"""종목군 분석 실행 — DeepLOB 교사·릿지 기준선을 학습하고 테스트일 예측값을 행별로 저장한다.

PREREG-GROUPS.md §3·§4 와 부록 A2~A5 를 코드로 옮긴 것이다. **판정은 하지 않는다** —
`group_analysis.py` 가 저장된 예측값만 읽어 판정한다(몇 번이고 다시 돌릴 수 있다).

분할 (종목은 틱 3분위별 같은 비율, 시드 0)
  학습 종목 2/3   20260316·17 → 교사 경사하강·릿지 적합
  선택 종목 1/6   20260318    → 교사 조기 종료·릿지 규제 강도 선택
  평가 종목 1/6   20260319    → 평가 ① (안 본 종목·안 본 날, 1차 결과)
  학습 종목       20260319    → 평가 ② (본 종목·안 본 날)

단계마다 `groups/out/` 에 저장하고, 다시 실행하면 끝난 단계는 건너뛴다(재부팅 대비).
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/scale")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
from sd import dimensionless, labels, ticks  # noqa: E402
from sd.manifold import select as manifold_select  # noqa: E402
from sd.sr.base import weighted_r2  # noqa: E402
from efficient_window import make_causal_windows_subset  # noqa: E402

ROOT = Path("/home/dgu/tick/symbolic/0911/groups")
OUT = ROOT / "out"
ASSIGN = ROOT / "data" / "group_assignment.csv"
PROGRESS = OUT / "group_run_progress.json"

TRAIN_DATES = ("20260316", "20260317")
SELECT_DATE = "20260318"
TEST_DATE = "20260319"
TICK_ORDER = ("큰틱", "중간틱", "작은틱")

SEED = 0
WINDOW = 16
FIT_ROWS = 2_000_000          # 사전등록 §4
SELECT_ROWS = 200_000
EPOCHS = 1500
EVAL_EVERY = 10
PATIENCE = 30                 # 부록 D: 10 은 일시적 하락 구간에서 멈춘다(곡선 50만 행 사례)
# 이 셸은 ~/.bashrc 가 CUDA_VISIBLE_DEVICES 를 물리 GPU1(ollama 용) 한 장으로 고정한다 — 그래서 "cuda:1" 은
# 존재하지 않는다(시험 운전에서 잡혔다). 기동할 때 CUDA_VISIBLE_DEVICES·GROUP_GPU_UUID 를 쓸 물리 GPU 로
# 지정하고, 여기서는 보이는 첫 장("cuda")을 쓴다. build_teacher 가 UUID 로 맞는 장치인지 확인한다.
DEVICE = "cuda"
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)   # 부록 A4. 표준화 후 행당 단위
BLOCKS_PER_SLICE = 100        # 평가 윈도우를 종목·날짜 묶음 단위로 잘라 만든다
PRED_CHUNK = 500_000          # predict_path 는 입력 전체를 한 번에 GPU 로 올린다
WORKERS = 12


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_progress(update: dict) -> None:
    state = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    state.update(update)
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=float))
    os.replace(tmp, PROGRESS)


# ---------------------------------------------------------------------------
# 적재 — scale/build_pool.py::load_symbol_day 와 같은 sd 호출이다(두 번째 진실 원천을 만들지 않는다).
# 그 모듈을 import 하지 않는 이유: 가져오는 순간 풀 생성이 돌 위험을 피하려는 것뿐이다.
# ---------------------------------------------------------------------------

def _load(pair):
    symbol, date = pair
    try:
        arrays = ticks.load_arrays(symbol, date)
    except (FileNotFoundError, ValueError):
        return None
    matrix, names = ticks.feature_matrix(arrays)
    X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
    lbl = labels.build(arrays)
    cols = {name: X_symbol[:, j] for j, name in enumerate(kept)}
    return (symbol, date, cols, np.asarray(lbl.y_path, float),
            np.asarray(lbl.y_fill, float), np.asarray(lbl.mask, bool))


def load_blocks(pairs):
    blocks = []
    with Pool(WORKERS) as pool:
        for r in pool.imap(_load, pairs, chunksize=4):
            if r is not None:
                blocks.append(r)
    return blocks


def assemble(blocks, feats):
    X = np.vstack([np.column_stack([b[2][f] for f in feats]) for b in blocks])
    y_path = np.concatenate([b[3] for b in blocks])
    y_fill = np.concatenate([b[4] for b in blocks])
    mask = np.concatenate([b[5] for b in blocks])
    sess = np.concatenate([np.full(len(b[3]), i, dtype=np.int64) for i, b in enumerate(blocks)])
    return X, y_path, y_fill, mask, sess


# ---------------------------------------------------------------------------

def make_split(assign: pd.DataFrame) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    split: dict[str, str] = {}
    for g in TICK_ORDER:
        syms = np.array(sorted(assign.index[assign.tick_group == g]))
        rng.shuffle(syms)
        n = len(syms)
        n_eval, n_sel = n // 6, n // 6
        split.update({s: "eval" for s in syms[:n_eval]})
        split.update({s: "select" for s in syms[n_eval:n_eval + n_sel]})
        split.update({s: "train" for s in syms[n_eval + n_sel:]})
    return split


def build_teacher(n_features: int):
    import torch
    from deeplob_gpu import DeepLOBCompactGPU   # CUDA 는 적재(fork) 뒤에만 건드린다
    want = os.environ.get("GROUP_GPU_UUID", "")
    got = str(torch.cuda.get_device_properties(0).uuid)
    norm = lambda u: u.lower().removeprefix("gpu-")
    if not want or norm(want) != norm(got):
        raise RuntimeError(f"의도한 GPU 가 아니다: GROUP_GPU_UUID={want!r}, 보이는 장치={got} — "
                           "다른 실행과 같은 GPU 를 쓰게 될 수 있어 멈춘다")
    log(f"GPU 확인: {torch.cuda.get_device_name(0)} {got}")
    return DeepLOBCompactGPU(n_features=n_features, bottleneck=2, seed=SEED,
                             window=WINDOW, device=DEVICE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assign = pd.read_csv(ASSIGN, index_col=0)
    assign.index = assign.index.astype(str).str.zfill(6)

    split = make_split(assign)
    by_role = {r: sorted(s for s, v in split.items() if v == r) for r in ("train", "select", "eval")}
    for r, syms in by_role.items():
        counts = assign.loc[syms].tick_group.value_counts().reindex(TICK_ORDER).to_dict()
        log(f"{r}: {len(syms)} 종목 {counts}")
    save_progress({"stage": "split", "split_counts": {r: len(s) for r, s in by_role.items()},
                   "split_symbols": by_role})

    # --- 적재: 필요한 (종목, 날짜) 만 --------------------------------------------------
    pairs = ([(s, d) for d in TRAIN_DATES for s in by_role["train"]]
             + [(s, SELECT_DATE) for s in by_role["select"]]
             + [(s, TEST_DATE) for s in by_role["eval"]]
             + [(s, TEST_DATE) for s in by_role["train"]])
    t0 = time.time()
    blocks = load_blocks(pairs)
    feats = sorted(set.intersection(*(set(b[2]) for b in blocks)))
    log(f"적재 {len(blocks)}/{len(pairs)} 블록, 공통 특징 {len(feats)} ({time.time()-t0:.0f}s)")
    save_progress({"stage": "loaded", "n_pairs": len(pairs), "n_blocks": len(blocks),
                   "n_missing_pairs": len(pairs) - len(blocks), "features": feats,
                   "load_seconds": time.time() - t0})

    def pick(role, dates):
        return [b for b in blocks if split[b[0]] == role and b[1] in dates]

    b_train, b_select = pick("train", TRAIN_DATES), pick("select", (SELECT_DATE,))
    b_eval1, b_eval2 = pick("eval", (TEST_DATE,)), pick("train", (TEST_DATE,))
    F = len(feats)

    # --- 교사·릿지 학습 (저장돼 있으면 불러온다) ---------------------------------------
    import torch
    ckpt = OUT / "deeplob.pt"
    ridge_path = OUT / "ridge.npz"
    if ckpt.exists() and ridge_path.exists():
        log("학습된 교사·릿지를 불러온다 — 학습 단계 건너뜀")
        ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        assert ck["features"] == feats, "저장된 교사의 특징 목록이 지금과 다르다"
        teacher = build_teacher(F)
        teacher._encoder.load_state_dict(ck["encoder"])
        teacher._head_path.load_state_dict(ck["head_path"])
        teacher._head_fill.load_state_dict(ck["head_fill"])
        teacher._mean, teacher._scale = ck["mean"], ck["scale"]
        rz = np.load(ridge_path)
        r_mu, r_sd, r_beta, r_ym = rz["mu"], rz["sd"], rz["beta"], float(rz["ym"])
    else:
        X_tr, yp_tr, yf_tr, m_tr, s_tr = assemble(b_train, feats)
        X_se, yp_se, _yf_se, m_se, s_se = assemble(b_select, feats)
        log(f"학습 풀 {len(X_tr):,}행, 선택 풀 {len(X_se):,}행")

        t0 = time.time()
        sel_fit = manifold_select(X_tr, m_tr, max_samples=FIT_ROWS, seed=SEED)
        fit_rows, fit_w = sel_fit.index[sel_fit.on_manifold], sel_fit.weight[sel_fit.on_manifold]
        sel_sel = manifold_select(X_se, m_se, max_samples=SELECT_ROWS, seed=SEED)
        sel_rows, sel_w = sel_sel.index[sel_sel.on_manifold], sel_sel.weight[sel_sel.on_manifold]
        Xw_fit = make_causal_windows_subset(X_tr, WINDOW, s_tr, fit_rows)
        Xw_sel = make_causal_windows_subset(X_se, WINDOW, s_se, sel_rows)
        y_fit, y_sel = yp_tr[fit_rows], yp_se[sel_rows]
        log(f"학습 표본 {len(fit_rows):,} · 선택 표본 {len(sel_rows):,} · 윈도우 {Xw_fit.shape} "
            f"({time.time()-t0:.0f}s)")
        del X_tr, X_se

        # 교사
        t0 = time.time()
        teacher = build_teacher(F).fit(
            Xw_fit, y_fit, yf_tr[fit_rows], fit_w, epochs=EPOCHS,
            X_path_select=Xw_sel, y_path_select=y_sel, weight_select=sel_w,
            eval_every=EVAL_EVERY, patience=PATIENCE)
        h = teacher.early_stop_history_
        teacher_info = {"fit_seconds": time.time() - t0, "best_epoch": h.best_epoch,
                        "best_score": h.best_score, "stopped_epoch": h.stopped_epoch,
                        "triggered": h.triggered, "eval_epochs": h.eval_epochs,
                        "eval_scores": h.eval_scores}
        log(f"교사 학습 {teacher_info['fit_seconds']:.0f}s, 최선 epoch {h.best_epoch}, "
            f"선택 R² {h.best_score:.5f}, 멈춤 {h.stopped_epoch}, 조기종료 {h.triggered}")
        torch.save({"encoder": teacher._encoder.state_dict(),
                    "head_path": teacher._head_path.state_dict(),
                    "head_fill": teacher._head_fill.state_dict(),
                    "mean": teacher._mean, "scale": teacher._scale, "features": feats}, ckpt)

        # 릿지 — 같은 304차원 윈도우 입력, 같은 학습 행·가중치 (부록 A4)
        t0 = time.time()
        w = fit_w.astype(float)
        r_mu = np.average(Xw_fit, axis=0, weights=w)
        r_sd = np.sqrt(np.average((Xw_fit - r_mu) ** 2, axis=0, weights=w))
        r_sd[r_sd == 0] = 1.0
        r_ym = float(np.average(y_fit, weights=w))
        D = Xw_fit.shape[1]
        XtWX, XtWy = np.zeros((D, D)), np.zeros(D)
        for i in range(0, len(Xw_fit), 250_000):
            Xc = (Xw_fit[i:i + 250_000] - r_mu) / r_sd
            wc = w[i:i + 250_000]
            XtWX += (Xc * wc[:, None]).T @ Xc
            XtWy += Xc.T @ (wc * (y_fit[i:i + 250_000] - r_ym))
        XtWX /= w.sum(); XtWy /= w.sum()
        Xs_sel = (Xw_sel - r_mu) / r_sd
        alpha_scores = {}
        best_alpha, r_beta, best_r2 = None, None, -np.inf
        for a in RIDGE_ALPHAS:
            beta = np.linalg.solve(XtWX + a * np.eye(D), XtWy)
            r2 = float(weighted_r2(y_sel, r_ym + Xs_sel @ beta, sel_w))
            alpha_scores[str(a)] = r2
            if r2 > best_r2:
                best_alpha, r_beta, best_r2 = a, beta, r2
        np.savez(ridge_path, mu=r_mu, sd=r_sd, beta=r_beta, ym=r_ym, alpha=best_alpha)
        log(f"릿지 규제 {best_alpha}, 선택 R² {best_r2:.5f} ({time.time()-t0:.0f}s)")
        save_progress({"stage": "trained", "n_fit_rows": int(len(fit_rows)),
                       "n_select_rows": int(len(sel_rows)), "teacher": teacher_info,
                       "ridge": {"alpha": best_alpha, "select_r2": best_r2,
                                 "alpha_scores": alpha_scores}})
        del Xw_fit, Xw_sel, Xs_sel

    # --- 테스트일 예측값 저장 (부록 A2: 유효 행 전체, 가중치 없음) ----------------------
    pred_path = OUT / "predictions.parquet"
    if pred_path.exists():
        log("예측값 파일이 이미 있다 — 예측 단계 건너뜀")
    else:
        t0 = time.time()
        tmp = pred_path.with_suffix(".tmp.parquet")
        schema = pa.schema([("symbol", pa.string()), ("date", pa.string()), ("eval_set", pa.string()),
                            ("y_path", pa.float64()), ("pred_deeplob", pa.float64()),
                            ("pred_ridge", pa.float64())])
        writer = pq.ParquetWriter(tmp, schema)
        n_rows = {}
        for set_name, bl in (("eval1", b_eval1), ("eval2", b_eval2)):
            n_rows[set_name] = 0
            for k in range(0, len(bl), BLOCKS_PER_SLICE):
                sl = bl[k:k + BLOCKS_PER_SLICE]
                X, yp, _yf, m, sess = assemble(sl, feats)
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                Xw = make_causal_windows_subset(X, WINDOW, sess, idx)
                p_d = np.concatenate([teacher.predict_path(Xw[i:i + PRED_CHUNK])
                                      for i in range(0, len(idx), PRED_CHUNK)])
                p_r = r_ym + ((Xw - r_mu) / r_sd) @ r_beta
                sym = np.array([b[0] for b in sl])[sess[idx]]
                writer.write_table(pa.table({
                    "symbol": sym, "date": np.full(len(idx), TEST_DATE),
                    "eval_set": np.full(len(idx), set_name), "y_path": yp[idx],
                    "pred_deeplob": p_d.astype(float), "pred_ridge": p_r}, schema=schema))
                n_rows[set_name] += len(idx)
                log(f"  {set_name} 묶음 {k // BLOCKS_PER_SLICE + 1}: 누적 {n_rows[set_name]:,}행")
        writer.close()
        os.replace(tmp, pred_path)
        save_progress({"stage": "predicted", "n_pred_rows": n_rows,
                       "predict_seconds": time.time() - t0})
        log(f"예측값 저장 {n_rows} ({time.time()-t0:.0f}s)")

    # --- 판정 (group_analysis.py) --------------------------------------------------
    sys.path.insert(0, str(ROOT))
    import group_analysis
    group_analysis.main()
    res = json.loads(group_analysis.OUT.read_text())
    save_progress({"stage": "analysed", "verdict_primary": res["verdict_primary"]})
    log("완료")


if __name__ == "__main__":
    main()
