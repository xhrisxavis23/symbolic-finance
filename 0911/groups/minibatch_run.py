"""미니배치 교사 실행 — PREREG-MINIBATCH.md §2·§3·§4 를 코드로 옮긴 것.

  python3 minibatch_run.py gate   1단계 동등성: 종목군 실행과 같은 학습 200만 행·가중치·선택 행
  python3 minibatch_run.py full   2단계: 학습 종목 × 3일(16·17·18) on-manifold 행 전부. 1단계 통과 시에만.

적재·분할·특징·표본 선별·교사 생성·예측 행 구성은 group_run.py 의 것을 그대로 쓴다.
학습률은 사전등록 두 후보를 모두 학습하고 선택 종목 R² 로 고른다(§1 표의 미니배치 절차, 두 단계 공통).
판정 규칙(음성 대조군)은 group_analysis.verdict 를 그대로 부르고, 짝지은 비교는 paired_analysis 를 쓴다.
출력 groups/out/mb_<단계>/ — 단계마다 저장하고, 다시 실행하면 끝난 단계는 건너뛴다(재부팅 대비).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_analysis as ga  # noqa: E402
import group_run as gr  # noqa: E402
import minibatch_teacher as mbt  # noqa: E402
import paired_analysis as paired  # noqa: E402

FULL_DATES = ("20260316", "20260317", "20260318")   # §3
GATE_RATIO = 0.9                                     # §2
REQUIRE_GATE_PASS = True                             # 시험 운전만 끈다
IDENTITY_RTOL = 1e-6


def out_dir(stage: str):
    return gr.OUT / f"mb_{stage}"


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


def save_progress(path, update: dict) -> None:
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(update)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=float))
    os.replace(tmp, path)


def read_assign() -> pd.DataFrame:
    assign = pd.read_csv(gr.ASSIGN, index_col=0)
    assign.index = assign.index.astype(str).str.zfill(6)
    return assign


def identity_check(teacher) -> dict | None:
    """1단계 전제("같은 학습 행") 확인 — 표준화 평균·척도는 학습 행 윈도우에서만 나오므로 기준 교사 저장본과
    같아야 한다. 척도는 거의 상수인 열(두 계산이 반올림 차이로 0 또는 극소값을 낼 수 있다)을 따로 센다."""
    import torch
    p = gr.OUT / "deeplob.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m0, s0, m1, s1 = ck["mean"], ck["scale"], teacher._mean, teacher._scale
    degenerate = (s0 <= 1e-6 * (1 + np.abs(m0))) | (s0 == 1.0) | (s1 == 1.0)
    res = {"mean_close": bool(np.allclose(m1, m0, rtol=IDENTITY_RTOL, atol=1e-9)),
           "scale_close": bool(np.allclose(s1[~degenerate], s0[~degenerate], rtol=IDENTITY_RTOL, atol=1e-9)),
           "max_abs_mean_diff": float(np.max(np.abs(m1 - m0))),
           "max_rel_scale_diff": float(np.max(np.abs(s1 - s0)[~degenerate] / s0[~degenerate])) if (~degenerate).any() else 0.0,
           "n_degenerate_columns": int(degenerate.sum())}
    res["close"] = res["mean_close"] and res["scale_close"]
    if not res["close"]:
        raise RuntimeError(f"학습 행 윈도우의 표준화 통계가 기준 교사와 다르다 — 같은 데이터가 아니다: {res}")
    return res


def load_teacher(path, F: int, feats: list[str]):
    import torch
    ck = torch.load(path, map_location=gr.DEVICE, weights_only=False)
    if ck["features"] != feats:
        raise RuntimeError("저장된 교사의 특징 목록이 지금과 다르다")
    t = gr.build_teacher(F)
    t._encoder.load_state_dict(ck["encoder"])
    t._head_path.load_state_dict(ck["head_path"])
    t._head_fill.load_state_dict(ck["head_fill"])
    t._mean, t._scale = ck["mean"], ck["scale"]
    return t


def main(stage: str) -> None:
    out = out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    prog = out / "progress.json"
    gate_prog = out_dir("gate") / "progress.json"
    if stage == "full" and REQUIRE_GATE_PASS:
        g = json.loads(gate_prog.read_text()) if gate_prog.exists() else {}
        if g.get("gate", {}).get("passed") is not True:
            sys.exit("1단계 동등성 검증을 통과하지 않았다 — 3일 전체 학습을 하지 않는다 (PREREG-MINIBATCH §2)")

    assign = read_assign()
    split = gr.make_split(assign)
    by_role = {r: sorted(s for s, v in split.items() if v == r) for r in ("train", "select", "eval")}
    if stage == "gate":
        train_dates, select_dates = gr.TRAIN_DATES, (gr.SELECT_DATE,)
    else:
        train_dates, select_dates = FULL_DATES, FULL_DATES
    save_progress(prog, {"stage": "split", "train_dates": train_dates, "select_dates": select_dates,
                         "test_date": gr.TEST_DATE, "split_counts": {r: len(s) for r, s in by_role.items()}})

    # 1단계는 종목군 실행과 같은 순서로 적재해야 같은 행이 뽑힌다(같은 입력·같은 시드).
    pairs = ([(s, d) for d in train_dates for s in by_role["train"]]
             + [(s, d) for d in select_dates for s in by_role["select"]]
             + [(s, gr.TEST_DATE) for s in by_role["eval"]]
             + [(s, gr.TEST_DATE) for s in by_role["train"]])
    t0 = time.time()
    blocks = gr.load_blocks(pairs)
    feats = sorted(set.intersection(*(set(b[2]) for b in blocks)))
    gr.log(f"[{stage}] 적재 {len(blocks)}/{len(pairs)} 블록, 공통 특징 {len(feats)} ({time.time()-t0:.0f}s)")
    base_prog = json.loads(gr.PROGRESS.read_text()) if gr.PROGRESS.exists() else {}
    if base_prog.get("features") and base_prog["features"] != feats:
        raise RuntimeError("특징 목록이 종목군 실행과 다르다")
    save_progress(prog, {"stage": "loaded", "n_pairs": len(pairs), "n_blocks": len(blocks),
                         "features": feats, "load_seconds": time.time() - t0})

    def pick(role, dates):
        return [b for b in blocks if split[b[0]] == role and b[1] in dates]

    b_train, b_select = pick("train", train_dates), pick("select", select_dates)
    b_eval1, b_eval2 = pick("eval", (gr.TEST_DATE,)), pick("train", (gr.TEST_DATE,))
    del blocks
    F = len(feats)

    import torch
    ckpts = {lr: out / f"teacher_lr{lr_tag(lr)}.pt" for lr in mbt.LR_CANDIDATES}
    if not all(p.exists() for p in ckpts.values()):
        X_tr, yp_tr, yf_tr, m_tr, s_tr = gr.assemble(b_train, feats)
        X_se, yp_se, _yf_se, m_se, s_se = gr.assemble(b_select, feats)
        del b_train, b_select
        t0 = time.time()
        cap = gr.FIT_ROWS if stage == "gate" else len(X_tr)      # full: 상한 없음 = on-manifold 행 전부
        sel_fit = gr.manifold_select(X_tr, m_tr, max_samples=cap, seed=gr.SEED)
        fit_rows, fit_w = sel_fit.index[sel_fit.on_manifold], sel_fit.weight[sel_fit.on_manifold]
        sel_sel = gr.manifold_select(X_se, m_se, max_samples=gr.SELECT_ROWS, seed=gr.SEED)
        sel_rows, sel_w = sel_sel.index[sel_sel.on_manifold], sel_sel.weight[sel_sel.on_manifold]
        Xw_sel = gr.make_causal_windows_subset(X_se, gr.WINDOW, s_se, sel_rows)
        windower = mbt.BatchWindower(X_tr, s_tr, gr.WINDOW)
        del X_tr, X_se
        sel_info = {"n_train_pool": int(len(yp_tr)), "n_select_pool": int(len(yp_se)),
                    "n_fit_rows": int(len(fit_rows)), "n_select_rows": int(len(sel_rows)),
                    "select_seconds": time.time() - t0}
        gr.log(f"[{stage}] 학습 풀 {len(yp_tr):,} → 학습 표본 {len(fit_rows):,} · 선택 표본 {len(sel_rows):,} "
               f"({sel_info['select_seconds']:.0f}s)")
        if stage == "gate":
            for k in ("n_fit_rows", "n_select_rows"):
                if base_prog.get(k) is not None and base_prog[k] != sel_info[k]:
                    raise RuntimeError(f"{k} 가 종목군 실행과 다르다: {sel_info[k]} vs {base_prog[k]}")
        save_progress(prog, {"stage": "selected", **sel_info})

        for lr, ck in ckpts.items():
            if ck.exists():
                continue
            teacher = gr.build_teacher(F)
            gr.log(f"[{stage}] 미니배치 학습 시작 lr={lr_tag(lr)}")
            info = mbt.fit_minibatch(teacher, windower, fit_rows, yp_tr[fit_rows], yf_tr[fit_rows], fit_w,
                                     Xw_select=Xw_sel, y_select=yp_se[sel_rows], w_select=sel_w, lr=lr,
                                     batch_size=mbt.BATCH_SIZE, max_epochs=mbt.MAX_EPOCHS,
                                     evals_per_epoch=mbt.EVALS_PER_EPOCH, patience=mbt.PATIENCE, seed=gr.SEED)
            if stage == "gate":
                info["identity_vs_fullbatch"] = identity_check(teacher)
            gr.log(f"[{stage}] lr={lr_tag(lr)} 학습 {info['fit_seconds']:.0f}s, 최선 epoch {info['best_epoch']}, "
                   f"선택 R² {info['best_score']:.5f}, 멈춤 {info['stopped_epoch']}, 조기종료 {info['triggered']}")
            torch.save({"encoder": teacher._encoder.state_dict(), "head_path": teacher._head_path.state_dict(),
                        "head_fill": teacher._head_fill.state_dict(), "mean": teacher._mean,
                        "scale": teacher._scale, "features": feats, "info": info}, ck)
            save_progress(prog, {"stage": f"trained_lr{lr_tag(lr)}", f"lr_{lr_tag(lr)}": info})
            del teacher
            torch.cuda.empty_cache()
        del windower, Xw_sel

    infos = {lr: torch.load(ck, map_location="cpu", weights_only=False)["info"] for lr, ck in ckpts.items()}
    chosen = max(mbt.LR_CANDIDATES, key=lambda lr: infos[lr]["best_score"])
    save_progress(prog, {"chosen_lr": chosen,
                         "select_r2_by_lr": {lr_tag(lr): infos[lr]["best_score"] for lr in mbt.LR_CANDIDATES}})
    gr.log(f"[{stage}] 학습률 선택 {lr_tag(chosen)} (선택 R² " +
           ", ".join(f"{lr_tag(lr)}: {infos[lr]['best_score']:.5f}" for lr in mbt.LR_CANDIDATES) + ")")

    # --- 테스트일 예측값 — group_run.py 와 같은 묶음·같은 행(유효 행 전체) ------------------------
    pred_path = out / "predictions.parquet"
    if pred_path.exists():
        gr.log(f"[{stage}] 예측값 파일이 이미 있다 — 예측 단계 건너뜀")
    else:
        teacher = load_teacher(ckpts[chosen], F, feats)
        t0 = time.time()
        tmp = pred_path.with_suffix(".tmp.parquet")
        schema = pa.schema([("symbol", pa.string()), ("date", pa.string()), ("eval_set", pa.string()),
                            ("y_path", pa.float64()), ("pred_deeplob", pa.float64())])
        writer = pq.ParquetWriter(tmp, schema)
        n_rows = {}
        for set_name, bl in (("eval1", b_eval1), ("eval2", b_eval2)):
            n_rows[set_name] = 0
            for k in range(0, len(bl), gr.BLOCKS_PER_SLICE):
                sl = bl[k:k + gr.BLOCKS_PER_SLICE]
                X, yp, _yf, m, sess = gr.assemble(sl, feats)
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                Xw = gr.make_causal_windows_subset(X, gr.WINDOW, sess, idx)
                p_d = np.concatenate([teacher.predict_path(Xw[i:i + gr.PRED_CHUNK])
                                      for i in range(0, len(idx), gr.PRED_CHUNK)])
                sym = np.array([b[0] for b in sl])[sess[idx]]
                writer.write_table(pa.table({
                    "symbol": sym, "date": np.full(len(idx), gr.TEST_DATE),
                    "eval_set": np.full(len(idx), set_name), "y_path": yp[idx],
                    "pred_deeplob": p_d.astype(float)}, schema=schema))
                n_rows[set_name] += len(idx)
        writer.close()
        os.replace(tmp, pred_path)
        save_progress(prog, {"stage": "predicted", "n_pred_rows": n_rows, "predict_seconds": time.time() - t0})
        gr.log(f"[{stage}] 예측값 저장 {n_rows} ({time.time()-t0:.0f}s)")

    analyse(stage, out, prog, infos[chosen], chosen)


def analyse(stage: str, out, prog, chosen_info: dict, chosen_lr: float) -> dict:
    assign = read_assign()
    df = pd.read_parquet(out / "predictions.parquet")
    df["symbol"] = df.symbol.astype(str).str.zfill(6)
    ok = np.isfinite(df.y_path) & np.isfinite(df.pred_deeplob)
    dropped = int((~ok).sum())
    df = df[ok & df.symbol.isin(assign.index)]

    rng = np.random.default_rng(ga.SEED)
    sets = {s: ga.analyse_set(df[df.eval_set == s], assign, "deeplob", rng) for s in ("eval1", "eval2")}
    # 부록 E 규칙을 옮겨 쓰지 않고 verdict 를 그대로 부른다. 릿지 자리에 같은 결과를 넣었으므로 deeplob 줄만 본다.
    violations = [v for v in ga.verdict(sets["eval1"], sets["eval1"])["violations"] if v.startswith("deeplob")]
    res: dict = {"stage": stage, "chosen_lr": chosen_lr, "chosen_info": chosen_info,
                 "n_rows_dropped_non_finite": dropped, "sets": sets,
                 "negative_control_violations": violations, "vs_fullbatch_2M": None}

    base = gr.OUT / "predictions.parquet"
    if base.exists():
        res["vs_fullbatch_2M"] = {}
        for s in ("eval1", "eval2"):
            sym, y, a, b, nd = paired.load_aligned(base, out / "predictions.parquet", s)
            d = paired.paired_delta(sym, y, a, b)
            d["pred_corr"] = float(np.corrcoef(a, b)[0, 1])
            d["n_rows_dropped_non_finite"] = nd
            if stage == "gate":
                d.pop("verdict")                   # 1단계의 차이는 판정이 아니라 보고다(§2)
            res["vs_fullbatch_2M"][s] = d

    update: dict = {"stage": "analysed", "negative_control_violations": violations}
    if stage == "gate":
        ref = json.loads(gr.PROGRESS.read_text()).get("teacher", {}).get("best_score") if gr.PROGRESS.exists() else None
        mb = chosen_info["best_score"]
        if ref is None:
            gate = {"passed": None, "reason": "기준값(종목군 실행 교사 선택 R²) 없음 — 판정 보류"}
        else:
            gate = {"reference_fullbatch_select_r2": ref, "minibatch_select_r2": mb,
                    "threshold": GATE_RATIO * ref, "passed": bool(mb >= GATE_RATIO * ref),
                    "minibatch_higher": bool(mb > ref), "reference_nonpositive": bool(ref <= 0)}
        res["gate"] = update["gate"] = gate
        gr.log(f"[gate] 판정 {gate}")
    else:
        if violations:
            verdict = "판정 보류 — 음성 대조군 위반"
        elif res["vs_fullbatch_2M"] is None:
            verdict = "판정 보류 — 200만 행 기준 예측 없음"
        else:
            verdict = res["vs_fullbatch_2M"]["eval1"]["verdict"]
        res["verdict"] = update["verdict"] = verdict
        gr.log(f"[full] 판정 {verdict}")
    if res["vs_fullbatch_2M"] is not None:
        update["vs_fullbatch_2M"] = res["vs_fullbatch_2M"]
        e1 = res["vs_fullbatch_2M"]["eval1"]
        gr.log(f"[{stage}] 평가① R² 전체배치 {e1['r2_a']:.5f} → 미니배치 {e1['r2_b']:.5f}, "
               f"차이 {e1['delta']:+.5f} [{e1['delta_lo']:+.5f}, {e1['delta_hi']:+.5f}], 예측 상관 {e1['pred_corr']:.3f}")
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    save_progress(prog, update)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gate", "full"))
    main(ap.parse_args().stage)
