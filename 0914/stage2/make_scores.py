#!/usr/bin/env python3
"""교사 점수 생성 + C1(재예측 일치). PREREG-TEACHER-ECON.md §3.2 · §3.4.

평가 종목 411개 × `20260319` 의 **모든 틱**에서 T2M · T3D · RDG 점수를 만든다 — 0911 과 같은 경로:
sd 적재 → 특징 → 무차원화 → 저장된 19개 특징 순서 → 16스텝 인과 윈도우(세션 안 앞채움) → 저장된 표준화 → 순전파.
C3 용으로 앞 20종목에는 FAKE 점수(= 정본 실행기가 계산한 book_imbalance)도 만든다.

C1: 유효 행(라벨 마스크)에서 재예측 점수가 0911 저장 예측과 최대 절대 차 1e-5 이내여야 한다.

    CUDA_VISIBLE_DEVICES=<GPU0 UUID> GROUP_GPU_UUID=<GPU0 UUID> python -m stage2.make_scores
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import common as C

from sd import dimensionless, labels, ticks  # noqa: E402
from framework import contract as fcontract  # noqa: E402
from efficient_window import make_causal_windows_subset  # noqa: E402

WINDOW = 16
PRED_CHUNK = 500_000
FEATS = json.loads(C.FEATURES_JSON.read_text())["features"]


def prepare(symbol: str):
    try:
        arrays = ticks.load_arrays(symbol, C.DATE)
    except (FileNotFoundError, ValueError) as error:
        return symbol, None, f"적재 실패: {type(error).__name__}: {error}"
    matrix, names = ticks.feature_matrix(arrays)
    X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
    cols = {name: X_symbol[:, j] for j, name in enumerate(kept)}
    missing = [f for f in FEATS if f not in cols]
    if missing:
        return symbol, None, f"특징 없음: {missing}"
    lbl = labels.build(arrays)
    bi = np.asarray(fcontract.ExpressionRuntime(arrays).evaluate(
        {"op": "primitive", "primitive_id": "book_imbalance"}), dtype=float)
    return symbol, {"X": np.column_stack([cols[f] for f in FEATS]),
                    "time_s": np.asarray(arrays["time_s"], dtype=float),
                    "bid1": np.asarray(arrays["bid_price"], dtype=float)[:, 0],
                    "mask": np.asarray(lbl.mask, bool), "y_path": np.asarray(lbl.y_path, float),
                    "book_imbalance": bi}, None


def main() -> None:
    want = os.environ.get("GROUP_GPU_UUID", "")
    if want != C.GPU_UUID or os.environ.get("CUDA_VISIBLE_DEVICES") != C.GPU_UUID:
        raise SystemExit(f"GPU0 를 지정해 기동해야 한다: CUDA_VISIBLE_DEVICES=GROUP_GPU_UUID={C.GPU_UUID}")
    t0 = time.time()
    symbols = C.eval_symbols()
    c3_symbols = set(symbols[:C.C3_SYMBOLS])
    for m in (*C.MODELS, "FAKE"):
        (C.SCORES / m).mkdir(parents=True, exist_ok=True)
    C.OUT.mkdir(parents=True, exist_ok=True)

    # 0911 저장 예측 (평가 ①) — 종목별 유효 행 순서
    a = pd.read_parquet(C.PRED_2M, columns=["symbol", "eval_set", "y_path", "pred_deeplob", "pred_ridge"])
    b = pd.read_parquet(C.PRED_3D, columns=["symbol", "eval_set", "y_path", "pred_deeplob"])
    if not (len(a) == len(b) and np.array_equal(a.y_path.to_numpy(float), b.y_path.to_numpy(float), equal_nan=True)):
        raise SystemExit("두 예측 파일의 행이 다르다")
    sel = (a.eval_set == "eval1").to_numpy()
    stored = pd.DataFrame({"symbol": a.symbol.astype(str).to_numpy()[sel], "y": a.y_path.to_numpy(float)[sel],
                           "T2M": a.pred_deeplob.to_numpy(float)[sel], "RDG": a.pred_ridge.to_numpy(float)[sel],
                           "T3D": b.pred_deeplob.to_numpy(float)[sel]})
    stored_idx = stored.groupby("symbol", sort=False).indices
    del a, b
    C.log("scores", f"저장 예측 평가 ① {len(stored):,}행 · {len(stored_idx)} 종목")

    pool = Pool(12)                      # CUDA 는 fork 뒤에만 건드린다
    import torch
    import group_run as gr

    teachers = {}
    for m, path in C.CKPT.items():
        ck = torch.load(path, map_location="cuda", weights_only=False)
        if ck["features"] != FEATS:
            raise SystemExit(f"{m} 교사의 특징 목록이 다르다")
        t = gr.build_teacher(len(FEATS))
        t._encoder.load_state_dict(ck["encoder"])
        t._head_path.load_state_dict(ck["head_path"])
        t._head_fill.load_state_dict(ck["head_fill"])
        t._mean, t._scale = ck["mean"], ck["scale"]
        teachers[m] = t
    rz = np.load(C.RIDGE)
    r_mu, r_sd, r_beta, r_ym = rz["mu"], rz["sd"], rz["beta"], float(rz["ym"])
    C.log("scores", f"교사 적재 T2M·T3D (GPU {torch.cuda.get_device_name(0)}), 릿지 α={float(rz['alpha'])}")

    c1 = {m: {"max_abs_diff": 0.0, "rows": 0} for m in C.MODELS}
    skipped, n_ticks, n_done, align_fail = {}, 0, 0, []
    for symbol, d, err in pool.imap(prepare, symbols, chunksize=2):
        if err is not None:
            skipped[symbol] = err
            continue
        n = len(d["X"])
        Xw = make_causal_windows_subset(d["X"], WINDOW, np.zeros(n, dtype=np.int64), np.arange(n))
        scores = {m: np.concatenate([teachers[m].predict_path(Xw[i:i + PRED_CHUNK])
                                     for i in range(0, n, PRED_CHUNK)]).astype(float)
                  for m in ("T2M", "T3D")}
        scores["RDG"] = r_ym + ((Xw - r_mu) / r_sd) @ r_beta
        for m, s in scores.items():
            np.savez(C.score_path(m, symbol), time_s=d["time_s"], bid1=d["bid1"], score=s)
        if symbol in c3_symbols:
            np.savez(C.score_path("FAKE", symbol), time_s=d["time_s"], bid1=d["bid1"], score=d["book_imbalance"])

        mask = d["mask"]
        idx = stored_idx.get(symbol)
        if idx is None or len(idx) != int(mask.sum()) or not np.array_equal(
                stored.y.to_numpy()[idx], d["y_path"][mask], equal_nan=True):
            align_fail.append(symbol)
        else:
            for m in C.MODELS:
                diff = np.abs(scores[m][mask] - stored[m].to_numpy()[idx])
                if len(diff):
                    c1[m]["max_abs_diff"] = max(c1[m]["max_abs_diff"], float(np.nanmax(diff)))
                    if np.isnan(diff).any():
                        c1[m]["nan_mismatch"] = c1[m].get("nan_mismatch", 0) + int(np.isnan(diff).sum())
                c1[m]["rows"] += int(len(diff))
        n_ticks += n
        n_done += 1
        if n_done % 50 == 0:
            C.log("scores", f"{n_done}/{len(symbols)} 종목, 틱 {n_ticks:,} ({time.time() - t0:.0f}s)")
    pool.close()
    pool.join()

    c1_pass = (not align_fail and not skipped
               and all(c1[m]["max_abs_diff"] <= C.C1_TOL and "nan_mismatch" not in c1[m] for m in C.MODELS)
               and sum(c1[m]["rows"] for m in C.MODELS) == 3 * len(stored))
    manifest = {"schema": "0914_stage2_scores.v1", "date": C.DATE, "features": FEATS,
                "n_symbols": len(symbols), "n_scored": n_done, "n_ticks": n_ticks, "skipped": skipped,
                "alignment_failures": align_fail, "c1": c1, "c1_tol": C.C1_TOL,
                "stored_eval1_rows": int(len(stored)), "c1_pass": bool(c1_pass),
                "fake_symbols": sorted(c3_symbols), "seconds": time.time() - t0}
    (C.OUT / "scores_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    C.log("scores", f"C1 {'통과' if c1_pass else '실패'}: {c1} · 건너뜀 {len(skipped)} · 정렬 실패 {len(align_fail)}")
    C.git_commit(["stage2/out/scores_manifest.json"],
                 f"run(0914): 교사 점수 생성 {n_done}종목·{n_ticks:,}틱, C1 {'통과' if c1_pass else '실패'}")
    if not c1_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
