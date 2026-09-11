#!/usr/bin/env python3
"""표본 크기 대비 R² 곡선용 데이터 풀을 만든다.

L(저마찰, 837종목) × 3일(20260316~18) 을 학습 후보 풀로, M(중간마찰,
834종목) × 1일(20260316) 을 **고정된 "안 본 종목" 평가 풀**로 쓴다. 평가
풀을 1일로 좁힌 이유: 이 진단의 목적은 "학습 표본을 늘리면 R² 가
오르는가"이고, 그 답을 얻으려면 평가 집합을 고정해 둔 채 학습 표본만
바꿔야 한다 — 평가 풀 자체의 규모는 이미 200,000행 상한으로 충분하다.

원본 배열(윈도우 이전)을 스크래치에 저장한다 — 윈도우는
`efficient_window.make_causal_windows_subset` 로 표집된 부분에만, 실험
스크립트에서 그때그때 만든다(전체에 윈도우를 씌우면 3일치 L 풀에서 출력이
수십 GB 가 된다, 0910/SCALE-MEASUREMENT.md §3.2).
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
from sd import dimensionless, labels, ticks, universe  # noqa: E402

STRATA_DATE = "20260316"
TRAIN_DATES = ["20260316", "20260317", "20260318"]
EVAL_DATE = "20260316"
OUT_DIR = "/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911_pool"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_symbol_day(symbol: str, date: str):
    try:
        arrays = ticks.load_arrays(symbol, date)
    except (FileNotFoundError, ValueError):
        return None
    matrix, names = ticks.feature_matrix(arrays)
    X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
    lbl = labels.build(arrays)
    return {name: X_symbol[:, j] for j, name in enumerate(kept)}, lbl


def load_pool(symbol_date_pairs):
    per_block = []
    n_loaded, n_skipped = 0, 0
    for symbol, date in symbol_date_pairs:
        result = load_symbol_day(symbol, date)
        if result is None:
            n_skipped += 1
            continue
        per_block.append(result)
        n_loaded += 1
    common = set(per_block[0][0])
    for columns, _l in per_block[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))
    return per_block, feature_names, n_loaded, n_skipped


def assemble(per_block, feature_names):
    X = np.vstack([np.column_stack([c[name] for name in feature_names])
                   for c, _l in per_block])
    y_path = np.concatenate([lbl.y_path for _c, lbl in per_block])
    y_fill = np.concatenate([lbl.y_fill for _c, lbl in per_block])
    mask = np.concatenate([lbl.mask for _c, lbl in per_block])
    session_ids = np.concatenate([
        np.full(len(lbl.y_path), i, dtype=int) for i, (_c, lbl) in enumerate(per_block)])
    return X, y_path, y_fill, mask, session_ids


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    strata = universe.assign_strata(
        universe.liquidity_stats(universe.stock_symbols(STRATA_DATE), STRATA_DATE))
    L = sorted(strata.index[strata.friction == "L"])
    M = sorted(strata.index[strata.friction == "M"])
    log(f"L={len(L)} M={len(M)}")

    L_pairs = [(s, d) for d in TRAIN_DATES for s in L]
    log(f"loading L pool: {len(L_pairs)} symbol-day pairs across {TRAIN_DATES}")
    t_l0 = time.time()
    L_blocks, L_features, L_loaded, L_skipped = load_pool(L_pairs)
    log(f"L pool loaded: {L_loaded} blocks, {L_skipped} skipped, "
        f"{len(L_features)} common features, {time.time()-t_l0:.1f}s")

    M_pairs = [(s, EVAL_DATE) for s in M]
    log(f"loading M pool: {len(M_pairs)} symbol-day pairs on {EVAL_DATE}")
    t_m0 = time.time()
    M_blocks, M_features, M_loaded, M_skipped = load_pool(M_pairs)
    log(f"M pool loaded: {M_loaded} blocks, {M_skipped} skipped, "
        f"{len(M_features)} common features, {time.time()-t_m0:.1f}s")

    common_features = tuple(sorted(set(L_features) & set(M_features)))
    log(f"common L∩M features: {len(common_features)}: {common_features}")

    X_L, y_path_L, y_fill_L, mask_L, sess_L = assemble(L_blocks, common_features)
    X_M, y_path_M, y_fill_M, mask_M, sess_M = assemble(M_blocks, common_features)
    log(f"assembled X_L={X_L.shape} X_M={X_M.shape}")

    np.savez(
        f"{OUT_DIR}/L_pool.npz", X=X_L, y_path=y_path_L, y_fill=y_fill_L,
        mask=mask_L, session_ids=sess_L)
    np.savez(
        f"{OUT_DIR}/M_pool.npz", X=X_M, y_path=y_path_M, y_fill=y_fill_M,
        mask=mask_M, session_ids=sess_M)
    meta = {
        "feature_names": list(common_features), "L_symbols": len(L), "M_symbols": len(M),
        "L_pairs": len(L_pairs), "M_pairs": len(M_pairs),
        "L_loaded": L_loaded, "L_skipped": L_skipped,
        "M_loaded": M_loaded, "M_skipped": M_skipped,
        "X_L_shape": list(X_L.shape), "X_M_shape": list(X_M.shape),
        "n_usable_L": int(mask_L.sum()), "n_usable_M": int(mask_M.sum()),
        "train_dates": TRAIN_DATES, "eval_date": EVAL_DATE,
        "build_seconds": time.time() - t0,
    }
    import json
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"saved pool to {OUT_DIR}, total {time.time()-t0:.1f}s")
    log(f"meta={meta}")


if __name__ == "__main__":
    main()
