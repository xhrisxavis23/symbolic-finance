#!/usr/bin/env python3
"""판정 R40 재현성 실험용 소규모(약 2만행) 데이터 준비.

L(저마찰) 종목 몇 개로 학습 표본(~20,000행), M(중간마찰) 종목 몇 개로
"안 본" 평가 표본(~5,000행)을 만든다. CPU/GPU 두 학습 스크립트가 **정확히
같은 입력**(같은 X, y, weight, window)을 쓰도록 여기서 한 번만 만들어
.npz 로 저장한다 — 데이터 준비 단계의 우연한 차이가 재현성 비교에 섞이지
않게 한다.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/scale")

from sd import dimensionless, labels, ticks, universe  # noqa: E402
from efficient_window import make_causal_windows_subset  # noqa: E402
from sd.manifold import select as manifold_select  # noqa: E402

DATE = "20260316"
OUT = "/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911_gpu_eval/repro_data.npz"
WINDOW = 16
TRAIN_MAX_SAMPLES = 20_000
SELECT_MAX_SAMPLES = 5_000
SEED = 0


def load_pool(symbols):
    per_symbol = []
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, DATE)
        except (FileNotFoundError, ValueError):
            continue
        matrix, names = ticks.feature_matrix(arrays)
        X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
        lbl = labels.build(arrays)
        per_symbol.append(({name: X_symbol[:, j] for j, name in enumerate(kept)}, lbl))
    common = set(per_symbol[0][0])
    for columns, _l in per_symbol[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))
    X = np.vstack([np.column_stack([c[name] for name in feature_names])
                   for c, _l in per_symbol])
    y_path = np.concatenate([lbl.y_path for _c, lbl in per_symbol])
    y_fill = np.concatenate([lbl.y_fill for _c, lbl in per_symbol])
    mask = np.concatenate([lbl.mask for _c, lbl in per_symbol])
    session_ids = np.concatenate([
        np.full(len(lbl.y_path), i, dtype=int) for i, (_c, lbl) in enumerate(per_symbol)])
    return X, y_path, y_fill, mask, feature_names, session_ids


def main() -> None:
    t0 = time.time()
    strata = universe.assign_strata(
        universe.liquidity_stats(universe.stock_symbols(DATE), DATE))
    L = sorted(strata.index[strata.friction == "L"])
    M = sorted(strata.index[strata.friction == "M"])

    # 소수 종목만 — 20,000/5,000행이면 충분하다. 앞에서부터 몇 개 골라
    # usable 행이 목표를 넘길 때까지 늘린다.
    def pick_enough(symbols, target):
        chosen = []
        total = 0
        for s in symbols:
            chosen.append(s)
            total += 1
            if total >= 6:  # 대략 6종목이면 20,000행 usable 을 넘긴다 (중앙값 3,118행/종목)
                break
        return chosen

    L_pick = pick_enough(L, TRAIN_MAX_SAMPLES)
    M_pick = pick_enough(M, SELECT_MAX_SAMPLES)
    print(f"[data] L_pick={L_pick}")
    print(f"[data] M_pick={M_pick}")

    X_L, yp_L, yf_L, mask_L, feat_L, sess_L = load_pool(L_pick)
    X_M, yp_M, yf_M, mask_M, feat_M, sess_M = load_pool(M_pick)
    common_features = tuple(sorted(set(feat_L) & set(feat_M)))
    idx_L = [feat_L.index(n) for n in common_features]
    idx_M = [feat_M.index(n) for n in common_features]
    X_L = X_L[:, idx_L]
    X_M = X_M[:, idx_M]
    print(f"[data] L usable rows={int(mask_L.sum()):,} pool={len(X_L):,} "
          f"M usable rows={int(mask_M.sum()):,} pool={len(X_M):,} "
          f"common_features={len(common_features)}")

    sel_L = manifold_select(X_L, mask_L, max_samples=TRAIN_MAX_SAMPLES, seed=SEED)
    fit_rows = sel_L.index[sel_L.on_manifold]
    fit_weight = sel_L.weight[sel_L.on_manifold]

    sel_M = manifold_select(X_M, mask_M, max_samples=SELECT_MAX_SAMPLES, seed=SEED)
    select_rows = sel_M.index[sel_M.on_manifold]
    select_weight = sel_M.weight[sel_M.on_manifold]

    print(f"[data] fit_rows={len(fit_rows):,} select_rows={len(select_rows):,}")

    Xw_fit = make_causal_windows_subset(X_L, window=WINDOW, session_ids=sess_L,
                                        indices=fit_rows)
    Xw_select = make_causal_windows_subset(X_M, window=WINDOW, session_ids=sess_M,
                                           indices=select_rows)

    np.savez(
        OUT,
        Xw_fit=Xw_fit, y_path_fit=yp_L[fit_rows], y_fill_fit=yf_L[fit_rows],
        weight_fit=fit_weight,
        Xw_select=Xw_select, y_path_select=yp_M[select_rows],
        y_fill_select=yf_M[select_rows], weight_select=select_weight,
        n_features=len(common_features), window=WINDOW,
    )
    print(f"[saved] {OUT} ({time.time()-t0:.1f}s)")
    print(f"[shapes] Xw_fit={Xw_fit.shape} Xw_select={Xw_select.shape}")


if __name__ == "__main__":
    main()
