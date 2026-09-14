#!/usr/bin/env python3
"""추가 측정 — niterations=200(계획서 원안), maxsize=18(0909 G2 선례), 200,000행
1개 지점만. R37-3: "시간 예측용으로만" 재측정한다. PREREG.md 의 타이밍 절 근거.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")

import numpy as np  # noqa: E402
from sd import config, dimensionless, labels, manifold, ticks, universe  # noqa: E402
from sd.teacher.shallow import ShallowMLP  # noqa: E402

OUT = Path("/home/dgu/tick/symbolic/0910/scale/data/m3b_pysr_niter200.json")
N_SYMBOLS_POOL = 300
MAX_SAMPLES = 200_000
NITERATIONS = 200
MAXSIZE = 18


def main() -> None:
    all_symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(all_symbols, config.DATE)
    strata = universe.assign_strata(stats)
    symbols = sorted(strata.index)[:N_SYMBOLS_POOL]

    per_symbol = []
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, config.DATE)
        except (FileNotFoundError, ValueError):
            continue
        matrix, names = ticks.feature_matrix(arrays)
        X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
        columns = {name: X_symbol[:, j] for j, name in enumerate(kept)}
        per_symbol.append((columns, labels.build(arrays)))

    common = set(per_symbol[0][0])
    for columns, _l in per_symbol[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))

    X = np.vstack([np.column_stack([c[name] for name in feature_names])
                   for c, _l in per_symbol])
    y_path = np.concatenate([l.y_path for _c, l in per_symbol])
    y_fill = np.concatenate([l.y_fill for _c, l in per_symbol])
    mask = np.concatenate([l.mask for _c, l in per_symbol])
    print(f"[pool] symbols={len(per_symbol)} rows={len(X):,}", flush=True)

    sel = manifold.select(X, mask, max_samples=MAX_SAMPLES, seed=0)
    fit_rows = sel.index[sel.on_manifold]
    fit_weight = sel.weight[sel.on_manifold]
    print(f"[select] n_fit={len(fit_rows)}", flush=True)

    teacher = ShallowMLP(n_features=X.shape[1], bottleneck=2, seed=0).fit(
        X[fit_rows], y_path[fit_rows], y_fill[fit_rows], fit_weight, epochs=100)
    target = teacher.predict_path(X[fit_rows])

    from sd.sr.pysr_backend import PySRBackend
    backend = PySRBackend(seed=0, deterministic=True, niterations=NITERATIONS,
                          maxsize=MAXSIZE)
    t0 = time.time()
    candidates = backend.fit(X[fit_rows], target, fit_weight, feature_names)
    t1 = time.time()

    result = {
        "niterations": NITERATIONS, "maxsize": MAXSIZE,
        "n_fit_rows": int(len(fit_rows)),
        "pysr_fit_seconds_wallclock": t1 - t0,
        "pysr_fit_seconds_internal": backend.diagnostics.get("fit_seconds"),
        "n_candidates": len(candidates),
    }
    print(f"[pysr200] n_fit={len(fit_rows)} wall={t1-t0:.1f}s "
          f"internal={backend.diagnostics.get('fit_seconds'):.1f}s "
          f"candidates={len(candidates)}", flush=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
