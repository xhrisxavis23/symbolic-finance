#!/usr/bin/env python3
"""M3 — 심볼릭 회귀(PySR) 1회 적합 시간 실측. 표본 크기를 몇 단계로 바꿔가며 잰다.

절차 (D13 과 같은 경로): 여러 종목을 로드 → 무차원 좌표 X, y_path, mask →
manifold.select(max_samples=...) → ShallowMLP 교사를 짧게 학습 → 교사 출력을
타깃으로 PySR fit. deterministic=True(직렬) — 계획서가 구조 복원율 판정을
위해 요구하는 기본값과 동일 조건.

같은 파이썬 프로세스 안에서 표본 크기를 늘려가며 반복 호출해 (a) 첫 호출의
Julia 워밍업 고정비용과 (b) 웜 상태에서 표본 크기에 따른 한계 비용을 분리한다.
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

OUT = Path("/home/dgu/tick/symbolic/0910/scale/data/m3_pysr_scale.json")

# 실제 데이터 풀을 만들기 위해 로드할 종목 수 (변수 개수 8~9개인 무차원 어휘 기준
# 다양한 max_samples 값을 시험하기에 충분한 풀을 만든다)
N_SYMBOLS_POOL = 300
SAMPLE_SIZES = [2_000, 20_000, 100_000, 200_000]
NITERATIONS = 40   # run_slice.py CLI 기본값과 동일
MAXSIZE = 20


def build_pool():
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
    print(f"[pool] symbols={len(per_symbol)} rows={len(X):,} maskable={int(mask.sum()):,} "
          f"features={len(feature_names)}", flush=True)
    return X, y_path, y_fill, mask, feature_names


def main() -> None:
    log = {"niterations": NITERATIONS, "maxsize": MAXSIZE,
           "n_symbols_pool": N_SYMBOLS_POOL, "runs": []}

    X, y_path, y_fill, mask, feature_names = build_pool()

    from sd.sr.pysr_backend import PySRBackend

    for max_samples in SAMPLE_SIZES:
        sel = manifold.select(X, mask, max_samples=max_samples, seed=0)
        fit_rows = sel.index[sel.on_manifold]
        fit_weight = sel.weight[sel.on_manifold]
        n_fit = len(fit_rows)
        print(f"[select] max_samples={max_samples} -> on_manifold rows={n_fit}", flush=True)

        t_teacher0 = time.time()
        teacher = ShallowMLP(n_features=X.shape[1], bottleneck=2, seed=0).fit(
            X[fit_rows], y_path[fit_rows], y_fill[fit_rows], fit_weight, epochs=100)
        t_teacher1 = time.time()
        target = teacher.predict_path(X[fit_rows])

        backend = PySRBackend(seed=0, deterministic=True, niterations=NITERATIONS,
                              maxsize=MAXSIZE)
        t0 = time.time()
        candidates = backend.fit(X[fit_rows], target, fit_weight, feature_names)
        t1 = time.time()

        entry = {
            "max_samples_requested": max_samples,
            "n_fit_rows": int(n_fit),
            "teacher_fit_seconds": t_teacher1 - t_teacher0,
            "pysr_fit_seconds_wallclock": t1 - t0,
            "pysr_fit_seconds_internal": backend.diagnostics.get("fit_seconds"),
            "n_candidates": len(candidates),
        }
        log["runs"].append(entry)
        print(f"[pysr] max_samples={max_samples} n_fit={n_fit} "
              f"wall={t1-t0:.1f}s internal={backend.diagnostics.get('fit_seconds'):.1f}s "
              f"candidates={len(candidates)}", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
