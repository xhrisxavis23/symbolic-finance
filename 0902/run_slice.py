#!/usr/bin/env python3
"""얇은 수직 슬라이스. DESIGN.md §5 데이터 흐름 그대로.

    python3 run_slice.py --per-stratum 5 --max-samples 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sd import config, dimensionless, labels, manifold, replay, report, select, ticks, universe
from sd.compile import compile_candidates
from sd.sr.naive import NaiveBackend
from sd.teacher.gate import evaluate as evaluate_gate
from sd.teacher.shallow import ShallowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=config.SLICE_PER_STRATUM)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bottleneck", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--workers", type=int, default=config.REPLAY_WORKERS)
    parser.add_argument("--ignore-gate", action="store_true",
                        help="S1 게이트 불통과를 무시한다. provenance 에 기록된다")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    settings = {k: v for k, v in vars(args).items()}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.RUNS_ROOT / f"{stamp}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    # S−1 우주와 층화
    all_symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(all_symbols, config.DATE)
    strata = universe.assign_strata(stats)
    symbols = universe.slice_symbols(strata, args.per_stratum)
    print(f"[S-1] 전종목 {len(all_symbols)} → 층화 {len(strata)} → 슬라이스 {len(symbols)}")

    boundaries = strata.groupby("friction").spread_bps.agg(["min", "max", "size"])
    universe_record = {
        "date": config.DATE,
        "total_symbols": len(all_symbols),
        "stratified_symbols": len(strata),
        "symbols": list(symbols),
        "per_stratum": args.per_stratum,
        "friction_boundaries": json.loads(boundaries.to_json(orient="index")),
        "stratum_of": {s: str(strata.loc[s, "stratum"]) for s in symbols},
    }

    # S0~S2 종목별 적재 → 라벨 → 무차원 → 표집
    #
    # 체결 행이 없는 종목-일은 체결 의존 feature 가 계산되지 않아 열이 줄어든다.
    # 그런 종목을 버리면 유동성 꼬리가 통째로 사라지므로, **열의 교집합**을 쓴다.
    per_symbol: list[tuple[dict[str, np.ndarray], object]] = []
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, config.DATE)
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] {symbol}: {type(error).__name__}: {error}")
            continue
        matrix, names = ticks.feature_matrix(arrays)
        X_symbol, kept, _meta = dimensionless.transform(matrix, names)
        columns = {name: X_symbol[:, j] for j, name in enumerate(kept)}
        per_symbol.append((columns, labels.build(arrays)))

    if not per_symbol:
        raise SystemExit("적재된 종목이 없다")

    common = set(per_symbol[0][0])
    for columns, _label in per_symbol[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))
    if not feature_names:
        raise SystemExit("모든 종목에 공통인 무차원 feature 가 없다")
    print(f"[S0] 공통 무차원 feature {len(feature_names)}: {', '.join(feature_names)}")

    X = np.vstack([np.column_stack([columns[name] for name in feature_names])
                   for columns, _label in per_symbol])
    y_path = np.concatenate([label.y_path for _columns, label in per_symbol])
    y_fill = np.concatenate([label.y_fill for _columns, label in per_symbol])
    mask = np.concatenate([label.mask for _columns, label in per_symbol])
    print(f"[S0] 행 {len(X):,} · 학습 가능 {int(mask.sum()):,}")

    selection = manifold.select(X, mask, max_samples=args.max_samples, seed=args.seed)
    print(f"[S2] 표집 {len(selection.index):,} · on-manifold "
          f"{int(selection.on_manifold.sum()):,}")

    # S1 교사 (on-manifold 만 적합에 쓴다)
    fit_rows = selection.index[selection.on_manifold]
    fit_weight = selection.weight[selection.on_manifold]
    teacher = ShallowMLP(n_features=X.shape[1], bottleneck=args.bottleneck,
                         seed=args.seed).fit(
        X[fit_rows], y_path[fit_rows], y_fill[fit_rows], fit_weight, epochs=args.epochs)
    print(f"[S1] 교사 학습 완료 (병목 {args.bottleneck})")

    gate_result = evaluate_gate(teacher, X[fit_rows], y_path[fit_rows],
                                y_fill[fit_rows], np.ones(len(fit_rows), dtype=bool))
    for name, check in gate_result.checks.items():
        print(f"[S1] {'통과' if check['passed'] else '실패'}  {name}: "
              + ", ".join(f"{k}={v}" for k, v in check.items()
                          if k not in {"passed", "curve", "why"}))
    if not gate_result.passed and not args.ignore_gate:
        raise SystemExit(
            "S1 교사 검증 게이트 불통과. 오염된 교사에서 증류하면 그 오염을 수식으로 "
            "고정할 뿐이다. 무시하려면 --ignore-gate 를 준다 (그 사실이 산출물에 남는다)")

    # S3 SR — feature 공간에서 교사 출력을 근사한다 (DESIGN.md D13)
    target = teacher.predict_path(X[fit_rows])
    candidates = NaiveBackend(seed=args.seed).fit(
        X[fit_rows], target, fit_weight, feature_names)
    print(f"[S3] SR 후보 {len(candidates)}")

    # S5 컴파일
    compiled, failures = compile_candidates(candidates)
    rate = (len(compiled) / len(candidates)) if candidates else 0.0
    print(f"[S5] 컴파일 성공 {len(compiled)} / {len(candidates)} ({rate:.0%}), 탈락 {len(failures)}")
    if not compiled:
        raise SystemExit("컴파일된 진입식이 없다. 계획서 F7 에 해당한다")

    # S5⑤ 정본 재생
    result = replay.run(compiled, symbols=symbols, date=config.DATE,
                        output_dir=run_dir, grid=config.QUANTILE_GRID,
                        workers=args.workers)
    ledger = pd.read_parquet(result.ledger_path)
    print(f"[S5] 원장 {len(ledger):,}행 · 시도 N={result.attempts}")

    # S4 선택
    summary = select.summarise(ledger)
    ranked = select.rank(summary)
    print(f"[S4] 자격 통과 {len(ranked)} / {len(summary)}")

    prov = report.provenance(symbols=symbols, seed=args.seed,
                             sr_backend=NaiveBackend.name, grid=config.QUANTILE_GRID,
                             attempts=result.attempts, bottleneck=args.bottleneck,
                             gate_passed=gate_result.passed,
                             gate_ignored=bool(args.ignore_gate),
                             gate_checks=gate_result.checks)
    summary.to_parquet(run_dir / "summary.parquet")
    path = report.write(run_dir, universe_record, candidates, compiled, failures,
                        ranked, prov)
    print(f"[done] {time.time() - started:.1f}s → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
