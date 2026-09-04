#!/usr/bin/env python3
"""E0 게이트. 계획서 §4 E0 · ROADMAP.md 3단계 — 알려진 다섯 법칙을 증류로
재발견할 수 있는가. **진입식을 만들지 않는다.** `sd/compile/`·`sd/replay.py`
를 타지 않는다.

    # 배관 검증 — 법칙 하나, 적은 종목, 적은 SR 반복
    python3 run_e0.py --laws L2 --per-stratum 1 --sr-niterations 8

    # 본 실행 (2026-09-04 실측: 5법칙×4트랙, 이 기본값으로 약 30~40분,
    # 머신 부하 20~30 기준 — 시간 측정을 회귀 판단에 쓰지 않는다)
    python3 run_e0.py
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from sd import config, universe
from sd.e0 import LAWS, report as e0_report, runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laws", nargs="+", default=list(LAWS), choices=list(LAWS))
    parser.add_argument("--per-stratum", type=int, default=6,
                        help="6개 층 × per-stratum 종목. 계획서 §3 S2 층화 그대로")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--bottleneck", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--sr-niterations", type=int, default=18)
    parser.add_argument("--sr-maxsize", type=int, default=18)
    parser.add_argument("--max-manifold-samples", type=int, default=1500,
                        help="법칙 하나당 SR 이 보는 최대 행 수. PySR fit 시간이 이 값에 "
                             "가장 민감하다 — 2026-09-04 실측(36종목, 이 기본값): 법칙 하나당 "
                             "약 200~400초(4트랙 합)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.RUNS_ROOT / f"e0-{stamp}"
    print(f"[e0] run_dir={run_dir}")

    all_symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(all_symbols, config.DATE)
    strata = universe.assign_strata(stats)
    symbols = universe.slice_symbols(strata, args.per_stratum)
    print(f"[e0] 전종목 {len(all_symbols)} -> 층화 {len(strata)} -> 슬라이스 {len(symbols)}")

    results = {}
    for law in args.laws:
        print(f"[e0] === {law} 시작 ===", flush=True)
        t0 = time.time()
        result = runner.run_law(
            law, symbols, config.DATE, seed=args.seed, bottleneck=args.bottleneck,
            epochs=args.epochs, sr_niterations=args.sr_niterations,
            sr_maxsize=args.sr_maxsize, max_manifold_samples=args.max_manifold_samples)
        dt = time.time() - t0
        main = result.tracks.get("main")
        print(f"[e0] {law} 완료 {dt:.1f}s — 종목 {len(result.symbols_used)}, "
              f"행 {result.n_rows_total:,}, main 후보 {len(main.candidates) if main else 0}, "
              f"main 복원={result.recovered}", flush=True)
        results[law] = result

    path = e0_report.write(run_dir, results, symbols=symbols, args=vars(args))
    n_recovered = sum(1 for r in results.values() if r.recovered)
    print(f"[e0] 게이트: {n_recovered} / {len(results)} 복원")
    print(f"[e0] done {time.time() - started:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
