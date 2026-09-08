#!/usr/bin/env python3
"""E0 게이트. 계획서 §4 E0 · ROADMAP.md 3단계 — 알려진 다섯 법칙을 증류로
재발견할 수 있는가. **진입식을 만들지 않는다.** `sd/compile/`·`sd/replay.py`
를 타지 않는다.

    # 배관 검증 — 법칙 하나, 적은 종목, 적은 SR 반복. `--per-stratum` 은
    # 최소 2 여야 한다 — 1이면 층마다 종목이 하나뿐이라 표본외 분할이
    # 선택 종목을 하나도 못 만든다(sd.e0.split 이 그 층 전부를 적합으로
    # 보낸다 — 모듈 docstring 참고).
    python3 run_e0.py --laws L2 --per-stratum 3 --sr-niterations 8

    # 본 실행 (2026-09-04 실측: 5법칙×4트랙, 이 기본값으로 약 30~40분,
    # 머신 부하 20~30 기준 — 시간 측정을 회귀 판단에 쓰지 않는다)
    python3 run_e0.py
"""

from __future__ import annotations

import argparse
import functools
import time
from datetime import datetime, timezone

from sd import config, universe
from sd.e0 import LAWS, report as e0_report, runner, split as e0_split
from sd.e0.teacher import ScalarDeepLOB, ScalarTeacher


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
    parser.add_argument("--fit-fraction", type=float, default=e0_split.DEFAULT_FIT_FRACTION,
                        help="종목 단위 표본외 분할에서 적합 종목 비율(층별로 유지). "
                             "PREREG-E0-V2.md §1-1 — 나머지가 후보 채점·판정에 쓰는 선택 "
                             "종목이다. 기본값 근거는 sd/e0/split.py 모듈 docstring")
    parser.add_argument("--teacher", choices=("shallow", "deeplob"), default="shallow",
                        help="증류 교사 선택(PREREG-E0-V2.md §1-2). shallow=ScalarTeacher"
                             "(사전등록 v2/v2b 기본값, 시간 윈도우 없음). "
                             "deeplob=ScalarDeepLOB(CNN→Inception→LSTM→병목, "
                             "sd.teacher.deeplob._DeepLOBEncoder 재사용 — DESIGN.md D5, "
                             "31e4a22). SR 이 보는 X 는 어느 쪽을 골라도 항상 원본 feature "
                             "공간이다 — 이 인자는 교사(증류 타깃)에만 영향을 준다.")
    parser.add_argument("--teacher-window", type=int, default=16,
                        help="--teacher deeplob 일 때 시간 윈도우 스텝 수"
                             "(sd.teacher.window.make_causal_windows). --teacher shallow 면 "
                             "무시된다(내부적으로 1 로 고정 — 윈도우 없음과 동일한 항등 경로).")
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

    # 종목 단위 표본외 분할 (PREREG-E0-V2.md §1-1) — 법칙 5개가 전부 같은
    # 분할을 공유한다. 법칙마다 다른 표본외 기준을 쓰면 게이트 판정이
    # 비교 불가능해진다.
    stratum_of = strata["stratum"].to_dict()
    symbol_split = e0_split.split_symbols(symbols, stratum_of, seed=args.seed,
                                          fit_fraction=args.fit_fraction)
    print(f"[e0] 표본외 분할(seed={args.seed}, fit_fraction={args.fit_fraction}): "
          f"적합 {len(symbol_split.fit)}종목 / 선택 {len(symbol_split.select)}종목 "
          f"— 층별 {symbol_split.per_stratum_counts}")

    # 교사 선택 (PREREG-E0-V2.md §1-2). shallow 는 윈도우가 없다 — teacher_window=1
    # 은 sd.e0.runner._maybe_window 산술상 항등 변환이라 v2/v2b 와 배선이 완전히
    # 같다. deeplob 은 window 를 --teacher-window 로 묶어(functools.partial)
    # run_law 호출부는 n_features·bottleneck·seed 만 넘기게 유지한다.
    if args.teacher == "deeplob":
        teacher_cls = functools.partial(ScalarDeepLOB, window=args.teacher_window)
        teacher_window = args.teacher_window
    else:
        teacher_cls = ScalarTeacher
        teacher_window = 1
    print(f"[e0] 교사={args.teacher}"
          + (f" (window={teacher_window})" if args.teacher == "deeplob" else ""))

    results = {}
    for law in args.laws:
        print(f"[e0] === {law} 시작 ===", flush=True)
        t0 = time.time()
        result = runner.run_law(
            law, symbol_split.fit, symbol_split.select, config.DATE, seed=args.seed,
            bottleneck=args.bottleneck, epochs=args.epochs, sr_niterations=args.sr_niterations,
            sr_maxsize=args.sr_maxsize, max_manifold_samples=args.max_manifold_samples,
            teacher_cls=teacher_cls, teacher_window=teacher_window)
        dt = time.time() - t0
        main = result.tracks.get("main")
        print(f"[e0] {law} 완료 {dt:.1f}s — 적합종목 {len(result.symbols_used)}, "
              f"선택종목 {len(result.select_symbols_used)}, 행 {result.n_rows_total:,}, "
              f"main 후보 {len(main.candidates) if main else 0}, "
              f"main 복원={result.recovered}", flush=True)
        results[law] = result

    path = e0_report.write(run_dir, results, symbols=symbols, args=vars(args),
                           symbol_split=symbol_split)
    n_recovered = sum(1 for r in results.values() if r.recovered)
    print(f"[e0] 게이트: {n_recovered} / {len(results)} 복원")
    print(f"[e0] done {time.time() - started:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
