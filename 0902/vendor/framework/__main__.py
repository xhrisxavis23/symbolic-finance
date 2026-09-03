"""정본 CLI. 연구 진입점은 자유 진입 조건 탐색 하나뿐이다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TICK_ROOT
from .workflows.free_research import (REFINEMENT_POPULATIONS, FreeResearchRequest,
                                      run as run_free_research)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m framework")
    commands = root.add_subparsers(dest="command", required=True)

    research = commands.add_parser(
        "research",
        help=("Agent가 만든 LONG 진입 조건을 Discovery에서 탐색·Refinement한 뒤 "
              "Validation/Final에 고정 재생한다"),
    )
    research.add_argument("--output", required=True, type=Path)
    research.add_argument("--symbol", action="append", required=True)
    research.add_argument("--discovery-date", action="append", required=True)
    research.add_argument("--validation-date", action="append", required=True)
    research.add_argument(
        "--final-date", "--backtest-date", dest="final_date", action="append", required=True
    )
    research.add_argument(
        "--strategy-source", type=Path,
        help="Agent 호출 대신 저장된 진입 조건 JSON을 그대로 사용한다",
    )
    research.add_argument("--strategy-count", type=int, default=8)
    research.add_argument("--model", default="gpt-5.6-luna")
    research.add_argument("--effort", default="high")
    research.add_argument(
        "--agent-timeout", type=int, default=480,
        help="Agent 호출 하나의 초 단위 한도. 프롬프트가 크고 effort 가 높으면 늘린다",
    )
    research.add_argument(
        "--trials-per-strategy", type=int, default=50,
        help="전략마다 허용할 파라미터 격자 상한. 격자는 전수로 돈다",
    )
    research.add_argument(
        "--optuna-processes", type=int, default=4,
        help="동시에 실행할 Optuna 백테스트 프로세스 수",
    )
    research.add_argument("--optuna-seed", type=int, default=1729)
    research.add_argument(
        "--refinement-rounds", type=int, default=1,
        help="Discovery 손실 장부 Refinement 라운드 수, 0은 건너뛰고 최대 10",
    )
    research.add_argument(
        "--refinement-max-proposals", type=int, default=3,
        help="각 Refinement 분기에서 Agent가 만들 최대 진입식 수",
    )
    research.add_argument(
        "--refinement-population", action="append", choices=REFINEMENT_POPULATIONS,
        help="손실 장부 관점. 생략하면 ALL_DECISIONS과 FILLED_ONLY 둘 다 사용",
    )
    research.add_argument(
        "--workers", type=int, default=8,
        help="Optuna 프로세스들이 나눠 쓰는 백테스트 worker 수",
    )

    cache = commands.add_parser(
        "common-stock-cache",
        help="연구에 필요한 주식 전종목 raw tick array cache만 준비한다",
    )
    source = cache.add_mutually_exclusive_group(required=True)
    source.add_argument("--listing", type=Path)
    source.add_argument("--stock-batch", action="store_true")
    cache.add_argument("--date", action="append", required=True)
    cache.add_argument("--output", required=True, type=Path)
    cache.add_argument("--tick-root", type=Path, default=TICK_ROOT)
    cache.add_argument("--workers", type=int, default=1)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "research":
        result = run_free_research(
            FreeResearchRequest(
                symbols=tuple(args.symbol),
                discovery_dates=tuple(args.discovery_date),
                validation_dates=tuple(args.validation_date),
                final_dates=tuple(args.final_date),
                strategy_source=args.strategy_source,
                strategy_count=int(args.strategy_count),
                model=str(args.model),
                effort=str(args.effort),
                agent_timeout_seconds=int(args.agent_timeout),
                workers=int(args.workers),
                trials_per_strategy=int(args.trials_per_strategy),
                optuna_processes=int(args.optuna_processes),
                optuna_seed=int(args.optuna_seed),
                refinement_rounds=int(args.refinement_rounds),
                refinement_max_proposals=int(args.refinement_max_proposals),
                refinement_populations=tuple(
                    args.refinement_population or REFINEMENT_POPULATIONS),
            ),
            args.output,
        )
        print(json.dumps({
            "state": result["state"], "artifacts": result["artifacts"],
            "output": str(args.output),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "common-stock-cache":
        from .modules import common_stock_cache

        result = common_stock_cache.materialize(
            args.listing,
            args.date,
            profile_store=args.output,
            root=args.tick_root,
            stock_batch=bool(args.stock_batch),
            raw_only=True,
            workers=int(args.workers),
        )
        print(json.dumps({
            "state": result["state"], "scope": result["scope"],
            "manifest": result["manifest"],
        }, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"알 수 없는 명령: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
