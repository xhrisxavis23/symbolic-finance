"""자유 진입 연구 1회 확인 뒤 같은 날짜 경계로 50회를 순차 실행한다."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Mapping

from ..config import TICK_ROOT, now_utc, read_json, sha256_file, sha256_json, write_json
from ..modules.common_stock_cache import stock_batch_symbols
from .free_research import FreeResearchRequest, run


def _symbols(date: str, root: Path) -> tuple[str, ...]:
    selected = set(stock_batch_symbols(date, root))
    counts: dict[str, int] = {}
    for path in (Path(root) / str(date)).glob("*.parquet"):
        match = re.fullmatch(r"([0-9]{6})_.*\.parquet", path.name)
        if match:
            symbol = match.group(1)
            counts[symbol] = counts.get(symbol, 0) + 1
    return tuple(sorted(symbol for symbol in selected if counts.get(symbol) == 1))


def _universes(dates: Mapping[str, str], root: Path) -> dict[str, tuple[str, ...]]:
    return {stage: _symbols(date, root) for stage, date in dates.items()}


def _request(dates: Mapping[str, str], universes: Mapping[str, tuple[str, ...]], *,
             seed: int, workers: int, optuna_batch_size: int) -> FreeResearchRequest:
    all_symbols = tuple(sorted(set().union(*map(set, universes.values()))))
    return FreeResearchRequest(
        symbols=all_symbols,
        discovery_dates=(dates["discovery"],),
        validation_dates=(dates["validation"],),
        final_dates=(dates["final"],),
        discovery_symbols=universes["discovery"],
        validation_symbols=universes["validation"],
        final_symbols=universes["final"],
        strategy_count=8,
        workers=int(workers),
        trials_per_strategy=50,
        optuna_processes=int(optuna_batch_size),
        optuna_seed=int(seed),
        refinement_rounds=1,
        refinement_max_proposals=3,
    )


def _complete(path: Path) -> bool:
    artifact = Path(path) / "research_run.json"
    return artifact.is_file() and read_json(artifact).get("payload", {}).get("state") == "FINAL_COMPLETE"


def _metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    values = summary.get("metrics") or {}
    return {
        "state": summary.get("state"),
        "net_bps_total": values.get("net_bps_total"),
        "net_bps_per_decision": values.get("net_bps_per_decision"),
        "net_bps_per_fill": values.get("net_bps_per_fill"),
        "decisions": values.get("scorable"),
        "fills": values.get("fills"),
        "fill_rate": values.get("fill_rate"),
        "ledger": summary.get("ledger"),
        "ledger_contract_id": summary.get("ledger_contract_id"),
    }


def aggregate(output: Path, run_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    completed = 0
    for index in range(1, int(run_count) + 1):
        run_root = Path(output) / "runs" / f"run_{index:03d}"
        final_path = run_root / "06_final" / "final_artifact.json"
        validation_path = run_root / "05_validation" / "validation_artifact.json"
        refinement_path = run_root / "03_refinement" / "refinement_artifact.json"
        if not (final_path.is_file() and validation_path.is_file() and refinement_path.is_file()):
            continue
        completed += 1
        final = read_json(final_path)["payload"]["units"]
        validation = read_json(validation_path)["payload"]["units"]
        refinement = read_json(refinement_path)["payload"]
        parent_ids = set(
            strategy_id for strategy_id, unit in refinement["units"].items()
            if "refinement_lineage" not in unit
        )
        for strategy_id, final_summary in final.items():
            row = {
                "run": index,
                "strategy_id": strategy_id,
                "candidate_kind": "PARENT" if strategy_id in parent_ids else "REFINED",
            }
            row.update({f"validation_{key}": value
                        for key, value in _metrics(validation[strategy_id]).items()})
            row.update({f"final_{key}": value
                        for key, value in _metrics(final_summary).items()})
            rows.append(row)

    csv_path = Path(output) / "campaign_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema": "free_research_campaign_summary.v1",
        "created_at": now_utc(),
        "requested_runs": int(run_count),
        "completed_runs": completed,
        "candidate_rows": len(rows),
        "parent_rows": sum(row["candidate_kind"] == "PARENT" for row in rows),
        "refined_rows": sum(row["candidate_kind"] == "REFINED" for row in rows),
        "final_positive_net_candidates": sum(
            row.get("final_net_bps_total") is not None
            and float(row["final_net_bps_total"]) > 0.0 for row in rows),
        "results_csv": str(csv_path),
        "results_sha256": sha256_file(csv_path) if csv_path.is_file() else None,
    }
    write_json(Path(output) / "campaign_summary.json", summary)
    return summary


def execute(output: Path, *, dates: Mapping[str, str], run_count: int = 50,
            workers: int = 32, optuna_batch_size: int = 32,
            seed: int = 1729, root: Path = TICK_ROOT) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    universes = _universes(dates, Path(root))
    universe_record = {
        "schema": "free_research_campaign_universe.v1",
        "created_at": now_utc(),
        "dates": dict(dates),
        "selection": "stock_batch asset_type=ST and exactly one tick parquet",
        "stages": {
            stage: {"count": len(symbols), "sha256": sha256_json(list(symbols)),
                    "symbols": list(symbols)}
            for stage, symbols in universes.items()
        },
    }
    write_json(output / "universe.json", universe_record)

    progress_path = output / "progress.json"
    progress = {
        "schema": "free_research_campaign_progress.v1",
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "state": "SMOKE_RUNNING",
        "dates": dict(dates),
        "requested_runs": int(run_count),
        "completed_runs": 0,
        "active_run": "smoke",
        "errors": [],
    }
    write_json(progress_path, progress)

    smoke_root = output / "smoke"
    if not _complete(smoke_root):
        started = time.monotonic()
        try:
            result = run(_request(dates, universes, seed=seed, workers=workers,
                                  optuna_batch_size=optuna_batch_size),
                         smoke_root, root=Path(root))
            if result["state"] != "FINAL_COMPLETE":
                raise RuntimeError(f"smoke가 Final까지 끝나지 않았다: {result['state']}")
        except Exception as error:
            progress.update({"updated_at": now_utc(), "state": "SMOKE_FAILED",
                             "active_run": None,
                             "errors": [f"{type(error).__name__}: {error}"]})
            write_json(progress_path, progress)
            raise
        progress["smoke_wall_seconds"] = time.monotonic() - started

    for index in range(1, int(run_count) + 1):
        run_root = output / "runs" / f"run_{index:03d}"
        if _complete(run_root):
            progress["completed_runs"] = index
            continue
        progress.update({"updated_at": now_utc(), "state": "CAMPAIGN_RUNNING",
                         "active_run": index, "completed_runs": index - 1})
        write_json(progress_path, progress)
        started = time.monotonic()
        try:
            result = run(_request(dates, universes, seed=seed + index, workers=workers,
                                  optuna_batch_size=optuna_batch_size),
                         run_root, root=Path(root))
            if result["state"] != "FINAL_COMPLETE":
                raise RuntimeError(
                    f"run_{index:03d}가 Final까지 끝나지 않았다: {result['state']}")
        except Exception as error:
            progress.update({"updated_at": now_utc(), "state": "CAMPAIGN_FAILED",
                             "active_run": index,
                             "errors": [f"run_{index:03d}: {type(error).__name__}: {error}"]})
            write_json(progress_path, progress)
            raise
        progress.update({"updated_at": now_utc(), "completed_runs": index,
                         "active_run": None, "last_run_wall_seconds": time.monotonic() - started})
        write_json(progress_path, progress)

    summary = aggregate(output, run_count)
    progress.update({"updated_at": now_utc(), "state": "CAMPAIGN_COMPLETE",
                     "completed_runs": int(run_count), "active_run": None,
                     "summary": summary})
    write_json(progress_path, progress)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--discovery-date", required=True)
    parser.add_argument("--validation-date", required=True)
    parser.add_argument("--final-date", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--optuna-batch-size", type=int, default=32)
    args = parser.parse_args()
    execute(
        args.output,
        dates={"discovery": args.discovery_date,
               "validation": args.validation_date,
               "final": args.final_date},
        run_count=int(args.runs), workers=int(args.workers),
        optuna_batch_size=int(args.optuna_batch_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
