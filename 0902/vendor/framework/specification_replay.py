"""저장된 구현 명세를 현재 q 정의와 전종목 cache로 다시 실행한다."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import executable
from .config import TICK_ROOT, now_utc, read_json, sha256_json, write_json
from .modules import backtest, parameter_search
from .modules.schedule import ResearchSchedule


SCHEMA = "specification_replay.v1"
SOURCE_GLOB = "*/03_implementation/units/*/source_specification.json"


@dataclass(frozen=True)
class SpecificationReplayRequest:
    source_root: Path
    universe_manifest: Path
    schedule: ResearchSchedule
    workers: int = 8
    resume: bool = True
    anchor_cache_root: Path | None = None
    source_manifest: Path | None = None


def _replace_prior_day_source(value: Any) -> Any:
    """저장 명세의 과거 q source만 현재 직전 100 tick source로 바꾼다."""
    if isinstance(value, list):
        return [_replace_prior_day_source(item) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, str):
            return (value.replace("전일 분위", "직전 100 tick 분위")
                    .replace("전일 분포", "직전 100 tick 분포"))
        return value
    out = {str(key): _replace_prior_day_source(item) for key, item in value.items()}
    source = out.get("threshold_source")
    if isinstance(source, dict) and source.get("kind") == executable.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE:
        source["kind"] = executable.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    return out


def _source_records(root: Path, source_manifest: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if source_manifest is None:
        paths = sorted(Path(root).glob(SOURCE_GLOB))
    else:
        manifest = read_json(Path(source_manifest))
        candidates = manifest.get("candidates") or []
        paths = [Path(str(candidate.get("source_specification_path") or ""))
                 for candidate in candidates]
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError("복합 후보 manifest의 source_specification_path를 읽을 수 없다: "
                             + str(source_manifest))
        paths = sorted(set(paths))
    for path in paths:
        source = read_json(path)
        cluster = path.parents[3].name
        original_id = str(source.get("hypothesis_id") or path.parent.name)
        migrated = _replace_prior_day_source(copy.deepcopy(source))
        signature = sha256_json({
            "signal_template": migrated.get("signal_template"),
            "allowed_parameters": (migrated.get("search_boundary") or {}).get("allowed_parameters"),
            "fixed_semantics": (migrated.get("search_boundary") or {}).get("fixed_semantics"),
        })[:16]
        records.append({
            "path": str(path), "cluster": cluster, "source_hypothesis_id": original_id,
            "signature": signature, "specification": migrated,
        })
    if not records:
        raise ValueError("03_implementation의 source_specification.json을 찾지 못했다: " + str(root))
    return records


def _replay_specification(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """동일 entry 식은 한 번만 실행할 수 있도록 충돌 없는 replay id를 붙인다."""
    replay_id = "Q100_" + str(record["signature"])
    spec = copy.deepcopy(record["specification"])
    spec["source_hypothesis_id"] = str(record["source_hypothesis_id"])
    spec["hypothesis_id"] = replay_id
    invariants = spec.get("semantic_invariants")
    if isinstance(invariants, dict):
        invariants["hypothesis_id"] = replay_id
    return replay_id, spec


def _universe(manifest_path: Path, schedule: ResearchSchedule) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    manifest = read_json(manifest_path)
    if manifest.get("state") != "COMPLETE":
        raise ValueError("전종목 raw cache manifest가 COMPLETE가 아니다: " + str(manifest_path))
    required = tuple(dict.fromkeys(
        schedule.search_fit_dates + schedule.search_confirm_dates + schedule.validation_dates
        + schedule.backtest_dates + schedule.terminal_oos_dates))
    records = manifest.get("dates") or {}
    missing = [date for date in required if date not in records]
    if missing:
        raise ValueError("전종목 raw cache에 필요한 날짜가 없다: " + ", ".join(missing))
    by_date = {date: list(records[date].get("symbols") or []) for date in required}
    incomplete = [date for date, symbols in by_date.items() if not symbols]
    if incomplete:
        raise ValueError("전종목 raw cache 종목 목록이 비었다: " + ", ".join(incomplete))
    symbols = tuple(sorted({symbol for values in by_date.values() for symbol in values}))
    return symbols, by_date


def _write_progress(output: Path, *, stage: str, completed: int, total: int) -> None:
    write_json(Path(output) / "replay_progress.json", {
        "schema": SCHEMA, "updated_at": now_utc(), "stage": stage,
        "completed": completed, "total": total,
    })


def _count_states(values: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("state") or "UNKNOWN")
                               for item in values.values()).items()))


def _validation_supported(summary: Mapping[str, Any]) -> bool:
    metrics = summary.get("metrics") or {}
    return (summary.get("state") == "BACKTEST_COMPLETE"
            and float(metrics.get("scorable") or 0) > 0
            and float(metrics.get("net_bps_per_decision") or 0) > 0)


def _run_batch_or_empty(searches: Mapping[str, Mapping[str, Any]], *, symbols: Sequence[str],
                        dates: Sequence[str], output: Path, root: Path, stage: str,
                        workers: int) -> dict[str, dict[str, Any]]:
    if not searches:
        return {}
    return backtest.run_batch(searches, symbols=symbols, dates=dates, output=output,
                              root=root, stage=stage, workers=workers)


def run(request: SpecificationReplayRequest, output: Path, *, root: Path = TICK_ROOT) -> dict[str, Any]:
    """새 q=직전 100 tick 정의로 Search→Validation→Final→OOS를 순서대로 실행한다."""
    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    source_records = _source_records(request.source_root, request.source_manifest)
    symbols, symbols_by_date = _universe(request.universe_manifest, request.schedule)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in source_records:
        grouped.setdefault(str(record["signature"]), []).append(record)
    representative = {signature: records[0] for signature, records in grouped.items()}
    # Parameter Search는 이제 30초 anchor 표를 읽지 않는다. 모든 q를 V9 full-tick
    # 원장으로 실행하므로, 기존 anchor cache가 있어도 선택 입력으로 쓰지 않는다.
    anchor_cache_root = Path(request.anchor_cache_root or output / "00_search_anchor_cache")
    source_manifest = {
        "schema": SCHEMA,
        "created_at": now_utc(),
        "source_root": str(Path(request.source_root).resolve()),
        "source_candidate_manifest": (str(Path(request.source_manifest).resolve())
                                      if request.source_manifest is not None else None),
        "universe_manifest": str(Path(request.universe_manifest).resolve()),
        "universe": {"unique_symbols": len(symbols), "symbols_by_date": symbols_by_date},
        "schedule": request.schedule.as_input(),
        "q_definition": "symbol/date current tick 제외 직전 100 tick 분위",
        "search_anchor_cache": {"path": str(anchor_cache_root.resolve()),
                                "state": "NOT_USED_BY_EXECUTION_PARAMETER_SEARCH",
                                "used_for_selection": False},
        "source_specifications": len(source_records),
        "unique_entry_expressions": len(representative),
        "sources": [{key: value for key, value in record.items() if key != "specification"}
                    for record in source_records],
    }
    write_json(output / "source_manifest.json", source_manifest)

    searches: dict[str, dict[str, Any]] = {}
    search_root = output / "01_parameter_search"
    for index, signature in enumerate(sorted(representative), start=1):
        record = representative[signature]
        replay_id, spec = _replay_specification(record)
        unit = search_root / "units" / replay_id
        result_path = unit / "search_result.json"
        existing = read_json(result_path) if request.resume and result_path.is_file() else None
        is_execution_selection = ((existing or {}).get("plan") or {}).get("selection_mode") \
            == parameter_search.SELECTION_MODE
        if existing is not None and existing.get("state") != "ERROR" and is_execution_selection:
            result = existing
        else:
            try:
                result = parameter_search.run(spec, symbols=symbols, schedule=request.schedule,
                                              output=unit, root=root,
                                              anchor_cache_root=anchor_cache_root)
            except Exception as error:  # 한 구현 명세 오류가 전체 재생을 멈추지 않는다.
                result = {"state": "ERROR", "error": f"{type(error).__name__}: {error}"}
            write_json(result_path, result)
        searches[replay_id] = result
        _write_progress(output, stage="PARAMETER_SEARCH", completed=index,
                        total=len(representative))
    write_json(output / "01_parameter_search" / "summary.json", {
        "schema": SCHEMA, "state_counts": _count_states(searches),
        "locked": sum(bool(value.get("lock")) for value in searches.values()),
    })

    locked = {key: value for key, value in searches.items() if value.get("lock")}
    validation_path = output / "02_validation" / "results.json"
    validation = (read_json(validation_path) if request.resume and validation_path.is_file()
                  else _run_batch_or_empty(
                      locked, symbols=symbols, dates=request.schedule.validation_dates,
                      output=output / "02_validation", root=root, stage="VALIDATION_BACKTEST",
                      workers=int(request.workers)))
    write_json(validation_path, validation)

    supported = {key: locked[key] for key, result in validation.items()
                 if key in locked and _validation_supported(result)}
    final_path = output / "03_final" / "results.json"
    final = (read_json(final_path) if request.resume and final_path.is_file()
             else _run_batch_or_empty(
                 supported, symbols=symbols, dates=request.schedule.backtest_dates,
                 output=output / "03_final", root=root, stage="FINAL_BACKTEST",
                 workers=int(request.workers)))
    write_json(final_path, final)

    oos_inputs = {key: supported[key] for key, result in final.items()
                  if key in supported and result.get("state") == "BACKTEST_COMPLETE"}
    oos_path = output / "04_terminal_oos" / "results.json"
    oos = (read_json(oos_path) if request.resume and oos_path.is_file()
           else _run_batch_or_empty(
               oos_inputs, symbols=symbols, dates=request.schedule.terminal_oos_dates,
               output=output / "04_terminal_oos", root=root, stage="TERMINAL_OOS_BACKTEST",
               workers=int(request.workers)))
    write_json(oos_path, oos)

    unit_results = {}
    for signature, records in grouped.items():
        replay_id = "Q100_" + signature
        unit_results[replay_id] = {
            "source_count": len(records),
            "source_hypothesis_ids": sorted({str(record["source_hypothesis_id"])
                                               for record in records}),
            "parameter_search": searches.get(replay_id),
            "validation": validation.get(replay_id),
            "final": final.get(replay_id),
            "terminal_oos": oos.get(replay_id),
        }
    summary = {
        **source_manifest,
        "state": "SPECIFICATION_REPLAY_COMPLETE",
        "parameter_search": {"state_counts": _count_states(searches), "locked": len(locked)},
        "validation": {"state_counts": _count_states(validation), "supported": len(supported)},
        "final": {"state_counts": _count_states(final), "executed": len(oos_inputs)},
        "terminal_oos": {"state_counts": _count_states(oos), "executed": len(oos)},
        "units": unit_results,
    }
    write_json(output / "summary.json", summary)
    _write_progress(output, stage="COMPLETE", completed=len(representative),
                    total=len(representative))
    return summary
