"""Parameter lock → V9·V8-900 Validation Backtest set artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .. import canonical as K, search as S
from ..config import TICK_ROOT
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import backtest
from ..modules.schedule import ResearchSchedule


ARTIFACT_NAME = "validation_artifact.json"


def _validation_result(summary: Mapping[str, Any]) -> str:
    if summary.get("state") != "BACKTEST_COMPLETE":
        return str(summary.get("state") or "NO_PARAMETER_LOCK")
    evaluation = summary.get("profit_target_evaluation")
    if not isinstance(evaluation, Mapping):
        return "VALIDATION_BACKTEST_INCONCLUSIVE"
    if evaluation.get("accounting_identity_ok") is not True:
        return "VALIDATION_BACKTEST_ACCOUNTING_FAILURE"
    metrics = summary.get("metrics") or {}
    if float(metrics.get("scorable") or 0) <= 0:
        return "VALIDATION_BACKTEST_INCONCLUSIVE"
    return ("VALIDATION_BACKTEST_SUPPORTED"
            if evaluation.get("success") is True
            else "VALIDATION_BACKTEST_NOT_SUPPORTED")


def _promotion_profile_ids(v9_state: str) -> list[str]:
    """정본 V9가 실행 손익으로 지지한 경우에만 승격 프로필을 남긴다."""
    return [K.PROFILE_ID] if v9_state == "VALIDATION_BACKTEST_SUPPORTED" else []


def run(parameter_search_path: Path, evidence_path: Path, output: Path, *,
        schedule: ResearchSchedule, root: Path = TICK_ROOT) -> dict[str, Any]:
    """잠긴 parameter를 새 날짜의 정본 Backtest로 검증한다."""
    search = read_artifact(parameter_search_path, kind="parameter_search_set")
    evidence = read_artifact(evidence_path, kind="evidence_package")
    if search["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Parameter Search와 Evidence의 lineage가 다르다")
    discovery_dates = tuple(evidence["payload"].get("profile_dates") or [])
    schedule.validate(discovery_dates)
    symbols = tuple(evidence["payload"].get("profile_symbols") or [])
    if not symbols:
        raise ValueError("Evidence artifact에 Profile 종목이 없다")
    output = Path(output)
    search_units = search["payload"].get("units", {})
    runnable = {
        str(hypothesis_id): result for hypothesis_id, result in search_units.items()
        if (result.get("lock") or {}).get("status") in S.PARAMETER_LOCK_STATUSES
    }
    batched = len(runnable) > 1
    primary: dict[str, dict[str, Any]] = (
        backtest.run_batch(runnable, symbols=symbols, dates=schedule.validation_dates,
                           output=output, root=Path(root), stage="VALIDATION_BACKTEST")
        if batched else {})
    completed = {
        hypothesis_id: runnable[hypothesis_id] for hypothesis_id, summary in primary.items()
        if summary.get("state") == "BACKTEST_COMPLETE" and hypothesis_id in runnable
    }
    comparison: dict[str, dict[str, Any]] = {}
    if batched and len(completed) > 1:
        comparison = backtest.run_batch(
            completed, symbols=symbols, dates=schedule.validation_dates, output=output,
            root=Path(root), stage="VALIDATION_BACKTEST_COMPARISON",
            canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
    elif batched and completed:
        hypothesis_id, search_result = next(iter(completed.items()))
        comparison[hypothesis_id] = backtest.run(
            search_result, symbols=symbols, dates=schedule.validation_dates,
            output=output / "units" / hypothesis_id / "v8_900_comparison", root=Path(root),
            stage="VALIDATION_BACKTEST_COMPARISON", canonical=K.V8_900_COMPARISON,
            comparison_of=K.CANONICAL)
    units: dict[str, Any] = {}
    for hypothesis_id, search_result in search_units.items():
        hypothesis_id = str(hypothesis_id)
        if batched:
            summary = primary.get(hypothesis_id)
            if summary is None:
                summary = backtest.run(
                    search_result, symbols=symbols, dates=schedule.validation_dates,
                    output=output / "units" / hypothesis_id, root=Path(root),
                    stage="VALIDATION_BACKTEST")
            comparison_result = comparison.get(hypothesis_id)
        else:
            summary = backtest.run(
                search_result, symbols=symbols, dates=schedule.validation_dates,
                output=output / "units" / hypothesis_id, root=Path(root),
                stage="VALIDATION_BACKTEST")
            comparison_result = None
            if summary.get("state") == "BACKTEST_COMPLETE":
                comparison_result = backtest.run(
                    search_result, symbols=symbols, dates=schedule.validation_dates,
                    output=output / "units" / hypothesis_id / "v8_900_comparison",
                    root=Path(root), stage="VALIDATION_BACKTEST_COMPARISON",
                    canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
        v9_state = _validation_result(summary)
        comparison_state = (_validation_result(comparison_result)
                            if comparison_result is not None else None)
        promotion_profile_ids = _promotion_profile_ids(v9_state)
        units[hypothesis_id] = {
            "state": v9_state,
            "backtest": summary,
            **({"v8_900_comparison_backtest": comparison_result}
               if comparison_result is not None else {}),
            "v9_validation_state": v9_state,
            "v8_900_validation_state": comparison_state,
            "promotion_profile_ids": promotion_profile_ids,
            "profit_target_evaluation": summary.get("profit_target_evaluation"),
            "validation_metric": "V9에서 profit_target_evaluation.success == true",
            "v8_900_role": "COMPARISON_ONLY_NO_PROMOTION_AUTHORITY",
        }
    supported = [hypothesis_id for hypothesis_id, unit in units.items()
                 if unit["state"] == "VALIDATION_BACKTEST_SUPPORTED"]
    state = "VALIDATION_BACKTEST_COMPLETE" if units else "NO_PARAMETER_LOCK"
    artifact = create_artifact("validation_backtest_set", {
        "state": state, "dates": list(schedule.validation_dates), "symbols": list(symbols),
        "units": units, "supported_hypothesis_ids": supported,
    }, parents={"parameter_search_set": search["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
