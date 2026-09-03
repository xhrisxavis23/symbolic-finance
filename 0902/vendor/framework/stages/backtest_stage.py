"""Supported Validation Backtest → Final Backtest set artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import canonical as K, search as S
from ..config import TICK_ROOT
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import backtest
from ..modules.schedule import ResearchSchedule


ARTIFACT_NAME = "backtest_artifact.json"


def _v9_validation_supported(unit: dict[str, Any]) -> bool:
    return (unit.get("state") == "VALIDATION_BACKTEST_SUPPORTED"
            and unit.get("v9_validation_state") == "VALIDATION_BACKTEST_SUPPORTED")


def _final_profit_target_state(summary: dict[str, Any]) -> str:
    if summary.get("state") != "BACKTEST_COMPLETE":
        return "FINAL_PROFIT_TARGET_NOT_EVALUATED"
    evaluation = summary.get("profit_target_evaluation")
    if not isinstance(evaluation, dict):
        return "FINAL_PROFIT_TARGET_INCONCLUSIVE"
    if evaluation.get("accounting_identity_ok") is not True:
        return "FINAL_PROFIT_TARGET_ACCOUNTING_FAILURE"
    return ("FINAL_PROFIT_TARGET_SUPPORTED" if evaluation.get("success") is True
            else "FINAL_PROFIT_TARGET_NOT_SUPPORTED")


def _final_state_fields(units: dict[str, Any]) -> dict[str, Any]:
    states = {
        hypothesis_id: str(unit["final_profit_target_state"])
        for hypothesis_id, unit in units.items()
        if unit.get("state") == "BACKTEST_COMPLETE"
    }
    return {
        "final_profit_target_states": states,
        "final_profit_target_supported_hypothesis_ids": [
            hypothesis_id for hypothesis_id, state in states.items()
            if state == "FINAL_PROFIT_TARGET_SUPPORTED"
        ],
    }


def run_from_validation_refinement(parameter_search_path: Path, evidence_path: Path,
                                   output: Path, *, schedule: ResearchSchedule,
                                   root: Path = TICK_ROOT) -> dict[str, Any]:
    """Validation을 개발 입력으로 쓴 새 q lock을 Final 날짜에서 처음 평가한다."""
    search = read_artifact(parameter_search_path, kind="parameter_search_set")
    evidence = read_artifact(evidence_path, kind="evidence_package")
    if search["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Parameter Search와 Evidence의 lineage가 다르다")
    discovery_dates = tuple(evidence["payload"].get("profile_dates") or [])
    schedule.validate(discovery_dates)
    symbols = tuple(evidence["payload"].get("profile_symbols") or [])
    output = Path(output)
    search_units = search["payload"].get("units", {})
    runnable = {
        str(hypothesis_id): result for hypothesis_id, result in search_units.items()
        if (result.get("lock") or {}).get("status") in S.PARAMETER_LOCK_STATUSES
    }
    batched = len(runnable) > 1
    primary: dict[str, dict[str, Any]] = (
        backtest.run_batch(runnable, symbols=symbols, dates=schedule.backtest_dates,
                           output=output, root=Path(root), stage="FINAL_BACKTEST")
        if batched else {})
    completed = {
        hypothesis_id: runnable[hypothesis_id] for hypothesis_id, summary in primary.items()
        if summary.get("state") == "BACKTEST_COMPLETE" and hypothesis_id in runnable
    }
    comparison: dict[str, dict[str, Any]] = {}
    if batched and len(completed) > 1:
        comparison = backtest.run_batch(
            completed, symbols=symbols, dates=schedule.backtest_dates, output=output,
            root=Path(root), stage="FINAL_BACKTEST_COMPARISON",
            canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
    elif batched and completed:
        hypothesis_id, search_result = next(iter(completed.items()))
        comparison[hypothesis_id] = backtest.run(
            search_result, symbols=symbols, dates=schedule.backtest_dates,
            output=output / "units" / hypothesis_id / "v8_900_comparison", root=Path(root),
            stage="FINAL_BACKTEST_COMPARISON", canonical=K.V8_900_COMPARISON,
            comparison_of=K.CANONICAL)
    units: dict[str, Any] = {}
    for hypothesis_id, search_result in search_units.items():
        hypothesis_id = str(hypothesis_id)
        if batched:
            summary = primary.get(hypothesis_id)
            if summary is None:
                summary = backtest.run(
                    search_result, symbols=symbols, dates=schedule.backtest_dates,
                    output=output / "units" / hypothesis_id, root=Path(root),
                    stage="FINAL_BACKTEST")
            comparison_result = comparison.get(hypothesis_id)
        else:
            summary = backtest.run(
                search_result, symbols=symbols, dates=schedule.backtest_dates,
                output=output / "units" / hypothesis_id, root=Path(root),
                stage="FINAL_BACKTEST")
            comparison_result = None
            if summary.get("state") == "BACKTEST_COMPLETE":
                comparison_result = backtest.run(
                    search_result, symbols=symbols, dates=schedule.backtest_dates,
                    output=output / "units" / hypothesis_id / "v8_900_comparison",
                    root=Path(root), stage="FINAL_BACKTEST_COMPARISON",
                    canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
        units[hypothesis_id] = {
            **summary,
            "execution_state": summary.get("state"),
            "final_profit_target_state": _final_profit_target_state(summary),
            "final_backtest_opened": True,
            **({"v8_900_comparison_backtest": comparison_result}
               if comparison_result is not None else {}),
            "final_backtest_policy": "VALIDATION_REFINEMENT_PARAMETER_LOCKS",
            "validation_state": "VALIDATION_USED_FOR_DISCOVERY_REFINEMENT",
        }
    executed = [hypothesis_id for hypothesis_id, unit in units.items()
                if unit.get("state") == "BACKTEST_COMPLETE"]
    state = "BACKTEST_COMPLETE" if executed else "NO_PARAMETER_LOCK"
    artifact = create_artifact("backtest_set", {
        "state": state,
        "execution_state": state,
        "units": units,
        "executed_hypothesis_ids": executed,
        **_final_state_fields(units),
        "input_boundary": (
            "이 후보는 Validation Backtest를 개발 입력으로 이미 읽었다. "
            "Validation을 promotion 관문으로 다시 쓰지 않고 Final 날짜에서 처음 평가한다."),
    }, parents={"parameter_search_set": search["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact


def run(parameter_search_path: Path, validation_path: Path, evidence_path: Path, output: Path, *,
        schedule: ResearchSchedule, root: Path = TICK_ROOT,
        final_backtest_for_all_validation: bool = False) -> dict[str, Any]:
    """기본은 Validation 지지 lock만, 요청하면 모든 lock을 Final Backtest로 보낸다."""
    search = read_artifact(parameter_search_path, kind="parameter_search_set")
    validation = read_artifact(validation_path, kind="validation_backtest_set")
    evidence = read_artifact(evidence_path, kind="evidence_package")
    if validation["parents"].get("parameter_search_set") != search["artifact_id"]:
        raise ValueError("Validation Backtest와 Parameter Search의 lineage가 다르다")
    if validation["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Validation Backtest와 Evidence의 lineage가 다르다")
    discovery_dates = tuple(evidence["payload"].get("profile_dates") or [])
    schedule.validate(discovery_dates)
    symbols = tuple(evidence["payload"].get("profile_symbols") or [])
    output = Path(output)
    validation_units = validation["payload"].get("units", {})
    search_units = search["payload"].get("units", {})
    runnable = {
        str(hypothesis_id): search_result
        for hypothesis_id, search_result in search_units.items()
        if (final_backtest_for_all_validation or _v9_validation_supported(
                validation_units.get(str(hypothesis_id)) or {}))
        and (search_result.get("lock") or {}).get("status") in S.PARAMETER_LOCK_STATUSES
    }
    batched = len(runnable) > 1
    primary: dict[str, dict[str, Any]] = (
        backtest.run_batch(runnable, symbols=symbols, dates=schedule.backtest_dates,
                           output=output, root=Path(root), stage="FINAL_BACKTEST")
        if batched else {})
    completed = {
        hypothesis_id: runnable[hypothesis_id] for hypothesis_id, summary in primary.items()
        if summary.get("state") == "BACKTEST_COMPLETE" and hypothesis_id in runnable
    }
    comparison: dict[str, dict[str, Any]] = {}
    if batched and len(completed) > 1:
        comparison = backtest.run_batch(
            completed, symbols=symbols, dates=schedule.backtest_dates, output=output,
            root=Path(root), stage="FINAL_BACKTEST_COMPARISON",
            canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
    elif batched and completed:
        hypothesis_id, search_result = next(iter(completed.items()))
        comparison[hypothesis_id] = backtest.run(
            search_result, symbols=symbols, dates=schedule.backtest_dates,
            output=output / "units" / hypothesis_id / "v8_900_comparison", root=Path(root),
            stage="FINAL_BACKTEST_COMPARISON", canonical=K.V8_900_COMPARISON,
            comparison_of=K.CANONICAL)
    units: dict[str, Any] = {}
    for hypothesis_id, search_result in search_units.items():
        hypothesis_id = str(hypothesis_id)
        validation_unit = validation_units.get(hypothesis_id) or {}
        validation_state = validation_unit.get("state", "VALIDATION_BACKTEST_MISSING")
        if (not _v9_validation_supported(validation_unit)
                and not final_backtest_for_all_validation):
            units[hypothesis_id] = {
                "state": validation_state, "final_backtest_opened": False,
                "reason": "정본 V9 Validation Backtest가 실행 손익으로 지지하지 않았다",
            }
            continue
        if batched:
            summary = primary.get(hypothesis_id)
            if summary is None:
                summary = backtest.run(
                    search_result, symbols=symbols, dates=schedule.backtest_dates,
                    output=output / "units" / hypothesis_id, root=Path(root),
                    stage="FINAL_BACKTEST")
            comparison_result = comparison.get(hypothesis_id)
        else:
            summary = backtest.run(
                search_result, symbols=symbols, dates=schedule.backtest_dates,
                output=output / "units" / hypothesis_id, root=Path(root),
                stage="FINAL_BACKTEST")
            comparison_result = None
            if summary.get("state") == "BACKTEST_COMPLETE":
                comparison_result = backtest.run(
                    search_result, symbols=symbols, dates=schedule.backtest_dates,
                    output=output / "units" / hypothesis_id / "v8_900_comparison",
                    root=Path(root), stage="FINAL_BACKTEST_COMPARISON",
                    canonical=K.V8_900_COMPARISON, comparison_of=K.CANONICAL)
        units[hypothesis_id] = {
            **summary,
            "execution_state": summary.get("state"),
            "final_profit_target_state": _final_profit_target_state(summary),
            "final_backtest_opened": True,
            **({"v8_900_comparison_backtest": comparison_result}
               if comparison_result is not None else {}),
            "final_backtest_policy": (
                "ALL_PARAMETER_LOCKS" if final_backtest_for_all_validation
                else "V9_VALIDATION_BACKTEST_SUPPORTED_ONLY"),
            "validation_state": validation_state,
            "validation_promotion_profile_ids": list(
                validation_unit.get("promotion_profile_ids") or []),
        }
    executed = [hypothesis_id for hypothesis_id, unit in units.items()
                if unit.get("state") == "BACKTEST_COMPLETE"]
    state = "BACKTEST_COMPLETE" if executed else "NO_VALIDATION_BACKTEST_SUPPORT"
    artifact = create_artifact("backtest_set", {
        "state": state,
        "execution_state": state,
        "units": units,
        "executed_hypothesis_ids": executed,
        **_final_state_fields(units),
    },
                               parents={"parameter_search_set": search["artifact_id"],
                                        "validation_backtest_set": validation["artifact_id"],
                                        "evidence_package": evidence["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
