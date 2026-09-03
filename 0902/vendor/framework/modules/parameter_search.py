"""Specification의 단일 q 축을 실제 V9 실행으로 탐색·확인·잠금한다.

공동 상태는 feature별 실제 임계를 가지지만 모든 상태가 하나의 q를 공유한다.
q 후보는 실제 BID/ASK·queue 원장에서 정본 profit_target을 통과해야 한다.
통과 후보 중 총 Net이 가장 큰 q를 고른다.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .. import canonical as K, catalog, contract, implementation as I, profit_target as PT, search as S
from ..config import now_utc, sha256_json, write_json
from . import backtest
from .schedule import ResearchSchedule


SELECTION_MODE = "FIXED_EXIT_EXECUTION_PROFIT_TARGET_V2"
PRIMARY_METRIC = PT.SEARCH_PRIMARY_METRIC
SELECTION_RULE = {
    "step_1": "Search Fit의 모든 q 후보를 같은 CANONICAL_QUEUE_V9 원장으로 실행한다",
    "step_2": "회계가 일치하고 mean Gross > 23bp이며 total Net > 0인 q만 남긴다",
    "step_3": "남은 후보 중 total_net_bps가 가장 큰 q를 고른다",
    "tie_break_1": "동률이면 net_bps_per_decision이 큰 q를 고른다",
    "tie_break_2": "그래도 동률이면 mean_gross_bps_per_fill이 큰 q를 고른다",
    "tie_break_3": "그래도 동률이면 체결이 많고 더 낮은 q를 고른다",
    "no_execution_result": "체결 후보가 없거나 profit_target 통과 후보가 없으면 q를 잠그지 않는다",
}


def candidate_summary(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """다음 Stage와 연구 장부가 바로 읽을 수 있는 q별 실행 요약.

    원본 후보와 전체 원장은 output sidecar에 계속 둔다. 여기에는 q별 선택·고정 청산
    결과만 넣어, `parameter_search_artifact.json`만 읽어도 왜 lock이 없었는지 알 수
    있게 한다.
    """
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        fixed = candidate.get("fixed_exit") or {}
        measured = fixed.get("metrics") or {}
        row = {
            "q": candidate.get("q"),
            "state": candidate.get("status"),
            "fixed_exit_state": fixed.get("state"),
            "fixed_exit_metrics": {
                key: measured.get(key)
                for key in ("decisions", "scorable", "fills", "gross_bps_total",
                            "mean_gross_bps_per_fill", "total_net_bps", "bps_per_decision")
            },
            "profit_target_evaluation": fixed.get("profit_target_evaluation"),
            "ledger": fixed.get("ledger"),
        }
        if candidate.get("parameter_values") is not None:
            row["parameter_values"] = candidate["parameter_values"]
        if candidate.get("candidate_id") is not None:
            row["candidate_id"] = candidate["candidate_id"]
        if candidate.get("entry_guards") is not None:
            row["entry_guards"] = [dict(guard) for guard in candidate["entry_guards"]]
        if fixed.get("entry_lifecycle_diagnostic") is not None:
            row["entry_lifecycle_diagnostic"] = dict(fixed["entry_lifecycle_diagnostic"])
        if fixed.get("q_exposure_diagnostic") is not None:
            row["q_exposure_diagnostic"] = dict(fixed["q_exposure_diagnostic"])
        rows.append(row)
    return rows


def candidate_summary_from_output(output: Path) -> list[dict[str, Any]]:
    """재개 시 sidecar의 response curve로 같은 요약을 복원한다."""
    path = Path(output) / "response_curve.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        source = csv.DictReader(handle)
        rows = []
        for row in source:
            summary = {
                "q": _number(row.get("q")),
                "candidate_id": row.get("후보") or None,
                "parameter_values": row.get("상태별q") or None,
                "state": row.get("판정") or None,
                "fixed_exit_state": row.get("fixed_exit_판정") or None,
                "fixed_exit_metrics": {
                    "decisions": _number(row.get("실행결정") or row.get("신호anchor")),
                    "scorable": _number(row.get("실행채점가능") or row.get("신호anchor")),
                    "fills": _number(row.get("실행체결")),
                    "gross_bps_total": _number(row.get("fixed_exit_gross합계")),
                    "mean_gross_bps_per_fill": _number(row.get("fixed_exit_gross체결당")),
                    "total_net_bps": _number(row.get("fixed_exit_net합계")),
                    "bps_per_decision": _number(row.get("fixed_exit_net결정당")),
                },
                "ledger": None,
            }
            summary["profit_target_evaluation"] = _profit_target_evaluation({
                "metrics": summary["fixed_exit_metrics"],
            })
            if "q노출_체결수" in row:
                summary["q_exposure_diagnostic"] = {
                    "selection_input": False,
                    "scope": "FILLED_ORDER_IDENTITY",
                    "candidate_id": row.get("후보") or None,
                    "fills": _number(row.get("q노출_체결수")),
                    "previous_candidate_id": row.get("q노출_직전후보") or None,
                    "shared_fills_with_previous": _number(row.get("q노출_직전공통체결")),
                    "current_share_of_previous": None,
                    "previous_share_retained": _number(row.get("q노출_직전체결유지비율")),
                    "identical_filled_set_to_previous": _boolean(
                        row.get("q노출_직전집합동일")),
                }
            rows.append(summary)
    return rows


def _number(value: Any) -> float | int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _boolean(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _declared_states(spec: Mapping[str, Any], allowed: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Parameter 항목에 방향이 생략되면 Specification의 고정 방향을 쓴다."""
    fixed = (spec.get("search_boundary") or {}).get("fixed_semantics") or {}
    direction = str(fixed.get("direction") or "")
    return {
        (str(item.get("feature")), str(item.get("direction") or direction))
        for item in allowed
    }


def _entry_expression_variants(config: S.SearchConfig,
                               variants: Mapping[str, Mapping[str, Any] | None] | None,
                               default: Mapping[str, Any] | None
                               ) -> dict[str, dict[str, Any] | None]:
    expected = {config.candidate_id(q) for q in config.grid}
    if variants is None:
        return {candidate_id: copy.deepcopy(default) for candidate_id in sorted(expected)}
    if set(map(str, variants)) != expected:
        raise ValueError("q별 Entry Refinement 진입식 후보가 Search grid와 다르다")
    return {
        candidate_id: copy.deepcopy(variants[candidate_id])
        for candidate_id in sorted(expected)
    }


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _apply_entry_expression(template: Mapping[str, Any],
                            expression: Mapping[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(dict(template))
    if expression is None:
        return result
    inferred = catalog.infer_expression_type(expression, allow_unresolved=True)
    if inferred.value_type not in {"boolean", "event"}:
        raise ValueError("Refinement entry_expression은 boolean/event여야 한다")
    placeholders = {
        value.removeprefix(catalog.UNRESOLVED_PREFIX)
        for value in _walk(expression)
        if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX)
    }
    unknown = placeholders - set(result.get("parameter_interface") or {})
    if unknown:
        raise ValueError(f"실행값 없는 새 UNRESOLVED가 있다: {sorted(unknown)}")
    result["parameter_interface"] = {
        name: item for name, item in (result.get("parameter_interface") or {}).items()
        if name in placeholders
    }
    result["entry_program"] = {
        "signal": copy.deepcopy(dict(expression)),
        "warmup_ticks": _expression_warmup(
            expression, rolling_quantile=contract.uses_rolling_quantiles(result)),
    }
    return result


def _expression_warmup(expression: Mapping[str, Any], *, rolling_quantile: bool) -> int:
    warmup = 100 if rolling_quantile else 0
    for node in _walk(expression):
        if not isinstance(node, Mapping):
            continue
        op = str(node.get("op") or "")
        if str(node.get("time_basis") or "") == "tick":
            if op == "difference":
                warmup = max(warmup, int(node.get("lag") or 0))
            elif op in {"rolling_zscore", "rolling_sum", "rolling_mean", "rolling_std",
                        "rolling_min", "rolling_max", "persistence"}:
                warmup = max(warmup, int(node.get("window") or 0))
            elif op in {"sequence", "sequence_once"}:
                warmup = max(warmup, int(node.get("max_lag") or 0))
        if op == "crossover":
            warmup = max(warmup, 1)
    return warmup


def _template_for_candidate(template: Mapping[str, Any], *, candidate_id: str,
                            guards: Sequence[Mapping[str, Any]],
                            entry_expression: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """같은 q 후보가 공유 원장에서 충돌하지 않게 계약 ID만 분리한다."""
    result = _apply_entry_expression(template, entry_expression)
    result["hypothesis_id"] = f"{template['hypothesis_id']}__{candidate_id}"
    result["entry_guards"] = [dict(guard) for guard in guards]
    result["contract_sha256"] = contract.contract_hash(result)[:16]
    return result


def _candidate_input(template: Mapping[str, Any], amended: Mapping[str, Any],
                     plan: Mapping[str, Any], value: float) -> dict[str, Any]:
    """V9 batch runner가 받는 최소 lock 모양. 아직 최종 선택 lock은 아니다."""
    parameters = {parameter: float(value) for _feature, _direction, parameter
                  in S.config_states(plan)}
    return {
        "lock": {
            "status": S.PARAMETER_LOCKED,
            "parameter": plan["grid_parameter"][0],
            "value": float(value),
            "parameters": parameters,
            "plan_sha256": sha256_json(plan)[:16],
        },
        "parameterized_template": dict(template),
        "amended_specification": dict(amended),
    }


def _fixed_exit(result: Mapping[str, Any] | None, *, ledger_path: Path) -> dict[str, Any]:
    """V9 결과를 Search response curve의 작은 공통 모양으로 줄인다."""
    result = dict(result or {})
    metrics = dict(result.get("metrics") or {})
    fills = int(metrics.get("fills") or 0)
    gross_total = metrics.get("gross_bps_total")
    mean_gross = (float(gross_total) / fills
                  if gross_total is not None and fills > 0 else None)
    evaluation = PT.evaluate(metrics)
    return {
        "state": result.get("state"),
        "metrics": {
            "decisions": metrics.get("eligible_decisions"),
            "scorable": metrics.get("scorable"),
            "fills": metrics.get("fills"),
            "gross_bps_total": gross_total,
            "mean_gross_bps_per_fill": mean_gross,
            "total_net_bps": metrics.get("net_bps_total"),
            "bps_per_decision": metrics.get("net_bps_per_decision"),
            "bps_per_fill": metrics.get("net_bps_per_fill"),
            "fill_rate": metrics.get("fill_rate"),
        },
        "profit_target_evaluation": evaluation,
        "ledger": str(ledger_path),
        "backtest_summary": result,
    }


def _profit_target_evaluation(fixed: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = fixed.get("profit_target_evaluation")
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    metrics = fixed.get("metrics") or {}
    return PT.evaluate({
        "fills": metrics.get("fills"),
        "gross_bps_total": metrics.get("gross_bps_total"),
        "net_bps_total": metrics.get("total_net_bps"),
    })


def _execution_select(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """정본 profit_target을 통과한 q 중 총 Net이 가장 큰 하나를 고른다."""
    scored = []
    for candidate in candidates:
        metrics = ((candidate.get("fixed_exit") or {}).get("metrics") or {})
        total = metrics.get("total_net_bps")
        fills = metrics.get("fills")
        if candidate.get("status") != "EXECUTION_SCORABLE" or total is None:
            continue
        if not math.isfinite(float(total)) or int(fills or 0) <= 0:
            continue
        scored.append(candidate)
    if not scored:
        return {
            "schema": "execution_parameter_selection.v1",
            "created_at": now_utc(),
            "status": S.INSUFFICIENT_SUPPORT,
            "selected_q": None,
            "selected_value": None,
            "rule": SELECTION_RULE,
            "direct_evidence_status": "NO_FILLED_EXECUTION_CANDIDATE",
            "reason": "Search Fit에서 실제 체결된 q 후보가 없다",
            "considered": len(candidates),
            "runner_up": [],
        }
    eligible = [candidate for candidate in scored
                if _profit_target_evaluation(candidate["fixed_exit"])["success"]]
    if not eligible:
        return {
            "schema": "execution_parameter_selection.v1",
            "created_at": now_utc(),
            "status": S.INSUFFICIENT_SUPPORT,
            "selected_q": None,
            "selected_value": None,
            "primary_metric": PRIMARY_METRIC,
            "hurdle_bps": float(PT.canonical_target()["hurdle_bps"]),
            "rule": SELECTION_RULE,
            "direct_evidence_status": "NO_PROFIT_TARGET_CANDIDATE",
            "reason": "Search Fit에서 정본 profit_target을 통과한 q가 없다",
            "considered": len(candidates),
            "filled_candidates": len(scored),
            "runner_up": [],
        }
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -float(candidate["fixed_exit"]["metrics"]["total_net_bps"]),
            -float(candidate["fixed_exit"]["metrics"].get("bps_per_decision") or 0.0),
            -float(candidate["fixed_exit"]["metrics"]["mean_gross_bps_per_fill"]),
            -int(candidate["fixed_exit"]["metrics"].get("fills") or 0),
            float(candidate["q"]),
        ),
    )
    winner = ranked[0]
    metrics = winner["fixed_exit"]["metrics"]
    evaluation = _profit_target_evaluation(winner["fixed_exit"])
    return {
        "schema": "execution_parameter_selection.v1",
        "created_at": now_utc(),
        "status": S.SEARCH_SELECTED,
        "selected_q": float(winner["q"]),
        "selected_value": float(winner["value"]),
        "primary_metric": PRIMARY_METRIC,
        "mean_gross_bps_per_fill": metrics["mean_gross_bps_per_fill"],
        "hurdle_bps": float(PT.canonical_target()["hurdle_bps"]),
        "net_bps_total": metrics["total_net_bps"],
        "scorable": metrics.get("scorable"),
        "fills": metrics.get("fills"),
        "profit_target_evaluation": evaluation,
        "rule": SELECTION_RULE,
        "direct_evidence_status": "EXECUTION_PROFIT_TARGET_SELECTED",
        "reason": (f"정본 profit_target을 통과한 q {len(ranked)}개 중 "
                   "total_net_bps가 가장 크다"),
        "considered": len(candidates),
        "runner_up": [float(candidate["q"]) for candidate in ranked[1:]],
    }


def _response_curve(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        fixed = candidate.get("fixed_exit") or {}
        metrics = fixed.get("metrics") or {}
        evaluation = _profit_target_evaluation(fixed)
        rows.append({
            "q": candidate.get("q"),
            "후보": candidate.get("candidate_id"),
            "상태별q": candidate.get("parameter_values"),
            "판정": candidate.get("status"),
            "fixed_exit_판정": (candidate.get("fixed_exit") or {}).get("state"),
            "fixed_exit_gross합계": metrics.get("gross_bps_total"),
            "fixed_exit_gross체결당": metrics.get("mean_gross_bps_per_fill"),
            "fixed_exit_net합계": metrics.get("total_net_bps"),
            "fixed_exit_net결정당": metrics.get("bps_per_decision"),
            "실행결정": metrics.get("decisions"),
            "실행채점가능": metrics.get("scorable"),
            "실행체결": metrics.get("fills"),
            "profit_target_성공": evaluation["success"],
            "profit_target_회계일치": evaluation["accounting_identity_ok"],
            "profit_target_Gross조건": evaluation["eligibility_condition_passed"],
            "profit_target_Net조건": evaluation["primary_condition_passed"],
        })
    return rows


def run(spec: Mapping[str, Any], *, symbols: Sequence[str], schedule: ResearchSchedule,
        output: Path, root: Path,
        anchor_cache_root: Path | None = None,
        entry_guards: Sequence[Mapping[str, Any]] = (),
        entry_guard_variants: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        entry_expression: Mapping[str, Any] | None = None,
        entry_expression_variants: Mapping[str, Mapping[str, Any] | None] | None = None,
        entry_refinement: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target = PT.require(spec, "Parameter Search Specification")
    allowed = list((spec.get("search_boundary") or {}).get("allowed_parameters") or [])
    config = S.config_for(spec)
    expected = {(feature, direction) for feature, direction, _parameter in config.states}
    declared = _declared_states(spec, allowed)
    if not allowed or declared != expected:
        raise ValueError(f"Specification과 Parameter Search의 상태가 다르다: {sorted(declared)} vs {sorted(expected)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source_specification.json"
    write_json(source, spec)
    blocks = {
        S.HYPOTHESIS_DEVELOPMENT: [],
        S.SEARCH_FIT: list(schedule.search_fit_dates),
        S.SEARCH_CONFIRM: list(schedule.search_confirm_dates),
        S.TERMINAL_OOS: list(schedule.terminal_oos_dates),
    }
    variants = S._entry_guard_variants(config, entry_guard_variants, entry_guards)
    expression_variants = _entry_expression_variants(
        config, entry_expression_variants, entry_expression)
    amended = S.amend_specification(spec, config)
    template = I.compile_contract(amended)
    template["entry_guards"] = [dict(guard) for guard in entry_guards]
    template["contract_sha256"] = contract.contract_hash(template)[:16]
    validity = I.implementation_validity(template)
    fidelity = I.fidelity_check(amended, template)
    if not validity["implementation_valid"] or fidelity["fidelity_status"] != "FIDELITY_PRESERVED":
        raise RuntimeError("매개화를 바꾼 뒤 충실도가 깨졌다. 탐색을 시작하지 않는다: "
                           + "; ".join(validity["problems"] + fidelity["problems"]))

    plan = S.plan(amended, blocks, list(symbols), config)
    plan.update({
        "selection_mode": SELECTION_MODE,
        "primary_metric": PRIMARY_METRIC,
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "selection_rule": SELECTION_RULE,
        "execution_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "search_input": "ALL_TICKS_CANONICAL_QUEUE_V9",
        "anchor_cache_used_for_selection": False,
        "not_done_here": ["새 feature 탐색", "격자 세분", "terminal OOS 평가"],
        "entry_refinement_branches": {
            "mode": ("Q_SPECIFIC_DISCOVERY_ENTRY" if (
                entry_guard_variants is not None or entry_expression_variants is not None)
                     else "COMMON_DISCOVERY_ENTRY"),
            "candidates": [{"candidate_id": config.candidate_id(q),
                            "entry_guards": variants[config.candidate_id(q)],
                            "entry_expression": expression_variants[config.candidate_id(q)]}
                           for q in config.grid],
        },
    })
    write_json(output / "parameter_search_plan.json", plan)

    candidate_inputs: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for q in config.grid:
        candidate_id = config.candidate_id(q)
        candidate_template = _template_for_candidate(
            template, candidate_id=candidate_id, guards=variants[candidate_id],
            entry_expression=expression_variants[candidate_id])
        candidate_inputs[candidate_id] = _candidate_input(candidate_template, amended, plan, float(q))
        candidates.append({
            "candidate_id": candidate_id,
            "q": float(q),
            "value": float(q),
            "parameter_values": {parameter: float(q) for _feature, _direction, parameter
                                 in config.states},
            "threshold_source": S.threshold_source(config, float(q)),
            "entry_guards": [dict(guard) for guard in variants[candidate_id]],
            "entry_expression": copy.deepcopy(expression_variants[candidate_id]),
        })

    fit_output = output / "execution_search_fit"
    fit_results = backtest.run_batch(
        candidate_inputs, symbols=list(symbols), dates=list(schedule.search_fit_dates),
        output=fit_output, root=root, stage="PARAMETER_SEARCH_FIT")
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        fixed = _fixed_exit(fit_results.get(candidate_id),
                            ledger_path=fit_output / "units" / candidate_id / "ledger.parquet")
        candidate["fixed_exit"] = fixed
        metrics = fixed["metrics"]
        candidate["status"] = (
            "EXECUTION_SCORABLE" if fixed.get("state") == "BACKTEST_COMPLETE"
            and int(metrics.get("fills") or 0) > 0 else
            "EXECUTION_NO_FILL" if fixed.get("state") == "BACKTEST_COMPLETE" else
            str(fixed.get("state") or "EXECUTION_FAILED"))

    selection = _execution_select(candidates)
    selected_guards: list[dict[str, Any]] = []
    selected_expression: dict[str, Any] | None = None
    confirmation: dict[str, Any] | None = None
    lock: dict[str, Any] | None = None
    if selection["status"] == S.SEARCH_SELECTED:
        selected_q = float(selection["selected_value"])
        selected_candidate_id = config.candidate_id(selected_q)
        selected_guards = [dict(guard) for guard in variants[selected_candidate_id]]
        selected_expression = copy.deepcopy(expression_variants[selected_candidate_id])
        template = _apply_entry_expression(template, selected_expression)
        template["entry_guards"] = selected_guards
        template["contract_sha256"] = contract.contract_hash(template)[:16]
        if schedule.search_confirm_dates:
            confirm_input = _candidate_input(template, amended, plan, selected_q)
            confirm_output = output / "execution_search_confirm"
            confirm_result = backtest.run_batch(
                {"selected": confirm_input}, symbols=list(symbols),
                dates=list(schedule.search_confirm_dates), output=confirm_output,
                root=root, stage="PARAMETER_SEARCH_CONFIRM").get("selected")
            fixed = _fixed_exit(confirm_result,
                                ledger_path=confirm_output / "units" / "selected" / "ledger.parquet")
            confirmed = (fixed.get("state") == "BACKTEST_COMPLETE"
                         and _profit_target_evaluation(fixed)["success"])
            confirmation = {
                "schema": "execution_parameter_confirmation.v1", "created_at": now_utc(),
                "q": selected_q, "value": selected_q,
                "threshold_source": S.threshold_source(config, selected_q),
                "status": (S.PARAMETER_CONFIRMED if confirmed
                           else S.PARAMETER_CONFIRMATION_FAILED),
                "fixed_exit": fixed,
                "only_one_candidate_opened": True,
                "selection_input": False,
                "why": ("선택에 쓰지 않은 날짜에서도 정본 profit_target을 통과했다"
                        if confirmed else
                        "선택에 쓰지 않은 날짜에서 정본 profit_target을 통과하지 못했다"),
            }
            if confirmed:
                lock = S.parameter_lock(plan, selection, confirmation, amended)
        elif schedule.discovery_only_lock:
            confirmation = {
                "schema": "execution_parameter_confirmation.v1", "created_at": now_utc(),
                "q": selected_q, "value": selected_q,
                "threshold_source": S.threshold_source(config, selected_q),
                "status": "NOT_RUN_SINGLE_DAY_DISCOVERY",
                "only_one_candidate_opened": False,
                "selection_input": False,
                "why": "승인된 discovery-only schedule이라 별도 Search Confirm 날짜가 없다",
            }
            lock = S.parameter_lock(plan, selection, confirmation, amended,
                                    status=S.PARAMETER_LOCKED_SINGLE_DAY)

    calibration = {
        "schema": "execution_search_calibration_manifest.v1", "created_at": now_utc(),
        **plan["calibration"],
        "selection_input": "V9 full-tick ledger",
        "anchor_cache_used_for_selection": False,
        "candidate_eligible_symbol_days": {
            candidate["candidate_id"]: ((candidate.get("fixed_exit") or {}).get("backtest_summary") or
                                        {}).get("eligible_symbol_days")
            for candidate in candidates
        },
    }
    audit = {
        "schema": "execution_parameter_search_audit.v1", "created_at": now_utc(),
        "selection_mode": SELECTION_MODE,
        "primary_metric": PRIMARY_METRIC,
        "search_fit_dates": list(schedule.search_fit_dates),
        "search_confirm_dates": list(schedule.search_confirm_dates),
        "candidate_count": len(candidates),
        "candidate_states": {candidate["candidate_id"]: candidate["status"]
                             for candidate in candidates},
        "future_label_used_for_selection": False,
        "anchor_cache_used_for_selection": False,
        "execution_profile": plan["execution_profile"],
        "terminal_oos_access_count": 0,
        "ok": True,
    }
    status = (lock["status"] if lock else confirmation["status"] if confirmation else
              selection["status"])

    write_json(output / "calibration_manifest.json", calibration)
    write_json(output / "executable_specification_amended.json", amended)
    write_json(output / "contract_template.json", template)
    write_json(output / "selected_parameter.json", selection)
    write_json(output / "search_audit.json", audit)
    if confirmation is not None:
        write_json(output / "confirmation_result.json", confirmation)
    if lock is not None:
        write_json(output / "parameter_lock.json", lock)
    curve = _response_curve(candidates)
    pd.DataFrame(curve).to_csv(output / "response_curve.csv", index=False)
    pd.DataFrame(curve).to_csv(output / "candidate_results.csv", index=False)
    (output / "parameter_search_report.md").write_text(
        "# Parameter Search\n\n"
        "모든 q 후보를 CANONICAL_QUEUE_V9 full-tick 실행으로 평가했다. "
        "정본 `profit_target`을 통과한 후보 중 "
        "총 Net이 가장 큰 q를 선택했다.\n",
        encoding="utf-8")
    return {
        "state": status, "plan": plan, "selection": selection,
        "confirmation": confirmation, "lock": lock,
        "audit": audit, "amended_specification": amended,
        "parameterized_template": template,
        "candidate_summary": candidate_summary(candidates),
        "entry_guards": selected_guards,
        "entry_expression": selected_expression,
        "entry_guard_variants": ({
            str(candidate_id): [dict(guard) for guard in guards]
            for candidate_id, guards in entry_guard_variants.items()
        } if entry_guard_variants is not None else None),
        "entry_expression_variants": ({
            str(candidate_id): copy.deepcopy(expression)
            for candidate_id, expression in entry_expression_variants.items()
        } if entry_expression_variants is not None else None),
        "entry_refinement": dict(entry_refinement) if entry_refinement is not None else None,
    }
