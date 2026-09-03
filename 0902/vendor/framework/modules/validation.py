"""ValidationPlan을 기존 검증 계산기에 연결하는 시스템용 어댑터."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import catalog, validation as V
from ..config import write_json
from .schedule import ResearchSchedule


_RUNTIME_TARGET = {
    "PATH_TARGET": "PATH_TARGET",
    "DOSE_RESPONSE_TARGET": "PATH_TARGET",
    "CONDITIONAL_EFFECT_TARGET": "INCREMENTAL_EFFECT_TARGET",
    "MATCHED_CONTROL_TARGET": "INCREMENTAL_EFFECT_TARGET",
    "INCREMENTAL_EFFECT_TARGET": "INCREMENTAL_EFFECT_TARGET",
    "ALREADY_PRICED_CHECK": "ALREADY_PRICED_CHECK",
    "TEMPORAL_PRECEDENCE_TARGET": "NOT_TESTABLE",
}


def _with_statistic(contract: Mapping[str, Any], target_type: str) -> dict[str, Any]:
    item = dict(contract)
    statistic = V.STATISTIC_KINDS[target_type]
    item.update({
        "target_type": target_type,
        "claim_asks": statistic["claim_asks"],
        "statistic_measures": statistic["statistic_measures"],
        "statistic": statistic["statistic"],
        "null": statistic["null"],
        "structural_terms_controlled": statistic["structural_terms_controlled"],
        "structural_note": statistic["structural_note"],
    })
    return item


def _runtime_contracts(grounded: Mapping[str, Any], hypothesis: Mapping[str, Any],
                       planned_targets: Sequence[Mapping[str, Any]],
                       config: V.ValidationConfig) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Plan의 target type을 보존해 기존 통계 중 하나에 명시적으로 연결한다."""
    source = V.build_targets({"hypotheses": [grounded]}, config,
                             str(hypothesis["hypothesis_id"]))
    planned_by_claim = {
        str(item["claim_id"]): item for item in planned_targets
        if item.get("claim_id") and item.get("target_type")
    }
    contracts: dict[str, dict[str, Any]] = {}
    executed_claims: set[str] = set()
    deferred: list[dict[str, Any]] = []
    deferred_ids: set[str] = set()
    for name, item in source.items():
        claim_id = str(item.get("claim_id") or "")
        planned = planned_by_claim.get(claim_id)
        if planned is None:
            continue
        runtime_type = _RUNTIME_TARGET.get(str(planned["target_type"]))
        if runtime_type is None:
            raise ValueError(f"지원하지 않는 ValidationPlan target: {planned['target_type']}")
        if runtime_type == "NOT_TESTABLE":
            deferred.append(dict(planned, status="NOT_EXECUTED_RUNTIME_UNAVAILABLE"))
            deferred_ids.add(str(planned.get("target_id")))
            continue
        contract = _with_statistic(item, runtime_type)
        contract["plan_target_id"] = planned.get("target_id")
        contract["plan_target_type"] = planned["target_type"]
        contracts[name] = contract
        executed_claims.add(claim_id)
    deferred += [dict(item, status="NOT_EXECUTED_NO_GROUNDED_CLAIM")
                 for item in planned_targets
                 if not item.get("claim_id") or str(item.get("claim_id")) not in executed_claims
                 if str(item.get("target_id")) not in deferred_ids]
    if not contracts:
        raise ValueError("ValidationPlan에 실행 가능한 grounded claim target이 없다")
    return contracts, deferred


def _config(hypothesis: Mapping[str, Any], package: Mapping[str, Any],
            discovery_dates: Sequence[str], schedule: ResearchSchedule,
            grounded: Mapping[str, Any]) -> V.ValidationConfig:
    source = dict(hypothesis)
    source.setdefault("evidence_roles", package.get("evidence_roles") or {})
    inferred = V.config_for(source)
    extra = tuple(sorted({str(claim.get("catalog_feature"))
                          for claim in grounded.get("claims") or []
                          if str(claim.get("catalog_feature")) in catalog.FEATURES}))
    return replace(inferred, discovery_dates=tuple(discovery_dates),
                   holdout_dates=tuple(schedule.validation_dates), extra_features=extra)


def run(*, plan_artifact_id: str, plan: Mapping[str, Any], hypotheses: Mapping[str, Any],
        grounded: Mapping[str, Any], package: Mapping[str, Any], symbols: Sequence[str],
        discovery_dates: Sequence[str], schedule: ResearchSchedule, output: Path,
        root: Path) -> dict[str, Any]:
    """가설별로 ValidationPlan을 실행한다. 첫 가설만 골라 쓰지 않는다."""
    schedule.validate(tuple(discovery_dates))
    source_hypotheses = {str(item.get("hypothesis_id")): item
                         for item in hypotheses.get("hypotheses") or []}
    source_grounded = {str(item.get("hypothesis_id")): item
                       for item in grounded.get("hypotheses") or []}
    targets_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
    for target in plan.get("targets") or []:
        targets_by_hypothesis.setdefault(str(target.get("hypothesis_id")), []).append(dict(target))
    output = Path(output)
    units: dict[str, Any] = {}
    for hypothesis_id in plan.get("hypothesis_ids") or []:
        hypothesis = source_hypotheses.get(str(hypothesis_id))
        grounded_item = source_grounded.get(str(hypothesis_id))
        if hypothesis is None or grounded_item is None:
            raise ValueError(f"ValidationPlan의 hypothesis_id를 입력 artifact에서 찾지 못했다: {hypothesis_id}")
        unit = output / "units" / str(hypothesis_id)
        inputs = unit / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        write_json(inputs / "hypotheses.json", {"status": hypotheses.get("status"), "hypotheses": [hypothesis]})
        write_json(inputs / "grounding.json", {"hypotheses": [grounded_item]})
        write_json(inputs / "evidence.json", package)
        config = _config(hypothesis, package, discovery_dates, schedule, grounded_item)
        contracts, deferred = _runtime_contracts(grounded_item, hypothesis,
                                                  targets_by_hypothesis.get(str(hypothesis_id), []),
                                                  config)
        ledger = {"consumed_dates": list(discovery_dates), "consumed_days": list(discovery_dates),
                  "remaining_fresh_days": list(schedule.validation_dates),
                  "terminal_oos_dates": list(schedule.terminal_oos_dates)}
        frozen = V.freeze(inputs / "hypotheses.json", inputs / "grounding.json",
                          inputs / "evidence.json", unit, config=config, ledger=ledger,
                          target_contracts=contracts,
                          validation_plan_artifact_id=plan_artifact_id)
        result = V.run(unit, config=config, symbols=list(symbols), root=root)
        units[str(hypothesis_id)] = {
            "state": "VALIDATED", "directory": f"units/{hypothesis_id}",
            "config": asdict(config), "frozen_plan_hash": frozen["validation_plan_hash"],
            "routing": result["routing"], "runtime_targets": sorted(contracts),
            "deferred_targets": deferred,
        }
    return {"state": "VALIDATION_COMPLETE", "schedule": schedule.as_input(),
            "discovery_dates": list(discovery_dates), "symbols": list(symbols), "units": units}
