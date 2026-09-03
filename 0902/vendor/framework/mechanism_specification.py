"""Hypothesis의 수익 가설을 Grounding 전과 후에 나눠 보존한다.

앞 단계는 관측 수익 가설과 주문 상태 계약을 고정한다. 경제적 주장은 선택적인 해석으로
보존한다. Grounding 뒤 단계는 그 계약이 Catalog 표현식과 체결 뒤 예측으로 바뀌지
않았는지만 확인한다. 둘 다 새 signal, threshold, exit rule을 만들지 않는다.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from . import catalog, mechanism_graph as M, sample_condition as SC
from .config import sha256_json


SCHEMA_VERSION = "mechanism_specification.v3"
READY_FOR_GROUNDING = "READY_FOR_GROUNDING"
READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
MATCHED_CONTROL_DIAGNOSTIC_REQUIRED = "MATCHED_CONTROL_DIAGNOSTIC_REQUIRED"
MECHANISM_DIAGNOSTIC_REQUIRED = "MECHANISM_DIAGNOSTIC_REQUIRED"
INVALID_MECHANISM_SPECIFICATION = "INVALID_MECHANISM_SPECIFICATION"


def _evidence_ids(hypothesis: Mapping[str, Any]) -> set[str]:
    return {str(item.get("evidence_id")) for item in hypothesis.get("evidence_basis") or []
            if item.get("evidence_id")}


def _prediction(grounded: Mapping[str, Any], *, target_type: str | None = None,
                direction: str | None = None) -> Mapping[str, Any]:
    """Graph가 지정한 미래 예측 하나를 Grounding claim에서 찾는다."""
    claims = grounded.get("claims") or []
    target_types = ((target_type,) if target_type in M.PREDICTION_TARGETS
                    else M.PREDICTION_TARGETS)
    for candidate_type in target_types:
        targets = [claim for claim in claims
                   if claim.get("grounding_status") == "VALIDATION_TARGET"
                   and claim.get("validation_target") == candidate_type
                   and claim.get("direction") in M.DIRECTIONS
                   and (direction is None or claim.get("direction") == direction)]
        core = [claim for claim in targets if claim.get("importance") == "CORE"]
        if len(core) == 1:
            return core[0]
        if len(core) > 1:
            raise ValueError(f"CORE {candidate_type} 예측이 하나여야 한다")
        if len(targets) == 1:
            return targets[0]
        if len(targets) > 1:
            raise ValueError(f"{candidate_type} 예측이 하나여야 한다")
    raise ValueError("PATH_TARGET 또는 MATCHED_CONTROL_TARGET 예측이 없다")


def _mechanism_blockers(grounded: Mapping[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for claim in grounded.get("claims") or []:
        if (claim.get("claim_type") != "MECHANISM_INTERPRETATION"
                or claim.get("importance") != "CORE"):
            continue
        status = str(claim.get("grounding_status") or "")
        if status != "GROUNDED_DERIVED":
            blockers.append({
                "claim_id": str(claim.get("claim_id") or ""),
                "grounding_status": status,
                "reason": str(claim.get("notes") or claim.get("claim_text") or ""),
            })
    return blockers


def _falsification(grounded: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{
        "claim_id": str(claim.get("claim_id") or ""),
        "target_type": str(claim.get("validation_target") or ""),
        "measure": claim.get("validation_measure"),
        "reference": claim.get("validation_reference"),
    } for claim in grounded.get("claims") or []
            if claim.get("grounding_status") == "VALIDATION_TARGET"
            and claim.get("validation_target") == "ALREADY_PRICED_CHECK"]


def _expression_features(expression: Mapping[str, Any]) -> set[str]:
    features: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("op") == "primitive" and value.get("primitive_id") in catalog.FEATURES:
                features.add(str(value["primitive_id"]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(expression)
    return features


def build(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """Grounding 전에 경제적 경로와 고정된 주문 상태 계약을 만든다."""
    sample_condition = SC.optional(hypothesis, "Hypothesis")
    graph = hypothesis.get("mechanism_graph")
    report = M.validate(graph, known_evidence_ids=_evidence_ids(hypothesis),
                        require_bindings=False)
    if report["problems"]:
        raise ValueError("MechanismGraph가 Hypothesis와 맞지 않는다: "
                         + "; ".join(report["problems"]))
    graph = dict(graph)
    execution = dict(graph["execution"])
    prediction = dict(graph["prediction"])
    value: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
        "state": READY_FOR_GROUNDING,
        "reason": "수익 가설·BID1 대기·체결 뒤 예측을 Grounding 전에 고정했다",
        "economic_mechanism": {
            "mechanistic_claim": hypothesis.get("mechanistic_claim"),
            "source_of_profit": hypothesis.get("source_of_profit"),
            "expected_observable_sequence": copy.deepcopy(
                hypothesis.get("expected_observable_sequence")),
            "already_priced_risk": hypothesis.get("already_priced_risk"),
            "alternative_explanations": copy.deepcopy(
                hypothesis.get("alternative_explanations") or []),
            "mechanism_tests": copy.deepcopy(hypothesis.get("mechanism_tests") or []),
            "implementation_directions": copy.deepcopy(
                hypothesis.get("implementation_directions") or []),
        },
        "grounding_contract": {
            "evidence_ids": sorted(_evidence_ids(hypothesis)),
            "grounding_requirements": copy.deepcopy(hypothesis.get("grounding_requirements") or []),
            "mechanism_graph": copy.deepcopy(graph),
        },
        "decision_to_fill": {
            "decision_state": execution.get("entry_state"),
            "pending_state": execution.get("pending_state"),
            "entry_action": execution.get("entry_action"),
            "entry_lifecycle_policy": copy.deepcopy(execution.get("entry_lifecycle_policy") or {}),
        },
        "post_fill_prediction": copy.deepcopy(prediction),
        "provenance": {
            "hypothesis_hash": sha256_json(hypothesis),
            "mechanism_graph_semantic_hash": report["semantic_hash"],
        },
    }
    if sample_condition is not None:
        value["executable_sample_condition"] = sample_condition
        value["executable_sample_condition_sha256"] = SC.condition_hash(sample_condition)
    value["mechanism_specification_sha256"] = sha256_json(value)[:16]
    return value


def assess_grounding(mechanism: Mapping[str, Any], hypothesis: Mapping[str, Any],
                     grounded: Mapping[str, Any]) -> dict[str, Any]:
    """Grounding이 앞에서 고정한 메커니즘 계약을 그대로 옮겼는지 확인한다."""
    if mechanism.get("state") != READY_FOR_GROUNDING:
        raise ValueError("Grounding 전 mechanism specification이 준비 상태가 아니다")
    if mechanism.get("hypothesis_id") != hypothesis.get("hypothesis_id"):
        raise ValueError("Mechanism specification의 hypothesis_id가 다르다")
    if mechanism.get("provenance", {}).get("hypothesis_hash") != sha256_json(hypothesis):
        raise ValueError("Mechanism specification의 Hypothesis 원문이 다르다")
    sample_condition = SC.optional(hypothesis, "Hypothesis")
    if mechanism.get("executable_sample_condition") != sample_condition:
        raise ValueError("Mechanism specification의 executable_sample_condition이 다르다")
    source_graph = (mechanism.get("grounding_contract") or {}).get("mechanism_graph")
    report = M.validate_preservation(source_graph, grounded.get("mechanism_graph"),
                                     known_evidence_ids=_evidence_ids(hypothesis),
                                     known_features=catalog.FEATURES)
    if report["problems"]:
        raise ValueError("MechanismGraph 보존 실패: " + "; ".join(report["problems"]))
    if report["source_semantic_hash"] != mechanism.get("provenance", {}).get("mechanism_graph_semantic_hash"):
        raise ValueError("Grounding 전 mechanism graph hash가 다르다")

    expression = grounded.get("operational_expression")
    if not isinstance(expression, Mapping):
        raise ValueError("Grounding에 operational_expression가 없다")
    expression_features = _expression_features(expression)
    graph = grounded["mechanism_graph"]
    binding_features = {str(item.get("catalog_feature"))
                        for item in graph.get("catalog_bindings") or []}
    if expression_features != binding_features:
        raise ValueError("operational_expression와 MechanismGraph binding feature가 다르다")
    prediction = _prediction(grounded, target_type=str(graph["prediction"].get("target_type") or ""),
                             direction=str(graph["prediction"].get("direction") or "") or None)
    if (prediction.get("validation_target") != graph["prediction"].get("target_type")
            or prediction.get("direction") != graph["prediction"].get("direction")):
        raise ValueError("MechanismGraph prediction과 Grounding prediction이 다르다")

    blockers = _mechanism_blockers(grounded)
    prediction_id = str(prediction.get("claim_id") or "")
    missing = sorted({str(item) for claim in grounded.get("claims") or []
                      if str(claim.get("claim_id") or "") == prediction_id
                      for item in claim.get("missing_capability") or []})
    target_type = str(prediction.get("validation_target"))
    required_diagnostics: list[dict[str, Any]] = []
    if target_type == "MATCHED_CONTROL_TARGET":
        required_diagnostics.append({
            "type": "MATCHED_CONTROL",
            "reason": "같은 구간의 실제 체결 matched control과 비교한다",
        })
    if blockers:
        required_diagnostics.append({
            "type": "ECONOMIC_INTERPRETATION",
            "reason": "경제적 해석은 아직 관측으로 식별되지 않았다",
            "claims": blockers,
        })
    if missing:
        required_diagnostics.append({
            "type": "MEASUREMENT_CAPABILITY",
            "reason": "체결 뒤 예측 측정에 필요한 capability가 아직 없다",
            "missing_capability": missing,
        })
    execution_state = READY_FOR_EXECUTION
    execution_reason = ("관측식·BID1 대기·체결 뒤 예측이 보존됐다. "
                        "진단은 실행 뒤 수익성 평가에서 수행한다")

    value: dict[str, Any] = {
        "schema": "grounding_mechanism_fidelity.v1",
        "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
        "state": "GROUNDING_FIDELITY_PRESERVED",
        "execution_state": execution_state,
        "execution_reason": execution_reason,
        "grounded_contract": {
            "observation": {
                "operational_expression": copy.deepcopy(dict(expression)),
                "operational_claim_ids": [str(item) for item in
                                            grounded.get("operational_claim_ids") or []],
                "executable_sample_condition": copy.deepcopy(sample_condition),
            },
            "decision_to_fill": copy.deepcopy(mechanism.get("decision_to_fill") or {}),
            "post_fill_prediction": {
                "claim_id": prediction_id,
                "target_type": target_type,
                "direction": str(prediction.get("direction") or ""),
                "measure": prediction.get("validation_measure"),
                "reference": prediction.get("validation_reference"),
                "required_data": list(prediction.get("required_data") or []),
                "missing_capability": missing,
            },
            "mechanism_blockers": blockers,
            "required_diagnostics": required_diagnostics,
            "falsification": _falsification(grounded),
        },
        "provenance": {
            "mechanism_specification_sha256": mechanism.get("mechanism_specification_sha256"),
            "hypothesis_hash": sha256_json(hypothesis),
            "grounding_hash": sha256_json(grounded),
            "mechanism_graph_semantic_hash": report["source_semantic_hash"],
            "grounded_mechanism_graph_hash": report["grounded_full_hash"],
        },
    }
    value["grounding_mechanism_fidelity_sha256"] = sha256_json(value)[:16]
    return value


def audit_execution_specification(fidelity: Mapping[str, Any],
                                  specification: Mapping[str, Any]) -> dict[str, Any]:
    """실행 명세가 Grounding 뒤 계약을 그대로 옮겼는지 확인한다."""
    invariants = specification.get("semantic_invariants") or {}
    contract = fidelity.get("grounded_contract") or {}
    post_fill = contract.get("post_fill_prediction") or {}
    decision_to_fill = contract.get("decision_to_fill") or {}
    expected = {
        "operational_expression": (contract.get("observation") or {}).get("operational_expression"),
        "executable_sample_condition": (
            contract.get("observation") or {}).get("executable_sample_condition"),
        "entry_lifecycle_policy": decision_to_fill.get("entry_lifecycle_policy"),
        "prediction_target": post_fill.get("target_type"),
        "prediction_direction": post_fill.get("direction"),
        "prediction_measure": post_fill.get("measure"),
        "prediction_reference": post_fill.get("reference"),
    }
    problems = [f"{key}가 Grounding 메커니즘 계약과 다르다"
                for key, value in expected.items() if invariants.get(key) != value]
    return {
        "state": "FIDELITY_PRESERVED" if not problems else "FIDELITY_VIOLATION",
        "problems": problems,
        "grounding_mechanism_fidelity_sha256": fidelity.get("grounding_mechanism_fidelity_sha256"),
        "specification_sha256": specification.get("specification_sha256"),
    }
