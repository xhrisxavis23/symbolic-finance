"""가설의 주장과 데이터 관측값을 맞추는 Agent 역할."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from dataclasses import asdict
from typing import Any, Mapping

from .. import grounding as G, hypothesis as H
from ..config import sha256_json
from .runtime import AgentInvocation, Runner, invoke


def normalize_relation_claim_encodings(value: Mapping[str, Any], package: Mapping[str, Any]
                                        ) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Joint evidence를 state feature로 잘못 적은 enum만 결정적으로 바로잡는다.

    `JEV_*`는 한 feature가 아니라 family 사이의 관측 관계다. Agent가 내용·근거·상태를
    이미 정확히 적고 enum만 `OBSERVABLE_STATE`로 둔 경우에만 `TEMPORAL_RELATION`으로
    바꾼다. 새 주장이나 새 관측을 만들지 않는다.
    """
    normalized = deepcopy(dict(value))
    joint_ids = {str(item.get("joint_evidence_id")) for item in package.get("joint_evidence") or []
                 if item.get("joint_evidence_id")}
    changes: list[dict[str, str]] = []
    for hypothesis in normalized.get("hypotheses") or []:
        if not isinstance(hypothesis, Mapping):
            continue
        for claim in hypothesis.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            relation_basis = set(map(str, claim.get("supporting_evidence_ids") or [])) & joint_ids
            if (relation_basis and claim.get("claim_type") == "OBSERVABLE_STATE"
                    and claim.get("epistemic_status") == "SUPPORTED_RELATION"
                    and claim.get("grounding_status") == "GROUNDED_DERIVED"
                    and not claim.get("catalog_feature")):
                claim["claim_type"] = "TEMPORAL_RELATION"
                changes.append({
                    "hypothesis_id": str(hypothesis.get("hypothesis_id")),
                    "claim_id": str(claim.get("claim_id")),
                    "change": "OBSERVABLE_STATE -> TEMPORAL_RELATION for JEV relation",
                })
    return normalized, changes


def normalize_single_direction_variants(value: Mapping[str, Any], hypotheses: Mapping[str, Any]
                                        ) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """기존 canonical Grounding을 단일 D1 방향으로만 호환시킨다.

    한 식으로 두 방향을 지어내면 안 된다. 따라서 H가 D1 하나만 선언한 경우에만
    그 canonical 식을 명시 variant로 올린다. 둘 이상 방향은 Agent가 각각을 Grounding
    해야 하며, 누락되면 결정적 계약 검사가 막는다.
    """
    normalized = deepcopy(dict(value))
    directions_by_hypothesis = {
        str(item.get("hypothesis_id")): list(item.get("implementation_directions") or [])
        for item in hypotheses.get("hypotheses") or [] if isinstance(item, Mapping)
    }
    changes: list[dict[str, str]] = []
    for grounded in normalized.get("hypotheses") or []:
        if not isinstance(grounded, dict) or "implementation_variants" in grounded:
            continue
        directions = directions_by_hypothesis.get(str(grounded.get("hypothesis_id")), [])
        if len(directions) != 1 or not isinstance(directions[0], Mapping):
            continue
        direction_id = str(directions[0].get("direction_id") or "")
        expression = grounded.get("operational_expression")
        if not direction_id or not isinstance(expression, Mapping):
            continue
        grounded["implementation_variants"] = [{
            "direction_id": direction_id,
            "operational_expression": deepcopy(expression),
            "operational_claim_ids": list(grounded.get("operational_claim_ids") or []),
            "operational_note": grounded.get("operational_note"),
        }]
        changes.append({
            "hypothesis_id": str(grounded.get("hypothesis_id")),
            "direction_id": direction_id,
            "change": "canonical operational_expression -> single implementation variant",
        })
    return normalized, changes


def _validation_and_state(grounded: Mapping[str, Any], hypotheses: Mapping[str, Any],
                          package: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    validation = G.validate(grounded, hypotheses, package)
    source_ids = {str(item.get("hypothesis_id")) for item in hypotheses.get("hypotheses") or []}
    grounded_ids = {str(item.get("hypothesis_id")) for item in grounded.get("hypotheses") or []}
    missing, unexpected = sorted(source_ids - grounded_ids), sorted(grounded_ids - source_ids)
    if missing or unexpected:
        validation = dict(validation)
        validation["problems"] = list(validation["problems"])
        if missing:
            validation["problems"].append(f"Grounding 결과에서 빠진 hypothesis_id: {missing}")
        if unexpected:
            validation["problems"].append(f"Grounding 결과에 원래 없던 hypothesis_id: {unexpected}")
    validation["hypothesis_coverage"] = {
        "requested": sorted(source_ids), "grounded": sorted(grounded_ids),
        "missing": missing, "unexpected": unexpected,
    }
    state = ("INVALID_OUTPUT" if not source_ids or validation["problems"]
             else "READY_FOR_VALIDATION_PLAN")
    return validation, state


def run(hypotheses: Mapping[str, Any], package: Mapping[str, Any], *, runner: Runner | None = None,
        mechanism_specifications: Mapping[str, Any] | None = None,
        config: G.GroundingConfig = G.GroundingConfig()) -> dict[str, Any]:
    """Grounding은 뜻을 고정한다. Validation 실행이나 파라미터 선택은 하지 않는다."""
    audit = G.input_audit(hypotheses, package)
    invocations: list[AgentInvocation] = []
    if not audit["ok"]:
        grounded = {"hypotheses": [], "blocked_at": "PASS_0_INPUT_AUDIT"}
    else:
        payload = G.grounding_input(hypotheses, package,
                                    mechanism_specifications=mechanism_specifications)
        generation_prompt = G.grounding_prompt(payload)
        try:
            draft, generated = invoke(runner, role="grounding_generation",
                                      prompt=generation_prompt,
                                      model=config.model, effort=config.effort)
        except H.AgentCallError as error:
            return _failure("grounding_generation", generation_prompt, error,
                            audit, payload, config, invocations)
        invocations.append(generated)
        audit_prompt = G.audit_prompt(payload, draft)
        try:
            grounded, audited = invoke(runner, role="grounding_semantic_audit",
                                       prompt=audit_prompt,
                                       model=config.model, effort=config.effort)
        except H.AgentCallError as error:
            return _failure("grounding_semantic_audit", audit_prompt, error,
                            audit, payload, config, invocations)
        invocations.append(audited)
    grounded, normalizations = normalize_relation_claim_encodings(grounded, package)
    grounded, direction_normalizations = normalize_single_direction_variants(grounded, hypotheses)
    normalizations.extend(direction_normalizations)
    validation, state = _validation_and_state(grounded, hypotheses, package)
    if audit["ok"] and validation["problems"]:
        repair_prompt = (
            "Repair this grounding JSON so it satisfies the listed contract problems. "
            "Do not add a hypothesis, feature, threshold, or validation result. Return JSON only.\n\n"
            f"Problems:\n{validation['problems']}\n\nPayload:\n{grounded}")
        try:
            repaired, repaired_invocation = invoke(
                runner, role="grounding_contract_repair", prompt=repair_prompt,
                model=config.model, effort=config.effort)
        except H.AgentCallError as error:
            return _failure("grounding_contract_repair", repair_prompt, error,
                            audit, payload, config, invocations)
        grounded, repaired_normalizations = normalize_relation_claim_encodings(repaired, package)
        grounded, repaired_direction_normalizations = normalize_single_direction_variants(
            grounded, hypotheses)
        normalizations.extend(repaired_normalizations)
        normalizations.extend(repaired_direction_normalizations)
        invocations.append(repaired_invocation)
        validation, state = _validation_and_state(grounded, hypotheses, package)
    return {
        "payload": grounded,
        "input_audit": audit,
        "validation": validation,
        "agent_invocations": [item.as_dict() for item in invocations],
        "normalizations": normalizations,
        "config": asdict(config),
        "state": state,
    }


def revalidate(source_grounded: Mapping[str, Any], hypotheses: Mapping[str, Any],
               package: Mapping[str, Any], *, source_artifact_id: str) -> dict[str, Any]:
    """Agent 원문을 다시 부르지 않고 현재의 결정적 Grounding 계약만 재적용한다."""
    grounded, normalizations = normalize_relation_claim_encodings(source_grounded, package)
    grounded, direction_normalizations = normalize_single_direction_variants(grounded, hypotheses)
    normalizations.extend(direction_normalizations)
    validation, state = _validation_and_state(grounded, hypotheses, package)
    return {
        "payload": grounded,
        "input_audit": G.input_audit(hypotheses, package),
        "validation": validation,
        "agent_invocations": [],
        "normalizations": normalizations,
        "revalidated_from": str(source_artifact_id),
        "state": state,
    }


def _failure(role: str, prompt: str, error: H.AgentCallError,
             audit: Mapping[str, Any], input_payload: Mapping[str, Any],
             config: G.GroundingConfig, invocations: list[AgentInvocation]) -> dict[str, Any]:
    return {
        "payload": {"hypotheses": []},
        "input_audit": audit,
        "grounding_input_sha256": sha256_json(input_payload)[:16],
        "validation": {"problems": ["Agent 호출 실패"], "warnings": []},
        "agent_invocations": [item.as_dict() for item in invocations],
        "agent_error": {
            "role": role,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            **error.as_dict(),
        },
        "config": asdict(config),
        "state": "AGENT_CALL_FAILED",
    }
