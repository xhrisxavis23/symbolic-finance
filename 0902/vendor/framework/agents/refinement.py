"""손실 장부 feedback에서 완전한 새 진입식을 만드는 Agent 역할."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .. import catalog, path_alignment as PA
from .runtime import (MODEL, REASONING_EFFORT, AgentInvocation, Runner,
                      invoke)


FEEDBACK_READY, PROPOSED, NO_CHANGE = "FEEDBACK_READY", "PROPOSED", "NO_CHANGE"


@dataclass(frozen=True)
class RefinementAgentConfig:
    """손실 장부 해석 호출의 모델 설정과 제안 수."""

    model: str = MODEL
    effort: str = REASONING_EFFORT
    max_proposals: int = 3


def _catalog_names(catalog_context: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("name"))
        for item in catalog_context.get("features") or []
        if isinstance(item, Mapping) and item.get("name")
    }


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _threshold_targets(expression: Mapping[str, Any]) -> set[str]:
    """임계 연산자의 직접 입력이 된 feature. 재료로 쓴 것은 잡지 않는다."""
    return {
        str(node["input"]["primitive_id"])
        for node in _walk(expression)
        if isinstance(node, Mapping) and node.get("op") in {"compare", "crossover"}
        and isinstance(node.get("input"), Mapping)
        and node["input"].get("op") == "primitive"
    }


def _feature_names(expression: Mapping[str, Any]) -> set[str]:
    return {
        str(node["primitive_id"])
        for node in _walk(expression)
        if isinstance(node, Mapping) and node.get("op") == "primitive"
    }


def _unresolved_names(expression: Mapping[str, Any]) -> set[str]:
    return {
        value.removeprefix(catalog.UNRESOLVED_PREFIX)
        for value in _walk(expression)
        if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX)
    }


def feedback_prompt(loss_profile: Mapping[str, Any], current_contract: Mapping[str, Any], *,
                    catalog_context: Mapping[str, Any],
                    parent_alignment: Mapping[str, Any] | None = None) -> str:
    """첫 호출은 관찰과 해석만 쓴다. 아직 수식을 만들지 않는다."""
    schema = {
        "state": "FEEDBACK_READY | NO_CHANGE",
        "observation": "손실 장부에서 직접 관찰된 차이",
        "loss_mechanism": "관찰을 설명할 수 있는 후보 메커니즘",
        "rival_explanation": "단순 거래 감소 등 경쟁 설명",
        "prediction": "후보 메커니즘이 맞다면 다음 재생에서 보여야 할 결과",
        "evidence_features": ["catalog feature name"],
    }
    return f"""# Role

Read a Discovery loss ledger and write natural-language feedback for entry refinement.
Do not write or select an entry formula in this step.

# Rules

- Separate the observed pattern, candidate mechanism, rival explanation, and prediction.
- Use only entry-time features present in the injected Catalog.
- Do not name a numeric threshold, comparator, formula, or exit change.
- `post_entry_feature_changes` explains a path after entry; it cannot become an entry condition.
- The primary result is 23bp-inclusive total net. A smaller loss caused only by fewer decisions
  is a rival explanation, not evidence that the mechanism improved.
- `path_observation` reports what the ledger actually showed on the six prediction fields. Name which of
  them your observation rests on.
- Capturing the spread is a legitimate mechanism, not a defect. So is landing on a few symbol-days,
  as long as the same expression was applied identically to every symbol. Neither is evidence
  against the hypothesis.
- The parent's own path prediction and whether each field held is given below. Where it did not
  hold, explain that first. A new direction that ignores a failed prediction is not an observation.
- Return `NO_CHANGE` when the ledger does not support a distinct, testable refinement direction.

# Output JSON schema

{json.dumps(schema, ensure_ascii=False, indent=1)}

# Current entry contract

{json.dumps(dict(current_contract), ensure_ascii=False, indent=1)}

# Parent path prediction and whether it held

{json.dumps(dict(parent_alignment) if parent_alignment else
            {"state": "NO_PREDICTION_ON_PARENT"}, ensure_ascii=False, indent=1)}

# Loss-ledger profile

{json.dumps(dict(loss_profile), ensure_ascii=False, indent=1)}

# Injected Catalog

{json.dumps(dict(catalog_context), ensure_ascii=False, indent=1)}
"""


def hypothesis_prompt(feedback: Mapping[str, Any], current_contract: Mapping[str, Any], *,
                      catalog_context: Mapping[str, Any],
                      config: RefinementAgentConfig) -> str:
    """두 번째 호출만 feedback을 완전한 실행 진입식으로 바꾼다."""
    schema = {
        "state": "PROPOSED | NO_CHANGE",
        "proposals": [{
            "hypothesis": "feedback에서 도출한 검증 가능한 진입 가설",
            "mechanism": "왜 이 entry-time 상태가 손실을 줄일 수 있는지",
            "rival_explanation": "거래 감소 등 경쟁 설명",
            "path_prediction": {name: " | ".join(values) for name, values in PA.FIELDS.items()},
            "evidence_features": ["Catalog feature"],
            "entry_expression": {"op": "any valid injected Catalog expression tree"},
        }],
    }
    return f"""# Role

Turn the frozen natural-language feedback into complete executable LONG entry hypotheses.
Each proposal replaces the complete current entry expression.

# Rules

- Create at most {config.max_proposals} materially different proposals.
- You may freely compose any injected operator, causal Catalog feature, raw causal input,
  comparator, fixed numeric value, rolling window, persistence, sequence, z-score, AND, or OR.
- `entry_expression` must be a complete boolean/event expression, not a patch or guard list.
- Existing `UNRESOLVED:<name>` values may be copied from the current expression. New unresolved
  names cannot be exact-replayed, so use executable numeric values for new thresholds and structure.
- Do not change LONG side, exit, fee, queue model, dates, symbols, or decision accounting.
- A separate exact canonical replay evaluates every proposal. It does not require a decision floor
  or improvement over the current parent.
- A raw threshold on a feature means the same number must be right for every symbol, and symbols
  differ by orders of magnitude. The injected operator list is the whole vocabulary.
  Check `observed` for the range before writing any number.
- Every proposal must carry `path_prediction`: what the ledger will show if this proposal is
  right. Same six prediction fields and same allowed values as the parent prediction above. It is checked
  mechanically after the replay.
  Three fields describe the trade (bid_move_bps, fill_slippage_bps, loss_kind); three compare
  the decisions that filled against the ones that did not (fill_rate, fill_spread_selection,
  fill_opportunity_selection). Every comparison uses a {PA.MOVE_DEAD_ZONE_BPS:g} bps tie band.
  No answer is preferred or penalised. Capturing the spread is a legitimate mechanism — such a
  proposal should say bid_move NEUTRAL and fill_spread_selection MORE. Answer NEUTRAL or SAME
  only where the proposal genuinely makes no claim on that field.
- Return `NO_CHANGE` with an empty proposal list if the feedback does not support a formula.

# Output JSON schema

{json.dumps(schema, ensure_ascii=False, indent=1)}

# Frozen natural-language feedback

{json.dumps(dict(feedback), ensure_ascii=False, indent=1)}

# Current entry contract

{json.dumps(dict(current_contract), ensure_ascii=False, indent=1)}

# Injected Catalog and expression DSL

{json.dumps(dict(catalog_context), ensure_ascii=False, indent=1)}
"""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_feedback(output: Mapping[str, Any], *,
                      catalog_context: Mapping[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    state = output.get("state")
    if state not in (FEEDBACK_READY, NO_CHANGE):
        problems.append("feedback state가 FEEDBACK_READY 또는 NO_CHANGE가 아니다")
    for key in ("observation", "loss_mechanism", "rival_explanation", "prediction"):
        if not _text(output.get(key)):
            problems.append(f"{key}가 비었다")
    features = output.get("evidence_features")
    allowed = _catalog_names(catalog_context)
    if state == FEEDBACK_READY and (not isinstance(features, list) or not features):
        problems.append("evidence_features가 비었다")
    elif isinstance(features, list) and any(str(feature) not in allowed for feature in features):
        problems.append("evidence_features에 Catalog 밖 feature가 있다")
    return {"ok": not problems, "problems": problems}


def validate_hypotheses(output: Mapping[str, Any], *, catalog_context: Mapping[str, Any],
                        current_contract: Mapping[str, Any],
                        config: RefinementAgentConfig) -> dict[str, Any]:
    problems: list[str] = []
    state = output.get("state")
    if state not in (PROPOSED, NO_CHANGE):
        problems.append("hypothesis state가 PROPOSED 또는 NO_CHANGE가 아니다")
    proposals = output.get("proposals")
    if not isinstance(proposals, list):
        problems.append("proposals가 배열이 아니다")
        proposals = []
    if state == NO_CHANGE and proposals:
        problems.append("NO_CHANGE에는 proposals가 없어야 한다")
    if state == PROPOSED and not proposals:
        problems.append("PROPOSED에는 proposal이 하나 이상 있어야 한다")
    if len(proposals) > int(config.max_proposals):
        problems.append(f"proposal은 최대 {config.max_proposals}개다")

    allowed = _catalog_names(catalog_context)
    current_expression = current_contract.get("entry_expression") or {}
    allowed_unresolved = (_unresolved_names(current_expression)
                          if isinstance(current_expression, Mapping) else set())
    fingerprints: set[str] = set()
    for index, proposal in enumerate(proposals):
        label = f"proposals[{index}]"
        if not isinstance(proposal, Mapping):
            problems.append(f"{label}가 객체가 아니다")
            continue
        for key in ("hypothesis", "mechanism", "rival_explanation"):
            if not _text(proposal.get(key)):
                problems.append(f"{label}.{key}가 비었다")
        prediction = proposal.get("path_prediction")
        if not isinstance(prediction, Mapping):
            problems.append(f"{label}.path_prediction이 객체가 아니다")
        else:
            for name, values in PA.FIELDS.items():
                actual = prediction.get(name)
                if actual is None:
                    problems.append(f"{label}.path_prediction에 {name}이 없다")
                elif str(actual) not in values:
                    problems.append(f"{label}.path_prediction.{name}이 허용값 밖이다: {actual!r}")
            for name in sorted(set(prediction) - set(PA.FIELDS)):
                problems.append(f"{label}.path_prediction에 모르는 항목이 있다: {name}")
        evidence = proposal.get("evidence_features")
        if not isinstance(evidence, list) or not evidence:
            problems.append(f"{label}.evidence_features가 비었다")
        elif any(str(feature) not in allowed for feature in evidence):
            problems.append(f"{label}.evidence_features에 Catalog 밖 feature가 있다")

        expression = proposal.get("entry_expression")
        if not isinstance(expression, Mapping):
            problems.append(f"{label}.entry_expression이 객체가 아니다")
            continue
        try:
            inferred = catalog.infer_expression_type(expression, allow_unresolved=True)
            if inferred.value_type not in {"boolean", "event"}:
                problems.append(f"{label}.entry_expression 결과가 boolean/event가 아니다")
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"{label}.entry_expression DSL 오류: {error}")
        for feature in sorted(_feature_names(expression)):
            try:
                resolved = catalog.resolve(feature)
            except (KeyError, ValueError):
                problems.append(f"{label}.entry_expression에 Catalog 밖 feature가 있다: {feature}")
                continue
            if resolved not in catalog.FEATURES or not catalog.FEATURES[resolved].causal:
                problems.append(f"{label}.entry_expression에 미래 feature가 있다: {feature}")
        for target in sorted(_threshold_targets(expression)):
            problem = catalog.condition_problem(target)
            if problem:
                problems.append(f"{label}.entry_expression: {problem}")
        new_unresolved = _unresolved_names(expression) - allowed_unresolved
        if new_unresolved:
            problems.append(f"{label}.entry_expression에 실행값 없는 새 UNRESOLVED가 있다: "
                            f"{sorted(new_unresolved)}")
        fingerprint = json.dumps(expression, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            problems.append(f"{label}가 앞 proposal과 같은 실행식이다")
        fingerprints.add(fingerprint)
    return {"ok": not problems, "problems": problems, "proposal_count": len(proposals)}


def repair_prompt(output: Mapping[str, Any], problems: Sequence[str], *, stage: str,
                  catalog_context: Mapping[str, Any], current_contract: Mapping[str, Any],
                  config: RefinementAgentConfig) -> str:
    """계약 위반만 한 번 고친다. Catalog는 repair에도 항상 다시 주입한다."""
    return f"""# Task

Repair the `{stage}` JSON below. Preserve its economic meaning and return JSON only.

# Problems

{json.dumps(list(problems), ensure_ascii=False, indent=1)}

# Limits

- Maximum proposals: {config.max_proposals}
- Each proposal contains one complete executable `entry_expression`.
- LONG side and the canonical exit/execution contract stay fixed.
- `path_prediction` allowed values:
{json.dumps({name: list(values) for name, values in PA.FIELDS.items()},
            ensure_ascii=False, indent=1)}

# Current entry contract

{json.dumps(dict(current_contract), ensure_ascii=False, indent=1)}

# Injected Catalog

{json.dumps(dict(catalog_context), ensure_ascii=False, indent=1)}

# Output to repair

{json.dumps(dict(output), ensure_ascii=False, indent=1)}
"""


def run(loss_profile: Mapping[str, Any], *, current_contract: Mapping[str, Any],
        catalog_context: Mapping[str, Any], runner: Runner | None = None,
        parent_alignment: Mapping[str, Any] | None = None,
        config: RefinementAgentConfig = RefinementAgentConfig()) -> dict[str, Any]:
    """자연어 feedback을 먼저 고정한 뒤 완전한 새 진입식을 만든다."""
    feedback, invocation = invoke(
        runner, role="entry_refinement_loss_feedback",
        prompt=feedback_prompt(loss_profile, current_contract,
                               catalog_context=catalog_context,
                               parent_alignment=parent_alignment),
        model=config.model, effort=config.effort)
    invocations: list[AgentInvocation] = [invocation]
    feedback_validation = validate_feedback(feedback, catalog_context=catalog_context)
    if feedback_validation["problems"]:
        feedback, repaired = invoke(
            runner, role="entry_refinement_feedback_repair",
            prompt=repair_prompt(feedback, feedback_validation["problems"], stage="feedback",
                                 catalog_context=catalog_context,
                                 current_contract=current_contract, config=config),
            model=config.model, effort=config.effort)
        invocations.append(repaired)
        feedback_validation = validate_feedback(feedback, catalog_context=catalog_context)
    if not feedback_validation["ok"]:
        return {
            "state": "INVALID_OUTPUT", "feedback": feedback, "payload": {"proposals": []},
            "validation": {"feedback": feedback_validation},
            "agent_invocations": [item.as_dict() for item in invocations],
            "config": asdict(config),
        }
    if feedback.get("state") == NO_CHANGE:
        return {
            "state": "READY", "feedback": feedback,
            "payload": {"state": NO_CHANGE, "proposals": []},
            "validation": {"feedback": feedback_validation,
                           "hypotheses": {"ok": True, "problems": [], "proposal_count": 0}},
            "agent_invocations": [item.as_dict() for item in invocations],
            "config": asdict(config),
        }

    hypotheses, generated = invoke(
        runner, role="entry_refinement_feedback_hypothesis",
        prompt=hypothesis_prompt(feedback, current_contract, catalog_context=catalog_context,
                                 config=config),
        model=config.model, effort=config.effort)
    invocations.append(generated)
    hypothesis_validation = validate_hypotheses(
        hypotheses, catalog_context=catalog_context, current_contract=current_contract,
        config=config)
    if hypothesis_validation["problems"]:
        hypotheses, repaired = invoke(
            runner, role="entry_refinement_hypothesis_repair",
            prompt=repair_prompt(hypotheses, hypothesis_validation["problems"], stage="hypotheses",
                                 catalog_context=catalog_context,
                                 current_contract=current_contract, config=config),
            model=config.model, effort=config.effort)
        invocations.append(repaired)
        hypothesis_validation = validate_hypotheses(
            hypotheses, catalog_context=catalog_context, current_contract=current_contract,
            config=config)
    return {
        "state": "READY" if hypothesis_validation["ok"] else "INVALID_OUTPUT",
        "feedback": feedback,
        "payload": hypotheses,
        "validation": {"feedback": feedback_validation, "hypotheses": hypothesis_validation},
        "agent_invocations": [item.as_dict() for item in invocations],
        "config": asdict(config),
    }
