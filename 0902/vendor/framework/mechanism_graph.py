"""가설·Grounding·실행 명세가 함께 쓰는 작은 메커니즘 계약.

이 graph는 전략 탐색 언어가 아니다. Evidence에서 말한 관측, 그것을 다시 볼 수 있는
주문 상태, 그리고 이미 허용된 BID1 대기 정책을 한 객체로 보존한다. 미래 결과는
``prediction``에서만 평가 대상으로 적을 수 있고 entry/pending 행동의 입력이 될 수 없다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "mechanism_graph.v1"
OBSERVATION_STATES = ("DECISION", "PENDING")
ENTRY_STATE = "DECISION"
PENDING_STATE = "PENDING"
FILLED_STATE = "FILLED"
ENTRY_ACTION = "BID1_QUEUE"
LIFECYCLE_MODES = ("HOLD_THROUGH", "CANCEL_WHEN_ENTRY_SIGNAL_FALSE")
RELATION_TYPES = ("CONTEXT_TRIGGER", "COUPLED", "CONTEXT_INDEPENDENT")
PREDICTION_TARGETS = ("PATH_TARGET", "MATCHED_CONTROL_TARGET")
DIRECTIONS = ("HIGHER", "LOWER")


def _canonical(value: Any) -> Any:
    """hash 이전의 순서만 고정한다. 의미 있는 list 순서는 바꾸지 않는다."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def semantic_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Grounding이 채우는 catalog binding을 제외한, 바뀌면 안 되는 graph 부분."""
    value = copy.deepcopy(dict(graph))
    value.pop("catalog_bindings", None)
    return _canonical(value)


def semantic_hash(graph: Mapping[str, Any]) -> str:
    return _hash(semantic_payload(graph))


def full_hash(graph: Mapping[str, Any]) -> str:
    return _hash(dict(graph))


def _strings(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if str(value)}


def validate(graph: Any, *, known_evidence_ids: Iterable[Any] = (),
             known_features: Iterable[Any] = (), require_bindings: bool = False) -> dict[str, Any]:
    """graph의 시간 경계·참조·실행 정책을 결정적으로 검사한다."""
    problems: list[str] = []
    if not isinstance(graph, Mapping):
        return {"problems": ["mechanism_graph가 객체가 아니다"], "semantic_hash": None,
                "full_hash": None, "observable_ids": [], "binding_count": 0}
    graph = dict(graph)
    allowed = {"schema", "observables", "relations", "execution", "prediction", "catalog_bindings"}
    extra = sorted(set(graph) - allowed)
    if extra:
        problems.append(f"mechanism_graph에 허용되지 않은 필드가 있다: {extra}")
    if graph.get("schema") != SCHEMA_VERSION:
        problems.append(f"mechanism_graph.schema가 {SCHEMA_VERSION!r}가 아니다")

    known_evidence = _strings(known_evidence_ids)
    known_feature_set = _strings(known_features)
    observables = graph.get("observables")
    if not isinstance(observables, list) or not observables:
        problems.append("mechanism_graph.observables가 비어 있다")
        observables = []
    observable_ids: set[str] = set()
    observable_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(observables):
        label = f"observables[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{label}가 객체가 아니다")
            continue
        if extra := sorted(set(item) - {"node_id", "evidence_id", "available_at"}):
            problems.append(f"{label}에 허용되지 않은 필드가 있다: {extra}")
        node_id = str(item.get("node_id") or "")
        if not node_id:
            problems.append(f"{label}.node_id가 없다")
        elif node_id in observable_ids:
            problems.append(f"observable node_id가 중복됐다: {node_id}")
        else:
            observable_ids.add(node_id)
            observable_by_id[node_id] = item
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id:
            problems.append(f"{label}.evidence_id가 없다")
        elif known_evidence and evidence_id not in known_evidence:
            problems.append(f"{label}.evidence_id가 입력 Evidence에 없다: {evidence_id}")
        if item.get("available_at") not in OBSERVATION_STATES:
            problems.append(f"{label}.available_at은 {OBSERVATION_STATES} 중 하나여야 한다")

    relations = graph.get("relations")
    if not isinstance(relations, list):
        problems.append("mechanism_graph.relations가 list가 아니다")
        relations = []
    relation_ids: set[str] = set()
    for index, item in enumerate(relations):
        label = f"relations[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{label}가 객체가 아니다")
            continue
        if extra := sorted(set(item) - {"node_id", "relation", "input_ids", "available_at"}):
            problems.append(f"{label}에 허용되지 않은 필드가 있다: {extra}")
        node_id = str(item.get("node_id") or "")
        if not node_id:
            problems.append(f"{label}.node_id가 없다")
        elif node_id in observable_ids or node_id in relation_ids:
            problems.append(f"graph node_id가 중복됐다: {node_id}")
        else:
            relation_ids.add(node_id)
        if item.get("relation") not in RELATION_TYPES:
            problems.append(f"{label}.relation이 유효하지 않다")
        inputs = item.get("input_ids")
        if not isinstance(inputs, list) or not inputs:
            problems.append(f"{label}.input_ids가 비어 있다")
        else:
            unknown = sorted(_strings(inputs) - observable_ids)
            if unknown:
                problems.append(f"{label}.input_ids에 없는 observable이 있다: {unknown}")
        if item.get("available_at") not in OBSERVATION_STATES:
            problems.append(f"{label}.available_at은 {OBSERVATION_STATES} 중 하나여야 한다")

    execution = graph.get("execution")
    if not isinstance(execution, Mapping):
        problems.append("mechanism_graph.execution이 객체가 아니다")
        execution = {}
    elif extra := sorted(set(execution) - {"entry_state", "pending_state", "entry_action",
                                            "entry_lifecycle_policy"}):
        problems.append(f"execution에 허용되지 않은 필드가 있다: {extra}")
    if execution.get("entry_state") != ENTRY_STATE:
        problems.append(f"execution.entry_state는 {ENTRY_STATE}여야 한다")
    if execution.get("pending_state") != PENDING_STATE:
        problems.append(f"execution.pending_state는 {PENDING_STATE}여야 한다")
    if execution.get("entry_action") != ENTRY_ACTION:
        problems.append(f"execution.entry_action은 {ENTRY_ACTION}여야 한다")
    lifecycle = execution.get("entry_lifecycle_policy")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("mode") not in LIFECYCLE_MODES:
        problems.append(f"execution.entry_lifecycle_policy.mode는 {LIFECYCLE_MODES} 중 하나여야 한다")
        lifecycle = {}
    elif extra := sorted(set(lifecycle) - {"mode", "pending_observable_ids"}):
        problems.append(f"entry_lifecycle_policy에 허용되지 않은 필드가 있다: {extra}")
    pending_ids = lifecycle.get("pending_observable_ids", [])
    if not isinstance(pending_ids, list):
        problems.append("entry_lifecycle_policy.pending_observable_ids가 list가 아니다")
        pending_ids = []
    unknown_pending = sorted(_strings(pending_ids) - observable_ids)
    if unknown_pending:
        problems.append(f"pending_observable_ids에 없는 observable이 있다: {unknown_pending}")
    if lifecycle.get("mode") == "HOLD_THROUGH" and pending_ids:
        problems.append("HOLD_THROUGH에는 pending_observable_ids가 있으면 안 된다")
    if lifecycle.get("mode") == "CANCEL_WHEN_ENTRY_SIGNAL_FALSE":
        if not pending_ids:
            problems.append("CANCEL_WHEN_ENTRY_SIGNAL_FALSE에는 pending_observable_ids가 필요하다")
        if set(_strings(pending_ids)) != observable_ids:
            problems.append("현재 실행기는 전체 entry signal을 다시 계산하므로 취소 정책은 모든 observable을 PENDING으로 선언해야 한다")
        not_pending = sorted(node_id for node_id in _strings(pending_ids)
                             if observable_by_id.get(node_id, {}).get("available_at") != PENDING_STATE)
        if not_pending:
            problems.append(f"취소 정책은 PENDING에서 관측 가능한 상태만 쓸 수 있다: {not_pending}")

    prediction = graph.get("prediction")
    if not isinstance(prediction, Mapping):
        problems.append("mechanism_graph.prediction이 객체가 아니다")
        prediction = {}
    elif extra := sorted(set(prediction) - {"input_ids", "evaluation_state", "target_type", "direction"}):
        problems.append(f"prediction에 허용되지 않은 필드가 있다: {extra}")
    prediction_inputs = prediction.get("input_ids")
    all_nodes = observable_ids | relation_ids
    if not isinstance(prediction_inputs, list) or not prediction_inputs:
        problems.append("prediction.input_ids가 비어 있다")
    elif (unknown := sorted(_strings(prediction_inputs) - all_nodes)):
        problems.append(f"prediction.input_ids에 없는 graph node가 있다: {unknown}")
    if prediction.get("evaluation_state") != FILLED_STATE:
        problems.append(f"prediction.evaluation_state는 {FILLED_STATE}여야 한다")
    if prediction.get("target_type") not in PREDICTION_TARGETS:
        problems.append(f"prediction.target_type은 {PREDICTION_TARGETS} 중 하나여야 한다")
    if prediction.get("direction") not in DIRECTIONS:
        problems.append(f"prediction.direction은 {DIRECTIONS} 중 하나여야 한다")

    bindings = graph.get("catalog_bindings", [])
    if not isinstance(bindings, list):
        problems.append("mechanism_graph.catalog_bindings가 list가 아니다")
        bindings = []
    binding_ids: set[str] = set()
    for index, item in enumerate(bindings):
        label = f"catalog_bindings[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{label}가 객체가 아니다")
            continue
        if extra := sorted(set(item) - {"observable_id", "catalog_feature"}):
            problems.append(f"{label}에 허용되지 않은 필드가 있다: {extra}")
        node_id = str(item.get("observable_id") or "")
        bound_feature = str(item.get("catalog_feature") or "")
        if node_id not in observable_ids:
            problems.append(f"{label}.observable_id가 graph observable에 없다: {node_id}")
        elif node_id in binding_ids:
            problems.append(f"{label}.observable_id가 중복됐다: {node_id}")
        else:
            binding_ids.add(node_id)
        if not bound_feature:
            problems.append(f"{label}.catalog_feature가 없다")
        elif known_feature_set and bound_feature not in known_feature_set:
            problems.append(f"{label}.catalog_feature가 알려진 feature가 아니다: {bound_feature}")
    if require_bindings and binding_ids != observable_ids:
        problems.append("Grounding graph에는 모든 observable의 catalog binding이 하나씩 필요하다")

    return {"problems": problems, "semantic_hash": semantic_hash(graph), "full_hash": full_hash(graph),
            "observable_ids": sorted(observable_ids), "binding_count": len(binding_ids)}


def validate_preservation(source: Any, grounded: Any, *, known_evidence_ids: Iterable[Any] = (),
                          known_features: Iterable[Any] = ()) -> dict[str, Any]:
    """Grounding이 graph의 뜻을 바꾸지 않고 catalog binding만 채웠는지 검사한다."""
    source_report = validate(source, known_evidence_ids=known_evidence_ids,
                             known_features=known_features, require_bindings=False)
    grounded_report = validate(grounded, known_evidence_ids=known_evidence_ids,
                               known_features=known_features, require_bindings=True)
    problems = [*(f"source graph: {problem}" for problem in source_report["problems"]),
                *(f"grounded graph: {problem}" for problem in grounded_report["problems"])]
    if not source_report["problems"] and not grounded_report["problems"]:
        if source_report["semantic_hash"] != grounded_report["semantic_hash"]:
            problems.append("Grounding이 MechanismGraph의 의미를 바꿨다")
    return {"problems": problems,
            "source_semantic_hash": source_report["semantic_hash"],
            "grounded_semantic_hash": grounded_report["semantic_hash"],
            "grounded_full_hash": grounded_report["full_hash"]}
