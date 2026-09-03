"""Evidence 관계 하나를 가설 탐색의 읽기 전용 초점으로 만든다.

이 모듈은 가설이나 entry rule을 만들지 않는다. Evidence Package가 이미 확정한
family 관계를 한 쌍씩 꺼내 Agent가 같은 넓은 이야기를 반복하지 않게 할 뿐이다.
"""

from __future__ import annotations

from typing import Any, Mapping

from .. import hypothesis as H


SUPPORTED_RELATIONS = {
    "COUPLED": "COUPLED_MECHANISM",
    "CONTEXT_TRIGGER": "CONTEXT_TRIGGER",
    "CONTEXT_INDEPENDENT": "CONTEXT_INDEPENDENT",
}


def build(package: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """입력 감사에서 살아남은 두 family의 관계만 lens로 남긴다."""
    usable_ids = {str(value) for value in audit.get("usable_evidence") or []}
    usable_by_family: dict[str, list[str]] = {}
    for item in package.get("evidence_families") or []:
        evidence_id, family = str(item.get("evidence_id")), str(item.get("family"))
        if evidence_id in usable_ids:
            usable_by_family.setdefault(family, []).append(evidence_id)

    lenses: list[dict[str, Any]] = []
    relation_types_by_family: dict[str, set[str]] = {}
    for relation in package.get("joint_evidence") or []:
        relation_type = str(relation.get("relation"))
        for family in (str(relation.get("family_a")), str(relation.get("family_b"))):
            relation_types_by_family.setdefault(family, set()).add(relation_type)
    for relation in package.get("joint_evidence") or []:
        family_a, family_b = str(relation.get("family_a")), str(relation.get("family_b"))
        relation_type = str(relation.get("relation"))
        structure = SUPPORTED_RELATIONS.get(relation_type)
        if structure is None or not usable_by_family.get(family_a) or not usable_by_family.get(family_b):
            continue
        lens_id = f"{relation_type.lower()}:{family_a}__{family_b}"
        lenses.append({
            "lens_id": lens_id,
            "joint_evidence_id": str(relation.get("joint_evidence_id")),
            "families": [family_a, family_b],
            "relation": relation_type,
            "allowed_hypothesis_structure": structure,
            "context_family": relation.get("context_family"),
            "trigger_family": relation.get("trigger_family"),
            "ordering_status": relation.get("ordering_status"),
            "relation_reason": relation.get("relation_reason"),
            "allowed_evidence_ids": {
                family_a: sorted(usable_by_family[family_a]),
                family_b: sorted(usable_by_family[family_b]),
            },
            "boundary": (
                "이 관계는 Evidence가 허용한 탐색 초점일 뿐이다. 관계를 인과로 올리거나 "
                "다른 family를 핵심 근거로 바꾸지 않는다. 메커니즘이 불명확하면 관계를 "
                "검증하는 낮은 확신도의 reduced-form 후보를 만든다."),
        })
    for family, evidence_ids in usable_by_family.items():
        if relation_types_by_family.get(family) == {"REDUNDANT"}:
            continue
        lenses.append({
            "lens_id": f"single:{family}",
            "joint_evidence_id": None,
            "families": [family],
            "relation": "SINGLE",
            "allowed_hypothesis_structure": "SINGLE_MECHANISM",
            "context_family": None,
            "trigger_family": None,
            "ordering_status": None,
            "relation_reason": "한 family의 직접 관측만으로 별도 메커니즘을 검토한다",
            "allowed_evidence_ids": {family: sorted(evidence_ids)},
            "boundary": (
                "이 lens는 한 family의 직접 관측만 쓴다. 다른 family를 근거에 덧붙여 "
                "약한 결합을 만들지 않는다. 단일 상태가 별도 경제 메커니즘과 반증 가능한 "
                "예측을 충분히 설명하지 못해도 단일 관측을 검증하는 후보를 만든다."),
        })
    # canonical anchor execution-state는 원 Profile oracle level과 표본·label이
    # 다르다. family-only lens에 섞으면 같은 family의 둘을 독립 확인처럼 쓰게 되므로
    # evidence id 하나를 고정한 별도 lens만 연다.
    for item in H.execution_state_evidence_items(package):
        evidence_id = str(item.get("evidence_id"))
        family = str(item.get("family"))
        if evidence_id not in usable_ids or not family:
            continue
        lenses.append({
            "lens_id": f"single-execution-state:{evidence_id}",
            "joint_evidence_id": None,
            "families": [family],
            "relation": "SINGLE",
            "allowed_hypothesis_structure": "SINGLE_MECHANISM",
            "context_family": None,
            "trigger_family": family,
            "ordering_status": "ANCHOR_STATE",
            "relation_reason": "개별 canonical anchor replay label과 t=0 상태 대조",
            "allowed_evidence_ids": {family: [evidence_id]},
            "required_evidence_ids": [evidence_id],
            "boundary": (
                f"이 lens는 `{evidence_id}` 하나만 근거로 쓴다. 원 Profile BACKGROUND "
                "anchor의 독립 canonical replay 진단이므로 전략 PnL이나 oracle level "
                "Evidence와 결합하지 않는다. 경제 메커니즘이 불명확하면 관측 상태 자체를 "
                "검증하는 낮은 확신도의 후보를 만든다."),
        })
    # raw 30초 경로 Evidence도 원 Profile oracle cohort와 label이 다르다. family lens에
    # 섞어 독립 확인처럼 세지 않고, cohort 대조 하나를 고정한 단일 lens로만 연다.
    for item in H.raw_path_evidence_items(package):
        evidence_id = str(item.get("evidence_id"))
        family = str(item.get("family"))
        if evidence_id not in usable_ids or not family:
            continue
        positive = str(item.get("raw_path_positive_label"))
        negative = str(item.get("raw_path_negative_label"))
        lenses.append({
            "lens_id": f"single-raw-path:{evidence_id}",
            "joint_evidence_id": None,
            "families": [family],
            "relation": "SINGLE",
            "allowed_hypothesis_structure": "SINGLE_MECHANISM",
            "context_family": None,
            "trigger_family": family,
            "ordering_status": "ANCHOR_STATE",
            "relation_reason": "체결 뒤 30초 BID1 raw path 두 cohort와 t=0 상태 대조",
            "allowed_evidence_ids": {family: [evidence_id]},
            "required_evidence_ids": [evidence_id],
            "raw_path_labels": {"positive": positive, "negative": negative},
            "boundary": (
                f"이 lens는 `{evidence_id}` 하나만 근거로 쓴다. BACKGROUND anchor의 "
                f"개별 canonical queue fill raw path `{positive}`/`{negative}` 대조이며, "
                "fixed exit PnL이나 oracle level Evidence와 결합하지 않는다. 별도 경제 "
                "메커니즘이 불명확하면 경로 대조 자체를 검증하는 후보를 만든다."),
        })
    # Temporal Evidence는 같은 feature라도 lag 하나마다 다른 entry-timing 가설이다.
    # family-only lens로 열면 Agent가 같은 family의 다른 lag를 바꿔 인용할 수 있으므로,
    # 여기서는 evidence id 하나를 고정한 단일 lens를 별도로 만든다.
    for item in H.execution_temporal_evidence_items(package):
        evidence_id = str(item.get("evidence_id"))
        family = str(item.get("family"))
        if evidence_id not in usable_ids or not family:
            continue
        expression = item.get("expression") or {}
        lag = expression.get("lag")
        lenses.append({
            "lens_id": f"single-temporal:{evidence_id}",
            "joint_evidence_id": None,
            "families": [family],
            "relation": "SINGLE",
            "allowed_hypothesis_structure": "SINGLE_MECHANISM",
            "context_family": None,
            "trigger_family": family,
            "ordering_status": "PRE_ANCHOR_CHANGE",
            "relation_reason": "저장된 Profile의 exact pre-anchor 변화량 관측",
            "allowed_evidence_ids": {family: [evidence_id]},
            "required_evidence_ids": [evidence_id],
            "temporal_expression": item.get("expression"),
            "boundary": (
                f"이 lens는 `{evidence_id}` 하나만 근거로 쓴다. 저장된 t-{lag}초→t "
                "difference 식·방향·TRIGGER 역할을 바꾸거나 같은 feature의 다른 lag를 "
                "섞지 않는다. 경제 메커니즘이 불명확하면 저장된 변화량 자체를 검증하는 "
                "후보를 만든다."),
        })
    lenses.sort(key=lambda item: item["lens_id"])
    return {
        "schema": "mechanism_lens_catalog.v1",
        "usable_evidence_ids": sorted(usable_ids),
        "lenses": lenses,
        "note": "lens는 새 Evidence나 가설이 아니다. 이미 허용된 family 관계의 탐색 순서다.",
    }


def select(catalog: Mapping[str, Any], lens_id: str) -> dict[str, Any]:
    """id가 정확히 일치하는 lens 하나만 연다."""
    matches = [dict(item) for item in catalog.get("lenses") or []
               if str(item.get("lens_id")) == str(lens_id)]
    if len(matches) != 1:
        available = ", ".join(str(item.get("lens_id")) for item in catalog.get("lenses") or [])
        raise ValueError(f"research lens '{lens_id}'가 없다. 사용 가능: {available or '없음'}")
    return matches[0]
