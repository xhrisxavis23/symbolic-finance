"""Grounding 결과를 Specification이 그대로 소비할 계획으로 고정한다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .. import grounding as G
from ..config import now_utc


PLAN_SCHEMA = "validation_plan.v1"


def select_hypotheses(payload: Mapping[str, Any], hypothesis_ids: Sequence[str] | None = None
                      ) -> dict[str, Any]:
    """순번이 아니라 id로 고른다. 중복 판정된 후보는 다음 단계로 넘기지 않는다."""
    items = list(payload.get("hypotheses") or [])
    source_ids = [str(item.get("hypothesis_id")) for item in items]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Hypothesis artifact 안에 hypothesis_id 가 중복됐다")
    duplicate = {str(item.get("hypothesis_id"))
                 for item in (payload.get("novelty") or {}).get("decisions") or []
                 if item.get("status") == "DUPLICATE_HYPOTHESIS"}
    available = {str(item.get("hypothesis_id")): item for item in items
                 if str(item.get("hypothesis_id")) not in duplicate}
    ids = list(hypothesis_ids) if hypothesis_ids is not None else list(available)
    if not ids:
        raise ValueError("Grounding 할 hypothesis 가 없다")
    unknown = [tag for tag in ids if tag not in available]
    if unknown:
        duplicate_requested = sorted(set(unknown) & duplicate)
        if duplicate_requested:
            raise ValueError(f"과거 후보와 중복된 hypothesis는 Grounding할 수 없다: {duplicate_requested}")
        raise ValueError(f"없는 hypothesis_id: {unknown}")
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis_id 가 중복됐다")
    selected = dict(payload)
    selected["hypotheses"] = [available[tag] for tag in ids]
    return selected


def build_validation_plan(grounded: Mapping[str, Any], hypotheses: Mapping[str, Any], *,
                          lineage: Mapping[str, str]) -> dict[str, Any]:
    """Grounding prediction을 적을 뿐, 날짜나 Backtest 결과를 열지 않는다."""
    draft = G.validation_plan(grounded, hypotheses)
    allowed = {str(item.get("hypothesis_id")) for item in grounded.get("hypotheses") or []}
    targets = []
    for index, target in enumerate(draft.get("targets") or [], start=1):
        if str(target.get("hypothesis_id")) not in allowed:
            continue
        item = dict(target)
        item["target_id"] = f"VP-{item['hypothesis_id']}-{index:03d}"
        item["status"] = "PLANNED"
        item["result"] = None
        targets.append(item)
    return {
        "schema": PLAN_SCHEMA,
        "created_at": now_utc(),
        "state": "READY_FOR_VALIDATION",
        "not_executed": True,
        "lineage": {str(key): str(value) for key, value in sorted(lineage.items())},
        "hypothesis_ids": sorted(allowed),
        "targets": targets,
        "target_count": len(targets),
        "note": "Grounding prediction 기록이다. Specification은 이 artifact의 hypothesis만 입력으로 받는다. canonical Validation Backtest는 parameter lock 뒤에 실행한다.",
    }
