"""Feature Profile이 정한 실행 가능한 표본 조건 계약."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from . import catalog
from .config import sha256_json


SCHEMA_VERSION = "executable_sample_condition.v2"


def m0_price_recovery() -> dict[str, Any]:
    """5-tick mid return이 0 이하에서 양수로 바뀌는 M0 사건."""
    return {
        "op": "crossover",
        "input": {"op": "primitive", "primitive_id": "mid_return_5t_bps"},
        "threshold": 0.0,
        "direction": "above",
        "equality": "from_equal",
    }


def for_event_source(event_source: str) -> dict[str, Any]:
    if str(event_source) == "m0_price_recovery":
        return m0_price_recovery()
    raise ValueError(f"지원하지 않는 Feature Profile event source: {event_source}")


def condition_hash(condition: Mapping[str, Any]) -> str:
    return sha256_json(dict(condition))[:16]


def warmup_ticks(condition: Mapping[str, Any] | None) -> int:
    if condition is None:
        return 0
    if dict(condition) == m0_price_recovery():
        return 20
    return 0


def validate(condition: Any) -> dict[str, Any]:
    problems: list[str] = []
    if not isinstance(condition, Mapping):
        return {"ok": False, "problems": ["executable_sample_condition이 객체가 아니다"]}
    try:
        inferred = catalog.infer_expression_type(condition, allow_unresolved=False)
        value_type = getattr(inferred, "value_type", inferred)
        if value_type not in {"boolean", "event"}:
            problems.append("executable_sample_condition은 boolean 또는 event 식이어야 한다")
    except catalog.ExpressionError as error:
        problems.append(f"executable_sample_condition DSL 검증 실패: {error}")
    if "UNRESOLVED:" in repr(condition):
        problems.append("executable_sample_condition에는 미결정 임계값을 둘 수 없다")
    return {"ok": not problems, "problems": problems}


def require(container: Mapping[str, Any], where: str) -> dict[str, Any]:
    condition = container.get("executable_sample_condition")
    report = validate(condition)
    if not report["ok"]:
        raise ValueError(f"{where}: " + "; ".join(report["problems"]))
    expected_hash = container.get("executable_sample_condition_sha256")
    if expected_hash is not None and expected_hash != condition_hash(condition):
        raise ValueError(f"{where}: executable_sample_condition_sha256가 조건과 다르다")
    return copy.deepcopy(dict(condition))


def require_for_event_source(container: Mapping[str, Any], where: str) -> dict[str, Any]:
    """저장 Profile 조건이 현재 event source의 정본 조건과 같은지 확인한다."""
    if container.get("executable_sample_condition_schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{where}: executable_sample_condition schema가 현재 정본과 다르다. "
            "Feature Profile을 다시 만들어야 한다")
    condition = require(container, where)
    expected = for_event_source(str(container.get("event_source") or ""))
    if condition != expected:
        raise ValueError(
            f"{where}: executable_sample_condition이 현재 event source 정본과 다르다. "
            "Feature Profile을 다시 만들어야 한다")
    return condition


def optional(container: Mapping[str, Any], where: str) -> dict[str, Any] | None:
    if container.get("executable_sample_condition") is None:
        return None
    return require(container, where)


def combine(condition: Mapping[str, Any] | None,
            candidate_expression: Mapping[str, Any]) -> dict[str, Any]:
    """Feature Profile 조건을 후보 조건보다 앞에 고정한다."""
    candidate = copy.deepcopy(dict(candidate_expression))
    if condition is None:
        return candidate
    fixed = copy.deepcopy(dict(condition))
    if fixed == candidate:
        return candidate
    return {"op": "all", "args": [fixed, candidate]}
