"""정적 검사. 계획서 §3 S5 ③.

이것이 파이프라인의 진짜 게이트다. S1 게이트가 교사 오염을 막는다면 여기는
**수식이 실행 계약을 몰래 바꾸는 것**을 막는다. 후자가 더 은밀하다 — 청산을
살짝 유리하게 바꾼 수식은 백테스트에서 아름답게 나온다.
"""

from __future__ import annotations

from typing import Any, Mapping

from .. import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402

# 진입식이 선언해서는 안 되는 키. 실행은 정본이 정한다.
FORBIDDEN_KEYS = ("execution_binding", "exit", "exit_rule", "stop_gross_bps",
                  "trailing_drawdown_gross_bps", "horizon_seconds", "fee_bps")


class CheckError(ValueError):
    """정적 검사 탈락. 사유가 메시지에 들어 있다."""


def static_check(ast: Mapping[str, Any], *, allow_unresolved: bool) -> None:
    _reject_execution_override(ast, "$")
    try:
        _catalog.infer_expression_type(ast, allow_unresolved=allow_unresolved)
    except _catalog.ExpressionError as error:
        raise CheckError(str(error)) from error
    except (KeyError, ValueError) as error:
        raise CheckError(f"어휘 또는 값 오류: {error}") from error
    _reject_banned_condition_targets(ast, "$")


def _reject_execution_override(node: Any, path: str) -> None:
    """진입식이 실행 계약을 정하려 들면 조용히 넘어가지 않는다."""
    if isinstance(node, Mapping):
        for key in FORBIDDEN_KEYS:
            if key in node:
                raise CheckError(
                    f"{path}: 진입식이 실행 결속 {key!r} 를 선언했다. "
                    "청산·큐·비용은 정본이 정하고 진입식은 정하지 못한다")
        for key, value in node.items():
            _reject_execution_override(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _reject_execution_override(value, f"{path}[{i}]")


# 리터럴 임계 하나와 값 하나를 직접 비교하는 노드만 "조건"이다.
# `compare_values` 는 두 표현식을 서로 비교할 뿐 리터럴 임계가 없어 대상이
# 아니다 (DESIGN.md D14 정정 · catalog.condition_problem 참고).
_THRESHOLD_NODE_OPS = ("compare", "crossover")

# `compare`/`crossover` 의 `input` 이 이 연산자로만 감싸여 있으면 "다른 것과
# 결합되지 않은 같은 축"이다 — negate 는 부호만, absolute 는 크기만 바꾼다.
# 실측 근거(DESIGN.md D14): 지난 실행에서 `queue_imbalance_best` 를 그대로
# 조건 대상으로 쓴 e001(`primitive`)과 `sqrt(Abs(...))`에서 sqrt 가 벗겨져
# 남은 e005(`absolute(primitive)`) 둘 다 "직접 대상"으로 지목됐다. 반면
# `add`·`multiply`·`ratio` 등으로 **다른 feature 와 결합**되면 그 순간부터는
# "재료"이지 "직접 대상"이 아니다 (`condition_problem` 의 docstring:
# "다른 식의 재료로는 쓸 수 있다") — 그래서 이 튜플에 넣지 않는다.
_IDENTITY_WRAPPER_OPS = ("negate", "absolute")


def _condition_target(node: Any) -> str | None:
    """`compare`/`crossover` 의 `input` 이 (identity wrapper 를 거쳐) 결국
    단일 primitive 하나뿐이면 그 feature 이름을. 다른 것과 결합됐으면 `None`."""
    while isinstance(node, Mapping) and node.get("op") in _IDENTITY_WRAPPER_OPS:
        node = node.get("input")
    if isinstance(node, Mapping) and node.get("op") == "primitive":
        return None if node.get("primitive_id") is None else str(node["primitive_id"])
    return None


def _reject_banned_condition_targets(node: Any, path: str) -> None:
    """정본 Catalog 의 `feature_use_policy`/`condition_problem` 을 강제한다.

    `queue_imbalance_best` 처럼 `derived_duplicate` 로 금지된 feature 가
    임계 조건의 **직접 대상**이 되는 것만 막는다 — 다른 식의 재료로 쓰는
    것(예: `book_imbalance + queue_imbalance_best`)은 그대로 허용한다.
    이 구분을 코드로 정확히 옮기지 않으면 과잉 거부가 되어 정당한 후보까지
    막힌다 (뮤테이션 자기검토 대상).
    """
    if isinstance(node, Mapping):
        op = node.get("op")
        if op in _THRESHOLD_NODE_OPS and "input" in node:
            target = _condition_target(node["input"])
            if target is not None:
                problem = _catalog.condition_problem(target)
                if problem is not None:
                    raise CheckError(f"{path}.input: {problem}")
        for key, value in node.items():
            _reject_banned_condition_targets(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _reject_banned_condition_targets(value, f"{path}[{i}]")
