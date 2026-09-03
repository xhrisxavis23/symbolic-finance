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
