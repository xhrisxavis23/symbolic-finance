"""sympy 수식을 Catalog AST 로 옮긴다. 계획서 §3 S5 ②.

번역할 수 없는 후보는 **예외로 죽이지 않고 상위에서 기록**한다. 탈락률 자체가
산출물이기 때문이다 (계획서 F7 — 컴파일 성공률 20% 미만이면 중단).
"""

from __future__ import annotations

import sympy

from .. import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402


class TranslationError(ValueError):
    """Catalog 문법으로 옮길 수 없는 수식."""


def translate(expr: sympy.Expr) -> dict:
    """numeric 을 내는 Catalog AST. boolean 으로 감싸는 것은 threshold 단계가 한다."""
    return _walk(sympy.sympify(expr))


def _walk(node: sympy.Expr) -> dict:
    if isinstance(node, sympy.Symbol):
        name = _catalog.resolve(str(node))
        if name not in _catalog.FEATURES:
            raise TranslationError(f"Catalog 에 없는 feature: {node}")
        return {"op": "primitive", "primitive_id": name}

    if node.is_number:
        raise TranslationError(
            "상수만으로 된 가지는 진입식이 될 수 없다. 상수는 임계로 흡수한다")

    if isinstance(node, sympy.Add):
        return _fold("add", "left", "right", [_walk(a) for a in node.args])

    if isinstance(node, sympy.Mul):
        constants = [a for a in node.args if a.is_number]
        rest = [a for a in node.args if not a.is_number]
        if not rest:
            raise TranslationError("상수 곱만 남았다")
        inner = _fold("multiply", "left", "right", [_walk(a) for a in rest])
        if not constants:
            return inner
        # 상수 배율의 **크기**는 분위 임계 아래에서 뜻이 없다.
        #   q 분위(c·u) = c · q 분위(u)  (c > 0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)]
        # 따라서 크기는 버리고 **부호만** 남긴다. 부호는 부등호 방향을 바꾼다.
        if sympy.Mul(*constants).is_negative:
            return {"op": "negate", "input": inner}
        return inner

    if isinstance(node, sympy.Pow):
        if node.exp == sympy.Rational(1, 2):
            return {"op": "sqrt", "input": _walk(node.base)}
        if node.exp == 2:
            child = _walk(node.base)
            return {"op": "multiply", "left": child, "right": child}
        raise TranslationError(
            f"지원하지 않는 지수: {node.exp}. 역수는 ratio/divide 로 명시해야 "
            "분모 0 정책이 붙는다")

    if isinstance(node, sympy.Abs):
        return {"op": "absolute", "input": _walk(node.args[0])}
    if isinstance(node, sympy.tanh):
        return {"op": "tanh", "input": _walk(node.args[0])}
    if isinstance(node, sympy.log):
        raise TranslationError("log 는 log1p 로 바꿔 쓴다. 인수 양수성이 보장되지 않는다")

    raise TranslationError(f"번역할 수 없는 노드: {type(node).__name__} ({node})")


def _fold(op: str, left_key: str, right_key: str, children: list[dict]) -> dict:
    """이항 연산자를 왼쪽으로 접는다. Catalog 에는 n항 산술이 없다."""
    if not children:
        raise TranslationError(f"{op} 에 인자가 없다")
    folded = children[0]
    for child in children[1:]:
        folded = {"op": op, left_key: folded, right_key: child}
    return folded
