"""sympy 수식을 Catalog AST 로 옮긴다. 계획서 §3 S5 ②.

번역할 수 없는 후보는 **예외로 죽이지 않고 상위에서 기록**한다. 탈락률 자체가
산출물이기 때문이다 (계획서 F7 — 컴파일 성공률 20% 미만이면 중단).
"""

from __future__ import annotations

import sympy

from .. import config, derived as _derived

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402


class TranslationError(ValueError):
    """Catalog 문법으로 옮길 수 없는 수식."""


def translate(expr: sympy.Expr) -> dict:
    """numeric 을 내는 Catalog AST. boolean 으로 감싸는 것은 threshold 단계가 한다.

    진입점이 **최외곽**이다 — `_walk` 에 `outermost=True` 로 들어간다. 최외곽인지
    깊이인지에 따라 상수 곱의 취급이 갈린다 (F1 리뷰, 아래 `_walk` 의 `Mul` 분기).
    """
    return _walk(sympy.sympify(expr), outermost=True)


def _walk(node: sympy.Expr, outermost: bool = False) -> dict:
    if isinstance(node, sympy.Symbol):
        name = _catalog.resolve(str(node))
        if name in _catalog.FEATURES:
            return {"op": "primitive", "primitive_id": name}
        if name in _derived.DERIVED:
            # 파생 레지스트리 폴백 (ROADMAP.md 1단계). Catalog feature 가 항상
            # 우선이다 — 위에서 먼저 찾아본다. 두 레지스트리는 이름이 겹치지
            # 않는다 (`sd/derived.py` 가 import 시점에 그것을 강제한다), 그래도
            # 순서를 Catalog 우선으로 고정해 둔다.
            return _derived.get(name)
        raise TranslationError(f"Catalog 에도 파생 레지스트리에도 없는 feature: {node}")

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
        product = sympy.Mul(*constants)
        if not outermost and abs(product) != 1:
            # q분위(c·u) = c·q분위(u) (c>0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)] 는
            # **최외곽** 부등식에만 적용된다 — 안쪽 상수는 함수의 모양 자체를
            # 바꾼다 (tanh(2·u) 의 포화 스케일은 tanh(u) 와 다르다: tanh(2)=0.964,
            # tanh(1)=0.762). Catalog 에는 산술 안에 수치 리터럴을 놓을 노드가
            # 없다 — `raw` 는 field 를 요구하고 `primitive` 는 feature 만 받는다.
            # 흡수할 곳이 없으므로 조용히 크기를 버리는 대신 번역을 거부한다.
            #
            # |c| == 1 (즉 부호만 있는 -1) 은 예외다 — `negate` 가 `-1·u` 를
            # **손실 없이 정확하게** 표현하므로 깊이든 어디든 항상 안전하다.
            # 이게 없으면 `book_imbalance - queue_imbalance_best`
            # (= `Add(bi, Mul(-1, qi))`, `-qi` 가 Add 의 항이라 깊이) 같은
            # 평범한 뺄셈까지 거부돼 버린다 — 실제로 F1 수정 직후 이 회귀를
            # 기존 산출물 재검증(9/12 파이프라인 회귀 확인)에서 잡았다.
            raise TranslationError(
                f"깊이(최외곽이 아닌 곳)의 상수배 {product} 는 옮길 수 없다: "
                "Catalog 에는 산술 안에 수치 리터럴을 놓을 노드가 없다 "
                f"({product}·{sympy.Mul(*rest) if len(rest) > 1 else rest[0]} 같은 "
                "깊이의 계수는 최외곽 분위 임계로 흡수되지 않는다)")
        # 상수 배율의 **크기**는 (최외곽에서만) 분위 임계 아래에서 뜻이 없다.
        #   q 분위(c·u) = c · q 분위(u)  (c > 0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)]
        # 따라서 크기는 버리고 **부호만** 남긴다. 부호는 부등호 방향을 바꾼다.
        if product.is_negative:
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
