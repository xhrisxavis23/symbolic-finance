"""정규형 환원 — 단조 흡수. 계획서 §3 S3.

    entry(x) = [ s(x) > τ ]
    s = g(u), g 가 순증가면  [g(u) > τ] ≡ [u > g⁻¹(τ)]

최외곽 순증가 껍질은 **구조가 아니라 임계**다. 벗겨서 중복을 병합하면
Pareto 후보 수 K 가 줄고, 그만큼 다중 검정 부담(계획서 §3 S4)이 줄어든다.

부호를 뒤집는 변환은 벗기지 않는다 — 부등호 방향이 바뀌기 때문이다.
"""

from __future__ import annotations

import sympy

# 정의역 전체에서 순증가인 단항 **클래스**만. `sympy.sqrt` 는 클래스가 아니라
# 함수라서 `isinstance` 에 넣을 수 없다 — 제곱근은 `Pow(x, 1/2)` 로 잡는다.
MONOTONE_UNARY = {sympy.tanh: "tanh", sympy.atan: "atan", sympy.log: "log"}


def strip_monotone(expr: sympy.Expr) -> tuple[sympy.Expr, list[str]]:
    """최외곽 순증가 껍질을 벗긴다. 바깥에서 안으로 반복한다."""
    shells: list[str] = []
    current = sympy.sympify(expr)
    while True:
        peeled, name = _peel_once(current)
        if name is None:
            return current, shells
        shells.append(name)
        current = peeled


def _peel_once(expr: sympy.Expr) -> tuple[sympy.Expr, str | None]:
    if isinstance(expr, sympy.Add):
        constants = [t for t in expr.args if t.is_number]
        rest = [t for t in expr.args if not t.is_number]
        if constants and rest:
            return sympy.Add(*rest), "offset"
    if isinstance(expr, sympy.Mul):
        constants = [t for t in expr.args if t.is_number]
        rest = [t for t in expr.args if not t.is_number]
        # 양수 상수배만. 음수면 부등호가 뒤집힌다.
        if constants and rest and sympy.Mul(*constants).is_positive:
            return sympy.Mul(*rest), "scale"
    if isinstance(expr, sympy.Pow) and expr.exp == sympy.Rational(1, 2):
        return expr.base, "sqrt"
    for function, name in MONOTONE_UNARY.items():
        if isinstance(expr, function):
            return expr.args[0], name
    return expr, None


def normal_form(expr: sympy.Expr) -> str:
    """중복 병합용 키. 단조 변형끼리는 같은 키를 낸다."""
    stripped, _shells = strip_monotone(expr)
    return sympy.srepr(sympy.simplify(stripped))
