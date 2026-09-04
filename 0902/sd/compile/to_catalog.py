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


# 나눗셈 → `ratio` 의 분모 0 정책 (사용자 추가 지시).
#
# Catalog 의 `ratio` 연산자는 `zero_policy ∈ {"nan", "reject", "zero"}` 를 요구한다
# (`vendor/framework/catalog.py`). 세 값 다 실행기(`ExpressionRuntime._op_ratio`)에
# 구현돼 있다 — 골라야 한다.
#
#   - "zero"   는 **쓰지 않는다.** DESIGN.md §7 이 이미 정확히 이 모양의 사고를
#     막아 뒀다: `labels.build` 는 CENSORED(관측 불가)를 0 으로 채우지 않고
#     마스크로 제외한다 — "0 으로 채우면 나쁜 거래만 사라진다"(음수로 나와야 할
#     사건이 무해한 0 으로 둔갑해 나쁜 쪽만 눈에 안 띄게 지워진다). 분모가
#     0/결측인 지점을 0 으로 채우면 같은 왜곡이 여기서도 재현된다.
#   - "reject" 는 **쓰지 않는다.** 하루 종일 관측된 실데이터에서 분모(임의의
#     Catalog feature 또는 그 조합)가 정확히 0 이거나 NaN 인 틱이 단 하나만
#     있어도 `ExpressionRuntime.evaluate` 가 예외를 던져 그 진입식 전체가
#     재생 단계에서 죽는다 — PySR 이 어떤 feature 를 분모로 고를지 미리 알 수
#     없으므로 이건 너무 부서지기 쉽다.
#   - "nan" 을 쓴다. 분모가 0/결측인 틱은 NaN 이 되어 하류(마스크·`weighted_r2`·
#     `select.summarise`)가 **이미 하는 대로** 관측 불가로 제외된다 —
#     `labels.build` 가 세운 "마스크로 제외, 0 으로 채우지 않는다" 원칙과
#     정확히 같은 모양이다. `sd/derived.py` 의 기존 파생 열(`ofi_over_qbar_5`
#     등)도 전부 `zero_policy="nan"` 을 쓴다 — 이 선택은 새 관례가 아니라
#     **이미 있는 관례를 따르는 것**이다. 참고로 Catalog 의 `divide` 연산자
#     (`_op_divide`)는 `zero_policy` 필드가 없지만 구현을 보면 "nan" 과
#     바이트 단위로 동일하다 — `ratio` 를 고른 것은 정책을 명시적으로 남기는
#     자리가 있어서다(나중에 재검토할 때 여기만 보면 된다).
RATIO_ZERO_POLICY = "nan"


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

        # 나눗셈 지원 (사용자 추가 지시). sympy 는 `a/b` 를 `Mul(a, Pow(b, -1))` 로
        # 표현한다 — `rest` 의 각 인자 중 `Pow(base, -1)` 꼴은 "분모" 로, 나머지는
        # "분자" 로 가른다. 실측(`sympy.srepr`): `x/(b*c)` 는 `Mul(x, Pow(b,-1),
        # Pow(c,-1))` 로 **분모마다 별도 Pow 노드**가 남는다(하나로 안 뭉친다) —
        # 그래서 분모 인자를 리스트로 모아 뒤에서 `multiply` 로 접는다. 지수가
        # 정확히 `-1` 인 것만 분모로 인정한다 — `-2` 등은 여기서 걸러지지 않고
        # 아래 `_walk(factor)` 를 그대로 타서 `Pow` 분기의 "지원하지 않는 지수"
        # 거부에 그대로 도달한다(기존 동작 유지, 사용자 지시: "-1 만 ratio 로
        # 표현 가능").
        numerator_factors = []
        denominator_bases = []
        for factor in rest:
            if isinstance(factor, sympy.Pow) and factor.exp == -1:
                denominator_bases.append(factor.base)
            else:
                numerator_factors.append(factor)

        if denominator_bases:
            if not numerator_factors:
                # `c/x`(전부 상수/변수 역수, 예: `2/x`, `1/x`, `2/(x*y)`) — 분자가
                # 상수(1 이든 2 든) 뿐이다. Catalog 에는 산술 안에 수치 리터럴을
                # 놓을 노드가 없다(`raw` 는 field 를, `primitive` 는 feature 만
                # 받는다) — 이 파일 전체를 관통하는 바로 그 제약이 여기서도
                # 똑같이 적용된다. 분자에 symbolic factor 가 최소 하나는 있어야
                # (`a/b`, `(a*b)/c` 처럼) `ratio` 로 옮길 수 있다.
                raise TranslationError(
                    f"{node} 의 분자가 상수뿐이다 — Catalog 에는 산술 안에 수치 "
                    "리터럴을 놓을 노드가 없어 분자만 상수인 나눗셈(c/x 꼴)은 "
                    "옮길 수 없다. 분자에 feature 가 최소 하나 있어야 한다")
            numerator_ast = _fold("multiply", "left", "right",
                                 [_walk(factor) for factor in numerator_factors])
            denominator_ast = _fold("multiply", "left", "right",
                                    [_walk(base) for base in denominator_bases])
            inner = {"op": "ratio", "numerator": numerator_ast,
                    "denominator": denominator_ast, "zero_policy": RATIO_ZERO_POLICY}
        else:
            inner = _fold("multiply", "left", "right", [_walk(a) for a in numerator_factors])

        if not constants:
            return inner
        product = sympy.Mul(*constants)
        if not outermost and abs(product) - 1 != 0:
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
            #
            # **R20 (SDD-LEDGER.md).** `abs(product) != 1` 은 `sympy.Float` 에서
            # 깨진다 — sympy 1.14 에서 `Float.__eq__` 는 파이썬 `int` 와 `float`
            # 를 다르게 다룬다: `sympy.Float(1.0) == 1` 은 `False` 이지만
            # `sympy.Float(1.0) == 1.0` 은 `True` 다(직접 실측 확인). `!=` 비교는
            # 그 반대(내부적으로 같은 `__eq__` 경유)라 `abs(sympy.Float(-1.0)) != 1`
            # 이 `True`(틀림)로 나와 부동소수 계수의 순수 부호반전(`-1.0`)을
            # 깊이 상수배로 오탈락시킨다. `product - 1` 뺄셈은 이 타입 불일치를
            # 피해 가고 `== 0` 비교는 Float/Integer 어느 쪽이든 정확하다
            # (`abs(sympy.Float(-1.0)) - 1 != 0` 은 `False` — 맞음). PySR 은
            # 계수를 상수로 낸다 — 이 결함이 고쳐지지 않으면 PySR 이 낸 순수
            # 부호반전 계수(`-1.0*queue_imbalance_best` 같은)가 평범한 뺄셈인데도
            # 전부 오탈락한다.
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
        if node.exp == -1:
            # 홀로 선 역수(`1/x`) — `a/b` 처럼 다른 인자와 곱해져 있으면(`Mul(a,
            # Pow(b,-1))`) 위 `Mul` 분기가 `ratio(a, b)` 로 옮긴다. 하지만
            # sympy 는 `1/x` 를 `Mul` 로 감싸지 않고 `Pow(x, -1)` 를 그대로
            # 최상위(또는 Add 의 항)에 남긴다 — 예: `translate(sa + 1/bi)` 는
            # `Add(sa, Pow(bi,-1))` 이라 이 분기가 직접 불린다. 분자가 상수 `1`
            # 뿐이라 위 Mul 분기의 "분자가 상수뿐이다" 거부와 **같은 이유**로
            # 옮길 수 없다 — Catalog 에는 산술 안에 수치 리터럴을 놓을 노드가
            # 없다.
            raise TranslationError(
                f"{node} — 분자가 상수 1 뿐인 홀로 선 역수는 옮길 수 없다. "
                "다른 feature 와 곱해진 나눗셈(a/b)은 Mul 단계에서 ratio 로 "
                "옮겨지지만, 분자에 symbolic factor 가 없으면(1/x 꼴) 흡수할 "
                "분자 노드가 없다")
        raise TranslationError(
            f"지원하지 않는 지수: {node.exp}. -1 은 나눗셈(ratio)으로, 그 외 음수는 "
            "지원하지 않는다")

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
