import sys
from pathlib import Path

import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.compile import normalize  # noqa: E402

a, b = sympy.symbols("book_imbalance ofi_depth_5")


def test_outer_sqrt_is_absorbed():
    stripped, shells = normalize.strip_monotone(sympy.sqrt(sympy.Abs(a)))
    assert stripped == sympy.Abs(a)
    assert shells == ["sqrt"]


def test_outer_tanh_is_absorbed():
    stripped, shells = normalize.strip_monotone(sympy.tanh(a + b))
    assert stripped == a + b
    assert shells == ["tanh"]


def test_positive_scale_and_offset_are_absorbed():
    stripped, shells = normalize.strip_monotone(3 * a + 5)
    assert stripped == a
    assert set(shells) == {"scale", "offset"}


def test_negative_scale_is_not_absorbed():
    """부호가 뒤집히면 부등호 방향이 바뀐다. 임계로 흡수할 수 없다."""
    stripped, shells = normalize.strip_monotone(-3 * a)
    assert stripped == -3 * a
    assert shells == []


def test_inner_sqrt_is_kept():
    expr = sympy.sqrt(sympy.Abs(a)) + b
    stripped, shells = normalize.strip_monotone(expr)
    assert stripped == expr
    assert shells == []


def test_nested_shells_are_stripped_outermost_first():
    stripped, shells = normalize.strip_monotone(sympy.tanh(2 * sympy.sqrt(sympy.Abs(a))))
    assert stripped == sympy.Abs(a)
    assert shells[0] == "tanh"
    assert "sqrt" in shells


def test_monotone_variants_share_one_normal_form():
    assert normalize.normal_form(a) == normalize.normal_form(3 * a + 5)
    assert normalize.normal_form(a) == normalize.normal_form(sympy.tanh(a))
    assert normalize.normal_form(a) != normalize.normal_form(-a)


def test_outer_log_is_not_absorbed():
    """log 는 정의역 일부(u<=0)에서 미정의다. to_catalog.translate 가 log 를
    명시적으로 거부하도록 설계돼 있으므로(계획서 Task 11), strip_monotone 이
    최외곽 log 를 먼저 벗겨 버리면 그 거부 로직이 우회된다 — 벗긴 뒤에는
    u 전체 분포로 분위수를 새로 매기게 되어, 원래 u<=0 에서 자동 배제되던
    행이 다시 선택 후보에 들어온다. 그래서 log 는 벗기지 않는다."""
    stripped, shells = normalize.strip_monotone(sympy.log(a))
    assert stripped == sympy.log(a)
    assert shells == []


def test_log_variant_does_not_share_normal_form():
    assert normalize.normal_form(a) != normalize.normal_form(sympy.log(a))


def test_even_power_is_not_treated_as_sqrt():
    """a**2 는 실수 전체에서 순증가가 아니다(음수 구간에서 감소). Pow 분기가
    지수를 1/2로 제한하지 않으면 부등호 의미가 뒤집힌 채로 흡수된다."""
    stripped, shells = normalize.strip_monotone(a**2)
    assert stripped == a**2
    assert shells == []


def test_outer_atan_is_absorbed():
    """atan 은 전체 실수에서 순증가라 흡수해도 안전하다."""
    stripped, shells = normalize.strip_monotone(sympy.atan(a))
    assert stripped == a
    assert shells == ["atan"]
