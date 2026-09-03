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
