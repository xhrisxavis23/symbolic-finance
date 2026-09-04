import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, derived  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402
from framework import contract as _contract  # noqa: E402


def _synthetic_book_arrays(n: int = 260, depth: int = 10, seed: int = 0):
    """호가만으로 이뤄진 최소 합성 데이터. 파생 열은 전부 호가 feature 에서만
    나오므로 체결(buy_volume/sell_volume) 없이도 전부 계산 가능해야 한다."""
    rng = np.random.default_rng(seed)
    tick = 5.0
    base = 10_000.0 + np.cumsum(rng.normal(0.0, 3.0, size=n))
    levels = np.arange(depth)
    bid_price = base[:, None] - tick * (levels[None, :] + 1)
    ask_price = base[:, None] + tick * (levels[None, :] + 1)
    bid_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    ask_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    return {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": bid_qty, "ask_qty": ask_qty}


# ---- 레지스트리 자체 -----------------------------------------------------------

def test_every_derived_column_is_dimensionless():
    for name, ast in derived.DERIVED.items():
        info = catalog.infer_expression_type(ast)
        assert info.dimension == "dimensionless", f"{name} 이 무차원이 아니다: {info}"


def test_derived_names_do_not_collide_with_catalog():
    assert set(derived.DERIVED) & set(catalog.FEATURES) == set()
    assert set(derived.DERIVED) & set(catalog.ALIASES) == set()


def test_derived_names_tuple_is_sorted_and_matches_registry():
    assert derived.DERIVED_NAMES == tuple(sorted(derived.DERIVED))


def test_every_derived_column_has_a_plan_mapping_note():
    for name in derived.DERIVED:
        assert name in derived.PLAN_MAPPING
        plan_item, why = derived.PLAN_MAPPING[name]
        assert plan_item and why


# ---- 등록 검증(_register) 이 실제로 거르는가 -----------------------------------

def test_register_rejects_name_colliding_with_catalog_feature():
    with pytest.raises(ValueError, match="겹친다"):
        derived._register("book_imbalance", derived._primitive("book_imbalance"),
                          plan_item="x", why="y")
    assert "book_imbalance" not in derived.DERIVED


def test_register_rejects_name_colliding_with_catalog_alias():
    # `bid_queue_depletion_5s_bps` 는 `bid_best_price_drop_count_50t` 의 구 별칭이다.
    with pytest.raises(ValueError, match="겹친다"):
        derived._register("bid_queue_depletion_5s_bps",
                          derived._primitive("book_imbalance"),
                          plan_item="x", why="y")


def test_register_rejects_duplicate_derived_name():
    with pytest.raises(ValueError, match="이미 등록"):
        derived._register("ofi_over_qbar_5", derived._primitive("book_imbalance"),
                          plan_item="x", why="y")


def test_register_rejects_a_dimensioned_ast():
    """무차원 검증이 실제로 거르는가 — 차원 있는 AST(quantity)를 등록하려 하면
    실패해야 한다. 이 검사를 지우면(또는 조건을 뒤집으면) 이 테스트가 잡는다."""
    with pytest.raises(ValueError, match="무차원이 아니다"):
        derived._register("__not_dimensionless_probe__",
                          derived._primitive("ofi_depth_5"),  # dimension == "quantity"
                          plan_item="x", why="y")
    assert "__not_dimensionless_probe__" not in derived.DERIVED


def test_register_rejects_a_dimensioned_composite_ast():
    """단일 primitive 뿐 아니라 조립된 AST 도 차원 검사를 통과해야 한다 —
    ratio 의 분자/분모 차원이 다르면 (quantity/price 처럼) dimensionless 가
    아닌 결과가 나온다."""
    bad = derived._ratio(derived._primitive("ofi_depth_5"), derived._primitive("mid_price"))
    with pytest.raises(ValueError, match="무차원이 아니다"):
        derived._register("__not_dimensionless_ratio_probe__", bad,
                          plan_item="x", why="y")


def test_get_returns_a_deep_copy_not_a_shared_reference():
    a = derived.get("ofi_over_qbar_5")
    a["numerator"]["primitive_id"] = "mutated"
    b = derived.get("ofi_over_qbar_5")
    assert b["numerator"]["primitive_id"] == "ofi_depth_5"
    assert derived.DERIVED["ofi_over_qbar_5"]["numerator"]["primitive_id"] == "ofi_depth_5"


# ---- 수치가 정말 등록된 그 AST 를 평가한 값인가 --------------------------------
#
# ExpressionRuntime 을 다시 부르는 것만으로는 "등록된 AST 를 평가했다"는 것만
# 확인할 뿐, "그 AST 가 실제로 원하는 양(OFI/Q̄, ΔP/s)을 계산하는가"는 확인하지
# 못한다. 그래서 여기서는 vendor 의 저수준 함수(depth_ofi·_lagged_return_bps)를
# 직접 불러 **독립적으로** 기대값을 계산하고 비교한다 — numerator/denominator를
# 바꿔치기하거나 depth 를 잘못 골라도 이 비교가 어긋나야 잡힌다.

def test_ofi_over_qbar_5_matches_independently_computed_ratio():
    arrays = _synthetic_book_arrays(n=260, seed=1)
    book = catalog.Book(arrays)
    numerator = catalog.depth_ofi(book, 5)
    qbar = pd.Series(book.bid_qty[:, 0]).rolling(
        window=derived.QBAR_WINDOW, min_periods=derived.QBAR_WINDOW).mean().to_numpy()
    valid = np.isfinite(numerator) & np.isfinite(qbar) & (qbar != 0)
    expected = np.full(book.n, np.nan)
    expected[valid] = numerator[valid] / qbar[valid]

    runtime = _contract.ExpressionRuntime(arrays)
    got = np.asarray(runtime.evaluate(derived.DERIVED["ofi_over_qbar_5"]))

    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got[valid], expected[valid])
    assert valid.sum() > 50, "표본이 너무 적어 이 검사가 사실상 아무것도 확인하지 않는다"


def test_ofi_over_qbar_10_uses_depth_10_not_depth_5():
    """`ofi_over_qbar_5` 와 `ofi_over_qbar_10` 은 서로 다른 값을 내야 한다 —
    둘이 같으면 depth 인자가 실수로 공유된 것이다."""
    arrays = _synthetic_book_arrays(n=260, seed=2)
    runtime = _contract.ExpressionRuntime(arrays)
    five = np.asarray(runtime.evaluate(derived.DERIVED["ofi_over_qbar_5"]))
    ten = np.asarray(runtime.evaluate(derived.DERIVED["ofi_over_qbar_10"]))
    both_finite = np.isfinite(five) & np.isfinite(ten)
    assert both_finite.sum() > 50
    assert not np.allclose(five[both_finite], ten[both_finite])


def test_mid_return_20t_over_spread_matches_independently_computed_ratio():
    arrays = _synthetic_book_arrays(n=260, seed=3)
    book = catalog.Book(arrays)
    mid_return = catalog._lagged_return_bps(book.mid, 20, book.n)
    spread_bps = np.divide((book.ask1 - book.bid1) * 10_000.0, book.mid,
                           out=book.empty(), where=book.mid > 0)
    valid = np.isfinite(mid_return) & np.isfinite(spread_bps) & (spread_bps != 0)
    expected = np.full(book.n, np.nan)
    expected[valid] = mid_return[valid] / spread_bps[valid]

    runtime = _contract.ExpressionRuntime(arrays)
    got = np.asarray(runtime.evaluate(derived.DERIVED["mid_return_20t_over_spread"]))

    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got[valid], expected[valid])


def test_mid_return_5t_and_100t_over_spread_are_not_the_same_column():
    """세 시간 스케일(5t/20t/100t)이 실제로 서로 다른 lag 를 쓰는지 — lag 인자가
    복사-붙여넣기로 굳어버리면 세 열이 전부 같은 값을 낸다."""
    arrays = _synthetic_book_arrays(n=260, seed=4)
    runtime = _contract.ExpressionRuntime(arrays)
    short = np.asarray(runtime.evaluate(derived.DERIVED["mid_return_5t_over_spread"]))
    long = np.asarray(runtime.evaluate(derived.DERIVED["mid_return_100t_over_spread"]))
    both_finite = np.isfinite(short) & np.isfinite(long)
    assert both_finite.sum() > 50
    assert not np.allclose(short[both_finite], long[both_finite])
