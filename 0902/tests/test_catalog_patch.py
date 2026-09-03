import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402

config.load_framework()
from framework import catalog, contract  # noqa: E402


def test_sqrt_and_tanh_are_registered():
    assert "sqrt" in catalog.OPERATORS
    assert "tanh" in catalog.OPERATORS


def test_sqrt_accepts_dimensionless_input():
    expr = {"op": "sqrt", "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    info = catalog.infer_expression_type(expr)
    assert info.value_type == "numeric"
    assert info.dimension == "dimensionless"


def test_sqrt_rejects_dimensionful_input():
    expr = {"op": "sqrt", "input": {"op": "primitive", "primitive_id": "mid_price"}}
    with pytest.raises(catalog.ExpressionError) as excinfo:
        catalog.infer_expression_type(expr)
    assert "무차원" in str(excinfo.value)


def test_tanh_rejects_dimensionful_input():
    expr = {"op": "tanh", "input": {"op": "primitive", "primitive_id": "bid_depth_total_5"}}
    with pytest.raises(catalog.ExpressionError):
        catalog.infer_expression_type(expr)


def test_sqrt_of_negative_is_nan_not_exception():
    n = 8
    book = _fixture_book(n)
    runtime = contract.ExpressionRuntime(book.data)
    expr = {"op": "sqrt",
            "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    out = runtime.evaluate(expr)
    assert out.shape == (n,)
    assert np.all(np.isnan(out) | (out >= 0.0))


def test_tanh_is_bounded():
    n = 8
    book = _fixture_book(n)
    runtime = contract.ExpressionRuntime(book.data)
    expr = {"op": "tanh",
            "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    out = runtime.evaluate(expr)
    finite = out[np.isfinite(out)]
    assert np.all(np.abs(finite) <= 1.0)


def test_catalog_hash_is_frozen():
    assert catalog.catalog_hash() == config.CATALOG_HASH
    assert catalog.catalog_hash() != "448d4a81d9341432"   # 원본 값에서 바뀌었다


def _fixture_book(n: int):
    """매수 잔량이 매도보다 많다가 뒤집히는 최소 호가창."""
    levels = 10
    bid_price = np.tile(np.arange(1000, 1000 - levels, -1, dtype=float), (n, 1))
    ask_price = np.tile(np.arange(1001, 1001 + levels, dtype=float), (n, 1))
    bid_qty = np.tile(np.full(levels, 100.0), (n, 1))
    ask_qty = np.tile(np.full(levels, 100.0), (n, 1))
    bid_qty[: n // 2, 0] = 300.0        # 앞 절반은 book_imbalance > 0
    ask_qty[n // 2 :, 0] = 300.0        # 뒤 절반은 < 0
    data = {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": bid_qty, "ask_qty": ask_qty,
            "time_s": np.arange(n, dtype=float),
            "local_time": np.arange(n, dtype=np.int64)}
    return catalog.Book(data)
