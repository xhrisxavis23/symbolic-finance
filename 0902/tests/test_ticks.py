import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, ticks  # noqa: E402


def test_feature_order_is_sorted_and_covers_catalog():
    config.load_framework()
    from framework import catalog
    assert ticks.FEATURE_ORDER == tuple(sorted(ticks.FEATURE_ORDER))
    assert set(ticks.FEATURE_ORDER) <= set(catalog.FEATURES)
    assert "book_imbalance" in ticks.FEATURE_ORDER
    assert "spread_to_round_trip_cost_ratio" in ticks.FEATURE_ORDER


@pytest.mark.slow
def test_load_arrays_returns_expected_keys():
    arrays = ticks.load_arrays("005930", config.DATE)
    for key in ("bid_price", "ask_price", "bid_qty", "ask_qty", "time_s", "local_time"):
        assert key in arrays
    assert arrays["bid_price"].ndim == 2 and arrays["bid_price"].shape[1] == 10
    assert len(arrays["time_s"]) > 100_000


@pytest.mark.slow
def test_feature_matrix_shape_and_names_agree():
    arrays = ticks.load_arrays("005930", config.DATE)
    matrix, names = ticks.feature_matrix(arrays)
    assert matrix.shape == (len(arrays["time_s"]), len(names))
    assert matrix.dtype == np.float64
    assert names == tuple(sorted(names))
    assert "book_imbalance" in names
    column = matrix[:, names.index("book_imbalance")]
    finite = column[np.isfinite(column)]
    assert np.all((finite >= -1.0) & (finite <= 1.0))
