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


def test_load_arrays_passes_cache_root_none(monkeypatch):
    """슬라이스는 원본 parquet 을 직접 읽고 캐시를 만들지 않는다 (브리핑 4번,
    DESIGN.md D10). `vendor.framework.data.load` 의 기본 `cache_root` 는
    `BACKTEST_CACHE_ROOT` 이므로, 명시적으로 `None` 을 넘기는지 여기서 고정한다."""
    captured = {}

    def fake_load(symbol, date, root=None, *, cache_root="__UNSET__"):
        captured["cache_root"] = cache_root
        return {"time_s": np.array([0.0])}, Path("/dev/null")

    monkeypatch.setattr(ticks._data, "load", fake_load)
    ticks.load_arrays("005930", config.DATE)
    assert captured["cache_root"] is None


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


@pytest.mark.slow
def test_feature_matrix_drops_trade_dependent_columns_when_no_trades():
    """체결 행이 없는 종목-일에서는 체결 의존 feature 가 0 으로 채워지지 않고 열에서
    빠진다 (브리핑 5번). `292770` (KODEX 국채선물3년인버스, 20260316) 은 체결 행이
    실제로 0인 것을 미리 스캔해 확인한 실 데이터 픽스처다."""
    arrays = ticks.load_arrays("292770", config.DATE)
    assert "buy_volume" not in arrays and "sell_volume" not in arrays

    matrix, names = ticks.feature_matrix(arrays)
    assert matrix.shape == (len(arrays["time_s"]), len(names))
    assert "book_imbalance" in names            # 호가만으로 계산 가능
    assert "vol_flow" not in names              # 체결 의존 — 지어내지 않는다
    assert "trade_event_indicator" not in names  # 체결 의존 — 지어내지 않는다
