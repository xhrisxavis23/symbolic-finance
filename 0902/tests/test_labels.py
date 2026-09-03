import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, labels  # noqa: E402


def _ramp_arrays(n=400, drift_bps=50.0):
    """가격이 완만히 오르는 합성 호가창. 30초 뒤 net 이 양수여야 한다."""
    levels = 10
    base = 10_000.0
    step = base * (drift_bps / 1e4) / n
    mid = base + step * np.arange(n)
    bid1 = np.round(mid - 5.0)
    ask1 = np.round(mid + 5.0)
    bid_price = bid1[:, None] - np.arange(levels)[None, :]
    ask_price = ask1[:, None] + np.arange(levels)[None, :]
    qty = np.full((n, levels), 100.0)
    return {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": qty, "ask_qty": qty,
            "time_s": np.arange(n, dtype=float) * 0.5,
            "local_time": (90000000000 + np.arange(n) * 500_000).astype(np.int64)}


def test_normalisation_divides_by_round_trip_friction():
    arrays = _ramp_arrays()
    out = labels.build(arrays)
    idx = np.flatnonzero(out.mask)[0]
    spread_bps = 10.0 / 10_000.0 * 1e4          # 10 KRW / 10,000 KRW
    expected = out.raw_net_bps[idx] / (spread_bps + 23.0)
    assert out.y_path[idx] == pytest.approx(expected, rel=1e-6)


def test_tail_of_day_is_masked_not_zeroed():
    arrays = _ramp_arrays(n=400)
    out = labels.build(arrays)
    assert not out.mask[-1]                      # 30초 창이 데이터 끝을 넘는다
    assert np.isnan(out.y_path[-1])              # 0 으로 채우지 않는다


def test_rising_market_gives_positive_labels_somewhere():
    arrays = _ramp_arrays(n=400, drift_bps=400.0)
    out = labels.build(arrays)
    assert out.mask.sum() > 0
    assert np.nanmax(out.y_path[out.mask]) > 0.0


def test_shapes_and_fill_label_are_binary():
    arrays = _ramp_arrays()
    out = labels.build(arrays)
    n = len(arrays["time_s"])
    assert out.y_path.shape == out.y_fill.shape == out.mask.shape == (n,)
    assert set(np.unique(out.y_fill)) <= {0.0, 1.0}


# --- 이 태스크에서 추가한 테스트: _fill_label 의미 보존 (njit 전환 검증용) ---

def _arrays_with_trades():
    """호가는 평탄, sell_min_price 로 체결 시나리오를 명시적으로 구성한다.

    dt=0.01초로 촘촘하게 잡아 시계 상한(10초=1000틱)이 틱 상한(100틱)보다
    항상 늦게 오도록 만든다 — 그러면 각 i 의 창은 정확히 [i+1, i+101) 이고,
    아래 세 이벤트(각각 100틱 이상 떨어져 있다)의 창이 서로 겹치지 않는다.
    """
    n = 400
    levels = 10
    bid1 = np.full(n, 9_995.0)
    ask1 = np.full(n, 10_005.0)
    bid_price = bid1[:, None] - np.arange(levels)[None, :]
    ask_price = ask1[:, None] + np.arange(levels)[None, :]
    qty = np.full((n, levels), 100.0)
    sell_min = np.full(n, np.nan)
    # i=0 의 창 [1, 101) 안에 9990 <= bid1[0]=9995 → 체결
    sell_min[5] = 9_990.0
    # i=120 의 창 [121, 221) 안 최솟값이 bid1 보다 큼 → 미체결
    sell_min[130] = 10_000.0
    sell_min[140] = 10_001.0
    # i=250 의 창 [251, 351) 안 값이 정확히 bid1 과 같음 → 체결 (등호 포함)
    sell_min[300] = 9_995.0
    arrays = {"bid_price": bid_price, "ask_price": ask_price,
              "bid_qty": qty, "ask_qty": qty,
              "time_s": np.arange(n, dtype=float) * 0.01,
              "local_time": (90000000000 + np.arange(n) * 10_000).astype(np.int64),
              "sell_min_price": sell_min}
    return arrays


def test_fill_label_uses_le_not_lt():
    arrays = _arrays_with_trades()
    out = labels.build(arrays)
    assert out.y_fill[0] == 1.0          # 9990 <= 9995
    assert out.y_fill[120] == 0.0        # 10000, 10001 모두 > 9995
    assert out.y_fill[250] == 1.0        # 9995 <= 9995 (등호 포함)


def test_fill_label_all_zero_when_no_sell_min_price_key():
    arrays = _ramp_arrays()
    assert "sell_min_price" not in arrays
    out = labels.build(arrays)
    assert np.all(out.y_fill == 0.0)


def test_fill_label_horizon_matches_clock_and_tick_bound():
    """window 는 [i+1, last[i]) 이고 last[i] 는 시계·틱 상한 중 더 이른 쪽이다."""
    n = 210
    levels = 10
    bid1 = np.full(n, 9_995.0)
    ask1 = np.full(n, 10_005.0)
    bid_price = bid1[:, None] - np.arange(levels)[None, :]
    ask_price = ask1[:, None] + np.arange(levels)[None, :]
    qty = np.full((n, levels), 100.0)
    # 촘촘한 시간 (0.01초 간격) → 시계 상한(10초)이 틱 상한(100틱)보다 훨씬 뒤에 옴.
    # i=0 에서 틱 상한 last_by_ticks = 0 + 100 + 1 = 101 이 시계 상한보다 작다.
    time_s = np.arange(n, dtype=float) * 0.01
    sell_min = np.full(n, np.nan)
    sell_min[105] = 1.0  # i=0 의 틱 창 [1, 101) 밖 → 체결로 잡히면 안 된다
    arrays = {"bid_price": bid_price, "ask_price": ask_price,
              "bid_qty": qty, "ask_qty": qty,
              "time_s": time_s,
              "local_time": (90000000000 + np.arange(n) * 10_000).astype(np.int64),
              "sell_min_price": sell_min}
    out = labels.build(arrays)
    assert out.y_fill[0] == 0.0


@pytest.mark.slow
def test_fill_label_perf_on_005930():
    """njit 전환 후 005930 (48만 틱) 에서 _fill_label 이 수 초 안에 끝나야 한다."""
    from sd import ticks
    arrays = ticks.load_arrays("005930", config.DATE)
    n = len(arrays["time_s"])
    assert n > 100_000
    # JIT 워밍업 (컴파일 비용 제외하고 스테디스테이트 시간을 재기 위해)
    labels.build({k: v[:1000] if hasattr(v, "__len__") else v for k, v in arrays.items()})
    t0 = time.perf_counter()
    out = labels.build(arrays)
    elapsed = time.perf_counter() - t0
    assert out.y_fill.shape == (n,)
    assert elapsed < 30.0, f"_fill_label 이 너무 느림: {elapsed:.1f}s (n={n})"
