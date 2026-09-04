"""`sd.e0.targets` 단위 테스트. 합성 배열로 인덱스 이동·나눗셈이 정확한지
확인한다 — 실데이터 적재(`ticks.load_arrays`)를 타는 통합 확인은
`tests/test_e0_targets_real_data.py`(slow) 쪽에 둔다."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import targets  # noqa: E402


def _synthetic_arrays(n=400, depth=10, seed=0):
    rng = np.random.default_rng(seed)
    tick = 5.0
    base = 10_000.0 + np.cumsum(rng.normal(0.0, 3.0, size=n))
    levels = np.arange(depth)
    bid_price = base[:, None] - tick * (levels[None, :] + 1)
    ask_price = base[:, None] + tick * (levels[None, :] + 1)
    bid_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    ask_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    time_s = np.arange(n, dtype=float) * 0.5
    buy_volume = rng.integers(0, 50, size=n).astype(float)
    sell_volume = rng.integers(0, 50, size=n).astype(float)
    return {"bid_price": bid_price, "ask_price": ask_price, "bid_qty": bid_qty,
            "ask_qty": ask_qty, "time_s": time_s,
            "local_time": (time_s * 1e6).astype(np.int64),
            "buy_volume": buy_volume, "sell_volume": sell_volume}


def test_forward_shift_moves_values_earlier_by_k():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    shifted = targets._forward_shift(x, 2)
    np.testing.assert_array_equal(shifted[:3], [3.0, 4.0, 5.0])
    assert np.all(np.isnan(shifted[3:]))


def test_forward_shift_k_zero_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(targets._forward_shift(x, 0), x)


def test_forward_shift_k_larger_than_length_is_all_nan():
    x = np.array([1.0, 2.0])
    shifted = targets._forward_shift(x, 5)
    assert shifted.shape == (2,)
    assert np.all(np.isnan(shifted))


def test_l1_target_uses_current_spread_not_future_spread():
    """`y_dimless` 의 분모가 **현재**(t) `spread_bps` 여야 한다 — 미래 시점의
    스프레드로 나누면 S0 의 '마찰은 결정 시점 기준' 철학이 깨진다. 직접
    분자·분모를 재구성해 대조한다."""
    arrays = _synthetic_arrays(n=300, seed=1)
    from sd import ticks
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]
    ret5 = matrix[:, idx["mid_return_5t_bps"]]

    sample = targets.build_l1(arrays)
    y_forward_expected = targets._forward_shift(ret5, targets.L1_FORWARD_TICKS)
    expected = np.where(spread_bps > 0, y_forward_expected / spread_bps, np.nan)
    finite = np.isfinite(sample.y_dimless) & np.isfinite(expected)
    assert finite.sum() > 50
    np.testing.assert_allclose(sample.y_dimless[finite], expected[finite])
    # raw 트랙은 정규화하지 않은 그대로.
    np.testing.assert_allclose(sample.y_raw[np.isfinite(sample.y_raw)],
                               y_forward_expected[np.isfinite(y_forward_expected)])


def test_l1_mask_excludes_rows_without_forward_horizon():
    arrays = _synthetic_arrays(n=50, seed=2)
    sample = targets.build_l1(arrays)
    assert not np.any(sample.mask[-targets.L1_FORWARD_TICKS:])


def test_l1_mask_excludes_zero_spread_rows():
    """실데이터 로더(`framework.data._load_parquet`)는 ASK1>BID1 인 틱만 들여오므로
    `spread_bps>0` 이 실제로는 항상 참이지만, 합성 배열은 그 보장이 없다 —
    분모가 0/음수가 될 수 있는 행을 마스크가 실제로 걷어내는지 직접 확인한다.
    이 가드를 지우면(`spread_bps > 0` 조건 삭제) 이 테스트가 잡는다(뮤테이션
    자기검토 확인, 커밋 전 자기검토 참고)."""
    arrays = _synthetic_arrays(n=100, seed=8)
    zero_spread_row = 40
    arrays["ask_price"][zero_spread_row, 0] = arrays["bid_price"][zero_spread_row, 0]
    sample = targets.build_l1(arrays)
    assert sample.mask[zero_spread_row] == False  # noqa: E712 — 넘파이 bool 명시 비교


def test_l2_target_is_contemporaneous_not_shifted():
    """L2 는 OFI 창과 ΔP 창이 **같은** 과거 구간이어야 한다(동시대 회귀) —
    미래로 이동시키면 안 된다. `mid_return_20t_bps` 원본과 그대로 비교한다."""
    arrays = _synthetic_arrays(n=300, seed=3)
    from sd import ticks
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]
    ret20 = matrix[:, idx["mid_return_20t_bps"]]
    expected = np.where(spread_bps > 0, ret20 / spread_bps, np.nan)

    sample = targets.build_l2(arrays)
    finite = np.isfinite(sample.y_dimless) & np.isfinite(expected)
    assert finite.sum() > 50
    np.testing.assert_allclose(sample.y_dimless[finite], expected[finite])


def test_l2_raw_track_is_ofi_without_qbar_division():
    arrays = _synthetic_arrays(n=300, seed=4)
    sample = targets.build_l2(arrays)
    # raw 는 OFI 절대값(부호 다양)이고, dimless 는 그것을 Q̄ 로 나눈 것이다 —
    # 둘이 같으면(스케일이 우연히 1이면) 판별력이 없으므로 분산 비율로 확인한다.
    raw_col = sample.X_raw[sample.mask, 0]
    dimless_col = sample.X_dimless[sample.mask, 0]
    assert not np.allclose(raw_col, dimless_col)


def test_l3_uses_signed_volume_and_rv_as_two_inputs():
    arrays = _synthetic_arrays(n=300, seed=5)
    sample = targets.build_l3(arrays)
    assert sample.X_dimless.shape[1] == 2
    assert sample.names_dimless == ("signed_volume_20t_over_qbar", "rv5_over_spread")


def test_l4_forward_target_matches_rv20_shifted_by_lookback():
    arrays = _synthetic_arrays(n=400, seed=6)
    from sd import ticks
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]

    sample = targets.build_l4(arrays)
    # 직접 재구성: rv20 을 다시 계산해 LOOKBACK 만큼 앞으로 당긴 값과 일치해야 한다.
    from framework import contract as _contract  # noqa
    rv20 = np.asarray(_contract.ExpressionRuntime(arrays).evaluate(
        targets._realized_vol_ast("mid_return_20t_bps", targets.RV_LOOKBACK)), dtype=float)
    y_forward_expected = targets._forward_shift(rv20, targets.L4_FORWARD_TICKS)
    expected = np.where(spread_bps > 0, y_forward_expected / spread_bps, np.nan)
    finite = np.isfinite(sample.y_dimless) & np.isfinite(expected)
    assert finite.sum() > 20
    np.testing.assert_allclose(sample.y_dimless[finite], expected[finite])


def test_l4_three_scales_are_genuinely_different_columns():
    arrays = _synthetic_arrays(n=400, seed=7)
    sample = targets.build_l4(arrays)
    finite = np.all(np.isfinite(sample.X_dimless), axis=1)
    assert finite.sum() > 20
    rv5, rv20, rv100 = sample.X_dimless[finite].T
    assert not np.allclose(rv5, rv20)
    assert not np.allclose(rv20, rv100)


def test_assemble_skips_failing_symbols_without_crashing(monkeypatch, tmp_path):
    """`assemble()` 은 종목 하나가 실패해도 나머지로 계속 진행해야 한다."""
    from sd import ticks as _ticks

    calls = {"n": 0}

    def fake_load(symbol, date):
        calls["n"] += 1
        if symbol == "BAD":
            raise FileNotFoundError("no such file")
        return _synthetic_arrays(n=200, seed=hash(symbol) % 1000)

    monkeypatch.setattr(_ticks, "load_arrays", fake_load)
    dataset = targets.assemble("L1", ["BAD", "GOOD1", "GOOD2"], "20260316")
    assert dataset.symbols_used == ("GOOD1", "GOOD2")
    assert "BAD" in dataset.symbols_skipped
    assert dataset.X_dimless.shape[0] == dataset.mask.shape[0]


def test_assemble_raises_when_every_symbol_fails(monkeypatch):
    from sd import ticks as _ticks

    def fake_load(symbol, date):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(_ticks, "load_arrays", fake_load)
    try:
        targets.assemble("L1", ["BAD1", "BAD2"], "20260316")
    except ValueError:
        pass
    else:
        raise AssertionError("모든 종목이 실패했는데 예외가 안 났다")
