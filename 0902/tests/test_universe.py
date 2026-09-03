import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, universe  # noqa: E402


def test_stock_symbols_matches_measured_count():
    symbols = universe.stock_symbols(config.DATE)
    assert len(symbols) == 2570
    assert symbols == tuple(sorted(symbols))
    assert all(len(s) == 6 and s.isdigit() for s in symbols)
    assert "005930" in symbols


def test_etf_codes_are_excluded():
    symbols = set(universe.stock_symbols(config.DATE))
    assert "0000D0" not in symbols          # TIGER ETF
    assert "33637K" not in symbols          # 영문자 포함 우선주 코드


def test_assign_strata_makes_six_balanced_groups():
    stats = pd.DataFrame({
        "n_quote":    [1000] * 12,
        "spread_bps": [5, 6, 7, 8, 20, 21, 22, 23, 50, 51, 52, 53],
        "ask1_depth": [10, 20, 300, 400] * 3,
        "rv20_bps":   [1.0] * 12,
    }, index=[f"{i:06d}" for i in range(12)])
    out = universe.assign_strata(stats)
    assert set(out.stratum.unique()) == set(universe.STRATA)
    assert out.stratum.value_counts().tolist() == [2] * 6


def test_slice_symbols_is_deterministic_and_balanced():
    stats = pd.DataFrame({
        "n_quote":    [1000] * 60,
        "spread_bps": ([5.0] * 20) + ([20.0] * 20) + ([50.0] * 20),
        "ask1_depth": ([10.0] * 10 + [900.0] * 10) * 3,
        "rv20_bps":   [1.0] * 60,
    }, index=[f"{i:06d}" for i in range(60)])
    strata = universe.assign_strata(stats)
    picked = universe.slice_symbols(strata, per_stratum=2)
    assert len(picked) == 12
    assert picked == universe.slice_symbols(strata, per_stratum=2)   # 재현
    assert picked == tuple(sorted(picked))
    counts = strata.loc[list(picked)].stratum.value_counts()
    assert counts.tolist() == [2] * 6


@pytest.mark.slow
def test_real_strata_boundaries_match_design():
    symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(symbols, config.DATE)
    assert len(stats) == 2507                      # 호가틱 200 미만 63개 제외
    strata = universe.assign_strata(stats)
    friction = strata.groupby("friction").size()
    assert friction["L"] == 837 and friction["M"] == 834 and friction["H"] == 836
    lo = strata.loc[strata.friction == "L", "spread_bps"].max()
    assert 18.3 < lo < 18.4


def test_slice_symbols_picks_ascending_front_not_arbitrary_subset():
    """`picked == tuple(sorted(picked))` 단언은 반환값이 항상 재정렬되므로 공허하다 —
    reverse=True 로 바꿔도 그 단언은 여전히 참이다. 실제로 어떤 종목이 뽑혔는지 값
    자체를 단언해 그 구멍을 막는다."""
    stats = pd.DataFrame({
        "n_quote":    [1000] * 60,
        "spread_bps": ([5.0] * 20) + ([20.0] * 20) + ([50.0] * 20),
        "ask1_depth": ([10.0] * 10 + [900.0] * 10) * 3,
        "rv20_bps":   [1.0] * 60,
    }, index=[f"{i:06d}" for i in range(60)])
    strata = universe.assign_strata(stats)
    picked = universe.slice_symbols(strata, per_stratum=2)
    expected = (
        "000000", "000001",   # L-shallow 오름차순 앞 2개
        "000010", "000011",   # L-deep
        "000020", "000021",   # M-shallow
        "000030", "000031",   # M-deep
        "000040", "000041",   # H-shallow
        "000050", "000051",   # H-deep
    )
    assert picked == tuple(sorted(expected))


def test_assign_strata_depth_split_uses_median_not_mean():
    """기존 픽스처의 ask1_depth 는 대칭이라 median<->mean 치환이 걸리지 않는다
    ([10,20,300,400]*3 과 [10]*10+[900]*10 모두 median==해당 분포의 중간값이 mean과
    같은 분할점을 낸다). [1,2,3,4,1000] 처럼 비대칭인 표본으로 그 구멍을 막는다 —
    median=3 이면 앞 3개가 shallow, 뒤 2개가 deep 이지만 mean=202 면 앞 4개가
    shallow, 마지막 1개만 deep 이 된다."""
    stats = pd.DataFrame({
        "n_quote":    [1000] * 15,
        "spread_bps": [5] * 5 + [20] * 5 + [50] * 5,
        "ask1_depth": [1.0, 2.0, 3.0, 4.0, 1000.0] * 3,
        "rv20_bps":   [1.0] * 15,
    }, index=[f"{i:06d}" for i in range(15)])
    out = universe.assign_strata(stats)
    l_group = out.loc[[f"{i:06d}" for i in range(5)]]
    assert l_group.depth.tolist() == ["shallow", "shallow", "shallow", "deep", "deep"]
