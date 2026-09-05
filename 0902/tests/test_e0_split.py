"""`sd.e0.split.split_symbols` — 종목 단위 표본외 분할 (PREREG-E0-V2.md §1-1).

세 가지를 확인한다: (1) 적합/선택 집합이 항상 서로소다, (2) 층화 구조가
보존된다(층마다 비율이 유지된다 — 전체를 한 번에 섞지 않는다), (3) `seed` 로
결정론적이다(같은 seed → 같은 분할, 다른 seed → 다른 분할)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import split  # noqa: E402

STRATA6 = ("L-shallow", "L-deep", "M-shallow", "M-deep", "H-shallow", "H-deep")


def _six_stratum_universe(per_stratum: int) -> tuple[tuple[str, ...], dict[str, str]]:
    """`per_stratum` 개씩 6개 층. 종목 코드는 층 인덱스·순번으로 결정론적으로
    만든다(정렬 순서가 코드 순서와 같게 — `slice_symbols` 관례를 흉내낸다)."""
    symbols: list[str] = []
    stratum_of: dict[str, str] = {}
    for s_idx, stratum in enumerate(STRATA6):
        for i in range(per_stratum):
            code = f"{s_idx}{i:05d}"
            symbols.append(code)
            stratum_of[code] = stratum
    return tuple(sorted(symbols)), stratum_of


def test_fit_and_select_are_always_disjoint():
    symbols, stratum_of = _six_stratum_universe(6)
    result = split.split_symbols(symbols, stratum_of, seed=0)
    assert set(result.fit) & set(result.select) == set()


def test_fit_and_select_together_cover_every_input_symbol_exactly_once():
    symbols, stratum_of = _six_stratum_universe(6)
    result = split.split_symbols(symbols, stratum_of, seed=0)
    assert sorted(result.fit) + sorted(result.select) != []
    combined = sorted(result.fit) + sorted(result.select)
    assert sorted(combined) == sorted(symbols)
    assert len(combined) == len(set(combined)) == len(symbols)


def test_per_stratum_ratio_is_preserved_not_a_global_shuffle():
    """뮤테이션 자기검토 대상: 분할이 층화를 무시하고 전체를 한 번에 섞으면
    (풀링) 이 테스트가 실패해야 한다 — 층마다 정확한 fit/select 개수를
    직접 확인한다(6개 층 모두 동일 크기이므로 "우연히 맞았다"의 여지가
    없다)."""
    symbols, stratum_of = _six_stratum_universe(6)
    result = split.split_symbols(symbols, stratum_of, seed=0, fit_fraction=2.0 / 3.0)
    assert set(result.per_stratum_counts) == set(STRATA6)
    for stratum in STRATA6:
        counts = result.per_stratum_counts[stratum]
        assert counts == {"fit": 4, "select": 2, "n": 6}
        members = [s for s in symbols if stratum_of[s] == stratum]
        assert sum(1 for s in members if s in result.select) == 2
        assert sum(1 for s in members if s in result.fit) == 4


def test_stratum_with_uneven_sizes_still_splits_each_independently():
    """층마다 크기가 다를 때도 각 층이 **독립으로** 나뉜다는 것을 확인한다 —
    전체를 풀링해서 나눴다면 작은 층이 통째로 한쪽에 쏠릴 수 있다."""
    symbols = tuple(sorted([f"A{i}" for i in range(5)] + [f"B{i}" for i in range(7)]))
    stratum_of = {**{f"A{i}": "small" for i in range(5)}, **{f"B{i}": "big" for i in range(7)}}
    result = split.split_symbols(symbols, stratum_of, seed=0)
    assert result.per_stratum_counts["small"] == {"fit": 3, "select": 2, "n": 5}
    assert result.per_stratum_counts["big"] == {"fit": 5, "select": 2, "n": 7}
    assert any(s.startswith("A") for s in result.select)
    assert any(s.startswith("B") for s in result.select)


def test_stratum_with_a_single_member_goes_entirely_to_fit():
    symbols = ("A0", "A1", "A2", "B0")
    stratum_of = {"A0": "multi", "A1": "multi", "A2": "multi", "B0": "singleton"}
    result = split.split_symbols(symbols, stratum_of, seed=0)
    assert "B0" in result.fit
    assert "B0" not in result.select
    assert result.per_stratum_counts["singleton"] == {"fit": 1, "select": 0, "n": 1}


def test_split_is_deterministic_given_the_same_seed():
    symbols, stratum_of = _six_stratum_universe(6)
    a = split.split_symbols(symbols, stratum_of, seed=7)
    b = split.split_symbols(symbols, stratum_of, seed=7)
    assert a.fit == b.fit
    assert a.select == b.select


def test_different_seeds_usually_give_different_splits():
    symbols, stratum_of = _six_stratum_universe(6)
    a = split.split_symbols(symbols, stratum_of, seed=0)
    b = split.split_symbols(symbols, stratum_of, seed=1)
    assert a.select != b.select


def test_fit_fraction_out_of_range_raises():
    symbols, stratum_of = _six_stratum_universe(6)
    with pytest.raises(ValueError):
        split.split_symbols(symbols, stratum_of, seed=0, fit_fraction=0.0)
    with pytest.raises(ValueError):
        split.split_symbols(symbols, stratum_of, seed=0, fit_fraction=1.0)


def test_symbol_split_rejects_overlap_if_constructed_directly():
    """`SymbolSplit.__post_init__` 자체도 겹침을 거부한다 — `split_symbols` 를
    안 거치고 직접 만들어도(예: 다른 코드가 나중에 재사용할 때) 같은
    불변식을 지킨다."""
    with pytest.raises(ValueError):
        split.SymbolSplit(fit=("A", "B"), select=("B", "C"), stratum_of={},
                          fit_fraction_requested=0.5, seed=0)
