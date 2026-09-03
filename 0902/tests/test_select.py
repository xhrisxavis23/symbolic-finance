import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import select  # noqa: E402


def _ledger(rows):
    return pd.DataFrame(rows)


def _row(contract, status, net=0.0, cohort="NET_RECOVERY"):
    return {"contract_id": contract, "status": status, "net_bps": net,
            "cohort": cohort if status == "FILLED" else None}


def test_summarise_counts_each_status_separately():
    ledger = _ledger([_row("a", "FILLED", 10.0), _row("a", "UNFILLED"),
                      _row("a", "CENSORED")])
    summary = select.summarise(ledger)
    assert summary.loc["a", "decisions"] == 3
    assert summary.loc["a", "fills"] == 1
    assert summary.loc["a", "unfilled"] == 1
    assert summary.loc["a", "censored"] == 1
    assert summary.loc["a", "scorable"] == 2       # CENSORED 는 분모에서 뺀다


def test_censored_is_excluded_from_denominator_not_zeroed():
    ledger = _ledger([_row("a", "FILLED", 20.0)] + [_row("a", "CENSORED")] * 99)
    summary = select.summarise(ledger)
    assert summary.loc["a", "scorable"] == 1
    assert summary.loc["a", "bps_per_decision"] == 20.0


def test_a_tiny_but_profitable_candidate_is_rejected():
    """계획서 §3 S4 의 분모 함정. 결정 3건에 양수인 후보는 발견이 아니다."""
    tiny = [_row("tiny", "FILLED", 500.0)] * 3
    wide = [_row("wide", "FILLED", 1.0)] * 300
    summary = select.summarise(_ledger(tiny + wide))
    ranked = select.rank(summary)
    assert "tiny" not in ranked.index
    assert ranked.index[0] == "wide"


def test_ranking_uses_total_net_not_ratio():
    """비율이 좋아도 총액이 음수면 이기지 못한다."""
    thin = [_row("thin", "FILLED", -1.0)] * 200 + [_row("thin", "UNFILLED")] * 100
    fat = [_row("fat", "FILLED", 2.0)] * 250 + [_row("fat", "UNFILLED")] * 50
    ranked = select.rank(select.summarise(_ledger(thin + fat)))
    assert ranked.index[0] == "fat"
    assert ranked.loc["fat", "total_net_bps"] == 500.0


def test_cohort_counts_are_reported():
    ledger = _ledger([_row("a", "FILLED", -5.0, "PERSISTENT_ADVERSE"),
                      _row("a", "FILLED", -1.0, "COST_INSUFFICIENT"),
                      _row("a", "FILLED", 3.0, "NET_RECOVERY")])
    summary = select.summarise(ledger)
    assert summary.loc["a", "cohort_PERSISTENT_ADVERSE"] == 1
    assert summary.loc["a", "cohort_COST_INSUFFICIENT"] == 1
    assert summary.loc["a", "cohort_NET_RECOVERY"] == 1
    assert summary.loc["a", "cohort_EARLY_RECOVERY_LATE_REVERSAL"] == 0


def test_ranking_prefers_total_net_bps_over_ratio_on_conflict():
    """분모 함정의 핵심: 총액과 비율이 서로 다른 승자를 고를 때 총액이 이겨야 한다.

    큰 후보(A) 는 결정당 비율은 낮지만 총액이 크다. 작은 후보(B) 는 비율은
    높지만 총액이 작다. `bps_per_decision` 을 1순위로 쓰면 B 가 이기는데,
    그것이 바로 이 태스크가 막으려는 실패다.
    """
    big = [_row("A", "FILLED", 0.25)] * 400     # total=100, ratio=0.25
    small = [_row("B", "FILLED", 0.45)] * 200    # total=90,  ratio=0.45
    ranked = select.rank(select.summarise(_ledger(big + small)))
    assert ranked.loc["A", "total_net_bps"] > ranked.loc["B", "total_net_bps"]
    assert ranked.loc["B", "bps_per_decision"] > ranked.loc["A", "bps_per_decision"]
    assert ranked.index[0] == "A"


def test_rank_tiebreak_is_deterministic_by_contract_id():
    """total_net_bps 와 bps_per_decision 이 모두 같으면 contract_id 오름차순."""
    rows = []
    for cid in ("z", "a", "m"):        # 입력 순서를 일부러 섞는다
        rows += [_row(cid, "FILLED", 1.0)] * 200   # 셋 다 total=200, ratio=1.0
    ranked = select.rank(select.summarise(_ledger(rows)))
    assert list(ranked.index) == ["a", "m", "z"]


def test_positive_requires_scorable_and_positive_total_net():
    ledger = _ledger(
        [_row("loser", "FILLED", -1.0)] * 250
        + [_row("winner", "FILLED", 1.0)] * 250
        + [_row("unobservable", "CENSORED")] * 250
    )
    summary = select.summarise(ledger)
    assert summary.loc["loser", "positive"] == False  # noqa: E712
    assert summary.loc["winner", "positive"] == True  # noqa: E712
    assert summary.loc["unobservable", "scorable"] == 0
    assert summary.loc["unobservable", "positive"] == False  # noqa: E712
    assert np.isnan(summary.loc["unobservable", "bps_per_decision"])
