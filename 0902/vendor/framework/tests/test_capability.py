"""연구 환경 프로필이 거짓말하지 않는가.

여기서 보는 것은 하나다 — **못 보는 것을 본다고 적지 않는가.** 프로필이 틀리면
그 위에 쌓는 모든 가설이 같이 틀린다.
"""

from __future__ import annotations

import json

from framework import capability as C, grounding as G


def test_profile_is_deterministic():
    """같은 저장소면 같은 프로필이 나와야 한다. 시각만 다르다."""
    a = C.research_capability_profile()
    b = C.research_capability_profile()
    for key in ("capabilities", "measurement_limits", "dataset_coverage",
                "max_observable_depth", "catalog_features"):
        assert a[key] == b[key]


def test_validation_only_is_not_unavailable():
    """가설 입력에서 가린 것과 데이터에 없는 것은 다르다."""
    assert C.lookup("post_anchor_executable_quotes") == C.VALIDATION_ONLY
    assert C.lookup("participant_identity") == C.UNAVAILABLE
    assert C.lookup("order_add_cancel_decomposition") == C.PROXY_ONLY


def test_grounding_reads_the_same_capability_table():
    """capability 정본은 하나다. grounding 이 따로 들고 있으면 갈라진다."""
    assert G.CAPABILITIES is C.CAPABILITIES


def test_depth_and_cadence_limits_are_stated():
    profile = C.research_capability_profile()
    assert profile["max_observable_depth"] == 10
    assert C.lookup("depth_beyond_l10") == C.UNAVAILABLE
    # 종목 간 호가 간격이 크게 다르다는 사실이 들어 있어야 한다
    ratio = profile["measurement_limits"]["median_quote_interval_seconds"]["spread_ratio"]
    assert ratio > 5
    assert "order-event log" in profile["data_type"]


def test_coverage_is_counted_from_files_not_typed_in():
    coverage = C.research_capability_profile()["dataset_coverage"]
    assert coverage["trading_days"] > 0
    assert coverage["first_date"] < coverage["last_date"]
    assert coverage["regime_diversity"] == "LIMITED"


def test_synthesized_dates_are_not_counted_as_trading_days():
    """합성 날짜를 실거래일로 세면 프로필이 그 자리에서 거짓이 된다 (§9~§10)."""
    from framework import data as tickdata
    assert "20260427" in tickdata.available_dates()          # 폴더는 있다
    coverage = C.dataset_coverage()
    assert coverage["last_date"] != "20260427"
    assert coverage["trading_days"] == len(tickdata.available_dates()) - 1
    # 위생 문제 자체는 Agent 가 보는 프로필에 들어가지 않는다
    profile = C.research_capability_profile()
    assert "20260427" not in json.dumps(profile, ensure_ascii=False)


def test_unknown_capability_is_caught():
    problems = C.validate_requirements([{"concept": "x", "capability": "telepathy",
                                         "availability": "DIRECT"}])
    assert len(problems) == 1 and "telepathy" in problems[0]


def test_claiming_a_missing_capability_as_direct_is_caught():
    problems = C.validate_requirements(
        [{"concept": "주문 주체", "capability": "participant_identity",
          "availability": "DIRECT"}])
    assert len(problems) == 1 and "UNAVAILABLE" in problems[0]


def test_correct_requirement_passes():
    assert C.validate_requirements(
        [{"concept": "호가 비대칭", "capability": "book_imbalance",
          "availability": "DERIVABLE"},
         {"concept": "미래 실행 호가", "capability": "post_anchor_executable_quotes",
          "availability": "VALIDATION_ONLY"}]) == []


def test_consumed_dates_travel_with_the_profile():
    """이미 결과를 본 날짜는 Agent 도 알아야 한다. 모르면 재사용을 제안한다."""
    profile = C.research_capability_profile(["20260316", "20260317"])
    assert profile["consumed_dates"] == ["20260316", "20260317"]


# ---- 실험 환경 ------------------------------------------------------------------

def test_environment_names_the_symbol_group_and_every_symbol():
    """보고서가 "24 종목" 같은 재현 불가능한 기록을 남기지 않게 한다."""
    env = C.experiment_environment(dates={"검증": ["20260407", "20260413"]})
    assert env["symbol_count"] == len(env["symbols"]) == 28
    assert env["symbol_count"] == sum(len(row["used"]) for row in env["clusters"])
    assert len(env["clusters"]) == 7
    assert env["unclustered_symbols"] == []
    assert 0 < env["coverage_ratio"] < 1


def test_environment_is_derived_not_typed_in():
    """종목이 바뀌면 환경 기술도 따라 바뀌어야 한다."""
    env = C.experiment_environment(dates={"검증": ["20260407"]},
                                   symbols=["469170", "017390"])
    assert env["symbol_count"] == 2
    used = [s for row in env["clusters"] for s in row["used"]]
    assert sorted(used) == ["017390", "469170"]
    assert "2종목" in env["known_limits"][0]


def test_environment_states_its_own_limits():
    env = C.experiment_environment(dates={"검증": ["20260407"]})
    limits = " ".join(env["known_limits"])
    assert "같은 28종목" in limits          # 프로필 측정과 검증이 같은 종목
    assert "과소대표" in limits             # 클러스터 크기가 다른데 같은 수를 뽑음
    assert "국면" in limits                 # 한 구간뿐


def test_environment_markdown_has_a_stable_shape():
    """보고서마다 모양이 다르면 비교가 안 된다."""
    text = C.environment_markdown(
        C.experiment_environment(dates={"검증": ["20260407", "20260413"]}))
    assert text.startswith("## 실험 환경")
    for heading in ("| 종목군 |", "| 종목 수 |", "| 기간 · 검증 |", "| 세션 |",
                    "| feature profile |", "### 종목", "### 이 환경의 한계"):
        assert heading in text, heading
    assert "MK01" in text and "MK07" in text
