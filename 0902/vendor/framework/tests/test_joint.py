"""공동 증거가 지켜야 할 것. 합성 anchor 로 검사한다.

여기서 보는 것은 **공동 증거의 계약**이다 — anchor 수준으로 붙였는가, 탐색이 끼지
않았는가, 음의 증거를 지웠는가, 그리고 Agent 가 근거 없이 둘을 묶었을 때 잡히는가.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from framework import joint as J
from framework.tests.test_hypothesis import family, package, good_hypothesis, generated


CONFIG = J.JointConfig()


def anchors(n=200, seed=0, link=True, cols=("ofi_depth_5", "vol_flow")):
    """PROFIT 은 두 상태가 함께, LOSS 는 따로 나타나게 만든 합성 anchor."""
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, share in (("PROFIT", 1.0), ("LOSS", 1.0), ("BACKGROUND", 1.0)):
        for k in range(n):
            if cohort == "PROFIT" and link:
                # 두 상태가 같은 anchor 에서 함께 나타난다
                together = rng.random() < 0.75
                a = -abs(rng.normal(2, 0.5)) if together else abs(rng.normal(2, 0.5))
                b = abs(rng.normal(2, 0.5)) if together else -abs(rng.normal(2, 0.5))
            elif cohort == "PROFIT":
                # 각자 PROFIT 에서 치우쳐 있지만 **서로 독립**이다.
                # 동시 발생률은 올라가도 독립 기대 대비 초과는 0 이다.
                a, b = rng.normal(-1.5, 1.0), rng.normal(1.5, 1.0)
            else:
                a, b = rng.normal(0, 1), rng.normal(0, 1)
            row = {"anchor_id": f"{cohort}-{k}", "cohort": cohort,
                   "symbol": f"{k % 6:06d}", "date": f"2026031{k % 5 + 1}",
                   "tick": k, "relative_time_ms": 0, "source_tick": k, "age_ms": 100.0,
                   cols[0]: a, cols[1]: b}
            # 대표가 어느 것으로 뽑히든 열이 있어야 한다
            for extra in ("ofi_depth_10", "ofi_cks_1", "vol_flow", "ofi_depth_5"):
                row.setdefault(extra, a if extra.startswith("ofi") else b)
            rows.append(row)
    return pd.DataFrame(rows)


def two_families():
    return package([
        family("EV_ORDER_FLOW_01", "ORDER_FLOW", "ofi_depth_5", 0.28, "lower_in_PROFIT"),
        family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19,
               "higher_in_PROFIT", pattern="PERSISTENT_HIGH")])


# ---- 단위와 범위 --------------------------------------------------------------

def test_only_pairs_of_family_representatives_are_built():
    pkg = package([
        family("EV_ORDER_FLOW_01", "ORDER_FLOW", "ofi_depth_5", 0.28),
        family("EV_ORDER_FLOW_02", "ORDER_FLOW", "ofi_depth_10", 0.29),
        family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19,
               "higher_in_PROFIT")])
    out = J.build_joint_evidence(pkg, anchors())
    assert len(out["usable_families"]) == 2                     # family 단위
    assert len(out["pairs"]) == 1                               # C(2,2)
    pair = out["pairs"][0]
    assert pair["family_a"] != pair["family_b"]                 # 같은 family 안은 제외
    # 같은 family 의 두 묶음 중 하나만 대표가 된다
    assert out["family_representatives"]["ORDER_FLOW"]["feature"] in ("ofi_depth_5",
                                                                      "ofi_depth_10")


def test_representative_is_not_chosen_by_effect_size():
    weak_but_covered = family("EV_ORDER_FLOW_01", "ORDER_FLOW", "ofi_cks_1", 0.21)
    strong = family("EV_ORDER_FLOW_02", "ORDER_FLOW", "ofi_depth_10", 0.29)
    strong["representative_feature"]["lookback"] = 5            # 더 복잡한 lineage
    pkg = package([weak_but_covered, strong,
                   family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19,
                          "higher_in_PROFIT")])
    out = J.build_joint_evidence(pkg, anchors())
    assert out["family_representatives"]["ORDER_FLOW"]["feature"] == "ofi_cks_1"


def test_joint_order_never_exceeds_two():
    assert J.MAX_JOINT_ORDER == 2
    pkg = package([family(f"EV_F{i}_01", f"FAM{i}", "ofi_depth_5", 0.2) for i in range(4)]
                  + [family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19,
                            "higher_in_PROFIT")])
    out = J.build_joint_evidence(pkg, anchors())
    assert all(len({p["family_a"], p["family_b"]}) == 2 for p in out["pairs"])


# ---- anchor 수준 결합 ---------------------------------------------------------

def test_joint_is_computed_at_anchor_level_not_from_cohort_summaries():
    """cohort 요약끼리 곱하면 anchor 안의 결합 여부를 구분할 수 없다."""
    linked = J.build_joint_evidence(two_families(), anchors(link=True))["pairs"][0]
    apart = J.build_joint_evidence(two_families(), anchors(link=False))["pairs"][0]
    assert linked["joint_state"]["excess_profit"] > apart["joint_state"]["excess_profit"]


def test_expected_from_marginals_is_the_product_of_marginals():
    pair = J.build_joint_evidence(two_families(), anchors())["pairs"][0]
    for cohort in ("PROFIT", "LOSS"):
        rates = pair["cohort_rates"][cohort]
        assert np.isclose(rates["expected_joint"], rates["marginal_a"] * rates["marginal_b"])
        assert np.isclose(rates["joint_excess"],
                          rates["observed_joint"] - rates["expected_joint"])


def test_nan_anchors_are_dropped_not_zero_filled():
    frame = anchors()
    frame.loc[frame.index[:150], "vol_flow"] = np.nan
    pair = J.build_joint_evidence(two_families(), frame)["pairs"][0]
    assert min(pair["coverage"].values()) < 1.0
    # 0 으로 채웠다면 vol_flow 의 상태 비율이 극단으로 튀었을 것이다
    assert 0.0 < pair["cohort_rates"]["PROFIT"]["marginal_b"] < 1.0


def test_pair_coverage_gap_raises_a_warning():
    frame = anchors()
    frame.loc[(frame.cohort == "LOSS"), "vol_flow"] = np.where(
        np.arange((frame.cohort == "LOSS").sum()) % 2, np.nan,
        frame.loc[frame.cohort == "LOSS", "vol_flow"])
    pair = J.build_joint_evidence(two_families(), frame)["pairs"][0]
    assert any("PAIR_MISSINGNESS_CONFOUND" in w for w in pair["warnings"])


# ---- 상태와 임계 --------------------------------------------------------------

def test_state_direction_comes_from_v1_not_from_a_new_search():
    pair = J.build_joint_evidence(two_families(), anchors())["pairs"][0]
    state = pair["joint_state"]
    assert state["direction_source"] == "profiler_v1_profit_vs_loss_direction"
    assert state["threshold_source"] == CONFIG.threshold_source
    order = {p: i for i, p in enumerate([pair["family_a"], pair["family_b"]])}
    assert state["state_a"] == ("LOW" if order else "LOW")     # lower_in_PROFIT -> LOW


def test_threshold_is_the_pooled_median_not_an_optimum():
    frame = anchors()
    pair = J.build_joint_evidence(two_families(), frame)["pairs"][0]
    pooled = float(np.nanmedian(frame[pair["feature_a"]].to_numpy(float)))
    assert np.isclose(pair["joint_state"]["threshold_a"], pooled)


def test_coarse_state_respects_direction():
    values = np.array([-2.0, 0.0, 2.0])
    assert list(J.coarse_state(values, "lower_in_PROFIT", 0.0)) == [True, True, False]
    assert list(J.coarse_state(values, "higher_in_PROFIT", 0.0)) == [False, True, True]
    assert not J.coarse_state(np.array([np.nan]), "lower_in_PROFIT", 0.0)[0]


# ---- 조건부 -------------------------------------------------------------------

def test_conditional_keeps_sample_size_and_both_directions():
    pair = J.build_joint_evidence(two_families(), anchors())["pairs"][0]
    assert set(pair["conditional"]) == {"a_given_b", "b_given_a"}
    for value in pair["conditional"].values():
        assert value["n_profit"] >= 0 and value["n_loss"] >= 0
        assert "retention" in value and "conditional_direction" in value


def test_small_conditional_samples_are_not_an_auc_gate():
    small = anchors(n=12)
    pair = J.build_joint_evidence(two_families(), small)["pairs"][0]
    assert not any("LOW_SUPPORT" in w for w in pair["warnings"])
    assert any(v["conditional_auc"] is not None for v in pair["conditional"].values())


# ---- 분류 --------------------------------------------------------------------

def test_linked_evidence_is_supported_and_independent_evidence_is_not():
    """주변분포만 치우치고 서로 독립이면 동시 발생률이 높아도 공동 구조가 아니다."""
    linked = J.build_joint_evidence(two_families(), anchors(link=True))["pairs"][0]
    apart = J.build_joint_evidence(two_families(), anchors(link=False, seed=3))["pairs"][0]
    assert linked["classification"] in (J.SUPPORTED, J.PARTIAL)
    assert apart["classification"] == J.NOT_SUPPORTED
    # 동시 발생률 자체는 둘 다 높을 수 있다 — 판정은 초과로 한다
    assert linked["joint_state"]["excess_diff"] > apart["joint_state"]["excess_diff"]
    assert abs(apart["joint_state"]["excess_diff"]) < 0.05


def test_raw_joint_rate_alone_never_makes_a_pair_supported():
    apart = J.build_joint_evidence(two_families(), anchors(link=False, seed=3))["pairs"][0]
    assert apart["joint_state"]["rate_diff"] > 0.2        # 동시 발생률 차이는 크지만
    assert apart["classification"] == J.NOT_SUPPORTED     # 공동 구조는 없다


def test_classification_is_one_of_the_three_labels():
    out = J.build_joint_evidence(two_families(), anchors())
    assert all(p["classification"] in J.CLASSIFICATIONS for p in out["pairs"])
    assert sum(out["counts"].values()) == len(out["pairs"])


def test_negative_evidence_is_kept_not_filtered_out():
    out = J.build_joint_evidence(two_families(), anchors(link=False, seed=3))
    assert out["counts"][J.NOT_SUPPORTED] >= 1
    leak = J.search_leakage_audit(out)
    assert leak["pairs_computed"] == leak["pairs_expected"]     # 상위 N 만 남기지 않았다
    assert out["pairs"], "음의 증거도 package 에 남아야 한다"


# ---- 탐색 누출 (§61) -----------------------------------------------------------

def test_search_leakage_audit_is_all_zero():
    out = J.build_joint_evidence(two_families(), anchors())
    leak = J.search_leakage_audit(out)
    for key in ("threshold_grid_search", "top_n_pair_selection",
                "profitability_based_selection", "post_anchor_outcome_used",
                "multiple_horizon_selection", "time_lag_search", "joint_order_above_two"):
        assert leak[key] == 0, key


def test_only_the_anchor_relative_time_is_used():
    frame = pd.concat([anchors(), anchors().assign(relative_time_ms=-5000)])
    out = J.build_joint_evidence(two_families(), frame)
    assert out["relative_time_ms"] == 0
    # -5000ms 행이 섞였다면 표본 수가 두 배가 됐을 것이다
    assert out["pairs"][0]["cohort_rates"]["PROFIT"]["n"] <= 200


# ---- Agent 출력 감사 (§52·§53) -------------------------------------------------

def joint_with(classification, relation=None):
    out = J.build_joint_evidence(two_families(),
                                 anchors(link=(classification == J.SUPPORTED), seed=1))
    out["pairs"][0]["classification"] = classification
    out["pairs"][0]["relation"] = relation or (
        J.COUPLED if classification == J.SUPPORTED else J.UNRELATED)
    return out


def test_unsupported_joint_claim_is_counted():
    report = J.audit_joint_claims(generated(), joint_with(J.NOT_SUPPORTED))
    assert report["unsupported_joint_claim_count"] == 1
    assert report["unsupported_joint_claims"][0]["classification"] == J.NOT_SUPPORTED


def test_supported_joint_claim_is_not_counted():
    report = J.audit_joint_claims(generated(), joint_with(J.SUPPORTED))
    assert report["unsupported_joint_claim_count"] == 0


def test_partial_joint_claim_is_allowed():
    report = J.audit_joint_claims(generated(), joint_with(J.PARTIAL, J.CONTEXT_INDEPENDENT))
    assert report["unsupported_joint_claim_count"] == 0


def test_context_trigger_relation_permits_linking_even_when_coupling_is_not_supported():
    """`NOT_SUPPORTED` 분류는 결합이 없다는 뜻이지 연결 금지가 아니다."""
    report = J.audit_joint_claims(generated(),
                                  joint_with(J.NOT_SUPPORTED, J.CONTEXT_TRIGGER))
    assert report["unsupported_joint_claim_count"] == 0      # 연결은 허용
    assert report["coupling_unsupported_pair_count"] == 1     # 결합은 여전히 미지지


def test_asserting_a_not_supported_pair_as_observed_is_flagged():
    report = J.audit_joint_claims(generated(), joint_with(J.NOT_SUPPORTED))
    assert any("OBSERVED" in p and "NOT_SUPPORTED" in p for p in report["problems"])


def test_unknown_joint_evidence_id_is_flagged():
    bad = good_hypothesis()
    bad["mechanistic_claim"] += " JEV_MADE_UP__NOWHERE 를 근거로 한다."
    report = J.audit_joint_claims(generated([bad]), joint_with(J.SUPPORTED))
    assert any("없는 joint evidence id" in p for p in report["problems"])


# ---- 배선 --------------------------------------------------------------------

def test_build_writes_all_artifacts_and_upgrades_the_package(tmp_path):
    from framework.config import write_json
    source = tmp_path / "evidence"
    source.mkdir()
    write_json(source / "hypothesis_evidence.json", two_families())
    anchors().to_parquet(source / "anchor_evidence.parquet", index=False)

    result = J.build(source, tmp_path / "joint")
    for name in ("joint_evidence.parquet", "conditional_evidence.parquet",
                 "joint_evidence.json", "hypothesis_evidence_v1_1.json",
                 "JOINT_EVIDENCE_VALIDATION.md", "joint_manifest.json"):
        assert (tmp_path / "joint" / name).exists(), name

    upgraded = result["package"]
    assert upgraded["schema"] == "hypothesis_evidence_package.v1_1"
    assert upgraded["evidence_families"] == two_families()["evidence_families"]  # v1 유지
    assert upgraded["joint_evidence"]
    assert "post-anchor path" in upgraded["not_provided"]


def test_digest_carries_relations_and_roles(tmp_path):
    from framework import hypothesis as H
    pkg = dict(two_families())
    pkg["joint_evidence"] = [{"joint_evidence_id": "JEV_A__B", "classification": "SUPPORTED",
                              "family_a": "ORDER_FLOW", "family_b": "TRADE_ACTIVITY",
                              "relation": "CONTEXT_TRIGGER", "role_a": "LATE_TRIGGER",
                              "role_b": "PERSISTENT_CONTEXT",
                              "context_family": "TRADE_ACTIVITY",
                              "trigger_family": "ORDER_FLOW",
                              "ordering_status": "CONTEXT_BEFORE_TRIGGER"}]
    digest = H.evidence_digest(pkg, H.input_audit(pkg))
    assert digest["evidence_relations"][0]["relation"] == "CONTEXT_TRIGGER"
    assert digest["evidence_relations"][0]["context_family"] == "TRADE_ACTIVITY"
    # 관계가 없는 package 는 관계 절을 싣지 않는다
    plain = H.evidence_digest(two_families(), H.input_audit(two_families()))
    assert "evidence_relations" not in plain


def test_grounding_requirement_role_schema_is_accepted():
    """`role` 은 증거에서의 역할, `importance` 가 CORE/SUPPORTING 이다 (§36·§37)."""
    from framework import hypothesis as H
    rich = good_hypothesis()
    rich["grounding_requirements"] = [
        {"concept": "매도 압력 소진", "role": "MECHANISM_INTERPRETATION",
         "importance": "CORE", "observability": "PROXY"},
        {"concept": "지속적 거래 활동", "role": "PERSISTENT_CONTEXT",
         "importance": "SUPPORTING", "observability": "DIRECT"}]
    assert H.validate_output(generated([rich]), package())["problems"] == []
    rich["grounding_requirements"] = [{"concept": "x", "role": "MANDATORY"}]
    assert any("grounding role" in p
               for p in H.validate_output(generated([rich]), package())["problems"])
    rich["grounding_requirements"] = [{"concept": "x", "role": "LATE_TRIGGER",
                                       "importance": "NICE_TO_HAVE"}]
    assert any("grounding importance" in p
               for p in H.validate_output(generated([rich]), package())["problems"])


def test_family_role_comes_from_the_package_role_map():
    """`evidence_families` 는 묶음 단위라 마지막 값이 이기면 틀린다."""
    from framework import hypothesis as H
    pkg = package()
    pkg["evidence_families"].append(
        family("EV_TRADE_ACTIVITY_09", "TRADE_ACTIVITY", "vol_flow", 0.02))
    pkg["evidence_families"][-1]["role"] = "UNRESOLVED"
    pkg["evidence_families"][1]["role"] = "PERSISTENT_CONTEXT"
    pkg["evidence_roles"] = {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT",
                             "ORDER_FLOW": "LATE_TRIGGER"}
    h = good_hypothesis()
    h["evidence_roles"] = {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT"}
    report = H.validate_output(generated([h]), pkg)
    assert not any("역할을" in p for p in report["problems"]), report["problems"]


def test_package_warning_words_are_not_invented_features():
    from framework import hypothesis as H
    pkg = package()
    pkg["global_warnings"] = ["MISSINGNESS_CONFOUND[anchor_alignment]: PROFIT/LOSS gap 0.2"]
    h = good_hypothesis()
    h["evidence_warnings"] = ["anchor_alignment 격차가 있다"]
    assert H.validate_output(generated([h]), pkg)["unknown_feature_count"] == 0


# ---- 관계 판정 (v1.1) ----------------------------------------------------------

def base_pair(classification="NOT_SUPPORTED", retention=(1.0, 1.0)):
    return {"family_a": "TRADE_ACTIVITY", "family_b": "ORDER_FLOW",
            "classification": classification,
            "conditional": {"a_given_b": {"retention": retention[0]},
                            "b_given_a": {"retention": retention[1]}}}


def test_context_and_trigger_roles_make_a_context_trigger_relation():
    out = J.relation_of(base_pair(),
                        {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT", "ORDER_FLOW": "LATE_TRIGGER"},
                        {"TRADE_ACTIVITY": -10000, "ORDER_FLOW": 0})
    assert out["relation"] == J.CONTEXT_TRIGGER
    assert out["context_family"] == "TRADE_ACTIVITY" and out["trigger_family"] == "ORDER_FLOW"
    assert out["ordering_status"] == "CONTEXT_BEFORE_TRIGGER"


def test_two_late_triggers_are_context_independent():
    out = J.relation_of(base_pair(),
                        {"TRADE_ACTIVITY": "LATE_TRIGGER", "ORDER_FLOW": "LATE_TRIGGER"},
                        {"TRADE_ACTIVITY": 0, "ORDER_FLOW": 0})
    assert out["relation"] == J.CONTEXT_INDEPENDENT
    assert out["context_family"] is None


def test_derived_family_makes_the_relation_redundant():
    out = J.relation_of(base_pair(),
                        {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT", "ORDER_FLOW": "REDUNDANT"},
                        {"TRADE_ACTIVITY": -10000, "ORDER_FLOW": 0})
    assert out["relation"] == J.REDUNDANT_RELATION


def test_coupled_only_when_the_pair_is_supported():
    roles = {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT", "ORDER_FLOW": "LATE_TRIGGER"}
    onsets = {"TRADE_ACTIVITY": -10000, "ORDER_FLOW": 0}
    assert J.relation_of(base_pair("SUPPORTED"), roles, onsets)["relation"] == J.COUPLED
    assert J.relation_of(base_pair("NOT_SUPPORTED"), roles, onsets)["relation"] != J.COUPLED


def test_context_after_trigger_is_not_a_context_trigger_relation():
    out = J.relation_of(base_pair(),
                        {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT", "ORDER_FLOW": "LATE_TRIGGER"},
                        {"TRADE_ACTIVITY": 0, "ORDER_FLOW": -10000})
    assert out["ordering_status"] != "CONTEXT_BEFORE_TRIGGER"
    assert out["relation"] != J.CONTEXT_TRIGGER


def test_collapsed_retention_does_not_block_independent_relation():
    out = J.relation_of(base_pair(retention=(0.1, 0.2)),
                        {"TRADE_ACTIVITY": "LATE_TRIGGER", "ORDER_FLOW": "LATE_TRIGGER"},
                        {"TRADE_ACTIVITY": 0, "ORDER_FLOW": 0})
    assert out["relation"] == J.CONTEXT_INDEPENDENT
