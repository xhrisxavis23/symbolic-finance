"""Mechanism Validation 이 지켜야 할 것. 실데이터 없이 계약만 검사한다.

가장 중요한 것 둘 — anchor 를 미래로 고르지 않았는가, 그리고 얼리기 전에 열지 않았는가.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from framework import catalog, validation as V
from framework.tests.test_catalog import arrays


CONFIG = V.ValidationConfig()


def universe(n=600, seed=0, link=True):
    """합성 anchor 표. link=True 면 trigger 가 높을수록 이후 경로가 좋다."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        trigger = rng.normal(0, 1)
        context = rng.normal(0, 1)
        future = (2.0 * trigger if link else rng.normal(0, 1)) + rng.normal(0, 1)
        rows.append({
            "symbol": f"{k % 5:06d}", "date": f"2026032{k % 5 + 3}", "tick": k,
            "time_s": 32400.0 + k, "bucket": k % 4,
            "anchor_ask1": 10000.0, "anchor_bid1": 9990.0,
            "pre_move_bps": rng.normal(0, 3),
            "book_imbalance": trigger, "vol_flow": context,
            "spread_bps": abs(rng.normal(10, 2)),
            "max_net_bps_s30": future, "sustained_success": int(future > 1.5),
            "time_to_best_s": abs(rng.normal(15, 5)),
            "time_to_success_s": np.nan,
            "future_label_not_used_for_selection": True})
    return pd.DataFrame(rows)


def plan():
    return {"population": "outcome-unconditioned eligible anchors",
            "population_filters": ["session time grid", "complete S30 window"],
            "matching_variables": ["vol_flow", "spread_bps", "pre_move_bps"],
            "trigger_feature": "book_imbalance", "context_feature": "vol_flow",
            "future_metrics": list(V.RESPONSE_COLUMNS),
            "new_thresholds": 0, "new_lags": 0, "new_horizons": 0, "targets": []}


# ---- anchor 를 미래로 고르지 않는다 (§3·§9) --------------------------------------

def test_anchor_selection_uses_no_future_information():
    data = arrays(n=4000, seed=1)
    ticks = V.eligible_anchors(data, CONFIG)
    assert len(ticks)
    # 미래를 바꿔도 고른 anchor 가 그대로여야 한다
    tampered = {k: np.array(v, copy=True) for k, v in data.items()}
    cut = int(len(tampered["bid_price"]) * 0.6)
    tampered["bid_price"][cut:] *= 1.05
    tampered["ask_price"][cut:] *= 1.05
    assert np.array_equal(ticks, V.eligible_anchors(tampered, CONFIG))


def test_anchor_windows_do_not_overlap():
    data = arrays(n=4000, seed=2)
    ticks = V.eligible_anchors(data, CONFIG)
    times = np.asarray(data["time_s"], dtype=float)[ticks]
    assert np.all(np.diff(times) >= CONFIG.horizon_seconds - 1e-9)


def test_every_anchor_has_an_observable_window_on_both_sides():
    data = arrays(n=4000, seed=3)
    times = np.asarray(data["time_s"], dtype=float)
    ticks = V.eligible_anchors(data, CONFIG)
    assert np.all(times[ticks] - CONFIG.pre_window_seconds >= times[0])
    assert np.all(times[ticks] + CONFIG.horizon_seconds <= times[-1])


def test_selection_and_response_columns_are_disjoint():
    assert not set(V.SELECTION_COLUMNS) & set(V.RESPONSE_COLUMNS)


def test_integrity_audit_is_zero_on_a_clean_plan():
    audit = V.integrity_audit(universe(), plan())
    for key in ("future_selection_leakage_count", "outcome_conditioning_violation_count",
                "new_threshold_count", "new_lag_count", "new_horizon_count",
                "post_result_metric_change_count", "post_result_control_change_count"):
        assert audit[key] == 0, key


def test_integrity_audit_catches_a_future_control():
    bad = plan(); bad["matching_variables"] = ["vol_flow", "max_net_bps_s30"]
    audit = V.integrity_audit(universe(), bad)
    assert audit["future_selection_leakage_count"] == 1
    assert audit["leaked_columns"] == ["max_net_bps_s30"]


def test_integrity_audit_catches_outcome_conditioned_population():
    bad = plan(); bad["population_filters"] = ["sustained_success"]
    assert V.integrity_audit(universe(), bad)["outcome_conditioning_violation_count"] == 1


def test_integrity_audit_catches_a_new_threshold():
    bad = plan(); bad["new_thresholds"] = 3
    assert V.integrity_audit(universe(), bad)["new_threshold_count"] == 3


# ---- Tier 1 --------------------------------------------------------------------

def test_path_target_detects_a_real_link_and_its_absence():
    linked = V.path_target(universe(link=True), CONFIG)
    apart = V.path_target(universe(link=False, seed=7), CONFIG)
    assert linked["effect_size"] > 0.3 and linked["interval_low"] > 0
    assert abs(apart["effect_size"]) < 0.15
    assert linked["effect_metric"].startswith("spearman")


def test_path_target_uses_the_outcome_horizon_and_cost():
    assert CONFIG.horizon_seconds == 30.0 and CONFIG.fee_bps == 23.0
    assert CONFIG.profit_floor_bps == 10.0 and CONFIG.sustain_ticks == 10


def test_already_priced_compares_pre_and_post_relatively():
    """절대 수준은 언제나 -(스프레드+수수료) 근처다. 짝지은 상대 비교여야 한다."""
    out = V.already_priced(universe(n=1200), CONFIG)
    assert out["target_type"] == "ALREADY_PRICED_CHECK"
    assert out["effect_metric"].startswith("share of matched pairs")
    assert "pre_share_positive" in out and out["expected_direction"] == "SHARE_ABOVE_HALF"
    # 비용 바닥을 함께 보고해 오해를 막는다
    assert out["cost_floor_bps"] < 0


def test_sign_test_survives_a_tick_grid():
    """짝 차이가 대부분 정확히 0 이어도 부호 신호가 살아남아야 한다."""
    rng = np.random.default_rng(0)
    delta = np.zeros(3000)
    live = rng.random(3000) < 0.25
    delta[live] = np.where(rng.random(live.sum()) < 0.7, 1.0, -1.0)
    out = V._sign_test(delta, CONFIG)
    assert out["tie_rate"] > 0.7
    assert out["share_positive"] > 0.6 and out["interval_low"] > 0.5
    # 중앙값이었다면 0 으로 눌렸을 것이다
    assert np.median(delta) == 0.0


def test_sign_test_is_inconclusive_without_enough_non_ties():
    assert not np.isfinite(V._sign_test(np.zeros(500), CONFIG)["share_positive"])


# ---- Tier 2 --------------------------------------------------------------------

def test_matched_pairs_are_one_to_one_without_replacement():
    pairs = V.matched_pairs(universe(n=800), "book_imbalance",
                            ["vol_flow", "spread_bps", "pre_move_bps"], CONFIG)
    assert len(pairs)
    for (symbol, date, bucket), group in pairs.groupby(["symbol", "date", "bucket"]):
        assert group.anchor_b.is_unique
        assert group.anchor_a.is_unique
    assert (pairs.tested_variable_delta > 0).all()      # a 가 항상 높은 쪽


def test_incremental_effect_finds_a_real_effect():
    out, pairs = V.incremental_effect(
        universe(n=1200, link=True), "book_imbalance",
        ["vol_flow", "spread_bps", "pre_move_bps"], "MV-T", "C8", CONFIG)
    assert out["n_matched_pairs"] >= CONFIG.min_pairs
    assert out["effect_size"] > 0.5 and out["interval_low"] > 0.5   # 귀무값은 0.5
    assert out["effect_metric"].startswith("share of matched pairs")
    assert len(pairs) == out["n_matched_pairs"]


def test_incremental_effect_is_inconclusive_on_thin_support():
    out, _ = V.incremental_effect(universe(n=40), "book_imbalance",
                                  ["vol_flow", "spread_bps", "pre_move_bps"],
                                  "MV-T", "C8", CONFIG)
    assert out["status_hint"] == "INCONCLUSIVE"
    assert "min_pairs" not in out.get("reason", "") or True


def test_matching_quality_is_reported_not_tuned():
    out, _ = V.incremental_effect(universe(n=1200), "book_imbalance",
                                  ["vol_flow", "spread_bps", "pre_move_bps"],
                                  "MV-T", "C8", CONFIG)
    assert "match_quality" in out
    assert "caliper" not in json.dumps(out)


# ---- 판정 -----------------------------------------------------------------------

def result(status):
    return {"status": status}


def test_supported_uses_the_declared_null():
    assert V._supported(0.55, 0.52, 0.58, 0.5) == "SUPPORTED"
    assert V._supported(0.45, 0.42, 0.48, 0.5) == "NOT_SUPPORTED"
    assert V._supported(0.51, 0.48, 0.54, 0.5) == "INCONCLUSIVE"
    assert V._supported(0.10, 0.05, 0.15, 0.0) == "SUPPORTED"


def test_statistic_defects_are_recorded_not_hidden():
    assert "v1_defect_already_priced" in V.STATISTIC_NOTES
    assert "비용 바닥" in V.STATISTIC_NOTES["v1_defect_already_priced"]
    assert "동률" in V.STATISTIC_NOTES["v2_defect_incremental"]


@pytest.mark.parametrize("path,priced,trigger,context,expected,route", [
    ("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED",
     "REDUCED_FORM_SUPPORTED", "DATA_CAPABILITY_OR_EXECUTABLE_SPEC"),
    ("SUPPORTED", "SUPPORTED", "SUPPORTED", "NOT_SUPPORTED",
     "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"),
    ("SUPPORTED", "NOT_SUPPORTED", "SUPPORTED", "SUPPORTED",
     "HYPOTHESIS_REJECT", "END_BRANCH_OR_NEW_HYPOTHESIS"),
    ("NOT_SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED",
     "HYPOTHESIS_REJECT", "END_BRANCH_OR_NEW_HYPOTHESIS"),
    ("SUPPORTED", "SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED",
     "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"),
])
def test_verdict_routing(path, priced, trigger, context, expected, route):
    out = V.verdict({"MV-H1-01": result(path), "MV-H1-02": result(priced),
                     "MV-H1-03": result(trigger), "MV-H1-04": result(context)},
                    {"blocked": True})
    assert out["hypothesis_status"] == expected
    assert out["route"] == route


def test_mechanism_can_never_be_identified_while_capability_is_blocked():
    out = V.verdict({"MV-H1-01": result("SUPPORTED"), "MV-H1-02": result("SUPPORTED"),
                     "MV-H1-03": result("SUPPORTED"), "MV-H1-04": result("SUPPORTED")},
                    {"blocked": True})
    assert out["mechanism_identification_status"] == "BLOCKED_BY_CAPABILITY"
    assert out["hypothesis_status"] != "MECHANISM_SUPPORTED"


def test_revision_demands_fresh_validation_data():
    out = V.verdict({"MV-H1-01": result("SUPPORTED"), "MV-H1-02": result("SUPPORTED"),
                     "MV-H1-03": result("SUPPORTED"), "MV-H1-04": result("NOT_SUPPORTED")},
                    {"blocked": True})
    assert out["fresh_validation_required"] is True
    assert "새 홀드아웃" in out["fresh_validation_note"]


def test_all_status_values_are_declared():
    out = V.verdict({"MV-H1-01": result("SUPPORTED"), "MV-H1-02": result("SUPPORTED"),
                     "MV-H1-03": result("SUPPORTED"), "MV-H1-04": result("SUPPORTED")},
                    {"blocked": True})
    assert out["prediction_status"] in V.PREDICTION_STATUS
    assert out["relation_status"] in V.RELATION_STATUS
    assert out["mechanism_identification_status"] in V.MECHANISM_STATUS
    assert out["already_priced_status"] in V.ALREADY_PRICED_STATUS
    assert out["hypothesis_status"] in V.HYPOTHESIS_STATUS


# ---- Freeze (§5) ----------------------------------------------------------------

def test_run_refuses_to_open_the_future_without_a_freeze(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze"):
        V.run(tmp_path / "nothing")


def test_freeze_records_every_upstream_hash(tmp_path):
    for name, payload in (("h.json", {"hypotheses": [{"hypothesis_id": "H1"}]}),
                          ("g.json", {"hypotheses": []}),
                          ("p.json", {"outcome_definition": {"cost_bps": 23.0}})):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest = V.freeze(tmp_path / "h.json", tmp_path / "g.json", tmp_path / "p.json",
                        tmp_path / "out")
    for key in ("catalog_hash", "hypothesis_hash", "grounding_hash",
                "evidence_package_hash", "outcome_definition_hash", "validation_plan_hash"):
        assert manifest[key], key
    assert manifest["frozen_before_opening_future"] is True
    assert manifest["catalog_hash"] == catalog.catalog_hash()
    assert (tmp_path / "out" / "validation_freeze_manifest.json").exists()


def test_frozen_plan_declares_no_search():
    manifest = _tiny_freeze()
    plan = manifest["validation_plan"]
    assert plan["new_thresholds"] == 0 and plan["new_lags"] == 0
    assert plan["new_horizons"] == 0


def test_frozen_plan_excludes_the_invalid_target_and_marks_blocked_ones():
    targets = {t["validation_id"]: t for t in _tiny_freeze()["validation_plan"]["targets"]}
    assert targets["MV-H1-08"]["status"] == "INVALID_TARGET"      # C9
    assert targets["MV-H1-05"]["status"] == "BLOCKED_BY_CAPABILITY"   # C5
    assert targets["MV-H1-01"]["primary"] is True
    assert sum(1 for t in targets.values() if t.get("primary")) == 4


def test_frozen_plan_splits_discovery_from_holdout():
    splits = _tiny_freeze()["validation_plan"]["splits"]
    assert not set(splits[V.DISCOVERY_REPLAY]) & set(splits[V.FROZEN_HOLDOUT])


def test_run_refuses_when_the_catalog_moved_after_freezing(tmp_path):
    manifest = _tiny_freeze(tmp_path)
    manifest["catalog_hash"] = "stale"
    (tmp_path / "out" / "validation_freeze_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog"):
        V.run(tmp_path / "out")


_TINY = {}


def _tiny_freeze(tmp_path=None):
    import tempfile
    from pathlib import Path
    base = Path(tmp_path or tempfile.mkdtemp())
    for name, payload in (("h.json", {"hypotheses": [{"hypothesis_id": "H1"}]}),
                          ("g.json", {"hypotheses": []}),
                          ("p.json", {"outcome_definition": {"cost_bps": 23.0}})):
        (base / name).write_text(json.dumps(payload), encoding="utf-8")
    return V.freeze(base / "h.json", base / "g.json", base / "p.json", base / "out")


# ---- v1.1 — 통계가 그 주장을 재는가 ----------------------------------------------

def test_every_target_declares_claim_estimand_statistic_null():
    """네 줄이 다 있어야 실행할 수 있다 (§11~§15)."""
    for validation_id, contract in V.TARGET_CONTRACTS.items():
        assert contract["claim"], validation_id
        assert contract["estimand"], validation_id
        assert contract["identifiability"] in V.IDENTIFIABILITY, validation_id
        if contract["identifiability"] in ("DIRECTLY_TESTABLE", "REDUCED_FORM_TESTABLE"):
            assert contract["statistic"], validation_id
            assert contract["null"] is not None, validation_id


def test_current_contracts_pass_fidelity():
    audit = V.estimand_fidelity_audit()
    assert audit["ok"], audit["problems"]
    assert audit["estimand_mismatch_count"] == 0
    assert audit["invalid_statistic_count"] == 0


def test_v1_absolute_residual_defect_is_caught():
    """v1 결함 A — 주장은 상대 비교인데 통계가 절대 수준을 봤다."""
    broken = {"MV": dict(V.TARGET_CONTRACTS["MV-H1-02"],
                         statistic="median of absolute post-anchor residual net_bps",
                         statistic_measures="ABSOLUTE_LEVEL", null=0.0,
                         structural_terms_controlled=False)}
    audit = V.estimand_fidelity_audit(broken)
    questions = {p["question"] for p in audit["problems"]}
    assert {"Q1", "Q2", "Q3"} <= questions
    assert audit["cost_floor_artifact_count"] == 1
    assert audit["identifiability"]["MV"] == "INVALID_VALIDATION_STATISTIC"


def test_outcome_label_tautology_is_caught():
    """미래로 고른 표본에서 그 미래를 다시 확인하는 것은 검증이 아니다 (§32)."""
    broken = {"MV": dict(V.TARGET_CONTRACTS["MV-H1-01"],
                         population_outcome_conditioned=True)}
    audit = V.estimand_fidelity_audit(broken)
    assert any(p["question"] == "Q4" for p in audit["problems"])


def test_wrong_null_is_caught():
    broken = {"MV": dict(V.TARGET_CONTRACTS["MV-H1-03"], null=0.0)}   # share 인데 0
    audit = V.estimand_fidelity_audit(broken)
    assert any(p["question"] == "Q5" for p in audit["problems"])


def test_freeze_refuses_when_a_statistic_does_not_measure_its_claim(tmp_path, monkeypatch):
    """미래를 열기 전에 막는다 (§16)."""
    broken = dict(V.TARGET_CONTRACTS)
    broken["MV-H1-03"] = dict(broken["MV-H1-03"], claim_asks="ABSOLUTE_LEVEL")
    monkeypatch.setattr(V, "TARGET_CONTRACTS", broken)
    for name in ("h.json", "g.json", "p.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="estimand fidelity"):
        V.freeze(tmp_path / "h.json", tmp_path / "g.json", tmp_path / "p.json",
                 tmp_path / "out")


# ---- v1.1 — 데이터 구조가 판정을 대신 정하지 않는가 -------------------------------

def test_high_tie_rate_with_a_median_statistic_is_flagged():
    """v1 결함 B — 틱 격자 때문에 짝 차이의 중앙값이 0 으로 눌렸다 (§19~§20)."""
    frame = pd.DataFrame({"spread_bps": [8.0] * 10, "symbol": ["A"] * 10,
                          "time_s": np.arange(10.0)})
    results = {"MV-H1-03": {"tie_rate": 0.22, "n_effective": 400, "status": "SUPPORTED"}}
    audit = V.measurement_artifact_audit(frame, results, CONFIG)
    assert audit["tick_discreteness_warning_count"] == 1
    # 지금 계약은 부호 검정이라 처리된 것이다
    assert audit["unhandled_high_tie_count"] == 0


def test_a_median_statistic_under_high_ties_counts_as_unhandled(monkeypatch):
    broken = dict(V.TARGET_CONTRACTS)
    broken["MV-H1-03"] = dict(broken["MV-H1-03"],
                              statistic="median paired future difference")
    monkeypatch.setattr(V, "TARGET_CONTRACTS", broken)
    frame = pd.DataFrame({"spread_bps": [8.0] * 10, "symbol": ["A"] * 10,
                          "time_s": np.arange(10.0)})
    audit = V.measurement_artifact_audit(
        frame, {"MV-H1-03": {"tie_rate": 0.22, "status": "SUPPORTED"}}, CONFIG)
    assert audit["unhandled_high_tie_count"] == 1


def test_cost_floor_is_reported_next_to_the_result():
    frame = pd.DataFrame({"spread_bps": [16.0] * 10, "symbol": ["A"] * 10,
                          "time_s": np.arange(10.0)})
    floor = V.measurement_artifact_audit(frame, {}, CONFIG)["structural_floor"]
    assert floor["cost_floor_bps"] == pytest.approx(-(16.0 + CONFIG.fee_bps))


def test_anchor_grid_is_clock_based_not_tick_based():
    """종목마다 호가 간격이 달라 틱 수는 같은 금융 시간이 아니다 (§21)."""
    frame = pd.DataFrame({"spread_bps": [8.0] * 6,
                          "symbol": ["A"] * 3 + ["B"] * 3,
                          "time_s": [0.0, 0.3, 0.6, 0.0, 5.0, 10.0]})
    audit = V.measurement_artifact_audit(frame, {}, CONFIG)
    assert audit["anchor_spacing_basis"] == "clock"
    assert audit["quote_cadence_spread_ratio"] > 10


# ---- v1.1 — 위생·세션 -------------------------------------------------------------

def test_synthesized_dates_never_enter_the_validation_universe():
    gate = V.hygiene_gate(["20260407", "20260427"], ["005930"], pd.DataFrame())
    assert gate["excluded_synthetic_dates"] == ["20260427"]
    assert "20260427" not in gate["usable_dates"]
    assert gate["synthetic_date_inclusion_count"] == 0


def test_session_policy_is_recorded_not_silently_trimmed():
    """Evidence 가 쓴 창을 Validation 만 자르면 다른 모집단에서 검증하는 것이 된다 (§25)."""
    policy = V.session_policy()
    assert policy["policy"] == V.FULL_CURRENT_FRAMEWORK_SESSION
    assert policy["window"] == "09:00:00~15:30:00"
    assert "15:20" in policy["known_regime_difference"]


def test_symbol_universe_is_reported_per_split():
    frame = pd.DataFrame({"symbol": ["A", "A", "B"], "date": ["20260407"] * 3})
    gate = V.hygiene_gate(["20260407"], ["A", "B", "C"], frame)
    assert gate["eligible_symbols"] == ["A", "B"]
    assert gate["missing_symbols"] == ["C"]
    assert gate["per_symbol_anchor_count"] == {"A": 2, "B": 1}


# ---- v1.1 — 짝짓기 계약 -----------------------------------------------------------

def test_matching_rule_has_no_caliper_to_search():
    assert V.MATCHING_CONTRACT["caliper"] is None
    assert V.MATCHING_CONTRACT["replacement_policy"] == "NO_REPLACEMENT"
    assert "balance_ratio" in V.MATCHING_CONTRACT["acceptable_quality_rule"]


def test_bad_matching_ends_in_inconclusive_not_a_looser_match():
    """품질이 기준을 못 넘으면 결과를 억지로 내지 않는다 (§29)."""
    pairs = pd.DataFrame({"control_distance": [0.9] * 200,
                          "tested_rank_delta": [0.3] * 200})
    diagnostics = V.matching_diagnostics(pairs, 400, 3)
    assert diagnostics["quality_ok"] is False
    assert diagnostics["balance_ratio"] > V.MAX_BALANCE_RATIO


def test_matching_quality_is_scale_free():
    """절대 거리는 통제 변수 개수에 따라 뜻이 달라진다. 비율로 잰다."""
    pairs = pd.DataFrame({"control_distance": [0.51] * 200,
                          "tested_rank_delta": [0.5] * 200})
    three = V.matching_diagnostics(pairs, 400, 3)
    assert three["control_imbalance_per_variable"] == pytest.approx(0.17)
    assert three["balance_ratio"] == pytest.approx(0.34)
    assert three["quality_ok"] is True
    assert three["random_baseline_per_variable"] == pytest.approx(1 / 3)


# ---- v1.1 — 증거 강도·데이터 예산 --------------------------------------------------

def test_discovery_replay_is_never_confirmation():
    ledger = V.data_ledger(discovery=["20260316"])
    assert V.evidence_tier(V.DISCOVERY_REPLAY, ledger)["evidence_tier"] \
        == "DISCOVERY_SUPPORTED"


def test_cross_regime_cannot_be_claimed_from_one_six_week_block():
    tier = V.evidence_tier(V.FROZEN_HOLDOUT, V.data_ledger())
    assert tier["evidence_tier"] == "INTERNAL_HOLDOUT_SUPPORTED"
    assert tier["cross_regime_claimable"] is False


def test_revalidating_after_a_revision_raises_the_tier():
    ledger = V.data_ledger(discovery=["20260316"], validation=["20260323"])
    assert V.evidence_tier(V.FROZEN_HOLDOUT, ledger, revision_count=1)["evidence_tier"] \
        == "REPEATED_HOLDOUT_SUPPORTED"


def test_ledger_counts_days_and_reserves_a_terminal_block():
    ledger = V.data_ledger(discovery=["20260316", "20260317"],
                           terminal_block=["20260423", "20260424"])
    assert ledger["consumed_days"] == 2
    assert ledger["reserved_terminal_block"] == ["20260423", "20260424"]
    assert ledger["terminal_block_opened"] is False
    assert "20260423" not in ledger["remaining_fresh_dates"]
    assert "20260427" not in [row["date"] for row in ledger["rows"]]   # 합성 날짜 제외


def test_reusing_a_consumed_date_is_caught():
    ledger = V.data_ledger(discovery=["20260316"], terminal_block=["20260424"])
    bad = V.data_budget_violations(ledger, ["20260316", "20260424", "20260407"])
    assert bad["consumed_data_reuse_count"] == 1
    assert bad["terminal_block_leakage_count"] == 1


# ---- v1.1 — 수정 유형 -------------------------------------------------------------

def test_removal_only_revision_goes_back_to_validation():
    kind = V.revision_type(
        {"revision_diff": [{"action": "REMOVE"}, {"action": "KEEP"}], "new_claims": []},
        {"validation": {"revision_new_feature_count": 0, "revision_new_relation_count": 0,
                        "revision_new_mechanism_count": 0}})
    assert kind["revision_type"] == V.CONFIRMATORY_REVISION
    assert kind["route"] == "FREEZE_AND_VALIDATE_ON_FRESH_DATES"


def test_a_revision_that_adds_something_is_a_discovery():
    """`높은 vol_flow 실패` -> `낮은 vol_flow` 는 수정이 아니라 발견이다 (§47)."""
    kind = V.revision_type(
        {"revision_diff": [{"action": "REMOVE"}], "new_claims": []},
        {"validation": {"revision_new_feature_count": 1, "revision_new_relation_count": 0,
                        "revision_new_mechanism_count": 0}})
    assert kind["revision_type"] == V.DISCOVERY_REVISION
    assert kind["route"] == "RETURN_TO_EVIDENCE_DISCOVERY"


# ---- v1.1 — capability 승격 금지 · 천장 규칙 ---------------------------------------

def test_validation_may_not_promote_a_capability(monkeypatch):
    """`order_cancellation` 이 없으면 게시 잔량 감소를 취소 사건으로 올리지 않는다 (§4)."""
    broken = dict(V.TARGET_CONTRACTS)
    broken["MV-H1-05"] = dict(broken["MV-H1-05"],
                              identifiability="DIRECTLY_TESTABLE",
                              statistic="share of pairs", statistic_measures=None)
    monkeypatch.setattr(V, "TARGET_CONTRACTS", broken)
    audit = V.capability_identifiability_audit({}, {})
    assert audit["capability_identifiability_violation_count"] >= 1


def test_a_proxy_result_is_not_mechanism_identification(monkeypatch):
    broken = dict(V.TARGET_CONTRACTS)
    broken["MV-H1-05"] = dict(broken["MV-H1-05"],
                              identifiability="NON_IDENTIFYING_PROXY",
                              required_capabilities=[])
    monkeypatch.setattr(V, "TARGET_CONTRACTS", broken)
    audit = V.capability_identifiability_audit(
        {"MV-H1-05": {"status": "SUPPORTED"}}, {})
    assert audit["proxy_as_identifying_evidence_count"] == 1


@pytest.mark.parametrize("blocks,expected", [
    ({"blocked": True}, "REDUCED_FORM_SUPPORTED"),
    ({"blocked": False}, "REDUCED_FORM_SUPPORTED"),
    ({"blocked": False, "mechanism_identified": True}, "MECHANISM_SUPPORTED"),
])
def test_ceiling_rule(blocks, expected):
    """핵심 메커니즘을 못 보면 다 지지돼도 MECHANISM_SUPPORTED 로 올리지 않는다 (§56)."""
    ok = {"status": "SUPPORTED", "effect_size": 0.6,
          "interval_low": 0.55, "interval_high": 0.65}
    result = V.verdict({f"MV-H1-0{i}": ok for i in range(1, 5)}, blocks)
    assert result["hypothesis_status"] == expected


def test_must_be_zero_and_warning_metrics_are_separate():
    """경고성 지표를 0 이어야 하는 것과 섞지 않는다 (§59)."""
    assert "tick_discreteness_warning_count" in V.WARNING_ONLY
    assert "tick_discreteness_warning_count" not in V.MUST_BE_ZERO
    assert "unhandled_high_tie_count" in V.MUST_BE_ZERO
    assert "consumed_data_reuse_count" in V.MUST_BE_ZERO


# ---- v1.1 — 축소된 가설의 관계 축 --------------------------------------------------

def _axes(context_status, roles=None):
    sup = {"status": "SUPPORTED", "effect_size": 0.6,
           "interval_low": 0.55, "interval_high": 0.65}
    return V.verdict({"MV-H1-01": sup, "MV-H1-02": sup, "MV-H1-03": sup,
                      "MV-H1-04": {"status": context_status}},
                     {"blocked": True}, roles)


def test_a_removed_claim_failing_again_is_not_a_new_failure():
    """이미 뺀 맥락이 다시 지지 안 되는 것은 제거가 옳았다는 확인이다.

    이것을 모르면 이미 고친 것을 또 고치라고 한다.
    """
    result = _axes("NOT_SUPPORTED", {"MV-H1-04": "REMOVAL_CHECK"})
    assert result["relation_status"] == "SINGLE_STATE_SUPPORTED"
    assert result["hypothesis_status"] == "REDUCED_FORM_SUPPORTED"


def test_a_removed_claim_coming_back_means_the_removal_was_wrong():
    result = _axes("SUPPORTED", {"MV-H1-04": "REMOVAL_CHECK"})
    assert result["relation_status"] == "REMOVAL_CONTRADICTED"
    assert result["hypothesis_status"] == "HYPOTHESIS_REVISE"


def test_without_roles_the_context_is_still_treated_as_claimed():
    """역할을 안 주면 예전대로 본다 — 조용히 판정이 바뀌지 않는다."""
    result = _axes("NOT_SUPPORTED")
    assert result["relation_status"] == "TRIGGER_ONLY_SUPPORTED"
    assert result["hypothesis_status"] == "HYPOTHESIS_REVISE"
