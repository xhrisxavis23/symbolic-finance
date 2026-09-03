"""Hypothesis Agent 가 지켜야 할 것. LLM 을 부르지 않고 계약만 검사한다.

여기서 보는 것은 **가설의 계약**이다 — 증거에서 나왔는가, 지어낸 feature 가 없는가,
전략을 가설로 위장하지 않았는가, 그리고 증거가 모자랄 때 만들지 않는가.
"""

from __future__ import annotations

import json

import pytest

from framework import (catalog, hypothesis as H, mechanism_graph as M, profit_target as PT,
                       sample_condition as SC)


def family(evidence_id="EV_ORDER_FLOW_01", fam="ORDER_FLOW", feature="ofi_depth_5",
           strength=0.28, direction="lower_in_PROFIT", date_ratio=1.0, symbol_ratio=0.93,
           pattern="LEVEL_SHIFT", related=()):
    return {
        "evidence_id": evidence_id, "evidence_group": evidence_id[3:], "family": fam,
        "representative_feature": {
            "name": feature, "definition_id": f"{feature}.v1", "dimension": "quantity",
            "unit": "quantity", "time_basis": "tick", "lookback": 1,
            "observable_description": "설명", "economic_interpretation": "해석",
            "invalid_policy": "정책"},
        "related_features": list(related),
        "snapshot_contrast": {
            "profit_vs_loss": {"auc": 0.5 - strength, "direction": direction,
                               "effect_strength": strength, "median_diff": -1.0},
            "profit_vs_background": {"auc": 0.5 - strength, "direction": direction,
                                     "effect_strength": strength, "median_diff": -1.0},
            "loss_vs_background": {"auc": 0.5, "direction": "no_separation",
                                   "effect_strength": 0.0, "median_diff": 0.0}},
        "temporal_profile": {"relative_times_ms": [-10000, -5000, -2000, -1000, 0],
                             "median_profit": [0, 0, 0, 0, -1], "median_loss": [0, 0, 0, 0, 0],
                             "separation_profit_vs_loss": [0.0, 0.0, 0.0, 0.0, -2 * strength],
                             "pattern": pattern},
        "coverage": {"PROFIT": 1.0, "LOSS": 1.0, "BACKGROUND": 1.0},
        "date_stability": {"agreement": 5, "total": 5, "ratio": date_ratio},
        "symbol_stability": {"agreement": 80, "total": 86, "ratio": symbol_ratio},
        "readiness": {"level_evidence_available": True, "temporal_evidence_available": True,
                      "stable_across_dates": True, "stable_across_symbols": True},
        "warnings": []}


def package(families=None, warnings=()):
    return {"schema": "hypothesis_evidence_package.v1", "catalog_sha256": catalog.catalog_hash(),
            "config": {"horizon_key": "S30"},
            "outcome_definition": {"side": "LONG", "entry_price": "ASK1",
                                   "exit_price": "BID1", "cost_bps": 23.0},
            "cohort_summary": {"PROFIT": {"anchors": 632}, "LOSS": {"anchors": 984},
                               "BACKGROUND": {"anchors": 1616}},
            "evidence_families": list(families if families is not None else [
                family(),
                family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19,
                       "higher_in_PROFIT", pattern="PERSISTENT_HIGH")]),
            "global_warnings": list(warnings), "not_provided": []}


def good_hypothesis(hid="H1"):
    return {
        "hypothesis_id": hid, "title": "매도 압력 소진 뒤 재가격",
        "profit_target": PT.canonical_target(),
        "mechanistic_claim": "거래가 활발한 상태에서 매도 주문흐름이 급격히 음으로 기울고, "
                             "그 압력이 일시적이라면 실행 가격이 단기 균형 아래로 밀렸다가 "
                             "되돌아올 수 있다.",
        "evidence_basis": [
            {"evidence_id": "EV_ORDER_FLOW_01", "family": "ORDER_FLOW",
             "representative_feature": "ofi_depth_5",
             "observation": "PROFIT 에서 더 낮고 anchor 직전에만 갈린다", "role": "TRIGGER"},
            {"evidence_id": "EV_TRADE_ACTIVITY_01", "family": "TRADE_ACTIVITY",
             "representative_feature": "vol_flow",
             "observation": "10초 내내 PROFIT 에서 높다", "role": "PERSISTENT_CONTEXT"}],
        "expected_observable_sequence": [
            {"stage": 1, "claim": "거래 활동이 높게 유지된다", "status": "OBSERVED",
             "evidence": ["EV_TRADE_ACTIVITY_01"]},
            {"stage": 2, "claim": "매도측 주문흐름이 급격히 기운다", "status": "OBSERVED",
             "evidence": ["EV_ORDER_FLOW_01"]},
            {"stage": 3, "claim": "추가 매도의 하방 충격이 줄어든다", "status": "TO_BE_TESTED",
             "evidence": []},
            {"stage": 4, "claim": "실행 가능한 가격이 되돌아온다", "status": "PREDICTED",
             "evidence": []}],
        "mechanism_graph": {
            "schema": M.SCHEMA_VERSION,
            "observables": [
                {"node_id": "O_CONTEXT", "evidence_id": "EV_TRADE_ACTIVITY_01",
                 "available_at": "DECISION"},
                {"node_id": "O_TRIGGER", "evidence_id": "EV_ORDER_FLOW_01",
                 "available_at": "PENDING"},
            ],
            "relations": [{"node_id": "R_CONTEXT_TRIGGER", "relation": "CONTEXT_TRIGGER",
                           "input_ids": ["O_CONTEXT", "O_TRIGGER"], "available_at": "DECISION"}],
            "execution": {"entry_state": "DECISION", "pending_state": "PENDING",
                          "entry_action": "BID1_QUEUE",
                          "entry_lifecycle_policy": {"mode": "HOLD_THROUGH",
                                                     "pending_observable_ids": []}},
            "prediction": {"input_ids": ["R_CONTEXT_TRIGGER"], "evaluation_state": "FILLED",
                           "target_type": "PATH_TARGET", "direction": "HIGHER"},
            "catalog_bindings": [],
        },
        "implementation_directions": [
            {"direction_id": "D1", "title": "관측 상태 표현",
             "representation": "LEVEL", "evidence_ids": ["EV_ORDER_FLOW_01"],
             "observation_focus": "anchor 직전 주문흐름 상태",
             "expected_execution_edge": "BID1 대기 체결 뒤 ASK1 경로가 비용을 넘는지 본다"},
        ],
        "source_of_franchise": None,
        "source_of_profit": "관측된 매도 압력이 정보에 기반한 것이 아니라 일시적이라면 가격이 "
                            "단기 균형 아래로 밀린 상태이고, 압력이 끝난 뒤 실행 가능한 호가가 "
                            "부분적으로 되돌아올 여지가 남는다.",
        "already_priced_risk": False,
        "mechanism_tests": [
            {"test_type": "MATCHED_CONTROL", "claim": "같은 매도 충격이라도 매수측 회복이 "
             "없으면 되돌림이 약하다", "control": "같은 충격, 매수측 회복 없음",
             "expected_result": "가설 일치 상태의 미래 실행 경로가 더 낫다",
             "required_evidence": "매수측 회복 대용치"},
            {"test_type": "TEMPORAL_PRECEDENCE", "claim": "회복이 되돌림보다 먼저 온다",
             "control": "동시점 또는 이후에만 나타나는 경우",
             "expected_result": "회복이 앞선다", "required_evidence": "시간 정렬 궤적"}],
        "alternative_explanations": [
            {"explanation": "가격이 멈춘 것은 흡수가 아니라 매도 자체가 사라졌기 때문",
             "why_plausible": "활동이 줄어도 하락이 멈춘다",
             "discriminating_test": "가격 저항 조건부의 이후 매도량"}],
        "observability": [
            {"concept": "매도 압력 소진", "level": "PROXY", "note": "직접 관측 아님"},
            {"concept": "체결 활동", "level": "DIRECT", "note": "체결량으로 직접"}],
        "grounding_requirements": ["매도 압력 소진", "매수측 회복력"],
        "evidence_warnings": [], "confidence": "MEDIUM"}


def generated(hypotheses=None):
    return {"status": H.GENERATED, "no_hypothesis_reason": None,
            "evidence_synthesis": {"TRIGGER": [{"evidence_id": "EV_ORDER_FLOW_01",
                                                "observation": "직전에만 갈린다"}]},
            "hypotheses": list(hypotheses if hypotheses is not None else [good_hypothesis()])}


# ---- PASS 0 ------------------------------------------------------------------

def test_audit_passes_on_two_stable_families():
    audit = H.input_audit(package())
    assert audit["ok"]
    assert sorted(audit["usable_families"]) == ["ORDER_FLOW", "TRADE_ACTIVITY"]


def test_price_path_observations_must_cover_profit_and_loss_and_cite_allowed_paths():
    pkg = package()
    pkg["path_observation"] = {
        "horizon_key": "S30",
        "anchors": [
            {"anchor_id": "P1", "cohort": "PROFIT", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 10},
            {"anchor_id": "L1", "cohort": "LOSS", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 20},
        ],
    }
    pkg["joint_evidence"] = [{
        "joint_evidence_id": "JEV_ORDER_FLOW__TRADE_ACTIVITY",
        "family_a": "ORDER_FLOW", "family_b": "TRADE_ACTIVITY",
        "relation": "CONTEXT_TRIGGER",
    }]
    payload = generated()
    payload["path_observations"] = [
        {"path_id": "P1:S30", "cohort": "PROFIT", "observations": ["ASK1 최고 시점"]},
        {"path_id": "L1:S30", "cohort": "LOSS", "observations": ["ASK1 유지 실패"]},
    ]
    payload["evidence_relation_map"] = [{
        "a": "ORDER_FLOW", "b": "TRADE_ACTIVITY", "relation": "CONTEXT_TRIGGER",
        "basis_evidence": ["JEV_ORDER_FLOW__TRADE_ACTIVITY"],
        "basis_paths": ["P1:S30", "L1:S30"], "observation": "시간 역할이 다르다",
    }]
    payload["hypotheses"][0]["expected_observable_sequence"][1]["evidence"] = [
        "JEV_ORDER_FLOW__TRADE_ACTIVITY"]

    report = H.validate_output(payload, pkg)
    assert report["path_observation_problem_count"] == 0
    assert report["relation_map_problem_count"] == 0


def test_price_path_relation_cannot_upgrade_the_evidence_relation():
    pkg = package()
    pkg["path_observation"] = {
        "horizon_key": "S30",
        "anchors": [
            {"anchor_id": "P1", "cohort": "PROFIT", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 10},
            {"anchor_id": "L1", "cohort": "LOSS", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 20},
        ],
    }
    pkg["joint_evidence"] = [{
        "family_a": "ORDER_FLOW", "family_b": "TRADE_ACTIVITY", "relation": "UNRESOLVED",
    }]
    payload = generated()
    payload["path_observations"] = [
        {"path_id": "P1:S30", "cohort": "PROFIT", "observations": ["관측"]},
        {"path_id": "L1:S30", "cohort": "LOSS", "observations": ["관측"]},
    ]
    payload["evidence_relation_map"] = [{
        "a": "ORDER_FLOW", "b": "TRADE_ACTIVITY", "relation": "COUPLED",
        "basis_evidence": [], "basis_paths": ["P1:S30"], "observation": "관측",
    }]

    report = H.validate_output(payload, pkg)
    assert report["relation_map_problem_count"] == 1
    assert "올렸다" in report["relation_map_problems"][0]


def test_audit_records_weak_contrast_without_blocking_the_agent():
    weak = [family(strength=0.02), family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY",
                                          "vol_flow", 0.03)]
    audit = H.input_audit(package(weak))
    assert audit["ok"]
    assert sorted(audit["usable_evidence"]) == ["EV_ORDER_FLOW_01", "EV_TRADE_ACTIVITY_01"]
    assert audit["rejected_evidence"] == []


def test_optional_strength_filter_still_records_weak_contrast():
    weak = [family(strength=0.02), family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY",
                                          "vol_flow", 0.03)]
    audit = H.input_audit(package(weak), H.AgentConfig(filter_evidence_by_strength=True))
    assert all(item["reasons"] for item in audit["rejected_evidence"])


def test_audit_keeps_one_family_evidence_for_the_agent():
    same = [family(), family("EV_ORDER_FLOW_02", "ORDER_FLOW", "ofi_depth_10")]
    audit = H.input_audit(package(same))
    assert audit["ok"]
    assert audit["usable_families"] == ["ORDER_FLOW"]


def test_audit_records_unstable_evidence_without_blocking_the_agent():
    shaky = [family(date_ratio=0.4), family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY",
                                            "vol_flow", 0.19, symbol_ratio=0.2)]
    audit = H.input_audit(package(shaky))
    assert audit["ok"]
    assert sorted(audit["usable_evidence"]) == ["EV_ORDER_FLOW_01", "EV_TRADE_ACTIVITY_01"]
    assert audit["rejected_evidence"] == []


def test_split_direction_inside_a_family_is_a_note_not_a_block():
    """한 family 의 서로 다른 묶음은 다른 양이다 — 수준과 변화가 반대여도 정합적이다."""
    split = [family("EV_MICROPRICE_01", "MICROPRICE", "microprice_dev_bps", 0.13,
                    "higher_in_PROFIT"),
             family("EV_MICROPRICE_02", "MICROPRICE", "microprice_velocity", 0.14,
                    "lower_in_PROFIT"),
             family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19)]
    audit = H.input_audit(package(split))
    assert audit["ok"]                                   # 막지 않는다
    assert any("MICROPRICE" in n and "다른 양" in n for n in audit["notes"])


def test_primary_and_secondary_disagreement_is_noted():
    item = family()
    item["snapshot_contrast"]["profit_vs_background"] = {
        "auc": 0.75, "direction": "higher_in_PROFIT", "effect_strength": 0.25,
        "median_diff": 1.0}
    audit = H.input_audit(package([item, family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY",
                                                "vol_flow", 0.19)]))
    assert audit["ok"]
    assert any("primary 를 따를 것" in n for n in audit["notes"])


def test_audit_notes_severe_missingness_without_blocking():
    audit = H.input_audit(package(warnings=[
        "MISSINGNESS_CONFOUND[anchor_alignment]: PROFIT/BACKGROUND @-1000ms gap 0.27~0.31"]))
    assert audit["ok"]
    assert audit["severe_missingness_warnings"]


# ---- Agent 가 보는 증거 --------------------------------------------------------

def test_digest_is_family_level_and_keeps_correlated_features_together():
    pkg = package([family(related=["ofi_depth_10"]),
                   family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19)])
    digest = H.evidence_digest(pkg, H.input_audit(pkg))
    flow = next(e for e in digest["evidence"] if e["evidence_id"] == "EV_ORDER_FLOW_01")
    assert flow["correlated_features_same_evidence"] == ["ofi_depth_10"]
    assert flow["tier"] == "PRIMARY"
    # 원시 anchor 값은 주지 않는다
    assert "anchor_evidence" not in json.dumps(digest)


def test_digest_keeps_low_strength_evidence_eligible_when_filter_is_disabled():
    pkg = package([family(), family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.19),
                   family("EV_SPREAD_01", "SPREAD", "spread_bps", 0.02)])
    digest = H.evidence_digest(pkg, H.input_audit(pkg))
    tiers = {e["evidence_id"]: e["tier"] for e in digest["evidence"]}
    assert tiers["EV_SPREAD_01"] == "PRIMARY" and tiers["EV_ORDER_FLOW_01"] == "PRIMARY"


# ---- 결정적 검증 ---------------------------------------------------------------

def test_clean_output_passes():
    assert H.validate_output(generated(), package())["problems"] == []


def test_profit_target_is_required_and_cannot_be_changed():
    missing = good_hypothesis()
    missing.pop("profit_target")
    assert any("profit_target" in problem
               for problem in H.validate_output(generated([missing]), package())["problems"])
    changed = good_hypothesis()
    changed["profit_target"]["explicit_cost_bps_per_fill"] = 0.0
    assert any("profit_target" in problem
               for problem in H.validate_output(generated([changed]), package())["problems"])


def test_feature_profile_executable_sample_condition_must_be_copied_exactly():
    pkg = package()
    pkg["executable_sample_condition"] = SC.m0_price_recovery()
    item = good_hypothesis()
    item["executable_sample_condition"] = SC.m0_price_recovery()
    assert H.validate_output(generated([item]), pkg)["problems"] == []

    changed = good_hypothesis()
    changed["executable_sample_condition"] = {
        **SC.m0_price_recovery(), "direction": "below"}
    assert any("Feature Profile 조건과 다르다" in problem
               for problem in H.validate_output(generated([changed]), pkg)["problems"])


def test_new_generation_requires_a_mechanism_graph_but_legacy_revalidation_can_read_old_output():
    item = dict(good_hypothesis())
    item.pop("mechanism_graph")
    payload = generated([item])
    assert any("mechanism_graph가 없다" in problem
               for problem in H.validate_output(payload, package())["problems"])
    legacy = H.validate_output(payload, package(),
                               config=H.AgentConfig(require_mechanism_graph=False))
    assert legacy["problems"] == []


def test_invented_feature_is_caught():
    bad = good_hypothesis()
    bad["evidence_basis"][0]["representative_feature"] = "downside_impact_per_sell_volume"
    report = H.validate_output(generated([bad]), package())
    assert any("카탈로그에 없는 feature" in p for p in report["problems"])
    assert report["unknown_feature_count"] >= 1


def test_package_vocabulary_is_not_counted_as_an_invented_feature():
    """Agent 가 \"near_miss 는 증거에서 뺐다\" 고 적는 것은 정확한 서술이다."""
    honest = good_hypothesis()
    honest["evidence_warnings"] = ["near_miss 와 mechanism_failure 는 증거에서 제외됐다"]
    pkg = package()
    pkg["not_provided"] = ["future-selected control cohorts (near_miss, mechanism_failure)"]
    report = H.validate_output(generated([honest]), pkg)
    assert report["unknown_feature_count"] == 0, report["unknown_features"]


def test_unknown_evidence_reference_is_caught():
    bad = good_hypothesis()
    bad["evidence_basis"][0]["evidence_id"] = "EV_MADE_UP_09"
    report = H.validate_output(generated([bad]), package())
    assert any("없는 evidence_id" in p for p in report["problems"])
    assert report["evidence_reference_validity"] == 0.0


def test_single_family_hypothesis_is_rejected():
    bad = good_hypothesis()
    bad["evidence_basis"] = bad["evidence_basis"][:1]
    report = H.validate_output(generated([bad]), package())
    assert any("evidence family" in p for p in report["problems"])


def test_pre_anchor_change_direction_requires_stored_change_evidence():
    bad = good_hypothesis()
    bad["implementation_directions"][0]["representation"] = "PRE_ANCHOR_CHANGE"
    report = H.validate_output(generated([bad]), package())
    assert any("PRE_ANCHOR_CHANGE" in problem for problem in report["problems"])


@pytest.mark.parametrize("text", [
    "OFI 가 q20 아래일 때 매수한다",
    "book_imbalance > 0.5 이면 진입",
    "buy when the flow reverses",
    "stop-loss 를 둔다",
])
def test_strategy_disguised_as_hypothesis_is_caught(text):
    bad = good_hypothesis()
    bad["mechanistic_claim"] = text
    report = H.validate_output(generated([bad]), package())
    assert any("구조 필드에" in p for p in report["problems"]), text


def test_absence_of_downstream_results_is_a_note_not_a_failure():
    """\"OOS 결과가 없다\" 는 누출의 반대다. 낱말만 보고 실패시키면 정직한 서술이 죽는다."""
    honest = good_hypothesis()
    honest["source_of_profit"] += " 현재 자료에는 OOS 결과가 없으므로 검증되지 않았다."
    report = H.validate_output(generated([honest]), package())
    assert report["problems"] == []
    assert report["downstream_result_mentions"]
    assert report["downstream_result_mentions"][0]["term"].upper() == "OOS"


def test_missing_falsification_tests_are_caught():
    bad = good_hypothesis()
    bad["mechanism_tests"] = [t for t in bad["mechanism_tests"]
                              if t["test_type"] != "TEMPORAL_PRECEDENCE"]
    report = H.validate_output(generated([bad]), package())
    assert any("TEMPORAL_PRECEDENCE" in p for p in report["problems"])


def test_sequence_without_any_observed_stage_is_caught():
    bad = good_hypothesis()
    for stage in bad["expected_observable_sequence"]:
        stage["status"] = "PREDICTED"
    report = H.validate_output(generated([bad]), package())
    assert any("OBSERVED 단계" in p for p in report["problems"])


def test_too_many_candidates_are_caught():
    many = [dict(good_hypothesis(), hypothesis_id=f"H{i}") for i in range(4)]
    report = H.validate_output(generated(many), package())
    assert any("최대" in p for p in report["problems"])


def test_already_priced_story_is_flagged():
    bad = good_hypothesis()
    for b in bad["evidence_basis"]:
        b["family"] = "PRICE_MOMENTUM"
    bad["evidence_basis"][1]["family"] = "MICROPRICE"
    report = H.validate_output(generated([bad]), package())
    assert report["already_priced_candidates"] == ["H1"]


def test_legacy_empty_result_is_not_a_valid_new_output():
    empty = {"status": H.LEGACY_NO_HYPOTHESIS, "hypotheses": []}
    problems = H.validate_output(empty, package())["problems"]
    assert any("status" in problem for problem in problems)


def test_generated_result_must_carry_at_least_one_hypothesis():
    empty = {"status": H.GENERATED, "hypotheses": []}
    assert any("가설이 없다" in p for p in H.validate_output(empty, package())["problems"])


# ---- 배선 --------------------------------------------------------------------

def test_run_calls_agent_even_when_all_evidence_is_weak(tmp_path):
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return generated()

    weak = package([family(strength=0.01),
                    family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.01)])
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(weak), encoding="utf-8")
    result = H.run(path, tmp_path / "out", agent=spy)
    assert len(calls) == 2                                 # 생성 1 + 증거 감사 1
    assert result["status"] == H.GENERATED
    assert "H1" in (tmp_path / "out" / "hypotheses.md").read_text(encoding="utf-8")


def test_run_does_not_persist_a_legacy_empty_result(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(package()), encoding="utf-8")
    empty = {"status": H.LEGACY_NO_HYPOTHESIS, "hypotheses": []}
    with pytest.raises(ValueError, match="최소 한 개"):
        H.run(path, tmp_path / "out", agent=lambda prompt: empty)
    assert not (tmp_path / "out" / "hypotheses.json").exists()


def test_run_uses_exactly_two_invocations(tmp_path):
    prompts = []

    def spy(prompt):
        prompts.append(prompt)
        return generated() if len(prompts) == 1 else dict(
            generated(), verdicts=[{"hypothesis_id": "H1", "verdict": "RETAIN", "reasons": []}])

    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(package()), encoding="utf-8")
    result = H.run(path, tmp_path / "out", agent=spy)
    assert len(prompts) == 2                               # 생성 1 + 증거 감사 1
    assert "audit it against the evidence" in prompts[1]
    assert "keep at least one exploratory candidate" in prompts[1]
    assert result["status"] == H.GENERATED
    assert result["audit"]["draft_count"] == 1
    assert result["audit"]["validation"]["problems"] == []


def test_run_writes_provenance(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(package()), encoding="utf-8")
    result = H.run(path, tmp_path / "out",
                   agent=lambda p: dict(generated(), verdicts=[]))
    manifest = result["manifest"]
    for key in ("catalog_hash", "evidence_package_hash", "model", "prompt_version",
                "prompt_sha256", "schema_version", "output_sha256", "reasoning_effort"):
        assert manifest[key], key
    assert manifest["catalog_hash"] == catalog.catalog_hash()
    for name in ("hypotheses.json", "hypotheses.md",
                 "hypothesis_generation_audit.json", "hypothesis_generation_manifest.json"):
        assert (tmp_path / "out" / name).exists(), name


def test_prompt_forbids_strategy_and_demands_falsification():
    digest = H.evidence_digest(package(), H.input_audit(package()))
    digest["price_path_tool"] = {"anchors_by_cohort": {"PROFIT": 1, "LOSS": 1}}
    prompt = H.generation_prompt(digest)
    for phrase in ("could become profitable execution candidates for strategy design",
                   "Do not invent executable features", "Do not choose thresholds",
                   "must emit at least one exploratory", "Feature Profile is a discovery hint",
                   "get_pre_anchor_microstructure",
                   "never count them as separate supporting evidence".lower()):
        assert phrase.lower() in prompt.lower(), phrase
    assert H.LEGACY_NO_HYPOTHESIS not in prompt


def test_contract_repair_prompt_contains_the_canonical_schema():
    digest = H.evidence_digest(package(), H.input_audit(package()))
    prompt = H.contract_repair_prompt(
        digest, {"status": "INVALID_OUTPUT", "hypotheses": []}, ["구조 오류"])
    assert '"status": "HYPOTHESES_GENERATED"' in prompt
    assert "SINGLE_MECHANISM" in prompt
    assert '"mechanism_graph"' in prompt
    assert '"validation_feasibility"' in prompt


def test_fixed_contract_fields_are_copied_by_code():
    pkg = package()
    pkg["executable_sample_condition"] = {
        "op": "crossover", "input": {"op": "primitive", "primitive_id": "mid_return_5t_bps"},
        "threshold": 0.0, "direction": "above", "equality": "from_equal",
    }
    payload = generated()
    payload["path_observations"] = [{"status": "NOT_OBSERVED"}]
    payload["hypotheses"][0]["profit_target"] = {"wrong": True}
    payload["hypotheses"][0]["executable_sample_condition"] = None
    fixed = H.apply_fixed_contracts(payload, pkg, path_observations_available=False)
    assert fixed["path_observations"] == []
    assert fixed["hypotheses"][0]["profit_target"] == PT.canonical_target()
    assert fixed["hypotheses"][0]["executable_sample_condition"] == pkg[
        "executable_sample_condition"]


def test_payload_parsing_survives_fences_and_prose():
    payload = H.parse_payload('설명입니다\n```json\n{"status": "HYPOTHESES_GENERATED"}\n```\n')
    assert payload["status"] == "HYPOTHESES_GENERATED"
    assert H.parse_payload('{"status": "HYPOTHESES_GENERATED"}')["status"] == "HYPOTHESES_GENERATED"


# ---- v1.2 데이터 인식 (§51) -----------------------------------------------------

def aware(hid="H1", **over):
    """capability 를 정확히 적은 가설."""
    item = {
        "hypothesis_id": hid,
        "capability_requirements": [
            {"concept": "호가 비대칭", "capability": "book_imbalance",
             "availability": "DERIVABLE"},
            {"concept": "미래 실행 호가", "capability": "post_anchor_executable_quotes",
             "availability": "VALIDATION_ONLY"},
            {"concept": "호가 갱신 비동기", "capability": "order_add_cancel_decomposition",
             "availability": "PROXY_ONLY"}],
        "observability": [{"concept": "호가 비대칭", "level": "DIRECT", "note": ""},
                          {"concept": "호가 갱신 비동기", "level": "WEAK_PROXY", "note": ""}],
        "expected_observable_sequence": [
            {"stage": 1, "claim": "호가 비대칭 이 높다", "status": "OBSERVED"},
            {"stage": 2, "claim": "미래 실행 호가 가 개선된다", "status": "PREDICTED"}],
        "data_fit": {"observable_core": "SUPPORTED",
                     "mechanism_observability": "UNRESOLVED",
                     "prediction_testability": "AVAILABLE",
                     "overall": "REDUCED_FORM_TESTABLE"},
        "validation_feasibility": "READY"}
    item.update(over)
    return item


def test_reduced_form_testable_is_a_normal_hypothesis():
    """메커니즘을 직접 못 봐도 예측을 시험할 수 있으면 정상이다 (§21)."""
    result = H.data_fit_checks([aware()])
    assert result["capability_violation_count"] == 0
    assert result["unobservable_as_direct_count"] == 0
    assert result["validation_only_leakage_count"] == 0
    assert result["data_fit_problem_count"] == 0
    assert result["data_fit_distribution"] == {"REDUCED_FORM_TESTABLE": 1}


def test_unavailable_concept_claimed_as_direct_is_caught():
    bad = aware(capability_requirements=[
        {"concept": "주문 주체", "capability": "participant_identity",
         "availability": "DIRECT"}],
        observability=[{"concept": "주문 주체", "level": "DIRECT", "note": ""}])
    result = H.data_fit_checks([bad])
    assert result["capability_violation_count"] == 1
    assert result["unobservable_as_direct_count"] == 1


def test_future_data_stated_as_observed_is_caught():
    """VALIDATION_ONLY 는 예측 대상이지 관측한 것이 아니다 (§8)."""
    bad = aware(expected_observable_sequence=[
        {"stage": 1, "claim": "미래 실행 호가 가 이미 개선된 것을 보았다",
         "status": "OBSERVED"}])
    assert H.data_fit_checks([bad])["validation_only_leakage_count"] == 1


def test_overall_fit_cannot_be_inflated():
    """메커니즘이 UNRESOLVED 면 FULLY_TESTABLE 이 될 수 없다."""
    bad = aware(data_fit={"observable_core": "SUPPORTED",
                          "mechanism_observability": "UNRESOLVED",
                          "prediction_testability": "AVAILABLE",
                          "overall": "FULLY_TESTABLE"})
    problems = H.data_fit_checks([bad])["data_fit_problems"]
    assert len(problems) == 1 and "REDUCED_FORM_TESTABLE" in problems[0]["problem"]


def test_blocked_prediction_makes_the_hypothesis_data_blocked():
    bad = aware(data_fit={"observable_core": "PARTIAL",
                          "mechanism_observability": "UNRESOLVED",
                          "prediction_testability": "UNAVAILABLE",
                          "overall": "REDUCED_FORM_TESTABLE"})
    assert H.data_fit_checks([bad])["data_fit_problem_count"] == 1
    assert H._derive_fit("UNRESOLVED", "UNAVAILABLE") == "DATA_BLOCKED"


def test_generation_prompt_carries_the_environment_but_not_its_values():
    prompt = H.generation_prompt({"evidence": []})
    assert "Research capability profile" in prompt
    assert "PASS 2b" in prompt
    # 정책 문장이 살아 있어야 한다 — 상상을 줄이라는 뜻이 아니다
    assert "not define which financial mechanisms you are allowed to CONSIDER" in prompt \
        or "does NOT" in prompt
    assert "VALIDATION_ONLY data exists but you cannot see its values" in prompt


# ---- v1.2 REVISE (§26~§53) ------------------------------------------------------

def parent():
    """검증에 들어갔던 원래 가설. CONTEXT_TRIGGER 구조다."""
    return {
        "hypothesis_id": "H1", "title": "맥락과 늦은 방아쇠",
        "hypothesis_structure": "CONTEXT_TRIGGER",
        "evidence_relations": [{"a": "TRADE_ACTIVITY", "b": "BOOK_IMBALANCE",
                                "relation": "CONTEXT_TRIGGER"}],
        "evidence_basis": [
            {"evidence_id": "EV_TRADE_ACTIVITY_01", "family": "TRADE_ACTIVITY",
             "representative_feature": "vol_flow", "observation": "높다",
             "role": "PERSISTENT_CONTEXT"},
            {"evidence_id": "EV_BOOK_IMBALANCE_01", "family": "BOOK_IMBALANCE",
             "representative_feature": "book_imbalance", "observation": "늦게 갈린다",
             "role": "TRIGGER"}],
        "mechanistic_claim": "높은 vol_flow 아래에서 book_imbalance 가 늦게 갈린다"}


def feedback():
    """§29 모양. 통계 수치는 없다."""
    return {"hypothesis_id": "H1", "hypothesis_status": "HYPOTHESIS_REVISE",
            "prediction_status": "SUPPORTED",
            "relation_status": "TRIGGER_ONLY_SUPPORTED",
            "mechanism_identification_status": "BLOCKED_BY_CAPABILITY",
            "already_priced_status": "RESIDUAL_MOVE_SUPPORTED",
            "route": "RETURN_TO_HYPOTHESIS", "primary_split": "FROZEN_HOLDOUT",
            "claim_feedback": [
                {"claim_id": "C6", "status": "SUPPORTED", "claim": "잔여 재가격"},
                {"claim_id": "C8", "status": "SUPPORTED", "claim": "늦은 상태 기여"},
                {"claim_id": "C5", "status": "BLOCKED_BY_CAPABILITY", "claim": "비동기 갱신"},
                {"claim_id": "STRUCTURE:CONTEXT_CONTRIBUTION", "status": "NOT_SUPPORTED",
                 "direction": "OPPOSITE", "claim": "맥락 기여"}]}


CONSUMED = ["20260316", "20260323", "20260330"]
NEXT = ["20260407", "20260408"]


def revised_payload(**over):
    payload = {
        "status": H.REVISED, "parent_hypothesis_id": "H1",
        "revision_reason": "맥락 기여가 지지되지 않아 구조만 줄였다",
        "revision_diff": [
            {"claim_id": "C6", "validation": "SUPPORTED", "action": "KEEP",
             "revised_claim": "잔여 재가격", "why": "지지됨"},
            {"claim_id": "C8", "validation": "SUPPORTED", "action": "KEEP",
             "revised_claim": "늦은 상태 기여", "why": "지지됨"},
            {"claim_id": "C5", "validation": "BLOCKED_BY_CAPABILITY",
             "action": "DOWNGRADE", "revised_claim": "미식별 해석으로 남긴다",
             "why": "못 본 것이지 틀린 것이 아니다"},
            {"claim_id": "STRUCTURE:CONTEXT_CONTRIBUTION", "validation": "NOT_SUPPORTED",
             "action": "REMOVE", "revised_claim": None, "why": "지지되지 않는다"}],
        "preserved_claims": ["C6", "C8"], "removed_claims": ["STRUCTURE:CONTEXT_CONTRIBUTION"],
        "downgraded_claims": ["C5"], "new_claims": [],
        "unresolved_mechanism": ["호가 갱신 비동기"],
        "revised_hypothesis": dict(aware(), **{
            "title": "늦은 호가 상태 하나",
            "hypothesis_structure": "SINGLE_MECHANISM",
            "evidence_relations": [],
            "evidence_basis": [
                {"evidence_id": "EV_BOOK_IMBALANCE_01", "family": "BOOK_IMBALANCE",
                 "representative_feature": "book_imbalance", "observation": "늦게 갈린다",
                 "role": "TRIGGER"}],
            "mechanistic_claim": "anchor 의 book_imbalance 만으로 설명한다"}),
        "requires_fresh_validation": True,
        "status_after_revision": H.REVISED_UNVALIDATED,
        "consumed_data_manifest": {"consumed_dates": CONSUMED,
                                   "next_validation_dates": NEXT}}
    payload.update(over)
    return payload


def check(payload=None, **over):
    return H.validate_revision(payload or revised_payload(**over), parent(), feedback(),
                               CONSUMED)


def test_minimal_revision_passes():
    result = check()
    assert result["problems"] == []
    for key in ("unsupported_claim_retained_count", "supported_claim_dropped_count",
                "blocked_claim_mishandled_count", "revision_new_feature_count",
                "revision_new_relation_count", "revision_new_mechanism_count",
                "validation_result_as_evidence_count", "fresh_validation_violation_count",
                "capability_violation_count", "unobservable_as_direct_count",
                "validation_only_leakage_count", "data_fit_problem_count"):
        assert result[key] == 0, (key, result.get(key.replace("_count", "s")))


def test_keeping_a_refuted_claim_is_caught():
    diff = revised_payload()["revision_diff"]
    diff[-1] = dict(diff[-1], action="KEEP", revised_claim="맥락은 여전히 기여한다")
    assert check(revision_diff=diff)["unsupported_claim_retained_count"] == 1


def test_dropping_a_supported_claim_is_caught():
    diff = revised_payload()["revision_diff"]
    diff[0] = dict(diff[0], action="REMOVE")
    assert check(revision_diff=diff)["supported_claim_dropped_count"] == 1


@pytest.mark.parametrize("action", ["KEEP", "REMOVE"])
def test_blocked_is_neither_a_fact_nor_a_falsehood(action):
    """BLOCKED_BY_CAPABILITY 는 거짓이 아니다. 지우지도, 확정으로 남기지도 않는다 (§33)."""
    diff = revised_payload()["revision_diff"]
    diff[2] = dict(diff[2], action=action)
    assert check(revision_diff=diff)["blocked_claim_mishandled_count"] == 1


def test_inventing_a_new_feature_is_caught():
    """검증에서 vol_flow 가 반대로 나왔다고 새 변수를 넣으면 그건 발견이다 (§35)."""
    revised = revised_payload()
    revised["revised_hypothesis"] = dict(
        revised["revised_hypothesis"],
        mechanistic_claim="낮은 spread_bps 와 높은 book_imbalance 가 함께 간다")
    assert check(revised)["revision_new_feature_count"] >= 1
    assert "spread_bps" in check(revised)["revision_new_features"]


def test_inventing_a_new_relation_is_caught():
    revised = revised_payload()
    revised["revised_hypothesis"] = dict(
        revised["revised_hypothesis"],
        evidence_relations=[{"a": "BOOK_IMBALANCE", "b": "PRICE_MOMENTUM",
                             "relation": "COUPLED"}])
    assert check(revised)["revision_new_relation_count"] == 1


def test_a_claim_id_that_was_never_validated_is_caught():
    diff = revised_payload()["revision_diff"] + [
        {"claim_id": "C99", "validation": "SUPPORTED", "action": "KEEP",
         "revised_claim": "새 주장", "why": ""}]
    assert check(revision_diff=diff)["revision_new_mechanism_count"] == 1


def test_quoting_the_validation_statistic_is_caught():
    """수정은 새 양적 패턴 채굴이 아니다 (§30)."""
    revised = revised_payload()
    revised["revised_hypothesis"] = dict(
        revised["revised_hypothesis"],
        mechanistic_claim="검증에서 짝 비율 0.527 이 나왔으므로 이 상태가 더 강하다")
    assert check(revised)["validation_result_as_evidence_count"] >= 1


def test_every_validated_claim_must_be_addressed():
    diff = revised_payload()["revision_diff"][:2]
    result = check(revision_diff=diff)
    assert result["uncovered_claims"] == ["C5", "STRUCTURE:CONTEXT_CONTRIBUTION"]


def test_fresh_validation_flags_are_set_by_code_not_by_the_agent(tmp_path):
    """Agent 가 빼먹어도 코드가 박는다 (§39~§41)."""
    calls = []

    def spy(prompt):
        calls.append(prompt)
        payload = revised_payload()
        payload.pop("requires_fresh_validation")       # Agent 가 안 넣었다고 치자
        payload.pop("status_after_revision")
        payload.pop("consumed_data_manifest")
        return payload

    result = H.run_revision(parent(), {"hypotheses": [{"hypothesis_id": "H1"}]},
                            feedback(), tmp_path / "rev", consumed_dates=CONSUMED,
                            next_validation_dates=NEXT, agent=spy)
    assert len(calls) == 2                              # 최소 수정 1 + 같은 역할 감사 1
    assert "Revision is not new hypothesis generation" in calls[0]
    assert "Remove what you invented" in calls[1]
    written = json.loads((tmp_path / "rev" / "hypothesis_revision.json").read_text())
    assert written["requires_fresh_validation"] is True
    assert written["status_after_revision"] == H.REVISED_UNVALIDATED
    assert written["consumed_data_manifest"]["consumed_dates"] == CONSUMED
    assert result["validation"]["fresh_validation_violation_count"] == 0


def test_next_validation_dates_may_not_reuse_consumed_dates():
    revised = revised_payload()
    revised["consumed_data_manifest"] = {"consumed_dates": CONSUMED,
                                         "next_validation_dates": ["20260330"]}
    result = check(revised)
    assert result["fresh_validation_violation_count"] == 1
    assert "20260330" in result["fresh_validation_problems"][0]


def test_new_discovery_required_is_a_normal_outcome():
    payload = revised_payload(status=H.NEW_DISCOVERY_REQUIRED,
                              new_discovery_reason="남는 주장만으로는 메커니즘이 안 된다",
                              revised_hypothesis=None, revision_diff=[])
    assert check(payload)["problems"] == []


def test_revision_refuses_when_validation_did_not_ask_for_it(tmp_path):
    """SUPPORTED 인데 수정을 부르면 Agent 를 부르지 않는다 (PASS R0)."""
    calls = []
    supported = dict(feedback(), hypothesis_status="REDUCED_FORM_SUPPORTED")
    result = H.run_revision(parent(), {"hypotheses": [{"hypothesis_id": "H1"}]},
                            supported, tmp_path / "rev", consumed_dates=CONSUMED,
                            next_validation_dates=NEXT,
                            agent=lambda p: calls.append(p))
    assert calls == []
    assert result["status"] == H.NEW_DISCOVERY_REQUIRED


def test_mode_routing_is_decided_by_the_workflow():
    """Agent 가 스스로 mode 를 고르지 않는다 (§43)."""
    assert H.route_mode({"hypothesis_status": "HYPOTHESIS_REVISE"}) == H.REVISE
    assert H.route_mode({"hypothesis_status": "HYPOTHESIS_REJECT"}) == H.GENERATE
    assert H.route_mode({"hypothesis_status": "REDUCED_FORM_SUPPORTED"}) is None
    assert H.route_mode({"hypothesis_status": "INCONCLUSIVE"}) is None


# ---- 스스로 선을 긋는 서술을 위반으로 세지 않는다 --------------------------------

@pytest.mark.parametrize("text,is_claim", [
    ("TRADE_ACTIVITY 가 ORDER_FLOW 변화를 유발한다.", True),
    ("TRADE_ACTIVITY 가 ORDER_FLOW 변화를 유발했다는 주장은 하지 않는다.", False),
    ("A causes B.", True),
    ("This does not claim that A causes B.", False),
    ("앞은 관측이다. A 가 B 를 유발한다. 나머지는 아니다.", True),
])
def test_a_disclaimer_is_not_a_causal_claim(text, is_claim):
    """`A 가 B 를 유발했다고 주장하지 않는다` 는 인과 주장이 아니라 그 반대다."""
    match = H.CAUSAL_WORDS.search(text)
    assert match is not None
    assert H._asserted(text, match) is is_claim


def test_capability_names_are_not_invented_features():
    """`participant_identity 는 관측할 수 없다` 는 정확한 서술이다. 지어낸 feature 가 아니다."""
    item = dict(good_hypothesis(), mechanistic_claim=(
        "participant_identity 와 order_cancellation 은 관측할 수 없어 "
        "post_anchor_executable_quotes 로만 예측을 시험한다"))
    result = H.validate_output(
        {"status": H.GENERATED, "hypotheses": [item]}, package())
    assert result["unknown_feature_count"] == 0
