"""Grounding 이 지켜야 할 것. LLM 을 부르지 않고 계약만 검사한다.

여기서 보는 것은 **의미 충실성의 계약**이다 — 카탈로그 밖 feature 를 만들지 않았는가,
proxy 를 직접 관측으로 둔갑시키지 않았는가, 예측을 관측으로 되돌리지 않았는가,
그리고 임계·전략이 끼어들지 않았는가.
"""

from __future__ import annotations

import json

import pytest

from framework import catalog, grounding as G, mechanism_graph as M
from framework.tests.test_hypothesis import family, package as base_package, good_hypothesis


def package():
    pkg = base_package()
    pkg["catalog_sha256"] = catalog.catalog_hash()
    pkg["evidence_roles"] = {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT",
                             "ORDER_FLOW": "LATE_TRIGGER", "MICROPRICE": "REDUNDANT"}
    pkg["independent_evidence_families"] = ["TRADE_ACTIVITY", "ORDER_FLOW"]
    pkg["joint_evidence"] = [{"joint_evidence_id": "JEV_ORDER_FLOW__TRADE_ACTIVITY",
                              "family_a": "ORDER_FLOW", "family_b": "TRADE_ACTIVITY",
                              "relation": "CONTEXT_TRIGGER",
                              "context_family": "TRADE_ACTIVITY",
                              "trigger_family": "ORDER_FLOW",
                              "ordering_status": "CONTEXT_BEFORE_TRIGGER"}]
    return pkg


def hypotheses():
    h = good_hypothesis()
    h.pop("implementation_directions")  # 기존 canonical Grounding 계약 fixture
    h["hypothesis_structure"] = "CONTEXT_TRIGGER"
    h["already_priced_risk"] = True
    return {"status": "HYPOTHESES_GENERATED", "no_hypothesis_reason": None, "hypotheses": [h]}


def claim(cid="C1", **kwargs):
    base = {"claim_id": cid, "claim_text": "거래 활동이 anchor 이전부터 높다",
            "claim_type": "OBSERVABLE_STATE", "epistemic_status": "OBSERVED",
            "importance": "CORE", "source_section": "expected_observable_sequence",
            "supporting_evidence_ids": ["EV_TRADE_ACTIVITY_01"], "depends_on": [],
            "grounding_status": "GROUNDED_DIRECT", "catalog_family": "TRADE_ACTIVITY",
            "catalog_feature": "vol_flow", "operator": None, "direction": "HIGHER",
            "proxy_target": None, "proxy_rationale": None, "proxy_limitation": None,
            "required_data": ["trade_prints"], "missing_capability": [],
            "validation_target": None, "validation_measure": None,
            "validation_reference": None, "grounding_ambiguity": [], "notes": ""}
    base.update(kwargs)
    return base


def grounded(claims=None, readiness="PARTIALLY_GROUNDED", temporal=None):
    items = claims if claims is not None else [claim()]
    observable = next((item for item in items
                       if item.get("claim_type") == "OBSERVABLE_STATE"
                       and item.get("grounding_status") in G.DIRECT_STATUS
                       and item.get("catalog_feature")), None)
    expression = None
    claim_ids = []
    note = None
    if observable is not None:
        feature = observable["catalog_feature"]
        direction = observable.get("direction")
        expression = {
            "op": "compare", "input": {"op": "primitive", "primitive_id": feature},
            "comparator": ">" if direction == "HIGHER" else "<",
            "value": f"UNRESOLVED:theta_{feature}",
        }
        claim_ids = [observable["claim_id"]]
        note = "관측적 상태이며 금융 메커니즘의 직접 증명은 아니다"
    return {"hypotheses": [{
        "hypothesis_id": "H1", "readiness": readiness,
        "readiness_reason": "핵심 관측은 붙었고 메커니즘은 미해결이다",
        "operational_expression": expression,
        "operational_claim_ids": claim_ids,
        "operational_note": note,
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
            "catalog_bindings": [
                {"observable_id": "O_CONTEXT", "catalog_feature": "vol_flow"},
                {"observable_id": "O_TRIGGER", "catalog_feature": "ofi_depth_5"},
            ],
        },
        "claims": items,
        "temporal_grounding": temporal if temporal is not None else {
            "reference_event": "profit anchor", "context_family": "TRADE_ACTIVITY",
            "trigger_family": "ORDER_FLOW", "relation": "CONTEXT_TRIGGER",
            "evidence_ordering": "TRADE_ACTIVITY separation 이 먼저 관측된다",
            "validation_ordering": "개별 사건 경로에서 확인해야 한다"},
        "discriminating_observations": [],
        "coverage": {"core_claims": 1, "grounded_direct": 1, "grounded_proxy": 0,
                     "validation_targets": 0, "unresolved_mechanisms": 0, "unobservable": 0}}]}


def composite_grounded():
    direct_flow = claim("C2", claim_text="anchor 이전 OFI가 낮다",
                        catalog_family="ORDER_FLOW", catalog_feature="ofi_depth_10",
                        direction="LOWER", supporting_evidence_ids=["EV_ORDER_FLOW_01"])
    relation = claim("C3", claim_text="두 관측의 기록된 관계",
                     claim_type="TEMPORAL_RELATION",
                     epistemic_status="SUPPORTED_RELATION",
                     importance="SUPPORTING",
                     grounding_status="GROUNDED_DERIVED",
                     catalog_family=None, catalog_feature=None, direction="NONE",
                     supporting_evidence_ids=["JEV_ORDER_FLOW__TRADE_ACTIVITY"])
    value = grounded([claim("C1"), direct_flow, relation])
    hypothesis = value["hypotheses"][0]
    hypothesis["operational_expression"] = {
        "op": "all", "args": [
            {"op": "compare", "input": {"op": "primitive", "primitive_id": "vol_flow"},
             "comparator": ">", "value": "UNRESOLVED:theta_vol_flow"},
            {"op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
             "comparator": "<", "value": "UNRESOLVED:theta_ofi_depth_10"},
        ]}
    hypothesis["operational_claim_ids"] = ["C1", "C2", "C3"]
    hypothesis["operational_note"] = "관측적 결합이며 메커니즘의 직접 증명은 아니다"
    return value


# ---- PASS 0 --------------------------------------------------------------------

def test_input_audit_passes_on_a_valid_hypothesis():
    audit = G.input_audit(hypotheses(), package())
    assert audit["ok"] and audit["hypotheses"] == ["H1"]


def test_input_audit_accepts_hypothesis_local_evidence_roles():
    hs = hypotheses()
    hs["hypotheses"][0]["evidence_roles"] = {"ORDER_FLOW": "UNRESOLVED"}
    pkg = package()
    pkg.pop("evidence_roles")

    assert G.input_audit(hs, pkg)["ok"]
    assert G.grounding_input(hs, pkg)["hypothesis_evidence_roles"] == {
        "H1": {"ORDER_FLOW": "UNRESOLVED"}
    }


def test_input_audit_blocks_when_there_is_nothing_to_ground():
    empty = {"status": "NO_HYPOTHESIS", "no_hypothesis_reason": "NO_PLAUSIBLE_ECONOMIC_MECHANISM",
             "hypotheses": []}
    assert not G.input_audit(empty, package())["ok"]


def test_input_audit_catches_a_catalog_hash_mismatch():
    pkg = package(); pkg["catalog_sha256"] = "stale"
    assert any("hash" in p for p in G.input_audit(hypotheses(), pkg)["problems"])


def test_run_blocks_at_pass_zero_without_calling_the_agent(tmp_path):
    calls = []
    empty = {"status": "NO_HYPOTHESIS", "no_hypothesis_reason": "NO_PLAUSIBLE_ECONOMIC_MECHANISM",
             "hypotheses": []}
    hp, pp = tmp_path / "h.json", tmp_path / "p.json"
    hp.write_text(json.dumps(empty), encoding="utf-8")
    pp.write_text(json.dumps(package()), encoding="utf-8")
    out = G.run(hp, pp, tmp_path / "out", agent=lambda p: calls.append(p) or grounded())
    assert calls == []
    assert out["audit"]["blocked"] is True


# ---- Agent 가 보는 것 -----------------------------------------------------------

def test_grounding_input_hides_post_anchor_and_thresholds():
    payload = G.grounding_input(hypotheses(), package())
    text = json.dumps(payload, ensure_ascii=False)
    assert "post-anchor price path" in payload["not_provided"]
    # 관측 통계·임계를 주지 않는다 — 주면 그 자리에서 숫자를 고르게 된다
    assert "observed" not in text.lower().split("observable_description")[0][-200:]
    for feature in payload["catalog"]["features"]:
        assert set(feature) == {"name", "family", "dimension", "unit", "time_basis",
                                "lookback", "observable_description", "definition_id",
                                "threshold_policy"}


def test_grounding_input_only_offers_catalog_names():
    payload = G.grounding_input(hypotheses(), package())
    assert all(f["name"] in catalog.FEATURES for f in payload["catalog"]["features"])
    assert all(o["op"] in catalog.OPERATORS for o in payload["catalog"]["operators"])


# ---- 데이터 capability ----------------------------------------------------------

def test_post_anchor_is_available_for_validation_not_unobservable():
    """가설 입력에서 가린 것과 데이터에 없는 것은 다르다."""
    inv = G.data_capability_inventory()
    assert inv["capabilities"]["post_anchor_executable_quotes"]["status"] \
        == "VALIDATION_ONLY"
    assert inv["capabilities"]["participant_identity"]["status"] == "UNAVAILABLE"
    assert inv["capabilities"]["order_add_cancel_decomposition"]["status"] == "PROXY_ONLY"


def test_unknown_requested_capability_is_reported_not_invented():
    inv = G.data_capability_inventory(["l1_quote", "telepathy"])
    assert inv["requested_but_unknown"] == ["telepathy"]


# ---- 결정적 검증 ----------------------------------------------------------------

def test_clean_grounding_passes():
    report = G.validate(grounded(), hypotheses(), package())
    assert report["problems"] == []
    assert report["structural_violation_count"] == 0


def test_progressable_grounding_requires_the_canonical_expression():
    bad = grounded()
    bad["hypotheses"][0]["operational_expression"] = None
    bad["hypotheses"][0]["operational_claim_ids"] = []
    bad["hypotheses"][0]["operational_note"] = None
    report = G.validate(bad, hypotheses(), package())
    assert report["invalid_operational_expression_count"] == 1
    assert any("operational_expression" in item for item in report["problems"])


def test_grounding_accepts_a_supported_composite_catalog_expression():
    report = G.validate(composite_grounded(), hypotheses(), package())
    assert report["problems"] == []
    assert report["invalid_operational_expression_count"] == 0


def test_composite_expression_cannot_use_an_uncited_feature():
    bad = composite_grounded()
    bad["hypotheses"][0]["operational_expression"]["args"][1]["input"]["primitive_id"] = "book_imbalance"
    report = G.validate(bad, hypotheses(), package())
    assert report["invalid_operational_expression_count"] >= 1
    assert any("인용한 직접 관측 claim에 없다" in item for item in report["problems"])


def test_composite_expression_cannot_combine_independent_observations():
    bad = composite_grounded()
    bad["hypotheses"][0]["temporal_grounding"]["relation"] = "CONTEXT_INDEPENDENT"
    report = G.validate(bad, hypotheses(), package())
    assert report["invalid_operational_expression_count"] == 1
    assert any("CONTEXT_INDEPENDENT" in item for item in report["problems"])


def test_composite_expression_cannot_choose_a_cutoff_in_grounding():
    bad = composite_grounded()
    bad["hypotheses"][0]["operational_expression"]["args"][0]["value"] = 0.5
    report = G.validate(bad, hypotheses(), package())
    assert report["invalid_operational_expression_count"] == 1
    assert any("cutoff" in item for item in report["problems"])


def test_operational_expression_cannot_reverse_a_claim_direction():
    bad = grounded([claim(catalog_feature="ofi_depth_10", catalog_family="ORDER_FLOW",
                         direction="LOWER")])
    bad["hypotheses"][0]["operational_expression"]["comparator"] = ">"
    report = G.validate(bad, hypotheses(), package())
    assert report["invalid_operational_expression_count"] == 1
    assert any("claim 방향 LOWER" in item for item in report["problems"])


def test_feature_outside_the_catalog_is_caught():
    bad = grounded([claim(catalog_feature="absorption_score")])
    report = G.validate(bad, hypotheses(), package())
    assert report["unknown_feature_count"] == 1
    assert any("카탈로그에 없는 feature" in p for p in report["problems"])


def test_legacy_alias_is_not_a_canonical_grounding():
    bad = grounded([claim(catalog_feature="microprice_velocity_1t")])
    report = G.validate(bad, hypotheses(), package())
    assert report["unknown_feature_count"] == 1
    assert any("별칭" in p for p in report["problems"])


def test_operator_outside_the_catalog_is_caught():
    bad = grounded([claim(operator="smooth")])
    assert G.validate(bad, hypotheses(), package())["invalid_operator_count"] == 1


def test_proxy_treated_as_direct_is_caught():
    bad = grounded([claim(grounding_status="GROUNDED_DIRECT",
                          proxy_target="bid-side depth persistence")])
    report = G.validate(bad, hypotheses(), package())
    assert report["proxy_as_direct_count"] == 1


def test_proxy_without_a_stated_limitation_is_caught():
    bad = grounded([claim(grounding_status="GROUNDED_PROXY", catalog_feature=None,
                          proxy_target="depth persistence", proxy_limitation=None)])
    assert any("한계가 비었다" in p for p in G.validate(bad, hypotheses(), package())["problems"])


def test_mechanism_interpretation_cannot_be_a_direct_observable():
    bad = grounded([claim(claim_type="MECHANISM_INTERPRETATION",
                          epistemic_status="HYPOTHESIZED",
                          grounding_status="GROUNDED_DIRECT")])
    report = G.validate(bad, hypotheses(), package())
    assert report["semantic_mismatch_count"] >= 1
    assert any("메커니즘 해석을 직접 관측" in p for p in report["problems"])


def test_prediction_marked_as_observed_is_future_leakage():
    bad = grounded([claim(claim_type="PREDICTED_CONSEQUENCE",
                          epistemic_status="OBSERVED",
                          grounding_status="GROUNDED_DIRECT",
                          catalog_feature="bid_queue_depth_at_l1")])
    report = G.validate(bad, hypotheses(), package())
    assert report["future_leakage_count"] >= 1


def test_validation_target_needs_a_target_type():
    bad = grounded([claim(claim_type="PREDICTED_CONSEQUENCE", epistemic_status="TO_BE_TESTED",
                          grounding_status="VALIDATION_TARGET", catalog_feature=None,
                          validation_target=None)])
    assert any("종류가 없다" in p for p in G.validate(bad, hypotheses(), package())["problems"])


def test_redundant_family_cannot_be_an_independent_grounding():
    bad = grounded([claim(catalog_family="MICROPRICE", catalog_feature="microprice_dev_bps")])
    report = G.validate(bad, hypotheses(), package())
    assert report["redundant_grounding_count"] == 1


@pytest.mark.parametrize("text", [
    "vol_flow > 124.3 일 때",
    "상위 q20 구간에서",
    "stop-loss 를 건다",
])
def test_threshold_or_strategy_leakage_is_caught(text):
    bad = grounded([claim(claim_text=text)])
    report = G.validate(bad, hypotheses(), package())
    assert report["threshold_leakage_count"] + report["strategy_leakage_count"] >= 1


def test_temporal_relation_conflicting_with_the_evidence_is_caught():
    bad = grounded(temporal={"reference_event": "profit anchor",
                             "context_family": "ORDER_FLOW", "trigger_family": "TRADE_ACTIVITY",
                             "relation": "COUPLED", "evidence_ordering": "-",
                             "validation_ordering": "-"})
    report = G.validate(bad, hypotheses(), package())
    assert report["semantic_mismatch_count"] >= 1


def test_unresolved_core_is_counted_but_is_not_a_violation():
    soft = grounded([claim(claim_type="MECHANISM_INTERPRETATION",
                           epistemic_status="HYPOTHESIZED", importance="CORE",
                           grounding_status="UNRESOLVED_MECHANISM",
                           catalog_family=None, catalog_feature=None, direction="NONE")],
                    readiness="BLOCKED_BY_CORE_UNOBSERVABLE")
    report = G.validate(soft, hypotheses(), package())
    assert report["unresolved_core_count"] == 1
    assert report["structural_violation_count"] == 0     # 연구 결과이지 위반이 아니다
    assert report["problems"] == []


def test_dangling_claim_dependency_is_caught():
    bad = grounded([claim(depends_on=["C9"])])
    assert any("없는 claim" in p for p in G.validate(bad, hypotheses(), package())["problems"])


# ---- 산출물 --------------------------------------------------------------------

def test_validation_plan_is_designed_but_never_run():
    target = claim("C5", claim_type="PREDICTED_CONSEQUENCE", epistemic_status="TO_BE_TESTED",
                   grounding_status="VALIDATION_TARGET", catalog_feature=None,
                   validation_target="PATH_TARGET",
                   validation_measure="anchor 이후 BID1 경로")
    plan = G.validation_plan(grounded([claim(), target]), hypotheses())
    assert plan["count"] >= 1
    assert all(t["status"] == "NOT_RUN" and t["result"] is None for t in plan["targets"])
    assert any(t["target_type"] == "PATH_TARGET" for t in plan["targets"])
    # 가설의 mechanism test 도 계획에 들어간다
    assert any(t["target_type"] == "TEMPORAL_PRECEDENCE" for t in plan["targets"])


def test_claim_table_carries_catalog_provenance():
    table = G.claim_table(grounded())
    assert table.definition_id.iloc[0] == catalog.FEATURES["vol_flow"].definition_id


def test_run_writes_all_artifacts(tmp_path):
    hp, pp = tmp_path / "h.json", tmp_path / "p.json"
    hp.write_text(json.dumps(hypotheses()), encoding="utf-8")
    pp.write_text(json.dumps(package()), encoding="utf-8")
    out = G.run(hp, pp, tmp_path / "out", agent=lambda prompt: grounded())
    for name in ("grounded_hypotheses.json", "grounded_hypotheses.md",
                 "claim_grounding.parquet", "data_capability_inventory.json",
                 "grounding_audit.json", "validation_plan_draft.json",
                 "grounding_manifest.json"):
        assert (tmp_path / "out" / name).exists(), name
    assert out["manifest"]["catalog_hash"] == catalog.catalog_hash()
    assert out["audit"]["validation"]["structural_violation_count"] == 0


def test_prompt_forbids_search_and_demands_fidelity():
    payload = G.grounding_input(hypotheses(), package())
    prompt = G.grounding_prompt(payload)
    for phrase in ("Do NOT invent features", "Do NOT choose numeric threshold values",
                   "Semantic fidelity beats implementability",
                   "You cannot see outcomes", "VALIDATION_TARGET"):
        assert phrase in prompt, phrase


def test_audit_prompt_forbids_adding_claims():
    payload = G.grounding_input(hypotheses(), package())
    prompt = G.audit_prompt(payload, grounded())
    assert "Do not add claims and do not add hypotheses" in prompt
