"""실행 명세가 가설의 의미를 지키는가.

여기서 보는 것은 하나다 — **어디까지 바꿔도 같은 가설인가.** 코드가 돈다는 것과
가설을 충실히 구현했다는 것은 다르다.
"""

from __future__ import annotations

import copy

import pytest

from framework import (catalog, executable as X, implementation as I, profit_target as PT,
                       sample_condition as SC)


def revision(feature="book_imbalance", family="BOOK_IMBALANCE"):
    return {
        "parent_hypothesis_id": "H1",
        "status_after_revision": "REVISED_UNVALIDATED",
        "removed_claims": ["STRUCTURE:CONTEXT_CONTRIBUTION"],
        "unresolved_mechanism": ["비동기 호가 갱신 여부"],
        "revised_hypothesis": {
            "hypothesis_id": "H1-R1",
            "hypothesis_structure": "SINGLE_MECHANISM",
            "evidence_roles": {family: "LATE_TRIGGER"},
            "evidence_basis": [
                {"evidence_id": f"EV_{family}_01", "family": family,
                 "representative_feature": feature, "role": "TRIGGER"}],
        },
    }


def routing(status="REDUCED_FORM_SUPPORTED", mechanism="BLOCKED_BY_CAPABILITY"):
    return {"hypothesis_id": "H1-R1", "hypothesis_status": status,
            "evidence_tier": "REPEATED_HOLDOUT_SUPPORTED",
            "mechanism_identification_status": mechanism,
            "primary_split": "FROZEN_HOLDOUT"}


def manifest():
    return {
        "catalog_hash": catalog.catalog_hash(),
        "grounding_hash": "abc123",
        "validation_plan_hash": "def456",
        "capability_profile_hash": "ghi789",
        "outcome_definition": {"side": "LONG", "entry_price": "ASK1",
                               "exit_price": "BID1", "cost_bps": 23.0},
        "validation_plan": {"context_feature": "vol_flow"},
        "config": {"horizon_seconds": 30.0, "anchor_spacing_seconds": None,
                   "context_feature": "vol_flow"},
        "revision": {"revision_hash": "r1", "revised_hypothesis_hash": "h1"},
    }


def results(effect=0.0677):
    return {"MV-H1-01": {"effect_size": effect, "null_value": 0.0},
            "MV-H1-04": {"effect_size": 0.4631, "null_value": 0.5,
                         "status": "NOT_SUPPORTED"}}


def spec(**over):
    built = X.build(revision(), routing(), manifest(), results())
    built.update(over)
    return built


def test_grounded_matched_control_prediction_can_define_the_specification():
    claim = {"claim_id": "C1", "grounding_status": "VALIDATION_TARGET",
             "validation_target": "MATCHED_CONTROL_TARGET", "direction": X.HIGHER}
    assert X._grounded_prediction({"claims": [claim]}) == claim


def test_grounded_path_prediction_prefers_the_core_target():
    core = {"claim_id": "C1", "grounding_status": "VALIDATION_TARGET",
            "validation_target": "PATH_TARGET", "direction": X.HIGHER,
            "importance": "CORE"}
    supporting = {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
                  "validation_target": "PATH_TARGET", "direction": X.HIGHER,
                  "importance": "SUPPORTING"}
    assert X._grounded_prediction({"claims": [supporting, core]}) == core


def test_grounded_prediction_follows_the_mechanism_graph_target():
    matched = {"claim_id": "C1", "grounding_status": "VALIDATION_TARGET",
               "validation_target": "MATCHED_CONTROL_TARGET", "direction": X.HIGHER,
               "importance": "CORE"}
    path = {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
            "validation_target": "PATH_TARGET", "direction": X.HIGHER,
            "importance": "SUPPORTING"}
    grounded = {"claims": [path, matched], "mechanism_graph": {
        "prediction": {"target_type": "MATCHED_CONTROL_TARGET", "direction": X.HIGHER}}}
    assert X._grounded_prediction(grounded) == matched


def test_grounded_spec_uses_signal_direction_not_future_path_direction():
    hypothesis = {"hypothesis_id": "H1", "evidence_roles": {"ORDER_FLOW": "LATE_TRIGGER"},
                  "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01",
                                      "family": "ORDER_FLOW",
                                      "representative_feature": "ofi_depth_10"}]}
    expression = {"op": "compare",
                  "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
                  "comparator": "<", "value": "UNRESOLVED:theta_ofi"}
    grounded = {"operational_expression": expression, "operational_claim_ids": ["C1"], "claims": [
        {"claim_id": "C1", "grounding_status": "GROUNDED_DIRECT",
         "catalog_feature": "ofi_depth_10", "direction": X.LOWER},
        {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
         "validation_target": "PATH_TARGET", "direction": X.HIGHER,
         "importance": "CORE"},
    ]}
    assert X.grounded_semantic_invariants(hypothesis, grounded)["direction"] == X.LOWER


def test_grounded_operational_claim_selects_weak_unresolved_evidence_feature():
    hypothesis = {
        "hypothesis_id": "H1",
        "evidence_roles": {"QUEUE_STATE": "UNRESOLVED"},
        "evidence_basis": [{
            "evidence_id": "EV_QUEUE_STATE_06", "family": "QUEUE_STATE",
            "representative_feature": "bid_queue_depth_at_l1",
            "role": "WEAK_OR_AMBIGUOUS",
        }],
    }
    grounded = {
        "operational_expression": {
            "op": "compare",
            "input": {"op": "primitive", "primitive_id": "bid_queue_depth_at_l1"},
            "comparator": ">", "value": "UNRESOLVED:theta_bid_queue_depth_at_l1",
        },
        "operational_claim_ids": ["C1"],
        "claims": [
            {"claim_id": "C1", "grounding_status": "GROUNDED_DIRECT",
             "catalog_feature": "bid_queue_depth_at_l1", "direction": X.HIGHER},
            {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
             "validation_target": "PATH_TARGET", "direction": X.HIGHER,
             "importance": "CORE", "supporting_evidence_ids": []},
        ],
    }

    invariants = X.grounded_semantic_invariants(hypothesis, grounded)
    assert invariants["canonical_feature"] == "bid_queue_depth_at_l1"
    assert invariants["feature_family"] == "QUEUE_STATE"


def test_grounding_composite_ast_reaches_specification_without_feature_collapse():
    hypothesis = {"hypothesis_id": "H-COMPOSITE",
                  "profit_target": PT.canonical_target(),
                  "evidence_roles": {"ORDER_FLOW": "LATE_TRIGGER"},
                  "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01",
                                      "family": "ORDER_FLOW",
                                      "representative_feature": "ofi_depth_10"}]}
    expression = {
        "op": "all", "args": [
            {"op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
             "comparator": "<", "value": "UNRESOLVED:theta_ofi"},
            {"op": "compare", "input": {"op": "primitive", "primitive_id": "microprice_dev_bps"},
             "comparator": ">", "value": "UNRESOLVED:theta_microprice"},
        ],
    }
    grounded = {"operational_expression": expression, "operational_claim_ids": ["C1", "C2"],
                "claims": [
                    {"claim_id": "C1", "grounding_status": "GROUNDED_DIRECT",
                     "catalog_feature": "ofi_depth_10", "direction": X.LOWER},
                    {"claim_id": "C2", "grounding_status": "GROUNDED_DIRECT",
                     "catalog_feature": "microprice_dev_bps", "direction": X.HIGHER},
                    {"claim_id": "C3", "grounding_status": "VALIDATION_TARGET",
                     "validation_target": "PATH_TARGET", "direction": X.HIGHER,
                     "importance": "CORE", "supporting_evidence_ids": ["EV_ORDER_FLOW_01"]},
                ]}
    built = X.build_from_grounding(hypothesis, grounded, {"catalog_sha256": "test"})
    assert built["signal_template"] == expression
    assert [item["name"] for item in built["search_boundary"]["allowed_parameters"]] == [
        "theta_microprice", "theta_ofi"]
    assert built["search_boundary"]["fixed_semantics"]["entry_features"] == [
        "microprice_dev_bps", "ofi_depth_10"]
    assert X.fidelity_audit(built, hypothesis, {}, expected_direction=X.LOWER)["fidelity_status"] \
        == X.FIDELITY_PRESERVED


def test_feature_profile_sample_condition_is_anded_into_the_runtime_entry_program():
    hypothesis = {
        "hypothesis_id": "H-M0", "profit_target": PT.canonical_target(),
        "executable_sample_condition": SC.m0_price_recovery(),
        "evidence_roles": {"ORDER_FLOW": "LATE_TRIGGER"},
        "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01", "family": "ORDER_FLOW",
                            "representative_feature": "ofi_depth_10"}],
    }
    candidate = {
        "op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
        "comparator": "<", "value": "UNRESOLVED:theta_ofi",
    }
    grounded = {
        "operational_expression": candidate, "operational_claim_ids": ["C1"],
        "claims": [
            {"claim_id": "C1", "grounding_status": "GROUNDED_DIRECT",
             "catalog_feature": "ofi_depth_10", "direction": X.LOWER},
            {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
             "validation_target": "PATH_TARGET", "direction": X.HIGHER,
             "importance": "CORE", "supporting_evidence_ids": ["EV_ORDER_FLOW_01"]},
        ],
    }
    specification = X.build_from_grounding(hypothesis, grounded, {"catalog_sha256": "test"})
    expected = {"op": "all", "args": [SC.m0_price_recovery(), candidate]}
    assert specification["signal_template"] == expected
    template = I.compile_contract(specification)
    assert template["entry_program"]["signal"] == expected
    assert template["executable_sample_condition"] == SC.m0_price_recovery()


def test_direct_evidence_fallback_keeps_the_same_feature_profile_sample_condition():
    candidate = {
        "hypothesis_id": "DE-M0", "profit_target": PT.canonical_target(),
        "executable_sample_condition": SC.m0_price_recovery(),
        "executable_sample_condition_sha256": SC.condition_hash(SC.m0_price_recovery()),
        "entry_states": [{"feature": "book_imbalance", "direction": X.HIGHER,
                          "family": "BOOK_IMBALANCE"}],
    }
    specification = X.build_from_direct_evidence(
        candidate, {"catalog_sha256": catalog.catalog_hash()})
    signal = specification["signal_template"]
    assert signal["op"] == "all"
    assert signal["args"][0] == SC.m0_price_recovery()
    assert signal["args"][1]["input"]["primitive_id"] == "book_imbalance"
    assert I.compile_contract(specification)["entry_program"]["warmup_ticks"] == 20


# ---- 무엇이 고정인가 ------------------------------------------------------------

def test_invariants_are_derived_from_validation_not_typed_in():
    """방향을 손으로 적으면 검증과 어긋날 수 있다. 통계 부호에서 읽는다."""
    up = X.build(revision(), routing(), manifest(), results(effect=+0.07))
    down = X.build(revision(), routing(), manifest(), results(effect=-0.07))
    assert up["semantic_invariants"]["direction"] == X.HIGHER
    assert down["semantic_invariants"]["direction"] == X.LOWER
    assert up["signal_template"]["comparator"] == ">"
    assert down["signal_template"]["comparator"] == "<"


def test_core_observable_comes_from_the_revised_hypothesis():
    built = spec()
    inv = built["semantic_invariants"]
    assert inv["canonical_feature"] == "book_imbalance"
    assert inv["feature_family"] == "BOOK_IMBALANCE"
    assert built["signal_template"]["input"]["primitive_id"] == "book_imbalance"


def test_execution_semantics_follow_the_outcome_definition():
    inv = spec()["semantic_invariants"]
    assert inv["side"] == "LONG"
    assert inv["entry_reference"] == "ASK1"
    assert inv["future_execution_reference"] == "BID1"
    assert inv["validation_horizon"] == "S30"


def test_rejected_context_is_recorded_as_an_invariant():
    """제거한 맥락은 "안 쓴다" 가 아니라 "쓰면 다른 가설" 이다."""
    inv = spec()["semantic_invariants"]
    assert inv["context_required"] is False
    assert inv["rejected_context_feature"] == "vol_flow"
    assert inv["rejected_context_family"] == "TRADE_ACTIVITY"


# ---- 임계는 여기서 고르지 않는다 --------------------------------------------------

def test_threshold_is_declared_not_chosen():
    template = spec()["signal_template"]
    assert isinstance(template["value"], str)
    assert template["value"].startswith(catalog.UNRESOLVED_PREFIX)
    assert template["_parameter"]["status"] == X.SEARCHABLE


def test_literal_threshold_source_is_declared_separately_from_the_expression():
    built = X.build(
        revision(), routing(), manifest(), results(),
        config=X.SpecConfig(threshold_kind=X.THRESHOLD_LITERAL,
                            threshold_grid=(0.50, 0.70)),
    )
    parameter = built["signal_template"]["_parameter"]
    assert parameter["threshold_source"] == {
        "kind": X.THRESHOLD_LITERAL, "grid": [0.50, 0.70]}
    assert built["search_boundary"]["allowed_parameters"][0]["threshold_source"] \
        == parameter["threshold_source"]


def test_the_placeholder_is_the_one_contract_already_understands():
    """명세가 새 표기법을 만들면 구현이 그것을 해석하면서 뜻이 갈린다."""
    from framework import contract
    template = spec()["signal_template"]
    nodes = contract._unresolved_nodes(
        {k: v for k, v in template.items() if not k.startswith("_")})
    assert [name for name, _ in nodes] == ["UNRESOLVED:theta_book_imbalance"]


def test_only_the_threshold_axis_is_open():
    boundary = spec()["search_boundary"]
    assert [p["name"] for p in boundary["allowed_parameters"]] \
        == ["theta_book_imbalance"]
    axes = " ".join(item["axis"] for item in boundary["forbidden_parameters"])
    for closed in ("persistence", "velocity", "평가 창", "방향", "결정 간격"):
        assert closed in axes, closed


# ---- 금지된 변형을 실제로 잡는가 (§24) --------------------------------------------

def audit_of(mutate):
    built = spec()
    mutate(built)
    return X.fidelity_audit(built, revision(), results())


def test_reintroducing_the_rejected_context_is_caught():
    result = audit_of(lambda s: s["signal_template"].update({
        "op": "all", "args": [dict(s["signal_template"]),
                              {"op": "compare",
                               "input": {"op": "primitive", "primitive_id": "vol_flow"},
                               "comparator": "<", "value": "UNRESOLVED:phi"}]}))
    assert result["fidelity_status"] == X.FIDELITY_VIOLATED
    assert result["rejected_context_reintroduced_count"] == 1


@pytest.mark.parametrize("swap", ["book_imbalance_velocity", "microprice_dev_bps",
                                  "queue_imbalance_best"])
def test_swapping_in_a_similar_feature_is_caught(swap):
    """비슷한 금융 개념이라고 같은 가설로 취급하지 않는다 (§4)."""
    result = audit_of(lambda s: s["signal_template"]["input"].update({"primitive_id": swap}))
    assert result["fidelity_status"] == X.FIDELITY_VIOLATED
    assert result["new_feature_count"] >= 1


def test_adding_persistence_is_caught():
    """이 가설은 지속을 검증하지 않았다. N 을 탐색하면 없던 시간 구조를 만든다 (§12)."""
    result = audit_of(lambda s: s["signal_template"].update({
        "op": "persistence", "condition": dict(s["signal_template"]), "window": 5}))
    assert result["new_relation_count"] == 1


def test_flipping_the_direction_is_caught():
    def flip(s):
        s["semantic_invariants"]["direction"] = X.LOWER
        s["signal_template"]["comparator"] = "<"
    result = audit_of(flip)
    assert result["direction_change_count"] >= 1


def test_changing_the_horizon_is_caught():
    result = audit_of(lambda s: s["exit_policy"].update({"evaluation_horizon": "S60"}))
    assert result["horizon_change_count"] == 1


def test_an_unapproved_search_axis_is_caught():
    result = audit_of(lambda s: s["signal_template"].update(
        {"value": "UNRESOLVED:theta_secret"}))
    assert result["unapproved_search_parameter_count"] == 1


def test_a_faithful_spec_passes_every_metric():
    result = X.fidelity_audit(spec(), revision(), results())
    assert result["fidelity_status"] == X.FIDELITY_PRESERVED
    for key in ("new_feature_count", "new_relation_count", "new_mechanism_count",
                "rejected_context_reintroduced_count", "direction_change_count",
                "horizon_change_count", "unapproved_search_parameter_count"):
        assert result[key] == 0, key


# ---- 금지 feature 목록이 정확한가 -------------------------------------------------

def test_forbidden_list_names_the_easy_swaps():
    named = {item["feature"] for item in spec()["forbidden_features"]}
    for feature in ("vol_flow", "book_imbalance_velocity", "microprice_dev_bps"):
        assert feature in named, feature


def test_derived_features_are_forbidden_with_their_identity():
    """MICROPRICE 는 BOOK_IMBALANCE 에서 결정적으로 나온다. 다른 이름의 같은 양이다."""
    item = next(f for f in spec()["forbidden_features"]
                if f["feature"] == "microprice_dev_bps")
    assert "결정적으로 파생" in item["why"]
    assert "spread_bps" in item["why"]


def test_unrelated_price_levels_are_not_called_siblings():
    """`family_of` 의 마지막 줄은 조건 없는 fallback 이다. 최우선호가까지 "같은 family"
    라고 부르면 이유가 거짓이 된다."""
    named = {item["feature"] for item in spec()["forbidden_features"]}
    for unrelated in ("ask_1", "bid_1", "mid_price"):
        assert unrelated not in named, unrelated


# ---- 실행 결속은 탐색 파라미터가 아니다 (§14~§19) ---------------------------------

def test_decision_cadence_is_bound_not_searched():
    policy = spec()["decision_opportunity_policy"]
    assert policy["status"] == X.EXECUTION_ASSUMPTION
    assert policy["not_a_search_parameter"] is True
    assert policy["initial_binding"] == "EVERY_TICK"
    assert policy["initial_spacing_seconds"] is None      # 간격이 없다
    assert "틱 수로" in policy["tick_count_warning"]


def test_entry_binding_belongs_to_the_canonical_profile():
    """체결 방식은 가설이 정하지 않는다. 모든 가설이 같은 장비를 써야 한다."""
    from framework import canonical as K
    entry = spec()["entry_execution_semantics"]
    assert entry["owner"] == "CANONICAL_BACKTEST_PROFILE"
    assert entry["fill_model"] == K.CANONICAL["entry"]["model"]
    assert entry["profile_sha256"] == K.profile_hash()
    assert entry["not_a_search_parameter"] is True
    # 검증이 쓴 기준은 따로 남는다 — 관계를 잰 자이지 체결 모형이 아니다
    assert entry["validation_reference"]["entry"] == "ASK1"


def test_decision_cadence_belongs_to_the_canonical_profile():
    from framework import canonical as K
    policy = spec()["decision_opportunity_policy"]
    assert policy["owner"] == "CANONICAL_BACKTEST_PROFILE"
    assert policy["initial_binding"] == K.CANONICAL["decision"]["basis"]
    # 검증 격자와 백테스트 표본은 다르다. 그것을 숨기지 않는다
    assert policy["backtest_sampling_differs"] is True


def test_undecided_execution_binding_blocks_the_next_stage():
    """실행 결속이 미정이면 충실도 위반이 아니라 **모호**다. 넘어가지는 못한다 (§28)."""
    def blank(s):
        s["decision_opportunity_policy"] = {}
        s["entry_execution_semantics"] = {}
    result = audit_of(blank)
    assert result["fidelity_status"] == X.FIDELITY_AMBIGUOUS
    assert X.readiness(result)["readiness"] == X.BLOCKED_BY_FIDELITY_AMBIGUITY


def test_exit_rule_is_out_of_scope_but_horizon_is_fixed():
    policy = spec()["exit_policy"]
    assert policy["horizon_status"] == X.FIXED_BY_VALIDATION
    assert policy["exit_rule_status"] == X.OUT_OF_SCOPE
    assert policy["evaluation_horizon"] == "S30"


# ---- 준비도 -------------------------------------------------------------------

def test_a_preserved_spec_goes_to_implementation_not_straight_to_search():
    """바로 파라미터 탐색으로 가지 않는다 (§42)."""
    ready = X.readiness(X.fidelity_audit(spec(), revision(), results()))
    assert ready["readiness"] == X.READY_FOR_IMPLEMENTATION
    assert X.READY_FOR_PARAMETER_SEARCH in ready["next"]
    assert "아직 시작하지 않는다" in ready["not_yet"]


def test_a_violated_spec_is_invalid():
    result = audit_of(lambda s: s["signal_template"]["input"].update({"primitive_id": "vol_flow"}))
    assert X.readiness(result)["readiness"] == X.INVALID_SPECIFICATION


# ---- 이 단계가 하지 않는 것 -------------------------------------------------------

def test_specification_picks_no_values():
    """임계·평가 창·청산 어느 것도 값을 고르지 않는다."""
    built = spec()
    assert "임계값 선택" in built["not_done_here"]
    assert "파라미터 탐색" in built["not_done_here"]
    assert built["exit_policy"]["not_yet_decided"]
    template = {k: v for k, v in built["signal_template"].items()
                if not k.startswith("_")}
    numbers = [v for v in template.values() if isinstance(v, (int, float))]
    assert numbers == []


def test_promising_forbidden_axis_becomes_a_discovery_candidate():
    """탐색이 좋은 것을 찾아도 이 가설의 결과로 저장하지 않는다 (§33)."""
    boundary = spec()["search_boundary"]
    assert boundary["on_promising_forbidden_axis"] == X.SEARCH_DISCOVERY_CANDIDATE


def test_provenance_pins_every_upstream_hash():
    provenance = spec()["provenance"]
    for key in ("hypothesis_hash", "revision_hash", "grounding_hash",
                "validation_plan_hash", "catalog_hash", "capability_profile_hash"):
        assert provenance[key], key
