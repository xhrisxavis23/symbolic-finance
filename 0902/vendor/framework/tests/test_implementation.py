"""계약이 명세의 경계를 한 글자도 넘지 않았는가.

여기서 보는 것은 하나다 — **코드가 도는 것과 그 가설을 구현한 것은 다르다.**
문법이 맞고 백테스트가 돌아도 다른 가설을 실행하고 있을 수 있다.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from framework import (catalog, executable as X, implementation as I, mechanism_graph as M,
                       profit_target as PT, sample_condition as SC)
from .test_executable import manifest, results, revision, routing


@pytest.fixture
def spec():
    return X.build(revision(), routing(), manifest(), results())


@pytest.fixture
def template(spec):
    built = I.compile_contract(spec)
    built["contract_sha256"] = "test"
    return built


def signal_of(template):
    return template["entry_program"]["signal"]


# ---- 컴파일 (§13~§15) -----------------------------------------------------------

def test_compiler_copies_only_what_the_spec_says(spec, template):
    signal = signal_of(template)
    assert signal["op"] == "compare"
    assert signal["input"]["primitive_id"] == "book_imbalance"
    assert signal["comparator"] == ">"
    assert template["side"] == "LONG"
    assert template["profit_target"] == PT.canonical_target()


def test_profit_target_change_breaks_fidelity(spec, template):
    changed = copy.deepcopy(template)
    changed["profit_target"]["explicit_cost_bps_per_fill"] = 0.0
    report = I.fidelity_check(spec, changed)
    assert report["fidelity_status"] == X.FIDELITY_VIOLATED
    assert report["profit_target_mismatch_count"] == 1


def test_the_threshold_stays_unresolved(template):
    """여기서 채우면 탐색을 미리 한 것이다 (§4)."""
    value = signal_of(template)["value"]
    assert isinstance(value, str)
    assert value == f"{catalog.UNRESOLVED_PREFIX}theta_book_imbalance"


def test_compiler_preserves_a_composite_grounding_ast_and_every_parameter_slot():
    hypothesis = {"hypothesis_id": "H-COMPOSITE",
                  "profit_target": PT.canonical_target(),
                  "evidence_roles": {"ORDER_FLOW": "LATE_TRIGGER"},
                  "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01",
                                      "family": "ORDER_FLOW",
                                      "representative_feature": "ofi_depth_10"}]}
    expression = {"op": "all", "args": [
        {"op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
         "comparator": "<", "value": "UNRESOLVED:theta_ofi"},
        {"op": "compare", "input": {"op": "primitive", "primitive_id": "microprice_dev_bps"},
         "comparator": ">", "value": "UNRESOLVED:theta_microprice"},
    ]}
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
    composite_spec = X.build_from_grounding(hypothesis, grounded, {"catalog_sha256": "test"})
    compiled = I.compile_contract(composite_spec)
    assert signal_of(compiled) == expression
    assert sorted(compiled["parameter_interface"]) == ["theta_microprice", "theta_ofi"]
    assert I.fidelity_check(composite_spec, compiled)["fidelity_status"] == X.FIDELITY_PRESERVED
    bound = I.bind_parameters(compiled, {"theta_ofi": 0.2, "theta_microprice": 1.5})
    assert not I._unresolved(signal_of(bound))


def test_grounded_pending_cancel_policy_reaches_the_compiled_contract():
    source_graph = {
        "schema": M.SCHEMA_VERSION,
        "observables": [{"node_id": "O1", "evidence_id": "EV_ORDER_FLOW_01",
                         "available_at": "PENDING"}],
        "relations": [],
        "execution": {"entry_state": "DECISION", "pending_state": "PENDING",
                      "entry_action": "BID1_QUEUE",
                      "entry_lifecycle_policy": {"mode": "CANCEL_WHEN_ENTRY_SIGNAL_FALSE",
                                                 "pending_observable_ids": ["O1"]}},
        "prediction": {"input_ids": ["O1"], "evaluation_state": "FILLED",
                       "target_type": "PATH_TARGET", "direction": X.HIGHER},
        "catalog_bindings": [],
    }
    hypothesis = {
        "hypothesis_id": "H-LIFECYCLE",
        "profit_target": PT.canonical_target(),
        "evidence_roles": {"ORDER_FLOW": "LATE_TRIGGER"},
        "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01", "family": "ORDER_FLOW",
                            "representative_feature": "ofi_depth_10"}],
        "mechanism_graph": source_graph,
    }
    expression = {"op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
                  "comparator": "<", "value": "UNRESOLVED:theta_ofi"}
    grounded = {
        "operational_expression": expression,
        "operational_claim_ids": ["C1"],
        "mechanism_graph": {**source_graph, "catalog_bindings": [
            {"observable_id": "O1", "catalog_feature": "ofi_depth_10"}]},
        "claims": [
            {"claim_id": "C1", "grounding_status": "GROUNDED_DIRECT",
             "catalog_feature": "ofi_depth_10", "direction": X.LOWER},
            {"claim_id": "C2", "grounding_status": "VALIDATION_TARGET",
             "validation_target": "PATH_TARGET", "direction": X.HIGHER,
             "importance": "CORE", "supporting_evidence_ids": ["EV_ORDER_FLOW_01"]},
        ],
    }
    compiled = I.compile_contract(X.build_from_grounding(
        hypothesis, grounded, {"catalog_sha256": "test"}))
    assert compiled["entry_lifecycle"] == {
        "mode": "CANCEL_WHEN_ENTRY_SIGNAL_FALSE", "pending_observable_ids": ["O1"]}


def test_warmup_is_read_from_the_tree_not_defaulted(template):
    """rolling 통계가 없으면 데울 구간도 없다. 기본값을 넣는 것이 아니다."""
    assert template["entry_program"]["warmup_ticks"] == 0


def test_compiler_refuses_to_invent_a_warmup(spec):
    """명세에 없는 것을 컴파일러가 정하지 않는다 (§15)."""
    broken = copy.deepcopy(spec)
    broken["signal_template"] = {"op": "persistence",
                                 "condition": broken["signal_template"], "window": 5}
    with pytest.raises(ValueError, match="warmup"):
        I.compile_contract(broken)


def test_execution_binding_comes_from_the_spec(template):
    binding = template["execution_binding"]
    from framework import canonical as K
    assert binding["decision_opportunity"] == K.CANONICAL["decision"]["basis"]
    assert binding["decision_spacing_seconds"] is None      # 간격이 없다 = 매 틱
    from framework import canonical as K
    assert binding["entry_execution"] == K.CANONICAL["entry"]["model"]
    assert binding["evaluation_horizon"] == "S30"
    assert binding["exit_rule"] is None       # 이 단계가 만들지 않는다


def test_contract_is_the_shape_the_runtime_reads(template):
    """`contract.entry_signal` 이 `entry_program.signal` 을 읽는다."""
    from framework import contract
    assert "entry_program" in template
    assert "signal" in template["entry_program"]
    info = catalog.infer_expression_type(signal_of(template), allow_unresolved=True)
    assert info.value_type == "boolean"
    nodes = contract._unresolved_nodes(template["entry_program"])
    assert [name for name, _ in nodes] == ["UNRESOLVED:theta_book_imbalance"]


# ---- 파라미터 인터페이스 (§16~§19) ------------------------------------------------

def test_only_one_parameter_is_exposed(template):
    assert sorted(template["parameter_interface"]) == ["theta_book_imbalance"]
    item = template["parameter_interface"]["theta_book_imbalance"]
    assert item["status"] == X.SEARCHABLE
    assert item["value"] is None
    assert item["role"] == "signal_threshold"


def test_binding_replaces_the_value_without_touching_structure(template):
    bound = I.bind_parameters(template, {"theta_book_imbalance": 0.4})
    assert signal_of(bound)["value"] == 0.4
    assert I.signal_ast({k: v for k, v in signal_of(bound).items() if k != "value"}) \
        == I.signal_ast({k: v for k, v in signal_of(template).items() if k != "value"})
    assert signal_of(template)["value"].startswith(catalog.UNRESOLVED_PREFIX)  # 원본 보존


def test_an_unapproved_parameter_is_refused(template):
    """"나중에 쓸지도 몰라서" 만든 자리도 탐색 표면을 넓힌다 (§41)."""
    for name in ("theta_vol_flow", "persistence_n", "holding_time", "spread_limit"):
        with pytest.raises(ValueError, match=I.UNAPPROVED_PARAMETER):
            I.bind_parameters(template, {name: 0.4})


def test_a_non_numeric_binding_is_refused(template):
    with pytest.raises(ValueError, match="숫자"):
        I.bind_parameters(template, {"theta_book_imbalance": "q80"})


# ---- 코드가 유효한가 (§20) -------------------------------------------------------

def test_a_compiled_template_is_valid(template):
    result = I.implementation_validity(template)
    assert result["implementation_valid"] is True
    for key in I.RUNTIME_METRICS:
        assert result[key] == 0, key


def test_a_non_boolean_signal_is_invalid(template):
    broken = copy.deepcopy(template)
    broken["entry_program"]["signal"] = {"op": "primitive",
                                         "primitive_id": "book_imbalance"}
    result = I.implementation_validity(broken)
    assert result["implementation_valid"] is False
    assert result["status"] == I.IMPLEMENTATION_INVALID


def test_an_empty_execution_binding_is_invalid(template):
    broken = copy.deepcopy(template)
    broken["execution_binding"]["entry_execution"] = None
    result = I.implementation_validity(broken)
    assert result["execution_binding_error_count"] == 1


def test_an_interface_that_disagrees_with_the_tree_is_invalid(template):
    broken = copy.deepcopy(template)
    broken["parameter_interface"]["theta_extra"] = {"status": X.SEARCHABLE}
    assert I.implementation_validity(broken)["parameter_binding_error_count"] == 1


# ---- 의미가 보존됐는가 (§21~§32) -------------------------------------------------

def test_a_faithful_contract_preserves_every_metric(spec, template):
    result = I.fidelity_check(spec, template)
    assert result["fidelity_status"] == X.FIDELITY_PRESERVED
    for key in I.STRUCTURE_METRICS:
        assert result[key] == 0, key


def test_sample_condition_feature_is_not_counted_as_a_new_or_forbidden_feature(spec):
    conditioned = copy.deepcopy(spec)
    sample = SC.m0_price_recovery()
    conditioned["executable_sample_condition"] = sample
    conditioned["semantic_invariants"]["executable_sample_condition"] = sample
    conditioned["semantic_invariants"]["temporal_warmup_ticks"] = SC.warmup_ticks(sample)
    conditioned["signal_template"] = SC.combine(sample, conditioned["signal_template"])
    conditioned["forbidden_features"] = [
        *conditioned.get("forbidden_features", []),
        {"feature": "mid_return_5t_bps"},
    ]

    result = I.fidelity_check(conditioned, I.compile_contract(conditioned))

    assert result["fidelity_status"] == X.FIDELITY_PRESERVED
    assert result["new_feature_count"] == 0
    assert result["problems"] == []


def test_the_diff_covers_every_semantic_item(spec, template):
    items = {row["item"] for row in I.fidelity_check(spec, template)["diff"]}
    for expected in ("feature", "direction", "side", "signal AST",
                     "evaluation horizon", "decision opportunity",
                     "entry execution", "threshold", "search surface"):
        assert expected in items, expected


def test_ast_comparison_ignores_key_order_but_not_values():
    a = {"op": "compare", "comparator": ">", "value": "UNRESOLVED:t"}
    b = {"value": "UNRESOLVED:t", "op": "compare", "comparator": ">"}
    c = {"op": "compare", "comparator": "<", "value": "UNRESOLVED:t"}
    assert I.signal_ast(a) == I.signal_ast(b)
    assert I.signal_ast(a) != I.signal_ast(c)


def test_private_keys_do_not_affect_the_ast():
    """명세 쪽 `_parameter` 같은 주석 필드가 비교를 흔들면 안 된다."""
    a = {"op": "compare", "_parameter": {"name": "t"}}
    b = {"op": "compare"}
    assert I.signal_ast(a) == I.signal_ast(b)


# ---- 일부러 틀린 계약 (§33) ------------------------------------------------------

def test_every_forbidden_mutation_is_caught(spec, template):
    """감사가 안 잡으면 그 감사는 없는 것과 같다."""
    tests = I.mutation_tests(spec, template)
    assert len(tests) == 9
    missed = [m["mutation"] for m in tests if not m["caught"]]
    assert missed == [], missed
    assert {m["verdict"] for m in tests} == {"FAIL_AS_EXPECTED"}


@pytest.mark.parametrize("letter,metric", [
    ("A", "feature_mismatch_count"),
    ("B", "direction_mismatch_count"),
    ("C", "side_mismatch_count"),
    ("D", "context_reintroduced_count"),
    ("E", "new_temporal_operator_count"),
    ("F", "horizon_mismatch_count"),
    ("G", "decision_cadence_mismatch_count"),
    ("H", "entry_execution_mismatch_count"),
    ("I", "premature_parameter_binding_count"),
])
def test_each_mutation_trips_its_own_metric(spec, template, letter, metric):
    tests = {m["mutation"][0]: m for m in I.mutation_tests(spec, template)}
    assert metric in tests[letter]["metrics"], tests[letter]


# ---- 실행 점검 (§34~§37) ---------------------------------------------------------

def test_smoke_test_runs_the_engine_without_market_data(template):
    result = I.smoke_test(template)
    assert result["compare_operator_ok"] is True
    assert result["parameter_binding_ok"] is True
    assert result["pnl_computed"] is False
    assert result["market_data_access_count"] == 0
    assert result["terminal_oos_access_count"] == 0


def test_smoke_signal_matches_the_comparison(template):
    result = I.smoke_test(template, fixture_threshold=0.3)
    values = np.array(result["synthetic_feature_values"])
    assert result["signal"] == (values > 0.3).tolist()
    assert result["long_decisions"] == int((values > 0.3).sum())


def test_the_fixture_threshold_is_labelled_as_a_fixture(template):
    """단위 테스트 임계를 '고른 임계' 처럼 남기지 않는다 (§37)."""
    result = I.smoke_test(template)
    assert result["fixture_only"] is True
    assert "탐색 결과가 아니" in result["note"]


def test_the_stored_template_is_never_bound(template):
    """실행 점검이 원본 템플릿을 채워 놓으면 안 된다."""
    I.smoke_test(template)
    assert signal_of(template)["value"].startswith(catalog.UNRESOLVED_PREFIX)
    assert template["parameter_interface"]["theta_book_imbalance"]["value"] is None


# ---- 관문 (§38~§39) --------------------------------------------------------------

def gate_for(spec, template, **over):
    validity = over.get("validity") or I.implementation_validity(template)
    fidelity = over.get("fidelity") or I.fidelity_check(spec, template)
    mutations = over.get("mutations", I.mutation_tests(spec, template))
    smoke = over.get("smoke") or I.smoke_test(template)
    return I.readiness_gate(validity, fidelity, template, mutations, smoke)


def test_all_four_conditions_open_the_gate(spec, template):
    gate = gate_for(spec, template)
    assert gate["status"] == I.READY_FOR_PARAMETER_SEARCH
    assert gate["implementation_valid"] and gate["fidelity_preserved"]
    assert gate["parameter_interface_valid"] and gate["execution_binding_complete"]
    assert gate["still_unresolved"] == ["UNRESOLVED:theta_book_imbalance"]


def test_a_bound_template_does_not_pass_the_gate(spec, template):
    """값이 채워져 있으면 이 단계가 끝난 것이 아니다 (§56)."""
    bound = I.bind_parameters(template, {"theta_book_imbalance": 0.4})
    gate = gate_for(spec, bound, fidelity=I.fidelity_check(spec, bound))
    assert gate["status"] != I.READY_FOR_PARAMETER_SEARCH
    assert gate["parameter_interface_valid"] is False


def test_a_fidelity_violation_blocks_the_gate(spec, template):
    broken = copy.deepcopy(template)
    broken["side"] = "SHORT"
    gate = gate_for(spec, broken, fidelity=I.fidelity_check(spec, broken))
    assert gate["status"] == I.IMPLEMENTATION_FIDELITY_VIOLATION


def test_an_uncaught_mutation_blocks_the_gate(spec, template):
    """감사가 못 잡는 변형이 하나라도 있으면 넘어가지 않는다."""
    fake = [{"mutation": "X. 못 잡는 것", "caught": False, "metrics": []}]
    gate = gate_for(spec, template, mutations=fake)
    assert gate["status"] == I.IMPLEMENTATION_FIDELITY_VIOLATION
    assert gate["mutations_all_caught"] is False


def test_an_invalid_dsl_blocks_the_gate(spec, template):
    broken = copy.deepcopy(template)
    broken["execution_binding"]["entry_execution"] = None
    gate = gate_for(spec, broken)
    assert gate["status"] == I.IMPLEMENTATION_INVALID


# ---- 이 단계가 하지 않는 것 -------------------------------------------------------

def test_no_threshold_is_chosen_anywhere(template):
    assert "임계값" in template["not_included"]
    assert template["parameter_interface"]["theta_book_imbalance"]["value"] is None


def test_no_exit_rule_is_created(template):
    assert template["execution_binding"]["exit_rule"] is None


def test_provenance_carries_the_specification_hash(spec, template):
    assert template["provenance"]["executable_specification_sha256"] \
        == spec["specification_sha256"]
    assert template["provenance"]["compiler"] == I.COMPILER_VERSION
