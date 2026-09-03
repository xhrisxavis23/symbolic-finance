"""Mechanism specification이 Grounding과 실행 명세 사이를 보존하는가."""

from __future__ import annotations

import pytest

from framework import (mechanism_graph as M, mechanism_specification as S,
                       profit_target as PT)
from framework.contracts.artifacts import create_artifact, write_artifact
from framework.stages import (grounding_fidelity_stage, mechanism_specification_stage,
                              specification_stage)


def source_and_grounded(*, target="PATH_TARGET"):
    graph = {
        "schema": M.SCHEMA_VERSION,
        "observables": [{"node_id": "O1", "evidence_id": "EV_ORDER_FLOW_01",
                         "available_at": "DECISION"}],
        "relations": [],
        "execution": {"entry_state": "DECISION", "pending_state": "PENDING",
                      "entry_action": "BID1_QUEUE",
                      "entry_lifecycle_policy": {"mode": "HOLD_THROUGH",
                                                 "pending_observable_ids": []}},
        "prediction": {"input_ids": ["O1"], "evaluation_state": "FILLED",
                       "target_type": target, "direction": "HIGHER"},
        "catalog_bindings": [],
    }
    hypothesis = {
        "hypothesis_id": "H1",
        "profit_target": PT.canonical_target(),
        "evidence_basis": [{"evidence_id": "EV_ORDER_FLOW_01", "family": "ORDER_FLOW",
                            "representative_feature": "ofi_depth_10"}],
        "mechanism_graph": graph,
    }
    grounded = {
        "hypothesis_id": "H1",
        "operational_expression": {
            "op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
            "comparator": "<", "value": "UNRESOLVED:theta_ofi_depth_10",
        },
        "operational_claim_ids": ["C1"],
        "mechanism_graph": {**graph, "catalog_bindings": [
            {"observable_id": "O1", "catalog_feature": "ofi_depth_10"}]},
        "claims": [
            {"claim_id": "C1", "claim_type": "OBSERVABLE_STATE",
             "grounding_status": "GROUNDED_DIRECT", "catalog_feature": "ofi_depth_10",
             "direction": "LOWER"},
            {"claim_id": "C2", "claim_type": "PREDICTED_CONSEQUENCE",
             "grounding_status": "VALIDATION_TARGET", "validation_target": target,
             "direction": "HIGHER", "importance": "CORE",
             "supporting_evidence_ids": ["EV_ORDER_FLOW_01"],
             "validation_measure": "fill 뒤 S30 ASK1 target", "validation_reference": "matched control",
             "required_data": ["best_quotes"], "missing_capability": []},
        ],
    }
    return hypothesis, grounded


def test_matched_control_prediction_is_a_post_execution_diagnostic():
    hypothesis, grounded = source_and_grounded(target="MATCHED_CONTROL_TARGET")
    result = S.assess_grounding(S.build(hypothesis), hypothesis, grounded)
    assert result["execution_state"] == S.READY_FOR_EXECUTION
    assert result["grounded_contract"]["post_fill_prediction"]["reference"] == "matched control"
    assert result["grounded_contract"]["required_diagnostics"][0]["type"] == "MATCHED_CONTROL"


def test_mechanism_prediction_follows_the_graph_not_a_supporting_path_target():
    hypothesis, grounded = source_and_grounded(target="MATCHED_CONTROL_TARGET")
    grounded["claims"].append({
        "claim_id": "C3", "claim_type": "PREDICTED_CONSEQUENCE",
        "grounding_status": "VALIDATION_TARGET", "validation_target": "PATH_TARGET",
        "direction": "HIGHER", "importance": "SUPPORTING",
    })
    result = S.assess_grounding(S.build(hypothesis), hypothesis, grounded)
    assert result["grounded_contract"]["post_fill_prediction"]["claim_id"] == "C2"


def test_mechanism_refuses_to_drop_a_graph_bound_feature_from_the_expression():
    hypothesis, grounded = source_and_grounded()
    grounded["operational_expression"] = {
        "op": "compare", "input": {"op": "primitive", "primitive_id": "mid_return_5t_bps"},
        "comparator": "<", "value": "UNRESOLVED:theta_mid_return_5t_bps",
    }
    with pytest.raises(ValueError, match="binding feature"):
        S.assess_grounding(S.build(hypothesis), hypothesis, grounded)


def test_ready_mechanism_is_preserved_in_executable_specification(tmp_path):
    hypothesis, grounded = source_and_grounded()
    evidence = create_artifact("evidence_package", {"package": {"catalog_sha256": "test"}})
    hypotheses = create_artifact("hypothesis_set", {
        "state": "READY_FOR_GROUNDING", "payload": {"hypotheses": [hypothesis]}},
                                 parents={"evidence_package": evidence["artifact_id"],
                                          "candidate_library": "candidate-library"})
    evidence_path, hypothesis_path = tmp_path / "evidence.json", tmp_path / "hypothesis.json"
    for path, artifact in ((evidence_path, evidence), (hypothesis_path, hypotheses)):
        write_artifact(path, artifact)
    mechanisms = mechanism_specification_stage.run(evidence_path, hypothesis_path, tmp_path / "mechanism")
    grounding = create_artifact("grounding_set", {"payload": {"hypotheses": [grounded]}}, parents={
        "evidence_package": evidence["artifact_id"], "hypothesis_set": hypotheses["artifact_id"],
        "mechanism_specification_set": mechanisms["artifact_id"]})
    plan = create_artifact("validation_plan", {"hypothesis_ids": ["H1"]},
                           parents={"grounding_set": grounding["artifact_id"]})
    grounding_path, plan_path = tmp_path / "grounding.json", tmp_path / "plan.json"
    for path, artifact in ((grounding_path, grounding), (plan_path, plan)):
        write_artifact(path, artifact)
    fidelity = grounding_fidelity_stage.run(
        evidence_path, hypothesis_path,
        tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
        grounding_path, plan_path, tmp_path / "fidelity")
    specifications = specification_stage.run(
        evidence_path, hypothesis_path, grounding_path, plan_path, tmp_path / "specification",
        mechanism_path=tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
        fidelity_path=tmp_path / "fidelity" / grounding_fidelity_stage.ARTIFACT_NAME)

    assert mechanisms["payload"]["units"]["H1"]["state"] == S.READY_FOR_GROUNDING
    assert fidelity["payload"]["units"]["H1"]["state"] == S.READY_FOR_EXECUTION
    unit = specifications["payload"]["units"]["H1"]
    assert unit["mechanism_fidelity"]["state"] == "FIDELITY_PRESERVED"
    assert unit["spec"]["semantic_invariants"]["prediction_reference"] == "matched control"


def test_explicit_directions_produce_separate_executable_specifications(tmp_path):
    hypothesis, grounded = source_and_grounded()
    hypothesis["implementation_directions"] = [
        {"direction_id": "D1", "representation": "LEVEL"},
        {"direction_id": "D2", "representation": "PRE_ANCHOR_CHANGE"},
    ]
    level = grounded["operational_expression"]
    change = {
        "op": "compare",
        "input": {"op": "difference", "input": {
            "op": "primitive", "primitive_id": "ofi_depth_10"},
                  "lag": 1, "time_basis": "tick"},
        "comparator": "<", "value": "UNRESOLVED:theta_change_ofi_depth_10",
    }
    grounded["implementation_variants"] = [
        {"direction_id": "D1", "operational_expression": level,
         "operational_claim_ids": ["C1"], "operational_note": "수준 표현"},
        {"direction_id": "D2", "operational_expression": change,
         "operational_claim_ids": ["C1"], "operational_note": "직전 틱 대비 변화 표현"},
    ]
    evidence = create_artifact("evidence_package", {"package": {"catalog_sha256": "test"}})
    hypotheses = create_artifact("hypothesis_set", {
        "state": "READY_FOR_GROUNDING", "payload": {"hypotheses": [hypothesis]}},
        parents={"evidence_package": evidence["artifact_id"]})
    evidence_path, hypothesis_path = tmp_path / "evidence.json", tmp_path / "hypothesis.json"
    for path, artifact in ((evidence_path, evidence), (hypothesis_path, hypotheses)):
        write_artifact(path, artifact)
    mechanisms = mechanism_specification_stage.run(evidence_path, hypothesis_path, tmp_path / "mechanism")
    grounding = create_artifact("grounding_set", {"payload": {"hypotheses": [grounded]}}, parents={
        "evidence_package": evidence["artifact_id"], "hypothesis_set": hypotheses["artifact_id"],
        "mechanism_specification_set": mechanisms["artifact_id"]})
    plan = create_artifact("validation_plan", {"hypothesis_ids": ["H1"]},
                           parents={"grounding_set": grounding["artifact_id"]})
    grounding_path, plan_path = tmp_path / "grounding.json", tmp_path / "plan.json"
    for path, artifact in ((grounding_path, grounding), (plan_path, plan)):
        write_artifact(path, artifact)
    grounding_fidelity_stage.run(
        evidence_path, hypothesis_path,
        tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
        grounding_path, plan_path, tmp_path / "fidelity")
    specifications = specification_stage.run(
        evidence_path, hypothesis_path, grounding_path, plan_path, tmp_path / "specification",
        mechanism_path=tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
        fidelity_path=tmp_path / "fidelity" / grounding_fidelity_stage.ARTIFACT_NAME)

    units = specifications["payload"]["units"]
    assert sorted(units) == ["H1.D1", "H1.D2"]
    assert units["H1.D1"]["parent_hypothesis_id"] == "H1"
    assert units["H1.D2"]["spec"]["signal_template"]["input"]["op"] == "difference"
