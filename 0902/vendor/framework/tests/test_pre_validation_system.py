"""새 Pre-validation 시스템의 artifact 연결만 검사한다. 실제 모델은 부르지 않는다."""

from __future__ import annotations

from copy import deepcopy

import pytest

from framework.contracts.artifacts import ArtifactError, create_artifact, read_artifact, write_artifact
from framework.agents import grounding as grounding_agent, hypothesis as hypothesis_agent
from framework.stages import grounding_stage, hypothesis_stage, mechanism_specification_stage
from framework.tests import test_grounding as grounding_fixture
from framework.tests import test_hypothesis as hypothesis_fixture


class FakeRunner:
    def __init__(self):
        h1 = grounding_fixture.grounded()
        h2 = deepcopy(h1)
        h2["hypotheses"][0]["hypothesis_id"] = "H2"
        self.grounded = {"hypotheses": h1["hypotheses"] + h2["hypotheses"]}
        self.calls: list[str] = []
        self.prompts: dict[str, str] = {}

    def run(self, *, role, prompt, model, effort):
        self.calls.append(role)
        self.prompts[role] = prompt
        if role == "hypothesis_generation":
            return hypothesis_fixture.generated([
                hypothesis_fixture.good_hypothesis("H1"), hypothesis_fixture.good_hypothesis("H2"),
            ])
        if role == "hypothesis_evidence_audit":
            return dict(hypothesis_fixture.generated([
                hypothesis_fixture.good_hypothesis("H1"), hypothesis_fixture.good_hypothesis("H2"),
            ]), verdicts=[
                {"hypothesis_id": "H1", "verdict": "RETAIN", "reasons": []},
                {"hypothesis_id": "H2", "verdict": "RETAIN", "reasons": []},
            ])
        if role in ("grounding_generation", "grounding_semantic_audit"):
            return deepcopy(self.grounded)
        raise AssertionError(role)


def _evidence_artifact(tmp_path):
    artifact = create_artifact("evidence_package", {"package": grounding_fixture.package()})
    path = tmp_path / "evidence.json"
    write_artifact(path, artifact)
    return path


def test_artifact_rejects_content_changed_after_its_id_was_made(tmp_path):
    path = tmp_path / "artifact.json"
    value = create_artifact("sample", {"answer": 1})
    write_artifact(path, value)
    value["payload"]["answer"] = 2
    with pytest.raises(ArtifactError):
        write_artifact(path, value)


def test_empty_legacy_result_is_repaired_or_marked_invalid():
    weak = hypothesis_fixture.package([
        hypothesis_fixture.family(strength=0.01),
        hypothesis_fixture.family("EV_TRADE_ACTIVITY_01", "TRADE_ACTIVITY", "vol_flow", 0.01),
    ])
    calls = []
    empty_legacy = {"status": "NO_HYPOTHESIS", "hypotheses": []}

    def runner(prompt):
        calls.append(prompt)
        return empty_legacy

    result = hypothesis_agent.run(weak, runner=runner)
    assert result["state"] == "INVALID_OUTPUT"
    assert result["payload"]["status"] == "INVALID_OUTPUT"
    assert "NO_HYPOTHESIS" not in str(result["payload"])
    assert len(calls) == 3
    assert len(result["agent_invocations"]) == 3


def test_hypothesis_contract_repair_keeps_the_evidence_role_map():
    good = hypothesis_fixture.good_hypothesis()
    good["evidence_roles"] = {"TRADE_ACTIVITY": "PERSISTENT_CONTEXT",
                              "ORDER_FLOW": "LATE_TRIGGER"}
    bad = deepcopy(good)
    bad["evidence_roles"]["TRADE_ACTIVITY"] = "LATE_TRIGGER"

    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, *, role, prompt, model, effort):
            self.calls.append(role)
            if role == "hypothesis_generation":
                return hypothesis_fixture.generated([good])
            if role == "hypothesis_evidence_audit":
                return dict(hypothesis_fixture.generated([bad]), verdicts=[])
            if role == "hypothesis_contract_repair":
                return dict(hypothesis_fixture.generated([good]), verdicts=[])
            raise AssertionError(role)

    runner = Runner()
    result = hypothesis_agent.run(grounding_fixture.package(), runner=runner)
    assert result["state"] == "READY_FOR_GROUNDING"
    assert result["payload"]["hypotheses"][0]["evidence_roles"] == good["evidence_roles"]
    assert runner.calls == ["hypothesis_generation", "hypothesis_evidence_audit",
                            "hypothesis_contract_repair"]


def test_unavailable_price_path_is_not_advertised_to_the_agent(monkeypatch):
    pkg = hypothesis_fixture.package()
    pkg["price_path_context_required"] = True
    pkg["path_observation"] = {
        "horizon_key": "S30",
        "anchors": [{"anchor_id": "P1", "cohort": "PROFIT"}],
    }
    monkeypatch.setattr(
        hypothesis_agent.price_path, "access_manifest",
        lambda package: (_ for _ in ()).throw(FileNotFoundError("cache 없음")))
    runner = FakeRunner()
    result = hypothesis_agent.run(pkg, runner=runner)
    prompt = runner.prompts["hypothesis_generation"]
    assert result["state"] == "READY_FOR_GROUNDING"
    assert result["price_path_context"]["status"] == "UNAVAILABLE"
    assert '"price_path_tool"' not in prompt
    assert "PASS 0 — Discovery price-path observation" not in prompt


def test_grounding_contract_repair_adds_the_missing_target_direction():
    missing_direction = grounding_fixture.claim(
        "C2", claim_type="PREDICTED_CONSEQUENCE", epistemic_status="PREDICTED",
        grounding_status="VALIDATION_TARGET", catalog_feature=None, catalog_family=None,
        direction="NONE", validation_target="PATH_TARGET",
        validation_measure="anchor 이후 비용 차감 ASK1 경로", validation_reference="같은 상태 대조군")
    invalid = grounding_fixture.grounded([grounding_fixture.claim(), missing_direction])
    repaired = deepcopy(invalid)
    repaired["hypotheses"][0]["claims"][1]["direction"] = "HIGHER"

    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, *, role, prompt, model, effort):
            self.calls.append(role)
            if role in ("grounding_generation", "grounding_semantic_audit"):
                return deepcopy(invalid)
            if role == "grounding_contract_repair":
                return deepcopy(repaired)
            raise AssertionError(role)

    runner = Runner()
    result = grounding_agent.run(grounding_fixture.hypotheses(), grounding_fixture.package(), runner=runner)
    assert result["state"] == "READY_FOR_VALIDATION_PLAN"
    target = result["payload"]["hypotheses"][0]["claims"][1]
    assert target["validation_target"] == "PATH_TARGET" and target["direction"] == "HIGHER"
    assert runner.calls == ["grounding_generation", "grounding_semantic_audit",
                            "grounding_contract_repair"]


def test_all_hypotheses_are_grounded_and_planned_without_positional_selection(tmp_path):
    runner = FakeRunner()
    evidence_path = _evidence_artifact(tmp_path)
    hypotheses = hypothesis_stage.run(evidence_path, tmp_path / "hypothesis", runner=runner)
    assert hypotheses["payload"]["state"] == "READY_FOR_GROUNDING"
    mechanism_specification_stage.run(
        evidence_path, tmp_path / "hypothesis" / hypothesis_stage.ARTIFACT_NAME,
        tmp_path / "mechanism")

    outputs = grounding_stage.run(
        evidence_path, tmp_path / "hypothesis" / hypothesis_stage.ARTIFACT_NAME,
        tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
        tmp_path / "grounding", runner=runner,
    )
    grounded = outputs["grounding"]["payload"]
    plan = outputs["validation_plan"]["payload"]
    assert grounded["selected_hypothesis_ids"] == ["H1", "H2"]
    assert grounded["validation"]["hypothesis_coverage"] == {
        "requested": ["H1", "H2"], "grounded": ["H1", "H2"],
        "missing": [], "unexpected": [],
    }
    assert plan["not_executed"] is True
    assert plan["hypothesis_ids"] == ["H1", "H2"]
    assert {target["hypothesis_id"] for target in plan["targets"]} == {"H1", "H2"}
    assert all(target["status"] == "PLANNED" and target["result"] is None
               for target in plan["targets"])
    assert "mechanism_specifications" in runner.prompts["grounding_generation"]
    assert runner.calls == ["hypothesis_generation", "hypothesis_evidence_audit",
                            "grounding_generation", "grounding_semantic_audit"]


def test_grounding_refuses_hypothesis_from_another_evidence_artifact(tmp_path):
    runner = FakeRunner()
    first = _evidence_artifact(tmp_path)
    hypothesis_stage.run(first, tmp_path / "hypothesis", runner=runner)
    mechanism_specification_stage.run(
        first, tmp_path / "hypothesis" / hypothesis_stage.ARTIFACT_NAME,
        tmp_path / "mechanism")
    other = tmp_path / "other.json"
    write_artifact(other, create_artifact("evidence_package", {"package": grounding_fixture.package(),
                                                                  "different": True}))
    with pytest.raises(ValueError, match="다른 Evidence"):
        grounding_stage.run(other, tmp_path / "hypothesis" / hypothesis_stage.ARTIFACT_NAME,
                            tmp_path / "mechanism" / mechanism_specification_stage.ARTIFACT_NAME,
                            tmp_path / "grounding", runner=runner)
