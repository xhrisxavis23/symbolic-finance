"""Evidence + Hypothesis artifact → Grounding과 ValidationPlan artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..agents import grounding
from ..agents.runtime import Runner
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..contracts.research import build_validation_plan, select_hypotheses
from .. import mechanism_specification


GROUNDING_ARTIFACT_NAME = "grounding_artifact.json"
PLAN_ARTIFACT_NAME = "validation_plan_artifact.json"


def run(evidence_path: Path, hypothesis_path: Path, mechanism_path: Path, output: Path, *,
        hypothesis_ids: Sequence[str] | None = None, runner: Runner | None = None) -> dict[str, Any]:
    evidence = read_artifact(evidence_path, kind="evidence_package")
    hypotheses_artifact = read_artifact(hypothesis_path, kind="hypothesis_set")
    mechanisms_artifact = read_artifact(mechanism_path, kind="mechanism_specification_set")
    if hypotheses_artifact["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Hypothesis artifact가 다른 Evidence artifact에서 만들어졌다")
    expected_mechanism_parents = {"evidence_package": evidence["artifact_id"],
                                  "hypothesis_set": hypotheses_artifact["artifact_id"]}
    if mechanisms_artifact["parents"] != expected_mechanism_parents:
        raise ValueError("Mechanism specification artifact의 입력 lineage가 다르다")
    hypothesis_result = hypotheses_artifact["payload"]
    if hypothesis_result.get("state") != "READY_FOR_GROUNDING":
        raise ValueError(f"Grounding 할 수 없는 Hypothesis 상태: {hypothesis_result.get('state')}")
    selection_payload = dict(hypothesis_result["payload"])
    # Novelty verdict는 Agent의 원문 payload 바깥에 기록된다. 여기서 함께 넘기지
    # 않으면 DUPLICATE_HYPOTHESIS가 Grounding·Search로 다시 흘러간다.
    if hypothesis_result.get("novelty", {}).get("status") != "NOT_REQUESTED":
        selection_payload["novelty"] = hypothesis_result["novelty"]
    selected = select_hypotheses(selection_payload, hypothesis_ids)
    mechanism_units = mechanisms_artifact["payload"].get("units") or {}
    selected_ids = [str(item.get("hypothesis_id")) for item in selected.get("hypotheses") or []]
    unavailable = [hypothesis_id for hypothesis_id in selected_ids
                   if mechanism_units.get(hypothesis_id, {}).get("state")
                   != mechanism_specification.READY_FOR_GROUNDING]
    if unavailable:
        result = {
            "payload": {"hypotheses": []},
            "validation": {"problems": ["Grounding 전 mechanism specification이 준비되지 않았다: "
                                        + ", ".join(unavailable)], "warnings": []},
            "state": "MECHANISM_SPECIFICATION_REQUIRED",
        }
    else:
        specifications = {hypothesis_id: mechanism_units[hypothesis_id]["mechanism_specification"]
                          for hypothesis_id in selected_ids}
        result = grounding.run(selected, evidence["payload"]["package"],
                               mechanism_specifications=specifications, runner=runner)
    grounded_artifact = create_artifact("grounding_set", {
        **result,
        "selected_hypothesis_ids": selected_ids,
    }, parents={"evidence_package": evidence["artifact_id"],
                "hypothesis_set": hypotheses_artifact["artifact_id"],
                "mechanism_specification_set": mechanisms_artifact["artifact_id"]})
    output = Path(output)
    write_artifact(output / GROUNDING_ARTIFACT_NAME, grounded_artifact)
    if result["state"] != "READY_FOR_VALIDATION_PLAN":
        return {"grounding": grounded_artifact, "validation_plan": None}
    plan = build_validation_plan(result["payload"], selected, lineage={
        "evidence_package": evidence["artifact_id"],
        "hypothesis_set": hypotheses_artifact["artifact_id"],
        "grounding_set": grounded_artifact["artifact_id"],
    })
    plan_artifact = create_artifact("validation_plan", plan, parents={
        "grounding_set": grounded_artifact["artifact_id"],
    })
    write_artifact(output / PLAN_ARTIFACT_NAME, plan_artifact)
    return {"grounding": grounded_artifact, "validation_plan": plan_artifact}


def revalidate(evidence_path: Path, hypothesis_path: Path, mechanism_path: Path, source_grounding_path: Path,
               output: Path) -> dict[str, Any]:
    """저장된 Grounding을 새 Agent 호출 없이 현재 계약으로 재판정한다."""
    evidence = read_artifact(evidence_path, kind="evidence_package")
    hypotheses_artifact = read_artifact(hypothesis_path, kind="hypothesis_set")
    mechanisms_artifact = read_artifact(mechanism_path, kind="mechanism_specification_set")
    source = read_artifact(source_grounding_path, kind="grounding_set")
    expected = {"evidence_package": evidence["artifact_id"],
                "hypothesis_set": hypotheses_artifact["artifact_id"]}
    if any(source["parents"].get(key) != value for key, value in expected.items()):
        raise ValueError("재검증 Grounding의 Evidence 또는 Hypothesis 부모가 다르다")
    expected_mechanisms = {**expected, "hypothesis_set": hypotheses_artifact["artifact_id"]}
    if mechanisms_artifact["parents"] != expected_mechanisms:
        raise ValueError("재검증 Mechanism specification의 입력 lineage가 다르다")
    selection_payload = dict(hypotheses_artifact["payload"]["payload"])
    if hypotheses_artifact["payload"].get("novelty", {}).get("status") != "NOT_REQUESTED":
        selection_payload["novelty"] = hypotheses_artifact["payload"]["novelty"]
    selected = select_hypotheses(
        selection_payload, source["payload"].get("selected_hypothesis_ids") or None)
    result = grounding.revalidate(source["payload"]["payload"], selected,
                                  evidence["payload"]["package"],
                                  source_artifact_id=source["artifact_id"])
    grounded_artifact = create_artifact("grounding_set", {
        **result,
        "selected_hypothesis_ids": [item.get("hypothesis_id")
                                    for item in selected.get("hypotheses") or []],
    }, parents={**expected, "mechanism_specification_set": mechanisms_artifact["artifact_id"]})
    output = Path(output)
    write_artifact(output / GROUNDING_ARTIFACT_NAME, grounded_artifact)
    if result["state"] != "READY_FOR_VALIDATION_PLAN":
        return {"grounding": grounded_artifact, "validation_plan": None}
    plan = build_validation_plan(result["payload"], selected, lineage={
        **expected, "grounding_set": grounded_artifact["artifact_id"],
    })
    plan_artifact = create_artifact("validation_plan", plan, parents={
        "grounding_set": grounded_artifact["artifact_id"],
    })
    write_artifact(output / PLAN_ARTIFACT_NAME, plan_artifact)
    return {"grounding": grounded_artifact, "validation_plan": plan_artifact}
