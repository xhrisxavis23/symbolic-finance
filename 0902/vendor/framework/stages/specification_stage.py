"""Grounding artifact → Executable Specification set artifact Stage."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

from .. import executable, grounding as grounding_core, mechanism_specification
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import specification


ARTIFACT_NAME = "specification_artifact.json"


def run(evidence_path: Path, hypothesis_path: Path, grounding_path: Path, plan_path: Path,
        output: Path, *, hypothesis_ids: Sequence[str] | None = None,
        mechanism_path: Path | None = None, fidelity_path: Path | None = None) -> dict[str, Any]:
    """옛 경로형 Validation을 거치지 않고 Grounding prediction으로 명세를 만든다."""
    evidence = read_artifact(evidence_path, kind="evidence_package")
    hypotheses = read_artifact(hypothesis_path, kind="hypothesis_set")
    grounding = read_artifact(grounding_path, kind="grounding_set")
    plan = read_artifact(plan_path, kind="validation_plan")
    if hypotheses["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Hypothesis artifact의 Evidence 부모가 다르다")
    expected_grounding_parents = {"evidence_package": evidence["artifact_id"],
                                  "hypothesis_set": hypotheses["artifact_id"]}
    if mechanism_path is not None:
        expected_grounding_parents["mechanism_specification_set"] = "__MECHANISM_ARTIFACT_ID__"
    if any(grounding["parents"].get(key) != value for key, value in expected_grounding_parents.items()
           if value != "__MECHANISM_ARTIFACT_ID__"):
        raise ValueError("Grounding artifact의 입력 lineage가 다르다")
    if plan["parents"].get("grounding_set") != grounding["artifact_id"]:
        raise ValueError("ValidationPlan의 Grounding 부모가 다르다")
    mechanisms = None
    if mechanism_path is not None:
        mechanisms = read_artifact(mechanism_path, kind="mechanism_specification_set")
        expected_mechanism_parents = {
            "evidence_package": evidence["artifact_id"],
            "hypothesis_set": hypotheses["artifact_id"],
        }
        if mechanisms["parents"] != expected_mechanism_parents:
            raise ValueError("Mechanism specification의 입력 lineage가 다르다")
        if grounding["parents"].get("mechanism_specification_set") != mechanisms["artifact_id"]:
            raise ValueError("Grounding artifact의 Mechanism specification 부모가 다르다")
        if fidelity_path is None:
            raise ValueError("Mechanism specification 경로의 Code 명세에는 Grounding fidelity가 필요하다")
    fidelities = None
    if fidelity_path is not None:
        if mechanisms is None:
            raise ValueError("Grounding fidelity에는 Mechanism specification이 필요하다")
        fidelities = read_artifact(fidelity_path, kind="grounding_mechanism_fidelity_set")
        expected_fidelity_parents = {
            "evidence_package": evidence["artifact_id"],
            "hypothesis_set": hypotheses["artifact_id"],
            "mechanism_specification_set": mechanisms["artifact_id"],
            "grounding_set": grounding["artifact_id"],
            "validation_plan": plan["artifact_id"],
        }
        if fidelities["parents"] != expected_fidelity_parents:
            raise ValueError("Grounding fidelity의 입력 lineage가 다르다")
    source = {str(item.get("hypothesis_id")): item
              for item in hypotheses["payload"]["payload"].get("hypotheses") or []}
    grounded = {str(item.get("hypothesis_id")): item
                for item in grounding["payload"]["payload"].get("hypotheses") or []}
    planned_ids = tuple(str(value) for value in plan["payload"].get("hypothesis_ids") or [])
    if hypothesis_ids is None:
        selected_ids = planned_ids
    else:
        selected_ids = tuple(str(value) for value in hypothesis_ids)
        unknown = [value for value in selected_ids if value not in planned_ids]
        if unknown:
            raise ValueError("ValidationPlan에 없는 hypothesis_id: " + ", ".join(unknown))
    output = Path(output)
    units: dict[str, Any] = {}
    for hypothesis_id in selected_ids:
        hypothesis = source.get(str(hypothesis_id))
        grounded_item = grounded.get(str(hypothesis_id))
        if hypothesis is None or grounded_item is None:
            raise ValueError(f"Specification source가 없다: {hypothesis_id}")
        fidelity_unit = ((fidelities["payload"].get("units") or {}).get(str(hypothesis_id))
                         if fidelities is not None else None)
        if fidelities is not None and fidelity_unit is None:
            raise ValueError(f"Grounding fidelity source가 없다: {hypothesis_id}")
        direction_metadata = {
            str(item.get("direction_id")): copy.deepcopy(dict(item))
            for item in hypothesis.get("implementation_directions") or []
            if isinstance(item, dict) and item.get("direction_id")
        }
        for variant in grounding_core.implementation_variants(grounded_item):
            direction_id = str(variant.get("direction_id") or "")
            candidate_id = (f"{hypothesis_id}.{direction_id}"
                            if variant.get("explicit") else str(hypothesis_id))
            if candidate_id in units:
                raise ValueError(f"Specification candidate_id가 중복됐다: {candidate_id}")
            variant_fidelity = None
            if fidelity_unit is not None:
                variant_fidelity = ((fidelity_unit.get("variants") or {}).get(direction_id)
                                    or (fidelity_unit.get("grounding_fidelity")
                                        if not variant.get("explicit") else None))
                if variant_fidelity is None:
                    units[candidate_id] = {
                        "state": "GROUNDING_FIDELITY_MISSING",
                        "reason": f"{hypothesis_id}/{direction_id} Grounding fidelity가 없다",
                    }
                    continue
                if variant_fidelity.get("state") != mechanism_specification.READY_FOR_EXECUTION:
                    units[candidate_id] = {
                        "state": variant_fidelity.get("state"),
                        "reason": variant_fidelity.get("reason"),
                        "grounding_fidelity": variant_fidelity.get("grounding_fidelity"),
                    }
                    continue
            variant_hypothesis = copy.deepcopy(hypothesis)
            variant_hypothesis["hypothesis_id"] = candidate_id
            variant_hypothesis["parent_hypothesis_id"] = str(hypothesis_id)
            variant_hypothesis["implementation_direction"] = direction_metadata.get(direction_id, {
                "direction_id": direction_id,
                "representation": "LEGACY_CANONICAL",
            })
            variant_grounded = grounding_core.materialize_implementation_variant(grounded_item, variant)
            variant_grounded["hypothesis_id"] = candidate_id
            try:
                unit = specification.run(
                    variant_hypothesis, variant_grounded, evidence["payload"]["package"],
                    output / "units" / candidate_id)
                unit["parent_hypothesis_id"] = str(hypothesis_id)
                unit["implementation_direction"] = variant_hypothesis["implementation_direction"]
                if variant_fidelity is not None:
                    fidelity = mechanism_specification.audit_execution_specification(
                        variant_fidelity["grounding_fidelity"], unit["spec"])
                    if fidelity["state"] != "FIDELITY_PRESERVED":
                        units[candidate_id] = {
                            "state": "MECHANISM_FIDELITY_VIOLATION",
                            "reason": "; ".join(fidelity["problems"]),
                            "grounding_fidelity": variant_fidelity["grounding_fidelity"],
                            "mechanism_fidelity": fidelity,
                        }
                        continue
                    unit["grounding_fidelity"] = variant_fidelity["grounding_fidelity"]
                    unit["mechanism_fidelity"] = fidelity
                units[candidate_id] = unit
            except executable.UnsupportedTemporalGrounding as exc:
                units[candidate_id] = {
                    "state": "UNIMPLEMENTABLE_TEMPORAL_ENTRY",
                    "reason": str(exc),
                    "parent_hypothesis_id": str(hypothesis_id),
                }
    state = "SPECIFICATION_COMPLETE" if units else "NO_GROUNDED_HYPOTHESIS"
    parents = {"evidence_package": evidence["artifact_id"],
               "hypothesis_set": hypotheses["artifact_id"],
               "grounding_set": grounding["artifact_id"],
               "validation_plan": plan["artifact_id"]}
    if mechanisms is not None:
        parents["mechanism_specification_set"] = mechanisms["artifact_id"]
    if fidelities is not None:
        parents["grounding_mechanism_fidelity_set"] = fidelities["artifact_id"]
    artifact = create_artifact("executable_specification_set", {"state": state, "units": units},
                               parents=parents)
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
