"""Grounding 뒤 mechanism specification 보존 검사 Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .. import grounding as grounding_core, mechanism_specification as core
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact


ARTIFACT_NAME = "grounding_mechanism_fidelity_artifact.json"


def run(evidence_path: Path, hypothesis_path: Path, mechanism_path: Path,
        grounding_path: Path, plan_path: Path, output: Path, *,
        hypothesis_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Grounding이 앞 명세의 뜻을 보존했는지와 코드 진입 가능성을 분리해 기록한다."""
    evidence = read_artifact(evidence_path, kind="evidence_package")
    hypotheses = read_artifact(hypothesis_path, kind="hypothesis_set")
    mechanisms = read_artifact(mechanism_path, kind="mechanism_specification_set")
    grounding = read_artifact(grounding_path, kind="grounding_set")
    plan = read_artifact(plan_path, kind="validation_plan")
    expected_hypotheses = {"evidence_package": evidence["artifact_id"]}
    expected_mechanisms = {**expected_hypotheses, "hypothesis_set": hypotheses["artifact_id"]}
    expected_grounding = {**expected_mechanisms,
                          "mechanism_specification_set": mechanisms["artifact_id"]}
    # Hypothesis 생성은 같은 Evidence에 candidate library·loss context 같은 읽기 전용
    # 부모를 함께 기록할 수 있다. Grounding 단계도 Evidence parent만 고정하므로, 여기서
    # 옛 단일-parent 형태를 완전 일치로 요구하면 정상 H→Grounding 경로가 끊긴다.
    if hypotheses["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Hypothesis artifact의 입력 lineage가 다르다")
    if mechanisms["parents"] != expected_mechanisms:
        raise ValueError("Mechanism specification의 입력 lineage가 다르다")
    if grounding["parents"] != expected_grounding:
        raise ValueError("Grounding artifact의 입력 lineage가 다르다")
    if plan["parents"].get("grounding_set") != grounding["artifact_id"]:
        raise ValueError("ValidationPlan의 Grounding 부모가 다르다")

    source = {str(item.get("hypothesis_id")): item
              for item in hypotheses["payload"]["payload"].get("hypotheses") or []}
    grounded = {str(item.get("hypothesis_id")): item
                for item in grounding["payload"]["payload"].get("hypotheses") or []}
    planned_ids = tuple(str(value) for value in plan["payload"].get("hypothesis_ids") or [])
    selected_ids = planned_ids if hypothesis_ids is None else tuple(str(value) for value in hypothesis_ids)
    unknown = [value for value in selected_ids if value not in planned_ids]
    if unknown:
        raise ValueError("ValidationPlan에 없는 hypothesis_id: " + ", ".join(unknown))

    units: dict[str, Any] = {}
    mechanism_units = mechanisms["payload"].get("units") or {}
    for hypothesis_id in selected_ids:
        hypothesis = source.get(hypothesis_id)
        grounded_item = grounded.get(hypothesis_id)
        mechanism_unit = mechanism_units.get(hypothesis_id)
        if hypothesis is None or grounded_item is None or mechanism_unit is None:
            raise ValueError(f"Grounding fidelity source가 없다: {hypothesis_id}")
        if mechanism_unit.get("state") != core.READY_FOR_GROUNDING:
            units[hypothesis_id] = {
                "state": mechanism_unit.get("state"),
                "reason": mechanism_unit.get("reason"),
            }
            continue
        try:
            variants: dict[str, Any] = {}
            for variant in grounding_core.implementation_variants(grounded_item):
                direction_id = str(variant.get("direction_id") or "")
                if not direction_id:
                    raise ValueError("Grounding implementation variant에 direction_id가 없다")
                materialized = grounding_core.materialize_implementation_variant(grounded_item, variant)
                fidelity = core.assess_grounding(mechanism_unit["mechanism_specification"],
                                                 hypothesis, materialized)
                variants[direction_id] = {
                    "state": fidelity["execution_state"],
                    "reason": fidelity["execution_reason"],
                    "grounding_fidelity": fidelity,
                    "implementation_variant": variant,
                }
            if not variants:
                raise ValueError("Grounding에서 실행할 implementation variant가 없다")
            first = next(iter(variants.values()))
            units[hypothesis_id] = {
                "state": (core.READY_FOR_EXECUTION if all(
                    item["state"] == core.READY_FOR_EXECUTION for item in variants.values())
                    else first["state"]),
                "reason": first["reason"],
                # v1 소비자는 한 canonical 표현을 계속 읽을 수 있다.
                "grounding_fidelity": first["grounding_fidelity"],
                "variants": variants,
            }
        except ValueError as exc:
            units[hypothesis_id] = {
                "state": "GROUNDING_FIDELITY_VIOLATION",
                "reason": str(exc),
            }
    artifact = create_artifact("grounding_mechanism_fidelity_set", {
        "state": "GROUNDING_FIDELITY_COMPLETE" if units else "NO_GROUNDED_HYPOTHESIS",
        "units": units,
    }, parents={**expected_grounding, "grounding_set": grounding["artifact_id"],
                "validation_plan": plan["artifact_id"]})
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact
