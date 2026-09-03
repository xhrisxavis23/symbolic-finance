"""Implementation과 Specification artifact → Parameter search/lock set artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import TICK_ROOT, read_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import parameter_search
from ..modules.schedule import ResearchSchedule


ARTIFACT_NAME = "parameter_search_artifact.json"


def run(specification_path: Path, implementation_path: Path, refinement_path: Path,
        evidence_path: Path, output: Path, *,
        schedule: ResearchSchedule, root: Path = TICK_ROOT) -> dict[str, Any]:
    specifications = read_artifact(specification_path, kind="executable_specification_set")
    implementations = read_artifact(implementation_path, kind="implementation_set")
    refinements = read_artifact(refinement_path, kind="entry_refinement_set")
    evidence = read_artifact(evidence_path, kind="evidence_package")
    if implementations["parents"].get("executable_specification_set") != specifications["artifact_id"]:
        raise ValueError("Implementation과 Specification의 lineage가 다르다")
    if specifications["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Specification과 Evidence의 lineage가 다르다")
    expected_refinement_parents = {
        "executable_specification_set": specifications["artifact_id"],
        "implementation_set": implementations["artifact_id"],
        "evidence_package": evidence["artifact_id"],
    }
    if refinements["parents"] != expected_refinement_parents:
        raise ValueError("Entry refinement의 lineage가 다르다")
    discovery_dates = tuple(evidence["payload"].get("profile_dates") or [])
    schedule.validate(discovery_dates)
    symbols = evidence["payload"].get("profile_symbols") or []
    if not symbols:
        raise ValueError("Evidence artifact에 Profile 종목이 없다")
    output = Path(output)
    units: dict[str, Any] = {}
    for hypothesis_id, specification in specifications["payload"].get("units", {}).items():
        if "spec" not in specification:
            units[str(hypothesis_id)] = {"state": specification.get("state"),
                                         "reason": specification.get("reason")}
            continue
        implementation_unit = implementations["payload"].get("units", {}).get(str(hypothesis_id))
        if implementation_unit is None:
            raise ValueError(f"Parameter Search의 Implementation이 없다: {hypothesis_id}")
        if implementation_unit.get("state") != "READY_FOR_PARAMETER_SEARCH":
            units[str(hypothesis_id)] = {"state": "IMPLEMENTATION_NOT_READY"}
            continue
        refinement_unit = refinements["payload"].get("units", {}).get(str(hypothesis_id))
        if refinement_unit is None:
            raise ValueError(f"Parameter Search의 Entry refinement가 없다: {hypothesis_id}")
        # Refinement Agent가 주 신호를 교체한 경우에만 그 새 명세를 Search로 넘긴다.
        effective_specification = refinement_unit.get("refined_specification") or specification["spec"]
        variants = refinement_unit.get("entry_guard_variants")
        expression_variants = refinement_unit.get("entry_expression_variants")
        existing = _existing_unit(output / "units" / str(hypothesis_id), effective_specification,
                                  refinement_unit.get("entry_guards") or [],
                                  entry_guard_variants=variants,
                                  entry_expression=refinement_unit.get("entry_expression"),
                                  entry_expression_variants=expression_variants)
        units[str(hypothesis_id)] = existing if existing is not None else parameter_search.run(
            effective_specification, symbols=symbols, schedule=schedule,
            output=output / "units" / str(hypothesis_id), root=Path(root),
            entry_guards=refinement_unit.get("entry_guards") or [],
            entry_guard_variants=variants,
            entry_expression=refinement_unit.get("entry_expression"),
            entry_expression_variants=expression_variants, entry_refinement={
                key: value for key, value in refinement_unit.items()
                if key in {"primary_override", "entry_guards", "entry_guard_variants",
                           "entry_expression", "entry_expression_variants"}})
    artifact = create_artifact("parameter_search_set", {"state": "PARAMETER_SEARCH_COMPLETE", "units": units},
                               parents={"executable_specification_set": specifications["artifact_id"],
                                        "implementation_set": implementations["artifact_id"],
                                        "entry_refinement_set": refinements["artifact_id"],
                                        "evidence_package": evidence["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact


def _existing_unit(output: Path, specification: dict[str, Any],
                   entry_guards: list[dict[str, Any]], *,
                   entry_guard_variants: dict[str, list[dict[str, Any]]] | None = None,
                   entry_expression: dict[str, Any] | None = None,
                   entry_expression_variants: dict[str, dict[str, Any] | None] | None = None,
                   ) -> dict[str, Any] | None:
    """중단된 Stage를 다시 열 때 동일 명세의 완료 후보만 재사용한다."""
    required = {
        "plan": output / "parameter_search_plan.json",
        "selection": output / "selected_parameter.json",
        "audit": output / "search_audit.json",
        "amended": output / "executable_specification_amended.json",
        "template": output / "contract_template.json",
    }
    if not all(path.is_file() for path in required.values()):
        return None
    plan = read_json(required["plan"])
    if plan.get("specification_sha256") != specification.get("specification_sha256"):
        return None
    if plan.get("selection_mode") != parameter_search.SELECTION_MODE:
        # 예전 30초 anchor Search 결과는 V9 full-tick execution 선택과 같은 결과가 아니다.
        return None
    if entry_guard_variants is not None or entry_expression_variants is not None:
        # branch Search는 q별 guard까지 계약이다. 이전 공통-guard 결과를 재사용하면
        # Search가 실제로 비교한 entry surface가 달라진다.
        branch = plan.get("entry_refinement_branches") or {}
        saved = {
            str(item.get("candidate_id")): item.get("entry_guards") or []
            for item in branch.get("candidates") or []
        }
        expected = ({str(candidate_id): [dict(guard) for guard in guards]
                     for candidate_id, guards in entry_guard_variants.items()}
                    if entry_guard_variants is not None else saved)
        saved_expressions = {
            str(item.get("candidate_id")): item.get("entry_expression")
            for item in branch.get("candidates") or []
        }
        expected_expressions = (entry_expression_variants
                                if entry_expression_variants is not None
                                else saved_expressions)
        if (branch.get("mode") != "Q_SPECIFIC_DISCOVERY_ENTRY"
                or saved != expected or saved_expressions != expected_expressions):
            return None
    selection = read_json(required["selection"])
    confirmation_path, lock_path = output / "confirmation_result.json", output / "parameter_lock.json"
    confirmation = read_json(confirmation_path) if confirmation_path.is_file() else None
    lock = read_json(lock_path) if lock_path.is_file() else None
    status = (lock.get("status") if lock else
              confirmation.get("status") if confirmation else selection.get("status"))
    if not status:
        return None
    template = read_json(required["template"])
    saved_guards = [dict(guard) for guard in template.get("entry_guards") or []]
    expected_guards = [dict(guard) for guard in entry_guards]
    if entry_guard_variants is None and saved_guards != expected_guards:
        return None
    saved_expression = (template.get("entry_program") or {}).get("signal")
    if entry_expression_variants is None and entry_expression is not None \
            and saved_expression != entry_expression:
        return None
    return {
        "state": status,
        "plan": plan,
        "selection": selection,
        "confirmation": confirmation,
        "lock": lock,
        "audit": read_json(required["audit"]),
        "amended_specification": read_json(required["amended"]),
        "parameterized_template": template,
        "candidate_summary": parameter_search.candidate_summary_from_output(output),
        "entry_guards": [dict(guard) for guard in template.get("entry_guards") or []],
        "entry_guard_variants": entry_guard_variants,
        "entry_expression": saved_expression if entry_expression is not None else None,
        "entry_expression_variants": entry_expression_variants,
    }
