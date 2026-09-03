"""Executable Specification set artifact → Implementation contract set artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import implementation as compiler
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import implementation


ARTIFACT_NAME = "implementation_artifact.json"


def run(specification_path: Path, output: Path, *, runs_root: Path | None = None,
        allow_equivalent_for_refinement: bool = False,
        allow_equivalent_for_refinement_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    specifications = read_artifact(specification_path, kind="executable_specification_set")
    output = Path(output)
    allowed_ids = frozenset(str(value) for value in allow_equivalent_for_refinement_ids)
    units: dict[str, Any] = {}
    for hypothesis_id, specification in specifications["payload"].get("units", {}).items():
        if "spec" not in specification:
            units[str(hypothesis_id)] = {"state": specification.get("state"),
                                         "reason": specification.get("reason")}
            continue
        unit_output = output / "units" / str(hypothesis_id)
        result = implementation.run(specification["spec"], unit_output)
        if result.get("state") == compiler.READY_FOR_PARAMETER_SEARCH:
            equivalent = implementation.equivalent_contracts(
                result["template"],
                runs_root=(Path(runs_root) if runs_root is not None
                           else Path(__file__).resolve().parents[2] / "runs"),
                exclude_path=unit_output / "contract_template.json")
            if equivalent["matches"]:
                if allow_equivalent_for_refinement or str(hypothesis_id) in allowed_ids:
                    # 기본 entry 계약은 같아도 Discovery-only q별 guard surface를
                    # 재평가할 수 있다. 이는 새 알고리즘이 아니라 같은 계약의
                    # 명시적 refinement 재실행이며, Final registry는 여전히 같은
                    # execution identity로 중복 집계하지 않는다.
                    units[str(hypothesis_id)] = {
                        **result,
                        "re_evaluation_of_execution_identity": equivalent,
                        "reason": (
                            "기존 Sandbox entry·execution 계약과 같지만, 요청된 "
                            "q별 Discovery Entry Refinement를 재평가한다"),
                    }
                    continue
                units[str(hypothesis_id)] = {
                    "state": "EXECUTION_EQUIVALENT",
                    "reason": "기존 Sandbox 계약과 같은 entry·execution·parameter interface다",
                    "execution_identity": equivalent,
                    "source_implementation": result,
                }
                continue
        units[str(hypothesis_id)] = result
    artifact = create_artifact("implementation_set", {"state": "IMPLEMENTATION_COMPLETE", "units": units},
                               parents={"executable_specification_set": specifications["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
