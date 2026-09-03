"""Hypothesis artifact → Grounding 전 mechanism specification artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .. import mechanism_specification as core
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..contracts.research import select_hypotheses


ARTIFACT_NAME = "mechanism_specification_artifact.json"


def run(evidence_path: Path, hypothesis_path: Path, output: Path, *,
        hypothesis_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """경제적 경로를 먼저 고정하고, 이 artifact 없이는 Grounding하지 않는다."""
    evidence = read_artifact(evidence_path, kind="evidence_package")
    hypotheses = read_artifact(hypothesis_path, kind="hypothesis_set")
    if hypotheses["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Hypothesis artifact의 Evidence 부모가 다르다")
    result = hypotheses["payload"]
    if result.get("state") != "READY_FOR_GROUNDING":
        raise ValueError(f"명세를 만들 수 없는 Hypothesis 상태: {result.get('state')}")
    payload = dict(result["payload"])
    novelty = result.get("novelty") or {}
    if novelty and novelty.get("status") != "NOT_REQUESTED":
        payload["novelty"] = novelty
    selected = select_hypotheses(payload, hypothesis_ids)
    units: dict[str, Any] = {}
    for hypothesis in selected.get("hypotheses") or []:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        try:
            specification = core.build(hypothesis)
            units[hypothesis_id] = {
                "state": specification["state"],
                "reason": specification["reason"],
                "mechanism_specification": specification,
            }
        except ValueError as exc:
            units[hypothesis_id] = {
                "state": core.INVALID_MECHANISM_SPECIFICATION,
                "reason": str(exc),
            }
    artifact = create_artifact("mechanism_specification_set", {
        "state": "MECHANISM_SPECIFICATION_COMPLETE" if units else "INVALID_INPUT",
        "selected_hypothesis_ids": [str(item.get("hypothesis_id"))
                                    for item in selected.get("hypotheses") or []],
        "units": units,
    }, parents={"evidence_package": evidence["artifact_id"],
                "hypothesis_set": hypotheses["artifact_id"]})
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact
