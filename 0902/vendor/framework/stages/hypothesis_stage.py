"""Evidence artifact → Hypothesis artifact Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .. import hypothesis as core
from ..agents import hypothesis
from ..agents.runtime import Runner
from ..contracts.artifacts import artifact_reference, create_artifact, read_artifact, write_artifact


ARTIFACT_NAME = "hypothesis_artifact.json"


def run(evidence_path: Path, output: Path, *, runner: Runner | None = None,
        discovery_loss_context: Mapping[str, Any] | None = None,
        development_validation_feedback: Mapping[str, Any] | None = None,
        candidate_library: Mapping[str, Any] | None = None,
        research_lens: Mapping[str, Any] | None = None) -> dict[str, Any]:
    evidence = read_artifact(evidence_path, kind="evidence_package")
    context_payload = discovery_loss_context.get("payload") if discovery_loss_context is not None else None
    result = hypothesis.run(evidence["payload"]["package"], runner=runner,
                            discovery_loss_context=context_payload,
                            development_validation_feedback=(
                                development_validation_feedback.get("payload")
                                if development_validation_feedback is not None else None),
                            candidate_library=(candidate_library or {}).get("payload"),
                            research_lens=research_lens)
    parents = {"evidence_package": evidence["artifact_id"]}
    if discovery_loss_context is not None:
        parents["discovery_loss_context"] = str(discovery_loss_context["artifact_id"])
    if development_validation_feedback is not None:
        parents["development_validation_feedback"] = str(
            development_validation_feedback["artifact_id"])
    if candidate_library is not None:
        parents["candidate_library"] = str(candidate_library["artifact_id"])
    artifact = create_artifact("hypothesis_set", result, parents={
        **parents,
    })
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def revalidate(evidence_path: Path, source_hypothesis_path: Path, output: Path) -> dict[str, Any]:
    """기존 Agent 응답을 현재 출력 계약으로만 다시 판단한다.

    Agent를 재호출하거나 payload를 고치지 않는다. validator 수정 뒤에는 이전의
    `INVALID_OUTPUT`이 실제 가설 실패인지, 계약 이음새인지 구별할 수 있어야 한다.
    """
    evidence = read_artifact(evidence_path, kind="evidence_package")
    source = read_artifact(source_hypothesis_path, kind="hypothesis_set")
    if source["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("재검증할 Hypothesis artifact가 다른 Evidence artifact에서 만들어졌다")

    result = dict(source["payload"])
    payload, temporal_role_changes = hypothesis._canonicalize_temporal_evidence_roles(
        result.get("payload") or {}, evidence["payload"]["package"])
    result["payload"] = payload
    result["temporal_role_canonicalization"] = temporal_role_changes
    validation = core.validate_output(
        payload, evidence["payload"]["package"],
        config=core.AgentConfig(require_mechanism_graph=False))
    if validation["problems"]:
        state = "INVALID_OUTPUT"
    elif result.get("novelty", {}).get("status") == core.DUPLICATE_HYPOTHESIS:
        # 원문을 고치지 않는 재검증이라도 이미 확정된 중복 판정은 보존한다.
        # READY로 올리면 Grounding이 선택 가능한 가설이 없다는 모순으로 중단된다.
        state = core.DUPLICATE_HYPOTHESIS
    else:
        state = "READY_FOR_GROUNDING"
    result.update({
        "validation": validation,
        "state": state,
        "revalidated_from": {
            "artifact_id": source["artifact_id"],
            "previous_state": source["payload"].get("state"),
        },
    })
    artifact = create_artifact("hypothesis_set", result, parents={
        "evidence_package": evidence["artifact_id"],
    })
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact


def novelty_revalidate(evidence_path: Path, source_hypothesis_path: Path,
                       candidate_library: Mapping[str, Any], output: Path, *,
                       runner: Runner | None = None) -> dict[str, Any]:
    """기존 가설 문장을 보존한 채 Sandbox historical candidate library와만 재비교한다."""
    evidence = read_artifact(evidence_path, kind="evidence_package")
    source = read_artifact(source_hypothesis_path, kind="hypothesis_set")
    if source["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("재비교할 Hypothesis artifact가 다른 Evidence artifact에서 만들어졌다")
    result = hypothesis.novelty_only(source["payload"], candidate_library["payload"], runner=runner)
    parents = {"evidence_package": evidence["artifact_id"],
               "candidate_library": str(candidate_library["artifact_id"]),
               "source_hypothesis_set": source["artifact_id"]}
    artifact = create_artifact("hypothesis_set", result, parents=parents)
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    return artifact
