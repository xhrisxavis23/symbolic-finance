"""Specification 직전까지의 단일 시스템 Workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..agents.runtime import Runner
from ..config import clean, execution_workers, now_utc, run_manifest, write_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from .. import hypothesis as hypothesis_core, sample_condition as SC
from ..modules import candidate_library, evidence_cache, mechanism_lenses, research_capacity
from ..modules.profile import ProfileSelection
from ..stages import (evidence_stage, execution_anchor_stage, grounding_stage,
                      hypothesis_stage, mechanism_specification_stage)


@dataclass(frozen=True)
class PreValidationRequest:
    profile: ProfileSelection
    hypothesis_ids: tuple[str, ...] = ()
    evidence_source: Path | None = None
    refresh_price_path_grounding: bool = False
    discovery_loss_context: Path | None = None
    development_validation_feedback: Path | None = None
    research_lens_id: str | None = None
    # Discovery loop가 첫 시점에 고정한 historical candidate snapshot. 후속 lens가
    # 병렬 연구의 새 산출물 때문에 history를 다시 열지 않게 한다.
    candidate_library_source: Path | None = None
    # Research 안에서 호출되면 부모 run이 replay slot의 소유자다. 따라서 여기서
    # anchor replay를 먼저 열어도 뒤 단계가 별도 slot을 소비하지 않는다.
    replay_owner: Path | None = None


PROGRESS_FILE = "pre_validation_progress.json"


def _write_progress(output: Path, stage: str, refs: dict[str, str] | None = None, *,
                    terminal_state: str | None = None) -> None:
    """긴 Agent Pre-validation 실행의 현재 Stage를 별도 원장에 남긴다."""
    write_json(Path(output) / PROGRESS_FILE, {
        "schema": "pre_validation_progress.v1",
        "updated_at": now_utc(),
        "state": "COMPLETE" if terminal_state is not None else "IN_PROGRESS",
        "stage": stage,
        "terminal_state": terminal_state,
        "owner_pid": os.getpid(),
        "artifacts": dict(refs or {}),
        "execution_workers": execution_workers(),
    })


def _capacity_waiting(request: PreValidationRequest, output: Path,
                      capacity: dict[str, Any]) -> dict[str, Any]:
    """새 execution-anchor replay를 열기 전에 남기는 terminal 기록."""
    state = "CAPACITY_WAITING"
    manifest = create_artifact("pre_validation_run", {
        "state": state,
        "request": {
            "profile": request.profile.as_input(),
            "hypothesis_ids": list(request.hypothesis_ids),
            "evidence_source": None,
            "replay_owner": (str(request.replay_owner)
                             if request.replay_owner is not None else None),
        },
        "artifacts": {},
        "next_stage": None,
        "validation_executed": False,
        "research_capacity_reservation": dict(capacity),
    })
    write_artifact(Path(output) / "pre_validation_run.json", manifest)
    write_json(Path(output) / "run_manifest.json", run_manifest(output, {
        "workflow": "pre_validation.v2", "state": state, "artifacts": {},
    }))
    _write_progress(output, "TERMINAL", {}, terminal_state=state)
    return {"state": state, "artifacts": {}, "run": manifest,
            "grounding": None, "capacity": capacity}


def _refresh_price_path_grounding(evidence: dict[str, Any], selection: ProfileSelection,
                                  output: Path, source_path: Path) -> tuple[dict[str, Any], Path]:
    """과거 공유 Evidence를 보존한 채 현재의 저장-anchor 경로 관측 정책만 붙인다.

    이 함수는 Feature Profile, cohort, Evidence 값 또는 execution replay를 다시
    계산하지 않는다. 과거 artifact가 cache 부재 때문에 경로 관측을 선택 사항으로
    남긴 경우에만, 같은 Profile의 이미 저장된 anchor를 raw quote fallback으로 열도록
    파생 artifact를 만든다.
    """
    package = dict((evidence.get("payload") or {}).get("package") or {})
    anchors = (package.get("path_observation") or {}).get("anchors") or []
    if package.get("price_path_context_required", True) or not anchors:
        return evidence, Path(source_path)
    profile_store = Path(selection.store)
    discovery_profit_root = (
        Path(selection.discovery_profit_root)
        if selection.discovery_profit_root is not None else profile_store / "profit")
    package["price_path_context_required"] = True
    package["discovery_profit_root"] = (
        str(discovery_profit_root) if discovery_profit_root.is_dir() else None)
    package["price_path_grounding_policy"] = {
        "schema": "stored_anchor_path_grounding_refresh.v1",
        "source_evidence_package": str(evidence["artifact_id"]),
        "allowed_anchor_source": "SAVED_FEATURE_PROFILE_ONLY",
        "cache_policy": "CURRENT_DIRECTION_CACHE_OR_RAW_TICK_ANCHOR_WINDOW",
        "materializes_profit_cache": False,
    }
    payload = dict(evidence["payload"])
    payload["package"] = package
    refreshed = create_artifact(
        "evidence_package", payload,
        parents={**dict(evidence.get("parents") or {}),
                 "source_evidence_package": str(evidence["artifact_id"])})
    target = Path(output) / "01_evidence" / evidence_stage.ARTIFACT_NAME
    write_artifact(target, refreshed)
    return refreshed, target


def run(request: PreValidationRequest, output: Path, *, runner: Runner | None = None) -> dict[str, Any]:
    """Evidence → Hypothesis → mechanism specification → Grounding → ValidationPlan."""
    output = Path(output)
    _write_progress(output, "EVIDENCE")
    evidence_source = (Path(request.evidence_source) if request.evidence_source is not None
                       else evidence_cache.find(request.profile))
    if evidence_source is not None:
        evidence_path = Path(evidence_source)
        evidence = read_artifact(evidence_path, kind="evidence_package")
        anchor_replay_id = str(evidence["parents"].get("execution_anchor_replay") or "")
        if not evidence_cache.same_selection(evidence["payload"].get("selection") or {},
                                             request.profile):
            raise ValueError("공유 Evidence의 Feature Profile 입력이 현재 요청과 다르다")
        if request.refresh_price_path_grounding:
            evidence, evidence_path = _refresh_price_path_grounding(
                evidence, request.profile, output, evidence_path)
    else:
        if str(request.profile.profile_kind) == "execution_aligned":
            # The stored Profile label already contains canonical queue entry and V9 exit.
            # Replaying balanced anchors again would be a redundant diagnostic, not new Evidence.
            # 새 날짜 조합도 여기서 shared cache에 남겨야 다음 Research가 곧바로
            # Hypothesis부터 시작할 수 있다.
            anchor_replay_id = ""
            cached = evidence_cache.materialize(request.profile)
            evidence_path = Path(cached["path"])
            evidence = read_artifact(evidence_path, kind="evidence_package")
        else:
            capacity = research_capacity.reserve(output=request.replay_owner or output)
            if capacity["state"] == "WAITING":
                return _capacity_waiting(request, output, capacity)
            anchor_replay = execution_anchor_stage.run(request.profile, output / "00_execution_anchor_replay")
            anchor_replay_id = anchor_replay["artifact_id"]
            evidence = evidence_stage.run(
                request.profile, output / "01_evidence",
                execution_anchor_replay_path=(output / "00_execution_anchor_replay"
                                              / execution_anchor_stage.ARTIFACT_NAME))
            evidence_path = output / "01_evidence" / evidence_stage.ARTIFACT_NAME
    if str(request.profile.profile_kind) == "execution_aligned":
        execution_contract = ((evidence.get("payload") or {}).get("package") or {}).get(
            "execution_aligned_profile") or {}
        SC.require_for_event_source(
            execution_contract, "Pre-validation execution-aligned Evidence")
    # Direct Evidence 후보는 현재 Evidence Package에서 파생된 것만 비교한다. 다른
    # Profile의 동일 feature 상태가 현재 entry logic을 미리 점유하지 않게 한다.
    if request.candidate_library_source is not None:
        library = read_artifact(Path(request.candidate_library_source), kind="candidate_library")
        library_payload = library["payload"]
        if (library_payload.get("direct_profile") is not None
                and library_payload.get("direct_profile") != clean(request.profile.as_input())):
            raise ValueError("고정 candidate library의 Feature Profile이 현재 요청과 다르다")
        saved_evidence = library_payload.get("direct_evidence_id")
        if saved_evidence is not None and str(saved_evidence) != str(evidence["artifact_id"]):
            raise ValueError("고정 candidate library의 Evidence와 현재 요청이 다르다")
        write_artifact(output / "00_candidate_library" / candidate_library.ARTIFACT_NAME, library)
    else:
        library = candidate_library.build(output / "00_candidate_library",
                                          direct_evidence_id=str(evidence["artifact_id"]),
                                          direct_profile=request.profile.as_input())
    lens_catalog = mechanism_lenses.build(evidence["payload"]["package"],
                                          hypothesis_core.input_audit(evidence["payload"]["package"]))
    write_json(output / "00_mechanism_lenses.json", lens_catalog)
    research_lens = (mechanism_lenses.select(lens_catalog, request.research_lens_id)
                     if request.research_lens_id else None)
    loss_context = None
    if request.discovery_loss_context is not None:
        loss_context = read_artifact(request.discovery_loss_context, kind="discovery_loss_context")
        source_profile = (loss_context["payload"].get("profile") or {})
        if source_profile != clean(request.profile.as_input()):
            raise ValueError("Discovery loss context와 현재 Feature Profile 입력이 다르다")
    validation_feedback = None
    if request.development_validation_feedback is not None:
        validation_feedback = read_artifact(
            request.development_validation_feedback, kind="development_validation_feedback")
        source_profile = (validation_feedback["payload"].get("profile") or {})
        if source_profile != clean(request.profile.as_input()):
            raise ValueError("Development validation feedback과 현재 Feature Profile 입력이 다르다")
    hypothesis_refs = {
        "candidate_library": library["artifact_id"],
        "evidence_package": evidence["artifact_id"],
        **({"execution_anchor_replay": anchor_replay_id} if anchor_replay_id else {}),
        **({"discovery_loss_context": loss_context["artifact_id"]} if loss_context else {}),
        **({"development_validation_feedback": validation_feedback["artifact_id"]}
           if validation_feedback else {}),
    }
    _write_progress(output, "HYPOTHESIS", hypothesis_refs)
    hypotheses = hypothesis_stage.run(evidence_path,
                                     output / "02_hypothesis", runner=runner,
                                     discovery_loss_context=loss_context,
                                     development_validation_feedback=validation_feedback,
                                     candidate_library=library,
                                     research_lens=research_lens)
    stage_refs: dict[str, Any] = {
        "candidate_library": library["artifact_id"],
        "evidence_package": evidence["artifact_id"],
        "hypothesis_set": hypotheses["artifact_id"],
    }
    if anchor_replay_id:
        stage_refs["execution_anchor_replay"] = anchor_replay_id
    if loss_context is not None:
        stage_refs["discovery_loss_context"] = loss_context["artifact_id"]
    if validation_feedback is not None:
        stage_refs["development_validation_feedback"] = validation_feedback["artifact_id"]
    grounding_result = None
    if hypotheses["payload"].get("state") == "READY_FOR_GROUNDING":
        _write_progress(output, "MECHANISM_SPECIFICATION", stage_refs)
        mechanisms = mechanism_specification_stage.run(
            evidence_path, output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification", hypothesis_ids=request.hypothesis_ids or None)
        stage_refs["mechanism_specification_set"] = mechanisms["artifact_id"]
        _write_progress(output, "GROUNDING", stage_refs)
        grounding_result = grounding_stage.run(
            evidence_path,
            output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification" / mechanism_specification_stage.ARTIFACT_NAME,
            output / "03_grounding",
            hypothesis_ids=request.hypothesis_ids or None,
            runner=runner,
        )
        stage_refs["grounding_set"] = grounding_result["grounding"]["artifact_id"]
        if grounding_result["validation_plan"] is not None:
            stage_refs["validation_plan"] = grounding_result["validation_plan"]["artifact_id"]
    state = "READY_FOR_SPECIFICATION" if "validation_plan" in stage_refs else (
        grounding_result["grounding"]["payload"].get("state", "STOPPED")
        if grounding_result is not None else hypotheses["payload"].get("state", "STOPPED"))
    manifest = create_artifact("pre_validation_run", {
        "state": state,
        "request": {"profile": request.profile.as_input(),
                    "hypothesis_ids": list(request.hypothesis_ids),
                    "evidence_source": (str(evidence_source)
                                        if evidence_source is not None else None),
                    "effective_evidence_path": str(evidence_path),
                    "refresh_price_path_grounding": bool(request.refresh_price_path_grounding),
                    "shared_evidence_execution_anchor_replay": (
                        (anchor_replay_id or None)
                        if request.evidence_source is not None else None),
                    "discovery_loss_context": (loss_context["artifact_id"] if loss_context else None),
                    "development_validation_feedback": (
                        validation_feedback["artifact_id"] if validation_feedback else None),
                    "research_lens_id": request.research_lens_id,
                    "candidate_library_source": (str(request.candidate_library_source)
                                                 if request.candidate_library_source is not None else None),
                    "replay_owner": (str(request.replay_owner)
                                     if request.replay_owner is not None else None)},
        "artifacts": stage_refs,
        "research_lens": research_lens,
        "next_stage": "specification" if state == "READY_FOR_SPECIFICATION" else None,
        "validation_executed": False,
    }, parents=stage_refs)
    write_artifact(output / "pre_validation_run.json", manifest)
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "pre_validation.v2", "state": state, "artifacts": stage_refs,
    }))
    _write_progress(output, "TERMINAL", stage_refs, terminal_state=state)
    return {"state": state, "artifacts": stage_refs, "run": manifest,
            "grounding": grounding_result}


def revalidate(request: PreValidationRequest, source: Path, output: Path, *,
               runner: Runner | None = None) -> dict[str, Any]:
    """같은 입력의 기존 Agent artifact를 현재 validator로 재판정한 뒤 이어서 수행한다."""
    source, output = Path(source), Path(output)
    _write_progress(output, "HYPOTHESIS")
    previous = read_artifact(source / "pre_validation_run.json", kind="pre_validation_run")
    source_request = previous["payload"].get("request") or {}
    if source_request.get("profile") != clean(request.profile.as_input()):
        raise ValueError("재검증 source의 Feature Profile이 현재 요청과 다르다")
    # 기존 실행에서 모든 후보를 만들었더라도 지금은 그중 일부만 Grounding할 수 있다.
    # 선택은 Hypothesis payload 안에서 `select_hypotheses`가 실제 존재 여부를 확인한다.

    evidence = read_artifact(source / "01_evidence" / evidence_stage.ARTIFACT_NAME,
                             kind="evidence_package")
    source_hypothesis = source / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME
    output_evidence = output / "01_evidence" / evidence_stage.ARTIFACT_NAME
    write_artifact(output_evidence, evidence)
    hypotheses = hypothesis_stage.revalidate(output_evidence, source_hypothesis,
                                              output / "02_hypothesis")
    stage_refs: dict[str, Any] = {
        "evidence_package": evidence["artifact_id"],
        "hypothesis_set": hypotheses["artifact_id"],
    }
    grounding_result = None
    if hypotheses["payload"].get("state") == "READY_FOR_GROUNDING":
        _write_progress(output, "MECHANISM_SPECIFICATION", stage_refs)
        mechanisms = mechanism_specification_stage.run(
            output_evidence, output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification", hypothesis_ids=request.hypothesis_ids or None)
        stage_refs["mechanism_specification_set"] = mechanisms["artifact_id"]
        _write_progress(output, "GROUNDING", stage_refs)
        grounding_result = grounding_stage.run(
            output_evidence, output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification" / mechanism_specification_stage.ARTIFACT_NAME,
            output / "03_grounding", hypothesis_ids=request.hypothesis_ids or None,
            runner=runner)
        stage_refs["grounding_set"] = grounding_result["grounding"]["artifact_id"]
        if grounding_result["validation_plan"] is not None:
            stage_refs["validation_plan"] = grounding_result["validation_plan"]["artifact_id"]
    state = "READY_FOR_SPECIFICATION" if "validation_plan" in stage_refs else (
        grounding_result["grounding"]["payload"].get("state", "STOPPED")
        if grounding_result is not None else hypotheses["payload"].get("state", "STOPPED"))
    manifest = create_artifact("pre_validation_run", {
        "state": state,
        "request": {"profile": request.profile.as_input(),
                    "hypothesis_ids": list(request.hypothesis_ids)},
        "artifacts": stage_refs,
        "next_stage": "specification" if state == "READY_FOR_SPECIFICATION" else None,
        "validation_executed": False,
        "revalidated_source": str(source),
    }, parents=stage_refs)
    write_artifact(output / "pre_validation_run.json", manifest)
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "pre_validation_revalidate.v1", "state": state, "artifacts": stage_refs,
    }))
    _write_progress(output, "TERMINAL", stage_refs, terminal_state=state)
    return {"state": state, "artifacts": stage_refs, "run": manifest,
            "grounding": grounding_result}


def grounding_revalidate(source: Path, output: Path) -> dict[str, Any]:
    """기존 Hypothesis는 보존하고 저장된 Grounding만 현재 계약으로 다시 읽는다."""
    source, output = Path(source), Path(output)
    _write_progress(output, "GROUNDING")
    previous = read_artifact(source / "pre_validation_run.json", kind="pre_validation_run")
    refs = dict(previous["payload"].get("artifacts") or {})
    evidence = read_artifact(source / "01_evidence" / evidence_stage.ARTIFACT_NAME,
                             kind="evidence_package")
    hypotheses = read_artifact(source / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
                               kind="hypothesis_set")
    source_grounding = source / "03_grounding" / grounding_stage.GROUNDING_ARTIFACT_NAME
    if not source_grounding.is_file():
        raise ValueError("재검증할 Grounding artifact가 없다")
    output_evidence = output / "01_evidence" / evidence_stage.ARTIFACT_NAME
    output_hypotheses = output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME
    write_artifact(output_evidence, evidence)
    write_artifact(output_hypotheses, hypotheses)
    mechanisms = mechanism_specification_stage.run(
        output_evidence, output_hypotheses,
        output / "02_mechanism_specification",
        hypothesis_ids=tuple(str(value) for value in
                             (read_artifact(source_grounding, kind="grounding_set")["payload"].get(
                                 "selected_hypothesis_ids") or [])))
    grounding_result = grounding_stage.revalidate(
        output_evidence, output_hypotheses,
        output / "02_mechanism_specification" / mechanism_specification_stage.ARTIFACT_NAME,
        source_grounding, output / "03_grounding")
    refs.update({"evidence_package": evidence["artifact_id"],
                 "hypothesis_set": hypotheses["artifact_id"],
                 "mechanism_specification_set": mechanisms["artifact_id"],
                 "grounding_set": grounding_result["grounding"]["artifact_id"]})
    if grounding_result["validation_plan"] is not None:
        refs["validation_plan"] = grounding_result["validation_plan"]["artifact_id"]
    else:
        refs.pop("validation_plan", None)
    state = ("READY_FOR_SPECIFICATION" if "validation_plan" in refs else
             grounding_result["grounding"]["payload"].get("state", "STOPPED"))
    manifest = create_artifact("pre_validation_run", {
        "state": state,
        "request": previous["payload"].get("request") or {},
        "artifacts": refs,
        "research_lens": previous["payload"].get("research_lens"),
        "next_stage": "specification" if state == "READY_FOR_SPECIFICATION" else None,
        "validation_executed": False,
        "grounding_revalidated_source": str(source),
    }, parents=refs)
    write_artifact(output / "pre_validation_run.json", manifest)
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "pre_validation_grounding_revalidate.v1", "state": state,
        "artifacts": refs,
    }))
    _write_progress(output, "TERMINAL", refs, terminal_state=state)
    return {"state": state, "artifacts": refs, "run": manifest,
            "grounding": grounding_result}


def novelty_revalidate(request: PreValidationRequest, source: Path, output: Path, *,
                       runner: Runner | None = None) -> dict[str, Any]:
    """기존 가설이 Sandbox 전체에서 정말 새 메커니즘인지 확인한 뒤에만 Grounding한다."""
    source, output = Path(source), Path(output)
    _write_progress(output, "HYPOTHESIS")
    previous = read_artifact(source / "pre_validation_run.json", kind="pre_validation_run")
    source_request = previous["payload"].get("request") or {}
    if source_request.get("profile") != clean(request.profile.as_input()):
        raise ValueError("Novelty 재비교 source의 Feature Profile이 현재 요청과 다르다")
    evidence = read_artifact(source / "01_evidence" / evidence_stage.ARTIFACT_NAME,
                             kind="evidence_package")
    library = candidate_library.build(
        output / "00_candidate_library",
        exclude_artifact_ids={str(previous["payload"]["artifacts"]["hypothesis_set"])},
        exclude_descendants_of={str(previous["payload"]["artifacts"]["hypothesis_set"])},
        direct_evidence_id=str(evidence["artifact_id"]),
        direct_profile=request.profile.as_input())
    output_evidence = output / "01_evidence" / evidence_stage.ARTIFACT_NAME
    write_artifact(output_evidence, evidence)
    hypotheses = hypothesis_stage.novelty_revalidate(
        output_evidence, source / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
        library, output / "02_hypothesis", runner=runner)
    stage_refs: dict[str, Any] = {"candidate_library": library["artifact_id"],
                                  "evidence_package": evidence["artifact_id"],
                                  "hypothesis_set": hypotheses["artifact_id"]}
    grounding_result = None
    if hypotheses["payload"].get("state") == "READY_FOR_GROUNDING":
        _write_progress(output, "MECHANISM_SPECIFICATION", stage_refs)
        mechanisms = mechanism_specification_stage.run(
            output_evidence, output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification", hypothesis_ids=request.hypothesis_ids or None)
        stage_refs["mechanism_specification_set"] = mechanisms["artifact_id"]
        _write_progress(output, "GROUNDING", stage_refs)
        grounding_result = grounding_stage.run(
            output_evidence, output / "02_hypothesis" / hypothesis_stage.ARTIFACT_NAME,
            output / "02_mechanism_specification" / mechanism_specification_stage.ARTIFACT_NAME,
            output / "03_grounding", hypothesis_ids=request.hypothesis_ids or None,
            runner=runner)
        stage_refs["grounding_set"] = grounding_result["grounding"]["artifact_id"]
        if grounding_result["validation_plan"] is not None:
            stage_refs["validation_plan"] = grounding_result["validation_plan"]["artifact_id"]
    state = "READY_FOR_SPECIFICATION" if "validation_plan" in stage_refs else (
        grounding_result["grounding"]["payload"].get("state", "STOPPED")
        if grounding_result is not None else hypotheses["payload"].get("state", "STOPPED"))
    manifest = create_artifact("pre_validation_run", {
        "state": state,
        "request": {"profile": request.profile.as_input(),
                    "hypothesis_ids": list(request.hypothesis_ids)},
        "artifacts": stage_refs,
        "next_stage": "specification" if state == "READY_FOR_SPECIFICATION" else None,
        "validation_executed": False,
        "novelty_revalidated_source": str(source),
    }, parents=stage_refs)
    write_artifact(output / "pre_validation_run.json", manifest)
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "pre_validation_novelty_revalidate.v1", "state": state, "artifacts": stage_refs,
    }))
    _write_progress(output, "TERMINAL", stage_refs, terminal_state=state)
    return {"state": state, "artifacts": stage_refs, "run": manifest,
            "grounding": grounding_result}
