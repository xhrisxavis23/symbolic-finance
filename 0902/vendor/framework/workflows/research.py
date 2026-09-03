"""FeatureProfile부터 Validation Backtest와 Final Backtest까지 잇는 연구 Workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..agents.runtime import Runner
from ..config import (TICK_ROOT, clean, execution_workers, now_utc, read_json,
                      run_manifest, write_json)
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import (candidate_library, direct_candidate_reservation, discovery_loss,
                       refinement, research_capacity)
from ..modules.profile import ProfileSelection
from ..modules.refinement import RefinementConfig
from ..modules.schedule import ResearchSchedule
from ..stages import (backtest_stage, grounding_stage, implementation_stage,
                      direct_evidence_stage, parameter_search_stage, refinement_stage, specification_stage,
                      grounding_fidelity_stage, registry_stage, validation_stage)
from .pre_validation import PreValidationRequest, run as run_pre_validation


@dataclass(frozen=True)
class ResearchRequest:
    profile: ProfileSelection
    schedule: ResearchSchedule
    hypothesis_ids: tuple[str, ...] = ()
    evidence_source: Path | None = None
    pre_validation_source: Path | None = None
    discovery_loss_context: Path | None = None
    research_lens_id: str | None = None
    direct_evidence_fallback: bool = True
    entry_refinement_branch_quantiles: tuple[float, ...] = ()
    auto_entry_refinement_branches: bool = True
    # Validation 실패를 읽어 만든 successor는 Validation을 다시 통과 관문으로 쓰지
    # 않는다. 새 명세·구현·q lock 뒤 Final 날짜에서 처음 평가한다.
    final_backtest_for_all_validation: bool = False
    # 이 run의 가설은 Validation 실패 장부를 개발 입력으로 읽고 만들어졌다.
    # 따라서 같은 Validation 날짜는 다시 promotion 관문으로 쓰지 않는다.
    validation_refinement_successor: bool = False
    # 기본 research는 Discovery 손실을 다음 Agent research로 계속 잇는다. loop가 만든
    # 하위 research는 이 값을 False로 받아 controller 하나만 세대를 관리한다.
    auto_discovery_refinement: bool = True


PROGRESS_FILE = "research_progress.json"


def _write_progress(output: Path, stage: str, refs: dict[str, str] | None = None, *,
                    terminal_state: str | None = None) -> None:
    """긴 Research 실행의 현재 Stage를 원장과 별도로 남긴다."""
    path = Path(output) / PROGRESS_FILE
    history: list[dict[str, str]] = []
    if path.is_file():
        try:
            prior = read_json(path)
            history = [dict(item) for item in prior.get("stage_history") or []
                       if isinstance(item, Mapping)]
        except (OSError, ValueError, TypeError):
            # 진행 표시는 관측용이다. 깨진 이전 표시가 실제 research를 막지 않는다.
            history = []
    timestamp = now_utc()
    if history and "finished_at" not in history[-1]:
        history[-1]["finished_at"] = timestamp
    history.append({"stage": stage, "started_at": timestamp})
    write_json(path, {
        "schema": "research_progress.v1",
        "updated_at": timestamp,
        "state": "COMPLETE" if terminal_state is not None else "IN_PROGRESS",
        "stage": stage,
        "terminal_state": terminal_state,
        "owner_pid": os.getpid(),
        "artifacts": dict(refs or {}),
        "execution_workers": execution_workers(),
        "stage_history": history,
    })


def _prior_direct_entry_logic_fingerprints(pre_root: Path) -> tuple[str, ...]:
    """같은 Feature Profile에서 이미 열린 직접 entry 계약만 fallback에서 뺀다."""
    path = Path(pre_root) / "00_candidate_library" / candidate_library.ARTIFACT_NAME
    if not path.is_file():
        return ()
    library = read_artifact(path, kind="candidate_library")
    return tuple(sorted({str(record.get("entry_logic_fingerprint") or "")
                         for record in library["payload"].get("records") or []
                         if record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"
                         and str(record.get("entry_logic_fingerprint") or "")}))


def _selected_plan_hypothesis_ids(pre_root: Path, refs: dict[str, str],
                                  requested: tuple[str, ...]) -> tuple[str, ...]:
    """재사용한 Pre-validation의 Grounding 승인 후보 중 요청한 것만 통과시킨다."""
    if not requested:
        return ()
    plan_path = Path(pre_root) / "03_grounding" / grounding_stage.PLAN_ARTIFACT_NAME
    plan = read_artifact(plan_path, kind="validation_plan")
    expected = str(refs.get("validation_plan") or "")
    if expected and str(plan["artifact_id"]) != expected:
        raise ValueError("재사용할 Pre-validation의 ValidationPlan artifact가 다르다")
    available = {str(value) for value in plan["payload"].get("hypothesis_ids") or []}
    unknown = [str(value) for value in requested if str(value) not in available]
    if unknown:
        raise ValueError("재사용할 Pre-validation에서 Grounding된 hypothesis_id가 아니다: "
                         + ", ".join(unknown))
    return tuple(str(value) for value in requested)


def _finish_pre_execution(
        output: Path, request: ResearchRequest, pre: Mapping[str, Any], pre_root: Path,
        effective_evidence_source: Path | None, refs: dict[str, str],
        *, state: str, reason: str, reservation: Mapping[str, Any] | None = None,
        capacity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """실행 자원 또는 Direct entry가 점유 중이면 구현 전에서 남긴다."""
    run_artifact = create_artifact("research_run", {
        "state": state,
        "pre_validation_state": pre["state"],
        "request": {
            "profile": request.profile.as_input(),
            "schedule": request.schedule.as_input(),
            "hypothesis_ids": list(request.hypothesis_ids),
            "evidence_source": (str(effective_evidence_source)
                                if effective_evidence_source is not None else None),
            "pre_validation_source": str(pre_root),
            "discovery_loss_context": (str(request.discovery_loss_context)
                                       if request.discovery_loss_context is not None else None),
            "research_lens_id": request.research_lens_id,
            "direct_evidence_fallback": request.direct_evidence_fallback,
            "entry_refinement_branch_quantiles": list(
                request.entry_refinement_branch_quantiles),
            "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
            "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs,
        },
        "artifacts": refs,
        "validation_backtest_executed": False,
        "final_backtest_executed": False,
        **({"direct_candidate_reservation": dict(reservation)} if reservation is not None else {}),
        **({"research_capacity_reservation": dict(capacity)} if capacity is not None else {}),
        "reason": reason,
    }, parents=refs)
    write_artifact(output / "research_run.json", run_artifact)
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "research.v2", "state": state, "artifacts": refs,
        "terminal_oos_opened": False,
    }))
    _write_progress(output, "TERMINAL", refs, terminal_state=state)
    return {"state": state, "artifacts": refs, "run": run_artifact}


def _run_discovery_refinement(request: ResearchRequest, output: Path,
                              refinements: Mapping[str, Any], validation: Mapping[str, Any], *,
                              root: Path, source_research_run_id: str | None = None) -> dict[str, Any]:
    """Validation 실패를 읽어 새 후보를 Final 전까지 실행한다."""
    if not request.auto_discovery_refinement:
        return {"state": "DISABLED", "units": []}
    # discovery_loop → agent_campaign → research 의 순환 import를 피한다. 이 controller는
    # root research가 research_run artifact를 쓴 뒤에만 열 수 있다.
    from . import discovery_loop

    units: list[dict[str, Any]] = []
    validation_units = (validation.get("payload") or {}).get("units") or {}
    for hypothesis_id, unit in sorted((refinements.get("payload") or {}).get("units", {}).items()):
        if not isinstance(unit, Mapping):
            continue
        validation_unit = validation_units.get(str(hypothesis_id))
        if isinstance(validation_unit, Mapping) \
                and validation_unit.get("state") == "VALIDATION_BACKTEST_SUPPORTED":
            units.append({"hypothesis_id": str(hypothesis_id), "state": "NOT_OPENED",
                          "reason": "Validation을 통과해 successor loss refinement 대상이 아니다"})
            continue
        has_parent = bool(unit.get("rounds"))
        if not has_parent and isinstance(unit.get("branch_records"), Mapping):
            selected = (Path(output) / "05_parameter_search" / "units" /
                        str(hypothesis_id) / "selected_parameter.json")
            has_parent = selected.is_file() and read_json(selected).get("status") == "SEARCH_SELECTED"
        if not has_parent:
            units.append({"hypothesis_id": str(hypothesis_id), "state": "NOT_OPENED",
                          "reason": "Discovery parent ledger 또는 선택된 q가 없다"})
            continue
        loop_root = Path(output) / "07_validation_discovery_refinement" / str(hypothesis_id)
        try:
            plan = discovery_loop.plan(discovery_loop.DiscoveryLoopPlanRequest(
                source_research=Path(output), source_hypothesis_id=str(hypothesis_id),
                direct_event_fallback=True, entry_lifecycle_fallback=True), loop_root)
            result = discovery_loop.run(loop_root / discovery_loop.PLAN_ARTIFACT_NAME, root=Path(root))
            units.append({"hypothesis_id": str(hypothesis_id), "state": str(result["state"]),
                          "plan_artifact": plan["artifact_id"], "result": result})
        except discovery_loss.DiscoveryParentLedgerUnavailable as error:
            units.append({"hypothesis_id": str(hypothesis_id), "state": "NOT_OPENED",
                          "reason": str(error)})
        except (OSError, ValueError, FileNotFoundError) as error:
            units.append({"hypothesis_id": str(hypothesis_id), "state": "ERROR",
                          "reason": f"{type(error).__name__}: {error}"})
    state = "COMPLETE" if all(unit["state"] != "ERROR" for unit in units) else "PARTIAL_ERROR"
    payload = {
        "schema": "validation_discovery_refinement.v1",
        "state": state,
        "input_boundary": "Validation Backtest의 실패 원장은 새 후보의 개발 입력이며 Final은 읽지 않는다.",
        "units": units,
    }
    parents = {
        "entry_refinement_set": str(refinements.get("artifact_id") or ""),
        "validation_backtest_set": str(validation.get("artifact_id") or ""),
    }
    if source_research_run_id:
        parents["validation_refinement_source_research_run"] = str(source_research_run_id)
    artifact = create_artifact("validation_discovery_refinement", payload,
                               parents={key: value for key, value in parents.items() if value})
    stage_root = Path(output) / "07_validation_discovery_refinement"
    write_artifact(stage_root / "validation_discovery_refinement_artifact.json", artifact)
    return {**payload, "artifact_id": artifact["artifact_id"]}


def run(request: ResearchRequest, output: Path, *, runner: Runner | None = None,
        root: Path = TICK_ROOT) -> dict[str, Any]:
    """각 Stage가 자기 artifact만 읽고, 다음 Stage는 그 artifact만 받는다."""
    output = Path(output)
    _write_progress(output, "PRE_VALIDATION")
    refinement_config = RefinementConfig(
        branch_quantiles=tuple(sorted({
            float(value) for value in request.entry_refinement_branch_quantiles
        })),
        auto_branch_search_quantiles=bool(request.auto_entry_refinement_branches))
    if request.pre_validation_source is None:
        pre_root = output / "01_pre_validation"
        pre = run_pre_validation(PreValidationRequest(
            profile=request.profile, hypothesis_ids=request.hypothesis_ids,
            evidence_source=request.evidence_source,
            discovery_loss_context=request.discovery_loss_context,
            research_lens_id=request.research_lens_id,
            replay_owner=output), pre_root,
            runner=runner)
        saved_effective_evidence_path = (
            (pre["run"].get("payload") or {}).get("request", {}).get("effective_evidence_path"))
        effective_evidence_source = (Path(saved_effective_evidence_path)
                                     if saved_effective_evidence_path else None)
    else:
        pre_root = Path(request.pre_validation_source)
        resumed = read_artifact(pre_root / "pre_validation_run.json", kind="pre_validation_run")
        saved_request = resumed["payload"].get("request") or {}
        saved_evidence_source = saved_request.get("evidence_source")
        if request.evidence_source is not None and str(request.evidence_source) != saved_evidence_source:
            raise ValueError("재사용할 Pre-validation의 공유 Evidence 입력이 현재 요청과 다르다")
        saved_effective_evidence_path = saved_request.get("effective_evidence_path")
        effective_evidence_source = (Path(saved_effective_evidence_path)
                                     if saved_effective_evidence_path else
                                     (Path(saved_evidence_source) if saved_evidence_source else None))
        expected_request = clean({"profile": request.profile.as_input()})
        saved_comparable = {key: saved_request.get(key) for key in expected_request}
        if saved_comparable != expected_request:
            raise ValueError("재사용할 Pre-validation artifact의 입력이 현재 Research 요청과 다르다")
        pre = {
            "state": resumed["payload"].get("state"),
            "artifacts": dict(resumed["payload"].get("artifacts") or {}),
            "run": resumed,
            "grounding": None,
        }
    evidence_path = (effective_evidence_source
                     if effective_evidence_source is not None
                     else pre_root / "01_evidence" / "evidence_artifact.json")
    refs: dict[str, str] = {
        "pre_validation_run": pre["run"]["artifact_id"],
        **pre["artifacts"],
    }
    if pre["state"] == "CAPACITY_WAITING":
        capacity = (pre.get("capacity")
                    or pre["run"]["payload"].get("research_capacity_reservation"))
        return _finish_pre_execution(
            output, request, pre, pre_root, effective_evidence_source, refs,
            state="CAPACITY_WAITING",
            reason="execution-anchor replay 전 전역 실행 slot이 모두 사용 중이다",
            capacity=capacity)
    selected_hypothesis_ids = _selected_plan_hypothesis_ids(
        pre_root, refs, request.hypothesis_ids) if "validation_plan" in refs else ()
    specification_path: Path | None = None
    specifications = None
    direct_candidates = None
    reservation = None
    capacity = None
    direct_evidence_state: str | None = None
    # 손실 장부 successor는 Agent가 손실 원인을 읽고 새 entry 가설을 만드는 경로다.
    # 여기서 일반 Direct Evidence 후보를 다시 열면, 중복 가설 뒤에 과거 후보의
    # 대량 재실행으로 바뀌어 loss context가 다음 연구에 반영되지 않는다.
    # research lens도 같은 경계다. lens가 있으면 Agent는 그 관측 관계만
    # 설명하거나 NO_HYPOTHESIS를 내야 하며, 무관한 일반 후보로 우회하면 안 된다.
    direct_evidence_fallback_allowed = (
        request.direct_evidence_fallback
        and request.discovery_loss_context is None
        and request.research_lens_id is None
    )
    if "validation_plan" not in refs and direct_evidence_fallback_allowed \
            and evidence_path.is_file():
        _write_progress(output, "SPECIFICATION", refs)
        direct = direct_evidence_stage.run(
            evidence_path,
            output / "02_direct_evidence_specification",
            exclude_entry_logic_fingerprints=_prior_direct_entry_logic_fingerprints(pre_root))
        refs["direct_evidence_candidate_set"] = direct["candidates"]["artifact_id"]
        direct_candidates = direct["candidates"]
        specifications = direct["specifications"]
        direct_evidence_state = str(specifications["payload"]["state"])
        refs["executable_specification_set"] = specifications["artifact_id"]
        specification_path = output / "02_direct_evidence_specification" / direct_evidence_stage.ARTIFACT_NAME
    if specifications is None and "validation_plan" not in refs:
        state = direct_evidence_state or pre["state"]
        run_artifact = create_artifact("research_run", {
            "state": state, "pre_validation_state": pre["state"], "request": {"profile": request.profile.as_input(),
                        "schedule": request.schedule.as_input(), "hypothesis_ids": list(request.hypothesis_ids),
                        "evidence_source": (str(effective_evidence_source)
                                            if effective_evidence_source is not None else None),
            "pre_validation_source": str(pre_root),
            "discovery_loss_context": (str(request.discovery_loss_context)
                                       if request.discovery_loss_context is not None else None),
            "research_lens_id": request.research_lens_id,
            "direct_evidence_fallback": request.direct_evidence_fallback,
            "entry_refinement_branch_quantiles": list(
                request.entry_refinement_branch_quantiles),
            "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
            "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs},
            "artifacts": refs, "validation_backtest_executed": False,
            "final_backtest_executed": False,
        }, parents=refs)
        write_artifact(output / "research_run.json", run_artifact)
        _write_progress(output, "TERMINAL", refs, terminal_state=state)
        return {"state": state, "artifacts": refs, "run": run_artifact}

    if specifications is None:
        mechanism_path = (pre_root / "02_mechanism_specification" /
                          "mechanism_specification_artifact.json")
        if not mechanism_path.is_file():
            return _finish_pre_execution(
                output, request, pre, pre_root, effective_evidence_source, refs,
                state="MECHANISM_SPECIFICATION_REQUIRED",
                reason="이 Pre-validation은 Grounding 전 mechanism specification artifact가 없다")
        _write_progress(output, "GROUNDING_FIDELITY", refs)
        fidelity = grounding_fidelity_stage.run(
            evidence_path,
            pre_root / "02_hypothesis" / "hypothesis_artifact.json",
            mechanism_path,
            pre_root / "03_grounding" / grounding_stage.GROUNDING_ARTIFACT_NAME,
            pre_root / "03_grounding" / grounding_stage.PLAN_ARTIFACT_NAME,
            output / "02_grounding_fidelity",
            hypothesis_ids=selected_hypothesis_ids or None)
        refs["mechanism_specification_set"] = read_artifact(
            mechanism_path, kind="mechanism_specification_set")["artifact_id"]
        refs["grounding_mechanism_fidelity_set"] = fidelity["artifact_id"]
        _write_progress(output, "SPECIFICATION", refs)
        specifications = specification_stage.run(
            evidence_path,
            pre_root / "02_hypothesis" / "hypothesis_artifact.json",
            pre_root / "03_grounding" / grounding_stage.GROUNDING_ARTIFACT_NAME,
            pre_root / "03_grounding" / grounding_stage.PLAN_ARTIFACT_NAME,
            output / "02_specification",
            hypothesis_ids=selected_hypothesis_ids or None,
            mechanism_path=mechanism_path,
            fidelity_path=(output / "02_grounding_fidelity" /
                           grounding_fidelity_stage.ARTIFACT_NAME))
        refs["executable_specification_set"] = specifications["artifact_id"]
        specification_path = output / "02_specification" / specification_stage.ARTIFACT_NAME
        if not any("spec" in unit for unit in specifications["payload"].get("units", {}).values()):
            return _finish_pre_execution(
                output, request, pre, pre_root, effective_evidence_source, refs,
                state="MECHANISM_DIAGNOSTIC_REQUIRED",
                reason="Grounding의 체결 뒤 예측을 먼저 matched control 또는 메커니즘 진단으로 확인해야 한다")
    assert specification_path is not None
    capacity = research_capacity.reserve(output=output)
    if capacity["state"] == "WAITING":
        return _finish_pre_execution(
            output, request, pre, pre_root, effective_evidence_source, refs,
            state="CAPACITY_WAITING",
            reason="다른 active 연구 replay가 전역 실행 slot을 사용 중이다",
            capacity=capacity)
    if direct_candidates is not None:
        reservation = direct_candidate_reservation.reserve(
            direct_candidates["payload"].get("candidates") or [],
            profile=request.profile.as_input(), output=output)
        if reservation["state"] == "CONFLICT":
            return _finish_pre_execution(
                output, request, pre, pre_root, effective_evidence_source, refs,
                state="ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE",
                reason="같은 Direct entry의 Entry Refinement가 다른 run에서 진행 중이다",
                reservation=reservation, capacity=capacity)
    branching_ids = refinement.branching_hypothesis_ids(
        specifications["payload"].get("units") or {}, refinement_config)
    _write_progress(output, "IMPLEMENTATION", refs)
    implementations = implementation_stage.run(
        specification_path,
        output / "03_implementation",
        allow_equivalent_for_refinement=bool(refinement_config.branch_quantiles),
        allow_equivalent_for_refinement_ids=branching_ids)
    implementation_path = output / "03_implementation" / implementation_stage.ARTIFACT_NAME
    refs["implementation_set"] = implementations["artifact_id"]
    implementation_units = list(implementations["payload"].get("units", {}).values())
    # Agent가 만든 설명이 기존 실행 계약으로 수렴해도, 같은 Evidence에는 아직 직접
    # 관측 후보가 남아 있을 수 있다. 이때 Agent 문장을 다시 고치지 말고 한 번만
    # Direct Evidence 경로를 열어 실제로 다른 entry 계약이 있는지 확인한다.
    if (implementation_units
            and all(item.get("state") == "EXECUTION_EQUIVALENT" for item in implementation_units)
            and direct_evidence_fallback_allowed
            and "direct_evidence_candidate_set" not in refs):
        _write_progress(output, "SPECIFICATION", refs)
        direct = direct_evidence_stage.run(
            evidence_path,
            output / "02_direct_evidence_specification",
            exclude_entry_logic_fingerprints=_prior_direct_entry_logic_fingerprints(pre_root))
        refs["direct_evidence_candidate_set"] = direct["candidates"]["artifact_id"]
        direct_candidates = direct["candidates"]
        direct_evidence_state = str(direct["specifications"]["payload"]["state"])
        if direct_evidence_state == "SPECIFICATION_COMPLETE":
            specifications = direct["specifications"]
            refs["executable_specification_set"] = specifications["artifact_id"]
            specification_path = (output / "02_direct_evidence_specification"
                                  / direct_evidence_stage.ARTIFACT_NAME)
            reservation = direct_candidate_reservation.reserve(
                direct_candidates["payload"].get("candidates") or [],
                profile=request.profile.as_input(), output=output)
            if reservation["state"] == "CONFLICT":
                return _finish_pre_execution(
                    output, request, pre, pre_root, effective_evidence_source, refs,
                    state="ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE",
                    reason="같은 Direct entry의 Entry Refinement가 다른 run에서 진행 중이다",
                    reservation=reservation, capacity=capacity)
            branching_ids = refinement.branching_hypothesis_ids(
                specifications["payload"].get("units") or {}, refinement_config)
            implementations = implementation_stage.run(
                specification_path,
                output / "03_implementation_direct_evidence",
                allow_equivalent_for_refinement=bool(refinement_config.branch_quantiles),
                allow_equivalent_for_refinement_ids=branching_ids)
            implementation_path = (output / "03_implementation_direct_evidence"
                                   / implementation_stage.ARTIFACT_NAME)
            refs["implementation_set"] = implementations["artifact_id"]
            implementation_units = list(implementations["payload"].get("units", {}).values())
    if implementation_units and all(
            item.get("state") == "EXECUTION_EQUIVALENT" for item in implementation_units) \
            and not refinement_config.branch_quantiles and not branching_ids:
        run_artifact = create_artifact("research_run", {
            "state": "EXECUTION_EQUIVALENT", "pre_validation_state": pre["state"],
            "request": {"profile": request.profile.as_input(), "schedule": request.schedule.as_input(),
                        "hypothesis_ids": list(request.hypothesis_ids),
                        "evidence_source": (str(effective_evidence_source)
                                            if effective_evidence_source is not None else None),
                        "pre_validation_source": str(pre_root),
                        "discovery_loss_context": (str(request.discovery_loss_context)
                                                   if request.discovery_loss_context is not None else None),
                        "research_lens_id": request.research_lens_id,
                        "direct_evidence_fallback": request.direct_evidence_fallback,
                        "entry_refinement_branch_quantiles": list(
                            request.entry_refinement_branch_quantiles),
                        "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
                        "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs},
            "artifacts": refs, "validation_backtest_executed": False,
            "final_backtest_executed": False,
            **({"direct_candidate_reservation": reservation}
               if reservation is not None else {}),
            **({"research_capacity_reservation": capacity}
               if capacity is not None else {}),
            "reason": "모든 구현 계약이 기존 Sandbox entry·execution 계약과 같다",
        }, parents=refs)
        write_artifact(output / "research_run.json", run_artifact)
        write_json(output / "run_manifest.json", run_manifest(output, {
            "workflow": "research.v2", "state": "EXECUTION_EQUIVALENT", "artifacts": refs,
            "terminal_oos_opened": False,
        }))
        _write_progress(output, "TERMINAL", refs, terminal_state="EXECUTION_EQUIVALENT")
        return {"state": "EXECUTION_EQUIVALENT", "artifacts": refs, "run": run_artifact}
    _write_progress(output, "ENTRY_REFINEMENT", refs)
    refinements = refinement_stage.run(
        specification_path,
        implementation_path,
        evidence_path,
        output / "04_entry_refinement", root=root, config=refinement_config, runner=runner)
    refs["entry_refinement_set"] = refinements["artifact_id"]
    _write_progress(output, "PARAMETER_SEARCH", refs)
    searches = parameter_search_stage.run(
        specification_path,
        implementation_path,
        output / "04_entry_refinement" / refinement_stage.ARTIFACT_NAME,
        evidence_path,
        output / "05_parameter_search", schedule=request.schedule, root=root)
    refs["parameter_search_set"] = searches["artifact_id"]
    if request.validation_refinement_successor:
        _write_progress(output, "FINAL_BACKTEST", refs)
        backtests = backtest_stage.run_from_validation_refinement(
            output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
            evidence_path,
            output / "08_final_backtest", schedule=request.schedule, root=root)
        refs["backtest_set"] = backtests["artifact_id"]
        backtest_state = backtests["payload"]["state"]
        final_backtest_executed = backtest_state == "BACKTEST_COMPLETE"
        run_artifact = create_artifact("research_run", {
            "state": backtest_state, "request": {
                "profile": request.profile.as_input(), "schedule": request.schedule.as_input(),
                "hypothesis_ids": list(request.hypothesis_ids),
                "evidence_source": (str(effective_evidence_source)
                                    if effective_evidence_source is not None else None),
                "pre_validation_source": str(pre_root),
                "discovery_loss_context": (str(request.discovery_loss_context)
                                           if request.discovery_loss_context is not None else None),
                "research_lens_id": request.research_lens_id,
                "direct_evidence_fallback": request.direct_evidence_fallback,
                "entry_refinement_branch_quantiles": list(
                    request.entry_refinement_branch_quantiles),
                "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
                "auto_discovery_refinement": request.auto_discovery_refinement,
                "final_backtest_for_all_validation": request.final_backtest_for_all_validation,
                "validation_refinement_successor": True,
                "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs},
            "artifacts": refs,
            "validation_backtest_executed": False,
            "final_backtest_executed": final_backtest_executed,
            "terminal_oos_opened": False,
            "input_boundary": (
                "Validation Backtest는 이 새 후보의 개발 입력이었다. 같은 날짜의 promotion을 "
                "다시 열지 않고 parameter lock 뒤 Final Backtest를 실행했다."),
            **({"direct_candidate_reservation": reservation}
               if reservation is not None else {}),
            **({"research_capacity_reservation": capacity}
               if capacity is not None else {}),
        }, parents=refs)
        write_artifact(output / "research_run.json", run_artifact)
        if final_backtest_executed:
            registry_refresh = registry_stage.run(
                run_artifact, backtest_set_id=backtests["artifact_id"],
                output=output / "08_final_registry_refresh")
            refs["final_registry_refresh"] = registry_refresh["artifact_id"]
        write_json(output / "run_manifest.json", run_manifest(output, {
            "workflow": "research.v2", "state": backtest_state, "artifacts": refs,
            "terminal_oos_opened": False,
        }))
        _write_progress(output, "TERMINAL", refs, terminal_state=backtest_state)
        return {"state": backtest_state, "artifacts": refs, "run": run_artifact,
                "validation": None, "specifications": specifications,
                "implementations": implementations, "searches": searches, "backtests": backtests,
                "discovery_refinement": {"state": "VALIDATION_INPUT_ALREADY_CONSUMED", "units": []}}
    _write_progress(output, "VALIDATION_BACKTEST", refs)
    validation = validation_stage.run(
        output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
        evidence_path,
        output / "06_validation_backtest", schedule=request.schedule, root=root)
    refs["validation_backtest_set"] = validation["artifact_id"]
    # Validation ledger는 여기서부터 새 후보를 만드는 개발 입력이다. parent Final은
    # 아직 열지 않으므로 successor가 Final 날짜를 보지 못한다.
    validation_source = create_artifact("research_run", {
        "state": "VALIDATION_DISCOVERY_REFINEMENT_IN_PROGRESS",
        "request": {"profile": request.profile.as_input(),
        "schedule": request.schedule.as_input(), "hypothesis_ids": list(request.hypothesis_ids),
        "evidence_source": (str(effective_evidence_source)
                            if effective_evidence_source is not None else None),
        "pre_validation_source": str(pre_root),
        "discovery_loss_context": (str(request.discovery_loss_context)
                                   if request.discovery_loss_context is not None else None),
        "research_lens_id": request.research_lens_id,
        "direct_evidence_fallback": request.direct_evidence_fallback,
        "entry_refinement_branch_quantiles": list(
            request.entry_refinement_branch_quantiles),
        "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
        "auto_discovery_refinement": request.auto_discovery_refinement,
        "final_backtest_for_all_validation": request.final_backtest_for_all_validation,
        "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs},
        "artifacts": refs,
        "validation_backtest_executed": any(
            (unit.get("backtest") or {}).get("state") == "BACKTEST_COMPLETE"
            for unit in validation["payload"].get("units", {}).values()),
        "final_backtest_executed": False,
        "terminal_oos_opened": False,
        **({"direct_candidate_reservation": reservation}
           if reservation is not None else {}),
        **({"research_capacity_reservation": capacity}
           if capacity is not None else {}),
    }, parents=refs)
    # 후속 후보가 읽은 parent 상태를 Final 결과와 분리해 보존한다.
    validation_refinement_root = output / "07_validation_discovery_refinement"
    write_artifact(validation_refinement_root / "parent_research_run.json", validation_source)
    write_artifact(output / "research_run.json", validation_source)
    _write_progress(output, "VALIDATION_DISCOVERY_REFINEMENT", refs)
    discovery_refinement = _run_discovery_refinement(
        request, output, refinements, validation, root=Path(root),
        source_research_run_id=validation_source["artifact_id"])
    if discovery_refinement.get("artifact_id"):
        refs["validation_discovery_refinement"] = discovery_refinement["artifact_id"]
    _write_progress(output, "FINAL_BACKTEST", refs)
    backtests = backtest_stage.run(
        output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
        output / "06_validation_backtest" / validation_stage.ARTIFACT_NAME,
        evidence_path,
        output / "08_final_backtest", schedule=request.schedule, root=root,
        final_backtest_for_all_validation=request.final_backtest_for_all_validation)
    refs["backtest_set"] = backtests["artifact_id"]
    backtest_state = backtests["payload"]["state"]
    final_backtest_executed = backtest_state == "BACKTEST_COMPLETE"
    validation_backtest_executed = any(
        (unit.get("backtest") or {}).get("state") == "BACKTEST_COMPLETE"
        for unit in validation["payload"].get("units", {}).values())
    run_artifact = create_artifact("research_run", {
        "state": backtest_state, "request": {"profile": request.profile.as_input(),
        "schedule": request.schedule.as_input(), "hypothesis_ids": list(request.hypothesis_ids),
        "evidence_source": (str(effective_evidence_source)
                            if effective_evidence_source is not None else None),
        "pre_validation_source": str(pre_root),
        "discovery_loss_context": (str(request.discovery_loss_context)
                                   if request.discovery_loss_context is not None else None),
        "research_lens_id": request.research_lens_id,
        "direct_evidence_fallback": request.direct_evidence_fallback,
        "entry_refinement_branch_quantiles": list(
            request.entry_refinement_branch_quantiles),
        "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
        "auto_discovery_refinement": request.auto_discovery_refinement,
        "final_backtest_for_all_validation": request.final_backtest_for_all_validation,
        "direct_evidence_fallback_used": "direct_evidence_candidate_set" in refs},
        "artifacts": refs,
        "validation_backtest_executed": validation_backtest_executed,
        "final_backtest_executed": final_backtest_executed,
        "terminal_oos_opened": False,
        **({"direct_candidate_reservation": reservation}
           if reservation is not None else {}),
        **({"research_capacity_reservation": capacity}
           if capacity is not None else {}),
    }, parents=refs)
    write_artifact(output / "research_run.json", run_artifact)
    if final_backtest_executed:
        registry_refresh = registry_stage.run(
            run_artifact, backtest_set_id=backtests["artifact_id"],
            output=output / "08_final_registry_refresh")
        refs["final_registry_refresh"] = registry_refresh["artifact_id"]
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "research.v2", "state": backtest_state, "artifacts": refs,
        "terminal_oos_opened": False,
    }))
    _write_progress(output, "TERMINAL", refs, terminal_state=backtest_state)
    return {"state": backtest_state, "artifacts": refs, "run": run_artifact,
            "validation": validation, "specifications": specifications,
            "implementations": implementations, "searches": searches, "backtests": backtests,
            "discovery_refinement": discovery_refinement}
