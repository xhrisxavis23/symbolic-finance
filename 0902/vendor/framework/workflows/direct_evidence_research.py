"""저장 Feature Profile의 직접 Evidence 후보만 끝까지 실행하는 보조 Workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import (ENTRY_REFINEMENT_POLICY_VERSION, TICK_ROOT, execution_workers, now_utc,
                      run_manifest, write_json)
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import (candidate_library, direct_candidate_reservation, refinement,
                       research_capacity)
from ..modules.profile import ProfileSelection
from ..modules.refinement import RefinementConfig, config_payload, resolve_config
from ..modules.schedule import ResearchSchedule
from ..stages import (backtest_stage, direct_evidence_stage, evidence_stage,
                      execution_anchor_stage,
                      implementation_stage, parameter_search_stage, refinement_stage,
                      registry_stage, validation_stage)


@dataclass(frozen=True)
class DirectEvidenceResearchRequest:
    profile: ProfileSelection
    schedule: ResearchSchedule
    # 같은 저장 Profile에서 이미 만든 Evidence를 공유한다. 후보별 Specification 이후
    # 계약은 분리하되, feature/evidence를 다시 빌드하지 않는다.
    evidence_source: Path | None = None
    relation_only: bool = False
    temporal_relation_only: bool = False
    temporal_episode_relation_only: bool = False
    incremental_relation_only: bool = False
    late_trigger_cross_only: bool = False
    execution_state_only: bool = False
    entry_lifecycle_continuity_only: bool = False
    execution_outcome_pair_only: bool = False
    execution_temporal_only: bool = False
    execution_temporal_fill_only: bool = False
    execution_stop_avoidance_only: bool = False
    execution_recovery_path_only: bool = False
    execution_early_reversal_path_only: bool = False
    execution_recovery_path_temporal_only: bool = False
    execution_recovery_path_fill_only: bool = False
    execution_recovery_path_temporal_fill_only: bool = False
    execution_stop_avoidance_fill_only: bool = False
    execution_outcome_fill_limit: int | None = None
    execution_stop_avoidance_fill_limit: int | None = None
    state_quantile_mode: str = "SHARED_STATE_QUANTILE"
    candidate_ids: tuple[str, ...] = ()
    # 비어 있으면 기존 단일 guard refinement다. 값이 있으면 Search grid의 모든 q를
    # Discovery 손실 장부로 각각 보완해, Search가 완결된 entry branch만 비교한다.
    entry_refinement_branch_quantiles: tuple[float, ...] = ()
    auto_entry_refinement_branches: bool = True
    # Validation 기반 Discovery Refinement에서 열린 후보는 새 q lock 뒤 Final로 간다.
    final_backtest_for_all_validation: bool = False
    validation_refinement_successor: bool = False
    # 후보별 campaign plan에서 온 실행이면, entry rule이 아니라 그 precommit의
    # provenance만 terminal artifact 부모에 남긴다.
    campaign_plan_id: str | None = None
    resume: bool = False


PROGRESS_FILE = "research_progress.json"


def _write_progress(output: Path, stage: str, refs: dict[str, str] | None = None, *,
                    terminal_state: str | None = None) -> None:
    """직접 Evidence 실행도 현재 Stage를 원장과 별도로 남긴다."""
    write_json(Path(output) / PROGRESS_FILE, {
        "schema": "research_progress.v1",
        "updated_at": now_utc(),
        "state": "COMPLETE" if terminal_state is not None else "IN_PROGRESS",
        "stage": stage,
        "terminal_state": terminal_state,
        "owner_pid": os.getpid(),
        "artifacts": dict(refs or {}),
        "execution_workers": execution_workers(),
    })


def _prior_direct_entry_logic_fingerprints(library: Mapping[str, Any]) -> tuple[str, ...]:
    """같은 Feature Profile에서 이미 연 직접 Evidence entry logic만 모은다.

    이력의 PnL·Search·Validation 결과는 읽지 않는다. q 또는 guard로 달라진 같은 base
    signal을 새 알고리즘처럼 다시 실행하지 않기 위한 구조 비교다.
    """
    return tuple(sorted({str(record.get("entry_logic_fingerprint") or "")
                         for record in library.get("records") or []
                         if record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"
                         and record.get("evaluation_opened")
                         and record.get("entry_logic_fingerprint")}))


def run(request: DirectEvidenceResearchRequest, output: Path, *,
        root: Path = TICK_ROOT) -> dict[str, Any]:
    """Agent 해석 없이, 저장 Profile의 안정적 단일 관측 상태를 같은 후속 단계로 보낸다."""
    output = Path(output)
    _write_progress(output, "EVIDENCE")
    refinement_config = RefinementConfig(
        branch_quantiles=tuple(sorted({
            float(value) for value in request.entry_refinement_branch_quantiles
        })),
        auto_branch_search_quantiles=bool(request.auto_entry_refinement_branches))
    reused: list[str] = []
    capacity = None
    if request.evidence_source is not None:
        evidence_path = Path(request.evidence_source)
        evidence = read_artifact(evidence_path, kind="evidence_package")
        anchor_replay_id = str(evidence["parents"].get("execution_anchor_replay") or "")
        if not anchor_replay_id:
            raise ValueError("공유 Evidence에 execution anchor lineage가 없다")
        if not _same_json_value(evidence["payload"].get("selection"), request.profile.as_input()):
            raise ValueError("공유 Evidence의 Feature Profile 입력이 현재 요청과 다르다")
        reused.append("shared_evidence")
    else:
        capacity = research_capacity.reserve(output=output)
        if capacity["state"] == "WAITING":
            return _finish(
                output, request, {}, "CAPACITY_WAITING", False, False, reused,
                reason="execution-anchor replay 전 전역 실행 slot이 모두 사용 중이다",
                capacity=capacity)
        anchor_replay = execution_anchor_stage.run(
            request.profile, output / "00_execution_anchor_replay", root=root)
        anchor_replay_id = anchor_replay["artifact_id"]
        evidence_path = output / "01_evidence" / evidence_stage.ARTIFACT_NAME
        if request.resume and evidence_path.is_file():
            evidence = read_artifact(evidence_path, kind="evidence_package")
            if (not _same_json_value(evidence["payload"].get("selection"), request.profile.as_input())
                    or evidence["parents"].get("execution_anchor_replay")
                    != anchor_replay_id):
                evidence = evidence_stage.run(
                    request.profile, output / "01_evidence",
                    execution_anchor_replay_path=(output / "00_execution_anchor_replay"
                                                  / execution_anchor_stage.ARTIFACT_NAME))
            else:
                reused.append("evidence")
        else:
            evidence = evidence_stage.run(
                request.profile, output / "01_evidence",
                execution_anchor_replay_path=(output / "00_execution_anchor_replay"
                                              / execution_anchor_stage.ARTIFACT_NAME))

    direct_candidates_path = output / "02_specification" / direct_evidence_stage.CANDIDATES_ARTIFACT_NAME
    specification_path = output / "02_specification" / direct_evidence_stage.ARTIFACT_NAME
    library_path = output / "00_candidate_library" / candidate_library.ARTIFACT_NAME
    # 이미 고정한 candidate/specification을 재개할 때는 새 library scan이 entry를
    # 다시 고르지 않도록, 같은 Profile의 저장 library를 그대로 쓴다. capacity 대기는
    # 재개할 때마다 수만 artifact를 다시 훑을 이유가 없다.
    if request.resume and direct_candidates_path.is_file() and specification_path.is_file() \
            and library_path.is_file():
        saved_library = read_artifact(library_path, kind="candidate_library")
        if _same_json_value(saved_library["payload"].get("direct_profile"),
                            request.profile.as_input()):
            library = saved_library
            reused.append("candidate_library")
        else:
            library = candidate_library.build(
                output / "00_candidate_library", direct_profile=request.profile.as_input())
    else:
        # Direct candidate도 다른 직접 후보의 동일 signal을 다시 실행하면 안 된다. 단,
        # 명시 q-branch는 이미 열린 base signal의 Discovery entry-refinement 연구이므로
        # 같은 알고리즘을 새로 세지 않는다는 전제 아래 그 실행만 허용한다.
        library = candidate_library.build(
            output / "00_candidate_library", direct_profile=request.profile.as_input())
    prior_entry_logic = (() if request.entry_refinement_branch_quantiles
                         else _prior_direct_entry_logic_fingerprints(library["payload"]))
    _write_progress(output, "SPECIFICATION")
    if request.resume and direct_candidates_path.is_file() and specification_path.is_file():
        candidates = read_artifact(direct_candidates_path, kind="direct_evidence_candidate_set")
        specifications = read_artifact(specification_path, kind="executable_specification_set")
        if candidates["parents"].get("evidence_package") != evidence["artifact_id"]:
            raise ValueError("재사용 Direct Evidence 후보와 Evidence의 lineage가 다르다")
        if specifications["parents"] != {
                "evidence_package": evidence["artifact_id"],
                "direct_evidence_candidate_set": candidates["artifact_id"]}:
            raise ValueError("재사용 Specification과 Direct Evidence 후보의 lineage가 다르다")
        wanted = {"source_types": list(_source_types(request))}
        if request.execution_outcome_fill_limit is not None:
            wanted["execution_outcome_fill_compound_limit"] = int(
                request.execution_outcome_fill_limit)
        if request.execution_stop_avoidance_fill_limit is not None:
            wanted["execution_stop_avoidance_fill_compound_limit"] = int(
                request.execution_stop_avoidance_fill_limit)
        if request.candidate_ids:
            wanted["candidate_ids"] = [str(value) for value in request.candidate_ids]
        if request.state_quantile_mode != "SHARED_STATE_QUANTILE":
            wanted["state_quantile_mode"] = request.state_quantile_mode
        if request.entry_lifecycle_continuity_only:
            wanted["entry_lifecycle_mode"] = "CANCEL_WHEN_ENTRY_SIGNAL_FALSE"
        # 이미 고정된 후보를 재개할 때 현재 library는 새 sibling 실행 때문에 달라질 수 있다.
        # 중복 제외 fingerprint는 *처음 후보를 고를 때*만 쓰인 내부 history 입력이므로,
        # 여기서 다시 계산한 값으로 immutable candidate set을 거부하지 않는다.
        actual_mode = dict(candidates["payload"].get("selection_mode") or {})
        actual_mode.pop("excluded_entry_logic_fingerprints", None)
        if actual_mode != wanted:
            raise ValueError("재사용 Direct Evidence 후보의 선택 모드가 현재 요청과 다르다")
        direct = {"candidates": candidates, "specifications": specifications}
        reused.append("direct_evidence_specification")
    else:
        direct = direct_evidence_stage.run(
            evidence_path, output / "02_specification",
            source_types=_source_types(request), candidate_ids=request.candidate_ids,
            exclude_entry_logic_fingerprints=prior_entry_logic,
            execution_outcome_fill_compound_limit=request.execution_outcome_fill_limit,
            execution_stop_avoidance_fill_compound_limit=(
                request.execution_stop_avoidance_fill_limit),
            state_quantile_mode=request.state_quantile_mode,
            entry_lifecycle_mode=("CANCEL_WHEN_ENTRY_SIGNAL_FALSE"
                                  if request.entry_lifecycle_continuity_only
                                  else "HOLD_THROUGH"))
    refs: dict[str, str] = {
        "execution_anchor_replay": anchor_replay_id,
        "evidence_package": evidence["artifact_id"],
        "candidate_library": library["artifact_id"],
        "direct_evidence_candidate_set": direct["candidates"]["artifact_id"],
        "executable_specification_set": direct["specifications"]["artifact_id"],
    }
    if request.campaign_plan_id is not None:
        refs["direct_evidence_campaign_plan"] = str(request.campaign_plan_id)
    specifications = direct["specifications"]
    if specifications["payload"].get("state") != "SPECIFICATION_COMPLETE":
        return _finish(output, request, refs, "NO_DIRECT_EVIDENCE_CANDIDATE", False, False, reused)
    if capacity is None:
        capacity = research_capacity.reserve(output=output)
    if capacity["state"] == "WAITING":
        return _finish(output, request, refs, "CAPACITY_WAITING", False, False, reused,
                       reason="다른 active 연구 replay가 전역 실행 slot을 사용 중이다",
                       capacity=capacity)
    reservation = direct_candidate_reservation.reserve(
        direct["candidates"]["payload"].get("candidates") or [],
        profile=request.profile.as_input(), output=output)
    if reservation["state"] == "CONFLICT":
        reason = "같은 Direct entry가 다른 active run에서 Entry Refinement 중이다"
        return _finish(output, request, refs, "ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE",
                       False, False, reused, reason=reason, reservation=reservation,
                       capacity=capacity)
    branching_ids = refinement.branching_hypothesis_ids(
        specifications["payload"].get("units") or {}, refinement_config)
    implementation_path = output / "03_implementation" / implementation_stage.ARTIFACT_NAME
    refinement_path = output / "04_entry_refinement" / refinement_stage.ARTIFACT_NAME
    _write_progress(output, "IMPLEMENTATION", refs)
    if request.resume and implementation_path.is_file():
        implementations = read_artifact(implementation_path, kind="implementation_set")
        if implementations["parents"] != {"executable_specification_set": specifications["artifact_id"]}:
            raise ValueError("재사용 Implementation과 Specification의 lineage가 다르다")
        has_equivalent = any(
            unit.get("state") == "EXECUTION_EQUIVALENT"
            for unit in implementations["payload"].get("units", {}).values()
            if isinstance(unit, Mapping))
        if (refinement_config.branch_quantiles or branching_ids) and has_equivalent:
            # 기존 기본 계약의 q별 refinement 재평가는 중복 알고리즘 생성이 아니다.
            # 따라서 base-contract 중복 차단만 한 번 우회해 이후 Search를 연다.
            implementations = implementation_stage.run(
                specification_path, output / "03_implementation",
                allow_equivalent_for_refinement=bool(refinement_config.branch_quantiles),
                allow_equivalent_for_refinement_ids=branching_ids)
        elif not refinement_path.is_file():
            # 예전 worker가 Implementation 뒤 Entry Refinement 전에 죽으면, 당시에는
            # 없던 completed-search contract가 뒤에 생겼을 수 있다. 그대로 stale READY
            # artifact를 쓰면 이미 끝난 같은 direct entry를 다시 장시간 replay한다.
            # 아직 downstream artifact가 없을 때만 현재 completed-search identity를
            # 다시 확인한다. refinement가 있으면 그 immutable 실행을 그대로 재개한다.
            implementations = implementation_stage.run(specification_path, output / "03_implementation")
        else:
            reused.append("implementation")
    else:
        implementations = implementation_stage.run(
            specification_path, output / "03_implementation",
            allow_equivalent_for_refinement=bool(refinement_config.branch_quantiles),
            allow_equivalent_for_refinement_ids=branching_ids)
    refs["implementation_set"] = implementations["artifact_id"]
    implementation_units = list(implementations["payload"].get("units", {}).values())
    # 같은 entry·execution·parameter interface면 새 가설이 아니라 같은 계약의
    # 재실행이다. 이를 "구현 미준비"로 흘려 보내면 Search/Validation을 열지 않았다는
    # 사실이 마치 성과 반증처럼 기록된다. Direct Evidence도 일반 Research workflow와
    # 같은 지점에서 명시적으로 멈춘다.
    if implementation_units and all(
            unit.get("state") == "EXECUTION_EQUIVALENT" for unit in implementation_units) \
            and not refinement_config.branch_quantiles and not branching_ids:
        return _finish(output, request, refs, "EXECUTION_EQUIVALENT", False, False, reused,
                       reason="모든 직접 Evidence 구현 계약이 기존 Sandbox entry·execution 계약과 같다",
                       reservation=reservation, capacity=capacity)
    expected_refinement_parents = {
        "executable_specification_set": specifications["artifact_id"],
        "implementation_set": implementations["artifact_id"],
        "evidence_package": evidence["artifact_id"],
    }
    _write_progress(output, "ENTRY_REFINEMENT", refs)
    if request.resume and refinement_path.is_file():
        refinements = read_artifact(refinement_path, kind="entry_refinement_set")
        if refinements["parents"] != expected_refinement_parents:
            raise ValueError("재사용 Entry refinement의 lineage가 다르다")
        if _refinement_policy_matches(
                refinements, implementations, refinement_config,
                specifications=specifications["payload"].get("units") or {}):
            reused.append("entry_refinement")
        else:
            # Entry Refinement 정책도 최종 entry 계약의 일부다. 이전 policy artifact를
            # 이어 쓰면 새로운 guard 탐색이 실행되지 않은 채 Search부터 재개된다.
            refinements = refinement_stage.run(
                specification_path, implementation_path, evidence_path,
                output / "04_entry_refinement", root=root, config=refinement_config)
    else:
        refinements = refinement_stage.run(
            specification_path, implementation_path, evidence_path,
                output / "04_entry_refinement", root=root, config=refinement_config)
    refs["entry_refinement_set"] = refinements["artifact_id"]
    _write_progress(output, "PARAMETER_SEARCH", refs)
    searches = parameter_search_stage.run(
        specification_path, output / "03_implementation" / implementation_stage.ARTIFACT_NAME,
        output / "04_entry_refinement" / refinement_stage.ARTIFACT_NAME,
        evidence_path,
        output / "05_parameter_search", schedule=request.schedule, root=root)
    refs["parameter_search_set"] = searches["artifact_id"]
    if request.validation_refinement_successor:
        _write_progress(output, "FINAL_BACKTEST", refs)
        backtests = backtest_stage.run_from_validation_refinement(
            output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
            evidence_path,
            output / "07_backtest", schedule=request.schedule, root=root)
        refs["backtest_set"] = backtests["artifact_id"]
        return _finish(output, request, refs, backtests["payload"]["state"], False,
                       backtests["payload"]["state"] == "BACKTEST_COMPLETE", reused,
                       reservation=reservation, capacity=capacity)
    _write_progress(output, "VALIDATION_BACKTEST", refs)
    validation = validation_stage.run(
        output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
        evidence_path,
        output / "06_validation_backtest", schedule=request.schedule, root=root)
    refs["validation_backtest_set"] = validation["artifact_id"]
    _write_progress(output, "FINAL_BACKTEST", refs)
    backtests = backtest_stage.run(
        output / "05_parameter_search" / parameter_search_stage.ARTIFACT_NAME,
        output / "06_validation_backtest" / validation_stage.ARTIFACT_NAME,
        evidence_path,
        output / "07_backtest", schedule=request.schedule, root=root,
        final_backtest_for_all_validation=request.final_backtest_for_all_validation)
    refs["backtest_set"] = backtests["artifact_id"]
    validation_executed = any(
        (unit.get("backtest") or {}).get("state") == "BACKTEST_COMPLETE"
        for unit in validation["payload"].get("units", {}).values())
    return _finish(output, request, refs, backtests["payload"]["state"], validation_executed,
                   backtests["payload"]["state"] == "BACKTEST_COMPLETE", reused,
                   reservation=reservation, capacity=capacity)


def _finish(output: Path, request: DirectEvidenceResearchRequest, refs: dict[str, str],
            state: str, validation_executed: bool, final_backtest_executed: bool,
            reused: list[str], reason: str | None = None,
            reservation: Mapping[str, Any] | None = None,
            capacity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_artifact = create_artifact("research_run", {
        "state": state,
        "source_type": "DIRECT_EVIDENCE_RESEARCH",
        "request": {"profile": request.profile.as_input(), "schedule": request.schedule.as_input(),
                    "evidence_source": (str(request.evidence_source)
                                        if request.evidence_source is not None else None),
                    "relation_only": request.relation_only,
                    "temporal_relation_only": request.temporal_relation_only,
                    "temporal_episode_relation_only": request.temporal_episode_relation_only,
                    "incremental_relation_only": request.incremental_relation_only,
                    "late_trigger_cross_only": request.late_trigger_cross_only,
                    "execution_state_only": request.execution_state_only,
                    "entry_lifecycle_continuity_only": request.entry_lifecycle_continuity_only,
                    "execution_outcome_pair_only": request.execution_outcome_pair_only,
                    "execution_temporal_only": request.execution_temporal_only,
                    "execution_temporal_fill_only": request.execution_temporal_fill_only,
                    "execution_stop_avoidance_only": request.execution_stop_avoidance_only,
                    "execution_recovery_path_only": request.execution_recovery_path_only,
                    "execution_early_reversal_path_only": (
                        request.execution_early_reversal_path_only),
                    "execution_recovery_path_temporal_only": (
                        request.execution_recovery_path_temporal_only),
                    "execution_recovery_path_fill_only": request.execution_recovery_path_fill_only,
                    "execution_recovery_path_temporal_fill_only": (
                        request.execution_recovery_path_temporal_fill_only),
                    "execution_stop_avoidance_fill_only": request.execution_stop_avoidance_fill_only,
                    "execution_outcome_fill_limit": request.execution_outcome_fill_limit,
                    "execution_stop_avoidance_fill_limit": (
                        request.execution_stop_avoidance_fill_limit),
                    "state_quantile_mode": request.state_quantile_mode,
                    "candidate_ids": [str(value) for value in request.candidate_ids],
                    "entry_refinement_branch_quantiles": list(
                        request.entry_refinement_branch_quantiles),
                    "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
                    "final_backtest_for_all_validation": request.final_backtest_for_all_validation,
                    "validation_refinement_successor": request.validation_refinement_successor,
                    "campaign_plan_id": request.campaign_plan_id,
                    "resume": request.resume},
        "artifacts": refs,
        "reused_pre_search_stages": list(reused),
        "validation_backtest_executed": validation_executed,
        "final_backtest_executed": final_backtest_executed,
        "terminal_oos_opened": False,
        **({"reason": str(reason)} if reason else {}),
        **({"direct_candidate_reservation": dict(reservation)} if reservation is not None else {}),
        **({"research_capacity_reservation": dict(capacity)} if capacity is not None else {}),
    }, parents=refs)
    write_artifact(output / "research_run.json", run_artifact)
    if final_backtest_executed:
        registry_refresh = registry_stage.run(
            run_artifact, backtest_set_id=refs["backtest_set"],
            output=output / "08_final_registry_refresh")
        refs["final_registry_refresh"] = registry_refresh["artifact_id"]
    write_json(output / "run_manifest.json", run_manifest(output, {
        "workflow": "direct_evidence_research.v3", "state": state, "artifacts": refs,
        "terminal_oos_opened": False,
    }))
    _write_progress(output, "TERMINAL", refs, terminal_state=state)
    return {"state": state, "artifacts": refs, "run": run_artifact}


def _same_json_value(left: Any, right: Any) -> bool:
    """저장 artifact(JSON)과 실행 요청(tuple)의 같은 구조를 비교한다."""
    return json.dumps(left, ensure_ascii=False, sort_keys=True, default=str) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, default=str)


def _refinement_policy_matches(refinements: Mapping[str, Any],
                               implementations: Mapping[str, Any],
                               config: RefinementConfig = RefinementConfig(), *,
                               specifications: Mapping[str, Any] | None = None) -> bool:
    """재개 대상의 실행 가능한 모든 후보가 현재 refinement policy로 만들어졌는지."""
    saved_units = (refinements.get("payload") or {}).get("units") or {}
    implementation_units = (implementations.get("payload") or {}).get("units") or {}
    for hypothesis_id, implementation in implementation_units.items():
        if not isinstance(implementation, Mapping):
            continue
        if implementation.get("state") != "READY_FOR_PARAMETER_SEARCH":
            continue
        unit = (specifications or {}).get(str(hypothesis_id)) or {}
        spec = unit.get("spec") if isinstance(unit, Mapping) else None
        effective_config = (resolve_config(spec, config)
                            if isinstance(spec, Mapping) else config)
        saved = saved_units.get(str(hypothesis_id))
        if (not isinstance(saved, Mapping)
                or saved.get("entry_refinement_policy_version")
                != ENTRY_REFINEMENT_POLICY_VERSION
                or saved.get("config") != config_payload(effective_config)):
            return False
    return True


def _source_types(request: DirectEvidenceResearchRequest) -> tuple[str, ...]:
    selected = sum(bool(value) for value in (
        request.relation_only, request.temporal_relation_only,
        request.temporal_episode_relation_only, request.incremental_relation_only,
        request.late_trigger_cross_only,
        request.execution_state_only, request.entry_lifecycle_continuity_only,
        request.execution_outcome_pair_only,
        request.execution_temporal_only, request.execution_temporal_fill_only,
        request.execution_stop_avoidance_only,
        request.execution_recovery_path_only,
        request.execution_early_reversal_path_only,
        request.execution_recovery_path_temporal_only,
        request.execution_recovery_path_fill_only,
        request.execution_recovery_path_temporal_fill_only,
        request.execution_stop_avoidance_fill_only))
    if selected > 1:
        raise ValueError("직접 Evidence 관계 후보의 선택 모드는 하나만 고를 수 있다")
    if request.temporal_episode_relation_only:
        return ("DIRECT_EVIDENCE_TEMPORAL_EPISODE_RELATION",)
    if request.incremental_relation_only:
        return ("DIRECT_EVIDENCE_INCREMENTAL_RELATION",)
    if request.late_trigger_cross_only:
        return ("DIRECT_EVIDENCE_LATE_TRIGGER_CROSS",)
    if request.execution_state_only:
        return ("DIRECT_DISCOVERY_ANCHOR_EXECUTION_STATE_EVIDENCE",)
    if request.entry_lifecycle_continuity_only:
        return ("DIRECT_DISCOVERY_ANCHOR_EXECUTION_STATE_EVIDENCE",)
    if request.execution_outcome_pair_only:
        return ("DIRECT_EXECUTION_OUTCOME_PAIR_EVIDENCE",)
    if request.execution_temporal_fill_only:
        return ("DIRECT_EXECUTION_TEMPORAL_FILL_COMPOUND_EVIDENCE",)
    if request.execution_stop_avoidance_only:
        return ("DIRECT_DISCOVERY_STOP_AVOIDANCE_EVIDENCE",)
    if request.execution_recovery_path_only:
        return ("DIRECT_DISCOVERY_RECOVERY_PATH_EVIDENCE",)
    if request.execution_early_reversal_path_only:
        return ("DIRECT_DISCOVERY_EARLY_REVERSAL_PATH_EVIDENCE",)
    if request.execution_recovery_path_temporal_only:
        return ("DIRECT_DISCOVERY_RECOVERY_PATH_TEMPORAL_EVIDENCE",)
    if request.execution_recovery_path_fill_only:
        return ("DIRECT_EXECUTION_RECOVERY_PATH_FILL_COMPOUND_EVIDENCE",)
    if request.execution_recovery_path_temporal_fill_only:
        return ("DIRECT_EXECUTION_RECOVERY_PATH_TEMPORAL_FILL_COMPOUND_EVIDENCE",)
    if request.execution_stop_avoidance_fill_only:
        return ("DIRECT_EXECUTION_STOP_AVOIDANCE_FILL_COMPOUND_EVIDENCE",)
    if request.execution_temporal_only:
        return ("DIRECT_DISCOVERY_EXECUTION_TEMPORAL_EVIDENCE",)
    if request.temporal_relation_only:
        return ("DIRECT_EVIDENCE_TEMPORAL_RELATION",)
    if request.relation_only:
        return ("DIRECT_EVIDENCE_RELATION",)
    return ()
