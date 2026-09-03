"""독립 Direct Evidence 후보를 재생 전에 분리·예약하는 Workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from collections import Counter

from ..config import TICK_ROOT, clean, read_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import candidate_library
from ..modules.profile import ProfileSelection
from ..modules.schedule import ResearchSchedule
from ..stages import direct_evidence_stage
from . import direct_evidence_research
from .direct_evidence_research import DirectEvidenceResearchRequest


PLAN_SCHEMA = "direct_evidence_campaign_plan.v1"
PLAN_ARTIFACT_NAME = "direct_evidence_campaign_plan.json"
SANDBOX_ROOT = Path(__file__).resolve().parents[2]


def _launch_priority(candidate: Mapping[str, Any]) -> tuple[int, str]:
    """재생 전 관측된 entry 정보량만으로 campaign 순서를 정한다.

    순서는 실행 결과가 아니라 candidate가 이미 보존한 Evidence source type만 쓴다.
    따라서 Search/Validation/Final을 보고 후보를 고르는 우회 통로가 아니다.
    """
    source_type = str(candidate.get("source_type") or "")
    if source_type.startswith("DIRECT_EXECUTION_"):
        return 0, "체결 상태와 고정-exit 결과를 함께 관측한 entry"
    if ("RECOVERY_PATH" in source_type or "STOP_AVOIDANCE" in source_type
            or "EARLY_REVERSAL" in source_type):
        return 1, "고정-exit 손실·회복 경로를 구분한 entry"
    if source_type == "DIRECT_EVIDENCE_TEMPORAL_EPISODE_RELATION":
        return 2, "context 구간당 한 번인 관측 순서 entry"
    if source_type == "DIRECT_EVIDENCE_TEMPORAL_RELATION":
        return 3, "관측된 context→trigger 순서 entry"
    if source_type == "DIRECT_EVIDENCE_INCREMENTAL_RELATION":
        return 4, "독립 상태 결합 entry"
    if source_type == "DIRECT_EVIDENCE_RELATION":
        return 5, "동시 관측 상태 관계 entry"
    return 6, "단일 직접 관측 상태 entry"


@dataclass(frozen=True)
class CampaignPlanRequest:
    """하나의 Evidence family를 후보별 독립 연구 run으로 나누는 입력."""

    request: DirectEvidenceResearchRequest
    candidate_output_root: Path


def plan(request: CampaignPlanRequest, output: Path) -> dict[str, Any]:
    """후보·일정을 먼저 고정한다. 여기서는 Entry Refinement replay를 시작하지 않는다."""
    base = request.request
    if base.evidence_source is None:
        raise ValueError("후보별 campaign plan은 이미 만든 동일 Profile Evidence를 요구한다")
    evidence = read_artifact(Path(base.evidence_source), kind="evidence_package")
    if not direct_evidence_research._same_json_value(
            evidence["payload"].get("selection"), base.profile.as_input()):
        raise ValueError("campaign plan Evidence와 Feature Profile 입력이 다르다")
    base.schedule.validate(base.profile.dates)

    output = Path(output)
    library = candidate_library.build(
        output / "00_candidate_library", direct_profile=base.profile.as_input())
    # q branch는 이미 열린 같은 base 계약의 재평가다. 새 base candidate를 제외할 때와
    # 달리, 이 계획 자체가 그 재평가를 명시했다면 candidate를 통과시킨다.
    excluded = (() if base.entry_refinement_branch_quantiles else
                direct_evidence_research._prior_direct_entry_logic_fingerprints(
                    library["payload"]))
    direct = direct_evidence_stage.run(
        Path(base.evidence_source), output / "01_candidate_selection",
        source_types=direct_evidence_research._source_types(base),
        candidate_ids=base.candidate_ids,
        exclude_entry_logic_fingerprints=excluded,
        execution_outcome_fill_compound_limit=base.execution_outcome_fill_limit,
        execution_stop_avoidance_fill_compound_limit=(
            base.execution_stop_avoidance_fill_limit),
        state_quantile_mode=base.state_quantile_mode)
    candidates = list(direct["candidates"]["payload"].get("candidates") or [])
    candidates.sort(key=lambda candidate: (
        _launch_priority(candidate)[0], str(candidate.get("hypothesis_id") or "")))
    candidate_output_root = Path(request.candidate_output_root).resolve()
    units = [_unit_input(base, candidate, candidate_output_root)
             for candidate in candidates]
    artifact = create_artifact("direct_evidence_campaign_plan", {
        "schema": PLAN_SCHEMA,
        "purpose": (
            "서로 다른 Direct Evidence 후보가 Entry Refinement 뒤 서로 기다리지 않도록, "
            "후보 identity와 Search·Validation·Final 일정을 재생 전에 고정한다."),
        "profile": base.profile.as_input(),
        "schedule": base.schedule.as_input(),
        "base_request": _request_input(base),
        "candidate_output_root": str(candidate_output_root),
        "candidate_units": units,
        "candidate_count": len(units),
        "no_replay_started": True,
        "boundary": (
            "이 plan은 후보를 고르기 위한 Discovery Evidence만 읽는다. 후보별 실행은 "
            "동일한 고정 canonical exit와 미리 적힌 Search·Validation·Final 날짜를 쓴다."),
    }, parents={
        "evidence_package": str(evidence["artifact_id"]),
        "candidate_library": str(library["artifact_id"]),
        "direct_evidence_candidate_set": str(direct["candidates"]["artifact_id"]),
    })
    write_artifact(output / PLAN_ARTIFACT_NAME, artifact)
    return artifact


def run_unit(plan_path: Path, candidate_id: str, output: Path | None = None, *,
             resume: bool = False, root: Path = TICK_ROOT) -> dict[str, Any]:
    """미리 고정한 후보 하나만 일반 Direct Evidence Workflow로 보낸다."""
    artifact = read_artifact(Path(plan_path), kind="direct_evidence_campaign_plan")
    payload = artifact["payload"]
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("Direct Evidence campaign plan 형식이 다르다")
    wanted = str(candidate_id)
    unit = next((item for item in payload.get("candidate_units") or []
                 if isinstance(item, Mapping) and str(item.get("candidate_id")) == wanted), None)
    if unit is None:
        raise ValueError(f"campaign plan에 없는 candidate_id: {wanted}")
    request = _request_from_input(unit.get("request") or {})
    expected = clean(unit.get("candidate") or {})
    if str(expected.get("hypothesis_id") or "") != wanted:
        raise ValueError("campaign plan candidate identity가 request와 다르다")
    destination = (Path(output).resolve() if output is not None
                   else _output_path(unit["planned_output"]))
    request = replace(request, campaign_plan_id=str(artifact["artifact_id"]), resume=bool(resume))
    result = direct_evidence_research.run(request, destination, root=Path(root))
    return {**result, "output": str(destination)}


def dispatch(plan_path: Path, output: Path | None = None, *,
             root: Path = TICK_ROOT) -> dict[str, Any]:
    """사전 고정 priority의 아직 열리지 않은 후보 하나만 시작하거나 재개한다."""
    current = status(plan_path)
    unit = next((item for item in current["units"]
                 if item["state"] in {"NOT_STARTED", "CAPACITY_WAITING",
                                      "ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE"}), None)
    if unit is None:
        return {"state": "NO_CAMPAIGN_UNIT_READY", "plan_artifact": current["plan_artifact"],
                "reason": "시작 또는 capacity 재개 가능한 campaign 후보가 없다"}
    candidate_id = str(unit["candidate_id"])
    resumed = unit["state"] != "NOT_STARTED"
    result = run_unit(plan_path, candidate_id, output=output, resume=resumed, root=root)
    return {"state": result["state"], "candidate_id": candidate_id, "resumed": resumed,
            "artifacts": result["artifacts"], "output": result["output"]}


def status(plan_path: Path) -> dict[str, Any]:
    """계획된 후보를 아직 시작하지 않음·진행 중·terminal로 구분해 읽는다."""
    artifact = read_artifact(Path(plan_path), kind="direct_evidence_campaign_plan")
    payload = artifact["payload"]
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("Direct Evidence campaign plan 형식이 다르다")
    units: list[dict[str, Any]] = []
    for item in payload.get("candidate_units") or []:
        if not isinstance(item, Mapping):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        output = _output_path(item.get("planned_output"))
        terminal = output / "research_run.json"
        progress = output / direct_evidence_research.PROGRESS_FILE
        saved_progress = read_json(progress) if progress.is_file() else {}
        if saved_progress.get("state") == "IN_PROGRESS":
            state = "IN_PROGRESS"
            stage = str(saved_progress.get("stage") or "")
        elif terminal.is_file():
            run = read_artifact(terminal, kind="research_run")
            run_payload = run["payload"]
            state = str(run_payload.get("state"))
            stage = "WAITING" if state in {
                "CAPACITY_WAITING", "ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE",
            } else "TERMINAL"
        elif saved_progress:
            state = str(saved_progress.get("state") or "IN_PROGRESS")
            stage = str(saved_progress.get("stage") or "")
        else:
            state, stage = "NOT_STARTED", ""
        priority, priority_reason = _launch_priority(item.get("candidate") or {})
        units.append({"candidate_id": candidate_id, "state": state, "stage": stage,
                      "launch_priority": int(item.get("launch_priority", priority)),
                      "launch_priority_reason": str(item.get("launch_priority_reason") or priority_reason),
                      "output": str(output)})
    units.sort(key=lambda unit: (int(unit["launch_priority"]), str(unit["candidate_id"])))
    states = Counter(item["state"] for item in units)
    return {"schema": "direct_evidence_campaign_status.v1",
            "plan_artifact": artifact["artifact_id"],
            "candidate_count": len(units), "states": dict(sorted(states.items())),
            "units": units}


def _unit_input(base: DirectEvidenceResearchRequest, candidate: Mapping[str, Any],
                output_root: Path) -> dict[str, Any]:
    candidate_id = str(candidate["hypothesis_id"])
    request = replace(base, candidate_ids=(candidate_id,), resume=False)
    priority, priority_reason = _launch_priority(candidate)
    return {
        "candidate_id": candidate_id,
        "entry_logic_fingerprint": str(candidate.get("entry_logic_fingerprint") or ""),
        "launch_priority": priority,
        "launch_priority_reason": priority_reason,
        "candidate": clean(dict(candidate)),
        "request": _request_input(request),
        "planned_output": str(Path(output_root) / candidate_id),
    }


def _request_input(request: DirectEvidenceResearchRequest) -> dict[str, Any]:
    return {
        "profile": request.profile.as_input(),
        "schedule": request.schedule.as_input(),
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
        "execution_early_reversal_path_only": request.execution_early_reversal_path_only,
        "execution_recovery_path_temporal_only": request.execution_recovery_path_temporal_only,
        "execution_recovery_path_fill_only": request.execution_recovery_path_fill_only,
        "execution_recovery_path_temporal_fill_only": (
            request.execution_recovery_path_temporal_fill_only),
        "execution_stop_avoidance_fill_only": request.execution_stop_avoidance_fill_only,
        "execution_outcome_fill_limit": request.execution_outcome_fill_limit,
        "execution_stop_avoidance_fill_limit": request.execution_stop_avoidance_fill_limit,
        "state_quantile_mode": request.state_quantile_mode,
        "candidate_ids": list(request.candidate_ids),
        "entry_refinement_branch_quantiles": list(request.entry_refinement_branch_quantiles),
        "auto_entry_refinement_branches": request.auto_entry_refinement_branches,
        "final_backtest_for_all_validation": request.final_backtest_for_all_validation,
        "validation_refinement_successor": request.validation_refinement_successor,
    }


def _request_from_input(payload: Mapping[str, Any]) -> DirectEvidenceResearchRequest:
    profile_input = payload.get("profile") or {}
    schedule_input = payload.get("schedule") or {}
    profile = ProfileSelection(
        clusters=tuple(profile_input.get("clusters") or []),
        symbols=tuple(profile_input.get("symbols") or []),
        dates=tuple(profile_input.get("dates") or []),
        store=Path(str(profile_input.get("store") or "")),
        discovery_profit_root=(Path(str(profile_input["discovery_profit_root"]))
                               if profile_input.get("discovery_profit_root") else None),
    )
    schedule = ResearchSchedule(
        validation_dates=tuple(schedule_input.get("validation_dates") or []),
        search_fit_dates=tuple(schedule_input.get("search_fit_dates") or []),
        search_confirm_dates=tuple(schedule_input.get("search_confirm_dates") or []),
        backtest_dates=tuple(schedule_input.get("backtest_dates") or []),
        terminal_oos_dates=tuple(schedule_input.get("terminal_oos_dates") or []),
        single_day_discovery_search=bool(schedule_input.get("single_day_discovery_search")),
        allow_discovery_only_lock=bool(schedule_input.get("allow_discovery_only_lock")),
    )
    fields = {
        "relation_only", "temporal_relation_only", "temporal_episode_relation_only",
        "incremental_relation_only", "late_trigger_cross_only", "execution_state_only",
        "entry_lifecycle_continuity_only",
        "execution_outcome_pair_only",
        "execution_temporal_only", "execution_temporal_fill_only",
        "execution_stop_avoidance_only", "execution_recovery_path_only",
        "execution_early_reversal_path_only", "execution_recovery_path_temporal_only",
        "execution_recovery_path_fill_only", "execution_recovery_path_temporal_fill_only",
        "execution_stop_avoidance_fill_only",
    }
    values: dict[str, Any] = {name: bool(payload.get(name)) for name in fields}
    values.update({
        "profile": profile,
        "schedule": schedule,
        "evidence_source": (Path(str(payload["evidence_source"]))
                            if payload.get("evidence_source") else None),
        "execution_outcome_fill_limit": payload.get("execution_outcome_fill_limit"),
        "execution_stop_avoidance_fill_limit": (
            payload.get("execution_stop_avoidance_fill_limit")),
        "state_quantile_mode": str(payload.get("state_quantile_mode")
                                    or "SHARED_STATE_QUANTILE"),
        "candidate_ids": tuple(str(value) for value in payload.get("candidate_ids") or []),
        "entry_refinement_branch_quantiles": tuple(
            float(value) for value in payload.get("entry_refinement_branch_quantiles") or []),
        "auto_entry_refinement_branches": bool(
            payload.get("auto_entry_refinement_branches", True)),
        "final_backtest_for_all_validation": bool(payload.get("final_backtest_for_all_validation")),
        "validation_refinement_successor": bool(payload.get("validation_refinement_successor")),
    })
    return DirectEvidenceResearchRequest(**values)


def _output_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else SANDBOX_ROOT / path
