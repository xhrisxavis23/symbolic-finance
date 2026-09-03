"""Grounding 승인 Agent 가설을 가설별 Research run으로 나눈다."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..config import TICK_ROOT, clean, read_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules.profile import ProfileSelection
from ..modules.schedule import ResearchSchedule
from . import research


PLAN_SCHEMA = "agent_campaign_plan.v1"
PLAN_ARTIFACT_NAME = "agent_campaign_plan.json"


@dataclass(frozen=True)
class AgentCampaignPlanRequest:
    profile: ProfileSelection
    schedule: ResearchSchedule
    pre_validation_source: Path
    candidate_output_root: Path
    # Validation 실패를 읽어 만든 새 후보는 그 날짜를 다시 promotion 관문으로 쓰지
    # 않고, parameter lock 뒤 Final 날짜에서 평가한다.
    final_backtest_for_all_validation: bool = False
    validation_refinement_successor: bool = False


def plan(request: AgentCampaignPlanRequest, output: Path) -> dict[str, Any]:
    """같은 Grounding Validation Plan의 승인 가설과 날짜를 실행 전에 고정한다."""
    pre_root = Path(request.pre_validation_source)
    pre = read_artifact(pre_root / "pre_validation_run.json", kind="pre_validation_run")
    saved = pre["payload"]
    if saved.get("state") != "READY_FOR_SPECIFICATION":
        raise ValueError("Agent campaign은 READY_FOR_SPECIFICATION Pre-validation만 사용한다")
    saved_profile = (saved.get("request") or {}).get("profile")
    if clean(saved_profile) != clean(request.profile.as_input()):
        raise ValueError("Agent campaign의 Feature Profile과 Pre-validation 입력이 다르다")
    request.schedule.validate(request.profile.dates)
    refs = dict(saved.get("artifacts") or {})
    plan_id = str(refs.get("validation_plan") or "")
    if not plan_id:
        raise ValueError("Pre-validation에 Grounding Validation Plan이 없다")
    validation_plan = read_artifact(
        pre_root / "03_grounding" / "validation_plan_artifact.json", kind="validation_plan")
    if validation_plan["artifact_id"] != plan_id:
        raise ValueError("Pre-validation과 Validation Plan artifact가 다르다")
    hypothesis_ids = tuple(str(value) for value in validation_plan["payload"].get("hypothesis_ids") or [])
    if not hypothesis_ids:
        raise ValueError("Validation Plan에 campaign할 Grounding 승인 가설이 없다")
    candidate_output_root = Path(request.candidate_output_root).resolve()
    units = [_unit_input(request, hypothesis_id, candidate_output_root, index)
             for index, hypothesis_id in enumerate(hypothesis_ids)]
    artifact = create_artifact("agent_campaign_plan", {
        "schema": PLAN_SCHEMA,
        "purpose": (
            "같은 Grounding Validation Plan의 서로 다른 Agent 가설이 Entry Refinement 뒤 "
            "서로 기다리지 않도록, 가설 identity와 Search·Validation·Final 일정을 먼저 고정한다."),
        "profile": request.profile.as_input(),
        "schedule": request.schedule.as_input(),
        "final_backtest_for_all_validation": bool(request.final_backtest_for_all_validation),
        "validation_refinement_successor": bool(request.validation_refinement_successor),
        "pre_validation_source": str(pre_root),
        "hypothesis_ids": list(hypothesis_ids),
        "candidate_output_root": str(candidate_output_root),
        "candidate_units": units,
        "candidate_count": len(units),
        "no_replay_started": True,
        "boundary": (
            "이 plan은 Grounding이 이미 승인한 가설 ID와 미리 적힌 날짜만 읽는다. "
            "후보 선택·재개는 Search·Validation·Final 손익을 읽지 않으며, 각 run은 "
            "동일한 고정 canonical exit를 쓴다."),
    }, parents={"pre_validation_run": pre["artifact_id"], "validation_plan": plan_id})
    write_artifact(Path(output) / PLAN_ARTIFACT_NAME, artifact)
    return artifact


def run_unit(plan_path: Path, hypothesis_id: str, output: Path | None = None, *,
             root: Path = TICK_ROOT) -> dict[str, Any]:
    """고정 Agent 가설 하나만 일반 Research workflow로 보낸다."""
    artifact = _read_plan(plan_path)
    wanted = str(hypothesis_id)
    unit = next((item for item in artifact["payload"]["candidate_units"]
                 if str(item.get("hypothesis_id") or "") == wanted), None)
    if unit is None:
        raise ValueError(f"Agent campaign plan에 없는 hypothesis_id: {wanted}")
    request = _request_from_input(unit.get("request") or {})
    destination = (Path(output).resolve() if output is not None
                   else Path(unit["planned_output"]).resolve())
    result = research.run(request, destination, root=Path(root))
    return {**result, "output": str(destination)}


def status(plan_path: Path) -> dict[str, Any]:
    """가설별 실제 진행 상태를 읽는다."""
    artifact = _read_plan(plan_path)
    units: list[dict[str, Any]] = []
    for item in artifact["payload"]["candidate_units"]:
        hypothesis_id = str(item["hypothesis_id"])
        output = Path(item["planned_output"])
        terminal = output / "research_run.json"
        progress = output / research.PROGRESS_FILE
        saved_progress = read_json(progress) if progress.is_file() else {}
        if saved_progress.get("state") == "IN_PROGRESS":
            state, stage = "IN_PROGRESS", str(saved_progress.get("stage") or "")
        elif terminal.is_file():
            run = read_artifact(terminal, kind="research_run")
            state = str(run["payload"].get("state") or "")
            stage = "WAITING" if state == "CAPACITY_WAITING" else "TERMINAL"
        elif saved_progress:
            state = str(saved_progress.get("state") or "IN_PROGRESS")
            stage = str(saved_progress.get("stage") or "")
        else:
            state, stage = "NOT_STARTED", ""
        units.append({"hypothesis_id": hypothesis_id, "state": state, "stage": stage,
                      "launch_index": int(item["launch_index"]), "output": str(output)})
    units.sort(key=lambda item: (item["launch_index"], item["hypothesis_id"]))
    state_counts: dict[str, int] = {}
    for item in units:
        state_counts[item["state"]] = state_counts.get(item["state"], 0) + 1
    return {"schema": "agent_campaign_status.v1", "plan_artifact": artifact["artifact_id"],
            "candidate_count": len(units), "states": dict(sorted(state_counts.items())),
            "units": units}


def dispatch(plan_path: Path, output: Path | None = None, *,
             root: Path = TICK_ROOT) -> dict[str, Any]:
    """Validation Plan의 고정 순서에서 시작 또는 capacity 재개 가능한 가설 하나를 실행한다."""
    current = status(plan_path)
    unit = next((item for item in current["units"]
                 if item["state"] in {"NOT_STARTED", "CAPACITY_WAITING"}), None)
    if unit is None:
        return {"state": "NO_AGENT_CAMPAIGN_UNIT_READY",
                "plan_artifact": current["plan_artifact"],
                "reason": "시작 또는 capacity 재개 가능한 Agent 가설이 없다"}
    result = run_unit(plan_path, unit["hypothesis_id"], output=output, root=root)
    return {"state": result["state"], "hypothesis_id": unit["hypothesis_id"],
            "resumed": unit["state"] == "CAPACITY_WAITING", "artifacts": result["artifacts"],
            "output": result["output"]}


def _read_plan(path: Path) -> dict[str, Any]:
    artifact = read_artifact(Path(path), kind="agent_campaign_plan")
    if artifact["payload"].get("schema") != PLAN_SCHEMA:
        raise ValueError("Agent campaign plan 형식이 다르다")
    return artifact


def _unit_input(request: AgentCampaignPlanRequest, hypothesis_id: str,
                output_root: Path, index: int) -> dict[str, Any]:
    return {
        "hypothesis_id": str(hypothesis_id), "launch_index": index,
        "planned_output": str(output_root / str(hypothesis_id)),
        "request": {
            "profile": request.profile.as_input(), "schedule": request.schedule.as_input(),
            "pre_validation_source": str(request.pre_validation_source),
            "hypothesis_id": str(hypothesis_id), "direct_evidence_fallback": False,
            "final_backtest_for_all_validation": bool(request.final_backtest_for_all_validation),
            "validation_refinement_successor": bool(request.validation_refinement_successor),
        },
    }


def _request_from_input(payload: Mapping[str, Any]) -> research.ResearchRequest:
    profile = payload.get("profile") or {}
    schedule = payload.get("schedule") or {}
    return research.ResearchRequest(
        profile=ProfileSelection(
            clusters=tuple(str(value) for value in profile.get("clusters") or []),
            symbols=tuple(str(value) for value in profile.get("symbols") or []),
            dates=tuple(str(value) for value in profile.get("dates") or []),
            store=Path(profile["store"]),
            discovery_profit_root=(Path(profile["discovery_profit_root"])
                                 if profile.get("discovery_profit_root") else None)),
        schedule=ResearchSchedule(
            validation_dates=tuple(str(value) for value in schedule.get("validation_dates") or []),
            search_fit_dates=tuple(str(value) for value in schedule.get("search_fit_dates") or []),
            search_confirm_dates=tuple(str(value) for value in schedule.get("search_confirm_dates") or []),
            backtest_dates=tuple(str(value) for value in schedule.get("backtest_dates") or []),
            terminal_oos_dates=tuple(str(value) for value in schedule.get("terminal_oos_dates") or []),
            single_day_discovery_search=bool(schedule.get("single_day_discovery_search")),
            allow_discovery_only_lock=bool(schedule.get("allow_discovery_only_lock"))),
        hypothesis_ids=(str(payload["hypothesis_id"]),),
        pre_validation_source=Path(payload["pre_validation_source"]),
        direct_evidence_fallback=False,
        final_backtest_for_all_validation=bool(payload.get("final_backtest_for_all_validation")),
        validation_refinement_successor=bool(payload.get("validation_refinement_successor")),
        auto_discovery_refinement=False)
