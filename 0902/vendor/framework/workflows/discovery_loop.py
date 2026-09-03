"""Discovery 손실과 Validation 탈락을 다음 진입 가설로 잇는 반복 controller.

후속 Agent는 저장 Evidence, 부모 Discovery 원장, 그리고 총손익 0 이하로 끝난
Validation 개발 피드백만 읽는다. Final artifact와 결과는 열지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import TICK_ROOT, clean, read_json, sha256_json, write_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import candidate_library, discovery_loss, validation_feedback
from ..modules.profile import ProfileSelection
from ..modules.schedule import ResearchSchedule
from . import agent_campaign, direct_evidence_campaign
from .direct_evidence_research import DirectEvidenceResearchRequest
from .pre_validation import PreValidationRequest, run as run_pre_validation


PLAN_SCHEMA = "discovery_learning_loop.v1"
PLAN_ARTIFACT_NAME = "discovery_learning_loop_plan.json"
STATE_FILE = "discovery_learning_loop_state.json"


@dataclass(frozen=True)
class DiscoveryLoopPlanRequest:
    source_research: Path
    source_hypothesis_id: str
    max_generations: int = 8
    max_lens_attempts: int = 8
    seed_source_limit: int = 4
    direct_event_fallback: bool = False
    entry_lifecycle_fallback: bool = False


def plan(request: DiscoveryLoopPlanRequest, output: Path) -> dict[str, Any]:
    """한 parent Discovery ledger에서 시작할 후속 세대 수와 평가 일정을 고정한다."""
    if int(request.max_generations) < 1:
        raise ValueError("Discovery loop의 max_generations는 1 이상이어야 한다")
    if int(request.max_lens_attempts) < 0:
        raise ValueError("Discovery loop의 max_lens_attempts는 0 이상이어야 한다")
    if int(request.seed_source_limit) < 1:
        raise ValueError("Discovery loop의 seed_source_limit는 1 이상이어야 한다")
    source = Path(request.source_research).resolve()
    run = read_artifact(source / "research_run.json", kind="research_run")
    profile = _profile_from_run(run)
    schedule = _schedule_from_run(run)
    schedule.validate(profile.dates)
    refs = dict(run["payload"].get("artifacts") or {})
    if not refs.get("entry_refinement_set"):
        raise ValueError("Discovery loop의 시작 run에는 Entry Refinement artifact가 필요하다")
    prior_library = _prior_loop_candidate_library(
        source, str(request.source_hypothesis_id), profile, exclude_output=Path(output))
    seed_sources = _matching_seed_sources(
        source, str(request.source_hypothesis_id), profile, schedule,
        limit=int(request.seed_source_limit))
    for index, item in enumerate(seed_sources):
        item["root_rank"] = index
    source_context_identities = {
        _source_context_key(item["research_output"], str(item["hypothesis_id"])):
        _source_context_identity(Path(str(item["research_output"])),
                                 str(item["hypothesis_id"]))
        for item in seed_sources
    }
    # 새 plan은 첫 root만이 아니라 모든 고정 seed root의 terminal attempt를 물려받는다.
    # 이전에는 plan의 declared source가 아니었던 root의 duplicate/no-hypothesis 결과를
    # 읽지 않아, 같은 Profile·일정 아래 새 plan이 그 base/lens를 다시 열 수 있었다.
    prior_attempts: list[dict[str, Any]] = []
    prior_hypothesis_paths: list[Path] = []
    for item in seed_sources:
        prior_attempts.extend(_prior_loop_attempts(
            Path(str(item["research_output"])), str(item["hypothesis_id"]),
            profile, schedule, exclude_output=Path(output),
            loss_context_identity=source_context_identities[
                _source_context_key(item["research_output"], str(item["hypothesis_id"]))]))
        prior_hypothesis_paths.extend(_prior_loop_hypothesis_paths(
            Path(str(item["research_output"])), str(item["hypothesis_id"]),
            profile, schedule, exclude_output=Path(output)))
    prior_attempts = [
        {key: value for key, value in item.items()}
        for _key, item in sorted({
            _source_key(item): item for item in prior_attempts
        }.items())
    ]
    prior_hypothesis_paths = sorted(set(prior_hypothesis_paths))
    parents = {"source_research_run": run["artifact_id"],
               "source_entry_refinement_set": str(refs["entry_refinement_set"])}
    if prior_library is not None:
        parents["candidate_library_snapshot"] = read_artifact(
            prior_library, kind="candidate_library")["artifact_id"]
    artifact = create_artifact("discovery_learning_loop_plan", {
        "schema": PLAN_SCHEMA,
        "source": {
            "research_output": str(source),
            "hypothesis_id": str(request.source_hypothesis_id),
            "generation": 0,
        },
        "seed_sources": seed_sources,
        "seed_source_limit": int(request.seed_source_limit),
        "seed_source_policy": (
            "같은 저장 Feature Profile과 동일하게 사전 고정된 Search·Validation·Final 날짜를 "
            "가진 완료 Entry Refinement source를 source 경로와 hypothesis ID 고정 순서로 최대 한도까지 쓴다. "
            "Final 지표나 PnL은 읽지 않는다. Validation 총손익 0 이하였던 source만 "
            "다음 세대의 개발 피드백으로 연결할 수 있다."),
        "source_context_identities": source_context_identities,
        "profile": profile.as_input(),
        "schedule": schedule.as_input(),
        "max_generations": int(request.max_generations),
        "max_lens_attempts": int(request.max_lens_attempts),
        "direct_event_fallback": bool(request.direct_event_fallback),
        "entry_lifecycle_fallback": bool(request.entry_lifecycle_fallback),
        "candidate_library_snapshot_source": (
            str(prior_library) if prior_library is not None else None),
        "prior_attempts": prior_attempts,
        "prior_hypothesis_paths": [str(path) for path in prior_hypothesis_paths],
        "candidate_order": "GROUNDING_VALIDATION_PLAN_ORDER",
        "evaluation_schedule_origin": "PARENT_RESEARCH_REQUEST",
        "boundary": (
            "각 successor는 parent Entry Refinement의 Discovery ledger와, Validation 총손익 0 이하인 "
            "경우의 개발 피드백만 Agent 입력으로 읽는다. Final artifact와 지표는 열지 않는다. "
            "읽은 Validation 날짜는 후속 세대에서 개발용 날짜로 취급한다. "
            "평가 날짜는 parent 연구가 실행 전에 request에 기록한 블록을 그대로 사용한다. "
            "중복 또는 가설 부재면 저장된 Evidence lens를 고정 순서로 바꿔 최대 횟수만 재시도한다. "
            "그 lens가 모두 비면 opt-in fallback이 같은 저장 Evidence의 Late Trigger 최초 교차 또는 "
            "지속 상태 신호 소실 시 BID1 주문 취소 Direct 후보를 별도 실행한다."),
    }, parents=parents)
    output = Path(output)
    write_artifact(output / PLAN_ARTIFACT_NAME, artifact)
    write_json(output / STATE_FILE, {
        "schema": "discovery_learning_loop_state.v1",
        "plan_artifact": artifact["artifact_id"], "steps": [], "skipped_sources": [],
    })
    return artifact


def status(plan_path: Path) -> dict[str, Any]:
    """현재 세대와 각 Agent campaign의 실제 진행 상태만 읽는다."""
    artifact = _read_plan(plan_path)
    state = _read_state(Path(plan_path).parent, artifact)
    steps: list[dict[str, Any]] = []
    for step in state["steps"]:
        report = dict(step)
        campaign_path = Path(str(step.get("agent_campaign_plan") or ""))
        if campaign_path.is_file():
            report["campaign"] = agent_campaign.status(campaign_path)
        elif step.get("pre_validation_output"):
            report["pre_validation"] = _pre_validation_status(
                Path(str(step["pre_validation_output"])))
        direct_campaign_path = Path(str(step.get("direct_event_campaign_plan")
                                        or step.get("entry_lifecycle_campaign_plan") or ""))
        if direct_campaign_path.is_file():
            key = ("direct_event_campaign" if step.get("direct_event_campaign_plan")
                   else "entry_lifecycle_campaign")
            report[key] = direct_evidence_campaign.status(direct_campaign_path)
        steps.append(report)
    return {
        "schema": "discovery_learning_loop_status.v1",
        "plan_artifact": artifact["artifact_id"],
        "max_generations": artifact["payload"]["max_generations"],
        "max_lens_attempts": int(artifact["payload"].get("max_lens_attempts", 4)),
        "seed_source_count": len(artifact["payload"].get("seed_sources") or []),
        "prior_attempt_count": len(artifact["payload"].get("prior_attempts") or []),
        "direct_event_fallback": bool(artifact["payload"].get("direct_event_fallback", False)),
        "entry_lifecycle_fallback": bool(artifact["payload"].get("entry_lifecycle_fallback", False)),
        "steps": steps,
        "skipped_sources": list(state["skipped_sources"]),
    }


def dispatch(plan_path: Path, *, root: Path = TICK_ROOT) -> dict[str, Any]:
    """고정 순서로 한 후보를 재개하거나, 다음 Discovery successor를 하나 만든다."""
    plan_path = Path(plan_path)
    artifact = _read_plan(plan_path)
    output = plan_path.parent
    state = _read_state(output, artifact)

    # 이미 만든 campaign에는 Grounding이 고정한 순서가 있다. 먼저 그 중 한 단위를 연다.
    for step in state["steps"]:
        campaign_path = Path(str(step.get("agent_campaign_plan") or ""))
        if not campaign_path.is_file():
            continue
        current = agent_campaign.status(campaign_path)
        if any(unit["state"] in {"NOT_STARTED", "CAPACITY_WAITING"}
               for unit in current["units"]):
            result = agent_campaign.dispatch(campaign_path, root=Path(root))
            return {"state": result["state"], "action": "DISPATCH_AGENT_CAMPAIGN_UNIT",
                    "generation": step["generation"] + 1,
                    "source_hypothesis_id": step["source_hypothesis_id"],
                    "result": result}

    # Grounding Agent가 아닌 event-form 후보도 plan된 뒤에는 후보 단위로 동일하게 재개한다.
    for step in state["steps"]:
        campaign_path = Path(str(step.get("direct_event_campaign_plan")
                                 or step.get("entry_lifecycle_campaign_plan") or ""))
        if not campaign_path.is_file():
            continue
        current = direct_evidence_campaign.status(campaign_path)
        if any(unit["state"] in {"NOT_STARTED", "CAPACITY_WAITING",
                                  "ENTRY_REFINEMENT_IN_PROGRESS_ELSEWHERE"}
               for unit in current["units"]):
            result = direct_evidence_campaign.dispatch(campaign_path, root=Path(root))
            step["state"] = str(result["state"])
            _write_state(output, artifact, state)
            action = ("DISPATCH_DIRECT_EVENT_UNIT" if step.get("direct_event_campaign_plan")
                      else "DISPATCH_ENTRY_LIFECYCLE_UNIT")
            return {"state": result["state"], "action": action,
                    "generation": step["generation"] + 1,
                    "source_hypothesis_id": step["source_hypothesis_id"],
                    "result": result}

    # process가 Pre-validation 중 끊기면 pending step만 남는다. 이 경우를 이미 시도한
    # source로만 취급하면 _next_source가 계속 건너뛰어 loop가 영구 정지한다. terminal
    # manifest가 없는 고아 step만, 이전과 동일한 immutable source로 다시 연다.
    for index, pending in enumerate(state["steps"]):
        if str(pending.get("state") or "") != "PRE_VALIDATION_IN_PROGRESS":
            continue
        pre_output = Path(str(pending.get("pre_validation_output") or ""))
        if _pre_validation_owner_active(pre_output):
            return {"state": "WAITING_FOR_PRE_VALIDATION",
                    "action": "WAIT_PRE_VALIDATION",
                    "generation": int(pending["generation"]) + 1,
                    "source_hypothesis_id": pending["source_hypothesis_id"]}
        source = _source_from_step(pending)
        prior_context = _reusable_context(state, source)
        prior_validation_feedback = _reusable_validation_feedback(state, source)
        prior_library = _reusable_candidate_library(state, source)
        prior_hypotheses = [Path(str(path)) for path in
                             artifact["payload"].get("prior_hypothesis_paths") or []]
        loop_hypotheses = sorted(set([*prior_hypotheses, *_loop_hypothesis_paths(state)]))
        resumed = _launch_successor(
            artifact, output, source, root=Path(root), prior_context=prior_context,
            prior_validation_feedback=prior_validation_feedback,
            prior_library=prior_library, loop_hypothesis_paths=loop_hypotheses)
        state["steps"][index] = resumed
        _write_state(output, artifact, state)
        return {"state": resumed["state"], "action": "RESUME_PRE_VALIDATION",
                "generation": resumed["generation"] + 1,
                "source_hypothesis_id": resumed["source_hypothesis_id"], "step": resumed}

    next_source = _next_source(artifact, state)
    if next_source is None:
        exhausted = _exhausted_lens_sources(artifact, state)
        if exhausted:
            fallback = (_launch_direct_event_fallback(artifact, output, state, exhausted,
                                                      root=Path(root))
                        or _launch_entry_lifecycle_fallback(artifact, output, state, exhausted,
                                                            root=Path(root)))
            if fallback is not None:
                state["steps"].append(fallback["step"])
                _write_state(output, artifact, state)
                return {"state": fallback["state"], "action": fallback["action"],
                        "generation": fallback["step"]["generation"] + 1,
                        "source_hypothesis_id": fallback["step"]["source_hypothesis_id"],
                        "result": fallback.get("result")}
            return {"state": "LENS_EXHAUSTED", "action": "EXHAUSTED_SOURCE",
                    "reason": "저장된 Evidence lens를 모두 시도했지만 successor 가설이 없었다",
                    "exhausted_sources": exhausted}
        return {"state": "NO_DISCOVERY_LOOP_WORK", "action": "IDLE",
                "reason": "새 successor source 또는 재개 가능한 Agent campaign unit이 없다"}
    if not _has_refinement(next_source["research_output"]):
        state["skipped_sources"].append({**next_source, "reason": "NO_ENTRY_REFINEMENT"})
        _write_state(output, artifact, state)
        return {"state": "NO_ENTRY_REFINEMENT", "action": "SKIP_SOURCE",
                "source_hypothesis_id": next_source["hypothesis_id"]}

    # Pre-validation은 Agent 호출까지 포함할 수 있다. 완료 뒤에만 state를 쓰면 실제로
    # 작업 중이어도 loop가 빈 것처럼 보인다. 먼저 source와 출력 위치를 남긴다.
    pending = _pending_step(output, next_source)
    state["steps"].append(pending)
    _write_state(output, artifact, state)
    prior_context = _reusable_context(state, next_source)
    prior_validation_feedback = _reusable_validation_feedback(state, next_source)
    prior_library = _reusable_candidate_library(state, next_source)
    prior_hypotheses = [Path(str(path)) for path in
                         artifact["payload"].get("prior_hypothesis_paths") or []]
    loop_hypotheses = sorted(set([*prior_hypotheses, *_loop_hypothesis_paths(state)]))
    step = _launch_successor(artifact, output, next_source, root=Path(root),
                             prior_context=prior_context,
                             prior_validation_feedback=prior_validation_feedback,
                             prior_library=prior_library,
                             loop_hypothesis_paths=loop_hypotheses)
    state["steps"][-1] = step
    _write_state(output, artifact, state)
    return {"state": step["state"], "action": "LAUNCH_SUCCESSOR",
            "generation": step["generation"] + 1,
            "source_hypothesis_id": step["source_hypothesis_id"], "step": step}


def run(plan_path: Path, *, root: Path = TICK_ROOT,
        max_actions: int | None = None) -> dict[str, Any]:
    """멈춘 campaign을 이어서, 가능한 한 다음 세대까지 계속 dispatch한다.

    각 unit 자체는 동기 Research run이다. 이미 실행 중인 unit이나 capacity 대기는
    기다리지 않고 상태를 돌려준다. 그래서 외부 scheduler는 이 명령만 다시 부르면 된다.
    """
    if max_actions is not None and int(max_actions) < 1:
        raise ValueError("max_actions는 1 이상이어야 한다")
    actions: list[dict[str, Any]] = []
    while max_actions is None or len(actions) < int(max_actions):
        report = status(plan_path)
        if any(unit["state"] == "IN_PROGRESS"
               for step in report["steps"]
               for unit in ((step.get("campaign") or {}).get("units") or [])):
            return {"state": "WAITING_FOR_AGENT_CAMPAIGN", "actions": actions,
                    "status": report}
        if any(unit["state"] == "IN_PROGRESS"
               for step in report["steps"]
               for unit in ((step.get("direct_event_campaign") or {}).get("units") or [])):
            return {"state": "WAITING_FOR_DIRECT_EVENT_CAMPAIGN", "actions": actions,
                    "status": report}
        if any(unit["state"] == "IN_PROGRESS"
               for step in report["steps"]
               for unit in ((step.get("entry_lifecycle_campaign") or {}).get("units") or [])):
            return {"state": "WAITING_FOR_ENTRY_LIFECYCLE_CAMPAIGN", "actions": actions,
                    "status": report}
        result = dispatch(plan_path, root=Path(root))
        actions.append(result)
        if result["state"] == "CAPACITY_WAITING":
            return {"state": "CAPACITY_WAITING", "actions": actions,
                    "status": status(plan_path)}
        if result["state"] == "WAITING_FOR_PRE_VALIDATION":
            return {"state": "WAITING_FOR_PRE_VALIDATION", "actions": actions,
                    "status": status(plan_path)}
        if result["state"] == "LENS_EXHAUSTED":
            return {"state": "LENS_EXHAUSTED", "actions": actions,
                    "status": status(plan_path),
                    "reason": result["reason"],
                    "exhausted_sources": result["exhausted_sources"]}
        if result["state"] == "NO_DISCOVERY_LOOP_WORK":
            return {"state": "IDLE", "actions": actions, "status": status(plan_path)}
    return {"state": "ACTION_LIMIT", "actions": actions, "status": status(plan_path)}


def _launch_successor(plan: Mapping[str, Any], output: Path,
                      source: Mapping[str, Any], *, root: Path,
                      prior_context: Mapping[str, Any] | None = None,
                      prior_validation_feedback: Mapping[str, Any] | None = None,
                      prior_library: Path | None = None,
                      loop_hypothesis_paths: list[Path] | None = None) -> dict[str, Any]:
    payload = plan["payload"]
    generation = int(source["generation"])
    step_root = output / "steps" / _step_name(source)
    parent_output = Path(str(source["research_output"])).resolve()
    parent_run = read_artifact(parent_output / "research_run.json", kind="research_run")
    profile = _profile_from_run(parent_run)
    schedule = _schedule_from_payload(payload["schedule"])
    if prior_context is not None:
        context = dict(prior_context)
        write_artifact(step_root / "00_discovery_loss_context" / discovery_loss.ARTIFACT_NAME,
                       context)
    else:
        context = discovery_loss.build(
            parent_output, str(source["hypothesis_id"]), step_root / "00_discovery_loss_context")
    context_path = step_root / "00_discovery_loss_context" / discovery_loss.ARTIFACT_NAME
    if prior_validation_feedback is not None:
        validation_context = dict(prior_validation_feedback)
        write_artifact(step_root / "00_development_validation_feedback" /
                       validation_feedback.ARTIFACT_NAME, validation_context)
    else:
        validation_context = validation_feedback.build(
            parent_output, str(source["hypothesis_id"]),
            step_root / "00_development_validation_feedback")
    validation_context_path = (
        step_root / "00_development_validation_feedback" / validation_feedback.ARTIFACT_NAME
        if validation_context is not None else None)
    evidence_source = _evidence_source_from_run(parent_output, parent_run)
    pre_output = step_root / "01_pre_validation"
    plan_source = payload["source"]
    uses_plan_snapshot = (
        str(source["research_output"]) == str(plan_source["research_output"])
        and str(source["hypothesis_id"]) == str(plan_source["hypothesis_id"]))
    base_library = (Path(str(payload["candidate_library_snapshot_source"])).resolve()
                    if payload.get("candidate_library_snapshot_source") else None)
    if base_library is not None and not uses_plan_snapshot and prior_library is None:
        target_evidence = read_artifact(evidence_source, kind="evidence_package")
        rebound_root = step_root / "00_root_candidate_library"
        candidate_library.rebind_direct_evidence(
            rebound_root, base_library, direct_evidence_id=str(target_evidence["artifact_id"]),
            direct_profile=profile.as_input())
        planned_library = rebound_root / candidate_library.ARTIFACT_NAME
    else:
        planned_library = base_library if uses_plan_snapshot else None
    effective_library = prior_library or planned_library
    if effective_library is not None and loop_hypothesis_paths:
        overlay_root = step_root / "00_loop_candidate_library"
        candidate_library.extend_snapshot(overlay_root, effective_library, loop_hypothesis_paths)
        effective_library = overlay_root / candidate_library.ARTIFACT_NAME
    pre = run_pre_validation(PreValidationRequest(
        profile=profile, evidence_source=evidence_source,
        discovery_loss_context=context_path,
        development_validation_feedback=validation_context_path,
        research_lens_id=(str(source["research_lens_id"])
                          if source.get("research_lens_id") else None),
        candidate_library_source=effective_library), pre_output)
    step: dict[str, Any] = {
        "generation": generation,
        "root_rank": int(source.get("root_rank", 0)),
        "source_research_output": str(parent_output),
        "source_hypothesis_id": str(source["hypothesis_id"]),
        "research_lens_id": source.get("research_lens_id"),
        "state": str(pre["state"]),
        "discovery_loss_context": context["artifact_id"],
        "development_validation_feedback": (
            validation_context["artifact_id"] if validation_context is not None else None),
        "candidate_library_source": (str(effective_library) if effective_library is not None else None),
        "pre_validation_output": str(pre_output),
        "pre_validation_run": pre["run"]["artifact_id"],
        "agent_campaign_plan": None,
        "next_schedule": schedule.as_input(),
    }
    if pre["state"] != "READY_FOR_SPECIFICATION":
        return step
    campaign_output = step_root / "02_agent_campaign"
    campaign = agent_campaign.plan(agent_campaign.AgentCampaignPlanRequest(
        profile=profile, schedule=schedule, pre_validation_source=pre_output,
        candidate_output_root=step_root / "03_research_units",
        final_backtest_for_all_validation=True,
        validation_refinement_successor=True), campaign_output)
    campaign_path = campaign_output / agent_campaign.PLAN_ARTIFACT_NAME
    step["agent_campaign_plan"] = str(campaign_path)
    step["agent_campaign_artifact"] = campaign["artifact_id"]
    result = agent_campaign.dispatch(campaign_path, root=root)
    step["state"] = str(result["state"])
    step["first_dispatch"] = {
        "state": str(result["state"]), "hypothesis_id": result.get("hypothesis_id"),
        "output": result.get("output"),
    }
    return step


def _launch_direct_event_fallback(plan: Mapping[str, Any], output: Path,
                                  state: Mapping[str, Any],
                                  exhausted_sources: list[Mapping[str, Any]], *,
                                  root: Path) -> dict[str, Any] | None:
    """Agent lens가 모두 비어 있을 때 저장된 Late Trigger의 사건형 entry를 연다.

    이것은 Agent가 금융 이야기를 새로 만든 대체 경로가 아니다. 동일 Evidence의
    `LATE_TRIGGER` 상태를 지속 level 대신 최초 prior-100-tick percentile crossover로
    표현해 보는, 이미 고정된 Direct Evidence 후보군이다.
    """
    if not bool((plan.get("payload") or {}).get("direct_event_fallback", False)):
        return None
    attempted = {
        (str(step.get("source_research_output") or ""),
         str(step.get("source_hypothesis_id") or ""))
        for step in state.get("steps") or []
        if str(step.get("kind") or "") == "DIRECT_EVENT_FALLBACK"
    }
    for item in sorted(exhausted_sources, key=lambda value: (
            int(value.get("root_rank", 0)), str(value.get("source_research_output") or ""),
            str(value.get("source_hypothesis_id") or ""))):
        parent_output = str(item.get("source_research_output") or "")
        hypothesis_id = str(item.get("source_hypothesis_id") or "")
        if not parent_output or not hypothesis_id or (parent_output, hypothesis_id) in attempted:
            continue
        parent = Path(parent_output).resolve()
        parent_run = read_artifact(parent / "research_run.json", kind="research_run")
        profile = _profile_from_run(parent_run)
        schedule = _schedule_from_payload((plan.get("payload") or {}).get("schedule") or {})
        source = {
            "generation": int(item.get("generation", 0)),
            "root_rank": int(item.get("root_rank", _root_rank(
                plan, state, parent_output, hypothesis_id))),
            "research_output": str(parent),
            "hypothesis_id": hypothesis_id,
        }
        step_root = output / "direct_event_fallback" / _step_name(source)
        request = DirectEvidenceResearchRequest(
            profile=profile, schedule=schedule,
            evidence_source=_evidence_source_from_run(parent, parent_run),
            late_trigger_cross_only=True,
            final_backtest_for_all_validation=True,
            validation_refinement_successor=True)
        campaign = direct_evidence_campaign.plan(
            direct_evidence_campaign.CampaignPlanRequest(
                request=request, candidate_output_root=step_root / "02_research_units"),
            step_root / "01_direct_event_campaign")
        campaign_path = step_root / "01_direct_event_campaign" / direct_evidence_campaign.PLAN_ARTIFACT_NAME
        if int(campaign["payload"].get("candidate_count") or 0) == 0:
            result: Mapping[str, Any] = {
                "state": "NO_DIRECT_EVENT_CANDIDATE",
                "reason": "저장 Evidence에 아직 열리지 않은 Late Trigger crossover 후보가 없다",
            }
        else:
            result = direct_evidence_campaign.dispatch(campaign_path, root=Path(root))
        step = {
            "kind": "DIRECT_EVENT_FALLBACK",
            "generation": int(source["generation"]),
            "root_rank": int(source["root_rank"]),
            "source_research_output": str(parent),
            "source_hypothesis_id": hypothesis_id,
            "research_lens_id": None,
            "state": str(result["state"]),
            "direct_event_campaign_plan": str(campaign_path),
            "direct_event_campaign_artifact": str(campaign["artifact_id"]),
            "direct_event_source_type": "DIRECT_EVIDENCE_LATE_TRIGGER_CROSS",
            "next_schedule": schedule.as_input(),
        }
        return {"state": str(result["state"]), "action": "LAUNCH_DIRECT_EVENT_FALLBACK",
                "step": step, "result": dict(result)}
    return None


def _launch_entry_lifecycle_fallback(plan: Mapping[str, Any], output: Path,
                                     state: Mapping[str, Any],
                                     exhausted_sources: list[Mapping[str, Any]], *,
                                     root: Path) -> dict[str, Any] | None:
    """지속 단일 상태의 order-lifetime entry를 같은 immutable Evidence에서 연다."""
    if not bool((plan.get("payload") or {}).get("entry_lifecycle_fallback", False)):
        return None
    attempted = {
        (str(step.get("source_research_output") or ""),
         str(step.get("source_hypothesis_id") or ""))
        for step in state.get("steps") or []
        if str(step.get("kind") or "") == "ENTRY_LIFECYCLE_FALLBACK"
    }
    for item in sorted(exhausted_sources, key=lambda value: (
            int(value.get("root_rank", 0)), str(value.get("source_research_output") or ""),
            str(value.get("source_hypothesis_id") or ""))):
        parent_output = str(item.get("source_research_output") or "")
        hypothesis_id = str(item.get("source_hypothesis_id") or "")
        if not parent_output or not hypothesis_id or (parent_output, hypothesis_id) in attempted:
            continue
        parent = Path(parent_output).resolve()
        parent_run = read_artifact(parent / "research_run.json", kind="research_run")
        profile = _profile_from_run(parent_run)
        schedule = _schedule_from_payload((plan.get("payload") or {}).get("schedule") or {})
        source = {
            "generation": int(item.get("generation", 0)),
            "root_rank": int(item.get("root_rank", _root_rank(
                plan, state, parent_output, hypothesis_id))),
            "research_output": str(parent), "hypothesis_id": hypothesis_id,
        }
        step_root = output / "entry_lifecycle_fallback" / _step_name(source)
        request = DirectEvidenceResearchRequest(
            profile=profile, schedule=schedule,
            evidence_source=_evidence_source_from_run(parent, parent_run),
            entry_lifecycle_continuity_only=True,
            final_backtest_for_all_validation=True,
            validation_refinement_successor=True)
        campaign = direct_evidence_campaign.plan(
            direct_evidence_campaign.CampaignPlanRequest(
                request=request, candidate_output_root=step_root / "02_research_units"),
            step_root / "01_entry_lifecycle_campaign")
        campaign_path = (step_root / "01_entry_lifecycle_campaign"
                         / direct_evidence_campaign.PLAN_ARTIFACT_NAME)
        if int(campaign["payload"].get("candidate_count") or 0) == 0:
            result: Mapping[str, Any] = {
                "state": "NO_ENTRY_LIFECYCLE_CANDIDATE",
                "reason": "저장 Evidence에 아직 열리지 않은 지속 단일 execution-state 후보가 없다",
            }
        else:
            result = direct_evidence_campaign.dispatch(campaign_path, root=Path(root))
        step = {
            "kind": "ENTRY_LIFECYCLE_FALLBACK",
            "generation": int(source["generation"]),
            "root_rank": int(source["root_rank"]),
            "source_research_output": str(parent),
            "source_hypothesis_id": hypothesis_id,
            "research_lens_id": None,
            "state": str(result["state"]),
            "entry_lifecycle_campaign_plan": str(campaign_path),
            "entry_lifecycle_campaign_artifact": str(campaign["artifact_id"]),
            "entry_lifecycle_source_type": "DIRECT_DISCOVERY_ANCHOR_EXECUTION_STATE_EVIDENCE",
            "next_schedule": schedule.as_input(),
        }
        return {"state": str(result["state"]), "action": "LAUNCH_ENTRY_LIFECYCLE_FALLBACK",
                "step": step, "result": dict(result)}
    return None


def _pending_step(output: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    step_root = output / "steps" / _step_name(source)
    return {
        "generation": int(source["generation"]),
        "root_rank": int(source.get("root_rank", 0)),
        "source_research_output": str(source["research_output"]),
        "source_hypothesis_id": str(source["hypothesis_id"]),
        "research_lens_id": source.get("research_lens_id"),
        "state": "PRE_VALIDATION_IN_PROGRESS",
        "discovery_loss_context": None,
        "development_validation_feedback": None,
        "candidate_library_source": None,
        "pre_validation_output": str(step_root / "01_pre_validation"),
        "pre_validation_run": None,
        "agent_campaign_plan": None,
        "next_schedule": None,
    }


def _source_from_step(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generation": int(step["generation"]),
        "root_rank": int(step.get("root_rank", 0)),
        "research_output": str(step["source_research_output"]),
        "hypothesis_id": str(step["source_hypothesis_id"]),
        "research_lens_id": step.get("research_lens_id"),
    }


def _pre_validation_owner_active(output: Path) -> bool:
    progress_path = Path(output) / "pre_validation_progress.json"
    if not progress_path.is_file():
        return False
    try:
        progress = read_json(progress_path)
        owner_pid = int(progress.get("owner_pid") or 0)
    except (OSError, ValueError, TypeError):
        return False
    if str(progress.get("state") or "") != "IN_PROGRESS" or owner_pid < 1:
        return False
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reusable_context(state: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any] | None:
    """같은 parent ledger를 다시 읽지 않고 이미 만든 immutable context를 재사용한다."""
    for step in state.get("steps") or []:
        if (int(step.get("generation", -1)) != int(source["generation"])
                or str(step.get("source_research_output") or "") != str(source["research_output"])
                or str(step.get("source_hypothesis_id") or "") != str(source["hypothesis_id"])):
            continue
        pre_output = Path(str(step.get("pre_validation_output") or ""))
        path = pre_output.parent / "00_discovery_loss_context" / discovery_loss.ARTIFACT_NAME
        if not path.is_file():
            continue
        try:
            context = read_artifact(path, kind="discovery_loss_context")
        except (OSError, ValueError):
            continue
        if str(context["payload"].get("source_hypothesis_id") or "") == str(source["hypothesis_id"]):
            return context
    return None


def _reusable_validation_feedback(state: Mapping[str, Any],
                                  source: Mapping[str, Any]) -> dict[str, Any] | None:
    """같은 Validation 탈락 source의 immutable 개발 피드백을 lens 사이에 재사용한다."""
    for step in state.get("steps") or []:
        if (int(step.get("generation", -1)) != int(source["generation"])
                or str(step.get("source_research_output") or "") != str(source["research_output"])
                or str(step.get("source_hypothesis_id") or "") != str(source["hypothesis_id"])):
            continue
        pre_output = Path(str(step.get("pre_validation_output") or ""))
        path = (pre_output.parent / "00_development_validation_feedback" /
                validation_feedback.ARTIFACT_NAME)
        if not path.is_file():
            continue
        try:
            context = read_artifact(path, kind="development_validation_feedback")
        except (OSError, ValueError):
            continue
        if str(context["payload"].get("source_hypothesis_id") or "") == str(source["hypothesis_id"]):
            return context
    return None


def _reusable_candidate_library(state: Mapping[str, Any], source: Mapping[str, Any]) -> Path | None:
    """같은 loss-source의 첫 Pre-validation candidate history를 immutable snapshot으로 쓴다."""
    for step in state.get("steps") or []:
        if (int(step.get("generation", -1)) != int(source["generation"])
                or str(step.get("source_research_output") or "") != str(source["research_output"])
                or str(step.get("source_hypothesis_id") or "") != str(source["hypothesis_id"])):
            continue
        pre_output = Path(str(step.get("pre_validation_output") or ""))
        path = pre_output / "00_candidate_library" / candidate_library.ARTIFACT_NAME
        if not path.is_file():
            continue
        try:
            read_artifact(path, kind="candidate_library")
        except (OSError, ValueError):
            continue
        return path
    return None


def _loop_hypothesis_paths(state: Mapping[str, Any]) -> list[Path]:
    """현재 plan에서 이미 끝난 Agent 초안만 다음 novelty snapshot에 넣는다."""
    paths: list[Path] = []
    for step in state.get("steps") or []:
        pre = Path(str(step.get("pre_validation_output") or ""))
        path = pre / "02_hypothesis" / "hypothesis_artifact.json"
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def _prior_loop_candidate_library(source_research: Path, hypothesis_id: str,
                                  profile: ProfileSelection, *, exclude_output: Path) -> Path | None:
    """같은 source의 과거 loop snapshot 중 plan 이전 마지막 것을 재사용한다.

    새 loop는 직전 loop가 이미 고정한 historical candidate baseline을 그대로 쓴다.
    따라서 동시 실행이 늘어도 새 plan의 첫 Agent 호출이 전역 history 재구축을 하지 않고,
    두 loop의 novelty 비교 범위도 artifact로 명시된다.
    """
    source = Path(source_research).resolve()
    excluded = Path(exclude_output).resolve()
    expected_profile = clean(profile.as_input())
    parent_run = read_artifact(source / "research_run.json", kind="research_run")
    evidence_path = _evidence_source_from_run(source, parent_run)
    evidence_id = read_artifact(evidence_path, kind="evidence_package")["artifact_id"]
    candidates: list[Path] = []
    runs_root = candidate_library.SANDBOX_ROOT / "runs"
    if not runs_root.is_dir():
        return None
    for plan_path in sorted(runs_root.glob(f"*/{PLAN_ARTIFACT_NAME}"),
                            key=lambda path: path.stat().st_mtime_ns):
        if plan_path.parent.resolve() == excluded:
            continue
        try:
            prior_plan = _read_plan(plan_path)
        except (OSError, ValueError):
            continue
        prior_source = (prior_plan.get("payload") or {}).get("source") or {}
        if (str(prior_source.get("research_output") or "") != str(source)
                or str(prior_source.get("hypothesis_id") or "") != str(hypothesis_id)):
            continue
        state_path = plan_path.parent / STATE_FILE
        raw = read_json(state_path) if state_path.is_file() else {}
        for step in raw.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            pre = Path(str(step.get("pre_validation_output") or ""))
            path = pre / "00_candidate_library" / candidate_library.ARTIFACT_NAME
            if not path.is_file():
                continue
            try:
                library = read_artifact(path, kind="candidate_library")
            except (OSError, ValueError):
                continue
            saved = library.get("payload") or {}
            if (saved.get("direct_profile") == expected_profile
                    and str(saved.get("direct_evidence_id") or "") == str(evidence_id)):
                candidates.append(path.resolve())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _prior_loop_attempts(source_research: Path, hypothesis_id: str,
                         profile: ProfileSelection, schedule: ResearchSchedule, *,
                         exclude_output: Path,
                         loss_context_identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    """같은 source에서 이미 끝난 duplicate/no-hypothesis lens만 새 plan에 물려준다."""
    source = Path(source_research).resolve()
    excluded = Path(exclude_output).resolve()
    attempts: dict[str, dict[str, Any]] = {}
    runs_root = candidate_library.SANDBOX_ROOT / "runs"
    if not runs_root.is_dir():
        return []
    # INVALID_OUTPUT은 Agent/계약 수리 뒤 재시도할 가치가 있으므로, 더 오래된
    # duplicate/no-hypothesis 결과가 같은 lens를 영구 제외하지 못하게 한다.
    terminal = {"DUPLICATE_HYPOTHESIS", "NO_HYPOTHESIS"}
    for plan_path in sorted(runs_root.glob(f"*/{PLAN_ARTIFACT_NAME}"),
                            key=lambda path: path.stat().st_mtime_ns):
        if plan_path.parent.resolve() == excluded:
            continue
        try:
            prior_plan = _read_plan(plan_path)
        except (OSError, ValueError):
            continue
        if not _matches_loop_context(prior_plan, profile, schedule):
            continue
        prior_identities = (prior_plan.get("payload") or {}).get("source_context_identities") or {}
        prior_identity = prior_identities.get(_source_context_key(source, hypothesis_id))
        # v1 plan은 개선 전 첫 round를 썼다. 현재 계약/guard가 바뀐 v2 질문에
        # 그 lens 실패를 물려주면, 새 loop가 질문 자체를 열어 보지 못한다.
        if clean(prior_identity or {}) != clean(loss_context_identity):
            continue
        state_path = plan_path.parent / STATE_FILE
        raw = read_json(state_path) if state_path.is_file() else {}
        for step in raw.get("steps") or []:
            if (not isinstance(step, Mapping)
                    or str(step.get("source_research_output") or "") != str(source)
                    or str(step.get("source_hypothesis_id") or "") != str(hypothesis_id)):
                continue
            state = str(step.get("state") or "")
            item = {
                "generation": int(step.get("generation", 0)),
                "research_output": str(source),
                "hypothesis_id": str(hypothesis_id),
                "research_lens_id": step.get("research_lens_id"),
                "state": state,
                "source_loop_plan": str(plan_path),
                "pre_validation_output": str(step.get("pre_validation_output") or ""),
            }
            key = _source_key(item)
            if state == "INVALID_OUTPUT":
                attempts.pop(key, None)
            elif state in terminal:
                attempts[key] = item
    return [attempts[key] for key in sorted(attempts)]


def _source_context_key(source_research: str | Path, hypothesis_id: str) -> str:
    return f"{Path(source_research).resolve()}|{str(hypothesis_id)}"


def _source_context_identity(source_research: Path, hypothesis_id: str) -> dict[str, Any]:
    """Discovery ledger와 Validation 실패가 모두 같은 질문인지 판단할 지문."""
    return {
        "discovery_loss": discovery_loss.context_identity(source_research, hypothesis_id),
        "development_validation_feedback": validation_feedback.context_identity(
            source_research, hypothesis_id),
    }


def _prior_loop_hypothesis_paths(source_research: Path, hypothesis_id: str,
                                 profile: ProfileSelection, schedule: ResearchSchedule, *,
                                 exclude_output: Path) -> list[Path]:
    """같은 source의 과거 loop가 실제 생성한 hypothesis artifact를 새 snapshot에 잇는다."""
    source = Path(source_research).resolve()
    excluded = Path(exclude_output).resolve()
    paths: list[Path] = []
    runs_root = candidate_library.SANDBOX_ROOT / "runs"
    if not runs_root.is_dir():
        return []
    for plan_path in sorted(runs_root.glob(f"*/{PLAN_ARTIFACT_NAME}")):
        if plan_path.parent.resolve() == excluded:
            continue
        try:
            prior_plan = _read_plan(plan_path)
        except (OSError, ValueError):
            continue
        if not _matches_loop_context(prior_plan, profile, schedule):
            continue
        state_path = plan_path.parent / STATE_FILE
        raw = read_json(state_path) if state_path.is_file() else {}
        for step in raw.get("steps") or []:
            if (not isinstance(step, Mapping)
                    or str(step.get("source_research_output") or "") != str(source)
                    or str(step.get("source_hypothesis_id") or "") != str(hypothesis_id)):
                continue
            path = Path(str(step.get("pre_validation_output") or "")) / "02_hypothesis" / "hypothesis_artifact.json"
            if path.is_file():
                paths.append(path.resolve())
    return sorted(set(paths))


def _matches_loop_context(plan: Mapping[str, Any], profile: ProfileSelection,
                          schedule: ResearchSchedule) -> bool:
    """다른 Frontier plan의 step도 같은 입력 경계일 때만 이어받는다."""
    payload = plan.get("payload") or {}
    return (clean(payload.get("profile") or {}) == clean(profile.as_input())
            and clean(payload.get("schedule") or {}) == clean(schedule.as_input()))


def _pre_validation_status(output: Path) -> dict[str, Any]:
    progress_path = Path(output) / "pre_validation_progress.json"
    run_path = Path(output) / "pre_validation_run.json"
    progress = read_json(progress_path) if progress_path.is_file() else {}
    if run_path.is_file():
        run = read_artifact(run_path, kind="pre_validation_run")
        return {"state": str(run["payload"].get("state") or ""), "stage": "TERMINAL"}
    return {"state": str(progress.get("state") or "IN_PROGRESS"),
            "stage": str(progress.get("stage") or "")}


def _next_source(plan: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = plan["payload"]
    stored_seeds = list(payload.get("seed_sources") or [])
    if not stored_seeds:  # schema v1 호환: 옛 plan은 단일 source만 가졌다.
        stored_seeds = [payload["source"]]
    sources = [{
        "generation": int(item.get("generation", 0)),
        "research_output": str(item["research_output"]),
        "hypothesis_id": str(item["hypothesis_id"]),
        "research_lens_id": None,
        "root_rank": int(item.get("root_rank", index)),
    } for index, item in enumerate(stored_seeds)]
    for step in state["steps"]:
        campaign_path = Path(str(step.get("agent_campaign_plan") or ""))
        if not campaign_path.is_file():
            continue
        for unit in agent_campaign.status(campaign_path)["units"]:
            candidate = {
                "generation": int(step["generation"]) + 1,
                "research_output": str(unit["output"]),
                "hypothesis_id": str(unit["hypothesis_id"]),
                "research_lens_id": None,
                "root_rank": int(step.get("root_rank", 0)),
            }
            if _terminal_research(candidate["research_output"]):
                sources.append(candidate)
    for step in state["steps"]:
        campaign_path = Path(str(step.get("direct_event_campaign_plan")
                                 or step.get("entry_lifecycle_campaign_plan") or ""))
        if not campaign_path.is_file():
            continue
        for unit in direct_evidence_campaign.status(campaign_path)["units"]:
            candidate = {
                "generation": int(step["generation"]) + 1,
                "research_output": str(unit["output"]),
                "hypothesis_id": str(unit["candidate_id"]),
                "research_lens_id": None,
                "root_rank": int(step.get("root_rank", 0)),
            }
            if _terminal_research(candidate["research_output"]):
                sources.append(candidate)
    # Duplicate는 진짜로 같은 가설을 막은 정상적인 연구 결과다. 그것을 terminal로만
    # 두면 loss ledger -> Agent loop가 한 번의 똑같은 초안에서 끝난다. 같은 Evidence의
    # 사전 계산된 관계 lens만 다음 입력으로 열어 다른 mechanism relation을 묻게 한다.
    # PnL·Search·Validation을 보고 재시도 순서를 고르지 않는다.
    sources.extend(_lens_retries(plan, state))
    known = {_source_key(item) for item in state["steps"]}
    known.update(_source_key(item) for item in payload.get("prior_attempts") or [])
    known.update(_source_key(item) for item in state["skipped_sources"])
    allowed = int(payload["max_generations"])
    candidates = [item for item in sources
                  if item["generation"] < allowed and _source_key(item) not in known]
    return min(candidates, key=lambda item: (int(item.get("root_rank", 0)),
                                             int(item["generation"]),
                                             str(item["research_output"]),
                                             str(item["hypothesis_id"]),
                                             str(item.get("research_lens_id") or "")), default=None)


def _lens_retries(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """중복/가설부재 source에 아직 열지 않은 Evidence lens 하나씩만 붙인다."""
    limit = int(plan["payload"].get("max_lens_attempts", 4))
    if limit == 0:
        return []
    retryable = {"DUPLICATE_HYPOTHESIS", "NO_HYPOTHESIS", "INVALID_OUTPUT"}
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for step in [*(state["steps"] or []), *(plan["payload"].get("prior_attempts") or [])]:
        if str(step.get("state") or "") not in retryable:
            continue
        research_output = str(step.get("source_research_output") or step.get("research_output") or "")
        hypothesis_id = str(step.get("source_hypothesis_id") or step.get("hypothesis_id") or "")
        if not research_output or not hypothesis_id:
            continue
        key = (int(step.get("generation", 0)), research_output, hypothesis_id)
        grouped.setdefault(key, []).append(step)
    retries: list[dict[str, Any]] = []
    for (generation, research_output, hypothesis_id), attempts in grouped.items():
        # 같은 plan 안의 INVALID_OUTPUT은 다음 lens로 넘기지만, 새 plan은 Agent/계약
        # 수리 뒤 그 lens를 한 번 다시 열 수 있다. 따라서 이전 invalid는 미시도로 둔다.
        attempted = {str(item["research_lens_id"]) for item in attempts
                     if item.get("research_lens_id")
                     and str(item.get("state") or "") != "INVALID_OUTPUT"}
        # 이전 plan의 terminal lens는 중복을 막는 history다. 그것을 새 plan의 retry
        # 예산까지 소진한 것으로 세면, 새 lens schedule·Agent 방법을 넣어도 과거 여덟
        # relation 때문에 temporal/single Evidence를 영원히 못 연다. 이번 plan이 실제로
        # 연 lens만 budget에 세고, history는 `attempted`에만 남긴다.
        current_attempted = {
            str(item["research_lens_id"]) for item in state.get("steps") or []
            if (str(item.get("source_research_output") or item.get("research_output") or "")
                == research_output)
            and str(item.get("source_hypothesis_id") or item.get("hypothesis_id") or "") == hypothesis_id
            and int(item.get("generation", 0)) == generation
            and item.get("research_lens_id")
            and str(item.get("state") or "") != "INVALID_OUTPUT"
        }
        if len(current_attempted) >= limit:
            continue
        latest = max(attempts, key=lambda item: str(item.get("pre_validation_output") or ""))
        lenses = _available_lenses(Path(str(latest.get("pre_validation_output") or "")))
        next_lens = next((lens for lens in lenses if lens not in attempted), None)
        if next_lens is not None:
            retries.append({"generation": generation, "research_output": research_output,
                            "hypothesis_id": hypothesis_id,
                            "research_lens_id": next_lens,
                            "root_rank": _root_rank(
                                plan, state, research_output, hypothesis_id)})
    return retries


def _exhausted_lens_sources(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """가설 부재를 일반 IDLE과 구분해, 다음 연구 설계가 필요함을 남긴다."""
    limit = int(plan["payload"].get("max_lens_attempts", 4))
    if limit == 0:
        return []
    retryable = {"DUPLICATE_HYPOTHESIS", "NO_HYPOTHESIS", "INVALID_OUTPUT"}
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for step in [*(state["steps"] or []), *(plan["payload"].get("prior_attempts") or [])]:
        research_output = str(step.get("source_research_output") or step.get("research_output") or "")
        hypothesis_id = str(step.get("source_hypothesis_id") or step.get("hypothesis_id") or "")
        if not research_output or not hypothesis_id:
            continue
        key = (int(step.get("generation", 0)), research_output, hypothesis_id)
        grouped.setdefault(key, []).append(step)
    exhausted: list[dict[str, Any]] = []
    for (generation, research_output, hypothesis_id), attempts in grouped.items():
        if not attempts or not all(str(item.get("state") or "") in retryable
                                   for item in attempts):
            continue
        attempted = [str(item["research_lens_id"]) for item in attempts
                     if item.get("research_lens_id")
                     and str(item.get("state") or "") != "INVALID_OUTPUT"]
        current_attempted = [str(item["research_lens_id"]) for item in state.get("steps") or []
                             if (str(item.get("source_research_output") or item.get("research_output") or "")
                                 == research_output)
                             and str(item.get("source_hypothesis_id") or item.get("hypothesis_id") or "")
                             == hypothesis_id
                             and int(item.get("generation", 0)) == generation
                             and item.get("research_lens_id")
                             and str(item.get("state") or "") != "INVALID_OUTPUT"]
        latest = max(attempts, key=lambda item: str(item.get("pre_validation_output") or ""))
        available = _available_lenses(Path(str(latest.get("pre_validation_output") or "")))
        remaining = [lens for lens in available if lens not in set(attempted)]
        if len(current_attempted) < limit and remaining:
            continue
        exhausted.append({
            "generation": generation,
            "source_research_output": research_output,
            "source_hypothesis_id": hypothesis_id,
            "attempted_lenses": attempted,
            "current_plan_attempted_lenses": current_attempted,
            "available_lens_count": len(available),
            "configured_lens_limit": limit,
        })
    return exhausted


def _available_lenses(pre_validation_output: Path) -> list[str]:
    path = Path(pre_validation_output) / "00_mechanism_lenses.json"
    if not path.is_file():
        return []
    try:
        catalog = read_json(path)
    except (OSError, ValueError):
        return []
    lenses = [item for item in catalog.get("lenses") or [] if isinstance(item, Mapping)]
    # 한 relation 종류를 전부 먼저 열면 작은 retry 한도 안에서 나머지 Evidence가
    # 영원히 닿지 않는다. 예컨대 CONTEXT_TRIGGER가 여덟 개면 temporal·single lens는
    # `max_lens_attempts=8`에서 한 번도 Agent 입력이 되지 못한다. 관계 구조를 먼저
    # 보되, 이후에는 lens kind별 첫 항목을 라운드로 섞는다. 이 순서는 저장 catalog만
    # 읽으며 Search·Validation·Final 또는 PnL을 보지 않는다.
    kinds = (
        "CONTEXT_TRIGGER", "CONTEXT_INDEPENDENT", "COUPLED",
        "EXECUTION_STATE", "RAW_PATH", "TEMPORAL", "SINGLE",
    )

    def kind(item: Mapping[str, Any]) -> str:
        lens_id = str(item.get("lens_id") or "")
        if lens_id.startswith("single-execution-state:"):
            return "EXECUTION_STATE"
        if lens_id.startswith("single-raw-path:"):
            return "RAW_PATH"
        if lens_id.startswith("single-temporal:"):
            return "TEMPORAL"
        relation = str(item.get("relation") or "")
        if relation in {"CONTEXT_TRIGGER", "CONTEXT_INDEPENDENT", "COUPLED"}:
            return relation
        return "SINGLE"

    buckets = {name: [] for name in kinds}
    for item in lenses:
        if not item.get("lens_id"):
            continue
        buckets[kind(item)].append(str(item["lens_id"]))
    for values in buckets.values():
        values.sort()
    ordered: list[str] = []
    while any(buckets.values()):
        for name in kinds:
            if buckets[name]:
                ordered.append(buckets[name].pop(0))
    return ordered


def _terminal_research(path: str) -> bool:
    return (Path(path) / "research_run.json").is_file()


def _matching_seed_sources(source_research: Path, hypothesis_id: str,
                           profile: ProfileSelection, schedule: ResearchSchedule, *,
                           limit: int) -> list[dict[str, Any]]:
    """동일한 입력 경계의 완료 Discovery source만 PnL 없이 seed queue로 고정한다."""
    wanted_profile = clean(profile.as_input())
    wanted_schedule = clean(schedule.as_input())
    initial = {"generation": 0, "research_output": str(Path(source_research).resolve()),
               "hypothesis_id": str(hypothesis_id)}
    candidates = [initial]
    seen = {(initial["research_output"], initial["hypothesis_id"])}
    if int(limit) == 1:
        return candidates
    runs_root = candidate_library.SANDBOX_ROOT / "runs"
    if runs_root.is_dir():
        # Frontier root와 한 단계 아래 campaign unit까지만 후보로 본다. loop child는
        # 그 부모 loop가 자체 generation으로 이어야 하므로, 깊은 rglob는 중복 source와
        # 대형 ledger tree 순회를 동시에 만든다.
        run_paths: list[Path] = []
        for root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            direct = root / "research_run.json"
            if direct.is_file():
                run_paths.append(direct)
            run_paths.extend(sorted(root.glob("*/research_run.json")))
        for run_path in run_paths:
            research_output = run_path.parent.resolve()
            try:
                run = read_artifact(run_path, kind="research_run")
            except (OSError, ValueError):
                continue
            request = run["payload"].get("request") or {}
            if (clean(request.get("profile") or {}) != wanted_profile
                    or clean(request.get("schedule") or {}) != wanted_schedule):
                continue
            refinement_ref = str((run["payload"].get("artifacts") or {}).get(
                "entry_refinement_set") or "")
            refinement_path = research_output / "04_entry_refinement" / "entry_refinement_artifact.json"
            if not refinement_ref or not refinement_path.is_file():
                continue
            try:
                refinement = read_artifact(refinement_path, kind="entry_refinement_set")
            except (OSError, ValueError):
                continue
            if refinement["artifact_id"] != refinement_ref:
                continue
            for candidate_id in sorted((refinement["payload"].get("units") or {}).keys()):
                unit_path = research_output / "04_entry_refinement" / "units" / str(candidate_id) / "entry_refinement.json"
                if unit_path.is_file():
                    key = (str(research_output), str(candidate_id))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({"generation": 0,
                                       "research_output": str(research_output),
                                       "hypothesis_id": str(candidate_id)})
                    if len(candidates) >= int(limit):
                        return candidates
    return candidates


def _root_rank(plan: Mapping[str, Any], state: Mapping[str, Any],
               research_output: str, hypothesis_id: str) -> int:
    """Seed 순서를 유지해 한 source의 lens를 먼저 소진한다."""
    wanted = (str(research_output), str(hypothesis_id))
    seeds = plan["payload"].get("seed_sources") or []
    for index, item in enumerate(seeds):
        if (str(item.get("research_output") or ""),
                str(item.get("hypothesis_id") or "")) == wanted:
            return int(item.get("root_rank", index))
    for step in state.get("steps") or []:
        if (str(step.get("source_research_output") or ""),
                str(step.get("source_hypothesis_id") or "")) == wanted:
            return int(step.get("root_rank", 0))
    return len(seeds)


def _has_refinement(path: str) -> bool:
    run_path = Path(path) / "research_run.json"
    if not run_path.is_file():
        return False
    run = read_artifact(run_path, kind="research_run")
    return bool((run["payload"].get("artifacts") or {}).get("entry_refinement_set"))


def _profile_from_run(run: Mapping[str, Any]) -> ProfileSelection:
    profile = (run["payload"].get("request") or {}).get("profile") or {}
    return ProfileSelection(
        clusters=tuple(str(value) for value in profile.get("clusters") or []),
        symbols=tuple(str(value) for value in profile.get("symbols") or []),
        dates=tuple(str(value) for value in profile.get("dates") or []),
        store=Path(str(profile["store"])),
        discovery_profit_root=(Path(str(profile["discovery_profit_root"]))
                               if profile.get("discovery_profit_root") else None),
        profile_kind=str(profile.get("profile_kind") or "legacy_oracle"))


def _schedule_from_run(run: Mapping[str, Any]) -> ResearchSchedule:
    return _schedule_from_payload(((run["payload"].get("request") or {}).get("schedule") or {}))


def _schedule_from_payload(payload: Mapping[str, Any]) -> ResearchSchedule:
    return ResearchSchedule(
        validation_dates=tuple(str(value) for value in payload.get("validation_dates") or []),
        search_fit_dates=tuple(str(value) for value in payload.get("search_fit_dates") or []),
        search_confirm_dates=tuple(str(value) for value in payload.get("search_confirm_dates") or []),
        backtest_dates=tuple(str(value) for value in payload.get("backtest_dates") or []),
        terminal_oos_dates=tuple(str(value) for value in payload.get("terminal_oos_dates") or []),
        single_day_discovery_search=bool(payload.get("single_day_discovery_search")),
        allow_discovery_only_lock=bool(payload.get("allow_discovery_only_lock")))


def _evidence_source_from_run(output: Path, run: Mapping[str, Any]) -> Path:
    request = run["payload"].get("request") or {}
    # Direct Evidence run은 Pre-validation을 만들지 않는다. 그 경우는 request에 저장한
    # shared Evidence가 정본이다. 구형 Direct Evidence run은 request에 경로를 남기지
    # 않고 output/01_evidence에 직접 Artifact를 썼으므로 그 저장 위치도 먼저 쓴다.
    # Agent Research run만 Pre-validation의 effective path를 먼저 읽는다.
    raw = request.get("evidence_source")
    direct_path = Path(output) / "01_evidence" / "evidence_artifact.json"
    if not raw and direct_path.is_file():
        raw = direct_path
    if not raw:
        pre_root = Path(str(request.get("pre_validation_source") or output / "01_pre_validation"))
        pre = read_artifact(pre_root / "pre_validation_run.json", kind="pre_validation_run")
        pre_request = pre["payload"].get("request") or {}
        raw = (pre_request.get("effective_evidence_path") or pre_request.get("evidence_source")
               or pre_root / "01_evidence" / "evidence_artifact.json")
    path = Path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(f"Discovery loop이 재사용할 Evidence artifact가 없다: {path}")
    return path


def _source_key(source: Mapping[str, Any]) -> str:
    return "|".join((str(source["generation"]), str(source.get("research_output")
                                                  or source.get("source_research_output")),
                     str(source.get("hypothesis_id") or source.get("source_hypothesis_id")),
                     str(source.get("research_lens_id") or "")))


def _step_name(source: Mapping[str, Any]) -> str:
    return f"g{int(source['generation'])}_from_{sha256_json(_source_key(source))[:12]}"


def _read_plan(path: Path) -> dict[str, Any]:
    artifact = read_artifact(Path(path), kind="discovery_learning_loop_plan")
    if artifact["payload"].get("schema") != PLAN_SCHEMA:
        raise ValueError("Discovery learning loop plan 형식이 다르다")
    return artifact


def _read_state(output: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(output) / STATE_FILE
    raw = read_json(path) if path.is_file() else {}
    if raw.get("plan_artifact") != plan["artifact_id"]:
        raise ValueError("Discovery learning loop state와 plan artifact가 다르다")
    steps = raw.get("steps") or []
    skipped = raw.get("skipped_sources") or []
    if not isinstance(steps, list) or not isinstance(skipped, list):
        raise ValueError("Discovery learning loop state 모양이 다르다")
    return {"steps": [dict(item) for item in steps if isinstance(item, Mapping)],
            "skipped_sources": [dict(item) for item in skipped if isinstance(item, Mapping)]}


def _write_state(output: Path, plan: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    write_json(Path(output) / STATE_FILE, {
        "schema": "discovery_learning_loop_state.v1",
        "plan_artifact": plan["artifact_id"], "steps": list(state["steps"]),
        "skipped_sources": list(state["skipped_sources"]),
    })
