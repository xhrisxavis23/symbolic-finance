"""Evidence를 금융 가설로 바꾸는 Agent 역할."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .. import capability as C, hypothesis as H
from ..config import sha256_json
from . import loss_ledger, price_path
from .runtime import AgentInvocation, CodexAgentRunner, Runner, invoke


def _invocation_record(item: AgentInvocation | Mapping[str, Any]) -> dict[str, Any]:
    """새 호출 객체와 저장된 과거 호출 기록을 같은 artifact 형태로 보존한다."""
    if isinstance(item, AgentInvocation):
        return item.as_dict()
    return dict(item)


def _clip(value: Any, maximum: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= maximum else text[:maximum - 1] + "…"


def _family_set(record: Mapping[str, Any]) -> set[str]:
    return {str(item.get("family")) for item in record.get("evidence_basis") or []
            if isinstance(item, Mapping) and item.get("family")}


def _feature_set(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    """후보가 실제로 쓴 family·feature 쌍만 중복 방지 우선순위에 쓴다."""
    features: set[tuple[str, str]] = set()
    for item in record.get("evidence_basis") or []:
        if not isinstance(item, Mapping) or not item.get("family"):
            continue
        raw = item.get("representative_feature")
        feature = raw.get("name") if isinstance(raw, Mapping) else raw
        if feature:
            features.add((str(item["family"]), str(feature)))
    return features


def _candidate_fingerprints(records: Sequence[Mapping[str, Any]], *,
                            families: set[str] | None,
                            features: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    """긴 과거 원문 대신 Agent가 중복을 피하는 데 필요한 지문만 준다.

    전체 원문은 artifact와 최종 novelty audit에 그대로 남는다. 여기서는 CLI argument
    한도를 넘기지 않으면서 현재 lens와 겹치는 금융 설명을 피하게 하는 것이 목적이다.
    """
    selected = [record for record in records
                if families is None or _family_set(record) & families]
    # 현재 lens와 정확히 같은 Direct 관계는 이미 실행 가능한 entry logic이다. 이
    # fingerprint가 prompt의 10개 상한 뒤로 밀리면 Agent가 같은 관계를 새로 발견한
    # 것처럼 다시 쓸 수 있다. 다만 Direct 후보는 mechanism이 아니므로 이들을 먼저
    # 보여 줄 뿐, Agent mechanism의 독립성 판단 자체를 대신하지 않는다.
    exact_direct = [] if families is None else [
        record for record in selected
        if record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"
        and _family_set(record) == families]
    other_direct = [] if families is None else [
        record for record in selected
        if record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"
        and _family_set(record) != families]
    agent_records = [record for record in selected
                     if record.get("candidate_kind") != "DIRECT_EVIDENCE_CANDIDATE"]
    ordered = selected if families is None else [*exact_direct, *agent_records, *other_direct]
    if features:
        # 같은 family만 공유하는 수십 개의 옛 후보보다, 현재 lens가 요구한 실제
        # feature를 가장 많이 공유하는 후보를 앞에 둔다. 그래야 generation Agent가
        # 그 후보를 못 본 채 같은 가설을 다시 만들고 뒤 novelty audit에서만 막히는
        # 일을 줄인다. direct fingerprint는 아래 예약 목록으로는 여전히 모두 보존한다.
        ordered = sorted(
            ordered,
            key=lambda record: (
                -len(_feature_set(record) & features),
                -int(_feature_set(record) == features),
                -int(_family_set(record) == (families or set())),
                -int(record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"),
                str(record.get("title") or ""),
            ))
    # 같은 artifact의 재검증/재시도는 같은 설명을 반복한다. 첫 지문 하나만 남긴다.
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...], str]] = set()
    for record in ordered:
        key = (str(record.get("candidate_kind") or "AGENT_HYPOTHESIS"),
               _clip(record.get("title"), 120), _clip(record.get("mechanistic_claim"), 160),
               tuple(sorted(_family_set(record))),
               _clip(record.get("entry_logic_fingerprint"), 160))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    shown = unique[:10]
    # Agent 설명은 최대 10개만 싣지만, Direct Evidence의 entry fingerprint는 이미
    # 구현 가능한 진입 계약 자체다. 뒤쪽에 밀린 fingerprint를 Agent가 새 가설처럼
    # 다시 쓰면 audit 뒤에야 같은 실행 계약임을 알게 된다. 따라서 이 작은 예약
    # 목록은 모두 보낸다. mechanism 원문이 아니라 fingerprint만 담으므로 prompt
    # 크기를 키우지 않는다.
    reserved_direct_entry_logic_fingerprints = sorted({
        str(record.get("entry_logic_fingerprint"))
        for record in selected
        if (record.get("candidate_kind") == "DIRECT_EVIDENCE_CANDIDATE"
            and str(record.get("entry_logic_fingerprint") or ""))
    })
    return {
        "note": "프롬프트용 중복 방지 지문이다. 전체 과거 후보 비교는 별도 novelty audit가 수행한다.",
        "matching_records": len(selected), "unique_records": len(unique),
        "records_omitted": max(0, len(unique) - len(shown)),
        "reserved_direct_entry_logic_fingerprints": reserved_direct_entry_logic_fingerprints,
        "records": [{
            "candidate_kind": record.get("candidate_kind", "AGENT_HYPOTHESIS"),
            "clusters": list(record.get("clusters") or []),
            "hypothesis_structure": record.get("hypothesis_structure"),
            "evidence_families": sorted(_family_set(record)),
            "entry_logic_fingerprint": _clip(record.get("entry_logic_fingerprint"), 120),
            "title": _clip(record.get("title"), 80),
            "mechanistic_claim": _clip(record.get("mechanistic_claim"), 120),
            "source_of_profit": _clip(record.get("source_of_profit"), 90),
            "mechanism_test": _clip(((record.get("mechanism_tests") or [{}])[0] or {}).get("claim"), 90),
        } for record in shown],
    }


def _compact_generation_digest(digest: Mapping[str, Any], *,
                               research_lens: Mapping[str, Any] | None) -> dict[str, Any]:
    """Agent 호출 전 transport용 context만 줄인다. 원본 Evidence artifact는 건드리지 않는다."""
    compact = dict(digest)
    # Agent는 가격 경로 전체를 prompt에서 해석하지 않고 read-only MCP로 필요한 사례를
    # 직접 열어야 한다. 여기에는 존재·길이만 남겨 prompt가 커지는 것을 막는다.
    opened_paths = []
    for item in compact.get("opened_discovery_price_paths") or []:
        if not isinstance(item, Mapping):
            continue
        summary = {key: value for key, value in item.items() if key != "points"}
        summary["available_point_count"] = len(item.get("points") or [])
        opened_paths.append(summary)
    if "opened_discovery_price_paths" in compact:
        compact["opened_discovery_price_paths"] = opened_paths
    families = ({str(value) for value in research_lens.get("families") or []}
                if research_lens is not None else None)
    required_evidence_ids = ({str(value) for value in research_lens.get("required_evidence_ids") or []}
                             if research_lens is not None else set())
    if families:
        compact["evidence"] = [item for item in compact.get("evidence") or []
                               if (str(item.get("evidence_id")) in required_evidence_ids
                                   if required_evidence_ids
                                   else str(item.get("family")) in families)]
        if len(families) == 1:
            compact.pop("evidence_relations", None)
            compact.pop("joint_evidence_note", None)
        else:
            compact["evidence_relations"] = [
                item for item in compact.get("evidence_relations") or []
                if {str(item.get("family_a")), str(item.get("family_b"))} == families]
        compact["independent_evidence_families"] = sorted(
            families & {str(value) for value in compact.get("independent_evidence_families") or []})
    library = compact.get("historical_candidate_library")
    if isinstance(library, Mapping):
        feature_pairs = _feature_set({"evidence_basis": compact.get("evidence") or []})
        compact["historical_candidate_library"] = _candidate_fingerprints(
            [record for record in library.get("records") or [] if isinstance(record, Mapping)],
            families=families, features=feature_pairs or None)
    return compact


def _access_counts(access: Mapping[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in (access or {}).get("anchors") or []:
        cohort = str(item.get("cohort"))
        counts[cohort] = counts.get(cohort, 0) + 1
    return dict(sorted(counts.items()))


def _path_contract(payload: Mapping[str, Any], package: Mapping[str, Any],
                   research_lens: Mapping[str, Any] | None) -> str:
    """최종 Evidence 또는 단일 execution/raw-path lens가 요구한 경로 계약을 고른다."""
    lens_id = str((research_lens or {}).get("lens_id") or "")
    if H.uses_raw_path_evidence(payload, package) or lens_id.startswith("single-raw-path:"):
        return "RAW_PATH"
    if H.uses_execution_state_evidence(payload, package) or lens_id.startswith(
            "single-execution-state:"):
        return "EXECUTION_STATE"
    return "PROFILE"


def _execution_state_path_contract(payload: Mapping[str, Any], package: Mapping[str, Any],
                                   research_lens: Mapping[str, Any] | None) -> bool:
    """과거 호출자가 쓰던 execution-state 여부 호환 helper."""
    return _path_contract(payload, package, research_lens) == "EXECUTION_STATE"


def _price_path_tool_usage(invocation: AgentInvocation, access: Mapping[str, Any] | None,
                           *, contract: str = "PROFILE",
                           required_raw_labels: set[str] | None = None,
                           execution_state: bool | None = None) -> dict[str, Any]:
    """실제 MCP 경로 조회만 가설의 경로 관측으로 인정한다."""
    if execution_state:
        contract = "EXECUTION_STATE"
    available = {str(item.get("cohort")) for item in (access or {}).get("anchors") or []}
    listed = any(
        call.get("server") == "price_path" and call.get("tool") == "list_price_paths"
        for call in invocation.tool_calls)
    reads: list[dict[str, Any]] = []
    microstructure_reads: list[dict[str, Any]] = []
    for call in invocation.tool_calls:
        if (call.get("server") != "price_path"
                or call.get("tool") not in {"get_price_path", "get_pre_anchor_microstructure"}):
            continue
        arguments = call.get("arguments") or {}
        try:
            row = price_path._allowed_anchor(  # access manifest 안의 anchor만 복원한다.
                access or {}, str(arguments.get("cluster")), str(arguments.get("symbol")),
                str(arguments.get("date")), int(arguments.get("anchor_tick")))
        except (TypeError, ValueError):
            continue
        target = microstructure_reads if call.get("tool") == "get_pre_anchor_microstructure" else reads
        target.append({"path_id": price_path.path_id(row, str((access or {}).get("horizon_key"))),
                       "cohort": str(row.get("cohort")),
                       "execution_label": str(
                           (row.get("canonical_execution") or {}).get("execution_label") or ""),
                       "diagnostic_cohort": str(
                           (row.get("canonical_execution") or {}).get("diagnostic_cohort") or "")})
    observed = {str(item["cohort"]) for item in reads}
    observed_execution_labels = {str(item["execution_label"]) for item in reads
                                 if item.get("execution_label")
                                 and str(item.get("cohort")) == "BACKGROUND"}
    observed_raw_labels = {str(item["diagnostic_cohort"]) for item in reads
                           if item.get("diagnostic_cohort")
                           and str(item.get("cohort")) == "BACKGROUND"}
    read_path_ids = {str(item["path_id"]) for item in reads}
    microstructure_path_ids = {str(item["path_id"]) for item in microstructure_reads}
    microstructure_cohorts = {str(item["cohort"]) for item in microstructure_reads}
    microstructure_execution_labels = {
        str(item["execution_label"]) for item in microstructure_reads
        if item.get("execution_label") and str(item.get("cohort")) == "BACKGROUND"}
    microstructure_raw_labels = {
        str(item["diagnostic_cohort"]) for item in microstructure_reads
        if item.get("diagnostic_cohort") and str(item.get("cohort")) == "BACKGROUND"}
    required = {cohort for cohort in ("PROFIT", "LOSS") if cohort in available}
    available_execution_labels = {
        str((item.get("canonical_execution") or {}).get("execution_label"))
        for item in (access or {}).get("anchors") or []
        if str(item.get("cohort")) == "BACKGROUND"
        and (item.get("canonical_execution") or {}).get("execution_label")
    }
    required_execution_labels = {
        label for label in ("EXECUTED_PROFIT", "EXECUTED_LOSS")
        if label in available_execution_labels
    }
    wanted_raw_labels = set(required_raw_labels or set())
    problems = (["list_price_paths MCP 호출이 없다"]
                if ((wanted_raw_labels if contract == "RAW_PATH" else
                     required_execution_labels if contract == "EXECUTION_STATE" else required)
                    and not listed) else [])
    if contract == "RAW_PATH":
        problems += [f"{label} BACKGROUND raw-path get_price_path MCP 호출이 없다"
                     for label in sorted(wanted_raw_labels - observed_raw_labels)]
    elif contract == "EXECUTION_STATE":
        problems += [f"{label} BACKGROUND get_price_path MCP 호출이 없다"
                     for label in sorted(required_execution_labels - observed_execution_labels)]
    else:
        problems += [f"{cohort} get_price_path MCP 호출이 없다"
                     for cohort in sorted(required - observed)]
    if contract == "RAW_PATH":
        problems += [f"{label} BACKGROUND get_pre_anchor_microstructure MCP 호출이 없다"
                     for label in sorted(wanted_raw_labels - microstructure_raw_labels)]
    elif contract == "EXECUTION_STATE":
        problems += [f"{label} BACKGROUND get_pre_anchor_microstructure MCP 호출이 없다"
                     for label in sorted(required_execution_labels - microstructure_execution_labels)]
    else:
        problems += [f"{cohort} get_pre_anchor_microstructure MCP 호출이 없다"
                     for cohort in sorted(required - microstructure_cohorts)]
    return {"status": "NOT_AVAILABLE" if access is None else "AVAILABLE",
            "listed_allowed_paths": listed,
            "required_cohorts": sorted(required), "observed_cohorts": sorted(observed),
            "required_execution_labels": sorted(required_execution_labels),
            "observed_execution_labels": sorted(observed_execution_labels),
            "required_raw_path_labels": sorted(wanted_raw_labels),
            "observed_raw_path_labels": sorted(observed_raw_labels),
            "path_contract": contract,
            "read_path_ids": sorted(read_path_ids),
            "reads": reads,
            "microstructure_path_ids": sorted(microstructure_path_ids),
            "microstructure_reads": microstructure_reads,
            "problems": problems}


def _path_observations_match_tool_reads(payload: Mapping[str, Any],
                                        usage: Mapping[str, Any], *,
                                        contract: str = "PROFILE",
                                        execution_state: bool | None = None) -> list[str]:
    """응답에 쓴 경로도 실제 MCP가 연 경로인지 대조한다."""
    if execution_state:
        contract = "EXECUTION_STATE"
    actual = {str(value) for value in usage.get("read_path_ids") or []}
    actual_execution = {
        (str(item.get("path_id")), str(item.get("execution_label")))
        for item in usage.get("reads") or [] if item.get("execution_label")
        and str(item.get("cohort")) == "BACKGROUND"
    }
    actual_raw = {
        (str(item.get("path_id")), str(item.get("diagnostic_cohort")))
        for item in usage.get("reads") or [] if item.get("diagnostic_cohort")
        and str(item.get("cohort")) == "BACKGROUND"
    }
    problems: list[str] = []
    for item in payload.get("path_observations") or []:
        if not isinstance(item, Mapping):
            continue
        path_id = str(item.get("path_id"))
        if contract == "RAW_PATH":
            if (str(item.get("cohort")) != "BACKGROUND"
                    or not str(item.get("diagnostic_cohort") or "")):
                continue
            if (path_id, str(item.get("diagnostic_cohort"))) not in actual_raw:
                problems.append(
                    f"{item.get('diagnostic_cohort')} BACKGROUND path_observations가 실제 "
                    f"get_price_path MCP 경로가 아니다: {path_id}")
        elif contract == "EXECUTION_STATE":
            if (str(item.get("cohort")) != "BACKGROUND"
                    or str(item.get("execution_label")) not in {"EXECUTED_PROFIT", "EXECUTED_LOSS"}):
                continue
            if (path_id, str(item.get("execution_label"))) not in actual_execution:
                problems.append(
                    f"{item.get('execution_label')} BACKGROUND path_observations가 실제 "
                    f"get_price_path MCP 경로가 아니다: {path_id}")
        elif str(item.get("cohort")) in {"PROFIT", "LOSS"} and path_id not in actual:
            problems.append(f"{item.get('cohort')} path_observations가 실제 get_price_path MCP 경로가 아니다: {path_id}")
    return problems


def _loss_ledger_tool_usage(invocation: AgentInvocation,
                            access: Mapping[str, Any] | None) -> dict[str, Any]:
    """손실 장부가 다음 가설의 질문에 쓰였는지 실제 MCP 기록으로만 남긴다."""
    case_cohorts = {str(item.get("case_id")): str(item.get("diagnostic_cohort") or "")
                    for item in (access or {}).get("cases") or [] if item.get("case_id")}
    case_ids = set(case_cohorts)
    listed = any(
        call.get("server") == "loss_ledger" and call.get("tool") == "list_execution_cases"
        for call in invocation.tool_calls)
    reads = [str((call.get("arguments") or {}).get("case_id"))
             for call in invocation.tool_calls
             if call.get("server") == "loss_ledger" and call.get("tool") == "get_execution_case"
             and str((call.get("arguments") or {}).get("case_id")) in case_ids]
    required_cohorts = sorted({cohort for cohort in ("PERSISTENT_ADVERSE", "NET_RECOVERY")
                               if cohort in set(case_cohorts.values())})
    observed_cohorts = sorted({case_cohorts[case_id] for case_id in reads
                               if case_id in case_cohorts})
    has_cases = bool(case_ids)
    return {
        "status": "NOT_AVAILABLE" if access is None else "AVAILABLE",
        "listed_execution_cases": listed,
        "read_case_ids": reads,
        "required_diagnostic_cohorts": required_cohorts,
        "observed_diagnostic_cohorts": observed_cohorts,
        "problems": (
            (["list_execution_cases MCP 호출이 없다"] if has_cases and not listed else [])
            + (["get_execution_case MCP 호출이 없다"] if has_cases and not reads else [])
            + [f"{cohort} get_execution_case MCP 호출이 없다"
               for cohort in required_cohorts if cohort not in observed_cohorts]
        ),
    }


def _loss_observations_match_tool_reads(payload: Mapping[str, Any],
                                        usage: Mapping[str, Any],
                                        access: Mapping[str, Any] | None) -> list[str]:
    """후속 가설이 실제로 연 Discovery 손실 사례를 연구 질문에 연결했는지 확인한다.

    사례는 새 가설의 Evidence가 아니다. 여기서 확인하는 것은 '무엇을 피하거나
    구별하려는 다음 질문인가'라는 provenance뿐이다. 실제로 열지 않은 사례 id나
    cohort를 응답에 적어 넣으면 Agent가 장부를 관찰한 것처럼 보이므로 막는다.
    """
    if access is None or payload.get("status") != H.GENERATED:
        return []
    observations = payload.get("loss_ledger_observations")
    if not isinstance(observations, list) or not observations:
        return ["Discovery loss ledger를 읽은 뒤의 loss_ledger_observations가 없다"]
    allowed = {str(item.get("case_id")): str(item.get("diagnostic_cohort"))
               for item in access.get("cases") or [] if item.get("case_id")}
    read_ids = {str(value) for value in usage.get("read_case_ids") or []}
    required_cohorts = {str(value) for value in usage.get("required_diagnostic_cohorts") or []}
    problems: list[str] = []
    observed_observation_cohorts: set[str] = set()
    for index, item in enumerate(observations):
        if not isinstance(item, Mapping):
            problems.append(f"loss_ledger_observations[{index}]가 객체가 아니다")
            continue
        case_id = str(item.get("case_id") or "")
        if case_id not in read_ids:
            problems.append(f"loss_ledger_observations[{index}]의 case_id가 실제 get_execution_case MCP 조회가 아니다")
        expected_cohort = allowed.get(case_id)
        if expected_cohort is None:
            problems.append(f"loss_ledger_observations[{index}]의 case_id가 허용된 Discovery 사례가 아니다")
        elif str(item.get("diagnostic_cohort") or "") != expected_cohort:
            problems.append(f"loss_ledger_observations[{index}]의 diagnostic_cohort가 tool 반환 사례와 다르다")
        else:
            observed_observation_cohorts.add(expected_cohort)
        if not str(item.get("observation") or "").strip():
            problems.append(f"loss_ledger_observations[{index}]의 observation이 비어 있다")
        if not str(item.get("research_question") or "").strip():
            problems.append(f"loss_ledger_observations[{index}]의 research_question이 비어 있다")
    for cohort in sorted(required_cohorts - observed_observation_cohorts):
        problems.append(f"{cohort} Discovery loss 관측을 loss_ledger_observations에 남기지 않았다")
    return problems


def _canonicalize_temporal_evidence_roles(payload: Mapping[str, Any], package: Mapping[str, Any]
                                          ) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """인용한 temporal/anchor-state Evidence의 고정 role을 Agent 표기보다 우선한다.

    `TRIGGER`와 `LATE_TRIGGER`를 섞어 쓰면 같은 저장 관측이 Stage마다 다른 역할처럼
    보인다. 이 role은 Agent의 금융 해석이 아니라 Evidence Package가 정한 관측 메타데이터라
    정확한 EVT/EVX id에서 결정적으로 복원한다.
    """
    result = copy.deepcopy(dict(payload))
    roles = {str(item.get("evidence_id")): str(item.get("role"))
             for item in [*H.execution_temporal_evidence_items(package),
                          *H.execution_state_evidence_items(package),
                          *H.raw_path_evidence_items(package)]
             if item.get("evidence_id") and item.get("role")}
    changes: list[dict[str, str]] = []
    for hypothesis in result.get("hypotheses") or []:
        if not isinstance(hypothesis, Mapping):
            continue
        basis = [item for item in hypothesis.get("evidence_basis") or []
                 if isinstance(item, Mapping)]
        grouped: dict[str, set[str]] = {}
        for item in basis:
            fixed = roles.get(str(item.get("evidence_id")))
            family = str(item.get("family") or "")
            if fixed and family:
                grouped.setdefault(family, set()).add(fixed)
        declared = hypothesis.get("evidence_roles")
        if not isinstance(declared, dict):
            continue
        for family, fixed_roles in grouped.items():
            if len(fixed_roles) != 1:
                continue
            fixed = next(iter(fixed_roles))
            if declared.get(family) != fixed:
                changes.append({"hypothesis_id": str(hypothesis.get("hypothesis_id")),
                                "family": family, "from": str(declared.get(family)),
                                "to": fixed})
                declared[family] = fixed
    return result, changes


def _preserve_generation_path_observations(final: Mapping[str, Any],
                                           draft: Mapping[str, Any]) -> dict[str, Any]:
    """감사 Agent가 도구로 실제 읽은 경로 기록을 삭제하지 못하게 한다.

    Evidence audit은 가설의 주장 범위를 줄이는 역할이다. 반면 ``path_observations``는
    generation Agent가 MCP에서 받은 사실의 원장이다. 최종 결과가 여전히 가설을 내는
    경우에는 이 원장을 generation draft에서 그대로 이어야 도구 사용 계약을 충족한다.
    경로가 허용된 것인지와 관측 문구가 비어 있지 않은지는 뒤의 결정적 validator가
    다시 검사한다.
    """
    result = copy.deepcopy(dict(final))
    if (result.get("status") == H.GENERATED
            and draft.get("status") == H.GENERATED
            and isinstance(draft.get("path_observations"), list)):
        result["path_observations"] = copy.deepcopy(draft["path_observations"])
    if (result.get("status") == H.GENERATED
            and draft.get("status") == H.GENERATED
            and isinstance(draft.get("loss_ledger_observations"), list)):
        result["loss_ledger_observations"] = copy.deepcopy(
            draft["loss_ledger_observations"])
    return result


def run(package: Mapping[str, Any], *, runner: Runner | None = None,
        config: H.AgentConfig = H.AgentConfig(),
        discovery_loss_context: Mapping[str, Any] | None = None,
        development_validation_feedback: Mapping[str, Any] | None = None,
        candidate_library: Mapping[str, Any] | None = None,
        research_lens: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """기존 프롬프트/결정적 검사를 재사용하고, 파일 기록은 Stage에만 맡긴다."""
    active_runner = runner or CodexAgentRunner(tool_timeout_seconds=config.tool_timeout_seconds)
    profile = C.research_capability_profile()
    audit = H.input_audit(package, config)
    digest = H.evidence_digest(package, audit)
    if not digest.get("evidence"):
        raise ValueError("Hypothesis 생성에 사용할 Evidence가 없다")
    if research_lens is not None:
        digest = dict(digest)
        digest["research_lens"] = dict(research_lens)
    path_access: Mapping[str, Any] | None = None
    loss_access = None
    path_context_status: dict[str, Any] = {"status": "NOT_REQUESTED"}
    if ((package.get("path_observation") or {}).get("anchors")
            and package.get("price_path_context_required", True)):
        try:
            opened = price_path.access_manifest(package) or {}
            if not opened.get("anchors"):
                raise ValueError("가격 경로 anchor가 0개다")
            path_access = opened
            path_context_status = {"status": "AVAILABLE",
                                   "anchors_by_cohort": _access_counts(path_access)}
            digest = dict(digest)
            # attached cache가 Profile의 일부일 수 있으므로, tool이 실제로 연 anchor 수를
            # Evidence package 전체 anchor 수 대신 정확히 알린다.
            digest["price_path_tool"] = {
                **dict(digest.get("price_path_tool") or {}),
                "anchors_by_cohort": _access_counts(path_access),
                "cache_limited": len(path_access.get("anchors") or []) < len(
                    (package.get("path_observation") or {}).get("anchors") or []),
            }
        except (FileNotFoundError, ValueError, price_path.profit.CacheError) as error:
            path_context_status = {"status": "UNAVAILABLE", "message": str(error)}
            digest = dict(digest)
            digest.pop("price_path_tool", None)
            digest["price_path_context"] = dict(path_context_status)
    elif (package.get("path_observation") or {}).get("anchors"):
        path_context_status = {
            "status": "UNAVAILABLE",
            "message": "저장된 FeatureProfile에 대응 discovery price-path cache가 없다",
        }
        digest = dict(digest)
        digest.pop("price_path_tool", None)
        digest["price_path_context"] = dict(path_context_status)
    if discovery_loss_context is not None:
        loss_access = loss_ledger.access_manifest(discovery_loss_context)
        digest = dict(digest)
        digest["discovery_execution_context"] = {
            "source_hypothesis_id": discovery_loss_context.get("source_hypothesis_id"),
            "interpretation_boundary": discovery_loss_context.get("interpretation_boundary"),
            "summary": discovery_loss_context.get("summary"),
            "prior_hypothesis": discovery_loss_context.get("prior_hypothesis"),
            "prior_entry_policy": {
                "refinement_round": (discovery_loss_context.get("source") or {}).get("round"),
                "entry_guards": (discovery_loss_context.get("source") or {}).get("entry_guards") or [],
            },
        }
        digest["loss_ledger_tool"] = {
            "source_hypothesis_id": discovery_loss_context.get("source_hypothesis_id"),
            "case_samples": len(loss_access.get("cases") or []),
            "policy": "허용된 Discovery 실행 사례만 읽는다. 이 결과는 새 가설 Evidence가 아니다.",
        }
    if development_validation_feedback is not None:
        source = development_validation_feedback.get("source") or {}
        cases = [item for item in development_validation_feedback.get("case_samples") or []
                 if isinstance(item, Mapping)]
        digest = dict(digest)
        digest["development_validation_feedback"] = {
            "source_hypothesis_id": development_validation_feedback.get("source_hypothesis_id"),
            "interpretation_boundary": development_validation_feedback.get(
                "interpretation_boundary"),
            "validation_dates": source.get("validation_dates") or [],
            "net_bps_total": source.get("net_bps_total"),
            "parameter_lock": source.get("parameter_lock") or {},
            "prior_entry_policy": {"entry_guards": source.get("entry_guards") or []},
            "summary": development_validation_feedback.get("summary") or {},
            "case_samples": [{key: case.get(key) for key in (
                "case_id", "diagnostic_cohort", "symbol", "date", "entry_tick",
                "fill_tick", "entry_status", "exit_reason", "net_bps", "gross_bps",
                "max_favorable_gross_bps", "max_adverse_gross_bps", "pre_entry_features")}
                             for case in cases[:8]],
        }
    library_records = list((candidate_library or {}).get("records") or [])
    if library_records:
        digest = dict(digest)
        digest["historical_candidate_library"] = {
            "note": (candidate_library or {}).get("note"),
            "records": library_records,
        }
    agent_digest = _compact_generation_digest(digest, research_lens=research_lens)
    invocations: list[AgentInvocation] = []
    generation_prompt = H.generation_prompt(agent_digest, config, profile)
    try:
        tools = {name: access for name, access in (("price_path", path_access),
                                                    ("loss_ledger", loss_access)) if access is not None}
        draft, generated = invoke(active_runner, role="hypothesis_generation",
                                  prompt=generation_prompt,
                                  model=config.model, effort=config.effort,
                                  tool_context=tools or None)
    except H.AgentCallError as error:
        return _failure("hypothesis_generation", generation_prompt, error,
                        audit, agent_digest, config, invocations)
    invocations.append(generated)
    loss_ledger_tool_usage = _loss_ledger_tool_usage(generated, loss_access)
    audit_prompt = H.audit_prompt(agent_digest, draft, profile)
    try:
        final, audited = invoke(active_runner, role="hypothesis_evidence_audit",
                                prompt=audit_prompt,
                                model=config.model, effort=config.effort)
    except H.AgentCallError as error:
        return _failure("hypothesis_evidence_audit", audit_prompt, error,
                        audit, agent_digest, config, invocations)
    final = _preserve_generation_path_observations(final, draft)
    final.setdefault("evidence_synthesis", draft.get("evidence_synthesis", {}))
    final = H.apply_fixed_contracts(
        final, package, path_observations_available=path_access is not None)
    payload, temporal_role_changes = _canonicalize_temporal_evidence_roles(final, package)
    invocations.append(audited)
    path_contract = _path_contract(
        payload, package, research_lens)
    price_path_tool_usage = _price_path_tool_usage(
        generated, path_access, contract=path_contract,
        required_raw_labels=H.raw_path_labels_used(payload, package))
    validation = H.validate_output(
        payload, package, config, path_observations_available=path_access is not None)
    if validation["problems"]:
        repair_prompt = H.contract_repair_prompt(
            agent_digest, payload, validation["problems"], profile)
        try:
            repaired, repaired_invocation = invoke(
                active_runner, role="hypothesis_contract_repair", prompt=repair_prompt,
                model=config.model, effort=config.effort)
        except H.AgentCallError as error:
            return _failure("hypothesis_contract_repair", repair_prompt, error,
                            audit, agent_digest, config, invocations)
        repaired = H.apply_fixed_contracts(
            repaired, package, path_observations_available=path_access is not None)
        payload, repaired_changes = _canonicalize_temporal_evidence_roles(repaired, package)
        temporal_role_changes.extend(repaired_changes)
        invocations.append(repaired_invocation)
        validation = H.validate_output(
            payload, package, config, path_observations_available=path_access is not None)
    if payload.get("status") == H.LEGACY_NO_HYPOTHESIS:
        payload = {key: value for key, value in payload.items()
                   if key != "no_hypothesis_reason"}
        payload["status"] = "INVALID_OUTPUT"
        validation = H.validate_output(
            payload, package, config, path_observations_available=path_access is not None)
    enforce_tool_contract = runner is None or isinstance(active_runner, CodexAgentRunner)
    if enforce_tool_contract and payload.get("status") == H.GENERATED:
        validation = dict(validation)
        validation["problems"] = [*validation.get("problems", []),
                                  *price_path_tool_usage["problems"],
                                  *_path_observations_match_tool_reads(
                                      payload, price_path_tool_usage, contract=path_contract),
                                  *loss_ledger_tool_usage["problems"],
                                  *_loss_observations_match_tool_reads(
                                      payload, loss_ledger_tool_usage, loss_access)]
    lens_problems = _validate_lens(payload, research_lens)
    if lens_problems:
        validation = dict(validation)
        validation["problems"] = [*validation.get("problems", []), *lens_problems]
    novelty: Mapping[str, Any] | None = None
    prior = discovery_loss_context.get("prior_hypothesis") if discovery_loss_context else None
    historical = ([prior] if isinstance(prior, Mapping) else []) + library_records
    if historical and payload.get("status") == H.GENERATED:
        novelty_prompt = H.novelty_prompt(historical, payload)
        try:
            novelty, novelty_invocation = invoke(active_runner, role="hypothesis_novelty_audit",
                                                  prompt=novelty_prompt,
                                                  model=config.model, effort=config.effort)
        except H.AgentCallError as error:
            return _failure("hypothesis_novelty_audit", novelty_prompt, error,
                            audit, digest, config, invocations)
        invocations.append(novelty_invocation)
        novelty = H.validate_novelty(novelty, payload)
    if validation["problems"]:
        state = "INVALID_OUTPUT"
    elif novelty is not None and novelty["status"] == H.DUPLICATE_HYPOTHESIS:
        state = H.DUPLICATE_HYPOTHESIS
    else:
        state = "READY_FOR_GROUNDING"
    return {
        "payload": payload,
        "input_audit": audit,
        "validation": validation,
        "price_path_context": path_context_status,
        "price_path_tool_usage": price_path_tool_usage,
        "loss_ledger_tool_usage": loss_ledger_tool_usage,
        "discovery_loss_context": ({"status": "OPENED",
                                      "source_hypothesis_id": discovery_loss_context.get("source_hypothesis_id")}
                                   if discovery_loss_context is not None else {"status": "NOT_REQUESTED"}),
        "development_validation_feedback": (
            {"status": "OPENED",
             "source_hypothesis_id": development_validation_feedback.get("source_hypothesis_id")}
            if development_validation_feedback is not None else {"status": "NOT_REQUESTED"}),
        "novelty": dict(novelty) if novelty is not None else {"status": "NOT_REQUESTED"},
        "candidate_library": ({"status": "OPENED", "records": len(library_records)}
                              if candidate_library is not None else {"status": "NOT_REQUESTED"}),
        "research_lens": (dict(research_lens) if research_lens is not None
                          else {"status": "NOT_REQUESTED"}),
        "temporal_role_canonicalization": temporal_role_changes,
        "agent_invocations": [_invocation_record(item) for item in invocations],
        "config": asdict(config),
        "state": state,
    }


def _validate_lens(payload: Mapping[str, Any], lens: Mapping[str, Any] | None) -> list[str]:
    """Lens를 열었으면 Agent가 다른 관계로 비켜가지 않았는지 확인한다."""
    if lens is None:
        return []
    families = {str(value) for value in lens.get("families") or []}
    required_evidence_ids = {str(value) for value in lens.get("required_evidence_ids") or []}
    relation = str(lens.get("relation"))
    structure = str(lens.get("allowed_hypothesis_structure"))
    hypotheses = list(payload.get("hypotheses") or [])
    problems: list[str] = []
    if len(hypotheses) > 1:
        problems.append("research lens는 한 관계당 가설 하나만 허용한다")
    for item in hypotheses:
        pair_found = relation == "SINGLE" or any(
            {str(value.get("a")), str(value.get("b"))} == families
            and str(value.get("relation")) == relation
            for value in item.get("evidence_relations") or [] if isinstance(value, Mapping))
        if not pair_found:
            problems.append(f"{item.get('hypothesis_id')}: research lens 관계를 쓰지 않았다")
        if str(item.get("hypothesis_structure")) != structure:
            problems.append(f"{item.get('hypothesis_id')}: lens 구조와 다른 hypothesis_structure")
        basis_families = {str(value.get("family")) for value in item.get("evidence_basis") or []
                          if isinstance(value, Mapping)}
        if basis_families != families:
            problems.append(f"{item.get('hypothesis_id')}: lens 밖 family를 핵심 basis로 섞었거나 두 family가 빠졌다")
        basis_ids = {str(value.get("evidence_id")) for value in item.get("evidence_basis") or []
                     if isinstance(value, Mapping) and value.get("evidence_id")}
        if required_evidence_ids and basis_ids != required_evidence_ids:
            problems.append(f"{item.get('hypothesis_id')}: lens가 고정한 evidence_id를 바꿨거나 다른 Evidence를 섞었다")
    return problems


def _failure(role: str, prompt: str, error: H.AgentCallError,
             audit: Mapping[str, Any], digest: Mapping[str, Any],
             config: H.AgentConfig, invocations: list[AgentInvocation]) -> dict[str, Any]:
    """실패도 Stage artifact로 남겨 재시도와 연구 기록의 근거로 쓴다."""
    return {
        "payload": {},
        "input_audit": audit,
        "evidence_digest_sha256": sha256_json(digest)[:16],
        "validation": {"problems": ["Agent 호출 실패"], "warnings": []},
        "agent_invocations": [_invocation_record(item) for item in invocations],
        "agent_error": {
            "role": role,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            **error.as_dict(),
        },
        "config": asdict(config),
        "state": "AGENT_CALL_FAILED",
    }


def novelty_only(source_result: Mapping[str, Any], candidate_library: Mapping[str, Any], *,
                 runner: Runner | None = None,
                 config: H.AgentConfig = H.AgentConfig()) -> dict[str, Any]:
    """기존 Agent payload는 고치지 않고 현재 historical library와만 다시 비교한다."""
    result = dict(source_result)
    payload = result.get("payload") or {}
    records = list(candidate_library.get("records") or [])
    if payload.get("status") != H.GENERATED or not records:
        result["novelty"] = {"status": "NOT_REQUESTED"}
        return result
    prompt = H.novelty_prompt(records, payload)
    invocations = list(result.get("agent_invocations") or [])
    try:
        response, invocation = invoke(runner, role="hypothesis_novelty_audit", prompt=prompt,
                                      model=config.model, effort=config.effort)
    except H.AgentCallError as error:
        return _failure("hypothesis_novelty_audit", prompt, error,
                        result.get("input_audit") or {}, {}, config, invocations)
    novelty = H.validate_novelty(response, payload)
    invocations.append(invocation)
    result.update({"novelty": novelty, "agent_invocations": [_invocation_record(item) for item in invocations],
                   "candidate_library": {"status": "OPENED", "records": len(records)},
                   "state": (H.DUPLICATE_HYPOTHESIS if novelty["status"] == H.DUPLICATE_HYPOTHESIS
                             else result.get("state", "READY_FOR_GROUNDING"))})
    return result
