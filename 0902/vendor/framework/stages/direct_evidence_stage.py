"""Evidence artifact에서 제한된 직접 관측 후보의 Specification을 만든다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import direct_evidence


CANDIDATES_ARTIFACT_NAME = "direct_evidence_candidates_artifact.json"
ARTIFACT_NAME = "specification_artifact.json"


def _matches_requested_id(candidate: dict[str, Any], requested: tuple[str, ...]) -> bool:
    """새 semantic id와 과거 CLI id 둘 다 선택한다."""
    identities = {str(candidate.get("hypothesis_id") or ""),
                  str(candidate.get("legacy_hypothesis_id") or "")}
    return bool(identities & set(requested))


def run(evidence_path: Path, output: Path, *, source_types: tuple[str, ...] = (),
        candidate_ids: tuple[str, ...] = (),
        exclude_entry_logic_fingerprints: tuple[str, ...] = (),
        execution_outcome_fill_compound_limit: int | None = None,
        execution_stop_avoidance_fill_compound_limit: int | None = None,
        state_quantile_mode: str = "SHARED_STATE_QUANTILE",
        entry_lifecycle_mode: str = "HOLD_THROUGH") -> dict[str, Any]:
    evidence = read_artifact(evidence_path, kind="evidence_package")
    output = Path(output)
    limits = {
        "execution_outcome_fill_compound_limit": execution_outcome_fill_compound_limit,
        "execution_stop_avoidance_fill_compound_limit": (
            execution_stop_avoidance_fill_compound_limit),
    }
    for name, value in limits.items():
        if value is not None and value < 1:
            raise ValueError(f"{name}은 1 이상이어야 한다")
    if state_quantile_mode not in {"SHARED_STATE_QUANTILE", "INDEPENDENT_STATE_QUANTILES"}:
        raise ValueError(f"state_quantile_mode이 유효하지 않다: {state_quantile_mode}")
    if entry_lifecycle_mode not in {"HOLD_THROUGH", "CANCEL_WHEN_ENTRY_SIGNAL_FALSE"}:
        raise ValueError(f"entry_lifecycle_mode이 유효하지 않다: {entry_lifecycle_mode}")
    config = direct_evidence.DirectEvidenceConfig(**{
        name: int(value) for name, value in limits.items() if value is not None})
    payload = direct_evidence.candidates(evidence["payload"]["package"], config)
    excluded = tuple(sorted({str(value) for value in exclude_entry_logic_fingerprints if str(value)}))
    if (source_types or candidate_ids or excluded
            or state_quantile_mode != "SHARED_STATE_QUANTILE"
            or entry_lifecycle_mode != "HOLD_THROUGH"):
        wanted = tuple(str(value) for value in source_types)
        wanted_ids = tuple(str(value) for value in candidate_ids)
        all_candidates = list(payload.get("candidates") or [])
        selection_mode = {"source_types": list(wanted)}
        if execution_outcome_fill_compound_limit is not None:
            selection_mode["execution_outcome_fill_compound_limit"] = int(
                execution_outcome_fill_compound_limit)
        if execution_stop_avoidance_fill_compound_limit is not None:
            selection_mode["execution_stop_avoidance_fill_compound_limit"] = int(
                execution_stop_avoidance_fill_compound_limit)
        if wanted_ids:
            selection_mode["candidate_ids"] = list(wanted_ids)
        if excluded:
            selection_mode["excluded_entry_logic_fingerprints"] = list(excluded)
        if state_quantile_mode != "SHARED_STATE_QUANTILE":
            selection_mode["state_quantile_mode"] = state_quantile_mode
        if entry_lifecycle_mode != "HOLD_THROUGH":
            selection_mode["entry_lifecycle_mode"] = entry_lifecycle_mode
        selected_candidates = []
        for candidate in all_candidates:
            selected = {**candidate, **({"state_quantile_mode": state_quantile_mode}
                                          if state_quantile_mode != "SHARED_STATE_QUANTILE" else {})}
            if entry_lifecycle_mode == "CANCEL_WHEN_ENTRY_SIGNAL_FALSE":
                selected = {
                    **selected,
                    "entry_lifecycle_mode": entry_lifecycle_mode,
                    "legacy_hypothesis_id": candidate["hypothesis_id"],
                    "hypothesis_id": str(candidate["hypothesis_id"])
                                     + "__LIFECYCLE_CONTINUITY",
                    "entry_logic_fingerprint": str(candidate.get("entry_logic_fingerprint") or "")
                                             + ":cancel_when_entry_signal_false",
                }
            if ((wanted and str(candidate.get("source_type")) not in wanted)
                    or (wanted_ids and not _matches_requested_id(candidate, wanted_ids))
                    or str(selected.get("entry_logic_fingerprint") or "") in excluded
                    or (entry_lifecycle_mode != "HOLD_THROUGH"
                        and str(candidate.get("source_type"))
                        != "DIRECT_DISCOVERY_ANCHOR_EXECUTION_STATE_EVIDENCE")):
                continue
            selected_candidates.append(selected)
        payload = {
            **payload,
            "selection_mode": selection_mode,
            "candidates": selected_candidates,
        }
        if wanted_ids:
            selected_ids = {identity for candidate in selected_candidates for identity in (
                str(candidate.get("hypothesis_id") or ""),
                str(candidate.get("legacy_hypothesis_id") or ""))}
            payload["unselected_requested_candidate_ids"] = [
                candidate_id for candidate_id in wanted_ids if candidate_id not in selected_ids]
        payload["excluded_existing_entry_logic_count"] = int(
            sum(str(candidate.get("entry_logic_fingerprint") or "") in excluded
                for candidate in all_candidates)) if excluded else 0
        payload["state"] = ("DIRECT_EVIDENCE_CANDIDATES_READY" if payload["candidates"]
                            else "NO_DIRECT_EVIDENCE_CANDIDATE")
    candidate_artifact = create_artifact(
        "direct_evidence_candidate_set",
        payload,
        parents={"evidence_package": evidence["artifact_id"]},
    )
    write_artifact(output / CANDIDATES_ARTIFACT_NAME, candidate_artifact)
    units: dict[str, Any] = {}
    for candidate in candidate_artifact["payload"].get("candidates") or []:
        hypothesis_id = str(candidate["hypothesis_id"])
        if candidate.get("temporal_entry"):
            # Parameter Search는 30초 anchor 표에서 상태 임계만 비교한다. tick/clock
            # 사건형 진입을 그 표에 억지로 축소하면 q가 실제 entry 계약과 달라진다.
            units[hypothesis_id] = {
                "state": "TEMPORAL_ENTRY_NOT_SUPPORTED_BY_PARAMETER_SEARCH",
                "reason": ("현재 Parameter Search는 level 상태만 같은 entry DSL로 "
                           "평가한다. temporal_entry는 전용 tick-level Search가 필요하다"),
            }
            continue
        units[hypothesis_id] = direct_evidence.run(
            candidate, evidence["payload"]["package"], output / "units" / hypothesis_id)
    state = "SPECIFICATION_COMPLETE" if units else "NO_DIRECT_EVIDENCE_CANDIDATE"
    specifications = create_artifact(
        "executable_specification_set", {"state": state, "units": units},
        parents={"evidence_package": evidence["artifact_id"],
                 "direct_evidence_candidate_set": candidate_artifact["artifact_id"]},
    )
    write_artifact(output / ARTIFACT_NAME, specifications)
    return {"candidates": candidate_artifact, "specifications": specifications}
