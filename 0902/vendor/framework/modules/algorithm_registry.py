"""완료된 Final Backtest를 entry 계약 단위로 읽기 전용 집계한다."""

from __future__ import annotations

import math
import fcntl
from pathlib import Path
from typing import Any, Mapping

from .. import canonical as K
from ..config import clean, sha256_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact


ARTIFACT_NAME = "final_algorithm_registry_artifact.json"
SCHEMA_VERSION = "final_algorithm_registry.v1"
TARGET_POSITIVE_ALGORITHMS = 50
CURRENT_OUTPUT_RELATIVE = Path("artifacts") / "final_algorithm_registry_current"
CURRENT_LOCK_NAME = ".final_algorithm_registry.lock"


def _entry_identity(template: Mapping[str, Any]) -> dict[str, Any]:
    """Search q·Entry Refinement guard와 무관한 실행 entry의 정체성이다."""
    binding = template.get("execution_binding") or {}
    program = template.get("entry_program") or {}
    return clean({
        "schema": "entry_algorithm_identity.v1",
        "side": template.get("side"),
        "signal": program.get("signal"),
        "warmup_ticks": program.get("warmup_ticks"),
        "decision_opportunity": binding.get("decision_opportunity"),
        "decision_spacing_seconds": binding.get("decision_spacing_seconds"),
        "decision_basis": binding.get("decision_basis"),
        "entry_execution": binding.get("entry_execution"),
        "entry_reference": binding.get("entry_reference"),
        "evaluation_horizon": binding.get("evaluation_horizon"),
    })


def _template_for(run_root: Path, hypothesis_id: str) -> Mapping[str, Any] | None:
    """Final unit에 대응하는 parameterized 실행 template 하나를 찾는다."""
    path = run_root / "05_parameter_search" / "parameter_search_artifact.json"
    if not path.is_file():
        return None
    try:
        search = read_artifact(path, kind="parameter_search_set")
    except (OSError, ValueError):
        return None
    unit = (search["payload"].get("units") or {}).get(str(hypothesis_id)) or {}
    template = unit.get("parameterized_template") or {}
    return dict(template) if isinstance(template, Mapping) else None


def _proposal_provenance(run_root: Path, hypothesis_id: str) -> dict[str, Any] | None:
    """Final entry가 어떤 서로 다른 연구 제안에서 왔는지 짧게 보존한다."""
    direct_path = run_root / "02_specification" / "direct_evidence_candidates_artifact.json"
    if direct_path.is_file():
        try:
            direct = read_artifact(direct_path, kind="direct_evidence_candidate_set")
        except (OSError, ValueError):
            direct = None
        for candidate in ((direct or {}).get("payload", {}).get("candidates") or []):
            if isinstance(candidate, Mapping) and str(candidate.get("hypothesis_id")) == hypothesis_id:
                basis = list(candidate.get("evidence_basis") or [])
                return {
                    "kind": "DIRECT_EVIDENCE_ENTRY_LOGIC",
                    "source_artifact": str(direct["artifact_id"]),
                    "source_type": str(candidate.get("source_type") or ""),
                    "entry_logic_fingerprint": str(candidate.get("entry_logic_fingerprint") or ""),
                    "evidence_ids": sorted(str(item.get("evidence_id")) for item in basis
                                           if isinstance(item, Mapping) and item.get("evidence_id")),
                }

    for path in sorted(run_root.rglob("hypothesis_artifact.json")):
        try:
            hypotheses = read_artifact(path, kind="hypothesis_set")
        except (OSError, ValueError):
            continue
        payload = hypotheses.get("payload") or {}
        for proposal in (payload.get("payload") or {}).get("hypotheses") or []:
            if not isinstance(proposal, Mapping) or str(proposal.get("hypothesis_id")) != hypothesis_id:
                continue
            basis = list(proposal.get("evidence_basis") or [])
            mechanism = {
                "hypothesis_structure": proposal.get("hypothesis_structure"),
                "mechanistic_claim": proposal.get("mechanistic_claim"),
                "source_of_profit": proposal.get("source_of_profit"),
                "evidence_relations": proposal.get("evidence_relations"),
            }
            return {
                "kind": "AGENT_HYPOTHESIS",
                "source_artifact": str(hypotheses["artifact_id"]),
                "hypothesis_structure": proposal.get("hypothesis_structure"),
                "title": proposal.get("title"),
                "evidence_families": sorted({str(item.get("family")) for item in basis
                                              if isinstance(item, Mapping) and item.get("family")}),
                "mechanism_sha256": sha256_json(mechanism),
            }
    return None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _final_record(run_root: Path, hypothesis_id: str, unit: Mapping[str, Any]) -> dict[str, Any]:
    """한 Final unit을 숨기지 않고, 카운트 가능 여부를 함께 기록한다."""
    record: dict[str, Any] = {
        "run": str(run_root),
        "hypothesis_id": str(hypothesis_id),
        "final_state": unit.get("state"),
        "final_backtest_opened": bool(unit.get("final_backtest_opened")),
    }
    if unit.get("state") != "BACKTEST_COMPLETE":
        return {**record, "count_status": "FINAL_NOT_EXECUTED"}

    metrics = unit.get("metrics") or {}
    total = _numeric(metrics.get("net_bps_total"))
    per_decision = _numeric(metrics.get("net_bps_per_decision"))
    record["metrics"] = {
        "net_bps_total": total,
        "net_bps_per_decision": per_decision,
        "scorable": metrics.get("scorable"),
        "fills": metrics.get("fills"),
    }
    contract = unit.get("contract_check") or {}
    profile_matches = (
        contract.get("profile_sha256") == K.CANONICAL["profile_sha256"]
        and bool(contract.get("ok"))
        and bool(unit.get("invariants_ok"))
    )
    record["canonical_profile_match"] = profile_matches
    template = _template_for(run_root, str(hypothesis_id))
    if template is None:
        return {**record, "count_status": "MISSING_ENTRY_TEMPLATE"}
    identity = _entry_identity(template)
    record["entry_identity"] = identity
    record["entry_identity_sha256"] = sha256_json(identity)
    provenance = _proposal_provenance(run_root, str(hypothesis_id))
    if provenance is None:
        return {**record, "count_status": "MISSING_HYPOTHESIS_PROVENANCE"}
    record["hypothesis_provenance"] = provenance
    if not profile_matches:
        return {**record, "count_status": "CANONICAL_PROFILE_MISMATCH"}
    if total is None:
        return {**record, "count_status": "MISSING_FINAL_METRIC"}
    if total <= 0:
        return {**record, "count_status": "FINAL_NOT_POSITIVE"}
    return {**record, "count_status": "POSITIVE_FINAL"}


def build(runs_root: Path, *, target: int = TARGET_POSITIVE_ALGORITHMS) -> dict[str, Any]:
    """`runs/*/07_backtest`를 읽어 서로 다른 양수 Final entry 계약을 센다."""
    root = Path(runs_root)
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for path in sorted(root.rglob("07_backtest/backtest_artifact.json")):
        run_root = path.parent.parent
        try:
            artifact = read_artifact(path, kind="backtest_set")
        except (OSError, ValueError) as exc:
            unreadable.append({"path": str(path), "reason": str(exc)})
            continue
        for hypothesis_id, unit in sorted((artifact["payload"].get("units") or {}).items()):
            if isinstance(unit, Mapping):
                records.append(_final_record(run_root, str(hypothesis_id), unit))
            else:
                records.append({
                    "run": str(run_root), "hypothesis_id": str(hypothesis_id),
                    "count_status": "INVALID_FINAL_UNIT",
                })

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("count_status") != "POSITIVE_FINAL":
            continue
        grouped.setdefault(str(record["entry_identity_sha256"]), []).append(record)
    positive_identities = []
    for identity_sha, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda item: (str(item["run"]), str(item["hypothesis_id"])))
        positive_identities.append({
            "entry_identity_sha256": identity_sha,
            "entry_identity": ordered[0]["entry_identity"],
            "distinctness_basis": ordered[0]["hypothesis_provenance"],
            "positive_final_observations": ordered,
            "duplicate_positive_final_observation_count": max(0, len(ordered) - 1),
        })
    completed = [record for record in records if record.get("final_state") == "BACKTEST_COMPLETE"]
    return {
        "schema": SCHEMA_VERSION,
        "runs_root": str(root),
        "canonical_profile": {
            "profile_id": K.CANONICAL["profile_id"],
            "profile_sha256": K.CANONICAL["profile_sha256"],
        },
        "positive_final_rule": "net_bps_total > 0",
        "identity_rule": (
            "same parameterized entry signal and canonical execution binding is one algorithm; "
            "q and Entry Refinement guards are not separate algorithms. Each counted identity "
            "must retain an Agent hypothesis or direct-Evidence entry-logic provenance."),
        "target_positive_algorithm_count": int(target),
        "positive_algorithm_count": len(positive_identities),
        "remaining_to_target": max(0, int(target) - len(positive_identities)),
        "final_observation_count": len(completed),
        "positive_final_observation_count": sum(
            record.get("count_status") == "POSITIVE_FINAL" for record in records),
        "positive_identities": positive_identities,
        "nonqualifying_observations": [
            record for record in records if record.get("count_status") != "POSITIVE_FINAL"],
        "unreadable_backtest_artifacts": unreadable,
        "use_boundary": (
            "이 registry는 Final 결과의 보고·중복 제거용이다. Final 결과를 다음 Discovery, "
            "Entry Refinement, Search 또는 Validation의 입력으로 쓰지 않는다."),
    }


def write(runs_root: Path, output: Path, *, target: int = TARGET_POSITIVE_ALGORITHMS) -> dict[str, Any]:
    """집계 결과를 재현 가능한 Sandbox artifact로 남긴다."""
    payload = build(runs_root, target=target)
    artifact = create_artifact("final_algorithm_registry", payload)
    return write_artifact(Path(output) / ARTIFACT_NAME, artifact)


def refresh_current(sandbox_root: Path | None = None, *,
                    target: int = TARGET_POSITIVE_ALGORITHMS) -> dict[str, Any]:
    """완료 Final 뒤 현재 Sandbox registry 하나를 직렬화해 갱신한다.

    여러 Research workflow가 거의 같은 때 끝나도 동일한 ``.tmp`` 파일을 쓰지 않게
    lock을 잡는다. registry는 보고용이라 Discovery·Search·Validation 입력에는 열지 않는다.
    """
    sandbox = Path(sandbox_root or Path(__file__).resolve().parents[2]).resolve()
    output = sandbox / CURRENT_OUTPUT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / CURRENT_LOCK_NAME
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return write(sandbox / "runs", output, target=target)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
