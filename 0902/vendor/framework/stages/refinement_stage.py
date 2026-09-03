"""Implementation 뒤, Parameter Search 앞의 고정-exit entry refinement Stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import search
from ..config import ENTRY_REFINEMENT_POLICY_VERSION, TICK_ROOT, read_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import refinement


ARTIFACT_NAME = "entry_refinement_artifact.json"


def run(specification_path: Path, implementation_path: Path, evidence_path: Path, output: Path, *,
        root: Path = TICK_ROOT,
        config: refinement.RefinementConfig = refinement.RefinementConfig(),
        runner: Any | None = None) -> dict[str, Any]:
    """실행 가능한 계약만 고정 청산 장부로 진입 조항을 보완한다."""
    specifications = read_artifact(specification_path, kind="executable_specification_set")
    implementations = read_artifact(implementation_path, kind="implementation_set")
    evidence = read_artifact(evidence_path, kind="evidence_package")
    if implementations["parents"].get("executable_specification_set") != specifications["artifact_id"]:
        raise ValueError("Implementation과 Specification의 lineage가 다르다")
    if specifications["parents"].get("evidence_package") != evidence["artifact_id"]:
        raise ValueError("Specification과 Evidence의 lineage가 다르다")
    dates = tuple(evidence["payload"].get("profile_dates") or [])
    symbols = tuple(evidence["payload"].get("profile_symbols") or [])
    if not dates or not symbols:
        raise ValueError("Entry refinement에 필요한 Feature Profile 날짜 또는 종목이 없다")

    output = Path(output)
    units: dict[str, Any] = {}
    pending: dict[str, tuple[dict[str, Any], refinement.RefinementConfig]] = {}
    for hypothesis_id, specification in specifications["payload"].get("units", {}).items():
        hypothesis_id = str(hypothesis_id)
        if "spec" not in specification:
            units[hypothesis_id] = {"state": specification.get("state"),
                                    "reason": specification.get("reason")}
            continue
        implementation = implementations["payload"].get("units", {}).get(hypothesis_id)
        if implementation is None:
            raise ValueError(f"Entry refinement의 Implementation이 없다: {hypothesis_id}")
        if implementation.get("state") != "READY_FOR_PARAMETER_SEARCH":
            units[hypothesis_id] = {"state": "IMPLEMENTATION_NOT_READY"}
            continue
        effective_config = refinement.resolve_config(specification["spec"], config)
        existing = output / "units" / hypothesis_id / "entry_refinement.json"
        if existing.is_file():
            saved = read_json(existing)
            amended = search.amend_specification(specification["spec"],
                                                  search.config_for(specification["spec"]))
            if (saved.get("hypothesis_id") == hypothesis_id
                    and saved.get("amended_specification_sha256")
                    == amended.get("specification_sha256")
                    and saved.get("entry_refinement_policy_version")
                    == ENTRY_REFINEMENT_POLICY_VERSION
                    and saved.get("config") == refinement.config_payload(effective_config)):
                units[hypothesis_id] = saved
                continue
        pending[hypothesis_id] = (dict(specification["spec"]), effective_config)

    # 서로 다른 후보는 signal·q·원장 행을 절대 공유하지 않는다. 다만 첫 부모
    # replay는 같은 (symbol, date)의 불변 quote/feature 배열을 읽으므로 둘 이상일
    # 때 한 batch로 묶으면 디스크 read와 worker 생성만 한 번이면 된다.
    prepared: dict[str, refinement.PreparedRefinement] = {}
    parents: dict[str, refinement.BootstrapParent] = {}
    unbranched = {hypothesis_id: item for hypothesis_id, item in pending.items()
                  if not item[1].branch_quantiles}
    if len(unbranched) > 1:
        prepared = {
            hypothesis_id: refinement.prepare(
                spec, symbols=symbols, dates=dates, root=Path(root), config=effective_config)
            for hypothesis_id, (spec, effective_config) in unbranched.items()
        }
        batch_key = refinement.bootstrap_batch_cache_key(
            prepared, symbols=symbols, dates=dates, root=Path(root))
        parents = refinement.run_bootstrap_batch(
            prepared, symbols=symbols, dates=dates,
            output=output / "bootstrap_parent_batches" / batch_key, root=Path(root))
    for hypothesis_id, (spec, effective_config) in pending.items():
        if effective_config.branch_quantiles:
            units[hypothesis_id] = refinement.run_q_branches(
                spec, symbols=symbols, dates=dates,
                output=output / "units" / hypothesis_id, root=Path(root), config=effective_config,
                runner=runner)
        else:
            units[hypothesis_id] = refinement.run(
                spec, symbols=symbols, dates=dates,
                output=output / "units" / hypothesis_id, root=Path(root), config=effective_config,
                prepared=prepared.get(hypothesis_id), initial_parent=parents.get(hypothesis_id),
                runner=runner)

    state = "ENTRY_REFINEMENT_COMPLETE" if units else "NO_IMPLEMENTED_SPECIFICATION"
    artifact = create_artifact("entry_refinement_set", {"state": state,
        "dates": list(dates), "symbols": list(symbols), "units": units},
        parents={"executable_specification_set": specifications["artifact_id"],
                 "implementation_set": implementations["artifact_id"],
                 "evidence_package": evidence["artifact_id"]})
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
