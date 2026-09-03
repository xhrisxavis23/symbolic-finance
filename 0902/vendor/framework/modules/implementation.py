"""Executable Specification을 실행 계약 템플릿으로 컴파일하는 어댑터."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .. import implementation as I
from ..config import read_json, sha256_json, write_json


def execution_identity(template: Mapping[str, Any]) -> dict[str, Any]:
    """가설 문장이 아니라 실제 진입·실행 계약으로 중복을 식별한다."""
    parameters = {
        str(name): {key: value for key, value in dict(value).items()
                    if key in {"comparator", "feature", "placeholder", "role", "status", "type", "value"}}
        for name, value in (template.get("parameter_interface") or {}).items()
        if isinstance(value, Mapping)
    }
    binding = {key: value for key, value in (template.get("execution_binding") or {}).items()
               if key != "exit_rule_note"}
    contract = {
        "schema": template.get("schema"),
        "side": template.get("side"),
        "entry_program": template.get("entry_program"),
        "execution_binding": binding,
        "parameter_interface": parameters,
        "parameter_search_policy": template.get("parameter_search_policy", {
            "selection_mode": "legacy-unknown",
            "primary_metric": "legacy-unknown",
            "profile_sha256": "legacy-unknown",
        }),
        # guard 선택 정책이 바뀌면 같은 base signal도 다른 최종 entry 연구가 된다.
        # 이전 template은 이 필드가 없으므로 legacy policy로만 서로 중복 처리한다.
        "entry_refinement_policy_version": template.get(
            "entry_refinement_policy_version", "legacy-unknown"),
    }
    return {"schema": "execution_identity.v1",
            "signature": sha256_json(contract)[:16],
            "contract": contract}


def equivalent_contracts(template: Mapping[str, Any], *, runs_root: Path,
                         exclude_path: Path) -> dict[str, Any]:
    """완료된 Search가 있는 같은 미해결 임계값 계약만 재실행하지 않는다."""
    identity = execution_identity(template)
    current = Path(exclude_path).resolve()
    matches: list[dict[str, Any]] = []
    for path in sorted(Path(runs_root).rglob("contract_template.json")):
        if path.resolve() == current:
            continue
        try:
            prior = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(prior, Mapping):
            continue
        if execution_identity(prior)["signature"] != identity["signature"]:
            continue
        if not _completed_search_for_prior(path, prior):
            continue
        matches.append({"path": str(path), "hypothesis_id": str(prior.get("hypothesis_id")),
                        "contract_sha256": str(prior.get("contract_sha256"))})
    return {**identity, "matches": matches}


def _completed_search_for_prior(template_path: Path, template: Mapping[str, Any]) -> bool:
    """계약 파일만 남긴 실패·진행 중 Implementation은 중복 실행이 아니다."""
    manifest_path = Path(template_path).with_name("implementation_manifest.json")
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return False
    if manifest.get("status") != I.READY_FOR_PARAMETER_SEARCH:
        return False
    run_root = Path(template_path).parents[3]
    search_path = run_root / "05_parameter_search" / "parameter_search_artifact.json"
    try:
        search_artifact = read_json(search_path)
    except (OSError, ValueError):
        return False
    units = ((search_artifact.get("payload") or {}).get("units") or {})
    return str(template.get("hypothesis_id") or "") in units


def run(spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source_specification.json"
    write_json(source, spec)
    result = I.run(source, output)
    return {"state": result["gate"]["status"], "template": result["template"],
            "validity": result["validity"], "fidelity": result["fidelity"],
            "gate": result["gate"]}
