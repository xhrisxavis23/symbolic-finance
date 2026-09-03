"""선택된 Feature Profile의 Evidence artifact cache.

Feature Profile은 종목-일 단위로 이미 저장돼 있다. 여기에는 그 Profile을 합쳐 만든
Evidence package만 따로 둔다. 따라서 cache가 맞으면 raw tick과 Evidence 집계를
다시 열지 않고 Hypothesis 단계로 바로 갈 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import read_json, sha256_file, sha256_json, write_json
from ..contracts.artifacts import read_artifact
from ..stages import evidence_stage
from .profile import ProfileSelection


SCHEMA = "feature_profile_evidence_cache.v2"
DIRECTORY = "EvidenceCache"
METADATA_NAME = "cache.json"


def _selection_input(selection: ProfileSelection | Mapping[str, Any]) -> dict[str, Any]:
    raw = (selection.as_input() if isinstance(selection, ProfileSelection) else dict(selection))
    return {
        "clusters": sorted(str(value) for value in raw.get("clusters") or ()),
        "symbols": sorted(str(value).zfill(6) for value in raw.get("symbols") or ()),
        "dates": sorted(str(value) for value in raw.get("dates") or ()),
        "store": str(raw.get("store")),
        "discovery_profit_root": (
            str(raw["discovery_profit_root"])
            if raw.get("discovery_profit_root") is not None else None),
        "profile_kind": str(raw.get("profile_kind")),
    }


def same_selection(left: ProfileSelection | Mapping[str, Any],
                   right: ProfileSelection | Mapping[str, Any]) -> bool:
    """순서만 다른 같은 종목·날짜 선택은 같은 입력이다."""
    return _selection_input(left) == _selection_input(right)


def _manifest_hash(selection: ProfileSelection) -> str | None:
    path = Path(selection.store) / "manifest.json"
    return sha256_file(path) if path.is_file() else None


def _key(selection: ProfileSelection) -> str:
    return sha256_json(_selection_input(selection))[:24]


def root(selection: ProfileSelection) -> Path:
    return Path(selection.store) / DIRECTORY


def artifact_path(selection: ProfileSelection) -> Path:
    return root(selection) / _key(selection) / evidence_stage.ARTIFACT_NAME


def find(selection: ProfileSelection) -> Path | None:
    """현재 Feature Profile manifest와 정확히 맞는 Evidence artifact만 돌려준다."""
    artifact = artifact_path(selection)
    metadata_path = artifact.parent / METADATA_NAME
    manifest_hash = _manifest_hash(selection)
    if manifest_hash is None or not artifact.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        stored = read_artifact(artifact, kind="evidence_package")
    except (OSError, ValueError):
        return None
    if metadata.get("schema") != SCHEMA:
        return None
    if metadata.get("profile_manifest_sha256") != manifest_hash:
        return None
    if not same_selection(metadata.get("selection") or {}, selection):
        return None
    if not same_selection((stored.get("payload") or {}).get("selection") or {}, selection):
        return None
    if metadata.get("evidence_artifact_id") != stored.get("artifact_id"):
        return None
    return artifact


def materialize(selection: ProfileSelection) -> dict[str, Any]:
    """선택 날짜의 Evidence package를 한 번 만들고 이후 run이 재사용하게 한다."""
    reused = find(selection)
    if reused is not None:
        artifact = read_artifact(reused, kind="evidence_package")
        return {"cache_state": "REUSED", "path": str(reused),
                "artifact_id": artifact["artifact_id"]}
    manifest_hash = _manifest_hash(selection)
    if manifest_hash is None:
        raise FileNotFoundError("Feature Profile manifest가 없다: " + str(Path(selection.store)))
    output = artifact_path(selection).parent
    artifact = evidence_stage.run(selection, output)
    path = output / evidence_stage.ARTIFACT_NAME
    write_json(output / METADATA_NAME, {
        "schema": SCHEMA,
        "selection": _selection_input(selection),
        "profile_manifest_sha256": manifest_hash,
        "evidence_artifact_id": artifact["artifact_id"],
    })
    return {"cache_state": "CREATED", "path": str(path),
            "artifact_id": artifact["artifact_id"]}
