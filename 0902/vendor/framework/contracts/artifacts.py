"""단계 산출물을 위치가 아니라 내용과 부모 산출물로 식별한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import clean, now_utc, read_json, sha256_json, write_json


SCHEMA = "framework_artifact.v1"


class ArtifactError(ValueError):
    """산출물 계약이 깨졌을 때 낸다."""


def _identity(kind: str, parents: Mapping[str, str], payload: Mapping[str, Any]) -> str:
    return sha256_json({"schema": SCHEMA, "kind": kind, "parents": dict(sorted(parents.items())),
                        "payload": clean(payload)})[:20]


def create_artifact(kind: str, payload: Mapping[str, Any], *,
                    parents: Mapping[str, str] | None = None) -> dict[str, Any]:
    """JSON 산출물 하나를 만든다. 생성 시각과 경로는 식별값에 넣지 않는다."""
    clean_payload = clean(dict(payload))
    clean_parents = {str(key): str(value) for key, value in sorted((parents or {}).items())}
    return {
        "schema": SCHEMA,
        "kind": str(kind),
        "artifact_id": _identity(str(kind), clean_parents, clean_payload),
        "created_at": now_utc(),
        "parents": clean_parents,
        "payload": clean_payload,
    }


def _validate(value: Mapping[str, Any], *, kind: str | None = None) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise ArtifactError(f"알 수 없는 artifact schema: {value.get('schema')!r}")
    if kind is not None and value.get("kind") != kind:
        raise ArtifactError(f"{kind} artifact 가 필요한데 {value.get('kind')!r} 이다")
    payload, parents = value.get("payload"), value.get("parents")
    if not isinstance(payload, Mapping) or not isinstance(parents, Mapping):
        raise ArtifactError("artifact payload 또는 parents 모양이 맞지 않는다")
    expected = _identity(str(value.get("kind")), {str(k): str(v) for k, v in parents.items()}, payload)
    if value.get("artifact_id") != expected:
        raise ArtifactError("artifact_id 가 내용과 맞지 않는다")
    return dict(value)


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """계약을 확인한 뒤 원자적으로 쓴다."""
    checked = _validate(artifact)
    write_json(Path(path), checked)
    return checked


def read_artifact(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    return _validate(read_json(Path(path)), kind=kind)


def artifact_reference(artifact: Mapping[str, Any]) -> dict[str, str]:
    checked = _validate(artifact)
    return {"kind": str(checked["kind"]), "artifact_id": str(checked["artifact_id"])}
