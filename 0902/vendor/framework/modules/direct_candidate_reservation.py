"""동시에 시작한 Direct Evidence run의 같은 entry replay를 한 번만 연다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import clean, now_utc, sha256_json, write_json


SCHEMA_VERSION = "direct_candidate_reservation.v1"
RESERVATION_DIR = ".direct_candidate_reservations"
RESULT_FILE = "direct_candidate_reservation.json"


def reserve(candidates: Sequence[Mapping[str, Any]], *, profile: Mapping[str, Any],
            output: Path) -> dict[str, Any]:
    """선택된 Direct candidate들의 Entry Refinement 권리를 원자적으로 잡는다.

    같은 저장 Feature Profile과 entry logic은 한 active run만 가질 수 있다. 이미 terminal
    artifact가 있는 이전 reservation이나 죽은 process reservation은 새 owner가 회수한다.
    q별 refinement도 같은 base entry의 재평가이므로 active base replay와는 동시에 열지 않는다.
    """
    output = Path(output).resolve()
    normalized = clean(dict(profile))
    requested = _requested(candidates, normalized)
    claims_root = _runs_root(output) / RESERVATION_DIR
    claims_root.mkdir(parents=True, exist_ok=True)
    acquired: list[Path] = []
    reused: list[Path] = []
    conflicts: list[dict[str, Any]] = []
    for item in requested:
        path = claims_root / f"{item['key']}.json"
        outcome = _reserve_one(path, item, output)
        if outcome["state"] == "ACQUIRED":
            acquired.append(path)
        elif outcome["state"] == "OWNED_BY_OUTPUT":
            reused.append(path)
        else:
            conflicts.append(outcome["conflict"])
    if conflicts:
        for path in acquired:
            _release_if_owned(path, output)
        result = {
            "schema": SCHEMA_VERSION,
            "state": "CONFLICT",
            "output": str(output),
            "requested": requested,
            "conflicts": conflicts,
            "selection_input": False,
            "note": "다른 active run의 같은 Direct entry replay를 기다린다.",
        }
    else:
        result = {
            "schema": SCHEMA_VERSION,
            "state": "RESERVED",
            "output": str(output),
            "requested": requested,
            "claim_paths": [str(path) for path in [*acquired, *reused]],
            "selection_input": False,
            "note": "동시 중복 replay 방지용 운영 예약이며 Search·PnL 선택 입력이 아니다.",
        }
    write_json(output / RESULT_FILE, result)
    return result


def _requested(candidates: Sequence[Mapping[str, Any]],
               profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        fingerprint = str(candidate.get("entry_logic_fingerprint") or "")
        if not fingerprint:
            raise ValueError("Direct candidate에 entry_logic_fingerprint가 없다")
        identity = {"profile": profile, "entry_logic_fingerprint": fingerprint}
        key = sha256_json(identity)[:24]
        values[key] = {
            "key": key,
            "hypothesis_id": str(candidate.get("hypothesis_id") or ""),
            "entry_logic_fingerprint": fingerprint,
            "profile": profile,
        }
    return [values[key] for key in sorted(values)]


def _reserve_one(path: Path, item: Mapping[str, Any], output: Path) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_VERSION,
        "state": "ACTIVE",
        "created_at": now_utc(),
        "owner_pid": os.getpid(),
        "output": str(output),
        **dict(item),
    }
    try:
        _write_exclusive(path, payload)
        return {"state": "ACQUIRED"}
    except FileExistsError:
        existing = _read(path)
        if _same_output(existing, output):
            return {"state": "OWNED_BY_OUTPUT"}
        if _can_reclaim(existing):
            path.unlink(missing_ok=True)
            return _reserve_one(path, item, output)
        return {"state": "CONFLICT", "conflict": {
            "key": str(item["key"]),
            "hypothesis_id": str(item["hypothesis_id"]),
            "entry_logic_fingerprint": str(item["entry_logic_fingerprint"]),
            "owner": existing or {"state": "RESERVING_OR_UNREADABLE"},
        }}


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _can_reclaim(existing: Mapping[str, Any] | None) -> bool:
    if existing is None:
        return False
    owner_raw = str(existing.get("output") or "")
    if owner_raw and (Path(owner_raw) / "research_run.json").is_file():
        return True
    pid = existing.get("owner_pid")
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return True
    except PermissionError:
        return False
    return False


def _release_if_owned(path: Path, output: Path) -> None:
    existing = _read(path)
    if _same_output(existing, output):
        path.unlink(missing_ok=True)


def _same_output(existing: Mapping[str, Any] | None, output: Path) -> bool:
    owner_raw = str((existing or {}).get("output") or "")
    return bool(owner_raw) and Path(owner_raw).resolve() == Path(output).resolve()


def _runs_root(output: Path) -> Path:
    for path in (output, *output.parents):
        if path.name == "runs":
            return path
    return output.parent
