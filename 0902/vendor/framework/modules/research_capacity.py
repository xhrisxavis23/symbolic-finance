"""동시에 실행되는 연구 replay 수를 Sandbox 전체에서 제한한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..config import now_utc, research_run_slots, write_json


SCHEMA_VERSION = "research_capacity_reservation.v1"
SLOT_DIR = ".research_capacity_slots"
RESULT_FILE = "research_capacity_reservation.json"


def reserve(*, output: Path) -> dict[str, Any]:
    """비싼 replay 이전에 한 run slot을 원자적으로 잡는다.

    자리가 없으면 계산을 시작하지 않고 caller가 ``CAPACITY_WAITING`` artifact를 남긴다.
    종료 artifact가 있고 더 최신의 active progress가 없거나 owner PID가 죽은 slot만
    다음 run이 회수한다.
    """
    output = Path(output).resolve()
    limit = research_run_slots()
    slots_root = _runs_root(output) / SLOT_DIR
    slots_root.mkdir(parents=True, exist_ok=True)
    runs_root = _runs_root(output).resolve()
    active_external_outputs = {
        candidate for candidate in _active_framework_outputs()
        if _runs_root(candidate).resolve() == runs_root and candidate != output
    }
    if len(active_external_outputs) >= limit:
        result = {
            "schema": SCHEMA_VERSION,
            "state": "WAITING",
            "output": str(output),
            "slot_limit": limit,
            "active_external_run_count": len(active_external_outputs),
            "selection_input": False,
            "note": "기존 framework replay가 전역 병렬도 한도를 사용 중이다. 이 run은 resume으로 이어야 한다.",
        }
        write_json(output / RESULT_FILE, result)
        return result
    for index in range(limit):
        path = slots_root / f"slot-{index:02d}.json"
        outcome = _reserve_one(path, index, output)
        if outcome["state"] in {"ACQUIRED", "OWNED_BY_OUTPUT"}:
            result = {
                "schema": SCHEMA_VERSION,
                "state": "RESERVED",
                "output": str(output),
                "slot": index,
                "slot_limit": limit,
                "active_external_run_count": len(active_external_outputs),
                "selection_input": False,
                "note": "전역 replay 병렬도 제한용 운영 slot이며 Search·PnL 선택 입력이 아니다.",
            }
            write_json(output / RESULT_FILE, result)
            return result
    result = {
        "schema": SCHEMA_VERSION,
        "state": "WAITING",
        "output": str(output),
        "slot_limit": limit,
        "active_external_run_count": len(active_external_outputs),
        "selection_input": False,
        "note": "다른 active 연구 replay가 slot을 사용 중이다. 이 run은 resume으로 이어야 한다.",
    }
    write_json(output / RESULT_FILE, result)
    return result


def _reserve_one(path: Path, slot: int, output: Path) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA_VERSION,
        "state": "ACTIVE",
        "slot": slot,
        "created_at": now_utc(),
        "owner_pid": os.getpid(),
        "output": str(output),
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
            return _reserve_one(path, slot, output)
        return {"state": "OCCUPIED"}


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


def _same_output(existing: Mapping[str, Any] | None, output: Path) -> bool:
    owner_raw = str((existing or {}).get("output") or "")
    return bool(owner_raw) and Path(owner_raw).resolve() == Path(output).resolve()


def _can_reclaim(existing: Mapping[str, Any] | None) -> bool:
    if existing is None:
        return False
    owner_raw = str(existing.get("output") or "")
    owner = Path(owner_raw) if owner_raw else None
    # `run()`은 동기 함수다. 같은 process가 다음 run의 slot을 요청했다면 앞 run은
    # 이미 예외로 caller에게 돌아온 것이다. progress만 IN_PROGRESS로 남겨 slot을
    # 영구 점유시키지 말고 그 실패 run의 slot을 다음 순서 run이 회수하게 한다.
    try:
        same_process = int(existing.get("owner_pid")) == os.getpid()
    except (TypeError, ValueError):
        same_process = False
    if same_process:
        return True
    if owner is not None and _has_active_progress(owner):
        return False
    if owner is not None and any((owner / name).is_file()
                                 for name in ("research_run.json", "pre_validation_run.json")):
        return True
    return not _pid_is_alive(existing.get("owner_pid"))


def _has_active_progress(output: Path) -> bool:
    for name in ("research_progress.json", "pre_validation_progress.json"):
        progress = _read(output / name)
        if progress is not None and progress.get("state") == "IN_PROGRESS" \
                and _pid_is_alive(progress.get("owner_pid")):
            return True
    return False


def _pid_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _active_framework_outputs() -> set[Path]:
    """slot 생성 전부터 살아 있던 framework 상위 process의 output도 센다."""
    outputs: set[Path] = set()
    proc_root = Path("/proc")
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return outputs
    for entry in entries:
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        arguments = _command_line(pid)
        if not _is_framework_command(arguments) or _is_framework_parent(pid):
            continue
        output = _command_output(pid, arguments)
        if output is not None:
            outputs.add(output)
    return outputs


def _command_line(pid: int) -> tuple[str, ...]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ()
    return tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def _is_framework_command(arguments: tuple[str, ...]) -> bool:
    return any(arguments[index:index + 2] == ("-m", "framework")
               for index in range(max(0, len(arguments) - 1)))


def _is_framework_parent(pid: int) -> bool:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        parent_pid = int(raw.rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return False
    return _is_framework_command(_command_line(parent_pid))


def _command_output(pid: int, arguments: tuple[str, ...]) -> Path | None:
    try:
        output = Path(arguments[arguments.index("--output") + 1])
    except (ValueError, IndexError):
        return None
    if output.is_absolute():
        return output.resolve()
    try:
        cwd = Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return None
    return (cwd / output).resolve()


def _runs_root(output: Path) -> Path:
    for path in (output, *output.parents):
        if path.name == "runs":
            return path
    return output.parent
