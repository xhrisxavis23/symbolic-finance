"""Agent 호출의 공용 경계. Codex 호출 자체와 역할·프롬프트·응답 해시를 한 곳에 남긴다."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..config import sha256_json
from . import loss_ledger, price_path


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "high"
# 도구 없이 답하는 Agent 호출의 운영 한도. 도구를 실제로 읽는 생성 호출은 AgentConfig의
# 별도 예산을 쓴다.
AGENT_TIMEOUT_SECONDS = 480


class AgentCallError(RuntimeError):
    """외부 Agent 호출 실패를 다음 Stage가 보존할 수 있는 형태로 남긴다."""

    def __init__(self, message: str, *, code: str, model: str, effort: str,
                 timeout_seconds: int | None = None, stdout_tail: str = "",
                 stderr_tail: str = "", response_present: bool = False):
        super().__init__(message)
        self.code = code
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.response_present = response_present

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "model": self.model,
            "effort": self.effort,
            "timeout_seconds": self.timeout_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "response_present": self.response_present,
        }


def _tail(value: str | bytes | None, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def run_agent(prompt: str, *, model: str = MODEL, effort: str = REASONING_EFFORT,
              extra_args: Sequence[str] = (),
              timeout_seconds: int = AGENT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Agent 한 번 호출. 응답 파일만 읽어 로그와 JSON을 섞지 않는다."""
    binary = shutil.which("codex")
    if binary is None:
        raise AgentCallError("codex 실행 파일을 찾지 못했다", code="EXECUTABLE_UNAVAILABLE",
                             model=model, effort=effort)
    with TemporaryDirectory(prefix="hypothesis-generation-") as directory:
        response = Path(directory) / "response.json"
        try:
            completed = subprocess.run(
                [binary, "exec", "--ephemeral", "--sandbox", "read-only",
                 "--skip-git-repo-check", "--color", "never", "-C", directory,
                 "-m", model, "-c", f'model_reasoning_effort="{effort}"',
                 *extra_args,
                 "-o", str(response)],
                input=prompt, text=True, capture_output=True,
                timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            raise AgentCallError(
                f"Codex Agent 호출이 {timeout_seconds}초 안에 끝나지 않았다",
                code="TIMEOUT", model=model, effort=effort, timeout_seconds=timeout_seconds,
                stdout_tail=_tail(error.stdout), stderr_tail=_tail(error.stderr),
                response_present=response.exists(),
            ) from error
        except OSError as error:
            raise AgentCallError(
                f"Codex Agent 호출을 시작할 수 없다: {error}", code="EXECUTION_ERROR",
                model=model, effort=effort, response_present=response.exists(),
            ) from error
        if completed.returncode != 0:
            raise AgentCallError("Codex Agent 호출 실패", code="NONZERO_EXIT",
                                 model=model, effort=effort,
                                 stdout_tail=_tail(completed.stdout),
                                 stderr_tail=_tail(completed.stderr),
                                 response_present=response.exists())
        if not response.exists():
            raise AgentCallError("Codex Agent 응답 파일이 없다", code="MISSING_RESPONSE",
                                 model=model, effort=effort,
                                 stdout_tail=_tail(completed.stdout),
                                 stderr_tail=_tail(completed.stderr))
        try:
            return parse_payload(response.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError) as error:
            raise AgentCallError("Codex Agent 응답이 JSON 객체가 아니다",
                                 code="INVALID_RESPONSE", model=model, effort=effort,
                                 stdout_tail=_tail(completed.stdout),
                                 stderr_tail=_tail(completed.stderr),
                                 response_present=True) from error


def parse_payload(text: str) -> dict[str, Any]:
    """모델 출력에서 JSON 하나를 꺼낸다. 코드 펜스는 벗긴다."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```", 2)[1]
        body = body.split("\n", 1)[1] if "\n" in body else body
        body = body.rstrip("`").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"JSON 을 찾을 수 없다: {text[:200]!r}")
    return json.loads(body[start:end + 1])


class Runner(Protocol):
    def run(self, *, role: str, prompt: str, model: str, effort: str,
            tool_context: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AgentInvocation:
    role: str
    model: str
    effort: str
    prompt_sha256: str
    response_sha256: str
    tool_calls: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodexAgentRunner:
    """기존 Codex 호출기를 시스템 Agent 인터페이스로 감싼 기본 구현.

    `timeout_seconds` 는 도구 없는 호출의 한도다. 프롬프트가 커지고 effort 가 높으면
    기본값으로는 모자랄 수 있어 호출하는 쪽이 정한다.
    """

    def __init__(self, *, timeout_seconds: int | None = None,
                 tool_timeout_seconds: int | None = None):
        self.timeout_seconds = timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds

    def run(self, *, role: str, prompt: str, model: str, effort: str,
            tool_context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        self.last_tool_calls: tuple[dict[str, Any], ...] = ()
        if tool_context is None:
            options: dict[str, Any] = {"model": model, "effort": effort}
            if self.timeout_seconds is not None:
                options["timeout_seconds"] = int(self.timeout_seconds)
            return run_agent(prompt, **options)
        with TemporaryDirectory(prefix="hypothesis-research-tools-") as directory:
            accesses = _normalise_tool_context(tool_context)
            arguments: list[str] = []
            call_logs: dict[str, Path] = {}
            for name, access in accesses.items():
                access_path = Path(directory) / f"{name}_access.json"
                call_log = Path(directory) / f"{name}_calls.jsonl"
                if name == "price_path":
                    price_path.write_access_manifest(access_path, access)
                    script = Path(price_path.__file__).resolve()
                else:
                    loss_ledger.write_access_manifest(access_path, access)
                    script = Path(loss_ledger.__file__).resolve()
                call_logs[name] = call_log
                arguments.extend(_mcp_args(name, script, access_path, call_log))
            options: dict[str, Any] = {"model": model, "effort": effort,
                                       "extra_args": tuple(arguments)}
            budget = self.tool_timeout_seconds or self.timeout_seconds
            if budget is not None:
                options["timeout_seconds"] = int(budget)
            result = run_agent(prompt, **options)
            self.last_tool_calls = _read_tool_calls(call_logs)
            return result


def invoke(runner: Runner | Callable[[str], Mapping[str, Any]] | None, *, role: str,
           prompt: str, model: str, effort: str,
           tool_context: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], AgentInvocation]:
    active = runner or CodexAgentRunner()
    if hasattr(active, "run"):
        arguments = {"role": role, "prompt": prompt, "model": model, "effort": effort}
        if tool_context is not None and _accepts_tool_context(getattr(active, "run")):
            arguments["tool_context"] = tool_context
        response = getattr(active, "run")(**arguments)
    else:
        response = active(prompt)  # 기존 테스트용 단일 인자 callable도 받는다.
    if not isinstance(response, Mapping):
        raise TypeError(f"{role} Agent 응답이 JSON 객체가 아니다")
    value = dict(response)
    return value, AgentInvocation(
        role=role, model=model, effort=effort,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        response_sha256=sha256_json(value)[:16],
        tool_calls=tuple(dict(item) for item in getattr(active, "last_tool_calls", ())
                         if isinstance(item, Mapping)),
    )


def _accepts_tool_context(call: Callable[..., Any]) -> bool:
    """기존 mock Runner는 네 인자만 받는다. 실행 중 TypeError를 잡아 숨기지 않는다."""
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        # 서명을 못 읽는 callable은 정본 Runner 계약을 따른다고 보고 전달한다.
        return True
    return any(parameter.name == "tool_context" or
               parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _normalise_tool_context(context: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """기존 가격 경로 access manifest 호출도 새 복수 tool 형식으로 읽는다."""
    if context.get("schema") == price_path.ACCESS_SCHEMA:
        return {"price_path": context}
    out: dict[str, Mapping[str, Any]] = {}
    for name in ("price_path", "loss_ledger"):
        value = context.get(name)
        if isinstance(value, Mapping):
            out[name] = value
    if not out:
        raise ValueError("Agent tool context에 읽기 전용 access manifest가 없다")
    return out


def _mcp_args(name: str, script: Path, access_path: Path, call_log: Path) -> tuple[str, ...]:
    """이 호출에만 읽기 전용 연구 MCP를 붙인다. 사용자 전역 설정은 건드리지 않는다."""
    command = json.dumps(str(Path(__import__("sys").executable)))
    arguments = json.dumps([str(script), "--access-manifest", str(Path(access_path)),
                            "--call-log", str(Path(call_log))])
    return (
        "-c", f"mcp_servers.discovery_{name}.command={command}",
        "-c", f"mcp_servers.discovery_{name}.args={arguments}",
        "-c", f'mcp_servers.discovery_{name}.default_tools_approval_mode="writes"',
    )


def _read_tool_calls(call_logs: Mapping[str, Path]) -> tuple[dict[str, Any], ...]:
    """이번 Agent 호출이 실제 MCP를 사용했는지 artifact에 남긴다."""
    records: list[dict[str, Any]] = []
    for server, path in call_logs.items():
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                records.append({"server": str(server), **dict(value)})
    return tuple(records)
