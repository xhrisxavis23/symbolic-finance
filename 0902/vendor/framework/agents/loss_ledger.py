"""Discovery 실행 장부의 제한된 사례를 Hypothesis Agent에 여는 읽기 전용 MCP tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from ..config import read_json
    from ..modules.discovery_loss import SCHEMA_VERSION
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from framework.config import read_json
    from framework.modules.discovery_loss import SCHEMA_VERSION


ACCESS_SCHEMA = "discovery_loss_ledger_access.v1"
TOOL_SCHEMA = "discovery_loss_ledger_tool.v1"
SUPPORTED_CONTEXT_SCHEMAS = {"discovery_loss_context.v1", SCHEMA_VERSION}


def access_manifest(context: Mapping[str, Any]) -> dict[str, Any]:
    if context.get("schema") not in SUPPORTED_CONTEXT_SCHEMAS:
        raise ValueError("Discovery loss context 형식이 다르다")
    cases = list(context.get("case_samples") or [])
    if not cases:
        raise ValueError("Discovery loss context에 열어 줄 사례가 없다")
    return {
        "schema": ACCESS_SCHEMA,
        "source_hypothesis_id": str(context.get("source_hypothesis_id")),
        "summary": dict(context.get("summary") or {}),
        "cases": cases,
        "interpretation_boundary": str(context.get("interpretation_boundary") or ""),
    }


def write_access_manifest(path: Path, access: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(access, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_access_manifest(path: Path) -> dict[str, Any]:
    value = read_json(Path(path))
    if value.get("schema") != ACCESS_SCHEMA or not value.get("cases"):
        raise ValueError("Discovery loss ledger tool 접근권 형식이 다르다")
    return value


def list_execution_cases(access: Mapping[str, Any], *, cohort: str | None = None,
                         limit: int = 8, offset: int = 0) -> dict[str, Any]:
    if not 1 <= int(limit) <= 20 or int(offset) < 0:
        raise ValueError("limit은 1~20, offset은 0 이상이어야 한다")
    rows = list(access.get("cases") or [])
    if cohort is not None:
        rows = [row for row in rows if row.get("diagnostic_cohort") == str(cohort)]
    total = len(rows)
    rows = rows[int(offset):int(offset) + int(limit)]
    return {
        "schema": TOOL_SCHEMA, "discovery_only": True,
        "execution_evidence": "DISCOVERY_DIAGNOSTIC_ONLY",
        "source_hypothesis_id": access["source_hypothesis_id"],
        "available": len(rows), "total": total, "offset": int(offset),
        "cases": [{key: row.get(key) for key in ("case_id", "diagnostic_cohort", "symbol", "date",
                                                    "decision_index", "entry_tick", "fill_tick",
                                                    "entry_signal_active_at_fill",
                                                    "entry_signal_persisted_until_fill",
                                                    "entry_signal_active_share_until_fill",
                                                    "exit_reason")}
                  for row in rows],
    }


def get_execution_case(access: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    for row in access.get("cases") or []:
        if row.get("case_id") == str(case_id):
            return {
                "schema": TOOL_SCHEMA, "discovery_only": True,
                "execution_evidence": "DISCOVERY_DIAGNOSTIC_ONLY",
                "interpretation_boundary": access["interpretation_boundary"],
                "case": row,
            }
    raise ValueError("허용된 Discovery loss 사례가 아니다")


TOOLS = [
    {
        "name": "list_execution_cases",
        "description": "직전 가설의 Discovery 실행 장부에서 허용된 손실·회복 사례 목록을 읽는다. 새 가설의 Evidence나 Validation 결과는 아니다.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "inputSchema": {"type": "object", "properties": {
            "cohort": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "offset": {"type": "integer", "minimum": 0}}, "additionalProperties": False},
    },
    {
        "name": "get_execution_case",
        "description": "허용된 한 Discovery 실행 사례의 실제 청산 진단과 진입 시점 feature 값을 읽는다. 다음 연구 질문용이며 Evidence 증명은 아니다.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "inputSchema": {"type": "object", "properties": {"case_id": {"type": "string"}},
                        "required": ["case_id"], "additionalProperties": False},
    },
]


def serve(access: Mapping[str, Any], *, call_log: Path | None = None) -> None:
    for line in sys.stdin:
        request: Mapping[str, Any] | None = None
        try:
            request = json.loads(line)
            method = request.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2025-03-26"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "discovery-loss-ledger", "version": "1.0"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                name, arguments = params.get("name"), params.get("arguments") or {}
                if name == "list_execution_cases":
                    value = list_execution_cases(access, **arguments)
                elif name == "get_execution_case":
                    value = get_execution_case(access, **arguments)
                else:
                    raise ValueError(f"모르는 tool: {name}")
                _record_call(call_log, str(name), arguments)
                result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                          "structuredContent": value}
            else:
                continue
            if "id" in request:
                _write({"jsonrpc": "2.0", "id": request["id"], "result": result})
        except Exception as error:
            if isinstance(request, Mapping) and "id" in request:
                _write({"jsonrpc": "2.0", "id": request["id"],
                        "error": {"code": -32000, "message": str(error)}})


def _write(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _record_call(path: Path | None, name: str, arguments: Mapping[str, Any]) -> None:
    if path is None:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": str(name), "arguments": dict(arguments)},
                                ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-manifest", required=True, type=Path)
    parser.add_argument("--call-log", type=Path, default=None)
    args = parser.parse_args()
    serve(read_access_manifest(args.access_manifest), call_log=args.call_log)


if __name__ == "__main__":
    main()
