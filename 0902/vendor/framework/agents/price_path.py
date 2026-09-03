"""Hypothesis Agent가 발굴 수익구간의 가격 흐름을 읽는 작은 MCP tool.

이 tool은 Feature Profile에 실제로 들어간 anchor만 연다. 그래서 Agent가 임의의
미래 구간을 찾아 고르는 대신, 이미 Profile이 고른 PROFIT/LOSS 사례를 대조해서 본다.
반환하는 미래 최고 ASK는 가격 기회 oracle이며 체결 증거가 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__:
    from .. import data as tickdata, profit
    from ..config import PROFIT_CACHE, TICK_ROOT, read_json
else:  # Codex가 MCP subprocess를 절대 경로 script로 실행할 때
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from framework import data as tickdata, profit
    from framework.config import PROFIT_CACHE, TICK_ROOT, read_json


ACCESS_SCHEMA = "discovery_price_path_access.v1"
TOOL_SCHEMA = "discovery_price_path_tool.v1"


def access_manifest(package: Mapping[str, Any], *, profit_root: Path = PROFIT_CACHE,
                    tick_root: Path = TICK_ROOT) -> dict[str, Any]:
    """Evidence Package의 Feature Profile anchor만 허용하는 읽기 권한을 만든다."""
    source = package.get("path_observation") or {}
    anchors = source.get("anchors") or []
    required = {"anchor_id", "cohort", "cluster_id", "symbol", "date", "tick"}
    execution = _execution_by_anchor(package)
    cleaned = []
    for row in anchors:
        if not isinstance(row, Mapping) or not required <= set(row):
            continue
        row = {
            "anchor_id": str(row["anchor_id"]), "cohort": str(row["cohort"]),
            "cluster_id": str(row["cluster_id"]), "symbol": str(row["symbol"]).zfill(6),
            "date": str(row["date"]), "tick": int(row["tick"]),
            "entry_queue_status": row.get("entry_queue_status"),
            "entry_fill_tick": _int_or_none(row.get("entry_fill_tick")),
            "entry_wait_ticks": _int_or_none(row.get("entry_wait_ticks")),
            "entry_wait_seconds": _float_or_none(row.get("entry_wait_seconds")),
            "entry_queue_fraction": _float_or_none(row.get("entry_queue_fraction")),
            "entry_oracle_ask_tick": _int_or_none(row.get("entry_oracle_ask_tick")),
            "entry_fill_before_oracle_ask": _bool_or_none(
                row.get("entry_fill_before_oracle_ask")),
        }
        if str(row["anchor_id"]) in execution:
            row["canonical_execution"] = execution[str(row["anchor_id"])]
        cleaned.append(row)
    if not cleaned:
        raise ValueError("가격 경로 tool에 열어 줄 Feature Profile anchor가 없다")
    horizon = str(source.get("horizon_key") or (package.get("config") or {}).get("horizon_key") or "")
    if not horizon:
        raise ValueError("가격 경로 tool의 horizon_key가 없다")
    raw_root = package.get("discovery_profit_root") or profit_root
    # Agent MCP는 임시 작업 폴더에서 실행된다. Profile이 넘긴 상대 경로를 그대로
    # manifest에 넣으면 그 폴더에서 cache를 찾게 되므로, access를 만들 때 고정한다.
    source_profit_root = Path(str(raw_root)).expanduser().resolve() if raw_root else None
    # 현재 BID1 진입·미래 ASK1 관측 계약과 같은 cache가 있을 때만 cache로 범위를 좁힌다.
    # 반대 가격 방향의 과거 cache는 읽지 않는다. cache가 없거나 계약이 다르면, 이미
    # Profile이 고른 anchor의 원본 quote 구간만 무저장으로 읽는다.
    available = (_cached_anchors(cleaned, horizon=horizon, profit_root=source_profit_root)
                 if source_profit_root is not None and source_profit_root.is_dir() else None)
    if available is not None:
        cleaned = available
    if not cleaned:
        raise FileNotFoundError(
            "연결한 discovery profit cache에 열 수 있는 Feature Profile anchor가 없다")
    return {
        "schema": ACCESS_SCHEMA,
        "horizon_key": horizon,
        "anchors": sorted(cleaned, key=lambda r: (r["cluster_id"], r["symbol"], r["date"],
                                                     r["tick"], r["anchor_id"])),
        "profit_root": str(source_profit_root) if source_profit_root is not None else None,
        "tick_root": str(Path(tick_root)),
        "path_source": ("CURRENT_DIRECTION_CACHE" if available is not None
                        else "RAW_TICK_ANCHOR_WINDOW"),
    }


def _execution_by_anchor(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Execution-anchor replay가 있을 때만 Profile anchor에 실제 fixed-exit 결과를 붙인다."""
    replay = package.get("canonical_execution_anchor_replay") or {}
    value = replay.get("outcomes_path")
    if not value:
        return {}
    path = _existing_path(value)
    if path is None:
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - execution tool 부재가 oracle path 접근을 막지 않는다.
        return {}
    required = {"anchor_id", "execution_label"}
    if not required <= set(frame):
        return {}
    fields = ("execution_label", "ledger_status", "fill_tick", "exit_tick", "entry_price",
              "net_bps", "gross_bps", "exit_reason", "holding_seconds", "diagnostic_cohort")
    return {
        str(row.anchor_id): {
            key: _number_or_none(getattr(row, key, None)) if key in {
                "fill_tick", "exit_tick", "entry_price", "net_bps", "gross_bps", "holding_seconds"}
            else (str(getattr(row, key)) if getattr(row, key, None) is not None
                  and not pd.isna(getattr(row, key)) else None)
            for key in fields if key in frame.columns
        }
        for row in frame.itertuples(index=False)
    }


def _existing_path(value: Any) -> Path | None:
    candidate = Path(str(value))
    choices = [candidate] if candidate.is_absolute() else [
        Path.cwd() / candidate,
        Path(__file__).resolve().parents[2] / candidate,
    ]
    return next((path for path in choices if path.is_file()), None)


def _cached_anchors(anchors: Sequence[Mapping[str, Any]], *, horizon: str,
                    profit_root: Path) -> list[dict[str, Any]] | None:
    """실제 cache가 있을 때만 해당 label에 존재하는 Profile anchor를 남긴다.

    `None`은 아직 materialize된 partition이 전혀 없는 호환 root를 뜻한다. 이 경우는
    호출자가 대표 cache를 나중에 붙이는 기존 단위 테스트·도구 계약을 유지한다.
    """
    partitions = Path(profit_root) / "partitions"
    if not partitions.is_dir():
        return None
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in anchors:
        key = (str(row["cluster_id"]), str(row["symbol"]).zfill(6), str(row["date"]))
        grouped.setdefault(key, []).append(row)
    available: list[dict[str, Any]] = []
    for (cluster, symbol, date), rows in grouped.items():
        label_path = partitions / cluster / symbol / date / "path_labels.parquet"
        if not label_path.exists():
            continue
        try:
            labels = profit.path_labels(cluster, symbol, date, horizon=horizon,
                                        root=Path(profit_root))
        except profit.CacheError:
            # 과거 cache가 반대 가격 방향이면 current discovery 관측에 쓰지 않는다.
            return None
        complete_column = f"complete__{horizon}"
        if complete_column not in labels:
            continue
        completed = {int(value) for value in labels.loc[labels[complete_column], "tick_idx"]}
        available.extend(dict(row) for row in rows if int(row["tick"]) in completed)
    return available


def write_access_manifest(path: Path, access: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(access, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_access_manifest(path: Path) -> dict[str, Any]:
    value = read_json(Path(path))
    if value.get("schema") != ACCESS_SCHEMA:
        raise ValueError("가격 경로 tool 접근권 형식이 다르다")
    if not value.get("anchors"):
        raise ValueError("가격 경로 tool 접근권이 비었다")
    return value


def list_price_paths(access: Mapping[str, Any], *, cohort: str | None = None,
                     execution_label: str | None = None, diagnostic_cohort: str | None = None,
                     cluster: str | None = None,
                     limit: int = 6, offset: int = 0) -> dict[str, Any]:
    """Agent가 실제로 열 수 있는 anchor를 Profile/실행 label별로 고른다.

    ``execution_label``은 원 Profile oracle cohort를 바꾸지 않는다. canonical 실행
    state Evidence처럼 BACKGROUND anchor 중 실제 queue replay 결과를 볼 때만 쓰는
    보조 filter다.
    """
    if limit < 1 or limit > 20:
        raise ValueError("limit은 1~20이어야 한다")
    if offset < 0:
        raise ValueError("offset은 0 이상이어야 한다")
    rows = list(access.get("anchors") or [])
    if cohort is not None:
        rows = [row for row in rows if row["cohort"] == str(cohort)]
    if execution_label is not None:
        rows = [row for row in rows
                if str((row.get("canonical_execution") or {}).get("execution_label"))
                == str(execution_label)]
    if diagnostic_cohort is not None:
        rows = [row for row in rows
                if str((row.get("canonical_execution") or {}).get("diagnostic_cohort"))
                == str(diagnostic_cohort)]
    if cluster is not None:
        rows = [row for row in rows if row["cluster_id"] == str(cluster)]
    total = len(rows)
    rows = rows[offset:offset + limit]
    return {
        "schema": TOOL_SCHEMA,
        "discovery_only": True,
        "horizon_key": access["horizon_key"],
        "available": len(rows), "total": total, "offset": offset,
        "paths": [{
            "path_id": path_id(row, str(access["horizon_key"])),
            "cohort": row["cohort"], "cluster_id": row["cluster_id"],
            "symbol": row["symbol"], "date": row["date"], "anchor_tick": row["tick"],
            "entry_queue_status": row.get("entry_queue_status"),
            **({"execution_label": row["canonical_execution"].get("execution_label")}
               if row.get("canonical_execution") is not None else {}),
            **({"diagnostic_cohort": row["canonical_execution"].get("diagnostic_cohort")}
               if row.get("canonical_execution") is not None else {}),
            **({"canonical_execution": row["canonical_execution"]}
               if row.get("canonical_execution") is not None else {}),
        } for row in rows],
    }


def get_price_path(access: Mapping[str, Any], *, cluster: str, symbol: str, date: str,
                   anchor_tick: int, horizon: str | None = None, pre_seconds: float = 10.0,
                   max_points: int = 160) -> dict[str, Any]:
    """허용된 한 anchor의 BID1/ASK1 흐름과 oracle 요약을 돌려준다."""
    key = str(horizon or access["horizon_key"])
    if key != str(access["horizon_key"]):
        raise ValueError(f"이 호출에서는 {access['horizon_key']} 경로만 열 수 있다")
    if not (0.0 <= float(pre_seconds) <= 60.0):
        raise ValueError("pre_seconds는 0~60초여야 한다")
    if not (8 <= int(max_points) <= 400):
        raise ValueError("max_points는 8~400이어야 한다")

    row = _allowed_anchor(access, cluster, symbol, date, anchor_tick)
    arrays, _source = tickdata.load(row["symbol"], row["date"], Path(access["tick_root"]))
    bid = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    time_s = np.asarray(arrays["time_s"], dtype=float)
    start_tick = int(row["tick"])
    cached = access.get("path_source", "CURRENT_DIRECTION_CACHE") == "CURRENT_DIRECTION_CACHE"
    label = None
    if cached:
        root = access.get("profit_root")
        if root is None:
            raise ValueError("가격 경로 cache root가 없다")
        labels = profit.path_labels(row["cluster_id"], row["symbol"], row["date"], horizon=key,
                                    root=Path(str(root)))
        selected = labels[labels.tick_idx == start_tick]
        if len(selected) != 1:
            raise ValueError("수익 경로 cache에 anchor tick이 없다")
        label = selected.iloc[0]
        if not bool(label[f"complete__{key}"]):
            raise ValueError("완결된 수익 경로가 아니다")
        end_tick = start_tick + int(label[f"window_end_delta_ticks__{key}"])
        oracle_tick = start_tick + int(label[f"best_ask_delta_ticks__{key}"])
    else:
        end_tick = _raw_window_end(time_s, start_tick, key)
        future_ask = ask[start_tick + 1:end_tick + 1]
        oracle_tick = start_tick + 1 + int(np.argmax(future_ask))
    if end_tick >= len(time_s):
        raise ValueError("원본 tick과 수익 경로의 길이가 다르다")
    before_tick = int(np.searchsorted(time_s, time_s[start_tick] - float(pre_seconds), side="left"))
    indices = _sample_indices(before_tick, end_tick, int(max_points),
                              required=(start_tick, oracle_tick, end_tick))
    points = [{
        "tick": int(idx),
        "elapsed_ms": round(float((time_s[idx] - time_s[start_tick]) * 1000.0), 3),
        "bid_1": _float_or_none(bid[idx]), "ask_1": _float_or_none(ask[idx]),
    } for idx in indices]
    return {
        "schema": TOOL_SCHEMA,
        "path_id": path_id(row, key),
        "discovery_only": True,
        "execution_proof": False,
        "path_source": str(access.get("path_source", "CURRENT_DIRECTION_CACHE")),
        "cohort": row["cohort"], "cluster_id": row["cluster_id"],
        "symbol": row["symbol"], "date": row["date"], "horizon_key": key,
        "anchor": {"tick": start_tick, "bid_1": _float_or_none(bid[start_tick]),
                   "ask_1": _float_or_none(ask[start_tick])},
        "price_opportunity": {
            "entry_price": "BID1 at anchor", "exit_price": "future best ASK1",
            "best_ask_tick": oracle_tick, "best_ask_1": _float_or_none(ask[oracle_tick]),
            "best_ask_gross_bps": _raw_bps(bid[start_tick], ask[oracle_tick]),
            "window_end_tick": end_tick,
            "fixed_ask_gross_bps": _raw_bps(bid[start_tick], ask[end_tick]),
        },
        "entry_queue": {
            "status": row.get("entry_queue_status"),
            "fill_tick": row.get("entry_fill_tick"),
            "wait_ticks": row.get("entry_wait_ticks"),
            "wait_seconds": row.get("entry_wait_seconds"),
            "queue_fraction_ahead": row.get("entry_queue_fraction"),
            "fill_before_oracle_ask": row.get("entry_fill_before_oracle_ask"),
        },
        **({"canonical_execution": {
                **dict(row["canonical_execution"]),
                "discovery_diagnostic_only": True,
            }} if row.get("canonical_execution") is not None else {}),
        "points": points,
    }


def get_pre_anchor_microstructure(access: Mapping[str, Any], *, cluster: str, symbol: str,
                                  date: str, anchor_tick: int, horizon: str | None = None,
                                  pre_seconds: float = 30.0,
                                  max_points: int = 160) -> dict[str, Any]:
    """허용 anchor 직전의 원시 호가·체결 상태만 읽는다.

    이 조회에는 anchor 뒤 tick을 절대로 넣지 않는다. Feature Profile이 골라 둔 사례에서
    "어떤 사전 상태가 있었는가"를 살피는 도구일 뿐, 미래 가격 결과를 다시 찾는 도구가
    아니다. 체결 방향 배열이 원본에 없으면 0으로 바꾸지 않고 ``None``으로 남긴다.
    """
    key = str(horizon or access["horizon_key"])
    if key != str(access["horizon_key"]):
        raise ValueError(f"이 호출에서는 {access['horizon_key']} 경로만 열 수 있다")
    if not (0.0 <= float(pre_seconds) <= 60.0):
        raise ValueError("pre_seconds는 0~60초여야 한다")
    if not (8 <= int(max_points) <= 400):
        raise ValueError("max_points는 8~400이어야 한다")

    row = _allowed_anchor(access, cluster, symbol, date, anchor_tick)
    arrays, _source = tickdata.load(row["symbol"], row["date"], Path(access["tick_root"]))
    bid = np.asarray(arrays["bid_price"], dtype=float)
    ask = np.asarray(arrays["ask_price"], dtype=float)
    bid_qty = np.asarray(arrays["bid_qty"], dtype=float)
    ask_qty = np.asarray(arrays["ask_qty"], dtype=float)
    time_s = np.asarray(arrays["time_s"], dtype=float)
    start_tick = int(row["tick"])
    if start_tick < 0 or start_tick >= len(time_s):
        raise ValueError("원본 tick에 anchor가 없다")
    before_tick = int(np.searchsorted(
        time_s, time_s[start_tick] - float(pre_seconds), side="left"))
    indices = _sample_indices(before_tick, start_tick, int(max_points), required=(start_tick,))
    buy = arrays.get("buy_volume")
    sell = arrays.get("sell_volume")
    has_trade_direction = buy is not None and sell is not None
    if has_trade_direction:
        buy = np.asarray(buy, dtype=float)
        sell = np.asarray(sell, dtype=float)

    def point(idx: int) -> dict[str, Any]:
        value = {
            "tick": int(idx),
            "elapsed_ms": round(float((time_s[idx] - time_s[start_tick]) * 1000.0), 3),
            "bid_1": _float_or_none(bid[idx, 0]),
            "ask_1": _float_or_none(ask[idx, 0]),
            "bid_qty_1": _float_or_none(bid_qty[idx, 0]),
            "ask_qty_1": _float_or_none(ask_qty[idx, 0]),
            "bid_depth_10": _float_or_none(np.sum(bid_qty[idx])),
            "ask_depth_10": _float_or_none(np.sum(ask_qty[idx])),
        }
        value["buy_volume"] = _float_or_none(buy[idx]) if has_trade_direction else None
        value["sell_volume"] = _float_or_none(sell[idx]) if has_trade_direction else None
        return value

    available_fields = ["bid_1", "ask_1", "bid_qty_1", "ask_qty_1",
                        "bid_depth_10", "ask_depth_10"]
    if has_trade_direction:
        available_fields.extend(("buy_volume", "sell_volume"))
    return {
        "schema": TOOL_SCHEMA,
        "path_id": path_id(row, key),
        "discovery_only": True,
        "execution_proof": False,
        "observation_boundary": "anchor tick 이하의 raw microstructure만 반환; post-anchor data 없음",
        "cohort": row["cohort"], "cluster_id": row["cluster_id"],
        "symbol": row["symbol"], "date": row["date"], "horizon_key": key,
        "anchor_tick": start_tick,
        "available_fields": available_fields,
        "points": [point(idx) for idx in indices],
    }


def _raw_window_end(time_s: np.ndarray, start_tick: int, horizon: str) -> int:
    """저장된 horizon만 사용해, Profile anchor의 원본 quote 창 끝을 읽는다."""
    if start_tick < 0 or start_tick >= len(time_s):
        raise ValueError("원본 tick에 anchor가 없다")
    prefix, value = str(horizon)[:1], str(horizon)[1:]
    if not value.isdigit() or prefix not in {"T", "S"}:
        raise ValueError(f"모르는 가격 경로 horizon: {horizon}")
    if prefix == "T":
        end = start_tick + int(value)
        complete = end < len(time_s)
    else:
        deadline = float(time_s[start_tick]) + float(int(value))
        end = int(np.searchsorted(time_s, deadline, side="right") - 1)
        complete = deadline <= float(time_s[-1]) and end > start_tick
    if not complete:
        raise ValueError("완결된 수익 경로가 아니다")
    return int(end)


def _raw_bps(entry_bid: float, exit_ask: float) -> float | None:
    if not np.isfinite(entry_bid) or not np.isfinite(exit_ask) or entry_bid <= 0.0:
        return None
    return _float_or_none((float(exit_ask) / float(entry_bid) - 1.0) * 1.0e4)


def representative_observations(access: Mapping[str, Any]) -> dict[str, Any]:
    """각 cohort의 허용된 한 경로를 실제로 열어 Agent의 관측 입력으로 고정한다.

    이것은 가설을 고르거나 future outcome을 평가하지 않는다. Agent가 빈 tool 호출을
    생략해도, Profile에 이미 포함된 PROFIT/LOSS 예시의 BID1·ASK1 흐름을 보지 않은 채
    이야기를 만들 수 없게 하는 최소 관측 단계다. 같은 access manifest는 Agent에도
    계속 붙여 두므로 추가 경로는 필요할 때 직접 조회할 수 있다.
    """
    paths, observations = [], []
    for cohort in ("PROFIT", "LOSS"):
        available = list_price_paths(access, cohort=cohort, limit=1).get("paths") or []
        if not available:
            raise ValueError(f"{cohort} 가격 경로가 없다")
        selected = available[0]
        path = get_price_path(
            access, cluster=str(selected["cluster_id"]), symbol=str(selected["symbol"]),
            date=str(selected["date"]), anchor_tick=int(selected["anchor_tick"]),
            max_points=80)
        paths.append(path)
        anchor, opportunity, queue = path["anchor"], path["price_opportunity"], path["entry_queue"]
        observations.append({
            "path_id": path["path_id"], "cohort": path["cohort"],
            "observations": [
                f"anchor tick {anchor['tick']}에서 BID1={anchor['bid_1']}, ASK1={anchor['ask_1']}",
                f"미래 최고 ASK1={opportunity['best_ask_1']}는 tick {opportunity['best_ask_tick']}에 있다 (discovery oracle)",
                f"entry queue status={queue['status']}, wait_ticks={queue['wait_ticks']}, fill_before_oracle_ask={queue['fill_before_oracle_ask']}",
            ],
        })
    return {"selection": "cohort별 허용 anchor 정렬순 첫 경로", "paths": paths,
            "path_observations": observations}


def path_id(row: Mapping[str, Any], horizon: str) -> str:
    return f"{row['anchor_id']}:{horizon}"


def _allowed_anchor(access: Mapping[str, Any], cluster: str, symbol: str, date: str,
                    anchor_tick: int) -> dict[str, Any]:
    target = (str(cluster), str(symbol).zfill(6), str(date), int(anchor_tick))
    for row in access.get("anchors") or []:
        key = (str(row["cluster_id"]), str(row["symbol"]).zfill(6), str(row["date"]),
               int(row["tick"]))
        if key == target:
            return dict(row)
    raise ValueError("Feature Profile 발굴 anchor가 아닌 경로는 열 수 없다")


def _sample_indices(start: int, end: int, maximum: int, *, required: Sequence[int]) -> list[int]:
    all_indices = np.arange(start, end + 1, dtype=int)
    if len(all_indices) > maximum:
        all_indices = np.linspace(start, end, maximum, dtype=int)
    return sorted(set(int(v) for v in all_indices) | {int(v) for v in required})


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _number_or_none(value: Any) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _bool_or_none(value: Any) -> bool | None:
    return bool(value) if isinstance(value, (bool, np.bool_)) else None


TOOLS = [
    {
        "name": "list_price_paths",
        "description": "발굴 Feature Profile anchor의 PROFIT/LOSS/BACKGROUND 가격 경로와 canonical 실행 diagnostic cohort 후보를 찾는다.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "cohort": {"type": "string", "enum": ["PROFIT", "LOSS", "BACKGROUND"]},
                "execution_label": {"type": "string"},
                "diagnostic_cohort": {"type": "string"},
                "cluster": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "offset": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_price_path",
        "description": "하나의 허용된 발굴 anchor에서 BID1/ASK1 가격 흐름과 oracle 가격 기회를 읽는다. 체결 증거는 아니다.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "cluster": {"type": "string"}, "symbol": {"type": "string"},
                "date": {"type": "string"}, "anchor_tick": {"type": "integer", "minimum": 0},
                "horizon": {"type": "string"},
                "pre_seconds": {"type": "number", "minimum": 0, "maximum": 60},
                "max_points": {"type": "integer", "minimum": 8, "maximum": 400},
            },
            "required": ["cluster", "symbol", "date", "anchor_tick"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_pre_anchor_microstructure",
        "description": "하나의 허용된 발굴 anchor 직전 raw 호가·호가잔량·방향별 체결량만 읽는다. anchor 뒤 가격과 미래 결과는 반환하지 않는다.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "cluster": {"type": "string"}, "symbol": {"type": "string"},
                "date": {"type": "string"}, "anchor_tick": {"type": "integer", "minimum": 0},
                "horizon": {"type": "string"},
                "pre_seconds": {"type": "number", "minimum": 0, "maximum": 60},
                "max_points": {"type": "integer", "minimum": 8, "maximum": 400},
            },
            "required": ["cluster", "symbol", "date", "anchor_tick"],
            "additionalProperties": False,
        },
    },
]


def serve(access: Mapping[str, Any], *, call_log: Path | None = None) -> None:
    """외부 패키지 없이 MCP stdio의 필요한 세 tool만 제공한다."""
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
                          "serverInfo": {"name": "discovery-price-path", "version": "1.0"},
                          "instructions": (
                              "These tools are read-only. They expose only selected discovery "
                              "Feature Profile anchors. Future best ASK values are oracle price "
                              "opportunities, never execution, validation, or OOS proof.")}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                name, arguments = params.get("name"), params.get("arguments") or {}
                if name == "list_price_paths":
                    result_value = list_price_paths(access, **arguments)
                elif name == "get_price_path":
                    result_value = get_price_path(access, **arguments)
                elif name == "get_pre_anchor_microstructure":
                    result_value = get_pre_anchor_microstructure(access, **arguments)
                else:
                    raise ValueError(f"모르는 tool: {name}")
                _record_call(call_log, str(name), arguments)
                result = {"content": [{"type": "text", "text": json.dumps(
                    result_value, ensure_ascii=False)}], "structuredContent": result_value}
            else:
                continue
            if "id" in request:
                _write({"jsonrpc": "2.0", "id": request["id"], "result": result})
        except Exception as exc:  # MCP는 모델에 읽을 수 있는 오류를 돌려준다.
            if isinstance(request, Mapping) and "id" in request:
                _write({"jsonrpc": "2.0", "id": request["id"],
                        "error": {"code": -32000, "message": str(exc)}})


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
