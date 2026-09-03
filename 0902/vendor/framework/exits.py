"""청산 규칙. 고정 평가 창 대신 조건부로 나가는 축.

진입은 그대로 두고 청산만 바꾸므로 **모든 청산 후보의 decision 수가 같다.** 진입 축과
달리 여기서는 분모 왜곡이 원리적으로 생기지 않는다 — 지표를 전 단계에 하나로 통일할
이유가 없다는 근거가 이것이다.

고정 horizon("무조건 더 일찍 판다")은 조건을 보지 않아 같은 이유로 진 거래와 아닌
거래를 구분하지 못한다. 그래서 규칙은 조건부여야 한다.

## 축이 넷이고 서로를 대신하지 못한다

  stop / take   손익 레벨. "얼마 잃었나 / 벌었나"
  trail         경로 의존. 고점에서 얼마나 반락했나 — 같은 net 이라도 고점이 다르면 답이 다르다
  state         시장 상태. "샀던 이유가 아직 유효한가". 손익을 보지 않는다

## 사전계산

각 수준에 처음 닿은 틱과 그때의 net 을 원장에 컬럼으로 남긴다. 후보마다 가격 경로를
다시 읽으면 격자 채점이 원장 재생성 비용이 된다 (후보 65개 × 종목-일 50개).
`trail` 은 경로 의존이라 쌍마다 따로 계산해야 한다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import FEE_BPS, PRIMARY_HORIZON_SECONDS, sha256_json
from .outcome import net_path


SCHEMA = "framework_exit_rule.v1"

STOP_LEVELS_BPS: tuple[float, ...] = (-80.0, -60.0, -40.0, -30.0, -20.0)
TAKE_LEVELS_BPS: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
TRAIL_SPECS: tuple[tuple[float, float], ...] = (
    (10.0, 5.0), (10.0, 10.0), (20.0, 5.0), (20.0, 10.0), (30.0, 10.0), (30.0, 15.0),
)
# 진입 근거가 사라졌다고 볼 상태들. 손익이 아니라 시장을 본다.
STATE_EXITS: tuple[dict[str, Any], ...] = (
    {"feature": "signed_aggr_flow_20", "operator": "<=", "threshold": -0.30},   # 매도 압력 재증가
    {"feature": "queue_imbalance_best", "operator": "<=", "threshold": 0.40},   # 매수 보충 소멸
    {"feature": "bid_queue_depletion_5s_bps", "operator": ">=", "threshold": 5.0},
)

REASON_STOP = "stop"
REASON_TAKE = "take_profit"
REASON_TRAIL = "trail"
REASON_STATE = "state"
REASON_CAP = "horizon_cap"


def rule(stop_net_bps: float | None = None, take_net_bps: float | None = None,
         *, trail: tuple[float, float] | None = None,
         state_exit: Sequence[Mapping[str, Any]] = (),
         cap_seconds: int = PRIMARY_HORIZON_SECONDS) -> dict[str, Any]:
    """청산 규칙 하나를 표현식으로."""
    return {
        "schema": SCHEMA,
        "stop_net_bps": None if stop_net_bps is None else float(stop_net_bps),
        "take_net_bps": None if take_net_bps is None else float(take_net_bps),
        "trail_start_bps": None if trail is None else float(trail[0]),
        "trail_giveback_bps": None if trail is None else float(trail[1]),
        "state_exit": [dict(c) for c in state_exit],
        "cap_seconds": int(cap_seconds),
        "decision_time": "first_tick_at_or_after_entry_meeting_a_level",
        "uses_current_or_past_only": True,
    }


def rule_id(expression: Mapping[str, Any]) -> str:
    """같은 규칙이면 같은 id."""
    parts = []
    if expression.get("stop_net_bps") is not None:
        parts.append(f"stop{float(expression['stop_net_bps']):g}")
    if expression.get("take_net_bps") is not None:
        parts.append(f"take{float(expression['take_net_bps']):g}")
    if expression.get("trail_start_bps") is not None:
        parts.append(f"trail{float(expression['trail_start_bps']):g}"
                     f"-{float(expression['trail_giveback_bps']):g}")
    if expression.get("state_exit"):
        parts.append(f"state{len(expression['state_exit'])}")
    return "exit|" + ("+".join(parts) or "none")


def candidates(stop_levels: Sequence[float] = STOP_LEVELS_BPS,
               take_levels: Sequence[float] = TAKE_LEVELS_BPS,
               trails: Sequence[tuple[float, float]] = TRAIL_SPECS) -> list[dict[str, Any]]:
    """시험할 규칙 격자. 손절만·익절만·트레일만을 조합과 따로 두어, 어느 쪽이
    일했는지 읽을 수 있게 한다.

    격자를 넓히면 그만큼 발굴 구간 최댓값이 낙관적이 된다. 넓힐 때는 이유를 남긴다.
    """
    out = [rule(stop_net_bps=s) for s in stop_levels]
    out += [rule(take_net_bps=t) for t in take_levels]
    out += [rule(trail=t) for t in trails]
    for stop in stop_levels:
        out += [rule(stop_net_bps=stop, take_net_bps=t) for t in take_levels]
        out += [rule(stop_net_bps=stop, trail=t) for t in trails]
    return out


def level_stems() -> list[tuple[str, float, str]]:
    """선언된 모든 수준을 `(컬럼 접두, 수준, 종류)` 로."""
    out = [(f"xstop{level:g}", float(level), "stop") for level in STOP_LEVELS_BPS]
    out += [(f"xtake{level:g}", float(level), "take") for level in TAKE_LEVELS_BPS]
    return out


def trail_stem(start: float, giveback: float) -> str:
    return f"xtrail{float(start):g}_{float(giveback):g}"


def crossing_columns() -> list[str]:
    """원장에 들어갈 crossing 컬럼 이름 전부. 원장 스키마가 이것을 따른다."""
    stems = [s for s, _l, _k in level_stems()] + [trail_stem(a, b) for a, b in TRAIL_SPECS]
    return ["cap:tick", "cap:net"] + [f"{s}:{k}" for s in stems for k in ("tick", "net")]


def _trail_mask(path: np.ndarray, start: float | None, giveback: float | None) -> np.ndarray | None:
    """이익이 `start` 에 닿은 뒤 고점에서 `giveback` 이상 반락한 틱.

    경로 의존이다 — 같은 net 이라도 그때까지의 고점이 다르면 답이 다르다.
    """
    if start is None or giveback is None:
        return None
    running = np.maximum.accumulate(np.where(np.isfinite(path), path, -np.inf))
    return np.isfinite(path) & (running >= float(start)) & (path <= running - float(giveback))


def state_exit_mask(features: Mapping[str, np.ndarray], start: int, end: int,
                    conditions: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    """진입 근거가 사라진 틱. 조건 중 하나라도 걸리면 참."""
    if not conditions:
        return None
    fired = np.zeros(max(int(end) - int(start) + 1, 0), dtype=bool)
    for condition in conditions:
        values = features.get(str(condition["feature"]))
        if values is None:
            continue
        window = np.asarray(values, dtype=float)[int(start): int(end) + 1]
        threshold = float(condition["threshold"])
        usable = np.isfinite(window)
        fired |= usable & ((window <= threshold) if str(condition["operator"]) == "<="
                           else (window >= threshold))
    return fired


def first_crossings(bid1: np.ndarray, entry_tick: int, cap_tick: int, entry_price: float,
                    *, fee_bps: float = FEE_BPS) -> dict[str, float]:
    """각 수준에 처음 닿은 틱과 그때의 net. 원장을 쓸 때 체결마다 한 번 계산한다.

    `<stem>:tick` 이 NaN 이면 평가 창 안에 그 수준에 닿지 않은 것이고, 그때는
    `cap:net`(평가 창 끝 net)으로 떨어진다.
    """
    stems = [s for s, _l, _k in level_stems()] + [trail_stem(a, b) for a, b in TRAIL_SPECS]
    out: dict[str, float] = {}
    if not np.isfinite(entry_price) or entry_price <= 0.0 or cap_tick <= entry_tick:
        out["cap:tick"] = out["cap:net"] = float("nan")
        for stem in stems:
            out[f"{stem}:tick"] = out[f"{stem}:net"] = float("nan")
        return out

    path = net_path(bid1, float(entry_price), int(entry_tick), int(cap_tick), fee_bps)
    out["cap:tick"] = float(cap_tick)
    out["cap:net"] = float(path[-1])

    def record(stem: str, hit: np.ndarray | None) -> None:
        where = np.flatnonzero(hit) if hit is not None else np.empty(0, dtype=int)
        if len(where):
            offset = int(where[0])
            out[f"{stem}:tick"] = float(int(entry_tick) + offset)
            out[f"{stem}:net"] = float(path[offset])
        else:
            out[f"{stem}:tick"] = out[f"{stem}:net"] = float("nan")

    for stem, level, kind in level_stems():
        record(stem, np.isfinite(path) & ((path <= level) if kind == "stop" else (path >= level)))
    for start, giveback in TRAIL_SPECS:
        record(trail_stem(start, giveback), _trail_mask(path, start, giveback))
    return out


def realised_net(frame: pd.DataFrame, expression: Mapping[str, Any]) -> np.ndarray:
    """규칙 하나가 각 체결에서 실현한 net(bps). 원장 컬럼만 읽는다.

    **먼저 닿은 수준이 이긴다.** 나중 틱의 정보로 되돌아가 고르지 않는다.
    어느 것도 닿지 않으면 평가 창 끝 시장가(`cap:net`).
    """
    stems: list[str] = []
    for level, kind in ((expression.get("stop_net_bps"), "stop"),
                        (expression.get("take_net_bps"), "take")):
        if level is not None:
            stems.append(f"x{kind}{float(level):g}")
    if expression.get("trail_start_bps") is not None:
        stems.append(trail_stem(expression["trail_start_bps"], expression["trail_giveback_bps"]))

    cap = pd.to_numeric(frame["cap:net"], errors="coerce").to_numpy(dtype=float)
    if not stems:
        return cap
    ticks = np.vstack([pd.to_numeric(frame[f"{s}:tick"], errors="coerce").to_numpy(dtype=float)
                       for s in stems])
    nets = np.vstack([pd.to_numeric(frame[f"{s}:net"], errors="coerce").to_numpy(dtype=float)
                      for s in stems])
    hit = np.any(np.isfinite(ticks), axis=0)
    winner = np.argmin(np.where(np.isfinite(ticks), ticks, np.inf), axis=0)
    chosen = nets[winner, np.arange(nets.shape[1])]
    return np.where(hit, chosen, cap)


def apply(bid1: np.ndarray, entry_tick: int, cap_tick: int, entry_price: float,
          expression: Mapping[str, Any], *, fee_bps: float = FEE_BPS,
          features: Mapping[str, np.ndarray] | None = None) -> tuple[int, float, str]:
    """규칙을 한 체결에 실제로 적용한다. `(청산 틱, net, 사유)`.

    진입 틱 자체부터 훑는다 — 진입 시점에 이미 수준을 넘긴 포지션은 평가 창 끝까지 들고
    가지 않고 즉시 나간다.

    `state_exit` 은 사전계산할 수 없으므로(feature 배열이 필요) 여기서만 쓴다.
    """
    if not np.isfinite(entry_price) or entry_price <= 0.0 or cap_tick <= entry_tick:
        return int(cap_tick), float("nan"), REASON_CAP
    path = net_path(bid1, float(entry_price), int(entry_tick), int(cap_tick), fee_bps)
    stop, take = expression.get("stop_net_bps"), expression.get("take_net_bps")
    hits = {
        REASON_STOP: np.isfinite(path) & (path <= float(stop)) if stop is not None else None,
        REASON_TAKE: np.isfinite(path) & (path >= float(take)) if take is not None else None,
        REASON_TRAIL: _trail_mask(path, expression.get("trail_start_bps"),
                                  expression.get("trail_giveback_bps")),
        REASON_STATE: (state_exit_mask(features, entry_tick, cap_tick, expression["state_exit"])
                       if features is not None and expression.get("state_exit") else None),
    }
    fired = np.zeros(len(path), dtype=bool)
    for mask in hits.values():
        if mask is not None:
            fired |= mask
    where = np.flatnonzero(fired)
    if not len(where):
        return int(cap_tick), float(path[-1]), REASON_CAP
    offset = int(where[0])
    reason = next((name for name, mask in hits.items() if mask is not None and mask[offset]),
                  REASON_CAP)
    return int(entry_tick) + offset, float(path[offset]), reason
