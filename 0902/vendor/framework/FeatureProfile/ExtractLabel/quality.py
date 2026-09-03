"""품질 통과 구간 — BID1 진입 뒤 ASK1 기회가 지속되는 것만 남긴다.

`extract.py` 의 수익 구간은 문턱이 낮다. **창 안에서 한 번이라도** 수수료를 넘으면
수익으로 센다. 실측하면 그런 구간의 30% 는 팔 수 있는 순간이 딱 한 틱이다 — 그 한 틱을
미리 알아야 성립하는 이익이라 실행이 아니라 결과론이다.

여기서 조건 셋을 더 건다.

    · 이익이 `min_profit_bps` 이상일 것            (조금 남는 정도는 뺀다)
    · 그것이 `min_profit_streak_ticks` 틱 연속될 것  (한 순간 반짝이면 뺀다)
    · 그 전에 `mae_floor_bps` 밑으로 안 깨질 것      (가는 길이 험하면 뺀다)

**확정 틱(confirmation)** 은 조건을 만족한 첫 연속 구간의 마지막 틱이다. 거기까지
버텼으면 그 이익은 한 순간이 아니었다는 뜻이다.

이 라벨도 미래를 보고 붙인다. 발굴에서만 쓴다 — 진입 규칙으로 쓰면 자기 답을 보고
쓰는 것이 된다.

수익은 `BID1` 매수 지정가 게시가와 그 뒤 `ASK1` 매도 지정가의 가격 기회로 잰다.
따라서 이 라벨은 큐 체결을 보장하지 않는다. 실제 체결·청산은 canonical backtest가
별도로 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .extract import (HorizonSpec, STANDARD_HORIZONS, _time_text, _validate_horizons,
                      fill_short_false_gaps, find_contiguous_runs, njit)


QUALITY_REGION_SCHEMA = "maker_quality_profit_region.v2"

# 진입 틱마다의 판정. 왜 떨어졌는지가 남아야 표본을 다시 짤 수 있다.
NO_LABEL = 0        # 창이 잘렸거나 호가가 없다
ROBUST_PROFIT = 1   # 셋 다 통과
RISK_FAIL = 2       # 이익은 났는데 가는 길에 너무 깨졌다
TRANSIENT = 3       # 이익이 났는데 연속되지 않았다 (한 순간 반짝)
THIN = 4            # 이익이 나긴 하는데 문턱에 못 미친다
ADVERSE = 5         # 창 안 최선으로 팔아도 손해
OUTCOME_NAMES = {NO_LABEL: "NO_LABEL", ROBUST_PROFIT: "ROBUST_PROFIT",
                 RISK_FAIL: "RISK_FAIL", TRANSIENT: "TRANSIENT",
                 THIN: "THIN", ADVERSE: "ADVERSE"}

QUALITY_REGION_COLUMNS = (
    "horizon_key", "horizon_kind", "horizon_value",
    "start_tick", "end_tick", "length_ticks",
    "start_time", "start_time_us", "end_time_us",
    "entry_bid", "window_end_tick", "window_end_seconds",
    "profit_streak_start_tick", "profit_streak_start_hold_ticks",
    "profit_streak_start_seconds",
    "confirmation_tick", "confirmation_hold_ticks", "confirmation_hold_seconds",
    "confirmation_ask", "confirmation_net_bps", "preconfirmation_mae_net_bps",
    "oracle_ask_tick", "oracle_hold_ticks", "oracle_hold_seconds",
    "oracle_best_ask_net_bps", "fixed_net_bps",
    "raw_qualified_ticks_in_region", "gap_filled_ticks_in_region",
)


@dataclass(frozen=True)
class QualityContract:
    """얼린 문턱. 결과를 보고 바꾸면 그 조정이 결과를 만든 것이 된다."""

    fee_bps_rt: float = 23.0
    min_profit_bps: float = 10.0        # 이 정도는 남아야 이익으로 센다
    min_profit_streak_ticks: int = 10   # 그것이 이만큼 연속돼야 한다
    mae_floor_bps: float = -60.0        # 확정 전까지 이 밑으로 깨지면 뺀다
    min_region_ticks: int = 5           # 구간이 되려면 이만큼 이어져야 한다
    max_gap_ticks: int = 2              # 이 이하로 끊긴 곳은 이어 붙인다

    def __post_init__(self) -> None:
        if self.fee_bps_rt < 0:
            raise ValueError("fee_bps_rt 는 0 이상이어야 한다")
        if self.min_profit_bps < 0:
            raise ValueError("min_profit_bps 는 0 이상이어야 한다")
        if self.min_profit_streak_ticks < 1:
            raise ValueError("min_profit_streak_ticks 는 1 이상이어야 한다")
        if self.mae_floor_bps > self.min_profit_bps:
            raise ValueError("mae_floor_bps 가 min_profit_bps 보다 클 수 없다")
        if self.min_region_ticks < 1:
            raise ValueError("min_region_ticks 는 1 이상이어야 한다")
        if self.max_gap_ticks < 0:
            raise ValueError("max_gap_ticks 는 0 이상이어야 한다")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": QUALITY_REGION_SCHEMA,
            "fee_bps_rt": float(self.fee_bps_rt),
            "min_profit_bps": float(self.min_profit_bps),
            "min_profit_streak_ticks": int(self.min_profit_streak_ticks),
            "mae_floor_bps": float(self.mae_floor_bps),
            "min_region_ticks": int(self.min_region_ticks),
            "max_gap_ticks": int(self.max_gap_ticks),
            "profit_streak_rule": "net ASK 가 min_profit_bps 이상인 틱이 연속",
            "mae_rule": "진입 다음 틱부터 확정 틱까지의 최저 net ASK 평가액",
            "confirmation_rule": "조건을 만족한 첫 연속 구간의 마지막 틱",
        }


# ---- 계산 도구 ----------------------------------------------------------------
# "앞으로 k 틱 동안 ASK 가 한 번도 X 밑으로 안 내려가는 첫 지점" 을 찾아야 한다.
# 시작 틱마다 창을 다시 훑으면 O(n * 창길이) 라, 미리 훑어 두고 조회만 한다.

@njit(cache=True)
def _rolling_min_forward(values: np.ndarray, window: int) -> np.ndarray:
    """`values[j:j+window]` 의 최솟값을 시작점 `j` 로 색인해 돌려준다."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[:] = -np.inf
    if window < 1 or n < window:
        return out
    deque = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        while tail > head and values[deque[tail - 1]] >= values[i]:
            tail -= 1
        deque[tail] = i
        tail += 1
        left = i - window + 1
        while tail > head and deque[head] < left:
            head += 1
        if left >= 0:
            out[left] = values[deque[head]]
    return out


@njit(cache=True)
def _tree_size(n: int) -> int:
    size = 1
    while size < n:
        size *= 2
    return size


@njit(cache=True)
def _build_max_tree(values: np.ndarray) -> tuple[np.ndarray, int]:
    size = _tree_size(len(values))
    tree = np.empty(size * 2, dtype=np.float64)
    tree[:] = -np.inf
    for i in range(len(values)):
        tree[size + i] = values[i]
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[i * 2], tree[i * 2 + 1])
    return tree, size


@njit(cache=True)
def _build_min_tree(values: np.ndarray) -> tuple[np.ndarray, int]:
    size = _tree_size(len(values))
    tree = np.empty(size * 2, dtype=np.float64)
    tree[:] = np.inf
    for i in range(len(values)):
        tree[size + i] = values[i]
    for i in range(size - 1, 0, -1):
        tree[i] = min(tree[i * 2], tree[i * 2 + 1])
    return tree, size


@njit(cache=True)
def _first_ge(tree: np.ndarray, size: int, left: int, right: int,
              threshold: float) -> int:
    """`[left, right]` 안에서 값이 `threshold` 이상인 **첫** 위치. 없으면 -1."""
    if left > right or right < 0 or left >= size or tree[1] < threshold:
        return -1
    left = max(left, 0)
    right = min(right, size - 1)
    nodes = np.empty(128, dtype=np.int64)
    lo = np.empty(128, dtype=np.int64)
    hi = np.empty(128, dtype=np.int64)
    top = 0
    nodes[top] = 1
    lo[top] = 0
    hi[top] = size - 1
    top += 1
    while top:
        top -= 1
        node = nodes[top]
        node_lo = lo[top]
        node_hi = hi[top]
        if node_hi < left or node_lo > right or tree[node] < threshold:
            continue
        if node_lo == node_hi:
            return node_lo
        middle = (node_lo + node_hi) // 2
        # 오른쪽을 먼저 쌓아야 왼쪽이 먼저 나온다 — "첫" 위치를 찾는 것이므로.
        nodes[top] = node * 2 + 1
        lo[top] = middle + 1
        hi[top] = node_hi
        top += 1
        nodes[top] = node * 2
        lo[top] = node_lo
        hi[top] = middle
        top += 1
    return -1


@njit(cache=True)
def _range_min(tree: np.ndarray, size: int, left: int, right: int) -> float:
    if left > right:
        return np.nan
    left += size
    right += size
    result = np.inf
    while left <= right:
        if left % 2 == 1:
            result = min(result, tree[left])
            left += 1
        if right % 2 == 0:
            result = min(result, tree[right])
            right -= 1
        left //= 2
        right //= 2
    return result


# ---- 진입 틱마다의 판정 --------------------------------------------------------

@njit(cache=True)
def _quality_outcomes(bid, ask, complete, window_end_delta, best_gross,
                      fee_bps_rt, min_profit_bps, min_profit_streak_ticks,
                      mae_floor_bps):
    """창이 온전한 모든 진입 틱을 여섯 갈래로 나눈다."""
    n = len(bid)
    outcome = np.full(n, NO_LABEL, dtype=np.int8)
    streak_start = np.full(n, -1, dtype=np.int64)
    confirmation = np.full(n, -1, dtype=np.int64)
    mae_net = np.full(n, np.nan, dtype=np.float64)

    streak_floor = _rolling_min_forward(ask, min_profit_streak_ticks)
    max_tree, max_size = _build_max_tree(streak_floor)
    min_tree, min_size = _build_min_tree(ask)
    required_best_gross = fee_bps_rt + min_profit_bps

    for i in range(n):
        if not complete[i] or bid[i] <= 0.0 or not np.isfinite(best_gross[i]):
            continue
        best_net = best_gross[i] - fee_bps_rt
        if best_net < 0.0:
            outcome[i] = ADVERSE
            continue
        if best_net < min_profit_bps:
            outcome[i] = THIN
            continue

        window_end = i + int(window_end_delta[i])
        latest_start = window_end - min_profit_streak_ticks + 1
        if latest_start < i + 1:
            outcome[i] = TRANSIENT
            continue
        required_ask = bid[i] * (1.0 + required_best_gross / 1.0e4)
        start = _first_ge(max_tree, max_size, i + 1, latest_start, required_ask)
        if start < 0:
            outcome[i] = TRANSIENT
            continue

        confirm = start + min_profit_streak_ticks - 1
        low_ask = _range_min(min_tree, min_size, i + 1, confirm)
        low_net = (low_ask - bid[i]) / bid[i] * 1.0e4 - fee_bps_rt
        streak_start[i] = start
        confirmation[i] = confirm
        mae_net[i] = low_net
        outcome[i] = RISK_FAIL if low_net < mae_floor_bps else ROBUST_PROFIT
    return outcome, streak_start, confirmation, mae_net


def classify_outcomes(labels: pd.DataFrame, horizon_key: str,
                      contract: QualityContract = QualityContract()
                      ) -> dict[str, np.ndarray]:
    """진입 틱마다 왜 통과했고 왜 떨어졌는가. `ROBUST_PROFIT` 이 구간의 재료다."""
    outcome, streak, confirm, mae = _quality_outcomes(
        labels["bid_1"].to_numpy(dtype=np.float64),
        labels["ask_1"].to_numpy(dtype=np.float64),
        labels[f"complete__{horizon_key}"].to_numpy(dtype=np.bool_),
        labels[f"window_end_delta_ticks__{horizon_key}"].to_numpy(dtype=np.int64),
        labels[f"best_ask_gross_bps__{horizon_key}"].to_numpy(dtype=np.float64),
        float(contract.fee_bps_rt), float(contract.min_profit_bps),
        int(contract.min_profit_streak_ticks), float(contract.mae_floor_bps))
    return {"outcome": outcome, "streak_start": streak,
            "confirmation": confirm, "preconfirmation_mae_net_bps": mae}


# ---- 층 2: 품질 통과 구간 ------------------------------------------------------

def quality_regions(labels: pd.DataFrame,
                    horizons: Iterable[HorizonSpec] = STANDARD_HORIZONS,
                    contract: QualityContract = QualityContract()
                    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """층 0 라벨에서 품질 통과 구간을 만든다.

    `ROBUST_PROFIT` 인 진입 틱을 이어 붙여 구간으로 만드는 것은 `extract.regions()`
    와 같은 방식이다 — 짧은 끊김을 메우고 `min_region_ticks` 이상만 남긴다.
    """
    specs = _validate_horizons(horizons)
    bid = labels["bid_1"].to_numpy(dtype=np.float64)
    ask = labels["ask_1"].to_numpy(dtype=np.float64)
    packed = labels["recv_ts_kst"].to_numpy(dtype=np.int64)
    time_us = labels["time_us"].to_numpy(dtype=np.int64)
    fee = float(contract.fee_bps_rt)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for spec in specs:
        key = spec.key
        complete = labels[f"complete__{key}"].to_numpy(dtype=bool)
        end_delta = labels[f"window_end_delta_ticks__{key}"].to_numpy(dtype=np.int64)
        best_gross = labels[f"best_ask_gross_bps__{key}"].to_numpy(dtype=np.float64)
        graded = classify_outcomes(labels, key, contract)
        outcome = graded["outcome"]
        streak_start = graded["streak_start"]
        confirmation = graded["confirmation"]
        mae_net = graded["preconfirmation_mae_net_bps"]
        raw = outcome == ROBUST_PROFIT

        filled = fill_short_false_gaps(raw, int(contract.max_gap_ticks))
        all_starts, all_ends = find_contiguous_runs(filled)
        keep = (all_ends - all_starts) >= int(contract.min_region_ticks)
        starts, ends = all_starts[keep], all_ends[keep]

        summaries[key] = {
            "horizon": spec.as_dict(),
            "n_valid_ticks": int(complete.sum()),
            "n_best_net_ge_threshold": int((
                complete & np.isfinite(best_gross)
                & (best_gross - fee >= float(contract.min_profit_bps))).sum()),
            "n_quality_ticks_raw": int(raw.sum()),
            "n_quality_ticks_after_gap_fill": int(filled.sum()),
            "n_runs_after_gap_fill": int(len(all_starts)),
            "n_regions": int(len(starts)),
            "region_total_ticks": int((ends - starts).sum()),
            "outcomes": {OUTCOME_NAMES[code]: int((outcome == code).sum())
                         for code in sorted(OUTCOME_NAMES)},
        }

        for start, end in zip(starts, ends):
            s, e = int(start), int(end)
            # 끊김 메우기는 구간의 **첫** 틱을 바꾸지 않는다. 바뀌었다면 계산이 틀렸다.
            if not raw[s]:
                raise AssertionError(f"구간 시작이 통과 틱이 아니다: {key}/{s}")
            streak = int(streak_start[s])
            confirm = int(confirmation[s])
            window_end = s + int(end_delta[s])
            oracle_delta = int(labels.at[s, f"best_ask_delta_ticks__{key}"])
            oracle_ask = s + oracle_delta
            region_last = min(e - 1, len(labels) - 1)
            rows.append({
                "horizon_key": key, "horizon_kind": spec.kind,
                "horizon_value": int(spec.value),
                "start_tick": s, "end_tick": e, "length_ticks": e - s,
                "start_time": _time_text(int(packed[s])),
                "start_time_us": int(time_us[s]),
                "end_time_us": int(time_us[region_last]),
                "entry_bid": float(bid[s]),
                "window_end_tick": window_end,
                "window_end_seconds": float((time_us[window_end] - time_us[s]) / 1e6),
                "profit_streak_start_tick": streak,
                "profit_streak_start_hold_ticks": streak - s,
                "profit_streak_start_seconds": float((time_us[streak] - time_us[s]) / 1e6),
                "confirmation_tick": confirm,
                "confirmation_hold_ticks": confirm - s,
                "confirmation_hold_seconds": float((time_us[confirm] - time_us[s]) / 1e6),
                "confirmation_ask": float(ask[confirm]),
                "confirmation_net_bps": float((ask[confirm] - bid[s]) / bid[s] * 1e4 - fee),
                "preconfirmation_mae_net_bps": float(mae_net[s]),
                "oracle_ask_tick": oracle_ask,
                "oracle_hold_ticks": oracle_delta,
                "oracle_hold_seconds": float((time_us[oracle_ask] - time_us[s]) / 1e6),
                "oracle_best_ask_net_bps": float(best_gross[s] - fee),
                "fixed_net_bps": float(labels.at[s, f"fixed_ask_gross_bps__{key}"] - fee),
                "raw_qualified_ticks_in_region": int(raw[s:e].sum()),
                "gap_filled_ticks_in_region": int((~raw[s:e]).sum()),
            })
    return pd.DataFrame(rows, columns=list(QUALITY_REGION_COLUMNS)), summaries


# ---- clean_safe: 마지막 걸러내기 ------------------------------------------------

# `clean_safe_manifest.json` 의 계약과 같은 값.
CLEAN_SAFE = {
    "additional_mae_floor_bps": -10.0,      # 진입 뒤 ASK1 가격 기회가 이보다 덜 나빠질 것
    "confirmation_horizon_fraction_max": 0.5,  # 창의 절반 안에 확정될 것
    "fixed_net_bps_min": 10.0,              # 창 끝까지 들고 있어도 남을 것
}


def clean_safe(regions: pd.DataFrame,
               rule: dict[str, float] | None = None) -> pd.DataFrame:
    """품질 통과 구간에서 다시 한 번 거른다. 통과한 것만 돌려준다.

    진입 전에는 매도 주문을 낼 수 없으므로, 진입 뒤 첫 ASK1부터의 최저 net을 그대로
    `additional_mae_bps`로 쓴다.
    """
    rule = {**CLEAN_SAFE, **(rule or {})}
    if regions.empty:
        return regions.copy()
    frame = regions.copy()
    additional_mae = frame["preconfirmation_mae_net_bps"].to_numpy(dtype=np.float64)
    horizon_value = frame["horizon_value"].to_numpy(dtype=np.float64)
    fraction = np.where(
        frame["horizon_kind"].astype(str).eq("ticks"),
        frame["confirmation_hold_ticks"].to_numpy(dtype=np.float64) / horizon_value,
        frame["confirmation_hold_seconds"].to_numpy(dtype=np.float64) / horizon_value)

    selected = ((additional_mae >= rule["additional_mae_floor_bps"])
                & (fraction <= rule["confirmation_horizon_fraction_max"])
                & (frame["fixed_net_bps"].to_numpy(dtype=np.float64)
                   >= rule["fixed_net_bps_min"]))
    frame = frame.loc[selected].copy()
    frame["additional_mae_bps"] = additional_mae[selected]
    frame["confirmation_horizon_fraction"] = fraction[selected]
    frame["clean_safe_selected"] = True
    return frame.reset_index(drop=True)
