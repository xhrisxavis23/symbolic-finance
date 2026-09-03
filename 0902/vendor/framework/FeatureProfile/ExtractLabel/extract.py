"""수익 구간 추출 — 틱마다의 미래 라벨(층 0)과 그 라벨에서 뽑은 구간(층 1).

## 무엇을 재는가

    진입   틱 t 의 BID1 (매수 지정가 게시가)
    청산   진입 뒤 창 안의 최고 ASK1
    gross  (최고_ASK1 / 진입_BID1 - 1) x 10^4
    net    gross - 왕복 수수료

**미래를 보고 붙인 라벨이다.** 발굴 구간에서 "무엇을 설명할 것인가" 를 정할 때만 쓴다.
진입 조건으로 쓰거나 OOS 증거로 내놓으면 그 구간이 검증이 아니게 된다.

## 층이 둘인 이유

층 0(`path_labels`)은 **수수료 전** gross 를 저장한다. 그래야 수수료율이나 구간 규칙을
바꿔도 원본 틱을 다시 읽지 않고 층 1만 다시 만들 수 있다.

층 1(`regions`)이 규칙을 적용한다.

    net > tradeable_edge_bps 인 틱을 "좋은 틱" 으로 표시
      -> 짧은 끊김(<= max_gap_ticks)은 메운다        (깜빡임 억제)
      -> 연속 길이 >= min_run_ticks 인 것만 구간으로

층 1의 판정 기준은 **창 안 최고 ASK1(oracle)** 이다. 매수·매도 주문이 실제로 모두
체결됐다는 뜻은 아니다. 발굴 라벨일 뿐이며, 실행은 canonical backtest가 따로 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:                                    # pragma: no cover
    def njit(*args, **kwargs):                         # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


PATH_SCHEMA = "maker_profit_path_cache.v2"
REGION_SCHEMA = "maker_profit_region_cache.v2"

REGION_COLUMNS = (
    "horizon_key", "horizon_kind", "horizon_value",
    "start_tick", "end_tick", "length_ticks",
    "start_time", "start_time_us", "end_time_us",
    "entry_bid", "region_end_ask", "region_end_net_bps",
    "window_end_tick", "window_end_seconds", "fixed_net_bps",
    "oracle_ask_tick", "oracle_hold_ticks", "oracle_hold_seconds",
    "oracle_best_ask_gross_bps", "oracle_best_ask_net_bps", "oracle_at_horizon_boundary",
    "first_profit_tick", "first_profit_hold_ticks", "first_profit_hold_seconds",
    "oracle_mae_gross_bps", "oracle_mfe_gross_bps",
)


# ---- 평가 창 ------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class HorizonSpec:
    """앞을 얼마나 볼 것인가. 틱 기준이거나 시계 기준이다.

    둘을 섞어 두는 이유는 같은 뜻이 아니기 때문이다 — 활동이 뜸한 종목에서 100틱은
    몇 분일 수 있다.
    """

    kind: str      # "ticks" | "seconds"
    value: int

    def __post_init__(self) -> None:
        if self.kind not in {"ticks", "seconds"}:
            raise ValueError(f"모르는 지평 종류: {self.kind}")
        if int(self.value) < 1:
            raise ValueError("지평 값은 1 이상이어야 한다")

    @classmethod
    def ticks(cls, value: int) -> "HorizonSpec":
        return cls("ticks", int(value))

    @classmethod
    def seconds(cls, value: int) -> "HorizonSpec":
        return cls("seconds", int(value))

    @property
    def key(self) -> str:
        return ("T" if self.kind == "ticks" else "S") + str(self.value)

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "value": self.value}


# 캐시가 담고 있는 일곱 개. `profit.horizons()` 와 같은 순서다.
STANDARD_HORIZONS = (
    HorizonSpec.ticks(100),
    HorizonSpec.seconds(30),
    HorizonSpec.seconds(60),
    HorizonSpec.seconds(90),
    HorizonSpec.seconds(120),
    HorizonSpec.seconds(180),
    HorizonSpec.seconds(240),
)


def _validate_horizons(horizons: Iterable[HorizonSpec]) -> tuple[HorizonSpec, ...]:
    result = tuple(horizons)
    if not result:
        raise ValueError("지평이 최소 하나는 있어야 한다")
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise ValueError(f"지평 키가 겹친다: {keys}")
    return result


def packed_time_to_us(values: np.ndarray) -> np.ndarray:
    """`HHMMSSuuuuuu` 정수를 자정부터의 마이크로초로.

    `data.packed_time_to_seconds()` 와 같은 변환인데 단위가 정수 마이크로초다.
    캐시의 `time_us` 열이 이 값이라 부동소수로 바꾸지 않는다.
    """
    packed = np.asarray(values, dtype=np.int64)
    hours = packed // 10_000_000_000
    minutes = (packed // 100_000_000) % 100
    seconds = (packed // 1_000_000) % 100
    micros = packed % 1_000_000
    return ((hours * 3600 + minutes * 60 + seconds) * 1_000_000 + micros).astype(np.int64)


# ---- 연속 구간 ----------------------------------------------------------------

def find_contiguous_runs(is_good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """참이 이어지는 구간의 (시작, 끝). `끝` 은 배타적이다."""
    if len(is_good) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    pad = np.concatenate([[False], is_good, [False]])
    transitions = np.diff(pad.astype(np.int8))
    return np.where(transitions == +1)[0], np.where(transitions == -1)[0]


def fill_short_false_gaps(mask: np.ndarray, max_gap_ticks: int) -> np.ndarray:
    """짧게 끊긴 곳을 메워 하나의 구간으로 잇는다 (깜빡임 억제).

    안쪽의 거짓 구간 길이가 `max_gap_ticks` 이하면 참으로 채운다. 배열 맨 앞과 맨 뒤의
    거짓은 채우지 않는다 — 그건 끊김이 아니라 아직 시작하지 않았거나 이미 끝난 것이다.

    `max_gap_ticks=5` 일 때::

        [T T T F T T T]        -> [T T T T T T T]        1틱 끊김, 메움
        [T T F F F F T T]      -> [T T T T T T T T]      4틱 끊김, 메움
        [T T F F F F F F T]    -> 그대로                  6틱 끊김, 안 메움
        [F F T T T F F]        -> 그대로                  앞뒤 끝은 그대로
    """
    if max_gap_ticks <= 0 or len(mask) == 0:
        return mask
    out = mask.copy()
    starts, ends = find_contiguous_runs(~mask)
    for start, end in zip(starts, ends):
        if start == 0 or end == len(mask):
            continue
        if (end - start) <= max_gap_ticks:
            out[start:end] = True
    return out


def fill_short_false_gaps_dual(mask: np.ndarray, time_us: np.ndarray, *,
                               max_gap_ticks: int, max_gap_seconds: float) -> np.ndarray:
    """틱 수와 실제 시간이 **둘 다** 한도 안일 때만 끊김을 잇는다.

    틱 조건만 걸면 호가가 뜸한 종목에서 2틱 끊김이 몇 분일 수 있다.
    """
    if len(mask) != len(time_us):
        raise ValueError("mask 와 time_us 의 길이가 다르다")
    if max_gap_ticks <= 0 or len(mask) == 0:
        return mask
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds 는 0 이상이어야 한다")
    out = mask.copy()
    starts, ends = find_contiguous_runs(~mask)
    for start, end in zip(starts, ends):
        if start == 0 or end == len(mask) or end - start > max_gap_ticks:
            continue
        if float(time_us[end] - time_us[start - 1]) / 1.0e6 <= max_gap_seconds:
            out[start:end] = True
    return out


# ---- 층 0: 틱마다의 미래 라벨 -------------------------------------------------

@njit(cache=True)
def _forward_best_details(bid: np.ndarray, ask: np.ndarray,
                          window_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 틱의 진입 뒤 창 안 최고 ASK1 과 그 시점까지의 거리.

    뒤에서 앞으로 훑으며 단조 감소 덱을 유지한다 — 창 끝이 틱마다 달라도 전체가
    O(n) 이다. 창마다 최댓값을 다시 구하면 O(n * 창길이) 가 된다.
    """
    n = len(bid)
    gross = np.empty(n, dtype=np.float64)
    best_delta = np.empty(n, dtype=np.int64)
    for i in range(n):
        gross[i] = np.nan
        best_delta[i] = -1
    if n <= 1:
        return gross, best_delta

    deque = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n - 2, -1, -1):
        j = i + 1
        # `<=` 라 값이 같으면 **이른** 쪽이 남는다. 같은 값이면 빨리 나가는 것이 낫다.
        while tail > head and ask[deque[tail - 1]] <= ask[j]:
            tail -= 1
        deque[tail] = j
        tail += 1

        end = int(window_end[i])
        while tail > head and deque[head] > end:
            head += 1
        if end >= j and tail > head and bid[i] > 0.0:
            best = int(deque[head])
            gross[i] = (ask[best] - bid[i]) / bid[i] * 1.0e4
            best_delta[i] = best - i
    return gross, best_delta


def _window_end_indices(time_us: np.ndarray,
                        horizon: HorizonSpec) -> tuple[np.ndarray, np.ndarray]:
    """각 틱의 창 끝 인덱스와 그 창이 온전한가.

    창이 장 끝을 넘으면 `complete=False` 이고 끝은 -1 이다. 잘린 창을 그대로 쓰면
    장 마감 직전 틱이 "짧은 창에서 수익" 으로 뽑힌다.
    """
    n = len(time_us)
    indices = np.arange(n, dtype=np.int64)
    if horizon.kind == "ticks":
        end = indices + horizon.value
        complete = end < n
    else:
        deadline = time_us + int(horizon.value) * 1_000_000
        end = np.searchsorted(time_us, deadline, side="right") - 1
        complete = (deadline <= time_us[-1]) & (end > indices)
    end = np.where(complete, end, -1).astype(np.int64)
    return end, complete.astype(bool)


def path_labels(arrays: Mapping[str, np.ndarray],
                horizons: Iterable[HorizonSpec] = STANDARD_HORIZONS) -> pd.DataFrame:
    """한 종목-일의 틱마다 미래 라벨. `data.load()` 가 준 배열을 그대로 받는다.

    **수수료를 빼지 않는다.** gross 로 저장해 두어야 수수료율을 바꿔도 층 1만 다시
    만들면 된다.

    열 이름과 dtype 은 캐시와 같다 — `profit.path_labels()` 가 읽는 그 모양이다.
    """
    specs = _validate_horizons(horizons)
    bid = np.asarray(arrays["bid_price"], dtype=np.float64)[:, 0]
    ask = np.asarray(arrays["ask_price"], dtype=np.float64)[:, 0]
    packed = np.asarray(arrays["local_time"], dtype=np.int64)
    time_us = packed_time_to_us(packed)
    if len(time_us) and np.any(np.diff(time_us) < 0):
        raise ValueError("틱이 시간 순으로 정렬돼 있지 않다")

    n = len(bid)
    out = pd.DataFrame({
        "tick_idx": np.arange(n, dtype=np.int32),
        "recv_ts_kst": packed,
        "time_us": time_us,
        "bid_1": bid.astype(np.float32),
        "ask_1": ask.astype(np.float32),
    })
    for spec in specs:
        key = spec.key
        end, complete = _window_end_indices(time_us, spec)
        gross, best_delta = _forward_best_details(bid, ask, end)

        fixed = np.full(n, np.nan, dtype=np.float64)
        valid = np.flatnonzero(complete)
        if len(valid):
            fixed[valid] = (ask[end[valid]] - bid[valid]) / bid[valid] * 1.0e4

        best_seconds = np.full(n, np.nan, dtype=np.float64)
        reached = np.flatnonzero(best_delta >= 1)
        if len(reached):
            exits = reached + best_delta[reached]
            best_seconds[reached] = (time_us[exits] - time_us[reached]) / 1.0e6

        out[f"complete__{key}"] = complete
        out[f"window_end_delta_ticks__{key}"] = np.where(
            complete, end - np.arange(n), -1).astype(np.int32)
        out[f"best_ask_gross_bps__{key}"] = gross.astype(np.float32)
        out[f"best_ask_delta_ticks__{key}"] = best_delta.astype(np.int32)
        out[f"best_ask_seconds__{key}"] = best_seconds.astype(np.float32)
        out[f"fixed_ask_gross_bps__{key}"] = fixed.astype(np.float32)
    return out


# ---- 층 1: 수익 구간 -----------------------------------------------------------

def _time_text(packed: int) -> str:
    value = int(packed)
    return (f"{value // 10_000_000_000:02d}:"
            f"{(value // 100_000_000) % 100:02d}:"
            f"{(value // 1_000_000) % 100:02d}")


def regions(labels: pd.DataFrame, horizons: Iterable[HorizonSpec] = STANDARD_HORIZONS, *,
            fee_bps_rt: float, tradeable_edge_bps: float,
            min_run_ticks: int, max_gap_ticks: int,
            max_gap_seconds: float | None = None
            ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """층 0 라벨에 수수료·문턱·연속 규칙을 적용해 구간을 만든다.

    구간이 하나도 안 나온 지평도 요약에 남긴다 — 빈 결과와 계산하지 않은 것은 다르다.
    """
    specs = _validate_horizons(horizons)
    bid = labels["bid_1"].to_numpy(dtype=np.float64)
    ask = labels["ask_1"].to_numpy(dtype=np.float64)
    packed = labels["recv_ts_kst"].to_numpy(dtype=np.int64)
    time_us = labels["time_us"].to_numpy(dtype=np.int64)
    n = len(labels)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for spec in specs:
        key = spec.key
        best_gross = labels[f"best_ask_gross_bps__{key}"].to_numpy(dtype=float)
        complete = labels[f"complete__{key}"].to_numpy(dtype=bool)
        best_net = best_gross - float(fee_bps_rt)

        good_raw = complete & np.isfinite(best_net) & (best_net > float(tradeable_edge_bps))
        good = (fill_short_false_gaps(good_raw, int(max_gap_ticks))
                if max_gap_seconds is None
                else fill_short_false_gaps_dual(
                    good_raw, time_us, max_gap_ticks=int(max_gap_ticks),
                    max_gap_seconds=float(max_gap_seconds)))
        starts, ends = find_contiguous_runs(good)
        long_enough = (ends - starts) >= int(min_run_ticks)

        summaries[key] = {
            "horizon": spec.as_dict(),
            "n_valid_ticks": int(complete.sum()),
            "n_good_ticks_raw": int(good_raw.sum()),
            "n_good_ticks_after_gap_fill": int(good.sum()),
            "n_runs": int(len(starts)),
            "n_long_runs": int(long_enough.sum()),
            "long_run_total_ticks": int((ends[long_enough] - starts[long_enough]).sum()),
            "max_gap_ticks": int(max_gap_ticks),
            "max_gap_seconds": None if max_gap_seconds is None else float(max_gap_seconds),
        }

        for start, end in zip(starts[long_enough], ends[long_enough]):
            s, e = int(start), int(end)
            best_delta = int(labels.at[s, f"best_ask_delta_ticks__{key}"])
            end_delta = int(labels.at[s, f"window_end_delta_ticks__{key}"])
            oracle_ask = s + best_delta
            window_end = s + end_delta
            if not (0 <= s < oracle_ask <= window_end < n):
                raise AssertionError(
                    f"층 0 라벨의 인덱스가 어긋난다 {key} tick={s}: "
                    f"oracle={oracle_ask} end={window_end} n={n}")

            gross_path = (ask[s + 1: window_end + 1] - bid[s]) / bid[s] * 1.0e4
            net_path = gross_path - float(fee_bps_rt)
            hit = np.flatnonzero(net_path > float(tradeable_edge_bps))
            first_offset = int(hit[0]) + 1 if len(hit) else None
            first_tick = s + first_offset if first_offset is not None else None
            region_last = min(e - 1, n - 1)

            rows.append({
                "horizon_key": key, "horizon_kind": spec.kind, "horizon_value": spec.value,
                "start_tick": s, "end_tick": e, "length_ticks": e - s,
                "start_time": _time_text(int(packed[s])),
                "start_time_us": int(time_us[s]), "end_time_us": int(time_us[region_last]),
                "entry_bid": float(bid[s]), "region_end_ask": float(ask[region_last]),
                "region_end_net_bps": float(
                    (ask[region_last] - bid[s]) / bid[s] * 1.0e4 - fee_bps_rt),
                "window_end_tick": window_end,
                "window_end_seconds": float((time_us[window_end] - time_us[s]) / 1.0e6),
                "fixed_net_bps": float(labels.at[s, f"fixed_ask_gross_bps__{key}"] - fee_bps_rt),
                "oracle_ask_tick": oracle_ask,
                "oracle_hold_ticks": best_delta,
                "oracle_hold_seconds": float(labels.at[s, f"best_ask_seconds__{key}"]),
                "oracle_best_ask_gross_bps": float(best_gross[s]),
                "oracle_best_ask_net_bps": float(best_net[s]),
                "oracle_at_horizon_boundary": oracle_ask == window_end,
                "first_profit_tick": first_tick,
                "first_profit_hold_ticks": first_offset,
                "first_profit_hold_seconds": (
                    float((time_us[first_tick] - time_us[s]) / 1.0e6)
                    if first_tick is not None else None),
                "oracle_mae_gross_bps": float(np.nanmin(gross_path)),
                "oracle_mfe_gross_bps": float(np.nanmax(gross_path)),
            })
    return pd.DataFrame(rows, columns=list(REGION_COLUMNS)), summaries
