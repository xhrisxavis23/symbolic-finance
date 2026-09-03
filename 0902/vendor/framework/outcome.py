"""체결 후 30초 동안 가격이 무엇을 했는가. 4분류의 단일 구현.

같은 산술을 여러 곳에서 다시 쓰면 한쪽만 고쳐진 채 조용히 갈라진다. 구 구조가
한 번 겪은 일이라 구현을 하나로 둔다.

기준: ASK1 로 산 가격 대비, 그 뒤 각 틱의 BID1 로 팔았을 때의 손익(bps).
왕복 수수료를 뺀 값을 net, 빼기 전을 gross 라 부른다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import EARLY_WINDOW_SECONDS, FEE_BPS, PATH_SECONDS, PRIMARY_HORIZON_SECONDS


NET_RECOVERY = "NET_RECOVERY"
EARLY_RECOVERY_LATE_REVERSAL = "EARLY_RECOVERY_LATE_REVERSAL"
COST_INSUFFICIENT = "COST_INSUFFICIENT"
PERSISTENT_ADVERSE = "PERSISTENT_ADVERSE"

COHORTS = (NET_RECOVERY, EARLY_RECOVERY_LATE_REVERSAL, COST_INSUFFICIENT, PERSISTENT_ADVERSE)
FAILURE_COHORTS = (PERSISTENT_ADVERSE, EARLY_RECOVERY_LATE_REVERSAL, COST_INSUFFICIENT)


def executable_marks(bid1: np.ndarray) -> np.ndarray:
    """체결 불가능한 호가를 NaN 으로.

    0 이거나 유한하지 않은 틱은 "그 가격에 팔 수 있다" 는 뜻이 아니다. 그대로 두면
    0 인 틱 하나가 -10,000 bp 로 계산돼 최대 손실 폭을 오염시킨다.
    """
    marks = np.asarray(bid1, dtype=float)
    return np.where(np.isfinite(marks) & (marks > 0.0), marks, np.nan)


def net_path(bid1: np.ndarray, entry_price: float, start: int, end: int,
             fee_bps: float = FEE_BPS) -> np.ndarray:
    """진입가 대비 `[start, end]` 각 틱의 net(bps). 양끝 포함.

    이 저장소의 유일한 구현이다.
    """
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return np.full(max(int(end) - int(start) + 1, 0), np.nan)
    marks = executable_marks(bid1)[int(start): int(end) + 1]
    return (marks / float(entry_price) - 1.0) * 10_000.0 - float(fee_bps)


def cohort(*, endpoint_net: float, early_max_net: float, max_gross: float,
           fee_bps: float = FEE_BPS) -> str:
    """겹치지 않는 네 갈래. 회복한 것부터 한 번도 안 오른 것 순으로 판정한다."""
    if endpoint_net > 0.0:
        return NET_RECOVERY
    if early_max_net > 0.0:
        return EARLY_RECOVERY_LATE_REVERSAL
    if max_gross > 0.0 and max_gross - fee_bps <= 0.0:
        return COST_INSUFFICIENT
    return PERSISTENT_ADVERSE


def path_outcome(
    data: Mapping[str, np.ndarray],
    entry_tick: int,
    *,
    entry_price: float | None = None,
    exit_price: float | None = None,
    exit_tick: int | None = None,
    horizons: Sequence[int] = PATH_SECONDS,
    fee_bps: float = FEE_BPS,
    early_window_seconds: float = EARLY_WINDOW_SECONDS,
) -> dict[str, Any] | None:
    """체결 하나의 경로. 관측할 수 없으면 None.

    기준 평가 창(30초)이 그날 데이터 끝을 넘으면 None 을 돌려준다. 짧게 잘라 기록하면
    관측하지 않은 것을 관측한 것처럼 만든다. 부르는 쪽은 이것을 CENSORED 로 센다.

    `exit_price`/`exit_tick` 이 오면 지정가 청산이 그 평가 창 안에 체결된 것이다. 그
    가격으로 나간다. 안 팔렸으면 평가 창 끝 BID1 시장가다 — 미체결 거래를 빼면 나쁜
    거래만 사라진다.

    `cohort` 는 실제 청산 손익으로 분류한다. 반면 `diagnostic_cohort` 는 청산을
    무시하고, 진입 뒤 30초 BID1 경로 자체만 본다. 진입 보완은 후자만 사용한다.
    """
    if PRIMARY_HORIZON_SECONDS not in tuple(horizons):
        raise ValueError("horizons 에 기준 평가 창(30초)이 있어야 한다")

    time_s = np.asarray(data["time_s"], dtype=float)
    bid1 = np.asarray(data["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(data["ask_price"], dtype=float)[:, 0]
    entry = int(entry_tick)
    bought_at = float(ask1[entry]) if entry_price is None else float(entry_price)

    last = int(np.searchsorted(time_s, time_s[entry] + float(PRIMARY_HORIZON_SECONDS), side="left"))
    if last >= len(time_s):
        return None   # 관측 불가 = CENSORED

    exits: dict[int, int] = {}
    gross: dict[int, float] = {}
    net: dict[int, float] = {}
    for seconds in horizons:
        index = int(np.searchsorted(time_s, time_s[entry] + float(seconds), side="left"))
        index = min(index, last)
        exits[int(seconds)] = index
        if exit_price is not None and exit_tick is not None and 0 <= int(exit_tick) <= index:
            realised = (float(exit_price) / bought_at - 1.0) * 10_000.0 - fee_bps
        else:
            realised = float(net_path(bid1, bought_at, index, index, fee_bps)[0])
        net[int(seconds)] = realised
        gross[int(seconds)] = realised + fee_bps

    path = net_path(bid1, bought_at, entry, last, fee_bps)
    if not np.any(np.isfinite(path)):
        return None

    early_last = int(np.searchsorted(time_s, time_s[entry] + float(early_window_seconds), side="right"))
    early = path[: max(1, early_last - entry)]
    early_max = float(np.nanmax(early)) if np.any(np.isfinite(early)) else float("nan")
    max_gross = float(np.nanmax(path)) + fee_bps
    min_gross = float(np.nanmin(path)) + fee_bps

    raw_endpoint_net = float(path[-1])
    return {
        "entry_price": bought_at,
        "exit_tick": exits,
        "gross_bps": gross,
        "net_bps": net,
        "max_favorable_gross_bps": max_gross,
        "max_adverse_gross_bps": min_gross,
        "early_max_net_bps": early_max,
        "ever_net_positive": max_gross > fee_bps,
        "cohort": cohort(endpoint_net=net[PRIMARY_HORIZON_SECONDS],
                         early_max_net=early_max, max_gross=max_gross, fee_bps=fee_bps),
        "diagnostic_cohort": cohort(endpoint_net=raw_endpoint_net,
                                    early_max_net=early_max, max_gross=max_gross,
                                    fee_bps=fee_bps),
    }
