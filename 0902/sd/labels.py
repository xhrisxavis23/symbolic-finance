"""라벨. 계획서 §3 S0 — 청산이 타깃을 정의한다.

두 가지를 만든다.

  y_path   진입 후 30초 BID1 경로의 net. **청산 규칙을 타지 않는다.**
           교사의 주 학습 신호다.
  y_fill   BID1 지정가가 10초/100틱 안에 체결됐을지의 근사.

`y_exec`(원장 net)는 여기서 만들지 않는다. 그것은 재생이 만들고 판정에만 쓴다.
교사를 `y_exec` 로 학습시키면 청산 규칙의 특이점을 외운다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from . import config

_framework = config.load_framework()
from framework.config import ENTRY_ORDER_MAX_SECONDS, ENTRY_ORDER_MAX_TICKS  # noqa: E402
from framework.config import FEE_BPS, PRIMARY_HORIZON_SECONDS                # noqa: E402


@dataclass(frozen=True)
class Labels:
    y_path: np.ndarray
    y_fill: np.ndarray
    mask: np.ndarray
    raw_net_bps: np.ndarray


def build(arrays: dict[str, np.ndarray]) -> Labels:
    time_s = np.asarray(arrays["time_s"], dtype=float)
    bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    n = len(time_s)

    # 평가 창 끝 틱. 데이터 끝을 넘으면 관측 불가 = 마스크 (CENSORED 와 같은 뜻).
    horizon = np.searchsorted(time_s, time_s + float(PRIMARY_HORIZON_SECONDS), side="left")
    observable = horizon < n

    # 체결 가능한 호가만. 0 이나 비유한은 "그 가격에 팔 수 있다" 가 아니다.
    sellable = np.where(np.isfinite(bid1) & (bid1 > 0.0), bid1, np.nan)
    buyable = np.where(np.isfinite(ask1) & (ask1 > 0.0), ask1, np.nan)

    exit_index = np.where(observable, np.clip(horizon, 0, n - 1), 0)
    raw = (sellable[exit_index] / buyable - 1.0) * 1e4 - FEE_BPS
    raw = np.where(observable, raw, np.nan)

    mid = (bid1 + ask1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_bps = (ask1 - bid1) / mid * 1e4
    friction = spread_bps + FEE_BPS
    y_path = np.where(friction > 0.0, raw / friction, np.nan)

    mask = observable & np.isfinite(y_path)
    return Labels(y_path=y_path, y_fill=_fill_label(arrays, time_s, bid1),
                  mask=mask, raw_net_bps=raw)


@njit(cache=True)
def _fill_label_inner(sell_min: np.ndarray, bid1: np.ndarray, last: np.ndarray) -> np.ndarray:
    """`_fill_label` 의 이중 루프 본체. 순수 njit — python 객체 없음.

    각 i 에 대해 [i+1, last[i]) 구간 안의 sell_min 값 중 유한하고 0 보다 큰
    것들의 최솟값이 bid1[i] 이하면 1.0. 부동소수 비교라 `v == v` 로 NaN 을
    거른다 (numba 는 이 관용구를 잘 컴파일한다).
    """
    n = sell_min.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        end = last[i]
        best = np.inf
        for j in range(i + 1, end):
            v = sell_min[j]
            if v == v and v > 0.0 and v < best:
                best = v
        if best <= bid1[i]:
            out[i] = 1.0
    return out


def _fill_label(arrays: dict[str, np.ndarray], time_s: np.ndarray,
                bid1: np.ndarray) -> np.ndarray:
    """BID1 지정가가 대기 창 안에 체결됐을지의 보수적 근사.

    정확한 큐 순번은 이 데이터에 없다. 재생이 쓰는 정본 큐 모델이 정답이고,
    이것은 **교사에게 줄 학습 신호**일 뿐이다. 판정에는 절대 쓰지 않는다.

    규칙: 게시 시점 BID1 가격 이하로 매도 주도 체결이 실제로 내려왔는가.
    체결 데이터가 없으면 전부 0 — 없는 관측을 지어내지 않는다.

    `last` 배열은 여기서 numpy 로 벡터화해 미리 계산한다. 이중 루프 본체만
    `_fill_label_inner` (njit) 로 뺀다 — 005930 은 (정규장 필터 후) 틱이
    193,602개라 순수 파이썬 이중 루프는 종목당 수 분이 걸린다.
    """
    n = len(time_s)
    sell_min = arrays.get("sell_min_price")
    if sell_min is None:
        return np.zeros(n, dtype=float)
    sell_min = np.ascontiguousarray(np.asarray(sell_min, dtype=np.float64))
    deadline = time_s + float(ENTRY_ORDER_MAX_SECONDS)
    last_by_clock = np.searchsorted(time_s, deadline, side="right")
    last_by_ticks = np.arange(n) + int(ENTRY_ORDER_MAX_TICKS) + 1
    last = np.minimum(np.minimum(last_by_clock, last_by_ticks), n).astype(np.int64)
    bid1 = np.ascontiguousarray(np.asarray(bid1, dtype=np.float64))
    return _fill_label_inner(sell_min, bid1, last)
