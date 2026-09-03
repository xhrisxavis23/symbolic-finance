"""지정가 주문이 체결됐는지 판정한다.

taker 진입만 계산하면 "최우선 매수호가에 지정가를 건다" 는 가설을 시험할 수 없다 —
수익원이 스프레드를 내지 않는 데 있는데 taker 는 스프레드를 전액 지불하기 때문이다.

## 큐를 무엇이 소진시키는가

게시 시점에 내 앞에는 이미 표시 잔량이 있다.

    remaining = qfrac × (게시 시점 최우선 잔량) + order_qty

매 틱 두 가지가 이것을 줄인다.

  · **내 가격에 도달한 반대편 체결** — 지정가 매수면 매도 주도 체결만 센다. 매수
    주도 체결은 반대편 호가를 소비하므로 내 큐와 무관하다. 그래서 로더가
    `sell_volume`/`sell_min_price` 를 따로 모은다.
  · **설명되지 않은 잔량 감소** — 표시 잔량이 체결로 설명되는 것보다 더 줄었으면
    취소이거나 관측되지 않은 체결이다. `unexplained_fill_fraction` 이 그중 얼마를
    체결로 볼지 정한다.

## hold-through 가 정본인 이유

최우선 호가가 바뀌어도 주문을 빼지 않고 게시가에서 기다린다. 그 뒤로는 **내 가격에
실제로 도달한 체결만** 큐를 소비한다 (잔량-감소 크레딧을 주지 않는다).

취소하는 쪽이 더 안전해 보이지만 반대다. 가격이 불리하게 움직이는 순간 주문을 빼면
**역선택을 공짜로 회피**하게 되어 net 이 과대평가된다. 실측: 취소 모델의 체결률
중앙값(~9%)이 실측 패시브 체결률(~45%)과 5배 어긋났고, hold-through 로 바꾸자 기존
"수익" 61건 중 27건만 살아남았다.

게시 시점 이후만 전방향으로 훑으므로 미래 정보를 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numba import njit


FILLED = "filled"
PRICE_CHANGED = "cancelled_price_moved"
TIMEOUT = "cancelled_timeout"
END_OF_DATA = "end_of_data"
SIGNAL_LOST = "cancelled_signal_lost"

_CODE_TO_NAME = {1: FILLED, 2: PRICE_CHANGED, 3: TIMEOUT, 4: END_OF_DATA,
                 5: SIGNAL_LOST}

DEFAULT_ORDER_QTY = 1.0
DEFAULT_QFRAC = 1.0
DEFAULT_UNEXPLAINED_FILL_FRACTION = 1.0
DEFAULT_MAX_TICKS = 100
DEFAULT_MAX_SECONDS = 10.0
CANCEL_ON_PRICE_CHANGE = False   # hold-through 가 정본

BUY = -1    # 지정가 매수: 매도 주도 체결이 내 큐를 소진
SELL = 1    # 지정가 매도: 매수 주도 체결이 내 큐를 소진


@njit(cache=True)
def _replay_one(own_px, own_depth, opp_v, opp_px, t_sec, post,
                cross_sign, order_qty, qfrac, unexplained, max_ticks, max_seconds, cancel,
                cancel_when_inactive):
    n = own_px.shape[0]
    post_price = own_px[post]
    ahead = qfrac * own_depth[post]      # 게시 시점 내 앞 잔량
    remaining = ahead + order_qty
    prev_depth = own_depth[post]
    deadline = t_sec[post] + max_seconds
    end = min(n - 1, post + max_ticks)
    last = post
    status = 3      # TIMEOUT
    fill = -1
    for j in range(post + 1, end + 1):
        if t_sec[j] > deadline:
            break
        last = j
        at_best = own_px[j] == post_price
        if (not at_best) and cancel > 0.5:
            status = 2      # PRICE_CHANGED
            break
        eligible = 0.0
        if opp_v[j] > 0.0 and cross_sign * (opp_px[j] - post_price) >= 0.0:
            eligible = opp_v[j]
        if at_best:
            depth = own_depth[j] if own_depth[j] > 0.0 else 0.0
            residual = depth - prev_depth + eligible
            depletion = -residual if residual < 0.0 else 0.0
            consumption = eligible + unexplained * depletion
            prev_depth = depth
        else:
            # 호가가 밀린 뒤에는 잔량-감소 크레딧을 주지 않는다. 그 감소는 내 가격이
            # 아니라 다른 가격에서 일어난 것이다.
            consumption = eligible
        remaining -= consumption
        if remaining <= 1e-12:
            status = 1      # FILLED
            fill = j
            break
        # 같은 snapshot에서 먼저 체결 가능성을 반영한 뒤, 여전히 미체결이고 조건이
        # 꺼졌다면 그 관측 뒤에 주문을 취소한다.
        if cancel_when_inactive[j] <= 0.5:
            status = 5      # SIGNAL_LOST
            break
    if status == 3 and last >= n - 1:
        status = 4          # END_OF_DATA
    return status, last, fill, ahead, remaining


@njit(cache=True)
def _replay_batch(own_px, own_depth, opp_v, opp_px, t_sec, posts,
                  cross_sign, order_qty, qfrac, unexplained, max_ticks, max_seconds, cancel,
                  cancel_when_inactive):
    count = posts.shape[0]
    statuses = np.empty(count, dtype=np.int64)
    terminals = np.empty(count, dtype=np.int64)
    fills = np.empty(count, dtype=np.int64)
    aheads = np.empty(count, dtype=np.float64)
    remainings = np.empty(count, dtype=np.float64)
    for k in range(count):
        s, term, fill, ahead, remaining = _replay_one(
            own_px, own_depth, opp_v, opp_px, t_sec, posts[k],
            cross_sign, order_qty, qfrac, unexplained, max_ticks, max_seconds, cancel,
            cancel_when_inactive)
        statuses[k] = s
        terminals[k] = term
        fills[k] = fill
        aheads[k] = ahead
        remainings[k] = remaining
    return statuses, terminals, fills, aheads, remainings


def replay(
    own_price: np.ndarray, own_depth: np.ndarray,
    opposite_volume: np.ndarray, opposite_price: np.ndarray,
    time_s: np.ndarray, post_ticks: np.ndarray,
    *,
    cross_sign: int,
    order_qty: float = DEFAULT_ORDER_QTY,
    qfrac: float = DEFAULT_QFRAC,
    unexplained_fill_fraction: float = DEFAULT_UNEXPLAINED_FILL_FRACTION,
    max_ticks: int = DEFAULT_MAX_TICKS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    cancel_on_price_change: bool = CANCEL_ON_PRICE_CHANGE,
    cancel_when_inactive: np.ndarray | None = None,
    with_queue_state: bool = False,
) -> tuple[np.ndarray, ...]:
    """게시 틱마다 `(상태, 종료 틱, 체결 틱)`. 미체결이면 체결 틱은 -1.

    `with_queue_state` 면 `(내 앞 잔량, 끝났을 때 남은 양)` 두 배열을 더 돌려준다.
    커널은 어차피 이 둘을 계산하므로 추가 비용이 없다. 미체결이 왜 미체결인지 —
    큐가 길었나, 시간이 없었나 — 는 이것 없이 가를 수 없다.
    """
    if cross_sign not in (BUY, SELL):
        raise ValueError("cross_sign 은 -1(지정가 매수) 또는 +1(지정가 매도)")
    if not 0.0 <= float(unexplained_fill_fraction) <= 1.0:
        raise ValueError("unexplained_fill_fraction 은 [0, 1]")
    arrays = [np.ascontiguousarray(np.asarray(a, dtype=np.float64))
              for a in (own_price, own_depth, opposite_volume, opposite_price, time_s)]
    n = len(arrays[0])
    if any(a.ndim != 1 or len(a) != n for a in arrays):
        raise ValueError("큐 배열은 1차원이고 길이가 같아야 한다")
    # 체결이 없는 틱은 소진에 기여하지 않는다. NaN 을 그대로 넘겨도 비교가 항상
    # 거짓이라 결과는 같지만, 의도를 코드로 못박는다.
    arrays[2] = np.where(np.isfinite(arrays[2]), arrays[2], 0.0)
    arrays[3] = np.where(np.isfinite(arrays[3]), arrays[3], np.inf)
    if cancel_when_inactive is None:
        active = np.ones(n, dtype=np.float64)
    else:
        active = np.ascontiguousarray(np.asarray(cancel_when_inactive, dtype=np.float64))
        if active.ndim != 1 or len(active) != n:
            raise ValueError("cancel_when_inactive 는 호가 배열과 같은 길이여야 한다")
        # NaN은 condition을 관측하지 못한 것이므로 주문을 유지하는 근거가 아니다.
        active = np.where(np.isfinite(active) & (active > 0.5), 1.0, 0.0)

    posts = np.ascontiguousarray(np.asarray(post_ticks, dtype=np.int64))
    if not len(posts):
        empty = np.empty(0, dtype=np.int64)
        blank = np.empty(0, dtype=object)
        if with_queue_state:
            zero = np.empty(0, dtype=np.float64)
            return blank, empty, empty, zero, zero
        return blank, empty, empty
    if posts.min() < 0 or posts.max() >= n:
        raise IndexError("post_ticks 가 배열 범위 밖이다")
    codes, terminals, fills, aheads, remainings = _replay_batch(
        *arrays, posts, float(cross_sign), float(order_qty), float(qfrac),
        float(unexplained_fill_fraction), int(max_ticks), float(max_seconds),
        1.0 if cancel_on_price_change else 0.0, active)
    names = np.array([_CODE_TO_NAME[int(c)] for c in codes], dtype=object)
    if with_queue_state:
        return names, terminals, fills, aheads, remainings
    return names, terminals, fills


def entry_fills(data: Mapping[str, np.ndarray], post_ticks: np.ndarray,
                **kwargs: Any) -> tuple[np.ndarray, ...]:
    """지정가 매수. 게시가는 그 틱의 BID1.

    체결 이벤트가 없는 소스는 거절한다. 큐를 소진시킬 근거가 없는데 체결을 추정하면
    근거 없는 체결률을 지어내게 된다.
    """
    missing = [k for k in ("sell_volume", "sell_min_price") if k not in data]
    if missing:
        raise ValueError(f"지정가 매수 큐 재생에는 체결 이벤트가 필요하다. 없는 키: {missing}")
    return replay(np.asarray(data["bid_price"])[:, 0], np.asarray(data["bid_qty"])[:, 0],
                  data["sell_volume"], data["sell_min_price"], data["time_s"],
                  post_ticks, cross_sign=BUY, **kwargs)


def exit_fills(data: Mapping[str, np.ndarray], post_ticks: np.ndarray,
               **kwargs: Any) -> tuple[np.ndarray, ...]:
    """지정가 매도. 진입과 대칭. 게시가는 그 틱의 ASK1.

    평가 창 안에 안 팔리면 미체결로 돌려준다. 부르는 쪽이 평가 창 끝 시장가로 처리한다 —
    미체결 거래를 통계에서 빼면 나쁜 거래만 사라진다.
    """
    missing = [k for k in ("buy_volume", "buy_max_price") if k not in data]
    if missing:
        raise ValueError(f"지정가 매도 큐 재생에는 체결 이벤트가 필요하다. 없는 키: {missing}")
    return replay(np.asarray(data["ask_price"])[:, 0], np.asarray(data["ask_qty"])[:, 0],
                  data["buy_volume"], data["buy_max_price"], data["time_s"],
                  post_ticks, cross_sign=SELL, **kwargs)


@njit(cache=True)
def _replay_repriced_exit(ask_price, ask_depth, buy_volume, buy_max_price, time_s, bid_price,
                          post, hard_stop_price, order_qty, qfrac, unexplained, max_seconds):
    """추적 청산용 매도 주문 하나를 재생한다.

    ASK1이 달라지면 기존 주문을 취소하고 그 틱의 새 ASK1 뒤에 다시 선다. 새 주문은
    그 틱 안에서 공짜로 체결시키지 않고, 다음 틱부터 큐를 소진한다.
    """
    n = len(ask_price)
    posted_price = ask_price[post]
    remaining = qfrac * ask_depth[post] + order_qty
    previous_depth = ask_depth[post]
    deadline = time_s[post] + max_seconds
    last = post
    reposts = 0
    for j in range(post + 1, n):
        if time_s[j] > deadline:
            break
        last = j
        if bid_price[j] <= hard_stop_price:
            return 5, j, -1, posted_price, reposts       # hard stop market exit
        if ask_price[j] != posted_price:
            posted_price = ask_price[j]
            remaining = qfrac * ask_depth[j] + order_qty
            previous_depth = ask_depth[j]
            reposts += 1
            continue
        eligible = buy_volume[j] if buy_max_price[j] >= posted_price else 0.0
        depth = ask_depth[j] if ask_depth[j] > 0.0 else 0.0
        residual = depth - previous_depth + eligible
        depletion = -residual if residual < 0.0 else 0.0
        remaining -= eligible + unexplained * depletion
        previous_depth = depth
        if remaining <= 1e-12:
            return 1, j, j, posted_price, reposts        # FILLED
    if last >= n - 1:
        return 4, last, -1, posted_price, reposts         # END_OF_DATA
    return 3, last, -1, posted_price, reposts             # TIMEOUT


def repriced_trailing_exit(data: Mapping[str, np.ndarray], post_tick: int, *,
                           hard_stop_price: float,
                           order_qty: float = DEFAULT_ORDER_QTY,
                           qfrac: float = DEFAULT_QFRAC,
                           unexplained_fill_fraction: float = DEFAULT_UNEXPLAINED_FILL_FRACTION,
                           max_seconds: float = DEFAULT_MAX_SECONDS
                           ) -> tuple[str, int, int, float, int]:
    """ASK1 변경 때마다 다시 게시하는 추적 청산 지정가.

    반환값은 `(상태, 종료 틱, 체결 틱, 마지막 게시가격, 다시_게시한_횟수)`다.
    """
    missing = [key for key in ("buy_volume", "buy_max_price") if key not in data]
    if missing:
        raise ValueError(f"추적 청산 큐 재생에는 체결 이벤트가 필요하다. 없는 키: {missing}")
    if not 0.0 <= float(unexplained_fill_fraction) <= 1.0:
        raise ValueError("unexplained_fill_fraction 은 [0, 1]")
    ask_price = np.ascontiguousarray(np.asarray(data["ask_price"], dtype=np.float64)[:, 0])
    ask_depth = np.ascontiguousarray(np.asarray(data["ask_qty"], dtype=np.float64)[:, 0])
    buy_volume = np.ascontiguousarray(np.where(
        np.isfinite(data["buy_volume"]), data["buy_volume"], 0.0), dtype=np.float64)
    buy_max_price = np.ascontiguousarray(np.where(
        np.isfinite(data["buy_max_price"]), data["buy_max_price"], -np.inf), dtype=np.float64)
    time_s = np.ascontiguousarray(np.asarray(data["time_s"], dtype=np.float64))
    bid_price = np.ascontiguousarray(np.asarray(data["bid_price"], dtype=np.float64)[:, 0])
    code, terminal, filled_at, price, reposts = _replay_repriced_exit(
        ask_price, ask_depth, buy_volume, buy_max_price, time_s, bid_price, int(post_tick),
        float(hard_stop_price), float(order_qty), float(qfrac),
        float(unexplained_fill_fraction), float(max_seconds))
    status = {1: FILLED, 3: TIMEOUT, 4: END_OF_DATA, 5: "hard_stop"}[int(code)]
    return status, int(terminal), int(filled_at), float(price), int(reposts)
