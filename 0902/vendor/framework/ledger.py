"""백테스트. 계약을 틱 위에서 돌려 **decision 단위** 원장을 만든다.

## 왜 decision 인가

구 원장은 신호가 켜진 틱을 전부 "시도(attempt)" 로 세고 그것을 성과의 분모로 썼다.
그런데 그 틱의 84.7% 는 이미 포지션을 들고 있어 아무 행동도 할 수 없는 틱이었다
(실측: signal tick 22,723,360 중 decision 3,485,535).

분모가 이러면 성과가 신호 지속시간에 좌우된다. 실측으로 같은 가설을 번역한 계약
둘이 **같은 거래를 하고도** 신호를 촘촘히 켠 쪽이 이겼다.

## 네 가지 상태

    BLOCKED   신호는 켜졌으나 포지션 보유중. **decision 이 아니다** (행으로 남기지 않음)
    CENSORED  flat 이었으나 30초 평가 창을 관측할 수 없다. decision 이지만 채점 불가
    FILLED    flat 이었고 체결됐다
    UNFILLED  flat 이었고 주문했으나 미체결 (지정가에서만)

불변식 — 모든 (계약, 종목, 날) 에서 정확히 성립해야 한다.

    decisions = FILLED + UNFILLED + CENSORED
    scorable  = FILLED + UNFILLED
    taker 진입이면 UNFILLED = 0 이므로 scorable == fills
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import catalog, contract, data as tickdata, exits, fill, outcome
from .config import (
    ENTRY_ORDER_MAX_SECONDS, ENTRY_ORDER_MAX_TICKS,
    FEE_BPS, LIMIT_ENTRY, LIMIT_EXIT, MIN_REFERENCE_SAMPLES, PRIMARY_HORIZON_SECONDS,
    QUANTILES, TICK_ROOT, now_utc, sha256_json, write_json,
)


BLOCKED = "BLOCKED"
CENSORED = "CENSORED"
FILLED = "FILLED"
UNFILLED = "UNFILLED"
# 원장 행이 가질 수 있는 상태. BLOCKED 는 행으로 남지 않으므로 여기 없다.
STATUSES = (FILLED, UNFILLED, CENSORED)

THRESHOLD_PREFIX = "pq"


@dataclass(frozen=True)
class Partition:
    """원장의 분할 단위. (계약, 종목, 날) 하나.

    구 구조는 이 키를 `_partition_code` 라는 암묵적 컬럼으로 들고 다녔고, 그 컬럼을
    붙이지 않은 경로에서 재현 확인이 KeyError 로 죽었다. 인자로 명시한다.
    """

    contract_id: str
    symbol: str
    date: str

    @property
    def key(self) -> str:
        return f"{self.contract_id}|{self.symbol}|{self.date}"


def threshold_column(feature: str, quantile: float) -> str:
    return f"{THRESHOLD_PREFIX}{quantile:.2f}:{feature}"


def _ledger_columns() -> tuple[str, ...]:
    """결정이 0개여도 다음 Stage가 읽을 수 있는 원장 schema.

    feature 열은 Catalog **전체**다. Agent 가 진입식에 쓸 수 있는 축과 원장이 남기는
    축이 다르면, 자기가 쓴 축으로 자기 실패를 읽을 수 없다. 임계 열은 guard 가 쓰는
    축에만 필요하므로 그대로 `guard_axes()` 를 쓴다.
    """
    axes = tuple(catalog.list_features())
    guard = tuple(catalog.guard_axes())
    return (
        "contract_id", "symbol", "date", "decision_index", "episode_index",
        "entry_tick", "status", "fill_tick", "exit_tick", "prior_reference_date",
        "entry_signal_active_at_fill", "entry_signal_persisted_until_fill",
        "entry_signal_active_share_until_fill", "order_type", "entry_order_terminal_status",
        "entry_price", "net_bps", "gross_bps", "cohort", "diagnostic_cohort",
        "max_favorable_gross_bps", "max_adverse_gross_bps", "early_max_net_bps",
        "exit_reason", "holding_seconds",
        # 경로 분해. 매수측·매도측을 따로 보고 게시-체결 사이의 역선택을 가른다.
        "fill_slippage_bps", "bid_move_bps", "ask_move_bps",
        "entry_spread_bps", "exit_spread_bps",
        # 이 종목-일의 호가 단위. bps 로 적힌 손절·추적·비용이 몇 틱인지 읽는 기준.
        "tick_size_bps",
        # 진입 지정가의 큐 상태. 미체결 원인을 큐 길이와 시간으로 가른다.
        "queue_ahead_at_post", "queue_remaining_at_end", "queue_consumed_share",
        # 이 종목-일이 어떤 종목-일이었나. 파티션 전체에 같은 값이 들어간다.
        "day_spread_bps_median", "day_tick_size_bps_median",
        "day_trade_tick_count", "day_mid_range_bps",
        # 미체결 결정의 반사실 경로. 관측이 아니라 "그 틱에 시장가로 샀다면" 이라는
        # 가정이며, 어떤 성과 합계에도 들어가지 않는다.
        "counterfactual_net_bps", "counterfactual_cohort",
        "counterfactual_max_favorable_gross_bps", "counterfactual_max_adverse_gross_bps",
        *axes,
        *(threshold_column(feature, quantile) for feature in guard for quantile in QUANTILES),
    )


def tick_size_bps(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """호가 단위를 mid 대비 bps 로. 진단 전용이고 Catalog feature 가 아니다.

    프로필의 손절 -120bp·추적 -30bp·비용 23bp 는 전부 bps 인데, 1틱이 몇 bps 인지는
    가격대마다 몇 배씩 다르다. 그래서 같은 -30bp 추적선이 어떤 종목에서는 여러 틱이고
    어떤 종목에서는 한 틱 남짓이 된다. 뒤쪽은 방향 예측 규칙이 아니라 호가 노이즈에
    반응하는 규칙이다. 그 구분을 하려면 원장에 이 값이 있어야 한다.

    인접 호가단 차이의 최솟값으로 잡는다. 빈 단이 있으면 그 간격은 틱의 배수라 최솟값이
    가장 좋은 추정이다. 양쪽 다 비어 있으면 NaN.
    """
    bid_px = np.asarray(arrays["bid_price"], dtype=float)
    ask_px = np.asarray(arrays["ask_price"], dtype=float)
    bid1, ask1 = bid_px[:, 0], ask_px[:, 0]
    mid = np.where(np.isfinite(bid1) & (bid1 > 0.0) & np.isfinite(ask1) & (ask1 > 0.0),
                   (bid1 + ask1) / 2.0, np.nan)
    gaps = np.concatenate(
        [bid_px[:, :-1] - bid_px[:, 1:], ask_px[:, 1:] - ask_px[:, :-1]], axis=1)
    gaps = np.where(np.isfinite(gaps) & (gaps > 0.0), gaps, np.inf)
    tick = gaps.min(axis=1)
    return np.where(np.isfinite(tick) & np.isfinite(mid) & (mid > 0.0),
                    tick / mid * 1e4, np.nan)


def day_profile(arrays: Mapping[str, np.ndarray],
                ticks_bps: np.ndarray | None = None) -> dict[str, float | None]:
    """이 종목-일이 어떤 종목-일이었는가. 진단 전용이고 Catalog feature 가 아니다.

    원장의 feature 는 전부 **진입 틱의 순간값**이라, 그날 그 종목이 넓었는지 활발했는지
    격자가 성겼는지가 남지 않았다. 같은 수식을 전 종목에 똑같이 적용해서 특정 종목에
    쏠렸다면 그것은 수식이 잡은 구조인데, 그 종목들이 어떤 종목인지 모르면 가설이 말한
    성격과 대조할 수가 없다.

    파티션 전체에 같은 값이 들어간다. 중복이지만 그래야 원장 하나만 읽고 대조된다.
    """
    bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    usable = np.isfinite(bid1) & (bid1 > 0.0) & np.isfinite(ask1) & (ask1 > 0.0)
    mid = np.where(usable, (bid1 + ask1) / 2.0, np.nan)
    spread = np.where(usable, (ask1 / bid1 - 1.0) * 1e4, np.nan)

    def median(values: np.ndarray) -> float | None:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if len(finite) else None

    middle = median(mid)
    span = (float(np.nanmax(mid) - np.nanmin(mid)) if np.isfinite(mid).any() else None)
    traded = None
    if "buy_volume" in arrays and "sell_volume" in arrays:
        buy = np.nan_to_num(np.asarray(arrays["buy_volume"], dtype=float))
        sell = np.nan_to_num(np.asarray(arrays["sell_volume"], dtype=float))
        traded = float(np.count_nonzero((buy > 0.0) | (sell > 0.0)))
    return {
        "day_spread_bps_median": median(spread),
        "day_tick_size_bps_median": (median(np.asarray(ticks_bps, dtype=float))
                                     if ticks_bps is not None else None),
        "day_trade_tick_count": traded,
        "day_mid_range_bps": (span / middle * 1e4
                              if span is not None and middle and middle > 0.0 else None),
    }


def horizon_exit_ticks(time_s: np.ndarray, seconds: int = PRIMARY_HORIZON_SECONDS) -> np.ndarray:
    """각 틱에서 `seconds` 뒤에 해당하는 틱 번호. 데이터 끝을 넘으면 n 이상.

    체결 여부와 무관하게 전 틱에 대해 한 번에 구한다 — 이 값이 곧 "그 진입이 언제
    끝나는가" 이고, 그것이 다음 decision 이 언제 가능한가를 정한다.
    """
    return np.searchsorted(np.asarray(time_s, dtype=float),
                           np.asarray(time_s, dtype=float) + float(seconds), side="left")


def decisions_from_signal(
    signal: np.ndarray,
    exit_ticks: np.ndarray,
    *,
    fill_ticks: np.ndarray | None = None,
    order_terminal: np.ndarray | None = None,
    observable: np.ndarray | None = None,
) -> pd.DataFrame:
    """신호 배열 하나를 decision 행으로 바꾼다. 이 모듈의 핵심.

    한 번 훑으면서 flat 여부를 추적한다. flat 이면 decision 을 하나 만들고, 포지션을
    잡으면 그 청산 틱 다음까지 flat 이 아니다.

    `fill_ticks` 는 그 틱에 낸 주문이 체결되는 틱이다 (taker 는 자기 자신, 지정가는
    나중이거나 -1=미체결). `order_terminal` 은 주문이 결판난 틱이다 — 미체결이어도
    그때까지는 주문이 살아 있으므로 새 진입 결정을 내릴 수 없다. taker 에서는 둘 다
    자기 자신이라 예전 동작과 완전히 같다.

    상태:
      FILLED    체결. 청산 틱 다음까지 flat 아님
      UNFILLED  주문했으나 미체결. 주문이 결판난 틱 다음까지 flat 아님
      CENSORED  체결했으나 30초 평가 창을 관측할 수 없다. 포지션을 잡지 않은 것으로 둔다

    `episode_index` 는 신호가 연속으로 켜진 구간의 번호다. episode 와 decision 은
    다르다 — 신호가 계속 켜져 있어도 30초마다 다시 살 수 있으므로 긴 episode 하나가
    decision 여럿을 만들고, 반대로 앞 포지션에 통째로 막힌 episode 는 decision 을
    하나도 만들지 못한다 (실측: episode 7,570,424개 → decision 3,485,535개).
    """
    signal = np.asarray(signal, dtype=bool)
    n = len(signal)
    exit_ticks = np.asarray(exit_ticks, dtype=np.int64)
    if observable is None:
        observable = exit_ticks < n
    observable = np.asarray(observable, dtype=bool)
    if fill_ticks is None:
        fill_ticks = np.arange(n, dtype=np.int64)          # taker: 신호 틱에 즉시 체결
    if order_terminal is None:
        order_terminal = np.arange(n, dtype=np.int64)
    fill_ticks = np.asarray(fill_ticks, dtype=np.int64)
    order_terminal = np.asarray(order_terminal, dtype=np.int64)

    ticks = np.flatnonzero(signal)
    columns = ["entry_tick", "episode_index", "status", "fill_tick", "exit_tick"]
    if not len(ticks):
        empty = pd.DataFrame(columns=columns)
        empty.attrs.update({"blocked_signal_ticks": 0, "raw_signal_ticks": 0,
                            "signal_episodes": 0})
        return empty

    # 연속 구간마다 episode 번호. 신호가 끊기면 새 episode.
    episode = np.concatenate(([0], np.cumsum(np.diff(ticks) != 1)))

    rows: list[tuple[int, int, str, int, int]] = []
    blocked = 0
    available = -1
    for position, raw in enumerate(ticks):
        tick = int(raw)
        if tick < available:
            blocked += 1
            continue
        episode_index = int(episode[position])
        executed = int(fill_ticks[tick])
        if executed < 0:
            # 주문은 냈으나 체결되지 않았다. 결판날 때까지는 새 결정을 못 한다.
            rows.append((tick, episode_index, UNFILLED, -1, -1))
            available = int(order_terminal[tick]) + 1
        elif observable[executed]:
            rows.append((tick, episode_index, FILLED, executed, int(exit_ticks[executed])))
            available = int(exit_ticks[executed]) + 1
        else:
            # 30초를 관측할 수 없다. 포지션을 잡지 않은 것으로 두어 flat 을 유지한다.
            rows.append((tick, episode_index, CENSORED, -1, -1))

    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["blocked_signal_ticks"] = blocked
    frame.attrs["raw_signal_ticks"] = int(len(ticks))
    frame.attrs["signal_episodes"] = int(episode[-1] + 1)
    return frame


def apply_guard(
    signal: np.ndarray,
    features: Mapping[str, np.ndarray],
    guard: Mapping[str, Any],
    thresholds: Mapping[str, float | np.ndarray] | None,
) -> np.ndarray:
    """가드 하나를 신호에 AND 로 붙인다.

    임계는 각 tick의 현재 tick을 제외한 직전 100개 관측 분위수다. 종목 공통 절대
    임계는 유동성에 비례하는 호가량에 대해 종목마다 전혀 다른 위치에 놓여, 상태가
    아니라 종목 크기를 거르는 필터가 된다.
    """
    if str(guard.get("kind", "")) == "sklearn_probability":
        if str(guard.get("operator", ">=")) != ">=":
            raise ValueError("sklearn_probability guard는 >= 연산자만 지원한다")
        active = np.flatnonzero(np.asarray(signal, dtype=bool))
        if not len(active):
            return np.zeros_like(signal, dtype=bool)
        names = tuple(map(str, guard["features"]))
        missing = sorted(set(names) - set(features))
        if missing:
            raise KeyError(f"분류기 입력 feature가 없다: {missing}")
        frame = pd.DataFrame(
            {name: np.asarray(features[name], dtype=float)[active] for name in names})
        frame = frame.replace([np.inf, -np.inf], np.nan)
        model = _load_probability_model(str(guard["model_path"]))
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        positive_class = guard.get("positive_class", 1)
        classes = np.asarray(model.classes_)
        matched = np.flatnonzero(classes == positive_class)
        if len(matched) != 1:
            raise ValueError(f"분류기 positive_class를 찾을 수 없다: {positive_class}")
        score = probabilities[:, int(matched[0])]
        keep = np.zeros_like(signal, dtype=bool)
        keep[active] = np.isfinite(score) & (score >= float(guard["threshold"]))
        return keep

    feature = str(guard["feature"])
    column = threshold_column(feature, float(guard["quantile"]))
    if not thresholds or column not in thresholds:
        return np.zeros_like(signal, dtype=bool)
    values = np.asarray(features[feature], dtype=float)
    cut = np.asarray(thresholds[column], dtype=float)
    if cut.ndim == 0:
        cut = np.full(len(values), float(cut))
    if cut.shape != values.shape:
        raise ValueError(f"guard 임계 배열 모양이 {cut.shape} 이다. {values.shape} 여야 한다")
    keep = values <= cut if str(guard["operator"]) == "<=" else values >= cut
    return np.asarray(signal, dtype=bool) & np.isfinite(values) & np.isfinite(cut) & keep


@lru_cache(maxsize=8)
def _load_probability_model(path: str) -> Any:
    """worker마다 고정된 분류기를 한 번만 읽는다."""
    import joblib
    return joblib.load(path)


def rolling_tick_thresholds(features: Mapping[str, np.ndarray], *,
                            guards: Sequence[Mapping[str, Any]] | None = None,
                            quantiles: Sequence[float] = QUANTILES) -> dict[str, np.ndarray]:
    """guard에 필요한 tick별 q cut만 만든다. 각 cut은 현재 tick을 보지 않는다."""
    requested: dict[str, set[float]] = {}
    if guards is None:
        requested = {name: {float(q) for q in quantiles} for name in features}
    else:
        for guard in guards:
            if str(guard.get("kind", "")) == "sklearn_probability":
                continue
            name = str(guard["feature"])
            if name in features:
                requested.setdefault(name, set()).add(float(guard["quantile"]))

    out: dict[str, np.ndarray] = {}
    for name, requested_quantiles in requested.items():
        values = np.asarray(features[name], dtype=float)
        requests = [(quantile, "higher") for quantile in sorted(requested_quantiles)]
        cuts = contract.prior_tick_quantile_grid(values, requests)
        for quantile, _method in requests:
            out[threshold_column(name, quantile)] = cuts[(quantile, "higher")]
    return out


def prior_day_thresholds(
    symbol: str,
    date: str,
    reference_dates: Sequence[str],
    *,
    features: Sequence[str] | None = None,
    quantiles: Sequence[float] = QUANTILES,
    min_samples: int = MIN_REFERENCE_SAMPLES,
    root: Path = TICK_ROOT,
) -> tuple[dict[str, float], str] | None:
    """그 종목의 직전 유효일에서 뜬 분위수. 표본이 모자라면 None.

    None 이면 부르는 쪽이 그 종목-일을 건너뛴다. 같은 날로 대체하지 않는다 —
    적용할 날의 분포로 그 날을 자르면 컷이 아니다.
    """
    features = tuple(features or catalog.guard_axes())
    prior = tickdata.prior_valid_day(symbol, date, list(reference_dates), root)
    while prior is not None:
        try:
            arrays, _ = tickdata.load(symbol, prior, root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData):
            prior = tickdata.prior_valid_day(symbol, prior, list(reference_dates), root)
            continue
        book = catalog.Book(arrays)
        values = catalog.compute_features(features, book)
        out: dict[str, float] = {}
        for name in features:
            column = values.get(name)
            finite = column[np.isfinite(column)] if column is not None else np.empty(0)
            for quantile in quantiles:
                out[threshold_column(name, quantile)] = (
                    float(np.quantile(finite, quantile)) if len(finite) >= min_samples else np.nan)
        return out, prior
    return None


def _resolve_exit(arrays, time_s, bid1, ask1, n, executed, entry_price,
                  horizon_tick, limit_exit, job, legacy_program=None):
    """포지션 하나가 언제 어느 가격에 끝나는가.

    `stop_gross_bps` 손절은 즉시 BID1 시장가다. 최고 ASK1 대비
    `trailing_drawdown_gross_bps` 추적선에 닿으면 ASK1 지정가를 건다. 추적 주문은
    새 ASK1이 생길 때마다 그 가격으로 다시 게시하고, 이 대기 중에도 손절은 우선한다.

        entry gross <= stop                       -> BID1 시장가
        running peak ASK1 drawdown <= trailing    -> ASK1 지정가 (최대 rest 초)
        추적 지정가가 미체결                      -> BID1 시장가
        평가 창 끝까지 무트리거                   -> BID1 시장가

    먼저 닿는 쪽이 이긴다 — 나중 틱의 정보로 되돌아가 고르지 않는다.
    """
    stop_level = job.get("stop_gross_bps")
    trailing_level = job.get("trailing_drawdown_gross_bps")
    horizon_observable = int(horizon_tick) < n
    cap = min(int(horizon_tick), n - 1)
    if job.get("ask1_exit"):
        # Sandbox_4의 스프레드 포획 가정: 매수 체결 직후 그 ASK1에 매도 지정가를
        # 게시한다. 평가 창 안에 매도 큐가 안 소진되면 마지막 BID1으로 정리한다.
        if cap <= executed:
            return float(bid1[executed]), int(executed), "ask1_no_horizon_bid1"
        max_ticks = min(cap - executed, int(job.get("ask1_exit_max_ticks") or (cap - executed)))
        max_seconds = min(
            float(time_s[cap] - time_s[executed]),
            float(job.get("ask1_exit_max_seconds") or (time_s[cap] - time_s[executed])),
        )
        status, terminal, sold = fill.exit_fills(
            arrays, np.asarray([executed], dtype=np.int64),
            max_ticks=max_ticks, max_seconds=max_seconds)
        if status[0] == fill.FILLED:
            return float(ask1[executed]), int(sold[0]), "ask1_queue"
        return float(bid1[int(terminal[0])]), int(terminal[0]), f"ask1_{status[0]}_bid1"

    triggered_at = trigger_reason = None
    if entry_price > 0 and cap > executed and (
            stop_level is not None or trailing_level is not None):
        hits = []
        if stop_level is not None:
            bid_prices = bid1[executed: cap + 1]
            stop_path = (bid_prices / entry_price - 1.0) * 1e4
            finite_bid = np.isfinite(stop_path) & (bid_prices > 0.0)
            where = np.flatnonzero(finite_bid & (stop_path <= float(stop_level)))
            if len(where):
                hits.append((int(where[0]), 0, "stop"))
        if trailing_level is not None:
            ask_prices = ask1[executed: cap + 1]
            finite_ask = np.isfinite(ask_prices) & (ask_prices > 0.0)
            peaks = np.maximum.accumulate(np.where(finite_ask, ask_prices, -np.inf))
            drawdown = (ask_prices / peaks - 1.0) * 1e4
            where = np.flatnonzero(finite_ask & np.isfinite(drawdown) &
                                   (drawdown <= float(trailing_level)))
            if len(where):
                hits.append((int(where[0]), 1, "trailing"))
        if hits:
            offset, _priority, trigger_reason = min(hits)
            triggered_at = executed + offset

    # 과거 AlgorithmPackage의 상태 청산은 현재 기본 청산과 별개 신호다. 둘 중
    # 먼저 **발화한** 쪽으로 끝낸다. 같은 틱이면 현재 기본 청산을 우선해, 결과가
    # 행 순서에 따라 바뀌지 않게 한다. 과거 상태 청산은 원래처럼 BID1 즉시 청산이다.
    legacy_at = None
    if legacy_program is not None:
        from . import legacy
        legacy_at = legacy.first_exit_trigger(arrays, legacy_program, executed, cap)
    if legacy_at is not None and (triggered_at is None or legacy_at < triggered_at):
        return float(bid1[legacy_at]), int(legacy_at), "legacy_state_bid1"

    if triggered_at is None:
        # 기본 정본은 S30 Feature Profile 라벨과 별개로 실제 청산 시간 상한의 끝값을 쓴다.
        # 하지만 명시적으로 다른 시간 상한을 준 단일 실험은 그 시점 BID1을 실제 청산가로
        # 남겨야 짧은 평가 창 결과를 실제 청산 결과로 잘못 재사용하지 않는다.
        if job.get("horizon_seconds") is not None:
            if not horizon_observable:
                return None, None, "horizon_unobserved"
            return float(bid1[cap]), int(cap), "horizon_cap"
        return None, None, "horizon_cap"

    # 진입가 손절은 **그 자리 BID1 시장가**다. 지정가로 걸면 "나가고 싶은데 좋은
    # 가격에만" 이 되어, 시장이 도망가는 바로 그 순간에 못 나간다.
    if trigger_reason == "stop" or not limit_exit:
        return float(bid1[triggered_at]), int(triggered_at), f"{trigger_reason}_bid1"

    rest = float(job.get("exit_limit_rest_seconds") or fill.DEFAULT_MAX_SECONDS)
    hard_stop_price = entry_price * (1.0 + float(stop_level) / 1e4)
    status, last, sold, sell_price, reposts = fill.repriced_trailing_exit(
        arrays, triggered_at, hard_stop_price=hard_stop_price, max_seconds=rest)

    def within_explicit_horizon(price: float, tick: int, reason: str) -> tuple[float, int, str]:
        if job.get("horizon_seconds") is not None and int(tick) > cap:
            return float(bid1[cap]), int(cap), "horizon_cap_after_trailing"
        return float(price), int(tick), reason

    if status == fill.FILLED:
        reason = "trailing_repriced_ask1" if reposts else "trailing_ask1"
        return within_explicit_horizon(sell_price, sold, reason)
    if status == "hard_stop":
        return within_explicit_horizon(float(bid1[last]), last, "stop_bid1_after_trailing")
    reason = "trailing_repriced_timeout_bid1" if reposts else "trailing_timeout_bid1"
    return within_explicit_horizon(float(bid1[last]), last, reason)


def _one_partition(job: dict[str, Any]) -> dict[str, Any]:
    """(계약, 종목, 날) 하나. 프로세스 경계를 넘으므로 인자는 전부 값이다."""
    part = Partition(job["contract_id"], job["symbol"], job["date"])
    if "_shared_arrays" in job:
        arrays, path = job["_shared_arrays"], Path(job["_shared_path"])
    else:
        try:
            arrays, path = tickdata.load(part.symbol, part.date, Path(job["root"]))
        except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
            # 데이터의 부재는 실패가 아니다. 조용히 사라지지 않게 세어만 둔다.
            return {"rows": [], "counts": None, "missing": {**part.__dict__, "reason": str(error)}}
        except Exception as error:  # noqa: BLE001 - 한 파일 스키마 오류가 전수 실행을 멈추지 않는다
            return {"rows": [], "counts": None,
                    "error": {**part.__dict__, "error": f"{type(error).__name__}: {error}"}}
    try:
        template = job["contract"]
        parameters = job.get("parameters") or {}
        required_parameters = set((template.get("parameter_interface") or {}).keys())
        unresolved = sorted(required_parameters - set(parameters))
        if unresolved:
            # q를 만들 수 없는 종목-일은 조용히 계약을 원값으로 실행하지 않는다.
            # parameter_table에 없는 경우도 이 경로로 남겨야 worker 오류가 아니다.
            return {"rows": [], "counts": None,
                    "missing": {**part.__dict__,
                                "reason": f"임계를 풀 수 없다: {unresolved}"}}
        if parameters:
            # rolling source는 q만 받는다. 실제 cut은 현재 tick을 빼고 직전 100 tick에서
            # 실행 중 만든다. 구 전일 source와 literal은 기존처럼 숫자 cut을 주입한다.
            missing = [name for name, value in parameters.items()
                       if value is None or not np.isfinite(float(value))]
            if missing:
                return {"rows": [], "counts": None,
                        "missing": {**part.__dict__,
                                    "reason": f"임계를 풀 수 없다: {missing}"}}
            if not contract.uses_rolling_quantiles(template):
                template = dict(template)
                template["entry_program"] = contract._substitute(
                    template["entry_program"],
                    {f"{catalog.UNRESOLVED_PREFIX}{k}": float(v)
                     for k, v in parameters.items()})
        legacy_program = template.get("legacy_program")
        legacy_threshold_detail = None
        legacy_reference_date = None
        if legacy_program is None:
            try:
                signal = contract.entry_signal(
                    arrays, template,
                    rolling_quantiles=parameters if contract.uses_rolling_quantiles(template) else None,
                    rolling_thresholds=job.get("_shared_rolling_thresholds"),
                    precomputed_expressions=job.get("_shared_entry_expression_cache"))
            except KeyError as error:
                # trade 분류 필드처럼 이 종목-일 원본에 없는 feature는 0으로 만들지 않는다.
                # q를 사전 보정표에서 제외하지 않아도 이 실행 원장에 명시적 missing으로 남긴다.
                return {"rows": [], "counts": None,
                        "missing": {**part.__dict__, "reason": str(error)}}
        else:
            from . import legacy
            bound = legacy.bind_prior_day_thresholds(
                legacy_program, symbol=part.symbol, date=part.date,
                reference_dates=job["reference_dates"], root=Path(job["root"]))
            if bound is None:
                return {"rows": [], "counts": None,
                        "missing": {**part.__dict__, "reason": "전일 종목별 분위수를 풀 수 없다"}}
            legacy_program, legacy_threshold_detail = bound
            legacy_reference_date = str(legacy_threshold_detail["reference_date"])
            signal = legacy.entry_signal(arrays, legacy_program)
        if "_shared_features" in job:
            features = job["_shared_features"]
        else:
            book = catalog.Book(arrays)
            axes = catalog.list_features()
            features = catalog.compute_features(axes, book)

        thresholds: dict[str, float | np.ndarray] = {}
        reference_date = ("PREVIOUS_100_TICKS" if legacy_program is None
                          and contract.uses_rolling_quantiles(template) else legacy_reference_date)
        if job["guards"] and legacy_program is None:
            thresholds = job.get("_shared_guard_thresholds") or rolling_tick_thresholds(
                features, guards=job["guards"])
            reference_date = "PREVIOUS_100_TICKS"

        for guard in job["guards"]:
            signal = apply_guard(signal, features, guard, thresholds)

        time_s = np.asarray(arrays["time_s"], dtype=float)
        # 주문을 낼 수 있는 시점과 주문을 유지할 조건은 다르다. 격자는 전자만 줄인다.
        entry_condition = np.asarray(signal, dtype=bool).copy()
        inactive_prefix = np.cumsum(~entry_condition, dtype=np.int64)
        # 결정 기회를 시계 격자로 자른다. 신호가 켜져도 격자 밖이면 결정이 아니다.
        # 검증이 이 표본 위에서 성립했으므로 백테스트도 같은 표본을 써야 한다.
        grid_seconds = job.get("decision_grid_seconds")
        if grid_seconds:
            from .canonical import decision_grid
            signal = np.asarray(signal, dtype=bool) & decision_grid(time_s, grid_seconds)
        bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
        ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
        ticks_bps = job.get("_shared_tick_size_bps")
        if ticks_bps is None:
            ticks_bps = tick_size_bps(arrays)
        day = job.get("_shared_day_profile")
        if day is None:
            day = day_profile(arrays, ticks_bps)
        n = len(time_s)
        horizon_seconds = job.get("horizon_seconds")
        exit_ticks = horizon_exit_ticks(
            time_s, seconds=PRIMARY_HORIZON_SECONDS if horizon_seconds is None else horizon_seconds)
        limit_entry = bool(job["limit_entry"])
        limit_exit = bool(job["limit_exit"])
        exit_rule = job.get("exit_rule")

        # ---- 진입 주문이 언제 체결되는가 -----------------------------------
        fill_ticks = order_terminal = None
        fill_status = queue_ahead = queue_left = None
        if limit_entry:
            posts = np.flatnonzero(signal)
            fill_ticks = np.full(n, -1, dtype=np.int64)
            order_terminal = np.arange(n, dtype=np.int64)
            fill_status = np.full(n, "", dtype=object)
            queue_ahead = np.full(n, np.nan, dtype=float)
            queue_left = np.full(n, np.nan, dtype=float)
            if len(posts):
                status, terminal, filled_at, ahead, left = fill.entry_fills(
                    arrays, posts, max_seconds=float(job["entry_max_rest_seconds"]),
                    max_ticks=int(job["entry_max_rest_ticks"]),
                    cancel_when_inactive=(entry_condition
                                          if job["cancel_when_entry_signal_false"] else None),
                    with_queue_state=True)
                fill_ticks[posts] = filled_at
                order_terminal[posts] = terminal
                fill_status[posts] = status
                # 게시 시점 내 앞 잔량과, 주문이 끝났을 때 남아 있던 양. 미체결의
                # 원인이 큐 길이인지 시간인지는 이 둘 없이 가를 수 없다.
                queue_ahead[posts] = ahead
                queue_left[posts] = left

        # ---- 청산이 언제 끝나는가 -------------------------------------------
        # **decision 을 만들기 전에** 푼다. 트리거 청산은 평가 창을 넘어갈 수 있고,
        # 그러면 평가 창 틱으로 다음 진입을 막는 것이 틀린다 — 포지션이 겹친다.
        resolved: dict[int, tuple[float | None, int | None, str]] = {}
        blocking = np.array(exit_ticks, dtype=np.int64, copy=True)
        for post in np.flatnonzero(signal):
            executed = int(fill_ticks[post]) if fill_ticks is not None else int(post)
            if executed < 0:
                continue
            entry = float(bid1[post]) if limit_entry else float(ask1[executed])
            outcome_of = _resolve_exit(arrays, time_s, bid1, ask1, n, executed, entry,
                                       int(exit_ticks[executed]), limit_exit, job,
                                       legacy_program=legacy_program)
            resolved[int(post)] = outcome_of
            _price, sell_tick, _reason = outcome_of
            if sell_tick is not None:
                # `decisions_from_signal` 은 **체결 틱** 자리를 읽는다. 게시 틱에 넣으면
                # 차단이 안 걸려 포지션이 겹친다.
                #
                # 여러 게시 틱이 같은 틱에 체결될 수 있다 (앞줄이 한 번에 빠질 때).
                # 진입가가 달라 청산도 달라지므로, 그중 **가장 늦게 끝나는 것**을 쓴다.
                # 마지막 게시 틱 값으로 덮어쓰면 실제로 선택된 것보다 일찍 풀린다.
                blocking[executed] = max(int(sell_tick), int(blocking[executed]))

        decisions = decisions_from_signal(
            signal, blocking, fill_ticks=fill_ticks, order_terminal=order_terminal)

        rows: list[dict[str, Any]] = []
        for record in decisions.itertuples(index=False):
            post = int(record.entry_tick)
            executed = int(record.fill_tick)
            row: dict[str, Any] = {
                "contract_id": part.contract_id, "symbol": part.symbol, "date": part.date,
                "decision_index": len(rows), "episode_index": int(record.episode_index),
                "entry_tick": post, "status": record.status,
                "fill_tick": executed, "exit_tick": int(record.exit_tick),
                "prior_reference_date": reference_date,
                "entry_signal_active_at_fill": None,
                "entry_signal_persisted_until_fill": None,
                "entry_signal_active_share_until_fill": None,
                "order_type": (
                    "limit_bid1_cancel_when_entry_signal_false"
                    if limit_entry and job["cancel_when_entry_signal_false"]
                    else "limit_bid1_hold_through" if limit_entry else "taker_ask1"),
                "entry_order_terminal_status": (
                    str(fill_status[post]) if fill_status is not None else "immediate_fill"),
            }
            if record.status == FILLED:
                row["entry_signal_active_at_fill"] = bool(entry_condition[executed])
                before_post = int(inactive_prefix[post - 1]) if post else 0
                inactive_until_fill = int(inactive_prefix[executed]) - before_post
                observed_order_ticks = executed - post + 1
                row["entry_signal_persisted_until_fill"] = bool(inactive_until_fill == 0)
                row["entry_signal_active_share_until_fill"] = float(
                    (observed_order_ticks - inactive_until_fill) / observed_order_ticks)
                # 지정가는 게시가(그 틱의 BID1)에 체결된다. taker 는 ASK1 즉시.
                entry_price = float(bid1[post]) if limit_entry else float(ask1[executed])
                sell_price, sell_tick, reason = resolved.get(
                    post, (None, None, "horizon_cap"))
                row["exit_reason_hint"] = reason

                result = outcome.path_outcome(arrays, executed, entry_price=entry_price,
                                              exit_price=sell_price, exit_tick=sell_tick)
                if result is None:
                    # 경로가 전부 결측. 관측 불가이므로 CENSORED 로 내린다.
                    row["status"] = CENSORED
                    row["fill_tick"] = row["exit_tick"] = -1
                else:
                    row["entry_price"] = result["entry_price"]
                    # 실현 손익은 **실제로 나간 가격**에서 온다. 트리거 청산이 평가 창을
                    # 넘어갈 수 있으므로 창 끝 값으로 대신하지 않는다.
                    if sell_price is not None:
                        net = (float(sell_price) / float(entry_price) - 1.0) * 1e4 - FEE_BPS
                    else:
                        net = result["net_bps"][PRIMARY_HORIZON_SECONDS]
                    row["net_bps"] = float(net)
                    row["gross_bps"] = float(net) + FEE_BPS
                    row["cohort"] = result["cohort"]
                    # 실제 청산을 포함한 `cohort` 는 장부 성과용이다. 진입 보완은
                    # 고정 청산이 만든 결과를 학습하지 않도록 날것의 30초 경로만 본다.
                    row["diagnostic_cohort"] = result["diagnostic_cohort"]
                    row["max_favorable_gross_bps"] = result["max_favorable_gross_bps"]
                    row["max_adverse_gross_bps"] = result["max_adverse_gross_bps"]
                    row["early_max_net_bps"] = result["early_max_net_bps"]
                    row["exit_reason"] = row.pop("exit_reason_hint", None) or "horizon_cap"
                    if sell_tick is not None:
                        row["exit_tick"] = int(sell_tick)
                        row["holding_seconds"] = float(
                            time_s[min(int(sell_tick), n - 1)] - time_s[executed])
                    if exit_rule is not None:
                        # 규칙 청산은 평가 창보다 일찍 끝날 수 있고, 그러면 다음 진입이
                        # 그만큼 빨리 가능해진다. 보유시간 변화가 거래 수를 바꾼다.
                        cap = min(int(record.exit_tick), n - 1)
                        tick, net, reason = exits.apply(
                            bid1, executed, cap, entry_price, exit_rule, features=features)
                        row["exit_tick"] = tick
                        row["net_bps"] = net
                        row["gross_bps"] = net + FEE_BPS
                        row["exit_reason"] = reason
                    if job.get("exit_crossings"):
                        # 청산 후보 격자가 원장만 읽고 채점할 수 있게 미리 남긴다.
                        row.update(exits.first_crossings(
                            bid1, executed, int(record.exit_tick), result["entry_price"]))
                    # ---- 경로 분해: 매수측과 매도측을 따로 ------------------
                    # mid 는 쓰지 않는다. 아무도 그 가격에 거래하지 않고, 성격이 다른
                    # 두 큐를 하나로 뭉갠다. 지정가 매수는 bid 쪽에서 사서 ask 쪽으로
                    # 나가므로 둘을 갈라 봐야 무슨 일이 있었는지 말할 수 있다.
                    #
                    # 기준점이 셋이다 — 게시(post) · 체결(fill) · 청산(exit).
                    # 게시가와 체결 시점 BID1 의 차이가 역선택이 사는 자리다. 음수면
                    # 시장이 내려간 뒤에 내 주문이 체결됐다는 뜻이다.
                    #
                    #   BID1 지정가 진입 기준 (정본):
                    #     ASK1 지정가 청산  gross ~= fill_slippage + entry_spread + ask_move
                    #                            ~= fill_slippage + bid_move + exit_spread
                    #     BID1 시장가 청산  gross ~= fill_slippage + bid_move
                    #
                    # bps 단위에서 교차항이 작아 근사식이다. taker 진입은 매수가가
                    # ASK1 이라 이 항등식이 성립하지 않는다. 열 자체는 그대로 경로 서술이다.
                    exit_mark_tick = min(int(row["exit_tick"]), n - 1)
                    b_post, b_fill = float(bid1[post]), float(bid1[executed])
                    b_exit = float(bid1[exit_mark_tick])
                    a_fill, a_exit = float(ask1[executed]), float(ask1[exit_mark_tick])
                    if all(np.isfinite(value) and value > 0.0
                           for value in (b_post, b_fill, b_exit, a_fill, a_exit)):
                        row["fill_slippage_bps"] = (b_fill / b_post - 1.0) * 1e4
                        row["bid_move_bps"] = (b_exit / b_fill - 1.0) * 1e4
                        row["ask_move_bps"] = (a_exit / a_fill - 1.0) * 1e4
                        row["entry_spread_bps"] = (a_fill / b_fill - 1.0) * 1e4
                        row["exit_spread_bps"] = (a_exit / b_exit - 1.0) * 1e4
            elif record.status == UNFILLED:
                # 못 산 결정. 지정가가 안 걸렸으니 실현 손익은 없다. 대신 **내가 건
                # 그 가격(게시 시점 BID1)에 큐가 빠졌다면** 을 남긴다. 가격은 실제로
                # 게시한 값 그대로이고 가정하는 것은 체결 하나뿐이다.
                #
                # 시장가(ASK1) 기준으로 물으면 스프레드 한 폭을 얹고 시작해 거의 항상
                # 음수가 나오고, 애초에 아무도 돌리지 않을 전략을 기준선으로 삼는 것이다.
                # 지정가 전략의 반사실은 "체결됐다면" 이지 "시장가로 샀다면" 이 아니다.
                #
                # 이것은 관측이 아니라 **가정**이다. 그래서 열 이름을 완전히 분리한다.
                # `metrics.measure` 는 status==FILLED 의 net_bps 만 더하므로 총합에
                # 섞이지 않는다.
                counterfactual = outcome.path_outcome(
                    arrays, post, entry_price=float(bid1[post]))
                if counterfactual is not None:
                    row["counterfactual_net_bps"] = float(
                        counterfactual["net_bps"][PRIMARY_HORIZON_SECONDS])
                    row["counterfactual_cohort"] = counterfactual["diagnostic_cohort"]
                    row["counterfactual_max_favorable_gross_bps"] = float(
                        counterfactual["max_favorable_gross_bps"])
                    row["counterfactual_max_adverse_gross_bps"] = float(
                        counterfactual["max_adverse_gross_bps"])
            row["tick_size_bps"] = float(ticks_bps[post])
            row.update(day)
            if queue_ahead is not None:
                ahead = float(queue_ahead[post])
                left = float(queue_left[post])
                total = ahead + fill.DEFAULT_ORDER_QTY
                row["queue_ahead_at_post"] = ahead
                row["queue_remaining_at_end"] = left
                row["queue_consumed_share"] = (
                    float(min(max((total - left) / total, 0.0), 1.0))
                    if np.isfinite(left) and total > 0.0 else None)
            row.update({name: float(values[post]) for name, values in features.items()})
            row.update({name: float(values[post]) for name, values in thresholds.items()})
            rows.append(row)

        counts = {
            **part.__dict__,
            "raw_signal_ticks": int(decisions.attrs.get("raw_signal_ticks", 0)),
            "signal_episodes": int(decisions.attrs.get("signal_episodes", 0)),
            "blocked_signal_ticks": int(decisions.attrs.get("blocked_signal_ticks", 0)),
            "decisions": len(rows),
            "filled": sum(1 for r in rows if r["status"] == FILLED),
            "censored": sum(1 for r in rows if r["status"] == CENSORED),
            "unfilled": sum(1 for r in rows if r["status"] == UNFILLED),
            "reference_date": reference_date,
            "legacy_thresholds": legacy_threshold_detail,
            "source": str(path),
            "source_identity": tickdata.source_identity(path),
        }
        return {"rows": rows, "counts": counts, "missing": None}
    except Exception as error:  # noqa: BLE001 - 한 종목-일의 실패가 원장을 멈추지 않는다
        return {"rows": [], "counts": None,
                "error": {**part.__dict__, "error": f"{type(error).__name__}: {error}"}}


def _one_symbol_day_group(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """같은 (종목, 날) 계약 묶음. 원본 호가와 guard feature는 한 번만 읽는다.

    계약별 signal·전일 threshold·체결 원장은 여전히 `_one_partition`에서 완전히
    분리한다. 공유하는 것은 바뀌지 않는 입력 배열뿐이다.
    """
    if not jobs:
        return {"results": [], "runtime": {"contracts": 0}}
    first = jobs[0]
    symbol, date, root = str(first["symbol"]), str(first["date"]), Path(first["root"])
    started = perf_counter()
    try:
        arrays, path = tickdata.load(symbol, date, root)
    except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
        return {
            "results": [{"rows": [], "counts": None,
                         "missing": {**Partition(job["contract_id"], symbol, date).__dict__,
                                     "reason": str(error)}} for job in jobs],
            "runtime": {"contracts": len(jobs), "load_seconds": perf_counter() - started,
                        "feature_seconds": 0.0, "execution_seconds": 0.0},
        }
    except Exception as error:  # noqa: BLE001 - 동일 원본 오류는 계약별로 남긴다
        return {
            "results": [{"rows": [], "counts": None,
                         "error": {**Partition(job["contract_id"], symbol, date).__dict__,
                                   "error": f"{type(error).__name__}: {error}"}} for job in jobs],
            "runtime": {"contracts": len(jobs), "load_seconds": perf_counter() - started,
                        "feature_seconds": 0.0, "execution_seconds": 0.0},
        }

    loaded = perf_counter()
    try:
        shared_features = catalog.compute_features(
            catalog.list_features(), catalog.Book(arrays))
    except Exception as error:  # noqa: BLE001 - 기존처럼 각 계약의 실행 오류로 기록한다
        return {
            "results": [{"rows": [], "counts": None,
                         "error": {**Partition(job["contract_id"], symbol, date).__dict__,
                                   "error": f"{type(error).__name__}: {error}"}} for job in jobs],
            "runtime": {"contracts": len(jobs), "load_seconds": loaded - started,
                        "feature_seconds": perf_counter() - loaded, "execution_seconds": 0.0},
        }
    featured = perf_counter()
    try:
        shared_tick_size_bps = tick_size_bps(arrays)
        shared_day_profile = day_profile(arrays, shared_tick_size_bps)
    except (KeyError, ValueError, IndexError):
        # 호가 배열이 없는 원본은 계약별 경로가 같은 오류를 원장에 기록한다.
        # 공유 계산 실패가 그룹 전체를 멈추지 않는다.
        shared_tick_size_bps = shared_day_profile = None
    all_guards = [guard for job in jobs for guard in job["guards"]]
    try:
        shared_guard_thresholds = (rolling_tick_thresholds(shared_features, guards=all_guards)
                                   if all_guards else None)
    except (KeyError, ValueError):
        # 공유 계산이 불가능하면 기존 계약별 경로가 해당 오류를 원장에 기록한다.
        shared_guard_thresholds = None
    # Parameter Search의 q 후보는 entry program만 같고 q만 다르다. 같은 입력 feature와
    # 직전 100 tick 순위는 여기서 한 번만 만든다. q별 bool signal·체결·청산은 독립이다.
    shared_q: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    q_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, job in enumerate(jobs):
        template = job["contract"]
        if template.get("legacy_program") is not None or not contract.uses_rolling_quantiles(template):
            continue
        key = sha256_json({
            "entry_program": template.get("entry_program"),
            "parameter_interface": template.get("parameter_interface"),
        })
        q_groups.setdefault(key, []).append((index, job))
    for group in q_groups.values():
        if len(group) < 2:
            continue
        first_template = group[0][1]["contract"]
        try:
            thresholds, expression_cache = contract.rolling_quantile_binding_grid(
                arrays, first_template["entry_program"],
                [dict(item.get("parameters") or {}) for _index, item in group])
        except (KeyError, ValueError):
            # 어떤 후보가 q를 풀 수 없는 경우는 기존 partition 경로가 계약별 missing/error로
            # 기록한다. 공유 최적화가 결과 상태를 바꾸면 안 된다.
            continue
        for (index, _job), threshold in zip(group, thresholds):
            shared_q[index] = (threshold, expression_cache)

    results = []
    for index, job in enumerate(jobs):
        shared_job = dict(job)
        shared_job["_shared_arrays"] = arrays
        shared_job["_shared_path"] = str(path)
        shared_job["_shared_features"] = shared_features
        if shared_tick_size_bps is not None:
            shared_job["_shared_tick_size_bps"] = shared_tick_size_bps
            shared_job["_shared_day_profile"] = shared_day_profile
        if shared_guard_thresholds is not None:
            shared_job["_shared_guard_thresholds"] = shared_guard_thresholds
        if index in shared_q:
            thresholds, expression_cache = shared_q[index]
            shared_job["_shared_rolling_thresholds"] = thresholds
            shared_job["_shared_entry_expression_cache"] = expression_cache
        results.append(_one_partition(shared_job))
    finished = perf_counter()
    return {
        "results": results,
        "runtime": {"contracts": len(jobs), "load_seconds": loaded - started,
                    "feature_seconds": featured - loaded,
                    "execution_seconds": finished - featured},
    }


def build_ledger(
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    members: Mapping[str, Sequence[str]],
    dates: Sequence[str],
    output: Path,
    workers: int = 8,
    guards: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    reference_dates: Sequence[str] = (),
    exit_crossings: bool = False,
    exit_rule: Mapping[str, Any] | None = None,
    limit_entry: bool = LIMIT_ENTRY,
    limit_exit: bool = LIMIT_EXIT,
    decision_grid_seconds: float | None = None,
    entry_max_rest_seconds: float = ENTRY_ORDER_MAX_SECONDS,
    entry_max_rest_ticks: int = ENTRY_ORDER_MAX_TICKS,
    parameter_table: Mapping[str, Mapping[str, float]] | None = None,
    stop_gross_bps: float | None = None,
    trailing_drawdown_gross_bps: float | None = None,
    exit_limit_rest_seconds: float | None = None,
    horizon_seconds: float | None = None,
    root: Path = TICK_ROOT,
) -> dict[str, Any]:
    """계약들을 지정한 날짜에 돌려 원장 parquet 을 쓴다. manifest 를 돌려준다.

    한 행 = 한 decision. BLOCKED 틱은 행으로 남기지 않고 partition 마다 개수만 센다.
    구 원장은 BLOCKED 틱까지 전부 행으로 남겨 22M 행이 됐고, 그 행 수가 곧 분모가
    되면서 문제가 시작됐다.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if horizon_seconds is not None and float(horizon_seconds) <= 0.0:
        raise ValueError("horizon_seconds는 양수여야 한다")
    if float(entry_max_rest_seconds) <= 0.0 or int(entry_max_rest_ticks) <= 0:
        raise ValueError("entry 주문 대기 시간과 tick은 양수여야 한다")
    jobs = [
        {"contract_id": key, "contract": dict(value), "symbol": symbol, "date": date,
         # Refinement가 고른 진입 조항은 계약의 일부다. 호출자가 임시로 주는 guard와
         # 합쳐 같은 원장 엔진에서만 적용한다.
         "guards": [*list(value.get("entry_guards") or []),
                    *list((guards or {}).get(key, []))], "exit_crossings": bool(exit_crossings),
         "exit_rule": dict(exit_rule) if exit_rule else None,
         "limit_entry": bool(limit_entry), "limit_exit": bool(limit_exit),
         "decision_grid_seconds": (None if decision_grid_seconds is None
                                   else float(decision_grid_seconds)),
         "entry_max_rest_seconds": float(entry_max_rest_seconds),
         "entry_max_rest_ticks": int(entry_max_rest_ticks),
         # batch refinement는 같은 종목-일에도 q 계약마다 다른 직전 100 tick 임계값을 쓴다.
         # 계약별 값을 먼저 읽고, 기존 단일 계약 표는 종목-일 값으로 계속 읽는다.
         "parameters": dict((parameter_table or {}).get(
             f"{key}:{symbol}:{date}", (parameter_table or {}).get(f"{symbol}:{date}", {}))),
         "stop_gross_bps": (None if stop_gross_bps is None else float(stop_gross_bps)),
         "trailing_drawdown_gross_bps": (
             None if trailing_drawdown_gross_bps is None else float(trailing_drawdown_gross_bps)),
         "exit_limit_rest_seconds": (None if exit_limit_rest_seconds is None
                                     else float(exit_limit_rest_seconds)),
         "horizon_seconds": (None if horizon_seconds is None else float(horizon_seconds)),
         "cancel_when_entry_signal_false": bool(
             (value.get("entry_lifecycle") or {}).get("mode")
             == "CANCEL_WHEN_ENTRY_SIGNAL_FALSE"),
         "reference_dates": list(reference_dates), "root": str(root)}
        for key, value in contracts.items()
        for symbol in members.get(key, ())
        for date in dates
    ]

    # 같은 (symbol, date)는 계약만 달라도 원본 호가와 guard feature가 같다.
    # 특히 Refinement exact batch처럼 계약이 많은 재생에서 이를 따로 읽으면 I/O와
    # worker 시작 비용만 계약 수만큼 반복된다. 계약별 원장 계산은 아래에서도 분리한다.
    grouped_jobs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        grouped_jobs.setdefault((str(job["symbol"]), str(job["date"])), []).append(job)

    rows: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    group_runtime: list[dict[str, Any]] = []
    started = perf_counter()

    def collect_group(result: Mapping[str, Any]) -> None:
        for item in result.get("results") or []:
            _collect(item, rows, counts, missing, errors)
        if isinstance(result.get("runtime"), Mapping):
            group_runtime.append(dict(result["runtime"]))

    if workers <= 1:
        for group in grouped_jobs.values():
            # 단일 계약은 기존 호출 경로를 보존한다. 테스트 mock과 단일 backtest의
            # 결과 형식도 그대로 두면서, 실제 batch에만 공유 입력을 적용한다.
            if len(group) == 1:
                _collect(_one_partition(group[0]), rows, counts, missing, errors)
            else:
                collect_group(_one_symbol_day_group(group))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_symbol_day_group, group)
                       for group in grouped_jobs.values()]
            for future in as_completed(futures):
                collect_group(future.result())
    elapsed = perf_counter() - started

    frame = pd.DataFrame(rows)
    # 신호가 없거나 전부 데이터·rolling q 부재면 rows가 비어도 된다. 다만 parquet이
    # 열 없는 표로 저장되면 다음 Stage가 contract_id·status를 읽다가 죽는다. 0행도
    # 보통 원장과 같은 schema로 남겨서 '거래 없음'으로 처리하게 한다.
    frame = frame.reindex(columns=tuple(dict.fromkeys([*frame.columns, *_ledger_columns()])))
    if len(frame):
        frame = frame.sort_values(["contract_id", "symbol", "date", "entry_tick"]).reset_index(drop=True)
    frame.to_parquet(output, index=False)

    totals = pd.DataFrame(counts)
    legacy_state_exit = any(
        bool((value.get("legacy_program") or {}).get("exit", {}).get("conditions"))
        for value in contracts.values())
    manifest = {
        "schema": "framework_decision_ledger.v1",
        "created_at": now_utc(),
        "output": str(output),
        "dates": list(dates),
        "reference_dates": list(reference_dates),
        "contracts": sorted(contracts),
        "entry_execution": (
            "taker_ask1" if not limit_entry else
            "limit_bid1_cancel_when_entry_signal_false"
            if jobs and all(job["cancel_when_entry_signal_false"] for job in jobs) else
            "limit_bid1_hold_through"
            if not any(job["cancel_when_entry_signal_false"] for job in jobs) else
            "limit_bid1_mixed_lifecycle"),
        "decision_grid_seconds": (None if decision_grid_seconds is None
                                  else float(decision_grid_seconds)),
        "entry_max_rest_seconds": float(entry_max_rest_seconds),
        "entry_max_rest_ticks": int(entry_max_rest_ticks),
        "stop_gross_bps": (None if stop_gross_bps is None else float(stop_gross_bps)),
        "trailing_drawdown_gross_bps": (
            None if trailing_drawdown_gross_bps is None else float(trailing_drawdown_gross_bps)),
        "exit_limit_rest_seconds": (None if exit_limit_rest_seconds is None
                                    else float(exit_limit_rest_seconds)),
        "horizon_seconds": (None if horizon_seconds is None else float(horizon_seconds)),
        "exit_execution": (
            "legacy_state_or_canonical_profile_first_trigger"
            if legacy_state_exit
            else ("exit_rule" if exit_rule else
                  ("limit_ask1" if limit_exit else "horizon_bid1"))),
        "exit_rule": dict(exit_rule) if exit_rule else None,
        "legacy_exit_overlay": {
            "enabled": legacy_state_exit,
            "rule": "legacy state exit and canonical default exit race by first trigger; canonical wins ties",
            "legacy_action": "BID1 market exit",
        },
        "legacy_programs": {
            key: dict(value["legacy_program"])
            for key, value in contracts.items() if value.get("legacy_program") is not None
        },
        "exit_crossings": bool(exit_crossings),
        "rows": int(len(frame)),
        "partitions": len(counts),
        "runtime": {
            "wall_seconds": elapsed,
            "execution_workers": int(workers),
            "contract_symbol_day_jobs": len(jobs),
            "symbol_day_groups": len(grouped_jobs),
            "raw_loads_avoided": len(jobs) - len(grouped_jobs),
            "group_load_seconds": sum(float(item.get("load_seconds") or 0.0)
                                      for item in group_runtime),
            "group_feature_seconds": sum(float(item.get("feature_seconds") or 0.0)
                                         for item in group_runtime),
            "group_execution_seconds": sum(float(item.get("execution_seconds") or 0.0)
                                           for item in group_runtime),
        },
        "totals": {k: int(totals[k].sum()) for k in
                   ("raw_signal_ticks", "signal_episodes", "blocked_signal_ticks",
                    "decisions", "filled", "censored", "unfilled")} if len(totals) else {},
        "by_partition": counts,
        "missing": missing,
        "errors": errors,
        **catalog.provenance(),
        # 원장 컬럼 하나하나가 어느 계산 정의로 만들어졌는지. 이것이 없으면
        # 나중에 "그때 그 book_imbalance" 가 무엇이었는지 말할 수 없다.
        "feature_definitions": catalog.definition_ids(catalog.list_features()),
    }
    write_json(output.with_name(output.stem + "_manifest.json"), manifest)
    verify_invariants(frame, manifest)
    return manifest


def _collect(result, rows, counts, missing, errors) -> None:
    rows.extend(result["rows"])
    if result.get("counts"):
        counts.append(result["counts"])
    if result.get("missing"):
        missing.append(result["missing"])
    if result.get("error"):
        errors.append(result["error"])


class InvariantViolation(AssertionError):
    """원장 회계가 스스로 모순된다. 여기서 막지 않으면 분모가 조용히 틀린다."""


def verify_invariants(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    """원장이 회계 불변식을 지키는지 검사한다. 어기면 예외."""
    problems: list[str] = []
    taker = manifest.get("entry_execution") == "taker_ask1"

    # 모르는 상태가 섞이면 그 행이 어느 분모에 들어가는지 말할 수 없다.
    if len(frame) and "status" in frame:
        unknown = sorted(set(frame["status"].astype(str)) - set(STATUSES))
        if unknown:
            problems.append(f"모르는 status 가 있다: {unknown}. 쓸 수 있는 것 "
                            f"{sorted(STATUSES)}")

    # manifest 의 합계가 분할 기록과 맞는가. 여기가 어긋나면 보고된 분모가 거짓이 된다.
    parts = manifest.get("by_partition") or []
    totals = manifest.get("totals") or {}
    for key in ("decisions", "filled", "censored", "unfilled",
                "raw_signal_ticks", "signal_episodes", "blocked_signal_ticks"):
        if key not in totals:
            continue
        summed = sum(int(p.get(key, 0)) for p in parts)
        if int(totals[key]) != summed:
            problems.append(f"totals.{key} 가 {totals[key]} 인데 분할 합은 {summed} 다")

    for part in parts:
        if part["decisions"] != part["filled"] + part["censored"] + part["unfilled"]:
            problems.append(f"{part['symbol']}/{part['date']}: decisions 가 상태 합과 다르다")
        if taker and part["unfilled"]:
            problems.append(f"{part['symbol']}/{part['date']}: taker 인데 UNFILLED 가 있다")

    if len(frame):
        observed = frame.groupby(["contract_id", "symbol", "date"]).size()
        declared = {(p["contract_id"], p["symbol"], p["date"]): p["decisions"]
                    for p in manifest.get("by_partition", [])}
        for key, size in observed.items():
            if declared.get(key) != int(size):
                problems.append(f"{key}: manifest decisions {declared.get(key)} != 행 수 {size}")
        # 체결 구간이 겹치지 않는가
        filled = frame.loc[frame["status"].eq(FILLED)]
        for key, group in filled.groupby(["contract_id", "symbol", "date"], sort=False):
            group = group.sort_values("entry_tick")
            if (group["entry_tick"].to_numpy()[1:] <= group["exit_tick"].to_numpy()[:-1]).any():
                problems.append(f"{key}: 체결 구간이 겹친다")
    if problems:
        raise InvariantViolation("원장 불변식 위반 " + str(len(problems)) + "건:\n  - "
                                 + "\n  - ".join(problems[:20]))
