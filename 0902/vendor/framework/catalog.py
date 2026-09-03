"""금융 어휘의 단일 정본. feature 와 연산자가 여기에만 있다.

구 구조에서 feature 이름은 다섯 곳에 흩어져 있었다 — 카탈로그 JSON, `catalog.py` 의
하드코딩 primitive 3개, `contract.py` 의 if/elif 14분기, `data.py` 의 계산식,
`ledger.py` 의 `FEATURE_COLUMNS` 19개. 어느 하나를 빠뜨려도 에러가 나지 않고 조용히
어긋났다 (실측: 계약 어휘 23개와 격자 축 19개 중 15개만 겹쳤다).

여기서는 **이름·타입·계산 함수·원본 의존·시간 의미가 한 행**이다. 행이 없으면 그
feature 는 존재하지 않는다.

## 카탈로그가 정하는 것과 정하지 않는 것

카탈로그는 **관측하고 계산하고 조합할 수 있는 것이 무엇인가** 만 정한다. 어떤 가설에
어떤 feature 가 맞는지, 임계가 얼마인지, 어떤 계약이 수익성 있는지는 정하지 않는다.

    카탈로그   "무엇을 관측하고 표현할 수 있는가"
    Grounding  "이 가설은 그 어휘로 무슨 뜻인가"
    Search     "허용된 구현 중 무엇이 잘 되는가"

## 관측 사실과 경제적 해석을 나눈다

`observable_description` 은 실제로 계산되는 값이고, `economic_interpretation` 은
그럴 수도 있다는 해석이다. 둘을 한 문장에 섞으면 이름부터 해석을 주장하게 된다.
그래서 구 이름 두 개를 바꿨다.

  · `bid_queue_depletion_5s_bps` → `bid_best_price_drop_count_50t`
  · `ask_queue_arrival_count_5s` → `ask_best_price_drop_count_50t`

단위가 bps 가 아니라 개수였고, 100ms 등간격이 보장되지 않으므로 50틱은 5초가 아니며,
큐 소진·주문 도착을 직접 관측하지도 않는다. 구 이름은 과거 계약을 읽기 위한 별칭으로
남기되 Agent 어휘에는 싣지 않는다.

## 정의와 관측은 다른 객체다

`Feature` 는 정의이고 `ObservationProfile` 은 이번 데이터에서 그 값이 어땠는가이다.
데이터가 바뀌면 profile 이 바뀌지 정의는 바뀌지 않는다. 마찬가지로 어느 workflow 가
어느 feature 를 쓸 수 있는가는 `WorkflowPolicy` 로 빼냈다.

## 타입

표현식은 문법뿐 아니라 타입으로도 검사한다. `all(mid_price, ...)` 같은 것은 실행
전에 결정적으로 실패한다. 출력 단위는 Agent 가 적지 않고 입력과 연산자에서 추론한다.

`value_type` 은 `numeric` · `boolean` · `event` 셋뿐이다. price·quantity 는 값의
종류가 아니라 차원이므로 `dimension` 에 둔다 — 한 뜻을 두 곳에 적지 않는다.
"""

from __future__ import annotations

import dataclasses

import numbers
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import FEE_BPS, sha256_json


EPS = 1.0e-12
DROP_COUNT_WINDOW = 50   # 틱 수. 등간격이 아니므로 초로 환산하지 않는다
SESSION_OPEN_SECONDS = 9 * 60 * 60

# 원본 배열의 어휘. `data.load()` 가 내놓는 키가 전부다.
RAW_FIELDS = frozenset({
    "bid_price", "ask_price", "bid_qty", "ask_qty",
    "time_s", "local_time",
    "buy_volume", "sell_volume", "buy_max_price", "sell_min_price",
})

BOOK_L1 = ("bid_price", "ask_price", "bid_qty", "ask_qty")
TRADES = ("buy_volume", "sell_volume")

VALUE_TYPES = frozenset({"numeric", "boolean", "event"})
DIMENSIONS = frozenset({
    "price", "return", "quantity", "dimensionless", "price_per_quantity", "count", "boolean",
})
TIME_BASES = frozenset({"tick", "event", "clock"})
NORMALIZATION_SCOPES = frozenset({"global", "symbol", "symbol_day", "not_comparable"})


# ---- 타입 ---------------------------------------------------------------------

@dataclass(frozen=True)
class TypeInfo:
    """표현식 한 노드가 내는 값의 종류. 단위는 표시용이고 검사는 차원으로 한다."""

    value_type: str
    dimension: str
    unit: str


BOOLEAN = TypeInfo("boolean", "boolean", "0/1")
EVENT = TypeInfo("event", "boolean", "0/1")
ZSCORE = TypeInfo("numeric", "dimensionless", "sigma")
DERIVED_NUMERIC = TypeInfo("numeric", "dimensionless", "derived")

RAW_FIELD_TYPES = {
    "bid_price": TypeInfo("numeric", "price", "KRW"),
    "ask_price": TypeInfo("numeric", "price", "KRW"),
    "bid_qty": TypeInfo("numeric", "quantity", "quantity"),
    "ask_qty": TypeInfo("numeric", "quantity", "quantity"),
    "buy_volume": TypeInfo("numeric", "quantity", "quantity"),
    "sell_volume": TypeInfo("numeric", "quantity", "quantity"),
    "buy_max_price": TypeInfo("numeric", "price", "KRW"),
    "sell_min_price": TypeInfo("numeric", "price", "KRW"),
    "time_s": TypeInfo("numeric", "count", "seconds"),
    "local_time": TypeInfo("numeric", "count", "HHMMSSuuuuuu"),
}


def _is_numeric(info: TypeInfo) -> bool:
    return info.value_type == "numeric"


def _is_integer(value: Any) -> bool:
    """구조 인자가 정수인가. bool 은 정수가 아니다 — 실수로 넣은 것이다."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_boolean(info: TypeInfo) -> bool:
    # event 는 순간에만 참인 bool 이다. bool 을 받는 자리에 그대로 들어간다.
    return info.value_type in ("boolean", "event")


# ---- 데이터가 무엇을 담고 있는가 ----------------------------------------------

@dataclass(frozen=True)
class DataCapabilities:
    """이 데이터셋으로 무엇을 계산할 수 있는가. feature 가용성을 여기서 판정한다."""

    fields: frozenset[str]
    book_depth: int
    has_trade_classification: bool
    has_timestamp: bool
    timestamp_resolution: str | None

    @classmethod
    def from_arrays(cls, data: Mapping[str, np.ndarray]) -> "DataCapabilities":
        depth = int(np.asarray(data["bid_price"]).shape[1]) if "bid_price" in data else 0
        return cls(
            fields=frozenset(str(k) for k in data),
            book_depth=depth,
            has_trade_classification=("buy_volume" in data and "sell_volume" in data),
            has_timestamp="time_s" in data,
            timestamp_resolution="microsecond" if "local_time" in data else None,
        )


# ---- 호가에서 매번 다시 만들던 중간값 -----------------------------------------

class Book:
    """한 종목-일의 공통 중간값. 구 코드는 분기마다 이 8줄을 다시 썼다."""

    __slots__ = ("data", "n", "bid_price", "ask_price", "bid_qty", "ask_qty",
                 "bid1", "ask1", "mid", "l1_total", "_microprice", "_capabilities")

    def __init__(self, data: Mapping[str, np.ndarray]) -> None:
        self.data = data
        self.bid_price = np.asarray(data["bid_price"], dtype=float)
        self.ask_price = np.asarray(data["ask_price"], dtype=float)
        raw_bid_qty = np.asarray(data["bid_qty"], dtype=float)
        raw_ask_qty = np.asarray(data["ask_qty"], dtype=float)
        # 음수·비유한 잔량은 0 으로. 그대로 두면 합이 오염된다.
        self.bid_qty = np.where(np.isfinite(raw_bid_qty) & (raw_bid_qty >= 0), raw_bid_qty, 0.0)
        self.ask_qty = np.where(np.isfinite(raw_ask_qty) & (raw_ask_qty >= 0), raw_ask_qty, 0.0)
        self.n = len(self.bid_price)
        self.bid1 = self.bid_price[:, 0]
        self.ask1 = self.ask_price[:, 0]
        self.mid = (self.bid1 + self.ask1) / 2.0
        self.l1_total = self.bid_qty[:, 0] + self.ask_qty[:, 0]
        self._microprice: np.ndarray | None = None
        self._capabilities: DataCapabilities | None = None

    @property
    def microprice(self) -> np.ndarray:
        """잔량 가중 중간가. 잔량이 없으면 mid 로 둔다."""
        if self._microprice is None:
            self._microprice = np.divide(
                self.ask1 * self.bid_qty[:, 0] + self.bid1 * self.ask_qty[:, 0],
                self.l1_total, out=self.mid.copy(), where=self.l1_total > 0)
        return self._microprice

    @property
    def capabilities(self) -> DataCapabilities:
        if self._capabilities is None:
            self._capabilities = DataCapabilities.from_arrays(self.data)
        return self._capabilities

    @property
    def has_trades(self) -> bool:
        return self.capabilities.has_trade_classification

    def empty(self, fill: float = np.nan) -> np.ndarray:
        return np.full(self.n, fill, dtype=float)


# ---- 계산 도우미 --------------------------------------------------------------

def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """과거 `window` 틱(현재 포함)의 합. 앞부분은 있는 만큼만."""
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    n = len(values)
    start = np.maximum(np.arange(n) + 1 - int(window), 0)
    return cumulative[np.arange(n) + 1] - cumulative[start]


def rolling_zscore(values: np.ndarray, *, window: int, min_observations: int) -> np.ndarray:
    """인과적 z-score. 현재 틱을 포함하고 이후는 보지 않는다."""
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(filled)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(filled * filled)))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    indices = np.arange(len(x), dtype=np.int64)
    starts = np.maximum(0, indices - int(window) + 1)
    count = counts[indices + 1] - counts[starts]
    safe = np.maximum(count, 1)
    mean = (prefix[indices + 1] - prefix[starts]) / safe
    variance = np.maximum((prefix_sq[indices + 1] - prefix_sq[starts]) / safe - mean * mean, 0.0)
    usable = valid & (count >= int(min_observations)) & (variance > EPS)
    out = np.zeros(len(x), dtype=float)
    out[usable] = (x[usable] - mean[usable]) / np.sqrt(variance[usable])
    return out


def depth_ofi(book: Book, depth: int) -> np.ndarray:
    """가격 조건부 CKS OFI. 레벨마다 독립적으로 유효성을 본다."""
    n = book.n
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result
    for level in range(depth):
        bp_now, bp_prior = book.bid_price[1:, level], book.bid_price[:-1, level]
        ap_now, ap_prior = book.ask_price[1:, level], book.ask_price[:-1, level]
        bq_now, bq_prior = book.bid_qty[1:, level], book.bid_qty[:-1, level]
        aq_now, aq_prior = book.ask_qty[1:, level], book.ask_qty[:-1, level]
        bid_valid = np.isfinite(bp_now) & np.isfinite(bp_prior) & (bp_now > 0) & (bp_prior > 0)
        ask_valid = np.isfinite(ap_now) & np.isfinite(ap_prior) & (ap_now > 0) & (ap_prior > 0)
        bid_event = np.zeros(n - 1, dtype=float)
        ask_event = np.zeros(n - 1, dtype=float)
        bid_event[bid_valid] = np.where(
            bp_now[bid_valid] > bp_prior[bid_valid], bq_now[bid_valid],
            np.where(bp_now[bid_valid] < bp_prior[bid_valid], -bq_prior[bid_valid],
                     bq_now[bid_valid] - bq_prior[bid_valid]))
        # 매도 호가가 사라지거나 올라가면 양수. CKS 규약대로 매수측에서 뺀다.
        ask_event[ask_valid] = np.where(
            ap_now[ask_valid] < ap_prior[ask_valid], aq_now[ask_valid],
            np.where(ap_now[ask_valid] > ap_prior[ask_valid], -aq_prior[ask_valid],
                     aq_now[ask_valid] - aq_prior[ask_valid]))
        result[1:] += bid_event - ask_event
    return result


def _tick_diff(values: np.ndarray, n: int) -> np.ndarray:
    """직전 틱과의 차. 첫 틱은 직전이 없으므로 NaN 이다.

    구 구현은 첫 틱을 0.0 으로 뒀는데, 0 은 "변화가 없었다" 는 관측을 지어내는 것이고
    `difference(lag=1)` 연산자와도 어긋났다. 정의와 구현이 같아야 §11 의 parity 가
    성립한다.
    """
    out = np.full(n, np.nan)
    if n > 1:
        valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        indices = np.flatnonzero(valid) + 1
        out[indices] = values[indices] - values[indices - 1]
    return out


def _lagged_return_bps(mid: np.ndarray, lag: int, n: int) -> np.ndarray:
    """`lag` 틱 전 대비 수익률(bps). 앞부분은 첫 값 기준으로 채운다."""
    out = np.full(n, np.nan)
    if n and mid[0] > 0:
        before = min(lag, n)
        out[:before] = (mid[:before] / mid[0] - 1.0) * 10_000.0
    if n > lag:
        valid = np.isfinite(mid[lag:]) & np.isfinite(mid[:-lag]) & (mid[:-lag] > 0)
        indices = np.flatnonzero(valid) + lag
        out[indices] = (mid[indices] / mid[indices - lag] - 1.0) * 10_000.0
    return out


def _drop_events(prices: np.ndarray) -> np.ndarray:
    """직전 틱보다 내려온 event 표시. 직전 값이 0이면 세지 않는다."""
    p = np.asarray(prices, dtype=float)
    out = np.zeros(len(p), dtype=float)
    if len(p) > 1:
        out[1:] = ((p[1:] < p[:-1]) & (p[:-1] > 0)).astype(float)
    return out


def _book_slope(book: Book, side: str) -> np.ndarray:
    """1~5호가의 가격 폭을 잔량으로 나눈 값. 호가창이 얼마나 가파른가."""
    price = book.bid_price if side == "bid" else book.ask_price
    qty = book.bid_qty if side == "bid" else book.ask_qty
    numerator = price[:, 0] - price[:, 4] if side == "bid" else price[:, 4] - price[:, 0]
    denominator = qty[:, :5].sum(axis=1)
    valid = np.isfinite(numerator) & (price[:, 4] > 0) & (denominator > 0)
    return np.divide(numerator, denominator, out=book.empty(), where=valid)


def _depth_total(book: Book, side: str, depth: int) -> np.ndarray:
    qty = book.bid_qty if side == "bid" else book.ask_qty
    return qty[:, :depth].sum(axis=1)


def _depth_concentration(book: Book, side: str) -> np.ndarray:
    qty = book.bid_qty if side == "bid" else book.ask_qty
    total = qty[:, :10].sum(axis=1)
    return np.divide(qty[:, 0], total, out=book.empty(), where=total > 0)


def _deep_depth_imbalance(book: Book) -> np.ndarray:
    bid = book.bid_qty[:, 5:10].sum(axis=1)
    ask = book.ask_qty[:, 5:10].sum(axis=1)
    total = bid + ask
    return np.divide(bid - ask, total, out=book.empty(), where=total > 0)


def _vamp(book: Book, depth: int) -> np.ndarray:
    bid_price, ask_price = book.bid_price[:, :depth], book.ask_price[:, :depth]
    bid_qty, ask_qty = book.bid_qty[:, :depth], book.ask_qty[:, :depth]
    total = bid_qty.sum(axis=1) + ask_qty.sum(axis=1)
    numerator = (bid_price * ask_qty).sum(axis=1) + (ask_price * bid_qty).sum(axis=1)
    valid_prices = np.all(np.isfinite(bid_price) & np.isfinite(ask_price)
                          & (bid_price > 0) & (ask_price > 0), axis=1)
    return np.divide(numerator, total, out=book.empty(), where=valid_prices & (total > 0))


def _l1_imbalance(book: Book) -> np.ndarray:
    return np.divide(book.bid_qty[:, 0] - book.ask_qty[:, 0], book.l1_total,
                     out=book.empty(), where=book.l1_total > 0)


def _spread_to_round_trip_cost_ratio(book: Book) -> np.ndarray:
    """현재 스프레드가 고정 왕복비용의 몇 배인가."""
    spread_bps = np.divide((book.ask1 - book.bid1) * 10_000.0, book.mid,
                           out=book.empty(), where=book.mid > 0)
    return spread_bps / FEE_BPS


def _signed_flow(book: Book, window: int) -> np.ndarray:
    buy = np.asarray(book.data["buy_volume"], dtype=float)
    sell = np.asarray(book.data["sell_volume"], dtype=float)
    b, s = rolling_sum(buy, window), rolling_sum(sell, window)
    return (b - s) / (b + s + EPS)


# ---- feature 레지스트리 -------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """feature 하나. 이 객체가 없으면 그 이름은 어디에서도 쓸 수 없다.

    `compute` 는 빠른 구현이고 `definition` 은 뜻의 정본이다. 둘 다 있으면 같은 값을
    내야 하며 그것을 테스트가 확인한다.
    """

    # 정체
    name: str
    version: int

    # 값의 뜻
    value_type: str
    dimension: str
    unit: str

    # 계산
    compute: Callable[[Book], np.ndarray]
    definition: Mapping[str, Any] | None = None
    # 0 나눗셈·비유한 호가·기록 부족을 무엇으로 내는가. 계산 정의의 일부라
    # 여기가 바뀌면 `version` 을 올린다.
    invalid_policy: str = ""

    # 원본 데이터 요구
    dependencies: tuple[str, ...] = ()
    min_book_depth: int | None = None

    # 시간 의미
    time_basis: str = "tick"
    lookback: int | float = 0
    current_included: bool = True
    causal: bool = True

    # 설명 — 관측 사실과 해석을 섞지 않는다
    observable_description: str = ""
    economic_interpretation: str | None = None

    # 정리
    normalization_scope: str = "global"
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def definition_id(self) -> str:
        """계산 의미의 버전. 수식·윈도우·결측정책이 바뀌면 `version` 을 올린다."""
        return f"{self.name}.v{self.version}"

    @property
    def type_info(self) -> TypeInfo:
        return TypeInfo(self.value_type, self.dimension, self.unit)


FEATURES: dict[str, Feature] = {}
ALIASES: dict[str, str] = {}


def register(feature: Feature) -> Feature:
    """레지스트리에 넣는다. 같은 이름을 두 번 넣으면 예외."""
    if feature.name in FEATURES or feature.name in ALIASES:
        raise ValueError(f"feature {feature.name} 이 이미 등록돼 있다")
    for alias in feature.aliases:
        if alias in ALIASES or alias in FEATURES:
            raise ValueError(f"별칭 {alias} 이 이미 쓰이고 있다")
        ALIASES[alias] = feature.name
    FEATURES[feature.name] = feature
    return feature


def resolve(name: str) -> str:
    """별칭을 정본 이름으로. 모르는 이름이면 그대로 돌려준다 (검증이 잡는다)."""
    return ALIASES.get(name, name)


def get_feature(name: str) -> Feature:
    """정본 이름으로 조회. 별칭은 받지 않는다."""
    return FEATURES[name]


def resolve_feature(name: str) -> Feature:
    """별칭을 포함해 조회."""
    return FEATURES[resolve(str(name))]


def list_features() -> tuple[str, ...]:
    return tuple(sorted(FEATURES))


# 호가만으로 계산되는 것 ------------------------------------------------------
register(Feature(
    "mid_price", 1, "numeric", "price", "KRW", lambda b: b.mid,
    dependencies=("bid_price", "ask_price"), min_book_depth=1,
    invalid_policy="BID1·ASK1 이 유한하고 양수인 틱만 원본에 들어온다 (`data.load` 가 거른다) — NaN 없음",
    observable_description="(BID1+ASK1)/2",
    economic_interpretation="체결 가능한 가격의 중심. 가격 수준 자체라 종목 간 비교는 뜻이 없다",
    normalization_scope="not_comparable", tags=("order_book", "price", "price_level")))
register(Feature(
    "bid_1", 1, "numeric", "price", "KRW", lambda b: b.bid1,
    dependencies=("bid_price",), min_book_depth=1,
    invalid_policy="BID1 이 유한하고 양수인 틱만 원본에 들어온다 — NaN 없음",
    observable_description="최우선 매수호가",
    economic_interpretation="지정가 매수 진입의 게시가",
    normalization_scope="not_comparable",
    tags=("order_book", "price", "price_level", "execution")))
register(Feature(
    "ask_1", 1, "numeric", "price", "KRW", lambda b: b.ask1,
    dependencies=("ask_price",), min_book_depth=1,
    invalid_policy="ASK1 이 유한하고 ASK1>BID1 인 틱만 원본에 들어온다 — NaN 없음",
    observable_description="최우선 매도호가",
    economic_interpretation="taker 매수 진입의 체결가",
    normalization_scope="not_comparable",
    tags=("order_book", "price", "price_level", "execution")))
register(Feature(
    "spread_bps", 1, "numeric", "return", "bps",
    lambda b: np.divide((b.ask1 - b.bid1) * 10_000.0, b.mid, out=b.empty(), where=b.mid > 0),
    dependencies=("bid_price", "ask_price"), min_book_depth=1,
    invalid_policy="mid <= 0 이면 NaN",
    observable_description="(ASK1-BID1)/mid, bps",
    economic_interpretation="taker 로 왕복할 때 즉시 지불하는 비용의 하한",
    tags=("order_book", "liquidity")))
register(Feature(
    "spread_to_round_trip_cost_ratio", 1, "numeric", "dimensionless", "ratio",
    _spread_to_round_trip_cost_ratio,
    dependencies=("bid_price", "ask_price"), min_book_depth=1,
    invalid_policy="mid <= 0 이면 NaN. 왕복비용은 config.FEE_BPS 의 고정값",
    observable_description="spread_bps / 왕복비용_bps",
    economic_interpretation="현재 한 번의 호가 스프레드가 고정 왕복비용 대비 얼마나 큰지",
    tags=("order_book", "liquidity", "normalized")))
register(Feature(
    "ofi_cks_1", 1, "numeric", "quantity", "quantity", lambda b: depth_ofi(b, 1),
    dependencies=BOOK_L1, min_book_depth=1, lookback=1,
    invalid_policy="레벨의 가격이 비유한이거나 0 이하이면 그 레벨 기여는 0. 첫 틱은 0",
    observable_description="최우선호가 잔량·가격 변화의 부호합 (CKS 규약)",
    economic_interpretation="매수측 순주문흐름일 수 있으나, 취소와 신규를 구분하지는 못한다",
    normalization_scope="symbol", tags=("order_book", "flow")))
register(Feature(
    "ofi_depth_5", 1, "numeric", "quantity", "quantity", lambda b: depth_ofi(b, 5),
    dependencies=BOOK_L1, min_book_depth=5, lookback=1,
    invalid_policy="레벨의 가격이 비유한이거나 0 이하이면 그 레벨 기여는 0. 첫 틱은 0",
    observable_description="1~5호가 각 레벨의 CKS OFI 합",
    economic_interpretation="더 깊은 구간까지의 순주문흐름일 수 있다",
    normalization_scope="symbol", tags=("order_book", "flow")))
register(Feature(
    "ofi_depth_10", 1, "numeric", "quantity", "quantity", lambda b: depth_ofi(b, 10),
    dependencies=BOOK_L1, min_book_depth=10, lookback=1,
    invalid_policy="레벨의 가격이 비유한이거나 0 이하이면 그 레벨 기여는 0. 첫 틱은 0",
    observable_description="1~10호가 각 레벨의 CKS OFI 합",
    economic_interpretation="호가창 전체의 순주문흐름일 수 있다",
    normalization_scope="symbol", tags=("order_book", "flow")))
register(Feature(
    "bid_depth_total_5", 1, "numeric", "quantity", "quantity",
    lambda b: _depth_total(b, "bid", 5),
    dependencies=("bid_qty",), min_book_depth=5,
    invalid_policy="음수·비유한 잔량은 `Book` 이 0 으로 정리한다 — NaN 없음",
    observable_description="1~5호가 매수 잔량의 합",
    economic_interpretation="가까운 매수측에 게시된 유동성의 총량",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "ask_depth_total_5", 1, "numeric", "quantity", "quantity",
    lambda b: _depth_total(b, "ask", 5),
    dependencies=("ask_qty",), min_book_depth=5,
    invalid_policy="음수·비유한 잔량은 `Book` 이 0 으로 정리한다 — NaN 없음",
    observable_description="1~5호가 매도 잔량의 합",
    economic_interpretation="가까운 매도측에 게시된 유동성의 총량",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "bid_depth_total_10", 1, "numeric", "quantity", "quantity",
    lambda b: _depth_total(b, "bid", 10),
    dependencies=("bid_qty",), min_book_depth=10,
    invalid_policy="음수·비유한 잔량은 `Book` 이 0 으로 정리한다 — NaN 없음",
    observable_description="1~10호가 매수 잔량의 합",
    economic_interpretation="호가창 전체 매수측에 게시된 유동성의 총량",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "ask_depth_total_10", 1, "numeric", "quantity", "quantity",
    lambda b: _depth_total(b, "ask", 10),
    dependencies=("ask_qty",), min_book_depth=10,
    invalid_policy="음수·비유한 잔량은 `Book` 이 0 으로 정리한다 — NaN 없음",
    observable_description="1~10호가 매도 잔량의 합",
    economic_interpretation="호가창 전체 매도측에 게시된 유동성의 총량",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "book_imbalance", 1, "numeric", "dimensionless", "ratio", _l1_imbalance,
    dependencies=("bid_qty", "ask_qty"), min_book_depth=1,
    invalid_policy="L1 잔량 합이 0 이면 NaN. 음수·비유한 잔량은 `Book` 이 0 으로 정리한다",
    observable_description="(BID1잔량-ASK1잔량)/(합). [-1, 1]",
    economic_interpretation="게시된 매수·매도 유동성의 비대칭. 미게시 주문은 보이지 않는다",
    tags=("order_book", "liquidity")))
register(Feature(
    "deep_depth_imbalance_6_10", 1, "numeric", "dimensionless", "ratio",
    _deep_depth_imbalance,
    dependencies=("bid_qty", "ask_qty"), min_book_depth=10,
    invalid_policy="6~10호가 양측 잔량 합이 0 이면 NaN. 음수·비유한 잔량은 `Book` 이 0 으로 정리한다",
    observable_description="6~10호가의 (매수 잔량 합-매도 잔량 합)/(양측 합). [-1, 1]",
    economic_interpretation="최우선호가보다 깊게 게시된 유동성의 방향 비대칭",
    tags=("order_book", "liquidity")))
register(Feature(
    "bid_depth_concentration", 1, "numeric", "dimensionless", "ratio",
    lambda b: _depth_concentration(b, "bid"),
    dependencies=("bid_qty",), min_book_depth=10,
    invalid_policy="1~10호가 매수 잔량 합이 0 이면 NaN. 음수·비유한 잔량은 `Book` 이 0 으로 정리한다",
    observable_description="BID1 잔량 / 1~10호가 매수 잔량 합. [0, 1]",
    economic_interpretation="매수측 유동성이 최우선호가에 얼마나 몰렸는지",
    tags=("order_book", "liquidity")))
register(Feature(
    "ask_depth_concentration", 1, "numeric", "dimensionless", "ratio",
    lambda b: _depth_concentration(b, "ask"),
    dependencies=("ask_qty",), min_book_depth=10,
    invalid_policy="1~10호가 매도 잔량 합이 0 이면 NaN. 음수·비유한 잔량은 `Book` 이 0 으로 정리한다",
    observable_description="ASK1 잔량 / 1~10호가 매도 잔량 합. [0, 1]",
    economic_interpretation="매도측 유동성이 최우선호가에 얼마나 몰렸는지",
    tags=("order_book", "liquidity")))
register(Feature(
    "book_imbalance_velocity", 1, "numeric", "dimensionless", "ratio",
    lambda b: _tick_diff(_l1_imbalance(b), b.n),
    definition={"op": "difference", "lag": 1, "time_basis": "tick",
                "input": {"op": "primitive", "primitive_id": "book_imbalance"}},
    dependencies=("bid_qty", "ask_qty"), min_book_depth=1, lookback=1,
    invalid_policy="직전 틱이 없거나 양쪽 중 하나가 NaN 이면 NaN",
    observable_description="book_imbalance 의 직전 틱 대비 차. 첫 틱은 NaN",
    economic_interpretation="유동성 비대칭이 어느 쪽으로 움직이는 중인지를 볼 수 있다",
    tags=("order_book", "liquidity", "derived")))
register(Feature(
    "microprice_dev_bps", 1, "numeric", "return", "bps",
    lambda b: np.divide((b.microprice - b.mid) * 10_000.0, b.mid, out=b.empty(), where=b.mid > 0),
    dependencies=BOOK_L1, min_book_depth=1,
    invalid_policy="mid <= 0 이면 NaN. L1 잔량 합이 0 이면 microprice 를 mid 로 둔다",
    observable_description="잔량가중 중간가와 mid 의 차, mid 대비 bps",
    economic_interpretation="다음 체결가가 어느 쪽으로 치우쳐 있는지의 대용치",
    tags=("order_book", "microprice")))
register(Feature(
    "vamp_bbo", 1, "numeric", "price", "KRW", lambda b: b.microprice,
    dependencies=BOOK_L1, min_book_depth=1,
    invalid_policy="L1 양측 잔량 합이 0 이면 mid 로 둔다",
    observable_description="(BID1가격×ASK1잔량 + ASK1가격×BID1잔량) / 양측 잔량 합",
    economic_interpretation="최우선호가 잔량으로 가중한 가격. microprice 와 같은 값",
    normalization_scope="not_comparable", tags=("order_book", "microprice", "price_level")))
register(Feature(
    "vamp_5", 1, "numeric", "price", "KRW", lambda b: _vamp(b, 5),
    dependencies=BOOK_L1, min_book_depth=5,
    invalid_policy="1~5호가 중 가격이 비유한·0 이하이거나 양측 잔량 합이 0 이면 NaN",
    observable_description="1~5호가에서 BID가격×ASK잔량과 ASK가격×BID잔량의 합을 양측 잔량 합으로 나눈 값",
    economic_interpretation="5단계 호가의 반대편 잔량 가중 가격",
    normalization_scope="not_comparable", tags=("order_book", "microprice", "price_level")))
register(Feature(
    "vamp_10", 1, "numeric", "price", "KRW", lambda b: _vamp(b, 10),
    dependencies=BOOK_L1, min_book_depth=10,
    invalid_policy="1~10호가 중 가격이 비유한·0 이하이거나 양측 잔량 합이 0 이면 NaN",
    observable_description="1~10호가에서 BID가격×ASK잔량과 ASK가격×BID잔량의 합을 양측 잔량 합으로 나눈 값",
    economic_interpretation="10단계 호가의 반대편 잔량 가중 가격",
    normalization_scope="not_comparable", tags=("order_book", "microprice", "price_level")))
register(Feature(
    "microprice_velocity", 1, "numeric", "return", "bps",
    lambda b: np.divide(_tick_diff(b.microprice, b.n) * 10_000.0, b.mid,
                        out=b.empty(), where=b.mid > 0),
    dependencies=BOOK_L1, min_book_depth=1, lookback=1,
    invalid_policy="mid <= 0 이거나 직전 틱이 없으면 NaN",
    observable_description="microprice 의 직전 틱 대비 차, mid 대비 bps. 첫 틱은 NaN",
    economic_interpretation="잔량가중 가격의 단기 표류",
    tags=("order_book", "microprice", "derived"),
    aliases=("microprice_velocity_1t", "microprice_velocity_1t_bps")))
register(Feature(
    "mid_return_5t_bps", 1, "numeric", "return", "bps",
    lambda b: _lagged_return_bps(b.mid, 5, b.n),
    dependencies=("bid_price", "ask_price"), min_book_depth=1, lookback=5,
    invalid_policy="5틱 전 mid 가 0 이하이거나 비유한이면 NaN. 앞 5틱은 첫 값 기준으로 채운다",
    observable_description="5틱 전 mid 대비 수익률(bps). 앞 5틱은 첫 값 기준",
    economic_interpretation="아주 짧은 구간의 가격 표류",
    tags=("order_book", "price")))
register(Feature(
    "mid_return_20t_bps", 1, "numeric", "return", "bps",
    lambda b: _lagged_return_bps(b.mid, 20, b.n),
    dependencies=("bid_price", "ask_price"), min_book_depth=1, lookback=20,
    invalid_policy="20틱 전 mid 가 0 이하이거나 비유한이면 NaN. 앞 20틱은 첫 값 기준으로 채운다",
    observable_description="20틱 전 mid 대비 수익률(bps). 앞 20틱은 첫 값 기준",
    economic_interpretation="짧은 구간의 가격 표류",
    tags=("order_book", "price")))
register(Feature(
    "mid_return_100t_bps", 1, "numeric", "return", "bps",
    lambda b: _lagged_return_bps(b.mid, 100, b.n),
    dependencies=("bid_price", "ask_price"), min_book_depth=1, lookback=100,
    invalid_policy="100틱 전 mid 가 0 이하이거나 비유한이면 NaN. 앞 100틱은 첫 값 기준으로 채운다",
    observable_description="100틱 전 mid 대비 수익률(bps). 앞 100틱은 첫 값 기준",
    economic_interpretation="더 긴 구간의 가격 표류",
    tags=("order_book", "price")))
register(Feature(
    "book_slope_bid", 1, "numeric", "price_per_quantity", "KRW/quantity",
    lambda b: _book_slope(b, "bid"),
    dependencies=("bid_price", "bid_qty"), min_book_depth=5,
    invalid_policy="BID5 가격이 0 이하이거나 1~5호가 매수잔량 합이 0 이면 NaN",
    observable_description="(BID1가격-BID5가격) / 1~5호가 매수잔량 합",
    economic_interpretation="매수측 호가창의 가파름. 큰 주문이 가격을 얼마나 밀지의 대용치",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "book_slope_ask", 1, "numeric", "price_per_quantity", "KRW/quantity",
    lambda b: _book_slope(b, "ask"),
    dependencies=("ask_price", "ask_qty"), min_book_depth=5,
    invalid_policy="ASK5 가격이 0 이하이거나 1~5호가 매도잔량 합이 0 이면 NaN",
    observable_description="(ASK5가격-ASK1가격) / 1~5호가 매도잔량 합",
    economic_interpretation="매도측 호가창의 가파름",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "queue_imbalance_best", 1, "numeric", "dimensionless", "ratio",
    lambda b: np.divide(b.bid_qty[:, 0], b.l1_total, out=b.empty(), where=b.l1_total > 0),
    dependencies=("bid_qty", "ask_qty"), min_book_depth=1,
    invalid_policy="L1 잔량 합이 0 이면 NaN",
    observable_description="BID1잔량 / (BID1+ASK1잔량). [0, 1]",
    economic_interpretation="book_imbalance 의 단조 변환(2q-1)이라 같은 정보를 담는다",
    tags=("order_book", "liquidity")))
register(Feature(
    "bid_queue_depth_at_l1", 1, "numeric", "quantity", "quantity", lambda b: b.bid_qty[:, 0],
    dependencies=("bid_qty",), min_book_depth=1,
    invalid_policy="음수·비유한 잔량은 `Book` 이 0 으로 정리한다 — NaN 없음",
    observable_description="최우선 매수 게시 잔량",
    economic_interpretation="지정가 매수 큐의 길이. 내 앞에 몇 주가 있는지의 상한",
    normalization_scope="symbol", tags=("order_book", "liquidity")))
register(Feature(
    "bid_best_price_drop_count_50t", 1, "numeric", "count", "event count",
    lambda b: rolling_sum(_drop_events(b.bid1), DROP_COUNT_WINDOW),
    dependencies=("bid_price",), min_book_depth=1, lookback=DROP_COUNT_WINDOW,
    invalid_policy="직전 값이 0 이하이면 세지 않는다. 앞 50틱은 있는 만큼만 더한다",
    observable_description="최근 50틱 중 BID1 이 직전 틱보다 내려간 횟수",
    economic_interpretation="매수측 재호가나 큐 소진을 반영할 수 있다. "
                            "큐 소진을 유일하게 식별하지는 않는다",
    tags=("order_book", "event"),
    aliases=("bid_queue_depletion_5s_bps",)))
register(Feature(
    "ask_best_price_drop_count_50t", 1, "numeric", "count", "event count",
    lambda b: rolling_sum(_drop_events(b.ask1), DROP_COUNT_WINDOW),
    dependencies=("ask_price",), min_book_depth=1, lookback=DROP_COUNT_WINDOW,
    invalid_policy="직전 값이 0 이하이면 세지 않는다. 앞 50틱은 있는 만큼만 더한다",
    observable_description="최근 50틱 중 ASK1 이 직전 틱보다 내려온 횟수",
    economic_interpretation="매도측 재호가나 게시 매도 유동성 변화를 반영할 수 있다. "
                            "신규 매도 주문의 도착을 유일하게 식별하지는 않는다",
    tags=("order_book", "event"),
    aliases=("ask_queue_arrival_count_5s",)))

# 시계 시간이 필요한 것 --------------------------------------------------------
register(Feature(
    "minute_of_session", 1, "numeric", "count", "minute",
    lambda b: (np.asarray(b.data["time_s"], dtype=float) - SESSION_OPEN_SECONDS) / 60.0,
    dependencies=("time_s",), time_basis="clock",
    invalid_policy="time_s 는 `data.load` 가 원본 시각에서 만든 초 단위 값이다 — 정규장 밖 값은 로더가 제외한다",
    observable_description="09:00 KST부터 지난 분. 정규장은 대략 0~390",
    economic_interpretation="장 초반·중반·마감처럼 시각에 따른 상태 차이를 나누는 기준",
    tags=("clock",)))

# 체결 이벤트가 필요한 것 ------------------------------------------------------
register(Feature(
    "signed_aggr_flow_20", 1, "numeric", "dimensionless", "ratio",
    lambda b: _signed_flow(b, 20),
    dependencies=TRADES, lookback=20,
    invalid_policy="구간에 체결이 없으면 분모의 EPS 때문에 0 을 낸다 — '균형' 이 아니라 '관측 없음' 인데 0 으로 나온다는 뜻이다",
    observable_description="최근 20틱 (매수주도-매도주도)/(합) 체결량. [-1, 1]",
    economic_interpretation="공격적 주문의 방향 편향",
    tags=("trade_flow",)))
register(Feature(
    "signed_aggr_flow_100", 1, "numeric", "dimensionless", "ratio",
    lambda b: _signed_flow(b, 100),
    dependencies=TRADES, lookback=100,
    invalid_policy="구간에 체결이 없으면 분모의 EPS 때문에 0 을 낸다 — '균형' 이 아니라 '관측 없음' 인데 0 으로 나온다는 뜻이다",
    observable_description="최근 100틱 (매수주도-매도주도)/(합) 체결량. [-1, 1]",
    economic_interpretation="더 긴 구간의 공격적 주문 방향 편향. "
                            "구 `trade_imbalance_signed` 와 계산이 같아 그 이름은 삭제했다",
    tags=("trade_flow",)))
register(Feature(
    "vol_flow", 1, "numeric", "quantity", "quantity",
    lambda b: rolling_sum(np.asarray(b.data["buy_volume"], dtype=float), 20)
    + rolling_sum(np.asarray(b.data["sell_volume"], dtype=float), 20),
    dependencies=TRADES, lookback=20,
    invalid_policy="체결 없는 틱은 0. 앞 20틱은 있는 만큼만 더한다",
    observable_description="최근 20틱 총 체결량",
    economic_interpretation="거래 활동성",
    normalization_scope="symbol", tags=("trade_flow",)))
register(Feature(
    "trade_event_indicator", 1, "boolean", "boolean", "0/1",
    lambda b: ((np.asarray(b.data["buy_volume"], dtype=float)
                + np.asarray(b.data["sell_volume"], dtype=float)) > 0.0).astype(float),
    dependencies=TRADES,
    invalid_policy="체결량 합이 0 이면 0",
    observable_description="그 틱에 체결이 하나라도 있었는가",
    economic_interpretation="상태가 아니라 표시다. 자를 축이 아니라 조건의 재료다",
    tags=("trade_flow", "event", "event_marker")))


# ---- feature 사용 정책 ---------------------------------------------------------

# 이름은 실행 명세의 threshold source와 맞춘다. 나머지는 "어떤 조건 형태만 허용하는가"를
# 나타내는 Catalog 정책값이다.
SEARCHABLE = "SEARCHABLE"          # 값이 달라져도 같은 가설이다
THRESHOLD_LITERAL = "literal"
THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE = "prior_valid_day_symbol_quantile"
THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE = "rolling_prior_100_ticks_quantile"
THRESHOLD_LITERAL_TIME_RANGE = "literal_time_range"
THRESHOLD_BOOLEAN = "boolean"
THRESHOLD_EXECUTION_ONLY = "execution_only"
THRESHOLD_DERIVED_INPUT_ONLY = "derived_input_only"
THRESHOLD_DERIVED_DUPLICATE = "derived_duplicate"
FEATURE_THRESHOLD_KINDS = frozenset({
    THRESHOLD_LITERAL,
    THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
    THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
    THRESHOLD_LITERAL_TIME_RANGE,
    THRESHOLD_BOOLEAN,
    THRESHOLD_EXECUTION_ONLY,
    THRESHOLD_DERIVED_INPUT_ONLY,
    THRESHOLD_DERIVED_DUPLICATE,
})


@dataclass(frozen=True)
class FeatureUsePolicy:
    """한 feature를 조건으로 쓰는 방법.

    계산값은 그대로 두고, 같은 의미를 원값·q·rolling z로 중복 탐색하지 않도록
    사용 방법만 제한한다. 현재 framework의 자동 탐색은 종목별 직전 100 tick q만
    구현하므로 OFI·잔량·queue·slope·vol_flow는 그 한 방법으로 고정한다.
    """

    threshold_kinds: tuple[str, ...]
    note: str
    natural_thresholds: tuple[float, ...] = ()
    required_context: tuple[str, ...] = ()
    duplicate_of: str | None = None
    automatic_q_search: bool = False

    def as_agent_dict(self) -> dict[str, Any]:
        return {
            "allowed": list(self.threshold_kinds),
            "natural_thresholds": list(self.natural_thresholds),
            "required_context": list(self.required_context),
            "duplicate_of": self.duplicate_of,
            "automatic_q_search": self.automatic_q_search,
            "note": self.note,
        }


def _use_policy(*threshold_kinds: str, note: str,
                natural_thresholds: tuple[float, ...] = (),
                required_context: tuple[str, ...] = (),
                duplicate_of: str | None = None,
                automatic_q_search: bool | None = None) -> FeatureUsePolicy:
    if automatic_q_search is None:
        automatic_q_search = (
            THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE in threshold_kinds
            and not required_context)
    return FeatureUsePolicy(
        threshold_kinds=tuple(threshold_kinds), note=note,
        natural_thresholds=natural_thresholds, required_context=required_context,
        duplicate_of=duplicate_of, automatic_q_search=automatic_q_search)


FEATURE_USE_POLICIES: dict[str, FeatureUsePolicy] = {
    # 가격 수준과 raw VAMP는 실행 가격이다. 종목의 가격 크기를 진입 신호로 만들지 않는다.
    "mid_price": _use_policy(THRESHOLD_EXECUTION_ONLY, note="가격 수준은 execution 전용"),
    "bid_1": _use_policy(THRESHOLD_EXECUTION_ONLY, note="진입 게시가/표시가"),
    "ask_1": _use_policy(THRESHOLD_EXECUTION_ONLY, note="진입 체결가/표시가"),
    "vamp_bbo": _use_policy(THRESHOLD_EXECUTION_ONLY, note="raw VAMP는 execution 전용"),
    "vamp_5": _use_policy(THRESHOLD_EXECUTION_ONLY, note="raw VAMP는 execution 전용"),
    "vamp_10": _use_policy(THRESHOLD_EXECUTION_ONLY, note="raw VAMP는 execution 전용"),

    # spread 자체가 아니라 비용으로 나눈 비율만 원값 조건으로 쓴다.
    "spread_bps": _use_policy(
        THRESHOLD_DERIVED_INPUT_ONLY,
        note="절대 q/원값 조건 금지. spread_to_round_trip_cost_ratio의 입력값"),
    "spread_to_round_trip_cost_ratio": _use_policy(
        THRESHOLD_LITERAL,
        note="고정 왕복비용 대비 비율만 원값 조건으로 허용"),

    # 같은 L1 비대칭의 두 표현은 하나만 탐색한다.
    "book_imbalance": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용",
        natural_thresholds=(0.0,)),
    "queue_imbalance_best": _use_policy(
        THRESHOLD_DERIVED_DUPLICATE,
        note="book_imbalance의 단조 변환이라 별도 축으로 쓰지 않음",
        duplicate_of="book_imbalance"),
    "microprice_dev_bps": _use_policy(
        THRESHOLD_DERIVED_DUPLICATE,
        note="microprice_dev_bps / spread_bps = book_imbalance / 2라 별도 축으로 쓰지 않음",
        duplicate_of="book_imbalance"),
    "deep_depth_imbalance_6_10": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용",
        natural_thresholds=(0.0,)),
    "bid_depth_concentration": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용"),
    "ask_depth_concentration": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용"),

    # 규모와 흐름은 직전 100 tick 종목 q 하나만 쓴다. rolling z를 동시에 열지 않는다.
    "ofi_cks_1": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                              note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "ofi_depth_5": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "ofi_depth_10": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                 note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "bid_depth_total_5": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                      note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "ask_depth_total_5": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                      note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "bid_depth_total_10": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                       note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "ask_depth_total_10": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                       note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "bid_queue_depth_at_l1": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                          note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "book_slope_bid": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                   note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "book_slope_ask": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                   note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "vol_flow": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                             note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),

    # 방향 비율과 수익률은 종목별 직전 100 tick q로만 비교한다.
    "signed_aggr_flow_20": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용", natural_thresholds=(0.0,)),
    "signed_aggr_flow_100": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE, THRESHOLD_LITERAL,
        note="원값 보존. 자동 탐색은 종목별 직전 100 tick q만 사용", natural_thresholds=(0.0,)),
    "book_imbalance_velocity": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
        note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "microprice_velocity": _use_policy(
        THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
        note="종목별 직전 100 tick q만 허용; rolling z와 중복 탐색 금지"),
    "mid_return_5t_bps": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                      note="종목별 직전 100 tick q만 허용"),
    "mid_return_20t_bps": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                       note="종목별 직전 100 tick q만 허용"),
    "mid_return_100t_bps": _use_policy(THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
                                        note="종목별 직전 100 tick q만 허용"),

    # 사건/시간은 분포의 상하가 아니라 고정된 문법으로만 쓴다.
    "bid_best_price_drop_count_50t": _use_policy(
        THRESHOLD_LITERAL, THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
        note="정수 count 원값만 허용. q는 고정된 유동성·시간대 맥락 안에서만 가능",
        required_context=("liquidity_regime", "minute_of_session")),
    "ask_best_price_drop_count_50t": _use_policy(
        THRESHOLD_LITERAL, THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
        note="정수 count 원값만 허용. q는 고정된 유동성·시간대 맥락 안에서만 가능",
        required_context=("liquidity_regime", "minute_of_session")),
    "minute_of_session": _use_policy(
        THRESHOLD_LITERAL_TIME_RANGE,
        note="신호가 아닌 거래 가능 시간 필터. 고정된 시작·끝 시각 범위만 허용"),
    "trade_event_indicator": _use_policy(
        THRESHOLD_BOOLEAN,
        note="True / False만 허용"),
}


def feature_use_policy(name: str) -> FeatureUsePolicy:
    """별칭까지 정본 feature의 사용 정책을 돌려준다."""
    return FEATURE_USE_POLICIES[resolve(str(name))]


def quantile_state_key(feature: str, direction: str) -> tuple[str, str]:
    """단조 중복 feature를 같은 직전 100 tick 분위수 상태로 묶는다."""
    name = resolve(str(feature))
    while FEATURE_USE_POLICIES[name].duplicate_of is not None:
        name = resolve(str(FEATURE_USE_POLICIES[name].duplicate_of))
    return name, str(direction)


# ---- 데이터셋 가용성 -----------------------------------------------------------

@dataclass(frozen=True)
class FeatureAvailability:
    """이 데이터셋에서 계산 가능한가. 불가능하면 왜 그런가.

    조용히 빼면 Agent 는 자기가 쓴 어휘가 왜 사라졌는지 알 수 없다.
    """

    feature: str
    available: bool
    reason: str | None = None


def feature_availability(capabilities: DataCapabilities) -> tuple[FeatureAvailability, ...]:
    """모든 feature 의 가용성과 이유."""
    out: list[FeatureAvailability] = []
    for name in sorted(FEATURES):
        spec = FEATURES[name]
        missing = tuple(d for d in spec.dependencies if d not in capabilities.fields)
        if missing:
            out.append(FeatureAvailability(name, False, "원본 필드 없음: " + ", ".join(missing)))
        elif spec.min_book_depth and capabilities.book_depth < spec.min_book_depth:
            out.append(FeatureAvailability(
                name, False,
                f"호가 depth {spec.min_book_depth} 필요, 데이터는 {capabilities.book_depth}"))
        else:
            out.append(FeatureAvailability(name, True))
    return tuple(out)


def available_features(capabilities: DataCapabilities) -> tuple[str, ...]:
    """이 데이터셋에서 실제로 계산되는 feature 이름."""
    return tuple(a.feature for a in feature_availability(capabilities) if a.available)


def compute_features(names: Iterable[str], book: Book) -> dict[str, np.ndarray]:
    """요청한 feature 를 계산한다. 이 데이터셋에서 못 만드는 것은 뺀다.

    0 으로 채우지 않는다 — 0 은 "매도 압력이 없었다" 는 관측을 지어내는 것이다.
    무엇이 왜 빠졌는지는 `feature_availability()` 로 본다.
    """
    usable = set(available_features(book.capabilities))
    out: dict[str, np.ndarray] = {}
    for raw in names:
        name = resolve(str(raw))
        if name not in FEATURES:
            raise KeyError(f"레지스트리에 없는 feature: {raw}")
        if name not in usable:
            continue
        out[name] = np.asarray(FEATURES[name].compute(book), dtype=float)
    return out


def definition_ids(names: Iterable[str] | None = None) -> dict[str, str]:
    """이름 -> definition_id. 원장 컬럼을 계산 정의에 잇는 고리다.

    원장 한 행의 `book_imbalance` 값이 **어느 정의로 계산된 값인지** 를 나중에
    말할 수 있어야 가설 → feature → 시장 증거 → 판단이 이어진다.
    """
    return {resolve(str(n)): FEATURES[resolve(str(n))].definition_id
            for n in (list_features() if names is None else names)}


# ---- 연산자 -------------------------------------------------------------------

@dataclass(frozen=True)
class Operator:
    """표현식 트리에 쓸 수 있는 연산자.

    `inputs` 가 어느 인자가 하위 표현식이고 무슨 타입이어야 하는지를 정한다.
    `output` 은 입력 타입에서 출력 타입을 만든다 — Agent 가 단위를 적지 않는다.
    """

    op: str
    required: tuple[str, ...]
    inputs: Mapping[str, str] = field(default_factory=dict)
    output: Callable[[Mapping[str, TypeInfo], Mapping[str, Any]], TypeInfo] | None = None
    domains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    thresholds: tuple[str, ...] = ()
    temporal: bool = False
    output_doc: str = ""
    doc: str = ""


def _ratio_type(kids: Mapping[str, TypeInfo], node: Mapping[str, Any]) -> TypeInfo:
    numerator, denominator = kids["numerator"], kids["denominator"]
    if numerator.dimension == denominator.dimension:
        return TypeInfo("numeric", "dimensionless", "ratio")
    return TypeInfo("numeric", f"{numerator.dimension}_per_{denominator.dimension}",
                    f"{numerator.unit}/{denominator.unit}")


def _raw_type(kids: Mapping[str, TypeInfo], node: Mapping[str, Any]) -> TypeInfo:
    field = str(node.get("field") or "")
    if field not in RAW_FIELD_TYPES:
        raise ValueError(f"raw field가 없다: {field!r}")
    if field in {"bid_price", "ask_price", "bid_qty", "ask_qty"}:
        level = node.get("level")
        if not _is_integer(level) or not 1 <= int(level) <= 10:
            raise ValueError("호가 raw field의 level은 1..10 정수여야 한다")
    elif "level" in node:
        raise ValueError(f"{field}에는 level을 붙이지 않는다")
    return RAW_FIELD_TYPES[field]


def _same_or_derived(kids: Mapping[str, TypeInfo], node: Mapping[str, Any]) -> TypeInfo:
    left, right = kids["left"], kids["right"]
    return left if left.dimension == right.dimension else DERIVED_NUMERIC


OPERATORS: dict[str, Operator] = {op.op: op for op in (
    Operator("primitive", ("primitive_id",),
             output=lambda kids, node: FEATURES[resolve(str(node["primitive_id"]))].type_info,
             output_doc="그 feature 의 타입", doc="feature 값 하나"),
    Operator("raw", ("field",), output=_raw_type,
             output_doc="원본 field 타입", doc="현재 tick의 raw 호가·체결 입력"),
    Operator("compare", ("input", "comparator", "value"),
             inputs={"input": "numeric"},
             output=lambda kids, node: BOOLEAN,
             domains={"comparator": ("<", "<=", "==", ">", ">=")},
             thresholds=("value",),
             output_doc="boolean",
             doc="값을 임계와 비교해 bool 로. `value` 만이 임계다"),
    Operator("compare_values", ("left", "right", "comparator"),
             inputs={"left": "numeric", "right": "numeric"},
             output=lambda kids, node: BOOLEAN,
             domains={"comparator": ("<", "<=", "==", ">", ">=")},
             output_doc="boolean", doc="두 인과적 표현식 값을 직접 비교"),
    Operator("all", ("args",), inputs={"args": "boolean_list"},
             output=lambda kids, node: BOOLEAN, output_doc="boolean", doc="전부 참"),
    Operator("any", ("args",), inputs={"args": "boolean_list"},
             output=lambda kids, node: BOOLEAN, output_doc="boolean", doc="하나라도 참"),
    Operator("ratio", ("numerator", "denominator", "zero_policy"),
             inputs={"numerator": "numeric", "denominator": "numeric"},
             output=_ratio_type,
             domains={"zero_policy": ("nan", "reject", "zero")},
             output_doc="차원이 같으면 dimensionless, 다르면 a_per_b", doc="나눗셈"),
    Operator("add", ("left", "right"), inputs={"left": "numeric", "right": "numeric"},
             output=_same_or_derived, output_doc="numeric", doc="덧셈"),
    Operator("subtract", ("left", "right"),
             inputs={"left": "numeric", "right": "numeric"},
             output=_same_or_derived, output_doc="numeric", doc="뺄셈"),
    Operator("multiply", ("left", "right"),
             inputs={"left": "numeric", "right": "numeric"},
             output=lambda kids, node: DERIVED_NUMERIC, output_doc="numeric", doc="곱셈"),
    Operator("divide", ("left", "right"),
             inputs={"left": "numeric", "right": "numeric"},
             output=lambda kids, node: DERIVED_NUMERIC, output_doc="numeric", doc="안전 나눗셈"),
    Operator("absolute", ("input",), inputs={"input": "numeric"},
             output=lambda kids, node: kids["input"], output_doc="numeric", doc="절댓값"),
    Operator("negate", ("input",), inputs={"input": "numeric"},
             output=lambda kids, node: kids["input"], output_doc="numeric", doc="부호 반전"),
    Operator("log1p", ("input",), inputs={"input": "numeric"},
             output=lambda kids, node: DERIVED_NUMERIC, output_doc="numeric", doc="log(1+x)"),
    Operator("difference", ("input", "lag", "time_basis"),
             inputs={"input": "numeric"},
             output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="입력과 같은 차원", doc="`lag` 만큼 전과의 차"),
    Operator("rolling_zscore", ("input", "window", "min_observations", "time_basis"),
             inputs={"input": "numeric"},
             output=lambda kids, node: ZSCORE,
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="dimensionless (sigma)",
             doc="인과적 z-score. 구 `zscore(x, n)` feature 3개를 대체한다"),
    Operator("rolling_sum", ("input", "window", "time_basis"),
             inputs={"input": "numeric"},
             output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="입력과 같은 차원", doc="구간 합"),
    Operator("rolling_mean", ("input", "window", "time_basis"),
             inputs={"input": "numeric"}, output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick",)}, temporal=True,
             output_doc="입력과 같은 차원", doc="인과적 구간 평균"),
    Operator("rolling_std", ("input", "window", "time_basis"),
             inputs={"input": "numeric"}, output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick",)}, temporal=True,
             output_doc="입력과 같은 차원", doc="인과적 구간 표준편차"),
    Operator("rolling_min", ("input", "window", "time_basis"),
             inputs={"input": "numeric"}, output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick",)}, temporal=True,
             output_doc="입력과 같은 차원", doc="인과적 구간 최솟값"),
    Operator("rolling_max", ("input", "window", "time_basis"),
             inputs={"input": "numeric"}, output=lambda kids, node: kids["input"],
             domains={"time_basis": ("tick",)}, temporal=True,
             output_doc="입력과 같은 차원", doc="인과적 구간 최댓값"),
    Operator("crossover", ("input", "threshold", "direction", "equality"),
             inputs={"input": "numeric"},
             output=lambda kids, node: EVENT,
             domains={"direction": ("above", "below"),
                      "equality": ("from_equal", "strict")},
             thresholds=("threshold",),
             temporal=True, output_doc="event (그 틱에만 참)", doc="임계를 넘는 순간"),
    Operator("persistence", ("condition", "window", "min_true", "time_basis"),
             inputs={"condition": "boolean"},
             output=lambda kids, node: BOOLEAN,
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="boolean",
             doc="구간 안에서 조건이 몇 번 이상 참"),
    Operator("sequence", ("setup", "trigger", "min_lag", "max_lag", "time_basis"),
             inputs={"setup": "boolean", "trigger": "boolean"},
             output=lambda kids, node: EVENT,
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="event (그 틱에만 참)",
             doc="A 다음 B 가 그 간격 안에"),
    Operator("sequence_once", ("setup", "trigger", "min_lag", "max_lag", "time_basis"),
             inputs={"setup": "boolean", "trigger": "boolean"},
             output=lambda kids, node: EVENT,
             domains={"time_basis": ("tick", "clock")},
             temporal=True, output_doc="event (그 틱에만 참)",
             doc="한 setup 연속 구간에서 A 다음 B 를 한 번만"),
    Operator("level_aggregate", ("side", "field", "levels", "reducer", "missing_policy"),
             output=lambda kids, node: (TypeInfo("numeric", "price", "KRW")
                                        if str(node["field"]) == "price"
                                        else TypeInfo("numeric", "quantity", "quantity")),
             domains={"side": ("ask", "bid", "both"), "field": ("price", "quantity"),
                      "reducer": ("first", "last", "mean", "sum"),
                      "missing_policy": ("mask", "reject", "zero")},
             output_doc="field 가 price 면 price, quantity 면 quantity",
             doc="여러 호가 레벨을 하나로"),
)}

# 임계가 아니라 구조를 정하는 정수. 분위수로 채울 수 없다 —
# "몇 틱" 에 대한 분포는 존재하지 않는다. 어느 것이 search 대상인지는 카탈로그가
# 정하지 않는다 — 그것은 Grounded Specification 의 몫이다.
STRUCTURAL_PARAMS = frozenset({
    "window", "min_observations", "lag", "min_lag", "max_lag", "min_true", "levels", "ddof",
})

UNRESOLVED_PREFIX = "UNRESOLVED:"


class ExpressionError(ValueError):
    """표현식이 실행 전 검사를 통과하지 못했다. 모든 이유를 한 번에 담는다."""


def validate_expression(expression: Mapping[str, Any], *, allow_unresolved: bool = False) -> None:
    """표현식 트리를 실행 전에 검사한다. 통과하지 못하면 예외.

    문법·허용값뿐 아니라 타입도 본다. `all(mid_price, ...)` 처럼 뜻이 안 되는 것은
    데이터를 읽기 전에 결정적으로 실패한다.

    통과 못 할 이유를 전부 모아 한 번에 보고한다. 하나씩 던지면 Agent 가 왕복만 한다.
    """
    infer_expression_type(expression, allow_unresolved=allow_unresolved)


def infer_expression_type(expression: Mapping[str, Any], *,
                          allow_unresolved: bool = False) -> TypeInfo:
    """표현식이 내는 값의 타입. 검사에 실패하면 `ExpressionError`."""
    problems: list[str] = []
    info = _infer(expression, "$", problems, allow_unresolved)
    if problems:
        raise ExpressionError("표현식 검증 실패 " + str(len(problems)) + "건:\n  - "
                              + "\n  - ".join(problems))
    assert info is not None   # problems 가 비었으면 타입이 나온다
    return info


def _threshold_problem(value: Any, path: str, allow_unresolved: bool) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return f"{path}: 임계는 숫자이거나 '{UNRESOLVED_PREFIX}<이름>' 이어야 한다"
    if isinstance(value, str):
        if not value.startswith(UNRESOLVED_PREFIX):
            return f"{path}: 임계는 숫자이거나 '{UNRESOLVED_PREFIX}<이름>' 이어야 한다"
        if not allow_unresolved:
            return f"{path}: 임계가 아직 {value} 이다"
    return None


def _infer(node: Any, path: str, problems: list[str], allow_unresolved: bool) -> TypeInfo | None:
    if not isinstance(node, Mapping):
        problems.append(f"{path}: 표현식 노드가 아니다 ({type(node).__name__})")
        return None
    op = node.get("op")
    if op is None:
        problems.append(f"{path}: 'op' 가 없다")
        return None
    operator = OPERATORS.get(str(op))
    if operator is None:
        problems.append(f"{path}: 모르는 연산자 '{op}'. 쓸 수 있는 것 {sorted(OPERATORS)}")
        return None

    ok = True
    for name in operator.required:
        if name not in node:
            problems.append(f"{path}: '{op}' 에 필수 인자 '{name}' 이 없다")
            ok = False
    for name, allowed in operator.domains.items():
        if name in node and node[name] not in allowed:
            problems.append(f"{path}.{name}: '{node[name]}' 는 허용값이 아니다. {list(allowed)}")
            ok = False
    for name in STRUCTURAL_PARAMS:
        if name not in node:
            continue
        value = node[name]
        if isinstance(value, str) and value.startswith(UNRESOLVED_PREFIX):
            problems.append(
                f"{path}.{name}: 구조 정수를 UNRESOLVED 로 뒀다. 분위수가 채울 수 없다")
            ok = False
        elif name == "levels":
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or not all(_is_integer(v) for v in value)):
                problems.append(f"{path}.levels: 정수 두 개의 목록이어야 한다 ({value!r})")
                ok = False
        elif (name in {"lag", "min_lag", "max_lag"}
              and str(node.get("time_basis")) == "clock"):
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(float(value))):
                problems.append(f"{path}.{name}: clock 간격은 유한한 숫자여야 한다 ({value!r})")
                ok = False
        elif not _is_integer(value):
            # 실행기가 `int(...)` 로 조용히 잘라내면 계약이 뜻과 다르게 돈다.
            problems.append(f"{path}.{name}: 구조 인자는 정수여야 한다 ({value!r})")
            ok = False
    for name in operator.thresholds:
        if name in node:
            problem = _threshold_problem(node[name], f"{path}.{name}", allow_unresolved)
            if problem:
                problems.append(problem)
                ok = False

    if str(op) == "primitive":
        name = resolve(str(node.get("primitive_id", "")))
        if name not in FEATURES:
            problems.append(f"{path}: 레지스트리에 없는 feature '{node.get('primitive_id')}'")
            return None

    kids: dict[str, TypeInfo] = {}
    for name, want in operator.inputs.items():
        if name not in node:
            ok = False
            continue
        if want == "boolean_list":
            items = node[name]
            if not isinstance(items, list) or not items:
                problems.append(f"{path}.{name}: bool 표현식 목록이어야 한다")
                ok = False
                continue
            for index, item in enumerate(items):
                info = _infer(item, f"{path}.{name}[{index}]", problems, allow_unresolved)
                if info is None:
                    ok = False
                elif not _is_boolean(info):
                    problems.append(f"{path}.{name}[{index}]: bool 이 필요한데 "
                                    f"{info.value_type}/{info.dimension} 이다")
                    ok = False
            continue
        info = _infer(node[name], f"{path}.{name}", problems, allow_unresolved)
        if info is None:
            ok = False
        elif want == "numeric" and not _is_numeric(info):
            problems.append(f"{path}.{name}: 수치가 필요한데 "
                            f"{info.value_type}/{info.dimension} 이다")
            ok = False
        elif want == "boolean" and not _is_boolean(info):
            problems.append(f"{path}.{name}: bool 이 필요한데 "
                            f"{info.value_type}/{info.dimension} 이다")
            ok = False
        else:
            kids[name] = info

    if not ok or operator.output is None:
        return None
    try:
        return operator.output(kids, node)
    except Exception as error:                          # pragma: no cover - 방어
        problems.append(f"{path}: 출력 타입을 정할 수 없다 ({error})")
        return None


# ---- 구 계약 읽기 --------------------------------------------------------------

# 구 계약이 쓰던 시간 인자. `unit` 은 출력 단위가 아니라 시간 기준이었다.
LEGACY_TIME_UNITS = {"ticks": "tick", "tick": "tick", "seconds": "clock", "sec": "clock",
                     "s": "clock", "clock": "clock"}


def migrate_expression(expression: Any) -> tuple[Any, list[dict[str, str]]]:
    """구 계약을 정본 어휘로 옮긴다. **무엇을 바꿨는지 함께 돌려준다.**

    `resolve()` 는 조용히 바꾼다 — 실행 경로라 그래야 한다. 하지만 과거 계약을 읽어
    다시 저장할 때 조용히 바꾸면, 그 계약이 원래 무엇이라고 쓰여 있었는지가 사라진다.
    돌려받은 기록은 run metadata 나 migration log 에 남긴다.

    새 계약에는 구 이름을 쓰지 않는다 — 옮긴 결과만 저장한다.
    """
    changes: list[dict[str, str]] = []

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, Mapping):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "primitive_id" and isinstance(value, str) and value in ALIASES:
                    changes.append({"path": f"{path}.{key}", "kind": "feature_alias",
                                    "legacy": value, "canonical": ALIASES[value]})
                    out[key] = ALIASES[value]
                elif (key == "unit" and isinstance(value, str) and "time_basis" not in node
                        and "time_basis" in OPERATORS.get(str(node.get("op")), Operator("", ())).domains):
                    canonical = LEGACY_TIME_UNITS.get(value, "clock")
                    changes.append({"path": f"{path}.unit", "kind": "operator_argument",
                                    "legacy": f"unit={value}", "canonical": f"time_basis={canonical}"})
                    out["time_basis"] = canonical
                else:
                    out[key] = walk(value, f"{path}.{key}")
            return out
        if isinstance(node, list):
            return [walk(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return node

    return walk(expression, "$"), changes


# ---- 관측 통계 (정의가 아니다) -------------------------------------------------

@dataclass(frozen=True)
class FeatureStats:
    """한 feature 가 특정 데이터셋에서 어떤 값을 가졌는가."""

    min: float
    max: float
    p1: float
    p99: float
    observations: int
    cases: int


@dataclass(frozen=True)
class ObservationProfile:
    """카탈로그가 아니라 **데이터**의 성질. 데이터가 바뀌면 이것만 바뀐다.

    `catalog_hash` 는 측정 당시의 카탈로그다. 지금 카탈로그와 다르면
    `audit()` 이 그 사실을 드러낸다.
    """

    profile_id: str
    catalog_hash: str
    dataset_hash: str
    market: str
    date_range: tuple[str, str]
    symbols: tuple[str, ...]
    stats: Mapping[str, FeatureStats]


def measure_profile(
    cases: Iterable[tuple[str, Book]],
    *,
    profile_id: str,
    dataset_hash: str,
    market: str,
    date_range: tuple[str, str],
    symbols: Sequence[str] = (),
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> ObservationProfile:
    """각 feature 의 관측 범위를 데이터에서 잰다.

    카탈로그에 숫자를 손으로 적으면 데이터가 바뀔 때 조용히 썩는다. 이 함수로 다시
    재고, 언제 무엇으로 쟀는지 profile 에 함께 남긴다.

    종목 간 비교가 되는지는 여기서 판단하지 않는다 — `Feature.normalization_scope`
    가 이미 그것을 말한다. 단위 문자열로 추측하던 구 로직은 없앴다.
    """
    collected: dict[str, list[np.ndarray]] = {}
    case_count = 0
    for _, book in cases:
        case_count += 1
        for name, values in compute_features(FEATURES.keys(), book).items():
            finite = values[np.isfinite(values)]
            if len(finite):
                collected.setdefault(name, []).append(finite)
    stats: dict[str, FeatureStats] = {}
    for name, chunks in collected.items():
        vector = np.concatenate(chunks)
        low, high = np.percentile(vector, percentiles)
        stats[name] = FeatureStats(
            min=float(vector.min()), max=float(vector.max()),
            p1=float(low), p99=float(high),
            observations=int(len(vector)), cases=case_count)
    return ObservationProfile(
        profile_id=profile_id, catalog_hash=catalog_hash(), dataset_hash=dataset_hash,
        market=market, date_range=date_range, symbols=tuple(symbols), stats=stats)


# 실측 관측 범위. `measure_profile()` 가 생성한 값이다. 손으로 고치지 말고 다시 재서
# 갈아끼운다. 반올림하지 않는다 — 반올림하면 다시 재도 이 숫자가 나오지 않아서
# `profile_hash()` 로 대조할 수 없다.
# 측정: 2026-08-19 · 정규장 · 클러스터당 앞 4종목(28개) × 2일 = 56 케이스 · 865,939 틱.
# 어느 종목이었는지를 여기 적는 이유는 그것 없이는 이 숫자를 다시 만들 수 없기
# 때문이다 — 구 측정은 "24 종목" 이라고만 적혀 있어 재현이 불가능했다.
DEFAULT_PROFILE_SYMBOLS = (
    "469170", "475050", "298380", "130680", "017390", "014790", "215200", "060590",
    "012170", "240600", "303030", "288330", "012200", "160190", "361390", "128820",
    "310960", "380340", "459790", "454320", "082210", "091440", "254120", "237750",
    "088350", "089030", "319400", "316140",
)

DEFAULT_PROFILE = ObservationProfile(
    profile_id="core4_20260317_20260319.v1",
    catalog_hash="57e864a1e235ffc3",
    dataset_hash=sha256_json({
        "universe_members_sha256":
            "1dc3b0cf996a94ccab92b1434813dbcd5292efb4c1af11bf1d8a18c6f3704b20",
        "symbols": list(DEFAULT_PROFILE_SYMBOLS),
        "dates": ["20260317", "20260319"], "session": "regular"})[:16],
    market="KRX",
    date_range=("20260317", "20260319"),
    symbols=DEFAULT_PROFILE_SYMBOLS,
    stats={name: FeatureStats(**values) for name, values in {
        "ask_1": {"min": 724.0, "max": 197800.0, "p1": 816.0, "p99": 194800.0,
                 "observations": 865939, "cases": 56},
        "ask_best_price_drop_count_50t": {"min": 0.0, "max": 28.0, "p1": 0.0, "p99": 14.0,
                 "observations": 865939, "cases": 56},
        "bid_1": {"min": 609.0, "max": 197700.0, "p1": 812.0, "p99": 194700.0,
                 "observations": 865939, "cases": 56},
        "bid_best_price_drop_count_50t": {"min": 0.0, "max": 26.0, "p1": 0.0, "p99": 13.0,
                 "observations": 865939, "cases": 56},
        "bid_queue_depth_at_l1": {"min": 1.0, "max": 123294.0, "p1": 1.0, "p99": 43520.77999999991,
                 "observations": 865939, "cases": 56},
        "book_imbalance": {"min": -0.9999406316789361, "max": 0.999980671473027, "p1": -0.9995983129142398, "p99": 0.9992622648469199,
                 "observations": 865939, "cases": 56},
        "book_imbalance_velocity": {"min": -1.9993457221356397, "max": 1.9995098491927985, "p1": -1.2807272740642808, "p99": 1.2722342556741675,
                 "observations": 865883, "cases": 56},
        "book_slope_ask": {"min": 4.8193351727129245e-05, "max": 120.0, "p1": 0.0002327692558366891, "p99": 3.0303030303030303,
                 "observations": 845860, "cases": 56},
        "book_slope_bid": {"min": 4.450031706475909e-05, "max": 55.55555555555556, "p1": 8.697391652243493e-05, "p99": 1.7167381974248928,
                 "observations": 845864, "cases": 56},
        "microprice_dev_bps": {"min": -534.6732552329133, "max": 1338.2779871046216, "p1": -18.912500612055393, "p99": 18.636398496816945,
                 "observations": 865939, "cases": 56},
        "microprice_velocity": {"min": -2150.6332230679473, "max": 1494.0465627166823, "p1": -11.823005265980877, "p99": 11.653354492624555,
                 "observations": 865883, "cases": 56},
        "mid_price": {"min": 723.0, "max": 197750.0, "p1": 814.5, "p99": 194750.0,
                 "observations": 865939, "cases": 56},
        "mid_return_20t_bps": {"min": -1646.6826538768987, "max": 1666.6666666666674, "p1": -39.99131924614525, "p99": 40.92071611253134,
                 "observations": 865939, "cases": 56},
        "mid_return_5t_bps": {"min": -1666.6666666666663, "max": 1594.5017182130593, "p1": -19.900497512437276, "p99": 18.963337547408532,
                 "observations": 865939, "cases": 56},
        "ofi_cks_1": {"min": -182544.0, "max": 167262.0, "p1": -10000.0, "p99": 10000.0,
                 "observations": 865939, "cases": 56},
        "ofi_depth_10": {"min": -685280.0, "max": 622507.0, "p1": -64268.43999999999, "p99": 63676.0,
                 "observations": 865939, "cases": 56},
        "ofi_depth_5": {"min": -380139.0, "max": 357916.0, "p1": -51125.0, "p99": 50066.47999999998,
                 "observations": 865939, "cases": 56},
        "queue_imbalance_best": {"min": 2.9684160531940157e-05, "max": 0.9999903357365135, "p1": 0.0002008435428800964, "p99": 0.99963113242346,
                 "observations": 865939, "cases": 56},
        "signed_aggr_flow_100": {"min": -1.0, "max": 1.0, "p1": -0.9999999999999915, "p99": 0.9999999999999906,
                 "observations": 865939, "cases": 56},
        "signed_aggr_flow_20": {"min": -1.0, "max": 1.0, "p1": -0.9999999999999992, "p99": 0.9999999999999992,
                 "observations": 865939, "cases": 56},
        "spread_bps": {"min": 0.971298139964062, "max": 3153.526970954357, "p1": 4.022930705018606, "p99": 74.21150278293135,
                 "observations": 865939, "cases": 56},
        "trade_event_indicator": {"min": 0.0, "max": 1.0, "p1": 0.0, "p99": 1.0,
                 "observations": 865939, "cases": 56},
        "vol_flow": {"min": 0.0, "max": 216116.0, "p1": 0.0, "p99": 13072.0,
                 "observations": 865939, "cases": 56},
    }.items()})


# ---- workflow 정책 (정의가 아니다) ---------------------------------------------

@dataclass(frozen=True)
class WorkflowPolicy:
    """어느 단계가 어느 feature 를 쓸 수 있는가. 카탈로그는 이것을 정하지 않는다."""

    hypothesis_visible: frozenset[str]
    grounding_visible: frozenset[str]
    guard_searchable: frozenset[str]
    execution_only: frozenset[str]


def default_policy() -> WorkflowPolicy:
    """Feature 사용 정책에서 유도한 기본 정책.

      · 미래를 보는 feature 는 어디에도 노출하지 않는다
      · `execution_only`·입력 전용·중복 feature 는 Agent/search에서 뺀다
      · 자동 guard 격자는 **직전 100 tick 종목 q**가 허용된 feature만 쓴다
    """
    causal = {n for n, f in FEATURES.items() if f.causal}
    visible = {
        n for n in causal
        if not set(feature_use_policy(n).threshold_kinds) & {
            THRESHOLD_EXECUTION_ONLY,
            THRESHOLD_DERIVED_INPUT_ONLY,
            THRESHOLD_DERIVED_DUPLICATE,
        }
    }
    q_searchable = {
        n for n in visible
        if (THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE in feature_use_policy(n).threshold_kinds
            and feature_use_policy(n).automatic_q_search)
    }
    execution_only = {
        n for n in causal
        if THRESHOLD_EXECUTION_ONLY in feature_use_policy(n).threshold_kinds
    }
    return WorkflowPolicy(
        hypothesis_visible=frozenset(visible),
        grounding_visible=frozenset(visible),
        guard_searchable=frozenset(q_searchable),
        execution_only=frozenset(execution_only),
    )


def guard_axes(policy: WorkflowPolicy | None = None) -> tuple[str, ...]:
    """후보 격자가 자를 수 있는 축. 정책에서 유도한다.

    구 `ledger.FEATURE_COLUMNS` 를 대체한다. 별도 상수로 두면 어휘와 어긋난다.
    """
    return tuple(sorted((policy or default_policy()).guard_searchable))


# ---- Agent 어휘 ---------------------------------------------------------------

# Agent 에게 조건으로 쓰라고 줄 수 없는 임계 종류. 어휘에는 남기되 임계 연산자의
# 직접 입력이 되는 것만 막는다 — 다른 식의 재료로는 쓸 수 있다.
NON_CONDITION_THRESHOLD_KINDS = frozenset({
    THRESHOLD_EXECUTION_ONLY,      # 가격 수준. 조건이 아니라 실행값이다
    THRESHOLD_DERIVED_DUPLICATE,   # 다른 feature 의 단조 변환. 같은 축을 두 번 건다
    THRESHOLD_DERIVED_INPUT_ONLY,  # 절대 원값 조건 금지. 비율 형태를 쓴다
})


# 한 가설을 서로 다른 형태로 표현한 것들. 같은 식을 이름만 바꾼 것(cosmetic rewrite)이
# 아니라 정말 다른 관측 형태여야 한다. 옛 Grounding 의 representation 정의를 그대로 쓴다.
REPRESENTATIONS = ("LEVEL", "PRE_ANCHOR_CHANGE", "PERSISTENCE", "SEQUENCE", "COMPOSITE")

# claim 이 관측을 만들 때 쓸 수 있는 연산. **같은 순간에 feature 를 합치는 것만** 둔다.
# 시간 연산(difference·persistence·sequence·rolling_*)은 variant 쪽 형태이므로 여기 없다.
# 그것까지 claim 으로 내리면 representation 이 뜻을 잃는다.
# 비교 연산도 없다 — claim 은 수량이지 조건이 아니다. 어디서 자를지는 variant 가 정한다.
CLAIM_OPS = frozenset({"primitive", "raw", "level_aggregate",
                       "ratio", "add", "subtract", "multiply", "divide",
                       "absolute", "negate", "log1p"})

_TEMPORAL_OPS = frozenset({"difference", "persistence", "sequence", "sequence_once"})


def expression_ops(expression: Any) -> set[str]:
    """식에 쓰인 연산자 이름."""
    out: set[str] = set()
    stack = [expression]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if node.get("op"):
                out.add(str(node["op"]))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return out


def representation_matches(representation: str, expression: Any) -> bool:
    """선언한 표현 형태가 식과 실제로 맞는가.

    `PRE_ANCHOR_CHANGE` 라고 써놓고 `difference` 를 안 쓰면 거짓이다. 선언만 바꾼
    variant 를 걸러내려고 기계적으로 확인한다.
    """
    ops = expression_ops(expression)
    if representation == "LEVEL":
        return not ops & _TEMPORAL_OPS
    if representation == "PRE_ANCHOR_CHANGE":
        return "difference" in ops
    if representation == "PERSISTENCE":
        return "persistence" in ops
    if representation == "SEQUENCE":
        return bool(ops & {"sequence", "sequence_once"})
    if representation == "COMPOSITE":
        return bool(ops & {"all", "any", "sequence", "sequence_once"})
    return False


def condition_problem(name: str) -> str | None:
    """이 feature 를 임계 조건의 직접 대상으로 쓸 수 있는가. 못 쓰면 이유와 대안.

    실측으로 확인한 중복이다 (005930 / 20260316 / 193,602틱):
      queue_imbalance_best 는 book_imbalance 와 Spearman 1.000000
      microprice_dev_bps 는 book_imbalance/2 x spread_bps 와 절대오차 0.000000
    """
    resolved = resolve(str(name))
    if resolved not in FEATURES:
        return None
    policy = feature_use_policy(resolved)
    blocked = set(policy.threshold_kinds) & NON_CONDITION_THRESHOLD_KINDS
    if not blocked:
        return None
    detail = policy.as_agent_dict()
    alternative = detail.get("duplicate_of")
    hint = f" 대신 {alternative} 를 쓴다." if alternative else ""
    return (f"{resolved} 는 임계 조건의 대상이 될 수 없다 ({sorted(blocked)[0]}).{hint} "
            f"규칙: {detail.get('note') or '카탈로그 정책'}. "
            f"다른 식의 재료로는 쓸 수 있다")


def agent_causal_policy() -> WorkflowPolicy:
    """Agent 어휘용 정책. causal feature 를 전부 보여준다.

    조건으로 못 쓰는 것도 어휘에 남긴다 — 숨기면 Agent 가 존재를 모르고, 실측상
    Agent 가 조합을 거의 만들지 못하므로(9개 전략 중 3개만) 숨기는 것이 곧 축을
    없애는 일이 된다. 대신 `threshold_policy` 로 규칙을 알려주고 검증이 확인한다.
    """
    causal = frozenset(name for name, spec in FEATURES.items() if spec.causal)
    return dataclasses.replace(default_policy(),
                               hypothesis_visible=causal, grounding_visible=causal)


def agent_vocabulary(
    *,
    profile: ObservationProfile | None = DEFAULT_PROFILE,
    policy: WorkflowPolicy | None = None,
    capabilities: DataCapabilities | None = None,
    visibility: str = "grounding_visible",
) -> dict[str, Any]:
    """Agent 에게 줄 어휘. 카탈로그 + 관측 profile + 정책의 조합이다.

    손으로 적으면 안 된다. 구 구조에서 두 번 다 손으로 적은 목록에서 사고가 났다 —
    실행되지 않는 이름을 줬고, 인자의 허용값을 틀렸다.

    별칭은 싣지 않는다. 실으면 Agent 가 다시 옛 이름을 만들어낸다 — 별칭은 과거 계약을
    읽을 때만 쓴다. 파이썬 구현·내부 함수명·해시도 싣지 않는다.

    `observed` 는 profile 에서 붙인다. 범위를 모르면 Agent 가 임계를 정의역 밖에 두어
    조건이 항상 참이거나 항상 거짓이 된다 (실측: 비율 feature 에 `>= -15`).
    """
    policy = policy or default_policy()
    visible = getattr(policy, visibility)
    reasons = ({a.feature: a for a in feature_availability(capabilities)}
               if capabilities is not None else {})
    features = []
    for name in sorted(visible):
        spec = FEATURES[name]
        stats = profile.stats.get(name) if profile else None
        availability = reasons.get(name)
        features.append({
            "name": spec.name,
            "definition_id": spec.definition_id,
            "value_type": spec.value_type,
            "dimension": spec.dimension,
            "unit": spec.unit,
            "observable_description": spec.observable_description,
            "economic_interpretation": spec.economic_interpretation,
            "invalid_policy": spec.invalid_policy,
            "dependencies": list(spec.dependencies),
            "time_basis": spec.time_basis,
            "lookback": spec.lookback,
            "current_included": spec.current_included,
            "normalization_scope": spec.normalization_scope,
            "tags": list(spec.tags),
            "threshold_policy": feature_use_policy(name).as_agent_dict(),
            "observed": (None if stats is None else
                         {"min": stats.min, "max": stats.max, "p1": stats.p1, "p99": stats.p99,
                          "observations": stats.observations, "cases": stats.cases}),
            "available": True if availability is None else availability.available,
            "unavailable_reason": None if availability is None else availability.reason,
        })
    for item in features:
        # 문제 없는 feature 에 available/unavailable_reason 을 실으면 프롬프트만 키운다.
        # definition_id 는 어휘 계약이라 남긴다 (test_catalog).
        if item.get("available") is True and item.get("unavailable_reason") is None:
            item.pop("available", None)
            item.pop("unavailable_reason", None)
    return {
        # 내부 해시는 싣지 않는다 — Agent 가 쓸 수 없고, run 출처는 `provenance()`
        # 가 manifest 에 따로 남긴다.
        "schema": "framework_catalog.v3",
        "observation_profile_id": profile.profile_id if profile else None,
        "features": features,
        "operators": [
            {"op": o.op, "required": list(o.required),
             "inputs": dict(o.inputs),
             "output": o.output_doc,
             "domains": {k: list(v) for k, v in o.domains.items()},
             "thresholds": list(o.thresholds),
             "temporal": o.temporal, "doc": o.doc}
            for o in sorted(OPERATORS.values(), key=lambda x: x.op)
        ],
        "rules": {
            "threshold_placeholder": UNRESOLVED_PREFIX + "<NAME>",
            "threshold_arguments": sorted(
                {f"{o.op}.{name}" for o in OPERATORS.values() for name in o.thresholds}),
            "structural_params_are_literal_integers": sorted(STRUCTURAL_PARAMS),
            "output_units_are_inferred": True,
            "economic_interpretation_is_not_identity":
                "`economic_interpretation` 은 해석이지 관측의 정체가 아니다. "
                "가설이 뜻하는 관측량은 `observable_description` 으로 고른다",
            "free_form_python": False,
        },
    }


# ---- 해시 ---------------------------------------------------------------------

def catalog_hash() -> str:
    """레지스트리 전체의 해시. **계산 정의가 바뀌면 값이 바뀐다.**

    구 해시는 이름·단위·flag 만 담아서 수식이 바뀌어도 같은 값이 나왔다. 여기에는
    의존·depth·시간 의미·정의 AST 까지 넣는다. 정책과 관측 profile 은 넣지 않는다 —
    그것은 카탈로그가 아니다.
    """
    return sha256_json({
        "round_trip_cost_bps": FEE_BPS,
        "features": [[f.name, f.definition_id, f.value_type, f.dimension, f.unit,
                      list(f.dependencies), f.min_book_depth, f.time_basis, f.lookback,
                      f.current_included, f.causal, f.definition, f.invalid_policy]
                     for f in sorted(FEATURES.values(), key=lambda x: x.name)],
        "operators": [[o.op, list(o.required), dict(o.inputs), o.output_doc,
                       {k: list(v) for k, v in o.domains.items()}, list(o.thresholds)]
                      for o in sorted(OPERATORS.values(), key=lambda x: x.op)],
    })[:16]


def policy_hash(policy: WorkflowPolicy | None = None) -> str:
    policy = policy or default_policy()
    return sha256_json({
        "hypothesis_visible": sorted(policy.hypothesis_visible),
        "grounding_visible": sorted(policy.grounding_visible),
        "guard_searchable": sorted(policy.guard_searchable),
        "execution_only": sorted(policy.execution_only),
        "feature_use_policies": {
            name: FEATURE_USE_POLICIES[name].as_agent_dict()
            for name in sorted(FEATURE_USE_POLICIES)
        },
    })[:16]


def profile_hash(profile: ObservationProfile | None = DEFAULT_PROFILE) -> str:
    if profile is None:
        return ""
    return sha256_json({
        "profile_id": profile.profile_id,
        "catalog_hash": profile.catalog_hash,
        "dataset_hash": profile.dataset_hash,
        "market": profile.market,
        "date_range": list(profile.date_range),
        "symbols": list(profile.symbols),
        "stats": {name: [s.min, s.max, s.p1, s.p99, s.observations, s.cases]
                  for name, s in sorted(profile.stats.items())},
    })[:16]


def provenance(profile: ObservationProfile | None = DEFAULT_PROFILE,
               policy: WorkflowPolicy | None = None) -> dict[str, Any]:
    """run 마다 남길 최소 출처. 어떤 어휘·통계·정책으로 실험했는지."""
    return {
        "catalog_sha256": catalog_hash(),
        "observation_profile_id": profile.profile_id if profile else None,
        "observation_profile_sha256": profile_hash(profile),
        "workflow_policy_sha256": policy_hash(policy),
    }


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    """NaN 자리까지 같은가. NaN != NaN 이라 그냥 비교하면 늘 다르다."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return False
    finite = ~np.isnan(a)
    return bool(np.array_equal(np.isnan(a), np.isnan(b))
                and np.allclose(a[finite], b[finite], rtol=0, atol=1e-12))


# ---- 기동 점검 ----------------------------------------------------------------

def audit(profile: ObservationProfile | None = DEFAULT_PROFILE,
          policy: WorkflowPolicy | None = None,
          parity_book: Book | None = None) -> dict[str, Any]:
    """레지스트리 정합성 점검. 기동 시 한 번 부른다.

    구 구조에서 조용히 어긋나 있던 것을 여기서 눈에 보이게 한다. `problems` 가 비어
    있지 않으면 카탈로그가 깨진 것이다.

    `parity_book` 을 주면 `definition` 이 있는 feature 에 대해 정의와 구현이 같은
    값을 내는지까지 본다. 데이터가 있어야 하는 검사라 기본은 건너뛴다 — 데이터
    없이 도는 검사와 섞으면 기동이 데이터에 묶인다.
    """
    policy = policy or default_policy()
    problems: list[str] = []

    definition_ids = [f.definition_id for f in FEATURES.values()]
    if len(set(definition_ids)) != len(definition_ids):
        problems.append("definition_id 가 중복됐다")
    for alias, target in ALIASES.items():
        if target not in FEATURES:
            problems.append(f"별칭 {alias} 이 없는 feature {target} 을 가리킨다")

    missing_policies = sorted(set(FEATURES) - set(FEATURE_USE_POLICIES))
    extra_policies = sorted(set(FEATURE_USE_POLICIES) - set(FEATURES))
    if missing_policies:
        problems.append(f"사용 정책이 없는 feature: {missing_policies}")
    if extra_policies:
        problems.append(f"없는 feature의 사용 정책: {extra_policies}")
    for name, use in FEATURE_USE_POLICIES.items():
        unknown_kinds = sorted(set(use.threshold_kinds) - FEATURE_THRESHOLD_KINDS)
        if unknown_kinds:
            problems.append(f"{name}: 모르는 threshold policy {unknown_kinds}")
        if not use.threshold_kinds:
            problems.append(f"{name}: threshold policy가 비었다")
        if (use.automatic_q_search and THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
                not in use.threshold_kinds):
            problems.append(f"{name}: 자동 q 탐색인데 직전 100 tick q가 허용되지 않았다")

    for name, spec in FEATURES.items():
        unknown = [d for d in spec.dependencies if d not in RAW_FIELDS]
        if unknown:
            problems.append(f"{name}: 모르는 원본 필드 {unknown}")
        if not spec.dependencies:
            problems.append(f"{name}: 원본 의존이 비어 있다")
        if spec.min_book_depth is not None and spec.min_book_depth < 1:
            problems.append(f"{name}: min_book_depth 가 {spec.min_book_depth} 다")
        if spec.value_type not in VALUE_TYPES:
            problems.append(f"{name}: 모르는 value_type {spec.value_type}")
        if spec.dimension not in DIMENSIONS:
            problems.append(f"{name}: 모르는 dimension {spec.dimension}")
        if spec.time_basis not in TIME_BASES:
            problems.append(f"{name}: 모르는 time_basis {spec.time_basis}")
        if spec.normalization_scope not in NORMALIZATION_SCOPES:
            problems.append(f"{name}: 모르는 normalization_scope {spec.normalization_scope}")
        if not spec.observable_description:
            problems.append(f"{name}: observable_description 이 비었다")
        if not spec.invalid_policy:
            problems.append(f"{name}: invalid_policy 가 비었다")
        if spec.definition is not None:
            try:
                info = infer_expression_type(spec.definition)
            except ExpressionError as error:
                problems.append(f"{name}: definition 이 검증을 통과하지 못한다 ({error})")
            else:
                if info != spec.type_info:
                    problems.append(f"{name}: definition 타입 {info} 이 선언 {spec.type_info} 과 다르다")

    for name in sorted(policy.hypothesis_visible | policy.grounding_visible
                       | policy.guard_searchable):
        if name not in FEATURES:
            problems.append(f"정책이 없는 feature {name} 을 노출한다")
        elif not FEATURES[name].causal:
            problems.append(f"{name}: 미래를 보는 feature 가 Agent/search 에 노출됐다")

    for name in sorted(policy.guard_searchable):
        if (name in FEATURES and THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
                not in feature_use_policy(name).threshold_kinds):
            problems.append(f"{name}: 직전 100 tick q가 허용되지 않았는데 자동 guard 축이다")
    for name in sorted(policy.execution_only):
        if (name in FEATURES and THRESHOLD_EXECUTION_ONLY
                not in feature_use_policy(name).threshold_kinds):
            problems.append(f"{name}: execution_only 정책과 다르다")

    for operator in OPERATORS.values():
        if operator.output is None:
            problems.append(f"연산자 {operator.op}: 출력 타입 규칙이 없다")
        for name in operator.inputs:
            if name not in operator.required:
                problems.append(f"연산자 {operator.op}: 입력 '{name}' 이 required 에 없다")

    parity = "not_checked"
    if parity_book is not None:
        from .contract import ExpressionRuntime   # 순환 import 를 피해 여기서 부른다
        runtime = ExpressionRuntime(parity_book.data)
        parity = "checked"
        for name, spec in sorted(FEATURES.items()):
            if spec.definition is None:
                continue
            if not _same_values(spec.compute(parity_book), runtime.evaluate(spec.definition)):
                problems.append(f"{name}: definition 과 compute 가 다른 값을 낸다")

    stale_profile = bool(profile) and profile.catalog_hash != catalog_hash()
    if profile:
        for name in profile.stats:
            if name not in FEATURES:
                problems.append(f"profile 에 없는 feature {name} 의 통계가 있다")

    guard = set(guard_axes(policy))
    return {
        "features": len(FEATURES),
        "grounding_visible": len(policy.grounding_visible),
        "guard_axes": len(guard),
        "needs_trades": sorted(n for n, f in FEATURES.items()
                               if set(f.dependencies) & set(TRADES)),
        "grounding_only": sorted(policy.grounding_visible - guard),
        "execution_only": sorted(policy.execution_only),
        "with_definition": sorted(n for n, f in FEATURES.items() if f.definition is not None),
        "aliases": dict(ALIASES),
        "grid_size": len(guard) * 5 * 2,   # 축 x 분위수 x 부등호
        "profile_stale": stale_profile,
        "definition_parity": parity,
        "problems": problems,
        **provenance(profile, policy),
    }
