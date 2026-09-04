"""다섯 법칙의 입력·타깃 조립. 계획서 §4 E0 표 · §3 S0 "E0 전용 타깃".

## 설계 원칙 — 기존 산술을 다시 짜지 않는다

L1·L2·L3·L4 는 정본 Catalog 가 이미 계산한 feature(`sd.ticks.feature_matrix`)와
`framework.contract.ExpressionRuntime`(`sd.derived`·`sd.dimensionless` 가 이미
쓰는 바로 그 실행기)만으로 조립한다. 여기서 하는 일은 인덱스 이동(미래
전진)과 나눗셈뿐이다 — OFI·마이크로프라이스·서명 체결량 자체의 정의는
전부 vendor Catalog 에서 그대로 가져온다.

**예외: L5.** `sd.ticks.load_arrays` (→ `framework.data.load`) 는 체결을 호가
격자에 **누적**한다("호가를 시간 순 격자로 삼고 체결을 그 격자에 누적한다" —
`framework/data.py` 모듈 docstring). Hawkes 커널 검사는 개별 체결 사이의
간격이 필요한데, 누적된 격자에는 그 정보가 없다. 그래서 L5 만 raw parquet 을
직접 읽어 체결행(`data_type==11`)의 `local_time` 을 그대로 쓴다 — 다만 시간
변환(`packed_time_to_seconds`)과 세션 경계(`SESSION_START/END`)는
`framework.data` 에서 그대로 가져와 같은 정의를 쓴다 (두 번째 진실 원천을
만들지 않는다).

## 무차원 vs raw 두 트랙

`LawSample` 은 항상 두 벌을 담는다 — `X_dimless`/`y_dimless`(본 트랙, S3 의
"마찰이 자연 단위다" 철학을 따른다)와 `X_raw`/`y_raw`(ablation "무차원화 없이
증류" 전용, 원 단위 그대로). 어느 쪽도 SR 입력이 유효하려면 `mask` 를
반드시 같이 봐야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import config, ticks

_framework = config.load_framework()
from framework import contract as _contract  # noqa: E402
from framework import data as _data          # noqa: E402

QBAR_WINDOW = 100          # sd.derived.QBAR_WINDOW 와 같은 관례("직전 몇 틱")를 쓴다.
OFI_WINDOW = 20            # L2·L3 "짧은 구간". mid_return_20t_bps 와 스케일을 맞춘다.
RV_LOOKBACK = 100          # L3·L4 실현변동성 추정 창.


@dataclass(frozen=True)
class LawSample:
    """법칙 하나·종목-일 하나의 입력·타깃. 행 수는 원본 tick 수와 같다."""

    names_dimless: tuple[str, ...]
    X_dimless: np.ndarray
    y_dimless: np.ndarray
    names_raw: tuple[str, ...]
    X_raw: np.ndarray
    y_raw: np.ndarray
    mask: np.ndarray
    y_description: str


@dataclass(frozen=True)
class Dataset:
    """여러 종목-일을 이어붙인 것. `assemble()` 의 반환값."""

    law: str
    names_dimless: tuple[str, ...]
    X_dimless: np.ndarray
    y_dimless: np.ndarray
    names_raw: tuple[str, ...]
    X_raw: np.ndarray
    y_raw: np.ndarray
    mask: np.ndarray
    y_description: str
    symbols_used: tuple[str, ...]
    symbols_skipped: dict[str, str]
    n_rows_total: int


# ---------------------------------------------------------------------------
# ExpressionRuntime AST 조립 헬퍼. sd/derived.py 와 같은 패턴 — 여기서도 AST 를
# 등록하고 ExpressionRuntime 으로 평가한다(별도 numpy 재구현을 두지 않는다).
# ---------------------------------------------------------------------------

def _primitive(name: str) -> dict:
    return {"op": "primitive", "primitive_id": name}


def _raw(field: str, level: int | None = None) -> dict:
    node: dict = {"op": "raw", "field": field}
    if level is not None:
        node["level"] = int(level)
    return node


def _ratio(numerator: dict, denominator: dict, *, zero_policy: str = "nan") -> dict:
    return {"op": "ratio", "numerator": numerator, "denominator": denominator,
            "zero_policy": zero_policy}


def _subtract(left: dict, right: dict) -> dict:
    return {"op": "subtract", "left": left, "right": right}


def _rolling(op: str, input_ast: dict, window: int) -> dict:
    return {"op": op, "input": input_ast, "window": int(window), "time_basis": "tick"}


def _qbar_ast() -> dict:
    return _rolling("rolling_mean", _primitive("bid_queue_depth_at_l1"), QBAR_WINDOW)


def _eval(arrays: dict, ast: dict) -> np.ndarray:
    return np.asarray(_contract.ExpressionRuntime(arrays).evaluate(ast), dtype=float)


def _forward_shift(x: np.ndarray, k: int) -> np.ndarray:
    """`y[t] = x[t+k]`. `x` 가 이미 Catalog 정의(예: `mid_return_5t_bps` — "5틱 전
    대비 수익률")를 쓴 배열이면, 이 함수는 그 값을 앞으로 당길 뿐이다 —
    `x[t+k]` 는 정의상 "t 에서 t+k 까지의 수익률"이므로 곧 t 시점에서 본
    **미래** 수익률이다. 새 산술을 더하지 않는다.
    """
    n = len(x)
    k = int(k)
    out = np.full(n, np.nan)
    if k < n:
        out[: n - k] = x[k:]
    return out


# ---------------------------------------------------------------------------
# L1 — 마이크로프라이스. (I, s/(s+c)) → 미래 5틱 mid 수익률. 계획서 §4 E0 표.
# ---------------------------------------------------------------------------

L1_FORWARD_TICKS = 5


def build_l1(arrays: dict[str, np.ndarray]) -> LawSample:
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    imbalance = matrix[:, idx["book_imbalance"]]
    friction_ratio = matrix[:, idx["spread_to_round_trip_cost_ratio"]]
    spread_bps = matrix[:, idx["spread_bps"]]
    ret_5t = matrix[:, idx["mid_return_5t_bps"]]

    y_forward_bps = _forward_shift(ret_5t, L1_FORWARD_TICKS)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_dimless = np.where(spread_bps > 0, y_forward_bps / spread_bps, np.nan)

    mask = (np.isfinite(imbalance) & np.isfinite(friction_ratio)
            & np.isfinite(spread_bps) & (spread_bps > 0) & np.isfinite(y_forward_bps))

    return LawSample(
        names_dimless=("book_imbalance", "spread_to_round_trip_cost_ratio"),
        X_dimless=np.column_stack([imbalance, friction_ratio]),
        y_dimless=y_dimless,
        names_raw=("book_imbalance", "spread_bps"),
        X_raw=np.column_stack([imbalance, spread_bps]),
        y_raw=y_forward_bps,
        mask=mask,
        y_description=f"forward {L1_FORWARD_TICKS}-tick mid return (bps), normalized by current spread_bps",
    )


# ---------------------------------------------------------------------------
# L2 — OFI 선형 법칙. rolling_sum(ofi_depth_5, 20)/Q̄ → 같은 20틱 창의 ΔP/s.
# 둘 다 t 에서 끝나는 과거 20틱 창이다(동시대 회귀) — CKS 원안의 구간-집계
# 형태.
# ---------------------------------------------------------------------------

def build_l2(arrays: dict[str, np.ndarray]) -> LawSample:
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]
    ret_20t = matrix[:, idx["mid_return_20t_bps"]]

    ofi_window = _eval(arrays, _rolling("rolling_sum", _primitive("ofi_depth_5"), OFI_WINDOW))
    qbar = _eval(arrays, _qbar_ast())
    with np.errstate(divide="ignore", invalid="ignore"):
        ofi_over_qbar = np.where(qbar > 0, ofi_window / qbar, np.nan)
        y_dimless = np.where(spread_bps > 0, ret_20t / spread_bps, np.nan)

    mask = (np.isfinite(ofi_over_qbar) & np.isfinite(spread_bps) & (spread_bps > 0)
            & np.isfinite(ret_20t) & np.isfinite(qbar) & (qbar > 0))

    return LawSample(
        names_dimless=(f"ofi_{OFI_WINDOW}t_over_qbar",),
        X_dimless=ofi_over_qbar.reshape(-1, 1),
        y_dimless=y_dimless,
        names_raw=(f"ofi_{OFI_WINDOW}t",),
        X_raw=ofi_window.reshape(-1, 1),
        y_raw=ret_20t,
        mask=mask,
        y_description=f"contemporaneous {OFI_WINDOW}-tick mid return (bps) / spread_bps",
    )


# ---------------------------------------------------------------------------
# L3′ — 집계 임팩트 제곱근 (프록시로 격하). (ΣV_signed/Q̄, σ) → 같은 창의 ΔP/s.
# ---------------------------------------------------------------------------

def _realized_vol_ast(lag_feature: str, lookback: int) -> dict:
    """`lag_feature`(예: mid_return_5t_bps) 계열의 직전 `lookback` 틱 표준편차.

    표준 RV 추정량(1틱 수익률의 구간 합)이 아니라 "이미 존재하는 k-스케일
    수익률의 국소 변동성"이다 — L4′ docstring 에 이 근사의 한계를 적어 뒀다.
    """
    return _rolling("rolling_std", _primitive(lag_feature), lookback)


def build_l3(arrays: dict[str, np.ndarray]) -> LawSample:
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]
    ret_20t = matrix[:, idx["mid_return_20t_bps"]]

    signed_volume = _eval(arrays, _rolling(
        "rolling_sum", _subtract(_raw("buy_volume"), _raw("sell_volume")), OFI_WINDOW))
    qbar = _eval(arrays, _qbar_ast())
    sigma_raw = _eval(arrays, _realized_vol_ast("mid_return_5t_bps", RV_LOOKBACK))

    with np.errstate(divide="ignore", invalid="ignore"):
        volume_over_qbar = np.where(qbar > 0, signed_volume / qbar, np.nan)
        sigma_dimless = np.where(spread_bps > 0, sigma_raw / spread_bps, np.nan)
        y_dimless = np.where(spread_bps > 0, ret_20t / spread_bps, np.nan)

    mask = (np.isfinite(volume_over_qbar) & np.isfinite(sigma_dimless)
            & np.isfinite(y_dimless) & (qbar > 0) & (spread_bps > 0))

    return LawSample(
        names_dimless=(f"signed_volume_{OFI_WINDOW}t_over_qbar", "rv5_over_spread"),
        X_dimless=np.column_stack([volume_over_qbar, sigma_dimless]),
        y_dimless=y_dimless,
        names_raw=(f"signed_volume_{OFI_WINDOW}t", "rv5_bps"),
        X_raw=np.column_stack([signed_volume, sigma_raw]),
        y_raw=ret_20t,
        mask=mask,
        y_description=f"contemporaneous {OFI_WINDOW}-tick mid return (bps) / spread_bps",
    )


# ---------------------------------------------------------------------------
# L4′ — 다중 스케일 RV 결합 (대체 형태). (RV5,RV20,RV100) → 다음 100틱 창의
# RV20-유사 추정량.
# ---------------------------------------------------------------------------

L4_FORWARD_TICKS = RV_LOOKBACK   # "다음 창" = 이 롤링 창의 길이만큼 앞.


def build_l4(arrays: dict[str, np.ndarray]) -> LawSample:
    matrix, names = ticks.feature_matrix(arrays)
    idx = {n: i for i, n in enumerate(names)}
    spread_bps = matrix[:, idx["spread_bps"]]

    rv5 = _eval(arrays, _realized_vol_ast("mid_return_5t_bps", RV_LOOKBACK))
    rv20 = _eval(arrays, _realized_vol_ast("mid_return_20t_bps", RV_LOOKBACK))
    rv100 = _eval(arrays, _realized_vol_ast("mid_return_100t_bps", RV_LOOKBACK))

    y_forward_raw = _forward_shift(rv20, L4_FORWARD_TICKS)
    with np.errstate(divide="ignore", invalid="ignore"):
        rv5_d = np.where(spread_bps > 0, rv5 / spread_bps, np.nan)
        rv20_d = np.where(spread_bps > 0, rv20 / spread_bps, np.nan)
        rv100_d = np.where(spread_bps > 0, rv100 / spread_bps, np.nan)
        y_dimless = np.where(spread_bps > 0, y_forward_raw / spread_bps, np.nan)

    mask = (np.isfinite(rv5_d) & np.isfinite(rv20_d) & np.isfinite(rv100_d)
            & np.isfinite(y_dimless) & (spread_bps > 0))

    return LawSample(
        names_dimless=("rv5_over_spread", "rv20_over_spread", "rv100_over_spread"),
        X_dimless=np.column_stack([rv5_d, rv20_d, rv100_d]),
        y_dimless=y_dimless,
        names_raw=("rv5_bps", "rv20_bps", "rv100_bps"),
        X_raw=np.column_stack([rv5, rv20, rv100]),
        y_raw=y_forward_raw,
        mask=mask,
        y_description=f"forward {L4_FORWARD_TICKS}-tick-ahead RV20-analogue (bps), /spread_bps",
    )


# ---------------------------------------------------------------------------
# L5 — Hawkes 커널. 체결행 원본 parquet 직접 읽기 (docstring 참조).
# ---------------------------------------------------------------------------

MIN_TRADE_GAP = 1e-6   # 초. 동시 체결(같은 local_time) 쌍은 뺀다 — 0으로 나누지 않는다.
RECENT_WINDOW_SECONDS = 1.0
INTENSITY_CLIP_QUANTILE = 0.99


def _load_trade_times(symbol: str, date: str) -> np.ndarray:
    """정규장 체결행의 `time_s`, 오름차순. `framework.data` 의 세션 경계·시간
    변환을 그대로 쓴다 — 새 정의를 만들지 않는다."""
    path = _data.find_parquet(symbol, date, root=config.TICK_ROOT)
    table = pq.read_table(path, columns=["data_type", "local_time"])
    table = table.filter(pc.and_(
        pc.and_(pc.greater_equal(table["local_time"], _data.SESSION_START),
                pc.less(table["local_time"], _data.SESSION_END)),
        pc.equal(table["data_type"], _data.TRADE_ROW)))
    local_time = np.array(table["local_time"].combine_chunks().to_numpy(zero_copy_only=False),
                          dtype=np.int64, copy=True)
    local_time.sort()
    return _data.packed_time_to_seconds(local_time)


def build_l5(symbol: str, date: str) -> LawSample:
    time_s = _load_trade_times(symbol, date)
    n = len(time_s)
    if n < 5:
        empty = np.zeros(0)
        return LawSample(names_dimless=("dt_prev_over_median", "recent_count_norm"),
                          X_dimless=empty.reshape(0, 2), y_dimless=empty,
                          names_raw=("dt_prev_seconds", "recent_count"),
                          X_raw=empty.reshape(0, 2), y_raw=empty,
                          mask=np.zeros(0, dtype=bool),
                          y_description="1/dt_next (Hz), interior trade events only")

    dt = np.diff(time_s)                                  # dt[i] = time_s[i+1]-time_s[i]
    dt_prev = np.concatenate([[np.nan], dt])               # dt_prev[i] = time_s[i]-time_s[i-1]
    dt_next = np.concatenate([dt, [np.nan]])               # dt_next[i] = time_s[i+1]-time_s[i]

    valid_gap = np.isfinite(dt_prev) & np.isfinite(dt_next) & (dt_prev > MIN_TRADE_GAP) & (dt_next > MIN_TRADE_GAP)
    positive_dt_prev = dt_prev[valid_gap & (dt_prev > 0)]
    median_dt = float(np.median(positive_dt_prev)) if positive_dt_prev.size else float("nan")

    # 직전 RECENT_WINDOW_SECONDS 안의 체결 수 — 오직 overcapacity 를 주기 위한
    # 두 번째 입력이다. 복원 판정은 dt_prev 축만 본다(다른 법칙들이 s·σ 를
    # 대표값에 고정하는 것과 같은 자세).
    left = np.searchsorted(time_s, time_s - RECENT_WINDOW_SECONDS, side="left")
    recent_count = (np.arange(n) - left).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_raw = np.where(dt_next > 0, 1.0 / dt_next, np.nan)
    if np.isfinite(y_raw).sum() > 10:
        cap = np.nanquantile(y_raw, INTENSITY_CLIP_QUANTILE)
        y_raw = np.minimum(y_raw, cap)

    dt_prev_norm = dt_prev / median_dt if median_dt > 0 else np.full(n, np.nan)
    y_dimless = y_raw * median_dt if median_dt > 0 else np.full(n, np.nan)
    recent_count_norm = recent_count * median_dt if median_dt > 0 else np.full(n, np.nan)

    mask = valid_gap & np.isfinite(y_raw) & np.isfinite(dt_prev_norm) & np.isfinite(median_dt)

    return LawSample(
        names_dimless=("dt_prev_over_median", "recent_count_norm"),
        X_dimless=np.column_stack([dt_prev_norm, recent_count_norm]),
        y_dimless=y_dimless,
        names_raw=("dt_prev_seconds", "recent_count"),
        X_raw=np.column_stack([dt_prev, recent_count]),
        y_raw=y_raw,
        mask=mask,
        y_description="1/dt_next (Hz, 상위 1% 분위로 clip), interior trade events only",
    )


# ---------------------------------------------------------------------------
# 종목 반복 → 하나의 Dataset. L1~L4 는 (symbol,date)→arrays→builder, L5 는
# (symbol,date)→builder 로 시그니처가 달라 두 갈래로 나눈다.
# ---------------------------------------------------------------------------

BUILDERS: dict[str, Callable] = {
    "L1": build_l1, "L2": build_l2, "L3": build_l3, "L4": build_l4,
}


def assemble(law: str, symbols: Sequence[str], date: str) -> Dataset:
    if law == "L5":
        return _assemble_l5(symbols, date)
    builder = BUILDERS[law]
    samples: list[LawSample] = []
    used: list[str] = []
    skipped: dict[str, str] = {}
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, date)
            sample = builder(arrays)
        except Exception as error:  # noqa: BLE001 — 종목 하나의 실패로 전체가 죽지 않는다
            skipped[symbol] = f"{type(error).__name__}: {error}"
            continue
        samples.append(sample)
        used.append(symbol)
    if not samples:
        raise ValueError(f"{law}: 적재된 종목이 하나도 없다 ({skipped})")
    return _concat(law, samples, tuple(used), skipped)


def _assemble_l5(symbols: Sequence[str], date: str) -> Dataset:
    samples: list[LawSample] = []
    used: list[str] = []
    skipped: dict[str, str] = {}
    for symbol in symbols:
        try:
            sample = build_l5(symbol, date)
        except Exception as error:  # noqa: BLE001
            skipped[symbol] = f"{type(error).__name__}: {error}"
            continue
        if sample.mask.sum() == 0:
            skipped[symbol] = "체결 이벤트가 5개 미만이거나 유효 간격이 없음"
            continue
        samples.append(sample)
        used.append(symbol)
    if not samples:
        raise ValueError(f"L5: 적재된 종목이 하나도 없다 ({skipped})")
    return _concat("L5", samples, tuple(used), skipped)


def _concat(law: str, samples: list[LawSample], used: tuple[str, ...],
           skipped: dict[str, str]) -> Dataset:
    first = samples[0]
    return Dataset(
        law=law,
        names_dimless=first.names_dimless,
        X_dimless=np.concatenate([s.X_dimless for s in samples], axis=0),
        y_dimless=np.concatenate([s.y_dimless for s in samples], axis=0),
        names_raw=first.names_raw,
        X_raw=np.concatenate([s.X_raw for s in samples], axis=0),
        y_raw=np.concatenate([s.y_raw for s in samples], axis=0),
        mask=np.concatenate([s.mask for s in samples], axis=0),
        y_description=first.y_description,
        symbols_used=used,
        symbols_skipped=skipped,
        n_rows_total=sum(len(s.mask) for s in samples),
    )
