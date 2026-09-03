"""Outcome 표본을 Hypothesis Agent 가 볼 증거로 바꾼다.

구 profile 경로는 88개 feature 를 실었는데 그중 71개가 `catalog.py` 에 없어서 계약이
실행할 수 없었고, 이름이 같은데 값이 다른 것이 4개 있었으며(`vol_flow` 는 전 표본
상수 0), 대조군 `near_miss` 의 97.2% 가 수익 구간 안에 있었고, `mechanism_failure` 는
같은 표본의 **미래 진단**으로 골라 놓고 그 축의 차이를 증거로 읽고 있었다.

그래서 고치지 않고 다시 만든다. 이 모듈이 지키는 것은 셋이다.

  1. **실행 어휘 하나** — 값은 전부 `catalog.compute_features` 에서 온다. 여기에
     feature 계산식을 새로 쓰지 않는다. 카탈로그에 없는 이름이 나오면 build 를 실패시킨다.
  2. **오염되지 않은 대조** — PROFIT / 진짜 LOSS / 시간대까지 맞춘 BACKGROUND.
     미래로 고른 대조군은 쓰지 않는다.
  3. **인과적 시간 증거** — anchor 이전 시각 격자에서 값을 읽는다. 미래 보간은 없고,
     읽은 값이 얼마나 오래된 것인지(`age_ms`)를 함께 남긴다.

경계는 이렇다.

    Profiler       "시장에서 무엇이 관측됐는가"
    Hypothesis     "왜 그랬을 수 있는가"

이 모듈은 경제적 해석을 쓰지 않는다. "매도세가 소진됐다" 는 문장은 여기서 나오지 않는다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import catalog, data as tickdata, fill, profit
from .config import (ENTRY_ORDER_MAX_SECONDS, ENTRY_ORDER_MAX_TICKS,
                     FEATURE_PROFILE_ENTRY_QUEUE_FRACTION, FEE_BPS, TICK_ROOT,
                     now_utc, write_json)


PROFIT, LOSS, BACKGROUND = "PROFIT", "LOSS", "BACKGROUND"
COHORTS = (PROFIT, LOSS, BACKGROUND)
PAIRS = ((PROFIT, LOSS), (PROFIT, BACKGROUND), (LOSS, BACKGROUND))


@dataclass(frozen=True)
class ProfilerConfig:
    """결정값을 전부 여기 둔다. 코드 안에 숨은 상수를 두지 않는다."""

    horizon_key: str = "S30"
    horizon_seconds: float = 30.0
    fee_bps: float = FEE_BPS
    entry_queue_fraction: float = FEATURE_PROFILE_ENTRY_QUEUE_FRACTION

    # 시간 격자. anchor 이전만 본다 — 0 보다 큰 값을 넣으면 build 가 거부한다.
    relative_times_ms: tuple[int, ...] = (-10_000, -5_000, -2_000, -1_000, 0)
    # 읽은 값이 이보다 오래됐으면 NaN. 호가가 드문 종목에서 옛날 값을 현재로 쓰지 않는다.
    freshness_limit_ms: int = 2_000

    # LOSS: 창 안 최고 ASK1 가격 기회도 net 이 이 값 이하. PROFIT의 +10bps와 같은 양의
    # 반대 부호다. 어느 시점에도 가격 기회가 없으므로 지속성 조건이 저절로 성립한다.
    loss_net_ceiling_bps: float = -10.0
    profit_net_floor_bps: float = 10.0     # PROFIT 표본이 만족해야 하는 값. 점검용
    max_anchors_per_partition: int = 3     # production 의 time-spread cap 과 같다

    background_bucket_minutes: int = 15

    correlation_threshold: float = 0.90
    coverage_floor: float = 0.50
    missingness_gap_warning: float = 0.15
    weighting_shift_warning_sd: float = 0.25
    min_group_anchors: int = 3             # 종목/날짜 방향을 셀 최소 표본


# ---- feature universe (§7) ----------------------------------------------------

# family 는 카탈로그 tag·dimension 에서 유도한다. feature 이름 목록을 여기 적기
# 시작하면 카탈로그가 정본이 아니게 된다.

def family_of(spec: catalog.Feature) -> str:
    """카탈로그 tag 와 dimension 에서 evidence family 를 유도한다."""
    tags, dim = set(spec.tags), spec.dimension
    if "trade_flow" in tags:
        return "TRADE_ACTIVITY"
    if "flow" in tags:
        return "ORDER_FLOW"
    if "microprice" in tags:
        return "MICROPRICE"
    if dim == "price_per_quantity":
        return "BOOK_SHAPE"
    if dim == "count":
        return "QUEUE_STATE"
    if dim == "return" and "liquidity" in tags:
        return "SPREAD"
    if dim == "return":
        return "PRICE_MOMENTUM"
    if dim == "quantity":
        return "QUEUE_STATE"
    return "BOOK_IMBALANCE"


def evidence_features(capabilities: catalog.DataCapabilities,
                      policy: catalog.WorkflowPolicy | None = None) -> tuple[str, ...]:
    """Agent 가 볼 수 있는 feature. 카탈로그와 정책과 데이터 가용성의 교집합이다.

    제외되는 것 — 미래를 보는 feature, 실행 전용(호가 수준), 이 데이터셋에서
    계산 불가능한 feature.
    """
    policy = policy or catalog.default_policy()
    visible = policy.grounding_visible - policy.execution_only
    available = set(catalog.available_features(capabilities))
    return tuple(sorted(n for n in visible
                        if n in available and catalog.FEATURES[n].causal))


# ---- 창 안 최고 ASK1 가격 기회 -------------------------------------------------

def _range_max(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """`[start, end]` 각 구간의 최대. sparse table 로 O(n log n)."""
    n = len(values)
    levels = [np.asarray(values, dtype=float)]
    size = 1
    while size * 2 <= n:
        prev = levels[-1]
        levels.append(np.maximum(prev[: len(prev) - size], prev[size:]))
        size *= 2
    length = np.maximum(ends - starts + 1, 1)
    k = np.floor(np.log2(length)).astype(int)
    k = np.clip(k, 0, len(levels) - 1)
    out = np.full(len(starts), np.nan)
    for level in np.unique(k):
        mask = k == level
        arr = levels[level]
        span = 1 << level
        left = np.clip(starts[mask], 0, len(arr) - 1)
        right = np.clip(ends[mask] - span + 1, 0, len(arr) - 1)
        out[mask] = np.maximum(arr[left], arr[right])
    return out


def window_end_ticks(time_s: np.ndarray, seconds: float) -> np.ndarray:
    """각 틱의 평가 창 끝. 창이 데이터 끝을 넘으면 -1 (관측 불가)."""
    end = np.searchsorted(time_s, time_s + float(seconds), side="left")
    return np.where(end >= len(time_s), -1, end)


def best_ask_net_bps(arrays: Mapping[str, np.ndarray], config: ProfilerConfig) -> np.ndarray:
    """각 틱의 BID1 진입 뒤 창 안 최고 ASK1 가격 기회의 net(bps). 창이 잘리면 NaN.

    이 값은 미래 oracle이며 체결 증명이 아니다. PROFIT과 LOSS는 같은 양 위에서만
    나눈다. 실제 지정가 체결은 backtest에서 따로 판정한다.
    """
    bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    time_s = np.asarray(arrays["time_s"], dtype=float)
    ends = window_end_ticks(time_s, config.horizon_seconds)
    n = len(bid1)
    out = np.full(n, np.nan)
    ok = (ends >= 0) & np.isfinite(bid1) & (bid1 > 0)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    peak = _range_max(np.where(np.isfinite(ask1) & (ask1 > 0), ask1, -np.inf),
                      idx + 1, ends[idx])
    with np.errstate(invalid="ignore"):
        net = (peak / bid1[idx] - 1.0) * 10_000.0 - config.fee_bps
    out[idx] = np.where(np.isfinite(peak), net, np.nan)
    return out


def _representatives(anchors: np.ndarray, window_end: np.ndarray, cap: int) -> list[int]:
    """창이 겹치는 덩어리마다 가운데 하나. 그다음 시간에 고르게 `cap` 개.

    production 의 `positive_event_rule` 과 같은 규칙이다 — 한 움직임을 여러 번 세지
    않으려는 것이지 표본을 늘리려는 것이 아니다.
    """
    if not len(anchors):
        return []
    blocks: list[list[int]] = [[int(anchors[0])]]
    reach = int(window_end[anchors[0]])
    for a in anchors[1:]:
        if int(a) <= reach:
            blocks[-1].append(int(a))
        else:
            blocks.append([int(a)])
        reach = max(reach, int(window_end[a]))
    picked = [block[len(block) // 2] for block in blocks]
    if len(picked) <= cap:
        return picked
    step = np.linspace(0, len(picked) - 1, cap).round().astype(int)
    return [picked[i] for i in dict.fromkeys(step)]


# ---- Feature Profile 보조 진입 정보 -------------------------------------------

ENTRY_QUEUE_COLUMNS = (
    "entry_queue_status", "entry_terminal_tick", "entry_fill_tick",
    "entry_wait_ticks", "entry_wait_seconds", "entry_queue_fraction",
    "entry_oracle_ask_tick", "entry_fill_before_oracle_ask",
)


def attach_entry_queue(anchors: pd.DataFrame, *, config: ProfilerConfig,
                       root: Path = TICK_ROOT) -> pd.DataFrame:
    """기존 anchor에만 BID1 20% 앞 큐 재생 결과를 붙인다.

    PROFIT/LOSS/BACKGROUND 선택이나 수익구간 산식은 전혀 바꾸지 않는다. `tick`에
    BID1 매수 주문을 냈다고 놓고, 실제 체결 시점과 기존 oracle ASK1 최고가보다
    먼저 체결됐는지만 Feature Profile의 보조 열로 남긴다. 체결 이벤트가 없는 데이터는
    기존 profile을 유지하고 `entry_queue_status=not_supported`로 남긴다.
    """
    if anchors.empty:
        return anchors.copy()
    if not 0.0 <= float(config.entry_queue_fraction) <= 1.0:
        raise ValueError("entry_queue_fraction 은 [0, 1]")
    out = anchors.copy()
    out["entry_queue_status"] = "not_supported"
    out["entry_terminal_tick"] = -1
    out["entry_fill_tick"] = -1
    out["entry_wait_ticks"] = -1
    out["entry_wait_seconds"] = np.nan
    out["entry_queue_fraction"] = float(config.entry_queue_fraction)
    out["entry_oracle_ask_tick"] = -1
    out["entry_fill_before_oracle_ask"] = False

    for (symbol, date), group in out.groupby(["symbol", "date"], sort=True):
        arrays, _ = tickdata.load(symbol, date, root)
        ticks = group.tick.to_numpy(dtype=np.int64)
        try:
            status, terminal, filled = fill.entry_fills(
                arrays, ticks, qfrac=float(config.entry_queue_fraction),
                max_ticks=ENTRY_ORDER_MAX_TICKS, max_seconds=ENTRY_ORDER_MAX_SECONDS)
        except ValueError as error:
            if "체결 이벤트" not in str(error):
                raise
            status = np.full(len(ticks), "not_supported", dtype=object)
            terminal = np.full(len(ticks), -1, dtype=np.int64)
            filled = np.full(len(ticks), -1, dtype=np.int64)
        time_s = np.asarray(arrays["time_s"], dtype=float)
        ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
        ends = window_end_ticks(time_s, config.horizon_seconds)
        oracle = np.full(len(ticks), -1, dtype=np.int64)
        for pos, (tick, end) in enumerate(zip(ticks, ends[ticks])):
            if end > tick:
                path = ask1[tick + 1:end + 1]
                if np.isfinite(path).any():
                    oracle[pos] = int(tick + 1 + np.nanargmax(path))

        idx = group.index.to_numpy()
        filled = np.asarray(filled, dtype=np.int64)
        out.loc[idx, "entry_queue_status"] = status.astype(str)
        out.loc[idx, "entry_terminal_tick"] = np.asarray(terminal, dtype=np.int64)
        out.loc[idx, "entry_fill_tick"] = filled
        out.loc[idx, "entry_wait_ticks"] = np.where(filled >= 0, filled - ticks, -1)
        wait = np.full(len(ticks), np.nan)
        valid = filled >= 0
        if valid.any():
            wait[valid] = time_s[filled[valid]] - time_s[ticks[valid]]
        out.loc[idx, "entry_wait_seconds"] = wait
        out.loc[idx, "entry_oracle_ask_tick"] = oracle
        out.loc[idx, "entry_fill_before_oracle_ask"] = (filled >= 0) & (filled < oracle)
    return out


# ---- cohort 구성 --------------------------------------------------------------

def build_anchors(config: ProfilerConfig = ProfilerConfig(),
                  clusters: Sequence[str] | None = None,
                  symbols: Sequence[str] | None = None,
                  dates: Sequence[str] | None = None,
                  root: Path = TICK_ROOT,
                  profit_root: Path = profit.PROFIT_CACHE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """PROFIT / LOSS / BACKGROUND anchor 를 만든다.

    PROFIT은 BID1 진입·ASK1 청산 기준 품질 수익구간에서 고른다. LOSS와 BACKGROUND는
    그 구간이 덮는 종목-일 안에서만 만든다 — 종목·날짜 구성이 달라져서
    생기는 차이를 메커니즘으로 착각하지 않으려는 것이다.

    품질 수익구간도 미래 oracle이므로 가설 발굴에만 쓴다. 실행 성과는 여기서 만들지 않는다.
    """
    keys = list(clusters or profit.quality_clusters(profit_root))
    frames = []
    for cluster in keys:
        frame = profit.quality_regions(cluster, horizon=config.horizon_key, root=profit_root)
        if symbols is not None:
            frame = frame[frame["symbol"].isin({str(value).zfill(6) for value in symbols})]
        if dates is not None:
            frame = frame[frame["date"].isin({str(value) for value in dates})]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("선택한 범위에 BID1 진입·ASK1 청산 품질 수익구간이 없다")
    seed = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    notes = {"partitions": 0, "skipped": [], "profit_floor_violations": 0}
    bucket = int(config.background_bucket_minutes) * 60

    for (symbol, date), group in seed.groupby(["symbol", "date"], sort=True):
        try:
            arrays, _ = tickdata.load(symbol, date, root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
            notes["skipped"].append({"symbol": symbol, "date": date, "reason": str(error)})
            continue
        notes["partitions"] += 1
        time_s = np.asarray(arrays["time_s"], dtype=float)
        ends = window_end_ticks(time_s, config.horizon_seconds)
        best = best_ask_net_bps(arrays, config)
        n = len(time_s)
        cluster = str(group.cluster_id.iloc[0])

        profit_ticks = _representatives(
            np.asarray(sorted({int(t) for t in group.start_tick if 0 <= int(t) < n}), dtype=int),
            ends, config.max_anchors_per_partition)
        for t in profit_ticks:
            # 저장소의 품질 수익구간이 이 양 위에서도 수익 조건을 만족하는지 확인한다.
            if not (np.isfinite(best[t]) and best[t] >= config.profit_net_floor_bps):
                notes["profit_floor_violations"] += 1
            rows.append({"cohort": PROFIT, "cluster_id": cluster, "symbol": symbol,
                         "date": date, "tick": t, "best_ask_net_bps": float(best[t])})

        # LOSS — 같은 eligibility·진입가·청산가·평가 창·비용. 미래 결과만 반대다.
        loss_mask = np.isfinite(best) & (best <= config.loss_net_ceiling_bps)
        loss = _representatives(np.flatnonzero(loss_mask), ends,
                                config.max_anchors_per_partition)
        for t in loss:
            rows.append({"cohort": LOSS, "cluster_id": cluster, "symbol": symbol,
                         "date": date, "tick": t, "best_ask_net_bps": float(best[t])})

        # BACKGROUND — 같은 종목·날·시간 버킷의 일반 상태. 뽑힌 anchor 의 창은 뺀다.
        taken = np.zeros(n, dtype=bool)
        for t in profit_ticks + loss:
            taken[t: max(int(ends[t]), t) + 1] = True
        eligible = (ends >= 0) & ~taken
        want: dict[int, int] = defaultdict(int)
        for t in profit_ticks + loss:
            want[int(time_s[t] // bucket)] += 1
        buckets = (time_s // bucket).astype(int)
        for key, count in sorted(want.items()):
            pool = np.flatnonzero(eligible & (buckets == key))
            if not len(pool):
                continue
            step = np.linspace(0, len(pool) - 1, min(count, len(pool))).round().astype(int)
            for i in dict.fromkeys(step):
                t = int(pool[i])
                rows.append({"cohort": BACKGROUND, "cluster_id": cluster, "symbol": symbol,
                             "date": date, "tick": t, "best_ask_net_bps": float(best[t])})

    anchors = pd.DataFrame(rows)
    anchors = attach_entry_queue(anchors, config=config, root=root)
    anchors["anchor_id"] = (anchors.symbol + ":" + anchors.date + ":"
                            + anchors.tick.astype(str) + ":" + anchors.cohort)
    return anchors, notes


# ---- 시각 정렬 (§16~§19) ------------------------------------------------------

def align_indices(time_s: np.ndarray, ticks: np.ndarray, offset_ms: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """`tick + offset` 시각 **이하**의 가장 최근 관측 인덱스와 그 나이(ms).

    미래 보간은 하지 않는다. 목표 시각보다 뒤의 관측은 쓰지 않는다.
    """
    target = time_s[ticks] + offset_ms / 1000.0
    idx = np.searchsorted(time_s, target, side="right") - 1
    age_ms = (target - time_s[np.clip(idx, 0, len(time_s) - 1)]) * 1000.0
    return idx, np.where(idx >= 0, age_ms, np.nan)


def profile_anchors(anchors: pd.DataFrame, config: ProfilerConfig = ProfilerConfig(),
                    features: Sequence[str] | None = None,
                    root: Path = TICK_ROOT) -> pd.DataFrame:
    """anchor 마다, 시각 격자마다, 카탈로그 feature 값을 읽는다.

    값은 전부 `catalog.compute_features` 에서 온다. 여기서 다시 계산하지 않는다.
    rolling feature 를 다시 평균하지도 않는다 — 그 시점의 값을 그대로 읽는다.
    """
    out: list[pd.DataFrame] = []
    for (symbol, date), group in anchors.groupby(["symbol", "date"], sort=True):
        arrays, _ = tickdata.load(symbol, date, root)
        book = catalog.Book(arrays)
        names = tuple(features) if features else evidence_features(book.capabilities)
        values = catalog.compute_features(names, book)
        missing = [n for n in names if n not in values]
        if missing:
            raise ValueError(f"{symbol}/{date}: 계산되지 않은 feature {missing}")
        time_s = np.asarray(arrays["time_s"], dtype=float)
        ticks = group.tick.to_numpy(int)
        for offset in config.relative_times_ms:
            idx, age = align_indices(time_s, ticks, offset)
            fresh = (idx >= 0) & (age <= config.freshness_limit_ms)
            safe = np.clip(idx, 0, len(time_s) - 1)
            block = pd.DataFrame({
                "anchor_id": group.anchor_id.to_numpy(), "cohort": group.cohort.to_numpy(),
                "cluster_id": group.cluster_id.to_numpy(), "symbol": symbol, "date": date,
                "tick": ticks, "relative_time_ms": int(offset),
                "source_tick": np.where(fresh, safe, -1),
                "age_ms": np.where(fresh, age, np.nan)})
            for name in names:
                block[name] = np.where(fresh, values[name][safe], np.nan)
            out.append(block)
    frame = pd.concat(out, ignore_index=True)
    return frame.sort_values(["anchor_id", "relative_time_ms"]).reset_index(drop=True)


# ---- 통계 --------------------------------------------------------------------

def _stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    good = x[finite]
    base = {"n": int(len(x)), "finite_n": int(finite.sum()),
            "coverage": float(finite.mean()) if len(x) else 0.0}
    if not len(good):
        return {**base, **{k: np.nan for k in ("mean", "median", "p25", "p75", "p1", "p99")}}
    return {**base, "mean": float(good.mean()), "median": float(np.percentile(good, 50)),
            "p25": float(np.percentile(good, 25)), "p75": float(np.percentile(good, 75)),
            "p1": float(np.percentile(good, 1)), "p99": float(np.percentile(good, 99))}


def canonical_profile(long: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """cohort × feature × relative_time 통계. NaN 을 0 으로 채우지 않는다."""
    rows = []
    for (cohort, rel), group in long.groupby(["cohort", "relative_time_ms"], sort=True):
        for name in features:
            spec = catalog.FEATURES[name]
            rows.append({"feature_name": name, "definition_id": spec.definition_id,
                         "catalog_hash": catalog.catalog_hash(), "dimension": spec.dimension,
                         "unit": spec.unit, "time_basis": spec.time_basis,
                         "family": family_of(spec), "cohort": cohort,
                         "relative_time_ms": int(rel),
                         "median_age_ms": float(np.nanmedian(group.age_ms))
                         if group.age_ms.notna().any() else np.nan,
                         **_stats(group[name].to_numpy(float))})
    return pd.DataFrame(rows)


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """a 의 값이 b 보다 클 확률. 0.5 면 구분하지 못한다."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    order = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return float((order[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def _agreement(long: pd.DataFrame, name: str, rel: int, a: str, b: str,
               axis: str, minimum: int) -> tuple[int, int, float]:
    """축(날짜·종목)마다 방향이 전체와 같은가. 소수 종목이 전체를 끌고 가는지 본다."""
    part = long[(long.relative_time_ms == rel) & long.cohort.isin([a, b])]
    med = part.groupby([axis, "cohort"])[name].median().unstack("cohort")
    cnt = part.groupby([axis, "cohort"])[name].count().unstack("cohort")
    if a not in med or b not in med:
        return 0, 0, np.nan
    ok = (cnt.get(a, 0) >= minimum) & (cnt.get(b, 0) >= minimum)
    diff = (med[a] - med[b])[ok].dropna()
    if not len(diff):
        return 0, 0, np.nan
    overall = np.sign(diff.median())
    agree = int((np.sign(diff) == overall).sum()) if overall else 0
    return agree, int(len(diff)), agree / len(diff)


def _direction(auc: float, cohort: str) -> str:
    """AUC 를 그대로 주면 0.23 을 "성능이 낮다" 로 읽는다. 방향과 세기를 나눠 적는다."""
    if not np.isfinite(auc):
        return "UNDETERMINED"
    if auc > 0.5:
        return f"higher_in_{cohort}"
    if auc < 0.5:
        return f"lower_in_{cohort}"
    return "no_separation"


def contrastive_profile(long: pd.DataFrame, features: Sequence[str],
                        config: ProfilerConfig = ProfilerConfig()) -> pd.DataFrame:
    """cohort 쌍마다 방향과 분리 크기. 유의성 선언이 아니라 방향과 크기를 적는다."""
    rows = []
    by_cohort = {c: long[long.cohort == c] for c in COHORTS}
    for a, b in PAIRS:
        for rel in sorted(long.relative_time_ms.unique()):
            fa = by_cohort[a][by_cohort[a].relative_time_ms == rel]
            fb = by_cohort[b][by_cohort[b].relative_time_ms == rel]
            for name in features:
                x, y = fa[name].to_numpy(float), fb[name].to_numpy(float)
                auc = _auc(x, y)
                da, dt, dr = _agreement(long, name, int(rel), a, b, "date",
                                        config.min_group_anchors)
                sa, st, sr = _agreement(long, name, int(rel), a, b, "symbol",
                                        config.min_group_anchors)
                ca = float(np.isfinite(x).mean()) if len(x) else 0.0
                cb = float(np.isfinite(y).mean()) if len(y) else 0.0
                rows.append({
                    "feature_name": name, "definition_id": catalog.FEATURES[name].definition_id,
                    "family": family_of(catalog.FEATURES[name]),
                    "relative_time_ms": int(rel), "cohort_a": a, "cohort_b": b,
                    "median_diff": float(np.nanmedian(x) - np.nanmedian(y))
                    if np.isfinite(x).any() and np.isfinite(y).any() else np.nan,
                    "auc": auc,
                    "direction": _direction(auc, a),
                    "effect_strength": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
                    "date_agreement_count": da, "date_total": dt, "date_agreement_ratio": dr,
                    "symbol_agreement_count": sa, "symbol_total": st,
                    "symbol_agreement_ratio": sr,
                    "coverage_a": ca, "coverage_b": cb,
                    "coverage_gap": abs(ca - cb),
                    "missingness_confound": abs(ca - cb) > config.missingness_gap_warning,
                })
    return pd.DataFrame(rows)


def balanced_effects(long: pd.DataFrame, features: Sequence[str], rel: int,
                     a: str = PROFIT, b: str = LOSS) -> pd.DataFrame:
    """anchor 단순 집계와 종목-일 균형 집계를 비교한다 (§29·§30).

    한 가격 움직임에서 anchor 가 여러 개 나오면 그 하나가 평균을 끌 수 있다.
    두 값이 크게 다르면 그 feature 의 대조는 표본 편중일 수 있다.
    """
    part = long[(long.relative_time_ms == rel) & long.cohort.isin([a, b])]
    rows = []
    for name in features:
        pooled = part.groupby("cohort")[name].median()
        cell = part.groupby(["symbol", "date", "cohort"])[name].median().unstack("cohort")
        scale = part[name].std()
        anchor_effect = pooled.get(a, np.nan) - pooled.get(b, np.nan)
        balanced = (cell[a].median() - cell[b].median()) if {a, b} <= set(cell.columns) else np.nan
        rows.append({"feature_name": name, "relative_time_ms": int(rel),
                     "anchor_weighted_diff": float(anchor_effect),
                     "symbol_day_balanced_diff": float(balanced),
                     "shift_sd": float((balanced - anchor_effect) / scale)
                     if scale and np.isfinite(scale) and scale > 0 else np.nan})
    return pd.DataFrame(rows)


# ---- 중복 묶기 (§31~§35) ------------------------------------------------------

def evidence_groups(long: pd.DataFrame, features: Sequence[str],
                    config: ProfilerConfig = ProfilerConfig(),
                    cohort: str = PROFIT, rel: int = 0) -> pd.DataFrame:
    """상관이 높은 feature 를 한 묶음으로. 같은 신호를 여러 증거로 세지 않으려는 것이다.

    묶음의 대표는 **효과가 큰 것으로 고르지 않는다.** 파생이 아닌 것, 관측이 많은 것,
    lookback 이 짧은 것 순이다 — 성과로 고르면 대표 선택 자체가 선택 편향이 된다.
    """
    part = long[(long.cohort == cohort) & (long.relative_time_ms == rel)]
    usable = [f for f in features if part[f].notna().sum() >= 5 and part[f].nunique() > 1]
    corr = part[usable].corr(method="spearman") if usable else pd.DataFrame()

    parent = {f: f for f in features}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, x in enumerate(usable):
        for y in usable[i + 1:]:
            rho = corr.loc[x, y]
            same_family = family_of(catalog.FEATURES[x]) == family_of(catalog.FEATURES[y])
            if np.isfinite(rho) and abs(rho) >= config.correlation_threshold and same_family:
                parent[find(x)] = find(y)

    members: dict[str, list[str]] = defaultdict(list)
    for f in features:
        members[find(f)].append(f)

    coverage = {f: float(part[f].notna().mean()) if len(part) else 0.0 for f in features}
    rows = []
    seen: dict[str, int] = defaultdict(int)
    for _, group in sorted(members.items(), key=lambda kv: sorted(kv[1])):
        family = family_of(catalog.FEATURES[sorted(group)[0]])
        seen[family] += 1
        group_id = f"{family}_{seen[family]:02d}"
        rep = sorted(group, key=lambda f: (
            "derived" in catalog.FEATURES[f].tags,      # 1. 파생이 아닌 것 우선
            -coverage[f],                               # 2. 관측이 많은 것
            float(catalog.FEATURES[f].lookback),        # 3. lineage 가 단순한 것
            f))                                         # 4. 결정적 tie-break
        for f in sorted(group):
            rows.append({"feature": f, "definition_id": catalog.FEATURES[f].definition_id,
                         "family": family_of(catalog.FEATURES[f]),
                         "evidence_id": f"EV_{group_id}", "correlation_group": group_id,
                         "representative": rep[0],
                         "is_representative": f == rep[0],
                         "related_features": ",".join(sorted(set(group) - {f})),
                         "coverage": coverage[f],
                         "max_abs_rho_in_group": float(max(
                             (abs(corr.loc[f, g]) for g in group
                              if g != f and f in corr.index and g in corr.columns),
                             default=np.nan))})
    return pd.DataFrame(rows)


# ---- 시간 패턴 (§36~§38) ------------------------------------------------------

# ---- 증거의 역할 (§3~§6) --------------------------------------------------------

# 분리 크기의 기준. Profiler 의 usable 기준(effect_strength >= 0.10)을 separation
# 단위로 옮긴 값 하나뿐이다 — separation = 2 x (AUC - 0.5). 새로 고르지 않는다.
MATERIAL_SEPARATION = 0.20

ROLES = ("PERSISTENT_CONTEXT", "SETUP_STATE", "LATE_TRIGGER", "SUPPORTING_STATE",
         "REDUNDANT", "UNRESOLVED")

# family 를 가로지르는 **대수적 종속**. 상관 묶기는 같은 family 안에서만 병합하므로
# 이런 항등식은 두 개의 독립 증거로 살아남는다. 실측으로 확인해 여기 적는다.
# (`test_evidence.py` 가 항등식을 데이터로 다시 검증한다.)
DERIVED_FAMILIES: dict[str, dict[str, Any]] = {
    "MICROPRICE": {
        "identity": "microprice_dev_bps == spread_bps / 2 * book_imbalance",
        "derived_from": ("SPREAD", "BOOK_IMBALANCE"),
        "dependency_type": "deterministic"},
}


def separation_onset(separation: Sequence[float | None], times: Sequence[int]) -> int | None:
    """분리가 기준을 처음 넘는 시각. 없으면 None."""
    for value, rel in zip(separation, times):
        if value is not None and np.isfinite(value) and abs(value) >= MATERIAL_SEPARATION:
            return int(rel)
    return None


def temporal_role(separation: Sequence[float | None], family: str = "") -> tuple[str, str, str]:
    """분리 궤적의 모양에서 역할을 정한다. 결정적이다 — 눈대중이 끼지 않는다.

    역할은 feature 의 영구 속성이 아니라 (feature, outcome 정의, run) 에 종속된다.
    다른 평가 창에서는 같은 feature 가 다른 역할을 가질 수 있다.
    """
    if family in DERIVED_FAMILIES:
        return "REDUNDANT", "HIGH", DERIVED_FAMILIES[family]["identity"]
    e = np.array([np.nan if v is None else float(v) for v in separation], dtype=float)
    if len(e) < 3 or not np.isfinite(e).any():
        return "UNRESOLVED", "LOW", "분리 궤적을 읽을 수 없다"
    early, mid, anchor = np.nanmean(e[:2]), np.nanmean(e[2:-1]), e[-1]
    shift = anchor - mid
    ref = np.sign(anchor) if np.isfinite(anchor) and anchor else 0.0
    consistent = bool(ref) and float(np.mean(np.sign(e[np.isfinite(e)]) == ref)) >= 0.8
    if max(abs(early), abs(mid), abs(anchor)) < MATERIAL_SEPARATION:
        return "UNRESOLVED", "LOW", "어느 시점에서도 분리가 기준에 못 미친다"
    if abs(early) >= MATERIAL_SEPARATION and abs(shift) < MATERIAL_SEPARATION and consistent:
        return ("PERSISTENT_CONTEXT", "HIGH",
                f"이른 구간부터 분리 {early:+.2f}, anchor 까지 변화 {shift:+.2f}")
    if abs(early) < MATERIAL_SEPARATION and abs(anchor) >= MATERIAL_SEPARATION \
            and abs(shift) >= MATERIAL_SEPARATION:
        return ("LATE_TRIGGER", "HIGH",
                f"이른 구간 {early:+.2f} -> anchor {anchor:+.2f} (변화 {shift:+.2f})")
    if abs(early) < MATERIAL_SEPARATION and abs(mid) >= MATERIAL_SEPARATION:
        return "SETUP_STATE", "MEDIUM", f"anchor 직전에 형성 (중간 {mid:+.2f})"
    if abs(anchor) >= MATERIAL_SEPARATION:
        return "SUPPORTING_STATE", "MEDIUM", "분리는 있으나 뚜렷한 시작점이 없다"
    return "UNRESOLVED", "LOW", "시간 모양이 어느 역할과도 맞지 않는다"


def evidence_relation_map(families: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """시간 역할에서 실제로 말할 수 있는 family 관계만 만든다.

    두 feature가 함께 보였다는 것만으로 결합·인과를 만들지 않는다. 이른 상태와 늦은
    trigger가 명확한 경우만 CONTEXT_TRIGGER로 적고, 나머지는 UNRESOLVED로 남긴다.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in families:
        grouped.setdefault(str(item["family"]), []).append(item)

    roles: dict[str, str] = {}
    details: dict[str, str] = {}
    for family, items in grouped.items():
        seen = {str(item.get("role")) for item in items}
        if len(seen) == 1:
            roles[family] = next(iter(seen))
            details[family] = str(items[0].get("role_reason", ""))
        else:
            roles[family] = "UNRESOLVED"
            details[family] = "같은 family 안의 대표 증거 역할이 하나로 정해지지 않는다"

    rows: list[dict[str, Any]] = []
    for index, family_a in enumerate(sorted(grouped)):
        for family_b in sorted(grouped)[index + 1:]:
            role_a, role_b = roles[family_a], roles[family_b]
            pair = {role_a, role_b}
            if "REDUNDANT" in pair:
                relation, ordering, reason = (
                    "REDUNDANT", "NOT_APPLICABLE",
                    "둘 중 하나가 파생 feature family라 독립 근거가 아니다")
            elif pair == {"PERSISTENT_CONTEXT", "LATE_TRIGGER"}:
                relation, ordering, reason = (
                    "CONTEXT_TRIGGER", "CONTEXT_EARLIER_THAN_TRIGGER",
                    "한 family는 관측 창의 이른 시점부터 분리되고, 다른 family는 anchor 근처에서만 분리된다")
            else:
                relation, ordering, reason = (
                    "UNRESOLVED", "NOT_ESTABLISHED",
                    "시간 역할만으로 두 family의 결합·독립·인과 관계를 정할 수 없다")
            rows.append({
                "joint_evidence_id": f"JR_{family_a}_{family_b}",
                "family_a": family_a, "family_b": family_b,
                "role_a": role_a, "role_b": role_b,
                "relation": relation, "relation_reason": reason,
                "relation_support": {"family_a": details[family_a], "family_b": details[family_b]},
                "ordering_status": ordering,
                "classification": "DETERMINISTIC_TEMPORAL_MAP",
                "summary": f"{family_a}/{family_b}: {relation}",
            })
    return roles, rows


def path_observation_anchors(anchors: pd.DataFrame, horizon_key: str) -> dict[str, Any]:
    """Agent tool이 열 수 있는 Feature Profile anchor 목록. 미래 경로 임의 탐색은 막는다."""
    optional = (
        "entry_queue_status", "entry_fill_tick", "entry_wait_ticks", "entry_wait_seconds",
        "entry_queue_fraction", "entry_oracle_ask_tick", "entry_fill_before_oracle_ask",
    )
    rows = []
    for index, row in anchors.sort_values(["cluster_id", "symbol", "date", "tick"]).iterrows():
        item = {
            "anchor_id": str(row.get("anchor_id", f"anchor-{index}")),
            "cohort": str(row["cohort"]), "cluster_id": str(row["cluster_id"]),
            "symbol": str(row["symbol"]).zfill(6), "date": str(row["date"]),
            "tick": int(row["tick"]),
        }
        for name in optional:
            if name in anchors.columns:
                item[name] = _clean(row.get(name)) if name != "entry_queue_status" else (
                    None if pd.isna(row.get(name)) else str(row.get(name)))
                if name == "entry_fill_before_oracle_ask" and not pd.isna(row.get(name)):
                    item[name] = bool(row.get(name))
        rows.append(item)
    return {
        "schema": "discovery_price_path_access.v1", "tool_name": "discovery_price_path",
        "horizon_key": str(horizon_key), "anchors": rows,
        "policy": "Feature Profile에 이미 포함된 discovery anchor만 읽는다. 반환 미래 가격은 oracle discovery label이며 체결·검증·OOS 증거가 아니다.",
    }


PATTERNS = ("MONOTONIC_INCREASE", "MONOTONIC_DECREASE", "REVERSAL", "CROSSOVER",
            "LEVEL_SHIFT", "PERSISTENT_HIGH", "PERSISTENT_LOW",
            "NO_CLEAR_TEMPORAL_PATTERN")


def temporal_pattern(diff: Sequence[float], threshold: float = 0.15) -> str:
    """cohort 간 차이의 시간 모양을 결정적 규칙으로 이름 붙인다.

    `diff` 는 시각 순서대로의 (cohort A 중앙 − cohort B 중앙) 을 척도로 나눈 값이다.
    LLM 으로 이름을 붙이지 않는다 — 같은 입력에 늘 같은 이름이 나와야 한다.
    """
    d = np.asarray(diff, dtype=float)
    if not np.isfinite(d).all() or len(d) < 3:
        return "NO_CLEAR_TEMPORAL_PATTERN"
    if np.all(np.abs(d) < threshold):
        return "NO_CLEAR_TEMPORAL_PATTERN"
    steps = np.diff(d)
    # 임계를 넘은 값만 부호로 센다. 그러지 않으면 1e-5 짜리 흔들림이 CROSSOVER 가 된다.
    strong = np.where(np.abs(d) >= threshold, np.sign(d), 0.0)
    if np.any(strong > 0) and np.any(strong < 0):
        return "CROSSOVER"
    half = len(steps) // 2 or 1
    early, late = steps[:half].sum(), steps[half:].sum()
    if abs(d[-1] - d[0]) >= threshold and np.all(steps >= 0):
        return "MONOTONIC_INCREASE"
    if abs(d[-1] - d[0]) >= threshold and np.all(steps <= 0):
        return "MONOTONIC_DECREASE"
    if early * late < 0 and min(abs(early), abs(late)) >= threshold:
        return "REVERSAL"
    if abs(steps[-1]) >= threshold and abs(steps[-1]) >= 2 * abs(steps[:-1]).max(initial=0.0):
        return "LEVEL_SHIFT"
    if np.all(d >= threshold):
        return "PERSISTENT_HIGH"
    if np.all(d <= -threshold):
        return "PERSISTENT_LOW"
    return "NO_CLEAR_TEMPORAL_PATTERN"


def temporal_summary(long: pd.DataFrame, features: Sequence[str], contrast: pd.DataFrame,
                     config: ProfilerConfig = ProfilerConfig(),
                     a: str = PROFIT, b: str = LOSS) -> pd.DataFrame:
    """feature 마다 시간 궤적과 그 모양, 그리고 증거 준비 상태.

    궤적은 중앙값 차이가 아니라 **분리도** `2 × (AUC − 0.5)` 로 잰다. OFI 처럼 값의
    절반 이상이 0 인 feature 는 두 cohort 의 중앙값이 모두 0 이 되어 차이가 사라진다 —
    구 profile 진단에서 시간 모양이 전부 0 으로 보였던 것이 그 때문이었다.
    순위 기반으로 재면 그 문제가 없다.
    """
    times = sorted(int(t) for t in long.relative_time_ms.unique())
    pair = contrast[(contrast.cohort_a == a) & (contrast.cohort_b == b)]
    lookup = {(r.feature_name, int(r.relative_time_ms)): r.auc for r in pair.itertuples()}
    rows = []
    for name in features:
        medians_a, medians_b, separation = [], [], []
        for rel in times:
            part = long[long.relative_time_ms == rel]
            with np.errstate(invalid="ignore"):
                medians_a.append(_clean(np.nanmedian(part.loc[part.cohort == a, name])))
                medians_b.append(_clean(np.nanmedian(part.loc[part.cohort == b, name])))
            auc = lookup.get((name, rel), np.nan)
            separation.append(2.0 * (auc - 0.5) if np.isfinite(auc) else np.nan)
        rows.append({"feature_name": name, "cohort_a": a, "cohort_b": b,
                     "relative_times_ms": times,
                     "median_a": medians_a, "median_b": medians_b,
                     "separation": [_clean(v) for v in separation],
                     "temporal_pattern": temporal_pattern(separation)})
    return pd.DataFrame(rows)


def alignment_coverage(long: pd.DataFrame) -> pd.DataFrame:
    """cohort × 시각마다 **anchor 자체**가 정렬됐는가.

    feature 별 coverage 가 20개 모두 같은 값으로 떨어지면 그것은 feature 문제가 아니라
    그 시각에 읽을 호가가 없었다는 뜻이다. 원인을 feature 에 잘못 붙이지 않으려고
    anchor 수준을 따로 센다.
    """
    rows = []
    for (cohort, rel), group in long.groupby(["cohort", "relative_time_ms"], sort=True):
        aligned = group.source_tick >= 0
        rows.append({"cohort": cohort, "relative_time_ms": int(rel),
                     "anchors": int(len(group)),
                     "aligned": int(aligned.sum()),
                     "alignment_coverage": float(aligned.mean()),
                     "median_age_ms": float(np.nanmedian(group.age_ms))
                     if group.age_ms.notna().any() else np.nan,
                     "p95_age_ms": float(np.nanpercentile(group.age_ms, 95))
                     if group.age_ms.notna().any() else np.nan})
    return pd.DataFrame(rows)


# ---- 게이트 (§45·§46) ---------------------------------------------------------

class EvidenceGateError(RuntimeError):
    """증거를 Hypothesis Agent 로 보내기 전 검사에 걸렸다. 이유를 전부 담는다."""


def quality_gate(long: pd.DataFrame, features: Sequence[str], anchors: pd.DataFrame,
                 config: ProfilerConfig, catalog_hash: str) -> tuple[list[str], list[str]]:
    """실패 조건과 경고를 나눠 돌려준다. 조용히 넘어가지 않는다."""
    fail, warn = [], []

    unknown = [f for f in features if f not in catalog.FEATURES]
    if unknown:
        fail.append(f"카탈로그에 없는 feature: {unknown}")
    if catalog_hash != catalog.catalog_hash():
        fail.append(f"catalog hash 불일치: {catalog_hash} != {catalog.catalog_hash()}")
    non_causal = [f for f in features if f in catalog.FEATURES and not catalog.FEATURES[f].causal]
    if non_causal:
        fail.append(f"미래를 보는 feature: {non_causal}")
    if any(int(t) > 0 for t in config.relative_times_ms):
        fail.append(f"시간 격자에 미래가 있다: {config.relative_times_ms}")
    missing_cohorts = [c for c in COHORTS if not (anchors.cohort == c).any()]
    if missing_cohorts:
        fail.append(f"cohort 가 비었다: {missing_cohorts}")
    if not long.empty and (long.age_ms.dropna() < 0).any():
        fail.append("미래 관측을 읽었다 (age_ms < 0)")

    at0 = long[long.relative_time_ms == 0]
    for name in [f for f in features if f in long.columns]:
        for cohort in COHORTS:
            part = at0[at0.cohort == cohort]
            if len(part) and part[name].notna().mean() < config.coverage_floor:
                warn.append(f"낮은 coverage: {name}/{cohort} "
                            f"{part[name].notna().mean():.1%}")
    counts = anchors.cohort.value_counts()
    for cohort in COHORTS:
        share = counts.get(cohort, 0) / max(len(anchors), 1)
        top = (anchors[anchors.cohort == cohort].symbol.value_counts(normalize=True).iloc[0]
               if counts.get(cohort, 0) else 0.0)
        if top > 0.10:
            warn.append(f"종목 집중: {cohort} 상위 종목이 {top:.1%}")
    return fail, warn


# ---- 최종 패키지 --------------------------------------------------------------

def evidence_package(canonical: pd.DataFrame, contrast: pd.DataFrame, groups: pd.DataFrame,
                     temporal: pd.DataFrame, anchors: pd.DataFrame, balanced: pd.DataFrame,
                     config: ProfilerConfig, warnings: Sequence[str],
                     alignment: pd.DataFrame) -> dict[str, Any]:
    """Hypothesis Agent 에게 그대로 넘길 구조. 관측만 담고 해석은 담지 않는다."""
    at0 = contrast[(contrast.relative_time_ms == 0)
                   & (contrast.cohort_a == PROFIT) & (contrast.cohort_b == LOSS)]
    reps = groups[groups.is_representative]
    rank = at0.merge(reps[["feature", "correlation_group", "related_features"]],
                     left_on="feature_name", right_on="feature", how="inner")
    # 순위는 대조 세기·coverage·날짜/종목 일관성으로 만든다. 수익성은 쓰지 않는다.
    rank = rank.assign(order=(rank.effect_strength.fillna(0)
                              * rank.coverage_a.fillna(0)
                              * rank.date_agreement_ratio.fillna(0.5)
                              * rank.symbol_agreement_ratio.fillna(0.5))
                       ).sort_values("order", ascending=False)

    families = []
    for _, row in rank.iterrows():
        name = row.feature_name
        temporal_row = temporal[temporal.feature_name == name].iloc[0]
        role, role_confidence, role_reason = temporal_role(temporal_row.separation, row.family)
        cover = canonical[(canonical.feature_name == name)
                          & (canonical.relative_time_ms == 0)]
        bal = balanced[balanced.feature_name == name]
        spec = catalog.FEATURES[name]
        families.append({
            # 안정적인 참조 ID. 가설의 모든 주장은 이 ID 를 인용해야 추적이 된다.
            "evidence_id": f"EV_{row.correlation_group}",
            "evidence_group": row.correlation_group,
            "family": row.family,
            "representative_feature": {
                "name": name, "definition_id": spec.definition_id,
                "dimension": spec.dimension, "unit": spec.unit,
                "time_basis": spec.time_basis, "lookback": spec.lookback,
                "observable_description": spec.observable_description,
                "economic_interpretation": spec.economic_interpretation,
                "invalid_policy": spec.invalid_policy},
            "related_features": [f for f in str(row.related_features).split(",") if f],
            "snapshot_contrast": {
                "profit_vs_loss": {
                    "auc": _clean(row.auc), "direction": row.direction,
                    "effect_strength": _clean(row.effect_strength),
                    "median_diff": _clean(row.median_diff)},
                "profit_vs_background": _pair(contrast, name, PROFIT, BACKGROUND),
                "loss_vs_background": _pair(contrast, name, LOSS, BACKGROUND)},
            "role": role, "role_confidence": role_confidence, "role_reason": role_reason,
            "separation_onset_ms": separation_onset(temporal_row.separation,
                                                    temporal_row.relative_times_ms),
            # 대수적으로 다른 family 에서 유도되는 증거는 독립 근거로 세지 않는다.
            "independent_evidence": row.family not in DERIVED_FAMILIES,
            "derived_from": list(DERIVED_FAMILIES.get(row.family, {}).get("derived_from", ())),
            "dependency_type": DERIVED_FAMILIES.get(row.family, {}).get("dependency_type"),
            "deterministic_dependency": DERIVED_FAMILIES.get(row.family, {}).get("identity"),
            "temporal_profile": {
                "relative_times_ms": temporal_row.relative_times_ms,
                "median_profit": [_clean(v) for v in temporal_row.median_a],
                "median_loss": [_clean(v) for v in temporal_row.median_b],
                "separation_profit_vs_loss": temporal_row.separation,
                "pattern": temporal_row.temporal_pattern},
            "coverage": {r.cohort: _clean(r.coverage) for r in cover.itertuples()},
            "date_stability": {"agreement": int(row.date_agreement_count),
                               "total": int(row.date_total),
                               "ratio": _clean(row.date_agreement_ratio)},
            "symbol_stability": {"agreement": int(row.symbol_agreement_count),
                                 "total": int(row.symbol_total),
                                 "ratio": _clean(row.symbol_agreement_ratio)},
            "readiness": {
                "level_evidence_available": bool(np.isfinite(row.effect_strength)),
                "temporal_evidence_available":
                    temporal_row.temporal_pattern != "NO_CLEAR_TEMPORAL_PATTERN",
                "stable_across_dates": bool((row.date_agreement_ratio or 0) >= 0.8),
                "stable_across_symbols": bool((row.symbol_agreement_ratio or 0) >= 0.7)},
            "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={row.coverage_gap:.2f}"]
                         if row.missingness_confound else [])
            + ([f"WEIGHTING_SHIFT {float(bal.shift_sd.iloc[0]):.2f}sd"]
               if len(bal) and abs(float(bal.shift_sd.iloc[0] or 0)) > config.weighting_shift_warning_sd
               else []),
        })

    roles, joint = evidence_relation_map(families)
    counts = anchors.cohort.value_counts()
    return {
        "schema": "hypothesis_evidence_package.v1",
        "created_at": now_utc(),
        "catalog_sha256": catalog.catalog_hash(),
        "config": asdict(config),
        "outcome_definition": {
            "side": "LONG", "entry_price": "BID1 at anchor tick",
            "exit_price": "future best ASK1 within the future window",
            "horizon": f"{config.horizon_key} = {config.horizon_seconds:g}s clock time",
            "cost_bps": config.fee_bps,
            "profit_rule": "price opportunity net >= 10bps sustained >= 10 ticks",
            "loss_rule": f"future best ASK1 in the window still nets <= {config.loss_net_ceiling_bps:g}bps",
            "background_rule": f"same symbol/date/{config.background_bucket_minutes}min bucket, "
                               "outside any selected anchor window",
            "future_information": "outcome labels only; feature evidence uses tau <= t",
            "execution_proof": False},
        "cohort_summary": {
            c: {"anchors": int(counts.get(c, 0)),
                "unique_symbols": int(anchors[anchors.cohort == c].symbol.nunique()),
                "unique_dates": int(anchors[anchors.cohort == c].date.nunique()),
                "symbol_days": int(len(anchors[anchors.cohort == c]
                                       .groupby(["symbol", "date"])))}
            for c in COHORTS},
        "alignment_coverage": [
            {"cohort": r.cohort, "relative_time_ms": int(r.relative_time_ms),
             "anchors": int(r.anchors), "coverage": _clean(r.alignment_coverage),
             "median_age_ms": _clean(r.median_age_ms), "p95_age_ms": _clean(r.p95_age_ms)}
            for r in alignment.itertuples()],
        "independent_evidence_families": sorted(
            {f["family"] for f in families if f["independent_evidence"]}),
        "evidence_families": families,
        "evidence_roles": roles,
        "joint_evidence": joint,
        "joint_evidence_note": (
            "관계 지도는 Feature Profile의 관측 시간 역할만으로 만들었다. "
            "UNRESOLVED를 결합·독립·인과로 읽지 않는다."),
        "path_observation": path_observation_anchors(anchors, config.horizon_key),
        "global_warnings": list(warnings),
        "not_provided": [
            "future path features outside the explicitly requested discovery price paths",
            "validation/OOS results", "strategy candidate performance",
            "future-selected control cohorts (near_miss, mechanism_failure)",
            "economic mechanism interpretation"],
    }


def _clean(value: Any) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def _pair(contrast: pd.DataFrame, name: str, a: str, b: str) -> dict[str, Any]:
    row = contrast[(contrast.feature_name == name) & (contrast.relative_time_ms == 0)
                   & (contrast.cohort_a == a) & (contrast.cohort_b == b)]
    if not len(row):
        return {}
    row = row.iloc[0]
    return {"auc": _clean(row.auc), "direction": row.direction,
            "effect_strength": _clean(row.effect_strength),
            "median_diff": _clean(row.median_diff)}


def dropped_features(capabilities: catalog.DataCapabilities,
                     policy: catalog.WorkflowPolicy | None = None) -> pd.DataFrame:
    """증거에서 빠진 feature 와 그 이유. 조용히 사라지지 않게 한 줄씩 남긴다."""
    policy = policy or catalog.default_policy()
    kept = set(evidence_features(capabilities, policy))
    reasons = {a.feature: a for a in catalog.feature_availability(capabilities)}
    rows = []
    for name in sorted(catalog.FEATURES):
        if name in kept:
            continue
        spec = catalog.FEATURES[name]
        if not reasons[name].available:
            why = f"이 데이터셋에서 계산 불가: {reasons[name].reason}"
        elif name in policy.execution_only:
            why = "실행 전용(호가 수준) — 종목마다 수준이 달라 증거가 아니다"
        elif not spec.causal:
            why = "미래를 본다"
        else:
            why = "정책에서 비노출"
        rows.append({"feature": name, "definition_id": spec.definition_id, "reason": why})
    return pd.DataFrame(rows)


def shadow_comparison(contrast: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    """구 profile 의 상위 증거가 canonical 조건에서도 반복되는지 (§49).

    구 경로는 `profit vs near_miss` 였고 near_miss 의 97% 가 수익 구간 안이었다.
    새 경로는 `PROFIT vs LOSS` 다. 두 대조가 다르므로 값이 아니라 **방향과 순위**를 본다.
    """
    new = contrast[(contrast.relative_time_ms == 0) & (contrast.cohort_a == PROFIT)
                   & (contrast.cohort_b == LOSS)].copy()
    new["canonical_rank"] = new.effect_strength.rank(ascending=False)
    old = legacy[(legacy.relative_time == "entry") & (legacy.granularity == "overall")
                 & (legacy.contrast == "profit_vs_near_miss")].copy()
    old["legacy_strength"] = (old.auc_vs_profit - 0.5).abs()
    old["legacy_rank"] = old.legacy_strength.rank(ascending=False)
    old["canonical_name"] = old.feature.map(
        lambda f: catalog.resolve(f) if catalog.resolve(f) in catalog.FEATURES else None)
    old = old.dropna(subset=["canonical_name"])
    merged = new.merge(old[["canonical_name", "auc_vs_profit", "legacy_strength", "legacy_rank"]],
                       left_on="feature_name", right_on="canonical_name", how="left")
    merged["legacy_direction"] = np.where(
        merged.auc_vs_profit.isna(), "NOT_IN_LEGACY",
        np.where(merged.auc_vs_profit < 0.5, "lower_in_PROFIT", "higher_in_PROFIT"))
    merged["direction_repeats"] = merged.direction == merged.legacy_direction
    return merged[["feature_name", "family", "auc", "direction", "effect_strength",
                   "canonical_rank", "auc_vs_profit", "legacy_strength", "legacy_rank",
                   "legacy_direction", "direction_repeats"]].rename(
        columns={"auc": "canonical_auc", "direction": "canonical_direction",
                 "effect_strength": "canonical_strength", "auc_vs_profit": "legacy_auc"})


def validation_report(package: Mapping[str, Any], canonical: pd.DataFrame,
                      contrast: pd.DataFrame, groups: pd.DataFrame, temporal: pd.DataFrame,
                      anchors: pd.DataFrame, dropped: pd.DataFrame, alignment: pd.DataFrame,
                      long: pd.DataFrame, balanced: pd.DataFrame,
                      shadow: pd.DataFrame | None = None) -> str:
    """사람이 읽을 검증 보고서. 관측만 적는다 — 메커니즘 해석은 쓰지 않는다."""
    L: list[str] = ["# PROFILE_VALIDATION.md", "",
                    "Canonical Evidence Profiler v1 이 만든 증거의 검증 기록.",
                    "관측 사실만 적는다 — \"왜 그런가\" 는 Hypothesis Agent 의 몫이다.", ""]

    L += ["## 1. Catalog provenance", "",
          f"- `catalog_sha256` = `{package['catalog_sha256']}`",
          f"- 생성 `{package['created_at']}`",
          f"- 평가 창 `{package['outcome_definition']['horizon']}` · 비용 "
          f"{package['outcome_definition']['cost_bps']:g}bps",
          f"- 시간 격자 {list(package['config']['relative_times_ms'])} ms · "
          f"freshness 한도 {package['config']['freshness_limit_ms']}ms", ""]

    L += ["## 2. Outcome 정의", "", "| 항목 | 값 |", "|---|---|"]
    for key, value in package["outcome_definition"].items():
        L.append(f"| `{key}` | {value} |")
    L.append("")

    used = sorted(canonical.feature_name.unique())
    L += ["## 3. Profile feature", "",
          f"**{len(used)}개, 전부 Catalog v2 정본이다.**", "",
          "| feature | definition_id | family | dimension |", "|---|---|---|---|"]
    for name in used:
        spec = catalog.FEATURES[name]
        L.append(f"| `{name}` | `{spec.definition_id}` | {family_of(spec)} | {spec.dimension} |")
    L += ["", f"### 빠진 feature {len(dropped)}개와 이유", "",
          "| feature | 이유 |", "|---|---|"]
    for row in dropped.itertuples():
        L.append(f"| `{row.feature}` | {row.reason} |")
    L.append("")

    L += ["## 4. Cohort 표본", "",
          "| cohort | anchor | 종목 | 날짜 | 종목-일 |", "|---|--:|--:|--:|--:|"]
    for cohort, info in package["cohort_summary"].items():
        L.append(f"| {cohort} | {info['anchors']} | {info['unique_symbols']} | "
                 f"{info['unique_dates']} | {info['symbol_days']} |")
    L.append("")

    L += ["## 5. Cohort 구성 (종목·날짜·시간대)", ""]
    for axis in ("date", "cluster_id"):
        share = (anchors.groupby(["cohort", axis]).size()
                 / anchors.groupby("cohort").size()).unstack(axis).round(3)
        L += [f"### {axis} 비중", "", share.to_markdown(), ""]
    top = anchors.groupby("cohort").symbol.value_counts(normalize=True).groupby("cohort").head(1)
    L += ["### 상위 종목 비중", "", "| cohort | 종목 | 비중 |", "|---|---|--:|"]
    for (cohort, symbol), value in top.items():
        L.append(f"| {cohort} | {symbol} | {value:.1%} |")
    L.append("")

    L += ["## 6. Leakage 검사", "", "| 항목 | 결과 |", "|---|---|",
          f"| 시간 격자에 미래가 있는가 | {'예' if any(t > 0 for t in package['config']['relative_times_ms']) else '아니오'} |",
          f"| 미래 관측을 읽었는가 (`age_ms < 0`) | {'예' if (long.age_ms.dropna() < 0).any() else '아니오'} |",
          f"| 미래를 보는 feature | {[n for n in used if not catalog.FEATURES[n].causal] or '없음'} |",
          "| 미래로 고른 대조군 | 없음 (`near_miss`·`mechanism_failure` 미사용) |",
          "| 미래를 쓰는 곳 | outcome 라벨뿐 |", ""]

    L += ["## 7. Coverage", "",
          "### anchor 정렬 (feature 가 아니라 anchor 자체가 정렬됐는가)", "",
          alignment.pivot_table(index="relative_time_ms", columns="cohort",
                                values="alignment_coverage").round(3).to_markdown(), "",
          "### 관측 나이 중앙(ms)", "",
          alignment.pivot_table(index="relative_time_ms", columns="cohort",
                                values="median_age_ms").round(0).to_markdown(), ""]

    at0 = contrast[(contrast.relative_time_ms == 0) & (contrast.cohort_a == PROFIT)
                   & (contrast.cohort_b == LOSS)].nlargest(12, "effect_strength")
    L += ["## 8. 가장 크게 갈리는 feature (PROFIT vs LOSS, t)", "",
          "AUC 를 그대로 읽지 않는다. 방향과 세기를 나눠 적었다.", "",
          "| feature | family | AUC | 방향 | 세기 | 날짜 일치 | 종목 일치 | 종목 수 |",
          "|---|---|--:|---|--:|--:|--:|--:|"]
    for r in at0.itertuples():
        L.append(f"| `{r.feature_name}` | {r.family} | {r.auc:.3f} | {r.direction} | "
                 f"{r.effect_strength:.3f} | {r.date_agreement_ratio:.0%} | "
                 f"{r.symbol_agreement_ratio:.0%} | {r.symbol_total} |")
    L.append("")

    L += ["## 9. 중복 묶음", "", "| 묶음 | 대표 | 관련 feature | 묶음 내 최대 |rho| |",
          "|---|---|---|--:|"]
    for gid, grp in groups.groupby("correlation_group"):
        rep = grp[grp.is_representative].feature.iloc[0]
        rest = sorted(set(grp.feature) - {rep})
        rho = grp.max_abs_rho_in_group.max()
        L.append(f"| EV_{gid} | `{rep}` | {', '.join(f'`{x}`' for x in rest) or '—'} | "
                 f"{'' if not np.isfinite(rho) else f'{rho:.2f}'} |")
    L += ["", "대표는 **효과가 큰 것으로 고르지 않는다** — 파생이 아닌 것, 관측이 많은 것, "
          "lookback 이 짧은 것 순이다.", ""]

    L += ["## 10. 시간 패턴 (PROFIT vs LOSS)", "",
          f"궤적은 중앙값 차이가 아니라 분리도 `2 × (AUC − 0.5)` 다. "
          f"시각 {package['config']['relative_times_ms']} ms 순서.", "",
          "| feature | 패턴 | 분리도 궤적 |", "|---|---|---|"]
    for r in temporal.sort_values("feature_name").itertuples():
        traj = ", ".join("—" if v is None else f"{v:+.2f}" for v in r.separation)
        L.append(f"| `{r.feature_name}` | {r.temporal_pattern} | {traj} |")
    L.append("")

    L += ["## 11. 날짜·종목 안정성", "",
          "| feature | 날짜 일치 | 종목 일치 | 종목 수 | anchor 집계 | 종목-일 균형 | 이동(SD) |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    merged = at0.merge(balanced, on="feature_name", how="left")
    for r in merged.itertuples():
        L.append(f"| `{r.feature_name}` | {r.date_agreement_ratio:.0%} | "
                 f"{r.symbol_agreement_ratio:.0%} | {r.symbol_total} | "
                 f"{r.anchor_weighted_diff:.4g} | {r.symbol_day_balanced_diff:.4g} | "
                 f"{r.shift_sd:+.2f} |")
    L.append("")

    L += ["## 12. 경고", ""]
    if package["global_warnings"]:
        L += [f"- {w}" for w in package["global_warnings"]]
    else:
        L.append("없음.")
    L.append("")

    L += ["## 13. 대표 raw 사례", "",
          "각 cohort 에서 두 개씩. 시각 격자별 값과 관측 나이를 그대로 보인다.", ""]
    show = ("ofi_depth_5", "mid_return_5t_bps", "book_imbalance", "spread_bps")
    show = [c for c in show if c in long.columns]
    for cohort in COHORTS:
        picks = long[long.cohort == cohort].anchor_id.drop_duplicates().head(2)
        for anchor_id in picks:
            part = long[long.anchor_id == anchor_id].sort_values("relative_time_ms")
            meta = anchors[anchors.anchor_id == anchor_id]
            best = float(meta.best_ask_net_bps.iloc[0]) if len(meta) else float("nan")
            L += [f"### {cohort} · `{anchor_id}` (창 안 최고 ASK1 가격 기회 net {best:+.1f} bps)", "",
                  "| Δt(ms) | age(ms) | " + " | ".join(f"`{c}`" for c in show) + " |",
                  "|--:|--:|" + "--:|" * len(show)]
            for r in part.itertuples():
                age = "—" if not np.isfinite(r.age_ms) else f"{r.age_ms:.0f}"
                cells = " | ".join("—" if not np.isfinite(getattr(r, c)) else f"{getattr(r, c):.4g}"
                                   for c in show)
                L.append(f"| {r.relative_time_ms} | {age} | {cells} |")
            L.append("")

    if shadow is not None and len(shadow):
        L += ["## 14. 구 profile 과의 shadow 비교 (§49)", "",
              "구 경로는 `profit vs near_miss` (near_miss 의 97% 가 수익 구간 안), "
              "새 경로는 `PROFIT vs LOSS` 다. 값이 아니라 **방향과 순위**를 본다.", "",
              "| feature | canonical AUC | 방향 | 순위 | legacy AUC | legacy 방향 | legacy 순위 | 방향 반복 |",
              "|---|--:|---|--:|--:|---|--:|:--:|"]
        for r in shadow.sort_values("canonical_rank").itertuples():
            legacy_auc = "—" if not np.isfinite(r.legacy_auc) else f"{r.legacy_auc:.3f}"
            legacy_rank = "—" if not np.isfinite(r.legacy_rank) else f"{r.legacy_rank:.0f}"
            mark = "✅" if r.direction_repeats else ("—" if r.legacy_direction == "NOT_IN_LEGACY" else "❌")
            L.append(f"| `{r.feature_name}` | {r.canonical_auc:.3f} | {r.canonical_direction} | "
                     f"{r.canonical_rank:.0f} | {legacy_auc} | {r.legacy_direction} | "
                     f"{legacy_rank} | {mark} |")
        L.append("")

    return "\n".join(L) + "\n"


# ---- 배선 --------------------------------------------------------------------

def build(output: Path, *, config: ProfilerConfig = ProfilerConfig(),
          clusters: Sequence[str] | None = None, root: Path = TICK_ROOT,
          legacy_profile: Path | None = None,
          profile: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict[str, Any]:
    """전 구간을 돌려 산출물 다섯 개를 쓴다. 게이트에 걸리면 예외.

    `profile` 로 `(long, anchors)` 를 주면 틱 데이터를 다시 읽지 않고 그것으로 위층을
    만든다. 이 계산의 바닥은 `anchor_evidence` 한 장이고, canonical·contrastive·
    groups·temporal·balanced·alignment 는 전부 거기서 유도되기 때문이다.

    저장해 둔 Feature Profile 을 그대로 넣어 돌릴 수 있게 하려는 것이다 — 종목 하나,
    기간 하나를 골라 파이프라인을 태우는 실험이 가능해진다.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = catalog.catalog_hash()

    if profile is not None:
        long, anchors = profile
        long, anchors = long.copy(), anchors.copy()
        notes = {"partitions": int(anchors.groupby(["symbol", "date"]).ngroups),
                 "skipped": [], "profit_floor_violations": 0,
                 "source": "stored_feature_profile"}
    else:
        anchors, notes = build_anchors(config, clusters, root)
        long = profile_anchors(anchors, config, root=root)
    features = tuple(sorted(set(long.columns)
                            & set(evidence_features(
                                catalog.DataCapabilities(
                                    fields=frozenset(catalog.RAW_FIELDS), book_depth=10,
                                    has_trade_classification=True, has_timestamp=True,
                                    timestamp_resolution="microsecond")))))

    fail, warn = quality_gate(long, features, anchors, config, stamp)
    if fail:
        raise EvidenceGateError("증거 게이트 실패 " + str(len(fail)) + "건:\n  - "
                                + "\n  - ".join(fail))

    canonical = canonical_profile(long, features)
    contrast = contrastive_profile(long, features, config)
    groups = evidence_groups(long, features, config)
    temporal = temporal_summary(long, features, contrast, config)
    balanced = balanced_effects(long, features, 0)
    alignment = alignment_coverage(long)

    shift = balanced[balanced.shift_sd.abs() > config.weighting_shift_warning_sd]
    warn += [f"WEIGHTING_SHIFT: {r.feature_name} {r.shift_sd:+.2f}sd" for r in shift.itertuples()]
    # 같은 (쌍, 시각) 에서 feature 20개가 똑같이 걸리면 원인은 anchor 정렬이다.
    flagged = contrast[contrast.missingness_confound]
    for (a, b, rel), part in flagged.groupby(["cohort_a", "cohort_b", "relative_time_ms"]):
        cause = "anchor_alignment" if len(part) == len(features) else "feature_specific"
        warn.append(f"MISSINGNESS_CONFOUND[{cause}]: {a}/{b} @{rel}ms "
                    f"feature {len(part)}/{len(features)} gap "
                    f"{part.coverage_gap.min():.2f}~{part.coverage_gap.max():.2f}")

    package = evidence_package(canonical, contrast, groups, temporal, anchors, balanced,
                               config, warn, alignment)

    canonical.to_parquet(output / "canonical_profile.parquet", index=False)
    contrast.to_parquet(output / "contrastive_profile.parquet", index=False)
    groups.to_parquet(output / "evidence_groups.parquet", index=False)
    temporal.to_parquet(output / "temporal_profile.parquet", index=False)
    alignment.to_parquet(output / "alignment_coverage.parquet", index=False)
    balanced.to_parquet(output / "balanced_effects.parquet", index=False)
    long.to_parquet(output / "anchor_evidence.parquet", index=False)
    anchors.to_parquet(output / "anchors.parquet", index=False)
    write_json(output / "hypothesis_evidence.json", package)

    capabilities = catalog.DataCapabilities(
        fields=frozenset(catalog.RAW_FIELDS), book_depth=10, has_trade_classification=True,
        has_timestamp=True, timestamp_resolution="microsecond")
    dropped = dropped_features(capabilities)
    dropped.to_parquet(output / "dropped_features.parquet", index=False)
    shadow = None
    if legacy_profile is not None and Path(legacy_profile).exists():
        shadow = shadow_comparison(contrast, pd.read_parquet(legacy_profile))
        shadow.to_parquet(output / "shadow_comparison.parquet", index=False)
    (output / "PROFILE_VALIDATION.md").write_text(
        validation_report(package, canonical, contrast, groups, temporal, anchors,
                          dropped, alignment, long, balanced, shadow), encoding="utf-8")

    return {"output": str(output), "catalog_sha256": stamp, "features": list(features),
            "anchors": anchors.cohort.value_counts().to_dict(), "notes": notes,
            "warnings": warn, "package": package,
            "tables": {"canonical": len(canonical), "contrast": len(contrast),
                       "groups": len(groups), "anchor_evidence": len(long),
                       "temporal": len(temporal), "alignment": len(alignment)},
            "alignment": alignment}
