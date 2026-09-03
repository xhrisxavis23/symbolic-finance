"""Execution-aligned Feature Profile used as the successor to the oracle profile.

The stored evidence shape remains ``anchor × pre-anchor time × catalog feature``.
Only the discovery label changes: an event is first posted at BID1, must actually fill
the canonical queue, and is then classified with the fixed Canonical Queue V9 exit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import (canonical as K, catalog, contract, data, evidence, featureprofile, fill, ledger,
               sample_condition as SC)
from .config import ENTRY_ORDER_MAX_SECONDS, ENTRY_ORDER_MAX_TICKS, FEE_BPS, TICK_ROOT, write_json


PROFILE_TYPE = "execution_aligned.v1"
STORE = featureprofile.STORE.with_name("FeatureProfileExecutionAligned")
STRONG_PROFIT, LOSS_TAIL, BACKGROUND, UNFILLED = (
    "STRONG_PROFIT", "LOSS_TAIL", "BACKGROUND", "UNFILLED")
CORE_COHORT = {STRONG_PROFIT: "PROFIT", LOSS_TAIL: "LOSS", BACKGROUND: "BACKGROUND"}


@dataclass(frozen=True)
class ExecutionProfileConfig:
    """Fixed discovery definition; thresholds are labels, never runtime parameters."""

    event_source: str = "m0_price_recovery"
    relative_times_ms: tuple[int, ...] = (-10_000, -5_000, -2_000, -1_000, 0)
    max_anchors_per_outcome_per_symbol_day: int = 3
    strong_profit_net_bps: float = 20.0
    loss_tail_net_bps: float = -30.0
    entry_queue_fraction: float = 1.0
    entry_max_seconds: float = ENTRY_ORDER_MAX_SECONDS
    entry_max_ticks: int = ENTRY_ORDER_MAX_TICKS


def _m0_price_recovery_ticks(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Predefined event source: the 5-tick mid return crosses from non-positive to positive."""
    expression = SC.m0_price_recovery()
    values = np.asarray(contract.ExpressionRuntime(arrays).evaluate(expression), dtype=bool)
    values[:20] = False
    return np.flatnonzero(values).astype(np.int64)


def _source_ticks(arrays: dict[str, np.ndarray], source: str) -> np.ndarray:
    if source == "m0_price_recovery":
        return _m0_price_recovery_ticks(arrays)
    raise ValueError(f"지원하지 않는 execution Profile event source: {source}")


def _cohort(net_bps: float, exit_reason: str, config: ExecutionProfileConfig) -> str:
    if net_bps >= config.strong_profit_net_bps:
        return STRONG_PROFIT
    if net_bps <= config.loss_tail_net_bps or str(exit_reason).startswith("stop"):
        return LOSS_TAIL
    return BACKGROUND


def _spaced(values: Iterable[int], cap: int) -> list[int]:
    values = np.asarray(sorted({int(v) for v in values}), dtype=np.int64)
    if not len(values):
        return []
    positions = np.linspace(0, len(values) - 1, min(int(cap), len(values))).round().astype(int)
    return [int(values[i]) for i in dict.fromkeys(positions)]


def _exit_job() -> dict[str, float]:
    spec = K.CANONICAL["exit"]
    return {
        "stop_gross_bps": float(spec["stop_gross_bps"]),
        "trailing_drawdown_gross_bps": float(spec["trailing_drawdown_gross_bps"]),
        "exit_limit_rest_seconds": float(spec["limit_rest_seconds"]),
        "horizon_seconds": float(spec["max_holding_seconds"]),
    }


def _missing_queue_events(error: ValueError) -> bool:
    """체결 원천이 없는 종목-일만 Profile에서 제외한다.

    체결량 열이 아예 없으면 BID1/ASK1 큐 체결을 판정할 수 없다. 이를 미체결로
    바꾸면 관측하지 못한 것을 0 체결로 지어내게 되므로, 전종목 cache에서는 해당
    종목-일을 명시적으로 건너뛴다.
    """
    return "큐 재생에는 체결 이벤트가 필요하다" in str(error)


def label_events(symbol: str, date: str, cluster: str, *, config: ExecutionProfileConfig,
                 root: Path = TICK_ROOT) -> pd.DataFrame:
    """Classify every predefined event using the real entry and fixed canonical exit."""
    arrays, _ = data.load(symbol, date, root)
    time_s = np.asarray(arrays["time_s"], dtype=float)
    bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    candidates = _source_ticks(arrays, config.event_source)
    # A label needs the complete entry wait and exit horizon. Truncated paths are excluded.
    candidates = candidates[time_s[candidates] + config.entry_max_seconds
                            + float(K.CANONICAL["exit"]["max_holding_seconds"]) <= time_s[-1]]
    status, terminal, filled = fill.entry_fills(
        arrays, candidates, qfrac=config.entry_queue_fraction,
        max_seconds=config.entry_max_seconds, max_ticks=config.entry_max_ticks)
    horizon = ledger.horizon_exit_ticks(
        time_s, seconds=float(K.CANONICAL["exit"]["max_holding_seconds"]))
    rows: list[dict[str, Any]] = []
    for pos, raw_tick in enumerate(candidates):
        tick = int(raw_tick)
        if not (np.isfinite(bid1[tick]) and bid1[tick] > 0):
            continue
        fill_tick = int(filled[pos])
        common = {
            "cluster_id": cluster, "symbol": str(symbol).zfill(6), "date": str(date), "tick": tick,
            "time_s": float(time_s[tick]), "anchor_source": config.event_source,
            "entry_assumption": "canonical_bid1_queue",
            "entry_queue_status": str(status[pos]), "entry_terminal_tick": int(terminal[pos]),
            "entry_fill_tick": fill_tick, "entry_wait_ticks": fill_tick - tick if fill_tick >= 0 else -1,
            "entry_wait_seconds": (float(time_s[fill_tick] - time_s[tick]) if fill_tick >= 0 else np.nan),
            "entry_queue_fraction": float(config.entry_queue_fraction),
            "entry_oracle_ask_tick": -1, "entry_fill_before_oracle_ask": False,
            "exit_profile_id": K.CANONICAL["profile_id"],
            # Legacy field remains diagnostic-only; it is intentionally not calculated as a label.
            "best_ask_net_bps": np.nan,
        }
        if fill_tick < 0:
            rows.append({**common, "execution_cohort": UNFILLED, "cohort": UNFILLED,
                         "canonical_exit_net_bps": np.nan, "canonical_exit_tick": -1,
                         "canonical_exit_reason": "entry_unfilled", "holding_seconds": np.nan,
                         "outcome_observable": False})
            continue
        price, exit_tick, reason = ledger._resolve_exit(
            arrays, time_s, bid1, ask1, len(time_s), fill_tick, float(bid1[tick]),
            int(horizon[fill_tick]), True, _exit_job())
        if price is None or exit_tick is None:
            continue
        net = (float(price) / float(bid1[tick]) - 1.0) * 10_000.0 - FEE_BPS
        semantic = _cohort(float(net), str(reason), config)
        rows.append({**common, "execution_cohort": semantic, "cohort": CORE_COHORT[semantic],
                     "canonical_exit_net_bps": float(net), "canonical_exit_tick": int(exit_tick),
                     "canonical_exit_reason": str(reason),
                     "holding_seconds": float(time_s[exit_tick] - time_s[fill_tick]),
                     "outcome_observable": True})
    return pd.DataFrame(rows)


def choose_anchors(labels: pd.DataFrame, *, config: ExecutionProfileConfig) -> pd.DataFrame:
    """Keep a balanced explanatory sample while retaining the full event-label table separately."""
    rows: list[dict[str, Any]] = []
    for _key, part in labels.groupby(["cluster_id", "symbol", "date"], sort=True):
        by_tick = part.set_index("tick", drop=False)
        # Strong-profit and loss-tail are the contrast sample. BACKGROUND is the full
        # matched middle outcome sample, so it receives one slot for every contrast
        # anchor (up to six), exactly as Sandbox_6 run_006 did.
        contrast_count = 0
        for semantic in (STRONG_PROFIT, LOSS_TAIL):
            picks = _spaced(part.loc[part.execution_cohort.eq(semantic), "tick"],
                            config.max_anchors_per_outcome_per_symbol_day)
            contrast_count += len(picks)
            rows.extend(by_tick.loc[tick].to_dict() for tick in picks)
        for tick in _spaced(part.loc[part.execution_cohort.eq(BACKGROUND), "tick"], contrast_count):
            rows.append(by_tick.loc[tick].to_dict())
        for tick in _spaced(part.loc[part.execution_cohort.eq(UNFILLED), "tick"],
                            config.max_anchors_per_outcome_per_symbol_day):
            rows.append(by_tick.loc[tick].to_dict())
    anchors = pd.DataFrame(rows, columns=list(labels.columns)) if rows else labels.iloc[0:0].copy()
    if not anchors.empty:
        anchors["anchor_id"] = (anchors.symbol.astype(str) + ":" + anchors.date.astype(str) + ":"
                                + anchors.tick.astype(str) + ":" + anchors.execution_cohort.astype(str))
    return anchors.sort_values(["cluster_id", "symbol", "date", "tick", "execution_cohort"]).reset_index(drop=True)


def materialize(cluster: str, symbols: Sequence[str], dates: Sequence[str], output: Path, *,
                config: ExecutionProfileConfig = ExecutionProfileConfig(),
                root: Path = TICK_ROOT) -> dict[str, Any]:
    """Build an execution-aligned store and its unthinned event-label ledger."""
    if not cluster or not symbols or not dates:
        raise ValueError("execution-aligned Profile은 cluster 하나와 symbols/dates를 명시해야 한다")
    output = Path(output)
    label_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for symbol in sorted({str(value).zfill(6) for value in symbols}):
        for date in sorted({str(value) for value in dates}):
            try:
                frame = label_events(symbol, date, str(cluster), config=config, root=Path(root))
            except ValueError as error:
                if not _missing_queue_events(error):
                    raise
                skipped.append({"symbol": symbol, "date": date,
                                "reason": "MISSING_QUEUE_TRADE_EVENTS",
                                "detail": str(error)})
                continue
            if not frame.empty:
                label_frames.append(frame)
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    if labels.empty:
        raise ValueError("선택 범위에 큐 체결 근거가 있는 complete-horizon execution event가 없다")
    anchors = choose_anchors(labels, config=config)
    if not {"PROFIT", "LOSS"} <= set(anchors.cohort):
        raise ValueError("execution Profile에 PROFIT/LOSS anchor가 모두 없다")
    profiler = evidence.ProfilerConfig(relative_times_ms=config.relative_times_ms)
    capabilities = catalog.DataCapabilities(fields=frozenset(catalog.RAW_FIELDS), book_depth=10,
                                             has_trade_classification=True, has_timestamp=True,
                                             timestamp_resolution="microsecond")
    fields = evidence.evidence_features(capabilities)
    long = evidence.profile_anchors(anchors, profiler, features=fields, root=Path(root))
    result = featureprofile.write(
        long, anchors, config=profiler, root=output, schema="execution_aligned_feature_profile_symbol_day.v1",
        metadata={"profile_type": PROFILE_TYPE, "execution_profile_config": asdict(config),
                  "label_contract": label_contract(config)})
    labels.to_parquet(output / "all_event_execution_labels.parquet", index=False)
    anchors.to_parquet(output / "anchors.parquet", index=False)
    write_json(output / "execution_profile_contract.json", label_contract(config))
    return {**result, "profile_type": PROFILE_TYPE, "event_count": int(len(labels)),
            "anchor_count": int(len(anchors)),
            "event_cohorts": {str(k): int(v) for k, v in labels.execution_cohort.value_counts().items()},
            "anchor_cohorts": {str(k): int(v) for k, v in anchors.execution_cohort.value_counts().items()},
            "skipped": skipped}


def label_contract(config: ExecutionProfileConfig) -> dict[str, Any]:
    sample_condition = SC.for_event_source(config.event_source)
    return {
        "profile_type": PROFILE_TYPE,
        "event_source": config.event_source,
        "executable_sample_condition_schema": SC.SCHEMA_VERSION,
        "executable_sample_condition": sample_condition,
        "executable_sample_condition_sha256": SC.condition_hash(sample_condition),
        "executable_sample_condition_warmup_ticks": SC.warmup_ticks(sample_condition),
        "entry": {"model": "canonical BID1 queue", "max_seconds": config.entry_max_seconds,
                  "max_ticks": config.entry_max_ticks, "queue_fraction": config.entry_queue_fraction},
        "exit": dict(K.CANONICAL["exit"]), "cost_bps": FEE_BPS,
        "cohorts": {"PROFIT": STRONG_PROFIT, "LOSS": LOSS_TAIL, "BACKGROUND": BACKGROUND,
                    "UNFILLED": UNFILLED},
        "cohort_rule": {STRONG_PROFIT: f"canonical_exit_net_bps >= {config.strong_profit_net_bps}",
                        LOSS_TAIL: f"canonical_exit_net_bps <= {config.loss_tail_net_bps} or stop exit"},
        "oracle": True, "not_live_profitability_evidence": True,
        "not_strategy_pnl": "Events are independently replayed; labels are discovery-only.",
    }
