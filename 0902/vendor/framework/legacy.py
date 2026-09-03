"""과거 AlgorithmPackage를 현재 정본 Backtest로 재생한다.

과거 패키지는 코드가 아니라 읽기 전용 JSON 산출물이다. 조건식과 분위수만 읽고,
실행일마다 그 종목의 직전 유효일 분포로 컷을 만든다. ``entry-only`` corpus는 과거
청산 규칙을 보존하지 않고, 현재 canonical 청산만 적용한다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import canonical as K, data as tickdata
from .config import (MIN_REFERENCE_SAMPLES, TICK_ROOT, now_utc, read_json, sha256_file,
                     sha256_json, write_json)


SCHEMA_VERSION = "legacy_algorithm_replay.v2"
ENTRY_ONLY_SCHEMA_VERSION = "legacy_entry_only_corpus.v1"
FIRST_TRIGGER_RULE = "LEGACY_STATE_OR_CANONICAL_DEFAULT_FIRST_TRIGGER"
EPS = 1.0e-9


def _book(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bid_qty = np.asarray(arrays["bid_qty"], dtype=float)
    ask_qty = np.asarray(arrays["ask_qty"], dtype=float)
    bid_price = np.asarray(arrays["bid_price"], dtype=float)
    ask_price = np.asarray(arrays["ask_price"], dtype=float)
    return bid_qty, ask_qty, bid_price, ask_price


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(len(values)) + 1
    starts = np.maximum(ends - int(window), 0)
    return prefix[ends] - prefix[starts]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count = np.minimum(np.arange(len(values)) + 1, int(window))
    return _rolling_sum(values, window) / count


def _imbalance(bid_qty: np.ndarray, ask_qty: np.ndarray, start: int, stop: int) -> np.ndarray:
    bid = bid_qty[:, start:stop].sum(axis=1)
    ask = ask_qty[:, start:stop].sum(axis=1)
    return (bid - ask) / (bid + ask + EPS)


def _mid_return(mid: np.ndarray, lag: int) -> np.ndarray:
    result = np.zeros(len(mid), dtype=float)
    if len(mid) > lag:
        before = mid[:-lag]
        valid = np.isfinite(mid[lag:]) & np.isfinite(before) & (before > 0.0)
        indices = np.flatnonzero(valid) + int(lag)
        result[indices] = (mid[indices] / mid[indices - int(lag)] - 1.0) * 10_000.0
    return result


def _lagged_difference(values: np.ndarray, lag: int, *, initial: float = np.nan) -> np.ndarray:
    """현재 값에서 정확히 ``lag`` tick 전 값을 뺀다."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), float(initial), dtype=float)
    if len(values) > int(lag):
        result[int(lag):] = values[int(lag):] - values[:-int(lag)]
    return result


def _relative_depth_change(depth: np.ndarray, lag: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=float)
    result = np.full(len(depth), np.nan, dtype=float)
    if len(depth) > int(lag):
        prior = depth[:-int(lag)]
        result[int(lag):] = (depth[int(lag):] - prior) / (prior + EPS)
    return result


def _positive_l1_build(bid_qty: np.ndarray, window: int) -> np.ndarray:
    delta = np.zeros(len(bid_qty), dtype=float)
    if len(delta) > 1:
        delta[1:] = np.maximum(bid_qty[1:, 0] - bid_qty[:-1, 0], 0.0)
    return _rolling_sum(delta, window)


def _microprice(bid_qty: np.ndarray, ask_qty: np.ndarray,
                bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    bid1 = np.asarray(bid_price[:, 0], dtype=float)
    ask1 = np.asarray(ask_price[:, 0], dtype=float)
    return ((bid1 * ask_qty[:, 0] + ask1 * bid_qty[:, 0])
            / (bid_qty[:, 0] + ask_qty[:, 0] + EPS))


def _vamp(bid_qty: np.ndarray, ask_qty: np.ndarray,
          bid_price: np.ndarray, ask_price: np.ndarray, depth: int) -> np.ndarray:
    stop = int(depth)
    numerator = (bid_price[:, :stop] * ask_qty[:, :stop]
                 + ask_price[:, :stop] * bid_qty[:, :stop]).sum(axis=1)
    denominator = (bid_qty[:, :stop] + ask_qty[:, :stop]).sum(axis=1)
    return numerator / (denominator + EPS)


def _cks_ofi(bid_qty: np.ndarray, ask_qty: np.ndarray,
             bid_price: np.ndarray, ask_price: np.ndarray, depth: int) -> np.ndarray:
    """가격 변화 조건을 포함한 Cont–Kukanov–Stoikov depth OFI."""
    result = np.zeros(len(bid_qty), dtype=float)
    if len(result) < 2:
        return result
    for level in range(int(depth)):
        bp_now, bp_before = bid_price[1:, level], bid_price[:-1, level]
        ap_now, ap_before = ask_price[1:, level], ask_price[:-1, level]
        bq_now, bq_before = bid_qty[1:, level], bid_qty[:-1, level]
        aq_now, aq_before = ask_qty[1:, level], ask_qty[:-1, level]
        bid_valid = (np.isfinite(bp_now) & np.isfinite(bp_before)
                     & (bp_now > 0.0) & (bp_before > 0.0))
        ask_valid = (np.isfinite(ap_now) & np.isfinite(ap_before)
                     & (ap_now > 0.0) & (ap_before > 0.0))
        bid_event = np.zeros(len(result) - 1, dtype=float)
        ask_event = np.zeros(len(result) - 1, dtype=float)
        bid_event[bid_valid] = np.where(
            bp_now[bid_valid] > bp_before[bid_valid], bq_now[bid_valid],
            np.where(bp_now[bid_valid] < bp_before[bid_valid], -bq_before[bid_valid],
                     bq_now[bid_valid] - bq_before[bid_valid]))
        ask_event[ask_valid] = np.where(
            ap_now[ask_valid] < ap_before[ask_valid], aq_now[ask_valid],
            np.where(ap_now[ask_valid] > ap_before[ask_valid], -aq_before[ask_valid],
                     aq_now[ask_valid] - aq_before[ask_valid]))
        result[1:] += bid_event - ask_event
    return result


def _trade_event_ratio(arrays: Mapping[str, np.ndarray], side: str, window: int = 50) -> np.ndarray:
    if "buy_volume" not in arrays or "sell_volume" not in arrays:
        raise KeyError("과거 taker 비율은 매수·매도 체결량이 필요하다")
    buy = (np.asarray(arrays["buy_volume"], dtype=float) > 0.0).astype(float)
    sell = (np.asarray(arrays["sell_volume"], dtype=float) > 0.0).astype(float)
    numerator = _rolling_sum(buy if side == "buy" else sell, window)
    denominator = _rolling_sum(buy + sell, window)
    return numerator / (denominator + EPS)


def _ask_vacuum(ask_qty: np.ndarray, window: int = 100) -> np.ndarray:
    delta = np.zeros(len(ask_qty), dtype=float)
    if len(delta) > 1:
        delta[1:] = np.maximum(ask_qty[:-1, 0] - ask_qty[1:, 0], 0.0)
    return _rolling_sum(delta, window)


def _ask_vacuum_20(ask_qty: np.ndarray) -> np.ndarray:
    """과거 Generation 패키지의 L1~L3 상대 매도잔량 감소율."""
    depth = np.asarray(ask_qty[:, :3], dtype=float).sum(axis=1)
    result = np.full(len(depth), np.nan)
    if len(depth) > 20:
        prior = depth[:-20]
        result[20:] = (prior - depth[20:]) / (prior + EPS)
    return result


def _spread_bps(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    bid1 = np.asarray(bid_price[:, 0], dtype=float)
    ask1 = np.asarray(ask_price[:, 0], dtype=float)
    return (ask1 - bid1) / ((ask1 + bid1) / 2.0 + EPS) * 10_000.0


def _microprice_deviation_bps(bid_qty: np.ndarray, ask_qty: np.ndarray,
                              bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    bid1 = np.asarray(bid_price[:, 0], dtype=float)
    ask1 = np.asarray(ask_price[:, 0], dtype=float)
    bid_depth = np.asarray(bid_qty[:, 0], dtype=float)
    ask_depth = np.asarray(ask_qty[:, 0], dtype=float)
    mid = (bid1 + ask1) / 2.0
    microprice = bid1 + (ask1 - bid1) * bid_depth / (bid_depth + ask_depth + EPS)
    return (microprice - mid) / (mid + EPS) * 10_000.0


def _ofi_depth_5(bid_qty: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
    """과거 strategy-search의 L1~L5 잔량 변화식. CKS OFI와 다르다."""
    imbalance = (np.asarray(bid_qty[:, :5], dtype=float).sum(axis=1)
                 - np.asarray(ask_qty[:, :5], dtype=float).sum(axis=1))
    result = np.zeros(len(imbalance), dtype=float)
    if len(result) > 1:
        result[1:] = imbalance[1:] - imbalance[:-1]
    return result


def _signed_aggr_flow(arrays: Mapping[str, np.ndarray], window: int) -> np.ndarray:
    if "buy_volume" not in arrays or "sell_volume" not in arrays:
        raise KeyError("과거 signed_aggr_flow는 매수·매도 체결량이 필요하다")
    buy = _rolling_sum(np.asarray(arrays["buy_volume"], dtype=float), window)
    sell = _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), window)
    return (buy - sell) / (buy + sell + EPS)


def _signed_volume(arrays: Mapping[str, np.ndarray], window: int) -> np.ndarray:
    if "buy_volume" not in arrays or "sell_volume" not in arrays:
        raise KeyError("과거 signed_volume_cumulative는 매수·매도 체결량이 필요하다")
    signed = (np.asarray(arrays["buy_volume"], dtype=float)
              - np.asarray(arrays["sell_volume"], dtype=float))
    return _rolling_sum(signed, window)


def _kyle_lambda(arrays: Mapping[str, np.ndarray], mid: np.ndarray) -> np.ndarray:
    total_volume = _rolling_sum(np.asarray(arrays["buy_volume"], dtype=float), 20)
    total_volume += _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 20)
    return np.abs(_mid_return(mid, 20)) / (total_volume + EPS)


def feature_values(arrays: Mapping[str, np.ndarray], feature: str) -> np.ndarray:
    """지원한 과거 feature의 원래 식. `raw:`는 이름 표기만 다르고 식은 같다."""
    name = str(feature)
    if name.startswith("raw:"):
        name = name[4:]
    bid_qty, ask_qty, bid_price, ask_price = _book(arrays)
    mid = (bid_price[:, 0] + ask_price[:, 0]) / 2.0
    spread = _spread_bps(bid_price, ask_price)
    obi1 = _imbalance(bid_qty, ask_qty, 0, 1)
    obi5 = _imbalance(bid_qty, ask_qty, 0, 5)
    bid_total_10 = bid_qty[:, :10].sum(axis=1)
    ask_total_10 = ask_qty[:, :10].sum(axis=1)

    if name == "ask_vacuum_20":
        return _ask_vacuum_20(ask_qty)
    if name == "ask_vacuum_100":
        return _ask_vacuum(ask_qty, 100)
    if name == "obi_3":
        return _imbalance(bid_qty, ask_qty, 0, 3)
    if name == "obi_5":
        return _imbalance(bid_qty, ask_qty, 0, 5)
    if name == "obi_10":
        return _imbalance(bid_qty, ask_qty, 0, 10)
    if name == "obi_ex_bbo":
        return _imbalance(bid_qty, ask_qty, 1, 10)
    if name == "deep_depth_imbalance_6_10":
        return _imbalance(bid_qty, ask_qty, 5, 10)
    if name == "ask_depth_total_5":
        return ask_qty[:, :5].sum(axis=1)
    if name == "ask_depth_total_10":
        return ask_total_10
    if name == "ask_depth_concentration":
        return ask_qty[:, 0] / (ask_total_10 + EPS)
    if name == "bid_depth_concentration":
        return bid_qty[:, 0] / (bid_total_10 + EPS)
    if name == "bid_depth_total_5":
        return bid_qty[:, :5].sum(axis=1)
    if name == "bid_depth_total_10":
        return bid_total_10
    if name == "bid_build_20":
        return _relative_depth_change(bid_qty[:, :5].sum(axis=1), 20)
    if name == "bid_build_100":
        return _positive_l1_build(bid_qty, 100)
    if name == "book_imbalance_velocity":
        return _lagged_difference(obi1, 1)
    if name == "book_skew_l1_vs_l2_3":
        # 해당 strategy-search 이름의 현재 정의는 L1 imbalance다.
        return obi1
    if name == "book_slope_ask":
        return (ask_price[:, 9] - ask_price[:, 0]) / (ask_total_10 + EPS)
    if name == "book_slope_bid":
        return ((bid_price[:, 0] - bid_price[:, 4])
                / (bid_qty[:, :5].sum(axis=1) + EPS))
    if name == "book_thickness":
        return bid_total_10 + ask_total_10
    if name == "ask_queue_depth_at_l1":
        return ask_qty[:, 0]
    if name == "bid_queue_depth_at_l1":
        return bid_qty[:, 0]
    if name == "ask_queue_arrival_count_5s":
        events = np.zeros(len(ask_price), dtype=float)
        if len(events) > 1:
            events[1:] = ((ask_price[1:, 0] < ask_price[:-1, 0])
                          & (ask_price[:-1, 0] > 0.0)).astype(float)
        return _rolling_sum(events, 50)
    if name == "bid_queue_depletion_5s_bps":
        events = np.zeros(len(bid_price), dtype=float)
        if len(events) > 1:
            events[1:] = ((bid_price[1:, 0] < bid_price[:-1, 0])
                          & (bid_price[:-1, 0] > 0.0)).astype(float)
        return _rolling_sum(events, 50)
    if name == "spread_bps":
        return spread
    if name == "spread_change_bps":
        return _lagged_difference(spread, 1, initial=0.0)
    if name == "spread_chg20_bps":
        return _lagged_difference(spread, 20, initial=0.0)
    if name == "spread_mean20_bps":
        return _rolling_mean(spread, 20)
    if name == "spread_mean50_bps":
        return _rolling_mean(spread, 50)
    if name == "microprice_dev_bps":
        return _microprice_deviation_bps(bid_qty, ask_qty, bid_price, ask_price)
    if name in ("microprice", "vamp_bbo"):
        return _microprice(bid_qty, ask_qty, bid_price, ask_price)
    if name == "vamp_5":
        return _vamp(bid_qty, ask_qty, bid_price, ask_price, 5)
    if name == "vamp_10":
        return _vamp(bid_qty, ask_qty, bid_price, ask_price, 10)
    if name == "microprice_velocity":
        micro = _microprice(bid_qty, ask_qty, bid_price, ask_price)
        return _lagged_difference(micro, 1) / (mid + EPS) * 10_000.0
    if name == "mid_px":
        return mid
    if name == "mid_range_bps_100":
        maximum = pd.Series(mid).rolling(100, min_periods=1).max().to_numpy()
        minimum = pd.Series(mid).rolling(100, min_periods=1).min().to_numpy()
        return (maximum - minimum) / (mid + EPS) * 10_000.0
    if name == "queue_imbalance_best":
        return bid_qty[:, 0] / (bid_qty[:, 0] + ask_qty[:, 0] + EPS)
    if name == "ofi_depth_5":
        return _ofi_depth_5(bid_qty, ask_qty)
    if name == "ofi_depth_10":
        return _cks_ofi(bid_qty, ask_qty, bid_price, ask_price, 10)
    if name == "ofi_proxy":
        bid_change = _lagged_difference(bid_total_10, 1, initial=0.0)
        ask_change = _lagged_difference(ask_total_10, 1, initial=0.0)
        return (bid_change - ask_change) / (np.abs(bid_change) + np.abs(ask_change) + EPS)
    if name == "mid_return_20t_bps":
        return _mid_return(mid, 20)
    if name == "mid_return_5t_bps":
        return _mid_return(mid, 5)
    if name == "mid_return_50t_bps":
        return _mid_return(mid, 50)
    if name == "mid_return_100t_bps":
        return _mid_return(mid, 100)
    if name == "minute_of_session":
        return np.asarray(arrays["time_s"], dtype=float) / 60.0 - 9.0 * 60.0
    if name == "obi5_velocity_20":
        return _lagged_difference(obi5, 20)
    if name == "obi5_velocity_100":
        return _lagged_difference(obi5, 100)
    if name == "obi_total":
        return (bid_total_10 - ask_total_10) / (bid_total_10 + ask_total_10 + EPS)
    if name == "signed_aggr_flow_20":
        return _signed_aggr_flow(arrays, 20)
    if name in ("signed_aggr_flow_100", "trade_imbalance_signed"):
        return _signed_aggr_flow(arrays, 100)
    if name == "recent_taker_buy_ratio":
        return _trade_event_ratio(arrays, "buy")
    if name == "recent_taker_sell_ratio":
        return _trade_event_ratio(arrays, "sell")
    if name == "maker_buy_fill_pressure":
        sells = _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 20)
        return sells / (bid_qty[:, 0] + EPS)
    if name == "signed_volume_cumulative":
        return _signed_volume(arrays, 100)
    if name == "sell100":
        return _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 100)
    if name == "sell_decel":
        sell20 = _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 20)
        sell100 = _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 100)
        return sell20 / (sell100 + EPS)
    if name == "buy_recover":
        buy20 = _rolling_sum(np.asarray(arrays["buy_volume"], dtype=float), 20)
        sell20 = _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 20)
        return (buy20 - sell20) / (buy20 + sell20 + EPS)
    if name == "vol_flow":
        return (_rolling_sum(np.asarray(arrays["buy_volume"], dtype=float), 20)
                + _rolling_sum(np.asarray(arrays["sell_volume"], dtype=float), 20))
    if name == "kyle_lambda_proxy":
        return _kyle_lambda(arrays, mid)
    raise KeyError(f"지원하지 않는 과거 feature: {feature}")


SUPPORTED_ENTRY_FEATURES = frozenset({
    "ask_depth_concentration", "ask_depth_total_5", "ask_depth_total_10",
    "ask_queue_arrival_count_5s", "ask_queue_depth_at_l1", "ask_vacuum_20",
    "ask_vacuum_100", "bid_build_20", "bid_build_100", "bid_depth_concentration",
    "bid_depth_total_5", "bid_depth_total_10", "bid_queue_depletion_5s_bps",
    "bid_queue_depth_at_l1", "book_imbalance_velocity", "book_skew_l1_vs_l2_3",
    "book_slope_ask", "book_slope_bid", "book_thickness", "buy_recover",
    "deep_depth_imbalance_6_10", "kyle_lambda_proxy", "maker_buy_fill_pressure",
    "microprice", "microprice_dev_bps", "microprice_velocity", "mid_px",
    "mid_range_bps_100", "mid_return_5t_bps", "mid_return_20t_bps",
    "mid_return_50t_bps", "mid_return_100t_bps", "minute_of_session",
    "obi5_velocity_20", "obi5_velocity_100", "obi_3", "obi_5", "obi_10",
    "obi_ex_bbo", "obi_total", "ofi_depth_5", "ofi_depth_10", "ofi_proxy",
    "queue_imbalance_best", "recent_taker_buy_ratio", "recent_taker_sell_ratio",
    "sell100", "sell_decel", "signed_aggr_flow_20", "signed_aggr_flow_100",
    "signed_volume_cumulative", "spread_bps", "spread_change_bps", "spread_chg20_bps",
    "spread_mean20_bps", "spread_mean50_bps", "trade_imbalance_signed", "vamp_5",
    "vamp_10", "vamp_bbo", "vol_flow",
})


def _bare_feature_name(feature: str) -> str:
    name = str(feature)
    return name[4:] if name.startswith("raw:") else name


def _conditions(rows: Sequence[Sequence[Any]], *, role: str,
                check_supported: bool = True) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"{role} 조건은 [feature, op, quantile]이어야 한다: {row!r}")
        raw_feature, comparator, quantile = row
        feature = str(raw_feature)
        for prefix in ("exit:", "pre_exit20t:", "pre_exit100t:"):
            if feature.startswith(prefix):
                feature = feature[len(prefix):]
                break
        if comparator not in ("<", ">"):
            raise ValueError(f"{role} comparator가 아니다: {comparator!r}")
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"{role} quantile은 (0, 1)이어야 한다: {quantile!r}")
        if check_supported and _bare_feature_name(feature) not in SUPPORTED_ENTRY_FEATURES:
            raise ValueError(f"현재 entry-only 재생기에 없는 feature: {raw_feature!r}")
        parsed.append({"source_feature": str(raw_feature), "feature": feature,
                       "comparator": str(comparator), "quantile": quantile})
    return parsed


def compile_package(path: Path, *, root: Path = TICK_ROOT,
                    entry_only: bool = False) -> dict[str, Any]:
    """과거 package의 조건식을 현재 백테스트의 전일-종목별 분위수 프로그램으로 만든다."""
    path = Path(path)
    package = read_json(path)
    entry_freeze_path = path.with_name("entry_freeze.json")
    entry_freeze = read_json(entry_freeze_path) if entry_freeze_path.exists() else {}
    symbols = tuple(str(value) for value in package.get("scope", {}).get("symbols", ()))
    if not symbols:
        raise ValueError("과거 package에 scope symbols가 없다")
    entry = _conditions(package.get("entry", {}).get("structure", ()), role="entry")

    program = {
        "schema": SCHEMA_VERSION,
        "legacy_package_path": str(path),
        "legacy_package_sha256": sha256_file(path),
        "legacy_package_id": str(package.get("id")),
        "legacy_content_hash": package.get("content_hash"),
        "symbols": list(symbols),
        "source_train_dates": list(map(str, entry_freeze.get("scope", {}).get("train_dates", ()))),
        "entry": {"threshold_policy": "PRIOR_VALID_DAY_PER_SYMBOL_QUANTILES",
                  "conditions": entry},
    }
    if entry_only:
        # 과거 exit rule/fallback/실행 정책은 새 corpus에 넣지 않는다.
        program["entry_only"] = True
        program["current_backtest_profile"] = {
            "canonical_profile_id": K.PROFILE_ID,
            "canonical_profile_sha256": K.profile_hash(),
        }
        return program

    exit_rows = package.get("exit", {}).get("rule", ())
    exit_conditions = _conditions(exit_rows, role="exit", check_supported=False)
    program["exit"] = {"threshold_policy": "PRIOR_VALID_DAY_PER_SYMBOL_QUANTILES",
                       "conditions": exit_conditions,
                       "action": "BID1_MARKET"}
    program["exit_overlay"] = {
        "rule": FIRST_TRIGGER_RULE,
        "tie_break": "CANONICAL_DEFAULT_EXIT",
        "legacy_state_action": "BID1_MARKET",
        "canonical_profile_id": K.PROFILE_ID,
        "canonical_profile_sha256": K.profile_hash(),
    }
    return program


def export_entry_only(package_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """중복 artifact를 content_hash 하나로 합쳐 entry-only corpus를 저장한다."""
    unique: dict[str, Path] = {}
    for candidate in sorted(map(Path, package_paths)):
        package = read_json(candidate)
        content_hash = str(package.get("content_hash") or "")
        if not content_hash:
            raise ValueError(f"content_hash가 없는 과거 package: {candidate}")
        unique.setdefault(content_hash, candidate)
    programs = [compile_package(path, entry_only=True)
                for _, path in sorted(unique.items())]
    ids = [str(program["legacy_content_hash"]) for program in programs]
    if len(ids) != len(set(ids)):
        raise ValueError("entry-only corpus에 중복 content_hash가 있다")
    payload = {
        "schema": ENTRY_ONLY_SCHEMA_VERSION,
        "created_at": now_utc(),
        "source": {
            "artifact_files_seen": int(len(package_paths)),
            "unique_content_hashes": int(len(programs)),
            "selection": "iterations/**/algorithm_package.final.json",
        },
        "current_backtest_profile": {
            "canonical_profile_id": K.PROFILE_ID,
            "canonical_profile_sha256": K.profile_hash(),
            "entry_threshold_policy": "PRIOR_VALID_DAY_PER_SYMBOL_QUANTILES",
        },
        "entry_feature_adapter": {
            "module": "framework.legacy.feature_values",
            "module_sha256": sha256_file(Path(__file__)),
            "supported_feature_count": len(SUPPORTED_ENTRY_FEATURES),
        },
        "entry_feature_names": sorted({
            _bare_feature_name(condition["feature"])
            for program in programs for condition in program["entry"]["conditions"]
        }),
        "programs": programs,
    }
    write_json(Path(output), payload)
    return {"path": str(output), "programs": len(programs),
            "features": len(payload["entry_feature_names"])}


def load_entry_only(path: Path) -> list[dict[str, Any]]:
    """저장된 corpus만 받아 현재 청산 규칙으로 재생할 때 쓴다."""
    payload = read_json(Path(path))
    if payload.get("schema") != ENTRY_ONLY_SCHEMA_VERSION:
        raise ValueError(f"entry-only corpus schema가 아니다: {path}")
    programs = list(payload.get("programs") or ())
    ids = [str(program.get("legacy_content_hash") or "") for program in programs]
    if not programs or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("entry-only corpus의 content_hash가 비었거나 중복이다")
    for program in programs:
        if not program.get("entry_only") or "exit" in program or "exit_overlay" in program:
            raise ValueError("entry-only corpus에 과거 청산 구조가 남아 있다")
        for condition in program.get("entry", {}).get("conditions", ()):
            if _bare_feature_name(str(condition["feature"])) not in SUPPORTED_ENTRY_FEATURES:
                raise ValueError(f"현재 재생기에 없는 entry feature: {condition['feature']}")
    return programs


def bind_prior_day_thresholds(program: Mapping[str, Any], *, symbol: str, date: str,
                              reference_dates: Sequence[str], root: Path = TICK_ROOT
                              ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """실행일 전일의 같은 종목 분포로 entry·exit 컷을 채운다."""
    prior = tickdata.prior_valid_day(str(symbol), str(date), list(reference_dates), root)
    while prior is not None:
        try:
            arrays, _ = tickdata.load(str(symbol), prior, root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData):
            prior = tickdata.prior_valid_day(str(symbol), prior, list(reference_dates), root)
            continue
        bound = dict(program)
        detail: dict[str, Any] = {"policy": "PRIOR_VALID_DAY_PER_SYMBOL_QUANTILES",
                                  "reference_date": str(prior), "conditions": {}}
        for role in ("entry", "exit"):
            role_body = dict(program.get(role, {}))
            conditions: list[dict[str, Any]] = []
            for condition in role_body.get("conditions", ()):
                value = feature_values(arrays, str(condition["feature"]))
                finite = np.asarray(value, dtype=float)
                finite = finite[np.isfinite(finite)]
                if len(finite) < MIN_REFERENCE_SAMPLES:
                    return None
                item = {**dict(condition), "threshold": float(np.quantile(
                    finite, float(condition["quantile"]))),
                        "reference_date": str(prior),
                        "reference_observations": int(len(finite))}
                conditions.append(item)
                detail["conditions"][f"{role}:{len(conditions) - 1}"] = {
                    "source_feature": item["source_feature"], "quantile": item["quantile"],
                    "threshold": item["threshold"],
                    "reference_observations": item["reference_observations"],
                }
            bound[role] = {**role_body, "conditions": conditions}
        return bound, detail
    return None


def _condition_mask(arrays: Mapping[str, np.ndarray], conditions: Sequence[Mapping[str, Any]],
                    *, exit_offset: int | None = None) -> np.ndarray:
    n = len(np.asarray(arrays["time_s"]))
    mask = np.ones(n, dtype=bool)
    for condition in conditions:
        values = feature_values(arrays, str(condition["feature"]))
        if exit_offset:
            shifted = np.full(n, np.nan)
            if n > int(exit_offset):
                shifted[int(exit_offset):] = values[:-int(exit_offset)]
            values = shifted
        threshold = float(condition["threshold"])
        current = values < threshold if condition["comparator"] == "<" else values > threshold
        mask &= np.isfinite(values) & current
    return mask


def entry_signal(arrays: Mapping[str, np.ndarray], program: Mapping[str, Any]) -> np.ndarray:
    return _condition_mask(arrays, program.get("entry", {}).get("conditions", ()))


def first_exit_trigger(arrays: Mapping[str, np.ndarray], program: Mapping[str, Any],
                       entry_tick: int, cap_tick: int) -> int | None:
    """legacy 상태 조건이 처음 참이 되는 tick. 빈 rule은 청산 신호가 없다."""
    conditions = program.get("exit", {}).get("conditions", ())
    if not conditions:
        return None
    # `exit:`은 현재 tick, `pre_exit20t:`·`pre_exit100t:`는 해당 tick의 과거값이다.
    masks = []
    for condition in conditions:
        source = str(condition["source_feature"])
        offset = 20 if source.startswith("pre_exit20t:") else (100 if source.startswith("pre_exit100t:") else 0)
        masks.append(_condition_mask(arrays, (condition,), exit_offset=offset))
    fired = np.column_stack(masks).all(axis=1)
    where = np.flatnonzero(fired[max(int(entry_tick), 0): min(int(cap_tick), len(fired) - 1) + 1])
    return (max(int(entry_tick), 0) + int(where[0])) if len(where) else None


def _source_oos(path: Path) -> dict[str, Any] | None:
    evaluation_path = Path(path).with_name("algorithm_package.oos_evaluation.json")
    if not evaluation_path.exists():
        return None
    payload = read_json(evaluation_path)
    return {"dates": payload.get("oos_dates"), "summary": payload.get("summary")}


def run_programs(programs: Sequence[Mapping[str, Any]], *, dates: Sequence[str], output: Path,
                 root: Path = TICK_ROOT, workers: int = 8) -> dict[str, Any]:
    """이미 컴파일된 프로그램을 같은 현재 정본 Backtest에 연결한다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    programs = [dict(program) for program in programs]
    ids = [f"LEGACY:{program['legacy_content_hash'] or program['legacy_package_id']}" for program in programs]
    if len(ids) != len(set(ids)):
        raise ValueError("같은 과거 package가 중복으로 들어왔다")
    contracts = {contract_id: {"legacy_program": program}
                 for contract_id, program in zip(ids, programs)}
    members = {contract_id: program["symbols"] for contract_id, program in zip(ids, programs)}
    ledger_path = output / "ledger.parquet"
    manifest = K.run_backtest(contracts, members, list(map(str, dates)), ledger_path,
                              reference_dates=tickdata.available_dates(root),
                              workers=workers, root=root)
    frame = pd.read_parquet(ledger_path)
    accounting = K.spread_accounting_audit(frame, manifest)
    from . import ledger
    ledger.verify_invariants(frame, manifest)
    metrics = K.metric_report(frame, manifest)
    filled = frame.loc[frame["status"].eq("FILLED")]
    exit_reasons = (Counter(filled["exit_reason"]) if "exit_reason" in filled else Counter())
    entry_only = bool(programs) and all(bool(program.get("entry_only")) for program in programs)
    result = {
        "schema": SCHEMA_VERSION,
        "created_at": now_utc(),
        "state": "ENTRY_ONLY_REPLAY_COMPLETE" if entry_only else "LEGACY_REPLAY_COMPLETE",
        "dates": list(map(str, dates)),
        "canonical_profile_id": K.PROFILE_ID,
        "canonical_profile_sha256": K.profile_hash(),
        "exit_overlay": "CANONICAL_DEFAULT_ONLY" if entry_only else FIRST_TRIGGER_RULE,
        "accounting": accounting,
        "metrics": metrics,
        "exit_reasons": {str(key): int(value) for key, value in exit_reasons.items()},
        "packages": [
            {"contract_id": contract_id, "program": program,
             "historical_oos": _source_oos(Path(program["legacy_package_path"]))}
            for contract_id, program in zip(ids, programs)
        ],
        "artifacts": {"ledger": str(ledger_path),
                      "ledger_manifest": str(ledger_path.with_name("ledger_manifest.json"))},
        "note": ("과거 수익 기록은 source artifact다. 조건의 분위수는 각 실행일의 "
                 "직전 유효일·같은 종목 분포로 계산한 현재 canonical profile 재생이다."),
    }
    write_json(output / "legacy_replay_summary.json", result)
    return result


def run(package_paths: Sequence[Path], *, dates: Sequence[str], output: Path,
        root: Path = TICK_ROOT, workers: int = 8, entry_only: bool = False) -> dict[str, Any]:
    """과거 package를 읽어 현재 정본 Backtest에 연결한다."""
    programs = [compile_package(path, root=root, entry_only=entry_only) for path in package_paths]
    return run_programs(programs, dates=dates, output=output, root=root, workers=workers)


def _entry_signature(program: Mapping[str, Any]) -> dict[str, Any]:
    """청산이 없는 현재 재생에서 결과를 공유할 수 있는 entry 식별자."""
    conditions = [
        {"feature": _bare_feature_name(str(condition["feature"])),
         "comparator": str(condition["comparator"]),
         "quantile": float(condition["quantile"])}
        for condition in program.get("entry", {}).get("conditions", ())
    ]
    return {"threshold_policy": program.get("entry", {}).get("threshold_policy"),
            "conditions": sorted(conditions,
                                 key=lambda item: (item["feature"], item["comparator"],
                                                   item["quantile"]))}


def _all_symbols(root: Path) -> list[str]:
    return sorted({
        path.name.split("_", 1)[0]
        for date in tickdata.available_dates(root)
        for path in (Path(root) / date).glob("*.parquet")
    })


def run_entry_only_universe(corpus: Path, *, dates: Sequence[str], output: Path,
                            root: Path = TICK_ROOT, workers: int = 8) -> dict[str, Any]:
    """entry-only corpus를 데이터 전체 종목에 재생하고, entry별 상세 ledger를 남긴다.

    같은 entry는 현재 threshold·실행·청산도 같으므로 한 번만 재생하고, source content_hash
    는 root manifest에서 그 ledger와 연결한다. 각 entry ledger를 독립 parquet으로 남겨
    전체 935개를 한 DataFrame으로 올리지 않는다.
    """
    root = Path(root)
    output = Path(output)
    programs = load_entry_only(corpus)
    symbols = _all_symbols(root)
    if not symbols:
        raise ValueError(f"전종목 universe가 비어 있다: {root}")
    target_dates = tuple(map(str, dates))
    available = set(tickdata.available_dates(root))
    missing_dates = sorted(set(target_dates) - available)
    if missing_dates:
        raise ValueError(f"데이터에 없는 재생일: {missing_dates}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for program in programs:
        signature = _entry_signature(program)
        rule_id = f"ENTRY:{sha256_json(signature)[:16]}"
        grouped.setdefault(rule_id, []).append(program)
        signatures[rule_id] = signature

    output.mkdir(parents=True, exist_ok=True)
    rules_dir = output / "rules"
    rules_dir.mkdir(exist_ok=True)
    program_to_rule = [
        {"legacy_content_hash": str(program["legacy_content_hash"]),
         "rule_id": rule_id}
        for rule_id, members in sorted(grouped.items()) for program in members
    ]
    run_manifest = {
        "schema": "legacy_entry_only_universe_replay.v1",
        "created_at": now_utc(),
        "state": "RUNNING",
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "dates": list(target_dates),
        "universe": {"symbols": len(symbols), "symbols_file": "universe_symbols.json"},
        "programs": len(programs),
        "unique_entries": len(grouped),
        "canonical_profile_id": K.PROFILE_ID,
        "canonical_profile_sha256": K.profile_hash(),
        "exit_overlay": "CANONICAL_DEFAULT_ONLY",
        "program_to_rule": program_to_rule,
        "rules": {rule_id: {"entry": signatures[rule_id],
                            "content_hashes": [str(p["legacy_content_hash"]) for p in members],
                            "ledger": str(rules_dir / rule_id.replace(":", "_") / "ledger.parquet")}
                  for rule_id, members in sorted(grouped.items())},
    }
    write_json(output / "universe_symbols.json", symbols)
    write_json(output / "universe_replay_manifest.json", run_manifest)

    completed = 0
    for ordinal, (rule_id, members) in enumerate(sorted(grouped.items()), start=1):
        rule_dir = rules_dir / rule_id.replace(":", "_")
        summary_path = rule_dir / "legacy_replay_summary.json"
        if summary_path.exists():
            previous = read_json(summary_path)
            if previous.get("state") == "ENTRY_ONLY_REPLAY_COMPLETE":
                completed += 1
                continue
        representative = dict(members[0])
        representative["symbols"] = list(symbols)
        result = run_programs([representative], dates=target_dates, output=rule_dir,
                              root=root, workers=workers)
        if result.get("state") != "ENTRY_ONLY_REPLAY_COMPLETE":
            raise RuntimeError(f"전종목 재생이 완료되지 않았다: {rule_id}")
        completed += 1
        write_json(output / "progress.json", {
            "schema": "legacy_entry_only_universe_progress.v1",
            "updated_at": now_utc(),
            "state": "RUNNING",
            "completed_unique_entries": completed,
            "total_unique_entries": len(grouped),
            "last_completed_rule": rule_id,
        })

    result = {**run_manifest, "state": "ENTRY_ONLY_UNIVERSE_REPLAY_COMPLETE",
              "completed_unique_entries": completed, "completed_at": now_utc()}
    write_json(output / "universe_replay_manifest.json", result)
    write_json(output / "progress.json", {
        "schema": "legacy_entry_only_universe_progress.v1",
        "updated_at": now_utc(),
        "state": "COMPLETE",
        "completed_unique_entries": completed,
        "total_unique_entries": len(grouped),
    })
    return result
