"""고정된 가설 안에서 **하나 남은 값**을 정한다.

묻는 것은 이것뿐이다.

    각 종목에서 어느 정도로 높은 상태부터 실제 신호로 취급할 것인가?

## 왜 원값이 아니라 분위인가

    book_imbalance > 0.4

는 종목마다 다른 뜻이다. 어떤 종목에선 극단값이고 어떤 종목에선 평범하다. 같은 숫자를
모두에게 적용하면 상태가 아니라 **종목 크기를 거르는 필터**가 된다.

    book_imbalance > 직전 100 tick q90

은 모든 종목에서 "그 종목의 방금 전 100개 관측 기준 상위 10%" 라는 같은 뜻을 갖는다.

## 분위로 바꾼다고 과적합이 사라지지 않는다

분위가 하는 일은 **종목 간 눈금 맞추기**지 과적합 방지가 아니다. 과적합은 따로 막는다 —
성긴 격자, 확대 금지, 새 데이터 확인, 손대지 않은 마지막 구간.

## 이 단계가 답하지 않는 것

    이 계약이 실제로 돈을 버는가?

청산 규칙이 아직 없다. 그래서 고른 값을 **수익 나는 임계** 라고 부르지 않는다.
`SELECTED_SIGNAL_CUTOFF` 다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import (capability as C, catalog, contract, data as tickdata, executable as X,
               implementation as I, validation as V)
from .config import TICK_ROOT, now_utc, read_json, sha256_json, write_json
# 임계 보정은 calibration.py 가 갖는다. 이 모듈의 나머지 코드가 그대로 쓴다.
from .calibration import (PARAMETER_LOCK_STATUSES, PARAMETER_LOCKED,  # noqa: F401
                         PARAMETER_LOCKED_SINGLE_DAY, SearchConfig, _state_config,
                         calibrate, calibration_table, config_for,
                         parameter_values_for_table, previous_real_trading_day,
                         threshold_column, threshold_source, trading_calendar,
                         amend_specification)


SCHEMA_VERSION = "parameter_search.v1"

# 진행 상태 (§62).
READY_TO_SEARCH = "READY_TO_SEARCH"
SEARCH_SELECTED = "SEARCH_SELECTED"
PARAMETER_CONFIRMED = "PARAMETER_CONFIRMED"
PARAMETER_CONFIRMATION_FAILED = "PARAMETER_CONFIRMATION_FAILED"
SEARCH_SPEC_CHANGE_REQUIRED = "SEARCH_SPEC_CHANGE_REQUIRED"

# 후보 판정.
SEARCH_SUPPORTED = "SEARCH_SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"
INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
MATCHING_FAILED = "MATCHING_FAILED"

# 날짜를 무엇에 쓰는가 (§58).
HYPOTHESIS_DEVELOPMENT = "HYPOTHESIS_DEVELOPMENT"
SEARCH_FIT = "SEARCH_FIT"
SEARCH_CONFIRM = "SEARCH_CONFIRM"
TERMINAL_OOS = "TERMINAL_OOS"

# 탐색 대상이 되면 안 되는 것들. 좋아 보여도 이 가설의 결과가 아니다 (§53).
SEARCH_DISCOVERY_CANDIDATE = X.SEARCH_DISCOVERY_CANDIDATE

QUANTILE_THRESHOLD_KINDS = frozenset({
    X.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
    X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
})


def is_quantile_threshold_kind(kind: str | None) -> bool:
    return str(kind) in QUANTILE_THRESHOLD_KINDS









# ---- 1. 종목별 직전 100 tick 분위 (§6~§13) --------------------------------------











# ---- 2. 신호와 짝짓기 (§29~§32) -------------------------------------------------

def _rolling_thresholds_for_frame(frame: pd.DataFrame, quantile: float,
                                  config: SearchConfig, root: Path,
                                  anchor_cache_root: Path | None = None) -> dict[str, np.ndarray]:
    """anchor의 tick 번호에 맞춘 직전 100 tick cut.

    anchor 표에는 일부 tick만 있으므로, 각 종목-일 원본에서 전체 tick cut을 만든 뒤
    해당 anchor 위치만 꺼낸다. 이때 현재 anchor나 미래 tick은 cut에 들어가지 않는다.
    """
    output = {f"{catalog.UNRESOLVED_PREFIX}{parameter}": np.full(len(frame), np.nan)
              for _feature, _direction, parameter in config.states}
    cached_states: set[tuple[str, str, str]] = set()
    if anchor_cache_root is not None:
        from .modules import search_anchor_cache
        for feature, direction, parameter in config.states:
            column = search_anchor_cache.threshold_column(feature, direction, quantile)
            if column in frame:
                output[f"{catalog.UNRESOLVED_PREFIX}{parameter}"] = frame[column].to_numpy(float)
                cached_states.add((feature, direction, parameter))
    for (symbol, date), positions in frame.groupby(["symbol", "date"], sort=False).indices.items():
        try:
            arrays, _ = tickdata.load(str(symbol), str(date), root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData):
            continue
        book = catalog.Book(arrays)
        ticks = frame.iloc[positions]["tick"].to_numpy(int)
        valid_ticks = (ticks >= 0) & (ticks < book.n)
        for feature, direction, parameter in config.states:
            if (feature, direction, parameter) in cached_states:
                continue
            values = catalog.compute_features((feature,), book).get(feature)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            lower = str(direction).upper() == "LOWER"
            q = 1.0 - float(quantile) if lower else float(quantile)
            cuts = contract.prior_tick_quantile(
                values, q, method="lower" if lower else config.quantile_method)
            output[f"{catalog.UNRESOLVED_PREFIX}{parameter}"][positions[valid_ticks]] = cuts[ticks[valid_ticks]]
    return output


def attach_signal(frame: pd.DataFrame, table: pd.DataFrame, quantile: float,
                  config: SearchConfig, *, root: Path = TICK_ROOT,
                  anchor_cache_root: Path | None = None) -> pd.DataFrame:
    """anchor 마다 그 종목·tick의 임계를 붙이고 신호를 켠다.

    rolling source는 같은 종목·날짜의 현재 tick 이전 100개만 쓴다. 적격하지 않은
    종목-일은 통째로 빠진다. 후보마다 같은 우주를 쓴다.
    """
    if frame.empty:
        empty = frame.copy()
        empty["signal"] = pd.Series(False, index=empty.index, dtype=bool)
        empty["threshold"] = np.nan
        return empty
    if config.is_rolling_quantile:
        keys = table.loc[table["eligible"].astype(bool), ["symbol", "date", "reference_date"]]
        merged = frame.merge(keys, on=["symbol", "date"], how="inner")
        thresholds = _rolling_thresholds_for_frame(
            merged, quantile, config, root, anchor_cache_root=anchor_cache_root)
        if len(config.states) > 1:
            merged["signal"] = _frame_expression(config.expression or {}, merged, thresholds)
            merged["threshold"] = np.nan
            return merged
        cut = thresholds[f"{catalog.UNRESOLVED_PREFIX}{config.parameter}"]
        merged["threshold"] = cut
        values = merged[config.feature].to_numpy(float)
        merged["signal"] = ((values < cut) if config.lower else (values > cut))
        return merged
    if len(config.states) > 1:
        columns = [threshold_column(quantile, config, parameter)
                   for _feature, _direction, parameter in config.states]
        keys = table.loc[table["eligible"].astype(bool),
                         ["symbol", "date", "reference_date", *columns]]
        merged = frame.merge(keys, on=["symbol", "date"], how="inner")
        thresholds = {f"{catalog.UNRESOLVED_PREFIX}{parameter}":
                      merged[threshold_column(quantile, config, parameter)].to_numpy(float)
                      for _feature, _direction, parameter in config.states}
        merged["signal"] = _frame_expression(config.expression or {}, merged, thresholds)
        merged["threshold"] = np.nan
        return merged
    column = threshold_column(quantile, config)
    keys = table.loc[table["eligible"], ["symbol", "date", column, "reference_date"]]
    merged = frame.merge(keys, on=["symbol", "date"], how="inner")
    merged = merged.rename(columns={column: "threshold"})
    values = merged[config.feature].to_numpy(float)
    cut = merged["threshold"].to_numpy(float)
    merged["signal"] = (values < cut) if config.lower else (values > cut)
    return merged


def _frame_expression(node: Any, frame: pd.DataFrame,
                      thresholds: Mapping[str, np.ndarray]) -> np.ndarray:
    """Search용 Catalog DSL 평가기. 값은 anchor frame, 임계는 전일 보정 표에서만 온다."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return np.full(len(frame), float(node))
    if not isinstance(node, Mapping):
        raise ValueError("joint Search expression이 Catalog DSL object가 아니다")
    op = str(node.get("op") or "")
    if op == "primitive":
        name = str(node.get("primitive_id") or "")
        if name not in frame:
            raise ValueError(f"joint Search anchor frame에 feature가 없다: {name}")
        return frame[name].to_numpy(float)
    if op in {"all", "any"}:
        values = [_frame_expression(item, frame, thresholds) for item in node.get("args") or []]
        if not values:
            raise ValueError(f"{op}에는 args가 필요하다")
        return np.logical_and.reduce(values) if op == "all" else np.logical_or.reduce(values)
    if op in {"difference", "ratio"}:
        left = _frame_expression(node.get("left"), frame, thresholds)
        right = _frame_expression(node.get("right"), frame, thresholds)
        return left - right if op == "difference" else np.divide(left, right,
            out=np.full(len(frame), np.nan), where=np.asarray(right) != 0)
    if op == "compare":
        left = _frame_expression(node.get("input"), frame, thresholds)
        raw = node.get("value")
        right = thresholds.get(raw) if isinstance(raw, str) else raw
        if right is None:
            raise ValueError(f"joint Search가 알 수 없는 threshold를 만났다: {raw}")
        comparator = str(node.get("comparator") or "")
        return {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}.get(
            comparator, lambda *_: (_ for _ in ()).throw(ValueError(f"비교 연산자가 유효하지 않다: {comparator}")))(left, right)
    raise ValueError(f"joint Search는 아직 `{op}` 연산자의 anchor 평가를 지원하지 않는다")


def matched_by_signal(frame: pd.DataFrame, controls: Sequence[str],
                      config: SearchConfig) -> pd.DataFrame:
    """신호가 켜진 anchor 와 안 켜진 anchor 를 층 안에서 1:1 로 짝짓는다.

    검증에서 쓴 것과 같은 규칙이다 — 같은 층, 순위 거리, 복원 없음, caliper 없음.
    결과가 좋아지도록 통제 변수나 거리를 바꾸지 않는다.
    """
    parts = []
    for (symbol, date, bucket), group in frame.groupby(["symbol", "date", "bucket"],
                                                       sort=True):
        high = group[group["signal"]]
        low = group[~group["signal"]]
        if high.empty or low.empty:
            continue
        ranked = group.copy()
        for name in [*controls, config.feature]:
            ranked[f"_r_{name}"] = ranked[name].rank(pct=True)
        high = ranked.loc[high.index]
        low = ranked.loc[low.index]
        cols = [f"_r_{c}" for c in controls]
        low_matrix = low[cols].to_numpy(float)
        high_matrix = high[cols].to_numpy(float)
        distances = np.abs(high_matrix[:, None, :] - low_matrix[None, :, :]).sum(axis=2)
        orders = np.argsort(distances, axis=1, kind="stable")
        used = np.zeros(len(low), dtype=bool)
        selected_high, selected_low = [], []
        for high_position, order in enumerate(orders):
            available = order[~used[order]]
            choice = int(available[0]) if len(available) else None
            if choice is None:
                break
            used[choice] = True
            selected_high.append(high_position)
            selected_low.append(choice)
        if not selected_high:
            continue
        high_selected = high.iloc[selected_high].reset_index(drop=True)
        low_selected = low.iloc[selected_low].reset_index(drop=True)
        part = pd.DataFrame({
            "pair_id": [f"{symbol}:{date}:{bucket}:{int(tick)}"
                        for tick in high_selected["tick"]],
            "symbol": symbol, "date": date, "bucket": bucket,
            "control_distance": distances[np.asarray(selected_high), np.asarray(selected_low)],
            # 검증과 같은 값을 쓴다 — 신호 유무를 1.0 대비로 두면 짝짓기
            # 품질 기준이 그만큼 헐거워진다.
            "tested_rank_delta": (high_selected[f"_r_{config.feature}"].to_numpy(float)
                                  - low_selected[f"_r_{config.feature}"].to_numpy(float)),
            "future_path_delta": (high_selected[config.response].to_numpy(float)
                                  - low_selected[config.response].to_numpy(float)),
            "success_delta": (high_selected["sustained_success"].to_numpy(int)
                              - low_selected["sustained_success"].to_numpy(int)),
            **{f"{name}_delta": (high_selected[name].to_numpy(float)
                                  - low_selected[name].to_numpy(float))
               for name in controls},
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---- 3. 후보 하나 채점 (§33~§35) ------------------------------------------------

def score_candidate(frame: pd.DataFrame, table: pd.DataFrame, quantile: float,
                    config: SearchConfig, controls: Sequence[str], *,
                    root: Path = TICK_ROOT,
                    anchor_cache_root: Path | None = None) -> dict[str, Any]:
    """후보 값 하나. 판정에 쓰는 통계는 하나지만 진단은 함께 낸다."""
    attached = attach_signal(frame, table, quantile, config, root=root,
                             anchor_cache_root=anchor_cache_root)
    signal = attached["signal"].to_numpy(bool)
    row: dict[str, Any] = {
        "q": quantile,
        "value": quantile,
        "threshold_source": threshold_source(config, quantile),
        "anchors": int(len(attached)),
        "signal_decisions": int(signal.sum()),
        "signal_rate": float(signal.mean()) if len(attached) else float("nan"),
        "eligible_symbol_days": int(table["eligible"].sum()),
    }
    if not signal.any() or signal.all():
        row.update({"status": INSUFFICIENT_SUPPORT, "n_matched_pairs": 0,
                    "reason": "신호가 전부 켜졌거나 전부 꺼져 대조군이 없다"})
        return row

    pairs = matched_by_signal(attached, controls, config)
    diagnostics = V.matching_diagnostics(pairs, len(attached), len(controls))
    row.update({"n_matched_pairs": int(len(pairs)),
                "matching_balance_ratio": diagnostics.get("balance_ratio"),
                "matching_quality_ok": diagnostics.get("quality_ok")})
    if len(pairs) < config.min_pairs:
        row.update({"status": INSUFFICIENT_SUPPORT,
                    "reason": f"짝이 {len(pairs)}개로 사전에 정한 {config.min_pairs} 미만"})
        return row
    if not diagnostics["quality_ok"]:
        row.update({"status": MATCHING_FAILED,
                    "reason": "짝짓기 품질이 사전 기준을 못 넘었다. 더 느슨한 짝짓기를 "
                              "찾지 않는다"})
        return row

    delta = pairs["future_path_delta"].to_numpy(float)
    sign = V._sign_test(delta, V.ValidationConfig(bootstrap=config.bootstrap,
                                                  seed=config.seed))
    status = V._supported(sign["share_positive"], sign["interval_low"],
                          sign["interval_high"], config.null_value)
    row.update({
        "effect_metric": config.primary_metric,
        "effect_size": sign["share_positive"],
        "interval_low": sign["interval_low"],
        "interval_high": sign["interval_high"],
        "null_value": config.null_value,
        "n_effective": sign["n_effective"],
        "tie_rate": sign["tie_rate"],
        "status": SEARCH_SUPPORTED if status == "SUPPORTED" else
                  NOT_SUPPORTED if status == "NOT_SUPPORTED" else INCONCLUSIVE,
        # 아래는 **진단**이다. 이걸 보고 q 를 고르지 않는다 (§35).
        "diag_mean_max_net_bps": float(attached.loc[signal, config.response].mean()),
        "diag_median_max_net_bps": float(attached.loc[signal, config.response].median()),
        "diag_sustained_success_rate": float(
            attached.loc[signal, "sustained_success"].mean()),
    })
    row["date_direction_agreement"] = _agreement(pairs, "date")
    row["cluster_direction_agreement"] = _agreement(pairs, "cluster")
    return row


def _agreement(pairs: pd.DataFrame, key: str) -> float | None:
    """방향이 한 날짜·한 클러스터에만 기대고 있지 않은가 (§34)."""
    if key == "cluster":
        from . import universe
        pairs = pairs.assign(cluster=[universe.cluster_of(s) for s in pairs["symbol"]])
    if key not in pairs:
        return None
    shares = []
    for _, group in pairs.groupby(key):
        delta = group["future_path_delta"].to_numpy(float)
        delta = delta[delta != 0]
        if len(delta) >= 20:
            shares.append(float((delta > 0).mean()))
    return float(np.mean([s > 0.5 for s in shares])) if shares else None


# ---- 4. 고르는 규칙 (§37~§39) ---------------------------------------------------

SELECTION_RULE = {
    "step_1": "효과를 계산할 수 있는 후보만 (짝 수 바닥·짝짓기 품질 통과)",
    "step_2": "직접 효과 판정은 후보를 버리는 조건이 아니라 기록이다",
    "step_3": "MATCHED_FUTURE_PATH_ADVANTAGE 가 가장 큰 것",
    "tie_break_1": "동률이면 동률 제외 유효 표본이 큰 쪽",
    "tie_break_2": "그래도 동률이면 더 낮은 threshold 값",
    "no_effect_estimate": "효과를 계산할 후보가 없으면 q=0.50에 가장 가까운 값을 쓴다. "
                          "직접 효과 판정은 기록만 남기고 Validation으로 보낸다",
}


def select(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """q 하나를 고정한다. 직접 효과 판정은 후보 탈락 사유가 아니다."""
    scored = [c for c in candidates
              if c.get("effect_size") is not None and np.isfinite(c["effect_size"])]
    if not scored:
        fallback = sorted(candidates,
                          key=lambda c: (abs(float(c["q"]) - 0.50), float(c["q"])))
        winner = fallback[0] if fallback else None
        value = float(winner["q"]) if winner else None
        return {"status": SEARCH_SELECTED, "selected_q": value,
                "selected_value": value, "rule": SELECTION_RULE,
                "direct_evidence_status": winner.get("status") if winner else None,
                "reason": ("직접 효과를 계산할 수 없어 중립 q에 가장 가까운 값을 고정했다"
                           if winner else "후보 격자가 비어 있다"),
                "considered": len(candidates), "runner_up": [c["q"] for c in fallback[1:]]}
    ranked = sorted(scored,
                    key=lambda c: (-float(c["effect_size"]),
                                   -int(c.get("n_effective") or 0),
                                   float(c["q"])))
    winner = ranked[0]
    value = float(winner["q"])
    return {"status": SEARCH_SELECTED, "selected_q": value, "selected_value": value,
            "rule": SELECTION_RULE,
            "effect_size": winner["effect_size"],
            "interval": [winner["interval_low"], winner["interval_high"]],
            "n_effective": winner.get("n_effective"),
            "direct_evidence_status": winner.get("status"),
            "reason": f"효과를 계산한 {len(scored)}개 중 짝 우위가 가장 크다",
            "considered": len(candidates),
            "runner_up": [c["q"] for c in ranked[1:]]}


# ---- 5. 새 데이터에서 확인 (§40~§44) --------------------------------------------

def confirm(frame: pd.DataFrame, table: pd.DataFrame, quantile: float,
            config: SearchConfig, controls: Sequence[str], *,
            root: Path = TICK_ROOT,
            anchor_cache_root: Path | None = None) -> dict[str, Any]:
    """고른 threshold 값 **하나만** 연다. 차점자도 격자 확장도 없다."""
    result = score_candidate(frame, table, quantile, config, controls, root=root,
                             anchor_cache_root=anchor_cache_root)
    confirmed = result.get("status") == SEARCH_SUPPORTED
    return {"schema": "parameter_confirmation.v1", "created_at": now_utc(),
            "q": quantile, "value": quantile,
            "threshold_source": threshold_source(config, quantile),
            "status": PARAMETER_CONFIRMED if confirmed else PARAMETER_CONFIRMATION_FAILED,
            "result": result,
            "only_one_candidate_opened": True,
            "why": ("같은 통계·같은 귀무값에서 방향이 재현됐다" if confirmed else
                    "재현되지 않았다"),
            "on_failure": "차점자 확인·격자 확장·다른 threshold source 시도를 하지 않는다. "
                          "이 데이터를 이미 봤기 때문이다",
            "next": ("파라미터를 잠근다" if confirmed else
                     "직접 증거는 미확인으로 남기고, 같은 고정 q를 Validation에서 본다")}


# ---- 6. 감사 (§60~§61) ----------------------------------------------------------

MUST_BE_ZERO = (
    "new_feature_count", "new_relation_count", "new_mechanism_count",
    "raw_threshold_search_count", "search_dimension_change_count",
    "current_day_calibration_leakage_count", "future_calibration_leakage_count",
    "zoom_in_search_count", "post_result_grid_change_count",
    "matching_rule_change_count", "metric_change_count",
    "terminal_oos_access_count",
)
WARNING_ONLY = (
    "calibration_insufficient_symbol_day_count", "threshold_tie_count",
    "low_signal_support_candidate_count", "high_signal_concentration_warning_count",
)


def search_audit(plan: Mapping[str, Any], table: pd.DataFrame,
                 candidates: Sequence[Mapping[str, Any]],
                 fidelity: Mapping[str, Any], opened: Sequence[str],
                 config: SearchConfig) -> dict[str, Any]:
    """탐색이 경계를 넘지 않았는가."""
    terminal = set(plan["blocks"][TERMINAL_OOS])
    leaked = sorted(set(opened) & terminal)
    # literal은 과거 데이터를 읽지 않으므로 날짜 보정 누출이 없다.
    same_day = (int((table["reference_date"] >= table["date"]).sum())
                if config.is_quantile and not config.is_rolling_quantile and len(table) else 0)
    planned_source = ((plan.get("threshold_source") or {}).get("kind")
                      or X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE)
    metrics = {
        "new_feature_count": int(fidelity.get("new_feature_count", 0)),
        "new_relation_count": int(fidelity.get("new_relation_count", 0)),
        "new_mechanism_count": int(fidelity.get("new_mechanism_count", 0)),
        "raw_threshold_search_count": int(planned_source != config.threshold_kind),
        "search_dimension_change_count": 0 if (
            int(plan.get("search_dimensions", 1)) == 1
            and list(plan.get("grid_parameter") or []) ==
                [parameter for _feature, _direction, parameter in config.states]
        ) else 1,
        "current_day_calibration_leakage_count": same_day,
        "future_calibration_leakage_count": same_day,
        "zoom_in_search_count": 0,
        "post_result_grid_change_count":
            0 if list(plan["grid"]) == list(config.grid) else 1,
        "matching_rule_change_count":
            0 if plan["matching_contract"] == V.MATCHING_CONTRACT else 1,
        "metric_change_count":
            0 if plan["primary_metric"] == config.primary_metric else 1,
        "terminal_oos_access_count": len(leaked),
        "calibration_insufficient_symbol_day_count":
            int((~table["eligible"]).sum()) if len(table) else 0,
        "threshold_tie_count": 0,
        "low_signal_support_candidate_count":
            sum(1 for c in candidates if c.get("status") == INSUFFICIENT_SUPPORT),
        "high_signal_concentration_warning_count":
            sum(1 for c in candidates
                if (c.get("signal_rate") or 0) > 0.5 or (c.get("signal_rate") or 1) < 0.01),
    }
    failed = {k: v for k, v in metrics.items() if k in MUST_BE_ZERO and v}
    return {"schema": "parameter_search_audit.v1", "created_at": now_utc(),
            "metrics": dict(sorted(metrics.items())),
            "must_be_zero": list(MUST_BE_ZERO), "warning_only": list(WARNING_ONLY),
            "failed": failed, "ok": not failed,
            "terminal_block_leaked": leaked,
            "note": "경고성 지표는 0 일 필요가 없다. 데이터 특성일 수 있다"}


# ---- 7. 계약 재컴파일 (§19~§22) -------------------------------------------------



# ---- 8. 계획 (§70) --------------------------------------------------------------

def plan(spec: Mapping[str, Any], blocks: Mapping[str, Sequence[str]],
         symbols: Sequence[str], config: SearchConfig = SearchConfig()) -> dict[str, Any]:
    """실행 전에 얼린다. 이 파일이 없으면 `run` 이 거부한다."""
    calibration = (
        {"scope": config.calibration_scope,
         "reference": config.calibration_reference,
         "quantile_method": config.quantile_method,
         "minimum_observations": config.min_calibration_obs,
         "minimum_observations_derivation": (
             f"현재 tick 제외 직전 {contract.ROLLING_QUANTILE_WINDOW} tick 고정"
             if config.is_rolling_quantile else
             f"ceil(10 / (1 - {config.max_quantile})) = {config.derived_min_obs()}"),
         "session": V.session_policy()["policy"],
         "no_fallback": (
             f"현재 tick 이전 {contract.ROLLING_QUANTILE_WINDOW}개 연속 관측이 없으면 그 tick은 뺀다"
             if config.is_rolling_quantile else
             "전일이 없거나 표본이 모자라면 그 종목-일을 뺀다"),
         "excluded_dates": sorted(C.NOT_REAL_DATES)}
        if config.is_quantile else
        {"scope": "GLOBAL_LITERAL",
         "reference": "NONE",
         "quantile_method": None,
         "minimum_observations": 0,
         "minimum_observations_derivation": "not applicable",
         "session": V.session_policy()["policy"],
         "no_fallback": "모든 종목-일에 잠근 literal 값을 그대로 쓴다",
         "excluded_dates": []})
    return {
        "schema": "parameter_search_plan.v1", "created_at": now_utc(),
        "hypothesis_id": spec["hypothesis_id"],
        "specification_sha256": spec["specification_sha256"],
        "grid_parameter": [parameter for _feature, _direction, parameter in config.states],
        "search_dimensions": 1,
        "joint_quantile": len(config.states) > 1,
        # 무엇을 어느 방향으로 자르는가. 보고서가 이것을 그대로 쓴다.
        "feature": config.feature,
        "direction": config.direction,
        "direction_word": "낮은" if config.lower else "높은",
        "comparator": "<" if config.lower else ">",
        "controls": list(config.controls),
        "grid": list(config.grid),
        "grid_note": "성기게 고정한다. 결과를 보고 세분하지 않는다",
        "threshold_source": threshold_source(config),
        "calibration": calibration,
        "primary_metric": config.primary_metric,
        "null_value": config.null_value,
        "matching_contract": V.MATCHING_CONTRACT,
        "selection_rule": SELECTION_RULE,
        "blocks": {k: list(v) for k, v in blocks.items()},
        "symbols": list(symbols),
        "config": asdict(config),
        "not_done_here": ["청산 규칙", "손익 최적화", "terminal OOS 평가",
                          "새 feature 탐색", "격자 세분"],
    }


# ---- 9. 잠금 (§45~§48) ----------------------------------------------------------

def parameter_lock(plan_payload: Mapping[str, Any], selection: Mapping[str, Any],
                   confirmation: Mapping[str, Any], spec: Mapping[str, Any], *,
                   status: str = PARAMETER_LOCKED) -> dict[str, Any]:
    """선택된 q를 잠근다. 직접 증거의 확인 결과는 별도로 보존한다."""
    source = dict(plan_payload.get("threshold_source") or {
        "kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE})
    is_quantile = is_quantile_threshold_kind(source.get("kind"))
    value = selection.get("selected_value", selection.get("selected_q"))
    source["q" if is_quantile else "value"] = value
    parameter_values = {parameter: value for _feature, _direction, parameter in config_states(plan_payload)}
    locked = {
        "schema": "parameter_lock.v1", "created_at": now_utc(),
        "hypothesis_id": plan_payload["hypothesis_id"],
        "parameter": plan_payload["grid_parameter"][0],
        "value": value,
        "parameters": parameter_values,
        "threshold_source": source,
        "status": status,
        "grid": plan_payload["grid"],
        "calibration": plan_payload["calibration"],
        "primary_metric": plan_payload["primary_metric"],
        "null_value": plan_payload["null_value"],
        "matching_contract": plan_payload["matching_contract"],
        "selection_rule": plan_payload["selection_rule"],
        "blocks": plan_payload["blocks"],
        "specification_sha256": spec["specification_sha256"],
        "plan_sha256": sha256_json(plan_payload)[:16],
        "confirmation_sha256": sha256_json(confirmation)[:16],
        "selection_direct_evidence_status": selection.get("direct_evidence_status"),
        "confirmation_status": confirmation.get("status"),
        "runtime_note": (
            "잠기는 것은 q 하나다. 실제 임계는 각 tick에서 현재 tick을 뺀 직전 100개 "
            "관측으로 다시 계산된다 — 값이 변하는 것이지 파라미터가 변하는 것이 아니다"
            if source.get("kind") == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE else
            "잠기는 것은 q 하나다. 각 상태의 종목-일 실제 임계는 전일 분포에서 결정적으로 "
            "다시 계산된다 — 값이 변하는 것이지 파라미터가 변하는 것이 아니다"
            if is_quantile else
            "잠기는 것은 literal 값 하나다. 모든 종목·날짜에 같은 임계가 들어간다"),
        "oos_rule": (
            "이후 어떤 구간에서도 q 를 다시 고르지 않는다. 각 tick의 임계만 그 직전 100개 "
            "관측에서 순차로 뜬다"
            if source.get("kind") == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE else
            "이후 어떤 구간에서도 q 를 다시 고르지 않는다. 날짜별 임계만 직전 실거래일에서 "
            "순차로 뜬다"
            if is_quantile else
            "이후 어떤 구간에서도 literal 값을 다시 고르거나 다시 보정하지 않는다"),
        "naming":
            "이 값을 '수익 나는 임계' 라고 부르지 않는다. 청산 규칙이 아직 없다. "
            "SELECTED_SIGNAL_CUTOFF 다",
    }
    locked["lock_sha256"] = sha256_json(
        {k: v for k, v in locked.items() if k != "created_at"})[:16]
    return locked


def config_states(plan_payload: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """계획 artifact만으로도 lock의 모든 parameter를 복원한다."""
    raw = (plan_payload.get("config") or {}).get("state_definitions") or []
    if raw:
        return tuple((str(item[0]), str(item[1]), str(item[2])) for item in raw)
    return ((str(plan_payload.get("feature")), str(plan_payload.get("direction")),
             str((plan_payload.get("grid_parameter") or [""])[0])),)


def sequential_thresholds(symbol: str, dates: Sequence[str], quantile: float,
                          config: SearchConfig, root: Path = TICK_ROOT) -> pd.DataFrame:
    """OOS에서 q를 다시 고르지 않고, 날짜별 실행 가능 여부만 확인한다."""
    calendar = trading_calendar(root)
    rows = []
    for date in dates:
        row = calibrate(symbol, date, config, calendar, root)
        rows.append({"symbol": symbol, "date": date,
                     "reference_date": row["reference_date"],
                     "threshold": (float(quantile) if config.is_rolling_quantile else
                                   row.get(threshold_column(quantile, config))),
                     "eligible": row["eligible"], "reason": row["reason"]})
    return pd.DataFrame(rows)




# ---- 10. 배선 ------------------------------------------------------------------

CONTROLS = SearchConfig.controls   # 하위 호환. 실제로는 config.controls 를 쓴다


def collect_anchors(dates: Sequence[str], symbols: Sequence[str],
                    root: Path = TICK_ROOT,
                    config: "SearchConfig | None" = None,
                    extra_features: Sequence[str] = (),
                    anchor_cache_root: Path | None = None) -> pd.DataFrame:
    """검증이 쓰던 것과 같은 anchor 우주. 미래로 고르지 않는다.

    시험할 feature 와 통제 변수는 가설마다 다르다. anchor 표에 그 열이 있어야 한다.
    """
    config = config or SearchConfig()
    base = V.ValidationConfig()
    state_features = tuple(feature for feature, _direction, _parameter in config.states)
    wanted = tuple(dict.fromkeys(
        state_features + tuple(c for c in config.controls if c in catalog.FEATURES)
        + tuple(str(c) for c in extra_features if str(c) in catalog.FEATURES)))
    if anchor_cache_root is not None:
        from .modules import search_anchor_cache
        cache_columns = tuple(dict.fromkeys(
            V.SELECTION_BASE_COLUMNS + V.RESPONSE_COLUMNS
            + ("future_label_not_used_for_selection",) + wanted
            + tuple(search_anchor_cache.threshold_column(feature, direction, q)
                    for feature, direction, _parameter in config.states
                    for q in config.grid)))
        return search_anchor_cache.anchors(anchor_cache_root, dates, symbols,
                                           columns=cache_columns)
    validation_config = dataclasses.replace(
        base, trigger_feature=config.feature,
        context_feature=next((c for c in wanted if c != config.feature),
                             base.context_feature),
        extra_features=wanted)
    return V.collect(dates, symbols, validation_config, root)


def _apply_entry_guards(frame: pd.DataFrame, symbols: Sequence[str], dates: Sequence[str],
                        guards: Sequence[Mapping[str, Any]], root: Path,
                        anchor_cache_root: Path | None = None) -> pd.DataFrame:
    """Refinement가 이미 고른 조항을 Parameter Search의 모든 후보에 똑같이 붙인다."""
    result = frame
    for guard in guards:
        feature, quantile = str(guard["feature"]), float(guard["quantile"])
        if feature not in catalog.FEATURES:
            raise ValueError(f"Refinement guard의 feature가 Catalog에 없다: {feature}")
        config = SearchConfig(parameter=f"guard_{feature}", feature=feature,
                              grid=(quantile,), direction="HIGHER")
        table = calibration_table(symbols, dates, config, root,
                                  anchor_cache_root=anchor_cache_root)
        thresholds = table.loc[table["eligible"], ["symbol", "date"]]
        result = result.merge(thresholds, on=["symbol", "date"], how="inner")
        if config.is_rolling_quantile:
            cut = _rolling_thresholds_for_frame(
                result, quantile, config, root,
                anchor_cache_root=anchor_cache_root)[f"{catalog.UNRESOLVED_PREFIX}{config.parameter}"]
        else:
            column = threshold_column(quantile, config)
            result = result.merge(
                table.loc[table["eligible"], ["symbol", "date", column]],
                on=["symbol", "date"], how="inner").rename(
                    columns={column: "_entry_guard_threshold"})
            cut = result["_entry_guard_threshold"].to_numpy(float)
        values = result[feature].to_numpy(float)
        keep = values <= cut if str(guard["operator"]) == "<=" else values >= cut
        result = result.loc[np.isfinite(values) & np.isfinite(cut) & keep]
        if "_entry_guard_threshold" in result:
            result = result.drop(columns=["_entry_guard_threshold"])
    return result


def _entry_guard_variants(config: SearchConfig,
                          variants: Mapping[str, Sequence[Mapping[str, Any]]] | None,
                          default: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """q별 Discovery guard를 q 후보와 정확히 한 번씩 연결한다."""
    expected = {config.candidate_id(q) for q in config.grid}
    if variants is None:
        return {candidate_id: [dict(guard) for guard in default]
                for candidate_id in sorted(expected)}
    received = {str(candidate_id) for candidate_id in variants}
    if received != expected:
        raise ValueError(f"q별 Entry Refinement guard 후보가 Search grid와 다르다: "
                         f"expected={sorted(expected)}, received={sorted(received)}")
    normalised: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in sorted(expected):
        guards = variants[candidate_id]
        if not isinstance(guards, Sequence) or isinstance(guards, (str, bytes)):
            raise ValueError(f"q별 Entry Refinement guard가 목록이 아니다: {candidate_id}")
        normalised[candidate_id] = [dict(guard) for guard in guards]
    return normalised


def run(spec_path: Path, output: Path, *, blocks: Mapping[str, Sequence[str]],
        symbols: Sequence[str] | None = None, config: SearchConfig | None = None,
        root: Path = TICK_ROOT, environment: str = "",
        anchor_cache_root: Path | None = None,
        allow_single_day_lock: bool = False,
        allow_discovery_only_lock: bool | None = None,
        entry_guards: Sequence[Mapping[str, Any]] = (),
        entry_guard_variants: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """계약 수정 → 충실도 재검사 → 격자 → 선택 → 새 데이터 확인 → 잠금.

    `config` 를 주지 않으면 **명세에서 만든다**. 기본값을 그냥 쓰면 명세가 말하는
    feature 와 다른 것을 탐색하고도 알아채지 못한다.
    """
    # Sandbox 확장 workflow가 쓰는 이름을 기존 단일-day lock 의미로만 연결한다.
    # 이 값이 없으면 기존 호출의 동작은 바뀌지 않는다.
    if allow_discovery_only_lock is not None:
        allow_single_day_lock = bool(allow_discovery_only_lock)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    symbols = list(symbols or catalog.DEFAULT_PROFILE_SYMBOLS)
    spec = read_json(Path(spec_path))
    config = config_for(spec) if config is None else config
    variants = _entry_guard_variants(config, entry_guard_variants, entry_guards)

    # 명세와 설정이 어긋나면 멈춘다. 조용히 맞추지 않는다.
    declared = (spec.get("semantic_invariants") or {}).get("canonical_feature")
    configured_features = {feature for feature, _direction, _parameter in config.states}
    if declared and str(declared) not in configured_features:
        raise RuntimeError(
            f"명세는 `{declared}` 를 말하는데 탐색 설정은 `{sorted(configured_features)}` 를 본다. "
            "조용히 맞추지 않는다")

    # 1. threshold source를 계약에 맞게 고정한다. 나머지는 건드리지 않는다.
    amended = amend_specification(spec, config)
    template = I.compile_contract(amended)
    template["entry_guards"] = [dict(guard) for guard in entry_guards]
    template["contract_sha256"] = I.contract.contract_hash(template)[:16]
    validity = I.implementation_validity(template)
    fidelity = I.fidelity_check(amended, template)
    if not validity["implementation_valid"] or \
            fidelity["fidelity_status"] != X.FIDELITY_PRESERVED:
        raise RuntimeError(
            "매개화를 바꾼 뒤 충실도가 깨졌다. 탐색을 시작하지 않는다:\n  "
            + "\n  ".join(validity["problems"] + fidelity["problems"]))

    payload = plan(amended, blocks, symbols, config)
    payload["entry_refinement_branches"] = {
        "mode": ("Q_SPECIFIC_DISCOVERY_GUARDS" if entry_guard_variants is not None
                 else "COMMON_ENTRY_GUARDS"),
        "candidates": [{"candidate_id": config.candidate_id(q),
                        "entry_guards": variants[config.candidate_id(q)]}
                       for q in config.grid],
    }
    write_json(output / "parameter_search_plan.json", payload)

    # 2. 보정. 후보 전체가 같은 우주를 쓴다
    fit_dates = list(blocks[SEARCH_FIT])
    confirm_dates = list(blocks[SEARCH_CONFIRM])
    table = calibration_table(symbols, fit_dates, config, root,
                              anchor_cache_root=anchor_cache_root)
    calibration = {
        "schema": "calibration_manifest.v1", "created_at": now_utc(),
        **payload["calibration"],
        "symbol_days": int(len(table)),
        "eligible_symbol_days": int(table["eligible"].sum()),
        "excluded_symbol_days": int((~table["eligible"]).sum()),
        "reasons": table.loc[~table["eligible"], "reason"].value_counts().to_dict(),
    }
    write_json(output / "calibration_manifest.json", calibration)

    # 3. 격자. 승자만이 아니라 곡선 전체를 남긴다
    all_guards = [guard for guards in variants.values() for guard in guards]
    guard_features = tuple(sorted({str(guard["feature"]) for guard in all_guards}))
    fit_anchors = collect_anchors(fit_dates, symbols, root, config,
                                  extra_features=guard_features,
                                  anchor_cache_root=anchor_cache_root)
    candidates = []
    for q in config.grid:
        candidate_id = config.candidate_id(q)
        guards = variants[candidate_id]
        fit_frame = _apply_entry_guards(fit_anchors, symbols, fit_dates, guards, root,
                                        anchor_cache_root=anchor_cache_root)
        candidate = score_candidate(fit_frame, table, q, config, config.controls, root=root,
                                    anchor_cache_root=anchor_cache_root)
        candidate["candidate_id"] = candidate_id
        candidate["entry_guards"] = [dict(guard) for guard in guards]
        candidates.append(candidate)
    selection = select(candidates)
    selected_guards = variants.get(config.candidate_id(selection["selected_q"]), []) \
        if selection.get("selected_q") is not None else []
    template["entry_guards"] = [dict(guard) for guard in selected_guards]
    template["contract_sha256"] = I.contract.contract_hash(template)[:16]

    # 4. 고른 하나만 새 데이터에서 연다
    confirmation: dict[str, Any] | None = None
    lock: dict[str, Any] | None = None
    opened = list(fit_dates)
    if selection["status"] == SEARCH_SELECTED:
        if confirm_dates:
            confirm_table = calibration_table(symbols, confirm_dates, config, root,
                                              anchor_cache_root=anchor_cache_root)
            confirm_frame = _apply_entry_guards(
                collect_anchors(confirm_dates, symbols, root, config,
                                extra_features=guard_features,
                                anchor_cache_root=anchor_cache_root),
                symbols, confirm_dates, selected_guards, root,
                anchor_cache_root=anchor_cache_root)
            confirmation = confirm(confirm_frame, confirm_table,
                                   selection["selected_value"], config, config.controls, root=root,
                                   anchor_cache_root=anchor_cache_root)
            opened += confirm_dates
            lock = parameter_lock(payload, selection, confirmation, amended)
        elif allow_single_day_lock:
            confirmation = {
                "schema": "parameter_confirmation.v1", "created_at": now_utc(),
                "q": selection["selected_q"], "value": selection["selected_value"],
                "threshold_source": threshold_source(config, selection["selected_value"]),
                "status": "NOT_RUN_SINGLE_DAY_DISCOVERY",
                "only_one_candidate_opened": False,
                "why": "사용자가 승인한 1일 Discovery 실험이라 별도 Search confirm 날짜가 없다",
                "next": "Validation Backtest에서 같은 canonical Backtest로 검증한다",
            }
            lock = parameter_lock(payload, selection, confirmation, amended,
                                  status=PARAMETER_LOCKED_SINGLE_DAY)

    audit = search_audit(payload, table, candidates, fidelity, opened, config)
    status = (lock["status"] if lock else
              confirmation["status"] if confirmation else selection["status"])

    write_json(output / "executable_specification_amended.json", amended)
    write_json(output / "contract_template.json", template)
    write_json(output / "selected_parameter.json", selection)
    write_json(output / "search_audit.json", audit)
    if confirmation:
        write_json(output / "confirmation_result.json", confirmation)
    if lock:
        write_json(output / "parameter_lock.json", lock)
    pd.DataFrame(candidates).to_csv(output / "candidate_results.csv", index=False)
    table.to_csv(output / "calibration_table.csv", index=False)

    result = {"status": status, "plan": payload, "calibration": calibration,
              "candidates": candidates, "selection": selection,
              "confirmation": confirmation, "lock": lock, "audit": audit,
              "amended": amended, "template": template, "fidelity": fidelity,
              "table": table, "output": str(output)}
    (output / "parameter_search_report.md").write_text(
        to_markdown(result, environment), encoding="utf-8")
    _companions(result, output)
    return result


def to_markdown(result: Mapping[str, Any], environment: str = "") -> str:
    payload, selection = result["plan"], result["selection"]
    audit, lock = result["audit"], result["lock"]
    confirmation = result["confirmation"]
    calibration = result["calibration"]
    source_kind = (payload.get("threshold_source") or {
        "kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE}).get("kind")
    is_quantile = is_quantile_threshold_kind(source_kind)
    rolling = source_kind == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    value_label = "q" if is_quantile else "literal"
    selected_value = selection.get("selected_value", selection.get("selected_q"))
    L = [f"# 파라미터 탐색 — {payload['hypothesis_id']}", "",
         "고정된 가설 안에서 **하나 남은 값**을 정한다. 새 전략을 찾는 단계가 아니다.", ""]
    if environment:
        L += [environment, "---", ""]
    L += ["## 0. 무엇을 묻는가", "",
          f"> 각 종목에서 **어느 정도로 {payload['direction_word']}** "
          f"`{payload['feature']}` 상태부터 실제 신호로 취급할 것인가?", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 탐색 축 | `{payload['grid_parameter'][0]}` — **1개** |",
          f"| 격자 | `{payload['grid']}` |",
          f"| 주 통계 | `{payload['primary_metric']}` (귀무값 {payload['null_value']}) |",
          f"| 상태 | **`{result['status']}`** |",
          f"| 계획 해시 | `{sha256_json(payload)[:16]}` |", ""]
    if is_quantile:
        L += ["## 1. 원값에서 분위로", "",
              "```",
              f"이전   {payload['feature']} {payload['comparator']} 0.4"
              f"            종목마다 다른 뜻",
              f"이후   {payload['feature']} {payload['comparator']} {'직전 100 tick q' if rolling else '전일 q'}"
              f"         모든 종목에서 같은 뜻",
              "```", "",
              "같은 원값이 어떤 종목에선 극단값이고 어떤 종목에선 평범하다. 그대로 쓰면 "
              "상태가 아니라 **종목 크기를 거르는 필터**가 된다.", "",
              "**탐색 차원은 늘지 않았다** — 하나에서 하나다. 바뀐 것은 그 하나가 무엇을 "
              "뜻하는가지 몇 개인가가 아니다.", "",
              "> 분위로 바꾼다고 과적합이 사라지지 않는다. 그건 성긴 격자·확대 금지·"
              "새 데이터 확인이 막는다.", "",
              "## 2. 임계를 언제 정하는가", "", "```",
              ("각 tick 직전 100개  →  q 분위 계산  →  현재 tick 판정" if rolling else
               "20260413 분포  →  q 분위 계산  →  20260414 장 전체에 고정"),
              "```", "",
              ("현재 tick과 이후 tick은 임계에 쓰지 않는다. 직전 100개만 쓴다." if rolling else
               "당일 데이터를 그날 임계에 쓰지 않는다. 장 시작 전에 이미 정해져 있어야 한다.")]
    else:
        L += ["## 1. 고정 원값", "",
              f"`{payload['feature']} {payload['comparator']} literal` 을 후보 격자에서 고른다.",
              "고른 값은 종목·날짜마다 다시 보정하지 않는다.", "",
              "## 2. 임계를 언제 정하는가", "", "```",
              "Discovery에서 literal 선택  →  Validation/OOS 전 기간에 같은 값",
              "```", "",
              "당일이나 전일 데이터는 이 임계를 만드는 데 쓰지 않는다."]
    L += ["", "| 항목 | 값 |", "|---|---|",
          f"| 범위 | `{calibration['scope']}` |",
          f"| 기준일 | `{calibration['reference']}` |",
          f"| threshold source | `{source_kind}` |",
          f"| 분위 계산법 | `{calibration['quantile_method']}`"
          f"{' (보간하지 않는다)' if is_quantile else ''} |",
          f"| 최소 관측 | {calibration['minimum_observations']} "
          f"— {calibration['minimum_observations_derivation']} |",
          f"| 적격 종목-일 | {calibration['eligible_symbol_days']:,} / "
          f"{calibration['symbol_days']:,} |", "",
          f"- {calibration['no_fallback']}"]
    if is_quantile:
        L += ["- 더 오래된 날로 물러서면 종목-일마다 보정 기간이 달라진다 — 그것은 새 "
              "자유도다."]
    L += [""]
    if calibration.get("reasons"):
        L += ["빠진 이유:", ""] + [f"- {k} — {v}건" for k, v in
                                  calibration["reasons"].items()] + [""]

    L += ["## 3. 응답 곡선", "",
          "승자만 남기지 않는다. 전체를 본다.", "",
          f"| {value_label} | 신호 비율 | 짝 | 짝 우위 | 95% 구간 | 판정 | 동률 | 균형비 |",
          "|---|---|---|---|---|---|---|---|"]
    for row in result["candidates"]:
        effect = row.get("effect_size")
        interval = ("—" if effect is None else
                    f"[{row['interval_low']:.3f}, {row['interval_high']:.3f}]")
        tie = row.get("tie_rate")
        balance = row.get("matching_balance_ratio")
        L.append("| **{q:.2f}** | {rate} | {pairs:,} | {effect} | {interval} | "
                 "`{status}` | {tie} | {balance} |".format(
                     q=row["q"],
                     rate=_pct_1(row.get("signal_rate")),
                     pairs=row.get("n_matched_pairs", 0),
                     effect="—" if effect is None else f"{effect:.4f}",
                     interval=interval,
                     status=row.get("status"),
                     tie="—" if tie is None else f"{tie:.1%}",
                     balance="—" if balance is None else f"{balance:.3f}"))
    L += ["", f"### 진단 — 이걸 보고 {value_label} 를 고르지 않는다", "",
          "청산 규칙이 아직 없어서 손익을 목적으로 삼을 수 없다.", "",
          f"| {value_label} | 신호 anchor 평균 net | 중앙 | 지속 성공률 | 날짜 일치 | 클러스터 일치 |",
          "|---|---|---|---|---|---|"]
    for row in result["candidates"]:
        if row.get("diag_mean_max_net_bps") is None:
            continue
        L.append(f"| {row['q']:.2f} | {row['diag_mean_max_net_bps']:.2f} | "
                 f"{row['diag_median_max_net_bps']:.2f} | "
                 f"{row['diag_sustained_success_rate']:.4%} | "
                 f"{_pct(row.get('date_direction_agreement'))} | "
                 f"{_pct(row.get('cluster_direction_agreement'))} |")

    # 곡선이 한쪽으로만 오르는가, 이긴 값이 격자 끝인가. 숨기면 안 되는 사실이다.
    scored = [r for r in result["candidates"] if r.get("effect_size") is not None]
    monotone = all(a["effect_size"] <= b["effect_size"]
                   for a, b in zip(scored, scored[1:])) if len(scored) > 1 else False
    at_edge = bool(scored) and selected_value == max(
        r["q"] for r in scored)
    if monotone or at_edge:
        L += ["", "### 격자 경계", ""]
        if monotone:
            L += ["곡선이 **한쪽으로만 오른다.** 안쪽에 봉우리가 없다.", ""]
        if at_edge:
            edge_note = ("더 높은 분위가 더 나을 수도 있다는 뜻" if is_quantile else
                         "격자 밖 값이 더 나을 수도 있다는 뜻")
            L += [f"이긴 값 `{selected_value}` 이 **격자의 끝**이다. {edge_note}이지만, "
                  "그것을 확인하려고 격자를 넓히지 않는다 — 결과를 보고 탐색 공간을 늘리는 "
                  "것이기 때문이다.", "",
                  "대신 두 가지를 함께 읽어야 한다.", "",
                  "| | |", "|---|---|"]
            edge = next(r for r in scored if r["q"] == selected_value)
            L += [f"| 신호 비율 | {edge['signal_rate']:.1%} — threshold를 높일수록 기회가 준다 |",
                  f"| 짝 | {edge['n_matched_pairs']:,} |",
                  f"| 동률 | {edge['tie_rate']:.1%} |", "",
                  "우위가 커지는 대신 표본이 줄어든다. 더 높은 값을 원하면 그것은 이 "
                  "탐색의 결과가 아니라 **별도 판의 설계 문제**다.", ""]

    L += ["", "## 4. 고르는 규칙 — 결과 보기 전에 얼렸다", "", "```"]
    L += [f"{k:12s} {v}" for k, v in payload["selection_rule"].items()]
    L += ["```", "",
          f"- 판정 `{selection['status']}` · 고른 값 "
          f"`{selected_value}`",
          f"- {selection['reason']}", ""]
    if selection.get("runner_up"):
        L += [f"- 차점자 `{selection['runner_up']}` — **열지 않는다**", ""]

    if confirmation and confirmation.get("result"):
        outcome = confirmation["result"]
        L += ["## 5. 새 데이터에서 확인", "",
              f"`{', '.join(payload['blocks'][SEARCH_CONFIRM])}` 에서 **고른 값 하나만** "
              "연다.", "", "| 항목 | 값 |", "|---|---|",
              f"| {value_label} | {confirmation['value']:.2f} |",
              f"| 짝 우위 | {outcome.get('effect_size', float('nan')):.4f} |",
              f"| 95% 구간 | [{outcome.get('interval_low', float('nan')):.3f}, "
              f"{outcome.get('interval_high', float('nan')):.3f}] |",
              f"| 짝 | {outcome.get('n_matched_pairs', 0):,} |",
              f"| 판정 | **`{confirmation['status']}`** |", "",
              f"{confirmation['why']}", "",
              f"실패했다면: {confirmation['on_failure']}", ""]
    elif confirmation:
        L += ["## 5. 새 데이터에서 확인", "",
              f"`{confirmation['status']}` — {confirmation['why']}", "",
              confirmation["next"], ""]
    else:
        L += ["## 5. 새 데이터에서 확인", "",
              "고른 값이 없어 확인 구간을 열지 않았다. **데이터를 아꼈다.**", ""]

    L += ["## 6. 감사", "", "### 0 이어야 하는 것", "", "| 지표 | 값 |", "|---|---|"]
    for key in MUST_BE_ZERO:
        L.append(f"| `{key}` | **{audit['metrics'].get(key, 0)}** |")
    L += ["", "### 경고성 — 0 일 필요 없다", "", "| 지표 | 값 |", "|---|---|"]
    for key in WARNING_ONLY:
        L.append(f"| `{key}` | {audit['metrics'].get(key, 0)} |")
    L += ["", f"- 종합 `{'통과' if audit['ok'] else '실패: ' + str(audit['failed'])}`", ""]

    L += ["## 7. 데이터 예산", "", "| 블록 | 날짜 | 용도 |", "|---|---|---|"]
    for name, dates in payload["blocks"].items():
        span = f"{dates[0]}~{dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "—")
        note = ("**열지 않았다**" if name == TERMINAL_OOS else
                "고른 값 하나만 열었다" if name == SEARCH_CONFIRM and confirmation else
                "열지 않았다" if name == SEARCH_CONFIRM else "후보 비교에 썼다")
        L.append(f"| `{name}` | `{span}` ({len(dates)}일) | {note} |")

    if lock:
        L += ["", "## 8. 잠금", "", "| 항목 | 값 |", "|---|---|",
              f"| `{lock['parameter']}` | **{lock['value']}** |",
              f"| 상태 | `{lock['status']}` |",
              f"| 잠금 해시 | `{lock['lock_sha256']}` |", "",
              f"- {lock['runtime_note']}",
              f"- {lock['oos_rule']}", "",
              f"**{lock['naming']}**", ""]
    L += ["## 9. 이 단계가 답하지 않는 것", "",
          "> 이 계약이 실제로 돈을 버는가?", "",
          "청산 규칙·보유 정책·포지션 크기가 아직 없다. 그래서 고른 값을 **수익 나는 "
          "임계**라고 부르지 않는다.", "",
          "아직 안 한 것: " + ", ".join(payload["not_done_here"]), ""]
    return "\n".join(L)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _pct_1(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.1%}"


def _companions(result: Mapping[str, Any], output: Path) -> None:
    payload, selection = result["plan"], result["selection"]
    audit, lock, confirmation = result["audit"], result["lock"], result["confirmation"]
    calibration = result["calibration"]
    source = payload.get("threshold_source") or {
        "kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE}
    is_quantile = is_quantile_threshold_kind(source.get("kind"))
    rolling = source.get("kind") == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    value_label = "q" if is_quantile else "literal"
    selected_value = selection.get("selected_value", selection.get("selected_q"))

    pd.DataFrame([{"항목": k, "값": v} for k, v in {
        "가설": payload["hypothesis_id"],
        "탐색축": payload["grid_parameter"][0],
        "탐색차원": 1,
        "격자": ", ".join(f"{q:.2f}" for q in payload["grid"]),
        "보정범위": calibration["scope"],
        "보정기준일": calibration["reference"],
        "분위계산법": calibration["quantile_method"],
        "최소관측": calibration["minimum_observations"],
        "적격종목일": f"{calibration['eligible_symbol_days']}/{calibration['symbol_days']}",
        "주통계": payload["primary_metric"],
        "귀무값": payload["null_value"],
        "선택상태": selection["status"],
        "선택된값": selected_value,
        "threshold_source": source["kind"],
        "확인상태": confirmation["status"] if confirmation else "열지 않음",
        "최종상태": result["status"],
        "잠긴값": lock["value"] if lock else None,
        "감사통과": audit["ok"],
        "terminal열었나": "아니오",
    }.items()]).to_csv(output / "summary.csv", index=False)

    pd.DataFrame([{"값": value, "상태": "후보",
                   "설명": ((f"직전 100 tick {value:.0%} 분위부터 신호" if rolling else
                             f"전일 {value:.0%} 분위부터 신호") if is_quantile else
                            f"고정 원값 {value:g}부터 신호")}
                  for value in payload["grid"]]).to_csv(output / "search_grid.csv", index=False)

    curve = [{value_label: r.get("value", r.get("q")), "신호비율": r.get("signal_rate"),
              "신호anchor": r.get("signal_decisions"), "짝": r.get("n_matched_pairs"),
              "짝우위": r.get("effect_size"), "구간하한": r.get("interval_low"),
              "구간상한": r.get("interval_high"), "귀무값": r.get("null_value"),
              "동률비율": r.get("tie_rate"), "동률제외n": r.get("n_effective"),
              "짝짓기균형비": r.get("matching_balance_ratio"),
              "판정": r.get("status"),
              "진단_평균net": r.get("diag_mean_max_net_bps"),
              "진단_중앙net": r.get("diag_median_max_net_bps"),
              "진단_지속성공률": r.get("diag_sustained_success_rate"),
              "날짜방향일치": r.get("date_direction_agreement"),
              "클러스터방향일치": r.get("cluster_direction_agreement")}
             for r in result["candidates"]]
    pd.DataFrame(curve).to_csv(output / "response_curve.csv", index=False)

    pd.DataFrame([{"단계": k, "규칙": v} for k, v in payload["selection_rule"].items()]
                 + [{"단계": "결과", "규칙": f"{selection['status']} · "
                                             f"{value_label}={selected_value}"}]
                 ).to_csv(output / "selection.csv", index=False)

    if confirmation and confirmation.get("result"):
        outcome = confirmation["result"]
        pd.DataFrame([{"항목": k, "값": v} for k, v in {
            value_label: confirmation["value"], "판정": confirmation["status"],
            "짝우위": outcome.get("effect_size"),
            "구간하한": outcome.get("interval_low"),
            "구간상한": outcome.get("interval_high"),
            "짝": outcome.get("n_matched_pairs"),
            "동률비율": outcome.get("tie_rate"),
            "차점자를_열었나": "아니오",
        }.items()]).to_csv(output / "confirmation.csv", index=False)

    meaning = {
        "new_feature_count": "가설에 없던 feature 를 신호에 넣었다",
        "new_relation_count": "신호 구조가 명세와 달라졌다",
        "new_mechanism_count": "미식별 메커니즘을 조건으로 만들었다",
        "raw_threshold_search_count": "계획과 다른 threshold source를 썼다",
        "search_dimension_change_count": "탐색 축이 하나가 아니다",
        "current_day_calibration_leakage_count": "그날 데이터로 그날 임계를 만들었다",
        "future_calibration_leakage_count": "미래 날짜를 임계 계산에 썼다",
        "zoom_in_search_count": "결과를 보고 격자를 세분했다",
        "post_result_grid_change_count": "결과를 보고 격자를 바꿨다",
        "matching_rule_change_count": "짝짓기 규칙을 바꿨다",
        "metric_change_count": "주 통계를 바꿨다",
        "terminal_oos_access_count": "예약한 마지막 구간을 열었다",
        "calibration_insufficient_symbol_day_count": "전일 표본이 모자라 뺀 종목-일",
        "threshold_tie_count": "임계가 동률인 경우",
        "low_signal_support_candidate_count": "짝이 모자라 판정 못 한 후보",
        "high_signal_concentration_warning_count": "신호가 너무 자주 또는 드물게 켜진 후보",
    }
    pd.DataFrame([{"지표": k, "값": v,
                   "기준": "0 이어야 함" if k in MUST_BE_ZERO else "경고성",
                   "무엇을_잡는가": meaning.get(k, "")}
                  for k, v in audit["metrics"].items()]
                 ).to_csv(output / "audit.csv", index=False)

    rows = []
    for name, dates in payload["blocks"].items():
        for date in dates:
            rows.append({"날짜": date, "용도": name,
                         "열었나": "아니오" if name == TERMINAL_OOS else
                                   ("고른 값만" if name == SEARCH_CONFIRM and confirmation
                                    else "아니오" if name == SEARCH_CONFIRM else "예")})
    pd.DataFrame(rows).to_csv(output / "data_ledger.csv", index=False)
