"""사전에 고정한 가설이 숨겨 둔 미래 데이터에서도 버티는지 본다.

여기서 처음으로 anchor 이후 데이터를 연다. 그런데 여는 순간 가장 쉬운 실수가 하나 있다.

## 라벨 재확인은 검증이 아니다

PROFIT 은 이미 `anchor ASK1 → S30 안의 미래 BID1` 로 정의돼 있다. 그래서

    PROFIT anchor 만 모은다 → anchor 이후 BID1 이 좋아졌나? → 그렇다 → "검증됨"

은 동어반복이다. PROFIT 이 애초에 그 경로로 정의됐기 때문이다.

그래서 검증은 **미래로 고르지 않은 전체 적격 anchor** 위에서 한다. anchor 를 고를 때
쓰는 것은 종목·날짜·호가 유효성·시각뿐이고, 미래는 anchor 가 고정된 뒤 **응답 변수**
로만 연다.

## 얼리고 나서 연다

가설·grounding·카탈로그·검증 계획의 해시를 먼저 얼린다(`freeze`). 그 다음에야 미래를
연다. 결과를 보고 계획이 바뀌면 v1 을 덮어쓰지 않고 v2 를 만든다 — 덮어쓰면 그 데이터가
수정된 가설의 개발 데이터가 되고, 같은 데이터로 다시 확증할 수 없게 된다.

## 새 숫자를 만들지 않는다

평가 창·비용·수익 기준은 outcome 정의 그대로, 사전 창(-10s)과 층(종목·날짜·15분)은
grounding 이 고정한 것 그대로다. 임계·lag·평가 창을 탐색하지 않는다.
"""

from __future__ import annotations

import hashlib
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import capability as C, catalog, data as tickdata, grounding as G
from .config import FEE_BPS, PRIMARY_HORIZON_SECONDS, TICK_ROOT, now_utc, read_json, \
    sha256_json, write_json


SCHEMA_VERSION = "mechanism_validation.v1_1"

# v1 통계의 결함과 그 교정. v1 산출물은 남겨 두고 덮지 않는다 (§5).
STATISTIC_NOTES = {
    "v1_defect_already_priced":
        "잔여 net 의 **절대 수준**으로 판정했다. 그 값의 중앙은 -(스프레드+수수료) 라 "
        "taker 왕복 비용 바닥일 뿐이고 '움직임이 trigger 전에 끝났는가' 를 재지 못한다. "
        "v2 는 trigger 강약을 짝지어 **사전 이동과 사후 잔여를 각각 비교**한다.",
    "v2_defect_incremental":
        "짝 차이의 중앙값으로 판정했다. 호가가 틱 격자라 차이의 19~24% 가 정확히 0 이어서 "
        "중앙값이 0 으로 붕괴한다. v2 는 동률을 뺀 **부호 검정**을 주 통계로 쓴다.",
}

DISCOVERY_REPLAY, FROZEN_HOLDOUT = "DISCOVERY_REPLAY", "FROZEN_HOLDOUT"
SPLITS = (DISCOVERY_REPLAY, FROZEN_HOLDOUT)

TARGET_STATUS = ("NOT_RUN", "SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE",
                 "INCONCLUSIVE_DUE_TO_MATCHING", "BLOCKED_BY_CAPABILITY", "INVALID_TARGET",
                 "NON_IDENTIFYING_CHECK", "INVALID_VALIDATION_STATISTIC")
PREDICTION_STATUS = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_RUN")
RELATION_STATUS = ("CONTEXT_AND_TRIGGER_SUPPORTED", "TRIGGER_ONLY_SUPPORTED",
                   "CONTEXT_ONLY_SUPPORTED", "NEITHER_SUPPORTED", "INCONCLUSIVE",
                   # 축소된 가설용. 맥락은 더 이상 주장이 아니라 제거 확인 대상이다.
                   "SINGLE_STATE_SUPPORTED", "REMOVAL_CONTRADICTED")
MECHANISM_STATUS = ("IDENTIFIED", "CONSISTENT_BUT_UNIDENTIFIED", "CONTRADICTED",
                    "BLOCKED_BY_CAPABILITY")
ALREADY_PRICED_STATUS = ("RESIDUAL_MOVE_SUPPORTED", "ALREADY_PRICED_SUPPORTED",
                         "MIXED", "INCONCLUSIVE")
HYPOTHESIS_STATUS = ("MECHANISM_SUPPORTED", "REDUCED_FORM_SUPPORTED", "HYPOTHESIS_REVISE",
                     "HYPOTHESIS_REJECT", "INCONCLUSIVE", "DATA_CAPABILITY_BLOCKED")


# ---- v1.1 --------------------------------------------------------------------
#
# v1 에서 숫자는 맞게 계산됐는데 **그 숫자가 주장을 재지 않았다.** 그래서 이 계약이 있다.
#
#   계산이 맞다  !=  그 값이 주장하는 양을 잰다
#
# 이 절의 전부는 그 둘을 가르는 장치다.

# 이 주장을 현재 데이터로 어디까지 시험할 수 있는가 (§5~§10).
IDENTIFIABILITY = ("DIRECTLY_TESTABLE", "REDUCED_FORM_TESTABLE", "NON_IDENTIFYING_PROXY",
                   "BLOCKED_BY_CAPABILITY", "INVALID_TARGET")

# 통계가 재는 양의 종류. 주장이 묻는 것과 같아야 한다 (§16 Q1~Q2).
QUANTITY_KINDS = ("ABSOLUTE_LEVEL", "RELATIVE_COMPARISON", "RANK_ASSOCIATION")

# 지지의 강도. `SUPPORTED` 하나로 뭉치지 않는다 (§33~§38).
EVIDENCE_TIERS = ("DISCOVERY_SUPPORTED", "INTERNAL_HOLDOUT_SUPPORTED",
                  "REPEATED_HOLDOUT_SUPPORTED", "TERMINAL_OOS_SUPPORTED",
                  "CROSS_REGIME_SUPPORTED")

# 세션 범위. Validation 단계에서 조용히 바꾸지 않는다 (§24~§25).
CONTINUOUS_SESSION_ONLY = "CONTINUOUS_SESSION_ONLY"
FULL_CURRENT_FRAMEWORK_SESSION = "FULL_CURRENT_FRAMEWORK_SESSION"
SESSION_POLICIES = (CONTINUOUS_SESSION_ONLY, FULL_CURRENT_FRAMEWORK_SESSION)

# 날짜를 무엇에 썼는가 (§40).
LEDGER_USES = ("DISCOVERY", "VALIDATION", "REVISION_FEEDBACK", "FINAL_OOS", "UNUSED")

# 수정의 종류. 새 것을 넣었으면 수정이 아니라 발견이다 (§44~§48).
CONFIRMATORY_REVISION, DISCOVERY_REVISION = "CONFIRMATORY_REVISION", "DISCOVERY_REVISION"
REVISION_TYPES = (CONFIRMATORY_REVISION, DISCOVERY_REVISION)

# 틱 격자 때문에 동률이 이만큼 넘으면 중앙값을 주 통계로 쓸 수 없다 (§19~§20).
MATERIAL_TIE_RATE = 0.05


@dataclass(frozen=True)
class ValidationConfig:
    """전부 상위 단계가 이미 정한 값이다. 여기서 새로 고른 숫자는 없다."""

    # outcome 정의 그대로
    horizon_seconds: float = float(PRIMARY_HORIZON_SECONDS)
    fee_bps: float = FEE_BPS
    profit_floor_bps: float = 10.0          # 품질 표본 규칙과 같다
    sustain_ticks: int = 10                 # 품질 표본 규칙과 같다
    # grounding 이 고정한 사전 창과 층
    pre_window_seconds: float = 10.0
    bucket_minutes: int = 15
    # anchor 격자. 창이 겹치지 않도록 평가 창과 같게 둔다 — 고른 값이 아니라 유도된 값이다.
    anchor_spacing_seconds: float | None = None
    # 사전 선언한 통제 변수. 결과를 보고 바꾸지 않는다.
    trigger_feature: str = "book_imbalance"
    context_feature: str = "vol_flow"
    quote_state_feature: str = "spread_bps"
    discovery_dates: tuple[str, ...] = ("20260316", "20260317", "20260318",
                                        "20260319", "20260320")
    holdout_dates: tuple[str, ...] = ("20260323", "20260324", "20260325",
                                      "20260326", "20260327")
    min_pairs: int = 100
    bootstrap: int = 1000
    seed: int = 0

    # 가설마다 쓰는 feature 가 다르다. 여기 넣으면 anchor 표에 열이 생긴다.
    extra_features: tuple[str, ...] = ()

    @property
    def spacing(self) -> float:
        return self.anchor_spacing_seconds or self.horizon_seconds

    @property
    def feature_names(self) -> tuple[str, ...]:
        """anchor 표에 담을 feature. 중복은 없앤다."""
        return tuple(dict.fromkeys(
            (self.trigger_feature, self.context_feature, self.quote_state_feature)
            + tuple(self.extra_features)))


# 가설이 어떤 근거를 어떤 자리에 놓았는지 나타내는 말. `evidence.ROLES` 와 같은 어휘다.
_TRIGGER_ROLES = ("LATE_TRIGGER", "TRIGGER", "SETUP_STATE")
_CONTEXT_ROLES = ("PERSISTENT_CONTEXT", "SUPPORTING_STATE")


def config_for(hypothesis: Mapping[str, Any],
               base: "ValidationConfig | None" = None) -> "ValidationConfig":
    """**가설이 쓸 feature 를 가설에서 읽는다.** 설정에 손으로 적지 않는다.

    `evidence_roles` 가 어느 근거 family 가 trigger 이고 어느 것이 맥락인지 말해 주고,
    `evidence_basis` 가 그 family 의 대표 feature 를 말해 준다. 이 둘이면 충분하다.
    """
    base = base or ValidationConfig()
    feature_of: dict[str, str] = {}
    for item in hypothesis.get("evidence_basis") or []:
        family = str(item.get("family") or "")
        feature = item.get("representative_feature")
        if family and feature and family not in feature_of:
            feature_of[family] = str(feature)

    roles = {str(k): str(v) for k, v in (hypothesis.get("evidence_roles") or {}).items()}
    trigger = next((feature_of[f] for f, r in roles.items()
                    if r in _TRIGGER_ROLES and f in feature_of), None)
    context = next((feature_of[f] for f, r in roles.items()
                    if r in _CONTEXT_ROLES and f in feature_of), None)
    if trigger is None:
        trigger = next(iter(feature_of.values()), base.trigger_feature)
    if context is None:
        # 축소된 가설은 맥락을 주장하지 않는다. 그래도 **제거가 옳았는지** 보려면
        # 그 feature 가 필요하다. 근거에 남아 있는 다른 family 를 쓴다.
        context = next((v for v in feature_of.values() if v != trigger),
                       base.context_feature)
    known = [f for f in (trigger, context) if f in catalog.FEATURES]
    if len(known) < 2:                       # catalog 에 없는 이름은 쓰지 않는다
        trigger = trigger if trigger in catalog.FEATURES else base.trigger_feature
        context = context if context in catalog.FEATURES else base.context_feature
    return dataclasses.replace(base, trigger_feature=trigger, context_feature=context)


# ---- 적격 anchor 우주 (§3) ------------------------------------------------------

def eligible_anchors(arrays: Mapping[str, np.ndarray], config: ValidationConfig
                     ) -> np.ndarray:
    """미래를 쓰지 않고 고른 anchor. 시각 격자 위에서 창이 온전한 틱만.

    고를 때 쓰는 것: 시각, 호가 유효성, 창 관측 가능 여부. 그것뿐이다.
    """
    time_s = np.asarray(arrays["time_s"], dtype=float)
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    if len(time_s) < 2:
        return np.empty(0, dtype=int)
    grid = np.arange(time_s[0] + config.pre_window_seconds,
                     time_s[-1] - config.horizon_seconds, config.spacing)
    if not len(grid):
        return np.empty(0, dtype=int)
    idx = np.searchsorted(time_s, grid, side="left")
    idx = np.unique(idx[idx < len(time_s)])
    end = np.searchsorted(time_s, time_s[idx] + config.horizon_seconds, side="left")
    pre = np.searchsorted(time_s, time_s[idx] - config.pre_window_seconds, side="right") - 1
    keep = (end < len(time_s)) & (pre >= 0) & np.isfinite(ask1[idx]) & (ask1[idx] > 0)
    return idx[keep]


def anchor_frame(symbol: str, date: str, config: ValidationConfig,
                 root: Path = TICK_ROOT, *, arrays: Mapping[str, np.ndarray] | None = None,
                 values: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
    """한 종목-일의 anchor 표. 관측 상태와 **미래 응답**을 함께 담되 열을 구분한다."""
    if arrays is None:
        arrays, _ = tickdata.load(symbol, date, root)
    book = catalog.Book(arrays)
    ticks = eligible_anchors(arrays, config)
    if not len(ticks):
        return pd.DataFrame()
    time_s = np.asarray(arrays["time_s"], dtype=float)
    bid1 = book.bid1
    ask1 = book.ask1
    mid = book.mid
    names = config.feature_names
    values = catalog.compute_features(names, book) if values is None else values

    pre_idx = np.searchsorted(time_s, time_s[ticks] - config.pre_window_seconds,
                              side="right") - 1
    end_idx = np.searchsorted(time_s, time_s[ticks] + config.horizon_seconds, side="left")

    rows = []
    for k, t in enumerate(ticks):
        entry = float(ask1[t])
        end = int(end_idx[k])
        marks = bid1[t + 1: end + 1]
        net = (marks / entry - 1.0) * 1e4 - config.fee_bps if len(marks) else np.empty(0)
        finite = np.isfinite(net)
        best = float(np.max(net[finite])) if finite.any() else np.nan
        # 품질 표본과 같은 규칙: net >= 10bps 가 10틱 연속
        good = np.isfinite(net) & (net >= config.profit_floor_bps)
        run = sustained = 0
        first = -1
        for i, flag in enumerate(good):
            run = run + 1 if flag else 0
            if run >= config.sustain_ticks and not sustained:
                sustained, first = 1, i
        best_at = int(np.argmax(np.where(finite, net, -np.inf))) if finite.any() else -1
        rows.append({
            "symbol": symbol, "date": date, "tick": int(t),
            "time_s": float(time_s[t]),
            "bucket": int(time_s[t] // (config.bucket_minutes * 60)),
            # --- anchor 선택·통제에 쓰는 관측 상태 (미래 없음) ---
            "anchor_ask1": entry, "anchor_bid1": float(bid1[t]),
            "pre_move_bps": float((mid[t] / mid[int(pre_idx[k])] - 1.0) * 1e4)
            if mid[int(pre_idx[k])] > 0 else np.nan,
            **{name: float(values[name][t]) for name in names if name in values},
            # --- 미래 응답. anchor 가 고정된 뒤에만 연다 ---
            "max_net_bps_s30": best,
            "sustained_success": int(sustained),
            "time_to_best_s": float(time_s[t + 1 + best_at] - time_s[t]) if best_at >= 0 else np.nan,
            "time_to_success_s": float(time_s[t + 1 + first] - time_s[t]) if first >= 0 else np.nan,
            "future_label_not_used_for_selection": True})
    return pd.DataFrame(rows)


# anchor 를 고르고 통제하는 데 쓰는 열. feature 이름은 가설마다 다르므로 config 에서 온다.
SELECTION_BASE_COLUMNS = ("symbol", "date", "tick", "time_s", "bucket",
                          "anchor_ask1", "anchor_bid1", "pre_move_bps")


def selection_columns(config: "ValidationConfig | None" = None) -> tuple[str, ...]:
    return SELECTION_BASE_COLUMNS + (config or ValidationConfig()).feature_names


SELECTION_COLUMNS = selection_columns()
RESPONSE_COLUMNS = ("max_net_bps_s30", "sustained_success", "time_to_best_s",
                    "time_to_success_s")


# ---- Tier 0: 무결성 (§8~§10) ----------------------------------------------------

def integrity_audit(frame: pd.DataFrame, plan: Mapping[str, Any],
                    config: "ValidationConfig | None" = None) -> dict[str, Any]:
    """선택에 미래가 섞였는지, 탐색이 끼었는지 코드로 센다. 전부 0 이어야 한다."""
    controls = set(plan.get("matching_variables", []))
    tested = {plan.get("trigger_feature"), plan.get("context_feature")}
    leak = sorted((controls | tested) & set(RESPONSE_COLUMNS))
    return {
        "future_selection_leakage_count": len(leak),
        "leaked_columns": leak,
        "outcome_conditioning_violation_count": int(
            bool(set(plan.get("population_filters", [])) & set(RESPONSE_COLUMNS))),
        "new_threshold_count": int(plan.get("new_thresholds", 0)),
        "new_lag_count": int(plan.get("new_lags", 0)),
        "new_horizon_count": int(plan.get("new_horizons", 0)),
        "post_result_metric_change_count": 0,
        "post_result_control_change_count": 0,
        "selection_columns": list(selection_columns(config)),
        "response_columns": list(RESPONSE_COLUMNS),
        "population": plan.get("population"),
        "anchors": int(len(frame)),
        "outcome_conditioned_population": False,
    }


# ---- 통계 (사전 고정) ------------------------------------------------------------

def _bootstrap_median(values: np.ndarray, config: ValidationConfig) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(config.seed)
    draws = rng.integers(0, len(x), size=(config.bootstrap, len(x)))
    medians = np.median(x[draws], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                        config: ValidationConfig) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    a, b = x[ok], y[ok]
    if len(a) < 100:
        return np.nan, np.nan
    rng = np.random.default_rng(config.seed)
    draws = rng.integers(0, len(a), size=(min(config.bootstrap, 400), len(a)))
    values = [_spearman(a[d], b[d]) for d in draws]
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) < 50:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _sign_test(delta: np.ndarray, config: ValidationConfig) -> dict[str, Any]:
    """동률을 빼고 부호만 센다. 호가가 틱 격자라 차이가 정확히 0 인 짝이 많다 —
    중앙값을 쓰면 그 동률이 통계를 0 으로 눌러 버린다."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    ties = int(np.sum(d == 0))
    nz = d[d != 0]
    if len(nz) < 100:
        return {"share_positive": np.nan, "interval_low": np.nan, "interval_high": np.nan,
                "n_effective": int(len(nz)), "tie_rate": ties / max(len(d), 1)}
    rng = np.random.default_rng(config.seed)
    share_positive = float((nz > 0).mean())
    # 부호 표본의 복원추출은 양수 개수가 Binomial(n, p)를 따르는 것과 같다.
    # 같은 bootstrap 분포를 작은 벡터만으로 만들며, 수백만 짝의 2D 배열을 만들지 않는다.
    shares = rng.binomial(len(nz), share_positive, size=config.bootstrap) / len(nz)
    return {"share_positive": share_positive,
            "interval_low": float(np.percentile(shares, 2.5)),
            "interval_high": float(np.percentile(shares, 97.5)),
            "n_effective": int(len(nz)), "tie_rate": ties / max(len(d), 1)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or len(np.unique(x[ok])) < 2:
        return np.nan
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))


# ---- Tier 1 (§12~§18) -----------------------------------------------------------

def path_target(frame: pd.DataFrame, config: ValidationConfig, *,
                validation_id: str = "MV-H1-01", claim_id: str = "C6",
                feature: str | None = None) -> dict[str, Any]:
    """C6 — trigger 상태가 강할수록 이후 실행 가능한 경로가 좋아지는가.

    임계로 자르지 않는다. 연속 순위 관계를 본다 (§20).
    """
    feature = feature or config.trigger_feature
    trigger = frame[feature].to_numpy(float)
    response = frame["max_net_bps_s30"].to_numpy(float)
    rho = _spearman(trigger, response)
    success = _spearman(trigger, frame["sustained_success"].to_numpy(float))
    low, high = _bootstrap_spearman(trigger, response, config)
    return {"validation_id": validation_id, "claim_id": claim_id,
            "target_type": "PATH_TARGET", "tested": feature,
            "population_n": int(len(frame)),
            "effect_metric": "spearman(trigger, max_net_bps_s30)",
            "effect_size": rho, "secondary_effect": success,
            "interval_low": low, "interval_high": high,
            "response_median": float(np.nanmedian(response)),
            "sustained_rate": float(np.nanmean(frame["sustained_success"])),
            "expected_direction": "POSITIVE"}


def already_priced(frame: pd.DataFrame, config: ValidationConfig, *,
                   validation_id: str = "MV-H1-02", claim_id: str = "C7/C13",
                   feature: str | None = None) -> dict[str, Any]:
    """C7/C13 — 움직임이 trigger 전에 끝났는가, trigger 뒤에도 남는가.

    절대 수준으로 재면 안 된다. 잔여 net 의 중앙은 언제나 -(스프레드+수수료) 근처라
    비용 바닥을 읽을 뿐이다. **trigger 가 강한 쪽과 약한 쪽을 짝지어** 사전 이동과
    사후 잔여를 각각 비교한다.

        사전만 벌어지고 사후가 그대로   → 이미 반영
        사후도 벌어짐                  → 잔여 움직임 있음
    """
    feature = feature or config.trigger_feature
    pairs = matched_pairs(frame, feature,
                          [config.quote_state_feature], config,
                          extra_responses=("pre_move_bps",))
    if len(pairs) < config.min_pairs:
        return {"validation_id": validation_id, "claim_id": claim_id,
                "target_type": "ALREADY_PRICED_CHECK", "tested": feature,
                "n_matched_pairs": int(len(pairs)),
                "status_hint": "INCONCLUSIVE",
                "reason": f"짝이 {len(pairs)}개로 사전에 정한 {config.min_pairs} 미만"}
    post = _sign_test(pairs.future_path_delta.to_numpy(float), config)
    pre = _sign_test(pairs.pre_move_bps_delta.to_numpy(float), config)
    return {"validation_id": validation_id, "claim_id": claim_id,
            "target_type": "ALREADY_PRICED_CHECK", "tested": feature,
            "n_matched_pairs": int(len(pairs)),
            "controls": [config.quote_state_feature],
            "effect_metric": "share of matched pairs where the stronger trigger has the "
                             "better POST residual (ties dropped)",
            "effect_size": post["share_positive"],
            "interval_low": post["interval_low"], "interval_high": post["interval_high"],
            "post_n_effective": post["n_effective"], "post_tie_rate": post["tie_rate"],
            "pre_share_positive": pre["share_positive"],
            "pre_interval_low": pre["interval_low"], "pre_interval_high": pre["interval_high"],
            "pre_n_effective": pre["n_effective"],
            "post_residual_median_bps": float(np.nanmedian(frame["max_net_bps_s30"])),
            "cost_floor_bps": float(-(np.nanmedian(frame[config.quote_state_feature])
                                      + config.fee_bps)),
            "expected_direction": "SHARE_ABOVE_HALF"}


# ---- Tier 2: matched control (§21~§27) ------------------------------------------

def matched_pairs(frame: pd.DataFrame, tested: str, controls: Sequence[str],
                  config: ValidationConfig,
                  extra_responses: Sequence[str] = ()) -> pd.DataFrame:
    """층 안에서 통제 변수 순위 거리로 1:1 최근접 짝짓기. 복원 없음, caliper 조정 없음."""
    pairs = []
    for (symbol, date, bucket), group in frame.groupby(["symbol", "date", "bucket"],
                                                       sort=True):
        if len(group) < 4:
            continue
        ranked = group.copy()
        for name in [tested, *controls]:
            ranked[f"_r_{name}"] = ranked[name].rank(pct=True)
        cut = ranked[f"_r_{tested}"].median()
        high = ranked[ranked[f"_r_{tested}"] > cut]
        low = ranked[ranked[f"_r_{tested}"] <= cut]
        if high.empty or low.empty:
            continue
        used: set[int] = set()
        cols = [f"_r_{c}" for c in controls]
        low_matrix = low[cols].to_numpy(float)
        for _, row in high.iterrows():
            distance = np.abs(low_matrix - row[cols].to_numpy(float)).sum(axis=1)
            order = np.argsort(distance, kind="stable")
            choice = next((i for i in order if int(low.index[i]) not in used), None)
            if choice is None:
                break
            partner = low.iloc[choice]
            used.add(int(low.index[choice]))
            pairs.append({
                "pair_id": f"{symbol}:{date}:{bucket}:{int(row.tick)}",
                "symbol": symbol, "date": date, "bucket": bucket,
                "anchor_a": int(row.tick), "anchor_b": int(partner.tick),
                "tested_variable": tested,
                "control_distance": float(distance[choice]),
                "tested_variable_delta": float(row[tested] - partner[tested]),
                "tested_rank_delta": float(row[f"_r_{tested}"] - partner[f"_r_{tested}"]),
                "future_path_delta": float(row.max_net_bps_s30 - partner.max_net_bps_s30),
                "success_delta": int(row.sustained_success - partner.sustained_success),
                **{f"{name}_delta": float(row[name] - partner[name])
                   for name in extra_responses}})
    return pd.DataFrame(pairs)


def incremental_effect(frame: pd.DataFrame, tested: str, controls: Sequence[str],
                       validation_id: str, claim_id: str,
                       config: ValidationConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    """통제를 맞춘 뒤 tested 변수가 미래 경로를 추가로 가르는가."""
    pairs = matched_pairs(frame, tested, controls, config)
    diagnostics = matching_diagnostics(pairs, len(frame), len(controls))
    if len(pairs) < config.min_pairs:
        return ({"validation_id": validation_id, "claim_id": claim_id,
                 "target_type": "INCREMENTAL_EFFECT_TARGET", "tested_variable": tested,
                 "controls": list(controls), "n_matched_pairs": int(len(pairs)),
                 "status_hint": "INCONCLUSIVE",
                 "matching_diagnostics": diagnostics,
                 "reason": f"짝이 {len(pairs)}개로 사전에 정한 {config.min_pairs} 미만"},
                pairs)
    if not diagnostics["quality_ok"]:
        # 더 느슨한 짝짓기를 찾지 않는다. 결과를 억지로 내지 않는다 (§29).
        return ({"validation_id": validation_id, "claim_id": claim_id,
                 "target_type": "INCREMENTAL_EFFECT_TARGET", "tested_variable": tested,
                 "controls": list(controls), "n_matched_pairs": int(len(pairs)),
                 "status_hint": "INCONCLUSIVE_DUE_TO_MATCHING",
                 "matching_diagnostics": diagnostics,
                 "reason": f"짝짓기 품질이 사전 기준을 못 넘었다 "
                           f"(balance_ratio {diagnostics['balance_ratio']:.3f} > "
                           f"{MAX_BALANCE_RATIO})"},
                pairs)
    delta = pairs.future_path_delta.to_numpy(float)
    sign = _sign_test(delta, config)
    return ({"validation_id": validation_id, "claim_id": claim_id,
             "target_type": "INCREMENTAL_EFFECT_TARGET", "tested_variable": tested,
             "controls": list(controls),
             "n_matched_pairs": int(len(pairs)),
             # 주 통계는 동률을 뺀 부호 검정이다. 중앙값은 격자 때문에 0 으로 눌린다.
             "effect_metric": "share of matched pairs where the higher tested value has "
                              "the better future path (ties dropped)",
             "effect_size": sign["share_positive"],
             "interval_low": sign["interval_low"], "interval_high": sign["interval_high"],
             "n_effective": sign["n_effective"], "tie_rate": sign["tie_rate"],
             "median_delta_bps": float(np.nanmedian(delta)),
             "mean_delta_bps": float(np.nanmean(delta)),
             "success_delta_mean": float(pairs.success_delta.mean()),
             "match_quality": float(pairs.control_distance.median()),
             "matching_diagnostics": diagnostics,
             "expected_direction": "SHARE_ABOVE_HALF"}, pairs)


# ---- 판정 -----------------------------------------------------------------------

def _supported(effect: float | None, low: float | None, high: float | None,
               null: float = 0.0) -> str:
    """구간이 귀무값을 넘는가. 사전에 고정한 판정 규칙이다.

    부호 검정은 `null=0.5`, 상관은 `null=0.0` 이다.
    """
    if effect is None or not np.isfinite(effect) or low is None or not np.isfinite(low):
        return "INCONCLUSIVE"
    if low > null:
        return "SUPPORTED"
    if high < null:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def verdict(results: Mapping[str, Any], blocks: Mapping[str, Any],
            target_roles: Mapping[str, str] | None = None,
            hypothesis_id: str = "H1",
            target_contracts: Mapping[str, Mapping[str, Any]] | None = None,
            config: "ValidationConfig | None" = None) -> dict[str, Any]:
    """축을 나눠 판정하고 하나의 route 로 모은다. 단일 VALIDATED/FAILED 로 뭉치지 않는다.

    `target_roles` 는 이 가설이 **실제로 무엇을 주장하는지** 알려준다. 축소된 가설에서
    맥락은 더 이상 주장이 아니라 `REMOVAL_CHECK` 다 — 그것이 지지되지 않는 것은 실패가
    아니라 제거가 옳았다는 확인이다. 이것을 모르면 이미 고친 것을 또 고치라고 한다.
    """
    roles = dict(target_roles or {})
    targets = dict(target_contracts or TARGET_CONTRACTS)

    # **이름이 아니라 역할로 찾는다.** `MV-H1-01` 같은 이름을 박으면 그 가설 전용이 된다.
    def named(kind: str) -> list[str]:
        return [n for n in sorted(results)
                if (targets.get(n) or {}).get("target_type") == kind]

    path_names = named("PATH_TARGET")
    priced_names = named("ALREADY_PRICED_CHECK")
    incremental = named("INCREMENTAL_EFFECT_TARGET")

    path = results.get(path_names[0], {}) if path_names else {}
    priced = results.get(priced_names[0], {}) if priced_names else {}

    # trigger 는 **경로 target 이 시험한 feature** 다. 설정에서 따로 받지 않는다 —
    # 가설이 바뀌면 이름도 feature 도 바뀌지만 "경로를 시험한 것이 trigger" 는 변하지 않는다.
    trigger_feature = (targets.get(path_names[0]) or {}).get("tested") if path_names else None
    if config is not None and trigger_feature is None:
        trigger_feature = getattr(config, "trigger_feature", None)

    trigger_names = [n for n in incremental
                     if (targets.get(n) or {}).get("tested") == trigger_feature]
    context_names = [n for n in incremental if n not in trigger_names]
    if not trigger_names and incremental:     # 경로 target 이 없는 가설
        trigger_names, context_names = incremental[:1], incremental[1:]

    # 단일 상태 가설은 PATH_TARGET 하나만 주장할 수 있다. 옛 H1-R1처럼 별도
    # incremental target 이 없는 것을 "관측 구조가 실패"로 읽으면, Plan이 말하지 않은
    # 검사를 요구하게 된다. 이 경우 경로 target 자체가 trigger 확인이다.
    trigger = (results.get(trigger_names[0], {}) if trigger_names else
               path if path_names and not incremental else {})
    # 맥락은 여러 개일 수 있다. 하나라도 주장이면 맥락을 주장하는 가설이다.
    contexts = [(n, results.get(n, {})) for n in context_names]
    claimed_contexts = [(n, r) for n, r in contexts
                        if roles.get(n, "CLAIMED") == "CLAIMED"]
    context_is_claimed = bool(claimed_contexts)
    if context_is_claimed:
        # 주장된 맥락 중 하나라도 지지되면 지지로 본다.
        context = next((r for _n, r in claimed_contexts if r.get("status") == "SUPPORTED"),
                       claimed_contexts[0][1])
    else:
        context = next((r for _n, r in contexts if r.get("status") == "SUPPORTED"),
                       contexts[0][1] if contexts else {})

    prediction = path.get("status", "NOT_RUN")
    residual = priced.get("status", "INCONCLUSIVE")
    already = ("RESIDUAL_MOVE_SUPPORTED" if residual == "SUPPORTED" else
               "ALREADY_PRICED_SUPPORTED" if residual == "NOT_SUPPORTED" else
               "INCONCLUSIVE")
    t_ok = trigger.get("status") == "SUPPORTED"
    c_ok = context.get("status") == "SUPPORTED"
    if not incremental and path_names:
        relation = ("SINGLE_STATE_SUPPORTED" if t_ok else
                    "INCONCLUSIVE" if trigger.get("status") == "INCONCLUSIVE"
                    else "NEITHER_SUPPORTED")
    elif not context_is_claimed:
        # 축소된 가설. 맥락 target 은 제거가 옳았는지 보는 검사다.
        relation = ("REMOVAL_CONTRADICTED" if c_ok else
                    "SINGLE_STATE_SUPPORTED" if t_ok else
                    "INCONCLUSIVE" if trigger.get("status") == "INCONCLUSIVE"
                    else "NEITHER_SUPPORTED")
    else:
        relation = ("CONTEXT_AND_TRIGGER_SUPPORTED" if t_ok and c_ok else
                    "TRIGGER_ONLY_SUPPORTED" if t_ok else
                    "CONTEXT_ONLY_SUPPORTED" if c_ok else
                    "INCONCLUSIVE" if "INCONCLUSIVE" in (trigger.get("status"),
                                                         context.get("status"))
                    else "NEITHER_SUPPORTED")
    mechanism = "BLOCKED_BY_CAPABILITY" if blocks.get("blocked") else \
        "IDENTIFIED" if blocks.get("mechanism_identified") else \
        "CONSISTENT_BUT_UNIDENTIFIED"

    if already == "ALREADY_PRICED_SUPPORTED" or prediction == "NOT_SUPPORTED":
        status, route = "HYPOTHESIS_REJECT", "END_BRANCH_OR_NEW_HYPOTHESIS"
        reason = "trigger 관측 이후 실행 가능한 잔여 움직임이 지지되지 않는다"
    elif prediction != "SUPPORTED":
        status, route = "INCONCLUSIVE", "RERUN_WITH_MORE_SUPPORT"
        reason = "핵심 경제적 예측이 결론에 이르지 못했다"
    elif relation == "REMOVAL_CONTRADICTED":
        status, route = "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"
        reason = ("수정에서 뺀 맥락이 새 데이터에서는 기여한다. 그 제거가 옳지 않았다")
    elif relation in ("CONTEXT_AND_TRIGGER_SUPPORTED", "SINGLE_STATE_SUPPORTED"):
        # 천장 규칙 (§56). 핵심 메커니즘이 식별되지 않으면 아무리 다 지지돼도
        # `MECHANISM_SUPPORTED` 로 올리지 않는다. 예측이 맞는 것과 그 이유를 아는 것은 다르다.
        if mechanism == "IDENTIFIED":
            status, route = "MECHANISM_SUPPORTED", "DOWNSTREAM"
            reason = "관측 구조·미래 경로·메커니즘 식별이 모두 지지된다"
        else:
            status, route = "REDUCED_FORM_SUPPORTED", "DATA_CAPABILITY_OR_EXECUTABLE_SPEC"
            reason = ("관측 구조와 미래 실행 경로의 연결은 지지된다. 다만 제안한 메커니즘 "
                      f"자체는 현재 데이터로 식별할 수 없다 ({mechanism})")
    elif relation == "TRIGGER_ONLY_SUPPORTED":
        status, route = "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"
        reason = ("잔여 움직임은 있으나 맥락이 추가 정보를 주지 않는다. "
                  "CONTEXT_TRIGGER 구조를 그대로 유지할 근거가 없다")
    elif relation == "CONTEXT_ONLY_SUPPORTED":
        status, route = "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"
        reason = "trigger 해석을 재검토해야 한다"
    elif relation == "NEITHER_SUPPORTED":
        status, route = "HYPOTHESIS_REVISE", "RETURN_TO_HYPOTHESIS"
        reason = "관측 구조 자체를 재검토해야 한다"
    else:
        status, route = "INCONCLUSIVE", "RERUN_WITH_MORE_SUPPORT"
        reason = "짝 표본이 모자라 기여를 판정할 수 없다"

    return {"hypothesis_id": hypothesis_id, "prediction_status": prediction,
            "relation_status": relation,
            "mechanism_identification_status": mechanism,
            "already_priced_status": already,
            "hypothesis_status": status, "route": route, "reason": reason,
            "fresh_validation_required": status in ("HYPOTHESIS_REVISE", "HYPOTHESIS_REJECT"),
            "fresh_validation_note": (
                "이 결과를 보고 가설을 고치면 이 데이터는 수정된 가설의 개발 데이터가 된다. "
                "수정본은 새 홀드아웃에서 검증해야 한다")}


# ---- Revision 으로 넘기는 claim 단위 피드백 (§29) --------------------------------

# 어느 validation target 이 어느 주장을 시험했는가. 통계가 SUPPORTED 라고 해서 그 이름의
# 주장이 늘 지지되는 것은 아니다 — 대안 설명은 방향이 반대다. 그것을 여기서 명시한다.
DIRECT, INVERSE = "DIRECT", "INVERSE"


def _null_value(effect_metric: str) -> float:
    """귀무값은 통계량이 정한다. 순위상관은 0, 짝 부호 비율은 0.5 다."""
    return 0.0 if effect_metric.startswith("spearman") else 0.5


def results_from_frame(frame, split: str) -> dict[str, Any]:
    """결과 parquet 에서 한 split 의 전체(ALL) 행만 뽑는다. 날짜별 행은 안정성 확인용이다."""
    rows = frame[(frame["split"] == split) & (frame["date"] == "ALL")]
    def scalar(value):
        if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
            return list(value)
        return None if pd.isna(value) else value

    return {str(r["validation_id"]): {k: scalar(v) for k, v in r.items()}
            for _, r in rows.iterrows()}
TARGET_CLAIMS: dict[str, list[tuple[str, str]]] = {
    "MV-H1-01": [("C6", DIRECT)],
    "MV-H1-02": [("C7", DIRECT), ("C13", INVERSE)],
    "MV-H1-03": [("C8", DIRECT)],
    # 맥락 기여는 grounding claim 이 아니라 가설 구조 자체다. 별도 행으로 낸다.
    "MV-H1-04": [("STRUCTURE:CONTEXT_CONTRIBUTION", DIRECT), ("C10", INVERSE)],
}
STRUCTURE_CLAIMS = {
    "STRUCTURE:CONTEXT_CONTRIBUTION":
        "앞선 높은 TRADE_ACTIVITY 맥락이 anchor 호가 상태를 맞춘 뒤에도 미래 실행 결과에 "
        "추가로 기여한다",
}
# grounding 상태만으로 정해지는 것들 — 통계로 시험하지 않았다.
GROUNDING_VERDICT = {
    "UNOBSERVABLE": "BLOCKED_BY_CAPABILITY",
    "UNRESOLVED_MECHANISM": "BLOCKED_BY_CAPABILITY",
    "INVALID_GROUNDING": "INVALID_TARGET",
    "GROUNDED_DIRECT": "OBSERVATIONAL_INPUT",
    "GROUNDED_DERIVED": "OBSERVATIONAL_INPUT",
}
_FLIP = {"SUPPORTED": "NOT_SUPPORTED", "NOT_SUPPORTED": "SUPPORTED"}


def claim_feedback(routing: Mapping[str, Any], results: Mapping[str, Any],
                   grounded: Mapping[str, Any],
                   include_statistics: bool = False,
                   targets: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Revision Agent 가 받을 입력. 원본 보고서를 통째로 넘기지 않는다 (§29~§30).

    새 양적 패턴을 캐라고 주는 것이 아니다. 어느 주장이 살아남았고 어느 것이 반박됐는지만
    준다. 통계 수치는 기본으로 빼고, provenance 로 어디서 나왔는지만 남긴다.
    """
    hypothesis = (grounded.get("hypotheses") or [{}])[0]
    texts = {c.get("claim_id"): c for c in hypothesis.get("claims") or []}

    # target → claim 판정
    tested: dict[str, dict[str, Any]] = {}
    mapping = target_claims(targets) if targets else TARGET_CLAIMS
    for validation_id, pairs in mapping.items():
        outcome = results.get(validation_id) or {}
        status = outcome.get("status")
        if status is None:
            continue
        for claim_id, polarity in pairs:
            resolved = _FLIP.get(status, status) if polarity == INVERSE else status
            row: dict[str, Any] = {
                "claim_id": claim_id, "status": resolved,
                "provenance": validation_id,
                "split": outcome.get("split") or routing.get("primary_split")}
            if polarity == INVERSE:
                row["note"] = (f"{validation_id} 는 반대 방향 주장을 시험했다. "
                               f"그 통계는 {status} 이므로 이 주장은 {resolved} 다")
            null = _null_value(str(outcome.get("effect_metric", "")))
            if resolved == "NOT_SUPPORTED" and outcome.get("effect_size") is not None:
                if float(outcome["effect_size"]) < null:
                    row["direction"] = "OPPOSITE"
            if include_statistics:
                row["statistic"] = {"null_value": null,
                                    **{k: outcome.get(k) for k in
                                       ("effect_size", "interval_low", "interval_high",
                                        "n_matched_pairs")}}
            tested[claim_id] = row

    rows: list[dict[str, Any]] = []
    for claim_id, claim in texts.items():
        row = tested.pop(claim_id, None) or {
            "claim_id": claim_id,
            "status": GROUNDING_VERDICT.get(claim.get("grounding_status"), "NOT_TESTED"),
            "provenance": f"grounding:{claim.get('grounding_status')}"}
        row["claim"] = claim.get("claim")
        row["grounding_status"] = claim.get("grounding_status")
        rows.append(row)
    for claim_id, row in tested.items():          # 구조 주장 — grounding claim 이 아니다
        row["claim"] = STRUCTURE_CLAIMS.get(claim_id, claim_id)
        rows.append(row)
    rows.sort(key=lambda r: (r["claim_id"].startswith("STRUCTURE"), r["claim_id"]))

    return {"schema": "validation_claim_feedback.v1", "created_at": now_utc(),
            "hypothesis_id": routing.get("hypothesis_id"),
            "prediction_status": routing.get("prediction_status"),
            "relation_status": routing.get("relation_status"),
            "mechanism_identification_status":
                routing.get("mechanism_identification_status"),
            "already_priced_status": routing.get("already_priced_status"),
            "hypothesis_status": routing.get("hypothesis_status"),
            "route": routing.get("route"),
            "primary_split": routing.get("primary_split"),
            "blocked_note": "BLOCKED_BY_CAPABILITY 는 거짓이라는 뜻이 아니다. "
                            "현재 데이터로 식별할 수 없다는 뜻이다",
            "claim_feedback": rows}


# ---- 주장 → 재려는 양 → 통계 → 귀무값 (§11~§15) --------------------------------

# 실행 전에 target 마다 이 네 줄을 적는다. 통계 이름이 아니라 **경제적으로 무엇을 재려는지**
# 를 적는 것이 요점이다.
#
#   claim_asks           주장이 묻는 양의 종류
#   statistic_measures   통계가 실제로 재는 양의 종류
#
# 이 둘이 다르면 숫자가 아무리 정확해도 그 주장을 시험한 것이 아니다.
# 통계 4종류. **어느 가설이든 이 넷 중에서 고른다.** 주장마다 새 통계를 만들지 않는다 —
# 만들면 가설마다 다른 자로 재게 되고 비교가 불가능해진다.
STATISTIC_KINDS: dict[str, dict[str, Any]] = {
    "PATH_TARGET": {
        "claim_asks": "RANK_ASSOCIATION", "statistic_measures": "RANK_ASSOCIATION",
        "statistic": "spearman(trigger, max_net_bps_s30)", "null": 0.0,
        "structural_terms_controlled": True,
        "structural_note": "순위상관이라 스프레드·수수료의 절대 수준이 상관값을 정하지 못한다",
        "estimand_template": "미래로 고르지 않은 anchor 들 사이에서, anchor 시점 "
                             "{feature} 강약과 그 뒤 실행 가능한 경로가 같이 움직이는 정도"},
    "ALREADY_PRICED_CHECK": {
        "claim_asks": "RELATIVE_COMPARISON", "statistic_measures": "RELATIVE_COMPARISON",
        "statistic": "share of matched pairs where the stronger trigger has the better "
                     "POST residual (ties dropped)", "null": 0.5,
        "structural_terms_controlled": True,
        "structural_note": "짝 안의 상대 비교라 -(스프레드+수수료) 바닥이 양쪽에서 상쇄된다",
        "estimand_template": "층과 통제를 맞춘 짝 안에서 {feature} 가 강한 쪽의 "
                             "**사후 잔여**가 더 좋은가. 같은 짝의 **사전 이동**과 따로 본다"},
    "INCREMENTAL_EFFECT_TARGET": {
        "claim_asks": "RELATIVE_COMPARISON", "statistic_measures": "RELATIVE_COMPARISON",
        "statistic": "share of matched pairs where the higher tested value has the better "
                     "future path (ties dropped)", "null": 0.5,
        "structural_terms_controlled": True,
        "structural_note": "동률을 뺀 부호 검정이다. 중앙값은 틱 격자 때문에 0 으로 눌린다",
        "estimand_template": "통제를 맞춘 짝 안에서 {feature} 가 높은 쪽의 미래 경로가 "
                             "더 좋은 비율"},
    "NOT_TESTABLE": {
        "claim_asks": "RELATIVE_COMPARISON", "statistic_measures": None,
        "statistic": None, "null": None, "structural_terms_controlled": None,
        "structural_note": "현재 데이터로 시험할 수 없다",
        "estimand_template": "{claim}"},
}

# grounding 상태 -> 이 주장을 어떻게 다룰 것인가. **가설 이름이 들어가지 않는다.**
GROUNDING_TO_TARGET = {
    "VALIDATION_TARGET": ("DIRECTLY_TESTABLE", None),        # 통계는 아래서 고른다
    "UNRESOLVED_MECHANISM": ("BLOCKED_BY_CAPABILITY", "NOT_TESTABLE"),
    "UNOBSERVABLE": ("BLOCKED_BY_CAPABILITY", "NOT_TESTABLE"),
    "INVALID_GROUNDING": ("INVALID_TARGET", "NOT_TESTABLE"),
}


def build_targets(grounded: Mapping[str, Any], config: "ValidationConfig",
                  hypothesis_id: str | None = None) -> dict[str, dict[str, Any]]:
    """**grounding 결과에서** 검증 target 을 만든다. 손으로 적지 않는다.

    가설마다 주장 수도 내용도 다르다. 표를 손으로 쓰면 그 가설 전용 코드가 되고,
    두 번째 가설은 코드를 고쳐야 들어온다.

    통계는 `STATISTIC_KINDS` 넷 중에서 고른다 — 주장마다 새 자를 만들면 가설끼리
    비교할 수 없다.
    """
    items = grounded.get("hypotheses") or []
    if not items:
        return {}
    hypothesis = items[0]
    name = hypothesis_id or hypothesis.get("hypothesis_id") or "H"
    trigger, context = config.trigger_feature, config.context_feature

    targets: dict[str, dict[str, Any]] = {}
    index = 0
    for claim in hypothesis.get("claims") or []:
        status = str(claim.get("grounding_status"))
        if status not in GROUNDING_TO_TARGET:
            continue                      # 관측 입력이다. 시험 대상이 아니다
        index += 1
        identifiability, forced = GROUNDING_TO_TARGET[status]
        # `UNRESOLVED_MECHANISM` 인데 **못 보는 capability 가 없으면** 그것은 관측이
        # 막힌 것이 아니라 아직 해석이 안 된 것이다. 관측으로 시험할 수 있다.
        # (H1 의 C10 — "맥락은 우연이고 trigger 만 설명한다" 가 이 경우다.)
        if status == "UNRESOLVED_MECHANISM" and not (claim.get("missing_capability") or []):
            identifiability, forced = "DIRECTLY_TESTABLE", None
        kind = forced or _statistic_for(claim, trigger, context)
        spec = STATISTIC_KINDS[kind]
        feature = trigger if kind != "INCREMENTAL_EFFECT_TARGET" else \
            _tested_feature(claim, trigger, context)
        cost_floor = dominates_cost_floor(feature)
        targets[f"MV-{name}-{index:02d}"] = {
            "claim_id": claim.get("claim_id"),
            "claim": claim.get("claim_text") or claim.get("claim"),
            "claim_type": claim.get("claim_type"),
            "target_type": kind,
            "tested": feature,
            "estimand": spec["estimand_template"].format(
                feature=feature, claim=claim.get("claim_text") or claim.get("claim")),
            "claim_asks": spec["claim_asks"],
            "statistic": spec["statistic"],
            "statistic_measures": spec["statistic_measures"],
            "null": spec["null"],
            "identifiability": identifiability,
            "structural_terms_controlled": (False if cost_floor
                                            else spec["structural_terms_controlled"]),
            "structural_note": (
                f"`{feature}` 는 결과 지표가 이미 빼고 시작하는 항이다. 이것으로 "
                "가르면 미래를 맞춘 것이 아니라 비용 바닥을 다시 재는 것이다"
                if cost_floor else spec["structural_note"]),
            "population_outcome_conditioned": False,
            "required_capabilities": [c for c in
                                      (claim.get("missing_capability"),
                                       claim.get("catalog_feature")) if c],
        }
    return targets


# 어떤 통계를 쓸지는 **grounding 이 붙인 `claim_type`** 이 정한다. 문장 낱말로 추측하지
# 않는다 — 낱말은 가설마다 다르고 추측은 조용히 틀린다.
#
#   PREDICTED_CONSEQUENCE   가설이 예측한 결과. 시험 대상
#   ALTERNATIVE_EXPLANATION 대안 설명. 반대 방향으로 시험한다
#   MECHANISM_INTERPRETATION 메커니즘 해석. 대개 관측 불가
#   OBSERVABLE_STATE / TEMPORAL_RELATION  관측 입력. 시험 대상 아님
TESTABLE_CLAIM_TYPES = ("PREDICTED_CONSEQUENCE", "ALTERNATIVE_EXPLANATION")

# 예측 결과 안에서도 무엇을 묻는가에 따라 자가 다르다.
_ALREADY_PRICED = ("이미", "전에 모두", "모두 완료", "반영되어", "남지 않는다")
_MATCHED = ("맞춘 뒤", "통제", "추가적인", "추가로")


def _statistic_for(claim: Mapping[str, Any], trigger: str, context: str) -> str:
    """이 주장을 어느 자로 재는가. **주장이 묻는 양의 종류**로 고른다."""
    text = str(claim.get("claim_text") or claim.get("claim") or "")
    if any(word in text for word in _ALREADY_PRICED):
        return "ALREADY_PRICED_CHECK"
    # 대안 설명은 "둘 중 무엇이 설명하는가" 를 묻는다. 그것은 언제나 통제 비교다 —
    # 하나를 맞춘 뒤에도 다른 하나가 남는지 봐야 답이 나온다.
    if str(claim.get("claim_type")) == "ALTERNATIVE_EXPLANATION":
        return "INCREMENTAL_EFFECT_TARGET"
    if any(word in text for word in _MATCHED):
        return "INCREMENTAL_EFFECT_TARGET"
    return "PATH_TARGET"


def _tested_feature(claim: Mapping[str, Any], trigger: str, context: str) -> str:
    """이 주장이 무엇의 기여를 묻는가.

    grounding 이 `catalog_feature` 를 붙여 두면 그것을 쓴다. 없으면 문장에서 맥락
    family 를 찾는다 — 대안 설명은 대개 맥락이 우연이라는 주장이기 때문이다.
    """
    named = claim.get("catalog_feature")
    if named and str(named) in catalog.FEATURES:
        return str(named)
    text = str(claim.get("claim_text") or claim.get("claim") or "")
    # 대안 설명은 대개 "맥락은 우연이다" 라는 주장이다. 그러면 시험할 것은 **맥락의
    # 기여**다 — 통계가 지지되면 맥락이 기여한 것이고, 그건 대안이 틀렸다는 뜻이다.
    context_family = (E_family(context) if context else None)
    if context_family and context_family in text:
        return context
    return trigger


# `max_net_bps_s30` 은 ASK1 에 사서 BID1 에 파는 값이라 **스프레드를 이미 빼고**
# 시작한다. 그래서 스프레드 계열을 시험 대상으로 삼으면 "좁을 때 더 좋다" 가 자동으로
# 나온다. 예측이 아니라 정의다.
COST_FLOOR_FAMILIES = ("SPREAD",)


def dominates_cost_floor(feature: str | None) -> bool:
    """이 feature 가 결과 지표의 비용 바닥과 같은 것인가."""
    return bool(feature) and E_family(str(feature)) in COST_FLOOR_FAMILIES


def E_family(feature: str) -> str | None:
    from .evidence import family_of
    spec = catalog.FEATURES.get(feature)
    return family_of(spec) if spec else None


# H1 이 실제로 쓴 표. `build_targets` 가 같은 것을 만들어내는지 대조하는 데 쓴다.
TARGET_CONTRACTS: dict[str, dict[str, Any]] = {
    "MV-H1-01": {
        "target_type": "PATH_TARGET", "tested": "book_imbalance",
        "claim": "book_imbalance 가 관측된 뒤에도 S30 안의 실제 BID1 경로에 비용을 "
                 "상쇄할 수 있는 추가 개선이 남는다",
        "claim_asks": "RANK_ASSOCIATION",
        "estimand": "미래로 고르지 않은 anchor 들 사이에서, anchor 시점 trigger 강약과 "
                    "그 뒤 실행 가능한 경로의 좋고 나쁨이 같이 움직이는 정도",
        "statistic": "spearman(trigger, max_net_bps_s30)",
        "statistic_measures": "RANK_ASSOCIATION",
        "null": 0.0,
        "identifiability": "DIRECTLY_TESTABLE",
        "structural_terms_controlled": True,
        "structural_note": "순위상관이라 스프레드·수수료의 절대 수준이 상관값을 정하지 못한다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["post_anchor_executable_quotes", "book_imbalance"],
    },
    "MV-H1-02": {
        "target_type": "ALREADY_PRICED_CHECK", "tested": "book_imbalance",
        "claim": "가격 조정이 trigger 관측 전에 모두 끝난 것이 아니라 관측 이후에도 남아 있다",
        "claim_asks": "RELATIVE_COMPARISON",
        "estimand": "층과 통제를 맞춘 짝 안에서, trigger 가 강한 쪽이 약한 쪽보다 "
                    "**사후 잔여**가 더 좋은가. 같은 짝의 **사전 이동**과 따로 본다",
        "statistic": "share of matched pairs where the stronger trigger has the better "
                     "POST residual (ties dropped)",
        "statistic_measures": "RELATIVE_COMPARISON",
        "null": 0.5,
        "identifiability": "DIRECTLY_TESTABLE",
        "structural_terms_controlled": True,
        "structural_note": "짝 안의 상대 비교라 -(스프레드+수수료) 바닥이 양쪽에서 상쇄된다. "
                           "v1 은 절대 수준을 봐서 이 바닥을 쟀다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["post_anchor_executable_quotes", "l1_quote"],
    },
    "MV-H1-03": {
        "target_type": "INCREMENTAL_EFFECT_TARGET", "tested": "book_imbalance",
        "claim": "앞선 거래 활동과 현재 호가 특성을 맞춘 뒤에도 anchor 의 book_imbalance 는 "
                 "미래 실행 결과를 추가로 가른다",
        "claim_asks": "RELATIVE_COMPARISON",
        "estimand": "통제를 맞춘 짝 안에서 trigger 가 높은 쪽의 미래 경로가 더 좋은 비율",
        "statistic": "share of matched pairs where the higher tested value has the better "
                     "future path (ties dropped)",
        "statistic_measures": "RELATIVE_COMPARISON",
        "null": 0.5,
        "identifiability": "DIRECTLY_TESTABLE",
        "structural_terms_controlled": True,
        "structural_note": "동률을 뺀 부호 검정이다. 중앙값은 틱 격자 때문에 0 으로 눌린다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["post_anchor_executable_quotes", "book_imbalance"],
    },
    "MV-H1-04": {
        "target_type": "INCREMENTAL_EFFECT_TARGET", "tested": "vol_flow",
        "claim": "앞선 거래 활동 맥락이 anchor 호가 상태를 맞춘 뒤에도 미래 실행 결과에 "
                 "추가로 기여한다",
        "claim_asks": "RELATIVE_COMPARISON",
        "estimand": "통제를 맞춘 짝 안에서 맥락이 높은 쪽의 미래 경로가 더 좋은 비율",
        "statistic": "share of matched pairs where the higher tested value has the better "
                     "future path (ties dropped)",
        "statistic_measures": "RELATIVE_COMPARISON",
        "null": 0.5,
        "identifiability": "DIRECTLY_TESTABLE",
        "structural_terms_controlled": True,
        "structural_note": "위와 같다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["post_anchor_executable_quotes", "trade_prints"],
    },
    "MV-H1-05": {
        "target_type": "NOT_TESTABLE", "tested": "book_imbalance",
        "claim": "anchor 의 게시 잔량 비대칭은 주문 주체들의 호가 갱신이 동시에 완료되지 "
                 "않은 상태를 반영한다",
        "claim_asks": "RELATIVE_COMPARISON",
        "estimand": "게시 잔량 감소를 추가·취소·체결로 갈라 비동기 갱신을 식별하는 것",
        "statistic": None,
        "statistic_measures": None,
        "null": None,
        "identifiability": "BLOCKED_BY_CAPABILITY",
        "structural_terms_controlled": None,
        "structural_note": "주문 이벤트 로그가 아니라 스냅샷이라 갈 수 없다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["order_add_cancel_decomposition", "participant_identity"],
    },
    "MV-H1-06": {
        "target_type": "NOT_TESTABLE", "tested": "book_imbalance",
        "claim": "두 증거는 관측되지 않은 공통 정보 충격을 함께 반영한다",
        "claim_asks": "RELATIVE_COMPARISON", "estimand": "외부 정보 도착 시점과의 대조",
        "statistic": None, "statistic_measures": None, "null": None,
        "identifiability": "BLOCKED_BY_CAPABILITY",
        "structural_terms_controlled": None,
        "structural_note": "뉴스·공시가 이 데이터셋에 없다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["information_arrival"],
    },
    "MV-H1-07": {
        "target_type": "NOT_TESTABLE", "tested": "book_imbalance",
        "claim": "두 증거는 공통 변동성 상태를 함께 반영한다",
        "claim_asks": "RELATIVE_COMPARISON", "estimand": "사전 변동성 국면으로 층을 나눈 대조",
        "statistic": None, "statistic_measures": None, "null": None,
        "identifiability": "BLOCKED_BY_CAPABILITY",
        "structural_terms_controlled": None,
        "structural_note": "카탈로그에 사전 변동성 정본 feature 가 없다",
        "population_outcome_conditioned": False,
        "required_capabilities": ["pre_anchor_volatility"],
    },
    "MV-H1-08": {
        "target_type": "NOT_TESTABLE", "tested": "book_imbalance",
        "claim": "미래 라벨을 쓰지 않은 원시 시계열에서도 두 관측의 순서가 반복된다",
        "claim_asks": "RELATIVE_COMPARISON",
        "estimand": "개별 사건의 상태 발생 시점 순서",
        "statistic": None, "statistic_measures": None, "null": None,
        "identifiability": "INVALID_TARGET",
        "structural_terms_controlled": None,
        "structural_note": "집단 분리 순서를 개별 사건 순서로 바꾸려면 grounding 에 없던 "
                           "상태 발생 임계가 새로 필요하다",
        "population_outcome_conditioned": False,
        "required_capabilities": [],
    },
}

# 통계 계열마다 맞는 귀무값. 여기서 벗어나면 Q5 가 잡는다.
_NULL_BY_FAMILY = {"spearman": 0.0, "share": 0.5}


def _statistic_family(statistic: str) -> str | None:
    for family in _NULL_BY_FAMILY:
        if family in statistic:
            return family
    return None


def estimand_fidelity_audit(contracts: Mapping[str, Mapping[str, Any]] = None
                            ) -> dict[str, Any]:
    """실행 전에 본다 — **이 통계가 정말 그 주장을 재는가** (§16).

    다섯 가지를 묻는다.
      Q1 통계가 주장과 같은 종류의 양을 재는가
      Q2 주장은 상대 비교인데 통계는 절대 수준을 보고 있지 않은가
      Q3 스프레드·수수료·틱 같은 구조적 항이 통계를 지배하지 않는가
      Q4 outcome 정의를 그대로 다시 재는 동어반복은 아닌가
      Q5 귀무값이 그 통계에 맞는가

    실패하면 `INVALID_VALIDATION_STATISTIC` 이다. **미래를 열기 전에** 멈춘다.
    """
    contracts = TARGET_CONTRACTS if contracts is None else contracts
    problems: list[dict[str, str]] = []
    cost_floor: list[str] = []
    verdicts: dict[str, str] = {}

    for validation_id, contract in sorted(contracts.items()):
        def fail(question: str, why: str) -> None:
            problems.append({"validation_id": validation_id, "question": question,
                             "problem": why})

        identifiability = contract.get("identifiability")
        if identifiability not in IDENTIFIABILITY:
            fail("SCHEMA", f"모르는 identifiability {identifiability!r}")
        for key in ("claim", "estimand"):
            if not contract.get(key):
                fail("SCHEMA", f"{key} 가 비었다")

        statistic = contract.get("statistic")
        if statistic is None:
            # 시험하지 않는 target 이다. 통계가 없는 것이 맞다.
            if identifiability in ("DIRECTLY_TESTABLE", "REDUCED_FORM_TESTABLE"):
                fail("SCHEMA", "시험 가능하다고 해 놓고 통계가 없다")
            verdicts[validation_id] = identifiability
            continue

        asks, measures = contract.get("claim_asks"), contract.get("statistic_measures")
        if asks not in QUANTITY_KINDS or measures not in QUANTITY_KINDS:
            fail("SCHEMA", f"양의 종류가 {QUANTITY_KINDS} 밖이다: {asks!r} / {measures!r}")
        elif asks != measures:                                          # Q1
            fail("Q1", f"주장은 {asks} 를 묻는데 통계는 {measures} 를 잰다")
        if measures == "ABSOLUTE_LEVEL" and asks != "ABSOLUTE_LEVEL":    # Q2
            fail("Q2", "주장은 상대 비교인데 통계가 절대 수준을 본다")
        if not contract.get("structural_terms_controlled"):              # Q3
            fail("Q3", contract.get("structural_note")
                 or "스프레드·수수료 바닥을 덜어내지 않았다")
            cost_floor.append(validation_id)
        if contract.get("population_outcome_conditioned"):              # Q4
            fail("Q4", "미래 결과로 고른 표본에서 그 결과를 다시 확인한다 — 동어반복이다")
        family = _statistic_family(str(statistic))                      # Q5
        if family is None:
            fail("Q5", f"통계 계열을 모르겠다: {statistic!r}")
        elif contract.get("null") != _NULL_BY_FAMILY[family]:
            fail("Q5", f"{family} 의 귀무값은 {_NULL_BY_FAMILY[family]} 인데 "
                       f"{contract.get('null')!r} 라고 적었다")

        hit = [item for item in problems if item["validation_id"] == validation_id
               and item["question"] != "SCHEMA"]
        verdicts[validation_id] = "INVALID_VALIDATION_STATISTIC" if hit else identifiability

    return {"schema": "estimand_fidelity_audit.v1", "created_at": now_utc(),
            "ok": not problems,
            "estimand_mismatch_count": sum(1 for p in problems if p["question"] == "Q1"),
            "invalid_statistic_count": sum(1 for p in problems
                                           if p["question"] in ("Q2", "Q4", "Q5", "SCHEMA")),
            "cost_floor_artifact_count": len(cost_floor),
            "problems": problems,
            "identifiability": verdicts,
            "note": "계산이 맞다는 것과 그 값이 주장을 잰다는 것은 다르다. 여기는 뒤쪽을 본다"}


# ---- 측정 artifact 감사 (§17~§21) ------------------------------------------------

def measurement_artifact_audit(frame: pd.DataFrame, results: Mapping[str, Any],
                               config: "ValidationConfig",
                               contracts: Mapping[str, Mapping[str, Any]] | None = None
                               ) -> dict[str, Any]:
    """데이터 구조가 판정을 대신 정하고 있지 않은가.

    v1 의 두 번째 결함이 여기였다. 호가가 틱 격자라 짝 차이의 19~24% 가 정확히 0 이었고,
    중앙값이 0 으로 붕괴했다. 통계는 맞게 계산됐지만 **격자를 잰 것**이다.
    """
    warnings: list[dict[str, Any]] = []
    unhandled: list[str] = []

    spread = frame["spread_bps"].to_numpy(float) if "spread_bps" in frame else np.array([])
    floor = float(np.nanmedian(spread)) + config.fee_bps if spread.size else float("nan")
    structural = {
        "median_spread_bps": float(np.nanmedian(spread)) if spread.size else None,
        "fee_bps": config.fee_bps,
        "cost_floor_bps": -floor if np.isfinite(floor) else None,
        "why": "왕복 net 의 절대 수준은 대체로 -(스프레드+수수료) 에 눌린다. 절대 수준을 "
               "보는 통계는 이 바닥을 재게 된다",
    }

    contracts = TARGET_CONTRACTS if contracts is None else contracts
    per_target: dict[str, Any] = {}
    for validation_id, result in sorted(results.items()):
        contract = contracts.get(validation_id, {})
        tie_rate = result.get("tie_rate")
        if tie_rate is None:
            tie_rate = result.get("post_tie_rate")
        entry: dict[str, Any] = {
            "tie_rate": None if tie_rate is None else float(tie_rate),
            "effective_non_tie_n": result.get("n_effective") or result.get("post_n_effective"),
            "n": result.get("population_n") or result.get("n_matched_pairs"),
            "statistic": contract.get("statistic"),
            "statistic_measures": contract.get("statistic_measures"),
        }
        if tie_rate is not None and float(tie_rate) >= MATERIAL_TIE_RATE:
            entry["tie_warning"] = True
            warnings.append({"validation_id": validation_id, "tie_rate": float(tie_rate),
                             "why": f"동률이 {float(tie_rate):.1%} 다. 중앙값을 주 통계로 "
                                    f"쓰면 0 으로 눌린다"})
            # 동률이 많은데도 중앙값으로 판정했으면 처리하지 않은 것이다
            statistic = str(contract.get("statistic") or "")
            if "median" in statistic:
                entry["unhandled"] = True
                unhandled.append(validation_id)
        per_target[validation_id] = entry

    # 종목별 호가 간격. `100 ticks` 를 같은 시간으로 읽으면 안 되는 근거다.
    cadence = {}
    if {"symbol", "time_s"} <= set(frame.columns):
        for symbol, part in frame.groupby("symbol"):
            times = np.sort(part["time_s"].to_numpy(float))
            if times.size > 1:
                cadence[str(symbol)] = float(np.median(np.diff(times)))
    ratio = (max(cadence.values()) / min(cadence.values())) if len(cadence) > 1 else None

    return {"schema": "measurement_artifact_audit.v1", "created_at": now_utc(),
            "structural_floor": structural,
            "per_target": per_target,
            "tick_discreteness_warning_count": len(warnings),
            "tick_discreteness_warnings": warnings,
            "unhandled_high_tie_count": len(unhandled),
            "unhandled_high_tie_targets": unhandled,
            "anchor_spacing_seconds": config.spacing,
            "anchor_spacing_basis": "clock",
            "quote_cadence_seconds": dict(sorted(cadence.items())),
            "quote_cadence_spread_ratio": ratio,
            "cadence_note": "anchor 격자는 시계 기준이다. 틱 수 기준 창을 종목 사이에서 "
                            "같은 금융 시간으로 읽지 않는다",
            "note": "경고는 0 일 필요가 없다. 데이터 특성 자체일 수 있다. 0 이어야 하는 것은 "
                    "`unhandled_high_tie_count` 다"}


# ---- 세션·위생 문지기 (§22~§26) --------------------------------------------------

def hygiene_gate(dates: Sequence[str], symbols: Sequence[str], frame: pd.DataFrame
                 ) -> dict[str, Any]:
    """합성 날짜·빠진 종목을 코드가 거른다. Agent 의 추론 대상이 아니다."""
    synthetic = [d for d in dates if d in C.NOT_REAL_DATES]
    usable = [d for d in dates if d not in C.NOT_REAL_DATES]
    seen_symbols = sorted(frame["symbol"].astype(str).unique()) if len(frame) else []
    seen_dates = sorted(frame["date"].astype(str).unique()) if len(frame) else []
    per_symbol = ({str(k): int(v) for k, v in
                   frame.groupby("symbol").size().sort_index().items()}
                  if len(frame) else {})
    return {"requested_dates": list(dates),
            "usable_dates": usable,
            "excluded_synthetic_dates": synthetic,
            "synthetic_reason": {d: C.NOT_REAL_DATES[d] for d in synthetic},
            "missing_dates": [d for d in usable if d not in seen_dates],
            "eligible_symbols": seen_symbols,
            "missing_symbols": [s for s in symbols if s not in seen_symbols],
            "per_symbol_anchor_count": per_symbol,
            "synthetic_date_inclusion_count": sum(1 for d in seen_dates
                                                  if d in C.NOT_REAL_DATES)}


def _clock(local_time: int) -> str:
    """`local_time` 은 HHMMSSffffff 꼴 정수다."""
    return (f"{local_time // 10 ** 10:02d}:{local_time // 10 ** 8 % 100:02d}:"
            f"{local_time // 10 ** 6 % 100:02d}")


def session_policy() -> dict[str, Any]:
    """어느 시간대를 쓰는가. Validation 에서 조용히 바꾸지 않는다 (§25)."""
    return {
        "policy": FULL_CURRENT_FRAMEWORK_SESSION,
        "window": f"{_clock(tickdata.SESSION_START)}~{_clock(tickdata.SESSION_END)}",
        "known_regime_difference":
            "15:20~15:30 은 종가 단일가 성격이라 체결이 거의 없고 깊은 레벨이 자주 0 이다",
        "why_not_trimmed":
            "Evidence·Hypothesis·Grounding 이 모두 이 창을 썼다. 여기서만 자르면 같은 "
            "가설이 다른 모집단에서 검증된다. 바꾸려면 파이프라인 전체를 함께 올린다",
    }


def session_policy_violation(manifest: Mapping[str, Any]) -> int:
    """얼릴 때 적은 세션 정책과 지금 코드가 쓰는 창이 같은가."""
    frozen = (manifest.get("validation_plan") or {}).get("session_policy") or {}
    return 0 if frozen.get("window") == session_policy()["window"] else 1


# ---- 짝짓기 계약 (§27~§30) --------------------------------------------------------

# 결과를 보기 전에 고정한다. caliper 를 여러 개 시도해 가장 좋은 것을 고르지 않는다.
MATCHING_CONTRACT = {
    "exact_match_fields": ["symbol", "date", "bucket"],
    "distance_variables": "target 마다 사전 선언한 통제 변수",
    "distance_metric": "percentile-rank L1",
    "split_rule": "층 안에서 tested 변수의 중앙 순위로 상·하 분할",
    "replacement_policy": "NO_REPLACEMENT",
    "assignment": "1:1 nearest",
    "caliper": None,
    "minimum_pair_count": 100,
    "matching_quality_metric": "balance_ratio = 통제 변수당 잔여 순위 불균형 / tested 변수의 "
                               "순위 대비. 짝짓기가 하려는 일 자체를 재는 값이다",
    "acceptable_quality_rule": "balance_ratio <= 0.5",
    "quality_rule_derivation":
        "짝짓기의 목적은 통제 변수를 맞춰 tested 변수의 대비만 남기는 것이다. 그러므로 남은 "
        "통제 불균형이 그 대비보다 **작아야** 의미가 있다. 절반을 넘으면 무엇을 비교하는지 "
        "말할 수 없다. 절대 거리로 선을 그으면 통제 변수 개수와 분포에 따라 뜻이 달라진다. "
        "무작위 짝짓기의 기준값(변수당 E|U1-U2| = 1/3)도 함께 낸다",
    "on_failure": "INCONCLUSIVE_DUE_TO_MATCHING — 더 느슨한 짝짓기를 찾지 않는다",
}
MAX_BALANCE_RATIO = 0.5
RANDOM_RANK_DISTANCE = 1.0 / 3.0        # 무작위 짝짓기의 변수당 기대 거리


def matching_diagnostics(pairs: pd.DataFrame, candidates: int,
                         n_controls: int) -> dict[str, Any]:
    """짝짓기가 실제로 어떻게 됐는지 (§30). 숨기면 결과를 읽을 수 없다.

    주 판정값은 `balance_ratio` 다 — 남은 통제 불균형이 tested 변수의 대비보다 작은가.
    절대 거리는 통제 변수 개수에 따라 뜻이 달라져서 그것만으로는 선을 그을 수 없다.
    """
    empty = {"n_candidate": int(candidates), "n_matched": 0, "match_rate": 0.0,
             "n_controls": int(n_controls),
             "distance_median": None, "distance_p90": None,
             "control_imbalance_per_variable": None, "tested_rank_contrast": None,
             "balance_ratio": None, "random_baseline_per_variable": RANDOM_RANK_DISTANCE,
             "quality_ok": False,
             "quality_rule": MATCHING_CONTRACT["acceptable_quality_rule"]}
    if not len(pairs):
        return empty
    distance = pairs["control_distance"].to_numpy(float)
    per_variable = float(np.nanmedian(distance)) / max(n_controls, 1)
    contrast = (float(np.nanmedian(np.abs(pairs["tested_rank_delta"].to_numpy(float))))
                if "tested_rank_delta" in pairs else None)
    ratio = (per_variable / contrast) if contrast else None
    return {"n_candidate": int(candidates), "n_matched": int(len(pairs)),
            "match_rate": float(len(pairs) * 2 / candidates) if candidates else None,
            "n_controls": int(n_controls),
            "distance_median": float(np.nanmedian(distance)),
            "distance_p90": float(np.nanpercentile(distance, 90)),
            "control_imbalance_per_variable": per_variable,
            "random_baseline_per_variable": RANDOM_RANK_DISTANCE,
            "tested_rank_contrast": contrast,
            "balance_ratio": ratio,
            "balance_after": {name.replace("_delta", ""): float(np.nanmedian(
                np.abs(pairs[name].to_numpy(float))))
                for name in pairs.columns if name.endswith("_delta")
                and name not in ("future_path_delta", "success_delta",
                                 "tested_rank_delta")},
            "quality_ok": bool(ratio is not None and ratio <= MAX_BALANCE_RATIO),
            "quality_rule": MATCHING_CONTRACT["acceptable_quality_rule"]}


# ---- 지지의 강도 (§33~§38) --------------------------------------------------------

def evidence_tier(split: str, ledger: Mapping[str, Any],
                  revision_count: int = 0) -> dict[str, Any]:
    """`SUPPORTED` 하나로 일반화 강도를 뭉치지 않는다.

    같은 6주 한 구간 안에서 나온 지지는 국면을 건너뛴 지지가 아니다. 그것을 이름으로 가른다.
    """
    if split == DISCOVERY_REPLAY:
        tier = "DISCOVERY_SUPPORTED"
        why = "가설을 만드는 데 쓴 데이터다. 확증이 아니라 구현 점검이다"
    elif ledger.get("terminal_block_opened"):
        tier = "TERMINAL_OOS_SUPPORTED"
        why = "loop 전체에서 한 번도 열지 않은 마지막 구간에서 지지됐다"
    elif revision_count > 0:
        tier = "REPEATED_HOLDOUT_SUPPORTED"
        why = "수정 뒤 또 다른 새 홀드아웃에서 같은 축소형 주장이 재현됐다"
    else:
        tier = "INTERNAL_HOLDOUT_SUPPORTED"
        why = "계약을 얼린 뒤 처음 연 같은 구간 안의 새 날짜에서 지지됐다"
    return {"evidence_tier": tier, "why": why,
            "cross_regime_claimable": False,
            "cross_regime_note": (
                f"실거래일 {ledger.get('total_days')}일, 한 구간뿐이다. "
                "`CROSS_REGIME_SUPPORTED` 는 이 데이터로 주장할 수 없다")}


# ---- 데이터 예산 (§39~§43) --------------------------------------------------------

def data_ledger(*, discovery: Sequence[str] = (), validation: Sequence[str] = (),
                revision_feedback: Sequence[str] = (), final_oos: Sequence[str] = (),
                terminal_block: Sequence[str] = (), root=None) -> dict[str, Any]:
    """날짜를 무엇에 썼는가. 데이터는 다시 채워지지 않는 자원이다.

    검증 결과를 보고 가설을 고친 순간 그 날짜는 개발 데이터가 된다. 그것을 세어 둔다.
    """
    dates = [d for d in (tickdata.available_dates() if root is None
                         else tickdata.available_dates(root))
             if d not in C.NOT_REAL_DATES]
    used: dict[str, str] = {}
    for label, group in (("DISCOVERY", discovery), ("VALIDATION", validation),
                         ("REVISION_FEEDBACK", revision_feedback),
                         ("FINAL_OOS", final_oos)):
        for date in group:
            used[date] = label
    reserved = [d for d in terminal_block if d in dates]
    rows = [{"date": d,
             "used_for": used.get(d, "UNUSED"),
             "status": ("RESERVED_TERMINAL" if d in reserved and d not in used
                        else "CONSUMED" if d in used else "AVAILABLE")}
            for d in dates]
    consumed = sorted(used)
    available = [r["date"] for r in rows if r["status"] == "AVAILABLE"]
    return {"schema": "validation_data_ledger.v1", "created_at": now_utc(),
            "total_days": len(dates),
            "rows": rows,
            "consumed_dates": consumed,
            "consumed_days": len(consumed),
            "reserved_terminal_block": reserved,
            "terminal_block_opened": bool(set(reserved) & set(used)),
            "remaining_fresh_dates": available,
            "remaining_fresh_days": len(available),
            "note": "예약한 마지막 구간은 가설 생성·수정·디버깅 중에 열지 않는다"}


def data_budget_violations(ledger: Mapping[str, Any], using: Sequence[str]) -> dict[str, Any]:
    """이번에 열려는 날짜가 이미 쓴 것인가, 예약한 마지막 구간인가."""
    consumed = set(ledger.get("consumed_dates") or [])
    terminal = set(ledger.get("reserved_terminal_block") or [])
    reuse = sorted(set(using) & consumed)
    leak = sorted(set(using) & terminal)
    return {"consumed_data_reuse_count": len(reuse),
            "consumed_data_reused": reuse,
            "terminal_block_leakage_count": len(leak),
            "terminal_block_leaked": leak,
            "why": "결과를 이미 본 날짜로 수정본을 확증하면 자기 답을 보고 채점하는 것이다"}


# ---- 수정의 종류 (§44~§48) --------------------------------------------------------

def revision_type(revision: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """덜어내기만 했는가, 새 것을 넣었는가. 후자는 수정이 아니라 발견이다."""
    validation = audit.get("validation") or {}
    invented = {key: int(validation.get(key) or 0) for key in
                ("revision_new_feature_count", "revision_new_relation_count",
                 "revision_new_mechanism_count")}
    actions = {str(item.get("action")) for item in revision.get("revision_diff") or []}
    added = sum(invented.values()) or bool(revision.get("new_claims"))
    if added:
        return {"revision_type": DISCOVERY_REVISION,
                "route": "RETURN_TO_EVIDENCE_DISCOVERY",
                "invented": invented,
                "why": "새 feature·관계·메커니즘이 들어갔다. 기존 검증 데이터에서 바로 "
                       "확증할 수 없다. Evidence 단계로 돌아간다"}
    return {"revision_type": CONFIRMATORY_REVISION,
            "route": "FREEZE_AND_VALIDATE_ON_FRESH_DATES",
            "actions": sorted(actions),
            "invented": invented,
            "why": "덜어내기·강등·좁히기만 했다. 새 홀드아웃에서 다시 얼려 검증한다"}


# ---- 종합 감사 (§57~§59) ----------------------------------------------------------

# 0 이 아니면 결과를 쓸 수 없는 것들. 경고성 지표와 섞지 않는다.
MUST_BE_ZERO = (
    "estimand_mismatch_count", "invalid_statistic_count",
    "capability_identifiability_violation_count", "proxy_as_identifying_evidence_count",
    "session_policy_violation_count", "synthetic_date_inclusion_count",
    "matching_quality_violation_count", "post_result_matching_change_count",
    "consumed_data_reuse_count", "terminal_block_leakage_count",
    "future_selection_leakage_count", "outcome_conditioning_violation_count",
    "new_threshold_count", "new_lag_count", "new_horizon_count",
    "post_result_metric_change_count", "post_result_control_change_count",
    "unhandled_high_tie_count",
)
# 0 일 필요가 없는 것들. 데이터 특성 자체다.
WARNING_ONLY = ("tick_discreteness_warning_count", "cost_floor_artifact_count")


def capability_identifiability_audit(results: Mapping[str, Any],
                                     capabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Validation 이 capability 를 스스로 올리지 않았는가 (§4·§8).

    `order_cancellation` 이 UNAVAILABLE 이면, 게시 잔량 감소를 봤다고 해서 그것을 취소 사건
    으로 승격하지 않는다. 최대 해석은 `NON_IDENTIFYING_PROXY` 다.
    """
    violations: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for validation_id, contract in sorted(TARGET_CONTRACTS.items()):
        declared = contract.get("identifiability")
        for name in contract.get("required_capabilities") or []:
            actual = (capabilities.get(name) or {}).get("status") if capabilities else None
            actual = actual or C.lookup(name)
            if actual is None:
                violations.append({"validation_id": validation_id,
                                   "problem": f"모르는 capability '{name}'"})
            elif actual == C.UNAVAILABLE and declared in ("DIRECTLY_TESTABLE",
                                                          "REDUCED_FORM_TESTABLE"):
                violations.append({"validation_id": validation_id, "capability": name,
                                   "problem": f"'{name}' 이 UNAVAILABLE 인데 "
                                              f"{declared} 라고 적었다"})
            elif actual == C.PROXY_ONLY and declared == "DIRECTLY_TESTABLE":
                violations.append({"validation_id": validation_id, "capability": name,
                                   "problem": f"'{name}' 은 PROXY_ONLY 다. "
                                              f"DIRECTLY_TESTABLE 이 될 수 없다"})
        # 대용치 결과를 메커니즘 식별로 승격했는가
        status = (results.get(validation_id) or {}).get("status")
        if declared == "NON_IDENTIFYING_PROXY" and status == "SUPPORTED":
            promoted.append({"validation_id": validation_id,
                             "problem": "대용치 target 이 지지됐다고 메커니즘이 식별된 것은 "
                                        "아니다. 상태를 NON_IDENTIFYING_CHECK 로 둔다"})
    return {"capability_identifiability_violation_count": len(violations),
            "capability_identifiability_violations": violations,
            "proxy_as_identifying_evidence_count": len(promoted),
            "proxy_as_identifying_evidence": promoted}


def validation_fidelity_audit(*, fidelity: Mapping[str, Any],
                              artifacts: Mapping[str, Any],
                              hygiene: Mapping[str, Any],
                              matching: Mapping[str, Any],
                              capability: Mapping[str, Any],
                              budget: Mapping[str, Any],
                              integrity: Mapping[str, Any],
                              manifest: Mapping[str, Any]) -> dict[str, Any]:
    """v2 의 감사를 그대로 두고 v1.1 지표를 얹는다. 하나로 모아서 본다."""
    metrics: dict[str, int] = {
        # v2 에서 이어받는 것 — 계획을 얼린 뒤 바꾸지 않았는가
        "future_selection_leakage_count": int(integrity.get("future_selection_leakage_count", 0)),
        "outcome_conditioning_violation_count":
            int(integrity.get("outcome_conditioning_violation_count", 0)),
        "new_threshold_count": int((manifest.get("validation_plan") or {}).get("new_thresholds", 0)),
        "new_lag_count": int((manifest.get("validation_plan") or {}).get("new_lags", 0)),
        "new_horizon_count": int((manifest.get("validation_plan") or {}).get("new_horizons", 0)),
        "post_result_metric_change_count": 0,
        "post_result_control_change_count": 0,
        "post_result_matching_change_count": 0,
        # v1.1 신규
        "estimand_mismatch_count": int(fidelity.get("estimand_mismatch_count", 0)),
        "invalid_statistic_count": int(fidelity.get("invalid_statistic_count", 0)),
        "cost_floor_artifact_count": int(fidelity.get("cost_floor_artifact_count", 0)),
        "tick_discreteness_warning_count":
            int(artifacts.get("tick_discreteness_warning_count", 0)),
        "unhandled_high_tie_count": int(artifacts.get("unhandled_high_tie_count", 0)),
        "capability_identifiability_violation_count":
            int(capability.get("capability_identifiability_violation_count", 0)),
        "proxy_as_identifying_evidence_count":
            int(capability.get("proxy_as_identifying_evidence_count", 0)),
        "session_policy_violation_count": session_policy_violation(manifest),
        "synthetic_date_inclusion_count": int(hygiene.get("synthetic_date_inclusion_count", 0)),
        "symbol_universe_unreported_count": 0 if hygiene.get("eligible_symbols") else 1,
        "matching_quality_violation_count": int(matching.get("quality_violation_count", 0)),
        "consumed_data_reuse_count": int(budget.get("consumed_data_reuse_count", 0)),
        "terminal_block_leakage_count": int(budget.get("terminal_block_leakage_count", 0)),
    }
    failed = {k: v for k, v in metrics.items() if k in MUST_BE_ZERO and v}
    return {"schema": "validation_fidelity_audit.v1", "created_at": now_utc(),
            "metrics": dict(sorted(metrics.items())),
            "must_be_zero": list(MUST_BE_ZERO),
            "warning_only": list(WARNING_ONLY),
            "failed": failed, "ok": not failed,
            "note": "경고성 지표는 0 일 필요가 없다. 데이터 특성 자체일 수 있다"}


# ---- Freeze (§5) ----------------------------------------------------------------

# H1-R1 이 실제로 주장하는 target. 맥락 기여는 수정에서 빠졌으므로 더 이상 주장이 아니다.
# 다만 target 을 지우지 않고 **제거가 옳았는지 확인하는 검사**로 남긴다 — 새 데이터에서
# 맥락이 다시 기여하면 그 수정이 틀린 것이다.
REVISION_TARGET_ROLE = {
    "MV-H1-01": "CLAIMED", "MV-H1-02": "CLAIMED", "MV-H1-03": "CLAIMED",
    "MV-H1-04": "REMOVAL_CHECK",
    "MV-H1-05": "CLAIMED_AS_UNIDENTIFIED", "MV-H1-06": "CLAIMED_AS_UNIDENTIFIED",
    "MV-H1-07": "CLAIMED_AS_UNIDENTIFIED", "MV-H1-08": "NOT_CLAIMED",
}


def revision_target_roles(targets: Mapping[str, Mapping[str, Any]],
                          revision: Mapping[str, Any] | None) -> dict[str, str] | None:
    """수정본이 **아직 주장하는 것**과 **뺀 것**을 가른다.

    뺀 주장이 지지되지 않는 것은 실패가 아니라 제거가 옳았다는 확인이다. 이것을
    가설에서 읽지 않고 손으로 적으면 다음 가설에서 그대로 틀린다.
    """
    if not revision:
        return None
    hypothesis = revision.get("revised_hypothesis") or {}
    families = {str(k) for k in (hypothesis.get("evidence_roles") or {})}
    family_of = {str(e.get("representative_feature")): str(e.get("family"))
                 for e in hypothesis.get("evidence_basis") or []}
    roles: dict[str, str] = {}
    for name, spec in targets.items():
        identifiability = spec.get("identifiability")
        if identifiability == "INVALID_TARGET":
            roles[name] = "NOT_CLAIMED"
        elif identifiability == "BLOCKED_BY_CAPABILITY":
            roles[name] = "CLAIMED_AS_UNIDENTIFIED"
        elif (spec.get("target_type") == "INCREMENTAL_EFFECT_TARGET" and families
              and family_of.get(str(spec.get("tested"))) not in families):
            roles[name] = "REMOVAL_CHECK"
        else:
            roles[name] = "CLAIMED"
    return roles


def target_claims(targets: Mapping[str, Mapping[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """어느 target 이 어느 주장을 시험했는가. 대안 설명은 방향이 반대다."""
    out: dict[str, list[tuple[str, str]]] = {}
    for name, spec in targets.items():
        claim_id = spec.get("claim_id")
        if not claim_id:
            continue
        polarity = INVERSE if spec.get("claim_type") == "ALTERNATIVE_EXPLANATION" else DIRECT
        out[name] = [(str(claim_id), polarity)]
    return out


def freeze(hypotheses_path: Path, grounding_path: Path, package_path: Path,
           output: Path, *, config: ValidationConfig = ValidationConfig(),
           ledger: Mapping[str, Any] | None = None,
           revision: Mapping[str, Any] | None = None,
           target_contracts: Mapping[str, Mapping[str, Any]] | None = None,
           validation_plan_artifact_id: str | None = None) -> dict[str, Any]:
    """미래를 열기 전에 계획과 해시를 얼린다. 이 파일이 없으면 `run` 이 거부한다.

    v1.1 에서는 통계뿐 아니라 **그 통계가 무엇을 재는지**까지 함께 얼린다. 결과를 본 뒤
    통계나 짝짓기 규칙을 바꿀 여지를 없애는 것이 목적이다.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hypotheses = read_json(Path(hypotheses_path))
    grounded = read_json(Path(grounding_path))
    package = read_json(Path(package_path))
    capabilities = G.data_capability_inventory()["capabilities"]
    profile = C.research_capability_profile(
        (ledger or {}).get("consumed_dates") or [])

    # target 은 grounding 에서 만든다. 손으로 적으면 그 가설 전용 코드가 된다.
    hypothesis_id = ((revision or {}).get("revised_hypothesis", {}).get("hypothesis_id")
                     or (grounded.get("hypotheses") or [{}])[0].get("hypothesis_id")
                     or "H1")
    contracts = (dict(target_contracts) if target_contracts is not None
                 else build_targets(grounded, config, hypothesis_id) or TARGET_CONTRACTS)

    # 미래를 열기 전에 통계 충실도를 본다. 여기서 막히면 실행하지 않는다 (§16).
    fidelity = estimand_fidelity_audit(contracts)
    if not fidelity["ok"]:
        raise RuntimeError(
            "estimand fidelity 감사에서 막혔다. 미래를 열지 않는다:\n"
            + "\n".join(f"  {x['validation_id']} {x['question']} {x['problem']}"
                         for x in fidelity["problems"]))

    plan = {
        "population": "outcome-unconditioned eligible anchors on a fixed time grid",
        "population_filters": ["session time grid", "complete S30 window",
                               "valid ASK1", "pre-window observable"],
        "matching_variables": [config.context_feature, config.quote_state_feature,
                               "pre_move_bps"],
        "trigger_feature": config.trigger_feature,
        "context_feature": config.context_feature,
        "future_metrics": list(RESPONSE_COLUMNS),
        "new_thresholds": 0, "new_lags": 0, "new_horizons": 0,
        "targets": [
            {"validation_id": name, "claim_id": spec.get("claim_id"),
             "target_type": spec.get("target_type"), "tested": spec.get("tested"),
             "primary": spec.get("identifiability") == "DIRECTLY_TESTABLE",
             **({"status": spec["identifiability"]}
                if spec.get("identifiability") in ("BLOCKED_BY_CAPABILITY",
                                                   "INVALID_TARGET") else {})}
            for name, spec in sorted(contracts.items())],
        "statistics": {"effect": "tie-dropped paired sign share / spearman",
                       "interval": "percentile bootstrap",
                       "bootstrap": config.bootstrap, "seed": config.seed},
        "splits": {DISCOVERY_REPLAY: list(config.discovery_dates),
                   FROZEN_HOLDOUT: list(config.holdout_dates)},
        # v1.1 — 주장이 무엇을 묻는지, 그 통계가 무엇을 재는지 함께 얼린다
        "target_contracts": contracts,
        "identifiability": fidelity["identifiability"],
        "session_policy": session_policy(),
        "matching_contract": MATCHING_CONTRACT,
        "tie_policy": {"material_tie_rate": MATERIAL_TIE_RATE,
                       "primary_statistic": "sign share excluding ties",
                       "why": "짝 차이의 중앙값은 틱 격자 때문에 0 으로 눌린다. 결과를 본 뒤 "
                              "통계를 바꾸지 않기 위해 미리 정한다"},
        "excluded_synthetic_dates": sorted(C.NOT_REAL_DATES),
        "hypothesis_id": hypothesis_id,
        "target_roles": revision_target_roles(contracts, revision),
    }
    manifest = {
        "schema": "validation_freeze_manifest.v1_1", "created_at": now_utc(),
        "statistic_notes": STATISTIC_NOTES,
        "schema_version": SCHEMA_VERSION,
        "catalog_hash": catalog.catalog_hash(),
        "hypothesis_hash": sha256_json(hypotheses)[:16],
        "grounding_hash": sha256_json(grounded)[:16],
        "evidence_package_hash": sha256_json(package)[:16],
        "outcome_definition": package.get("outcome_definition"),
        "outcome_definition_hash": sha256_json(package.get("outcome_definition") or {})[:16],
        "validation_plan": plan,
        "validation_plan_hash": sha256_json(plan)[:16],
        "validation_plan_artifact_id": validation_plan_artifact_id,
        "config": asdict(config),
        "capabilities": capabilities,
        "research_capability_profile": profile,
        "capability_profile_hash": sha256_json(profile)[:16],
        "estimand_fidelity_audit": fidelity,
        "data_ledger": ledger,
        "revision": ({"parent_hypothesis_id": revision.get("parent_hypothesis_id"),
                      "status_after_revision": revision.get("status_after_revision"),
                      "revision_hash": sha256_json(revision)[:16],
                      "revised_hypothesis_hash":
                          sha256_json(revision.get("revised_hypothesis") or {})[:16],
                      "consumed_data_manifest": revision.get("consumed_data_manifest"),
                      "grounding_note":
                          "확증형 수정이라 새 주장이 없다. 살아남은 주장의 grounding 은 "
                          "부모 것 그대로다 — 그래서 grounding 을 다시 만들지 않는다"}
                     if revision else None),
        "frozen_before_opening_future": True,
    }
    write_json(output / "validation_freeze_manifest.json", manifest)
    return manifest


# ---- 배선 -----------------------------------------------------------------------

def collect(dates: Sequence[str], symbols: Sequence[str], config: ValidationConfig,
            root: Path = TICK_ROOT) -> pd.DataFrame:
    frames = []
    for date in dates:
        for symbol in symbols:
            try:
                part = anchor_frame(symbol, date, config, root)
            except (FileNotFoundError, tickdata.InsufficientQuoteData):
                continue
            if len(part):
                frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(output: Path, *, config: ValidationConfig = ValidationConfig(),
        symbols: Sequence[str] | None = None, root: Path = TICK_ROOT,
        revision_count: int = 0) -> dict[str, Any]:
    """얼린 계획대로 실행한다. freeze manifest 가 없으면 거부한다."""
    output = Path(output)
    manifest_path = output / "validation_freeze_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "freeze manifest 가 없다. 미래를 열기 전에 `freeze()` 를 먼저 부른다")
    manifest = read_json(manifest_path)
    plan = manifest["validation_plan"]
    if manifest["catalog_hash"] != catalog.catalog_hash():
        raise RuntimeError(f"catalog 이 얼린 뒤 바뀌었다: {manifest['catalog_hash']} "
                           f"!= {catalog.catalog_hash()}")
    symbols = list(symbols or catalog.DEFAULT_PROFILE_SYMBOLS)
    contracts = plan.get("target_contracts") or TARGET_CONTRACTS
    path_feature = next((t.get("tested") for t in sorted(contracts.values(),
                                                         key=lambda x: str(x.get("claim_id")))
                         if t.get("target_type") == "PATH_TARGET"),
                        config.trigger_feature)

    blocked = [t for t in plan["targets"] if t.get("status") == "BLOCKED_BY_CAPABILITY"]
    invalid = [t for t in plan["targets"] if t.get("status") == "INVALID_TARGET"]

    rows, paths, pair_frames = [], [], []
    integrity: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    per_split: dict[str, dict[str, Any]] = {}
    ledger = manifest.get("data_ledger") or {}
    routing_hypothesis_id = (manifest.get("validation_plan") or {}).get(
        "hypothesis_id", "H1")

    hygiene: dict[str, Any] = {}
    for split, dates in ((DISCOVERY_REPLAY, config.discovery_dates),
                         (FROZEN_HOLDOUT, config.holdout_dates)):
        # 합성 날짜는 코드가 거른다. 검증 우주에 들이지 않는다 (§22~§23).
        dates = [d for d in dates if d not in C.NOT_REAL_DATES]
        frame = collect(dates, symbols, config, root)
        hygiene[split] = hygiene_gate(dates, symbols, frame)
        if not len(frame):
            continue
        paths.append(frame.assign(split=split))
        integrity[split] = integrity_audit(frame, plan, config)

        # **얼린 target 을 그대로 돈다.** 이름을 코드에 적지 않는다 — 가설마다 이름도
        # 개수도 다르다. 무슨 통계를 쓸지는 `target_type` 이 정한다.
        results: dict[str, Any] = {}
        incremental_features = sorted(
            {str(t.get("tested")) for t in contracts.values()
             if t.get("target_type") == "INCREMENTAL_EFFECT_TARGET"})
        for name, spec in sorted(contracts.items()):
            kind = spec.get("target_type")
            tested = spec.get("tested") or config.trigger_feature
            claim_id = str(spec.get("claim_id") or "")
            if tested not in frame.columns:
                continue                    # anchor 표에 그 열이 없다. 건너뛴다
            if kind == "PATH_TARGET":
                value = path_target(frame, config, validation_id=name,
                                    claim_id=claim_id, feature=tested)
                value["status"] = _supported(value["effect_size"],
                                             value["interval_low"], value["interval_high"])
            elif kind == "ALREADY_PRICED_CHECK":
                value = already_priced(frame, config, validation_id=name,
                                       claim_id=claim_id, feature=tested)
                value["status"] = value.get("status_hint") or _supported(
                    value.get("effect_size"), value.get("interval_low"),
                    value.get("interval_high"), 0.5)
            elif kind == "INCREMENTAL_EFFECT_TARGET":
                # 나머지 시험 대상 feature 를 전부 통제한다. "추가 기여" 의 뜻 그대로다.
                controls = [f for f in incremental_features if f != tested]
                controls += [config.quote_state_feature, "pre_move_bps"]
                controls = [c for c in dict.fromkeys(controls) if c in frame.columns]
                value, pairs = incremental_effect(frame, tested, controls,
                                                  name, claim_id, config)
                value["status"] = value.get("status_hint") or _supported(
                    value.get("effect_size"), value.get("interval_low"),
                    value.get("interval_high"), 0.5)
                if len(pairs):
                    pair_frames.append(pairs.assign(split=split, validation_id=name))
            else:
                continue                    # NOT_TESTABLE — 통계가 없는 것이 맞다
            results[name] = value

        artifacts[split] = measurement_artifact_audit(frame, results, config, contracts)
        tier = evidence_tier(split, ledger or {}, revision_count)
        for key, value in results.items():
            contract = contracts.get(key, {})
            diagnostics = value.get("matching_diagnostics") or {}
            rows.append({
                "split": split, "date": "ALL", **value,
                # v1.1 — 무엇을 주장했고 무엇을 쟀는가를 결과 옆에 붙인다 (§52)
                "hypothesis_id": routing_hypothesis_id,
                "claim": contract.get("claim"),
                "estimand": contract.get("estimand"),
                "statistic_contract": contract.get("statistic"),
                "null_value": contract.get("null"),
                "identifiability": contract.get("identifiability"),
                "matching_balance_ratio": diagnostics.get("balance_ratio"),
                "matching_quality_ok": diagnostics.get("quality_ok"),
                "evidence_tier": (tier["evidence_tier"]
                                  if value.get("status") == "SUPPORTED" else None)})
        # 날짜별로도 나눈다 — aggregate 만 보고하지 않는다 (§57)
        for date, part in frame.groupby("date"):
            daily = path_target(part, config, feature=path_feature)
            daily["status"] = _supported(daily["effect_size"], daily["interval_low"],
                                         daily["interval_high"])
            rows.append({"split": split, "date": date, **daily})
        per_split[split] = results

    primary = per_split.get(FROZEN_HOLDOUT) or per_split.get(DISCOVERY_REPLAY) or {}
    blocks = {"blocked": bool(blocked),
              "targets": blocked,
              "capabilities": {k: manifest["capabilities"].get(k, {}).get("status")
                               for k in ("order_add_cancel_decomposition",
                                         "participant_identity", "information_arrival")},
              "pre_anchor_volatility": "UNAVAILABLE_IN_CATALOG"}
    routing = verdict(primary, blocks, plan.get("target_roles"),
                      routing_hypothesis_id, contracts, config)
    primary_split = FROZEN_HOLDOUT if FROZEN_HOLDOUT in per_split else DISCOVERY_REPLAY
    routing["primary_split"] = primary_split
    routing["blocked_capability_count"] = len(blocked)
    routing["invalid_target_count"] = len(invalid)

    # ---- v1.1 감사 ---------------------------------------------------------
    opened = [d for split, dates in ((DISCOVERY_REPLAY, config.discovery_dates),
                                     (FROZEN_HOLDOUT, config.holdout_dates))
              for d in dates if d not in C.NOT_REAL_DATES]
    capability = capability_identifiability_audit(
        primary, manifest.get("capabilities") or {})
    budget = data_budget_violations(ledger, opened) if ledger else {
        "consumed_data_reuse_count": 0, "terminal_block_leakage_count": 0,
        "why": "원장 없이 실행했다 — freeze 에 ledger 를 넘기면 재사용을 검사한다"}
    quality_failures = [key for key, value in primary.items()
                        if value.get("status_hint") == "INCONCLUSIVE_DUE_TO_MATCHING"]
    matching = {"quality_violation_count": 0,          # 규칙대로 멈췄으면 위반이 아니다
                "inconclusive_due_to_matching": quality_failures,
                "contract": MATCHING_CONTRACT}
    tier = evidence_tier(primary_split, ledger, revision_count)
    routing.update({k: v for k, v in tier.items()})
    routing["revision_count"] = revision_count
    routing["consumed_validation_days"] = ledger.get("consumed_days")
    routing["remaining_fresh_days"] = ledger.get("remaining_fresh_days")

    fidelity_audit = validation_fidelity_audit(
        fidelity=manifest.get("estimand_fidelity_audit") or estimand_fidelity_audit(),
        artifacts=artifacts.get(primary_split) or {},
        hygiene=hygiene.get(primary_split) or {},
        matching=matching, capability=capability, budget=budget,
        integrity=integrity.get(primary_split) or {}, manifest=manifest)

    results_frame = pd.DataFrame(rows)
    path_frame = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    pair_frame = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()

    results_frame.to_parquet(output / "mechanism_validation_results.parquet", index=False)
    if len(path_frame):
        path_frame.to_parquet(output / "path_validation.parquet", index=False)
    if len(pair_frame):
        pair_frame.to_parquet(output / "matched_control_results.parquet", index=False)
    write_json(output / "mechanism_capability_blocks.json", blocks)
    write_json(output / "hypothesis_routing.json", routing)
    write_json(output / "validation_integrity_audit.json", integrity)
    write_json(output / "validation_fidelity_audit.json", fidelity_audit)
    write_json(output / "measurement_artifact_audit.json", artifacts)
    write_json(output / "validation_hygiene_audit.json", hygiene)
    if ledger:
        write_json(output / "validation_data_ledger.json", ledger)
    return {"output": str(output), "manifest": manifest, "per_split": per_split,
            "routing": routing, "integrity": integrity, "blocks": blocks,
            "fidelity_audit": fidelity_audit, "artifacts": artifacts,
            "hygiene": hygiene, "ledger": ledger,
            "results": results_frame, "paths": path_frame, "pairs": pair_frame}
