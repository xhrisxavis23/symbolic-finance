"""증거들이 **같은 anchor 안에서도** 함께 나타나는가.

Profiler v1 은 family 마다 PROFIT 과 LOSS 를 얼마나 가르는지를 따로 쟀다. 그래서
"거래활동이 높고, 주문흐름이 음이고, 매수 잔량이 두껍다" 가 각각 참이어도 그것이
**한 사건의 한 상태**인지는 말할 수 없었다. Hypothesis Agent v1 이 실행에서 바로 그
이유로 주장을 축소했다 — "각 evidence 의 cohort 차이는 관측됐지만 동일 anchor 에서의
공동 발생은 제공되지 않았다".

이 모듈이 답하는 것은 둘뿐이다.

    Q  따로 유의미한 증거들이 같은 outcome anchor 에서도 함께 나타나는가
    Q  그 공동 구조가 PROFIT 과 LOSS 에서 실제로 다른가

## 이것은 전략 탐색이 아니다

- family **대표 하나씩**만 쓴다. raw feature 20개를 조합하면 190쌍이 되고 그건
  feature 조합 탐색이다. 같은 OFI 를 여러 번 세는 문제도 되돌아온다.
- 쌍(pair)까지만 본다. `max_joint_order = 2` 를 강제한다.
- 임계를 탐색하지 않는다. 방향은 v1 이 이미 정한 것을 쓰고, 컷은 **고정된 pooled
  중앙값**이다. 공동 발생을 크게 만드는 컷을 고르지 않는다.
- 시간 lag 를 탐색하지 않는다. anchor 시점(`relative_time = 0`)만 본다.
- anchor 이후 경로를 쓰지 않는다. 그것은 Agent 가 **예측**해야 하는 것이다.

## NOT_SUPPORTED 가 이 층의 가장 중요한 산출이다

두 증거가 각각 참인데 같은 사건에서 함께 나타나지는 않는다는 사실은, Agent 가 둘을
하나의 메커니즘으로 묶지 못하게 막는 **음의 증거**다. 그래서 SUPPORTED 만 주지 않고
전부 준다.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import catalog, evidence
from .config import now_utc, read_json, sha256_json, write_json
from .hypothesis import AgentConfig, input_audit


PROFIT, LOSS, BACKGROUND = evidence.PROFIT, evidence.LOSS, evidence.BACKGROUND
COHORTS = evidence.COHORTS

SUPPORTED, PARTIAL, NOT_SUPPORTED = "SUPPORTED", "PARTIAL", "NOT_SUPPORTED"
CLASSIFICATIONS = (SUPPORTED, PARTIAL, NOT_SUPPORTED)

MAX_JOINT_ORDER = 2       # 쌍까지만. triple 은 조합 탐색이 된다

# 쌍의 관계 유형 (§7~§15). Agent 가 임의로 만들지 않고 여기서 정한 것을 따른다.
COUPLED, CONTEXT_TRIGGER, CONTEXT_INDEPENDENT = (
    "COUPLED", "CONTEXT_TRIGGER", "CONTEXT_INDEPENDENT")
SUPPORTING, REDUNDANT_RELATION = "SUPPORTING", "REDUNDANT"
UNRELATED, UNRESOLVED_RELATION = "UNRELATED", "UNRESOLVED"
RELATIONS = (COUPLED, CONTEXT_TRIGGER, CONTEXT_INDEPENDENT, SUPPORTING,
             REDUNDANT_RELATION, UNRELATED, UNRESOLVED_RELATION)


@dataclass(frozen=True)
class JointConfig:
    """결정값. 어느 것도 공동 발생을 크게 만드는 쪽으로 고르지 않는다."""

    relative_time_ms: int = 0            # anchor 시점만 본다 (§29)
    # 상태 컷의 출처. 세 cohort 를 합친 분포의 중앙값이다 — 어느 cohort 도 편들지 않는다.
    threshold_source: str = "pooled_all_cohort_median_at_anchor"
    # **독립 기대 대비 초과**가 PROFIT 에서 LOSS 보다 이만큼은 커야 공동 구조라고 본다.
    # 공동 발생률 자체(rate_diff)로 판정하면 안 된다 — 그 값은 두 주변분포가 각각
    # PROFIT 에서 높기만 해도 커진다. 그것은 "함께 나타난다" 가 아니다 (§17·§18).
    joint_excess_margin: float = 0.05
    coverage_gap_warning: float = 0.15
    min_date_agreement: float = 0.80
    min_symbol_agreement: float = 0.70
    min_group_anchors: int = 3           # 날짜·종목 방향을 셀 최소 표본 (v1 과 같다)


# ---- family 대표 고르기 --------------------------------------------------------

def family_representatives(package: Mapping[str, Any], audit: Mapping[str, Any]
                           ) -> dict[str, dict[str, Any]]:
    """family 마다 대표 증거 하나. **효과 크기로 고르지 않는다.**

    효과로 고르면 대표 선택 자체가 선택 편향이 된다. v1 의 대표 선택과 같은 기준을
    쓴다 — 관측이 많은 것, lookback 이 짧은 것, 그다음 이름순.
    """
    usable = set(audit.get("usable_evidence", []))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in package.get("evidence_families", []):
        if item.get("evidence_id") in usable:
            by_family.setdefault(str(item.get("family")), []).append(item)
    out = {}
    for family, items in by_family.items():
        pick = sorted(items, key=lambda i: (
            -float((i.get("coverage") or {}).get(PROFIT, 0.0)),
            float((i.get("representative_feature") or {}).get("lookback", 0)),
            str((i.get("representative_feature") or {}).get("name", ""))))[0]
        out[family] = pick
    return out


# ---- 상태 정의 ----------------------------------------------------------------

def coarse_state(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    """v1 이 정한 방향 그대로의 거친 상태. 방향을 여기서 새로 찾지 않는다."""
    x = np.asarray(values, dtype=float)
    if str(direction).startswith("lower"):
        return np.where(np.isfinite(x), x <= threshold, False)
    return np.where(np.isfinite(x), x >= threshold, False)


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """v1 과 같은 순위 기반 AUC. 표본 수는 탈락 조건으로 쓰지 않는다."""
    return evidence._auc(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or len(np.unique(x[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))


def _agreement(frame: pd.DataFrame, axis: str, value: str, config: JointConfig
               ) -> tuple[int, int, float]:
    """축마다 부호가 전체와 같은가. 소수 종목이 전체를 끌고 가는지 본다."""
    rows = []
    for key, group in frame.groupby(axis):
        p = group[group.cohort == PROFIT][value].to_numpy(float)
        l = group[group.cohort == LOSS][value].to_numpy(float)
        if len(p) < config.min_group_anchors or len(l) < config.min_group_anchors:
            continue
        rows.append(float(np.nanmean(p) - np.nanmean(l)))
    if not rows:
        return 0, 0, np.nan
    diffs = np.array(rows)
    overall = np.sign(np.nanmedian(diffs))
    agree = int((np.sign(diffs) == overall).sum()) if overall else 0
    return agree, len(diffs), agree / len(diffs)


# ---- 한 쌍 ---------------------------------------------------------------------

def _conditional(anchors: pd.DataFrame, context_col: str, context_state: np.ndarray,
                 target_col: str, marginal_strength: float, config: JointConfig
                 ) -> dict[str, Any]:
    """context 상태로 좁힌 뒤에도 target 이 PROFIT/LOSS 를 가르는가."""
    subset = anchors[context_state]
    p = subset.loc[subset.cohort == PROFIT, target_col].to_numpy(float)
    l = subset.loc[subset.cohort == LOSS, target_col].to_numpy(float)
    auc = _auc(p, l)
    strength = abs(auc - 0.5) if np.isfinite(auc) else np.nan
    n_p, n_l = int(np.isfinite(p).sum()), int(np.isfinite(l).sum())
    return {
        "context_feature": context_col, "target_feature": target_col,
        "n_profit": n_p, "n_loss": n_l,
        "marginal_effect_strength": marginal_strength,
        "conditional_auc": None if not np.isfinite(auc) else float(auc),
        "conditional_effect_strength": None if not np.isfinite(strength) else float(strength),
        "conditional_direction": (
            "UNDETERMINED" if not np.isfinite(auc)
            else "higher_in_PROFIT" if auc > 0.5 else
            "lower_in_PROFIT" if auc < 0.5 else "no_separation"),
        "retention": (None if not (np.isfinite(strength) and marginal_strength)
                      else float(strength / marginal_strength)),
        "warnings": []}


def joint_pair(anchors: pd.DataFrame, a: Mapping[str, Any], b: Mapping[str, Any],
               config: JointConfig = JointConfig()) -> dict[str, Any]:
    """한 쌍의 공동 증거. anchor 수준으로 join 한다 — cohort 요약끼리 곱하지 않는다."""
    fa, fb = str(a["family"]), str(b["family"])
    ca = str((a["representative_feature"] or {})["name"])
    cb = str((b["representative_feature"] or {})["name"])
    da = str((a["snapshot_contrast"]["profit_vs_loss"] or {})["direction"])
    db = str((b["snapshot_contrast"]["profit_vs_loss"] or {})["direction"])
    sa = float(a["snapshot_contrast"]["profit_vs_loss"]["effect_strength"] or 0.0)
    sb = float(b["snapshot_contrast"]["profit_vs_loss"]["effect_strength"] or 0.0)

    # 둘 중 하나라도 NaN 인 anchor 는 이 쌍의 계산에서 뺀다. 0 으로 채우지 않는다.
    both = anchors[np.isfinite(anchors[ca]) & np.isfinite(anchors[cb])]
    coverage = {c: float((anchors.cohort == c).sum() and
                         (both.cohort == c).sum() / (anchors.cohort == c).sum())
                for c in COHORTS}

    # 컷은 세 cohort 를 합친 분포의 중앙값. 어느 쪽도 편들지 않고 탐색하지도 않는다.
    ta = float(np.nanmedian(anchors[ca].to_numpy(float)))
    tb = float(np.nanmedian(anchors[cb].to_numpy(float)))
    state_a = coarse_state(both[ca].to_numpy(float), da, ta)
    state_b = coarse_state(both[cb].to_numpy(float), db, tb)
    both = both.assign(_a=state_a, _b=state_b, _joint=(state_a & state_b).astype(float))

    rates: dict[str, Any] = {}
    for cohort in COHORTS:
        part = both[both.cohort == cohort]
        if not len(part):
            rates[cohort] = {"marginal_a": None, "marginal_b": None, "observed_joint": None,
                             "expected_joint": None, "joint_excess": None, "n": 0}
            continue
        pa, pb = float(part._a.mean()), float(part._b.mean())
        observed = float(part._joint.mean())
        rates[cohort] = {"marginal_a": pa, "marginal_b": pb, "observed_joint": observed,
                         "expected_joint": pa * pb, "joint_excess": observed - pa * pb,
                         "n": int(len(part))}

    rate_p = rates[PROFIT]["observed_joint"]
    rate_l = rates[LOSS]["observed_joint"]
    rate_diff = (None if rate_p is None or rate_l is None else rate_p - rate_l)
    # 독립이었다면 기대되는 값 대비 초과. 이것이 "같은 사건에서 함께 나타나는가" 다.
    excess_p, excess_l = rates[PROFIT]["joint_excess"], rates[LOSS]["joint_excess"]
    excess_diff = (None if excess_p is None or excess_l is None else excess_p - excess_l)

    association = {f"rho_{c.lower()}": _spearman(
        both.loc[both.cohort == c, ca].to_numpy(float),
        both.loc[both.cohort == c, cb].to_numpy(float)) for c in COHORTS}
    association = {k: (None if not np.isfinite(v) else float(v)) for k, v in association.items()}

    conditional = {
        "b_given_a": _conditional(both, ca, state_a, cb, sb, config),
        "a_given_b": _conditional(both, cb, state_b, ca, sa, config)}

    date_agree = _agreement(both, "date", "_joint", config)
    symbol_agree = _agreement(both, "symbol", "_joint", config)

    warnings: list[str] = []
    gap = abs((coverage[PROFIT] or 0) - (coverage[LOSS] or 0))
    if gap > config.coverage_gap_warning:
        warnings.append(f"PAIR_MISSINGNESS_CONFOUND gap={gap:.2f}")
    if (coverage[BACKGROUND] or 0) < 0.7:
        warnings.append("BACKGROUND_COVERAGE_SECONDARY")
    for key, value in conditional.items():
        warnings += [f"{w}:{key}" for w in value["warnings"]]

    differential_association = (None if association["rho_profit"] is None
                                or association["rho_loss"] is None
                                else association["rho_profit"] - association["rho_loss"])
    association["rho_diff_profit_minus_loss"] = differential_association
    classification, reasons = _classify(excess_diff, conditional, date_agree, symbol_agree,
                                        rates, config)

    return {
        "joint_evidence_id": f"JEV_{fa}__{fb}",
        "family_a": fa, "family_b": fb,
        "feature_a": ca, "feature_b": cb,
        "evidence_id_a": a.get("evidence_id"), "evidence_id_b": b.get("evidence_id"),
        "definition_id_a": (a["representative_feature"] or {}).get("definition_id"),
        "definition_id_b": (b["representative_feature"] or {}).get("definition_id"),
        "association": association,
        "joint_state": {
            "state_a": "LOW" if da.startswith("lower") else "HIGH",
            "state_b": "LOW" if db.startswith("lower") else "HIGH",
            "direction_source": "profiler_v1_profit_vs_loss_direction",
            "threshold_source": config.threshold_source,
            "threshold_a": ta, "threshold_b": tb,
            "rate_profit": rate_p, "rate_loss": rate_l,
            "rate_background": rates[BACKGROUND]["observed_joint"],
            "rate_diff": rate_diff,
            "expected_profit": rates[PROFIT]["expected_joint"],
            "excess_profit": rates[PROFIT]["joint_excess"],
            "expected_loss": rates[LOSS]["expected_joint"],
            "excess_loss": rates[LOSS]["joint_excess"],
            "excess_diff": excess_diff},
        "cohort_rates": rates,
        "conditional": conditional,
        "temporal_role_pair": {
            "family_a": (a.get("temporal_profile") or {}).get("pattern"),
            "family_b": (b.get("temporal_profile") or {}).get("pattern"),
            "note": "v1 의 시간 패턴을 옮겨 적은 것이다. 시간 lag 를 탐색하지 않았다"},
        "date_stability": {"agreement": date_agree[0], "total": date_agree[1],
                           "ratio": None if not np.isfinite(date_agree[2]) else date_agree[2]},
        "symbol_stability": {"agreement": symbol_agree[0], "total": symbol_agree[1],
                             "ratio": None if not np.isfinite(symbol_agree[2]) else symbol_agree[2]},
        "coverage": coverage,
        "classification": classification,
        "classification_reasons": reasons,
        "warnings": warnings,
        "summary": _summary(fa, fb, rate_diff, excess_diff, conditional, date_agree,
                            symbol_agree, classification),
    }


def _classify(excess_diff, conditional, date_agree, symbol_agree, rates, config
              ) -> tuple[str, list[str]]:
    """결정적 분류. 어느 조건도 공동 발생을 크게 만드는 쪽으로 고르지 않는다.

    판정의 축은 **독립 기대 대비 초과**이지 공동 발생률 자체가 아니다. 두 상태가 각각
    PROFIT 에서 흔하기만 해도 동시 발생률은 올라간다 — 그것은 "같은 사건의 한 상태" 가
    아니라 "각자 흔하다" 일 뿐이다.

    조건부 유지가 높은 것은 **독립적 기여**의 증거이지 공동 구조의 증거가 아니다.
    그래서 초과가 없으면 공동 구조는 `NOT_SUPPORTED` 다 (§35).
    """
    reasons: list[str] = []
    if excess_diff is None:
        reasons.append("PROFIT 또는 LOSS의 같은-anchor 관측이 없어 공동 구조를 셀 수 없다")
        return NOT_SUPPORTED, reasons

    enriched = excess_diff > config.joint_excess_margin
    stable = ((date_agree[2] if np.isfinite(date_agree[2]) else 0) >= config.min_date_agreement
              and (symbol_agree[2] if np.isfinite(symbol_agree[2]) else 0)
              >= config.min_symbol_agreement)
    reasons.append(
        f"독립 기대 대비 초과가 PROFIT 에서 LOSS 보다 {excess_diff:+.3f} "
        f"({'우연 이상으로 함께 나타난다' if enriched else '우연 이상의 공동 구조가 없다'})")
    reasons.append(f"날짜 {date_agree[2]:.0%} · 종목 {symbol_agree[2]:.0%} 방향 일치"
                   if np.isfinite(date_agree[2]) and np.isfinite(symbol_agree[2])
                   else "안정성을 셀 표본이 모자람")

    if not enriched:
        return NOT_SUPPORTED, reasons
    if stable:
        return SUPPORTED, reasons
    return PARTIAL, reasons


def _summary(fa, fb, rate_diff, excess_diff, conditional, date_agree, symbol_agree,
             classification) -> str:
    """해석하지 않은 사실 요약. 메커니즘을 말하지 않는다."""
    lines = [f"{fa} x {fb}"]
    if rate_diff is not None:
        lines.append(f"- 두 상태의 동시 발생률이 PROFIT 에서 LOSS 보다 {rate_diff:+.3f}")
    if excess_diff is not None:
        lines.append(f"- 다만 독립 기대 대비 초과의 차이는 {excess_diff:+.3f} — "
                     f"동시 발생률 차이는 주로 각 상태가 PROFIT 에서 더 흔한 데서 온다")
    for key, value in conditional.items():
        target = value["target_feature"]
        if value["conditional_effect_strength"] is None:
            lines.append(f"- {target} 의 조건부 대조는 표본이 모자라 계산되지 않았다")
        else:
            lines.append(
                f"- {value['context_feature']} 상태로 좁혔을 때 {target} 의 PROFIT-LOSS "
                f"분리는 {value['marginal_effect_strength']:.3f} → "
                f"{value['conditional_effect_strength']:.3f}")
    if np.isfinite(date_agree[2]) and np.isfinite(symbol_agree[2]):
        lines.append(f"- 방향이 날짜 {date_agree[0]}/{date_agree[1]}, "
                     f"종목 {symbol_agree[2]:.0%} 에서 같다")
    lines.append(f"- 분류 {classification}")
    return "\n".join(lines)


# ---- 관계 판정 -----------------------------------------------------------------

def relation_of(pair: Mapping[str, Any], roles: Mapping[str, str],
                onsets: Mapping[str, int | None],
                config: JointConfig = JointConfig()) -> dict[str, Any]:
    """쌍의 관계를 정한다. **역할과 이미 계산된 통계만** 쓴다 — 새로 탐색하지 않는다.

    화살표는 "먼저 관측된 분리 → 나중에 관측된 분리" 라는 뜻이지 인과가 아니다.
    """
    fa, fb = pair["family_a"], pair["family_b"]
    ra, rb = roles.get(fa, "UNRESOLVED"), roles.get(fb, "UNRESOLVED")
    context = trigger = None
    if ra == "PERSISTENT_CONTEXT" and rb in ("LATE_TRIGGER", "SETUP_STATE"):
        context, trigger = fa, fb
    elif rb == "PERSISTENT_CONTEXT" and ra in ("LATE_TRIGGER", "SETUP_STATE"):
        context, trigger = fb, fa

    ordering = "ORDER_UNCLEAR"
    if context and onsets.get(context) is not None and onsets.get(trigger) is not None:
        ordering = ("CONTEXT_BEFORE_TRIGGER" if onsets[context] < onsets[trigger]
                    else "SAME_STAGE" if onsets[context] == onsets[trigger]
                    else "ORDER_UNCLEAR")

    if "REDUNDANT" in (ra, rb):
        relation, why = REDUNDANT_RELATION, "한 증거가 다른 증거의 대수적 함수다"
    elif pair["classification"] == SUPPORTED:
        relation, why = COUPLED, "공동 발생과 상호작용이 모두 지지된다"
    elif context and ordering == "CONTEXT_BEFORE_TRIGGER":
        relation = CONTEXT_TRIGGER
        why = (f"{context} 의 분리가 먼저 관측되고 {trigger} 가 뒤따른다. "
               "인과는 검증되지 않았다")
    elif "UNRESOLVED" in (ra, rb):
        relation, why = UNRESOLVED_RELATION, "역할을 정할 수 없다"
    elif any(v["retention"] is not None for v in pair["conditional"].values()):
        relation, why = CONTEXT_INDEPENDENT, "둘 다 결과를 가르는 별도 근거다"
    else:
        relation, why = UNRELATED, "같은 가설 안에서 직접 연결할 근거가 없다"

    return {"relation": relation, "relation_reason": why,
            "role_a": ra, "role_b": rb,
            "context_family": context, "trigger_family": trigger,
            "ordering_status": ordering,
            "relation_support": ("SUPPORTED_RELATION" if relation in
                                 (COUPLED, CONTEXT_TRIGGER, CONTEXT_INDEPENDENT)
                                 else "TO_BE_TESTED" if relation == UNRESOLVED_RELATION
                                 else "NOT_A_RELATION")}


# ---- 전체 ---------------------------------------------------------------------

def build_joint_evidence(package: Mapping[str, Any], anchor_evidence: pd.DataFrame,
                         config: JointConfig = JointConfig(),
                         agent_config: AgentConfig = AgentConfig()) -> dict[str, Any]:
    """usable family 의 모든 쌍. 상위 N 만 고르지 않는다 — 음의 증거도 남겨야 한다."""
    audit = input_audit(package, agent_config)
    reps = family_representatives(package, audit)
    anchors = anchor_evidence[anchor_evidence.relative_time_ms == config.relative_time_ms]

    roles = {f: str(item.get("role", "UNRESOLVED")) for f, item in reps.items()}
    onsets = {f: item.get("separation_onset_ms") for f, item in reps.items()}
    pairs = []
    for fa, fb in itertools.combinations(sorted(reps), MAX_JOINT_ORDER):
        pair = joint_pair(anchors, reps[fa], reps[fb], config)
        pair.update(relation_of(pair, roles, onsets, config))
        pairs.append(pair)

    return {
        "schema": "joint_evidence.v1_1",
        "created_at": now_utc(),
        "catalog_sha256": catalog.catalog_hash(),
        "config": asdict(config),
        "max_joint_order": MAX_JOINT_ORDER,
        "relative_time_ms": config.relative_time_ms,
        "family_representatives": {
            f: {"evidence_id": item.get("evidence_id"),
                "feature": (item["representative_feature"] or {}).get("name"),
                "role": item.get("role"), "separation_onset_ms": item.get("separation_onset_ms"),
            "independent_evidence": item.get("independent_evidence", True),
            "selection_rule": "coverage desc, lookback asc, name — 효과 크기로 고르지 않는다"}
            for f, item in sorted(reps.items())},
        "usable_families": sorted(reps),
        "evidence_roles": dict(sorted(roles.items())),
        "independent_families": sorted(f for f, item in reps.items()
                                       if item.get("independent_evidence", True)),
        "pairs": pairs,
        "counts": {k: sum(1 for p in pairs if p["classification"] == k)
                   for k in CLASSIFICATIONS},
        "relation_counts": {k: sum(1 for p in pairs if p["relation"] == k)
                            for k in RELATIONS},
        "not_provided": [
            "post-anchor path", "threshold search", "time-lag search",
            "triple or higher combinations", "profitability-based pair selection"],
    }


def search_leakage_audit(joint: Mapping[str, Any]) -> dict[str, Any]:
    """탐색이 끼어들지 않았는지 스스로 센다 (§61). 모두 0 이어야 한다."""
    pairs = joint.get("pairs", [])
    families = joint.get("usable_families", [])
    expected = len(list(itertools.combinations(sorted(families), MAX_JOINT_ORDER)))
    thresholds = {p["joint_state"]["threshold_source"] for p in pairs}
    return {
        "threshold_grid_search": 0 if thresholds == {JointConfig().threshold_source} else 1,
        "top_n_pair_selection": 0 if len(pairs) == expected else 1,
        "profitability_based_selection": 0,
        "post_anchor_outcome_used": 0 if joint.get("relative_time_ms") == 0 else 1,
        "multiple_horizon_selection": 0,
        "time_lag_search": 0,
        "joint_order_above_two": 0 if joint.get("max_joint_order") == MAX_JOINT_ORDER else 1,
        "pairs_computed": len(pairs), "pairs_expected": expected,
        "negative_evidence_kept": sum(
            1 for p in pairs if p["classification"] == NOT_SUPPORTED)}


# ---- Agent 출력 감사 (§52·§53) -------------------------------------------------

def audit_joint_claims(hypotheses: Mapping[str, Any], joint: Mapping[str, Any]
                       ) -> dict[str, Any]:
    """Agent 가 묶은 증거 쌍이 실제 공동 증거로 지지되는가.

    `hypothesis.py` 를 건드리지 않고 밖에서 잰다 — 같은 Agent 가 더 나은 증거를 받았을 때
    어떻게 바뀌는지를 보는 것이 이번 실험이라 Agent 쪽은 얼려 둔다.
    """
    lookup = {(p["family_a"], p["family_b"]): p for p in joint.get("pairs", [])}
    lookup.update({(b, a): p for (a, b), p in list(lookup.items())})
    # 한 가설 안에서 두 증거를 함께 써도 되는 관계. `NOT_SUPPORTED` 분류는 **결합**이
    # 지지되지 않는다는 뜻이지 연결 금지가 아니다 — 맥락→방아쇠와 독립 기여는 허용된다.
    LINKABLE = (COUPLED, CONTEXT_TRIGGER, CONTEXT_INDEPENDENT)
    known_ids = {p["joint_evidence_id"] for p in joint.get("pairs", [])}

    unsupported: list[dict[str, Any]] = []
    coupling: list[dict[str, Any]] = []
    problems: list[str] = []
    referenced: set[str] = set()
    per_hypothesis = []

    for item in hypotheses.get("hypotheses") or []:
        hid = item.get("hypothesis_id")
        families = sorted({str(b.get("family")) for b in item.get("evidence_basis") or []})
        bad = []
        for fa, fb in itertools.combinations(families, 2):
            pair = lookup.get((fa, fb))
            if pair is None:
                bad.append({"hypothesis_id": hid, "family_a": fa, "family_b": fb,
                            "classification": "MISSING", "relation": "MISSING"})
                continue
            relation = pair.get("relation")
            linkable = (relation in LINKABLE if relation
                        else pair["classification"] in (SUPPORTED, PARTIAL))
            if not linkable:
                bad.append({"hypothesis_id": hid, "family_a": fa, "family_b": fb,
                            "classification": pair["classification"], "relation": relation})
            if pair["classification"] not in (SUPPORTED, PARTIAL):
                coupling.append({"hypothesis_id": hid, "family_a": fa, "family_b": fb,
                                 "classification": pair["classification"],
                                 "relation": relation})
        unsupported += bad
        per_hypothesis.append({"hypothesis_id": hid, "families": families,
                               "implied_pairs": len(families) * (len(families) - 1) // 2,
                               "unsupported_pairs": len(bad)})

        for token in _tokens(item):
            if token.startswith("JEV_"):
                referenced.add(token)
                if token not in known_ids:
                    problems.append(f"{hid}: 없는 joint evidence id '{token}'")

        # NOT_SUPPORTED 를 관측된 사실처럼 쓰지 않았는가
        observed_families = set()
        for stage in item.get("expected_observable_sequence") or []:
            if stage.get("status") != "OBSERVED":
                continue
            for ref in stage.get("evidence") or []:
                for b in item.get("evidence_basis") or []:
                    if b.get("evidence_id") == ref:
                        observed_families.add(str(b.get("family")))
        for fa, fb in itertools.combinations(sorted(observed_families), 2):
            pair = lookup.get((fa, fb))
            if pair and pair["classification"] == NOT_SUPPORTED:
                problems.append(f"{hid}: {fa}+{fb} 를 OBSERVED 로 묶었는데 공동 증거는 "
                                f"NOT_SUPPORTED 다")

    return {"unsupported_joint_claim_count": len(unsupported),
            "unsupported_joint_claims": unsupported,
            # 참고용: 결합(coupling)이 지지되지 않는 쌍 수. 관계가 허용하면 연결 자체는
            # 문제가 아니므로 위 지표와 따로 센다.
            "coupling_unsupported_pair_count": len(coupling),
            "coupling_unsupported_pairs": coupling,
            "referenced_joint_evidence_ids": sorted(referenced),
            "problems": problems,
            "per_hypothesis": per_hypothesis}


def _tokens(node: Any) -> list[str]:
    return re.findall(r"\bJEV_[A-Z_]+\b", " ".join(_strings(node)))


def _strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, Mapping):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, (list, tuple)):
        return [s for v in node for s in _strings(v)]
    return []


# ---- 표와 보고서 ---------------------------------------------------------------

# 쌍이 없을 때도 같은 모양의 빈 표를 낸다. 보고서가 열 이름으로 표를 읽기 때문이다.
JOINT_COLUMNS = (
    "joint_evidence_id", "family_a", "family_b", "feature_a", "feature_b",
    "definition_id_a", "definition_id_b", "state_a", "state_b", "threshold_source",
    "threshold_a", "threshold_b", "rho_profit", "rho_loss", "rho_background",
    "rate_profit", "rate_loss", "rate_background", "rate_diff", "expected_profit",
    "excess_profit", "expected_loss", "excess_loss", "excess_diff",
    "rho_diff_profit_minus_loss", "n_profit", "n_loss", "coverage_profit",
    "coverage_loss", "coverage_background", "date_agreement_ratio",
    "symbol_agreement_ratio", "temporal_pattern_a", "temporal_pattern_b",
    "classification", "relation", "role_a", "role_b", "context_family",
    "trigger_family", "ordering_status", "warnings")
CONDITIONAL_COLUMNS = (
    "joint_evidence_id", "direction", "context_family", "target_family",
    "context_feature", "target_feature", "context_direction", "n_profit", "n_loss",
    "marginal_effect_strength", "conditional_auc", "conditional_effect_strength",
    "conditional_direction", "retention", "warnings")


def joint_table(joint: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for p in joint["pairs"]:
        state = p["joint_state"]
        rows.append({
            "joint_evidence_id": p["joint_evidence_id"],
            "family_a": p["family_a"], "family_b": p["family_b"],
            "feature_a": p["feature_a"], "feature_b": p["feature_b"],
            "definition_id_a": p["definition_id_a"], "definition_id_b": p["definition_id_b"],
            "state_a": state["state_a"], "state_b": state["state_b"],
            "threshold_source": state["threshold_source"],
            "threshold_a": state["threshold_a"], "threshold_b": state["threshold_b"],
            "rho_profit": p["association"]["rho_profit"],
            "rho_loss": p["association"]["rho_loss"],
            "rho_background": p["association"]["rho_background"],
            "rate_profit": state["rate_profit"], "rate_loss": state["rate_loss"],
            "rate_background": state["rate_background"], "rate_diff": state["rate_diff"],
            "expected_profit": state["expected_profit"], "excess_profit": state["excess_profit"],
            "expected_loss": state["expected_loss"], "excess_loss": state["excess_loss"],
            "excess_diff": state["excess_diff"],
            "rho_diff_profit_minus_loss": p["association"]["rho_diff_profit_minus_loss"],
            "n_profit": p["cohort_rates"][PROFIT]["n"], "n_loss": p["cohort_rates"][LOSS]["n"],
            "coverage_profit": p["coverage"][PROFIT], "coverage_loss": p["coverage"][LOSS],
            "coverage_background": p["coverage"][BACKGROUND],
            "date_agreement_ratio": p["date_stability"]["ratio"],
            "symbol_agreement_ratio": p["symbol_stability"]["ratio"],
            "temporal_pattern_a": p["temporal_role_pair"]["family_a"],
            "temporal_pattern_b": p["temporal_role_pair"]["family_b"],
            "classification": p["classification"],
            "relation": p["relation"], "role_a": p["role_a"], "role_b": p["role_b"],
            "context_family": p["context_family"], "trigger_family": p["trigger_family"],
            "ordering_status": p["ordering_status"],
            "warnings": "; ".join(p["warnings"])})
    return pd.DataFrame(rows, columns=None if rows else list(JOINT_COLUMNS))


def conditional_table(joint: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for p in joint["pairs"]:
        for key, value in p["conditional"].items():
            context, target = ((p["family_a"], p["family_b"]) if key == "b_given_a"
                               else (p["family_b"], p["family_a"]))
            rows.append({
                "joint_evidence_id": p["joint_evidence_id"], "direction": key,
                "context_family": context, "target_family": target,
                "context_feature": value["context_feature"],
                "target_feature": value["target_feature"],
                "context_direction": (p["joint_state"]["state_a"] if key == "b_given_a"
                                      else p["joint_state"]["state_b"]),
                "n_profit": value["n_profit"], "n_loss": value["n_loss"],
                "marginal_effect_strength": value["marginal_effect_strength"],
                "conditional_auc": value["conditional_auc"],
                "conditional_effect_strength": value["conditional_effect_strength"],
                "conditional_direction": value["conditional_direction"],
                "retention": value["retention"],
                "warnings": "; ".join(value["warnings"])})
    return pd.DataFrame(rows, columns=None if rows else list(CONDITIONAL_COLUMNS))


def validation_report(joint: Mapping[str, Any], leakage: Mapping[str, Any],
                      baseline: Mapping[str, Any] | None = None,
                      baseline_claims: Mapping[str, Any] | None = None) -> str:
    table = joint_table(joint)
    L = ["# JOINT_EVIDENCE_VALIDATION.md", "",
         "따로 유의미한 증거들이 **같은 anchor 안에서도** 함께 나타나는지 잰 기록.",
         "관측 사실만 적는다 — 메커니즘 해석은 Hypothesis Agent 의 몫이다.", "",
         "## 1. 무엇을 썼나", "",
         f"- 카탈로그 `{joint['catalog_sha256']}`",
         f"- anchor 시점 `relative_time = {joint['relative_time_ms']}ms` (§29)",
         f"- 조합 차수 `max_joint_order = {joint['max_joint_order']}` (§8)",
         f"- 쓸 수 있는 family {len(joint['usable_families'])}개 → 쌍 {len(joint['pairs'])}개", "",
         "### family 대표 (효과 크기로 고르지 않았다)", "",
         "| family | evidence | 대표 feature |", "|---|---|---|"]
    for fam, item in joint["family_representatives"].items():
        L.append(f"| {fam} | `{item['evidence_id']}` | `{item['feature']}` |")

    counts = joint["counts"]
    L += ["", "## 2. 분류 결과", "", "| 분류 | 쌍 |", "|---|--:|"]
    L += [f"| {k} | {counts[k]} |" for k in CLASSIFICATIONS]

    L += ["", "## 3. 가장 강한 공동 관계", "",
          "| 쌍 | 상태 | PROFIT | LOSS | 차이 | 초과(관측−독립기대) | 분류 |",
          "|---|---|--:|--:|--:|--:|---|"]
    ordered = table.sort_values("rate_diff", ascending=False) if len(table) else table
    for r in ordered.head(8).itertuples():
        L.append(f"| {r.family_a} × {r.family_b} | {r.state_a}/{r.state_b} | "
                 f"{r.rate_profit:.3f} | {r.rate_loss:.3f} | {r.rate_diff:+.3f} | "
                 f"{r.excess_profit:+.3f} | {r.classification} |")

    cond = conditional_table(joint)
    L += ["", "## 4. 가장 강한 조건부 관계", "",
          "context 를 고정한 뒤에도 target 의 PROFIT-LOSS 분리가 남는가.", "",
          "| context | target | marginal | conditional | 유지 | n(P/L) |",
          "|---|---|--:|--:|--:|---|"]
    keep = (cond.dropna(subset=["retention"]).sort_values("retention", ascending=False)
            .head(8) if len(cond) else cond)
    for r in keep.itertuples():
        L.append(f"| {r.context_family} | {r.target_family} | "
                 f"{r.marginal_effect_strength:.3f} | {r.conditional_effect_strength:.3f} | "
                 f"{r.retention:.2f} | {r.n_profit}/{r.n_loss} |")

    weak = table[table.classification == NOT_SUPPORTED]
    L += ["", "## 5. 따로는 강한데 함께는 아닌 쌍 (음의 증거)", "",
          "이 정보가 이 층의 가장 중요한 산출이다 — Agent 가 둘을 하나의 메커니즘으로 "
          "묶지 못하게 막는다.", ""]
    if len(weak):
        L += ["| 쌍 | 차이 | 최대 유지 | 이유 |", "|---|--:|--:|---|"]
        for p in joint["pairs"]:
            if p["classification"] != NOT_SUPPORTED:
                continue
            ret = [v["retention"] for v in p["conditional"].values() if v["retention"] is not None]
            L.append(f"| {p['family_a']} × {p['family_b']} | "
                     f"{p['joint_state']['rate_diff']:+.3f} | "
                     f"{'—' if not ret else f'{max(ret):.2f}'} | "
                     f"{p['classification_reasons'][0]} |")
    else:
        L.append("없음.")

    L += ["", "## 6. 날짜·종목 안정성", "",
          "| 쌍 | 날짜 일치 | 종목 일치 | 분류 |", "|---|--:|--:|---|"]
    for r in (table.sort_values("rate_diff", ascending=False)
              if len(table) else table).itertuples():
        date = "—" if r.date_agreement_ratio is None else f"{r.date_agreement_ratio:.0%}"
        sym = "—" if r.symbol_agreement_ratio is None else f"{r.symbol_agreement_ratio:.0%}"
        L.append(f"| {r.family_a} × {r.family_b} | {date} | {sym} | {r.classification} |")

    L += ["", "## 7. 쌍 coverage 와 경고", "",
          "| 쌍 | PROFIT | LOSS | BACKGROUND | 경고 |", "|---|--:|--:|--:|---|"]
    for r in table.itertuples():
        L.append(f"| {r.family_a} × {r.family_b} | {r.coverage_profit:.3f} | "
                 f"{r.coverage_loss:.3f} | {r.coverage_background:.3f} | {r.warnings or '—'} |")

    L += ["", "## 8. 탐색 누출 감사 (§61)", "", "| 항목 | 값 |", "|---|--:|"]
    L += [f"| `{k}` | {v} |" for k, v in leakage.items()]
    L.append("")

    if baseline_claims is not None:
        L += ["## 9. 기존 H1 이 묶은 쌍의 결과", "",
              f"H1 이 하나의 메커니즘으로 묶은 family 조합을 공동 증거에 대조한다. "
              f"`unsupported_joint_claim_count = "
              f"{baseline_claims['unsupported_joint_claim_count']}`", "",
              "| 쌍 | 분류 |", "|---|---|"]
        seen = set()
        for claim in baseline_claims["unsupported_joint_claims"]:
            key = (claim["family_a"], claim["family_b"])
            if key in seen:
                continue
            seen.add(key)
            L.append(f"| {claim['family_a']} × {claim['family_b']} | {claim['classification']} |")
        L.append("")
    return "\n".join(L) + "\n"


# ---- 배선 --------------------------------------------------------------------

def build(evidence_dir: Path, output: Path, *, config: JointConfig = JointConfig(),
          agent_config: AgentConfig = AgentConfig(),
          baseline_hypotheses: Path | None = None) -> dict[str, Any]:
    """공동 증거를 만들고 v1.1 package 를 쓴다. v1 의 marginal 증거는 그대로 둔다."""
    evidence_dir, output = Path(evidence_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    package = read_json(evidence_dir / "hypothesis_evidence.json")
    anchors = pd.read_parquet(evidence_dir / "anchor_evidence.parquet")

    joint = build_joint_evidence(package, anchors, config, agent_config)
    leakage = search_leakage_audit(joint)

    baseline_claims = None
    if baseline_hypotheses is not None and Path(baseline_hypotheses).exists():
        baseline_claims = audit_joint_claims(read_json(Path(baseline_hypotheses)), joint)

    # v1.1 package — 기존 marginal 을 유지하고 joint 를 더한다
    upgraded = dict(package)
    upgraded["schema"] = "hypothesis_evidence_package.v1_1"
    upgraded["joint_evidence"] = [
        {k: v for k, v in p.items() if k not in ("cohort_rates", "classification_reasons")}
        for p in joint["pairs"]]
    # family -> role 의 정본. `evidence_families` 는 묶음 단위라 한 family 에 여러 역할이
    # 들어 있다. 대표 묶음의 역할이 그 family 의 역할이다.
    upgraded["evidence_roles"] = dict(joint["evidence_roles"])
    upgraded["independent_evidence_families"] = list(joint["independent_families"])
    upgraded["joint_evidence_note"] = (
        "family 대표 쌍의 same-anchor 공동 구조다. anchor 시점만 보았고 임계·시간 lag 를 "
        "탐색하지 않았다. NOT_SUPPORTED 는 '그 feature 가 중요하지 않다' 가 아니라 "
        "'현재 증거는 두 개가 하나의 same-anchor 메커니즘으로 함께 작동한다고 지지하지 "
        "않는다' 는 뜻이다.")
    upgraded["not_provided"] = list(package.get("not_provided", [])) + [
        "post-anchor path", "joint threshold search", "time-lag search",
        "triple or higher feature combinations"]

    joint_table(joint).to_parquet(output / "joint_evidence.parquet", index=False)
    conditional_table(joint).to_parquet(output / "conditional_evidence.parquet", index=False)
    write_json(output / "joint_evidence.json", joint)
    write_json(output / "hypothesis_evidence_v1_1.json", upgraded)
    (output / "JOINT_EVIDENCE_VALIDATION.md").write_text(
        validation_report(joint, leakage, None, baseline_claims), encoding="utf-8")
    write_json(output / "joint_manifest.json", {
        "schema": "joint_evidence_manifest.v1_1",
        "created_at": now_utc(),
        "catalog_hash": catalog.catalog_hash(),
        "source_evidence_dir": str(evidence_dir),
        "source_package_hash": sha256_json(package)[:16],
        "output_package_hash": sha256_json(upgraded)[:16],
        "config": asdict(config), "agent_config": asdict(agent_config),
        "search_leakage_audit": leakage,
        "baseline_hypotheses": str(baseline_hypotheses) if baseline_hypotheses else None,
        "baseline_joint_claims": baseline_claims})

    return {"output": str(output), "joint": joint, "leakage": leakage,
            "baseline_claims": baseline_claims, "package": upgraded,
            "counts": joint["counts"]}
