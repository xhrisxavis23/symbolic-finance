"""진 거래를 대조해 손실 유형을 설명하고 entry guard 후보를 만든다.

**이 모듈은 Entry Refinement Stage가 쓴다.** Implementation 뒤, Parameter Search 앞에서
고정 청산 장부의 진입 가드만 고르고 라운드를 반복한다. 청산 후보는 만들지 않는다.

`route()`는 raw price path cohort를 분류하는 연구 진단이다.

    PERSISTENT_ADVERSE            진입 전 가격 경로가 계속 불리함
    EARLY_RECOVERY_LATE_REVERSAL  초반 회복 뒤 되돌림
    COST_INSUFFICIENT             gross 이동이 비용보다 작음
    (회복이 거의 없음)             가설의 관측 움직임이 희박함

현재 Entry Refinement는 이 분류를 보고 artifact에 보존할 뿐, 중단 근거로 쓰지 않는다.
실제 고정 canonical exit의 `net_bps` 부호가 guard 탐색의 유일한 Discovery 선택 표적이다.

## 격자 채점을 두 단계로 하는 이유

가드 후보는 190개다. 각각에 대해 원장을 다시 만들면 라운드 하나가 원장 190벌이 된다.
그래서 1단계는 원장 위에 마스크만 씌워 싸게 재고(`metrics.measure_masked_screening`, 근사),
2단계에서 상위 몇 개만 가드를 실제로 붙여 원장을 다시 만들어 확정한다.

근사인 이유와 실측 오차는 `metrics.measure_masked` 에 적혀 있다 — decision 수는 최대
4.6배까지 어긋나지만 `bps_per_decision` 은 0.4~2.7% 안에 들었다.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import catalog, metrics
from .config import MIN_DECISIONS, MIN_DECISION_FRACTION, QUANTILES, sha256_json
from .ledger import FILLED, UNFILLED, threshold_column
from .metrics import Metrics
from .outcome import (COHORTS, COST_INSUFFICIENT, EARLY_RECOVERY_LATE_REVERSAL,
                      FAILURE_COHORTS, NET_RECOVERY, PERSISTENT_ADVERSE)


ROUTE_ENTRY = "ENTRY_GUARD"
ROUTE_FIXED_EXIT = "FIXED_EXIT_DIAGNOSTIC"
ROUTE_COST = "OUT_OF_LOOP_COST_RESEARCH"
ROUTE_REFUTED = "HYPOTHESIS_REFUTED"
ROUTE_NOTHING = "NO_FAILING_DECISIONS"

ROUTE_BY_COHORT = {
    PERSISTENT_ADVERSE: ROUTE_ENTRY,
    EARLY_RECOVERY_LATE_REVERSAL: ROUTE_FIXED_EXIT,
    COST_INSUFFICIENT: ROUTE_COST,
}

# 가설 기각 임계. 발견이 아니라 정책 선택이므로 이름을 붙여 인자로 둔다.
REFUTED_RECOVERY_RATE = 0.01
REFUTED_ADVERSE_RATE = 0.90

DEFAULT_SHORTLIST = 10
MIN_SEPARATION = 0.0
RETENTION_BOUNDS = (0.0, 1.0)
PAIRWISE_SEED_LIMIT = 16
COHORT_SELECTION = "COHORT"
FIXED_EXIT_NET_SELECTION = "FIXED_EXIT_NET"


def route(frame: pd.DataFrame, *, recovery_rate: float = REFUTED_RECOVERY_RATE,
          adverse_rate: float = REFUTED_ADVERSE_RATE) -> dict[str, Any]:
    """원시 price-path cohort의 개수와 설명용 분류를 돌려준다.

    `modules.refinement`는 이 값을 guard 탐색 중단에 쓰지 않는다. 이는 fixed-exit
    실제 손실 장부와 raw price path를 같은 학습 표적으로 섞지 않기 위한 분리다.
    """
    filled = frame.loc[frame["status"].eq(FILLED)] if len(frame) else frame
    counts = Counter(str(v) for v in filled["cohort"]) if len(filled) else Counter()
    total = sum(counts[c] for c in COHORTS)

    verdict: dict[str, Any] = {"refuted": False, "fills": total}
    if total:
        recovered = counts[NET_RECOVERY] / total
        adverse = counts[PERSISTENT_ADVERSE] / total
        verdict.update({
            "recovery_rate": recovered, "persistent_adverse_rate": adverse,
            "thresholds": {"recovery_below": recovery_rate, "adverse_above": adverse_rate},
            "refuted": bool(recovered < recovery_rate and adverse > adverse_rate),
        })
        verdict["reason"] = ("예측한 움직임이 없다. 타이밍 문제가 아니다"
                             if verdict["refuted"] else
                             "가드가 고를 만큼 회복하는 체결이 있다")

    failures = {c: counts[c] for c in FAILURE_COHORTS}
    if not total:
        primary = ROUTE_NOTHING
    elif verdict["refuted"]:
        primary = ROUTE_REFUTED
    elif not any(failures.values()):
        primary = ROUTE_NOTHING
    else:
        primary = ROUTE_BY_COHORT[max(failures, key=lambda c: failures[c])]

    return {"route": primary, "cohort_counts": dict(counts), "failures": failures,
            "verdict": verdict,
            "decision_sha256": sha256_json({"counts": dict(counts), "route": primary})[:16]}


# ---- 진입 가드 후보 -----------------------------------------------------------

def guard_mask(frame: pd.DataFrame, expression: Mapping[str, Any]) -> np.ndarray:
    """조항 하나를 원장 행에 적용한 마스크. **원장 컬럼만 읽는다.**

    구 구조는 격자 함수가 붙여 둔 임시 컬럼(`_partition_code`)에 기대어, 그 컬럼이
    없는 경로에서 재현 확인이 항상 KeyError 로 죽었다. 여기서는 원장이 스스로 담고
    있는 feature 값과 전날 임계 컬럼만 쓴다.
    """
    feature = str(expression["feature"])
    column = threshold_column(feature, float(expression["quantile"]))
    if feature not in frame.columns or column not in frame.columns:
        raise KeyError(f"원장에 {feature} 또는 {column} 이 없다")
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
    cut = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(values) & np.isfinite(cut)
    keep = values <= cut if str(expression["operator"]) == "<=" else values >= cut
    return usable & keep


def _mask_fingerprint(keep: np.ndarray) -> str:
    """같은 guard mask를 큰 index 목록 없이 식별한다."""
    bits = np.packbits(np.asarray(keep, dtype=bool), bitorder="little").tobytes()
    size = int(len(keep)).to_bytes(8, byteorder="little", signed=False)
    return hashlib.sha256(size + bits).hexdigest()[:16]


def _selection_masks(frame: pd.DataFrame, selection_target: str) -> tuple[np.ndarray, np.ndarray]:
    """후보가 보존해야 할 고정 exit 수익과 제거해야 할 손실을 정한다.

    ``FIXED_EXIT_NET``은 Discovery에만 쓰는 진입 학습 표적이다. 고정 canonical
    exit가 낸 실제 ``net_bps``의 부호를 그대로 쓰므로 COST_INSUFFICIENT와
    EARLY_RECOVERY_LATE_REVERSAL도 손실에서 빠지지 않는다. 지정가 미체결은
    canonical 성과에서 순손익 0인 scorable decision이므로, 이 단계에서도
    보존할 수익 거래가 아니라 피할 non-positive decision으로 센다.
    """
    filled = frame["status"].eq(FILLED).to_numpy()
    net = pd.to_numeric(frame.get("net_bps"), errors="coerce").to_numpy(dtype=float)
    if selection_target == FIXED_EXIT_NET_SELECTION:
        observed = filled & np.isfinite(net)
        unfilled = frame["status"].eq(UNFILLED).to_numpy()
        return observed & (net > 0.0), (observed & (net <= 0.0)) | unfilled
    if selection_target == COHORT_SELECTION:
        cohort = frame["cohort"].to_numpy() if "cohort" in frame.columns else np.array([""] * len(frame))
        return filled & (cohort == NET_RECOVERY), filled & (cohort == PERSISTENT_ADVERSE)
    raise ValueError(f"알 수 없는 entry guard selection target: {selection_target}")


def guard_candidates(frame: pd.DataFrame, manifest: Mapping[str, Any],
                     parent: Metrics | None = None,
                     axes: Sequence[str] | None = None,
                     quantiles: Sequence[float] = QUANTILES,
                     selection_target: str = COHORT_SELECTION) -> list[dict[str, Any]]:
    """진입 가드 후보 격자. `catalog.guard_axes()` × 분위수 × 부등호.

    임계는 고정값이 아니라 **그 종목의 전날 분위수 레벨**이다. 종목 공통 절대 임계는
    유동성에 비례하는 호가량에 대해 종목마다 전혀 다른 위치에 놓여, 상태가 아니라
    종목 크기를 거르는 필터가 된다.

    후보마다 남길 것 — **가드가 진짜 필터인지 표본 축소인지는 남긴 것과 지운 것을
    비교해야만 안다.** 실측으로 어떤 가드는 승자 67.6%·패자 75.4% 를 지웠다. 차이
    7.8%p 로는 필터라 부를 수 없다.
    """
    parent = parent or metrics.measure(frame, manifest)
    axes = tuple(axes or catalog.guard_axes())
    # 후보 190개는 feature cut만 다르고 회계 열은 같다. 전체 feature frame을 매번
    # 정렬·복사하면 대규모 Discovery에서 행 수보다 열 수가 memory를 결정한다.
    # `measure_masked`와 retention 계산에 필요한 열만 한 번 남긴다.
    metric_columns = [name for name in (
        "contract_id", "symbol", "date", "entry_tick", "exit_tick", "episode_index",
        "status", "net_bps", "gross_bps",
    ) if name in frame]
    metric_frame = frame.loc[:, metric_columns].copy()
    filled = frame["status"].eq(FILLED).to_numpy()
    net = pd.to_numeric(frame.get("net_bps"), errors="coerce").to_numpy(dtype=float)
    recovery, adverse = _selection_masks(frame, selection_target)

    # 마스크가 완전히 같은 후보는 하나만 남긴다. 실측: `book_imbalance` 는
    # `queue_imbalance_best` 의 단조 변환(2q-1)이라 10개 컷이 전부 같은 마스크를 냈다.
    # 중복을 그대로 두면 같은 가설을 두 번 시험하면서 다중검정 개수만 부풀린다.
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    out: list[dict[str, Any]] = []
    for feature in axes:
        if feature not in frame.columns:
            continue
        for quantile in quantiles:
            if threshold_column(feature, quantile) not in frame.columns:
                continue
            for operator in ("<=", ">="):
                expression = {"feature": feature, "operator": operator,
                              "quantile": float(quantile),
                              "threshold_policy": "prior_valid_day_per_symbol",
                              "decision_time": "same_tick",
                              "uses_current_or_past_only": True}
                keep = guard_mask(frame.loc[:, [feature, threshold_column(feature, quantile)]],
                                  expression)
                fingerprint = _mask_fingerprint(keep)
                candidate_id = f"{feature}|{operator}|q{quantile:.2f}"
                if fingerprint in seen:
                    duplicates.append({"candidate_id": candidate_id,
                                       "same_as": seen[fingerprint]})
                    continue
                seen[fingerprint] = candidate_id
                # 이 값은 shortlist 순위만 정한다. guard로 새로 열리는 signal은
                # 기존 decision 원장에 없으므로, 채택은 아래 Stage의 actual replay다.
                scored = metrics.measure_masked_screening(metric_frame, keep, manifest)
                kept_net = net[keep & filled]
                removed_net = net[~keep & filled]
                out.append({
                    "candidate_id": candidate_id,
                    "expression": expression,
                    "selection_target": selection_target,
                    "metrics": scored,
                    "recovery_retention": float(keep[recovery].mean()) if recovery.any() else 0.0,
                    "adverse_retention": float(keep[adverse].mean()) if adverse.any() else 0.0,
                    "decision_retention": float(keep.mean()) if len(keep) else 0.0,
                    # 남긴 것 vs 지운 것. 이 둘이 비슷하면 필터가 아니라 표본 축소다.
                    "kept": _side(kept_net),
                    "removed": _side(removed_net),
                    "delta_total_net_bps": float(scored.total_net_bps - parent.total_net_bps),
                    "delta_bps_per_decision": _delta(scored.bps_per_decision, parent.bps_per_decision),
                    "decomposition": metrics.compare(scored, parent),
                })
    for row in out:
        row["separation"] = row["recovery_retention"] - row["adverse_retention"]
        row["duplicates_removed"] = duplicates
    return out


def guard_pair_candidates(frame: pd.DataFrame, manifest: Mapping[str, Any],
                          candidates: Sequence[Mapping[str, Any]],
                          parent: Metrics | None = None,
                          *, seed_limit: int = PAIRWISE_SEED_LIMIT,
                          selection_target: str = COHORT_SELECTION) -> list[dict[str, Any]]:
    """단독으로는 약해도 함께 손실을 거르는 두 guard를 Discovery에서만 선별한다.

    후보 두 개를 조합하면 격자가 급격히 커진다. 그래서 recovery/adverse 분리가 있는
    단일 guard 상위 `seed_limit`개만 조합한다. 이 결과도 기존 원장 마스크의 **근사
    순위**일 뿐이며, Entry Refinement Stage가 shortlist에 실제 guard 두 개를 붙여
    동일 canonical Backtest로 다시 실행해야만 채택할 수 있다.

    두 guard는 모두 같은 tick의 현재 feature와 그 tick 이전 100개 관측 분위수만 읽는다. 새
    데이터·미래 값·청산 값은 쓰지 않는다.
    """
    parent = parent or metrics.measure(frame, manifest)
    low, high = RETENTION_BOUNDS
    eligible = [row for row in candidates
                if isinstance(row.get("expression"), Mapping)
                and float(row.get("separation") or 0.0) > MIN_SEPARATION
                and low <= float(row.get("decision_retention") or 0.0) <= high]
    # 분리력이 큰 조항부터 보되, 동률은 후보 id로 고정해 실행 순서를 재현한다.
    eligible.sort(key=lambda row: (-float(row.get("separation") or 0.0),
                                   str(row.get("candidate_id") or "")))
    seeds = eligible[:max(0, int(seed_limit))]
    if len(seeds) < 2:
        return []

    metric_columns = [name for name in (
        "contract_id", "symbol", "date", "entry_tick", "exit_tick", "episode_index",
        "status", "net_bps", "gross_bps",
    ) if name in frame]
    metric_frame = frame.loc[:, metric_columns].copy()
    filled = frame["status"].eq(FILLED).to_numpy()
    net = pd.to_numeric(frame.get("net_bps"), errors="coerce").to_numpy(dtype=float)
    recovery, adverse = _selection_masks(frame, selection_target)
    masks = {
        str(row["candidate_id"]): guard_mask(frame, row["expression"])
        for row in seeds
    }
    # 단일 mask와 완전히 같은 pair는 새 조합이 아니다.
    seen = {_mask_fingerprint(mask) for mask in masks.values()}
    out: list[dict[str, Any]] = []
    for left, right in combinations(seeds, 2):
        left_id, right_id = str(left["candidate_id"]), str(right["candidate_id"])
        keep = masks[left_id] & masks[right_id]
        fingerprint = _mask_fingerprint(keep)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        scored = metrics.measure_masked_screening(metric_frame, keep, manifest)
        kept_net = net[keep & filled]
        removed_net = net[~keep & filled]
        expressions = [dict(left["expression"]), dict(right["expression"])]
        out.append({
            "candidate_id": f"{left_id}&&{right_id}",
            "expressions": expressions,
            "candidate_kind": "PAIRWISE_ENTRY_GUARD",
            "selection_target": selection_target,
            "metrics": scored,
            "recovery_retention": float(keep[recovery].mean()) if recovery.any() else 0.0,
            "adverse_retention": float(keep[adverse].mean()) if adverse.any() else 0.0,
            "decision_retention": float(keep.mean()) if len(keep) else 0.0,
            "kept": _side(kept_net),
            "removed": _side(removed_net),
            "delta_total_net_bps": float(scored.total_net_bps - parent.total_net_bps),
            "delta_bps_per_decision": _delta(scored.bps_per_decision, parent.bps_per_decision),
            "decomposition": metrics.compare(scored, parent),
        })
    for row in out:
        row["separation"] = row["recovery_retention"] - row["adverse_retention"]
    return out


def _side(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"fills": 0, "mean_net_bps": None, "positive_share": None}
    return {"fills": int(len(finite)), "mean_net_bps": float(finite.mean()),
            "median_net_bps": float(np.median(finite)),
            "positive_share": float((finite > 0).mean())}


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def choose_guard(
    candidates: Sequence[Mapping[str, Any]],
    parent: Metrics,
    *,
    baseline_decisions: int,
    min_decisions: int = MIN_DECISIONS,
    min_decision_fraction: float = MIN_DECISION_FRACTION,
    shortlist: int = DEFAULT_SHORTLIST,
) -> dict[str, Any]:
    """후보를 거르고 상위 몇 개를 남긴다. **하나를 확정하지 않는다.**

    순서가 중요하다.
      1. 표본 바닥으로 **먼저 거른다.** 고른 뒤 검사하면 욕심낸 픽 하나가 루프를
         끝내고, 바닥을 넘으면서 개선도 한 후보를 전부 버리게 된다.
      2. 바닥은 decision 기준이다. signal tick 기준이면 신호 지속시간이 바닥을 정한다.
      3. 총 순손익이 Parent보다 나쁘면 버린다. 이 Stage의 목표와 Search의 주
         선택 기준은 같은 fixed-exit 총 순손익이다.
      4. 동률이면 결정당 수익으로만 순서를 보조하고, 그래도 동률이면 임의로 깨지지 않는다.

    돌려주는 `shortlist` 는 **근사값 기준 순위**다. 확정하려면 부르는 쪽이 이것들에
    가드를 실제로 붙여 원장을 다시 만들어야 한다 (`exact=False` 가 그 표시).
    """
    floor = max(int(min_decisions), int(baseline_decisions * float(min_decision_fraction)))
    reasons = Counter()
    surviving = []
    for row in candidates:
        scored: Metrics = row["metrics"]
        if scored.scorable < floor:
            reasons["BELOW_SAMPLE_FLOOR"] += 1
            continue
        if row["separation"] <= MIN_SEPARATION:
            reasons["NO_SEPARATION"] += 1
            continue
        low, high = RETENTION_BOUNDS
        if not low <= row["decision_retention"] <= high:
            reasons["RETENTION_OUT_OF_BOUNDS"] += 1
            continue
        if scored.total_net_bps <= parent.total_net_bps:
            reasons["NO_IMPROVEMENT"] += 1
            continue
        surviving.append(row)

    surviving.sort(key=lambda r: (-float(r["metrics"].total_net_bps),
                                  -float(r["metrics"].bps_per_decision or 0.0),
                                  str(r["candidate_id"])))
    top = surviving[: int(shortlist)]
    return {
        "decision_floor": floor,
        "candidates": len(candidates),
        "surviving": len(surviving),
        "rejected": dict(reasons),
        "shortlist": top,
        "shortlist_ids": [r["candidate_id"] for r in top],
        "exact": False,
        "note": "기존 decision 원장의 총 순손익 screening 순위다. 확정하려면 가드를 붙여 같은 fixed-exit Backtest를 다시 만들어야 한다",
    }


def score_on(frame: pd.DataFrame, manifest: Mapping[str, Any],
             expression: Mapping[str, Any]) -> Metrics:
    """이미 고른 조항 하나를 **다른 날의 원장**에서 채점한다.

    격자를 다시 만들지 않는다 — 다시 만들면 그 날에서 또 최댓값을 고르게 되어 같은
    문제가 반복된다. 조항은 발굴일에서 고정된 것 하나뿐이다.

    원장 컬럼만 읽는다. 격자 함수가 붙여 둔 임시 컬럼에 기대지 않는다.
    """
    return metrics.measure_masked(frame, guard_mask(frame, expression), manifest)


def replication_verdict(discovery: Metrics, per_day: Sequence[Metrics],
                        *, minimum_retained_fraction: float = 0.5) -> dict[str, Any]:
    """고른 조항이 발굴 구간 안의 **다른 날** 에서 다시 나오는가.

    후보 190개 중 최댓값을 고르면 진짜 우위가 0이어도 큰 값이 나온다. 그 값 자체로는
    아무것도 말할 수 없다.

    두 조건을 모두 요구한다.
      · 확인일마다 양수 — 날마다 독립 시도이므로 우연 통과 확률이 곱해져 줄어든다
      · 평균이 발굴 값의 일정 비율 이상 — 부호만 맞고 크기가 사라지면 재현이 아니다

    확인일은 발굴 구간 안이되 후보를 고른 날과 겹치지 않아야 한다 (`config.Blocks`
    가 강제한다). Val·OOS 는 여기 들어오지 않는다.
    """
    values = [m.bps_per_decision for m in per_day]
    usable = [v for v in values if v is not None]
    base = discovery.bps_per_decision
    if not usable or base is None:
        return {"replicates": False, "reason": "잴 수 없는 날이 있다",
                "per_day": values, "discovery": base}
    mean = float(np.mean(usable))
    all_positive = all(v > 0.0 for v in usable)
    floor = float(minimum_retained_fraction) * base
    keeps = mean >= floor if base > 0 else mean > 0.0
    return {
        "per_day_bps_per_decision": values,
        "per_day_decisions": [m.scorable for m in per_day],
        "mean_bps_per_decision": mean,
        "discovery_bps_per_decision": base,
        "all_days_positive": all_positive,
        "retained_fraction": mean / base if base else None,
        "keeps_magnitude": bool(keeps),
        "replicates": bool(all_positive and keeps),
        "minimum_retained_fraction": float(minimum_retained_fraction),
    }
