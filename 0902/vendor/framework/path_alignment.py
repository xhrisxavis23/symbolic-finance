"""가설이 미리 말한 경로와 원장이 실제로 보여준 경로를 대조한다.

지금까지 가설이 갖고 있던 것은 `rationale` 자유 문장 하나였다. 그래서 결과가 나쁘면
"임계가 안 맞았다", 좋으면 "역시 그렇다" 로 사후에 어느 쪽으로든 붙일 수 있었고,
프레임워크가 물을 수 있는 것은 `net_bps_total > 0` 하나뿐이었다. 그것은 "벌었나" 이지
"가설이 맞았나" 가 아니다.

여기서는 가설이 **원장 열의 언어로** 미리 말하게 하고, 재생 직후 같은 열에서 실제 값을
뽑아 맞춰 본다. 판정은 두 갈래로 갈린다.

              수익 O                        수익 X
    정합 O    가설이 맞고 돈도 된다         현상은 맞는데 비용을 못 넘는다
    정합 X    다른 이유로 벌었다            기각

왼쪽 아래가 지금까지 안 보이던 칸이다. net 만 보면 성공으로 읽히지만 가설이 말한 것과
다른 일이 일어나고 있다.

**정합은 선택에 개입하지 않는다.** Discovery 선택은 계속 `net_bps_total` 하나다.
채점이 선택을 바꾸면 무난한 예측을 쓸 이유가 생긴다.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from . import ledger, outcome


SCHEMA_VERSION = "path_alignment.v2"

# 용어: 여기서 다루는 여섯은 **예측 항목**(prediction field)이다. `catalog.guard_axes()`
# 의 **축**(자를 수 있는 feature)과 다른 것이므로 같은 말로 부르지 않는다.
#
# v1 의 여섯 중 셋은 죽은 질문이었다. 실측 12개 후보에서 관측이 사실상 고정이었다.
#
#   ask_move_bps      NEGATIVE 12/12   BID1 지정가는 누가 나에게 팔아야 체결된다.
#                                      체결됐다는 것이 이미 매도 압력을 뜻한다.
#   entry_spread_bps  WIDER 11/12      스프레드가 넓으면 큐가 짧아 잘 체결된다.
#                                      체결됐다는 사실이 넓은 자리를 골라낸다.
#   dominant_cohort   PERSISTENT 11/12 NET_RECOVERY 는 최대 28%인데 PERSISTENT 가
#                                      평균 52%다. 최빈이 될 수 없다. 게다가 이기는
#                                      가설은 NET_RECOVERY 라 답할 수밖에 없으므로
#                                      정직할수록 반드시 틀리는 질문이었다.
#
# 셋 다 진입식이 아니라 **실행 규칙이 정하는 값**이라 가설을 구분하지 못했다.
# v2 는 그 자리를 비교형 질문으로 바꿨다. 비교형이라 데이터에서 뽑은 임계가 없다 —
# 동률 밴드만 둔다.

# 판정 임계. 결과를 보고 정하면 채점이 아니라 탐색이 된다. 발굴을 열기 전에 고정한다.
MOVE_DEAD_ZONE_BPS = 1.0     # 이동값의 절대값이 이보다 작으면 NEUTRAL/SAME
SHARE_TIE_BAND = 0.05        # 과반 비교의 동률 밴드 (0.45~0.55 는 MIXED)
FILL_RATE_HIGH = 0.2
ALIGNED_MIN_HITS = 5         # 여섯 항목 중
CONTRADICTED_MAX_HITS = 2

ALIGNED, PARTIAL, CONTRADICTED = "ALIGNED", "PARTIAL", "CONTRADICTED"
NOT_MEASURABLE = "NOT_MEASURABLE"

MOVE_VALUES = ("POSITIVE", "NEGATIVE", "NEUTRAL")
# BID1 지정가는 시장이 내려와야 체결된다. 게시가보다 올라간 뒤 체결되는 일은 구조적으로
# 없다 — tick250 캠페인 162건에서 POSITIVE 가 0건이다. 고를 수 있게 두면 Agent 가
# 맞출 수 없는 칸을 계속 고른다.
SLIPPAGE_VALUES = ("NEGATIVE", "NEUTRAL")
COMPARE_VALUES = ("MORE", "SAME", "LESS")
LOSS_KIND_VALUES = ("PERSISTENT_ADVERSE", "ROSE_THEN_LOST", "MIXED")
RATE_VALUES = ("HIGH", "LOW")

# 예측 항목 -> 허용값. `validate_strategy` 와 프롬프트가 같은 표를 쓴다.
FIELDS: dict[str, tuple[str, ...]] = {
    # 체결 후 매수측이 움직이나
    "bid_move_bps": MOVE_VALUES,
    # 게시가 대비 체결 시점 매수측. 음수면 시장이 내려간 뒤 체결됐다는 뜻이다
    "fill_slippage_bps": SLIPPAGE_VALUES,
    # 체결이 결정 전체에서 차지하는 비율
    "fill_rate": RATE_VALUES,
    # 체결된 자리가 못 체결한 자리보다 넓은 스프레드인가
    "fill_spread_selection": COMPARE_VALUES,
    # 체결된 것이 못 체결한 것보다 더 올랐나 (기회 단위 선택)
    "fill_opportunity_selection": COMPARE_VALUES,
    # 손실이 "한 번도 안 오름" 인가 "올랐는데 부족" 인가
    "loss_kind": LOSS_KIND_VALUES,
}


# 각 항목에서 **가설에 유리한 답**. 적중과 별개로, 예측이 유리한 쪽으로 쏠렸는지를 센다.
# 정합 점수가 낮을 때 "가설이 틀렸다" 와 "낙관해서 틀렸다" 를 가르는 데 쓴다.
# `fill_rate` 는 높은 쪽이 유리하다고 말할 수 없어 뺀다 — 많이 체결되는 것이
# 좋은 것인지가 그 자체로 이 연구의 질문이다.
FAVOURABLE = {
    "bid_move_bps": "POSITIVE",              # 사고 나서 오른다
    "fill_slippage_bps": "NEUTRAL",          # 불리한 값에 체결되지 않는다
    "fill_spread_selection": "MORE",         # 넓은 스프레드 자리를 골라 잡는다
    "fill_opportunity_selection": "MORE",    # 체결된 것이 못 체결한 것보다 더 오른다
    "loss_kind": "ROSE_THEN_LOST",           # 져도 한 번은 올랐다
}


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.loc[values.notna()]
    return float(values.median()) if len(values) else None


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    """진입 후 방향만 평균으로 잰다.

    체결당 bid 이동은 작은 손실이 많고 큰 이익이 적은 모양이다 (OOS 35,404건에서
    중앙값 -10.08, 평균 -0.12, 양수 33.8%). 중앙값은 "보통 거래" 를, 평균은 "돈이
    되는 쪽" 을 말한다. 가설이 "오른다" 고 할 때 뜻하는 것은 후자다. 중앙값으로 재면
    POSITIVE 가 162건 내내 한 번도 안 나와 예측이 구조적으로 빗나갔다.
    """
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.loc[values.notna()]
    return float(values.mean()) if len(values) else None


def _move(value: float | None) -> str | None:
    if value is None:
        return None
    if value > MOVE_DEAD_ZONE_BPS:
        return "POSITIVE"
    if value < -MOVE_DEAD_ZONE_BPS:
        return "NEGATIVE"
    return "NEUTRAL"


def observe(frame: pd.DataFrame, *, unfilled_available: bool = True) -> dict[str, Any]:
    """원장에서 예측 여섯 항목의 실제 값을 뽑는다. 못 재면 None.

    `unfilled_available=False` 는 부르는 쪽이 이미 체결만 남기고 걸렀다는 뜻이다.
    미체결과 견주는 세 항목은 그 프레임에서 잴 수 없다. 걸러서 100%가 나온 값을
    관측인 척하지 않고 None 을 낸다.
    """
    if not len(frame) or "status" not in frame.columns:
        return {name: None for name in FIELDS}
    status = frame["status"].astype(str)
    filled = frame.loc[status.eq(ledger.FILLED)]
    unfilled = frame.loc[status.eq(ledger.UNFILLED)]
    decisions = int(len(filled)) + int(len(unfilled))
    usable_unfilled = unfilled_available and len(unfilled) > 0

    def compare(left: float | None, right: float | None) -> str | None:
        if left is None or right is None:
            return None
        gap = left - right
        return ("MORE" if gap > MOVE_DEAD_ZONE_BPS
                else "LESS" if gap < -MOVE_DEAD_ZONE_BPS else "SAME")

    loss_kind = None
    if "diagnostic_cohort" in filled.columns and len(filled):
        cohorts = filled["diagnostic_cohort"].astype(str)
        never = int(cohorts.eq(outcome.PERSISTENT_ADVERSE).sum())
        rose = int(cohorts.isin((outcome.COST_INSUFFICIENT,
                                 outcome.EARLY_RECOVERY_LATE_REVERSAL)).sum())
        if never + rose:
            share = never / (never + rose)
            loss_kind = ("PERSISTENT_ADVERSE" if share > 0.5 + SHARE_TIE_BAND
                         else "ROSE_THEN_LOST" if share < 0.5 - SHARE_TIE_BAND else "MIXED")

    return {
        "bid_move_bps": _move(_mean(filled, "bid_move_bps")),
        "fill_slippage_bps": _move(_median(filled, "fill_slippage_bps")),
        "fill_rate": (("HIGH" if len(filled) / decisions > FILL_RATE_HIGH else "LOW")
                      if decisions and usable_unfilled else None),
        "fill_spread_selection": (compare(_median(filled, "spread_bps"),
                                          _median(unfilled, "spread_bps"))
                                  if usable_unfilled else None),
        "fill_opportunity_selection": (
            compare(_median(filled, "max_favorable_gross_bps"),
                    _median(unfilled, "counterfactual_max_favorable_gross_bps"))
            if usable_unfilled else None),
        "loss_kind": loss_kind,
    }


def evaluate(frame: pd.DataFrame,
             prediction: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """예측과 관측을 항목별로 맞춘다. 예측이 없으면 None."""
    if not isinstance(prediction, Mapping) or not prediction:
        return None
    actual = observe(frame)
    items = []
    hits = measurable = 0
    for name in FIELDS:
        predicted = prediction.get(name)
        observed = actual.get(name)
        hit = None if (predicted is None or observed is None) else bool(predicted == observed)
        if hit is not None:
            measurable += 1
            hits += int(hit)
        items.append({"field": name, "predicted": predicted, "actual": observed, "hit": hit})
    scored = [item for item in items if item["hit"] is not None
              and item["field"] in FAVOURABLE]
    optimism = {
        "fields": len(scored),
        "predicted_favourable": sum(1 for item in scored
                                    if item["predicted"] == FAVOURABLE[item["field"]]),
        "actual_favourable": sum(1 for item in scored
                                 if item["actual"] == FAVOURABLE[item["field"]]),
        "note": "예측이 유리한 답으로 쏠린 정도. 적중과 별개이며 선택에 쓰지 않는다",
    }
    if not measurable:
        verdict = NOT_MEASURABLE
    elif hits >= ALIGNED_MIN_HITS:
        verdict = ALIGNED
    elif hits <= CONTRADICTED_MAX_HITS:
        verdict = CONTRADICTED
    else:
        verdict = PARTIAL
    return {
        "schema": SCHEMA_VERSION,
        "items": items,
        "hit_count": hits,
        "measurable_count": measurable,
        "total": len(FIELDS),
        "verdict": verdict,
        "optimism": optimism,
        "thresholds": {
            "move_dead_zone_bps": MOVE_DEAD_ZONE_BPS,
            "share_tie_band": SHARE_TIE_BAND,
            "fill_rate_high": FILL_RATE_HIGH,
            "aligned_min_hits": ALIGNED_MIN_HITS,
            "contradicted_max_hits": CONTRADICTED_MAX_HITS,
        },
        "note": "정합은 기록이다. Discovery 선택은 net_bps_total 하나로 한다",
    }
