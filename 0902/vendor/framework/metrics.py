"""성과 지표. **분모 정의가 이 파일에만 있다.**

구 구조는 같은 이름의 지표를 세 곳에서 따로 구현했다 (`ledger._metrics`,
`diagnose._economic_replay`, 청산 후보 채점부). 그것을 쓰는 곳은 여섯 곳이었다 —
계약 선택, 가드 선택, 청산 선택, 재현 확인, "양수 관측" 판정, Parent 대조. 분모를
바꾸려면 여섯 곳을 다 찾아야 했고 하나라도 놓치면 조용히 갈라졌다.

## 왜 분모가 문제인가

구 분모는 신호가 켜진 틱 수였다. 그 값은 다음에 좌우된다.

  · 신호를 얼마나 오래 켜 두는가 (계약마다 다름)
  · 포지션 보유중 신호가 얼마나 겹치는가 (실측 중앙값 79%, 어떤 계약은 96%)

실측으로 드러난 것 둘.

  1. 이 저장소의 계약 1,025개가 전부 손실이었다. 분자가 음수면 `net/분모` 는 분모가
     클수록 0 에 가까워져 좋아 보인다. 그래서 계약 선택에서는 **더 많이 잃은 쪽이
     신호를 촘촘히 켰다는 이유로** 이겼다 (갈라진 record 50개 중 40개).
  2. 가드는 반대로 분모를 줄여서 이긴다. 검증 구간에서 두 지표의 부호가 갈렸다 —
     bps/attempt 로는 악화, bps/decision 으로는 개선.

그래서 지표를 하나로 정하지 않고 **함께 보고**하며, 어떤 지표로 골랐는지 기록한다.
손실 구간에서 비율 지표만 보면 방향을 잃는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .ledger import CENSORED, FILLED, UNFILLED


TIE = 1e-9


@dataclass(frozen=True)
class Metrics:
    """한 계약(또는 후보)의 성과. 분모가 다른 수를 나란히 들고 다닌다."""

    decisions: int          # 새 진입을 결정할 수 있었던 횟수
    scorable: int           # 그중 결과를 관측할 수 있는 것 = 분모
    fills: int
    censored: int
    unfilled: int
    blocked_ticks: int      # 원장에 행이 없다. 진단용
    signal_ticks: int       # 구 `signal_attempts`. 비교용으로만
    episodes: int
    total_net_bps: float
    total_gross_bps: float
    days: int
    symbols: int
    exact: bool = True      # 마스크 근사로 잰 값이면 False

    # -- 지표 ---------------------------------------------------------------
    @staticmethod
    def _ratio(numerator: float, denominator: int) -> float | None:
        """분모가 0 이면 None. 0.0 으로 두면 음수인 정상 후보를 전부 이긴다."""
        return float(numerator) / denominator if denominator else None

    @property
    def bps_per_decision(self) -> float | None:
        """기본 지표. 채점 가능한 결정 하나당 손익."""
        return self._ratio(self.total_net_bps, self.scorable)

    @property
    def bps_per_fill(self) -> float | None:
        """체결 하나당 손익. taker 에서는 `bps_per_decision` 과 같다 (미체결이 0이므로)."""
        return self._ratio(self.total_net_bps, self.fills)

    @property
    def bps_per_day(self) -> float | None:
        """기간 규모. 비율 지표가 부호 때문에 뒤집힐 때 방향을 준다."""
        return self._ratio(self.total_net_bps, self.days)

    @property
    def legacy_bps_per_attempt(self) -> float | None:
        """구 지표. **선택에 쓰지 않는다.** 구 결과와 대조할 때만."""
        return self._ratio(self.total_net_bps, self.signal_ticks)

    @property
    def decisions_per_episode(self) -> float | None:
        """긴 신호 하나가 몇 번 재진입했나. episode 와 decision 은 다르다."""
        return self._ratio(self.decisions, self.episodes)

    @property
    def blocked_ratio(self) -> float | None:
        """신호 틱 중 포지션 때문에 아무것도 못 한 비율."""
        return self._ratio(self.blocked_ticks, self.signal_ticks)

    @property
    def positive(self) -> bool:
        """돈을 벌었나. 총액과 결정당 둘 다 양수여야 한다.

        구 구조는 `bps_per_attempt` 까지 셋을 다 봤는데, 그 수는 분모 때문에 총액과
        부호가 갈릴 수 있어 판정에 넣으면 안 된다.
        """
        return bool(self.scorable and self.total_net_bps > 0.0
                    and (self.bps_per_decision or 0.0) > 0.0)

    def as_dict(self) -> dict[str, Any]:
        """기록용. 파생 지표를 함께 담는다 — 하나만 남기면 나중에 재계산 못 한다."""
        return {
            **asdict(self),
            "bps_per_decision": self.bps_per_decision,
            "bps_per_fill": self.bps_per_fill,
            "bps_per_day": self.bps_per_day,
            "legacy_bps_per_attempt": self.legacy_bps_per_attempt,
            "decisions_per_episode": self.decisions_per_episode,
            "blocked_ratio": self.blocked_ratio,
            "positive": self.positive,
        }


EMPTY = Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0)


def _partition_totals(manifest: Mapping[str, Any], contract_id: str | None,
                      keys: set[tuple[str, str]] | None = None) -> dict[str, int]:
    """manifest 에서 신호·episode·blocked 를 모은다. 이 셋은 원장에 행이 없다."""
    out = {"raw_signal_ticks": 0, "signal_episodes": 0, "blocked_signal_ticks": 0}
    for part in manifest.get("by_partition", []):
        if contract_id is not None and part["contract_id"] != contract_id:
            continue
        if keys is not None and (part["symbol"], part["date"]) not in keys:
            continue
        for key in out:
            out[key] += int(part.get(key, 0))
    return out


def measure(frame: pd.DataFrame, manifest: Mapping[str, Any],
            contract_id: str | None = None) -> Metrics:
    """원장 조각 하나를 잰다. 정확한 값.

    `blocked_ticks`·`signal_ticks`·`episodes` 는 원장에 행이 없으므로 manifest 에서
    읽는다. 그래서 manifest 없이는 잴 수 없다 — 일부러 그렇게 뒀다. 구 지표를
    대조하려면 그 셋이 있어야 하고, 없이 재면 다시 분모가 흩어진다.
    """
    if contract_id is not None and len(frame):
        frame = frame.loc[frame["contract_id"].eq(contract_id)]
    if not len(frame):
        return Metrics(**{**asdict(EMPTY), **_totals_to_fields(_partition_totals(manifest, contract_id))})

    status = frame["status"].to_numpy()
    filled = status == FILLED
    net = pd.to_numeric(frame.get("net_bps"), errors="coerce").to_numpy(dtype=float) \
        if "net_bps" in frame.columns else np.zeros(len(frame))
    gross = pd.to_numeric(frame.get("gross_bps"), errors="coerce").to_numpy(dtype=float) \
        if "gross_bps" in frame.columns else np.zeros(len(frame))
    totals = _partition_totals(manifest, contract_id)
    return Metrics(
        decisions=int(len(frame)),
        scorable=int((filled | (status == UNFILLED)).sum()),
        fills=int(filled.sum()),
        censored=int((status == CENSORED).sum()),
        unfilled=int((status == UNFILLED).sum()),
        blocked_ticks=totals["blocked_signal_ticks"],
        signal_ticks=totals["raw_signal_ticks"],
        episodes=totals["signal_episodes"],
        total_net_bps=float(np.nansum(net[filled])),
        total_gross_bps=float(np.nansum(gross[filled])),
        days=int(frame["date"].nunique()),
        symbols=int(frame["symbol"].nunique()),
    )


def _totals_to_fields(totals: Mapping[str, int]) -> dict[str, int]:
    return {"blocked_ticks": totals["blocked_signal_ticks"],
            "signal_ticks": totals["raw_signal_ticks"],
            "episodes": totals["signal_episodes"]}


def measure_masked(frame: pd.DataFrame, keep: np.ndarray,
                   manifest: Mapping[str, Any], contract_id: str | None = None) -> Metrics:
    """조건 하나를 씌운 뒤의 성과. **근사값이다 (`exact=False`).**

    남은 decision 사이의 겹침은 다시 계산한다 — 앞 진입이 지워지면 그 뒤에 가려져
    있던 decision 이 살아난다.

    근사인 이유: 이 원장은 BLOCKED 틱을 행으로 들고 있지 않다. 가드가 어떤 진입을
    지우면, 그 진입이 막고 있던 **신호 틱**중 일부가 새 decision 이 될 수 있는데
    그 틱들이 여기 없다. 구 원장은 그것까지 들고 있어서(그래서 22M 행이었다) 정확히
    다시 훑을 수 있었다.

    그래서 격자 채점은 두 단계로 한다.
      1. 여기서 190개 후보를 싸게 재고 순위를 매긴다 (근사)
      2. 상위 몇 개만 가드를 실제로 붙여 원장을 다시 만들고 `measure` 로 잰다 (정확)

    이 함수의 값으로 최종 판단을 하면 안 된다. `exact=False` 가 그 표시다.
    """
    if contract_id is not None and len(frame):
        mask = frame["contract_id"].eq(contract_id).to_numpy()
        frame, keep = frame.loc[mask], np.asarray(keep, dtype=bool)[mask]
    keep = np.asarray(keep, dtype=bool)
    if not len(frame):
        return Metrics(**{**asdict(EMPTY), **_totals_to_fields(_partition_totals(manifest, contract_id)),
                          "exact": False})

    ordered = frame.assign(_keep=keep).sort_values(["symbol", "date", "entry_tick"])
    kept = ordered.loc[ordered["_keep"]]
    if not len(kept):
        return Metrics(**{**asdict(EMPTY), **_totals_to_fields(_partition_totals(manifest, contract_id)),
                          "exact": False})

    # 겹침 재계산. 같은 (종목, 날) 안에서 앞 체결의 청산 틱까지는 다시 못 산다.
    partition = pd.factorize(pd.MultiIndex.from_frame(kept[["symbol", "date"]]), sort=False)[0]
    entry = kept["entry_tick"].to_numpy(dtype=np.int64)
    exit_tick = kept["exit_tick"].to_numpy(dtype=np.int64)
    status = kept["status"].to_numpy()
    net = pd.to_numeric(kept["net_bps"], errors="coerce").to_numpy(dtype=float)
    gross = pd.to_numeric(kept["gross_bps"], errors="coerce").to_numpy(dtype=float)

    decisions = fills = censored = 0
    total_net = total_gross = 0.0
    current, available = -1, -1
    for index in range(len(kept)):
        if partition[index] != current:
            current, available = int(partition[index]), -1
        if entry[index] < available:
            continue          # 남은 앞 진입에 가려졌다
        decisions += 1
        if status[index] == FILLED and np.isfinite(net[index]):
            fills += 1
            total_net += float(net[index])
            total_gross += float(gross[index]) if np.isfinite(gross[index]) else 0.0
            available = int(exit_tick[index]) + 1
        else:
            censored += 1
    totals = _partition_totals(manifest, contract_id)
    return Metrics(
        decisions=decisions, scorable=fills, fills=fills, censored=censored, unfilled=0,
        blocked_ticks=totals["blocked_signal_ticks"],
        signal_ticks=int(keep.sum()),   # 마스크 뒤의 신호 수는 알 수 없다. decision 기준으로 둔다
        episodes=int(kept["episode_index"].nunique()),
        total_net_bps=total_net, total_gross_bps=total_gross,
        days=int(kept["date"].nunique()), symbols=int(kept["symbol"].nunique()),
        exact=False)


def measure_masked_screening(frame: pd.DataFrame, keep: np.ndarray,
                             manifest: Mapping[str, Any], contract_id: str | None = None) -> Metrics:
    """Guard 후보 순위용 근사 지표다. 최종 선택은 실제 재생으로 다시 잰다."""
    return measure_masked(frame, keep, manifest, contract_id)


def compare(left: Metrics, right: Metrics, *, denominator: str = "scorable") -> dict[str, Any]:
    """두 성과의 차이를 **분자 효과와 분모 효과로 가른다.**

        점수차 = (net_L − net_R)/분모_L  +  net_R × (1/분모_L − 1/분모_R)
                 └── 분자 효과 ──┘        └──── 분모 효과 ────┘

    상관계수로는 이걸 못 본다. 같은 가설·같은 종목·같은 날 안에서 짝지어 봐야 한다.
    분모 효과 비중이 크면 그 선택은 거래 품질이 아니라 회계가 만든 것이다.
    """
    a, b = int(getattr(left, denominator)), int(getattr(right, denominator))
    if not a or not b:
        return {"comparable": False, "reason": f"{denominator} 가 0 인 쪽이 있다"}
    numerator = (left.total_net_bps - right.total_net_bps) / a
    denom = right.total_net_bps * (1.0 / a - 1.0 / b)
    magnitude = abs(numerator) + abs(denom)
    return {
        "comparable": True,
        "denominator": denominator,
        "score_gap": left.total_net_bps / a - right.total_net_bps / b,
        "numerator_effect": numerator,
        "denominator_effect": denom,
        "denominator_share": abs(denom) / magnitude if magnitude > 0 else None,
        "left_denominator": a, "right_denominator": b,
        "left_net_bps": left.total_net_bps, "right_net_bps": right.total_net_bps,
    }


def top_set(candidates: Mapping[str, Metrics], metric: str = "bps_per_decision") -> set[str]:
    """그 지표의 공동 1등 전부.

    `max()` 로 하나를 강제하면 동률이 임의로 깨진다. 계약 5개가 같은 신호를 내는
    경우가 record 의 70% 였으므로 동률은 예외가 아니라 기본이다.

    분모가 0 이라 `None` 인 후보는 애초에 후보가 아니다.
    """
    scored = {key: getattr(value, metric) for key, value in candidates.items()}
    scored = {k: float(v) for k, v in scored.items() if v is not None}
    if not scored:
        return set()
    best = max(scored.values())
    return {k for k, v in scored.items() if v >= best - TIE * max(abs(best), 1.0)}


def rank(candidates: Mapping[str, Metrics], metric: str = "bps_per_decision") -> dict[str, int]:
    """동률을 같은 순위로 주는 경쟁 순위 (1 = 공동 1등)."""
    scored = {k: getattr(v, metric) for k, v in candidates.items()}
    scored = {k: float(v) for k, v in scored.items() if v is not None}
    return {key: sum(1 for other in scored.values()
                     if other > value + TIE * max(abs(value), 1.0)) + 1
            for key, value in scored.items()}
