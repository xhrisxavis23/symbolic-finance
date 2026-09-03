"""후보 선택. 계획서 §3 S4 — 분모 함정.

정본 저장소가 계약 1,025개에서 실측한 것: 전부 손실 구간이었고, 분자가 음수면
`net/분모` 는 분모가 클수록 0 에 가까워져 좋아 보인다. 그래서 **더 많이 잃은
후보가 신호를 촘촘히 켰다는 이유로** 이겼다. 가드는 정반대로 분모를 줄여서 이긴다.

그래서 두 가지를 건다 — `scorable` 하한, 그리고 부호 판정은 `total_net_bps` 로만.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SCORABLE = 200
COHORTS = ("NET_RECOVERY", "EARLY_RECOVERY_LATE_REVERSAL",
           "COST_INSUFFICIENT", "PERSISTENT_ADVERSE")


def summarise(ledger: pd.DataFrame) -> pd.DataFrame:
    """계약별 성과. 분모가 다른 수를 나란히 들고 다닌다."""
    rows = []
    for contract, group in ledger.groupby("contract_id", sort=True):
        status = group["status"]
        fills = int((status == "FILLED").sum())
        unfilled = int((status == "UNFILLED").sum())
        censored = int((status == "CENSORED").sum())
        scorable = fills + unfilled          # CENSORED 는 관측 불가라 분모에서 뺀다
        net = float(group.loc[status == "FILLED", "net_bps"].sum())
        record = {
            "contract_id": contract,
            "decisions": int(len(group)),
            "scorable": scorable,
            "fills": fills,
            "unfilled": unfilled,
            "censored": censored,
            "total_net_bps": net,
            "bps_per_decision": (net / scorable) if scorable else np.nan,
            "positive": bool(scorable and net > 0.0),
        }
        cohorts = group.loc[status == "FILLED", "cohort"] if "cohort" in group else pd.Series(dtype=object)
        for name in COHORTS:
            record[f"cohort_{name}"] = int((cohorts == name).sum())
        rows.append(record)
    return pd.DataFrame(rows).set_index("contract_id")


def rank(summary: pd.DataFrame, min_scorable: int = MIN_SCORABLE) -> pd.DataFrame:
    """자격을 통과한 후보만, 총액 기준으로.

    비율 지표는 동률 처리에만 쓴다. 손실 구간에서 비율만 보면 방향을 잃는다.
    """
    eligible = summary[summary["scorable"] >= int(min_scorable)].copy()
    return eligible.sort_values(
        by=["total_net_bps", "bps_per_decision"],
        ascending=[False, False], kind="mergesort")
