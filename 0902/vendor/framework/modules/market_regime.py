"""종목-일 수익률 라벨에서 날짜 단위 시장 상태를 고정한다."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..config import now_utc, write_json


UP, DOWN, SIDEWAYS = "UP", "DOWN", "SIDEWAYS"
THRESHOLD_BPS = 20.0


def build(source: Path, output: Path, *, threshold_bps: float = THRESHOLD_BPS) -> pd.DataFrame:
    """그날의 전체 종목 수익률 중앙값으로 날짜 상태를 만든다.

    개별 종목 라벨의 다수결이 아니라 원 수익률 중앙값을 쓴다. 급등락 소수 종목이
    시장 상태를 뒤집지 않게 하기 위해서다.
    """
    source, output = Path(source), Path(output)
    frame = pd.read_csv(source, dtype={"date": str})
    usable = frame.loc[frame["status"].eq("LABELED"), ["date", "return_bps"]].copy()
    grouped = usable.groupby("date", as_index=False)["return_bps"].agg(
        labeled_symbols="count", median_return_bps="median", mean_return_bps="mean")
    grouped["regime"] = grouped["median_return_bps"].map(
        lambda value: UP if value > threshold_bps else
        (DOWN if value < -threshold_bps else SIDEWAYS))
    grouped = grouped.sort_values("date").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output / "market_day_labels.csv", index=False)
    write_json(output / "market_day_labels.json", {
        "schema": "market_day_labels.v1", "created_at": now_utc(),
        "source": str(source), "method": "cross_sectional_median_return_bps",
        "threshold_bps": float(threshold_bps), "rows": grouped.to_dict(orient="records"),
    })
    return grouped


def choose(frame: pd.DataFrame, dates: Sequence[str], regime: str, count: int) -> tuple[str, ...]:
    """허용 구간에서 상태가 같은 날짜를 시간순으로 고른다."""
    matched = frame.loc[frame["date"].isin({str(date) for date in dates})
                        & frame["regime"].eq(str(regime)), "date"].tolist()
    if len(matched) < count:
        raise ValueError(f"{regime} 날짜가 {count}일보다 적다: {matched}")
    return tuple(matched[:count])
