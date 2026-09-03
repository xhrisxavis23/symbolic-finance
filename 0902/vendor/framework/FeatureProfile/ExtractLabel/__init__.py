"""BID1 매수 지정가·미래 ASK1 매도 지정가 기준 수익 구간 추출.

    from framework.FeatureProfile.ExtractLabel import extract, extract_quality
    labels, regions, meta = extract("005930", "20260317")          # 수익 구간
    labels, quality, meta = extract_quality("005930", "20260317")  # 품질 통과 구간

수익 구간은 문턱이 낮다 — 창 안에서 한 번이라도 수수료를 넘으면 센다. 품질 통과 구간은
그중 ASK1 가격 기회가 지속되는 것만 남긴다. 가설의 anchor 로 쓰는 것은 후자다
(`framework/evidence.py` 의 `build_anchors` 참고).

가격 기회 라벨은 미래를 보므로 발굴에만 쓴다. 실제 지정가 체결은 canonical backtest가
큐 재생으로 따로 판정한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ... import data as tickdata
from ...config import FEE_BPS, TICK_ROOT
from .extract import (
    PATH_SCHEMA, REGION_SCHEMA, REGION_COLUMNS, STANDARD_HORIZONS, HorizonSpec,
    fill_short_false_gaps, fill_short_false_gaps_dual, find_contiguous_runs,
    packed_time_to_us, path_labels, regions,
)
from .quality import (
    CLEAN_SAFE, OUTCOME_NAMES, QUALITY_REGION_COLUMNS, QUALITY_REGION_SCHEMA,
    QualityContract, classify_outcomes, clean_safe, quality_regions,
)

__all__ = [
    "PATH_SCHEMA", "REGION_SCHEMA", "REGION_COLUMNS", "STANDARD_HORIZONS", "HorizonSpec",
    "DEFAULT_CONTRACT", "fill_short_false_gaps", "fill_short_false_gaps_dual",
    "find_contiguous_runs", "packed_time_to_us", "path_labels", "regions", "extract",
    "QUALITY_REGION_SCHEMA", "QUALITY_REGION_COLUMNS", "OUTCOME_NAMES", "CLEAN_SAFE",
    "QualityContract", "classify_outcomes", "quality_regions", "clean_safe",
    "extract_quality",
]


DEFAULT_CONTRACT = {
    "fee_bps_rt": FEE_BPS,
    "tradeable_edge_bps": 0.0,
    "min_run_ticks": 5,
    "max_gap_ticks": 2,
}


def extract(symbol: str, date: str, *,
            horizons: Iterable[HorizonSpec] = STANDARD_HORIZONS,
            root: Path = TICK_ROOT,
            **contract: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """한 종목-일을 읽어 층 0 라벨과 층 1 구간을 만든다.

    `contract` 로 넘긴 것만 `DEFAULT_CONTRACT` 를 덮는다. 무엇으로 계산했는지는
    돌려주는 요약의 `contract` 에 남는다.
    """
    rule = {**DEFAULT_CONTRACT, **contract}
    arrays, path = tickdata.load(symbol, date, root)
    labels = path_labels(arrays, horizons)
    frame, summaries = regions(labels, horizons, **rule)
    return labels, frame, {"symbol": str(symbol), "date": str(date),
                           "source": str(path), "ticks": int(len(labels)),
                           "contract": rule, "horizons": summaries}


def extract_quality(symbol: str, date: str, *,
                    horizons: Iterable[HorizonSpec] = STANDARD_HORIZONS,
                    contract: QualityContract | None = None,
                    apply_clean_safe: bool = True,
                    root: Path = TICK_ROOT
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """한 종목-일의 **품질 통과 구간**. 가설 anchor 로 쓰는 층이다.

    `apply_clean_safe` 는 마지막 걸러내기까지 적용한다. 끄면 걸러내기 전 구간이 나오고,
    `clean_safe()` 를 직접 불러 조건을 바꿔 볼 수 있다.
    """
    rule = contract or QualityContract()
    arrays, path = tickdata.load(symbol, date, root)
    labels = path_labels(arrays, horizons)
    frame, summaries = quality_regions(labels, horizons, rule)
    meta: dict[str, Any] = {"symbol": str(symbol), "date": str(date),
                            "source": str(path), "ticks": int(len(labels)),
                            "contract": rule.as_dict(), "horizons": summaries,
                            "clean_safe": None}
    if apply_clean_safe and len(frame):
        before = len(frame)
        frame = clean_safe(frame.assign(fee_bps_rt=rule.fee_bps_rt))
        meta["clean_safe"] = {"rule": dict(CLEAN_SAFE),
                              "before": before, "after": int(len(frame))}
    return labels, frame, meta
