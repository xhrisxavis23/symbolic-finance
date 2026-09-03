"""FeatureProfile 저장소에서 단계 입력을 명시적으로 고른다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .. import evidence, featureprofile, profit
from ..config import PROFIT_CACHE, TICK_ROOT


@dataclass(frozen=True)
class ProfileSelection:
    clusters: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    store: Path = featureprofile.STORE
    discovery_profit_root: Path | None = None
    profile_kind: str = "legacy_oracle"

    def as_input(self) -> dict[str, Any]:
        value = asdict(self)
        value["store"] = str(self.store)
        value["discovery_profit_root"] = (
            str(self.discovery_profit_root) if self.discovery_profit_root is not None else None)
        return value


def load(selection: ProfileSelection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """선택된 profile만 읽는다. 비어 있으면 저장소의 모든 종목으로 넓히지 않는다."""
    if not selection.clusters:
        raise ValueError("최소 한 cluster를 지정해야 한다")
    return featureprofile.read(
        clusters=selection.clusters,
        symbols=selection.symbols or None,
        dates=selection.dates or None,
        root=Path(selection.store),
    )


def materialize_execution_aligned(selection: ProfileSelection, output: Path, *,
                                    root: Path = TICK_ROOT, config: Any = None) -> dict[str, Any]:
    """Sandbox_6 run_006 contract, promoted to a first-class framework Profile builder."""
    if len(selection.clusters) != 1 or not selection.symbols or not selection.dates:
        raise ValueError("execution-aligned Profile은 cluster 하나와 symbols/dates를 명시해야 한다")
    from .. import execution_profile
    active = execution_profile.ExecutionProfileConfig() if config is None else config
    result = execution_profile.materialize(
        str(selection.clusters[0]), selection.symbols, selection.dates, Path(output),
        config=active, root=Path(root))
    return {**result, "selection": selection.as_input()}


def materialize(selection: ProfileSelection, output: Path, *, root: Path = TICK_ROOT,
                profit_root: Path = PROFIT_CACHE,
                config: evidence.ProfilerConfig = evidence.ProfilerConfig()) -> dict[str, Any]:
    """요청한 중심 종목·발굴일의 FeatureProfile을 독립 저장소에 만든다.

    먼저 BID1 진입·ASK1 청산 기준 수익구간을 만든다. 선택 종목 중 하나라도 그 날의
    PROFIT anchor가 없으면 조용히 빼지 않고 실패로 돌려준다.
    """
    if str(selection.profile_kind) == "execution_aligned":
        return materialize_execution_aligned(selection, output, root=Path(root))
    if str(selection.profile_kind) != "legacy_oracle":
        raise ValueError(f"알 수 없는 profile_kind: {selection.profile_kind}")
    if len(selection.clusters) != 1 or not selection.symbols or not selection.dates:
        raise ValueError("실험 Profile은 cluster 하나와 종목·발굴 날짜를 명시해야 한다")
    cluster = str(selection.clusters[0])
    symbols = tuple(str(symbol).zfill(6) for symbol in selection.symbols)
    dates = tuple(str(date) for date in selection.dates)
    for symbol in symbols:
        for date in dates:
            profit.materialize(cluster, symbol, date, root=Path(profit_root), tick_root=Path(root))
    anchors, notes = evidence.build_anchors(
        config, clusters=(cluster,), symbols=symbols, dates=dates, root=Path(root),
        profit_root=Path(profit_root))
    found = {(str(symbol), str(date)) for symbol, date in
             anchors[["symbol", "date"]].drop_duplicates().to_numpy()}
    expected = {(symbol, date) for symbol in symbols for date in dates}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"quality discovery sample이 없는 중심 종목-일: {missing}")
    long = evidence.profile_anchors(anchors, config, root=Path(root))
    result = featureprofile.write(long, anchors, config=config, root=Path(output))
    return {**result, "anchors": int(len(anchors)), "notes": notes,
            "selection": selection.as_input(), "profit_root": str(profit_root)}
