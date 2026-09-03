"""종목군-종목-일 단위로 쪼갠 Feature Profile 저장소.

`evidence.build()` 는 클러스터 하나를 통째로 계산해 산출물을 낸다. 그 계산의 바닥에
있는 것은 `anchor_evidence` — (anchor x 시점 x feature) 한 장이고, 위층(canonical ·
contrastive · groups · temporal · balanced · alignment)은 전부 이것에서 유도된다.
그래서 이 한 장만 쪼개 두면 나머지는 언제든 다시 만든다.

    framework/FeatureProfile/<종목군>/<종목>/<MM_DD>/profile.parquet
                                                    meta.json

종목군이 맨 위인 이유는 그것이 이 연구의 분석 단위이기 때문이다 — 증거·가설·검증이
전부 클러스터마다 따로 돈다. 종목 하나만 보려면 `read(symbols=[...])` 가 전 종목군을
훑으므로 경로를 몰라도 된다.

폴더 이름에 연도가 없다. 원 날짜(`20260316`)는 `profile.parquet` 의 `date` 열과
`meta.json` 에 그대로 남으므로 정보가 사라지지는 않는다.

`read()` 가 돌려주는 `(long, anchors)` 는 `evidence.build()` 안에서 쓰이는 것과 같은
모양이다. 그래서 캐시나 틱 데이터를 다시 읽지 않고 그 자리에서 위층을 만든다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import catalog
from .config import now_utc, read_json, write_json


STORE = Path(__file__).with_name("FeatureProfile")

# `anchors` 표를 복원하는 데 필요한 열. `evidence.build_anchors` 의 출력과 같다.
OUTCOME_COLUMNS = (
    # Legacy oracle diagnostic fields. They remain nullable in execution-aligned stores.
    "best_ask_net_bps", "entry_queue_status", "entry_terminal_tick", "entry_fill_tick",
    "entry_wait_ticks", "entry_wait_seconds", "entry_queue_fraction",
    "entry_oracle_ask_tick", "entry_fill_before_oracle_ask",
    # Execution-aligned label provenance. Keeping these beside generic cohort preserves
    # old Evidence code while making the stronger cohort semantics explicit to the Agent.
    "execution_cohort", "entry_assumption", "exit_profile_id",
    "canonical_exit_net_bps", "canonical_exit_tick", "canonical_exit_reason",
    "holding_seconds", "outcome_observable", "anchor_source",
)
ANCHOR_COLUMNS = ("cohort", "cluster_id", "symbol", "date", "tick",
                  *OUTCOME_COLUMNS, "anchor_id")


def folder(cluster: str, symbol: str, date: str, root: Path = STORE) -> Path:
    """`MK01, 003380, 20260316` -> `<root>/MK01/003380/03_16`."""
    day = str(date)
    return Path(root) / str(cluster) / str(symbol).zfill(6) / f"{day[4:6]}_{day[6:8]}"


def write(long: pd.DataFrame, anchors: pd.DataFrame, *, config: Any,
          root: Path = STORE, schema: str = "feature_profile_symbol_day.v1",
          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """`anchor_evidence` 를 종목군-종목-일로 쪼개 쓴다. 같은 칸이 있으면 덮어쓴다."""
    outcome = anchors.reindex(columns=["anchor_id", *OUTCOME_COLUMNS])
    merged = long.merge(outcome, on="anchor_id", how="left")
    written = []
    for (cluster, symbol, date), part in merged.groupby(
            ["cluster_id", "symbol", "date"], sort=True):
        out = folder(cluster, symbol, date, root)
        out.mkdir(parents=True, exist_ok=True)
        part = part.sort_values(["anchor_id", "relative_time_ms"]).reset_index(drop=True)
        part.to_parquet(out / "profile.parquet", index=False)
        counts = part.drop_duplicates("anchor_id").cohort.value_counts()
        write_json(out / "meta.json", {
            "schema": str(schema),
            "created_at": now_utc(),
            "cluster_id": str(cluster), "symbol": str(symbol), "date": str(date),
            "anchors": int(part.anchor_id.nunique()),
            "cohorts": {str(k): int(v) for k, v in counts.items()},
            "relative_times_ms": sorted(int(v) for v in part.relative_time_ms.unique()),
            "catalog_hash": catalog.catalog_hash(),
            "profiler_config": {k: list(v) if isinstance(v, tuple) else v
                                for k, v in vars(config).items()},
            **dict(metadata or {}),
        })
        written.append({"cluster_id": str(cluster), "symbol": str(symbol),
                        "date": str(date), "rows": int(len(part))})
    manifest(root)
    return {"root": str(root), "written": len(written), "rows": int(len(merged)),
            "entries": written}


def manifest(root: Path = STORE) -> dict[str, Any]:
    """저장소 전체를 훑어 `manifest.json` 을 다시 쓴다.

    넘겨받은 목록이 아니라 **파일에서** 만든다. 종목군을 나눠 여러 번 `write()` 해도
    마지막 호출이 앞의 기록을 지우지 않는다.
    """
    entries = [dict(read_json(folder(c, s, d, root) / "meta.json"),
                    path=str(folder(c, s, d, root)))
               for c, s, d in available(root)]
    payload = {
        "schema": "feature_profile_store.v1",
        "created_at": now_utc(),
        "catalog_hash": catalog.catalog_hash(),
        "layout": "<종목군>/<종목>/<MM_DD>",
        "symbol_days": len(entries),
        "clusters": sorted({e["cluster_id"] for e in entries}),
        "symbols": sorted({e["symbol"] for e in entries}),
        "dates": sorted({e["date"] for e in entries}),
        "anchors": sum(int(e["anchors"]) for e in entries),
        "entries": entries,
    }
    write_json(Path(root) / "manifest.json", payload)
    return payload


def available(root: Path = STORE) -> list[tuple[str, str, str]]:
    """저장소에 실제로 있는 (종목군, 종목, 날짜). manifest 가 아니라 파일에서 센다."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        (cluster.name, symbol.name, str(read_json(day / "meta.json")["date"]))
        for cluster in root.iterdir() if cluster.is_dir()
        for symbol in cluster.iterdir() if symbol.is_dir()
        for day in symbol.iterdir() if (day / "meta.json").exists())


def read(clusters: Sequence[str] | None = None, symbols: Sequence[str] | None = None,
         dates: Sequence[str] | None = None,
         root: Path = STORE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """저장소에서 `(long, anchors)` 를 복원한다. `evidence` 의 위층에 그대로 넣는다."""
    keep = available(root)
    if clusters is not None:
        want = {str(c) for c in clusters}
        keep = [t for t in keep if t[0] in want]
    if symbols is not None:
        want = {str(s).zfill(6) for s in symbols}
        keep = [t for t in keep if t[1] in want]
    if dates is not None:
        want = {str(d) for d in dates}
        keep = [t for t in keep if t[2] in want]
    if not keep:
        raise FileNotFoundError(
            f"저장소에 맞는 칸이 없다: clusters={clusters} symbols={symbols} dates={dates}")
    merged = pd.concat(
        [pd.read_parquet(folder(c, s, d, root) / "profile.parquet") for c, s, d in keep],
        ignore_index=True)
    for column in OUTCOME_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    anchors = merged.drop_duplicates("anchor_id")[list(ANCHOR_COLUMNS)].reset_index(drop=True)
    long = merged.drop(columns=[column for column in OUTCOME_COLUMNS if column in merged]).reset_index(drop=True)
    return long, anchors


def feature_columns(long: pd.DataFrame) -> tuple[str, ...]:
    """저장된 표에서 카탈로그 feature 열만."""
    return tuple(c for c in long.columns if c in catalog.FEATURES)
