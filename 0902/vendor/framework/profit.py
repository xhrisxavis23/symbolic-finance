"""BID1 매수 지정가와 미래 ASK1 매도 지정가 기준 수익 구간 저장소.

수익구간은 가격 기회를 찾는 미래 라벨이다. 실제 매수·매도 체결은 여기서 가정하지
않고 canonical backtest의 큐 재생에서만 판정한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from . import data as tickdata, universe
from .FeatureProfile.ExtractLabel import (DEFAULT_CONTRACT, STANDARD_HORIZONS,
                                          HorizonSpec, path_labels as build_path_labels,
                                          regions as extract_regions)
from .FeatureProfile.ExtractLabel.quality import (QualityContract, clean_safe,
                                                   quality_regions as extract_quality)
from .config import PROFIT_CACHE, TICK_ROOT, now_utc, read_json, sha256_file, write_json


CACHE_SCHEMA = "maker_profit_region_store.v2"
PATH_BASE_COLUMNS = ("tick_idx", "recv_ts_kst", "time_us", "bid_1", "ask_1")
PATH_HORIZON_FIELDS = (
    "complete", "window_end_delta_ticks", "best_ask_gross_bps",
    "best_ask_delta_ticks", "best_ask_seconds", "fixed_ask_gross_bps",
)


class CacheError(RuntimeError):
    """수익 구간 저장소가 현재 가격 방향과 맞지 않는다."""


def _config_path(root: Path) -> Path:
    return Path(root) / "cache_config.json"


def _read_contract(root: Path) -> dict[str, Any]:
    path = _config_path(root)
    if not path.exists():
        raise CacheError(f"수익 구간이 아직 없다: {path.parent}. materialize()로 먼저 만든다")
    value = read_json(path)
    if value.get("schema") != CACHE_SCHEMA:
        raise CacheError("BID1 진입·ASK1 청산 기준이 아닌 수익 구간은 읽지 않는다")
    price = value.get("price_opportunity") or {}
    if price.get("entry_price") != "BID1" or price.get("exit_price") != "FUTURE_BEST_ASK1":
        raise CacheError("수익 구간 가격 방향이 현재 canonical 실행 모델과 다르다")
    return value


def contract(root: Path = PROFIT_CACHE) -> dict[str, Any]:
    return _read_contract(Path(root))


def dates(root: Path = PROFIT_CACHE) -> tuple[str, ...]:
    return tuple(str(value) for value in contract(root).get("dates", ()))


def horizons(root: Path = PROFIT_CACHE) -> tuple[str, ...]:
    return tuple(str(value["key"]) for value in contract(root)["horizons"])


def horizon_seconds(key: str, root: Path = PROFIT_CACHE) -> float | None:
    for row in contract(root)["horizons"]:
        if str(row["key"]) == str(key):
            return float(row["value"]) if row["kind"] == "seconds" else None
    raise KeyError(f"수익 구간에 없는 평가 창: {key}")


def _profile_scope(scope: str) -> str:
    """기존 클러스터나 독립 Profile 종목군에 쓸 저장소 이름을 확인한다."""
    value = str(scope)
    if value in universe.CLUSTERS:
        return value
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,31}", value):
        raise ValueError(f"Profile 종목군 이름이 올바르지 않다: {scope}")
    return value


def partition(cluster: str, symbol: str, date: str, root: Path = PROFIT_CACHE) -> Path:
    """클러스터 또는 명시적 Profile 종목군의 수익 구간 위치."""
    scope = _profile_scope(cluster)
    return Path(root) / "partitions" / scope / str(symbol).zfill(6) / str(date)


def available(cluster: str | None = None, root: Path = PROFIT_CACHE) -> list[tuple[str, str, str]]:
    base = Path(root) / "partitions"
    if not base.is_dir():
        return []
    groups = [base / cluster] if cluster is not None else sorted(base.iterdir())
    result = []
    for group in groups:
        if not group.is_dir():
            continue
        for symbol in sorted(group.iterdir()):
            if not symbol.is_dir():
                continue
            for day in sorted(symbol.iterdir()):
                if day.is_dir() and (day / "quality_regions.parquet").exists():
                    result.append((group.name, symbol.name, day.name))
    return result


def _read_partitions(filename: str, *, cluster: str, symbol: str | None, date: str | None,
                     root: Path) -> pd.DataFrame:
    pairs = [item for item in available(cluster, root)
             if (symbol is None or item[1] == str(symbol).zfill(6))
             and (date is None or item[2] == str(date))]
    if not pairs:
        raise CacheError(f"수익 구간이 없다: {cluster}/{symbol or '*'}/{date or '*'}")
    frames = [pd.read_parquet(partition(c, s, d, root) / filename) for c, s, d in pairs]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def path_labels(cluster: str, symbol: str, date: str, *,
                horizon: Sequence[str] | str | None = None,
                columns: Sequence[str] | None = None,
                root: Path = PROFIT_CACHE) -> pd.DataFrame:
    _read_contract(Path(root))
    path = partition(cluster, symbol, date, root) / "path_labels.parquet"
    if not path.exists():
        raise CacheError(f"수익 구간이 없다: {cluster}/{symbol}/{date}")
    if columns is None and horizon is not None:
        keys = (horizon,) if isinstance(horizon, str) else tuple(horizon)
        unknown = set(keys) - set(horizons(root))
        if unknown:
            raise KeyError(f"수익 구간에 없는 평가 창: {sorted(unknown)}")
        columns = [*PATH_BASE_COLUMNS,
                   *(f"{field}__{key}" for key in keys for field in PATH_HORIZON_FIELDS)]
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def regions(cluster: str, symbol: str | None = None, date: str | None = None, *,
            horizon: Sequence[str] | str | None = None,
            root: Path = PROFIT_CACHE) -> pd.DataFrame:
    _read_contract(Path(root))
    frame = _read_partitions("regions.parquet", cluster=cluster, symbol=symbol, date=date,
                             root=Path(root))
    if horizon is not None:
        keys = {horizon} if isinstance(horizon, str) else set(horizon)
        frame = frame[frame["horizon_key"].isin(keys)].copy()
    return frame


def quality_regions(cluster: str, symbol: str | None = None, date: str | None = None, *,
                    horizon: Sequence[str] | str | None = None,
                    root: Path = PROFIT_CACHE) -> pd.DataFrame:
    _read_contract(Path(root))
    frame = _read_partitions("quality_regions.parquet", cluster=cluster, symbol=symbol, date=date,
                             root=Path(root))
    if horizon is not None:
        keys = {horizon} if isinstance(horizon, str) else set(horizon)
        frame = frame[frame["horizon_key"].isin(keys)].copy()
    return frame


def quality_clusters(root: Path = PROFIT_CACHE) -> tuple[str, ...]:
    return tuple(sorted({cluster for cluster, _symbol, _date in available(root=root)}))


def _write_contract(root: Path, *, specs: tuple[HorizonSpec, ...],
                    region_contract: dict[str, Any], quality_contract: QualityContract) -> None:
    payload = {
        "schema": CACHE_SCHEMA,
        "created_at": now_utc(),
        "price_opportunity": {
            "entry_price": "BID1",
            "entry_order": "LIMIT_BUY_POST",
            "exit_price": "FUTURE_BEST_ASK1",
            "exit_order": "LIMIT_SELL_PRICE_OPPORTUNITY",
            "formula": "(best_ask_after_entry / entry_bid - 1) * 10000 - fee_bps_rt",
            "oracle": True,
            "execution_proof": False,
        },
        "horizons": [spec.as_dict() for spec in specs],
        "region_contract": dict(region_contract),
        "quality_contract": quality_contract.as_dict(),
        "dates": sorted({date for _cluster, _symbol, date in available(root=root)}),
        "clusters": sorted({cluster for cluster, _symbol, _date in available(root=root)}),
    }
    path = _config_path(root)
    if path.exists():
        previous = read_json(path)
        immutable = ("schema", "price_opportunity", "horizons", "region_contract", "quality_contract")
        if any(previous.get(key) != payload.get(key) for key in immutable):
            raise CacheError("다른 수익구간 계약이 이미 있다. 새 저장소를 지정한다")
        payload["created_at"] = previous.get("created_at", payload["created_at"])
    write_json(path, payload)


def materialize(cluster: str, symbol: str, date: str, *,
                horizons_: Iterable[HorizonSpec] = STANDARD_HORIZONS,
                region_contract: dict[str, Any] | None = None,
                quality_contract: QualityContract = QualityContract(),
                root: Path = PROFIT_CACHE, tick_root: Path = TICK_ROOT) -> dict[str, Any]:
    """한 종목-일의 새 가격 방향 수익구간·품질구간을 저장한다."""
    cluster = _profile_scope(cluster)
    specs = tuple(horizons_)
    region_rule = {**DEFAULT_CONTRACT, **(region_contract or {})}
    arrays, source = tickdata.load(symbol, date, Path(tick_root))
    labels = build_path_labels(arrays, specs)
    raw, raw_summary = extract_regions(labels, specs, **region_rule)
    quality, quality_summary = extract_quality(labels, specs, quality_contract)
    quality = clean_safe(quality.assign(fee_bps_rt=quality_contract.fee_bps_rt))
    for frame in (raw, quality):
        frame.insert(0, "cluster_id", str(cluster))
        frame.insert(1, "symbol", str(symbol).zfill(6))
        frame.insert(2, "date", str(date))
    out = partition(cluster, symbol, date, root)
    out.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out / "path_labels.parquet", index=False)
    raw.to_parquet(out / "regions.parquet", index=False)
    quality.to_parquet(out / "quality_regions.parquet", index=False)
    write_json(out / "manifest.json", {
        "schema": CACHE_SCHEMA,
        "cluster_id": str(cluster), "symbol": str(symbol).zfill(6), "date": str(date),
        "source": {"path": str(source), "bytes": source.stat().st_size,
                   "sha256": sha256_file(source)},
        "ticks": int(len(labels)), "regions": int(len(raw)), "quality_regions": int(len(quality)),
        "region_summary": raw_summary, "quality_summary": quality_summary,
    })
    _write_contract(Path(root), specs=specs, region_contract=region_rule,
                    quality_contract=quality_contract)
    return {"cluster_id": str(cluster), "symbol": str(symbol).zfill(6), "date": str(date),
            "path": str(out), "ticks": int(len(labels)), "regions": int(len(raw)),
            "quality_regions": int(len(quality))}
