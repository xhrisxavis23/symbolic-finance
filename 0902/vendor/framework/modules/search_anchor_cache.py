"""전종목 Parameter Search가 함께 쓰는 날짜별 anchor/cut cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import catalog, contract, data as tickdata, validation as V
from ..config import now_utc, read_json, sha256_file, write_json


SCHEMA = "search_anchor_cache.v1"
MANIFEST = "anchor_manifest.json"
ANCHOR_DIRECTORY = "anchors"
ELIGIBILITY_DIRECTORY = "eligibility"


def threshold_column(feature: str, direction: str, quantile: float) -> str:
    """직전 100 tick cut의 안정적인 cache 열 이름."""
    return f"__prior100_cut__{feature}__{str(direction).upper()}__q{float(quantile):.2f}"


def _anchor_path(output: Path, date: str) -> Path:
    return Path(output) / ANCHOR_DIRECTORY / f"{date}.parquet"


def _eligibility_path(output: Path, date: str) -> Path:
    return Path(output) / ELIGIBILITY_DIRECTORY / f"{date}.parquet"


def _requirements(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {str(feature): tuple(sorted({str(direction).upper() for direction in directions}))
            for feature, directions in sorted(value.items())}


def _cut_columns(requirements: Mapping[str, Sequence[str]], grid: Sequence[float]) -> tuple[str, ...]:
    return tuple(threshold_column(feature, direction, q)
                 for feature, directions in _requirements(requirements).items()
                 for direction in directions for q in grid)


def _eligibility_columns(features: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for feature in features for column in (
        f"q_eligible__{feature}", f"observations__{feature}", f"reason__{feature}"))


def _anchor_schema(columns: Sequence[str]) -> pa.Schema:
    fields = []
    for name in columns:
        if name in {"symbol", "date"}:
            kind = pa.string()
        elif name in {"tick", "bucket", "sustained_success"}:
            kind = pa.int64()
        elif name == "future_label_not_used_for_selection":
            kind = pa.bool_()
        else:
            kind = pa.float64()
        fields.append(pa.field(name, kind))
    return pa.schema(fields)


def _eligibility_schema(features: Sequence[str]) -> pa.Schema:
    fields = [pa.field("symbol", pa.string())]
    for feature in features:
        fields.extend((pa.field(f"q_eligible__{feature}", pa.bool_()),
                       pa.field(f"observations__{feature}", pa.int64()),
                       pa.field(f"reason__{feature}", pa.string())))
    return pa.schema(fields)


def _q_eligible(values: np.ndarray) -> bool:
    """원래 calibrate와 같은 기준: 연속된 직전 100개가 한 번이라도 있는가."""
    window = int(contract.ROLLING_QUANTILE_WINDOW)
    if len(values) < window:
        return False
    valid = np.isfinite(values).astype(np.int8)
    return bool(np.any(np.convolve(valid, np.ones(window, dtype=np.int8), mode="valid") == window))


def _thresholds_at_anchors(values: np.ndarray, ticks: np.ndarray,
                           directions: Sequence[str], grid: Sequence[float]) -> dict[str, np.ndarray]:
    """anchor tick에서만 정확한 pandas higher/lower order statistic을 만든다."""
    values = np.asarray(values, dtype=float)
    ticks = np.asarray(ticks, dtype=int)
    window = int(contract.ROLLING_QUANTILE_WINDOW)
    result = {threshold_column("", direction, q): np.full(len(ticks), np.nan)
              for direction in directions for q in grid}
    usable = (ticks >= window) & (ticks < len(values))
    if not usable.any():
        return result
    positions = np.flatnonzero(usable)
    selected = ticks[positions]
    offsets = np.arange(window, 0, -1, dtype=int)
    windows = values[selected[:, None] - offsets[None, :]]
    finite = np.isfinite(windows).all(axis=1)
    if not finite.any():
        return result
    valid_positions = positions[finite]
    windows = windows[finite]
    # pandas interpolation=higher/lower가 고르는 0-base order statistic 위치.
    order_positions = sorted({
        *(int(np.ceil((window - 1) * float(q))) for q in grid),
        *(int(np.floor((window - 1) * (1.0 - float(q)))) for q in grid),
    })
    partitioned = np.partition(windows, kth=order_positions, axis=1)
    for direction in directions:
        lower = str(direction).upper() == "LOWER"
        for q in grid:
            order = (int(np.floor((window - 1) * (1.0 - float(q)))) if lower
                     else int(np.ceil((window - 1) * float(q))))
            result[threshold_column("", direction, q)][valid_positions] = partitioned[:, order]
    return result


def _feature_eligibility(features: Sequence[str], values: Mapping[str, np.ndarray],
                         book: catalog.Book) -> dict[str, Any]:
    availability = {item.feature: item for item in catalog.feature_availability(book.capabilities)}
    row: dict[str, Any] = {}
    for feature in features:
        series = values.get(feature)
        if series is None:
            detail = availability.get(feature)
            row.update({f"q_eligible__{feature}": False,
                        f"observations__{feature}": 0,
                        f"reason__{feature}": (detail.reason if detail is not None else
                                               "feature를 계산할 수 없다")})
            continue
        series = np.asarray(series, dtype=float)
        eligible = _q_eligible(series)
        row.update({f"q_eligible__{feature}": eligible,
                    f"observations__{feature}": int(np.isfinite(series).sum()),
                    f"reason__{feature}": None if eligible else
                    f"직전 {contract.ROLLING_QUANTILE_WINDOW} tick 연속 관측이 없다"})
    return row


def _write_date(date: str, symbols: Sequence[str], *, output: Path,
                features: Sequence[str], requirements: Mapping[str, Sequence[str]],
                grid: Sequence[float], root: Path) -> dict[str, Any]:
    config = V.ValidationConfig(extra_features=tuple(features))
    anchor_columns = tuple(dict.fromkeys(
        V.SELECTION_BASE_COLUMNS + config.feature_names + V.RESPONSE_COLUMNS
        + ("future_label_not_used_for_selection",) + _cut_columns(requirements, grid)))
    anchor_schema = _anchor_schema(anchor_columns)
    eligibility_schema = _eligibility_schema(features)
    anchor_path, eligibility_path = _anchor_path(output, date), _eligibility_path(output, date)
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_tmp = anchor_path.with_suffix(".parquet.partial")
    eligibility_tmp = eligibility_path.with_suffix(".parquet.partial")
    anchor_tmp.unlink(missing_ok=True)
    eligibility_tmp.unlink(missing_ok=True)
    anchor_writer = pq.ParquetWriter(anchor_tmp, anchor_schema, compression="zstd")
    eligibility_writer = pq.ParquetWriter(eligibility_tmp, eligibility_schema, compression="zstd")
    anchors = 0
    usable_symbols = 0
    unavailable: list[dict[str, str]] = []
    try:
        for symbol in symbols:
            try:
                arrays, _ = tickdata.load(symbol, date, root)
            except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
                unavailable.append({"symbol": str(symbol), "error": f"{type(error).__name__}: {error}"})
                continue
            book = catalog.Book(arrays)
            values = catalog.compute_features(config.feature_names, book)
            eligibility = {"symbol": str(symbol), **_feature_eligibility(features, values, book)}
            eligibility_part = pd.DataFrame([eligibility]).reindex(
                columns=[field.name for field in eligibility_schema])
            eligibility_writer.write_table(pa.Table.from_pandas(
                eligibility_part, schema=eligibility_schema, preserve_index=False, safe=False))
            part = V.anchor_frame(symbol, date, config, root, arrays=arrays, values=values)
            if part.empty:
                continue
            ticks = part["tick"].to_numpy(int)
            cached_cuts: dict[str, np.ndarray] = {}
            for feature, directions in _requirements(requirements).items():
                series = values.get(feature)
                if series is None:
                    continue
                cuts = _thresholds_at_anchors(series, ticks, directions, grid)
                for direction in directions:
                    for q in grid:
                        cached_cuts[threshold_column(feature, direction, q)] = cuts[
                            threshold_column("", direction, q)]
            if cached_cuts:
                part = pd.concat((part, pd.DataFrame(cached_cuts, index=part.index)), axis=1)
            part = part.reindex(columns=anchor_columns).copy()
            anchor_writer.write_table(pa.Table.from_pandas(
                part, schema=anchor_schema, preserve_index=False, safe=False))
            anchors += len(part)
            usable_symbols += 1
    finally:
        anchor_writer.close()
        eligibility_writer.close()
    anchor_tmp.replace(anchor_path)
    eligibility_tmp.replace(eligibility_path)
    return {"state": "COMPLETE", "symbols": len(symbols), "usable_symbols": usable_symbols,
            "anchors": anchors, "unavailable": unavailable,
            "anchor_file": str(anchor_path), "eligibility_file": str(eligibility_path)}


def materialize(universe_manifest: Path, dates: Sequence[str], *, output: Path,
                features: Sequence[str], threshold_requirements: Mapping[str, Sequence[str]],
                grid: Sequence[float], root: Path) -> dict[str, Any]:
    """전종목 raw cache에서 Search 공용 표를 날짜별로 한 번만 만든다."""
    output, universe_manifest, root = Path(output), Path(universe_manifest), Path(root)
    source = read_json(universe_manifest)
    requested_dates = tuple(sorted({str(date) for date in dates}))
    features = tuple(sorted({str(feature) for feature in features}))
    requirements = _requirements(threshold_requirements)
    input_signature = {
        "universe_manifest": str(universe_manifest.resolve()),
        "universe_manifest_sha256": sha256_file(universe_manifest),
        "features": list(features),
        "threshold_requirements": {feature: list(directions)
                                   for feature, directions in requirements.items()},
        "grid": [float(q) for q in grid], "window": contract.ROLLING_QUANTILE_WINDOW,
        "anchor_config": {"horizon_seconds": V.ValidationConfig().horizon_seconds,
                          "pre_window_seconds": V.ValidationConfig().pre_window_seconds,
                          "bucket_minutes": V.ValidationConfig().bucket_minutes},
    }
    path = output / MANIFEST
    existing = read_json(path) if path.is_file() else None
    if existing is not None and existing.get("input") != input_signature:
        raise ValueError("다른 Search anchor 입력을 같은 cache에 섞을 수 없다: " + str(output))
    manifest = existing or {"schema": SCHEMA, "input": input_signature, "dates": {}}
    manifest.update({"state": "IN_PROGRESS", "updated_at": now_utc()})
    write_json(path, manifest)
    for date in requested_dates:
        symbols = list((source.get("dates") or {}).get(date, {}).get("symbols") or [])
        if not symbols:
            raise ValueError("전종목 raw cache에 종목이 없는 날짜: " + date)
        prior = (manifest.get("dates") or {}).get(date) or {}
        if (prior.get("state") == "COMPLETE" and _anchor_path(output, date).is_file()
                and _eligibility_path(output, date).is_file()):
            continue
        manifest["dates"][date] = {"state": "IN_PROGRESS", "symbols": len(symbols)}
        manifest["updated_at"] = now_utc()
        write_json(path, manifest)
        manifest["dates"][date] = _write_date(
            date, symbols, output=output, features=features, requirements=requirements,
            grid=grid, root=root)
        manifest["updated_at"] = now_utc()
        write_json(path, manifest)
    manifest.update({"state": "COMPLETE", "updated_at": now_utc()})
    write_json(path, manifest)
    return manifest


def anchors(output: Path, dates: Sequence[str], symbols: Sequence[str], *,
            columns: Sequence[str] | None = None) -> pd.DataFrame:
    """완료된 날짜 파일만 읽어 기존 collect와 같은 DataFrame을 돌려준다."""
    output = Path(output)
    parts = []
    wanted = set(str(symbol) for symbol in symbols)
    selected = None if columns is None else list(dict.fromkeys(("symbol", *columns)))
    for date in dates:
        path = _anchor_path(output, str(date))
        if not path.is_file():
            raise FileNotFoundError("Search anchor cache가 없다: " + str(path))
        part = pd.read_parquet(path, columns=selected)
        parts.append(part.loc[part["symbol"].isin(wanted)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def calibration(output: Path, dates: Sequence[str], symbols: Sequence[str], config: Any) -> pd.DataFrame:
    """원래 rolling calibrate의 eligible/reason만 cache에서 정확히 복원한다."""
    output = Path(output)
    rows = pd.MultiIndex.from_product(
        [[str(date) for date in dates], [str(symbol) for symbol in symbols]],
        names=["date", "symbol"]).to_frame(index=False)[["symbol", "date"]]
    feature = str(config.feature)
    pieces = []
    columns = ["symbol", f"q_eligible__{feature}", f"observations__{feature}",
               f"reason__{feature}"]
    for date in dates:
        path = _eligibility_path(output, str(date))
        if not path.is_file():
            raise FileNotFoundError("Search eligibility cache가 없다: " + str(path))
        part = pd.read_parquet(path, columns=columns)
        part["date"] = str(date)
        pieces.append(part)
    known = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["symbol", "date"])
    result = rows.merge(known, on=["symbol", "date"], how="left")
    result["reference_date"] = None
    result["eligible"] = result[f"q_eligible__{feature}"].fillna(False).astype(bool)
    result["reason"] = result[f"reason__{feature}"].where(
        result[f"reason__{feature}"].notna(), "anchor cache에 해당 종목-일이 없다")
    result.loc[result["eligible"], "reason"] = None
    result["observations"] = result[f"observations__{feature}"].fillna(0).astype(int)
    result["window_ticks"] = int(contract.ROLLING_QUANTILE_WINDOW)
    for q in config.grid:
        result[f"{config.parameter}__q{float(q):.2f}"] = float(q)
    return result
