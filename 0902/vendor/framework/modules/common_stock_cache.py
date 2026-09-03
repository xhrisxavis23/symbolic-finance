"""날짜별 KRX 주식 종목군의 raw/Profile/Evidence cache를 만든다."""

from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from .. import catalog, data, execution_profile, featureprofile
from ..config import TICK_ROOT, now_utc, read_json, sha256_file, write_json
from . import evidence_cache
from .profile import ProfileSelection


SCHEMA = "common_stock_cache.v2"
SCOPE = "ALL_COMMON_STOCKS"
STOCK_BATCH_SCOPE = "ALL_STOCKS_ASSET_TYPE_ST"
DIRECTORY = "CommonStockCache"
MANIFEST_NAME = "universe_manifest.json"
PROFILE_INPUT_DIRECTORY = "profile_inputs"


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()


def listing_symbols(path: Path) -> tuple[str, ...]:
    """KRX 상장법인목록의 회사별 6자리 보통주 코드만 읽는다."""
    source = Path(path).read_text(encoding="euc-kr")
    symbols: set[str] = set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", source, re.DOTALL):
        cells = [_text(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)]
        if len(cells) == 10 and re.fullmatch(r"[0-9]{6}", cells[2]):
            symbols.add(cells[2])
    if not symbols:
        raise ValueError("KRX 상장법인목록에서 보통주 코드를 읽지 못했다: " + str(path))
    return tuple(sorted(symbols))


def stock_batch_symbols(date: str, root: Path) -> tuple[str, ...]:
    """그 날짜 종목표에서 주식(ST) 6자리 코드만 읽는다.

    원본 tick 폴더의 ``stock_batch_YYYYMMDD.parquet`` 는 날짜별 상품 분류를 함께
    가진다. ST만 고르면 ETF(EF), ETN(EN) 등은 이 시점에서 제외된다.
    """
    path = Path(root) / str(date) / f"stock_batch_{date}.parquet"
    table = pq.read_table(path, columns=["issue_code", "asset_type"])
    symbols = {
        str(code) for code, asset_type in zip(table["issue_code"].to_pylist(),
                                               table["asset_type"].to_pylist())
        if str(asset_type) == "ST" and re.fullmatch(r"[0-9]{6}", str(code))
    }
    if not symbols:
        raise ValueError("stock_batch에서 6자리 주식(ST) 코드를 읽지 못했다: " + str(path))
    return tuple(sorted(symbols))


def _tick_symbols(date: str, root: Path) -> tuple[set[str], set[str]]:
    counts: dict[str, int] = {}
    for path in (Path(root) / str(date)).glob("*.parquet"):
        match = re.fullmatch(r"([0-9]{6})_.*\.parquet", path.name)
        if match:
            symbol = match.group(1)
            counts[symbol] = counts.get(symbol, 0) + 1
    return ({symbol for symbol, count in counts.items() if count == 1},
            {symbol for symbol, count in counts.items() if count != 1})


def _source(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _listing_for_date(listing: Path, date: str) -> Path:
    listing = Path(listing)
    if listing.is_file():
        return listing
    for suffix in (".xls", ".html", ".htm"):
        path = listing / f"{date}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"날짜별 KRX 상장법인목록이 없다: {listing}/{date}.xls")


def _manifest_path(profile_store: Path, scope: str) -> Path:
    return Path(profile_store) / DIRECTORY / str(scope) / MANIFEST_NAME


def _load_manifest(profile_store: Path, *, scope: str, selection_input: Path,
                   selection_definition: str) -> dict[str, Any]:
    path = _manifest_path(profile_store, scope)
    if path.is_file():
        value = read_json(path)
        if (value.get("schema") == SCHEMA
                and value.get("selection_input") == str(Path(selection_input).resolve())
                and value.get("scope") == scope):
            return value
        if value.get("schema") == SCHEMA:
            raise ValueError("다른 종목군 입력을 같은 cache 범위에 섞을 수 없다: "
                             + str(path.parent))
    return {
        "schema": SCHEMA,
        "scope": scope,
        "selection_definition": selection_definition,
        "selection_input": str(Path(selection_input).resolve()),
        "dates": {},
    }


def _profile_input_path(profile_store: Path, scope: str, date: str) -> Path:
    return Path(profile_store) / DIRECTORY / str(scope) / PROFILE_INPUT_DIRECTORY / f"{date}.json"


def _profile_input(symbols: Sequence[str], sources: Sequence[dict[str, Any]]) -> dict[str, Any]:
    config = execution_profile.ExecutionProfileConfig()
    return {
        "symbols": list(symbols),
        "raw_sources": list(sources),
        "catalog_hash": catalog.catalog_hash(),
        "execution_profile_contract": execution_profile.label_contract(config),
    }


def _profile_cached(profile_store: Path, scope: str, date: str, expected: dict[str, Any]) -> bool:
    path = _profile_input_path(profile_store, scope, date)
    if not path.is_file():
        return False
    try:
        stored = read_json(path)
    except (OSError, ValueError):
        return False
    if stored.get("input") != expected:
        return False
    expected_symbols = {str(symbol).zfill(6) for symbol in expected["symbols"]}
    skipped_symbols = {str(symbol).zfill(6) for symbol in stored.get("skipped_symbols") or []}
    if not skipped_symbols <= expected_symbols:
        return False
    return all((featureprofile.folder(scope, symbol, date, profile_store) / "profile.parquet").is_file()
               for symbol in sorted(expected_symbols - skipped_symbols))


def _raw_cache_complete(record: Mapping[str, Any]) -> bool:
    """기존 raw cache가 전종목 선택을 이미 끝냈는지 본다.

    Profile만 추가할 때는 raw universe가 변하지 않는다. 이 상태를 다시 IN_PROGRESS로
    바꾸면 동시에 시작한 Backtest가 raw cache를 읽을 수 없게 되므로 분리한다.
    """
    symbols = record.get("symbols") or []
    states = record.get("raw_cache") or {}
    return bool(symbols) and sum(int(value or 0) for value in states.values()) == len(symbols)


def _materialize_raw(symbol: str, date: str, root: Path) -> dict[str, Any]:
    try:
        return {"symbol": symbol, "result": data.materialize_cache(symbol, date, root=root)}
    except (OSError, ValueError) as error:
        return {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}


def _profile_raw(symbol: str, date: str, root: Path) -> dict[str, Any]:
    """Profile 계산 전에는 npz를 미리 풀지 않는다.

    캐시가 현재 원본과 맞으면 Profile 계산의 ``data.load()``가 처음으로 배열을 읽는다.
    없거나 오래됐으면 기존 raw materialization 경로로 보완한다.
    """
    source = data.cached_source(symbol, date)
    if source is not None:
        return {"symbol": symbol, "result": {
            "cache_state": "REUSED", "symbol": str(symbol).zfill(6), "date": str(date),
            "source": source,
        }}
    return _materialize_raw(symbol, date, root)


def materialize(listing: Path | None, dates: Sequence[str], *, profile_store: Path,
                root: Path = TICK_ROOT, stock_batch: bool = False,
                raw_only: bool = False, workers: int = 1) -> dict[str, Any]:
    """요청 날짜의 주식 전종목 raw cache와 필요시 Profile/Evidence cache를 준비한다."""
    if stock_batch == (listing is not None):
        raise ValueError("KRX listing 또는 날짜별 stock_batch 중 하나만 지정해야 한다")
    listing = Path(listing) if listing is not None else None
    profile_store, root = Path(profile_store), Path(root)
    scope = STOCK_BATCH_SCOPE if stock_batch else SCOPE
    selection_input = root if stock_batch else listing
    assert selection_input is not None
    selection_definition = (
        "날짜별 stock_batch의 asset_type=ST 이면서 6자리 issue_code (ETF/ETN 제외)"
        if stock_batch else "KRX 상장법인목록의 6자리 회사별 보통주 코드")
    manifest = _load_manifest(profile_store, scope=scope, selection_input=selection_input,
                              selection_definition=selection_definition)
    requested_dates = tuple(sorted({str(value) for value in dates}))
    raw_already_complete = all(_raw_cache_complete((manifest.get("dates") or {}).get(date) or {})
                               for date in requested_dates)
    if raw_only or not raw_already_complete:
        manifest["state"] = "IN_PROGRESS"
    else:
        # Backtest는 raw cache가 COMPLETE일 때만 시작한다. Profile 생성은 raw arrays를
        # 바꾸지 않으므로 별도 상태로 기록한다.
        manifest["state"] = "COMPLETE"
        manifest["profile_state"] = "IN_PROGRESS"
    manifest["updated_at"] = now_utc()
    write_json(_manifest_path(profile_store, scope), manifest)
    prepared_dates: list[str] = []
    date_records: list[dict[str, Any]] = []
    for date in requested_dates:
        if stock_batch:
            selection_path = root / date / f"stock_batch_{date}.parquet"
            selected = set(stock_batch_symbols(date, root))
            selection_source = _source(selection_path)
        else:
            assert listing is not None
            date_listing = _listing_for_date(listing, date)
            selected = set(listing_symbols(date_listing))
            selection_source = _source(date_listing)
        prior = (manifest.get("dates") or {}).get(date) or {}
        if prior.get("selection") is not None and prior.get("selection") != selection_source:
            raise ValueError("같은 날짜의 종목군 입력이 바뀌었다. 새 Profile store를 써야 한다: "
                             + date)
        available, ambiguous = _tick_symbols(date, root)
        symbols = sorted(selected & available)
        failures: list[dict[str, str]] = []
        states = {"CREATED": 0, "REUSED": 0}
        cached_symbols: list[str] = []
        sources: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []
        raw_job = _materialize_raw if raw_only else _profile_raw
        if int(workers) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                futures = [pool.submit(raw_job, symbol, date, root) for symbol in symbols]
                raw_results = [future.result() for future in as_completed(futures)]
        else:
            raw_results = [raw_job(symbol, date, root) for symbol in symbols]
        for item in sorted(raw_results, key=lambda value: str(value["symbol"])):
            symbol = str(item["symbol"])
            if item.get("error") is not None:
                failures.append({"symbol": symbol, "error": str(item["error"])})
                continue
            result = dict(item["result"])
            states[str(result["cache_state"])] = states.get(str(result["cache_state"]), 0) + 1
            cached_symbols.append(symbol)
            sources.append({"symbol": symbol, "source": dict(result.get("source") or {})})
        profile_input = _profile_input(cached_symbols, sources)
        profile_result: dict[str, Any]
        if raw_only:
            profile_result = {"state": "SKIPPED_RAW_ONLY", "written": 0}
        elif _profile_cached(profile_store, scope, date, profile_input):
            profile_result = {"state": "REUSED", "written": len(cached_symbols)}
            prepared_dates.append(date)
        else:
            try:
                profile_result = execution_profile.materialize(scope, cached_symbols, (date,),
                                                                profile_store, root=root)
                write_json(_profile_input_path(profile_store, scope, date), {
                    "input": profile_input,
                    "skipped_symbols": sorted({str(item["symbol"]).zfill(6)
                                               for item in profile_result.get("skipped") or []}),
                })
                prepared_dates.append(date)
            except (OSError, ValueError) as error:
                profile_result = {"state": "ERROR", "error": f"{type(error).__name__}: {error}"}
        record = {
            "selection": selection_source,
            "selected_stock_count": len(selected),
            "symbols": cached_symbols,
            "source_file_symbols": len(available),
            "excluded_non_stock_symbols": len(available - selected),
            **({"excluded_non_common_stock_symbols": len(available - selected)}
               if not stock_batch else {}),
            "ambiguous_source_symbols": sorted(ambiguous),
            "raw_cache": states,
            "raw_cache_failures": failures,
            "profile": profile_result,
        }
        manifest["dates"][date] = record
        manifest["updated_at"] = now_utc()
        write_json(_manifest_path(profile_store, scope), manifest)
        date_records.append({"date": date, **record})
    manifest["updated_at"] = now_utc()
    write_json(_manifest_path(profile_store, scope), manifest)

    evidence: list[dict[str, Any]] = []
    for date in prepared_dates:
        selection = ProfileSelection(
            clusters=(scope,), dates=(date,), store=profile_store,
            profile_kind="execution_aligned")
        evidence.append({"dates": [date], **evidence_cache.materialize(selection)})
    if len(prepared_dates) > 1:
        selection = ProfileSelection(
            clusters=(scope,), dates=tuple(prepared_dates), store=profile_store,
            profile_kind="execution_aligned")
        evidence.append({"dates": list(prepared_dates), **evidence_cache.materialize(selection)})
    manifest["state"] = "COMPLETE"
    if not raw_only:
        manifest["profile_state"] = "COMPLETE"
    manifest["updated_at"] = now_utc()
    write_json(_manifest_path(profile_store, scope), manifest)
    return {"state": ("RAW_STOCK_CACHE_MATERIALIZED" if raw_only
                      else "COMMON_STOCK_CACHE_MATERIALIZED"),
            "scope": scope, "manifest": str(_manifest_path(profile_store, scope)),
            "dates": date_records, "evidence": evidence}
