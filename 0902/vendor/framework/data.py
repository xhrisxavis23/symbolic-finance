"""틱 원본을 배열로 읽는다. feature 계산은 하지 않는다.

구 `data.py` 는 로딩과 feature 계산을 함께 들고 있어 766줄이었다. 계산은
`catalog.py` 의 각 `Feature.compute` 로 갔고, 여기에는 읽기만 남는다.

원본 parquet 은 한 파일에 호가(`data_type==12`)와 체결(`data_type==11`) 이 섞여
있다. 호가를 시간 순 격자로 삼고 체결을 그 격자에 누적한다.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import BACKTEST_CACHE_ROOT, TICK_ROOT, read_json, write_json


N_LEVELS = 10
QUOTE_ROW = 12
TRADE_ROW = 11
SESSION_START = 90000000000    # 09:00:00.000000
SESSION_END = 153000000000     # 15:30:00.000000
CACHE_SCHEMA = "backtest_tick_array_cache.v1"
_REQUIRED_ARRAY_KEYS = (
    "bid_price", "ask_price", "bid_qty", "ask_qty", "local_time", "time_s",
)

# askbid_type: 1 = 매도 주도, 2 = 매수 주도
SELL_INITIATED = 1
BUY_INITIATED = 2


class InsufficientQuoteData(ValueError):
    """유효 호가가 거의 없다 (거래정지·정리매매). 실패가 아니라 부재다."""


def find_parquet(symbol: str, date: str, root: Path = TICK_ROOT) -> Path:
    """(종목, 날) 의 parquet 경로. 없으면 FileNotFoundError, 둘 이상이면 예외."""
    pattern = str(Path(root) / date / f"{symbol}_*.parquet")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"parquet 없음: symbol={symbol} date={date} ({pattern})")
    if len(matches) > 1:
        raise RuntimeError(f"parquet 이 둘 이상: {symbol}/{date}: {matches}")
    return Path(matches[0])


def _column(table: Any, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def packed_time_to_seconds(values: np.ndarray) -> np.ndarray:
    """HHMMSSuuuuuu 정수를 장 시작부터의 초로."""
    packed = np.asarray(values, dtype=np.int64)
    hours = packed // 10_000_000_000
    minutes = (packed // 100_000_000) % 100
    seconds = (packed // 1_000_000) % 100
    micros = packed % 1_000_000
    return (hours * 3600 + minutes * 60 + seconds).astype(float) + micros.astype(float) / 1.0e6


def load(symbol: str, date: str, root: Path = TICK_ROOT, *,
         cache_root: Path | None = BACKTEST_CACHE_ROOT) -> tuple[dict[str, np.ndarray], Path]:
    """한 종목-일을 배열로. 정규장 구간만.

    돌려주는 키:
      bid_price, ask_price, bid_qty, ask_qty   (n, N_LEVELS)
      time_s, local_time                       (n,)
      buy_volume, sell_volume                  (n,)  매수/매도 주도 체결량
      buy_max_price, sell_min_price            (n,)  큐 판정용 극단가

    체결 행이 하나도 없으면 체결 키를 **넣지 않는다.** 0 으로 채우면
    "매도 압력이 없었다" 는 관측을 지어내게 된다. `catalog.compute_features` 가 그 경우
    체결 의존 feature 를 건너뛴다 (`catalog.feature_availability` 가 이유를 남긴다).

    매도 주도 체결의 최저가와 매수 주도 체결의 최고가를 따로 두는 이유: 지정가 매수의
    큐는 '내 가격에 도달한 매도 체결' 이 소진시킨다. 평균가는 도달 여부를 말해주지
    않는다.
    """
    # 전종목 반복에서는 ``date/`` 폴더를 symbol마다 glob 하는 비용이 커진다. 명시적으로
    # 만든 array cache에는 이미 원본의 절대 경로와 stat identity가 있으므로, 먼저 그 경로로
    # cache를 확인한다. 원본이 바뀌었거나 cache가 없을 때만 기존 glob 탐색으로 되돌아간다.
    if cache_root is not None:
        _arrays_path, metadata_path = _cache_paths(symbol, date, Path(cache_root))
        if metadata_path.is_file():
            try:
                cached_source = Path(str((read_json(metadata_path).get("source") or {}).get("path") or ""))
                if cached_source.is_file():
                    cached = _read_cache(cached_source, symbol, date, cache_root)
                    if cached is not None:
                        return cached, cached_source
            except (OSError, ValueError, TypeError):
                pass
    path = find_parquet(symbol, date, root)
    cached = _read_cache(path, symbol, date, cache_root)
    if cached is not None:
        return cached, path
    return _load_parquet(path, symbol, date)


def materialize_cache(symbol: str, date: str, *, root: Path = TICK_ROOT,
                      cache_root: Path = BACKTEST_CACHE_ROOT) -> dict[str, Any]:
    """정본 Backtest가 읽는 한 종목-일 배열을 명시적으로 cache에 만든다."""
    path = find_parquet(symbol, date, root)
    cached = _read_cache(path, symbol, date, cache_root)
    if cached is not None:
        return {"cache_state": "REUSED", "symbol": str(symbol).zfill(6), "date": str(date),
                "source": _source_identity(path), "array_count": len(cached)}
    arrays, _ = _load_parquet(path, symbol, date)
    _write_cache(path, symbol, date, arrays, cache_root)
    return {"cache_state": "CREATED", "symbol": str(symbol).zfill(6), "date": str(date),
            "source": _source_identity(path), "array_count": len(arrays)}


def _load_parquet(path: Path, symbol: str, date: str) -> tuple[dict[str, np.ndarray], Path]:
    """원본 parquet을 정규장 Backtest 배열로 바꾼다."""
    levels = [c for level in range(1, N_LEVELS + 1)
              for c in (f"bid{level}_price", f"ask{level}_price",
                        f"bid{level}_quantity", f"ask{level}_quantity")]
    table = pq.read_table(path, columns=[
        "data_type", "local_time", "current_price", "trading_volume", "askbid_type", *levels])
    table = table.filter(pc.and_(
        pc.greater_equal(table["local_time"], SESSION_START),
        pc.less(table["local_time"], SESSION_END)))

    quotes = table.filter(pc.equal(table["data_type"], QUOTE_ROW))
    quotes = quotes.take(pc.sort_indices(quotes, sort_keys=[("local_time", "ascending")]))
    bid1 = _column(quotes, "bid1_price", float)
    ask1 = _column(quotes, "ask1_price", float)
    # 스프레드가 음수이거나 호가가 0 인 틱은 체결 가능한 시장이 아니다.
    quotes = quotes.filter(np.isfinite(bid1) & np.isfinite(ask1) & (bid1 > 0) & (ask1 > bid1))
    if len(quotes) < 2:
        raise InsufficientQuoteData(f"유효 호가 2틱 미만: symbol={symbol} date={date}")

    def stack(prefix: str, suffix: str) -> np.ndarray:
        return np.column_stack([_column(quotes, f"{prefix}{level}_{suffix}", float)
                                for level in range(1, N_LEVELS + 1)])

    local_time = _column(quotes, "local_time", np.int64)
    out: dict[str, np.ndarray] = {
        "bid_price": stack("bid", "price"),
        "ask_price": stack("ask", "price"),
        "bid_qty": stack("bid", "quantity"),
        "ask_qty": stack("ask", "quantity"),
        "local_time": local_time,
        "time_s": packed_time_to_seconds(local_time),
    }

    trades = table.filter(pc.equal(table["data_type"], TRADE_ROW))
    if len(trades):
        times = _column(trades, "local_time", np.int64)
        index = np.searchsorted(local_time, times, side="left")
        keep = (index >= 0) & (index < len(local_time))
        index = index[keep]
        volume = _column(trades, "trading_volume", float)[keep]
        price = _column(trades, "current_price", float)[keep]
        side = _column(trades, "askbid_type", int)[keep]

        n = len(local_time)
        buy_volume = np.zeros(n)
        sell_volume = np.zeros(n)
        buy_max = np.full(n, -np.inf)
        sell_min = np.full(n, np.inf)
        is_sell = (side == SELL_INITIATED) & (price > 0.0)
        is_buy = (side == BUY_INITIATED) & (price > 0.0)
        np.add.at(sell_volume, index[is_sell], volume[is_sell])
        np.minimum.at(sell_min, index[is_sell], price[is_sell])
        np.add.at(buy_volume, index[is_buy], volume[is_buy])
        np.maximum.at(buy_max, index[is_buy], price[is_buy])
        out["buy_volume"] = buy_volume
        out["sell_volume"] = sell_volume
        out["buy_max_price"] = np.where(np.isfinite(buy_max), buy_max, np.nan)
        out["sell_min_price"] = np.where(np.isfinite(sell_min), sell_min, np.nan)
    return out, path


def _cache_paths(symbol: str, date: str, root: Path) -> tuple[Path, Path]:
    base = Path(root) / str(date) / str(symbol).zfill(6)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _source_identity(path: Path) -> dict[str, Any]:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def source_identity(path: Path) -> dict[str, Any]:
    """재생 manifest에 남길 원본 파일 identity."""
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "missing": True}
    return _source_identity(path)


def _read_cache(source: Path, symbol: str, date: str,
                cache_root: Path | None) -> dict[str, np.ndarray] | None:
    if cache_root is None:
        return None
    arrays_path, metadata_path = _cache_paths(symbol, date, Path(cache_root))
    if not arrays_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        if (metadata.get("schema") != CACHE_SCHEMA
                or metadata.get("source") != _source_identity(source)):
            return None
        with np.load(arrays_path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        if (set(metadata.get("array_keys") or []) != set(arrays)
                or not all(name in arrays for name in _REQUIRED_ARRAY_KEYS)):
            return None
        return arrays
    except (OSError, ValueError, KeyError):
        return None


def cached_source(symbol: str, date: str, *,
                  cache_root: Path | None = BACKTEST_CACHE_ROOT) -> dict[str, Any] | None:
    """배열을 풀지 않고 현재 원본과 맞는 raw cache인지 확인한다.

    전종목 Feature Profile 생성은 이어서 ``load()``로 배열 전체를 읽는다. 그 전에
    같은 npz를 한 번 더 풀어 raw cache를 검사할 필요는 없으므로, 여기서는 메타데이터와
    원본 파일 identity만 확인한다. npz가 깨진 경우에도 이후 ``load()``가 원본으로
    안전하게 되돌아간다.
    """
    if cache_root is None:
        return None
    arrays_path, metadata_path = _cache_paths(symbol, date, Path(cache_root))
    if not arrays_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        source = Path(str((metadata.get("source") or {}).get("path") or ""))
        if not source.is_file() or metadata.get("schema") != CACHE_SCHEMA:
            return None
        identity = _source_identity(source)
        if metadata.get("source") != identity:
            return None
        keys = set(metadata.get("array_keys") or [])
        if not all(name in keys for name in _REQUIRED_ARRAY_KEYS):
            return None
        return identity
    except (OSError, ValueError, TypeError):
        return None


def _write_cache(source: Path, symbol: str, date: str, arrays: dict[str, np.ndarray],
                 cache_root: Path) -> None:
    arrays_path, metadata_path = _cache_paths(symbol, date, Path(cache_root))
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_name(arrays_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{name: np.asarray(value) for name, value in arrays.items()})
    temporary.replace(arrays_path)
    write_json(metadata_path, {
        "schema": CACHE_SCHEMA,
        "symbol": str(symbol).zfill(6),
        "date": str(date),
        "source": _source_identity(source),
        "array_keys": sorted(arrays),
    })


def available_dates(root: Path = TICK_ROOT) -> list[str]:
    """읽을 수 있는 날짜 목록. 블록 검증이 실재하는 날만 쓰게 한다."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def prior_valid_day(symbol: str, date: str, among: list[str], root: Path = TICK_ROOT) -> str | None:
    """그 종목의 직전 유효 거래일. 없으면 None.

    같은 날로 되돌아가지 않는다 — 적용할 날의 분포로 그 날을 자르면 컷이 아니다.
    파일 존재만 본다. 표본 수 검사는 부르는 쪽 몫이다.
    """
    earlier = sorted(d for d in among if d < date)
    for candidate in reversed(earlier):
        try:
            find_parquet(symbol, candidate, root)
        except FileNotFoundError:
            continue
        return candidate
    return None
