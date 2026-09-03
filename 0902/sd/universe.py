"""종목 우주와 층화. 계획서 §3 S−1 ① · §3 S2.

층은 이 날짜 이 우주에서 직접 계산한다. 정본 `framework/universe.py` 의
MK01~MK07 을 쓰지 않는 이유는 DESIGN.md D6 에 있다 — 그 명단은 ST+ETF 혼합
우주에서 나왔고, ETF 를 빼면 MK05 가 1종목으로 붕괴한다.
"""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

QUOTE_ROW = 12
SESSION_START = 90000000000        # 09:00:00.000000
SESSION_END = 153000000000         # 15:30:00.000000
STAT_COLUMNS = ("data_type", "local_time", "bid1_price", "ask1_price", "ask1_quantity")
RV_LAG = 20

STRATA = ("L-shallow", "L-deep", "M-shallow", "M-deep", "H-shallow", "H-deep")
FRICTIONS = ("L", "M", "H")


def _date_dir(date: str) -> Path:
    return config.TICK_ROOT / str(date)


def stock_symbols(date: str) -> tuple[str, ...]:
    """선택 사다리 3단계까지. ST ∩ 6자리 숫자 ∩ 틱 parquet 정확히 1개.

    정본 `modules/common_stock_cache.py` 와 같은 규칙이다. 증류가 학습한 종목과
    백테스트가 실행하는 종목이 어긋나면 비교가 성립하지 않는다.
    """
    folder = _date_dir(date)
    batch = pd.read_parquet(folder / f"stock_batch_{date}.parquet",
                            columns=["issue_code", "asset_type"])
    selected = {
        str(code) for code, asset in zip(batch.issue_code, batch.asset_type)
        if str(asset) == "ST" and re.fullmatch(r"[0-9]{6}", str(code))
    }
    counts: dict[str, int] = {}
    for path in folder.glob("*.parquet"):
        match = re.fullmatch(r"([0-9]{6})_.*\.parquet", path.name)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    available = {symbol for symbol, count in counts.items() if count == 1}
    return tuple(sorted(selected & available))


def _one_symbol_stats(symbol: str, date: str) -> dict | None:
    """한 종목의 3축 통계. 유효 호가틱이 모자라면 None — 부재이지 실패가 아니다."""
    matches = glob.glob(str(_date_dir(date) / f"{symbol}_*.parquet"))
    if len(matches) != 1:
        return None
    table = pq.read_table(matches[0], columns=list(STAT_COLUMNS))
    data_type = table["data_type"].combine_chunks().to_numpy()
    local_time = table["local_time"].combine_chunks().to_numpy()
    keep = (data_type == QUOTE_ROW) & (local_time >= SESSION_START) & (local_time <= SESSION_END)
    if int(keep.sum()) < config.MIN_QUOTE_TICKS:
        return None
    bid = table["bid1_price"].combine_chunks().to_numpy()[keep].astype(float)
    ask = table["ask1_price"].combine_chunks().to_numpy()[keep].astype(float)
    ask_qty = table["ask1_quantity"].combine_chunks().to_numpy()[keep].astype(float)
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    bid, ask, ask_qty = bid[valid], ask[valid], ask_qty[valid]
    if len(bid) < config.MIN_QUOTE_TICKS:
        return None
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid * 1e4
    moves = (np.abs(mid[RV_LAG:] / mid[:-RV_LAG] - 1.0) * 1e4
             if len(mid) > RV_LAG else np.zeros(1))
    return {"symbol": symbol, "n_quote": int(len(bid)),
            "spread_bps": float(np.median(spread)),
            "ask1_depth": float(np.median(ask_qty)),
            "rv20_bps": float(np.median(moves))}


def liquidity_stats(symbols: Sequence[str], date: str, workers: int = 32) -> pd.DataFrame:
    """종목별 (스프레드, ASK1 잔량, 20틱 변화) 중앙값. 미래 라벨을 쓰지 않는다."""
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        rows = list(pool.map(lambda s: _one_symbol_stats(s, date), symbols))
    frame = pd.DataFrame([r for r in rows if r is not None])
    return frame.set_index("symbol").sort_index()


def assign_strata(stats: pd.DataFrame) -> pd.DataFrame:
    """마찰 3분위 × 잔량 2분할 = 6층. 경계는 이 표본에서 나온다."""
    out = stats.copy()
    low, high = out.spread_bps.quantile([1 / 3, 2 / 3])
    out["friction"] = np.where(out.spread_bps <= low, "L",
                               np.where(out.spread_bps <= high, "M", "H"))
    out["depth"] = out.groupby("friction").ask1_depth.transform(
        lambda column: np.where(column <= column.median(), "shallow", "deep"))
    out["stratum"] = out.friction + "-" + out.depth
    return out


def slice_symbols(strata: pd.DataFrame, per_stratum: int) -> tuple[str, ...]:
    """층마다 코드 오름차순 앞에서 n 개. 무작위 추출을 쓰지 않는다 — 재현성."""
    picked: list[str] = []
    for name in STRATA:
        members = sorted(strata.index[strata.stratum == name])
        picked.extend(members[: int(per_stratum)])
    return tuple(sorted(picked))
