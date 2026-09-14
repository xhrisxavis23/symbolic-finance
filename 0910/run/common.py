"""E1 실행 공용 헬퍼. `0902/`(sd 패키지)를 읽기 전용으로 import 한다.

이 파일은 `0910/scale/measure_*.py` 가 이미 검증한 로드 경로
(ticks.load_arrays → feature_matrix → dimensionless.transform → labels.build)
를 그대로 재사용하고, 시간창 필터와 3층(L/M/H) 조립만 더한다.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sd import config, dimensionless, labels, manifold, ticks, universe  # noqa: E402

from . import locked_params as P  # noqa: E402

RUN_ROOT = Path("/home/dgu/tick/symbolic/0910/run")
STATE_DIR = RUN_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def git_commit(paths: Sequence[str], message: str) -> None:
    """스테이지 산출물을 커밋한다 — 세션이 끊겨도 진행 상황이 저장소에 남는다.

    실패해도(예: 변경 없음) 예외를 던지지 않는다 — 실행 자체를 막을 이유가
    아니다. stdout/stderr 는 로그에 남긴다.
    """
    repo = "/home/dgu/tick/symbolic/0910"
    full_message = message + "\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
    try:
        subprocess.run(["git", "add", *paths], cwd=repo, check=True,
                       capture_output=True, text=True)
        result = subprocess.run(
            ["git", "commit", "-m", full_message,
             "--author=E1 실행 파이프라인 <noreply@anthropic.com>"],
            cwd=repo, capture_output=True, text=True)
        log("git", f"commit rc={result.returncode} {result.stdout.strip()[:200]} "
                    f"{result.stderr.strip()[:200]}")
    except Exception as error:  # noqa: BLE001 - 커밋 실패로 실행을 막지 않는다
        log("git", f"commit 실패(무시하고 계속): {type(error).__name__}: {error}")


def stratified_universe() -> pd.DataFrame:
    """2,507종목의 층 배정과 통계. `sd.universe` 를 그대로 쓴다."""
    all_symbols = universe.stock_symbols(P.DATE)
    stats = universe.liquidity_stats(all_symbols, P.DATE)
    strata = universe.assign_strata(stats)
    return strata


def friction_group_symbols(strata: pd.DataFrame) -> dict[str, list[str]]:
    """마찰 3분위(L/M/H)로 묶은 종목 명단. `strata['friction']` 은
    `sd.universe.assign_strata` 가 이미 L/M/H 로 이름 붙여 둔 열이다."""
    groups: dict[str, list[str]] = {}
    for friction in ("L", "M", "H"):
        groups[friction] = sorted(strata.index[strata.friction == friction])
    return groups


def load_windowed(symbols: Sequence[str], time_window: tuple[float, float] | None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             tuple[str, ...], np.ndarray]:
    """종목 목록을 로드해 `(X, y_path, y_fill, mask, feature_names, session_ids)`.

    `time_window`(lo, hi) 를 주면 `time_s` 가 그 구간(`[lo, hi)`) 인 행만
    쓴다 — feature·라벨은 **하루 전체 연속 배열**에서 먼저 계산한 뒤
    시간창으로 자른다(롤링 계산이 세션 경계를 넘지 않게). `None` 이면
    하루 전체(09:00~15:30)를 쓴다.
    """
    per_symbol: list[tuple[dict, object, np.ndarray]] = []
    n_loaded, n_skipped = 0, 0
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, P.DATE)
        except (FileNotFoundError, ValueError) as error:
            n_skipped += 1
            continue
        matrix, names = ticks.feature_matrix(arrays)
        X_symbol, kept, _meta = dimensionless.transform(matrix, names, arrays=arrays)
        lbl = labels.build(arrays)
        time_s = np.asarray(arrays["time_s"], dtype=float)
        if time_window is not None:
            lo, hi = time_window
            keep = (time_s >= lo) & (time_s < hi)
        else:
            keep = np.ones_like(time_s, dtype=bool)
        columns = {name: X_symbol[keep, j] for j, name in enumerate(kept)}
        per_symbol.append((columns, lbl, keep))
        n_loaded += 1

    if not per_symbol:
        raise SystemExit("적재된 종목이 없다")

    common = set(per_symbol[0][0])
    for columns, _l, _k in per_symbol[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))

    X = np.vstack([np.column_stack([c[name] for name in feature_names])
                   for c, _l, _k in per_symbol])
    y_path = np.concatenate([lbl.y_path[k] for _c, lbl, k in per_symbol])
    y_fill = np.concatenate([lbl.y_fill[k] for _c, lbl, k in per_symbol])
    mask = np.concatenate([lbl.mask[k] for _c, lbl, k in per_symbol])
    session_ids = np.concatenate([
        np.full(int(k.sum()), i, dtype=int) for i, (_c, _l, k) in enumerate(per_symbol)])

    log("load", f"symbols loaded={n_loaded} skipped={n_skipped} rows={len(X):,} "
                f"features={len(feature_names)} window={time_window}")
    return X, y_path, y_fill, mask, feature_names, session_ids
