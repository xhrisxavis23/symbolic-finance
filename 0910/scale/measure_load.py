#!/usr/bin/env python3
"""M1 — 전종목(층화 2,507종목) 데이터 적재 시간·메모리 실측.

run_slice.py 의 per-symbol 루프(ticks.load_arrays -> feature_matrix ->
dimensionless.transform -> labels.build)를 그대로 전종목에 대해 돈다.
0902/ 는 읽기 전용으로 import 만 한다 — 아무 파일도 쓰지 않는다.

산출: JSON (종목별 timing, 누적 시간, RSS 스냅샷).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")

import numpy as np  # noqa: E402
from sd import config, dimensionless, labels, ticks, universe  # noqa: E402

OUT = Path("/home/dgu/tick/symbolic/0910/scale/data/m1_load_full.json")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    t_start = time.time()
    log = {"stage_times": {}, "per_symbol": [], "rss_checkpoints": []}

    t0 = time.time()
    all_symbols = universe.stock_symbols(config.DATE)
    t1 = time.time()
    log["stage_times"]["stock_symbols"] = {"n": len(all_symbols), "seconds": t1 - t0}
    print(f"[stock_symbols] {len(all_symbols)} in {t1-t0:.2f}s", flush=True)

    stats = universe.liquidity_stats(all_symbols, config.DATE)
    t2 = time.time()
    log["stage_times"]["liquidity_stats"] = {"n": len(stats), "seconds": t2 - t1}
    print(f"[liquidity_stats] {len(stats)} in {t2-t1:.2f}s", flush=True)

    strata = universe.assign_strata(stats)
    t3 = time.time()
    log["stage_times"]["assign_strata"] = {"n": len(strata), "seconds": t3 - t2}
    print(f"[assign_strata] {len(strata)} in {t3-t2:.2f}s", flush=True)

    symbols = sorted(strata.index)
    log["n_symbols_to_load"] = len(symbols)
    log["rss_checkpoints"].append({"n_done": 0, "rss_mb": rss_mb(), "t": 0.0})

    per_symbol_arrays: list[tuple[dict, object]] = []
    load_loop_t0 = time.time()
    n_skipped = 0
    for i, symbol in enumerate(symbols, start=1):
        ts = time.time()
        try:
            arrays = ticks.load_arrays(symbol, config.DATE)
        except (FileNotFoundError, ValueError) as error:
            n_skipped += 1
            log["per_symbol"].append({
                "symbol": symbol, "error": f"{type(error).__name__}: {error}"})
            continue
        t_load = time.time()
        matrix, names = ticks.feature_matrix(arrays)
        t_feat = time.time()
        X_symbol, kept, dim_meta = dimensionless.transform(matrix, names, arrays=arrays)
        t_dim = time.time()
        lbl = labels.build(arrays)
        t_lbl = time.time()

        n_rows = int(len(arrays.get("time_s", [])))
        per_symbol_arrays.append(({name: X_symbol[:, j] for j, name in enumerate(kept)}, lbl))

        log["per_symbol"].append({
            "symbol": symbol, "n_rows": n_rows,
            "t_load": t_load - ts, "t_feat": t_feat - t_load,
            "t_dim": t_dim - t_feat, "t_lbl": t_lbl - t_dim,
            "t_total": t_lbl - ts,
        })

        if i % 100 == 0 or i == len(symbols):
            elapsed = time.time() - load_loop_t0
            log["rss_checkpoints"].append({"n_done": i, "rss_mb": rss_mb(), "t": elapsed})
            print(f"[{i}/{len(symbols)}] elapsed={elapsed:.1f}s rss={rss_mb():.0f}MB "
                  f"last={symbol} n_rows={n_rows}", flush=True)
            OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

    load_loop_t1 = time.time()
    log["stage_times"]["load_loop_total"] = {
        "n_loaded": len(per_symbol_arrays), "n_skipped": n_skipped,
        "seconds": load_loop_t1 - load_loop_t0}
    print(f"[load_loop] loaded={len(per_symbol_arrays)} skipped={n_skipped} "
          f"in {load_loop_t1 - load_loop_t0:.1f}s", flush=True)

    # 공통 feature 교집합 + vstack (run_slice.py 와 동일)
    t_common0 = time.time()
    common = set(per_symbol_arrays[0][0])
    for columns, _lbl in per_symbol_arrays[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))
    t_common1 = time.time()

    X = np.vstack([np.column_stack([columns[name] for name in feature_names])
                   for columns, _lbl in per_symbol_arrays])
    t_vstack = time.time()
    y_path = np.concatenate([lbl.y_path for _c, lbl in per_symbol_arrays])
    mask = np.concatenate([lbl.mask for _c, lbl in per_symbol_arrays])
    t_concat = time.time()

    log["stage_times"]["common_features"] = {
        "n_features": len(feature_names), "seconds": t_common1 - t_common0}
    log["stage_times"]["vstack"] = {"seconds": t_vstack - t_common1, "shape": list(X.shape)}
    log["stage_times"]["concat_labels"] = {"seconds": t_concat - t_vstack}
    log["feature_names"] = list(feature_names)
    log["total_rows"] = int(X.shape[0])
    log["total_maskable_rows"] = int(mask.sum())
    log["rss_final_mb"] = rss_mb()
    log["total_wall_seconds"] = time.time() - t_start

    print(f"[done] total_rows={X.shape[0]:,} maskable={int(mask.sum()):,} "
          f"features={len(feature_names)} rss={rss_mb():.0f}MB "
          f"wall={log['total_wall_seconds']:.1f}s", flush=True)

    OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
