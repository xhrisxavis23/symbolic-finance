#!/usr/bin/env python3
"""M1(3일) — 3일치(20260316~18) 전종목 데이터 적재 시간·메모리·층별 행 수 실측.

0910/scale/measure_load.py 의 방법(순차 로드, 표본을 곱하지 않음)을 그대로
3일로 확장한다. 층 배정은 20260316 하루로 고정한다 (locked_params.py 의
기존 관례를 따른다 — 종목 우주가 날짜마다 크게 안 변한다는 전제, 이 스크립트가
그 전제 자체도 확인한다: 날짜별 종목 목록 교집합/차집합을 기록한다).

0902/ 는 읽기 전용으로 import 만 한다. 큰 배열은 저장하지 않고 JSON 요약만
0911/scale/data/ 에 남긴다.
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

DATES = ["20260316", "20260317", "20260318"]
STRATA_DATE = "20260316"
OUT = Path("/home/dgu/tick/symbolic/0911/scale/data/m1_3day_load.json")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    t_start = time.time()
    log: dict = {"dates": DATES, "strata_date": STRATA_DATE, "per_date": {},
                 "rss_checkpoints": []}

    # ---- 층 배정 (20260316 하루로 고정) ----
    all_symbols = universe.stock_symbols(STRATA_DATE)
    stats = universe.liquidity_stats(all_symbols, STRATA_DATE)
    strata = universe.assign_strata(stats)
    symbols = sorted(strata.index)
    log["n_symbols_universe"] = len(all_symbols)
    log["n_symbols_stratified"] = len(symbols)
    log["stratum_counts"] = strata["stratum"].value_counts().to_dict()
    log["friction_counts"] = strata["friction"].value_counts().to_dict()
    print(f"[strata] universe={len(all_symbols)} stratified={len(symbols)} "
          f"friction={log['friction_counts']}", flush=True)

    # 종목 우주가 날짜마다 같은지 확인 (다른 날짜에서 stock_symbols 만 다시 계산)
    log["symbol_universe_by_date"] = {}
    for date in DATES:
        syms = set(universe.stock_symbols(date))
        overlap = len(syms & set(all_symbols))
        log["symbol_universe_by_date"][date] = {
            "n": len(syms), "overlap_with_strata_date": overlap,
            "missing_from_strata_date": len(set(all_symbols) - syms),
            "new_vs_strata_date": len(syms - set(all_symbols)),
        }
        print(f"[universe-check] {date}: n={len(syms)} overlap={overlap}", flush=True)

    grand_total_rows = 0
    grand_total_maskable = 0
    stratum_row_totals: dict[str, int] = {s: 0 for s in strata["stratum"].unique()}
    stratum_maskable_totals: dict[str, int] = {s: 0 for s in strata["stratum"].unique()}
    friction_row_totals: dict[str, int] = {f: 0 for f in ("L", "M", "H")}
    friction_maskable_totals: dict[str, int] = {f: 0 for f in ("L", "M", "H")}

    for date in DATES:
        date_t0 = time.time()
        per_symbol_info = []
        n_loaded, n_skipped = 0, 0
        date_rows = 0
        date_maskable = 0
        for i, symbol in enumerate(symbols, start=1):
            ts = time.time()
            try:
                arrays = ticks.load_arrays(symbol, date)
            except (FileNotFoundError, ValueError) as error:
                n_skipped += 1
                continue
            t_load = time.time()
            matrix, names = ticks.feature_matrix(arrays)
            t_feat = time.time()
            X_symbol, kept, dim_meta = dimensionless.transform(matrix, names, arrays=arrays)
            t_dim = time.time()
            lbl = labels.build(arrays)
            t_lbl = time.time()

            n_rows = int(len(arrays.get("time_s", [])))
            n_mask = int(lbl.mask.sum())
            date_rows += n_rows
            date_maskable += n_mask
            stratum = str(strata.loc[symbol, "stratum"])
            friction = str(strata.loc[symbol, "friction"])
            stratum_row_totals[stratum] += n_rows
            stratum_maskable_totals[stratum] += n_mask
            friction_row_totals[friction] += n_rows
            friction_maskable_totals[friction] += n_mask
            n_loaded += 1

            per_symbol_info.append({
                "symbol": symbol, "n_rows": n_rows, "n_maskable": n_mask,
                "t_total": t_lbl - ts,
            })

            if i % 500 == 0 or i == len(symbols):
                elapsed = time.time() - date_t0
                log["rss_checkpoints"].append(
                    {"date": date, "n_done": i, "rss_mb": rss_mb(), "t": elapsed})
                print(f"[{date} {i}/{len(symbols)}] elapsed={elapsed:.1f}s "
                      f"rss={rss_mb():.0f}MB rows_so_far={date_rows:,}", flush=True)
                OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

        date_t1 = time.time()
        log["per_date"][date] = {
            "n_loaded": n_loaded, "n_skipped": n_skipped,
            "total_rows": date_rows, "total_maskable": date_maskable,
            "wall_seconds": date_t1 - date_t0,
            "rss_mb_after": rss_mb(),
            # 대표 몇 종목의 세부 타이밍만 남긴다 (전부 남기면 파일이 커진다)
            "sample_symbol_timings": per_symbol_info[:5] + per_symbol_info[-5:],
        }
        grand_total_rows += date_rows
        grand_total_maskable += date_maskable
        print(f"[done {date}] rows={date_rows:,} maskable={date_maskable:,} "
              f"loaded={n_loaded} skipped={n_skipped} wall={date_t1-date_t0:.1f}s "
              f"rss={rss_mb():.0f}MB", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

    log["grand_total_rows_3day"] = grand_total_rows
    log["grand_total_maskable_3day"] = grand_total_maskable
    log["stratum_row_totals_3day"] = stratum_row_totals
    log["stratum_maskable_totals_3day"] = stratum_maskable_totals
    log["friction_row_totals_3day"] = friction_row_totals
    log["friction_maskable_totals_3day"] = friction_maskable_totals
    log["total_wall_seconds"] = time.time() - t_start
    log["rss_final_mb"] = rss_mb()

    print(f"[ALL DONE] 3-day total rows={grand_total_rows:,} maskable={grand_total_maskable:,} "
          f"friction_totals={friction_row_totals} wall={log['total_wall_seconds']:.1f}s "
          f"rss={rss_mb():.0f}MB", flush=True)
    OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
