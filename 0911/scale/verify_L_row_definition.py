#!/usr/bin/env python3
"""0910 이 보고한 "저마찰층 가용 행 8,151,756"이 하루 전체인지 아침 창
(09:00~12:00, `L_TRAIN_WINDOW`)만인지 확인한다. 0911 측정치(3일 합계에서
역산한 하루평균 약 1,524만행)와 크게 달라서, 정의(하루 전체 vs 학습
시간창)가 다른 것인지 실제 불일치인지 가려야 한다.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
from sd import labels, ticks, universe  # noqa: E402

DATE = "20260316"
T_0900, T_1200 = 32400.0, 43200.0


def main() -> None:
    t0 = time.time()
    strata = universe.assign_strata(
        universe.liquidity_stats(universe.stock_symbols(DATE), DATE))
    L = sorted(strata.index[strata.friction == "L"])
    print(f"L symbols: {len(L)}")

    full_day_mask_sum = 0
    morning_mask_sum = 0
    full_day_rows = 0
    morning_rows = 0
    n_loaded = 0
    for symbol in L:
        try:
            arrays = ticks.load_arrays(symbol, DATE)
        except (FileNotFoundError, ValueError):
            continue
        lbl = labels.build(arrays)
        time_s = np.asarray(arrays["time_s"], dtype=float)
        in_morning = (time_s >= T_0900) & (time_s < T_1200)
        full_day_rows += len(time_s)
        morning_rows += int(in_morning.sum())
        full_day_mask_sum += int(lbl.mask.sum())
        morning_mask_sum += int((lbl.mask & in_morning).sum())
        n_loaded += 1

    print(f"loaded={n_loaded}")
    print(f"FULL DAY  : rows={full_day_rows:,} usable(mask)={full_day_mask_sum:,}")
    print(f"MORNING   : rows={morning_rows:,} usable(mask)={morning_mask_sum:,}")
    print(f"ratio morning/full (rows)   = {morning_rows/full_day_rows:.4f}")
    print(f"ratio morning/full (usable) = {morning_mask_sum/full_day_mask_sum:.4f}")
    print(f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
