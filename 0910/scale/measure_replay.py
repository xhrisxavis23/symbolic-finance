#!/usr/bin/env python3
"""M4 — 컴파일 + 정본 백테스트 재생 시간 실측. 종목 수를 몇 단계로 늘려가며 잰다.

가장 무거운 종목(005930, 정규장 유효 호가틱 193,602 — 전종목 최대)을 모든
체크포인트에 강제로 포함시킨다 — 대표성 없는 표본으로 곱해 추정하지 않기
위해서다. 마지막 체크포인트는 전종목(2,570)이다 — 가장 무거운 구간을 실제로
돈다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")

import sympy  # noqa: E402
from sd import config, replay, universe  # noqa: E402
from sd.compile import compile_candidates  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402

OUT = Path("/home/dgu/tick/symbolic/0910/scale/data/m4_replay_scale.json")
HEAVIEST_SYMBOL = "005930"

CHECKPOINTS = [10, 50, 200, 500, 1000, 2570]
WORKERS = 32


def build_entry():
    expr = sympy.Symbol("book_imbalance")
    candidate = Candidate(expr=expr, complexity=1, in_sample_score=0.5,
                          backend="naive", seed=0)
    ok, failed = compile_candidates([candidate])
    if not ok:
        raise SystemExit(f"컴파일 실패: {failed}")
    return ok


def main() -> None:
    log = {"workers": WORKERS, "heaviest_symbol": HEAVIEST_SYMBOL,
           "grid": [0.85], "checkpoints": []}

    all_symbols = list(universe.stock_symbols(config.DATE))
    all_symbols_sorted = sorted(all_symbols)
    print(f"[universe] {len(all_symbols_sorted)} symbols total, "
          f"heaviest included={HEAVIEST_SYMBOL in all_symbols_sorted}", flush=True)

    entries = build_entry()
    print(f"[compile] entries={len(entries)}", flush=True)

    for n in CHECKPOINTS:
        n = min(n, len(all_symbols_sorted))
        symbols = list(all_symbols_sorted[:n])
        if HEAVIEST_SYMBOL not in symbols and HEAVIEST_SYMBOL in all_symbols_sorted:
            symbols[-1] = HEAVIEST_SYMBOL
        symbols = sorted(set(symbols))

        out_dir = Path(f"/tmp/m4_replay_out_{n}")
        t0 = time.time()
        result = replay.run(entries, symbols=symbols, date=config.DATE,
                            output_dir=out_dir, grid=[0.85], workers=WORKERS)
        t1 = time.time()

        import pandas as pd
        ledger = pd.read_parquet(result.ledger_path)
        entry = {
            "n_symbols_requested": n, "n_symbols_actual": len(symbols),
            "heaviest_included": HEAVIEST_SYMBOL in symbols,
            "seconds": t1 - t0, "ledger_rows": int(len(ledger)),
            "attempts": result.attempts,
        }
        log["checkpoints"].append(entry)
        print(f"[replay] n_symbols={len(symbols)} -> {t1-t0:.1f}s "
              f"ledger_rows={len(ledger):,} attempts={result.attempts}", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

        # 산출물 정리 — 디스크 여유가 45GB뿐이다
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
