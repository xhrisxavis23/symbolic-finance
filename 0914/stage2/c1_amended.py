#!/usr/bin/env python3
"""C1′ — 정정한 재예측 점검 (PREREG-TEACHER-ECON.md 부록 A.3).

1. 정렬: 411종목 전부 저장 행 수 = 유효 행 수, y_path 완전 일치 (y_path 는 정렬 확인에만 쓴다 — 손익은 계산하지 않는다)
2. RDG: 최대 절대 차 ≤ 1e-5
3. T2M·T3D: 유효 행 위 규칙 4개 신호가 0911 저장 예측의 신호와 99.5% 이상 일치
4. 판별력 음성 대조: 교사를 맞바꾸면(새 T2M vs 저장 T3D, 새 T3D vs 저장 T2M) Q70 신호 일치 < 99.5%

    python -m stage2.c1_amended
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import common as C

from framework.contract import prior_tick_quantile_grid  # noqa: E402
from sd import labels, ticks  # noqa: E402

AGREE_MIN = 0.995
RDG_TOL = 1e-5
RULES = ("ABS", "Q70", "Q85", "Q95")
PAIRS = (("T2M", "T2M"), ("T3D", "T3D"), ("T2M", "T3D"), ("T3D", "T2M"))   # (새 점수, 저장 예측)


def mask_of(symbol: str):
    arrays = ticks.load_arrays(symbol, C.DATE)
    lbl = labels.build(arrays)
    return symbol, np.asarray(lbl.mask, bool), np.asarray(lbl.y_path, float), np.asarray(arrays["time_s"], float)


def signals(p: np.ndarray) -> dict[str, np.ndarray]:
    fin = np.isfinite(p)
    out = {"ABS": fin & (p > 0.0)}
    cuts = prior_tick_quantile_grid(p, [(q, "higher") for q in C.QS])
    for q in C.QS:
        c = cuts[(q, "higher")]
        out[f"Q{int(round(q * 100))}"] = fin & np.isfinite(c) & (p > c)
    return out


def main() -> None:
    t0 = time.time()
    symbols = C.eval_symbols()
    a = pd.read_parquet(C.PRED_2M, columns=["symbol", "eval_set", "y_path", "pred_deeplob", "pred_ridge"],
                        filters=[("eval_set", "==", "eval1")])
    b = pd.read_parquet(C.PRED_3D, columns=["symbol", "eval_set", "y_path", "pred_deeplob"],
                        filters=[("eval_set", "==", "eval1")])
    if not np.array_equal(a.y_path.to_numpy(float), b.y_path.to_numpy(float), equal_nan=True):
        raise SystemExit("두 예측 파일의 평가 ① 행이 다르다")
    stored = {"T2M": a.pred_deeplob.to_numpy(float), "RDG": a.pred_ridge.to_numpy(float),
              "T3D": b.pred_deeplob.to_numpy(float)}
    y_stored = a.y_path.to_numpy(float)
    idx_by_symbol = a.groupby(a.symbol.astype(str), sort=False).indices
    del a, b

    agree = {f"{f}|{s}": {r: [0, 0] for r in RULES} for f, s in PAIRS}
    rdg_max, align_fail, rows = 0.0, [], 0
    abs_diff = {m: {"max": 0.0, "gt_1e-3": 0} for m in ("T2M", "T3D")}
    with Pool(12) as pool:
        for sym, mask, y, time_s in pool.imap(mask_of, symbols, chunksize=2):
            idx = idx_by_symbol.get(sym)
            z = {m: np.load(C.score_path(m, sym)) for m in C.MODELS}
            if (idx is None or len(idx) != int(mask.sum())
                    or not np.array_equal(y_stored[idx], y[mask], equal_nan=True)
                    or any(not np.array_equal(z[m]["time_s"], time_s) for m in C.MODELS)):
                align_fail.append(sym)
                continue
            fresh = {m: z[m]["score"][mask] for m in C.MODELS}
            ref = {m: stored[m][idx] for m in C.MODELS}
            rdg_max = max(rdg_max, float(np.max(np.abs(fresh["RDG"] - ref["RDG"]))) if len(idx) else 0.0)
            for m in ("T2M", "T3D"):
                d = np.abs(fresh[m] - ref[m])
                if len(d):
                    abs_diff[m]["max"] = max(abs_diff[m]["max"], float(d.max()))
                    abs_diff[m]["gt_1e-3"] += int((d > 1e-3).sum())
            sig_f = {m: signals(fresh[m]) for m in ("T2M", "T3D")}
            sig_r = {m: signals(ref[m]) for m in ("T2M", "T3D")}
            for f, s in PAIRS:
                for r in RULES:
                    agree[f"{f}|{s}"][r][0] += int((sig_f[f][r] == sig_r[s][r]).sum())
                    agree[f"{f}|{s}"][r][1] += int(len(idx))
            rows += int(len(idx))

    rates = {k: {r: (c[0] / c[1] if c[1] else None) for r, c in v.items()} for k, v in agree.items()}
    cond = {
        "1_alignment": not align_fail and rows == len(y_stored),
        "2_rdg": rdg_max <= RDG_TOL,
        "3_same_teacher_agreement": all(rates[f"{m}|{m}"][r] is not None and rates[f"{m}|{m}"][r] >= AGREE_MIN
                                        for m in ("T2M", "T3D") for r in RULES),
        "4_swapped_teacher_rejected": all(rates[k]["Q70"] is not None and rates[k]["Q70"] < AGREE_MIN
                                          for k in ("T2M|T3D", "T3D|T2M")),
    }
    result = {"schema": "0914_c1_amended.v1", "prereg": "PREREG-TEACHER-ECON.md 부록 A.3", "pass": all(cond.values()),
              "conditions": cond, "agree_min": AGREE_MIN, "rows": rows, "alignment_failures": align_fail,
              "rdg_max_abs_diff": rdg_max, "deeplob_abs_diff": abs_diff, "signal_agreement": rates,
              "seconds": time.time() - t0}
    (C.OUT / "c1_amended.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    C.log("c1'", f"{'통과' if result['pass'] else '실패'} {cond}")
    C.log("c1'", f"일치율 {rates}")
    C.git_commit(["stage2/out/c1_amended.json"], f"run(0914): C1′ {'통과' if result['pass'] else '실패'} — {cond}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
