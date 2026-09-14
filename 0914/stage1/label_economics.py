#!/usr/bin/env python3
"""1단계 — 라벨 기준 경제성 (PREREG-TEACHER-ECON.md §2).

읽기만 한다: 0911/groups/out/predictions.parquet (T2M `pred_deeplob` · RDG `pred_ridge`),
            0911/groups/out/mb_full/predictions.parquet (T3D `pred_deeplob`).
쓰는 것:   stage1/label_economics.json

신호가 켜진 유효 행의 실제 y_path(30초 즉시 매수·매도의 마찰 정규화 순수익)를 본다.
분위 규칙은 정본과 같은 정의(framework.contract.prior_tick_quantile_grid — 직전 100행, higher 보간,
창에 결측이 있으면 cut 없음)를 **유효 행 위에서** 계산한다. 정본은 모든 틱을 창에 넣는다(사전등록 §2.1).

    cd /home/dgu/tick/symbolic/0914 && python3 stage1/label_economics.py
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
from sd import config  # noqa: E402

config.load_framework()
from framework.contract import prior_tick_quantile, prior_tick_quantile_grid  # noqa: E402

ROOT = Path("/home/dgu/tick/symbolic/0914")
P_2M = Path("/home/dgu/tick/symbolic/0911/groups/out/predictions.parquet")
P_3D = Path("/home/dgu/tick/symbolic/0911/groups/out/mb_full/predictions.parquet")
OUT = ROOT / "stage1" / "label_economics.json"

DATE = "20260319"
MODELS = ("T2M", "T3D", "RDG")
QS = (0.70, 0.85, 0.95)
RULES = ("ABS",) + tuple(f"Q{int(round(q * 100))}" for q in QS)
N_BOOT = 10_000
SEED = 0
N_TESTS = len(MODELS) * len(RULES)          # 12 — 사전등록 §1.3
BONF = 0.05 / N_TESTS
MIN_ROWS, MIN_SYMBOLS = 1_000, 20            # 사전등록 §2.3 신호 부족 기준
TOP_VAR_SYMBOL = "261780"                    # 0911 평가 ① 변동 1위 종목 (서술용 제외 분석)
WORKERS = 16

_Y: np.ndarray | None = None
_P: dict[str, np.ndarray] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load() -> pd.DataFrame:
    a = pd.read_parquet(P_2M)
    b = pd.read_parquet(P_3D, columns=["symbol", "date", "eval_set", "y_path", "pred_deeplob"])
    same = (len(a) == len(b)
            and bool((a.symbol.astype(str).values == b.symbol.astype(str).values).all())
            and bool((a.eval_set.values == b.eval_set.values).all())
            and np.array_equal(a.y_path.to_numpy(float), b.y_path.to_numpy(float), equal_nan=True))
    if not same:
        raise SystemExit("두 예측 파일의 행이 같지 않다 — 규칙을 짝지어 비교할 수 없어 멈춘다")
    dates = set(a.date.astype(str).unique()) | set(b.date.astype(str).unique())
    if dates != {DATE}:
        raise SystemExit(f"예측 파일의 날짜가 {DATE} 하나가 아니다: {dates}")
    return pd.DataFrame({"eval_set": a.eval_set.values, "symbol": a.symbol.astype(str).values,
                         "y": a.y_path.to_numpy(float), "T2M": a.pred_deeplob.to_numpy(float),
                         "RDG": a.pred_ridge.to_numpy(float), "T3D": b.pred_deeplob.to_numpy(float)})


def _signals(p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    fin = np.isfinite(p) & np.isfinite(y)
    out = {"ALL": fin, "ABS": fin & (p > 0.0)}
    cuts = prior_tick_quantile_grid(p, [(q, "higher") for q in QS])
    for q in QS:
        c = cuts[(q, "higher")]
        out[f"Q{int(round(q * 100))}"] = fin & np.isfinite(c) & (p > c)
    return out


def symbol_stats(idx: np.ndarray) -> dict[tuple[str, str], tuple[int, float, int, float]]:
    """한 종목의 (모델, 규칙)별 (신호 행 수, y 합, y>0 수, 예측 합)."""
    y = _Y[idx]
    out = {}
    for m in MODELS:
        p = _P[m][idx]
        for r, s in _signals(p, y).items():
            ys = y[s]
            out[(m, r)] = (int(s.sum()), float(ys.sum()), int((ys > 0).sum()), float(p[s].sum()))
    return out


def self_check(groups: dict, n_symbols: int = 3) -> dict:
    """grid 분위(한 번 정렬)가 정본 단일 함수(pandas rolling)와 같은가 — 몇 종목에서 대조."""
    worst = 0.0
    checked = 0
    for (_es, _sym), idx in list(groups.items())[:n_symbols]:
        for m in MODELS:
            p = _P[m][idx]
            grid = prior_tick_quantile_grid(p, [(q, "higher") for q in QS])
            for q in QS:
                single = prior_tick_quantile(p, q, method="higher")
                a, b = grid[(q, "higher")], single
                if not np.array_equal(np.isnan(a), np.isnan(b)):
                    raise SystemExit(f"분위 grid 와 단일 함수의 결측 위치가 다르다 ({m}, q={q})")
                both = np.isfinite(a)
                if both.any():
                    worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
                checked += 1
    if worst != 0.0:
        raise SystemExit(f"분위 grid 와 단일 함수 값이 다르다: 최대 차 {worst}")
    return {"checked_series": checked, "max_abs_diff": worst}


def boot_counts(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.empty((N_BOOT, n_symbols), dtype=np.float64)
    for b in range(N_BOOT):
        counts[b] = np.bincount(rng.integers(0, n_symbols, size=n_symbols), minlength=n_symbols)
    return counts


def ratio_ci(counts: np.ndarray, num: np.ndarray, den: np.ndarray) -> dict:
    N, D = counts @ num, counts @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(D > 0, N / D, np.nan)
    nan_reps = int(np.isnan(r).sum())
    if nan_reps == len(r):
        return {"ci95": [None, None], "ci_bonf": [None, None], "nan_reps": nan_reps}
    return {"ci95": [float(x) for x in np.nanpercentile(r, [2.5, 97.5])],
            "ci_bonf": [float(x) for x in np.nanpercentile(r, [100 * BONF / 2, 100 * (1 - BONF / 2)])],
            "nan_reps": nan_reps}


def table(stats_by_symbol: list[dict], symbols: list[str], counts: np.ndarray, total_rows: dict) -> dict:
    out = {}
    for m in MODELS:
        out[m] = {}
        for r in ("ALL",) + RULES:
            arr = np.array([s[(m, r)] for s in stats_by_symbol], dtype=np.float64)   # (S, 4)
            n, sy, npos, sp = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            N = float(n.sum())
            rec = {"n_rows": int(N), "n_symbols": int((n > 0).sum()),
                   "share_of_rows": N / total_rows[m] if total_rows[m] else None,
                   "mean_y": float(sy.sum() / N) if N else None,
                   "pos_share": float(npos.sum() / N) if N else None,
                   "mean_pred": float(sp.sum() / N) if N else None,
                   "mean_y_boot": ratio_ci(counts, sy, n),
                   "pos_share_boot": ratio_ci(counts, npos, n)}
            out[m][r] = rec
    return out


def verdict(rec: dict) -> str:
    if rec["n_rows"] < MIN_ROWS or rec["n_symbols"] < MIN_SYMBOLS:
        return "신호 부족"
    lo, hi = rec["mean_y_boot"]["ci_bonf"]
    if lo is not None and lo > 0:
        return "라벨 기준 이익"
    if hi is not None and hi < 0:
        return "라벨 기준 손실"
    return "판정 불가"


def calibration(df: pd.DataFrame, exclude: str | None = None) -> dict:
    d = df if exclude is None else df[df.symbol != exclude]
    out = {}
    for m in MODELS:
        p, y = d[m].to_numpy(float), d["y"].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        edges = np.quantile(p, np.linspace(0, 1, 11))
        bins = []
        for k in range(10):
            s = (p >= edges[k]) & ((p < edges[k + 1]) if k < 9 else (p <= edges[k + 1]))
            bins.append({"bin": f"D{k + 1}", "lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(s.sum()),
                         "mean_pred": float(p[s].mean()), "mean_y": float(y[s].mean()),
                         "pos_share": float((y[s] > 0).mean())})
        for label, qq in (("top1%", 0.99), ("top0.1%", 0.999)):
            cut = float(np.quantile(p, qq))
            s = p >= cut
            bins.append({"bin": label, "lo": cut, "hi": float(p.max()), "n": int(s.sum()),
                         "mean_pred": float(p[s].mean()), "mean_y": float(y[s].mean()),
                         "pos_share": float((y[s] > 0).mean())})
        out[m] = {"n": int(ok.sum()), "pred_quantiles": {str(q): float(np.quantile(p, q))
                                                          for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                  "bins": bins}
    return out


def main() -> None:
    global _Y, _P
    t0 = time.time()
    df = load()
    log(f"적재 {len(df):,}행 (평가 ① {int((df.eval_set == 'eval1').sum()):,} · 평가 ② {int((df.eval_set == 'eval2').sum()):,})")
    _Y = df["y"].to_numpy(float)
    _P = {m: df[m].to_numpy(float) for m in MODELS}
    groups = df.groupby(["eval_set", "symbol"], sort=True).indices
    for key, idx in groups.items():
        if len(idx) > 1 and np.any(np.diff(idx) != 1):
            raise SystemExit(f"{key} 의 행이 연속하지 않는다 — 틱 순서를 보장할 수 없어 멈춘다")
    check = self_check(groups)
    log(f"분위 자기 점검 통과 {check}")

    keys = list(groups.keys())
    with Pool(WORKERS) as pool:
        stats = pool.map(symbol_stats, [groups[k] for k in keys], chunksize=8)
    log(f"종목별 신호 집계 {len(keys)} 종목 ({time.time() - t0:.0f}s)")

    result = {"schema": "0914_stage1_label_economics.v1", "date": DATE, "prereg": "PREREG-TEACHER-ECON.md §2",
              "n_boot": N_BOOT, "seed": SEED, "bonferroni_tests": N_TESTS, "bonferroni_level": 1 - BONF,
              "min_rows": MIN_ROWS, "min_symbols": MIN_SYMBOLS, "quantile_self_check": check, "sets": {}}
    rng = np.random.default_rng(SEED)
    for es in ("eval1", "eval2"):
        idx_sets = [i for i, k in enumerate(keys) if k[0] == es]
        syms = [keys[i][1] for i in idx_sets]
        st = [stats[i] for i in idx_sets]
        total = {m: sum(s[(m, "ALL")][0] for s in st) for m in MODELS}
        counts = boot_counts(len(syms), rng)
        tab = table(st, syms, counts, total)
        rec = {"n_symbols": len(syms), "total_rows": total, "rules": tab}
        if es == "eval1":
            rec["verdicts"] = {m: {r: verdict(tab[m][r]) for r in RULES} for m in MODELS}
            if TOP_VAR_SYMBOL in syms:
                keep = [i for i, s in enumerate(syms) if s != TOP_VAR_SYMBOL]
                st2 = [st[i] for i in keep]
                total2 = {m: sum(s[(m, "ALL")][0] for s in st2) for m in MODELS}
                counts2 = boot_counts(len(keep), rng)
                rec["rules_excluding_" + TOP_VAR_SYMBOL] = table(st2, [syms[i] for i in keep], counts2, total2)
        rec["calibration"] = calibration(df[df.eval_set == es])
        if es == "eval1":
            rec["calibration_excluding_" + TOP_VAR_SYMBOL] = calibration(df[df.eval_set == es], exclude=TOP_VAR_SYMBOL)
        result["sets"][es] = rec
        log(f"{es} 완료 ({time.time() - t0:.0f}s)")

    result["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log(f"저장 {OUT}")
    for m in MODELS:
        log(f"  {m}: " + " · ".join(f"{r} {result['sets']['eval1']['verdicts'][m][r]}" for r in RULES))


if __name__ == "__main__":
    main()
