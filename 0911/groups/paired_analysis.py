"""두 교사를 같은 평가 행에서 비교한다 — 종목 단위 짝지은 부트스트랩 (PREREG-MINIBATCH.md §4).

  ΔR² = R²(비교 대상 B) − R²(기준 A)

두 R² 를 따로 구간 추정해 비교하지 않는다. 같은 행·같은 y 이므로 서로 독립이 아니다
(PREREG-GROUPS.md 부록 C 의 교훈). 같은 부트스트랩 표본에서 두 R² 를 함께 계산해 차이를 잰다.

평가 행이 다르면 짝지을 수 없으므로 **계산하지 않고 거부한다.**

사용: python3 paired_analysis.py --a 기준.parquet --b 비교.parquet --out 결과.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_analysis as ga  # noqa: E402  같은 충분통계·R² 식을 쓴다(두 번째 진실 원천을 만들지 않는다)

N_BOOT = 1000
SEED = 0


def load_aligned(path_a, path_b, eval_set: str = "eval1", col: str = "pred_deeplob"):
    a = pd.read_parquet(path_a)
    b = pd.read_parquet(path_b)
    a = a[a.eval_set == eval_set].reset_index(drop=True)
    b = b[b.eval_set == eval_set].reset_index(drop=True)
    same = (len(a) == len(b)
            and bool((a.symbol.astype(str).values == b.symbol.astype(str).values).all())
            and np.array_equal(a.y_path.to_numpy(float), b.y_path.to_numpy(float), equal_nan=True))
    if not same:
        raise ValueError(f"두 예측 파일의 {eval_set} 평가 행이 같지 않다 — 짝지은 비교를 할 수 없다 "
                         f"(행 수 {len(a)} vs {len(b)})")
    ok = (np.isfinite(a.y_path) & np.isfinite(a[col]) & np.isfinite(b[col])).to_numpy()
    return (a.symbol.astype(str).to_numpy()[ok], a.y_path.to_numpy(float)[ok],
            a[col].to_numpy(float)[ok], b[col].to_numpy(float)[ok], int((~ok).sum()))


def paired_delta(symbols, y, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    cat = pd.Categorical(symbols)
    codes = cat.codes.astype(np.int64)
    S = len(cat.categories)
    sa = ga.symbol_stats(codes, y, pred_a, S)
    sb = ga.symbol_stats(codes, y, pred_b, S)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(n_boot, S))
    counts = np.zeros((n_boot, S))
    np.add.at(counts, (np.repeat(np.arange(n_boot), S), idx.ravel()), 1.0)
    A, B = counts @ sa, counts @ sb
    sst = A[:, 2] - A[:, 1] ** 2 / A[:, 0]          # y 가 같으므로 A·B 공통
    delta = (1.0 - B[:, 3] / sst) - (1.0 - A[:, 3] / sst)
    lo, hi = np.nanpercentile(delta, [2.5, 97.5])
    r2a, r2b = ga.r2_from_stats(sa), ga.r2_from_stats(sb)
    verdict = ("데이터 증가 효과 있음" if lo > 0 else ("오히려 나빠짐" if hi < 0 else "효과 없음"))
    return {"r2_a": r2a, "r2_b": r2b, "delta": r2b - r2a, "delta_lo": float(lo), "delta_hi": float(hi),
            "n_symbols": int(S), "n_rows": int(len(y)), "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 예측(200만 행 전체 배치)")
    ap.add_argument("--b", required=True, help="비교 예측(3일 전체 미니배치)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for eval_set in ("eval1", "eval2"):
        sym, y, pa, pb, dropped = load_aligned(args.a, args.b, eval_set)
        res[eval_set] = paired_delta(sym, y, pa, pb) | {"n_rows_dropped_non_finite": dropped}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
