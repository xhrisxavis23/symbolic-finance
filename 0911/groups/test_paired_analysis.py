"""짝지은 부트스트랩 점검 — 알려진 답이 있는 합성 데이터로 (PREREG-MINIBATCH.md §4).

  같은 예측                         → 효과 없음 (Δ 가 모든 표본에서 정확히 0)
  B 에만 신호 추가                   → 데이터 증가 효과 있음
  B 가 잡음                         → 오히려 나빠짐
  A·B 가 같은 신호, 다른 잡음 (적대적) → 효과 있음이 나오면 안 된다 (시드 3개)
  평가 행이 다름                     → 계산하지 않고 거부
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import paired_analysis as pa  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  통과  " if cond else "  실패  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def data(seed: int, S: int = 400, n: int = 250):
    rng = np.random.default_rng(seed)
    sym = np.repeat([f"{i:06d}" for i in range(S)], n)
    level = np.repeat(rng.normal(size=S) * 0.3, n)
    x = rng.normal(size=S * n)
    y = level + 0.4 * x + rng.normal(size=S * n)
    return rng, sym, x, y


rng, sym, x, y = data(1)
base = 0.2 * x

r = pa.paired_delta(sym, y, base, base.copy())
print(f"[같은 예측] Δ {r['delta']:+.5f} [{r['delta_lo']:+.5f}, {r['delta_hi']:+.5f}]")
check(r["verdict"] == "효과 없음" and r["delta_lo"] == 0 and r["delta_hi"] == 0, f"효과 없음, 구간 정확히 0 ({r['verdict']})")

r = pa.paired_delta(sym, y, base, 0.4 * x)
print(f"[B 에 신호 추가] A {r['r2_a']:+.4f} B {r['r2_b']:+.4f} Δ [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]")
check(r["verdict"] == "데이터 증가 효과 있음", f"효과 있음 ({r['verdict']})")

r = pa.paired_delta(sym, y, base, 0.2 * rng.normal(size=len(y)))
print(f"[B 가 잡음] Δ [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]")
check(r["verdict"] == "오히려 나빠짐", f"나빠짐 ({r['verdict']})")

print("[적대적] A·B 같은 신호 + 서로 다른 작은 잡음")
for seed in (2, 3, 4):
    rng, sym, x, y = data(seed)
    pa_ = 0.4 * x + 0.05 * rng.normal(size=len(y))
    pb_ = 0.4 * x + 0.05 * rng.normal(size=len(y))
    r = pa.paired_delta(sym, y, pa_, pb_)
    check(r["verdict"] != "데이터 증가 효과 있음",
          f"seed={seed} → {r['verdict']} Δ [{r['delta_lo']:+.5f}, {r['delta_hi']:+.5f}]")

print("[평가 행 불일치 거부]")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    df = pd.DataFrame({"symbol": sym, "date": "20260319", "eval_set": "eval1", "y_path": y,
                       "pred_deeplob": base, "pred_ridge": base})
    df.to_parquet(d / "a.parquet")
    df.iloc[1:].to_parquet(d / "b_short.parquet")
    df2 = df.copy(); df2.loc[5, "y_path"] += 1.0
    df2.to_parquet(d / "b_y.parquet")
    for name in ("b_short", "b_y"):
        try:
            pa.load_aligned(d / "a.parquet", d / f"{name}.parquet")
            check(False, f"{name}: 거부해야 하는데 통과시켰다")
        except ValueError:
            check(True, f"{name}: 거부한다")
    s, yy, a_, b_, dropped = pa.load_aligned(d / "a.parquet", d / "a.parquet")
    check(len(yy) == len(y) and dropped == 0, "같은 파일은 받아들인다")

print()
if failures:
    print(f"실패 {len(failures)}건:"); [print("  -", f) for f in failures]; sys.exit(1)
print("전부 통과")
