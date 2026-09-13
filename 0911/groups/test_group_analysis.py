"""group_analysis.py 를 알려진 답이 있는 합성 데이터로 점검한다 — 실측 예측값에 쓰기 전에.

세 상황
  hetero    큰틱만 종목 안 신호가 있다         → 선결 조건 통과, 이질성 있음, 가장 좋은 종목군 큰틱
  flat      어디에도 신호가 없다               → 이질성 있음이 나오면 안 된다
  confound  예측이 종목 수준 평균만 타고 있다   → 종목 안 섞기 대조군이 잡아야 한다(판정 보류)
                                              전체 섞기 대조군은 못 잡는다는 것도 확인한다
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ga", "/home/dgu/tick/symbolic/0911/groups/group_analysis.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)


def make(scenario: str, seed: int = 7):
    rng = np.random.default_rng(seed)
    S, n = 450, 300
    tick = np.array(ga.TICK_ORDER)[np.arange(S) % 3]
    liq = np.array(ga.LIQ_ORDER)[(np.arange(S) // 3) % 3]
    syms = [f"{i:06d}" for i in range(S)]
    parts = []
    for i, s in enumerate(syms):
        x = rng.normal(size=n)
        level = rng.normal() * 2.0                      # 종목마다 다른 y 수준
        noise_pred = 0.05 * rng.normal(size=n)          # 신호 없는 예측 = 작은 잡음
        if scenario == "hetero":
            beta = 0.6 if tick[i] == "큰틱" else 0.0
            y = level + beta * x + rng.normal(size=n)
            pred = beta * x if beta else noise_pred
        elif scenario == "flat":
            y = level + rng.normal(size=n)
            pred = noise_pred
        elif scenario == "confound":
            y = level + rng.normal(size=n)
            pred = np.full(n, level)                    # 종목 평균만 맞힌다 — 행 단위 신호 없음
        else:
            raise ValueError(scenario)
        parts.append(pd.DataFrame({"symbol": s, "date": "20260319", "y_path": y,
                                   "pred_deeplob": pred, "pred_ridge": pred}))
    one = pd.concat(parts, ignore_index=True)
    df = pd.concat([one.assign(eval_set="eval1"), one.assign(eval_set="eval2")], ignore_index=True)
    assign = pd.DataFrame({"tick_group": tick, "liq_group": liq}, index=syms)
    return df, assign


def run(scenario: str) -> dict:
    df, assign = make(scenario)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        df.to_parquet(d / "p.parquet")
        assign.to_csv(d / "a.csv")
        ga.PRED, ga.ASSIGN, ga.OUT = d / "p.parquet", d / "a.csv", d / "r.json"
        with redirect_stdout(io.StringIO()):
            ga.main()
        return json.loads((d / "r.json").read_text())


failures = []


def check(cond: bool, msg: str) -> None:
    print(("  통과  " if cond else "  실패  ") + msg)
    if not cond:
        failures.append(msg)


print("[hetero] 큰틱만 신호")
r = run("hetero")
v = r["verdict_primary"]
t = r["sets"]["eval1"]["deeplob"]["tick"]
print(f"    큰틱 {t['큰틱']['r2']:+.4f} [{t['큰틱']['lo']:+.4f},{t['큰틱']['hi']:+.4f}]  "
      f"작은틱 {t['작은틱']['r2']:+.4f} [{t['작은틱']['lo']:+.4f},{t['작은틱']['hi']:+.4f}]  "
      f"전체 {v.get('pooled_r2', float('nan')):+.4f}")
check(v["precondition_passed"], "음성 대조군 통과")
check(v["heterogeneity"] == "이질성 있음", f"이질성 있음 (실제: {v['heterogeneity']})")
check(v.get("best_group") == "큰틱", f"가장 좋은 종목군 큰틱 (실제: {v.get('best_group')})")
check(v.get("nature") == "본질", f"릿지도 같은 기울기 → 본질 (실제: {v.get('nature')})")

print("[flat] 신호 없음")
v = run("flat")["verdict_primary"]
check(v["precondition_passed"], "음성 대조군 통과")
check(v["heterogeneity"] != "이질성 있음", f"이질성 있음이 나오지 않는다 (실제: {v['heterogeneity']})")

print("[confound] 예측이 종목 수준만 탄다")
r = run("confound")
v = r["verdict_primary"]
nc = r["sets"]["eval1"]["deeplob"]["negative_control"]
within_hi = max(e["pooled"]["r2"] for e in nc["within_symbol"])
global_hi = max(e["pooled"]["r2"] for e in nc["global"])
print(f"    실측 R² {r['sets']['eval1']['deeplob']['pooled']['r2']:+.4f}  "
      f"종목안섞기 R² 최대 {within_hi:+.4f}  전체섞기 R² 최대 {global_hi:+.4f}")
check(not v["precondition_passed"], "종목 안 섞기 대조군이 교란을 잡아 판정을 보류한다")
check(within_hi > 0, "종목 안 섞기에서 R² 가 양수로 드러난다")
check(global_hi <= 0, "전체 섞기만으로는 이 교란을 못 잡는다 (종목 안 섞기가 필요한 이유)")

print()
if failures:
    print(f"실패 {len(failures)}건"); sys.exit(1)
print("전부 통과")
