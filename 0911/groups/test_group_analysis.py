"""group_analysis.py 를 알려진 답이 있는 합성 데이터로 점검한다 — 실측 예측값에 쓰기 전에.

부록 C 의 규칙 수정은 이 점검의 적대적 상황을 전부 통과할 때만 채택한다.

  hetero      큰틱만 종목 안 신호                → 이질성 있음, 최선 큰틱, 본질
  model_only  큰틱만 신호, 릿지는 아무것도 못 배움 → 이질성 있음, 모델 부가가치
  flat        어디에도 신호 없음 (적대적)         → 이질성 있음이 절대 나오면 안 된다
  equal       모든 종목군에 같은 신호 (적대적)     → 이질성 있음이 절대 나오면 안 된다
  confound    예측이 종목 수준 평균만 탄다         → 종목 안 섞기 대조군이 잡아 판정 보류

종목 수준 y 분산(level_sd)을 작게(0.2)·크게(2.0) 둘 다 본다 — 실제 y 가 어느 쪽인지 모르므로.
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


def make(scenario: str, seed: int, level_sd: float):
    rng = np.random.default_rng(seed)
    S, n = 450, 300
    tick = np.array(ga.TICK_ORDER)[np.arange(S) % 3]
    liq = np.array(ga.LIQ_ORDER)[(np.arange(S) // 3) % 3]
    syms = [f"{i:06d}" for i in range(S)]
    parts = []
    for i, s in enumerate(syms):
        x = rng.normal(size=n)
        level = rng.normal() * level_sd
        noise_pred = 0.05 * rng.normal(size=n)
        big = tick[i] == "큰틱"
        if scenario == "hetero":
            beta = 0.6 if big else 0.0
            y = level + beta * x + rng.normal(size=n)
            p_d = p_r = beta * x if beta else noise_pred
        elif scenario == "model_only":
            beta = 0.6 if big else 0.0
            y = level + beta * x + rng.normal(size=n)
            p_d = beta * x if beta else noise_pred
            p_r = noise_pred
        elif scenario == "flat":
            y = level + rng.normal(size=n)
            p_d = p_r = noise_pred
        elif scenario == "equal":
            y = level + 0.3 * x + rng.normal(size=n)
            p_d = p_r = 0.3 * x
        elif scenario == "weak_confound":
            y = level + rng.normal(size=n)
            p_d = p_r = 0.3 * level + noise_pred
        elif scenario == "confound":
            y = level + rng.normal(size=n)
            p_d = p_r = np.full(n, level)
        else:
            raise ValueError(scenario)
        parts.append(pd.DataFrame({"symbol": s, "date": "20260319", "y_path": y,
                                   "pred_deeplob": p_d, "pred_ridge": p_r}))
    one = pd.concat(parts, ignore_index=True)
    df = pd.concat([one.assign(eval_set="eval1"), one.assign(eval_set="eval2")], ignore_index=True)
    return df, pd.DataFrame({"tick_group": tick, "liq_group": liq}, index=syms)


def run(scenario: str, seed: int = 7, level_sd: float = 0.2) -> dict:
    df, assign = make(scenario, seed, level_sd)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        df.to_parquet(d / "p.parquet")
        assign.to_csv(d / "a.csv")
        ga.PRED, ga.ASSIGN, ga.OUT = d / "p.parquet", d / "a.csv", d / "r.json"
        with redirect_stdout(io.StringIO()):
            ga.main()
        return json.loads((d / "r.json").read_text())


failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  통과  " if cond else "  실패  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def brief(r: dict) -> str:
    t = r["sets"]["eval1"]["deeplob"]["tick"]
    return "  ".join(f"{g} {t[g]['r2']:+.4f}[{t[g]['lo']:+.4f},{t[g]['hi']:+.4f}]" for g in ga.TICK_ORDER)


for lsd in (0.2, 2.0):
    r = run("hetero", level_sd=lsd); v = r["verdict_primary"]
    print(f"[hetero level_sd={lsd}] {brief(r)}")
    check(v["precondition_passed"], "음성 대조군 통과")
    check(v["heterogeneity"] == "이질성 있음", f"이질성 있음 (실제: {v['heterogeneity']})")
    check(v.get("best_group") == "큰틱", f"최선 큰틱 (실제: {v.get('best_group')})")
    check(v.get("nature") == "본질", f"본질 (실제: {v.get('nature')})")

r = run("model_only"); v = r["verdict_primary"]
print(f"[model_only] {brief(r)}")
check(v["heterogeneity"] == "이질성 있음", f"이질성 있음 (실제: {v['heterogeneity']})")
check(v.get("nature") == "모델 부가가치", f"릿지는 무신호 → 모델 부가가치 (실제: {v.get('nature')})")

print("[flat — 적대적] 신호 없음에서 거짓 이질성이 나오면 부록 C 수정 기각")
for lsd in (0.2, 2.0):
    for seed in (1, 2, 3):
        v = run("flat", seed=seed, level_sd=lsd)["verdict_primary"]
        check(v["heterogeneity"] != "이질성 있음", f"seed={seed} level_sd={lsd} → {v['heterogeneity']}")
        check(v["precondition_passed"], f"seed={seed} level_sd={lsd} → 대조군 거짓 경보 없음 (부록 E)")

print("[equal — 적대적] 모든 종목군 같은 신호에서 거짓 이질성이 나오면 부록 C 수정 기각")
for seed in (1, 2, 3):
    r = run("equal", seed=seed); v = r["verdict_primary"]
    check(v["heterogeneity"] != "이질성 있음", f"seed={seed} → {v['heterogeneity']}  {brief(r)}")
    check(v["precondition_passed"], f"seed={seed} → 대조군 거짓 경보 없음 (부록 E)")

r = run("confound"); v = r["verdict_primary"]
nc = r["sets"]["eval1"]["deeplob"]["negative_control"]
within_hi = max(e["pooled"]["r2"] for e in nc["within_symbol"])
global_hi = max(e["pooled"]["r2"] for e in nc["global"])
print(f"[confound] 종목안섞기 R² 최대 {within_hi:+.4f}  전체섞기 R² 최대 {global_hi:+.4f}")
check(not v["precondition_passed"], "종목 안 섞기 대조군이 교란을 잡아 판정 보류")
check(within_hi > 0 and global_hi <= 0, "전체 섞기만으로는 못 잡는다 — 종목 안 섞기가 필요한 이유")

print("[weak_confound — 검출력 기록, 합격선 아님] 예측이 종목 수준의 30% 만 탄다")
for lsd in (0.2, 2.0):
    for seed in (1, 2, 3):
        r = run("weak_confound", seed=seed, level_sd=lsd); v = r["verdict_primary"]
        nc = r["sets"]["eval1"]["deeplob"]["negative_control"]["within_symbol"]
        lo = max(e["pooled"]["lo"] for e in nc); pt = max(e["pooled"]["r2"] for e in nc)
        print(f"    level_sd={lsd} seed={seed}: 종목안섞기 R² 최대 {pt:+.4f} (하한 최대 {lo:+.4f}) → "
              f"{'검출' if not v['precondition_passed'] else '놓침'}", flush=True)

print()
if failures:
    print(f"실패 {len(failures)}건:"); [print("  -", f) for f in failures]; sys.exit(1)
print("전부 통과")
