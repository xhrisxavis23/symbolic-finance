"""종목군 분석 — 저장된 예측값에서 종목군별 R² 와 판정을 낸다 (PREREG-GROUPS.md 를 코드로 옮긴 것).

재학습하지 않는다. `group_run.py` 가 저장한 행별 예측값만 읽으므로 몇 번이고 다시 돌릴 수 있다.

입력
  groups/out/predictions.parquet   열: symbol(str), date, eval_set('eval1'|'eval2'), y_path, pred_deeplob, pred_ridge
  groups/data/group_assignment.csv 열: tick_group, liq_group (색인 = 종목)
출력
  groups/out/group_results.json    전 수치와 판정

사전등록 대응
  §5  행 단위 R², 종목 단위 부트스트랩 1000회
  §6  음성 대조군 = 고정 예측 대 종목 안에서 섞은 y (부록 A3). 섞은 y 에서 R² 가 유의하게 양수(CI 하한 > 0)인 칸이 없어야 판정 (부록 E)
  §7  이질성 판정, 본질 vs 모델 판정
  부록 A1  틱×유동성 9칸 (서술, 판정 아님). 평가 종목 20개 미만 칸은 해석 제외
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/dgu/tick/symbolic/0911/groups")
PRED = ROOT / "out" / "predictions.parquet"
ASSIGN = ROOT / "data" / "group_assignment.csv"
OUT = ROOT / "out" / "group_results.json"

N_BOOT = 1000
N_PERM = 5
SEED = 0
MIN_CELL_SYMBOLS = 20
TICK_ORDER = ["큰틱", "중간틱", "작은틱"]
LIQ_ORDER = ["저유동", "중유동", "고유동"]


# ---------------------------------------------------------------------------
# 종목별 충분통계 — R² 를 종목 단위로 재표집할 때 행을 다시 모을 필요가 없게 한다.
#   R² = 1 − ΣSSE / (Σy² − (Σy)²/Σn)
# ---------------------------------------------------------------------------

def symbol_stats(sym_codes: np.ndarray, y: np.ndarray, pred: np.ndarray, n_sym: int) -> np.ndarray:
    """(n_sym, 4) = [n, Σy, Σy², Σ(y−ŷ)²]"""
    out = np.zeros((n_sym, 4))
    out[:, 0] = np.bincount(sym_codes, minlength=n_sym)
    out[:, 1] = np.bincount(sym_codes, weights=y, minlength=n_sym)
    out[:, 2] = np.bincount(sym_codes, weights=y * y, minlength=n_sym)
    out[:, 3] = np.bincount(sym_codes, weights=(y - pred) ** 2, minlength=n_sym)
    return out


def r2_from_stats(s: np.ndarray) -> float:
    n, sy, syy, sse = s.sum(axis=0) if s.ndim == 2 else s
    sst = syy - sy * sy / n if n > 0 else np.nan
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def boot_ci(stats: np.ndarray, members: np.ndarray, rng: np.random.Generator) -> dict:
    """종목을 복원추출해 R² 분포를 만든다. members = 이 종목군에 속한 종목 번호."""
    s = stats[members]
    s = s[s[:, 0] > 0]
    k = len(s)
    if k < 2:
        return {"r2": r2_from_stats(s) if k else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_symbols": int(k), "n_rows": int(s[:, 0].sum()) if k else 0}
    idx = rng.integers(0, k, size=(N_BOOT, k))
    counts = np.zeros((N_BOOT, k))
    np.add.at(counts, (np.repeat(np.arange(N_BOOT), k), idx.ravel()), 1.0)
    agg = counts @ s                                    # (N_BOOT, 4)
    sst = agg[:, 2] - agg[:, 1] ** 2 / agg[:, 0]
    r2 = 1.0 - agg[:, 3] / sst
    lo, hi = np.nanpercentile(r2, [2.5, 97.5])
    return {"r2": r2_from_stats(s), "lo": float(lo), "hi": float(hi),
            "n_symbols": int(k), "n_rows": int(s[:, 0].sum())}


def overlap(a: dict, b: dict) -> bool:
    return not (a["lo"] > b["hi"] or b["lo"] > a["hi"])


# ---------------------------------------------------------------------------

def analyse_set(df: pd.DataFrame, assign: pd.DataFrame, model: str, rng: np.random.Generator) -> dict:
    sym_cat = pd.Categorical(df.symbol)
    codes = sym_cat.codes.astype(np.int64)
    symbols = np.asarray(sym_cat.categories)
    n_sym = len(symbols)
    y = df.y_path.to_numpy(float)
    pred = df[f"pred_{model}"].to_numpy(float)

    grp = assign.reindex(symbols)
    tick = grp.tick_group.to_numpy()
    liq = grp.liq_group.to_numpy()
    all_members = np.arange(n_sym)

    real = symbol_stats(codes, y, pred, n_sym)
    res: dict = {"pooled": boot_ci(real, all_members, rng), "tick": {}, "liq": {}, "cells": {}}
    for g in TICK_ORDER:
        res["tick"][g] = boot_ci(real, np.flatnonzero(tick == g), rng)
    for g in LIQ_ORDER:
        res["liq"][g] = boot_ci(real, np.flatnonzero(liq == g), rng)
    for tg in TICK_ORDER:
        for lg in LIQ_ORDER:
            cell = boot_ci(real, np.flatnonzero((tick == tg) & (liq == lg)), rng)
            cell["interpretable"] = cell["n_symbols"] >= MIN_CELL_SYMBOLS
            res["cells"][f"{tg}×{lg}"] = cell

    # 음성 대조군 (부록 A3) — 예측은 고정, y 만 섞는다
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    neg = {"within_symbol": [], "global": []}
    for p in range(N_PERM):
        prng = np.random.default_rng(SEED + 1000 + p)
        y_w = y.copy()
        for chunk in np.split(order, bounds):
            y_w[chunk] = y[prng.permutation(chunk)]
        y_g = y[prng.permutation(len(y))]
        for kind, yy in (("within_symbol", y_w), ("global", y_g)):
            st = symbol_stats(codes, yy, pred, n_sym)
            entry = {"pooled": boot_ci(st, all_members, rng),
                     "tick": {g: boot_ci(st, np.flatnonzero(tick == g), rng) for g in TICK_ORDER}}
            neg[kind].append(entry)
    res["negative_control"] = neg
    return res


def verdict(primary: dict, ridge: dict) -> dict:
    """§6 선결 조건 → §7 이질성 판정 → 본질 vs 모델. 전부 사전등록 문장 그대로."""
    violations = []
    for kind in ("within_symbol",):                          # §6 은 종목 안 섞기가 조건이다
        for p, entry in enumerate(primary["negative_control"][kind]):
            for g in ["pooled"] + TICK_ORDER:
                ci = entry["pooled"] if g == "pooled" else entry["tick"][g]
                if ci["lo"] > 0:   # 부록 E: 섞은 y 에서 유의하게 양수일 때만 경보
                    violations.append(f"deeplob {kind} perm{p} {g}: CI 하한 {ci['lo']:.5f} > 0")
        for p, entry in enumerate(ridge["negative_control"][kind]):
            for g in ["pooled"] + TICK_ORDER:
                ci = entry["pooled"] if g == "pooled" else entry["tick"][g]
                if ci["lo"] > 0:   # 부록 E: 섞은 y 에서 유의하게 양수일 때만 경보
                    violations.append(f"ridge {kind} perm{p} {g}: CI 하한 {ci['lo']:.5f} > 0")
    if violations:
        return {"precondition_passed": False, "violations": violations,
                "heterogeneity": "판정 보류 — 음성 대조군 위반", "nature": None}

    t = primary["tick"]
    best = max(TICK_ORDER, key=lambda g: t[g]["r2"])
    worst = min(TICK_ORDER, key=lambda g: t[g]["r2"])
    pooled = primary["pooled"]["r2"]
    # 부록 B: 가장 나쁜 종목군 조건은 "0 보다 유의하게 낫지 않다"(CI 하한 ≤ 0).
    # 부록 C: "가장 좋은 종목군 CI 하한 > 전체 R² 점추정" 은 구조적으로 과엄격하다 — 전체 R² 에는 그 종목군
    #         자신이 섞여 있어, 종목 단위 CI 폭이 종목군-전체 차이보다 넓으면 참 이질성도 떨어뜨린다.
    #         의도("한 종목군은 되고 다른 종목군은 안 된다")를 직접 묻는다.
    het_yes = (not overlap(t[best], t[worst])) and t[best]["lo"] > 0 and t[worst]["lo"] <= 0
    het_no = all(overlap(t[a], t[b]) for i, a in enumerate(TICK_ORDER) for b in TICK_ORDER[i + 1:])
    heterogeneity = "이질성 있음" if het_yes else ("이질성 없음" if het_no else "애매")

    nature = None
    if het_yes:
        rt = ridge["tick"]
        r_best = max(TICK_ORDER, key=lambda g: rt[g]["r2"])
        r_worst = min(TICK_ORDER, key=lambda g: rt[g]["r2"])
        nature = "본질" if (r_best == best and not overlap(rt[r_best], rt[r_worst])) else "모델 부가가치"

    # 부록 A1 — 서술만. 유동성 칸 안에서 큰틱−작은틱 부호가 1차 기울기와 같은가
    sign_primary = np.sign(t["큰틱"]["r2"] - t["작은틱"]["r2"])
    within = {}
    for lg in LIQ_ORDER:
        a, b = primary["cells"][f"큰틱×{lg}"], primary["cells"][f"작은틱×{lg}"]
        usable = a["interpretable"] and b["interpretable"]
        within[lg] = {"diff": a["r2"] - b["r2"], "same_sign": bool(np.sign(a["r2"] - b["r2"]) == sign_primary),
                      "interpretable": bool(usable)}
    n_same = sum(v["same_sign"] for v in within.values() if v["interpretable"])
    n_usable = sum(v["interpretable"] for v in within.values())
    return {"precondition_passed": True, "violations": [],
            "best_group": best, "worst_group": worst, "pooled_r2": pooled,
            "heterogeneity": heterogeneity, "nature": nature,
            "appendix_A1": {"within_liquidity": within, "same_sign_count": int(n_same),
                            "usable_cells": int(n_usable),
                            "statement": ("틱 효과가 유동성을 통제해도 남는다" if n_usable >= 2 and n_same >= 2
                                          else "유동성 통제 후 틱 효과가 유지된다고 말할 수 없다")}}


def main() -> None:
    if not PRED.exists():
        sys.exit(f"예측값 파일이 없다: {PRED}")
    df = pd.read_parquet(PRED)
    df["symbol"] = df.symbol.astype(str).str.zfill(6)
    assign = pd.read_csv(ASSIGN, index_col=0)
    assign.index = assign.index.astype(str).str.zfill(6)

    finite = np.isfinite(df.y_path) & np.isfinite(df.pred_deeplob) & np.isfinite(df.pred_ridge)
    dropped = int((~finite).sum())
    df = df[finite & df.symbol.isin(assign.index)]

    rng = np.random.default_rng(SEED)
    results: dict = {"n_rows_dropped_non_finite": dropped, "sets": {}}
    for eval_set in ("eval1", "eval2"):
        part = df[df.eval_set == eval_set]
        results["sets"][eval_set] = {m: analyse_set(part, assign, m, rng) for m in ("deeplob", "ridge")}
    s1 = results["sets"]["eval1"]
    results["verdict_primary"] = verdict(s1["deeplob"], s1["ridge"])
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    print(f"비유한 제외 {dropped:,}행")
    for eval_set, label in (("eval1", "① 안 본 종목·안 본 날 (1차)"), ("eval2", "② 본 종목·안 본 날")):
        print(f"\n=== {label} ===")
        for m in ("deeplob", "ridge"):
            r = results["sets"][eval_set][m]
            print(f"  [{m}] 전체 R² {r['pooled']['r2']:+.5f} [{r['pooled']['lo']:+.5f}, {r['pooled']['hi']:+.5f}]"
                  f"  종목 {r['pooled']['n_symbols']}")
            for axis, order in (("tick", TICK_ORDER), ("liq", LIQ_ORDER)):
                for g in order:
                    c = r[axis][g]
                    print(f"      {g:5s} {c['r2']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}]  종목 {c['n_symbols']}")
    print("\n=== 판정 (§6·§7, 1차: 평가①·DeepLOB·틱 축) ===")
    print(json.dumps(results["verdict_primary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
