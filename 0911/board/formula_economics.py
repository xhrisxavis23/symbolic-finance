#!/usr/bin/env python3
"""0910 본 실험에서 증류된 진입 수식과 그 경제성 — 보고서 '증류된 진입 수식과 경제성' 절의 원자료.

읽기만 한다: 0910/run/state/{seed_*.json, select.json, curve.json, replay_L|H/ledger.parquet},
0911/groups/data/group_assignment.csv. 쓰는 것: board/formula_economics.json.

주의: 0911 은 수식을 새로 증류하지 않았다. 여기의 수식은 전부 0910 의 하루치 교사(저마찰 837종목, 학습 20만 행)에서 나왔다.

체결 단위 경제성의 구간은 종목 단위 부트스트랩 1000회(rng 시드 0, 층 L → H, 식 이름순, 식마다 gross → net 순서로 뽑는다).
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STATE = "/home/dgu/tick/symbolic/0910/run/state"
ASSIGN = "/home/dgu/tick/symbolic/0911/groups/data/group_assignment.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formula_economics.json")
FEE_BPS = 23.0
FEATS = ["book_imbalance", "signed_aggr_flow_20", "signed_aggr_flow_100", "spread_to_round_trip_cost_ratio",
         "queue_imbalance_best", "microprice_velocity", "spread_bps", "book_imbalance_velocity"]
COHORTS = ["NET_RECOVERY", "EARLY_RECOVERY_LATE_REVERSAL", "COST_INSUFFICIENT", "PERSISTENT_ADVERSE"]

res = {"note": "0910 하루치 교사에서 증류된 수식. 0911 은 새로 증류하지 않았다.", "fee_bps": FEE_BPS}

# --- 1. 후보 수식 (시드별) ---------------------------------------------------------------
sel = json.load(open(f"{STATE}/select.json"))
cands, prefix_of = [], {}
idx = 0
for s in (0, 1, 2):
    d = json.load(open(f"{STATE}/seed_{s}.json"))
    res[f"seed{s}_meta"] = {k: d[k] for k in ("n_candidates", "n_compiled", "n_compile_failed", "n_fit_rows",
                                              "teacher_fit_seconds", "sr_fit_seconds")}
    res[f"seed{s}_meta"]["teacher_early_stop"] = d["teacher_early_stop"]
    res[f"seed{s}_meta"]["operators"] = [d["sr_diagnostics"]["binary_operators"], d["sr_diagnostics"]["unary_operators"]]
    res[f"seed{s}_failures"] = d["compile_failure_reasons"]
    for e in d["compiled_entries"]:
        p = f"e{idx:03d}"; idx += 1
        cands.append({"prefix": p, "seed": s, "expr_str": e["expr_str"], "normal_form": e["normal_form"],
                      "shells": e["shells"], "complexity": e["complexity"], "in_sample_score": e["in_sample_score"],
                      "ast": e["ast"]})
cands.append({"prefix": "e018", "seed": "naive", "expr_str": "book_imbalance", "normal_form": "Symbol('book_imbalance')"})
cands.append({"prefix": "e019", "seed": "null", "expr_str": "book_imbalance_velocity", "normal_form": "Symbol('book_imbalance_velocity')"})
assert all(v["expr_str"] == next(c["expr_str"] for c in cands if c["prefix"] == v["contract_id"].split(":")[0])
           for v in sel["primaries"].values()), "선택 단계 계약 번호와 후보 순서가 맞지 않는다"
by_prefix = {c["prefix"]: c for c in cands}
for r in sel["all_M_summary"]:
    c = by_prefix[r["contract_id"].split(":")[0]]
    q = r["contract_id"].split(":q")[1]
    c.setdefault("M", {})[q] = {k: r[k] for k in ("decisions", "fills", "total_net_bps", "bps_per_decision")} | {
        "cohorts": {k: r[f"cohort_{k}"] for k in COHORTS},
        "adverse_share": r["cohort_PERSISTENT_ADVERSE"] / r["fills"] if r["fills"] else None}
res["candidates"] = cands
res["primaries"] = {k: v["contract_id"] for k, v in sel["primaries"].items()}
res["structural_recovery"] = sel["structural_recovery"]
res["M_replay_seconds"] = sel["replay_seconds"]

# --- 2. 층별 곡선 ------------------------------------------------------------------------
curve = json.load(open(f"{STATE}/curve.json"))
res["curve_L_M_H"] = curve["curve_L_M_H"]
res["H_after_1300"] = curve["H_intraday_after_1300"]
res["seed_verdicts"] = curve["seed_verdicts"]
lab = {r["contract_id"].split(":")[0]: r["label"] for r in curve["H_intraday_after_1300"]}
res["curve_contract_labels"] = lab
man = json.load(open(f"{STATE}/replay_H/ledger_manifest.json"))
res["execution"] = {k: man[k] for k in ("entry_execution", "entry_max_rest_seconds", "entry_max_rest_ticks", "stop_gross_bps",
                                        "trailing_drawdown_gross_bps", "exit_limit_rest_seconds", "horizon_seconds",
                                        "exit_execution")}

# --- 3. 체결 단위 경제성 (분위 0.95 동결 계약) ----------------------------------------------
rng = np.random.default_rng(0)


def boot_mean_by_symbol(f, col, B=1000):
    g = f.groupby("symbol")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    k = len(g)
    ix = rng.integers(0, k, size=(B, k))
    cnt = np.zeros((B, k))
    np.add.at(cnt, (np.repeat(np.arange(B), k), ix.ravel()), 1.0)
    m = (cnt @ s) / (cnt @ n)
    return [float(x) for x in np.percentile(m, [2.5, 97.5])], int(k)


assign = pd.read_csv(ASSIGN, index_col=0)
assign.index = assign.index.astype(str).str.zfill(6)
cols = ["contract_id", "symbol", "status", "net_bps", "gross_bps", "cohort", "exit_reason", "holding_seconds",
        "max_favorable_gross_bps", "max_adverse_gross_bps", "entry_spread_bps"] + FEATS
res["fills"], res["groups"] = {}, {}
for layer in ("L", "H"):
    t = pq.read_table(f"{STATE}/replay_{layer}/ledger.parquet", columns=cols,
                      filters=[("contract_id", "in", [f"{p}:q0.95" for p in lab])]).to_pandas()
    t["label"] = t.contract_id.str.split(":").str[0].map(lab)
    res["fills"][layer] = {}
    for lb, d in t.groupby("label"):
        f = d[d.status == "FILLED"]
        ci_g, k = boot_mean_by_symbol(f, "gross_bps")
        ci_n, _ = boot_mean_by_symbol(f, "net_bps")
        res["fills"][layer][lb] = {
            "decisions": int(len(d)), "fills": int(len(f)), "fill_rate": len(f) / len(d), "symbols_filled": k,
            "gross_mean": float(f.gross_bps.mean()), "gross_ci": ci_g, "net_mean": float(f.net_bps.mean()), "net_ci": ci_n,
            "gross_quantiles": {str(q): float(v) for q, v in zip((10, 25, 50, 75, 90), np.percentile(f.gross_bps, [10, 25, 50, 75, 90]))},
            "share_gross_ge_fee": float((f.gross_bps >= FEE_BPS).mean()), "share_gross_pos": float((f.gross_bps > 0).mean()),
            "share_gross_zero": float((f.gross_bps == 0).mean()), "share_net_pos": float((f.net_bps > 0).mean()),
            "net_total": float(f.net_bps.sum()), "holding_median_s": float(f.holding_seconds.median()),
            "mfe_median": float(f.max_favorable_gross_bps.median()), "mae_median": float(f.max_adverse_gross_bps.median()),
            "entry_spread_median": float(f.entry_spread_bps.median()),
            "cohort_share": {k2: float(v) for k2, v in f.cohort.value_counts(normalize=True).items()},
            "exit_share": {k2: float(v) for k2, v in f.exit_reason.value_counts(normalize=True).items()},
            "feature_median_decision": {k2: float(d[k2].median()) for k2 in FEATS},
            "feature_median_fill": {k2: float(f[k2].median()) for k2 in FEATS}}
    t["symbol"] = t.symbol.astype(str).str.zfill(6)
    t = t.join(assign[["tick_group", "liq_group"]], on="symbol")
    res["groups"][layer] = {}
    for axis in ("tick_group", "liq_group"):
        rows = []
        for (lb, g), d in t.dropna(subset=[axis]).groupby(["label", axis]):
            f = d[d.status == "FILLED"]
            rows.append({"label": lb, "group": g, "symbols": int(d.symbol.nunique()), "decisions": int(len(d)),
                         "fills": int(len(f)), "gross_mean": float(f.gross_bps.mean()), "net_mean": float(f.net_bps.mean()),
                         "share_net_pos": float((f.net_bps > 0).mean()), "net_total": float(f.net_bps.sum())})
        res["groups"][layer][axis] = rows

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)

# --- 요약 출력 ---------------------------------------------------------------------------
print("선택 단계(M) 후보별: 계약 · 시드 · 복잡도 · 결정당 q0.70/0.85/0.95 · q0.95 체결 · q0.95 총순 · q0.95 방향오류비율")
for c in cands:
    m = c["M"]
    print(f"  {c['prefix']} {str(c['seed']):>5s} {str(c.get('complexity','')):>3s} "
          f"{m['0.70']['bps_per_decision']:+.3f}/{m['0.85']['bps_per_decision']:+.3f}/{m['0.95']['bps_per_decision']:+.3f} "
          f"{m['0.95']['fills']:>5d} {m['0.95']['total_net_bps']:>9.0f} {m['0.95']['adverse_share']:.1%}  {c['normal_form'][:95]}")
for layer in ("L", "H"):
    print(f"층 {layer}")
    for lb, v in res["fills"][layer].items():
        print(f"  {lb:6s} 체결 {v['fills']:,} ({v['fill_rate']:.1%}) gross {v['gross_mean']:+.2f} [{v['gross_ci'][0]:+.2f},{v['gross_ci'][1]:+.2f}] "
              f"net {v['net_mean']:+.2f} [{v['net_ci'][0]:+.2f},{v['net_ci'][1]:+.2f}] ≥23 {v['share_gross_ge_fee']:.1%} "
              f"방향오류 {v['cohort_share'].get('PERSISTENT_ADVERSE',0):.1%}")
print("저장:", OUT)
