#!/usr/bin/env python3
"""2단계 — 정본 재생 27계약 + 요약·판정. PREREG-TEACHER-ECON.md §3.

    python -m stage2.run_replay --smoke   # 3종목, 구조만 확인(행 수·오류·결측). 성과는 보지 않는다
    python -m stage2.run_replay           # 평가 종목 411개 본 실행 → out/main/summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import common as C
from . import inject

from framework import canonical  # noqa: E402
from sd import config  # noqa: E402
from sd import select as sd_select  # noqa: E402
from sd.compile import contract_template  # noqa: E402


def build(symbols: list[str]):
    select = json.loads(C.SELECT_0910.read_text())
    contracts, members, table, meta = {}, {}, {}, {}

    def add_q(label: str, template: dict, group: str) -> None:
        for q in C.QS:
            cid = f"{label}:q{q:.2f}"
            contracts[cid], members[cid] = template, list(symbols)
            table.update({f"{cid}:{s}:{C.DATE}": {C.THETA: q} for s in symbols})
            meta[cid] = {"label": label, "group": group, "rule": f"Q{int(round(q * 100))}", "q": q}

    for m in C.MODELS:
        cid = f"{m}:abs"
        contracts[cid], members[cid] = C.abs_template(m), list(symbols)
        meta[cid] = {"label": m, "group": "teacher", "rule": "ABS", "q": None}
        add_q(m, C.q_template(C.MARKERS[m]), "teacher")
    add_q("naive", contract_template(select["naive_primary"]["ast"], C.PARAM), "baseline")
    add_q("null", contract_template(select["null_primary"]["ast"], C.PARAM), "baseline")
    for s in ("0", "1", "2"):
        add_q(f"seed{s}", contract_template(select["primaries"][s]["ast"], C.PARAM), "formula0910")
    if len(contracts) != 27:
        raise SystemExit(f"계약이 27개가 아니다: {len(contracts)}")
    return contracts, members, table, meta


def replay(symbols: list[str], out_dir, workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts, members, table, meta = build(symbols)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    ledger_path = out_dir / "ledger.parquet"
    t0 = time.time()
    manifest = canonical.run_backtest(contracts=contracts, members=members, dates=[C.DATE], output=ledger_path,
                                      parameter_table=table, workers=workers, root=config.TICK_ROOT)
    C.log("replay", f"재생 {len(symbols)}종목 × {len(contracts)}계약 {time.time() - t0:.0f}s")
    if manifest.get("errors"):
        raise RuntimeError("재생 오류 — 원장 일부만 집계하지 않는다:\n  "
                           + "\n  ".join(str(e) for e in manifest["errors"][:20]))
    return ledger_path, manifest, meta


def missing_summary(manifest: dict) -> dict:
    missing = manifest.get("missing") or []
    by_contract = Counter(str(m.get("contract_id")) for m in missing if isinstance(m, dict))
    reasons = Counter(str(m.get("reason"))[:120] for m in missing if isinstance(m, dict))
    return {"n": len(missing), "by_contract": dict(by_contract), "top_reasons": reasons.most_common(8)}


def boot_counts(n: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.empty((C.N_BOOT, n), dtype=np.float64)
    for b in range(C.N_BOOT):
        counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    return counts


def _ratio_reps(counts, num, den):
    N, D = counts @ num, counts @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(D > 0, N / D, np.nan)


def _ci(reps: np.ndarray) -> dict:
    nan = int(np.isnan(reps).sum())
    if nan == len(reps):
        return {"ci95": [None, None], "ci_bonf": [None, None], "nan_reps": nan}
    return {"ci95": [float(x) for x in np.nanpercentile(reps, [2.5, 97.5])],
            "ci_bonf": [float(x) for x in np.nanpercentile(reps, [100 * C.BONF / 2, 100 * (1 - C.BONF / 2)])],
            "nan_reps": nan}


def verdict(rec: dict) -> str:
    if rec["scorable"] < C.MIN_SCORABLE:
        return "거래 부족"
    lo, hi = rec["bps_per_decision_boot"]["ci_bonf"]
    if rec["total_net_bps"] > 0 and lo is not None and lo > 0:
        return "수익"
    if hi is not None and hi < 0:
        return "손실"
    return "판정 불가"


def summarise(ledger_path, manifest: dict, meta: dict, symbols: list[str]) -> dict:
    cols = ["contract_id", "symbol", "status", "net_bps", "gross_bps", "cohort", "holding_seconds"]
    led = pd.read_parquet(ledger_path, columns=cols)
    led["symbol"] = led["symbol"].astype(str)
    base = sd_select.summarise(led) if len(led) else pd.DataFrame()
    S = len(symbols)
    pos = {s: i for i, s in enumerate(symbols)}
    counts = boot_counts(S, np.random.default_rng(C.SEED))

    per_symbol, contracts = {}, {}
    for cid, info in meta.items():
        g = led[led.contract_id == cid]
        filled = g[g.status == "FILLED"]
        vec = {k: np.zeros(S) for k in ("net", "gross", "fills", "scorable", "decisions")}
        for key, series in (("net", filled.groupby("symbol").net_bps.sum()),
                            ("gross", filled.groupby("symbol").gross_bps.sum()),
                            ("fills", filled.groupby("symbol").size()),
                            ("scorable", g[g.status.isin(["FILLED", "UNFILLED"])].groupby("symbol").size()),
                            ("decisions", g.groupby("symbol").size())):
            for s, v in series.items():
                vec[key][pos[s]] = float(v)
        per_symbol[cid] = vec
        row = base.loc[cid].to_dict() if cid in base.index else {
            "decisions": 0, "scorable": 0, "fills": 0, "unfilled": 0, "censored": 0,
            "total_net_bps": 0.0, "bps_per_decision": float("nan"), "positive": False,
            **{f"cohort_{c}": 0 for c in sd_select.COHORTS}}
        rec = {**info, **{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}}
        rec["bps_per_decision_boot"] = _ci(_ratio_reps(counts, vec["net"], vec["scorable"]))
        rec["net_per_fill"] = float(filled.net_bps.mean()) if len(filled) else None
        rec["gross_per_fill"] = float(filled.gross_bps.mean()) if len(filled) else None
        rec["net_per_fill_boot"] = _ci(_ratio_reps(counts, vec["net"], vec["fills"]))
        rec["gross_per_fill_boot"] = _ci(_ratio_reps(counts, vec["gross"], vec["fills"]))
        rec["fill_rate"] = rec["fills"] / rec["scorable"] if rec["scorable"] else None
        rec["holding_seconds_median"] = float(filled.holding_seconds.median()) if len(filled) else None
        rec["symbols_with_decisions"] = int((vec["decisions"] > 0).sum())
        rec["cohort_share"] = ({c: rec[f"cohort_{c}"] / rec["fills"] for c in sd_select.COHORTS}
                               if rec["fills"] else None)
        if info["group"] == "teacher":
            rec["verdict"] = verdict(rec)
        contracts[cid] = rec

    comparisons = {}
    for m in C.MODELS:
        for q in C.QS:
            a = f"{m}:q{q:.2f}"
            ra = _ratio_reps(counts, per_symbol[a]["net"], per_symbol[a]["scorable"])
            for other in ("naive", "null", "seed0", "seed1", "seed2"):
                b = f"{other}:q{q:.2f}"
                rb = _ratio_reps(counts, per_symbol[b]["net"], per_symbol[b]["scorable"])
                pa, pb = contracts[a]["bps_per_decision"], contracts[b]["bps_per_decision"]
                comparisons[f"{a} − {b}"] = {"diff": (pa - pb) if pa == pa and pb == pb else None,
                                            "ci95": _ci(ra - rb)["ci95"]}

    teacher = {cid: r["verdict"] for cid, r in contracts.items() if r["group"] == "teacher"}
    profitable = sorted(cid for cid, v in teacher.items() if v == "수익")
    answer = ("교사 자체는 정본 계약에서 번다 — 해당 계약에 한해서: " + ", ".join(profitable) if profitable
              else "교사 자체도 정본 계약에서 벌지 못한다")
    status_counts = led.status.value_counts().to_dict()
    return {"schema": "0914_stage2_summary.v1", "date": C.DATE, "n_symbols": S, "n_boot": C.N_BOOT,
            "seed": C.SEED, "bonferroni_level": 1 - C.BONF, "min_scorable": C.MIN_SCORABLE,
            "ledger_rows": int(len(led)), "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "missing": missing_summary(manifest), "contracts": contracts, "teacher_verdicts": teacher,
            "answer": answer, "comparisons_same_q": comparisons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    checks = json.loads((C.OUT / "checks.json").read_text())
    if not (C.c1_passed() and checks.get("c3_pass") and checks.get("c4_pass")):
        raise SystemExit("실행 전 점검(C1′·C3·C4)을 통과하지 않았다 — 재생하지 않는다")
    inject.install()
    symbols = C.eval_symbols()

    if args.smoke:
        ledger_path, manifest, meta = replay(symbols[:3], C.OUT / "smoke", workers=3)
        led = pd.read_parquet(ledger_path, columns=["contract_id", "status"])
        structure = {"rows_by_contract": {k: int(v) for k, v in led.contract_id.value_counts().items()},
                     "contracts_without_rows": sorted(set(meta) - set(led.contract_id.unique())),
                     "statuses": sorted(led.status.astype(str).unique().tolist()),
                     "missing": missing_summary(manifest), "errors": len(manifest.get("errors") or []),
                     "wall_seconds": manifest.get("wall_seconds"), "runtime": manifest.get("runtime")}
        (C.OUT / "smoke" / "structure.json").write_text(json.dumps(structure, ensure_ascii=False, indent=1,
                                                                    default=str))
        C.log("smoke", json.dumps({k: structure[k] for k in ("contracts_without_rows", "statuses", "errors")},
                                  ensure_ascii=False))
        C.log("smoke", f"missing {structure['missing']['n']} · 계약별 행 {structure['rows_by_contract']}")
        return

    t0 = time.time()
    ledger_path, manifest, meta = replay(symbols, C.OUT / "main", C.REPLAY_WORKERS)
    summary = summarise(ledger_path, manifest, meta, symbols)
    summary["seconds"] = time.time() - t0
    path = C.OUT / "main" / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    C.log("replay", f"판정: {summary['answer']}")
    for cid, v in summary["teacher_verdicts"].items():
        r = summary["contracts"][cid]
        C.log("replay", f"  {cid}: {v} · scorable {r['scorable']} · total {r['total_net_bps']:.0f} · "
                        f"bps/decision {r['bps_per_decision']:.3f} {r['bps_per_decision_boot']['ci_bonf']}")
    C.git_commit(["stage2/out/main/summary.json", "stage2/out/main/meta.json", "stage2/out/main/ledger_manifest.json"],
                 f"run(0914): 2단계 정본 재생 411종목×27계약 완료 — {summary['answer']}")


if __name__ == "__main__":
    main()
