#!/usr/bin/env python3
"""Stage C — L·M·H 저하 곡선(R37-2) + 최종 판정.

L(837, 학습에 쓴 층 — 저하 곡선의 기준점이지 성과 근거가 아니다)·
H(836, 봉인 최종 평가)에서 동결된 후보들을 재생한다. M 은 stage_select 가
이미 쟀다. H 원장은 `minute_of_session` 으로 잘라 13:00 이후(일중 외삽,
H2②) 도 함께 본다.

**이 스테이지는 `20260317`(날짜 축 봉인 홀드아웃)을 열지 않는다.** 그것은
DESIGN.md §5 의 별도 절차이고 이 스테이지의 판정이 그 조건을 만족하는지만
확인한다.

    python -m run.stage_curve
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pandas as pd

from . import common, locked_params as P
from .common import STATE_DIR, log

H2_INTRADAY_MINUTE_CUTOFF = (P.H_INTRADAY_WINDOW[0] - P.T_0900) / 60.0  # 13:00 -> 240분


def _frozen_entries(select_data: dict) -> list[tuple[str, SimpleNamespace, float]]:
    """(label, entry, resolved_quantile) 목록. label 은 'seed0'·'naive'·'null'."""
    out: list[tuple[str, SimpleNamespace, float]] = []
    for seed_str, primary in select_data["primaries"].items():
        if not primary.get("eligible"):
            continue
        q = float(primary["contract_id"].split(":q")[1])
        out.append((f"seed{seed_str}", SimpleNamespace(ast=primary["ast"]), q))
    for label in ("naive", "null"):
        primary = select_data[f"{label}_primary"]
        if not primary.get("eligible"):
            continue
        q = float(primary["contract_id"].split(":q")[1])
        out.append((label, SimpleNamespace(ast=primary["ast"]), q))
    return out


def _replay_stratum(entries: list[tuple[str, SimpleNamespace, float]], symbols: list[str],
                    stratum_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sd import replay as sd_replay
    from sd import select as sd_select

    entries_only = [e for _label, e, _q in entries]
    out_dir = STATE_DIR / f"replay_{stratum_name}"
    t0 = time.time()
    result = sd_replay.run(entries_only, symbols=symbols, date=P.DATE,
                           output_dir=out_dir, grid=list(P.QUANTILE_GRID), workers=32)
    t1 = time.time()
    log("curve", f"{stratum_name} 재생 완료 {t1-t0:.1f}s attempts={result.attempts}")

    ledger = pd.read_parquet(result.ledger_path)
    summary = sd_select.summarise(ledger).reset_index()
    summary["prefix"] = summary["contract_id"].str.split(":").str[0]
    prefix_to_label = {f"e{idx:03d}": label for idx, (label, _e, _q) in enumerate(entries)}
    prefix_to_q = {f"e{idx:03d}": q for idx, (_l, _e, q) in enumerate(entries)}
    summary["label"] = summary["prefix"].map(prefix_to_label)
    summary["quantile"] = summary["prefix"].map(prefix_to_q)

    # 동결된(=M 에서 뽑힌) 분위만 골라 "그 후보 하나"의 공식 성과로 삼는다.
    summary["resolved_q"] = summary["contract_id"].str.split(":q").str[1].astype(float)
    official = summary[summary["quantile"] == summary["resolved_q"]].copy()

    return official, ledger


def main() -> int:
    out_path = STATE_DIR / "curve.json"
    if out_path.exists():
        log("curve", "이미 완료 — 스킵(재개)")
        return 0

    select_path = STATE_DIR / "select.json"
    if not select_path.exists():
        raise SystemExit("select.json 이 없다 — stage_select 를 먼저 돌려라")
    select_data = json.loads(select_path.read_text())

    entries = _frozen_entries(select_data)
    if not entries:
        raise SystemExit("동결된 후보가 하나도 없다 — 모든 시드·기준선이 자격 미달")
    log("curve", f"동결 후보 {len(entries)}개: {[l for l, _e, _q in entries]}")

    strata_df = common.stratified_universe()
    groups = common.friction_group_symbols(strata_df)
    L, H = groups["L"], groups["H"]

    L_official, _L_ledger = _replay_stratum(entries, L, "L")
    H_official, H_ledger = _replay_stratum(entries, H, "H")

    # H2② — H 원장을 13:00 이후로 잘라 일중 외삽을 본다 (minute_of_session
    # 이 없으면 건너뛰고 이유를 기록한다 — 이 스테이지를 막을 이유는 아니다).
    intraday_summary = None
    if "minute_of_session" in H_ledger.columns:
        from sd import select as sd_select
        late = H_ledger[H_ledger["minute_of_session"] >= H2_INTRADAY_MINUTE_CUTOFF].copy()
        if len(late):
            s = sd_select.summarise(late).reset_index()
            s["prefix"] = s["contract_id"].str.split(":").str[0]
            prefix_to_label = {f"e{idx:03d}": label for idx, (label, _e, _q) in enumerate(entries)}
            prefix_to_q = {f"e{idx:03d}": q for idx, (_l, _e, q) in enumerate(entries)}
            s["label"] = s["prefix"].map(prefix_to_label)
            s["resolved_q"] = s["contract_id"].str.split(":q").str[1].astype(float)
            s["quantile"] = s["prefix"].map(prefix_to_q)
            intraday_summary = s[s["quantile"] == s["resolved_q"]].to_dict("records")
    else:
        log("curve", "경고: 원장에 minute_of_session 열이 없다 — H2② 일중 진단을 건너뛴다")

    # M 결과는 stage_select 가 이미 저장했다 — 여기서는 라벨만 맞춰 합친다.
    M_by_label: dict[str, dict] = {}
    for seed_str, primary in select_data["primaries"].items():
        if primary.get("eligible"):
            M_by_label[f"seed{seed_str}"] = primary
    for label in ("naive", "null"):
        primary = select_data[f"{label}_primary"]
        if primary.get("eligible"):
            M_by_label[label] = primary

    def _row(df: pd.DataFrame, label: str) -> dict | None:
        rows = df[df["label"] == label]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {"total_net_bps": float(r["total_net_bps"]),
               "bps_per_decision": float(r["bps_per_decision"]),
               "scorable": int(r["scorable"]), "decisions": int(r["decisions"]),
               "fills": int(r["fills"]), "unfilled": int(r["unfilled"]),
               "censored": int(r["censored"]),
               "cohort_NET_RECOVERY": int(r.get("cohort_NET_RECOVERY", 0)),
               "cohort_EARLY_RECOVERY_LATE_REVERSAL": int(
                   r.get("cohort_EARLY_RECOVERY_LATE_REVERSAL", 0)),
               "cohort_COST_INSUFFICIENT": int(r.get("cohort_COST_INSUFFICIENT", 0)),
               "cohort_PERSISTENT_ADVERSE": int(r.get("cohort_PERSISTENT_ADVERSE", 0))}

    labels_present = [label for label, _e, _q in entries]
    curve = {}
    for label in labels_present:
        m_row = M_by_label.get(label)
        curve[label] = {
            "L": _row(L_official, label),
            "M": {"total_net_bps": m_row["M_total_net_bps"],
                 "bps_per_decision": m_row["M_bps_per_decision"],
                 "scorable": m_row["M_scorable"], "decisions": m_row["M_decisions"]}
                 if m_row else None,
            "H": _row(H_official, label),
        }

    # ---- 최종 판정 (PREREG.md 성공/실패 기준을 그대로 코드로 옮긴다) ----
    null_H = curve.get("null", {}).get("H")
    verdicts: dict[str, dict] = {}
    for seed_int in P.SEEDS:
        label = f"seed{seed_int}"
        seed_json = json.loads((STATE_DIR / f"seed_{seed_int}.json").read_text())
        gate_passed = bool(seed_json["gate_passed"])
        h_row = curve.get(label, {}).get("H")
        eligible_M = label in M_by_label
        if not eligible_M or h_row is None:
            verdicts[label] = {"pass": False, "reason": "M 자격 미달 또는 후보 없음",
                              "gate_passed": gate_passed}
            continue
        beats_null = (null_H is not None and h_row["total_net_bps"] > null_H["total_net_bps"])
        positive = h_row["total_net_bps"] > 0.0
        enough_scorable = h_row["scorable"] >= P.MIN_SCORABLE
        seed_pass = bool(gate_passed and positive and enough_scorable and beats_null)
        verdicts[label] = {
            "pass": seed_pass, "gate_passed": gate_passed,
            "H_total_net_bps_positive": positive,
            "H_scorable_sufficient": enough_scorable,
            "beats_null_on_H": beats_null,
            "H_total_net_bps": h_row["total_net_bps"],
            "null_H_total_net_bps": null_H["total_net_bps"] if null_H else None,
        }

    all_seeds_pass = all(v["pass"] for v in verdicts.values()) and len(verdicts) == len(P.SEEDS)
    final_claim = ("성공 — 3개 시드 전부 H 통과선을 넘었다" if all_seeds_pass else
                   "성공 주장 불가 — 3개 시드 중 전부가 통과하지 못했다 (R37-1)")
    log("curve", f"최종 판정: {final_claim}")
    for label, v in verdicts.items():
        log("curve", f"  {label}: {v}")

    output = {
        "schema": "e1_stage_curve.v1",
        "L_symbols": len(L), "M_symbols": select_data["M_symbols"], "H_symbols": len(H),
        "curve_L_M_H": curve,
        "H_intraday_after_1300": intraday_summary,
        "seed_verdicts": verdicts,
        "all_three_seeds_pass": all_seeds_pass,
        "final_claim": final_claim,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    log("curve", f"완료 -> {out_path}")

    common.git_commit(["run/state/curve.json"],
                      f"E1 실행: L/M/H 저하 곡선 + 최종 판정 완료 — {final_claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
