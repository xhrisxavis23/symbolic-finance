#!/usr/bin/env python3
"""Stage B — 시드별 컴파일 후보를 M(834종목, 전일)에서 재생해 1순위 후보를
고른다. 나이브(f)·영점(h) 비교군도 같은 재생 호출에 얹는다. 이 재생의
M 성과가 L/M/H 저하 곡선(R37-2)의 M 지점이기도 하다.

    python -m run.stage_select

`state/select.json` 이 있으면 스킵한다. `state/seed_{N}.json` 셋(전 시드)이
전부 있어야 한다.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pandas as pd
import sympy

from . import common, locked_params as P
from .common import STATE_DIR, log


def _load_seed(seed: int) -> tuple[dict, list[SimpleNamespace]] | None:
    """`None` 이면 이 시드가 아직(또는 영영) 없다는 뜻 — 호출부가 건너뛴다.

    3개 시드 전부를 요구하지 않는다: 한 시드가 복구 불가능하게 실패해도
    나머지로 부분 결과를 낼 수 있어야 한다(R37-1 은 "3개 전부 통과해야
    성공을 주장한다"이지 "1개라도 없으면 아무 결과도 안 낸다"가 아니다).
    최종 판정은 어차피 `len(verdicts)==3 and all(pass)` 로 걸린다 — 시드가
    비면 자동으로 성공 주장이 불가능해진다."""
    path = STATE_DIR / f"seed_{seed}.json"
    if not path.exists():
        log("select", f"경고: seed_{seed}.json 이 없다 — 이 시드는 빼고 진행한다")
        return None
    data = json.loads(path.read_text())
    entries = [SimpleNamespace(ast=e["ast"], normal_form=e["normal_form"],
                               complexity=e["complexity"], expr_str=e["expr_str"])
               for e in data["compiled_entries"]]
    return data, entries


def main() -> int:
    out_path = STATE_DIR / "select.json"
    if out_path.exists():
        log("select", "이미 완료 — 스킵(재개)")
        return 0

    from sd import replay as sd_replay
    from sd import select as sd_select
    from sd.compile import compile_candidates
    from sd.sr.base import Candidate

    strata_df = common.stratified_universe()
    groups = common.friction_group_symbols(strata_df)
    M = groups["M"]

    seed_data: dict[int, dict] = {}
    entries_flat: list[tuple[str, int, SimpleNamespace]] = []  # (kind, seed_or_-1, entry)
    for seed in P.SEEDS:
        loaded = _load_seed(seed)
        if loaded is None:
            continue
        data, entries = loaded
        seed_data[seed] = data
        for e in entries:
            entries_flat.append(("sr", seed, e))
        log("select", f"seed={seed} gate_passed={data['gate_passed']} "
                      f"compiled={data['n_compiled']}")
    if len(seed_data) < len(P.SEEDS):
        log("select", f"경고: {len(P.SEEDS)}개 시드 중 {len(seed_data)}개만 있다 — "
                      f"R37-1 상 '3개 전부 통과' 주장은 이 실행에서 불가능하다. "
                      f"부분 결과만 낸다")

    naive_candidate = Candidate(expr=sympy.Symbol(P.NAIVE_FEATURE), complexity=1,
                                in_sample_score=float("nan"), backend="baseline_naive",
                                seed=-1)
    null_candidate = Candidate(expr=sympy.Symbol(P.NULL_FEATURE), complexity=1,
                               in_sample_score=float("nan"), backend="baseline_null",
                               seed=-1)
    baseline_compiled, baseline_failed = compile_candidates([naive_candidate, null_candidate])
    if baseline_failed:
        raise SystemExit(f"기준선 컴파일 실패 — 이건 있어서는 안 된다: {baseline_failed}")
    for e in baseline_compiled:
        kind = "naive" if e.normal_form.count(P.NAIVE_FEATURE) else "null"
        entries_flat.append((kind, -1, SimpleNamespace(
            ast=e.ast, normal_form=e.normal_form, complexity=e.source.complexity,
            expr_str=str(e.source.expr))))

    if not any(kind == "sr" for kind, _s, _e in entries_flat):
        raise SystemExit("SR 후보가 하나도 컴파일되지 않았다 — 모든 시드가 실패했다 (F7 해당 가능)")

    entries_only = [e for _k, _s, e in entries_flat]
    prefix_map = {f"e{idx:03d}": (kind, seed) for idx, (kind, seed, _e) in enumerate(entries_flat)}

    out_dir = STATE_DIR / "replay_select"
    t0 = time.time()
    result = sd_replay.run(entries_only, symbols=M, date=P.DATE,
                           output_dir=out_dir, grid=list(P.QUANTILE_GRID), workers=32)
    t1 = time.time()
    log("select", f"M 재생 완료 {t1-t0:.1f}s attempts={result.attempts} "
                  f"entries={len(entries_only)}")

    ledger = pd.read_parquet(result.ledger_path)
    summary = sd_select.summarise(ledger)
    summary = summary.reset_index()
    summary["prefix"] = summary["contract_id"].str.split(":").str[0]
    summary["kind"] = summary["prefix"].map(lambda p: prefix_map[p][0])
    summary["seed"] = summary["prefix"].map(lambda p: prefix_map[p][1])

    primaries: dict[int, dict] = {}
    for seed in P.SEEDS:
        seed_summary = summary[(summary["kind"] == "sr") & (summary["seed"] == seed)]
        seed_summary = seed_summary.set_index("contract_id")
        ranked = sd_select.rank(seed_summary, min_scorable=P.MIN_SCORABLE)
        if ranked.empty:
            log("select", f"seed={seed}: scorable>={P.MIN_SCORABLE} 통과 후보 없음 — 자격 미달")
            primaries[seed] = {"eligible": False}
            continue
        top = ranked.iloc[0]
        contract_id = ranked.index[0]
        # contract_id 형식 "e{idx:03d}:q{quantile:.2f}" 에서 entry index 를 되찾는다
        entry_prefix = contract_id.split(":")[0]
        entry_idx = int(entry_prefix[1:])
        entry = entries_only[entry_idx]
        primaries[seed] = {
            "eligible": True,
            "contract_id": contract_id,
            "ast": entry.ast,
            "normal_form": entry.normal_form,
            "expr_str": entry.expr_str,
            "M_total_net_bps": float(top["total_net_bps"]),
            "M_bps_per_decision": float(top["bps_per_decision"]),
            "M_scorable": int(top["scorable"]),
            "M_decisions": int(top["decisions"]),
        }
        log("select", f"seed={seed} 1순위 {contract_id} "
                      f"net_bps={top['total_net_bps']:.2f} scorable={top['scorable']}")

    # 구조 복원율 — 시드 간 같은 normal_form 이 몇 번 나왔는가(전체 컴파일 후보 기준,
    # 1순위로 뽑힌 것만이 아니라 — 계획서 §4 S4 ⑤).
    all_normal_forms = [e.normal_form for kind, seed, e in entries_flat if kind == "sr"]
    from collections import Counter
    nf_counts = Counter(all_normal_forms)
    structural_recovery = {
        "n_unique_structures": len(nf_counts),
        "n_total_candidates": len(all_normal_forms),
        "top_structures": nf_counts.most_common(10),
    }

    def _pick_primary(kind: str) -> dict:
        """SR 후보와 완전히 같은 규칙(S4: scorable>=200, total_net_bps 최대)으로
        나이브·영점 기준선도 격자 중 하나를 고른다 — 비교 대상에게만 다른
        규칙을 쓰면 비교가 불공정해진다."""
        rows = summary[summary["kind"] == kind].set_index("contract_id")
        ranked = sd_select.rank(rows, min_scorable=P.MIN_SCORABLE)
        if ranked.empty:
            return {"eligible": False}
        top = ranked.iloc[0]
        contract_id = ranked.index[0]
        entry_idx = int(contract_id.split(":")[0][1:])
        entry = entries_only[entry_idx]
        return {"eligible": True, "contract_id": contract_id, "ast": entry.ast,
               "normal_form": entry.normal_form, "expr_str": entry.expr_str,
               "M_total_net_bps": float(top["total_net_bps"]),
               "M_bps_per_decision": float(top["bps_per_decision"]),
               "M_scorable": int(top["scorable"]),
               "M_decisions": int(top["decisions"])}

    naive_primary = _pick_primary("naive")
    null_primary = _pick_primary("null")
    log("select", f"나이브(f) 1순위: {naive_primary.get('contract_id', '자격없음')}")
    log("select", f"영점(h) 1순위: {null_primary.get('contract_id', '자격없음')}")

    naive_summary = summary[summary["kind"] == "naive"].to_dict("records")
    null_summary = summary[summary["kind"] == "null"].to_dict("records")

    output = {
        "schema": "e1_stage_select.v1",
        "M_symbols": len(M),
        "replay_seconds": t1 - t0,
        "attempts": result.attempts,
        "primaries": primaries,
        "naive_primary": naive_primary,
        "null_primary": null_primary,
        "structural_recovery": structural_recovery,
        "naive_M_summary": naive_summary,
        "null_M_summary": null_summary,
        "all_M_summary": summary.to_dict("records"),
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    log("select", f"완료 -> {out_path}")

    # M 선택 재생은 낙선 후보가 대부분이라(96개 안팎 계약 중 최종 동결은
    # 시드당 최대 1개) 원장을 계속 들고 있을 값어치가 적다. 필요한 숫자는
    # 이미 select.json 에 다 옮겼다. 디스크 여유 45GB 를 감안해 지운다
    # (SCALE-MEASUREMENT.md §1) — L/M/H 최종 곡선 원장(stage_curve)은 후보
    # 수가 적어(최대 5개) 남겨 둔다.
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    log("select", f"원장 정리 완료 ({out_dir})")

    common.git_commit(["run/state/select.json"],
                      f"E1 실행: M 선택 완료 (자격 후보 "
                      f"{sum(1 for p in primaries.values() if p.get('eligible'))}/{len(P.SEEDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
