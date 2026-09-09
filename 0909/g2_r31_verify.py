#!/usr/bin/env python3
"""Ruling R31 조건 이행 확인. `PROGRESS.md` 참고.

1. L4 를 종목 내 중심화로 고친 뒤 음성 대조군을 다시 돌려 0/5 로
   떨어지는지 확인한다(조건 1).
2. L1 양성 대조군이 여전히 자격을 유지하는지 확인한다(조건 2) — L4 만
   고쳤으므로 L1 의 코드 경로는 바뀌지 않았지만, 다시 실행해 재확인한다.
3. L4 의 실제(안 섞은) 데이터 참조천장을 다시 재서, 고친 뒤 자격이
   어떻게 나오는지 있는 그대로 기록한다(조건 3 — 자격 없음으로 나와도
   그대로 보고).

전체 실행 확정 설정(층당 참조 20, cap 20000, K=5, REPEATS=20)에서 돈다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd import config, universe, manifold      # noqa: E402
from sd.e0 import targets                       # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
RESULTS_DIR = REPO_0909 / "results"

import g2_split       # noqa: E402
import g2_reference   # noqa: E402

DATE = config.DATE
assert DATE == "20260316", f"허용되지 않은 날짜: {DATE}"

N_REF_PER_STRATUM = 20
N_DISTILL_PER_STRATUM = 6
MAX_MANIFOLD_SAMPLES_REFERENCE = 20000
KFOLD_K = 5
KFOLD_REPEATS = 20
N_PERMUTATIONS = 5
SEED = config.SEED


def _reference_arrays(law: str, ref_symbols: tuple):
    ref_ds = targets.assemble(law, ref_symbols, DATE)
    selection = manifold.select(ref_ds.X_dimless, ref_ds.mask,
                                max_samples=MAX_MANIFOLD_SAMPLES_REFERENCE, seed=SEED)
    on_rows = selection.index[selection.on_manifold]
    return (ref_ds.X_dimless[on_rows], ref_ds.y_dimless[on_rows],
            ref_ds.symbol_ids[on_rows], selection.weight[selection.on_manifold])


def _negative_control(law: str, X, y, sid, w) -> dict:
    out = {}
    for scope in ("within_symbol", "global"):
        results = []
        for perm_i in range(N_PERMUTATIONS):
            y_perm = g2_reference.permute_y(y, sid, scope=scope, seed=1000 * perm_i + 7)
            mask = np.ones(len(y_perm), dtype=bool)
            ceiling = g2_reference.repeated_kfold_ceiling(
                law, X, y_perm, mask, sid, w, k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
            eligible = bool(ceiling.get("ok") and ceiling["r2_ci_low"] > 0.0)
            results.append({"perm_index": perm_i, "eligible": eligible,
                            "ceiling_summary": {k: ceiling.get(k) for k in
                                                ("ok", "r2_median", "r2_ci_low", "r2_ci_high")}})
        out[scope] = results
    return out


def main() -> None:
    t0 = time.time()
    all_symbols = universe.stock_symbols(DATE)
    split = g2_split.build_three_way_split(
        all_symbols, DATE, n_ref_per_stratum=N_REF_PER_STRATUM,
        n_distill_per_stratum=N_DISTILL_PER_STRATUM)
    print(f"참조 종목 {len(split.reference)}개\n", flush=True)

    report = {}

    # --- 조건 1: L4 음성 대조군 재확인 ---------------------------------------
    print("=== 조건 1 — L4 음성 대조군 재확인 (수정 후) ===", flush=True)
    X4, y4, sid4, w4 = _reference_arrays("L4", split.reference)
    neg_l4 = _negative_control("L4", X4, y4, sid4, w4)
    for scope, results in neg_l4.items():
        n_elig = sum(r["eligible"] for r in results)
        print(f"  L4 {scope}: {n_elig}/{N_PERMUTATIONS} 허위양성")
        for r in results:
            c = r["ceiling_summary"]
            print(f"    perm#{r['perm_index']}: eligible={r['eligible']} "
                  f"median={c.get('r2_median')} CI=[{c.get('r2_ci_low')},{c.get('r2_ci_high')}]")
    report["negative_control_L4_after_fix"] = neg_l4

    # --- 조건 2: L1 양성 대조군 재확인 ---------------------------------------
    print("\n=== 조건 2 — L1 양성 대조군 재확인 ===", flush=True)
    X1, y1, sid1, w1 = _reference_arrays("L1", split.reference)
    mask1 = np.ones(len(y1), dtype=bool)
    ceiling_l1 = g2_reference.repeated_kfold_ceiling(
        "L1", X1, y1, mask1, sid1, w1, k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
    l1_eligible = bool(ceiling_l1.get("ok") and ceiling_l1["r2_ci_low"] > 0.0)
    print(f"  L1 실측(안 섞음) 참조천장: median={ceiling_l1.get('r2_median')} "
          f"CI=[{ceiling_l1.get('r2_ci_low')},{ceiling_l1.get('r2_ci_high')}] eligible={l1_eligible}")
    neg_l1 = _negative_control("L1", X1, y1, sid1, w1)
    for scope, results in neg_l1.items():
        n_elig = sum(r["eligible"] for r in results)
        print(f"  L1 {scope} 음성대조군: {n_elig}/{N_PERMUTATIONS} 허위양성")
    report["l1_positive_control_ceiling"] = ceiling_l1
    report["l1_positive_control_eligible"] = l1_eligible
    report["negative_control_L1_recheck"] = neg_l1

    # --- 조건 3: L4 실측(안 섞음) 참조천장 — 있는 그대로 보고 ------------------
    print("\n=== 조건 3 — L4 실측(안 섞음) 참조천장 (수정 후) ===", flush=True)
    mask4 = np.ones(len(y4), dtype=bool)
    ceiling_l4_real = g2_reference.repeated_kfold_ceiling(
        "L4", X4, y4, mask4, sid4, w4, k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
    l4_eligible_real = bool(ceiling_l4_real.get("ok") and ceiling_l4_real["r2_ci_low"] > 0.0)
    print(f"  L4 실측 참조천장(수정 후): median={ceiling_l4_real.get('r2_median')} "
          f"CI=[{ceiling_l4_real.get('r2_ci_low')},{ceiling_l4_real.get('r2_ci_high')}] "
          f"eligible={l4_eligible_real}")
    report["l4_real_data_ceiling_after_fix"] = ceiling_l4_real
    report["l4_real_data_eligible_after_fix"] = l4_eligible_real

    total_s = time.time() - t0
    l4_precondition_ok = all(r["eligible"] is False for r in neg_l4["within_symbol"])
    l1_precondition_ok = all(r["eligible"] is False for r in neg_l1["within_symbol"]) and l1_eligible

    report["summary"] = {
        "l4_within_symbol_false_positive_after_fix": sum(r["eligible"] for r in neg_l4["within_symbol"]),
        "l4_negative_control_now_clean": l4_precondition_ok,
        "l1_still_eligible_and_clean": l1_precondition_ok,
        "l4_real_data_eligible_after_fix": l4_eligible_real,
    }
    report["total_seconds"] = total_s
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = RESULTS_DIR / "g2_r31_verify.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    print("\n=== 요약 ===")
    print(f"  조건1 L4 음성대조군(within_symbol) 허위양성: "
          f"{report['summary']['l4_within_symbol_false_positive_after_fix']}/5 "
          f"({'통과' if l4_precondition_ok else '실패 — 원인이 다른 데 있음'})")
    print(f"  조건2 L1 양성대조군 유지: {'통과' if l1_precondition_ok else '실패'}")
    print(f"  조건3 L4 실측 자격(수정 후): {l4_eligible_real} "
          f"(참고: L0 직접측정 R²=0.335, E0 밴드 0.85 밖)")
    print(f"총 소요: {total_s:.1f}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
