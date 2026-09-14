#!/usr/bin/env python3
"""G2 — 음성 대조군. Ruling R30(코디네이터, `PROGRESS.md`)의 요구 이행.

"측정이 고장났는데 L1 만 우연히 통과하는 경우를 못 잡는다" — 양성
대조군(L1) 하나만으로는 이 위험을 못 막는다. 그래서 **실제 데이터에서
`y` 를 섞은(permuted) 관계**를 §3.1 의 반복 K-fold 참조천장에 그대로
통과시켜, **반드시 자격 없음(CI 하한 ≤ 0)이 나와야 한다**는 선결 조건을
추가한다.

섞기 범위(`scope`)는 `g2_reference.permute_y` 의 docstring 에 근거를
적었다 — 요약: `within_symbol`(종목 안에서만 섞음, §4.1 선결 조건이 요구
하는 쪽 — 이 설계의 종목단위 K-fold 가 만들 수 있는 "종목 수준 생태학적
상관" 허위양성까지 잡는 더 엄격한 검사)과 `global`(참조 전체를 섞음,
표준적인 자기검산, 보조 진단)을 둘 다 낸다.

**교사·SR 은 전혀 안 부른다** — 참조 측정만(코디네이터: "참조 측정만이면
지금 해도 됩니다"). 전체 실행 확정 설정(층당 참조 20, manifold cap
20000, K=5, REPEATS=20)에서 돌린다 — 그래야 이 음성 대조군이 실제로
전체 실행에 쓰일 그 절차를 검사한 게 된다.
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

LAWS = ("L1", "L2", "L3", "L4", "L5")   # L1 은 §4.1 필수 선결 조건, 나머지는 코드 커버리지 확장(추가 안심)
SCOPES = ("within_symbol", "global")     # within_symbol 이 선결 조건(§4.1) 대상, global 은 보조 진단
N_PERMUTATIONS = 5                       # 섞기 한 번의 우연을 배제 — 독립 시드 5개

# --- 전체 실행 확정 설정(민감도 확인·Ruling R30 근거) ------------------------
N_REF_PER_STRATUM = 20
N_DISTILL_PER_STRATUM = 6     # 이 확인에선 증류 쪽을 안 쓰지만 3분할 함수 인자로 필요 — 최소값만
MAX_MANIFOLD_SAMPLES_REFERENCE = 20000
KFOLD_K = 5
KFOLD_REPEATS = 20
SEED = config.SEED


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_symbols = universe.stock_symbols(DATE)
    split = g2_split.build_three_way_split(
        all_symbols, DATE, n_ref_per_stratum=N_REF_PER_STRATUM,
        n_distill_per_stratum=N_DISTILL_PER_STRATUM)
    print(f"참조 종목 {len(split.reference)}개 (전체 실행 확정 설정과 동일)\n", flush=True)

    results: dict = {}
    for law in LAWS:
        results[law] = {}
        ref_ds = targets.assemble(law, split.reference, DATE)
        selection = manifold.select(ref_ds.X_dimless, ref_ds.mask,
                                    max_samples=MAX_MANIFOLD_SAMPLES_REFERENCE, seed=SEED)
        on_rows = selection.index[selection.on_manifold]
        X_R = ref_ds.X_dimless[on_rows]
        y_R = ref_ds.y_dimless[on_rows]
        sid_R = ref_ds.symbol_ids[on_rows]
        w_R = selection.weight[selection.on_manifold]
        mask_R = np.ones(len(y_R), dtype=bool)
        print(f"=== {law} (n_rows={len(y_R)}, n_symbols={len(np.unique(sid_R))}) ===", flush=True)

        for scope in SCOPES:
            results[law][scope] = []
            for perm_i in range(N_PERMUTATIONS):
                t_iter = time.time()
                y_perm = g2_reference.permute_y(y_R, sid_R, scope=scope, seed=1000 * perm_i + 7)
                ceiling = g2_reference.repeated_kfold_ceiling(
                    law, X_R, y_perm, mask_R, sid_R, w_R, k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
                eligible = bool(ceiling.get("ok") and ceiling["r2_ci_low"] > 0.0)
                elapsed = time.time() - t_iter
                if ceiling.get("ok"):
                    print(f"  {scope} perm#{perm_i}: median={ceiling['r2_median']:.4f} "
                          f"CI=[{ceiling['r2_ci_low']:.4f},{ceiling['r2_ci_high']:.4f}] "
                          f"eligible(허위양성이면 문제)={eligible} ({elapsed:.1f}s)")
                else:
                    print(f"  {scope} perm#{perm_i}: 실패({ceiling.get('reason')}) — "
                          f"자격 없음으로 취급 ({elapsed:.1f}s)")
                results[law][scope].append({"perm_index": perm_i, "ceiling": ceiling,
                                            "eligible": eligible, "elapsed_seconds": elapsed})
        print()

    total_s = time.time() - t0

    # --- 요약: within_symbol 스코프에서 단 하나라도 자격 있음(허위양성)이면 위험 신호 ---
    summary = {}
    any_false_positive_within_symbol = False
    any_false_positive_global = False
    for law in LAWS:
        elig_within = [r["eligible"] for r in results[law]["within_symbol"]]
        elig_global = [r["eligible"] for r in results[law]["global"]]
        summary[law] = {"within_symbol_any_eligible": any(elig_within),
                        "within_symbol_n_eligible_of": f"{sum(elig_within)}/{len(elig_within)}",
                        "global_any_eligible": any(elig_global),
                        "global_n_eligible_of": f"{sum(elig_global)}/{len(elig_global)}"}
        if any(elig_within):
            any_false_positive_within_symbol = True
        if any(elig_global):
            any_false_positive_global = True

    # §4.1 필수 선결 조건 — L1, within_symbol, 5회 전부 자격 없음이어야 통과
    l1_precondition_ok = not summary["L1"]["within_symbol_any_eligible"]

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "date": DATE,
        "purpose": "Ruling R30 조건 이행 — 음성 대조군(permuted y), §4.1 선결 조건",
        "laws": list(LAWS), "scopes": list(SCOPES), "n_permutations": N_PERMUTATIONS,
        "config": {"n_ref_per_stratum": N_REF_PER_STRATUM,
                  "max_manifold_samples_reference": MAX_MANIFOLD_SAMPLES_REFERENCE,
                  "kfold_k": KFOLD_K, "kfold_repeats": KFOLD_REPEATS, "seed": SEED},
        "results": results, "summary": summary,
        "precondition": {"law": "L1", "scope": "within_symbol",
                        "required": "5/5 회 자격 없음(허위양성 0)",
                        "passed": l1_precondition_ok},
        "any_false_positive_within_symbol": any_false_positive_within_symbol,
        "any_false_positive_global": any_false_positive_global,
        "total_seconds": total_s,
    }
    out_path = RESULTS_DIR / "g2_negative_control.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    print("=== 요약 ===")
    for law in LAWS:
        s = summary[law]
        print(f"  {law}: within_symbol 허위양성={s['within_symbol_n_eligible_of']} "
              f"/ global 허위양성={s['global_n_eligible_of']}")
    print(f"\n§4.1 선결 조건(L1, within_symbol, 5/5 자격없음 요구): "
          f"{'통과' if l1_precondition_ok else '실패 — 측정 코드 재검토 필요'}")
    print(f"총 소요: {total_s:.1f}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
