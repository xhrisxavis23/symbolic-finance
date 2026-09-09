#!/usr/bin/env python3
"""G2 — 참조 규모 민감도 확인. `PROGRESS.md` Ruling R29 조건의 이행.

코디네이터 검토(커밋 `1d38aa7`, 이후 메시지로 재확인)가 승인 조건으로
요구했다: "층당 참조 15/20/25 에서 **L1·L5** 대조군의 CI 가 얼마나
흔들리는지 재라. SR·교사 없이 참조 측정만." (`run_law` 는 전혀 안 부른다
— §3.1 의 반복 K-fold 천장 계산만.)

**두 축을 동시에 본다:**
  (1) `n_ref_per_stratum` — 15/20/25 (코디네이터가 지정)
  (2) `max_manifold_samples` — 원래 설계값(4000, 증류 SR 적합과 같은 값을
      그냥 재사용했었다) vs **참조 천장 전용으로 분리한 값**(20000)

(2)를 같이 보는 이유: 배관 확인 단계에서 4000 cap 으로 L1 을 15/20/25
재봤더니 **자격(있음/없음) 자체는 항상 안정**이었지만 CI 폭 자체가 꽤
흔들렸다(아래 결과 참고) — 원인을 진단해 보니 `max_manifold_samples`
가 참조 집합 크기와 무관하게 표본을 4000행으로 캡핑해서(종목을 몇 개
넣든 반복 K-fold 가 보는 실제 행 수가 거의 안 늘어난다) 종목 수를 늘려도
안정화 효과가 제한적이었다. 참조 천장 계산은 SR·교사가 없어 계산비용이
싸므로 이 cap 을 낮게 유지할 이유가 약하다 — 그래서 분리해서 같이 잰다.
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

LAWS = ("L1", "L5")                 # 코디네이터 지정 대조군 둘 다
PER_STRATUM_VALUES = (15, 20, 25)   # 코디네이터 지정
N_DISTILL_PER_STRATUM = 6           # 3분할 함수 인자 — 이 확인에선 안 쓰지만 최소값만 채운다
MANIFOLD_CAPS = (4000, 20000)       # 원래 설계값 vs 참조 전용 분리값(원인 진단)
KFOLD_K = 5                         # 전체 실행 사전등록값(§3.1)
KFOLD_REPEATS = 20                  # 전체 실행 사전등록값(§3.1)
SEED = config.SEED


def _one(law: str, ref_symbols: tuple, max_manifold_samples: int) -> dict:
    ref_ds = targets.assemble(law, ref_symbols, DATE)
    selection = manifold.select(ref_ds.X_dimless, ref_ds.mask,
                                max_samples=max_manifold_samples, seed=SEED)
    on_rows = selection.index[selection.on_manifold]
    X_R = ref_ds.X_dimless[on_rows]
    y_R = ref_ds.y_dimless[on_rows]
    sid_R = ref_ds.symbol_ids[on_rows]
    w_R = selection.weight[selection.on_manifold]
    mask_R = np.ones(len(y_R), dtype=bool)

    ceiling = g2_reference.repeated_kfold_ceiling(
        law, X_R, y_R, mask_R, sid_R, w_R, k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
    eligible = bool(ceiling.get("ok") and ceiling["r2_ci_low"] > 0.0)
    return {"n_reference_symbols_used": len(ref_ds.symbols_used),
            "n_on_manifold_rows": int(len(on_rows)), "ceiling": ceiling, "eligible": eligible}


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_symbols = universe.stock_symbols(DATE)

    # 참조 종목 집합은 per_stratum 마다 한 번만 뽑는다(두 cap 이 같은 종목
    # 집합을 공유해야 "cap 만 바꿨을 때" 비교가 깨끗하다).
    splits = {}
    for per_stratum in PER_STRATUM_VALUES:
        splits[per_stratum] = g2_split.build_three_way_split(
            all_symbols, DATE, n_ref_per_stratum=per_stratum,
            n_distill_per_stratum=N_DISTILL_PER_STRATUM)

    results: dict = {}
    for law in LAWS:
        results[law] = {}
        for per_stratum in PER_STRATUM_VALUES:
            results[law][per_stratum] = {}
            ref_symbols = splits[per_stratum].reference
            for cap in MANIFOLD_CAPS:
                t_iter = time.time()
                print(f"=== {law} / 층당 참조 {per_stratum} / cap={cap} ===", flush=True)
                out = _one(law, ref_symbols, cap)
                elapsed = time.time() - t_iter
                out["elapsed_seconds"] = elapsed
                c = out["ceiling"]
                if c.get("ok"):
                    print(f"  n_symbols={out['n_reference_symbols_used']} "
                          f"n_rows={out['n_on_manifold_rows']} "
                          f"median={c['r2_median']:.4f} "
                          f"CI=[{c['r2_ci_low']:.4f},{c['r2_ci_high']:.4f}] "
                          f"eligible={out['eligible']} ({elapsed:.1f}s)")
                else:
                    print(f"  실패: {c.get('reason')} ({elapsed:.1f}s)")
                results[law][per_stratum][cap] = out

    total_s = time.time() - t0

    # --- 요약: 각 (law, cap) 에서 per_stratum 15/20/25 에 걸친 CI 스프레드 ---
    summary = {}
    for law in LAWS:
        summary[law] = {}
        for cap in MANIFOLD_CAPS:
            lows = [results[law][p][cap]["ceiling"].get("r2_ci_low") for p in PER_STRATUM_VALUES
                   if results[law][p][cap]["ceiling"].get("ok")]
            highs = [results[law][p][cap]["ceiling"].get("r2_ci_high") for p in PER_STRATUM_VALUES
                    if results[law][p][cap]["ceiling"].get("ok")]
            all_elig = all(results[law][p][cap]["eligible"] for p in PER_STRATUM_VALUES)
            summary[law][cap] = {
                "all_eligible_at_15_20_25": all_elig,
                "ci_low_spread": (max(lows) - min(lows)) if len(lows) == 3 else None,
                "ci_high_spread": (max(highs) - min(highs)) if len(highs) == 3 else None,
                "ci_low_values": lows, "ci_high_values": highs,
            }

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "date": DATE,
        "purpose": "PROGRESS.md Ruling R29 조건 이행 — 참조 규모(층당 15/20/25) x "
                   "manifold cap(4000 vs 20000) 민감도, L1·L5 대조군",
        "laws": list(LAWS), "per_stratum_values": list(PER_STRATUM_VALUES),
        "manifold_caps": list(MANIFOLD_CAPS),
        "config": {"kfold_k": KFOLD_K, "kfold_repeats": KFOLD_REPEATS, "seed": SEED},
        "results": results, "summary": summary, "total_seconds": total_s,
    }
    out_path = RESULTS_DIR / "g2_sensitivity_check.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    print("\n=== 요약 ===")
    for law in LAWS:
        for cap in MANIFOLD_CAPS:
            s = summary[law][cap]
            print(f"  {law} cap={cap}: 15/20/25 모두 자격있음={s['all_eligible_at_15_20_25']} "
                  f"CI하한값={['%.3f' % v for v in s['ci_low_values']]} "
                  f"(스프레드 {s['ci_low_spread']:.4f})" if s['ci_low_spread'] is not None else
                  f"  {law} cap={cap}: 실패 있음")
    print(f"\n총 소요: {total_s:.1f}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
