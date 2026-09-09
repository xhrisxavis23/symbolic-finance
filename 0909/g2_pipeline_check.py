#!/usr/bin/env python3
"""G2 배관 확인 — `PREREG-G2.md` §6. **작은 규모로 배관만 확인한다. 결과가
아니다.** 전체 실행은 승인 후.

확인하는 것:
  1. 참조/증류-적합/증류-선택 3분할이 실제로 서로소로 나오는가 (assert)
  2. 참조 집합 안에서 반복 K-fold 로 "천장" CI 가 실제로 계산되는가
  3. `sd.e0.runner.run_law` (기존 코드, 손대지 않음) 가 낸 primary 후보를
     참조 집합에서 실측 y 로 재채점하는 배선이 끊기지 않는가
  4. "단순회귀 기준선"(§2.3)이 같은 방식으로 계산되는가
  5. 전체 소요시간이 선형 외삽했을 때 합리적인 범위인가

SR 백엔드는 `naive`(Julia 불필요) 로 **임시 교체**한다 — PySR 자체의 통합은
0902 가 11회 E0 실행으로 이미 검증했으므로, 여기서 다시 검증할 대상이
아니다(`PREREG-G2.md` §6). `sd.e0.runner` 모듈 파일은 전혀 안 건드린다 —
이 스크립트 안에서 그 모듈이 참조하는 `_build_backend` 함수 객체만 실행 중에
바꿔치기한다(런타임 몽키패치, 파일 수정 아님).
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

from sd import config, universe, manifold           # noqa: E402
from sd.e0 import targets, runner as runner_mod      # noqa: E402
from sd.sr.naive import NaiveBackend                 # noqa: E402
from sd.sr.base import score_candidate               # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
RESULTS_DIR = REPO_0909 / "results"

import g2_split                                        # noqa: E402
import g2_reference                                    # noqa: E402

DATE = config.DATE
assert DATE == "20260316", f"허용되지 않은 날짜: {DATE}"

# --- 배관 확인 전용 소규모 (§6) — 전체 실행값(참조 20/증류 24 층당, 총 ≤264
# 종목)이 아니다. 처음엔 층당 참조 3/증류 6(총 54종목, manifold cap 800)으로
# 시도했는데 L1(대조군, L0이 이미 참으로 확인)의 참조 천장 CI가 0을 가로질러
# "자격 없음"으로 나왔다 — `G2-DESIGN-NOTES.md` §"발견 2"에 기록된 대로 이건
# 표본이 너무 작아서였다(795행/18종목, 이중 구간화가 잡음을 증폭). 층당
# 참조 10/증류 10(총 120종목), manifold cap 2500 으로 올리자 L1 이 CI=[0.62,
# 0.79]로 뚜렷이 자격을 얻었다 — 그래서 이 값을 배관 확인의 최종 규모로 쓴다.
# 그래도 전체 실행값(264종목)의 절반 이하다.
N_REF_PER_STRATUM = 10
N_DISTILL_PER_STRATUM = 10
MAX_MANIFOLD_SAMPLES_CHECK = 2500
KFOLD_K = 3
KFOLD_REPEATS = 6                    # 전체 실행값(20)보다 작게 — 속도
TEACHER_EPOCHS_CHECK = 20            # 전체 실행값(300)보다 작게 — 배관만 본다
LAWS_TO_CHECK = ("L1", "L2", "L4")   # L1=대조군(존재 확인), L2=L0에서 "없음", L4=L0에서 "중간" — 세 상태를 다 보려는 선택


class _NaiveBackendWithDiagnostics(NaiveBackend):
    """`PySRBackend` 는 `.diagnostics` 속성을 낸다(`runner._fit_track` 이 그걸
    읽는다) — `NaiveBackend` 는 그 계약이 없다(배관 검증용으로만 쓰였던
    이력이라 애초에 필요 없었다). `runner.py` 를 고치지 않고 여기서 얇은
    어댑터로 메꾼다."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diagnostics: dict = {"backend": self.name, "seed": self.seed}


def _naive_backend(law: str, seed: int, niterations: int, maxsize: int):
    return _NaiveBackendWithDiagnostics(seed=seed)


def _score_on_reference(law: str, ref_symbols: tuple, date: str, *,
                        max_manifold_samples: int, kfold_k: int, kfold_repeats: list[int],
                        distill_fit: tuple, distill_select: tuple, seed: int) -> dict:
    t0 = time.time()
    ref_ds = targets.assemble(law, ref_symbols, date)
    names = ref_ds.names_dimless
    assemble_s = time.time() - t0

    # 참조 집합의 on-manifold 부분집합 — L0 과 같은 신뢰반경 철학 (PREREG-G2 §3.5).
    selection = manifold.select(ref_ds.X_dimless, ref_ds.mask, max_samples=max_manifold_samples, seed=seed)
    on_rows = selection.index[selection.on_manifold]
    on_weight = selection.weight[selection.on_manifold]
    X_R = ref_ds.X_dimless[on_rows]
    y_R = ref_ds.y_dimless[on_rows]
    sid_R = ref_ds.symbol_ids[on_rows]
    mask_R = np.ones(len(y_R), dtype=bool)

    ceilings = {}
    for reps in kfold_repeats:
        ceilings[reps] = g2_reference.repeated_kfold_ceiling(
            law, X_R, y_R, mask_R, sid_R, on_weight, k=kfold_k, repeats=reps, seed=seed)
    ceiling_main = ceilings[kfold_repeats[-1]]

    eligible = bool(ceiling_main.get("ok") and ceiling_main["r2_ci_low"] > 0.0)

    # --- 증류: 기존 run_law 그대로, naive 백엔드로 임시 교체(속도) ------------
    original_build_backend = runner_mod._build_backend
    runner_mod._build_backend = _naive_backend
    try:
        t1 = time.time()
        law_result = runner_mod.run_law(
            law, distill_fit, distill_select, date, seed=seed,
            epochs=TEACHER_EPOCHS_CHECK, sr_niterations=5, sr_maxsize=12,
            max_manifold_samples=max_manifold_samples)
        distill_s = time.time() - t1
    finally:
        runner_mod._build_backend = original_build_backend

    main_track = law_result.tracks.get("main")
    nodist_track = law_result.tracks.get("no_distillation")

    def _primary_score(track):
        if track is None or track.primary_index is None:
            return None, None, None
        cand = track.candidates[track.primary_index]
        s_binned = g2_reference.score_candidate_binned(cand, names, X_R, y_R, on_weight)
        s_row = score_candidate(cand, names, X_R, y_R, on_weight)   # 진단용, 판정에는 안 씀 (§3.1 binned 채택 근거)
        return (str(cand.expr), (float(s_binned) if np.isfinite(s_binned) else None),
                (float(s_row) if np.isfinite(s_row) else None))

    main_expr, score_main, score_main_row = _primary_score(main_track)
    nodist_expr, score_nodist, score_nodist_row = _primary_score(nodist_track)

    # --- 단순회귀 기준선: D_fit∪D_select 전체에 적합, R 에서 채점 ---------------
    fit_ds = targets.assemble(law, distill_fit, date)
    select_ds = targets.assemble(law, distill_select, date)
    X_D = np.concatenate([fit_ds.X_dimless, select_ds.X_dimless], axis=0)
    y_D = np.concatenate([fit_ds.y_dimless, select_ds.y_dimless], axis=0)
    mask_D = np.concatenate([fit_ds.mask, select_ds.mask], axis=0)
    sid_D = np.concatenate([fit_ds.symbol_ids, select_ds.symbol_ids], axis=0)
    pred_baseline = g2_reference.fit_on_all_and_predict(law, X_D, y_D, mask_D, X_R,
                                                        symbol_ids_fit=sid_D, symbol_ids_score=sid_R)
    score_baseline = None
    if pred_baseline is not None:
        r2b = g2_reference.binned_r2(y_R, pred_baseline, on_weight)
        score_baseline = float(r2b) if np.isfinite(r2b) else None

    recovered = bool(eligible and score_main is not None and score_main >= ceiling_main["r2_ci_low"])
    exceeds_ceiling = bool(eligible and score_main is not None and score_main > ceiling_main["r2_ci_high"])
    no_added_value = bool(score_baseline is not None and score_main is not None
                          and score_baseline > score_main)
    teacher_harmed = bool(score_main is not None and score_nodist is not None
                          and score_nodist > score_main + 0.05)

    return {
        "law": law, "assemble_seconds": assemble_s, "distill_seconds": distill_s,
        "n_reference_symbols_used": len(ref_ds.symbols_used),
        "n_reference_rows_total": ref_ds.n_rows_total, "n_reference_on_manifold": int(len(on_rows)),
        "ceilings_by_repeats": ceilings, "eligible": eligible,
        "main_expr": main_expr, "score_main_on_reference": score_main,
        "score_main_on_reference_row_diagnostic": score_main_row,
        "no_distillation_expr": nodist_expr, "score_no_distillation_on_reference": score_nodist,
        "score_no_distillation_on_reference_row_diagnostic": score_nodist_row,
        "score_baseline_on_reference": score_baseline,
        "recovered": recovered, "exceeds_ceiling": exceeds_ceiling,
        "no_added_value": no_added_value, "teacher_harmed": teacher_harmed,
        "n_distill_fit_symbols_used": len(fit_ds.symbols_used),
        "n_distill_select_symbols_used": len(select_ds.symbols_used),
    }


def main() -> None:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. 3분할 조립 ===", flush=True)
    all_symbols = universe.stock_symbols(DATE)
    split = g2_split.build_three_way_split(
        all_symbols, DATE, n_ref_per_stratum=N_REF_PER_STRATUM,
        n_distill_per_stratum=N_DISTILL_PER_STRATUM)
    print(f"  참조 {len(split.reference)} / 증류-적합 {len(split.distill_fit)} / "
          f"증류-선택 {len(split.distill_select)}")

    # 서로소 확인 — dataclass __post_init__ 이 이미 검사하지만, 스크립트에서도
    # 명시적으로 다시 확인해 배관 확인 로그에 결과를 남긴다.
    r, f, s = set(split.reference), set(split.distill_fit), set(split.distill_select)
    assert not (r & f) and not (r & s) and not (f & s), "서로소 위반 — 3분할 버그"
    print(f"  서로소 확인: 통과 (|R∩Fit|={len(r&f)}, |R∩Select|={len(r&s)}, |Fit∩Select|={len(f&s)})\n")

    results = {}
    for law in LAWS_TO_CHECK:
        print(f"=== {law} ===", flush=True)
        t_law = time.time()
        try:
            res = _score_on_reference(
                law, split.reference, DATE, max_manifold_samples=MAX_MANIFOLD_SAMPLES_CHECK,
                kfold_k=KFOLD_K, kfold_repeats=[5, KFOLD_REPEATS],
                distill_fit=split.distill_fit, distill_select=split.distill_select, seed=config.SEED)
            results[law] = res
            ceiling = res["ceilings_by_repeats"][KFOLD_REPEATS]
            print(f"  참조 천장(repeats={KFOLD_REPEATS}): "
                  f"median={ceiling.get('r2_median'):.4f} "
                  f"CI=[{ceiling.get('r2_ci_low'):.4f}, {ceiling.get('r2_ci_high'):.4f}]"
                  if ceiling.get("ok") else f"  참조 천장 계산 실패: {ceiling.get('reason')}")
            print(f"  자격(eligible): {res['eligible']}")
            print(f"  score_main={res['score_main_on_reference']} "
                  f"score_no_distillation={res['score_no_distillation_on_reference']} "
                  f"score_baseline={res['score_baseline_on_reference']}")
            print(f"  recovered={res['recovered']} exceeds_ceiling={res['exceeds_ceiling']} "
                  f"no_added_value={res['no_added_value']} teacher_harmed={res['teacher_harmed']}")
        except Exception as error:  # noqa: BLE001 — 배관 확인이므로 한 법칙 실패해도 나머지는 계속
            results[law] = {"law": law, "error": f"{type(error).__name__}: {error}"}
            print(f"  !!! 실패: {type(error).__name__}: {error}")
        print(f"  소요: {time.time() - t_law:.1f}s\n", flush=True)

    total_s = time.time() - t_start
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "date": DATE,
        "scale": {"n_ref_per_stratum": N_REF_PER_STRATUM,
                  "n_distill_per_stratum": N_DISTILL_PER_STRATUM,
                  "max_manifold_samples": MAX_MANIFOLD_SAMPLES_CHECK,
                  "kfold_k": KFOLD_K, "kfold_repeats": KFOLD_REPEATS,
                  "teacher_epochs": TEACHER_EPOCHS_CHECK, "sr_backend": "naive(임시)"},
        "split": {"reference": list(split.reference), "distill_fit": list(split.distill_fit),
                  "distill_select": list(split.distill_select),
                  "per_stratum_counts": split.per_stratum_counts},
        "disjoint_check": {"reference_x_fit": len(r & f), "reference_x_select": len(r & s),
                           "fit_x_select": len(f & s)},
        "results": results, "total_seconds": total_s,
    }
    out_path = RESULTS_DIR / "g2_pipeline_check.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"=== 완료. 총 소요 {total_s:.1f}s. 결과: {out_path} ===")


if __name__ == "__main__":
    main()
