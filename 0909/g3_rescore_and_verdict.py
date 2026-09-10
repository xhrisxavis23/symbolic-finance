#!/usr/bin/env python3
"""G3 — 4단계(재채점) + 5단계(판정). `PREREG-G3.md` §2 원칙3의 마지막
단계다 — **여기서 처음으로 L1 이 통과하는지 본다.** 그 전에
`g3_reference_and_controls.py`(대칭 천장·음성 대조군)와
`g3_synthetic_check.py`(합성 검산)가 **둘 다 끝나고 통과했는지 확인한
뒤에만** 이 스크립트를 돌린다 — 이 스크립트 자신도 그 순서를 강제한다
(아래 `_enforce_order`).

이미 있는 증류 결과(`results/g2_full_run/03_distill_L*.json`, PySR 3.7시간
포함)를 **다시 돌리지 않고 그대로 읽는다** — 바뀐 것은 채점 방식(원칙1·2)
뿐이므로 후보 자체를 다시 만들 이유가 없다. 참조 집합에서 재채점만
새로 한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd.e0 import runner as runner_mod        # noqa: E402
from sd.sr.base import Candidate, score_candidate, weighted_r2  # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
OLD_OUT_DIR = REPO_0909 / "results" / "g2_full_run"
OUT_DIR = REPO_0909 / "results" / "g3_symmetric_run"

import g2_reference               # noqa: E402
from g3_reference_and_controls import (   # noqa: E402
    _reference_arrays, _distill_pool_arrays, _load_split, SEED, LAWS, VETO_LAW,
)

GRADED_LAWS = ("L2", "L3", "L4", "L5")
TEACHER_HARM_MARGIN = 0.05   # G2 원안 그대로 유지 — 진단 플래그 전용(판정에 안 씀), 결과 보고 조정 안 함


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _enforce_order() -> None:
    """PREREG-G3.md §2 원칙3 순서 강제 — 음성 대조군·합성 검산이 없거나
    실패했으면 여기서 죽는다. 결과가 나쁘다고 이 함수를 우회하지 않는다."""
    controls_path = OUT_DIR / "DONE_controls.json"
    synth_path = REPO_0909 / "results" / "g3_synthetic_check.json"
    if not controls_path.exists():
        raise SystemExit("g3_reference_and_controls.py 를 먼저 돌려야 한다 (원칙3 순서).")
    if not synth_path.exists():
        raise SystemExit("g3_synthetic_check.py 를 먼저 돌려야 한다 (원칙3 순서).")
    controls = json.loads(controls_path.read_text())
    synth = json.loads(synth_path.read_text())
    if not controls.get("l1_within_symbol_clean"):
        _log("!!! 경고: L1 음성 대조군이 깨끗하지 않다 — 판정은 계산하되 STOP 으로 귀결될 것이다.")
    if not synth.get("all_checks_passed"):
        raise SystemExit("합성 검산이 실패했다 — PREREG-G3.md §2 원칙3에 따라 재채점으로 진행하지 않는다.")
    _log(f"순서 확인: 음성대조군 완료(L1 정상={controls.get('l1_within_symbol_clean')}), "
         f"합성검산 전체통과={synth.get('all_checks_passed')} — 재채점으로 진행한다.")


def _load_distilled() -> dict:
    distilled = {}
    for law in LAWS:
        p = OLD_OUT_DIR / f"03_distill_{law}.json"
        distilled[law] = json.loads(p.read_text())
    return distilled


def _score_track(track_payload, names, X_R, y_R, w_R):
    if track_payload is None:
        return None
    expr = sympy.sympify(track_payload["expr"])
    cand = Candidate(expr=expr, complexity=track_payload["complexity"],
                     in_sample_score=0.0, backend="pysr", seed=SEED)
    s_row = score_candidate(cand, names, X_R, y_R, w_R)              # 1차(원칙2)
    s_binned = g2_reference.score_candidate_binned(cand, names, X_R, y_R, w_R)  # 보조
    row_v = float(s_row) if np.isfinite(s_row) else None
    binned_v = float(s_binned) if np.isfinite(s_binned) else None
    amp = (binned_v / row_v) if (row_v is not None and row_v > 1e-12 and binned_v is not None) else None
    diag = None
    try:
        diag = runner_mod._judge(track_payload.get("law", "?") or "", expr, names, X_R)
    except Exception as error:  # noqa: BLE001
        diag = {"error": f"{type(error).__name__}: {error}"}
    return {"score_row": row_v, "score_binned": binned_v, "amplification_factor": amp,
           "structural_diagnostic": diag, "expr": track_payload["expr"]}


def phase4_rescore(split, distilled: dict) -> dict:
    _log("=== G3 단계 4: 참조 집합 재채점 (1차 지표 = 행단위 R², 원칙2) ===")
    rescored = {}
    for law in LAWS:
        t0 = time.time()
        d = distilled[law]
        if "error" in d:
            rescored[law] = {"skipped_reason": f"증류 실패: {d['error']}"}
            _log(f"  {law}: 증류 실패로 재채점 건너뜀")
            continue

        arrs = _reference_arrays(law, split.reference)
        darrs = _distill_pool_arrays(law, split.distill_fit, split.distill_select)
        names = arrs["names"]
        X_R, y_R, w_R = arrs["X"], arrs["y"], arrs["w"]

        main_res = _score_track({**d["main"], "law": law} if d.get("main") else None, names, X_R, y_R, w_R)
        nodist_res = _score_track({**d["no_distillation"], "law": law} if d.get("no_distillation") else None,
                                  names, X_R, y_R, w_R)

        pred_baseline = g2_reference.fit_on_all_and_predict(
            law, darrs["X"], darrs["y"], darrs["mask"], X_R,
            symbol_ids_fit=darrs["sid"], symbol_ids_score=arrs["sid"])
        score_baseline_row = score_baseline_binned = baseline_amp = None
        if pred_baseline is not None:
            r2row = weighted_r2(y_R, pred_baseline, w_R)
            r2bin = g2_reference.binned_r2(y_R, pred_baseline, w_R)
            score_baseline_row = float(r2row) if np.isfinite(r2row) else None
            score_baseline_binned = float(r2bin) if np.isfinite(r2bin) else None
            if score_baseline_row is not None and score_baseline_row > 1e-12 and score_baseline_binned is not None:
                baseline_amp = score_baseline_binned / score_baseline_row

        rescored[law] = {
            "main": main_res, "no_distillation": nodist_res,
            "baseline": {"score_row": score_baseline_row, "score_binned": score_baseline_binned,
                        "amplification_factor": baseline_amp},
            "elapsed_seconds": time.time() - t0,
        }
        _log(f"  {law}: score_main_row={main_res['score_row'] if main_res else None} "
             f"score_nodist_row={nodist_res['score_row'] if nodist_res else None} "
             f"score_baseline_row={score_baseline_row} ({time.time()-t0:.1f}s)")
    _write_json(OUT_DIR / "03_rescored.json", rescored)
    return rescored


def phase5_verdict(ceilings: dict, neg_control: dict, rescored: dict) -> dict:
    _log("=== G3 단계 5: 판정 ===")
    untrustworthy = set(neg_control["untrustworthy_laws_within_symbol"])

    per_law = {}
    for law in GRADED_LAWS:
        c = ceilings[law]
        r = rescored.get(law, {})
        if law in untrustworthy:
            per_law[law] = {"status": "untrusted_negative_control_failed",
                            "reason": "이 법칙의 음성 대조군(within_symbol)이 허위양성을 냈다."}
            continue
        if "skipped_reason" in r:
            per_law[law] = {"status": "distillation_failed", "reason": r["skipped_reason"]}
            continue
        eligible = c["eligible"]
        if not eligible:
            per_law[law] = {"status": "ineligible", "ceiling": c["ceiling"],
                            "reason": "자격 검사(행단위 R² CI 하한 <= 0) 미달"}
            continue
        main = r.get("main")
        score_main = main["score_row"] if main else None
        ci_low = c["ceiling"]["row_r2_ci_low"]
        ci_high = c["ceiling"]["row_r2_ci_high"]
        recovered = bool(score_main is not None and score_main >= ci_low)
        exceeds_ceiling = bool(score_main is not None and score_main > ci_high)
        baseline = r.get("baseline", {})
        score_baseline = baseline.get("score_row")
        no_added_value = bool(score_baseline is not None and score_main is not None
                              and score_baseline > score_main)
        nodist = r.get("no_distillation")
        score_nodist = nodist["score_row"] if nodist else None
        teacher_harmed = bool(score_main is not None and score_nodist is not None
                              and score_nodist > score_main + TEACHER_HARM_MARGIN)
        per_law[law] = {
            "status": "graded", "eligible": True, "recovered": recovered,
            "exceeds_ceiling": exceeds_ceiling, "no_added_value": no_added_value,
            "teacher_harmed": teacher_harmed, "score_main_row": score_main,
            "score_main_binned": main["score_binned"] if main else None,
            "score_main_amplification": main["amplification_factor"] if main else None,
            "score_no_distillation_row": score_nodist, "score_baseline_row": score_baseline,
            "ceiling_row_r2_ci_low": ci_low, "ceiling_row_r2_ci_high": ci_high,
            "ceiling_row_r2_point": c["ceiling"]["row_r2_point"],
            "ceiling_binned_r2_point": c["ceiling"]["binned_r2_point"],
            "ceiling_amplification_factor": c["ceiling"]["amplification_factor"],
        }

    graded_eligible = {law: v for law, v in per_law.items()
                       if v.get("status") == "graded" and v.get("eligible")}
    E_prime = len(graded_eligible)
    K_prime = sum(1 for v in graded_eligible.values() if v.get("recovered"))

    l1_ceiling = ceilings[VETO_LAW]
    l1_rescore = rescored.get(VETO_LAW, {})
    l1_main_recovered = None
    l1_main_score = None
    if l1_rescore.get("main"):
        l1_main_score = l1_rescore["main"]["score_row"]
        cl = l1_ceiling["ceiling"]["row_r2_ci_low"]
        l1_main_recovered = bool(l1_main_score is not None and l1_main_score >= cl)

    l1_eligible = l1_ceiling["eligible"]
    l1_neg_clean = neg_control["l1_within_symbol_clean"]
    l1_precondition_ok = bool(l1_eligible and l1_neg_clean)
    precondition_violated = (not l1_precondition_ok) or (l1_main_recovered is False)

    if precondition_violated:
        verdict = "STOP"
        verdict_reason = "선결 조건(§4.1) 위반 — L1 자격/음성대조군/주후보 복원 중 하나가 깨졌다."
    elif E_prime == 0:
        verdict = "INSUFFICIENT_TARGETS"
        verdict_reason = "L1 외엔 이 표본으로 잴 신호 자체가 없다(자격 있는 채점대상 법칙 0개)."
    elif K_prime == E_prime:
        verdict = "PROCEED"
        verdict_reason = f"자격 있는 {E_prime}개 법칙을 전부 재현했다."
    elif K_prime == 0:
        verdict = "STOP"
        verdict_reason = f"자격 있는 {E_prime}개 법칙 중 하나도 재현 못 했다."
    else:
        verdict = "HOLD"
        verdict_reason = f"자격 있는 {E_prime}개 중 {K_prime}개만 재현했다."

    # 증폭 배율 종합표(원칙2 요구 — 모든 법칙에 대해 보고)
    amplification_table = {}
    for law in LAWS:
        c = ceilings[law]["ceiling"]
        r = rescored.get(law, {})
        main = r.get("main") if isinstance(r, dict) else None
        baseline = r.get("baseline") if isinstance(r, dict) else None
        amplification_table[law] = {
            "ceiling_amplification": c.get("amplification_factor"),
            "candidate_main_amplification": main.get("amplification_factor") if main else None,
            "baseline_amplification": baseline.get("amplification_factor") if baseline else None,
        }

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "l1_positive_control": {"eligible": l1_eligible, "negative_control_clean": l1_neg_clean,
                                "main_track_score_row": l1_main_score,
                                "main_track_recovered": l1_main_recovered},
        "untrustworthy_laws": list(untrustworthy),
        "per_law": per_law, "E_prime": E_prime, "K_prime": K_prime,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "amplification_table": amplification_table,
    }
    _write_json(OUT_DIR / "04_verdict.json", result)
    _log(f"최종 판정: {verdict} — {verdict_reason}")
    return result


def main() -> None:
    _enforce_order()
    t_start = time.time()
    split = _load_split()
    ceilings = json.loads((OUT_DIR / "01_reference_ceilings.json").read_text())
    neg_control = json.loads((OUT_DIR / "02_negative_control.json").read_text())
    distilled = _load_distilled()

    rescored = phase4_rescore(split, distilled)
    verdict = phase5_verdict(ceilings, neg_control, rescored)

    total_s = time.time() - t_start
    _write_json(OUT_DIR / "DONE.json", {"total_seconds": total_s, "verdict": verdict["verdict"]})
    _log(f"=== 4·5단계 완료. 총 {total_s:.1f}s. 최종 판정: {verdict['verdict']} ===")


if __name__ == "__main__":
    main()
