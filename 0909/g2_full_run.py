#!/usr/bin/env python3
"""G2 전체 실행. `PREREG-G2.md` 최종본(Ruling R30·R31 반영)의 구현.
Ruling R32(코디네이터 승인)로 실행한다.

**절대 하지 않는 것(§7, 사전등록 그대로):**
  - 0902 수정 (읽기 전용 재사용만)
  - 봉인 홀드아웃(20260317 이후) 열기
  - 밴드를 증류 실행 후 조정
  - 결과가 나쁘면 설정을 바꿔 재실행

**단계 (순서가 설계의 일부다 — 밴드를 증류 결과보다 먼저 확정한다):**

  0. 3분할 조립(참조 20/증류 24 층당) + disjoint assert
  1. 참조 천장(밴드) 계산 — 5법칙, 회귀만(교사·SR 없음). 여기서 확정한
     CI 를 이후 그 무엇도 손대지 않는다.
  2. 선결 조건 확인 — L1 자격+양성(§4.1), 다섯 법칙 전부 음성 대조군
     (within_symbol/global × 5회). **L1 쪽(자격/양성/음성) 중 하나라도
     깨지면 여기서 멈춘다 — 3단계(비싼 증류)로 넘어가지 않는다.** L1 이
     아닌 법칙의 음성 대조군이 깨지면 그 법칙만 "신뢰 불가"로 표시하고
     계속한다(§4.1 해석 문단 — L4 때의 선례와 같다).
  3. 증류 — `sd.e0.runner.run_law`, 실제 PySR. 법칙마다 끝나는 즉시
     JSON 으로 저장한다(중간 산출 보존 — 프로세스가 죽어도 완료된 법칙은
     남는다).
  4. 재채점 — R 에서 `score_main`/`score_no_distillation`/기준선,
     구조진단(§3.6, 비판정).
  5. 판정 — §4.1~4.4 규칙을 기계적으로 적용해 최종 게이트를 만든다.

각 단계는 `results/g2_full_run/` 아래에 자신의 산출물을 즉시 쓴다 —
이 스크립트가 중간에 죽어도 이미 쓰인 파일은 유효한 부분 결과다.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

import sympy                                          # noqa: E402

from sd import config, universe, manifold            # noqa: E402
from sd.e0 import targets, criteria, runner as runner_mod  # noqa: E402
from sd.sr.base import Candidate, score_candidate     # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
OUT_DIR = REPO_0909 / "results" / "g2_full_run"

import g2_split       # noqa: E402
import g2_reference   # noqa: E402

DATE = config.DATE
assert DATE == "20260316", f"허용되지 않은 날짜: {DATE}"

LAWS = ("L1", "L2", "L3", "L4", "L5")
VETO_LAW = "L1"                       # 유일한 양성 대조군 (Ruling R30)
GRADED_LAWS = ("L2", "L3", "L4", "L5")  # E'/K' 분모 후보 (Ruling R30 — L5 포함)

# --- 전체 실행 확정 설정 (PREREG-G2.md §1.3·§3.1, R29/R30/R31 로 검증됨) ----
N_REF_PER_STRATUM = 20
N_DISTILL_PER_STRATUM = 24
MAX_MANIFOLD_SAMPLES_REFERENCE = 20000   # 참조천장·재채점 전용 (증류측과 분리, R29)
MAX_MANIFOLD_SAMPLES_DISTILL = 4000      # sd.e0.runner 기본값 그대로 (SR 적합 예산)
KFOLD_K = 5
KFOLD_REPEATS = 20
N_PERMUTATIONS = 5
SR_NITERATIONS = 25
SR_MAXSIZE = 20
SEED = config.SEED

NO_ADDED_VALUE_NONE = None  # placeholder for clarity in dict literals


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ===========================================================================
# 단계 0 — 3분할
# ===========================================================================

def phase0_split() -> g2_split.ThreeWaySplit:
    _log("=== 단계 0: 3분할 조립 ===")
    all_symbols = universe.stock_symbols(DATE)
    split = g2_split.build_three_way_split(
        all_symbols, DATE, n_ref_per_stratum=N_REF_PER_STRATUM,
        n_distill_per_stratum=N_DISTILL_PER_STRATUM)
    r, f, s = set(split.reference), set(split.distill_fit), set(split.distill_select)
    assert not (r & f) and not (r & s) and not (f & s), "서로소 위반 — 실행 중단"
    _log(f"참조 {len(split.reference)} / 증류-적합 {len(split.distill_fit)} / "
         f"증류-선택 {len(split.distill_select)} — 서로소 확인 통과")
    _write_json(OUT_DIR / "00_split.json", {
        "reference": list(split.reference), "distill_fit": list(split.distill_fit),
        "distill_select": list(split.distill_select),
        "per_stratum_counts": split.per_stratum_counts,
        "disjoint_check": {"ref_x_fit": len(r & f), "ref_x_select": len(r & s), "fit_x_select": len(f & s)},
    })
    return split


# ===========================================================================
# 단계 1 — 참조 천장(밴드). 증류 실행 전 확정. 이후 아무도 안 건드린다.
# ===========================================================================

def _reference_arrays(law: str, ref_symbols: tuple):
    ref_ds = targets.assemble(law, ref_symbols, DATE)
    selection = manifold.select(ref_ds.X_dimless, ref_ds.mask,
                                max_samples=MAX_MANIFOLD_SAMPLES_REFERENCE, seed=SEED)
    on_rows = selection.index[selection.on_manifold]
    return {"names": ref_ds.names_dimless, "X": ref_ds.X_dimless[on_rows],
            "y": ref_ds.y_dimless[on_rows], "sid": ref_ds.symbol_ids[on_rows],
            "w": selection.weight[selection.on_manifold],
            "n_symbols_used": len(ref_ds.symbols_used), "n_rows_total": ref_ds.n_rows_total}


def phase1_reference_ceilings(split: g2_split.ThreeWaySplit) -> dict:
    _log("=== 단계 1: 참조 천장(밴드) — 증류 실행 전 확정 ===")
    ceilings = {}
    ref_arrays = {}
    for law in LAWS:
        t0 = time.time()
        arrs = _reference_arrays(law, split.reference)
        ref_arrays[law] = arrs
        mask = np.ones(len(arrs["y"]), dtype=bool)
        ceiling = g2_reference.repeated_kfold_ceiling(
            law, arrs["X"], arrs["y"], mask, arrs["sid"], arrs["w"],
            k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
        eligible = bool(ceiling.get("ok") and ceiling["r2_ci_low"] > 0.0)
        ceilings[law] = {"ceiling": ceiling, "eligible": eligible,
                         "n_symbols_used": arrs["n_symbols_used"], "n_rows_total": arrs["n_rows_total"],
                         "n_on_manifold": int(len(arrs["y"]))}
        _log(f"  {law}: median={ceiling.get('r2_median')} "
             f"CI=[{ceiling.get('r2_ci_low')},{ceiling.get('r2_ci_high')}] "
             f"eligible={eligible} ({time.time()-t0:.1f}s)")
    _write_json(OUT_DIR / "01_reference_ceilings.json", ceilings)
    _log("밴드 확정 — 이 시점 이후 어떤 이유로도 수정하지 않는다.")
    return {"ceilings": ceilings, "ref_arrays": ref_arrays}


# ===========================================================================
# 단계 2 — 선결 조건(양성+음성 대조군). L1 쪽이 깨지면 여기서 멈춘다.
# ===========================================================================

def _negative_control_for_law(law: str, arrs: dict) -> dict:
    out = {}
    for scope in ("within_symbol", "global"):
        results = []
        for perm_i in range(N_PERMUTATIONS):
            y_perm = g2_reference.permute_y(arrs["y"], arrs["sid"], scope=scope, seed=1000 * perm_i + 7)
            mask = np.ones(len(y_perm), dtype=bool)
            ceiling = g2_reference.repeated_kfold_ceiling(
                law, arrs["X"], y_perm, mask, arrs["sid"], arrs["w"],
                k=KFOLD_K, repeats=KFOLD_REPEATS, seed=SEED)
            eligible = bool(ceiling.get("ok") and ceiling["r2_ci_low"] > 0.0)
            results.append({"perm_index": perm_i, "eligible": eligible,
                            "r2_median": ceiling.get("r2_median"), "r2_ci_low": ceiling.get("r2_ci_low"),
                            "r2_ci_high": ceiling.get("r2_ci_high")})
        out[scope] = results
    return out


def phase2_preconditions(phase1: dict) -> dict:
    _log("=== 단계 2: 선결 조건(양성+음성 대조군) ===")
    ref_arrays = phase1["ref_arrays"]
    ceilings = phase1["ceilings"]

    neg_controls = {}
    for law in LAWS:
        t0 = time.time()
        neg_controls[law] = _negative_control_for_law(law, ref_arrays[law])
        n_within = sum(r["eligible"] for r in neg_controls[law]["within_symbol"])
        n_global = sum(r["eligible"] for r in neg_controls[law]["global"])
        _log(f"  {law} 음성대조군: within_symbol {n_within}/{N_PERMUTATIONS}, "
             f"global {n_global}/{N_PERMUTATIONS} "
             f"({time.time()-t0:.1f}s)")

    l1_eligible = ceilings[VETO_LAW]["eligible"]
    l1_neg_within_clean = sum(r["eligible"] for r in neg_controls[VETO_LAW]["within_symbol"]) == 0
    l1_precondition_ok = bool(l1_eligible and l1_neg_within_clean)

    untrustworthy_laws = [law for law in LAWS
                          if sum(r["eligible"] for r in neg_controls[law]["within_symbol"]) > 0]

    result = {
        "l1_eligible": l1_eligible, "l1_negative_control_clean": l1_neg_within_clean,
        "l1_precondition_ok": l1_precondition_ok,
        "untrustworthy_laws_within_symbol": untrustworthy_laws,
        "negative_controls": neg_controls,
        "halt": not l1_precondition_ok,
    }
    _write_json(OUT_DIR / "02_preconditions.json", result)

    if not l1_precondition_ok:
        _log("!!! L1 선결 조건 실패 — 증류(단계 3)로 진행하지 않고 여기서 멈춘다.")
    elif untrustworthy_laws:
        _log(f"주의: {untrustworthy_laws} 는 음성 대조군에서 문제 발견 — "
             f"이 법칙(들)은 뒤에서 '신뢰 불가'로 표시하고 E'/K' 에서 제외한다.")
    else:
        _log("선결 조건 통과 — 다섯 법칙 전부 음성 대조군 정상, L1 양성 정상.")
    return result


# ===========================================================================
# 단계 3 — 증류(비싼 부분, 실제 PySR). 법칙마다 즉시 저장한다.
# ===========================================================================

def phase3_distillation(split: g2_split.ThreeWaySplit) -> dict:
    _log("=== 단계 3: 증류(PySR) — 법칙마다 즉시 저장 ===")
    distilled = {}
    for law in LAWS:
        out_path = OUT_DIR / f"03_distill_{law}.json"
        if out_path.exists():
            _log(f"  {law}: 이미 완료된 산출물 발견 — 재사용(재실행 안 함)")
            distilled[law] = json.loads(out_path.read_text())
            continue
        t0 = time.time()
        _log(f"  {law}: run_law 시작 (niterations={SR_NITERATIONS}, maxsize={SR_MAXSIZE}, "
             f"max_manifold_samples={MAX_MANIFOLD_SAMPLES_DISTILL})")
        try:
            result = runner_mod.run_law(
                law, split.distill_fit, split.distill_select, DATE, seed=SEED,
                sr_niterations=SR_NITERATIONS, sr_maxsize=SR_MAXSIZE,
                max_manifold_samples=MAX_MANIFOLD_SAMPLES_DISTILL)
        except Exception as error:  # noqa: BLE001 — 한 법칙 실패가 전체를 죽이면 안 된다
            elapsed = time.time() - t0
            _log(f"  !!! {law} 증류 실패: {type(error).__name__}: {error} ({elapsed:.1f}s)")
            payload = {"law": law, "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc(), "elapsed_seconds": elapsed}
            _write_json(out_path, payload)
            distilled[law] = payload
            continue
        elapsed = time.time() - t0

        main_track = result.tracks.get("main")
        nodist_track = result.tracks.get("no_distillation")

        def _track_payload(track):
            if track is None or track.primary_index is None:
                return None
            cand = track.candidates[track.primary_index]
            return {"expr": str(cand.expr), "complexity": cand.complexity,
                    "primary_index": track.primary_index,
                    "out_of_sample_score_on_select": track.out_of_sample_scores[track.primary_index],
                    "n_candidates": len(track.candidates), "fit_seconds": track.fit_seconds,
                    "n_select_rows": track.n_select_rows}

        payload = {
            # names(feature 이름)는 여기 안 담는다 — 4단계가 targets.assemble 로
            # 같은 법칙에 대해 다시 얻는다(law 마다 고정이므로 항상 같은 값).
            "law": law, "elapsed_seconds": elapsed,
            "main": _track_payload(main_track), "no_distillation": _track_payload(nodist_track),
            "teacher_dimless_diag": result.teacher_dimless_diag,
            "symbols_used": list(result.symbols_used), "select_symbols_used": list(result.select_symbols_used),
        }
        _write_json(out_path, payload)
        distilled[law] = payload
        _log(f"  {law}: 완료 ({elapsed:.1f}s) — main_expr={payload['main']['expr'] if payload['main'] else None}")
    return distilled


# ===========================================================================
# 단계 4 — 참조 집합에서 재채점 + 구조진단(비판정)
# ===========================================================================

def phase4_rescore(split: g2_split.ThreeWaySplit, phase1: dict, distilled: dict) -> dict:
    _log("=== 단계 4: 참조 집합 재채점 ===")
    ref_arrays = phase1["ref_arrays"]
    rescored = {}
    for law in LAWS:
        t0 = time.time()
        d = distilled[law]
        if "error" in d:
            rescored[law] = {"skipped_reason": f"증류 실패: {d['error']}"}
            _log(f"  {law}: 증류 실패로 재채점 건너뜀")
            continue

        arrs = ref_arrays[law]
        names = arrs["names"]
        X_R, y_R, w_R, sid_R = arrs["X"], arrs["y"], arrs["w"], arrs["sid"]

        # main/no_distillation 후보를 다시 sympy 로 파싱해 R 에서 채점한다.
        def _score(track_payload):
            if track_payload is None:
                return None, None, None
            expr = sympy.sympify(track_payload["expr"])
            cand = Candidate(expr=expr, complexity=track_payload["complexity"],
                             in_sample_score=0.0, backend="pysr", seed=SEED)
            s_binned = g2_reference.score_candidate_binned(cand, names, X_R, y_R, w_R)
            s_row = score_candidate(cand, names, X_R, y_R, w_R)
            diag = None
            try:
                diag = runner_mod._judge(law, expr, names, X_R)
            except Exception as error:  # noqa: BLE001 — 진단 실패는 비치명적
                diag = {"error": f"{type(error).__name__}: {error}"}
            return ((float(s_binned) if np.isfinite(s_binned) else None),
                    (float(s_row) if np.isfinite(s_row) else None), diag)

        score_main, score_main_row, diag_main = _score(d["main"])
        score_nodist, score_nodist_row, diag_nodist = _score(d["no_distillation"])

        # 단순회귀 기준선: D_fit∪D_select 전체에 적합, R 에서 채점
        fit_ds = targets.assemble(law, split.distill_fit, DATE)
        select_ds = targets.assemble(law, split.distill_select, DATE)
        X_D = np.concatenate([fit_ds.X_dimless, select_ds.X_dimless], axis=0)
        y_D = np.concatenate([fit_ds.y_dimless, select_ds.y_dimless], axis=0)
        mask_D = np.concatenate([fit_ds.mask, select_ds.mask], axis=0)
        sid_D = np.concatenate([fit_ds.symbol_ids, select_ds.symbol_ids], axis=0)
        pred_baseline = g2_reference.fit_on_all_and_predict(
            law, X_D, y_D, mask_D, X_R, symbol_ids_fit=sid_D, symbol_ids_score=sid_R)
        score_baseline = None
        if pred_baseline is not None:
            r2b = g2_reference.binned_r2(y_R, pred_baseline, w_R)
            score_baseline = float(r2b) if np.isfinite(r2b) else None

        rescored[law] = {
            "score_main_on_reference": score_main, "score_main_row_diagnostic": score_main_row,
            "score_no_distillation_on_reference": score_nodist, "score_no_distillation_row_diagnostic": score_nodist_row,
            "score_baseline_on_reference": score_baseline,
            "structural_diagnostic_main": diag_main, "structural_diagnostic_no_distillation": diag_nodist,
            "elapsed_seconds": time.time() - t0,
        }
        _log(f"  {law}: score_main={score_main} score_no_distillation={score_nodist} "
             f"score_baseline={score_baseline} ({time.time()-t0:.1f}s)")
    _write_json(OUT_DIR / "04_rescored.json", rescored)
    return rescored


# ===========================================================================
# 단계 5 — 판정 (§4.1~4.4)
# ===========================================================================

def phase5_verdict(phase1: dict, phase2: dict, rescored: dict) -> dict:
    _log("=== 단계 5: 판정 ===")
    ceilings = phase1["ceilings"]
    untrustworthy = set(phase2["untrustworthy_laws_within_symbol"])

    per_law = {}
    for law in GRADED_LAWS:
        c = ceilings[law]
        r = rescored.get(law, {})
        if law in untrustworthy:
            per_law[law] = {"status": "untrusted_negative_control_failed",
                            "reason": "이 법칙의 음성 대조군(within_symbol)이 허위양성을 냈다 — "
                                     "참조측 코드 재검토 전까지 이 법칙의 어떤 판정도 신뢰하지 않는다."}
            continue
        if "skipped_reason" in r:
            per_law[law] = {"status": "distillation_failed", "reason": r["skipped_reason"]}
            continue
        eligible = c["eligible"]
        if not eligible:
            per_law[law] = {"status": "ineligible",
                            "ceiling": c["ceiling"], "reason": "자격 검사(§3.3) 미달 — CI 하한 <= 0"}
            continue
        score_main = r.get("score_main_on_reference")
        ci_low = c["ceiling"]["r2_ci_low"]
        ci_high = c["ceiling"]["r2_ci_high"]
        recovered = bool(score_main is not None and score_main >= ci_low)
        exceeds_ceiling = bool(score_main is not None and score_main > ci_high)
        score_baseline = r.get("score_baseline_on_reference")
        no_added_value = bool(score_baseline is not None and score_main is not None
                              and score_baseline > score_main)
        score_nodist = r.get("score_no_distillation_on_reference")
        teacher_harmed = bool(score_main is not None and score_nodist is not None
                              and score_nodist > score_main + 0.05)
        per_law[law] = {
            "status": "graded", "eligible": True, "recovered": recovered,
            "exceeds_ceiling": exceeds_ceiling, "no_added_value": no_added_value,
            "teacher_harmed": teacher_harmed, "score_main": score_main,
            "score_no_distillation": score_nodist, "score_baseline": score_baseline,
            "ceiling_ci_low": ci_low, "ceiling_ci_high": ci_high,
        }

    graded_eligible = {law: v for law, v in per_law.items()
                       if v.get("status") == "graded" and v.get("eligible")}
    E_prime = len(graded_eligible)
    K_prime = sum(1 for v in graded_eligible.values() if v.get("recovered"))

    l1_main_recovered = None
    l1_rescore = rescored.get(VETO_LAW, {})
    if "score_main_on_reference" in l1_rescore:
        sm = l1_rescore["score_main_on_reference"]
        cl = ceilings[VETO_LAW]["ceiling"]["r2_ci_low"]
        l1_main_recovered = bool(sm is not None and sm >= cl)

    precondition_violated = (not phase2["l1_precondition_ok"]) or (l1_main_recovered is False)

    if precondition_violated:
        verdict = "STOP"
        verdict_reason = "선결 조건(§4.1) 위반 — L1 자격/양성/음성대조군 중 하나가 깨졌다."
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
        verdict_reason = f"자격 있는 {E_prime}개 중 {K_prime}개만 재현했다 — 법칙별 원인 조사 필요."

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "l1_positive_control": {
            "eligible": phase2["l1_eligible"], "negative_control_clean": phase2["l1_negative_control_clean"],
            "main_track_recovered": l1_main_recovered,
        },
        "untrustworthy_laws": list(untrustworthy),
        "per_law": per_law, "E_prime": E_prime, "K_prime": K_prime,
        "verdict": verdict, "verdict_reason": verdict_reason,
    }
    _write_json(OUT_DIR / "05_verdict.json", result)
    _log(f"최종 판정: {verdict} — {verdict_reason}")
    return result


# ===========================================================================
# 메인
# ===========================================================================

def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"G2 전체 실행 시작. PID={os.getpid()}")

    split = phase0_split()
    phase1 = phase1_reference_ceilings(split)
    phase2 = phase2_preconditions(phase1)

    if phase2["halt"]:
        _log("!!! L1 선결 조건 실패로 실행을 중단한다. 증류(3~5단계)는 돌리지 않는다.")
        _write_json(OUT_DIR / "HALTED.json", {
            "reason": "L1 선결 조건 실패", "phase2": phase2,
            "total_seconds": time.time() - t_start})
        _log(f"총 소요: {time.time() - t_start:.1f}s (중단됨)")
        return

    distilled = phase3_distillation(split)
    rescored = phase4_rescore(split, phase1, distilled)
    verdict = phase5_verdict(phase1, phase2, rescored)

    total_s = time.time() - t_start
    _write_json(OUT_DIR / "DONE.json", {"total_seconds": total_s, "verdict": verdict["verdict"]})
    _log(f"=== 전체 완료. 총 소요 {total_s:.1f}s ({total_s/60:.1f}분). 최종 판정: {verdict['verdict']} ===")


if __name__ == "__main__":
    main()
