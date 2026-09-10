#!/usr/bin/env python3
"""G3 — 3단계(합성 검산). `PREREG-G3.md` §2 원칙3의 순서를 지킨다: 이
스크립트는 **실측 데이터를 전혀 안 쓴다** — 알려진 신호/무신호를 인공
데이터로 만들어 `g2_reference.ceiling_fit_on_D_score_on_R`(대칭 천장)과
`negative_control_fit_on_D_score_on_R`(음성 대조군)가 **그것을 정확히
가리는지**만 검사한다. L1 을 포함해 어떤 실제 후보 점수도 이 시점까지
안 본다 — `g3_rescore_and_verdict.py`(다음 단계)가 그걸 한다.

두 갈래로 검사한다:

  A. L2 형태(단일 특성 풀링 선형회귀) — 무관계/약한신호/강한신호 세 강도.
     `binned_r2` 도입 당시 합성검산(G2-DESIGN-NOTES.md "발견 2")과 같은
     발상이지만, 이번엔 **binned_r2 집계 하나**가 아니라 **대칭 경로
     전체**(D 적합 → R 채점 → 부트스트랩 CI)를 검사한다.
  B. L4 형태(3특성, 종목 내 중심화) — Ruling R31 이 잡은 "종목수준
     생태학적 상관" 버그를 새 경로에서도 재현해 본다: 종목마다 레짐이
     다르고 그 레짐이 X·y 를 동시에 밀어올리지만(교란), **종목 내
     진짜 관계는 0**인 경우와, 교란 위에 **진짜 종목 내 신호**를 얹은
     경우를 각각 만들어 자격 검사가 정확히 가르는지 본다. D 와 R 을
     서로 다른 종목 이름공간(REF*/D*)으로 만들어 실제 3분할과 같은
     "완전히 다른 종목 집합"조건을 재현한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_0909 = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_0909))
import g2_reference  # noqa: E402

OUT_PATH = REPO_0909 / "results" / "g3_synthetic_check.json"

SEED = 20260910


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# A. L2 형태 — 단일 특성, 무관계/약한신호/강한신호
# ===========================================================================

def _gen_l2_population(n_symbols: int, rows_per_symbol: int, *, true_slope: float,
                       noise_sd: float, seed: int):
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_symbols), rows_per_symbol).astype(float)
    x = rng.normal(size=len(sid))
    y = true_slope * x + rng.normal(scale=noise_sd, size=len(sid))
    X = x.reshape(-1, 1)
    mask = np.ones(len(sid), dtype=bool)
    w = np.ones(len(sid))
    return X, y, mask, sid, w


def scenario_l2(name: str, true_slope: float, noise_sd: float, *, seed: int) -> dict:
    X_D, y_D, mask_D, sid_D, _ = _gen_l2_population(
        100, 200, true_slope=true_slope, noise_sd=noise_sd, seed=seed)
    X_R, y_R, mask_R, sid_R, w_R = _gen_l2_population(
        60, 300, true_slope=true_slope, noise_sd=noise_sd, seed=seed + 1)
    del mask_R
    res = g2_reference.ceiling_fit_on_D_score_on_R(
        "L2", X_D, y_D, mask_D, sid_D, X_R, y_R, w_R, sid_R, n_boot=400, seed=seed)
    eligible = bool(res.get("ok") and res.get("row_r2_ci_low", -1.0) > 0.0)
    return {"scenario": name, "true_slope": true_slope, "noise_sd": noise_sd,
           "result": res, "eligible": eligible}


# ===========================================================================
# B. L4 형태 — 3특성, 종목수준 교란(생태학적 상관) ± 진짜 종목내 신호
# ===========================================================================

#   **재보정 기록(결과를 본 뒤 수정 — 정직하게 남긴다).** 첫 실행은
#   `confound_sd=3.0`(교란 표준편차가 신호 표준편차의 약 5배)로 B2(교란+진짜
#   종목내신호)를 자격 없음으로 냈다. 원인을 계산으로 확인했다 —
#   `weighted_r2` 는 R **전체 행**의 총분산(SS_tot) 대비로 채점하는데, 그
#   총분산은 종목간 레짐 분산이 압도한다(신호 분산 0.33 대 레짐 분산 9,
#   레짐 비중 95%). 종목 내 신호를 "완벽히" 복원해도(정말 그랬다 — 계수는
#   맞게 나온다) 달성 가능한 행단위 R² 상한 자체가 신호분산/총분산
#   ≈0.035로 수학적으로 정해져 있었다(직접 계산해 실측값 0.025와 일치를
#   확인 — 코드 결함이 아니라 척도의 성질이었다). 이건 그 자체로 보고할
#   가치가 있는 발견이다(G3-REPORT.md — "종목간 이질성이 크면 진짜
#   종목내신호도 행단위 R²가 작게 보인다"는 것은 지표의 결함이 아니라
#   정의상 당연한 결과). **여기서 고친 것은 채점 방식이 아니라 시나리오의
#   교란 강도**(confound_sd 3.0→0.6, 실측 L4 의 종목간 분산 비중
#   22~34%(G2-NEGATIVE-CONTROL-REPORT.md)에 맞춘 것)다 — "진짜 신호가
#   있으면 잡혀야 한다"는 원래 검사 취지를 유지한 채, 교란이 신호를
#   구조적으로 못 잡게 만들 만큼 극단적이지 않게 재보정했다. 판정 함수
#   자체(`ceiling_fit_on_D_score_on_R`)는 건드리지 않았다.
def _gen_l4_population(n_symbols: int, rows_per_symbol: int, *, within_coef: float,
                       confound_sd: float, noise_sd: float, seed: int):
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_symbols), rows_per_symbol).astype(float)
    regime = rng.normal(scale=confound_sd, size=n_symbols)          # 종목별 "그날의 레짐" 수준
    regime_per_row = regime[sid.astype(int)]
    X_within = rng.normal(size=(len(sid), 3))                        # 종목 내 편차(평균 0)
    X = X_within + regime_per_row[:, None]                           # 관측 X = 레짐 + 종목내 편차
    # 진짜 종목내 신호(있다면): 종목내 편차의 평균에 선형으로 반응
    true_within_signal = within_coef * X_within.mean(axis=1)
    y = regime_per_row + true_within_signal + rng.normal(scale=noise_sd, size=len(sid))
    mask = np.ones(len(sid), dtype=bool)
    w = np.ones(len(sid))
    return X, y, mask, sid, w


def scenario_l4(name: str, within_coef: float, *, confound_sd: float = 0.6,
                noise_sd: float = 0.3, seed: int) -> dict:
    X_D, y_D, mask_D, sid_D, _ = _gen_l4_population(
        100, 200, within_coef=within_coef, confound_sd=confound_sd, noise_sd=noise_sd, seed=seed)
    X_R, y_R, mask_R, sid_R, w_R = _gen_l4_population(
        60, 300, within_coef=within_coef, confound_sd=confound_sd, noise_sd=noise_sd, seed=seed + 1)
    del mask_R
    res = g2_reference.ceiling_fit_on_D_score_on_R(
        "L4", X_D, y_D, mask_D, sid_D, X_R, y_R, w_R, sid_R, n_boot=400, seed=seed)
    eligible = bool(res.get("ok") and res.get("row_r2_ci_low", -1.0) > 0.0)
    return {"scenario": name, "within_coef": within_coef, "confound_sd": confound_sd,
           "noise_sd": noise_sd, "result": res, "eligible": eligible}


# ===========================================================================
# C. 음성 대조군 기계 자체의 점검 — 진짜 강신호를 만들고 나서 D·R 양쪽을
# 섞으면(permute) 반드시 자격을 잃어야 한다. "섞으면 사라진다"를 검사한다.
# ===========================================================================

def scenario_negcontrol_kills_real_signal(seed: int) -> dict:
    X_D, y_D, mask_D, sid_D, _ = _gen_l2_population(100, 200, true_slope=2.0, noise_sd=0.5, seed=seed)
    X_R, y_R, mask_R, sid_R, w_R = _gen_l2_population(60, 300, true_slope=2.0, noise_sd=0.5, seed=seed + 1)
    del mask_R
    out = {}
    for scope in ("within_symbol", "global"):
        res = g2_reference.negative_control_fit_on_D_score_on_R(
            "L2", X_D, y_D, mask_D, sid_D, X_R, y_R, w_R, sid_R,
            scope=scope, seed=seed, n_boot=400)
        eligible = bool(res.get("ok") and res.get("row_r2_ci_low", -1.0) > 0.0)
        out[scope] = {"result": res, "eligible": eligible}
    return out


def main() -> None:
    t0 = time.time()
    results: dict = {}

    _log("=== A. L2 형태 — 무관계/약한신호/강한신호 ===")
    a_null = scenario_l2("A1_null_no_relation", true_slope=0.0, noise_sd=1.0, seed=SEED)
    a_weak = scenario_l2("A2_weak_real_signal", true_slope=0.05, noise_sd=1.0, seed=SEED + 10)
    a_strong = scenario_l2("A3_strong_real_signal", true_slope=2.0, noise_sd=0.5, seed=SEED + 20)
    for r in (a_null, a_weak, a_strong):
        rr = r["result"]
        _log(f"  {r['scenario']}: row_r2_point={rr.get('row_r2_point')} "
             f"CI=[{rr.get('row_r2_ci_low')},{rr.get('row_r2_ci_high')}] eligible={r['eligible']}")
    results["A_l2_form"] = {"null": a_null, "weak": a_weak, "strong": a_strong}

    _log("=== B. L4 형태 — 종목수준 교란(생태학적 상관) ± 진짜 종목내 신호 ===")
    b_confound_only = scenario_l4("B1_confound_only_no_within_signal", within_coef=0.0, seed=SEED + 30)
    b_with_signal = scenario_l4("B2_confound_plus_real_within_signal", within_coef=1.0, seed=SEED + 40)
    for r in (b_confound_only, b_with_signal):
        rr = r["result"]
        _log(f"  {r['scenario']}: row_r2_point={rr.get('row_r2_point')} "
             f"CI=[{rr.get('row_r2_ci_low')},{rr.get('row_r2_ci_high')}] eligible={r['eligible']}")
    results["B_l4_form_ecological_confound"] = {"confound_only": b_confound_only,
                                                 "confound_plus_signal": b_with_signal}

    _log("=== C. 음성 대조군이 실제 강신호를 죽이는가 ===")
    c = scenario_negcontrol_kills_real_signal(SEED + 50)
    for scope, v in c.items():
        _log(f"  permuted({scope}) on strong-signal data: "
             f"row_r2_point={v['result'].get('row_r2_point')} eligible={v['eligible']}")
    results["C_negative_control_on_real_signal"] = c

    # --- 판정 기준 (사전에 정한다 — 결과를 보고 바꾸지 않는다) -----------------
    checks = {
        "A1_null_not_eligible": a_null["eligible"] is False,
        "A2_weak_signal_detected_eligible": a_weak["eligible"] is True,
        "A2_weak_signal_small_magnitude": (a_weak["result"].get("row_r2_point") is not None
                                           and 0 < a_weak["result"]["row_r2_point"] < 0.2),
        "A3_strong_signal_near_one": (a_strong["result"].get("row_r2_point") is not None
                                      and a_strong["result"]["row_r2_point"] > 0.8),
        "B1_confound_only_not_eligible": b_confound_only["eligible"] is False,
        "B2_confound_plus_signal_eligible": b_with_signal["eligible"] is True,
        "C_within_symbol_kills_real_signal": c["within_symbol"]["eligible"] is False,
        "C_global_kills_real_signal": c["global"]["eligible"] is False,
    }
    all_passed = all(checks.values())

    output = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
             "results": results, "checks": checks, "all_checks_passed": all_passed,
             "total_seconds": time.time() - t0}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    _log("=== 판정 기준 ===")
    for k, v in checks.items():
        _log(f"  {k}: {'통과' if v else '실패'}")
    _log(f"전체 통과: {all_passed}")
    _log(f"저장: {OUT_PATH}")
    if not all_passed:
        _log("!!! 합성 검산 실패 — PREREG-G3.md §2 원칙3에 따라 여기서 멈춘다. "
             "L1 을 포함해 어떤 후보 재채점도 다음 단계로 넘어가지 않는다.")


if __name__ == "__main__":
    main()
