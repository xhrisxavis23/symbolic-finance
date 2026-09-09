#!/usr/bin/env python3
"""L0 — 법칙 직접 측정. PREREG-L0.md 의 구현.

**증류도, 교사도, 심볼릭 회귀도 쓰지 않는다.** `0902/sd/*` 를 읽기 전용으로
재사용해 데이터를 조립하고(`sd.e0.targets`), 곧바로 회귀로 잰다.

두 가지 방식(PREREG §2):

  (a) 직접  — 전체 종목 풀에서, 층화 없이, `targets` 가 주는 `mask` 만 적용해
              회귀.
  (b) E0 표본 — `sd.universe`(층화) → `sd.e0.split`(적합/선택 분할) →
              `sd.manifold.select`(on-manifold 표집)까지 E0 와 똑같이 적용한
              뒤 같은 회귀.

밴드는 전부 `sd.e0.criteria` 에서 읽는다 — 이 파일에 숫자를 손으로 옮기지
않는다 (READ_BAND_NAMES 를 통해 프로그램으로 끌어온다).

L1·L5 는 판정 대상이 아니라 측정 자체의 대조군이다 (PREREG §4). E0 는
후보 심볼릭 표현식을 그리드에서 평가해 판정하지만(`criteria.judge_l*`),
여기서는 후보 표현식이 없다 — 대신 실측 데이터를 그대로 쓴다:

  - L1: book_imbalance 를 대칭 구간에서 40개 구간으로 나눠 구간별
    조건부평균 y(이것이 "직접 회귀로 잰 g(I)"다)를 만들고, 그 위에
    `criteria` 의 단조성(Spearman)·홀함수(대칭오차) 판정을 그대로 적용한다.
  - L5: dt_prev 를 로그 간격 50개 구간으로 나눠 구간별 조건부평균 y 를
    만들고, `criteria._fit_exponential`/`_fit_power_law`/`_linear_r2` 를
    그 위에 그대로 적용한다(두 번째 진실 원천을 만들지 않는다 — 피팅
    함수 자체를 재사용).

L2·L3·L4 는 PREREG §1 이 명시한 그대로: 각각 종목별 단순회귀의 CV,
로그-로그 회귀, 다중선형회귀다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 0902 를 읽기 전용으로 붙인다. 두 번째 진실 원천을 만들지 않는다.
# ---------------------------------------------------------------------------
REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd import config, universe, manifold, dimensionless  # noqa: E402  (dimensionless: 참고용, 아래 NOTE)
from sd.e0 import targets, split, criteria               # noqa: E402

# NOTE — sd.dimensionless 는 import 만 하고 쓰지 않는다: `sd/e0/targets.py` 가
# 만드는 X_dimless/y_dimless 는 `dimensionless.transform`(일반 SR 어휘용,
# `run_slice.py` 가 쓴다)을 타지 않고, 법칙별로 직접 비(比)를 구성한다
# (targets.py 모듈 docstring, "설계 원칙" 참고). 그래서 E0 의 법칙 게이트
# 파이프라인을 그대로 재현하는 이 스크립트는 이 모듈을 실제로 호출할 지점이
# 없다 — 확인차 import 만 해 둔다(요청된 재사용 목록에 있으므로).

REPO_0909 = Path(__file__).resolve().parent
RESULTS_DIR = REPO_0909 / "results"

# ---------------------------------------------------------------------------
# 측정 규모. "최소 수십 종목", "몇 분~수십 분" (지시사항).
# ---------------------------------------------------------------------------
DATE = config.DATE                 # "20260316" — 유일하게 허용된 날짜
SEED = config.SEED                 # 0 — sd.config 의 정본 시드
assert DATE == "20260316", f"허용되지 않은 날짜: {DATE}"

N_DIRECT_SYMBOLS = 200             # (a) 직접 풀 — 층화 없이 무작위 표본
PER_STRATUM_E0 = 15                # (b) E0 풀 — 6층 × 15 = 최대 90종목
MAX_MANIFOLD_SAMPLES = 4000        # sd.e0.runner.run_law 기본값과 동일

MIN_ROWS_PER_SYMBOL_L2 = 20        # 종목별 기울기 하나를 신뢰하기 위한 최소 행 수
MIN_ROWS_GENERIC = 50              # 풀링 회귀(L3·L4) 최소 표본
N_BINS_L1 = 40
MIN_BIN_COUNT_L1 = 5
N_BINS_L5 = 50
MIN_BIN_COUNT_L5 = 5

LAWS = ("L1", "L2", "L3", "L4", "L5")

# 밴드를 손으로 옮기지 않는다 — criteria.py 에서 프로그램으로 끌어온다.
BAND_CONSTANT_NAMES = (
    "AFFINE_CV_MAX", "L3_EXPONENT_RANGE", "L3_MIN_LOG_LOG_R2",
    "L4_MIN_GLOBAL_R2", "L4_MIN_ABS_SLOPE", "L4_QUASI_LINEAR_CV_MAX",
    "MONOTONIC_SPEARMAN_MIN", "ODD_RELATIVE_ERROR_MAX", "DEGENERATE_SCALE_MIN",
    "L5_MIN_R2", "L5_MIN_CURVATURE_MARGIN",
)


def read_bands() -> dict:
    return {name: getattr(criteria, name) for name in BAND_CONSTANT_NAMES}


# ===========================================================================
# 법칙별 직접-회귀 측정 함수. 전부 (X, y, mask, symbol_ids) 를 받는 공통 시그니처.
# ===========================================================================

def _finite_mask(*cols: np.ndarray) -> np.ndarray:
    out = np.isfinite(cols[0])
    for c in cols[1:]:
        out &= np.isfinite(c)
    return out


def measure_l1(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                imbalance_idx: int = 0, *, min_bin_count: int = MIN_BIN_COUNT_L1,
                n_bins: int = N_BINS_L1) -> dict:
    """book_imbalance 의 대칭 구간별 조건부평균 y → 단조성·홀함수(criteria 밴드)."""
    x = X[:, imbalance_idx]
    valid = mask & _finite_mask(x, y)
    x, y = x[valid], y[valid]
    n = int(valid.sum())
    if n < MIN_ROWS_GENERIC:
        return {"passed": False, "ambiguous": True, "reason": "표본 부족", "n": n}

    i_max = float(np.nanpercentile(np.abs(x), 95))
    if i_max <= 0:
        return {"passed": False, "ambiguous": True, "reason": "book_imbalance 범위가 0", "n": n}

    edges = np.linspace(-i_max, i_max, n_bins + 1)      # 대칭 — edges[::-1] == -edges
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0            # centers[::-1] == -centers

    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bin_idx == b
        counts[b] = int(sel.sum())
        if counts[b] >= min_bin_count:
            means[b] = float(np.mean(y[sel]))

    finite = np.isfinite(means)
    if finite.sum() < 0.8 * n_bins:
        return {"passed": False, "ambiguous": True, "reason": "구간 대부분 표본부족/비유한",
                "n": n, "n_bins_filled": int(finite.sum())}

    g = means
    scale = float(np.nanmax(np.abs(g[finite])))
    if scale < criteria.DEGENERATE_SCALE_MIN:
        return {"passed": False, "ambiguous": False, "reason": "상수로 퇴화", "n": n}

    rho, _p = stats.spearmanr(centers[finite], g[finite])
    mirror = g[::-1]
    both = finite & np.isfinite(mirror)
    denom = np.maximum(np.abs(g), np.abs(mirror))
    denom = np.where(denom < criteria.DEGENERATE_SCALE_MIN, 1.0, denom)
    odd_error = float(np.nanmedian(np.abs(g[both] + mirror[both]) / denom[both]))

    monotonic = bool(np.isfinite(rho) and abs(rho) >= criteria.MONOTONIC_SPEARMAN_MIN)
    odd = bool(odd_error <= criteria.ODD_RELATIVE_ERROR_MAX)
    return {"passed": bool(monotonic and odd), "ambiguous": False, "n": n,
            "n_bins_filled": int(finite.sum()), "spearman": float(rho) if np.isfinite(rho) else None,
            "monotonic": monotonic, "odd_relative_error": odd_error, "odd": odd,
            "grid_max_abs_I": i_max}


def measure_l2(X: np.ndarray, y: np.ndarray, mask: np.ndarray, symbol_ids: np.ndarray,
                ofi_idx: int = 0, *, min_rows: int = MIN_ROWS_PER_SYMBOL_L2) -> dict:
    """종목별 단순회귀 기울기 → 평균·표준편차·CV (PREREG §1 L2, criteria.AFFINE_CV_MAX)."""
    x = X[:, ofi_idx]
    valid = mask & _finite_mask(x, y)
    slopes: dict[int, float] = {}
    n_rows_by_symbol: dict[int, int] = {}
    for sid in np.unique(symbol_ids[valid]):
        sel = valid & (symbol_ids == sid)
        n_sid = int(sel.sum())
        if n_sid < min_rows:
            continue
        xs, ys = x[sel], y[sel]
        if np.std(xs) < 1e-12:
            continue
        slope, _intercept, _r, _p, _se = stats.linregress(xs, ys)
        if np.isfinite(slope):
            slopes[int(sid)] = float(slope)
            n_rows_by_symbol[int(sid)] = n_sid

    n_symbols = len(slopes)
    if n_symbols < 3:
        return {"passed": False, "ambiguous": True, "reason": "종목이 3개 미만",
                "n_symbols": n_symbols, "n_rows_valid": int(valid.sum())}

    values = np.array(list(slopes.values()), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))                      # criteria.approximately_affine 과 같은 정의(ddof=0)
    cv = float(abs(std / mean)) if abs(mean) > criteria.DEGENERATE_SCALE_MIN else float("inf")
    passed = bool(cv <= criteria.AFFINE_CV_MAX and mean > 0.0)
    return {"passed": passed, "ambiguous": False, "n_symbols": n_symbols,
            "n_rows_valid": int(valid.sum()), "slope_mean": mean, "slope_std": std,
            "slope_cv": cv, "positive": bool(mean > 0.0),
            "per_symbol_slopes": {str(k): v for k, v in sorted(slopes.items())},
            "per_symbol_n_rows": {str(k): v for k, v in sorted(n_rows_by_symbol.items())}}


def measure_l3(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                volume_idx: int = 0) -> dict:
    """|volume proxy| vs |y| 의 log-log 회귀 → 지수·R² (PREREG §1 L3)."""
    v = X[:, volume_idx]
    valid = mask & _finite_mask(v, y)
    vv, yy = v[valid], y[valid]
    x = np.abs(vv)
    ycol = np.abs(yy)
    keep = (x > 0) & (ycol > 0) & np.isfinite(x) & np.isfinite(ycol)
    x, ycol = x[keep], ycol[keep]
    n = int(keep.sum())
    if n < MIN_ROWS_GENERIC:
        return {"passed": False, "ambiguous": True, "reason": "유효 표본 부족(0 제외 후)",
                "n": n, "n_before_zero_filter": int(valid.sum())}

    log_x, log_y = np.log(x), np.log(ycol)
    slope, intercept, r_value, _p, _se = stats.linregress(log_x, log_y)
    r2 = float(r_value ** 2)
    lo, hi = criteria.L3_EXPONENT_RANGE
    ambiguous = bool(r2 < criteria.L3_MIN_LOG_LOG_R2)
    passed = bool(lo <= slope <= hi) and not ambiguous
    return {"passed": passed, "ambiguous": ambiguous, "n": n,
            "n_before_zero_filter": int(valid.sum()), "exponent": float(slope),
            "intercept": float(intercept), "log_log_r2": r2, "range": list(criteria.L3_EXPONENT_RANGE)}


def measure_l4(X3: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """세 스케일 다중선형회귀 → 계수 부호·크기, 전체 R² (PREREG §1 L4)."""
    valid = mask & np.all(np.isfinite(X3), axis=1) & np.isfinite(y)
    n = int(valid.sum())
    if n < MIN_ROWS_GENERIC:
        return {"passed": False, "ambiguous": True, "reason": "표본 부족", "n": n}

    Xv, yv = X3[valid], y[valid]
    design = np.column_stack([Xv, np.ones(len(yv))])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    prediction = design @ coef
    ss_res = float(np.sum((yv - prediction) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
    scale_coefs = coef[:-1]

    uses_all_three = bool(all(abs(c) > criteria.L4_MIN_ABS_SLOPE for c in scale_coefs))
    all_positive = bool(all(c > 0.0 for c in scale_coefs))
    ambiguous = bool(r2 == float("-inf"))
    passed = bool(not ambiguous and r2 >= criteria.L4_MIN_GLOBAL_R2 and uses_all_three and all_positive)
    return {"passed": passed, "ambiguous": ambiguous, "n": n, "linear_r2": float(r2),
            "coefficients": [float(c) for c in scale_coefs], "intercept": float(coef[-1]),
            "uses_all_three": uses_all_three, "all_positive": all_positive,
            "note": "선형모델 자체가 후보이므로 스케일별 도함수는 정의상 상수(cv=0) — "
                    "criteria.judge_l4 의 quasi_linear 검사는 자동으로 충족된다"}


def measure_l5(X: np.ndarray, y: np.ndarray, mask: np.ndarray, dt_idx: int = 0, *,
                n_bins: int = N_BINS_L5, min_bin_count: int = MIN_BIN_COUNT_L5) -> dict:
    """dt_prev 의 로그구간별 조건부평균 y → criteria 의 exp/power 피팅 함수 재사용."""
    dt = X[:, dt_idx]
    valid = mask & _finite_mask(dt, y) & (dt > 0)
    dtv, yv = dt[valid], y[valid]
    n = int(valid.sum())
    if n < MIN_ROWS_GENERIC:
        return {"passed": False, "ambiguous": True, "reason": "표본 부족", "n": n}

    lo, hi = np.nanpercentile(dtv, [2, 98])
    if not (hi > lo > 0):
        return {"passed": False, "ambiguous": True, "reason": "dt_prev 범위가 퇴화했다", "n": n}

    edges = np.geomspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dtv, edges) - 1, 0, n_bins - 1)
    t_pts, y_pts, counts = [], [], []
    for b in range(n_bins):
        sel = bin_idx == b
        c = int(sel.sum())
        if c >= min_bin_count:
            t_pts.append(float(np.mean(dtv[sel])))
            y_pts.append(float(np.mean(yv[sel])))
            counts.append(c)
    if len(t_pts) < 0.6 * n_bins:
        return {"passed": False, "ambiguous": True, "reason": "구간 대부분 표본부족",
                "n": n, "n_bins_filled": len(t_pts)}

    t_arr, y_arr = np.array(t_pts), np.array(y_pts)
    exp_fit = criteria._fit_exponential(t_arr, y_arr)     # criteria 의 피팅 함수를 그대로 재사용
    pow_fit = criteria._fit_power_law(t_arr, y_arr)
    r2_linear = criteria._linear_r2(t_arr, y_arr)
    best = max((exp_fit, pow_fit), key=lambda d: d["r2"])
    curvature_margin = best["r2"] - r2_linear
    ambiguous = bool(best["r2"] < criteria.L5_MIN_R2)
    genuine_curvature = bool(curvature_margin >= criteria.L5_MIN_CURVATURE_MARGIN)
    passed = bool(not ambiguous and best.get("decay_positive") and genuine_curvature)
    return {"passed": passed, "ambiguous": ambiguous, "n": n, "n_bins_filled": len(t_pts),
            "chosen": best["form"], "exponential": exp_fit, "power_law": pow_fit,
            "linear_r2": r2_linear, "curvature_margin": float(curvature_margin)}


def measure_law(law: str, names: tuple, X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 symbol_ids: np.ndarray | None) -> dict:
    if law == "L1":
        return measure_l1(X, y, mask, imbalance_idx=names.index("book_imbalance"))
    if law == "L2":
        assert symbol_ids is not None
        return measure_l2(X, y, mask, symbol_ids, ofi_idx=0)
    if law == "L3":
        return measure_l3(X, y, mask, volume_idx=0)
    if law == "L4":
        return measure_l4(X, y, mask)
    if law == "L5":
        return measure_l5(X, y, mask, dt_idx=names.index("dt_prev_over_median"))
    raise ValueError(f"모르는 법칙: {law}")


def in_band(result: dict) -> bool:
    return bool(result.get("passed")) and not bool(result.get("ambiguous"))


VERDICT_TABLE = {
    (True, True): "법칙 성립. 증류가 못 찾은 것 → 방법 문제",
    (True, False): "E0 의 표집·무차원화가 왜곡 → 파이프라인 문제",
    (False, False): "법칙이 이 데이터에서 성립 안 함 → 게이트 기준이 틀림",
    (False, True): "설명 필요. 우연일 가능성 높으니 시드를 바꿔 재확인",
}


# ===========================================================================
# 자기검산 — PREREG "검증" 절. 합성 데이터로 측정 함수 자체를 검사한다.
# ===========================================================================

def run_self_checks() -> dict:
    checks = {}
    rng = np.random.default_rng(12345)   # 자기검산 전용 시드 — 본 측정 SEED 와 분리

    # --- L2: 알려진 기울기 β=2.0, 종목 4개, 기울기를 조금씩 흔든다 -----------
    n_symbols, n_per = 4, 500
    true_slopes = np.array([1.9, 2.0, 2.1, 2.0])
    xs, ys, sids = [], [], []
    for i, b in enumerate(true_slopes):
        x = rng.uniform(-1, 1, n_per)
        y = b * x + rng.normal(0, 1e-6, n_per)
        xs.append(x); ys.append(y); sids.append(np.full(n_per, i))
    X = np.concatenate(xs).reshape(-1, 1)
    y = np.concatenate(ys)
    sid = np.concatenate(sids)
    mask = np.ones(len(y), dtype=bool)
    res = measure_l2(X, y, mask, sid, min_rows=10)
    expected_mean = float(np.mean(true_slopes))
    expected_cv = float(abs(np.std(true_slopes) / np.mean(true_slopes)))
    ok = (abs(res["slope_mean"] - expected_mean) < 1e-3 and abs(res["slope_cv"] - expected_cv) < 1e-3
          and res["passed"] is True)
    checks["L2_known_slope"] = {"ok": bool(ok), "expected_mean": expected_mean,
                                 "expected_cv": expected_cv, "got": res}

    # --- L3: 알려진 지수 γ=0.5, 노이즈 없음 ---------------------------------
    x = rng.uniform(1.0, 100.0, 4000)
    y = 3.0 * x ** 0.5
    Xv = x.reshape(-1, 1)
    mask = np.ones(len(y), dtype=bool)
    res = measure_l3(Xv, y, mask, volume_idx=0)
    ok = abs(res["exponent"] - 0.5) < 1e-2 and res["log_log_r2"] > 0.999 and res["passed"] is True
    checks["L3_known_exponent_0p5"] = {"ok": bool(ok), "expected_exponent": 0.5, "got": res}

    # --- L4: 알려진 양의 계수 3개 --------------------------------------------
    n = 4000
    X3 = rng.uniform(0.0, 1.0, size=(n, 3))
    true_coef = np.array([2.0, 3.0, 1.5])
    y = X3 @ true_coef + rng.normal(0, 1e-6, n)
    mask = np.ones(n, dtype=bool)
    res = measure_l4(X3, y, mask)
    ok = (np.allclose(res["coefficients"], true_coef, atol=1e-2) and res["linear_r2"] > 0.999
          and res["passed"] is True)
    checks["L4_known_coefficients"] = {"ok": bool(ok), "expected": true_coef.tolist(), "got": res}

    # --- L1: 홀·단조 함수(y=I)는 통과, 짝함수(y=I^2)는 홀함수 검사에서 탈락 --
    n = 20000
    I = rng.uniform(-1, 1, n)
    X1 = np.column_stack([I, np.zeros(n)])
    y_odd = I.copy()
    mask = np.ones(n, dtype=bool)
    res_odd = measure_l1(X1, y_odd, mask, imbalance_idx=0)
    y_even = I ** 2
    res_even = measure_l1(X1, y_even, mask, imbalance_idx=0)
    ok = bool(res_odd["passed"] is True and res_even["odd"] is False)
    checks["L1_odd_vs_even"] = {"ok": ok, "odd_case": res_odd, "even_case": res_even}

    # --- L5: 알려진 지수감쇠는 통과, 순수 직선(5*t)은 곡률 부족으로 탈락 -----
    # criteria.py 자체가 이 함정을 문서화한다(순수 직선이 β≈3.8e-5, R²≈1.0 으로
    # "가짜 통과"할 뻔했던 사례, A7).
    t = rng.uniform(0.01, 5.0, 6000)
    y_decay = 0.1 + 2.0 * np.exp(-1.5 * t)
    X5 = t.reshape(-1, 1)
    mask = np.ones(len(t), dtype=bool)
    res_decay = measure_l5(X5, y_decay, mask, dt_idx=0)
    y_linear = 5.0 * t
    res_linear = measure_l5(X5, y_linear, mask, dt_idx=0)
    ok = bool(res_decay["passed"] is True and res_linear["passed"] is False)
    checks["L5_decay_vs_pure_linear"] = {"ok": ok, "decay_case": res_decay, "linear_case": res_linear}

    # --- 뮤테이션: x·y 를 바꿔치기하면 결과가 달라져야 한다 -------------------
    x = rng.uniform(1.0, 50.0, 3000)
    y = 4.0 * x ** 0.5 + rng.normal(0, 0.05, 3000)
    y = np.abs(y)
    mask = np.ones(len(y), dtype=bool)
    forward = measure_l3(x.reshape(-1, 1), y, mask, volume_idx=0)
    swapped = measure_l3(y.reshape(-1, 1), x, mask, volume_idx=0)
    ok = abs(forward["exponent"] - swapped["exponent"]) > 0.05
    checks["mutation_swap_xy_changes_result"] = {
        "ok": bool(ok), "forward_exponent": forward["exponent"], "swapped_exponent": swapped["exponent"]}

    # --- 마스크 무시 검산: mask=False 인 이상치 행을 포함시키면 결과가 달라져야 함 -
    # measure_l2 는 종목 3개 이상을 요구하므로(단일 그룹으로는 CV 정의가 안 됨),
    # 대신 measure_l4(풀링 다중선형회귀)로 마스크 적용 여부의 효과를 검사한다.
    n_good = 2000
    X_good = rng.uniform(0.0, 1.0, size=(n_good, 3))
    true_coef = np.array([2.0, 3.0, 1.5])
    y_good = X_good @ true_coef + rng.normal(0, 1e-3, n_good)
    # 극단치 — 부호가 뒤집힌 관계라 포함되면 계수(특히 첫째 열)를 크게 흔든다.
    X_outlier = np.array([[50.0, 0.0, 0.0], [80.0, 0.0, 0.0], [65.0, 0.0, 0.0]])
    y_outlier = np.array([-900.0, -1400.0, -1100.0])
    X_all = np.concatenate([X_good, X_outlier], axis=0)
    y_all = np.concatenate([y_good, y_outlier])
    mask_true = np.concatenate([np.ones(n_good, dtype=bool), np.zeros(3, dtype=bool)])
    mask_ignored = np.ones(len(y_all), dtype=bool)
    res_masked = measure_l4(X_all, y_all, mask_true)
    res_unmasked = measure_l4(X_all, y_all, mask_ignored)
    ok = abs(res_masked["coefficients"][0] - res_unmasked["coefficients"][0]) > 0.1
    checks["mask_matters"] = {"ok": bool(ok), "masked_coef0": res_masked["coefficients"][0],
                               "mask_ignored_coef0": res_unmasked["coefficients"][0]}

    all_ok = all(c["ok"] for c in checks.values())
    return {"all_ok": bool(all_ok), "checks": checks}


# ===========================================================================
# 데이터 조립 — (a) 직접 풀, (b) E0 표본
# ===========================================================================

def build_direct_pool(all_symbols: tuple, seed: int, n: int) -> tuple:
    """층화 없이 전체 종목 풀에서 무작위로 n개(결정론적, seed 고정)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_symbols), size=min(n, len(all_symbols)), replace=False)
    return tuple(sorted(all_symbols[i] for i in idx))


def build_e0_pool(all_symbols: tuple, date: str, seed: int, per_stratum: int) -> dict:
    """`sd.universe` 층화 → `sd.e0.split` 적합/선택 분할. E0 가 실제로 쓰는 종목 선정."""
    stats_df = universe.liquidity_stats(all_symbols, date, workers=32)
    strata = universe.assign_strata(stats_df)
    e0_symbols = universe.slice_symbols(strata, per_stratum=per_stratum)
    stratum_of = strata["stratum"].to_dict()
    stratum_of = {s: stratum_of[s] for s in e0_symbols}
    symsplit = split.split_symbols(e0_symbols, stratum_of, seed=seed)
    return {"e0_symbols": e0_symbols, "fit_symbols": symsplit.fit, "select_symbols": symsplit.select,
            "per_stratum_counts": symsplit.per_stratum_counts, "stratum_of": stratum_of}


def _manifold_on_manifold_rows(X: np.ndarray, mask: np.ndarray, *, max_samples: int, seed: int):
    selection = manifold.select(X, mask, max_samples=max_samples, seed=seed)
    rows = selection.index[selection.on_manifold]
    return rows


def assemble_e0_sample(law: str, fit_symbols: tuple, select_symbols: tuple, date: str, seed: int,
                        max_manifold_samples: int = MAX_MANIFOLD_SAMPLES) -> dict:
    """E0 가 실제로 쓰는 표본: 적합·선택 종목 각각 on-manifold 표집 후 합친다."""
    fit_ds = targets.assemble(law, fit_symbols, date)
    select_ds = targets.assemble(law, select_symbols, date)

    fit_rows = _manifold_on_manifold_rows(fit_ds.X_dimless, fit_ds.mask,
                                           max_samples=max_manifold_samples, seed=seed)
    select_rows = _manifold_on_manifold_rows(select_ds.X_dimless, select_ds.mask,
                                              max_samples=max_manifold_samples, seed=seed)

    X = np.concatenate([fit_ds.X_dimless[fit_rows], select_ds.X_dimless[select_rows]], axis=0)
    y = np.concatenate([fit_ds.y_dimless[fit_rows], select_ds.y_dimless[select_rows]], axis=0)
    mask = np.ones(len(y), dtype=bool)   # manifold.select 의 usable 정의상 이미 mask&isfinite 만 통과

    fit_sid = fit_ds.symbol_ids[fit_rows] if fit_ds.symbol_ids is not None else np.zeros(len(fit_rows), dtype=int)
    offset = int(fit_sid.max()) + 1 if len(fit_sid) else 0
    select_sid = (select_ds.symbol_ids[select_rows] if select_ds.symbol_ids is not None
                  else np.zeros(len(select_rows), dtype=int)) + offset
    symbol_ids = np.concatenate([fit_sid, select_sid])

    meta = {
        "names_dimless": list(fit_ds.names_dimless),
        "n_fit_symbols_used": len(fit_ds.symbols_used), "n_select_symbols_used": len(select_ds.symbols_used),
        "fit_symbols_skipped": fit_ds.symbols_skipped, "select_symbols_skipped": select_ds.symbols_skipped,
        "n_fit_rows_total": fit_ds.n_rows_total, "n_select_rows_total": select_ds.n_rows_total,
        "n_fit_on_manifold": int(len(fit_rows)), "n_select_on_manifold": int(len(select_rows)),
        "n_rows_used": int(len(y)),
    }
    return {"names": fit_ds.names_dimless, "X": X, "y": y, "mask": mask, "symbol_ids": symbol_ids, "meta": meta}


def assemble_direct_sample(law: str, symbols: tuple, date: str) -> dict:
    ds = targets.assemble(law, symbols, date)
    meta = {
        "names_dimless": list(ds.names_dimless), "n_symbols_used": len(ds.symbols_used),
        "symbols_skipped": ds.symbols_skipped, "n_rows_total": ds.n_rows_total,
        "n_rows_valid_mask": int(np.count_nonzero(ds.mask)),
    }
    return {"names": ds.names_dimless, "X": ds.X_dimless, "y": ds.y_dimless, "mask": ds.mask,
            "symbol_ids": ds.symbol_ids, "meta": meta}


# ===========================================================================
# 메인
# ===========================================================================

def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 자기검산 (합성 데이터) ===", flush=True)
    selfcheck = run_self_checks()
    for name, c in selfcheck["checks"].items():
        print(f"  {'OK ' if c['ok'] else 'FAIL'} {name}")
    if not selfcheck["all_ok"]:
        print("!!! 자기검산 실패 — 측정 코드에 버그가 있다. 결과 해석을 중단한다.")
    print(f"자기검산 전체: {'통과' if selfcheck['all_ok'] else '실패'}\n", flush=True)

    print("=== 종목 풀 조립 ===", flush=True)
    t_pool = time.time()
    all_symbols = universe.stock_symbols(DATE)
    direct_symbols = build_direct_pool(all_symbols, SEED, N_DIRECT_SYMBOLS)
    e0_pool = build_e0_pool(all_symbols, DATE, SEED, PER_STRATUM_E0)
    pool_elapsed = time.time() - t_pool
    print(f"  전체 우주: {len(all_symbols)} 종목")
    print(f"  (a) 직접 풀: {len(direct_symbols)} 종목 (층화 없이 무작위)")
    print(f"  (b) E0 풀: {len(e0_pool['e0_symbols'])} 종목 "
          f"(적합 {len(e0_pool['fit_symbols'])} / 선택 {len(e0_pool['select_symbols'])})")
    print(f"  풀 조립 소요: {pool_elapsed:.1f}s\n", flush=True)

    bands = read_bands()

    results: dict[str, dict] = {}
    for law in LAWS:
        print(f"=== {law} ===", flush=True)
        t_law = time.time()

        direct = assemble_direct_sample(law, direct_symbols, DATE)
        t_direct = time.time()
        res_a = measure_law(law, direct["names"], direct["X"], direct["y"], direct["mask"], direct["symbol_ids"])
        print(f"  (a) 직접: n_symbols_used={direct['meta']['n_symbols_used']} "
              f"n_rows_total={direct['meta']['n_rows_total']} "
              f"n_rows_valid={direct['meta']['n_rows_valid_mask']} "
              f"passed={res_a.get('passed')} ambiguous={res_a.get('ambiguous')} "
              f"({time.time() - t_direct:.1f}s)")

        t_e0 = time.time()
        e0_sample = assemble_e0_sample(law, e0_pool["fit_symbols"], e0_pool["select_symbols"], DATE, SEED)
        res_b = measure_law(law, e0_sample["names"], e0_sample["X"], e0_sample["y"], e0_sample["mask"],
                             e0_sample["symbol_ids"])
        print(f"  (b) E0표본: n_rows_used={e0_sample['meta']['n_rows_used']} "
              f"(적합 on-manifold={e0_sample['meta']['n_fit_on_manifold']}, "
              f"선택 on-manifold={e0_sample['meta']['n_select_on_manifold']}) "
              f"passed={res_b.get('passed')} ambiguous={res_b.get('ambiguous')} "
              f"({time.time() - t_e0:.1f}s)")

        a_in_band = in_band(res_a)
        b_in_band = in_band(res_b)
        verdict = VERDICT_TABLE[(a_in_band, b_in_band)] if law in ("L2", "L3", "L4") else None
        control_ok = (a_in_band and b_in_band) if law in ("L1", "L5") else None

        results[law] = {
            "direct_meta": direct["meta"], "e0_sample_meta": e0_sample["meta"],
            "direct": res_a, "e0_sample": res_b,
            "a_in_band": a_in_band, "b_in_band": b_in_band,
            "verdict": verdict, "control_ok": control_ok,
            "elapsed_seconds": time.time() - t_law,
        }
        print(f"  법칙 소요: {time.time() - t_law:.1f}s\n", flush=True)

    total_elapsed = time.time() - t0

    n_rows_total_direct = sum(results[law]["direct_meta"]["n_rows_total"] for law in LAWS)
    n_rows_total_e0 = sum(results[law]["e0_sample_meta"]["n_fit_rows_total"]
                          + results[law]["e0_sample_meta"]["n_select_rows_total"] for law in LAWS)

    output = {
        "date": DATE, "seed": SEED, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_direct_symbols_requested": N_DIRECT_SYMBOLS, "per_stratum_e0": PER_STRATUM_E0,
                   "max_manifold_samples": MAX_MANIFOLD_SAMPLES,
                   "min_rows_per_symbol_l2": MIN_ROWS_PER_SYMBOL_L2, "min_rows_generic": MIN_ROWS_GENERIC,
                   "n_bins_l1": N_BINS_L1, "n_bins_l5": N_BINS_L5},
        "universe": {"n_total": len(all_symbols), "direct_symbols": list(direct_symbols),
                     "e0_symbols": list(e0_pool["e0_symbols"]), "fit_symbols": list(e0_pool["fit_symbols"]),
                     "select_symbols": list(e0_pool["select_symbols"]),
                     "per_stratum_counts": e0_pool["per_stratum_counts"]},
        "bands": bands,
        "self_check": selfcheck,
        "results": results,
        "totals": {"n_rows_total_direct_pool_summed_over_laws": n_rows_total_direct,
                   "n_rows_total_e0_pool_summed_over_laws": n_rows_total_e0,
                   "elapsed_seconds": total_elapsed},
    }

    out_path = RESULTS_DIR / "l0_direct.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    print("=== 요약 ===")
    print(f"자기검산: {'통과' if selfcheck['all_ok'] else '실패 — 결과 신뢰 불가'}")
    for law in LAWS:
        r = results[law]
        tag = r["verdict"] if r["verdict"] else ("대조군 " + ("OK" if r["control_ok"] else "밴드 밖!"))
        print(f"  {law}: a_in_band={r['a_in_band']} b_in_band={r['b_in_band']}  -> {tag}")
    print(f"\n총 소요: {total_elapsed:.1f}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
