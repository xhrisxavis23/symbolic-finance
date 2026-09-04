"""복원 판정. 계획서 §4 E0 표의 "복원 판정 기준" 열을 코드로 옮긴다.

**이 파일이 게이트의 실체다.** 사람이 수식을 보고 "그럴듯하다"고 판단하는
것이 아니라, 아래 함수들이 숫자로 판정한다. 판정 기준이 애매한 경우
(로그-로그 적합이 나쁘다, 그리드 대부분이 비유한이다 등)는 `ambiguous=True`
로 명시하고 그런 경우는 **복원되지 않은 것으로 센다** — 애매함을 관대함으로
바꾸지 않는다.

모든 함수는 `(passed: bool, ambiguous: bool, **근거 숫자)` 형태의 dict 를
돌려준다. `passed and not ambiguous` 가 최종 판정이다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import sympy
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# 공통 유틸 — sympy 식을 feature 그리드에서 평가한다.
# ---------------------------------------------------------------------------

def _lambdify(expr: sympy.Expr, names: Sequence[str]):
    used = {str(s) for s in expr.free_symbols}
    if not used <= set(names):
        raise ValueError(f"expr 이 names 밖의 심볼을 쓴다: {used - set(names)}")
    symbols = [sympy.Symbol(n) for n in names]
    return sympy.lambdify(symbols, expr, "numpy"), symbols


def grid_eval(expr: sympy.Expr, names: Sequence[str], fixed: dict[str, float],
             sweep_name: str, sweep_values: np.ndarray) -> np.ndarray:
    """`sweep_name` 만 `sweep_values` 로 바꾸고 나머지는 `fixed` 에 고정해 평가한다."""
    fn, _ = _lambdify(expr, names)
    args = []
    for name in names:
        if name == sweep_name:
            args.append(np.asarray(sweep_values, dtype=float))
        else:
            args.append(np.full(len(sweep_values), float(fixed.get(name, 0.0))))
    with np.errstate(all="ignore"):
        out = np.asarray(fn(*args), dtype=float)
    return np.broadcast_to(out, np.shape(sweep_values)).astype(float)


def full_eval(expr: sympy.Expr, names: Sequence[str], X: np.ndarray) -> np.ndarray:
    fn, _ = _lambdify(expr, names)
    args = [X[:, names.index(n)] for n in names]
    with np.errstate(all="ignore"):
        out = np.asarray(fn(*args), dtype=float)
    return np.broadcast_to(out, (X.shape[0],)).astype(float)


def depends_on(expr: sympy.Expr, name: str) -> bool:
    return sympy.Symbol(name) in expr.free_symbols


# ---------------------------------------------------------------------------
# L1 — I 에 대한 단조 홀함수. 계획서: "수식이 I 에 대한 단조 홀함수 형태로
# 나오는가". S(2번째 인수, 마찰비)는 대표값 몇 개에 고정하고 훑는다.
# ---------------------------------------------------------------------------

MONOTONIC_SPEARMAN_MIN = 0.9
ODD_RELATIVE_ERROR_MAX = 0.3
DEGENERATE_SCALE_MIN = 1e-9


def judge_l1(expr: sympy.Expr, names: Sequence[str], X: np.ndarray,
             *, imbalance_name: str = "book_imbalance") -> dict:
    other = [n for n in names if n != imbalance_name]
    i_col = X[:, names.index(imbalance_name)]
    i_col = i_col[np.isfinite(i_col)]
    if i_col.size == 0:
        return {"passed": False, "ambiguous": True, "reason": "book_imbalance 관측이 없다"}
    i_max = float(np.nanpercentile(np.abs(i_col), 95))
    if i_max <= 0:
        return {"passed": False, "ambiguous": True, "reason": "book_imbalance 범위가 0이다"}
    grid = np.linspace(-i_max, i_max, 41)  # 대칭 그리드 — grid[::-1] == -grid

    if not depends_on(expr, imbalance_name):
        return {"passed": False, "ambiguous": False,
                "reason": f"expr 이 {imbalance_name} 에 의존하지 않는다"}

    representative = ([float(np.nanmedian(X[:, names.index(n)])) for n in other]
                      if other else [0.0])
    fixed_options = []
    if other:
        n0 = other[0]
        col = X[:, names.index(n0)]
        col = col[np.isfinite(col)]
        for q in (25, 50, 75):
            fixed_options.append({n0: float(np.nanpercentile(col, q))})
    else:
        fixed_options.append({})

    slices = []
    for fixed in fixed_options:
        g = grid_eval(expr, names, fixed, imbalance_name, grid)
        finite = np.isfinite(g)
        if finite.sum() < 0.8 * len(grid):
            slices.append({"fixed": fixed, "ambiguous": True, "reason": "grid 대부분 비유한"})
            continue
        scale = float(np.nanmax(np.abs(g[finite])))
        if scale < DEGENERATE_SCALE_MIN:
            slices.append({"fixed": fixed, "ambiguous": False, "monotonic": False, "odd": False,
                          "reason": "expr 이 상수로 퇴화했다"})
            continue
        rho, _ = stats.spearmanr(grid, g)
        mirror = g[::-1]
        denom = np.maximum(np.abs(g), np.abs(mirror))
        denom = np.where(denom < DEGENERATE_SCALE_MIN, 1.0, denom)
        odd_error = float(np.nanmedian(np.abs(g + mirror) / denom))
        slices.append({"fixed": fixed, "ambiguous": False,
                      "spearman": float(rho) if np.isfinite(rho) else None,
                      "monotonic": bool(np.isfinite(rho) and abs(rho) >= MONOTONIC_SPEARMAN_MIN),
                      "odd_relative_error": odd_error,
                      "odd": bool(odd_error <= ODD_RELATIVE_ERROR_MAX)})

    ambiguous = any(s.get("ambiguous") for s in slices)
    evaluated = [s for s in slices if not s.get("ambiguous")]
    passed = bool(evaluated) and all(s.get("monotonic") and s.get("odd") for s in evaluated)
    return {"passed": passed, "ambiguous": ambiguous, "slices": slices,
            "grid_max_abs_I": i_max}


# ---------------------------------------------------------------------------
# L2 — 선형: ΔP ≈ β·OFI/Q̄, β > 0. "근사 아핀"을 도함수의 항상성으로 잰다 —
# 도함수가 그리드 전체에서 (거의) 일정하면 그 함수는 그 구간에서 아핀이다.
# ---------------------------------------------------------------------------

AFFINE_CV_MAX = 0.15


def approximately_affine(expr: sympy.Expr, names: Sequence[str], symbol_name: str,
                         X: np.ndarray, *, cv_tolerance: float = AFFINE_CV_MAX,
                         grid_size: int = 41) -> dict:
    """`symbol_name` 방향의 도함수가 그리드 전체에서 (거의) 일정한가.

    **수치 중심차분을 쓴다 — `sympy.diff` 를 쓰지 않는다.** `sign`·`Abs` 처럼
    도함수가 분포(Dirac delta)를 내거나 sympy 가 미분을 접지 못하는 노드에서
    `sympy.diff` 는 평가 불가능한 `Derivative(...)` 를 그대로 남기고,
    `lambdify` 가 그것을 만나면 죽는다. 중심차분은 함수 자체만 평가하므로
    이런 노드에서도(꺾이는 점을 정확히 밟지만 않으면) 항상 값을 낸다 —
    PySR 이 실제로 낼 수 있는 임의의 식에 대해 이 판정이 죽지 않아야 한다.
    """
    if not depends_on(expr, symbol_name):
        return {"passed": False, "ambiguous": False, "slope_mean": 0.0, "slope_cv": None,
                "reason": f"expr 이 {symbol_name} 에 의존하지 않는다"}

    col = X[:, names.index(symbol_name)]
    col = col[np.isfinite(col)]
    if col.size < 10:
        return {"passed": False, "ambiguous": True, "reason": "표본이 모자라다"}
    lo, hi = np.nanpercentile(col, [2, 98])
    if not (hi > lo):
        return {"passed": False, "ambiguous": True, "reason": f"{symbol_name} 범위가 퇴화했다"}
    grid = np.linspace(lo, hi, grid_size)
    step = (hi - lo) * 1e-4

    other = [n for n in names if n != symbol_name]
    fixed = {n: float(np.nanmedian(X[:, names.index(n)])) for n in other}
    g_plus = grid_eval(expr, names, fixed, symbol_name, grid + step)
    g_minus = grid_eval(expr, names, fixed, symbol_name, grid - step)
    values = (g_plus - g_minus) / (2.0 * step)
    finite = np.isfinite(values)
    if finite.sum() < 0.8 * len(grid):
        return {"passed": False, "ambiguous": True, "reason": "도함수 대부분 비유한"}

    values = values[finite]
    mean = float(np.mean(values))
    std = float(np.std(values))
    cv = float(abs(std / mean)) if abs(mean) > DEGENERATE_SCALE_MIN else float("inf")
    passed = bool(cv <= cv_tolerance and mean > 0.0)
    return {"passed": passed, "ambiguous": False, "slope_mean": mean, "slope_cv": cv,
            "positive": bool(mean > 0.0)}


def judge_l2(expr: sympy.Expr, names: Sequence[str], X: np.ndarray) -> dict:
    symbol_name = names[0]
    result = approximately_affine(expr, names, symbol_name, X)
    result["symbol"] = symbol_name
    return result


# ---------------------------------------------------------------------------
# L3′ — 지수가 0.4~0.7. 로그-로그 회귀로 국소 지수를 잰다. σ 는 중앙값에
# 고정한다.
# ---------------------------------------------------------------------------

L3_EXPONENT_RANGE = (0.4, 0.7)
L3_MIN_LOG_LOG_R2 = 0.6


def judge_l3(expr: sympy.Expr, names: Sequence[str], X: np.ndarray,
             *, volume_name: str, sigma_name: str,
             exponent_range: tuple[float, float] = L3_EXPONENT_RANGE) -> dict:
    if not depends_on(expr, volume_name):
        return {"passed": False, "ambiguous": False,
                "reason": f"expr 이 {volume_name} 에 의존하지 않는다"}

    x_col = X[:, names.index(volume_name)]
    x_col = np.abs(x_col[np.isfinite(x_col)])
    x_col = x_col[x_col > 0]
    if x_col.size < 10:
        return {"passed": False, "ambiguous": True, "reason": "유효 표본이 모자라다"}
    lo, hi = np.nanpercentile(x_col, [5, 95])
    if not (hi > lo > 0):
        return {"passed": False, "ambiguous": True, "reason": "범위가 퇴화했다"}
    grid = np.geomspace(lo, hi, 40)

    sigma_col = X[:, names.index(sigma_name)] if sigma_name in names else None
    fixed = {}
    if sigma_col is not None:
        finite_sigma = sigma_col[np.isfinite(sigma_col)]
        fixed[sigma_name] = float(np.nanmedian(finite_sigma)) if finite_sigma.size else 0.0

    g = grid_eval(expr, names, fixed, volume_name, grid)
    finite = np.isfinite(g) & (np.abs(g) > DEGENERATE_SCALE_MIN)
    if finite.sum() < 0.6 * len(grid):
        return {"passed": False, "ambiguous": True, "reason": "grid 상당수가 비유한/0"}

    log_x = np.log(grid[finite])
    log_g = np.log(np.abs(g[finite]))
    slope, intercept, r_value, _p, _se = stats.linregress(log_x, log_g)
    r2 = float(r_value ** 2)
    ambiguous = r2 < L3_MIN_LOG_LOG_R2
    passed = bool(exponent_range[0] <= slope <= exponent_range[1]) and not ambiguous
    return {"passed": passed, "ambiguous": ambiguous, "exponent": float(slope),
            "log_log_r2": r2, "range": exponent_range}


# ---------------------------------------------------------------------------
# L4′ — 세 스케일의 (준)선형 결합. 전역 다중선형 R² + 각 스케일의 도함수
# 항상성(더 느슨한 허용치) + 세 스케일 모두 실질적으로 쓰이는가.
# ---------------------------------------------------------------------------

L4_QUASI_LINEAR_CV_MAX = 0.35
L4_MIN_GLOBAL_R2 = 0.85
L4_MIN_ABS_SLOPE = 1e-6


def judge_l4(expr: sympy.Expr, names: Sequence[str], X: np.ndarray) -> dict:
    per_scale = {}
    for name in names:
        per_scale[name] = approximately_affine(expr, names, name, X,
                                               cv_tolerance=L4_QUASI_LINEAR_CV_MAX)

    finite_rows = np.all(np.isfinite(X), axis=1)
    if finite_rows.sum() < 10:
        return {"passed": False, "ambiguous": True, "reason": "표본이 모자라다",
                "per_scale": per_scale}
    g_full = full_eval(expr, names, X[finite_rows])
    finite_g = np.isfinite(g_full)
    if finite_g.sum() < 10:
        return {"passed": False, "ambiguous": True, "reason": "expr 평가가 대부분 비유한",
                "per_scale": per_scale}

    design = np.column_stack([X[finite_rows][finite_g], np.ones(int(finite_g.sum()))])
    target = g_full[finite_g]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coef
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")

    uses_all_three = all(abs(per_scale[n].get("slope_mean") or 0.0) > L4_MIN_ABS_SLOPE
                         for n in names)
    all_positive = all((per_scale[n].get("slope_mean") or 0.0) > 0.0 for n in names)
    quasi_linear = all(per_scale[n].get("passed") for n in names)
    ambiguous = r2 == float("-inf")
    passed = bool(not ambiguous and r2 >= L4_MIN_GLOBAL_R2 and uses_all_three
                 and all_positive and quasi_linear)
    return {"passed": passed, "ambiguous": ambiguous, "linear_r2": float(r2),
            "coefficients": {n: float(c) for n, c in zip(names, coef[:-1])},
            "intercept": float(coef[-1]), "uses_all_three": uses_all_three,
            "all_positive": all_positive, "per_scale": per_scale}


# ---------------------------------------------------------------------------
# L5 — exp(-βΔt) 또는 멱함수 커널. 둘 다 시도해 더 잘 맞는 쪽으로 판정한다.
# ---------------------------------------------------------------------------

L5_MIN_R2 = 0.7
L5_MIN_CURVATURE_MARGIN = 0.03


def _linear_r2(t: np.ndarray, y: np.ndarray) -> float:
    slope, intercept, r_value, _p, _se = stats.linregress(t, y)
    return float(r_value ** 2)


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")


def _fit_exponential(t: np.ndarray, y: np.ndarray) -> dict:
    def model(t, c0, c1, beta):
        return c0 + c1 * np.exp(-beta * t)

    c0_guess, c1_guess = float(np.min(y)), float(np.max(y) - np.min(y))
    try:
        params, _ = optimize.curve_fit(
            model, t, y, p0=[c0_guess, c1_guess, 1.0 / max(np.median(t), 1e-6)],
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), maxfev=5000)
    except (RuntimeError, ValueError):
        return {"form": "exponential", "r2": float("-inf"), "beta": None}
    r2 = _r2(y, model(t, *params))
    return {"form": "exponential", "r2": float(r2), "beta": float(params[2]),
            "decay_positive": bool(params[2] > 0.0)}


def _fit_power_law(t: np.ndarray, y: np.ndarray, *, eps: float = 1e-6) -> dict:
    def model(t, c0, c1, gamma):
        return c0 + c1 * np.power(t + eps, -gamma)

    c0_guess, c1_guess = float(np.min(y)), float(np.max(y) - np.min(y))
    try:
        params, _ = optimize.curve_fit(
            model, t, y, p0=[c0_guess, c1_guess, 0.5],
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 10.0]), maxfev=5000)
    except (RuntimeError, ValueError):
        return {"form": "power_law", "r2": float("-inf"), "gamma": None}
    r2 = _r2(y, model(t, *params))
    return {"form": "power_law", "r2": float(r2), "gamma": float(params[2]),
            "decay_positive": bool(params[2] > 0.0)}


def judge_l5(expr: sympy.Expr, names: Sequence[str], X: np.ndarray,
             *, dt_name: str, min_r2: float = L5_MIN_R2) -> dict:
    if not depends_on(expr, dt_name):
        return {"passed": False, "ambiguous": False,
                "reason": f"expr 이 {dt_name} 에 의존하지 않는다"}

    dt_col = X[:, names.index(dt_name)]
    dt_col = dt_col[np.isfinite(dt_col) & (dt_col > 0)]
    if dt_col.size < 10:
        return {"passed": False, "ambiguous": True, "reason": "유효 표본이 모자라다"}
    lo, hi = np.nanpercentile(dt_col, [2, 98])
    if not (hi > lo > 0):
        return {"passed": False, "ambiguous": True, "reason": "범위가 퇴화했다"}
    grid = np.geomspace(lo, hi, 50)

    other = [n for n in names if n != dt_name]
    fixed = {n: float(np.nanmedian(X[:, names.index(n)])) for n in other}
    g = grid_eval(expr, names, fixed, dt_name, grid)
    finite = np.isfinite(g)
    if finite.sum() < 0.6 * len(grid):
        return {"passed": False, "ambiguous": True, "reason": "grid 대부분 비유한"}

    t, y = grid[finite], g[finite]
    exp_fit = _fit_exponential(t, y)
    pow_fit = _fit_power_law(t, y)
    r2_linear = _linear_r2(t, y)
    best = max((exp_fit, pow_fit), key=lambda d: d["r2"])
    # `beta`(또는 `gamma`)가 0에 가까우면 exp(-βΔt)·(Δt)^-γ 는 각각 상수·직선으로
    # 퇴화한다 — 그 경우 부호 무관하게 어떤 단조 추세든 흉내낼 수 있어
    # "decay_positive"(β>0) 만으로는 가짜 통과를 막지 못한다(직접 실측: 순수
    # 증가 직선 `5*Δt` 가 β≈3.8e-5 로 R²≈1.0 을 냈다). 그래서 그 형태가 단순
    # 직선보다 **실제로 더 잘 맞는지**(곡률이 있는지)까지 함께 요구한다.
    curvature_margin = best["r2"] - r2_linear
    ambiguous = best["r2"] < min_r2
    genuine_curvature = curvature_margin >= L5_MIN_CURVATURE_MARGIN
    passed = bool(not ambiguous and best.get("decay_positive") and genuine_curvature)
    return {"passed": passed, "ambiguous": ambiguous, "chosen": best["form"],
            "exponential": exp_fit, "power_law": pow_fit, "linear_r2": r2_linear,
            "curvature_margin": float(curvature_margin)}


JUDGES = {"L1": judge_l1, "L2": judge_l2, "L3": judge_l3, "L4": judge_l4, "L5": judge_l5}
