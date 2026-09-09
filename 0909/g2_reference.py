"""G2 — 법칙별 "단순 참조모델"의 적합/예측과 반복 K-fold 참조 천장.
`PREREG-G2.md` §3.1·§3.2 의 구현.

`l0_measure.py` 가 이미 자기검산까지 마친 다섯 법칙의 측정 정의를 그대로
쓰되, "전체에 대해 한 번에 잰다"(L0) 대신 **적합(train) → 예측(held-out)**
두 단계로 나눈다 — 반복 K-fold 로 "이 데이터에서 달성 가능한 표본외 R²"의
신뢰구간을 내려면 훈련과 평가가 분리돼 있어야 한다.

새 회귀 기법을 도입하지 않는다: L1·L5 는 구간별 조건부평균(+`criteria.py`
의 exp/power 피팅 함수 재사용), L2·L4 는 풀링 선형/다중선형회귀, L3 는
로그-로그 회귀 + 부호 복원 규칙. 전부 `l0_measure.py`/`criteria.py` 가
이미 검증한 정의의 재사용이다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import sympy
from scipy import stats

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd.e0 import criteria                    # noqa: E402  (exp/power 피팅 함수 재사용)
from sd.sr.base import weighted_r2            # noqa: E402  (재사용 — 새 채점식 안 만든다)

REPO_0909 = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_0909))
import l0_measure  # noqa: E402  (측정 정의 재사용 — DEGENERATE_SCALE_MIN 등)


# ===========================================================================
# 법칙별 fit(train) / predict(held-out) 쌍. 전부 "params dict 를 돌려주는
# fit"과 "params 를 받아 y_pred(NaN 허용)를 돌려주는 predict"로 통일한다.
# ===========================================================================

def fit_l1(X: np.ndarray, y: np.ndarray, *, imbalance_idx: int = 0,
           n_bins: int = l0_measure.N_BINS_L1,
           min_bin_count: int = l0_measure.MIN_BIN_COUNT_L1) -> dict | None:
    x = X[:, imbalance_idx]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC:
        return None
    i_max = float(np.nanpercentile(np.abs(x), 95))
    if i_max <= 0:
        return None
    edges = np.linspace(-i_max, i_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() >= min_bin_count:
            means[b] = float(np.mean(y[sel]))
    return {"law": "L1", "edges": edges, "means": means, "n_bins": n_bins,
            "imbalance_idx": imbalance_idx}


def predict_l1(params: dict, X: np.ndarray) -> np.ndarray:
    x = X[:, params["imbalance_idx"]]
    edges, means, n_bins = params["edges"], params["means"], params["n_bins"]
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    out = means[bin_idx]
    out = np.where(np.isfinite(x), out, np.nan)
    return out


def fit_l2(X: np.ndarray, y: np.ndarray, *, ofi_idx: int = 0) -> dict | None:
    x = X[:, ofi_idx]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC or np.std(x) < 1e-12:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    return {"law": "L2", "slope": float(slope), "intercept": float(intercept), "ofi_idx": ofi_idx}


def predict_l2(params: dict, X: np.ndarray) -> np.ndarray:
    x = X[:, params["ofi_idx"]]
    return params["slope"] * x + params["intercept"]


def fit_l3(X: np.ndarray, y: np.ndarray, *, volume_idx: int = 0) -> dict | None:
    v = X[:, volume_idx]
    valid = np.isfinite(v) & np.isfinite(y)
    v, yv = v[valid], y[valid]
    xabs, yabs = np.abs(v), np.abs(yv)
    keep = (xabs > 0) & (yabs > 0) & np.isfinite(xabs) & np.isfinite(yabs)
    if keep.sum() < l0_measure.MIN_ROWS_GENERIC:
        return None
    log_x, log_y = np.log(xabs[keep]), np.log(yabs[keep])
    slope, intercept, *_ = stats.linregress(log_x, log_y)
    # 부호 복원 규칙 — train 에서 sign(y) 와 sign(volume) 의 관계를 그대로 잰다
    # (법칙이 원래 가정하는 "거래량 부호를 따른다"는 방향과 같은 것을 확인
    # 차원에서 다시 재는 것이지 새 가정을 더하는 게 아니다).
    sign_v, sign_y = np.sign(v[keep]), np.sign(yv[keep])
    agreement = float(np.mean(sign_v == sign_y))
    sign_rule = 1.0 if agreement >= 0.5 else -1.0
    return {"law": "L3", "slope": float(slope), "intercept": float(intercept),
            "sign_rule": sign_rule, "sign_agreement": agreement, "volume_idx": volume_idx}


def predict_l3(params: dict, X: np.ndarray) -> np.ndarray:
    v = X[:, params["volume_idx"]]
    out = np.full(len(v), np.nan)
    nz = np.isfinite(v) & (v != 0)
    log_pred = params["slope"] * np.log(np.abs(v[nz])) + params["intercept"]
    magnitude = np.exp(log_pred)
    out[nz] = params["sign_rule"] * np.sign(v[nz]) * magnitude
    return out


def fit_l4(X3: np.ndarray, y: np.ndarray) -> dict | None:
    valid = np.all(np.isfinite(X3), axis=1) & np.isfinite(y)
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC:
        return None
    Xv, yv = X3[valid], y[valid]
    design = np.column_stack([Xv, np.ones(len(yv))])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return {"law": "L4", "coef": coef[:-1].tolist(), "intercept": float(coef[-1])}


def predict_l4(params: dict, X3: np.ndarray) -> np.ndarray:
    coef = np.asarray(params["coef"])
    valid = np.all(np.isfinite(X3), axis=1)
    out = np.full(X3.shape[0], np.nan)
    out[valid] = X3[valid] @ coef + params["intercept"]
    return out


# criteria._fit_exponential/_fit_power_law 는 r2/beta(gamma)/decay_positive 만
# 돌려주고 predict 에 필요한 c0/c1(오프셋·진폭)은 안 준다. 그래서 아래
# `_refit_with_full_params` 가 criteria 의 내부 모델 함수 정의
# (c0+c1*exp(-beta t) 등)를 그대로 다시 적용해 c0/c1 까지 포함한 전체
# 파라미터를 얻는다 — 두 번째 진실 원천을 만드는 게 아니라 같은 정의를
# 한 번 더 curve_fit 하는 것뿐이다(모델 함수 자체는 criteria.py 와 동일).

def _refit_with_full_params(t: np.ndarray, y: np.ndarray, form: str) -> dict:
    from scipy import optimize

    if form == "exponential":
        def model(t, c0, c1, beta):
            return c0 + c1 * np.exp(-beta * t)
        c0_guess, c1_guess = float(np.min(y)), float(np.max(y) - np.min(y))
        params, _ = optimize.curve_fit(
            model, t, y, p0=[c0_guess, c1_guess, 1.0 / max(np.median(t), 1e-6)],
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), maxfev=5000)
        return {"form": "exponential", "c0": float(params[0]), "c1": float(params[1]),
                "beta": float(params[2])}
    else:
        def model(t, c0, c1, gamma):
            return c0 + c1 * np.power(t + 1e-6, -gamma)
        c0_guess, c1_guess = float(np.min(y)), float(np.max(y) - np.min(y))
        params, _ = optimize.curve_fit(
            model, t, y, p0=[c0_guess, c1_guess, 0.5],
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 10.0]), maxfev=5000)
        return {"form": "power_law", "c0": float(params[0]), "c1": float(params[1]),
                "gamma": float(params[2])}


def fit_l5(X: np.ndarray, y: np.ndarray, *, dt_idx: int = 0,
           n_bins: int = l0_measure.N_BINS_L5,
           min_bin_count: int = l0_measure.MIN_BIN_COUNT_L5) -> dict | None:
    dt = X[:, dt_idx]
    valid = np.isfinite(dt) & np.isfinite(y) & (dt > 0)
    dtv, yv = dt[valid], y[valid]
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC:
        return None
    lo, hi = np.nanpercentile(dtv, [2, 98])
    if not (hi > lo > 0):
        return None
    edges = np.geomspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dtv, edges) - 1, 0, n_bins - 1)
    t_pts, y_pts = [], []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() >= min_bin_count:
            t_pts.append(float(np.mean(dtv[sel])))
            y_pts.append(float(np.mean(yv[sel])))
    if len(t_pts) < 0.6 * n_bins:
        return None
    t_arr, y_arr = np.array(t_pts), np.array(y_pts)
    exp_summary = criteria._fit_exponential(t_arr, y_arr)
    pow_summary = criteria._fit_power_law(t_arr, y_arr)
    best_summary = max((exp_summary, pow_summary), key=lambda d: d["r2"])
    try:
        full = _refit_with_full_params(t_arr, y_arr, best_summary["form"])
    except (RuntimeError, ValueError):
        return None
    return {"law": "L5", "chosen": best_summary["form"], "full": full, "dt_idx": dt_idx}


def predict_l5(params: dict, X: np.ndarray) -> np.ndarray:
    dt = X[:, params["dt_idx"]]
    out = np.full(len(dt), np.nan)
    pos = np.isfinite(dt) & (dt > 0)
    f = params["full"]
    if f["form"] == "exponential":
        out[pos] = f["c0"] + f["c1"] * np.exp(-f["beta"] * dt[pos])
    else:
        out[pos] = f["c0"] + f["c1"] * np.power(dt[pos] + 1e-6, -f["gamma"])
    return out


FIT: dict[str, Callable] = {"L1": fit_l1, "L2": fit_l2, "L3": fit_l3, "L4": fit_l4, "L5": fit_l5}
PREDICT: dict[str, Callable] = {"L1": predict_l1, "L2": predict_l2, "L3": predict_l3,
                                "L4": predict_l4, "L5": predict_l5}


# ===========================================================================
# 반복 K-fold — 참조 집합 안에서 "종목 단위" 훈련/평가를 반복해 표본외 R² 의
# 분포를 얻는다. `PREREG-G2.md` §3.1.
# ===========================================================================

MIN_ROWS_PER_BIN = 5


def binned_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, *,
             n_bins: int = 20) -> float:
    """예측값으로 정렬한 분위 구간(quantile bin)별 실측평균 vs 예측평균의 R².

    **왜 행 단위(row-level) R² 만으로는 안 되는가 — 배관 확인에서 발견**
    (`G2-DESIGN-NOTES.md` 참고). L1 처럼 L0 이 이미 참으로 확인한 관계
    (직접측정 Spearman ρ=0.997)도, 배관 확인에서 참조 천장을 행 단위 R² 로
    재니 CI 가 0 을 가로질렀다(참조 795행, CI=[-0.027, 0.020]). 이유는
    L0 의 ρ=0.997 자체가 **구간별 조건부평균**(40개 점) 사이의 상관이었지
    개별 틱의 `y` 를 설명하는 비율이 아니었기 때문이다 — 개별 틱의 미래
    수익률은 압도적으로 잡음이고, "법칙"은 그 잡음을 없애 주는 게 아니라
    조건부 기댓값의 형태를 말해 줄 뿐이다. 행 단위 R² 를 유일한 존재 기준
    으로 쓰면 이 프로젝트가 이미 참으로 확인한 관계조차 "신호 없음"으로
    조기 배제될 위험이 있다(`PREREG-G2.md` §5-3 이 예견한 위험이 실제로
    관측됐다).

    그래서 이 함수는 예측값 기준 분위 구간으로 실측 `y` 를 묶어 **구간
    평균 대 구간 평균**의 R² 를 잰다 — calibration/binned-scatter 표준
    기법이다. `judge_l1`/`judge_l4` 의 자기참조 결함(자기 예측을 자기
    예측에 회귀)과는 다르다: 여기서 구간의 "실측값"은 **참조 집합의 진짜
    관측 `y`** 다(모델이 만든 값이 아니다) — 그저 잡음을 줄이려고 평균
    낼 뿐이다.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weight)
    if valid.sum() < n_bins * MIN_ROWS_PER_BIN:
        n_bins = max(1, int(valid.sum() // MIN_ROWS_PER_BIN))
    if valid.sum() < 2 or n_bins < 2:
        return float("-inf")
    yt, yp, w = y_true[valid], y_pred[valid], weight[valid]
    order = np.argsort(yp, kind="mergesort")
    yt, yp, w = yt[order], yp[order], w[order]
    edges = np.array_split(np.arange(len(yp)), n_bins)
    bin_true, bin_pred, bin_w = [], [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        bw = w[idx]
        bin_true.append(float(np.average(yt[idx], weights=bw)))
        bin_pred.append(float(np.average(yp[idx], weights=bw)))
        bin_w.append(float(bw.sum()))
    if len(bin_true) < 2:
        return float("-inf")
    return weighted_r2(np.array(bin_true), np.array(bin_pred), np.array(bin_w))


def repeated_kfold_ceiling(law: str, X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                           symbol_ids: np.ndarray, weight: np.ndarray | None = None, *,
                           k: int = 5, repeats: int = 20, seed: int = 0,
                           metric: str = "binned", n_bins: int = 20) -> dict:
    """참조 집합 R **내부**에서만 도는 반복 K-fold. 증류는 관여하지 않는다.

    `metric`: `"binned"`(기본, §3.1 최종 설계 — `binned_r2` 재사용) 또는
    `"row"`(행 단위 `weighted_r2` — 참고용으로만 같이 낸다, 진단 목적).
    `weight` 를 안 주면 균등 가중(1.0)을 쓴다.
    """
    fit_fn, predict_fn = FIT[law], PREDICT[law]
    if weight is None:
        weight = np.ones(len(y), dtype=float)

    finite_rows = mask & np.all(np.isfinite(np.atleast_2d(X.T).T), axis=1) & np.isfinite(y)
    X, y, symbol_ids, weight = X[finite_rows], y[finite_rows], symbol_ids[finite_rows], weight[finite_rows]

    unique_symbols = np.unique(symbol_ids)
    if len(unique_symbols) < k:
        return {"ok": False, "reason": f"참조 종목이 {len(unique_symbols)}개뿐이라 "
                                        f"{k}-fold 를 만들 수 없다", "n_symbols": int(len(unique_symbols))}

    rng = np.random.default_rng(int(seed))
    r2_values: list[float] = []
    r2_row_values: list[float] = []
    n_fold_fit_failures = 0
    for _rep in range(int(repeats)):
        perm = rng.permutation(len(unique_symbols))
        folds = np.array_split(perm, k)
        preds_all = np.full(len(y), np.nan)
        touched = np.zeros(len(y), dtype=bool)
        for fold in folds:
            test_syms = unique_symbols[fold]
            test_row = np.isin(symbol_ids, test_syms)
            train_row = ~test_row
            if train_row.sum() < l0_measure.MIN_ROWS_GENERIC or test_row.sum() == 0:
                continue
            params = fit_fn(X[train_row], y[train_row])
            if params is None:
                n_fold_fit_failures += 1
                continue
            preds_all[test_row] = predict_fn(params, X[test_row])
            touched[test_row] = True
        valid = touched & np.isfinite(preds_all) & np.isfinite(y)
        if valid.sum() < l0_measure.MIN_ROWS_GENERIC:
            continue
        r2b = binned_r2(y[valid], preds_all[valid], weight[valid], n_bins=n_bins)
        r2r = weighted_r2(y[valid], preds_all[valid], weight[valid])
        if np.isfinite(r2b):
            r2_values.append(float(r2b))
        if np.isfinite(r2r):
            r2_row_values.append(float(r2r))

    chosen = r2_values if metric == "binned" else r2_row_values
    if not chosen:
        return {"ok": False, "reason": "모든 반복에서 유효한 R² 를 못 얻었다",
                "n_fold_fit_failures": n_fold_fit_failures}

    arr = np.array(chosen)
    out = {"ok": True, "n_repeats_used": len(arr), "n_repeats_requested": int(repeats),
           "n_fold_fit_failures": n_fold_fit_failures, "metric": metric,
           "r2_median": float(np.median(arr)), "r2_ci_low": float(np.percentile(arr, 2.5)),
           "r2_ci_high": float(np.percentile(arr, 97.5)), "r2_values": arr.tolist(),
           "n_symbols_reference": int(len(unique_symbols)), "n_rows_reference": int(len(y))}
    if r2_row_values:
        row_arr = np.array(r2_row_values)
        out["row_r2_median_diagnostic"] = float(np.median(row_arr))
        out["row_r2_ci_low_diagnostic"] = float(np.percentile(row_arr, 2.5))
    return out


def fit_on_all_and_predict(law: str, X_fit: np.ndarray, y_fit: np.ndarray, mask_fit: np.ndarray,
                           X_score: np.ndarray) -> np.ndarray | None:
    """§2.3 의 "단순회귀 기준선" — 증류 집합(D_fit∪D_select) 전체에 참조모델을
    적합하고, 참조 집합(R)에 예측값을 낸다. 반환값이 None 이면 적합 실패."""
    fit_fn, predict_fn = FIT[law], PREDICT[law]
    finite_rows = mask_fit & np.all(np.isfinite(np.atleast_2d(X_fit.T).T), axis=1) & np.isfinite(y_fit)
    params = fit_fn(X_fit[finite_rows], y_fit[finite_rows])
    if params is None:
        return None
    return predict_fn(params, X_score)


def score_candidate_binned(candidate, feature_names, X: np.ndarray, y: np.ndarray,
                           weight: np.ndarray, *, n_bins: int = 20) -> float:
    """`sd.sr.base.score_candidate` 와 같은 평가 방식(lambdify → broadcast)
    이지만 마지막 채점 함수만 `weighted_r2` 대신 `binned_r2` 를 쓴다 —
    `PREREG-G2.md` §3.5 최종 설계(배관 확인에서 행 단위 R² 의 한계가
    드러난 뒤 채택, `G2-DESIGN-NOTES.md` 참고). **후보 표현식을 실제
    관측 `y`(참조 집합, 증류가 한 번도 못 본 데이터)에 평가하는 것은
    그대로다** — 바뀐 것은 잡음을 줄이는 집계 방식뿐이고, 자기참조(후보를
    후보 자신에 회귀)는 여전히 안 쓴다."""
    try:
        symbols = [sympy.Symbol(n) for n in feature_names]
        fn = sympy.lambdify(symbols, candidate.expr, "numpy")
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(*[X[:, j] for j in range(len(feature_names))]), dtype=float)
        prediction = np.broadcast_to(raw, np.asarray(y).shape).astype(float)
    except Exception:  # noqa: BLE001 — 평가 실패는 -inf 로, 예외로 죽지 않는다(score_candidate 와 같은 관례)
        return float("-inf")
    return binned_r2(y, prediction, weight, n_bins=n_bins)
