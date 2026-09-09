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
           min_bin_count: int = l0_measure.MIN_BIN_COUNT_L1,
           symbol_ids: np.ndarray | None = None) -> dict | None:
    # symbol_ids: 이 법칙엔 안 쓴다 — L4(Ruling R31)만 종목 내 중심화가
    # 필요해서 생긴 인자다. 다섯 법칙의 fit_fn 호출부(§ repeated_kfold_
    # ceiling/fit_on_all_and_predict)를 하나의 시그니처로 통일하려고
    # 받기만 하고 무시한다.
    del symbol_ids
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


def predict_l1(params: dict, X: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> np.ndarray:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일
    x = X[:, params["imbalance_idx"]]
    edges, means, n_bins = params["edges"], params["means"], params["n_bins"]
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    out = means[bin_idx]
    out = np.where(np.isfinite(x), out, np.nan)
    return out


def _clip_to_domain(x: np.ndarray, lo, hi) -> np.ndarray:
    """held-out 입력을 **train 이 실제로 관측한 범위**로 자른다 — `sd/manifold.py`
    의 원칙("넷이 모르는 곳의 답을 수식으로 만들지 않는다")을 참조모델의
    예측 단계에도 그대로 적용한 것. 발견 3(`G2-DESIGN-NOTES.md`)의 원인이자
    수정: 이게 없으면 `predict_l5` 의 멱함수 항이 train 범위 밖의 극단
    `dt`(특히 0 에 가까운 값)에서 발산해 참조 천장이 통째로 무너진다."""
    return np.clip(x, lo, hi)


def fit_l2(X: np.ndarray, y: np.ndarray, *, ofi_idx: int = 0,
           symbol_ids: np.ndarray | None = None) -> dict | None:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일 (위 fit_l1 주석 참고)
    x = X[:, ofi_idx]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC or np.std(x) < 1e-12:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    return {"law": "L2", "slope": float(slope), "intercept": float(intercept), "ofi_idx": ofi_idx,
            "domain_lo": float(np.min(x)), "domain_hi": float(np.max(x))}


def predict_l2(params: dict, X: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> np.ndarray:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일
    x = _clip_to_domain(X[:, params["ofi_idx"]], params["domain_lo"], params["domain_hi"])
    return params["slope"] * x + params["intercept"]


def fit_l3(X: np.ndarray, y: np.ndarray, *, volume_idx: int = 0,
           symbol_ids: np.ndarray | None = None) -> dict | None:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일 (위 fit_l1 주석 참고)
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
            "sign_rule": sign_rule, "sign_agreement": agreement, "volume_idx": volume_idx,
            "domain_lo": float(np.min(xabs[keep])), "domain_hi": float(np.max(xabs[keep]))}


def predict_l3(params: dict, X: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> np.ndarray:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일
    v = X[:, params["volume_idx"]]
    out = np.full(len(v), np.nan)
    nz = np.isfinite(v) & (v != 0)
    abs_v_clipped = _clip_to_domain(np.abs(v[nz]), params["domain_lo"], params["domain_hi"])
    log_pred = params["slope"] * np.log(abs_v_clipped) + params["intercept"]
    magnitude = np.exp(log_pred)
    out[nz] = params["sign_rule"] * np.sign(v[nz]) * magnitude
    return out


def fit_l4(X3: np.ndarray, y: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> dict | None:
    """**Ruling R31 수정 — 종목 내 중심화(within-symbol demeaning).**

    원래 구현(풀링 다중선형회귀, 종목 구분 없음)은 음성 대조군(permuted y,
    `within_symbol` 스코프)에서 5/5 허위양성을 냈다
    (`G2-NEGATIVE-CONTROL-REPORT.md`) — 원인을 종목수준 상관으로 실측
    확인했다(mean(rv5..100)-mean(y) 상관 0.77~0.84, between-symbol 분산
    비중 X 29~34%·y 22%). L4 의 세 입력(RV5·RV20·RV100)과 타깃(미래
    RV20-유사)이 전부 "실현변동성"이라는 같은 종류의 양이라, 종목마다
    그날의 변동성 레짐이 다르면 종목 간 비교만으로도(틱 단위 인과관계
    없이) 신호처럼 보인다.

    L4 가 실제로 주장하는 것은 **종목 안에서** 스케일들이 미래 변동성을
    예측한다는 것이다 — 그래서 종목별 평균(그 종목의 그날 변동성 레짐)을
    먼저 빼고, 남는 **종목 내 편차**만으로 적합한다(패널 데이터의
    고정효과 처리와 같다). `symbol_ids` 없이는 중심화를 할 수 없으므로
    (다른 네 법칙과의 시그니처 호환을 위한 기본값 `None`을 받았을 때)
    조용히 오염된 방식으로 되돌아가지 않고 실패로 처리한다.
    """
    if symbol_ids is None:
        return None
    valid = np.all(np.isfinite(X3), axis=1) & np.isfinite(y) & np.isfinite(symbol_ids)
    if valid.sum() < l0_measure.MIN_ROWS_GENERIC:
        return None
    Xv, yv, sid = X3[valid], y[valid], symbol_ids[valid]

    grand_mean_x = Xv.mean(axis=0)
    grand_mean_y = float(yv.mean())

    X_within = np.empty_like(Xv)
    y_within = np.empty_like(yv)
    for s in np.unique(sid):
        rows = sid == s
        X_within[rows] = Xv[rows] - Xv[rows].mean(axis=0)
        y_within[rows] = yv[rows] - yv[rows].mean()

    # 이미 종목별로 중심화됐으므로(각 종목 안에서 평균 0) 별도 절편 열이
    # 필요 없다 — 절편을 추가하면 수치상 0 근처로 나올 뿐 아무것도 안
    # 바꾼다. 계수만 lstsq 로 구한다.
    coef, *_ = np.linalg.lstsq(X_within, y_within, rcond=None)
    return {"law": "L4", "coef": coef.tolist(), "grand_mean_x": grand_mean_x.tolist(),
            "grand_mean_y": grand_mean_y,
            "domain_lo": X_within.min(axis=0).tolist(), "domain_hi": X_within.max(axis=0).tolist()}


def predict_l4(params: dict, X3: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> np.ndarray:
    """예측 시점엔 held-out 종목 **자신의 y 평균**(고정효과)은 모르지만,
    그 종목 **자신의 X 평균**은 안다 — X 는 관측된 공변량이고, held-out
    분할이 그 종목의 행을 여러 개 준다(참조천장의 K-fold 는 종목당
    수백~수천 행을 남긴다). 그래서 `symbol_ids` 가 주어지면 **그 종목
    자신의 관측된 X 평균**으로 중심화한다 — train 과 정확히 같은 변환을
    test 종목에도 적용하는 것이다. `y` 는 여전히 한 번도 안 본다(정보
    누출이 아니다) — 그저 X 를 그 종목의 X 분포 기준으로 다시 표현할
    뿐이다.

    **첫 구현은 이걸 몰랐다** — `grand_mean_x`(전체 평균)로만 중심화
    했는데, 종목 간 X 수준 차이가 종목 내 편차보다 훨씬 크면(합성
    데이터로 확인: between-symbol sd 3.0 vs within-symbol sd 1.0) 이
    "벗어난 정도"가 실제로는 거의 전부 종목 수준 차이가 되어 학습한
    기울기를 완전히 잘못된 양에 곱하는 꼴이 된다(합성 검증에서 R²=-82
    로 발견 — Ruling R31 구현 중 자체 발견, `G2-DESIGN-NOTES.md` 참고).
    `symbol_ids` 없이 불리면(단일 행 채점 등, 종목 정보가 없는 호출)
    `grand_mean_x` 로 물러난다 — 최선이 아니지만 유일하게 쓸 수 있는
    값이다.
    """
    coef = np.asarray(params["coef"])
    grand_mean_x = np.asarray(params["grand_mean_x"])
    n_features = len(grand_mean_x)
    valid = np.all(np.isfinite(X3), axis=1)
    out = np.full(X3.shape[0], np.nan)
    lo, hi = np.asarray(params["domain_lo"]), np.asarray(params["domain_hi"])

    center = np.tile(grand_mean_x, (X3.shape[0], 1)).astype(float)
    if symbol_ids is not None:
        valid_idx = np.flatnonzero(valid)
        sid_valid = symbol_ids[valid_idx]
        for s in np.unique(sid_valid):
            rows_global = valid_idx[sid_valid == s]
            center[rows_global] = X3[rows_global].mean(axis=0)

    X_centered = np.clip(X3[valid] - center[valid], lo, hi)
    out[valid] = params["grand_mean_y"] + X_centered @ coef
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
           min_bin_count: int = l0_measure.MIN_BIN_COUNT_L5,
           symbol_ids: np.ndarray | None = None) -> dict | None:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일 (fit_l1 주석 참고)
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
    return {"law": "L5", "chosen": best_summary["form"], "full": full, "dt_idx": dt_idx,
            "domain_lo": float(lo), "domain_hi": float(hi)}


def predict_l5(params: dict, X: np.ndarray, *, symbol_ids: np.ndarray | None = None) -> np.ndarray:
    del symbol_ids  # 안 쓴다 — 시그니처만 L4 와 통일
    """**발견 3(`G2-DESIGN-NOTES.md`)의 수정 지점.** train 범위 밖(특히 0 에
    가까운 `dt`)로 멱함수 항을 그대로 외삽하면 `(dt+eps)^-gamma` 가 발산해
    참조 천장 전체가 무너진다(민감도 확인에서 실측: median R² 가
    -3,110,500까지 나왔다). `_clip_to_domain` 으로 train 이 실제로 본 범위
    밖의 `dt` 는 가장 가까운 경계값으로 자른 뒤 평가한다 — 지수함수 항은
    발산하지 않지만(감쇠하므로) 일관성을 위해 같은 자름을 적용한다."""
    dt = X[:, params["dt_idx"]]
    out = np.full(len(dt), np.nan)
    pos = np.isfinite(dt) & (dt > 0)
    dt_clipped = _clip_to_domain(dt[pos], params["domain_lo"], params["domain_hi"])
    f = params["full"]
    if f["form"] == "exponential":
        out[pos] = f["c0"] + f["c1"] * np.exp(-f["beta"] * dt_clipped)
    else:
        out[pos] = f["c0"] + f["c1"] * np.power(dt_clipped + 1e-6, -f["gamma"])
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


def permute_y(y: np.ndarray, symbol_ids: np.ndarray, *, scope: str, seed: int) -> np.ndarray:
    """음성 대조군(Ruling R30) — `y` 를 실제 X 와의 관계에서 떼어낸다.

    **범위 선택 근거.** 두 스코프를 다 지원한다:

    - `"within_symbol"`(기본으로 쓴다) — 종목마다 **그 종목 안에서만**
      `y` 를 섞는다. 각 종목의 `y` 주변분포(평균·분산 — "이 종목은
      원래 변동이 크다" 같은 종목 고유 수준)는 그대로 두고, **틱 단위
      X-y 짝만** 끊는다. 이게 더 엄격한 검사인 이유: 이 설계의 반복
      K-fold(§3.1)는 **종목 단위**로 학습/평가를 나눈다. 만약 종목마다
      `y` 수준이 원래 다르고 그 수준이 (우연히든 아니든) 그 종목의 X
      분포와 얽혀 있다면, 진짜 틱 단위 인과관계가 전혀 없어도 "종목
      A 는 X 도 크고 y 도 크다"는 **생태학적 상관**만으로 참조천장이
      허위로 자격을 얻을 수 있다 — 이 스코프가 정확히 그 경로를 막고
      남겨서, 그것만으로 통과하는지를 시험한다.
    - `"global"` — 참조 집합 전체에서 종목 구분 없이 섞는다. 종목별
      주변분포까지 깨는 훨씬 단순하고 약한(=더 쉽게 기각되는) 귀무가설
      이다 — 표준적인 "무작위로 섞으면 신호가 사라져야 한다" 자기검산과
      같은 종류다. 이것만으로는 위 생태학적 상관 문제를 못 잡는다.

    **선결 조건(§4.1)은 `within_symbol` 을 요구한다** — 더 엄격한 쪽이
    참조 측정 코드의 결백을 더 강하게 보증하기 때문이다. `global` 은
    보조 진단으로 같이 낸다(둘 다 자격 없음이면 이중으로 안심할 수 있고,
    `global` 만 자격 없음·`within_symbol` 만 자격 있음이면 정확히 위에서
    설명한 생태학적 상관 문제를 의심해야 한다).
    """
    rng = np.random.default_rng(int(seed))
    y_perm = np.array(y, dtype=float, copy=True)
    if scope == "global":
        y_perm = rng.permutation(y_perm)
    elif scope == "within_symbol":
        for sid in np.unique(symbol_ids):
            idx = np.flatnonzero(symbol_ids == sid)
            if len(idx) > 1:
                y_perm[idx] = rng.permutation(y_perm[idx])
    else:
        raise ValueError(f"모르는 scope: {scope}")
    return y_perm


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
            params = fit_fn(X[train_row], y[train_row], symbol_ids=symbol_ids[train_row])
            if params is None:
                n_fold_fit_failures += 1
                continue
            preds_all[test_row] = predict_fn(params, X[test_row], symbol_ids=symbol_ids[test_row])
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
                           X_score: np.ndarray, symbol_ids_fit: np.ndarray | None = None,
                           symbol_ids_score: np.ndarray | None = None) -> np.ndarray | None:
    """§2.3 의 "단순회귀 기준선" — 증류 집합(D_fit∪D_select) 전체에 참조모델을
    적합하고, 참조 집합(R)에 예측값을 낸다. 반환값이 None 이면 적합 실패.

    `symbol_ids_fit`/`symbol_ids_score` — L4(Ruling R31, 종목 내 중심화)에
    필요하다. `fit` 쪽이 없으면 `fit_l4` 가 (오염된 방식으로 조용히
    돌아가는 대신) `None` 을 돌려준다. `score` 쪽(R 자신의 종목 식별자)이
    없으면 `predict_l4` 가 R 의 모집단 평균으로 물러난다(발견, `predict_l4`
    docstring 참고) — 가능하면 항상 넘겨야 한다."""
    fit_fn, predict_fn = FIT[law], PREDICT[law]
    finite_rows = mask_fit & np.all(np.isfinite(np.atleast_2d(X_fit.T).T), axis=1) & np.isfinite(y_fit)
    sid_arg = symbol_ids_fit[finite_rows] if symbol_ids_fit is not None else None
    params = fit_fn(X_fit[finite_rows], y_fit[finite_rows], symbol_ids=sid_arg)
    if params is None:
        return None
    return predict_fn(params, X_score, symbol_ids=symbol_ids_score)


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
