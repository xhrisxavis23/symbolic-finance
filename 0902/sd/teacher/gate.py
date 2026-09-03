"""S1 교사 검증 게이트. 계획서 §3 S1.

> 오염된 교사에서 증류한 수식은 인샘플에서 훌륭하고 해석도 그럴듯하다.
> 이 게이트를 형식적으로 통과시키지 말 것.

세 가지를 본다. 계획서가 열거한 여섯 검사 중, **이 슬라이스에서 실제로 계산되는
것**만 넣었다. 대칭성·스케일 준불변·무시 변수 확인은 DeepLOB 교사가 들어올 때
추가한다 — 얕은 MLP 에 걸어 봐야 뜻이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ADVERSE_SELECTION = "adverse_selection_sign"
FILL_CALIBRATION = "fill_calibration"
BEATS_CONSTANT = "beats_constant"

MIN_CORRELATION = 0.05        # 상수 예측기보다 나은가
MAX_CALIBRATION_GAP = 0.25    # 신뢰도 곡선이 대각선에서 얼마나 벗어나도 되는가


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, dict]


def evaluate(teacher, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
             mask: np.ndarray, deciles: int = 10) -> GateResult:
    X = np.asarray(X, dtype=float)
    usable = np.asarray(mask, dtype=bool) & np.all(np.isfinite(X), axis=1)
    usable &= np.isfinite(y_path) & np.isfinite(y_fill)
    if usable.sum() < deciles * 10:
        raise ValueError("게이트를 재기에 표본이 모자란다")

    Xs = X[usable]
    truth_path = np.asarray(y_path, dtype=float)[usable]
    truth_fill = np.asarray(y_fill, dtype=float)[usable]
    predicted_path = np.asarray(teacher.predict_path(Xs), dtype=float)
    predicted_fill = np.asarray(teacher.predict_fill(Xs), dtype=float)

    checks = {
        BEATS_CONSTANT: _beats_constant(predicted_path, truth_path),
        FILL_CALIBRATION: _calibration(predicted_fill, truth_fill, deciles),
        ADVERSE_SELECTION: _adverse_selection(predicted_fill, truth_path, deciles),
    }
    return GateResult(passed=all(c["passed"] for c in checks.values()), checks=checks)


def _beats_constant(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """상수를 말하는 교사는 증류할 것이 없다."""
    if np.std(prediction) == 0.0:
        return {"passed": False, "correlation": 0.0,
                "why": "교사가 상수를 예측한다. 증류할 함수가 없다"}
    correlation = float(np.corrcoef(prediction, truth)[0, 1])
    return {"passed": bool(correlation > MIN_CORRELATION),
            "correlation": correlation, "threshold": MIN_CORRELATION}


def _calibration(predicted: np.ndarray, truth: np.ndarray, deciles: int) -> dict:
    """예측 체결확률 십분위별 실제 체결률. 대각선 부근이어야 한다."""
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, deciles + 1))
    curve = []
    for i in range(deciles):
        lower, upper = edges[i], edges[i + 1]
        inside = (predicted >= lower) & (predicted <= upper if i == deciles - 1
                                         else predicted < upper)
        if inside.sum() == 0:
            curve.append({"bucket": i, "predicted": float("nan"), "actual": float("nan")})
            continue
        curve.append({"bucket": i,
                      "predicted": float(np.mean(predicted[inside])),
                      "actual": float(np.mean(truth[inside]))})
    gaps = [abs(p["predicted"] - p["actual"]) for p in curve
            if np.isfinite(p["predicted"]) and np.isfinite(p["actual"])]
    worst = float(max(gaps)) if gaps else float("inf")
    return {"passed": bool(worst <= MAX_CALIBRATION_GAP),
            "max_abs_gap": worst, "threshold": MAX_CALIBRATION_GAP, "curve": curve}


def _adverse_selection(predicted_fill: np.ndarray, truth_path: np.ndarray,
                       deciles: int) -> dict:
    """체결이 잘 되는 구간의 실제 결과는 **음수여야 정상**이다.

    지정가 매수가 잘 체결되는 순간은 대체로 파는 쪽이 급한 순간이다. 이 구간에서
    결과가 양수로 나오면 그것은 발견이 아니라 **큐 모델이 낙관적이거나 라벨에
    미래가 새고 있다는 신호**다. 정본 큐는 이미 보수적이므로 라벨을 먼저 의심한다.
    """
    cut = float(np.quantile(predicted_fill, 1.0 - 1.0 / deciles))
    top = predicted_fill >= cut
    if top.sum() == 0:
        return {"passed": False, "top_decile_mean_y_path": float("nan"),
                "why": "상위 십분위가 비었다"}
    mean = float(np.mean(truth_path[top]))
    return {"passed": bool(mean <= 0.0), "top_decile_mean_y_path": mean,
            "top_decile_size": int(top.sum()),
            "why": "양수면 큐 모델 낙관 또는 라벨 누수를 의심한다"}
