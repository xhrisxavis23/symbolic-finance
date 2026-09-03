import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher import gate  # noqa: E402


class _FakeTeacher:
    """게이트가 무엇을 보는지 고정하기 위한 대역. 학습하지 않는다."""

    bottleneck = 1

    def __init__(self, path_fn, fill_fn):
        self._path_fn, self._fill_fn = path_fn, fill_fn

    def z(self, X):
        return X[:, :1]

    def predict_path(self, X):
        return self._path_fn(X)

    def predict_fill(self, X):
        return self._fill_fn(X)


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    # 역선택: 체결이 잘 되는 상태일수록 실제 결과가 나쁘다
    y_fill = (X[:, 0] > 0).astype(float)
    y_path = -1.0 * X[:, 0] + 0.2 * rng.normal(size=n)
    return X, y_path, y_fill, np.ones(n, dtype=bool)


def test_healthy_teacher_passes_all_checks():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert result.passed
    assert result.checks[gate.ADVERSE_SELECTION]["passed"]


def test_positive_outcome_in_high_fill_bucket_fails_the_gate():
    """체결 잘 되는 구간에서 결과가 양수면 큐 모델이나 라벨을 의심한다."""
    X, _y_path, y_fill, mask = _data()
    y_path = +1.0 * X[:, 0]                      # 부호를 뒤집어 오염을 흉내낸다
    teacher = _FakeTeacher(lambda x: x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.passed
    assert not result.checks[gate.ADVERSE_SELECTION]["passed"]
    assert result.checks[gate.ADVERSE_SELECTION]["top_decile_mean_y_path"] > 0


def test_constant_predictor_fails_beats_constant():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: np.zeros(len(x)), lambda x: np.full(len(x), 0.5))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.checks[gate.BEATS_CONSTANT]["passed"]


def test_miscalibrated_fill_head_is_caught():
    X, y_path, y_fill, mask = _data()
    # 실제 체결률과 무관하게 항상 0.99 를 말하는 헤드
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: np.full(len(x), 0.99))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.checks[gate.FILL_CALIBRATION]["passed"]


def test_checks_report_numbers_not_just_booleans():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert "correlation" in result.checks[gate.BEATS_CONSTANT]
    assert "max_abs_gap" in result.checks[gate.FILL_CALIBRATION]
    assert len(result.checks[gate.FILL_CALIBRATION]["curve"]) == 10


# -- 추가: 브리핑의 5개 테스트는 `_calibration` 이 십분위를 실제로 나누는지
# 검사하지 못한다. `test_miscalibrated_fill_head_is_caught` 는 예측이 상수라
# 십분위를 몽땅 한 버킷에 몰아넣어도(전체 평균만 봐도) 어긋남이 드러난다.
# 사용자 지시(뮤테이션 자기검토)가 "전부 한 버킷에 넣어도 통과하는가" 를 정확히
# 꼽았고, 실제로 `inside = np.ones_like(predicted, dtype=bool)` 로 바꿔도
# 브리핑의 5개 테스트는 전부 그대로 통과했다 — 그 빈틈을 이 테스트로 메운다.
def test_calibration_actually_bins_by_decile_not_the_overall_average():
    """예측이 0~1 에 고르게 퍼져 있고, 상위 십분위에서는 실제가 크게 낮고
    하위 십분위에서는 실제가 크게 높아 **전체 평균은 우연히 맞아떨어지지만**
    (예측 평균 0.5 ≈ 실제 평균 0.5) 십분위별로는 완전히 어긋나는 경우를 만든다.

    십분위를 실제로 나누는 구현은 상위/하위 버킷에서 큰 gap 을 잡아 낙제시켜야
    한다. 모든 표본을 버킷 하나에 몰아넣는(전체 평균만 보는) 구현은 gap 을
    0 으로 계산해 잘못 통과시킨다.
    """
    rng = np.random.default_rng(3)
    n = 5000
    X = rng.normal(size=(n, 2))
    rank = (np.argsort(np.argsort(X[:, 0])) + 0.5) / n   # 0~1 에 고르게 퍼진 예측
    predicted_fill = rank
    actual_fill = 1.0 - rank                             # 대칭 반전: 전체 평균은 0.5 대 0.5
    y_path = -1.0 * X[:, 0]                               # adverse_selection 은 건강하게 유지
    mask = np.ones(n, dtype=bool)
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: predicted_fill)
    result = gate.evaluate(teacher, X, y_path, actual_fill, mask)
    assert not result.checks[gate.FILL_CALIBRATION]["passed"]
    assert result.checks[gate.FILL_CALIBRATION]["max_abs_gap"] > 0.5
