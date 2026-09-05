import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher.deeplob import DeepLOBCompact  # noqa: E402
from sd.teacher.window import make_causal_windows  # noqa: E402


def _linear_problem(n=2000, seed=0, window=4):
    """`ShallowMLP` 테스트(`tests/test_teacher.py::_linear_problem`)와 같은 문제를
    윈도우로 감싼 것 — 신호가 현재 행에만 있으므로 윈도우 유무와 무관하게
    풀려야 한다(순수 배관 검증용)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y_path = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    y_fill = (X[:, 2] > 0.0).astype(float)
    Xw = make_causal_windows(X, window=window)
    return Xw, y_path, y_fill


def _kwargs(**overrides):
    base = dict(n_features=4, bottleneck=2, window=4, seed=0)
    base.update(overrides)
    return base


def test_bottleneck_shape_is_respected():
    Xw, y_path, y_fill = _linear_problem()
    model = DeepLOBCompact(**_kwargs()).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=30)
    assert model.z(Xw).shape == (len(Xw), 2)
    assert model.bottleneck == 2


@pytest.mark.parametrize("bottleneck", [1, 2, 5])
def test_bottleneck_dimension_cannot_be_bypassed(bottleneck):
    """넓은 표현(LSTM 은닉 상태 그대로)을 내보내는 버그를 잡는 회귀 —
    `bottleneck` 을 무엇으로 주든 `z()` 의 열 수는 정확히 그 값이어야 한다.
    `lstm_hidden`(기본 16) 과 다른 값을 여러 개 시험해 "우연히 16 이라
    통과"하는 요행을 배제한다."""
    Xw, y_path, y_fill = _linear_problem()
    model = DeepLOBCompact(**_kwargs(bottleneck=bottleneck)).fit(
        Xw, y_path, y_fill, np.ones(len(Xw)), epochs=10)
    assert model.z(Xw).shape == (len(Xw), bottleneck)


def test_predictions_have_right_shape_and_range():
    Xw, y_path, y_fill = _linear_problem()
    model = DeepLOBCompact(**_kwargs()).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=30)
    path = model.predict_path(Xw)
    fill = model.predict_fill(Xw)
    assert path.shape == fill.shape == (len(Xw),)
    assert np.all((fill >= 0.0) & (fill <= 1.0))


def test_teacher_learns_a_linear_signal():
    Xw, y_path, y_fill = _linear_problem()
    model = DeepLOBCompact(**_kwargs()).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=250)
    predicted = model.predict_path(Xw)
    correlation = float(np.corrcoef(predicted, y_path)[0, 1])
    assert correlation > 0.8


def test_shape_mismatch_raises_clear_error():
    """`X` 가 `window*n_features` 열이 아니면(윈도우를 안 만들고 raw 를 바로
    주는 실수) 조용히 잘못된 reshape 를 하지 말고 바로 에러여야 한다."""
    model = DeepLOBCompact(**_kwargs(n_features=4, window=4))
    X_unwindowed = np.zeros((10, 4))          # window 적용 전 raw 모양
    with pytest.raises(ValueError):
        model.fit(X_unwindowed, np.zeros(10), np.zeros(10), np.ones(10), epochs=1)


# -- 핵심 차별점: DeepLOBCompact 는 과거를 본다, ShallowMLP 는 못 본다 --------

def test_deeplob_recovers_a_lagged_signal_that_needs_history():
    """타깃이 **현재가 아니라 과거** 관측에 의존하는 문제를 만든다
    (`y_path[i] = 3 * raw[i - lag]`). 이 신호는 행 i 자신의 feature 안에는
    없다 — 윈도우로 과거를 들여다볼 때만 복원 가능하다. `ShallowMLP` 류(현재
    행만 보는 구조)는 구조적으로 이 신호에 접근할 수 없다는 것까지 같이
    확인한다 — "윈도우를 무시하고 마지막 시점만 쓰는" 퇴화도 이 테스트가
    잡는다(마지막 시점의 raw 값은 lag 전 값과 독립이므로 상관이 낮게 나옴)."""
    rng = np.random.default_rng(7)
    n = 4000
    lag = 5
    window = 12
    raw = rng.normal(size=(n, 1))
    y_path = np.empty(n)
    y_path[lag:] = 3.0 * raw[:-lag, 0]
    y_path[:lag] = 3.0 * raw[0, 0]
    y_fill = (raw[:, 0] > 0.0).astype(float)          # 현재 행에만 의존(대조용)

    Xw = make_causal_windows(raw, window=window)
    model = DeepLOBCompact(n_features=1, bottleneck=2, window=window, seed=0).fit(
        Xw, y_path, y_fill, np.ones(n), epochs=400)
    corr = float(np.corrcoef(model.predict_path(Xw), y_path)[0, 1])
    assert corr > 0.55

    # 대조: 윈도우를 주지 않고(=현재 행만) 같은 구조로 학습하면 이 신호에
    # 접근할 수 없다 — window=1 은 곧 "현재 행만 보는 ShallowMLP 급 정보량"과
    # 같다.
    Xw_current_only = make_causal_windows(raw, window=1)
    naive = DeepLOBCompact(n_features=1, bottleneck=2, window=1, seed=0).fit(
        Xw_current_only, y_path, y_fill, np.ones(n), epochs=400)
    corr_naive = float(np.corrcoef(naive.predict_path(Xw_current_only), y_path)[0, 1])
    assert corr_naive < 0.3
    assert corr - corr_naive > 0.3


# -- 사용자 지시 뮤테이션 자기검토 항목들의 상시 회귀 버전 -------------------

def test_heads_are_not_aliased():
    """두 헤드가 서로의 복사가 아닌지 확인한다. `y_path` 와 `y_fill` 이 현재
    행의 **서로 다른** 열에 의존하도록 만들어(`ShallowMLP` 쪽
    `tests/test_teacher.py::_linear_problem` 과 같은 구성 — lag 는 여기서
    변수가 아니다, 그건 `test_deeplob_recovers_a_lagged_signal_...` 이 이미
    맡는다) 두 헤드가 실제로 분업해야만 둘 다 통과하게 짠다."""
    rng = np.random.default_rng(3)
    n = 3000
    window = 4
    raw = rng.normal(size=(n, 4))
    y_path = 2.0 * raw[:, 0]
    y_fill = (raw[:, 2] > 0.0).astype(float)

    # 실측(자기검토 중 발견): path 의 MSE 손실 스케일이 fill 의 BCE 보다 커서
    # 공유 optimiser 가 초반 수백 epoch 을 path 에만 쓰고 fill 로짓이 한동안
    # 거의 0 부근에 머문다(에폭별 corr_fill: 50→0.02, 350→0.09, 700→0.9998) —
    # 버그가 아니라 두 손실 스케일이 다른 co-training 의 실제 수렴 속도다.
    # 400 epoch 로는 이 시드에서 fill 이 아직 덜 여문 채였다(corr≈0.02) — 이
    # 테스트가 잡으려는 것(아예 분업하지 않는 aliasing)과는 다른 실패라
    # epoch 을 늘려 수렴 뒤에 판정한다.
    Xw = make_causal_windows(raw, window=window)
    model = DeepLOBCompact(n_features=4, bottleneck=2, window=window, seed=0).fit(
        Xw, y_path, y_fill, np.ones(n), epochs=700)

    weight_path = model._head_path.weight.detach().numpy()
    weight_fill = model._head_fill.weight.detach().numpy()
    assert model._head_path is not model._head_fill
    assert not np.allclose(weight_path, weight_fill)

    predicted_path = model.predict_path(Xw)
    predicted_fill = model.predict_fill(Xw)
    corr_path_own = float(np.corrcoef(predicted_path, y_path)[0, 1])
    corr_fill_own = float(np.corrcoef(predicted_fill, y_fill)[0, 1])
    corr_path_cross = float(np.corrcoef(predicted_path, y_fill)[0, 1])
    corr_fill_cross = float(np.corrcoef(predicted_fill, y_path)[0, 1])

    assert corr_path_own > 0.5
    assert corr_fill_own > 0.5
    assert corr_path_own - abs(corr_path_cross) > 0.2
    assert corr_fill_own - abs(corr_fill_cross) > 0.2


def test_weight_actually_changes_the_fit():
    """`ShallowMLP` 버전(`tests/test_teacher.py::test_weight_actually_changes_the_fit`)
    과 같은 구성 — 두 반쪽이 정반대 기울기를 갖는 문제에서 `weight` 를
    어느 반쪽에 쏠리게 주느냐로 적합된 부호가 실제로 뒤집히는지 확인한다."""
    rng = np.random.default_rng(11)
    n = 2000
    window = 4
    X = rng.normal(size=(n, 4))
    half = n // 2
    y_path = np.empty(n)
    y_path[:half] = 3.0 * X[:half, 0]
    y_path[half:] = -3.0 * X[half:, 0]
    y_fill = (X[:, 2] > 0.0).astype(float)
    Xw = make_causal_windows(X, window=window)

    weight_first_half = np.concatenate([np.full(half, 20.0), np.full(n - half, 0.05)])
    weight_second_half = np.concatenate([np.full(half, 0.05), np.full(n - half, 20.0)])

    kwargs = _kwargs(window=window, seed=5)
    model_first = DeepLOBCompact(**kwargs).fit(Xw, y_path, y_fill, weight_first_half, epochs=400)
    model_second = DeepLOBCompact(**kwargs).fit(Xw, y_path, y_fill, weight_second_half, epochs=400)

    corr_first = float(np.corrcoef(model_first.predict_path(Xw), X[:, 0])[0, 1])
    corr_second = float(np.corrcoef(model_second.predict_path(Xw), X[:, 0])[0, 1])

    assert corr_first > 0.5
    assert corr_second < -0.5


def test_same_seed_gives_same_model():
    Xw, y_path, y_fill = _linear_problem(seed=0)
    kwargs = _kwargs(seed=3)
    a = DeepLOBCompact(**kwargs).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=20)
    b = DeepLOBCompact(**kwargs).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=20)
    np.testing.assert_allclose(a.predict_path(Xw), b.predict_path(Xw), rtol=1e-5)


def test_different_seeds_give_different_predictions():
    Xw, y_path, y_fill = _linear_problem(seed=0)
    weight = np.ones(len(Xw))
    a = DeepLOBCompact(**_kwargs(seed=1)).fit(Xw, y_path, y_fill, weight, epochs=20)
    b = DeepLOBCompact(**_kwargs(seed=2)).fit(Xw, y_path, y_fill, weight, epochs=20)
    assert not np.allclose(a.predict_path(Xw), b.predict_path(Xw), rtol=1e-6)


def test_l1_penalty_shrinks_the_bottleneck():
    Xw, y_path, y_fill = _linear_problem()
    weight = np.ones(len(Xw))
    no_l1 = DeepLOBCompact(**_kwargs(l1=0.0)).fit(Xw, y_path, y_fill, weight, epochs=250)
    heavy_l1 = DeepLOBCompact(**_kwargs(l1=5.0)).fit(Xw, y_path, y_fill, weight, epochs=250)

    no_l1_magnitude = float(np.mean(np.abs(no_l1.z(Xw))))
    heavy_l1_magnitude = float(np.mean(np.abs(heavy_l1.z(Xw))))
    assert heavy_l1_magnitude < 0.5 * no_l1_magnitude


def test_predict_fill_stays_in_unit_interval_with_confident_head():
    Xw, y_path, y_fill = _linear_problem(n=3000, seed=7)
    model = DeepLOBCompact(**_kwargs()).fit(Xw, y_path, y_fill, np.ones(len(Xw)), epochs=500)
    fill = model.predict_fill(Xw)
    assert fill.min() >= 0.0 and fill.max() <= 1.0
    assert fill.max() > 0.9
    assert fill.min() < 0.1
