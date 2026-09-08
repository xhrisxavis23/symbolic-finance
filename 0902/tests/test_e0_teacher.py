"""`sd.e0.teacher.ScalarTeacher`·`ScalarDeepLOB` 테스트. `tests/test_teacher.py`
(ShallowMLP)와 같은 자세 — 가중치·시드·L1 이 실제로 손실에 배선돼 있는지까지
본다(단순히 "돌아간다"만 확인하는 테스트는 뮤테이션에 안 잡힌다).

`ScalarDeepLOB` 쪽은 `tests/test_teacher_deeplob.py`(DeepLOBCompact, 두 헤드)의
같은 시나리오를 헤드 하나짜리로 반복한다 — 특히 "윈도우가 실제로 과거를
보게 하는가"(`test_scalar_deeplob_recovers_a_lagged_signal_that_needs_history`)
가 이 클래스의 존재 이유를 직접 검증한다."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0.teacher import ScalarDeepLOB, ScalarTeacher  # noqa: E402
from sd.teacher.window import make_causal_windows  # noqa: E402


def _linear_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_bottleneck_shape_is_respected():
    X, y = _linear_problem()
    model = ScalarTeacher(n_features=3, bottleneck=2, seed=0).fit(X, y, np.ones(len(X)), epochs=50)
    assert model.z(X).shape == (len(X), 2)
    assert model.bottleneck == 2


def test_prediction_has_right_shape():
    X, y = _linear_problem()
    model = ScalarTeacher(n_features=3, bottleneck=2, seed=0).fit(X, y, np.ones(len(X)), epochs=50)
    assert model.predict(X).shape == (len(X),)


def test_teacher_learns_a_linear_signal():
    X, y = _linear_problem()
    model = ScalarTeacher(n_features=3, bottleneck=2, seed=0).fit(X, y, np.ones(len(X)), epochs=400)
    correlation = float(np.corrcoef(model.predict(X), y)[0, 1])
    assert correlation > 0.9


def test_same_seed_gives_same_model():
    X, y = _linear_problem()
    kwargs = dict(n_features=3, bottleneck=2, seed=3)
    a = ScalarTeacher(**kwargs).fit(X, y, np.ones(len(X)), epochs=30)
    b = ScalarTeacher(**kwargs).fit(X, y, np.ones(len(X)), epochs=30)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=1e-6)


def test_different_seeds_give_different_predictions():
    X, y = _linear_problem()
    weight = np.ones(len(X))
    a = ScalarTeacher(n_features=3, bottleneck=2, seed=1).fit(X, y, weight, epochs=30)
    b = ScalarTeacher(n_features=3, bottleneck=2, seed=2).fit(X, y, weight, epochs=30)
    assert not np.allclose(a.predict(X), b.predict(X), rtol=1e-6)


def test_weight_actually_changes_the_fit():
    """두 반쪽이 정반대 기울기를 갖는 문제에서 가중치 쏠림에 따라 적합된
    부호가 실제로 뒤집히는지 — `weight` 를 손실에서 무시하면 이 테스트가 잡는다
    (`tests/test_teacher.py::test_weight_actually_changes_the_fit` 와 같은 설계)."""
    rng = np.random.default_rng(11)
    n = 1000
    X = rng.normal(size=(n, 3))
    half = n // 2
    y = np.empty(n)
    y[:half] = 3.0 * X[:half, 0]
    y[half:] = -3.0 * X[half:, 0]

    weight_first_half = np.concatenate([np.full(half, 20.0), np.full(n - half, 0.05)])
    weight_second_half = np.concatenate([np.full(half, 0.05), np.full(n - half, 20.0)])

    kwargs = dict(n_features=3, bottleneck=2, seed=5)
    model_first = ScalarTeacher(**kwargs).fit(X, y, weight_first_half, epochs=400)
    model_second = ScalarTeacher(**kwargs).fit(X, y, weight_second_half, epochs=400)

    corr_first = float(np.corrcoef(model_first.predict(X), X[:, 0])[0, 1])
    corr_second = float(np.corrcoef(model_second.predict(X), X[:, 0])[0, 1])
    assert corr_first > 0.5
    assert corr_second < -0.5


def test_l1_penalty_shrinks_the_bottleneck():
    X, y = _linear_problem()
    weight = np.ones(len(X))
    no_l1 = ScalarTeacher(n_features=3, bottleneck=2, seed=0, l1=0.0).fit(
        X, y, weight, epochs=300)
    heavy_l1 = ScalarTeacher(n_features=3, bottleneck=2, seed=0, l1=5.0).fit(
        X, y, weight, epochs=300)
    no_l1_magnitude = float(np.mean(np.abs(no_l1.z(X))))
    heavy_l1_magnitude = float(np.mean(np.abs(heavy_l1.z(X))))
    assert heavy_l1_magnitude < 0.5 * no_l1_magnitude


# ---------------------------------------------------------------------------
# ScalarDeepLOB — DeepLOBCompact 인코더 재사용 + 헤드 하나. PREREG-E0-V2.md §1-2.
# ---------------------------------------------------------------------------

def _windowed_linear_problem(n=2000, seed=0, window=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    Xw = make_causal_windows(X, window=window)
    return Xw, y


def _deeplob_kwargs(**overrides):
    base = dict(n_features=3, bottleneck=2, window=4, seed=0)
    base.update(overrides)
    return base


def test_scalar_deeplob_bottleneck_shape_is_respected():
    Xw, y = _windowed_linear_problem()
    model = ScalarDeepLOB(**_deeplob_kwargs()).fit(Xw, y, np.ones(len(Xw)), epochs=30)
    assert model.z(Xw).shape == (len(Xw), 2)
    assert model.bottleneck == 2


def test_scalar_deeplob_prediction_has_right_shape():
    Xw, y = _windowed_linear_problem()
    model = ScalarDeepLOB(**_deeplob_kwargs()).fit(Xw, y, np.ones(len(Xw)), epochs=30)
    assert model.predict(Xw).shape == (len(Xw),)


def test_scalar_deeplob_shape_mismatch_raises_clear_error():
    """`X` 가 `window*n_features` 열이 아니면(윈도우를 안 씌우고 raw 를 바로
    준 실수) 조용히 잘못된 reshape 를 하지 말고 바로 에러여야 한다 —
    `DeepLOBCompact` 와 같은 계약(`tests/test_teacher_deeplob.py::
    test_shape_mismatch_raises_clear_error`)."""
    model = ScalarDeepLOB(**_deeplob_kwargs(n_features=3, window=4))
    X_unwindowed = np.zeros((10, 3))
    try:
        model.fit(X_unwindowed, np.zeros(10), np.ones(10), epochs=1)
    except ValueError:
        pass
    else:
        raise AssertionError("윈도우 없는 X 인데 예외가 안 났다")


def test_scalar_deeplob_learns_a_linear_signal():
    Xw, y = _windowed_linear_problem()
    model = ScalarDeepLOB(**_deeplob_kwargs()).fit(Xw, y, np.ones(len(Xw)), epochs=400)
    correlation = float(np.corrcoef(model.predict(Xw), y)[0, 1])
    assert correlation > 0.8


def test_scalar_deeplob_recovers_a_lagged_signal_that_needs_history():
    """`ScalarDeepLOB` 존재 이유의 핵심 검증 —
    `tests/test_teacher_deeplob.py::test_deeplob_recovers_a_lagged_signal_...`
    를 헤드 하나짜리로 반복한다. 타깃이 현재 행이 아니라 5틱 전 값에만
    의존한다 — 윈도우로 과거를 볼 때만 복원 가능하다."""
    rng = np.random.default_rng(7)
    n = 4000
    lag = 5
    window = 12
    raw = rng.normal(size=(n, 1))
    y = np.empty(n)
    y[lag:] = 3.0 * raw[:-lag, 0]
    y[:lag] = 3.0 * raw[0, 0]

    Xw = make_causal_windows(raw, window=window)
    model = ScalarDeepLOB(n_features=1, bottleneck=2, window=window, seed=0).fit(
        Xw, y, np.ones(n), epochs=400)
    corr = float(np.corrcoef(model.predict(Xw), y)[0, 1])
    assert corr > 0.55

    Xw_current_only = make_causal_windows(raw, window=1)
    naive = ScalarDeepLOB(n_features=1, bottleneck=2, window=1, seed=0).fit(
        Xw_current_only, y, np.ones(n), epochs=400)
    corr_naive = float(np.corrcoef(naive.predict(Xw_current_only), y)[0, 1])
    assert corr_naive < 0.3
    assert corr - corr_naive > 0.3


def test_scalar_deeplob_same_seed_gives_same_model():
    Xw, y = _windowed_linear_problem(seed=0)
    kwargs = _deeplob_kwargs(seed=3)
    a = ScalarDeepLOB(**kwargs).fit(Xw, y, np.ones(len(Xw)), epochs=20)
    b = ScalarDeepLOB(**kwargs).fit(Xw, y, np.ones(len(Xw)), epochs=20)
    np.testing.assert_allclose(a.predict(Xw), b.predict(Xw), rtol=1e-5)


def test_scalar_deeplob_different_seeds_give_different_predictions():
    Xw, y = _windowed_linear_problem(seed=0)
    weight = np.ones(len(Xw))
    a = ScalarDeepLOB(**_deeplob_kwargs(seed=1)).fit(Xw, y, weight, epochs=20)
    b = ScalarDeepLOB(**_deeplob_kwargs(seed=2)).fit(Xw, y, weight, epochs=20)
    assert not np.allclose(a.predict(Xw), b.predict(Xw), rtol=1e-6)


def test_scalar_deeplob_weight_actually_changes_the_fit():
    rng = np.random.default_rng(11)
    n = 2000
    window = 4
    X = rng.normal(size=(n, 3))
    half = n // 2
    y = np.empty(n)
    y[:half] = 3.0 * X[:half, 0]
    y[half:] = -3.0 * X[half:, 0]
    Xw = make_causal_windows(X, window=window)

    weight_first_half = np.concatenate([np.full(half, 20.0), np.full(n - half, 0.05)])
    weight_second_half = np.concatenate([np.full(half, 0.05), np.full(n - half, 20.0)])

    kwargs = _deeplob_kwargs(window=window, seed=5)
    model_first = ScalarDeepLOB(**kwargs).fit(Xw, y, weight_first_half, epochs=400)
    model_second = ScalarDeepLOB(**kwargs).fit(Xw, y, weight_second_half, epochs=400)

    corr_first = float(np.corrcoef(model_first.predict(Xw), X[:, 0])[0, 1])
    corr_second = float(np.corrcoef(model_second.predict(Xw), X[:, 0])[0, 1])
    assert corr_first > 0.5
    assert corr_second < -0.5


def test_scalar_deeplob_l1_penalty_shrinks_the_bottleneck():
    Xw, y = _windowed_linear_problem()
    weight = np.ones(len(Xw))
    no_l1 = ScalarDeepLOB(**_deeplob_kwargs(l1=0.0)).fit(Xw, y, weight, epochs=250)
    heavy_l1 = ScalarDeepLOB(**_deeplob_kwargs(l1=5.0)).fit(Xw, y, weight, epochs=250)
    no_l1_magnitude = float(np.mean(np.abs(no_l1.z(Xw))))
    heavy_l1_magnitude = float(np.mean(np.abs(heavy_l1.z(Xw))))
    assert heavy_l1_magnitude < 0.5 * no_l1_magnitude
