"""`sd.e0.teacher.ScalarTeacher` 테스트. `tests/test_teacher.py`(ShallowMLP)와
같은 자세 — 가중치·시드·L1 이 실제로 손실에 배선돼 있는지까지 본다(단순히
"돌아간다"만 확인하는 테스트는 뮤테이션에 안 잡힌다)."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0.teacher import ScalarTeacher  # noqa: E402


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
