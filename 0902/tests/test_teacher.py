import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher.shallow import ShallowMLP  # noqa: E402


def _linear_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y_path = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    y_fill = (X[:, 2] > 0.0).astype(float)
    return X, y_path, y_fill


def test_bottleneck_shape_is_respected():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=50)
    assert model.z(X).shape == (len(X), 2)
    assert model.bottleneck == 2


def test_predictions_have_right_shape_and_range():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=50)
    path = model.predict_path(X)
    fill = model.predict_fill(X)
    assert path.shape == fill.shape == (len(X),)
    assert np.all((fill >= 0.0) & (fill <= 1.0))


def test_teacher_learns_a_linear_signal():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=400)
    predicted = model.predict_path(X)
    correlation = float(np.corrcoef(predicted, y_path)[0, 1])
    assert correlation > 0.9


def test_same_seed_gives_same_model():
    X, y_path, y_fill = _linear_problem()
    kwargs = dict(n_features=4, bottleneck=2, seed=3)
    a = ShallowMLP(**kwargs).fit(X, y_path, y_fill, np.ones(len(X)), epochs=30)
    b = ShallowMLP(**kwargs).fit(X, y_path, y_fill, np.ones(len(X)), epochs=30)
    np.testing.assert_allclose(a.predict_path(X), b.predict_path(X), rtol=1e-6)


# -- 추가: 브리핑의 4개 테스트는 전부 weight=np.ones(...) 로만 학습해 가중치가
# 실제로 손실에 반영되는지, 다른 seed 가 실제로 다른 결과를 내는지, l1 이
# 실제로 병목을 조이는지를 검증하지 못한다. 사용자 지시(뮤테이션 자기검토)가
# 정확히 이 세 가지를 꼽았으므로 별도 테스트로 메운다.

def test_weight_actually_changes_the_fit():
    """두 반쪽이 정반대 기울기를 갖는 문제에서, 가중치를 어느 반쪽에
    쏠리게 주느냐에 따라 적합된 부호가 실제로 뒤집히는지 확인한다.

    `weight` 를 손실에서 무시하면(예: 전부 1로 취급) 두 학습은 항상
    같은(상쇄된) 결과를 내 이 테스트가 잡아낸다.
    """
    rng = np.random.default_rng(11)
    n = 1000
    X = rng.normal(size=(n, 4))
    half = n // 2
    y_path = np.empty(n)
    y_path[:half] = 3.0 * X[:half, 0]
    y_path[half:] = -3.0 * X[half:, 0]
    y_fill = (X[:, 2] > 0.0).astype(float)

    weight_first_half = np.concatenate([np.full(half, 20.0), np.full(n - half, 0.05)])
    weight_second_half = np.concatenate([np.full(half, 0.05), np.full(n - half, 20.0)])

    kwargs = dict(n_features=4, bottleneck=2, seed=5)
    model_first = ShallowMLP(**kwargs).fit(X, y_path, y_fill, weight_first_half, epochs=400)
    model_second = ShallowMLP(**kwargs).fit(X, y_path, y_fill, weight_second_half, epochs=400)

    corr_first = float(np.corrcoef(model_first.predict_path(X), X[:, 0])[0, 1])
    corr_second = float(np.corrcoef(model_second.predict_path(X), X[:, 0])[0, 1])

    assert corr_first > 0.5
    assert corr_second < -0.5


def test_different_seeds_give_different_predictions():
    """`test_same_seed_gives_same_model` 은 seed 가 실제로 쓰이는지는 보장하지
    않는다 — 구현이 seed 를 완전히 무시해도 "같은 seed → 같은 결과" 는
    자명하게 참이다. 서로 다른 seed 가 서로 다른 결과를 내는지까지 봐야
    seed 가 실제로 초기화·학습에 배선돼 있음을 확인할 수 있다.
    """
    X, y_path, y_fill = _linear_problem()
    weight = np.ones(len(X))
    a = ShallowMLP(n_features=4, bottleneck=2, seed=1).fit(X, y_path, y_fill, weight, epochs=30)
    b = ShallowMLP(n_features=4, bottleneck=2, seed=2).fit(X, y_path, y_fill, weight, epochs=30)
    assert not np.allclose(a.predict_path(X), b.predict_path(X), rtol=1e-6)


def test_l1_penalty_shrinks_the_bottleneck():
    """`l1` 인자가 실제로 손실에 걸리는지 확인한다. `l1=0` 대비 큰 `l1` 이
    병목 활성값의 평균 절대크기를 눈에 띄게 줄여야 한다.
    """
    X, y_path, y_fill = _linear_problem()
    weight = np.ones(len(X))
    no_l1 = ShallowMLP(n_features=4, bottleneck=2, seed=0, l1=0.0).fit(
        X, y_path, y_fill, weight, epochs=300)
    heavy_l1 = ShallowMLP(n_features=4, bottleneck=2, seed=0, l1=5.0).fit(
        X, y_path, y_fill, weight, epochs=300)

    no_l1_magnitude = float(np.mean(np.abs(no_l1.z(X))))
    heavy_l1_magnitude = float(np.mean(np.abs(heavy_l1.z(X))))
    assert heavy_l1_magnitude < 0.5 * no_l1_magnitude


# ---------------------------------------------------------------------------
# A10 조기 종료 (PREREG-T1.md §1) — `sd.e0.teacher.ScalarTeacher` 와 같은
# 능력(경로 헤드 R² 기준)이 `ShallowMLP` 에도 있다. 전체 뮤테이션 자기검토는
# `tests/test_e0_teacher.py`(ScalarTeacher)·`tests/test_teacher_early_stopping.py`
# (스케줄러 자체)가 담당한다 — 여기서는 하위 호환성과 배선만 확인한다.
# ---------------------------------------------------------------------------

def test_fit_without_select_data_disables_early_stopping():
    """`X_path_select` 를 안 주면(기존 모든 호출부, `run_slice.py` 포함)
    조기 종료가 완전히 꺼지고 `epochs` 를 전부 돈다."""
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=37)
    history = model.early_stop_history_
    assert history.early_stopping_enabled is False
    assert history.stopped_epoch == 37


def test_fit_with_select_data_tracks_path_head_r2():
    """경로 헤드가 적합-정반대 선택 관계에서 이른 epoch 을 최선으로 골라야
    한다(`tests/test_e0_teacher.py` 의 같은 설계, 경로 헤드만 본다 — 체결확률
    헤드는 이 기준에 관여하지 않는다)."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    y_path = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.05 * rng.normal(size=n)
    y_fill = (X[:, 2] > 0.0).astype(float)
    X_select = rng.normal(size=(n, 4))
    y_path_select = -(2.0 * X_select[:, 0] - 1.0 * X_select[:, 1]) + 0.05 * rng.normal(size=n)
    weight = np.ones(n)

    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, weight, epochs=300,
        X_path_select=X_select, y_path_select=y_path_select, weight_select=weight,
        eval_every=10, patience=1000)

    history = model.early_stop_history_
    assert history.early_stopping_enabled is True
    assert len(history.eval_epochs) >= 5
    assert history.best_epoch == history.eval_epochs[0], (
        f"최선 시점이 학습 초반이 아니다(궤적: {list(zip(history.eval_epochs, history.eval_scores))})")


def test_predict_fill_stays_in_unit_interval_with_confident_head():
    """`predict_fill` 이 [0,1] 을 벗어날 수 없는지, 그리고 헤드가 실제로
    분리 가능한 신호에서 확신 있는(0 또는 1 근처) 값을 내는지 확인한다.
    후자가 없으면 sigmoid 를 제거해도(로짓이 우연히 작아) 통과할 수 있다.
    """
    X, y_path, y_fill = _linear_problem(n=3000, seed=7)
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=800)
    fill = model.predict_fill(X)
    assert fill.min() >= 0.0 and fill.max() <= 1.0
    assert fill.max() > 0.9
    assert fill.min() < 0.1
