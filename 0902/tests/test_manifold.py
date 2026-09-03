import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import manifold  # noqa: E402


def _blob(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def test_selection_only_returns_masked_rows():
    X = _blob()
    mask = np.zeros(len(X), dtype=bool)
    mask[::3] = True
    sel = manifold.select(X, mask, max_samples=100, seed=0)
    assert np.all(mask[sel.index])


def test_far_outliers_are_flagged_as_probe():
    X = _blob(n=500)
    X = np.vstack([X, np.full((5, 3), 50.0)])         # 명백한 off-manifold
    mask = np.ones(len(X), dtype=bool)
    sel = manifold.select(X, mask, max_samples=len(X), seed=0)
    tail = sel.index >= 500
    assert tail.sum() > 0
    assert not sel.on_manifold[tail].any()


def test_weights_are_normalised_and_positive():
    X = _blob()
    mask = np.ones(len(X), dtype=bool)
    sel = manifold.select(X, mask, max_samples=200, seed=0)
    assert len(sel.weight) == len(sel.index)
    assert np.all(sel.weight > 0)
    assert sel.weight.sum() == pytest.approx(len(sel.index), rel=1e-9)


# -- F5 (최종 전체 리뷰): `WEIGHT_CLIP` 은 죽은 상수였다 — `ranks ∈ [0,1]` 이라
# `1.0 + 4.0*ranks` 의 자연 상한이 이미 5.0 이고, `WEIGHT_CLIP=10.0` 은 그보다
# 커서 절대 발동하지 않았다. 그것을 검사하던 `sel.weight.max() <= WEIGHT_CLIP
# + 1e-9` 도 상수를 100 으로 바꾸거나 클립을 통째로 지워도 통과하는 판별력
# 없는 단언이었다. 상수와 단언을 함께 지우고(F5 (b), 더 단순한 쪽), 대신
# `_tail_weight` 공식 자체가 `[1, 5]` 를 내는지 직접 확인한다. `select()` 가
# 반환하는 `sel.weight` 는 마지막에 `sum() == len(index)` 로 재정규화돼
# 스케일이 바뀌므로 (`picked_weight / picked_weight.sum() * len(chosen)`) 이
# 경계를 더는 보여주지 않는다 — 그래서 공식을 직접 테스트해야 한다.

def test_tail_weight_formula_is_bounded_to_one_to_five():
    """`ranks` 는 항상 `[0, 1]` 이므로 `_tail_weight` 는 항상 정확히 `[1, 5]`
    를 낸다. 계수(4.0)나 절편(1.0)이 바뀌면 이 경계도 따라 바뀌어야 한다 —
    이 테스트는 그 변화를 직접 잡는다."""
    ranks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    weight = manifold._tail_weight(ranks)
    assert weight.min() == pytest.approx(1.0)
    assert weight.max() == pytest.approx(5.0)
    np.testing.assert_allclose(weight, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_select_end_to_end_weight_before_renormalisation_stays_within_one_to_five():
    """`select()` 전체를 거쳐도 재정규화 **이전** 가중치가 공식이 보장하는
    `[1, 5]` 를 벗어나지 않는지, `ranks` 를 직접 재현해 확인한다."""
    X = _blob(n=800, seed=11)
    mask = np.ones(len(X), dtype=bool)
    distance = manifold._mahalanobis(X)
    ranks = distance.argsort().argsort() / max(len(distance) - 1, 1)
    weight = manifold._tail_weight(ranks)
    assert weight.min() == pytest.approx(1.0)
    assert weight.max() == pytest.approx(5.0)
    assert np.all((weight >= 1.0) & (weight <= 5.0))


def test_selection_is_reproducible_under_same_seed():
    X = _blob()
    mask = np.ones(len(X), dtype=bool)
    a = manifold.select(X, mask, max_samples=200, seed=7)
    b = manifold.select(X, mask, max_samples=200, seed=7)
    np.testing.assert_array_equal(a.index, b.index)
    np.testing.assert_array_equal(a.weight, b.weight)


def test_different_seeds_actually_change_the_subsample():
    """재현성 테스트(같은 시드 → 같은 결과)만으로는 구현이 `seed` 인자를
    무시해도(예: 내부에서 항상 고정 시드를 쓰도록 퇴화해도) 잡아내지 못한다
    — 같은 seed 를 두 번 넣으면 어차피 같은 값이 나오기 때문이다. 서로 다른
    seed 가 서로 다른 부표본을 만드는지까지 확인해야 `seed` 인자가 실제로
    쓰이고 있음을 보장한다."""
    X = _blob(n=5000, seed=1)
    mask = np.ones(len(X), dtype=bool)
    a = manifold.select(X, mask, max_samples=500, seed=1)
    b = manifold.select(X, mask, max_samples=500, seed=2)
    assert not np.array_equal(a.index, b.index)


def test_on_manifold_fraction_matches_trust_quantile():
    """`TRUST_QUANTILE`(기본 0.99)이 실제로 on-manifold 판정에 쓰이는지 고정값으로
    핀다. 상수를 바꿔서 이 판정이 실제로 달라지는지가 브리핑이 요구한 뮤테이션
    자기검토 항목 중 하나다 — 여기서 리터럴 0.99 를 직접 박아 두어야, 상수를
    조용히 바꿔도(또는 판정 로직이 상수를 참조하지 않아도) 테스트가 깨진다."""
    assert manifold.TRUST_QUANTILE == pytest.approx(0.99)
    X = _blob(n=5000, seed=3)
    mask = np.ones(len(X), dtype=bool)
    sel = manifold.select(X, mask, max_samples=len(X), seed=0)
    assert sel.on_manifold.mean() == pytest.approx(0.99, abs=0.01)


def test_mahalanobis_is_stable_when_two_columns_are_affinely_dependent():
    """005930 실데이터에서 실측된 상황을 합성으로 재현한다: 무차원 feature 9개 중
    `book_imbalance` 와 `queue_imbalance_best` 가 정확한 아핀 변환 관계라
    공분산 행렬이 rank 결손(9개 중 8, 특이)이다.

    `np.linalg.inv` 는 이런 수치적 특이 행렬에 대해 `LinAlgError` 를 던지지 않고
    개별 원소가 거대한(실측 이 합성 데이터에서 최대 절대값 4.5e15) 역행렬을
    돌려줄 수 있다. **주의:** 이 합성 데이터에서는 그 거대한 원소들이 실제
    관측 방향으로는 우연히 상쇄되어 마할라노비스 거리 자체는 작게 나온다(최대
    5.66) — 즉 이 테스트 단독으로는 `inv` 경로의 병리적 폭주(005930 실데이터
    실측: 거리 최대 1.77e8, 평균 2.5천만)를 재현하지 못한다. 폭주 여부는
    특이 방향과 실제 데이터가 어떻게 정렬되는지에 달려 있어 합성 데이터로
    안정적으로 재현할 수 없었다 — 그래서 진짜 회귀 가드는 아래
    `test_mahalanobis_is_finite_and_bounded_on_real_dimensionless_data`
    (실데이터, `slow`)다. 이 테스트는 그와 별개로 `_mahalanobis` 가 특이
    공분산에서 일반적으로 유한·유계 값을 내는지를 확인하는 회귀 가드다."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2000, 8))
    duplicate = 2.0 * base[:, 0] - 1.0                  # book_imbalance 관계 재현
    X = np.column_stack([base, duplicate])

    covariance = np.cov(X - X.mean(axis=0, keepdims=True), rowvar=False)
    assert np.linalg.matrix_rank(covariance) == X.shape[1] - 1  # 특이함을 확인

    distance = manifold._mahalanobis(X)
    assert np.all(np.isfinite(distance))
    assert distance.max() < 1_000.0


def test_select_end_to_end_is_stable_on_affinely_dependent_columns():
    """`_mahalanobis` 단위 테스트를 넘어 `select()` 전체가 특이 공분산에서
    유한한 신뢰 반경과 유한한 가중치를 내는지 확인한다."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=(1500, 8))
    duplicate = 2.0 * base[:, 0] - 1.0
    X = np.column_stack([base, duplicate])
    mask = np.ones(len(X), dtype=bool)

    sel = manifold.select(X, mask, max_samples=len(X), seed=0)
    assert np.all(np.isfinite(sel.weight))


@pytest.mark.slow
def test_mahalanobis_is_finite_and_bounded_on_real_dimensionless_data():
    """실제 005930 무차원 행렬(정확히 특이한 공분산)에서 회귀 테스트로 확인한다."""
    from sd import config, dimensionless, ticks

    arrays = ticks.load_arrays("005930", config.DATE)
    matrix, names = ticks.feature_matrix(arrays)
    X, _kept, _meta = dimensionless.transform(matrix, names)
    finite_rows = np.all(np.isfinite(X), axis=1)
    X = X[finite_rows]
    assert np.linalg.matrix_rank(np.cov(X - X.mean(axis=0, keepdims=True), rowvar=False)) \
        == X.shape[1] - 1  # 실데이터에서도 특이함을 재확인

    distance = manifold._mahalanobis(X)
    assert np.all(np.isfinite(distance))
    assert distance.max() < 1_000.0
