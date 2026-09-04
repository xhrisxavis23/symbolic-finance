import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import ablation  # noqa: E402


def test_uniform_box_sample_respects_percentile_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 2))
    sample = ablation.uniform_box_sample(X, n=2000, seed=1)
    lo = np.percentile(X, ablation.BOX_PERCENTILE, axis=0)
    hi = np.percentile(X, 100 - ablation.BOX_PERCENTILE, axis=0)
    assert sample.shape == (2000, 2)
    assert np.all(sample >= lo - 1e-9)
    assert np.all(sample <= hi + 1e-9)


def test_uniform_box_sample_ignores_joint_correlation():
    """핵심 성질: 입력이 강한 상관을 가져도(예: x1 ≈ x0) 합성 표본은 그 상관을
    지키지 않는다 — 이것이 곧 off-manifold 를 의도적으로 만드는 지점이다.
    상관을 지키면(예: 원본 행을 그대로 재표집하면) 이 테스트가 잡는다."""
    rng = np.random.default_rng(0)
    n = 5000
    x0 = rng.normal(size=n)
    x1 = x0 + rng.normal(scale=0.01, size=n)   # 거의 완전 상관
    X = np.column_stack([x0, x1])
    sample = ablation.uniform_box_sample(X, n=n, seed=2)

    original_corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    synthetic_corr = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
    assert original_corr > 0.99
    assert synthetic_corr < 0.3


def test_uniform_box_sample_is_seed_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    a = ablation.uniform_box_sample(X, n=500, seed=7)
    b = ablation.uniform_box_sample(X, n=500, seed=7)
    np.testing.assert_array_equal(a, b)


def test_uniform_box_sample_different_seeds_differ():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    a = ablation.uniform_box_sample(X, n=500, seed=1)
    b = ablation.uniform_box_sample(X, n=500, seed=2)
    assert not np.allclose(a, b)


def test_uniform_box_sample_ignores_non_finite_reference_rows():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0], [2.0, 2.0]])
    sample = ablation.uniform_box_sample(X, n=100, seed=0)
    assert np.all(np.isfinite(sample))


def test_uniform_box_sample_raises_when_no_finite_rows():
    X = np.full((5, 2), np.nan)
    try:
        ablation.uniform_box_sample(X, n=10, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("유한 행이 없는데 예외가 안 났다")
