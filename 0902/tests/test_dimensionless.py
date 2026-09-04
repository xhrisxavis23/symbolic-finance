import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, derived, dimensionless, ticks  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402
from framework import contract as _contract  # noqa: E402


def _synthetic_book_arrays(n: int = 260, depth: int = 10, seed: int = 0):
    """호가만으로 이뤄진 최소 합성 데이터 (tests/test_derived.py 와 같은 픽스처).
    체결 데이터가 없어도 파생 열은 전부 계산돼야 한다 — 전부 호가 feature 다."""
    rng = np.random.default_rng(seed)
    tick = 5.0
    base = 10_000.0 + np.cumsum(rng.normal(0.0, 3.0, size=n))
    levels = np.arange(depth)
    bid_price = base[:, None] - tick * (levels[None, :] + 1)
    ask_price = base[:, None] + tick * (levels[None, :] + 1)
    bid_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    ask_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    return {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": bid_qty, "ask_qty": ask_qty}


def test_only_dimensionless_features_are_kept():
    for name in dimensionless.DIMENSIONLESS_FEATURES:
        assert catalog.FEATURES[name].type_info.dimension == "dimensionless"
    assert "book_imbalance" in dimensionless.DIMENSIONLESS_FEATURES
    assert "mid_price" not in dimensionless.DIMENSIONLESS_FEATURES
    assert "bid_depth_total_5" not in dimensionless.DIMENSIONLESS_FEATURES


def test_transform_drops_dimensionful_columns_and_records_why():
    names = ("book_imbalance", "mid_price", "queue_imbalance_best")
    matrix = np.arange(12, dtype=float).reshape(4, 3)
    kept_matrix, kept_names, meta = dimensionless.transform(matrix, names)
    assert kept_names == ("book_imbalance", "queue_imbalance_best")
    assert kept_matrix.shape == (4, 2)
    assert meta["dropped"] == {"mid_price": "price"}
    # 열 순서가 이름 순서와 실제로 대응하는지 — 값이 다른 두 열 모두 확인한다.
    # book_imbalance 는 원본 열 0(0,3,6,9), queue_imbalance_best 는 원본 열 2
    # (2,5,8,11) 이고 mid_price(원본 열 1: 1,4,7,10)가 사이에서 빠진다. 열 0만
    # 검사하면 우연히 위치가 맞아도(0번째가 그대로 0번째) 통과해 버리므로
    # 값이 다른 두 번째 열까지 반드시 함께 검사한다.
    np.testing.assert_array_equal(kept_matrix[:, 0], matrix[:, 0])
    np.testing.assert_array_equal(kept_matrix[:, 1], matrix[:, 2])
    assert not np.array_equal(kept_matrix[:, 1], matrix[:, 1])  # mid_price 열이 아님을 확인


def test_transform_is_order_stable():
    names = ("queue_imbalance_best", "book_imbalance")
    matrix = np.zeros((3, 2))
    _kept, kept_names, _meta = dimensionless.transform(matrix, names)
    assert kept_names == tuple(sorted(kept_names))


def test_transform_raises_when_nothing_dimensionless_remains():
    names = ("mid_price", "bid_depth_total_5")
    matrix = np.ones((2, 2))
    try:
        dimensionless.transform(matrix, names)
    except ValueError:
        pass
    else:
        raise AssertionError("무차원 feature 가 없는데도 ValueError 가 나지 않았다")


def test_dropped_records_actual_dimension_not_placeholder():
    names = ("book_imbalance", "bid_depth_total_5")
    _kept, _kept_names, meta = dimensionless.transform(matrix=np.zeros((2, 2)), names=names)
    assert meta["dropped"] == {"bid_depth_total_5": "quantity"}


# ---- 파생 열 (ROADMAP.md 1단계) ------------------------------------------------
#
# `arrays` 를 주지 않으면 위 테스트들처럼 예전과 완전히 같아야 한다 — 이미
# 그렇게 통과했다. 아래는 `arrays` 를 준 새 경로만 다룬다.

def test_transform_without_arrays_never_adds_derived_columns():
    """기존 호출 방식(위치 인자 두 개)은 파생 열을 전혀 계산하지 않는다 —
    `run_slice.py` 를 포함해 기존 호출부가 하나도 안 바뀐 것처럼 동작해야 한다."""
    names = ("book_imbalance", "mid_price", "queue_imbalance_best")
    matrix = np.arange(12, dtype=float).reshape(4, 3)
    _matrix, kept_names, meta = dimensionless.transform(matrix, names)
    assert meta["derived"] == []
    assert meta["derived_failed"] == {}
    assert kept_names == ("book_imbalance", "queue_imbalance_best")


def test_transform_with_arrays_appends_every_registered_derived_column():
    arrays = _synthetic_book_arrays(n=260, seed=0)
    matrix, names = ticks.feature_matrix(arrays)
    result, kept_names, meta = dimensionless.transform(matrix, names, arrays=arrays)

    assert meta["derived_failed"] == {}
    assert set(derived.DERIVED) == set(meta["derived"])
    assert set(derived.DERIVED) <= set(kept_names)
    assert kept_names == tuple(sorted(kept_names))
    assert result.shape == (matrix.shape[0], len(kept_names))


def test_transform_derived_column_values_match_direct_expression_runtime_evaluation():
    """`dimensionless.transform` 이 실제로 그 AST 를 평가한 값을 내는가 — 다른
    계산을 넣어도 이 테스트가 눈치채지 못하면 뮤테이션에 안 잡힌다. 독립적으로
    `ExpressionRuntime` 을 다시 호출해 같은 파생 열을 평가하고 열 단위로 비교한다."""
    arrays = _synthetic_book_arrays(n=260, seed=5)
    matrix, names = ticks.feature_matrix(arrays)
    result, kept_names, _meta = dimensionless.transform(matrix, names, arrays=arrays)

    runtime = _contract.ExpressionRuntime(arrays)
    for name in ("ofi_over_qbar_5", "mid_return_20t_over_spread", "ofi_depth_5_zscore"):
        expected = np.asarray(runtime.evaluate(derived.DERIVED[name]))
        column = result[:, kept_names.index(name)]
        np.testing.assert_array_equal(np.isnan(column), np.isnan(expected))
        np.testing.assert_allclose(column[np.isfinite(column)], expected[np.isfinite(expected)])


def test_transform_column_order_matches_names_for_a_mix_of_raw_and_derived():
    """핵심 계약: 행렬의 열 순서와 반환된 이름 순서가 정확히 대응해야 한다.
    값이 서로 다른 원본 열 하나와 파생 열 하나를 모두 확인한다 — 하나만 보면
    우연히 위치가 맞아도 통과해 버린다 (이 저장소에서 실제로 있었던 실수)."""
    arrays = _synthetic_book_arrays(n=260, seed=6)
    matrix, names = ticks.feature_matrix(arrays)
    result, kept_names, _meta = dimensionless.transform(matrix, names, arrays=arrays)

    raw_idx = kept_names.index("book_imbalance")
    raw_original_idx = names.index("book_imbalance")
    np.testing.assert_array_equal(result[:, raw_idx], matrix[:, raw_original_idx])

    derived_idx = kept_names.index("ofi_over_qbar_5")
    runtime = _contract.ExpressionRuntime(arrays)
    expected_derived = np.asarray(runtime.evaluate(derived.DERIVED["ofi_over_qbar_5"]))
    finite = np.isfinite(expected_derived)
    np.testing.assert_allclose(result[finite, derived_idx], expected_derived[finite])

    # 두 열이 실제로 다른 값이어야 판별력이 있다 — 우연히 같은 값이면 순서가
    # 뒤바뀌어도 이 테스트가 눈치채지 못한다.
    finite_both = np.isfinite(result[:, raw_idx]) & np.isfinite(result[:, derived_idx])
    assert not np.allclose(result[finite_both, raw_idx], result[finite_both, derived_idx])


def test_transform_reports_partial_derived_failures_without_crashing():
    """호가 깊이가 5뿐이면 `ofi_depth_10` 이 필요한 파생 열만 실패하고 나머지는
    성공해야 한다 — 하나가 죽는다고 전체가 죽으면 안 된다 (`compute_features`
    가 계산 불가 feature 를 조용히 빼는 것과 같은 태도)."""
    arrays = _synthetic_book_arrays(n=260, depth=5, seed=7)
    matrix, names = ticks.feature_matrix(arrays)
    _result, kept_names, meta = dimensionless.transform(matrix, names, arrays=arrays)

    assert "ofi_over_qbar_10" in meta["derived_failed"]
    assert "ofi_over_qbar_10" not in meta["derived"]
    assert "ofi_over_qbar_10" not in kept_names
    assert "ofi_over_qbar_5" in meta["derived"]
    assert "ofi_over_qbar_5" in kept_names
