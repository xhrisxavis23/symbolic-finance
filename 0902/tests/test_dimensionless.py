import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, dimensionless  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402


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
