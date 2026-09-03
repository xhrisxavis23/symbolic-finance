"""틱 적재와 feature 행렬. 계산은 vendor Catalog 가 한다 — 여기서 다시 짜지 않는다.

feature 정의를 우리 쪽에서 구현하면 백테스트가 쓰는 정의와 조용히 갈라진다.
그러면 진입식이 학습 때와 재생 때 다른 값을 본다.
"""

from __future__ import annotations

import numpy as np

from . import config

_framework = config.load_framework()
from framework import catalog as _catalog        # noqa: E402
from framework import data as _data              # noqa: E402

# 열 순서를 이름 정렬로 고정한다. 종목마다 순서가 달라지면 행렬이 뜻을 잃는다.
FEATURE_ORDER: tuple[str, ...] = tuple(sorted(_catalog.FEATURES))


def load_arrays(symbol: str, date: str) -> dict[str, np.ndarray]:
    """한 종목-일의 원본 배열. 정규장 구간만. 캐시는 쓰지 않는다 (DESIGN.md D10)."""
    arrays, _source = _data.load(symbol, date, root=config.TICK_ROOT, cache_root=None)
    return arrays


def feature_matrix(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    """`(n, k)` 행렬과 열 이름.

    체결 행이 없는 종목-일에서는 체결 의존 feature 가 계산되지 않는다. 0 으로
    채우지 않고 **열에서 뺀다** — 0 은 "매도 압력이 없었다" 는 관측을 지어내는 것이다.
    """
    book = _catalog.Book(arrays)
    computed = _catalog.compute_features(FEATURE_ORDER, book)
    names = tuple(sorted(computed))
    if not names:
        raise ValueError("계산된 feature 가 하나도 없다")
    matrix = np.column_stack([np.asarray(computed[name], dtype=float) for name in names])
    return matrix, names
