"""무차원 좌표. 계획서 §3 S3.

규칙 하나다 — **차원 있는 feature 는 SR 입력이 될 수 없다.** 종목 스케일이
수식에 들어오면 SR 은 유동성 등급을 설명하는 항을 붙이고 그것을 알파로 착각한다.

Catalog 가 이미 모든 feature 에 `dimension` 을 붙여 두었으므로 여기서 다시
분류하지 않는다. Task 2 에서 `sqrt`·`tanh` 를 무차원 입력만 받게 만든 것도
같은 규칙을 타입 수준에서 강제한 것이다.

슬라이스에서는 **이미 무차원인 feature 만** 쓴다. 차원 있는 feature 를
`ratio` 나 `rolling_zscore` 로 무차원화해 파생 열을 만드는 것은 다음 단계다.
"""

from __future__ import annotations

import numpy as np

from . import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402

DIMENSIONLESS_FEATURES: tuple[str, ...] = tuple(sorted(
    name for name, spec in _catalog.FEATURES.items()
    if spec.type_info.dimension == "dimensionless"
))


def transform(matrix: np.ndarray,
              names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """무차원 열만 남긴다. 무엇을 왜 뺐는지 함께 돌려준다."""
    keep = tuple(sorted(n for n in names if n in set(DIMENSIONLESS_FEATURES)))
    dropped = {n: _catalog.FEATURES[n].type_info.dimension
               for n in names if n not in set(DIMENSIONLESS_FEATURES)}
    if not keep:
        raise ValueError("무차원 feature 가 하나도 남지 않았다")
    index = [names.index(n) for n in keep]
    meta = {"kept": list(keep), "dropped": dropped,
            "rule": "Catalog dimension == 'dimensionless' 인 feature 만 SR 입력"}
    return matrix[:, index], keep, meta
