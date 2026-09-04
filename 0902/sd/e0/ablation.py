"""E0 에서 미리 돌리는 세 ablation. 계획서: "증류 없이 원데이터 직접 SR,
무차원화 없이 증류, on-manifold 없이 균등 샘플링. E0 에서 이미 세 기여의
가치가 측정된다."

세 ablation은 각각 본 트랙(main)에서 **정확히 하나만** 바꾼다:

| 트랙 | 입력 X | 타깃 y | 표집 |
| --- | --- | --- | --- |
| `main` | 무차원 | 교사 증류 | on-manifold |
| `no_distillation` | 무차원 | **실측 y** | on-manifold |
| `no_dimensionless` | **raw** | 교사 증류(raw 교사) | on-manifold(raw 공간) |
| `uniform_off_manifold` | 무차원 | 교사 증류 | **균등 박스 합성** |

`uniform_box_sample` 만 이 파일의 새 로직이다 — 나머지 세 트랙은 이미 있는
데이터(`Dataset.X_dimless/y_dimless/X_raw/y_raw`)와 교사 출력을 조합하는
것뿐이라 `runner.py` 가 직접 만든다.
"""

from __future__ import annotations

import numpy as np

BOX_PERCENTILE = 1.0    # 각 열의 [1, 99] 분위를 박스 경계로 쓴다 — 극단 이상치 배제.


def uniform_box_sample(X_reference: np.ndarray, n: int, seed: int) -> np.ndarray:
    """관측된 결합분포를 무시하고 **열마다 독립** 균등분포에서 뽑는다.

    이것이 바로 계획서 §3 S2 가 경고하는 off-manifold 아티팩트를 의도적으로
    재현하는 절차다 — "스프레드만 5배 키우고 잔량을 고정하면 실존하지 않는
    호가창이 된다"의 저차원 버전. 열이 두 개 이상이면 결합 표본 다수가
    실제로 관측된 적 없는 조합이 된다.
    """
    finite_rows = np.all(np.isfinite(X_reference), axis=1)
    reference = X_reference[finite_rows]
    if reference.shape[0] == 0:
        raise ValueError("균등 표집 기준이 될 유한 행이 없다")
    lo = np.nanpercentile(reference, BOX_PERCENTILE, axis=0)
    hi = np.nanpercentile(reference, 100.0 - BOX_PERCENTILE, axis=0)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lo, hi, size=(int(n), reference.shape[1]))
