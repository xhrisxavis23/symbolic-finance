"""교사 계약. 내부는 교체 가능하되 이 세 메서드는 고정이다 (DESIGN.md D5).

`z` 는 병목이고 진단·H4 ablation 에만 쓴다. **SR 의 입력이 아니다** — `z` 는
Catalog feature 가 아니라 백테스트가 계산할 수 없기 때문이다 (D13).
SR 은 `predict_path` 를 타깃으로 삼아 feature 공간에서 적합한다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Teacher(Protocol):
    bottleneck: int

    def z(self, X: np.ndarray) -> np.ndarray: ...
    def predict_path(self, X: np.ndarray) -> np.ndarray: ...
    def predict_fill(self, X: np.ndarray) -> np.ndarray: ...
