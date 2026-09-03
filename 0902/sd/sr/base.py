"""SR 백엔드 계약 (DESIGN.md D4).

`backend` 필드가 반드시 채워져야 한다. `naive` 로 나온 진입식은 **어떤 가설
판정에도 쓰지 않는다** — 배관 검증용이다. 리포트가 그것을 눈에 띄게 표시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import sympy


@dataclass(frozen=True)
class Candidate:
    expr: sympy.Expr
    complexity: int
    in_sample_score: float
    backend: str
    seed: int


class SRBackend(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]: ...


def complexity_of(expr: sympy.Expr) -> int:
    """AST 노드 수. Pareto 의 x 축이다."""
    return sum(1 for _ in sympy.preorder_traversal(expr))


def weighted_r2(y: np.ndarray, prediction: np.ndarray, w: np.ndarray) -> float:
    """가중 R². 가중치를 손실에 명시적으로 넘긴다 (계획서 §3 S2 ③)."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(w)
    if valid.sum() < 2:
        return float("-inf")
    y, prediction, w = y[valid], prediction[valid], w[valid]
    mean = float(np.average(y, weights=w))
    total = float(np.average((y - mean) ** 2, weights=w))
    residual = float(np.average((y - prediction) ** 2, weights=w))
    return 1.0 - residual / total if total > 0 else float("-inf")
