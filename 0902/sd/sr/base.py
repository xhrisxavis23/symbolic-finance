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


def score_candidate(candidate: Candidate, feature_names: Sequence[str],
                    X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """`candidate.expr` 을 `(X, feature_names)` 에서 평가해 `(y, w)` 에 대한
    `weighted_r2` 를 낸다.

    `candidate.in_sample_score` 는 그 후보를 **적합한** 데이터에서 이미 굳어진
    값이다 — 다른 데이터(예: 표본외 선택 종목)에서 다시 채점하려면 식을 다시
    평가해야 한다. E0 표본외 후보 선택(PREREG-E0-V2.md §1-1)이 이 함수로
    "선택 종목" 점수를 만든다.

    `pysr_backend._extract_candidate` 와 같은 lambdify→broadcast 패턴이지만
    PySR 을 몰라도 되는 순수 함수다 — 평가가 실패하면(정의역 밖 등)
    `-inf`(가장 나쁜 점수, `weighted_r2` 가 유효 표본 부족일 때 이미 쓰는
    관례와 같다) 를 돌려주지 예외를 올리지 않는다. 후보가 새 데이터에서
    죽는 것 자체가 "표본외로 일반화하지 않는다"는 판정의 일부이지 버그가
    아니다.
    """
    try:
        symbols = [sympy.Symbol(n) for n in feature_names]
        fn = sympy.lambdify(symbols, candidate.expr, "numpy")
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(*[X[:, j] for j in range(len(feature_names))]), dtype=float)
        prediction = np.broadcast_to(raw, np.asarray(y).shape).astype(float)
    except Exception:  # noqa: BLE001 — 평가 실패는 -inf 로, 예외로 죽지 않는다
        return float("-inf")
    return weighted_r2(y, prediction, w)
