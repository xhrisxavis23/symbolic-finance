"""배관 검증용 최소 SR 백엔드 (DESIGN.md D4).

템플릿 격자를 훑고 상수만 최소제곱으로 맞춘다. 진짜 탐색이 아니다 —
컴파일러가 다뤄야 할 **수식의 모양**(선형결합·비·sqrt·tanh·곱)을 전부
내보내는 것이 목적이다. 본 실험에서는 PySRBackend 로 교체한다.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import sympy

from .base import Candidate, complexity_of, weighted_r2

# (이름, 심볼 몇 개를 쓰는가, 수식 만드는 함수)
TEMPLATES: tuple[tuple[str, int, Callable[..., sympy.Expr]], ...] = (
    ("linear1", 1, lambda a: a),
    ("neg1", 1, lambda a: -a),
    ("abs1", 1, lambda a: sympy.Abs(a)),
    ("linear2", 2, lambda a, b: a + b),
    ("diff2", 2, lambda a, b: a - b),
    ("product2", 2, lambda a, b: a * b),
    ("ratio2", 2, lambda a, b: a / (1 + sympy.Abs(b))),
    ("sqrt_abs", 1, lambda a: sympy.sqrt(sympy.Abs(a))),
    ("tanh1", 1, lambda a: sympy.tanh(a)),
    ("linear3", 3, lambda a, b, c: a + b + c),
    ("mixed3", 3, lambda a, b, c: a * b + c),
    ("saturated2", 2, lambda a, b: sympy.tanh(a) * b),
)


class NaiveBackend:
    name = "naive"

    def __init__(self, seed: int = 0, max_candidates: int = 12,
                 top_features: int = 4) -> None:
        self.seed = int(seed)
        self.max_candidates = int(max_candidates)
        self.top_features = int(top_features)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        names = tuple(str(n) for n in feature_names)

        ranked = self._rank_features(X, y, w, names)
        symbols = {name: sympy.Symbol(name) for name in names}
        found: list[Candidate] = []

        for _label, arity, build in TEMPLATES:
            for combo in self._combinations(ranked, arity):
                expr = build(*[symbols[name] for name in combo])
                scored = self._score(expr, X, y, w, names)
                if scored is not None:
                    found.append(scored)

        # 같은 정규형은 하나만 남긴다 — 중복은 N 회계를 부풀린다.
        unique: dict[str, Candidate] = {}
        for candidate in found:
            key = sympy.srepr(sympy.simplify(candidate.expr))
            best = unique.get(key)
            if best is None or candidate.in_sample_score > best.in_sample_score:
                unique[key] = candidate

        ordered = sorted(unique.values(),
                         key=lambda c: (-c.in_sample_score, c.complexity))
        top = ordered[: self.max_candidates]
        return sorted(top, key=lambda c: c.complexity)

    # -- 내부 ---------------------------------------------------------------
    def _rank_features(self, X, y, w, names) -> tuple[str, ...]:
        """단변량 가중 상관 절댓값 상위. 결정론적이다.

        F1 수정 (Task 9 리뷰): 계획서 §3 S2 ③ 이 가중치를 SR 손실에 명시적으로
        넘기라고 못 박았는데, 이전 구현은 채점(`weighted_r2`, `_score`)에는
        `w` 를 썼지만 이 feature 선택 단계에서는 `np.corrcoef`(비가중)를 써
        탐색의 첫 관문에서 그 원칙이 무너졌다. on-manifold 표집이 만든
        [1, 5] 범위 가중치를 여기서도 반영해야, 꼬리에서만 신호를 내는
        feature 가 상위 4개에서 부당하게 탈락하지 않는다.
        """
        scores = []
        for j, name in enumerate(names):
            column = X[:, j]
            valid = np.isfinite(column) & np.isfinite(y) & np.isfinite(w)
            if valid.sum() < 10 or np.std(column[valid]) == 0:
                scores.append((0.0, name))
                continue
            x = column[valid]
            yy = y[valid]
            ww = w[valid]
            mx = np.average(x, weights=ww)
            my = np.average(yy, weights=ww)
            cov = np.average((x - mx) * (yy - my), weights=ww)
            sx = np.sqrt(np.average((x - mx) ** 2, weights=ww))
            sy = np.sqrt(np.average((yy - my) ** 2, weights=ww))
            if sx == 0 or sy == 0:
                scores.append((0.0, name))
                continue
            scores.append((abs(float(cov / (sx * sy))), name))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return tuple(name for _score, name in scores[: self.top_features])

    @staticmethod
    def _combinations(ranked: tuple[str, ...], arity: int) -> list[tuple[str, ...]]:
        import itertools
        if arity > len(ranked):
            return []
        return [tuple(c) for c in itertools.combinations(ranked, arity)]

    def _score(self, expr, X, y, w, names) -> Candidate | None:
        """상수 스케일·절편만 최소제곱으로 맞춘다. 구조는 템플릿이 정한다."""
        try:
            function = sympy.lambdify([sympy.Symbol(n) for n in names], expr, "numpy")
            raw = np.asarray(function(*[X[:, j] for j in range(len(names))]), dtype=float)
        except Exception:
            return None
        raw = np.broadcast_to(raw, y.shape).astype(float)
        valid = np.isfinite(raw) & np.isfinite(y) & np.isfinite(w)
        if valid.sum() < 10 or np.std(raw[valid]) == 0:
            return None
        design = np.column_stack([raw[valid], np.ones(valid.sum())])
        sqrt_w = np.sqrt(w[valid])
        coefficients, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y[valid] * sqrt_w,
                                           rcond=None)
        slope, intercept = float(coefficients[0]), float(coefficients[1])
        prediction = np.full_like(y, np.nan)
        prediction[valid] = slope * raw[valid] + intercept
        score = weighted_r2(y, prediction, w)
        if not np.isfinite(score):
            return None
        # 상수는 진입식에서 임계로 흡수되므로 구조에 남기지 않는다 (계획서 S3 단조 흡수).
        return Candidate(expr=expr, complexity=complexity_of(expr),
                         in_sample_score=float(score), backend=self.name, seed=self.seed)
