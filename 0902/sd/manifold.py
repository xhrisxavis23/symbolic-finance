"""On-manifold 표집. 계획서 §3 S2.

원칙 한 줄 — **넷이 모르는 곳의 답을 수식으로 만들지 않는다.**

슬라이스에서는 세 선택지 중 가장 안전한 것을 쓴다 (계획서 §3 S2 ①):
관측을 재사용하고 합성 상태를 만들지 않는다. 관측이 2,860만 호가틱이므로
밀도가 부족한 것은 전체가 아니라 꼬리뿐이고, 꼬리는 가중치로 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRUST_QUANTILE = 0.99      # 마할라노비스 거리 이 분위 안이면 on-manifold
WEIGHT_CLIP = 10.0         # 꼬리 가중치 상한. 한 점이 손실을 지배하지 않게


@dataclass(frozen=True)
class Selection:
    index: np.ndarray
    weight: np.ndarray
    on_manifold: np.ndarray


def _mahalanobis(X: np.ndarray) -> np.ndarray:
    """공분산 구조를 반영한 중심 거리.

    **항상 유사역행렬(`pinv`)을 쓴다.** 브리핑 초안은 `np.linalg.inv` 를 먼저
    시도하고 `LinAlgError` 에서만 `pinv` 로 물러났는데, 이 슬라이스의 무차원
    행렬로 실측한 결과 그 가드는 절대 걸리지 않는다:

    9개 무차원 feature 중 `book_imbalance = (Q_b − Q_a)/(Q_b + Q_a)` 와
    `queue_imbalance_best = Q_b/(Q_b + Q_a)` 가 정확한 아핀 변환
    (`book_imbalance = 2·queue_imbalance_best − 1`, 최대 절대오차 1.11e-16)
    이라 공분산 행렬이 수치적으로 특이하다 (rank 8/9, 조건수 ~1.2e17).

    `np.linalg.inv` 는 이런 수치적 특이 행렬에서 예외를 던지지 않고 거대한
    쓰레기 값을 조용히 돌려준다 — 005930 실측: `inv` 최대 절대값 1.27e32,
    그 결과 마할라노비스 거리 최대 1.77e8(평균 2,460만). 같은 데이터에서
    `pinv` 는 최대 절대값 2,172, 거리는 0.27~32.6 범위로 유한하고 합리적이다.
    `inv`→`LinAlgError`→`pinv` 가드는 예외가 나지 않으므로 절대 `pinv` 로
    물러나지 못하고, 신뢰 반경이 조용히 오염된다. 그래서 분기 없이 `pinv` 를
    무조건 쓴다.
    """
    centered = X - X.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    covariance = np.atleast_2d(covariance)
    inverse = np.linalg.pinv(covariance)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0.0))


def select(X: np.ndarray, mask: np.ndarray, max_samples: int, seed: int) -> Selection:
    """학습에 쓸 행과 가중치.

    `probe`(신뢰 반경 밖)는 버리지 않고 플래그만 단다 — H2 검증에 통제된
    외삽 질의가 필요하기 때문이다. 다만 **SR 적합에는 쓰지 않는다.**
    """
    X = np.asarray(X, dtype=float)
    usable = np.flatnonzero(mask & np.all(np.isfinite(X), axis=1))
    if usable.size == 0:
        raise ValueError("사용 가능한 행이 없다")

    distance = _mahalanobis(X[usable])
    radius = float(np.quantile(distance, TRUST_QUANTILE))
    on_manifold = distance <= radius

    # 층화 가중치: 밀도가 낮은 곳(거리 상위)에 더 큰 가중치. 상한으로 자른다.
    ranks = distance.argsort().argsort() / max(len(distance) - 1, 1)
    weight = np.clip(1.0 + 4.0 * ranks, 1.0, WEIGHT_CLIP)

    if usable.size > max_samples:
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(usable.size, size=int(max_samples), replace=False))
    else:
        chosen = np.arange(usable.size)

    picked_weight = weight[chosen]
    picked_weight = picked_weight / picked_weight.sum() * len(chosen)
    return Selection(index=usable[chosen], weight=picked_weight,
                     on_manifold=on_manifold[chosen])
