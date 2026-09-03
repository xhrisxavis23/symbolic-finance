"""배관 검증용 얕은 교사. 본 실험에서는 DeepLOBCompact 로 교체한다 (DESIGN.md D3).

이 모델의 성능은 의미가 없다. 확인하는 것은 **병목을 가진 교사가 두 헤드를 내고
그 출력이 SR 로 흘러간다**는 계약뿐이다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ShallowMLP:
    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 hidden: int = 32, l1: float = 1e-4) -> None:
        self.bottleneck = int(bottleneck)
        self.l1 = float(l1)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(int(n_features), hidden), nn.ReLU(),
            nn.Linear(hidden, self.bottleneck))
        self._head_path = nn.Linear(self.bottleneck, 1)
        self._head_fill = nn.Linear(self.bottleneck, 1)
        self._mean = np.zeros(int(n_features))
        self._scale = np.ones(int(n_features))

    # -- 학습 ---------------------------------------------------------------
    def fit(self, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
            weight: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> "ShallowMLP":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

        inputs = torch.tensor(self._standardise(X), dtype=torch.float32)
        target_path = torch.tensor(np.asarray(y_path, dtype=float), dtype=torch.float32)
        target_fill = torch.tensor(np.asarray(y_fill, dtype=float), dtype=torch.float32)
        weights = torch.tensor(np.asarray(weight, dtype=float), dtype=torch.float32)

        parameters = (list(self._encoder.parameters())
                      + list(self._head_path.parameters())
                      + list(self._head_fill.parameters()))
        optimiser = torch.optim.Adam(parameters, lr=lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for _ in range(int(epochs)):
            optimiser.zero_grad()
            latent = self._encoder(inputs)
            path = self._head_path(latent).squeeze(-1)
            fill = self._head_fill(latent).squeeze(-1)
            loss_path = (weights * (path - target_path) ** 2).mean()
            loss_fill = (weights * bce(fill, target_fill)).mean()
            # 병목에 L1. Cranmer 방법의 핵심이 이 한 줄이다.
            loss = loss_path + loss_fill + self.l1 * latent.abs().mean()
            loss.backward()
            optimiser.step()
        return self

    # -- 추론 ---------------------------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _latent(self, X: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self._encoder(torch.tensor(self._standardise(X), dtype=torch.float32))

    def z(self, X: np.ndarray) -> np.ndarray:
        return self._latent(X).numpy()

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head_path(self._latent(X)).squeeze(-1).numpy()

    def predict_fill(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self._head_fill(self._latent(X))).squeeze(-1).numpy()
