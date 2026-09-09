"""배관 검증용 얕은 교사. 본 실험에서는 DeepLOBCompact 로 교체한다 (DESIGN.md D3).

이 모델의 성능은 의미가 없다. 확인하는 것은 **병목을 가진 교사가 두 헤드를 내고
그 출력이 SR 로 흘러간다**는 계약뿐이다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..sr.base import weighted_r2
from .early_stopping import run_with_early_stopping, torch_snapshot_functions

# A10/PREREG-T1.md 조기 종료 기본값 — `sd.e0.teacher` 의 `EVAL_EVERY_DEFAULT`/
# `PATIENCE_DEFAULT` 와 의도적으로 같은 값(같은 근거, 그 파일 docstring 참고).
# 이 클래스는 이번 T1 실행(run_e0.py)이 실제로 쓰지 않는다 — S0/S1 파이프라인
# (`run_slice.py`)용이고, `run_slice.py` 는 아직 select 데이터를 안 넘기므로
# `X_path_select=None` 기본값 그대로 켜지지 않는다(기존 동작 보존). 그래도
# PREREG-T1.md §1 이 "구현 위치: ShallowMLP·DeepLOBCompact 공통"이라고 못박아
# 여기도 같은 능력을 넣는다.
EVAL_EVERY_DEFAULT = 10
PATIENCE_DEFAULT = 10


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
            weight: np.ndarray, epochs: int = 200, lr: float = 1e-2, *,
            X_path_select: np.ndarray | None = None,
            y_path_select: np.ndarray | None = None,
            weight_select: np.ndarray | None = None,
            eval_every: int = EVAL_EVERY_DEFAULT,
            patience: int = PATIENCE_DEFAULT) -> "ShallowMLP":
        """`X_path_select`/`y_path_select`/`weight_select` 를 주면 A10 조기
        종료를 켠다(경로 헤드 R² 기준 — 체결확률 헤드는 BCE 확률이라 회귀
        R² 개념이 없다, `ScalarTeacher.fit` 과 같은 자세). 안 주면(기본)
        예전과 완전히 같은 전체-epoch 학습이다."""
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

        def train_step() -> None:
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

        evaluate = None
        if X_path_select is not None:
            y_select_arr = np.asarray(y_path_select, dtype=float)
            weight_select_arr = np.asarray(weight_select, dtype=float)
            evaluate = lambda: weighted_r2(       # noqa: E731
                y_select_arr, self.predict_path(X_path_select), weight_select_arr)

        get_state, set_state = torch_snapshot_functions({
            "encoder": self._encoder, "head_path": self._head_path,
            "head_fill": self._head_fill})
        self.early_stop_history_ = run_with_early_stopping(
            total_epochs=int(epochs), eval_every=int(eval_every), patience=int(patience),
            train_step=train_step, evaluate=evaluate, get_state=get_state, set_state=set_state)
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
