"""`sd/teacher/shallow.py` 의 GPU 실험용 사본 — 공수 비교용. `deeplob_gpu.py` 와
같은 방식(`device` 파라미터만 추가, 그 외 전부 동일)이다.

ShallowMLP 는 배관 검증용이라 본 실험 R² 곡선에는 안 쓰지만, "교사 코드를
GPU 로 옮기는 공수"를 두 파일 모두에서 실측하라는 지시(README.md 작업2)에
따라 만들었다. 실제로 옮겨 보니 `deeplob_gpu.py` 와 똑같은 패턴 — 생성자에
`device` 인자 추가, `.to(device)`, `torch.tensor(..., device=device)` 뿐이다.
코드 줄 수 기준 변경량: 원본 대비 +6줄(디바이스 인자 1개 + `.to()`/`device=`
호출 5곳), 삭제 0줄. `DeepLOBCompact` 도 동일한 규모(+7줄)였다 — **두 파일
모두 GPU 이전 자체의 공수는 사소하다**(구조 변경 없음, 순수 배선).
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
from sd.sr.base import weighted_r2  # noqa: E402
from sd.teacher.early_stopping import run_with_early_stopping, torch_snapshot_functions  # noqa: E402

EVAL_EVERY_DEFAULT = 10
PATIENCE_DEFAULT = 10


class ShallowMLPGPU:
    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 hidden: int = 32, l1: float = 1e-4, device: str = "cpu") -> None:
        self.bottleneck = int(bottleneck)
        self.l1 = float(l1)
        self._seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(int(n_features), hidden), nn.ReLU(),
            nn.Linear(hidden, self.bottleneck)).to(self.device)
        self._head_path = nn.Linear(self.bottleneck, 1).to(self.device)
        self._head_fill = nn.Linear(self.bottleneck, 1).to(self.device)
        self._mean = np.zeros(int(n_features))
        self._scale = np.ones(int(n_features))

    def fit(self, X, y_path, y_fill, weight, epochs=200, lr=1e-2, *,
            X_path_select=None, y_path_select=None, weight_select=None,
            eval_every=EVAL_EVERY_DEFAULT, patience=PATIENCE_DEFAULT):
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

        inputs = torch.tensor(self._standardise(X), dtype=torch.float32, device=self.device)
        target_path = torch.tensor(np.asarray(y_path, dtype=float), dtype=torch.float32,
                                   device=self.device)
        target_fill = torch.tensor(np.asarray(y_fill, dtype=float), dtype=torch.float32,
                                   device=self.device)
        weights = torch.tensor(np.asarray(weight, dtype=float), dtype=torch.float32,
                               device=self.device)

        parameters = (list(self._encoder.parameters())
                      + list(self._head_path.parameters())
                      + list(self._head_fill.parameters()))
        optimiser = torch.optim.Adam(parameters, lr=lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        def train_step():
            optimiser.zero_grad()
            latent = self._encoder(inputs)
            path = self._head_path(latent).squeeze(-1)
            fill = self._head_fill(latent).squeeze(-1)
            loss_path = (weights * (path - target_path) ** 2).mean()
            loss_fill = (weights * bce(fill, target_fill)).mean()
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

    def _standardise(self, X):
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _latent(self, X):
        with torch.no_grad():
            return self._encoder(torch.tensor(self._standardise(X), dtype=torch.float32,
                                              device=self.device))

    def z(self, X):
        return self._latent(X).cpu().numpy()

    def predict_path(self, X):
        with torch.no_grad():
            return self._head_path(self._latent(X)).squeeze(-1).cpu().numpy()

    def predict_fill(self, X):
        with torch.no_grad():
            return torch.sigmoid(self._head_fill(self._latent(X))).squeeze(-1).cpu().numpy()
