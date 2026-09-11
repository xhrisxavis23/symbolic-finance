"""`sd/teacher/deeplob.py` 의 GPU 실험용 사본. 0911 판정 R40 전용.

**0902/ 는 건드리지 않는다.** README.md 가 명시적으로 허용한 예외 조항
("GPU 평가를 위해 교사 코드를 복사해 0911 안에서 실험하는 것은 됩니다")에
따라 이 파일을 만들었다. 원본과의 유일한 차이는 `device` 파라미터를 받아
텐서·모듈을 그 장치로 옮기는 것뿐이다 — 아키텍처(레이어 크기·순서),
초기화 순서(`torch.manual_seed` 를 모듈 생성 **전에** 호출), 손실 함수,
옵티마이저, 학습 루프(`run_with_early_stopping`)는 원본과 한 글자도 다르지
않다.

**왜 초기 가중치가 CPU/GPU 사이에 동일한가:** `torch.manual_seed(seed)` 는
파라미터 초기화에 쓰는 RNG 상태를 고정한다. `nn.Linear`/`nn.Conv1d`/`nn.LSTM`
생성자는 항상 CPU 텐서로 초기화되고(파이토치 기본 동작), 그 다음에
`.to(device)` 로 값을 복사할 뿐이다 — 값 자체는 바뀌지 않는다. 그래서 같은
시드로 만든 CPU 모델과 GPU 모델은 **학습 시작 전 가중치가 bit-for-bit
동일**하다. 학습 후 갈라진다면 그 원인은 순전파/역전파의 수치 연산
(cuBLAS/cuDNN vs MKL, 리덕션 순서 등) 자체이지 초기화가 아니다 — 이 구분이
판정 R40 의 핵심이다.
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


class _TemporalConvBlock(nn.Module):
    def __init__(self, n_features: int, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class _InceptionLite(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        c = branch_channels
        self.branch1 = nn.Conv1d(in_channels, c, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.Conv1d(c, c, kernel_size=3, padding=1))
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),
            nn.Conv1d(c, c, kernel_size=5, padding=2))
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c, kernel_size=1))
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)]
        return self.act(torch.cat(parts, dim=1))


class _DeepLOBEncoder(nn.Module):
    def __init__(self, n_features: int, bottleneck: int, conv_channels: int,
                 inception_channels: int, lstm_hidden: int) -> None:
        super().__init__()
        self.conv = _TemporalConvBlock(n_features, conv_channels)
        self.inception = _InceptionLite(conv_channels, inception_channels)
        lstm_input = inception_channels * 4
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden, batch_first=True)
        self.to_bottleneck = nn.Linear(lstm_hidden, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.inception(h)
        h = h.permute(0, 2, 1)
        _output, (h_n, _c_n) = self.lstm(h)
        last = h_n[-1]
        return self.to_bottleneck(last)


class DeepLOBCompactGPU:
    """`sd.teacher.deeplob.DeepLOBCompact` 와 계약이 같다. 유일한 추가 인자는
    `device`(`"cpu"` 또는 `"cuda"`)."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 window: int = 16, conv_channels: int = 8,
                 inception_channels: int = 4, lstm_hidden: int = 16,
                 l1: float = 1e-4, device: str = "cpu") -> None:
        self.bottleneck = int(bottleneck)
        self.window = int(window)
        self.n_features = int(n_features)
        self.l1 = float(l1)
        self._seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self._seed)          # 원본과 동일 — 모듈 생성 전에 시드
        self._encoder = _DeepLOBEncoder(
            n_features=self.n_features, bottleneck=self.bottleneck,
            conv_channels=conv_channels, inception_channels=inception_channels,
            lstm_hidden=lstm_hidden).to(self.device)   # 값 복사만, 재초기화 아님
        self._head_path = nn.Linear(self.bottleneck, 1).to(self.device)
        self._head_fill = nn.Linear(self.bottleneck, 1).to(self.device)
        flat_dim = self.window * self.n_features
        self._mean = np.zeros(flat_dim)
        self._scale = np.ones(flat_dim)

    def fit(self, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
            weight: np.ndarray, epochs: int = 200, lr: float = 1e-2, *,
            X_path_select: np.ndarray | None = None,
            y_path_select: np.ndarray | None = None,
            weight_select: np.ndarray | None = None,
            eval_every: int = EVAL_EVERY_DEFAULT,
            patience: int = PATIENCE_DEFAULT) -> "DeepLOBCompactGPU":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._check_shape(X)
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

        def train_step() -> None:
            optimiser.zero_grad()
            latent = self._encode(inputs)
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

    def _check_shape(self, X: np.ndarray) -> None:
        expected = self.window * self.n_features
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(
                f"X 열 수가 window*n_features 와 다르다: got {X.shape}, "
                f"expected (n, {expected})")

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        x = inputs.view(batch, self.window, self.n_features).permute(0, 2, 1)
        return self._encoder(x)

    def _latent(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=float)
        self._check_shape(X)
        with torch.no_grad():
            inputs = torch.tensor(self._standardise(X), dtype=torch.float32, device=self.device)
            return self._encode(inputs)

    def z(self, X: np.ndarray) -> np.ndarray:
        return self._latent(X).cpu().numpy()

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head_path(self._latent(X)).squeeze(-1).cpu().numpy()

    def predict_fill(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self._head_fill(self._latent(X))).squeeze(-1).cpu().numpy()
