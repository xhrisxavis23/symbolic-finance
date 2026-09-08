"""E0 전용 단일-출력 과잉용량 교사. 계획서 §4 E0 · §3 S1.

`sd.teacher.shallow.ShallowMLP` 는 병목에서 두 헤드(path/fill)를 강제로
갈라내는 구조다 — S0 의 "청산이 타깃을 정의한다"는 진입식 특유의 분해다.
E0 는 진입식을 만들지 않는다(태스크 지시 "스칼라 함수 복원만 본다"). 다섯
법칙 각각은 체결확률이라는 개념 자체가 없는 스칼라 회귀이므로, `y_fill` 을
억지로 채우면(예: `y_path` 를 재사용) BCE 손실이 인코더에 이 법칙과 무관한
기울기를 얹어 "교사가 이 함수 하나를 얼마나 깨끗이 배웠는가"라는 E0 의
질문 자체를 오염시킨다. 그래서 헤드가 하나뿐인 얇은 변형을 새로 둔다.

**이것이 재사용 원칙 위반이 아닌 이유.** 다시 짜는 것은 신경망 학습
루프(표준 MLP + Adam + L1 페널티)이지, 금융 산술이 아니다 — OFI·마이크로
프라이스·무차원화·on-manifold 표집·SR 은 전부 기존 모듈(`sd.ticks`,
`sd.dimensionless`, `sd.derived`, `sd.manifold`, `sd.sr`)에서 그대로 온다.
아키텍처 패턴(표준화 → Linear-ReLU-Linear 인코더 → 병목에 L1 → 선형 헤드)은
`ShallowMLP` 와 의도적으로 같게 맞췄다 — 같은 회사에서 결과를 비교할 때
"헤드가 하나냐 둘이냐"만 차이여야 하기 때문이다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn


class ScalarTeacherProtocol(Protocol):
    """`sd.e0.runner.run_law` 의 `teacher_cls` 가 요구하는 계약.

    `sd.teacher.base.Teacher`(D5, `predict_path`/`predict_fill` 두 헤드)와는
    다른 계약이다 — E0 는 헤드가 하나뿐인 스칼라 회귀만 본다(이 파일 상단
    docstring). `run_law` 는 이 계약만 지키면 어떤 교사 구현이든(예: 계획된
    `DeepLOBCompact`, PREREG-E0-V2.md §1-2 — 다른 작업자 담당) 인자로 갈아
    끼울 수 있다 — 그래서 `ScalarTeacher` 를 여기 직접 박아 넣지 않고
    `runner.run_law(..., teacher_cls=...)` 로 주입받는다."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray,
            epochs: int = 300) -> "ScalarTeacherProtocol": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def z(self, X: np.ndarray) -> np.ndarray: ...


class ScalarTeacher:
    """`X -> y` 스칼라 회귀. `bottleneck` 은 S1 이 규정한 정규화 장치이지 SR
    입력이 아니다(`z()` 는 진단용). SR 은 `predict(X)` 를 타깃으로 원래 feature
    공간에서 적합한다 — `ShallowMLP` 와 같은 설계(D13)."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 hidden: int = 32, l1: float = 1e-4) -> None:
        self.bottleneck = int(bottleneck)
        self.l1 = float(l1)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(int(n_features), hidden), nn.ReLU(),
            nn.Linear(hidden, self.bottleneck))
        self._head = nn.Linear(self.bottleneck, 1)
        self._mean = np.zeros(int(n_features))
        self._scale = np.ones(int(n_features))

    def fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray,
            epochs: int = 300, lr: float = 1e-2) -> "ScalarTeacher":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

        inputs = torch.tensor(self._standardise(X), dtype=torch.float32)
        target = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        weights = torch.tensor(np.asarray(weight, dtype=float), dtype=torch.float32)

        parameters = list(self._encoder.parameters()) + list(self._head.parameters())
        optimiser = torch.optim.Adam(parameters, lr=lr)

        for _ in range(int(epochs)):
            optimiser.zero_grad()
            latent = self._encoder(inputs)
            prediction = self._head(latent).squeeze(-1)
            loss = (weights * (prediction - target) ** 2).mean() + self.l1 * latent.abs().mean()
            loss.backward()
            optimiser.step()
        return self

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _latent(self, X: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self._encoder(torch.tensor(self._standardise(X), dtype=torch.float32))

    def z(self, X: np.ndarray) -> np.ndarray:
        return self._latent(X).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head(self._latent(X)).squeeze(-1).numpy()


class ScalarDeepLOB:
    """`sd.teacher.deeplob.DeepLOBCompact` 의 인코더(CNN(시간) → Inception-lite
    → LSTM → 병목)를 재사용하되 헤드가 **하나뿐**인 E0 전용 교사.

    `DeepLOBCompact` 를 그대로 `teacher_cls` 에 꽂지 않는 이유는 `ScalarTeacher`
    가 `ShallowMLP` 를 그대로 안 쓰는 이유와 정확히 같다(이 파일 상단 모듈
    docstring) — `DeepLOBCompact.fit` 은 `predict_fill` 용 BCE 타깃을 강제로
    요구하는데, E0 의 다섯 법칙 중 어느 것도 체결확률 개념이 없다. 억지로
    채운 `y_fill` 이 공유 인코더에 이 법칙과 무관한 그래디언트를 얹는다.

    새로 짜는 것은 이 파일의 `ScalarTeacher` 와 똑같이 "헤드 하나짜리 학습
    루프"뿐이다 — CNN/Inception/LSTM 자체는 `sd.teacher.deeplob._DeepLOBEncoder`
    를 그대로 가져다 쓴다(새 시계열 아키텍처를 다시 짜지 않는다, `deeplob.py`
    는 한 바이트도 바꾸지 않는다).

    **입력 계약이 `ScalarTeacher` 와 다르다.** `X` 는 이미 `sd.teacher.window.
    make_causal_windows` 로 만든 `(n, window*n_features)` 평평한 행렬이어야
    한다 — `DeepLOBCompact` 와 동일(모듈 docstring 참고). 윈도우를 스스로
    만들지 않는다: 표집(subsample) 이후의 행렬에 씌우면 "시간"이라는 말이
    거짓이 되기 때문이다(`sd/teacher/window.py` 모듈 docstring) — 그래서 호출자
    (`sd.e0.runner.run_law`)가 표집 **이전**의 연속 행렬에 윈도우를 씌워서
    넘겨야 한다."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 window: int = 16, conv_channels: int = 8,
                 inception_channels: int = 4, lstm_hidden: int = 16,
                 l1: float = 1e-4) -> None:
        from ..teacher.deeplob import _DeepLOBEncoder  # 여기서만 import — CNN/LSTM 코드를 재사용

        self.bottleneck = int(bottleneck)
        self.window = int(window)
        self.n_features = int(n_features)          # 한 시점당(윈도우 이전) feature 수
        self.l1 = float(l1)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._encoder = _DeepLOBEncoder(
            n_features=self.n_features, bottleneck=self.bottleneck,
            conv_channels=conv_channels, inception_channels=inception_channels,
            lstm_hidden=lstm_hidden)
        self._head = nn.Linear(self.bottleneck, 1)
        flat_dim = self.window * self.n_features
        self._mean = np.zeros(flat_dim)
        self._scale = np.ones(flat_dim)

    def _check_shape(self, X: np.ndarray) -> None:
        expected = self.window * self.n_features
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(
                f"X 열 수가 window*n_features 와 다르다: got {X.shape}, "
                f"expected (n, {expected}) (window={self.window}, "
                f"n_features={self.n_features}). sd.teacher.window."
                "make_causal_windows 로 먼저 시간 윈도우를 만들었는지 확인하라")

    def fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray,
            epochs: int = 300, lr: float = 1e-2) -> "ScalarDeepLOB":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._check_shape(X)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

        inputs = torch.tensor(self._standardise(X), dtype=torch.float32)
        target = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        weights = torch.tensor(np.asarray(weight, dtype=float), dtype=torch.float32)

        parameters = list(self._encoder.parameters()) + list(self._head.parameters())
        optimiser = torch.optim.Adam(parameters, lr=lr)

        for _ in range(int(epochs)):
            optimiser.zero_grad()
            latent = self._encode(inputs)
            prediction = self._head(latent).squeeze(-1)
            loss = (weights * (prediction - target) ** 2).mean() + self.l1 * latent.abs().mean()
            loss.backward()
            optimiser.step()
        return self

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        # 평평한 (batch, window*F) -> (batch, window, F) -> (batch, F, window) —
        # make_causal_windows 가 낸 순서(옛것→최신, C-order reshape)와 맞다
        # (deeplob.py::DeepLOBCompact._encode 와 동일한 배선).
        x = inputs.view(batch, self.window, self.n_features).permute(0, 2, 1)
        return self._encoder(x)

    def _latent(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=float)
        self._check_shape(X)
        with torch.no_grad():
            inputs = torch.tensor(self._standardise(X), dtype=torch.float32)
            return self._encode(inputs)

    def z(self, X: np.ndarray) -> np.ndarray:
        return self._latent(X).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head(self._latent(X)).squeeze(-1).numpy()
