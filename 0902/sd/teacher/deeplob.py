"""압축한 DeepLOB 계열 교사. PREREG-E0-V2.md §1-2, DESIGN.md D5.

원 DeepLOB(Zhang, Zohren & Roberts, 2019, *DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books*)은 `(100 스텝 × 10 레벨 × {가격,수량})` 텐서를
CNN(레벨 축 압축) → Inception 모듈 → LSTM(64) 으로 3-class(상승/보합/하락)
분류한다. 이 저장소에 맞춰 압축하며 뺀 것 — **무엇을 왜 뺐는지**:

1. **레벨-압축 컨볼루션을 통째로 뺐다.** 이 교사가 받는 입력은 원시
   (가격,수량) 페어가 레벨 순서로 나열된 텐서가 아니라, `sd.ticks` 가
   Catalog 에서 뽑은 feature 벡터다 — 열 순서는 **알파벳 정렬**이다
   (`FEATURE_ORDER = tuple(sorted(_catalog.FEATURES))`, `sd/ticks.py:18`).
   "인접한 두 열이 인접한 호가 레벨이다"라는 원 DeepLOB 의 전제가 아예 없다.
   그 전제 위에서만 뜻이 있는 `kernel=(1,2), stride=(1,2)` 레벨-압축
   컨볼루션을 그대로 가져오면 알파벳 순서로 우연히 이웃한, 서로 무관한 두
   값(예: `book_imbalance` 옆의 `dt`)을 섞을 뿐이다. 그래서 레벨 축
   컨볼루션은 없다 — feature 축은 시간 컨볼루션의 **채널**로만 쓴다.
2. **Inception 을 얕게(lite) 줄였다.** 원본은 32~64 채널 가지를 여러 깊이로
   쌓는다(레벨 압축 뒤에 세 번). 여기서는 가지당 `inception_channels`
   (기본 4) 채널짜리 층 하나만 둔다 — "여러 시간 스케일(1·3·5 틱, 풀링)을
   한 층에서 본다"는 구조적 특징만 보존한다. 병목이 이미 좁으므로(S1) 그
   앞에 큰 표현력을 쌓아 봐야 L1 이 결국 짓누른다.
3. **LSTM 을 대폭 줄였다** (64 → `lstm_hidden`, 기본 16, 단일 층). 병목
   (`bottleneck`) 뒤에 오는 것은 `ShallowMLP` 와 똑같은 좁은 선형 헤드
   둘뿐이라, LSTM 자체를 넓게 둘 이유가 없다.
4. **3-class 분류 헤드를 뺐다.** 이 저장소의 계약(`sd/teacher/base.py`)은
   스칼라 회귀(`predict_path`) + 체결확률(`predict_fill`) 두 헤드다. 원본의
   상승/보합/하락 분류는 이 태스크에 없다.
5. **BatchNorm 을 뺐다.** on-manifold 표집 후 표본은 수천 행 규모이고
   (`sd/manifold.py` 의 `max_manifold_samples`), 표집 가중치가 불균등하다
   (`_tail_weight` 가 `[1,5]` 범위, `sd/manifold.py`) — 배치 통계가 미니배치
   구성에 따라 흔들리기 쉽다. 이 저장소는 애초에 미니배치도 안 쓴다
   (`ShallowMLP` 처럼 전체 배치 full-batch 학습, DESIGN.md 가 배치를 규정한
   적이 없다) — 배치 하나뿐이면 BatchNorm 은 표준화를 다시 하는 것과
   같아지고, `_standardise`(평균/표준편차) 로 이미 하고 있다.

## 시간 윈도우 — 입력 형태 간극

원 DeepLOB 은 시간 축이 있는 텐서를 받지만 이 파이프라인의 `X` 는 시간 축이
없는 `(n, F)` 행렬이다(`sd/e0/runner.py`, `run_slice.py` 참조 — 읽기만 했다).
이 클래스는 그 간극을 **직접 메우지 않는다** — `X` 는 **이미
`sd.teacher.window.make_causal_windows` 로 만든** 평평한
`(n, window*n_features)` 행렬이어야 한다. 윈도우 생성과 모델을 분리한 이유는
`window.py` 의 모듈 docstring에 적었다: 윈도우는 표집(subsample) **이전**의
연속 행렬에 대해서만 뜻이 있고, 그 순서를 강제하려면 별도 함수로 빼는 편이
"모델 안에 숨겨서 호출자가 잊어도 그럴듯하게 도는" 것보다 안전하다 —
`fit`/`z`/`predict_path`/`predict_fill` 은 열 수가 `window*n_features` 와
다르면 바로 `ValueError` 를 던진다(조용히 잘못된 reshape 를 하지 않는다).

## 병목

`ShallowMLP` 와 계약이 같다 — `z()` 는 항상 `(n, bottleneck)`, 병목에 L1
벌점이 걸린다(Cranmer 방법, `ShallowMLP` 와 동일한 자리: `loss + l1 *
latent.abs().mean()`). LSTM 최종 은닉 상태 → `Linear(lstm_hidden,
bottleneck)`(활성함수 없음, `ShallowMLP._encoder` 의 마지막 `Linear` 와 같은
자리) → 두 개의 독립된 선형 헤드(`predict_path`, `predict_fill`).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _TemporalConvBlock(nn.Module):
    """시간 축 위에서만 컨볼루션한다(레벨 축 컨볼루션 없음 — 모듈 docstring §1)."""

    def __init__(self, n_features: int, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, F, window)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x                                          # (batch, channels, window)


class _InceptionLite(nn.Module):
    """DeepLOB Inception 모듈의 압축판 — 가지당 채널 하나, 깊이 하나(§2)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, in_channels, window)
        parts = [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)]
        return self.act(torch.cat(parts, dim=1))          # (batch, 4*c, window)


class _DeepLOBEncoder(nn.Module):
    """CNN(시간) → Inception-lite → LSTM → 병목. `z` 를 낸다(헤드는 바깥에 둔다,
    `ShallowMLP` 와 같은 자리 배치 — 헤드 두 개를 독립 모듈로 유지해야 두
    헤드가 서로의 복사가 되는 사고를 구조적으로 막기 쉽다)."""

    def __init__(self, n_features: int, bottleneck: int, conv_channels: int,
                 inception_channels: int, lstm_hidden: int) -> None:
        super().__init__()
        self.conv = _TemporalConvBlock(n_features, conv_channels)
        self.inception = _InceptionLite(conv_channels, inception_channels)
        lstm_input = inception_channels * 4
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden, batch_first=True)
        self.to_bottleneck = nn.Linear(lstm_hidden, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, F, window)
        h = self.conv(x)                                  # (batch, conv_channels, window)
        h = self.inception(h)                             # (batch, 4*inception_channels, window)
        h = h.permute(0, 2, 1)                             # (batch, window, channels) — LSTM batch_first
        _output, (h_n, _c_n) = self.lstm(h)
        last = h_n[-1]                                     # 마지막 층의 최종 은닉 상태 (batch, lstm_hidden)
        return self.to_bottleneck(last)                     # (batch, bottleneck)


class DeepLOBCompact:
    """`X -> (z, path, fill)`. `X` 는 `sd.teacher.window.make_causal_windows`
    로 만든 `(n, window*n_features)` 평평한 행렬이어야 한다(모듈 docstring
    참조). `n_features` 는 한 시점당 feature 수(윈도우 이전 열 수)다."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 window: int = 16, conv_channels: int = 8,
                 inception_channels: int = 4, lstm_hidden: int = 16,
                 l1: float = 1e-4) -> None:
        self.bottleneck = int(bottleneck)
        self.window = int(window)
        self.n_features = int(n_features)          # 한 시점당 feature 수
        self.l1 = float(l1)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._encoder = _DeepLOBEncoder(
            n_features=self.n_features, bottleneck=self.bottleneck,
            conv_channels=conv_channels, inception_channels=inception_channels,
            lstm_hidden=lstm_hidden)
        self._head_path = nn.Linear(self.bottleneck, 1)
        self._head_fill = nn.Linear(self.bottleneck, 1)
        flat_dim = self.window * self.n_features
        self._mean = np.zeros(flat_dim)
        self._scale = np.ones(flat_dim)

    # -- 학습 ---------------------------------------------------------------
    def fit(self, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
            weight: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> "DeepLOBCompact":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._check_shape(X)
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
            latent = self._encode(inputs)
            path = self._head_path(latent).squeeze(-1)
            fill = self._head_fill(latent).squeeze(-1)
            loss_path = (weights * (path - target_path) ** 2).mean()
            loss_fill = (weights * bce(fill, target_fill)).mean()
            # 병목에 L1. Cranmer 방법의 핵심이 이 한 줄이다(ShallowMLP 와 동일).
            loss = loss_path + loss_fill + self.l1 * latent.abs().mean()
            loss.backward()
            optimiser.step()
        return self

    # -- 추론 ---------------------------------------------------------------
    def _check_shape(self, X: np.ndarray) -> None:
        expected = self.window * self.n_features
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(
                f"X 열 수가 window*n_features 와 다르다: got {X.shape}, "
                f"expected (n, {expected}) (window={self.window}, "
                f"n_features={self.n_features}). sd.teacher.window."
                "make_causal_windows 로 먼저 시간 윈도우를 만들었는지 확인하라")

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        # 평평한 (batch, window*F) -> (batch, window, F) -> (batch, F, window).
        # make_causal_windows 가 만든 순서(옛것→최신, C-order reshape)와 맞다.
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

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head_path(self._latent(X)).squeeze(-1).numpy()

    def predict_fill(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self._head_fill(self._latent(X))).squeeze(-1).numpy()
