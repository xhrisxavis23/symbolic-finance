"""미니배치 교사 학습 — PREREG-MINIBATCH.md §1 을 코드로 옮긴 것.

**바꾸는 것은 학습 절차뿐이다.** 모델 구조·초기화·손실·표준화 방식·예측 함수는
`gpu_eval/deeplob_gpu.py::DeepLOBCompactGPU` 인스턴스의 것을 그대로 쓴다(그 모듈을 받아 학습한다).

윈도우는 묶음마다 원본 배열에서 즉석으로 만든다. `scale/efficient_window.py::make_causal_windows_subset`
은 호출마다 전체 배열의 세션 시작점과 비유한 값 채우기를 다시 계산하므로 수천만 행에서 묶음마다 부르면
안 된다. 같은 함수(`session_starts`, `forward_fill_non_finite`)로 전처리를 **한 번만** 하고
행 선택만 묶음마다 한다 — 값은 같아야 하며 `test_minibatch_teacher.py` 가 비트 단위로 확인한다.

호출자 책임: 같은 초기값으로 비교하려면 교사 인스턴스를 만들기 **전에** `torch.manual_seed(seed)`.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/scale")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
from efficient_window import forward_fill_non_finite, session_starts  # noqa: E402
from sd.sr.base import weighted_r2  # noqa: E402
from sd.teacher.early_stopping import run_with_early_stopping, torch_snapshot_functions  # noqa: E402

BATCH_SIZE = 8192
LR_CANDIDATES = (1e-3, 3e-4)      # 사전등록 §1 — 결과 보기 전 고정한 두 후보
EVALS_PER_EPOCH = 4               # 0.25 epoch 마다 평가
PATIENCE = 20                     # 평가 20회 = 5 epoch
MAX_EPOCHS = 30
PRED_CHUNK = 500_000
STD_ZERO_TOL = 1e-9               # 흘려 계산한 분산의 부동소수 잔차를 0 으로 본다


class BatchWindower:
    """`make_causal_windows_subset(X, window, session_ids, idx)` 와 같은 값을 낸다.
    전처리(세션 시작점·비유한 값 채우기)는 생성 시 한 번만 한다."""

    def __init__(self, X: np.ndarray, session_ids: np.ndarray, window: int) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 는 2차원이어야 한다: {X.shape}")
        self.window = int(window)
        self.n_features = X.shape[1]
        self.start = session_starts(np.asarray(session_ids))
        self.X_filled = forward_fill_non_finite(X, self.start)
        self.lag = np.arange(self.window - 1, -1, -1)

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        raw = idx[:, None] - self.lag[None, :]
        clamped = np.maximum(raw, self.start[idx][:, None])
        return self.X_filled[clamped].reshape(len(idx), self.window * self.n_features)


def window_stats(windower: BatchWindower, idx: np.ndarray, chunk: int = 500_000):
    """학습 행 윈도우의 평균·표준편차를 흘려 계산한다. 전체 배치의
    `X.mean(axis=0)`, `np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)` 과 같은 값."""
    D = windower.window * windower.n_features
    s1, s2, n = np.zeros(D), np.zeros(D), 0
    for i in range(0, len(idx), chunk):
        w = windower(idx[i:i + chunk])
        s1 += w.sum(axis=0)
        s2 += (w * w).sum(axis=0)
        n += len(w)
    mean = s1 / n
    var = s2 / n - mean ** 2
    zero = var <= STD_ZERO_TOL * (1.0 + mean ** 2)
    std = np.sqrt(np.where(zero, 0.0, var))
    return mean, np.where(std > 0, std, 1.0)


def fit_minibatch(teacher, windower: BatchWindower, fit_idx, y_path, y_fill, weight, *,
                  Xw_select, y_select, w_select, lr: float, batch_size: int = BATCH_SIZE,
                  max_epochs: int = MAX_EPOCHS, evals_per_epoch: int = EVALS_PER_EPOCH,
                  patience: int = PATIENCE, seed: int = 0) -> dict:
    """`teacher`(DeepLOBCompactGPU)의 모듈을 미니배치로 학습한다. y·가중치는 fit_idx 와 같은 순서.

    손실은 전체 배치 `fit` 과 같다: 가중 MSE(경로) + 가중 BCE(체결) + l1·|잠재|.
    조기 종료는 파이프라인의 `run_with_early_stopping` 을 그대로 쓰고, 한 단위 = 1/evals_per_epoch epoch.
    """
    import torch
    import torch.nn as nn

    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    y_path = np.asarray(y_path, dtype=np.float32)
    y_fill = np.asarray(y_fill, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if not (len(fit_idx) == len(y_path) == len(y_fill) == len(weight)):
        raise ValueError("fit_idx·y_path·y_fill·weight 길이가 다르다")

    t_stats = time.time()
    teacher._mean, teacher._scale = window_stats(windower, fit_idx)
    stats_seconds = time.time() - t_stats
    mean32, scale32 = teacher._mean.astype(np.float32), teacher._scale.astype(np.float32)

    dev = teacher.device
    params = (list(teacher._encoder.parameters()) + list(teacher._head_path.parameters())
              + list(teacher._head_fill.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    rng = np.random.default_rng(seed)
    n = len(fit_idx)
    per_unit = int(np.ceil(np.ceil(n / batch_size) / evals_per_epoch))
    cursor = {"perm": rng.permutation(n), "pos": 0}

    def next_batch() -> np.ndarray:
        if cursor["pos"] >= n:
            cursor["perm"], cursor["pos"] = rng.permutation(n), 0
        b = cursor["perm"][cursor["pos"]:cursor["pos"] + batch_size]
        cursor["pos"] += batch_size
        return b

    def train_step() -> None:                        # 1 단위 = 1/evals_per_epoch epoch
        for _ in range(per_unit):
            b = next_batch()
            xb = (windower(fit_idx[b]).astype(np.float32) - mean32) / scale32
            inputs = torch.from_numpy(xb).to(dev)
            tp = torch.from_numpy(y_path[b]).to(dev)
            tf = torch.from_numpy(y_fill[b]).to(dev)
            wt = torch.from_numpy(weight[b]).to(dev)
            opt.zero_grad()
            latent = teacher._encode(inputs)
            path = teacher._head_path(latent).squeeze(-1)
            fill = teacher._head_fill(latent).squeeze(-1)
            loss = ((wt * (path - tp) ** 2).mean() + (wt * bce(fill, tf)).mean()
                    + teacher.l1 * latent.abs().mean())
            loss.backward()
            opt.step()

    y_sel = np.asarray(y_select, dtype=float)
    w_sel = np.asarray(w_select, dtype=float)

    def evaluate() -> float:
        pred = np.concatenate([teacher.predict_path(Xw_select[i:i + PRED_CHUNK])
                               for i in range(0, len(Xw_select), PRED_CHUNK)])
        return weighted_r2(y_sel, pred, w_sel)

    get_state, set_state = torch_snapshot_functions({
        "encoder": teacher._encoder, "head_path": teacher._head_path, "head_fill": teacher._head_fill})
    t0 = time.time()
    hist = run_with_early_stopping(total_epochs=max_epochs * evals_per_epoch, eval_every=1,
                                   patience=patience, train_step=train_step, evaluate=evaluate,
                                   get_state=get_state, set_state=set_state)
    teacher.early_stop_history_ = hist
    return {"lr": lr, "batch_size": batch_size, "n_fit_rows": int(n), "batches_per_eval": per_unit,
            "stats_seconds": stats_seconds, "fit_seconds": time.time() - t0,
            "best_epoch": hist.best_epoch / evals_per_epoch, "best_score": hist.best_score,
            "stopped_epoch": hist.stopped_epoch / evals_per_epoch, "triggered": hist.triggered,
            "eval_scores": [float(s) for s in hist.eval_scores]}
