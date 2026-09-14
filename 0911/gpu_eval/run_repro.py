#!/usr/bin/env python3
"""판정 R40 — CPU vs GPU 재현성 실험 본체. 같은 시드·같은 데이터로 한 쪽만
장치를 바꿔 돌리고, 학습 곡선(early-stop 궤적)과 최종 R² 를 JSON 으로 남긴다.

    python3 run_repro.py --device cpu --seed 0
    python3 run_repro.py --device cuda --seed 0

결정론 설정: `CUBLAS_WORKSPACE_CONFIG` 는 **torch/CUDA 초기화 전에** 설정해야
해서 이 파일 맨 위, 어떤 torch import 보다도 먼저 둔다.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
from deeplob_gpu import DeepLOBCompactGPU  # noqa: E402

DATA_PATH = "/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911_gpu_eval/repro_data.npz"


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:  # noqa: BLE001 - 일부 연산에 결정론 구현이 없으면 경고만
        print(f"[warn] use_deterministic_algorithms(True) 실패, warn_only 로 재시도: {error}")
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    set_determinism(args.seed)

    data = np.load(DATA_PATH)
    Xw_fit = data["Xw_fit"]
    y_path_fit = data["y_path_fit"]
    y_fill_fit = data["y_fill_fit"]
    weight_fit = data["weight_fit"]
    Xw_select = data["Xw_select"]
    y_path_select = data["y_path_select"]
    y_fill_select = data["y_fill_select"]
    weight_select = data["weight_select"]
    n_features = int(data["n_features"])
    window = int(data["window"])

    print(f"[repro] device={args.device} seed={args.seed} "
          f"Xw_fit={Xw_fit.shape} Xw_select={Xw_select.shape}")

    t0 = time.time()
    teacher = DeepLOBCompactGPU(n_features=n_features, bottleneck=2, seed=args.seed,
                               window=window, device=args.device)
    teacher.fit(Xw_fit, y_path_fit, y_fill_fit, weight_fit, epochs=args.epochs,
               X_path_select=Xw_select, y_path_select=y_path_select,
               weight_select=weight_select, eval_every=args.eval_every,
               patience=args.patience)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    history = teacher.early_stop_history_
    final_r2_recheck = None
    try:
        from sd.sr.base import weighted_r2
        final_r2_recheck = float(weighted_r2(
            y_path_select, teacher.predict_path(Xw_select), weight_select))
    except Exception as error:  # noqa: BLE001
        print(f"[warn] 재확인 R2 계산 실패: {error}")

    result = {
        "device": args.device, "seed": args.seed, "epochs_budget": args.epochs,
        "eval_every": args.eval_every, "patience": args.patience,
        "fit_seconds": t1 - t0,
        "n_fit_rows": int(Xw_fit.shape[0]), "n_select_rows": int(Xw_select.shape[0]),
        "eval_epochs": history.eval_epochs,
        "eval_scores": history.eval_scores,
        "best_epoch": history.best_epoch,
        "best_score": history.best_score,
        "stopped_epoch": history.stopped_epoch,
        "triggered": history.triggered,
        "reverted": history.reverted,
        "final_r2_recheck": final_r2_recheck,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[repro] done device={args.device} fit_seconds={t1-t0:.2f}s "
          f"best_epoch={history.best_epoch} best_score={history.best_score:.6f} "
          f"stopped_epoch={history.stopped_epoch} final_recheck={final_r2_recheck}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
