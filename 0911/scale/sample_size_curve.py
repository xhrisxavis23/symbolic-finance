#!/usr/bin/env python3
"""표본 크기 대비 교사 R² 곡선 — 실측 (0911 작업 1+2 핵심 산출물).

`build_pool.py` 가 만든 L(학습 후보 풀, 3일)·M(고정 평가 풀, 1일) 을 읽어,
`max_samples` 를 늘려가며:

  1. `sd.manifold.select` 로 학습 표본을 뽑는다 (seed 고정)
  2. 교사를 학습한다 (ShallowMLP=원본 그대로, DeepLOB=`gpu_eval/deeplob_gpu.py`
     의 `device` 파라미터 버전 — 아키텍처·초기화·손실은 원본과 동일)
  3. **고정된** M 평가 표본(모든 사이즈에서 동일)에서 가중 R² 를 잰다
  4. 학습 시간·표본 크기·R² 를 함께 JSON 에 남긴다 — "이것이 작업2의 R²
     곡선과 같은 실행이므로 한 번에 하라"는 지시를 그대로 따른다.

    python3 sample_size_curve.py --model shallow --sizes 200000,500000,1000000,2000000,5000000
    python3 sample_size_curve.py --model deeplob --device cpu --sizes 200000,500000
    python3 sample_size_curve.py --model deeplob --device cuda --sizes 1000000,2000000,5000000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/scale")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")

from sd.manifold import select as manifold_select  # noqa: E402
from sd.sr.base import weighted_r2  # noqa: E402
from sd.teacher.shallow import ShallowMLP  # noqa: E402
from efficient_window import make_causal_windows_subset  # noqa: E402
from deeplob_gpu import DeepLOBCompactGPU  # noqa: E402

POOL_DIR = "/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad/0911_pool"
EVAL_MAX_SAMPLES = 200_000
WINDOW = 16
SEED = 0
EVAL_EVERY = 10
PATIENCE = 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool_arrays():
    L = np.load(f"{POOL_DIR}/L_pool.npz")
    M = np.load(f"{POOL_DIR}/M_pool.npz")
    meta = json.loads(Path(f"{POOL_DIR}/meta.json").read_text())
    return L, M, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["shallow", "deeplob"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sizes", type=str, required=True,
                        help="쉼표로 구분한 max_samples 목록")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    L, M, meta = load_pool_arrays()
    X_L, y_path_L, y_fill_L, mask_L = L["X"], L["y_path"], L["y_fill"], L["mask"]
    sess_L = L["session_ids"]
    X_M, y_path_M, y_fill_M, mask_M = M["X"], M["y_path"], M["y_fill"], M["mask"]
    sess_M = M["session_ids"]
    n_features = X_L.shape[1]
    log(f"pool loaded: X_L={X_L.shape} X_M={X_M.shape} n_features={n_features} "
        f"usable_L={int(mask_L.sum()):,} usable_M={int(mask_M.sum()):,}")

    # ---- 고정 평가 표본 (모든 사이즈에서 동일 — 재사용) ----
    t0 = time.time()
    sel_eval = manifold_select(X_M, mask_M, max_samples=EVAL_MAX_SAMPLES, seed=SEED)
    select_rows = sel_eval.index[sel_eval.on_manifold]
    select_weight = sel_eval.weight[sel_eval.on_manifold]
    log(f"fixed eval set: select_rows={len(select_rows):,} "
        f"(select() took {time.time()-t0:.1f}s)")

    X_eval = X_M[select_rows]
    if args.model == "deeplob":
        Xw_eval = make_causal_windows_subset(X_M, window=WINDOW, session_ids=sess_M,
                                             indices=select_rows)
        log(f"Xw_eval shape={Xw_eval.shape}")

    out_path = Path(args.out)
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done_sizes = {r["max_samples"] for r in results
                     if r["model"] == args.model and r["device"] == args.device}
        log(f"기존 결과 {len(results)}개 발견, 이미 완료된 사이즈: {done_sizes}")
    else:
        done_sizes = set()

    for size in sizes:
        if size in done_sizes:
            log(f"size={size:,} 이미 완료 — 스킵")
            continue
        log(f"=== size={size:,} model={args.model} device={args.device} ===")
        t_sel0 = time.time()
        sel_fit = manifold_select(X_L, mask_L, max_samples=size, seed=SEED)
        fit_rows = sel_fit.index[sel_fit.on_manifold]
        fit_weight = sel_fit.weight[sel_fit.on_manifold]
        t_sel1 = time.time()
        log(f"select() -> fit_rows={len(fit_rows):,} ({t_sel1-t_sel0:.1f}s)")

        record = {
            "model": args.model, "device": args.device, "max_samples": size,
            "n_fit_rows": int(len(fit_rows)), "select_seconds": t_sel1 - t_sel0,
            "epochs_budget": args.epochs, "eval_every": args.eval_every,
            "patience": args.patience, "n_eval_rows": int(len(select_rows)),
        }

        if args.model == "shallow":
            t0 = time.time()
            teacher = ShallowMLP(n_features=n_features, bottleneck=2, seed=SEED).fit(
                X_L[fit_rows], y_path_L[fit_rows], y_fill_L[fit_rows], fit_weight,
                epochs=args.epochs, X_path_select=X_eval, y_path_select=y_path_M[select_rows],
                weight_select=select_weight, eval_every=args.eval_every,
                patience=args.patience)
            t1 = time.time()
            history = teacher.early_stop_history_
            final_r2 = float(weighted_r2(y_path_M[select_rows], teacher.predict_path(X_eval),
                                         select_weight))
        else:
            t_w0 = time.time()
            Xw_fit = make_causal_windows_subset(X_L, window=WINDOW, session_ids=sess_L,
                                                indices=fit_rows)
            t_w1 = time.time()
            record["window_seconds"] = t_w1 - t_w0
            log(f"windowing fit set -> {Xw_fit.shape} ({t_w1-t_w0:.1f}s)")
            t0 = time.time()
            teacher = DeepLOBCompactGPU(n_features=n_features, bottleneck=2, seed=SEED,
                                        window=WINDOW, device=args.device).fit(
                Xw_fit, y_path_L[fit_rows], y_fill_L[fit_rows], fit_weight,
                epochs=args.epochs, X_path_select=Xw_eval, y_path_select=y_path_M[select_rows],
                weight_select=select_weight, eval_every=args.eval_every,
                patience=args.patience)
            t1 = time.time()
            history = teacher.early_stop_history_
            final_r2 = float(weighted_r2(y_path_M[select_rows], teacher.predict_path(Xw_eval),
                                         select_weight))

        record.update({
            "fit_seconds": t1 - t0,
            "eval_epochs": history.eval_epochs, "eval_scores": history.eval_scores,
            "best_epoch": history.best_epoch, "best_score": history.best_score,
            "stopped_epoch": history.stopped_epoch, "triggered": history.triggered,
            "final_r2_recheck": final_r2,
        })
        log(f"size={size:,} fit_seconds={t1-t0:.1f}s best_epoch={history.best_epoch} "
            f"best_score={history.best_score:.6f} stopped_epoch={history.stopped_epoch} "
            f"final_recheck={final_r2:.6f}")

        results.append(record)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        log(f"checkpoint saved -> {out_path}")

    log("all sizes done")


if __name__ == "__main__":
    main()
