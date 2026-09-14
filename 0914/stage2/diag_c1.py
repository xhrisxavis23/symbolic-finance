#!/usr/bin/env python3
"""C1 실패 진단 — 재예측 차이가 모델·특징·정렬 오류인가, GPU 순전파의 묶음 크기 의존 잡음인가.

성과(y_path)는 쓰지 않는다. 같은 행을 묶음 크기·TF32 설정만 바꿔 다시 예측하고,
(1) 변형끼리의 차이(같은 GPU 의 자기 잡음)와 (2) 0911 저장 예측과의 차이를 나란히 잰다.
신호 일치: `score > 0`, 유효 행 위 직전 100행 Q95 신호가 저장 예측과 몇 % 같은가.

    CUDA_VISIBLE_DEVICES=<GPU0> GROUP_GPU_UUID=<GPU0> python -m stage2.diag_c1
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import common as C
from .make_scores import FEATS, WINDOW, prepare

from efficient_window import make_causal_windows_subset  # noqa: E402
from framework.contract import prior_tick_quantile  # noqa: E402

N_SYMBOLS = 12


def main() -> None:
    if os.environ.get("GROUP_GPU_UUID") != C.GPU_UUID:
        raise SystemExit("GPU0 를 지정해 기동해야 한다")
    import torch
    import group_run as gr

    t0 = time.time()
    all_symbols = C.eval_symbols()
    symbols = all_symbols[::len(all_symbols) // N_SYMBOLS][:N_SYMBOLS]
    a = pd.read_parquet(C.PRED_2M, columns=["symbol", "eval_set", "pred_deeplob"],
                        filters=[("eval_set", "==", "eval1"), ("symbol", "in", symbols)])
    b = pd.read_parquet(C.PRED_3D, columns=["symbol", "eval_set", "pred_deeplob"],
                        filters=[("eval_set", "==", "eval1"), ("symbol", "in", symbols)])
    stored = {"T2M": a, "T3D": b}

    teachers = {}
    for m, path in C.CKPT.items():
        ck = torch.load(path, map_location="cuda", weights_only=False)
        t = gr.build_teacher(len(FEATS))
        t._encoder.load_state_dict(ck["encoder"])
        t._head_path.load_state_dict(ck["head_path"])
        t._head_fill.load_state_dict(ck["head_fill"])
        t._mean, t._scale = ck["mean"], ck["scale"]
        teachers[m] = t

    def predict(t, Xw, chunk):
        return np.concatenate([t.predict_path(Xw[i:i + chunk]) for i in range(0, len(Xw), chunk)]).astype(float)

    flags = {"cudnn_tf32_default": bool(torch.backends.cudnn.allow_tf32),
             "matmul_tf32_default": bool(torch.backends.cuda.matmul.allow_tf32),
             "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    variants = ("full", "b4096", "b65536", "full_repeat", "full_notf32")
    diffs = {m: {v: [] for v in variants} for m in teachers}          # vs stored (mask rows)
    self_noise = {m: {v: [] for v in variants if v != "full"} for m in teachers}   # vs full (all rows)
    agree = {m: {v: {"abs": [0, 0], "q95": [0, 0]} for v in variants} for m in teachers}

    for sym in symbols:
        _s, d, err = prepare(sym)
        if err:
            raise SystemExit(f"{sym}: {err}")
        n = len(d["X"])
        Xw = make_causal_windows_subset(d["X"], WINDOW, np.zeros(n, dtype=np.int64), np.arange(n))
        mask = d["mask"]
        for m, t in teachers.items():
            ref = stored[m][stored[m].symbol.astype(str) == sym].pred_deeplob.to_numpy(float)
            if len(ref) != int(mask.sum()):
                raise SystemExit(f"{sym} {m}: 저장 행 수 불일치")
            out = {"full": predict(t, Xw, n), "b4096": predict(t, Xw, 4096), "b65536": predict(t, Xw, 65536),
                   "full_repeat": predict(t, Xw, n)}
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            out["full_notf32"] = predict(t, Xw, n)
            torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32_default"]
            torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32_default"]
            ref_cut = prior_tick_quantile(ref, 0.95)
            ref_q = np.isfinite(ref_cut) & (ref > ref_cut)
            for v, p in out.items():
                pm = p[mask]
                diffs[m][v].append(np.abs(pm - ref))
                if v != "full":
                    self_noise[m][v].append(np.abs(p - out["full"]))
                agree[m][v]["abs"][0] += int(((pm > 0) == (ref > 0)).sum())
                agree[m][v]["abs"][1] += len(ref)
                cut = prior_tick_quantile(pm, 0.95)
                q = np.isfinite(cut) & (pm > cut)
                agree[m][v]["q95"][0] += int((q == ref_q).sum())
                agree[m][v]["q95"][1] += len(ref)
        C.log("diag", f"{sym} n={n}")

    def dist(parts):
        x = np.concatenate(parts)
        return {"max": float(x.max()), "q999": float(np.quantile(x, 0.999)), "q99": float(np.quantile(x, 0.99)),
                "median": float(np.median(x)), "frac_gt_1e-5": float((x > 1e-5).mean()), "n": int(len(x))}

    result = {"schema": "0914_diag_c1.v1", "symbols": symbols, "torch_flags": flags,
              "vs_stored": {m: {v: dist(diffs[m][v]) for v in variants} for m in teachers},
              "self_noise_vs_full": {m: {v: dist(self_noise[m][v]) for v in self_noise[m]} for m in teachers},
              "signal_agreement_vs_stored": {m: {v: {k: c[0] / c[1] for k, c in agree[m][v].items()}
                                                 for v in variants} for m in teachers},
              "seconds": time.time() - t0}
    (C.OUT / "diag_c1.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for m in teachers:
        C.log("diag", f"{m} vs 저장: " + " · ".join(f"{v} max {result['vs_stored'][m][v]['max']:.2e}"
                                                    for v in variants))
        C.log("diag", f"{m} 자기 잡음: " + " · ".join(f"{v} max {result['self_noise_vs_full'][m][v]['max']:.2e}"
                                                    for v in self_noise[m]))


if __name__ == "__main__":
    main()
