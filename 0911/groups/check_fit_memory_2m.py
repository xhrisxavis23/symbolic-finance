"""종목군 실행 재기동 전 점검 — 200만 행 전체 배치 학습이 GPU0 에서 메모리 부족 없이 도는가.

첫 실행(2026-09-14 00:52 KST 종료)은 첫 역전파에서 메모리 부족으로 죽었다: 할당 33.23GB, **예약됐지만
쓰지 않는 조각 37.77GB**, 42.51GB 요청 실패. 총량이 아니라 조각화 문제다. 파이토치 할당기 설정
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 로 조각화를 줄인다 — 계산 값은 바뀌지 않는다.

실제 실행과 같은 모양(학습 1,979,970 × 304, 선택 197,998 × 304)의 무작위 배열로, 선택 평가가 학습 사이에
끼는 순서까지 같게 몇 epoch 돌려 최대 메모리를 잰다.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
from deeplob_gpu import DeepLOBCompactGPU  # noqa: E402

N_FIT, N_SEL, F, W = 1_979_970, 197_998, 19, 16
EPOCHS, EVAL_EVERY = 12, 4

assert torch.cuda.device_count() == 1, torch.cuda.device_count()
uuid = str(torch.cuda.get_device_properties(0).uuid)
conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
print("GPU", torch.cuda.get_device_name(0), uuid, "ALLOC_CONF=", conf, flush=True)
rng = np.random.default_rng(0)
X = rng.standard_normal((N_FIT, F * W))
Xs = rng.standard_normal((N_SEL, F * W))
yp, ys = rng.standard_normal(N_FIT), rng.standard_normal(N_SEL)
yf, w, ws = (rng.random(N_FIT) < 0.5).astype(float), np.ones(N_FIT), np.ones(N_SEL)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
t = DeepLOBCompactGPU(n_features=F, bottleneck=2, seed=0, window=W, device="cuda")
t.fit(X, yp, yf, w, epochs=EPOCHS, X_path_select=Xs, y_path_select=ys, weight_select=ws,
      eval_every=EVAL_EVERY, patience=1000)
torch.cuda.synchronize()
res = {"gpu": uuid, "alloc_conf": conf, "n_fit": N_FIT, "n_select": N_SEL, "epochs": EPOCHS,
       "eval_every": EVAL_EVERY, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
       "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
       "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
       "seconds": time.time() - t0, "ok": True}
print(res, flush=True)
json.dump(res, open("/home/dgu/tick/symbolic/0911/groups/data/check_fit_memory_2m.json", "w"), indent=1)
