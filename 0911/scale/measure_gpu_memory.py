"""전체 배치 교사 학습의 GPU 메모리를 실측한다 — 곱해서 추정하지 않는다.

교사는 학습 데이터 전체를 한 번에 GPU 로 올리고 매 epoch 전 행에 대해 경사 한 번을 계산한다.
메모리는 값이 아니라 모양(행 수 × 304)에만 달려 있으므로 무작위 배열로 잰다.
여러 크기에서 재서 선을 긋고, 200만 행 관측치(곡선 측정 49.5GB)와 대조해 투영을 검증한다.
"""
import json, sys, time
import numpy as np, torch

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
from deeplob_gpu import DeepLOBCompactGPU

assert torch.cuda.device_count() == 1, torch.cuda.device_count()
uuid = str(torch.cuda.get_device_properties(0).uuid)
print("GPU", torch.cuda.get_device_name(0), uuid, flush=True)
F, W = 19, 16
points = []
for n in (100_000, 200_000, 400_000, 800_000, 1_200_000):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, F * W)).astype(np.float32)
    yp = rng.standard_normal(n); yf = (rng.random(n) < 0.5).astype(float); w = np.ones(n)
    t = DeepLOBCompactGPU(n_features=F, bottleneck=2, seed=0, window=W, device="cuda")
    t0 = time.time(); t.fit(X, yp, yf, w, epochs=3); torch.cuda.synchronize()
    p = {"n": n, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
         "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
         "sec_per_epoch": (time.time() - t0) / 3}
    points.append(p); print(p, flush=True)
    del t, X; torch.cuda.empty_cache()

ns = np.array([p["n"] for p in points], float)
res = {}
for key in ("peak_alloc_gb", "peak_reserved_gb"):
    a, b = np.polyfit(ns, np.array([p[key] for p in points]), 1)
    res[key] = {"gb_per_million_rows": a * 1e6, "intercept_gb": b,
                "projection_gb": {str(k): a * k + b for k in (2_000_000, 5_000_000, 10_000_000, 45_734_085)}}
    print(key, "백만 행당", round(a * 1e6, 2), "GB, 투영",
          {k: round(v, 1) for k, v in res[key]["projection_gb"].items()}, flush=True)
json.dump({"gpu": uuid, "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
           "points": points, "fits": res, "observed_2M_nvidia_smi_gb": 49.5},
          open("/home/dgu/tick/symbolic/0911/scale/data/gpu_memory_fullbatch.json", "w"), ensure_ascii=False, indent=1)
