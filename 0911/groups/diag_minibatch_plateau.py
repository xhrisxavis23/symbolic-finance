"""미니배치 학습 점검 실패 진단: 학습기 결함인가, 학습 횟수 부족인가.
같은 합성 데이터에서 (a) 기존 전체 배치 학습기, (b) 미니배치를 길게 돌린 궤적을 비교한다."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
import minibatch_teacher as mbt
from deeplob_gpu import DeepLOBCompactGPU
from sd.sr.base import weighted_r2

rng = np.random.default_rng(0)
W, F = 16, 19
# 테스트와 같은 난수 소비를 재현하기 위해 테스트 1·2 단계의 난수 사용을 그대로 반복
lengths = [40, 3, 900, 1, 1500, 17, 2539]
sess = np.concatenate([np.full(L, i) for i, L in enumerate(lengths)])
X = rng.standard_normal((len(sess), F)); rng.random(X.shape)
rng.choice(len(sess), 3000, replace=False)
N = 24_000
sess2 = np.repeat(np.arange(N // 400), 400)
X2 = rng.standard_normal((N, F))
fit_idx, sel_idx = np.arange(18_000), np.arange(18_000, N)
wdr2 = mbt.BatchWindower(X2, sess2, W)
Xw = wdr2(np.arange(N))
signal = np.tanh(Xw[:, :F].sum(axis=1) * 0.3 + Xw[:, -F:].mean(axis=1))
y = signal + 0.3 * rng.standard_normal(N)
yf = (signal > 0).astype(float); w = np.ones(N)
print("이론 상한 R² ≈", round(np.var(signal) / np.var(y), 3), flush=True)

t0 = time.time()
fb = DeepLOBCompactGPU(n_features=F, bottleneck=2, seed=0, window=W, device="cpu")
fb.fit(Xw[fit_idx], y[fit_idx], yf[fit_idx], w[fit_idx], epochs=600, X_path_select=Xw[sel_idx],
       y_path_select=y[sel_idx], weight_select=w[sel_idx], eval_every=10, patience=30)
h = fb.early_stop_history_
print(f"(a) 전체 배치 lr1e-2: 최선 {h.best_score:+.4f} @epoch {h.best_epoch} 멈춤 {h.stopped_epoch} ({time.time()-t0:.0f}s)", flush=True)
print("    궤적", [round(s, 3) for s in h.eval_scores[::5]], flush=True)

for lr in (1e-3, 3e-3):
    t0 = time.time()
    torch.manual_seed(0)
    t = DeepLOBCompactGPU(n_features=F, bottleneck=2, seed=0, window=W, device="cpu")
    info = mbt.fit_minibatch(t, wdr2, fit_idx, y[fit_idx], yf[fit_idx], w[fit_idx],
                             Xw_select=Xw[sel_idx], y_select=y[sel_idx], w_select=w[sel_idx],
                             lr=lr, batch_size=512, max_epochs=40, patience=1000)
    print(f"(b) 미니배치 lr{lr}: 최선 {info['best_score']:+.4f} @epoch {info['best_epoch']} ({time.time()-t0:.0f}s)", flush=True)
    print("    궤적(epoch마다)", [round(s, 3) for s in info["eval_scores"][3::4]], flush=True)
