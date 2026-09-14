"""미니배치 학습기 점검 — 실측에 쓰기 전에.

1. 즉석 윈도우가 `make_causal_windows_subset` 과 **비트 단위로 같은가** (비유한 값·세션 경계 포함)
2. 흘려 계산한 표준화 통계가 전체 배치 방식과 같은가 (상수 열 포함)
3. 알려진 신호가 있으면 배우고, **y 를 섞으면 못 배우는가** (학습기가 가짜로 점수를 내지 않는가)
CPU 에서 작은 규모로 돈다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/scale")
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/gpu_eval")
import minibatch_teacher as mbt  # noqa: E402
from efficient_window import make_causal_windows_subset  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  통과  " if cond else "  실패  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


rng = np.random.default_rng(0)
W, F = 16, 19

# --- 1. 윈도우 동일성 ---------------------------------------------------------
lengths = [40, 3, 900, 1, 1500, 17, 2539]
sess = np.concatenate([np.full(L, i) for i, L in enumerate(lengths)])
X = rng.standard_normal((len(sess), F))
X[rng.random(X.shape) < 0.05] = np.nan                  # 비유한 값
X[np.cumsum([0] + lengths[:-1]), :] = np.nan            # 세션 첫 행이 비유한 → 과거 없음 → 0 채움 경로
X[:, 3] = 2.0                                           # 상수 열
idx = np.concatenate([rng.choice(len(sess), 3000, replace=False), np.cumsum([0] + lengths[:-1])])
ref = make_causal_windows_subset(X, W, sess, idx)
wdr = mbt.BatchWindower(X, sess, W)
got = wdr(idx)
print("[1] 즉석 윈도우 vs make_causal_windows_subset")
check(got.shape == ref.shape, f"모양 같음 {got.shape}")
check(np.array_equal(got, ref), "값이 비트 단위로 같다")
check(np.isfinite(got).all(), "비유한 값이 남지 않는다")

# --- 2. 표준화 통계 -----------------------------------------------------------
print("[2] 흘려 계산한 표준화 통계 vs 전체 배치 방식")
mean_ref = ref.mean(axis=0)
scale_ref = np.where(ref.std(axis=0) > 0, ref.std(axis=0), 1.0)
mean_got, scale_got = mbt.window_stats(wdr, idx, chunk=333)
check(np.allclose(mean_got, mean_ref, rtol=1e-10, atol=1e-12), "평균 같음")
check(np.allclose(scale_got, scale_ref, rtol=1e-8, atol=1e-10), "표준편차 같음")
const_cols = np.flatnonzero(ref.std(axis=0) == 0)
check(len(const_cols) > 0 and np.all(scale_got[const_cols] == 1.0), f"상수 열 {len(const_cols)}개의 척도가 정확히 1")

# --- 3. 학습기가 알려진 신호를 배우고, 섞은 y 로는 못 배우는가 ----------------
print("[3] 학습 점검 (CPU, 작은 규모)")
import torch  # noqa: E402
from deeplob_gpu import DeepLOBCompactGPU  # noqa: E402

N = 24_000
sess2 = np.repeat(np.arange(N // 400), 400)
X2 = rng.standard_normal((N, F))
fit_idx = np.arange(0, 18_000)
sel_idx = np.arange(18_000, N)
wdr2 = mbt.BatchWindower(X2, sess2, W)
Xw_all = wdr2(np.arange(N))
signal = np.tanh(Xw_all[:, :F].sum(axis=1) * 0.3 + Xw_all[:, -F:].mean(axis=1))  # 과거·현재 특징의 비선형 결합
y = signal + 0.3 * rng.standard_normal(N)
yf = (signal > 0).astype(float)
w = np.ones(N)


# 예산 40 epoch·조기 종료 끔: 이 합성 신호는 전체 배치·미니배치 모두 긴 정체(R² 0.02) 뒤에 급등한다
# (diag_minibatch_plateau.log — 미니배치 16 epoch 부근, 전체 배치 110~160 epoch). 첫 실행의 6 epoch 는 정체 안에서 끝났다.
# 여기서 보는 것은 학습기가 배우는가이지 조기 종료 규칙이 아니다.
def train(y_train, label):
    torch.manual_seed(0)
    t = DeepLOBCompactGPU(n_features=F, bottleneck=2, seed=0, window=W, device="cpu")
    info = mbt.fit_minibatch(t, wdr2, fit_idx, y_train[fit_idx], yf[fit_idx], w[fit_idx],
                             Xw_select=Xw_all[sel_idx], y_select=y[sel_idx], w_select=w[sel_idx],
                             lr=1e-3, batch_size=512, max_epochs=40, patience=1000)
    print(f"    {label}: 선택 R² {info['best_score']:+.4f}, 최선 epoch {info['best_epoch']}, "
          f"멈춤 {info['stopped_epoch']}, 조기종료 {info['triggered']}", flush=True)
    return info, t


info_real, t_real = train(y, "진짜 y")
check(info_real["best_score"] > 0.3, "알려진 신호를 배운다 (선택 R² > 0.3)")
check(len(info_real["eval_scores"]) > 0, "조기 종료 평가 기록이 남는다")
pred_now = t_real.predict_path(Xw_all[sel_idx])
from sd.sr.base import weighted_r2  # noqa: E402
check(abs(weighted_r2(y[sel_idx], pred_now, w[sel_idx]) - info_real["best_score"]) < 1e-6,
      "학습 뒤 모델이 최선 시점 상태로 되돌려져 있다")

y_shuf = y.copy()
y_shuf[fit_idx] = y[rng.permutation(fit_idx)]
info_shuf, _ = train(y_shuf, "섞은 y")
check(info_shuf["best_score"] < 0.05, "섞은 y 로는 못 배운다 (선택 R² < 0.05)")

print()
if failures:
    print(f"실패 {len(failures)}건:"); [print("  -", f) for f in failures]; sys.exit(1)
print("전부 통과")
