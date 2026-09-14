#!/usr/bin/env python3
"""M2 — 교사 학습 시간의 규모 실측. 표본 크기(=사실상 종목 수가 max_samples 로
바뀌는 지점)와 전체 행 수(윈도우·마할라노비스 계산)를 분리해서 잰다.

방법론 노트 (왜 합성 배열을 쓰는가): `manifold.select`(마할라노비스 거리
계산)·`make_causal_windows`(시간 윈도우 gather)·`ShallowMLP.fit`/
`DeepLOBCompact.fit`(순전파·역전파) 은 전부 **모양(shape)과 dtype 에 의해
비용이 결정되는 벡터화 연산**이다 — 실제 시장 데이터의 *값*이 아니라
*행 수·열 수*가 시간을 결정한다. 그래서 실제 2,507종목을 다시 로드하는 대신,
동일한 모양의 합성 배열로 **목표 규모(23,091,003행) 자체에서 직접** 측정한다.
이것은 "표본에서 곱해 추정"이 아니다 — 작은 표본의 결과를 곱하는 게 아니라
**목표 규모 그 자체를 실행해서 잰다.** 데이터 적재(I/O·feature 계산)는 실제
값에 의존하므로 이 스크립트가 아니라 `measure_load.py` 가 실제 종목으로 잰다.

메모리가 위험한 구간(전체 행 수에서의 윈도우 생성)만 몇 단계로 나눠 재고
선형성을 확인한 뒤 가장 큰 지점을 추정한다 — 그 사실을 결과에 명시한다.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")

import numpy as np  # noqa: E402
from sd.manifold import select as manifold_select  # noqa: E402
from sd.teacher.shallow import ShallowMLP  # noqa: E402
from sd.teacher.deeplob import DeepLOBCompact  # noqa: E402
from sd.teacher.window import make_causal_windows  # noqa: E402

OUT = Path("/home/dgu/tick/symbolic/0910/scale/data/m2_teacher_scale.json")
F = 19          # 실측 어휘 크기 (M3 pool 로 확인: 무차원 9 + 파생 10)
WINDOW = 16
FIXED_EPOCHS = 20
BOTTLENECK = 2
SEED = 0

TOTAL_ROWS_FULL = 23_091_003   # S-1 계약: 정규장 유효 호가틱 전종목 합계


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def synth(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, F)).astype(float)
    y_path = rng.normal(scale=0.5, size=n)
    y_fill = rng.integers(0, 2, size=n).astype(float)
    mask = np.ones(n, dtype=bool)
    # 종목 경계를 흉내낸다 — 약 4,000행짜리 세션 다수(중앙값 종목 규모와 비슷)
    session_len = 4000
    session_ids = (np.arange(n) // session_len).astype(int)
    return X, y_path, y_fill, mask, session_ids


def main() -> None:
    log = {"F": F, "window": WINDOW, "fixed_epochs": FIXED_EPOCHS,
           "total_rows_full_universe": TOTAL_ROWS_FULL,
           "manifold_select_at_scale": [], "window_gen_at_scale": [],
           "teacher_fit_at_scale": []}

    # ---- (a) manifold.select 비용 — 목표 규모(23,091,003행)에서 직접 잰다 ----
    for n in [50_000, 500_000, 5_000_000, TOTAL_ROWS_FULL]:
        X, y_path, y_fill, mask, session_ids = synth(n)
        t0 = time.time()
        sel = manifold_select(X, mask, max_samples=200_000, seed=SEED)
        t1 = time.time()
        entry = {"n_rows": n, "seconds": t1 - t0, "n_on_manifold_capped": int(len(sel.index)),
                 "rss_mb": rss_mb()}
        log["manifold_select_at_scale"].append(entry)
        print(f"[select] n={n:,} -> {t1-t0:.2f}s rss={rss_mb():.0f}MB", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))
        del X, y_path, y_fill, mask, session_ids, sel

    # ---- (b) make_causal_windows 비용 — 메모리 위험 구간이라 몇 단계로 재고
    #      선형성을 확인한 뒤 전체 규모를 그 기울기로 추정한다 (라벨을 남긴다) ----
    window_points = [50_000, 500_000, 2_000_000, 5_000_000]
    for n in window_points:
        X, y_path, y_fill, mask, session_ids = synth(n)
        t0 = time.time()
        Xw = make_causal_windows(X, window=WINDOW, session_ids=session_ids)
        t1 = time.time()
        entry = {"n_rows": n, "seconds": t1 - t0, "output_shape": list(Xw.shape),
                 "output_bytes": int(Xw.nbytes), "rss_mb": rss_mb()}
        log["window_gen_at_scale"].append(entry)
        print(f"[window] n={n:,} -> {t1-t0:.2f}s output={Xw.nbytes/1e9:.2f}GB "
              f"rss={rss_mb():.0f}MB", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))
        del X, y_path, y_fill, mask, session_ids, Xw

    # 선형 회귀로 초 단위 기울기(초/행) 추정 -> 전체 규모 추정치 (extrapolated 라고 표시)
    ns = np.array([e["n_rows"] for e in log["window_gen_at_scale"]], dtype=float)
    secs = np.array([e["seconds"] for e in log["window_gen_at_scale"]], dtype=float)
    slope = float(np.polyfit(ns, secs, 1)[0])
    log["window_gen_extrapolated_full_universe"] = {
        "method": "measured_points_linear_fit_NOT_directly_measured_at_full_scale",
        "slope_seconds_per_row": slope,
        "estimated_seconds_at_full_universe": slope * TOTAL_ROWS_FULL,
        "estimated_output_gb_at_full_universe": TOTAL_ROWS_FULL * WINDOW * F * 8 / 1e9,
    }
    print(f"[window-extrapolation] slope={slope:.3e}s/row -> full universe est "
          f"{slope*TOTAL_ROWS_FULL:.0f}s, output size "
          f"{TOTAL_ROWS_FULL*WINDOW*F*8/1e9:.1f}GB", flush=True)
    OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))

    # ---- (c) 교사 학습 — max_samples(=fit 표본 크기)를 바꿔가며 고정 epoch 비용 ----
    for n_fit in [5_000, 50_000, 200_000, 1_000_000]:
        X, y_path, y_fill, mask, session_ids = synth(n_fit, seed=1)
        weight = np.ones(n_fit)

        t0 = time.time()
        shallow = ShallowMLP(n_features=F, bottleneck=BOTTLENECK, seed=SEED).fit(
            X, y_path, y_fill, weight, epochs=FIXED_EPOCHS)
        t1 = time.time()
        shallow_seconds = t1 - t0

        Xw = make_causal_windows(X, window=WINDOW, session_ids=session_ids)
        t2 = time.time()
        deeplob = DeepLOBCompact(n_features=F, bottleneck=BOTTLENECK, seed=SEED,
                                 window=WINDOW).fit(
            Xw, y_path, y_fill, weight, epochs=FIXED_EPOCHS)
        t3 = time.time()
        deeplob_seconds = t3 - t2
        window_prep_seconds = t2 - t1

        entry = {
            "n_fit_rows": n_fit,
            "shallow_mlp_seconds": shallow_seconds,
            "shallow_mlp_seconds_per_epoch": shallow_seconds / FIXED_EPOCHS,
            "deeplob_window_prep_seconds": window_prep_seconds,
            "deeplob_fit_seconds": deeplob_seconds,
            "deeplob_fit_seconds_per_epoch": deeplob_seconds / FIXED_EPOCHS,
            "rss_mb": rss_mb(),
        }
        log["teacher_fit_at_scale"].append(entry)
        print(f"[teacher] n_fit={n_fit:,} shallow={shallow_seconds:.2f}s "
              f"({shallow_seconds/FIXED_EPOCHS:.4f}s/epoch) "
              f"deeplob={deeplob_seconds:.2f}s ({deeplob_seconds/FIXED_EPOCHS:.4f}s/epoch) "
              f"window_prep={window_prep_seconds:.2f}s rss={rss_mb():.0f}MB", flush=True)
        OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str))
        del X, y_path, y_fill, mask, session_ids, weight, Xw, shallow, deeplob

    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
