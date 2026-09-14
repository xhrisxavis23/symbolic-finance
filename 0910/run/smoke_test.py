#!/usr/bin/env python3
"""데이터 흐름 전체를 소규모(소수 종목·짧은 epoch·낮은 niterations)로
한 번 통과시켜 본다 — 몇 시간짜리 실행을 시작하기 전에 로직 버그를 잡는다.
실제 상태 파일(state/)에는 아무것도 안 쓴다(별도 tmp 디렉터리).
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/home/dgu/tick/symbolic/0910")

from run import common, locked_params as P  # noqa: E402

t0 = time.time()
strata_df = common.stratified_universe()
groups = common.friction_group_symbols(strata_df)
L_small = groups["L"][:8]
M_small = groups["M"][:8]
H_small = groups["H"][:4]
print(f"L_small={L_small}\nM_small={M_small}\nH_small={H_small}")

X_L, y_path_L, y_fill_L, mask_L, feat_L, sess_L = common.load_windowed(
    L_small, P.L_TRAIN_WINDOW)
X_M, y_path_M, y_fill_M, mask_M, feat_M, sess_M = common.load_windowed(
    M_small, P.M_SELECT_WINDOW)
print(f"X_L={X_L.shape} X_M={X_M.shape} feat_L==feat_M: {feat_L==feat_M}")
assert feat_L == feat_M, "feature 불일치 — 실제 실행에서 교집합 로직이 필요하다"

from sd.manifold import select as manifold_select
from sd.teacher.window import make_causal_windows
from sd.teacher.deeplob import DeepLOBCompact
from sd.teacher.gate import evaluate as evaluate_gate
import numpy as np

sel_L = manifold_select(X_L, mask_L, max_samples=5000, seed=0)
fit_rows = sel_L.index[sel_L.on_manifold]
fit_weight = sel_L.weight[sel_L.on_manifold]
sel_M = manifold_select(X_M, mask_M, max_samples=5000, seed=0)
select_rows = sel_M.index[sel_M.on_manifold]
select_weight = sel_M.weight[sel_M.on_manifold]
print(f"fit_rows={len(fit_rows)} select_rows={len(select_rows)}")

Xw_L = make_causal_windows(X_L, window=P.TEACHER_WINDOW, session_ids=sess_L)
Xw_M = make_causal_windows(X_M, window=P.TEACHER_WINDOW, session_ids=sess_M)

teacher = DeepLOBCompact(n_features=len(feat_L), bottleneck=P.BOTTLENECK, seed=0,
                         window=P.TEACHER_WINDOW).fit(
    Xw_L[fit_rows], y_path_L[fit_rows], y_fill_L[fit_rows], fit_weight,
    epochs=20, X_path_select=Xw_M[select_rows], y_path_select=y_path_M[select_rows],
    weight_select=select_weight, eval_every=5, patience=2)
print("teacher fit OK, early_stop:", teacher.early_stop_history_.stopped_epoch,
     teacher.early_stop_history_.best_epoch)

gate_result = evaluate_gate(teacher, Xw_M[select_rows], y_path_M[select_rows],
                            y_fill_M[select_rows], np.ones(len(select_rows), dtype=bool))
print("gate passed:", gate_result.passed)

target = teacher.predict_path(Xw_L[fit_rows])
print("target shape:", target.shape, "finite:", np.isfinite(target).all())

from sd.sr.pysr_backend import PySRBackend
from sd.compile import compile_candidates

backend = PySRBackend(seed=0, deterministic=True, niterations=5, maxsize=P.SR_MAXSIZE)
candidates = backend.fit(X_L[fit_rows], target, fit_weight, feat_L)
print("SR candidates:", len(candidates))
compiled, failures = compile_candidates(candidates)
print(f"compiled: {len(compiled)}/{len(candidates)} failed={len(failures)}")
if failures:
    for f in failures[:3]:
        print("  fail:", f.stage, f.reason[:150])

if compiled:
    from sd import replay as sd_replay, select as sd_select
    entries = [SimpleNamespace(ast=e.ast) for e in compiled[:2]]
    result = sd_replay.run(entries, symbols=M_small, date=P.DATE,
                           output_dir="/tmp/smoke_replay", grid=[0.70, 0.85],
                           workers=4)
    print("replay attempts:", result.attempts)
    import pandas as pd
    ledger = pd.read_parquet(result.ledger_path)
    summary = sd_select.summarise(ledger)
    print(summary[["decisions", "scorable", "total_net_bps"]])
    ranked = sd_select.rank(summary, min_scorable=1)
    print("ranked rows:", len(ranked))

    # naive/null 컴파일 확인
    import sympy
    from sd.sr.base import Candidate
    naive_c = Candidate(expr=sympy.Symbol(P.NAIVE_FEATURE), complexity=1,
                        in_sample_score=float("nan"), backend="x", seed=-1)
    null_c = Candidate(expr=sympy.Symbol(P.NULL_FEATURE), complexity=1,
                       in_sample_score=float("nan"), backend="x", seed=-1)
    bcompiled, bfailed = compile_candidates([naive_c, null_c])
    print("baseline compiled:", len(bcompiled), "failed:", len(bfailed))

# minute_of_session 확인
from sd import replay as sd_replay
entries2 = [SimpleNamespace(ast=compiled[0].ast)] if compiled else []
if entries2:
    result2 = sd_replay.run(entries2, symbols=H_small, date=P.DATE,
                            output_dir="/tmp/smoke_replay_h", grid=[0.70],
                            workers=4)
    import pandas as pd
    ledger2 = pd.read_parquet(result2.ledger_path)
    print("H ledger cols has minute_of_session:", "minute_of_session" in ledger2.columns)
    if len(ledger2):
        print(ledger2["minute_of_session"].describe() if "minute_of_session" in ledger2 else "n/a")

print(f"\nSMOKE TEST DONE in {time.time()-t0:.1f}s")
