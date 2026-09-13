"""group_run.py 소규모 시험 운전 — 본 실행(수 시간) 전에 끝까지 도는지, 재시작 경로가 같은 예측을 내는지 본다.

틱 3분위에서 12종목씩(36종목), 학습 표본 2만 행, 30 epoch. 출력은 스크래치 임시 폴더.
1회차: 처음부터. 2회차: 예측값만 지우고 다시 — 저장된 교사·릿지를 불러와 예측한다.
두 회차 예측이 같아야 한다(다르면 교사 복원이 조용히 틀린 것이다).
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_run as gr
import group_analysis as ga

tmp = Path(tempfile.mkdtemp(prefix="smoke_group_",
           dir="/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad"))
assign = pd.read_csv(gr.ASSIGN, index_col=0)
assign.index = assign.index.astype(str).str.zfill(6)
sub = pd.concat([assign[assign.tick_group == g].sort_index().head(12) for g in gr.TICK_ORDER])
sub.to_csv(tmp / "assign.csv")

gr.OUT = tmp / "out"; gr.PROGRESS = gr.OUT / "group_run_progress.json"; gr.ASSIGN = tmp / "assign.csv"
gr.FIT_ROWS, gr.SELECT_ROWS, gr.EPOCHS = 20_000, 5_000, 30
gr.BLOCKS_PER_SLICE, gr.WORKERS = 5, 8
ga.PRED = gr.OUT / "predictions.parquet"; ga.ASSIGN = gr.ASSIGN; ga.OUT = gr.OUT / "group_results.json"

t0 = time.time(); gr.main(); print(f"=== 1회차(처음부터) {time.time()-t0:.0f}s ===", flush=True)
a = pd.read_parquet(ga.PRED)
shutil.copy(ga.PRED, tmp / "pred_run1.parquet"); ga.PRED.unlink()
t0 = time.time(); gr.main(); print(f"=== 2회차(저장본 복원) {time.time()-t0:.0f}s ===", flush=True)
b = pd.read_parquet(ga.PRED)

assert len(a) == len(b), f"행 수가 다르다 {len(a)} vs {len(b)}"
dd = float(np.nanmax(np.abs(a.pred_deeplob.to_numpy() - b.pred_deeplob.to_numpy())))
dr = float(np.nanmax(np.abs(a.pred_ridge.to_numpy() - b.pred_ridge.to_numpy())))
print(f"예측 행 {len(a):,} (eval1 {int((a.eval_set=='eval1').sum()):,}, eval2 {int((a.eval_set=='eval2').sum()):,})")
print(f"복원 재현성: DeepLOB 최대 차이 {dd:.3e}, 릿지 최대 차이 {dr:.3e}")
print(f"비유한 예측: DeepLOB {int((~np.isfinite(a.pred_deeplob)).sum())}, 릿지 {int((~np.isfinite(a.pred_ridge)).sum())}, "
      f"y {int((~np.isfinite(a.y_path)).sum())}")
prog = json.loads(gr.PROGRESS.read_text())
print("단계:", prog["stage"], "| 교사:", {k: prog.get("teacher", {}).get(k) for k in ("best_epoch", "best_score", "triggered")})
print("복원 판정:", "통과" if dd < 1e-5 and dr < 1e-8 else "실패")
print("임시 폴더:", tmp)
