"""minibatch_run.py 시험 운전 — 본 실행 전에 두 단계가 끝까지 돌고, 복원 경로가 같은 예측을 내는지 본다.

틱 3분위에서 12종목씩(36종목). 기준(전체 배치) 실행은 group_run 시험 운전과 같은 축소 설정.
미니배치는 사전등록 묶음 크기 8,192 그대로 두고 상한만 1 epoch 로 줄인다. 출력은 스크래치 임시 폴더.

점검
  1. 1단계 학습·선택 행이 기준 실행과 같다 — 행 수, 그리고 표준화 평균·척도가 기준 교사 저장본과 일치
  2. 1단계 예측을 지우고 다시 돌리면 저장된 교사를 불러와 같은 예측을 낸다
  3. 1단계 판정·짝지은 비교가 나온다
  4. 2단계(3일)가 끝까지 돌고, 학습 날짜가 3일이며 학습 행이 1단계보다 많고, 판정 문자열이 나온다
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_analysis as ga  # noqa: E402
import group_run as gr  # noqa: E402
import minibatch_run as mr  # noqa: E402
import minibatch_teacher as mbt  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  통과  " if cond else "  실패  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="smoke_mb_",
           dir="/tmp/claude-1003/-home-dgu-tick-symbolic/4a3e1c4b-3ad5-47d8-a5f5-5cdf1d5a7bb6/scratchpad"))
assign = pd.read_csv(gr.ASSIGN, index_col=0)
assign.index = assign.index.astype(str).str.zfill(6)
sub = pd.concat([assign[assign.tick_group == g].sort_index().head(12) for g in gr.TICK_ORDER])
sub.to_csv(tmp / "assign.csv")

gr.OUT = tmp / "out"; gr.PROGRESS = gr.OUT / "group_run_progress.json"; gr.ASSIGN = tmp / "assign.csv"
gr.FIT_ROWS, gr.SELECT_ROWS, gr.EPOCHS = 20_000, 5_000, 30
gr.BLOCKS_PER_SLICE, gr.WORKERS = 5, 8
ga.PRED = gr.OUT / "predictions.parquet"; ga.ASSIGN = gr.ASSIGN; ga.OUT = gr.OUT / "group_results.json"
mbt.MAX_EPOCHS = 1
mr.REQUIRE_GATE_PASS = False

t0 = time.time(); gr.main(); print(f"=== 기준(전체 배치) {time.time()-t0:.0f}s ===", flush=True)
t0 = time.time(); mr.main("gate"); print(f"=== 1단계 1회차 {time.time()-t0:.0f}s ===", flush=True)

base = json.loads(gr.PROGRESS.read_text())
gp = json.loads((mr.out_dir("gate") / "progress.json").read_text())
print("[1] 1단계 = 기준 실행과 같은 데이터")
check(gp["n_fit_rows"] == base["n_fit_rows"] and gp["n_select_rows"] == base["n_select_rows"],
      f"학습·선택 행 수 같음 ({gp['n_fit_rows']}, {gp['n_select_rows']})")
ident = [gp[f"lr_{mr.lr_tag(lr)}"]["identity_vs_fullbatch"] for lr in mbt.LR_CANDIDATES]
check(all(i is not None and i["close"] for i in ident), f"표준화 통계가 기준 교사 저장본과 일치 {ident[0]}")

print("[2] 복원 경로")
pred = mr.out_dir("gate") / "predictions.parquet"
a = pd.read_parquet(pred); pred.unlink()
t0 = time.time(); mr.main("gate"); print(f"=== 1단계 2회차(저장본 복원) {time.time()-t0:.0f}s ===", flush=True)
b = pd.read_parquet(pred)
d = float(np.nanmax(np.abs(a.pred_deeplob.to_numpy() - b.pred_deeplob.to_numpy()))) if len(a) == len(b) else np.inf
check(len(a) == len(b) and d < 1e-5, f"같은 예측 (행 {len(a):,}, 최대 차이 {d:.3e})")

print("[3] 1단계 판정·비교")
gres = json.loads((mr.out_dir("gate") / "results.json").read_text())
check(gres["gate"].get("passed") in (True, False), f"판정이 나온다 {gres['gate']}")
check(gres["vs_fullbatch_2M"] is not None and "verdict" not in gres["vs_fullbatch_2M"]["eval1"],
      "기준 예측과 행을 맞춰 비교했고, 1단계에는 판정 문구를 붙이지 않는다")

print("[4] 2단계 (3일)")
t0 = time.time(); mr.main("full"); print(f"=== 2단계 {time.time()-t0:.0f}s ===", flush=True)
fp = json.loads((mr.out_dir("full") / "progress.json").read_text())
fres = json.loads((mr.out_dir("full") / "results.json").read_text())
check(fp["train_dates"] == list(mr.FULL_DATES) and fp["select_dates"] == list(mr.FULL_DATES), f"학습·선택 날짜 {fp['train_dates']}")
check(fp["n_fit_rows"] > gp["n_fit_rows"], f"학습 행이 1단계보다 많다 ({fp['n_fit_rows']:,} > {gp['n_fit_rows']:,})")
check(isinstance(fres.get("verdict"), str), f"판정 문자열 '{fres.get('verdict')}'")
e1 = fres["vs_fullbatch_2M"]["eval1"]
print(f"    평가① R² 기준 {e1['r2_a']:+.5f} / 3일 {e1['r2_b']:+.5f}, 차이 [{e1['delta_lo']:+.5f}, {e1['delta_hi']:+.5f}], "
      f"대조군 위반 {len(fres['negative_control_violations'])}건")

print("\n임시 폴더:", tmp)
if failures:
    print(f"실패 {len(failures)}건:"); [print("  -", f) for f in failures]; sys.exit(1)
print("전부 통과")
