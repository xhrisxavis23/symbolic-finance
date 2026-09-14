"""2단계 공용 — 경로·상수·표식 AST·평가 종목·계약. PREREG-TEACHER-ECON.md §3.

`0902/` 의 sd 패키지와 vendor 프레임워크, `0911/groups` 의 분할 함수를 읽기 전용으로 쓴다.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

for _p in ("/home/dgu/tick/symbolic/0902", "/home/dgu/tick/symbolic/0911/groups",
           "/home/dgu/tick/symbolic/0911/scale", "/home/dgu/tick/symbolic/0911/gpu_eval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
from sd import config  # noqa: E402

config.load_framework()

ROOT = Path("/home/dgu/tick/symbolic/0914")
STAGE2 = ROOT / "stage2"
SCORES = STAGE2 / "scores"
OUT = STAGE2 / "out"
LOGS = ROOT / "logs"

DATE = "20260319"
QS = (0.70, 0.85, 0.95)
PARAM = "score"
THETA = f"theta_{PARAM}"
MODELS = ("T2M", "T3D", "RDG")

G = Path("/home/dgu/tick/symbolic/0911/groups")
CKPT = {"T2M": G / "out" / "deeplob.pt", "T3D": G / "out" / "mb_full" / "teacher_lr0.001.pt"}
RIDGE = G / "out" / "ridge.npz"
ASSIGN = G / "data" / "group_assignment.csv"
FEATURES_JSON = G / "out" / "mb_full" / "progress.json"
PRED_2M = G / "out" / "predictions.parquet"
PRED_3D = G / "out" / "mb_full" / "predictions.parquet"
SELECT_0910 = Path("/home/dgu/tick/symbolic/0910/run/state/select.json")

N_BOOT = 10_000
SEED = 0
N_TESTS = 12
BONF = 0.05 / N_TESTS
MIN_SCORABLE = 200
C1_TOL = 1e-5
C3_SYMBOLS = 20
C4_SYMBOLS = 5
REPLAY_WORKERS = 32
GPU_UUID = "GPU-dcd70e26-59e0-7835-941a-4a0aabb186fa"   # 물리 GPU0


def _twice(primitive_id: str) -> dict:
    """표식 AST — 비교군이 쓰지 않는 `x + x` 꼴. 값은 계산되지 않고 교사 점수로 대체된다."""
    p = {"op": "primitive", "primitive_id": primitive_id}
    return {"op": "add", "left": p, "right": dict(p)}


MARKERS = {
    "T2M": _twice("book_imbalance"),
    "T3D": _twice("queue_imbalance_best"),
    "RDG": _twice("spread_to_round_trip_cost_ratio"),
    "FAKE": _twice("signed_aggr_flow_20"),       # C3 전용: 점수 = book_imbalance
}


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def eval_symbols() -> list[str]:
    """0911 분할(group_run.make_split, 시드 0)의 평가 종목 411개."""
    import group_run as gr
    assign = pd.read_csv(ASSIGN, index_col=0)
    assign.index = assign.index.astype(str).str.zfill(6)
    split = gr.make_split(assign)
    symbols = sorted(s for s, v in split.items() if v == "eval")
    if len(symbols) != 411:
        raise SystemExit(f"평가 종목이 411개가 아니다: {len(symbols)}")
    return symbols


def score_path(model: str, symbol: str) -> Path:
    return SCORES / model / f"{symbol}.npz"


def compare_q(input_ast: dict) -> dict:
    return {"op": "compare", "input": input_ast, "comparator": ">", "value": f"UNRESOLVED:{THETA}"}


def abs_template(model: str) -> dict:
    """ABS — 분위 파라미터 없이 숫자 임계 0. 실행 결속을 선언하지 않는다(정본이 정한다)."""
    return {"entry_program": {"signal": {"op": "compare", "input": MARKERS[model], "comparator": ">",
                                         "value": 0.0}, "warmup_ticks": 0}}


def q_template(input_ast: dict) -> dict:
    from sd.compile import contract_template
    return contract_template(compare_q(input_ast), PARAM)


def git_commit(paths: list[str], message: str) -> None:
    """산출물을 0914 저장소에 커밋한다. 실패해도 실행을 막지 않는다."""
    body = (message + "\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_018orTRo7BHm9dsHymtfVwVh")
    try:
        subprocess.run(["git", "add", *paths], cwd=ROOT, check=True, capture_output=True, text=True)
        r = subprocess.run(["git", "commit", "-q", "-m", body], cwd=ROOT, capture_output=True, text=True)
        log("git", f"commit rc={r.returncode} {r.stderr.strip()[:200]}")
    except Exception as error:  # noqa: BLE001
        log("git", f"commit 실패(계속): {type(error).__name__}: {error}")
