"""경로·상수·해시의 단일 진실 원천. 다른 모듈은 값을 자기 안에 두지 않는다."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO / "vendor"
RUNS_ROOT = REPO / "runs"

TICK_ROOT = Path("/home/dgu/tick/tickdata_krx")
DATE = "20260316"

# SR 백엔드 선택 (ROADMAP.md 2단계). `naive` 가 기본값이다 — PySR 은 느리고
# (첫 fit 118초, Julia JIT 워밍업이 대부분), 배관을 빠르게 검증하는 경로를
# 그대로 유지해야 한다.
SR_BACKENDS: tuple[str, ...] = ("naive", "pysr")
DEFAULT_SR_BACKEND = "naive"

# PySR(juliacall)이 Julia 를 설치/캐시할 프로젝트 디렉터리.
#
# `juliapkg` 의 기본값은 conda 환경 안(`$CONDA_PREFIX/julia_env`)인데 그
# 디렉터리가 쓰기 불가라 **import 단계에서** `PermissionError` 로 죽는다
# (ROADMAP.md 2단계 실측). 사용자가 셸에서 환경변수를 기억해야 하는 설계는
# 피한다 — `load_framework()` 가 vendor 경로를 배선하는 것과 같은 자세로,
# `pysr` 를 import 하기 **전에** 여기서 배선한다.
PYSR_JULIA_PROJECT = Path.home() / ".julia_pysr_env"

# 정본 실행 프로필. Catalog 확장으로 바뀌지 않음을 실측 확인했다.
PROFILE_HASH = "d0b91a03541c0cec"

# Catalog 어휘 해시. sqrt·tanh 추가 후 값이며 실험 종료까지 동결한다.
# 원본(패치 전) 값은 448d4a81d9341432 였다.
CATALOG_HASH = "73344500dc6b2d7f"

# 슬라이스 전용 값 (DESIGN.md D7·D8).
QUANTILE_GRID = (0.70, 0.85, 0.95)
SLICE_PER_STRATUM = 5          # 6개 층 × 5 = 30종목
MIN_QUOTE_TICKS = 200          # 층화 대상 최소 정규장 호가틱
REPLAY_WORKERS = 32
SEED = 0


def load_framework() -> ModuleType:
    """vendor 사본을 import 가능하게 만들고 `framework` 패키지를 돌려준다.

    상위 저장소를 import 하지 않는다. `sys.path` 맨 앞에 vendor 를 넣어,
    같은 이름의 다른 `framework` 가 있어도 사본이 이긴다.
    """
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import framework                      # noqa: E402
    loaded = Path(framework.__file__).resolve()
    if VENDOR_ROOT not in loaded.parents:
        raise RuntimeError(f"vendor 가 아닌 framework 를 잡았다: {loaded}")
    return framework


def ensure_pysr_env() -> Path:
    """`import pysr` 보다 먼저 불러야 한다 — juliapkg 프로젝트 경로와 juliacall 의
    신호 처리를 배선한다.

    `PYTHON_JULIAPKG_PROJECT` 가 이미 설정돼 있으면(사용자가 직접 지정했든,
    이전 호출이 이미 배선했든) 그 값을 존중하고 덮어쓰지 않는다 — `os.environ`
    은 프로세스 전역이라 여러 번 불러도 안전해야 한다. 설정돼 있지 않으면
    `PYSR_JULIA_PROJECT`(쓰기 가능한 홈 디렉터리 아래)로 기본값을 채운다.

    `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` — juliacall 자신의 문서(`juliacall/
    __init__.py` 의 경고 문구)가 권하는 완화책이다: Julia 를 다중 스레드로
    시작했는데 `opt_handle_signals` 가 설정 안 돼 있으면 "세그폴트나 다른
    크래시를 겪을 수 있다"고 직접 경고한다. `run_slice.py` 는 `--sr-backend
    pysr` 를 고르기 전에 이미 `sd.teacher.shallow.ShallowMLP`(torch 기반)를
    임포트해 둔 상태라 juliacall 자신의 **다른** 경고("torch was imported
    before juliacall")도 뜬다 — 이 신호처리 설정이 **그 경고 자체를 없애지는
    않는다**(그 경고는 import 순서만 보고 무조건 뜬다, `handle_signals` 와
    무관). 이 설정이 완화하는 것은 신호 처리로 인한 크래시 위험이지 경고
    문구가 아니다 — 둘을 혼동하지 않도록 여기 적어 둔다. 실측(ROADMAP.md
    2단계 검증 run): 이 값을 켜도 그 경고는 실제로 그대로 뜬다.
    """
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(PYSR_JULIA_PROJECT))
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    return Path(os.environ["PYTHON_JULIAPKG_PROJECT"])
