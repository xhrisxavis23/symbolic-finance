"""경로·상수·해시의 단일 진실 원천. 다른 모듈은 값을 자기 안에 두지 않는다."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO / "vendor"
RUNS_ROOT = REPO / "runs"

TICK_ROOT = Path("/home/dgu/tick/tickdata_krx")
DATE = "20260316"

# 정본 실행 프로필. Catalog 확장으로 바뀌지 않음을 실측 확인했다.
PROFILE_HASH = "d0b91a03541c0cec"

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
