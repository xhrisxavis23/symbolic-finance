"""결정값 한 곳.

구 구조에서는 수수료가 `ledger.py`, 평가 창이 `ledger.py`, 표본 바닥이 `agents.py` 와
`loop.py` 두 곳, 날짜 블록이 `__main__.py`, 틱 경로가 `data.py` 에 절대경로로 박혀
있었다. 재현에 필요한 값이 흩어져 있으면 run 하나를 설명할 수 없다.

숫자를 여기 두는 것은 그것이 **연구 판단**이기 때문이다. 계산 편의상 생긴 상수는
쓰는 모듈에 두고 여기 올리지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def execution_workers(default: int | None = None) -> int:
    """연구 replay 한 건이 사용할 worker 수."""
    raw = os.environ.get("FRAMEWORK_WORKERS")
    if raw is None:
        if default is not None:
            return int(default)
        return min(8, max(1, int(os.cpu_count() or 1) // 8))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("FRAMEWORK_WORKERS는 양의 정수여야 한다") from error
    if workers < 1:
        raise ValueError("FRAMEWORK_WORKERS는 1 이상이어야 한다")
    return workers


def research_run_slots(default: int | None = None) -> int:
    """동시에 실행할 수 있는 연구 replay 수."""
    raw = os.environ.get("FRAMEWORK_RUN_SLOTS")
    if raw is None:
        if default is not None:
            return int(default)
        cores = max(1, int(os.cpu_count() or 1))
        return min(4, max(1, cores // max(1, execution_workers() * 2)))
    try:
        slots = int(raw)
    except ValueError as error:
        raise ValueError("FRAMEWORK_RUN_SLOTS는 양의 정수여야 한다") from error
    if slots < 1:
        raise ValueError("FRAMEWORK_RUN_SLOTS는 1 이상이어야 한다")
    return slots


# ---- 데이터 위치 -------------------------------------------------------------
# 절대경로를 코드에 박으면 다른 계정에서 돌지 않는다.
TICK_ROOT = _env_path("TICK_ROOT", "~/tick/tickdata_krx")
# Backtest가 반복해 쓰는 정규장 호가·체결 배열 저장소. 없거나 원본과 다르면 자동으로
# 원본 parquet을 읽으므로, cache 자체가 데이터 원본이나 실행 결과를 바꾸지 않는다.
BACKTEST_CACHE_ROOT = _env_path(
    "BACKTEST_CACHE_ROOT", str(REPO / "framework/BacktestCache/data_pair_v9"))
# 카탈로그는 파일이 아니라 `catalog.py` 의 레지스트리가 정본이다. 경로 상수는 두지 않는다.

# 수익 구간 저장소. BID1 매수 지정가와 미래 ASK1 매도 지정가의 가격 기회 라벨만 담는다.
# 과거 `iterations/` 캐시는 반대 가격 방향이라 읽지 않는다.
PROFIT_CACHE = _env_path("PROFIT_CACHE", str(REPO / "framework/ProfitRegion"))


# ---- 거래 비용과 평가 창 --------------------------------------------------------
FEE_BPS = 23.0                      # 왕복. 진입·청산 합계
PRIMARY_HORIZON_SECONDS = 30        # 성과를 재는 기준 평가 창
PATH_SECONDS = (5, 10, 15, 20, 30)  # 경로를 기록할 시점
EARLY_WINDOW_SECONDS = 10.0         # "초반 반등" 의 경계


# ---- 실행 ---------------------------------------------------------------------
# 지정가가 기본이다. 수익원이 스프레드를 내지 않는 데 있는 가설을 taker 로 재면
# 그 수익원을 재는 것 자체가 불가능하다. 큐 커널이 있으므로 "못 사는 것" 과
# "못 파는 것" 이 결과의 일부로 계산된다 — 그게 지정가 전략의 실제 경제다.
LIMIT_ENTRY = True
LIMIT_EXIT = True

# 진입 지정가를 얼마나 오래 세워 둘 것인가. 이건 전략 선택이다 — 오래 둘수록
# 체결률은 오르고 역선택도 커진다.
ENTRY_ORDER_MAX_SECONDS = 10.0
ENTRY_ORDER_MAX_TICKS = 100

# Feature Profile에 붙이는 BID1 앞 대기 물량 가정. 수익 구간 자체는 바꾸지 않고,
# 그 구간의 anchor에서 주문을 냈을 때의 체결 시점만 보조 정보로 남긴다.
FEATURE_PROFILE_ENTRY_QUEUE_FRACTION = 0.20

# 청산의 최대 보유 시간. S30 Feature Profile 라벨과 별개로, 실제 체결 뒤에는 이 시간까지
# 손절·추적 청산을 기다린다.
EXIT_MAX_HOLD_SECONDS = 900.0

# 청산 지정가는 최대 보유 시간 끝까지 세워 둔다. 중간에 거두면 그때부터 시간 상한 끝까지가
# 무엇도 아닌 구간이 된다.
EXIT_ORDER_RESTS_TO_HORIZON = True

# 추적 청산. **gross 기준** — 진입 뒤 가장 높았던 ASK1 에서 이만큼 밀리면 ASK1 에
# 지정가 청산을 시작한다. 최고가는 언제나 지금까지 관측된 ASK1 만 쓴다.
TRAILING_DRAWDOWN_GROSS_BPS = -30.0
# 트리거 뒤 ASK1 지정가를 얼마나 세워 둘 것인가. 안 팔리면 BID1 시장가.
EXIT_LIMIT_REST_SECONDS = 60.0

# 손절. **gross 기준** — 가격이 이만큼 밀리면 그 자리 BID1 에 던진다.
# net 으로 잡으면 BID1 에 사는 순간 이미 -수수료 에서 시작해 여유가 그만큼 줄어든다.
# 실측: gross -30 은 19.6%, net -30(= gross -7)은 56.5% 에서 발동했다.
# 진입가 대비 gross -120bp 이하면 즉시 시장가 청산.
STOP_GROSS_BPS = -120.0


# ---- 표본 바닥 ---------------------------------------------------------------
# decision 기준이다. signal tick 기준이 아니다 — tick 은 신호 지속시간에 비례해
# 부풀고, 그러면 바닥이 "촘촘한 신호" 를 통과시키는 필터가 된다.
MIN_DECISIONS = 200
MIN_DECISION_FRACTION = 0.10        # 라운드를 거듭해도 원래의 이 비율 아래로는 안 간다
QUANTILES = (0.10, 0.20, 0.25, 0.40, 0.50, 0.75, 0.80, 0.90)  # 직전 100 tick 임계 분위수
MIN_REFERENCE_SAMPLES = 200         # 전날 분위수를 인정할 최소 관측 수

# Sandbox 확장 Entry Refinement의 결과·재개 identity. 실행 프로필은 바꾸지 않는다.
ENTRY_REFINEMENT_POLICY_VERSION = "feedback-full-entry-expression.v13"


class BlockOverlap(ValueError):
    """블록끼리 날짜가 겹친다. 한 블록이 다른 블록의 증거를 빌려 쓰게 된다."""


@dataclass(frozen=True)
class Blocks:
    """날짜 블록."""

    discovery: tuple[str, ...]
    validation: tuple[str, ...] = ()
    oos: tuple[str, ...] = ()
    reference: tuple[str, ...] = ()      # 전날 분위수를 뜰 때만 읽는 날
    replication: tuple[str, ...] = ()    # 후보 재현을 확인할 날

    def validate(self) -> None:
        """블록끼리 날짜가 겹치는지 검사한다. 겹치면 예외.

        `replication` 은 예외다 — 설계상 발굴 구간 안의 날을 쓴다. 다만 발굴 구간과
        완전히 같으면 재현 확인이 되지 않으므로 그 경우도 예외로 막는다.
        """
        named = {
            "discovery": set(self.discovery),
            "validation": set(self.validation),
            "oos": set(self.oos),
        }
        for left in ("discovery", "validation"):
            for right in ("validation", "oos"):
                if left == right:
                    continue
                shared = named[left] & named[right]
                if shared:
                    raise BlockOverlap(
                        f"{left} 와 {right} 가 날짜를 공유한다: {sorted(shared)}"
                    )
        if self.replication:
            outside = set(self.replication) - set(self.discovery)
            if outside:
                raise BlockOverlap(
                    f"재현 확인일이 발굴 구간 밖이다: {sorted(outside)}. "
                    "검증·OOS 를 쓰면 그 구간이 검증이 아니게 된다"
                )
            if set(self.replication) == set(self.discovery):
                raise BlockOverlap(
                    "재현 확인일이 발굴 구간과 같다. 후보를 고른 데이터로 다시 재는 "
                    "것이라 아무것도 걸러내지 못한다"
                )

    @property
    def reference_pool(self) -> tuple[str, ...]:
        """전날 분위수를 찾을 때 훑을 날짜 전부. 정렬해 중복 제거."""
        return tuple(sorted({*self.reference, *self.discovery, *self.validation, *self.oos}))


DEFAULT_BLOCKS = Blocks(
    discovery=("20260316", "20260317", "20260318", "20260319", "20260320"),
    validation=("20260330", "20260331", "20260401"),
    oos=("20260413", "20260414", "20260415"),
    reference=("20260316",),
    replication=(),
)


# ---- 직렬화 ------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> Any:
    """JSON 으로 쓸 수 있게 정리. NaN·Inf 는 null 로, numpy 스칼라는 파이썬 값으로."""
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """원자적 저장. 부분 기록된 JSON 이 남지 않게 임시 파일에 쓰고 교체한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_json(value: Any) -> str:
    """정렬된 JSON 의 해시. 같은 내용이면 항상 같은 값."""
    payload = json.dumps(
        clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(repo: Path = REPO) -> dict[str, Any]:
    """커밋과 작업 트리 상태. dirty 인 run 은 재현되지 않는다는 뜻이다."""

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=False
        )
        return completed.stdout.rstrip("\n")

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


# ---- run 기록 ----------------------------------------------------------------

def run_manifest(output: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """이 run 을 재현하는 데 필요한 전부. 산출물마다 한 부씩 남긴다."""
    from . import catalog, universe  # 순환 import 를 피해 여기서 부른다

    return {
        "schema": "framework_run_manifest.v1",
        "created_at": now_utc(),
        "output": str(output),
        "git": git_snapshot(),
        "tick_root": str(TICK_ROOT),
        # 어떤 어휘·어떤 관측 통계·어떤 가용 정책으로 돌았는가. 셋이 다 있어야 재현된다.
        **catalog.provenance(),
        "universe_sha256": universe.members_sha256(),
        "execution": {
            "limit_entry": LIMIT_ENTRY, "limit_exit": LIMIT_EXIT,
            "entry_order_max_seconds": ENTRY_ORDER_MAX_SECONDS,
            "entry_order_max_ticks": ENTRY_ORDER_MAX_TICKS,
            "exit_order_rests_to_horizon": EXIT_ORDER_RESTS_TO_HORIZON,
            "exit_max_hold_seconds": EXIT_MAX_HOLD_SECONDS,
            "stop_gross_bps": STOP_GROSS_BPS,
            "trailing_drawdown_gross_bps": TRAILING_DRAWDOWN_GROSS_BPS,
            "exit_limit_rest_seconds": EXIT_LIMIT_REST_SECONDS,
            "cancel_on_price_change": False,
        },
        "costs": {
            "fee_bps": FEE_BPS,
            "primary_horizon_seconds": PRIMARY_HORIZON_SECONDS,
            "path_seconds": list(PATH_SECONDS),
            "early_window_seconds": EARLY_WINDOW_SECONDS,
        },
        "sample_floor": {
            "min_decisions": MIN_DECISIONS,
            "min_decision_fraction": MIN_DECISION_FRACTION,
            "denominator": "scorable_decisions",
        },
        "thresholds": {
            "quantiles": list(QUANTILES),
            "min_reference_samples": MIN_REFERENCE_SAMPLES,
            "policy": "prior_valid_day_per_symbol",
        },
        **(dict(extra) if extra else {}),
    }
