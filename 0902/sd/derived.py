"""파생 무차원 열 — 이름 하나당 정의도 하나. 계획서 §3 S3 "무차원 그룹 사전".

## 왜 이 파일이 있는가 (ROADMAP.md 1단계 · DESIGN.md D14)

SR 이 보는 무차원 어휘는 원래 9개뿐이었고, 그 중 둘은 아핀 중복이라 실효
8개였다. `return`(가격 움직임) 6개와 OFI 3개가 통째로 빠져 있어 계획서
§3 S3 의 대표 무차원 그룹 — `OFI / Q̄`, `ΔP / s` — 을 SR 이 표현조차
할 수 없었다.

## 설계 — AST 가 곧 정의다

파생 열은 두 곳에서 쓰인다: SR 이 적합할 **행렬의 열**(수치)과, 컴파일러가
진입식으로 옮길 **Catalog AST**. 두 정의를 따로 두면 조용히 갈라진다 — 이
저장소가 참조하는 정본 프레임워크가 `metrics.py` 에서 겪은 사고와 같은
형태다 (같은 지표를 세 곳에 구현했다가 하나만 고쳐져 갈라졌다).

그래서 여기서는 **AST 만** 등록한다. 수치는 이 AST 를
`framework.contract.ExpressionRuntime` 으로 평가해 얻는다 (`sd/dimensionless.py`
가 그 호출자다) — 백테스트가 실제로 쓰는 바로 그 실행기다. 별도 numpy
구현은 두지 않는다. 컴파일러(`sd/compile/to_catalog.py`)는 심볼 조회가
Catalog `FEATURES` 에서 실패하면 이 레지스트리를 폴백으로 보고, 등록된
AST 를 그대로 진입식에 옮긴다.

## 각 항목이 무엇에 대응하는가

`_register()` 호출마다 `plan_item`(계획서 §3 S3 무차원 그룹 사전의 행 이름)과
`why`(왜 이 조합인지)를 남긴다. 대응하는 사전 항목이 없으면 여기 넣지
않는다 — 아래에 사전 11항목 전체와 이 파일에서 다루는 범위를 적어둔다.

| 사전 항목 | 이 파일에서 |
| --- | --- |
| 큐 불균형 `I` | 이미 `book_imbalance`·`queue_imbalance_best` 로 존재. 파생 불필요 |
| 마찰비 `s/(s+c)` | 이미 `spread_to_round_trip_cost_ratio` 로 존재. 파생 불필요 |
| 정규화 스프레드 `s/δ` | **제외.** `δ`(KRX 호가단위)는 가격대별 계단함수라 Catalog raw/feature 로 없다. 손으로 계산하면 두 번째 진실 원천이 된다 |
| 심도 집중도 `Q_1/Σ Q_k` | 이미 `ask_depth_concentration`·`bid_depth_concentration` 로 존재. 파생 불필요 |
| 정규화 심도 `Q_k/Q̄` | `bid_l1_depth_over_qbar` |
| 정규화 주문흐름 `OFI/Q̄` | `ofi_l1_over_qbar` · `ofi_over_qbar_5`(태스크 지시 예시) · `ofi_over_qbar_10` |
| 서명 체결 강도 | 이미 `signed_aggr_flow_20`·`signed_aggr_flow_100` 로 존재. 파생 불필요 |
| 정규화 가격변화 `ΔP/s` | `mid_return_5t_over_spread`·`mid_return_20t_over_spread`·`mid_return_100t_over_spread`(태스크 지시 예시) · `microprice_velocity_over_spread`(대안 ΔP) |
| z-정규화 상태 `(x-μ)/σ` | `ofi_depth_5_zscore` · `microprice_velocity_zscore` |
| 깊은 층 불균형 | 이미 `deep_depth_imbalance_6_10` 로 존재. 파생 불필요 |
| 불균형 강도비 `α/β`(Hawkes) | **제외.** 계획서가 명시적으로 "E2 전용"이라 적었다 — 이 슬라이스(S3) 범위 밖 |

`microprice_dev_bps / spread_bps` 는 일부러 넣지 않았다 — Catalog 자신의
`feature_use_policy('microprice_dev_bps').note` 가 "microprice_dev_bps /
spread_bps = book_imbalance / 2" 라고 실측으로 적어 둔 **알려진 정확한
중복**이다. 그것을 파생 열로 등록하면 D14 가 경고한 바로 그 아핀 중복
축을 하나 더 만드는 것이다.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from . import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402

# Q̄(최우선 매수 잔량의 롤링 평균)·rolling_zscore 둘 다 "직전 몇 틱을 보는가"가
# 구조 파라미터다 (Catalog STRUCTURAL_PARAMS — 탐색 대상이 아니다). 정본
# 프레임워크가 이미 "직전 100틱"을 프레임워크 전역 관례로 못박아 뒀다
# (vendor/framework/contract.py ROLLING_QUANTILE_WINDOW = 100,
# THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE). 여기서 다른 창을 고르면 같은
# "직전 몇 틱" 질문에 두 번째 답을 만드는 것이므로 그 값을 그대로 재사용한다.
QBAR_WINDOW = 100

# z-score 창도 같은 이유로 100. min_observations 는 그 20% — 정본 테스트
# 픽스처(`vendor/framework/tests/test_catalog.py`)가 window=50/min_obs=20,
# window=37/min_obs=15 로 40% 안팎을 쓰길래 그 비율대의 하한을 골랐다.
# 너무 크게 잡으면(예: window 전체) 장 초반 표본이 전부 0(미사용)으로
# 죽고, 너무 작게 잡으면 분산 추정이 불안정해 z 가 튄다.
ZSCORE_WINDOW = 100
ZSCORE_MIN_OBSERVATIONS = 20


DERIVED: dict[str, dict[str, Any]] = {}
# 이름 -> (계획서 사전 항목, 왜 이 조합인가). 디버그·리포트용이지 검증에는 안 쓴다.
PLAN_MAPPING: dict[str, tuple[str, str]] = {}


def _primitive(name: str) -> dict[str, Any]:
    return {"op": "primitive", "primitive_id": name}


def _ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any],
           *, zero_policy: str = "nan") -> dict[str, Any]:
    return {"op": "ratio", "numerator": dict(numerator), "denominator": dict(denominator),
            "zero_policy": zero_policy}


def _rolling_mean(input_ast: Mapping[str, Any], *, window: int) -> dict[str, Any]:
    return {"op": "rolling_mean", "input": dict(input_ast), "window": int(window),
            "time_basis": "tick"}


def _rolling_zscore(input_ast: Mapping[str, Any], *, window: int,
                     min_observations: int) -> dict[str, Any]:
    return {"op": "rolling_zscore", "input": dict(input_ast), "window": int(window),
            "min_observations": int(min_observations), "time_basis": "tick"}


def _qbar_l1() -> dict[str, Any]:
    """Q̄ — 최우선 매수 게시잔량의 직전 `QBAR_WINDOW` 틱 평균.

    `bid_queue_depth_at_l1` 을 쓰는 이유: 매 틱 항상 계산되는(체결 유무와
    무관한) 유일한 단일 큐 깊이 feature 다. ask 쪽 대응 feature 는 Catalog에
    없다 — 만들면 그 자체가 두 번째 진실 원천이 되므로 만들지 않는다.
    """
    return _rolling_mean(_primitive("bid_queue_depth_at_l1"), window=QBAR_WINDOW)


def _register(name: str, ast: Mapping[str, Any], *, plan_item: str, why: str) -> None:
    """레지스트리에 넣는다. 이름 충돌과 차원을 **여기서** 검사한다 — import 시점에.

    잘못된 항목이 조용히 들어가면 안 된다. 어느 하나라도 실패하면 이 모듈을
    import 하는 즉시(테스트 수집 시점 포함) 예외가 나야 한다.
    """
    if name in _catalog.FEATURES or name in _catalog.ALIASES:
        raise ValueError(
            f"파생 열 이름 {name!r} 이 Catalog feature/별칭과 겹친다 — "
            "이름을 바꾸거나 그 Catalog feature 를 직접 써야 한다")
    if name in DERIVED:
        raise ValueError(f"파생 열 {name!r} 이 이미 등록돼 있다")
    info = _catalog.infer_expression_type(ast)
    if info.dimension != "dimensionless":
        raise ValueError(
            f"파생 열 {name!r} 의 AST 가 무차원이 아니다 (dimension={info.dimension!r}). "
            "무차원이 아닌 파생 열은 SR 입력이 될 수 없다 (DESIGN.md D14 규칙)")
    DERIVED[name] = dict(ast)
    PLAN_MAPPING[name] = (plan_item, why)


# ---------------------------------------------------------------------------
# 정규화 주문흐름 OFI / Q̄ — L2 법칙(Cont-Kukanov-Stoikov)의 인수.
# ---------------------------------------------------------------------------
_register(
    "ofi_over_qbar_5",
    _ratio(_primitive("ofi_depth_5"), _qbar_l1()),
    plan_item="정규화 주문흐름 OFI/Q̄",
    why="태스크가 지시한 필수 예시: ratio(ofi_depth_5, rolling_mean(bid_queue_depth_at_l1))")
_register(
    "ofi_l1_over_qbar",
    _ratio(_primitive("ofi_cks_1"), _qbar_l1()),
    plan_item="정규화 주문흐름 OFI/Q̄",
    why="최우선호가만의 원형 CKS OFI(ofi_cks_1)를 같은 Q̄로 무차원화 — "
        "L2 법칙이 원래 가리키는 가장 얕은 형태")
_register(
    "ofi_over_qbar_10",
    _ratio(_primitive("ofi_depth_10"), _qbar_l1()),
    plan_item="정규화 주문흐름 OFI/Q̄",
    why="1~10호가까지의 OFI를 같은 Q̄로 무차원화 — 더 깊은 구간의 순주문흐름")

# ---------------------------------------------------------------------------
# 정규화 가격변화 ΔP / s.
# ---------------------------------------------------------------------------
for _lag in (5, 20, 100):
    _register(
        f"mid_return_{_lag}t_over_spread",
        _ratio(_primitive(f"mid_return_{_lag}t_bps"), _primitive("spread_bps")),
        plan_item="정규화 가격변화 ΔP/s",
        why=f"태스크가 지시한 필수 예시: mid_return_{_lag}t_bps / spread_bps "
            f"(시간 스케일 {_lag}t)")
del _lag
_register(
    "microprice_velocity_over_spread",
    _ratio(_primitive("microprice_velocity"), _primitive("spread_bps")),
    plan_item="정규화 가격변화 ΔP/s",
    why="ΔP 를 mid 의 5/20/100틱 지연수익률이 아니라 microprice 의 1틱 변화로 "
        "잡은 대안 — 더 짧은 시간 스케일의 가격변화")

# ---------------------------------------------------------------------------
# 정규화 심도 Q_k / Q̄.
# ---------------------------------------------------------------------------
_register(
    "bid_l1_depth_over_qbar",
    _ratio(_primitive("bid_queue_depth_at_l1"), _qbar_l1()),
    plan_item="정규화 심도 Q_k/Q̄",
    why="최우선 매수 잔량이 자신의 직전 100틱 평균 대비 지금 얼마나 두꺼운가. "
        "OFI/Q̄ 와 분모(Q̄)를 공유하는 상태(state) 짝")

# ---------------------------------------------------------------------------
# z-정규화 상태 (x-μ_w)/σ_w — "rolling_zscore 가 무차원화의 만능 어댑터".
# ratio 계열(OFI/Q̄, ΔP/s)과는 다른 정규화 기준(외부 스케일이 아니라 자기
# 자신의 최근 분포)이라 정보가 겹치지 않는다. 무작정 모든 feature 에 씌우지
# 않고, OFI·가격변화 그룹 각각에서 하나씩만 대표로 추가한다.
# ---------------------------------------------------------------------------
_register(
    "ofi_depth_5_zscore",
    _rolling_zscore(_primitive("ofi_depth_5"), window=ZSCORE_WINDOW,
                     min_observations=ZSCORE_MIN_OBSERVATIONS),
    plan_item="z-정규화 상태 (x-μ_w)/σ_w",
    why="ofi_depth_5(quantity)를 자기 분포 기준으로 정규화 — OFI/Q̄(외부 스케일) "
        "와 다른 정규화 축")
_register(
    "microprice_velocity_zscore",
    _rolling_zscore(_primitive("microprice_velocity"), window=ZSCORE_WINDOW,
                     min_observations=ZSCORE_MIN_OBSERVATIONS),
    plan_item="z-정규화 상태 (x-μ_w)/σ_w",
    why="microprice_velocity(return, bps)를 자기 분포 기준으로 정규화 — "
        "ΔP/s(외부 스케일인 spread_bps)와 다른 정규화 축")


DERIVED_NAMES: tuple[str, ...] = tuple(sorted(DERIVED))


def get(name: str) -> dict[str, Any]:
    """등록된 AST 의 깊은 복사본. 호출자가 반환값을 변형해도 레지스트리는 그대로다."""
    return copy.deepcopy(DERIVED[name])
