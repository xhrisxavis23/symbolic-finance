"""무차원 좌표. 계획서 §3 S3.

규칙 하나다 — **차원 있는 feature 는 SR 입력이 될 수 없다.** 종목 스케일이
수식에 들어오면 SR 은 유동성 등급을 설명하는 항을 붙이고 그것을 알파로 착각한다.

Catalog 가 이미 모든 feature 에 `dimension` 을 붙여 두었으므로 여기서 다시
분류하지 않는다. Task 2 에서 `sqrt`·`tanh` 를 무차원 입력만 받게 만든 것도
같은 규칙을 타입 수준에서 강제한 것이다.

## 파생 열 (ROADMAP.md 1단계)

이미 무차원인 Catalog feature 만으로는 어휘가 9개(실효 8개)뿐이라 가격
움직임·OFI 그룹을 SR 이 표현할 수 없었다 (DESIGN.md D14). `sd/derived.py`
가 `ratio`·`rolling_zscore` 로 조립한 파생 무차원 열을 등록해 두면, 여기서
그 AST 를 `framework.contract.ExpressionRuntime` — 백테스트가 쓰는 바로 그
실행기 — 로 평가해 열을 만든다. 별도 numpy 구현을 두지 않는다.

`arrays`(한 종목-일의 원본 tick 배열)를 넘겨야 파생 열이 계산된다. 넘기지
않으면(기본값 `None`) 예전과 완전히 같게 동작한다 — 기존 시그니처·반환
형태·호출자를 깨지 않는다.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from . import config, derived as _derived

_framework = config.load_framework()
from framework import catalog as _catalog      # noqa: E402
from framework import contract as _contract    # noqa: E402

DIMENSIONLESS_FEATURES: tuple[str, ...] = tuple(sorted(
    name for name, spec in _catalog.FEATURES.items()
    if spec.type_info.dimension == "dimensionless"
))


def _compute_derived(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """등록된 파생 열 전부를 `ExpressionRuntime` 으로 평가한다.

    한 종목-일에서 계산이 안 되는 것(예: 필요한 원본 필드가 없는 데이터셋)은
    예외를 삼키지 않고 실패 사유를 함께 돌려준다 — `compute_features` 가 계산
    불가 feature 를 조용히 빼는 것과 같은 태도다.
    """
    values: dict[str, np.ndarray] = {}
    failed: dict[str, str] = {}
    runtime = _contract.ExpressionRuntime(arrays)
    for name, ast in _derived.DERIVED.items():
        try:
            values[name] = np.asarray(runtime.evaluate(ast), dtype=float)
        except Exception as error:  # noqa: BLE001 - 실패 사유를 그대로 보고한다
            failed[name] = f"{type(error).__name__}: {error}"
    return values, failed


def transform(matrix: np.ndarray, names: tuple[str, ...], *,
              arrays: Mapping[str, np.ndarray] | None = None,
              ) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """무차원 열(원본 + 파생)만 남긴다. 무엇을 왜 뺐는지 함께 돌려준다.

    `arrays` 를 주면 `sd/derived.py` 의 파생 무차원 열도 계산해 덧붙인다.
    주지 않으면 원본 무차원 feature 만 거르던 예전 동작 그대로다.

    **열 순서 계약**: 반환하는 `names` 는 항상 정렬돼 있고, 반환하는 행렬의
    열 `j` 는 정확히 `names[j]` 의 값이다. 원본 열이든 파생 열이든 예외 없다.
    """
    dimensionless = set(DIMENSIONLESS_FEATURES)
    keep = tuple(sorted(n for n in names if n in dimensionless))
    dropped = {n: _catalog.FEATURES[n].type_info.dimension
               for n in names if n not in dimensionless}

    derived_values: dict[str, np.ndarray] = {}
    derived_failed: dict[str, str] = {}
    if arrays is not None:
        derived_values, derived_failed = _compute_derived(arrays)

    if not keep and not derived_values:
        raise ValueError("무차원 feature 가 하나도 남지 않았다 (원본도 파생도 없다)")

    all_names = tuple(sorted(set(keep) | set(derived_values)))
    columns = []
    for name in all_names:
        if name in derived_values:
            columns.append(derived_values[name])
        else:
            columns.append(matrix[:, names.index(name)])
    result = np.column_stack(columns)

    meta = {
        "kept": list(keep),
        "dropped": dropped,
        "derived": sorted(derived_values),
        "derived_failed": derived_failed,
        "rule": "Catalog dimension == 'dimensionless' 인 feature + "
                "sd.derived 레지스트리의 무차원 파생 열만 SR 입력",
    }
    return result, all_names, meta
