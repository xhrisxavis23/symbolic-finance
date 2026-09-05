"""종목 단위 표본외 분할. 사전등록 §1-1(PREREG-E0-V2.md).

"현재에도 있는 층화 코드"를 다시 만들지 않는다 — `sd.universe.assign_strata`
가 이미 계산한 층 딱지를 그대로 받아쓴다. 이 모듈이 하는 일은 그 층 안에서
종목을 적합(fit)/선택(select) 두 서로소 집합으로 나누는 것뿐이다.

## 설계 판단 (사전등록에 숫자가 없다 — `.stage4a-report.md` §"분할 설계"에
근거와 틀렸을 때의 비용을 적는다)

- **비율**: 적합:선택 = 2:1 (`DEFAULT_FIT_FRACTION = 2/3`). SR 적합은 다양한
  종목 조합을 봐야 구조를 찾는다(적합 종목이 너무 적으면 애초에 못 찾는
  문제가 된다 — 이건 채점 규칙으로 못 고친다). 선택 쪽은 층당 최소 2종목을
  가능한 한 확보해 종목 하나의 특이값이 판정 전체를 좌우하지 않게 한다.
  `per_stratum=6`(실행 계획값) 기준 층당 4적합/2선택.
- **표본이 모자란 층(층 내 종목 1개)**: 나눌 수 없다 — 전부 적합으로 보낸다.
  선택 집합에서 그 층은 대표되지 않는다. 대안(선택으로 보내기)은 그 층에서
  SR 이 아예 fit 할 데이터가 없어지므로 더 나쁘다.
- **분할축**: 종목. 시간 축은 쓰지 않는다 — PREREG-E0-V2.md §1-1 근거
  ("종목 고유 노이즈 암기"가 실패 양상이었고 본 주장이 "종목을 가로질러
  이식되는 진입 조건"이기 때문) 그대로.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# 적합:선택 비율. (0,1) 열린 구간 — 양쪽 다 최소 하나는 있어야 분할의 의미가
# 있다. 근거는 모듈 docstring.
DEFAULT_FIT_FRACTION = 2.0 / 3.0

# 층 내 종목 수가 이 미만이면 그 층은 나누지 않고 전부 적합으로 보낸다.
MIN_MEMBERS_TO_SPLIT = 2


@dataclass(frozen=True)
class SymbolSplit:
    fit: tuple[str, ...]
    select: tuple[str, ...]
    stratum_of: dict[str, str]
    fit_fraction_requested: float
    seed: int
    per_stratum_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.fit) & set(self.select)
        if overlap:
            raise ValueError(f"적합·선택 집합이 겹친다 (표본 외 분할이 아니다): {sorted(overlap)}")


def split_symbols(symbols: Sequence[str], stratum_of: Mapping[str, str], *,
                  seed: int, fit_fraction: float = DEFAULT_FIT_FRACTION) -> SymbolSplit:
    """`symbols` 를 층화 구조를 유지하며 적합/선택 두 서로소 집합으로 나눈다.

    각 층 안에서 독립으로 나눈다(전체를 한 번에 섞지 않는다) — 그래야 층별
    비율이 보존된다. 층을 순회하는 순서·층 안의 종목 순서를 정렬로 고정한
    뒤 시드 하나로 만든 `Generator` 를 그 순서대로 소비한다 — 그래서 같은
    `(symbols, stratum_of, seed, fit_fraction)` 이면 항상 같은 분할이 나오고,
    `seed` 가 다르면(다른 난수열) 분할도 달라진다.
    """
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError(f"fit_fraction 은 (0,1) 열린 구간이어야 한다: {fit_fraction}")

    by_stratum: dict[str, list[str]] = {}
    for symbol in symbols:
        stratum = stratum_of[symbol]
        by_stratum.setdefault(stratum, []).append(symbol)

    rng = np.random.default_rng(int(seed))
    fit: list[str] = []
    select: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for stratum in sorted(by_stratum):                # 결정론적 순회 순서
        members = sorted(by_stratum[stratum])          # 결정론적 입력 순서
        n = len(members)
        if n < MIN_MEMBERS_TO_SPLIT:
            fit.extend(members)
            counts[stratum] = {"fit": n, "select": 0, "n": n}
            continue
        order = rng.permutation(n)                      # seed 로 결정되는 유일한 난수 소비
        n_select = max(1, round(n * (1.0 - fit_fraction)))
        n_select = min(n_select, n - 1)                 # 적합 쪽에 최소 1개는 남긴다
        select_positions = set(order[:n_select].tolist())
        for i, symbol in enumerate(members):
            (select if i in select_positions else fit).append(symbol)
        counts[stratum] = {"fit": n - n_select, "select": n_select, "n": n}

    return SymbolSplit(fit=tuple(sorted(fit)), select=tuple(sorted(select)),
                       stratum_of={s: stratum_of[s] for s in symbols},
                       fit_fraction_requested=float(fit_fraction), seed=int(seed),
                       per_stratum_counts=counts)
