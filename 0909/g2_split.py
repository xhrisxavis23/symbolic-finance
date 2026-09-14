"""G2 — 참조/증류-적합/증류-선택 세 갈래 분할. `PREREG-G2.md` §1 의 구현.

`sd.universe`(층화) · `sd.e0.split.split_symbols`(적합/선택 분할)를
읽기 전용으로 재사용한다. 이 파일이 새로 하는 일은 딱 하나 — 층 안에서
시드로 섞은 뒤 앞부분을 참조(R) 로 떼어내고, 나머지만 `split_symbols` 에
넘기는 것.

**코드순이 아니라 시드 셔플을 쓰는 이유**: `sd.universe.slice_symbols` 는
"층마다 코드 오름차순 앞에서 n개"(재현성 근거)를 쓴다. 참조/증류를 가르는
분할에 그 순서를 그대로 쓰면 "종목 코드 순서"라는 잠재 교란(상장 시점·
거래소 구분과 상관될 수 있다)이 참조 집합과 증류 집합을 체계적으로 갈라
놓을 위험이 있다. 그래서 여기서는 층 안에서 결정론적 시드로 섞은 뒤
앞부분을 R 에 배정한다 — 재현성은 유지하면서 순서 편향을 없앤다
(`PREREG-G2.md` §1.1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd import universe                      # noqa: E402
from sd.e0 import split as e0_split           # noqa: E402

# 참조/증류 분할 전용 시드 — `sd.config.SEED`(=0, E0/L0 의 적합·선택 분할과
# on-manifold 표집이 쓰는 시드)와 분리한다. 같은 시드를 재사용하면 "참조를
# 가르는 난수"와 "적합/선택을 가르는 난수"가 같은 스트림을 공유하게 되어
# 두 분할 사이에 숨은 상관을 만들 위험이 있다.
G2_SPLIT_SEED = 20260909


@dataclass(frozen=True)
class ThreeWaySplit:
    reference: tuple[str, ...]
    distill_fit: tuple[str, ...]
    distill_select: tuple[str, ...]
    stratum_of: dict[str, str]
    per_stratum_counts: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        r, f, s = set(self.reference), set(self.distill_fit), set(self.distill_select)
        overlaps = {"reference&fit": r & f, "reference&select": r & s, "fit&select": f & s}
        bad = {k: sorted(v) for k, v in overlaps.items() if v}
        if bad:
            raise ValueError(f"세 집합이 겹친다(서로소 위반): {bad}")


def _stratified_shuffle_split(strata_of_symbol: dict, symbols_by_stratum: dict,
                              n_ref_per_stratum: int, seed: int) -> tuple[list, list, dict]:
    """층마다 시드로 섞고 앞 n_ref_per_stratum 개를 참조로, 나머지를 증류 풀로."""
    rng = np.random.default_rng(int(seed))
    reference: list[str] = []
    remaining: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for stratum in sorted(symbols_by_stratum):          # 결정론적 순회 순서
        members = sorted(symbols_by_stratum[stratum])    # 결정론적 입력 순서
        order = rng.permutation(len(members))             # seed 로 결정되는 유일한 난수 소비
        shuffled = [members[i] for i in order]
        n_ref = min(int(n_ref_per_stratum), len(shuffled))
        reference.extend(shuffled[:n_ref])
        remaining.extend(shuffled[n_ref:])
        counts[stratum] = {"n_members": len(members), "n_reference": n_ref,
                           "n_remaining_for_distill": len(shuffled) - n_ref}
    return reference, remaining, counts


def build_three_way_split(all_symbols: tuple, date: str, *,
                          n_ref_per_stratum: int, n_distill_per_stratum: int,
                          g2_split_seed: int = G2_SPLIT_SEED,
                          distill_split_seed: int = 0,
                          fit_fraction: float = e0_split.DEFAULT_FIT_FRACTION,
                          workers: int = 32) -> ThreeWaySplit:
    """참조(R) / 증류-적합(D_fit) / 증류-선택(D_select) 를 만든다.

    각 층에서 최대 `n_ref_per_stratum + n_distill_per_stratum` 개를 쓴다
    (있는 만큼만 — 층이 그보다 작으면 경고 없이 조용히 줄지 않는다, 호출부가
    `per_stratum_counts` 를 보고 확인해야 한다).
    """
    stats_df = universe.liquidity_stats(all_symbols, date, workers=workers)
    strata = universe.assign_strata(stats_df)
    stratum_of_all = strata["stratum"].to_dict()

    # 층마다 최대 (n_ref+n_distill) 개까지만 후보로 삼는다 — `slice_symbols`
    # 를 재사용해 그 상한을 구한다(코드순으로 뽑지만, 이 뒤에 우리가 다시
    # 시드로 섞으므로 최종 R/D 배정은 코드순과 무관해진다).
    cap = int(n_ref_per_stratum) + int(n_distill_per_stratum)
    candidate_symbols = universe.slice_symbols(strata, per_stratum=cap)
    symbols_by_stratum: dict[str, list[str]] = {}
    for symbol in candidate_symbols:
        symbols_by_stratum.setdefault(stratum_of_all[symbol], []).append(symbol)

    reference, remaining, counts = _stratified_shuffle_split(
        stratum_of_all, symbols_by_stratum, n_ref_per_stratum, g2_split_seed)

    for stratum, c in counts.items():
        if c["n_members"] < cap:
            print(f"  [경고] 층 {stratum}: 요청 {cap}개 중 {c['n_members']}개만 있음 "
                  f"(참조 {c['n_reference']} / 증류풀 {c['n_remaining_for_distill']})")

    stratum_of_remaining = {s: stratum_of_all[s] for s in remaining}
    symsplit = e0_split.split_symbols(remaining, stratum_of_remaining,
                                      seed=distill_split_seed, fit_fraction=fit_fraction)

    return ThreeWaySplit(
        reference=tuple(sorted(reference)),
        distill_fit=symsplit.fit, distill_select=symsplit.select,
        stratum_of={s: stratum_of_all[s] for s in candidate_symbols},
        per_stratum_counts=counts)
