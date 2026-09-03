"""정본 재생. 계획서 §3 S5 ⑤.

실행 설정을 인자로 받지 않는다 — 정본 프로필이 정한다. 여기서 하는 일은
계약을 모아 넘기고, 돌아온 원장이 성립하는지 확인하는 것뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config
from .compile import EntryExpression, contract_id, contract_template, parameter_table
from .compile.pipeline import PARAMETER

_framework = config.load_framework()
from framework import canonical as _canonical  # noqa: E402


@dataclass(frozen=True)
class ReplayResult:
    ledger_path: Path
    manifest: dict
    contract_ids: tuple[str, ...]
    attempts: int


def attempt_count(entries: Sequence[EntryExpression], grid: Sequence[float]) -> int:
    """계획서 §4 가 보고를 요구하는 시도 횟수 N."""
    return len(entries) * len(grid)


def run(entries: Sequence[EntryExpression], symbols: Sequence[str], date: str,
        output_dir: Path, grid: Sequence[float] = config.QUANTILE_GRID,
        workers: int = config.REPLAY_WORKERS) -> ReplayResult:
    if not entries:
        raise ValueError("재생할 진입식이 없다")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contracts: dict[str, dict] = {}
    members: dict[str, tuple[str, ...]] = {}
    table: dict[str, dict[str, float]] = {}

    for index, entry in enumerate(entries):
        prefix = f"e{index:03d}"
        identifiers = tuple(contract_id(prefix, q) for q in grid)
        template = contract_template(entry.ast, PARAMETER)
        for identifier in identifiers:
            if identifier in contracts:
                # F6 리뷰: `contract_id` 는 분위를 소수 2자리로 포맷한다. 임의
                # 격자에서 두 분위가 같은 문자열이 되면 이 대입이 계약 하나를
                # 조용히 덮어쓰는데, `attempt_count` 는 `len(entries)*len(grid)`
                # 를 그대로 돌려주므로 보고되는 시도 횟수 N 이 실제보다 커진다
                # — N 은 계획서가 "미보고 시 성과 주장 무효"로 못 박은 값이다.
                raise ValueError(
                    f"계약 ID 충돌: '{identifier}' 가 이미 있다. 분위 격자 {grid} "
                    "에서 서로 다른 분위가 `contract_id` (소수 2자리 포맷) 로는 "
                    "같은 문자열이 됐다 — 격자를 조정하거나 포맷 정밀도를 높여야 "
                    "한다. 조용히 덮어쓰면 계약이 사라지고 attempt_count 가 실제 "
                    "재생 시도보다 크게 보고된다.")
            contracts[identifier] = template
            members[identifier] = tuple(symbols)
        table.update(parameter_table(identifiers, symbols, date, PARAMETER, grid))

    ledger_path = output_dir / "ledger.parquet"
    manifest = _canonical.run_backtest(
        contracts=contracts, members=members, dates=[str(date)],
        output=ledger_path, parameter_table=table,
        workers=int(workers), root=config.TICK_ROOT)

    if manifest.get("errors"):
        raise RuntimeError(
            "재생에 오류가 있다. 원장 일부만 집계하면 조용히 틀린 수가 나온다:\n  "
            + "\n  ".join(str(e) for e in manifest["errors"]))

    return ReplayResult(ledger_path=ledger_path, manifest=manifest,
                        contract_ids=tuple(sorted(contracts)),
                        attempts=attempt_count(entries, grid))
