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
