"""임계 부착과 분위 격자 전개. 계획서 §3 S5 ④ · DESIGN.md D8·D9.

임계는 절대값이 아니라 **분위수**다. 스프레드 중앙값이 13bp 인 저마찰층과
43bp 인 고마찰층에 같은 절대 임계를 걸면 전혀 다른 사건을 고르게 된다.
"""

from __future__ import annotations

from typing import Sequence

THETA_PREFIX = "UNRESOLVED:theta_"
THRESHOLD_KIND = "rolling_prior_100_ticks_quantile"


def attach(numeric_ast: dict, parameter: str) -> dict:
    """numeric AST 를 boolean 진입식으로. 임계는 미결로 남긴다."""
    return {"op": "compare", "input": numeric_ast, "comparator": ">",
            "value": f"{THETA_PREFIX}{parameter}"}


def contract_template(entry_ast: dict, parameter: str) -> dict:
    """`canonical.run_backtest` 가 받는 계약. 실행 결속을 선언하지 않는다."""
    return {
        "entry_program": {"signal": entry_ast, "warmup_ticks": 0},
        "parameter_interface": {
            f"theta_{parameter}": {"threshold_source": {"kind": THRESHOLD_KIND}}},
    }


def contract_id(prefix: str, quantile: float) -> str:
    return f"{prefix}:q{float(quantile):.2f}"


def parameter_table(contract_ids: Sequence[str], symbols: Sequence[str], date: str,
                    parameter: str, grid: Sequence[float]) -> dict[str, dict[str, float]]:
    """`"{contract_id}:{symbol}:{date}" -> {theta: quantile}`.

    계약 ID 로 분위를 나누면 한 번의 재생 호출로 격자 전체를 돈다.
    """
    if len(contract_ids) != len(grid):
        raise ValueError("계약 ID 와 분위 격자의 길이가 다르다")
    table: dict[str, dict[str, float]] = {}
    for identifier, quantile in zip(contract_ids, grid):
        for symbol in symbols:
            table[f"{identifier}:{symbol}:{date}"] = {f"theta_{parameter}": float(quantile)}
    return table
