"""Grounding을 Parameter Search용 Executable Specification으로 옮기는 어댑터."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .. import executable as X


def run(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
        package: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """명세는 Grounding의 예측만 읽는다. Validation은 parameter lock 뒤 Backtest다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = X.run_from_grounding(hypothesis, grounded, package, output)
    return {"state": result["readiness"]["readiness"], "spec": result["spec"],
            "audit": result["audit"], "readiness": result["readiness"]}
