"""Final Backtest 뒤 Sandbox 전체 registry를 보고용으로 갱신한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..contracts.artifacts import create_artifact, write_artifact
from ..modules import algorithm_registry


ARTIFACT_NAME = "registry_refresh_artifact.json"


def run(research_run: Mapping[str, Any], *, backtest_set_id: str, output: Path,
        sandbox_root: Path | None = None) -> dict[str, Any]:
    """Final 결과를 읽기 전용으로 집계한다. 어떤 다음 진입도 선택하지 않는다."""
    registry = algorithm_registry.refresh_current(sandbox_root)
    payload = registry["payload"]
    artifact = create_artifact("final_registry_refresh", {
        "state": "FINAL_REGISTRY_REFRESHED",
        "registry_artifact_id": registry["artifact_id"],
        "positive_algorithm_count": payload["positive_algorithm_count"],
        "remaining_to_target": payload["remaining_to_target"],
        "final_observation_count": payload["final_observation_count"],
        "reporting_only": True,
        "boundary": (
            "이 refresh는 완료된 Final Backtest의 집계·중복 제거용이다. Final 결과를 "
            "Discovery, Entry Refinement, Parameter Search 또는 Validation의 입력으로 쓰지 않는다."),
    }, parents={
        "research_run": str(research_run["artifact_id"]),
        "backtest_set": str(backtest_set_id),
        "final_algorithm_registry": str(registry["artifact_id"]),
    })
    return write_artifact(Path(output) / ARTIFACT_NAME, artifact)
