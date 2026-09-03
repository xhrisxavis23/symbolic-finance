"""FeatureProfile → Evidence artifact Stage."""

from __future__ import annotations

from pathlib import Path

from ..config import TICK_ROOT
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import evidence, execution_anchor
from ..modules.profile import ProfileSelection


ARTIFACT_NAME = "evidence_artifact.json"


def run(selection: ProfileSelection, output: Path, *,
        execution_anchor_replay_path: Path | None = None) -> dict:
    output = Path(output)
    replay = None
    parents = {}
    if execution_anchor_replay_path is not None:
        replay = read_artifact(execution_anchor_replay_path, kind="execution_anchor_replay")
        if replay["payload"].get("input") != execution_anchor.input_signature(selection):
            raise ValueError("Execution anchor replay와 Evidence의 Feature Profile 입력이 다르다")
        parents["execution_anchor_replay"] = replay["artifact_id"]
    result = evidence.build(selection, output,
                            execution_anchor_replay=(replay or {}).get("payload"))
    artifact = create_artifact("evidence_package", result, parents=parents)
    write_artifact(output / ARTIFACT_NAME, artifact)
    return artifact
