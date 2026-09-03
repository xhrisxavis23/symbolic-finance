"""Stored Feature Profile anchor → canonical execution-alignment artifact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import canonical as K
from ..config import TICK_ROOT, write_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..modules import execution_anchor
from ..modules.profile import ProfileSelection


ARTIFACT_NAME = "execution_anchor_replay_artifact.json"
CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache" / "execution_anchor_replay"


def run(selection: ProfileSelection, output: Path, *, root: Path = TICK_ROOT,
        workers: int | None = None) -> dict:
    """같은 stored anchor 입력의 현재 replay만 run-local 또는 shared cache에서 재사용한다."""
    output = Path(output)
    signature = execution_anchor.input_signature(selection)
    path = output / ARTIFACT_NAME
    if path.is_file():
        saved = read_artifact(path, kind="execution_anchor_replay")
        if (saved["payload"].get("input") == signature
                and _current_replay_payload(saved["payload"])):
            return saved
    cache_output = CACHE_ROOT / (str(signature["anchor_sha256"]) + "-"
                                 + str(K.CANONICAL["profile_sha256"]))
    cache_path = cache_output / ARTIFACT_NAME
    if cache_path.is_file():
        cached = read_artifact(cache_path, kind="execution_anchor_replay")
        if (cached["payload"].get("input") == signature
                and _current_replay_payload(cached["payload"])):
            write_artifact(path, cached)
            return cached
    composed = _compose_from_shared_anchor_outcomes(signature, execution_anchor.anchor_records(selection),
                                                    cache_output)
    if composed is not None:
        artifact = create_artifact("execution_anchor_replay", composed)
        write_artifact(cache_path, artifact)
        write_artifact(path, artifact)
        return artifact
    payload = execution_anchor.run(selection, cache_output, root=Path(root), workers=workers)
    artifact = create_artifact("execution_anchor_replay", payload)
    write_artifact(cache_path, artifact)
    write_artifact(path, artifact)
    return artifact


def _current_replay_payload(payload: dict) -> bool:
    """현 Evidence가 요구하는 raw-path label까지 보존한 replay인가."""
    if payload.get("schema") != execution_anchor.SCHEMA_VERSION:
        return False
    outcome_path = Path(str(payload.get("outcomes_path") or ""))
    if not outcome_path.is_file():
        return False
    try:
        pd.read_parquet(outcome_path, columns=["diagnostic_cohort"])
    except Exception:  # noqa: BLE001 - 동시 작성 중인 cache는 재사용하지 않는다.
        return False
    return True


def _compose_from_shared_anchor_outcomes(signature: dict, records: list[dict],
                                         cache_output: Path) -> dict | None:
    """더 넓은 Profile 선택이 기존 exact anchor replay를 다시 돌리지 않게 합친다."""
    if not records:
        return None
    keys = ["anchor_id", "cluster_id", "symbol", "date", "tick", "profile_cohort"]
    desired = pd.DataFrame(records).rename(columns={"cohort": "profile_cohort"})
    desired["symbol"] = desired["symbol"].astype(str).str.zfill(6)
    desired["date"] = desired["date"].astype(str)
    desired["tick"] = desired["tick"].astype(int)
    available: list[pd.DataFrame] = []
    for candidate in sorted(CACHE_ROOT.glob(f"*/{ARTIFACT_NAME}")):
        if candidate.parent == cache_output:
            continue
        cached = read_artifact(candidate, kind="execution_anchor_replay")
        payload = cached["payload"]
        if payload.get("canonical_backtest_profile_hash") != K.CANONICAL["profile_sha256"]:
            continue
        if not _current_replay_payload(payload):
            continue
        outcome_path = Path(str(payload.get("outcomes_path") or ""))
        if not outcome_path.is_file():
            continue
        try:
            frame = pd.read_parquet(outcome_path)
        except Exception:  # noqa: BLE001 - 동시 작성 중인 cache는 재사용하지 않는다.
            continue
        if set(keys) <= set(frame):
            available.append(frame)
    if not available:
        return None
    known = pd.concat(available, ignore_index=True).drop_duplicates(keys, keep="last")
    merged = desired.merge(known, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(desired) or merged["execution_label"].isna().any():
        return None
    cache_output.mkdir(parents=True, exist_ok=True)
    outcomes_path = cache_output / "anchor_execution_outcomes.parquet"
    merged.to_parquet(outcomes_path, index=False)
    summary = execution_anchor._summary(merged, [])
    write_json(cache_output / "execution_anchor_summary.json", summary)
    return {
        "schema": execution_anchor.SCHEMA_VERSION,
        "input": signature,
        "canonical_backtest_profile_id": K.CANONICAL["profile_id"],
        "canonical_backtest_profile_hash": K.CANONICAL["profile_sha256"],
        "execution_definition": {
            "entry": "stored anchor tick에서 canonical BID1 queue entry",
            "exit": "unchanged canonical fixed exit",
            "independence": "각 anchor를 독립 주문으로 재생; anchor 사이 position blocking을 공유하지 않음",
            "use": "Discovery execution-alignment diagnostic only",
            "not_a_strategy_pnl": True,
        },
        "summary": summary,
        "outcomes_path": str(outcomes_path),
        "ledger_manifests": [],
        "cache_reuse": {"mode": "COMPOSED_FROM_EXACT_ANCHOR_OUTCOMES",
                        "source_cache_count": len(available)},
    }
