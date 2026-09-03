from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework import contract as C, evidence, execution_profile, featureprofile, sample_condition as SC
from framework.config import sha256_json
from framework.tests.test_catalog import arrays


def _anchor(anchor_id: str, cohort: str, semantic: str, tick: int) -> dict:
    return {
        "anchor_id": anchor_id, "cohort": cohort, "execution_cohort": semantic,
        "cluster_id": "EA01", "symbol": "000001", "date": "20260316", "tick": tick,
        "best_ask_net_bps": float("nan"), "entry_queue_status": "filled",
        "entry_terminal_tick": tick + 1, "entry_fill_tick": tick + 1,
        "entry_wait_ticks": 1, "entry_wait_seconds": .1, "entry_queue_fraction": 1.0,
        "entry_oracle_ask_tick": -1, "entry_fill_before_oracle_ask": False,
        "entry_assumption": "canonical_bid1_queue", "exit_profile_id": "CANONICAL_QUEUE_V9",
        "canonical_exit_net_bps": 25.0 if semantic == "STRONG_PROFIT" else -40.0,
        "canonical_exit_tick": tick + 5, "canonical_exit_reason": "trailing_ask1",
        "holding_seconds": 1.0, "outcome_observable": True, "anchor_source": "m0_price_recovery",
    }


def test_execution_profile_anchor_sampling_preserves_semantic_cohorts():
    labels = pd.DataFrame([
        _anchor("p", "PROFIT", "STRONG_PROFIT", 10),
        _anchor("l", "LOSS", "LOSS_TAIL", 20),
        _anchor("b", "BACKGROUND", "BACKGROUND", 30),
        {**_anchor("u", "UNFILLED", "UNFILLED", 40), "outcome_observable": False},
    ])
    result = execution_profile.choose_anchors(labels, config=execution_profile.ExecutionProfileConfig())
    assert set(result.cohort) == {"PROFIT", "LOSS", "BACKGROUND", "UNFILLED"}
    assert set(result.execution_cohort) == {
        "STRONG_PROFIT", "LOSS_TAIL", "BACKGROUND", "UNFILLED"}


def test_execution_profile_contract_carries_the_executable_m0_condition():
    contract = execution_profile.label_contract(execution_profile.ExecutionProfileConfig())
    assert contract["executable_sample_condition"] == SC.m0_price_recovery()
    assert contract["executable_sample_condition_sha256"] == SC.condition_hash(
        contract["executable_sample_condition"])
    assert contract["executable_sample_condition_warmup_ticks"] == 20


def test_m0_includes_zero_to_positive_transition():
    primitive = {"op": "primitive", "primitive_id": "mid_return_5t_bps"}
    runtime = C.ExpressionRuntime(
        arrays(n=2),
        precomputed_expressions={sha256_json(primitive): np.array([0.0, 1.0])},
    )

    assert runtime.evaluate(SC.m0_price_recovery()).tolist() == [False, True]


def test_legacy_strict_m0_profile_requires_regeneration():
    stored = execution_profile.label_contract(execution_profile.ExecutionProfileConfig())
    stored["executable_sample_condition_schema"] = "executable_sample_condition.v1"
    stored["executable_sample_condition"]["equality"] = "strict"
    stored["executable_sample_condition_sha256"] = SC.condition_hash(
        stored["executable_sample_condition"])

    with pytest.raises(ValueError, match="Feature Profile을 다시 만들어야 한다"):
        SC.require_for_event_source(stored, "stored Profile")


def test_execution_profile_store_roundtrip_preserves_execution_provenance(tmp_path: Path):
    anchors = pd.DataFrame([
        _anchor("p", "PROFIT", "STRONG_PROFIT", 10),
        _anchor("l", "LOSS", "LOSS_TAIL", 20),
        _anchor("b", "BACKGROUND", "BACKGROUND", 30),
    ])
    long = pd.DataFrame([
        {"anchor_id": row.anchor_id, "cohort": row.cohort, "cluster_id": row.cluster_id,
         "symbol": row.symbol, "date": row.date, "tick": row.tick,
         "relative_time_ms": 0, "age_ms": 0.0, "book_imbalance": 0.1}
        for row in anchors.itertuples()
    ])
    featureprofile.write(long, anchors, config=evidence.ProfilerConfig(), root=tmp_path,
                         schema="execution_aligned_feature_profile_symbol_day.v1",
                         metadata={"profile_type": execution_profile.PROFILE_TYPE})
    restored_long, restored_anchors = featureprofile.read(clusters=["EA01"], root=tmp_path)
    assert set(restored_anchors.execution_cohort) == {"STRONG_PROFIT", "LOSS_TAIL", "BACKGROUND"}
    assert "execution_cohort" not in restored_long
    assert len(restored_long) == len(long)


def test_execution_profile_skips_only_symbol_days_without_queue_trade_events(tmp_path: Path, monkeypatch):
    labels = pd.DataFrame([
        _anchor("p", "PROFIT", "STRONG_PROFIT", 10),
        _anchor("l", "LOSS", "LOSS_TAIL", 20),
        _anchor("b", "BACKGROUND", "BACKGROUND", 30),
    ])
    labels["cluster_id"] = "ALL_STOCKS_ASSET_TYPE_ST"

    def fake_labels(symbol, *_args, **_kwargs):
        if symbol == "000002":
            raise ValueError("지정가 매수 큐 재생에는 체결 이벤트가 필요하다. 없는 키: ['sell_volume']")
        return labels

    def fake_profile(anchors, *_args, **_kwargs):
        return pd.DataFrame([
            {"anchor_id": row.anchor_id, "cohort": row.cohort,
             "cluster_id": row.cluster_id, "symbol": row.symbol, "date": row.date,
             "tick": row.tick, "relative_time_ms": 0, "age_ms": 0.0,
             "book_imbalance": 0.1}
            for row in anchors.itertuples()
        ])

    monkeypatch.setattr(execution_profile, "label_events", fake_labels)
    monkeypatch.setattr(execution_profile.evidence, "profile_anchors", fake_profile)
    result = execution_profile.materialize(
        "ALL_STOCKS_ASSET_TYPE_ST", ("000001", "000002"), ("20260316",), tmp_path)

    assert result["written"] == 1
    assert result["skipped"][0]["symbol"] == "000002"
    assert result["skipped"][0]["reason"] == "MISSING_QUEUE_TRADE_EVENTS"
    assert (featureprofile.folder("ALL_STOCKS_ASSET_TYPE_ST", "000001", "20260316", tmp_path)
            / "profile.parquet").is_file()
