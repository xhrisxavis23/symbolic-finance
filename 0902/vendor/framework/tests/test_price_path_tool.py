"""발굴 가격 경로 tool의 읽기 권한과 출력 계약."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framework.agents import hypothesis as hypothesis_agent, price_path, runtime


def access():
    return {
        "schema": price_path.ACCESS_SCHEMA,
        "horizon_key": "S30",
        "profit_root": "profit-root", "tick_root": "tick-root",
        "anchors": [
            {"anchor_id": "P1", "cohort": "PROFIT", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 2,
             "entry_queue_status": "filled", "entry_fill_tick": 3,
             "entry_wait_ticks": 1, "entry_wait_seconds": 0.1,
             "entry_queue_fraction": 0.2, "entry_fill_before_oracle_ask": True},
            {"anchor_id": "L1", "cohort": "LOSS", "cluster_id": "MK01",
             "symbol": "000001", "date": "20260316", "tick": 4},
        ],
    }


def test_list_price_paths_only_exposes_profile_anchors():
    result = price_path.list_price_paths(access(), cohort="PROFIT", limit=3)
    assert result["discovery_only"] is True
    assert result["paths"] == [{
        "path_id": "P1:S30", "cohort": "PROFIT", "cluster_id": "MK01",
        "symbol": "000001", "date": "20260316", "anchor_tick": 2,
        "entry_queue_status": "filled",
    }]


def test_price_path_reads_bid_ask_and_keeps_oracle_separate(monkeypatch):
    labels = pd.DataFrame({
        "tick_idx": [2], "complete__S30": [True], "window_end_delta_ticks__S30": [4],
        "best_ask_delta_ticks__S30": [2], "best_ask_gross_bps__S30": [300.0],
        "fixed_ask_gross_bps__S30": [20.0],
    })
    arrays = {
        "bid_price": np.array([[99.0], [100.0], [101.0], [102.0], [103.0], [104.0], [105.0]]),
        "ask_price": np.array([[100.0], [101.0], [102.0], [103.0], [105.0], [105.0], [106.0]]),
        "time_s": np.arange(7, dtype=float) * 0.1,
    }
    monkeypatch.setattr(price_path.profit, "path_labels", lambda *args, **kwargs: labels)
    monkeypatch.setattr(price_path.tickdata, "load", lambda *args, **kwargs: (arrays, None))

    result = price_path.get_price_path(access(), cluster="MK01", symbol="1", date="20260316",
                                       anchor_tick=2, max_points=8)

    assert result["path_id"] == "P1:S30"
    assert result["execution_proof"] is False
    assert result["anchor"] == {"tick": 2, "bid_1": 101.0, "ask_1": 102.0}
    assert result["price_opportunity"]["best_ask_tick"] == 4
    assert result["price_opportunity"]["best_ask_1"] == 105.0
    assert result["entry_queue"]["fill_before_oracle_ask"] is True
    assert [point["tick"] for point in result["points"]] == list(range(7))


def test_price_path_rejects_an_anchor_the_profile_did_not_open():
    with pytest.raises(ValueError, match="Feature Profile 발굴 anchor"):
        price_path.get_price_path(access(), cluster="MK01", symbol="000001", date="20260316",
                                  anchor_tick=3)


def test_microstructure_probe_returns_only_pre_anchor_raw_data(monkeypatch):
    arrays = {
        "bid_price": np.array([[99.0], [100.0], [101.0], [102.0], [103.0]]),
        "ask_price": np.array([[100.0], [101.0], [102.0], [103.0], [104.0]]),
        "bid_qty": np.array([[10.0] * 10, [20.0] * 10, [30.0] * 10,
                             [40.0] * 10, [50.0] * 10]),
        "ask_qty": np.array([[11.0] * 10, [21.0] * 10, [31.0] * 10,
                             [41.0] * 10, [51.0] * 10]),
        "buy_volume": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "sell_volume": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        "time_s": np.arange(5, dtype=float) * 0.1,
    }
    monkeypatch.setattr(price_path.tickdata, "load", lambda *args, **kwargs: (arrays, None))

    result = price_path.get_pre_anchor_microstructure(
        access(), cluster="MK01", symbol="000001", date="20260316", anchor_tick=2,
        max_points=8)

    assert result["path_id"] == "P1:S30"
    assert result["observation_boundary"].endswith("post-anchor data 없음")
    assert [point["tick"] for point in result["points"]] == [0, 1, 2]
    assert all(point["elapsed_ms"] <= 0.0 for point in result["points"])
    assert result["points"][-1]["bid_depth_10"] == 300.0
    assert result["points"][-1]["sell_volume"] == 3.0


def test_hypothesis_tool_contract_requires_profit_and_loss_microstructure_reads():
    def call(tool, tick):
        return {"server": "price_path", "tool": tool, "arguments": {
            "cluster": "MK01", "symbol": "000001", "date": "20260316", "anchor_tick": tick}}

    invocation = runtime.AgentInvocation(
        role="hypothesis_generation", model="test", effort="low", prompt_sha256="p",
        response_sha256="r", tool_calls=(
            {"server": "price_path", "tool": "list_price_paths", "arguments": {}},
            call("get_price_path", 2), call("get_price_path", 4)))
    missing = hypothesis_agent._price_path_tool_usage(invocation, access())
    assert missing["problems"] == [
        "LOSS get_pre_anchor_microstructure MCP 호출이 없다",
        "PROFIT get_pre_anchor_microstructure MCP 호출이 없다"]

    inspected = runtime.AgentInvocation(
        role="hypothesis_generation", model="test", effort="low", prompt_sha256="p",
        response_sha256="r", tool_calls=(
            {"server": "price_path", "tool": "list_price_paths", "arguments": {}},
            call("get_price_path", 2), call("get_price_path", 4),
            call("get_pre_anchor_microstructure", 2),
            call("get_pre_anchor_microstructure", 4)))
    assert hypothesis_agent._price_path_tool_usage(inspected, access())["problems"] == []


def test_runtime_attaches_only_a_per_call_mcp_server(monkeypatch):
    calls = []

    def fake_run_agent(prompt, *, model, effort, extra_args=()):
        calls.append({"prompt": prompt, "model": model, "effort": effort,
                      "extra_args": extra_args})
        return {"status": "ok"}

    monkeypatch.setattr(runtime, "run_agent", fake_run_agent)
    result = runtime.CodexAgentRunner().run(
        role="hypothesis_generation", prompt="prompt", model="test-model", effort="low",
        tool_context=access())

    assert result == {"status": "ok"}
    assert calls[0]["extra_args"][:2] == ("-c", calls[0]["extra_args"][1])
    assert "mcp_servers.discovery_price_path.command" in calls[0]["extra_args"][1]
    assert "mcp_servers.discovery_price_path.args" in calls[0]["extra_args"][3]
    assert calls[0]["extra_args"][-1].endswith('default_tools_approval_mode="writes"')
