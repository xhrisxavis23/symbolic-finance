from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd

from framework.modules import market_regime
from framework.modules.schedule import ResearchSchedule
from framework.workflows import experiments


def test_market_regime_uses_cross_sectional_median_not_label_majority(tmp_path):
    source = tmp_path / "symbol_day_labels.csv"
    pd.DataFrame([
        {"date": "20260316", "status": "LABELED", "return_bps": 100.0},
        {"date": "20260316", "status": "LABELED", "return_bps": -10.0},
        {"date": "20260316", "status": "LABELED", "return_bps": -10.0},
    ]).to_csv(source, index=False)
    labels = market_regime.build(source, tmp_path / "out")
    assert labels.loc[0, "regime"] == market_regime.SIDEWAYS
    assert market_regime.choose(labels, ("20260316",), market_regime.SIDEWAYS, 1) == (
        "20260316",)


def test_experiment_runs_ready_units_in_parallel(monkeypatch, tmp_path):
    schedule = ResearchSchedule(
        validation_dates=("20260401",), search_fit_dates=("20260320",),
        search_confirm_dates=(), backtest_dates=("20260413",), terminal_oos_dates=(),
        single_day_discovery_search=True).as_input()
    units = [{"regime": "UP", "cluster": cluster, "symbol_count": 1, "state": "READY",
              "schedule": schedule, "discovery_date": "20260320",
              "eligible_core_symbols": [symbol]}
             for cluster, symbol in (("MK01", "001"), ("MK02", "002"))]
    monkeypatch.setattr(experiments, "plan", lambda request, output: {"units": units})
    lock = threading.Lock()
    active = maximum = 0

    def fake_run(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"state": "COMPLETE", "artifacts": {}}

    monkeypatch.setattr(experiments, "run_research", fake_run)
    request = experiments.MarketExperimentRequest(
        clusters=("MK01", "MK02"), profile_store=tmp_path, symbol_day_labels=tmp_path,
        discovery_dates=(), validation_dates=(), test_dates=(), max_workers=2)
    result = experiments.run(request, tmp_path / "output")
    assert maximum == 2
    assert [record["state"] for record in result["records"]] == ["COMPLETE", "COMPLETE"]


def test_two_day_schedule_allows_each_market_regime():
    labels = pd.DataFrame([
        {"date": "20260316", "regime": market_regime.DOWN},
        {"date": "20260318", "regime": market_regime.SIDEWAYS},
        {"date": "20260330", "regime": market_regime.SIDEWAYS},
        {"date": "20260331", "regime": market_regime.DOWN},
        {"date": "20260402", "regime": market_regime.DOWN},
        {"date": "20260403", "regime": market_regime.SIDEWAYS},
        {"date": "20260414", "regime": market_regime.SIDEWAYS},
        {"date": "20260415", "regime": market_regime.SIDEWAYS},
        {"date": "20260421", "regime": market_regime.DOWN},
        {"date": "20260423", "regime": market_regime.DOWN},
    ])
    request = experiments.MarketExperimentRequest(
        clusters=(), profile_store=Path("."), symbol_day_labels=Path("."),
        discovery_dates=("20260316", "20260318"),
        validation_dates=("20260330", "20260331", "20260402", "20260403"),
        test_dates=("20260414", "20260415", "20260421", "20260423"),
        validation_day_count=2, test_day_count=2)
    down, down_error = experiments._schedule(labels, request, market_regime.DOWN)
    sideways, sideways_error = experiments._schedule(labels, request, market_regime.SIDEWAYS)
    assert down_error is None and down is not None
    assert sideways_error is None and sideways is not None
    assert down.backtest_dates == ("20260421", "20260423")
    assert sideways.validation_dates == ("20260330", "20260403")
