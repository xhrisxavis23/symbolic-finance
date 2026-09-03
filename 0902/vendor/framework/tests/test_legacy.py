"""과거 package adapter의 식과 청산 결합만 확인한다."""

from __future__ import annotations

import numpy as np
import pytest

from framework import legacy


def arrays():
    n = 5
    bid = np.array([100.0, 100.0, 101.0, 101.0, 101.0])
    ask = bid + 1.0
    return {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.tile(bid[:, None], (1, 10)),
        "ask_price": np.tile(ask[:, None], (1, 10)),
        "bid_qty": np.tile(np.arange(1.0, 11.0), (n, 1)),
        "ask_qty": np.tile(np.arange(11.0, 21.0), (n, 1)),
        "buy_volume": np.array([0.0, 20.0, 0.0, 0.0, 0.0]),
        "sell_volume": np.array([0.0, 0.0, 10.0, 0.0, 0.0]),
    }


def test_legacy_feature_formulas_keep_their_named_meaning():
    data = arrays()
    assert legacy.feature_values(data, "ask_depth_total_10")[0] == 155.0
    assert legacy.feature_values(data, "obi_10")[0] == pytest.approx(-100.0 / 210.0)


def test_legacy_exit_is_a_first_trigger_not_a_replacement():
    data = arrays()
    program = {"exit": {"conditions": [{"source_feature": "exit:obi_10", "feature": "obi_10",
                                          "comparator": "<", "threshold": 0.0}]}}
    # bid depth is always below ask depth, so the state rule fires at entry.
    assert legacy.first_exit_trigger(data, program, 0, 4) == 0


def test_legacy_quantiles_bind_to_the_same_symbols_prior_valid_day(monkeypatch):
    prior = arrays()
    n = 300
    prior["time_s"] = np.arange(n, dtype=float)
    prior["bid_qty"] = np.tile(np.arange(1.0, 11.0), (n, 1))
    prior["ask_qty"] = np.tile(np.arange(11.0, 21.0), (n, 1))
    prior["bid_qty"][:, 0] = np.arange(n, dtype=float)
    monkeypatch.setattr(legacy.tickdata, "prior_valid_day",
                        lambda symbol, date, among, root: "20260420")
    monkeypatch.setattr(legacy.tickdata, "load", lambda symbol, date, root: (prior, None))
    program = {"entry": {"conditions": [{"source_feature": "obi_10", "feature": "obi_10",
                                             "comparator": ">", "quantile": 0.5}]},
               "exit": {"conditions": []}}

    bound = legacy.bind_prior_day_thresholds(
        program, symbol="A", date="20260421", reference_dates=["20260420"])

    assert bound is not None
    resolved, detail = bound
    assert detail["reference_date"] == "20260420"
    assert resolved["entry"]["conditions"][0]["threshold"] == pytest.approx(
        np.quantile(legacy.feature_values(prior, "obi_10"), 0.5))
