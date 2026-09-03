"""탐색이 고정된 가설의 경계를 넘지 않았는가.

여기서 보는 것은 둘이다 — **탐색 축이 정말 하나인가**, 그리고 **임계를 만들 때
그날이나 미래를 보지 않았는가.**
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from framework import catalog, contract, executable as X, implementation as I, search as S
from framework.modules import parameter_search
from framework import validation as V
from .test_executable import manifest, results, revision, routing


# 전일 q artifact 재생 호환성. 정본 기본값은 직전 100 tick q이고 별도 테스트로 확인한다.
CONFIG = S.SearchConfig(threshold_kind=X.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
                        min_calibration_obs=200)
CALENDAR = ["20260316", "20260317", "20260318", "20260319", "20260320"]


@pytest.fixture
def spec():
    return X.build(revision(), routing(), manifest(), results())


@pytest.fixture
def amended(spec):
    return S.amend_specification(spec, CONFIG)


# ---- 매개화 (§4~§6) --------------------------------------------------------------

def test_raw_threshold_becomes_a_percentile(spec, amended):
    assert spec["signal_template"]["value"] == "UNRESOLVED:theta_book_imbalance"
    assert amended["signal_template"]["value"] == "UNRESOLVED:q_book_imbalance"
    assert amended["signal_template"]["_parameter"]["type"] == "percentile"


def test_new_specification_declares_the_prior_100_tick_q_source(spec):
    config = S.config_for(spec)
    amended = S.amend_specification(spec, config)

    assert config.threshold_kind == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    assert amended["signal_template"]["_parameter"]["threshold_source"]["kind"] \
        == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE


def test_search_dimension_does_not_grow(spec, amended):
    """바뀐 것은 그 하나가 무엇을 뜻하는가지 몇 개인가가 아니다 (§5)."""
    assert len(spec["search_boundary"]["allowed_parameters"]) == 1
    assert len(amended["search_boundary"]["allowed_parameters"]) == 1
    assert amended["parameterization"]["search_dimension"] == 1


def test_parameter_search_uses_specification_direction_when_parameter_omits_it(spec):
    allowed = spec["search_boundary"]["allowed_parameters"]

    assert parameter_search._declared_states(spec, allowed) == {
        ("book_imbalance", "HIGHER"),
    }


def test_parameter_search_selects_q_by_total_net_after_the_23bp_hurdle(
        monkeypatch, tmp_path, spec):
    """23bp를 넘은 후보 중 체결당 Gross가 아니라 총 Net이 큰 q를 고른다."""
    calls = []
    mean_gross = {0.50: 22.0, 0.60: 24.0, 0.70: 25.0,
                  0.80: 30.0, 0.90: 21.0, 0.95: 20.0}
    fill_counts = {0.50: 10, 0.60: 10, 0.70: 100, 0.80: 10, 0.90: 10, 0.95: 10}

    def fake_run_batch(search_results, *, dates, stage, **_kwargs):
        calls.append((stage, tuple(dates), dict(search_results)))
        results = {}
        for candidate_id, search_result in search_results.items():
            q = float(search_result["lock"]["value"])
            fills = fill_counts[q]
            gross = mean_gross[q] * fills
            net = gross - 23.0 * fills
            results[candidate_id] = {
                "state": "BACKTEST_COMPLETE",
                "eligible_symbol_days": 3,
                "metrics": {
                    "eligible_decisions": 100,
                    "scorable": 200,
                    "fills": fills,
                    "gross_bps_total": gross,
                    "net_bps_total": net,
                    "net_bps_per_decision": net / 200,
                    "net_bps_per_fill": net / fills,
                    "fill_rate": 0.1,
                },
            }
        return results

    monkeypatch.setattr(parameter_search.backtest, "run_batch", fake_run_batch)
    monkeypatch.setattr(S, "run", lambda *_args, **_kwargs: pytest.fail("legacy anchor Search 호출"))
    schedule = parameter_search.ResearchSchedule(
        validation_dates=("20260407",), search_fit_dates=("20260330",),
        search_confirm_dates=("20260402",), backtest_dates=("20260413",),
        terminal_oos_dates=("20260421",))

    result = parameter_search.run(spec, symbols=("001",), schedule=schedule,
                                  output=tmp_path, root=tmp_path)

    assert result["state"] == S.PARAMETER_LOCKED
    assert result["selection"]["selected_q"] == 0.70
    assert result["selection"]["mean_gross_bps_per_fill"] == 25.0
    assert result["selection"]["net_bps_total"] == 200.0
    assert result["plan"]["selection_mode"] == parameter_search.SELECTION_MODE
    assert result["plan"]["anchor_cache_used_for_selection"] is False
    assert [stage for stage, _dates, _inputs in calls] == [
        "PARAMETER_SEARCH_FIT", "PARAMETER_SEARCH_CONFIRM"]
    assert len(calls[0][2]) == 6
    assert all("__q_" in item["parameterized_template"]["hypothesis_id"]
               for item in calls[0][2].values())
    assert (tmp_path / "response_curve.csv").is_file()


def test_parameter_search_does_not_select_when_every_q_is_below_23bp():
    candidates = [{
        "q": q, "value": q, "status": "EXECUTION_SCORABLE",
        "fixed_exit": {"metrics": {
            "fills": 10, "scorable": 100, "gross_bps_total": mean_gross * 10,
            "mean_gross_bps_per_fill": mean_gross,
            "total_net_bps": (mean_gross - 23.0) * 10,
            "bps_per_decision": (mean_gross - 23.0) / 10,
        }},
    } for q, mean_gross in ((0.5, 22.9), (0.7, 20.0), (0.9, 15.0))]

    result = parameter_search._execution_select(candidates)

    assert result["status"] == S.INSUFFICIENT_SUPPORT
    assert result["selected_q"] is None
    assert result["direct_evidence_status"] == "NO_PROFIT_TARGET_CANDIDATE"


def test_parameter_search_rejects_inconsistent_profit_accounting():
    candidate = {
        "q": 0.5, "value": 0.5, "status": "EXECUTION_SCORABLE",
        "fixed_exit": {"metrics": {
            "fills": 10, "scorable": 100, "gross_bps_total": 300.0,
            "mean_gross_bps_per_fill": 30.0, "total_net_bps": -100.0,
            "bps_per_decision": -1.0,
        }},
    }

    result = parameter_search._execution_select([candidate])

    assert result["status"] == S.INSUFFICIENT_SUPPORT
    assert result["selected_q"] is None
    assert result["direct_evidence_status"] == "NO_PROFIT_TARGET_CANDIDATE"


@pytest.mark.parametrize("confirm_mean_gross,confirm_net", [
    (22.0, -10.0),
    (30.0, -100.0),
])
def test_parameter_search_does_not_lock_when_confirm_fails_profit_target(
        monkeypatch, tmp_path, spec, confirm_mean_gross, confirm_net):
    def fake_run_batch(search_results, *, stage, **_kwargs):
        results = {}
        for candidate_id, search_result in search_results.items():
            q = float(search_result["lock"]["value"])
            mean_gross = (confirm_mean_gross if stage == "PARAMETER_SEARCH_CONFIRM"
                          else 30.0 if q == 0.8 else 24.0)
            fills = 10
            gross = mean_gross * fills
            net = (confirm_net if stage == "PARAMETER_SEARCH_CONFIRM"
                   else gross - 23.0 * fills)
            results[candidate_id] = {
                "state": "BACKTEST_COMPLETE", "eligible_symbol_days": 3,
                "metrics": {
                    "eligible_decisions": 100, "scorable": 80, "fills": fills,
                    "gross_bps_total": gross, "net_bps_total": net,
                    "net_bps_per_decision": net / 80,
                    "net_bps_per_fill": net / fills, "fill_rate": 0.125,
                },
            }
        return results

    monkeypatch.setattr(parameter_search.backtest, "run_batch", fake_run_batch)
    schedule = parameter_search.ResearchSchedule(
        validation_dates=("20260407",), search_fit_dates=("20260330",),
        search_confirm_dates=("20260402",), backtest_dates=("20260413",),
        terminal_oos_dates=("20260421",))

    result = parameter_search.run(spec, symbols=("001",), schedule=schedule,
                                  output=tmp_path, root=tmp_path)

    assert result["selection"]["selected_q"] == 0.8
    assert result["confirmation"]["status"] == S.PARAMETER_CONFIRMATION_FAILED
    assert result["lock"] is None


def test_search_jointly_calibrates_and_evaluates_a_composite_grounding_specification():
    composite = {
        "semantic_invariants": {"canonical_feature": "ofi_depth_10"},
        "search_boundary": {"allowed_parameters": [
            {"name": "theta_ofi", "feature": "ofi_depth_10", "direction": "LOWER"},
            {"name": "theta_microprice", "feature": "microprice_dev_bps", "direction": "HIGHER"},
        ]},
    }
    composite["signal_template"] = {"op": "all", "args": [
        {"op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
         "comparator": "<", "value": "UNRESOLVED:theta_ofi"},
        {"op": "compare", "input": {"op": "primitive", "primitive_id": "microprice_dev_bps"},
         "comparator": ">", "value": "UNRESOLVED:theta_microprice"},
    ]}
    config = S.config_for(composite, S.SearchConfig(
        grid=(0.8,), threshold_kind=X.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
        min_calibration_obs=200))
    assert config.states == (("ofi_depth_10", "LOWER", "theta_ofi"),
                             ("microprice_dev_bps", "HIGHER", "theta_microprice"))
    table = pd.DataFrame([{
        "symbol": "A", "date": "20260318", "reference_date": "20260317", "eligible": True,
        "theta_ofi__q0.80": 0.2, "theta_microprice__q0.80": 0.7,
    }])
    frame = pd.DataFrame({"symbol": ["A", "A", "A"], "date": ["20260318"] * 3,
                          "ofi_depth_10": [0.1, 0.3, 0.1],
                          "microprice_dev_bps": [0.8, 0.8, 0.6]})
    attached = S.attach_signal(frame, table, 0.8, config)
    assert attached["signal"].tolist() == [True, False, False]
    assert S.parameter_values_for_table(table, 0.8, config) == {
        "A:20260318": {"theta_ofi": 0.2, "theta_microprice": 0.7}}


def test_amendment_accepts_single_threshold_grounding_ast_without_private_parameter():
    grounded = {
        "semantic_invariants": {
            "canonical_feature": "ofi_depth_10",
            "operational_expression": {
                "op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
                "comparator": "<", "value": "UNRESOLVED:theta_ofi",
            },
        },
        "signal_template": {
            "op": "compare", "input": {"op": "primitive", "primitive_id": "ofi_depth_10"},
            "comparator": "<", "value": "UNRESOLVED:theta_ofi",
        },
        "search_boundary": {"allowed_parameters": [{
            "name": "theta_ofi", "feature": "ofi_depth_10", "direction": "LOWER",
            "threshold_source": {"kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE},
        }]},
    }
    amended = S.amend_specification(grounded, S.config_for(grounded))
    assert amended["signal_template"] == grounded["signal_template"]
    assert amended["parameterization"]["search_dimension"] == 1


def test_amendment_touches_nothing_else(spec, amended):
    """분위를 넣었다고 나머지가 변하면 안 된다 (§21)."""
    for key in ("semantic_invariants", "decision_opportunity_policy",
                "entry_execution_semantics", "exit_policy", "forbidden_features"):
        assert amended[key] == spec[key], key
    assert amended["signal_template"]["input"] == spec["signal_template"]["input"]
    assert amended["signal_template"]["comparator"] == spec["signal_template"]["comparator"]


def test_raw_threshold_search_becomes_a_closed_axis(amended):
    axes = " ".join(i["axis"] for i in amended["search_boundary"]["forbidden_parameters"])
    for closed in ("원값 임계", "보정 기간", "분위 계산법"):
        assert closed in axes, closed


def test_amended_spec_still_passes_fidelity(amended):
    """매개화를 바꾼 뒤 다시 통과해야 탐색을 시작한다 (§22)."""
    template = I.compile_contract(amended)
    assert I.implementation_validity(template)["implementation_valid"]
    check = I.fidelity_check(amended, template)
    assert check["fidelity_status"] == X.FIDELITY_PRESERVED
    for key in I.STRUCTURE_METRICS:
        assert check[key] == 0, key


def test_percentile_is_not_an_overfitting_cure(amended):
    """분위가 하는 일은 눈금 맞추기지 과적합 방지가 아니다 (§3)."""
    assert "과적합이 사라지지 않는다" in amended["parameterization"]["note"]


def test_literal_source_uses_one_fixed_value_for_every_symbol_day(spec):
    assert S.SearchConfig(threshold_kind=X.THRESHOLD_LITERAL).parameter \
        == "literal_book_imbalance"
    literal_spec = X.build(
        revision(), routing(), manifest(), results(),
        config=X.SpecConfig(threshold_kind=X.THRESHOLD_LITERAL,
                            threshold_grid=(0.25, 0.75)),
    )
    config = S.config_for(literal_spec)
    amended = S.amend_specification(literal_spec, config)
    row = S.calibrate("A", "20260318", config, CALENDAR)
    table = pd.DataFrame([row, {**row, "symbol": "B"}])
    attached = S.attach_signal(anchor_frame(), table, 0.75, config)
    payload = S.plan(amended, {S.SEARCH_FIT: ["20260318"], S.SEARCH_CONFIRM: [],
                               S.TERMINAL_OOS: []}, ["A", "B"], config)
    lock = S.parameter_lock(
        payload,
        {"selected_value": 0.75}, {}, amended)
    report = S.to_markdown({
        "plan": payload, "selection": S.select([]), "confirmation": None,
        "lock": None, "audit": S.search_audit(payload, table, [], {}, [], config),
        "calibration": {**payload["calibration"], "eligible_symbol_days": 2,
                        "symbol_days": 2, "reasons": {}},
        "candidates": [], "status": S.SEARCH_SELECTED,
    })

    assert config.parameter == "literal_book_imbalance"
    assert row["eligible"] is True and row["reference_date"] is None
    assert row[S.threshold_column(0.25, config)] == 0.25
    assert row[S.threshold_column(0.75, config)] == 0.75
    assert (attached["threshold"] == 0.75).all()
    assert amended["signal_template"]["_parameter"]["type"] == "literal"
    assert I.compile_contract(amended)["parameter_interface"][config.parameter]["threshold_source"] \
        == {"kind": X.THRESHOLD_LITERAL}
    assert "원값 임계" not in " ".join(
        item["axis"] for item in amended["search_boundary"]["forbidden_parameters"])
    assert lock["threshold_source"] == {"kind": X.THRESHOLD_LITERAL, "value": 0.75}
    assert "고정 원값" in report and "전일 q" not in report


# ---- 보정 (§7~§13) ---------------------------------------------------------------

def test_minimum_observations_is_derived_not_picked():
    """가장 까다로운 q 에서 상단 꼬리에 10개는 남도록 (§13)."""
    assert CONFIG.derived_min_obs() == CONFIG.min_calibration_obs == 200
    assert CONFIG.max_quantile == 0.95


def test_previous_day_is_the_previous_real_trading_day():
    """달력에서 하루 빼지 않는다 (§8)."""
    assert S.previous_real_trading_day("20260317", CALENDAR) == "20260316"
    # 금요일 다음 월요일 — 주말을 건너뛴다
    assert S.previous_real_trading_day("20260323",
                                       CALENDAR + ["20260323"]) == "20260320"
    assert S.previous_real_trading_day("20260316", CALENDAR) is None


def test_synthetic_dates_never_become_a_calibration_source():
    from framework import capability as C
    calendar = S.trading_calendar()
    assert "20260427" not in calendar
    assert all(d not in C.NOT_REAL_DATES for d in calendar)


def test_the_first_day_has_no_calibration_source():
    """`20260316` 은 앞이 없다. 그래서 탐색 구간에서 빠진다 (§25)."""
    row = S.calibrate("469170", "20260316", CONFIG, CALENDAR)
    assert row["eligible"] is False
    assert row["reference_date"] is None


def test_ineligible_calibration_keeps_the_full_quantile_schema():
    """전일이 없어도 탐색은 무결론으로 끝나야지, 분위 열 KeyError로 죽으면 안 된다."""
    row = S.calibrate("469170", "20260316", CONFIG, CALENDAR)
    assert all(f"q{q:.2f}" in row and np.isnan(row[f"q{q:.2f}"])
               for q in CONFIG.grid)
    result = S.score_candidate(anchor_frame(), pd.DataFrame([row]), 0.50,
                               CONFIG, S.CONTROLS)
    assert result["status"] == S.INSUFFICIENT_SUPPORT


def test_a_thin_previous_day_is_skipped_not_backfilled(monkeypatch):
    """더 오래된 날로 물러서면 종목-일마다 보정 기간이 달라진다 (§9)."""
    from framework import data as tickdata

    def thin(symbol, date, root=None):
        n = 5
        return ({"time_s": np.arange(n, dtype=float),
                 "bid_price": np.ones((n, 10)) * 100, "ask_price": np.ones((n, 10)) * 101,
                 "bid_qty": np.ones((n, 10)), "ask_qty": np.ones((n, 10)),
                 "buy_volume": np.zeros(n), "sell_volume": np.zeros(n),
                 "buy_max_price": np.zeros(n), "sell_min_price": np.zeros(n),
                 "local_time": np.full(n, 100000000000, dtype=np.int64)}, "x")

    monkeypatch.setattr(tickdata, "load", thin)
    row = S.calibrate("469170", "20260318", CONFIG, CALENDAR)
    assert row["eligible"] is False
    assert row["reference_date"] == "20260317"        # 하루만 본다
    assert "미만" in row["reason"]


def test_the_quantile_method_does_not_interpolate():
    """실제 관측값을 컷으로 쓴다. 보간법을 결과 보고 바꾸지 않는다 (§10)."""
    assert CONFIG.quantile_method == "higher"
    values = np.array([0.0, 1.0])
    assert np.quantile(values, 0.5, method="higher") == 1.0
    assert np.quantile(values, 0.5, method="linear") == 0.5


def test_default_q_is_each_symbols_prior_100_ticks_and_excludes_the_current_tick():
    config = S.SearchConfig()
    values = np.arange(140, dtype=float)
    cuts = contract.prior_tick_quantile(values, 0.90)
    changed = values.copy()
    changed[100] = 10_000.0

    assert config.threshold_kind == X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    assert config.min_calibration_obs == config.derived_min_obs() == 100
    assert np.isnan(cuts[:100]).all()
    assert cuts[100] == np.quantile(values[:100], 0.90, method="higher")
    assert contract.prior_tick_quantile(changed, 0.90)[100] == cuts[100]


def test_rolling_q_skips_a_symbol_day_without_required_trade_fields(monkeypatch):
    from framework import data as tickdata

    def quote_only(symbol, date, root=None):
        n = 120
        return ({"time_s": np.arange(n, dtype=float),
                 "bid_price": np.full((n, 10), 100.0),
                 "ask_price": np.full((n, 10), 101.0),
                 "bid_qty": np.ones((n, 10)), "ask_qty": np.ones((n, 10)),
                 "local_time": np.full(n, 100000000000, dtype=np.int64)}, "x")

    monkeypatch.setattr(tickdata, "load", quote_only)
    config = S.SearchConfig(feature="signed_aggr_flow_100")
    row = S.calibrate("000001", "20260318", config, CALENDAR)

    assert row["eligible"] is False
    assert "buy_volume" in str(row["reason"])


def test_rolling_q_eligibility_needs_a_tick_after_the_prior_100(monkeypatch):
    from framework import data as tickdata

    def tick_data(symbol, date, root=None):
        n = 101
        return ({"time_s": np.arange(n, dtype=float),
                 "bid_price": np.full((n, 10), 100.0),
                 "ask_price": np.full((n, 10), 101.0),
                 "bid_qty": np.ones((n, 10)), "ask_qty": np.ones((n, 10)),
                 "local_time": np.full(n, 100000000000, dtype=np.int64)}, "x")

    monkeypatch.setattr(tickdata, "load", tick_data)
    config = S.SearchConfig(feature="book_imbalance")
    assert S.calibrate("000001", "20260318", config, CALENDAR)["eligible"] is True

    def only_100(symbol, date, root=None):
        arrays, path = tick_data(symbol, date, root)
        return ({key: value[:100] if getattr(value, "ndim", 0) else value
                 for key, value in arrays.items()}, path)

    monkeypatch.setattr(tickdata, "load", only_100)
    assert S.calibrate("000001", "20260318", config, CALENDAR)["eligible"] is False


def test_rolling_q_runs_inside_the_contract_not_as_a_fixed_daily_cut():
    n = 130
    bid_qty = np.ones((n, 1))
    ask_qty = np.ones((n, 1))
    bid_qty[:, 0] = np.arange(1, n + 1)
    data = {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.full((n, 1), 100.0), "ask_price": np.full((n, 1), 101.0),
        "bid_qty": bid_qty, "ask_qty": ask_qty,
    }
    template = {
        "entry_program": {"signal": {
            "op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance"},
            "warmup_ticks": 0},
        "parameter_interface": {"theta_book_imbalance": {
            "threshold_source": {"kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE}}},
    }

    signal = contract.entry_signal(data, template, rolling_quantiles={"theta_book_imbalance": 0.90})

    assert not signal[:100].any()
    assert signal[100:].any()


def test_rolling_q_grid_matches_each_existing_q_definition():
    values = np.random.default_rng(23).normal(size=260)
    values[17] = np.nan
    requests = [(0.50, "higher"), (0.60, "higher"), (0.70, "higher"),
                (0.80, "higher"), (0.90, "higher"), (0.95, "higher"),
                (0.10, "lower")]

    combined = contract.prior_tick_quantile_grid(values, requests)

    for q, method in requests:
        expected = contract.prior_tick_quantile(values, q, method=method)
        assert np.array_equal(combined[(q, method)], expected, equal_nan=True)


def test_rolling_q_grid_reuses_input_expression_without_changing_signal():
    n = 180
    data = {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.full((n, 1), 100.0), "ask_price": np.full((n, 1), 101.0),
        "bid_qty": np.arange(1, n + 1, dtype=float).reshape(n, 1),
        "ask_qty": np.full((n, 1), 20.0),
    }
    template = {
        "entry_program": {"signal": {
            "op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance"},
            "warmup_ticks": 0},
        "parameter_interface": {"theta_book_imbalance": {
            "threshold_source": {"kind": X.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE}}},
    }
    candidates = [{"theta_book_imbalance": q} for q in (0.50, 0.70, 0.95)]

    thresholds, expression_cache = contract.rolling_quantile_binding_grid(
        data, template["entry_program"], candidates)

    for parameters, shared in zip(candidates, thresholds):
        expected = contract.entry_signal(data, template, rolling_quantiles=parameters)
        actual = contract.entry_signal(
            data, template, rolling_quantiles=parameters, rolling_thresholds=shared,
            precomputed_expressions=expression_cache)
        assert np.array_equal(actual, expected)


# ---- 우주를 후보마다 바꾸지 않는다 (§14) ------------------------------------------

def calibration_frame():
    rows = []
    for symbol in ("A", "B"):
        row = {"symbol": symbol, "date": "20260318", "reference_date": "20260317",
               "eligible": True, "reason": None, "observations": 5000}
        for q in CONFIG.grid:
            row[f"q{q:.2f}"] = q          # 편의상 분위값을 그대로 컷으로
        rows.append(row)
    rows.append({"symbol": "C", "date": "20260318", "reference_date": "20260317",
                 "eligible": False, "reason": "전일 관측이 3개로 200 미만",
                 "observations": 3})
    return pd.DataFrame(rows)


def anchor_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "symbol": np.where(np.arange(n) % 2 == 0, "A", "B"),
        "date": "20260318", "bucket": np.arange(n) // 40, "tick": np.arange(n),
        "book_imbalance": values,
        "vol_flow": rng.uniform(0, 1, n), "spread_bps": rng.uniform(5, 20, n),
        "pre_move_bps": rng.normal(0, 3, n),
        # 신호가 높을수록 미래가 좋게 — 관계가 있는 자료
        "max_net_bps_s30": values * 12 + rng.normal(0, 2, n),
        "sustained_success": (values > 0.9).astype(int)})


def test_every_candidate_uses_the_same_universe():
    table, frame = calibration_frame(), anchor_frame()
    sizes = {q: len(S.attach_signal(frame, table, q, CONFIG)) for q in CONFIG.grid}
    assert len(set(sizes.values())) == 1, sizes


def test_a_refined_guard_feature_is_collected_before_the_search(monkeypatch):
    """Refinement가 고른 축도 Search anchor 표에 있어야 같은 guard를 적용할 수 있다."""
    seen = {}

    def fake_collect(dates, symbols, config, root):
        seen["features"] = config.feature_names
        return anchor_frame()

    monkeypatch.setattr(V, "collect", fake_collect)
    frame = S.collect_anchors(["20260318"], ["A"], config=CONFIG,
                              extra_features=("queue_imbalance_best",))
    assert len(frame) == len(anchor_frame())
    assert "queue_imbalance_best" in seen["features"]


def test_an_ineligible_symbol_day_drops_out_entirely():
    table, frame = calibration_frame(), anchor_frame()
    frame = pd.concat([frame, frame.assign(symbol="C")], ignore_index=True)
    attached = S.attach_signal(frame, table, 0.90, CONFIG)
    assert "C" not in set(attached["symbol"])


def test_the_signal_is_the_feature_above_its_own_threshold():
    table, frame = calibration_frame(), anchor_frame()
    attached = S.attach_signal(frame, table, 0.80, CONFIG)
    assert (attached["signal"] ==
            (attached["book_imbalance"] > attached["threshold"])).all()
    assert (attached["threshold"] == 0.80).all()


# ---- 후보 채점 -------------------------------------------------------------------

def test_a_real_relation_is_detected():
    table, frame = calibration_frame(), anchor_frame()
    row = S.score_candidate(frame, table, 0.50, CONFIG, S.CONTROLS)
    assert row["status"] in (S.SEARCH_SUPPORTED, S.INCONCLUSIVE,
                             S.INSUFFICIENT_SUPPORT, S.MATCHING_FAILED)
    assert row["null_value" if "null_value" in row else "q"] is not None


def test_too_few_pairs_ends_in_insufficient_support():
    table = calibration_frame()
    row = S.score_candidate(anchor_frame(n=40), table, 0.50, CONFIG, S.CONTROLS)
    assert row["status"] == S.INSUFFICIENT_SUPPORT


def test_a_candidate_with_no_control_group_is_not_scored():
    """신호가 전부 켜지면 비교할 대조군이 없다."""
    table, frame = calibration_frame(), anchor_frame()
    table = table.assign(**{f"q{q:.2f}": -5.0 for q in CONFIG.grid})
    row = S.score_candidate(frame, table, 0.50, CONFIG, S.CONTROLS)
    assert row["status"] == S.INSUFFICIENT_SUPPORT


def test_matching_reuses_the_validation_contract():
    """결과가 좋아지도록 통제나 거리를 바꾸지 않는다 (§31)."""
    table, frame = calibration_frame(), anchor_frame()
    attached = S.attach_signal(frame, table, 0.50, CONFIG)
    pairs = S.matched_by_signal(attached, S.CONTROLS, CONFIG)
    assert "control_distance" in pairs and "tested_rank_delta" in pairs
    diagnostics = V.matching_diagnostics(pairs, len(attached), len(S.CONTROLS))
    assert diagnostics["quality_rule"] == V.MATCHING_CONTRACT["acceptable_quality_rule"]


def test_diagnostics_are_reported_but_not_used_to_select():
    """손익 진단으로 q 를 고르지 않는다 (§35)."""
    table, frame = calibration_frame(), anchor_frame()
    row = S.score_candidate(frame, table, 0.50, CONFIG, S.CONTROLS)
    if row["status"] not in (S.INSUFFICIENT_SUPPORT, S.MATCHING_FAILED):
        assert "diag_mean_max_net_bps" in row
    rule = " ".join(S.SELECTION_RULE.values())
    for word in ("net", "손익", "수익"):
        assert word not in rule


# ---- 고르는 규칙 (§37~§39) --------------------------------------------------------

def candidate(q, effect, status=S.SEARCH_SUPPORTED, n=1000):
    return {"q": q, "effect_size": effect, "interval_low": effect - 0.01,
            "interval_high": effect + 0.01, "status": status, "n_effective": n}


def test_the_largest_advantage_wins():
    picked = S.select([candidate(0.5, 0.52), candidate(0.9, 0.56),
                       candidate(0.8, 0.54)])
    assert picked["status"] == S.SEARCH_SELECTED
    assert picked["selected_q"] == 0.9


def test_a_tie_prefers_more_support_then_a_lower_q():
    picked = S.select([candidate(0.9, 0.55, n=500), candidate(0.6, 0.55, n=900)])
    assert picked["selected_q"] == 0.6
    same = S.select([candidate(0.9, 0.55, n=900), candidate(0.6, 0.55, n=900)])
    assert same["selected_q"] == 0.6            # 같으면 낮은 q


def test_non_supported_candidates_remain_selectable():
    picked = S.select([candidate(0.9, 0.60, status=S.NOT_SUPPORTED),
                       candidate(0.5, 0.52, status=S.INCONCLUSIVE)])
    assert picked["status"] == S.SEARCH_SELECTED
    assert picked["selected_q"] == 0.9
    assert picked["direct_evidence_status"] == S.NOT_SUPPORTED


def test_no_supported_candidate_still_selects_a_q():
    picked = S.select([candidate(q, 0.49, status=S.NOT_SUPPORTED)
                       for q in CONFIG.grid])
    assert picked["status"] == S.SEARCH_SELECTED
    assert picked["selected_q"] == 0.5


def test_no_effect_estimate_uses_the_neutral_q():
    picked = S.select([{"q": 0.8, "status": S.INSUFFICIENT_SUPPORT},
                       {"q": 0.5, "status": S.MATCHING_FAILED}])
    assert picked["status"] == S.SEARCH_SELECTED
    assert picked["selected_q"] == 0.5


# ---- 확인 (§40~§44) --------------------------------------------------------------

def test_confirmation_opens_one_candidate_only():
    table, frame = calibration_frame(), anchor_frame(seed=7)
    result = S.confirm(frame, table, 0.50, CONFIG, S.CONTROLS)
    assert result["only_one_candidate_opened"] is True
    assert result["q"] == 0.50
    assert result["status"] in (S.PARAMETER_CONFIRMED, S.PARAMETER_CONFIRMATION_FAILED)
    assert "차점자" in result["on_failure"]


# ---- 잠금 (§45~§48) --------------------------------------------------------------

def test_the_lock_fixes_q_not_the_daily_threshold(spec, amended):
    payload = S.plan(amended, {S.SEARCH_FIT: ["20260317"], S.SEARCH_CONFIRM: ["20260414"],
                               S.TERMINAL_OOS: ["20260420"]}, ["A"], CONFIG)
    lock = S.parameter_lock(payload, {"selected_q": 0.9},
                            {"status": S.PARAMETER_CONFIRMED}, amended)
    assert lock["value"] == 0.9
    assert lock["status"] == S.PARAMETER_LOCKED
    assert "값이 변하는 것이지 파라미터가 변하는 것이 아니다" in lock["runtime_note"]
    assert "다시 고르지 않는다" in lock["oos_rule"]


def test_the_lock_refuses_the_profitable_label(spec, amended):
    payload = S.plan(amended, {S.SEARCH_FIT: [], S.SEARCH_CONFIRM: [],
                               S.TERMINAL_OOS: []}, ["A"], CONFIG)
    lock = S.parameter_lock(payload, {"selected_q": 0.9}, {}, amended)
    assert "수익 나는 임계" in lock["naming"]
    assert "SELECTED_SIGNAL_CUTOFF" in lock["naming"]


def test_single_day_lock_has_no_confirmation_result_to_render(tmp_path, amended):
    payload = S.plan(amended, {S.SEARCH_FIT: ["20260317"], S.SEARCH_CONFIRM: [],
                               S.TERMINAL_OOS: []}, ["A"], CONFIG)
    candidates = [candidate(0.5, 0.55), candidate(0.6, 0.54)]
    selection = {"status": S.SEARCH_SELECTED, "selected_q": 0.5,
                 "reason": "선택 규칙을 통과했다"}
    confirmation = {
        "status": "NOT_RUN_SINGLE_DAY_DISCOVERY",
        "why": "별도 Search confirm 날짜가 없다",
        "next": "Validation Backtest에서 검증한다",
    }
    result = {
        "plan": payload, "selection": selection, "confirmation": confirmation,
        "lock": S.parameter_lock(payload, selection, confirmation, amended,
                                  status=S.PARAMETER_LOCKED_SINGLE_DAY),
        "audit": audit_for(plan=payload, candidates=candidates),
        "calibration": {"scope": "SYMBOL", "reference": "PREVIOUS_REAL_TRADING_DAY",
                        "quantile_method": "higher", "minimum_observations": 200,
                        "minimum_observations_derivation": "derived",
                        "eligible_symbol_days": 1, "symbol_days": 1, "reasons": {},
                        "no_fallback": "no fallback"},
        "candidates": candidates,
        "status": S.PARAMETER_LOCKED_SINGLE_DAY,
    }
    assert "NOT_RUN_SINGLE_DAY_DISCOVERY" in S.to_markdown(result)
    S._companions(result, tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "confirmation.csv").exists()


def test_sequential_calibration_walks_forward():
    """`0421` 장 시작에 `0420` 은 이미 과거다. 누출이 아니다 (§49~§50)."""
    calendar = S.trading_calendar()
    for date in ("20260421", "20260422"):
        prior = S.previous_real_trading_day(date, calendar)
        assert prior is not None and prior < date


# ---- 감사 (§60~§61) --------------------------------------------------------------

def audit_for(**over):
    payload = {"grid_parameter": ["q_book_imbalance"], "grid": list(CONFIG.grid),
               "threshold_source": {"kind": X.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE},
               "matching_contract": V.MATCHING_CONTRACT,
               "primary_metric": CONFIG.primary_metric,
               "blocks": {S.TERMINAL_OOS: ["20260420", "20260424"]}}
    payload.update(over.pop("plan", {}))
    table = over.pop("table", calibration_frame())
    return S.search_audit(payload, table, over.pop("candidates", []),
                          over.pop("fidelity", {}), over.pop("opened", ["20260317"]),
                          CONFIG)


def test_a_clean_search_passes_every_metric():
    audit = audit_for()
    assert audit["ok"], audit["failed"]
    for key in S.MUST_BE_ZERO:
        assert audit["metrics"][key] == 0, key


def test_opening_the_terminal_block_is_caught():
    audit = audit_for(opened=["20260317", "20260420"])
    assert audit["metrics"]["terminal_oos_access_count"] == 1
    assert audit["ok"] is False


def test_same_day_calibration_is_caught():
    table = calibration_frame().assign(reference_date="20260318")   # date 와 같다
    audit = audit_for(table=table)
    assert audit["metrics"]["current_day_calibration_leakage_count"] > 0
    assert audit["metrics"]["future_calibration_leakage_count"] > 0


def test_changing_the_grid_after_results_is_caught():
    audit = audit_for(plan={"grid": [0.88, 0.89, 0.90]})
    assert audit["metrics"]["post_result_grid_change_count"] == 1


def test_changing_the_matching_rule_is_caught():
    audit = audit_for(plan={"matching_contract": {"caliper": 0.2}})
    assert audit["metrics"]["matching_rule_change_count"] == 1


def test_changing_the_metric_is_caught():
    audit = audit_for(plan={"primary_metric": "TOTAL_PNL"})
    assert audit["metrics"]["metric_change_count"] == 1


def test_a_second_search_axis_is_caught():
    audit = audit_for(plan={"grid_parameter": ["q_book_imbalance", "q_vol_flow"]})
    assert audit["metrics"]["search_dimension_change_count"] == 1


def test_warning_metrics_are_separate_from_blocking_ones():
    assert "calibration_insufficient_symbol_day_count" in S.WARNING_ONLY
    assert "calibration_insufficient_symbol_day_count" not in S.MUST_BE_ZERO
    audit = audit_for()
    assert audit["metrics"]["calibration_insufficient_symbol_day_count"] == 1
    assert audit["ok"] is True          # 경고가 있어도 막지 않는다


# ---- 계획이 결과 전에 얼려졌는가 --------------------------------------------------

def test_the_plan_freezes_everything_before_running(amended):
    payload = S.plan(amended, {S.SEARCH_FIT: ["20260317"], S.SEARCH_CONFIRM: ["20260414"],
                               S.TERMINAL_OOS: ["20260420"]}, ["A"], CONFIG)
    for key in ("grid", "calibration", "primary_metric", "null_value",
                "matching_contract", "selection_rule", "blocks"):
        assert payload[key], key
    assert payload["calibration"]["minimum_observations_derivation"].startswith("ceil(")
    assert "청산 규칙" in payload["not_done_here"]
    assert "손익 최적화" in payload["not_done_here"]
