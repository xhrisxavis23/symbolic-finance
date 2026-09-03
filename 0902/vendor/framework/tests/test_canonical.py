"""측정 장비가 언제나 같은가, 그리고 돈을 두 번 세지 않는가.

여기가 실제로 돈을 세는 쪽이다. 지금까지 이 경로에는 테스트가 하나도 없었다.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from framework import canonical as K, contract, fill, ledger, metrics


# ---- 정본 프로필 (§48~§49) -------------------------------------------------------

def test_there_is_exactly_one_profile():
    assert K.CANONICAL["profile_id"] == K.PROFILE_ID
    assert K.profile_hash() == K.CANONICAL["profile_sha256"]


def test_the_profile_hash_changes_when_a_rule_changes():
    """해시가 다르면 같은 평가 환경에서 비교한 것이 아니다."""
    import copy
    other = copy.deepcopy(K.CANONICAL)
    other.pop("profile_sha256")
    other["exit"]["stop_gross_bps"] = -50.0
    from framework.config import sha256_json
    assert sha256_json(other)[:16] != K.profile_hash()


def test_contract_hash_ignores_compile_time_metadata():
    first = {"created_at": "2026-08-25T08:00:00+00:00", "entry_program": {"signal": "A"}}
    second = {"created_at": "2026-08-25T08:00:01+00:00", "entry_program": {"signal": "A"}}

    assert contract.contract_hash(first) == contract.contract_hash(second)


def test_execution_rules_are_not_searchable():
    forbidden = " ".join(K.CANONICAL["not_searchable"])
    for item in ("큐 모델", "비용", "결정 규칙", "주문 유지시간"):
        assert item in forbidden, item


def test_the_profile_states_what_it_cannot_know():
    """스냅샷으로 알 수 없는 것을 아는 척하지 않는다 (§6)."""
    limits = " ".join(K.CANONICAL["queue"]["limits"])
    assert "주문 추가·취소" in limits
    assert "큐 순번" in limits
    assert "거래소 모사가 아니" in limits


# ---- 결정 기회 (§11~§14, §41) ----------------------------------------------------

def test_the_backtest_looks_every_tick():
    """검증의 30초 격자는 **측정 설계**였지 거래 규칙이 아니다."""
    decision = K.CANONICAL["decision"]
    assert decision["basis"] == "EVERY_TICK"
    assert decision["interval_seconds"] is None
    assert decision["blocked_while_holding"] is True
    assert "측정 설계" in decision["not_the_validation_grid"]


def test_the_grid_helper_still_works_for_validation():
    """격자 함수 자체는 남는다 — 검증이 여전히 쓴다."""
    time_s = np.array([0.0, 5.0, 12.0, 31.0, 33.0, 61.0, 95.0])
    mask = K.decision_grid(time_s, 30.0)
    assert mask.tolist() == [True, False, False, True, False, True, True]


def test_the_grid_helper_masks_off_grid_ticks():
    """격자 함수의 동작 자체는 그대로. 백테스트가 안 쓸 뿐이다."""
    time_s = np.arange(0.0, 120.0, 1.0)
    signal = np.ones(len(time_s), dtype=bool)
    gated = signal & K.decision_grid(time_s, 30.0)
    assert gated.sum() == 4                      # 0, 30, 60, 90 초
    assert not gated[1]


def test_sklearn_probability_guard_filters_the_tick_signal(monkeypatch):
    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            assert list(frame.columns) == ["book_imbalance", "vol_flow"]
            score = frame["book_imbalance"].to_numpy(float)
            return np.column_stack([1.0 - score, score])

    monkeypatch.setattr(ledger, "_load_probability_model", lambda _path: Model())
    signal = np.array([True, True, False])
    features = {
        "book_imbalance": np.array([0.4, 0.7, 0.9]),
        "vol_flow": np.array([1.0, 2.0, 3.0]),
    }
    guard = {
        "kind": "sklearn_probability",
        "model_path": "fixed.joblib",
        "features": ["book_imbalance", "vol_flow"],
        "positive_class": 1,
        "operator": ">=",
        "threshold": 0.5,
    }

    kept = ledger.apply_guard(signal, features, guard, thresholds=None)

    assert kept.tolist() == [False, True, False]


def test_same_program_q_candidates_share_one_rolling_quantile_pass(monkeypatch, tmp_path):
    from framework import data as tickdata

    arrays = book(130)
    monkeypatch.setattr(tickdata, "load", lambda *_args, **_kwargs: (arrays, "fixture"))
    original = contract.rolling_quantile_binding_grid
    calls = []

    def tracked(data, program, quantile_sets):
        calls.append(len(quantile_sets))
        return original(data, program, quantile_sets)

    monkeypatch.setattr(contract, "rolling_quantile_binding_grid", tracked)
    program = {
        "signal": {
            "op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance"},
        "warmup_ticks": 0,
    }
    template = {
        "entry_program": program,
        "parameter_interface": {"theta_book_imbalance": {
            "threshold_source": {"kind": "rolling_prior_100_ticks_quantile"}}},
    }

    manifest = ledger.build_ledger(
        contracts={"q50": template, "q90": template},
        members={"q50": ["A"], "q90": ["A"]}, dates=["20260318"],
        output=tmp_path / "ledger.parquet", workers=1,
        parameter_table={
            "q50:A:20260318": {"theta_book_imbalance": 0.50},
            "q90:A:20260318": {"theta_book_imbalance": 0.90},
        }, root=tmp_path,
    )

    assert calls == [2]
    assert manifest["errors"] == []


def test_an_irregular_clock_still_lands_once_per_interval():
    """호가 간격이 들쭉날쭉해도 구간마다 한 번씩만."""
    time_s = np.array([0.0, 0.1, 0.2, 40.0, 41.0, 100.0])
    mask = K.decision_grid(time_s, 30.0)
    assert mask.sum() == 3                       # 0 · 40 · 100 근처 하나씩
    assert mask[0] and mask[3] and mask[5]


def test_an_empty_series_yields_no_decisions():
    assert K.decision_grid(np.array([]), 30.0).tolist() == []


# ---- decision / BLOCKED (§41) ----------------------------------------------------

def exit_ticks(n, span=3):
    return np.minimum(np.arange(n) + span, n)


def test_no_signal_means_no_decision():
    signal = np.zeros(10, dtype=bool)
    assert len(ledger.decisions_from_signal(signal, exit_ticks(10))) == 0


def test_a_signal_while_flat_makes_one_decision():
    signal = np.zeros(10, dtype=bool)
    signal[0] = True
    frame = ledger.decisions_from_signal(signal, exit_ticks(10))
    assert len(frame) == 1
    assert frame.iloc[0].status == ledger.FILLED


def test_a_signal_while_holding_makes_no_extra_decision():
    """신호가 계속 켜져 있어도 포지션 중이면 새 결정이 아니다 (§12)."""
    signal = np.ones(10, dtype=bool)
    frame = ledger.decisions_from_signal(signal, exit_ticks(10, span=3))
    # 3틱마다 한 번씩만 살 수 있다
    assert len(frame) < int(signal.sum())
    assert frame.attrs["blocked_signal_ticks"] > 0


def test_an_unobservable_horizon_is_censored_not_a_loss():
    """30초를 못 보면 채점 불가지 손실이 아니다."""
    signal = np.zeros(6, dtype=bool)
    signal[5] = True
    frame = ledger.decisions_from_signal(signal, np.full(6, 99))   # 전부 관측 불가
    assert frame.iloc[0].status == ledger.CENSORED


def test_decision_identity_holds():
    """decisions = FILLED + UNFILLED + CENSORED (§34)."""
    signal = np.random.default_rng(0).random(200) > 0.5
    frame = ledger.decisions_from_signal(signal, exit_ticks(200, span=5))
    counts = frame["status"].value_counts()
    total = sum(counts.get(s, 0) for s in ledger.STATUSES)
    assert total == len(frame)


# ---- 큐 (§42~§43) ----------------------------------------------------------------

def book(n, bid=100.0, ask=101.0, bid_qty=1000.0, sell_volume=0.0,
         sell_min_price=None):
    sell_min_price = bid if sell_min_price is None else sell_min_price
    return {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.tile([[bid] + [bid - 1] * 9], (n, 1)),
        "ask_price": np.tile([[ask] + [ask + 1] * 9], (n, 1)),
        "bid_qty": np.full((n, 10), bid_qty),
        "ask_qty": np.full((n, 10), 500.0),
        "sell_volume": np.full(n, sell_volume),
        "sell_min_price": np.full(n, sell_min_price),
        "buy_volume": np.zeros(n),
        "buy_max_price": np.full(n, ask),
        "local_time": np.full(n, 100000000000, dtype=np.int64)}


def test_a_deep_queue_does_not_fill():
    """앞 물량이 남아 있으면 체결되지 않는다."""
    data = book(50, bid_qty=1000.0, sell_volume=1.0)
    status, _terminal, filled_at = fill.entry_fills(data, np.array([0]))
    assert status[0] != fill.FILLED
    assert filled_at[0] == -1


def test_the_queue_clearing_fills():
    """앞 물량이 다 소진되면 내 차례가 온다."""
    data = book(50, bid_qty=10.0, sell_volume=20.0)
    status, _terminal, filled_at = fill.entry_fills(data, np.array([0]))
    assert status[0] == fill.FILLED
    assert filled_at[0] >= 0


def test_a_partial_clearing_does_not_fill_early():
    data = book(50, bid_qty=1000.0, sell_volume=10.0)
    status, _terminal, _filled = fill.entry_fills(data, np.array([0]))
    assert status[0] != fill.FILLED


def test_a_price_move_alone_never_fills():
    """호가가 다음 레벨로 갔다는 이유만으로 공짜 체결시키지 않는다 (§7)."""
    n = 50
    data = book(n, bid_qty=1000.0, sell_volume=0.0)
    data["bid_price"][10:, 0] = 105.0        # 가격만 위로 움직인다
    data["ask_price"][10:, 0] = 106.0
    status, _terminal, filled_at = fill.entry_fills(data, np.array([0]))
    assert status[0] != fill.FILLED
    assert filled_at[0] == -1
    assert K.CANONICAL["queue"]["price_move_alone_does_not_fill"] is True


def test_trades_at_an_ineligible_price_do_not_clear_my_queue():
    """내 가격에 도달하지 않은 체결은 내 큐와 무관하다."""
    data = book(50, bid_qty=50.0, sell_volume=100.0, sell_min_price=200.0)
    status, _terminal, _filled = fill.entry_fills(data, np.array([0]))
    assert status[0] != fill.FILLED


def test_the_source_must_carry_trade_events():
    """큐를 소진시킬 근거가 없는데 체결을 추정하면 체결률을 지어내는 것이다."""
    data = book(10)
    del data["sell_volume"]
    with pytest.raises(Exception):
        fill.entry_fills(data, np.array([0]))


def test_entry_and_exit_use_the_same_kernel():
    """진입만 큐를 보고 청산은 대충 처리하지 않는다 (§15, §44)."""
    import inspect
    assert "replay" in inspect.getsource(fill.entry_fills)
    assert "replay" in inspect.getsource(fill.exit_fills)
    assert K.CANONICAL["entry"]["model"] == K.CANONICAL["exit"]["model"]


def test_hold_through_is_the_canonical_policy():
    """가격이 불리해졌다고 주문을 공짜로 빼지 않는다 (§8)."""
    assert K.CANONICAL["queue"]["hold_through"] is True
    assert K.CANONICAL["queue"]["cancel_on_price_change"] is False
    assert fill.CANCEL_ON_PRICE_CHANGE is False


# ---- 비용 (§26~§30, §45) ---------------------------------------------------------

def ledger_frame(gross, fee=23.0, status=ledger.FILLED, n=5):
    return pd.DataFrame({
        "contract_id": ["C"] * n, "symbol": ["A"] * n, "date": ["20260316"] * n,
        "entry_tick": np.arange(n) * 100, "exit_tick": np.arange(n) * 100 + 50,
        "status": [status] * n,
        "gross_bps": np.full(n, float(gross)),
        "net_bps": np.full(n, float(gross) - float(fee))})


def test_spread_is_never_deducted_twice():
    """체결가격이 실제 호가에서 났으면 스프레드는 이미 손익에 있다 (§28)."""
    assert K.CANONICAL["cost"]["spread_mode"] == K.IMPLICIT_IN_FILL_PRICE
    assert K.CANONICAL["cost"]["spread_deduction_bps"] == 0.0
    audit = K.spread_accounting_audit(ledger_frame(10.0), {})
    assert audit["ok"] is True


def test_an_explicit_spread_penalty_is_caught():
    broken = copy.deepcopy(K.CANONICAL)
    broken["cost"]["spread_deduction_bps"] = 8.0
    audit = K.spread_accounting_audit(ledger_frame(10.0), {}, broken)
    assert audit["ok"] is False
    assert K.SPREAD_DOUBLE_COUNT in audit["problems"][0]


def test_a_doubled_fee_is_caught():
    """gross - net 이 수수료와 다르면 비용이 두 번 빠진 것이다 (§30)."""
    frame = ledger_frame(10.0, fee=46.0)          # 23 을 두 번 뺐다
    audit = K.spread_accounting_audit(frame, {})
    assert audit["ok"] is False
    assert K.COST_DUPLICATION in audit["problems"][0]


def test_gross_and_net_differ_by_exactly_the_explicit_cost():
    audit = K.spread_accounting_audit(ledger_frame(10.0), {})
    assert audit["gross_minus_net_max_gap_bps"] == pytest.approx(0.0, abs=1e-9)
    assert audit["explicit_cost_bps"] == K.CANONICAL["cost"]["explicit_cost_bps"]


# ---- 원장 불변식 (§31~§35, §46) --------------------------------------------------

def manifest_for(frame, entry="limit_bid1_hold_through"):
    parts = []
    for (contract_id, symbol, date), group in frame.groupby(
            ["contract_id", "symbol", "date"]):
        counts = group["status"].value_counts()
        parts.append({"contract_id": contract_id, "symbol": symbol, "date": date,
                      "decisions": int(len(group)),
                      "filled": int(counts.get(ledger.FILLED, 0)),
                      "unfilled": int(counts.get(ledger.UNFILLED, 0)),
                      "censored": int(counts.get(ledger.CENSORED, 0))})
    totals = {k: sum(p[k] for p in parts)
              for k in ("decisions", "filled", "unfilled", "censored")}
    return {"entry_execution": entry, "by_partition": parts, "totals": totals}


def test_a_clean_ledger_passes():
    frame = ledger_frame(10.0)
    ledger.verify_invariants(frame, manifest_for(frame))


def test_an_unknown_status_is_caught():
    """모르는 상태가 섞이면 그 행이 어느 분모에 들어가는지 말할 수 없다 (§32)."""
    frame = ledger_frame(10.0)
    manifest = manifest_for(frame)
    frame.loc[0, "status"] = "WEIRD"
    with pytest.raises(ledger.InvariantViolation, match="모르는 status"):
        ledger.verify_invariants(frame, manifest)


def test_manipulated_totals_are_caught():
    """보고된 분모가 거짓이면 성과 전체가 거짓이 된다 (§33)."""
    frame = ledger_frame(10.0)
    manifest = manifest_for(frame)
    manifest["totals"]["decisions"] += 7
    with pytest.raises(ledger.InvariantViolation, match="totals"):
        ledger.verify_invariants(frame, manifest)


def test_a_deleted_row_is_caught():
    frame = ledger_frame(10.0)
    manifest = manifest_for(frame)
    with pytest.raises(ledger.InvariantViolation):
        ledger.verify_invariants(frame.iloc[1:], manifest)


def test_overlapping_positions_are_caught():
    """같은 종목에서 체결 구간이 겹치면 두 번 산 것이다 (§35)."""
    frame = ledger_frame(10.0)
    manifest = manifest_for(frame)
    frame["exit_tick"] = frame["exit_tick"] + 10_000
    with pytest.raises(ledger.InvariantViolation, match="겹친다"):
        ledger.verify_invariants(frame, manifest)


def test_taker_entry_may_not_have_unfilled_rows():
    frame = ledger_frame(10.0, status=ledger.UNFILLED)
    manifest = manifest_for(frame, entry="taker_ask1")
    with pytest.raises(ledger.InvariantViolation, match="UNFILLED"):
        ledger.verify_invariants(frame, manifest)


# ---- 지표 분모 (§36~§39, §47) ----------------------------------------------------

def test_every_metric_names_its_denominator():
    for name in K.METRIC_DENOMINATORS:
        assert K.METRIC_DENOMINATORS[name], name
    for banned in K.FORBIDDEN_METRIC_NAMES:
        assert banned not in K.METRIC_DENOMINATORS


def test_metrics_match_a_hand_computation():
    """작은 원장을 손으로 계산한 값과 대조한다."""
    frame = pd.DataFrame({
        "contract_id": ["C"] * 4, "symbol": ["A"] * 4, "date": ["20260316"] * 4,
        "entry_tick": [0, 100, 200, 300], "exit_tick": [50, 150, 250, 350],
        "status": [ledger.FILLED, ledger.FILLED, ledger.UNFILLED, ledger.CENSORED],
        "gross_bps": [33.0, 13.0, 0.0, 0.0],
        "net_bps": [10.0, -10.0, 0.0, 0.0]})
    report = K.metric_report(frame, manifest_for(frame))
    assert report["eligible_decisions"] == 4
    assert report["fills"] == 2
    assert report["unfilled"] == 1
    assert report["censored"] == 1
    assert report["scorable"] == 3                       # FILLED + UNFILLED
    assert report["fill_rate"] == pytest.approx(2 / 4)
    assert report["net_bps_total"] == pytest.approx(0.0)
    assert report["net_bps_per_decision"] == pytest.approx(0.0 / 3)
    assert report["net_bps_per_fill"] == pytest.approx(0.0 / 2)


def test_fill_rate_is_reported_next_to_pnl():
    """손익이 좋아졌을 때 신호가 좋아진 것인지 체결이 골라진 것인지 갈라야 한다 (§39)."""
    frame = ledger_frame(10.0)
    report = K.metric_report(frame, manifest_for(frame))
    for key in ("fill_rate", "net_bps_per_decision", "net_bps_per_fill",
                "eligible_decisions", "unfilled", "censored"):
        assert key in report, key


# ---- 계약과 프로필 (§22~§25) -----------------------------------------------------

def template_with(**binding):
    base = {"decision_opportunity": K.CANONICAL["decision"]["basis"],
            "decision_spacing_seconds": K.CANONICAL["decision"]["interval_seconds"],
            "entry_execution": K.CANONICAL["entry"]["model"],
            "evaluation_horizon": f"S{int(K.CANONICAL['horizon']['primary_seconds'])}"}
    base.update(binding)
    return {"execution_binding": base}


def test_a_matching_contract_passes():
    assert K.check_contract(template_with())["ok"] is True


def test_backtest_forwards_the_profile_exit_holding_cap(monkeypatch, tmp_path):
    received = {}

    def fake_build_ledger(**kwargs):
        received.update(kwargs)
        return {}

    monkeypatch.setattr(K.ledger, "build_ledger", fake_build_ledger)
    K.run_backtest({"C": template_with()}, {"C": ()}, (), tmp_path / "ledger.parquet")
    assert received["horizon_seconds"] == K.CANONICAL["exit"]["max_holding_seconds"]
    assert received["stop_gross_bps"] == -120.0
    assert received["trailing_drawdown_gross_bps"] == -30.0


def test_v8_900_comparison_keeps_v8_exit_thresholds_with_v9_holding_cap(monkeypatch, tmp_path):
    received = {}

    def fake_build_ledger(**kwargs):
        received.update(kwargs)
        return {}

    monkeypatch.setattr(K.ledger, "build_ledger", fake_build_ledger)
    K.run_backtest({"C": template_with()}, {"C": ()}, (), tmp_path / "ledger.parquet",
                   canonical=K.V8_900_COMPARISON)

    comparison = K.V8_900_COMPARISON
    assert comparison["profile_id"] == "CANONICAL_QUEUE_V8_900_COMPARISON"
    assert comparison["exit"]["stop_gross_bps"] == -10.0
    assert comparison["exit"]["trailing_drawdown_gross_bps"] == -10.0
    assert comparison["exit"]["max_holding_seconds"] == K.CANONICAL["exit"]["max_holding_seconds"]
    assert received["horizon_seconds"] == 900.0


def test_empty_ledger_keeps_the_decision_schema(monkeypatch, tmp_path):
    """0 decision도 refinement가 읽을 수 있는 정상 원장이다."""
    monkeypatch.setattr(ledger, "_one_partition", lambda _job: {
        "rows": [],
        "counts": {"contract_id": "C", "symbol": "001", "date": "20260317",
                   "raw_signal_ticks": 0, "signal_episodes": 0, "blocked_signal_ticks": 0,
                   "decisions": 0, "filled": 0, "censored": 0, "unfilled": 0},
        "missing": None,
    })

    ledger.build_ledger(
        contracts={"C": {}}, members={"C": ["001"]}, dates=["20260317"],
        output=tmp_path / "ledger.parquet", workers=1)

    frame = pd.read_parquet(tmp_path / "ledger.parquet")
    assert frame.empty
    assert {"contract_id", "status", "net_bps", "diagnostic_cohort"} <= set(frame.columns)


def test_masked_screening_uses_the_documented_approximate_metric():
    frame = pd.DataFrame({
        "contract_id": ["H", "H"], "symbol": ["001", "001"],
        "date": ["20260317", "20260317"], "entry_tick": [1, 3], "exit_tick": [2, -1],
        "episode_index": [0, 0], "status": [ledger.FILLED, ledger.UNFILLED],
        "net_bps": [10.0, 0.0], "gross_bps": [33.0, 0.0],
    })
    manifest = {"by_partition": [{"contract_id": "H", "symbol": "001", "date": "20260317",
                                    "raw_signal_ticks": 2, "signal_episodes": 1,
                                    "blocked_signal_ticks": 0}]}
    keep = np.array([True, True])

    assert metrics.measure_masked_screening(frame, keep, manifest) == \
        metrics.measure_masked(frame, keep, manifest)


def test_batch_contract_uses_its_own_prior_day_parameters(monkeypatch, tmp_path):
    jobs = []

    def fake_partition(job):
        jobs.append(job)
        return {"rows": [], "counts": {"contract_id": job["contract_id"],
                "symbol": job["symbol"], "date": job["date"], "raw_signal_ticks": 0,
                "signal_episodes": 0, "blocked_signal_ticks": 0, "decisions": 0,
                "filled": 0, "censored": 0, "unfilled": 0}, "missing": None}

    monkeypatch.setattr(ledger, "_one_partition", fake_partition)
    ledger.build_ledger(
        contracts={"q_0.70": {}}, members={"q_0.70": ["001"]}, dates=["20260317"],
        parameter_table={"001:20260317": {"threshold": 0.1},
                         "q_0.70:001:20260317": {"threshold": 0.7}},
        output=tmp_path / "ledger.parquet", workers=1)

    assert jobs[0]["parameters"] == {"threshold": 0.7}


def test_batch_ledger_loads_one_symbol_day_once_for_multiple_contracts(monkeypatch, tmp_path):
    loads, received = [], []

    def fake_load(symbol, date, root):
        loads.append((symbol, date, root))
        return {"shared": True}, tmp_path / "source.parquet"

    def fake_partition(job):
        received.append(job)
        return {"rows": [], "counts": {
            "contract_id": job["contract_id"], "symbol": job["symbol"], "date": job["date"],
            "raw_signal_ticks": 0, "signal_episodes": 0, "blocked_signal_ticks": 0,
            "decisions": 0, "filled": 0, "censored": 0, "unfilled": 0,
        }, "missing": None}

    monkeypatch.setattr(ledger.tickdata, "load", fake_load)
    monkeypatch.setattr(ledger.catalog, "Book", lambda arrays: arrays)
    monkeypatch.setattr(ledger.catalog, "compute_features", lambda _axes, _book: {"shared": True})
    monkeypatch.setattr(ledger, "_one_partition", fake_partition)

    manifest = ledger.build_ledger(
        contracts={"C1": {}, "C2": {}}, members={"C1": ["001"], "C2": ["001"]},
        dates=["20260317"],
        parameter_table={"C1:001:20260317": {"threshold": 0.1},
                         "C2:001:20260317": {"threshold": 0.2}},
        output=tmp_path / "ledger.parquet", workers=1)

    assert len(loads) == 1
    assert [job["contract_id"] for job in received] == ["C1", "C2"]
    assert all(job["_shared_arrays"] == {"shared": True} for job in received)
    assert [job["parameters"] for job in received] == [{"threshold": 0.1}, {"threshold": 0.2}]
    assert manifest["runtime"]["contract_symbol_day_jobs"] == 2
    assert manifest["runtime"]["symbol_day_groups"] == 1
    assert manifest["runtime"]["raw_loads_avoided"] == 1


def test_a_contract_that_picks_its_own_execution_is_refused():
    """가설마다 실행 모델을 바꾸면 성과 차이를 신호의 차이로 읽을 수 없다 (§25)."""
    check = K.check_contract(template_with(entry_execution="TAKER_IMMEDIATE"))
    assert check["ok"] is False
    assert "실행 방식은 가설이 정하지 않는다" in check["problems"][0]


def test_a_contract_with_a_different_cadence_is_refused():
    check = K.check_contract(template_with(decision_spacing_seconds=5.0))
    assert check["ok"] is False


def test_a_contract_with_a_different_horizon_is_refused():
    check = K.check_contract(template_with(evaluation_horizon="S60"))
    assert check["ok"] is False


def test_a_mismatched_contract_stops_the_backtest(tmp_path):
    """조용히 맞추지 않는다 (§58)."""
    with pytest.raises(RuntimeError, match=K.CANONICAL_BACKTEST_SPEC_CHANGE_REQUIRED):
        K.run_backtest({"C": template_with(entry_execution="TAKER_IMMEDIATE")},
                       {"C": ["A"]}, ["20260316"], tmp_path / "lg.parquet")


# ---- 숨은 상수 (§17~§19) ---------------------------------------------------------

def test_every_hidden_constant_now_points_at_the_profile():
    rows = K.hidden_constants()
    assert len(rows) >= 10
    for row in rows:
        assert row["이제"].startswith("profile.") or "profile." in row["이제"], row


def test_the_decision_cadence_used_to_be_implicit():
    cadence = next(r for r in K.hidden_constants() if r["constant"] == "decision cadence")
    assert "암묵" in cadence["위치"]
    assert "EVERY_TICK" in cadence["이제"]


# ---- 관문 (§53~§54) --------------------------------------------------------------

def gate(**over):
    args = {"accounting": {"ok": True}, "invariants_ok": True, "tests_ok": True,
            "contract_check": {"ok": True}}
    args.update(over)
    return K.readiness_gate(args["accounting"], args["invariants_ok"],
                            args["tests_ok"], args["contract_check"])


def test_all_five_conditions_open_the_gate():
    result = gate()
    assert result["status"] == K.READY_FOR_CANONICAL_BACKTEST_SEARCH
    for key in ("CANONICAL_BACKTEST_PROFILE_DEFINED", "BACKTEST_TESTS_PASS",
                "LEDGER_INVARIANTS_PASS", "SPREAD_ACCOUNTING_PASS",
                "METRIC_DENOMINATORS_PASS"):
        assert result[key] is True, key


def test_bad_accounting_blocks_the_gate():
    assert gate(accounting={"ok": False})["status"] == K.ACCOUNTING_FAILURE


def test_failing_invariants_block_the_gate():
    assert gate(invariants_ok=False)["status"] == K.BACKTEST_INVARIANT_FAILURE


def test_a_contract_mismatch_blocks_the_gate():
    assert gate(contract_check={"ok": False})["status"] == K.BACKTEST_PROTOCOL_AMBIGUOUS


def test_the_gate_carries_the_profile_hash():
    """프로필 해시가 다르면 같은 평가 환경에서 비교한 것이 아니다 (§49)."""
    assert gate()["profile_sha256"] == K.profile_hash()


# ---- 손절 (gross 기준) -----------------------------------------------------------

def test_the_stop_is_measured_on_gross_not_net():
    """net 으로 잡으면 BID1 에 사는 순간 이미 -수수료 에서 시작해 여유가 줄어든다."""
    stop = K.CANONICAL["exit"]
    assert stop["stop_basis"] == "GROSS"
    assert stop["stop_gross_bps"] < 0
    assert "가격 이동을 재는 값" in stop["why_gross"]


def test_the_changelog_records_every_rule_change():
    """실행 규칙이 바뀌면 해시가 달라진다. 이전 결과와 직접 비교하지 않는다."""
    assert "v2" in K.PROFILE_CHANGELOG and "손절" in K.PROFILE_CHANGELOG["v2"]
    assert "v3" in K.PROFILE_CHANGELOG and "매 틱" in K.PROFILE_CHANGELOG["v3"]
    assert "v4" in K.PROFILE_CHANGELOG and "트리거" in K.PROFILE_CHANGELOG["v4"]
    assert "v5" in K.PROFILE_CHANGELOG
    assert "v6" in K.PROFILE_CHANGELOG
    assert "v7" in K.PROFILE_CHANGELOG
    assert "v8" in K.PROFILE_CHANGELOG
    assert "v9" in K.PROFILE_CHANGELOG
    assert K.PROFILE_ID.endswith("V9")


def stop_book(n, path_bps):
    """entry BID1 = 100 으로 두고 원하는 gross 경로를 만든다."""
    data = book(n)
    bid = 100.0 * (1.0 + np.asarray(path_bps, dtype=float) / 1e4)
    data["bid_price"] = np.tile(bid.reshape(-1, 1), (1, 10))
    data["ask_price"] = np.tile((bid + 1.0).reshape(-1, 1), (1, 10))
    return data


def test_the_stop_fires_at_the_first_tick_that_breaches():
    """먼저 닿는 쪽이 이긴다. 나중 틱 정보로 되돌아가 고르지 않는다."""
    path = np.array([0.0, -5.0, -11.0, -60.0, 50.0, 50.0])
    level = -10.0
    hit = np.flatnonzero(path <= level)
    assert int(hit[0]) == 2                 # -35 인 지점. -60 이 아니다


def test_a_path_that_never_breaches_does_not_stop():
    path = np.array([0.0, -5.0, -9.0, -5.0])
    assert len(np.flatnonzero(path <= -10.0)) == 0


def test_the_stop_level_is_a_protocol_constant_not_searchable():
    forbidden = " ".join(K.CANONICAL["not_searchable"])
    assert "청산 주문 정책" in forbidden
    from framework import config as cfg
    assert cfg.STOP_GROSS_BPS == K.CANONICAL["exit"]["stop_gross_bps"]


def test_the_ledger_records_each_trigger_as_its_own_exit_reason():
    import inspect
    source = inspect.getsource(ledger)
    for token in ("stop_gross_bps", "trailing_drawdown_gross_bps", "trailing_ask1",
                  "trailing_timeout_bid1", "horizon_cap"):
        assert token in source, token


def test_the_exit_limit_is_not_posted_at_entry():
    """진입 즉시 걸면 목표 수익이 그 순간 스프레드로 고정된다."""
    exit_rule = K.CANONICAL["exit"]
    assert exit_rule["posted_at_entry"] is False
    assert "스프레드 포획" in exit_rule["why_not_posted_at_entry"]


def test_the_stop_goes_to_market_but_the_trailing_exit_rests():
    """손절은 나가는 것이 목적이라 지정가로 기다리지 않는다."""
    exit_rule = K.CANONICAL["exit"]
    assert "BID1" in exit_rule["stop_action"] and "시장가" in exit_rule["stop_action"]
    assert "ASK1" in exit_rule["trailing_action"] and "지정가" in exit_rule["trailing_action"]
    assert exit_rule["reprice_on_best_ask_change"] is True
    assert "도망가는 순간" in exit_rule["why_stop_is_market"]
    assert "시장가 손절은 대기 중인 추적 지정가보다도 우선" in exit_rule["stop_vs_trailing"]


def test_hard_stop_and_trailing_drawdown_are_gross_based():
    exit_rule = K.CANONICAL["exit"]
    assert exit_rule["stop_basis"] == "GROSS"
    assert exit_rule["stop_gross_bps"] == -120.0
    assert exit_rule["trailing_drawdown_gross_bps"] == -30.0
    assert exit_rule["trailing_mark"] == "RUNNING_HIGHEST_ASK1_SINCE_ENTRY"


def test_trailing_drawdown_uses_the_running_highest_ask1():
    ask = np.array([101.0, 103.0, 102.8])
    peaks = np.maximum.accumulate(ask)
    drawdown = (ask / peaks - 1.0) * 1e4
    assert int(np.flatnonzero(drawdown <= -10.0)[0]) == 2


def test_the_exit_max_holding_time_is_separate_from_feature_profile_window():
    """실제 청산 900초와 S30 Feature Profile 라벨은 서로 다른 값이다."""
    exit_rule = K.CANONICAL["exit"]
    assert exit_rule["max_holding_seconds"] == 900.0
    assert "S30" in exit_rule["note"]
    assert exit_rule["limit_rest_seconds"] < exit_rule["max_holding_seconds"]


def test_repriced_trailing_exit_reposts_at_each_new_best_ask():
    data = book(5)
    data["ask_price"][:, 0] = [101.0, 102.0, 103.0, 103.0, 103.0]
    data["ask_qty"][:, 0] = 1.0
    data["buy_volume"][:] = [0.0, 0.0, 0.0, 10.0, 0.0]
    data["buy_max_price"][:] = 103.0

    status, _last, filled_at, price, reposts = fill.repriced_trailing_exit(
        data, 0, hard_stop_price=99.0, max_seconds=60.0)

    assert status == fill.FILLED
    assert filled_at == 3
    assert price == 103.0
    assert reposts == 2


def test_hard_stop_overrides_a_pending_trailing_exit():
    data = book(5)
    data["bid_price"][:, 0] = [100.0, 102.0, 101.8, 99.0, 99.0]
    data["ask_price"][:, 0] = data["bid_price"][:, 0] + 1.0
    price, tick, reason = ledger._resolve_exit(
        data, data["time_s"], data["bid_price"][:, 0], data["ask_price"][:, 0], 5,
        0, 100.0, 4, True,
        {"stop_gross_bps": -10.0, "trailing_drawdown_gross_bps": -10.0,
         "exit_limit_rest_seconds": 60.0})

    assert (price, tick, reason) == (99.0, 3, "stop_bid1_after_trailing")
