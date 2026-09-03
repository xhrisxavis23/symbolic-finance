"""The only research path searches entry conditions under one fixed execution profile."""

from __future__ import annotations

import argparse
import json

import pytest

from framework import canonical as K
from framework import contract
from framework.__main__ import parser
from framework.config import read_json
from framework.workflows import free_research as F
from framework.workflows import free_research_campaign as C


PREDICTION = {
    "bid_move_bps": "POSITIVE", "fill_slippage_bps": "NEUTRAL",
    "fill_rate": "LOW", "fill_spread_selection": "SAME",
    "fill_opportunity_selection": "MORE", "loss_kind": "ROSE_THEN_LOST",
}


def hypothesis(values=(0.3, 0.8), variants=1):
    """Agent 응답 모양. 가설 하나가 variant 마다 전략 하나로 펼쳐진다."""
    level = {
        "variant_id": "D1", "representation": "LEVEL", "claim_ids": ["C1"],
        "entry_expression": {
            "op": "compare",
            "input": {"op": "primitive", "primitive_id": "book_imbalance"},
            "comparator": ">",
            "value": "UNRESOLVED:imbalance_q",
        },
        "parameters": [{"name": "imbalance_q", "kind": "rolling_quantile",
                        "values": list(values)}],
    }
    persistence = {
        "variant_id": "D2", "representation": "PERSISTENCE", "claim_ids": ["C1"],
        "entry_expression": {
            "op": "persistence", "window": 10, "min_true": 6, "time_basis": "tick",
            "condition": level["entry_expression"],
        },
        "parameters": list(level["parameters"]),
    }
    return {
        "strategy_id": "KS001",
        "title": "imbalance continuation",
        "rationale": "own-knowledge proposal",
        "claims": [{"claim_id": "C1", "text": "displayed bid imbalance is unusually high",
                    "feature": "book_imbalance"}],
        "path_prediction": dict(PREDICTION),
        "variants": [level] if variants == 1 else [level, persistence],
    }


def strategy(values=(0.3, 0.8)):
    """펼쳐진 전략 하나. `validate_strategy` 를 직접 부르는 테스트용."""
    expanded, problems = F.expand_hypothesis(hypothesis(values))
    assert not problems, problems
    return expanded[0]


def request(source):
    return F.FreeResearchRequest(
        symbols=("001",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        strategy_source=source, workers=1, refinement_rounds=0)


def test_strategy_accepts_multidimensional_temporal_catalog_expression():
    raw = strategy()
    raw["entry_expression"] = {
        "op": "all",
        "args": [
            raw["entry_expression"],
            {"op": "persistence", "window": 5, "min_true": 3, "time_basis": "tick",
             "condition": {"op": "compare",
                           "input": {"op": "primitive", "primitive_id": "signed_aggr_flow_20"},
                           "comparator": ">", "value": "UNRESOLVED:flow_q"}},
        ],
    }
    raw["parameters"].append(
        {"name": "flow_q", "kind": "rolling_quantile", "values": [0.4, 0.7]})

    checked, problems = F.validate_strategy(raw)

    assert problems == []
    assert checked is not None
    assert {item["name"] for item in checked["parameters"]} == {"imbalance_q", "flow_q"}


def test_only_non_executable_or_future_unsafe_shapes_are_rejected():
    raw = strategy()
    raw["entry_expression"]["input"]["primitive_id"] = "future_best_ask"

    checked, problems = F.validate_strategy(raw)

    assert checked is None
    assert any("Catalog" in problem or "DSL" in problem for problem in problems)


def test_dates_must_be_disjoint():
    with pytest.raises(ValueError, match="겹친다"):
        F._validate_dates(F.FreeResearchRequest(
            symbols=("001",), discovery_dates=("20260316",),
            validation_dates=("20260316",), final_dates=("20260318",)))


def test_dates_must_be_chronological():
    with pytest.raises(ValueError, match="앞서야"):
        F._validate_dates(F.FreeResearchRequest(
            symbols=("001",), discovery_dates=("20260318",),
            validation_dates=("20260317",), final_dates=("20260319",)))


def test_dates_must_use_yyyymmdd():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        F._validate_dates(F.FreeResearchRequest(
            symbols=("001",), discovery_dates=("2026-03-16",),
            validation_dates=("20260317",), final_dates=("20260318",)))


def test_refinement_rounds_are_bounded_to_ten():
    configured = F.FreeResearchRequest(
        symbols=("001",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        refinement_rounds=11)

    with pytest.raises(ValueError, match="0..10"):
        F._validate_refinement(configured)


def test_cli_exposes_only_research_and_raw_cache():
    subparsers = next(action for action in parser()._actions
                      if isinstance(action, argparse._SubParsersAction))

    assert set(subparsers.choices) == {"research", "common-stock-cache"}
    with pytest.raises(SystemExit):
        parser().parse_args(["pre-validation"])


def test_execution_grid_is_rejected_because_canonical_execution_is_fixed():
    raw = strategy()
    raw["execution_grid"] = {"exit_model": ["BID1_MARKET"]}

    checked, problems = F.validate_strategy(raw)

    assert checked is None
    assert any("execution_grid" in problem for problem in problems)


def test_raw_inputs_and_structural_entry_parameters_are_searchable():
    raw = strategy()
    raw["entry_expression"] = {
        "op": "compare",
        "input": {
            "op": "rolling_mean",
            "input": {"op": "raw", "field": "bid_qty", "level": "UNRESOLVED:level"},
            "window": "UNRESOLVED:window",
            "time_basis": "tick",
        },
        "comparator": ">",
        "value": "UNRESOLVED:quantity_q",
    }
    raw["parameters"] = [
        {"name": "level", "kind": "literal", "values": [1, 10]},
        {"name": "window", "kind": "literal", "values": [5, 20]},
        {"name": "quantity_q", "kind": "rolling_quantile", "values": [0.5, 0.9]},
    ]

    checked, problems = F.validate_strategy(raw)

    assert problems == []
    assert checked is not None
    assert F._grid_size(checked) == 8
    first_parameters = {"level": 1, "window": 5, "quantity_q": 0.5}
    signal = F._contract(checked, first_parameters)["entry_program"]["signal"]
    assert signal["input"]["input"]["level"] == 1


def test_fractional_clock_lag_is_a_searchable_entry_parameter():
    raw = strategy()
    raw["entry_expression"]["input"] = {
        "op": "difference",
        "input": {"op": "raw", "field": "bid_price", "level": 1},
        "lag": "UNRESOLVED:clock_lag",
        "time_basis": "clock",
    }
    raw["parameters"].append(
        {"name": "clock_lag", "kind": "literal", "values": [0.25, 1.5]})

    checked, problems = F.validate_strategy(raw)

    assert problems == []
    assert checked is not None


def test_invalid_rolling_window_is_rejected_before_backtest():
    raw = strategy()
    raw["entry_expression"]["input"] = {
        "op": "rolling_mean",
        "input": {"op": "raw", "field": "bid_qty", "level": 1},
        "window": "UNRESOLVED:window",
        "time_basis": "tick",
    }
    raw["parameters"].append(
        {"name": "window", "kind": "literal", "values": [0]})

    checked, problems = F.validate_strategy(raw)

    assert checked is None
    assert any("window는 1 이상" in problem for problem in problems)


def test_canonical_passes_strategy_entry_rest_to_ledger(tmp_path, monkeypatch):
    captured = {}

    def fake_build_ledger(**kwargs):
        captured.update(kwargs)
        return {"errors": []}

    monkeypatch.setattr(K.ledger, "build_ledger", fake_build_ledger)
    profile = dict(K.CANONICAL)
    profile["entry"] = {**profile["entry"], "max_rest_seconds": 3.0, "max_rest_ticks": 17}

    template = {"entry_program": {"signal": {
        "op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
        "comparator": ">", "value": 0.0,
    }}}
    manifest = K.run_backtest(
        {"KS001": template}, {"KS001": ()}, (), tmp_path / "ledger.parquet",
        canonical=profile,
    )

    assert captured["entry_max_rest_seconds"] == 3.0
    assert captured["entry_max_rest_ticks"] == 17
    assert manifest["contract_sha256"]["KS001"] == contract.contract_hash(template)
    assert len(manifest["contracts_sha256"]) == 64


def test_discovery_freezes_best_net_and_validation_cannot_change_final(tmp_path):
    source = tmp_path / "strategies.json"
    source.write_text(json.dumps({"strategies": [hypothesis()]}), encoding="utf-8")
    calls = []

    def fake_replay(frozen, symbols, dates, output, workers, root):
        calls.append({"q": frozen["parameters"]["imbalance_q"],
                      "dates": tuple(dates), "frozen": frozen})
        q = float(frozen["parameters"]["imbalance_q"])
        if tuple(dates) == ("20260316",):
            net = q * 100.0
        elif tuple(dates) == ("20260317",):
            net = -999.0
        else:
            net = 7.0
        return {"state": "BACKTEST_COMPLETE",
                "metrics": {"net_bps_total": net, "scorable": 2}}

    result = F.run(request(source), tmp_path / "run", replay=fake_replay, root=tmp_path)

    assert result["state"] == "FINAL_COMPLETE"
    refinement = read_json(tmp_path / "run" / "03_refinement" / "refinement_artifact.json")
    assert refinement["payload"]["state"] == "REFINEMENT_SKIPPED"
    frozen = read_json(tmp_path / "run" / "04_frozen_strategies.json")
    assert frozen["payload"]["units"]["KS001__D1"]["parameters"]["imbalance_q"] == 0.8
    assert [call["dates"] for call in calls] == [
        ("20260316",), ("20260316",), ("20260317",), ("20260318",)]
    assert calls[-2]["frozen"] == calls[-1]["frozen"]
    final = read_json(tmp_path / "run" / "06_final" / "final_artifact.json")
    assert final["payload"]["validation_used_for_selection"] is False
    context = read_json(tmp_path / "run" / "00_research_context.json")
    assert context["payload"]["constraints"] == [
        "DECISION_TIME_CAUSAL_INPUTS_ONLY",
        "ACTUAL_BID_ASK_AND_EXPLICIT_23BPS",
        "DISJOINT_DISCOVERY_VALIDATION_FINAL_DATES",
        "REFINEMENT_USES_DISCOVERY_ONLY",
        "FULL_STRATEGY_FROZEN_BEFORE_VALIDATION",
        "LONG_ONLY",
        "CANONICAL_ENTRY_AND_EXIT_EXECUTION_FIXED",
    ]
    assert context["payload"]["search_space"] == "ENTRY_CONDITION_AND_ITS_PARAMETERS_ONLY"
    assert context["payload"]["optimizer"] == {
        "name": "OPTUNA",
        "trials_per_strategy": 50,
        "processes": 4,
        "seed": 1729,
        "objective": "DISCOVERY_NET_BPS_TOTAL",
    }
    assert len(context["payload"]["runtime_identity"]["framework_code_sha256"]) == 64
    assert context["payload"]["runtime_identity"]["dependencies"]["optuna"]
    discovery = read_json(tmp_path / "run" / "02_discovery" / "discovery_artifact.json")
    unit = discovery["payload"]["units"]["KS001__D1"]
    assert unit["optimizer"] == "OPTUNA"
    assert unit["sampler"] == "GridSampler"
    assert unit["completed_trials"] == 2
    assert (tmp_path / "run" / "02_discovery" / "units" / "KS001__D1" /
            "optuna_journal.log").exists()


def test_discovery_never_freezes_a_failed_replay_over_a_negative_complete_one(tmp_path):
    source = tmp_path / "strategies.json"
    source.write_text(json.dumps({"strategies": [hypothesis()]}), encoding="utf-8")

    def fake_replay(frozen, symbols, dates, output, workers, root):
        q = float(frozen["parameters"]["imbalance_q"])
        if tuple(dates) == ("20260316",) and q == 0.8:
            return {"state": "BACKTEST_ERROR", "errors": ["broken"]}
        return {"state": "BACKTEST_COMPLETE",
                "metrics": {"net_bps_total": -10.0, "scorable": 1}}

    F.run(request(source), tmp_path / "run", replay=fake_replay, root=tmp_path)

    frozen = read_json(tmp_path / "run" / "04_frozen_strategies.json")
    assert frozen["payload"]["units"]["KS001__D1"]["parameters"]["imbalance_q"] == 0.3


def test_optuna_enumerates_the_whole_grid_and_rejects_one_over_the_cap(tmp_path):
    def fake_replay(frozen, symbols, dates, output, workers, root):
        calls.append(tuple(dates))
        return {"state": "BACKTEST_COMPLETE", "metrics": {
            "net_bps_total": float(frozen["parameters"]["imbalance_q"]), "scorable": 1,
        }}

    def configured(source):
        return F.FreeResearchRequest(
            symbols=("001",), discovery_dates=("20260316",),
            validation_dates=("20260317",), final_dates=("20260318",),
            strategy_source=source, workers=1, trials_per_strategy=3,
            optuna_processes=2, optuna_seed=11, refinement_rounds=0)

    calls = []
    over = tmp_path / "over.json"
    over.write_text(json.dumps({"strategies": [hypothesis(tuple(i / 10 for i in range(9)))]}),
                    encoding="utf-8")
    F.run(configured(over), tmp_path / "over_run", replay=fake_replay, root=tmp_path)
    rejected = read_json(tmp_path / "over_run" / "01_knowledge_strategies.json")
    payload = rejected["payload"]
    assert payload["state"] == "NO_EXECUTABLE_STRATEGY"
    assert any("격자가 9칸" in problem
               for item in payload["rejected"] for problem in item["problems"])
    assert calls == []

    source = tmp_path / "strategies.json"
    source.write_text(json.dumps({"strategies": [hypothesis((0.3, 0.6, 0.9))]}),
                      encoding="utf-8")
    output = tmp_path / "run"
    F.run(configured(source), output, replay=fake_replay, root=tmp_path)

    snapshot = read_json(output / "02_discovery" / "units" / "KS001__D1" /
                         "optuna_study.json")
    assert snapshot["sampler"] == "GridSampler"
    assert snapshot["trial_budget"] == 3
    assert len(snapshot["trials"]) == 3
    # 전수라서 격자의 세 점이 정확히 한 번씩 나온다.
    assert sorted(trial["parameters"]["imbalance_q"]
                  for trial in snapshot["trials"]) == [0.3, 0.6, 0.9]
    assert len([dates for dates in calls if dates == ("20260316",)]) == 3

    calls.clear()
    F.run(configured(source), output, replay=fake_replay, root=tmp_path)
    assert ("20260316",) not in calls


def test_optuna_refuses_resume_with_a_different_discovery_scope(tmp_path):
    source = tmp_path / "strategies.json"
    source.write_text(json.dumps({"strategies": [hypothesis()]}), encoding="utf-8")

    def fake_replay(frozen, symbols, dates, output, workers, root):
        return {"state": "BACKTEST_COMPLETE", "metrics": {
            "net_bps_total": 1.0, "scorable": 1,
        }}

    output = tmp_path / "run"
    F.run(request(source), output, replay=fake_replay, root=tmp_path)
    changed = F.FreeResearchRequest(
        symbols=("002",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        strategy_source=source, workers=1, refinement_rounds=0)

    with pytest.raises(ValueError, match="discovery_scope_sha256"):
        F.run(changed, output, replay=fake_replay, root=tmp_path)


def test_refinement_uses_discovery_reoptimizes_and_freezes_before_validation(
        tmp_path, monkeypatch):
    source = tmp_path / "strategies.json"
    source.write_text(json.dumps({"strategies": [hypothesis()]}), encoding="utf-8")
    calls = []

    def fake_replay(frozen, symbols, dates, output, workers, root):
        calls.append({
            "strategy_id": frozen["strategy"]["strategy_id"],
            "dates": tuple(dates),
            "frozen": frozen,
        })
        q = float(frozen["parameters"]["imbalance_q"])
        return {"state": "BACKTEST_COMPLETE", "metrics": {
            "net_bps_total": q * 100.0,
            "scorable": 10,
        }}

    monkeypatch.setattr(
        F,
        "_refinement_loss_profile",
        lambda frozen, population, root: (
            {"decision_summary": {"rows": 10}, "population": population},
            {"population": population, "ledger": "fake.parquet",
             "ledger_sha256": "a" * 64, "rows": 10},
        ),
    )

    def fake_refinement_agent(*args, **kwargs):
        return {
            "state": "READY",
            "feedback": {"state": "FEEDBACK_READY"},
            "payload": {"state": "PROPOSED", "proposals": [{
                "hypothesis": "negative imbalance reversal",
                "mechanism": "loss-side imbalance reversal",
                "rival_explanation": "fewer decisions",
                "path_prediction": {
                    "bid_move_bps": "POSITIVE", "fill_slippage_bps": "NEUTRAL",
                    "fill_rate": "LOW", "fill_spread_selection": "SAME",
                    "fill_opportunity_selection": "MORE", "loss_kind": "ROSE_THEN_LOST"},
                "evidence_features": ["book_imbalance"],
                "entry_expression": {
                    "op": "compare",
                    "input": {"op": "primitive", "primitive_id": "book_imbalance"},
                    "comparator": "<",
                    "value": "UNRESOLVED:imbalance_q",
                },
            }]},
            "validation": {"feedback": {"ok": True}, "hypotheses": {"ok": True}},
            "agent_invocations": [],
            "config": {"max_proposals": 1},
        }

    monkeypatch.setattr(F.refinement_agent, "run", fake_refinement_agent)
    configured = F.FreeResearchRequest(
        symbols=("001",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        strategy_source=source, workers=1, refinement_rounds=1,
        refinement_max_proposals=1,
    )

    result = F.run(configured, tmp_path / "run", replay=fake_replay, root=tmp_path)

    assert result["state"] == "FINAL_COMPLETE"
    refinement = read_json(
        tmp_path / "run" / "03_refinement" / "refinement_artifact.json")
    assert refinement["kind"] == "refined_strategy_set"
    assert refinement["payload"]["state"] == "REFINEMENT_COMPLETE"
    assert refinement["payload"]["refined_candidate_count"] == 2
    assert refinement["payload"]["parent_retained_count"] == 1
    assert set(refinement["payload"]["units"]) == {
        "KS001__D1", "KS001__D1__R01__AD01", "KS001__D1__R01__FO01"}
    frozen = read_json(tmp_path / "run" / "04_frozen_strategies.json")
    assert frozen["parents"]["refined_strategy_set"] == refinement["artifact_id"]
    validation_ids = {
        call["strategy_id"] for call in calls if call["dates"] == ("20260317",)}
    final_ids = {
        call["strategy_id"] for call in calls if call["dates"] == ("20260318",)}
    assert validation_ids == final_ids == set(refinement["payload"]["units"])
    assert all(call["dates"] == ("20260316",) for call in calls[:6])
    final = read_json(tmp_path / "run" / "06_final" / "final_artifact.json")
    assert final["payload"]["validation_used_for_selection"] is False


def test_cli_accepts_optuna_settings():
    args = parser().parse_args([
        "research", "--output", "run", "--symbol", "005930",
        "--discovery-date", "20260828", "--validation-date", "20260829",
        "--final-date", "20260831", "--trials-per-strategy", "12",
        "--optuna-processes", "3", "--optuna-seed", "77",
        "--refinement-rounds", "2", "--refinement-max-proposals", "4",
        "--refinement-population", "FILLED_ONLY",
    ])

    assert args.trials_per_strategy == 12
    assert args.optuna_processes == 3
    assert args.optuna_seed == 77
    assert args.refinement_rounds == 2
    assert args.refinement_max_proposals == 4
    assert args.refinement_population == ["FILLED_ONLY"]


def test_saved_agent_response_is_bound_to_the_generation_request(tmp_path):
    class FakeRunner:
        def run(self, **kwargs):
            return {"strategies": [hypothesis()]}

    configured = F.FreeResearchRequest(
        symbols=("001",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        strategy_count=1)
    F._load_or_generate(configured, tmp_path, FakeRunner())
    changed = F.FreeResearchRequest(
        symbols=("001",), discovery_dates=("20260316",),
        validation_dates=("20260317",), final_dates=("20260318",),
        strategy_count=2)

    with pytest.raises(ValueError, match="prompt/model identity"):
        F._load_or_generate(changed, tmp_path, FakeRunner())


def test_campaign_reads_the_canonical_fills_metric():
    measured = C._metrics({"state": "BACKTEST_COMPLETE", "metrics": {
        "scorable": 9, "fills": 4, "fill_rate": 4 / 9,
    }})

    assert measured["decisions"] == 9
    assert measured["fills"] == 4
