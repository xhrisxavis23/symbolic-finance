"""고정 청산 장부 기반 진입 보완의 직접 계약 검사."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from framework import canonical as K, ledger, outcome, profit_target as PT
from framework.config import write_json
from framework.agents import refinement as refinement_agent
from framework.contracts.artifacts import create_artifact, write_artifact
from framework.modules import discovery_loss, parameter_search, refinement
from framework.modules.schedule import ResearchSchedule
from framework.stages import parameter_search_stage, refinement_stage
from framework.workflows import discovery_loop, research


def _metrics(value: float) -> refinement.metrics.Metrics:
    return refinement.metrics.Metrics(
        decisions=300, scorable=300, fills=300, censored=0, unfilled=0,
        blocked_ticks=0, signal_ticks=300, episodes=300,
        total_net_bps=value * 300, total_gross_bps=(value + 23.0) * 300,
        days=1, symbols=1,
    )


def test_raw_diagnostic_path_is_not_changed_by_actual_exit():
    """고정 exit가 일찍 팔아도 진입 보완은 30초 원시 BID1 경로를 본다."""
    n = 40
    data = {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.tile(np.linspace(100.0, 98.0, n)[:, None], (1, 10)),
        "ask_price": np.tile(np.linspace(101.0, 99.0, n)[:, None], (1, 10)),
    }
    # 실제 exit는 첫 틱의 높은 가격으로, raw 30초 경로 끝은 손실로 만든다.
    value = outcome.path_outcome(data, 0, entry_price=100.0, exit_price=101.0, exit_tick=1)
    assert value is not None
    assert value["cohort"] == outcome.NET_RECOVERY
    assert value["diagnostic_cohort"] == outcome.PERSISTENT_ADVERSE


def test_bootstrap_parameters_reads_the_search_quantile_column(monkeypatch, tmp_path):
    spec = {
        "semantic_invariants": {"canonical_feature": "book_imbalance", "direction": "HIGHER"},
        "signal_template": {"_parameter": {"threshold_source": {
            "kind": "prior_valid_day_symbol_quantile"}}},
    }
    monkeypatch.setattr(refinement.S, "calibration_table", lambda *args, **kwargs: pd.DataFrame([
        {"symbol": "001", "date": "20260317", "eligible": True, "q0.80": 0.25},
        {"symbol": "002", "date": "20260317", "eligible": False, "q0.80": np.nan},
    ]))

    parameters, details = refinement._bootstrap_parameters(
        spec, ("001", "002"), ("20260317",), tmp_path, 0.80)

    assert parameters == {"001:20260317": {"q_book_imbalance": 0.25}}
    assert details["eligible_symbol_days"] == 1


def test_shared_refinement_replays_use_the_canonical_backtest_api(monkeypatch, tmp_path):
    """공유 부모·후보 재생은 canonical API에 없는 옵션을 넘기지 않는다."""
    calls = []
    frame = pd.DataFrame({
        "contract_id": ["H1"], "symbol": ["001"], "date": ["20260317"],
        "status": [ledger.UNFILLED], "entry_tick": [0], "exit_tick": [-1],
        "net_bps": [0.0], "gross_bps": [0.0],
    })
    manifest = {"errors": [], "by_partition": [], "missing": []}

    def fake_backtest(*args, **kwargs):
        calls.append(kwargs)
        return manifest

    prepared = refinement.PreparedRefinement(
        amended={"hypothesis_id": "H1"}, parameters={"001:20260317": {}}, calibration={},
        parent_template={"hypothesis_id": "H1", "contract_sha256": "parent"},
        identity={"parameter_table_sha256": "parameters"},
    )
    monkeypatch.setattr(refinement.K, "check_contract", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(refinement.K, "run_backtest", fake_backtest)
    monkeypatch.setattr(refinement.S, "trading_calendar", lambda *_args: ("20260316", "20260317"))
    monkeypatch.setattr(refinement.pd, "read_parquet", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(refinement.ledger, "verify_invariants", lambda *_args, **_kwargs: None)

    refinement.run_bootstrap_batch(
        {"H1": prepared}, symbols=("001",), dates=("20260317",), output=tmp_path, root=tmp_path)
    refinement._run_exact_batch(
        (("candidate", {"hypothesis_id": "H1"}),), symbols=("001",), dates=("20260317",),
        parameters={"001:20260317": {}}, output=tmp_path / "exact", root=tmp_path)

    assert len(calls) == 2
    assert all("record_features" not in kwargs for kwargs in calls)


def test_zero_round_refinement_keeps_the_exit_and_original_entry(monkeypatch, tmp_path):
    calls = []
    frame = pd.DataFrame({
        "status": [ledger.FILLED] * 3,
        "cohort": [outcome.NET_RECOVERY] * 3,
        "diagnostic_cohort": [outcome.PERSISTENT_ADVERSE, outcome.PERSISTENT_ADVERSE,
                                outcome.NET_RECOVERY],
        "vol_flow": [0.2, 0.4, 0.6], "pq0.50:vol_flow": [0.5] * 3,
        "net_bps": [-1.0] * 3, "gross_bps": [22.0] * 3,
        "symbol": ["001"] * 3, "date": ["20260317"] * 3,
        "entry_tick": [0, 40, 80], "exit_tick": [30, 70, 110],
        "contract_id": ["H1"] * 3,
    })
    expression = {"feature": "vol_flow", "operator": "<=", "quantile": 0.50,
                  "threshold_policy": "prior_valid_day_per_symbol",
                  "decision_time": "same_tick", "uses_current_or_past_only": True}

    monkeypatch.setattr(refinement, "_bootstrap_parameters", lambda *args, **kwargs: (
        {"001:20260317": {"q_book_imbalance": 0.8}}, {"parameter": "q_book_imbalance"}))

    def fake_ledger(template, **kwargs):
        calls.append(template)
        # ADD_GUARD만 이기고, 같은 feature의 band·주 신호 교체는 이 fixture에서
        # 이기지 않게 해 정확 재생 승자가 하나인지 확인한다.
        measured = _metrics(1.0 if len(template.get("entry_guards") or []) == 1 else -1.0)
        return frame, {"errors": [], "missing": []}, measured

    monkeypatch.setattr(refinement, "_run_ledger", fake_ledger)
    monkeypatch.setattr(refinement.diagnose, "guard_candidates", lambda *args: [{
        "candidate_id": "vol_flow|<=|q0.50", "expression": expression,
        "metrics": _metrics(0.5),
    }])
    monkeypatch.setattr(refinement.diagnose, "choose_guard", lambda *args, **kwargs: {
        "decision_floor": 200, "shortlist": [{"candidate_id": "vol_flow|<=|q0.50",
        "expression": expression, "metrics": _metrics(0.5)}], "candidates": 1,
        "surviving": 1, "rejected": {}, "shortlist_ids": ["vol_flow|<=|q0.50"], "exact": False,
    })
    spec = {
            "hypothesis_id": "H1", "profit_target": PT.canonical_target(),
            "semantic_invariants": {"side": "LONG", "direction": "HIGHER"},
        "signal_template": {"op": "compare", "input": {"primitive_id": "book_imbalance"},
                            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance",
                            "_parameter": {"name": "theta_book_imbalance", "status": "SEARCHABLE",
                                           "meaning": "fixture"}},
        "decision_opportunity_policy": {"initial_binding": "EVERY_TICK", "initial_spacing_seconds": None},
        "entry_execution_semantics": {"fill_model": "BEST_QUOTE_QUEUE", "entry_reference": "BID1"},
        "exit_policy": {"evaluation_horizon": "S30"},
        "search_boundary": {"allowed_parameters": [{"name": "theta_book_imbalance", "feature": "book_imbalance",
                                                        "status": "SEARCHABLE", "meaning": "fixture"}],
                            "forbidden_parameters": []},
        "specification_sha256": "fixture",
    }
    result = refinement.run(spec, symbols=("001",), dates=("20260317",), output=tmp_path,
                            config=refinement.RefinementConfig(max_rounds=0))
    assert result["state"] == refinement.ENTRY_REFINEMENT_COMPLETE
    assert result["fixed_exit_profile"]["profile_id"] == K.PROFILE_ID
    assert result["entry_guards"] == []
    assert result["entry_expression"] is None
    assert calls == []


def test_zero_round_refinement_has_no_primary_override(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "status": [ledger.FILLED] * 3,
        "cohort": [outcome.NET_RECOVERY] * 3,
        "diagnostic_cohort": [outcome.PERSISTENT_ADVERSE, outcome.PERSISTENT_ADVERSE,
                                outcome.NET_RECOVERY],
        "vol_flow": [0.2, 0.4, 0.6], "pq0.50:vol_flow": [0.5] * 3,
        "net_bps": [-1.0] * 3, "gross_bps": [22.0] * 3,
        "symbol": ["001"] * 3, "date": ["20260317"] * 3,
        "entry_tick": [0, 40, 80], "exit_tick": [30, 70, 110],
        "contract_id": ["H1"] * 3,
    })
    expression = {"feature": "vol_flow", "operator": "<=", "quantile": 0.50,
                  "threshold_policy": "prior_valid_day_per_symbol",
                  "decision_time": "same_tick", "uses_current_or_past_only": True}
    monkeypatch.setattr(refinement, "_bootstrap_parameters", lambda *args, **kwargs: (
        {"001:20260317": {"q_book_imbalance": 0.8}}, {"parameter": "fixture"}))

    def fake_ledger(template, **kwargs):
        primary = template["entry_program"]["signal"]["input"]["primitive_id"]
        return frame, {"errors": [], "missing": []}, _metrics(2.0 if primary == "vol_flow" else -1.0)

    monkeypatch.setattr(refinement, "_run_ledger", fake_ledger)
    monkeypatch.setattr(refinement.diagnose, "guard_candidates", lambda *args: [{
        "candidate_id": "vol_flow|<=|q0.50", "expression": expression,
        "metrics": _metrics(0.5),
    }])
    monkeypatch.setattr(refinement.diagnose, "choose_guard", lambda *args, **kwargs: {
        "decision_floor": 200, "shortlist": [{"candidate_id": "vol_flow|<=|q0.50",
        "expression": expression, "metrics": _metrics(0.5)}], "candidates": 1,
        "surviving": 1, "rejected": {}, "shortlist_ids": ["vol_flow|<=|q0.50"], "exact": False,
    })
    spec = {
            "hypothesis_id": "H1", "profit_target": PT.canonical_target(),
            "semantic_invariants": {"side": "LONG", "direction": "HIGHER",
                                                            "canonical_feature": "book_imbalance"},
        "signal_template": {"op": "compare", "input": {"op": "primitive", "primitive_id": "book_imbalance"},
                            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance",
                            "_parameter": {"name": "theta_book_imbalance", "status": "SEARCHABLE",
                                           "meaning": "fixture"}},
        "decision_opportunity_policy": {"initial_binding": "EVERY_TICK", "initial_spacing_seconds": None},
        "entry_execution_semantics": {"fill_model": "BEST_QUOTE_QUEUE", "entry_reference": "BID1"},
        "exit_policy": {"evaluation_horizon": "S30"},
        "search_boundary": {"allowed_parameters": [{"name": "theta_book_imbalance", "feature": "book_imbalance",
                                                        "status": "SEARCHABLE", "meaning": "fixture"}],
                            "forbidden_parameters": []},
        "specification_sha256": "fixture",
    }
    result = refinement.run(spec, symbols=("001",), dates=("20260317",), output=tmp_path,
                            config=refinement.RefinementConfig(max_rounds=0))
    assert result["state"] == refinement.ENTRY_REFINEMENT_COMPLETE
    assert result["primary_override"] is None
    assert result["entry_guards"] == []
    assert result["refined_specification"] is None


def test_parameter_search_receives_the_refined_guards(monkeypatch, tmp_path):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    specification = create_artifact("executable_specification_set", {
        "units": {"H1": {"spec": {"hypothesis_id": "H1"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    implementation = create_artifact("implementation_set", {
        "units": {"H1": {"state": "READY_FOR_PARAMETER_SEARCH"}},
    }, parents={"executable_specification_set": specification["artifact_id"]})
    guard = {"feature": "vol_flow", "operator": "<=", "quantile": 0.5}
    entry_expression = {"op": "compare", "input": {"op": "raw", "field": "time_s"},
                        "comparator": ">", "value": 60.0}
    refined_specification = {"hypothesis_id": "H1", "specification_sha256": "refined-specification"}
    refined = create_artifact("entry_refinement_set", {
        "units": {"H1": {"entry_guards": [guard], "entry_expression": entry_expression,
                         "primary_override": {"feature": "vol_flow"},
                         "refined_specification": refined_specification}},
    }, parents={"executable_specification_set": specification["artifact_id"],
                "implementation_set": implementation["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    evidence_path, specification_path = tmp_path / "evidence.json", tmp_path / "specification.json"
    implementation_path, refined_path = tmp_path / "implementation.json", tmp_path / "refined.json"
    for path, artifact in ((evidence_path, evidence), (specification_path, specification),
                           (implementation_path, implementation), (refined_path, refined)):
        write_artifact(path, artifact)
    received = []
    monkeypatch.setattr(parameter_search_stage.parameter_search, "run", lambda spec, **kwargs: (
        received.append((spec, kwargs)) or {"state": "SEARCH_SELECTED"}))
    schedule = ResearchSchedule(
        validation_dates=("20260323",), search_fit_dates=("20260330",),
        search_confirm_dates=("20260401",), backtest_dates=("20260407",),
        terminal_oos_dates=("20260420",))
    result = parameter_search_stage.run(specification_path, implementation_path, refined_path,
                                        evidence_path, tmp_path / "search", schedule=schedule)
    assert received[0][0] == refined_specification
    assert received[0][1]["entry_guards"] == [guard]
    assert received[0][1]["entry_expression"] == entry_expression
    assert received[0][1]["entry_refinement"]["primary_override"] == {"feature": "vol_flow"}
    assert result["parents"]["entry_refinement_set"] == refined["artifact_id"]


def test_parameter_search_candidate_binds_the_full_refinement_expression():
    expression = {
        "op": "sequence",
        "setup": {"op": "compare", "input": {"op": "raw", "field": "buy_volume"},
                  "comparator": ">", "value": 10.0},
        "trigger": {"op": "compare", "input": {"op": "raw", "field": "sell_volume"},
                    "comparator": "<", "value": 5.0},
        "min_lag": 1, "max_lag": 23, "time_basis": "tick",
    }
    template = {
        "hypothesis_id": "H1",
        "entry_program": {"signal": {"op": "compare", "input": {
            "op": "primitive", "primitive_id": "book_imbalance"},
            "comparator": ">", "value": "UNRESOLVED:q_book_imbalance"},
            "warmup_ticks": 100},
        "parameter_interface": {"q_book_imbalance": {"threshold_source": {
            "kind": "rolling_prior_100_ticks_quantile"}}},
    }

    candidate = parameter_search._template_for_candidate(
        template, candidate_id="q_0.50", guards=(), entry_expression=expression)

    assert candidate["entry_program"]["signal"] == expression
    assert candidate["entry_program"]["warmup_ticks"] == 23
    assert candidate["parameter_interface"] == {}


def test_loss_ledger_agent_freezes_feedback_before_generating_from_catalog():
    expression = {
        "op": "compare",
        "input": {"op": "rolling_zscore",
                  "input": {"op": "primitive", "primitive_id": "vol_flow"},
                  "window": 37, "min_observations": 15, "time_basis": "tick"},
        "comparator": "<=", "value": -0.4,
    }

    class FakeRunner:
        def __init__(self):
            self.roles = []
            self.prompts = []

        def run(self, **kwargs):
            self.roles.append(kwargs["role"])
            self.prompts.append(kwargs["prompt"])
            if kwargs["role"] == "entry_refinement_loss_feedback":
                return {"state": "FEEDBACK_READY", "observation": "손실에서 매도 흐름이 컸다",
                        "loss_mechanism": "매도 압력이 이어졌다", "rival_explanation": "거래 감소",
                        "prediction": "총 순이익이 개선된다", "evidence_features": ["vol_flow"]}
            return {"state": "PROPOSED", "proposals": [{
                "hypothesis": "매도 흐름이 약할 때만 진입한다", "mechanism": "역선택을 줄인다",
                "rival_explanation": "거래 감소", "path_prediction": {"bid_move_bps": "POSITIVE",
                                    "fill_slippage_bps": "NEUTRAL",
                                    "fill_rate": "LOW",
                                    "fill_spread_selection": "SAME",
                                    "fill_opportunity_selection": "MORE",
                                    "loss_kind": "ROSE_THEN_LOST"},
                "evidence_features": ["vol_flow"],
                "entry_expression": expression,
            }]}

    catalog_context = {"features": [{"name": "vol_flow"}], "operators": []}
    runner = FakeRunner()
    result = refinement_agent.run(
        {"scope": "raw BID1 path", "entry_feature_contrasts": []},
        current_contract={"entry_expression": expression}, catalog_context=catalog_context,
        runner=runner)
    assert result["state"] == "READY"
    assert result["payload"]["proposals"][0]["entry_expression"] == expression
    assert runner.roles == ["entry_refinement_loss_feedback",
                            "entry_refinement_feedback_hypothesis"]
    assert all("# Injected Catalog" in prompt and '"vol_flow"' in prompt
               for prompt in runner.prompts)


def test_refinement_accepts_agent_expression_without_floor_or_parent_improvement(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "status": [ledger.FILLED] * 3,
        "cohort": [outcome.PERSISTENT_ADVERSE] * 3,
        "diagnostic_cohort": [outcome.PERSISTENT_ADVERSE] * 3,
        "vol_flow": [0.2, 0.4, 0.6], "pq0.50:vol_flow": [0.5] * 3,
        "net_bps": [-1.0] * 3, "gross_bps": [22.0] * 3,
        "symbol": ["001"] * 3, "date": ["20260317"] * 3,
        "entry_tick": [0, 40, 80], "exit_tick": [30, 70, 110],
        "contract_id": ["H1"] * 3,
    })
    expression = {
        "op": "persistence",
        "condition": {"op": "compare",
                      "input": {"op": "primitive", "primitive_id": "vol_flow"},
                      "comparator": "<", "value": 0.5},
        "window": 17, "min_true": 9, "time_basis": "tick",
    }
    monkeypatch.setattr(refinement, "_bootstrap_parameters", lambda *args, **kwargs: (
        {"001:20260317": {"q_book_imbalance": 0.8}}, {"parameter": "q_book_imbalance"}))
    monkeypatch.setattr(refinement, "_run_ledger", lambda *args, **kwargs: (
        frame, {"errors": [], "missing": []}, _metrics(-1.0)))
    small_negative = refinement.metrics.Metrics(
        decisions=50, scorable=50, fills=20, censored=0, unfilled=30,
        blocked_ticks=0, signal_ticks=50, episodes=50,
        total_net_bps=-600.0, total_gross_bps=-140.0, days=1, symbols=1)
    monkeypatch.setattr(refinement, "_run_exact_batch", lambda templates, **kwargs: {
        candidate_id: {"contract_id": "H1", "metrics": small_negative, "rows": 3}
        for candidate_id, _template in templates
    } | {"_manifest": {"missing": []}})

    class FakeRunner:
        def __init__(self):
            self.roles = []

        def run(self, **kwargs):
            self.roles.append(kwargs["role"])
            if kwargs["role"] == "entry_refinement_loss_feedback":
                return {"state": "FEEDBACK_READY", "observation": "손실에서 매도 흐름이 컸다",
                        "loss_mechanism": "매도 압력이 이어졌다", "rival_explanation": "거래 감소",
                        "prediction": "총 순이익이 개선된다", "evidence_features": ["vol_flow"]}
            return {"state": "PROPOSED", "proposals": [{
                "hypothesis": "매도 흐름이 약할 때만 진입한다", "mechanism": "역선택을 줄인다",
                "rival_explanation": "거래 감소", "path_prediction": {"bid_move_bps": "POSITIVE",
                                    "fill_slippage_bps": "NEUTRAL",
                                    "fill_rate": "LOW",
                                    "fill_spread_selection": "SAME",
                                    "fill_opportunity_selection": "MORE",
                                    "loss_kind": "ROSE_THEN_LOST"},
                "evidence_features": ["vol_flow"],
                "entry_expression": expression,
            }]}

    spec = {
            "hypothesis_id": "H1", "profit_target": PT.canonical_target(),
            "semantic_invariants": {"side": "LONG", "direction": "HIGHER"},
        "signal_template": {"op": "compare", "input": {"primitive_id": "book_imbalance"},
                            "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance",
                            "_parameter": {"name": "theta_book_imbalance", "status": "SEARCHABLE"}},
        "decision_opportunity_policy": {"initial_binding": "EVERY_TICK", "initial_spacing_seconds": None},
        "entry_execution_semantics": {"fill_model": "BEST_QUOTE_QUEUE", "entry_reference": "BID1"},
        "exit_policy": {"evaluation_horizon": "S30"},
        "search_boundary": {"allowed_parameters": [{"name": "theta_book_imbalance",
                                                        "feature": "book_imbalance", "status": "SEARCHABLE"}],
                            "forbidden_parameters": []},
        "specification_sha256": "fixture",
    }
    runner = FakeRunner()
    result = refinement.run(spec, symbols=("001",), dates=("20260317",), output=tmp_path,
                            config=refinement.RefinementConfig(max_rounds=1), runner=runner)

    assert runner.roles == ["entry_refinement_loss_feedback",
                            "entry_refinement_feedback_hypothesis"]
    assert result["entry_expression"] == expression
    assert result["entry_guards"] == []
    assert result["rounds"][0]["candidate_selection"]["decision_floor"] is None
    assert result["rounds"][0]["candidate_selection"]["parent_improvement_required"] is False
    assert result["rounds"][0]["candidate_selection"]["prebuilt_candidate_list"] is False
    assert result["rounds"][0]["catalog_injection"]["feature_count"] > 0
    assert result["rounds"][0]["agent_selected_candidate_ids"][0].startswith("feedback:")
    assert refinement.RefinementConfig().max_rounds == 10


def test_discovery_loss_uses_the_search_selected_q_refinement_branch(tmp_path):
    selected_path = tmp_path / "05_parameter_search" / "units" / "H1" / "selected_parameter.json"
    branch_path = (tmp_path / "04_entry_refinement" / "units" / "H1" / "branches" /
                   "q_0.70" / "entry_refinement.json")
    write_json(selected_path, {"status": "SEARCH_SELECTED", "selected_q": 0.70})
    write_json(branch_path, {"rounds": [{"round": 1}], "entry_guards": [{"feature": "ofi"}]})
    selected, branch_id = discovery_loss._selected_refinement_branch(
        tmp_path, "H1", {"branch_records": {"q_0.70": {"state": "ENTRY_REFINEMENT_COMPLETE"}}})

    assert branch_id == "q_0.70"
    assert selected["entry_guards"] == [{"feature": "ofi"}]


def test_root_research_continues_failure_into_discovery_refinement(monkeypatch, tmp_path):
    planned = []
    monkeypatch.setattr(discovery_loop, "plan", lambda request, output: (
        planned.append((request, output)) or {"artifact_id": "loop-plan"}))
    monkeypatch.setattr(discovery_loop, "run", lambda *args, **kwargs: {"state": "IDLE", "actions": []})
    result = research._run_discovery_refinement(
        SimpleNamespace(auto_discovery_refinement=True), tmp_path,
        {"payload": {"units": {"H1": {"rounds": [{"round": 1}]}}}},
        {"payload": {"units": {"H1": {"state": "NO_PARAMETER_LOCK"}}}}, root=tmp_path)

    assert planned and planned[0][0].source_hypothesis_id == "H1"
    assert result["units"][0]["state"] == "IDLE"
    assert (tmp_path / "07_validation_discovery_refinement" /
            "validation_discovery_refinement_artifact.json").is_file()


def test_root_research_marks_empty_discovery_parent_as_not_opened(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery_loop, "plan", lambda *args, **kwargs: {"artifact_id": "loop-plan"})

    def no_parent(*args, **kwargs):
        raise discovery_loss.DiscoveryParentLedgerUnavailable("전일 기준이 없어 decision이 없다")

    monkeypatch.setattr(discovery_loop, "run", no_parent)
    result = research._run_discovery_refinement(
        SimpleNamespace(auto_discovery_refinement=True), tmp_path,
        {"payload": {"units": {"H1": {"rounds": [{"round": 1}]}}}},
        {"payload": {"units": {"H1": {"state": "NO_PARAMETER_LOCK"}}}}, root=tmp_path)

    assert result["state"] == "COMPLETE"
    assert result["units"] == [{"hypothesis_id": "H1", "state": "NOT_OPENED",
                                "reason": "전일 기준이 없어 decision이 없다"}]


def test_loss_ledger_profile_includes_post_entry_feature_changes(monkeypatch, tmp_path):
    n = 40
    arrays = {
        "time_s": np.arange(n, dtype=float),
        "bid_price": np.tile(np.linspace(100.0, 101.0, n)[:, None], (1, 10)),
        "ask_price": np.tile(np.linspace(101.0, 102.0, n)[:, None], (1, 10)),
        "bid_qty": np.tile(np.arange(1, n + 1, dtype=float)[:, None], (1, 10)),
        "ask_qty": np.tile(np.arange(n, 0, -1, dtype=float)[:, None], (1, 10)),
    }
    monkeypatch.setattr(refinement.tickdata, "load", lambda *args, **kwargs: (arrays, tmp_path / "x.parquet"))
    frame = pd.DataFrame({
        "status": [ledger.FILLED, ledger.FILLED],
        "cohort": [outcome.PERSISTENT_ADVERSE, outcome.NET_RECOVERY],
        "symbol": ["001", "001"], "date": ["20260317", "20260317"],
        "entry_tick": [1, 3], "fill_tick": [1, 3],
        "max_favorable_gross_bps": [2.0, 10.0], "max_adverse_gross_bps": [-20.0, -2.0],
        "early_max_net_bps": [-4.0, 3.0], "book_imbalance": [-0.5, 0.5],
    })
    profile = refinement._loss_ledger_profile(
        frame, [{"candidate_id": "book", "expression": {"feature": "book_imbalance"}}],
        accepted_guards=(), primary_override=None,
        spec={"semantic_invariants": {"canonical_feature": "book_imbalance"}}, root=tmp_path)
    changes = profile["post_entry_feature_changes"]
    assert changes["paths_used"]
    assert any(item["feature"] == "book_imbalance" for item in changes["changes"])


def test_refinement_stage_passes_the_workflow_runner_to_loss_agent(monkeypatch, tmp_path):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    specifications = create_artifact("executable_specification_set", {
        "units": {"H1": {"spec": {"hypothesis_id": "H1"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    implementations = create_artifact("implementation_set", {
        "units": {"H1": {"state": "READY_FOR_PARAMETER_SEARCH"}},
    }, parents={"executable_specification_set": specifications["artifact_id"]})
    evidence_path, spec_path, implementation_path = (tmp_path / "evidence.json", tmp_path / "spec.json",
                                                       tmp_path / "implementation.json")
    for path, artifact in ((evidence_path, evidence), (spec_path, specifications),
                           (implementation_path, implementations)):
        write_artifact(path, artifact)
    received = []
    monkeypatch.setattr(refinement_stage.refinement, "run", lambda *args, **kwargs: (
        received.append(kwargs["runner"]) or {"state": "ENTRY_REFINEMENT_NO_CHANGE"}))
    runner = object()
    refinement_stage.run(spec_path, implementation_path, evidence_path, tmp_path / "refinement", runner=runner)
    assert received == [runner]
