"""Validation 이후 시스템 연결의 작은 계약 검사."""

from __future__ import annotations

import pytest
import pandas as pd

from framework import profit_target as PT, validation as V
from framework.contracts.artifacts import create_artifact, write_artifact
from framework.modules import backtest, validation
from framework.modules.schedule import ResearchSchedule
from framework.stages import backtest_stage, validation_stage
from framework.tests import test_grounding as grounding_fixture
from framework.tests import test_hypothesis as hypothesis_fixture
from framework.workflows import research as research_workflow


def schedule() -> ResearchSchedule:
    return ResearchSchedule(
        validation_dates=("20260323",),
        search_fit_dates=("20260330",),
        search_confirm_dates=("20260401",),
        backtest_dates=("20260407",),
        terminal_oos_dates=("20260420",),
    )


def test_schedule_keeps_discovery_search_and_terminal_oos_separate():
    schedule().validate(("20260316", "20260317"))
    bad = ResearchSchedule(
        validation_dates=("20260323",),
        search_fit_dates=("20260323",),
        search_confirm_dates=("20260401",),
        backtest_dates=("20260407",),
        terminal_oos_dates=("20260420",),
    )
    with pytest.raises(ValueError, match="겹친다"):
        bad.validate(("20260316",))
    bad_final = ResearchSchedule(
        validation_dates=("20260323",),
        search_fit_dates=("20260330",),
        search_confirm_dates=("20260401",),
        backtest_dates=("20260323",),
        terminal_oos_dates=("20260420",),
    )
    with pytest.raises(ValueError, match="겹친다"):
        bad_final.validate(("20260316",))


def test_profit_target_uses_the_fill_conditional_gross_hurdle():
    result = PT.evaluate({"fills": 2, "gross_bps_total": 50.0, "net_bps_total": 4.0})
    assert result["accounting_identity_ok"] is True
    assert result["mean_gross_bps_per_fill"] == 25.0
    assert result["eligibility_condition_passed"] is True
    assert result["primary_condition_passed"] is True
    assert result["success"] is True


def test_single_day_discovery_search_can_share_only_the_discovery_day():
    single = ResearchSchedule(
        validation_dates=("20260323",),
        search_fit_dates=("20260316",),
        search_confirm_dates=(),
        backtest_dates=("20260407",),
        terminal_oos_dates=("20260420",),
        single_day_discovery_search=True,
    )
    single.validate(("20260316",))


def test_multiple_profile_days_can_search_only_inside_discovery():
    schedule = ResearchSchedule(
        validation_dates=("20260323", "20260324"),
        search_fit_dates=("20260316", "20260317", "20260319"),
        search_confirm_dates=(),
        backtest_dates=("20260326", "20260331"),
        terminal_oos_dates=(),
        search_on_discovery_dates=True,
    )
    schedule.validate(("20260316", "20260317", "20260319"))


def test_research_progress_keeps_stage_timing_history(tmp_path):
    research_workflow._write_progress(tmp_path, "ENTRY_REFINEMENT", {"implementation_set": "I1"})
    research_workflow._write_progress(tmp_path, "PARAMETER_SEARCH", {"entry_refinement_set": "R1"})
    progress = (tmp_path / research_workflow.PROGRESS_FILE).read_text()

    import json
    history = json.loads(progress)["stage_history"]
    assert history[0]["stage"] == "ENTRY_REFINEMENT"
    assert "finished_at" in history[0]
    assert history[1]["stage"] == "PARAMETER_SEARCH"
    assert "finished_at" not in history[1]


def test_validation_plan_target_is_the_runtime_target_not_a_hardcoded_hypothesis():
    target = grounding_fixture.claim(
        "C3", claim_type="PREDICTED_CONSEQUENCE", epistemic_status="TO_BE_TESTED",
        grounding_status="VALIDATION_TARGET", catalog_family=None, catalog_feature=None,
        validation_target="PATH_TARGET", validation_measure="future path",
        validation_reference="anchor")
    grounded = grounding_fixture.grounded([grounding_fixture.claim(), target])
    hypothesis = hypothesis_fixture.good_hypothesis("H1")
    plan_targets = [
        {"target_id": "VP-H1-001", "hypothesis_id": "H1", "claim_id": "C3",
         "target_type": "PATH_TARGET"},
        {"target_id": "VP-H1-002", "hypothesis_id": "H1", "claim_id": None,
         "target_type": "MATCHED_CONTROL"},
    ]
    contracts, deferred = validation._runtime_contracts(
        grounded["hypotheses"][0], hypothesis, plan_targets,
        V.ValidationConfig(trigger_feature="ofi_depth_5", context_feature="vol_flow"))

    assert len(contracts) == 1
    contract = next(iter(contracts.values()))
    assert contract["plan_target_id"] == "VP-H1-001"
    assert contract["plan_target_type"] == "PATH_TARGET"
    assert deferred == [{"target_id": "VP-H1-002", "hypothesis_id": "H1",
                         "claim_id": None, "target_type": "MATCHED_CONTROL",
                         "status": "NOT_EXECUTED_NO_GROUNDED_CLAIM"}]


def test_path_only_plan_routes_as_a_single_state_hypothesis():
    result = V.verdict(
        {"MV-H1-01": {"status": "SUPPORTED"}}, {"blocked": False},
        hypothesis_id="H1",
        target_contracts={"MV-H1-01": {
            "target_type": "PATH_TARGET", "tested": "ofi_depth_5"}},
        config=V.ValidationConfig(trigger_feature="ofi_depth_5"))
    assert result["relation_status"] == "SINGLE_STATE_SUPPORTED"
    assert result["hypothesis_status"] == "REDUCED_FORM_SUPPORTED"
    assert result["route"] == "DATA_CAPABILITY_OR_EXECUTABLE_SPEC"


def test_backtest_stage_does_not_claim_execution_without_parameter_lock(tmp_path):
    evidence_artifact = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence_artifact)
    validation_path = tmp_path / "validation.json"
    search_artifact = create_artifact("parameter_search_set", {
        "units": {"H1": {"state": "SEARCH_SELECTED", "lock": None}},
    }, parents={"evidence_package": evidence_artifact["artifact_id"]})
    validation_artifact = create_artifact("validation_backtest_set", {
        "units": {"H1": {"state": "NO_PARAMETER_LOCK"}},
    }, parents={"parameter_search_set": search_artifact["artifact_id"],
                "evidence_package": evidence_artifact["artifact_id"]})
    write_artifact(validation_path, validation_artifact)
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search_artifact)

    result = backtest_stage.run(search_path, validation_path, evidence_path, tmp_path / "backtest",
                                schedule=schedule())
    assert result["payload"]["state"] == "NO_VALIDATION_BACKTEST_SUPPORT"
    assert result["payload"]["executed_hypothesis_ids"] == []
    assert result["payload"]["units"]["H1"]["state"] == "NO_PARAMETER_LOCK"


def test_backtest_module_keeps_missing_lock_as_a_non_execution(tmp_path):
    result = backtest.run({"lock": None}, symbols=("001",), dates=("20260407",),
                          output=tmp_path / "backtest", root=tmp_path,
                          stage="FINAL_BACKTEST")
    assert result == {"state": "NO_PARAMETER_LOCK", "reason": "확인된 parameter lock이 없다"}


def test_batch_backtest_splits_a_shared_ledger_back_into_contract_artifacts(tmp_path, monkeypatch):
    calls = []
    calibration_grids = []

    def fake_run_backtest(contracts, members, dates, output, **kwargs):
        calls.append((dict(contracts), dict(members), tuple(dates), kwargs["workers"]))
        rows = []
        parts = []
        for contract_id in contracts:
            rows.append({"contract_id": contract_id, "symbol": "001", "date": dates[0],
                         "entry_tick": 1, "exit_tick": -1, "episode_index": 0,
                         "status": "UNFILLED", "net_bps": 0.0, "gross_bps": 0.0})
            parts.append({"contract_id": contract_id, "symbol": "001", "date": dates[0],
                          "raw_signal_ticks": 1, "signal_episodes": 1,
                          "blocked_signal_ticks": 0, "decisions": 1, "filled": 0,
                          "censored": 0, "unfilled": 1})
        pd.DataFrame(rows).to_parquet(output, index=False)
        return {
            "contracts": sorted(contracts), "rows": len(rows), "partitions": len(parts),
            "by_partition": parts, "totals": {
                "raw_signal_ticks": len(parts), "signal_episodes": len(parts),
                "blocked_signal_ticks": 0, "decisions": len(parts), "filled": 0,
                "censored": 0, "unfilled": len(parts)},
            "entry_execution": "limit_bid1_hold_through", "missing": [], "errors": [],
            "legacy_programs": {}, "runtime": {"raw_loads_avoided": 1},
        }

    monkeypatch.setattr(backtest.K, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(backtest, "execution_workers", lambda: 2)
    monkeypatch.setattr(backtest.K, "check_contract", lambda *_args, **_kwargs: {"ok": True, "problems": []})
    monkeypatch.setattr(backtest.S, "calibration_table",
                        lambda _symbols, _dates, config, _root: (
                            calibration_grids.append(config.grid) or pd.DataFrame()))
    monkeypatch.setattr(backtest.S, "parameter_values_for_table",
                        lambda *_args, **_kwargs: {"001:20260407": {"q_book_imbalance": 0.5}})
    monkeypatch.setattr(backtest.S, "trading_calendar", lambda *_args, **_kwargs: ("20260406",))

    def search_result(hypothesis_id, value):
        return {
            "lock": {"status": "PARAMETER_LOCKED", "parameter": "q_book_imbalance",
                     "value": value, "parameters": {"q_book_imbalance": value}},
            "parameterized_template": {"hypothesis_id": hypothesis_id,
                                       "profit_target": PT.canonical_target()},
            "amended_specification": {"profit_target": PT.canonical_target(),
                                      "search_boundary": {"allowed_parameters": [{
                "name": "q_book_imbalance", "feature": "book_imbalance",
                "threshold_source": {"kind": "prior_valid_day_symbol_quantile"}}]}},
        }

    results = backtest.run_batch(
        {"H1": search_result("H1", 0.5), "H2": search_result("H2", 0.6)}, symbols=("001",),
        dates=("20260407",), output=tmp_path / "validation", root=tmp_path,
        stage="VALIDATION_BACKTEST")

    assert len(calls) == 1
    assert calls[0][3] == 2
    assert calibration_grids == []  # rolling q는 V9 실행 중에만 푼다
    assert set(calls[0][0]) == {"H1", "H2"}
    assert set(results) == {"H1", "H2"}
    for hypothesis_id in ("H1", "H2"):
        assert (tmp_path / "validation" / "units" / hypothesis_id / "ledger.parquet").is_file()
        assert results[hypothesis_id]["metrics"]["scorable"] == 1


def test_validation_does_not_promote_a_v8_900_only_supported_exit(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    calls = []

    def fake_run(search_result, *, symbols, dates, output, root, stage,
                 canonical=None, comparison_of=None):
        profile = canonical or validation_stage.K.CANONICAL
        calls.append((stage, profile["profile_id"]))
        return {
            "state": "BACKTEST_COMPLETE",
            "metrics": {"scorable": 1,
                        "net_bps_per_decision": -99.0 if comparison_of is None else 1.0},
            "profit_target_evaluation": {
                "accounting_identity_ok": True,
                "success": comparison_of is not None,
            },
        }

    monkeypatch.setattr(validation_stage.backtest, "run", fake_run)
    result = validation_stage.run(search_path, evidence_path, tmp_path / "validation",
                                  schedule=schedule())
    unit = result["payload"]["units"]["H1"]

    assert calls == [
        ("VALIDATION_BACKTEST", "CANONICAL_QUEUE_V9"),
        ("VALIDATION_BACKTEST_COMPARISON", "CANONICAL_QUEUE_V8_900_COMPARISON"),
    ]
    assert unit["state"] == "VALIDATION_BACKTEST_NOT_SUPPORTED"
    assert unit["v9_validation_state"] == "VALIDATION_BACKTEST_NOT_SUPPORTED"
    assert unit["v8_900_validation_state"] == "VALIDATION_BACKTEST_SUPPORTED"
    assert unit["promotion_profile_ids"] == []
    assert unit["v8_900_role"] == "COMPARISON_ONLY_NO_PROMOTION_AUTHORITY"


def test_validation_keeps_v9_support_when_v8_900_does_not_support(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)

    def fake_run(search_result, *, symbols, dates, output, root, stage,
                 canonical=None, comparison_of=None):
        return {
            "state": "BACKTEST_COMPLETE",
            "metrics": {"scorable": 1,
                        "net_bps_per_decision": 1.0 if comparison_of is None else -99.0},
            "profit_target_evaluation": {
                "accounting_identity_ok": True,
                "success": comparison_of is None,
            },
        }

    monkeypatch.setattr(validation_stage.backtest, "run", fake_run)
    result = validation_stage.run(search_path, evidence_path, tmp_path / "validation",
                                  schedule=schedule())
    unit = result["payload"]["units"]["H1"]

    assert unit["state"] == "VALIDATION_BACKTEST_SUPPORTED"
    assert unit["v9_validation_state"] == "VALIDATION_BACKTEST_SUPPORTED"
    assert unit["v8_900_validation_state"] == "VALIDATION_BACKTEST_NOT_SUPPORTED"
    assert unit["promotion_profile_ids"] == ["CANONICAL_QUEUE_V9"]


def test_validation_batches_multiple_parameter_locks(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}},
                  "H2": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    calls = []

    def fake_run_batch(search_results, *, stage, canonical=None, comparison_of=None, **_kwargs):
        calls.append((stage, tuple(sorted(search_results)),
                      (canonical or validation_stage.K.CANONICAL)["profile_id"]))
        return {hypothesis_id: {"state": "BACKTEST_COMPLETE",
                                "metrics": {"scorable": 1, "net_bps_per_decision": 1.0},
                                "profit_target_evaluation": {
                                    "accounting_identity_ok": True, "success": True}}
                for hypothesis_id in search_results}

    monkeypatch.setattr(validation_stage.backtest, "run_batch", fake_run_batch)
    result = validation_stage.run(search_path, evidence_path, tmp_path / "validation",
                                  schedule=schedule())

    assert calls == [
        ("VALIDATION_BACKTEST", ("H1", "H2"), "CANONICAL_QUEUE_V9"),
        ("VALIDATION_BACKTEST_COMPARISON", ("H1", "H2"), "CANONICAL_QUEUE_V8_900_COMPARISON"),
    ]
    assert set(result["payload"]["supported_hypothesis_ids"]) == {"H1", "H2"}


def test_validation_rejects_profit_target_accounting_failure():
    summary = {
        "state": "BACKTEST_COMPLETE",
        "metrics": {"scorable": 10, "net_bps_per_decision": 5.0},
        "profit_target_evaluation": {
            "accounting_identity_ok": False,
            "success": False,
        },
    }

    assert (validation_stage._validation_result(summary)
            == "VALIDATION_BACKTEST_ACCOUNTING_FAILURE")


def test_final_backtest_uses_new_dates_after_supported_validation(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    validation = create_artifact("validation_backtest_set", {
        "units": {"H1": {
            "state": "VALIDATION_BACKTEST_SUPPORTED",
            "v9_validation_state": "VALIDATION_BACKTEST_SUPPORTED",
            "promotion_profile_ids": ["CANONICAL_QUEUE_V9"],
        }},
    }, parents={"parameter_search_set": search["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    validation_path = tmp_path / "validation.json"
    write_artifact(validation_path, validation)
    calls = []

    def fake_run(search_result, *, symbols, dates, output, root, stage,
                 canonical=None, comparison_of=None):
        calls.append({"symbols": symbols, "dates": dates, "stage": stage,
                      "profile": (canonical or backtest_stage.K.CANONICAL)["profile_id"],
                      "comparison_of": (comparison_of or {}).get("profile_id")})
        return {
            "state": "BACKTEST_COMPLETE",
            "metrics": {"scorable": 1},
            "profit_target_evaluation": {
                "accounting_identity_ok": True,
                "success": comparison_of is None,
            },
        }

    monkeypatch.setattr(backtest_stage.backtest, "run", fake_run)
    result = backtest_stage.run(search_path, validation_path, evidence_path,
                                tmp_path / "backtest", schedule=schedule())
    assert calls == [
        {"symbols": ("001",), "dates": ("20260407",), "stage": "FINAL_BACKTEST",
         "profile": "CANONICAL_QUEUE_V9", "comparison_of": None},
        {"symbols": ("001",), "dates": ("20260407",),
         "stage": "FINAL_BACKTEST_COMPARISON",
         "profile": "CANONICAL_QUEUE_V8_900_COMPARISON",
         "comparison_of": "CANONICAL_QUEUE_V9"},
    ]
    assert result["payload"]["state"] == "BACKTEST_COMPLETE"
    assert result["payload"]["execution_state"] == "BACKTEST_COMPLETE"
    assert result["payload"]["units"]["H1"]["final_profit_target_state"] == (
        "FINAL_PROFIT_TARGET_SUPPORTED")
    assert result["payload"]["final_profit_target_supported_hypothesis_ids"] == ["H1"]
    assert (result["payload"]["units"]["H1"]["validation_promotion_profile_ids"] ==
            ["CANONICAL_QUEUE_V9"])


def test_final_separates_execution_complete_from_profit_not_supported():
    summary = {
        "state": "BACKTEST_COMPLETE",
        "profit_target_evaluation": {
            "accounting_identity_ok": True,
            "success": False,
        },
    }

    assert (backtest_stage._final_profit_target_state(summary)
            == "FINAL_PROFIT_TARGET_NOT_SUPPORTED")


def test_final_marks_profit_target_accounting_failure():
    summary = {
        "state": "BACKTEST_COMPLETE",
        "profit_target_evaluation": {
            "accounting_identity_ok": False,
            "success": False,
        },
    }

    assert (backtest_stage._final_profit_target_state(summary)
            == "FINAL_PROFIT_TARGET_ACCOUNTING_FAILURE")


def test_final_backtest_rejects_v8_900_only_validation_support(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    validation = create_artifact("validation_backtest_set", {
        "units": {"H1": {
            "state": "VALIDATION_BACKTEST_SUPPORTED",
            "v9_validation_state": "VALIDATION_BACKTEST_NOT_SUPPORTED",
            "v8_900_validation_state": "VALIDATION_BACKTEST_SUPPORTED",
            "promotion_profile_ids": ["CANONICAL_QUEUE_V8_900_COMPARISON"],
        }},
    }, parents={"parameter_search_set": search["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    validation_path = tmp_path / "validation.json"
    write_artifact(validation_path, validation)
    monkeypatch.setattr(
        backtest_stage.backtest, "run",
        lambda *_args, **_kwargs: pytest.fail("V8-900-only support opened Final"))

    result = backtest_stage.run(search_path, validation_path, evidence_path,
                                tmp_path / "backtest", schedule=schedule())

    unit = result["payload"]["units"]["H1"]
    assert result["payload"]["state"] == "NO_VALIDATION_BACKTEST_SUPPORT"
    assert unit["final_backtest_opened"] is False


def test_final_backtest_batches_multiple_supported_locks(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}},
                  "H2": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    validation = create_artifact("validation_backtest_set", {
        "units": {"H1": {"state": "VALIDATION_BACKTEST_SUPPORTED",
                           "v9_validation_state": "VALIDATION_BACKTEST_SUPPORTED"},
                  "H2": {"state": "VALIDATION_BACKTEST_SUPPORTED",
                           "v9_validation_state": "VALIDATION_BACKTEST_SUPPORTED"}},
    }, parents={"parameter_search_set": search["artifact_id"],
                "evidence_package": evidence["artifact_id"]})
    validation_path = tmp_path / "validation.json"
    write_artifact(validation_path, validation)
    calls = []

    def fake_run_batch(search_results, *, stage, canonical=None, **_kwargs):
        calls.append((stage, tuple(sorted(search_results)),
                      (canonical or backtest_stage.K.CANONICAL)["profile_id"]))
        return {hypothesis_id: {"state": "BACKTEST_COMPLETE", "metrics": {"scorable": 1}}
                for hypothesis_id in search_results}

    monkeypatch.setattr(backtest_stage.backtest, "run_batch", fake_run_batch)
    result = backtest_stage.run(search_path, validation_path, evidence_path,
                                tmp_path / "backtest", schedule=schedule())

    assert calls == [
        ("FINAL_BACKTEST", ("H1", "H2"), "CANONICAL_QUEUE_V9"),
        ("FINAL_BACKTEST_COMPARISON", ("H1", "H2"), "CANONICAL_QUEUE_V8_900_COMPARISON"),
    ]
    assert set(result["payload"]["executed_hypothesis_ids"]) == {"H1", "H2"}


def test_validation_refinement_lock_goes_directly_to_final(tmp_path, monkeypatch):
    evidence = create_artifact("evidence_package", {
        "profile_dates": ["20260316"], "profile_symbols": ["001"],
    })
    evidence_path = tmp_path / "evidence.json"
    write_artifact(evidence_path, evidence)
    search = create_artifact("parameter_search_set", {
        "units": {"H1": {"lock": {"status": "PARAMETER_LOCKED"}}},
    }, parents={"evidence_package": evidence["artifact_id"]})
    search_path = tmp_path / "search.json"
    write_artifact(search_path, search)
    calls = []

    def fake_run(search_result, *, symbols, dates, output, root, stage,
                 canonical=None, comparison_of=None):
        calls.append(stage)
        return {"state": "BACKTEST_COMPLETE", "metrics": {"scorable": 1}}

    monkeypatch.setattr(backtest_stage.backtest, "run", fake_run)
    result = backtest_stage.run_from_validation_refinement(
        search_path, evidence_path, tmp_path / "backtest", schedule=schedule())

    assert calls == ["FINAL_BACKTEST", "FINAL_BACKTEST_COMPARISON"]
    assert result["payload"]["units"]["H1"]["final_backtest_policy"] == (
        "VALIDATION_REFINEMENT_PARAMETER_LOCKS")
