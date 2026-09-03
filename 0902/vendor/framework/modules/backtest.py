"""잠긴 파라미터와 parameterized contract를 정본 Backtest에 연결한다."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .. import calibration as S, canonical as K, ledger, profit_target as PT
from ..config import execution_workers, now_utc, write_json


def _contract_manifest(manifest: Mapping[str, Any], frame: pd.DataFrame,
                       contract_id: str) -> dict[str, Any]:
    """공유 원장을 계약 하나의 기존 backtest artifact 모양으로 잘라 낸다."""
    parts = [dict(part) for part in manifest.get("by_partition") or []
             if str(part.get("contract_id")) == str(contract_id)]
    totals = {
        key: sum(int(part.get(key, 0)) for part in parts)
        for key in ("raw_signal_ticks", "signal_episodes", "blocked_signal_ticks",
                    "decisions", "filled", "censored", "unfilled")
    }
    result = dict(manifest)
    result.update({
        "contracts": [str(contract_id)],
        "rows": int(len(frame)),
        "partitions": len(parts),
        "by_partition": parts,
        "totals": totals,
        "missing": [dict(item) for item in manifest.get("missing") or []
                    if str(item.get("contract_id")) == str(contract_id)],
        "errors": [dict(item) for item in manifest.get("errors") or []
                   if str(item.get("contract_id")) == str(contract_id)],
        "legacy_programs": {
            str(contract_id): dict((manifest.get("legacy_programs") or {}).get(contract_id) or {})
        } if str(contract_id) in (manifest.get("legacy_programs") or {}) else {},
    })
    runtime = dict(manifest.get("runtime") or {})
    runtime.update({"batch_contracts": len(manifest.get("contracts") or []),
                    "contract_id": str(contract_id)})
    result["runtime"] = runtime
    return result


def _backtest_summary(*, hypothesis_id: str, lock: Mapping[str, Any],
                      template: Mapping[str, Any], dates: Sequence[str], symbols: Sequence[str],
                      parameters: Mapping[str, Mapping[str, float]], manifest: Mapping[str, Any],
                      frame: pd.DataFrame, canonical: Mapping[str, Any], stage: str,
                      comparison_of: Mapping[str, Any] | None) -> dict[str, Any]:
    target = PT.require(template, "Backtest contract")
    accounting = K.spread_accounting_audit(frame, manifest, canonical)
    try:
        ledger.verify_invariants(frame, manifest)
        invariants_ok, invariant_error = True, None
    except ledger.InvariantViolation as error:
        invariants_ok, invariant_error = False, str(error)
    metrics = K.metric_report(frame, manifest)
    return {
        "schema": "system_backtest.v2", "created_at": now_utc(),
        "state": "BACKTEST_COMPLETE", "hypothesis_id": str(hypothesis_id),
        "stage": stage,
        "parameter_lock": {"parameter": lock["parameter"], "value": lock["value"],
                           "parameters": dict(lock.get("parameters") or {}),
                           "plan_sha256": lock.get("plan_sha256")},
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "profit_target_evaluation": PT.evaluate(metrics),
        "entry_guards": list(template.get("entry_guards") or []),
        "dates": list(dates), "symbols": list(symbols),
        "execution_profile": {"profile_id": canonical["profile_id"],
                              "profile_sha256": canonical["profile_sha256"],
                              "role": "COMPARISON_ONLY" if comparison_of is not None else "SELECTION"},
        **({"comparison_of_profile": {
            "profile_id": comparison_of["profile_id"],
            "profile_sha256": comparison_of["profile_sha256"],
        }} if comparison_of is not None else {}),
        "eligible_symbol_days": len(parameters), "contract_check": K.check_contract(template, canonical),
        "accounting": accounting, "invariants_ok": invariants_ok,
        "invariant_error": invariant_error, "metrics": metrics,
        "evidence_note": (
            "Parameter lock 뒤, Final Backtest와 같은 V9 selection profile로 검증한 날짜다."
            if stage == "VALIDATION_BACKTEST" else
            ("V9가 고른 entry·q를 V8-900 청산으로만 다시 재생한 비교 결과다."
             if comparison_of is not None else
             "Validation Backtest와 다른 날짜에서 같은 V9 selection profile로 실행한 결과다."))
    }


def run_batch(search_results: Mapping[str, Mapping[str, Any]], *, symbols: Sequence[str],
              dates: Sequence[str], output: Path, root: Path, stage: str,
              workers: int | None = None, canonical: Mapping[str, Any] | None = None,
              comparison_of: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """여러 잠긴 계약을 한 원장으로 재생하되, 결과 artifact는 계약별로 유지한다.

    계약 선택·직전 100 tick q·지표·불변식은 기존 `run()`과 같다. 묶는 것은 같은 날짜의 원본
    호가를 읽는 실행뿐이며, batch ledger는 다시 계약별 ledger와 manifest로 분리한다.
    """
    canonical = K.CANONICAL if canonical is None else canonical
    workers = execution_workers() if workers is None else int(workers)
    output = Path(output)
    dates = tuple(str(value) for value in dates)
    if not dates:
        raise ValueError("Backtest 날짜가 비었다")

    results: dict[str, dict[str, Any]] = {}
    prepared: list[dict[str, Any]] = []
    seen_contracts: set[str] = set()
    for result_key, search_result in search_results.items():
        key = str(result_key)
        lock = search_result.get("lock")
        if not lock or lock.get("status") not in S.PARAMETER_LOCK_STATUSES:
            results[key] = {"state": "NO_PARAMETER_LOCK", "reason": "확인된 parameter lock이 없다"}
            continue
        template = search_result.get("parameterized_template")
        if not isinstance(template, Mapping):
            raise ValueError("Parameter Search가 만든 contract template이 없다")
        specification = search_result.get("amended_specification") or {}
        PT.require(specification, "Backtest amended Specification")
        PT.require(template, "Backtest contract")
        config = dataclasses.replace(S.config_for(specification), grid=(float(lock["value"]),))
        locked_parameters = set((lock.get("parameters") or {
            lock.get("parameter"): lock.get("value")}).keys())
        if locked_parameters != {parameter for _feature, _direction, parameter in config.states}:
            raise ValueError("parameter lock과 parameterized contract의 이름이 다르다")
        check = K.check_contract(template, canonical)
        if not check["ok"]:
            raise RuntimeError("Backtest contract가 정본 실행 프로필과 다르다: "
                               + "; ".join(check["problems"]))
        contract_id = str(template["hypothesis_id"])
        if contract_id in seen_contracts:
            raise ValueError(f"batch Backtest 계약 ID가 중복된다: {contract_id}")
        seen_contracts.add(contract_id)
        prepared.append({"key": key, "contract_id": contract_id, "lock": lock,
                         "template": dict(template), "config": config})

    if not prepared:
        return results

    # q 후보들은 threshold grid만 다르고, 종목·날짜·feature 보정은 같다. 이 표를 q마다
    # 다시 만들면 같은 raw tick을 여러 번 읽는다. 같은 보정 정의는 grid를 합쳐 한 번만
    # 만들고, 각 계약에는 자기 q 열만 다시 꺼낸다.
    calibration_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in prepared:
        config = item["config"]
        if config.is_rolling_quantile:
            # rolling q의 실제 cut은 아래 V9 실행에서 각 tick 직전 100개로 계산된다.
            # 여기서 같은 array를 한 번 더 풀어 "가능 여부"만 확인하면 전종목 Search가
            # 거의 두 번 돈다. 모든 입력에 잠긴 q를 넘기고, feature/관측 부족은 실행
            # partition이 missing 또는 신호 없음으로 정확히 남긴다.
            values = {parameter: float(item["lock"]["value"])
                      for _feature, _direction, parameter in config.states}
            item["parameters"] = {
                f"{symbol}:{date}": dict(values)
                for date in dates for symbol in symbols
            }
            continue
        key = (config.parameter, config.feature, config.threshold_kind,
               config.min_calibration_obs, config.quantile_method,
               config.calibration_scope, config.calibration_reference,
               config.direction, config.states)
        calibration_groups.setdefault(key, []).append(item)
    for group in calibration_groups.values():
        base = group[0]["config"]
        grid = tuple(sorted({float(item["lock"]["value"]) for item in group}))
        table = S.calibration_table(symbols, dates, dataclasses.replace(base, grid=grid), root)
        for item in group:
            item["parameters"] = S.parameter_values_for_table(
                table, float(item["lock"]["value"]), item["config"])

    contracts = {item["contract_id"]: item["template"] for item in prepared}
    members = {item["contract_id"]: list(symbols) for item in prepared}
    parameter_table = {
        f"{item['contract_id']}:{symbol_day}": dict(values)
        for item in prepared for symbol_day, values in item["parameters"].items()
    }
    batch_output = output / "_canonical_batches" / stage / "ledger.parquet"
    batch_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = K.run_backtest(
        contracts, members, list(dates), batch_output, parameter_table=parameter_table,
        root=root, workers=workers, reference_dates=S.trading_calendar(root), canonical=canonical)
    batch_frame = pd.read_parquet(batch_output)

    for item in prepared:
        contract_id = item["contract_id"]
        frame = batch_frame.loc[batch_frame["contract_id"].eq(contract_id)].copy()
        unit_manifest = _contract_manifest(manifest, frame, contract_id)
        unit_output = output / "units" / item["key"]
        unit_output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(unit_output / "ledger.parquet", index=False)
        write_json(unit_output / "ledger_manifest.json", unit_manifest)
        pd.DataFrame(
            [{"symbol": symbol_date.split(":", 1)[0], "date": symbol_date.split(":", 1)[1],
              **values} for symbol_date, values in item["parameters"].items()]
        ).to_csv(unit_output / "backtest_calibration.csv", index=False)
        results[item["key"]] = _backtest_summary(
            hypothesis_id=contract_id, lock=item["lock"], template=item["template"],
            dates=dates, symbols=symbols, parameters=item["parameters"], manifest=unit_manifest,
            frame=frame, canonical=canonical, stage=stage, comparison_of=comparison_of)
        write_json(unit_output / "backtest_summary.json", results[item["key"]])
    return results


def run(search_result: Mapping[str, Any], *, symbols: Sequence[str],
        dates: Sequence[str], output: Path, root: Path, stage: str,
        workers: int | None = None, canonical: Mapping[str, Any] | None = None,
        comparison_of: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """lock이 없으면 Backtest를 만들지 않는다. 임의 fixture 값으로 대체하지 않는다."""
    canonical = K.CANONICAL if canonical is None else canonical
    workers = execution_workers() if workers is None else int(workers)
    lock = search_result.get("lock")
    if not lock or lock.get("status") not in S.PARAMETER_LOCK_STATUSES:
        return {"state": "NO_PARAMETER_LOCK", "reason": "확인된 parameter lock이 없다"}
    template = search_result.get("parameterized_template")
    if not isinstance(template, Mapping):
        raise ValueError("Parameter Search가 만든 contract template이 없다")
    spec = search_result.get("amended_specification") or {}
    target = PT.require(spec, "Backtest amended Specification")
    PT.require(template, "Backtest contract")
    config = S.config_for(spec)
    # 잠근 후보 하나를 바로 Backtest한다. q는 직전 100 tick에서 풀고, literal은 그대로 쓴다.
    # grid에 그 값이 없으면 threshold 표에 필요한 열이 생기지 않는다.
    config = dataclasses.replace(config, grid=(float(lock["value"]),))
    locked_parameters = set((lock.get("parameters") or {lock.get("parameter"): lock.get("value")}).keys())
    if locked_parameters != {parameter for _feature, _direction, parameter in config.states}:
        raise ValueError("parameter lock과 parameterized contract의 이름이 다르다")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dates = tuple(str(value) for value in dates)
    if not dates:
        raise ValueError("Backtest 날짜가 비었다")
    calibration = S.calibration_table(symbols, dates, config, root)
    parameters = S.parameter_values_for_table(calibration, float(lock["value"]), config)
    check = K.check_contract(template, canonical)
    if not check["ok"]:
        raise RuntimeError("Backtest contract가 정본 실행 프로필과 다르다: " + "; ".join(check["problems"]))
    hypothesis_id = str(template["hypothesis_id"])
    manifest = K.run_backtest({hypothesis_id: template}, {hypothesis_id: list(symbols)},
                              list(dates), output / "ledger.parquet",
                              parameter_table=parameters, root=root,
                              workers=workers,
                              # Legacy compatibility 값이다. 정본 rolling q/guard는 이 날짜 목록을
                              # 읽지 않고 각 tick 직전 100개 관측에서 풀린다.
                              reference_dates=S.trading_calendar(root), canonical=canonical)
    frame = pd.read_parquet(output / "ledger.parquet")
    accounting = K.spread_accounting_audit(frame, manifest, canonical)
    try:
        ledger.verify_invariants(frame, manifest)
        invariants_ok, invariant_error = True, None
    except ledger.InvariantViolation as error:
        invariants_ok, invariant_error = False, str(error)
    metrics = K.metric_report(frame, manifest)
    summary = {
        "schema": "system_backtest.v2", "created_at": now_utc(),
        "state": "BACKTEST_COMPLETE", "hypothesis_id": hypothesis_id,
        "stage": stage,
        "parameter_lock": {"parameter": lock["parameter"], "value": lock["value"],
                           "parameters": dict(lock.get("parameters") or {}),
                           "plan_sha256": lock.get("plan_sha256")},
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "profit_target_evaluation": PT.evaluate(metrics),
        "entry_guards": list(template.get("entry_guards") or []),
        "dates": list(dates), "symbols": list(symbols),
        "execution_profile": {"profile_id": canonical["profile_id"],
                              "profile_sha256": canonical["profile_sha256"],
                              "role": "COMPARISON_ONLY" if comparison_of is not None else "SELECTION"},
        **({"comparison_of_profile": {
            "profile_id": comparison_of["profile_id"],
            "profile_sha256": comparison_of["profile_sha256"],
        }} if comparison_of is not None else {}),
        "eligible_symbol_days": len(parameters), "contract_check": check,
        "accounting": accounting, "invariants_ok": invariants_ok,
        "invariant_error": invariant_error, "metrics": metrics,
        "evidence_note": (
            "Parameter lock 뒤, Final Backtest와 같은 V9 selection profile로 검증한 날짜다."
            if stage == "VALIDATION_BACKTEST" else
            ("V9가 고른 entry·q를 V8-900 청산으로만 다시 재생한 비교 결과다."
             if comparison_of is not None else
             "Validation Backtest와 다른 날짜에서 같은 V9 selection profile로 실행한 결과다."))
    }
    write_json(output / "backtest_summary.json", summary)
    calibration.to_csv(output / "backtest_calibration.csv", index=False)
    return summary
