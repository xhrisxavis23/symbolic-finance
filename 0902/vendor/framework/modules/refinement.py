"""고정 청산 손실 장부의 자연어 feedback에서 새 진입 가설을 만든다.

청산은 `canonical.CANONICAL` 하나로 고정한다. Agent는 손실 장부를 먼저 자연어로
해석하고, 다음 호출에서 주입된 모든 인과 feature·raw 입력·DSL로 완전한 진입식을
새로 만든다. 미리 만든 후보 목록은 주지 않는다. 같은 canonical Backtest로 정확
재생한 후보 중 23bp 포함 총 순이익이 가장 큰 식을 다음 round의 parent로 쓴다.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import (calibration as S, canonical as K, catalog, contract, data as tickdata,
                diagnose, implementation as I, ledger, metrics, outcome)
from ..agents import refinement as refinement_agent
from ..config import (ENTRY_REFINEMENT_POLICY_VERSION, QUANTILES, TICK_ROOT,
                      execution_workers, now_utc, read_json, sha256_json, write_json)


SCHEMA_VERSION = "entry_refinement.v6"
BOOTSTRAP_QUANTILE = 0.80
# 매 round마다 Agent가 완전한 진입식을 다시 만들고, 정확 재생 결과를 다음 parent로 쓴다.
MAX_ROUNDS = 10
SELECTION_TARGET = diagnose.FIXED_EXIT_NET_SELECTION
SELECTION_METRIC = "TOTAL_NET_BPS"

ENTRY_REFINEMENT_COMPLETE = "ENTRY_REFINEMENT_COMPLETE"
ENTRY_REFINEMENT_NO_CHANGE = "ENTRY_REFINEMENT_NO_CHANGE"
ENTRY_REFINEMENT_FIXED_EXIT = "ENTRY_REFINEMENT_FIXED_EXIT"
ENTRY_REFINEMENT_REFUTED = "ENTRY_REFINEMENT_REFUTED"
ENTRY_REFINEMENT_NO_DECISIONS = "ENTRY_REFINEMENT_NO_DECISIONS"
ENTRY_REFINEMENT_TIED_EXACT = "ENTRY_REFINEMENT_TIED_EXACT"
ENTRY_REFINEMENT_AGENT_INVALID_OUTPUT = "ENTRY_REFINEMENT_AGENT_INVALID_OUTPUT"
PROGRESS_FILE = "entry_refinement_progress.json"
BOOTSTRAP_BATCH_FILE = "bootstrap_parent_batch.json"
BOOTSTRAP_BATCH_PROGRESS_FILE = "bootstrap_parent_batch_progress.json"


@dataclass(frozen=True)
class RefinementConfig:
    """고정 실행 규칙 위에서 진입 보완에만 쓰는 값."""

    bootstrap_quantile: float = BOOTSTRAP_QUANTILE
    max_rounds: int = MAX_ROUNDS
    # branch를 명시하면 각 q의 Discovery 손실 장부에서 완전한 진입식을 따로 만든다.
    branch_quantiles: tuple[float, ...] = ()
    # shared-q Search는 q마다 다른 Discovery 손실 구조를 가질 수 있다. 이 값을 켜면
    # Specification의 고정 Search grid 전체를 branch로 푼다. 상태별 독립 q Cartesian
    # grid는 별도 surface라 여기서 묵시적으로 36개 branch를 만들지 않는다.
    auto_branch_search_quantiles: bool = False


def _config_payload(config: RefinementConfig) -> dict[str, Any]:
    """JSON artifact와 재개 identity가 같은 값 표현을 쓰게 한다."""
    payload = asdict(config)
    payload["branch_quantiles"] = list(config.branch_quantiles)
    return payload


def config_payload(config: RefinementConfig) -> dict[str, Any]:
    """Stage cache도 refinement 결과와 같은 설정 지문을 쓴다."""
    return _config_payload(config)


def resolve_config(spec: Mapping[str, Any], config: RefinementConfig) -> RefinementConfig:
    """한 Specification이 실제로 쓸 q별 refinement policy를 결정한다.

    직접 q 목록을 준 경우에는 그것이 의도다. 자동 모드에서는 shared-q Search의
    사전 고정 grid만 branch로 펼친다. independent-state q는 후보가 Cartesian surface라
    별도의 branch 설계 없이 자동으로 같은 guard를 복제하지 않는다.
    """
    if config.branch_quantiles or not config.auto_branch_search_quantiles:
        return config
    search_config = S.config_for(spec)
    if search_config.independent_state_quantiles:
        return config
    return replace(config, branch_quantiles=tuple(search_config.grid))


def branching_hypothesis_ids(specification_units: Mapping[str, Any],
                             config: RefinementConfig) -> tuple[str, ...]:
    """자동 q별 재평가가 필요한 실행 가능한 Specification id만 고른다."""
    return tuple(
        str(hypothesis_id)
        for hypothesis_id, unit in specification_units.items()
        if isinstance(unit, Mapping)
        and isinstance(unit.get("spec"), Mapping)
        and bool(resolve_config(unit["spec"], config).branch_quantiles)
    )


@dataclass(frozen=True)
class PreparedRefinement:
    """같은 Discovery 부모 장부를 후보 사이에 묶기 위한 고정 입력.

    여기에는 후보 신호·직전 100 tick 분위수·고정 exit가 모두 들어 있다. Stage가 이 값들을
    한 canonical 재생에 묶어도, 각 contract의 신호와 결과 행은 섞이지 않는다.
    """

    amended: dict[str, Any]
    parameters: dict[str, dict[str, float]]
    calibration: dict[str, Any]
    parent_template: dict[str, Any]
    identity: dict[str, Any]


@dataclass(frozen=True)
class BootstrapParent:
    """공유 canonical 부모 장부에서 한 hypothesis만 떼어 낸 입력."""

    contract_id: str
    contract_sha256: str
    frame: pd.DataFrame
    manifest: dict[str, Any]
    ledger_path: str
    cache_state: str
    # q branch batch에서는 ledger의 contract id가 q별로 고유해야 한다. 실제 entry
    # template이 같은 경우에도 이 hash로 q별 prior-100-tick parameter table을 확인한다.
    parameter_table_sha256: str = ""


def _progress_identity(amended: Mapping[str, Any], *, symbols: Sequence[str], dates: Sequence[str],
                       config: RefinementConfig) -> dict[str, Any]:
    """재개해도 같은 Discovery·고정 청산 계산임을 확인하는 최소 입력."""
    return {
        "hypothesis_id": str(amended["hypothesis_id"]),
        "amended_specification_sha256": str(amended["specification_sha256"]),
        "symbols": list(map(str, symbols)),
        "dates": list(map(str, dates)),
        "config": _config_payload(config),
        "entry_refinement_policy_version": ENTRY_REFINEMENT_POLICY_VERSION,
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
    }


def prepare(spec: Mapping[str, Any], *, symbols: Sequence[str], dates: Sequence[str],
            root: Path = TICK_ROOT, config: RefinementConfig = RefinementConfig()) -> PreparedRefinement:
    """후보 하나의 q0.80 부모 계약을 결정한다.

    ``run`` 밖으로 꺼낸 이유는 여러 후보가 같은 종목·날짜를 볼 때 부모 재생만 한
    batch로 묶기 위해서다. preparation 자체는 기존 ``run``이 하던 일과 같다.
    """
    symbols, dates = tuple(map(str, symbols)), tuple(map(str, dates))
    amended = S.amend_specification(spec, S.config_for(spec))
    parameters, calibration = _bootstrap_parameters(
        amended, symbols, dates, Path(root), config.bootstrap_quantile)
    parent_template = _template(amended, [])
    identity = {
        "hypothesis_id": str(amended["hypothesis_id"]),
        "source_specification_sha256": str(spec.get("specification_sha256") or ""),
        "amended_specification_sha256": str(amended["specification_sha256"]),
        "symbols": list(symbols),
        "dates": list(dates),
        "config": _config_payload(config),
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "parent_contract_sha256": str(parent_template["contract_sha256"]),
        "parameter_table_sha256": sha256_json(parameters),
    }
    return PreparedRefinement(
        amended=dict(amended), parameters={key: dict(value) for key, value in parameters.items()},
        calibration=dict(calibration), parent_template=dict(parent_template), identity=identity)


def _prepared_matches(prepared: PreparedRefinement, spec: Mapping[str, Any], *,
                      symbols: Sequence[str], dates: Sequence[str], config: RefinementConfig) -> bool:
    """공유 batch 입력을 다른 후보·날짜에 잘못 붙이지 않는다."""
    identity = dict(prepared.identity)
    expected = {
        "hypothesis_id": str(spec["hypothesis_id"]),
        "source_specification_sha256": str(spec.get("specification_sha256") or ""),
        "symbols": list(map(str, symbols)),
        "dates": list(map(str, dates)),
        "config": _config_payload(config),
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "amended_specification_sha256": str(prepared.amended.get("specification_sha256") or ""),
        "parent_contract_sha256": str(prepared.parent_template.get("contract_sha256") or ""),
        "parameter_table_sha256": sha256_json(prepared.parameters),
    }
    return identity == expected


def _load_progress(output: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """현재 입력과 정확히 같은 미완료 refinement만 재개한다."""
    path = Path(output) / PROGRESS_FILE
    if not path.is_file():
        return None
    saved = read_json(path)
    if not isinstance(saved, Mapping) or saved.get("identity") != dict(identity):
        return None
    if saved.get("state") != "IN_PROGRESS":
        return None
    rounds = saved.get("rounds")
    guards = saved.get("accepted_guards")
    if not isinstance(rounds, list) or not isinstance(guards, list):
        return None
    expression = saved.get("entry_expression")
    if expression is not None and not isinstance(expression, Mapping):
        return None
    return dict(saved)


def _completed_result(output: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """완료한 동일 refinement는 q-branch 재개 때 다시 replay하지 않는다."""
    path = Path(output) / "entry_refinement.json"
    if not path.is_file():
        return None
    try:
        saved = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(saved, Mapping) or saved.get("input_identity") != dict(identity):
        return None
    return dict(saved)


def _write_progress(output: Path, identity: Mapping[str, Any], *, rounds: Sequence[Mapping[str, Any]],
                    accepted_guards: Sequence[Mapping[str, Any]],
                    entry_expression: Mapping[str, Any] | None,
                    initial_scorable: int | None) -> None:
    """확정된 진입식까지만 원자적으로 남긴다. 끊긴 round만 다시 돈다."""
    write_json(Path(output) / PROGRESS_FILE, {
        "schema": "entry_refinement_progress.v1",
        "created_at": now_utc(),
        "state": "IN_PROGRESS",
        "identity": dict(identity),
        "rounds": [dict(round_record) for round_record in rounds],
        "accepted_guards": [dict(guard) for guard in accepted_guards],
        "entry_expression": copy.deepcopy(entry_expression),
        "initial_scorable": initial_scorable,
        "next_round": len(rounds) + 1,
        "execution_workers": execution_workers(),
    })


def _close_progress(output: Path, identity: Mapping[str, Any], *, rounds: Sequence[Mapping[str, Any]],
                    accepted_guards: Sequence[Mapping[str, Any]],
                    entry_expression: Mapping[str, Any] | None,
                    initial_scorable: int | None,
                    terminal_state: str) -> None:
    """성공적으로 끝난 checkpoint를 더 이상 재개 대상으로 보이지 않게 닫는다."""
    path = Path(output) / PROGRESS_FILE
    if not path.is_file():
        return
    write_json(path, {
        "schema": "entry_refinement_progress.v1",
        "created_at": now_utc(),
        "state": "COMPLETE",
        "terminal_state": str(terminal_state),
        "identity": dict(identity),
        "rounds": [dict(round_record) for round_record in rounds],
        "accepted_guards": [dict(guard) for guard in accepted_guards],
        "entry_expression": copy.deepcopy(entry_expression),
        "initial_scorable": initial_scorable,
        "next_round": None,
        "execution_workers": execution_workers(),
    })


def _expression_placeholders(expression: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX):
            values.add(value.removeprefix(catalog.UNRESOLVED_PREFIX))
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(expression)
    return values


def _expression_warmup(expression: Mapping[str, Any], *, rolling_quantile: bool) -> int:
    warmup = 100 if rolling_quantile else 0

    def visit(value: Any) -> None:
        nonlocal warmup
        if isinstance(value, Mapping):
            op = str(value.get("op") or "")
            if str(value.get("time_basis") or "") == "tick":
                if op == "difference":
                    warmup = max(warmup, int(value.get("lag") or 0))
                elif op in {"rolling_zscore", "rolling_sum", "rolling_mean", "rolling_std",
                            "rolling_min", "rolling_max", "persistence"}:
                    warmup = max(warmup, int(value.get("window") or 0))
                elif op in {"sequence", "sequence_once"}:
                    warmup = max(warmup, int(value.get("max_lag") or 0))
            if op == "crossover":
                warmup = max(warmup, 1)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(expression)
    return warmup


def _template(spec: Mapping[str, Any], guards: Sequence[Mapping[str, Any]] = (), *,
              entry_expression: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """q 값은 비워 두되, 선택된 완전한 진입식은 계약에 그대로 결속한다."""
    template = I.compile_contract(spec)
    template["entry_guards"] = [dict(guard) for guard in guards]
    if entry_expression is not None:
        expression = copy.deepcopy(dict(entry_expression))
        inferred = catalog.infer_expression_type(expression, allow_unresolved=True)
        if inferred.value_type not in {"boolean", "event"}:
            raise ValueError("Refinement entry_expression은 boolean/event여야 한다")
        placeholders = _expression_placeholders(expression)
        unknown = placeholders - set(template.get("parameter_interface") or {})
        if unknown:
            raise ValueError(f"실행값 없는 새 UNRESOLVED가 있다: {sorted(unknown)}")
        template["parameter_interface"] = {
            name: item for name, item in (template.get("parameter_interface") or {}).items()
            if name in placeholders
        }
        template["entry_program"] = {
            "signal": expression,
            "warmup_ticks": _expression_warmup(
                expression, rolling_quantile=contract.uses_rolling_quantiles(template)),
        }
    template["contract_sha256"] = contract.contract_hash(template)[:16]
    return template


def _bootstrap_parameters(spec: Mapping[str, Any], symbols: Sequence[str], dates: Sequence[str],
                          root: Path, quantile: float) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """주 신호의 q0.80은 시작점일 뿐, refinement가 탐색하는 값이 아니다."""
    base = S.config_for(spec)
    config = replace(base, grid=(float(quantile),))
    table = S.calibration_table(symbols, dates, config, root)
    columns = [S.threshold_column(float(quantile), config, parameter)
               for _feature, _direction, parameter in config.states]
    usable = table.loc[table["eligible"].astype(bool), ["symbol", "date", *columns]]
    parameters = {}
    for row in usable.to_dict(orient="records"):
        parameters[f"{row['symbol']}:{row['date']}"] = {
            parameter: float(row[S.threshold_column(float(quantile), config, parameter)])
            for feature, _direction, parameter in config.states
        }
    return parameters, {
        "parameter": config.parameter,
        "state_parameters": [parameter for _feature, _direction, parameter in config.states],
        "quantile": float(quantile),
        "eligible_symbol_days": int(len(parameters)),
        "calibration": table.to_dict(orient="records"),
    }


def _parent_ledger_columns() -> list[str]:
    """Entry Refinement가 실제로 읽는 원장 열만 고른다.

    원장은 audit 용 exit·경로 열도 함께 기록한다. 하지만 이 Stage는 고정 exit를
    바꾸지 않는다. feature 는 Catalog 전체를 읽는다 — Agent 가 진입식에 쓴 축이
    여기 없으면 손실 대비가 그 축을 조용히 건너뛴다. 임계는 guard 축에만 있다.
    """
    axes = catalog.list_features()
    guard = catalog.guard_axes()
    return [
        "contract_id", "symbol", "date", "status", "entry_tick", "exit_tick",
        "fill_tick", "entry_signal_active_at_fill", "entry_signal_persisted_until_fill",
        "entry_signal_active_share_until_fill",
        "episode_index", "net_bps", "gross_bps",
        "cohort", "diagnostic_cohort",
        *axes,
        *(ledger.threshold_column(feature, quantile)
          for feature in guard for quantile in QUANTILES),
    ]


def _run_ledger(template: Mapping[str, Any], *, symbols: Sequence[str], dates: Sequence[str],
                parameters: Mapping[str, Mapping[str, float]], output: Path,
                root: Path) -> tuple[pd.DataFrame, dict[str, Any], metrics.Metrics]:
    """모든 라운드가 정확히 같은 고정 청산 프로필을 쓴다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hypothesis_id = str(template["hypothesis_id"])
    check = K.check_contract(template)
    if not check["ok"]:
        raise RuntimeError("Entry refinement contract가 canonical Backtest와 다르다: "
                           + "; ".join(check["problems"]))
    manifest = K.run_backtest(
        {hypothesis_id: template}, {hypothesis_id: list(symbols)}, list(dates),
        output / "ledger.parquet", parameter_table=parameters, root=root,
        reference_dates=S.trading_calendar(root))
    if manifest.get("errors"):
        raise RuntimeError("Entry refinement 장부 실행 오류: " + str(manifest["errors"][:3]))
    frame = pd.read_parquet(output / "ledger.parquet", columns=_parent_ledger_columns())
    ledger.verify_invariants(frame, manifest)
    return frame, manifest, metrics.measure(frame, manifest)


def _bootstrap_batch_identity(prepared: Mapping[str, PreparedRefinement], *,
                              symbols: Sequence[str], dates: Sequence[str], root: Path) -> dict[str, Any]:
    """공유 parent ledger가 정확히 같은 signal·q·날짜에서 온 것인지 식별한다."""
    return {
        "schema": "entry_refinement_bootstrap_batch.v1",
        "prepared": {hypothesis_id: dict(value.identity)
                     for hypothesis_id, value in sorted(prepared.items())},
        "symbols": list(map(str, symbols)),
        "dates": list(map(str, dates)),
        "reference_dates": list(S.trading_calendar(root)),
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "entry_refinement_policy_version": ENTRY_REFINEMENT_POLICY_VERSION,
    }


def bootstrap_batch_cache_key(prepared: Mapping[str, PreparedRefinement], *,
                              symbols: Sequence[str], dates: Sequence[str], root: Path) -> str:
    """서로 다른 초기 parent 계약이 같은 batch 원장을 덮지 않게 하는 입력 지문."""
    identity = _bootstrap_batch_identity(prepared, symbols=symbols, dates=dates, root=Path(root))
    return sha256_json(identity)[:16]


def _manifest_for_contract(manifest: Mapping[str, Any], contract_id: str) -> dict[str, Any]:
    """한 후보의 screening 지표가 이웃 후보 partition을 세지 않게 만든다."""
    out = dict(manifest)
    out["contracts"] = [str(contract_id)]
    for key in ("by_partition", "missing", "errors"):
        values = list(manifest.get(key) or [])
        out[key] = [dict(value) for value in values
                    if str(value.get("contract_id") or "") == str(contract_id)]
    return out


def _read_bootstrap_batch(output: Path, identity: Mapping[str, Any],
                          prepared: Mapping[str, PreparedRefinement]) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """완료된 동일 batch만 재개한다. partial parquet은 다시 계산한다."""
    output = Path(output)
    metadata_path = output / BOOTSTRAP_BATCH_FILE
    ledger_path = output / "ledger.parquet"
    manifest_path = output / "ledger_manifest.json"
    if not (metadata_path.is_file() and ledger_path.is_file() and manifest_path.is_file()):
        return None
    metadata = read_json(metadata_path)
    if metadata.get("identity") != dict(identity):
        return None
    try:
        manifest = read_json(manifest_path)
        frame = pd.read_parquet(ledger_path)
        ledger.verify_invariants(frame, manifest)
    except Exception:
        return None
    actual = set(map(str, frame.get("contract_id", pd.Series(dtype=object)).unique()))
    if actual != set(map(str, prepared)):
        return None
    return frame, manifest


def run_bootstrap_batch(prepared: Mapping[str, PreparedRefinement], *,
                        symbols: Sequence[str], dates: Sequence[str], output: Path,
                        root: Path = TICK_ROOT) -> dict[str, BootstrapParent]:
    """여러 초기 부모 계약을 한 quote load로 실행한다.

    후보별 신호·큐·고정 exit replay는 각각 그대로 수행한다. 단지 불변인 같은
    ``(symbol, date)`` 호가 배열과 feature 배열을 worker 안에서 한 번 읽어 공유한다.
    """
    prepared = {str(hypothesis_id): value for hypothesis_id, value in prepared.items()}
    if not prepared:
        return {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    symbols, dates = tuple(map(str, symbols)), tuple(map(str, dates))
    identity = _bootstrap_batch_identity(prepared, symbols=symbols, dates=dates, root=Path(root))
    cached = _read_bootstrap_batch(output, identity, prepared)
    cache_state = "REUSED" if cached is not None else "CREATED"
    if cached is None:
        _write_bootstrap_batch_progress(
            output, identity, prepared, symbols=symbols, dates=dates,
            state="IN_PROGRESS", cache_state=None)
        contracts: dict[str, dict[str, Any]] = {}
        members: dict[str, list[str]] = {}
        parameter_table: dict[str, dict[str, float]] = {}
        for hypothesis_id, value in prepared.items():
            template = dict(value.parent_template)
            check = K.check_contract(template)
            if not check["ok"]:
                raise RuntimeError("Entry refinement contract가 canonical Backtest와 다르다: "
                                   + "; ".join(check["problems"]))
            contracts[hypothesis_id] = template
            members[hypothesis_id] = list(symbols)
            for symbol_day, parameters in value.parameters.items():
                parameter_table[f"{hypothesis_id}:{symbol_day}"] = dict(parameters)
        manifest = K.run_backtest(
            contracts, members, list(dates), output / "ledger.parquet",
            parameter_table=parameter_table, root=root,
            reference_dates=S.trading_calendar(root))
        if manifest.get("errors"):
            raise RuntimeError("Entry refinement bootstrap batch 장부 실행 오류: "
                               + str(manifest["errors"][:3]))
        frame = pd.read_parquet(output / "ledger.parquet")
        ledger.verify_invariants(frame, manifest)
        write_json(output / BOOTSTRAP_BATCH_FILE, {
            "schema": "entry_refinement_bootstrap_batch.v1",
            "created_at": now_utc(),
            "identity": identity,
            "ledger": str(output / "ledger.parquet"),
            "contracts": sorted(prepared),
        })
    else:
        frame, manifest = cached
    _write_bootstrap_batch_progress(
        output, identity, prepared, symbols=symbols, dates=dates,
        state="COMPLETE", cache_state=cache_state)
    result: dict[str, BootstrapParent] = {}
    for hypothesis_id, value in prepared.items():
        contract_id = str(hypothesis_id)
        result[hypothesis_id] = BootstrapParent(
            contract_id=contract_id,
            contract_sha256=str(value.parent_template["contract_sha256"]),
            frame=frame.loc[frame["contract_id"].eq(contract_id)].copy(),
            manifest=_manifest_for_contract(manifest, contract_id),
            ledger_path=str(output / "ledger.parquet"),
            cache_state=cache_state,
            parameter_table_sha256=str(value.identity["parameter_table_sha256"]),
        )
    return result


def _write_bootstrap_batch_progress(
        output: Path, identity: Mapping[str, Any], prepared: Mapping[str, PreparedRefinement], *,
        symbols: Sequence[str], dates: Sequence[str], state: str,
        cache_state: str | None) -> None:
    """공유 부모 replay가 계산 중인지 cache에서 끝났는지 남긴다."""
    write_json(Path(output) / BOOTSTRAP_BATCH_PROGRESS_FILE, {
        "schema": "entry_refinement_bootstrap_batch_progress.v1",
        "updated_at": now_utc(),
        "state": str(state),
        "phase": "CANONICAL_PARENT_REPLAY",
        "identity": dict(identity),
        "hypothesis_ids": sorted(map(str, prepared)),
        "symbols": list(map(str, symbols)),
        "dates": list(map(str, dates)),
        "cache_state": cache_state,
        "execution_workers": execution_workers(),
    })


def _run_exact_batch(
        templates: Sequence[tuple[str, Mapping[str, Any]]], *,
        symbols: Sequence[str], dates: Sequence[str],
        parameters: Mapping[str, Mapping[str, float]], output: Path, root: Path,
) -> dict[str, dict[str, Any]]:
    """한 refinement shortlist를 종목일당 quote 한 번으로 실제 재실행한다.

    후보마다 guard만 다르고 entry q, queue 체결, fixed exit는 같다. 따라서 quote 입력을
    공유해도 각각의 계약·원장 행은 분리된다. 이 단계의 판정에는 전 feature frame이
    아니라 contract별 status·net만 필요하므로 그것만 다시 읽는다.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contracts: dict[str, dict[str, Any]] = {}
    members: dict[str, list[str]] = {}
    parameter_table: dict[str, dict[str, float]] = {}
    candidate_contracts: dict[str, str] = {}
    for rank, (candidate_id, template) in enumerate(templates, start=1):
        check = K.check_contract(template)
        if not check["ok"]:
            raise RuntimeError("Entry refinement contract가 canonical Backtest와 다르다: "
                               + "; ".join(check["problems"]))
        contract_id = f"{template['hypothesis_id']}__REFINE_{rank:02d}"
        contracts[contract_id] = dict(template)
        members[contract_id] = list(symbols)
        candidate_contracts[str(candidate_id)] = contract_id
        for symbol_day, values in parameters.items():
            parameter_table[f"{contract_id}:{symbol_day}"] = dict(values)
    manifest = K.run_backtest(
        contracts, members, list(dates), output / "ledger.parquet",
        parameter_table=parameter_table, root=root,
        reference_dates=S.trading_calendar(root))
    if manifest.get("errors"):
        raise RuntimeError("Entry refinement batch 장부 실행 오류: " + str(manifest["errors"][:3]))
    frame = pd.read_parquet(output / "ledger.parquet",
                            columns=["contract_id", "symbol", "date", "status", "net_bps",
                                     "gross_bps"])
    return {
        candidate_id: {
            "contract_id": contract_id,
            "metrics": metrics.measure(frame, manifest, contract_id=contract_id),
            "rows": int(frame["contract_id"].eq(contract_id).sum()),
        }
        for candidate_id, contract_id in candidate_contracts.items()
    } | {"_manifest": manifest}


def _loss_ledger_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """고정 canonical exit 결정 장부를 entry guard 학습 표적으로 쓴다.

    여기서 쓰는 ``cohort``와 ``net_bps``는 모두 이미 고정된 canonical exit의 결과다.
    따라서 이 단계는 손실 체결과 0원 지정가 미체결을 피할 entry 상태만 고를 뿐,
    exit 조건·가격·시간을 바꾸거나 학습하지 않는다. ``diagnostic_cohort``는 이후
    가격 경로 설명용으로만 원장에 보존한다.
    """
    if not len(frame):
        return frame
    if "cohort" not in frame.columns:
        raise RuntimeError("원장에 fixed-exit loss ledger cohort가 없다")
    return frame


def _loss_ledger_profile(frame: pd.DataFrame, candidates: Sequence[Mapping[str, Any]], *,
                         accepted_guards: Sequence[Mapping[str, Any]],
                         primary_override: Mapping[str, Any] | None,
                         spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Agent가 손실 원인과 허용된 구조 후보를 연결할 수 있는 읽기 전용 요약."""
    features = {str((spec.get("semantic_invariants") or {}).get("canonical_feature") or "")}
    for candidate in [*candidates, *accepted_guards]:
        expression = candidate.get("expression", candidate) if isinstance(candidate, Mapping) else {}
        if isinstance(expression, Mapping) and expression.get("feature"):
            features.add(str(expression["feature"]))
    if isinstance(primary_override, Mapping) and primary_override.get("feature"):
        features.add(str(primary_override["feature"]))
    features.discard("")

    totals = {feature: [] for feature in sorted(features)}
    paths_used = 0
    filled = frame.loc[frame.get("status", pd.Series(index=frame.index, dtype=object)).eq(ledger.FILLED)]
    for row in filled.to_dict(orient="records"):
        try:
            arrays, _ = tickdata.load(str(row["symbol"]), str(row["date"]), Path(root))
            values = catalog.compute_features(features, catalog.Book(arrays))
            start = int(row.get("fill_tick", row.get("entry_tick", 0)))
            time_s = np.asarray(arrays["time_s"], dtype=float)
            if start < 0 or start >= len(time_s):
                continue
            end = min(len(time_s) - 1, int(np.searchsorted(
                time_s, time_s[start] + outcome.PRIMARY_HORIZON_SECONDS, side="left")))
            for feature, series in values.items():
                if start < len(series) and end < len(series):
                    before, after = float(series[start]), float(series[end])
                    if np.isfinite(before) and np.isfinite(after):
                        totals[str(feature)].append(after - before)
            paths_used += 1
        except (FileNotFoundError, KeyError, ValueError, tickdata.InsufficientQuoteData):
            continue
    changes = [
        {"feature": feature, "mean_change": float(np.mean(values)), "observations": len(values)}
        for feature, values in sorted(totals.items()) if values
    ]
    status = frame.get("status", pd.Series(index=frame.index, dtype=object)).astype(str)
    net = pd.to_numeric(frame.get("net_bps", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    filled = status.eq(ledger.FILLED)
    diagnostic = frame.get("diagnostic_cohort", frame.get(
        "cohort", pd.Series(index=frame.index, dtype=object)))
    cohorts = diagnostic.fillna("UNKNOWN").astype(str).value_counts().to_dict()
    nonpositive = (filled & net.le(0.0)) | status.eq(ledger.UNFILLED)
    profitable = filled & net.gt(0.0)
    entry_feature_contrasts = []
    for feature in catalog.list_features():
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        loss_values = values.loc[nonpositive & values.notna()]
        profit_values = values.loc[profitable & values.notna()]
        entry_feature_contrasts.append({
            "feature": feature,
            "nonpositive_observations": int(len(loss_values)),
            "profitable_observations": int(len(profit_values)),
            "nonpositive_median": (float(loss_values.median()) if len(loss_values) else None),
            "profitable_median": (float(profit_values.median()) if len(profit_values) else None),
            "nonpositive_mean": (float(loss_values.mean()) if len(loss_values) else None),
            "profitable_mean": (float(profit_values.mean()) if len(profit_values) else None),
        })
    return {
        "scope": "fixed canonical exit Discovery loss ledger",
        "decision_summary": {
            "rows": int(len(frame)),
            "filled": int(filled.sum()),
            "loss_or_flat_filled": int((filled & net.le(0.0)).sum()),
            "profitable_filled": int((filled & net.gt(0.0)).sum()),
            "unfilled": int(status.eq(ledger.UNFILLED).sum()),
        },
        "diagnostic_cohort_counts": {str(key): int(value) for key, value in cohorts.items()},
        "entry_feature_contrasts": entry_feature_contrasts,
        "post_entry_feature_changes": {"paths_used": paths_used, "changes": changes},
    }


def _catalog_context() -> dict[str, Any]:
    """Agent 어휘. 손으로 적지 않는다 — `catalog.agent_vocabulary()` 가 정본이다.

    구 구조에서 손수 적은 목록 때문에 사고가 났고(실행되지 않는 이름, 틀린 허용값),
    이 Workflow 도 같은 실수를 반복하고 있었다. 손수 목록은 `observed` 관측 범위와
    `threshold_policy` 를 통째로 버려서 Agent 가 스케일을 모르고 임계를 박았다.
    """
    vocabulary = catalog.agent_vocabulary(policy=catalog.agent_causal_policy(),
                                          visibility="grounding_visible")
    return {
        **vocabulary,
        "catalog_sha256": catalog.catalog_hash(),
        "catalog_policy_sha256": catalog.policy_hash(),
        "raw_causal_inputs": {
            "book_arrays_level_1_to_10": ["bid_price", "ask_price", "bid_qty", "ask_qty"],
            "current_tick_scalars": [
                "time_s", "local_time", "buy_volume", "sell_volume",
                "buy_max_price", "sell_min_price",
            ],
        },
    }


def _current_contract_context(amended: Mapping[str, Any],
                              accepted_guards: Sequence[Mapping[str, Any]],
                              primary_override: Mapping[str, Any] | None,
                              entry_expression: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Agent가 바꿀 수 있는 현재 entry 의미만 보여 준다."""
    return {
        "hypothesis_id": str(amended.get("hypothesis_id") or ""),
        "semantic_invariants": dict(amended.get("semantic_invariants") or {}),
        "signal_template": copy.deepcopy(amended.get("signal_template")),
        "entry_expression": copy.deepcopy(
            entry_expression or I.compile_contract(amended)["entry_program"]["signal"]),
        "entry_guards": [dict(guard) for guard in accepted_guards],
        "primary_override": (dict(primary_override) if primary_override else None),
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "round_trip_cost_bps": float(catalog.FEE_BPS),
    }


def _generated_candidate_id(proposal: Mapping[str, Any]) -> str:
    return "feedback:" + sha256_json(proposal.get("entry_expression") or {})[:12]


def run(spec: Mapping[str, Any], *, symbols: Sequence[str], dates: Sequence[str], output: Path,
        root: Path = TICK_ROOT, config: RefinementConfig = RefinementConfig(),
        prepared: PreparedRefinement | None = None,
        initial_parent: BootstrapParent | None = None,
        runner: Any | None = None) -> dict[str, Any]:
    """Agent의 완전한 진입식 제안과 정확 재생을 최대 10회 반복한다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    symbols, dates = tuple(map(str, symbols)), tuple(map(str, dates))
    if not symbols or not dates:
        raise ValueError("Entry refinement에 종목과 Discovery 날짜가 모두 필요하다")

    # Parameter Search와 같은 q-parameter 계약을 먼저 만든다. 원 Specification의
    # `theta_*` 자리를 그대로 두면 q0.80 보정값을 줘도 이름이 달라 종목-일이 빠진다.
    prepared = prepared or prepare(spec, symbols=symbols, dates=dates, root=Path(root), config=config)
    if not _prepared_matches(prepared, spec, symbols=symbols, dates=dates, config=config):
        raise ValueError("Entry refinement batch preparation이 현재 specification·기간과 다르다")
    amended, parameters, calibration = prepared.amended, prepared.parameters, prepared.calibration
    progress_identity = _progress_identity(amended, symbols=symbols, dates=dates, config=config)
    completed = _completed_result(output, progress_identity)
    if completed is not None:
        return completed
    progress = _load_progress(output, progress_identity)
    accepted_guards: list[dict[str, Any]] = (
        [dict(guard) for guard in progress["accepted_guards"]] if progress else [])
    current_entry_expression: dict[str, Any] | None = (
        copy.deepcopy(progress.get("entry_expression")) if progress else None)
    rounds: list[dict[str, Any]] = (
        [dict(round_record) for round_record in progress["rounds"]] if progress else [])
    terminal_state = ENTRY_REFINEMENT_NO_CHANGE
    primary_override: dict[str, Any] | None = None
    refined_specification: dict[str, Any] | None = None
    saved_initial = progress.get("initial_scorable") if progress else None
    initial_scorable: int | None = int(saved_initial) if saved_initial is not None else None
    start_round = int(progress.get("next_round") or (len(rounds) + 1)) if progress else 1

    for index in range(start_round, int(config.max_rounds) + 1):
        round_output = output / "rounds" / f"{index:02d}"
        parent_template = _template(
            amended, accepted_guards, entry_expression=current_entry_expression)
        uses_initial_parent = bool(index == 1 and not accepted_guards
                                   and current_entry_expression is None
                                   and initial_parent is not None)
        if uses_initial_parent:
            if initial_parent.contract_sha256 != str(parent_template["contract_sha256"]):
                raise ValueError("공유 bootstrap parent가 현재 entry 계약과 다르다")
            if (initial_parent.parameter_table_sha256
                    and initial_parent.parameter_table_sha256 != sha256_json(parameters)):
                raise ValueError("공유 bootstrap parent의 prior-100-tick parameter가 현재 q와 다르다")
            parent_frame, parent_manifest = initial_parent.frame, initial_parent.manifest
            parent_metrics = metrics.measure(parent_frame, parent_manifest,
                                             contract_id=initial_parent.contract_id)
            parent_ledger_path = initial_parent.ledger_path
            parent_contract_id = initial_parent.contract_id
        else:
            parent_frame, parent_manifest, parent_metrics = _run_ledger(
                parent_template, symbols=symbols, dates=dates, parameters=parameters,
                output=round_output / "parent", root=Path(root))
            parent_ledger_path = str(round_output / "parent" / "ledger.parquet")
            parent_contract_id = str(parent_template["hypothesis_id"])
        loss_ledger = _loss_ledger_frame(parent_frame)
        routing = diagnose.route(loss_ledger)
        round_record: dict[str, Any] = {
            "round": index,
            "parent_contract_sha256": parent_template["contract_sha256"],
            "entry_guards": copy.deepcopy(accepted_guards),
            "entry_expression": copy.deepcopy(
                current_entry_expression or parent_template["entry_program"]["signal"]),
            "parent_metrics": parent_metrics.as_dict(),
            "selection_target": SELECTION_TARGET,
            "selection_metric": SELECTION_METRIC,
            "diagnostic_route": routing,
            "ledger": {
                "path": parent_ledger_path,
                "contract_id": parent_contract_id,
                "manifest_errors": list(parent_manifest.get("errors") or []),
                "missing_symbol_days": list(parent_manifest.get("missing") or []),
                "shared_bootstrap_batch": uses_initial_parent,
                "bootstrap_cache_state": (initial_parent.cache_state if uses_initial_parent else None),
            },
        }
        if parent_metrics.scorable == 0:
            terminal_state = ENTRY_REFINEMENT_NO_DECISIONS
            rounds.append(round_record)
            break
        if initial_scorable is None:
            initial_scorable = int(parent_metrics.scorable)
        # raw 가격 경로 cohort는 손실 유형을 설명하는 연구 진단일 뿐, entry guard
        # 탐색을 멈추게 하지 않는다. 고정 exit를 바꾸지 않는다는 것은 exit 설정을
        # 탐색하지 않는다는 뜻이지, 그 exit 아래 실제 손실 난 진입을 구별하지
        # 않는다는 뜻은 아니다. 이 Stage의 학습 표적은 실제 fixed-exit net뿐이다.
        net = pd.to_numeric(
            loss_ledger.get("net_bps", pd.Series(float("nan"), index=loss_ledger.index)),
            errors="coerce")
        filled = loss_ledger.get("status", pd.Series(index=loss_ledger.index, dtype=object)).eq(ledger.FILLED)
        observed = filled & net.notna()
        loss_count = int((observed & net.le(0.0)).sum())
        profit_count = int((observed & net.gt(0.0)).sum())
        unfilled_count = int(loss_ledger.get(
            "status", pd.Series(index=loss_ledger.index, dtype=object)).eq(ledger.UNFILLED).sum())
        round_record["fixed_exit_net_ledger"] = {
            "filled_with_observed_net": int(observed.sum()),
            "loss_or_flat_filled": loss_count,
            "profitable": profit_count,
            "unfilled_zero_return": unfilled_count,
            "nonpositive_scorable_decisions": loss_count + unfilled_count,
            "diagnostic_route_is_report_only": True,
        }
        if len(loss_ledger) and loss_count == 0 and unfilled_count == 0:
            terminal_state = (ENTRY_REFINEMENT_COMPLETE if (
                accepted_guards or current_entry_expression is not None)
                              else ENTRY_REFINEMENT_NO_CHANGE)
            round_record["terminal_reason"] = (
                "고정 canonical exit 아래 손실·보합 체결 또는 0원 미체결 결정이 없어 "
                "추가 entry refinement가 필요 없다")
            rounds.append(round_record)
            break
        if routing["route"] in (diagnose.ROUTE_REFUTED, diagnose.ROUTE_FIXED_EXIT,
                                diagnose.ROUTE_COST, diagnose.ROUTE_NOTHING):
            round_record["diagnostic_note"] = (
                "원시 가격 경로 진단은 보존했지만, fixed-exit 손실 장부 기반 entry expression "
                "탐색은 계속한다")

        exact_rows: list[dict[str, Any]] = []
        templates: list[tuple[str, Mapping[str, Any]]] = []
        actions: dict[str, dict[str, Any]] = {}
        catalog_context = _catalog_context()
        current_contract = _current_contract_context(
            amended, accepted_guards, primary_override,
            entry_expression=current_entry_expression)
        loss_profile = _loss_ledger_profile(
            loss_ledger, (), accepted_guards=accepted_guards,
            primary_override=primary_override, spec=amended, root=Path(root))
        feedback = refinement_agent.run(
            loss_profile, current_contract=current_contract,
            catalog_context=catalog_context, runner=runner)
        round_record["agent_feedback"] = feedback
        invocation_roles = [str(item.get("role"))
                            for item in feedback.get("agent_invocations") or []]
        round_record["catalog_injection"] = {
            "catalog_sha256": catalog_context["catalog_sha256"],
            "catalog_policy_sha256": catalog_context["catalog_policy_sha256"],
            "feature_count": len(catalog_context.get("features") or []),
            "injected_into": invocation_roles,
        }
        if feedback["state"] != "READY":
            terminal_state = ENTRY_REFINEMENT_AGENT_INVALID_OUTPUT
            rounds.append(round_record)
            break
        proposals = [dict(item) for item in feedback["payload"].get("proposals") or []]
        for proposal in proposals:
            candidate_id = _generated_candidate_id(proposal)
            expression = copy.deepcopy(dict(proposal["entry_expression"]))
            if expression == parent_template["entry_program"]["signal"]:
                continue
            templates.append((candidate_id, _template(
                amended, entry_expression=expression)))
            actions[candidate_id] = {
                "candidate_id": candidate_id,
                "source": "FEEDBACK_GENERATED_HYPOTHESIS",
                "action_kind": "REPLACE_ENTRY_EXPRESSION",
                "entry_expression": expression,
                "hypothesis": proposal,
            }
        round_record["candidate_selection"] = {
            "source": "FEEDBACK_GENERATED_HYPOTHESES",
            "prebuilt_candidate_list": False,
            "generated": len(proposals),
            "decision_floor": None,
            "parent_improvement_required": False,
            "selection_metric": SELECTION_METRIC,
        }

        exact: dict[str, Any] = {}
        if templates:
            exact = _run_exact_batch(
                templates, symbols=symbols, dates=dates, parameters=parameters,
                output=round_output / "exact", root=Path(root))
        for candidate_id, child_template in templates:
            result = exact[candidate_id]
            exact_rows.append({
                **actions[candidate_id],
                "contract_sha256": child_template["contract_sha256"],
                "exact_metrics": result["metrics"],
                "ledger": {
                    "path": str(round_output / "exact" / "ledger.parquet"),
                    "contract_id": result["contract_id"],
                    "missing_symbol_days": list((exact.get("_manifest") or {}).get("missing") or []),
                    "rows": int(result["rows"]),
                },
            })
        round_record["exact_candidates"] = [
            {**{key: value for key, value in row.items() if key != "exact_metrics"},
             "exact_metrics": row["exact_metrics"].as_dict()}
            for row in exact_rows
        ]
        selected_ids = [str(row["candidate_id"]) for row in exact_rows]
        selected = exact_rows
        round_record["agent_selected_candidate_ids"] = selected_ids
        if not selected_ids:
            terminal_state = (ENTRY_REFINEMENT_COMPLETE if (
                accepted_guards or current_entry_expression is not None)
                              else ENTRY_REFINEMENT_NO_CHANGE)
            round_record["terminal_reason"] = "자연어 feedback에서 실행 가능한 새 진입 가설이 나오지 않았다"
            rounds.append(round_record)
            break

        best = max(float(row["exact_metrics"].total_net_bps) for row in selected)
        tied = [row for row in selected
                if float(row["exact_metrics"].total_net_bps) == best]
        if len(tied) != 1:
            terminal_state = ENTRY_REFINEMENT_TIED_EXACT
            round_record["exact_tie_candidate_ids"] = [row["candidate_id"] for row in tied]
            rounds.append(round_record)
            break
        chosen = tied[0]
        if chosen.get("action_kind") == "REPLACE_ENTRY_EXPRESSION":
            current_entry_expression = copy.deepcopy(chosen["entry_expression"])
            accepted_guards = []
            primary_override = None
            refined_specification = None
            round_record["accepted_feedback_hypothesis"] = dict(chosen["hypothesis"])
            round_record["accepted_entry_expression"] = copy.deepcopy(
                current_entry_expression)
            rounds.append(round_record)
            _write_progress(
                output, progress_identity, rounds=rounds, accepted_guards=accepted_guards,
                entry_expression=current_entry_expression,
                initial_scorable=initial_scorable)
            continue
    else:
        terminal_state = ENTRY_REFINEMENT_COMPLETE

    result = {
        "schema": SCHEMA_VERSION,
        "created_at": now_utc(),
        "state": terminal_state,
        "hypothesis_id": str(spec["hypothesis_id"]),
        "input_identity": progress_identity,
        "amended_specification_sha256": str(amended["specification_sha256"]),
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "entry_refinement_policy_version": ENTRY_REFINEMENT_POLICY_VERSION,
        "bootstrap": calibration,
        "config": _config_payload(config),
        "entry_guards": accepted_guards,
        "entry_expression": copy.deepcopy(current_entry_expression),
        "primary_override": primary_override,
        "refined_specification": refined_specification,
        "rounds": rounds,
        "note": (
            "청산은 모든 장부에서 같은 canonical profile이다. 진입 보완은 그 고정 exit의 "
            "Discovery 손실 장부를 자연어 feedback으로 먼저 고정하고, 항상 주입된 Catalog로 "
            "새 진입 가설을 만든다. exit 자체는 바꾸지 않는다."
        ),
        "selection_target": SELECTION_TARGET,
        "selection_metric": SELECTION_METRIC,
    }
    write_json(output / "entry_refinement.json", result)
    _close_progress(output, progress_identity, rounds=rounds,
                    accepted_guards=accepted_guards,
                    entry_expression=current_entry_expression,
                    initial_scorable=initial_scorable,
                    terminal_state=terminal_state)
    return result


def run_q_branches(spec: Mapping[str, Any], *, symbols: Sequence[str], dates: Sequence[str], output: Path,
                   root: Path = TICK_ROOT, config: RefinementConfig = RefinementConfig(),
                   runner: Any | None = None) -> dict[str, Any]:
    """미리 고정한 q마다 별도 Discovery 손실 장부로 entry guard를 만든다.

    q와 guard를 Search 구간에서 함께 학습하지 않는다. 각 branch는 Discovery만 보고
    완결하고, Parameter Search는 그 완결된 entry 계약들을 한 번 비교할 뿐이다.
    """
    values = tuple(sorted({float(value) for value in config.branch_quantiles}))
    if not values:
        return run(spec, symbols=symbols, dates=dates, output=output, root=root, config=config,
                   runner=runner)
    search_config = S.config_for(spec)
    if search_config.independent_state_quantiles:
        raise ValueError("상태별 독립 q 후보의 branch refinement는 아직 지원하지 않는다")
    if any(value not in search_config.grid for value in values):
        raise ValueError("Entry Refinement branch q가 Parameter Search의 고정 grid 밖에 있다")
    if set(values) != set(search_config.grid):
        raise ValueError("q별 Entry Refinement는 Parameter Search의 모든 고정 q를 만들어야 한다")

    output = Path(output)
    branches: dict[str, dict[str, Any]] = {}
    prepared_by_candidate: dict[str, PreparedRefinement] = {}
    configs: dict[str, RefinementConfig] = {}
    outputs: dict[str, Path] = {}
    needs_parent: dict[str, PreparedRefinement] = {}
    for value in values:
        candidate_id = search_config.candidate_id(value)
        branch_config = replace(config, bootstrap_quantile=value, branch_quantiles=(),
                                auto_branch_search_quantiles=False)
        branch_output = output / "branches" / candidate_id
        prepared = prepare(spec, symbols=symbols, dates=dates, root=Path(root), config=branch_config)
        progress_identity = _progress_identity(
            prepared.amended, symbols=symbols, dates=dates, config=branch_config)
        prepared_by_candidate[candidate_id] = prepared
        configs[candidate_id] = branch_config
        outputs[candidate_id] = branch_output
        # 완료 branch는 `run()`이 input identity로 바로 재사용한다. IN_PROGRESS branch는
        # 다음 round부터만 이어야 하므로 새 parent batch에 섞지 않는다.
        if (_completed_result(branch_output, progress_identity) is None
                and _load_progress(branch_output, progress_identity) is None):
            needs_parent[candidate_id] = prepared
    parents: dict[str, BootstrapParent] = {}
    if needs_parent:
        batch_key = bootstrap_batch_cache_key(
            needs_parent, symbols=symbols, dates=dates, root=Path(root))
        parents = run_bootstrap_batch(
            needs_parent, symbols=symbols, dates=dates,
            output=output / "bootstrap_parent_batches" / batch_key, root=Path(root))
    for value in values:
        candidate_id = search_config.candidate_id(value)
        branch = run(spec, symbols=symbols, dates=dates,
                     output=outputs[candidate_id], root=root,
                     config=configs[candidate_id],
                     prepared=prepared_by_candidate[candidate_id],
                     initial_parent=parents.get(candidate_id), runner=runner)
        branches[candidate_id] = branch

    representative_id = search_config.candidate_id(
        config.bootstrap_quantile if config.bootstrap_quantile in values else values[0])
    representative = branches[representative_id]
    result = {
        "schema": "entry_refinement_branches.v1",
        "created_at": now_utc(),
        "state": ENTRY_REFINEMENT_COMPLETE,
        "hypothesis_id": str(spec["hypothesis_id"]),
        "amended_specification_sha256": representative["amended_specification_sha256"],
        "fixed_exit_profile": {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()},
        "entry_refinement_policy_version": ENTRY_REFINEMENT_POLICY_VERSION,
        "config": _config_payload(config),
        "entry_refinement_branch_quantiles": list(values),
        "entry_guards": [dict(guard) for guard in representative.get("entry_guards") or []],
        "entry_expression": copy.deepcopy(representative.get("entry_expression")),
        "entry_guard_variants": {
            candidate_id: [dict(guard) for guard in branch.get("entry_guards") or []]
            for candidate_id, branch in sorted(branches.items())
        },
        "entry_expression_variants": {
            candidate_id: copy.deepcopy(branch.get("entry_expression"))
            for candidate_id, branch in sorted(branches.items())
        },
        "branch_records": {
            candidate_id: {
                "state": branch.get("state"),
                "output": str(output / "branches" / candidate_id),
                "entry_guards": [dict(guard) for guard in branch.get("entry_guards") or []],
                "entry_guards_sha256": sha256_json(branch.get("entry_guards") or [])[:16],
                "entry_expression": copy.deepcopy(branch.get("entry_expression")),
                "entry_expression_sha256": sha256_json(
                    branch.get("entry_expression"))[:16],
                "bootstrap_quantile": branch.get("bootstrap", {}).get("quantile"),
            }
            for candidate_id, branch in sorted(branches.items())
        },
        "note": (
            "각 branch의 guard는 해당 q의 Discovery fixed-exit 손실 장부에서만 만들었다. "
            "Search는 q별로 이미 고정된 entry 계약을 비교하고, 선택된 하나만 새 날짜에서 확인한다."
        ),
        "selection_target": SELECTION_TARGET,
    }
    write_json(output / "entry_refinement.json", result)
    return result
