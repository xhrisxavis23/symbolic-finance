"""Discovery 실행 장부를 다음 가설의 제한된 연구 입력으로 만든다.

이 모듈은 이미 실행한 후보의 **Discovery 구간** 원장만 읽는다. 여기서 나온
손실/회복 차이는 다음 가설이 살펴볼 현상이지, 새 가설을 지지하는 Evidence나
Validation 결과가 아니다. 따라서 Search·Validation·Final 날짜와 원장은 열지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import catalog
from ..config import clean, read_json, sha256_json
from ..contracts.artifacts import create_artifact, read_artifact, write_artifact
from ..ledger import FILLED


SCHEMA_VERSION = "discovery_loss_context.v2"
ARTIFACT_NAME = "discovery_loss_context_artifact.json"
MAX_CASES_PER_COHORT = 8
MAX_FEATURE_CONTRASTS = 12
SANDBOX_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = SANDBOX_ROOT / "cache" / "discovery_loss"


class DiscoveryParentLedgerUnavailable(ValueError):
    """전일 기준이 없어 다음 Discovery refinement를 열 수 없는 정상 상태."""


def build(source_research: Path, hypothesis_id: str, output: Path, *,
          cache_root: Path | None = None) -> dict[str, Any]:
    """현재 Entry Refinement 계약의 Discovery ledger를 다음 연구 입력으로 저장한다."""
    source_research = Path(source_research).resolve()
    run = read_artifact(source_research / "research_run.json", kind="research_run")
    refinement_ref = str((run["payload"].get("artifacts") or {}).get("entry_refinement_set") or "")
    if not refinement_ref:
        raise ValueError("source research run에 Entry Refinement artifact가 없다")
    refinement_set = read_artifact(
        source_research / "04_entry_refinement" / "entry_refinement_artifact.json",
        kind="entry_refinement_set")
    if refinement_set["artifact_id"] != refinement_ref:
        raise ValueError("research run과 Entry Refinement artifact의 lineage가 다르다")
    unit = (refinement_set["payload"].get("units") or {}).get(str(hypothesis_id))
    if not isinstance(unit, Mapping):
        raise ValueError(f"Entry Refinement에 {hypothesis_id} 가설이 없다")
    refinement_path = source_research / "04_entry_refinement" / "units" / str(hypothesis_id) / "entry_refinement.json"
    refinement, branch_id = _selected_refinement_branch(
        source_research, str(hypothesis_id), read_json(refinement_path))
    rounds = list(refinement.get("rounds") or [])
    if not rounds:
        raise ValueError("Entry Refinement에 Discovery parent ledger가 없다")
    current_round = _current_refinement_round(rounds)
    parent = current_round.get("ledger") or {}
    ledger_path = _inside_source(source_research, parent.get("path"))
    parents = {"research_run": run["artifact_id"],
               "entry_refinement_set": refinement_set["artifact_id"]}
    cache_path = _cache_path(cache_root or CACHE_ROOT, source_research, hypothesis_id,
                             parents=parents)
    cached = _read_cached_context(cache_path, parents=parents)
    if cached is not None:
        write_artifact(Path(output) / ARTIFACT_NAME, cached)
        return cached
    frame = pd.read_parquet(ledger_path)
    contract_id = str(parent.get("contract_id") or "")
    if contract_id:
        if "contract_id" not in frame.columns:
            raise ValueError("Discovery parent ledger에 contract_id가 없다")
        frame = frame.loc[frame["contract_id"].astype(str).eq(contract_id)].copy()
        if frame.empty:
            raise DiscoveryParentLedgerUnavailable(
                "Discovery parent ledger에 이 hypothesis contract의 실행 가능한 decision이 없다")
    elif frame.get("contract_id", pd.Series(dtype=object)).nunique() > 1:
        raise ValueError("공유 Discovery parent ledger에는 hypothesis contract_id가 필요하다")
    if "diagnostic_cohort" not in frame.columns:
        raise ValueError("Discovery ledger에 diagnostic_cohort가 없다")
    profile = ((run["payload"].get("request") or {}).get("profile") or {})
    payload = {
        "schema": SCHEMA_VERSION,
        "source_hypothesis_id": str(hypothesis_id),
        "profile": clean(profile),
        "discovery_only": True,
        "source": {
            "research_run_artifact": run["artifact_id"],
            "entry_refinement_artifact": refinement_set["artifact_id"],
            "parent_ledger": str(ledger_path),
            "parent_contract_id": (contract_id or None),
            "round": int(current_round.get("round") or 1),
            "entry_refinement_branch_id": branch_id,
            "entry_guards": [dict(guard) for guard in current_round.get("entry_guards") or []],
            "fixed_exit_profile": refinement.get("fixed_exit_profile"),
        },
        "interpretation_boundary": (
            "이 context는 고정 canonical exit로 실행된 Discovery 원장의 실패·회복 진단이다. "
            "손실 cohort와 feature contrast는 다음 가설의 질문을 정하는 데만 쓰며, "
            "그 자체가 새 가설의 Evidence, 미래 예측, Validation 또는 Final 수익 증거는 아니다."
        ),
        "summary": _summary(frame),
        "case_samples": _cases(frame),
        "prior_hypothesis": _prior_hypothesis(source_research, run, str(hypothesis_id)),
    }
    artifact = create_artifact("discovery_loss_context", payload, parents=parents)
    write_artifact(Path(output) / ARTIFACT_NAME, artifact)
    write_artifact(cache_path, artifact)
    return artifact


def context_identity(source_research: Path, hypothesis_id: str) -> dict[str, Any]:
    """어느 refinement 계약의 손실 장부인지 plan 단계에서 판별할 최소 지문."""
    source_research = Path(source_research).resolve()
    run = read_artifact(source_research / "research_run.json", kind="research_run")
    refinement_ref = str((run["payload"].get("artifacts") or {}).get("entry_refinement_set") or "")
    if not refinement_ref:
        raise ValueError("source research run에 Entry Refinement artifact가 없다")
    refinement_artifact_path = source_research / "04_entry_refinement" / "entry_refinement_artifact.json"
    refinement_path = (source_research / "04_entry_refinement" / "units" /
                       str(hypothesis_id) / "entry_refinement.json")
    # 최소 source는 lineage ID만 남길 수 있다. 실제 dispatch는 build()가 상세
    # ledger를 검증하고, plan은 같은 minimal lineage의 attempt만 구분하면 된다.
    if not refinement_artifact_path.is_file() or not refinement_path.is_file():
        return {
            "schema": SCHEMA_VERSION,
            "research_run_artifact": str(run["artifact_id"]),
            "entry_refinement_artifact": refinement_ref,
            "hypothesis_id": str(hypothesis_id),
            "state": "MINIMAL_REFINEMENT_LINEAGE",
        }
    refinement_set = read_artifact(
        refinement_artifact_path,
        kind="entry_refinement_set")
    if refinement_set["artifact_id"] != refinement_ref:
        raise ValueError("research run과 Entry Refinement artifact의 lineage가 다르다")
    refinement, branch_id = _selected_refinement_branch(
        source_research, str(hypothesis_id), read_json(refinement_path))
    try:
        current_round = _current_refinement_round(list(refinement.get("rounds") or []))
    except ValueError:
        return {
            "schema": SCHEMA_VERSION,
            "research_run_artifact": str(run["artifact_id"]),
            "entry_refinement_artifact": str(refinement_set["artifact_id"]),
            "hypothesis_id": str(hypothesis_id),
            "state": "MINIMAL_REFINEMENT_LINEAGE",
        }
    ledger = current_round.get("ledger") or {}
    return {
        "schema": SCHEMA_VERSION,
        "research_run_artifact": str(run["artifact_id"]),
        "entry_refinement_artifact": str(refinement_set["artifact_id"]),
        "hypothesis_id": str(hypothesis_id),
        "round": int(current_round.get("round") or 1),
        "parent_ledger": str(ledger.get("path") or ""),
        "parent_contract_id": str(ledger.get("contract_id") or ""),
        "entry_refinement_branch_id": branch_id,
        "entry_guards": [dict(guard) for guard in current_round.get("entry_guards") or []],
    }


def _selected_refinement_branch(source_research: Path, hypothesis_id: str,
                                refinement: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """q별 Refinement라면 Search가 실제 선택한 q의 Discovery ledger만 다음 Agent에 준다."""
    value = dict(refinement)
    if value.get("rounds"):
        return value, None
    records = value.get("branch_records") or {}
    if not isinstance(records, Mapping):
        return value, None
    selected_path = (Path(source_research) / "05_parameter_search" / "units" /
                     str(hypothesis_id) / "selected_parameter.json")
    if not selected_path.is_file():
        raise ValueError("q별 Entry Refinement의 선택된 q가 없어 Discovery loss context를 만들 수 없다")
    selected = read_json(selected_path)
    if selected.get("status") != "SEARCH_SELECTED":
        raise ValueError("q별 Entry Refinement는 Parameter Search가 q를 고른 뒤에만 successor로 넘긴다")
    try:
        branch_id = f"q_{float(selected['selected_q']):.2f}"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Parameter Search의 selected_q가 없다") from error
    branch = records.get(branch_id)
    if not isinstance(branch, Mapping):
        raise ValueError(f"선택된 q의 Entry Refinement branch가 없다: {branch_id}")
    branch_path = (Path(source_research) / "04_entry_refinement" / "units" /
                   str(hypothesis_id) / "branches" / branch_id / "entry_refinement.json")
    if not branch_path.is_file():
        raise FileNotFoundError(f"선택된 q의 Entry Refinement 결과가 없다: {branch_path}")
    return dict(read_json(branch_path)), branch_id


def _current_refinement_round(rounds: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """마지막으로 실제 재생된 entry 계약을 고른다.

    round N의 parent ledger는 N-1까지 채택된 guard를 적용한 현재 계약이다. 첫 round를
    고르면 이미 수용한 guard를 잊고 개선 전 손실을 다음 Agent에 다시 주게 된다.
    """
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(rounds):
        if not isinstance(item, Mapping):
            continue
        ledger = item.get("ledger")
        if not isinstance(ledger, Mapping) or not ledger.get("path"):
            continue
        try:
            round_number = int(item.get("round") or index + 1)
        except (TypeError, ValueError):
            round_number = index + 1
        candidates.append((round_number, index, item))
    if not candidates:
        raise ValueError("Entry Refinement에 현재 Discovery parent ledger가 없다")
    return max(candidates, key=lambda value: (value[0], value[1]))[2]


def _cache_path(cache_root: Path, source_research: Path, hypothesis_id: str, *,
                parents: Mapping[str, str]) -> Path:
    key = sha256_json({"schema": SCHEMA_VERSION, "source": str(source_research),
                       "hypothesis_id": str(hypothesis_id), "parents": dict(parents)})[:24]
    return Path(cache_root) / f"{key}.json"


def _read_cached_context(path: Path, *, parents: Mapping[str, str]) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        artifact = read_artifact(path, kind="discovery_loss_context")
    except (OSError, ValueError):
        return None
    if dict(artifact.get("parents") or {}) != dict(parents):
        return None
    return artifact


def _prior_hypothesis(source: Path, run: Mapping[str, Any], hypothesis_id: str) -> dict[str, Any] | None:
    """손실 장부를 만든 부모 가설을 다음 Agent의 중복 비교용으로만 보존한다."""
    request = run["payload"].get("request") or {}
    pre_source = request.get("pre_validation_source")
    pre_root = (Path(str(pre_source)) if pre_source else source / "01_pre_validation")
    if not pre_root.is_absolute():
        pre_root = (Path.cwd() / pre_root).resolve()
    path = pre_root / "02_hypothesis" / "hypothesis_artifact.json"
    if not path.is_file():
        return None
    artifact = read_artifact(path, kind="hypothesis_set")
    hypotheses = ((artifact["payload"].get("payload") or {}).get("hypotheses") or [])
    parent = next((item for item in hypotheses if str(item.get("hypothesis_id")) == hypothesis_id), None)
    if not isinstance(parent, Mapping):
        return None
    return {
        "hypothesis_id": str(parent.get("hypothesis_id")),
        "title": str(parent.get("title")),
        "hypothesis_structure": parent.get("hypothesis_structure"),
        "evidence_basis": [{key: item.get(key) for key in ("family", "representative_feature")}
                           for item in parent.get("evidence_basis") or []],
        "mechanistic_claim": parent.get("mechanistic_claim"),
        "source_of_profit": parent.get("source_of_profit"),
        "mechanism_tests": [{key: item.get(key) for key in ("test_type", "claim")}
                            for item in parent.get("mechanism_tests") or []],
    }


def _inside_source(source: Path, value: Any) -> Path:
    if not value:
        raise ValueError("Entry Refinement parent ledger 경로가 없다")
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(source)
    except ValueError as error:
        raise ValueError("Discovery loss context는 source research run 내부 ledger만 읽는다") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Discovery parent ledger가 없다: {candidate}")
    return candidate


def _filled(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        raise ValueError("Discovery ledger에 status가 없다")
    return frame.loc[frame["status"].eq(FILLED)].copy()


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    filled = _filled(frame)
    cohorts = filled["diagnostic_cohort"].fillna("UNKNOWN").astype(str)
    net = pd.to_numeric(filled.get("net_bps"), errors="coerce")
    return {
        "decisions": int(len(frame)),
        "fills": int(len(filled)),
        "diagnostic_cohort_counts": {str(k): int(v) for k, v in cohorts.value_counts().sort_index().items()},
        "filled_net_bps": {
            "mean": _number(net.mean()), "median": _number(net.median()),
            "positive_share": _number((net > 0.0).mean()),
        },
        "entry_signal_at_fill": _entry_signal_at_fill_summary(filled),
        "entry_signal_active_share_until_fill": _active_share_until_fill_summary(filled),
        "fill_wait_ticks": _fill_wait_ticks_summary(filled),
        "adverse_vs_recovery_feature_contrasts": _feature_contrasts(filled),
        "case_selection": (
            "각 diagnostic cohort에서 (symbol, date, decision_index) 정렬 뒤 동일 간격으로 "
            f"최대 {MAX_CASES_PER_COHORT}개를 고른다. 손익 크기로 사례를 고르지 않는다."
        ),
    }


def _entry_signal_at_fill_summary(filled: pd.DataFrame) -> dict[str, Any]:
    """체결 뒤가 아니라, **체결 순간** entry condition이 남았는지 Discovery에만 남긴다."""
    if "entry_signal_active_at_fill" not in filled:
        return {"available": False, "observed_fills": 0}
    active = filled["entry_signal_active_at_fill"].astype("boolean")
    observed = active.dropna()
    if observed.empty:
        return {"available": True, "observed_fills": 0}
    out: dict[str, Any] = {
        "available": True,
        "observed_fills": int(len(observed)),
        "active_share": _number(observed.astype(float).mean()),
    }
    for cohort, key in (("PERSISTENT_ADVERSE", "persistent_adverse_active_share"),
                        ("NET_RECOVERY", "net_recovery_active_share")):
        values = active.loc[filled["diagnostic_cohort"].eq(cohort)].dropna()
        out[key] = _number(values.astype(float).mean()) if len(values) else None
    out["persistence_until_fill"] = _entry_signal_persistence_summary(filled)
    return out


def _entry_signal_persistence_summary(filled: pd.DataFrame) -> dict[str, Any]:
    """게시부터 체결까지 신호가 한 번도 꺼지지 않았는지 별도로 남긴다."""
    if "entry_signal_persisted_until_fill" not in filled:
        return {"available": False, "observed_fills": 0}
    persisted = filled["entry_signal_persisted_until_fill"].astype("boolean")
    observed = persisted.dropna()
    if observed.empty:
        return {"available": True, "observed_fills": 0}
    out: dict[str, Any] = {
        "available": True,
        "observed_fills": int(len(observed)),
        "persisted_share": _number(observed.astype(float).mean()),
    }
    for cohort, key in (("PERSISTENT_ADVERSE", "persistent_adverse_persisted_share"),
                        ("NET_RECOVERY", "net_recovery_persisted_share")):
        values = persisted.loc[filled["diagnostic_cohort"].eq(cohort)].dropna()
        out[key] = _number(values.astype(float).mean()) if len(values) else None
    return out


def _active_share_until_fill_summary(filled: pd.DataFrame) -> dict[str, Any]:
    """게시부터 체결까지 signal이 켜져 있던 비율을 연구용으로만 요약한다."""
    column = "entry_signal_active_share_until_fill"
    if column not in filled:
        return {"available": False, "observed_fills": 0, "mean_active_share": None}
    active_share = pd.to_numeric(filled[column], errors="coerce")
    observed = active_share.dropna()
    out: dict[str, Any] = {
        "available": True,
        "observed_fills": int(len(observed)),
        "mean_active_share": _number(observed.mean()) if len(observed) else None,
    }
    for cohort, key in (("PERSISTENT_ADVERSE", "persistent_adverse_mean_active_share"),
                        ("NET_RECOVERY", "net_recovery_mean_active_share")):
        values = active_share.loc[filled["diagnostic_cohort"].eq(cohort)].dropna()
        out[key] = _number(values.mean()) if len(values) else None
    return out


def _fill_wait_ticks_summary(filled: pd.DataFrame) -> dict[str, Any]:
    """게시부터 실제 체결까지 걸린 tick 수를 연구용으로만 요약한다."""
    if "entry_tick" not in filled or "fill_tick" not in filled:
        return {"available": False, "observed_fills": 0, "median_ticks": None}
    entry = pd.to_numeric(filled["entry_tick"], errors="coerce")
    executed = pd.to_numeric(filled["fill_tick"], errors="coerce")
    waits = (executed - entry).loc[(entry >= 0) & (executed >= entry)].dropna()
    out: dict[str, Any] = {
        "available": True,
        "observed_fills": int(len(waits)),
        "median_ticks": _number(waits.median()) if len(waits) else None,
    }
    for cohort, key in (("PERSISTENT_ADVERSE", "persistent_adverse_median_ticks"),
                        ("NET_RECOVERY", "net_recovery_median_ticks")):
        values = waits.loc[filled.loc[waits.index, "diagnostic_cohort"].eq(cohort)]
        out[key] = _number(values.median()) if len(values) else None
    return out


def _feature_contrasts(filled: pd.DataFrame) -> list[dict[str, Any]]:
    adverse = filled.loc[filled["diagnostic_cohort"].eq("PERSISTENT_ADVERSE")]
    recovery = filled.loc[filled["diagnostic_cohort"].eq("NET_RECOVERY")]
    rows: list[dict[str, Any]] = []
    for feature in catalog.guard_axes():
        if feature not in filled.columns:
            continue
        all_values = pd.to_numeric(filled[feature], errors="coerce")
        left = pd.to_numeric(adverse[feature], errors="coerce").dropna()
        right = pd.to_numeric(recovery[feature], errors="coerce").dropna()
        if not len(left) or not len(right):
            continue
        iqr = float(all_values.quantile(0.75) - all_values.quantile(0.25))
        if not np.isfinite(iqr) or iqr <= 0.0:
            continue
        difference = float(left.median() - right.median())
        rows.append({
            "feature": str(feature),
            "persistent_adverse_median": _number(left.median()),
            "net_recovery_median": _number(right.median()),
            "adverse_minus_recovery_iqr": _number(difference / iqr),
            "adverse_observations": int(len(left)),
            "recovery_observations": int(len(right)),
        })
    rows.sort(key=lambda row: (-abs(float(row["adverse_minus_recovery_iqr"] or 0.0)), row["feature"]))
    return rows[:MAX_FEATURE_CONTRASTS]


def _cases(frame: pd.DataFrame) -> list[dict[str, Any]]:
    filled = _filled(frame)
    features = [feature for feature in catalog.guard_axes() if feature in filled.columns]
    cases: list[dict[str, Any]] = []
    for cohort in sorted(str(v) for v in filled["diagnostic_cohort"].dropna().unique()):
        group = filled.loc[filled["diagnostic_cohort"].eq(cohort)].copy()
        if group.empty:
            continue
        sort_columns = [name for name in ("symbol", "date", "decision_index") if name in group.columns]
        group = group.sort_values(sort_columns, kind="stable")
        positions = np.linspace(0, len(group) - 1, min(MAX_CASES_PER_COHORT, len(group)), dtype=int)
        for _, row in group.iloc[sorted(set(int(v) for v in positions))].iterrows():
            symbol, date, decision = str(row["symbol"]), str(row["date"]), int(row["decision_index"])
            cases.append({
                "case_id": f"{symbol}:{date}:{decision}",
                "diagnostic_cohort": cohort,
                "symbol": symbol, "date": date, "decision_index": decision,
                "entry_tick": _integer(row.get("entry_tick")),
                "fill_tick": _integer(row.get("fill_tick")),
                "entry_signal_active_at_fill": _boolean(row.get("entry_signal_active_at_fill")),
                "entry_signal_persisted_until_fill": _boolean(
                    row.get("entry_signal_persisted_until_fill")),
                "entry_signal_active_share_until_fill": _number(
                    row.get("entry_signal_active_share_until_fill")),
                "entry_status": str(row.get("status")),
                "exit_reason": _string(row.get("exit_reason")),
                "holding_seconds": _number(row.get("holding_seconds")),
                "net_bps": _number(row.get("net_bps")),
                "gross_bps": _number(row.get("gross_bps")),
                "max_favorable_gross_bps": _number(row.get("max_favorable_gross_bps")),
                "max_adverse_gross_bps": _number(row.get("max_adverse_gross_bps")),
                "pre_entry_features": {feature: _number(row.get(feature)) for feature in features},
            })
    return cases


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _string(value: Any) -> str | None:
    return str(value) if value is not None and not pd.isna(value) else None
