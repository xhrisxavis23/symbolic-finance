"""저장된 Feature Profile anchor의 canonical 실행 결과를 읽는 Discovery Stage.

Feature Profile의 PROFIT/LOSS는 BID1→미래 최고 ASK1 가격 기회로 만든 oracle label이다.
이 모듈은 그 Profile을 고치지 않고, 각 anchor tick에서 실제 canonical BID1 queue
진입과 고정 exit을 한 번씩 독립 재생해 그 차이를 artifact로 남긴다.

anchor는 서로 다른 가설의 signal이 아니라 이미 선택된 관찰 지점이다. 따라서 여기의
합산 손익은 전략 성과가 아니며, Discovery Evidence가 실행을 뜻한다고 잘못 읽지 않게
하는 진단 입력으로만 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import canonical as K
from ..config import TICK_ROOT, clean, execution_workers, sha256_json, write_json
from ..ledger import CENSORED, FILLED, UNFILLED
from ..outcome import EARLY_RECOVERY_LATE_REVERSAL, NET_RECOVERY, PERSISTENT_ADVERSE
from . import profile
from .profile import ProfileSelection


SCHEMA_VERSION = "execution_anchor_replay.v2"
# 원 Profile의 PROFIT/LOSS anchor는 미래 가격 기회로 이미 고른 표본이다. 실제 체결
# 결과를 Evidence label로 바꿔도 그 선택은 사라지지 않는다. 실행 Evidence의 주 입력은
# 같은 시간대에 outcome으로 직접 고르지 않은 Profile BACKGROUND anchor만 쓴다.
EXECUTION_EVIDENCE_SOURCE_PROFILE_COHORTS = ("BACKGROUND",)
STOP_EXIT_REASONS = frozenset({"stop_bid1", "stop_bid1_after_trailing"})


def input_signature(selection: ProfileSelection) -> dict[str, Any]:
    """실제 실행 대상인 anchor 식별자만으로 cache 입력을 묶는다."""
    records = anchor_records(selection)
    return {
        "selection": clean(selection.as_input()),
        "anchor_count": len(records),
        "anchor_sha256": sha256_json(records),
    }


def anchor_records(selection: ProfileSelection) -> list[dict[str, Any]]:
    """공유 cache 조합에도 쓰는 저장 anchor의 결정적 식별자 목록."""
    _long, anchors = profile.load(selection)
    return _anchor_records(anchors)


def run(selection: ProfileSelection, output: Path, *, root: Path = TICK_ROOT,
        workers: int | None = None) -> dict[str, Any]:
    """각 저장 anchor를 하나의 독립 canonical 주문으로 재생한다."""
    workers = execution_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError("execution anchor worker 수는 1 이상이어야 한다")
    _long, anchors = profile.load(selection)
    records = _anchor_records(anchors)
    signature = {
        "selection": clean(selection.as_input()),
        "anchor_count": len(records),
        "anchor_sha256": sha256_json(records),
    }
    output = Path(output)
    ledger_root = output / "ledgers"
    ledger_root.mkdir(parents=True, exist_ok=True)

    indexed = _indexed_anchors(anchors)
    ledgers: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for date, group in indexed.groupby("date", sort=True):
        contracts, members = _anchor_contracts(group)
        ledger_path = ledger_root / f"{date}.parquet"
        manifest = K.run_backtest(
            contracts, members, [str(date)], ledger_path, workers=workers, root=Path(root),
            # `diagnostic_frozen_entry_ticks` is intentionally not an executable contract.
            # The Stage observes a stored anchor; candidate contracts stay strict elsewhere.
            strict_contract=False,
        )
        manifests.append(_manifest_summary(manifest, date=str(date), ledger_path=ledger_path))
        for item in [*(manifest.get("missing") or []), *(manifest.get("errors") or [])]:
            contract_id = str(item.get("contract_id") or "")
            if contract_id:
                unavailable[contract_id] = str(item.get("reason") or item.get("error") or "unavailable")
        frame = pd.read_parquet(ledger_path)
        if len(frame):
            ledgers.append(frame)

    replayed = (pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame())
    outcomes = _outcomes(indexed, replayed, unavailable)
    outcomes_path = output / "anchor_execution_outcomes.parquet"
    outcomes.to_parquet(outcomes_path, index=False)
    summary = _summary(outcomes, manifests)
    write_json(output / "execution_anchor_summary.json", summary)
    return {
        "schema": SCHEMA_VERSION,
        "input": signature,
        "canonical_backtest_profile_id": K.CANONICAL["profile_id"],
        "canonical_backtest_profile_hash": K.CANONICAL["profile_sha256"],
        "execution_definition": {
            "entry": "stored anchor tick에서 canonical BID1 queue entry",
            "exit": "unchanged canonical fixed exit",
            "independence": "각 anchor를 독립 주문으로 재생; anchor 사이 position blocking을 공유하지 않음",
            "use": "Discovery execution-alignment diagnostic only",
            "not_a_strategy_pnl": True,
        },
        "summary": summary,
        "outcomes_path": str(outcomes_path),
        "ledger_manifests": manifests,
    }


def execution_labelled_profile(
        selection: ProfileSelection, replay: Mapping[str, Any], *,
        source_profile_cohorts: tuple[str, ...] = EXECUTION_EVIDENCE_SOURCE_PROFILE_COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """저장 Profile을 고치지 않고, outcome으로 고르지 않은 source 표본에 실행 label을 붙인다.

    세 cohort 이름은 기존 Evidence 계산기가 요구하는 어휘를 재사용한다. 이때 결과
    BACKGROUND는 일반 시장 상태가 아니라 `ENTRY_UNFILLED` anchor라는 점을 context에
    명시한다. 직접 후보 선택은 계속 실제 실행 PROFIT/LOSS 대조만 사용한다.
    """
    signature = input_signature(selection)
    if replay.get("input") != signature:
        raise ValueError("Execution replay와 execution Evidence의 Feature Profile 입력이 다르다")
    outcome_path = Path(str(replay.get("outcomes_path") or ""))
    if not outcome_path.is_file():
        raise FileNotFoundError("Execution replay의 anchor outcome 원장이 없다")
    outcomes = pd.read_parquet(outcome_path)
    needed = {"anchor_id", "profile_cohort", "execution_label"}
    if not needed <= set(outcomes):
        raise ValueError("Execution replay outcome 원장에 execution label이 없다")
    source_cohorts = tuple(dict.fromkeys(str(value) for value in source_profile_cohorts))
    if not source_cohorts:
        raise ValueError("Execution Evidence의 원 Profile cohort가 비었다")
    label_map = {
        "EXECUTED_PROFIT": "PROFIT",
        "EXECUTED_LOSS": "LOSS",
        "ENTRY_UNFILLED": "BACKGROUND",
    }
    source_outcomes = outcomes.loc[outcomes["profile_cohort"].isin(source_cohorts)].copy()
    labels = source_outcomes.loc[source_outcomes["execution_label"].isin(label_map),
                          ["anchor_id", "profile_cohort", "execution_label"]].copy()
    if labels["anchor_id"].duplicated().any():
        raise ValueError("Execution replay outcome 원장에 anchor가 중복됐다")
    labels["cohort"] = labels["execution_label"].map(label_map)
    long, anchors = profile.load(selection)
    anchors = anchors.drop(columns=["cohort"]).merge(
        labels[["anchor_id", "profile_cohort", "cohort"]], on="anchor_id", how="inner")
    long = long.drop(columns=["cohort"]).merge(labels[["anchor_id", "cohort"]],
                                                 on="anchor_id", how="inner")
    long = long.merge(labels[["anchor_id", "profile_cohort"]], on="anchor_id", how="inner")
    counts = anchors["cohort"].value_counts().to_dict()
    # 이 결과는 주 Evidence가 아니라 실행 진단이다. 예컨대 모든 queue 체결이 손실일
    # 수 있으며, 그때 PROFIT label이 없다는 사실이 저장 Profile Evidence 자체를
    # 만들지 못하게 해서는 안 된다. 불완전한 실행 contrast는 그대로 표시하고,
    # 실행 변화량 항목은 방향 미결정으로 남는다.
    complete = set(counts) == {"PROFIT", "LOSS", "BACKGROUND"}
    context = {
        "state": ("EXECUTION_EVIDENCE_READY" if complete
                  else "EXECUTION_EVIDENCE_INCOMPLETE"),
        "label_source": "canonical_execution_anchor_replay",
        "source_profile_cohorts": list(source_cohorts),
        "source_anchor_count": int(len(source_outcomes)),
        "source_execution_label_counts": {
            str(key): int(value)
            for key, value in source_outcomes["execution_label"].value_counts().sort_index().items()},
        "profit_rule": "canonical order filled and net_bps > 0",
        "loss_rule": "canonical order filled and net_bps <= 0",
        "background_rule": (
            "source is original Profile BACKGROUND; within that source, canonical entry order was unfilled"),
        "original_profile_preserved": True,
        "execution_label_counts": {str(key): int(value)
                                   for key, value in outcomes["execution_label"].value_counts().sort_index().items()},
        "evidence_cohort_counts": {str(key): int(value) for key, value in counts.items()},
        "interpretation_boundary": (
            "이 실행 label은 Discovery anchor별 독립 canonical replay에서 온다. "
            "원 Profile의 BACKGROUND는 수익/손실 결과로 선택하지 않은 시간대 매칭 표본이며, "
            "그 표본 안에서만 canonical 체결 결과를 PROFIT/LOSS/미체결 BACKGROUND로 바꿨다. "
            "Feature는 anchor 시점 이전 관측만 쓰지만, label은 미래 실행 결과다."
        ),
    }
    return long.reset_index(drop=True), anchors.reset_index(drop=True), context


def execution_stop_avoidance_profile(
        selection: ProfileSelection, replay: Mapping[str, Any], *,
        source_profile_cohorts: tuple[str, ...] = EXECUTION_EVIDENCE_SOURCE_PROFILE_COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """고정 청산의 즉시 손절 여부와 anchor 시점 상태를 Discovery에서만 연결한다.

    이 label은 청산 규칙을 고르거나 바꾸지 않는다. 이미 고정된 canonical exit에서
    채결된 주문이 `stop_bid1` 계열로 끝났는지만 보고, 다음 entry 상태가 그 손실 경로를
    피할 수 있는지 후속 Search가 반증할 수 있게 만든다.
    """
    signature = input_signature(selection)
    if replay.get("input") != signature:
        raise ValueError("Execution replay와 stop-avoidance Evidence의 Feature Profile 입력이 다르다")
    outcome_path = Path(str(replay.get("outcomes_path") or ""))
    if not outcome_path.is_file():
        raise FileNotFoundError("Execution replay outcome 원장이 없다")
    outcomes = pd.read_parquet(outcome_path)
    needed = {"anchor_id", "profile_cohort", "execution_label", "exit_reason"}
    if not needed <= set(outcomes):
        raise ValueError("Execution replay outcome 원장에 stop-avoidance label 입력이 없다")
    source_cohorts = tuple(dict.fromkeys(str(value) for value in source_profile_cohorts))
    if not source_cohorts:
        raise ValueError("Stop-avoidance Evidence의 원 Profile cohort가 비었다")
    source = outcomes.loc[
        outcomes["profile_cohort"].isin(source_cohorts)
        & outcomes["execution_label"].isin(["EXECUTED_PROFIT", "EXECUTED_LOSS"])
        & outcomes["exit_reason"].notna(),
        ["anchor_id", "profile_cohort", "exit_reason"],
    ].copy()
    if source["anchor_id"].duplicated().any():
        raise ValueError("Execution replay outcome 원장에 anchor가 중복됐다")
    source["stop_avoidance_label"] = np.where(
        source["exit_reason"].astype(str).isin(STOP_EXIT_REASONS), "STOP", "NON_STOP")
    # 기존 Evidence 계산기의 두 cohort 어휘를 재사용할 뿐, NON_STOP을 수익이라고
    # 부르지 않는다. label의 실제 뜻은 context와 후속 Evidence artifact에 남긴다.
    source["cohort"] = np.where(source["stop_avoidance_label"].eq("NON_STOP"),
                                "PROFIT", "LOSS")
    long, anchors = profile.load(selection)
    labels = source[["anchor_id", "profile_cohort", "cohort", "stop_avoidance_label"]]
    anchors = anchors.drop(columns=["cohort"]).merge(
        labels[["anchor_id", "profile_cohort", "cohort", "stop_avoidance_label"]],
        on="anchor_id", how="inner")
    long = long.drop(columns=["cohort"]).merge(
        labels[["anchor_id", "cohort", "stop_avoidance_label"]], on="anchor_id", how="inner")
    counts = source["stop_avoidance_label"].value_counts().sort_index()
    context = {
        "state": ("EXECUTION_STOP_AVOIDANCE_EVIDENCE_READY"
                  if set(counts.index) == {"STOP", "NON_STOP"}
                  else "EXECUTION_STOP_AVOIDANCE_EVIDENCE_INCOMPLETE"),
        "label_source": "canonical_execution_anchor_replay",
        "source_profile_cohorts": list(source_cohorts),
        "source_anchor_count": int(len(source)),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "stop_exit_reasons": sorted(STOP_EXIT_REASONS),
        "positive_label": "NON_STOP",
        "negative_label": "STOP",
        "interpretation_boundary": (
            "NON_STOP은 수익 label이 아니다. 고정 canonical exit에서 즉시 손절 계열로 "
            "끝나지 않은 체결일 뿐이며, 이 대조는 Discovery entry-harm 진단으로만 쓴다. "
            "청산 규칙은 그대로다."),
    }
    return long.reset_index(drop=True), anchors.reset_index(drop=True), context


def execution_recovery_path_profile(
        selection: ProfileSelection, replay: Mapping[str, Any], *,
        source_profile_cohorts: tuple[str, ...] = EXECUTION_EVIDENCE_SOURCE_PROFILE_COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """고정 청산과 분리한 원시 30초 가격경로의 회복/지속악화 대조를 만든다.

    ``diagnostic_cohort``는 canonical 주문의 exit reason이나 실현 손익을 쓰지 않는다.
    실제 queue 체결 뒤 entry price 대비 30초 BID1 경로만 분류한 진단 label이므로,
    이 함수는 다음 진입 상태를 찾는 Discovery 입력일 뿐 exit 학습이 아니다.
    """
    return _raw_path_profile(
        selection, replay, source_profile_cohorts=source_profile_cohorts,
        positive_label=NET_RECOVERY, negative_label=PERSISTENT_ADVERSE,
        state_prefix="EXECUTION_RECOVERY_PATH_EVIDENCE",
        name="Recovery-path",
        interpretation=(
            "NET_RECOVERY/PERSISTENT_ADVERSE는 체결 뒤 entry price 대비 30초 BID1 raw path의 "
            "endpoint와 경로만으로 정한 진단 label이다. canonical exit, exit reason, 실현 PnL은 "
            "이 label에 쓰지 않으며, 이 대조는 Discovery entry 상태 관측으로만 쓴다. "
            "청산 규칙은 그대로다."),
    )


def execution_early_reversal_path_profile(
        selection: ProfileSelection, replay: Mapping[str, Any], *,
        source_profile_cohorts: tuple[str, ...] = EXECUTION_EVIDENCE_SOURCE_PROFILE_COHORTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """초기 순회복 뒤 30초 내 재하락과 지속 악화를 분리한 Discovery 대조다.

    첫 양수 경로를 고정 exit 결과로 바꾸지 않는다. 체결 뒤 30초 BID1 raw path에서
    비용을 넘긴 지점이 한 번 있었지만 endpoint는 음수인 경우만 양성 label로 쓴다.
    고정 trailing exit이 그 임시 반등을 실제로 포착하는지는 후속 Backtest가 판정한다.
    """
    return _raw_path_profile(
        selection, replay, source_profile_cohorts=source_profile_cohorts,
        positive_label=EARLY_RECOVERY_LATE_REVERSAL, negative_label=PERSISTENT_ADVERSE,
        state_prefix="EXECUTION_EARLY_REVERSAL_PATH_EVIDENCE",
        name="Early-reversal-path",
        interpretation=(
            "EARLY_RECOVERY_LATE_REVERSAL/PERSISTENT_ADVERSE는 체결 뒤 entry price 대비 30초 "
            "BID1 raw path만으로 정한 진단 label이다. 양성은 30초 끝 전에는 비용을 넘겨 "
            "회복했지만 endpoint는 음수였던 경우다. canonical exit, exit reason, 실현 PnL은 "
            "이 label에 쓰지 않으며, 이 대조는 고정 trailing exit이 포착할 수 있는 entry "
            "상태를 Discovery에서 관측할 뿐 청산 규칙을 학습하지 않는다."),
    )


def _raw_path_profile(
        selection: ProfileSelection, replay: Mapping[str, Any], *,
        source_profile_cohorts: tuple[str, ...], positive_label: str, negative_label: str,
        state_prefix: str, name: str, interpretation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """동일한 raw path 두 cohort를 Feature Profile 행에 안전하게 붙인다."""
    signature = input_signature(selection)
    if replay.get("input") != signature:
        raise ValueError(f"Execution replay와 {name} Evidence의 Feature Profile 입력이 다르다")
    outcome_path = Path(str(replay.get("outcomes_path") or ""))
    if not outcome_path.is_file():
        raise FileNotFoundError("Execution replay outcome 원장이 없다")
    outcomes = pd.read_parquet(outcome_path)
    needed = {"anchor_id", "profile_cohort", "execution_label"}
    if not needed <= set(outcomes):
        raise ValueError(f"Execution replay outcome 원장에 {name} label 입력이 없다")
    source_cohorts = tuple(dict.fromkeys(str(value) for value in source_profile_cohorts))
    if not source_cohorts:
        raise ValueError(f"{name} Evidence의 원 Profile cohort가 비었다")
    long, anchors = profile.load(selection)
    if "diagnostic_cohort" not in outcomes:
        return long.iloc[0:0].copy(), anchors.iloc[0:0].copy(), {
            "state": f"{state_prefix}_UNAVAILABLE",
            "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
            "source_profile_cohorts": list(source_cohorts),
            "reason": "execution anchor replay outcome에 diagnostic_cohort가 없다",
            "original_profile_preserved": True,
            "interpretation_boundary": (
                "이 replay artifact는 원시 30초 경로 label을 보존하기 전 버전이다. "
                "기존 실행/청산 Evidence는 그대로 쓸 수 있지만 raw-path Evidence는 만들지 않는다."),
        }
    source = outcomes.loc[
        outcomes["profile_cohort"].isin(source_cohorts)
        & outcomes["execution_label"].isin(["EXECUTED_PROFIT", "EXECUTED_LOSS"])
        & outcomes["diagnostic_cohort"].isin([positive_label, negative_label]),
        ["anchor_id", "profile_cohort", "diagnostic_cohort"],
    ].copy()
    if source["anchor_id"].duplicated().any():
        raise ValueError("Execution replay outcome 원장에 anchor가 중복됐다")
    source["cohort"] = np.where(source["diagnostic_cohort"].eq(positive_label),
                                 "PROFIT", "LOSS")
    labels = source[["anchor_id", "profile_cohort", "cohort", "diagnostic_cohort"]]
    anchors = anchors.drop(columns=["cohort"]).merge(labels, on="anchor_id", how="inner")
    long = long.drop(columns=["cohort"]).merge(
        labels[["anchor_id", "cohort", "diagnostic_cohort"]], on="anchor_id", how="inner")
    counts = source["diagnostic_cohort"].value_counts().sort_index()
    context = {
        "state": (f"{state_prefix}_READY"
                  if set(counts.index) == {positive_label, negative_label}
                  else f"{state_prefix}_INCOMPLETE"),
        "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
        "source_profile_cohorts": list(source_cohorts),
        "source_anchor_count": int(len(source)),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "positive_label": positive_label,
        "negative_label": negative_label,
        "original_profile_preserved": True,
        "interpretation_boundary": interpretation,
    }
    return long.reset_index(drop=True), anchors.reset_index(drop=True), context


def _anchor_records(anchors: pd.DataFrame) -> list[dict[str, Any]]:
    required = ("anchor_id", "cohort", "cluster_id", "symbol", "date", "tick")
    missing = [name for name in required if name not in anchors]
    if missing:
        raise ValueError(f"Execution anchor replay에 필요한 Profile 열이 없다: {missing}")
    frame = anchors.loc[:, list(required)].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["date"] = frame["date"].astype(str)
    frame["tick"] = pd.to_numeric(frame["tick"], errors="raise").astype(int)
    return [{key: (int(value) if key == "tick" else str(value)) for key, value in row.items()}
            for row in frame.sort_values(["cluster_id", "symbol", "date", "tick", "cohort", "anchor_id"])
            .to_dict(orient="records")]


def _indexed_anchors(anchors: pd.DataFrame) -> pd.DataFrame:
    records = _anchor_records(anchors)
    frame = pd.DataFrame(records)
    frame.insert(0, "contract_id", [f"EAR_{index:06d}" for index in range(len(frame))])
    if frame["anchor_id"].duplicated().any():
        raise ValueError("Feature Profile anchor_id가 중복됐다")
    return frame


def _anchor_contracts(group: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    contracts: dict[str, dict[str, Any]] = {}
    members: dict[str, list[str]] = {}
    for row in group.itertuples(index=False):
        contract_id = str(row.contract_id)
        contracts[contract_id] = {
            "hypothesis_id": contract_id,
            "diagnostic_frozen_entry_ticks": [int(row.tick)],
            "diagnostic_anchor_id": str(row.anchor_id),
        }
        members[contract_id] = [str(row.symbol).zfill(6)]
    return contracts, members


def _manifest_summary(manifest: Mapping[str, Any], *, date: str, ledger_path: Path) -> dict[str, Any]:
    return {
        "date": str(date), "ledger_path": str(ledger_path),
        "rows": int(manifest.get("rows") or 0),
        "totals": dict(manifest.get("totals") or {}),
        "missing_count": len(manifest.get("missing") or []),
        "error_count": len(manifest.get("errors") or []),
    }


def _outcomes(indexed: pd.DataFrame, replayed: pd.DataFrame,
              unavailable: Mapping[str, str]) -> pd.DataFrame:
    rows = indexed.copy()
    columns = ["contract_id", "status", "fill_tick", "exit_tick", "entry_price", "net_bps",
               "gross_bps", "exit_reason", "holding_seconds", "diagnostic_cohort",
               "max_favorable_gross_bps", "max_adverse_gross_bps"]
    available = [name for name in columns if name in replayed]
    ledger_rows = replayed.loc[:, available].copy() if available else pd.DataFrame(columns=columns)
    if len(ledger_rows) and ledger_rows["contract_id"].duplicated().any():
        raise ValueError("anchor replay contract당 decision이 둘 이상이다")
    rows = rows.merge(ledger_rows, on="contract_id", how="left")
    rows = rows.rename(columns={"cohort": "profile_cohort", "status": "ledger_status"})
    rows["execution_label"] = rows.apply(
        lambda row: _execution_label(row.get("ledger_status"), row.get("net_bps"),
                                     str(row["contract_id"]) in unavailable), axis=1)
    rows["unavailable_reason"] = rows["contract_id"].map(dict(unavailable))
    return rows.sort_values(["cluster_id", "symbol", "date", "tick", "profile_cohort"]).reset_index(drop=True)


def _execution_label(status: Any, net_bps: Any, unavailable: bool) -> str:
    if unavailable or pd.isna(status):
        return "DATA_UNAVAILABLE"
    if str(status) == FILLED:
        value = float(net_bps)
        return "EXECUTED_PROFIT" if np.isfinite(value) and value > 0.0 else "EXECUTED_LOSS"
    if str(status) == UNFILLED:
        return "ENTRY_UNFILLED"
    if str(status) == CENSORED:
        return "EXECUTION_CENSORED"
    return "DATA_UNAVAILABLE"


def _summary(outcomes: pd.DataFrame, manifests: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = outcomes["execution_label"].value_counts().sort_index()
    filled = outcomes.loc[outcomes["execution_label"].isin(["EXECUTED_PROFIT", "EXECUTED_LOSS"])].copy()
    net = pd.to_numeric(filled.get("net_bps"), errors="coerce")
    counts = {str(key): int(value) for key, value in labels.items()}
    by_profile = (outcomes.groupby(["profile_cohort", "execution_label"]).size().unstack(fill_value=0)
                  if len(outcomes) else pd.DataFrame())
    return {
        "anchor_count": int(len(outcomes)),
        "execution_label_counts": counts,
        "filled_anchor_count": int(len(filled)),
        "filled_anchor_rate": (float(len(filled) / len(outcomes)) if len(outcomes) else None),
        "executed_profit_rate_among_fills": (
            float((filled["execution_label"] == "EXECUTED_PROFIT").mean()) if len(filled) else None),
        "filled_net_bps": {
            "mean": _number(net.mean()), "median": _number(net.median()),
            "positive_count": int((net > 0.0).sum()), "nonpositive_count": int((net <= 0.0).sum()),
        },
        "profile_cohort_by_execution_label": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in by_profile.iterrows()
        },
        "ledger_dates": [dict(value) for value in manifests],
        "interpretation_boundary": (
            "저장 Feature Profile anchor마다 독립 canonical 주문을 재생한 Discovery 진단이다. "
            "anchor들이 서로 position blocking을 공유하지 않으므로 순손익 합계·평균은 "
            "실행 가능한 전략의 PnL, Validation, Final 또는 OOS 증거가 아니다."
        ),
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
