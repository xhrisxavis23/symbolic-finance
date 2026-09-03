"""저장된 FeatureProfile에서 Evidence와 Joint Evidence를 만든다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import catalog, evidence as evidence_module, joint, sample_condition as SC
from ..config import read_json, write_json
from . import execution_anchor
from .profile import ProfileSelection, load


def build(selection: ProfileSelection, output: Path, *,
          execution_anchor_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """기존 계산을 재사용하되, 입력 선택과 산출 위치를 호출자가 준다."""
    output = Path(output)
    stored_long, stored_anchors = load(selection)
    # 주 Evidence는 저장된 Feature Profile의 cohort를 그대로 쓴다. canonical
    # 실행 replay는 anchor별 체결 가능성을 보는 진단이며, 그것을 다시 cohort label로
    # 바꾸면 동일 Feature Profile의 Evidence와 직접 후보가 실행 모드에 따라 달라진다.
    long, anchors = stored_long, stored_anchors
    execution_long = None
    execution_context = None
    execution_stop_long = None
    execution_stop_context = None
    execution_recovery_long = None
    execution_recovery_context = None
    execution_early_reversal_long = None
    execution_early_reversal_context = None
    if execution_anchor_replay is not None:
        execution_long, _execution_anchors, execution_context = (
            execution_anchor.execution_labelled_profile(selection, execution_anchor_replay))
        execution_stop_long, _execution_stop_anchors, execution_stop_context = (
            execution_anchor.execution_stop_avoidance_profile(selection, execution_anchor_replay))
        execution_recovery_long, _execution_recovery_anchors, execution_recovery_context = (
            execution_anchor.execution_recovery_path_profile(selection, execution_anchor_replay))
        execution_early_reversal_long, _execution_early_reversal_anchors, execution_early_reversal_context = (
            execution_anchor.execution_early_reversal_path_profile(selection, execution_anchor_replay))
    evidence_result = evidence_module.build(
        output / "evidence", profile=(long, anchors))
    joint_result = joint.build(output / "evidence", output / "joint")
    # Profile과 같은 저장 단위에 만든 discovery profit cache가 있으면 Agent의 읽기 전용
    # 가격 경로 tool도 그 cache를 써야 한다. 전역 기본 cache를 추측하면 profile의 anchor와
    # 다른 label을 읽을 수 있다.
    profile_store = Path(selection.store)
    discovery_profit_root = (
        Path(selection.discovery_profit_root)
        if selection.discovery_profit_root is not None else profile_store / "profit")
    package = dict(joint_result["package"])
    if str(selection.profile_kind) == "execution_aligned":
        contract_path = profile_store / "execution_profile_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError("execution-aligned Profile contract가 없다: " + str(contract_path))
        execution_contract = read_json(contract_path)
        sample_condition = SC.require_for_event_source(
            execution_contract, "execution-aligned Feature Profile")
        package["outcome_definition"] = {
            "profile_label": "execution-aligned canonical queue outcome",
            "entry": execution_contract.get("entry"),
            "exit": execution_contract.get("exit"),
            "cost_bps": execution_contract.get("cost_bps"),
            "cohorts": execution_contract.get("cohorts"),
            "cohort_rule": execution_contract.get("cohort_rule"),
            "boundary": "future execution labels are Discovery-only; they are not strategy PnL",
        }
        package["execution_aligned_profile"] = execution_contract
        package["executable_sample_condition"] = sample_condition
        package["executable_sample_condition_sha256"] = SC.condition_hash(sample_condition)
    package["profile_store"] = str(profile_store)
    package["discovery_profit_root"] = (
        str(discovery_profit_root) if discovery_profit_root.is_dir() else None)
    # 가격 경로는 cache가 같은 가격 방향일 때 먼저 읽고, 없거나 반대 방향인 옛 cache면
    # 저장된 anchor의 원본 quote 창을 무저장으로 읽는다. 어느 쪽도 cohort를 새로 만들지 않는다.
    package["price_path_context_required"] = True
    package["profile_compatibility"] = {
        "profile_kind": str(selection.profile_kind),
        "missing_outcome_columns": list(stored_long.attrs.get("missing_outcome_columns", [])),
        "note": "없는 보조 outcome 열은 결측으로 유지했다. feature·cohort 증거를 바꾸지 않는다.",
        "entry_execution_alignment": entry_execution_alignment(
            stored_anchors, missing_outcome_columns=stored_long.attrs.get("missing_outcome_columns", [])),
    }
    if execution_anchor_replay is not None:
        # 실제 queue fill과 fixed exit 결과는 별도의 진단이다. 원래 Profile cohort를
        # 바꾸지 않아야 Evidence와 직접 후보가 저장 입력에 대해 결정적이다.
        assert execution_context is not None
        execution_context = {
            **execution_context,
            "source_profile_cohort_contrasts": _mark_profile_selection_confound(
                package, execution_long),
        }
        package["execution_evidence"] = execution_context
        package["execution_state_evidence"] = _execution_state_evidence(execution_long)
        package["execution_fill_evidence"] = _execution_fill_evidence(execution_long)
        package["execution_temporal_evidence"] = _execution_temporal_evidence(execution_long)
        assert execution_stop_long is not None and execution_stop_context is not None
        package["execution_stop_avoidance_evidence"] = _execution_stop_avoidance_evidence(
            execution_stop_long, execution_stop_context)
        assert execution_recovery_context is not None
        package["execution_recovery_path_evidence"] = (
            _execution_recovery_path_evidence(execution_recovery_long, execution_recovery_context)
            if execution_recovery_context["state"] == "EXECUTION_RECOVERY_PATH_EVIDENCE_READY"
            else _unavailable_execution_recovery_path_evidence(execution_recovery_context))
        package["execution_recovery_path_temporal_evidence"] = (
            _execution_recovery_path_temporal_evidence(
                execution_recovery_long, execution_recovery_context)
            if execution_recovery_context["state"] == "EXECUTION_RECOVERY_PATH_EVIDENCE_READY"
            else _unavailable_execution_recovery_path_temporal_evidence(execution_recovery_context))
        assert execution_early_reversal_context is not None
        package["execution_early_reversal_path_evidence"] = (
            _execution_early_reversal_path_evidence(
                execution_early_reversal_long, execution_early_reversal_context)
            if execution_early_reversal_context["state"]
            == "EXECUTION_EARLY_REVERSAL_PATH_EVIDENCE_READY"
            else _unavailable_execution_early_reversal_path_evidence(
                execution_early_reversal_context))
        package["not_provided"] = [
            *list(package.get("not_provided") or []),
            "aggregated execution-anchor replay PnL", "Validation/Final/OOS execution evidence",
        ]
        package["canonical_execution_anchor_replay"] = {
            "schema": execution_anchor_replay.get("schema"),
            "canonical_backtest_profile_id": execution_anchor_replay.get("canonical_backtest_profile_id"),
            "canonical_backtest_profile_hash": execution_anchor_replay.get("canonical_backtest_profile_hash"),
            "execution_definition": execution_anchor_replay.get("execution_definition"),
            "summary": execution_anchor_replay.get("summary"),
            "outcomes_path": execution_anchor_replay.get("outcomes_path"),
        }
        # `joint.build()`가 읽은 원본 package도 Artifact와 같은 label 정의를 가리키게
        # 한다. 그렇지 않으면 디스크 Evidence와 Agent Evidence가 서로 다른 말을 한다.
        write_json(output / "evidence" / "hypothesis_evidence.json", package)
    return {
        "package": package,
        "selection": selection.as_input(),
        "profile_dates": sorted(str(value) for value in stored_anchors["date"].unique()),
        "profile_symbols": sorted(str(value) for value in stored_anchors["symbol"].unique()),
        "evidence_output": "evidence",
        "joint_output": "joint",
        "profile_rows": int(len(stored_long)),
        "anchor_count": int(len(stored_anchors)),
        "evidence": {key: value for key, value in evidence_result.items()
                     if key in ("catalog_sha256", "features", "anchors", "notes", "warnings", "tables")},
        "joint": {key: value for key, value in joint_result.items()
                  if key in ("counts", "leakage")},
        "execution_anchor_replay": (
            {"state": "OPENED", "summary": execution_anchor_replay.get("summary")}
            if execution_anchor_replay is not None else {"state": "NOT_REQUESTED"}),
    }


def _mark_profile_selection_confound(package: dict[str, Any], long: pd.DataFrame) -> list[dict[str, Any]]:
    """원 oracle cohort를 섞어서만 생긴 execution 효과를 Evidence에 표시한다."""
    if "profile_cohort" not in long:
        return []
    at0 = long.loc[(long["relative_time_ms"] == 0)
                   & long["cohort"].isin(["PROFIT", "LOSS"])].copy()
    rows: list[dict[str, Any]] = []
    by_feature: dict[str, dict[str, Any]] = {}
    for item in package.get("evidence_families") or []:
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        if feature not in at0:
            continue
        contrasts = []
        signs: set[str] = set()
        for source, group in at0.groupby("profile_cohort", sort=True):
            profit = group.loc[group["cohort"].eq("PROFIT"), feature].to_numpy(float)
            loss = group.loc[group["cohort"].eq("LOSS"), feature].to_numpy(float)
            auc = evidence_module._auc(profit, loss)
            direction = ("UNDETERMINED" if not pd.notna(auc) else
                         "higher_in_PROFIT" if float(auc) > 0.5 else
                         "lower_in_PROFIT" if float(auc) < 0.5 else "no_separation")
            if direction in {"higher_in_PROFIT", "lower_in_PROFIT"}:
                signs.add(direction)
            contrasts.append({"profile_cohort": str(source),
                              "profit_anchors": int(np.isfinite(profit).sum()),
                              "loss_anchors": int(np.isfinite(loss).sum()),
                              "auc": _finite_number(auc), "direction": direction})
        confounded = len(signs) > 1
        row = {"feature": feature, "profile_cohort_contrasts": contrasts,
               "profile_selection_confound": confounded}
        rows.append(row)
        by_feature[feature] = row
    for item in package.get("evidence_families") or []:
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        detail = by_feature.get(feature)
        if detail is None or not detail["profile_selection_confound"]:
            continue
        directions = ", ".join(
            f"{row['profile_cohort']}:{row['direction']}" for row in detail["profile_cohort_contrasts"])
        item["warnings"] = [*list(item.get("warnings") or []),
                            f"PROFILE_SELECTION_CONFOUND {directions}"]
    return rows


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _execution_state_evidence(long: pd.DataFrame) -> dict[str, Any]:
    """실제 canonical anchor replay label과 anchor 시점 상태를 대조한다.

    이 Evidence는 원 Profile의 oracle PROFIT/LOSS를 쓰지 않는다. outcome으로 고르지
    않은 BACKGROUND anchor에 붙인 개별 canonical replay label만 사용한다. 각 anchor를
    독립 주문으로 재생한 진단이므로 전략 PnL로 해석하지 않는다.
    """
    at0 = long.loc[long["relative_time_ms"].eq(0)].copy()
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in at0]
    items: list[dict[str, Any]] = []
    for feature in feature_names:
        contrast = at0.loc[at0["cohort"].isin(["PROFIT", "LOSS"])].copy()
        profit = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("PROFIT"), feature], errors="coerce").to_numpy(float)
        loss = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("LOSS"), feature], errors="coerce").to_numpy(float)
        auc = evidence_module._auc(profit, loss)
        direction = ("higher_in_PROFIT" if np.isfinite(auc) and auc > 0.5 else
                     "lower_in_PROFIT" if np.isfinite(auc) and auc < 0.5 else
                     "no_separation" if np.isfinite(auc) else "UNDETERMINED")
        coverage_profit = float(np.isfinite(profit).mean()) if len(profit) else 0.0
        coverage_loss = float(np.isfinite(loss).mean()) if len(loss) else 0.0
        gap = abs(coverage_profit - coverage_loss)
        spec = catalog.FEATURES[feature]
        items.append({
            "evidence_id": f"EVX_STATE_{feature.upper()}",
            "family": evidence_module.family_of(spec),
            "representative_feature": {"name": feature, "definition_id": spec.definition_id},
            "observation_window": {"relative_time_ms": 0,
                                   "operator": "anchor_value_at_t"},
            "snapshot_contrast": {"profit_vs_loss": {
                "auc": _finite_number(auc), "direction": direction,
                "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                "median_diff": _finite_number(np.nanmedian(profit) - np.nanmedian(loss)
                                                if np.isfinite(profit).any() and np.isfinite(loss).any()
                                                else np.nan),
            }},
            "date_stability": _state_agreement(contrast, "date", feature, direction),
            "symbol_stability": _state_agreement(contrast, "symbol", feature, direction),
            "sample_counts": {"profit": int(np.isfinite(profit).sum()),
                              "loss": int(np.isfinite(loss).sum())},
            "coverage": {"PROFIT": coverage_profit, "LOSS": coverage_loss, "gap": gap},
            "independent_evidence": True,
            "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"]
                         if gap > 0.30 else []),
        })
    return {
        "schema": "execution_state_evidence.v1",
        "source": "stored FeatureProfile anchor timepoint t=0 only",
        "label_source": "canonical_execution_anchor_replay",
        "operator": "snapshot",
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": (
            "t=0은 저장된 Feature Profile의 anchor 시점 값이다. PROFIT/LOSS는 개별 "
            "anchor canonical fixed-exit replay label이며 전략 PnL이 아니다."),
    }
 

def _execution_stop_avoidance_evidence(long: pd.DataFrame,
                                       context: Mapping[str, Any]) -> dict[str, Any]:
    """고정 exit의 즉시 손절 여부를 피할 entry 상태를 Discovery에서만 기록한다."""
    at0 = long.loc[long["relative_time_ms"].eq(0)].copy()
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in at0]
    items: list[dict[str, Any]] = []
    for feature in feature_names:
        contrast = at0.loc[at0["cohort"].isin(["PROFIT", "LOSS"])].copy()
        non_stop = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("PROFIT"), feature], errors="coerce").to_numpy(float)
        stop = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("LOSS"), feature], errors="coerce").to_numpy(float)
        auc = evidence_module._auc(non_stop, stop)
        direction = ("higher_in_NONSTOP" if np.isfinite(auc) and auc > 0.5 else
                     "lower_in_NONSTOP" if np.isfinite(auc) and auc < 0.5 else
                     "no_separation" if np.isfinite(auc) else "UNDETERMINED")
        agreement_direction = ("higher_in_PROFIT" if direction == "higher_in_NONSTOP" else
                               "lower_in_PROFIT" if direction == "lower_in_NONSTOP" else direction)
        coverage_non_stop = float(np.isfinite(non_stop).mean()) if len(non_stop) else 0.0
        coverage_stop = float(np.isfinite(stop).mean()) if len(stop) else 0.0
        gap = abs(coverage_non_stop - coverage_stop)
        spec = catalog.FEATURES[feature]
        items.append({
            "evidence_id": f"EVX_STOP_AVOID_{feature.upper()}",
            "family": evidence_module.family_of(spec),
            "representative_feature": {"name": feature, "definition_id": spec.definition_id},
            "observation_window": {"relative_time_ms": 0,
                                   "operator": "anchor_value_at_t"},
            "snapshot_contrast": {"nonstop_vs_stop": {
                "auc": _finite_number(auc), "direction": direction,
                "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                "median_diff": _finite_number(np.nanmedian(non_stop) - np.nanmedian(stop)
                                                if np.isfinite(non_stop).any() and np.isfinite(stop).any()
                                                else np.nan),
            }},
            "date_stability": _state_agreement(contrast, "date", feature, agreement_direction),
            "symbol_stability": _state_agreement(contrast, "symbol", feature, agreement_direction),
            "sample_counts": {"non_stop": int(np.isfinite(non_stop).sum()),
                              "stop": int(np.isfinite(stop).sum())},
            "coverage": {"NON_STOP": coverage_non_stop, "STOP": coverage_stop, "gap": gap},
            "independent_evidence": True,
            "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"]
                         if gap > 0.30 else []),
        })
    return {
        "schema": "execution_stop_avoidance_evidence.v1",
        "source": "stored FeatureProfile anchor timepoint t=0 only",
        "label_source": "canonical_execution_anchor_replay",
        "operator": "snapshot",
        "label_definition": dict(context),
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": (
            "NON_STOP/STOP은 고정 canonical exit의 exit reason에서 온 Discovery label이다. "
            "NON_STOP은 수익이 아니며, 이 Evidence는 즉시 손절 경로를 피할 entry 상태를 "
            "관측할 뿐 exit 정책을 학습하거나 바꾸지 않는다."),
    }


def _execution_recovery_path_evidence(long: pd.DataFrame,
                                      context: Mapping[str, Any]) -> dict[str, Any]:
    """raw 30초 경로의 회복과 지속 악화를 가르는 anchor 시점 상태를 기록한다."""
    return _raw_path_snapshot_evidence(
        long, context,
        schema="execution_recovery_path_evidence.v1",
        evidence_prefix="EVX_RECOVERY_PATH",
        contrast_key="recovery_vs_persistent_adverse",
        positive_label="NET_RECOVERY",
        negative_label="PERSISTENT_ADVERSE",
        positive_count_key="net_recovery",
        negative_count_key="persistent_adverse",
        source_description=(
            "NET_RECOVERY/PERSISTENT_ADVERSE는 canonical exit을 무시한 체결 뒤 30초 BID1 "
            "경로 진단이다. 수익/실현 PnL 또는 exit 정책 label이 아니며, 이 Evidence는 "
            "다음 entry 상태만 관측한다."),
    )


def _execution_early_reversal_path_evidence(long: pd.DataFrame,
                                            context: Mapping[str, Any]) -> dict[str, Any]:
    """초기 순회복 뒤 재하락과 지속 악화를 가르는 anchor 시점 상태를 기록한다."""
    return _raw_path_snapshot_evidence(
        long, context,
        schema="execution_early_reversal_path_evidence.v1",
        evidence_prefix="EVX_EARLY_REVERSAL_PATH",
        contrast_key="early_reversal_vs_persistent_adverse",
        positive_label="EARLY_RECOVERY_LATE_REVERSAL",
        negative_label="PERSISTENT_ADVERSE",
        positive_count_key="early_recovery_late_reversal",
        negative_count_key="persistent_adverse",
        source_description=(
            "EARLY_RECOVERY_LATE_REVERSAL/PERSISTENT_ADVERSE는 canonical exit을 무시한 체결 뒤 "
            "30초 BID1 경로 진단이다. 양성은 초기에 비용을 넘겼지만 30초 endpoint는 음수인 "
            "경우이며, 이 Evidence는 고정 trailing exit이 포착할 수 있는 다음 entry 상태만 "
            "관측한다."),
    )


def _raw_path_snapshot_evidence(
        long: pd.DataFrame, context: Mapping[str, Any], *,
        schema: str, evidence_prefix: str, contrast_key: str,
        positive_label: str, negative_label: str,
        positive_count_key: str, negative_count_key: str,
        source_description: str,
) -> dict[str, Any]:
    """두 raw path cohort의 anchor 시점 상태를 같은 Evidence 형식으로 기록한다."""
    at0 = long.loc[long["relative_time_ms"].eq(0)].copy()
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in at0]
    items: list[dict[str, Any]] = []
    for feature in feature_names:
        contrast = at0.loc[at0["cohort"].isin(["PROFIT", "LOSS"])].copy()
        recovery = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("PROFIT"), feature], errors="coerce").to_numpy(float)
        adverse = pd.to_numeric(
            contrast.loc[contrast["cohort"].eq("LOSS"), feature], errors="coerce").to_numpy(float)
        auc = evidence_module._auc(recovery, adverse)
        direction = (f"higher_in_{positive_label}" if np.isfinite(auc) and auc > 0.5 else
                     f"lower_in_{positive_label}" if np.isfinite(auc) and auc < 0.5 else
                     "no_separation" if np.isfinite(auc) else "UNDETERMINED")
        agreement_direction = ("higher_in_PROFIT" if direction == f"higher_in_{positive_label}" else
                               "lower_in_PROFIT" if direction == f"lower_in_{positive_label}" else direction)
        coverage_recovery = float(np.isfinite(recovery).mean()) if len(recovery) else 0.0
        coverage_adverse = float(np.isfinite(adverse).mean()) if len(adverse) else 0.0
        gap = abs(coverage_recovery - coverage_adverse)
        spec = catalog.FEATURES[feature]
        items.append({
            "evidence_id": f"{evidence_prefix}_{feature.upper()}",
            "family": evidence_module.family_of(spec),
            "representative_feature": {"name": feature, "definition_id": spec.definition_id},
            "observation_window": {"relative_time_ms": 0, "operator": "anchor_value_at_t"},
            "snapshot_contrast": {contrast_key: {
                "auc": _finite_number(auc), "direction": direction,
                "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                "median_diff": _finite_number(np.nanmedian(recovery) - np.nanmedian(adverse)
                                                if np.isfinite(recovery).any() and np.isfinite(adverse).any()
                                                else np.nan),
            }},
            "date_stability": _state_agreement(contrast, "date", feature, agreement_direction),
            "symbol_stability": _state_agreement(contrast, "symbol", feature, agreement_direction),
            "sample_counts": {positive_count_key: int(np.isfinite(recovery).sum()),
                              negative_count_key: int(np.isfinite(adverse).sum())},
            "coverage": {positive_label: coverage_recovery,
                         negative_label: coverage_adverse, "gap": gap},
            "independent_evidence": True,
            "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"] if gap > 0.30 else []),
        })
    return {
        "schema": schema,
        "source": "stored FeatureProfile anchor timepoint t=0 only",
        "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
        "operator": "snapshot",
        "label_definition": dict(context),
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": source_description,
    }


def _unavailable_execution_recovery_path_evidence(context: Mapping[str, Any]) -> dict[str, Any]:
    return _unavailable_raw_path_snapshot_evidence(
        context, schema="execution_recovery_path_evidence.v1",
        default_state="EXECUTION_RECOVERY_PATH_EVIDENCE_UNAVAILABLE")


def _unavailable_execution_early_reversal_path_evidence(
        context: Mapping[str, Any]) -> dict[str, Any]:
    return _unavailable_raw_path_snapshot_evidence(
        context, schema="execution_early_reversal_path_evidence.v1",
        default_state="EXECUTION_EARLY_REVERSAL_PATH_EVIDENCE_UNAVAILABLE")


def _unavailable_raw_path_snapshot_evidence(
        context: Mapping[str, Any], *, schema: str, default_state: str,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "state": str(context.get("state") or default_state),
        "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
        "operator": "snapshot",
        "label_definition": dict(context),
        "items": [],
        "interpretation_boundary": str(context.get("interpretation_boundary") or ""),
    }


def _execution_recovery_path_temporal_evidence(long: pd.DataFrame,
                                               context: Mapping[str, Any]) -> dict[str, Any]:
    """raw 30초 회복/지속악화와 저장된 사전 변화량을 연결한다.

    저장 Feature Profile의 `t-lag`와 `t`만 읽는다. 새 feature나 원시 quote 재처리는
    하지 않고, 변화 폭도 Search parameter가 아니라 각각 별도 entry timing 가설이다.
    """
    meta = ["anchor_id", "symbol", "date", "cohort"]
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in long]
    items: list[dict[str, Any]] = []
    for lag_seconds in (1, 2, 5, 10):
        start_ms, end_ms = -int(lag_seconds * 1000), 0
        profile = long.loc[long["relative_time_ms"].isin([start_ms, end_ms])].copy()
        for feature in feature_names:
            values = profile.pivot_table(index=meta, columns="relative_time_ms",
                                         values=feature, aggfunc="first")
            if start_ms not in values or end_ms not in values:
                continue
            frame = values.reset_index()
            frame["difference"] = pd.to_numeric(frame[end_ms], errors="coerce") - pd.to_numeric(
                frame[start_ms], errors="coerce")
            contrast = frame.loc[frame["cohort"].isin(["PROFIT", "LOSS"])].copy()
            recovery = contrast.loc[contrast["cohort"].eq("PROFIT"), "difference"].to_numpy(float)
            adverse = contrast.loc[contrast["cohort"].eq("LOSS"), "difference"].to_numpy(float)
            auc = evidence_module._auc(recovery, adverse)
            direction = ("higher_in_NET_RECOVERY" if np.isfinite(auc) and auc > 0.5 else
                         "lower_in_NET_RECOVERY" if np.isfinite(auc) and auc < 0.5 else
                         "no_separation" if np.isfinite(auc) else "UNDETERMINED")
            agreement_direction = ("higher_in_PROFIT" if direction == "higher_in_NET_RECOVERY" else
                                   "lower_in_PROFIT" if direction == "lower_in_NET_RECOVERY" else direction)
            coverage_recovery = float(np.isfinite(recovery).mean()) if len(recovery) else 0.0
            coverage_adverse = float(np.isfinite(adverse).mean()) if len(adverse) else 0.0
            gap = abs(coverage_recovery - coverage_adverse)
            spec = catalog.FEATURES[feature]
            items.append({
                "evidence_id": f"EVT_RECOVERY_PATH_DIFFERENCE_{lag_seconds}S_{feature.upper()}",
                "family": evidence_module.family_of(spec),
                "representative_feature": {"name": feature, "definition_id": spec.definition_id},
                "expression": {"op": "difference",
                               "input": {"op": "primitive", "primitive_id": feature},
                               "lag": lag_seconds, "time_basis": "clock"},
                "observation_window": {"from_relative_time_ms": start_ms,
                                       "to_relative_time_ms": end_ms,
                                       "operator": "anchor_value_at_t_minus_lag_to_t"},
                "snapshot_contrast": {"recovery_vs_persistent_adverse": {
                    "auc": _finite_number(auc), "direction": direction,
                    "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                    "median_diff": _finite_number(np.nanmedian(recovery) - np.nanmedian(adverse)
                                                    if np.isfinite(recovery).any() and np.isfinite(adverse).any()
                                                    else np.nan),
                }},
                "date_stability": _temporal_agreement(contrast, "date", agreement_direction),
                "symbol_stability": _temporal_agreement(contrast, "symbol", agreement_direction),
                "sample_counts": {"net_recovery": int(np.isfinite(recovery).sum()),
                                  "persistent_adverse": int(np.isfinite(adverse).sum())},
                "coverage": {"NET_RECOVERY": coverage_recovery,
                             "PERSISTENT_ADVERSE": coverage_adverse, "gap": gap},
                "independent_evidence": True,
                "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"]
                             if gap > 0.30 else []),
            })
    return {
        "schema": "execution_recovery_path_temporal_evidence.v1",
        "source": "stored FeatureProfile pre-anchor timepoints only",
        "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
        "operator": "difference", "time_basis": "clock",
        "available_lag_seconds": [1, 2, 5, 10],
        "label_definition": dict(context),
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": (
            "변화량은 저장된 Feature Profile의 사전 관측값이고, NET_RECOVERY/"
            "PERSISTENT_ADVERSE는 canonical exit을 무시한 체결 뒤 30초 BID1 raw path "
            "진단이다. 이 Evidence는 exit/PnL을 학습하지 않고 entry timing만 관측한다."),
    }


def _unavailable_execution_recovery_path_temporal_evidence(
        context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "execution_recovery_path_temporal_evidence.v1",
        "state": str(context.get("state") or "EXECUTION_RECOVERY_PATH_EVIDENCE_UNAVAILABLE"),
        "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
        "operator": "difference", "time_basis": "clock",
        "label_definition": dict(context), "items": [],
        "interpretation_boundary": str(context.get("interpretation_boundary") or ""),
    }


def _execution_fill_evidence(long: pd.DataFrame) -> dict[str, Any]:
    """같은 독립 canonical replay에서 체결과 미체결을 가르는 anchor 상태를 기록한다.

    이 항목은 수익을 뜻하지 않는다. ``FILLED``에는 수익·손실 체결을 함께 넣고,
    ``UNFILLED``은 당시 BID1 지정가가 제한 시간 안에 체결되지 않은 anchor다. 따라서
    뒤 단계가 수익 상태와 체결 가능 상태를 각각 확인한 뒤에만 둘을 결합할 수 있다.
    """
    at0 = long.loc[long["relative_time_ms"].eq(0)].copy()
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in at0]
    items: list[dict[str, Any]] = []
    for feature in feature_names:
        filled = pd.to_numeric(at0.loc[at0["cohort"].isin(["PROFIT", "LOSS"]), feature],
                               errors="coerce").to_numpy(float)
        unfilled = pd.to_numeric(at0.loc[at0["cohort"].eq("BACKGROUND"), feature],
                                 errors="coerce").to_numpy(float)
        auc = evidence_module._auc(filled, unfilled)
        direction = ("higher_in_FILLED" if np.isfinite(auc) and auc > 0.5 else
                     "lower_in_FILLED" if np.isfinite(auc) and auc < 0.5 else
                     "no_separation" if np.isfinite(auc) else "UNDETERMINED")
        coverage_filled = float(np.isfinite(filled).mean()) if len(filled) else 0.0
        coverage_unfilled = float(np.isfinite(unfilled).mean()) if len(unfilled) else 0.0
        gap = abs(coverage_filled - coverage_unfilled)
        spec = catalog.FEATURES[feature]
        items.append({
            "evidence_id": f"EVX_FILL_{feature.upper()}",
            "family": evidence_module.family_of(spec),
            "representative_feature": {"name": feature, "definition_id": spec.definition_id},
            "observation_window": {"relative_time_ms": 0,
                                   "operator": "anchor_value_at_t"},
            "snapshot_contrast": {"filled_vs_unfilled": {
                "auc": _finite_number(auc), "direction": direction,
                "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                "median_diff": _finite_number(np.nanmedian(filled) - np.nanmedian(unfilled)
                                                if np.isfinite(filled).any() and np.isfinite(unfilled).any()
                                                else np.nan),
            }},
            "date_stability": _execution_fill_agreement(at0, "date", feature, direction),
            "symbol_stability": _execution_fill_agreement(at0, "symbol", feature, direction),
            "sample_counts": {"filled": int(np.isfinite(filled).sum()),
                              "unfilled": int(np.isfinite(unfilled).sum())},
            "coverage": {"FILLED": coverage_filled, "UNFILLED": coverage_unfilled, "gap": gap},
            "independent_evidence": True,
            "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"]
                         if gap > 0.30 else []),
        })
    return {
        "schema": "execution_fill_evidence.v1",
        "source": "stored FeatureProfile anchor timepoint t=0 only",
        "label_source": "canonical_execution_anchor_replay",
        "operator": "snapshot",
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": (
            "FILLED/UNFILLED은 원 Profile BACKGROUND anchor의 독립 canonical BID1 queue "
            "replay 결과다. 체결 가능성 진단일 뿐 전략 PnL이나 수익 예측이 아니다."),
    }


def _execution_temporal_evidence(long: pd.DataFrame) -> dict[str, Any]:
    """저장된 Profile의 사전 시점 두 개로만 만든 변화량 Evidence.

    새 tick feature를 만들거나 원시 quote를 다시 읽지 않는다. 각 anchor의 저장된
    사전 시점과 `t`의 차이만 사용하며 실행 replay label은 PROFIT/LOSS 대조에만 쓴다.
    이후 계약도 같은 `difference(..., clock)` 식을 평가해야 한다.
    """
    meta = ["anchor_id", "symbol", "date", "cohort"]
    feature_names = [name for name, spec in catalog.FEATURES.items()
                     if spec.value_type == "numeric" and name in long]
    items: list[dict[str, Any]] = []
    # 저장 Profile 자체가 가진 사전 시점 네 개다. 이 시간 폭들은 Search parameter가
    # 아니며 각 폭은 별도의 entry-timing 가설이다.
    for lag_seconds in (1, 2, 5, 10):
        start_ms, end_ms = -int(lag_seconds * 1000), 0
        profile = long.loc[long["relative_time_ms"].isin([start_ms, end_ms])].copy()
        for feature in feature_names:
            values = profile.pivot_table(index=meta, columns="relative_time_ms",
                                         values=feature, aggfunc="first")
            if start_ms not in values or end_ms not in values:
                continue
            frame = values.reset_index()
            frame["difference"] = pd.to_numeric(frame[end_ms], errors="coerce") - pd.to_numeric(
                frame[start_ms], errors="coerce")
            contrast = frame.loc[frame["cohort"].isin(["PROFIT", "LOSS"])].copy()
            profit = contrast.loc[contrast["cohort"].eq("PROFIT"), "difference"].to_numpy(float)
            loss = contrast.loc[contrast["cohort"].eq("LOSS"), "difference"].to_numpy(float)
            auc = evidence_module._auc(profit, loss)
            direction = ("higher_in_PROFIT" if np.isfinite(auc) and auc > 0.5 else
                         "lower_in_PROFIT" if np.isfinite(auc) and auc < 0.5 else
                         "no_separation" if np.isfinite(auc) else "UNDETERMINED")
            date_stability = _temporal_agreement(contrast, "date", direction)
            symbol_stability = _temporal_agreement(contrast, "symbol", direction)
            coverage_profit = float(np.isfinite(profit).mean()) if len(profit) else 0.0
            coverage_loss = float(np.isfinite(loss).mean()) if len(loss) else 0.0
            gap = abs(coverage_profit - coverage_loss)
            spec = catalog.FEATURES[feature]
            items.append({
                "evidence_id": f"EVT_DIFFERENCE_{lag_seconds}S_{feature.upper()}",
                "family": evidence_module.family_of(spec),
                "representative_feature": {"name": feature, "definition_id": spec.definition_id},
                "expression": {"op": "difference",
                               "input": {"op": "primitive", "primitive_id": feature},
                               "lag": lag_seconds, "time_basis": "clock"},
                "observation_window": {"from_relative_time_ms": start_ms,
                                       "to_relative_time_ms": end_ms,
                                       "operator": "anchor_value_at_t_minus_lag_to_t"},
                "snapshot_contrast": {"profit_vs_loss": {
                    "auc": _finite_number(auc), "direction": direction,
                    "effect_strength": _finite_number(abs(auc - 0.5) if np.isfinite(auc) else np.nan),
                    "median_diff": _finite_number(np.nanmedian(profit) - np.nanmedian(loss)
                                                    if np.isfinite(profit).any() and np.isfinite(loss).any()
                                                    else np.nan),
                }},
                "date_stability": date_stability,
                "symbol_stability": symbol_stability,
                "sample_counts": {"profit": int(np.isfinite(profit).sum()),
                                  "loss": int(np.isfinite(loss).sum())},
                "coverage": {"PROFIT": coverage_profit, "LOSS": coverage_loss, "gap": gap},
                "independent_evidence": True,
                "warnings": ([f"MISSINGNESS_CONFOUND coverage_gap={gap:.2f}"]
                             if gap > 0.30 else []),
            })
    return {
        "schema": "execution_temporal_evidence.v1",
        "source": "stored FeatureProfile pre-anchor timepoints only",
        "label_source": "canonical_execution_anchor_replay",
        "operator": "difference", "time_basis": "clock",
        "available_lag_seconds": [1, 2, 5, 10],
        "items": sorted(items, key=lambda item: str(item["evidence_id"])),
        "interpretation_boundary": (
            "`t-lag`와 `t`는 저장된 Feature Profile의 사전 관측값이다. "
            "PROFIT/LOSS는 개별 anchor canonical fixed-exit replay label이며 전략 PnL이 아니다."),
    }


def _temporal_agreement(frame: pd.DataFrame, axis: str, direction: str) -> dict[str, Any]:
    """각 날짜·종목에서 pooled PROFIT/LOSS 방향이 다시 나오는 비율."""
    agrees, total = 0, 0
    for _name, group in frame.groupby(axis, sort=True):
        profit = group.loc[group["cohort"].eq("PROFIT"), "difference"].to_numpy(float)
        loss = group.loc[group["cohort"].eq("LOSS"), "difference"].to_numpy(float)
        auc = evidence_module._auc(profit, loss)
        local = ("higher_in_PROFIT" if np.isfinite(auc) and auc > 0.5 else
                 "lower_in_PROFIT" if np.isfinite(auc) and auc < 0.5 else None)
        if local is None:
            continue
        total += 1
        agrees += int(local == direction)
    return {"agreement": int(agrees), "total": int(total),
            "ratio": _finite_number(agrees / total if total else np.nan)}


def _state_agreement(frame: pd.DataFrame, axis: str, feature: str, direction: str) -> dict[str, Any]:
    """날짜·종목별 anchor 시점 상태에서도 pooled 방향이 유지되는 비율."""
    agrees, total = 0, 0
    for _name, group in frame.groupby(axis, sort=True):
        profit = pd.to_numeric(group.loc[group["cohort"].eq("PROFIT"), feature],
                               errors="coerce").to_numpy(float)
        loss = pd.to_numeric(group.loc[group["cohort"].eq("LOSS"), feature],
                             errors="coerce").to_numpy(float)
        auc = evidence_module._auc(profit, loss)
        local = ("higher_in_PROFIT" if np.isfinite(auc) and auc > 0.5 else
                 "lower_in_PROFIT" if np.isfinite(auc) and auc < 0.5 else None)
        if local is None:
            continue
        total += 1
        agrees += int(local == direction)
    return {"agreement": int(agrees), "total": int(total),
            "ratio": _finite_number(agrees / total if total else np.nan)}


def _execution_fill_agreement(frame: pd.DataFrame, axis: str, feature: str,
                              direction: str) -> dict[str, Any]:
    """각 날짜·종목에서 체결/미체결 대조 방향이 pooled 결과와 같은 비율."""
    agrees, total = 0, 0
    for _name, group in frame.groupby(axis, sort=True):
        filled = pd.to_numeric(group.loc[group["cohort"].isin(["PROFIT", "LOSS"]), feature],
                               errors="coerce").to_numpy(float)
        unfilled = pd.to_numeric(group.loc[group["cohort"].eq("BACKGROUND"), feature],
                                 errors="coerce").to_numpy(float)
        auc = evidence_module._auc(filled, unfilled)
        local = ("higher_in_FILLED" if np.isfinite(auc) and auc > 0.5 else
                 "lower_in_FILLED" if np.isfinite(auc) and auc < 0.5 else None)
        if local is None:
            continue
        total += 1
        agrees += int(local == direction)
    return {"agreement": int(agrees), "total": int(total),
            "ratio": _finite_number(agrees / total if total else np.nan)}


def entry_execution_alignment(anchors: pd.DataFrame, *,
                              missing_outcome_columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """저장 Feature Profile이 BID1 queue 진입과 연결되는지 사실만 요약한다.

    Profile cohort를 고치거나 새 label을 만들지 않는다. 이 요약은 Agent와 이후 Stage가
    PROFIT oracle 기회를 실제 주문 체결 증거로 오해하지 않게 하는 capability 기록이다.
    """
    required = {"entry_queue_status", "entry_queue_fraction", "entry_fill_before_oracle_ask"}
    missing = sorted(required & {str(value) for value in missing_outcome_columns})
    profit = anchors.loc[anchors["cohort"].astype(str).eq("PROFIT")].copy()
    known = (not missing and not profit.empty
             and profit["entry_queue_status"].notna().any()
             and profit["entry_fill_before_oracle_ask"].notna().any())
    if not known:
        return {
            "state": "ENTRY_QUEUE_UNOBSERVED",
            "profit_anchor_count": int(len(profit)),
            "fillable_profit_anchor_count": 0,
            "fillable_profit_anchor_rate": None,
            "missing_columns": missing,
            "note": ("저장 Profile에 BID1 queue 체결과 oracle ASK 이전 체결 여부가 없어, "
                     "PROFIT cohort를 executable entry 증거로 읽을 수 없다."),
        }
    status = profit["entry_queue_status"].fillna("missing").astype(str)
    fillable = profit["entry_fill_before_oracle_ask"].fillna(False).astype(bool)
    count = int(fillable.sum())
    return {
        "state": "ENTRY_QUEUE_OBSERVED",
        "profit_anchor_count": int(len(profit)),
        "fillable_profit_anchor_count": count,
        "fillable_profit_anchor_rate": float(count / len(profit)) if len(profit) else None,
        "entry_queue_status_counts": {str(key): int(value)
                                      for key, value in status.value_counts().sort_index().items()},
        "queue_fractions": sorted({float(value) for value in profit["entry_queue_fraction"].dropna()}),
        "missing_columns": [],
        "note": ("이 정보는 저장 Profile의 BID1 queue 체결 보조 기록이다. canonical exit 수익이나 "
                 "Validation/Final 결과를 뜻하지 않는다."),
    }
