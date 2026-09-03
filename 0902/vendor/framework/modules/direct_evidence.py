"""Agent가 멈춘 뒤에도 관측된 상태 자체를 제한적으로 시험하는 후보 경로."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .. import (catalog, executable as X, hypothesis as H, profit_target as PT,
                sample_condition as SC)


HIGHER_IN_PROFIT = "higher_in_PROFIT"
LOWER_IN_PROFIT = "lower_in_PROFIT"
_DIRECTION = {HIGHER_IN_PROFIT: X.HIGHER, LOWER_IN_PROFIT: X.LOWER}
_FILL_DIRECTION = {"higher_in_FILLED": X.HIGHER, "lower_in_FILLED": X.LOWER}
_STOP_AVOIDANCE_DIRECTION = {"higher_in_NONSTOP": X.HIGHER, "lower_in_NONSTOP": X.LOWER}
_RECOVERY_PATH_DIRECTION = {"higher_in_NET_RECOVERY": X.HIGHER,
                            "lower_in_NET_RECOVERY": X.LOWER}
_EARLY_REVERSAL_PATH_DIRECTION = {
    "higher_in_EARLY_RECOVERY_LATE_REVERSAL": X.HIGHER,
    "lower_in_EARLY_RECOVERY_LATE_REVERSAL": X.LOWER,
}
_STATE_DIRECTION = {"HIGH": X.HIGHER, "LOW": X.LOWER}
EXECUTION_OUTCOME_FILL_COMPOUND_LIMIT = 12
EXECUTION_OUTCOME_PAIR_COMPOUND_LIMIT = 8
EXECUTION_STOP_AVOIDANCE_FILL_COMPOUND_LIMIT = 8
EXECUTION_RECOVERY_PATH_FILL_COMPOUND_LIMIT = 8
EXECUTION_RECOVERY_PATH_TEMPORAL_FILL_COMPOUND_LIMIT = 8
EXECUTION_TEMPORAL_FILL_COMPOUND_LIMIT = 8


def _direction_variants(direction: str | None) -> tuple[str, ...]:
    """관측 방향이 없으면 양쪽 분위 꼬리를 모두 Search 후보로 연다."""
    return (direction,) if direction is not None else (X.HIGHER, X.LOWER)


def _state_equivalence_fingerprint(state: Mapping[str, Any]) -> str:
    """이름이 달라도 직전 100 tick 분위 신호가 완전히 같은 상태의 공통 지문."""
    group, direction = catalog.quantile_state_key(
        str(state.get("feature") or ""), str(state.get("direction") or ""))
    return f"{group}:{direction}"


def _same_quantile_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return catalog.same_quantile_state(
        str(left.get("feature") or ""), str(left.get("direction") or ""),
        str(right.get("feature") or ""), str(right.get("direction") or ""))


def _same_quantile_feature_group(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """방향과 무관하게 같은 rank 축인가.

    같은 축의 HIGHER와 LOWER를 AND로 묶으면 같은 상태의 중복이거나 서로 양립할 수
    없는 조건이다. 어느 쪽이든 별도 두-state 가설로 만들지 않는다.
    """
    left_group, _ = catalog.quantile_state_key(
        str(left.get("feature") or ""), str(left.get("direction") or ""))
    right_group, _ = catalog.quantile_state_key(
        str(right.get("feature") or ""), str(right.get("direction") or ""))
    return left_group == right_group


def _states_equivalence_fingerprint(states: list[Mapping[str, Any]]) -> str:
    return "&".join(sorted(_state_equivalence_fingerprint(state) for state in states))


def _temporal_state_equivalence_fingerprint(state: Mapping[str, Any], *,
                                            lag_seconds: int) -> str:
    """같은 clock 변화량 분위 상태를 같은 신호로 묶는다.

    양의 단조 변환의 차분도 같은 종목·일 분위 꼬리 조건을 만든다. 예를 들어
    ``book_imbalance = 2 * queue_imbalance_best - 1`` 이므로 두 1초 변화량은
    같은 entry signal이다.
    """
    return f"difference:clock:{int(lag_seconds)}:{_state_equivalence_fingerprint(state)}"


def _semantic_hypothesis_ids(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """후보 이름도 실제 entry 논리를 가리키게 고정한다.

    Evidence family id는 실행마다 같은데 대표 feature나 방향은 달라질 수 있다. 그 id만
    hypothesis id로 쓰면 서로 다른 entry 계약이 같은 이름을 공유한다. 기존 id는 CLI
    선택 호환용 alias로 보존하고, 새 id에는 이미 고정된 entry fingerprint를 붙인다.
    """
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in candidates:
        candidate = dict(source)
        legacy_id = str(candidate.get("hypothesis_id") or "")
        fingerprint = str(candidate.get("entry_logic_fingerprint") or "")
        if not legacy_id or not fingerprint:
            raise ValueError("직접 Evidence 후보에 hypothesis id 또는 entry logic fingerprint가 없다")
        fragment = re.sub(r"[^A-Za-z0-9]+", "_", fingerprint).strip("_").upper()
        hypothesis_id = f"{legacy_id}__{fragment}"
        if hypothesis_id in seen:
            raise ValueError(f"서로 다른 직접 Evidence 후보가 같은 entry identity를 쓴다: {hypothesis_id}")
        seen.add(hypothesis_id)
        candidate["legacy_hypothesis_id"] = legacy_id
        candidate["hypothesis_id"] = hypothesis_id
        resolved.append(candidate)
    return resolved


def _deduplicate_single_state_candidates(candidates: list[dict[str, Any]],
                                         rejected: list[dict[str, Any]], *,
                                         reason: str) -> list[dict[str, Any]]:
    """같은 entry signal의 이름만 다른 단일 상태 후보를 하나로 남긴다."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(
            candidates,
            key=lambda item: (-float((item.get("selection_metrics") or {}).get(
                "effect_strength") or 0.0), str(item.get("hypothesis_id") or ""))):
        states = list(candidate.get("entry_states") or [])
        if len(states) != 1:
            raise ValueError("단일 상태 동치 제거에 entry state가 하나여야 한다")
        temporal = candidate.get("temporal_entry") or {}
        if temporal.get("operator") == "DIFFERENCE":
            fingerprint = _temporal_state_equivalence_fingerprint(
                states[0], lag_seconds=int(temporal.get("lag_seconds") or 0))
        else:
            fingerprint = _state_equivalence_fingerprint(states[0])
        candidate["entry_signal_equivalence_fingerprint"] = fingerprint
        if fingerprint in seen:
            basis = list(candidate.get("evidence_basis") or [])
            rejected.append({
                "evidence_id": basis[0].get("evidence_id") if basis else None,
                "reasons": [reason],
            })
            continue
        seen.add(fingerprint)
        selected.append(candidate)
    return selected


@dataclass(frozen=True)
class DirectEvidenceConfig:
    """직접 후보 생성의 수치 설정."""

    min_effect_strength: float = 0.0
    min_date_agreement: float = H.AgentConfig().min_date_agreement
    min_symbol_agreement: float = H.AgentConfig().min_symbol_agreement
    severe_coverage_gap: float = H.AgentConfig().severe_coverage_gap
    execution_state_min_effect_strength: float = 0.0
    # outcome × fill 조합은 기본적으로 상위 12개만 연다. 앞 묶음이 끝난 뒤의 별도
    # 연구 요청은 이 값을 키워, 이미 실행한 ID를 제외한 다음 관측 관계를 보낼 수 있다.
    execution_outcome_fill_compound_limit: int = EXECUTION_OUTCOME_FILL_COMPOUND_LIMIT
    # STOP 회피 × fill도 같은 원리다. 기본 후보군 뒤의 별도 요청에서만 다음 순위를 연다.
    execution_stop_avoidance_fill_compound_limit: int = (
        EXECUTION_STOP_AVOIDANCE_FILL_COMPOUND_LIMIT)


def candidates(package: Mapping[str, Any], config: DirectEvidenceConfig = DirectEvidenceConfig()
               ) -> dict[str, Any]:
    """모든 수치 단일 상태를 후보로 열고 Search에서 반증한다.

    경제적 해석을 임의로 만들지 않는다. 일반 feature family의 결합은 이 경로에서
    만들지 않는다. 단, 같은 canonical replay에서 독립적으로 관측된 실행 결과 상태와
    entry 체결 상태의 쌍은 별도 compound 후보로 제한해 Search에서 반증할 수 있다.
    """
    selected, rejected = [], []
    for item in sorted(package.get("evidence_families") or [],
                       key=lambda value: str(value.get("evidence_id", ""))):
        contrast = (item.get("snapshot_contrast") or {}).get("profit_vs_loss") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        observed_direction = _DIRECTION.get(str(contrast.get("direction") or ""))
        directions = _direction_variants(observed_direction)
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if not item.get("independent_evidence", True):
            reasons.append("DERIVED_OR_REDUNDANT")
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            # 이 경로의 유일한 탐색축은 종목별 직전 100 tick 분위 q다. boolean은 q 임계값을
            # 뜻 있게 가질 수 없으므로 0/1 수치로 조용히 바꾸지 않는다.
            reasons.append("NON_NUMERIC_FEATURE")
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if any(str(warning).startswith("PROFILE_SELECTION_CONFOUND")
               for warning in item.get("warnings") or []):
            reasons.append("PROFILE_SELECTION_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        evidence_id = str(item["evidence_id"])
        family = str(item.get("family") or "")
        candidate_id = f"DE_{evidence_id}"
        for direction in directions:
            selected.append({
                "hypothesis_id": candidate_id,
                "title": f"직접 관측 상태 — {feature} {direction}",
                "source_type": "DIRECT_DISCOVERY_PROFILE_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": f"{feature}:{direction}",
                "evidence_basis": [{
                    "evidence_id": evidence_id, "family": family,
                    "representative_feature": feature, "role": item.get("role"),
                    "observation": (
                        "Discovery PROFIT/LOSS 대조에서 확인된 관측이다. "
                        "경제적 메커니즘은 이 후보가 주장하지 않는다."),
                }],
                "entry_states": [{"feature": feature, "direction": direction, "family": family}],
                "selection_metrics": {
                    "effect_strength": (float(effect)
                                        if isinstance(effect, (int, float)) else None),
                    "date_agreement": (float(date_ratio)
                                       if isinstance(date_ratio, (int, float)) else None),
                    "symbol_agreement": (float(symbol_ratio)
                                         if isinstance(symbol_ratio, (int, float)) else None),
                },
                "warnings": [*list(item.get("warnings") or []),
                             *(["DIRECTION_UNSPECIFIED_BOTH_SIDES"]
                               if observed_direction is None else [])],
                "limits": [
                    "Discovery outcome으로 선택된 관측 상태다.",
                    "경제적 메커니즘, 인과, 체결 우위는 주장하지 않는다.",
                    "고정 canonical exit의 Search·Validation·Final Backtest가 별도로 반증할 수 있다.",
                ],
            })
    selected = _deduplicate_single_state_candidates(
        selected, rejected, reason="MONOTONICALLY_EQUIVALENT_PROFILE_STATE")
    late_trigger_cross = _late_trigger_cross_candidates(package, selected)
    execution_state, execution_state_rejected = _execution_state_candidates(package, config)
    stop_avoidance, stop_avoidance_rejected = _execution_stop_avoidance_candidates(package, config)
    recovery_path, recovery_path_rejected = _execution_recovery_path_candidates(package, config)
    early_reversal_path, early_reversal_path_rejected = (
        _execution_early_reversal_path_candidates(package, config))
    recovery_path_temporal, recovery_path_temporal_rejected = (
        _execution_recovery_path_temporal_candidates(package, config))
    recovery_path_fill, recovery_path_fill_rejected = _execution_recovery_path_fill_compounds(
        package, recovery_path, config)
    recovery_path_temporal_fill, recovery_path_temporal_fill_rejected = (
        _execution_recovery_path_temporal_fill_compounds(
            package, recovery_path_temporal, config))
    execution_pairs, execution_pair_rejected = _execution_outcome_pair_compounds(execution_state)
    execution_compounds, execution_fill_rejected = _execution_outcome_fill_compounds(
        package, execution_state, config)
    stop_fill_compounds, stop_fill_rejected = _execution_stop_avoidance_fill_compounds(
        package, stop_avoidance, config)
    temporal, temporal_rejected = _execution_temporal_candidates(package, config)
    execution_temporal_fill, execution_temporal_fill_rejected = (
        _execution_temporal_fill_compounds(package, temporal, config))
    relations, relation_rejected = _relation_candidates(package, selected)
    incremental, incremental_rejected = _incremental_relation_candidates(package, selected)
    candidates = _semantic_hypothesis_ids([
        *selected, *late_trigger_cross,
        *execution_state, *stop_avoidance, *recovery_path, *early_reversal_path,
        *recovery_path_temporal, *recovery_path_fill, *recovery_path_temporal_fill,
        *execution_temporal_fill,
        *execution_pairs, *execution_compounds, *stop_fill_compounds,
        *temporal, *relations, *incremental,
    ])
    for candidate in candidates:
        candidate["profit_target"] = PT.canonical_target()
        sample_condition = package.get("executable_sample_condition")
        if sample_condition is not None:
            report = SC.validate(sample_condition)
            if not report["ok"]:
                raise ValueError("Evidence executable_sample_condition이 유효하지 않다: "
                                 + "; ".join(report["problems"]))
            candidate["executable_sample_condition"] = dict(sample_condition)
            candidate["executable_sample_condition_sha256"] = SC.condition_hash(
                sample_condition)
    return {
        "schema": "direct_evidence_candidate_set.v3",
        "state": "DIRECT_EVIDENCE_CANDIDATES_READY" if candidates else "NO_DIRECT_EVIDENCE_CANDIDATE",
        "selection_rule": {
            "independent_evidence": True,
            "effect_strength_gte": config.min_effect_strength,
            "execution_state_effect_strength_gte": config.execution_state_min_effect_strength,
            "execution_outcome_fill_compound_limit": config.execution_outcome_fill_compound_limit,
            "execution_outcome_pair_compound_limit": EXECUTION_OUTCOME_PAIR_COMPOUND_LIMIT,
            "execution_stop_avoidance_fill_compound_limit": (
                config.execution_stop_avoidance_fill_compound_limit),
            "execution_recovery_path_fill_compound_limit": (
                EXECUTION_RECOVERY_PATH_FILL_COMPOUND_LIMIT),
            "execution_recovery_path_temporal_fill_compound_limit": (
                EXECUTION_RECOVERY_PATH_TEMPORAL_FILL_COMPOUND_LIMIT),
            "execution_temporal_fill_compound_limit": (
                EXECUTION_TEMPORAL_FILL_COMPOUND_LIMIT),
            "severe_coverage_gap_lt": config.severe_coverage_gap,
        },
        "candidates": candidates,
        "rejected": [*rejected, *execution_state_rejected, *stop_avoidance_rejected,
                     *recovery_path_rejected,
                     *early_reversal_path_rejected,
                     *recovery_path_temporal_rejected,
                     *recovery_path_fill_rejected,
                     *recovery_path_temporal_fill_rejected,
                     *execution_temporal_fill_rejected,
                     *execution_pair_rejected,
                     *execution_fill_rejected, *stop_fill_rejected,
                     *temporal_rejected],
        "relation_rejected": [*relation_rejected, *incremental_rejected],
        "note": (
            "Agent의 경제적 가설을 대신하지 않는다. Agent 후보 생성이 실패했을 때에도, "
            "그 Agent가 이미 읽은 동일 Evidence에서 관측 상태 또는 명시된 관계를 제한적으로 시험하는 경로다."),
    }


def _late_trigger_cross_candidates(package: Mapping[str, Any],
                                   single_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """저장된 LATE_TRIGGER 상태를 반복 level이 아닌 최초 q-cross 사건으로도 연다.

    Evidence가 직접 관측한 것은 상태의 near-anchor 역할이다. 실제 q-cross 자체를
    관측 사실처럼 쓰지 않으며, 같은 q를 새로 넘는 사건형 entry가 level-state보다
    실행 손실을 줄이는지만 별도 고정-exit 후보로 반증한다.
    """
    by_evidence = {
        str(item.get("evidence_id") or ""): item
        for item in package.get("evidence_families") or [] if isinstance(item, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for source in single_candidates:
        basis = list(source.get("evidence_basis") or [])
        if len(basis) != 1:
            continue
        evidence = by_evidence.get(str(basis[0].get("evidence_id") or "")) or {}
        if str(evidence.get("role") or "") not in {"LATE_TRIGGER", "TRIGGER"}:
            continue
        state = dict((source.get("entry_states") or [{}])[0])
        if not state.get("feature") or not state.get("direction"):
            continue
        fingerprint = f"cross:{state['feature']}:{state['direction']}"
        candidates.append({
            "hypothesis_id": f"DE_ONSET_{basis[0].get('evidence_id')}",
            "title": f"직접 관측 late trigger 사건 — {state['feature']} 경계 통과",
            "source_type": "DIRECT_EVIDENCE_LATE_TRIGGER_CROSS",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": fingerprint,
            "entry_signal_equivalence_fingerprint": fingerprint,
            "evidence_basis": [{
                **dict(basis[0]), "role": "LATE_TRIGGER_EVENT_FORM",
                "observation": (
                    "저장된 Discovery Evidence에서 anchor 근처 LATE_TRIGGER 역할로 분류된 상태다. "
                    "이 후보는 같은 직전 100 tick 분위 경계를 새로 넘는 사건형 진입을 별도로 시험한다."),
            }],
            "entry_states": [state],
            "temporal_entry": {
                "operator": "CROSSOVER",
                "time_basis": "tick",
                "trigger_event": "CROSS_PRIOR_100_TICKS_PERCENTILE",
                "warmup_ticks": 1,
                "why_fixed": (
                    "Evidence의 LATE_TRIGGER 역할을 반복 level이 아닌 최초 경계 통과 사건으로 "
                    "표현한다. 시간 폭·지속·재진입 간격은 추가로 탐색하지 않는다."),
            },
            "selection_metrics": dict(source.get("selection_metrics") or {}),
            "warnings": list(source.get("warnings") or []),
            "limits": [
                "LATE_TRIGGER 역할은 직접 관측됐지만 q-cross 자체를 관측 사실로 주장하지 않는다.",
                "동일한 상태 level 후보와 별도의 entry-event 논리이며, q 하나만 Search한다.",
                "고정 canonical exit Search·Validation·Final Backtest가 사건형 진입을 반증한다.",
            ],
        })
    return candidates


def _execution_state_candidates(package: Mapping[str, Any], config: DirectEvidenceConfig
                                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """개별 canonical anchor replay label의 t=0 상태 후보만 만든다."""
    selected, rejected = [], []
    evidence = package.get("execution_state_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get("profit_vs_loss") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        observed_direction = _DIRECTION.get(str(contrast.get("direction") or ""))
        directions = _direction_variants(observed_direction)
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        evidence_id = str(item.get("evidence_id") or "")
        family = str(item.get("family") or "")
        for direction in directions:
            selected.append({
                "hypothesis_id": f"DE_{evidence_id}",
                "title": f"실행 anchor 직접 관측 상태 — {feature} {direction}",
                "source_type": "DIRECT_DISCOVERY_ANCHOR_EXECUTION_STATE_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": f"execution-anchor-state:{feature}:{direction}",
                "evidence_basis": [{
                    "evidence_id": evidence_id, "family": family,
                    "representative_feature": feature, "role": "EXECUTION_ANCHOR_STATE",
                    "observation": (
                        "원 Profile BACKGROUND anchor의 t=0 상태와 개별 canonical "
                        "fixed-exit replay label의 대조다."),
                }],
                "entry_states": [{"feature": feature, "direction": direction, "family": family}],
                "selection_metrics": {
                    "effect_strength": (float(effect)
                                        if isinstance(effect, (int, float)) else None),
                    "date_agreement": (float(date_ratio)
                                       if isinstance(date_ratio, (int, float)) else None),
                    "symbol_agreement": (float(symbol_ratio)
                                         if isinstance(symbol_ratio, (int, float)) else None),
                },
                "warnings": [*list(item.get("warnings") or []),
                             *(["DIRECTION_UNSPECIFIED_BOTH_SIDES"]
                               if observed_direction is None else []),
                             *(["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
                "limits": [
                    "원 Profile BACKGROUND anchor만 쓴 개별 canonical execution replay 진단이다.",
                    "anchor 사이 position blocking을 공유하지 않으므로 전략 PnL이 아니다.",
                    "경제적 메커니즘·인과·체결 우위를 주장하지 않는다.",
                    "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증할 수 있다.",
                ],
            })
    return _deduplicate_single_state_candidates(
        selected, rejected, reason="MONOTONICALLY_EQUIVALENT_EXECUTION_STATE"), rejected


def _execution_stop_avoidance_candidates(
        package: Mapping[str, Any], config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """즉시 손절을 덜 맞은 체결 상태를 entry-harm filter 후보로만 연다."""
    selected, rejected = [], []
    evidence = package.get("execution_stop_avoidance_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get("nonstop_vs_stop") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _STOP_AVOIDANCE_DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        if direction is None:
            reasons.append("NO_NONSTOP_STOP_DIRECTION")
        if (not isinstance(effect, (int, float))
                or float(effect) < config.execution_state_min_effect_strength):
            reasons.append("WEAK_EFFECT")
        # STOP/NON_STOP 표본은 체결된 Discovery anchor로만 좁아져 종목별 contrast가
        # 희소하다. 안정성은 artifact 경고로 보존하고, 이 새 entry-harm 가설을 여기서
        # 표본 부족만으로 닫지 않는다. 이후 별도 날짜의 fixed-exit Search/Validation이
        # 실제 실행 성과를 반증한다.
        date_unobservable = not isinstance(date_ratio, (int, float))
        date_unstable = (not date_unobservable and float(date_ratio) < config.min_date_agreement)
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        symbol_unstable = (not symbol_unobservable
                           and float(symbol_ratio) < config.min_symbol_agreement)
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        evidence_id = str(item.get("evidence_id") or "")
        family = str(item.get("family") or "")
        selected.append({
            "hypothesis_id": f"DE_{evidence_id}",
            "title": f"고정 exit 즉시 손절 회피 상태 — {feature} {direction}",
            "source_type": "DIRECT_DISCOVERY_STOP_AVOIDANCE_EVIDENCE",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": f"execution-stop-avoidance:{feature}:{direction}",
            "evidence_basis": [{
                "evidence_id": evidence_id, "family": family,
                "representative_feature": feature, "role": "FIXED_EXIT_STOP_AVOIDANCE_STATE",
                "observation": (
                    "원 Profile BACKGROUND anchor의 체결 거래를 고정 canonical exit으로 재생했을 때, "
                    "즉시 손절 계열이 아닌 체결과 손절 체결을 가르는 t=0 상태다."),
            }],
            "entry_states": [{"feature": feature, "direction": direction, "family": family}],
            "selection_metrics": {
                "effect_strength": float(effect),
                "date_agreement": (float(date_ratio)
                                   if isinstance(date_ratio, (int, float)) else None),
                "symbol_agreement": (float(symbol_ratio)
                                   if isinstance(symbol_ratio, (int, float)) else None),
            },
            "warnings": [*list(item.get("warnings") or []),
                         *(["DATE_STABILITY_UNOBSERVABLE"] if date_unobservable else []),
                         *(["DATE_DIRECTION_UNSTABLE"] if date_unstable else []),
                         *(["SYMBOL_DIRECTION_UNSTABLE"] if symbol_unstable else []),
                         *(["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
            "limits": [
                "NON_STOP은 수익이 아니라 고정 canonical exit의 즉시 손절 계열이 아닌 체결이다.",
                "이 후보는 청산 규칙을 바꾸거나 최적화하지 않고 entry 상태만 시험한다.",
                "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증한다.",
            ],
        })
    return _deduplicate_single_state_candidates(
        selected, rejected, reason="MONOTONICALLY_EQUIVALENT_STOP_AVOIDANCE_STATE"), rejected


def _execution_recovery_path_candidates(
        package: Mapping[str, Any], config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """고정 exit을 쓰지 않은 raw 경로 회복/지속악화 대조를 entry 후보로 연다."""
    ranked, rejected = [], []
    evidence = package.get("execution_recovery_path_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get(
            "recovery_vs_persistent_adverse") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _RECOVERY_PATH_DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        if direction is None:
            reasons.append("NO_RECOVERY_PATH_DIRECTION")
        if (not isinstance(effect, (int, float))
                or float(effect) < config.execution_state_min_effect_strength):
            reasons.append("WEAK_EFFECT")
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        date_unobservable = not isinstance(date_ratio, (int, float))
        date_unstable = (not date_unobservable and float(date_ratio) < config.min_date_agreement)
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        symbol_unstable = (not symbol_unobservable
                           and float(symbol_ratio) < config.min_symbol_agreement)
        evidence_id = str(item.get("evidence_id") or "")
        family = str(item.get("family") or "")
        state = {"feature": feature, "direction": direction, "family": family}
        ranked.append((float(effect), {
            "hypothesis_id": f"DE_{evidence_id}",
            "title": f"원시 30초 회복 경로 상태 — {feature} {direction}",
            "source_type": "DIRECT_DISCOVERY_RECOVERY_PATH_EVIDENCE",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": f"execution-recovery-path:{feature}:{direction}",
            "entry_signal_equivalence_fingerprint": _state_equivalence_fingerprint(state),
            "evidence_basis": [{
                "evidence_id": evidence_id, "family": family,
                "representative_feature": feature, "role": "RAW_30S_RECOVERY_PATH_STATE",
                "observation": (
                    "원 Profile BACKGROUND anchor의 queue 체결 뒤, canonical exit을 무시한 "
                    "30초 BID1 raw path에서 NET_RECOVERY와 PERSISTENT_ADVERSE를 가르는 t=0 상태다."),
            }],
            "entry_states": [state],
            "selection_metrics": {
                "effect_strength": float(effect),
                "date_agreement": (float(date_ratio)
                                   if isinstance(date_ratio, (int, float)) else None),
                "symbol_agreement": (float(symbol_ratio)
                                   if isinstance(symbol_ratio, (int, float)) else None),
            },
            "warnings": [*list(item.get("warnings") or []),
                         *(["DATE_STABILITY_UNOBSERVABLE"] if date_unobservable else []),
                         *(["DATE_DIRECTION_UNSTABLE"] if date_unstable else []),
                         *(["SYMBOL_DIRECTION_UNSTABLE"] if symbol_unstable else []),
                         *(["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
            "limits": [
                "NET_RECOVERY/PERSISTENT_ADVERSE는 canonical exit과 무관한 체결 뒤 30초 BID1 raw path label이다.",
                "이 후보는 청산 규칙을 바꾸거나 최적화하지 않고 entry 상태만 시험한다.",
                "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증한다.",
            ],
        }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate["entry_signal_equivalence_fingerprint"])
        if fingerprint in seen:
            rejected.append({
                "evidence_id": candidate["evidence_basis"][0]["evidence_id"],
                "reasons": ["MONOTONICALLY_EQUIVALENT_STATE"],
            })
            continue
        seen.add(fingerprint)
        selected.append(candidate)
    return selected, rejected


def _execution_early_reversal_path_candidates(
        package: Mapping[str, Any], config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """초기 순회복 뒤 재하락/지속악화 raw path 대조를 정적 entry 후보로 연다."""
    ranked, rejected = [], []
    evidence = package.get("execution_early_reversal_path_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get(
            "early_reversal_vs_persistent_adverse") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _EARLY_REVERSAL_PATH_DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        if direction is None:
            reasons.append("NO_EARLY_REVERSAL_PATH_DIRECTION")
        if (not isinstance(effect, (int, float))
                or float(effect) < config.execution_state_min_effect_strength):
            reasons.append("WEAK_EFFECT")
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        date_unobservable = not isinstance(date_ratio, (int, float))
        date_unstable = (not date_unobservable and float(date_ratio) < config.min_date_agreement)
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        symbol_unstable = (not symbol_unobservable
                           and float(symbol_ratio) < config.min_symbol_agreement)
        evidence_id = str(item.get("evidence_id") or "")
        family = str(item.get("family") or "")
        state = {"feature": feature, "direction": direction, "family": family}
        ranked.append((float(effect), {
            "hypothesis_id": f"DE_{evidence_id}",
            "title": f"초기 회복 뒤 재하락 경로 상태 — {feature} {direction}",
            "source_type": "DIRECT_DISCOVERY_EARLY_REVERSAL_PATH_EVIDENCE",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": f"execution-early-reversal-path:{feature}:{direction}",
            "entry_signal_equivalence_fingerprint": _state_equivalence_fingerprint(state),
            "evidence_basis": [{
                "evidence_id": evidence_id, "family": family,
                "representative_feature": feature,
                "role": "RAW_30S_EARLY_REVERSAL_PATH_STATE",
                "observation": (
                    "원 Profile BACKGROUND anchor의 queue 체결 뒤, canonical exit을 무시한 "
                    "30초 BID1 raw path에서 EARLY_RECOVERY_LATE_REVERSAL과 "
                    "PERSISTENT_ADVERSE를 가르는 t=0 상태다."),
            }],
            "entry_states": [state],
            "selection_metrics": {
                "effect_strength": float(effect),
                "date_agreement": (float(date_ratio)
                                   if isinstance(date_ratio, (int, float)) else None),
                "symbol_agreement": (float(symbol_ratio)
                                   if isinstance(symbol_ratio, (int, float)) else None),
            },
            "warnings": [*list(item.get("warnings") or []),
                         *(["DATE_STABILITY_UNOBSERVABLE"] if date_unobservable else []),
                         *(["DATE_DIRECTION_UNSTABLE"] if date_unstable else []),
                         *(["SYMBOL_DIRECTION_UNSTABLE"] if symbol_unstable else []),
                         *(["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
            "limits": [
                "EARLY_RECOVERY_LATE_REVERSAL/PERSISTENT_ADVERSE는 canonical exit과 무관한 체결 뒤 30초 BID1 raw path label이다.",
                "이 후보는 청산 규칙을 바꾸거나 최적화하지 않고 entry 상태만 시험한다.",
                "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증한다.",
            ],
        }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate["entry_signal_equivalence_fingerprint"])
        if fingerprint in seen:
            rejected.append({
                "evidence_id": candidate["evidence_basis"][0]["evidence_id"],
                "reasons": ["MONOTONICALLY_EQUIVALENT_EARLY_REVERSAL_PATH_STATE"],
            })
            continue
        seen.add(fingerprint)
        selected.append(candidate)
    return selected, rejected


def _execution_recovery_path_temporal_candidates(
        package: Mapping[str, Any], config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """raw 30초 회복/지속악화와 사전 변화량이 만나는 timing 후보를 만든다.

    같은 raw-path label의 t=0 상태와 달리 차분 상태만 일반 Evidence 0.10
    효과 기준으로 막으면, 표현 방식 때문에 관측 가능한 timing 가설을 잃는다.
    raw execution label의 공통 0.05 기준을 쓰고 불안정성은 artifact 경고로 남긴다.
    """
    ranked, rejected = [], []
    evidence = package.get("execution_recovery_path_temporal_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get(
            "recovery_vs_persistent_adverse") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _RECOVERY_PATH_DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        if direction is None:
            reasons.append("NO_RECOVERY_PATH_DIRECTION")
        if (not isinstance(effect, (int, float))
                or float(effect) < config.execution_state_min_effect_strength):
            reasons.append("WEAK_EFFECT")
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        expression = dict(item.get("expression") or {})
        try:
            catalog.validate_expression(expression)
        except catalog.ExpressionError:
            rejected.append({"evidence_id": item.get("evidence_id"),
                             "reasons": ["INVALID_TEMPORAL_EXPRESSION"]})
            continue
        lag_seconds = int(expression.get("lag") or 0)
        if lag_seconds <= 0:
            rejected.append({"evidence_id": item.get("evidence_id"),
                             "reasons": ["INVALID_TEMPORAL_LAG"]})
            continue
        date_unobservable = not isinstance(date_ratio, (int, float))
        date_unstable = (not date_unobservable
                         and float(date_ratio) < config.min_date_agreement)
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        symbol_unstable = (not symbol_unobservable
                           and float(symbol_ratio) < config.min_symbol_agreement)
        family = str(item.get("family") or "")
        evidence_id = str(item.get("evidence_id") or "")
        state = {"feature": feature, "direction": direction, "family": family,
                 "expression": expression,
                 "parameter": f"q_recovery_difference_{lag_seconds}s_{feature}"}
        ranked.append((float(effect), {
            "hypothesis_id": f"DE_{evidence_id}",
            "title": f"원시 30초 회복 경로 변화 — {feature} {lag_seconds}초 변화 {direction}",
            "source_type": "DIRECT_DISCOVERY_RECOVERY_PATH_TEMPORAL_EVIDENCE",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": (
                f"recovery-path-difference:clock:{lag_seconds}:{feature}:{direction}"),
            "entry_signal_equivalence_fingerprint": _temporal_state_equivalence_fingerprint(
                state, lag_seconds=lag_seconds),
            "evidence_basis": [{
                "evidence_id": evidence_id, "family": family,
                "representative_feature": feature, "role": "RAW_30S_RECOVERY_PATH_TEMPORAL_STATE",
                "observation": (
                    f"저장된 Profile의 anchor t-{lag_seconds}초→t 변화량과 canonical exit을 "
                    "무시한 30초 BID1 raw path의 NET_RECOVERY/PERSISTENT_ADVERSE 대조다."),
            }],
            "entry_states": [state],
            "temporal_entry": {"operator": "DIFFERENCE", "lag_seconds": lag_seconds,
                               "time_basis": "clock", "warmup_ticks": 0,
                               "why_fixed": (
                                   f"저장된 Feature Profile의 두 사전 시점 t-{lag_seconds}초와 t를 쓴다.")},
            "selection_metrics": {"effect_strength": float(effect),
                                  "date_agreement": (float(date_ratio)
                                                     if isinstance(date_ratio, (int, float)) else None),
                                  "symbol_agreement": (float(symbol_ratio)
                                                       if isinstance(symbol_ratio, (int, float)) else None)},
            "warnings": [*list(item.get("warnings") or []),
                         *(["DATE_STABILITY_UNOBSERVABLE"] if date_unobservable else []),
                         *(["DATE_DIRECTION_UNSTABLE"] if date_unstable else []),
                         *(["SYMBOL_DIRECTION_UNSTABLE"] if symbol_unstable else []),
                         *(["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
            "limits": [
                "NET_RECOVERY/PERSISTENT_ADVERSE는 canonical exit과 무관한 체결 뒤 30초 BID1 raw path label이다.",
                "변화량은 저장된 사전 Profile 값만 쓰며, 이 후보는 entry timing만 시험한다.",
                "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증한다.",
            ],
        }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate["entry_signal_equivalence_fingerprint"])
        if fingerprint in seen:
            rejected.append({
                "evidence_id": candidate["evidence_basis"][0]["evidence_id"],
                "reasons": ["MONOTONICALLY_EQUIVALENT_TEMPORAL_STATE"],
            })
            continue
        seen.add(fingerprint)
        selected.append(candidate)
    return selected, rejected


def _execution_recovery_path_fill_compounds(
        package: Mapping[str, Any], recovery_path: list[dict[str, Any]],
        config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """원시 회복 경로 상태와 BID1 queue 체결 상태를 같은 tick에서 함께 시험한다.

    첫 상태는 fixed exit을 쓰지 않은 체결 뒤 30초 경로 대조이고, 두 번째 상태는
    실제 queue 체결 가능성 대조다. 둘 다 entry의 서로 다른 질문일 뿐, 청산 또는
    전략 PnL을 학습하는 결합이 아니다.
    """
    fill_states, rejected = _execution_fill_states(package, config)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for recovery in recovery_path:
        recovery_state = dict((recovery.get("entry_states") or [{}])[0])
        recovery_basis = dict((recovery.get("evidence_basis") or [{}])[0])
        recovery_effect = float(
            (recovery.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for fill in fill_states:
            if _same_quantile_feature_group(recovery_state, fill["state"]):
                rejected.append({
                    "recovery_path_evidence_id": recovery_basis.get("evidence_id"),
                    "fill_evidence_id": fill["evidence_id"],
                    "reasons": ["DUPLICATE_OR_MONOTONICALLY_EQUIVALENT_FEATURE"],
                })
                continue
            states = sorted([recovery_state, dict(fill["state"])],
                            key=lambda item: (str(item.get("feature")), str(item.get("direction"))))
            fingerprint = "&".join(
                f"{state['feature']}:{state['direction']}" for state in states)
            recovery_id = str(recovery_basis.get("evidence_id"))
            fill_id = str(fill["evidence_id"])
            score = recovery_effect * float(fill["effect_strength"])
            ranked.append((score, {
                "hypothesis_id": f"DE_RECOVERY_PATH_FILL__{recovery_id}__{fill_id}",
                "title": (f"원시 회복 경로와 queue 체결 상태 — {recovery_state['feature']} "
                          f"+ {fill['state']['feature']}"),
                "source_type": "DIRECT_EXECUTION_RECOVERY_PATH_FILL_COMPOUND_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": _states_equivalence_fingerprint(states),
                "evidence_basis": [
                    {**recovery_basis, "role": "RAW_30S_RECOVERY_PATH_STATE"},
                    {"evidence_id": fill_id, "family": fill["family"],
                     "representative_feature": fill["state"]["feature"],
                     "role": "ENTRY_FILL_STATE", "observation": fill["observation"]},
                ],
                "entry_states": states,
                "selection_metrics": {
                    "recovery_path_effect_strength": recovery_effect,
                    "fill_effect_strength": float(fill["effect_strength"]),
                    "compound_priority": score,
                },
                "warnings": [*list(recovery.get("warnings") or []), *list(fill["warnings"])],
                "limits": [
                    "NET_RECOVERY/PERSISTENT_ADVERSE는 canonical exit과 무관한 체결 뒤 30초 BID1 raw path label이다.",
                    "FILLED/ENTRY_UNFILLED은 BID1 queue 체결 가능성 대조이며, 두 상태 모두 entry만 제한한다.",
                    "두 상태가 같은 결정 tick에 함께 충족할 때만 진입하고, fixed canonical exit은 변경하지 않는다.",
                    "고정 canonical exit Search·Validation·Final Backtest가 별도로 반증한다.",
                ],
            }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate.get("entry_signal_equivalence_fingerprint")
                          or candidate["entry_logic_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= EXECUTION_RECOVERY_PATH_FILL_COMPOUND_LIMIT:
            break
    return selected, rejected


def _execution_recovery_path_temporal_fill_compounds(
        package: Mapping[str, Any], recovery_path_temporal: list[dict[str, Any]],
        config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """원시 회복 timing 변화와 현재 BID1 queue 체결 상태를 함께 시험한다.

    변화량은 ``t-lag → t``의 저장된 사전 Profile 값이고, fill 상태는 현재 queue
    접근성이다. 둘은 같은 raw recovery label을 직접 조건에 넣지 않으며, Search부터
    같은 canonical fixed exit로 entry logic만 반증한다.
    """
    fill_states, rejected = _execution_fill_states(package, config)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for recovery in recovery_path_temporal:
        recovery_state = dict((recovery.get("entry_states") or [{}])[0])
        recovery_basis = dict((recovery.get("evidence_basis") or [{}])[0])
        temporal = dict(recovery.get("temporal_entry") or {})
        lag_seconds = int(temporal.get("lag_seconds") or 0)
        if temporal.get("operator") != "DIFFERENCE" or lag_seconds <= 0:
            rejected.append({
                "recovery_path_evidence_id": recovery_basis.get("evidence_id"),
                "reasons": ["INVALID_TEMPORAL_RECOVERY_PATH_CANDIDATE"],
            })
            continue
        recovery_effect = float(
            (recovery.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for fill in fill_states:
            # 같은 수준 축의 정적 tail을 차분 tail과 묶으면 두번째 관측이 새로운
            # entry 접근성 설명이 아니라 같은 feature 이름의 중복 조건이 된다.
            if _same_quantile_feature_group(recovery_state, fill["state"]):
                rejected.append({
                    "recovery_path_evidence_id": recovery_basis.get("evidence_id"),
                    "fill_evidence_id": fill["evidence_id"],
                    "reasons": ["DUPLICATE_OR_MONOTONICALLY_EQUIVALENT_FEATURE"],
                })
                continue
            fill_state = dict(fill["state"])
            states = [recovery_state, fill_state]
            recovery_id, fill_id = str(recovery_basis.get("evidence_id")), str(fill["evidence_id"])
            fingerprint = "&".join(sorted((
                _temporal_state_equivalence_fingerprint(recovery_state, lag_seconds=lag_seconds),
                _state_equivalence_fingerprint(fill_state),
            )))
            score = recovery_effect * float(fill["effect_strength"])
            ranked.append((score, {
                "hypothesis_id": f"DE_RECOVERY_PATH_TEMPORAL_FILL__{recovery_id}__{fill_id}",
                "title": (f"원시 회복 변화와 queue 체결 상태 — {recovery_state['feature']} "
                          f"{lag_seconds}초 변화 + {fill_state['feature']}"),
                "source_type": "DIRECT_EXECUTION_RECOVERY_PATH_TEMPORAL_FILL_COMPOUND_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": fingerprint,
                "evidence_basis": [
                    {**recovery_basis, "role": "RAW_30S_RECOVERY_PATH_TEMPORAL_STATE"},
                    {"evidence_id": fill_id, "family": fill["family"],
                     "representative_feature": fill_state["feature"],
                     "role": "ENTRY_FILL_STATE", "observation": fill["observation"]},
                ],
                "entry_states": states,
                "temporal_entry": temporal,
                "selection_metrics": {
                    "recovery_path_temporal_effect_strength": recovery_effect,
                    "fill_effect_strength": float(fill["effect_strength"]),
                    "compound_priority": score,
                },
                "warnings": [*list(recovery.get("warnings") or []), *list(fill["warnings"])],
                "limits": [
                    "NET_RECOVERY/PERSISTENT_ADVERSE는 canonical exit과 무관한 체결 뒤 30초 BID1 raw path label이다.",
                    "변화량은 저장된 사전 Profile 값이고, FILLED/ENTRY_UNFILLED은 현재 BID1 queue 체결 가능성 대조다.",
                    "두 조건이 같은 결정 tick에 함께 충족할 때만 진입하며, fixed canonical exit은 변경하지 않는다.",
                    "고정 canonical exit Search·Validation·Final Backtest가 별도로 반증한다.",
                ],
            }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate["entry_signal_equivalence_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= EXECUTION_RECOVERY_PATH_TEMPORAL_FILL_COMPOUND_LIMIT:
            break
    return selected, rejected


def _execution_temporal_fill_compounds(
        package: Mapping[str, Any], temporal_candidates: list[dict[str, Any]],
        config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """실제 fixed-exit outcome의 사전 변화량과 현재 queue 체결 상태를 함께 연다.

    이 경로는 raw 30초 회복 라벨을 쓰지 않는다. 첫 상태는 원 Profile BACKGROUND
    anchor의 canonical fixed-exit PROFIT/LOSS 대조에서 나온 clock difference이고,
    두 번째 상태는 같은 anchor에서 BID1 queue order가 FILLED인지의 대조다.
    """
    fill_states, rejected = _execution_fill_states(package, config)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for temporal_candidate in temporal_candidates:
        temporal_state = dict((temporal_candidate.get("entry_states") or [{}])[0])
        temporal_basis = dict((temporal_candidate.get("evidence_basis") or [{}])[0])
        temporal = dict(temporal_candidate.get("temporal_entry") or {})
        lag_seconds = int(temporal.get("lag_seconds") or 0)
        if temporal.get("operator") != "DIFFERENCE" or lag_seconds <= 0:
            rejected.append({
                "execution_temporal_evidence_id": temporal_basis.get("evidence_id"),
                "reasons": ["INVALID_EXECUTION_TEMPORAL_CANDIDATE"],
            })
            continue
        temporal_effect = float(
            (temporal_candidate.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for fill in fill_states:
            if _same_quantile_feature_group(temporal_state, fill["state"]):
                rejected.append({
                    "execution_temporal_evidence_id": temporal_basis.get("evidence_id"),
                    "fill_evidence_id": fill["evidence_id"],
                    "reasons": ["DUPLICATE_OR_MONOTONICALLY_EQUIVALENT_FEATURE"],
                })
                continue
            fill_state = dict(fill["state"])
            temporal_id, fill_id = str(temporal_basis.get("evidence_id")), str(fill["evidence_id"])
            fingerprint = "&".join(sorted((
                _temporal_state_equivalence_fingerprint(temporal_state, lag_seconds=lag_seconds),
                _state_equivalence_fingerprint(fill_state),
            )))
            score = temporal_effect * float(fill["effect_strength"])
            ranked.append((score, {
                "hypothesis_id": f"DE_EXECUTION_TEMPORAL_FILL__{temporal_id}__{fill_id}",
                "title": (f"실행 사전 변화와 queue 체결 상태 — {temporal_state['feature']} "
                          f"{lag_seconds}초 변화 + {fill_state['feature']}"),
                "source_type": "DIRECT_EXECUTION_TEMPORAL_FILL_COMPOUND_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": fingerprint,
                "evidence_basis": [
                    {**temporal_basis, "role": "EXECUTED_OUTCOME_TEMPORAL_STATE"},
                    {"evidence_id": fill_id, "family": fill["family"],
                     "representative_feature": fill_state["feature"],
                     "role": "ENTRY_FILL_STATE", "observation": fill["observation"]},
                ],
                "entry_states": [temporal_state, fill_state],
                "temporal_entry": temporal,
                "selection_metrics": {
                    "execution_temporal_effect_strength": temporal_effect,
                    "fill_effect_strength": float(fill["effect_strength"]),
                    "compound_priority": score,
                },
                "warnings": [*list(temporal_candidate.get("warnings") or []),
                             *list(fill["warnings"])],
                "limits": [
                    "변화량은 원 Profile BACKGROUND anchor의 canonical fixed-exit 수익/손실 대조에서 왔다.",
                    "FILLED/ENTRY_UNFILLED은 현재 BID1 queue 체결 가능성 대조이며, 두 상태 모두 entry만 제한한다.",
                    "두 조건이 같은 결정 tick에 함께 충족할 때만 진입하며 fixed canonical exit은 변경하지 않는다.",
                    "고정 canonical exit Search·Validation·Final Backtest가 별도로 반증한다.",
                ],
            }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate["entry_signal_equivalence_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= EXECUTION_TEMPORAL_FILL_COMPOUND_LIMIT:
            break
    return selected, rejected


def _execution_outcome_pair_compounds(
        outcome_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """같은 fixed-exit outcome 대조에서 나온 서로 다른 두 상태를 함께 시험한다.

    기존 outcome×fill은 “수익 상태”와 “체결 가능 상태”라는 다른 질문의 결합이다.
    이 경로는 그와 달리 실행 수익/손실 대조에서 각각 독립적으로 남은 두 상태가 같은
    결정 tick에 동시에 있을 때만 진입한다. 효과 곱은 실행 순서를 정할 뿐 수익 판정은
    아니며, Search가 두 상태가 함께 남는지 별도로 반증한다.
    """
    ranked: list[tuple[float, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for index, left in enumerate(outcome_candidates):
        left_state = dict((left.get("entry_states") or [{}])[0])
        left_basis = dict((left.get("evidence_basis") or [{}])[0])
        left_effect = float((left.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for right in outcome_candidates[index + 1:]:
            right_state = dict((right.get("entry_states") or [{}])[0])
            right_basis = dict((right.get("evidence_basis") or [{}])[0])
            if (str(left_state.get("feature")) == str(right_state.get("feature"))
                    or _same_quantile_feature_group(left_state, right_state)
                    or str(left_state.get("family")) == str(right_state.get("family"))):
                rejected.append({
                    "left_evidence_id": left_basis.get("evidence_id"),
                    "right_evidence_id": right_basis.get("evidence_id"),
                    "reasons": ["SAME_FEATURE_OR_MONOTONIC_EQUIVALENT_OR_FAMILY"],
                })
                continue
            states = sorted([left_state, right_state],
                            key=lambda item: (str(item.get("feature")), str(item.get("direction"))))
            fingerprint = "&".join(
                f"{state['feature']}:{state['direction']}" for state in states)
            left_id, right_id = sorted((str(left_basis.get("evidence_id")),
                                        str(right_basis.get("evidence_id"))))
            right_effect = float((right.get("selection_metrics") or {}).get("effect_strength") or 0.0)
            score = left_effect * right_effect
            ranked.append((score, {
                "hypothesis_id": f"DE_EXECUTION_OUTCOME_PAIR__{left_id}__{right_id}",
                "title": (f"실행 수익 상태의 동시 조건 — {left_state['feature']} "
                          f"+ {right_state['feature']}"),
                "source_type": "DIRECT_EXECUTION_OUTCOME_PAIR_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": _states_equivalence_fingerprint(states),
                "evidence_basis": [
                    {**left_basis, "role": "EXECUTED_OUTCOME_PAIR_STATE_A"},
                    {**right_basis, "role": "EXECUTED_OUTCOME_PAIR_STATE_B"},
                ],
                "entry_states": states,
                "selection_metrics": {
                    "left_effect_strength": left_effect,
                    "right_effect_strength": right_effect,
                    "compound_priority": score,
                },
                "warnings": [*list(left.get("warnings") or []),
                             *list(right.get("warnings") or [])],
                "limits": [
                    "두 상태는 원 Profile BACKGROUND anchor의 독립 canonical replay 수익/손실 대조에서 왔다.",
                    "같은 결정 tick의 동시 조건이며, 시간 순서·인과·전략 PnL을 주장하지 않는다.",
                    "효과 곱은 실행 후보 순서일 뿐 결합 효과나 수익을 뜻하지 않는다.",
                    "고정 canonical exit Search·Validation·Final Backtest가 두 상태의 결합을 별도로 반증한다.",
                ],
            }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate.get("entry_signal_equivalence_fingerprint")
                          or candidate["entry_logic_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= EXECUTION_OUTCOME_PAIR_COMPOUND_LIMIT:
            break
    return selected, rejected


def _execution_outcome_fill_compounds(
        package: Mapping[str, Any], outcome_candidates: list[dict[str, Any]],
        config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """수익/손실 대조 상태와 체결/미체결 대조 상태를 같은 tick에서 함께 요구한다.

    둘은 같은 독립 canonical anchor replay에서 왔지만 라벨 질문이 다르다. 앞은 체결된
    거래 중 어떤 상태가 수익과 갈리는지, 뒤는 주문이 실제로 queue를 통과했는지를 본다.
    각각의 효과가 있는 경우에만 AND 후보를 만들고, 최종 성과 판단은 별도 strategy
    backtest에 남긴다.
    """
    fill_states, rejected = _execution_fill_states(package, config)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for outcome in outcome_candidates:
        outcome_state = dict((outcome.get("entry_states") or [{}])[0])
        outcome_basis = dict((outcome.get("evidence_basis") or [{}])[0])
        outcome_effect = float((outcome.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for fill in fill_states:
            if _same_quantile_feature_group(outcome_state, fill["state"]):
                rejected.append({
                    "outcome_evidence_id": outcome_basis.get("evidence_id"),
                    "fill_evidence_id": fill["evidence_id"],
                    "reasons": ["DUPLICATE_OR_MONOTONICALLY_EQUIVALENT_FEATURE"],
                })
                continue
            states = sorted([outcome_state, dict(fill["state"])],
                            key=lambda item: (str(item.get("feature")), str(item.get("direction"))))
            fingerprint = "&".join(
                f"{state['feature']}:{state['direction']}" for state in states)
            outcome_id, fill_id = str(outcome_basis.get("evidence_id")), str(fill["evidence_id"])
            score = outcome_effect * float(fill["effect_strength"])
            ranked.append((score, {
                "hypothesis_id": f"DE_EXECUTION_OUTCOME_FILL__{outcome_id}__{fill_id}",
                "title": (f"실행 수익 상태와 queue 체결 상태 — {outcome_state['feature']} "
                          f"+ {fill['state']['feature']}"),
                "source_type": "DIRECT_EXECUTION_OUTCOME_FILL_COMPOUND_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": _states_equivalence_fingerprint(states),
                "evidence_basis": [
                    {**outcome_basis, "role": "EXECUTED_OUTCOME_STATE"},
                    {"evidence_id": fill_id, "family": fill["family"],
                     "representative_feature": fill["state"]["feature"],
                     "role": "ENTRY_FILL_STATE", "observation": fill["observation"]},
                ],
                "entry_states": states,
                "selection_metrics": {
                    "outcome_effect_strength": outcome_effect,
                    "fill_effect_strength": float(fill["effect_strength"]),
                    "compound_priority": score,
                },
                "warnings": [*list(outcome.get("warnings") or []), *list(fill["warnings"])],
                "limits": [
                    "두 상태는 원 Profile BACKGROUND anchor의 독립 canonical replay에서 왔다.",
                    "수익/손실과 체결/미체결은 다른 대조 질문으로 기록했으며, 이 후보 자체는 전략 PnL이 아니다.",
                    "같은 결정 tick에서 두 상태가 함께 충족해야 하며 q 하나를 공유한다.",
                    "고정 canonical exit Search·Validation·Final Backtest가 별도로 반증한다.",
                ],
            }))
    # 가능한 조합 전체를 모두 실행하면 한 Profile의 약한 파생 조합이 비용을 독점한다.
    # 두 독립 contrast의 효과 곱을 미리 정한 순서로 쓰고, 같은 실행 fingerprint는 한 번만
    # 남긴다. 이는 수익 선택이 아니라 full backtest에 보낼 제한된 후보 목록이다.
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate.get("entry_signal_equivalence_fingerprint")
                          or candidate["entry_logic_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= config.execution_outcome_fill_compound_limit:
            break
    return selected, rejected


def _execution_stop_avoidance_fill_compounds(
        package: Mapping[str, Any], stop_avoidance: list[dict[str, Any]],
        config: DirectEvidenceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """즉시 손절 회피 상태와 queue 체결 상태가 함께 있는 entry만 별도로 시험한다."""
    fill_states, rejected = _execution_fill_states(package, config)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for avoidance in stop_avoidance:
        avoidance_state = dict((avoidance.get("entry_states") or [{}])[0])
        avoidance_basis = dict((avoidance.get("evidence_basis") or [{}])[0])
        avoidance_effect = float(
            (avoidance.get("selection_metrics") or {}).get("effect_strength") or 0.0)
        for fill in fill_states:
            if _same_quantile_feature_group(avoidance_state, fill["state"]):
                rejected.append({
                    "stop_avoidance_evidence_id": avoidance_basis.get("evidence_id"),
                    "fill_evidence_id": fill["evidence_id"],
                    "reasons": ["DUPLICATE_OR_MONOTONICALLY_EQUIVALENT_FEATURE"],
                })
                continue
            states = sorted([avoidance_state, dict(fill["state"])],
                            key=lambda item: (str(item.get("feature")), str(item.get("direction"))))
            fingerprint = "&".join(
                f"{state['feature']}:{state['direction']}" for state in states)
            avoidance_id, fill_id = str(avoidance_basis.get("evidence_id")), str(fill["evidence_id"])
            score = avoidance_effect * float(fill["effect_strength"])
            ranked.append((score, {
                "hypothesis_id": (
                    f"DE_STOP_AVOIDANCE_FILL__{avoidance_id}__{fill_id}"),
                "title": (f"즉시 손절 회피와 queue 체결 상태 — {avoidance_state['feature']} "
                          f"+ {fill['state']['feature']}"),
                "source_type": "DIRECT_EXECUTION_STOP_AVOIDANCE_FILL_COMPOUND_EVIDENCE",
                "mechanism_status": "NOT_INTERPRETED",
                "entry_logic_fingerprint": fingerprint,
                "entry_signal_equivalence_fingerprint": _states_equivalence_fingerprint(states),
                "evidence_basis": [
                    {**avoidance_basis, "role": "FIXED_EXIT_STOP_AVOIDANCE_STATE"},
                    {"evidence_id": fill_id, "family": fill["family"],
                     "representative_feature": fill["state"]["feature"],
                     "role": "ENTRY_FILL_STATE", "observation": fill["observation"]},
                ],
                "entry_states": states,
                "selection_metrics": {
                    "stop_avoidance_effect_strength": avoidance_effect,
                    "fill_effect_strength": float(fill["effect_strength"]),
                    "compound_priority": score,
                },
                "warnings": [*list(avoidance.get("warnings") or []), *list(fill["warnings"])],
                "limits": [
                    "NON_STOP/STOP과 FILLED/ENTRY_UNFILLED은 서로 다른 fixed-exit Discovery 질문이다.",
                    "NON_STOP은 수익이 아니며, 이 후보는 청산이 아닌 entry 상태만 시험한다.",
                    "두 상태가 같은 결정 tick에 함께 충족할 때만 진입하고, 고정 canonical exit "
                    "Search·Validation·Final Backtest가 별도로 반증한다.",
                ],
            }))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, candidate in sorted(ranked, key=lambda row: (-row[0], row[1]["hypothesis_id"])):
        fingerprint = str(candidate.get("entry_signal_equivalence_fingerprint")
                          or candidate["entry_logic_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(candidate)
        if len(selected) >= config.execution_stop_avoidance_fill_compound_limit:
            break
    return selected, rejected


def _execution_fill_states(package: Mapping[str, Any], config: DirectEvidenceConfig
                           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """체결/미체결 contrast가 안정적인 숫자 상태만 compound의 두 번째 축으로 쓴다."""
    selected, rejected = [], []
    evidence = package.get("execution_fill_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get("filled_vs_unfilled") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _FILL_DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        elif catalog.FEATURES[feature].value_type != "numeric":
            reasons.append("NON_NUMERIC_FEATURE")
        if direction is None:
            reasons.append("NO_FILLED_UNFILLED_DIRECTION")
        if not isinstance(effect, (int, float)) or float(effect) < config.execution_state_min_effect_strength:
            reasons.append("WEAK_EFFECT")
        symbol_unobservable = not isinstance(symbol_ratio, (int, float))
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        selected.append({
            "evidence_id": str(item.get("evidence_id") or ""),
            "family": str(item.get("family") or ""),
            "state": {"feature": feature, "direction": direction,
                      "family": str(item.get("family") or "")},
            "effect_strength": float(effect),
            "warnings": [*list(item.get("warnings") or []),
                         *( ["SYMBOL_STABILITY_UNOBSERVABLE"] if symbol_unobservable else [])],
            "observation": (
                "원 Profile BACKGROUND anchor에서 canonical BID1 queue order의 "
                "FILLED/ENTRY_UNFILLED 대조로 확인된 상태다."),
        })
    return selected, rejected


def _execution_temporal_candidates(package: Mapping[str, Any], config: DirectEvidenceConfig
                                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """실행 label과 저장된 사전 Profile 변화량이 직접 만나는 단일 상태 후보."""
    selected, rejected = [], []
    evidence = package.get("execution_temporal_evidence") or {}
    for item in evidence.get("items") or []:
        contrast = (item.get("snapshot_contrast") or {}).get("profit_vs_loss") or {}
        feature = str((item.get("representative_feature") or {}).get("name") or "")
        direction = _DIRECTION.get(str(contrast.get("direction") or ""))
        effect = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        reasons = []
        if feature not in catalog.FEATURES:
            reasons.append("FEATURE_NOT_IN_CATALOG")
        if direction is None:
            reasons.append("NO_PROFIT_LOSS_DIRECTION")
        if not isinstance(effect, (int, float)) or float(effect) < config.min_effect_strength:
            reasons.append("WEAK_EFFECT")
        if _severe_missingness(item.get("warnings") or [], config.severe_coverage_gap):
            reasons.append("SEVERE_MISSINGNESS_CONFOUND")
        if reasons:
            rejected.append({"evidence_id": item.get("evidence_id"), "reasons": reasons})
            continue
        expression = dict(item.get("expression") or {})
        try:
            catalog.validate_expression(expression)
        except catalog.ExpressionError:
            rejected.append({"evidence_id": item.get("evidence_id"),
                             "reasons": ["INVALID_TEMPORAL_EXPRESSION"]})
            continue
        family = str(item.get("family") or "")
        evidence_id = str(item.get("evidence_id") or "")
        lag_seconds = int((expression or {}).get("lag") or 0)
        if lag_seconds <= 0:
            rejected.append({"evidence_id": item.get("evidence_id"),
                             "reasons": ["INVALID_TEMPORAL_LAG"]})
            continue
        parameter = f"q_difference_{lag_seconds}s_{feature}"
        selected.append({
            "hypothesis_id": f"DE_{evidence_id}",
            "title": f"직접 관측 변화 — {feature} {lag_seconds}초 변화 {direction}",
            "source_type": "DIRECT_DISCOVERY_EXECUTION_TEMPORAL_EVIDENCE",
            "mechanism_status": "NOT_INTERPRETED",
            "entry_logic_fingerprint": f"difference:clock:{lag_seconds}:{feature}:{direction}",
            "evidence_basis": [{
                "evidence_id": evidence_id, "family": family,
                "representative_feature": feature, "role": "TEMPORAL_STATE",
                "observation": f"저장된 Profile의 anchor t-{lag_seconds}초→t 변화량과 canonical 실행 label의 대조다.",
            }],
            "entry_states": [{"feature": feature, "direction": direction, "family": family,
                              "expression": expression, "parameter": parameter}],
            "temporal_entry": {"operator": "DIFFERENCE", "lag_seconds": lag_seconds,
                               "time_basis": "clock", "warmup_ticks": 0,
                               "why_fixed": f"저장된 Feature Profile의 두 사전 시점 t-{lag_seconds}초와 t를 쓴다."},
            "selection_metrics": {"effect_strength": float(effect),
                                  "date_agreement": (float(date_ratio)
                                                     if isinstance(date_ratio, (int, float)) else None),
                                  "symbol_agreement": (float(symbol_ratio)
                                                       if isinstance(symbol_ratio, (int, float)) else None)},
            "warnings": list(item.get("warnings") or []),
            "limits": [
                "저장된 사전 Profile 변화량으로 선택된 Discovery 관측이다.",
                "경제적 메커니즘·인과·체결 우위를 주장하지 않는다.",
                "동일한 canonical fixed exit Search·Validation·Final Backtest가 별도로 반증할 수 있다.",
            ],
        })
    return _deduplicate_single_state_candidates(
        selected, rejected, reason="MONOTONICALLY_EQUIVALENT_EXECUTION_TEMPORAL_STATE"), rejected


def _relation_candidates(package: Mapping[str, Any], single_candidates: list[dict[str, Any]]
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evidence가 직접 명시한 context→trigger 관계를 두 진입 구조로 보존한다.

    단일 상태 둘을 임의로 묶지 않는다. 각 상태가 단독으로도 안정적인 직접 Evidence여야
    하고, profiler가 관계와 시간 순서를 모두 남긴 경우만 연다. 기존 공동 상태는 같은
    틱 기준선으로 남기고, onset 간격도 실제로 기록됐을 때만 사건형 후보를 별도로 연다.
    """
    by_evidence = {
        str(item["evidence_basis"][0]["evidence_id"]): item
        for item in single_candidates if item.get("evidence_basis")
    }
    selected, rejected = [], []
    onset_by_evidence = {
        str(item.get("evidence_id") or ""): item.get("separation_onset_ms")
        for item in package.get("evidence_families") or []
    }
    for joint in sorted(package.get("joint_evidence") or [],
                        key=lambda item: str(item.get("joint_evidence_id", ""))):
        relation_id = str(joint.get("joint_evidence_id") or "")
        reasons = []
        if joint.get("relation_support") != "SUPPORTED_RELATION":
            reasons.append("RELATION_NOT_SUPPORTED")
        if joint.get("relation") != "CONTEXT_TRIGGER":
            reasons.append("NOT_CONTEXT_TRIGGER")
        if joint.get("ordering_status") != "CONTEXT_BEFORE_TRIGGER":
            reasons.append("ORDER_NOT_OBSERVED")
        context_family = str(joint.get("context_family") or "")
        trigger_family = str(joint.get("trigger_family") or "")
        if not context_family or not trigger_family:
            reasons.append("MISSING_CONTEXT_OR_TRIGGER")
        family_rows = {
            str(joint.get("family_a") or ""): {
                "feature": str(joint.get("feature_a") or ""),
                "evidence_id": str(joint.get("evidence_id_a") or ""),
                "state": str((joint.get("joint_state") or {}).get("state_a") or ""),
            },
            str(joint.get("family_b") or ""): {
                "feature": str(joint.get("feature_b") or ""),
                "evidence_id": str(joint.get("evidence_id_b") or ""),
                "state": str((joint.get("joint_state") or {}).get("state_b") or ""),
            },
        }
        context = family_rows.get(context_family)
        trigger = family_rows.get(trigger_family)
        if context is None or trigger is None:
            reasons.append("RELATION_FAMILY_MISMATCH")
        if reasons:
            rejected.append({"joint_evidence_id": relation_id, "reasons": reasons})
            continue
        states: list[dict[str, str]] = []
        for family, row in ((context_family, context), (trigger_family, trigger)):
            direct = by_evidence.get(row["evidence_id"])
            direction = _STATE_DIRECTION.get(row["state"])
            direct_state = ((direct or {}).get("entry_states") or [{}])[0]
            if (direct is None or direction is None or direct_state.get("feature") != row["feature"]
                    or direct_state.get("direction") != direction):
                reasons.append(f"STATE_NOT_IN_DIRECT_EVIDENCE:{family}")
                break
            states.append({"feature": row["feature"], "direction": direction, "family": family})
        if reasons:
            rejected.append({"joint_evidence_id": relation_id, "reasons": reasons})
            continue
        if len({state["feature"] for state in states}) != 2:
            rejected.append({"joint_evidence_id": relation_id, "reasons": ["DUPLICATE_FEATURE"]})
            continue
        static = {
            "hypothesis_id": f"DE_{relation_id}",
            "title": f"직접 관측 관계 — {states[0]['feature']} → {states[1]['feature']}",
            "source_type": "DIRECT_EVIDENCE_RELATION",
            "mechanism_status": "OBSERVED_RELATION_NOT_INTERPRETED",
            "entry_logic_fingerprint": "&".join(
                f"{state['feature']}:{state['direction']}" for state in states),
            "evidence_basis": [
                {"evidence_id": context["evidence_id"], "family": context_family,
                 "representative_feature": context["feature"], "role": "PERSISTENT_CONTEXT"},
                {"evidence_id": trigger["evidence_id"], "family": trigger_family,
                 "representative_feature": trigger["feature"], "role": "SETUP_STATE"},
            ],
            "relation_basis": {
                "joint_evidence_id": relation_id,
                "relation": "CONTEXT_TRIGGER",
                "ordering_status": "CONTEXT_BEFORE_TRIGGER",
                "context_family": context_family,
                "trigger_family": trigger_family,
                "relation_reason": joint.get("relation_reason"),
            },
            "entry_states": states,
            "selection_metrics": {
                "joint_rate_diff": (joint.get("joint_state") or {}).get("rate_diff"),
                "date_agreement": (joint.get("date_stability") or {}).get("ratio"),
                "symbol_agreement": (joint.get("symbol_stability") or {}).get("ratio"),
            },
            "warnings": list(joint.get("warnings") or []),
            "limits": [
                "Discovery Evidence에서 명시된 context→trigger 관측 관계다.",
                "경제적 메커니즘, 인과, 체결 우위는 이 후보가 주장하지 않는다.",
                "두 상태는 같은 결정 시점에 함께 충족해야 하며, q 하나를 공유한다.",
            ],
        }
        selected.append(static)
        temporal = _temporal_relation_candidate(
            relation_id, context, trigger, states, joint, onset_by_evidence)
        if temporal is not None:
            selected.append(temporal)
            selected.append(_temporal_episode_relation_candidate(temporal))
    return selected, rejected


def _incremental_relation_candidates(package: Mapping[str, Any], single_candidates: list[dict[str, Any]]
                                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """시간 순서가 없는, 직접 지지된 두 독립 상태의 동시 관측 후보.

    `CONTEXT_INDEPENDENT`는 원인·순서를 뜻하지 않는다. 두 상태를 같은 tick에 함께
    관측하는 것이 단독 상태보다 정보가 있는지만 별도 후보로 둔다.
    """
    by_evidence = {
        str(item["evidence_basis"][0]["evidence_id"]): item
        for item in single_candidates if item.get("evidence_basis")
    }
    selected, rejected = [], []
    for joint in sorted(package.get("joint_evidence") or [],
                        key=lambda item: str(item.get("joint_evidence_id", ""))):
        relation_id = str(joint.get("joint_evidence_id") or "")
        reasons = []
        if joint.get("relation_support") != "SUPPORTED_RELATION":
            reasons.append("RELATION_NOT_SUPPORTED")
        if joint.get("relation") != "CONTEXT_INDEPENDENT":
            reasons.append("NOT_CONTEXT_INDEPENDENT")
        if joint.get("ordering_status") != "ORDER_UNCLEAR":
            reasons.append("ORDER_NOT_UNCLEAR")
        rows = [
            (str(joint.get("family_a") or ""), str(joint.get("feature_a") or ""),
             str(joint.get("evidence_id_a") or ""), str((joint.get("joint_state") or {}).get("state_a") or "")),
            (str(joint.get("family_b") or ""), str(joint.get("feature_b") or ""),
             str(joint.get("evidence_id_b") or ""), str((joint.get("joint_state") or {}).get("state_b") or "")),
        ]
        states: list[dict[str, str]] = []
        for family, feature, evidence_id, state in rows:
            direct = by_evidence.get(evidence_id)
            direction = _STATE_DIRECTION.get(state)
            direct_state = ((direct or {}).get("entry_states") or [{}])[0]
            if (not family or direct is None or direction is None
                    or direct_state.get("feature") != feature or direct_state.get("direction") != direction):
                reasons.append(f"STATE_NOT_IN_DIRECT_EVIDENCE:{family or 'UNKNOWN'}")
                break
            states.append({"feature": feature, "direction": direction, "family": family})
        if len({state["feature"] for state in states}) != 2:
            reasons.append("DUPLICATE_FEATURE")
        if reasons:
            rejected.append({"joint_evidence_id": relation_id, "reasons": reasons})
            continue
        selected.append({
            "hypothesis_id": f"DE_INC_{relation_id}",
            "title": f"직접 관측 독립 상태 결합 — {states[0]['feature']} & {states[1]['feature']}",
            "source_type": "DIRECT_EVIDENCE_INCREMENTAL_RELATION",
            "mechanism_status": "OBSERVED_INCREMENTAL_RELATION_NOT_INTERPRETED",
            "entry_logic_fingerprint": "&".join(
                f"{state['feature']}:{state['direction']}" for state in states),
            "evidence_basis": [
                {"evidence_id": evidence_id, "family": family,
                 "representative_feature": feature, "role": "SUPPORTING_STATE"}
                for family, feature, evidence_id, _state in rows
            ],
            "relation_basis": {
                "joint_evidence_id": relation_id,
                "relation": "CONTEXT_INDEPENDENT",
                "ordering_status": "ORDER_UNCLEAR",
                "families": [state["family"] for state in states],
                "relation_reason": joint.get("relation_reason"),
            },
            "entry_states": states,
            "selection_metrics": {
                "joint_rate_diff": (joint.get("joint_state") or {}).get("rate_diff"),
                "date_agreement": (joint.get("date_stability") or {}).get("ratio"),
                "symbol_agreement": (joint.get("symbol_stability") or {}).get("ratio"),
            },
            "warnings": list(joint.get("warnings") or []),
            "limits": [
                "Discovery Evidence에서 독립 상태의 결합이 직접 지지됐다.",
                "두 상태 사이의 시간 순서·인과·경제적 메커니즘은 주장하지 않는다.",
                "같은 결정 tick의 동시 상태만 사용하며, q 하나를 공유한다.",
            ],
        })
    return selected, rejected


def _temporal_relation_candidate(relation_id: str, context: Mapping[str, Any],
                                 trigger: Mapping[str, Any], states: list[dict[str, str]],
                                 joint: Mapping[str, Any], onset_by_evidence: Mapping[str, Any]
                                 ) -> dict[str, Any] | None:
    """기록된 onset 차이만 사건형 구조의 고정 시간 폭으로 쓴다.

    이 시간 폭은 q와 별개다. Discovery 결과를 보고 3초·5초처럼 다시 고르는 축을
    만들지 않는다. `sequence`의 setup은 최근 context 상태, trigger는 trigger 상태가
    직전 100 tick 분위 경계를 새로 넘는 순간이다.
    """
    context_onset = onset_by_evidence.get(str(context.get("evidence_id") or ""))
    trigger_onset = onset_by_evidence.get(str(trigger.get("evidence_id") or ""))
    if (not isinstance(context_onset, (int, float))
            or not isinstance(trigger_onset, (int, float))
            or float(context_onset) >= float(trigger_onset)):
        return None
    lag_ms = float(trigger_onset) - float(context_onset)
    if lag_ms <= 0.0 or lag_ms % 1000.0:
        return None
    lag_seconds = int(lag_ms / 1000.0)
    context_state, trigger_state = states
    return {
        "hypothesis_id": f"DE_TEV_{relation_id}",
        "title": (f"직접 관측 사건 관계 — {context_state['feature']} 뒤 "
                  f"{trigger_state['feature']} 경계 통과"),
        "source_type": "DIRECT_EVIDENCE_TEMPORAL_RELATION",
        "mechanism_status": "OBSERVED_TEMPORAL_RELATION_NOT_INTERPRETED",
        "entry_logic_fingerprint": (
            f"{context_state['feature']}:{context_state['direction']}"
            f"->cross:{trigger_state['feature']}:{trigger_state['direction']}"
            f"@{lag_seconds:g}s"),
        "evidence_basis": [
            {"evidence_id": context["evidence_id"], "family": context_state["family"],
             "representative_feature": context_state["feature"], "role": "PERSISTENT_CONTEXT"},
            {"evidence_id": trigger["evidence_id"], "family": trigger_state["family"],
             "representative_feature": trigger_state["feature"], "role": "TRIGGER_EVENT"},
        ],
        "relation_basis": {
            "joint_evidence_id": relation_id,
            "relation": "CONTEXT_TRIGGER",
            "ordering_status": "CONTEXT_BEFORE_TRIGGER",
            "context_family": context_state["family"],
            "trigger_family": trigger_state["family"],
            "relation_reason": joint.get("relation_reason"),
        },
        "entry_states": states,
        "temporal_entry": {
            "operator": "SEQUENCE",
            "time_basis": "clock",
            "min_lag_seconds": 0.0,
            "max_lag_seconds": lag_seconds,
            "context_onset_ms": int(context_onset),
            "trigger_onset_ms": int(trigger_onset),
            "trigger_event": "CROSS_PRIOR_100_TICKS_PERCENTILE",
            "warmup_ticks": 1,
            "why_fixed": (
                "Evidence가 기록한 context와 trigger의 separation onset 차이다. "
                "별도 시간 폭 탐색은 하지 않는다."),
        },
        "selection_metrics": {
            "joint_rate_diff": (joint.get("joint_state") or {}).get("rate_diff"),
            "date_agreement": (joint.get("date_stability") or {}).get("ratio"),
            "symbol_agreement": (joint.get("symbol_stability") or {}).get("ratio"),
        },
        "warnings": list(joint.get("warnings") or []),
        "limits": [
            "Discovery Evidence에서 명시된 context→trigger 관측 관계다.",
            "경제적 메커니즘, 인과, 체결 우위는 이 후보가 주장하지 않는다.",
            "Search는 직전 100 tick 분위 q 하나만 연다. onset 차이·결정 기회·체결·청산은 고정한다.",
        ],
    }


def _temporal_episode_relation_candidate(temporal: Mapping[str, Any]) -> dict[str, Any]:
    """같은 관측 사건을 context 연속 구간당 한 번만 소비하는 별도 후보로 보존한다."""
    candidate = {key: value for key, value in temporal.items()}
    entry = dict(candidate["temporal_entry"])
    entry["operator"] = "SEQUENCE_ONCE_PER_CONTEXT_EPISODE"
    entry["why_fixed"] = (
        "context→trigger의 Evidence 시간 폭은 그대로 쓰고, 같은 context 연속 구간에서 "
        "trigger를 한 번만 사건으로 소비한다. 재진입 간격은 추가하지 않는다.")
    candidate.update({
        "hypothesis_id": str(candidate["hypothesis_id"]).replace("DE_TEV_", "DE_TEE_", 1),
        "title": str(candidate["title"]).replace("사건 관계", "context 구간 사건 관계", 1),
        "source_type": "DIRECT_EVIDENCE_TEMPORAL_EPISODE_RELATION",
        "entry_logic_fingerprint": str(candidate["entry_logic_fingerprint"]) + ":once_per_context_episode",
        "temporal_entry": entry,
        "limits": [
            *list(candidate.get("limits") or []),
            "한 context 연속 구간에서 trigger를 한 번만 소비한다. 이 의미는 별도 후보로 검증한다.",
        ],
    })
    return candidate


def run(candidate: Mapping[str, Any], package: Mapping[str, Any], output: Any) -> dict[str, Any]:
    return X.run_from_direct_evidence(candidate, package, output)


def _severe_missingness(warnings: list[Any], threshold: float) -> bool:
    for warning in warnings:
        if "MISSINGNESS_CONFOUND" not in str(warning):
            continue
        values = [float(value) for value in re.findall(r"(\d+\.\d+)", str(warning))]
        if values and max(values) >= threshold:
            return True
    return False
