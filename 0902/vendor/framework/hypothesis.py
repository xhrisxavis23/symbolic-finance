"""관측된 증거에서 실행 가능한 수익 가설을 만든다. 전략 파라미터는 만들지 않는다.

## 왜 Agent 를 여러 개 두지 않는가

구 구조는 같은 문장을 Agent 5개가 각자 옮기게 한 뒤 **발굴 PnL 로 하나를 골랐다.**
그런데 PnL 은 번역이 맞다는 증거가 아니고, 다수 합의도 수익성의 증거가 아니다.
여기서는 **한 역할**이 정해진 순서를 밟는다 — 생성 한 번, 같은 역할의 증거 감사 한 번.
그러면 "Agent × N → winner" 로 되돌아갈 자리가 아예 없다.

    PASS 0  입력 감사        코드. 증거가 가설을 지탱하는가
    PASS 1  증거 정리        LLM. 해석 전에 관측을 역할별로 나눈다
    PASS 2  수익 가설 구성  LLM. 실행 뒤 순수익이 남을 수 있는 관측 상태를 정한다
    PASS 3  시간 이야기      LLM. profiler 가 준 시간 증거만
    PASS 4  구현 방향        LLM. 같은 수익 가설을 관측식으로 어떻게 다르게 시험할지
    PASS 5  반증             LLM. 무엇이 관측되면 틀린 것인가
    PASS 6  최종 컴파일      LLM(감사) + 코드(결정적 검증)

## 관측은 탐색 후보로 남긴다

입력 감사에서 사용할 수 있는 Evidence가 있으면 최소 하나의 검증 가능한 후보를 만든다.
약하거나 불안정한 관측은 생성 거절 사유가 아니다. `LOW` confidence와 경고를 붙인
`REDUCED_FORM_TESTABLE` 가설로 남기고, 수익성은 이후 Search가 판단한다.

## 이 모듈이 만들지 않는 것

임계·진입 규칙·청산 규칙·보유 기간·가드 조합·파라미터 탐색·계약. 그것은 이후
Finance-Grounded Specification 과 Contract 의 몫이다. 여기서 경제적 설명은 선택적인
해석이며, 실행 뒤 순수익 가능성은 아직 검증 전 가설이다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (capability as C, catalog, mechanism_graph as M, profit_target as PT,
               sample_condition as SC)
from .config import now_utc, read_json, sha256_json, write_json
# Codex 호출 자체는 agents/runtime.py 가 갖는다. 과거 artifact 를 읽는 코드가
# `H.run_agent` 로 부르고 있어 이름만 여기서 다시 내보낸다.
from .agents.runtime import (AGENT_TIMEOUT_SECONDS, MODEL, REASONING_EFFORT,  # noqa: F401
                            AgentCallError, parse_payload, run_agent)


SCHEMA_VERSION = "hypothesis_generation.v3"
PROMPT_VERSION = "v5"

GENERATED = "HYPOTHESES_GENERATED"
# 과거 artifact를 읽기 위한 값이다. 새 Hypothesis 실행은 이 상태를 만들지 않는다.
LEGACY_NO_HYPOTHESIS = "NO_HYPOTHESIS"
DUPLICATE_HYPOTHESIS = "DUPLICATE_HYPOTHESIS"
STATUSES = (GENERATED,)

EVIDENCE_ROLES = ("PERSISTENT_CONTEXT", "SETUP", "TRIGGER", "SUPPORTING_STATE",
                  "WEAK_OR_AMBIGUOUS")
SEQUENCE_STATUS = ("OBSERVED", "PREDICTED", "TO_BE_TESTED")
TEST_TYPES = ("MATCHED_CONTROL", "TEMPORAL_PRECEDENCE", "DOSE_RESPONSE",
              "INCREMENTAL_EFFECT", "PATH_TEST")
OBSERVABILITY = ("DIRECT", "PROXY", "WEAK_PROXY", "UNOBSERVABLE")
CONFIDENCE = ("HIGH", "MEDIUM", "LOW")
# grounding 개념이 없으면 메커니즘이 성립하지 않는가(CORE), 설명만 돕는가(SUPPORTING).
# v1.1 에서는 스키마만 받아 둔다 — 이것으로 가설을 자동 제거하지 않는다 (§58).
GROUNDING_ROLES = ("CORE", "SUPPORTING")
IMPLEMENTATION_REPRESENTATIONS = ("LEVEL", "PRE_ANCHOR_CHANGE", "PERSISTENCE",
                                  "SEQUENCE", "COMPOSITE")

# 가설이 고를 수 있는 증거 구조 (§17~§21). 증거 관계가 허용하는 것만 고를 수 있다.
STRUCTURES = ("COUPLED_MECHANISM", "CONTEXT_TRIGGER", "CONTEXT_INDEPENDENT",
              "SINGLE_MECHANISM")
EVIDENCE_ROLES_IN = ("PERSISTENT_CONTEXT", "SETUP_STATE", "LATE_TRIGGER",
                     "SUPPORTING_STATE", "REDUNDANT", "UNRESOLVED")
RELATION_TYPES = ("COUPLED", "CONTEXT_TRIGGER", "CONTEXT_INDEPENDENT", "SUPPORTING",
                  "REDUNDANT", "UNRELATED", "UNRESOLVED")
RELATION_STATUS = ("OBSERVED_ORDERING", "SUPPORTED_RELATION", "INDEPENDENT",
                   "TO_BE_TESTED", "REDUNDANT")
OBSERVABILITY_ROLES = ("PERSISTENT_CONTEXT", "SETUP_STATE", "LATE_TRIGGER",
                       "MECHANISM_INTERPRETATION", "SUPPORTING_STATE")

# ---- v1.2 ---------------------------------------------------------------------
# 같은 Agent 가 두 가지 일을 한다. 어느 쪽인지는 workflow 가 정한다 — Agent 가 고르지 않는다.
GENERATE, REVISE = "GENERATE", "REVISE"
MODES = (GENERATE, REVISE)

# 이 데이터 환경에서 가설을 어디까지 시험할 수 있는가 (§20).
# 메커니즘을 직접 못 봐도 예측을 시험할 수 있으면 정상 가설이다 (§21).
DATA_FIT = ("FULLY_TESTABLE", "REDUCED_FORM_TESTABLE", "DATA_BLOCKED")
CORE_OBSERVABILITY = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")
MECHANISM_OBSERVABILITY = ("IDENTIFIABLE", "PROXY_ONLY", "UNRESOLVED")
PREDICTION_TESTABILITY = ("AVAILABLE", "PARTIAL", "UNAVAILABLE")
VALIDATION_FEASIBILITY = ("READY", "PARTIAL", "BLOCKED")

# Revision 에서 각 주장에 붙이는 처리 (§37). 새 주장은 만들 수 없다.
CHANGE_TYPES = ("KEEP", "REMOVE", "DOWNGRADE", "NARROW", "REPHRASE")
CLAIM_STATUS = ("SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED", "BLOCKED_BY_CAPABILITY",
                "INCONCLUSIVE")

# Revision 결과 상태 (§36, §41).
REVISED, NEW_DISCOVERY_REQUIRED = "HYPOTHESIS_REVISED", "NEW_DISCOVERY_REQUIRED"
REVISION_STATUSES = (REVISED, NEW_DISCOVERY_REQUIRED)
REVISED_UNVALIDATED = "REVISED_UNVALIDATED"

# 반박된 주장을 그대로 들고 있으면 안 되는 것들 (§32).
CONTRADICTED_STATUS = ("NOT_SUPPORTED", "CONTRADICTED")

MAX_HYPOTHESES = 3

# 오프라인 품질 rubric (§62). H01–H09 가 갖춘 속성을 축으로 삼는다 — 다만 그 문서를
# production 프롬프트의 few-shot 으로 넣지 않는다. 넣으면 흡수·모주문·유동성 공백
# 세 이야기로 Agent 를 몰아간다. 채점은 사람이 한다. 총점 컷은 코드에 박지 않는다.
RUBRIC_AXES = (
    ("execution_edge", "BID1 대기·비용 뒤에도 수익이 남을 수 있는 관측 상태인가"),
    ("temporal_coherence", "관측된 시간 증거와 이야기의 순서가 맞는가"),
    ("economic_edge", "왜 미래의 실행 가능한 움직임이 남는지 설명하는가"),
    ("falsifiability", "틀렸다고 볼 관측이 실제로 실패할 수 있는가"),
    ("alternative_quality", "대안 설명과 그것을 구분할 관측이 있는가"),
    ("observability_awareness", "proxy 위험을 스스로 표시하는가"),
)
RUBRIC_MAX_PER_AXIS = 2
RUBRIC_REFERENCE = "docs/project_portal_mockups/hybrid_site/tick-profit-hypotheses-H01-H09.ko.md"


def rubric_template(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """사람이 채울 채점표. 자동 채점하지 않는다 — 메커니즘의 질은 코드가 못 잰다."""
    return [{"hypothesis_id": item.get("hypothesis_id"),
             "title": item.get("title"),
             "scores": {axis: None for axis, _ in RUBRIC_AXES},
             "max_per_axis": RUBRIC_MAX_PER_AXIS,
             "max_total": RUBRIC_MAX_PER_AXIS * len(RUBRIC_AXES),
             "reference": RUBRIC_REFERENCE,
             "axis_meaning": dict(RUBRIC_AXES)}
            for item in payload.get("hypotheses") or []]


@dataclass(frozen=True)
class AgentConfig:
    """PASS 0 의 기준과 호출 설정. 코드에 숨은 상수를 두지 않는다."""

    filter_evidence_by_strength: bool = False
    min_effect_strength: float = 0.10      # |AUC - 0.5|
    min_date_agreement: float = 0.80
    min_symbol_agreement: float = 0.70
    execution_state_min_effect_strength: float = 0.05
    min_usable_families: int = 2
    severe_coverage_gap: float = 0.30
    model: str = MODEL
    effort: str = REASONING_EFFORT
    max_hypotheses: int = MAX_HYPOTHESES
    tool_timeout_seconds: int = 720
    # 새 생성은 실행 상태까지 같은 계약으로 남긴다. 과거 artifact 재검증은 명시적으로
    # False를 써서 원문을 고치지 않고 읽을 수 있다.
    require_mechanism_graph: bool = True




# ---- 입력 증거 기록 (코드) -----------------------------------------------------

def execution_temporal_evidence_items(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """실행 label과 결합된 저장 Profile 변화량을 Agent 계약용 Evidence로 펼친다.

    원본 ``evidence_families``를 바꾸지 않는다. 변화량은 같은 family의 level Evidence와
    다른 시간 표현이므로, 기존 Evidence Package의 정본 구조를 보존한 채 별도 source를
    표시한다. 각 항목은 Profile에 실제 저장된 ``t-lag``와 ``t``만 쓴 pre-anchor
    관측이며, Agent가 임의 lag나 새 feature를 만들 수 없다.
    """
    items: list[dict[str, Any]] = []
    for raw in ((package.get("execution_temporal_evidence") or {}).get("items") or []):
        if not isinstance(raw, Mapping):
            continue
        feature = str((raw.get("representative_feature") or {}).get("name") or "")
        if feature not in catalog.FEATURES or not raw.get("evidence_id"):
            continue
        spec = catalog.FEATURES[feature]
        representative = {
            **dict(raw.get("representative_feature") or {}),
            "name": feature,
            "observable_description": spec.observable_description,
            "economic_interpretation": spec.economic_interpretation,
        }
        items.append({
            **dict(raw),
            "representative_feature": representative,
            # t 직전 변화는 장기 context가 아니라 anchor에 가까운 관측 상태다. Agent가
            # 이 힌트를 더 느린 persistent context로 바꿔 적을 수는 없다.
            "role": "TRIGGER",
            "evidence_source": "EXECUTION_TEMPORAL_PREANCHOR",
        })
    return items


def execution_state_evidence_items(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """개별 canonical anchor replay label과 t=0 상태를 Agent Evidence로 펼친다.

    이 항목은 원 Profile oracle cohort와 다른 BACKGROUND anchor 표본에서 온다. 각
    anchor가 독립 주문이므로 전략 PnL은 아니지만, 명시된 상태 대 실행 label 대조는
    Agent가 관측 사실로 인용할 수 있다.
    """
    items: list[dict[str, Any]] = []
    for raw in ((package.get("execution_state_evidence") or {}).get("items") or []):
        if not isinstance(raw, Mapping):
            continue
        feature = str((raw.get("representative_feature") or {}).get("name") or "")
        if feature not in catalog.FEATURES or not raw.get("evidence_id"):
            continue
        spec = catalog.FEATURES[feature]
        representative = {
            **dict(raw.get("representative_feature") or {}),
            "name": feature,
            "observable_description": spec.observable_description,
            "economic_interpretation": spec.economic_interpretation,
        }
        items.append({
            **dict(raw),
            "representative_feature": representative,
            "role": "TRIGGER",
            "evidence_source": "CANONICAL_EXECUTION_ANCHOR_STATE",
        })
    return items


def raw_path_evidence_items(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """고정 exit과 분리한 30초 raw path 대조를 Agent Evidence로 펼친다.

    이 label은 다음 진입 상태를 찾는 Discovery 진단이다. canonical exit의 실현
    결과나 정책을 Agent가 학습하는 입력이 아니며, 각 path cohort를 실제 tool로
    관측한 뒤에만 메커니즘 설명에 쓸 수 있다.
    """
    sources = (
        ("execution_recovery_path_evidence", "recovery_vs_persistent_adverse",
         "RAW_30S_NET_RECOVERY_PATH", "NET_RECOVERY", "PERSISTENT_ADVERSE"),
        ("execution_early_reversal_path_evidence", "early_reversal_vs_persistent_adverse",
         "RAW_30S_EARLY_REVERSAL_PATH", "EARLY_RECOVERY_LATE_REVERSAL",
         "PERSISTENT_ADVERSE"),
    )
    items: list[dict[str, Any]] = []
    for package_key, contrast_key, source, positive_label, negative_label in sources:
        for raw in ((package.get(package_key) or {}).get("items") or []):
            if not isinstance(raw, Mapping):
                continue
            feature = str((raw.get("representative_feature") or {}).get("name") or "")
            if feature not in catalog.FEATURES or not raw.get("evidence_id"):
                continue
            spec = catalog.FEATURES[feature]
            representative = {
                **dict(raw.get("representative_feature") or {}),
                "name": feature,
                "observable_description": spec.observable_description,
                "economic_interpretation": spec.economic_interpretation,
            }
            items.append({
                **dict(raw),
                "representative_feature": representative,
                "role": "TRIGGER",
                "evidence_source": source,
                "primary_contrast_key": contrast_key,
                "raw_path_positive_label": positive_label,
                "raw_path_negative_label": negative_label,
            })
    return items


def _agent_evidence_items(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """가설 Agent가 인용할 수 있는 모든 Evidence. Package 원문은 변형하지 않는다."""
    return [*list(package.get("evidence_families", []) or []),
            *execution_state_evidence_items(package),
            *execution_temporal_evidence_items(package),
            *raw_path_evidence_items(package)]


def _primary_contrast(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Evidence source마다 정한 두 cohort 대조만 primary로 읽는다."""
    snapshot = item.get("snapshot_contrast") or {}
    key = str(item.get("primary_contrast_key") or "profit_vs_loss")
    value = snapshot.get(key) or {}
    return value if isinstance(value, Mapping) else {}


def input_audit(package: Mapping[str, Any], config: AgentConfig = AgentConfig()
                ) -> dict[str, Any]:
    """증거의 강도와 안정성을 기록한다. 이 단계는 Agent 호출을 막지 않는다."""
    families = _agent_evidence_items(package)
    notes: list[str] = []
    usable, rejected = [], []

    for item in families:
        contrast = _primary_contrast(item)
        strength = contrast.get("effect_strength")
        date_ratio = (item.get("date_stability") or {}).get("ratio")
        symbol_ratio = (item.get("symbol_stability") or {}).get("ratio")
        why = []
        temporal = item.get("evidence_source") == "EXECUTION_TEMPORAL_PREANCHOR"
        execution_state = item.get("evidence_source") == "CANONICAL_EXECUTION_ANCHOR_STATE"
        raw_path = str(item.get("evidence_source") or "").startswith("RAW_30S_")
        execution_labelled = temporal or execution_state or raw_path
        if config.filter_evidence_by_strength:
            minimum_effect = (config.execution_state_min_effect_strength
                              if (execution_state or raw_path) else config.min_effect_strength)
            if not isinstance(strength, (int, float)) or strength < minimum_effect:
                why.append(f"effect_strength {strength} < {minimum_effect}")
            # 실행 변화량은 각 날짜에 PROFIT/LOSS가 함께 있어야만 재계산된다. 없는 비율은
            # "반대"가 아니므로 후보를 억지로 막지 않는다. 실제로 계산되어 방향이 반대인
            # 경우만 약한 Evidence로 내린다.
            if ((not isinstance(date_ratio, (int, float)) and not execution_labelled)
                    or (isinstance(date_ratio, (int, float))
                        and date_ratio < config.min_date_agreement)):
                why.append(f"date_agreement {date_ratio} < {config.min_date_agreement}")
            if ((not isinstance(symbol_ratio, (int, float)) and not execution_labelled)
                    or (isinstance(symbol_ratio, (int, float))
                        and symbol_ratio < config.min_symbol_agreement)):
                why.append(f"symbol_agreement {symbol_ratio} < {config.min_symbol_agreement}")
        coverage_gap = ((item.get("coverage") or {}).get("gap"))
        if (execution_labelled and isinstance(coverage_gap, (int, float))
                and coverage_gap >= config.severe_coverage_gap):
            why.append(f"coverage_gap {coverage_gap} >= {config.severe_coverage_gap}")
        (usable if not why else rejected).append(
            {"evidence_id": item.get("evidence_id"), "family": item.get("family"),
             "reasons": why})

    severe = [w for w in package.get("global_warnings", [])
              if "MISSINGNESS_CONFOUND" in str(w)
              and _worst_gap(str(w)) >= config.severe_coverage_gap]
    if severe:
        notes.append(f"심한 결측 격차 경고 {len(severe)}건 — 그 시각의 증거는 보조로만")

    # family 안에서 방향이 갈리는 것은 모순이 아니다. 한 family 의 서로 다른 상관 묶음은
    # **다른 양**이다 — `microprice_dev_bps`(수준)와 `microprice_velocity`(변화)가
    # 반대 방향인 것은 그 자체로 정합적이다. 모순으로 막으면 멀쩡한 증거가 죽는다.
    # 대신 Agent 가 두 양을 하나로 묶어 읽지 않게 사실로만 남긴다.
    split = {}
    for item in families:
        if item.get("evidence_id") in set(audit_usable := {u["evidence_id"] for u in usable}):
            contrast = _primary_contrast(item)
            split.setdefault(item.get("family"), set()).add(contrast.get("direction"))
    for name in sorted(f for f, d in split.items() if len(d) > 1):
        notes.append(f"{name} 안에서 묶음마다 방향이 다르다 — 서로 다른 양이므로 "
                     f"하나의 증거로 합쳐 읽지 말 것")

    # 진짜 모순: 같은 묶음이 primary 와 secondary 대조에서 반대를 가리키는데 둘 다 세다.
    # §6 대로 PROFIT-vs-LOSS 를 우선하므로 막지 않고 사실로만 남긴다.
    for item in families:
        if item.get("evidence_id") not in audit_usable:
            continue
        snapshot = item.get("snapshot_contrast") or {}
        primary, secondary = snapshot.get("profit_vs_loss") or {}, snapshot.get("profit_vs_background") or {}
        if (primary.get("direction") and secondary.get("direction")
                and primary["direction"] != secondary["direction"]
                and (secondary.get("effect_strength") or 0) >= config.min_effect_strength):
            notes.append(f"{item.get('evidence_id')}: PROFIT-vs-LOSS 와 "
                         f"PROFIT-vs-BACKGROUND 방향이 다르다 — primary 를 따를 것")

    return {"ok": True, "usable_evidence": [u["evidence_id"] for u in usable],
            "rejected_evidence": rejected,
            "usable_families": sorted({u["family"] for u in usable}),
            "severe_missingness_warnings": severe, "notes": notes}


def _worst_gap(warning: str) -> float:
    numbers = [float(x) for x in re.findall(r"(\d+\.\d+)", warning)]
    return max(numbers) if numbers else 0.0


# ---- Agent 가 보는 증거 --------------------------------------------------------

def evidence_digest(package: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Agent 에게 줄 증거. **family 단위**이고 개별 feature 20개를 흩뿌리지 않는다.

    같은 상관 묶음 안의 feature 는 관련 목록으로 붙여서 준다 — 세 개를 독립 증거
    세 개로 세지 못하게 하려는 것이다.
    """
    usable = set(audit.get("usable_evidence", []))
    rejected = {str(item.get("evidence_id")): list(item.get("reasons") or [])
                for item in audit.get("rejected_evidence", []) if item.get("evidence_id")}
    items = []
    omitted_pre_anchor_changes = []
    for item in _agent_evidence_items(package):
        snapshot = item.get("snapshot_contrast") or {}
        temporal = item.get("temporal_profile") or {}
        rep = item.get("representative_feature") or {}
        is_pre_anchor_change = item.get("evidence_source") == "EXECUTION_TEMPORAL_PREANCHOR"
        is_execution_state = item.get("evidence_source") == "CANONICAL_EXECUTION_ANCHOR_STATE"
        is_raw_path = str(item.get("evidence_source") or "").startswith("RAW_30S_")
        # temporal item은 input audit에서 이미 개별적으로 탈락할 수 있다. 그런 항목을
        # 수십 개 그대로 prompt에 넣으면, 정상적인 level Evidence까지 결측 confound로
        # 보이게 된다. 값은 가설에 쓰지 않고, 제외 사실·이유만 압축해 보존한다.
        if is_pre_anchor_change and item.get("evidence_id") not in usable:
            omitted_pre_anchor_changes.append({
                "evidence_id": item.get("evidence_id"), "family": item.get("family"),
                "reasons": rejected.get(str(item.get("evidence_id")), []),
            })
            continue
        items.append({
            "evidence_id": item.get("evidence_id"),
            "family": item.get("family"),
            "tier": "PRIMARY" if item.get("evidence_id") in usable else "WEAK",
            "evidence_source": item.get("evidence_source", "PROFILE_STATE"),
            "representative_feature": rep.get("name"),
            "observable_description": rep.get("observable_description"),
            "catalog_note": rep.get("economic_interpretation"),
            "correlated_features_same_evidence": item.get("related_features", []),
            "profit_vs_loss": snapshot.get("profit_vs_loss"),
            "profit_vs_background": snapshot.get("profit_vs_background"),
            "primary_contrast": _primary_contrast(item),
            "temporal": {"relative_times_ms": temporal.get("relative_times_ms"),
                         "separation_profit_vs_loss": temporal.get("separation_profit_vs_loss"),
                         "pattern": temporal.get("pattern")},
            "role": item.get("role"), "role_reason": item.get("role_reason"),
            "separation_onset_ms": item.get("separation_onset_ms"),
            "independent_evidence": item.get("independent_evidence", True),
            "derived_from": item.get("derived_from") or None,
            "deterministic_dependency": item.get("deterministic_dependency"),
            "coverage": item.get("coverage"),
            "date_stability": item.get("date_stability"),
            "symbol_stability": item.get("symbol_stability"),
            "warnings": item.get("warnings", []),
            # 변화량은 원 Profile의 모든 시점 곡선이 아니라, contract가 그대로 재현해야
            # 하는 둘의 pre-anchor 관측과 exact DSL만 보인다.
            "pre_anchor_change": ({
                "expression": item.get("expression"),
                "observation_window": item.get("observation_window"),
                "role_constraint": "TRIGGER",
                "interpretation_boundary": (
                    "stored Profile t-lag to t change only; no post-anchor observation "
                    "or lag selection"),
            } if is_pre_anchor_change else None),
            "execution_anchor_state": ({
                "observation_window": item.get("observation_window"),
                "label_source": "canonical_execution_anchor_replay",
                "interpretation_boundary": (
                    "individual anchor replay diagnostic; not aggregate strategy PnL"),
            } if is_execution_state else None),
            "raw_path_state": ({
                "observation_window": item.get("observation_window"),
                "label_source": "canonical_execution_anchor_replay.raw_30s_diagnostic_path",
                "positive_label": item.get("raw_path_positive_label"),
                "negative_label": item.get("raw_path_negative_label"),
                "interpretation_boundary": (
                    "individual canonical queue fill's 30-second BID1 raw-path diagnostic; "
                    "not canonical exit PnL or an exit-policy target"),
            } if is_raw_path else None)})
    execution_evidence = package.get("execution_evidence")
    digest = {"outcome_definition": package.get("outcome_definition"),
              "executable_sample_condition": package.get("executable_sample_condition"),
              "executable_sample_condition_sha256": package.get(
                  "executable_sample_condition_sha256"),
              "cohort_summary": package.get("cohort_summary"),
              "evidence": items,
              "global_warnings": package.get("global_warnings", []),
              "not_provided": package.get("not_provided", []),
              "profile_compatibility": package.get("profile_compatibility", {}),
              "canonical_execution_anchor_replay": package.get(
                  "canonical_execution_anchor_replay")}
    if omitted_pre_anchor_changes:
        digest["omitted_pre_anchor_change_evidence"] = {
            "count": len(omitted_pre_anchor_changes),
            "policy": ("input audit에서 탈락한 pre-anchor change는 가설 근거에 쓸 수 없다. "
                       "다른 PRIMARY level Evidence를 무효로 만들지는 않는다."),
            "items": omitted_pre_anchor_changes,
        }
    if isinstance(execution_evidence, Mapping):
        digest["execution_evidence"] = {
            key: execution_evidence.get(key)
            for key in ("state", "label_source", "source_profile_cohorts", "source_anchor_count",
                        "source_execution_label_counts", "evidence_cohort_counts",
                        "profit_rule", "loss_rule", "background_rule",
                        "interpretation_boundary")
        }
    # package 가 공동 증거를 담고 있으면 그대로 싣는다. 프롬프트 문구는 바꾸지 않는다 —
    # 같은 Agent 가 더 나은 증거를 받았을 때 어떻게 달라지는지가 이번 실험이다.
    if package.get("joint_evidence"):
        digest["evidence_relations"] = [
            {k: v for k, v in item.items()
             if k in ("joint_evidence_id", "family_a", "family_b", "relation",
                      "relation_reason", "relation_support", "role_a", "role_b",
                      "context_family", "trigger_family", "ordering_status",
                      "classification", "summary")}
            for item in package["joint_evidence"]]
        digest["joint_evidence_note"] = package.get("joint_evidence_note")
        digest["independent_evidence_families"] = package.get("independent_evidence_families")
    path_access = package.get("path_observation") or {}
    if path_access.get("anchors"):
        counts: dict[str, int] = {}
        for item in path_access["anchors"]:
            cohort = str(item.get("cohort"))
            counts[cohort] = counts.get(cohort, 0) + 1
        if package.get("price_path_context_required", True):
            digest["price_path_tool"] = {
                "tool_server": path_access.get("tool_name"),
                "horizon_key": path_access.get("horizon_key"),
                "anchors_by_cohort": dict(sorted(counts.items())),
                "policy": path_access.get("policy"),
            }
        else:
            digest["price_path_context"] = {
                "status": "UNAVAILABLE",
                "reason": "저장된 FeatureProfile에 대응 discovery price-path cache가 없다",
            }
    return digest


# ---- 프롬프트 -----------------------------------------------------------------

ROLE = """\
You search for market states that could become profitable execution candidates for strategy design.
Your primary goal is a falsifiable, fee-inclusive net-profit hypothesis under the fixed BID1 queue
execution model. Economic mechanism is an optional interpretation, not an eligibility requirement.
You do not design or optimize a trading strategy.

When at least one usable Evidence item is supplied, you must emit at least one exploratory
hypothesis. Weak effect size, unstable diagnostics, missing temporal ordering, unavailable raw
paths, or an unresolved economic mechanism lower confidence and become explicit limitations;
they do not authorize an empty result. Search, not this Agent, determines profitability.

Hard rules:
- Copy the supplied `profit_target` exactly. It is the fixed result contract, not a
  strategy parameter or an observed result.
- When the Evidence supplies `executable_sample_condition`, copy it exactly. It is the
  Feature Profile's fixed sample condition and is applied before the hypothesis's
  additional observable condition.
- Separate observations from interpretations. Never state a prediction as an observation.
- Do not invent executable features. If a financial concept has no catalog observable,
  name the concept in `grounding_requirements` and leave it unresolved.
- Do not choose thresholds, quantiles, entry rules, exit rules, holding periods,
  position sizes, or any trading parameter. That belongs to a later stage. The only
  execution choice allowed in `mechanism_graph` is one already-supported pending-order
  mode: HOLD_THROUGH, or CANCEL_WHEN_ENTRY_SIGNAL_FALSE using the complete PENDING
  entry signal.
  It never changes the BID1 entry, exit, horizon, or a threshold.
- Features inside one `evidence_id` are correlated measurements of one thing.
  Never count them as separate supporting evidence.
- The Feature Profile is a discovery hint and an anchor sampler, not a causal explanation.
  Inspect raw pre-anchor episodes when the tool is available. When it is unavailable,
  construct the narrowest reduced-form candidate from the supplied Profile Evidence and
  preserve the missing path as a limitation.
- Do not combine evidence families merely because their cards exist. `SINGLE_MECHANISM`
  is allowed whenever one directly observed, non-redundant family supports a distinct
  falsifiable mechanism. Use multiple families only when their supplied relation adds a
  necessary observed part of the same path. A conjunction of feature conditions is not
  a mechanism.
- State what future execution result would show the profit hypothesis is wrong.
- A missing economic explanation is acceptable when a distinct, observable, execution-aware
  profit prediction remains. Mark it unresolved instead of inventing one.
- A reduced-form prediction is allowed when one observable condition is measurable.
  Do not turn two `CONTEXT_INDEPENDENT` or `ORDER_UNCLEAR` Profile cards into a claimed
  coupled mechanism. Keep them separate or use the strongest non-redundant item as a
  single-observable exploratory candidate.

Optional reasoning examples — use these only as a vocabulary and structure guide, not as
templates, facts, feature requirements, or trading rules. A simple observation can be
the correct final hypothesis when that is all the Evidence supports. Do not pad it into
a richer story. Conversely, when Evidence genuinely supports multiple families and their
ordering, do not compress the idea into an unrelated list of conditions.

- Simple observation: "anchor 직전에 ofi_depth_5가 PROFIT에서 더 높다." This is a
  useful observed state, but by itself it does not say whether flow was persistent, was
  a reversal, coincided with a book change, or could still leave an executable move.
- Richer *possible* reading: "매도 압력이 일정 구간에 남아 있고, 그 뒤 bid 깊이의
  회복과 ask 측 약화가 관측되며, 마지막에 단기 가격 변화가 상향 전환한다." This may be
  described as an earlier context followed by a later trigger only when the supplied
  Evidence relation supports that order; it does not establish causal order.
- Another possible reading: "BID1 대기 물량이 먼저 줄어든 뒤 다시 보강되고, 동시에 ask
  측 물량 또는 집중도가 낮아진 상태에서 호가 불균형이 개선된다." This is not evidence of
  participant intent or hidden liquidity. Treat those as unresolved interpretations and
  state the directly observable transition separately.
- Another possible reading: "짧은 하락 또는 공격적 매도 흐름이 지속된 뒤, OFI와
  microprice 변화가 함께 회복 방향을 보인다." It needs an observed temporal relation;
  simultaneous late states must remain CONTEXT_INDEPENDENT when ordering is absent.
- Another possible reading: "깊은 호가의 상대적 bid 지지와 과도하지 않은 spread/cost
  상태가 이미 있고, 이후 단기 회복 event가 나타난다." The depth/spread state is a
  context or execution-relevance question, not proof that an order will fill.

The Catalog can later express supported versions of these shapes with `difference`,
`rolling_zscore`, `rolling_sum`, `persistence`, `crossover`, `sequence`, `ratio`, and
`level_aggregate`. You do not need to emit that DSL or choose its thresholds here.
Describe the economic observation and the falsifiable temporal relation precisely enough
that Grounding can preserve it. Do not mention an operator merely to sound sophisticated:
if Evidence supports only a level state, say only a level state.

Reading the numbers:
- `auc` is the probability a PROFIT anchor ranks above a LOSS anchor. It is NOT a
  model score. `direction` already states the sign; `effect_strength` is |auc - 0.5|.
- `effect_strength`, `date_stability`, and `symbol_stability` are descriptive diagnostics,
  not eligibility filters. Preserve weak or unstable values as limitations. They may support
  an exploratory, falsifiable candidate but never a claim that the relation is strong or stable.
- `separation_profit_vs_loss` is 2*(auc-0.5) at each relative time, oldest first.
  `pattern` is a deterministic label over that trajectory.
- For `PROFILE_STATE`, PROFIT vs LOSS is the PRIMARY contrast: identical eligibility,
  entry price, exit price, horizon, cost and sampling quality — only the future outcome
  differs. For an execution or raw-path Evidence item, its `primary_contrast` and the
  labels stated in `execution_anchor_state` or `raw_path_state` are authoritative;
  never relabel that item as PROFIT vs LOSS.
- PROFIT vs BACKGROUND is SECONDARY context only. Where the two disagree, trust
  PROFIT vs LOSS.

Evidence relationships have explicit meanings. They are given to you; do not invent them.

- `evidence_roles` in a hypothesis contains only the families it actually cites. The
  input map is the fixed role of each evidence family for a multi-family relation. For a
  SINGLE_MECHANISM, copy the role of its exact `EV_*` item; do not replace a persistent
  item with the shorter-lookback representative role of the same family.
- A joint relation's `context_family` or `trigger_family` describes only that pair's
  observed ordering. It never changes the role of the evidence item that supports it.

- PERSISTENT_CONTEXT describes a market condition that appears earlier and remains present.
- LATE_TRIGGER describes a later state change near the outcome anchor.
- SETUP_STATE forms after the context but before the late trigger.
- CONTEXT_TRIGGER permits a temporal description: the context is observed earlier and the
  trigger later. It does NOT permit the claim that the context causes the trigger.
- CONTEXT_INDEPENDENT means both observations may be used in one hypothesis, but their
  contributions must be described independently, never as one coupled state.
- COUPLED may only be claimed where the evidence states relation COUPLED. Do not create it.
- REDUNDANT evidence must not be counted as a separate source of support. One of the pair
  is a deterministic function of the other.
- UNRELATED pairs must not be linked inside one mechanism chain.
- UNRESOLVED relations may be used only when marked TO_BE_TESTED.

Do not turn observed ordering into causal ordering. "A separates earlier than B" is an
observation; "A causes B", "A induces B", "the effect of A operates through B" and
"B is the consequence of A" are not supported by this evidence.

Vocabulary contract. The deterministic contract rejects causal wording in every retained
field. Do not use `때문에`, `으로 인해`, `유발`, `일으키`, `결과이다`, `causes`,
`induces`, `consequence of`, or `operates through`. This includes an otherwise harmless
epistemic sentence such as "관측값이 X를 보여주지 않기 때문에 ..."; rewrite it as
"관측값만으로 X를 알 수 없다. 따라서 ...". State only observed order, a possible
interpretation, or a falsifiable prediction.

What the data environment does and does not decide:

- The research capability profile defines what you may claim to OBSERVE. It does NOT
  define which financial mechanisms you are allowed to CONSIDER.
- You may propose a mechanism that is not directly observable here — temporary liquidity
  pressure, liquidity resilience, inventory adjustment, stale quote adjustment. Label it
  as an interpretation and give at least one falsifiable prediction that can be measured
  under the stated capabilities.
- A mechanism whose core is UNAVAILABLE and which yields no distinct testable prediction
  is a poor primary hypothesis. Say so instead of dressing it up.
- Never replace an unavailable financial concept with an arbitrary available feature just
  to make the hypothesis executable. Leave the concept unresolved instead.
- Never label an UNAVAILABLE or PROXY_ONLY capability as a direct observation.
- VALIDATION_ONLY data exists but you cannot see its values. You may build a prediction
  about it. You may never state it as something you observed.
- The sample is one short period. Do not claim generality "across market regimes".
- Quote cadence differs by symbol by more than an order of magnitude, so a tick-count
  window is not the same amount of financial time in two different symbols.

Prose fields must be written in Korean. Enum values, ids and feature names stay as given.
Return one JSON object and nothing else. No markdown fence, no commentary.
"""

_SCHEMA = """\
{
  "status": "HYPOTHESES_GENERATED",
  "path_observations": [
    {"path_id": str, "cohort": "PROFIT" | "LOSS" | "BACKGROUND",
     "execution_label": optional "EXECUTED_PROFIT" | "EXECUTED_LOSS",
     "diagnostic_cohort": optional raw-path diagnostic cohort,
     "observations": [str]}
  ],
  "loss_ledger_observations": [
    {"case_id": str, "diagnostic_cohort": str, "observation": str,
     "research_question": str}
  ],
  "evidence_relation_map": [
    {"a": family, "b": family, "relation": one of %(relations)s,
     "basis_evidence": [evidence_id or joint_evidence_id], "basis_paths": [path_id], "observation": str}
  ],
  "evidence_synthesis": {
    "PERSISTENT_CONTEXT": [{"evidence_id": str, "observation": str}],
    "SETUP": [...], "TRIGGER": [...], "SUPPORTING_STATE": [...],
    "WEAK_OR_AMBIGUOUS": [...]
  },
  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "title": str,
      "profit_target": %(profit_target)s,
      "executable_sample_condition": Catalog DSL object | null,
      "hypothesis_structure": one of %(structures)s,
      "evidence_roles": {family: one of %(eroles)s},
      "evidence_relations": [{"a": family, "b": family, "relation": one of %(relations)s}],
      "mechanistic_claim": str | null,
      "evidence_basis": [
        {"evidence_id": str, "family": str, "representative_feature": str,
         "observation": str, "role": one of %(roles)s}
      ],
      "expected_observable_sequence": [
        {"stage": int, "claim": str, "status": one of %(seq)s, "evidence": [evidence_id],
         "relation_status": one of %(rstatus)s}
      ],
      "mechanism_graph": {
        "schema": "mechanism_graph.v1",
        "observables": [
          {"node_id": str, "evidence_id": str,
           "available_at": "DECISION" | "PENDING"}
        ],
        "relations": [
          {"node_id": str, "relation": "CONTEXT_TRIGGER" | "COUPLED" | "CONTEXT_INDEPENDENT",
           "input_ids": [observable_node_id], "available_at": "DECISION" | "PENDING"}
        ],
        "execution": {
          "entry_state": "DECISION", "pending_state": "PENDING", "entry_action": "BID1_QUEUE",
          "entry_lifecycle_policy": {
            "mode": "HOLD_THROUGH" | "CANCEL_WHEN_ENTRY_SIGNAL_FALSE",
            "pending_observable_ids": [observable_node_id]
          }
        },
        "prediction": {"input_ids": [observable_or_relation_node_id], "evaluation_state": "FILLED",
                       "target_type": "PATH_TARGET" | "MATCHED_CONTROL_TARGET",
                       "direction": "HIGHER" | "LOWER"},
        "catalog_bindings": []
      },
      "implementation_directions": [
        {"direction_id": "D1", "title": str,
         "representation": one of %(representations)s,
         "evidence_ids": [evidence_id], "observation_focus": str,
         "expected_execution_edge": str}
      ],
      "source_of_profit": str,
      "already_priced_risk": bool,
      "mechanism_tests": [
        {"test_type": one of %(tests)s, "claim": str, "control": str,
         "expected_result": str, "required_evidence": str}
      ],
      "alternative_explanations": [
        {"explanation": str, "why_plausible": str, "discriminating_test": str}
      ],
      "observability": [
        {"concept": str, "level": one of %(obs)s, "note": str}
      ],
      "grounding_requirements": [
        {"concept": str, "role": one of %(oroles)s,
         "importance": one of %(ground)s, "observability": one of %(obs)s}
      ],
      "evidence_warnings": [str],
      "data_fit": {
        "observable_core": one of %(core)s,
        "mechanism_observability": one of %(mobs)s,
        "prediction_testability": one of %(ptest)s,
        "overall": one of %(fit)s
      },
      "capability_requirements": [
        {"concept": str, "capability": <name from the capability profile>,
         "availability": <its status in the profile>}
      ],
      "validation_feasibility": one of %(feas)s,
      "confidence": one of %(conf)s
    }
  ]
}""" % {"roles": list(EVIDENCE_ROLES),
        "seq": list(SEQUENCE_STATUS), "tests": list(TEST_TYPES),
        "obs": list(OBSERVABILITY), "conf": list(CONFIDENCE),
        "structures": list(STRUCTURES), "eroles": list(EVIDENCE_ROLES_IN),
        "relations": list(RELATION_TYPES), "rstatus": list(RELATION_STATUS),
        "oroles": list(OBSERVABILITY_ROLES), "ground": list(GROUNDING_ROLES),
        "representations": list(IMPLEMENTATION_REPRESENTATIONS),
        "core": list(CORE_OBSERVABILITY), "mobs": list(MECHANISM_OBSERVABILITY),
        "ptest": list(PREDICTION_TESTABILITY), "fit": list(DATA_FIT),
        "feas": list(VALIDATION_FEASIBILITY),
        "profit_target": json.dumps(PT.canonical_target(), ensure_ascii=False)}


def generation_prompt(digest: Mapping[str, Any], config: AgentConfig = AgentConfig(),
                      profile: Mapping[str, Any] | None = None) -> str:
    """Invocation A — 초안 생성. 환경 프로필을 같이 준다 (§18)."""
    profile = C.research_capability_profile() if profile is None else profile
    path_tool = digest.get("price_path_tool")
    alignment = ((digest.get("profile_compatibility") or {}).get("entry_execution_alignment") or {})
    alignment_pass = "" if not alignment else f"""
PASS 0.1 — Entry-execution alignment. `profile_compatibility.entry_execution_alignment`
states whether this stored Feature Profile recorded BID1 queue fills before the discovery
oracle ASK. It is an input capability fact, not a result. When its state is
`ENTRY_QUEUE_UNOBSERVED`, do not present a PROFIT oracle path as evidence that the
canonical BID1 queue entry could have filled in time. A mechanism whose executable edge
depends on that missing fact is not defensible from this input.

{json.dumps(alignment, ensure_ascii=False, indent=1)}
"""
    execution_replay = digest.get("canonical_execution_anchor_replay")
    execution_replay_pass = "" if not execution_replay else f"""
PASS 0.15 — Canonical anchor execution replay. This is a fixed-exit, independent
canonical queue replay at every stored Feature Profile anchor. It resolves the missing
fill fact for this Discovery input, but the anchors overlap and do not share position
blocking. Therefore it is not strategy PnL, Validation, Final, OOS, or Evidence for a
new mechanism. Use it only to avoid describing an oracle price opportunity as an
executable fill. Do not cite it in `evidence_basis`, and do not make aggregate replay
PnL a prediction or a source of profit.

When a `get_price_path` response includes `canonical_execution`, it is the same
single-anchor Discovery diagnostic. You may distinguish an oracle price path from its
actual queue fill and fixed exit, but it cannot itself support a new mechanism.

{json.dumps(execution_replay, ensure_ascii=False, indent=1)}
"""
    execution_evidence = digest.get("execution_evidence")
    execution_evidence_pass = "" if not execution_evidence else f"""
PASS 0.2 — Canonical execution diagnostic. This is a separate replay of stored anchors
with the canonical BID1 queue entry and unchanged fixed exit. It does **not** replace the
stored Feature Profile's original PROFIT/LOSS cohort or the primary `EV_*` Evidence
contrast shown below. Do not describe a raw `canonical_execution` path, its aggregate,
or this diagnostic summary as Evidence for a mechanism, strategy PnL, Validation, Final,
or OOS. Its only role is distinguishing a discovery oracle opportunity from an actual
queue fill when that fact is available.

{json.dumps(execution_evidence, ensure_ascii=False, indent=1)}
"""
    execution_anchor_states = [item for item in digest.get("evidence") or []
                               if item.get("evidence_source") == "CANONICAL_EXECUTION_ANCHOR_STATE"]
    execution_state_pass = "" if not execution_anchor_states else """
PASS 0.22 — Canonical anchor execution-state Evidence. Each supplied `EVX_STATE_*`
item compares one stored `t=0` Feature Profile state between independently replayed
canonical execution PROFIT and LOSS anchors. These explicitly listed items may be cited
as direct observations, unlike the aggregate replay summary above. They come only from
original Profile BACKGROUND anchors and are not strategy PnL: never use their aggregate,
fill count, or a raw path as profitability or OOS evidence.

For one such item, write a SINGLE_MECHANISM using exactly that one family. Its observed
state and direction are fixed. Do not mix it with an
oracle `EV_*` level of the same family as if they were independent confirmation, and do
not introduce another family merely to make the mechanism sound richer.
"""
    raw_path_states = [item for item in digest.get("evidence") or []
                       if str(item.get("evidence_source") or "").startswith("RAW_30S_")]
    raw_path_pass = "" if not raw_path_states else """
PASS 0.23 — Raw 30-second path Evidence. Each supplied raw-path item compares an
independently filled Profile BACKGROUND anchor's post-entry 30-second BID1 path between
the exact `raw_path_state.positive_label` and `negative_label` shown in its item. This
is neither canonical exit PnL nor an exit-policy target. It may be cited only as a
Discovery entry-state observation. For one item, write a SINGLE_MECHANISM using exactly
that one family. Do not mix it with an oracle `EV_*`
level of the same family as independent confirmation, and do not tune an exit around
the path label.
"""
    pre_anchor_changes = [item for item in digest.get("evidence") or []
                          if item.get("evidence_source") == "EXECUTION_TEMPORAL_PREANCHOR"]
    temporal_change_pass = "" if not pre_anchor_changes else """
PASS 0.25 — Stored pre-anchor change Evidence. Each supplied `EVT_DIFFERENCE_*` item is
a directly observed change between exactly two stored Feature Profile times, `t-lag` and
`t`, before the anchor. Its expression, lag and TRIGGER role are fixed observations, not
search choices. It uses the same canonical fixed-exit Discovery labels as PASS 0.2 and
may be cited in `evidence_basis` and an OBSERVED sequence.

For one such item, write a SINGLE_MECHANISM using exactly that one evidence family.
Do not silently change its lag, treat it as a persistent context,
or combine multiple lags of the same feature as independent support. The later contract
will replay the exact supplied difference expression; you must not choose a threshold or
an alternative timing expression here.
"""
    execution_state_path_grounding = any(
        item.get("evidence_source") == "CANONICAL_EXECUTION_ANCHOR_STATE"
        for item in digest.get("evidence") or []) and str(
            ((digest.get("research_lens") or {}).get("lens_id") or "")).startswith(
                "single-execution-state:")
    raw_path_path_grounding = bool(raw_path_states) and str(
        ((digest.get("research_lens") or {}).get("lens_id") or "")).startswith(
            "single-raw-path:")
    path_pass = "" if path_tool is None else ("""
PASS 0 — Raw-path price-path observation. This lens uses original Profile `BACKGROUND`
anchors labelled by the canonical queue replay's 30-second BID1 raw path. Before emitting
a hypothesis, call the read-only `list_price_paths` and then `get_price_path` for at
least one `cohort="BACKGROUND", diagnostic_cohort=<positive_label>` path and one
`cohort="BACKGROUND", diagnostic_cohort=<negative_label>` path, using the exact labels
in the cited raw-path Evidence. Record each returned path id with `cohort="BACKGROUND"`
and its exact `diagnostic_cohort` in `path_observations`. Record only returned BID1/ASK1
facts; do not invent or alter them. These are individual Discovery diagnostics, not
strategy PnL, Validation, Final, OOS, or proof that the fixed exit captures the path.
""" if raw_path_path_grounding else """
PASS 0 — Canonical execution-state price-path observation. This lens uses original
Profile `BACKGROUND` anchors, labelled by the canonical fixed-exit anchor replay.
Before emitting a hypothesis, call the read-only `list_price_paths` and then
`get_price_path` for at least one `cohort="BACKGROUND", execution_label="EXECUTED_PROFIT"`
path and one `cohort="BACKGROUND", execution_label="EXECUTED_LOSS"` path. Record each
returned path id with `cohort="BACKGROUND"` and its exact `execution_label` in
`path_observations`. Record only returned BID1/ASK1 facts; do not invent or alter them.
The system records these MCP calls, so a generated hypothesis without both actual reads
is not valid. These are individual anchor diagnostics, not strategy PnL, Validation,
Final, or OOS. Price points and future oracle peaks remain illustrations, never evidence
for the mechanism or execution proof.
""" if execution_state_path_grounding else """
PASS 0 — Evidence-selected Discovery price-path observation. Choose the price paths that
match the Evidence you actually cite in the final hypothesis:

* If it cites any `EVX_STATE_*` canonical execution-state Evidence, first call the
  read-only `list_price_paths` and then `get_price_path` for at least one
  `cohort="BACKGROUND", execution_label="EXECUTED_PROFIT"` path and one
  `cohort="BACKGROUND", execution_label="EXECUTED_LOSS"` path. Record each returned
  path id with `cohort="BACKGROUND"` and its exact `execution_label` in
  `path_observations`.
* If it cites raw 30-second path Evidence, first call `list_price_paths` with the exact
  `diagnostic_cohort` labels from that Evidence and then read one returned BACKGROUND
  path for each of its positive and negative raw-path labels. Record each path id with
  `cohort="BACKGROUND"` and its exact `diagnostic_cohort` in `path_observations`.
* Otherwise, first call `list_price_paths` and then `get_price_path` for at least one
  allowed PROFIT path and one allowed LOSS path. Record their returned path ids and
  observed BID1/ASK1 facts in `path_observations`.

Do not mix the two path contracts. Record only returned facts; do not invent or alter
them. The system records these MCP calls, so a generated hypothesis without the reads
matching its Evidence is not valid. Price paths are individual Discovery diagnostics,
not strategy PnL, Validation, Final, OOS, or execution proof. Future peaks remain
discovery-oracle labels, never fills.
""" if (execution_anchor_states or raw_path_states) else """
PASS 0 — Discovery price-path observation. Before emitting any hypothesis, call the
read-only `list_price_paths` and then `get_price_path` for at least one allowed PROFIT
path and one allowed LOSS path. Record only the returned path ids and observed BID1/ASK1
facts in `path_observations`; do not invent or alter them. The system records those MCP
calls, so a generated hypothesis without those two actual reads is not valid. You may
read additional allowed paths when useful, but cannot select a path outside the tool
result. Future peaks remain discovery-oracle labels, never fills, validation, OOS, or
execution proof.
""")
    mechanism_probe_pass = "" if path_tool is None else """
PASS 0.05 — Pre-anchor mechanism probe. Feature Profile cards are only leads. For the
same required PROFIT/LOSS (or BACKGROUND execution/raw-path) cases from PASS 0, call
`get_pre_anchor_microstructure`. It returns only ticks at or before the anchor: BID1,
ASK1, L1/depth-10 queue, and, when available, buy/sell initiated volume. It never
returns a future price or outcome.

In `path_observations`, record only exact tool-returned pre-anchor facts as well as the
price-path facts. Use those facts to ask one narrow question: what observed pre-anchor
state could leave a BID1 fill exposed to a later reprice, and what economic actor or
liquidity process would have to be true for that to happen? Do not claim that actor was
observed. Do not use a raw field as a new executable feature, choose a threshold, or
search additional anchors for a better result. Any later executable condition still
needs Catalog grounding.

If the PROFIT/LOSS episodes do not support one coherent execution edge, do not rescue it
by joining independent Feature Profile cards. Keep one single-observable reduced-form
prediction and record the missing coherent path as a limitation.
"""
    loss_tool = digest.get("loss_ledger_tool")
    loss_pass = "" if loss_tool is None else """
PASS 0.45 — Current parent entry policy. `discovery_execution_context.prior_entry_policy`
is the earlier algorithm's last Discovery-tested entry guard set, and its listed
`refinement_round` is the exact parent ledger behind the loss tool. It tells you which
entry policy has already been tried. It is neither Evidence nor a claim that these
guards will transfer to a successor. Do not call the policy profitable, do not cite it
in `evidence_basis`, and do not merely rename its mechanism. Use it only to make the
next research question change a real part of the earlier entry logic.

PASS 0.4 — Prior-hypothesis boundary. `discovery_execution_context.prior_hypothesis`
describes the earlier mechanism whose Discovery execution ledger you are reading. Do
not restate that mechanism with new wording. A hypothesis using the same Evidence may
have a genuinely different execution edge or falsifiable prediction. When the mechanism
remains unresolved, retain the narrowest observable candidate and mark the novelty risk.

PASS 0.5 — Discovery execution-loss observation. `discovery_execution_context` is a
fixed-exit execution diagnostic for an earlier hypothesis on the same Discovery input.
Read its aggregate contrast first. Before emitting a hypothesis, call
`list_execution_cases` and then `get_execution_case` for at least one returned case. If
both `PERSISTENT_ADVERSE` and `NET_RECOVERY` cases are available, open one of each and
record both sides; one loss case alone cannot establish what distinguishes recovery.
The system records those MCP calls, so a generated successor without that concrete
comparison is not valid. In `loss_ledger_observations`, record each exact returned
`case_id`, its exact `diagnostic_cohort`, a short observation, and the one next research
question that observation motivates. This context may tell you what question the next
mechanism should answer; it is NOT Evidence for the new hypothesis. Do not cite it in
`evidence_basis`, do not turn its net/gross PnL, exit reason, or cohort into a prediction,
and do not claim it validates a new mechanism. Every retained hypothesis must still be
supported by the supplied Feature Profile Evidence and relation map.

When a returned case has `entry_signal_active_at_fill`, it says only whether the earlier
entry condition was true at that case's actual `fill_tick`; the optional
`entry_signal_persisted_until_fill` says whether it stayed true continuously from posting
through that fill. `entry_signal_active_share_until_fill` is the fraction of that same
posting-to-fill interval for which it was true. All three use only then-available data. They
can motivate one explicit next research question about an entry-order lifecycle. They are not Evidence: do not infer
participant intent, claim cancellation would help, or change this hypothesis's execution
or exit from them. A missing/null value supports no statement.
"""
    validation_feedback = digest.get("development_validation_feedback")
    validation_feedback_pass = "" if validation_feedback is None else f"""
PASS 0.55 — Development validation feedback. `development_validation_feedback` is the
earlier algorithm's fixed-exit Validation Backtest result with total net bps at or below
zero. It tells you that the parent entry mechanism did not transfer to these already-used
development dates. Its current parameter lock, entry guards, aggregate and sampled cases
may only help choose what part of the *next entry question* must materially change.

It is not Feature Profile Evidence, does not support any factual claim or prediction,
and must never appear in `evidence_basis`, `evidence_relations`, source of profit, or a
mechanism test. Do not call it independent validation or profitability. The listed dates
are now development feedback for this successor; Final Backtest has not been opened and
is not available here. Preserve the fixed canonical exit. Every retained hypothesis must
still be justified only by the supplied Feature Profile Evidence and relation map.

{json.dumps(validation_feedback, ensure_ascii=False, indent=1)}
"""
    library_pass = "" if not digest.get("historical_candidate_library") else """
PASS 0.3 — Historical candidate boundary. `historical_candidate_library` is a compact
fingerprint subset of earlier Sandbox proposals. It is not evidence and may be wrong, but
do not make a prior financial mechanism look new by changing its prose, feature threshold,
or cluster name. A later novelty audit compares the retained candidate with the complete
history. Use this fingerprint to seek a materially different mechanism, source of
executable movement, and falsifiable prediction. If the current Evidence supports only a
similar observable, retain it as a low-confidence candidate for the novelty audit.

Some records have `candidate_kind=DIRECT_EVIDENCE_CANDIDATE`. They are prior observed
entry logics, not financial mechanism claims. `reserved_direct_entry_logic_fingerprints`
lists **all** prior Direct Evidence entry contracts, including records omitted from the
compact prose subset. Do not emit a new hypothesis whose executable core is one of those
fingerprints as if it were new: different prose, thresholds, financial stories, or
falsification wording do not make a new algorithm. Generate a different testable
interpretation when supported; otherwise retain the observable candidate for the
explicit novelty audit.
"""
    lens = digest.get("research_lens")
    lens_pass = "" if lens is None else f"""
PASS 0.2 — Targeted Evidence relation. This invocation is restricted to the supplied
`research_lens`. It is not a new observation and it does not prove a mechanism.
Emit **exactly one** hypothesis. It must use exactly the listed
`families` in its `evidence_basis` and use the exact `allowed_hypothesis_structure`.
When `required_evidence_ids` is present, cite exactly those ids and no other Evidence
in `evidence_basis`; this pins a stored temporal expression rather than letting you
substitute another lag of the same feature.
For a two-family lens, include its exact `relation` in `evidence_relations`. For a
`SINGLE` lens, do not invent a relationship. Do not borrow another family to make the
story sound stronger, do not change a relation into causality, and do not escape to
another relation if this lens is weak. Emit a low-confidence reduced-form hypothesis
when the mechanism is unresolved; it may not turn the relation into causality.

Research lens:
{json.dumps(lens, ensure_ascii=False, indent=1)}
"""
    return f"""{ROLE}
# Task

Work through these passes in order, then return only the final JSON.

{path_pass}
{mechanism_probe_pass}
{alignment_pass}
{execution_replay_pass}
{execution_evidence_pass}
{execution_state_pass}
{raw_path_pass}
{temporal_change_pass}
{lens_pass}
{library_pass}
{loss_pass}
{validation_feedback_pass}
PASS 0.26 — Missingness scope. An omitted pre-anchor change item failed the input audit
for its own value/coverage only. It is not Evidence and must not be cited. Its omission
does not invalidate a different PRIMARY level Evidence item. Work from the remaining
PRIMARY Evidence and record missingness as a limitation.
PASS 1 — Evidence synthesis. Before any interpretation, sort the evidence into
  PERSISTENT_CONTEXT (differs long before the anchor), SETUP, TRIGGER (separates
  sharply near the anchor), SUPPORTING_STATE, WEAK_OR_AMBIGUOUS. Use no economic
  vocabulary yet — no "absorption", "exhaustion", "institutional", "vacuum".

PASS 1.5 — Evidence relation mapping. For a hypothesis using two or more distinct
evidence families, write `evidence_relation_map` before making a mechanism. The supplied
relation map is the maximum supported relation: copy it or downgrade it to UNRESOLVED,
never upgrade it. For a SINGLE_MECHANISM, an empty relation map is correct. A path may
illustrate a relation but may not turn coincidence into coupling or causal order.
PERSISTENT_CONTEXT means only "earlier within the observed Profile window"; it is not a
multi-day or multi-week state. In `basis_evidence`, cite the supplied
`joint_evidence_id` for a family relation, or the underlying `EV_*` evidence_id for an
individual observation. Both are valid identifiers.

PASS 2 — Profit-hypothesis construction. Start with the pre-anchor tool observations,
  then ask what observable state could leave positive fee-inclusive net PnL after the
  fixed BID1 queue fill, spread already in the fill, and explicit 23bp cost. When the
  input carries `executable_sample_condition`, that Feature Profile condition
  is already fixed and must be copied unchanged. The later executable entry is that
  condition AND the additional observable condition derived from Profile Evidence.
  An economic mechanism may explain the state, but it is optional. If it is unresolved,
  write that plainly and retain the narrower observable profit hypothesis.
  First pick a `hypothesis_structure` the relations allow:
  COUPLED_MECHANISM only where a COUPLED relation exists; CONTEXT_TRIGGER where a
  context is observed earlier than a trigger; CONTEXT_INDEPENDENT where both contribute
  independently; SINGLE_MECHANISM when one evidence suffices — do not pad with
  correlated or redundant evidence. Then prefer the shape:
  market context -> pressure/shock -> market response -> state transition -> repricing.
  Each non-SINGLE_MECHANISM hypothesis must explain every necessary evidence family.
  A SINGLE_MECHANISM must stay within exactly one directly observed, non-redundant
  family and make a distinct falsifiable execution prediction. Never add a family for decoration.

PASS 2b — Data awareness. For each hypothesis answer these before writing it:
  1. What is directly observed?
  2. At what exact point could the BID1 order fill, and what post-fill price path could
     still leave net profit after the stated cost?
  3. Is the economic interpretation observable, proxy-only, or unresolved?
  4. What distinct execution result would refute the profit hypothesis?
  5. Is that result measurable with existing or VALIDATION_ONLY data?
  6. Would the observable profit hypothesis still be meaningful if the mechanism stays unidentified?
  Record the outcome in `data_fit` and `capability_requirements`. Every capability you
  name must exist in the profile and carry the availability the profile gives it.
  `overall` is FULLY_TESTABLE only when the mechanism itself is identifiable here;
  REDUCED_FORM_TESTABLE when the mechanism is unresolved but the prediction is testable;
  DATA_BLOCKED when neither. REDUCED_FORM_TESTABLE is a normal hypothesis, not a defect.

An unresolved financial mechanism is normal at this stage. A single observable condition
must remain `REDUCED_FORM_TESTABLE`. Independent Profile cards without a tool-observed
path are not one coupled condition: keep their predictions separate or choose the
strongest non-redundant observable for the candidate.

PASS 3 — Temporal story. Use only the temporal evidence given. Write the sequence by
  role: persistent context first, later trigger after, and mark each stage's
  `relation_status`. Do not invent timing the evidence does not contain, and do not
  convert an observed ordering into a causal one. A `JEV_*` relation item belongs only
  in `evidence_relation_map`; never put it in an `expected_observable_sequence.evidence`
  list. Sequence stages cite their individual `EV_*` observations only.

PASS 4 — Execution edge and implementation directions. Fill `source_of_profit` as a
testable execution thesis: (1) returned pre-anchor state, (2) fixed BID1 queue/fill,
(3) post-fill BID1/ASK1 path, and (4) explicit 23bp cost. It is a hypothesis, not a
profit claim. An unobserved participant story is optional and must stay unresolved.
"Feature X predicts returns" alone is not enough: name the expected net path and the
failure observation.

Emit one to three `implementation_directions` for the SAME profit hypothesis. They are
alternative observation representations, not separate mechanisms or parameter searches.
Each direction names only evidence already in `evidence_basis`, a representation
(`LEVEL`, `PRE_ANCHOR_CHANGE`, `PERSISTENCE`, `SEQUENCE`, or `COMPOSITE`), and the
execution edge it is meant to test. Do not choose a lag, window, threshold, entry, exit,
or guard. Use `PRE_ANCHOR_CHANGE` only when the cited Evidence is an exact stored
pre-anchor change. Directions must differ in the observable representation, not merely
in prose. Grounding will either turn each direction into a Catalog expression or record
why that representation is unsupported.
Set `already_priced_risk` true when the observed state might already be reflected in the
quote; that risk requires a later comparison but does not block code generation.

PASS 4.5 — Distinguishing prediction. Check that the hypothesis makes a prediction
that differs from a simpler baseline explanation such as unconditional short-horizon
momentum or a general market-state effect. This is not a claim of academic novelty;
state the concrete matched comparison that would distinguish the two.

PASS 5 — Falsification. At least one MATCHED_CONTROL (or equivalent) test and at
  least one alternative explanation with a discriminating test are required for every
  hypothesis. A CONTEXT_TRIGGER structure additionally requires a TEMPORAL_PRECEDENCE
  test; prefer a context-present/trigger-absent control and a trigger-present/context-
  absent control. For CONTEXT_INDEPENDENT with ORDER_UNCLEAR, do not invent a temporal
  order merely to fill this slot: use INCREMENTAL_EFFECT or PATH_TEST to test each
  independent contribution instead. At least one alternative must consider that the
  context is merely an incidental regime, or that both are consequences of an
  unobserved third condition.

PASS 5.5 — MechanismGraph. For every retained hypothesis emit the exact graph in the
output schema. Every observable must cite one item already in `evidence_basis`. Use
`available_at="PENDING"` only if the same observable is readable while the BID1 order waits. For
`CANCEL_WHEN_ENTRY_SIGNAL_FALSE`, every graph observable must be PENDING because the
current runtime recomputes the complete entry signal. The graph always
uses `BID1_QUEUE`; do not invent order, exit, hold, or sizing rules. A cancellation
mode must name the PENDING observable(s) that are checked, and HOLD_THROUGH must leave
that list empty. `prediction` is evaluated only after FILLED and cannot be an input to
entry or cancellation.

PASS 6 — Emit {1 if lens is not None else '1 to ' + str(config.max_hypotheses)} hypotheses.
Do not pad to {config.max_hypotheses}. Multiple hypotheses must differ in observed
Evidence relation, execution edge, or financial interpretation, not in parameter
strength. Weak, contradictory, unstable, or mechanism-unresolved Evidence still produces
the narrowest measurable reduced-form candidate. Record every weakness in
`evidence_warnings`, set confidence accordingly, and leave profitability to Search.

# Output schema

{_SCHEMA}

# Research capability profile

What this environment can and cannot observe. Values of VALIDATION_ONLY data are not
shown to you and must not be treated as observed.

{json.dumps(profile, ensure_ascii=False, indent=1)}

# Evidence

{json.dumps(digest, ensure_ascii=False, indent=1)}
"""


def audit_prompt(digest: Mapping[str, Any], drafts: Mapping[str, Any],
                 profile: Mapping[str, Any] | None = None) -> str:
    """Invocation B — 초안을 증거에 맞게 좁히되 최소 후보 하나를 보존한다."""
    profile = C.research_capability_profile() if profile is None else profile
    return f"""{ROLE}
# Task

You wrote the draft below. Now audit it against the evidence. You are the same role,
not a judge: your job is to keep at least one exploratory candidate while narrowing
every claim to what the Evidence carries. Do not add a richer mechanism. If the draft
is empty, create one minimal `REDUCED_FORM_TESTABLE` candidate from the strongest
non-redundant Evidence item and mark its limitations and `LOW` confidence.

Check every hypothesis for:
- unsupported claims — a claim with no evidence carrying it
- feature hallucination — a feature name not present in the evidence
- redundant evidence counting — features inside one evidence_id treated as separate support
- temporal invention — timing the evidence does not contain
- missing or weak execution edge — does it address post-fill net outcome and cost?
- weak falsifiability — is there a test that could actually fail?
- implementation leakage — thresholds, quantiles, entry/exit rules, sizing, PnL
- context confused with cause — did the draft turn an observed ordering into causality?
- independent late triggers merged into one coupled state
- redundant evidence counted twice as separate support
- an UNRESOLVED or UNRELATED relation stated as an observed fact
- a capability named with an availability the profile does not give it
- an UNAVAILABLE or PROXY_ONLY concept described as a direct observation
- VALIDATION_ONLY data described as something already observed
- an unavailable financial concept quietly swapped for an available feature
- a generality claim the 6-week single-period sample cannot carry
- a `path_observations` entry without a path_id from the draft, or a future oracle
  price path described as a fill, validation result, OOS result, or execution proof
- a `path_observations` entry that is an exact, tool-returned BID1/ASK1 or pre-anchor
  raw microstructure observation from the draft: retain it verbatim when it is valid. It records what the generation
  Agent actually inspected; audit the hypothesis claims around it, not the observation
  itself.
- a `loss_ledger_observations` entry that is an exact, tool-returned Discovery diagnostic
  from the draft: retain it verbatim when it is valid. It records the prior failure and
  the next research question, never Evidence or proof for the new mechanism.
- an `evidence_relation_map` relation upgraded beyond the supplied evidence relation
- an `evidence_roles` entry that differs from the exact cited `EV_*` role in a
  SINGLE_MECHANISM, or from the supplied family role in a multi-family relation; a joint
  relation's context/trigger label is not permission to change that role
- a Discovery execution-loss diagnostic cited as new Evidence, a prediction, Validation,
  Final result, or direct support for the new mechanism

For each draft hypothesis decide RETAIN, REVISE or DROP. REVISE means narrow the claim
to what the evidence carries — never broaden it. Do not drop the last candidate: revise
it to the smallest measurable profit prediction. Return `HYPOTHESES_GENERATED` with at
least one hypothesis.

# Output schema

Same schema as the draft, plus a top-level "verdicts" array:

"verdicts": [{{"hypothesis_id": str, "verdict": "RETAIN"|"REVISE"|"DROP",
               "reasons": [str]}}]

{_SCHEMA}

# Research capability profile

{json.dumps(profile, ensure_ascii=False, indent=1)}

# Evidence

{json.dumps(digest, ensure_ascii=False, indent=1)}

# Draft to audit

{json.dumps(drafts, ensure_ascii=False, indent=1)}
"""


def contract_repair_prompt(digest: Mapping[str, Any], payload: Mapping[str, Any],
                           problems: Sequence[str],
                           profile: Mapping[str, Any] | None = None) -> str:
    """잘못된 Agent JSON을 같은 정본 Schema 안에서만 수리한다."""
    profile = C.research_capability_profile() if profile is None else profile
    return f"""{ROLE}
# Task

Repair the payload so it satisfies every listed problem and the exact canonical schema below.
Return `HYPOTHESES_GENERATED` with at least one evidence-bounded exploratory hypothesis.
Use only enum values shown in the schema. Do not rename fields or invent a new structure.
Do not add a feature, threshold, relation, path observation, or result not present in Evidence.
When `price_path_tool` is absent from Evidence, `path_observations` must be an empty array.
Weak Evidence remains a LOW-confidence `REDUCED_FORM_TESTABLE` candidate; Search later decides
profitability. Preserve valid draft meaning and make the smallest structural repair.

# Problems

{json.dumps(list(problems), ensure_ascii=False, indent=1)}

# Canonical output schema

{_SCHEMA}

# Research capability profile

{json.dumps(profile, ensure_ascii=False, indent=1)}

# Evidence

{json.dumps(digest, ensure_ascii=False, indent=1)}

# Payload to repair

{json.dumps(payload, ensure_ascii=False, indent=1)}
"""


def apply_fixed_contracts(payload: Mapping[str, Any], package: Mapping[str, Any], *,
                          path_observations_available: bool = True) -> dict[str, Any]:
    """Agent 판단이 아닌 고정 계약값을 정본 입력에서 복사한다."""
    result = copy.deepcopy(dict(payload))
    if not path_observations_available:
        result["path_observations"] = []
    sample_condition = package.get("executable_sample_condition")
    for hypothesis in result.get("hypotheses") or []:
        if not isinstance(hypothesis, dict):
            continue
        hypothesis["profit_target"] = PT.canonical_target()
        hypothesis["executable_sample_condition"] = copy.deepcopy(sample_condition)
        graph = hypothesis.get("mechanism_graph")
        if not isinstance(graph, dict):
            continue
        execution = dict(graph.get("execution") or {})
        execution.update({"entry_state": "DECISION", "pending_state": "PENDING",
                          "entry_action": "BID1_QUEUE"})
        graph["execution"] = execution
    return result


def novelty_prompt(prior_hypotheses: Sequence[Mapping[str, Any]], candidate_payload: Mapping[str, Any]) -> str:
    """기존 문장을 바꿔 쓴 후보를 다음 단계 전에 막는 별도 Agent 역할."""
    return f"""You compare a prior financial hypothesis with proposed successors.

Return one JSON object only:
{{"status":"RETAIN"|"DUPLICATE_HYPOTHESIS", "decisions":[{{"hypothesis_id":str,
"status":"RETAIN"|"DUPLICATE_HYPOTHESIS", "reason":str}}]}}

Mark DUPLICATE_HYPOTHESIS when a proposal merely rephrases the same financial mechanism,
source of profit, and falsification logic. Some historical records are
DIRECT_EVIDENCE_CANDIDATE: they reserve an existing `entry_logic_fingerprint` but make no
mechanism claim. Do not mark a proposal duplicate solely because it uses the same direct
logic; do mark it duplicate if it only supplies new prose or a threshold change for that
logic without a materially different source of profit and distinguishing test. Shared
features alone do not make two ideas duplicates; retain one only when its financial cause,
why executable movement remains, and a test that could distinguish it are materially
different. Do not create, revise, or score any hypothesis. The earlier execution diagnostic
is not Evidence for either side.

# Historical hypotheses
{json.dumps(list(prior_hypotheses), ensure_ascii=False, indent=1)}

# Proposed hypotheses
{json.dumps(candidate_payload, ensure_ascii=False, indent=1)}
"""


def validate_novelty(value: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Novelty Agent의 작은 출력 계약. 후보 전체가 중복이면 다음 Stage를 열지 않는다."""
    allowed = {str(item.get("hypothesis_id")) for item in payload.get("hypotheses") or []}
    decisions = list(value.get("decisions") or [])
    by_id = {str(item.get("hypothesis_id")): item for item in decisions if isinstance(item, Mapping)}
    missing = sorted(allowed - set(by_id))
    unknown = sorted(set(by_id) - allowed)
    if missing or unknown:
        raise ValueError(f"Novelty audit 가설 범위가 다르다: missing={missing}, unknown={unknown}")
    statuses = {str(item.get("status")) for item in by_id.values()}
    if not statuses <= {"RETAIN", DUPLICATE_HYPOTHESIS}:
        raise ValueError("Novelty audit status가 다르다")
    state = DUPLICATE_HYPOTHESIS if statuses == {DUPLICATE_HYPOTHESIS} else "RETAIN"
    return {"status": state, "decisions": [dict(by_id[identifier]) for identifier in sorted(allowed)]}


# ---- 호출 --------------------------------------------------------------------







# ---- 결정적 출력 검증 (§59) ----------------------------------------------------

# 전략을 가설로 위장한 출력. 이것은 **실패**다 — 임계·규칙·수량은 다음 단계의 몫이다.
FORBIDDEN = (
    (re.compile(r"\bq\d{1,2}\b", re.I), "분위수 컷"),
    (re.compile(r"[<>]=?\s*[-+]?\d"), "임계 비교"),
    (re.compile(r"(?:buy|sell|enter|exit|매수|매도|진입|청산)\s*(?:when|if|at|하면|할 때|한다)", re.I),
     "진입·청산 규칙"),
    (re.compile(r"\b(stop[- ]?loss|take[- ]?profit|position siz|손절|익절)", re.I), "실행 규칙"),
    (re.compile(r"\b\d{1,3}(?:th|st|nd|rd)\s+percentile\b|\bpercentile\s*\d", re.I), "분위수"),
)

# 하류 성과를 나타내는 낱말. 이것만으로는 실패로 보지 않는다 — "OOS 결과가 없다" 처럼
# **부재를 적는 것**과 "OOS 성적이 좋았다" 는 정반대인데 정규식은 둘을 못 가른다.
# 세어서 경고로 남기고 사람이 본다.
# 관측된 순서를 인과로 바꾸는 표현. CONTEXT_TRIGGER 는 순서까지만 허용한다 (§11·§26).
CAUSAL_WORDS = re.compile(
    r"(causes?\b|induces?\b|caused by|consequence of|operates? through|"
    r"때문에 |으로 인해|를 일으키|가 일으키|를 유발|가 유발|의 결과이다|때문이다)", re.I)
# "A 가 B 를 유발했다고 주장하지 않는다" 는 인과 주장이 아니라 그 반대다. 부정을 못 읽으면
# 스스로 선을 긋는 정확한 서술이 위반으로 잡힌다.
DISCLAIMER = re.compile(
    r"(주장(?:은|을)? ?하지 않|말하지 않|하지 않는다|아니(?:다|며|고|면서)|없다|do(?:es)? not (?:claim|assert|imply)|"
    r"is not (?:claimed|asserted|implied)|no claim)")

# 관측값을 인과로 단정하지 않으면서, 이후 grounding으로 시험할 **가능한 해석**을 적는
# 문장이다. 이 표지는 인과 주장의 면제가 아니라 가설적 설명이라는 경계다.
HYPOTHETICAL_INTERPRETATION = re.compile(
    r"(가능한 해석|금융적 해석|가능성(?:은|이)?|일 수 있다|일 수 있으며|may be|possible interpretation)", re.I)


def _asserted(text: str, match: re.Match[str]) -> bool:
    """그 표현이 실제 주장인가, 아니면 하지 않겠다는 선언인가.

    표현이 들어 있는 **문장 하나**만 본다. 문단 전체를 보면 뒤 문장의 부정이 앞 문장의
    주장을 덮어 버린다.
    """
    start = max(text.rfind(".", 0, match.start()), text.rfind("다 ", 0, match.start())) + 1
    end = text.find(".", match.end())
    sentence = text[start: end if end != -1 else len(text)]
    return not (DISCLAIMER.search(sentence) or HYPOTHETICAL_INTERPRETATION.search(sentence))


# 두 증거를 하나의 결합 상태로 묶는 표현. COUPLED 관계가 있을 때만 허용한다 (§8·§12).
COUPLING_WORDS = re.compile(
    r"(coupled state|joint (?:latent )?state|one (?:single )?latent state|"
    r"결합 상태|하나의 상태|단일 잠재 상태|하나의 잠재 상태|공동 상태)", re.I)

RESULT_WORDS = re.compile(
    r"\b(sharpe|pnl|p&l|backtest|out[- ]of[- ]sample|oos|백테스트|수익률 곡선)", re.I)

# feature 이름처럼 생긴 토큰. 카탈로그에 없으면 지어낸 것으로 본다.
_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_ALLOWED_TOKENS = frozenset({
    "evidence_id", "profit_vs_loss", "profit_vs_background", "loss_vs_background",
    "source_of_profit", "already_priced_risk", "mechanism_tests", "effect_strength",
    "date_stability", "symbol_stability", "grounding_requirements", "time_basis",
    "relative_times_ms", "separation_profit_vs_loss", "observable_description",
    "expected_observable_sequence", "evidence_basis", "mechanistic_claim",
    "alternative_explanations", "evidence_warnings", "hypothesis_id",
    "implementation_directions", "direction_id", "evidence_ids", "observation_focus",
    "expected_execution_edge", "pre_anchor_change",
    "matched_control", "temporal_precedence", "dose_response", "incremental_effect",
    "path_test", "to_be_tested", "cohort_summary",
})


def _combinations(items: Sequence[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, Mapping):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, (list, tuple)):
        return [s for v in node for s in _strings(v)]
    return []


# capability 이름은 연구 환경의 어휘다. 가설이 지어낸 feature 가 아니다.
# 이것을 세면 "participant_identity 는 관측할 수 없다" 라는 정확한 서술이 위반으로 잡힌다.
def _environment_vocabulary() -> set[str]:
    return set(C.CAPABILITIES) | {status.lower() for status in C.STATUSES}


def _package_vocabulary(package: Mapping[str, Any]) -> set[str]:
    """package 가 스스로 쓰는 낱말. 이것을 지어낸 feature 로 세면 안 된다 —
    Agent 가 "near_miss 는 쓰지 않았다" 고 적는 것은 정확한 서술이다."""
    words = set(_ALLOWED_TOKENS) | _environment_vocabulary()
    for text in _strings({k: v for k, v in package.items()
                          if k in ("not_provided", "outcome_definition", "cohort_summary",
                                   "global_warnings", "joint_evidence_note")}):
        words |= set(_TOKEN.findall(text))
    words |= {str(k).lower() for k in (package.get("cohort_summary") or {})}
    words |= {"near_miss", "mechanism_failure", "context_failure", "legacy_time_baseline"}
    return words


# `data_fit.overall` 은 앞의 두 축에서 정해진다. Agent 가 마음대로 올릴 수 없다 (§20).
def _derive_fit(mechanism: str, prediction: str) -> str:
    if prediction == "UNAVAILABLE":
        return "DATA_BLOCKED"
    if mechanism == "IDENTIFIABLE":
        return "FULLY_TESTABLE"
    return "REDUCED_FORM_TESTABLE"


def data_fit_checks(hypotheses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """가설이 데이터 환경을 정확히 말했는가. 여기가 정본이다 (§51).

    세 가지를 본다.
      - capability 이름과 등급이 프로필과 맞는가
      - 없는 것(UNAVAILABLE)·대용치(PROXY_ONLY)를 직접 관측이라 했는가
      - 아직 열지 않은 미래 데이터를 이미 봤다고 했는가
    """
    capability_problems: list[dict[str, Any]] = []
    as_direct: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    fit_problems: list[dict[str, Any]] = []

    for item in hypotheses:
        tag = str(item.get("hypothesis_id"))
        requirements = item.get("capability_requirements") or []
        for message in C.validate_requirements(requirements):
            capability_problems.append({"hypothesis_id": tag, "problem": message})

        # 개념 이름으로 capability 요구와 observability 선언을 맞춰 본다.
        # LLM 출력이라 모양이 어긋날 수 있다. 여기서 죽으면 검사 자체를 못 한다.
        levels = {str(o.get("concept")): str(o.get("level"))
                  for o in item.get("observability") or [] if isinstance(o, Mapping)}
        levels.update({str(g.get("concept")): str(g.get("observability"))
                       for g in item.get("grounding_requirements") or []
                       if isinstance(g, Mapping)})
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                capability_problems.append({"hypothesis_id": tag,
                                            "problem": f"capability 요구가 객체가 아니다: "
                                                       f"{requirement!r}"})
                continue
            concept = str(requirement.get("concept"))
            actual = C.lookup(str(requirement.get("capability")))
            level = levels.get(concept)
            if level != "DIRECT":
                continue
            if actual in (C.UNAVAILABLE, C.PROXY_ONLY):
                as_direct.append({"hypothesis_id": tag, "concept": concept,
                                  "capability_status": actual, "claimed_level": level})
            elif actual == C.VALIDATION_ONLY:
                leakage.append({"hypothesis_id": tag, "concept": concept,
                                "why": "아직 열지 않은 미래 데이터를 직접 관측이라 했다"})

        # 미래 데이터에 기대는 단계를 이미 관측했다고 적었는가
        future = {str(r.get("concept")) for r in requirements
                  if C.lookup(str(r.get("capability"))) == C.VALIDATION_ONLY}
        for stage in item.get("expected_observable_sequence") or []:
            if stage.get("status") != "OBSERVED":
                continue
            claim = str(stage.get("claim", ""))
            hit = [c for c in future if c and c in claim]
            if hit:
                leakage.append({"hypothesis_id": tag, "stage": stage.get("stage"),
                                "concept": hit[0],
                                "why": "VALIDATION_ONLY 인 것을 OBSERVED 로 적었다"})

        fit = item.get("data_fit") or {}
        if not isinstance(fit, Mapping):
            fit_problems.append({"hypothesis_id": tag, "problem": "data_fit 이 객체가 아니다"})
            continue
        if not fit:
            fit_problems.append({"hypothesis_id": tag, "problem": "data_fit 이 없다"})
            continue
        axes = {"observable_core": CORE_OBSERVABILITY,
                "mechanism_observability": MECHANISM_OBSERVABILITY,
                "prediction_testability": PREDICTION_TESTABILITY,
                "overall": DATA_FIT}
        for key, allowed in axes.items():
            if fit.get(key) not in allowed:
                fit_problems.append({"hypothesis_id": tag,
                                     "problem": f"data_fit.{key} 가 {allowed} 밖이다: "
                                                f"{fit.get(key)!r}"})
        derived = _derive_fit(str(fit.get("mechanism_observability")),
                              str(fit.get("prediction_testability")))
        if fit.get("overall") in DATA_FIT and fit.get("overall") != derived:
            fit_problems.append({"hypothesis_id": tag,
                                 "problem": f"data_fit.overall 이 {fit.get('overall')} 인데 "
                                            f"두 축에서 따라 나오는 값은 {derived} 다"})
        if item.get("validation_feasibility") not in VALIDATION_FEASIBILITY:
            fit_problems.append({"hypothesis_id": tag,
                                 "problem": f"validation_feasibility 가 "
                                            f"{VALIDATION_FEASIBILITY} 밖이다: "
                                            f"{item.get('validation_feasibility')!r}"})

    return {"capability_violation_count": len(capability_problems),
            "capability_violations": capability_problems,
            "unobservable_as_direct_count": len(as_direct),
            "unobservable_as_direct": as_direct,
            "validation_only_leakage_count": len(leakage),
            "validation_only_leakage": leakage,
            "data_fit_problem_count": len(fit_problems),
            "data_fit_problems": fit_problems,
            "data_fit_distribution": _counts(
                [str((h.get("data_fit") or {}).get("overall")) for h in hypotheses])}


def _counts(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def uses_execution_state_evidence(payload: Mapping[str, Any],
                                  package: Mapping[str, Any]) -> bool:
    """최종 가설이 execution-state Evidence를 실제로 인용했는지 판별한다."""
    execution_state_ids = {
        str(item.get("evidence_id"))
        for item in execution_state_evidence_items(package)
    }
    return any(
        str(basis.get("evidence_id")) in execution_state_ids
        for hypothesis in payload.get("hypotheses") or [] if isinstance(hypothesis, Mapping)
        for basis in (hypothesis.get("evidence_basis") or []) if isinstance(basis, Mapping))


def raw_path_labels_used(payload: Mapping[str, Any], package: Mapping[str, Any]) -> set[str]:
    """최종 가설이 실제 인용한 raw path Evidence의 두 경로 label."""
    cited = {
        str(basis.get("evidence_id"))
        for hypothesis in payload.get("hypotheses") or [] if isinstance(hypothesis, Mapping)
        for basis in (hypothesis.get("evidence_basis") or []) if isinstance(basis, Mapping)
    }
    labels: set[str] = set()
    for item in raw_path_evidence_items(package):
        if str(item.get("evidence_id")) in cited:
            labels.add(str(item.get("raw_path_positive_label")))
            labels.add(str(item.get("raw_path_negative_label")))
    return {label for label in labels if label}


def uses_raw_path_evidence(payload: Mapping[str, Any], package: Mapping[str, Any]) -> bool:
    return bool(raw_path_labels_used(payload, package))


def validate_output(payload: Mapping[str, Any], package: Mapping[str, Any],
                    config: AgentConfig = AgentConfig(), *,
                    path_observations_available: bool | None = None) -> dict[str, Any]:
    """스키마·증거 참조·feature 이름·금지 표현을 코드로 검사한다.

    LLM 이 지키겠다고 말한 것과 실제로 지킨 것은 다르다. 여기가 정본이다.
    """
    problems: list[str] = []
    evidence_items = _agent_evidence_items(package)
    known = {item.get("evidence_id") for item in evidence_items}
    known_features = {item.get("representative_feature", {}).get("name")
                      for item in evidence_items}
    known_features |= {f for item in evidence_items
                       for f in item.get("related_features", [])}
    vocabulary = _package_vocabulary(package)

    status = payload.get("status")
    if status not in STATUSES:
        problems.append(f"status 가 {STATUSES} 중 하나가 아니다: {status!r}")
    hypotheses = payload.get("hypotheses") or []
    if status == GENERATED and not hypotheses:
        problems.append("HYPOTHESES_GENERATED 인데 가설이 없다")
    if len(hypotheses) > config.max_hypotheses:
        problems.append(f"가설이 {len(hypotheses)}개다. 최대 {config.max_hypotheses}개")

    # 새 Agent 경로에서는 가격 path를 실제로 본 뒤 가설을 쓴다. path는 discovery oracle
    # 이므로 가격 기회 관측을 체결/검증으로 바꾸지는 못하게 참조만 고정한다.
    path_source = package.get("path_observation") or {}
    allowed_paths = {
        f"{item.get('anchor_id')}:{path_source.get('horizon_key')}"
        for item in path_source.get("anchors") or []
    }
    path_observations = payload.get("path_observations") or []
    raw_path_labels = raw_path_labels_used(payload, package)
    uses_raw_path = bool(raw_path_labels)
    uses_execution_state = uses_execution_state_evidence(payload, package)
    observed_path_ids: set[str] = set()
    path_observation_problems: list[str] = []
    for item in path_observations:
        if not isinstance(item, Mapping):
            path_observation_problems.append("path_observations 항목이 객체가 아니다")
            continue
        path = str(item.get("path_id"))
        if path not in allowed_paths:
            path_observation_problems.append(f"허용되지 않은 path_id '{path}'")
        else:
            observed_path_ids.add(path)
        if item.get("cohort") not in {"PROFIT", "LOSS", "BACKGROUND"}:
            path_observation_problems.append(f"path {path}: 모르는 cohort")
        if uses_raw_path:
            diagnostic = str(item.get("diagnostic_cohort") or "")
            if item.get("cohort") != "BACKGROUND":
                path_observation_problems.append(
                    f"path {path}: raw-path Evidence는 BACKGROUND anchor 경로만 쓴다")
            if diagnostic not in raw_path_labels:
                path_observation_problems.append(
                    f"path {path}: raw-path diagnostic_cohort가 없다 또는 인용 Evidence와 다르다")
        elif uses_execution_state:
            label = str(item.get("execution_label") or "")
            if item.get("cohort") != "BACKGROUND":
                path_observation_problems.append(
                    f"path {path}: execution-state Evidence는 BACKGROUND anchor 경로만 쓴다")
            if label not in {"EXECUTED_PROFIT", "EXECUTED_LOSS"}:
                path_observation_problems.append(
                    f"path {path}: execution-state 경로의 execution_label이 없다")
        if not item.get("observations"):
            path_observation_problems.append(f"path {path}: 관측 사실이 없다")
    path_observations_required = (package.get("price_path_context_required", True)
                                  if path_observations_available is None
                                  else path_observations_available)
    if allowed_paths and path_observations_required and status == GENERATED:
        if uses_raw_path:
            seen_raw_labels = {str(item.get("diagnostic_cohort") or "")
                               for item in path_observations if isinstance(item, Mapping)}
            for label in sorted(raw_path_labels - seen_raw_labels):
                path_observation_problems.append(f"{label} raw-path 가격 경로를 관측하지 않았다")
        elif uses_execution_state:
            seen_execution_labels = {str(item.get("execution_label"))
                                     for item in path_observations if isinstance(item, Mapping)}
            for label in ("EXECUTED_PROFIT", "EXECUTED_LOSS"):
                if label not in seen_execution_labels:
                    path_observation_problems.append(f"{label} 가격 경로를 관측하지 않았다")
        else:
            seen_cohorts = {str(item.get("cohort")) for item in path_observations
                            if isinstance(item, Mapping)}
            for cohort in ("PROFIT", "LOSS"):
                if cohort not in seen_cohorts:
                    path_observation_problems.append(f"{cohort} 가격 경로를 관측하지 않았다")
        hypothesis_family_counts = {
            len({str(b.get("family")) for b in (hypothesis.get("evidence_basis") or [])})
            for hypothesis in hypotheses if isinstance(hypothesis, Mapping)
        }
        if any(count >= 2 for count in hypothesis_family_counts) and not payload.get("evidence_relation_map"):
            path_observation_problems.append("가격 경로를 본 뒤 Evidence 관계 지도를 쓰지 않았다")

    relation_source = {
        tuple(sorted((str(item.get("family_a")), str(item.get("family_b"))))):
        str(item.get("relation"))
        for item in package.get("joint_evidence", []) or []
    }
    # 관계 지도는 family 단위 joint evidence에서 출발한다. 따라서 개별 evidence_id와
    # joint_evidence_id 모두 실제 근거다. 전자는 feature-level 관측, 후자는 관계-level
    # 관측을 뜻하므로 둘 중 하나만 허용하면 prompt와 validator가 서로 다른 말을 한다.
    known_relation_evidence = known | {
        str(item.get("joint_evidence_id"))
        for item in package.get("joint_evidence", []) or []
    }
    relation_map_problems: list[str] = []
    for item in payload.get("evidence_relation_map") or []:
        if not isinstance(item, Mapping):
            relation_map_problems.append("evidence_relation_map 항목이 객체가 아니다")
            continue
        pair = tuple(sorted((str(item.get("a")), str(item.get("b")))))
        source_relation = relation_source.get(pair)
        relation = str(item.get("relation"))
        if source_relation is None:
            relation_map_problems.append(f"관계 지도의 근거가 없는 family 쌍 {pair}")
        elif relation not in (source_relation, "UNRESOLVED"):
            relation_map_problems.append(
                f"{pair} 관계를 {relation}로 올렸다 (근거는 {source_relation})")
        for path in item.get("basis_paths") or []:
            if str(path) not in observed_path_ids:
                relation_map_problems.append(f"{pair}: 관측하지 않은 path_id '{path}'를 인용했다")
        for evidence_id in item.get("basis_evidence") or []:
            if str(evidence_id) not in known_relation_evidence:
                relation_map_problems.append(f"{pair}: 없는 evidence_id '{evidence_id}'")
    problems.extend(path_observation_problems)
    problems.extend(relation_map_problems)

    required = ("hypothesis_id", "title", "profit_target", "evidence_basis",
                "expected_observable_sequence", "source_of_profit", "mechanism_tests",
                "alternative_explanations", "observability", "grounding_requirements",
                "implementation_directions", "confidence")
    unknown_features: set[str] = set()
    invalid_refs: set[str] = set()
    result_mentions: list[dict[str, str]] = []
    mechanism_graph_problem_count = 0

    for index, item in enumerate(hypotheses):
        tag = f"hypotheses[{index}]"
        for field in required:
            if not item.get(field):
                problems.append(f"{tag}: 필수 항목 '{field}' 이 비었다")
        target_validation = PT.validate(item.get("profit_target"))
        problems.extend(f"{tag}: {problem}" for problem in target_validation["problems"])
        expected_sample_condition = package.get("executable_sample_condition")
        actual_sample_condition = item.get("executable_sample_condition")
        if expected_sample_condition is not None:
            sample_validation = SC.validate(actual_sample_condition)
            problems.extend(f"{tag}: {problem}" for problem in sample_validation["problems"])
            if actual_sample_condition != expected_sample_condition:
                problems.append(
                    f"{tag}: executable_sample_condition이 Feature Profile 조건과 다르다")
        elif actual_sample_condition is not None:
            problems.append(f"{tag}: Feature Profile에 없는 executable_sample_condition을 만들었다")

        basis = item.get("evidence_basis") or []
        families = {str(b.get("family")) for b in basis}
        structure = item.get("hypothesis_structure")
        groups = {str(b.get("evidence_id")) for b in basis}
        if structure == "SINGLE_MECHANISM" and len(families) != 1:
            problems.append(f"{tag}: SINGLE_MECHANISM 인데 evidence family 가 {len(families)}개다")
        elif structure != "SINGLE_MECHANISM" and len(families) < 2:
            problems.append(f"{tag}: evidence family 가 {len(families)}개다. 최소 2개")
        for ref in groups:
            if ref not in known:
                invalid_refs.add(ref)
                problems.append(f"{tag}: 없는 evidence_id '{ref}'")
        for b in basis:
            if b.get("role") not in EVIDENCE_ROLES:
                problems.append(f"{tag}: 모르는 role {b.get('role')!r}")
            name = b.get("representative_feature")
            if name and name not in catalog.FEATURES:
                unknown_features.add(str(name))
                problems.append(f"{tag}: 카탈로그에 없는 feature '{name}'")

        directions = item.get("implementation_directions") or []
        if not isinstance(directions, list) or not (1 <= len(directions) <= 3):
            problems.append(f"{tag}: implementation_directions는 1~3개여야 한다")
        else:
            direction_ids: set[str] = set()
            fingerprints: set[tuple[str, tuple[str, ...]]] = set()
            temporal_ids = {str(evidence.get("evidence_id")) for evidence in evidence_items
                            if evidence.get("evidence_source") == "EXECUTION_TEMPORAL_PREANCHOR"}
            for direction_index, direction in enumerate(directions):
                label = f"{tag}: implementation_directions[{direction_index}]"
                if not isinstance(direction, Mapping):
                    problems.append(f"{label}가 객체가 아니다")
                    continue
                direction_id = str(direction.get("direction_id") or "")
                if not direction_id or direction_id in direction_ids:
                    problems.append(f"{label}.direction_id가 없거나 중복됐다")
                direction_ids.add(direction_id)
                representation = str(direction.get("representation") or "")
                if representation not in IMPLEMENTATION_REPRESENTATIONS:
                    problems.append(f"{label}.representation이 유효하지 않다: {representation!r}")
                evidence_ids = tuple(sorted(str(value) for value in direction.get("evidence_ids") or []))
                if not evidence_ids or any(value not in groups for value in evidence_ids):
                    problems.append(f"{label}.evidence_ids는 이 가설의 evidence_basis만 써야 한다")
                if representation == "PRE_ANCHOR_CHANGE" and not set(evidence_ids) <= temporal_ids:
                    problems.append(f"{label}: PRE_ANCHOR_CHANGE는 저장된 pre-anchor change Evidence만 쓴다")
                fingerprint = (representation, evidence_ids)
                if fingerprint in fingerprints:
                    problems.append(f"{label}: 이전 방향과 같은 관측 표현이다")
                fingerprints.add(fingerprint)
                for field in ("title", "observation_focus", "expected_execution_edge"):
                    if not str(direction.get(field) or "").strip():
                        problems.append(f"{label}.{field}이 비었다")

        graph = item.get("mechanism_graph")
        if graph is None:
            if config.require_mechanism_graph:
                problems.append(f"{tag}: mechanism_graph가 없다")
                mechanism_graph_problem_count += 1
        else:
            graph_report = M.validate(
                graph,
                known_evidence_ids={str(b.get("evidence_id")) for b in basis},
                known_features={str(b.get("representative_feature")) for b in basis},
            )
            if graph_report["problems"]:
                mechanism_graph_problem_count += len(graph_report["problems"])
                problems.extend(f"{tag}: {problem}" for problem in graph_report["problems"])

        for stage in item.get("expected_observable_sequence") or []:
            if stage.get("status") not in SEQUENCE_STATUS:
                problems.append(f"{tag}: 모르는 stage status {stage.get('status')!r}")
            for ref in stage.get("evidence") or []:
                # JEV 는 두 family의 관계를 직접 관측한 근거다. 그러므로 관계 단계에서만
                # 허용한다. 개별 feature 관측처럼 OBSERVED_ORDERING에 섞어 쓰는 것은 막는다.
                is_relation_evidence = (
                    ref in known_relation_evidence
                    and stage.get("relation_status") in ("SUPPORTED_RELATION", "INDEPENDENT")
                )
                if ref not in known and not is_relation_evidence:
                    invalid_refs.add(str(ref))
                    problems.append(f"{tag}: stage 가 없는 evidence_id '{ref}' 를 가리킨다")
        if not any(s.get("status") == "OBSERVED"
                   for s in item.get("expected_observable_sequence") or []):
            problems.append(f"{tag}: OBSERVED 단계가 하나도 없다")

        types = {t.get("test_type") for t in item.get("mechanism_tests") or []}
        if not types & {"MATCHED_CONTROL", "INCREMENTAL_EFFECT", "PATH_TEST"}:
            problems.append(f"{tag}: 대조 성격의 반증 시험이 없다")
        relations = {str(relation.get("relation"))
                     for relation in item.get("evidence_relations") or []}
        if (len(families) >= 2 or structure == "CONTEXT_TRIGGER" or "CONTEXT_TRIGGER" in relations) \
                and "TEMPORAL_PRECEDENCE" not in types:
            problems.append(f"{tag}: TEMPORAL_PRECEDENCE 시험이 없다")
        for t in item.get("mechanism_tests") or []:
            if t.get("test_type") not in TEST_TYPES:
                problems.append(f"{tag}: 모르는 test_type {t.get('test_type')!r}")
        if not item.get("alternative_explanations"):
            problems.append(f"{tag}: 대안 설명이 없다")
        for need in item.get("grounding_requirements") or []:
            if not isinstance(need, Mapping):
                continue
            # `role` 은 증거에서의 역할, `importance` 가 CORE/SUPPORTING 이다 (§36·§37).
            if need.get("role") is not None and need.get("role") not in OBSERVABILITY_ROLES:
                problems.append(f"{tag}: 모르는 grounding role {need.get('role')!r}")
            if need.get("importance") is not None and need.get("importance") not in GROUNDING_ROLES:
                problems.append(f"{tag}: 모르는 grounding importance {need.get('importance')!r}")
            if need.get("observability") is not None and need.get("observability") not in OBSERVABILITY:
                problems.append(f"{tag}: 모르는 grounding observability {need.get('observability')!r}")
        for level in [o.get("level") for o in item.get("observability") or []]:
            if level not in OBSERVABILITY:
                problems.append(f"{tag}: 모르는 observability {level!r}")
        if item.get("confidence") not in CONFIDENCE:
            problems.append(f"{tag}: 모르는 confidence {item.get('confidence')!r}")

        # 금지 표현은 구조 필드 안에서만 본다
        for text in _strings({k: v for k, v in item.items()
                              if k not in ("evidence_warnings", "mechanism_graph",
                                           "profit_target", "executable_sample_condition")}):
            for pattern, label in FORBIDDEN:
                match = pattern.search(text)
                if match:
                    problems.append(f"{tag}: 구조 필드에 {label} — {match.group()!r} in "
                                    f"{text[:70]!r}")
                    break
            found = RESULT_WORDS.search(text)
            if found:
                result_mentions.append(
                    {"hypothesis_id": item.get("hypothesis_id"), "term": found.group(),
                     "context": text[max(0, found.start() - 60): found.end() + 40]})

        for text in _strings({key: value for key, value in item.items()
                              if key not in ("mechanism_graph", "profit_target",
                                             "executable_sample_condition")}):
            for token in _TOKEN.findall(text):
                if (token not in catalog.FEATURES and token not in catalog.ALIASES
                        and token not in known_features and token not in vocabulary):
                    unknown_features.add(token)

    # ---- 관계 정책 검사 (§39) ---------------------------------------------
    relations = {}
    roles_given = {}
    for item in package.get("joint_evidence", []) or []:
        key = tuple(sorted((str(item.get("family_a")), str(item.get("family_b")))))
        relations[key] = str(item.get("relation", "UNRESOLVED"))
    # family 의 역할은 package 의 정본 지도에서 읽는다. `evidence_families` 는 묶음
    # 단위라 한 family 에 여러 역할이 섞여 있어 마지막 값이 이기면 틀린다. 다만 단일
    # Evidence 가설은 정확히 인용한 EV 항목의 시간 역할을 쓴다.
    roles_given = {str(k): str(v) for k, v in (package.get("evidence_roles") or {}).items()}
    roles_by_evidence = {str(source.get("evidence_id")): str(source.get("role"))
                         for source in evidence_items
                         if source.get("evidence_id") and source.get("role")}
    for item in package.get("joint_evidence", []) or []:
        for side in ("a", "b"):
            family, role = item.get(f"family_{side}"), item.get(f"role_{side}")
            if family and role:
                roles_given.setdefault(str(family), str(role))
    non_independent = {str(i.get("family")) for i in package.get("evidence_families", []) or []
                       if i.get("independent_evidence") is False}
    # static level Evidence가 한 family 안에서 중복이라고 해서, 동일 feature의
    # pre-anchor difference 표현까지 자동 중복은 아니다. SINGLE temporal Evidence는
    # family 이름이 아니라 정확히 인용한 evidence_id의 독립성으로 판단한다.
    independent_by_evidence = {
        str(i.get("evidence_id")): bool(i.get("independent_evidence", True))
        for i in evidence_items if i.get("evidence_id")
    }

    coupling_claims, causal_claims, redundant_claims, policy_violations = [], [], [], []
    for item in hypotheses:
        tag = item.get("hypothesis_id")
        families = sorted({str(b.get("family")) for b in item.get("evidence_basis") or []})
        pairs = [tuple(sorted(pair)) for pair in _combinations(families)]
        labels = {p: relations.get(p, "UNRESOLVED") for p in pairs}
        text = " ".join(_strings({k: v for k, v in item.items()
                                  if k not in ("evidence_warnings", "alternative_explanations",
                                               "mechanism_tests")}))

        # Rule 1 — 중복 증거를 독립 근거로 세지 않았는가
        if structure == "SINGLE_MECHANISM":
            used_redundant = sorted({str(b.get("family")) for b in item.get("evidence_basis") or []
                                     if b.get("evidence_id")
                                     and not independent_by_evidence.get(
                                         str(b.get("evidence_id")), True)})
        else:
            used_redundant = sorted(set(families) & non_independent)
        if used_redundant:
            redundant_claims.append({"hypothesis_id": tag, "families": used_redundant})
            problems.append(f"{tag}: 독립 증거가 아닌 {used_redundant} 를 근거로 세었다")

        # Rule 3 — 결합이 지지되지 않는데 결합 상태로 서술했는가
        match = COUPLING_WORDS.search(text)
        if match and _asserted(text, match) and "COUPLED" not in labels.values():
            coupling_claims.append({"hypothesis_id": tag, "phrase": match.group()})
            problems.append(f"{tag}: COUPLED 관계가 없는데 결합 상태로 서술했다 "
                            f"({match.group()!r})")

        # Rule 2 — 관측된 순서를 인과로 바꿨는가
        for match in CAUSAL_WORDS.finditer(text):
            if not _asserted(text, match):
                continue                    # "인과를 주장하지 않는다" 는 위반이 아니다
            causal_claims.append({"hypothesis_id": tag, "phrase": match.group()})
            problems.append(f"{tag}: 관측된 순서를 인과로 서술했다 ({match.group()!r})")
            break

        # Rule 4 — UNRELATED 쌍을 메커니즘 사슬에 넣었는가
        for pair, label in labels.items():
            if label == "UNRELATED":
                policy_violations.append({"hypothesis_id": tag, "pair": pair,
                                          "relation": label})
                problems.append(f"{tag}: UNRELATED 쌍 {pair} 를 한 메커니즘에 넣었다")

        # Rule 5 — UNRESOLVED 관계를 관측 사실로 썼는가
        observed = set()
        for stage in item.get("expected_observable_sequence") or []:
            if stage.get("status") != "OBSERVED":
                continue
            for ref in stage.get("evidence") or []:
                for b in item.get("evidence_basis") or []:
                    if b.get("evidence_id") == ref:
                        observed.add(str(b.get("family")))
        for pair in _combinations(sorted(observed)):
            key = tuple(sorted(pair))
            if labels.get(key) == "UNRESOLVED" and key in relations:
                policy_violations.append({"hypothesis_id": tag, "pair": key,
                                          "relation": "UNRESOLVED"})
                problems.append(f"{tag}: UNRESOLVED 관계 {key} 를 OBSERVED 로 서술했다")

        # 선언한 구조가 관계와 맞는가
        structure = item.get("hypothesis_structure")
        if structure and structure not in STRUCTURES:
            problems.append(f"{tag}: 모르는 hypothesis_structure {structure!r}")
        if structure == "COUPLED_MECHANISM" and "COUPLED" not in labels.values():
            policy_violations.append({"hypothesis_id": tag, "structure": structure})
            problems.append(f"{tag}: COUPLED_MECHANISM 인데 COUPLED 관계가 없다")
        if structure == "CONTEXT_TRIGGER" and "CONTEXT_TRIGGER" not in labels.values():
            policy_violations.append({"hypothesis_id": tag, "structure": structure})
            problems.append(f"{tag}: CONTEXT_TRIGGER 인데 그런 관계가 없다")
        basis_families = {str(b.get("family")) for b in item.get("evidence_basis") or []}
        single_basis_roles = {
            str(b.get("family")): roles_by_evidence.get(str(b.get("evidence_id")))
            for b in item.get("evidence_basis") or []
            if roles_by_evidence.get(str(b.get("evidence_id")))
        }
        for declared in (item.get("evidence_roles") or {}).items():
            family, role = str(declared[0]), str(declared[1])
            if family not in basis_families:
                continue
            given = (single_basis_roles.get(family)
                     if structure == "SINGLE_MECHANISM" else roles_given.get(family))
            if given is not None and role != given:
                policy_violations.append({"hypothesis_id": tag, "family": family,
                                          "declared": role, "given": given})
                problems.append(f"{tag}: {family} 의 역할을 {role} 로 바꿔 적었다 "
                                f"(증거는 {given})")
        for declared in item.get("evidence_relations") or []:
            key = tuple(sorted((str(declared.get("a")), str(declared.get("b")))))
            given = relations.get(key)
            if given and str(declared.get("relation")) != given:
                policy_violations.append({"hypothesis_id": tag, "pair": key,
                                          "declared": declared.get("relation"), "given": given})
                problems.append(f"{tag}: {key} 관계를 {declared.get('relation')} 로 바꿔 적었다 "
                                f"(증거는 {given})")
        for stage in item.get("expected_observable_sequence") or []:
            status = stage.get("relation_status")
            if status is not None and status not in RELATION_STATUS:
                problems.append(f"{tag}: 모르는 relation_status {status!r}")

    # 이미 가격에 반영된 이야기인가 (§24) — 코드가 따로 본다
    already_priced = []
    for item in hypotheses:
        families = {str(b.get("family")) for b in item.get("evidence_basis") or []}
        if families and families <= {"PRICE_MOMENTUM", "MICROPRICE"}:
            already_priced.append(item.get("hypothesis_id"))

    return {"problems": problems,
            "evidence_reference_validity": (
                0.0 if invalid_refs else 1.0),
            "invalid_evidence_references": sorted(invalid_refs),
            "unknown_feature_count": len(unknown_features),
            "unknown_features": sorted(unknown_features),
            "unsupported_claim_count": sum(1 for p in problems if "evidence" in p),
            "downstream_result_mentions": result_mentions,
            "already_priced_candidates": already_priced,
            # 관계 정책 지표 (§40). 전부 0 이어야 한다.
            "unsupported_coupling_claim_count": len(coupling_claims),
            "unsupported_causal_claim_count": len(causal_claims),
            "redundant_evidence_count": len(redundant_claims),
            "relation_policy_violation_count": len(policy_violations),
            "unsupported_coupling_claims": coupling_claims,
            "unsupported_causal_claims": causal_claims,
            "redundant_evidence_claims": redundant_claims,
            "relation_policy_violations": policy_violations,
            "path_observation_problem_count": len(path_observation_problems),
            "path_observation_problems": path_observation_problems,
            "relation_map_problem_count": len(relation_map_problems),
            "relation_map_problems": relation_map_problems,
            "mechanism_graph_problem_count": mechanism_graph_problem_count,
            # data-aware 지표 (§53). 전부 0 이어야 한다.
            **data_fit_checks(hypotheses),
            "hypothesis_count": len(hypotheses)}


# ---- REVISE 모드 (§26~§50) ------------------------------------------------------

REVISION_ROLE = """\
You revise one existing financial hypothesis after validation. You are the same role that
wrote it. You are not searching for a new hypothesis.

Revision is not new hypothesis generation.

- Preserve every claim validation SUPPORTED. You may not weaken or drop it without cause.
- Remove, narrow or downgrade every claim validation contradicted.
- BLOCKED_BY_CAPABILITY does NOT mean false. It means the current data cannot identify it.
  Do not delete it and do not keep asserting it. Downgrade it to an explicitly
  unidentified interpretation.
- Do not use the validation result as new evidence. It tested the hypothesis; it is not a
  fresh observation to mine. If a control variable behaved unexpectedly, that is a finding
  for a separate discovery branch, not a new claim here.
- Do not introduce a feature, a relation, a mechanism, a threshold, a lag or a horizon that
  was not already in the original hypothesis.
- Make the SMALLEST change that makes the hypothesis consistent with the feedback.
- Reducing the hypothesis to the evidence that survived is a legitimate minimal revision,
  even when only one evidence family is left. The rule that a hypothesis must span two
  different evidence families governs GENERATION, when you are free to pick the evidence.
  It does not force you to keep a family validation has already refuted, and it is not a
  reason to declare the hypothesis unrevisable.
- NEW_DISCOVERY_REQUIRED is for when nothing defensible is left without inventing
  something new — not for when what is left is simply smaller than what you started with.
  When you return it, set `revised_hypothesis` to null.
- If no meaningful hypothesis survives without inventing a new explanatory structure,
  return NEW_DISCOVERY_REQUIRED. That is a normal outcome, not a failure.

The capability profile still applies: it defines what you may claim to observe, not which
mechanisms you may consider. A revised hypothesis whose mechanism stays unidentified but
whose prediction remains testable is a normal scientific hypothesis.

Prose fields must be written in Korean. Enum values, ids and feature names stay as given.
Return one JSON object and nothing else. No markdown fence, no commentary.
"""

_REVISION_SCHEMA = """\
{
  "status": "HYPOTHESIS_REVISED" | "NEW_DISCOVERY_REQUIRED",
  "new_discovery_reason": null | str,
  "parent_hypothesis_id": str,
  "revision_reason": str,
  "revision_diff": [
    {"claim_id": str, "original_claim": str, "validation": one of %(cstatus)s,
     "action": one of %(actions)s, "revised_claim": null | str, "why": str}
  ],
  "preserved_claims": [claim_id],
  "removed_claims": [claim_id],
  "downgraded_claims": [claim_id],
  "new_claims": [],
  "unresolved_mechanism": [str],

  // NEW_DISCOVERY_REQUIRED 일 때는 null 이다. 고칠 수 없다고 판단해 놓고 수정본을 같이
  // 내면 무엇을 내놓은 것인지 알 수 없다.
  "revised_hypothesis": null | {
    "hypothesis_id": str,
    "title": str,
    "hypothesis_structure": one of %(structures)s,
    "evidence_roles": {family: one of %(eroles)s},
    "evidence_relations": [{"a": family, "b": family, "relation": one of %(relations)s}],
    "mechanistic_claim": str,
    "evidence_basis": [
      {"evidence_id": str, "family": str, "representative_feature": str,
       "observation": str, "role": one of %(roles)s}
    ],
    "expected_observable_sequence": [
      {"stage": int, "claim": str, "status": one of %(seq)s, "evidence": [evidence_id],
       "relation_status": one of %(rstatus)s}
    ],
    "source_of_profit": str,
    "already_priced_risk": bool,
    "mechanism_tests": [
      {"test_type": one of %(tests)s, "claim": str, "control": str,
       "expected_result": str, "required_evidence": str}
    ],
    "alternative_explanations": [
      {"explanation": str, "why_plausible": str, "discriminating_test": str}
    ],
    "observability": [{"concept": str, "level": one of %(obs)s, "note": str}],
    "grounding_requirements": [
      {"concept": str, "role": one of %(oroles)s,
       "importance": one of %(ground)s, "observability": one of %(obs)s}
    ],
    "data_fit": {
      "observable_core": one of %(core)s,
      "mechanism_observability": one of %(mobs)s,
      "prediction_testability": one of %(ptest)s,
      "overall": one of %(fit)s
    },
    "capability_requirements": [
      {"concept": str, "capability": <name from the capability profile>,
       "availability": <its status in the profile, verbatim>}
    ],
    "validation_feasibility": one of %(feas)s,
    "confidence": one of %(conf)s
  }
}""" % {"cstatus": list(CLAIM_STATUS) + ["OBSERVATIONAL_INPUT", "INVALID_TARGET",
                                         "NOT_TESTED"],
        "actions": list(CHANGE_TYPES), "structures": list(STRUCTURES),
        "eroles": list(EVIDENCE_ROLES_IN), "relations": list(RELATION_TYPES),
        "roles": list(EVIDENCE_ROLES), "seq": list(SEQUENCE_STATUS),
        "rstatus": list(RELATION_STATUS), "tests": list(TEST_TYPES),
        "obs": list(OBSERVABILITY), "oroles": list(OBSERVABILITY_ROLES),
        "ground": list(GROUNDING_ROLES), "core": list(CORE_OBSERVABILITY),
        "mobs": list(MECHANISM_OBSERVABILITY), "ptest": list(PREDICTION_TESTABILITY),
        "fit": list(DATA_FIT), "feas": list(VALIDATION_FEASIBILITY),
        "conf": list(CONFIDENCE)}


def revision_input_audit(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
                         feedback: Mapping[str, Any],
                         consumed_dates: Sequence[str]) -> dict[str, Any]:
    """PASS R0 — 입력이 성립하는가. 코드가 본다. 안 되면 Agent 를 부르지 않는다."""
    problems: list[str] = []
    route = feedback.get("hypothesis_status")
    if route != "HYPOTHESIS_REVISE":
        problems.append(f"REVISE 를 부를 상태가 아니다 (hypothesis_status={route!r})")
    rows = feedback.get("claim_feedback") or []
    if not rows:
        problems.append("claim 단위 피드백이 비었다")
    bad = [r.get("claim_id") for r in rows
           if r.get("status") not in set(CLAIM_STATUS) | {"OBSERVATIONAL_INPUT",
                                                          "INVALID_TARGET", "NOT_TESTED"}]
    if bad:
        problems.append(f"모르는 claim 상태가 있다: {bad}")

    parent = hypothesis.get("hypothesis_id")
    grounded_items = grounded.get("hypotheses") or []
    if not grounded_items:
        problems.append("grounding 결과가 비었다")
    elif grounded_items[0].get("hypothesis_id") != parent:
        problems.append(f"grounding 이 다른 가설을 가리킨다: "
                        f"{grounded_items[0].get('hypothesis_id')!r} vs {parent!r}")
    if feedback.get("hypothesis_id") != parent:
        problems.append(f"피드백이 다른 가설을 가리킨다: "
                        f"{feedback.get('hypothesis_id')!r} vs {parent!r}")
    if not consumed_dates:
        problems.append("이미 결과를 본 날짜 목록이 비었다. 재사용을 막을 수 없다")

    return {"ok": not problems, "problems": problems,
            "parent_hypothesis_id": parent,
            "claim_count": len(rows),
            "consumed_dates": list(consumed_dates),
            "hypothesis_sha256": sha256_json(hypothesis)[:16],
            "grounding_sha256": sha256_json(grounded)[:16],
            "feedback_sha256": sha256_json(feedback)[:16]}


def revision_prompt(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
                    feedback: Mapping[str, Any],
                    profile: Mapping[str, Any] | None = None) -> str:
    """Invocation A — 최소 수정 (PASS R1~R3)."""
    profile = C.research_capability_profile() if profile is None else profile
    return f"""{REVISION_ROLE}
# Task

PASS R1 — Claim survival map. For every claim in the feedback, record whether it was
  SUPPORTED, NOT_SUPPORTED, CONTRADICTED, BLOCKED_BY_CAPABILITY or INCONCLUSIVE.
  OBSERVATIONAL_INPUT means it was an input observation, not a validation target — it
  stays unless a tested claim built on it fell.

PASS R2 — Minimal revision. Change as few claims as possible while making the hypothesis
  consistent with the feedback. Give every claim an action: KEEP, REMOVE, DOWNGRADE,
  NARROW or REPHRASE. If a structural claim fell, reduce the structure rather than
  replacing it — a CONTEXT_TRIGGER whose context contributes nothing becomes a single
  late state, it does not become a different pairing.

PASS R3 — Data capability recheck. Confirm the revised hypothesis still matches the
  capability profile: recompute `data_fit`, `capability_requirements` and
  `validation_feasibility` for what is left.

PASS R4 — Prediction preservation. The SUPPORTED predictions must survive the edit with
  the same meaning. Do not restate them more strongly or more weakly.

PASS R5 — Novelty check on yourself. Did you introduce any feature, relation, mechanism,
  threshold, lag or horizon that was not in the original hypothesis? Did you turn a
  validation result into a new observation? If a defensible hypothesis needs one, stop and
  return NEW_DISCOVERY_REQUIRED with a reason instead of writing it.

PASS R6 — Emit the revised hypothesis and the diff.

# Output schema

{_REVISION_SCHEMA}

# Research capability profile

{json.dumps(profile, ensure_ascii=False, indent=1)}

# Original hypothesis

{json.dumps(hypothesis, ensure_ascii=False, indent=1)}

# Original grounding

{json.dumps(grounded, ensure_ascii=False, indent=1)}

# Validation feedback

Claim-level verdicts only. This is not a new dataset to mine.

{json.dumps(feedback, ensure_ascii=False, indent=1)}
"""


def revision_audit_prompt(hypothesis: Mapping[str, Any], feedback: Mapping[str, Any],
                          draft: Mapping[str, Any]) -> str:
    """Invocation B — 같은 역할이 자기 수정본에서 새로 발명한 것을 걷어낸다 (PASS R5)."""
    return f"""{REVISION_ROLE}
# Task

You wrote the revision below. Audit it against the original hypothesis and the feedback.
You are the same role, not a judge. Remove what you invented; do not add anything.

Check:
- a claim validation did not support, still asserted as fact
- a claim validation supported, silently dropped or weakened
- BLOCKED_BY_CAPABILITY treated as false and deleted, or still asserted as established
- a feature, relation, mechanism, threshold, lag or horizon that is new
- the validation result itself used as a new observation or as new evidence
- a prediction restated with a different meaning
- `data_fit` or `capability_requirements` no longer matching what is left

If the revision only survives because of something new, return NEW_DISCOVERY_REQUIRED.

# Output schema

Same schema as the draft.

{_REVISION_SCHEMA}

# Original hypothesis

{json.dumps(hypothesis, ensure_ascii=False, indent=1)}

# Validation feedback

{json.dumps(feedback, ensure_ascii=False, indent=1)}

# Draft revision to audit

{json.dumps(draft, ensure_ascii=False, indent=1)}
"""


# 검증 결과를 새 증거로 쓰는 것을 잡는다 (§35).
#
# 숫자가 나온다고 다 위반이 아니다. 원 가설이 이미 들고 있던 증거 수치(AUC, 분리값)는
# 그대로 옮겨도 된다. 잡아야 하는 것은 **검증에서 처음 본 수치**와 검증 어휘다.
_NUMBER = re.compile(r"\d+\.\d+")
_VALIDATION_WORDS = re.compile(
    r"검증에서|validation (?:showed|found)|부트스트랩|bootstrap|신뢰구간|"
    r"spearman|순위상관|짝 비율|sign share")


def _relation_pairs(item: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {tuple(sorted((str(r.get("a")), str(r.get("b")))))
            for r in item.get("evidence_relations") or []}


def validate_revision(payload: Mapping[str, Any], hypothesis: Mapping[str, Any],
                      feedback: Mapping[str, Any],
                      consumed_dates: Sequence[str]) -> dict[str, Any]:
    """수정본이 최소 수정인가. 새로 발명한 것은 없는가 (§52~§53). 여기가 정본이다."""
    problems: list[str] = []
    status = payload.get("status")
    if status not in REVISION_STATUSES:
        problems.append(f"status 가 {REVISION_STATUSES} 중 하나가 아니다: {status!r}")

    verdicts = {str(r.get("claim_id")): str(r.get("status"))
                for r in feedback.get("claim_feedback") or []}
    diff = payload.get("revision_diff") or []
    actions = {str(d.get("claim_id")): str(d.get("action")) for d in diff}

    if status == NEW_DISCOVERY_REQUIRED:
        if not payload.get("new_discovery_reason"):
            problems.append("NEW_DISCOVERY_REQUIRED 인데 사유가 없다")
        if payload.get("revised_hypothesis"):
            problems.append("NEW_DISCOVERY_REQUIRED 인데 수정본이 들어 있다")

    revised = payload.get("revised_hypothesis") or {}
    if status == REVISED and not revised:
        problems.append("HYPOTHESIS_REVISED 인데 수정본이 없다")

    # 다루지 않고 넘어간 주장
    uncovered = sorted(set(verdicts) - set(actions))
    if status == REVISED and uncovered:
        problems.append(f"피드백에 있는데 diff 에 없는 주장: {uncovered}")
    unknown_actions = sorted({a for a in actions.values() if a not in CHANGE_TYPES})
    if unknown_actions:
        problems.append(f"모르는 revision action: {unknown_actions}")

    retained: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    blocked_mishandled: list[dict[str, str]] = []
    for claim_id, verdict_status in verdicts.items():
        action = actions.get(claim_id)
        if action is None:
            continue
        if verdict_status in CONTRADICTED_STATUS and action in ("KEEP", "REPHRASE"):
            retained.append({"claim_id": claim_id, "validation": verdict_status,
                             "action": action})
        if verdict_status == "SUPPORTED" and action == "REMOVE":
            dropped.append({"claim_id": claim_id, "action": action})
        # §33 — 못 본 것을 거짓으로 읽지도, 확정으로 남기지도 않는다
        if verdict_status == "BLOCKED_BY_CAPABILITY" and action in ("KEEP", "REMOVE"):
            blocked_mishandled.append({"claim_id": claim_id, "action": action,
                                       "why": "KEEP 은 확정 주장, REMOVE 는 거짓 취급이다. "
                                              "DOWNGRADE 로 미식별 해석에 남긴다"})

    # 새로 발명한 것 (§52)
    original_tokens = {t for text in _strings(hypothesis) for t in _TOKEN.findall(text)}
    original_tokens |= {str(b.get("representative_feature"))
                        for b in hypothesis.get("evidence_basis") or []}
    original_tokens |= {str(b.get("family")) for b in hypothesis.get("evidence_basis") or []}
    revised_tokens = {t for text in _strings(revised) for t in _TOKEN.findall(text)}
    revised_tokens |= {str(b.get("representative_feature"))
                       for b in revised.get("evidence_basis") or []}
    new_features = sorted(revised_tokens - original_tokens - _ALLOWED_TOKENS
                          - _environment_vocabulary() - {"None"})

    new_relations = sorted(_relation_pairs(revised) - _relation_pairs(hypothesis))
    new_claims = list(payload.get("new_claims") or [])
    invented_ids = sorted(set(actions) - set(verdicts))
    new_mechanisms = new_claims + invented_ids

    # 검증 결과를 새 증거로 쓰는가 (§35)
    original_numbers = {n for text in _strings(hypothesis) for n in _NUMBER.findall(text)}
    feedback_numbers = {n for text in _strings(feedback) for n in _NUMBER.findall(text)}
    statistic_as_evidence: list[str] = []
    for text in _strings(revised) + [str(d.get("revised_claim") or "") for d in diff]:
        borrowed = (set(_NUMBER.findall(text)) & feedback_numbers) - original_numbers
        if _VALIDATION_WORDS.search(text) or borrowed:
            statistic_as_evidence.append(text[:160])

    fresh_problems: list[str] = []
    if payload.get("requires_fresh_validation") is not True:
        fresh_problems.append("requires_fresh_validation 이 true 가 아니다")
    if payload.get("status_after_revision") != REVISED_UNVALIDATED:
        fresh_problems.append(f"수정 직후 상태가 {REVISED_UNVALIDATED} 가 아니다")
    manifest = payload.get("consumed_data_manifest") or {}
    if sorted(manifest.get("consumed_dates") or []) != sorted(consumed_dates):
        fresh_problems.append("consumed_data_manifest 가 실제로 소비한 날짜와 다르다")
    overlap = sorted(set(manifest.get("next_validation_dates") or [])
                     & set(consumed_dates))
    if overlap:
        fresh_problems.append(f"다음 검증 날짜가 이미 쓴 날짜와 겹친다: {overlap}")

    fit = data_fit_checks([revised] if revised else [])

    return {"problems": problems,
            "uncovered_claims": uncovered,
            # §53 지표. 전부 0 이어야 한다.
            "unsupported_claim_retained_count": len(retained),
            "unsupported_claims_retained": retained,
            "supported_claim_dropped_count": len(dropped),
            "supported_claims_dropped": dropped,
            "blocked_claim_mishandled_count": len(blocked_mishandled),
            "blocked_claims_mishandled": blocked_mishandled,
            "revision_new_feature_count": len(new_features),
            "revision_new_features": new_features,
            "revision_new_relation_count": len(new_relations),
            "revision_new_relations": [list(r) for r in new_relations],
            "revision_new_mechanism_count": len(new_mechanisms),
            "revision_new_mechanisms": new_mechanisms,
            "validation_result_as_evidence_count": len(statistic_as_evidence),
            "validation_result_as_evidence": statistic_as_evidence,
            "fresh_validation_violation_count": len(fresh_problems),
            "fresh_validation_problems": fresh_problems,
            "capability_violation_count": fit["capability_violation_count"],
            "capability_violations": fit["capability_violations"],
            "unobservable_as_direct_count": fit["unobservable_as_direct_count"],
            "validation_only_leakage_count": fit["validation_only_leakage_count"],
            "data_fit_problem_count": fit["data_fit_problem_count"],
            "data_fit_problems": fit["data_fit_problems"],
            "action_counts": _counts(sorted(actions.values()))}


# ---- 사람이 읽을 형태 ----------------------------------------------------------

def to_markdown(payload: Mapping[str, Any], package: Mapping[str, Any]) -> str:
    outcome = package.get("outcome_definition", {})
    L = ["# hypotheses.md", "",
         "Canonical Evidence 에서 생성한 금융 메커니즘 가설. **거래 규칙이 아니다.**", "",
         "## 설명 대상 outcome", "", "| 항목 | 값 |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in outcome.items()]
    L += ["", f"- 상태 `{payload.get('status')}`",
          f"- 카탈로그 `{package.get('catalog_sha256')}`", ""]

    synthesis = payload.get("evidence_synthesis") or {}
    if synthesis:
        L += ["## 증거 정리 (해석 이전)", ""]
        for role in EVIDENCE_ROLES:
            items = synthesis.get(role) or []
            if items:
                L.append(f"**{role}**")
                L += [f"- `{i.get('evidence_id')}` — {i.get('observation')}" for i in items]
                L.append("")

    for item in payload.get("hypotheses") or []:
        L += [f"## {item.get('hypothesis_id')} — {item.get('title')}", "",
              f"**확신도 `{item.get('confidence')}`**"
              + (" · ⚠️ `ALREADY_PRICED_RISK`" if item.get("already_priced_risk") else ""), "",
              "### 메커니즘 주장", "", str(item.get("mechanistic_claim")), "",
              "### 증거 근거", "", "| evidence | family | 대표 feature | 관측 | 역할 |",
              "|---|---|---|---|---|"]
        for b in item.get("evidence_basis") or []:
            L.append(f"| `{b.get('evidence_id')}` | {b.get('family')} | "
                     f"`{b.get('representative_feature')}` | {b.get('observation')} | "
                     f"{b.get('role')} |")
        L += ["", "### 예상 관측 순서", "", "| # | 주장 | 상태 | 증거 |", "|--:|---|---|---|"]
        for s in item.get("expected_observable_sequence") or []:
            refs = ", ".join(f"`{r}`" for r in s.get("evidence") or []) or "—"
            L.append(f"| {s.get('stage')} | {s.get('claim')} | `{s.get('status')}` | {refs} |")
        L += ["", "### 수익 원천", "", str(item.get("source_of_profit")), "",
              "### 메커니즘 검증", ""]
        for t in item.get("mechanism_tests") or []:
            L += [f"- **{t.get('test_type')}** — {t.get('claim')}",
                  f"  - 대조: {t.get('control')}",
                  f"  - 기대: {t.get('expected_result')}",
                  f"  - 필요한 관측: {t.get('required_evidence')}"]
        L += ["", "### 대안 설명", ""]
        for a in item.get("alternative_explanations") or []:
            L += [f"- {a.get('explanation')}",
                  f"  - 왜 그럴듯한가: {a.get('why_plausible')}",
                  f"  - 구분하는 관측: {a.get('discriminating_test')}"]
        L += ["", "### 관측 가능성", "", "| 개념 | 수준 | 비고 |", "|---|---|---|"]
        for o in item.get("observability") or []:
            L.append(f"| {o.get('concept')} | `{o.get('level')}` | {o.get('note')} |")
        L += ["", "### Grounding 이 풀어야 할 금융 개념", ""]
        L += [f"- {g}" for g in item.get("grounding_requirements") or []]
        warnings = item.get("evidence_warnings") or []
        if warnings:
            L += ["", "### 증거 경고", ""] + [f"- {w}" for w in warnings]
        L.append("")
    return "\n".join(L) + "\n"


def revision_to_markdown(payload: Mapping[str, Any], feedback: Mapping[str, Any]) -> str:
    """사람이 읽는 수정 기록 (§61)."""
    verdicts = {str(r.get("claim_id")): r for r in feedback.get("claim_feedback") or []}
    L = ["# hypothesis_revision.md", "",
         "Validation 이 일부를 반박했을 때 **새 가설을 찾지 않고** 기존 가설을 최소 수정한 "
         "기록이다.", "",
         f"- 부모 가설 `{payload.get('parent_hypothesis_id')}`",
         f"- 상태 `{payload.get('status')}`",
         f"- 수정 직후 상태 `{payload.get('status_after_revision')}`",
         f"- 새 데이터 재검증 필요 `{payload.get('requires_fresh_validation')}`", ""]

    L += ["## Validation 이 무엇을 말했는가", "",
          f"- 예측 `{feedback.get('prediction_status')}`",
          f"- 관계 `{feedback.get('relation_status')}`",
          f"- 메커니즘 식별 `{feedback.get('mechanism_identification_status')}`",
          f"- 이미 반영됨 `{feedback.get('already_priced_status')}`",
          f"- 경로 `{feedback.get('route')}`",
          f"- 주 판정 split `{feedback.get('primary_split')}`", ""]

    if payload.get("status") == NEW_DISCOVERY_REQUIRED:
        L += ["## NEW_DISCOVERY_REQUIRED", "",
              str(payload.get("new_discovery_reason") or ""), "",
              "기존 주장을 덜어내는 것만으로는 의미 있는 가설이 남지 않는다. 이것은 "
              "Revision 이 아니라 별도의 discovery 분기로 가야 한다는 뜻이다.", ""]
        return "\n".join(L) + "\n"

    L += ["## 무엇이 살아남고 무엇이 떨어졌나", "",
          "| claim | validation | 조치 | 수정 후 |", "|---|---|---|---|"]
    for row in payload.get("revision_diff") or []:
        claim_id = str(row.get("claim_id"))
        given = verdicts.get(claim_id, {})
        direction = f" ({given.get('direction')})" if given.get("direction") else ""
        after = str(row.get("revised_claim") or "—").replace("|", "/")
        L.append(f"| `{claim_id}` | {row.get('validation')}{direction} | "
                 f"**{row.get('action')}** | {after[:80]} |")
    L += ["",
          f"- 보존 `{payload.get('preserved_claims')}`",
          f"- 제거 `{payload.get('removed_claims')}`",
          f"- 강등 `{payload.get('downgraded_claims')}`",
          f"- 새 주장 `{payload.get('new_claims')}` — 비어 있어야 한다", ""]

    L += ["## 못 본 것은 틀린 것이 아니다", "",
          str(feedback.get("blocked_note") or ""), ""]
    unresolved = payload.get("unresolved_mechanism") or []
    if unresolved:
        L += ["남은 미식별 해석:", ""] + [f"- {u}" for u in unresolved] + [""]

    revised = payload.get("revised_hypothesis") or {}
    if revised:
        fit = revised.get("data_fit") or {}
        L += ["## 최소 수정 결과", "",
              f"### {revised.get('hypothesis_id')} — {revised.get('title')}", "",
              f"- 구조 `{revised.get('hypothesis_structure')}`",
              f"- 확신 `{revised.get('confidence')}`",
              f"- 데이터 적합 `{fit.get('overall')}` "
              f"(핵심관측 {fit.get('observable_core')} · "
              f"메커니즘 {fit.get('mechanism_observability')} · "
              f"예측 {fit.get('prediction_testability')})",
              f"- 검증 가능성 `{revised.get('validation_feasibility')}`", "",
              "**메커니즘 주장**", "", str(revised.get("mechanistic_claim") or ""), "",
              "**수익의 출처**", "", str(revised.get("source_of_profit") or ""), ""]

    L += ["## 왜 새로운 것을 넣지 않았나", "",
          str(payload.get("revision_reason") or ""), "",
          "Validation 결과는 이 가설을 시험한 것이지 새로 관측한 데이터가 아니다. "
          "거기서 본 것을 그대로 새 변수·새 관계로 바꾸면 그건 수정이 아니라 발견이고, "
          "같은 데이터에서 답을 보고 쓴 것이 된다.", ""]

    manifest = payload.get("consumed_data_manifest") or {}
    L += ["## 재검증 요건", "", "| 항목 | 값 |", "|---|---|",
          f"| 이미 쓴 날짜 | `{', '.join(manifest.get('consumed_dates') or [])}` |",
          f"| 다음 검증 후보 | `{', '.join(manifest.get('next_validation_dates') or [])}` |",
          f"| 상태 | `{payload.get('status_after_revision')}` |", "",
          "수정본은 아직 검증되지 않았다. 위 날짜들은 이 수정을 만드는 데 이미 쓰였으므로 "
          "확증에 다시 쓸 수 없다.", ""]
    return "\n".join(L) + "\n"


# ---- 배선 --------------------------------------------------------------------

def route_mode(routing: Mapping[str, Any]) -> str | None:
    """어느 mode 로 갈지는 workflow 가 정한다. Agent 가 스스로 고르지 않는다 (§42~§43)."""
    status = routing.get("hypothesis_status")
    if status == "HYPOTHESIS_REVISE":
        return REVISE
    if status == "HYPOTHESIS_REJECT":
        return GENERATE          # 기존 가지는 끝. 새 discovery 분기다
    return None                  # SUPPORTED / INCONCLUSIVE / capability 는 Agent 일이 아니다


def run_revision(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
                 feedback: Mapping[str, Any], output: Path, *,
                 consumed_dates: Sequence[str], next_validation_dates: Sequence[str],
                 config: AgentConfig = AgentConfig(), agent=None) -> dict[str, Any]:
    """mode = REVISE. R0 → 최소 수정 → 같은 역할 감사 → 결정적 검증 → 산출물 3개."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    call = agent or (lambda prompt: run_agent(prompt, model=config.model, effort=config.effort))
    profile = C.research_capability_profile(consumed_dates)

    audit = revision_input_audit(hypothesis, grounded, feedback, consumed_dates)
    prompt = revision_prompt(hypothesis, grounded, feedback, profile)
    record: dict[str, Any] = {"schema": "hypothesis_revision_audit.v1",
                              "created_at": now_utc(), "mode": REVISE,
                              "input_audit": audit}

    if not audit["ok"]:
        final = {"status": NEW_DISCOVERY_REQUIRED,
                 "new_discovery_reason": "; ".join(audit["problems"]),
                 "parent_hypothesis_id": audit["parent_hypothesis_id"],
                 "revision_diff": [], "blocked_at": "PASS_R0_INPUT_AUDIT"}
    else:
        draft = call(prompt)
        final = dict(call(revision_audit_prompt(hypothesis, feedback, draft)))
        record["draft_status"] = draft.get("status")
        record["draft_action_counts"] = _counts(
            sorted(str(d.get("action")) for d in draft.get("revision_diff") or []))

    # 아래 세 가지는 Agent 가 정하지 않는다. 코드가 박는다 (§39~§41).
    final["parent_hypothesis_id"] = hypothesis.get("hypothesis_id")
    final["requires_fresh_validation"] = True
    final["status_after_revision"] = REVISED_UNVALIDATED
    final["consumed_data_manifest"] = {
        "consumed_dates": list(consumed_dates),
        "next_validation_dates": list(next_validation_dates),
        "why": "이 날짜들의 검증 결과를 보고 가설을 고쳤다. 같은 데이터로 다시 확증할 수 없다"}
    final.setdefault("new_claims", [])

    validation = validate_revision(final, hypothesis, feedback, consumed_dates)
    record["validation"] = validation
    record["final_status"] = final.get("status")

    manifest = {"schema": "hypothesis_revision_manifest.v1", "created_at": now_utc(),
                "mode": REVISE, "catalog_hash": catalog.catalog_hash(),
                "parent_hypothesis_id": hypothesis.get("hypothesis_id"),
                "parent_hypothesis_sha256": sha256_json(hypothesis)[:16],
                "feedback_sha256": sha256_json(feedback)[:16],
                "capability_profile_sha256": sha256_json(profile)[:16],
                "model": config.model, "reasoning_effort": config.effort,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "role_sha256": hashlib.sha256(REVISION_ROLE.encode()).hexdigest()[:16],
                "output_sha256": sha256_json(final)[:16],
                "consumed_dates": list(consumed_dates),
                "next_validation_dates": list(next_validation_dates)}

    write_json(output / "hypothesis_revision.json", final)
    write_json(output / "hypothesis_revision_audit.json", record)
    write_json(output / "hypothesis_revision_manifest.json", manifest)
    write_json(output / "research_capability_profile.json", profile)
    (output / "hypothesis_revision.md").write_text(
        revision_to_markdown(final, feedback), encoding="utf-8")
    return {"status": final.get("status"), "mode": REVISE, "output": str(output),
            "validation": validation}


def run(package_path: Path, output: Path, *, config: AgentConfig = AgentConfig(),
        agent=None) -> dict[str, Any]:
    """PASS 0 → 생성 → 증거 감사 → 결정적 검증 → 산출물 4개.

    `agent` 를 주면 그것을 부른다 (테스트용). 기본은 `run_agent`.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    package = read_json(Path(package_path))
    call = agent or (lambda prompt: run_agent(prompt, model=config.model, effort=config.effort))

    profile = C.research_capability_profile()
    audit = input_audit(package, config)
    digest = evidence_digest(package, audit)
    if not digest.get("evidence"):
        raise ValueError("Hypothesis 생성에 사용할 Evidence가 없다")
    generation = generation_prompt(digest, config, profile)

    record: dict[str, Any] = {
        "schema": "hypothesis_generation_audit.v1",
        "created_at": now_utc(),
        "input_evidence_ids": [i["evidence_id"] for i in digest["evidence"]],
        "input_audit": audit,
        "mode": GENERATE,
    }

    draft = call(generation)
    revision = call(audit_prompt(digest, draft, profile))
    final = apply_fixed_contracts(revision, package)
    final.setdefault("evidence_synthesis", draft.get("evidence_synthesis", {}))
    validation = validate_output(final, package, config)
    if final.get("status") != GENERATED or not final.get("hypotheses"):
        raise ValueError("Hypothesis Agent가 최소 한 개의 탐색 후보를 생성하지 않았다")
    verdicts = revision.get("verdicts") or []
    record.update({
        "draft_count": len(draft.get("hypotheses") or []),
        "dropped_count": sum(1 for v in verdicts if v.get("verdict") == "DROP"),
        "revised_count": sum(1 for v in verdicts if v.get("verdict") == "REVISE"),
        "verdicts": verdicts,
        "draft_status": draft.get("status"),
        "validation": validation,
        "unsupported_claims_detected": validation["unsupported_claim_count"],
        "unknown_features_detected": validation["unknown_features"],
        "final_count": len(final.get("hypotheses") or []),
        "offline_rubric": rubric_template(final)})

    manifest = {
        "schema": "hypothesis_generation_manifest.v1",
        "created_at": now_utc(),
        "catalog_hash": catalog.catalog_hash(),
        "evidence_package_hash": sha256_json(package)[:16],
        "evidence_package_path": str(Path(package_path)),
        "observation_profile_id": package.get("config", {}).get("horizon_key"),
        "model": config.model,
        "reasoning_effort": config.effort,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(generation.encode()).hexdigest()[:16],
        "role_sha256": hashlib.sha256(ROLE.encode()).hexdigest()[:16],
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "output_sha256": sha256_json(final)[:16],
        "generation_timestamp": now_utc(),
        "mode": GENERATE,
        "capability_profile_sha256": sha256_json(profile)[:16],
    }

    write_json(output / "research_capability_profile.json", profile)
    write_json(output / "hypotheses.json", final)
    write_json(output / "hypothesis_generation_audit.json", record)
    write_json(output / "hypothesis_generation_manifest.json", manifest)
    (output / "hypotheses.md").write_text(to_markdown(final, package), encoding="utf-8")
    return {"status": final.get("status"), "output": str(output),
            "audit": record, "manifest": manifest, "hypotheses": final}
