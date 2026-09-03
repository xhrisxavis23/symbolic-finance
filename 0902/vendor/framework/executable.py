"""검증을 통과한 가설을 **의미를 바꾸지 않고** 실행 계약으로 옮긴다.

이 단계는 코드를 짜는 단계의 시작이지만, 더 정확히는 **코드와 탐색이 가설을 몰래 바꾸지
못하도록 경계를 먼저 잠그는 단계**다.

## 여기서 하지 않는 것

    임계값 고르기 · 파라미터 탐색 · 손익 최적화 · terminal OOS 열기

## 답해야 하는 한 문장

    파라미터 탐색에서 아무리 좋은 결과가 나와도,
    **어디까지 바뀌면 여전히 같은 가설이고 어디부터는 새 가설인가?**

## 네 가지를 가른다

    HYPOTHESIS_INVARIANT      검증이 지켜낸 의미. 바꾸면 다른 가설이다
    SEARCHABLE_PARAMETER      값이 바뀌어도 의미가 유지되는 것
    EXECUTION_BINDING         실행하려면 정해야 하지만 탐색에 넘기면 안 되는 것
    FORBIDDEN_TRANSFORMATION  하면 다른 가설이 되는 변형

핵심은 세 번째다. 진입 시점을 어떻게 잡을지 같은 것을 "파라미터" 로 두면, 검증이 서 있던
표본 규칙이 조용히 달라진다. 그건 탐색이 아니라 다른 실험이다.
"""

from __future__ import annotations

import hashlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (canonical as K, capability as C, catalog, contract, evidence as E,
               mechanism_graph as M, profit_target as PT, sample_condition as SC)
from .config import now_utc, read_json, sha256_json, write_json


SCHEMA_VERSION = "executable_specification.v1"

# 각 항목이 이후 단계에서 어떤 대접을 받는가 (§13).
FIXED_BY_HYPOTHESIS = "FIXED_BY_HYPOTHESIS"      # 가설이 그렇게 말했다
FIXED_BY_VALIDATION = "FIXED_BY_VALIDATION"      # 검증이 그 방향을 지켜냈다
SEARCHABLE = catalog.SEARCHABLE
EXECUTION_ASSUMPTION = "EXECUTION_ASSUMPTION"    # 실행하려면 정해야 한다. 탐색 대상 아님
OUT_OF_SCOPE = "OUT_OF_SCOPE"                    # 이 단계가 정할 일이 아니다
FORBIDDEN = "FORBIDDEN"                          # 넣으면 다른 가설이다
PARAMETER_STATUS = (FIXED_BY_HYPOTHESIS, FIXED_BY_VALIDATION, SEARCHABLE,
                    EXECUTION_ASSUMPTION, OUT_OF_SCOPE, FORBIDDEN)

# 명세가 가설과 같은 뜻인가 (§26~§29).
FIDELITY_PRESERVED = "FIDELITY_PRESERVED"
FIDELITY_AMBIGUOUS = "FIDELITY_AMBIGUOUS"
FIDELITY_VIOLATED = "FIDELITY_VIOLATED"
FIDELITY_STATUS = (FIDELITY_PRESERVED, FIDELITY_AMBIGUOUS, FIDELITY_VIOLATED)


class UnsupportedTemporalGrounding(ValueError):
    """Grounding의 시간 관계를 현재 Catalog DSL로 보존할 수 없을 때만 쓴다."""

# 다음 단계로 갈 수 있는가 (§41~§42).
READY_FOR_IMPLEMENTATION = "READY_FOR_IMPLEMENTATION"
READY_FOR_PARAMETER_SEARCH = "READY_FOR_PARAMETER_SEARCH"
BLOCKED_BY_FIDELITY_AMBIGUITY = "BLOCKED_BY_FIDELITY_AMBIGUITY"
INVALID_SPECIFICATION = "INVALID_SPECIFICATION"
READINESS = (READY_FOR_IMPLEMENTATION, READY_FOR_PARAMETER_SEARCH,
             BLOCKED_BY_FIDELITY_AMBIGUITY, INVALID_SPECIFICATION)

# 탐색이 좋은 것을 찾아도 그것이 이 가설의 결과가 아닐 때 (§33).
SEARCH_DISCOVERY_CANDIDATE = "SEARCH_DISCOVERY_CANDIDATE"

# 검증이 지켜낸 방향. 통계의 부호에서 읽는다 — 손으로 적지 않는다.
HIGHER, LOWER = "HIGHER", "LOWER"

# 부등호는 방향에서 따라 나온다. 고르는 것이 아니다.
COMPARATOR = {HIGHER: ">", LOWER: "<"}

# 임계의 숫자가 어디에서 오는가. expression 자체는 `value` 한 자리만 가지며,
# 이 정보는 Specification/Search가 그 자리를 어떻게 채울지 정한다.
THRESHOLD_LITERAL = catalog.THRESHOLD_LITERAL
THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE = catalog.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE
THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE = catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
THRESHOLD_KINDS = (THRESHOLD_LITERAL, THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
                   THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE)


@dataclass(frozen=True)
class SpecConfig:
    """이 단계가 새로 고르는 값은 없다. 전부 위 단계에서 온다."""

    threshold_prefix: str = "theta_"
    threshold_kind: str = THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    # 빈 값이면 SearchConfig의 해당 source 기본 격자를 쓴다.
    threshold_grid: tuple[float, ...] = ()
    # 방향은 **경로 target** 이 읽는다. 이름이 아니라 종류로 찾는다 — 가설마다 이름이
    # 다르고 이름을 박으면 그 가설 전용 코드가 된다.
    primary_target_type: str = "PATH_TARGET"


# ---- 1. 바꾸면 안 되는 의미 (§3~§7) ---------------------------------------------

def _core_feature(revised: Mapping[str, Any],
                  tested: str | None = None) -> tuple[str, str]:
    """가설이 실제로 기댄 관측 하나. 역할이 trigger 인 증거에서 읽는다.

    `tested` 를 주면 그것이 정본이다 — **검증이 실제로 시험한 feature** 이기 때문이다.
    trigger 후보가 여럿인 가설에서 여기와 검증이 각자 고르면 서로 다른 것을 가리킨다.
    """
    basis = revised.get("evidence_basis") or []
    if tested:
        for item in basis:
            if str(item.get("representative_feature")) == str(tested):
                return str(item.get("family")), str(tested)
    roles = revised.get("evidence_roles") or {}
    for role in ("LATE_TRIGGER", "TRIGGER", "SETUP_STATE"):
        trigger = {family for family, assigned in roles.items() if str(assigned) == role}
        for item in basis:
            if str(item.get("family")) in trigger:
                return str(item.get("family")), str(item.get("representative_feature"))
    for item in basis:                       # 역할 표가 비면 TRIGGER 로 적힌 것을 본다
        if str(item.get("role")) in ("TRIGGER", "LATE_TRIGGER"):
            return str(item.get("family")), str(item.get("representative_feature"))
    raise ValueError("수정본에서 핵심 관측을 찾지 못했다. 역할 표가 비어 있다")


def _direction(results: Mapping[str, Any], config: SpecConfig,
               contracts: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """방향은 검증 통계의 부호가 정한다. 명세가 고르는 것이 아니다.

    어느 결과에서 읽는지는 **경로 target** 이라는 종류로 찾는다. 이름은 가설마다 다르다.
    """
    contracts = dict(contracts or {})

    def kind(name: str) -> str | None:
        return ((results.get(name) or {}).get("target_type")
                or (contracts.get(name) or {}).get("target_type"))

    ordered = sorted(results)
    target = next((results[n] for n in ordered
                   if kind(n) == config.primary_target_type), None)
    if target is None and not any(kind(n) for n in ordered):
        target = results[ordered[0]] if ordered else {}     # 종류가 안 적힌 옛 결과
    target = target or {}
    effect, null = target.get("effect_size"), target.get("null_value")
    if effect is None:
        raise ValueError(f"{config.primary_target_type} 결과가 없어 방향을 읽을 수 없다")
    null = 0.0 if null is None else float(null)
    return HIGHER if float(effect) > null else LOWER


def semantic_invariants(revision: Mapping[str, Any], routing: Mapping[str, Any],
                        manifest: Mapping[str, Any], results: Mapping[str, Any],
                        config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """검증이 지켜낸 것. 이후 어떤 단계도 이것을 바꿀 수 없다."""
    revised = revision.get("revised_hypothesis") or (
        dict(revision) if revision.get("hypothesis_id")
        else dict((revision.get("hypotheses") or [{}])[0]))
    outcome = manifest.get("outcome_definition") or {}
    plan = manifest.get("validation_plan") or {}
    cfg = manifest.get("config") or {}
    # 검증이 시험한 feature 가 정본이다. 여기서 다시 고르지 않는다.
    validated_feature = plan.get("trigger_feature") or cfg.get("trigger_feature")
    family, feature = _core_feature(revised, validated_feature)

    removed = sorted({str(c) for c in revision.get("removed_claims") or []})
    context_feature = str(plan.get("context_feature") or cfg.get("context_feature") or "")
    context_family = (E.family_of(catalog.FEATURES[context_feature])
                      if context_feature in catalog.FEATURES else None)

    horizon = float(cfg.get("horizon_seconds") or 30.0)
    return {
        "hypothesis_id": routing.get("hypothesis_id"),
        "hypothesis_status": routing.get("hypothesis_status"),
        "evidence_tier": routing.get("evidence_tier"),
        # 검증이 실제로 시험한 feature. 명세도 감사도 이것을 본다.
        "validated_feature": validated_feature,
        # -- 절대 바꾸면 안 되는 것 --------------------------------------------
        "feature_family": family,
        "canonical_feature": feature,
        "role": "late observable state",
        "direction": _direction(results, config, plan.get("target_contracts")),
        "side": str(outcome.get("side") or "LONG"),
        "context_required": False,
        "rejected_context_family": context_family,
        "rejected_context_feature": context_feature,
        "mechanism_identified": routing.get("mechanism_identification_status") == "IDENTIFIED",
        "unresolved_mechanism": list(revision.get("unresolved_mechanism") or []),
        "removed_claims": removed,
        # -- 검증이 서 있던 실행 의미 ------------------------------------------
        "validation_horizon": f"S{int(horizon)}",
        "validation_horizon_seconds": horizon,
        "entry_reference": str(outcome.get("entry_price") or "ASK1"),
        "future_execution_reference": str(outcome.get("exit_price") or "BID1"),
        "cost_bps": outcome.get("cost_bps"),
        "why_fixed": {
            "canonical_feature": "이 관측 하나로 검증했다. 비슷한 금융 개념이라고 다른 "
                                 "feature 로 바꾸면 검증한 적 없는 것을 실행하는 것이다",
            "direction": "여러 홀드아웃에서 이 방향으로 지지됐다. 뒤집으면 새 발견이다",
            "context_required": "맥락 기여는 세 구간 모두에서 지지되지 않아 제거했다. "
                                "다시 넣는 것도, 반대로 뒤집어 넣는 것도 새 발견이다",
            "validation_horizon": "이 평가 창에서 살아남은 주장이다. 평가 창을 바꾸면 그 주장이 "
                                  "아니다",
        },
    }


# ---- 2. 같은 family 의 다른 feature 는 같은 가설이 아니다 (§4) --------------------

def sibling_features(invariants: Mapping[str, Any]) -> list[str]:
    """검증한 관측과 **같은 것을 재는 이웃**. 몰래 갈아끼우기 가장 쉬운 자리다.

    family 만으로 고르지 않는다. `evidence.family_of` 의 마지막 줄은 조건 없는 fallback
    이라 어디에도 안 걸린 feature 가 전부 BOOK_IMBALANCE 로 떨어진다 — 최우선호가나
    중간가격까지 "같은 family" 라고 부르면 이유가 거짓이 된다.

    그래서 family 가 같고 **변별력 있는 태그를 공유하는 것**만 이웃으로 본다. 거의 모든
    feature 가 다는 태그(`order_book` 같은)는 아무것도 구별하지 못하므로 뺀다.
    """
    family, canonical = invariants["feature_family"], invariants["canonical_feature"]
    spec = catalog.FEATURES.get(canonical)
    if spec is None:
        return []
    tags = set(spec.tags) - _ubiquitous_tags()
    if not tags:
        return []
    return sorted(name for name, other in catalog.FEATURES.items()
                  if E.family_of(other) == family and name != canonical
                  and (tags & set(other.tags)))


def _ubiquitous_tags(threshold: float = 0.5) -> set[str]:
    """절반 넘는 feature 가 다는 태그. 붙어 있어도 무엇과도 구별해 주지 않는다."""
    total = len(catalog.FEATURES)
    counts: dict[str, int] = {}
    for spec in catalog.FEATURES.values():
        for tag in spec.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return {tag for tag, n in counts.items() if n > threshold * total}


def forbidden_features(invariants: Mapping[str, Any]) -> list[dict[str, str]]:
    """넣으면 다른 가설이 되는 feature 들과 그 이유 (§23~§24)."""
    out: list[dict[str, str]] = []
    rejected = invariants.get("rejected_context_feature")
    if rejected:
        out.append({"feature": rejected,
                    "family": invariants.get("rejected_context_family") or "",
                    "why": "검증에서 기여가 지지되지 않아 제거한 맥락이다. 다시 넣는 것도 "
                           "부등호를 뒤집어 넣는 것도 새 발견이다"})
    for name in sibling_features(invariants):
        out.append({"feature": name, "family": invariants["feature_family"],
                    "why": f"`{invariants['canonical_feature']}` 와 같은 것을 재는 "
                           "이웃이다. 비슷하다고 갈아끼우면 검증한 적 없는 것을 "
                           "실행하는 것이다"})
    # 핵심 family 에서 결정적으로 파생되는 것 — 다른 이름이지만 같은 양이다
    for family, info in sorted(E.DERIVED_FAMILIES.items()):
        if invariants["feature_family"] not in info.get("derived_from", ()):
            continue
        for name, spec in sorted(catalog.FEATURES.items()):
            if E.family_of(spec) != family:
                continue
            out.append({"feature": name, "family": family,
                        "why": f"`{invariants['feature_family']}` 에서 결정적으로 파생된다 "
                               f"({info['identity']}). 다른 이름이지만 같은 양이라 "
                               "새 증거가 아니다"})
    if not invariants.get("mechanism_identified"):
        out.append({"feature": "<mechanism proxy>", "family": "—",
                    "why": "메커니즘이 식별되지 않았다. 그것을 나타낸다며 임의의 feature 를 "
                           "조건에 더하면, 검증한 적 없는 것을 실행 조건으로 만드는 것이다"})
    return out


# ---- 3. 실행 신호의 최소 형태 (§8~§9) -------------------------------------------

def signal_template(invariants: Mapping[str, Any],
                    config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """가장 단순한 충실한 표현. **임계값을 여기서 고르지 않는다.**

    `contract.py` 가 이미 `UNRESOLVED:<name>` 자리를 안다. 그 자리를 그대로 쓴다 —
    명세가 새 표기법을 만들면 구현이 그것을 해석하면서 뜻이 갈린다.
    """
    if config.threshold_kind not in THRESHOLD_KINDS:
        raise ValueError(f"알 수 없는 threshold source: {config.threshold_kind!r}")
    # Grounding이 이미 Catalog DSL 식을 만들었다면 여기서 한 primitive 비교로
    # 다시 번역하지 않는다. 그렇게 하면 ``A AND B``가 ``A``가 되는, 가장 위험한
    # 의미 손실이 생긴다. 아래의 옛 단일-feature 경로는 Validation 산출물용이다.
    operational = invariants.get("operational_expression")
    if operational is not None:
        return SC.combine(invariants.get("executable_sample_condition"), operational)
    feature = invariants["canonical_feature"]
    name = f"{config.threshold_prefix}{feature}"
    source: dict[str, Any] = {"kind": config.threshold_kind}
    if config.threshold_grid:
        source["grid"] = [float(value) for value in config.threshold_grid]
    meaning = (
        "그 종목의 현재 장에서 **현재 tick을 뺀 직전 100 tick** 분포의 몇 분위부터 "
        "'높다' 고 부를 것인가"
        if config.threshold_kind == THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE else
        "그 종목의 **전일** 분포에서 몇 분위부터 '높다' 고 부를 것인가. 값이 달라져도 "
        "'높을수록 이후 실행 경로가 낫다' 는 주장은 그대로다"
        if config.threshold_kind == THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE else
        "이 상태를 '높다' 고 부를 고정 원값. 모든 종목·날짜에서 같은 값을 쓴다")
    candidate = {
        "op": "compare",
        # `primitive_id` 다. 실행기가 읽는 이름 그대로 쓴다 — 명세가 다른 키를 쓰면
        # 구현이 옮겨 적으면서 어긋난다.
        "input": {"op": "primitive", "primitive_id": feature},
        "comparator": COMPARATOR[invariants["direction"]],
        "value": f"{catalog.UNRESOLVED_PREFIX}{name}",
        "_parameter": {"name": name, "status": SEARCHABLE,
                       "threshold_source": source, "meaning": meaning},
    }
    return SC.combine(invariants.get("executable_sample_condition"), candidate)


# ---- 4. 언제 결정하는가 (§14~§16) ------------------------------------------------

def decision_opportunity_policy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """결정 기회는 **가설이 정하지 않는다.** 정본 백테스트 프로필이 정한다.

    가설마다 실행 방식이 다르면 성과 차이를 신호의 차이로 읽을 수 없다. 여기서는 그
    프로필이 무엇인지 적어 둘 뿐이고, 검증이 선 표본과 같은지만 확인한다.
    """
    canonical_validation = bool(manifest.get("canonical_backtest_validation"))
    cfg = manifest.get("config") or {}
    horizon = float(cfg.get("horizon_seconds") or 30.0)
    spacing = cfg.get("anchor_spacing_seconds") or horizon
    decided = K.CANONICAL["decision"]
    return {
        "status": EXECUTION_ASSUMPTION,
        "owner": "CANONICAL_BACKTEST_PROFILE",
        "profile_id": K.CANONICAL["profile_id"],
        "profile_sha256": K.CANONICAL["profile_sha256"],
        "validated_sampling": (
            "Validation Backtest도 Final Backtest와 같은 canonical Backtest profile을 쓴다"
            if canonical_validation else
            f"{spacing:g}초 시계 격자 위의, 미래를 쓰지 않고 고른 anchor"),
        "initial_binding": decided["basis"],
        "initial_spacing_seconds": decided["interval_seconds"],
        "backtest_sampling_differs": not canonical_validation,
        "why": ("Validation과 Final Backtest는 날짜만 다르고 같은 실행 규칙을 쓴다"
                if canonical_validation else
                "검증의 격자는 **창이 겹치지 않게 통계를 재려는 측정 설계**였다. "
                "백테스트는 '실제로 무엇을 할 수 있었나' 를 묻으므로 계속 본다. "
                "다른 질문이라 다른 표본을 쓰는 것이 맞다"),
        "not_a_search_parameter": True,
        "why_not_searchable":
            "결정 간격을 바꾸면 기회 수와 상관 구조가 함께 바뀐다. 그건 같은 가설의 "
            "파라미터가 아니라 다른 표본에서의 다른 실험이다",
        "tick_count_warning":
            f"틱 수로 간격을 정하지 않는다. 종목 간 호가 간격이 "
            f"{C.MEASURED['median_quote_interval_seconds']['spread_ratio']}배까지 "
            "벌어져 같은 틱 수가 같은 금융 시간이 아니다",
        "change_requires": "이 값을 바꾸려면 바꾼 표본에서 다시 검증해야 한다",
    }


def entry_execution_semantics(invariants: Mapping[str, Any]) -> dict[str, Any]:
    """체결 방식도 **가설이 정하지 않는다.** 정본 프로필이 정한다.

    검증은 `ASK1 → BID1` 기준으로 관계를 쟀다. 그것은 그 관계를 재는 자였고, 실제로
    주문이 어떻게 체결되는가는 별개다. 둘을 섞으면 가설마다 다른 장비로 재게 된다.
    """
    return {
        "status": EXECUTION_ASSUMPTION,
        "owner": "CANONICAL_BACKTEST_PROFILE",
        "profile_id": K.CANONICAL["profile_id"],
        "profile_sha256": K.CANONICAL["profile_sha256"],
        "fill_model": K.CANONICAL["entry"]["model"],
        "entry_reference": K.CANONICAL["entry"]["post_price"],
        "exit_reference": K.CANONICAL["exit"]["post_price"],
        "validation_reference": {
            "entry": invariants["entry_reference"],
            "exit": invariants["future_execution_reference"],
            "note": "검증이 관계를 잴 때 쓴 기준이다. 체결 모형이 아니다"},
        "spread_mode": K.CANONICAL["cost"]["spread_mode"],
        "cost_bps": K.CANONICAL["cost"]["explicit_cost_bps"],
        "why": "실행은 모든 가설에 대해 같아야 한다. 그래야 성과 차이를 신호의 차이로 "
               "읽을 수 있다",
        "not_a_search_parameter": True,
        "limit_entry_note":
            "큐 모델·체결 가정·비용은 프로토콜이다. 가설이나 탐색이 건드리지 않는다",
    }


def exit_policy(invariants: Mapping[str, Any]) -> dict[str, Any]:
    """S30 은 **검증 평가 창**이지 청산 규칙이 아니다 (§20~§22)."""
    return {
        "evaluation_horizon": invariants["validation_horizon"],
        "horizon_status": invariants.get("horizon_status", FIXED_BY_VALIDATION),
        "exit_rule_status": OUT_OF_SCOPE,
        "why": "이 평가 창에서 살아남은 주장이다. 평가는 같은 평가 창에서 한다",
        "not_yet_decided": ["정확한 청산 시점", "이익실현 폭", "손절 폭", "보유 기간"],
        "horizon_search_forbidden":
            "여러 평가 창을 돌려 가장 좋은 손익을 고르면, 검증한 주장이 아니라 그 탐색의 "
            "결과를 보고하는 것이 된다",
    }


# ---- 5. 탐색 경계 (§31~§32) -----------------------------------------------------

def _parameter_entries_from_expression(expression: Mapping[str, Any],
                                       config: SpecConfig) -> list[dict[str, Any]]:
    """AST의 unresolved threshold를 *그 AST에 붙은* 탐색 축으로 읽는다.

    한 parameter가 서로 다른 feature/방향을 뜻하면 Search가 값을 하나 골라도 뜻이
    정해지지 않는다. 그런 식은 이 단계에서 명시적으로 거절한다.
    """
    entries: dict[str, dict[str, Any]] = {}

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            if isinstance(node, list):
                for item in node:
                    visit(item)
            return
        op = str(node.get("op") or "")
        value = node.get("threshold" if op == "crossover" else "value")
        if op in {"compare", "crossover"} and isinstance(value, str) \
                and value.startswith(catalog.UNRESOLVED_PREFIX):
            name = value[len(catalog.UNRESOLVED_PREFIX):]
            source = node.get("input")
            features = sorted(_feature_tokens(source))
            if not name or not features:
                raise ValueError("Grounding threshold는 이름과 Catalog 입력을 모두 가져야 한다")
            comparator = str(node.get("direction") if op == "crossover"
                             else node.get("comparator") or "")
            direction = ({">": HIGHER, "<": LOWER, "above": HIGHER,
                          "below": LOWER}.get(comparator))
            if direction is None:
                raise ValueError(f"Grounding threshold `{name}`의 비교 방향을 읽을 수 없다")
            entry = {
                "name": name,
                "status": SEARCHABLE,
                "feature": features[0] if len(features) == 1 else " + ".join(features),
                "features": features,
                "direction": direction,
                "threshold_source": {"kind": config.threshold_kind},
                "meaning": "Grounding AST가 선언한 이 비교의 임계값. 식의 구조·방향은 고정한다",
            }
            if config.threshold_grid:
                entry["threshold_source"]["grid"] = [float(v) for v in config.threshold_grid]
            previous = entries.get(name)
            if previous and {k: previous[k] for k in ("features", "direction")} != \
                    {k: entry[k] for k in ("features", "direction")}:
                raise ValueError(f"Grounding parameter `{name}`가 서로 다른 비교를 뜻한다")
            entries[name] = entry
        for child in node.values():
            visit(child)

    visit(expression)
    declared = _unresolved_names(expression)
    if declared != set(entries):
        unknown = sorted(declared - set(entries))
        raise ValueError(f"비교 임계값이 아닌 unresolved parameter는 아직 지원하지 않는다: {unknown}")
    if not entries:
        raise ValueError("Grounding operational_expression에는 탐색할 unresolved threshold가 하나 이상 필요하다")
    return [entries[name] for name in sorted(entries)]


def search_boundary(invariants: Mapping[str, Any], template: Mapping[str, Any],
                    config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """값이 아니라 **탐색이 허용되는 축**을 낸다."""
    composite = invariants.get("operational_expression") is not None
    parameter = template.get("_parameter")
    forbidden_axes = [
        {"axis": "persistence / N ticks", "why": "이 가설은 지속을 검증하지 않았다. "
                                                 "N 을 탐색하면 없던 시간 구조를 만든다"},
        {"axis": "rolling window / velocity", "why": "검증한 것은 상태 수준이지 변화율이 "
                                                     "아니다"},
        {"axis": "AST 밖 추가 feature 조건", "why": "조건을 더하면 검증한 규칙이 아니다"},
        {"axis": "맥락 임계", "why": "그 맥락은 기여가 지지되지 않아 제거됐다"},
        {"axis": "스프레드 조건", "why": "가설에 없다. 넣으면 새 규칙이다"},
        {"axis": "평가 창", "why": "검증 의미가 달라진다"},
        {"axis": "방향", "why": "뒤집으면 다른 가설이다"},
        {"axis": "매매 방향(side)", "why": "가설이 정한 것이다"},
        {"axis": "결정 간격", "why": "검증이 선 표본이 달라진다. 실행 결속이지 파라미터가 "
                                    "아니다"},
        {"axis": "진입 체결 방식", "why": "위와 같다"},
    ]
    if composite:
        allowed_parameters = _parameter_entries_from_expression(template, config)
    else:
        assert parameter is not None
        allowed = {"name": parameter["name"], "status": SEARCHABLE,
                   "feature": invariants["canonical_feature"],
                   "features": [invariants["canonical_feature"]],
                   "direction": invariants["direction"], "meaning": parameter["meaning"]}
        if parameter.get("threshold_source"):
            allowed["threshold_source"] = dict(parameter["threshold_source"])
        allowed_parameters = [allowed]
    entry_features = sorted(_feature_tokens(template))
    return {
        "allowed_parameters": allowed_parameters,
        "forbidden_parameters": forbidden_axes,
        "fixed_semantics": {
            "canonical_feature": invariants["canonical_feature"],
            "direction": invariants["direction"],
            "side": invariants["side"],
            "evaluation_horizon": invariants["validation_horizon"],
            "entry_reference": invariants["entry_reference"],
            "context_required": len(entry_features) > 1,
            "entry_features": entry_features,
            "operational_expression": copy.deepcopy(dict(template)) if composite else None,
            "executable_sample_condition": copy.deepcopy(
                invariants.get("executable_sample_condition")),
        },
        "on_promising_forbidden_axis": SEARCH_DISCOVERY_CANDIDATE,
        "on_promising_note":
            "탐색 중 금지 축에서 좋은 결과가 나와도 이 가설의 결과로 저장하지 않는다. "
            "별도 후보로 남겨 Evidence 단계의 입력으로만 쓴다",
        "minimal_surface_note":
            "Grounding AST의 각 unresolved 임계만 연다. 그 밖의 축을 늘리면 '같은 가설' "
            "이라 부를 수 있는 범위가 흐려진다",
    }


# ---- 6. 명세가 가설과 같은 뜻인가 (§25~§29) --------------------------------------

def fidelity_audit(spec: Mapping[str, Any], revision: Mapping[str, Any],
                   results: Mapping[str, Any], config: SpecConfig = SpecConfig(), *,
                   expected_direction: str | None = None) -> dict[str, Any]:
    """여섯 질문. 코드가 본다 — 명세가 스스로 지켰다고 말하는 것과 다르다."""
    invariants = spec["semantic_invariants"]
    template = spec["signal_template"]
    boundary = spec["search_boundary"]
    revised = revision.get("revised_hypothesis") or (
        dict(revision) if revision.get("hypothesis_id")
        else dict((revision.get("hypotheses") or [{}])[0]))
    problems: list[dict[str, str]] = []

    def fail(question: str, why: str) -> None:
        problems.append({"question": question, "problem": why})

    grounded_expression = invariants.get("operational_expression")
    if grounded_expression is not None:
        # 복합식에서는 “대표 feature 하나”가 정본이 아니다. Grounding이 승인한 AST
        # 전체(연산자·관계·방향 포함)가 정본이다.
        expected_template = _public_expression(SC.combine(
            invariants.get("executable_sample_condition"), grounded_expression))
        actual_template = _public_expression(template)
        expected_features = _feature_tokens(expected_template)
        actual_features = _feature_tokens(actual_template)
        new_feature = len(actual_features - expected_features)
        if actual_features != expected_features:
            fail("Q1", "Grounding AST의 feature 집합과 명세 signal_template이 다르다")
        direction_change = 0
        if expected_template != actual_template:
            fail("Q2", "Grounding이 승인한 비교 방향·관계·구조와 명세 signal_template이 다르다")
            direction_change = 1
        expected_temporal = _ops(expected_template) & {"persistence", "sequence", "sequence_once", "rolling_sum",
                                                         "rolling_zscore", "crossover", "difference"}
        actual_temporal = _ops(actual_template) & {"persistence", "sequence", "sequence_once", "rolling_sum",
                                                     "rolling_zscore", "crossover", "difference"}
        new_relation = len(actual_temporal - expected_temporal)
        if new_relation:
            fail("Q5", f"Grounding AST에 없던 시간 구조를 더했다: {sorted(actual_temporal - expected_temporal)}")
    else:
        # Q1 — 검증한 feature 를 그대로 쓰는가. 검증이 시험한 것이 정본이다.
        _, validated = _core_feature(revised, invariants.get("validated_feature"))
        used = str((template.get("input") or {}).get("primitive_id"))
        new_feature = 0
        if used != validated:
            fail("Q1", f"검증한 것은 `{validated}` 인데 명세는 `{used}` 를 쓴다")
            new_feature += 1
        if used != invariants["canonical_feature"]:
            fail("Q1", "불변식과 신호 템플릿의 feature 가 다르다")
            new_feature += 1

        # Q2 — 방향이 유지되는가
        direction_change = 0
        expected = expected_direction or _direction(results, config)
        if invariants["direction"] != expected:
            source = "Grounding 예측" if expected_direction else "검증 통계"
            fail("Q2", f"{source}의 방향은 {expected} 인데 명세는 "
                       f"{invariants['direction']} 라고 적었다")
            direction_change += 1
        if template.get("comparator") != COMPARATOR[invariants["direction"]]:
            fail("Q2", f"방향 {invariants['direction']} 에 맞는 부등호는 "
                       f"{COMPARATOR[invariants['direction']]} 다")
            direction_change += 1
        ops = _ops(template)
        temporal = sorted(ops & {"persistence", "sequence", "sequence_once", "rolling_sum", "rolling_zscore",
                                 "crossover", "difference"})
        new_relation = len(temporal)
        if temporal:
            fail("Q5", f"가설에 없던 시간 구조를 더했다: {temporal}")

    # Q3 — 기각된 맥락이 다시 들어왔는가
    rejected = str(invariants.get("rejected_context_feature") or "")
    tokens = _feature_tokens(template)
    reintroduced = 0
    if rejected and rejected in tokens:
        fail("Q3", f"제거한 맥락 `{rejected}` 가 신호에 다시 들어왔다")
        reintroduced += 1

    # Q4 — 미식별 메커니즘을 임의 feature 로 대체했는가
    banned = {item["feature"] for item in spec.get("forbidden_features") or []}
    mechanism_proxy = sorted(tokens & banned - {rejected})
    if mechanism_proxy:
        fail("Q4", f"금지된 feature 가 신호에 들어왔다: {mechanism_proxy}")

    # Q6 — 검증 평가 창을 몰래 바꿨는가
    horizon_change = 0
    if spec["exit_policy"]["evaluation_horizon"] != invariants["validation_horizon"]:
        fail("Q6", "평가 창이 검증 평가 창과 다르다")
        horizon_change += 1

    # 승인하지 않은 탐색 축
    allowed = {item["name"] for item in boundary["allowed_parameters"]}
    declared = {node for node in _unresolved_names(template)}
    unapproved = sorted(declared - allowed)
    if unapproved:
        fail("BOUNDARY", f"탐색 축으로 승인되지 않은 자리가 있다: {unapproved}")

    # 아직 정하지 않아 넘어갈 수 없는 것 (§28)
    ambiguous: list[str] = []
    for key in ("decision_opportunity_policy", "entry_execution_semantics"):
        section = spec.get(key) or {}
        if not section.get("initial_binding") and not section.get("fill_model"):
            ambiguous.append(key)

    if problems:
        status = FIDELITY_VIOLATED
    elif ambiguous:
        status = FIDELITY_AMBIGUOUS
    else:
        status = FIDELITY_PRESERVED

    return {"schema": "specification_audit.v1", "created_at": now_utc(),
            "fidelity_status": status,
            "problems": problems,
            "ambiguous": ambiguous,
            # §46 — 전부 0 이어야 한다
            "new_feature_count": new_feature,
            "new_relation_count": new_relation,
            "new_mechanism_count": len(mechanism_proxy),
            "rejected_context_reintroduced_count": reintroduced,
            "direction_change_count": direction_change,
            "horizon_change_count": horizon_change,
            "unapproved_search_parameter_count": len(unapproved),
            "unapproved_search_parameters": unapproved,
            "declared_search_parameters": sorted(declared)}


def _public_expression(node: Any) -> Any:
    """계약 밖 메타데이터를 빼고 비교할 수 있는 DSL 표현으로 만든다."""
    if isinstance(node, Mapping):
        return {str(key): _public_expression(value) for key, value in node.items()
                if not str(key).startswith("_")}
    if isinstance(node, list):
        return [_public_expression(value) for value in node]
    return node


def _feature_tokens(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        if str(node.get("op")) == "primitive" and node.get("primitive_id"):
            out.add(str(node["primitive_id"]))
        for child in node.values():
            _feature_tokens(child, out)
    elif isinstance(node, list):
        for child in node:
            _feature_tokens(child, out)
    return out


def _ops(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        if node.get("op"):
            out.add(str(node["op"]))
        for child in node.values():
            _ops(child, out)
    elif isinstance(node, list):
        for child in node:
            _ops(child, out)
    return out


def _unresolved_names(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        for key, value in node.items():
            if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX):
                out.add(value[len(catalog.UNRESOLVED_PREFIX):])
            else:
                _unresolved_names(value, out)
    elif isinstance(node, list):
        for child in node:
            _unresolved_names(child, out)
    return out


# 구현이 명세와 같은가는 `implementation.py` 가 본다 — 계약을 만드는 쪽에 함께 둔다.
# 여기서 또 정의하면 같은 개념이 두 벌이 되고, 한쪽만 고쳐지는 날이 온다.


def readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    """다음 단계로 갈 수 있는가. 바로 파라미터 탐색으로 가지 않는다 (§42)."""
    status = audit["fidelity_status"]
    if status == FIDELITY_VIOLATED:
        return {"readiness": INVALID_SPECIFICATION,
                "why": "명세가 가설과 다른 것을 말한다. 구현으로 넘기지 않는다",
                "next": "명세를 고치거나, 정말 다른 것을 시험하려면 새 가설로 분기한다"}
    if status == FIDELITY_AMBIGUOUS:
        return {"readiness": BLOCKED_BY_FIDELITY_AMBIGUITY,
                "why": "실행 결속이 아직 정해지지 않았다",
                "next": "결정 간격·진입 체결을 명시한 뒤 다시 본다"}
    return {"readiness": READY_FOR_IMPLEMENTATION,
            "why": "의미가 보존됐다. 이제 코드로 옮길 수 있다",
            "next": f"구현 → `implementation_fidelity_check` 통과 → "
                    f"그때 `{READY_FOR_PARAMETER_SEARCH}` 로 올린다",
            "not_yet": "파라미터 탐색은 아직 시작하지 않는다"}


# ---- 8. 조립 -------------------------------------------------------------------

def build(revision: Mapping[str, Any], routing: Mapping[str, Any],
          manifest: Mapping[str, Any], results: Mapping[str, Any],
          config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """검증 산출물에서 실행 계약을 만든다. 값은 하나도 새로 고르지 않는다."""
    invariants = semantic_invariants(revision, routing, manifest, results, config)
    template = signal_template(invariants, config)
    revised = revision.get("revised_hypothesis") or revision
    target = (PT.require(revised, "Hypothesis revision")
              if revised.get("profit_target") is not None else PT.canonical_target())
    spec: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at": now_utc(),
        "hypothesis_id": invariants["hypothesis_id"],
        "hypothesis_status": invariants["hypothesis_status"],
        "evidence_tier": invariants["evidence_tier"],
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "semantic_invariants": invariants,
        "signal_template": template,
        "forbidden_features": forbidden_features(invariants),
        "decision_opportunity_policy": decision_opportunity_policy(manifest),
        "entry_execution_semantics": entry_execution_semantics(invariants),
        "exit_policy": exit_policy(invariants),
        "validation_semantics": {
            "horizon": invariants["validation_horizon"],
            "entry_reference": invariants["entry_reference"],
            "future_execution_reference": invariants["future_execution_reference"],
            "cost_bps": invariants["cost_bps"],
        },
        "provenance": {
            "hypothesis_hash": (manifest.get("revision") or {}).get(
                "revised_hypothesis_hash"),
            "revision_hash": (manifest.get("revision") or {}).get("revision_hash"),
            "grounding_hash": manifest.get("grounding_hash"),
            "validation_plan_hash": manifest.get("validation_plan_hash"),
            "catalog_hash": manifest.get("catalog_hash"),
            "capability_profile_hash": manifest.get("capability_profile_hash"),
        },
        "not_done_here": ["임계값 선택", "파라미터 탐색", "손익 최적화",
                          "청산 규칙", "포지션 크기", "terminal OOS 접근"],
    }
    spec["search_boundary"] = search_boundary(invariants, template, config)
    spec["specification_sha256"] = sha256_json(
        {k: v for k, v in spec.items() if k != "created_at"})[:16]
    return spec


def _grounded_prediction(grounded: Mapping[str, Any]) -> Mapping[str, Any]:
    """명세가 사용할 하나의 예측 주장. 결과가 아니라 Grounding의 입력이다."""
    claims = grounded.get("claims") or []
    graph_prediction = (grounded.get("mechanism_graph") or {}).get("prediction") or {}
    graph_target = graph_prediction.get("target_type")
    graph_direction = graph_prediction.get("direction")
    target_types = ((graph_target,) if graph_target in ("PATH_TARGET", "MATCHED_CONTROL_TARGET")
                    else ("PATH_TARGET", "MATCHED_CONTROL_TARGET"))
    for target_type in target_types:
        targets = [claim for claim in claims
                   if claim.get("grounding_status") == "VALIDATION_TARGET"
                   and claim.get("validation_target") == target_type
                   and claim.get("direction") in (HIGHER, LOWER)
                   and (graph_direction not in (HIGHER, LOWER)
                        or claim.get("direction") == graph_direction)]
        core = [claim for claim in targets if claim.get("importance") == "CORE"]
        if len(core) == 1:
            return core[0]
        if len(core) > 1:
            raise ValueError(f"실행 명세에는 CORE {target_type} 예측이 하나만 필요하다")
        if len(targets) == 1:
            return targets[0]
        if len(targets) > 1:
            raise ValueError(f"실행 명세에는 방향이 있는 {target_type} 예측이 하나만 필요하다")
    raise ValueError("실행 명세에는 방향이 있는 PATH_TARGET 또는 MATCHED_CONTROL_TARGET 예측이 필요하다")


def _grounded_lifecycle(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any], *,
                        expression_features: Sequence[str], prediction: Mapping[str, Any]
                        ) -> tuple[dict[str, Any], dict[str, str]]:
    """새 graph가 있으면 Grounding의 binding·pending 정책을 그대로 실행 명세에 옮긴다.

    graph가 없는 과거 artifact는 기존 HOLD_THROUGH 의미를 보존한다. 새 Agent 경로는
    Hypothesis validator가 graph를 요구하므로 이 호환 분기는 과거 재생 전용이다.
    """
    source = hypothesis.get("mechanism_graph")
    grounded_graph = grounded.get("mechanism_graph")
    if source is None and grounded_graph is None:
        return {"mode": "HOLD_THROUGH"}, {}
    if source is None or grounded_graph is None:
        raise ValueError("Hypothesis와 Grounding은 MechanismGraph를 함께 가져야 한다")
    evidence_ids = {str(item.get("evidence_id")) for item in hypothesis.get("evidence_basis") or []}
    report = M.validate_preservation(source, grounded_graph, known_evidence_ids=evidence_ids,
                                     known_features=catalog.FEATURES)
    if report["problems"]:
        raise ValueError("MechanismGraph 보존 실패: " + "; ".join(report["problems"]))
    bindings = grounded_graph.get("catalog_bindings") or []
    bound_features = {str(item.get("catalog_feature")) for item in bindings}
    if bound_features != set(expression_features):
        raise ValueError("Grounding operational_expression와 MechanismGraph binding feature가 다르다")
    graph_prediction = grounded_graph.get("prediction") or {}
    if (graph_prediction.get("target_type") != prediction.get("validation_target")
            or graph_prediction.get("direction") != prediction.get("direction")):
        raise ValueError("MechanismGraph prediction이 Grounding validation target과 다르다")
    lifecycle = dict((grounded_graph.get("execution") or {}).get("entry_lifecycle_policy") or {})
    return lifecycle, {
        "mechanism_graph_semantic_hash": str(report["source_semantic_hash"]),
        "grounded_mechanism_graph_hash": str(report["grounded_full_hash"]),
    }


def grounded_semantic_invariants(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any]
                                 ) -> dict[str, Any]:
    """아직 손익을 보지 않은 Grounding에서 탐색할 의미만 고정한다.

    옛 경로형 Validation은 여기서 쓰지 않는다. Grounding이 지정한 예측의 방향으로
    명세를 만들고, 그 명세와 parameter lock을 같은 canonical Backtest에서 검증한다.
    """
    sample_condition = SC.optional(hypothesis, "Hypothesis")
    operational = grounded.get("operational_expression")
    if not isinstance(operational, Mapping):
        raise ValueError("새 Grounding 계약에는 operational_expression가 필요하다")
    operational = _public_expression(operational)
    inferred = catalog.infer_expression_type(operational, allow_unresolved=True)
    if getattr(inferred, "value_type", inferred) != "boolean":
        raise ValueError("Grounding operational_expression는 bool Catalog DSL 식이어야 한다")
    expression_features = sorted(_feature_tokens(operational))
    if not expression_features:
        raise ValueError("Grounding operational_expression에 Catalog feature가 없다")

    prediction = _grounded_prediction(grounded)
    evidence_ids = {str(value) for value in prediction.get("supporting_evidence_ids") or []}
    basis = list(hypothesis.get("evidence_basis") or [])
    matching = [item for item in basis if str(item.get("evidence_id")) in evidence_ids]
    if len(matching) != 1:
        # 미래 수익 claim은 supporting evidence가 비어 있을 수 있다. 그때는 Grounding이
        # operational claim으로 고정한 직접 관측을 먼저 읽는다.
        operational_claim_ids = {
            str(value) for value in grounded.get("operational_claim_ids") or []
        }
        operational_features = {
            str(claim.get("catalog_feature"))
            for claim in grounded.get("claims") or []
            if str(claim.get("claim_id")) in operational_claim_ids
            and claim.get("grounding_status") in ("GROUNDED_DIRECT", "GROUNDED_DERIVED")
            and claim.get("catalog_feature")
        }
        operational_basis = [
            item for item in basis
            if str(item.get("representative_feature")) in operational_features
        ]
        if len(operational_basis) == 1:
            family, feature = (str(operational_basis[0].get("family")),
                               str(operational_basis[0].get("representative_feature")))
        else:
            # 옛 artifact는 operational claim 연결이 없으므로 기존 role 규칙을 쓴다.
            family, feature = _core_feature(hypothesis)
    else:
        family, feature = (str(matching[0].get("family")),
                           str(matching[0].get("representative_feature")))
    if feature not in catalog.FEATURES:
        raise ValueError(f"Grounding prediction의 핵심 feature가 Catalog에 없다: {feature}")
    if feature not in expression_features:
        raise ValueError("Grounding prediction의 핵심 관측이 operational_expression에 없다")
    signal_claims = [claim for claim in grounded.get("claims") or []
                     if claim.get("grounding_status") == "GROUNDED_DIRECT"
                     and claim.get("catalog_feature") in expression_features
                     and claim.get("direction") in (HIGHER, LOWER)]
    directions = {str(claim["catalog_feature"]): str(claim["direction"])
                  for claim in signal_claims}
    signal_direction = directions.get(feature, str(prediction["direction"]))
    entry_states = [{"feature": name, "direction": directions.get(name)}
                    for name in expression_features]
    lifecycle, graph_provenance = _grounded_lifecycle(
        hypothesis, grounded, expression_features=expression_features, prediction=prediction)
    seconds = float(K.CANONICAL["horizon"]["primary_seconds"])
    return {
        "hypothesis_id": str(hypothesis.get("hypothesis_id")),
        "hypothesis_status": "GROUNDED_FOR_PARAMETER_SEARCH",
        "evidence_tier": "GROUNDED_PREDICTION",
        "validated_feature": feature,
        "feature_family": family,
        "canonical_feature": feature,
        "entry_features": expression_features,
        "entry_states": entry_states,
        "entry_lifecycle_policy": lifecycle,
        "operational_expression": copy.deepcopy(operational),
        "executable_sample_condition": copy.deepcopy(sample_condition),
        "operational_claim_ids": [str(value) for value in
                                    (grounded.get("operational_claim_ids") or [])],
        "prediction_target": str(prediction.get("validation_target")),
        "prediction_direction": str(prediction.get("direction")),
        "prediction_measure": prediction.get("validation_measure"),
        "prediction_reference": prediction.get("validation_reference"),
        "role": "late observable state",
        "direction": signal_direction,
        "direction_status": FIXED_BY_HYPOTHESIS,
        "side": "LONG",
        "context_required": len(expression_features) > 1,
        "rejected_context_family": None,
        "rejected_context_feature": "",
        "mechanism_identified": False,
        "unresolved_mechanism": [],
        "removed_claims": [],
        "validation_horizon": f"S{int(seconds)}",
        "validation_horizon_seconds": seconds,
        "horizon_status": EXECUTION_ASSUMPTION,
        "entry_reference": str(K.CANONICAL["entry"]["post_price"]),
        "future_execution_reference": str(K.CANONICAL["exit"]["post_price"]),
        "cost_bps": K.CANONICAL["cost"]["explicit_cost_bps"],
        **({"temporal_warmup_ticks": SC.warmup_ticks(sample_condition)}
           if sample_condition is not None else {}),
        **graph_provenance,
        "why_fixed": {
            "canonical_feature": "Grounding prediction이 지정한 핵심 관측이다. 복합식 전체는 operational_expression에 고정한다",
            "direction": "대표 관측의 Grounding 방향이다. 복합식의 각 비교 방향은 operational_expression에 고정한다",
            "context_required": "Grounding AST에 둘 이상의 관측이 있어야만 참이다",
            "validation_horizon": "Validation과 Final Backtest는 같은 canonical Backtest profile의 보유 한도를 쓴다",
        },
    }


def build_from_grounding(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
                         package: Mapping[str, Any],
                         config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """Grounding → parameter-search Specification. 아직 Validation 결과는 입력하지 않는다."""
    target = PT.require(hypothesis, "Hypothesis")
    invariants = grounded_semantic_invariants(hypothesis, grounded)
    template = signal_template(invariants, config)
    # 동일 AST 안의 다른 feature는 "추가 feature"가 아니라 Grounding이 승인한
    # 조건이다. 금지목록에서 빼되, fidelity audit는 AST 전체 동일성으로 보호한다.
    expression_features = set(invariants.get("entry_features") or [])
    forbidden = [item for item in forbidden_features(invariants)
                 if item.get("feature") not in expression_features]
    spec: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at": now_utc(),
        "hypothesis_id": invariants["hypothesis_id"],
        "hypothesis_status": invariants["hypothesis_status"],
        "evidence_tier": invariants["evidence_tier"],
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "executable_sample_condition": copy.deepcopy(
            invariants.get("executable_sample_condition")),
        "executable_sample_condition_sha256": (
            SC.condition_hash(invariants["executable_sample_condition"])
            if invariants.get("executable_sample_condition") is not None else None),
        "semantic_invariants": invariants,
        "signal_template": template,
        "forbidden_features": forbidden,
        "decision_opportunity_policy": decision_opportunity_policy(
            {"canonical_backtest_validation": True}),
        "entry_execution_semantics": entry_execution_semantics(invariants),
        "entry_lifecycle_policy": dict(invariants["entry_lifecycle_policy"]),
        "exit_policy": exit_policy(invariants),
        "validation_semantics": {
            "horizon": invariants["validation_horizon"],
            "entry_reference": invariants["entry_reference"],
            "future_execution_reference": invariants["future_execution_reference"],
            "cost_bps": invariants["cost_bps"],
        },
        "provenance": {
            "hypothesis_hash": sha256_json(hypothesis),
            "profit_target_sha256": PT.target_hash(target),
            "grounding_hash": sha256_json(grounded),
            "catalog_hash": package.get("catalog_sha256"),
            **{key: invariants[key] for key in ("mechanism_graph_semantic_hash",
                                                "grounded_mechanism_graph_hash")
               if key in invariants},
        },
        "not_done_here": ["임계값 선택", "파라미터 탐색", "Validation Backtest",
                          "Final Backtest", "terminal OOS 접근"],
    }
    spec["search_boundary"] = search_boundary(invariants, template, config)
    spec["specification_sha256"] = sha256_json(
        {key: value for key, value in spec.items() if key != "created_at"})[:16]
    return spec


def _direct_evidence_expression(candidate: Mapping[str, Any], config: SpecConfig) -> dict[str, Any]:
    """Direct Evidence 후보가 이미 고정한 상태식만 Catalog DSL로 옮긴다."""
    states = list(candidate.get("entry_states") or [])
    if not states:
        raise ValueError("Direct Evidence 후보에 entry_states가 없다")
    temporal = candidate.get("temporal_entry") or {}

    def condition(state: Mapping[str, Any], *, event: bool = False) -> dict[str, Any]:
        feature = str(state.get("feature") or "")
        direction = str(state.get("direction") or "")
        if feature not in catalog.FEATURES or direction not in COMPARATOR:
            raise ValueError(f"Direct Evidence 상태가 유효하지 않다: {feature} {direction}")
        parameter = str(state.get("parameter") or f"{config.threshold_prefix}{feature}")
        source = dict(state.get("expression") or {"op": "primitive", "primitive_id": feature})
        metadata = {"name": parameter, "status": SEARCHABLE,
                    "threshold_source": {"kind": config.threshold_kind},
                    "meaning": "Direct Evidence 상태의 전일 분위 임계값"}
        if event:
            return {
                "op": "crossover",
                "input": source,
                "threshold": f"{catalog.UNRESOLVED_PREFIX}{parameter}",
                "direction": "above" if direction == HIGHER else "below",
                "equality": "strict",
                "_parameter": metadata,
            }
        return {
            "op": "compare",
            "input": source,
            "comparator": COMPARATOR[direction],
            "value": f"{catalog.UNRESOLVED_PREFIX}{parameter}",
            "_parameter": metadata,
        }

    operator = str(temporal.get("operator") or "")
    if not operator:
        expressions = [condition(state) for state in states]
        return expressions[0] if len(expressions) == 1 else {"op": "all", "args": expressions}
    if operator == "CROSSOVER":
        if len(states) != 1:
            raise ValueError("CROSSOVER Direct Evidence 후보는 상태 하나여야 한다")
        return condition(states[0], event=True)
    if operator == "DIFFERENCE":
        expressions = [condition(states[0])]
        expressions.extend(condition(state) for state in states[1:])
        return expressions[0] if len(expressions) == 1 else {"op": "all", "args": expressions}
    if operator in {"SEQUENCE", "SEQUENCE_ONCE_PER_CONTEXT_EPISODE"}:
        if len(states) != 2:
            raise ValueError("SEQUENCE Direct Evidence 후보는 context와 trigger 상태가 필요하다")
        return {
            "op": "sequence_once" if operator == "SEQUENCE_ONCE_PER_CONTEXT_EPISODE" else "sequence",
            "setup": condition(states[0]),
            "trigger": condition(states[1], event=True),
            "min_lag": int(float(temporal.get("min_lag_seconds", 0.0))),
            "max_lag": int(float(temporal.get("max_lag_seconds", 0.0))),
            "time_basis": str(temporal.get("time_basis") or "clock"),
        }
    raise ValueError(f"Direct Evidence 시간 연산자가 유효하지 않다: {operator}")


def direct_evidence_semantic_invariants(candidate: Mapping[str, Any],
                                         config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """Profile에서 직접 관측된 entry 상태를 실행 명세의 고정 의미로 만든다."""
    sample_condition = SC.optional(candidate, "Direct Evidence candidate")
    states = [dict(state) for state in candidate.get("entry_states") or []]
    expression = _direct_evidence_expression(candidate, config)
    primary = states[0]
    feature = str(primary["feature"])
    direction = str(primary["direction"])
    temporal = dict(candidate.get("temporal_entry") or {})
    seconds = float(K.CANONICAL["horizon"]["primary_seconds"])
    return {
        "hypothesis_id": str(candidate.get("hypothesis_id")),
        "hypothesis_status": "DIRECT_EVIDENCE_FOR_PARAMETER_SEARCH",
        "evidence_tier": "DIRECT_DISCOVERY_EVIDENCE",
        "validated_feature": feature,
        "feature_family": str(primary.get("family") or E.family_of(catalog.FEATURES[feature])),
        "canonical_feature": feature,
        "entry_features": sorted(str(state["feature"]) for state in states),
        "entry_states": states,
        "operational_expression": expression,
        "executable_sample_condition": copy.deepcopy(sample_condition),
        "role": "late observable state",
        "direction": direction,
        "direction_status": FIXED_BY_HYPOTHESIS,
        "side": "LONG",
        "context_required": len(states) > 1,
        "rejected_context_family": None,
        "rejected_context_feature": "",
        "mechanism_identified": False,
        "unresolved_mechanism": [],
        "removed_claims": [],
        "validation_horizon": f"S{int(seconds)}",
        "validation_horizon_seconds": seconds,
        "horizon_status": EXECUTION_ASSUMPTION,
        "entry_reference": str(K.CANONICAL["entry"]["post_price"]),
        "future_execution_reference": str(K.CANONICAL["exit"]["post_price"]),
        "cost_bps": K.CANONICAL["cost"]["explicit_cost_bps"],
        **({"temporal_warmup_ticks": max(
            SC.warmup_ticks(sample_condition), int(temporal.get("warmup_ticks", 0)))}
           if temporal or sample_condition is not None else {}),
        "why_fixed": {
            "canonical_feature": "Direct Evidence가 관측한 entry 상태다. 복합식 전체는 operational_expression에 고정한다",
            "direction": "Direct Evidence가 관측한 각 상태의 방향이다. 임계값만 Search가 고른다",
            "context_required": "직접 관측된 복합 상태가 둘 이상일 때만 참이다",
            "validation_horizon": "Validation과 Final Backtest는 같은 canonical Backtest profile의 보유 한도를 쓴다",
        },
    }


def build_from_direct_evidence(candidate: Mapping[str, Any], package: Mapping[str, Any],
                               config: SpecConfig = SpecConfig()) -> dict[str, Any]:
    """Direct Evidence 후보 → parameter-search Specification."""
    target = PT.require(candidate, "Direct Evidence candidate")
    invariants = direct_evidence_semantic_invariants(candidate, config)
    template = signal_template(invariants, config)
    expression_features = set(invariants["entry_features"])
    forbidden = [item for item in forbidden_features(invariants)
                 if item.get("feature") not in expression_features]
    spec: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at": now_utc(),
        "hypothesis_id": invariants["hypothesis_id"],
        "hypothesis_status": invariants["hypothesis_status"],
        "evidence_tier": invariants["evidence_tier"],
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "executable_sample_condition": copy.deepcopy(
            invariants.get("executable_sample_condition")),
        "executable_sample_condition_sha256": (
            SC.condition_hash(invariants["executable_sample_condition"])
            if invariants.get("executable_sample_condition") is not None else None),
        "semantic_invariants": invariants,
        "signal_template": template,
        "forbidden_features": forbidden,
        "decision_opportunity_policy": decision_opportunity_policy(
            {"canonical_backtest_validation": True}),
        "entry_execution_semantics": entry_execution_semantics(invariants),
        "entry_lifecycle_policy": {"mode": str(candidate.get("entry_lifecycle_mode") or "HOLD_THROUGH")},
        "exit_policy": exit_policy(invariants),
        "validation_semantics": {
            "horizon": invariants["validation_horizon"],
            "entry_reference": invariants["entry_reference"],
            "future_execution_reference": invariants["future_execution_reference"],
            "cost_bps": invariants["cost_bps"],
        },
        "provenance": {
            "direct_evidence_candidate_hash": sha256_json(candidate),
            "profit_target_sha256": PT.target_hash(target),
            "catalog_hash": package.get("catalog_sha256"),
        },
        "not_done_here": ["임계값 선택", "파라미터 탐색", "Validation Backtest",
                          "Final Backtest", "terminal OOS 접근"],
    }
    spec["search_boundary"] = search_boundary(invariants, template, config)
    spec["specification_sha256"] = sha256_json(
        {key: value for key, value in spec.items() if key != "created_at"})[:16]
    return spec


# ---- 9. 사람이 읽을 형태 --------------------------------------------------------

def parameter_table(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """§44 의 표. 항목마다 상태와 **이유**를 붙인다."""
    inv = spec["semantic_invariants"]
    why = inv["why_fixed"]
    composite = inv.get("operational_expression") is not None
    feature_label = ", ".join(f"`{feature}`" for feature in inv.get("entry_features") or
                              [inv["canonical_feature"]])
    rows = [
        {"항목": "Grounding 입력 AST" if composite else feature_label, "값": "사용",
         "상태": FIXED_BY_HYPOTHESIS, "이유": why["canonical_feature"]},
        {"항목": "비교 방향·연산자" if composite else f"방향 {inv['direction']}",
         "값": "AST에 고정" if composite else COMPARATOR[inv["direction"]],
         "상태": inv.get("direction_status", FIXED_BY_VALIDATION), "이유": why["direction"]},
        {"항목": f"매매 방향 {inv['side']}", "값": inv["side"],
         "상태": FIXED_BY_HYPOTHESIS, "이유": "가설이 정한 방향이다"},
        {"항목": f"`{inv['rejected_context_feature']}` 맥락", "값": "미사용",
         "상태": FORBIDDEN, "이유": why["context_required"]},
        {"항목": "지속(N틱)", "값": "미사용", "상태": FORBIDDEN,
         "이유": "이 가설은 지속을 검증하지 않았다"},
        {"항목": f"평가 창 {inv['validation_horizon']}", "값": "유지",
         "상태": inv.get("horizon_status", FIXED_BY_VALIDATION), "이유": why["validation_horizon"]},
        {"항목": f"진입 {inv['entry_reference']}", "값": "즉시 실행",
         "상태": EXECUTION_ASSUMPTION,
         "이유": "검증 결과가 이 기준 위에서 계산됐다"},
        {"항목": "결정 간격",
         "값": str(spec["decision_opportunity_policy"]["initial_binding"]),
         "상태": EXECUTION_ASSUMPTION,
         "이유": spec["decision_opportunity_policy"]["why_not_searchable"]},
        {"항목": "청산 규칙", "값": "미정", "상태": OUT_OF_SCOPE,
         "이유": "이 단계가 정할 일이 아니다"},
    ]
    thresholds = [{"항목": f"임계 `{item['name']}`", "값": "미정",
                   "상태": SEARCHABLE, "이유": item["meaning"]}
                  for item in spec["search_boundary"]["allowed_parameters"]]
    rows[3:3] = thresholds
    return rows


def to_markdown(spec: Mapping[str, Any], audit: Mapping[str, Any],
                environment: str = "") -> str:
    inv = spec["semantic_invariants"]
    boundary = spec["search_boundary"]
    L = [f"# 실행 명세 — {spec['hypothesis_id']}", "",
         "검증을 통과한 가설을 **의미를 바꾸지 않고** 실행 계약으로 옮긴다. 코드를 짜는 "
         "단계의 시작이지만, 더 정확히는 **코드와 탐색이 가설을 몰래 바꾸지 못하도록 "
         "경계를 먼저 잠그는 단계**다.", ""]
    if environment:
        L += [environment, "---", ""]
    L += ["## 0. 답해야 하는 한 문장", "",
          "> 파라미터 탐색에서 아무리 좋은 결과가 나와도, **어디까지 바뀌면 여전히 "
          f"{spec['hypothesis_id']} 이고 어디부터는 새 가설인가?**", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 입력 가설 | `{spec['hypothesis_id']}` |",
          f"| 검증 상태 | `{spec['hypothesis_status']}` |",
          f"| 증거 강도 | `{spec['evidence_tier']}` |",
          f"| 명세 해시 | `{spec['specification_sha256']}` |",
          f"| 충실도 | **`{audit['fidelity_status']}`** |", "",
          "## 1. 무엇이 고정이고 무엇만 탐색인가", "",
          "| 항목 | 값 | 상태 | 이유 |", "|---|---|---|---|"]
    for row in parameter_table(spec):
        L.append(f"| {row['항목']} | {row['값']} | `{row['상태']}` | {row['이유']} |")

    L += ["", "## 2. 실행 신호", "",
          "임계값을 여기서 고르지 않는다. **자리만 선언한다.**", "", "```json",
          _pretty(spec["signal_template"]), "```", "",
          f"`{catalog.UNRESOLVED_PREFIX}...` 자리는 `contract.py` 가 이미 아는 표기다. "
          "명세가 새 표기법을 만들면 구현이 그것을 해석하면서 뜻이 갈린다.", "",
          "## 3. 탐색 경계", "",
          "### 열린 축", "", "| 파라미터 | feature | 뜻 |", "|---|---|---|"]
    for item in boundary["allowed_parameters"]:
        L.append(f"| `{item['name']}` | `{item['feature']}` | {item['meaning']} |")
    L += ["", "### 닫힌 축", "", "| 축 | 왜 |", "|---|---|"]
    for item in boundary["forbidden_parameters"]:
        L.append(f"| {item['axis']} | {item['why']} |")
    L += ["", f"탐색 중 닫힌 축에서 좋은 결과가 나오면 `{SEARCH_DISCOVERY_CANDIDATE}` 로 "
          "남긴다. 이 가설의 결과로 저장하지 않는다.", "",
          "## 4. 넣으면 다른 가설이 되는 feature", "",
          "| feature | family | 왜 |", "|---|---|---|"]
    for item in spec["forbidden_features"]:
        L.append(f"| `{item['feature']}` | {item['family']} | {item['why']} |")

    decision = spec["decision_opportunity_policy"]
    entry = spec["entry_execution_semantics"]
    L += ["", "## 5. 실행 결속 — 정해야 하지만 탐색에 넘기지 않는 것", "",
          "이게 이 명세의 핵심이다. 진입 시점 같은 것을 '파라미터' 로 두면 검증이 서 있던 "
          "표본 규칙이 조용히 달라진다.", "", "| 항목 | 값 | 왜 탐색이 아닌가 |",
          "|---|---|---|",
          f"| 결정 기회 | {decision['initial_binding']} | "
          f"{decision['why_not_searchable']} |",
          f"| 진입 체결 | {entry['fill_model']} @ {entry['entry_reference']} | "
          f"{entry['limit_entry_note']} |", "",
          f"- 검증 표본: {decision['validated_sampling']}",
          f"- {decision['tick_count_warning']}",
          f"- 바꾸려면: {decision['change_requires']}", "",
          "## 6. 청산과 평가 창", "",
          f"`{inv['validation_horizon']}` 은 **검증 평가 창**이지 청산 규칙이 아니다.", "",
          f"- 아직 정하지 않은 것: {', '.join(spec['exit_policy']['not_yet_decided'])}",
          f"- {spec['exit_policy']['horizon_search_forbidden']}", "",
          "## 7. 충실도 검사", "", "| 지표 | 값 |", "|---|---|"]
    for key in ("new_feature_count", "new_relation_count", "new_mechanism_count",
                "rejected_context_reintroduced_count", "direction_change_count",
                "horizon_change_count", "unapproved_search_parameter_count"):
        L.append(f"| `{key}` | **{audit[key]}** |")
    L += ["", f"- 판정 `{audit['fidelity_status']}`",
          f"- 문제 `{audit['problems'] or '없음'}`", ""]
    if inv["unresolved_mechanism"]:
        L += ["## 8. 여전히 모르는 것", "",
              "메커니즘이 식별되지 않았다. 그것을 나타낸다며 임의의 feature 를 조건에 "
              "더하지 않는다.", ""]
        L += [f"- {item}" for item in inv["unresolved_mechanism"]] + [""]
    L += ["## 9. 이 단계에서 하지 않은 것", ""]
    L += [f"- {item}" for item in spec["not_done_here"]] + [""]
    return "\n".join(L)


def _pretty(node: Any, indent: int = 0) -> str:
    import json
    return json.dumps({k: v for k, v in node.items() if not k.startswith("_")},
                      ensure_ascii=False, indent=1)


# ---- 10. 배선 ------------------------------------------------------------------

def run(validation_dir: Path, revision_path: Path, output: Path, *,
        config: SpecConfig = SpecConfig(), environment: str = "") -> dict[str, Any]:
    """검증 산출물 → 실행 명세. 산출물 4개 + 온보딩 CSV."""
    import pandas as pd
    from . import validation as V

    validation_dir, output = Path(validation_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(validation_dir / "validation_freeze_manifest.json")
    routing = read_json(validation_dir / "hypothesis_routing.json")
    revision = read_json(Path(revision_path))
    frame = pd.read_parquet(validation_dir / "mechanism_validation_results.parquet")
    results = V.results_from_frame(frame, routing["primary_split"])
    # 결정 규칙이 쓰는 귀무값을 계약에서 그대로 읽는다
    contracts = (manifest.get("validation_plan") or {}).get("target_contracts") \
        or V.TARGET_CONTRACTS
    for validation_id, row in results.items():
        row.setdefault("null_value", (contracts.get(validation_id) or {}).get("null"))

    spec = build(revision, routing, manifest, results, config)
    audit = fidelity_audit(spec, revision, results, config)
    ready = readiness(audit)
    spec["readiness"] = ready

    write_json(output / "executable_specification.json", spec)
    write_json(output / "search_boundary.json", spec["search_boundary"])
    write_json(output / "specification_audit.json", audit)
    (output / "executable_specification.md").write_text(
        to_markdown(spec, audit, environment), encoding="utf-8")
    _companions(spec, audit, output)
    return {"output": str(output), "spec": spec, "audit": audit, "readiness": ready}


def run_from_grounding(hypothesis: Mapping[str, Any], grounded: Mapping[str, Any],
                       package: Mapping[str, Any], output: Path, *,
                       config: SpecConfig = SpecConfig(), environment: str = "") -> dict[str, Any]:
    """Grounding에서 명세를 만들되, 옛 경로형 Validation 결과는 읽지 않는다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "source_hypothesis.json", hypothesis)
    write_json(output / "source_grounding.json", grounded)
    spec = build_from_grounding(hypothesis, grounded, package, config)
    audit = fidelity_audit(spec, hypothesis, {}, config,
                           expected_direction=spec["semantic_invariants"]["direction"])
    ready = readiness(audit)
    spec["readiness"] = ready
    write_json(output / "executable_specification.json", spec)
    write_json(output / "search_boundary.json", spec["search_boundary"])
    write_json(output / "specification_audit.json", audit)
    (output / "executable_specification.md").write_text(
        to_markdown(spec, audit, environment), encoding="utf-8")
    _companions(spec, audit, output)
    return {"output": str(output), "spec": spec, "audit": audit, "readiness": ready}


def run_from_direct_evidence(candidate: Mapping[str, Any], package: Mapping[str, Any],
                             output: Path, *, config: SpecConfig = SpecConfig(),
                             environment: str = "") -> dict[str, Any]:
    """Direct Evidence 후보에서 명세를 만들고, 같은 구현 전 계약을 남긴다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "source_direct_evidence_candidate.json", candidate)
    spec = build_from_direct_evidence(candidate, package, config)
    audit = fidelity_audit(spec, candidate, {}, config,
                           expected_direction=spec["semantic_invariants"]["direction"])
    ready = readiness(audit)
    spec["readiness"] = ready
    write_json(output / "executable_specification.json", spec)
    write_json(output / "search_boundary.json", spec["search_boundary"])
    write_json(output / "specification_audit.json", audit)
    (output / "executable_specification.md").write_text(
        to_markdown(spec, audit, environment), encoding="utf-8")
    _companions(spec, audit, output)
    return {"output": str(output), "spec": spec, "audit": audit, "readiness": ready}


def _companions(spec: Mapping[str, Any], audit: Mapping[str, Any], output: Path) -> None:
    """보고서와 별개로 두는, 표로 열어 훑을 결과물."""
    import pandas as pd

    inv = spec["semantic_invariants"]
    pd.DataFrame([{"항목": k, "값": v} for k, v in {
        "가설": spec["hypothesis_id"],
        "검증상태": spec["hypothesis_status"],
        "증거강도": spec["evidence_tier"],
        "핵심관측": inv["canonical_feature"],
        "family": inv["feature_family"],
        "방향": inv["direction"],
        "매매방향": inv["side"],
        "맥락필요": inv["context_required"],
        "메커니즘식별": inv["mechanism_identified"],
        "평가창": inv["validation_horizon"],
        "진입기준": inv["entry_reference"],
        "청산기준": inv["future_execution_reference"],
        "열린탐색축": len(spec["search_boundary"]["allowed_parameters"]),
        "닫힌탐색축": len(spec["search_boundary"]["forbidden_parameters"]),
        "충실도": audit["fidelity_status"],
        "준비도": spec["readiness"]["readiness"],
        "명세해시": spec["specification_sha256"],
    }.items()]).to_csv(output / "summary.csv", index=False)

    pd.DataFrame(parameter_table(spec)).to_csv(
        output / "semantic_invariants.csv", index=False)

    pd.DataFrame([{"파라미터": item["name"], "상태": item["status"],
                   "feature": item["feature"], "뜻": item["meaning"]}
                  for item in spec["search_boundary"]["allowed_parameters"]]
                 ).to_csv(output / "searchable_parameters.csv", index=False)

    rows = [{"종류": "탐색 축", "대상": item["axis"], "왜": item["why"]}
            for item in spec["search_boundary"]["forbidden_parameters"]]
    rows += [{"종류": "feature", "대상": item["feature"], "왜": item["why"]}
             for item in spec["forbidden_features"]]
    pd.DataFrame(rows).to_csv(output / "forbidden_transformations.csv", index=False)

    meaning = {
        "new_feature_count": "검증하지 않은 feature 를 신호에 넣었다",
        "new_relation_count": "가설에 없던 시간 구조를 더했다",
        "new_mechanism_count": "미식별 메커니즘을 임의 feature 로 대체했다",
        "rejected_context_reintroduced_count": "제거한 맥락이 다시 들어왔다",
        "direction_change_count": "검증이 지켜낸 방향과 다르다",
        "horizon_change_count": "검증 평가 창을 몰래 바꿨다",
        "unapproved_search_parameter_count": "승인하지 않은 자리를 탐색 축으로 열었다",
    }
    pd.DataFrame([{"지표": k, "값": audit[k], "기준": "0 이어야 함",
                   "무엇을_잡는가": v} for k, v in meaning.items()]
                 ).to_csv(output / "audit.csv", index=False)
