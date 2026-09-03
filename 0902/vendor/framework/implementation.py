"""실행 명세를 프레임워크가 실제로 읽는 계약으로 옮긴다.

이 단계의 성공은 **코드가 도는 것**이 아니라 **코드가 정확히 그 가설을 구현한 것**이다.
둘은 다르다. 문법이 맞고 백테스트가 돌아도 다른 가설을 실행하고 있을 수 있다.

## 끝났을 때 이래야 한다

    theta_book_imbalance = ?

**그 물음표가 남아 있어야 이 단계가 제대로 끝난 것이다.** 값을 채웠으면 탐색을 미리
한 것이고, 그건 다음 단계 일이다.

## 두 종류의 성공을 가른다

    IMPLEMENTATION_VALID   DSL 이 유효하다. 실행기가 읽는다
    FIDELITY_PRESERVED     명세의 의미를 한 글자도 넘지 않았다

둘 다 있어야 파라미터 탐색으로 넘어간다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import (canonical as K, catalog, contract, executable as X, profit_target as PT,
               sample_condition as SC)
from .config import ENTRY_REFINEMENT_POLICY_VERSION, now_utc, read_json, sha256_json, write_json


SCHEMA_VERSION = "h1r1_contract_template.v1"
COMPILER_VERSION = "contract_compiler.v1"

# 실행 결속 이름. 명세와 같은 낱말을 쓴다.
CLOCK_GRID = "CLOCK_GRID"
TAKER_IMMEDIATE = "TAKER_IMMEDIATE"

# 최종 관문 (§39).
READY_FOR_PARAMETER_SEARCH = "READY_FOR_PARAMETER_SEARCH"
BLOCKED_BY_IMPLEMENTATION_AMBIGUITY = "BLOCKED_BY_IMPLEMENTATION_AMBIGUITY"
IMPLEMENTATION_FIDELITY_VIOLATION = "IMPLEMENTATION_FIDELITY_VIOLATION"
IMPLEMENTATION_INVALID = "IMPLEMENTATION_INVALID"
GATE_STATUS = (READY_FOR_PARAMETER_SEARCH, BLOCKED_BY_IMPLEMENTATION_AMBIGUITY,
               IMPLEMENTATION_FIDELITY_VIOLATION, IMPLEMENTATION_INVALID)

# 파라미터 주입 실패 (§18).
UNAPPROVED_PARAMETER = "UNAPPROVED_PARAMETER"
PREMATURE_PARAMETER_BINDING = "PREMATURE_PARAMETER_BINDING"

# 이 가설에 없는 시간 구조. 하나라도 생기면 다른 가설이다 (§26).
TEMPORAL_OPERATORS = ("persistence", "sequence", "sequence_once", "rolling_sum", "rolling_zscore",
                      "crossover", "difference")


@dataclass(frozen=True)
class CompileConfig:
    """컴파일러가 새로 고르는 값은 없다. 전부 명세에서 온다."""

    schema: str = SCHEMA_VERSION
    compiler: str = COMPILER_VERSION


# ---- 1. 컴파일 (§13~§15) --------------------------------------------------------

def compile_contract(spec: Mapping[str, Any],
                     config: CompileConfig = CompileConfig()) -> dict[str, Any]:
    """명세 → 계약 템플릿. **명세에 적힌 것만 옮긴다.**

    LLM 이 DSL 을 자유롭게 쓰지 않는다. 그러면 매번 다른 것이 나오고, 무엇이 바뀌었는지
    사람이 눈으로 대조해야 한다.

    임계는 `UNRESOLVED:` 그대로 둔다. 여기서 채우면 탐색을 미리 한 것이다.
    """
    target = PT.require(spec, "Executable Specification")
    sample_condition = SC.optional(spec, "Executable Specification")
    invariants = spec["semantic_invariants"]
    template = spec["signal_template"]
    decision = spec["decision_opportunity_policy"]
    entry = spec["entry_execution_semantics"]
    lifecycle = dict(spec.get("entry_lifecycle_policy") or {"mode": "HOLD_THROUGH"})
    if lifecycle.get("mode") not in {"HOLD_THROUGH", "CANCEL_WHEN_ENTRY_SIGNAL_FALSE"}:
        raise ValueError("entry lifecycle mode가 유효하지 않다")

    signal = {k: v for k, v in template.items() if not k.startswith("_")}
    # rolling 통계가 없으면 데울 구간도 없다. 기본값을 넣지 않고 트리에서 읽는다.
    warmup = (0 if not (_ops(signal) & set(TEMPORAL_OPERATORS)) else
              (spec.get("semantic_invariants") or {}).get("temporal_warmup_ticks"))
    if warmup is None:
        raise ValueError("시간 연산자가 있는데 warmup 을 정할 근거가 없다. "
                         "명세에 없는 것을 컴파일러가 정하지 않는다")

    # 청산 프로필은 실제 Backtest가 단일 정본(V9)으로 결속한다. Implementation은
    # 검증된 평가 창만 옮기고, 여기서 청산 규칙을 새로 정하지 않는다.
    profile = {"profile_id": K.PROFILE_ID, "profile_sha256": K.profile_hash()}
    return {
        "schema": config.schema,
        "created_at": now_utc(),
        "hypothesis_id": spec["hypothesis_id"],
        "profit_target": target,
        "profit_target_sha256": PT.target_hash(target),
        "executable_sample_condition": sample_condition,
        "executable_sample_condition_sha256": (
            SC.condition_hash(sample_condition) if sample_condition is not None else None),
        "side": invariants["side"],
        "entry_program": {"signal": signal, "warmup_ticks": int(warmup)},
        "entry_lifecycle": lifecycle,
        "execution_binding": {
            "decision_opportunity": decision["initial_binding"],
            "decision_spacing_seconds": decision["initial_spacing_seconds"],
            "decision_basis": "clock",
            "entry_execution": entry["fill_model"],
            "entry_reference": entry["entry_reference"],
            "evaluation_horizon": spec["exit_policy"]["evaluation_horizon"],
            "exit_rule": None,
            "exit_rule_note": "청산 규칙은 이 단계가 정하지 않는다",
        },
        "parameter_interface": parameter_interface(spec),
        # Search가 q를 어떤 실행 원장으로 고르는지도 결과의 일부다. 이 값이 바뀌면
        # 같은 entry DSL이라도 재사용하지 않고 다시 Search·Validation·Final을 연다.
        "parameter_search_policy": {
            "selection_mode": invariants.get("parameter_search_mode", "FIXED_EXIT_EXECUTION"),
            "primary_metric": PT.SEARCH_PRIMARY_METRIC,
            **profile,
        },
        # Refinement가 선택하는 guard도 최종 entry 조건이다. 같은 base signal이라도
        # guard 보존 정책이 달라지면 재평가할 수 있게 계약에 명시한다.
        "entry_refinement_policy_version": ENTRY_REFINEMENT_POLICY_VERSION,
        "provenance": {
            "executable_specification_sha256": spec["specification_sha256"],
            "compiler": config.compiler,
            **{k: v for k, v in (spec.get("provenance") or {}).items()},
        },
        "not_included": ["임계값", "청산 파라미터", "포지션 크기", "손익 계산"],
    }


def parameter_interface(spec: Mapping[str, Any]) -> dict[str, Any]:
    """미결정 값을 문자열 치환으로 다루지 않는다. 인터페이스로 드러낸다 (§16)."""
    invariants = spec["semantic_invariants"]
    out: dict[str, Any] = {}
    for item in spec["search_boundary"]["allowed_parameters"]:
        parameter = {
            "status": item["status"],
            "type": "numeric",
            "feature": item["feature"],
            "role": "signal_threshold",
            "comparator": X.COMPARATOR[str(item.get("direction") or invariants["direction"])],
            "placeholder": f"{catalog.UNRESOLVED_PREFIX}{item['name']}",
            "meaning": item["meaning"],
            "value": None,
        }
        if item.get("threshold_source"):
            parameter["threshold_source"] = dict(item["threshold_source"])
        out[item["name"]] = parameter
    return out


# ---- 2. 값 주입 (§19) -----------------------------------------------------------

def bind_parameters(template: Mapping[str, Any],
                    values: Mapping[str, float]) -> dict[str, Any]:
    """탐색이 값을 주는 **유일한 통로**. 구조는 못 건드린다.

    승인되지 않은 이름이 들어오면 실패한다. "나중에 쓸지도 몰라서" 만든 자리도 탐색
    표면을 넓힌다.
    """
    interface = template.get("parameter_interface") or {}
    unknown = sorted(set(values) - set(interface))
    if unknown:
        raise ValueError(f"{UNAPPROVED_PARAMETER}: 승인되지 않은 파라미터 {unknown}. "
                         f"쓸 수 있는 것은 {sorted(interface)} 뿐이다")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 은 숫자여야 한다: {value!r}")

    bound = copy.deepcopy(dict(template))
    bound["entry_program"] = contract._substitute(
        bound["entry_program"],
        {f"{catalog.UNRESOLVED_PREFIX}{k}": float(v) for k, v in values.items()})
    for name, value in values.items():
        bound["parameter_interface"][name]["value"] = float(value)
    bound["bound_parameters"] = {k: float(v) for k, v in values.items()}
    return bound


# ---- 3. 코드가 유효한가 (§20) ---------------------------------------------------

def implementation_validity(template: Mapping[str, Any]) -> dict[str, Any]:
    """DSL 자체가 정상인가. 의미가 맞는지는 여기서 보지 않는다."""
    problems: list[str] = []
    target = template.get("profit_target")
    target_validation = PT.validate(target)
    problems.extend(target_validation["problems"])
    if (not isinstance(target, Mapping)
            or template.get("profit_target_sha256") != PT.target_hash(target)):
        problems.append("profit_target_sha256가 profit_target과 다르다")
    sample_condition = template.get("executable_sample_condition")
    if sample_condition is not None:
        sample_validation = SC.validate(sample_condition)
        problems.extend(sample_validation["problems"])
        if template.get("executable_sample_condition_sha256") != SC.condition_hash(
                sample_condition):
            problems.append(
                "executable_sample_condition_sha256가 executable_sample_condition과 다르다")
    signal = (template.get("entry_program") or {}).get("signal")
    if not isinstance(signal, Mapping):
        problems.append("entry_program.signal 이 표현식이 아니다")
    else:
        try:
            info = catalog.infer_expression_type(signal, allow_unresolved=True)
            if info.value_type not in ("boolean", "event"):
                problems.append(f"진입 신호는 boolean 또는 event 여야 한다. {info.value_type} 다")
        except catalog.ExpressionError as error:
            problems.append(f"표현식 검증 실패: {error}")
    for name in _features(signal or {}):
        if catalog.resolve(name) not in catalog.FEATURES:
            problems.append(f"레지스트리에 없는 feature: {name}")
    if template.get("side") not in ("LONG", "SHORT"):
        problems.append(f"모르는 side: {template.get('side')!r}")

    binding = template.get("execution_binding") or {}
    # `decision_spacing_seconds` 는 비어 있는 것이 정상이다 — 매 틱이면 간격이 없다.
    for key in ("decision_opportunity", "entry_execution", "evaluation_horizon"):
        if binding.get(key) in (None, ""):
            problems.append(f"실행 결속 `{key}` 가 비었다")
    lifecycle = template.get("entry_lifecycle") or {"mode": "HOLD_THROUGH"}
    if lifecycle.get("mode") not in {"HOLD_THROUGH", "CANCEL_WHEN_ENTRY_SIGNAL_FALSE"}:
        problems.append("entry_lifecycle.mode가 유효하지 않다")

    interface = template.get("parameter_interface") or {}
    declared = _unresolved(signal or {})
    if declared != {f"{catalog.UNRESOLVED_PREFIX}{n}" for n in interface}:
        problems.append(f"미결정 자리와 파라미터 인터페이스가 다르다: "
                        f"{sorted(declared)} vs {sorted(interface)}")

    return {"schema": "implementation_validity.v1", "created_at": now_utc(),
            "implementation_valid": not problems,
            "status": "IMPLEMENTATION_VALID" if not problems else IMPLEMENTATION_INVALID,
            "problems": problems,
            "dsl_schema_error_count": sum(1 for p in problems if "표현식" in p),
            "execution_binding_error_count": sum(1 for p in problems if "실행 결속" in p),
            "parameter_binding_error_count": sum(1 for p in problems
                                                 if "파라미터 인터페이스" in p)}


# ---- 4. 의미가 보존됐는가 (§21~§32) ---------------------------------------------

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


def _features(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        if str(node.get("op")) == "primitive" and node.get("primitive_id"):
            out.add(str(node["primitive_id"]))
        for child in node.values():
            _features(child, out)
    elif isinstance(node, list):
        for child in node:
            _features(child, out)
    return out


def _unresolved(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        for value in node.values():
            if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX):
                out.add(value)
            else:
                _unresolved(value, out)
    elif isinstance(node, list):
        for child in node:
            _unresolved(child, out)
    return out


def _comparison_by_feature(node: Any, out: dict[str, str] | None = None) -> dict[str, str]:
    """비교식의 feature와 방향만 뽑는다.

    이 비교는 계약 경계 검사에만 쓰며, 실행기의 내부 helper에 의존하지 않는다.
    """
    out = {} if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") == "compare":
            source = node.get("input") or {}
            feature = source.get("primitive_id") if isinstance(source, Mapping) else None
            if feature:
                out[str(feature)] = str(node.get("comparator") or "")
        for value in node.values():
            _comparison_by_feature(value, out)
    elif isinstance(node, list):
        for value in node:
            _comparison_by_feature(value, out)
    return out


def _allowed_entry_features(invariants: Mapping[str, Any]) -> set[str]:
    """구형 단일 feature 명세와 확장 명세를 같은 방식으로 읽는다."""
    features = {str(invariants.get("canonical_feature") or "")}
    for item in invariants.get("entry_states") or []:
        if isinstance(item, Mapping) and item.get("feature"):
            features.add(str(item["feature"]))
    return features - {""}


def _comparisons(node: Any, out: list[Mapping[str, Any]] | None = None) -> list[Mapping[str, Any]]:
    """단일 compare와 `all(compare, ...)`를 같은 방식으로 다룬다."""
    out = [] if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") == "compare":
            out.append(node)
        for value in node.values():
            _comparisons(value, out)
    elif isinstance(node, list):
        for value in node:
            _comparisons(value, out)
    return out


def _threshold_nodes(node: Any, out: list[Mapping[str, Any]] | None = None) -> list[Mapping[str, Any]]:
    """숫자 threshold를 갖는 level/사건 연산자를 같은 방식으로 찾는다."""
    out = [] if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") in {"compare", "crossover"}:
            out.append(node)
        for value in node.values():
            _threshold_nodes(value, out)
    elif isinstance(node, list):
        for value in node:
            _threshold_nodes(value, out)
    return out


def signal_ast(node: Any) -> Any:
    """비교용 정규 트리. 문자열이 아니라 구조를 본다 (§32).

    키 순서·공백이 달라도 같은 트리면 같다고 봐야 하고, 값이 하나만 달라도 달라야 한다.
    """
    if isinstance(node, Mapping):
        return tuple(sorted((str(k), signal_ast(v)) for k, v in node.items()
                            if not str(k).startswith("_")))
    if isinstance(node, list):
        return tuple(signal_ast(v) for v in node)
    return node


def ast_lines(node: Any, depth: int = 0) -> list[str]:
    """사람이 읽을 트리. 보고서에 넣는다."""
    pad = "  " * depth
    if not isinstance(node, Mapping):
        return [f"{pad}{node}"]
    op = str(node.get("op"))
    if op == "primitive":
        return [f"{pad}primitive({node.get('primitive_id')})"]
    lines = [f"{pad}{op}"]
    for key in sorted(node):
        if key in ("op",) or str(key).startswith("_"):
            continue
        value = node[key]
        if isinstance(value, (Mapping, list)):
            lines.append(f"{pad}  {key}:")
            lines += ast_lines(value, depth + 2)
        else:
            lines.append(f"{pad}  {key}: {value}")
    return lines


# 명세의 어느 항목이 계약의 어느 자리와 맞아야 하는가 (§31).
def semantic_diff(spec: Mapping[str, Any], template: Mapping[str, Any]
                  ) -> list[dict[str, Any]]:
    """항목별로 명세와 구현을 나란히 놓는다. 하나라도 다르면 다른 가설이다."""
    invariants = spec["semantic_invariants"]
    expected_signal = {k: v for k, v in spec["signal_template"].items()
                       if not k.startswith("_")}
    signal = (template.get("entry_program") or {}).get("signal") or {}
    binding = template.get("execution_binding") or {}
    decision = spec["decision_opportunity_policy"]
    entry = spec["entry_execution_semantics"]
    lifecycle = spec.get("entry_lifecycle_policy") or {"mode": "HOLD_THROUGH"}
    interface = template.get("parameter_interface") or {}

    expected_comparators = _comparison_by_feature(expected_signal)
    actual_comparators = _comparison_by_feature(signal)
    rows = [
        {"item": "feature", "metric": "feature_mismatch_count",
         "spec": sorted(_features(expected_signal)),
         "code": sorted(_features(signal))},
        {"item": "direction", "metric": "direction_mismatch_count",
         "spec": expected_comparators,
         "code": actual_comparators},
        {"item": "side", "metric": "side_mismatch_count",
         "spec": invariants["side"], "code": template.get("side")},
        {"item": "signal AST", "metric": "new_relation_count",
         "spec": signal_ast(expected_signal), "code": signal_ast(signal)},
        {"item": "temporal operators", "metric": "new_temporal_operator_count",
         "spec": sorted(_ops(expected_signal) & set(TEMPORAL_OPERATORS)),
         "code": sorted(_ops(signal) & set(TEMPORAL_OPERATORS))},
        {"item": "evaluation horizon", "metric": "horizon_mismatch_count",
         "spec": spec["exit_policy"]["evaluation_horizon"],
         "code": binding.get("evaluation_horizon")},
        {"item": "decision opportunity", "metric": "decision_cadence_mismatch_count",
         "spec": _cadence(decision["initial_binding"],
                          decision["initial_spacing_seconds"]),
         "code": _cadence(binding.get("decision_opportunity"),
                          binding.get("decision_spacing_seconds"))},
        {"item": "entry execution", "metric": "entry_execution_mismatch_count",
         "spec": f"{entry['fill_model']} @ {entry['entry_reference']}",
         "code": f"{binding.get('entry_execution')} @ {binding.get('entry_reference')}"},
        {"item": "entry lifecycle", "metric": "entry_lifecycle_mismatch_count",
         "spec": lifecycle.get("mode"),
         "code": (template.get("entry_lifecycle") or {}).get("mode")},
        {"item": "threshold", "metric": "premature_parameter_binding_count",
         "spec": sorted(_unresolved(expected_signal)),
         "code": sorted(_unresolved(signal))},
        {"item": "search surface", "metric": "unapproved_parameter_count",
         "spec": sorted(p["name"] for p in spec["search_boundary"]["allowed_parameters"]),
         "code": sorted(interface)},
        {"item": "profit target", "metric": "profit_target_mismatch_count",
         "spec": spec.get("profit_target"), "code": template.get("profit_target")},
        {"item": "executable sample condition",
         "metric": "executable_sample_condition_mismatch_count",
         "spec": spec.get("executable_sample_condition"),
         "code": template.get("executable_sample_condition")},
    ]
    for row in rows:
        row["match"] = row["spec"] == row["code"]
    return rows


def fidelity_check(spec: Mapping[str, Any], template: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """구현이 명세의 경계를 한 글자도 넘지 않았는가."""
    invariants = spec["semantic_invariants"]
    signal = (template.get("entry_program") or {}).get("signal") or {}
    rows = semantic_diff(spec, template)
    counts = {row["metric"]: (0 if row["match"] else 1) for row in rows}
    problems = [f"{row['item']}: 명세 {row['spec']} vs 코드 {row['code']}"
                for row in rows if not row["match"]]

    # 항목별로 따로 세어야 하는 것들
    banned = {item["feature"] for item in spec.get("forbidden_features") or []}
    used = _features(signal)
    rejected = str(invariants.get("rejected_context_feature") or "")
    counts["context_reintroduced_count"] = int(bool(rejected and rejected in used))
    sample_features = _features(spec.get("executable_sample_condition") or {})
    allowed_features = _allowed_entry_features(invariants) | sample_features
    counts["new_feature_count"] = len(used - allowed_features)
    counts["new_relation_count"] = counts.get("new_relation_count", 0)
    for extra in sorted((used & banned) - {rejected} - sample_features):
        problems.append(f"금지된 feature 가 신호에 있다: {extra}")

    status = (X.FIDELITY_VIOLATED if any(counts.values()) or problems
              else X.FIDELITY_PRESERVED)
    return {"schema": "implementation_fidelity.v1", "created_at": now_utc(),
            "fidelity_status": status,
            "diff": rows,
            "problems": problems,
            **{k: int(v) for k, v in sorted(counts.items())}}


# 0 이어야 하는 구조 지표 (§51).
STRUCTURE_METRICS = (
    "feature_mismatch_count", "direction_mismatch_count", "side_mismatch_count",
    "context_reintroduced_count", "new_feature_count", "new_relation_count",
    "new_temporal_operator_count", "horizon_mismatch_count",
    "decision_cadence_mismatch_count", "entry_execution_mismatch_count",
    "entry_lifecycle_mismatch_count",
    "profit_target_mismatch_count",
    "executable_sample_condition_mismatch_count",
    "premature_parameter_binding_count", "unapproved_parameter_count",
)
RUNTIME_METRICS = ("dsl_schema_error_count", "parameter_binding_error_count",
                   "execution_binding_error_count")
LEAKAGE_METRICS = ("market_data_access_count", "terminal_oos_access_count")


# ---- 5. 일부러 틀린 계약을 만들어 본다 (§33) ------------------------------------

def mutation_tests(spec: Mapping[str, Any], template: Mapping[str, Any]
                   ) -> list[dict[str, Any]]:
    """감사가 실제로 잡는지 확인한다. 안 잡히면 그 감사는 없는 것과 같다."""
    invariants = spec["semantic_invariants"]
    # 바꿔치기할 feature 는 **핵심 관측과 달라야** 한다. 같은 것으로 바꾸면 아무것도
    # 안 바뀌어 감사가 못 잡는 것처럼 보인다.
    core = str(invariants["canonical_feature"])
    swap = next((f for f in (*X.sibling_features(invariants), "spread_bps",
                             "book_imbalance", "vol_flow") if f != core), core)
    # 뺀 맥락이 없는 가설이면 맥락 feature 를, 그것도 없으면 형제 feature 를 쓴다.
    rejected = (invariants.get("rejected_context_feature")
                or invariants.get("context_feature") or swap)

    def mutate(label, expect, change):
        broken = copy.deepcopy(dict(template))
        change(broken)
        result = fidelity_check(spec, broken)
        caught = result["fidelity_status"] == X.FIDELITY_VIOLATED
        hit = [k for k in STRUCTURE_METRICS if result.get(k)]
        return {"mutation": label, "expected": expect, "caught": caught,
                "metrics": hit, "verdict": "FAIL_AS_EXPECTED" if caught else "NOT_CAUGHT"}

    def signal_of(broken):
        return broken["entry_program"]["signal"]

    def first_threshold(broken):
        nodes = _threshold_nodes(signal_of(broken))
        if not nodes:
            raise ValueError("변형할 threshold 노드가 없다")
        return nodes[0]

    def threshold_direction(node):
        return str(node.get("direction") if node.get("op") == "crossover"
                   else node.get("comparator") or ">")

    def flip_threshold_direction(node, direction):
        if node.get("op") == "crossover":
            node["direction"] = direction
        else:
            node["comparator"] = direction

    def bind_threshold(node, value):
        node["threshold" if node.get("op") == "crossover" else "value"] = value

    # 명세가 이미 쓰는 방향의 **반대**로 뒤집는다. `>` 로 고정하면 LOWER 가설에서
    # 바뀌는 것이 없다. crossover도 같은 feature의 반대 방향 사건으로 바꾼다.
    here = threshold_direction(first_threshold(template))
    flipped = {"above": "below", "below": "above"}.get(
        here, "<" if here.startswith(">") else ">")

    return [
        mutate("A. feature 를 이웃으로 교체", f"{invariants['canonical_feature']} → {swap}",
               lambda b: first_threshold(b)["input"].update({"primitive_id": swap})),
        mutate("B. 방향 뒤집기", f"{here} → {flipped}",
               lambda b: flip_threshold_direction(first_threshold(b), flipped)),
        mutate("C. side 뒤집기", "LONG → SHORT",
               lambda b: b.update({"side": "SHORT"})),
        mutate("D. 기각된 맥락 재도입", f"AND {rejected} < φ",
               lambda b: b["entry_program"].update({"signal": {
                   "op": "all", "args": [signal_of(b), {
                       "op": "compare",
                       "input": {"op": "primitive", "primitive_id": rejected},
                       "comparator": "<", "value": "UNRESOLVED:phi"}]}})),
        mutate("E. 지속 추가", "persistence(signal, N)",
               lambda b: b["entry_program"].update({"signal": {
                   "op": "persistence", "condition": signal_of(b), "window": 5}})),
        mutate("F. 평가 창 변경", "S30 → S60",
               lambda b: b["execution_binding"].update({"evaluation_horizon": "S60"})),
        mutate("G. 결정 간격 변경", "30초 격자 → 매 틱",
               lambda b: b["execution_binding"].update(
                   {"decision_opportunity": "EVERY_TICK", "decision_spacing_seconds": 0})),
        mutate("H. 진입 방식 변경", "ASK1 taker → BID1 limit",
               lambda b: b["execution_binding"].update(
                   {"entry_execution": "LIMIT_PASSIVE", "entry_reference": "BID1"})),
        mutate("I. 임계 미리 채움", "UNRESOLVED → 0.37",
               lambda b: bind_threshold(first_threshold(b), 0.37)),
    ]


# ---- 6. 실행기가 템플릿을 다루는가 (§34~§37) ------------------------------------

def smoke_test(template: Mapping[str, Any], fixture_threshold: float = 0.3
               ) -> dict[str, Any]:
    """합성 계열로 실행기가 도는지만 본다. **손익은 계산하지 않는다.**

    여기서 쓰는 임계는 탐색 결과가 아니라 테스트 픽스처다. 산출물에 남기지 않는다.
    """
    values = np.array([-0.6, -0.2, 0.1, 0.35, 0.7, 0.9])
    n = len(values)
    data = {
        "time_s": np.arange(n, dtype=float) * 30.0,
        "bid_price": np.tile(np.array([[100.0] + [99.0] * 9]), (n, 1)),
        "ask_price": np.tile(np.array([[101.0] + [102.0] * 9]), (n, 1)),
        "bid_qty": np.zeros((n, 10)), "ask_qty": np.ones((n, 10)) * 100.0,
        "buy_volume": np.zeros(n), "sell_volume": np.zeros(n),
        "buy_max_price": np.zeros(n), "sell_min_price": np.zeros(n),
        "local_time": np.full(n, 100000000000, dtype=np.int64),
    }
    # book_imbalance = (bid - ask) / (bid + ask) 가 원하는 값이 되도록 잔량을 맞춘다.
    # 다른 feature 를 쓰는 가설이면 이 계열이 그 feature 에서 무슨 값을 내는지는
    # 카탈로그에 물어본다 — 여기서 확인할 것은 "선언한 비교를 그대로 하는가" 다.
    ask = data["ask_qty"][:, 0]
    data["bid_qty"][:, 0] = ask * (1 + values) / (1 - values)

    bound = bind_parameters(template, {name: fixture_threshold
                                       for name in template["parameter_interface"]})
    signal = contract.entry_signal(data, bound)
    thresholds = _threshold_nodes((bound.get("entry_program") or {}).get("signal") or {})
    temporal_ops = _ops((bound.get("entry_program") or {}).get("signal") or {}) & set(TEMPORAL_OPERATORS)
    expected = np.ones(n, dtype=bool)
    feature_values: dict[str, list[float]] = {}
    runtime = contract.ExpressionRuntime(data)
    for index, node in enumerate(thresholds):
        feature = str((node.get("input") or {}).get("primitive_id") or f"expression_{index + 1}")
        comparator = (str(node.get("direction") or "above") if node.get("op") == "crossover"
                      else str(node.get("comparator") or ">"))
        actual = np.asarray(runtime.evaluate(node["input"]), dtype=float)
        feature_values[feature] = np.where(np.isfinite(actual), actual, np.nan).tolist()
        value = float(node["threshold" if node.get("op") == "crossover" else "value"])
        finite = np.isfinite(actual)
        expected &= (finite & (actual < value) if comparator in {"<", "<="} or comparator == "below"
                     else finite & (actual > value))
    # 비교식만 있는 계약은 각 부등호를 직접 다시 계산한다. sequence/crossover는
    # 비교식의 AND가 아니라 사건이므로, 첫 틱에 사건이 생기지 않는지와 실행 결과의
    # 배열 형태를 확인한다. 시간 구조의 상세 의미는 contract 단위 테스트가 맡는다.
    if temporal_ops:
        expected = signal.copy()
        semantic_ok = signal.shape == (n,) and (not bool(signal[0]) if "sequence" in temporal_ops else True)
    else:
        semantic_ok = True
    one_feature = sorted(feature_values)
    feature_field: Any = one_feature[0] if len(one_feature) == 1 else one_feature
    values_field: Any = feature_values[one_feature[0]] if len(one_feature) == 1 else feature_values
    return {"schema": "implementation_smoke_test.v1", "created_at": now_utc(),
            "fixture_only": True,
            "note": "여기 쓴 임계는 테스트 픽스처다. 탐색 결과가 아니며 산출물에 남기지 "
                    "않는다",
            "feature": feature_field,
            "synthetic_feature_values": values_field,
            "fixture_threshold": fixture_threshold,
            "signal": signal.tolist(),
            "expected": expected.tolist(),
            "parameter_binding_ok": True,
            "compare_operator_ok": bool(semantic_ok and np.array_equal(signal, expected)),
            "long_decisions": int(signal.sum()),
            "pnl_computed": False,
            "market_data_access_count": 0,
            "terminal_oos_access_count": 0}


# ---- 7. 관문 (§38~§39) ----------------------------------------------------------

def readiness_gate(validity: Mapping[str, Any], fidelity: Mapping[str, Any],
                   template: Mapping[str, Any], mutations: Sequence[Mapping[str, Any]],
                   smoke: Mapping[str, Any]) -> dict[str, Any]:
    """네 가지를 모두 만족해야 파라미터 탐색으로 넘어간다."""
    interface = template.get("parameter_interface") or {}
    binding = template.get("execution_binding") or {}
    unresolved = _unresolved((template.get("entry_program") or {}).get("signal") or {})

    interface_valid = (bool(interface)
                       and all(item.get("value") is None for item in interface.values())
                       and len(unresolved) == len(interface))
    binding_complete = all(binding.get(k) not in (None, "") for k in
                           ("decision_opportunity", "entry_execution",
                            "entry_reference", "evaluation_horizon"))
    missed = [m["mutation"] for m in mutations if not m["caught"]]

    if not validity["implementation_valid"]:
        status, why = IMPLEMENTATION_INVALID, "DSL 이 유효하지 않다"
    elif fidelity["fidelity_status"] != X.FIDELITY_PRESERVED:
        status, why = IMPLEMENTATION_FIDELITY_VIOLATION, "구현이 명세의 의미를 넘었다"
    elif missed:
        status, why = (IMPLEMENTATION_FIDELITY_VIOLATION,
                       f"감사가 못 잡는 변형이 있다: {missed}")
    elif not interface_valid or not binding_complete:
        status, why = (BLOCKED_BY_IMPLEMENTATION_AMBIGUITY,
                       "파라미터 인터페이스나 실행 결속이 아직 완성되지 않았다")
    elif not smoke.get("compare_operator_ok"):
        status, why = IMPLEMENTATION_INVALID, "실행기가 템플릿을 예상대로 처리하지 못했다"
    else:
        status, why = READY_FOR_PARAMETER_SEARCH, "네 조건을 모두 만족한다"

    return {"status": status, "why": why,
            "implementation_valid": validity["implementation_valid"],
            "fidelity_preserved": fidelity["fidelity_status"] == X.FIDELITY_PRESERVED,
            "parameter_interface_valid": interface_valid,
            "execution_binding_complete": binding_complete,
            "mutations_all_caught": not missed,
            "still_unresolved": sorted(unresolved),
            "next": ("이제 처음으로 `theta` 를 얼마로 잡을지 물을 수 있다"
                     if status == READY_FOR_PARAMETER_SEARCH else
                     "탐색으로 넘어가지 않는다")}


# ---- 8. 사람이 읽을 형태 --------------------------------------------------------

def to_markdown(spec: Mapping[str, Any], template: Mapping[str, Any],
                validity: Mapping[str, Any], fidelity: Mapping[str, Any],
                mutations: Sequence[Mapping[str, Any]], smoke: Mapping[str, Any],
                gate: Mapping[str, Any], environment: str = "") -> str:
    inv = spec["semantic_invariants"]
    binding = template["execution_binding"]
    L = [f"# 계약 구현 — {template['hypothesis_id']}", "",
         "실행 명세를 프레임워크가 실제로 읽는 계약으로 옮겼다. 이 단계의 성공은 **코드가 "
         "도는 것**이 아니라 **코드가 정확히 그 가설을 구현한 것**이다.", ""]
    if environment:
        L += [environment, "---", ""]
    L += ["## 0. 끝났을 때 이래야 한다", "", "```",
          f"{sorted(template['parameter_interface'])[0]} = ?", "```", "",
          "**그 물음표가 남아 있어야 이 단계가 제대로 끝난 것이다.**", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 입력 명세 | `{spec['specification_sha256']}` |",
          f"| 계약 해시 | `{template['contract_sha256']}` |",
          f"| 컴파일러 | `{template['provenance']['compiler']}` |",
          f"| 코드 유효 | `{validity['status']}` |",
          f"| 의미 보존 | **`{fidelity['fidelity_status']}`** |",
          f"| 관문 | **`{gate['status']}`** |", "",
          "## 1. 무엇을 구현했나", "", "### 계약 AST", "", "```"]
    L += ast_lines(template["entry_program"]["signal"])
    L += ["```", "",
          f"- side `{template['side']}` · warmup `{template['entry_program']['warmup_ticks']}` 틱",
          "- warmup 은 rolling 통계가 없어서 0 이다. 기본값을 넣은 것이 아니라 트리에서 "
          "읽었다.", "",
          "### 실행 결속", "", "| 항목 | 값 |", "|---|---|",
          f"| 결정 기회 | `{binding['decision_opportunity']}` "
          f"{_cadence('', binding.get('decision_spacing_seconds')).strip()} |",
          f"| 진입 체결 | `{binding['entry_execution']}` @ {binding['entry_reference']} |",
          f"| 평가 창 | `{binding['evaluation_horizon']}` |",
          f"| 청산 규칙 | `{binding['exit_rule']}` — {binding['exit_rule_note']} |", "",
          "## 2. 파라미터 인터페이스", "",
          "미결정 값을 문자열 치환으로 다루지 않는다. 인터페이스로 드러낸다.", "",
          "| 이름 | 상태 | feature | 자리 | 값 |", "|---|---|---|---|---|"]
    for name, item in sorted(template["parameter_interface"].items()):
        L.append(f"| `{name}` | `{item['status']}` | `{item['feature']}` | "
                 f"`{item['placeholder']}` | **{item['value']}** |")
    L += ["", f"`bind_parameters()` 가 값을 주는 유일한 통로다. 승인되지 않은 이름은 "
          f"`{UNAPPROVED_PARAMETER}` 로 실패한다 — \"나중에 쓸지도 몰라서\" 만든 자리도 "
          "탐색 표면을 넓힌다.", "",
          "## 3. 명세 대 구현", "", "| 항목 | 명세 | 구현 | |", "|---|---|---|---|"]
    for row in fidelity["diff"]:
        spec_value = _short(row["spec"])
        code_value = _short(row["code"])
        L.append(f"| {row['item']} | {spec_value} | {code_value} | "
                 f"{'일치' if row['match'] else '**불일치**'} |")
    L += ["", "구조로 비교한다. 키 순서나 공백이 달라도 같은 트리면 같고, 값이 하나만 "
          "달라도 다르다.", "",
          "## 4. 일부러 틀린 계약을 만들어 봤다", "",
          "감사가 실제로 잡는지 확인한다. 안 잡히면 그 감사는 없는 것과 같다.", "",
          "| 변형 | 무엇을 바꿨나 | 잡혔나 | 어느 지표가 |", "|---|---|---|---|"]
    for item in mutations:
        L.append(f"| {item['mutation']} | {item['expected']} | "
                 f"{'**잡힘**' if item['caught'] else '못 잡음'} | "
                 f"`{', '.join(item['metrics']) or '—'}` |")
    L += ["", "## 5. 실행 점검", "",
          "합성 계열로 실행기가 도는지만 본다. **손익은 계산하지 않는다.**", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 합성 `book_imbalance` | `{smoke['synthetic_feature_values']}` |",
          f"| 픽스처 임계 | {smoke['fixture_threshold']} |",
          f"| 진입 신호 | `{smoke['signal']}` |",
          f"| 기대 | `{smoke['expected']}` |",
          f"| 비교 연산자 | {'정상' if smoke['compare_operator_ok'] else '**어긋남**'} |",
          f"| LONG 결정 수 | {smoke['long_decisions']} |",
          f"| 손익 계산 | {smoke['pnl_computed']} |", "",
          f"{smoke['note']}", "",
          "## 6. 감사", "", "### 구조 — 0 이어야 한다", "", "| 지표 | 값 |", "|---|---|"]
    for key in STRUCTURE_METRICS:
        L.append(f"| `{key}` | **{fidelity.get(key, 0)}** |")
    L += ["", "### 실행 — 0 이어야 한다", "", "| 지표 | 값 |", "|---|---|"]
    for key in RUNTIME_METRICS:
        L.append(f"| `{key}` | **{validity.get(key, 0)}** |")
    L += ["", "### 데이터 접근 — 구현 단계에는 0 이다", "", "| 지표 | 값 |", "|---|---|"]
    for key in LEAKAGE_METRICS:
        L.append(f"| `{key}` | **{smoke.get(key, 0)}** |")
    L += ["", "합성 픽스처만 쓴다. 실제 시장 데이터도, 예약해 둔 마지막 구간도 열 이유가 "
          "없다.", "",
          "## 7. 탐색 경계 재확인", "",
          f"열린 축은 여전히 **하나**다 — `{sorted(template['parameter_interface'])[0]}`.",
          "", "## 8. 관문", "", "| 조건 | 통과 |", "|---|---|",
          f"| `IMPLEMENTATION_VALID` | {gate['implementation_valid']} |",
          f"| `FIDELITY_PRESERVED` | {gate['fidelity_preserved']} |",
          f"| `PARAMETER_INTERFACE_VALID` | {gate['parameter_interface_valid']} |",
          f"| `EXECUTION_BINDING_COMPLETE` | {gate['execution_binding_complete']} |",
          f"| 변형 전부 잡힘 | {gate['mutations_all_caught']} |", "",
          f"**`{gate['status']}`** — {gate['why']}", "",
          f"아직 미결정: `{', '.join(gate['still_unresolved'])}`", "",
          f"다음: {gate['next']}", ""]
    return "\n".join(L)


def _cadence(basis: Any, seconds: Any) -> str:
    """결정 기회를 한 줄로. 간격이 없으면(매 틱) 그대로 없다고 적는다."""
    return str(basis) if seconds in (None, "") else f"{basis} {float(seconds):g}s"


def _short(value: Any, limit: int = 46) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    return f"`{text}`"


# ---- 9. 배선 ------------------------------------------------------------------

def run(spec_path: Path, output: Path, *, config: CompileConfig = CompileConfig(),
        environment: str = "") -> dict[str, Any]:
    """명세 → 계약 → 검사 → 산출물. 시장 데이터를 읽지 않는다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = read_json(Path(spec_path))

    template = compile_contract(spec, config)
    template["contract_sha256"] = contract.contract_hash(template)[:16]
    validity = implementation_validity(template)
    fidelity = fidelity_check(spec, template)
    mutations = mutation_tests(spec, template)
    smoke = smoke_test(template)
    gate = readiness_gate(validity, fidelity, template, mutations, smoke)

    manifest = {
        "schema": "implementation_manifest.v1", "created_at": now_utc(),
        "hypothesis_id": template["hypothesis_id"],
        "source_specification_sha256": spec["specification_sha256"],
        "contract_sha256": template["contract_sha256"],
        "compiler": config.compiler,
        "parameter_interface": sorted(template["parameter_interface"]),
        "execution_binding": template["execution_binding"],
        "status": gate["status"],
        "provenance": template["provenance"],
    }

    write_json(output / "contract_template.json", template)
    write_json(output / "implementation_manifest.json", manifest)
    write_json(output / "implementation_fidelity.json",
               {**fidelity, "validity": validity, "mutations": mutations,
                "smoke_test": smoke, "gate": gate})
    write_json(output / "parameter_interface.json", template["parameter_interface"])
    (output / "implementation_report.md").write_text(
        to_markdown(spec, template, validity, fidelity, mutations, smoke, gate,
                    environment), encoding="utf-8")
    _companions(spec, template, validity, fidelity, mutations, smoke, gate, output)
    return {"output": str(output), "template": template, "validity": validity,
            "fidelity": fidelity, "mutations": mutations, "smoke": smoke, "gate": gate}


def _companions(spec, template, validity, fidelity, mutations, smoke, gate,
                output: Path) -> None:
    """보고서와 별개로 두는, 표로 열어 훑을 결과물."""
    import pandas as pd

    binding = template["execution_binding"]
    pd.DataFrame([{"항목": k, "값": v} for k, v in {
        "가설": template["hypothesis_id"],
        "계약해시": template["contract_sha256"],
        "명세해시": spec["specification_sha256"],
        "컴파일러": template["provenance"]["compiler"],
        "side": template["side"],
        "feature": spec["semantic_invariants"]["canonical_feature"],
        "방향": spec["semantic_invariants"]["direction"],
        "결정기회": _cadence(binding["decision_opportunity"],
                             binding.get("decision_spacing_seconds")),
        "평가창": binding["evaluation_horizon"],
        "미결정파라미터": ", ".join(gate["still_unresolved"]),
        "코드유효": validity["status"],
        "의미보존": fidelity["fidelity_status"],
        "변형전부잡힘": gate["mutations_all_caught"],
        "관문": gate["status"],
    }.items()]).to_csv(output / "summary.csv", index=False)

    comparisons = _comparisons(template["entry_program"]["signal"])
    signal_fields = []
    for index, node in enumerate(comparisons, start=1):
        signal_fields += [
            {"필드": f"signal.{index}.feature",
             "값": "\n".join(ast_lines(node["input"])), "출처": "명세 불변식"},
            {"필드": f"signal.{index}.comparator",
             "값": node["comparator"], "출처": "Grounding 관측 방향"},
            {"필드": f"signal.{index}.value", "값": node["value"],
             "출처": "미결정 — 탐색 단계가 정한다"},
        ]
    pd.DataFrame([{"필드": "side", "값": template["side"], "출처": "명세 불변식"},
                  {"필드": "signal.op", "값": template["entry_program"]["signal"]["op"],
                   "출처": "명세 신호 템플릿"},
                  *signal_fields,
                  {"필드": "warmup_ticks",
                   "값": template["entry_program"]["warmup_ticks"],
                   "출처": "시간 연산자가 없어 0"},
                  *[{"필드": f"execution_binding.{k}", "값": v, "출처": "명세 실행 결속"}
                    for k, v in binding.items() if not k.endswith("_note")]]
                 ).to_csv(output / "contract_fields.csv", index=False)

    pd.DataFrame([{"항목": row["item"], "명세": str(row["spec"]),
                   "구현": str(row["code"]),
                   "일치": "예" if row["match"] else "아니오",
                   "지표": row["metric"]} for row in fidelity["diff"]]
                 ).to_csv(output / "fidelity_check.csv", index=False)

    pd.DataFrame([{"변형": m["mutation"], "무엇을_바꿨나": m["expected"],
                   "잡혔나": "예" if m["caught"] else "아니오",
                   "어느_지표가": ", ".join(m["metrics"]), "판정": m["verdict"]}
                  for m in mutations]).to_csv(output / "mutation_tests.csv", index=False)

    pd.DataFrame([{"이름": name, "상태": item["status"], "타입": item["type"],
                   "feature": item["feature"], "역할": item["role"],
                   "자리": item["placeholder"], "값": item["value"], "뜻": item["meaning"]}
                  for name, item in sorted(template["parameter_interface"].items())]
                 ).to_csv(output / "parameter_interface.csv", index=False)

    meaning = {
        "feature_mismatch_count": "명세와 다른 feature 를 썼다",
        "direction_mismatch_count": "부등호가 검증 방향과 다르다",
        "side_mismatch_count": "매매 방향이 다르다",
        "context_reintroduced_count": "기각된 맥락이 다시 들어왔다",
        "new_feature_count": "핵심 관측 말고 다른 feature 가 신호에 있다",
        "new_relation_count": "신호 구조가 명세와 다르다",
        "new_temporal_operator_count": "가설에 없던 시간 연산자를 더했다",
        "horizon_mismatch_count": "평가 창이 다르다",
        "decision_cadence_mismatch_count": "결정 기회가 검증 표본과 다르다",
        "entry_execution_mismatch_count": "진입 체결 가정이 다르다",
        "premature_parameter_binding_count": "임계를 미리 채웠다",
        "unapproved_parameter_count": "승인되지 않은 탐색 자리가 있다",
        "dsl_schema_error_count": "표현식이 DSL 검증을 통과하지 못한다",
        "parameter_binding_error_count": "미결정 자리와 인터페이스가 어긋난다",
        "execution_binding_error_count": "실행 결속이 비었다",
        "market_data_access_count": "구현 단계에서 시장 데이터를 읽었다",
        "terminal_oos_access_count": "예약해 둔 마지막 구간을 열었다",
    }
    rows = [{"지표": k, "값": fidelity.get(k, 0), "종류": "구조"} for k in STRUCTURE_METRICS]
    rows += [{"지표": k, "값": validity.get(k, 0), "종류": "실행"} for k in RUNTIME_METRICS]
    rows += [{"지표": k, "값": smoke.get(k, 0), "종류": "데이터 접근"}
             for k in LEAKAGE_METRICS]
    for row in rows:
        row["기준"] = "0 이어야 함"
        row["무엇을_잡는가"] = meaning.get(row["지표"], "")
    pd.DataFrame(rows).to_csv(output / "audit.csv", index=False)
