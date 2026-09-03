"""가설의 각 주장을 데이터의 무엇으로 볼 것인지 고정한다. 검증하지는 않는다.

가설 한 문장에는 성격이 다른 것이 섞여 있다.

    관측 사실   "거래 활동이 anchor 이전부터 높다"
    금융 해석   "호가 갱신이 비동기적일 수 있다"
    예측       "anchor 이후에도 BID1 개선이 남는다"

이 셋을 같은 방식으로 데이터에 붙이면 안 된다. 첫째는 카탈로그 feature 로 바로 가고,
둘째는 proxy 이거나 아예 못 보는 것이고, 셋째는 **지금 보면 안 되는 것**이다 —
그것을 보는 순간 가설이 자기 답을 보고 쓰인 것이 된다.

## 이 단계가 하지 않는 것

가설을 새로 만들거나 더 그럴듯하게 고치지 않는다. 임계를 정하지 않는다. 진입·청산
규칙을 만들지 않는다. 그리고 **여러 proxy 를 만들어 성적으로 고르지 않는다** — 그건
구 Code Agent ×5 를 이름만 바꿔 되살리는 것이다.

## 가장 중요한 원칙

    grounding 은 되지만 뜻이 달라진 것
        보다
    지금은 grounding 할 수 없는 것

이 더 올바른 결과다. 의미 충실성이 실행 가능성보다 앞선다.

    Grounding    "이 주장은 데이터의 무엇인가"
    Validation   "그래서 실제로 그런가"

둘을 합치지 않는다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import capability as C, catalog, hypothesis as H, mechanism_graph as M
from .config import now_utc, read_json, sha256_json, write_json


SCHEMA_VERSION = "finance_grounded_specification.v1"
PROMPT_VERSION = "v1"

CLAIM_TYPES = ("OBSERVABLE_STATE", "TEMPORAL_RELATION", "MECHANISM_INTERPRETATION",
               "PREDICTED_CONSEQUENCE", "ALTERNATIVE_EXPLANATION")
EPISTEMIC = ("OBSERVED", "SUPPORTED_RELATION", "HYPOTHESIZED", "PREDICTED", "TO_BE_TESTED")
GROUNDING_STATUS = ("GROUNDED_DIRECT", "GROUNDED_DERIVED", "GROUNDED_PROXY",
                    "VALIDATION_TARGET", "UNRESOLVED_MECHANISM", "UNOBSERVABLE",
                    "INVALID_GROUNDING")
VALIDATION_TARGETS = ("PATH_TARGET", "TEMPORAL_PRECEDENCE_TARGET", "CONDITIONAL_EFFECT_TARGET",
                      "MATCHED_CONTROL_TARGET", "DOSE_RESPONSE_TARGET",
                      "INCREMENTAL_EFFECT_TARGET", "ALREADY_PRICED_CHECK")
IMPORTANCE = ("CORE", "SUPPORTING")
DIRECTIONS = ("HIGHER", "LOWER", "NONE")
READINESS = ("READY_FOR_VALIDATION", "PARTIALLY_GROUNDED", "BLOCKED_BY_CORE_UNOBSERVABLE",
             "INVALID_HYPOTHESIS_MAPPING")

# grounding 이 직접 관측이라 주장할 수 있는 상태
DIRECT_STATUS = ("GROUNDED_DIRECT", "GROUNDED_DERIVED")
MAX_IMPLEMENTATION_VARIANTS = 3


@dataclass(frozen=True)
class GroundingConfig:
    model: str = H.MODEL
    effort: str = H.REASONING_EFFORT


# ---- 데이터 capability (결정적) -------------------------------------------------

# 정본은 `capability.py` 하나다. 여기서는 그것을 그대로 쓴다.
CAPABILITIES = C.CAPABILITIES


def data_capability_inventory(required: Sequence[str] = ()) -> dict[str, Any]:
    """필요한 capability 와 현재 지원 여부. 요청된 것 중 모르는 이름은 그대로 남긴다."""
    unknown = [name for name in required if name not in CAPABILITIES]
    return {"schema": "data_capability_inventory.v1", "created_at": now_utc(),
            "source": "framework/capability.py — data.py 가 실제로 읽는 것에서 유도",
            "capabilities": {k: {kk: vv for kk, vv in v.items() if kk != "group"}
                             for k, v in sorted(CAPABILITIES.items())},
            "requested_but_unknown": unknown,
            "counts": {status: sum(1 for v in CAPABILITIES.values() if v["status"] == status)
                       for status in sorted({v["status"] for v in CAPABILITIES.values()})}}


# ---- PASS 0: 입력 감사 (코드) ---------------------------------------------------

def _evidence_roles_for(item: Mapping[str, Any],
                        package: Mapping[str, Any]) -> dict[str, Any]:
    """새 계약은 가설별 역할을 쓰고, 옛 Evidence 역할 지도는 fallback 으로 읽는다."""
    return dict(item.get("evidence_roles") or package.get("evidence_roles") or {})

def input_audit(hypotheses: Mapping[str, Any], package: Mapping[str, Any]) -> dict[str, Any]:
    """가설과 증거가 grounding 을 지탱하는가."""
    problems: list[str] = []
    items = hypotheses.get("hypotheses") or []
    if hypotheses.get("status") != H.GENERATED or not items:
        problems.append(f"grounding 할 가설이 없다 (status={hypotheses.get('status')})")
    known = {f.get("evidence_id") for f in package.get("evidence_families", [])}
    known |= {f.get("evidence_id") for f in H.execution_state_evidence_items(package)}
    known |= {f.get("evidence_id") for f in H.execution_temporal_evidence_items(package)}
    known |= {f.get("evidence_id") for f in H.raw_path_evidence_items(package)}
    for item in items:
        for b in item.get("evidence_basis") or []:
            if b.get("evidence_id") not in known:
                problems.append(f"{item.get('hypothesis_id')}: 없는 evidence_id "
                                f"{b.get('evidence_id')}")
        graph = item.get("mechanism_graph")
        if graph is not None:
            graph_report = M.validate(graph, known_evidence_ids=known,
                                      known_features=catalog.FEATURES)
            problems.extend(f"{item.get('hypothesis_id')}: {problem}"
                            for problem in graph_report["problems"])
        if not _evidence_roles_for(item, package):
            problems.append(f"{item.get('hypothesis_id')}: 증거 역할 지도가 없다")
    if not package.get("outcome_definition"):
        problems.append("outcome 정의가 없다")
    stored = package.get("catalog_sha256")
    if stored and stored != catalog.catalog_hash():
        problems.append(f"catalog hash 불일치: {stored} != {catalog.catalog_hash()}")
    return {"ok": not problems, "problems": problems,
            "hypotheses": [i.get("hypothesis_id") for i in items]}


# ---- Agent 가 보는 것 -----------------------------------------------------------

def grounding_input(hypotheses: Mapping[str, Any], package: Mapping[str, Any], *,
                    mechanism_specifications: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Grounding Agent 에게 줄 것. **post-anchor 결과는 넣지 않는다.**

    카탈로그 어휘는 이름·설명·차원만 준다. 임계도 관측 통계도 주지 않는다 — 여기서
    숫자를 고르면 그것이 곧 탐색이다.
    """
    capabilities = data_capability_inventory()
    temporal_evidence = [{
        "evidence_id": item.get("evidence_id"),
        "family": item.get("family"),
        "feature": (item.get("representative_feature") or {}).get("name"),
        "expression": item.get("expression"),
        "observation_window": item.get("observation_window"),
        "role_constraint": item.get("role"),
        "boundary": "stored pre-anchor Profile change; exact expression only",
    } for item in H.execution_temporal_evidence_items(package)]
    hypothesis_items = hypotheses.get("hypotheses") or []
    return {
        "outcome_definition": package.get("outcome_definition"),
        "evidence_roles": package.get("evidence_roles"),
        "hypothesis_evidence_roles": {
            str(item.get("hypothesis_id")): _evidence_roles_for(item, package)
            for item in hypothesis_items
        },
        "independent_evidence_families": package.get("independent_evidence_families"),
        "evidence_relations": [
            {k: v for k, v in item.items()
             if k in ("joint_evidence_id", "family_a", "family_b", "relation",
                      "context_family", "trigger_family", "ordering_status")}
            for item in package.get("joint_evidence", []) or []],
        "catalog": {
            "features": [
                {"name": f.name, "family": None, "dimension": f.dimension, "unit": f.unit,
                 "time_basis": f.time_basis, "lookback": f.lookback,
                 "observable_description": f.observable_description,
                 "definition_id": f.definition_id,
                 "threshold_policy": catalog.feature_use_policy(f.name).as_agent_dict()}
                for f in sorted(catalog.FEATURES.values(), key=lambda x: x.name)
                if f.name in set(catalog.guard_axes()) | set(
                    catalog.default_policy().grounding_visible - catalog.default_policy().execution_only)],
            "operators": [{"op": o.op, "required": list(o.required), "doc": o.doc}
                          for o in sorted(catalog.OPERATORS.values(), key=lambda x: x.op)]},
        "data_capabilities": capabilities["capabilities"],
        "hypotheses": hypothesis_items,
        "mechanism_specifications": dict(mechanism_specifications or {}),
        # Grounding은 result statistic을 보지 않는다. 다만 Hypothesis가 EVT를 인용했다면
        # exact pre-anchor difference 식을 다른 primitive state로 바꾸지 않게 보존한다.
        "execution_temporal_evidence": temporal_evidence,
        "not_provided": ["post-anchor price path", "validation results", "PnL",
                           "observed statistics", "thresholds"],
    }


# ---- 프롬프트 -------------------------------------------------------------------

ROLE = """\
You align a financial hypothesis with the data. You do not test it, and you do not improve it.

Your questions for each hypothesis are:
"What exactly in this dataset corresponds to each claim, and what does not?"
"What Catalog expression represents the supported observable state without claiming the mechanism itself?"

Hard rules:
- Do NOT create a new hypothesis, add a mechanism, or make an existing claim stronger.
- Do NOT invent features. Only names present in the given catalog may be used. If a
  financial concept has no catalog observable, say so — that is a valid, useful result.
- Do NOT choose numeric threshold values, cutoffs, quantiles, exit rules, or holding
  periods. Grounding fixes atomic claim -> feature and the observable expression; it
  does not choose the expression's unresolved parameter values or execution policy.
- Do NOT pick among candidate mappings by expected performance. You cannot see outcomes.
  If several mappings fit, record the ambiguity and compare only semantic fit,
  observability and limitations.
- Direction may be preserved (HIGHER / LOWER) because the evidence states it.
  Magnitude may not.
- A proxy is never the concept itself. If you use one, name what it misses.
- Semantic fidelity beats implementability. "Groundable but means something else" is
  WORSE than "cannot be grounded yet".

Separate the three kinds of sentence and never treat them alike:
- an OBSERVED state seen in the evidence,
- a HYPOTHESIZED mechanism the agent proposed to explain it,
- a PREDICTED consequence that must be checked later.

`operational_expression` is the canonical observable form of every hypothesis that can
proceed. It is a Catalog DSL expression for the reduced-form state that the hypothesis
will later test. It is never a direct observation of a financial mechanism. Keep atomic
claims atomic: each OBSERVABLE_STATE still names one catalog feature. The expression may
combine those already-grounded atomic claims with Catalog operators.

Rules for an operational_expression:
- Only use features named by the hypothesis's GROUNDED_DIRECT or GROUNDED_DERIVED
  OBSERVABLE_STATE claims. List every such claim id in `operational_claim_ids`.
- A multi-feature `all`/`any` expression needs a cited TEMPORAL_RELATION claim or
  an explicit evidence relation. Do not combine CONTEXT_INDEPENDENT observations.
- `sequence` needs a cited relation that supports the ordering. Do not infer an
  order from SETUP_STATE or LATE_TRIGGER role names alone.
- Keep every numeric cutoff unresolved as `UNRESOLVED:<parameter_name>`. Do not
  choose its value here. Boolean event primitives need no cutoff.
- Do not use an expression to turn a proxy, an unobservable mechanism, actual queue
  position, hidden liquidity, or order-event decomposition into a direct observable.
- For READY_FOR_VALIDATION or PARTIALLY_GROUNDED, emit an expression whenever a direct
  or derived observable state exists. A one-feature expression is valid when Evidence
  does not support a combination; it still uses the same AST contract as a composite
  expression. Only BLOCKED_BY_CORE_UNOBSERVABLE may leave the field null.

A predicted consequence is a VALIDATION_TARGET. You define how it would be measured;
you never look at it. Post-anchor data is deliberately withheld.

Prose fields must be written in Korean. Enum values, ids and feature names stay as given.
Return one JSON object and nothing else. No markdown fence, no commentary.
"""

_SCHEMA = """\
{
  "hypotheses": [
    {
      "hypothesis_id": str,
      "readiness": one of %(readiness)s,
      "readiness_reason": str,
      "operational_expression": Catalog DSL object | null,
      "operational_claim_ids": [claim_id],
      "operational_note": str | null,
      "implementation_variants": [
        {"direction_id": str, "operational_expression": Catalog DSL object,
         "operational_claim_ids": [claim_id], "operational_note": str}
      ],
      "mechanism_graph": {
        "schema": "mechanism_graph.v1",
        "observables": [{"node_id": str, "evidence_id": str,
                          "available_at": "DECISION" | "PENDING"}],
        "relations": [{"node_id": str, "relation": str, "input_ids": [str],
                       "available_at": "DECISION" | "PENDING"}],
        "execution": {"entry_state": "DECISION", "pending_state": "PENDING",
                      "entry_action": "BID1_QUEUE",
                      "entry_lifecycle_policy": {"mode": "HOLD_THROUGH" | "CANCEL_WHEN_ENTRY_SIGNAL_FALSE",
                                                 "pending_observable_ids": [str]}},
        "prediction": {"input_ids": [str], "evaluation_state": "FILLED",
                       "target_type": "PATH_TARGET" | "MATCHED_CONTROL_TARGET",
                       "direction": "HIGHER" | "LOWER"},
        "catalog_bindings": [{"observable_id": str, "catalog_feature": str}]
      } | null,
      "claims": [
        {
          "claim_id": "C1",
          "claim_text": str,
          "claim_type": one of %(types)s,
          "epistemic_status": one of %(epi)s,
          "importance": one of %(importance)s,
          "source_section": str,
          "supporting_evidence_ids": [evidence_id],
          "depends_on": [claim_id],

          "grounding_status": one of %(status)s,
          "catalog_family": str | null,
          "catalog_feature": str | null,
          "operator": str | null,
          "direction": one of %(dirs)s,

          "proxy_target": str | null,
          "proxy_rationale": str | null,
          "proxy_limitation": str | null,

          "required_data": [capability_name],
          "missing_capability": [capability_name],

          "validation_target": one of %(targets)s | null,
          "validation_measure": str | null,
          "validation_reference": str | null,

          "grounding_ambiguity": [{"candidate": str, "semantic_fit": str,
                                   "limitation": str}],
          "notes": str
        }
      ],
      "temporal_grounding": {
        "reference_event": str,
        "context_family": str | null,
        "trigger_family": str | null,
        "relation": str | null,
        "evidence_ordering": str,
        "validation_ordering": str
      },
      "discriminating_observations": [
        {"alternative": str, "observation": str, "supports_hypothesis_if": str,
         "supports_alternative_if": str, "grounding_status": one of %(status)s}
      ],
      "coverage": {"core_claims": int, "grounded_direct": int, "grounded_proxy": int,
                   "validation_targets": int, "unresolved_mechanisms": int,
                   "unobservable": int}
    }
  ]
}""" % {"readiness": list(READINESS), "types": list(CLAIM_TYPES), "epi": list(EPISTEMIC),
        "status": list(GROUNDING_STATUS), "importance": list(IMPORTANCE),
        "dirs": list(DIRECTIONS), "targets": list(VALIDATION_TARGETS)}


def grounding_prompt(payload: Mapping[str, Any]) -> str:
    return f"""{ROLE}
# Task

PASS 1 — Claim decomposition. Break each hypothesis into atomic claims. One claim must
  not mix an observation, an interpretation and a prediction. Give each an id, a type,
  an epistemic status and CORE/SUPPORTING importance, and record which claims it
  depends on. The dependency graph is the hypothesis's logical structure, NOT a causal
  graph.

PASS 2 — Catalog alignment. For every OBSERVABLE_STATE claim, name the catalog feature
  (and operator, if the claim needs a catalog operator to express it). Use only names
  from the given catalog. Preserve direction where the evidence states one.
  A fact about a relationship between two evidence families is not an OBSERVABLE_STATE:
  use TEMPORAL_RELATION with SUPPORTED_RELATION and GROUNDED_DERIVED, and leave its
  catalog_feature empty. OBSERVABLE_STATE always names exactly one catalog feature.
  When a supporting evidence id appears in `execution_temporal_evidence`, it is a direct
  observation of that exact stored pre-anchor `difference` expression. Keep its feature,
  operator and fixed lag intact; do not replace it with a level state or choose another
  lag. The deterministic implementation later restores this exact expression from the
  cited evidence id, so do not invent a threshold here.

  After the atomic mappings, write the hypothesis's `operational_expression`. This is
  the canonical observable form and the place for a supported Catalog composition such
  as `all`, `sequence`, or `persistence`; it does not change any atomic claim into a
  mechanism. Put every source claim id in `operational_claim_ids`. Use a one-feature
  AST when Evidence does not support a composition. Leave it null only if the hypothesis
  is BLOCKED_BY_CORE_UNOBSERVABLE and no direct/derived observable state can support it.

  Then materialize every input `implementation_directions` item in
  `implementation_variants`. A variant keeps its exact `direction_id` and can use only
  direct/derived observable claims already decomposed for this hypothesis. Its expression
  must be a genuinely supported representation, not a cosmetic rewrite. For
  `PRE_ANCHOR_CHANGE`, copy the cited stored exact difference AST and fixed lag. If the
  requested representation is unsupported, omit that variant and say why in
  `operational_note`; do not invent a lag, window, cutoff or feature. The top-level
  canonical expression remains required and must equal one emitted variant expression.

PASS 2.5 — MechanismGraph binding. When the input hypothesis has `mechanism_graph`,
copy its semantic fields exactly: observables, relations, execution, prediction, and
every state/action enum. Add exactly one `catalog_bindings` item per observable, using
the catalog feature that grounds the same observable claim. Do not add, remove, reorder,
or reinterpret graph nodes. The graph is not a new strategy: its only action is the
already fixed BID1 queue entry and its only optional lifecycle is the supplied pending
cancel mode. For a cancel mode, every graph observable must remain PENDING because the
runtime recomputes the complete entry signal. Leave `mechanism_graph` null only when the
input hypothesis had none.

PASS 2.6 — Pre-grounding mechanism contract. `mechanism_specifications` is frozen
before this step. For the matching hypothesis_id, preserve its economic mechanism,
decision/pending/BID1 contract, and post-fill prediction exactly. Only add Catalog
bindings and operational_expression; do not replace the declared relationship with an
easier observable or change the target type or direction.

PASS 3 — Observability classification. Assign a grounding_status. Use
  GROUNDED_PROXY only with an explicit proxy_target and proxy_limitation.
  Use UNRESOLVED_MECHANISM for a meaningful financial claim that has no observable.
  Use UNOBSERVABLE where the data structure itself cannot carry the claim, and list the
  missing capability. Use INVALID_GROUNDING if a candidate mapping would change what the
  claim means. Record required_data and missing_capability from the given capability list.

PASS 4 — Validation target design. Every PREDICTED_CONSEQUENCE becomes a
  VALIDATION_TARGET with a measure and a reference point — how it WOULD be measured.
  A CORE prediction about a future price/quote path or a conditional/incremental effect
  must state whether that future measure is expected to be HIGHER or LOWER. `NONE` is
  only for directionless temporal-precedence or already-priced checks.
  Do not evaluate it. If the hypothesis carries already_priced_risk, produce an
  ALREADY_PRICED_CHECK target that separates adjustment before the trigger from
  adjustment after it. Also ground each alternative explanation far enough to state a
  discriminating observation.

PASS 5 — Readiness. READY_FOR_VALIDATION when the core observables and the main
  prediction can be checked, even if the mechanism itself is only interpretation.
  PARTIALLY_GROUNDED when some CORE concept is proxy or unresolved but the main
  prediction is still checkable. BLOCKED_BY_CORE_UNOBSERVABLE only when the core claim
  produces no observable prediction at all and cannot be told apart from its
  alternatives. Do not drop a hypothesis merely because its mechanism is unresolved.

Temporal grounding rule: `temporal_grounding.relation` must copy the exact relation of
the Evidence pair. A `COUPLED` pair may still have an earlier context-family observation
and a later trigger-family observation. In that case keep `relation` as `COUPLED` and
write the earlier/later fact only in `evidence_ordering`; never replace the pair relation
with `CONTEXT_BEFORE_TRIGGER`.

# Output schema

{_SCHEMA}

# Input

{json.dumps(payload, ensure_ascii=False, indent=1)}
"""


def audit_prompt(payload: Mapping[str, Any], draft: Mapping[str, Any]) -> str:
    return f"""{ROLE}
# Task

You produced the grounding below. Audit it for semantic fidelity. You are the same role,
not a judge. **Do not add claims and do not add hypotheses.** Narrow or correct what is
already there, then emit the corrected grounding in the same schema.

Check every claim for:
- Did the grounding change the original financial meaning?
- Was a proxy treated as a direct observable?
- Was an unobservable mechanism quietly replaced with whatever feature was available?
- Did a numeric threshold, cutoff, or execution rule appear? Unresolved placeholders
  inside operational_expression are required and are not chosen thresholds.
- Was a mapping chosen using future outcome information?
- Was a redundant evidence family used as an independent grounding?
- Was a prediction marked OBSERVED instead of VALIDATION_TARGET?
- Does every CORE directional future prediction use HIGHER or LOWER rather than NONE?
- Is a claim mixing observation, interpretation and prediction in one sentence?
- Does operational_expression use only its cited direct/derived observable claims, keep
  every numeric cutoff unresolved, and avoid turning a proxy or mechanism into an
  observable rule?
- Does a multi-feature expression have an Evidence relation, and does sequence preserve
  the recorded ordering rather than invent one?
- Does `temporal_grounding.relation` exactly copy the Evidence pair relation? For a
  COUPLED pair, earlier/later family observations belong only in `evidence_ordering`; the
  relation remains COUPLED.

Correct a wrong status rather than deleting the claim. If a concept genuinely has no
observable, UNRESOLVED_MECHANISM or UNOBSERVABLE is the right answer — do not force a fit.

# Output schema

{_SCHEMA}

# Input

{json.dumps(payload, ensure_ascii=False, indent=1)}

# Draft to audit

{json.dumps(draft, ensure_ascii=False, indent=1)}
"""


# ---- 결정적 검증 (§54) -----------------------------------------------------------

def _expression_features(node: Any, out: set[str] | None = None) -> set[str]:
    """Catalog expression이 실제로 읽는 primitive 이름을 모은다."""
    out = set() if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") == "primitive" and node.get("primitive_id"):
            out.add(str(node["primitive_id"]))
        for value in node.values():
            _expression_features(value, out)
    elif isinstance(node, list):
        for value in node:
            _expression_features(value, out)
    return out


def _expression_ops(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, Mapping):
        if node.get("op"):
            out.add(str(node["op"]))
        for value in node.values():
            _expression_ops(value, out)
    elif isinstance(node, list):
        for value in node:
            _expression_ops(value, out)
    return out


def _numeric_cutoffs(node: Any, out: list[Any] | None = None) -> list[Any]:
    """Grounding이 고르면 안 되는 compare/crossover 숫자를 찾는다."""
    out = [] if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") == "compare" and isinstance(node.get("value"), (int, float)) \
                and not isinstance(node.get("value"), bool):
            out.append(node["value"])
        if node.get("op") == "crossover" and isinstance(node.get("threshold"), (int, float)) \
                and not isinstance(node.get("threshold"), bool):
            out.append(node["threshold"])
        for value in node.values():
            _numeric_cutoffs(value, out)
    elif isinstance(node, list):
        for value in node:
            _numeric_cutoffs(value, out)
    return out


def _comparison_nodes(node: Any, out: list[Mapping[str, Any]] | None = None
                      ) -> list[Mapping[str, Any]]:
    out = [] if out is None else out
    if isinstance(node, Mapping):
        if node.get("op") in {"compare", "crossover"}:
            out.append(node)
        for value in node.values():
            _comparison_nodes(value, out)
    elif isinstance(node, list):
        for value in node:
            _comparison_nodes(value, out)
    return out


def _operational_expression_problems(hypothesis: Mapping[str, Any]) -> list[str]:
    """복합 관측식이 원자 Grounding을 넘어서지 않는지 결정적으로 확인한다."""
    expression = hypothesis.get("operational_expression")
    claim_ids = [str(value) for value in hypothesis.get("operational_claim_ids") or []]
    claims = {str(claim.get("claim_id")): claim
              for claim in hypothesis.get("claims") or [] if claim.get("claim_id")}
    if expression is None:
        problems = (["operational_expression 이 없는데 operational_claim_ids 가 있다"]
                    if claim_ids else [])
        readiness = str(hypothesis.get("readiness") or "")
        if readiness not in {"BLOCKED_BY_CORE_UNOBSERVABLE", "INVALID_HYPOTHESIS_MAPPING"}:
            problems.append("진행 가능한 가설에는 operational_expression 이 필요하다")
        return problems
    if not isinstance(expression, Mapping):
        return ["operational_expression 은 Catalog DSL object 여야 한다"]
    if not claim_ids:
        return ["operational_expression 에 operational_claim_ids 가 없다"]

    problems: list[str] = []
    try:
        info = catalog.infer_expression_type(expression, allow_unresolved=True)
        if info.value_type not in ("boolean", "event"):
            problems.append("operational_expression 의 결과는 boolean 또는 event 여야 한다")
    except catalog.ExpressionError as error:
        problems.append(f"operational_expression DSL 검증 실패: {error}")
        return problems

    missing = sorted(set(claim_ids) - set(claims))
    if missing:
        problems.append(f"operational_claim_ids 에 없는 claim 이 있다: {missing}")
    selected = [claims[claim_id] for claim_id in claim_ids if claim_id in claims]
    observable = [claim for claim in selected
                  if claim.get("claim_type") == "OBSERVABLE_STATE"
                  and claim.get("grounding_status") in DIRECT_STATUS
                  and claim.get("catalog_feature")]
    invalid = sorted(str(claim.get("claim_id")) for claim in selected if claim not in observable
                     and claim.get("claim_type") != "TEMPORAL_RELATION")
    if invalid:
        problems.append("operational_expression 은 직접/파생 OBSERVABLE_STATE claim 만 "
                        f"feature로 쓸 수 있다: {invalid}")
    allowed_features = {str(claim["catalog_feature"]) for claim in observable}
    used_features = _expression_features(expression)
    unused = sorted(allowed_features - used_features)
    if unused:
        problems.append("operational_claim_ids 의 직접 관측 feature가 expression에 빠졌다: "
                        f"{unused}")
    extra = sorted(used_features - allowed_features)
    if extra:
        problems.append("operational_expression feature가 인용한 직접 관측 claim에 없다: "
                        f"{extra}")

    declared_directions = {str(claim["catalog_feature"]): str(claim.get("direction") or "")
                           for claim in observable}
    for comparison in _comparison_nodes(expression):
        source = comparison.get("input") or {}
        features = _expression_features(source)
        if len(features) != 1:
            continue
        feature = next(iter(features))
        expected = declared_directions.get(feature)
        if expected not in ("HIGHER", "LOWER"):
            continue
        actual = (str(comparison.get("direction")) if comparison.get("op") == "crossover"
                  else str(comparison.get("comparator") or ""))
        compatible = ((expected == "HIGHER" and actual in {">", ">=", "above"}) or
                      (expected == "LOWER" and actual in {"<", "<=", "below"}))
        if not compatible:
            problems.append(f"operational_expression 의 `{feature}` 방향 {actual!r} 이 "
                            f"claim 방향 {expected} 과 다르다")

    ops = _expression_ops(expression)
    has_relation = any(claim.get("claim_type") == "TEMPORAL_RELATION" for claim in selected)
    relation = str((hypothesis.get("temporal_grounding") or {}).get("relation") or "")
    if len(used_features) > 1 and not has_relation and not relation:
        problems.append("다중 feature operational_expression 에 Evidence relation claim이 없다")
    if len(used_features) > 1 and relation == "CONTEXT_INDEPENDENT":
        problems.append("CONTEXT_INDEPENDENT 관측을 operational_expression 으로 결합할 수 없다")
    ordering = str((hypothesis.get("temporal_grounding") or {}).get("evidence_ordering") or "")
    if "sequence" in ops and (not has_relation or relation == "CONTEXT_INDEPENDENT"
                               or not ordering or ordering == "-"):
        problems.append("sequence 는 인용한 시간 관계와 Evidence ordering이 필요하다")
    cutoffs = _numeric_cutoffs(expression)
    if cutoffs:
        problems.append(f"Grounding이 operational_expression cutoff를 미리 골랐다: {cutoffs}")
    if not str(hypothesis.get("operational_note") or "").strip():
        problems.append("operational_expression 의 reduced-form 한계를 operational_note 에 적어야 한다")
    return problems


def implementation_variants(grounded_hypothesis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Grounding이 실제 코드 후보로 분기한 관측 표현을 돌려준다.

    이전 artifact에는 variants가 없으므로 그 경우 기존 canonical expression 하나를
    D1로 읽는다. 새 artifact는 Grounding이 명시한 variant만 다음 단계로 보낸다.
    """
    raw = grounded_hypothesis.get("implementation_variants")
    if raw is None:
        return [{
            "direction_id": "D1",
            "operational_expression": copy.deepcopy(grounded_hypothesis.get("operational_expression")),
            "operational_claim_ids": list(grounded_hypothesis.get("operational_claim_ids") or []),
            "operational_note": grounded_hypothesis.get("operational_note"),
            "explicit": False,
        }]
    return [{**copy.deepcopy(dict(item)), "explicit": True}
            for item in raw if isinstance(item, Mapping)]


def materialize_implementation_variant(grounded_hypothesis: Mapping[str, Any],
                                       variant: Mapping[str, Any]) -> dict[str, Any]:
    """한 variant를 기존 Executable 입력 모양으로 펼친다."""
    value = copy.deepcopy(dict(grounded_hypothesis))
    for field in ("operational_expression", "operational_claim_ids", "operational_note"):
        value[field] = copy.deepcopy(variant.get(field))
    value["implementation_direction_id"] = str(variant.get("direction_id") or "")
    return value


def _representation_matches_expression(representation: str, expression: Any) -> bool:
    ops = _expression_ops(expression)
    if representation == "LEVEL":
        return not ops.intersection({"difference", "persistence", "sequence", "sequence_once"})
    if representation == "PRE_ANCHOR_CHANGE":
        return "difference" in ops
    if representation == "PERSISTENCE":
        return "persistence" in ops
    if representation == "SEQUENCE":
        return bool(ops.intersection({"sequence", "sequence_once"}))
    if representation == "COMPOSITE":
        return bool(ops.intersection({"all", "any", "sequence", "sequence_once"}))
    return False


def _implementation_variant_problems(grounded_hypothesis: Mapping[str, Any],
                                     source_hypothesis: Mapping[str, Any] | None) -> list[str]:
    raw = grounded_hypothesis.get("implementation_variants")
    source_directions = {
        str(item.get("direction_id")): str(item.get("representation") or "")
        for item in (source_hypothesis or {}).get("implementation_directions") or []
        if isinstance(item, Mapping) and item.get("direction_id")
    }
    if raw is None:  # v1 artifact — canonical expression 하나가 D1이다.
        return (["Hypothesis implementation_directions에 맞는 implementation_variants가 없다"]
                if source_directions else [])
    if not isinstance(raw, list) or not (1 <= len(raw) <= MAX_IMPLEMENTATION_VARIANTS):
        return ["implementation_variants는 1~3개여야 한다"]

    problems: list[str] = []
    direction_ids: set[str] = set()
    top_expression = grounded_hypothesis.get("operational_expression")
    has_canonical_variant = False
    for index, variant in enumerate(raw):
        label = f"implementation_variants[{index}]"
        if not isinstance(variant, Mapping):
            problems.append(f"{label}가 객체가 아니다")
            continue
        direction_id = str(variant.get("direction_id") or "")
        if not direction_id or direction_id in direction_ids:
            problems.append(f"{label}.direction_id가 없거나 중복됐다")
        direction_ids.add(direction_id)
        if source_directions and direction_id not in source_directions:
            problems.append(f"{label}.direction_id가 Hypothesis 방향에 없다: {direction_id}")
        candidate = materialize_implementation_variant(grounded_hypothesis, variant)
        problems.extend(f"{label}: {problem}" for problem in _operational_expression_problems(candidate))
        representation = source_directions.get(direction_id)
        if representation and not _representation_matches_expression(
                representation, variant.get("operational_expression")):
            problems.append(f"{label}: {representation} 방향과 Catalog 표현이 맞지 않는다")
        if variant.get("operational_expression") == top_expression:
            has_canonical_variant = True
    if not has_canonical_variant:
        problems.append("top-level operational_expression과 같은 implementation variant가 없다")
    return problems

def validate(grounded: Mapping[str, Any], hypotheses: Mapping[str, Any],
             package: Mapping[str, Any]) -> dict[str, Any]:
    """카탈로그 소속·연산자·의미 일관성·누출을 코드로 본다.

    LLM 이 지키겠다고 말한 것과 실제로 지킨 것은 다르다. 여기가 정본이다.
    """
    problems: list[str] = []
    metrics = {k: 0 for k in ("unknown_feature_count", "invalid_operator_count",
                              "semantic_mismatch_count", "proxy_as_direct_count",
                              "unresolved_core_count", "future_leakage_count",
                              "threshold_leakage_count", "strategy_leakage_count",
                              "redundant_grounding_count", "directionless_core_prediction_count",
                              "invalid_operational_expression_count",
                              "mechanism_graph_violation_count")}
    source = {i.get("hypothesis_id"): i for i in hypotheses.get("hypotheses") or []}
    graph_evidence_ids = {
        str(item.get("evidence_id")) for item in package.get("evidence_families", [])
        if item.get("evidence_id")}
    for items in (H.execution_state_evidence_items(package),
                  H.execution_temporal_evidence_items(package),
                  H.raw_path_evidence_items(package)):
        graph_evidence_ids.update(str(item.get("evidence_id")) for item in items
                                  if item.get("evidence_id"))
    relations = {tuple(sorted((str(p.get("family_a")), str(p.get("family_b"))))): p
                 for p in package.get("joint_evidence", []) or []}
    for item in grounded.get("hypotheses") or []:
        tag = item.get("hypothesis_id")
        non_independent = {
            family for family, role in _evidence_roles_for(source.get(tag) or {}, package).items()
            if role == "REDUNDANT"
        }
        if tag not in source:
            problems.append(f"{tag}: 원래 가설에 없는 id 다")
        if item.get("readiness") not in READINESS:
            problems.append(f"{tag}: 모르는 readiness {item.get('readiness')!r}")
        expression_problems = _operational_expression_problems(item)
        if expression_problems:
            metrics["invalid_operational_expression_count"] += len(expression_problems)
            problems.extend(f"{tag}: {problem}" for problem in expression_problems)
        variant_problems = _implementation_variant_problems(item, source.get(tag))
        if variant_problems:
            metrics["invalid_operational_expression_count"] += len(variant_problems)
            problems.extend(f"{tag}: {problem}" for problem in variant_problems)
        source_graph = (source.get(tag) or {}).get("mechanism_graph")
        grounded_graph = item.get("mechanism_graph")
        if source_graph is not None:
            graph_report = M.validate_preservation(
                source_graph, grounded_graph, known_evidence_ids=graph_evidence_ids,
                known_features=catalog.FEATURES)
            if graph_report["problems"]:
                metrics["mechanism_graph_violation_count"] += len(graph_report["problems"])
                problems.extend(f"{tag}: {problem}" for problem in graph_report["problems"])
        claims = item.get("claims") or []
        if not claims:
            problems.append(f"{tag}: claim 이 하나도 없다")
        ids = {c.get("claim_id") for c in claims}

        for claim in claims:
            cid = f"{tag}/{claim.get('claim_id')}"
            for field, allowed in (("claim_type", CLAIM_TYPES), ("epistemic_status", EPISTEMIC),
                                   ("grounding_status", GROUNDING_STATUS),
                                   ("importance", IMPORTANCE)):
                if claim.get(field) not in allowed:
                    problems.append(f"{cid}: 모르는 {field} {claim.get(field)!r}")
            if claim.get("direction") is not None and claim.get("direction") not in DIRECTIONS:
                problems.append(f"{cid}: 모르는 direction {claim.get('direction')!r}")
            for dep in claim.get("depends_on") or []:
                if dep not in ids:
                    problems.append(f"{cid}: 없는 claim 을 참조한다 ({dep})")

            status = claim.get("grounding_status")
            feature = claim.get("catalog_feature")
            operator = claim.get("operator")

            # Rule 1·2 — 카탈로그에 있고 별칭이 아닌 정본인가. 둘을 겹쳐 세지 않는다.
            if feature and feature not in catalog.FEATURES:
                metrics["unknown_feature_count"] += 1
                problems.append(
                    f"{cid}: 정본이 아닌 별칭 '{feature}' "
                    f"(정본은 '{catalog.resolve(feature)}')" if feature in catalog.ALIASES
                    else f"{cid}: 카탈로그에 없는 feature '{feature}'")
            # Rule 3 — operator 가 카탈로그 DSL 에 있는가
            if operator and operator not in catalog.OPERATORS:
                metrics["invalid_operator_count"] += 1
                problems.append(f"{cid}: 카탈로그에 없는 연산자 '{operator}'")

            # Rule 4 — DIRECT/DERIVED 는 실제로 feature 를 걸어야 한다
            if status in DIRECT_STATUS and claim.get("claim_type") == "OBSERVABLE_STATE" \
                    and not feature:
                metrics["semantic_mismatch_count"] += 1
                problems.append(f"{cid}: {status} 인데 카탈로그 feature 가 없다")
            if status == "GROUNDED_DIRECT" and claim.get("proxy_target"):
                metrics["proxy_as_direct_count"] += 1
                problems.append(f"{cid}: proxy 를 직접 관측으로 적었다")
            if status == "GROUNDED_PROXY" and not (claim.get("proxy_target")
                                                   and claim.get("proxy_limitation")):
                problems.append(f"{cid}: proxy 인데 대상이나 한계가 비었다")
            # 해석·예측을 직접 관측으로 둔갑시키지 않았는가
            if claim.get("claim_type") == "MECHANISM_INTERPRETATION" and status == "GROUNDED_DIRECT":
                metrics["semantic_mismatch_count"] += 1
                problems.append(f"{cid}: 메커니즘 해석을 직접 관측으로 적었다")

            # Rule 5 — 중복 family 를 독립 grounding 으로 쓰지 않았는가
            if claim.get("catalog_family") in non_independent and status in DIRECT_STATUS:
                metrics["redundant_grounding_count"] += 1
                problems.append(f"{cid}: 독립 증거가 아닌 family "
                                f"{claim.get('catalog_family')} 를 grounding 으로 썼다")

            # Rule 7 — 예측을 관측으로 되돌리지 않았는가
            if claim.get("claim_type") == "PREDICTED_CONSEQUENCE":
                if claim.get("epistemic_status") == "OBSERVED" or status in DIRECT_STATUS:
                    metrics["future_leakage_count"] += 1
                    problems.append(f"{cid}: 예측을 관측으로 적었다 "
                                    f"({claim.get('epistemic_status')}/{status})")
                if status == "VALIDATION_TARGET" and not claim.get("validation_target"):
                    problems.append(f"{cid}: VALIDATION_TARGET 인데 종류가 없다")
                directional_targets = {"PATH_TARGET", "MATCHED_CONTROL_TARGET",
                                       "CONDITIONAL_EFFECT_TARGET", "DOSE_RESPONSE_TARGET",
                                       "INCREMENTAL_EFFECT_TARGET"}
                if (claim.get("importance") == "CORE" and status == "VALIDATION_TARGET"
                        and claim.get("validation_target") in directional_targets
                        and claim.get("direction") not in ("HIGHER", "LOWER")):
                    metrics["directionless_core_prediction_count"] += 1
                    problems.append(f"{cid}: CORE 미래 경로/효과 예측에는 HIGHER 또는 LOWER 방향이 필요하다")
            if claim.get("validation_target") and claim["validation_target"] not in VALIDATION_TARGETS:
                problems.append(f"{cid}: 모르는 validation_target {claim['validation_target']!r}")

            if status == "UNRESOLVED_MECHANISM" and claim.get("importance") == "CORE":
                metrics["unresolved_core_count"] += 1

            # Rule 8 — 임계·전략 표현이 들어오지 않았는가
            for text in H._strings({k: v for k, v in claim.items()
                                    if k not in ("notes", "proxy_limitation")}):
                for pattern, label in H.FORBIDDEN:
                    match = pattern.search(text)
                    if match:
                        key = ("threshold_leakage_count" if "임계" in label or "분위수" in label
                               else "strategy_leakage_count")
                        metrics[key] += 1
                        problems.append(f"{cid}: {label} — {match.group()!r}")
                        break

        # Rule 6 — 시간 관계가 증거의 관계와 어긋나지 않는가
        temporal = item.get("temporal_grounding") or {}
        pair = tuple(sorted((str(temporal.get("context_family")),
                             str(temporal.get("trigger_family")))))
        given = relations.get(pair)
        if temporal.get("relation") and given and temporal["relation"] != given.get("relation"):
            metrics["semantic_mismatch_count"] += 1
            problems.append(f"{tag}: 시간 관계를 {temporal['relation']} 로 적었는데 "
                            f"증거는 {given.get('relation')} 다")
        if given and temporal.get("context_family") and \
                given.get("context_family") and \
                temporal["context_family"] != given["context_family"]:
            metrics["semantic_mismatch_count"] += 1
            problems.append(f"{tag}: 맥락 family 를 바꿔 적었다 "
                            f"({temporal['context_family']} vs {given['context_family']})")

    metrics["structural_violation_count"] = sum(
        metrics[k] for k in ("unknown_feature_count", "invalid_operator_count",
                             "semantic_mismatch_count", "proxy_as_direct_count",
                             "future_leakage_count", "threshold_leakage_count",
                             "strategy_leakage_count", "redundant_grounding_count",
                             "invalid_operational_expression_count",
                             "mechanism_graph_violation_count"))
    return {"problems": problems, **metrics}


# ---- 표와 보고서 -----------------------------------------------------------------

def claim_table(grounded: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in grounded.get("hypotheses") or []:
        for claim in item.get("claims") or []:
            rows.append({
                "hypothesis_id": item.get("hypothesis_id"),
                "claim_id": claim.get("claim_id"),
                "claim_type": claim.get("claim_type"),
                "claim_text": claim.get("claim_text"),
                "epistemic_status": claim.get("epistemic_status"),
                "importance": claim.get("importance"),
                "grounding_status": claim.get("grounding_status"),
                "catalog_family": claim.get("catalog_family"),
                "catalog_feature": claim.get("catalog_feature"),
                "definition_id": (catalog.FEATURES[claim["catalog_feature"]].definition_id
                                  if claim.get("catalog_feature") in catalog.FEATURES else None),
                "operator": claim.get("operator"),
                "direction": claim.get("direction"),
                "proxy_target": claim.get("proxy_target"),
                "proxy_limitation": claim.get("proxy_limitation"),
                "required_data": ",".join(claim.get("required_data") or []),
                "missing_capability": ",".join(claim.get("missing_capability") or []),
                "validation_target": claim.get("validation_target"),
                "validation_measure": claim.get("validation_measure"),
                "depends_on": ",".join(claim.get("depends_on") or []),
                "ambiguity": len(claim.get("grounding_ambiguity") or [])})
    return pd.DataFrame(rows)


def validation_plan(grounded: Mapping[str, Any], hypotheses: Mapping[str, Any]) -> dict[str, Any]:
    """아직 실행하지 않은 검증 목록. 결과 칸은 비워 둔다."""
    tests = {i.get("hypothesis_id"): i.get("mechanism_tests") or []
             for i in hypotheses.get("hypotheses") or []}
    plan = []
    for item in grounded.get("hypotheses") or []:
        tag = item.get("hypothesis_id")
        for claim in item.get("claims") or []:
            if claim.get("grounding_status") != "VALIDATION_TARGET":
                continue
            plan.append({
                "hypothesis_id": tag, "claim_id": claim.get("claim_id"),
                "target_type": claim.get("validation_target"),
                "claim": claim.get("claim_text"),
                "measure": claim.get("validation_measure"),
                "reference": claim.get("validation_reference"),
                "required_data": claim.get("required_data") or [],
                "importance": claim.get("importance"),
                "status": "NOT_RUN", "result": None})
        for test in tests.get(tag, []):
            plan.append({
                "hypothesis_id": tag, "claim_id": None,
                "target_type": test.get("test_type"),
                "claim": test.get("claim"), "measure": test.get("expected_result"),
                "reference": test.get("control"),
                "required_data": [test.get("required_evidence")],
                "importance": "CORE", "status": "NOT_RUN", "result": None})
        for disc in item.get("discriminating_observations") or []:
            plan.append({
                "hypothesis_id": tag, "claim_id": None,
                "target_type": "DISCRIMINATING_OBSERVATION",
                "claim": disc.get("alternative"), "measure": disc.get("observation"),
                "reference": disc.get("supports_hypothesis_if"),
                "required_data": [], "importance": "CORE",
                "status": "NOT_RUN", "result": None})
    return {"schema": "validation_plan_draft.v1", "created_at": now_utc(),
            "note": "설계만 했고 실행하지 않았다. 결과는 Mechanism Validation 의 몫이다.",
            "targets": plan, "count": len(plan)}


def to_markdown(grounded: Mapping[str, Any], hypotheses: Mapping[str, Any],
                package: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    table = claim_table(grounded)
    source = {i.get("hypothesis_id"): i for i in hypotheses.get("hypotheses") or []}
    L = ["# grounded_hypotheses.md", "",
         "가설의 각 주장을 데이터의 무엇으로 볼 것인지 고정한 기록. **검증하지 않았다.**", "",
         f"- 카탈로그 `{catalog.catalog_hash()}`",
         f"- outcome `{(package.get('outcome_definition') or {}).get('horizon')}`",
         f"- 구조 위반 `{audit['structural_violation_count']}` · "
         f"미해결 CORE 메커니즘 `{audit['unresolved_core_count']}`", ""]
    for item in grounded.get("hypotheses") or []:
        tag = item.get("hypothesis_id")
        origin = source.get(tag, {})
        part = table[table.hypothesis_id == tag]
        L += [f"## {tag} — {origin.get('title', '')}", "",
              f"**readiness `{item.get('readiness')}`** · 원래 confidence "
              f"`{origin.get('confidence')}`"
              + (" · ⚠️ `already_priced_risk`" if origin.get("already_priced_risk") else ""), "",
              item.get("readiness_reason", ""), "",
              "### 핵심 표", "",
              "| Claim | Type | Evidence status | Grounding | Observable | Next action |",
              "|---|---|---|---|---|---|"]
        for r in part.itertuples():
            observable = (f"`{r.catalog_feature}`" if r.catalog_feature else
                          (r.proxy_target or "—"))
            action = {"GROUNDED_DIRECT": "그대로 쓴다", "GROUNDED_DERIVED": "그대로 쓴다",
                      "GROUNDED_PROXY": "한계를 명시하고 쓴다",
                      "VALIDATION_TARGET": "Mechanism Validation",
                      "UNRESOLVED_MECHANISM": "capability 확인",
                      "UNOBSERVABLE": "데이터 확장 필요",
                      "INVALID_GROUNDING": "매핑 폐기"}.get(r.grounding_status, "—")
            L.append(f"| {r.claim_id} {str(r.claim_text)[:44]} | {r.claim_type.split('_')[0]} | "
                     f"`{r.epistemic_status}` | `{r.grounding_status}` | {observable} | {action} |")
        coverage = item.get("coverage") or {}
        L += ["", "### Grounding coverage", "", "| 항목 | 수 |", "|---|--:|"]
        L += [f"| {k} | {v} |" for k, v in coverage.items()]
        temporal = item.get("temporal_grounding") or {}
        if temporal:
            L += ["", "### 시간 grounding", "", "| 항목 | 값 |", "|---|---|"]
            L += [f"| `{k}` | {v} |" for k, v in temporal.items()]
        L += ["", "### 주장별 상세", ""]
        for claim in item.get("claims") or []:
            L += [f"**{claim.get('claim_id')}** ({claim.get('importance')}) — "
                  f"{claim.get('claim_text')}", "",
                  f"- type `{claim.get('claim_type')}` · epistemic "
                  f"`{claim.get('epistemic_status')}` · grounding "
                  f"`{claim.get('grounding_status')}`"]
            if claim.get("catalog_feature"):
                L.append(f"- feature `{claim['catalog_feature']}`"
                         + (f" · 방향 `{claim.get('direction')}`" if claim.get("direction") else "")
                         + (f" · 연산자 `{claim['operator']}`" if claim.get("operator") else ""))
            if claim.get("proxy_target"):
                L += [f"- proxy `{claim['proxy_target']}` — {claim.get('proxy_rationale')}",
                      f"- 한계: {claim.get('proxy_limitation')}"]
            if claim.get("validation_target"):
                L += [f"- 검증 대상 `{claim['validation_target']}` — {claim.get('validation_measure')}"]
            if claim.get("missing_capability"):
                L.append(f"- 없는 데이터: {', '.join(claim['missing_capability'])}")
            if claim.get("grounding_ambiguity"):
                L.append(f"- ⚠️ `GROUNDING_AMBIGUITY` 후보 "
                         f"{len(claim['grounding_ambiguity'])}개 — 성적으로 고르지 않았다")
            L.append("")
        # §66 — grounding 이 끝난 뒤 반드시 답해야 하는 것
        by_status: dict[str, list[str]] = {}
        for r in part.itertuples():
            by_status.setdefault(r.grounding_status, []).append(
                f"{r.claim_id} {r.catalog_feature or r.proxy_target or str(r.claim_text)[:28]}")
        say = lambda k: ", ".join(by_status.get(k, [])) or "없음"
        roles = _evidence_roles_for(origin, package)
        temporal = item.get("temporal_grounding") or {}
        L += ["### Validation readiness 질문", "", "| 질문 | 답 |", "|---|---|",
              f"| Q1 관측 가능한 맥락은 무엇인가 | {temporal.get('context_family') or '—'} |",
              f"| Q2 trigger 는 무엇으로 관측하는가 | {temporal.get('trigger_family') or '—'} |",
              f"| Q3 시간 관계는 어떻게 정의되는가 | `{temporal.get('relation') or '—'}` · "
              f"{temporal.get('evidence_ordering', '')[:70]} |",
              f"| Q4 직접 관측 가능한 것 | {say('GROUNDED_DIRECT')} |",
              f"| Q5 proxy 만 가능한 것 | {say('GROUNDED_PROXY')} |",
              f"| Q6 지금 전혀 관측 못 하는 것 | {say('UNOBSERVABLE')} · "
              f"(미해결 메커니즘 {say('UNRESOLVED_MECHANISM')}) |",
              f"| Q7 이후 반드시 나타나야 하는 예측 | {say('VALIDATION_TARGET')} |",
              f"| Q8 already-priced 대안을 어떻게 기각하나 | "
              f"{'`ALREADY_PRICED_CHECK` 목표 있음' if any(c.get('validation_target') == 'ALREADY_PRICED_CHECK' for c in item.get('claims') or []) else '**목표 없음**'} |",
              f"| Q9 대안과 구분할 관측 | {len(item.get('discriminating_observations') or [])}개 정의됨 |",
              f"| Q10 지금 데이터로 Validation 가능한가 | "
              f"`{item.get('readiness')}` |", ""]
        invalid = by_status.get("INVALID_GROUNDING")
        if invalid:
            L += ["> ⚠️ `INVALID_GROUNDING` — 억지로 붙이면 뜻이 달라지는 매핑이 있다: "
                  f"{', '.join(invalid)}", ""]

        disc = item.get("discriminating_observations") or []
        if disc:
            L += ["### 대안과 구분하는 관측", "",
                  "| 대안 | 구분 관측 | 가설을 지지하면 | 대안을 지지하면 |", "|---|---|---|---|"]
            for d in disc:
                L.append(f"| {d.get('alternative')} | {d.get('observation')} | "
                         f"{d.get('supports_hypothesis_if')} | {d.get('supports_alternative_if')} |")
            L.append("")
    return "\n".join(L) + "\n"


# ---- 배선 -----------------------------------------------------------------------

def run(hypotheses_path: Path, package_path: Path, output: Path, *,
        config: GroundingConfig = GroundingConfig(), agent=None) -> dict[str, Any]:
    """PASS 0 → grounding → 의미 충실성 감사 → 결정적 검증 → 산출물 6개."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hypotheses = read_json(Path(hypotheses_path))
    package = read_json(Path(package_path))
    call = agent or (lambda prompt: H.run_agent(prompt, model=config.model,
                                                effort=config.effort))

    audit_in = input_audit(hypotheses, package)
    payload = grounding_input(hypotheses, package)
    prompt = grounding_prompt(payload)

    record: dict[str, Any] = {
        "schema": "grounding_audit.v1", "created_at": now_utc(),
        "input_audit": audit_in,
        "hypotheses": audit_in["hypotheses"]}

    if not audit_in["ok"]:
        grounded = {"hypotheses": [], "blocked_at": "PASS_0_INPUT_AUDIT"}
        record.update({"validation": validate(grounded, hypotheses, package),
                       "blocked": True})
    else:
        draft = call(prompt)
        revised = call(audit_prompt(payload, draft))
        grounded = dict(revised)
        report = validate(grounded, hypotheses, package)
        record.update({
            "draft_claims": sum(len(i.get("claims") or [])
                                for i in draft.get("hypotheses") or []),
            "final_claims": sum(len(i.get("claims") or [])
                                for i in grounded.get("hypotheses") or []),
            "validation": report, "blocked": False})

    inventory = data_capability_inventory(
        sorted({c for i in grounded.get("hypotheses") or []
                for claim in i.get("claims") or []
                for c in (claim.get("required_data") or []) + (claim.get("missing_capability") or [])}))
    plan = validation_plan(grounded, hypotheses)
    table = claim_table(grounded)

    manifest = {
        "schema": "grounding_manifest.v1", "created_at": now_utc(),
        "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "catalog_hash": catalog.catalog_hash(),
        "hypotheses_path": str(Path(hypotheses_path)),
        "evidence_package_path": str(Path(package_path)),
        "hypotheses_hash": sha256_json(hypotheses)[:16],
        "package_hash": sha256_json(package)[:16],
        "model": config.model, "reasoning_effort": config.effort,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "role_sha256": hashlib.sha256(ROLE.encode()).hexdigest()[:16],
        "output_sha256": sha256_json(grounded)[:16],
        "config": asdict(config)}

    write_json(output / "grounded_hypotheses.json", grounded)
    write_json(output / "grounding_audit.json", record)
    write_json(output / "data_capability_inventory.json", inventory)
    write_json(output / "validation_plan_draft.json", plan)
    write_json(output / "grounding_manifest.json", manifest)
    if len(table):
        table.to_parquet(output / "claim_grounding.parquet", index=False)
    (output / "grounded_hypotheses.md").write_text(
        to_markdown(grounded, hypotheses, package, record["validation"]), encoding="utf-8")

    return {"output": str(output), "grounded": grounded, "audit": record,
            "inventory": inventory, "plan": plan, "manifest": manifest,
            "claims": len(table)}
