"""Agent knowledge -> Discovery search -> Refinement -> frozen Validation and Final replay.

This is the only research path.  It keeps these explicit boundaries:

* decision-time inputs are causal;
* fills use actual BID/ASK prices and exactly 23 bps explicit cost;
* Discovery, Validation, and Final dates do not overlap;
* Refinement reads Discovery ledgers only;
* the complete refined strategy is frozen before Validation opens;
* side is LONG and the canonical entry/exit execution profile never changes.

Feature Profile, Evidence, mechanism stories, Grounding, fidelity gates, novelty
checks, and the old matched-path q selection are not inputs to this workflow.
"""

from __future__ import annotations

import copy
import importlib.metadata
import itertools
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from .. import (canonical as K, catalog, contract, ledger,
                path_alignment as PA, profit_target as PT)
from ..agents import refinement as refinement_agent
from ..agents.runtime import (AGENT_TIMEOUT_SECONDS, MODEL, REASONING_EFFORT,
                              CodexAgentRunner, Runner, invoke)
from ..config import (REPO, TICK_ROOT, clean, git_snapshot, now_utc, read_json,
                      sha256_file, sha256_json, write_json)
from ..contracts.artifacts import create_artifact, write_artifact
from ..modules import backtest as backtest_module
from ..modules import refinement as refinement_module


PARAMETER_KINDS = ("rolling_quantile", "literal")
UNRESOLVED = catalog.UNRESOLVED_PREFIX
REFINEMENT_POPULATIONS = ("ALL_DECISIONS", "FILLED_ONLY")
MAX_REFINEMENT_ROUNDS = 10

# 실행 의미를 바꿀 수 있는 파일. `python -m framework` 의 두 명령이 실제로
# 로드하는 모듈 + 진입점이다. 사슬로만 끌려오던 옛 Workflow 파일은 뺐다.
RUNTIME_SOURCE_FILES = (
    "framework/FeatureProfile/ExtractLabel/__init__.py",
    "framework/FeatureProfile/ExtractLabel/extract.py",
    "framework/FeatureProfile/ExtractLabel/quality.py",
    "framework/__init__.py",
    "framework/__main__.py",
    "framework/agents/__init__.py",
    "framework/agents/loss_ledger.py",
    "framework/agents/price_path.py",
    "framework/agents/refinement.py",
    "framework/agents/runtime.py",
    "framework/calibration.py",
    "framework/canonical.py",
    "framework/capability.py",
    "framework/catalog.py",
    "framework/config.py",
    "framework/contract.py",
    "framework/contracts/__init__.py",
    "framework/contracts/artifacts.py",
    "framework/data.py",
    "framework/diagnose.py",
    "framework/evidence.py",
    "framework/executable.py",
    "framework/execution_profile.py",
    "framework/exits.py",
    "framework/featureprofile.py",
    "framework/fill.py",
    "framework/hypothesis.py",
    "framework/implementation.py",
    "framework/joint.py",
    "framework/ledger.py",
    "framework/mechanism_graph.py",
    "framework/metrics.py",
    "framework/modules/__init__.py",
    "framework/modules/backtest.py",
    "framework/modules/common_stock_cache.py",
    "framework/modules/discovery_loss.py",
    "framework/modules/evidence.py",
    "framework/modules/evidence_cache.py",
    "framework/modules/execution_anchor.py",
    "framework/modules/profile.py",
    "framework/modules/refinement.py",
    "framework/outcome.py",
    "framework/path_alignment.py",
    "framework/profit.py",
    "framework/profit_target.py",
    "framework/sample_condition.py",
    "framework/stages/__init__.py",
    "framework/stages/evidence_stage.py",
    "framework/universe.py",
    "framework/workflows/__init__.py",
    "framework/workflows/free_research.py",
    "framework/workflows/free_research_campaign.py",
)


def runtime_identity() -> dict[str, Any]:
    source_files = {
        relative: sha256_file(REPO / relative)
        for relative in RUNTIME_SOURCE_FILES
    }
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "optuna", "pandas", "pyarrow")
    }
    return {
        "git": git_snapshot(),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "source_files": source_files,
        "framework_code_sha256": sha256_json(source_files),
        "canonical_profile_id": K.CANONICAL["profile_id"],
        "canonical_profile_sha256": K.CANONICAL["profile_sha256"],
    }


@dataclass(frozen=True)
class FreeResearchRequest:
    symbols: tuple[str, ...]
    discovery_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    final_dates: tuple[str, ...]
    discovery_symbols: tuple[str, ...] | None = None
    validation_symbols: tuple[str, ...] | None = None
    final_symbols: tuple[str, ...] | None = None
    strategy_source: Path | None = None
    strategy_count: int = 8
    model: str = MODEL
    effort: str = REASONING_EFFORT
    agent_timeout_seconds: int = AGENT_TIMEOUT_SECONDS
    workers: int = 8
    trials_per_strategy: int = 50
    optuna_processes: int = 4
    optuna_seed: int = 1729
    refinement_rounds: int = 1
    refinement_max_proposals: int = 3
    refinement_populations: tuple[str, ...] = REFINEMENT_POPULATIONS


Replay = Callable[[Mapping[str, Any], Sequence[str], Sequence[str], Path, int, Path], dict[str, Any]]


def _normalise_symbols(values: Sequence[str]) -> tuple[str, ...]:
    symbols = tuple(dict.fromkeys(str(value).zfill(6) for value in values))
    if not symbols:
        raise ValueError("research에는 symbol이 하나 이상 필요하다")
    return symbols


def _stage_symbols(request: FreeResearchRequest) -> dict[str, tuple[str, ...]]:
    base = _normalise_symbols(request.symbols)
    return {
        "discovery": _normalise_symbols(request.discovery_symbols or base),
        "validation": _normalise_symbols(request.validation_symbols or base),
        "final": _normalise_symbols(request.final_symbols or base),
    }


def _validate_dates(request: FreeResearchRequest) -> None:
    blocks = {
        "discovery": set(map(str, request.discovery_dates)),
        "validation": set(map(str, request.validation_dates)),
        "final": set(map(str, request.final_dates)),
    }
    for name, dates in blocks.items():
        if not dates:
            raise ValueError(f"{name} 날짜가 비었다")
        invalid = sorted(value for value in dates if len(value) != 8 or not value.isdigit())
        if invalid:
            raise ValueError(f"{name} 날짜는 YYYYMMDD여야 한다: {invalid}")
    names = tuple(blocks)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(blocks[left] & blocks[right])
            if shared:
                raise ValueError(f"{left}와 {right} 날짜가 겹친다: {shared}")
    if max(blocks["discovery"]) >= min(blocks["validation"]):
        raise ValueError("Discovery 날짜는 Validation 날짜보다 앞서야 한다")
    if max(blocks["validation"]) >= min(blocks["final"]):
        raise ValueError("Validation 날짜는 Final 날짜보다 앞서야 한다")


def _validate_refinement(request: FreeResearchRequest) -> None:
    rounds = int(request.refinement_rounds)
    if rounds < 0 or rounds > MAX_REFINEMENT_ROUNDS:
        raise ValueError(f"refinement_rounds는 0..{MAX_REFINEMENT_ROUNDS}이어야 한다")
    if int(request.refinement_max_proposals) < 1:
        raise ValueError("refinement_max_proposals는 1 이상이어야 한다")
    populations = tuple(dict.fromkeys(map(str, request.refinement_populations)))
    if not populations:
        raise ValueError("refinement_populations가 비었다")
    unknown = sorted(set(populations) - set(REFINEMENT_POPULATIONS))
    if unknown:
        raise ValueError(f"지원하지 않는 refinement population: {unknown}")


def _catalog_for_agent() -> dict[str, Any]:
    """Agent 어휘. 정제와 **같은 함수**를 쓴다.

    전에는 두 곳이 각자 손으로 목록을 만들어 필드 이름까지 달랐고, 둘 다 관측 범위와
    임계 규칙을 버렸다. 그래서 Agent 가 정수 count 에 `> 0.9` 를, 관측 p1=-40 인
    feature 에 `> 0.5` 를 박았다.
    """
    return catalog.agent_vocabulary(policy=catalog.agent_causal_policy(),
                                    visibility="grounding_visible")


def generation_prompt(strategy_count: int, grid_cap: int = 50) -> str:
    claim_ops = " ".join(sorted(catalog.CLAIM_OPS - {"primitive", "raw", "level_aggregate"}))
    schema = {
        "strategies": [{
            "strategy_id": "KS001",
            "title": "short title",
            "rationale": "your market knowledge; no claim that it already made money",
            "claims": [{
                "claim_id": "C1",
                "text": "one atomic observable claim, no interpretation, no prediction",
                "feature": "catalog_feature",
            }],
            "path_prediction": {name: " | ".join(values)
                                for name, values in PA.FIELDS.items()},
            "variants": [{
                "variant_id": "D1",
                "representation": " | ".join(catalog.REPRESENTATIONS),
                "claim_ids": ["C1"],
                "entry_expression": {
                    "op": "compare",
                    "input": {"op": "primitive", "primitive_id": "catalog_feature"},
                    "comparator": ">",
                    "value": "UNRESOLVED:parameter_name",
                },
                "parameters": [{
                    "name": "parameter_name",
                    "kind": "rolling_quantile or literal",
                    "values": [0.5, 0.7, 0.9],
                }],
            }],
        }]
    }
    raw_inputs = {
        "book arrays; level is 1..10": ["bid_price", "ask_price", "bid_qty", "ask_qty"],
        "current tick scalars": [
            "time_s", "local_time", "buy_volume", "sell_volume",
            "buy_max_price", "sell_min_price",
        ],
    }
    return f"""Create {int(strategy_count)} materially different LONG tick-trading entry conditions
from your own market knowledge. You receive no market outcomes, PROFIT/LOSS labels, Evidence, prior
results, or successful candidates. Compose any causal expression from raw inputs, convenience
features, and the operators below.

Optuna searches only the entry condition and its parameters on Discovery net PnL. LONG side,
decision timing, order handling, entry execution, stop, trailing exit, holding time, fill model, and
23 bps cost are fixed by the canonical backtest and cannot appear in your JSON. The selected entry
condition is frozen and replayed unchanged on Validation and Final dates. Do not simplify ideas to
satisfy a mechanism story. Do not cite evidence. Return JSON only.

A raw threshold on a feature means the same number must be right for every symbol, and symbols
differ by orders of magnitude. The operator list below is the whole vocabulary — the schema example
shows the smallest possible expression, not a preferred one.

Each feature carries `observed` (its actual min/max/p1/p99) and `threshold_policy` (which kinds of
threshold it accepts, and what to use instead when it accepts none). Read both before writing any
number. A feature whose `threshold_policy.allowed` contains `execution_only`, `derived_duplicate`
or `derived_input_only` cannot be the direct target of a comparison — use the named alternative,
or use it as material inside a ratio or difference.

# Claims and variants

Split each hypothesis into atomic `claims` before writing any expression. One claim states one
observable thing. A claim is not an interpretation and not a prediction — "aggressive selling is
sustained" is a claim, "sellers are exhausted" is not.

A claim's `feature` is either a catalog name, or an expression that combines catalog features into
one quantity the catalog does not already hold. Combining is how you observe something the 36
features cannot say on their own:

  {{"op": "ratio", "numerator": {{"op": "primitive", "primitive_id": "bid_depth_total_5"}},
    "denominator": {{"op": "primitive", "primitive_id": "ask_depth_total_5"}},
    "zero_policy": "nan"}}
  {{"op": "subtract", "left": {{"op": "primitive", "primitive_id": "book_slope_bid"}},
    "right": {{"op": "primitive", "primitive_id": "book_slope_ask"}}}}

These two are shapes, not recommendations — the operators you combine with and the features you
combine are yours to choose. Only same-instant operators are allowed here ({claim_ops}); time
operators and comparisons belong to variants, and a claim is a quantity, not a condition.

A combined quantity has no `observed` range, so it must be cut with a `rolling_quantile` parameter,
never a literal number. That is also the point of combining: a literal has to be right for every
symbol at once, and a rolling quantile does not.

Then write `variants`: the same hypothesis expressed in genuinely different observable forms.

  LEVEL              a state right now, no temporal operator
  PRE_ANCHOR_CHANGE  a change against a fixed lag, uses `difference`
  PERSISTENCE        holds over a window, uses `persistence`
  SEQUENCE           one thing after another, uses `sequence`
  COMPOSITE          a combination

Each variant declares which claims it rests on. **Its expression may only use features named by
those claims** — that is checked mechanically, and so is the match between the declared
representation and the operators actually used. Two variants must not share a representation:
renaming an expression is not a variant.

At most {MAX_VARIANTS_PER_HYPOTHESIS} variants. Write fewer when the hypothesis genuinely has
only one honest form; a forced variant is worse than none.

`path_prediction` sits at the hypothesis level and is shared by every variant of it. It is a claim
about what the market does, not about how you wrote the expression. If two forms of the same
hypothesis would predict different ledgers, they are different hypotheses.

Every threshold that should be searched must be `UNRESOLVED:<name>` and have one matching
parameters entry. `rolling_quantile` values must be in [0,1). `literal` values may also replace
window, lag, level, min_true, or other numeric positions, so entry structure itself can be searched.
All rolling operators use `time_basis: tick`; difference and sequence may also use `clock`.
The whole parameter grid is enumerated, so its size is the number of backtests this variant costs.
Keep the product of all `values` lengths at or below {grid_cap}. A variant above that is rejected.

Every strategy must also state what the execution ledger will look like if the idea is right.
That is `path_prediction`. It is not a profit forecast — it describes the trade path, and it is
checked mechanically against the ledger after the backtest runs.

The execution is a BID1 limit buy that exits by a fixed canonical rule. Six ledger quantities
describe that path. Three are about the trade itself, three compare filled decisions against the
ones that never filled.

  bid_move_bps        BID1 at exit vs BID1 at fill. Did the side you bought on move?
  fill_slippage_bps   BID1 at fill vs BID1 at post. NEGATIVE means the market fell while your
                      order rested — you were filled by someone selling into you.
  loss_kind           Among the losing fills, which failure dominates:
                        PERSISTENT_ADVERSE  never rose at all
                        ROSE_THEN_LOST      rose but gave it back or never cleared 23 bps
                        MIXED               neither dominates

  fill_rate                   fills / decisions. HIGH above 20%.
  fill_spread_selection       Spread where you filled vs spread where you did not.
                              MORE = your fills land in wider books than your misses.
  fill_opportunity_selection  How far price rose after your fills vs after your misses.
                              LESS = the ones you caught were the worse opportunities.

Every comparison uses a {PA.MOVE_DEAD_ZONE_BPS:g} bps tie band; inside it the answer is SAME.

There is no preferred answer and no answer is penalised. Capturing the spread is a legitimate
mechanism: a spread-capture idea should say bid_move NEUTRAL and entry_spread WIDER. A directional
idea should say bid_move POSITIVE. Answer NEUTRAL or TYPICAL only where your idea genuinely makes
no claim on that field — these values are recorded, not scored against you.

State what your idea actually implies. A prediction the ledger contradicts is a useful result.
A prediction that could not have been wrong is not.

JSON schema example. The `entry_expression` here is the smallest expression the schema accepts,
shown to fix the JSON shape — not a template to follow:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Raw causal inputs:
{json.dumps(raw_inputs, ensure_ascii=False, indent=2)}

Catalog features, operators and their rules:
{json.dumps(_catalog_for_agent(), ensure_ascii=False, indent=1)}
"""


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _placeholder_names(expression: Mapping[str, Any]) -> set[str]:
    return {
        value.removeprefix(UNRESOLVED)
        for value in _walk(expression)
        if isinstance(value, str) and value.startswith(UNRESOLVED)
    }


def _feature_names(expression: Mapping[str, Any]) -> set[str]:
    return {
        str(value["primitive_id"])
        for value in _walk(expression)
        if isinstance(value, Mapping) and value.get("op") == "primitive"
    }


def _threshold_targets(expression: Mapping[str, Any]) -> set[str]:
    """임계 연산자의 **직접** 입력이 된 feature 이름.

    `compare(spread_bps, 20)` 은 잡고 `compare(ratio(spread_bps, x), 3)` 은 안 잡는다.
    정책이 금지하는 것은 절대 원값 조건이지 재료로 쓰는 것이 아니다.
    """
    out: set[str] = set()
    for node in _walk(expression):
        if not isinstance(node, Mapping) or node.get("op") not in {"compare", "crossover"}:
            continue
        target = node.get("input")
        if isinstance(target, Mapping) and target.get("op") == "primitive":
            out.add(str(target["primitive_id"]))
    return out


def _combined_threshold_problems(expression: Mapping[str, Any],
                                 kinds: Mapping[str, str]) -> list[str]:
    """조합해서 만든 관측은 rolling_quantile 로만 자른다.

    Catalog feature 에는 실측 범위(`observed`)가 있어 절대 숫자를 쓸 근거가 되지만,
    조합해서 만든 수량에는 그 범위가 없다. 게다가 조합을 쓰는 이유가 "숫자 하나가 모든
    종목에서 맞아야 하는" 문제를 피하려는 것인데, 거기에 다시 절대 숫자를 적으면 원래
    문제로 돌아간다. rolling_quantile 은 그 수량의 직전 100틱 분포에서 자르므로 절대
    크기를 몰라도 된다.
    """
    problems: list[str] = []
    for node in _walk(expression):
        if not isinstance(node, Mapping) or node.get("op") not in {"compare", "crossover"}:
            continue
        target = node.get("input")
        if not isinstance(target, Mapping):
            continue
        combining = catalog.expression_ops(target) - {"primitive", "raw", "level_aggregate"}
        if not combining:
            continue
        cut = node.get("value") if node.get("op") == "compare" else node.get("threshold")
        name = (str(cut).removeprefix(catalog.UNRESOLVED_PREFIX)
                if isinstance(cut, str) and str(cut).startswith(catalog.UNRESOLVED_PREFIX)
                else None)
        if name is None or kinds.get(name) != "rolling_quantile":
            problems.append(
                f"조합해서 만든 관측({sorted(combining)})을 절대 숫자로 자른다: {cut!r}. "
                "rolling_quantile 파라미터로 자르라 — 그 값의 실측 범위가 없다")
    return problems


def _causal_problems(expression: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    try:
        catalog.validate_expression(expression, allow_unresolved=True)
        inferred = catalog.infer_expression_type(expression, allow_unresolved=True)
        if inferred.value_type not in {"boolean", "event"}:
            problems.append("entry_expression 결과가 boolean/event가 아니다")
    except (KeyError, TypeError, ValueError) as error:
        problems.append(f"entry_expression DSL 오류: {error}")
    for feature in sorted(_feature_names(expression)):
        try:
            resolved = catalog.resolve(feature)
        except (KeyError, ValueError):
            problems.append(f"Catalog에 없는 feature: {feature}")
            continue
        if resolved not in catalog.FEATURES:
            problems.append(f"Catalog에 없는 feature: {feature}")
            continue
        if not catalog.FEATURES[resolved].causal:
            problems.append(f"미래 feature는 사용할 수 없다: {feature}")
    for feature in sorted(_threshold_targets(expression)):
        problem = catalog.condition_problem(feature)
        if problem:
            problems.append(problem)
    for node in _walk(expression):
        if not isinstance(node, Mapping):
            continue
        op = str(node.get("op") or "")
        if op == "difference" and float(node.get("lag", 0)) <= 0:
            problems.append("difference lag는 0보다 커야 한다")
        if op in {"rolling_zscore", "rolling_sum", "rolling_mean", "rolling_std",
                  "rolling_min", "rolling_max", "persistence"}:
            window = int(node.get("window", 0))
            if window < 1:
                problems.append(f"{op} window는 1 이상이어야 한다")
            minimum = int(node.get("min_observations", window))
            if minimum < 1 or minimum > window:
                problems.append(f"{op} min_observations는 1..window여야 한다")
            if op == "persistence" and not 1 <= int(node.get("min_true", 0)) <= window:
                problems.append("persistence min_true는 1..window여야 한다")
        if op == "sequence":
            minimum, maximum = float(node.get("min_lag", -1)), float(node.get("max_lag", -1))
            if minimum < 0 or maximum < minimum:
                problems.append("sequence lag는 0 <= min_lag <= max_lag여야 한다")
    return problems


MAX_VARIANTS_PER_HYPOTHESIS = 4


def _claim_problems(raw: Mapping[str, Any]) -> tuple[dict[str, frozenset[str]], list[str]]:
    """가설을 원자 주장으로 쪼갠 것. 각 주장은 하나의 관측을 가리킨다.

    자유 문장 `rationale` 만으로는 "이 식이 그 가설을 표현하나" 를 기계로 물을 수 없다.
    주장마다 feature 를 적게 하면 식에 쓰인 feature 와 대조할 수 있다.
    """
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        return {}, ["claims가 비었다. 가설을 원자 주장으로 쪼개야 한다"]
    out: dict[str, frozenset[str]] = {}
    problems: list[str] = []
    for item in claims:
        if not isinstance(item, Mapping):
            problems.append("claim이 객체가 아니다")
            continue
        claim_id = str(item.get("claim_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not claim_id or claim_id in out:
            problems.append(f"claim_id가 없거나 중복됐다: {claim_id!r}")
            continue
        if not text:
            problems.append(f"{claim_id}.text가 비었다")
        names, trouble = _claim_features(claim_id, item.get("feature"))
        if trouble:
            problems.extend(trouble)
            continue
        out[claim_id] = names
    return out, problems


def _claim_features(claim_id: str, feature: Any) -> tuple[frozenset[str], list[str]]:
    """claim 이 지목한 관측의 feature 들.

    Catalog 이름 하나이거나, 그 이름들을 같은 순간에 합친 식이다. 합치면 카탈로그에
    없던 관측을 만들 수 있다 — `ratio(bid_depth_total_5, ask_depth_total_5)` 는
    "매수 깊이가 매도보다 두껍다" 를 종목 크기와 무관하게 재는 하나의 수량이다.
    """
    if isinstance(feature, str):
        name = feature.strip()
        try:
            resolved = catalog.resolve(name)
        except (KeyError, ValueError):
            return frozenset(), [f"{claim_id}.feature가 Catalog에 없다: {name!r}"]
        if resolved not in catalog.FEATURES:
            return frozenset(), [f"{claim_id}.feature가 Catalog에 없다: {name!r}"]
        return frozenset({resolved}), []
    if not isinstance(feature, Mapping):
        return frozenset(), [f"{claim_id}.feature가 Catalog 이름도 조합식도 아니다"]
    ops = catalog.expression_ops(feature)
    outside = sorted(ops - catalog.CLAIM_OPS)
    if outside:
        return frozenset(), [
            f"{claim_id}.feature 조합에 쓸 수 없는 연산이 있다: {outside}. "
            f"claim 은 같은 순간의 수량이다. 시간 연산과 비교는 variant 쪽이다"]
    try:
        catalog.validate_expression(feature, allow_unresolved=False)
        if catalog.infer_expression_type(feature, allow_unresolved=False).value_type != "numeric":
            return frozenset(), [f"{claim_id}.feature 조합이 수량이 아니다"]
    except (KeyError, TypeError, ValueError) as error:
        return frozenset(), [f"{claim_id}.feature 조합 DSL 오류: {error}"]
    names: set[str] = set()
    for name in _feature_names(feature):
        try:
            resolved = catalog.resolve(name)
        except (KeyError, ValueError):
            return frozenset(), [f"{claim_id}.feature 조합에 Catalog에 없는 feature: {name}"]
        if resolved not in catalog.FEATURES:
            return frozenset(), [f"{claim_id}.feature 조합에 Catalog에 없는 feature: {name}"]
        if not catalog.FEATURES[resolved].causal:
            return frozenset(), [f"{claim_id}.feature 조합에 미래를 보는 feature: {resolved}"]
        names.add(resolved)
    if not names:
        return frozenset(), [f"{claim_id}.feature 조합이 feature를 하나도 안 쓴다"]
    return frozenset(names), []


def _variant_problems(raw: Mapping[str, Any], claims: Mapping[str, str]) -> list[str]:
    """variant 는 같은 가설의 **다른 표현**이어야 한다. 이름만 바꾼 것은 거절한다."""
    variants = raw.get("variants")
    if not isinstance(variants, list) or not variants:
        return ["variants가 비었다"]
    problems: list[str] = []
    if len(variants) > MAX_VARIANTS_PER_HYPOTHESIS:
        problems.append(f"variant는 최대 {MAX_VARIANTS_PER_HYPOTHESIS}개다")
    seen_ids: set[str] = set()
    seen_representations: set[str] = set()
    for item in variants:
        if not isinstance(item, Mapping):
            problems.append("variant가 객체가 아니다")
            continue
        vid = str(item.get("variant_id") or "").strip()
        label = f"variant[{vid or '?'}]"
        if not vid or vid in seen_ids:
            problems.append(f"variant_id가 없거나 중복됐다: {vid!r}")
            continue
        seen_ids.add(vid)
        representation = str(item.get("representation") or "")
        if representation not in catalog.REPRESENTATIONS:
            problems.append(f"{label}.representation이 허용값 밖이다: {representation!r}")
        elif representation in seen_representations:
            problems.append(
                f"{label}.representation이 다른 variant와 같다: {representation}. "
                "같은 가설의 다른 표현이어야 한다")
        else:
            seen_representations.add(representation)
        expression = item.get("entry_expression")
        if not isinstance(expression, Mapping):
            problems.append(f"{label}.entry_expression이 객체가 아니다")
            continue
        if (representation in catalog.REPRESENTATIONS
                and not catalog.representation_matches(representation, expression)):
            problems.append(
                f"{label}이 {representation}이라고 했는데 식이 그 형태가 아니다: "
                f"{sorted(catalog.expression_ops(expression))}")
        cited = [str(c) for c in (item.get("claim_ids") or [])]
        missing = sorted(set(cited) - set(claims))
        if not cited:
            problems.append(f"{label}.claim_ids가 비었다")
        elif missing:
            problems.append(f"{label}.claim_ids가 없는 claim을 가리킨다: {missing}")
        # 말과 식의 기계적 대조 — 식에 쓴 feature 는 인용한 주장의 feature 여야 한다.
        allowed = {name for c in cited if c in claims for name in claims[c]}
        used = {catalog.resolve(name) for name in _feature_names(expression)}
        extra = sorted(used - allowed)
        if extra and allowed:
            problems.append(
                f"{label}의 식이 인용하지 않은 주장의 feature를 쓴다: {extra}. "
                f"claim_ids에 넣거나 식에서 빼라")
    return problems


def _path_prediction_problems(raw: Mapping[str, Any], *, required: bool) -> list[str]:
    """예측은 원장 열과 1:1 인 범주값이어야 채점된다. 자유 문장은 받지 않는다."""
    value = raw.get("path_prediction")
    if value is None:
        return ["path_prediction이 없다"] if required else []
    if not isinstance(value, Mapping):
        return ["path_prediction이 객체가 아니다"]
    problems: list[str] = []
    for name, allowed in PA.FIELDS.items():
        actual = value.get(name)
        if actual is None:
            problems.append(f"path_prediction에 {name}이 없다")
        elif str(actual) not in allowed:
            problems.append(f"path_prediction.{name}이 허용값 밖이다: {actual!r}")
    for name in sorted(set(value) - set(PA.FIELDS)):
        problems.append(f"path_prediction에 모르는 항목이 있다: {name}")
    return problems


def validate_strategy(raw: Mapping[str, Any], *,
                      require_path_prediction: bool = True) -> tuple[dict[str, Any] | None,
                                                                     list[str]]:
    problems: list[str] = []
    strategy_id = str(raw.get("strategy_id") or "").strip()
    if not strategy_id:
        problems.append("strategy_id가 없다")
    expression = raw.get("entry_expression")
    if not isinstance(expression, Mapping):
        problems.append("entry_expression이 객체가 아니다")
        return None, problems
    if "execution_grid" in raw:
        problems.append("execution_grid는 사용할 수 없다. 정본 실행 규칙이 고정된다")
    if "side" in raw or "exit_expression" in raw:
        problems.append("side와 청산식은 전략 입력이 아니다. LONG과 정본 청산식이 고정된다")

    parameters: dict[str, dict[str, Any]] = {}
    for item in raw.get("parameters") or []:
        if not isinstance(item, Mapping):
            problems.append("parameter가 객체가 아니다")
            continue
        name, kind = str(item.get("name") or ""), str(item.get("kind") or "")
        values = list(item.get("values") or [])
        if not name or name in parameters:
            problems.append(f"parameter 이름이 없거나 중복됐다: {name!r}")
            continue
        if kind not in PARAMETER_KINDS:
            problems.append(f"지원하지 않는 parameter kind: {kind!r}")
            continue
        if not values:
            problems.append(f"parameter grid가 비었다: {name}")
            continue
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError):
            problems.append(f"parameter 값이 숫자가 아니다: {name}")
            continue
        if any(isinstance(value, bool) for value in values) or not all(
                math.isfinite(value) for value in floats):
            problems.append(f"parameter 값이 유한하지 않다: {name}")
            continue
        numeric = [int(value) if value.is_integer() else value for value in floats]
        if kind == "rolling_quantile" and not all(0.0 <= value < 1.0 for value in floats):
            problems.append(f"rolling_quantile은 0 이상 1 미만이어야 한다: {name}")
        parameters[name] = {"name": name, "kind": kind, "values": numeric}
    problems.extend(_path_prediction_problems(raw, required=require_path_prediction))
    placeholders = _placeholder_names(expression)
    if placeholders != set(parameters):
        problems.append(
            f"UNRESOLVED 자리와 parameter가 다르다: expression={sorted(placeholders)}, "
            f"parameters={sorted(parameters)}")
    problems.extend(_combined_threshold_problems(
        expression, {name: str(item["kind"]) for name, item in parameters.items()}))
    literal_parameters = [item for item in parameters.values() if item["kind"] == "literal"]
    names = [str(item["name"]) for item in literal_parameters]
    grids = [list(item["values"]) for item in literal_parameters]
    combinations = itertools.product(*grids) if grids else [()]
    for combination in combinations:
        candidate = _replace_literals(expression, dict(zip(names, combination)))
        problems.extend(_causal_problems(candidate))
    problems = list(dict.fromkeys(problems))
    if problems:
        return None, problems
    return clean({
        "strategy_id": strategy_id,
        "title": str(raw.get("title") or strategy_id),
        "rationale": str(raw.get("rationale") or ""),
        "entry_expression": dict(expression),
        "parameters": [parameters[name] for name in sorted(parameters)],
        "path_prediction": ({name: str(dict(raw["path_prediction"])[name])
                             for name in PA.FIELDS}
                            if isinstance(raw.get("path_prediction"), Mapping) else None),
        # 가설 계보. 같은 hypothesis_id 를 가진 전략들은 한 가설의 다른 표현이다.
        "hypothesis_id": (str(raw["hypothesis_id"]) if raw.get("hypothesis_id") else None),
        "representation": (str(raw["representation"]) if raw.get("representation") else None),
        "claims": ([dict(c) for c in raw["claims"]] if isinstance(raw.get("claims"), list)
                   else None),
        "claim_ids": ([str(c) for c in raw["claim_ids"]]
                      if isinstance(raw.get("claim_ids"), list) else None),
    }), []


def expand_hypothesis(raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """가설 하나를 variant 마다 전략 하나로 펼친다.

    아래 단계(Optuna·Refinement·동결·재생)는 그대로 둔다. 펼친 전략은 예전과 같은
    모양이고, `hypothesis_id`·`representation`·`claims` 로 계보만 더 붙는다.
    같은 가설의 variant 들은 `path_prediction` 을 공유한다 — 예측은 가설 수준의
    주장이지 표현 수준의 주장이 아니다.
    """
    base = str(raw.get("strategy_id") or "").strip()
    if not base:
        return [], ["strategy_id가 없다"]
    claims, problems = _claim_problems(raw)
    problems = list(problems) + _variant_problems(raw, claims)
    if problems:
        return [], problems
    shared = {
        "title": str(raw.get("title") or base),
        "rationale": str(raw.get("rationale") or ""),
        "claims": [dict(item) for item in raw["claims"]],
        "path_prediction": raw.get("path_prediction"),
    }
    out: list[dict[str, Any]] = []
    for item in raw["variants"]:
        vid = str(item["variant_id"])
        out.append({
            **shared,
            "strategy_id": f"{base}__{vid}",
            "hypothesis_id": base,
            "representation": str(item["representation"]),
            "claim_ids": [str(c) for c in item.get("claim_ids") or []],
            "entry_expression": dict(item["entry_expression"]),
            "parameters": list(item.get("parameters") or []),
        })
    return out, []


def _replace_literals(value: Any, literals: Mapping[str, int | float]) -> Any:
    if isinstance(value, str) and value.startswith(UNRESOLVED):
        name = value.removeprefix(UNRESOLVED)
        return literals[name] if name in literals else value
    if isinstance(value, Mapping):
        return {key: _replace_literals(child, literals) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_literals(child, literals) for child in value]
    return copy.deepcopy(value)


def _warmup_ticks(expression: Mapping[str, Any], rolling: bool) -> int:
    warmup = 100 if rolling else 0
    for node in _walk(expression):
        if not isinstance(node, Mapping):
            continue
        op = str(node.get("op") or "")
        if op == "difference" and str(node.get("time_basis")) == "tick":
            warmup = max(warmup, int(node.get("lag") or 0))
        if op in {"rolling_zscore", "rolling_sum", "rolling_mean", "rolling_std",
                  "rolling_min", "rolling_max", "persistence"} \
                and str(node.get("time_basis")) == "tick":
            warmup = max(warmup, int(node.get("window") or 0))
        if op == "sequence" and str(node.get("time_basis")) == "tick":
            warmup = max(warmup, int(node.get("max_lag") or 0))
    return warmup


def _contract(strategy: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
    kinds = {str(item["name"]): str(item["kind"]) for item in strategy["parameters"]}
    literals = {name: value for name, value in values.items() if kinds[name] == "literal"}
    expression = _replace_literals(strategy["entry_expression"], literals)
    interfaces = {}
    for name, kind in kinds.items():
        if kind != "rolling_quantile":
            continue
        interfaces[name] = {
            "status": "SEARCHABLE", "value": None, "role": "signal_threshold",
            "threshold_source": {"kind": catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE},
        }
    template = {
        "schema": "free_strategy_contract.v1",
        "hypothesis_id": str(strategy["strategy_id"]),
        "side": "LONG",
        "entry_program": {
            "signal": expression,
            "warmup_ticks": _warmup_ticks(expression, bool(interfaces)),
        },
        "entry_lifecycle": {"mode": "HOLD_THROUGH", "pending_observable_ids": []},
        "parameter_interface": interfaces,
        "profit_target": PT.canonical_target(),
        "provenance": {"source": "AGENT_OWN_KNOWLEDGE", "strategy_sha256": sha256_json(strategy)},
    }
    template["contract_sha256"] = contract.contract_hash(template)
    return template


def _grid_size(strategy: Mapping[str, Any]) -> int:
    sizes = [len(item["values"]) for item in strategy["parameters"]]
    return math.prod(sizes) if sizes else 1


def _suggest_parameters(trial: optuna.Trial,
                        strategy: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        str(item["name"]): trial.suggest_categorical(str(item["name"]), list(item["values"]))
        for item in strategy["parameters"]
    }


def _frozen_strategy(strategy: Mapping[str, Any],
                     values: Mapping[str, int | float]) -> dict[str, Any]:
    variant_id = sha256_json({"parameters": values})[:12]
    return {
        "variant_id": variant_id,
        "strategy": strategy,
        "parameters": dict(values),
        "contract": _contract(strategy, values),
        "execution_profile": copy.deepcopy(K.CANONICAL),
    }


def _objective(summary: Mapping[str, Any]) -> float | None:
    if summary.get("state") != "BACKTEST_COMPLETE":
        return None
    return float((summary.get("metrics") or {}).get("net_bps_total") or 0.0)


def _read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def _tell(study: optuna.Study, trial_number: int, value: float | None = None,
          state: TrialState | None = None) -> None:
    try:
        study.tell(trial_number, value, state=state)
    except RuntimeError as error:
        # GridSampler calls Study.stop() after its last grid point. Optuna's
        # ask/tell API has already stored that trial before stop() rejects this context.
        stored = study.trials[trial_number]
        if "Study.stop" not in str(error) or not stored.state.is_finished():
            raise


def _recover_running_trials(study: optuna.Study) -> None:
    """Finish interrupted trials from their summary, or mark them failed."""
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        summary_path = trial.user_attrs.get("summary_path")
        summary = _read_summary(Path(summary_path)) if summary_path else None
        value = _objective(summary or {})
        if value is None:
            _tell(study, trial.number, state=TrialState.FAIL)
        else:
            _tell(study, trial.number, value)


def _study_snapshot(study: optuna.Study, *, sampler_name: str, seed: int,
                    requested_trials: int, trial_budget: int, grid_size: int,
                    processes: int) -> dict[str, Any]:
    trials = [{
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "parameters": trial.params,
        "variant_id": trial.user_attrs.get("variant_id"),
        "summary_path": trial.user_attrs.get("summary_path"),
    } for trial in study.get_trials(deepcopy=False)]
    complete = [trial for trial in trials if trial["state"] == "COMPLETE"]
    best = max(complete, key=lambda item: float(item["value"])) if complete else None
    return {
        "schema": "optuna_discovery_study.v1",
        "study_name": study.study_name,
        "sampler": sampler_name,
        "seed": int(seed),
        "requested_trials": int(requested_trials),
        "trial_budget": int(trial_budget),
        "total_grid_size": int(grid_size),
        "processes": int(processes),
        "trials": trials,
        "best_trial_number": best["number"] if best else None,
        "best_variant_id": best["variant_id"] if best else None,
    }


def _optimize_strategy(strategy: Mapping[str, Any], request: FreeResearchRequest,
                       symbols: Sequence[str], output: Path, replay: Replay,
                       root: Path, runtime: Mapping[str, Any], *,
                       unit_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    strategy_id = str(strategy["strategy_id"])
    strategy_dir = (Path(unit_root) if unit_root is not None else
                    Path(output) / "02_discovery" / "units" / strategy_id)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    grid_size = _grid_size(strategy)
    requested_trials = int(request.trials_per_strategy)
    # 격자를 전수로 돈다. TPE는 초반 10회가 무작위라 작은 격자에서 같은 점을 되뽑아
    # 예산만 태웠다 (729칸 격자에 50시도로 서로 다른 점 17~19개, 한 점이 22회까지 반복).
    # 격자가 requested_trials를 넘는 전략은 생성 단계에서 이미 걸러진다.
    trial_budget = grid_size
    search_space = {
        str(item["name"]): list(item["values"])
        for item in strategy["parameters"]
    }
    if search_space:
        sampler: optuna.samplers.BaseSampler = optuna.samplers.GridSampler(
            search_space, seed=int(request.optuna_seed))
        sampler_name = "GridSampler"
    else:
        sampler = optuna.samplers.TPESampler(seed=int(request.optuna_seed))
        sampler_name = "TPESampler"
    journal_path = strategy_dir / "optuna_journal.log"
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    study_name = f"{strategy_id}-{sha256_json(strategy)[:12]}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    immutable = {
        "strategy_sha256": sha256_json(strategy),
        "search_space_sha256": sha256_json(search_space),
        "discovery_scope_sha256": sha256_json({
            "symbols": list(map(str, symbols)),
            "dates": list(map(str, request.discovery_dates)),
            "tick_root": str(Path(root).resolve()),
            "canonical_profile_sha256": K.CANONICAL["profile_sha256"],
            "framework_code_sha256": str(runtime["framework_code_sha256"]),
        }),
        "seed": int(request.optuna_seed),
        "sampler": sampler_name,
    }
    for key, expected in immutable.items():
        actual = study.user_attrs.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"기존 Optuna study의 {key}가 현재 요청과 다르다")
        study.set_user_attr(key, expected)
    _recover_running_trials(study)

    use_batch = replay is replay_variant and int(request.optuna_processes) > 1
    processes = min(int(request.optuna_processes), trial_budget) if use_batch else 1
    inner_workers = int(request.workers) if use_batch else max(1, int(request.workers))
    summary_by_variant: dict[str, dict[str, Any]] = {}
    frozen_by_variant: dict[str, dict[str, Any]] = {}
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        variant_id = trial.user_attrs.get("variant_id")
        summary_path = trial.user_attrs.get("summary_path")
        summary = _read_summary(Path(summary_path)) if summary_path else None
        if variant_id and summary is not None:
            summary_by_variant[str(variant_id)] = summary
            frozen_by_variant[str(variant_id)] = _frozen_strategy(strategy, trial.params)

    while len(study.trials) < trial_budget:
        batch_size = min(processes, trial_budget - len(study.trials))
        batch: list[tuple[optuna.Trial, str, dict[str, Any], Path]] = []
        pending: dict[str, tuple[dict[str, Any], Path]] = {}
        for _ in range(batch_size):
            trial = study.ask()
            values = _suggest_parameters(trial, strategy)
            frozen = _frozen_strategy(strategy, values)
            variant_id = str(frozen["variant_id"])
            variant_dir = strategy_dir / "variants" / variant_id
            summary_path = variant_dir / "summary.json"
            trial.set_user_attr("variant_id", variant_id)
            trial.set_user_attr("summary_path", str(summary_path))
            frozen_by_variant[variant_id] = frozen
            batch.append((trial, variant_id, frozen, summary_path))
            if variant_id in summary_by_variant or variant_id in pending:
                continue
            pending[variant_id] = (frozen, summary_path)

        if use_batch and pending:
            batch_root = strategy_dir / "batches" / (
                f"{min(item[0].number for item in batch):04d}_"
                f"{max(item[0].number for item in batch):04d}")
            try:
                summaries = replay_variant_batch(
                    {variant_id: frozen for variant_id, (frozen, _path) in pending.items()},
                    symbols, request.discovery_dates, batch_root, inner_workers, Path(root))
            except Exception as error:
                summaries = {
                    variant_id: {"state": "BACKTEST_ERROR", "errors": [repr(error)]}
                    for variant_id in pending
                }
            for variant_id, (_frozen, summary_path) in pending.items():
                summary = summaries[variant_id]
                write_json(summary_path, summary)
                summary_by_variant[variant_id] = summary
        else:
            for variant_id, (frozen, summary_path) in pending.items():
                try:
                    summary = replay(
                        frozen, symbols, request.discovery_dates,
                        summary_path.parent, inner_workers, Path(root))
                except Exception as error:
                    summary = {"state": "BACKTEST_ERROR", "errors": [repr(error)]}
                write_json(summary_path, summary)
                summary_by_variant[variant_id] = summary

        for trial, variant_id, _frozen, _summary_path in sorted(
                batch, key=lambda item: item[0].number):
            value = _objective(summary_by_variant[variant_id])
            if value is None:
                _tell(study, trial.number, state=TrialState.FAIL)
            else:
                _tell(study, trial.number, value)
        write_json(
            strategy_dir / "optuna_study.json",
            _study_snapshot(
                study, sampler_name=sampler_name, seed=request.optuna_seed,
                requested_trials=requested_trials, trial_budget=trial_budget,
                grid_size=grid_size, processes=processes))
    write_json(
        strategy_dir / "optuna_study.json",
        _study_snapshot(
            study, sampler_name=sampler_name, seed=request.optuna_seed,
            requested_trials=requested_trials, trial_budget=trial_budget,
            grid_size=grid_size, processes=processes))

    candidates = []
    seen_variants: set[str] = set()
    for trial in study.get_trials(deepcopy=False):
        variant_id = str(trial.user_attrs.get("variant_id") or "")
        if not variant_id or variant_id in seen_variants:
            continue
        summary = summary_by_variant.get(variant_id)
        if summary is None:
            summary_path = trial.user_attrs.get("summary_path")
            summary = _read_summary(Path(summary_path)) if summary_path else None
        if summary is None:
            continue
        seen_variants.add(variant_id)
        candidates.append({
            "trial_number": trial.number,
            "variant_id": variant_id,
            "parameters": dict(trial.params),
            "summary": summary,
            "frozen": frozen_by_variant.get(variant_id) or _frozen_strategy(strategy, trial.params),
        })
    completed = [item for item in candidates if _objective(item["summary"]) is not None]
    ranked = sorted(completed, key=lambda item: (
        -float(_objective(item["summary"]) or 0.0),
        -int((item["summary"].get("metrics") or {}).get("scorable") or 0),
        str(item["variant_id"]),
    ))
    selected = ranked[0] if ranked else None
    unit = {
        "state": "DISCOVERY_COMPLETE" if selected else "NO_VARIANT",
        "optimizer": "OPTUNA",
        "sampler": sampler_name,
        "seed": int(request.optuna_seed),
        "requested_trials": requested_trials,
        "trial_budget": trial_budget,
        "completed_trials": sum(
            trial.state == TrialState.COMPLETE for trial in study.get_trials(deepcopy=False)),
        "total_grid_size": grid_size,
        "processes": processes,
        "journal": str(journal_path),
        "study_snapshot": str(strategy_dir / "optuna_study.json"),
        "candidates": [{key: value for key, value in item.items() if key != "frozen"}
                       for item in candidates],
        "selected_trial_number": selected["trial_number"] if selected else None,
        "selected_variant_id": selected["variant_id"] if selected else None,
        "selection_metric": "net_bps_total",
    }
    frozen = ({**selected["frozen"], "discovery_summary": selected["summary"]}
              if selected else None)
    return unit, frozen


def _parameter_table(strategy: Mapping[str, Any], values: Mapping[str, float],
                     symbols: Sequence[str], dates: Sequence[str]) -> dict[str, dict[str, float]]:
    rolling = {
        str(item["name"]): float(values[str(item["name"])])
        for item in strategy["parameters"] if item["kind"] == "rolling_quantile"
    }
    return {f"{symbol}:{date}": dict(rolling) for symbol in symbols for date in dates} if rolling else {}


def replay_variant(frozen: Mapping[str, Any], symbols: Sequence[str], dates: Sequence[str],
                   output: Path, workers: int, root: Path) -> dict[str, Any]:
    """Replay one complete strategy with actual quotes and 23 bps accounting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    strategy = frozen["strategy"]
    values = frozen["parameters"]
    template = frozen.get("contract") or _contract(strategy, values)
    profile = copy.deepcopy(K.CANONICAL)
    contract_id = str(strategy["strategy_id"])
    ledger_path = output / "ledger.parquet"
    manifest = K.run_backtest(
        {contract_id: template}, {contract_id: list(symbols)}, list(dates), ledger_path,
        canonical=profile,
        parameter_table=_parameter_table(strategy, values, symbols, dates),
        workers=int(workers), root=Path(root), strict_contract=True)
    if manifest.get("errors"):
        result = {"state": "BACKTEST_ERROR", "errors": list(manifest["errors"]),
                  "ledger": str(ledger_path)}
        write_json(output / "summary.json", result)
        return result
    frame = pd.read_parquet(ledger_path)
    ledger.verify_invariants(frame, manifest)
    measured = K.metric_report(frame, manifest)
    accounting = K.spread_accounting_audit(frame, manifest, profile)
    result = {
        "state": "BACKTEST_COMPLETE",
        "dates": list(map(str, dates)),
        "ledger": str(ledger_path),
        "metrics": measured,
        "accounting": accounting,
        "profit_target_evaluation": PT.evaluate(measured),
        # 가설이 미리 말한 경로와 원장이 보여준 경로의 대조. 기록이며 선택에 쓰지 않는다.
        "path_alignment": PA.evaluate(frame, strategy.get("path_prediction")),
        "variant_id": str(frozen["variant_id"]),
        "strategy_sha256": sha256_json(strategy),
        "contract_sha256": str(template["contract_sha256"]),
        "frozen_strategy_sha256": sha256_json(frozen),
        "execution_profile_id": profile["profile_id"],
        "execution_profile_sha256": profile["profile_sha256"],
    }
    write_json(output / "summary.json", result)
    return result


def replay_variant_batch(frozen_by_variant: Mapping[str, Mapping[str, Any]],
                         symbols: Sequence[str], dates: Sequence[str], output: Path,
                         workers: int, root: Path) -> dict[str, dict[str, Any]]:
    """Optuna 후보들을 한 원본 로드로 재생하고 후보별 지표를 돌려준다."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract_ids = {
        str(variant_id): (
            f"{frozen['strategy']['strategy_id']}__V{variant_id}")
        for variant_id, frozen in frozen_by_variant.items()
    }
    contracts = {
        contract_ids[str(variant_id)]: frozen["contract"]
        for variant_id, frozen in frozen_by_variant.items()
    }
    members = {contract_id: list(symbols) for contract_id in contracts}
    parameter_table: dict[str, dict[str, float]] = {}
    for variant_id, frozen in frozen_by_variant.items():
        contract_id = contract_ids[str(variant_id)]
        rolling = {
            str(item["name"]): float(frozen["parameters"][str(item["name"])])
            for item in frozen["strategy"]["parameters"]
            if str(item["kind"]) == "rolling_quantile"
        }
        for symbol in symbols:
            for date in dates:
                if rolling:
                    parameter_table[f"{contract_id}:{symbol}:{date}"] = dict(rolling)

    ledger_path = output / "ledger.parquet"
    manifest = K.run_backtest(
        contracts, members, list(dates), ledger_path,
        canonical=K.CANONICAL,
        parameter_table=parameter_table,
        workers=int(workers), root=Path(root), strict_contract=True)
    frame = pd.read_parquet(ledger_path)
    ledger.verify_invariants(frame, manifest)

    summaries: dict[str, dict[str, Any]] = {}
    for variant_id, frozen in frozen_by_variant.items():
        contract_id = contract_ids[str(variant_id)]
        unit_frame = frame.loc[frame["contract_id"].eq(contract_id)].copy()
        unit_manifest = backtest_module._contract_manifest(manifest, unit_frame, contract_id)
        if unit_manifest.get("errors"):
            summaries[str(variant_id)] = {
                "state": "BACKTEST_ERROR",
                "errors": list(unit_manifest["errors"]),
                "ledger": str(ledger_path),
                "ledger_contract_id": contract_id,
            }
            continue
        measured = K.metric_report(unit_frame, unit_manifest)
        summaries[str(variant_id)] = {
            "state": "BACKTEST_COMPLETE",
            "dates": list(map(str, dates)),
            "ledger": str(ledger_path),
            "ledger_contract_id": contract_id,
            "metrics": measured,
            "accounting": K.spread_accounting_audit(unit_frame, unit_manifest, K.CANONICAL),
            "profit_target_evaluation": PT.evaluate(measured),
            "path_alignment": PA.evaluate(
                unit_frame, (frozen.get("strategy") or {}).get("path_prediction")),
            "variant_id": str(frozen["variant_id"]),
            "strategy_sha256": sha256_json(frozen["strategy"]),
            "contract_sha256": str(frozen["contract"]["contract_sha256"]),
            "frozen_strategy_sha256": sha256_json(frozen),
            "execution_profile_id": K.CANONICAL["profile_id"],
            "execution_profile_sha256": K.CANONICAL["profile_sha256"],
        }
    return summaries


def _load_or_generate(request: FreeResearchRequest, output: Path,
                      runner: Runner | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if request.strategy_source is not None:
        return read_json(Path(request.strategy_source)), []
    response_path = Path(output) / "strategy_agent_response.json"
    invocation_path = Path(output) / "strategy_agent_invocation.json"
    request_path = Path(output) / "strategy_agent_request.json"
    prompt = generation_prompt(request.strategy_count, int(request.trials_per_strategy))
    request_identity = {
        "schema": "knowledge_strategy_agent_request.v1",
        "model": str(request.model),
        "effort": str(request.effort),
        "strategy_count": int(request.strategy_count),
        "prompt_sha256": sha256_json({"prompt": prompt}),
    }
    if response_path.is_file():
        if not request_path.is_file() or read_json(request_path) != request_identity:
            raise ValueError(
                "기존 Knowledge Strategy Agent 응답의 prompt/model identity가 현재 요청과 다르다")
        invocations = read_json(invocation_path) if invocation_path.is_file() else []
        return read_json(response_path), list(invocations)
    write_json(request_path, request_identity)
    response, invocation = invoke(
        runner, role="knowledge_strategy_generation", prompt=prompt,
        model=request.model, effort=request.effort)
    write_json(response_path, response)
    write_json(invocation_path, [invocation.as_dict()])
    return response, [invocation.as_dict()]


def _refinement_ledger_path(summary: Mapping[str, Any]) -> Path:
    value = summary.get("ledger")
    if not value:
        raise ValueError("Refinement 부모 Discovery summary에 ledger가 없다")
    path = Path(str(value))
    if path.is_file():
        return path
    repository_path = REPO / path
    if repository_path.is_file():
        return repository_path
    raise FileNotFoundError(f"Refinement 부모 Discovery ledger를 찾을 수 없다: {value}")


def _refinement_contract(frozen: Mapping[str, Any]) -> dict[str, Any]:
    strategy = frozen["strategy"]
    template = frozen["contract"]
    return {
        "hypothesis_id": str(strategy["strategy_id"]),
        "title": str(strategy.get("title") or strategy["strategy_id"]),
        "rationale": str(strategy.get("rationale") or ""),
        "side": str(template["side"]),
        "entry_expression": copy.deepcopy(template["entry_program"]["signal"]),
        "selected_parameters": copy.deepcopy(frozen["parameters"]),
        "fixed_exit_profile": {
            "profile_id": K.CANONICAL["profile_id"],
            "profile_sha256": K.CANONICAL["profile_sha256"],
        },
        "round_trip_cost_bps": float(catalog.FEE_BPS),
        "entry_replacement_scope": "COMPLETE_EXPRESSION",
        "path_prediction": copy.deepcopy(strategy.get("path_prediction")),
    }


def _refinement_loss_profile(frozen: Mapping[str, Any], population: str, *,
                             root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = frozen.get("discovery_summary") or {}
    ledger_path = _refinement_ledger_path(summary)
    frame = pd.read_parquet(ledger_path)
    ledger_contract_id = summary.get("ledger_contract_id")
    if ledger_contract_id is not None:
        frame = frame.loc[frame["contract_id"].eq(str(ledger_contract_id))].copy()
    if population == "FILLED_ONLY":
        frame = frame.loc[frame["status"].eq(ledger.FILLED)].copy()
    loss_ledger = refinement_module._loss_ledger_frame(frame)
    strategy_id = str(frozen["strategy"]["strategy_id"])
    profile = refinement_module._loss_ledger_profile(
        loss_ledger,
        (),
        accepted_guards=(),
        primary_override=None,
        spec={"hypothesis_id": strategy_id, "semantic_invariants": {}},
        root=Path(root),
    )
    # Agent 가 경로를 근거로 삼으려면 그 경로가 프로필에 있어야 한다. 규칙에서
    # `path_observation` 을 말하면서 값을 안 주면 프롬프트가 거짓말이 된다.
    # FILLED_ONLY 는 체결만 남긴 프레임이라 미체결과 견주는 세 항목을 잴 수 없다.
    # 재면 필터가 만든 값이 나오는데 그것을 관측인 척하면 안 된다.
    profile = {**profile, "path_observation": PA.observe(
        frame, unfilled_available=(population != "FILLED_ONLY"))}
    return profile, {
        "population": population,
        "ledger": str(ledger_path),
        "ledger_contract_id": ledger_contract_id,
        "ledger_sha256": sha256_file(ledger_path),
        "rows": int(len(loss_ledger)),
    }


def _refined_strategy(parent: Mapping[str, Any], proposal: Mapping[str, Any],
                      strategy_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    expression = proposal.get("entry_expression")
    if not isinstance(expression, Mapping):
        return None, ["entry_expression이 객체가 아니다"]
    parameters = {
        str(item["name"]): dict(item)
        for item in parent["strategy"].get("parameters") or []
    }
    placeholders = _placeholder_names(expression)
    missing = sorted(placeholders - set(parameters))
    if missing:
        return None, [f"실행값 없는 새 UNRESOLVED가 있다: {missing}"]
    raw = {
        "strategy_id": strategy_id,
        "title": str(proposal.get("hypothesis") or strategy_id),
        "rationale": str(proposal.get("mechanism") or ""),
        "entry_expression": copy.deepcopy(dict(expression)),
        "parameters": [parameters[name] for name in sorted(placeholders)],
        "path_prediction": copy.deepcopy(proposal.get("path_prediction")),
    }
    return validate_strategy(raw)


def _refinement_net(frozen: Mapping[str, Any]) -> float:
    return float(_objective(frozen.get("discovery_summary") or {}) or 0.0)


def _run_refinement(request: FreeResearchRequest, output: Path, *,
                    symbols: Sequence[str], parents: Mapping[str, Mapping[str, Any]],
                    context_id: str, strategy_set_id: str, discovery_id: str,
                    runner: Runner | None, replay: Replay, root: Path,
                    runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Discovery 손실 장부만 읽어 새 진입식을 만들고 Optuna로 다시 탐색한다."""
    stage_root = Path(output) / "03_refinement"
    rounds_requested = int(request.refinement_rounds)
    populations = tuple(dict.fromkeys(map(str, request.refinement_populations)))
    if rounds_requested == 0:
        units = {str(key): copy.deepcopy(dict(value)) for key, value in parents.items()}
        artifact = create_artifact("refined_strategy_set", {
            "state": "REFINEMENT_SKIPPED",
            "rounds_requested": 0,
            "rounds_completed": 0,
            "populations": list(populations),
            "parent_strategy_count": len(parents),
            "refined_candidate_count": 0,
            "parent_retained_count": len(parents),
            "selection_policy": "NO_PARENT_IMPROVEMENT_OR_DECISION_FLOOR",
            "rounds": [],
            "units": units,
        }, parents={
            "research_context": context_id,
            "knowledge_strategy_set": strategy_set_id,
            "direct_net_discovery_set": discovery_id,
        })
        write_artifact(stage_root / "refinement_artifact.json", artifact)
        return artifact

    catalog_context = refinement_module._catalog_context()
    config = refinement_agent.RefinementAgentConfig(
        model=request.model,
        effort=request.effort,
        max_proposals=int(request.refinement_max_proposals),
    )
    active = {
        f"{strategy_id}::{population}": {
            "root_strategy_id": str(strategy_id),
            "population": population,
            "frozen": copy.deepcopy(dict(frozen)),
        }
        for strategy_id, frozen in parents.items()
        for population in populations
    }
    refined_units: dict[str, dict[str, Any]] = {}
    round_records: list[dict[str, Any]] = []

    for round_index in range(1, rounds_requested + 1):
        next_active: dict[str, dict[str, Any]] = {}
        branch_records: list[dict[str, Any]] = []
        for branch_id, branch in sorted(active.items()):
            parent = branch["frozen"]
            root_strategy_id = str(branch["root_strategy_id"])
            population = str(branch["population"])
            loss_profile, ledger_source = _refinement_loss_profile(
                parent, population, root=Path(root))
            current_contract = _refinement_contract(parent)
            agent_result = refinement_agent.run(
                loss_profile,
                current_contract=current_contract,
                catalog_context=catalog_context,
                runner=runner,
                parent_alignment=(parent.get("discovery_summary") or {}).get("path_alignment"),
                config=config,
            )
            proposal_values = ((agent_result.get("payload") or {}).get("proposals") or []
                               if agent_result.get("state") == "READY" else [])
            proposals = [dict(value) for value in proposal_values
                         if isinstance(value, Mapping)]
            population_code = "AD" if population == "ALL_DECISIONS" else "FO"
            candidates: list[dict[str, Any]] = []
            candidate_frozen: list[dict[str, Any]] = []
            for rank, proposal in enumerate(proposals, start=1):
                candidate_id = (
                    f"{root_strategy_id}__R{round_index:02d}__{population_code}{rank:02d}")
                candidate, problems = _refined_strategy(parent, proposal, candidate_id)
                if candidate is None:
                    candidates.append({
                        "strategy_id": candidate_id,
                        "state": "INVALID_STRATEGY",
                        "problems": problems,
                        "proposal": proposal,
                    })
                    continue
                unit_root = (stage_root / "rounds" / f"{round_index:02d}" /
                             population / root_strategy_id / "candidates" / candidate_id)
                optimization, frozen = _optimize_strategy(
                    candidate,
                    request,
                    symbols,
                    output,
                    replay,
                    Path(root),
                    runtime,
                    unit_root=unit_root,
                )
                record = {
                    "strategy_id": candidate_id,
                    "state": optimization["state"],
                    "proposal": proposal,
                    "strategy": candidate,
                    "optimization": optimization,
                }
                candidates.append(record)
                if frozen is not None:
                    frozen = {
                        **frozen,
                        "refinement_lineage": {
                            "root_strategy_id": root_strategy_id,
                            "parent_strategy_id": str(parent["strategy"]["strategy_id"]),
                            "round": round_index,
                            "population": population,
                            "proposal_rank": rank,
                        },
                    }
                    refined_units[candidate_id] = frozen
                    candidate_frozen.append(frozen)
            selected = max(
                candidate_frozen,
                key=lambda value: (
                    _refinement_net(value),
                    int(((value.get("discovery_summary") or {}).get("metrics") or {})
                        .get("scorable") or 0),
                    str(value["strategy"]["strategy_id"]),
                ),
                default=None,
            )
            if selected is not None:
                next_active[branch_id] = {
                    "root_strategy_id": root_strategy_id,
                    "population": population,
                    "frozen": selected,
                }
            branch_records.append({
                "branch_id": branch_id,
                "root_strategy_id": root_strategy_id,
                "parent_strategy_id": str(parent["strategy"]["strategy_id"]),
                "population": population,
                "parent_discovery_metrics": copy.deepcopy(
                    (parent.get("discovery_summary") or {}).get("metrics") or {}),
                "ledger_source": ledger_source,
                "loss_profile": loss_profile,
                "current_contract": current_contract,
                "agent": agent_result,
                "candidates": candidates,
                "next_round_strategy_id": (
                    str(selected["strategy"]["strategy_id"]) if selected is not None else None),
                "parent_improvement_required": False,
                "decision_floor": None,
            })
        round_records.append({
            "round": round_index,
            "branch_count": len(branch_records),
            "candidate_count": sum(len(value["candidates"]) for value in branch_records),
            "next_round_branch_count": len(next_active),
            "branches": branch_records,
        })
        active = next_active
        if not active:
            break

    parent_units = {
        str(strategy_id): copy.deepcopy(dict(frozen))
        for strategy_id, frozen in parents.items()
    }
    units = {**parent_units, **refined_units}
    artifact = create_artifact("refined_strategy_set", {
        "state": ("REFINEMENT_COMPLETE" if refined_units else "REFINEMENT_NO_CHANGE"),
        "rounds_requested": rounds_requested,
        "rounds_completed": len(round_records),
        "populations": list(populations),
        "parent_strategy_count": len(parents),
        "refined_candidate_count": len(refined_units),
        "parent_retained_count": len(parent_units),
        "selection_policy": (
            "KEEP_EVERY_PARENT_AND_EXECUTABLE_PROPOSAL__"
            "BEST_DISCOVERY_NET_CONTINUES_NEXT_ROUND__NO_PARENT_IMPROVEMENT_OR_DECISION_FLOOR"),
        "catalog_identity": {
            "catalog_sha256": catalog_context["catalog_sha256"],
            "catalog_policy_sha256": catalog_context["catalog_policy_sha256"],
        },
        "rounds": round_records,
        "units": units,
    }, parents={
        "research_context": context_id,
        "knowledge_strategy_set": strategy_set_id,
        "direct_net_discovery_set": discovery_id,
    })
    write_artifact(stage_root / "refinement_artifact.json", artifact)
    return artifact


def run(request: FreeResearchRequest, output: Path, *, runner: Runner | None = None,
        replay: Replay = replay_variant, root: Path = TICK_ROOT) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if request.strategy_source is None and int(request.strategy_count) < 1:
        raise ValueError("strategy_count는 1 이상이어야 한다")
    if int(request.trials_per_strategy) < 1:
        raise ValueError("trials_per_strategy는 1 이상이어야 한다")
    if int(request.optuna_processes) < 1:
        raise ValueError("optuna_processes는 1 이상이어야 한다")
    if int(request.workers) < 1:
        raise ValueError("workers는 1 이상이어야 한다")
    if runner is None:
        # 기본 Runner 를 여기서 한 번 만든다. invoke 가 호출마다 새로 만들면
        # 타임아웃을 넘길 자리가 없다.
        runner = CodexAgentRunner(timeout_seconds=int(request.agent_timeout_seconds))
    stage_symbols = _stage_symbols(request)
    symbols = tuple(sorted(set().union(*map(set, stage_symbols.values()))))
    _validate_dates(request)
    _validate_refinement(request)
    runtime = runtime_identity()
    context = create_artifact("research_context", {
        "symbols": list(symbols),
        "stage_symbols": {
            name: {
                "symbols": list(values),
                "count": len(values),
                "sha256": sha256_json(list(values)),
            }
            for name, values in stage_symbols.items()
        },
        "discovery_dates": list(map(str, request.discovery_dates)),
        "validation_dates": list(map(str, request.validation_dates)),
        "final_dates": list(map(str, request.final_dates)),
        "constraints": [
            "DECISION_TIME_CAUSAL_INPUTS_ONLY",
            "ACTUAL_BID_ASK_AND_EXPLICIT_23BPS",
            "DISJOINT_DISCOVERY_VALIDATION_FINAL_DATES",
            "REFINEMENT_USES_DISCOVERY_ONLY",
            "FULL_STRATEGY_FROZEN_BEFORE_VALIDATION",
            "LONG_ONLY",
            "CANONICAL_ENTRY_AND_EXIT_EXECUTION_FIXED",
        ],
        "search_space": "ENTRY_CONDITION_AND_ITS_PARAMETERS_ONLY",
        "agent": {
            "model": str(request.model), "effort": str(request.effort),
            "timeout_seconds": int(request.agent_timeout_seconds),
        },
        "optimizer": {
            "name": "OPTUNA",
            "trials_per_strategy": int(request.trials_per_strategy),
            "processes": int(request.optuna_processes),
            "seed": int(request.optuna_seed),
            "objective": "DISCOVERY_NET_BPS_TOTAL",
        },
        "refinement": {
            "rounds": int(request.refinement_rounds),
            "max_rounds": MAX_REFINEMENT_ROUNDS,
            "max_proposals_per_branch": int(request.refinement_max_proposals),
            "populations": list(map(str, request.refinement_populations)),
            "parent_improvement_required": False,
            "decision_floor": None,
        },
        "runtime_identity": runtime,
    })
    write_artifact(output / "00_research_context.json", context)

    raw, invocations = _load_or_generate(request, output, runner)
    raw_strategies = raw.get("strategies") if isinstance(raw, Mapping) else None
    if not isinstance(raw_strategies, list):
        raise ValueError("Knowledge Strategy Agent 응답에 strategies 목록이 없다")
    strategies: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    expanded: list[Mapping[str, Any]] = []
    for raw_hypothesis in raw_strategies:
        if not isinstance(raw_hypothesis, Mapping):
            rejected.append({"strategy_id": None, "problems": ["strategy가 객체가 아니다"]})
            continue
        variants, problems = expand_hypothesis(raw_hypothesis)
        if problems:
            rejected.append({"strategy_id": str(raw_hypothesis.get("strategy_id") or "") or None,
                             "problems": problems})
            continue
        expanded.extend(variants)
    for raw_strategy in expanded:
        strategy, problems = validate_strategy(raw_strategy)
        strategy_id = str(raw_strategy.get("strategy_id") or "")
        if strategy is not None and _grid_size(strategy) > int(request.trials_per_strategy):
            problems = [*problems, f"파라미터 격자가 {_grid_size(strategy)}칸이라 전수 탐색 한도 "
                                   f"{request.trials_per_strategy}칸을 넘는다. values를 줄여라"]
            strategy = None
        if strategy_id in seen:
            problems = [*problems, f"strategy_id가 중복됐다: {strategy_id}"]
            strategy = None
        if strategy is None:
            rejected.append({"strategy_id": strategy_id or None, "problems": problems})
            continue
        seen.add(strategy_id)
        strategies.append(strategy)
    combining = [s for s in strategies
                 if catalog.expression_ops(s["entry_expression"]) & (
                     catalog.CLAIM_OPS - {"primitive", "raw", "level_aggregate"})]
    strategy_set = create_artifact("knowledge_strategy_set", {
        "state": "STRATEGIES_READY" if strategies else "NO_EXECUTABLE_STRATEGY",
        # 카탈로그에 없는 관측을 스스로 만든 비율. 26개 연산자 중 실제로 쓰는 것이
        # 몇 개인지를 캠페인마다 재려고 남긴다.
        "combined_observation_count": len(combining),
        "hypothesis_count": len({str(s.get("hypothesis_id") or s["strategy_id"])
                                 for s in strategies}),
        "source": "AGENT_OWN_KNOWLEDGE" if request.strategy_source is None else "SAVED_JSON",
        "strategies": strategies,
        "rejected": rejected,
        "agent_invocations": invocations,
    }, parents={"research_context": context["artifact_id"]})
    write_artifact(output / "01_knowledge_strategies.json", strategy_set)

    discovery_units: dict[str, Any] = {}
    discovery_frozen_units: dict[str, Any] = {}
    for strategy in strategies:
        strategy_id = str(strategy["strategy_id"])
        discovery_unit, frozen = _optimize_strategy(
            strategy, request, stage_symbols["discovery"], output, replay, Path(root), runtime)
        discovery_units[strategy_id] = discovery_unit
        if frozen is not None:
            discovery_frozen_units[strategy_id] = frozen
    discovery = create_artifact("direct_net_discovery_set", {
        "state": "DISCOVERY_COMPLETE", "units": discovery_units,
    }, parents={"research_context": context["artifact_id"],
                "knowledge_strategy_set": strategy_set["artifact_id"]})
    write_artifact(output / "02_discovery" / "discovery_artifact.json", discovery)

    refinement = _run_refinement(
        request,
        output,
        symbols=stage_symbols["discovery"],
        parents=discovery_frozen_units,
        context_id=context["artifact_id"],
        strategy_set_id=strategy_set["artifact_id"],
        discovery_id=discovery["artifact_id"],
        runner=runner,
        replay=replay,
        root=Path(root),
        runtime=runtime,
    )
    frozen_units = refinement["payload"]["units"]
    frozen_set = create_artifact("frozen_strategy_set", {
        "state": "STRATEGIES_FROZEN" if frozen_units else "NO_STRATEGY_TO_FREEZE",
        "frozen_before_validation": True,
        "units": frozen_units,
    }, parents={"research_context": context["artifact_id"],
                "knowledge_strategy_set": strategy_set["artifact_id"],
                "direct_net_discovery_set": discovery["artifact_id"],
                "refined_strategy_set": refinement["artifact_id"]})
    write_artifact(output / "04_frozen_strategies.json", frozen_set)

    if replay is replay_variant and frozen_units:
        validation_units = replay_variant_batch(
            frozen_units, stage_symbols["validation"], request.validation_dates,
            output / "05_validation" / "batch", int(request.workers), Path(root))
        for strategy_id, summary in validation_units.items():
            write_json(output / "05_validation" / "units" / strategy_id / "summary.json", summary)
    else:
        validation_units = {
            strategy_id: replay(
                frozen, stage_symbols["validation"], request.validation_dates,
                output / "05_validation" / "units" / strategy_id,
                int(request.workers), Path(root))
            for strategy_id, frozen in frozen_units.items()
        }
    validation = create_artifact("validation_replay_set", {
        "state": "VALIDATION_COMPLETE", "units": validation_units,
        "strategy_changed": False,
    }, parents={"research_context": context["artifact_id"],
                "frozen_strategy_set": frozen_set["artifact_id"]})
    write_artifact(output / "05_validation" / "validation_artifact.json", validation)

    # Final opens every frozen strategy. Validation results do not select, repair, or mutate it.
    if replay is replay_variant and frozen_units:
        final_units = replay_variant_batch(
            frozen_units, stage_symbols["final"], request.final_dates,
            output / "06_final" / "batch", int(request.workers), Path(root))
        for strategy_id, summary in final_units.items():
            write_json(output / "06_final" / "units" / strategy_id / "summary.json", summary)
    else:
        final_units = {
            strategy_id: replay(
                frozen, stage_symbols["final"], request.final_dates,
                output / "06_final" / "units" / strategy_id,
                int(request.workers), Path(root))
            for strategy_id, frozen in frozen_units.items()
        }
    final = create_artifact("final_replay_set", {
        "state": "FINAL_COMPLETE", "units": final_units,
        "strategy_changed_after_freeze": False,
        "validation_used_for_selection": False,
    }, parents={"research_context": context["artifact_id"],
                "frozen_strategy_set": frozen_set["artifact_id"],
                "validation_replay_set": validation["artifact_id"]})
    write_artifact(output / "06_final" / "final_artifact.json", final)
    state = "FINAL_COMPLETE" if frozen_units else "NO_EXECUTABLE_STRATEGY"
    run_artifact = create_artifact("free_research_run", {
        "state": state,
        "created_at": now_utc(),
        "strategy_count": len(strategies),
        "discovery_frozen_strategy_count": len(discovery_frozen_units),
        "refined_strategy_count": int(refinement["payload"]["refined_candidate_count"]),
        "frozen_strategy_count": len(frozen_units),
        "rejected_strategy_count": len(rejected),
        "artifacts": {
            "research_context": context["artifact_id"],
            "knowledge_strategy_set": strategy_set["artifact_id"],
            "direct_net_discovery_set": discovery["artifact_id"],
            "refined_strategy_set": refinement["artifact_id"],
            "frozen_strategy_set": frozen_set["artifact_id"],
            "validation_replay_set": validation["artifact_id"],
            "final_replay_set": final["artifact_id"],
        },
    }, parents={
        "research_context": context["artifact_id"],
        "knowledge_strategy_set": strategy_set["artifact_id"],
        "direct_net_discovery_set": discovery["artifact_id"],
        "refined_strategy_set": refinement["artifact_id"],
        "frozen_strategy_set": frozen_set["artifact_id"],
        "validation_replay_set": validation["artifact_id"],
        "final_replay_set": final["artifact_id"],
    })
    write_artifact(output / "research_run.json", run_artifact)
    return {"state": state, "artifacts": run_artifact["payload"]["artifacts"],
            "run": run_artifact, "refinement": refinement,
            "validation": validation, "final": final}
