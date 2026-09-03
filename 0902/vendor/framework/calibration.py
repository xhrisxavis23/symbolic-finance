"""임계 보정. 종목-일마다 진입 임계를 만들고 실행에 넘길 값으로 바꾼다.

`search.py` 안에 있었지만, 백테스트가 쓰는 것은 여기 12개뿐이다. 그것 하나 때문에
옛 Workflow(`executable` -> `validation` -> `grounding`) 전체를 import 하고 있었다.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import capability as C, catalog, contract, data as tickdata
from .config import TICK_ROOT, sha256_json


THRESHOLD_KINDS = (catalog.THRESHOLD_LITERAL,
                   catalog.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
                   catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE)


PARAMETER_LOCKED = "PARAMETER_LOCKED"


PARAMETER_LOCKED_SINGLE_DAY = "PARAMETER_LOCKED_SINGLE_DAY"


PARAMETER_LOCK_STATUSES = (PARAMETER_LOCKED, PARAMETER_LOCKED_SINGLE_DAY)



@dataclass(frozen=True)
class SearchConfig:
    """실행 전에 얼린다. 결과를 보고 하나도 바꾸지 않는다."""

    parameter: str | None = None
    feature: str = "book_imbalance"
    threshold_kind: str = catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE
    # 성긴 격자. 중앙 이상부터 상단 꼬리까지 (§15~§16).
    grid: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    # 현재 tick을 빼고 직전 100 tick만 쓴다. 첫 100 tick에는 cut이 없다.
    min_calibration_obs: int = 100
    # 보간하지 않는다. 실제 관측값을 컷으로 쓴다 (§10).
    quantile_method: str = "higher"
    calibration_scope: str = "SYMBOL"
    calibration_reference: str = "PREVIOUS_100_TICKS_SAME_SYMBOL_DAY"
    # 검증에서 쓰던 것을 그대로 (§31~§32).
    min_pairs: int = 100
    bootstrap: int = 1000
    seed: int = 0
    null_value: float = 0.5
    primary_metric: str = "MATCHED_FUTURE_PATH_ADVANTAGE"
    response: str = "max_net_bps_s30"
    # 짝을 맞출 때 통제할 것. 가설마다 맥락 feature 가 다르므로 설정으로 받는다.
    controls: tuple[str, ...] = ("vol_flow", "spread_bps", "pre_move_bps")
    # 명세가 정한 방향. HIGHER 면 위 꼬리를 자르고 `>`, LOWER 면 아래 꼬리를 자르고 `<`.
    direction: str = "HIGHER"
    # (feature, direction, parameter). 비어 있으면 기존 단일 상태를 쓴다. 여러
    # 상태도 q 하나를 공유한다. 이는 q^N 격자를 새로 열지 않기 위한 고정 규칙이다.
    state_definitions: tuple[tuple[str, str, str], ...] = ()
    # Search가 anchor frame 위에서 평가할 정본 Catalog AST. 단일식은 None이다.
    expression: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.parameter is None:
            prefix = "q" if self.is_quantile else "literal"
            object.__setattr__(self, "parameter", f"{prefix}_{self.feature}")
        if self.threshold_kind not in THRESHOLD_KINDS:
            raise ValueError(f"알 수 없는 threshold source: {self.threshold_kind!r}")
        if not self.grid:
            raise ValueError("threshold grid가 비었다")
        if not all(math.isfinite(float(value)) for value in self.grid):
            raise ValueError("threshold grid에는 유한한 숫자만 넣을 수 있다")
        if self.is_quantile and not all(0.0 <= float(value) < 1.0 for value in self.grid):
            raise ValueError("분위 q는 0 이상 1 미만이어야 한다")
        states = self.state_definitions or ((str(self.feature), str(self.direction),
                                             str(self.parameter)),)
        if len({name for _feature, _direction, name in states}) != len(states):
            raise ValueError("복합 Search parameter 이름이 중복됐다")
        if any(feature not in catalog.FEATURES or direction not in ("HIGHER", "LOWER")
               or not name for feature, direction, name in states):
            raise ValueError("복합 Search state 정의가 유효하지 않다")

    @property
    def is_quantile(self) -> bool:
        return self.threshold_kind in {
            catalog.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
            catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
        }

    @property
    def is_rolling_quantile(self) -> bool:
        return self.threshold_kind == catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE

    @property
    def lower(self) -> bool:
        return str(self.direction).upper() == "LOWER"

    @property
    def states(self) -> tuple[tuple[str, str, str], ...]:
        return self.state_definitions or ((str(self.feature), str(self.direction),
                                           str(self.parameter)),)

    @property
    def independent_state_quantiles(self) -> bool:
        """현재 정본 Search는 하나의 q만 고정한다."""
        return False

    def candidate_id(self, value: float) -> str:
        """q별 refinement artifact와 Search 후보를 잇는 안정된 이름."""
        return f"q_{float(value):.2f}"

    def tail(self, quantile: float) -> float:
        """격자 값은 '얼마나 꼬리로 갈 것인가' 다. 방향에 따라 어느 쪽 꼬리인지 정한다."""
        if not self.is_quantile:
            raise ValueError("literal threshold에는 quantile tail이 없다")
        return (1.0 - float(quantile)) if self.lower else float(quantile)

    @property
    def max_quantile(self) -> float:
        return max(self.grid)

    def derived_min_obs(self) -> int:
        """최소 관측 수가 어디서 나왔는지. 손으로 고른 값이 아니다."""
        if not self.is_quantile:
            return 0
        if self.is_rolling_quantile:
            return contract.ROLLING_QUANTILE_WINDOW
        return int(math.ceil(10 / (1 - self.max_quantile)))


def threshold_column(value: float, config: SearchConfig, parameter: str | None = None) -> str:
    """보정 표에서 한 후보 값이 쓰는 열 이름."""
    base = f"q{float(value):.2f}" if config.is_quantile else f"literal_{float(value):.12g}"
    # 기존 단일 표의 열 이름을 보존한다. 복합 표만 parameter prefix를 붙인다.
    if parameter is not None and len(config.states) > 1:
        return f"{parameter}__{base}"
    return base


def config_for(spec: Mapping[str, Any],
               base: "SearchConfig | None" = None) -> "SearchConfig":
    """**탐색할 feature 를 명세에서 읽는다.** 설정에 손으로 적지 않는다."""
    base = base or SearchConfig()
    allowed = list((spec.get("search_boundary") or {}).get("allowed_parameters") or [])
    invariants = spec.get("semantic_invariants") or {}
    if len(allowed) > 1 or invariants.get("operational_expression") is not None:
        states = []
        for item in allowed:
            features = list(item.get("features") or [item.get("feature")])
            if len(features) != 1 or features[0] not in catalog.FEATURES:
                raise ValueError("Search의 각 threshold는 Catalog feature 하나에 대응해야 한다")
            direction = str(item.get("direction") or invariants.get("direction") or "")
            states.append((str(features[0]), direction, str(item.get("name") or "")))
        source = dict(allowed[0].get("threshold_source") or {})
        if any((item.get("threshold_source") or {}).get("kind", source.get("kind")) !=
               source.get("kind") for item in allowed):
            raise ValueError("Search는 threshold source 하나만 공유할 수 있다")
        kind = str(source.get("kind") or base.threshold_kind)
        grid = tuple(float(value) for value in source.get("grid") or base.grid)
        features = {feature for feature, _direction, _parameter in states}
        context = invariants.get("context_feature") or invariants.get("rejected_context_feature")
        controls = [c for c in (context, "spread_bps", "pre_move_bps")
                    if c and c not in features]
        return dataclasses.replace(base, parameter=states[0][2], feature=states[0][0],
                                   direction=states[0][1], threshold_kind=kind, grid=grid,
                                   controls=tuple(dict.fromkeys(controls)),
                                   state_definitions=(tuple(states) if len(states) > 1 else ()),
                                   expression=dict(spec.get("signal_template") or {}))
    feature = invariants.get("canonical_feature")
    if not feature:
        node = (spec.get("signal_template") or {}).get("input") or {}
        feature = node.get("primitive_id")
    if not feature:
        return base
    context = (invariants.get("context_feature")
               or invariants.get("rejected_context_feature"))
    # 통제에 trigger 자신을 넣지 않는다. `spread_bps` 로 나누면서 `spread_bps` 를
    # 맞추라는 요구는 모순이라 짝짓기가 구조적으로 실패한다.
    controls = [c for c in (context, "spread_bps", "pre_move_bps")
                if c and c != feature]
    source = ((spec.get("signal_template") or {}).get("_parameter") or
              {}).get("threshold_source") or {}
    kind = str(source.get("kind") or base.threshold_kind)
    grid = tuple(float(value) for value in source.get("grid") or base.grid)
    parameter_prefix = "q" if kind in {
        catalog.THRESHOLD_PRIOR_VALID_DAY_SYMBOL_QUANTILE,
        catalog.THRESHOLD_ROLLING_PRIOR_100_TICKS_QUANTILE,
    } else "literal"
    return dataclasses.replace(base, parameter=f"{parameter_prefix}_{feature}", feature=str(feature),
                               threshold_kind=kind, grid=grid,
                               controls=tuple(dict.fromkeys(controls)),
                               direction=str(invariants.get("direction") or "HIGHER"))


def previous_real_trading_day(date: str, calendar: Sequence[str]) -> str | None:
    """직전 **실거래일**. 달력에서 하루 빼지 않는다. 합성 날짜는 애초에 달력에 없다."""
    earlier = [d for d in calendar if d < date]
    return max(earlier) if earlier else None


def trading_calendar(root: Path = TICK_ROOT) -> list[str]:
    return [d for d in tickdata.available_dates(root) if d not in C.NOT_REAL_DATES]


def calibrate(symbol: str, date: str, config: SearchConfig,
              calendar: Sequence[str], root: Path = TICK_ROOT) -> dict[str, Any]:
    """그 종목·그 날이 직전 100 tick q를 만들 수 있는지 확인한다.

    rolling q는 날짜 전체의 분포를 한 번에 보지 않는다. 각 tick에서 앞 100개만 본다.
    이 표는 그 날에 유효한 cut이 하나라도 있는지만 적고, 실제 tick별 cut은 신호를
    계산할 때 만든다.
    """
    prior = previous_real_trading_day(date, calendar) if (
        config.is_quantile and not config.is_rolling_quantile) else None
    row: dict[str, Any] = {"symbol": symbol, "date": date, "reference_date": prior,
                           "eligible": False, "reason": None, "observations": 0,
                           "window_ticks": (contract.ROLLING_QUANTILE_WINDOW
                                            if config.is_rolling_quantile else None)}
    for value in config.grid:
        row[threshold_column(value, config)] = float("nan")
    if not config.is_quantile:
        row["eligible"] = True
        for value in config.grid:
            row[threshold_column(value, config)] = float(value)
        return row
    if config.is_rolling_quantile:
        try:
            arrays, _ = tickdata.load(symbol, date, root)
        except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
            row["reason"] = f"당일 tick을 읽을 수 없다: {error}"
            return row
        book = catalog.Book(arrays)
        values = catalog.compute_features((config.feature,), book).get(config.feature)
        if values is None:
            availability = {item.feature: item for item in catalog.feature_availability(book.capabilities)}
            detail = availability.get(config.feature)
            row["reason"] = ((detail.reason if detail is not None else None)
                             or f"feature를 계산할 수 없다: {config.feature}")
            return row
        values = np.asarray(values, dtype=float)
        row["observations"] = int(np.isfinite(values).sum())
        # 이 표는 q cut 자체가 아니라 그 날에 cut을 만들 수 있는지만 적는다. 그래서
        # rolling quantile 배열 전체를 계산할 필요가 없다. 현재 tick은 창에 안 들어가므로
        # 마지막 관측으로 끝나는 100개 연속 구간만으로는 아직 cut 하나가 생기지 않는다.
        window = contract.ROLLING_QUANTILE_WINDOW
        valid = np.isfinite(values)
        consecutive = (np.convolve(valid.astype(np.int16), np.ones(window, dtype=np.int16),
                                   mode="valid") if len(values) >= window else np.array([], dtype=np.int16))
        if len(values) <= window or not np.any(consecutive[:-1] == window):
            row["reason"] = f"직전 {contract.ROLLING_QUANTILE_WINDOW} tick 연속 관측이 없다"
            return row
        row["eligible"] = True
        # q는 종목-일마다 고정 cut이 아니라 런타임 파라미터다. 표에는 선택할 q만 남긴다.
        for quantile in config.grid:
            row[threshold_column(quantile, config)] = float(quantile)
        return row
    if prior is None:
        row["reason"] = "직전 실거래일이 데이터셋에 없다"
        return row
    try:
        arrays, _ = tickdata.load(symbol, prior, root)
    except (FileNotFoundError, tickdata.InsufficientQuoteData) as error:
        row["reason"] = f"전일을 읽을 수 없다: {error}"
        return row

    values = catalog.FEATURES[config.feature].compute(catalog.Book(arrays))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    row["observations"] = int(len(finite))
    if len(finite) < config.min_calibration_obs:
        row["reason"] = (f"전일 관측이 {len(finite)}개로 "
                         f"{config.min_calibration_obs} 미만")
        return row

    row["eligible"] = True
    method = "lower" if config.lower else config.quantile_method
    for quantile in config.grid:
        row[threshold_column(quantile, config)] = float(
            np.quantile(finite, config.tail(quantile), method=method))
    return row


def _state_config(config: SearchConfig, state: tuple[str, str, str]) -> SearchConfig:
    """한 joint-q 상태의 보정만 읽는, 기존 단일 보정 호환 설정."""
    feature, direction, parameter = state
    return dataclasses.replace(config, feature=feature, direction=direction,
                               parameter=parameter, state_definitions=(), expression=None)


def calibration_table(symbols: Sequence[str], dates: Sequence[str],
                      config: SearchConfig, root: Path = TICK_ROOT,
                      anchor_cache_root: Path | None = None) -> pd.DataFrame:
    """모든 종목-일의 임계. 후보마다 우주를 바꾸지 않기 위해 한 번에 만든다 (§14)."""
    if len(config.states) > 1:
        tables: list[pd.DataFrame] = []
        for state in config.states:
            feature, _direction, parameter = state
            part = calibration_table(symbols, dates, _state_config(config, state), root,
                                     anchor_cache_root=anchor_cache_root)
            rename = {threshold_column(q, config): threshold_column(q, config, parameter)
                      for q in config.grid}
            part = part.rename(columns=rename).rename(columns={
                "eligible": f"eligible__{parameter}",
                "reason": f"reason__{parameter}",
                "observations": f"observations__{parameter}",
            })
            tables.append(part)
        keys = ["symbol", "date", "reference_date"]
        merged = tables[0]
        for part in tables[1:]:
            merged = merged.merge(part, on=keys, how="outer", validate="one_to_one")
        eligible_columns = [f"eligible__{parameter}" for _feature, _direction, parameter in config.states]
        reason_columns = [f"reason__{parameter}" for _feature, _direction, parameter in config.states]
        merged["eligible"] = merged[eligible_columns].fillna(False).all(axis=1)
        merged["reason"] = merged.apply(
            lambda row: "; ".join(
                str(row[name]) for name in reason_columns
                if pd.notna(row.get(name)) and str(row.get(name)).strip()), axis=1)
        merged.loc[merged["reason"] == "", "reason"] = None
        merged["observations"] = merged[[f"observations__{p}" for _f, _d, p in config.states]].min(axis=1)
        return merged
    if config.is_rolling_quantile and anchor_cache_root is not None:
        from .modules import search_anchor_cache
        return search_anchor_cache.calibration(anchor_cache_root, dates, symbols, config)
    calendar = trading_calendar(root)
    rows = [calibrate(symbol, date, config, calendar, root)
            for date in dates for symbol in symbols]
    return pd.DataFrame(rows)


def parameter_values_for_table(table: pd.DataFrame, quantile: float,
                               config: SearchConfig) -> dict[str, dict[str, float]]:
    """종목-일별 runtime parameter.

    rolling source에서는 숫자 cut이 아니라 잠긴 q를 넘긴다. 실제 cut은 해당 종목·날짜
    실행 중 현재 tick 이전 100개 관측에서 만든다.
    """
    values: dict[str, dict[str, float]] = {}
    usable = table.loc[table["eligible"].astype(bool)]
    for _, row in usable.iterrows():
        state_values = ({parameter: float(quantile)
                         for _feature, _direction, parameter in config.states}
                        if config.is_rolling_quantile else {
                            parameter: float(row[threshold_column(quantile, config, parameter)])
                            for _feature, _direction, parameter in config.states})
        values[f"{row['symbol']}:{row['date']}"] = state_values
    return values


def threshold_source(config: SearchConfig, value: float | None = None) -> dict[str, Any]:
    """원장과 lock이 같은 source를 읽게 하는 작은 공통 표현."""
    source: dict[str, Any] = {"kind": config.threshold_kind}
    if value is not None:
        source["q" if config.is_quantile else "value"] = float(value)
    return source


def amend_specification(spec: Mapping[str, Any],
                        config: SearchConfig = SearchConfig()) -> dict[str, Any]:
    """임계의 source를 정한다. **나머지는 하나도 건드리지 않는다.**

    바뀌는 것은 그 하나가 무엇을 뜻하는가지, 몇 개인가가 아니다.
    """
    amended = {k: v for k, v in spec.items() if k != "specification_sha256"}
    # Grounding AST 명세는 단일 비교여도 parameter metadata를
    # ``search_boundary``에 둔다. 그 경우 ``signal_template._parameter``를
    # 전제하면 안 된다.
    if len(config.states) > 1 or (spec.get("semantic_invariants") or {}).get("operational_expression") is not None:
        boundary = {k: v for k, v in spec["search_boundary"].items()}
        sources = {name: threshold_source(config) for _feature, _direction, name in config.states}
        boundary["allowed_parameters"] = [{**dict(item), "threshold_source": sources[str(item["name"])]}
                                          for item in boundary.get("allowed_parameters") or []]
        amended["search_boundary"] = boundary
        amended["parameterization"] = {
            "changed_from": "unresolved joint thresholds",
            "changed_to": ("state-local prior-100-tick percentile with one shared q"
                           if config.is_rolling_quantile else
                           "state-local previous-day percentile with one shared q"),
            "why": ("각 feature는 현재 tick을 뺀 자기 직전 100 tick 분포에서 보정하되, "
                    "하나의 q만 선택해 복합식의 극단성 수준을 같이 고정한다"
                    if config.is_rolling_quantile else
                    "각 feature는 자기 전일 분포에서 보정하되, 하나의 q만 선택해 복합식의 "
                    "극단성 수준을 같이 고정한다"),
            "search_dimension": 1,
            "joint_parameters": [name for _feature, _direction, name in config.states],
            "note": "feature별 q 조합 격자를 열지 않는다. 그러면 같은 Grounding AST 안에서도 "
                    "탐색 차원이 늘어난다",
        }
        amended["specification_sha256"] = sha256_json(
            {k: v for k, v in amended.items() if k != "created_at"})[:16]
        return amended
    template = dict(spec["signal_template"])
    parameter = dict(template["_parameter"])
    template["value"] = f"{catalog.UNRESOLVED_PREFIX}{config.parameter}"
    source = threshold_source(config)
    if config.is_quantile:
        parameter_type = "percentile"
        meaning = (f"그 종목의 **현재 tick을 뺀 직전 100 tick** {config.feature} 분포에서 "
                   "몇 분위부터 '높다' 고 부를 것인가"
                   if config.is_rolling_quantile else
                   "그 종목의 **전일** 분포에서 몇 분위부터 '높다' 고 부를 것인가. "
                   "값이 달라져도 '높을수록 이후 실행 경로가 낫다' 는 주장은 그대로다")
        runtime_value = (f"theta(symbol, date, tick) = 직전 100 tick {config.feature} 의 q 분위"
                         if config.is_rolling_quantile else
                         f"theta(symbol, date) = 전일 {config.feature} 의 q 분위")
    else:
        parameter_type = "literal"
        meaning = "모든 종목·날짜에서 같은 원값으로 '높다' 고 부를 지점"
        runtime_value = "theta(symbol, date) = 잠근 literal 값"
    parameter.update({
        "name": config.parameter,
        "type": parameter_type,
        "threshold_source": source,
        "meaning": meaning,
        "derived_runtime_value": runtime_value,
    })
    template["_parameter"] = parameter
    amended["signal_template"] = template

    boundary = {k: v for k, v in spec["search_boundary"].items()}
    boundary["allowed_parameters"] = [{
        "name": config.parameter, "status": catalog.SEARCHABLE,
        "feature": config.feature, "meaning": parameter["meaning"],
        "threshold_source": source}]
    forbidden = list(boundary["forbidden_parameters"])
    if config.is_quantile:
        forbidden += [
            {"axis": "원값 임계", "why": "분위와 원값을 같이 돌려 좋은 쪽을 고르면 "
                                        "매개화 자체를 탐색한 것이 된다"},
            {"axis": "보정 기간", "why": ("직전 100 tick 창은 고정한다. 더 긴·짧은 창을 "
                                        "결과 보고 바꾸지 않는다" if config.is_rolling_quantile else
                                       "전일이 없으면 건너뛴다. 더 오래된 날로 물러서면 "
                                       "종목-일마다 보정 기간이 달라진다")},
            {"axis": "분위 계산법", "why": "보간법을 결과 보고 바꾸면 그것도 탐색이다"},
        ]
    else:
        forbidden += [
            {"axis": "분위 보정", "why": "고정값과 분위수를 같이 돌려 좋은 쪽을 "
                                      "고르면 threshold source 자체를 탐색한 것이 된다"},
        ]
    boundary["forbidden_parameters"] = forbidden
    amended["search_boundary"] = boundary
    amended["parameterization"] = (
        {"changed_from": "raw numeric threshold",
         "changed_to": ("symbol-local prior-100-tick percentile"
                        if config.is_rolling_quantile else "symbol-local previous-day percentile"),
         "why": "같은 원값이 종목마다 다른 뜻이다. 상태가 아니라 종목 크기를 거르는 "
                "필터가 된다",
         "search_dimension": 1,
         "note": "분위로 바꾼다고 과적합이 사라지지 않는다. 그건 성긴 격자·확대 금지·"
                 "새 데이터 확인이 막는다"}
        if config.is_quantile else
        {"changed_from": "unresolved threshold",
         "changed_to": "fixed literal",
         "why": "후보 원값 하나를 고른 뒤 모든 종목·날짜에 그대로 쓴다",
         "search_dimension": 1,
         "note": "고정값과 전일 분위는 서로 다른 threshold source다. 같은 판에서 "
                 "둘 다 돌리지 않는다"})
    amended["specification_sha256"] = sha256_json(
        {k: v for k, v in amended.items() if k != "created_at"})[:16]
    return amended
