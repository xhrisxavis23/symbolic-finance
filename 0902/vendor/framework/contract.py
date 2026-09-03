"""계약(표현식 트리)을 틱 위에서 실행한다.

구 `contract.py` 는 primitive 이름마다 if/elif 분기 14개를 들고 있었다. 여기서는
`catalog.FEATURES` 조회 한 번이다 — 분기를 지우는 것이 목적이 아니라, 이름이
레지스트리에만 있게 하는 것이 목적이다.

모든 연산자는 인과적이다. 틱 t 의 값이 t 이후를 보지 않는다.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import catalog
from .config import sha256_json


ROLLING_QUANTILE_WINDOW = 100


class ExpressionRuntime:
    """한 종목-일 위에서 표현식을 계산한다. feature 와 부분식을 한 번만 계산한다."""

    def __init__(self, data: Mapping[str, np.ndarray], *,
                 thresholds: Mapping[str, float | np.ndarray] | None = None,
                 precomputed_expressions: Mapping[str, np.ndarray] | None = None) -> None:
        self.book = catalog.Book(data)
        self.n = self.book.n
        # 이 소스에서 무엇이 왜 안 되는지. 조용히 빠지면 계약이 왜 죽었는지 알 수 없다.
        self.availability = {a.feature: a for a in
                             catalog.feature_availability(self.book.capabilities)}
        self._features: dict[str, np.ndarray] = {}
        # q 후보 묶음은 같은 입력식을 공유한다. threshold가 다른 비교 결과까지 공유하면
        # 안 되므로, 여기에는 q와 무관한 입력식 값만 caller가 넣는다.
        self._cache: dict[str, np.ndarray] = {
            str(key): np.asarray(value) for key, value in (precomputed_expressions or {}).items()
        }
        self.thresholds = dict(thresholds or {})

    def feature(self, name: str) -> np.ndarray:
        name = catalog.resolve(str(name))
        if name not in self._features:
            spec = catalog.FEATURES.get(name)
            if spec is None:
                raise KeyError(f"레지스트리에 없는 feature: {name}")
            state = self.availability[name]
            if not state.available:
                raise KeyError(f"{name} 은 이 소스에서 만들 수 없다: {state.reason}")
            self._features[name] = np.asarray(spec.compute(self.book), dtype=float)
        return self._features[name]

    def evaluate(self, expression: Mapping[str, Any]) -> np.ndarray:
        """표현식 트리 하나를 배열로. 불리언 노드면 bool 배열."""
        catalog.validate_expression(expression, allow_unresolved=bool(self.thresholds))
        key = sha256_json(expression)
        if key not in self._cache:
            result = np.asarray(self._evaluate(expression))
            if result.shape != (self.n,):
                raise ValueError(f"표현식 {key[:12]} 가 {result.shape} 를 냈다. ({self.n},) 여야 한다")
            self._cache[key] = result
        return self._cache[key]

    # -- 연산자 --------------------------------------------------------------
    def _evaluate(self, node: Mapping[str, Any]) -> np.ndarray:
        op = str(node["op"])
        method = getattr(self, f"_op_{op}", None)
        if method is None:
            raise ValueError(f"실행기가 모르는 연산자 {op}")
        return method(node)

    def _op_primitive(self, node):
        return self.feature(node["primitive_id"])

    def _op_raw(self, node):
        field = str(node["field"])
        if field not in self.book.data:
            raise KeyError(f"raw field가 이 소스에 없다: {field}")
        values = np.asarray(self.book.data[field], dtype=float)
        if field in {"bid_price", "ask_price", "bid_qty", "ask_qty"}:
            return values[:, int(node["level"]) - 1]
        return values

    def _op_compare(self, node):
        values = np.asarray(self.evaluate(node["input"]), dtype=float)
        threshold = self._threshold(node["value"])
        finite = np.isfinite(values)
        # NaN 은 어떤 비교도 통과시키지 않는다. 관측이 없는 것은 조건 충족이 아니다.
        return {"<": finite & (values < threshold),
                "<=": finite & (values <= threshold),
                ">": finite & (values > threshold),
                ">=": finite & (values >= threshold),
                "==": finite & (values == threshold)}[str(node["comparator"])]

    def _op_compare_values(self, node):
        left = np.asarray(self.evaluate(node["left"]), dtype=float)
        right = np.asarray(self.evaluate(node["right"]), dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        return {"<": finite & (left < right),
                "<=": finite & (left <= right),
                ">": finite & (left > right),
                ">=": finite & (left >= right),
                "==": finite & (left == right)}[str(node["comparator"])]

    def _op_all(self, node):
        return np.column_stack([np.asarray(self.evaluate(a), dtype=bool)
                                for a in node["args"]]).all(axis=1)

    def _op_any(self, node):
        return np.column_stack([np.asarray(self.evaluate(a), dtype=bool)
                                for a in node["args"]]).any(axis=1)

    def _op_ratio(self, node):
        numerator = np.asarray(self.evaluate(node["numerator"]), dtype=float)
        denominator = np.asarray(self.evaluate(node["denominator"]), dtype=float)
        policy = str(node["zero_policy"])
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        if policy == "reject" and not valid.all():
            raise ValueError("ratio 분모가 0 이거나 결측인데 zero_policy=reject 다")
        out = np.zeros(self.n) if policy == "zero" else np.full(self.n, np.nan)
        out[valid] = numerator[valid] / denominator[valid]
        return out

    def _binary(self, node, operation):
        left = np.asarray(self.evaluate(node["left"]), dtype=float)
        right = np.asarray(self.evaluate(node["right"]), dtype=float)
        out = np.full(self.n, np.nan)
        valid = np.isfinite(left) & np.isfinite(right)
        out[valid] = operation(left[valid], right[valid])
        return out

    def _op_add(self, node):
        return self._binary(node, np.add)

    def _op_subtract(self, node):
        return self._binary(node, np.subtract)

    def _op_multiply(self, node):
        return self._binary(node, np.multiply)

    def _op_divide(self, node):
        left = np.asarray(self.evaluate(node["left"]), dtype=float)
        right = np.asarray(self.evaluate(node["right"]), dtype=float)
        out = np.full(self.n, np.nan)
        valid = np.isfinite(left) & np.isfinite(right) & (right != 0)
        out[valid] = left[valid] / right[valid]
        return out

    def _op_absolute(self, node):
        return np.abs(np.asarray(self.evaluate(node["input"]), dtype=float))

    def _op_negate(self, node):
        return -np.asarray(self.evaluate(node["input"]), dtype=float)

    def _op_log1p(self, node):
        values = np.asarray(self.evaluate(node["input"]), dtype=float)
        out = np.full(self.n, np.nan)
        valid = np.isfinite(values) & (values > -1.0)
        out[valid] = np.log1p(values[valid])
        return out

    def _op_difference(self, node):
        source = np.asarray(self.evaluate(node["input"]), dtype=float)
        lag = int(node["lag"])
        if str(node["time_basis"]) == "tick":
            out = np.full(self.n, np.nan)
            if self.n > lag:
                out[lag:] = source[lag:] - source[:-lag]
            return out
        time_s = np.asarray(self.book.data["time_s"], dtype=float)
        prior = np.searchsorted(time_s, time_s - float(lag), side="right") - 1
        out = np.full(self.n, np.nan)
        valid = (prior >= 0) & np.isfinite(source)
        out[valid] = source[valid] - source[prior[valid]]
        return out

    def _op_rolling_zscore(self, node):
        if str(node["time_basis"]) != "tick":
            raise ValueError("clock 기준 z-score 는 별도 확장이 필요하다")
        return _zscore(np.asarray(self.evaluate(node["input"]), dtype=float),
                       window=int(node["window"]),
                       min_observations=int(node["min_observations"]),
                       ddof=int(node.get("ddof", 1)))

    def _op_rolling_sum(self, node):
        if str(node["time_basis"]) != "tick":
            raise ValueError("clock 기준 rolling sum 은 별도 확장이 필요하다")
        return _rolling_sum(np.asarray(self.evaluate(node["input"]), dtype=float),
                            window=int(node["window"]),
                            min_observations=int(node.get("min_observations", node["window"])))

    def _rolling(self, node, operation: str):
        if str(node["time_basis"]) != "tick":
            raise ValueError(f"clock 기준 {operation}은 지원하지 않는다")
        values = pd.Series(np.asarray(self.evaluate(node["input"]), dtype=float))
        rolling = values.rolling(
            window=int(node["window"]),
            min_periods=int(node.get("min_observations", node["window"])),
        )
        return np.asarray(getattr(rolling, operation)(), dtype=float)

    def _op_rolling_mean(self, node):
        return self._rolling(node, "mean")

    def _op_rolling_std(self, node):
        return self._rolling(node, "std")

    def _op_rolling_min(self, node):
        return self._rolling(node, "min")

    def _op_rolling_max(self, node):
        return self._rolling(node, "max")

    def _op_crossover(self, node):
        source = np.asarray(self.evaluate(node["input"]), dtype=float)
        threshold = self._threshold(node["threshold"])
        out = np.zeros(self.n, dtype=bool)
        if self.n < 2:
            return out
        strict = str(node["equality"]) == "strict"
        if str(node["direction"]) == "above":
            before = source[:-1] < threshold[:-1] if strict else source[:-1] <= threshold[:-1]
            out[1:] = before & (source[1:] > threshold[1:])
        else:
            before = source[:-1] > threshold[:-1] if strict else source[:-1] >= threshold[:-1]
            out[1:] = before & (source[1:] < threshold[1:])
        return out & np.isfinite(source)

    def _threshold(self, value: Any) -> np.ndarray:
        """고정 숫자 또는 tick별 임계 배열을 같은 모양으로 돌려준다."""
        if isinstance(value, str):
            if value not in self.thresholds:
                raise ValueError(f"실행 시점에도 풀리지 않은 임계: {value}")
            value = self.thresholds[value]
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(self.n, float(values))
        if values.shape != (self.n,):
            raise ValueError(f"임계 배열 모양이 {values.shape} 이다. ({self.n},) 여야 한다")
        return values

    def _op_persistence(self, node):
        if str(node["time_basis"]) != "tick":
            raise ValueError("clock 기준 persistence 는 별도 확장이 필요하다")
        condition = np.asarray(self.evaluate(node["condition"]), dtype=bool)
        window = int(node["window"])
        counts = _rolling_sum(condition.astype(float), window=window, min_observations=window)
        return np.isfinite(counts) & (counts >= int(node["min_true"]))

    def _op_sequence(self, node):
        """A 가 켜진 뒤 `[min_lag, max_lag]` 안에 B 가 켜지면 참.

        setup 이나 trigger 가 켜진 틱에서만 상태가 바뀌므로 그 틱만 순회한다.
        """
        setup = np.asarray(self.evaluate(node["setup"]), dtype=bool)
        trigger = np.asarray(self.evaluate(node["trigger"]), dtype=bool)
        min_lag, max_lag = float(node["min_lag"]), float(node["max_lag"])
        ticks = str(node["time_basis"]) == "tick"
        time_s = np.asarray(self.book.data["time_s"], dtype=float)
        out = np.zeros(self.n, dtype=bool)
        armed: int | None = None
        for raw in np.flatnonzero(setup | trigger):
            index = int(raw)
            if armed is not None:
                distance = float(index - armed) if ticks else float(time_s[index] - time_s[armed])
                if distance > max_lag:
                    armed = None
                elif min_lag <= distance and trigger[index]:
                    out[index] = True
                    armed = None
                    continue
            if setup[index]:
                armed = index
        return out

    def _op_sequence_once(self, node):
        """한 setup 연속 구간에서는 첫 sequence 사건만 남긴다."""
        setup = np.asarray(self.evaluate(node["setup"]), dtype=bool)
        sequence = self._op_sequence(node)
        out = np.zeros(self.n, dtype=bool)
        consumed = False
        was_setup = False
        for index in range(self.n):
            if setup[index] and not was_setup:
                consumed = False
            if sequence[index] and not consumed:
                out[index] = True
                consumed = True
            was_setup = bool(setup[index])
        return out

    def _op_level_aggregate(self, node):
        start, end = (int(v) for v in node["levels"])
        field = str(node["field"])
        policy = str(node["missing_policy"])
        reducer = str(node["reducer"])
        sides = ["bid", "ask"] if str(node["side"]) == "both" else [str(node["side"])]
        results = []
        for side in sides:
            key = f"{side}_{'price' if field == 'price' else 'qty'}"
            array = np.asarray(self.book.data[key], dtype=float)[:, start - 1: end]
            valid = np.isfinite(array) & (array > 0 if field == "price" else array >= 0)
            if policy == "reject" and not valid.all():
                raise ValueError("호가 레벨이 비었는데 missing_policy=reject 다")
            filled = np.where(valid, array, 0.0 if policy == "zero" else np.nan)
            results.append({"sum": lambda a: np.nansum(a, axis=1),
                            "mean": lambda a: np.nanmean(a, axis=1),
                            "first": lambda a: a[:, 0],
                            "last": lambda a: a[:, -1]}[reducer](filled))
        return results[0] if len(results) == 1 else np.column_stack(results).sum(axis=1)


def _rolling_sum(values: np.ndarray, *, window: int, min_observations: int) -> np.ndarray:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(filled)))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    total = prefix[indices + 1] - prefix[starts]
    count = counts[indices + 1] - counts[starts]
    out = np.full(len(values), np.nan)
    enough = count >= min_observations
    out[enough] = total[enough]
    return out


def _zscore(values: np.ndarray, *, window: int, min_observations: int, ddof: int) -> np.ndarray:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(filled)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(filled * filled)))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    count = counts[indices + 1] - counts[starts]
    total = prefix[indices + 1] - prefix[starts]
    total_sq = prefix_sq[indices + 1] - prefix_sq[starts]
    zeros = np.zeros(len(values))
    mean = np.divide(total, count, out=zeros.copy(), where=count > 0)
    centered = total_sq - np.divide(total * total, count, out=zeros.copy(), where=count > 0)
    denominator = count - ddof
    variance = np.maximum(np.divide(centered, denominator, out=zeros.copy(), where=denominator > 0), 0.0)
    usable = valid & (count >= min_observations) & (denominator > 0) & (variance > 1.0e-12)
    out = np.full(len(values), np.nan)
    out[usable] = (values[usable] - mean[usable]) / np.sqrt(variance[usable])
    return out


def prior_tick_quantile(values: np.ndarray, quantile: float, *,
                        method: str = "higher",
                        window: int = ROLLING_QUANTILE_WINDOW) -> np.ndarray:
    """tick t의 cut을 ``t-100``부터 ``t-1``까지만으로 계산한다.

    현재 tick과 이후 tick은 창에 들어가지 않는다. 결측이 하나라도 있으면 그 100-tick
    창은 cut을 만들지 않는다.
    """
    if window < 1:
        raise ValueError("rolling quantile window는 1 이상이어야 한다")
    if not 0.0 <= float(quantile) < 1.0:
        raise ValueError("rolling quantile q는 0 이상 1 미만이어야 한다")
    return pd.Series(np.asarray(values, dtype=float)).shift(1).rolling(
        window=window, min_periods=window).quantile(float(quantile), interpolation=method).to_numpy()


def prior_tick_quantile_grid(values: np.ndarray, quantiles: Sequence[tuple[float, str]], *,
                             window: int = ROLLING_QUANTILE_WINDOW) -> dict[tuple[float, str], np.ndarray]:
    """같은 직전 tick 창에서 여러 q cut을 한 번에 구한다.

    `prior_tick_quantile`과 같은 ``higher``/``lower`` 정의다. q 후보마다 창을 다시
    정렬하지 않고, 필요한 순위만 한 번의 partial sort에서 꺼낸다.
    """
    if window < 1:
        raise ValueError("rolling quantile window는 1 이상이어야 한다")
    requests = tuple((float(q), str(method)) for q, method in quantiles)
    for q, method in requests:
        if not 0.0 <= q < 1.0:
            raise ValueError("rolling quantile q는 0 이상 1 미만이어야 한다")
        if method not in {"higher", "lower"}:
            raise ValueError(f"rolling quantile method가 없다: {method}")

    source = np.asarray(values, dtype=float)
    result = {request: np.full(len(source), np.nan) for request in requests}
    if not requests or len(source) <= window:
        return result

    # tick t의 창은 values[t-window:t]다. 마지막 sliding window는 존재하지 않는
    # tick n의 창이므로 제외한다.
    windows = np.lib.stride_tricks.sliding_window_view(source, window_shape=window)[:-1]
    positions = {
        (q, method): (int(np.ceil((window - 1) * q)) if method == "higher"
                      else int(np.floor((window - 1) * q)))
        for q, method in requests
    }
    partitioned = np.partition(windows, kth=sorted(set(positions.values())), axis=1)
    finite = np.isfinite(source).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(finite)))
    complete = (prefix[window:len(source)] - prefix[:len(source) - window]) == window
    for request, position in positions.items():
        tail = result[request][window:]
        tail[complete] = partitioned[complete, position]
    return result


def _rolling_quantile_bindings(data: Mapping[str, np.ndarray], program: Mapping[str, Any],
                               quantiles: Mapping[str, float]) -> dict[str, np.ndarray]:
    """미결정 자리마다 그 자리의 입력식으로 직전 100 tick cut을 만든다."""
    runtime = ExpressionRuntime(data)
    bindings: dict[str, np.ndarray] = {}
    for placeholder, node in _unresolved_nodes(program["signal"]):
        name = placeholder.removeprefix(catalog.UNRESOLVED_PREFIX)
        if name not in quantiles:
            raise ValueError(f"rolling q가 없는 파라미터: {name}")
        selector = _selector(node)
        lower = selector in {"<", "<=", "below"}
        q = 1.0 - float(quantiles[name]) if lower else float(quantiles[name])
        method = "lower" if lower else "higher"
        values = np.asarray(runtime.evaluate(node["input"]), dtype=float)
        binding = prior_tick_quantile(values, q, method=method)
        if placeholder in bindings and not np.array_equal(bindings[placeholder], binding, equal_nan=True):
            raise ValueError(f"같은 파라미터 {name}가 서로 다른 tick 기준을 가리킨다")
        bindings[placeholder] = binding
    return bindings


def rolling_quantile_binding_grid(
    data: Mapping[str, np.ndarray], program: Mapping[str, Any],
    quantile_sets: Sequence[Mapping[str, float]],
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """동일한 진입식의 q 후보들에 필요한 tick cut을 함께 만든다.

    반환값 앞부분은 q 후보 순서와 같은 threshold 묶음이고, 뒷부분은 q와 무관한
    입력식 cache다. 최종 bool signal과 원장은 후보별로 별도 계산한다.
    """
    runtime = ExpressionRuntime(data)
    bindings = [{} for _ in quantile_sets]
    expression_cache: dict[str, np.ndarray] = {}
    for placeholder, node in _unresolved_nodes(program["signal"]):
        name = placeholder.removeprefix(catalog.UNRESOLVED_PREFIX)
        selector = _selector(node)
        lower = selector in {"<", "<=", "below"}
        method = "lower" if lower else "higher"
        values = np.asarray(runtime.evaluate(node["input"]), dtype=float)
        expression_cache[sha256_json(node["input"])] = values
        requests = []
        for quantiles in quantile_sets:
            if name not in quantiles:
                raise ValueError(f"rolling q가 없는 파라미터: {name}")
            q = 1.0 - float(quantiles[name]) if lower else float(quantiles[name])
            requests.append((q, method))
        cuts = prior_tick_quantile_grid(values, requests)
        for index, request in enumerate(requests):
            binding = cuts[request]
            if (placeholder in bindings[index]
                    and not np.array_equal(bindings[index][placeholder], binding, equal_nan=True)):
                raise ValueError(f"같은 파라미터 {name}가 서로 다른 tick 기준을 가리킨다")
            bindings[index][placeholder] = binding
    return bindings, expression_cache


def uses_rolling_quantiles(contract: Mapping[str, Any]) -> bool:
    """이 계약의 미결정 임계가 직전 100 tick q인지 확인한다."""
    kinds = {
        str((item.get("threshold_source") or {}).get("kind"))
        for item in (contract.get("parameter_interface") or {}).values()
    }
    return "rolling_prior_100_ticks_quantile" in kinds


def entry_signal(data: Mapping[str, np.ndarray], contract: Mapping[str, Any], *,
                 rolling_quantiles: Mapping[str, float] | None = None,
                 rolling_thresholds: Mapping[str, np.ndarray] | None = None,
                 precomputed_expressions: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """계약의 진입 신호. 틱마다 bool.

    "그 틱에 사고 싶다" 는 뜻이지 "샀다" 는 뜻이 아니다. 포지션 상태와 체결 여부는
    `ledger.py` 가 정한다.

    `warmup_ticks` 만큼 앞부분은 끈다. rolling 통계가 자리를 잡기 전의 값은 관측이
    아니라 초기화 잔재다.
    """
    program = contract["entry_program"]
    thresholds = (dict(rolling_thresholds) if rolling_thresholds is not None else
                  _rolling_quantile_bindings(data, program, rolling_quantiles or {})
                  if uses_rolling_quantiles(contract) else None)
    signal = np.asarray(ExpressionRuntime(
        data, thresholds=thresholds, precomputed_expressions=precomputed_expressions,
    ).evaluate(program["signal"]), dtype=bool)
    warmup = int(program.get("warmup_ticks", 0))
    if warmup > 0:
        signal[:min(warmup, len(signal))] = False
    return signal


def resolve_thresholds(
    program: Mapping[str, Any],
    cases: Sequence[tuple[str, Mapping[str, np.ndarray]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`"UNRESOLVED:<name>"` 자리를 발굴 구간 분위수로 채운다.

    Agent 가 숫자를 찍게 두면 저마다 다른 값을 쓰고, 값 범위를 모르면 조건이 항상
    참이거나 항상 거짓이 된다. 그래서 임계는 데이터가 정한다.

    방향에 따라 분위수를 고른다 — `<`/`<=` 는 0.25, `>`/`>=` 는 0.75, `==` 는 0.50.
    `crossover` 도 임계 자리다 — `below` 는 0.25, `above` 는 0.75 이고, 행의
    `comparator` 에는 그 방향을 적는다.
    고른 분위수와 표본 수를 행으로 남긴다. 남기지 않으면 그 계약을 설명할 수 없다.
    """
    result = deepcopy(dict(program))
    nodes = _unresolved_nodes(result["signal"])
    if not nodes:
        return result, []

    samples: dict[str, list[np.ndarray]] = {name: [] for name, _ in nodes}
    inputs = {name: node["input"] for name, node in nodes}
    ops = {name: str(node["op"]) for name, node in nodes}
    comparators = {name: _selector(node) for name, node in nodes}
    for case_id, data in cases:
        runtime = ExpressionRuntime(data)
        for name, expression in inputs.items():
            try:
                values = np.asarray(runtime.evaluate(expression), dtype=float)
            except Exception:
                continue   # 그 종목-일에서 못 만드는 feature 는 건너뛴다
            finite = values[np.isfinite(values)]
            if len(finite):
                samples[name].append(finite)

    chosen: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for name in sorted(samples):
        usable = [s for s in samples[name] if len(s)]
        if not usable:
            raise ValueError(f"{name} 을 채울 발굴 관측이 없다")
        vector = np.concatenate(usable)
        quantile = QUANTILE_BY_SELECTOR[comparators[name]]
        value = float(np.quantile(vector, quantile))
        chosen[name] = value
        rows.append({"placeholder": name, "op": ops[name], "comparator": comparators[name],
                     "quantile": quantile, "value": value,
                     "observations": int(len(vector)), "cases": len(usable),
                     "uses_future_returns": False, "uses_oos": False})
    return _substitute(result, chosen), rows


# 임계 자리를 가진 연산자와, 그 자리를 담은 인자 이름. 카탈로그가 정본이다.
THRESHOLD_ARGUMENTS = {op.op: op.thresholds for op in catalog.OPERATORS.values() if op.thresholds}

QUANTILE_BY_SELECTOR = {"<": 0.25, "<=": 0.25, ">": 0.75, ">=": 0.75, "==": 0.50,
                        "below": 0.25, "above": 0.75}


def _selector(node: Mapping[str, Any]) -> str:
    """분위수를 고를 근거. `compare` 는 부등호, `crossover` 는 방향이다."""
    return str(node["direction"] if str(node["op"]) == "crossover" else node["comparator"])


def _unresolved_nodes(node: Any, out: list | None = None) -> list[tuple[str, Mapping[str, Any]]]:
    out = [] if out is None else out
    if isinstance(node, Mapping):
        for name in THRESHOLD_ARGUMENTS.get(str(node.get("op")), ()):
            value = node.get(name)
            if isinstance(value, str) and value.startswith(catalog.UNRESOLVED_PREFIX):
                out.append((value, node))
        for child in node.values():
            _unresolved_nodes(child, out)
    elif isinstance(node, list):
        for child in node:
            _unresolved_nodes(child, out)
    return out


def _substitute(node: Any, chosen: Mapping[str, float]) -> Any:
    if isinstance(node, Mapping):
        return {k: _substitute(v, chosen) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, chosen) for v in node]
    if isinstance(node, str) and node in chosen:
        return chosen[node]
    return node


def contract_hash(contract: Mapping[str, Any]) -> str:
    """계약의 해시. 임계까지 포함하므로 해소 전후가 다른 값이다."""
    # 생성 시각은 계약의 실행 의미가 아니다. 이것까지 해시하면 같은 계약을 다시
    # 컴파일할 때마다 다른 계약으로 보여 shared replay를 다시 쓰거나 거부하게 된다.
    return sha256_json({k: v for k, v in contract.items()
                        if k not in {"contract_sha256", "created_at"}})
