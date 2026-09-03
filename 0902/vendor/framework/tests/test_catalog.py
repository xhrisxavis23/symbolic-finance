"""카탈로그가 지켜야 할 것. 데이터 없이 도는 합성 호가로 검사한다.

여기서 보는 것은 카탈로그의 **계약**이지 어떤 feature 가 좋은가가 아니다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from framework import catalog
from framework.config import FEE_BPS, QUANTILES
from framework.contract import ExpressionRuntime


LEVELS = 10
TICK = 10.0


def arrays(n: int = 400, seed: int = 0, levels: int = LEVELS, trades: bool = True):
    """합성 호가-체결. 실데이터가 없어도 도는 최소 격자."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.integers(-1, 2, size=n))
    bid1 = 10_000.0 + walk * TICK
    ask1 = bid1 + TICK * rng.integers(1, 3, size=n)
    ladder = TICK * np.arange(levels, dtype=float)[None, :]
    data = {
        "bid_price": bid1[:, None] - ladder,
        "ask_price": ask1[:, None] + ladder,
        "bid_qty": rng.integers(1, 500, size=(n, levels)).astype(float),
        "ask_qty": rng.integers(1, 500, size=(n, levels)).astype(float),
        "local_time": np.arange(n, dtype=np.int64) * 100_000 + 90_000_000_000,
        "time_s": np.arange(n, dtype=float) * 0.1,
    }
    if trades:
        data["buy_volume"] = rng.integers(0, 50, size=n).astype(float)
        data["sell_volume"] = rng.integers(0, 50, size=n).astype(float)
    return data


def book(**kwargs) -> catalog.Book:
    return catalog.Book(arrays(**kwargs))


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN 자리까지 같은가. NaN != NaN 이라 그냥 비교하면 늘 다르다."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(
        np.array_equal(np.isnan(a), np.isnan(b))
        and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12))


# ---- 정체 --------------------------------------------------------------------

def test_registry_names_and_aliases_are_unique():
    assert len(catalog.FEATURES) == len(set(catalog.FEATURES))
    assert not set(catalog.ALIASES) & set(catalog.FEATURES)
    ids = [f.definition_id for f in catalog.FEATURES.values()]
    assert len(ids) == len(set(ids))


def test_aliases_resolve_to_canonical_names():
    assert catalog.resolve("bid_queue_depletion_5s_bps") == "bid_best_price_drop_count_50t"
    assert catalog.resolve("ask_queue_arrival_count_5s") == "ask_best_price_drop_count_50t"
    assert catalog.resolve("microprice_velocity_1t") == "microprice_velocity"
    assert catalog.resolve_feature("microprice_velocity_1t_bps").name == "microprice_velocity"


def test_agent_vocabulary_hides_aliases_and_implementation():
    vocabulary = catalog.agent_vocabulary()
    names = {f["name"] for f in vocabulary["features"]}
    assert not names & set(catalog.ALIASES)
    for entry in vocabulary["features"]:
        assert "compute" not in entry
        assert entry["observable_description"]
        assert "definition_id" in entry and "dimension" in entry
        assert entry["threshold_policy"]["allowed"]


def test_audit_reports_no_problems():
    assert catalog.audit()["problems"] == []


# ---- 타입 --------------------------------------------------------------------

def primitive(name: str) -> dict:
    return {"op": "primitive", "primitive_id": name}


def condition(name: str, value: float = 0.0) -> dict:
    return {"op": "compare", "input": primitive(name), "comparator": ">", "value": value}


def test_operator_infers_output_unit():
    zscore = {"op": "rolling_zscore", "input": primitive("mid_price"),
              "window": 50, "min_observations": 20, "time_basis": "tick"}
    assert catalog.infer_expression_type(zscore) == catalog.ZSCORE

    difference = {"op": "difference", "input": primitive("mid_price"),
                  "lag": 1, "time_basis": "tick"}
    assert catalog.infer_expression_type(difference).dimension == "price"

    total = {"op": "rolling_sum", "input": primitive("vol_flow"),
             "window": 10, "time_basis": "tick"}
    assert catalog.infer_expression_type(total).dimension == "quantity"

    assert catalog.infer_expression_type(condition("book_imbalance")) == catalog.BOOLEAN


def test_operators_take_no_free_form_unit():
    for operator in catalog.OPERATORS.values():
        assert "unit" not in operator.required
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression({"op": "difference", "input": primitive("mid_price"),
                                     "lag": 1, "unit": "ticks"})


def test_type_mismatch_fails_before_execution():
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression({"op": "all", "args": [primitive("mid_price"),
                                                           condition("book_imbalance")]})
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression({"op": "sequence", "setup": primitive("mid_price"),
                                     "trigger": primitive("book_imbalance"),
                                     "min_lag": 1, "max_lag": 8, "time_basis": "tick"})
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression({"op": "compare", "input": condition("book_imbalance"),
                                     "comparator": ">", "value": 0.5})


def test_unknown_feature_and_operator_fail():
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression(primitive("queue_arrival_rate"))
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression({"op": "smooth", "input": primitive("mid_price")})


def test_thresholds_and_structural_params_are_separate():
    unresolved = {"op": "compare", "input": primitive("book_imbalance"),
                  "comparator": ">", "value": "UNRESOLVED:pressure"}
    catalog.validate_expression(unresolved, allow_unresolved=True)
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression(unresolved)
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression(
            {"op": "persistence", "condition": condition("book_imbalance"),
             "window": "UNRESOLVED:window", "min_true": 3, "time_basis": "tick"},
            allow_unresolved=True)


# ---- 데이터 의존 --------------------------------------------------------------

def test_shallow_book_hides_deep_features():
    capabilities = catalog.DataCapabilities.from_arrays(arrays(levels=5))
    available = catalog.available_features(capabilities)
    assert "ofi_depth_5" in available
    assert "ofi_depth_10" not in available
    reason = {a.feature: a for a in catalog.feature_availability(capabilities)}["ofi_depth_10"]
    assert not reason.available and "depth" in reason.reason


def test_missing_trades_are_reported_not_silently_dropped():
    quote_only = book(trades=False)
    reasons = {a.feature: a for a in catalog.feature_availability(quote_only.capabilities)}
    assert not reasons["signed_aggr_flow_20"].available
    assert "buy_volume" in reasons["signed_aggr_flow_20"].reason
    assert "signed_aggr_flow_20" not in catalog.compute_features(catalog.list_features(), quote_only)
    assert "book_imbalance" in catalog.compute_features(catalog.list_features(), quote_only)


# ---- 인과성 ------------------------------------------------------------------

def test_future_ticks_do_not_change_past_values():
    base = arrays(n=400, seed=3)
    tampered = {k: np.array(v, copy=True) for k, v in base.items()}
    cut = 300
    tampered["bid_price"][cut:] += 500.0
    tampered["ask_price"][cut:] += 500.0
    tampered["bid_qty"][cut:] = 1.0
    tampered["ask_qty"][cut:] = 999.0
    tampered["buy_volume"][cut:] = 0.0
    tampered["sell_volume"][cut:] = 1000.0

    before = catalog.compute_features(catalog.list_features(), catalog.Book(base))
    after = catalog.compute_features(catalog.list_features(), catalog.Book(tampered))
    assert set(before) == set(after)
    for name, values in before.items():
        assert same(values[:cut], after[name][:cut]), name


def test_every_agent_visible_feature_is_causal():
    policy = catalog.default_policy()
    exposed = policy.hypothesis_visible | policy.grounding_visible | policy.guard_searchable
    assert all(catalog.FEATURES[name].causal for name in exposed)


# ---- 정의와 구현 --------------------------------------------------------------

def test_definition_matches_optimized_compute():
    data = arrays(n=300, seed=5)
    runtime = ExpressionRuntime(data)
    subject = catalog.Book(data)
    defined = [f for f in catalog.FEATURES.values() if f.definition is not None]
    assert defined, "definition 을 가진 feature 가 하나도 없다"
    for spec in defined:
        assert same(spec.compute(subject), runtime.evaluate(spec.definition)), spec.name
        assert catalog.infer_expression_type(spec.definition) == spec.type_info


# ---- 해시와 정책 --------------------------------------------------------------

def test_catalog_hash_detects_definition_change():
    before = catalog.catalog_hash()
    original = catalog.FEATURES["book_imbalance"]
    catalog.FEATURES["book_imbalance"] = replace(original, lookback=7)
    try:
        assert catalog.catalog_hash() != before
    finally:
        catalog.FEATURES["book_imbalance"] = original
    assert catalog.catalog_hash() == before


def test_policy_is_not_part_of_catalog_hash():
    before = catalog.catalog_hash()
    narrow = replace(catalog.default_policy(), guard_searchable=frozenset({"book_imbalance"}))
    assert catalog.guard_axes(narrow) == ("book_imbalance",)
    assert catalog.policy_hash(narrow) != catalog.policy_hash()
    assert catalog.catalog_hash() == before


def test_guard_axes_follow_feature_use_policy():
    axes = set(catalog.guard_axes())
    assert not axes & {"mid_price", "bid_1", "ask_1", "trade_event_indicator",
                       "spread_bps", "spread_to_round_trip_cost_ratio",
                       "microprice_dev_bps", "queue_imbalance_best"}
    assert "book_imbalance" in axes and "ofi_cks_1" in axes


def test_feature_use_policy_keeps_one_axis_for_l1_imbalance():
    microprice = catalog.feature_use_policy("microprice_dev_bps")
    queue = catalog.feature_use_policy("queue_imbalance_best")
    names = {f["name"] for f in catalog.agent_vocabulary()["features"]}
    assert microprice.duplicate_of == queue.duplicate_of == "book_imbalance"
    assert "book_imbalance" in names
    assert not {"microprice_dev_bps", "queue_imbalance_best"} & names


def test_spread_cost_ratio_uses_the_configured_round_trip_cost():
    subject = book(seed=14)
    values = catalog.compute_features(
        ("spread_bps", "spread_to_round_trip_cost_ratio"), subject)
    assert same(values["spread_to_round_trip_cost_ratio"], values["spread_bps"] / FEE_BPS)


def test_quintile_boundaries_are_available_for_prior_day_guards():
    assert {0.20, 0.40, 0.80} <= set(QUANTILES)


# ---- 관측 profile -------------------------------------------------------------

def measured(seed: int, dataset_hash: str) -> catalog.ObservationProfile:
    return catalog.measure_profile(
        [("A/1", book(seed=seed))],
        profile_id=f"synthetic_{seed}", dataset_hash=dataset_hash,
        market="TEST", date_range=("20260101", "20260101"), symbols=("A",))


def test_same_catalog_different_dataset_changes_only_the_profile():
    first, second = measured(11, "d1"), measured(12, "d2")
    assert first.catalog_hash == second.catalog_hash == catalog.catalog_hash()
    assert catalog.profile_hash(first) != catalog.profile_hash(second)


def test_same_dataset_reproduces_the_same_profile():
    assert catalog.profile_hash(measured(11, "d1")) == catalog.profile_hash(measured(11, "d1"))


def test_profile_carries_statistics_not_definitions():
    profile = measured(11, "d1")
    assert set(profile.stats) <= set(catalog.FEATURES)
    assert all(not hasattr(f, "observed_range") for f in catalog.FEATURES.values())
    stats = profile.stats["book_imbalance"]
    assert stats.min <= stats.p1 <= stats.p99 <= stats.max
    assert stats.observations > 0


def test_structural_arguments_must_be_literal_integers():
    """실행기가 `int(...)` 로 조용히 잘라내기 전에 잡아야 한다."""
    def total(window):
        return {"op": "rolling_sum", "input": primitive("mid_price"),
                "window": window, "time_basis": "tick"}
    catalog.validate_expression(total(10))
    for bad in (2.5, "8", True, None):
        with pytest.raises(catalog.ExpressionError):
            catalog.validate_expression(total(bad))
    with pytest.raises(catalog.ExpressionError):
        catalog.validate_expression(
            {"op": "level_aggregate", "side": "bid", "field": "quantity",
             "levels": [1, 2, 3], "reducer": "sum", "missing_policy": "mask"})
    catalog.validate_expression(
        {"op": "level_aggregate", "side": "bid", "field": "quantity",
         "levels": [1, 5], "reducer": "sum", "missing_policy": "mask"})


def test_every_feature_states_its_invalid_value_policy():
    assert all(f.invalid_policy for f in catalog.FEATURES.values())
    before = catalog.catalog_hash()
    original = catalog.FEATURES["book_imbalance"]
    catalog.FEATURES["book_imbalance"] = replace(original, invalid_policy="0 으로 채운다")
    try:
        assert catalog.catalog_hash() != before   # 결측 정책은 계산 정의의 일부다
    finally:
        catalog.FEATURES["book_imbalance"] = original


def test_legacy_contract_migrates_with_a_record():
    legacy = {"op": "all", "args": [
        {"op": "compare", "input": primitive("ask_queue_arrival_count_5s"),
         "comparator": ">", "value": 3},
        {"op": "compare", "comparator": "<", "value": 0.0, "input":
            {"op": "difference", "input": primitive("microprice_velocity_1t_bps"),
             "lag": 1, "unit": "ticks"}}]}
    migrated, changes = catalog.migrate_expression(legacy)
    catalog.validate_expression(migrated)          # 옮긴 결과는 그대로 실행 가능
    assert {c["legacy"] for c in changes} == {
        "ask_queue_arrival_count_5s", "microprice_velocity_1t_bps", "unit=ticks"}
    text = repr(migrated)
    assert "ask_queue_arrival_count_5s" not in text and "unit" not in text


def test_audit_can_check_definition_against_compute():
    clean = catalog.audit(parity_book=book(seed=9))
    assert clean["definition_parity"] == "checked" and clean["problems"] == []
    original = catalog.FEATURES["book_imbalance_velocity"]
    catalog.FEATURES["book_imbalance_velocity"] = replace(
        original, definition={"op": "difference", "lag": 2, "time_basis": "tick",
                              "input": primitive("book_imbalance")})
    try:
        broken = catalog.audit(parity_book=book(seed=9))
        assert any("definition 과 compute" in p for p in broken["problems"])
    finally:
        catalog.FEATURES["book_imbalance_velocity"] = original


def test_ledger_columns_can_be_traced_to_a_definition():
    ids = catalog.definition_ids(catalog.guard_axes())
    assert set(ids) == set(catalog.guard_axes())
    assert ids["book_imbalance"] == "book_imbalance.v1"


def test_agent_vocabulary_carries_no_internal_hashes():
    v = catalog.agent_vocabulary()
    assert not [k for k in v if "sha" in k]
    assert all(f["invalid_policy"] for f in v["features"])


def test_provenance_names_catalog_profile_and_policy():
    keys = set(catalog.provenance())
    assert keys == {"catalog_sha256", "observation_profile_id",
                    "observation_profile_sha256", "workflow_policy_sha256"}
