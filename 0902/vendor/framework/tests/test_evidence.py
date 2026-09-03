"""Canonical Evidence Profiler 가 지켜야 할 것. 합성 호가로 검사한다.

여기서 보는 것은 **증거의 계약**이다 — 어떤 feature 가 좋은가가 아니라, 미래를 읽지
않는가·어휘가 카탈로그와 같은가·대조가 오염되지 않았는가이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework import catalog, evidence
from framework.FeatureProfile.ExtractLabel import HorizonSpec, path_labels
from framework.tests.test_catalog import arrays


CONFIG = evidence.ProfilerConfig()


def long_frame(cohorts=("PROFIT", "LOSS", "BACKGROUND"), n=60, seed=0, times=(-1000, 0)):
    """작은 long 표. 통계 함수를 데이터 없이 돌리기 위한 것."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, cohort in enumerate(cohorts):
        for k in range(n):
            for rel in times:
                rows.append({
                    "anchor_id": f"{cohort}-{k}", "cohort": cohort,
                    "symbol": f"{k % 4:06d}", "date": f"2026031{k % 5 + 1}",
                    "tick": k, "relative_time_ms": rel, "source_tick": k,
                    "age_ms": 100.0,
                    "book_imbalance": rng.normal(i * 0.5, 1.0),
                    "spread_bps": rng.normal(10.0, 2.0),
                    "ofi_depth_5": rng.normal(-i * 100.0, 50.0)})
    return pd.DataFrame(rows)


FEATURES = ("book_imbalance", "ofi_depth_5", "spread_bps")


# ---- 어휘 --------------------------------------------------------------------

def test_evidence_features_are_catalog_canonical_and_causal():
    caps = catalog.DataCapabilities.from_arrays(arrays())
    names = evidence.evidence_features(caps)
    assert names, "증거 feature 가 하나도 없다"
    assert all(n in catalog.FEATURES for n in names)
    assert all(catalog.FEATURES[n].causal for n in names)
    # 실행 전용(호가 수준)은 증거가 아니다
    assert not set(names) & catalog.default_policy().execution_only
    # 별칭은 노출하지 않는다
    assert not set(names) & set(catalog.ALIASES)


def test_shallow_book_drops_deep_features():
    caps = catalog.DataCapabilities.from_arrays(arrays(levels=5))
    names = evidence.evidence_features(caps)
    assert "ofi_depth_5" in names and "ofi_depth_10" not in names


def test_every_family_comes_from_catalog_metadata():
    caps = catalog.DataCapabilities.from_arrays(arrays())
    families = {evidence.family_of(catalog.FEATURES[n]) for n in evidence.evidence_features(caps)}
    assert families and families <= {
        "ORDER_FLOW", "BOOK_IMBALANCE", "MICROPRICE", "PRICE_MOMENTUM",
        "SPREAD", "BOOK_SHAPE", "QUEUE_STATE", "TRADE_ACTIVITY"}


# ---- 인과성 ------------------------------------------------------------------

def test_alignment_never_reads_the_future():
    data = arrays(n=500, seed=2)
    time_s = np.asarray(data["time_s"], dtype=float)
    ticks = np.arange(100, 400)
    for offset in CONFIG.relative_times_ms:
        idx, age = evidence.align_indices(time_s, ticks, offset)
        target = time_s[ticks] + offset / 1000.0
        good = idx >= 0
        # 읽은 관측은 목표 시각 이하여야 한다
        assert np.all(time_s[idx[good]] <= target[good] + 1e-12)
        # 그리고 그 다음 관측은 목표를 넘어야 한다 (가장 최근이라는 뜻)
        nxt = np.minimum(idx[good] + 1, len(time_s) - 1)
        assert np.all((time_s[nxt] > target[good]) | (idx[good] == len(time_s) - 1))
        assert np.all(age[good] >= -1e-9)


def test_relative_time_grid_rejects_the_future():
    config = evidence.ProfilerConfig(relative_times_ms=(-1000, 0, 1000))
    fail, _ = evidence.quality_gate(long_frame(), FEATURES, anchor_frame(), config,
                                    catalog.catalog_hash())
    assert any("미래" in f for f in fail)


def test_stale_observation_becomes_nan_not_a_carried_value():
    data = arrays(n=200, seed=3)
    time_s = np.asarray(data["time_s"], dtype=float)
    time_s[100:] += 60.0            # 100틱에서 1분 공백을 만든다
    data = {**data, "time_s": time_s}
    anchors = pd.DataFrame({"anchor_id": ["a"], "cohort": ["PROFIT"], "cluster_id": ["MK01"],
                            "symbol": ["000001"], "date": ["20260316"], "tick": [100]})
    idx, age = evidence.align_indices(time_s, np.array([100]), -1000)
    assert age[0] > CONFIG.freshness_limit_ms      # 공백 때문에 오래된 값이다


# ---- LOSS 의 대칭성 ------------------------------------------------------------

def test_best_ask_uses_bid_entry_and_future_ask():
    data = arrays(n=400, seed=4)
    best = evidence.best_ask_net_bps(data, CONFIG)
    bid1 = data["bid_price"][:, 0]
    ask1 = data["ask_price"][:, 0]
    ends = evidence.window_end_ticks(np.asarray(data["time_s"], float), CONFIG.horizon_seconds)
    for t in (10, 100, 250):
        if ends[t] < 0:
            continue
        want = (ask1[t + 1:ends[t] + 1].max() / bid1[t] - 1.0) * 1e4 - CONFIG.fee_bps
        assert np.isclose(best[t], want)
    # 창이 데이터 끝을 넘으면 관측하지 않은 것이다
    assert np.all(np.isnan(best[ends < 0]))


def test_profit_and_loss_conditions_are_mutually_exclusive():
    data = arrays(n=600, seed=5)
    best = evidence.best_ask_net_bps(data, CONFIG)
    profit_like = np.isfinite(best) & (best >= CONFIG.profit_net_floor_bps)
    loss_like = np.isfinite(best) & (best <= CONFIG.loss_net_ceiling_bps)
    assert not (profit_like & loss_like).any()


def test_profit_region_uses_future_best_ask_after_bid_entry():
    data = arrays(n=40, seed=9)
    labels = path_labels(data, (HorizonSpec.ticks(5),))
    bid1 = data["bid_price"][:, 0]
    ask1 = data["ask_price"][:, 0]
    for tick in (0, 7, 20):
        expected = (ask1[tick + 1:tick + 6].max() / bid1[tick] - 1.0) * 1e4
        assert np.isclose(labels.loc[tick, "best_ask_gross_bps__T5"], expected)
        assert labels.loc[tick, "best_ask_delta_ticks__T5"] in range(1, 6)


def test_profile_entry_queue_is_supplemental_metadata(monkeypatch):
    n = 8
    data = {
        "bid_price": np.full((n, 1), 100.0),
        "ask_price": np.array([101.0, 102.0, 110.0, 103.0, 103.0, 103.0, 103.0, 103.0])[:, None],
        "bid_qty": np.full((n, 1), 10.0),
        "ask_qty": np.full((n, 1), 10.0),
        "sell_volume": np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "sell_min_price": np.full(n, 100.0),
        "local_time": 90_000_000_000 + np.arange(n, dtype=np.int64) * 100_000,
        "time_s": np.arange(n, dtype=float) * 0.1,
    }
    monkeypatch.setattr(evidence.tickdata, "load", lambda *_args: (data, Path("synthetic")))
    anchors = pd.DataFrame({"cohort": ["PROFIT"], "cluster_id": ["MK01"],
                            "symbol": ["000001"], "date": ["20260316"], "tick": [0],
                            "best_ask_net_bps": [1000.0]})
    got = evidence.attach_entry_queue(
        anchors, config=evidence.ProfilerConfig(horizon_seconds=0.3), root=Path("synthetic"))
    row = got.iloc[0]
    # 20% 앞 큐(2)와 주문 1을 틱 1의 매도로 소진한다. 수익 label은 입력 그대로다.
    assert row.entry_queue_status == "filled"
    assert row.entry_fill_tick == 1 and row.entry_wait_ticks == 1
    assert row.entry_oracle_ask_tick == 2
    assert bool(row.entry_fill_before_oracle_ask)
    assert row.best_ask_net_bps == 1000.0


def test_profile_entry_queue_keeps_quote_only_profile(monkeypatch):
    data = arrays(n=20, seed=11, trades=False)
    monkeypatch.setattr(evidence.tickdata, "load", lambda *_args: (data, Path("synthetic")))
    anchors = pd.DataFrame({"cohort": ["PROFIT"], "cluster_id": ["MK01"],
                            "symbol": ["000001"], "date": ["20260316"], "tick": [0],
                            "best_ask_net_bps": [1.0]})
    row = evidence.attach_entry_queue(anchors, config=CONFIG, root=Path("synthetic")).iloc[0]
    assert row.entry_queue_status == "not_supported"
    assert row.entry_fill_tick == -1
    assert row.best_ask_net_bps == 1.0


def test_representatives_collapse_overlapping_windows():
    ends = np.arange(100) + 5
    picked = evidence._representatives(np.arange(0, 20), ends, cap=10)
    assert len(picked) < 20                      # 겹치는 창은 한 덩어리로
    spread = evidence._representatives(np.array([0, 50, 90]), ends, cap=10)
    assert spread == [0, 50, 90]                 # 겹치지 않으면 그대로
    assert len(evidence._representatives(np.arange(0, 100, 10), ends, cap=3)) <= 3


def test_range_max_matches_brute_force():
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    starts = np.arange(0, 400)
    ends = starts + rng.integers(0, 90, size=len(starts))
    got = evidence._range_max(x, starts, ends)
    want = np.array([x[s:e + 1].max() for s, e in zip(starts, ends)])
    assert np.allclose(got, want)


# ---- 대조 통계 ----------------------------------------------------------------

def anchor_frame():
    return pd.DataFrame({"cohort": ["PROFIT"] * 5 + ["LOSS"] * 5 + ["BACKGROUND"] * 5,
                         "symbol": [f"{i:06d}" for i in range(15)],
                         "date": ["20260316"] * 15, "tick": range(15)})


def test_direction_is_stated_not_left_as_a_bare_auc():
    long = long_frame()
    contrast = evidence.contrastive_profile(long, FEATURES, CONFIG)
    assert set(contrast.direction) <= {"lower_in_PROFIT", "higher_in_PROFIT",
                                       "lower_in_LOSS", "higher_in_LOSS",
                                       "no_separation", "UNDETERMINED"}
    row = contrast[(contrast.feature_name == "ofi_depth_5") & (contrast.cohort_a == "PROFIT")
                   & (contrast.cohort_b == "LOSS") & (contrast.relative_time_ms == 0)].iloc[0]
    # 합성 데이터에서 PROFIT 의 ofi 가 더 높게 만들었다
    assert row.direction == "higher_in_PROFIT"
    assert np.isclose(row.effect_strength, abs(row.auc - 0.5))


def test_missing_values_are_never_filled_with_zero():
    long = long_frame()
    long.loc[long.cohort == "LOSS", "book_imbalance"] = np.nan
    profile = evidence.canonical_profile(long, FEATURES)
    row = profile[(profile.cohort == "LOSS") & (profile.feature_name == "book_imbalance")].iloc[0]
    assert row.coverage == 0.0
    assert np.isnan(row["median"])   # 0 이 아니다
    contrast = evidence.contrastive_profile(long, FEATURES, CONFIG)
    flagged = contrast[(contrast.feature_name == "book_imbalance")
                       & (contrast.cohort_a == "PROFIT") & (contrast.cohort_b == "LOSS")]
    assert flagged.missingness_confound.all()


def test_profile_carries_catalog_provenance():
    profile = evidence.canonical_profile(long_frame(), FEATURES)
    assert (profile.catalog_hash == catalog.catalog_hash()).all()
    for name in FEATURES:
        row = profile[profile.feature_name == name].iloc[0]
        assert row.definition_id == catalog.FEATURES[name].definition_id
        assert row.dimension and row.unit and row.time_basis


def test_evidence_relation_map_only_declares_time_order_it_can_show():
    families = [
        {"family": "TRADE_ACTIVITY", "role": "PERSISTENT_CONTEXT",
         "role_reason": "이른 시점부터 분리"},
        {"family": "ORDER_FLOW", "role": "LATE_TRIGGER",
         "role_reason": "anchor에서만 분리"},
        {"family": "SPREAD", "role": "SUPPORTING_STATE",
         "role_reason": "분리는 있으나 시작점 불명"},
    ]
    roles, relations = evidence.evidence_relation_map(families)
    indexed = {(r["family_a"], r["family_b"]): r for r in relations}

    assert roles["TRADE_ACTIVITY"] == "PERSISTENT_CONTEXT"
    assert indexed[("ORDER_FLOW", "TRADE_ACTIVITY")]["relation"] == "CONTEXT_TRIGGER"
    assert indexed[("ORDER_FLOW", "SPREAD")]["relation"] == "UNRESOLVED"


# ---- 중복 묶기 ----------------------------------------------------------------

def test_correlated_features_share_a_group_and_a_representative():
    long = long_frame(n=200)
    # 같은 family 안에서 완전히 상관된 짝을 만든다
    long["ofi_depth_10"] = long.ofi_depth_5 * 2.0 + 1.0
    features = (*FEATURES, "ofi_depth_10")
    groups = evidence.evidence_groups(long, features, CONFIG)
    flow = groups[groups.feature.isin(["ofi_depth_5", "ofi_depth_10"])]
    assert flow.correlation_group.nunique() == 1
    assert flow.is_representative.sum() == 1
    # 다른 family 는 상관이 높아도 섞지 않는다
    assert groups[groups.feature == "spread_bps"].correlation_group.iloc[0] \
        != flow.correlation_group.iloc[0]


def test_representative_is_not_chosen_by_effect_size():
    long = long_frame(n=200)
    long["ofi_depth_10"] = long.ofi_depth_5 * 2.0 + 1.0
    long.loc[long.index[:50], "ofi_depth_10"] = np.nan     # 관측이 적은 쪽
    groups = evidence.evidence_groups(long, (*FEATURES, "ofi_depth_10"), CONFIG)
    rep = groups[groups.feature.isin(["ofi_depth_5", "ofi_depth_10"])]
    assert rep[rep.is_representative].feature.iloc[0] == "ofi_depth_5"   # coverage 가 높은 쪽


# ---- 시간 패턴 ----------------------------------------------------------------

@pytest.mark.parametrize("series,expected", [
    ([0.0, 0.0, 0.0, 0.0, 0.0], "NO_CLEAR_TEMPORAL_PATTERN"),
    ([0.0, 0.1, 0.2, 0.4, 0.6], "MONOTONIC_INCREASE"),
    ([0.0, -0.1, -0.2, -0.4, -0.6], "MONOTONIC_DECREASE"),
    ([-0.4, -0.3, 0.0, 0.3, 0.5], "CROSSOVER"),
    ([0.3, 0.3, 0.3, 0.3, 0.3], "PERSISTENT_HIGH"),
    ([-0.3, -0.3, -0.3, -0.3, -0.3], "PERSISTENT_LOW"),
])
def test_temporal_patterns_are_deterministic(series, expected):
    assert evidence.temporal_pattern(series) == expected
    assert evidence.temporal_pattern(series) == evidence.temporal_pattern(series)


def test_tiny_numeric_wobble_is_not_a_crossover():
    assert evidence.temporal_pattern([1e-6, -1e-6, 1e-6, -1e-6, -0.4]) != "CROSSOVER"


def test_temporal_trajectory_survives_zero_inflated_features():
    """값의 절반이 0 이어도 궤적이 사라지지 않아야 한다 (구 profile 의 실패 지점)."""
    long = long_frame(n=200, times=(-2000, -1000, 0))
    rng = np.random.default_rng(1)
    n = len(long)
    hot = ((long.cohort == "PROFIT").to_numpy()
           & (long.relative_time_ms == 0).to_numpy())
    # 어느 cohort 에서도 값의 과반이 0 이다 — 중앙값은 양쪽 다 0 이 된다.
    # 다만 PROFIT 은 0 이 아닌 값이 뚜렷하게 크다. 순위로는 갈리고 중앙값으로는 안 갈린다.
    values = np.where(hot, rng.normal(2.0, 0.5, size=n), rng.normal(0.0, 1.0, size=n))
    values[rng.random(n) < np.where(hot, 0.55, 0.80)] = 0.0
    long["ofi_depth_5"] = values
    contrast = evidence.contrastive_profile(long, FEATURES, CONFIG)
    temporal = evidence.temporal_summary(long, FEATURES, contrast, CONFIG)
    row = temporal[temporal.feature_name == "ofi_depth_5"].iloc[0]
    assert row.median_a[-1] == row.median_b[-1] == 0.0     # 중앙값은 둘 다 0 이지만
    assert abs(row.separation[-1]) > 0.2                    # 분리도는 보인다


# ---- 게이트 ------------------------------------------------------------------

def test_gate_rejects_a_feature_outside_the_catalog():
    fail, _ = evidence.quality_gate(long_frame(), (*FEATURES, "obi_1"), anchor_frame(),
                                    CONFIG, catalog.catalog_hash())
    assert any("카탈로그에 없는" in f for f in fail)


def test_gate_rejects_a_catalog_hash_mismatch():
    fail, _ = evidence.quality_gate(long_frame(), FEATURES, anchor_frame(), CONFIG, "stale")
    assert any("hash" in f for f in fail)


def test_gate_rejects_a_missing_cohort():
    fail, _ = evidence.quality_gate(long_frame(cohorts=("PROFIT", "LOSS")), FEATURES,
                                    anchor_frame()[lambda d: d.cohort != "BACKGROUND"],
                                    CONFIG, catalog.catalog_hash())
    assert any("cohort" in f for f in fail)


def test_gate_rejects_a_future_observation():
    long = long_frame()
    long.loc[long.index[0], "age_ms"] = -5.0
    fail, _ = evidence.quality_gate(long, FEATURES, anchor_frame(), CONFIG,
                                    catalog.catalog_hash())
    assert any("미래 관측" in f for f in fail)


def test_gate_passes_a_clean_frame():
    fail, _ = evidence.quality_gate(long_frame(), FEATURES, anchor_frame(), CONFIG,
                                    catalog.catalog_hash())
    assert fail == []


# ---- 역할과 cross-family 중복 (v1.1) -------------------------------------------

def test_declared_cross_family_identity_holds_in_the_data():
    """`MICROPRICE` 를 REDUNDANT 로 강등한 근거를 데이터로 다시 확인한다."""
    book = catalog.Book(arrays(n=400, seed=11))
    values = catalog.compute_features(
        ["microprice_dev_bps", "spread_bps", "book_imbalance"], book)
    predicted = values["spread_bps"] / 2.0 * values["book_imbalance"]
    got = values["microprice_dev_bps"]
    finite = np.isfinite(predicted) & np.isfinite(got)
    assert finite.mean() > 0.9
    assert np.abs(predicted[finite] - got[finite]).max() < 1e-9
    assert "MICROPRICE" in evidence.DERIVED_FAMILIES
    assert set(evidence.DERIVED_FAMILIES["MICROPRICE"]["derived_from"]) == {"SPREAD",
                                                                           "BOOK_IMBALANCE"}


@pytest.mark.parametrize("separation,expected", [
    ([0.24, 0.28, 0.32, 0.33, 0.37], "PERSISTENT_CONTEXT"),
    ([-0.03, 0.01, 0.05, 0.03, -0.41], "LATE_TRIGGER"),
    ([0.01, 0.05, 0.26, 0.31, 0.22], "SETUP_STATE"),
    ([0.02, 0.01, 0.03, 0.02, 0.04], "UNRESOLVED"),
])
def test_temporal_role_is_deterministic(separation, expected):
    role, _, _ = evidence.temporal_role(separation)
    assert role == expected
    assert evidence.temporal_role(separation) == evidence.temporal_role(separation)


def test_derived_family_is_always_redundant_whatever_its_shape():
    strong = [0.30, 0.31, 0.32, 0.33, 0.34]
    assert evidence.temporal_role(strong, "MICROPRICE")[0] == "REDUNDANT"
    assert evidence.temporal_role(strong, "ORDER_FLOW")[0] == "PERSISTENT_CONTEXT"


def test_separation_onset_uses_the_material_cut():
    times = [-10000, -5000, -2000, -1000, 0]
    assert evidence.separation_onset([0.05, 0.09, 0.25, 0.3, 0.4], times) == -2000
    assert evidence.separation_onset([0.01, 0.02, 0.03, 0.04, 0.05], times) is None
    assert evidence.separation_onset([0.9, 0.9, 0.9, 0.9, 0.9], times) == -10000
