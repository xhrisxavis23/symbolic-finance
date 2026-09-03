import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import report  # noqa: E402


def test_provenance_records_every_required_field():
    record = report.provenance(symbols=("005930",), seed=0, sr_backend="naive",
                               grid=(0.7, 0.85, 0.95), attempts=9, bottleneck=2)
    for key in ("profile_hash", "catalog_hash", "vendor_manifest_sha256",
                "symbols", "seed", "sr_backend", "quantile_grid", "attempts",
                "bottleneck", "date", "python"):
        assert key in record, key
    assert record["attempts"] == 9


def test_report_flags_naive_backend_loudly(tmp_path):
    ranked = pd.DataFrame(
        {"decisions": [10], "scorable": [8], "fills": [3], "unfilled": [5],
         "censored": [2], "total_net_bps": [1.5], "bps_per_decision": [0.19],
         "positive": [True], "cohort_NET_RECOVERY": [1],
         "cohort_EARLY_RECOVERY_LATE_REVERSAL": [0],
         "cohort_COST_INSUFFICIENT": [1], "cohort_PERSISTENT_ADVERSE": [1]},
        index=pd.Index(["e000:q0.85"], name="contract_id"))
    path = report.write(
        tmp_path, universe={"symbols": ["005930"], "strata": {}},
        candidates=[], compiled=[], failures=[], ranked=ranked,
        prov=report.provenance(symbols=("005930",), seed=0, sr_backend="naive",
                               grid=(0.85,), attempts=1, bottleneck=2))
    text = path.read_text(encoding="utf-8")
    assert "naive" in text
    assert "배관 검증용" in text
    assert json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))


def test_report_omits_naive_warning_for_a_real_sr_backend(tmp_path):
    """뮤테이션 자기검토: `if prov["sr_backend"] == "naive":` 를 `if True:` 로 바꿔도
    `test_report_flags_naive_backend_loudly` 는 여전히 통과한다 — 그 테스트는
    `naive` 를 준 경우만 보고 다른 백엔드를 준 경우를 보지 않기 때문이다.
    이 테스트가 그 빈틈을 메운다: `naive` 가 아닌 백엔드에서는 경고가 없어야 한다."""
    ranked = pd.DataFrame(
        {"decisions": [10], "scorable": [8], "fills": [3], "unfilled": [5],
         "censored": [2], "total_net_bps": [1.5], "bps_per_decision": [0.19],
         "positive": [True], "cohort_NET_RECOVERY": [1],
         "cohort_EARLY_RECOVERY_LATE_REVERSAL": [0],
         "cohort_COST_INSUFFICIENT": [1], "cohort_PERSISTENT_ADVERSE": [1]},
        index=pd.Index(["e000:q0.85"], name="contract_id"))
    path = report.write(
        tmp_path, universe={"symbols": ["005930"], "strata": {}},
        candidates=[], compiled=[], failures=[], ranked=ranked,
        prov=report.provenance(symbols=("005930",), seed=0, sr_backend="pysr",
                               grid=(0.85,), attempts=1, bottleneck=2))
    text = path.read_text(encoding="utf-8")
    assert "배관 검증용" not in text
    assert "⚠️" not in text


def test_compile_report_success_rate_reflects_actual_ratio(tmp_path):
    """뮤테이션 자기검토: `success_rate` 를 상수(예 1.0)로 반환해도
    `test_report_flags_naive_backend_loudly` 는 잡지 못한다 — `candidates`·
    `compiled` 가 둘 다 빈 리스트라 실제 비율을 검사하지 않기 때문이다.
    서로 다른 비율을 만드는 두 번의 `write()` 로 상수 반환을 잡는다."""
    def _write(run_id: str, n_candidates: int, n_compiled: int) -> float:
        candidates = [SimpleNamespace(expr=f"c{i}", complexity=1, in_sample_score=0.1,
                                      backend="naive", seed=0)
                     for i in range(n_candidates)]
        compiled = [SimpleNamespace(
            ast={"op": "primitive", "primitive_id": "book_imbalance"},
            thresholds={"theta_score": "rolling_prior_100_ticks_quantile"},
            shells=(), source=SimpleNamespace(expr=f"c{i}", backend="naive"))
            for i in range(n_compiled)]
        run_dir = tmp_path / run_id
        report.write(run_dir, universe={}, candidates=candidates, compiled=compiled,
                    failures=[], ranked=pd.DataFrame(),
                    prov=report.provenance(symbols=(), seed=0, sr_backend="naive",
                                           grid=(0.85,), attempts=1, bottleneck=2))
        payload = json.loads((run_dir / "compile_report.json").read_text(encoding="utf-8"))
        return payload["success_rate"]

    rate_a = _write("a", n_candidates=4, n_compiled=1)
    rate_b = _write("b", n_candidates=2, n_compiled=2)
    assert rate_a == pytest.approx(0.25)
    assert rate_b == pytest.approx(1.0)
    assert rate_a != rate_b


def _fake_candidates(n: int) -> list:
    return [SimpleNamespace(expr=f"c{i}", complexity=1, in_sample_score=0.1,
                            backend="naive", seed=0) for i in range(n)]


def _fake_compiled(n: int) -> list:
    return [SimpleNamespace(
        ast={"op": "primitive", "primitive_id": "book_imbalance"},
        thresholds={"theta_score": "rolling_prior_100_ticks_quantile"},
        shells=(), source=SimpleNamespace(expr=f"c{i}", backend="naive"))
        for i in range(n)]


def _fake_failures(n: int) -> list:
    return [SimpleNamespace(candidate=SimpleNamespace(expr=f"bad{i}"),
                            stage="translate", reason="지원하지 않는 노드")
            for i in range(n)]


def test_compile_report_reconciles_merged_duplicates(tmp_path):
    """F1: `attempted - succeeded - failed` 가 `merged_duplicates` 로 이름 붙어
    산출물에 남아야 한다 — 정규형이 같아 병합된 후보는 실패로 기록되지 않는다."""
    run_dir = tmp_path / "run"
    path = report.write(
        run_dir, universe={}, candidates=_fake_candidates(5), compiled=_fake_compiled(2),
        failures=_fake_failures(1), ranked=pd.DataFrame(),
        prov=report.provenance(symbols=(), seed=0, sr_backend="naive",
                               grid=(0.85,), attempts=1, bottleneck=2))
    payload = json.loads((run_dir / "compile_report.json").read_text(encoding="utf-8"))
    assert payload["merged_duplicates"] == 5 - 2 - 1 == 2
    text = path.read_text(encoding="utf-8")
    assert "2개 병합" in text
    assert "5 = 2 + 1 + 2" in text


def test_report_flags_negative_merged_duplicates_visibly(tmp_path):
    """F1: `merged_duplicates` 가 음수면 그 자체가 버그 신호이므로 눈에 띄게
    표시해야 한다. (정상 경로에서는 음수가 나올 수 없지만, 산술이 깨졌을 때
    조용히 넘어가지 않는지는 인위적으로 만든 입력으로만 확인할 수 있다.)"""
    run_dir = tmp_path / "run"
    path = report.write(
        run_dir, universe={}, candidates=_fake_candidates(1), compiled=_fake_compiled(1),
        failures=_fake_failures(1), ranked=pd.DataFrame(),
        prov=report.provenance(symbols=(), seed=0, sr_backend="naive",
                               grid=(0.85,), attempts=1, bottleneck=2))
    payload = json.loads((run_dir / "compile_report.json").read_text(encoding="utf-8"))
    assert payload["merged_duplicates"] == -1
    text = path.read_text(encoding="utf-8")
    assert "⚠️" in text
    assert "음수" in text


def _ranked_frame(rows: dict[str, dict]) -> pd.DataFrame:
    """`contract_id -> {열: 값}` 에서 F2 테스트용 `ranked` 프레임을 만든다."""
    base = {"decisions": 100, "scorable": 100, "fills": 10, "unfilled": 90,
           "censored": 0, "total_net_bps": -1.0, "bps_per_decision": -0.01,
           "positive": False, "cohort_NET_RECOVERY": 0,
           "cohort_EARLY_RECOVERY_LATE_REVERSAL": 0,
           "cohort_COST_INSUFFICIENT": 0, "cohort_PERSISTENT_ADVERSE": 0}
    records = []
    for contract_id, overrides in rows.items():
        row = dict(base)
        row.update(overrides)
        row["contract_id"] = contract_id
        records.append(row)
    return pd.DataFrame(records).set_index("contract_id")


def test_duplicate_signal_groups_groups_metric_identical_contracts():
    """F2: 지표(`total_net_bps`·`scorable`·`fills`·`unfilled`·`censored`·코호트 4열)가
    완전히 같은 계약은 한 그룹으로 묶이고, 다른 계약은 별도 그룹이어야 한다."""
    ranked = _ranked_frame({
        "e000:q0.85": {"total_net_bps": -100.0, "fills": 5},
        "e001:q0.85": {"total_net_bps": -100.0, "fills": 5},   # e000 과 지표 동일
        "e002:q0.85": {"total_net_bps": -50.0, "fills": 3},    # 다름
    })
    groups = report.duplicate_signal_groups(ranked)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
    duplicate = next(g for g in groups if len(g) == 2)
    assert duplicate == ["e000:q0.85", "e001:q0.85"]


def test_report_surfaces_independent_contract_count_and_duplicate_groups(tmp_path):
    """F2: `write()` 가 provenance.json 에 실질 독립 계약 수를 별도 항목으로 남기고
    (`attempts` 는 건드리지 않는다), report.md 가 어느 계약이 묶였는지 나열해야 한다."""
    ranked = _ranked_frame({
        "e000:q0.85": {"total_net_bps": -100.0, "fills": 5},
        "e001:q0.85": {"total_net_bps": -100.0, "fills": 5},
        "e002:q0.85": {"total_net_bps": -50.0, "fills": 3},
    })
    run_dir = tmp_path / "run"
    prov = report.provenance(symbols=(), seed=0, sr_backend="naive",
                             grid=(0.85,), attempts=27, bottleneck=2)
    path = report.write(run_dir, universe={}, candidates=[], compiled=[], failures=[],
                        ranked=ranked, prov=prov)
    written_prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert written_prov["attempts"] == 27                       # attempts 는 그대로
    assert written_prov["independent_contract_count"] == 2      # 별도 항목
    assert written_prov["independent_contract_count_of"] == 3
    text = path.read_text(encoding="utf-8")
    assert "e000:q0.85, e001:q0.85" in text


def test_report_states_explicitly_when_no_duplicates_found(tmp_path):
    """F2: 중복이 하나도 없으면 "안 세어본 것"과 구분되게 그 사실을 명시해야 한다."""
    ranked = _ranked_frame({
        "e000:q0.85": {"total_net_bps": -100.0},
        "e001:q0.85": {"total_net_bps": -50.0},
    })
    run_dir = tmp_path / "run"
    path = report.write(
        run_dir, universe={}, candidates=[], compiled=[], failures=[], ranked=ranked,
        prov=report.provenance(symbols=(), seed=0, sr_backend="naive",
                               grid=(0.85,), attempts=2, bottleneck=2))
    text = path.read_text(encoding="utf-8")
    assert "지표가 동일한 계약 쌍 없음" in text


def test_duplicate_signal_groups_treats_missing_metric_columns_as_all_distinct():
    """방어 경로: `DUPLICATE_METRIC_COLUMNS` 가 하나도 없는(구버전) `ranked` 가
    들어오면 비교할 지표가 없으니 모든 계약을 별개 그룹으로 본다 — 안전한 방향
    (과소 병합)으로 fail 한다. 열이 있는데 값을 안 읽는 회귀도 이 테스트가 잡는다:
    조용히 하나로 뭉뚱그리면(과대 병합) 아래 assert 가 깨진다."""
    ranked = pd.DataFrame(
        {"unrelated_column": [1, 2, 3]},
        index=pd.Index(["e000:q0.85", "e001:q0.85", "e002:q0.85"], name="contract_id"))
    groups = report.duplicate_signal_groups(ranked)
    assert len(groups) == 3
    assert sorted(g[0] for g in groups) == ["e000:q0.85", "e001:q0.85", "e002:q0.85"]
