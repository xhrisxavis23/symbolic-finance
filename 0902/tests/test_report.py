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
