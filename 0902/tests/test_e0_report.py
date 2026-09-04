"""`sd.e0.report` 의 게이트 판정 로직. 계획서 §4 게이트 규칙 세 줄을 그대로
숫자로 검사한다 — "5개 중 3개 이상 → 진행", "2개 이하 → 중단",
"L1·L2 동시 실패 → 경고(hold_veto)"."""

import json
import sys
from pathlib import Path

import numpy as np
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import report as e0_report  # noqa: E402
from sd.e0 import runner  # noqa: E402

X1 = sympy.Symbol("x")


def _track(recovered: bool) -> runner.TrackResult:
    candidate = runner.Candidate(expr=X1, complexity=1, in_sample_score=0.5,
                                 backend="pysr", seed=0)
    judged = {"passed": recovered, "ambiguous": False}
    return runner.TrackResult(name="main", n_rows=10, candidates=[candidate],
                              diagnostics={}, judged=[judged], primary_index=0,
                              fit_seconds=1.0)


def _law_result(law: str, recovered: bool) -> runner.LawResult:
    return runner.LawResult(law=law, n_rows_total=10, symbols_used=("000020",),
                            symbols_skipped={}, teacher_dimless_diag={},
                            teacher_raw_diag={}, tracks={"main": _track(recovered)})


def _write_and_load_gate(tmp_path, results):
    run_dir = tmp_path / "run"
    e0_report.write(run_dir, results, symbols=["000020"], args={})
    return json.loads((run_dir / "gate.json").read_text())


def test_three_of_five_recovered_proceeds(tmp_path):
    results = {law: _law_result(law, recovered) for law, recovered in
              [("L1", True), ("L2", True), ("L3", True), ("L4", False), ("L5", False)]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 3
    assert gate["verdict"] == "proceed"
    assert gate["veto_triggered"] is False


def test_two_of_five_recovered_stops_even_without_veto(tmp_path):
    """카운트 규칙이 거부권보다 먼저다 — L1·L2 가 둘 다 살아 있어도 전체
    복원이 2개뿐이면 중단이다."""
    results = {law: _law_result(law, recovered) for law, recovered in
              [("L1", True), ("L2", True), ("L3", False), ("L4", False), ("L5", False)]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 2
    assert gate["verdict"] == "stop"
    assert gate["veto_triggered"] is False


def test_l1_and_l2_both_failing_holds_veto_even_with_three_recovered(tmp_path):
    """3개 이상 복원돼도 L1·L2 가 동시에 실패하면 '진행'이 아니라 '경고
    (hold_veto)' 다 — 계획서: "원인 규명 전 E1 보류"."""
    results = {law: _law_result(law, recovered) for law, recovered in
              [("L1", False), ("L2", False), ("L3", True), ("L4", True), ("L5", True)]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 3
    assert gate["veto_triggered"] is True
    assert gate["verdict"] == "hold_veto"
    assert set(gate["veto_laws_failed"]) == {"L1", "L2"}


def test_only_l1_failing_does_not_trigger_veto(tmp_path):
    """거부권은 L1·L2 가 **함께** 실패할 때만 발동한다 — 하나만 실패하면
    발동하지 않는다."""
    results = {law: _law_result(law, recovered) for law, recovered in
              [("L1", False), ("L2", True), ("L3", True), ("L4", True), ("L5", False)]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 3
    assert gate["veto_triggered"] is False
    assert gate["verdict"] == "proceed"


def test_zero_recovered_stops(tmp_path):
    results = {law: _law_result(law, False) for law in ["L1", "L2", "L3", "L4", "L5"]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 0
    assert gate["verdict"] == "stop"


def test_all_five_recovered_proceeds(tmp_path):
    results = {law: _law_result(law, True) for law in ["L1", "L2", "L3", "L4", "L5"]}
    gate = _write_and_load_gate(tmp_path, results)
    assert gate["n_recovered"] == 5
    assert gate["verdict"] == "proceed"
    assert gate["veto_triggered"] is False
