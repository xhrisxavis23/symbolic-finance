"""`sd.e0.runner` 의 PySR 을 타지 않는 부분 — `_judge` 디스패치와
`TrackResult`/`LawResult` 의 파생 프로퍼티(`recovered`·`primary_verdict`·
`any_candidate_recovered`). 실제 `PySRBackend.fit()` 을 부르는 통합 확인은
`run_e0.py` 실행(느림)으로만 검증한다 — 여기서는 순수 로직만 본다."""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import runner  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402

X1 = sympy.Symbol("ofi_20t_over_qbar")


def _candidate(expr, complexity, score, seed=0) -> Candidate:
    return Candidate(expr=expr, complexity=complexity, in_sample_score=score,
                     backend="pysr", seed=seed)


def test_judge_dispatches_l2_to_approximately_affine_logic():
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    verdict = runner._judge("L2", 0.5 * X1, ("ofi_20t_over_qbar",), X)
    assert verdict["passed"] is True
    assert verdict["slope_mean"] == pytest.approx(0.5, abs=1e-6)


def test_judge_raises_for_unknown_law():
    X = np.zeros((10, 1))
    try:
        runner._judge("L99", X1, ("ofi_20t_over_qbar",), X)
    except ValueError:
        pass
    else:
        raise AssertionError("모르는 법칙인데 예외가 안 났다")


def test_track_result_primary_is_the_highest_scoring_candidate():
    candidates = [_candidate(X1, 1, 0.2), _candidate(2 * X1, 3, 0.9), _candidate(X1**2, 5, 0.5)]
    judged = [{"passed": False, "ambiguous": False},
             {"passed": True, "ambiguous": False},
             {"passed": True, "ambiguous": False}]
    track = runner.TrackResult(name="main", n_rows=200, candidates=candidates,
                               diagnostics={}, judged=judged, primary_index=None,
                               fit_seconds=1.0)
    track.primary_index = int(np.argmax([c.in_sample_score for c in candidates]))
    assert track.primary_index == 1
    assert track.recovered is True
    assert track.any_candidate_recovered is True


def test_track_result_recovered_is_false_when_primary_verdict_fails():
    """가장 점수 높은 후보가 판정에서 떨어지면, 다른 후보가 통과해도(있다면)
    `recovered` 는 여전히 False 다 — `any_candidate_recovered` 만 True 다.
    (기준 후보 선택이 실제로 게이트에 반영되는지 확인)"""
    candidates = [_candidate(2 * X1, 3, 0.95), _candidate(X1**2, 5, 0.3)]
    judged = [{"passed": False, "ambiguous": False},   # 최고 점수인데 판정 실패
             {"passed": True, "ambiguous": False}]      # 낮은 점수인데 판정 통과
    track = runner.TrackResult(name="main", n_rows=200, candidates=candidates,
                               diagnostics={}, judged=judged, primary_index=0,
                               fit_seconds=1.0)
    assert track.recovered is False
    assert track.any_candidate_recovered is True


def test_track_result_with_no_candidates_is_not_recovered_and_does_not_crash():
    track = runner.TrackResult(name="main", n_rows=0, candidates=[], diagnostics={},
                               judged=[], primary_index=None, fit_seconds=0.0)
    assert track.primary_verdict is None
    assert track.recovered is False
    assert track.any_candidate_recovered is False


def test_any_candidate_recovered_ignores_structurally_passing_but_badly_fit_candidates():
    """실측 발견(2026-09-04 첫 전체 E0 실행, L2): 원본 변수를 그대로 돌려주는
    항등함수(`f(x)=x`)는 구조 판정(근사 아핀)을 언제나 통과하지만, 실제
    `in_sample_score` 가 평균 예측보다도 나쁠(음수) 수 있다 — 그런 후보만
    있을 때 `any_candidate_recovered` 가 True 로 나오면 안 된다."""
    degenerate = _candidate(X1, 1, -850.0)     # 구조는 통과하지만 적합이 끔찍하다
    saturating = _candidate(sympy.tanh(X1), 4, 0.98)  # 적합은 좋지만 구조 판정 실패
    judged = [{"passed": True, "ambiguous": False},     # degenerate 의 판정 (구조만 봄)
             {"passed": False, "ambiguous": False}]      # saturating 의 판정
    track = runner.TrackResult(name="main", n_rows=100, candidates=[degenerate, saturating],
                               diagnostics={}, judged=judged, primary_index=1, fit_seconds=1.0)
    assert track.any_candidate_recovered is False
    assert track.fraction_of_front_recovered == 0.0
    assert track.recovered_indices == []


def test_any_candidate_recovered_counts_a_well_fit_structurally_passing_candidate():
    good = _candidate(0.5 * X1, 3, 0.7)
    judged = [{"passed": True, "ambiguous": False}]
    track = runner.TrackResult(name="main", n_rows=100, candidates=[good], diagnostics={},
                               judged=judged, primary_index=0, fit_seconds=1.0)
    assert track.any_candidate_recovered is True
    assert track.fraction_of_front_recovered == 1.0
    assert track.recovered_indices == [0]


def test_recovered_property_also_requires_well_fit_primary():
    """`recovered`(primary 기준)도 같은 문턱을 받아야 한다 — 구조만 보고
    적합도를 무시하면 최고 점수 후보가 우연히 degenerate 일 때 잘못된
    '복원'을 낼 수 있다."""
    degenerate = _candidate(X1, 1, -5.0)
    judged = [{"passed": True, "ambiguous": False}]
    track = runner.TrackResult(name="main", n_rows=100, candidates=[degenerate], diagnostics={},
                               judged=judged, primary_index=0, fit_seconds=1.0)
    assert track.recovered is False


def test_ambiguous_primary_verdict_does_not_count_as_recovered():
    """`passed=True` 인데 `ambiguous=True` 면 복원으로 세지 않는다 — 애매함을
    관대함으로 바꾸지 않는다는 규칙이 `TrackResult` 수준에서도 지켜지는지."""
    candidates = [_candidate(X1, 1, 0.9)]
    judged = [{"passed": True, "ambiguous": True}]
    track = runner.TrackResult(name="main", n_rows=10, candidates=candidates,
                               diagnostics={}, judged=judged, primary_index=0,
                               fit_seconds=1.0)
    assert track.recovered is False


def test_law_result_recovered_reflects_main_track_only():
    main_candidates = [_candidate(X1, 1, 0.9)]
    main_judged = [{"passed": True, "ambiguous": False}]
    main_track = runner.TrackResult(name="main", n_rows=10, candidates=main_candidates,
                                    diagnostics={}, judged=main_judged, primary_index=0,
                                    fit_seconds=1.0)
    other_candidates = [_candidate(X1, 1, 0.1)]
    other_judged = [{"passed": False, "ambiguous": False}]
    other_track = runner.TrackResult(name="no_distillation", n_rows=10,
                                     candidates=other_candidates, diagnostics={},
                                     judged=other_judged, primary_index=0, fit_seconds=1.0)
    result = runner.LawResult(law="L2", n_rows_total=10, symbols_used=("000020",),
                              symbols_skipped={}, teacher_dimless_diag={}, teacher_raw_diag={},
                              tracks={"main": main_track, "no_distillation": other_track})
    assert result.recovered is True


def test_law_result_recovered_is_false_when_main_track_missing():
    result = runner.LawResult(law="L2", n_rows_total=10, symbols_used=(), symbols_skipped={},
                              teacher_dimless_diag={}, teacher_raw_diag={}, tracks={})
    assert result.recovered is False
