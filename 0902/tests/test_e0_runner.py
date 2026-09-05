"""`sd.e0.runner` 의 PySR 을 타지 않는 부분 — `_judge` 디스패치와
`TrackResult`/`LawResult` 의 파생 프로퍼티(`recovered`·`primary_verdict`·
`any_candidate_recovered`), 그리고 v2 표본외 후보 선택(PREREG-E0-V2.md §1-1)의
핵심 로직(`_score_and_judge`·`_fit_track`·`run_law` 의 배선). 실제
`PySRBackend.fit()` 을 부르는 통합 확인은 `run_e0.py` 실행(느림)으로만
검증한다 — 여기서는 순수 로직 + 가짜(fake) 백엔드/교사/조립기로 배선만
본다."""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import runner  # noqa: E402
from sd.e0 import targets  # noqa: E402
from sd.sr.base import Candidate, score_candidate  # noqa: E402

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
    out_of_sample_scores = [0.2, 0.9, 0.5]
    track = runner.TrackResult(name="main", n_rows=200, candidates=candidates,
                               diagnostics={}, judged=judged, primary_index=None,
                               fit_seconds=1.0, out_of_sample_scores=out_of_sample_scores)
    track.primary_index = int(np.argmax(out_of_sample_scores))
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
                               fit_seconds=1.0, out_of_sample_scores=[0.95, 0.3])
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
    점수가 평균 예측보다도 나쁠(음수) 수 있다 — 그런 후보만 있을 때
    `any_candidate_recovered` 가 True 로 나오면 안 된다. v2 에서는 이 점수가
    선택 종목에서 계산한 `out_of_sample_score` 다."""
    degenerate = _candidate(X1, 1, -850.0)     # 구조는 통과하지만 적합이 끔찍하다
    saturating = _candidate(sympy.tanh(X1), 4, 0.98)  # 적합은 좋지만 구조 판정 실패
    judged = [{"passed": True, "ambiguous": False},     # degenerate 의 판정 (구조만 봄)
             {"passed": False, "ambiguous": False}]      # saturating 의 판정
    track = runner.TrackResult(name="main", n_rows=100, candidates=[degenerate, saturating],
                               diagnostics={}, judged=judged, primary_index=1, fit_seconds=1.0,
                               out_of_sample_scores=[-850.0, 0.98])
    assert track.any_candidate_recovered is False
    assert track.fraction_of_front_recovered == 0.0
    assert track.recovered_indices == []


def test_any_candidate_recovered_counts_a_well_fit_structurally_passing_candidate():
    good = _candidate(0.5 * X1, 3, 0.7)
    judged = [{"passed": True, "ambiguous": False}]
    track = runner.TrackResult(name="main", n_rows=100, candidates=[good], diagnostics={},
                               judged=judged, primary_index=0, fit_seconds=1.0,
                               out_of_sample_scores=[0.7])
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
                               judged=judged, primary_index=0, fit_seconds=1.0,
                               out_of_sample_scores=[-5.0])
    assert track.recovered is False


def test_ambiguous_primary_verdict_does_not_count_as_recovered():
    """`passed=True` 인데 `ambiguous=True` 면 복원으로 세지 않는다 — 애매함을
    관대함으로 바꾸지 않는다는 규칙이 `TrackResult` 수준에서도 지켜지는지."""
    candidates = [_candidate(X1, 1, 0.9)]
    judged = [{"passed": True, "ambiguous": True}]
    track = runner.TrackResult(name="main", n_rows=10, candidates=candidates,
                               diagnostics={}, judged=judged, primary_index=0,
                               fit_seconds=1.0, out_of_sample_scores=[0.9])
    assert track.recovered is False


def test_law_result_recovered_reflects_main_track_only():
    main_candidates = [_candidate(X1, 1, 0.9)]
    main_judged = [{"passed": True, "ambiguous": False}]
    main_track = runner.TrackResult(name="main", n_rows=10, candidates=main_candidates,
                                    diagnostics={}, judged=main_judged, primary_index=0,
                                    fit_seconds=1.0, out_of_sample_scores=[0.9])
    other_candidates = [_candidate(X1, 1, 0.1)]
    other_judged = [{"passed": False, "ambiguous": False}]
    other_track = runner.TrackResult(name="no_distillation", n_rows=10,
                                     candidates=other_candidates, diagnostics={},
                                     judged=other_judged, primary_index=0, fit_seconds=1.0,
                                     out_of_sample_scores=[0.1])
    result = runner.LawResult(law="L2", n_rows_total=10, symbols_used=("000020",),
                              symbols_skipped={}, teacher_dimless_diag={}, teacher_raw_diag={},
                              tracks={"main": main_track, "no_distillation": other_track})
    assert result.recovered is True


def test_law_result_recovered_is_false_when_main_track_missing():
    result = runner.LawResult(law="L2", n_rows_total=10, symbols_used=(), symbols_skipped={},
                              teacher_dimless_diag={}, teacher_raw_diag={}, tracks={})
    assert result.recovered is False


# ---------------------------------------------------------------------------
# v2 표본외 후보 선택 (PREREG-E0-V2.md §1-1) — `score_candidate`·
# `_score_and_judge`·`_fit_track`·`run_law` 배선.
# ---------------------------------------------------------------------------

def test_score_candidate_evaluates_expr_on_given_X_ignoring_in_sample_score():
    """`score_candidate` 는 `candidate.in_sample_score`(적합 데이터에서 이미
    굳어진 값)를 전혀 안 본다 — `expr` 을 주어진 `(X, y, w)` 에서 다시
    평가한다."""
    x = sympy.Symbol("x")
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 4.0 * X[:, 0]
    w = np.ones(len(y))
    candidate = _candidate(4 * x, 2, score=-999.0)   # in_sample_score 라벨은 완전히 틀렸다
    assert score_candidate(candidate, ("x",), X, y, w) == pytest.approx(1.0, abs=1e-9)


def test_score_candidate_returns_negative_infinity_on_evaluation_failure():
    """`expr` 이 주어진 `feature_names` 밖의 심볼을 쓰면(또는 평가가 죽으면)
    예외를 올리지 않고 -inf 를 돌려준다 — 표본외 평가가 실패하는 것도
    "일반화하지 않는다"는 판정의 일부다."""
    bogus = sympy.Symbol("not_a_feature")
    candidate = _candidate(bogus, 1, score=0.5)
    X = np.zeros((20, 1))
    y = np.ones(20)
    w = np.ones(20)
    assert score_candidate(candidate, ("x",), X, y, w) == float("-inf")


def test_primary_index_is_chosen_by_out_of_sample_score_not_in_sample_score():
    """뮤테이션 자기검토 대상: `primary_index` 를 표본 내 점수(argmax)로
    되돌리면 이 테스트가 실패해야 한다. `in_sample_score` 라벨을 실제와
    반대로 매겨도(거짓말) 결과가 바뀌면 안 된다 — `_score_and_judge` 가
    `candidate.expr` 을 `X_select` 에서 다시 평가해 고르기 때문이다."""
    x = sympy.Symbol("x")
    X_select = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
    y_select = 3.0 * X_select[:, 0]                 # 참값: y = 3x
    w_select = np.ones(len(y_select))

    good = _candidate(3 * x, 2, score=0.01)          # 참값과 일치 — 라벨은 "나쁘다"고 우긴다
    bad = _candidate(3 * x + 100, 3, score=0.99)     # 오프셋으로 어긋난다 — 라벨은 "좋다"고 우긴다

    track = runner._score_and_judge(
        "L2", "main", [good, bad], ("x",), X_select, y_select, w_select,
        fit_seconds=1.0, n_fit_rows=999, diagnostics={})

    assert track.primary_index == 0
    assert track.out_of_sample_scores[0] > track.out_of_sample_scores[1]


def test_fit_track_judges_using_select_data_not_fit_data(monkeypatch):
    """뮤테이션 자기검토 대상: `_judge` 를 적합 집합 X 로 되돌리면 이 테스트가
    실패해야 한다.

    `Abs(x)` 는 전부 양수인 구간에서는 완벽한 아핀(기울기 1, 꺾임이 안
    보인다)이지만 0을 가로지르는 구간에서는 꺾인다. 적합 데이터는 순수
    양수 구간(0.5~1.5), 선택 데이터는 대칭 구간(-1~1)으로 둬서 두 구간의
    판정이 실제로 갈리게 만든다 — 판정이 선택 구간 기준(불합격)으로 나오는지
    확인한다."""
    x = sympy.Symbol("x")
    kink_candidate = _candidate(sympy.Abs(x), 2, score=0.5)

    class _FakeBackend:
        diagnostics: dict = {}

        def fit(self, X, y, w, names):
            return [kink_candidate]

    monkeypatch.setattr(runner, "_build_backend",
                        lambda law, seed, niterations, maxsize: _FakeBackend())

    X_fit = np.linspace(0.5, 1.5, 200).reshape(-1, 1)          # 순수 양수 — Abs(x)=x, 꺾임이 안 보인다
    y_fit = X_fit[:, 0].copy()
    w_fit = np.ones(len(y_fit))

    X_select = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)       # 0을 가로지른다 — 꺾임이 보인다
    y_select = np.abs(X_select[:, 0])
    w_select = np.ones(len(y_select))

    track = runner._fit_track("L2", "main", X_fit, y_fit, w_fit, ("x",),
                              X_select, y_select, w_select, seed=0, niterations=1, maxsize=5)

    assert track.judged[0]["passed"] is False


class _FakeTeacher:
    """`sd.e0.teacher.ScalarTeacherProtocol` 계약만 지키는 최소 스텁 — 진짜
    torch 학습을 피해서 배선 테스트를 빠르고 결정론적으로 만든다."""

    def __init__(self, n_features: int, bottleneck: int, seed: int = 0) -> None:
        self.n_features = n_features

    def fit(self, X, y, weight, epochs: int = 300) -> "_FakeTeacher":
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] * 2.0

    def z(self, X):
        return np.asarray(X)


def _fake_dataset(law: str, symbols, date: str, *, offset: float) -> targets.Dataset:
    """`targets.assemble` 스텁. 값의 범위가 종목 집합마다 다르게 표시돼 있어서
    ("적합"은 0 근처, "선택"은 1000 근처) 두 데이터가 섞이면 바로 드러난다."""
    n = max(len(symbols), 1) * 50
    rng = np.random.default_rng(0)
    x = offset + rng.uniform(0.0, 1.0, size=n)
    y = 2.0 * x
    return targets.Dataset(
        law=law, names_dimless=("x",), X_dimless=x.reshape(-1, 1), y_dimless=y,
        names_raw=("x",), X_raw=x.reshape(-1, 1), y_raw=y,
        mask=np.ones(n, dtype=bool), y_description="synthetic",
        symbols_used=tuple(symbols), symbols_skipped={}, n_rows_total=n)


def _patch_run_law_fakes(monkeypatch, fit_symbols, seen_X: list[np.ndarray]):
    """`run_law` 의 무거운 의존성(틱 로딩·manifold·PySR·교사 학습)을 전부
    가짜로 바꿔치기한다 — 배선 테스트를 빠르고 결정론적으로 만든다. 실제
    PySR/Julia 를 절대 안 타므로 뮤테이션 자기검토(§ 아래 두 테스트)에도
    안전하게 쓴다: 가드가 사라져도 무거운 계산으로 새지 않고 곧바로
    관측 가능한 결과(예외가 안 남·값 범위가 섞임)로 드러난다."""
    def fake_assemble(law, symbols, date):
        offset = 0.0 if set(symbols) == set(fit_symbols) else 1000.0
        return _fake_dataset(law, symbols, date, offset=offset)

    def fake_manifold_select(X, mask, max_samples, seed):
        return runner.manifold.Selection(index=np.arange(len(X)), weight=np.ones(len(X)),
                                         on_manifold=np.ones(len(X), dtype=bool))

    class _RecordingBackend:
        diagnostics: dict = {}

        def fit(self, X, y, w, names):
            seen_X.append(np.asarray(X))
            return [Candidate(expr=sympy.Symbol(names[0]), complexity=1,
                              in_sample_score=0.0, backend="fake", seed=0)]

    monkeypatch.setattr(runner.targets, "assemble", fake_assemble)
    monkeypatch.setattr(runner.manifold, "select", fake_manifold_select)
    monkeypatch.setattr(runner, "_build_backend",
                        lambda law, seed, niterations, maxsize: _RecordingBackend())


def test_run_law_fits_sr_only_on_fit_symbols_never_select_symbols(monkeypatch):
    """뮤테이션 자기검토 대상: 적합 종목과 선택 종목을 같게(또는 섞이게) 만들면
    이 테스트가 실패해야 한다. `_build_backend` 가 낸 (가짜) 백엔드의
    `.fit()` 이 실제로 받는 X 가 전부 적합 종목 쪽 값 범위(<10)뿐이고 선택
    종목 쪽 범위(>=1000)는 한 번도 안 섞이는지 배선 수준에서 확인한다
    (PREREG-E0-V2.md §1-1 "SR 적합은 적합 종목으로만 한다")."""
    seen_X: list[np.ndarray] = []
    fit_symbols = ("F1", "F2")
    select_symbols = ("S1",)
    _patch_run_law_fakes(monkeypatch, fit_symbols, seen_X)

    result = runner.run_law("L2", fit_symbols, select_symbols, "20260316", seed=0, bottleneck=1,
                            epochs=1, sr_niterations=1, sr_maxsize=5, max_manifold_samples=1000,
                            teacher_cls=_FakeTeacher)

    assert seen_X, "가짜 백엔드 fit() 이 한 번도 안 불렸다 — 테스트가 아무것도 확인 못 했다"
    for X in seen_X:
        assert np.all(X < 10.0), "선택 종목 값 범위(>=1000)가 SR 적합 입력에 섞였다"
    assert result.tracks["main"].n_select_rows > 0


def test_run_law_rejects_overlapping_fit_and_select_symbols(monkeypatch):
    """뮤테이션 자기검토 대상: 적합/선택 집합을 겹치게(=표본 외가 아니게)
    만들면 `run_law` 이 명시적으로 거부해야 한다 — 조용히 섞인 채로 돌아가면
    안 된다. 무거운 의존성을 전부 가짜로 바꿔서, 가드가 사라져도(뮤테이션)
    실제 PySR/틱 로딩으로 새지 않고 이 테스트 자체가 빠르게 끝난다 —
    가드가 사라지면 예외 없이 정상 종료하므로 `pytest.raises` 가 바로
    실패로 잡는다."""
    seen_X: list[np.ndarray] = []
    fit_symbols = ("F1", "F2")
    overlapping_select_symbols = ("F2", "S1")     # F2 가 양쪽에 다 있다
    _patch_run_law_fakes(monkeypatch, fit_symbols, seen_X)

    with pytest.raises(ValueError):
        runner.run_law("L2", fit_symbols, overlapping_select_symbols, "20260316", seed=0,
                       bottleneck=1, epochs=1, sr_niterations=1, sr_maxsize=5,
                       max_manifold_samples=1000, teacher_cls=_FakeTeacher)
