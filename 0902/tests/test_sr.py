import sys
from pathlib import Path

import numpy as np
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.sr.base import complexity_of  # noqa: E402
from sd.sr.naive import NaiveBackend  # noqa: E402


def _problem(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    names = ("book_imbalance", "queue_imbalance_best", "signed_aggr_flow_20")
    y = 3.0 * X[:, 0] - 2.0 * X[:, 2]
    return X, y, np.ones(n), names


def test_candidates_are_marked_with_backend_name():
    X, y, w, names = _problem()
    for candidate in NaiveBackend(seed=0).fit(X, y, w, names):
        assert candidate.backend == "naive"
        assert candidate.seed == 0


def test_expressions_use_only_given_feature_symbols():
    X, y, w, names = _problem()
    allowed = set(names)
    for candidate in NaiveBackend(seed=0).fit(X, y, w, names):
        used = {str(s) for s in candidate.expr.free_symbols}
        assert used <= allowed


def test_recovers_a_linear_signal_with_high_score():
    X, y, w, names = _problem()
    best = max(NaiveBackend(seed=0).fit(X, y, w, names),
               key=lambda c: c.in_sample_score)
    assert best.in_sample_score > 0.95


def test_complexity_is_positive_and_pareto_is_sorted():
    X, y, w, names = _problem()
    candidates = NaiveBackend(seed=0).fit(X, y, w, names)
    assert candidates
    assert all(c.complexity > 0 for c in candidates)
    assert [c.complexity for c in candidates] == sorted(c.complexity for c in candidates)


def test_same_seed_is_reproducible():
    X, y, w, names = _problem()
    first = [sympy.srepr(c.expr) for c in NaiveBackend(seed=1).fit(X, y, w, names)]
    second = [sympy.srepr(c.expr) for c in NaiveBackend(seed=1).fit(X, y, w, names)]
    assert first == second


# -- 뮤테이션 자기검토로 드러난 빈틈을 메우는 보강 테스트 -----------------------
#
# 위 5개는 브리핑이 준 그대로다. 커밋 전 뮤테이션 자기검토에서 아래 네 지점이
# 실제로 `NaiveBackend(seed=0)` / `w = np.ones(n)` 균일 가중치만 쓰는 `_problem()`
# 으로는 잡히지 않는다는 것을 확인했다 (각 뮤테이션을 넣고 FAILED 를 본 뒤
# 원복·재확인함). `test_manifold.py` 가 이미 같은 방식으로 보강한 전례를 따른다.


def test_seed_field_reflects_the_callers_seed_not_a_hardcoded_default():
    """`Candidate(..., seed=0)` 처럼 seed 필드를 상수로 박아도 주어진 5개는 전부
    통과한다 — `test_candidates_are_marked_with_backend_name` 은 seed=0 으로만
    부르고, `test_same_seed_is_reproducible` 은 seed=1 을 쓰지만 `c.seed` 를
    검사하지 않는다. `NaiveBackend` 는 결정론적 격자 탐색이라 서로 다른 seed 가
    수식 자체를 바꾸지는 않는다(이것은 버그가 아니라 이 백엔드에 확률적 요소가
    없다는 사실의 반영이다) — 그래서 "다른 시드 → 다른 결과" 짝 검사를 그대로
    옮기는 대신, 여기서는 "다른 시드 → `Candidate.seed` 필드가 그 시드를
    정확히 반영하는지"를 확인한다."""
    X, y, w, names = _problem()
    for seed in (3, 17):
        candidates = NaiveBackend(seed=seed).fit(X, y, w, names)
        assert candidates
        assert {c.seed for c in candidates} == {seed}

    exprs_a = [sympy.srepr(c.expr) for c in NaiveBackend(seed=3).fit(X, y, w, names)]
    exprs_b = [sympy.srepr(c.expr) for c in NaiveBackend(seed=17).fit(X, y, w, names)]
    assert exprs_a == exprs_b  # 격자 탐색 자체는 seed 에 의존하지 않는다 — 의도된 동작


def test_weighted_r2_actually_uses_the_weights():
    """5개 테스트가 전부 `w = np.ones(n)` 균일 가중치만 쓰기 때문에, `weighted_r2`
    가 `w` 를 완전히 무시하고 (`np.mean` 으로 퇴화해도) 5개 전부 통과한다(뮤테이션
    자기검토로 확인). 잔차가 큰 구간에 가중치를 몰아주면 R² 가 뚜렷이 떨어져야
    한다는 것을 직접 확인한다."""
    from sd.sr.base import weighted_r2

    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(size=n)
    prediction = y.copy()
    prediction[n // 2:] += 5.0                       # 뒤쪽 절반에 큰 잔차를 심는다
    uniform = np.ones(n)
    skewed = np.where(np.arange(n) < n // 2, 1.0, 100.0)  # 잔차가 큰 쪽에 가중치 집중

    r2_uniform = weighted_r2(y, prediction, uniform)
    r2_skewed = weighted_r2(y, prediction, skewed)
    assert r2_skewed < r2_uniform - 1.0


def test_score_uses_sample_weights_in_the_least_squares_fit():
    """`weighted_r2` 단위 테스트를 넘어, `NaiveBackend._score` 의 `lstsq` 호출까지
    가중치가 실제로 타는지 확인한다. 앞쪽 절반은 `y = 5·x`, 뒤쪽 절반은
    `y = -5·x` 로 반대 부호 기울기를 심어 균일 가중치에서는 상쇄되어 거의
    0에 가까운 점수가 나오게 하고, 앞쪽에 가중치를 100배 몰아주면 그 구간의
    기울기를 회복해 점수가 뚜렷이 올라가야 한다."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    names = ("book_imbalance", "queue_imbalance_best", "signed_aggr_flow_20")
    y = np.where(np.arange(n) < n // 2, 5.0 * X[:, 0], -5.0 * X[:, 0])

    backend = NaiveBackend(seed=0)
    expr = sympy.Symbol("book_imbalance")
    uniform = backend._score(expr, X, y, np.ones(n), names)
    concentrated = backend._score(
        expr, X, y, np.where(np.arange(n) < n // 2, 100.0, 1.0), names)

    assert uniform is not None and concentrated is not None
    assert uniform.in_sample_score < 0.1
    assert concentrated.in_sample_score > 0.9


def test_complexity_of_counts_actual_ast_nodes_not_a_constant():
    """`complexity_of` 를 상수(예: 항상 1)로 바꿔도 `test_complexity_is_positive_and_pareto_is_sorted`
    는 전부 같은 값이라 "양수"·"정렬됨" 둘 다 트리비얼하게 통과한다(뮤테이션
    자기검토로 확인). 서로 다른 크기의 수식이 실제로 다른 노드 수를 내는지
    리터럴 값으로 고정해 확인한다."""
    a, b = sympy.symbols("a b")
    assert complexity_of(a) == 1
    assert complexity_of(a + b) == 3
    assert complexity_of(a * b + a) == 5
    assert complexity_of(a) < complexity_of(a + b) < complexity_of(a * b + a)


def test_rank_features_uses_weighted_correlation_not_plain_corrcoef():
    """F1 (Task 9 리뷰): `_rank_features` 가 `w` 를 받고도 `np.corrcoef`(비가중)만
    써서 계획서 §3 S2 ③("가중치를 SR 손실에 명시적으로 넘긴다")이 feature 선택
    단계에서 무너지던 결함을 고친 뒤의 판별 테스트다. 균일 가중치 테스트만으로는
    가중 상관과 비가중 상관이 우연히 같은 순서를 내므로 이 결함을 잡을 수 없다
    — 그래서 순서가 실제로 뒤집히는 상황을 만든다.

    `tail_only_feature`: 표본 대부분(95%)에서 `y`와 무관하고, 꼬리 5%에서만
    강하게(계수 3) 정렬된다. `weak_broad_feature`: 전체에서 약하지만(계수
    0.15) 고르게 상관된다. 비가중 상관에서는 다수인 무관 구간이 신호를
    희석해 `weak_broad_feature`가 이긴다. 꼬리에 `manifold.select`의 실제
    상한([1, 5])과 맞춘 가중치 5.0 을 주면, 가중 상관에서는 `tail_only_feature`
    가 이겨야 한다 — on-manifold 표집이 꼬리에 준 가중치가 feature 선택
    단계에서도 실제로 반영된다는 뜻이다."""
    rng = np.random.default_rng(0)
    n = 2000
    tail = n - 100  # 마지막 100행(5%)이 꼬리

    y = rng.normal(size=n)

    tail_only = rng.normal(scale=1.0, size=n)
    tail_only[tail:] = y[tail:] * 3.0 + rng.normal(scale=0.05, size=n - tail)

    weak_broad = 0.15 * y + rng.normal(scale=1.0, size=n)

    names = ("tail_only_feature", "weak_broad_feature")
    X = np.column_stack([tail_only, weak_broad])

    w_uniform = np.ones(n)
    w_tail = np.ones(n)
    w_tail[tail:] = 5.0  # manifold.select 의 실제 가중치 상한

    backend = NaiveBackend(seed=0, top_features=4)
    unweighted_ranked = backend._rank_features(X, y, w_uniform, names)
    weighted_ranked = backend._rank_features(X, y, w_tail, names)

    assert unweighted_ranked[0] == "weak_broad_feature"
    assert weighted_ranked[0] == "tail_only_feature"
