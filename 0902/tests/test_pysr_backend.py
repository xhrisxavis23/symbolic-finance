"""PySRBackend 테스트.

`pysr`(juliacall) 자체를 부르는 테스트는 `@pytest.mark.slow` 로만 표시한다 —
Julia JIT 워밍업이 첫 `fit()` 마다 수십~백여 초 걸린다(ROADMAP.md 2단계 실측).
나머지는 전부 순수 로직(`_extract_candidate`, `verify_operator_roundtrip`,
`PySRBackend.__init__`, `sd.config.ensure_pysr_env`)만 건드려 `pysr` 없이도
빠르게 돈다.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sd import config  # noqa: E402
from sd.sr.base import Candidate, complexity_of  # noqa: E402
from sd.sr.pysr_backend import (  # noqa: E402
    BINARY_OPERATORS, EXCLUDED_OPERATORS, MAXSIZE_CEILING, UNARY_OPERATORS,
    PySRBackend, _extract_candidate, verify_operator_roundtrip,
)

bi = sympy.Symbol("book_imbalance")
qi = sympy.Symbol("queue_imbalance_best")
NAMES = ("book_imbalance", "queue_imbalance_best")


# ---------------------------------------------------------------------------
# 연산자 집합 — `to_catalog.translate` 왕복 실측.
# ---------------------------------------------------------------------------

def test_declared_operator_sets_actually_compile():
    """`BINARY_OPERATORS`·`UNARY_OPERATORS` 로 선언한 연산자는 전부 `to_catalog`
    를 통과해야 한다 — 통과하지 않는 연산자를 탐색에 준다는 것은 탐색 예산
    낭비다(브리핑의 핵심 요구)."""
    report = verify_operator_roundtrip()
    for op in BINARY_OPERATORS:
        assert report[op] is True, f"{op} 가 선언과 달리 컴파일되지 않는다"
    for op in UNARY_OPERATORS:
        assert report[op] is True, f"{op} 가 선언과 달리 컴파일되지 않는다"


def test_excluded_operators_actually_fail_to_compile():
    """`EXCLUDED_OPERATORS` 에 적은 연산자(`/`·`log`·`sign`)는 실제로 거부돼야
    한다 — 뺀 이유가 사실이 아니면(예: Catalog 가 나중에 지원을 추가했는데
    이 표를 안 고치면) 이 테스트가 잡는다."""
    report = verify_operator_roundtrip()
    for op in EXCLUDED_OPERATORS:
        assert report[op] is False, f"{op} 가 이제 컴파일된다 — EXCLUDED_OPERATORS 갱신 필요"


def test_operator_roundtrip_report_covers_every_declared_and_excluded_operator():
    """선언한 연산자 집합과 뺀 집합을 합치면 `verify_operator_roundtrip` 이 실제로
    검사한 키 전체와 정확히 일치해야 한다 — 표에 없는 연산자가 조용히 섞여
    들어오는 것을 막는다."""
    report = verify_operator_roundtrip()
    declared = set(BINARY_OPERATORS) | set(UNARY_OPERATORS) | set(EXCLUDED_OPERATORS)
    assert set(report) == declared


# ---------------------------------------------------------------------------
# 착수 조건 2 — free_symbols 명시 검증. `pysr` 없이 `_extract_candidate` 를
# 직접 불러 검사한다 (PySR 이 낸 한 행을 흉내낸 최소 입력만 준다).
# ---------------------------------------------------------------------------

def _xyw(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 2.0 * X[:, 0]
    w = np.ones(n)
    return X, y, w


def test_extract_candidate_rejects_symbol_outside_feature_names():
    """`sympy_format` 이 `feature_names` 에 없는 심볼(`x2`)을 쓰면 명시적으로
    버려야 한다 — `NaiveBackend` 처럼 `lambdify`+`except Exception` 의 우연한
    가드에 기대지 않는다."""
    X, y, w = _xyw()
    rogue = sympy.Symbol("x2")  # allowed_names 에 없는 심볼
    outcome = _extract_candidate(
        bi + rogue, pysr_complexity=3, allowed_names=frozenset(NAMES),
        backend_name="pysr", seed=0, X=X, y=y, w=w, names=NAMES)
    assert outcome.candidate is None
    assert outcome.discard["reason"] == "free_symbols_outside_feature_names"
    assert "x2" in outcome.discard["unknown_symbols"]


def test_free_symbol_check_short_circuits_before_any_lambdify_attempt(monkeypatch):
    """"결과적으로 버려지는가"가 아니라 "명시 검증이 `lambdify` 시도보다 **먼저**
    발동하는가"를 직접 확인한다 — 이 구분이 중요한 이유: `sympy.Symbol("x2")`
    같은 흔한 미지 심볼은 검증을 지워도 `lambdify`+`float()` 캐스트가 (다른
    사유로) 여전히 실패해서 우연히 버려진다(직접 실측 확인: sympy 의
    `lambdify(numpy)` 는 인자 목록에 없는 자유 심볼을 만나면 그 심볼 자체를
    네임스페이스에 넣어 예외 없이 object-dtype 배열을 만들고, `np.asarray(...,
    dtype=float)` 단계에서 `TypeError` 로 실패한다 — `NaiveBackend` 가 우연히
    기대는 것과 같은 종류의 사고다). 이 "결과가 같다"는 우연 때문에 단순히
    `discard is not None` 만 보면 명시 검증이 실제로 도는지 알 수 없다.

    그래서 여기서는 `sympy.lambdify` 자체를 스파이로 바꿔, 범위 밖 심볼이 있는
    후보에서는 **호출되지도 않아야 한다**는 것을 직접 증명한다.
    `_extract_candidate` 의 `free_symbols` 검증 분기를 지우면(`if False:` 로
    바꾸면) `lambdify` 가 실제로 불려서 이 테스트가 FAILED 로 떨어진다 — 직접
    확인함(커밋 전 자기검토 참고)."""
    import sd.sr.pysr_backend as backend_module

    def _boom(*args, **kwargs):
        raise AssertionError(
            "free_symbols 검증을 통과하지 못한 후보에서 sympy.lambdify 가 호출됐다 — "
            "명시 검증이 아니라 lambdify 실패에 기대는 우연한 가드로 퇴화했다")

    monkeypatch.setattr(backend_module.sympy, "lambdify", _boom)

    X, y, w = _xyw()
    rogue = sympy.Symbol("x2")
    outcome = _extract_candidate(
        bi + rogue, pysr_complexity=3, allowed_names=frozenset(NAMES),
        backend_name="pysr", seed=0, X=X, y=y, w=w, names=NAMES)
    assert outcome.candidate is None
    assert outcome.discard["reason"] == "free_symbols_outside_feature_names"
    assert "x2" in outcome.discard["unknown_symbols"]


def test_extract_candidate_keeps_expression_using_only_allowed_symbols():
    X, y, w = _xyw()
    outcome = _extract_candidate(
        2 * bi, pysr_complexity=3, allowed_names=frozenset(NAMES),
        backend_name="pysr", seed=7, X=X, y=y, w=w, names=NAMES)
    assert outcome.discard is None
    assert outcome.candidate is not None
    assert outcome.candidate.backend == "pysr"
    assert outcome.candidate.seed == 7


def test_extract_candidate_rejects_only_when_the_symbol_is_truly_outside_not_a_subset_edge_case():
    """부분집합 검사가 `==` 가 아니라 실제로 `<=` 인지 — 후보가 허용 심볼의
    **진짜 부분집합**(전부는 안 씀)이면 통과해야 한다."""
    X, y, w = _xyw()
    outcome = _extract_candidate(
        bi, pysr_complexity=1, allowed_names=frozenset(NAMES),  # qi 는 안 씀
        backend_name="pysr", seed=0, X=X, y=y, w=w, names=NAMES)
    assert outcome.discard is None
    assert outcome.candidate is not None


# ---------------------------------------------------------------------------
# in_sample_score — 수식을 직접 평가한 값으로 잰다. NaiveBackend 의 "기울기를
# 밖에서 맞추느라 부호 맹목" 결함을 물려받지 않는지 확인한다.
# ---------------------------------------------------------------------------

def test_score_is_sign_aware_unlike_naive_backends_external_refit():
    """`y = 2*x0` 인데 후보가 `-2*x0`(부호 반대)면 `NaiveBackend` 는 밖에서
    기울기를 다시 맞춰(lstsq) 부호를 뒤집고 거의 완벽한 점수를 낸다 — 부호
    맹목이다. `PySRBackend` 는 수식을 있는 그대로 평가해야 하므로 이 경우
    점수가 뚜렷이 나빠야 한다(오히려 크게 음수)."""
    n = 500
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 2.0 * X[:, 0]
    w = np.ones(n)

    correct = _extract_candidate(2 * bi, pysr_complexity=3,
                                 allowed_names=frozenset(NAMES), backend_name="pysr",
                                 seed=0, X=X, y=y, w=w, names=NAMES)
    flipped = _extract_candidate(-2 * bi, pysr_complexity=3,
                                 allowed_names=frozenset(NAMES), backend_name="pysr",
                                 seed=0, X=X, y=y, w=w, names=NAMES)
    assert correct.candidate.in_sample_score > 0.99
    assert flipped.candidate.in_sample_score < -0.9   # 부호가 반대라 크게 나쁘다


# ---------------------------------------------------------------------------
# complexity — `base.complexity_of` 로 재계산. PySR 값과 다르면 note 에 남긴다.
# ---------------------------------------------------------------------------

def test_complexity_is_recomputed_not_taken_from_pysr():
    """`pysr_complexity` 로 명백히 틀린 값(예: 999)을 줘도 `Candidate.complexity`
    는 `complexity_of(expr)` 로 재계산된 값이어야 한다 — PySR 이 준 값을 그대로
    쓰면 이 테스트가 잡는다."""
    X, y, w = _xyw()
    outcome = _extract_candidate(
        bi + qi, pysr_complexity=999, allowed_names=frozenset(NAMES),
        backend_name="pysr", seed=0, X=X, y=y, w=w, names=NAMES)
    assert outcome.candidate.complexity == complexity_of(bi + qi)
    assert outcome.candidate.complexity != 999
    assert outcome.note is not None
    assert outcome.note["pysr_complexity"] == 999
    assert outcome.note["recomputed_complexity"] == complexity_of(bi + qi)


def test_complexity_mismatch_note_is_absent_when_values_agree():
    X, y, w = _xyw()
    true_complexity = complexity_of(bi)
    outcome = _extract_candidate(
        bi, pysr_complexity=true_complexity, allowed_names=frozenset(NAMES),
        backend_name="pysr", seed=0, X=X, y=y, w=w, names=NAMES)
    assert outcome.note is None


# ---------------------------------------------------------------------------
# 생성자 검증 — maxsize 상한(계획서 §3, 복잡도 크리프 방지).
# ---------------------------------------------------------------------------

def test_maxsize_above_plan_ceiling_is_rejected():
    with pytest.raises(ValueError):
        PySRBackend(maxsize=MAXSIZE_CEILING + 1)


def test_maxsize_at_ceiling_is_accepted():
    PySRBackend(maxsize=MAXSIZE_CEILING)  # 예외가 없어야 한다


def test_backend_name_and_default_diagnostics():
    backend = PySRBackend(seed=3)
    assert backend.name == "pysr"
    assert backend.seed == 3
    assert backend.diagnostics == {}


# ---------------------------------------------------------------------------
# `sd.config.ensure_pysr_env` — import pysr 전에 환경변수를 배선한다.
# ---------------------------------------------------------------------------

def test_ensure_pysr_env_sets_the_variable_when_absent(monkeypatch):
    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    result = config.ensure_pysr_env()
    assert os.environ["PYTHON_JULIAPKG_PROJECT"] == str(config.PYSR_JULIA_PROJECT)
    assert result == config.PYSR_JULIA_PROJECT


def test_ensure_pysr_env_does_not_override_an_existing_value(monkeypatch):
    """사용자가 이미 다른 값으로 설정해 뒀으면 존중해야 한다 — 덮어쓰면
    사용자가 의도적으로 고른 Julia 프로젝트 경로를 조용히 무시하는 것이다."""
    monkeypatch.setenv("PYTHON_JULIAPKG_PROJECT", "/some/custom/path")
    result = config.ensure_pysr_env()
    assert os.environ["PYTHON_JULIAPKG_PROJECT"] == "/some/custom/path"
    assert result == Path("/some/custom/path")


def test_ensure_pysr_env_also_sets_juliacall_signal_handling(monkeypatch):
    """`PYTHON_JULIACALL_HANDLE_SIGNALS=yes` (사용자 추가 지시) — juliacall 자신의
    문서가 권하는 세그폴트 완화책이다. `PYTHON_JULIAPKG_PROJECT` 와 **같은 호출**
    에서 배선돼야 한다 — 둘을 따로 부르게 하면 하나만 배선하고 잊어버리는
    사고가 나기 쉽다."""
    monkeypatch.delenv("PYTHON_JULIACALL_HANDLE_SIGNALS", raising=False)
    config.ensure_pysr_env()
    assert os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "yes"


def test_ensure_pysr_env_does_not_override_an_existing_signal_handling_choice(monkeypatch):
    """사용자가 `PYTHON_JULIACALL_HANDLE_SIGNALS=no` 를 이미 정해 뒀으면(예:
    juliacall 자신의 경고가 권하듯 Ctrl-C 를 KeyboardInterrupt 로 받고 싶어서)
    존중해야 한다."""
    monkeypatch.setenv("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")
    config.ensure_pysr_env()
    assert os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "no"


# ---------------------------------------------------------------------------
# 느린 테스트 — 실제 `pysr` 를 부른다 (Julia JIT 워밍업 포함, 수십~수백 초).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weights_are_actually_passed_to_pysr_loss():
    """`X`·`y`·`seed`·`deterministic` 를 전부 고정하고 `w` 만 바꾼다.
    `deterministic=True`(직렬)이므로 PySR 자체는 결정론적이다 — 그러므로 두
    호출의 유일한 차이는 `w` 뿐이고, `model.fit(..., weights=w_fit, ...)` 가
    실제로 그 `w` 를 손실에 쓴다면 두 호출의 Pareto front(각 복잡도의 상수)가
    달라져야 한다. `model.fit()` 에서 `weights=w_fit` 을 빼는 뮤테이션을
    넣으면 두 호출이 완전히 같은 입력을 받게 되어 **바이트 단위로 동일한**
    결과가 나온다 — 이 테스트가 그것을 잡는다(직접 뮤테이션 적용해 FAILED
    확인함, 커밋 전 자기검토 참고).

    가중치는 절대 0을 포함하지 않는다 — `PySRBackend.fit()` 의 `valid` 마스크가
    `w > 0` 도 걸러내므로, 0을 넣으면 행 필터링 차이와 손실-가중치 차이가
    뒤섞여 이 테스트가 무엇을 검증하는지 불분명해진다.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = np.where(np.arange(n) < n // 2, 5.0 * X[:, 0], -5.0 * X[:, 0])

    uniform_w = np.ones(n)
    front_heavy_w = np.where(np.arange(n) < n // 2, 50.0, 1.0)

    common = dict(seed=0, deterministic=True, niterations=6, maxsize=10, verbosity=0)
    candidates_uniform = PySRBackend(**common).fit(X, y, uniform_w, NAMES)
    candidates_front = PySRBackend(**common).fit(X, y, front_heavy_w, NAMES)

    exprs_uniform = sorted(str(c.expr) for c in candidates_uniform)
    exprs_front = sorted(str(c.expr) for c in candidates_front)
    assert exprs_uniform != exprs_front, (
        "가중치를 바꿨는데 결과가 완전히 같다 — weights= 가 PySR 손실에 전달되지 "
        "않고 있을 가능성이 있다")


@pytest.mark.slow
def test_end_to_end_fit_recovers_a_simple_signal_and_tags_candidates_correctly():
    """`y = 2*x0*x1` — ROADMAP.md 2단계가 실측한 바로 그 문제. 컴파일 가능한
    연산자만으로 실제 `pysr` 를 돌려 배선이 끝까지 살아있는지 확인한다."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    y = 2.0 * X[:, 0] * X[:, 1]
    w = np.ones(n)
    names = ("book_imbalance", "queue_imbalance_best", "signed_aggr_flow_20")

    backend = PySRBackend(seed=0, deterministic=True, niterations=6, maxsize=12)
    candidates = backend.fit(X, y, w, names)

    assert candidates
    for candidate in candidates:
        assert candidate.backend == "pysr"
        assert candidate.seed == 0
        used = {str(s) for s in candidate.expr.free_symbols}
        assert used <= set(names)          # 착수 조건 2가 실제로 지켜졌다

    best = max(candidates, key=lambda c: c.in_sample_score)
    assert best.in_sample_score > 0.9

    assert backend.diagnostics["deterministic"] is True
    assert backend.diagnostics["parallelism"] == "serial"
    assert backend.diagnostics["binary_operators"] == list(BINARY_OPERATORS)
    assert backend.diagnostics["unary_operators"] == list(UNARY_OPERATORS)
