"""E0 복원 판정 함수 테스트. 태스크 지시의 핵심 요구:

    "복원 판정 함수가 실제로 판별하는가 — 명백히 틀린 수식을 넣어도
    '복원됨'이 나오는가. 이게 가장 중요하다."

그래서 각 판정 함수마다 **양성**(진짜 그 형태)과 **음성**(형태가 명백히
다른) 짝을 반드시 함께 둔다. 음성 쪽이 통과하면 그 테스트가 바로 잡는다.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.e0 import criteria  # noqa: E402

I, S = sympy.symbols("book_imbalance spread_to_round_trip_cost_ratio")
X1 = sympy.symbols("ofi_20t_over_qbar")
V, SIG = sympy.symbols("signed_volume_20t_over_qbar rv5_over_spread")
RV5, RV20, RV100 = sympy.symbols("rv5_over_spread rv20_over_spread rv100_over_spread")
DT, CNT = sympy.symbols("dt_prev_over_median recent_count_norm")


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# L1 — 단조 홀함수 in I.
# ---------------------------------------------------------------------------

def _l1_data(n=2000, seed=0):
    rng = _rng(seed)
    i_col = rng.uniform(-0.95, 0.95, n)
    s_col = rng.uniform(1.0, 5.0, n)
    return np.column_stack([i_col, s_col])


def test_l1_true_odd_monotonic_function_is_recovered():
    X = _l1_data()
    expr = 3.0 * I + 0.1 * I * S          # I 에 대해 단조증가, 홀함수 (S 고정 시)
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    assert result["passed"] is True
    assert result["ambiguous"] is False


def test_l1_even_function_is_rejected():
    """짝함수(부호 대칭 없음)는 복원되지 않아야 한다 — 가장 기본적인 오탐 방지."""
    X = _l1_data()
    expr = I**2 + S            # 짝함수. 단조도 아니고 홀함수도 아니다
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    assert result["passed"] is False


def test_l1_non_monotonic_cubic_within_range_is_rejected():
    """`I - 3*I**3` 은 홀함수이지만 |I|~1 범위에서 극값을 지나 비단조다."""
    X = _l1_data()
    expr = I - 3.0 * I**3
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    assert result["passed"] is False
    # 홀함수 성질 자체는 참이어야 한다 (판정이 단조에서만 걸렸는지 확인)
    evaluated = [s for s in result["slices"] if not s.get("ambiguous")]
    assert evaluated and all(s["odd"] for s in evaluated)
    assert evaluated and not any(s["monotonic"] for s in evaluated)


def test_l1_expression_ignoring_imbalance_is_rejected_and_not_ambiguous():
    X = _l1_data()
    expr = S + 1.0     # book_imbalance 를 아예 안 쓴다
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    assert result["passed"] is False
    assert result["ambiguous"] is False
    assert "의존하지 않는다" in result["reason"]


def test_l1_constant_expression_is_rejected():
    X = _l1_data()
    expr = 0.0 * I + 5.0
    # depends_on(expr, "book_imbalance") 는 False 다(0*I 는 sympy 가 0 으로 접는다) —
    # 그래도 "의존하지 않는다" 경로로 확실히 거부되는지 확인한다.
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    assert result["passed"] is False


def test_l1_offset_breaks_oddness():
    """홀함수에 상수를 더하면(수직 이동) 더 이상 홀함수가 아니다 — 오프셋에
    민감한지 확인한다(뮤테이션: odd 판정에서 상수항을 무시하면 이 테스트가
    깨진다)."""
    X = _l1_data()
    expr = 3.0 * I + 2.0     # 단조는 유지되지만 홀함수는 깨진다
    result = criteria.judge_l1(expr, ("book_imbalance", "spread_to_round_trip_cost_ratio"), X)
    evaluated = [s for s in result["slices"] if not s.get("ambiguous")]
    assert evaluated and all(s["monotonic"] for s in evaluated)
    assert evaluated and not any(s["odd"] for s in evaluated)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# L2 — 근사 아핀, 기울기 양수.
# ---------------------------------------------------------------------------

def _l2_data(n=1000, seed=0):
    rng = _rng(seed)
    return rng.uniform(-2.0, 2.0, n).reshape(-1, 1)


def test_l2_true_linear_positive_slope_is_recovered():
    X = _l2_data()
    expr = 0.73 * X1 + 0.02
    result = criteria.judge_l2(expr, ("ofi_20t_over_qbar",), X)
    assert result["passed"] is True
    assert result["slope_mean"] == pytest.approx(0.73, abs=1e-6)
    assert result["slope_cv"] == pytest.approx(0.0, abs=1e-6)


def test_l2_negative_slope_is_rejected():
    """부호가 반대인 선형 법칙은 'β > 0' 조건에서 명백히 실패해야 한다."""
    X = _l2_data()
    expr = -0.5 * X1
    result = criteria.judge_l2(expr, ("ofi_20t_over_qbar",), X)
    assert result["passed"] is False
    assert result["slope_mean"] < 0


def test_l2_quadratic_is_rejected_as_not_affine():
    X = _l2_data()
    expr = X1**2
    result = criteria.judge_l2(expr, ("ofi_20t_over_qbar",), X)
    assert result["passed"] is False
    assert result["slope_cv"] > criteria.AFFINE_CV_MAX


def test_l2_sqrt_like_nonlinearity_is_rejected():
    X = _l2_data()
    expr = sympy.sign(X1) * sympy.sqrt(sympy.Abs(X1))
    result = criteria.judge_l2(expr, ("ofi_20t_over_qbar",), X)
    assert result["passed"] is False


def test_l2_expression_ignoring_input_is_rejected():
    X = _l2_data()
    expr = sympy.Integer(5)
    result = criteria.judge_l2(expr, ("ofi_20t_over_qbar",), X)
    assert result["passed"] is False
    assert result["slope_mean"] == 0.0


# ---------------------------------------------------------------------------
# L3′ — 로그-로그 지수가 0.4~0.7.
# ---------------------------------------------------------------------------

def _l3_data(n=1000, seed=0):
    rng = _rng(seed)
    v = rng.uniform(-5.0, 5.0, n)
    sig = rng.uniform(0.5, 3.0, n)
    return np.column_stack([v, sig])


def test_l3_square_root_exponent_is_recovered():
    X = _l3_data()
    expr = sympy.sign(V) * sympy.sqrt(sympy.Abs(V)) * SIG
    result = criteria.judge_l3(expr, ("signed_volume_20t_over_qbar", "rv5_over_spread"), X,
                               volume_name="signed_volume_20t_over_qbar", sigma_name="rv5_over_spread")
    assert result["passed"] is True
    assert result["exponent"] == pytest.approx(0.5, abs=0.05)


def test_l3_linear_exponent_one_is_rejected():
    """지수 1(선형 임팩트)은 0.4~0.7 구간 밖이라 거부돼야 한다."""
    X = _l3_data()
    expr = 2.0 * V
    result = criteria.judge_l3(expr, ("signed_volume_20t_over_qbar", "rv5_over_spread"), X,
                               volume_name="signed_volume_20t_over_qbar", sigma_name="rv5_over_spread")
    assert result["passed"] is False
    assert result["exponent"] == pytest.approx(1.0, abs=0.05)


def test_l3_exponent_far_below_range_is_rejected():
    X = _l3_data()
    expr = sympy.sign(V) * sympy.Abs(V) ** sympy.Float(0.1)
    result = criteria.judge_l3(expr, ("signed_volume_20t_over_qbar", "rv5_over_spread"), X,
                               volume_name="signed_volume_20t_over_qbar", sigma_name="rv5_over_spread")
    assert result["passed"] is False
    assert result["exponent"] < 0.4


def test_l3_constant_expression_is_ambiguous_not_passed():
    """상수는 로그-로그 회귀 자체가 무의미하다(R² ~ 0) — ambiguous 로 표시하고
    통과시키지 않아야 한다."""
    X = _l3_data()
    expr = sympy.Integer(3)
    result = criteria.judge_l3(expr, ("signed_volume_20t_over_qbar", "rv5_over_spread"), X,
                               volume_name="signed_volume_20t_over_qbar", sigma_name="rv5_over_spread")
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# L4′ — 세 스케일의 (준)선형 결합, 전부 양의 기여.
# ---------------------------------------------------------------------------

def _l4_data(n=1000, seed=0):
    rng = _rng(seed)
    return rng.uniform(0.1, 5.0, size=(n, 3))


NAMES4 = ("rv5_over_spread", "rv20_over_spread", "rv100_over_spread")


def test_l4_true_positive_linear_combination_is_recovered():
    X = _l4_data()
    expr = 0.5 * RV5 + 0.3 * RV20 + 0.2 * RV100 + 0.1
    result = criteria.judge_l4(expr, NAMES4, X)
    assert result["passed"] is True
    assert result["uses_all_three"] is True
    assert result["all_positive"] is True


def test_l4_negative_coefficient_on_one_scale_is_rejected():
    X = _l4_data()
    expr = 0.5 * RV5 - 0.3 * RV20 + 0.2 * RV100
    result = criteria.judge_l4(expr, NAMES4, X)
    assert result["passed"] is False
    assert result["all_positive"] is False


def test_l4_using_only_one_scale_is_rejected_as_not_multiscale():
    """다중 스케일이라 부르려면 세 스케일 모두 실질적으로 쓰여야 한다."""
    X = _l4_data()
    expr = 0.9 * RV5     # RV20·RV100 은 계수가 0
    result = criteria.judge_l4(expr, NAMES4, X)
    assert result["passed"] is False
    assert result["uses_all_three"] is False


def test_l4_strongly_nonlinear_combination_is_rejected():
    X = _l4_data()
    expr = RV5 * RV20 * RV100    # 곱셈 — 선형결합이 아니다
    result = criteria.judge_l4(expr, NAMES4, X)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# L5 — exp(-βΔt) 또는 멱함수 커널, 감쇠(β>0 혹은 γ>0).
# ---------------------------------------------------------------------------

def _l5_data(n=1000, seed=0):
    rng = _rng(seed)
    dt = rng.uniform(0.01, 5.0, n)
    cnt = rng.uniform(0.0, 3.0, n)
    return np.column_stack([dt, cnt])


NAMES5 = ("dt_prev_over_median", "recent_count_norm")


def test_l5_true_exponential_decay_is_recovered():
    X = _l5_data()
    expr = 1.0 + 2.0 * sympy.exp(-0.5 * DT)
    result = criteria.judge_l5(expr, NAMES5, X, dt_name="dt_prev_over_median")
    assert result["passed"] is True
    assert result["chosen"] == "exponential"
    assert result["exponential"]["beta"] == pytest.approx(0.5, rel=0.2)


def test_l5_growing_exponential_is_rejected_despite_good_fit():
    """지수함수 형태는 맞지만 부호가 반대(감쇠가 아니라 성장)면 커널이 아니다."""
    X = _l5_data()
    expr = 1.0 + 0.1 * sympy.exp(0.5 * DT)
    result = criteria.judge_l5(expr, NAMES5, X, dt_name="dt_prev_over_median")
    assert result["passed"] is False


def test_l5_power_law_decay_is_recovered_via_power_law_branch():
    X = _l5_data()
    expr = 0.5 + 3.0 / (DT + 0.1)      # ~ (Δt)^-1 형태의 멱함수 감쇠
    result = criteria.judge_l5(expr, NAMES5, X, dt_name="dt_prev_over_median")
    assert result["passed"] is True
    assert result["chosen"] == "power_law"


def test_l5_expression_ignoring_dt_is_rejected():
    X = _l5_data()
    expr = 1.0 + 0.2 * CNT
    result = criteria.judge_l5(expr, NAMES5, X, dt_name="dt_prev_over_median")
    assert result["passed"] is False
    assert result["ambiguous"] is False


def test_l5_pure_linear_growth_in_dt_is_rejected_as_neither_form():
    """감쇠도 성장도 아니라 그냥 선형이면 지수/멱함수 어느 쪽으로도 잘 안 맞아야
    한다(멱함수·지수 둘 다 완만한 곡률을 가정하는데 순수 선형은 다르다)."""
    X = _l5_data()
    expr = 5.0 * DT
    result = criteria.judge_l5(expr, NAMES5, X, dt_name="dt_prev_over_median")
    # 선형 성장은 감쇠(decay_positive)가 아니므로 최소한 무조건 통과해서는 안 된다.
    assert result["passed"] is False
