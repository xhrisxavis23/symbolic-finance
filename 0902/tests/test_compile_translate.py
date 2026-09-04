import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, derived  # noqa: E402
from sd.compile import check, normalize, to_catalog  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402
from framework import contract as _contract  # noqa: E402

bi = sympy.Symbol("book_imbalance")
qi = sympy.Symbol("queue_imbalance_best")
rc = sympy.Symbol("spread_to_round_trip_cost_ratio")


def test_symbol_becomes_primitive_node():
    assert to_catalog.translate(bi) == {"op": "primitive", "primitive_id": "book_imbalance"}


def test_sum_becomes_add_nodes():
    node = to_catalog.translate(bi + qi)
    assert node["op"] == "add"
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_product_becomes_multiply_node():
    node = to_catalog.translate(bi * qi)
    assert node["op"] == "multiply"


def test_three_term_sum_folds_as_nested_binary_add():
    """Catalog 에는 n항 산술이 없다 — `_fold` 는 3항 이상을 이항으로 왼쪽부터
    접어야 한다. 항을 하나라도 빠뜨리거나 평평한 구조로 무너지면 여기서
    걸린다."""
    node = to_catalog.translate(bi + qi + rc)
    assert node == {
        "op": "add",
        "left": {"op": "add",
                  "left": {"op": "primitive", "primitive_id": "book_imbalance"},
                  "right": {"op": "primitive", "primitive_id": "queue_imbalance_best"}},
        "right": {"op": "primitive", "primitive_id": "spread_to_round_trip_cost_ratio"},
    }
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_three_term_product_folds_as_nested_binary_multiply():
    node = to_catalog.translate(bi * qi * rc)
    assert node == {
        "op": "multiply",
        "left": {"op": "multiply",
                  "left": {"op": "primitive", "primitive_id": "book_imbalance"},
                  "right": {"op": "primitive", "primitive_id": "queue_imbalance_best"}},
        "right": {"op": "primitive", "primitive_id": "spread_to_round_trip_cost_ratio"},
    }
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_abs_becomes_absolute_node():
    node = to_catalog.translate(sympy.Abs(bi))
    assert node == {"op": "absolute", "input": {"op": "primitive", "primitive_id": "book_imbalance"}}


def test_inner_sqrt_becomes_sqrt_node():
    node = to_catalog.translate(sympy.sqrt(sympy.Abs(bi)) + qi)
    assert node["op"] == "add"
    assert catalog.infer_expression_type(node).value_type == "numeric"


# ---- Pow 분기의 지원 지수(1/2, 2)와 미지원 지수(그 외) 경계.
#
# `_walk` 의 `Pow` 처리는 `exp == Rational(1, 2)` 는 `sqrt`, `exp == 2` 는
# 자기 자신과의 `multiply` 로 옮기고, 그 외 지수는 명시적으로 거부한다
# (`raise TranslationError(f"지원하지 않는 지수: {node.exp}. ...")`). 이
# 거부 분기 자체가 없으면 홀수 지수(`x**3`)가 조용히 `x**2` 취급돼 부호
# 정보가 사라진다 — `book_imbalance**3` 이 항상 양수인 `book_imbalance**2`
# 와 같은 것으로 번역되면 진입식이 반대 구간을 고르는데 예외도 경고도
# 없다. 지원 지수 둘이 여전히 통과하는지도 함께 확인해 거부가 과하게
# 넓어지지 않았는지 본다.

def test_sqrt_exponent_becomes_sqrt_node_directly():
    node = to_catalog.translate(sympy.sqrt(sympy.Abs(bi)))
    assert node == {"op": "sqrt",
                     "input": {"op": "absolute",
                                "input": {"op": "primitive", "primitive_id": "book_imbalance"}}}


def test_square_exponent_becomes_self_multiply():
    node = to_catalog.translate(bi**2)
    primitive = {"op": "primitive", "primitive_id": "book_imbalance"}
    assert node == {"op": "multiply", "left": primitive, "right": primitive}


def test_odd_exponent_is_rejected():
    """`x**3` 은 `sqrt`(1/2)도 `multiply(x,x)`(2)도 아니다 — 명시적으로
    거부해야 한다. 거부 대신 `x**2` 로 눙치면 부호가 소실된다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(bi**3)
    assert "3" in str(excinfo.value)


def test_negative_exponent_is_rejected():
    """역수(`x**-1`)는 `ratio`/`divide` 로 명시해야 분모 0 정책이 붙는다 —
    `Pow` 로 몰래 넘기면 그 정책을 우회한다."""
    with pytest.raises(to_catalog.TranslationError):
        to_catalog.translate(bi**-1)


def test_tanh_becomes_tanh_node():
    node = to_catalog.translate(sympy.tanh(bi))
    assert node["op"] == "tanh"


def test_positive_constant_scale_drops_magnitude_and_stays_unwrapped():
    """q 분위(c·u) = c·q 분위(u) (c>0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)] —
    크기는 선택 집합을 바꾸지 않으므로 버려도 되고, 부호가 양수면 부등호
    방향도 바뀌지 않으므로 negate 로 감싸지 않는다.

    **주의 (F1 리뷰):** 이 등식은 `translate()` 의 **최외곽**에서만 성립한다.
    `translate(3 * bi)` 는 `3*bi` 가 진입식 전체이므로 최외곽 Mul 이고, 이
    테스트는 그 경로를 직접 부른다. 파이프라인(`strip_monotone` → `translate`)
    은 실제로 이 경로를 거의 타지 않는다 — `strip_monotone` 이 최외곽 양수
    상수배를 먼저 "scale" 껍질로 벗겨 버리기 때문이다. 그래도 `translate` 를
    직접 부르는 경로(예: 이 테스트, 또는 향후 다른 호출자)가 있을 수 있으므로
    최외곽 동작 자체는 유지한다. 깊이의 상수배(최외곽이 아닌 곳)는 이 등식이
    성립하지 않아 전혀 다르게 취급된다 — 아래 `test_depth_constant_scale_*`
    를 보라."""
    node = to_catalog.translate(3 * bi)
    assert node == {"op": "primitive", "primitive_id": "book_imbalance"}


def test_negative_constant_scale_preserves_sign_via_negate():
    """상수 배율의 부호는 부등호 방향을 바꾸므로 크기와 달리 버릴 수 없다 —
    `negate` 노드로 보존해야 한다. `negate` 를 빼면 `-3*book_imbalance` 와
    `book_imbalance` 가 같은 AST 로 번역돼 선택 집합이 달라진다."""
    node = to_catalog.translate(-3 * bi)
    assert node == {"op": "negate",
                     "input": {"op": "primitive", "primitive_id": "book_imbalance"}}


# ---- F1 (최종 전체 리뷰): 깊이의 상수배는 최외곽과 다르다.
#
# `q분위(c·u) = c·q분위(u)` (c>0) 는 **진입식 전체**에 대한 부등식에서만
# 성립한다. 안쪽 상수는 함수의 모양을 바꾼다 — `tanh(2*x)` 의 포화 스케일은
# `tanh(x)` 와 다르다(`tanh(2)=0.964` vs `tanh(1)=0.762`, 포화점이 이동한다).
# Catalog 에는 산술 안에 수치 리터럴을 놓을 노드가 없다(`raw` 는 field 를
# 요구하고 `primitive` 는 feature 만 받는다) — 그러니 깊이의 상수는 흡수할
# 곳이 없다. 유일하게 옳은 선택은 `TranslationError` 로 거부하는 것이다.
#
# 지금(NaiveBackend)은 이 경로가 발동하지 않는다 — 템플릿에 수치 계수가
# 없기 때문이다. PySR 로 바꾸면 거의 모든 후보에서 발동한다.

sa = sympy.Symbol("signed_aggr_flow_20")


def test_depth_constant_scale_inside_tanh_is_rejected():
    """`sa * tanh(2*bi)` — `tanh` 안의 `2` 는 최외곽이 아니다. 크기를 버리면
    `tanh(2*bi)` 와 `tanh(bi)` 가 같은 AST 로 번역되는데, 이 둘은 실제로는
    포화 스케일이 다른 서로 다른 함수다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sa * sympy.tanh(2 * bi))
    message = str(excinfo.value)
    assert "2" in message
    assert "수치 리터럴" in message


def test_depth_constant_scale_inside_add_term_is_rejected():
    """`2*bi + 3*sa` — 최외곽은 Add 이지 Mul 이 아니다. 각 항의 상수배는
    깊이에 있으므로 거부돼야 한다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(2 * bi + 3 * sa)
    assert "수치 리터럴" in str(excinfo.value)


def test_depth_constant_division_inside_add_term_is_rejected():
    """`bi/2 + sa` — `bi/2` 는 sympy 에서 `Mul(1/2, bi)` 이고 Add 의 항이라
    깊이다. 나눗셈으로 쓴 상수 계수도 곱과 같은 이유로 거부돼야 한다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(bi / 2 + sa)
    assert "수치 리터럴" in str(excinfo.value)


def test_depth_sign_flip_via_subtraction_is_still_translatable():
    """`bi - qi` 는 sympy 에서 `Add(bi, Mul(-1, qi))` — `-qi` 의 상수는 깊이(Add
    의 항)에 있지만 `|c| = 1` 이라 정보 손실 없이 `negate` 로 정확히 옮길 수
    있다. 깊이 상수를 무조건 거부하면 평범한 뺄셈까지 깨진다 — 실제로 F1
    수정 직후 기존 산출물 재검증에서 `book_imbalance - queue_imbalance_best`
    가 이 경로로 잘못 거부되는 회귀가 났었다."""
    node = to_catalog.translate(bi - qi)
    assert node == {
        "op": "add",
        "left": {"op": "primitive", "primitive_id": "book_imbalance"},
        "right": {"op": "negate",
                  "input": {"op": "primitive", "primitive_id": "queue_imbalance_best"}},
    }


# ---- R20 (SDD-LEDGER.md) — 깊이 상수배의 |c|==1 예외가 `sympy.Float` 에서
# 깨졌었다. `abs(product) != 1` 은 sympy 1.14 에서 `Float.__eq__` 가 파이썬
# `int` 와 `float` 를 다르게 다뤄 실패한다: 직접 실측하면
#   abs(sympy.Float(-1.0)) != 1        → True   (틀림)
#   abs(sympy.Float(-1.0)) - 1 != 0    → False  (맞음)
# PySR 은 계수를 항상 부동소수로 낸다 — 이 결함이 고쳐지지 않으면 PySR 이 낸
# 순수 부호반전 계수(정수 -1 이 아니라 -1.0)까지 평범한 뺄셈으로서 오탈락한다.

def test_depth_sign_flip_via_subtraction_with_float_coefficient_is_still_translatable():
    """`book_imbalance - 1.0*queue_imbalance_best` — 계수가 정수 `-1` 이 아니라
    `sympy.Float(-1.0)` 이어도 `bi - qi` 와 정확히 같은 AST 로 옮겨져야 한다.
    옛 비교(`abs(product) != 1`)로 되돌리면 이 테스트가 TranslationError 로
    떨어진다 — PySR 출력 형태를 그대로 흉내낸 회귀 판별 테스트다."""
    node = to_catalog.translate(bi - 1.0 * qi)
    assert node == {
        "op": "add",
        "left": {"op": "primitive", "primitive_id": "book_imbalance"},
        "right": {"op": "negate",
                  "input": {"op": "primitive", "primitive_id": "queue_imbalance_best"}},
    }


def test_depth_float_coefficient_of_negative_one_point_zero_is_translatable():
    """부호가 아니라 `-1.0` 자체를 곱으로 직접 구성해도(뺄셈을 거치지 않고)
    같은 예외가 적용돼야 한다 — `Mul(sympy.Float(-1.0), qi)` 를 Add 항으로
    직접 만든 경우."""
    node = to_catalog.translate(bi + sympy.Float(-1.0) * qi)
    assert node == {
        "op": "add",
        "left": {"op": "primitive", "primitive_id": "book_imbalance"},
        "right": {"op": "negate",
                  "input": {"op": "primitive", "primitive_id": "queue_imbalance_best"}},
    }


def test_depth_float_coefficient_other_than_unit_magnitude_is_still_rejected():
    """R20 수정이 방향을 반대로 뒤집어 과잉 승인하지 않는지 확인한다 — 부동소수
    계수라도 `|c| != 1` 이면 여전히 거부돼야 한다. `abs(product) - 1 != 0` 대신
    항상 `False`를 내는 뮤테이션(검사를 무력화)을 넣으면 이 테스트가 잡는다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(bi + 2.0 * qi)
    assert "수치 리터럴" in str(excinfo.value)


def test_unknown_function_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.exp(bi))
    assert "exp" in str(excinfo.value)


def test_unknown_symbol_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.Symbol("not_a_feature"))
    assert "not_a_feature" in str(excinfo.value)


# ---- 파생 레지스트리 폴백 (ROADMAP.md 1단계). Catalog 에 없는 심볼은 이제
# `sd/derived.py` 를 본 뒤에야 포기한다.

def test_derived_symbol_translates_to_its_registered_ast():
    node = to_catalog.translate(sympy.Symbol("ofi_over_qbar_5"))
    assert node == derived.DERIVED["ofi_over_qbar_5"]
    assert node["op"] == "ratio"                      # primitive 가 아니라 완전히 펼쳐진 AST


def test_derived_symbol_inside_a_larger_expression_still_translates():
    node = to_catalog.translate(bi + sympy.Symbol("ofi_over_qbar_5"))
    assert node["op"] == "add"
    assert node["right"] == derived.DERIVED["ofi_over_qbar_5"]
    assert catalog.infer_expression_type(node).dimension == "dimensionless"


def test_catalog_feature_wins_over_a_colliding_derived_entry(monkeypatch):
    """Catalog feature 가 항상 우선이어야 한다 — `_walk` 가 순서를 뒤집으면
    (파생을 먼저 본다) 이 테스트가 잡는다. 실제로는 `sd/derived.py` 의 import
    시점 검사가 이름 충돌을 막아 이 상황이 일어날 수 없으므로, 여기서는
    `to_catalog._derived.DERIVED` 를 직접 몽키패치해 순서 자체를 검사한다."""
    fake = dict(to_catalog._derived.DERIVED)
    fake["book_imbalance"] = {"op": "primitive", "primitive_id": "queue_imbalance_best"}
    monkeypatch.setattr(to_catalog._derived, "DERIVED", fake)
    node = to_catalog.translate(bi)
    assert node == {"op": "primitive", "primitive_id": "book_imbalance"}


def test_unknown_symbol_error_mentions_both_registries():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.Symbol("nowhere_at_all"))
    message = str(excinfo.value)
    assert "Catalog" in message and "파생" in message


def test_static_check_rejects_type_error():
    bad = {"op": "all", "args": [{"op": "primitive", "primitive_id": "mid_price"}]}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "bool" in str(excinfo.value)


def test_static_check_rejects_vocabulary_outside_catalog():
    bad = {"op": "primitive", "primitive_id": "invented_feature"}
    with pytest.raises(check.CheckError):
        check.static_check(bad, allow_unresolved=False)


def test_static_check_accepts_valid_boolean_entry_expression():
    good = {"op": "all", "args": [
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "book_imbalance"},
         "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance"},
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "spread_to_round_trip_cost_ratio"},
         "comparator": "<", "value": 0.6},
    ]}
    check.static_check(good, allow_unresolved=True)          # 예외가 없어야 한다
    assert catalog.infer_expression_type(good, allow_unresolved=True).value_type == "boolean"


def test_static_check_rejects_execution_binding_override():
    bad = {"op": "compare",
           "input": {"op": "primitive", "primitive_id": "book_imbalance"},
           "comparator": ">", "value": 0.5,
           "execution_binding": {"entry_execution": "MARKET"}}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "실행" in str(excinfo.value)


def test_static_check_rejects_execution_binding_hidden_at_depth_three():
    """FORBIDDEN_KEYS 검사는 재귀적이어야 한다 — 최상위가 아니라 깊이 3에
    숨긴 execution_binding 도 잡아야 한다. 최상위만 보는 얕은 검사는 여기서
    통과해 버려 실행 계약이 몰래 바뀐 진입식이 그대로 넘어간다."""
    bad = {"op": "all", "args": [
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "book_imbalance",
                    "execution_binding": {"entry_execution": "MARKET"}},
         "comparator": ">", "value": 0.5},
    ]}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "실행" in str(excinfo.value)


def test_static_check_rejects_execution_binding_hidden_inside_a_list():
    """리스트(`args`) 안에 숨긴 노드의 forbidden key 도 잡아야 한다. 딕셔너리만
    재귀하고 리스트를 안 훑으면 `all`/`any` 의 args 안에 숨긴 것을 놓친다."""
    bad = {"op": "all", "args": [
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "book_imbalance"},
         "comparator": ">", "value": 0.5},
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "spread_to_round_trip_cost_ratio"},
         "comparator": "<", "value": 0.6,
         "fee_bps": 0.0},
    ]}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "실행" in str(excinfo.value)


# ---- catalog.condition_problem() 검사 (ROADMAP.md 1단계 · DESIGN.md D14 정정).
#
# `queue_imbalance_best` 는 `book_imbalance` 의 정확한 아핀 변환이라 Catalog
# 정책이 `derived_duplicate` 로 임계 조건의 **직접 대상**이 되는 것을 금지한다
# (재료로 쓰는 것은 허용). 실측(D14): 지난 실행에서 `queue_imbalance_best` 를
# 직접 대상으로 쓴 진입식은 e001(그대로)과 e005(`sqrt(Abs(...))`에서 sqrt 가
# 벗겨져 `absolute(primitive(...))`로 남은 것) 둘이었다. 아래 테스트들이 그
# 정확한 두 AST 모양을 재현한다.

def _compare(input_ast, value=0.5):
    return {"op": "compare", "input": input_ast, "comparator": ">", "value": value}


def test_static_check_rejects_bare_derived_duplicate_as_direct_condition_target():
    """D14 의 e001: `queue_imbalance_best` 를 그대로 임계 조건에 걸었다."""
    bad = _compare({"op": "primitive", "primitive_id": "queue_imbalance_best"})
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    message = str(excinfo.value)
    assert "queue_imbalance_best" in message
    assert "book_imbalance" in message          # 대안 힌트


def test_static_check_rejects_absolute_wrapped_derived_duplicate_as_direct_target():
    """D14 의 e005: `sqrt(Abs(queue_imbalance_best))` 에서 최외곽 sqrt 가
    strip_monotone 에 벗겨지고 `absolute(primitive(...))` 만 남는다. absolute
    는 부호가 아니라 크기만 바꾸는 wrapper 라 여전히 "같은 축"이다."""
    bad = _compare({"op": "absolute",
                    "input": {"op": "primitive", "primitive_id": "queue_imbalance_best"}})
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "queue_imbalance_best" in str(excinfo.value)


def test_static_check_rejects_negate_wrapped_direct_target():
    """negate wrapper 도 같은 축이다 — `derived_duplicate` 가 아니라
    `execution_only`(가격 수준) 축으로 한 번 더 확인한다."""
    bad = _compare({"op": "negate",
                    "input": {"op": "primitive", "primitive_id": "mid_price"}})
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "mid_price" in str(excinfo.value)


def test_static_check_rejects_banned_target_nested_inside_all():
    """`all`/`any` 의 `args` 안에 숨은 `compare` 도 잡아야 한다 — forbidden key
    검사와 같은 재귀 요구사항이다."""
    bad = {"op": "all", "args": [
        _compare({"op": "primitive", "primitive_id": "book_imbalance"}),
        _compare({"op": "primitive", "primitive_id": "queue_imbalance_best"}),
    ]}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "queue_imbalance_best" in str(excinfo.value)


def test_static_check_rejects_crossover_on_a_banned_target():
    """`crossover` 도 `compare` 와 마찬가지로 `input` 을 리터럴 임계와 직접
    비교하는 노드다 — 같은 규칙이 적용돼야 한다."""
    bad = {"op": "crossover",
           "input": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
           "threshold": 0.0, "direction": "above", "equality": "strict"}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "queue_imbalance_best" in str(excinfo.value)


def test_static_check_allows_banned_feature_used_as_material_not_as_target():
    """핵심 구분: `book_imbalance + queue_imbalance_best` 처럼 다른 feature 와
    **결합**되면 그 순간부터 "재료"이지 "직접 대상"이 아니다 — D14 의 e002·e008
    이 실제로 컴파일에 성공했던 것과 같은 모양이다. 과잉 거부하면 이 테스트가
    잡는다."""
    good = _compare({"op": "add",
                     "left": {"op": "primitive", "primitive_id": "book_imbalance"},
                     "right": {"op": "primitive", "primitive_id": "queue_imbalance_best"}})
    check.static_check(good, allow_unresolved=False)   # 예외가 없어야 한다


def test_static_check_allows_negated_material_combination():
    """D14 의 e008 그대로: `book_imbalance - queue_imbalance_best`. 뺄셈은
    sympy 에서 `Add(bi, Mul(-1, qi))` 이므로 `negate` 가 `add` 의 자식으로
    깊이 있다 — `add` 가 최외곽이라 여전히 재료다."""
    good = _compare({"op": "add",
                     "left": {"op": "primitive", "primitive_id": "book_imbalance"},
                     "right": {"op": "negate",
                               "input": {"op": "primitive",
                                        "primitive_id": "queue_imbalance_best"}}})
    check.static_check(good, allow_unresolved=False)   # 예외가 없어야 한다


def test_static_check_does_not_police_compare_values_targets():
    """`compare_values` 는 리터럴 임계가 없다 — 두 표현식을 서로 비교할 뿐이다.
    태스크 범위를 `compare`(와 `crossover`)로 좁힌 의도적 결정이다. 이 테스트가
    없으면 나중에 누가 "당연히 여기도 막아야지" 하고 조용히 넓혀도 아무도
    모른다."""
    ok = {"op": "compare_values",
          "left": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
          "right": {"op": "primitive", "primitive_id": "book_imbalance"},
          "comparator": ">"}
    check.static_check(ok, allow_unresolved=False)     # 예외가 없어야 한다


def test_static_check_allows_direct_condition_on_a_rolling_quantile_only_feature():
    """`ofi_depth_5` 의 정책은 `rolling_prior_100_ticks_quantile` 만 허용이고
    `NON_CONDITION_THRESHOLD_KINDS` 와 겹치지 않는다 — `condition_problem` 이
    `None` 을 돌려주고 직접 대상으로도 허용돼야 한다. 파생 열 그룹(OFI/Q̄)의
    분자로 쓰는 바로 그 feature 라 이 축이 막히면 파생 열 확장의 의미가
    없어진다."""
    assert catalog.condition_problem("ofi_depth_5") is None
    good = _compare({"op": "primitive", "primitive_id": "ofi_depth_5"})
    check.static_check(good, allow_unresolved=False)   # 예외가 없어야 한다


# ---- 교차 태스크 계약: log 는 strip_monotone → translate 파이프라인을 종단으로
# 통과하지 못한다. Task 10 에서 MONOTONE_UNARY 에서 log 를 뺀 이유가 바로 이것
# (최외곽 log 가 먼저 벗겨지면 translate 의 명시적 거부에 도달하지 못해 우회됨).
# 어느 한쪽 모듈만 보면 이 계약이 실제로 지켜지는지 알 수 없다 — 두 모듈을
# 실제로 이어 붙여 확인해야 한다.

def test_log_survives_strip_monotone_unstripped():
    """strip_monotone 은 최외곽 log 를 벗기지 않고 그대로 돌려준다."""
    expr = sympy.log(bi)
    stripped, shells = normalize.strip_monotone(expr)
    assert stripped == expr
    assert shells == []


def test_log_is_rejected_by_translate_after_strip_monotone():
    """strip_monotone 을 통과한 결과를 translate 에 넣으면 TranslationError.

    즉 log(book_imbalance) 같은 후보는 파이프라인 순서(strip_monotone 뒤
    translate)를 그대로 따라가도 실제로 탈락해야 한다 — 두 모듈을 각각
    단위 테스트로만 봐서는 이 배선이 살아있는지 확인할 수 없다.
    """
    expr = sympy.log(bi)
    stripped, _shells = normalize.strip_monotone(expr)
    with pytest.raises(to_catalog.TranslationError):
        to_catalog.translate(stripped)


def test_log_inside_larger_expression_is_also_rejected_end_to_end():
    """log 가 최외곽이 아니라 더 큰 식 안에 있어도 파이프라인을 거쳐 탈락한다."""
    expr = sympy.tanh(sympy.log(bi) + qi)
    stripped, shells = normalize.strip_monotone(expr)
    # 바깥 tanh 는 순증가라 벗겨지지만, 안의 log 는 구조로 남는다.
    assert shells == ["tanh"]
    with pytest.raises(to_catalog.TranslationError):
        to_catalog.translate(stripped)


def test_translation_error_and_check_error_are_distinct_types():
    """번역 실패(`TranslationError`)와 정적 검사 실패(`CheckError`)는 서로 다른
    예외 클래스라야 한다. 둘을 하나로 합치면(또는 하나가 다른 하나의 서브클래스가
    되면) 호출자가 "sympy 식을 Catalog 문법으로 아예 못 옮겼다"와 "옮기긴
    했는데 실행 계약을 어겼다"를 구분할 수 없다 — 둘은 파이프라인의 다른
    단계에서 나는, 다른 조치를 요구하는 실패다."""
    assert check.CheckError is not to_catalog.TranslationError
    assert not issubclass(check.CheckError, to_catalog.TranslationError)
    assert not issubclass(to_catalog.TranslationError, check.CheckError)

    # translate 의 실패는 CheckError 로는 안 잡힌다.
    with pytest.raises(to_catalog.TranslationError):
        try:
            to_catalog.translate(sympy.Symbol("not_a_feature"))
        except check.CheckError:  # pragma: no cover - 잡히면 안 된다
            pytest.fail("번역 실패가 CheckError 로 위장됐다")

    # static_check 의 실패는 TranslationError 로는 안 잡힌다.
    bad = {"op": "primitive", "primitive_id": "invented_feature"}
    with pytest.raises(check.CheckError):
        try:
            check.static_check(bad, allow_unresolved=False)
        except to_catalog.TranslationError:  # pragma: no cover - 잡히면 안 된다
            pytest.fail("정적 검사 실패가 TranslationError 로 위장됐다")


# ---- 나눗셈 → `ratio` (사용자 추가 지시). sympy 는 `a/b` 를 `Mul(a, Pow(b,-1))`
# 로 표현한다. 거부 사유 자체가 해법을 말하고 있었다: Catalog 의 `ratio` 가 이미
# "분모 0 정책"(`zero_policy`)을 갖고 있는데 `_walk` 에 매핑이 없어 나눗셈
# 전체가 막혀 있었다. `RATIO_ZERO_POLICY = "nan"` 을 쓴다 — DESIGN.md §7 이 세운
# "0 으로 채우면 나쁜 거래만 사라진다" 원칙과 같은 이유로 "zero" 를 피하고,
# 실데이터의 우연한 분모 0/결측 한 틱 때문에 진입식 전체가 죽는 것을 피하려고
# "reject" 도 피한다. `sd/derived.py` 의 기존 파생 열 전부가 이미 "nan" 을
# 쓰므로 새 관례가 아니라 기존 관례를 따르는 것이다.

qb = sympy.Symbol("queue_imbalance_best")


def test_simple_division_becomes_ratio_node():
    """`a/b` → `ratio(numerator=a, denominator=b, zero_policy="nan")`."""
    node = to_catalog.translate(bi / qb)
    assert node == {
        "op": "ratio",
        "numerator": {"op": "primitive", "primitive_id": "book_imbalance"},
        "denominator": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
        "zero_policy": "nan",
    }
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_division_by_a_product_folds_denominator_with_multiply():
    """`a/(b*c)` — sympy 는 이것을 `Mul(a, Pow(b,-1), Pow(c,-1))` 로, 분모마다
    별도 `Pow` 노드로 남긴다(하나로 뭉치지 않는다, `sympy.srepr` 로 직접 확인).
    분모 인자를 전부 모아 `multiply` 로 접어야 한다 — 하나라도 빠뜨리면 분모가
    틀린다."""
    node = to_catalog.translate(bi / (qb * sa))
    assert node["op"] == "ratio"
    assert node["numerator"] == {"op": "primitive", "primitive_id": "book_imbalance"}
    assert node["denominator"] == {
        "op": "multiply",
        "left": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
        "right": {"op": "primitive", "primitive_id": "signed_aggr_flow_20"},
    }
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_product_divided_by_a_symbol_folds_numerator_with_multiply():
    """`(a*b)/c` — 분자 쪽에 여러 인자가 있으면 `multiply` 로 접어야 한다."""
    node = to_catalog.translate((bi * qb) / rc)
    assert node["op"] == "ratio"
    assert node["numerator"] == {
        "op": "multiply",
        "left": {"op": "primitive", "primitive_id": "book_imbalance"},
        "right": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
    }
    assert node["denominator"] == {"op": "primitive",
                                   "primitive_id": "spread_to_round_trip_cost_ratio"}


def test_division_inside_a_larger_sum_still_translates():
    """`a/b + c` — `ratio` 노드가 `add` 의 자식으로 깊이 있어도 옮겨져야 한다."""
    node = to_catalog.translate(bi / qb + sa)
    assert node["op"] == "add"
    ratio_side = node["left"] if node["left"]["op"] == "ratio" else node["right"]
    assert ratio_side == {
        "op": "ratio",
        "numerator": {"op": "primitive", "primitive_id": "book_imbalance"},
        "denominator": {"op": "primitive", "primitive_id": "queue_imbalance_best"},
        "zero_policy": "nan",
    }


def test_constant_over_symbol_is_rejected_numerator_is_bare_constant():
    """`2/a` — 분자가 상수 `2` 뿐이다(다른 feature 와 곱해지지 않았다). Catalog
    에는 산술 안에 수치 리터럴을 놓을 노드가 없어 옮길 수 없다 — 최외곽에서도
    (양수 상수배를 버리는 규칙은 "u 가 이미 옮길 수 있을 때"만 뜻이 있다. u 자체가
    옮길 수 없으면 버릴 크기도 없다)."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(2 / bi)
    assert "분자" in str(excinfo.value)


def test_constant_over_symbol_is_rejected_even_at_depth():
    """`sa + 2/a` — 같은 거부가 깊이(Add 의 항)에서도 성립해야 한다. 깊이 상수배
    규칙(F1·R20)과 분모 없음 규칙이 상호작용해도 여전히 명확히 거부돼야 한다 —
    조용히 다른 값으로 새지 않는지가 핵심이다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sa + 2 / bi)
    assert "분자" in str(excinfo.value)


def test_bare_reciprocal_without_any_multiplier_is_rejected():
    """`1/a` — sympy 는 이것을 `Mul` 로 감싸지 않고 `Pow(a,-1)` 을 그대로 최상위에
    남긴다(`Mul` 분기를 아예 타지 않는다). `Pow` 분기에 별도 처리가 없으면 이
    경로가 조용히 다른 지수 취급으로 새거나 예외 메시지가 엉뚱해진다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(1 / bi)
    assert "역수" in str(excinfo.value) or "분자" in str(excinfo.value)


def test_bare_reciprocal_inside_add_is_also_rejected():
    """`sa + 1/a` — `Add(sa, Pow(bi,-1))` 형태로 `Pow` 분기가 Add 의 항으로서
    직접 불린다."""
    with pytest.raises(to_catalog.TranslationError):
        to_catalog.translate(sa + 1 / bi)


def test_negative_two_exponent_is_still_rejected_not_silently_treated_as_ratio():
    """`a**-2` — `-1` 만 `ratio` 로 표현 가능하다. `-2` 를 조용히 받아들이면
    (예: 분모 두 번 곱하기로 오인) 의미가 완전히 달라진다."""
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(bi**-2)
    assert "-2" in str(excinfo.value)


def test_division_where_denominator_is_negated_still_translates():
    """`a/(-b)` = `Mul(a, Pow(Mul(-1,b), -1))` — sympy 가 부호를 분모 안으로
    끌고 들어갈 수도 있다(정확한 내부 표현은 버전에 따라 다를 수 있으니, 여기서는
    번역이 **성공하고** 부호를 잃지 않는지만 확인한다 — `infer_expression_type`
    이 통과하고, 실제 배열 평가가 `a/(-b)` 와 일치하는지는 아래 시맨틱 보존
    테스트가 더 강하게 확인한다)."""
    node = to_catalog.translate(bi / (-qb))
    assert catalog.infer_expression_type(node).value_type == "numeric"


# ---- 의미 보존: `ExpressionRuntime` 실제 배열 평가를 sympy 평가값과 대조한다.
#
# AST 모양만 맞고 실행 결과가 다르면 아무 의미가 없다 — `test_derived.py` 의
# `_synthetic_book_arrays` 와 같은 방식으로 합성 호가 데이터를 만들고, Catalog
# feature 값을 직접 얻어 `numerator/denominator` 를 numpy 로 독립적으로 나눈
# 값과 `ExpressionRuntime.evaluate(ast)` 결과를 비교한다.

def _synthetic_book_arrays(n: int = 300, depth: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    tick = 5.0
    base = 10_000.0 + np.cumsum(rng.normal(0.0, 3.0, size=n))
    levels = np.arange(depth)
    bid_price = base[:, None] - tick * (levels[None, :] + 1)
    ask_price = base[:, None] + tick * (levels[None, :] + 1)
    bid_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    ask_qty = rng.integers(1, 200, size=(n, depth)).astype(float)
    return {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": bid_qty, "ask_qty": ask_qty}


def test_ratio_translation_matches_independently_computed_division():
    """`book_imbalance / queue_imbalance_best` 를 실제 합성 데이터 위에서
    평가해, 두 feature 값을 직접 얻어 numpy 로 나눈 값과 일치하는지 확인한다.
    `numerator`/`denominator` 가 뒤바뀌거나 엉뚱한 feature 를 참조하면 이
    비교가 어긋난다."""
    arrays = _synthetic_book_arrays(seed=11)
    runtime = _contract.ExpressionRuntime(arrays)

    a = np.asarray(runtime.feature("book_imbalance"))
    b = np.asarray(runtime.feature("queue_imbalance_best"))
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    expected = np.full(len(a), np.nan)
    expected[valid] = a[valid] / b[valid]

    node = to_catalog.translate(bi / qb)
    got = np.asarray(runtime.evaluate(node))

    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got[valid], expected[valid])
    assert valid.sum() > 100, "표본이 너무 적어 이 검사가 사실상 아무것도 확인하지 않는다"


def test_ratio_translation_of_product_over_symbol_matches_independent_computation():
    """`(book_imbalance * queue_imbalance_best) / spread_to_round_trip_cost_ratio`
    — 분자가 `multiply` 로 접힌 경우도 값이 맞는지 확인한다."""
    arrays = _synthetic_book_arrays(seed=12)
    runtime = _contract.ExpressionRuntime(arrays)

    a = np.asarray(runtime.feature("book_imbalance"))
    b = np.asarray(runtime.feature("queue_imbalance_best"))
    c = np.asarray(runtime.feature("spread_to_round_trip_cost_ratio"))
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (c != 0)
    expected = np.full(len(a), np.nan)
    expected[valid] = (a[valid] * b[valid]) / c[valid]

    node = to_catalog.translate((bi * qb) / rc)
    got = np.asarray(runtime.evaluate(node))

    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got[valid], expected[valid])
    assert valid.sum() > 100


def test_ratio_zero_policy_is_nan_not_zero_or_reject():
    """`RATIO_ZERO_POLICY` 상수 자체가 실제로 "nan" 인지 — 이 값이 나중에 실수로
    "zero" 나 "reject" 로 바뀌면 이 테스트가 잡는다. 주석의 근거(DESIGN.md §7
    "0 으로 채우면 나쁜 거래만 사라진다")가 코드에 실제로 반영됐는지의 최소
    확인이다."""
    assert to_catalog.RATIO_ZERO_POLICY == "nan"
    node = to_catalog.translate(bi / qb)
    assert node["zero_policy"] == to_catalog.RATIO_ZERO_POLICY
