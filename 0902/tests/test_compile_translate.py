import sys
from pathlib import Path

import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402
from sd.compile import check, normalize, to_catalog  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402

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


def test_tanh_becomes_tanh_node():
    node = to_catalog.translate(sympy.tanh(bi))
    assert node["op"] == "tanh"


def test_positive_constant_scale_drops_magnitude_and_stays_unwrapped():
    """q 분위(c·u) = c·q 분위(u) (c>0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)] —
    크기는 선택 집합을 바꾸지 않으므로 버려도 되고, 부호가 양수면 부등호
    방향도 바뀌지 않으므로 negate 로 감싸지 않는다."""
    node = to_catalog.translate(3 * bi)
    assert node == {"op": "primitive", "primitive_id": "book_imbalance"}


def test_negative_constant_scale_preserves_sign_via_negate():
    """상수 배율의 부호는 부등호 방향을 바꾸므로 크기와 달리 버릴 수 없다 —
    `negate` 노드로 보존해야 한다. `negate` 를 빼면 `-3*book_imbalance` 와
    `book_imbalance` 가 같은 AST 로 번역돼 선택 집합이 달라진다."""
    node = to_catalog.translate(-3 * bi)
    assert node == {"op": "negate",
                     "input": {"op": "primitive", "primitive_id": "book_imbalance"}}


def test_unknown_function_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.exp(bi))
    assert "exp" in str(excinfo.value)


def test_unknown_symbol_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.Symbol("not_a_feature"))
    assert "not_a_feature" in str(excinfo.value)


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
