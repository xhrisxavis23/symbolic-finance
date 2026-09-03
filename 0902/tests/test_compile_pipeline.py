import sys
from pathlib import Path

import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402
from sd.compile import pipeline, threshold  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402

bi = sympy.Symbol("book_imbalance")


def _candidate(expr, score=0.5):
    return Candidate(expr=expr, complexity=3, in_sample_score=score,
                     backend="naive", seed=0)


def test_attach_makes_a_boolean_expression():
    numeric = {"op": "primitive", "primitive_id": "book_imbalance"}
    entry = threshold.attach(numeric, "score")
    assert catalog.infer_expression_type(entry, allow_unresolved=True).value_type == "boolean"
    assert entry["value"] == "UNRESOLVED:theta_score"


def test_contract_template_matches_backtest_interface():
    entry = threshold.attach({"op": "primitive", "primitive_id": "book_imbalance"}, "score")
    template = threshold.contract_template(entry, "score")
    assert template["entry_program"]["signal"] == entry
    assert template["entry_program"]["warmup_ticks"] == 0
    source = template["parameter_interface"]["theta_score"]["threshold_source"]
    assert source["kind"] == "rolling_prior_100_ticks_quantile"


def test_parameter_table_keys_are_contract_symbol_date():
    table = threshold.parameter_table(
        contract_ids=("q0.70",), symbols=("005930",), date="20260316",
        parameter="score", grid=(0.70,))
    assert table == {"q0.70:005930:20260316": {"theta_score": 0.70}}


def test_compile_succeeds_for_translatable_candidate():
    ok, failed = pipeline.compile_candidates([_candidate(3 * bi + 5)])
    assert len(ok) == 1 and not failed
    assert catalog.infer_expression_type(ok[0].ast, allow_unresolved=True).value_type == "boolean"
    assert ok[0].source.backend == "naive"


def test_compile_records_failure_instead_of_raising():
    ok, failed = pipeline.compile_candidates([_candidate(sympy.exp(bi))])
    assert not ok and len(failed) == 1
    assert failed[0].stage == "translate"
    assert "exp" in failed[0].reason


def test_monotone_variants_collapse_to_one_expression():
    candidates = [_candidate(bi, 0.5), _candidate(3 * bi + 5, 0.6), _candidate(sympy.tanh(bi), 0.4)]
    ok, _failed = pipeline.compile_candidates(candidates)
    assert len(ok) == 1
    assert ok[0].source.in_sample_score == 0.6      # 같은 정규형 중 최고 점수가 남는다


def test_normalize_failure_routes_to_normalize_stage(monkeypatch):
    """`stage` 는 실패를 귀속하는 필드다 — 라우팅 자체를 검사한다.

    현재 입력면에서 `strip_monotone` 은 예외를 던지지 않는 순수 AST 순회라
    이 경로는 도달 불가능에 가깝다(pipeline.py 의 `# pragma: no cover - 방어`
    참고). 그래도 stage 필드가 장식이 되지 않으려면, 억지로 예외를 주입했을
    때 pipeline 이 "normalize" 로 정확히 귀속하고 예외를 삼키는지는 검사돼야
    한다.
    """
    def _boom(expr):
        raise RuntimeError("boom-normalize")
    monkeypatch.setattr(pipeline.normalize, "strip_monotone", _boom)

    ok, failed = pipeline.compile_candidates([_candidate(bi)])  # 예외가 밖으로 새면 여기서 에러로 뜬다

    assert not ok
    assert len(failed) == 1
    assert failed[0].stage == "normalize"
    assert "boom-normalize" in failed[0].reason


def test_check_failure_routes_to_check_stage(monkeypatch):
    """`static_check` 이 실패해도 "check" 로 귀속되는지 — translate 와 헷갈리지 않는지."""
    def _boom(ast, *, allow_unresolved):
        raise pipeline.check.CheckError("boom-check")
    monkeypatch.setattr(pipeline.check, "static_check", _boom)

    ok, failed = pipeline.compile_candidates([_candidate(bi)])  # 예외가 밖으로 새면 여기서 에러로 뜬다

    assert not ok
    assert len(failed) == 1
    assert failed[0].stage == "check"
    assert "boom-check" in failed[0].reason
