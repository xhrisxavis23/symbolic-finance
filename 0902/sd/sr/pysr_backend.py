"""PySR SR 백엔드 — 본 실험용 (ROADMAP.md 2단계 · DESIGN.md D4).

`NaiveBackend`(배관 검증용 템플릿 격자)를 `SRBackend` 프로토콜(`sd/sr/base.py`)을
그대로 지키며 교체한다. `pysr`(juliacall)은 무거운 선택적 의존성이므로 이
모듈 자체는 **`fit()` 안에서만** `import pysr` 한다 — 이 파일을 그냥 import
하는 것(예: 다른 테스트가 `sd.sr` 패키지를 건드릴 때)은 Julia 를 요구하지
않는다.

## 환경변수 배선

`pysr`(juliacall)은 import 시점에 Julia 프로젝트 디렉터리를 만들려 한다.
`juliapkg` 기본값은 conda 환경 안(`$CONDA_PREFIX/julia_env`)인데 이 저장소가
도는 환경에서는 그 디렉터리가 쓰기 불가라 `PermissionError` 로 죽는다(ROADMAP.md
2단계 실측). `sd.config.ensure_pysr_env()` 가 `PYTHON_JULIAPKG_PROJECT` 를
쓰기 가능한 홈 디렉터리 아래로 배선한다 — `fit()` 이 `import pysr` 보다
**먼저** 그것을 부른다. 사용자가 셸에서 환경변수를 기억해야 하는 설계를
피한다.

## 연산자 집합 — `to_catalog.translate` 왕복으로 실측 확정했다

연구계획서 SR 사양(§3 S3)은

    binary_operators = ["+", "-", "*", "/"]
    unary_operators  = ["sqrt", "tanh", "log", "abs", "sign", "square"]

을 제안한다. 하지만 이 백엔드가 낸 후보는 결국 `sd/compile/to_catalog.py` 를
왕복해야 정본 백테스트에 들어간다 — 컴파일러가 받아들이지 않는 연산자를
탐색에 주는 것은 탐색 예산 낭비다. 각 연산자를 실제로 `sympy` 식으로 만들어
`to_catalog.translate()` 에 직접 통과시켜 확인한 결과(이 파일의
`verify_operator_roundtrip()` 이 이 표를 실행 가능한 자기검사로 유지한다):

| 연산자 | 결과 | 실측 근거 |
| --- | --- | --- |
| `+` | ✅ 통과 | `sympy.Add` → `add` 노드 |
| `-` | ✅ 통과 | sympy 에서 `Add(a, Mul(-1, b))` — 부호는 `negate`, 크기는 절댓값이 1이라 손실 없다 |
| `*` | ✅ 통과 | `sympy.Mul`(비수 인자만) → `multiply` 노드 |
| `/` | ✅ 통과 (2차 수정) | sympy 는 `a/b` 를 `Mul(a, Pow(b, -1))` 로 표현한다. 처음에는 |
|      |                  | `to_catalog._walk` 의 `Pow` 분기가 지수 `-1` 을 전부 거부해서(Catalog 에 |
|      |                  | `ratio`/`divide` 연산자가 있는데도 `_walk` 가 그것을 만들어 내는 경로가 |
|      |                  | 없었다) 제외했었다. 사용자 지시로 `Mul` 처리에 분자/분모 분리를 추가해 |
|      |                  | `a/b → ratio(a, b, zero_policy="nan")` 로 옮기도록 고쳤다(`RATIO_ZERO_POLICY` |
|      |                  | 상수와 그 옆 주석 참고) — 이제 변수/변수 나눗셈이 통과한다. **여전히 안 |
|      |                  | 되는 것**: 분자가 상수뿐인 나눗셈(`2/x`, `1/x`) — Catalog 에 산술 안에 |
|      |                  | 수치 리터럴을 놓을 노드가 없어서다. `-1` 이 아닌 음수 지수(`x**-2`)도 |
|      |                  | 여전히 거부된다 |
| `sqrt` | ✅ 통과 | `Pow(x, 1/2)` → `sqrt` 노드. 밑이 음수일 수 있어도 컴파일 자체는 거부하지 |
|        |        | 않는다(Catalog `sqrt` 연산자에 양수 도메인 정적 강제가 없다 — `catalog.py` |
|        |        | 확인) — 실행 시 `nan` 이 나올 뿐이고, 그 행은 기존 파이프라인의 유한값 필터가 |
|        |        | 흡수한다. 이 백엔드의 in-sample 채점(`weighted_r2`)도 같은 방식으로 처리한다 |
| `tanh` | ✅ 통과 | `tanh` 노드 |
| `abs`  | ✅ 통과 | `absolute` 노드 |
| `square` | ✅ 통과 | PySR 의 내장 sympy 매핑(`pysr/export_sympy.py`)이 `square(x)` 를 정확히 |
|          |        | `x**2` 로 낸다 → `Pow(x, 2)` → `multiply(x, x)` |
| `log`  | ❌ 제외 | `to_catalog._walk` 가 명시적으로 거부한다("log 는 log1p 로 바꿔 쓴다. 인수 |
|        |        | 양수성이 보장되지 않는다") — Task 11 결정 그대로, 우회할 방법이 없다 |
| `sign` | ❌ 제외 | `to_catalog._walk` 에 `sympy.sign` 에 대응하는 분기가 없다 |
|        |        | ("번역할 수 없는 노드: sign") — 계획서는 "compare + all/any 로 우회"를 |
|        |        | 제안했지만 그 우회는 **구현돼 있지 않다** |

따라서 이 백엔드가 실제로 PySR 에 넘기는 연산자 집합은(2단계 최종):

    binary_operators = ["+", "-", "*", "/"]
    unary_operators  = ["sqrt", "tanh", "abs", "square"]

## 결정론 vs 병렬성

PySR 은 `deterministic=True` 를 쓰려면 `parallelism="serial"` 이어야 한다
(`pysr.PySRRegressor.fit` 이 그렇지 않으면 즉시 `ValueError` 를 던진다).
연구계획서가 구조 복원율(시드 R 회 중 같은 구조가 나오는 비율)을 1급
선택 기준으로 요구하므로 **기본값은 결정론(`deterministic=True`, 직렬)**
이다. `deterministic=False` 로 인자를 바꾸면 병렬(`multithreading`)로 돈다 —
어느 쪽이었는지는 `PySRBackend.diagnostics["deterministic"]`/`["parallelism"]`
와 `run_slice.py` 가 쓰는 `report.provenance(sr_deterministic=...)` 산출물에
남는다(`Candidate` 자체는 두 백엔드가 공유하는 5-필드 계약이라 여기에
넣지 않는다 — DESIGN.md D4).
"""

from __future__ import annotations

import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sympy

from .. import config as _config
from .base import Candidate, complexity_of, weighted_r2

# 계획서 SR 사양에서 실제로 살아남는 연산자 (위 docstring 표 참고).
# `/` 는 2차 수정으로 되돌아왔다 — `to_catalog._walk` 가 `Mul(a, Pow(b,-1))` 를
# `ratio(a, b, zero_policy="nan")` 로 옮기도록 고쳐졌다. 여전히 분자가 상수뿐인
# 나눗셈(`2/x`)은 안 되고, `-1` 이 아닌 음수 지수도 안 된다 — `to_catalog.py`
# 참고.
BINARY_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")
UNARY_OPERATORS: tuple[str, ...] = ("sqrt", "tanh", "abs", "square")

# 뺀 연산자와 이유 — 산출물·리포트가 "왜 안 줬는가"를 그대로 인용할 수 있게.
EXCLUDED_OPERATORS: dict[str, str] = {
    "log": "to_catalog._walk 가 최외곽이든 깊이든 명시적으로 거부한다 (인수 양수성 미보장)",
    "sign": "to_catalog._walk 에 sympy.sign 에 대응하는 노드가 없다",
}

MAXSIZE_CEILING = 20  # 계획서 §3: 압축이 목적이므로 복잡도 상한을 사전에 고정한다.


class PySRBackend:
    """본 실험용 SR 백엔드. `SRBackend` 프로토콜(`sd/sr/base.py`)을 구현한다."""

    name = "pysr"

    def __init__(self, seed: int = 0, deterministic: bool = True,
                 niterations: int = 40, maxsize: int = MAXSIZE_CEILING,
                 complexity_of_constants: int = 2,
                 verbosity: int = 0, progress: bool = False,
                 output_directory: str | Path | None = None,
                 binary_operators: Sequence[str] | None = None,
                 unary_operators: Sequence[str] | None = None,
                 extra_kwargs: dict[str, Any] | None = None) -> None:
        """`binary_operators`/`unary_operators` — 기본값은 모듈 상수(컴파일러 왕복이
        확인된 집합)다. E0(ROADMAP.md 3단계)는 진입식으로 컴파일하지 않으므로
        `to_catalog` 제약을 받지 않는다 — 계획서 §3 S3 원안의 전체 SR 사양
        (`log`·`sign` 포함, L5 는 `exp` 도)을 그대로 쓸 수 있어야 해서 여기서
        재정의 가능하게 열어 둔다. 이 백엔드가 낸 sympy 식 자체는 연산자가
        무엇이든 그대로이므로(`complexity_of`·`weighted_r2` 는 연산자에 무관하다),
        기본값을 바꾸지 않는 한 기존 호출부·테스트는 전혀 영향받지 않는다.
        """
        if maxsize > MAXSIZE_CEILING:
            raise ValueError(
                f"maxsize={maxsize} 가 계획서 §3 상한({MAXSIZE_CEILING})을 넘는다 — "
                "복잡도 크리프를 막기 위해 사전 등록한 상한이다 (PLAN.md '복잡도 크리프' "
                "위험 항목)")
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.niterations = int(niterations)
        self.maxsize = int(maxsize)
        self.complexity_of_constants = int(complexity_of_constants)
        self.verbosity = int(verbosity)
        self.progress = bool(progress)
        self.output_directory = output_directory
        self.binary_operators = (tuple(binary_operators) if binary_operators is not None
                                 else BINARY_OPERATORS)
        self.unary_operators = (tuple(unary_operators) if unary_operators is not None
                                else UNARY_OPERATORS)
        self.extra_kwargs = dict(extra_kwargs or {})
        # 마지막 fit() 의 진단 기록. 실패/탈락 사유가 여기 남는다 (착수 조건 2).
        self.diagnostics: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]:
        # import pysr 보다 반드시 먼저다 — 그렇지 않으면 juliapkg 가 쓰기 불가한
        # conda 경로에 기본값을 잡아 PermissionError 로 죽는다 (ROADMAP.md 2단계).
        _config.ensure_pysr_env()
        import pysr  # noqa: PLC0415 — 무거운 선택적 의존성, 여기서만 로드한다

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        names = tuple(str(n) for n in feature_names)
        allowed = frozenset(names)

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        n_valid = int(valid.sum())
        if n_valid < 10:
            self.diagnostics = {"backend": self.name, "seed": self.seed,
                               "skipped": True, "reason": "유효 표본이 10개 미만",
                               "n_rows_total": int(len(X)), "n_rows_valid": n_valid}
            return []
        X_fit, y_fit, w_fit = X[valid], y[valid], w[valid]

        parallelism = "serial" if self.deterministic else "multithreading"
        output_directory = str(self.output_directory or tempfile.mkdtemp(prefix="pysr_"))

        model = pysr.PySRRegressor(
            niterations=self.niterations,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            maxsize=self.maxsize,
            complexity_of_constants=self.complexity_of_constants,
            model_selection="best",
            deterministic=self.deterministic,
            parallelism=parallelism,
            random_state=self.seed,
            verbosity=self.verbosity,
            progress=self.progress,
            output_directory=output_directory,
            **self.extra_kwargs,
        )

        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # weights=w_fit — 계획서 §3 S2 ③: "암묵적 리샘플링은 나중에 무엇을
            # 최적화했는지 알 수 없게 만든다". PySR 손실에 명시적으로 넘긴다.
            model.fit(X_fit, y_fit, weights=w_fit, variable_names=list(names))
        elapsed = time.time() - started

        candidates: list[Candidate] = []
        discarded: list[dict[str, Any]] = []
        complexity_mismatches: list[dict[str, Any]] = []
        with np.errstate(all="ignore"):  # sqrt(음수) 등은 nan 으로 조용히 처리한다
            for _, row in model.equations_.iterrows():
                outcome = _extract_candidate(
                    row["sympy_format"], pysr_complexity=int(row["complexity"]),
                    allowed_names=allowed, backend_name=self.name, seed=self.seed,
                    X=X_fit, y=y_fit, w=w_fit, names=names)
                if outcome.candidate is not None:
                    candidates.append(outcome.candidate)
                    if outcome.note is not None:
                        complexity_mismatches.append(outcome.note)
                else:
                    discarded.append(outcome.discard)

        self.diagnostics = {
            "backend": self.name,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "parallelism": parallelism,
            "fit_seconds": elapsed,
            "niterations": self.niterations,
            "maxsize": self.maxsize,
            "complexity_of_constants": self.complexity_of_constants,
            "binary_operators": list(self.binary_operators),
            "unary_operators": list(self.unary_operators),
            "excluded_operators": dict(EXCLUDED_OPERATORS),
            "n_rows_total": int(len(X)),
            "n_rows_valid": n_valid,
            "n_equations_from_pysr": int(len(model.equations_)),
            "n_candidates_kept": len(candidates),
            "n_discarded_free_symbols": sum(
                1 for d in discarded if d["reason"] == "free_symbols_outside_feature_names"),
            "n_discarded_other": sum(
                1 for d in discarded if d["reason"] != "free_symbols_outside_feature_names"),
            "discarded": discarded,
            "complexity_mismatches": complexity_mismatches,
            "output_directory": output_directory,
        }
        if discarded:
            print(f"[pysr] {len(discarded)}개 후보를 fit() 단계에서 버렸다:")
            for item in discarded[:10]:
                extra = item.get("unknown_symbols") or item.get("error", "")
                print(f"  - {item['reason']}: {item.get('expr')!r} {extra}")
        if complexity_mismatches:
            print(f"[pysr] complexity_of 재계산이 PySR 값과 다른 후보 "
                  f"{len(complexity_mismatches)}개 — Pareto 비교에는 재계산 값만 쓴다")

        return sorted(candidates, key=lambda c: c.complexity)


@dataclass(frozen=True)
class _ExtractOutcome:
    candidate: Candidate | None
    discard: dict[str, Any] | None
    note: dict[str, Any] | None


def _extract_candidate(sympy_expr: Any, *, pysr_complexity: int,
                        allowed_names: frozenset[str], backend_name: str, seed: int,
                        X: np.ndarray, y: np.ndarray, w: np.ndarray,
                        names: Sequence[str]) -> _ExtractOutcome:
    """PySR Pareto front 한 행을 `Candidate` 로 바꾸거나 사유와 함께 버린다.

    **착수 조건 2 (명시 검증).** `NaiveBackend` 는 템플릿에 없는 심볼이 들어와도
    `lambdify` 가 object dtype 을 내고 `float()` 캐스트가 `TypeError` 를 던지는
    것을 `_score` 의 `except Exception` 이 우연히 삼켜서 결과적으로 걸러질
    뿐이다 — **우연한 가드이지 명시적 검증이 아니다.** 여기서는 그 가드에
    기대지 않고, `expr.free_symbols` 가 `allowed_names` 의 부분집합인지
    **직접** 비교해서 아니면 즉시 버리고 사유를 남긴다.

    이 함수는 순수 함수다(PySR 을 몰라도 된다) — `sympy_expr` 은 `pysr` 의
    `equations_["sympy_format"]` 한 셀이지만, 테스트는 이 함수를 PySR 없이
    직접 부를 수 있다.
    """
    expr = sympy.sympify(sympy_expr)
    used = {str(s) for s in expr.free_symbols}
    if not used <= set(allowed_names):
        unknown = sorted(used - set(allowed_names))
        return _ExtractOutcome(None, {
            "reason": "free_symbols_outside_feature_names",
            "expr": str(expr), "unknown_symbols": unknown}, None)

    try:
        function = sympy.lambdify([sympy.Symbol(n) for n in names], expr, "numpy")
        raw = np.asarray(function(*[X[:, j] for j in range(len(names))]), dtype=float)
    except Exception as error:                                  # pragma: no cover - 방어
        return _ExtractOutcome(None, {
            "reason": "lambdify_or_eval_failed", "expr": str(expr),
            "error": f"{type(error).__name__}: {error}"}, None)

    raw = np.broadcast_to(raw, y.shape).astype(float)

    # `NaiveBackend._score` 는 raw 를 기울기·절편으로 다시 맞춰(lstsq) 채점한다
    # (부호에 맹목이다 — `-raw` 도 똑같이 잘 맞는다). PySR 은 상수를 스스로
    # 맞춰 왔으므로 그 결함을 물려받지 않는다: 수식이 낸 값을 **그대로** 쓴다.
    score = weighted_r2(y, raw, w)
    if not np.isfinite(score):
        return _ExtractOutcome(None, {
            "reason": "non_finite_in_sample_score", "expr": str(expr)}, None)

    # `complexity` — PySR 이 주는 값이 아니라 `base.complexity_of` 로 재계산한다.
    # 두 백엔드가 같은 자로 재야 Pareto 비교가 성립한다 (DESIGN.md D4).
    recomputed_complexity = complexity_of(expr)
    candidate = Candidate(expr=expr, complexity=recomputed_complexity,
                          in_sample_score=float(score), backend=backend_name,
                          seed=int(seed))
    note = None
    if recomputed_complexity != pysr_complexity:
        note = {"expr": str(expr), "pysr_complexity": int(pysr_complexity),
                "recomputed_complexity": recomputed_complexity}
    return _ExtractOutcome(candidate, None, note)


def verify_operator_roundtrip() -> dict[str, bool]:
    """모듈 docstring 의 연산자 표가 실제로 참인지 `to_catalog.translate` 로 직접
    확인한다. `pysr` 없이 돈다 — Catalog 가 나중에 바뀌면(예: `divide` 지원 추가)
    여기서 먼저 드러난다."""
    from ..compile import to_catalog  # noqa: PLC0415 — 순환 import 회피

    a = sympy.Symbol("book_imbalance")
    b = sympy.Symbol("queue_imbalance_best")
    samples: dict[str, sympy.Expr] = {
        "+": a + b, "-": a - b, "*": a * b, "/": a / b,
        "sqrt": sympy.sqrt(a), "tanh": sympy.tanh(a), "abs": sympy.Abs(a),
        "square": a**2, "log": sympy.log(sympy.Abs(a) + 1), "sign": sympy.sign(a),
    }
    result: dict[str, bool] = {}
    for op, expr in samples.items():
        try:
            to_catalog.translate(expr)
            result[op] = True
        except to_catalog.TranslationError:
            result[op] = False
    return result
