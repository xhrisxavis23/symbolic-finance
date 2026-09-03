"""컴파일 파이프라인. 계획서 §3 S5 ①~④.

후보 하나가 실패해도 예외로 죽지 않는다. **실패율 자체가 산출물**이다
(계획서 F7 — Pareto 후보의 컴파일 성공률이 20% 미만이면 중단).

정규형 키는 `normalize.normal_form()` 을 그대로 부른다 — 여기서 같은 산술을
`sympy.simplify`/`sympy.srepr` 로 다시 짜지 않는다. 같은 산술이 두 곳에 있으면
한쪽만 고쳐진 채 조용히 갈라진다: `vendor/framework/metrics.py` 의 분모가
`ledger._metrics`·`diagnose._economic_replay`·청산 후보 채점부 세 곳에 따로
있다가 계약 선택이 "더 많이 잃은 쪽"을 고르는 사고로 이어졌다 (그 파일 docstring
참고). `normal_form` 은 내부에서 `strip_monotone` 을 다시 호출하므로 원본
후보(`candidate.expr`)를 넘긴다 — 이미 벗긴 `stripped` 를 넘기면 이중으로
벗기게 되어 다른 값이 나올 위험이 있다 (단, `strip_monotone` 은 멱등이라 이
경우엔 실제로 같은 값이 나온다. 그래도 원본을 넘기는 쪽이 계약이 하나뿐이라
더 안전하다).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..sr.base import Candidate
from . import check, normalize, threshold, to_catalog

PARAMETER = "score"


@dataclass(frozen=True)
class EntryExpression:
    ast: dict
    thresholds: dict[str, str]
    source: Candidate
    normal_form: str
    shells: tuple[str, ...]


@dataclass(frozen=True)
class CompileFailure:
    candidate: Candidate
    stage: str
    reason: str


def compile_candidates(
        candidates: list[Candidate]) -> tuple[list[EntryExpression], list[CompileFailure]]:
    succeeded: dict[str, EntryExpression] = {}
    failed: list[CompileFailure] = []

    for candidate in candidates:
        try:
            stripped, shells = normalize.strip_monotone(candidate.expr)
        except Exception as error:                       # pragma: no cover - 방어
            failed.append(CompileFailure(candidate, "normalize", str(error)))
            continue

        try:
            numeric = to_catalog.translate(stripped)
        except to_catalog.TranslationError as error:
            failed.append(CompileFailure(candidate, "translate", str(error)))
            continue

        entry = threshold.attach(numeric, PARAMETER)
        try:
            check.static_check(entry, allow_unresolved=True)
        except check.CheckError as error:
            failed.append(CompileFailure(candidate, "check", str(error)))
            continue

        # Task 10 의 normal_form() 을 그대로 쓴다 — 원본 후보를 넘긴다.
        # normal_form 이 내부에서 strip_monotone 을 다시 부르므로 위에서 이미
        # 벗긴 stripped 와 동일한 값에 도달한다.
        key = normalize.normal_form(candidate.expr)
        existing = succeeded.get(key)
        if existing is None or candidate.in_sample_score > existing.source.in_sample_score:
            succeeded[key] = EntryExpression(
                ast=entry,
                thresholds={f"theta_{PARAMETER}": threshold.THRESHOLD_KIND},
                source=candidate, normal_form=key, shells=tuple(shells))

    ordered = sorted(succeeded.values(), key=lambda e: (e.source.complexity, e.normal_form))
    return ordered, failed
