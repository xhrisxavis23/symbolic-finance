"""E0 오케스트레이션. 법칙 하나를 끝까지 돌린다: 데이터 조립 → 종목 단위
표본외 분할 → 교사(무차원·raw 두 벌, **적합 종목으로만**) → 네 트랙(본트랙 +
ablation 3종, SR 적합도 적합 종목으로만) → 후보 전원을 **선택 종목**에서
채점·판정 → 게이트 집계.

`sd.manifold.select`·`sd.sr.pysr_backend.PySRBackend`·`sd.e0.teacher.ScalarTeacher`
·`sd.e0.criteria`·`sd.e0.ablation`·`sd.e0.targets`·`sd.e0.split` 를 그대로
엮는다 — 여기서 새 산술을 만들지 않는다.

## 표본외 후보 선택 (PREREG-E0-V2.md §1-1)

v1(ROADMAP.md 3단계)은 `argmax(in_sample_score)` 로 `primary_index` 를 뽑고
`_judge` 도 적합에 쓴 그 `X` 로 돌렸다 — 교사 R² 가 0.05~0.58 인 데이터에서
"적합 데이터에 제일 잘 맞는 후보"를 뽑으면 노이즈를 제일 잘 외운 후보가
뽑힌다(v1 이 실제로 관측한 탈락 사유: 정답 수식에 `+0.089` 오프셋 하나가
붙어 대칭을 깨는 대신 근소하게 더 잘 맞았다). 이 파일은 그 규칙만 바꾼다:

- `sd.e0.split.split_symbols` 로 종목을 적합/선택 두 서로소 집합으로
  나눈다(층화 유지, `seed` 결정론).
- SR 적합(`backend.fit`)은 **적합 종목**의 데이터로만 한다.
- `primary_index`·`_judge` 는 **선택 종목**의 데이터로 계산한다
  (`_score_and_judge`).

바뀌지 않은 것: `sd.e0.criteria` 의 판정 함수·밴드(한 바이트도), 게이트 규칙,
심사 대상은 `primary` 후보 하나, seed 0, 층화 구조.

## 교사 시간 윈도우 (PREREG-E0-V2.md §1-2, `run_law(..., teacher_window=...)`)

`sd.e0.teacher.ScalarDeepLOB` 처럼 시간 윈도우가 필요한 교사를 주입하면
`_maybe_window` 가 `sd.teacher.window.make_causal_windows` 를 태운다 — **교사
입력에만** 적용되고 SR 이 보는 X 는 항상 원본이다. 윈도우는 `_manifold_pick`
(표집)보다 먼저, 종목별 연속 행렬(`fit_dataset`/`select_dataset` 전체)에
씌운 뒤 그 결과를 표집이 고른 행 인덱스로 슬라이스한다 — 표집 후 행렬에
씌우면 "시간 윈도우"라는 말 자체가 거짓이 된다(`sd/teacher/window.py`).
`teacher_window=1`(기본)은 이 파일 안 `_maybe_window` 산술상 완전한
항등 변환이라 `ScalarTeacher` 경로의 동작은 이 변경으로 전혀 안 바뀐다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .. import manifold
from ..sr.base import Candidate, score_candidate, weighted_r2
from . import ablation, criteria, targets
from .teacher import ScalarTeacher, ScalarTeacherProtocol

# 계획서 §3 S3 원안 SR 사양 그대로(`sqrt`·`tanh`·`log`·`abs`·`sign`·`square`).
# E0 는 `to_catalog` 컴파일을 타지 않으므로 `sd.sr.pysr_backend` 의 컴파일러용
# 제한 집합(`log`·`sign` 제외 — 컴파일러가 거부해서 뺐다)을 쓰지 않고 원안
# 전체를 그대로 쓴다. 실측 확인: `sign`·`log` 둘 다 PySR(Julia
# SymbolicRegression.jl) 자체는 문제없이 받는다(2026-09-04, niterations=6
# 스모크 fit 성공) — `to_catalog` 가 거부하는 것과 PySR 이 못 내는 것은
# 별개다. `exp` 만 원안이 명시적으로 금지하고(외삽 폭발) L5(Hawkes) 에서
# 예외를 허용한다.
DEFAULT_UNARY: tuple[str, ...] = ("sqrt", "square", "abs", "tanh", "log", "sign")
L5_UNARY: tuple[str, ...] = DEFAULT_UNARY + ("exp",)
BINARY_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")

TRACKS: tuple[str, ...] = ("main", "no_distillation", "no_dimensionless", "uniform_off_manifold")


# 구조("근사 아핀"·"단조 홀함수" 등)만 보는 판정은 적합도에 맹목이다 — 원본
# 변수를 그대로 돌려주는 항등함수(`f(x)=x`)는 언제나 완벽하게 아핀이지만
# 그것이 좋은 근사라는 뜻은 아니다. 2026-09-04 첫 전체 실행에서 L2 의
# `any_candidate_recovered=True` 가 정확히 이 사례(cx=1, `in_sample_score=-850`,
# 즉 예측이 평균보다도 나쁘다)로만 성립한 것을 발견했다 — 그래서 "복원"으로
# 세려면 구조 판정과 별개로 **평균보다는 나은 적합**(R² > 0, 통계적으로 이미
# 정의된 자연스러운 문턱)도 요구한다. 모든 법칙에 동일하게 적용한다 — 이미
# 실패로 잡힌 L2 를 더 실패로 만들 뿐, 어떤 법칙도 이 변경으로 새로 통과하지
# 않는다(엄격화이지 완화가 아니다).
#
# v2: 이 문턱이 보는 점수가 `candidate.in_sample_score`(적합 데이터)에서
# `out_of_sample_score`(선택 데이터, § 아래)로 바뀌었다 — 판정 전체를 선택
# 종목 기준으로 옮기는 이상, "평균보다는 나은 적합"도 같은 데이터에서 재야
# 한다. 문턱 값(0.0)은 그대로다.
MIN_MEANINGFUL_SCORE = 0.0

# 선택 종목 쪽 유효 행이 이 미만이면 판정 자체가 불안정하다고 보고 멈춘다
# (조용히 전부 `ambiguous` 로 새는 대신 사유를 명시한다). `criteria.py` 안의
# 개별 문턱(대부분 `< 10`)보다 여유를 둔 값 — 표본이 10개를 겨우 넘겨
# 통과하는 것과 판정이 실제로 안정적인 것은 다르다.
MIN_SELECT_ROWS = 30


@dataclass
class TrackResult:
    name: str
    n_rows: int
    candidates: list[Candidate]
    diagnostics: dict[str, Any]
    judged: list[dict[str, Any]]           # 후보와 판정을 나란히 (같은 인덱스)
    primary_index: int | None              # out_of_sample_score 최고 후보의 인덱스
    fit_seconds: float
    out_of_sample_scores: list[float] = field(default_factory=list)  # 후보와 같은 인덱스
    n_select_rows: int = 0                 # 채점·판정에 쓴 선택 종목 행 수
    skipped_reason: str | None = None

    def _well_fit(self, index: int) -> bool:
        return self.out_of_sample_scores[index] > MIN_MEANINGFUL_SCORE

    @property
    def primary_verdict(self) -> dict | None:
        if self.primary_index is None:
            return None
        return self.judged[self.primary_index]

    @property
    def recovered(self) -> bool:
        if self.primary_index is None:
            return False
        verdict = self.primary_verdict
        return bool(verdict and verdict.get("passed") and not verdict.get("ambiguous")
                   and self._well_fit(self.primary_index))

    @property
    def recovered_indices(self) -> list[int]:
        """구조 판정 통과 **and** 평균보다는 나은 적합(§ `MIN_MEANINGFUL_SCORE`,
        선택 종목 기준) 둘 다인 후보의 인덱스. `any_candidate_recovered`·
        `fraction_of_front_recovered` 가 공유하는 단일 정의 — 두 곳에 따로
        두면 조용히 갈라진다."""
        return [i for i, j in enumerate(self.judged)
                if j.get("passed") and not j.get("ambiguous") and self._well_fit(i)]

    @property
    def any_candidate_recovered(self) -> bool:
        return len(self.recovered_indices) > 0

    @property
    def fraction_of_front_recovered(self) -> float | None:
        if not self.judged:
            return None
        return len(self.recovered_indices) / len(self.judged)


@dataclass
class LawResult:
    law: str
    n_rows_total: int
    symbols_used: tuple[str, ...]                  # 적합 종목 중 실제 로드된 것
    symbols_skipped: dict[str, str]
    teacher_dimless_diag: dict[str, Any]
    teacher_raw_diag: dict[str, Any]
    tracks: dict[str, TrackResult] = field(default_factory=dict)
    select_symbols_used: tuple[str, ...] = field(default_factory=tuple)
    select_symbols_skipped: dict[str, str] = field(default_factory=dict)

    @property
    def recovered(self) -> bool:
        return self.tracks["main"].recovered if "main" in self.tracks else False


def _judge(law: str, expr, names: Sequence[str], X: np.ndarray) -> dict:
    if law == "L1":
        return criteria.judge_l1(expr, names, X)
    if law == "L2":
        return criteria.judge_l2(expr, names, X)
    if law == "L3":
        return criteria.judge_l3(expr, names, X, volume_name=names[0], sigma_name=names[1])
    if law == "L4":
        return criteria.judge_l4(expr, names, X)
    if law == "L5":
        return criteria.judge_l5(expr, names, X, dt_name=names[0])
    raise ValueError(f"모르는 법칙: {law}")


def _build_backend(law: str, seed: int, niterations: int, maxsize: int):
    from ..sr.pysr_backend import PySRBackend  # 여기서만 import — juliacall 을 요구한다

    unary = L5_UNARY if law == "L5" else DEFAULT_UNARY
    return PySRBackend(seed=seed, deterministic=True, niterations=niterations,
                       maxsize=maxsize, binary_operators=BINARY_OPERATORS,
                       unary_operators=unary)


def _score_and_judge(law: str, name: str, candidates: list[Candidate], names: Sequence[str],
                     X_select: np.ndarray, y_select: np.ndarray, w_select: np.ndarray, *,
                     fit_seconds: float, n_fit_rows: int, diagnostics: dict[str, Any]) -> TrackResult:
    """SR 이 이미 낸 `candidates`(적합 종목에서 나왔다)를 **선택 종목**
    `(X_select, y_select, w_select)` 에서 채점·판정한다.

    `backend.fit()` 을 몰라도 되는 순수 함수다 — PySR 없이 단위 테스트가
    이 함수를 직접 부른다(`tests/test_e0_runner.py`)."""
    out_of_sample_scores = [score_candidate(c, names, X_select, y_select, w_select)
                            for c in candidates]
    judged = [_judge(law, c.expr, names, X_select) for c in candidates]
    primary_index = int(np.argmax(out_of_sample_scores)) if candidates else None
    return TrackResult(name=name, n_rows=n_fit_rows, candidates=candidates,
                       diagnostics=diagnostics, judged=judged, primary_index=primary_index,
                       fit_seconds=fit_seconds, out_of_sample_scores=out_of_sample_scores,
                       n_select_rows=int(len(y_select)))


def _fit_track(law: str, name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               names: Sequence[str], X_select: np.ndarray, y_select: np.ndarray,
               w_select: np.ndarray, *, seed: int, niterations: int, maxsize: int) -> TrackResult:
    backend = _build_backend(law, seed, niterations, maxsize)
    started = time.time()
    candidates = backend.fit(X, y, w, names)          # 적합 종목만 본다
    elapsed = time.time() - started
    return _score_and_judge(law, name, candidates, names, X_select, y_select, w_select,
                            fit_seconds=elapsed, n_fit_rows=int(len(y)),
                            diagnostics=dict(backend.diagnostics))


def _maybe_window(X: np.ndarray, session_ids: np.ndarray | None, window: int) -> np.ndarray:
    """`window<=1` 이면 `X` 를 그대로 돌려준다 — 기본 교사(`ScalarTeacher`,
    `window=1`)의 산출물이 이 함수가 생기기 전과 바이트 단위로 같아야 한다.
    이는 우연이 아니라 산술로 보장된다: `mask=True`(선택 후보가 될 수 있는
    유일한 조건, `sd.manifold.select` 참고)인 행은 정의상 이미 유한하므로
    `window=1` 에서 켜지는 세션-내 forward-fill(비유한 값 대체)이 그런 행의
    값을 바꿀 일이 없다 — 바뀌는 값은 애초에 선택될 수 없는(mask=False) 행의
    값뿐이다.

    `window>1` 일 때만 `sd.teacher.window.make_causal_windows` 를 태운다.
    **호출자 책임**: 표집(`_manifold_pick`) **이전**의 연속 행렬에 대해서만
    이 함수를 불러야 한다(PREREG-E0-V2.md §1-2) — `run_law` 는 실제로 그
    순서를 지킨다(아래에서 `_manifold_pick` 보다 먼저 부른다)."""
    if window <= 1:
        return X
    from ..teacher.window import make_causal_windows  # 필요할 때만 import(window>1 인 실행만 쓴다)
    return make_causal_windows(X, window=window, session_ids=session_ids)


def _manifold_pick(X: np.ndarray, mask: np.ndarray, *, max_samples: int, seed: int,
                   min_rows: int | None = None, what: str = "") -> tuple[np.ndarray, np.ndarray]:
    """`manifold.select` 를 부르고 on-manifold 행·가중치만 돌려준다.

    `min_rows` 가 주어지면 그 미만일 때 명시적으로 멈춘다 — 선택 종목 쪽이
    너무 작아 판정이 불안정해지는 경우를 조용히 삼키지 않는다(§
    `MIN_SELECT_ROWS`, PREREG-E0-V2.md §1-1 "설계 판단이 필요한 지점")."""
    selection = manifold.select(X, mask, max_samples=max_samples, seed=seed)
    rows = selection.index[selection.on_manifold]
    weight = selection.weight[selection.on_manifold]
    if min_rows is not None and len(rows) < min_rows:
        raise ValueError(
            f"{what}: on-manifold 행이 {len(rows)}개뿐이다(최소 {min_rows}) — "
            "선택 종목 표본이 너무 작아 채점·판정이 불안정하다. --per-stratum 을 "
            "올리거나 sd.e0.split.DEFAULT_FIT_FRACTION 을 낮춰라.")
    return rows, weight


def run_law(law: str, fit_symbols: Sequence[str], select_symbols: Sequence[str], date: str, *,
           seed: int = 0, bottleneck: int = 2, epochs: int = 300,
           sr_niterations: int = 25, sr_maxsize: int = 20,
           max_manifold_samples: int = 4000,
           teacher_cls: Callable[..., ScalarTeacherProtocol] = ScalarTeacher,
           teacher_window: int = 1) -> LawResult:
    """`fit_symbols`·`select_symbols` 는 서로소여야 한다(호출부가 보증한다 —
    보통 `sd.e0.split.split_symbols` 의 산출물, 법칙 5개가 모두 같은 분할을
    공유한다: 게이트 판정이 법칙마다 다른 표본외 기준으로 나오면 비교가
    안 선다). `teacher_cls` 는 `sd.e0.teacher.ScalarTeacher` 와 같은 계약
    (`__init__(n_features, bottleneck, seed)` → `.fit(X,y,weight,epochs)` →
    `.predict(X)`)을 지키는 교사 클래스면 무엇이든 받는다 — 여기서 구체
    교사를 고르지 않는다(PREREG-E0-V2.md 1-2 는 별도 작업).

    `teacher_window`(기본 1) 는 교사에게 넘길 입력에 `sd.teacher.window.
    make_causal_windows` 를 몇 스텝짜리로 씌울지다. **SR 이 보는 X 는 이 값과
    무관하게 항상 원본(윈도우 없는) feature 공간이다** — 윈도우는 오직 교사
    (증류 타깃을 만드는 신경망)의 입력에만 적용된다. `teacher_window=1`(기본,
    `ScalarTeacher` 용)은 `_maybe_window` 산술상 완전한 항등 변환이라 기존
    동작을 바이트 단위로 보존한다. `sd.e0.teacher.ScalarDeepLOB` 처럼 시간
    윈도우가 필요한 교사를 쓰려면 `teacher_window>1` 을 같이 넘겨야 한다 —
    윈도우는 **표집 전, 종목별 연속 행렬**(`fit_dataset.X_dimless` 전체)에
    씌운 뒤에야 `_manifold_pick` 이 고른 행 인덱스로 슬라이스한다(아래 코드
    순서가 그 순서를 그대로 지킨다) — 표집 후 행렬에 씌우면 "시간 윈도우"라는
    말 자체가 거짓이 된다(`sd/teacher/window.py` 모듈 docstring, PREREG-E0-V2.md
    §1-2)."""
    if set(fit_symbols) & set(select_symbols):
        raise ValueError("fit_symbols·select_symbols 가 겹친다 — 표본 외 분할이 아니다")

    fit_dataset = targets.assemble(law, fit_symbols, date)
    select_dataset = targets.assemble(law, select_symbols, date)
    names = fit_dataset.names_dimless
    names_raw = fit_dataset.names_raw

    # 교사 전용 입력 — 표집 **전**, 종목별 연속 행렬에 윈도우를 씌운다(위
    # docstring 참고). SR 이 쓰는 `fit_dataset.X_dimless`/`X_raw` 자체는 아래에서
    # 전혀 바뀌지 않는다 — 이 두 변수만 교사 fit/predict 호출에 쓰인다.
    fit_X_dimless_teacher = _maybe_window(fit_dataset.X_dimless, fit_dataset.symbol_ids,
                                          teacher_window)
    fit_X_raw_teacher = _maybe_window(fit_dataset.X_raw, fit_dataset.symbol_ids, teacher_window)
    select_X_dimless_teacher = _maybe_window(select_dataset.X_dimless, select_dataset.symbol_ids,
                                             teacher_window)
    select_X_raw_teacher = _maybe_window(select_dataset.X_raw, select_dataset.symbol_ids,
                                         teacher_window)

    # -- 무차원 트랙 교사 (적합 종목으로만) -----------------------------------
    fit_rows, fit_weight = _manifold_pick(
        fit_dataset.X_dimless, fit_dataset.mask, max_samples=max_manifold_samples, seed=seed,
        what=f"{law}/적합/dimless")
    teacher = teacher_cls(n_features=fit_dataset.X_dimless.shape[1], bottleneck=bottleneck,
                          seed=seed).fit(fit_X_dimless_teacher[fit_rows], fit_dataset.y_dimless[fit_rows],
                                        fit_weight, epochs=epochs)
    teacher_pred = teacher.predict(fit_X_dimless_teacher[fit_rows])

    # -- raw 트랙 교사 (ablation "무차원화 없이 증류" 전용, 적합 종목으로만) --
    fit_rows_raw, fit_weight_raw = _manifold_pick(
        fit_dataset.X_raw, fit_dataset.mask, max_samples=max_manifold_samples, seed=seed,
        what=f"{law}/적합/raw")
    teacher_raw = teacher_cls(n_features=fit_dataset.X_raw.shape[1], bottleneck=bottleneck,
                              seed=seed).fit(fit_X_raw_teacher[fit_rows_raw],
                                            fit_dataset.y_raw[fit_rows_raw],
                                            fit_weight_raw, epochs=epochs)
    teacher_raw_pred = teacher_raw.predict(fit_X_raw_teacher[fit_rows_raw])

    # -- 선택 종목 데이터: 후보 채점·판정 전용. on-manifold 로 다시 고른다 —
    # 훈련 쪽과 같은 신뢰반경 철학(믿음의 반경 밖 극단치가 percentile 기반
    # 판정·R² 를 왜곡하지 않게)을 그대로 적용한다. -----------------------
    select_rows, select_weight = _manifold_pick(
        select_dataset.X_dimless, select_dataset.mask, max_samples=max_manifold_samples,
        seed=seed, min_rows=MIN_SELECT_ROWS, what=f"{law}/선택/dimless")
    select_rows_raw, select_weight_raw = _manifold_pick(
        select_dataset.X_raw, select_dataset.mask, max_samples=max_manifold_samples,
        seed=seed, min_rows=MIN_SELECT_ROWS, what=f"{law}/선택/raw")

    select_X_dimless = select_dataset.X_dimless[select_rows]
    select_y_dimless = select_dataset.y_dimless[select_rows]
    # 교사 입력만 윈도우 버전으로 — SR 채점용 select_X_dimless(위 줄)는 그대로
    # 원본 feature 공간이다(`names` 열 수와 일치해야 한다).
    select_teacher_pred = teacher.predict(select_X_dimless_teacher[select_rows])  # main·uniform_off_manifold 채점용
    select_X_raw = select_dataset.X_raw[select_rows_raw]
    select_teacher_raw_pred = teacher_raw.predict(select_X_raw_teacher[select_rows_raw])  # no_dimensionless 채점용

    teacher_diag = {
        "track": "dimless", "n_fit_rows": int(len(fit_rows)),
        "r2_vs_real_y": weighted_r2(fit_dataset.y_dimless[fit_rows], teacher_pred, fit_weight),
        "correlation_vs_real_y": float(np.corrcoef(teacher_pred, fit_dataset.y_dimless[fit_rows])[0, 1])
                                 if np.std(teacher_pred) > 0 else 0.0,
        # 참고용 — 게이트 어떤 판정에도 쓰지 않는다. 교사 자체가 held-out
        # 종목에서도 실측 y 를 설명하는지 보여주는 진단일 뿐이다.
        "n_select_rows": int(len(select_rows)),
        "r2_vs_real_y_select_symbols": weighted_r2(select_y_dimless, select_teacher_pred, select_weight),
        "bottleneck": bottleneck, "epochs": epochs,
    }
    teacher_raw_diag = {
        "track": "raw", "n_fit_rows": int(len(fit_rows_raw)),
        "r2_vs_real_y": weighted_r2(fit_dataset.y_raw[fit_rows_raw], teacher_raw_pred, fit_weight_raw),
        "correlation_vs_real_y": float(np.corrcoef(teacher_raw_pred,
                                                    fit_dataset.y_raw[fit_rows_raw])[0, 1])
                                 if np.std(teacher_raw_pred) > 0 else 0.0,
        "n_select_rows": int(len(select_rows_raw)),
        "r2_vs_real_y_select_symbols": weighted_r2(select_dataset.y_raw[select_rows_raw],
                                                   select_teacher_raw_pred, select_weight_raw),
        "bottleneck": bottleneck, "epochs": epochs,
    }

    result = LawResult(law=law, n_rows_total=fit_dataset.n_rows_total + select_dataset.n_rows_total,
                       symbols_used=fit_dataset.symbols_used,
                       symbols_skipped=fit_dataset.symbols_skipped,
                       select_symbols_used=select_dataset.symbols_used,
                       select_symbols_skipped=select_dataset.symbols_skipped,
                       teacher_dimless_diag=teacher_diag, teacher_raw_diag=teacher_raw_diag)

    common = dict(seed=seed, niterations=sr_niterations, maxsize=sr_maxsize)

    # main: on-manifold, 무차원, 증류(교사 출력). 채점·판정은 선택 종목에서.
    result.tracks["main"] = _fit_track(
        law, "main", fit_dataset.X_dimless[fit_rows], teacher_pred, fit_weight, names,
        select_X_dimless, select_teacher_pred, select_weight, **common)

    # no_distillation: 같은 표집·같은 X, 타깃만 실측 y 로.
    result.tracks["no_distillation"] = _fit_track(
        law, "no_distillation", fit_dataset.X_dimless[fit_rows], fit_dataset.y_dimless[fit_rows],
        fit_weight, names, select_X_dimless, select_y_dimless, select_weight, **common)

    # no_dimensionless: raw 공간에서 표집·raw 교사로 증류.
    result.tracks["no_dimensionless"] = _fit_track(
        law, "no_dimensionless", fit_dataset.X_raw[fit_rows_raw], teacher_raw_pred,
        fit_weight_raw, names_raw, select_X_raw, select_teacher_raw_pred, select_weight_raw,
        **common)

    # uniform_off_manifold: 같은 무차원 X 의 열별 범위(적합 종목 기준)에서
    # 균등 박스 합성, 같은(무차원) 교사에 질의 — 표집 절차만 바꾼다. 채점은
    # main 과 같은 선택 종목 실측 데이터에서 한다(이 ablation 이 진짜로
    # 묻는 것 — "균등 합성으로 적합해도 실제 held-out 종목에서 통하는가").
    synthetic_X = ablation.uniform_box_sample(fit_dataset.X_dimless[fit_rows], n=len(fit_rows),
                                              seed=seed)
    # 합성 표본은 시간상 이웃이 없다(열마다 독립 균등분포에서 뽑은 점) — 각
    # 행을 자기 혼자만의 세션으로 취급해 윈도우를 씌운다. 세션 길이가 1이면
    # `make_causal_windows` 의 패딩 규칙("세션 첫 행을 반복")이 그 행 자신을
    # window 번 반복하는 것으로 자연히 축약된다 — 다른 합성 행의 값이
    # 섞여 들어올 길이 없다.
    synthetic_X_teacher = _maybe_window(synthetic_X, np.arange(len(synthetic_X)), teacher_window)
    synthetic_y = teacher.predict(synthetic_X_teacher)
    result.tracks["uniform_off_manifold"] = _fit_track(
        law, "uniform_off_manifold", synthetic_X, synthetic_y,
        np.ones(len(synthetic_X)), names, select_X_dimless, select_teacher_pred, select_weight,
        **common)

    return result
