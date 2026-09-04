"""E0 오케스트레이션. 법칙 하나를 끝까지 돌린다: 데이터 조립 → 교사(무차원·raw
두 벌) → 네 트랙(본트랙 + ablation 3종) 표집·SR → 후보 전원 판정 → 게이트 집계.

`sd.manifold.select`·`sd.sr.pysr_backend.PySRBackend`·`sd.e0.teacher.ScalarTeacher`
·`sd.e0.criteria`·`sd.e0.ablation`·`sd.e0.targets` 를 그대로 엮는다 — 여기서
새 산술을 만들지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .. import manifold
from ..sr.base import Candidate, weighted_r2
from . import ablation, criteria, targets
from .teacher import ScalarTeacher

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
MIN_MEANINGFUL_SCORE = 0.0


@dataclass
class TrackResult:
    name: str
    n_rows: int
    candidates: list[Candidate]
    diagnostics: dict[str, Any]
    judged: list[dict[str, Any]]           # 후보와 판정을 나란히 (같은 인덱스)
    primary_index: int | None              # in_sample_score 최고 후보의 인덱스
    fit_seconds: float
    skipped_reason: str | None = None

    def _well_fit(self, index: int) -> bool:
        return self.candidates[index].in_sample_score > MIN_MEANINGFUL_SCORE

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
        """구조 판정 통과 **and** 평균보다는 나은 적합(§ `MIN_MEANINGFUL_SCORE`)
        둘 다인 후보의 인덱스. `any_candidate_recovered`·`fraction_of_front_recovered`
        가 공유하는 단일 정의 — 두 곳에 따로 두면 조용히 갈라진다."""
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
    symbols_used: tuple[str, ...]
    symbols_skipped: dict[str, str]
    teacher_dimless_diag: dict[str, Any]
    teacher_raw_diag: dict[str, Any]
    tracks: dict[str, TrackResult] = field(default_factory=dict)

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


def _fit_track(law: str, name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               names: Sequence[str], *, seed: int, niterations: int, maxsize: int) -> TrackResult:
    backend = _build_backend(law, seed, niterations, maxsize)
    started = time.time()
    candidates = backend.fit(X, y, w, names)
    elapsed = time.time() - started
    judged = [_judge(law, c.expr, names, X) for c in candidates]
    primary_index = (int(np.argmax([c.in_sample_score for c in candidates]))
                     if candidates else None)
    return TrackResult(name=name, n_rows=int(len(y)), candidates=candidates,
                       diagnostics=dict(backend.diagnostics), judged=judged,
                       primary_index=primary_index, fit_seconds=elapsed)


def run_law(law: str, symbols: Sequence[str], date: str, *, seed: int = 0,
           bottleneck: int = 2, epochs: int = 300,
           sr_niterations: int = 25, sr_maxsize: int = 20,
           max_manifold_samples: int = 4000) -> LawResult:
    dataset = targets.assemble(law, symbols, date)
    names = dataset.names_dimless
    names_raw = dataset.names_raw

    # -- 무차원 트랙 교사 -----------------------------------------------------
    selection = manifold.select(dataset.X_dimless, dataset.mask,
                                max_samples=max_manifold_samples, seed=seed)
    fit_rows = selection.index[selection.on_manifold]
    fit_weight = selection.weight[selection.on_manifold]
    teacher = ScalarTeacher(n_features=dataset.X_dimless.shape[1], bottleneck=bottleneck,
                            seed=seed).fit(dataset.X_dimless[fit_rows], dataset.y_dimless[fit_rows],
                                          fit_weight, epochs=epochs)
    teacher_pred = teacher.predict(dataset.X_dimless[fit_rows])
    teacher_diag = {
        "track": "dimless", "n_fit_rows": int(len(fit_rows)),
        "r2_vs_real_y": weighted_r2(dataset.y_dimless[fit_rows], teacher_pred, fit_weight),
        "correlation_vs_real_y": float(np.corrcoef(teacher_pred, dataset.y_dimless[fit_rows])[0, 1])
                                 if np.std(teacher_pred) > 0 else 0.0,
        "bottleneck": bottleneck, "epochs": epochs,
    }

    # -- raw 트랙 교사 (ablation "무차원화 없이 증류" 전용) -------------------
    selection_raw = manifold.select(dataset.X_raw, dataset.mask,
                                    max_samples=max_manifold_samples, seed=seed)
    fit_rows_raw = selection_raw.index[selection_raw.on_manifold]
    fit_weight_raw = selection_raw.weight[selection_raw.on_manifold]
    teacher_raw = ScalarTeacher(n_features=dataset.X_raw.shape[1], bottleneck=bottleneck,
                                seed=seed).fit(dataset.X_raw[fit_rows_raw],
                                              dataset.y_raw[fit_rows_raw],
                                              fit_weight_raw, epochs=epochs)
    teacher_raw_pred = teacher_raw.predict(dataset.X_raw[fit_rows_raw])
    teacher_raw_diag = {
        "track": "raw", "n_fit_rows": int(len(fit_rows_raw)),
        "r2_vs_real_y": weighted_r2(dataset.y_raw[fit_rows_raw], teacher_raw_pred, fit_weight_raw),
        "correlation_vs_real_y": float(np.corrcoef(teacher_raw_pred,
                                                    dataset.y_raw[fit_rows_raw])[0, 1])
                                 if np.std(teacher_raw_pred) > 0 else 0.0,
        "bottleneck": bottleneck, "epochs": epochs,
    }

    result = LawResult(law=law, n_rows_total=dataset.n_rows_total,
                       symbols_used=dataset.symbols_used,
                       symbols_skipped=dataset.symbols_skipped,
                       teacher_dimless_diag=teacher_diag, teacher_raw_diag=teacher_raw_diag)

    common = dict(seed=seed, niterations=sr_niterations, maxsize=sr_maxsize)

    # main: on-manifold, 무차원, 증류(교사 출력)
    result.tracks["main"] = _fit_track(
        law, "main", dataset.X_dimless[fit_rows], teacher_pred, fit_weight, names, **common)

    # no_distillation: 같은 표집·같은 X, 타깃만 실측 y 로.
    result.tracks["no_distillation"] = _fit_track(
        law, "no_distillation", dataset.X_dimless[fit_rows], dataset.y_dimless[fit_rows],
        fit_weight, names, **common)

    # no_dimensionless: raw 공간에서 표집·raw 교사로 증류.
    result.tracks["no_dimensionless"] = _fit_track(
        law, "no_dimensionless", dataset.X_raw[fit_rows_raw], teacher_raw_pred,
        fit_weight_raw, names_raw, **common)

    # uniform_off_manifold: 같은 무차원 X 의 열별 범위에서 균등 박스 합성,
    # 같은(무차원) 교사에 질의 — 표집 절차만 바꾼다.
    synthetic_X = ablation.uniform_box_sample(dataset.X_dimless[fit_rows], n=len(fit_rows),
                                              seed=seed)
    synthetic_y = teacher.predict(synthetic_X)
    result.tracks["uniform_off_manifold"] = _fit_track(
        law, "uniform_off_manifold", synthetic_X, synthetic_y,
        np.ones(len(synthetic_X)), names, **common)

    return result
