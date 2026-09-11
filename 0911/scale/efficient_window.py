"""부분(subset) 인과적 시간 윈도우 생성 — 0902 의 `sd.teacher.window.make_causal_windows`
와 **수학적으로 동일한 결과**를 내지만, 표집으로 고른 행에 대해서만 출력을
만들어 메모리를 `pool_size` 가 아니라 `len(indices)` 에 비례하게 줄인다.

왜 필요한가 (0910/ANALYSIS.md §3.2, 0910/SCALE-MEASUREMENT.md 의 권고사항):
`make_causal_windows` 는 표집 **이전**의 연속 행렬 전체에 윈도우를 씌운다 —
출력이 `(n_pool, window*F)` 이라 전종목·3일치 규모(수천만 행)에서는 출력
배열 하나가 수십~수백 GB 가 된다. 실제로 필요한 것은 `manifold.select` 가
고른 `max_samples` 개 행의 윈도우뿐이다. 이 파일은 **0902 를 건드리지
않고** 그 최적화를 0911 쪽에서 구현한다 — 새 코드이므로 정확성을 반드시
`make_causal_windows` 와 직접 대조해서 검증한다(이 파일 자신의
`_selftest()` 및 `verify_against_reference.py`).

로직은 `sd/teacher/window.py` 의 클램프 공식을 그대로 옮긴 것이다:
`idx = max(i - lag, session_start[i])`, forward-fill 도 동일한 누적 최댓값
방식이다. 차이는 마지막에 **전체 n 이 아니라 `indices` 로 지정된 행만
gather** 한다는 것 하나뿐이다.
"""
from __future__ import annotations

import numpy as np


def session_starts(session_ids: np.ndarray) -> np.ndarray:
    """`sd.teacher.window.make_causal_windows` 와 동일한 세션 시작 인덱스 계산."""
    session_ids = np.asarray(session_ids)
    n = len(session_ids)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = session_ids[1:] != session_ids[:-1]
    run_of = np.cumsum(change) - 1
    run_starts = np.flatnonzero(change)
    return run_starts[run_of]


def forward_fill_non_finite(X: np.ndarray, session_start: np.ndarray) -> np.ndarray:
    """`sd/teacher/window.py::_forward_fill_non_finite` 와 동일 (그대로 복제)."""
    n, n_features = X.shape
    finite = np.isfinite(X)
    if finite.all():
        return X
    rows = np.arange(n)
    row_if_finite = np.where(finite, rows[:, None], -1)
    last_finite_row = np.maximum.accumulate(row_if_finite, axis=0)
    has_prior_finite_in_session = last_finite_row >= session_start[:, None]
    source_row = np.where(has_prior_finite_in_session, last_finite_row, 0)
    filled_from_past = np.take_along_axis(X, source_row, axis=0)
    return np.where(finite, X, np.where(has_prior_finite_in_session, filled_from_past, 0.0))


def make_causal_windows_subset(X: np.ndarray, window: int, session_ids: np.ndarray,
                               indices: np.ndarray) -> np.ndarray:
    """`make_causal_windows(X, window, session_ids)[indices]` 와 동일한 값을,
    `(len(indices), window*F)` 짜리 출력만 만들어서 낸다 — `(len(X), window*F)`
    전체를 만들지 않는다.

    선행 조건: `X`, `session_ids` 는 원본 연속(정렬된) 배열이어야 한다
    (표집 이후 뒤섞인 배열이 아니라) — `sd/teacher/window.py` 모듈
    docstring 의 요구사항과 같다.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 는 2차원이어야 한다: got shape {X.shape}")
    n, n_base_features = X.shape
    window = int(window)
    if window < 1:
        raise ValueError(f"window 는 1 이상이어야 한다: got {window}")

    session_start_full = session_starts(session_ids)
    X_filled = forward_fill_non_finite(X, session_start_full)

    indices = np.asarray(indices, dtype=np.int64)
    lag = np.arange(window - 1, -1, -1)                          # (window,)
    raw_idx = indices[:, None] - lag[None, :]                     # (m, window)
    clamped = np.maximum(raw_idx, session_start_full[indices][:, None])
    gathered = X_filled[clamped]                                  # (m, window, F)
    return gathered.reshape(len(indices), window * n_base_features)


def _selftest() -> None:
    """`make_causal_windows` 와 직접 대조 — 무작위 데이터·세션 경계·비유한 값
    섞어서 완전 일치를 확인한다."""
    import sys
    sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
    from sd.teacher.window import make_causal_windows

    rng = np.random.default_rng(42)
    n, F, window = 5000, 7, 16
    X = rng.normal(size=(n, F))
    # 비유한 값 몇 개 섞는다 (forward-fill 경로 검증)
    nan_mask = rng.random(size=(n, F)) < 0.05
    X[nan_mask] = np.nan
    # 세션 경계: 여러 길이의 런
    session_ids = np.repeat(np.arange(37), np.diff(
        np.sort(rng.choice(np.arange(1, n), size=36, replace=False)).tolist() + [n],
        prepend=0))[:n]
    # 길이 안 맞으면 패딩
    if len(session_ids) < n:
        session_ids = np.concatenate([session_ids, np.full(n - len(session_ids),
                                                             session_ids[-1])])

    reference_full = make_causal_windows(X, window=window, session_ids=session_ids)

    # 무작위 부분집합 (표집을 흉내낸다 — 순서 뒤섞임 포함)
    indices = rng.choice(n, size=1500, replace=False)
    subset_result = make_causal_windows_subset(X, window=window, session_ids=session_ids,
                                               indices=indices)
    reference_subset = reference_full[indices]

    same = np.allclose(subset_result, reference_subset, equal_nan=True)
    max_diff = np.nanmax(np.abs(subset_result - reference_subset))
    print(f"[selftest] shapes match: {subset_result.shape == reference_subset.shape}")
    print(f"[selftest] allclose: {same}, max_abs_diff: {max_diff}")
    assert subset_result.shape == reference_subset.shape
    assert same, f"불일치! max_diff={max_diff}"
    print("[selftest] PASS — make_causal_windows_subset 이 참조 구현과 완전히 일치")


if __name__ == "__main__":
    _selftest()
