"""시간 윈도우 생성 — DeepLOBCompact 의 입력 간극을 메운다. PREREG-E0-V2.md §1-2.

지금 파이프라인의 교사는 feature 행렬 `X: (n, F)` 를 받는다(`sd/ticks.py`,
`sd/e0/runner.py`, `run_slice.py`) — 한 행이 한 시각의 관측이고 시간 축이
없다. 원 DeepLOB(Zhang, Zohren & Roberts, 2019) 은
`(시간 윈도우 × 호가 레벨 × {가격,수량})` 텐서를 받는다. 이 함수가 그 간극을
메운다: 행 `i` 를 그 행과 **과거 `window-1` 개** 행을 이어붙인
`(window·F,)` 벡터로 바꾼다. `DeepLOBCompact` 는 이 평평한 벡터를 받아
내부에서 `(window, F)` 로 되돌린다(`deeplob.py` 참조).

**반드시 표집(`sd.manifold.select`) 이전에, 원본 연속 행렬에 대해서만
호출한다.** `manifold.select` 는 무작위로 행을 골라 재배열한다(`chosen =
rng.choice(...)`, `sd/manifold.py`) — 표집 후 "바로 위 행"은 시간상 이웃이
아니라 무작위로 뽑힌 다른 시각(심지어 다른 종목)의 행일 수 있다. 표집 후에
윈도우를 만들면 미래 누설은 아니어도("과거"의 정의 자체가 무작위 순서가
되므로) "시간 윈도우"라는 말이 거짓이 된다. 그래서 윈도우 생성을 표집과
분리된 독립 함수로 두어 호출 순서를 구조로 강제한다 — `manifold.select` 안에
넣지 않는다(그러면 이미 늦다).

## 인과성

행 `i` 의 윈도우는 인덱스 `[i-window+1, ..., i]` 만 본다 — 자기 자신을
포함하고 미래는 배제한다. `session_ids` 로 준 연속 구간(세션·종목) 경계를
넘지 않는다: 세션 시작 이전으로 모자란 만큼은 **그 세션의 첫 행을 반복**해
채운다. 0 패딩을 쓰지 않은 이유 — 표준화(평균 차감) 후 0 은 "평균적인 시장
상태였다"는 관측을 지어내지만, 반복 패딩은 "그 시점엔 그 관측이 최선의
추정이었다"는 사실 그대로다(0 이 우연히 평균 근처일 이유가 없는 원시
feature 에서 특히 그렇다).

구현은 클램프 하나로 인과성을 **구조적으로** 보장한다 — 테스트가 아니라
코드 자체가 증명이다. `raw_idx = i - lag` (`lag >= 0`) 는 언제나 `<= i` 이고,
`session_start[i] <= i` 도 항상 참이므로 `idx = max(raw_idx, session_start[i])`
는 두 값 다 `<= i` 인 것들의 최댓값이라 반드시 `<= i` 다 — `window` 나
`session_ids` 값이 무엇이든 미래 인덱스가 나올 수 없다.
`tests/test_teacher_window.py` 가 그래도 실측으로 재확인한다: 미래 행을
오염시켜도 과거 시점의 윈도우 출력이 바뀌지 않아야 통과한다.

## 비유한(non-finite) 값 — 실측에서 발견

`manifold.select` 는 **행 i 자신**의 feature 가 비유한이면 그 행을 표집에서
뺀다(`np.all(np.isfinite(X), axis=1)`). 하지만 윈도우는 i 의 **과거** 행도
끌어오고, 과거 행은 그 필터를 통과하지 않았다 — 실측(20260316, 12종목,
`ofi_over_qbar_5` 등 유동성 분모 비율계 열)으로 확인한 결과 전체 행의
약 4.2%가 비유한이었다. window=16 이면 "현재 행은 유한하지만 과거 16개 중
하나가 비유한"일 확률이 대략 1-(1-0.042)^16 ≈ 50%다 — 이걸 그대로 넘기면
`DeepLOBCompact.fit` 의 손실이 그 배치 하나 때문에 전부 NaN 이 된다(실측
확인 — 첫 실전 비교(§ 비교표)에서 `r2_path=-inf` 로 재현됐다).

그래서 게이트-전 단계로 **세션 내 인과적 forward-fill**을 넣는다: 열 f, 행
i 가 비유한이면 같은 세션 안에서 `i` **이전**의 가장 최근 유한값으로
채운다(세션 경계는 넘지 않는다 — 다른 종목의 값을 빌리는 것도 일종의
누설이다). 세션 시작부터 한 번도 유한값이 없었으면(그 시점까지 빌릴 과거가
없다) `0.0` 으로 둔다 — 이 경우만은 표준화 후 "평균적인 시장 상태"를
지어낼 위험보다 "이 세션엔 아직 유효한 관측이 없었다"는 사실이 더 크다고
보고, 어차피 이런 사례는 세션의 맨 앞 몇 틱에서만 나온다(실측: 12종목
어디에서도 세션 첫 행 자체가 비유한이었던 사례는 없었다). forward-fill 은
**과거 값만** 쓰므로 인과성을 깨지 않는다 — `test_forward_fill_is_causal_*`
가 미래 오염으로 재확인한다.
"""

from __future__ import annotations

import numpy as np


def make_causal_windows(X: np.ndarray, window: int,
                        session_ids: np.ndarray | None = None) -> np.ndarray:
    """`(n, F)` → `(n, window*F)`. 시간축은 각 행 안에서 옛것→최신 순으로 눕는다.

    `session_ids` 를 주면 **같은 값이 연속으로 나타나는 구간(런)** 을 하나의
    세션으로 본다 — 이 파이프라인이 종목별로 행을 이어붙이는 방식(연속
    블록, `run_slice.py` 의 `np.vstack([...])`)과 맞다. 같은 id 가 서로 떨어진
    두 구간에 따로 나타나도(비연속) 각 구간은 독립된 런으로 취급되므로
    잘못 섞이지 않는다 — 다만 그런 배치라면 애초에 "세션"의 뜻이 흐려지니
    호출자가 연속 블록으로 준비했는지 스스로 확인해야 한다.

    `session_ids` 가 `None` 이면 `X` 전체를 세션 하나로 본다(단일 종목·단일
    연속열을 다룰 때).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 는 2차원이어야 한다: got shape {X.shape}")
    n, n_base_features = X.shape
    window = int(window)
    if window < 1:
        raise ValueError(f"window 는 1 이상이어야 한다: got {window}")
    if n == 0:
        return np.zeros((0, window * n_base_features), dtype=float)

    if session_ids is None:
        session_start = np.zeros(n, dtype=int)
    else:
        session_ids = np.asarray(session_ids)
        if len(session_ids) != n:
            raise ValueError(
                f"session_ids 길이({len(session_ids)})가 X 행 수({n})와 다르다")
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = session_ids[1:] != session_ids[:-1]
        run_of = np.cumsum(change) - 1
        run_starts = np.flatnonzero(change)
        session_start = run_starts[run_of]

    X = _forward_fill_non_finite(X, session_start)

    rows = np.arange(n)
    lag = np.arange(window - 1, -1, -1)                     # window-1..0, 옛것부터
    raw_idx = rows[:, None] - lag[None, :]                    # (n, window), 미래 아님(lag>=0)
    clamped = np.maximum(raw_idx, session_start[:, None])     # 세션 시작 이전 금지
    gathered = X[clamped]                                     # (n, window, F)
    return gathered.reshape(n, window * n_base_features)


def _forward_fill_non_finite(X: np.ndarray, session_start: np.ndarray) -> np.ndarray:
    """열별로, 세션 안에서만, 비유한 값을 **직전 유한값**으로 채운다.

    세션 시작부터 유한값이 한 번도 없었던 자리는 `0.0`(모듈 docstring
    "비유한 값" 절 참조 — 세션 맨 앞 몇 틱에서만 나오는 예외적 경우다).
    미래 행을 보지 않는다 — `last_finite_row` 는 오직 `arange(n)` 의 누적
    최댓값이라 행 i 자리를 채우는 값은 항상 `<= i` 인 행에서 온다.
    """
    n, n_features = X.shape
    finite = np.isfinite(X)
    if finite.all():
        return X

    rows = np.arange(n)
    row_if_finite = np.where(finite, rows[:, None], -1)          # (n, F)
    last_finite_row = np.maximum.accumulate(row_if_finite, axis=0)  # 전역 누적 최댓값(과거만)
    has_prior_finite_in_session = last_finite_row >= session_start[:, None]

    source_row = np.where(has_prior_finite_in_session, last_finite_row, 0)
    filled_from_past = np.take_along_axis(X, source_row, axis=0)   # (n, F)
    return np.where(finite, X, np.where(has_prior_finite_in_session, filled_from_past, 0.0))
