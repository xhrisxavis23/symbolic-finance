import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher.window import make_causal_windows  # noqa: E402


def test_output_shape():
    X = np.arange(20.0).reshape(10, 2)
    Xw = make_causal_windows(X, window=3)
    assert Xw.shape == (10, 6)


def test_current_row_is_last_block():
    """윈도우의 마지막 블록(가장 최근)은 그 행 자신의 원본 feature 여야 한다."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    window = 5
    Xw = make_causal_windows(X, window=window)
    last_block = Xw[:, (window - 1) * 4:window * 4]
    np.testing.assert_allclose(last_block, X)


def test_padding_repeats_earliest_row_within_session_not_zero():
    """행 수가 window 보다 적은 초반부는 세션의 **첫 행을 반복**해 채운다.

    0 패딩이었다면 아래 첫 행(전부 7.0)의 가장 오래된 블록이 0 이 됐을 것이다.
    """
    X = np.array([[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]])
    Xw = make_causal_windows(X, window=4)
    oldest_block_row0 = Xw[0, 0:2]
    np.testing.assert_allclose(oldest_block_row0, [7.0, 7.0])
    # 두 번째로 오래된 블록도 아직 첫 행 반복이어야 한다(window=4, 실제 역사는 1개뿐).
    second_oldest_row0 = Xw[0, 2:4]
    np.testing.assert_allclose(second_oldest_row0, [7.0, 7.0])


def test_window_one_is_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    Xw = make_causal_windows(X, window=1)
    np.testing.assert_allclose(Xw, X)


def test_no_future_leakage_via_corruption():
    """미래 행을 오염시켜도 과거 시점의 윈도우 출력이 바뀌면 안 된다.

    이것이 이 저장소 전체의 핵심 불변식이다 — 사용자 지시: "시간 윈도우에
    미래 틱이 새어 들어가면 이 실험 전체가 무의미해진다." 행 값 자체를
    행 인덱스로 인코딩해(`X[i] = i`) 어느 행이 섞였는지 바로 드러나게 한다.
    """
    n, window = 200, 10
    X_before = np.tile(np.arange(n, dtype=float)[:, None], (1, 3))
    Xw_before = make_causal_windows(X_before, window=window)

    future_start = 120
    X_after = X_before.copy()
    X_after[future_start:] = -999.0                 # "미래" 오염
    Xw_after = make_causal_windows(X_after, window=window)

    # 오염된 행보다 앞선 모든 행은 미래를 보지 않으므로 완전히 동일해야 한다.
    np.testing.assert_allclose(Xw_before[:future_start], Xw_after[:future_start])

    # 오염이 실제로 무언가를 바꾸긴 하는지(공허한 통과 방지) — 오염된 행
    # 자신의 윈도우(현재 블록)는 반드시 달라져야 한다.
    assert not np.allclose(Xw_before[future_start], Xw_after[future_start])


def test_session_boundary_not_crossed():
    """세션 경계를 넘어 과거를 빌려오면 안 된다 — 다른 종목의 역사가 섞이는 것도
    일종의 누설이다(사용자 지시: 세션 경계도 인과성 보장의 일부).

    세션1 값은 전부 1000 이상, 세션2 값은 전부 100 미만으로 둬서 섞이면
    바로 드러나게 한다.
    """
    session1 = np.full((30, 2), 1000.0) + np.arange(30)[:, None]
    session2 = np.full((5, 2), 1.0) + np.arange(5)[:, None]
    X = np.vstack([session1, session2])
    session_ids = np.array(["A"] * 30 + ["B"] * 5)

    window = 8
    Xw = make_causal_windows(X, window=window, session_ids=session_ids)

    # 세션2의 첫 행(전체에서 30번째 행) 윈도우는 window=8 이라 앞의 7개 행이
    # 필요하지만 세션2엔 자기 자신 하나뿐이다 — 전부 세션2 첫 행 반복이어야
    # 하고, 세션1(1000 이상) 값이 하나도 섞이면 안 된다.
    first_of_session2 = Xw[30]
    assert np.all(first_of_session2 < 100.0)

    # 오염 테스트로 재확인: 세션1을 통째로 오염시켜도 세션2 윈도우는 불변.
    X_corrupt = X.copy()
    X_corrupt[:30] = -12345.0
    Xw_corrupt = make_causal_windows(X_corrupt, window=window, session_ids=session_ids)
    np.testing.assert_allclose(Xw[30:], Xw_corrupt[30:])


def test_two_separate_runs_of_the_same_session_id_do_not_merge():
    """같은 id 가 비연속으로 다시 나타나면(런이 두 개) 각 런을 독립으로 본다 —
    뒤 런이 앞 런의 역사를 빌리면 안 된다(앞 런은 이미 끝난 세션이다)."""
    run1 = np.full((5, 1), 111.0)
    other = np.full((3, 1), 5.0)
    run2 = np.full((5, 1), 222.0)
    X = np.vstack([run1, other, run2])
    session_ids = np.array(["A"] * 5 + ["B"] * 3 + ["A"] * 5)

    Xw = make_causal_windows(X, window=10, session_ids=session_ids)
    # run2 의 첫 행(인덱스 8)은 자기 세션(A, 두번째 런)의 시작이므로, 자기
    # 자신(222)만 반복해야 한다 — 첫 런의 111 이 섞이면 안 된다.
    assert np.all(Xw[8] == 222.0)


def test_window_must_be_positive():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        make_causal_windows(X, window=0)


def test_session_ids_length_mismatch_raises():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        make_causal_windows(X, window=2, session_ids=np.array(["A", "B"]))


def test_empty_input():
    X = np.zeros((0, 3))
    Xw = make_causal_windows(X, window=4)
    assert Xw.shape == (0, 12)


def test_non_finite_history_is_forward_filled_not_left_nan():
    """실측(20260316 실데이터)에서 발견: 전체 행의 ~4.2%가 비유한 feature 값을
    가진다(`ofi_over_qbar_5` 등 유동성 분모 비율계 열). window=16 이면
    "현재 행은 유한하지만 과거 어딘가 하나가 비유한"일 확률이 대략 50% —
    그대로 두면 `DeepLOBCompact.fit` 의 배치 손실이 그 한 표본 때문에 전부
    NaN 이 된다(실측: 첫 실전 비교에서 `r2_path=-inf` 로 재현됐다). 그
    회귀를 이 테스트로 고정한다."""
    X = np.array([[1.0], [2.0], [np.nan], [np.nan], [5.0], [6.0]])
    Xw = make_causal_windows(X, window=3)
    assert np.all(np.isfinite(Xw))
    # 행 2(값 nan)의 윈도우: [행0=1, 행1=2, 행2=직전 유한값인 2 로 채움].
    np.testing.assert_allclose(Xw[2], [1.0, 2.0, 2.0])
    # 행 3(값 nan)도 마찬가지로 직전 유한값(행1=2, forward-fill 이 누적)을 본다.
    np.testing.assert_allclose(Xw[3], [2.0, 2.0, 2.0])
    # 행 4(값 5, 유한)부터는 다시 원래 값이 보인다.
    np.testing.assert_allclose(Xw[4], [2.0, 2.0, 5.0])


def test_leading_non_finite_falls_back_to_zero_not_borrowed_from_elsewhere():
    """세션 시작부터 유한값이 한 번도 없으면 빌려올 과거가 없다 — 0 폴백."""
    X = np.array([[np.nan], [np.nan], [3.0], [4.0]])
    Xw = make_causal_windows(X, window=2)
    assert np.all(np.isfinite(Xw))
    np.testing.assert_allclose(Xw[0], [0.0, 0.0])
    np.testing.assert_allclose(Xw[1], [0.0, 0.0])
    np.testing.assert_allclose(Xw[2], [0.0, 3.0])


def test_forward_fill_does_not_cross_session_boundary():
    """세션2 시작이 비유한이면 세션1의 마지막 유한값을 빌려오면 안 된다 —
    그것도 세션 경계를 넘는 누설이다. 0 폴백이어야 한다."""
    X = np.array([[100.0], [200.0], [np.nan], [7.0]])
    session_ids = np.array(["A", "A", "B", "B"])
    Xw = make_causal_windows(X, window=2, session_ids=session_ids)
    assert np.all(np.isfinite(Xw))
    # 행 2(세션B 첫 행, nan): 세션A 의 200 을 빌리면 안 된다 — 0 이어야 한다.
    np.testing.assert_allclose(Xw[2], [0.0, 0.0])
    np.testing.assert_allclose(Xw[3], [0.0, 7.0])


def test_forward_fill_is_causal_under_future_corruption():
    """forward-fill 채움값이 미래에서 오지 않는지 오염 테스트로 재확인한다
    (구조로는 이미 보장되지만 — `last_finite_row` 가 `arange` 의 누적
    최댓값이라 항상 과거만 본다 — 실측으로도 남긴다)."""
    n = 50
    X = np.arange(n, dtype=float)[:, None]
    X[10] = np.nan
    Xw_before = make_causal_windows(X, window=4)

    X_after = X.copy()
    X_after[30:] = -999.0
    Xw_after = make_causal_windows(X_after, window=4)

    np.testing.assert_allclose(Xw_before[:30], Xw_after[:30])


def test_must_be_called_before_subsampling_not_after_reorder():
    """표집 후 재배열된 행에 윈도우를 씌우면 "시간 윈도우"가 거짓이 됨을
    직접 보여준다 — 문서화된 사용 규칙(모듈 docstring)의 근거를 실측으로
    남긴다. 정순(원본) 대비 무작위로 뒤섞은 순서에 윈도우를 씌우면 각 행의
    "과거"가 실제 과거와 무관해진다(오래된 블록이 현재 블록과 같아지는
    비율이 원본보다 뚜렷이 낮아짐)."""
    n = 500
    X = np.arange(n, dtype=float)[:, None]
    Xw_correct = make_causal_windows(X, window=5)
    # 정순: 오래된 블록 = 최근 블록 - 4 (항상 성립, 초반 패딩 구간 제외).
    tail_correct = Xw_correct[10:]
    assert np.all(tail_correct[:, -1] - tail_correct[:, 0] == 4.0)

    rng = np.random.default_rng(0)
    shuffled_idx = rng.permutation(n)
    X_shuffled = X[shuffled_idx]
    Xw_wrong = make_causal_windows(X_shuffled, window=5)
    tail_wrong = Xw_wrong[10:]
    # 뒤섞은 뒤에 씌우면 "옛것 블록 = 최근 블록 - 4" 관계가 거의 깨진다.
    matches = np.mean(tail_wrong[:, -1] - tail_wrong[:, 0] == 4.0)
    assert matches < 0.1
