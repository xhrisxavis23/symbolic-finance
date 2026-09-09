"""`sd.teacher.early_stopping.run_with_early_stopping` — 순수 스케줄링 로직만
(torch 없이) 검증한다. `get_state`/`set_state` 는 정수 태그를 담은 딕셔너리로
대신해, "어느 시점 상태로 되돌렸는가"를 값 비교만으로 확인한다.

사용자 지시의 뮤테이션 자기검토 대상 세 가지 중 둘(#2·#3)이 이 파일에서
직접 잡힌다:
  #2 조기 종료가 아예 작동하지 않게(항상 마지막 epoch 사용) 만들면 —
     `test_reverts_to_best_epoch_not_last_epoch_when_validation_collapses`.
  #3 되돌릴 가중치를 최선 시점이 아니라 마지막 시점 것으로 바꾸면 — 같은
     테스트 + `test_never_triggers_but_still_reverts_to_best_not_final`.
(#1 "조기 종료 기준을 적합 종목 R² 로 되돌리면"은 이 모듈이 R² 를 전혀
모른다 — 그 배선은 `tests/test_e0_teacher.py`/`tests/test_e0_runner.py` 가
담당한다.)
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher.early_stopping import (  # noqa: E402
    run_with_early_stopping, torch_snapshot_functions)


def _make_state_probe():
    """`train_step`이 부를 때마다 카운터를 올리고, `get_state`/`set_state` 는
    "그 시점의 카운터 값"을 스냅샷으로 삼는 최소 상태 기계."""
    counter = {"epoch": 0}

    def train_step() -> None:
        counter["epoch"] += 1

    def get_state():
        return counter["epoch"]

    def set_state(value) -> None:
        counter["epoch"] = value

    return counter, train_step, get_state, set_state


def test_runs_all_epochs_when_evaluate_is_none_and_disables_early_stopping():
    """`evaluate=None` 은 기존 호출부(select 데이터를 안 주는 곳)의 동작을
    바꾸지 않아야 한다 — patience 개념 자체가 없다."""
    counter, train_step, get_state, set_state = _make_state_probe()
    history = run_with_early_stopping(
        total_epochs=25, eval_every=5, patience=2, train_step=train_step,
        evaluate=None, get_state=get_state, set_state=set_state)
    assert counter["epoch"] == 25
    assert history.stopped_epoch == 25
    assert history.early_stopping_enabled is False
    assert history.triggered is False
    assert history.reverted is False


def test_reverts_to_best_epoch_not_last_epoch_when_validation_collapses():
    """점수가 [평가1: 상승] → [평가2: 최고] → 이후 계속 하락하는 궤적을 만든다
    (A10 이 실측한 "훈련 손실은 계속 내려가는데 표본외는 무너진다" 모양을
    스케줄러 수준에서 흉내낸 것). patience=3(평가 3회 연속 무개선)이면
    평가5(하락 3연속)에서 멈추고, 되돌린 상태는 평가2(최고점)의 것이어야
    한다 — 평가5(마지막으로 본 시점)도, epoch 전체를 다 돈 상태도 아니다."""
    counter, train_step, get_state, set_state = _make_state_probe()
    scores = iter([0.3, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01])   # 평가마다 하나씩 소비

    def evaluate() -> float:
        return next(scores)

    history = run_with_early_stopping(
        total_epochs=100, eval_every=10, patience=3, train_step=train_step,
        evaluate=evaluate, get_state=get_state, set_state=set_state)

    # 평가는 epoch 10,20,...에 일어난다. 평가2(epoch20, 점수0.8)가 최고,
    # 평가3(30,0.5)·평가4(40,0.2)·평가5(50,0.1) 세 번 연속 무개선 -> epoch50 에서 멈춘다.
    assert history.eval_epochs == [10, 20, 30, 40, 50]
    assert history.best_epoch == 20
    assert history.best_score == pytest.approx(0.8)
    assert history.triggered is True
    assert history.stopped_epoch == 50
    # 핵심 단언: 되돌린 상태 == 최선 시점(20)의 상태. 마지막 평가 시점(50)도,
    # counter 의 "현재" 값(학습이 멈춘 채 되돌리지 않은 50)도 아니다.
    assert counter["epoch"] == 20


def test_never_triggers_but_still_reverts_to_best_not_final():
    """patience 를 한 번도 소진하지 않고 `total_epochs` 를 다 돌아도(가장 흔한
    실전 경우), 마지막 epoch 이 최고점이 아니었다면 여전히 최선으로 되돌려야
    한다 — "조기 종료가 트리거되지 않았으니 마지막 것을 쓴다"는 뮤테이션을
    잡는다."""
    counter, train_step, get_state, set_state = _make_state_probe()
    # 5번 평가(epoch 10,20,30,40,50=total), 매번 조금씩 나아지다가 마지막에 소폭 하락.
    scores = iter([0.1, 0.4, 0.7, 0.9, 0.85])

    def evaluate() -> float:
        return next(scores)

    history = run_with_early_stopping(
        total_epochs=50, eval_every=10, patience=10, train_step=train_step,
        evaluate=evaluate, get_state=get_state, set_state=set_state)

    assert history.triggered is False               # patience(10) 를 절대 못 채운다(최대 4)
    assert history.stopped_epoch == 50               # 끝까지 돌았다
    assert history.best_epoch == 40                  # 0.9 가 최고
    assert counter["epoch"] == 40                    # 마지막(50)이 아니라 최선(40)으로 되돌림


def test_reverts_last_state_when_final_epoch_is_the_best():
    """최선이 정확히 마지막 epoch 이면(붕괴가 전혀 없는 경우) 되돌린 상태와
    "그대로 뒀을 때"의 상태가 우연히 같아진다 — 이 경우에도 `best_epoch`
    이 실제로 마지막 평가를 가리키는지 확인해 위 테스트와 짝을 이룬다."""
    counter, train_step, get_state, set_state = _make_state_probe()
    scores = iter([0.1, 0.4, 0.7, 0.95])

    def evaluate() -> float:
        return next(scores)

    history = run_with_early_stopping(
        total_epochs=40, eval_every=10, patience=10, train_step=train_step,
        evaluate=evaluate, get_state=get_state, set_state=set_state)
    assert history.best_epoch == 40
    assert counter["epoch"] == 40
    assert history.triggered is False


def test_non_finite_scores_count_toward_patience_but_never_become_best():
    """`weighted_r2` 는 표본이 모자라거나 분산이 0이면 `-inf` 를 돌려준다
    (A9 관례). `-inf` 가 "개선"으로 잘못 기록돼 무작위 초기 상태로 되돌리는
    사고를 막는지 확인한다."""
    counter, train_step, get_state, set_state = _make_state_probe()
    scores = iter([float("-inf"), float("nan"), float("-inf")])

    def evaluate() -> float:
        return next(scores)

    history = run_with_early_stopping(
        total_epochs=30, eval_every=10, patience=3, train_step=train_step,
        evaluate=evaluate, get_state=get_state, set_state=set_state)

    assert history.best_epoch == -1
    assert history.reverted is False    # 유효 개선이 한 번도 없었다 — 되돌리지 않는다
    assert counter["epoch"] == 30        # 마지막 학습 상태 그대로 남는다(무작위 초기화 아님)
    assert history.triggered is True     # patience(3) 는 여전히 소진된다(무개선 3연속)


def test_eval_every_larger_than_total_epochs_still_evaluates_final_epoch():
    """`eval_every` 가 `total_epochs` 보다 크면 정기 평가 시점이 한 번도 안
    오지만, 마지막 epoch 은 항상 평가해야 한다(그렇지 않으면 evaluate 가
    한 번도 안 불려 조기 종료가 사실상 무의미해진다)."""
    counter, train_step, get_state, set_state = _make_state_probe()
    history = run_with_early_stopping(
        total_epochs=5, eval_every=100, patience=1, train_step=train_step,
        evaluate=lambda: 0.42, get_state=get_state, set_state=set_state)
    assert history.eval_epochs == [5]
    assert history.best_epoch == 5


@pytest.mark.parametrize("kwargs", [
    dict(total_epochs=0, eval_every=1, patience=1),
    dict(total_epochs=1, eval_every=0, patience=1),
    dict(total_epochs=1, eval_every=1, patience=0),
])
def test_rejects_non_positive_arguments(kwargs):
    with pytest.raises(ValueError):
        run_with_early_stopping(train_step=lambda: None, evaluate=lambda: 0.0,
                                get_state=lambda: None, set_state=lambda s: None, **kwargs)


class _FakeModule:
    """`torch.nn.Module` 계약(`state_dict`/`load_state_dict`) 만 흉내내는 스텁 —
    이 헬퍼가 실제로 여러 모듈을 하나의 스냅샷으로 묶는지만 확인한다(torch
    불필요)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def state_dict(self) -> dict:
        return {"value": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.value = state["value"]


def test_torch_snapshot_functions_round_trip_multiple_modules():
    encoder = _FakeModule(1.0)
    head = _FakeModule(2.0)
    get_state, set_state = torch_snapshot_functions({"encoder": encoder, "head": head})

    snapshot = get_state()
    encoder.value, head.value = 99.0, 98.0
    assert encoder.value == 99.0 and head.value == 98.0

    set_state(snapshot)
    assert encoder.value == 1.0
    assert head.value == 2.0


def test_torch_snapshot_functions_snapshot_is_independent_of_later_mutation():
    """`get_state` 가 참조가 아니라 깊은 복사를 남기는지 — 그렇지 않으면
    다음 평가에서 "최선"으로 저장한 스냅샷이 그 뒤 학습으로 몰래 바뀐다."""
    encoder = _FakeModule(1.0)
    get_state, _set_state = torch_snapshot_functions({"encoder": encoder})
    snapshot = get_state()
    encoder.value = 500.0
    assert snapshot["encoder"]["value"] == 1.0
