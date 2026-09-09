"""공용 조기 종료 루프. PREREG-T1.md §1 (A10 — 교사 표본 내 채점).

## 왜 이 파일이 필요한가

`sd.teacher.shallow.ShallowMLP`·`sd.teacher.deeplob.DeepLOBCompact`·
`sd.e0.teacher.ScalarTeacher`·`sd.e0.teacher.ScalarDeepLOB` 넷 다 지금까지
"적합 종목에서 정해진 epoch 수만큼 학습하고 끝"이었다 — 붕괴가 훈련 중
손실 곡선에 보이지 않는데도(A10, DeepLOB 표본외 R² −1.72/−0.40/−13.86) 그냥
마지막 epoch 의 가중치를 썼다. 이 모듈은 "몇 epoch 마다 held-out 점수를
재고, 개선이 patience 만큼 멈추면 그 시점의 가중치로 되돌려 종료한다"는
스케줄링 로직 **하나만** 담당한다 — 옵티마이저 스텝 자체(모델 순전파·역전파)
는 호출자가 갖고 있고, 이 함수는 콜백만 받는다. 그래서 torch 를 몰라도
단위 테스트가 가능하다(`tests/test_teacher_early_stopping.py` 는 실제로
torch 를 한 번도 import 하지 않는다).

## 설계 판단 (PREREG-T1.md "설계 판단이 필요한 지점", 결과를 보기 전에 고정)

- **되돌릴 가중치는 최선 시점 스냅샷 하나만 보관한다** (전체 히스토리를
  들고 있지 않는다). 이 저장소에 "최선 이후의 중간 체크포인트를 나중에
  다시 본다"는 용도가 없다 — 항상 "지금까지 관측한 최선으로 되돌린다"만
  필요하다. 매 평가마다 `deepcopy(state_dict())` 를 새로 만들지만 이전
  것을 계속 들고 있지는 않는다(가장 최근 최선 하나만 참조가 남는다).
- **평가 점수가 유한하지 않으면(`-inf`/`nan`, `weighted_r2` 가 표본 부족일
  때 이미 쓰는 관례) 그 평가는 "개선"으로 세지 않는다** — patience 카운터는
  증가하지만 최선 스냅샷을 갱신하지도, 초기 미학습 상태로 되돌리지도
  않는다. 유효한 개선이 단 한 번도 없었다면(실전에서는 거의 없다 — 선택
  종목 최소 30행 이상이 이미 `sd.e0.runner.MIN_SELECT_ROWS` 로 보장된다)
  최종적으로 되돌리지 않고 마지막 학습 상태를 그대로 둔다 — 무작위 초기화
  상태로 되돌리는 것보다 안전하다.
- **`total_epochs` 를 끝까지 다 돌 때까지 patience 를 한 번도 소진하지
  않아도(가장 흔한 경우일 것 — 붕괴가 항상 오지는 않는다) 여전히 최선
  스냅샷으로 되돌린다** — "마지막 epoch"이 아니라 "관측된 최선 시점"이
  항상 이긴다. 이것이 사용자 지시의 뮤테이션 자기검토 대상 #3
  ("되돌릴 가중치를 최선 시점이 아니라 마지막 시점 것으로 바꾸면 잡는가")
  이 겨냥하는 바로 그 지점이다.
- **`evaluate=None` 이면 조기 종료를 완전히 끈다** — 기존 호출부(선택
  종목 데이터를 안 주는 곳)의 동작을 한 비트도 바꾸지 않는다. 이 인자를
  옵션으로 둔 이유는 `ShallowMLP`/`DeepLOBCompact` 처럼 아직 select 데이터를
  안 받는 호출부(예: `run_slice.py`)가 이 함수를 써도 회귀가 없게 하기
  위함이다.
- 평가 주기(`eval_every`)·patience 의 구체적인 값은 여기서 정하지 않는다
  — 호출부(`sd.e0.teacher`)가 정하고 그 근거를 그 파일에 적는다. 이 모듈은
  값에 대해 아무 의견이 없다(일반 스케줄러).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class EarlyStopHistory:
    """진단용 궤적 — 저장소 관례(A9, "진단 수치는 영속화한다")를 따라 호출부가
    이 값을 `teacher_diag` 등에 그대로 실어 JSON 으로 남긴다."""

    eval_every: int = 0
    patience: int = 0
    eval_epochs: list[int] = field(default_factory=list)
    eval_scores: list[float] = field(default_factory=list)
    best_epoch: int = -1                    # 최선이 한 번도 없었으면 -1
    best_score: float = float("-inf")
    stopped_epoch: int = -1                 # 학습 루프가 실제로 멈춘 epoch
    triggered: bool = False                 # patience 소진으로 조기 종료했는가
    reverted: bool = False                  # 최선 스냅샷으로 되돌렸는가(유효 개선이 있었으면 항상 True)
    early_stopping_enabled: bool = True     # evaluate=None 이면 False


def run_with_early_stopping(*, total_epochs: int, eval_every: int, patience: int,
                            train_step: Callable[[], None],
                            evaluate: Callable[[], float] | None,
                            get_state: Callable[[], Any],
                            set_state: Callable[[Any], None]) -> EarlyStopHistory:
    """`train_step()` 을 최대 `total_epochs` 번 부르고, `evaluate` 가 주어지면
    `eval_every` epoch 마다(+ 항상 마지막 epoch 에) 평가해 조기 종료를 관리한다.

    Args:
        total_epochs: 최대 epoch 수.
        eval_every: 몇 epoch 마다 `evaluate()` 를 부를지. `total_epochs` 는
            이 값의 배수가 아니어도 된다 — 마지막 epoch 은 항상 평가한다.
        patience: 연속으로 개선이 없는 평가가 이 횟수에 도달하면 멈춘다.
        train_step: epoch 하나(순전파+역전파+옵티마이저 스텝)를 실행한다.
        evaluate: `None` 이면 조기 종료를 끈다(호출부 기존 동작 보존).
            아니면 held-out 점수(클수록 좋다, 예: 가중 R²)를 돌려준다.
        get_state/set_state: 학습 가능한 파라미터 전체의 스냅샷을 얻고/되돌린다.

    Returns:
        `EarlyStopHistory` — 평가 궤적과 최종적으로 무슨 일이 있었는지.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs 는 1 이상이어야 한다: {total_epochs}")
    if eval_every <= 0:
        raise ValueError(f"eval_every 는 1 이상이어야 한다: {eval_every}")
    if patience <= 0:
        raise ValueError(f"patience 는 1 이상이어야 한다: {patience}")

    history = EarlyStopHistory(eval_every=int(eval_every), patience=int(patience))

    if evaluate is None:
        history.early_stopping_enabled = False
        for _ in range(total_epochs):
            train_step()
        history.stopped_epoch = total_epochs
        return history

    best_state: Any = None
    since_improved = 0
    stopped_epoch = total_epochs

    for epoch in range(1, total_epochs + 1):
        train_step()
        due_for_eval = (epoch % eval_every == 0) or (epoch == total_epochs)
        if not due_for_eval:
            continue

        score = float(evaluate())
        history.eval_epochs.append(epoch)
        history.eval_scores.append(score)

        if math.isfinite(score) and (best_state is None or score > history.best_score):
            history.best_score = score
            history.best_epoch = epoch
            best_state = get_state()
            since_improved = 0
        else:
            since_improved += 1

        if since_improved >= patience:
            history.triggered = True
            stopped_epoch = epoch
            break

    history.stopped_epoch = stopped_epoch
    if best_state is not None:
        set_state(best_state)
        history.reverted = True
    return history


def torch_snapshot_functions(
        modules: Mapping[str, Any]) -> tuple[Callable[[], dict], Callable[[dict], None]]:
    """`{이름: torch.nn.Module}` 여러 개를 하나의 스냅샷으로 묶는 `get_state`/
    `set_state` 쌍을 만든다. `run_with_early_stopping` 에 그대로 넘긴다 —
    이 헬퍼가 없으면 교사 클래스 네 개가 각자 `copy.deepcopy(state_dict())`
    배선을 반복해야 한다."""

    def get_state() -> dict:
        return {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}

    def set_state(state: dict) -> None:
        for name, module in modules.items():
            module.load_state_dict(state[name])

    return get_state, set_state
