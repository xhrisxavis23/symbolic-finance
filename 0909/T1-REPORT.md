# T1 — 교사 조기 종료를 선택 종목 R² 로 (A10)

작성: 2026-09-09. 사전등록: `PREREG-T1.md`(2026-09-09, 결과를 보기 전에 커밋).
구현·검증은 결과를 보기 전에 끝냈다 — 이 문서의 "실행 결과" 절은 두 실행
(`--teacher shallow`, `--teacher deeplob`)이 각각 끝난 뒤에 채운다.

---

## 0. 무엇을 고쳤나

`sd/e0/runner.py::run_law` 의 교사(`teacher`·`teacher_raw` 두 벌)가 예전에는
적합 종목에서 정해진 epoch 수(700)만큼 학습하고 끝이었다. 손실은 계속
내려가는데 표본외(선택 종목) R² 는 조용히 무너질 수 있었다 — DeepLOB 실측
(2026-09-08 `runs/e0-20260908T182910Z`): 표본내 +0.74~+0.95, 표본외
L1 **−1.72**, L2 +0.207, L3 **−0.40**, L4 +0.02, L5 **−13.86**.

지금은 두 교사 모두 `.fit()` 에 선택 종목 데이터(`X_select`/`y_select`/
`weight_select`)를 같이 받아 매 `eval_every` epoch 마다 선택 종목 R² 를
재고, `patience` 만큼 개선이 없으면 그 시점 가중치로 되돌려 종료한다.

### 바꾼 파일 (전부 소유 범위 안)

| 파일 | 내용 |
| --- | --- |
| `0902/sd/teacher/early_stopping.py` (신규) | 공용 조기 종료 스케줄러. torch 를 모른다(콜백만 받는다) — `run_with_early_stopping`, `torch_snapshot_functions` |
| `0902/sd/e0/teacher.py` | `ScalarTeacher.fit`/`ScalarDeepLOB.fit` 에 `X_select`/`y_select`/`weight_select`/`eval_every`/`patience` 추가. `EVAL_EVERY_DEFAULT=10`, `PATIENCE_DEFAULT=10` |
| `0902/sd/e0/runner.py` | `run_law` — 선택 종목 on-manifold 표집을 교사 학습 **전**으로 옮겨 `.fit()` 에 넘긴다. `teacher_diag`/`teacher_raw_diag` 에 `early_stop`·`contamination_note` 필드 추가(A9 관례: 진단 영속화) |
| `0902/sd/teacher/shallow.py`, `0902/sd/teacher/deeplob.py` | `ShallowMLP.fit`/`DeepLOBCompact.fit` 에 같은 능력(경로 헤드 R² 기준) — PREREG-T1.md §1 "구현 위치: ShallowMLP·DeepLOBCompact 공통"을 따름. 이번 E0 실행은 이 두 클래스를 직접 쓰지 않는다(`run_slice.py` 전용, 건드리지 않았다) — 선택 인자 기본값이 `None`이라 기존 호출부는 바이트 단위로 동작 보존 |
| `0902/tests/test_teacher_early_stopping.py` (신규) | 스케줄러 단위 테스트 9개 |
| `0902/tests/test_e0_teacher.py`, `test_teacher.py`, `test_teacher_deeplob.py`, `test_e0_runner.py` | 배선·회귀 테스트 추가 |

`sd/e0/criteria.py` 는 한 바이트도 건드리지 않았다. `0909/` 의 다른 md 파일은
건드리지 않았다(현재 다른 작업자가 `PREREG-G2.md`를 커밋 중인 것을 확인했고,
그 파일·`g2_*.py`는 전혀 손대지 않았다).

---

## 1. 설계 판단 (결과를 보기 전에 고정, PREREG-T1.md §1·§4)

### 1.1 평가 주기·patience

**`eval_every=10`, `patience=10`(=100 epoch 무개선)**, `--teacher shallow`·
`--teacher deeplob` 양쪽에 동일 적용.

- 평가는 순전파 한 번이라 비용이 무시할 수준이다 — PySR 적합이 법칙당
  2,400~2,800초인데(2026-09-08 실측, 아래 §4) 조기 종료 평가는 그 앞의 교사
  학습 단계 안에서 epoch 당 forward pass 하나 추가일 뿐이다.
- `epochs=700` 예산에서 70개 평가점 해상도면 A10 이 실측한(수백 epoch
  규모) 표본외 붕괴를 놓치지 않는다.
- 선택 종목 R² 는 `MIN_SELECT_ROWS`(30행)보다 훨씬 큰 표본(최대
  `max_manifold_samples=1500`행)에서 재는 통계라 잡음이 비교적 작지만,
  patience 를 1~2 평가로 너무 짧게 두면 잡음 하나에 멈출 위험이 있다.
  700 epoch 전부를 patience 로 삼으면(사실상 무한) A10 이 관측한 붕괴를
  못 막는다 — 100 epoch(예산의 ~14%)를 절충값으로 정했다.
- **틀렸을 때 비용**: patience 가 너무 짧으면 아직 개선 여지가 있는 모델을
  잡음으로 일찍 죽인다(과소적합 방향 오류) — shallow 처럼 표본내/외 R² 차이가
  원래 작은 교사는 이 위험이 거의 없다. patience 가 너무 길면 A10 이 실측한
  붕괴를 못 막는다(이번 수정의 목적 자체가 무효화). 100 epoch 은 "무한"(700)의
  1/7 수준이라 후자 쪽 위험이 더 크지 않다고 판단했다.

### 1.2 되돌릴 가중치 보관 방식

**최선 시점 스냅샷 하나만 보관**(`copy.deepcopy(state_dict())`, 매 평가에서
개선이 있을 때만 갱신, 이전 스냅샷은 버린다) — 전체 히스토리를 들고 있지
않는다.

- 이 저장소에 "최선 이후 중간 체크포인트를 나중에 다시 본다"는 용도가 없다.
  항상 "지금까지 관측한 최선으로 되돌린다"만 필요하다.
- 모델이 작아(병목 2, 은닉 16~32) 메모리는 어느 쪽이든 문제가 안 되지만,
  최선-only 가 더 단순하고 "어느 스냅샷을 최종으로 쓰는가"에 대한 모호함이
  구조적으로 없다(전체 히스토리를 보관하면 "복원 로직이 최댓값을 못 찾는다"는
  또 다른 뮤테이션 표면이 생긴다).

### 1.3 오염 — 가장 어려운 판단

**선택지 (i)를 골랐다: 오염을 감수하고 기록한다.** 셋(적합/조기종료/채점)으로
쪼개지 않았다.

**근거 (실측, 2026-09-09):**

```
python3 -c "
from sd import config, universe
from sd.e0 import split as e0_split
symbols = universe.slice_symbols(universe.assign_strata(
    universe.liquidity_stats(universe.stock_symbols(config.DATE), config.DATE)), 9)
stratum_of = ...assign_strata 결과의 stratum 열...
sp = e0_split.split_symbols(symbols, stratum_of, seed=0, fit_fraction=2/3)
"
결과: fit 36종목 / select 18종목, 층별 {fit:6, select:3, n:9} × 6층 (전부 동일)
```

`--per-stratum 9`(이번 실행 인자, 직전 실행과 동일해야 한다 — 사전등록 §4
"설정은 결과를 보기 전에 정한다")에서 선택 종목은 **층당 정확히 3개**뿐이다.
셋으로 쪼개면(예: 조기종료 1 / 채점 2, 또는 반대) 한쪽은 **층당 1개**가
된다 — `sd/e0/split.py` 모듈 docstring 이 명시한 분할 설계의 전제("선택
쪽은 층당 최소 2종목을 가능한 한 확보해 종목 하나의 특이값이 판정 전체를
좌우하지 않게 한다", `MIN_MEMBERS_TO_SPLIT=2`)가 정확히 이 지점에서 깨진다.
종목 수를 늘리려면 `--per-stratum` 을 올려야 하는데, 그러면 이번 실행이
"직전 실행과 동일한 인자"가 아니게 되고 사전등록 §4 를 어기게 된다 — 그래서
이번 실행에서는 (ii)를 선택지에서 제외했다(다음 라운드에서 `--per-stratum`
을 올릴 때 재고할 문제로 남긴다).

**자유도 비교.** 조기 종료가 고르는 것은 "이 학습에서 어느 epoch 의 가중치를
쓸까" 하나뿐이다(모델 구조·후보 자체를 고르지 않는다, 평가 지점 최대
70개 중 하나를 고르는 정도의 자유도). 반면 후보 primary 선택(SR 이 낸
10~16개 후보 중 `out_of_sample_score` 최댓값 하나를 고르는 것)은 이미
같은 선택 종목에 노출된 훨씬 큰 자유도다 — A1 에서 "표본 내 채점"을 고친
뒤에도 여전히 선택 종목 위에서 최댓값을 고른다는 점은 동일하다(다만 그
후보들 자체는 선택 종목을 한 번도 못 본 적합 종목에서만 생성됐다). 조기
종료가 추가하는 자유도는 이 기존 구조 위에 "교사가 선택 종목을 한 번 더,
훨씬 좁은 방식으로 본다"는 것 하나가 얹히는 정도라고 본다.

**틀렸을 때 비용.** 이 변경 이후 게이트가 새로 "복원"을 보고하면, 그
복원은 진짜 표본외 일반화가 아니라 교사 조기 종료와 후보 선택 두 층이
동시에 선택 종목의 잡음에 맞춰진 결과일 수 있다 — 액면 그대로 믿지 말고
별도 홀드아웃(다른 날짜, 지금은 `20260317` 이후로 봉인)으로 재확인해야
한다. **특히 L2·L3·L4** 는 L0 직접 측정(200종목·130만 행, 표집 왜곡 없음)
에서 이미 법칙 자체가 이 데이터에 없다고 나왔다(`L0-REPORT.md`) — 이
수정 이후 이 셋 중 하나가 "복원"으로 나온다면 그것은 교사 개선의 성과가
아니라 오염의 신호일 가능성이 높다는 뜻으로 읽어야 한다. 반대로, 게이트가
안 열리는 결과(표본외 R² 음수 붕괴가 사라지는지)는 이 오염으로 설명되지
않는다 — 오염은 지표를 낙관적인 쪽으로만 밀 수 있지 비관적인 쪽으로 밀
근거가 없다(선택 종목을 조기 종료에 한 번 더 노출시키는 것이 모델을
"실제보다 더 나빠 보이게" 만들 메커니즘은 없다).

---

## 2. 검증 — 뮤테이션 자기검토

세 가지 전부 수동으로 코드에 주입 → 관련 테스트만 실행 → 실패 확인 →
되돌림 → `md5sum` 대조로 원상복구 확인. 순서대로 진행(동시에 두 개를
넣지 않았다 — 어느 뮤테이션이 어느 실패를 냈는지 헷갈리지 않기 위해).

| # | 뮤테이션 | 적용 위치 | 결과 |
| --- | --- | --- | --- |
| 1 | 조기 종료 기준을 적합 종목 R² 로 되돌림 | `sd/e0/teacher.py::ScalarTeacher.fit` 의 `evaluate` 클로저를 `X_select/y_select` 대신 `X/y`(적합 데이터)로 교체 | **잡힘.** `test_scalar_teacher_early_stopping_picks_epoch_by_select_r2_not_fit_r2` 실패(최선 시점이 epoch 300 — 적합 R² 는 끝까지 계속 좋아지므로), `test_scalar_teacher_early_stopping_can_actually_trigger_and_stop_before_total_epochs` 실패(`triggered=False`, 전체 epoch 을 다 돎) |
| 2 | 조기 종료가 아예 작동하지 않게(항상 마지막 epoch) | `sd/teacher/early_stopping.py::run_with_early_stopping` 끝의 `set_state(best_state)` 호출을 주석 처리 | **잡힘.** `test_teacher_early_stopping.py` 의 `test_reverts_to_best_epoch_not_last_epoch_when_validation_collapses`·`test_never_triggers_but_still_reverts_to_best_not_final` 2건 + `test_e0_teacher.py` 의 select-R² 궤적 테스트 2건(ScalarTeacher·ScalarDeepLOB), 총 4건 실패 |
| 3 | 되돌릴 가중치를 최선이 아니라 마지막 시점 것으로 | 같은 함수의 `best_state` 갱신 조건(`score > best_score`일 때만)을 지우고 매 평가마다 무조건 갱신 | **잡힘.** 위와 동일한 4건 실패(마지막 평가 시점 상태로 되돌아가 `best_score` 와 실제 선택 R² 재계산값이 2.4~2.9 차이) |

세 경우 모두 되돌린 뒤 `md5sum -c` 로 `sd/teacher/early_stopping.py`·
`sd/e0/teacher.py` 가 뮤테이션 전 체크섬과 정확히 일치함을 확인했다.

**판별력에 대한 소견.** 처음 설계한 테스트 중 다수가 "적합-선택 관계를
정반대로 만든 데이터"(모델이 적합에 맞춰갈수록 선택 R² 가 나빠지는 인공
시나리오, `_adversarial_fit_and_select`)에 의존한다 — 실측(2026-09-09)으로
궤적이 예상대로(첫 평가가 최선, 이후 단조에 가깝게 악화) 나오는지 먼저
확인한 뒤 임계값을 정했다(결과를 보고 나서야 테스트를 쓴 것 아니냐는
우려가 있을 수 있어 명시한다 — 다만 이건 테스트 데이터 자체의 튜닝이지
A10 수정의 실제 실행 결과를 보고 기준을 바꾼 것은 아니다).

전체 회귀: `python3 -m pytest -q`(0902) 통과 건수는 §4 실행 직전에
확정한다(아래 채운다).

---

## 3. 실행 전 시간 산출

2026-09-08 에 **정확히 같은 인자**로 이미 두 번 돌린 기록이 있다
(`runs/e0-shallow700.log`, `runs/e0-deeplob700.log` — A10 수정 전, 이번
작업이 고치는 바로 그 붕괴를 낸 실행):

```
--teacher shallow   : 12,722.4초 (3시간 32분)  게이트 1/5
--teacher deeplob   : 13,169.1초 (3시간 39분)  게이트 0/5
```

PySR 적합(법칙당 2,400~2,800초, `deterministic=True` → `parallelism="serial"`,
단일 스레드)이 전체 시간을 지배한다 — 조기 종료가 교사 학습 epoch 수를
줄이더라도(patience 소진 시) 절약되는 시간은 전체의 일부에 불과하다.
**두 실행 모두 이전과 비슷하거나 그보다 짧을 것으로 예상**하고, 늘어날
근거는 없다(추가 연산은 forward pass 하나 + `deepcopy` 하나, epoch 당
무시할 수준).

**용량 확인(2026-09-09 07:10 UTC 실측):** CPU 64코어, load average
23.8/33.3/33.2(다른 작업자 프로세스 포함), 메모리 여유 348G, 디스크 여유
171G. PySR 은 프로세스당 직렬(단일 스레드)이라 두 실행을 **동시에** 띄워도
CPU 경합이 크지 않다 — 그래서 **순차(총 7~8시간)가 아니라 병렬(총
4~5시간)로 띄운다.**

---

## 4. 실행

전체 회귀(`python3 -m pytest -q`, 0902, 구현 완료 후): **385 passed**
(1065.17초). 직접 영향받는 파일만 다시 별도 실행: **81 passed**(779.48초,
`test_teacher_early_stopping.py`·`test_e0_teacher.py`·`test_e0_runner.py`·
`test_teacher.py`·`test_teacher_deeplob.py`). 뮤테이션 3건은 이 재실행
전에 이미 주입·확인·복구했다(§2) — 즉 이 통과는 뮤테이션이 전혀 없는
깨끗한 코드에서 나온 결과다.

구현 커밋: `0902` `9733242`("feat: 교사 조기 종료를 선택 종목 R² 로").

시작 시각: 2026-09-09 07:28 UTC (두 실행 모두, 병렬 기동).

```
shallow : run_dir runs/e0-20260909T072818Z, PID 699180
          PID 파일 runs/e0-shallow700-t1.pid, 로그 runs/e0-shallow700-t1.log
deeplob : run_dir runs/e0-20260909T072840Z, PID 699797
          PID 파일 runs/e0-deeplob700-t1.pid, 로그 runs/e0-deeplob700-t1.log
```

둘 다 `setsid nohup ... &` 로 새 세션에 분리해 띄웠다(`PPID=1`, `SID`=자기
PID 로 확인 — 이 대화 세션이 끊겨도 안 죽는다). 첫 시도에서 `setsid` 가
내부적으로 fork 해 `$!` 가 잘못된(이미 종료한 부모) PID 를 가리키는 문제를
겪어 중복 프로세스 2개가 떴다 — `pgrep -f`/`ps` 로 실제 살아있는 PID 를
확인해 죽이고 로그 파일을 지운 뒤 다시 깨끗하게 띄웠다(교훈: `setsid` 뒤
`$!` 를 그대로 믿지 말고 `pgrep -f`/`ps -o ppid,sid` 로 재확인할 것).

---

## 5. 실행 결과 (채운다 — 각 실행이 끝난 뒤)

### shallow

*(채운다)*

### deeplob

*(채운다)*

### 표본외 교사 R² — 수정 전/후 대조

| 법칙 | deeplob 수정 전(표본외) | deeplob 수정 후(표본외) |
| --- | --- | --- |
| L1 | −1.720 | *(채운다)* |
| L2 | +0.207 | *(채운다)* |
| L3 | −0.397 | *(채운다)* |
| L4 | +0.020 | *(채운다)* |
| L5 | −13.864 | *(채운다)* |

---

## 6. 사전등록 §3 예측과의 대조

*(채운다)*

---

## 7. 우려사항

- **오염(§1.3)**: 이 실행 이후 어떤 "복원"도 완전한 표본외 증거로 액면
  그대로 받아들이면 안 된다 — 특히 L2·L3·L4.
- **`--per-stratum` 을 올리는 다음 라운드에서는 (ii) 삼분할을 재고해야
  한다** — 이번에 (ii)를 뺀 이유(층당 종목 수 부족)는 `--per-stratum` 이
  커지면 사라진다.
- 조기 종료 hyperparameter(주기 10·patience 10)는 5개 법칙·2개 교사
  전부에 동일하게 고정했다 — 법칙마다 최적값이 다를 수 있지만, 사전등록
  §4("결과를 보기 전에 정하고 그 값으로 간다")를 지키려면 법칙별 튜닝은
  이번 라운드에서 하지 않는다.
