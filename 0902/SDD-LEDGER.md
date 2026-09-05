# SDD ledger — plan: /home/dgu/tick/symbolic/0902/PLAN.md

## Setup

Workspace: 0902/ 자체가 격리 작업공간이다. 상위 `symbolic/` 은 git 저장소가 아니고
0902/ 도 아니었다. 보호할 공유 브랜치가 없다.

Ruling R1: `git init` 을 controller 가 먼저 실행했다 — `sdd-workspace` 가 git 저장소를
요구하는데 그것을 만드는 것이 Task 1 이라 순환이 생긴다. Task 1 의 나머지(디렉터리·
.gitignore·config.py)는 그대로 implementer 몫이고 `git init` 은 멱등이다.
틀렸을 때의 비용: 없음. Task 1 이 다시 실행해도 무해하다.

## 사전 충돌 스캔

### A. 파일을 공유하는 태스크 쌍

| 파일 | 생성 | 이후 수정 | 확인 결과 |
|---|---|---|---|
| `sd/config.py` | T1 | T2 (`CATALOG_HASH` 추가) | 정합. T2 테스트가 `config.CATALOG_HASH` 를 읽고 T2 Step 2 가 그 AttributeError 를 기대 실패로 명시 |
| `vendor/PATCHES.md` | T1 (빈 목록) | T2 (catalog.py·contract.py 등재) | 정합. T1 의 `--check` 는 빈 목록에서 "차이 없음", T2 후에는 두 파일만 예외 |
| `vendor/framework/{catalog,contract}.py` | T1 (복사) | T2 (연산자 2개) | 정합 |
| `sd/compile/__init__.py` | T10 | T11 덮어씀 → T12 덮어씀 | 정합. 각 단계가 직전 export 를 포함해 누적 |
| `sd/teacher/__init__.py` | T8 | T16 덮어씀 | 정합. T16 이 `gate` 를 추가하고 T8 의 두 export 유지 |
| `run_slice.py` | T15 | T16 (게이트 호출·인자·provenance) | 정합. T16 이 앵커로 삼은 `[S1] 교사 학습 완료` print 가 T15 에 실재 |
| `sd/report.py` | T15 | T16 (provenance 2필드) | 정합. 기본값이 있어 T15 테스트가 계속 통과 |
| `pytest.ini` | T3 Step 5 | — | **결함** → Ruling R4 |

### B. 인터페이스 생산↔소비

| 생산 | 소비 | 확인 결과 |
|---|---|---|
| T1 `config.load_framework()`·상수 | T2·4·5·6·13·15·16 | 정합. 반복 호출 멱등 |
| T2 `sqrt`/`tanh` (무차원 입력만) | T11 `translate` 가 두 노드를 방출 | 정합. T11 테스트의 인수가 전부 무차원 feature |
| T3 `stock_symbols`·`liquidity_stats`·`assign_strata`·`slice_symbols` | T15 | 정합 |
| T4 `load_arrays`·`feature_matrix` | T15 | 정합 |
| T5 `labels.build(arrays) -> Labels` | T15·T16 | 정합 (계획 자체검토에서 시그니처 불일치 1건 이미 수정) |
| T6 `transform(matrix, names)` | T15 | 정합 |
| T7 `select(X, mask, max_samples, seed)` | T15 | 정합. `index`·`weight`·`on_manifold` 길이 일치 확인 |
| T8 `ShallowMLP`·`Teacher` | T15·T16 | 정합 |
| T9 `Candidate`·`NaiveBackend.name` | T12·T15 | 정합 |
| T10 `strip_monotone`·`normal_form` | T12 | **중복** → Ruling R3 |
| T11 `translate`·`static_check` + 예외형 | T12 | 정합 |
| T12 `compile_candidates`·`threshold.*`·`PARAMETER` | T13·T15 | 정합. T13 이 `PARAMETER` 를 `.compile.pipeline` 에서 직접 import |
| T13 `replay.run`·`attempt_count` | T15 | 정합 |
| T14 `summarise`·`rank` | T15 | 정합. 원장 컬럼 `contract_id/status/net_bps/cohort` 는 스파이크에서 실재 확인, status 값도 대문자 3종 일치 |
| T15 `report.provenance`·`write` | T16 | 정합 |

### C. 태스크 자체 정합 (텍스트 vs 테스트 vs 코드)

| 태스크 | 확인 결과 |
|---|---|
| T1 | 정합. `vendor_sync --check` 가 PATCHES.md 목록만 예외 처리 |
| T2 | 정합. 무차원 강제 동작을 실제 vendor 에서 미리 재현해 확인 |
| T3 | 실측치(2570·2507·837/834/836·경계 18.37) 가 테스트 기대값과 일치 |
| T4 | 정합 |
| T5 | **성능 결함** → Ruling R2 |
| T6 | 정합 |
| T7 | 정합 |
| T8 | 정합 |
| T9 | 정합 |
| T10 | 정규형 환원 7개 기대값을 실제 sympy 로 전수 확인 |
| T11 | 정합. `sympy.sqrt` 가 함수인 문제는 계획 자체검토에서 이미 수정 |
| T12 | 정합 (R3 제외) |
| T13 | 정합. `run_backtest` 시그니처는 스파이크에서 실호출 확인 |
| T14 | 정합 |
| T15 | 정합. `to_markdown` 이 요구하는 tabulate 0.9.0 설치 확인 |
| T16 | 정합 |

### 사전 스캔 rulings

Ruling R2 (Task 5): `_fill_label` 의 순수 파이썬 이중 루프는 O(n·W) 다. `005930` 은
호가틱 48만이라 최악의 경우 종목당 수 분이 걸린다. **`numba.njit` 로 감싼 헬퍼로
바꾼다** (numba 0.61 설치 확인). 근거: 이 함수는 교사 체결 헤드의 유일한 학습 신호라
빼거나 성기게 만들 수 없고, 의미는 그대로 두고 속도만 바꾸는 것이 가장 작은 변경이다.
틀렸을 때의 비용: njit 이 이 코드를 못 받으면 구현자가 벡터화로 우회해야 한다 — 한 라운드.

Ruling R3 (Task 12): `pipeline.compile_candidates` 가 정규형 키를
`sympy.srepr(sympy.simplify(stripped))` 로 다시 만들지 말고 **`normalize.normal_form(
candidate.expr)` 를 호출**한다. 근거: T10 이 그 함수를 이미 정의했고, 같은 산술을 두
곳에 두면 한쪽만 고쳐진 채 조용히 갈라진다 (프레임워크가 `metrics.py` 에서 겪은 일).
틀렸을 때의 비용: 없음. 두 표현은 같은 값을 낸다.

Ruling R4 (Task 3): `pytest.ini` 생성을 Step 5 에서 **Step 1 로 앞당긴다**. 근거:
Step 4 가 `-m "not slow"` 를 marker 등록 전에 실행해 경고가 난다. 기능 문제는 아니지만
구현자가 경고를 실패로 오인할 수 있다.
틀렸을 때의 비용: 없음.

## 실행

Task 1: dispatched (sonnet). BASE=4b825dc642cb6eb9a060e54bf8d69288fbee4904 (빈 트리 — 첫 커밋 전)
Task 1: implementer DONE (커밋 7172095, 4 passed, 우려 없음)

Ruling R5 (Task 1 리뷰 범위): 전체 차분이 2.5MB 였고 그중 122개가 `vendor/framework/**`
— 상위 저장소에서 **그대로 복사한 사본**이다. 리뷰 패키지에서 그 경로를 제외하고
9.5KB 로 좁혀 dispatch 했다. 근거: 사본의 내용은 우리가 쓴 코드가 아니라 리뷰 대상이
아니며, 원본 일치는 `vendor_sync.py --check` 와 `test_vendor.py` 가 파일별 sha256 으로
기계 검증한다. 2.5MB 를 읽히면 리뷰어가 정작 우리 코드 6개 파일을 못 본다.
틀렸을 때의 비용: 사본에 숨은 변경이 있어도 리뷰어가 눈으로 못 잡는다 — 다만 sha256
검사가 그것을 잡으므로, 남는 위험은 PATCHES.md 에 등재된 파일(현재 0개)뿐이다.
리뷰어에게 그 취약점을 직접 물었다.

Task 1: 리뷰 — 사양 ✅ / 품질 변경 필요. Important 2 · Minor 2. 둘 다 plan-mandated.

Ruling R6 (Task 1 Important 2건 — 수정한다): 스펙(DESIGN.md D2)의 불변식은
"원본과의 차이는 PATCHES.md 에 적힌 것뿐이어야 하고, **테스트가 그것을 검사한다**"
이다. 리뷰어가 실측한 두 우회 경로는 그 불변식을 정면으로 깬다.
  (1) 매니페스트에 없는 **새 파일**을 vendor 에 넣어도 `--check` 가 통과한다.
      새 파일은 원본과의 차이이고 PATCHES.md 에 없다 → 불변식 위반.
      게다가 `load_framework()` 가 vendor 를 sys.path 맨 앞에 꽂으므로 로드 경로가 열려 있다.
  (2) PATCHES.md 등재 파일은 **존재 여부조차** 검사하지 않는다. 삭제해도 통과했다.
      Task 2 가 catalog.py·contract.py 를 등재하는 순간 이것이 실제 위험이 된다 —
      Task 2 가 동결하는 `catalog_hash()` 는 catalog.py 의 OPERATORS 만 덮고
      contract.py 의 런타임 메서드는 아무도 안 본다.
계획서가 그 코드를 시킨 것은 맞지만, 스펙이 구속력을 갖는다. 지금 고치지 않으면
Task 2 부터 load-bearing 이 된다.
수정 범위: (a) DEST 독립 스캔으로 미등재 파일 보고, (b) 등재 파일도 존재 확인 +
`patched_sha256` 항목이 있으면 일치 확인 (Task 2 가 그 항목을 채운다).
틀렸을 때의 비용: `vendor_sync.py` 30줄이 늘어난다. 되돌리기 쉽다.

Task 1: minor (deferred): tests/test_vendor.py 의 `import pytest` 미사용 (F401 소지)
Task 1: minor (deferred): 저장소 루트에서 인자 없이 `pytest` 를 돌리면
  vendor/framework/tests/ 25개가 수집돼 ModuleNotFoundError. 경로를 명시하는
  `make test`/`pytest tests/` 에는 영향 없음.

Ruling R7 (R4 개정): Task 3 의 `pytest.ini` 에 `norecursedirs = vendor .superpowers`
를 함께 넣는다. 근거: 위 deferred minor 를 별도 라운드 없이 없애고, Task 1 과
Task 3 이 같은 파일을 두고 다투지 않게 한다.
틀렸을 때의 비용: 없음.

Task 1: fix round 1/5 — dispatched (FIX_BASE=7172095)
Task 1: fix round 1/5 (2 addressed, 0 open; commits 7172095..0c3d206)
  재리뷰가 fault injection 으로 F1·F2 해소를 실측했고, 신규 테스트 3개가 수정을
  되돌리면 실제로 FAILED 로 떨어지는 것까지 확인했다.
Task 1: complete (commits 7172095..0c3d206, review clean)

Task 2: dispatched (sonnet). BASE=0c3d206
  carry-forward: Task 1 이 만든 `patched_sha256` 는 아직 빈 dict 다. Task 2 가
  catalog.py·contract.py 를 PATCHES.md 에 등재하므로 **그 두 파일의 패치 후 sha256 을
  반드시 채워야** F2 방어가 완전해진다. 안 채우면 존재 확인만 하는 약한 모드로 남고,
  contract.py 의 `_op_sqrt`/`_op_tanh` 가 조용히 바뀌어도 아무도 못 잡는다
  (catalog_hash() 는 OPERATORS 만 덮는다).
Task 2: implementer DONE (커밋 e26f428, 16 passed, CATALOG_HASH=73344500dc6b2d7f)
  patched_sha256 채움 확인. PROFILE_HASH d0b91a03541c0cec 불변 재확인.

Ruling R8 (CLAIM_OPS 는 건드리지 않는다): 구현자가 `catalog.CLAIM_OPS` 에 `log1p` 는
있는데 `sqrt`·`tanh` 는 없다고 보고했다. 확인 결과 그 상수는 `workflows/
free_research.py` 세 곳에서만 쓰인다 — 프레임워크 자체의 **Agent 주도 연구 경로**다.
우리 `sd/` 는 `canonical.run_backtest` 를 직접 부르고 `free_research.py` 를 import
하지 않으므로 **실행 경로 밖**이다.
게다가 그대로 두는 편이 더 옳다: DESIGN.md 가 `free_research.py` 를 E1 비교군 (g)
"Agent 산 진입식" 으로 쓴다고 적어 두었는데, (g) 는 프레임워크 자신의 기준선이므로
우리가 증류를 위해 넓힌 어휘를 **조용히 물려받으면 안 된다**. 같은 자로 재려면
기준선의 어휘는 원래대로여야 한다.
틀렸을 때의 비용: 비교군 (g) 를 만들 때(이 계획 범위 밖) Agent 가 claim 수준 식에서
두 연산자를 못 쓴다 — 그것이 의도다. 뒤집으려면 한 줄이다.
Task 2: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 2.
  리뷰어가 patched_sha256 을 {} 로 되돌려 "약한 모드" 회귀를 재현했고 신규 테스트가
  정확히 FAILED 로 잡는 것을 확인. 무차원 가드 제거도 FAILED 재현.
  상위 저장소 무손상을 `git -C proj_claude_tick_finance diff` 빈 결과로 확정.
  리뷰어가 독립적으로 CLAIM_OPS 를 "브리핑 미지시 → 건드리지 않음" 으로 판단 —
  Ruling R8 과 같은 결론에 도달했다.
Task 2: minor (deferred): `_op_sqrt` 의 음수 가드를 제거해도
  test_sqrt_of_negative_is_nan_not_exception 이 FAILED 로 안 떨어진다. numpy 기본
  동작이 이미 NaN 을 내기 때문. 사양 요구는 어느 쪽이든 충족되고 가드는 방어적
  여분(경고 억제)이라 비하중. 판별력 있는 테스트로 바꾸려면 filterwarnings=error 필요.
Task 2: complete (commits 0c3d206..e26f428, review clean)

Task 3: dispatched (sonnet). BASE=e26f428
Task 3: implementer DONE_WITH_CONCERNS (커밋 89daea3, 21 passed, 층 837/834/836 일치)
  우려: 층화 통계가 브리핑 기대 8초 대비 실측 108~128초 (15배).
  구현자 진단: GIL 경합 + 무관 프로세스의 CPU 점유.

Controller 실측 (구현자가 놓친 원인): `_one_symbol_stats` 가 **종목마다 glob** 을 돈다.
  300종목 표본에서 glob-per-symbol 2.93s vs 사전 경로맵 0.04s → **78배**.
  2570종목 환산 시 순수 디렉터리 스캔만 25초. 3,726개 파일 폴더를 2,570번 훑는다.
  정본 `modules/common_stock_cache.py` 가 바로 이 문제를 주석으로 남겨 두었다:
  "전종목 반복에서는 date/ 폴더를 symbol마다 glob 하는 비용이 커진다".
  머신 부하도 실재한다 — load average 10.88, python 프로세스 3개가 각 ~110% CPU.
  → 내가 계획서에 쓴 8초는 **내 벤치마크 스크립트**(경로맵 사전 구축) 수치였고,
    계획서 코드는 다른 구현이다. 문서의 수치가 오도한다.
Task 3: 리뷰 — 사양 ✅ / 품질 변경 필요. Important 2 · Minor 3. 둘 다 plan-mandated.

Ruling R9 (glob-per-symbol 을 지금 고친다): 리뷰어가 독립 측정으로 178배를 확인했다
(내 측정 78배와 같은 자릿수). 구현자가 `pa.set_cpu_count(1)` 로도 108초가 유지되는
반증을 스스로 보고하고도 GIL 로 재귀속한 것을 리뷰어가 짚었다.
"Task 15 착수 전에" 가 아니라 **지금** 고친다. 근거: (1) 5줄짜리 변경이고 지금 그 파일을
열고 있다, (2) 미루면 Task 15 구현자가 알려진 결함을 물려받는다, (3) 이 함수는 매 실행
2,570 종목에 불리므로 실험 반복마다 100초씩 누적된다.
틀렸을 때의 비용: 경로맵 주입이 인터페이스를 하나 바꾼다 — 되돌리기 쉽다.

Ruling R10 (무방비 테스트 2개를 보강한다): 뮤테이션으로 실증됐다.
  (a) `slice_symbols` 의 `sorted(...)` → `reverse=True` 로 바꿔도 4개 테스트 전부 통과.
      `picked == sorted(picked)` 단언이 공허하다 — 반환값이 항상 재정렬되기 때문.
      브리핑이 "층마다 issue_code 오름차순 앞에서" 를 명시했는데 아무도 검사하지 않는다.
  (b) `assign_strata` 의 `median()` → `mean()` 으로 바꿔도 전부 통과. 픽스처 둘 다
      대칭이라 두 통계량이 같은 분할점을 낸다.
근거: **실패할 수 없는 테스트는 없는 테스트보다 나쁘다** — 거짓 확신을 준다.
층화는 이후 실험의 횡단면 외삽 축이라 분할 규칙이 조용히 틀리면 H5 판정이 통째로 오염된다.
틀렸을 때의 비용: 테스트 2개 추가. 없다.

Task 3: minor (deferred): `sd/universe.py` 의 `import os` 미사용, `FRICTIONS` 상수 미사용
Task 3: minor (deferred): 브리핑 Step 5 의 "Expected: 2 passed" 오기 — 실제로는
  "1 passed, 4 deselected" 가 맞다. 구현자가 추측으로 테스트를 늘리지 않은 판단은 적절.
Task 3: fix round 1/5 — dispatched (FIX_BASE=89daea3)
Task 3: fix round 1/5 (3 addressed, 0 open; commits 89daea3..2f6f510)
  재리뷰가 뮤테이션 2건을 직접 넣어 새 테스트가 정확히 FAILED 로 떨어지는 것을 확인.
  실측 재현: workers=32 → 9.79s·10.35s, workers=1 → 28.94s. 역전 해소(2.9배 정상 스케일링).
  정확도 회귀 없음: rows=2507, L837/M834/H836, L경계 18.365472910927455.
Task 3: complete (commits e26f428..2f6f510, review clean)

Controller: DESIGN.md §2.3 과 계획서 §8 의 "7.7초" 를 실측값(약 10초)으로 정정했고,
  왜 틀렸는지(어느 구현의 값이었는지)를 함께 남겼다.

Task 3: minor (deferred): `stock_symbols` 와 `liquidity_stats` 를 순차 호출하면
  `_scan_symbol_paths` 가 폴더를 2번 스캔한다. O(N×M) 대비 미미하지만 맵을 공유하면
  더 줄일 수 있다. 재리뷰의 범위 밖 관찰.

Task 4: dispatched (sonnet). BASE=2f6f510
Task 4: implementer DONE (커밋 100acae + c80c444, 28 passed, 005930 feature 36열)
  요청한 뮤테이션 자기검토를 실제로 수행. 브리핑의 3개 테스트가 두 규칙을 못 지킨다는
  것을 스스로 발견하고 보강했다 — (1) 체결 없는 종목-일에서 "0으로 채우지 않고 열을 뺀다",
  (2) `cache_root=None` 강제. 005930 은 체결이 있고 캐시도 없어서 이 환경의 우연으로
  가려져 있었다. 체결 행 0인 실제 종목(292770)을 찾아 쓰고, `_data.load` 를 monkeypatch 했다.
  → 구현자 단계에서 뮤테이션을 돌리게 한 지시가 실제로 라운드를 줄였는지 리뷰가 판정한다.
Task 4: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 2.
  리뷰어가 구현자의 뮤테이션 주장 2건을 독립 재현했고, 두 번째 커밋을
  "브리핑 침범이 아니라 정당한 보강" 으로 판정했다 — 새 요구사항이나 새 프로덕션
  코드가 아니라 브리핑 문서가 이미 명시한 설계 결정을 검증하는 테스트이므로.
  → 구현자 단계 뮤테이션 지시가 리뷰 라운드를 실제로 줄였다. 이후 태스크에도 유지한다.
Task 4: minor (deferred): 열-순서 검증이 취약하다. 무관한 두 열을 뒤바꿔도 통과한다.
  다만 구현이 단일 컴프리헨션(`computed[name] for name in names`)이라 **구조적으로**
  desync 가 불가능하다 — 코드는 안전하고 테스트가 그것을 증명하지 못할 뿐.
Task 4: minor (deferred): `names = tuple(sorted(computed))` 재정렬이 현재 항상 no-op.
  `FEATURE_ORDER` 가 이미 정렬돼 있고 `compute_features` 가 순서를 보존하기 때문.
  방어적 코드로 무해하나 그 줄의 의도를 뒷받침하는 테스트가 없다.
Task 4: complete (commits 2f6f510..c80c444, review clean)

Task 5: dispatched (sonnet). BASE=c80c444
  Ruling R2 를 브리핑 외 지시로 전달한다 — `_fill_label` 순수 파이썬 이중 루프를 numba njit 으로.
Task 5: implementer DONE (커밋 09e56de, 36 passed)
  njit 전환 성공 — `_fill_label` on 005930: 3.58초 → 0.085초 (42배).
  두 구현의 출력을 `np.array_equal` 로 전수 대조해 동일함을 확인. njit 거부 없음.
  뮤테이션 자기검토 5라운드 전부 테스트가 잡음.

Controller: 구현자가 내 수치 오류를 잡았다. 실측 재확인 결과 **두 군데 틀렸다**.
  (1) `005930` 호가틱은 48만이 아니라 **193,602**. 48만(482,280)은 전체 행이고
      그중 호가행 201,080 · 체결행 281,200 — 삼성전자는 **체결행이 호가행보다 많다.**
      정규장 필터 후 193,602.
  (2) 전종목 "약 2,860만 호가틱" 도 과대. 실제 정규장 유효 호가틱 합계는
      **23,091,003 (약 2,310만)**. 25종목 표본의 호가행 비율 76.5% 를 전체 행에
      곱해서 얻은 값이었는데 그 표본이 편향됐다 (거래 활발한 종목일수록 체결행 비중이 크다).
  → 계획서 12곳 + DESIGN.md 2곳 정정. 유도식("76.5% 곱하기")을 지우고 직접 센 값으로 대체.
  이번 세션에서 내 측정 오류가 두 번째다(앞은 glob). **표본에서 전체를 곱해 추정하지 말고 센다.**
Task 5: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 2.
  리뷰어가 njit 등가성을 독립 재현했다 — 순수 파이썬 버전을 되살려 005930(193,602틱),
  292770(체결 0건), 합성 경계(NaN 밀집·sell_min==bid1 등호·0/음수 혼입)에서
  `np.array_equal` 전수 대조 모두 True. 뮤테이션 4종도 직접 재현.
  회계 확인: 스프레드가 분자에 없고 분모(`spread_bps + FEE_BPS`)에만 있다 — 이중 차감 없음.
Task 5: minor (deferred): `sd/labels.py:97` docstring 의 "005930 은 호가틱이 48만" 이
  stale. 실측 193,602. **최종 리뷰에서 반드시 고칠 것** — 코드 주석에 남은 거짓 사실이다.
Task 5: complete (commits c80c444..09e56de, review clean)

## 상위 저장소 드리프트 조사 (리뷰어가 발견)

리뷰어가 `proj_claude_tick_finance` 에 커밋 안 된 변경 46파일을 보고했다. 조사 결과:

- **우리가 건드린 것이 아니다.** `git status --short -- Sandbox_12/framework` → 0건.
  46파일은 docs 목업·.coverage·.gitignore 이고 mtime 이 전부 어제(09-02)다.
- **그러나 상위가 우리 복사 이후 실제로 움직였다.** vendor 복사 시각 09-03T04:21 이후
  상위에 커밋 2개가 들어왔다 — 5529302(04:54), 55c37fb(05:52).
  그 커밋들이 `agents/runtime.py` · `tests/test_free_research.py` ·
  `workflows/free_research.py` 세 파일의 내용을 바꿨다.
- 검증: 세 파일 모두 **상위 현재 ≠ 매니페스트 기록**, **vendor 사본 == 매니페스트 기록**.
  즉 우리 스냅샷은 온전하고 상위만 앞으로 갔다. 나머지 119개 파일은 드리프트 없음.
- 세 파일 모두 `sd/` 가 참조하지 않는다 — 실행 경로 밖이다 (Ruling R8 참조).

Ruling R11 (조치 없음, 기록만): **이것이 D2(vendor 복사)를 정당화한 사건이다.**
D2 의 근거가 "Sandbox_12 는 활성 저장소라 라이브 import 는 실험 도중 코드 해시를
움직인다" 였는데, 복사 90분 만에 그 일이 실제로 벌어졌다. 라이브 import 였다면
Task 3~5 가 Task 1~2 와 다른 코드 위에서 돌았을 것이고 계획서 §5 의 재현성 요구가
깨졌을 것이다. 사본을 갱신하지 않는다 — 갱신하는 순간 같은 문제가 생긴다.
틀렸을 때의 비용: 상위의 개선(임계 규칙 문서화, variant 최소 2개 강제 등)을
이 실험에서 못 받는다. 실험이 끝난 뒤 `make vendor-diff` 로 확인하고 의도적으로 옮긴다.

Task 6: dispatched (sonnet). BASE=09e56de
Task 6: implementer DONE (커밋 429520a, 41 passed, 무차원 feature 9개)
  뮤테이션 자기검토가 브리핑 테스트의 실제 맹점을 잡았다 — 위치 인덱싱 버그
  (`names.index()` 무시)가 열 0만 검사하는 테스트를 우연히 통과했다. 열 1을 인접
  오답 열과 대조하도록 보강해 이제 잡힌다. ValueError 분기와 `dropped` 내용도 보강.

Controller 관측 → Ruling R12 (기록만, 조치 없음): 무차원 feature 가 9개뿐이고
  **가격 움직임(`return` 6개)과 OFI(`quantity` 안 3개)가 통째로 빠진다.**
  남는 9개는 전부 호가창 상태·흐름 불균형이다. 그래서 슬라이스의 SR 은 계획서 §3 S3 이
  앞세운 두 무차원 그룹 — `OFI / Q̄` 와 `ΔP / s` — 을 **표현조차 할 수 없다.** 둘 다
  파생량이고 슬라이스는 파생을 미뤘기 때문이다.
  결함이 아니라 D3 의 결과다. 다만 슬라이스 결과를 방법론 증거로 읽으면 안 되는 근거가
  하나 더 늘었다 — 최소 교사·최소 SR 에 더해 **어휘 자체가 가설을 담지 못한다.**
  DESIGN.md 에 D14 로 기록하고 다음 단계(파생 무차원 열)의 구체적 형태를 적어 뒀다.
  `return` 을 무차원으로 승격하지 않는 이유도 명시: bps 수익률은 물리적으로 무차원이지만
  종목 간 비교 가능하지 않다 — 스프레드 5bp 종목의 10bp 이동과 50bp 종목의 10bp 이동은
  다른 사건이다. 승격이 아니라 비를 만들어야 한다.
  틀렸을 때의 비용: 없다. 슬라이스 범위를 넓히지 않았고 기록만 했다.
Task 6: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 2.
  리뷰어가 브리핑 원본 테스트가 위치 인덱싱 뮤테이션을 실제로 통과해버림을 재현했고,
  구현자가 강화한 어서션이 그것을 잡는 것을 확인했다. Task 4 리뷰가 지적한 약점이
  Task 6 에서 재발했다가 구현자 단계에서 진단·수정됐다.
  실 틱 데이터로 (193602, 36) → (193602, 9), dropped 27, 9개 열 전부 일치 독립 재현.
Task 6: minor (deferred): `dropped` 컴프리헨션이 Catalog 밖 이름에 KeyError.
  `ticks.feature_matrix` 계약이 지켜지는 한 발생하지 않음. 브리핑 코드 그대로.
Task 6: minor (deferred): 새 테스트 하나만 키워드 인자 스타일 (사소한 불일치)
Task 6: complete (commits 09e56de..429520a, review clean)

Ruling R13 (선형종속을 지금 고치지 않고 거리 계산을 견디게 만든다):
  실측 — `book_imbalance = 2 × queue_imbalance_best − 1` 이 기계 정밀도(1.11e-16)로
  성립한다. r=+1.000000, 공분산 rank 8/9 로 **특이**다. 실효 어휘는 9가 아니라 8.
  `dimensionless.py` 에서 고치지 않는 이유: 그 모듈은 Catalog 의 `dimension` 만 보는
  얇은 필터다. 어느 feature 가 다른 것의 아핀 변환인지는 Catalog 가 말해 주지 않으므로
  판단하려면 feature 관계표를 새로 만들어야 하고, **그 표가 곧 두 번째 진실 원천이 된다.**
  대신 (1) Task 7 의 마할라노비스를 rank 결손에 견디게 만들고, (2) 사실을 DESIGN.md D14
  에 기록해 SR 단계가 알고 있게 한다. SR 쪽 영향은 제한적이다 — 둘의 아핀 결합은
  단조 흡수(S3)가 임계로 걷어내므로 진입식 수준에서 새 구조가 되지 않는다.
  틀렸을 때의 비용: SR 이 중복 차원을 하나 더 뒤진다 (탐색 시간만 낭비).

Task 7: dispatched (sonnet). BASE=429520a
Task 7: implementer DONE (커밋 0a6e600, 50 passed) — **inv → pinv 무조건 전환**

  실측(005930): `np.linalg.inv` 가 **예외를 던지지 않고** 최대 절대값 1.27e32 의 쓰레기를
  반환했다. 마할라노비스 거리가 최대 1.77억까지 폭주했고 신뢰 반경 0.99 분위가
  **1.15억** 이라는 무의미한 값이 됐다. `pinv` 는 같은 데이터에서 최대 절대값 2,172,
  거리 0.27~32.6 의 유한하고 합리적인 값을 낸다.

  → 브리핑의 `try: inv / except LinAlgError: pinv` 폴백이 **한 번도 발동하지 않는다.**
    신뢰 반경이 1.15억이면 모든 점이 on-manifold 로 판정되고 **S2 단계 전체가 no-op** 이
    된다. 아무 예외도 없고 아무 경고도 없이. 계획서 C1(on-manifold 샘플링)의 기여가
    통째로 사라지는데 그것을 알 방법이 없다.

  Ruling R13 의 지시(rank 결손에 견디게)가 정확했고, 구현자가 실측으로 확인해 고쳤다.

  구현자 우려(타당함, 이월): 이 버그는 **합성 데이터로 안정적으로 재현되지 않는다** —
  inv 원소가 거대해도 실제 관측 벡터와의 정렬에 따라 이차형식이 우연히 상쇄될 수 있다.
  실제로 잡아낸 것은 005930 실데이터 회귀 테스트(`@pytest.mark.slow`) 하나뿐이다.
  **그 slow 테스트를 스위트에서 빼면 안전망이 사라진다.**

  뮤테이션에서 또 하나: 브리핑의 "같은 seed → 같은 결과" 테스트만으로는
  **"seed 를 무시해도 통과"** 하는 거짓 안전이 재현됐다. 구현자가
  `test_different_seeds_actually_change_the_subsample` 을 추가해 메웠다.
Task 7: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 1.
  리뷰어가 세 주장을 전부 독립 재현했다 (소수점까지 일치):
    rank 8/9, 조건수 1.18e17, inv 최대 1.2659596788281945e+32,
    d_inv q99 = 115,291,555  vs  d_pinv q99 = 9.26
  `pinv` 를 `inv` 로 되돌리면 오직 slow 회귀 테스트 하나만 FAILED
  (`assert 176802312.375 < 1000.0`) — "유일한 안전망" 주장도 사실로 확인.
  뮤테이션 4종 독립 재현, 매번 md5 로 원복 확인.
  리뷰어 관찰: DESIGN.md 가 이 문제를 이미 예견해 두었다("공분산이 특이해 inv 가
  못 쓰인다. 유사역행렬로 물러나야 한다"). 브리핑의 try/except 가드가 그 설계 의도를
  달성하지 못했음을 실측이 밝혔다.
Task 7: minor (deferred): `WEIGHT_CLIP=10.0` 이 **죽은 상수**다.
  `1.0 + 4.0*ranks` 는 ranks∈[0,1] 이라 최대 5.0 이므로 clip 이 절대 발동하지 않는다.
  브리핑 원본 코드의 결함. 동작은 틀리지 않았고(가중치 범위 [1,5] 는 의도대로 작동)
  상수와 주석의 의도만 어긋난다. Task 9(SR)가 가중치를 소비하므로 그때 재확인.
Task 7: complete (commits 429520a..0a6e600, review clean)

Task 8: dispatched (sonnet). BASE=0a6e600
Task 8: implementer DONE (커밋 e62dd2d, 58 passed)
  구현자 우려 3건(전부 타당, 이월):
  (1) `fit()` 안의 `torch.manual_seed` 가 **죽은 코드**다. 재현성은 전부 `__init__` 에서
      온다 — 이 아키텍처에 학습 루프 확률 연산(dropout·셔플)이 없기 때문. **DeepLOBCompact
      으로 교체하면 이 가정이 깨진다.** `Teacher` 프로토콜이 시딩을 강제하지 않는다.
  (2) `l1=1e-4` 가 손실에 배선된 것은 뮤테이션으로 확인됐지만, 그 크기가 H4(병목 필요성)
      목표에 맞는지는 검증 안 됨 — 범위 밖.
  (3) `fit()` 이 전체를 한 배치로 텐서화한다. 슬라이스 규모에서는 괜찮으나 실제 교사로
      바꾸면 메모리 문제가 될 수 있다.
Task 8: 리뷰 — 사양 ✅ / 품질 승인. Important 1(지금은 문제 아님), Minor 3.
  리뷰어가 5개 뮤테이션과 구현자 우려(1)을 전부 독립 재현. `fit()` 의 두 번째
  `torch.manual_seed` 를 삭제해도 8개 전부 통과 → 죽은 코드 확정. 다만 브리핑 코드
  자체에 있던 것이고 DeepLOB 교체 시점의 문제이므로 지금 고칠 근거 없음으로 판정.
  선형종속 열에서 표준화·학습이 안 깨짐을 직접 시뮬레이션으로 확인.
  GPU 미사용 확인 (`torch.cuda.is_available()==True` 인데도 안 끌어들임).

  ⚠️ 리뷰어 지적 — **내 브리핑 테스트 설계가 같은 맹점을 반복 생산하고 있다.**
  "같은 seed → 같은 결과" 만 보는 테스트가 Task 7 과 Task 8 에서 연속으로
  "seed 를 무시해도 통과" 했다. 재현성 테스트를 쓸 때는 **다른 시드가 다른 결과를 내는지**
  를 반드시 짝으로 넣어야 한다. 남은 태스크 브리핑에 이 경계를 명시한다.
Task 8: complete (commits 0a6e600..e62dd2d, review clean)

Task 9: dispatched (sonnet). BASE=e62dd2d
Task 9: implementer DONE (커밋 0deb748, 67 passed)
  성능 실측: 현실 규모(20만행 × 9feature)에서 fit 약 5.0초, 원시 후보 58개 → 12개 반환.
    78% lambdify+평가+lstsq, 18% simplify 중복병합, 4% 랭킹. 병목 아님.
  선형종속 쌍: `_rank_features` 상관 랭킹에서 정확히 동점(차이 ~3e-16), 알파벳순으로
    결정론적 tie-break, **둘 다 상위 4개에 든다.** `linear2`/`diff2` 가 그 쌍에 걸리면
    한 변수의 아핀 변환이 되고 점수도 같다 — 후보 풀의 **가짜 다양성**. 치명적이지 않다
    (컴파일러의 단조 흡수가 걷어낸다).
  뮤테이션이 브리핑 테스트의 실제 맹점 2건을 또 잡았다:
    (1) `Candidate.seed=0` 하드코딩이 5개 테스트를 전부 통과 — 비기본 시드를 검사하는
        테스트가 없었다. **내가 미리 경고한 그 함정이 이번엔 seed 필드 쪽에서 나왔다.**
    (2) `weighted_r2` 가 `w` 를 통째로 무시해도 통과 — 주어진 테스트가 전부 균일 가중치.
    (3) `complexity_of` 가 상수를 반환해도 통과.
    셋 다 추가 테스트로 메움.

Task 9: minor (deferred): `_rank_features` 가 `w` 를 인자로 받고 **쓰지 않는다.**
  docstring 은 "단변량 가중 상관" 이라 하는데 구현은 순수 `np.corrcoef` 다.
  → **최종 리뷰 수정 목록**: 이 docstring 과 `sd/labels.py:97` 의 "48만" 둘 다
    코드 주석에 남은 거짓 진술이다. 함께 고친다.
Task 9: 리뷰 — 사양 ✅ / 품질 승인. Important 1(plan-mandated), Minor 2.
  리뷰어가 뮤테이션 6개를 전부 독립 재현(표까지 일치). 선형종속 쌍의 상관 차이가
  `0.000e+00` 로 완전 동일하고 세 후보가 같은 점수(0.899582)를 내는 것도 재현.
  심볼 오염 메커니즘(lambdify → object dtype → float 캐스트 TypeError → except 삼킴)도
  직접 재현해 "우연한 가드" 판정에 동의.

  ⚠️ 성능 주장은 **미확인(inconclusive)** 으로 남았다. 리뷰어 측정은 92~93초로
  구현자 보고(5.0초)의 18배였다. 다만 리뷰어가 원인을 조사해 **다른 세션의 대형 동시
  작업**(proj_claude_tick_finance 의 `framework research` optuna 캠페인, workers 6~8,
  optuna-processes 3~4)으로 load average 12~14 였음을 `ps aux`·`uptime` 으로 확인했고,
  개별 연산 단가(lstsq 14ms, simplify 14.7ms, lambdify+평가 18.9ms)는 보고서와 거의
  정확히 일치했다. 그래서 회귀 근거로 채택하지 않되 5.0초도 확인으로 판정하지 않았다.
  → 이 판단이 옳다. 조용한 머신에서 재측정할 기회가 있으면 확정한다.

Ruling R14 (`_rank_features` 의 `w` 미사용을 **지금 고친다**):
  리뷰어가 Important 로 매기고 "다음 태스크로 이월" 을 권했으나, 이월하지 않는다.
  근거 셋.
  (1) **스펙 위반이다.** 계획서 §3 S2 ③ 이 "가중치를 SR 손실에 **명시적으로** 넘긴다.
      암묵적 리샘플링은 나중에 무엇을 최적화했는지 알 수 없게 만든다" 고 못 박았다.
      `weighted_r2`(채점)는 가중치를 쓰는데 `_rank_features`(feature 선택)는 안 쓴다 —
      **탐색의 첫 관문에서 원칙이 무너진다.**
  (2) **C1 의 기여를 정면으로 되돌린다.** on-manifold 표집의 요점이 "드물지만 경제적으로
      중요한 꼬리 상태에 가중치를 준다" 인데, 꼬리에서만 신호를 내는 feature 가
      비가중 상관으로 상위 4개에서 탈락하면 SR 은 그 feature 를 **아예 보지 못한다.**
  (3) **이월할 곳이 없다.** 남은 태스크 10~12 는 컴파일러라 `sr/` 를 건드리지 않는다.
      "다음 태스크" 는 실질적으로 존재하지 않는다.
  덤: docstring 이 이미 "단변량 가중 상관" 이라고 주장하므로, 고치면 코드가 자기 계약과
  일치하게 된다.
  틀렸을 때의 비용: 3줄 + 테스트 하나. 사소하다.

Task 9: minor (deferred): 심볼 제약이 `except Exception` 의 우연한 가드에 의존.
  PySR 어댑터 작성 시 `free_symbols ⊆ allowed` 명시 검증을 반드시 넣을 것.
Task 9: fix round 1/5 — dispatched (FIX_BASE=0deb748)
Task 9: fix round 1/5 (1 addressed, 0 open; commits 0deb748..8e82ff1)
  재리뷰가 가중치 무시 뮤테이션으로 새 테스트가 정확히 떨어지는 것을 확인
  (`assert 'weak_broad_feature' == 'tail_only_feature'`), 나머지 9개는 영향 없음.
  가중 상관 수식이 균일 가중치에서 `np.corrcoef` 와 일치(0.9606981088731766 vs …69).
  실제 아핀 종속 쌍이 비균일 가중에서도 가중 |상관| 완전 동일(0.8963628653798377)하고
  알파벳순으로 갈림 — 결정론 유지.
  "가중 표준편차 0 → 점수 0.0" 가드가 죽은 코드가 아님을 실증(가중치 0 주입으로 재현).
Task 9: minor (deferred): `w` 합이 0이면 `np.average` 가 ZeroDivisionError.
  이번 diff 가 만든 것이 아니다 — `weighted_r2`(원본)도 같은 조건에서 이미 크래시했다.
  `manifold.select` 가 [1,5] 를 보장하므로 실사용 경로에 도달하지 않는다. 다만 그것은
  **계약이 아니라 우연**이다. 최종 리뷰에서 가중치 계약을 명시할지 판단할 것.
Task 9: complete (commits e62dd2d..8e82ff1, review clean)

Task 10: dispatched (sonnet). BASE=8e82ff1
  ★ 여기서부터 컴파일러 3부작 — DESIGN.md D11 이 "이 실험의 유일한 신규 위험 지점" 으로
  지목하고 테스트 우선을 명시한 구간이다. 실패 모드가 조용하다: 잘못 번역된 진입식은
  예외 없이 그럴듯한 원장을 뱉는다.
Task 10: 리뷰 — 사양 ✅ / 품질 변경 필요. Important 2, Minor 0. 둘 다 plan-mandated.

Ruling R15 (`MONOTONE_UNARY` 에서 `log` 를 뺀다):
  리뷰어가 **Task 11 의 안전장치가 우회되는 경로**를 찾았다. `to_catalog.translate` 는
  `log` 를 명시적으로 거부한다("인수 양수성이 보장되지 않는다"). 그런데 Task 12 파이프라인이
      stripped, shells = normalize.strip_monotone(candidate.expr)
      numeric = to_catalog.translate(stripped)
  순서라, **최외곽 `log` 는 translate 가 보기도 전에 strip_monotone 이 제거한다.**
  거부 로직이 중첩된 log 에만 걸리고 최외곽에는 발동하지 않는다.

  리뷰어 수치 실증 (u 에 음수 40% 섞인 10,000행, q=0.7):
    올바른 역변환(τ' = exp(τ_log), 항상 양수라 u≤0 자동 배제) → 원본과 일치 1797 = 1797행
    Task 12 가 실제로 하는 것(벗겨진 u 에 대해 분위수를 새로 추정) → 3000행
    **1203행(12%) 불일치.** u ≤ 0 인 행이 원본에서는 log 미정의로 절대 진입하지 않는데
    벗긴 뒤에는 u 전체 분포로 분위를 새로 매기므로 그 배제가 깨진다.

  `tanh`·`atan`(전체 실수에서 순증가)와 `sqrt(Abs(·))`(인수가 항상 ≥0)에는 이 문제가 없다.
  **`log` 만 유일하게 부분 정의역을 갖는다.**
  근거: 두 모듈이 같은 것에 대해 반대로 행동하고 있다 — 하나는 거부하고 하나는 흡수한다.
  일치시키는 가장 싼 방법이 `MONOTONE_UNARY` 에서 log 를 빼는 것이고, 그러면 log 후보가
  translate 에 도달해 설계대로 CompileFailure 로 기록된다.
  틀렸을 때의 비용: 최외곽 log 후보가 흡수 대신 탈락한다 — 그것이 의도다.

Ruling R16 (`Pow` 지수 가드에 테스트를 붙인다):
  리뷰어가 `isinstance(expr, sympy.Pow)` 만 남기고 지수 제한을 없애도 **7개 테스트가 전부
  통과**함을 재현했다. `x**2` 는 실수 전체에서 순증가가 아닌데(음수에서 감소) 흡수하면
  부등호 의미가 뒤집힌다. 게다가 `to_catalog` 가 `exp == 2` 를 내부 노드로 명시 지원하므로
  최외곽 `(feature)**2` 후보가 나오는 것이 비현실적이지 않다.
  `Mul` 의 `is_positive` 검사는 브리핑이 테스트로 고정했는데 이것만 비대칭으로 비어 있다.
  틀렸을 때의 비용: 테스트 하나. 없다.

Task 10: 이월 (PySR 전환 시): 계획서 SR 사양의 `unary_operators` 에 `"log"` 가 있는데
  `to_catalog` 가 거부하므로 log 후보는 항상 컴파일 실패한다. PySR 탐색 예산 낭비다.
  전환 시 `log` 를 빼거나 `log1p` 로 바꿀 것.
Task 10: fix round 1/5 — dispatched (FIX_BASE=b436faa)
Task 10: fix round 1/5 (2 addressed + atan 테스트; commits b436faa..d0f45c0)
  재리뷰가 뮤테이션 3건을 독립 재현했고 각각 **정확히 해당 테스트만** 잡히는 것을 확인:
    log 재추가 → test_outer_log_is_not_absorbed + test_log_variant_does_not_share_normal_form
    Pow 지수제한 제거 → test_even_power_is_not_treated_as_sqrt
    atan 제거 → test_outer_atan_is_absorbed
  매번 원복 후 diff 무출력 + md5 fbe43e9506ed301e8609f4904b9d94d9 일치.
  F2 는 코드 변경 없이 테스트만 추가 — 브리핑 코드가 이미 올발랐다.
Task 10: complete (commits 8e82ff1..d0f45c0, review clean)

Task 11: dispatched (sonnet). BASE=d0f45c0
  ★ 교차 태스크 계약 검증을 요구사항으로 넣는다: Task 10 이 최외곽 log 흡수를 막았으므로
    이제 log 후보가 실제로 `translate` 에 도달해 `TranslationError` 로 탈락하는지
    **종단으로 확인**해야 한다. 두 모듈 사이의 계약이라 어느 한쪽만 봐서는 확인이 안 된다.
Task 11: implementer DONE (커밋 d26f879, 101 passed — 브리핑 12 + log 종단 3 + 보강 7)

  ★ **교차 태스크 계약 확인됨.** `strip_monotone(log(bi))` 가 벗기지 않고 그대로 돌려주고,
    그 결과가 `translate` 에서 `TranslationError` 로 탈락한다. 최외곽 단독과
    `tanh(log(bi)+qi)` 처럼 더 큰 식 안에 있는 경우 모두 파이프라인 실제 순서로 종단 검증.

  ★★ **브리핑의 `FORBIDDEN_KEYS` 테스트가 최상위 노드만 검사했다.** 깊이 3 중첩이나
    `all.args` 리스트 안에 숨긴 `execution_binding` 을 놓쳤다. 구현자가 뮤테이션으로
    재귀 제거를 시험해 이것을 발견하고 보강했다.
    → 이것이 **수식이 청산 규칙을 몰래 바꾸는 것을 막는 유일한 장치**다. 내가 브리핑에
      테스트를 얕게 쓴 탓에 그 장치가 사실상 무방비였다. 구현자 단계에서 잡혀 다행이다.

  뮤테이션 5건 중 3건이 브리핑 12개로 안 잡혔다. 전부 보강 후 md5 원복 확인.
  구현자 우려 2건(평가 대상):
    (a) `Pow` 미지원 지수(`x**3`) 거부 분기에 전용 테스트 없음 — Task 10 F2 와 같은 패턴
    (b) `check.py` 의 `except (KeyError, ValueError)` 폴백이 발동한 적 없음
Task 11: 리뷰 — 사양 ✅ / 품질 변경 필요. Important 1, Minor 2.
  리뷰어가 뮤테이션 5건을 전부 독립 재현(md5 값까지 보고서와 일치).
  **뮤테이션 1(재귀 제거)에서 브리핑의 최상위 전용 테스트는 통과해 버리고, 구현자가
  추가한 depth-3·리스트 은닉 테스트 2건만 정확히 잡는 것을 확인.** 내가 지목한 지점이
  실제로 브리핑 테스트만으로는 새고 있었다.
  log 종단 계약을 리뷰어가 3개 변형으로 독립 재현 (단독 / tanh 중첩 / 단조 껍질 없는 합).

Ruling R17 (`Pow` 미지원 지수 거부에 테스트를 붙인다 — Task 10 F2 와 같은 처리):
  리뷰어가 거부 분기를 지우고 **모든 미지원 지수를 `x**2` 처럼 자기 자신과 곱하도록**
  바꿔도 22개 테스트가 전부 통과함을 확인했다. `book_imbalance**3` 같은 홀수 지수가
  `**2` 로 등가 취급되면 **부호 정보가 소실되어 항상 양수**가 된다 — 진입식이 반대
  구간을 고르는데 예외가 없다.
  근거: Task 10 에서 정확히 같은 패턴(`normalize.py` 의 Pow 짝수지수 가드 미테스트)을
  Important 로 판정하고 실제로 고쳤다. **같은 결함에 다른 기준을 적용하지 않는다.**
  구현이 아니라 테스트만 없으므로 Task 10 F2 와 동일하게 테스트만 추가한다.
  틀렸을 때의 비용: 테스트 1~2개. 없다.

Task 11: minor (deferred): `check.py` 의 `except (KeyError, ValueError)` 는 도달 불가.
  `_infer` 가 이미 `try/except Exception` 으로 감싸 `ExpressionError` 로 내보내고,
  유일한 KeyError 발생원(`compute_features`)은 이 경로에서 호출되지 않는다. 죽은 방어 코드.
Task 11: minor (deferred): `to_catalog.py` 의 "상수 곱만 남았다" 분기도 도달 불가.
  sympy 가 순수 상수 Mul 을 `is_number == True` 로 평가해 `_walk` 최상단이 먼저 잡는다.
Task 11: fix round 1/5 — dispatched (FIX_BASE=d26f879)
Task 11: fix round 1/5 (1 addressed, 0 open; commits d26f879..97d43b0)
  재리뷰가 뮤테이션을 독립 재현 — 정확히 새 테스트 2개만 DID NOT RAISE 로 떨어지고
  지원 지수 테스트 2개 포함 나머지 24개는 통과 (2 failed, 24 passed).
  구현 무변경 확인: diff 는 tests 파일 39줄 추가뿐, to_catalog.py·check.py md5 원값 일치.
Task 11: complete (commits d0f45c0..97d43b0, review clean)

Task 12: dispatched (sonnet). BASE=97d43b0
  ★ 사전 스캔의 Ruling R3 을 여기서 적용한다 — `pipeline.compile_candidates` 가 정규형
    키를 `sympy.srepr(sympy.simplify(stripped))` 로 다시 만들지 말고
    `normalize.normal_form()` 을 부를 것. 같은 산술이 두 곳에 있으면 한쪽만 고쳐진 채
    갈라진다 (프레임워크가 metrics.py 에서 실제로 겪은 일).
Task 12: implementer DONE (커밋 0f76a01, 111 passed)
  Ruling R3 적용 확인: 인라인 `srepr(simplify(stripped))` → `normal_form(candidate.expr)`.
  12개 다양한 식(스케일·오프셋·tanh·atan·sqrt·제곱·음수배 조합)에서 두 방식이
  **전부 동일한 키** — 갈라지는 사례 없음. `strip_monotone` 이 멱등이라 값은 같지만,
  앞으로 `normal_form` 정의가 바뀔 때 파이프라인이 자동으로 따라가도록 단일 계약으로 통합.
  log 후보(단독 + `3*log(bi)+5` 중첩)가 예외 없이 `CompileFailure(stage="translate")` 로 기록됨.
  뮤테이션 5건 전부 잡힘 (parameter_table 키 순서 / threshold kind 오타 / 중복병합 최저점수 /
  stage 통일 / attach 의 boolean 미생성).
  구현자 우려 2건: (a) `stage` 중 "normalize"·"check" 는 도달 불가에 가까운 방어 경로라
  개별 테스트 없음, (b) `strip_monotone` 이 후보당 두 번 호출됨(직접 + normal_form 내부).
Task 12: 리뷰 — 사양 ✅ / 품질 승인. Important 1, Minor 3.
  리뷰어가 `normal_form` 교체를 20개 식(자기 8개 경계 케이스 추가)으로 독립 재현 —
  전부 일치. `strip_monotone` 이 고정점까지 반복하는 while-loop 라 원본이든 벗긴 값이든
  같은 고정점에 도달한다는 **구조적 근거**까지 제시했다.
  log 후보 4변형(단독/스케일+오프셋/인수 시프트/tanh 중첩) 전부 stage="translate" 확인.
  뮤테이션 5건 독립 재현, 매번 md5 원복 확인.
  리뷰어가 `metrics.py` 인용이 **실제 문장인지** 대조까지 했다 — 조작된 근거가 아님을 확인.
  `contract_id()` 가 Task 12 에서 안 쓰이는 것도 죽은 코드가 아니라 Task 13 replay.py 가
  소비하는 **계획된 선행 인터페이스**임을 PLAN.md 로 확인.

Controller: DESIGN.md §4.1 을 갱신했다. `EntryExpression` 에 `shells` 가 빠져 있었고
  `CompileFailure.stage` 에 도달 불가능한 `'threshold'` 가 적혀 있었다. 리뷰어 지적대로
  브리핑/PLAN.md 가 맞고 DESIGN.md 스케치가 낡았다. 설계 문서가 구현보다 뒤처진 것이다.

Ruling R18 (`stage` 라우팅 회귀 테스트를 붙인다):
  리뷰어가 `"normalize"`·`"check"` 리터럴만 `"same"` 으로 바꿔도 111개가 전부 통과함을
  확인했다. 두 경로가 현재 입력면(순수 sympy Candidate)에서 구조적으로 도달 불가인 것도
  맞다 — 그래서 지금은 버그가 아니다.
  그래도 채우는 이유: `stage` 필드의 존재 이유가 **실패를 단계별로 귀속하는 것**이고,
  그 귀속이 맞는지 아무도 검사하지 않으면 필드가 장식이 된다. PySR 로 교체하면 후보
  생성 방식이 바뀌어 실제 위험이 된다. monkeypatch 로 각 단계를 실패시켜 **라우팅**을
  검사하면 되고, 그것은 해당 모듈이 아니라 파이프라인의 책임을 검사하는 정당한 테스트다.
  틀렸을 때의 비용: 테스트 2개. 없다.
Task 12: minor (deferred): `contract_id()` 단위 테스트 없음 (한 줄 포맷 함수, Task 13 이 소비)
Task 12: minor (deferred): `strip_monotone` 후보당 2회 호출 — 무거운 `simplify` 는
  `normal_form` 안에서 1회뿐이고 중복은 가벼운 순회다. 의도적 트레이드오프.
Task 12: fix round 1/5 — dispatched (FIX_BASE=0f76a01)
Task 12: fix round 1/5 (1 addressed, 0 open; commits 0f76a01..c614f70)
  재리뷰가 뮤테이션 독립 재현 — 신규 2개만 정확히 FAILED
  (`assert 'same' == 'normalize'`, `assert 'same' == 'check'`), 기존 6개는 무감.
  monkeypatch 깊이도 검증: `pipeline.py` 가 `from . import check, normalize` 로 모듈을
  바인딩하므로 `pipeline.normalize`/`pipeline.check` 패치가 내부 호출에 실제로 반영된다.
  `_boom` 시그니처가 실제 호출 시그니처와 일치. 예외 미누출·`reason` 보존도 확인.
Task 12: complete (commits 97d43b0..c614f70, review clean)
  ★ 컴파일러 3부작 완료. Task 10(단조 흡수) → 11(번역·정적검사) → 12(임계·파이프라인).

Controller 관찰: Task 10 F2 · Task 11 F1 · Task 12 F1 이 **모두 같은 형태**였다 —
  구현은 올발랐고 그것을 지키는 테스트만 없었다. 세 번 연속 나온 것은 내 계획서
  테스트가 "정상 경로는 잘 덮고 **방어 경로는 안 덮는**" 편향을 가졌다는 뜻이다.
  남은 태스크 브리핑에 이 경계를 명시한다.

Task 13: dispatched (sonnet). BASE=c614f70
Task 13: implementer DONE (커밋 5dbf925, 121 passed — 브리핑 2 + 방어 테스트 6)
  ★ **실데이터 재생 성공.** 2종목(000660/035420) × 분위 0.85, workers=2 → 53.69초,
    `errors == []`, 프로필 해시 일치. 원장 823행 254열,
    status UNFILLED 793 · FILLED 29 · CENSORED 1.
  내가 알려준 편향(방어 경로 미검증)을 의식적으로 뒤집었다 — 5건 전부 뮤테이션으로 확인:
    errors 가드 / 빈 entries 가드 / attempt_count 실제 곱셈 / 분위별 계약 ID 상이성 /
    parameter_table 전체 조합 커버리지.
  머신 부하 주의: load average **18~21** (앞선 태스크의 12~14 보다 높다).
  다른 세션 optuna 캠페인이 계속 돌고 있다. Task 15 엔드투엔드가 가장 무거우므로
  그때 시간이 크게 늘어도 회귀로 단정하지 않는다.
Task 13: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 3.
  리뷰어가 뮤테이션 5건을 전부 독립 재현(매번 md5 원복 확인).
  ★ **실행 설정 비유입을 코드로 검증했다** — `replay.run` 시그니처에 청산·큐·비용을
    바꿀 인자가 없고, `run_backtest` 의 `canonical` 인자를 노출하지도 전달하지도 않으며,
    `stop_gross_bps` 등이 전부 `canonical["exit"]` 에서만 나온다. 이것이 이 태스크의 핵심
    계약인데 실제로 지켜진다.
  `workers` 는 병렬성일 뿐 회계 무관, `grid` 는 계획서가 명시한 유일한 탐색 대상이라
  노출이 맞다는 판정도 정확하다.
Task 13: minor (deferred): 빈 `grid=()` 에 가드가 없다 — 빈 `entries` 가드와 비대칭.
  `contracts={}` 로 조용히 진행된다. 정본 격자는 항상 비지 않으므로 실사용 위험은 낮다.
Task 13: minor (deferred): `contract_id` 가 분위를 소수 2자리로 포맷 — 임의 격자에서
  `0.701`/`0.704` 가 같은 ID 로 충돌해 조용히 덮어쓸 수 있다. Task 12 상속.
Task 13: minor (deferred): `from .compile.pipeline import PARAMETER` 가 `__all__` 밖
  내부 심볼을 직접 import. 브리핑 코드 그대로.
Task 13: complete (commits c614f70..5dbf925, review clean)

Task 14: dispatched (sonnet). BASE=5dbf925
Task 14: implementer DONE (커밋 41638ef, 129 passed)

  ★★ **내 브리핑의 가장 중요한 테스트가 공허했다.** `test_ranking_uses_total_net_not_ratio`
    가 분모 함정을 잡으라고 쓴 것인데, fat/thin 픽스처가 **두 지표 모두에서 이겨서**
    정렬 키 우선순위를 바꿔도 조용히 통과했다. 즉 이 태스크의 존재 이유인 그 규칙을
    아무도 검사하지 않고 있었다. 구현자가 진짜 충돌 케이스를 만들어 메웠다
    (`test_ranking_prefers_total_net_bps_over_ratio_on_conflict`).
    → 정본 프레임워크가 계약 1,025개에서 실측한 실패 양식을 막는 코드인데, 그 방어를
      검증하는 테스트가 검증을 안 하고 있었다.
  정렬 동률 결정론 테스트도 없어서 추가했고, 그것은 실제 회귀(`groupby(sort=False)`)를 잡는다.

  구현자 우려 3건(전부 정직하고 타당함):
  (a) `kind="mergesort"` 제거 뮤테이션을 잡는 테스트가 없다 — pandas 기본 quicksort 가
      작은 완전 동률 픽스처에서 우연히 순서를 보존했다. 소스에 명시돼 있어 계약은 서지만
      순수 블랙박스로는 증명 불가.
  (b) `positive` 의 "총액과 결정당 둘 다" 뮤테이션이 **수학적으로 no-op** 이다 —
      `net` 이 FILLED 행만 더하고 `scorable ≥ fills` 이라 `scorable==0 ⇒ net==0.0` 이
      항상 성립한다. 판별력을 가질 수 없는 조건이다. 그래도 필드 자체 커버리지는 추가.
  (c) 원장에 `cohort` 컬럼이 아예 없는 분기는 미검증 — 실제 원장은 항상 갖고 있다(823행 254열 확인).
Task 14: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 3.
  ★ 리뷰어가 **공허한 테스트 주장을 독립 재현**했다. 정렬 키를 뒤집었을 때
    브리핑 원본 `test_ranking_uses_total_net_not_ratio` 는 **PASSED**(공허 확인),
    신규 충돌 테스트는 **FAILED**(`assert 'B' == 'A'`). 구현자 보고와 100% 일치.
  뮤테이션 4건 추가 재현(하한 0 / CENSORED 분모 포함 / 코호트 == → != / groupby sort=False).
  ★ 구현자의 수학 논증도 검증: `scorable = fills + unfilled` 이고 `net` 은 FILLED 만
    합산하므로 `scorable==0 ⇒ fills==0 ⇒ net==0.0`. 따라서 `bool(scorable and net>0)` 과
    `bool(net>0)` 이 이 구현에서 항상 동치 — 게이트 제거 뮤테이션이 8개 전부 통과함을
    재현해 **판별 불가라는 주장이 참임을 확인**했다.
  status 3종·코호트 4종 이름이 vendor 정본 상수와 일치함도 대조.
Task 14: minor (deferred): `rank()` 가 `contract_id` 동률 규칙을 정렬 키에 명시하지 않고
  `groupby(sort=True)` 출력 순서 + mergesort 안정성이라는 암묵 전제에 의존. 브리핑 설계.
Task 14: minor (deferred): 브리핑 인터페이스 서술의 컬럼 순서와 코드의 순서가 불일치
  (`censored, unfilled` vs `unfilled, censored`). 브리핑 자체의 내적 불일치, 기능 무관.
Task 14: minor (deferred): `kind="mergesort"` 제거 뮤테이션은 어떤 테스트로도 안 잡힘.
  알려진 잔여 리스크.
Task 14: complete (commits 5dbf925..41638ef, review clean)

Task 15: dispatched (sonnet). BASE=41638ef
  ★ 엔드투엔드. 파이프라인 전체가 처음으로 실제 데이터에서 관통한다.
Task 15: implementer DONE (커밋 89c7bdb, 133 passed) — ★ 엔드투엔드 관통 성공

  본 실행 (30종목, 기본 설정, runs/20260903T094959Z-7ddbf870/):
    후보 12 → 컴파일 9 (성공률 75%), 2개 translate 탈락 (미지원 `^-1` 지수)
    원장 143,955행 (UNFILLED 138,260 / FILLED 4,630 / CENSORED 1,065), errors []
    프로필 해시 d0b91a03541c0cec 일치
    27/27 계약이 scorable ≥ 200 통과, **0/27 net 양수** (설계상 예상된 결과)
    소요 204.7초, 스모크 런(6종목) 25.1초. feature 교집합은 9개 전부 유지.
  뮤테이션 자기검토가 브리핑 테스트 2개의 맹점을 또 잡았다 — `naive` 경고를 항상 내도록
  바꿔도, `success_rate` 를 상수로 바꿔도 원본 2개가 조용히 통과했다.

★★ Ruling R13 정정 — 내가 틀렸다.
  R13 에서 "선형종속 쌍의 아핀 결합은 단조 흡수(S3)가 임계로 걷어내므로 진입식 수준에서
  새 구조가 되지 않는다" 고 판단했다. **실행이 반증했다.**
  실측: 컴파일된 9개 진입식 중 **5개가 바이트 단위로 동일한 원장**을 냈다.
  이유: 중복 병합이 **sympy 표현식 수준**(symbolic)에서 일어나는데,
  `book_imbalance` 와 `queue_imbalance_best` 는 **서로 다른 심볼**이라
  `sympy.simplify` 가 `bi = 2·qib − 1` 관계를 알지 못한다. 그래서
  `bi + qib`(= `3·qib − 1`)와 `qib` 는 다른 AST 로 남아 둘 다 컴파일되지만,
  분위 임계 아래에서 아핀 변환은 **같은 행을 고르므로** 원장이 동일해진다.
  → 단조 흡수는 **같은 식의 껍질**만 벗긴다. **다른 심볼 간의 아핀 관계**는 못 본다.
  영향: 후보 슬롯의 절반 이상이 낭비되고, 보고되는 시도 횟수 N 이 실제 독립 시도보다
  부풀려진다. N 과대는 다중 검정 할인을 **과하게** 하므로 안전한 방향이지만,
  탐색 예산과 Pareto 슬롯은 실제로 낭비된다.
  틀렸을 때의 비용: 이미 치렀다 — 이 슬라이스에서 9개 중 5개가 중복이었다.
  PySR 전환 시 조치: 어휘에서 중복 쌍 하나를 빼거나, 중복 병합을 **원장 수준**
  (또는 무차원 좌표에서의 아핀 동치)에서 하도록 바꾼다.
Task 15: 리뷰 — 사양 ✅ / 품질 승인. Important 2, Minor 1.
  리뷰어가 "9개 중 5개 동일 원장" 을 **세 층위로** 독립 재현했다:
   (1) report.md summary 테이블에서 e000/e001/e002/e005/e008 이 세 분위 전부 수치 동일
   (2) ledger.parquet 을 열어 (symbol,date,decision_index,status,net,fill_tick,exit_tick)
       튜플 해시로 계약 간 비교 — 세 분위 모두 정확히 한 그룹
   (3) AST 대조로 수학적 확인: qib∈[0,1] 이라
       e000(bi=2qib−1) · e001(qib, tanh 흡수) · e002(bi+qib=3qib−1) ·
       e005(|qib|=qib) · e008(bi−qib=qib−1) 이 **전부 qib 의 단조증가 함수**
  → 내 R13 정정이 맞았고, 리뷰어가 (3)으로 **왜** 그런지까지 밝혔다.
  산출물 9종 전부 확인, AST 9개 전부 `boolean` 타입 독립 검증, 날짜 봉인(20260317 이후
  참조 전무) 확인, 뮤테이션 3건 재현.

Controller: DESIGN.md·PLAN.md 를 커밋했다(86cc9ea). README 가 링크하는데 untracked 라
  새 클론에서 깨졌다 — 리뷰어 Minor 지적.

Ruling R19 (산출물의 두 신뢰도 구멍을 report.py 에서만 메운다):
  (a) `compile_report.json` 이 attempted 12 · succeeded 9 · failed 2 를 내는데
      **12−9−2=1 이 설명되지 않는다.** 원인은 정규형 중복 시 승자만 남기고 패자를
      `failed` 에 넣지 않는 것(Task 12 설계). report.md 를 읽는 사람은 알 길이 없다.
      숫자가 앞뒤가 안 맞으면 **그 리포트의 다른 모든 숫자도 의심받는다.**
  (b) `attempts=27` 이 실질 독립 시도(5×3=15)를 반영하지 않는다. 계획서가 시도 횟수
      보고를 **방법론 요구사항**으로 못 박았고(미보고 시 성과 주장 무효), 그 수가
      과대하면 다중 검정 할인이 과해진다 — 안전한 방향이지만 정확하지 않다.
  둘 다 **`report.py` 안에서만** 메울 수 있다. (a)는 차이를 빼서 "중복 병합" 으로
  이름 붙이면 되고, (b)는 `summary` 에서 지표 튜플이 동일한 계약을 묶어 세면 된다.
  파이프라인을 건드리지 않으므로 이미 검증된 것이 흔들리지 않는다.
  틀렸을 때의 비용: 리포트에 두 줄. 없다.
Task 15: minor (deferred): 단계별 시간 계측이 없다 — 부하 20 환경에서는 의미가 적다.
Task 15: fix round 1/5 — dispatched (FIX_BASE=86cc9ea)
Task 15: fix round 1/5 (2 addressed, 0 open; commits 86cc9ea..383aa13)
  재리뷰가 두 숫자를 **산출물에서 직접 계산해** 독립 재현했다:
    12−9−2=1 (candidates.json + compile_report.json 에서 직접)
    summary.parquet 27행을 9개 지표열로 groupby → 그룹 15개, 크기-5 중복 3개,
    계약 ID(e000/e001/e002/e005/e008 × 3분위)까지 일치
  뮤테이션 2건 재현(merged_duplicates 상수화, duplicate_signal_groups 항상 빈 결과).
  파이프라인 무변경 확인. 기존 report.md 항목이 하나도 사라지지 않았음을 diff 로 확인
  (순수 추가분만).
  문구 구분도 확인: "지표가 같다는 것은 증거이지 증명은 아니다 — '지표가 동일한 계약'
  이지 '같은 진입식'이 아니다".
Task 15: complete (commits 41638ef..383aa13, review clean)

Task 16: dispatched (sonnet). BASE=383aa13 — **마지막 구현 태스크**
Task 16: implementer DONE (커밋 1ee3644, 149 passed — 신규 10)

  ★ **실데이터 게이트 세 검사 전부 통과** (20260316, 30종목, 병목 2, 시드 0):
    beats_constant          correlation = 0.1988    (임계 > 0.05)
    fill_calibration        max_abs_gap = 0.0341    (임계 ≤ 0.25)
    adverse_selection_sign  top_decile_mean_y_path = **−1.0532**  (임계 ≤ 0, 표본 19,796)

  세 번째가 단순한 green check 이 아니다. **체결 확률 상위 십분위에서 실제 경로 결과가
  음수**라는 것은 계획서가 예측한 역선택이 데이터에서 실제로 관측된다는 뜻이다 —
  지정가 매수가 잘 체결되는 순간은 파는 쪽이 급한 순간이다. 이 부호가 양수였다면
  큐 모델이 낙관적이거나 라벨에 미래가 새고 있다는 신호였을 것이다.
  즉 이 검사는 **"통과했다"가 아니라 "라벨과 큐 모델이 명백히 새지는 않는다"** 는
  증거를 준다. 표본 19,796 으로 잰 값이다.

  `--ignore-gate` 경로도 확인 — `s1_gate_ignored=true` 와 경고문이 provenance.json 과
  report.md 에 실제로 기록된다. 게이트 불통과 → SystemExit 경로는 이번 시드가 우연히
  건강해 실데이터로 재현 못 했다. **임계를 낮추지 않았다** (지시 준수).

  뮤테이션이 브리핑 테스트의 빈틈 2개를 또 잡았다:
   (1) `_calibration` 이 모든 버킷을 `np.ones_like(...)` 로 뭉개도(십분위 무시) 5개 전부 통과
   (2) `sd/report.py` 의 게이트 기록 로직을 **통째로 삭제해도** 기존 10개가 전부 통과
  둘 다 보강. 나머지 3건(부호 뒤집기 / all→any / 상수 예측기)은 브리핑 테스트가 즉시 잡음.
Task 16: 리뷰 — 사양 ✅ / 품질 승인. Critical·Important 0, Minor 2.
  ★ 리뷰어가 **임계 완화 흔적이 없음을 확인**했다 — MIN_CORRELATION=0.05,
    MAX_CALIBRATION_GAP=0.25 가 브리핑 값 그대로이고 역선택 판정 `mean <= 0.0` 도
    뒤집히지 않았다. `git diff -- sd/config.py` 가 비어 정본 상수도 무손상.
  뮤테이션 5건 전부 독립 재현(역선택 부호 뒤집기 포함). Task 15 산출물
  (merged_duplicates·duplicate_signal_groups·independent_contract_count) 무손상 확인.
  실데이터 두 run 의 provenance.json 을 직접 열어 s1_gate_passed=false(정상경로) /
  s1_gate_ignored=true(--ignore-gate 경로) 가 보고와 일치함을 확인.
Task 16: minor (deferred): 게이트 세 검사의 **실제 숫자가 영속화되지 않는다.**
  provenance.json 에는 불리언 둘만 남고 correlation·max_abs_gap·top_decile_mean_y_path 는
  stdout 에만 있다. 나중에 특정 run 을 감사할 때 "왜 통과했는가" 의 근거가 복구 불가.
  브리핑이 불리언 둘만 요구했으므로 사양 위반은 아니다. **최종 리뷰 판단 대상.**
Task 16: minor (deferred): 뮤테이션 원복 절차는 리뷰어가 독립 재현했으나 구현자 기록
  자체를 별도 트랜잭션으로 검증한 것은 아니다.
Task 16: complete (commits 383aa13..1ee3644, review clean)

## 16/16 구현 완료. 최종 전체 리뷰로 넘어간다.

## 최종 전체 리뷰 + 수정 웨이브 + 재리뷰

최종 리뷰(opus): 수정 필요 — Critical 1 · Important 7 · Minor 다수.
  ★ C1: `to_catalog._walk` 가 **깊이의 양수 상수배 크기를 조용히 버린다.**
    내가 계획서에 쓴 근거 `q분위(c·u)=c·q분위(u)` 는 **최외곽에서만** 성립하는데
    깊이에 적용했다. 실측 `sa*tanh(2*bi) → multiply(sa, tanh(bi))`,
    `tanh(2)=0.9640 ≠ tanh(1)=0.7616` — **다른 함수, 포화점 이동.**
    16번의 태스크 리뷰가 전부 놓쳤다. 이유: 그 동작을 잠근 테스트가
    **파이프라인이 절대 가지 않는 경로**(`translate(3*bi)`)를 테스트하고 있었다.
    `strip_monotone` 이 최외곽을 먼저 벗기므로 그 분기에 도달하는 건 깊이의 계수뿐이다.
    → 테스트된 자리에서는 도달 불가, 도달 가능한 모든 자리에서는 틀렸다.

수정 웨이브 (커밋 f4672e2·bd58dee·c4537db·005be45·71d5de8 + 문서 1ed5aef): 163 passed.
  컴파일 성공률 9/12 유지. 수정자가 자기 회귀(평범한 뺄셈 거부)를 스스로 잡아 `|c|==1`
  예외를 넣었다 — 내 지시가 너무 넓었다. 규칙을 근거에서 다시 유도하지 않고 결론만 옮겼다.

재리뷰: F1~F6 전부 ADDRESSED. 단 **잔여 결함 1건 발견.**

Ruling R20 (잔여 결함을 park 하고 사용자에게 올린다 — 두 번째 수정 웨이브를 돌리지 않는다):
  발견: F1 의 `abs(product) != 1` 비교가 `sympy.Float` 계수에서 깨진다. 내가 직접 확인:
      abs(sympy.Float(-1.0)) != 1      → True   (틀림)
      abs(sympy.Float(-1.0)) - 1 != 0  → False  (맞음)
      translate("bi - queue_imbalance_best")      → OK        (정수 -1)
      translate("bi - 1.0*queue_imbalance_best")  → 거부       (부동소수 -1.0)
  즉 F1 이 고친 "평범한 뺄셈 거부" 회귀가 **부동소수 표현에서 그대로 재발**한다.
  방향은 과잉 거부(안전 쪽)이지 크기 손실이 아니다.

  park 하는 이유 셋.
  (1) **지금은 죽은 코드다.** NaiveBackend 는 수치 계수를 내지 않아 이 경로가 한 번도
      발동하지 않는다. 이 브랜치에서 하중이 아니다.
  (2) **고치려면 PySR 이 실제로 무엇을 내는지 알아야 한다.** 재리뷰어의 "PySR 은 통상
      부동소수로 낸다" 는 추정이고, PySR 은 설치돼 있지도 않다. 지금 고치면 **설치되지
      않은 백엔드의 출력 형태에 대한 내 추정에 맞춰 테스트를 쓰는 것**이 된다 —
      이 세션에서 반복해서 사고를 낸 바로 그 패턴(검증되지 않은 가정을 코드에 박기)이다.
  (3) 스킬이 두 번째 수정 웨이브를 금지하고, 잔여 하중 발견은 사용자에게 올리라고 한다.

  **PySR 교체 태스크의 착수 조건으로 잡는다.** 그때 `abs(product) - 1 != 0` 로 바꾸고
  PySR 의 실제 출력으로 검증한다. 결함 자체(sympy Float 비교)는 백엔드와 무관하게
  참이므로 진단은 확정이다 — 미룬 것은 수정이 아니라 **검증 방법**이다.
  틀렸을 때의 비용: PySR 전환 시 순수 부호 반전 후보가 대량 오탈락해 컴파일 성공률이
  실제보다 낮게 나온다. 안전 방향이고 한 줄로 고쳐진다.

## 확장 단계 (1~5) 실행

1단계 어휘 넓히기: 커밋 a7008c2, 199 passed. 파생 열 10개(AST 가 곧 정의, 수치는
  ExpressionRuntime 평가). SR 입력 9 → 19. `check.py` 에 condition_problem 추가
  (대상/재료 구분). 리뷰 승인, Minor 1(주석 근거 서술 과장).
  리뷰어 실측: tanh/sqrt(Abs) 단조 중복은 1단계가 만든 게 아니다 — 기존 Catalog
  feature 3개가 이미 [0,1] 범위로 같은 취약점을 갖고 있었고 신규 10개 중 1개만 노출.

2단계 PySR 교체: 커밋 3810b77, 238 passed. 리뷰 승인, A~G 전부 통과.
  ★ **F7 첫 진짜 측정** — 나눗셈 전 4/15=27%, 후 8/15=53%. 둘 다 폐기선(20%) 상회.
  구현자가 상승 원인을 잘못 귀속하지 않았다: 나눗셈 후보 둘은 c/x(상수 분자)라 거부됐고,
  상승은 PySR 이 연산자 하나를 더 받아 탐색을 다르게 한 결과다. 리뷰어가 산출물로 대조 확인.
  내 계획서 구멍 하나 더 발견: 나눗셈이 통째로 막혀 있었다. `a/b = Mul(a, Pow(b,-1))`
  인데 지수 -1 을 거부했고, 거부 사유("ratio 로 명시해야 분모 0 정책이 붙는다")가
  가리키는 `ratio` 연산자가 Catalog 에 zero_policy 필드까지 갖고 있었다.
  juliacall 경고는 안 사라졌다 — 구현자가 소스를 추적해 HANDLE_SIGNALS 와 무관함을
  확인하고 **고쳐진 척하지 않고** 남는 위험으로 기록했다.

Ruling R21 (PySR diagnostics 영속화를 3단계 이후로 미룬다):
  2단계 리뷰가 Important 로 지목: `pysr_backend.fit()` 의 diagnostics(fit 시간,
  complexity 불일치, 버려진 후보 수)가 어떤 산출물에도 안 남고 stdout 한 줄뿐이다.
  그래서 2단계 보고서의 F7 감사 근거(불일치 7/15·11/15, 277.9s/279.5s)를 저장소만으로
  재현할 수 없다. **근거로 쓰는 숫자는 영속화해야 한다** — S1 게이트 숫자에서 이미
  같은 판정을 내렸다(최종 수정 웨이브 F4).
  지금 고치지 않는 이유: 3단계 구현자가 `sd/sr/pysr_backend.py` 를 이미 만지고 있다
  (리뷰어가 커밋 안 된 생성자 변경을 발견 — E0 용 연산자 인자로 보임). 지금 손대면 충돌한다.
  3단계 완료 직후 작은 수정으로 처리한다.
  틀렸을 때의 비용: 3단계 산출물의 PySR 진단 숫자도 stdout 에만 남는다 — 3단계 보고서를
  읽을 때 그것을 감안해야 한다.

3단계 E0 게이트: 커밋 7fbc700, 298 passed (+ slow PySR 2건 183.5s).
  ★ **게이트 판정 `stop`** — 5개 중 1개(L5) 복원, L1·L2 동시 실패로 거부권 발동.
  실행 규모: 36종목 310,359행, 5법칙 × 4트랙 = PySR fit 20회, 22.3분.

  트랙별 복원 (main / no_distill / no_dimless / uniform_off_manifold):
    L1  F(any=T) / F(any=T) / F(any=T) / F(any=T)
    L2  F(any=F) / F(any=**T**) / F(any=**T**) / F(any=F)
    L3  전부 F
    L4  전부 F
    L5  **T** / **T** / **T** / **F**

  발견 1 — L1 은 파레토 12개 중 8개가 구조적으로 옳았다(단조 홀함수).
    게이트가 최고점수 후보 하나만 심사하기 때문에 탈락했고, 그 후보의 결격 사유는
    대칭파괴 오프셋 `I+0.089` 하나뿐이었다. 방법 실패가 아니라 심사 설계 문제다.
  발견 2 — **L2 에서 ablation 이 본 트랙을 이겼다.** no_distillation·no_dimensionless
    가 법칙을 복원했는데 main 은 못 했다. C1(증류가 돕는다)·C2(무차원화가 돕는다)에
    **반대 방향 증거**다. 이 실행에서 가장 값진 결과 — 우리 가설을 우리가 반증했다.
  발견 3 — L5 에서 uniform_off_manifold 만 유일하게 실패했다. on-manifold 표집이
    실제로 일을 하고 있다는 직접 증거(S2 지지).

  구현자가 판정 함수 자체의 결함 2건을 뮤테이션으로 찾아 **엄격화 방향으로** 고쳤다:
    L5 의 β→0 퇴화(순수 증가 직선 `5*Δt` 가 β≈3.8e-5, R²≈1.0 으로 가짜 통과)와
    any_candidate_recovered 가 degenerate 항등함수로 부풀려지던 빈틈.

Ruling R22 (E0 를 계획서 규모로 한 번 더 돌려 `stop` 의 원인을 가른다):
  이 실행은 `niterations=18` 로 돌았다 — 계획서 원안 200 의 1/11. SR 적합 표본도
  1,487행(전체의 0.5%). 구현자는 이 값을 **결과를 보기 전에** 고정했고 그렇게
  보고했다(사후 변명이 아니다). 그래도 11배 격차는 남는다.
  `stop` 이 (a) 방법이 안 통한다 인지 (b) 충분히 안 돌렸다 인지 이 실행 하나로는
  구분되지 않는다. 0.5% 표본에 탐색예산 9% 로 돌리고 연구를 접는 것은, 결과를
  부풀리는 것과 같은 종류의 부정직한 보고다.
  **사전 약속(실행 전에 못박는다):** 판정 코드는 한 바이트도 바꾸지 않는다.
  밴드도 넓히지 않는다(L3 지수 0.849 를 보고 [0.4,0.9] 로 늘리는 것은 내가
  구현자에게 금지한 바로 그 체리피킹이다). L1 심사를 primary → any_candidate 로
  바꾸지도 않는다. 계획서가 원래 지정한 200 으로만 올린다.
  **200 에서도 L1·L2 가 실패하면 `stop` 은 방법에 대한 판정이고, 우리는 멈춘다.**
  4·5단계는 이 재실행 결과가 나오기 전까지 착수하지 않는다 — 게이트를 열어 두고
  통과하는 것이 게이트를 무의미하게 만든다.
  틀렸을 때의 비용: 재실행 비용(추정 수 시간)을 쓰고도 같은 답이 나올 수 있다.
  그 경우에도 "예산 때문이 아니었다"가 확정되므로 낭비가 아니다.

Ruling R23 (R21 diagnostics 영속화를 재실행 **전에** 처리한다):
  재실행은 이 프로젝트에서 가장 판돈이 큰 측정이다. 그 실행의 PySR 진단이 또
  stdout 에만 남으면 안 된다. 다만 두 E0 실행 사이에 코드가 바뀌면 비교가
  흔들리므로, 고친 뒤 **같은 seed 로 파레토 front 가 바이트 동일한지 먼저 확인**하고
  나서 재실행한다.

3b단계 진행 (R21 + R22):
  작업 A 완료 — `sr_diagnostics.json` 이 run_e0(`e0-20260904T053118Z`)·run_slice
    (`20260904T054222Z-09489c82`) 양쪽에 실제로 떨어진다. 수정 전 기준선 run
    (`e0-20260904T051856Z`)을 따로 남겨 바이트 비교했다.
  작업 B 시작 — 2026-09-04 06:15:24 UTC(KST 15:15) 기동, PID 1352077.
    인자 대조: `--per-stratum 6 --seed 0 --bottleneck 2 --epochs 250
    --sr-maxsize 18 --max-manifold-samples 1500` 전부 직전 실행과 동일,
    `--sr-niterations` 만 18 → 200. 사전 약속이 지켜졌다.
    실측 fit 시간 niter=200: 263.6초 (직전 L2 main 은 48.0초).
    예상 소요 3.5~4.7시간 → KST 18:45~20:00 완료 예상.

내 지시 결함 (기록): 시간 측정 법칙으로 **L2 를 고른 것이 잘못**이었다.
  직전 실행의 main 트랙 fit 시간은 L1 222.8s / L2 48.0s / L3 54.9s / L4 54.6s /
  L5 64.8s 로, L2 가 5개 중 가장 빠르다. 가장 빠른 법칙으로 재서 곱하면
  과소추정된다. 측정 대상은 가장 느린 것(L1) 이거나 전체 평균이어야 했다.
  에이전트가 이를 감안해 범위를 넓게(3.5~4.7h) 잡아 실질 피해는 없었다.
  일반화: **대표성 없는 표본으로 재고 곱하는 것도 "곱해서 추정"의 한 형태다.**

★★★ R22 결론 — E0 재실행(niterations=200) 완료, 판정 `stop` 확정 ★★★
  run: runs/e0-20260904T061532Z, 소요 12,537초(3시간 29분), 종목 목록 직전과 동일.
  바뀐 인자는 --sr-niterations 18 → 200 하나뿐. 판정 코드·밴드·seed 전부 그대로.

  게이트  18회: 1/5 [L5] veto=True stop
  게이트 200회: 1/5 [L5] veto=True stop     ← 동일

  20개 트랙 중 **19개가 동일 판정**. 유일한 변화는 L5/no_distillation 이 T→F
  (엄격해진 방향). 후보 수는 전 트랙에서 늘었다(예: L3 main 8→14) — 탐색은
  실제로 넓어졌는데 **어떤 법칙도 복원 쪽으로 뒤집히지 않았다.**

  → 예산 부족 가설은 반증됐다. `stop` 은 방법에 대한 판정이다.
  → 사전 약속(R22)대로 **4·5단계는 착수하지 않는다.**

  재현된 발견 2건 (두 실행 모두에서 같은 방향):
  (1) **L2 에서 ablation 이 main 을 이긴다** — no_distillation·no_dimensionless 는
      any=T, main 은 any=F. 18회·200회 양쪽에서 재현. C1(증류가 돕는다)·
      C2(무차원화가 돕는다)에 대한 반대 증거이며, 이제 일회성이 아니다.
  (2) **on-manifold 표집은 일을 한다** — L5 의 uniform_off_manifold 만 두 실행
      모두에서 실패(any=F). S2 를 지지하는 직접 증거.

  ★ 가장 유용한 진단 — L1 은 **8개 트랙 전부에서 any_candidate=True** 다(양쪽 실행,
    4트랙×2실행 = 8/8). 법칙의 형태는 언제나 찾아진다. 그런데 최고점수 후보로는
    한 번도 안 올라온다. 이것은 **발견의 실패가 아니라 선택(스코어링)의 실패**다.
    파레토 상에서 옳은 점이 존재하는데 우리 점수 함수가 다른 점을 고른다.
    (이 진단을 근거로 게이트를 통과시키지는 않는다 — 심사 기준을 결과를 보고
     바꾸는 것이기 때문. 발견으로만 기록한다.)

4b단계 DeepLOBCompact: 커밋 31e4a22, 전체 355 passed.
  구현: sd/teacher/window.py(인과 윈도우) + sd/teacher/deeplob.py.
  Teacher 계약(z/predict_path/predict_fill/bottleneck) 동일 — 갈아끼울 수 있다.
  뮤테이션 6건 전부 잡힘(미래 훔쳐보기, 병목 우회, head 별칭, weights 미도달,
  forward-fill 세션 경계 무시 포함).

  ★ 실데이터에서 진짜 버그 하나 발견 — feature 행렬의 **4.2% 행이 비유한(non-finite)**
    이다(유동성 비율 열이 0 근처로 나눔). 행 단위로는 마스크로 넘어가지만 16스텝
    윈도우를 씌우면 창 하나가 그걸 물 확률이 ~50% 라 DeepLOB 손실이 통째로 NaN 이
    됐다. 인과적·세션 한정 forward-fill 로 수정.
    → 이건 DeepLOB 만의 문제가 아니다. 4.2% 비유한은 파이프라인 전체의 성질이다.

  비교 (12종목, 같은 표본·같은 가중치·seed 0, bottleneck=2):
                    epochs=300              epochs=700
    r2_path    Shallow 0.200 / Deep 0.171   Shallow 0.183 / Deep **0.305**
    r2_fill    Shallow 0.363 / Deep 0.254   Shallow 0.420 / Deep **0.479**
    학습시간   7.4s / 24.0s                  9.1s / 56.0s
    z 상관     0.557 / **0.974**             0.588 / **0.956**

  해석 (구현자, 실측 근거 있음): 같은 epoch 에서는 Shallow 우위, 예산을 늘리면
    Deep 이 역전한다. 원인은 co-training 수렴 속도 — 공유 인코더에서 path(MSE)와
    fill(BCE)을 같이 학습하면 Deep 은 초반 수백 epoch 을 path 에만 쓴다
    (같은 시드 corr_fill: epoch 50→0.02, 350→0.09, 700→0.9998, 실측 로그).
    **"같은 epoch"이 "같은 예산"이 아니다.**

  ⚠ 가장 큰 유보 — **원 DeepLOB 의 핵심 귀납 편향인 레벨-압축 컨볼루션이 없다.**
    입력이 원시 (가격,수량) 레벨 순서 텐서가 아니라 Catalog feature 벡터이고
    열 순서가 **알파벳 정렬**이다(`FEATURE_ORDER = tuple(sorted(...))`, sd/ticks.py:18).
    "인접한 두 열이 인접한 호가 레벨"이라는 전제가 없으므로 그 연산을 옮기면
    무관한 값끼리 섞는다. 구현자의 판단이 옳다 — 다만 그 결과 이것은
    **깊은 시계열 모델이지 DeepLOB 본체가 아니다.** 진짜 DeepLOB 을 하려면
    sd/ticks.py 가 레벨 순서 원시 LOB 텐서를 내도록 바꿔야 한다.

  ⚠ z 붕괴 — Deep 의 병목 두 축 상관이 0.95~0.97 이다(Shallow 0.56~0.59).
    폭은 2인데 유효 차원은 1에 가깝다. 증류 대상이 사실상 스칼라라는 뜻이라
    좋은 신호인지 인코더가 둘째 축을 못 쓰는 것인지 이 실험만으로는 못 가린다.

Ruling R24 (DeepLOB E0 실행의 epoch 을 700 으로 올린다 — 사전등록에서 벗어난다):
  사전등록은 epochs=250 을 고정했다. 그런데 250 근방(=300 측정)에서는 DeepLOB 이
  Shallow 보다 **나쁘다.** 그 설정으로 E0 를 돌리면 "더 좋은 교사를 써도 판정이
  안 바뀐다"는 예측을 검증하는 게 아니라 **더 나쁜 교사를 시험**하는 것이 된다.
  epoch 결정 근거는 **게이트 결과와 무관한 교사 적합 품질**이고, 그 측정은
  E0 를 돌리기 **전에** 나왔다(4b 비교표). 결과를 보고 맞춘 것이 아니다.
  교란을 없애기 위해 **두 교사 모두 700 으로** 돌린다:
    (a) v2-shallow@250 — A 의 진행 중 실행 (사전등록 그대로)
    (b) v2-shallow@700 — 대조군, epoch 효과 분리
    (c) v2-deeplob@700 — 교사 효과
  (b) 가 없으면 교사와 epoch 이 섞여 어느 쪽이 원인인지 못 가린다.
  틀렸을 때의 비용: E0 실행 2회(약 8시간) 추가. 그래도 교란된 결론보다 싸다.
  A 의 실행이 끝난 뒤 순차 실행한다 — 동시에 돌리면 CPU 경쟁으로 시간 측정이
  왜곡된다(4b 비교표에서 이미 관측: 같은 epoch 인데 배율이 3.2~6.2배로 흔들렸다).

★★★ 4a단계 — 표본 외 평가 규칙, E0 v2 실행 결과 ★★★
  커밋 c9b28f6(구현) + 13caaaf(보고서). 실행 3h16m, runs/e0-20260905T083613Z/.
  분할: 적합 24종목 / 선택 12종목, 6개 층 각각 4:2. 뮤테이션 4/4 잡힘.
  게이트: **1/5 (L1), verdict stop, 거부권 미발동** (L1·L2 중 L2 만 실패).

              v1 (표본내, 적합36)      v2 (표본외, 적합24)
    L1   복원 False any=True      →   복원 **True**  any=True     ← 바뀜
    L2   복원 False any=False     →   복원 False  any=**True**
    L3   복원 False any=False     →   복원 False  any=False
    L4   복원 False any=False     →   복원 False  any=False
    L5   복원 **True**  any=True  →   복원 **False** any=**False** ← 바뀜

  ★ 사전등록 §3 예측 대조:
    L1 복원된다     → **맞음.** 그리고 기전까지 확인됐다:
       홀함수 오차 v1 0.359/0.325/0.184 (3중 1통과)
                  v2 0.078/0.078/0.078 (3중 3통과)
       간신히 넘은 게 아니라 세 슬라이스가 **동일한 값**이다 — 진짜 홀함수의 모습.
       "표본 내 채점이 노이즈를 외운 후보를 뽑는다"는 진단이 검증됐다.
    L3·L4 여전히 실패 → **맞음** (생성 문제, 채점으로 해결 불가).
    L2 불확실        → any 가 False→True 로 바뀌었다. 정답 후보가 이제 파레토에
       존재하지만 표본 외에서도 1등이 아니다. 즉 L2 는 L1 과 달리 **암기 문제가
       아니라 진짜로 R² 경쟁력 있는 비선형 대안이 있는 것**이다. 채점 규칙만으로
       안 풀린다.
    L5 유지된다      → **틀렸다.** v1 두 실행 모두 복원했는데 v2 에서 실패했고,
       any 까지 False 다 — 파레토에 지수/멱함수 형태 후보가 아예 없다. 생성 실패.

Ruling R25 (L5 퇴행은 교란이다 — 내 설계 결함이고, 적합 종목 수를 36으로 맞춰 재실행한다):
  **내 지시의 결함이다.** "종목 단위로 분할하라"고만 했고, 그러면 적합 집합이
  36 → 24 로 줄어든다는 것을 계산에 넣지 않았다. 그래서 v2 는 **두 가지를 동시에
  바꿨다** — 채점 규칙(표본 내→외)과 적합 종목 수(36→24). 교란된 실험이다.
  구현자는 이 위험을 착수 전에 §1-2 에 적어 두었고, 원인을 못 가린다고 정직하게
  보고했다. 지시가 잘못이었지 실행이 잘못이 아니다.
  L5 만 이 교란에 걸린 이유의 가설: Hawkes 커널은 체결 간격 분포에서 나오는데,
  종목이 줄면 간격 표본의 꼬리가 얇아져 감쇠 형태 자체가 안 드러날 수 있다.
  L1~L4 는 행 단위 feature 라 종목 감소에 덜 민감하다.
  **수정:** per_stratum 을 6 → 9 로 올려 54종목을 뽑고 36 적합 / 18 선택으로 나눈다.
  적합 집합 크기가 v1 과 **정확히 같아지므로**, 남는 차이는 채점 규칙 하나뿐이다.
  틀렸을 때의 비용: E0 실행 1회(약 4~5시간). 이걸 안 하면 v2 의 L5 결과는
  해석 불가능한 채로 남는다 — 그게 더 비싸다.
  우선순위: 이 실행이 DeepLOB E0 보다 먼저다. 이것 없이는 v2 자체가 못 읽힌다.
