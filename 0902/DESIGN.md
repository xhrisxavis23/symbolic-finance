# 0902 — 심볼릭 증류 실험 세팅 설계

_[`../symbolic-distillation-for-market-microstructure.md`](../symbolic-distillation-for-market-microstructure.md) 를 실행 가능한 코드로 옮기기 위한 설계 문서_

| | |
| --- | --- |
| **작성일** | 2026-09-03 |
| **문서 성격** | 설계 사양 + **결정 기록**. 계획서(무엇을 왜)와 구현(어떻게 짜는가) 사이 |
| **범위** | 첫 마일스톤 — 얇은 수직 슬라이스 하나. P0~P9 전체가 아니다 |
| **선행 문서** | 계획서 §3 S−1 ~ S5, §8 구현 계획 |
| **읽는 법** | §1은 무엇이 확정됐는가, §2는 무엇을 실측했는가, §3~§6은 어떻게 짜는가, §7은 완료 기준, §8은 계획서에 되먹여야 할 것 |

---

## 0. 한 문장

**`20260316` KRX 전종목 중 결정론적으로 고른 30종목으로, 데이터 적재부터 정본 백테스트 원장까지 파이프라인 전 단계를 관통시킨다.** 교사와 SR은 인터페이스 뒤에 최소 구현으로 두고, 배관이 검증된 뒤에 진짜 것으로 교체한다.

---

## 1. 결정 기록

브레인스토밍에서 내린 결정을 전부 남긴다. **근거와 함께 남기는 이유는, 나중에 이 중 하나를 뒤집을 때 무엇이 함께 무너지는지 알기 위해서다.**

### D1 — 첫 마일스톤은 얇은 수직 슬라이스다

| | |
| --- | --- |
| **결정** | 소수 종목(30)으로 S−1 → S0 → S1 → S2 → S3 → S5 전 단계를 관통. 진입식 하나가 실제로 정본 백테스트 원장을 뱉는 것까지 |
| **대안** | ① E0 게이트부터 (계획서 P0~P2) ② 전체 P0~P6 ③ 스캐폴드만 |
| **근거** | 계획서가 F7(컴파일 성공률 20% 미만이면 중단)을 폐기 조건으로 두고 P5를 P6 앞에 배치했다. 가장 큰 통합 위험이 "SR 출력 → Catalog AST → 정본 재생"이므로 그것을 먼저 태운다 |
| **버리는 것** | E0 게이트 판정이 뒤로 밀린다. 슬라이스가 통과해도 방법론이 검증된 것은 아니다 |

### D2 — 정본 프레임워크는 vendor 복사한다

| | |
| --- | --- |
| **결정** | `~/tick/proj_claude_tick_finance/Sandbox_12/framework` 의 `.py` (2.5MB) 를 `0902/vendor/framework/` 로 복사. 원본은 건드리지 않는다 |
| **대안** | ① 읽기전용 import + 런타임 확장 ② 상위 폴더 직접 수정 ③ 최소 재구현 |
| **근거** | 세 가지. (1) 작업 범위가 `~/tick/symbolic` 안으로 제한되어 상위 폴더를 수정할 수 없다. (2) `Sandbox_12` 는 어제도 파일이 바뀐 **활성 저장소**라, 라이브 import 는 실험 도중 코드 해시를 움직여 계획서 §5 재현성 요구와 충돌한다. (3) 계획서 P0 이 요구하는 `sqrt`·`tanh` 추가를 사본에서 하면 "P0 이전 1회, 이후 동결" 규칙이 자연스럽게 지켜진다 |
| **버리는 것** | 상위 저장소가 개선되어도 자동으로 따라가지 않는다. `make vendor-diff` 로 수동 확인한다 |
| **불변식** | `VENDOR_MANIFEST.json` 이 원본 경로·복사 시각·파일별 sha256 을 기록한다. `tools/vendor_sync.py --check` 는 사본을 **그 매니페스트 스냅샷**과 대조한다 — **상위 원본과 대조하는 것이 아니다** |

> ⚠️ **정정 (최종 전체 리뷰).** 이 항목은 원래 "원본과의 차이는 PATCHES.md 에 적힌 것뿐이어야 하고, 테스트가 그것을 검사한다" 로 적혀 있었다. 실제로 검사하는 것은 **복사 시점의 스냅샷과의 차이**다. 상위가 그 뒤 움직여도 `--check` 는 통과한다 — 그리고 실제로 움직였다(§8 드리프트 조사). **그것이 D2 가 의도한 동작이다**(동결). 문구만 틀렸다. `vendor_sync.py` 의 성공 메시지와 `README.md` 도 같은 정정이 필요하다.

### D3 — 컴파일러 우선 (접근 B)

| | |
| --- | --- |
| **결정** | 교사는 얕은 MLP, SR 은 최소 백엔드로 두고 **배관을 먼저 완주**시킨다. 그 다음 인터페이스 뒤에서 교체 |
| **대안** | 교사 우선 (계획서 순서 그대로 DeepLOB + PySR 부터) |
| **근거** | (1) 계획서 자신이 P5 를 P6 앞에 두었다. (2) PySR 은 Julia 백엔드라 설치가 무겁고 실패할 수 있는데, 배관이 거기 막히면 안 된다. (3) 재생이 느리다(§2.3) — 배관을 세워야 캐시·병렬화를 어디에 걸지 안다. (4) GPU 여유가 카드당 10.7GB 뿐이라 큰 교사부터 시작하면 배관 검증 전에 자원 문제를 만난다 |
| **버리는 것** | 슬라이스 결과의 **성과 숫자는 의미가 없다.** 최소 교사·최소 SR 이 만든 진입식이므로. 슬라이스가 검증하는 것은 배관이지 알파가 아니다 |

### D4 — SR 백엔드는 플러그형

| | |
| --- | --- |
| **결정** | `SRBackend` 인터페이스: `fit(X, y, w) -> list[Candidate]`, `Candidate = (sympy_expr, complexity, in_sample_score)`. 구현체 둘 — `NaiveBackend`(배관용), `PySRBackend`(설치되면) |
| **근거** | PySR 설치 성패와 배관 진척을 분리한다. 계획서 §8 이 PySR 을 표준으로 지목했으므로 어댑터만 두고 기본값은 나중에 바꾼다 |
| **주의** | `NaiveBackend` 결과로 나온 진입식을 **어떤 가설 판정에도 쓰지 않는다.** 산출물에 `sr_backend` 를 반드시 기록하고, 리포트가 `naive` 인 run 을 눈에 띄게 표시한다 |

### D5 — 교사도 플러그형

| | |
| --- | --- |
| **결정** | `Teacher` 인터페이스: `.z(X)`, `.g_path(z)`, `.g_fill(z)`. 구현체 둘 — `ShallowMLP`(배관용), `DeepLOBCompact`(본 실험) |
| **근거** | D3 과 같다. 병목 `z` 와 두 헤드라는 **계약**만 고정하면 내부는 교체 가능하다 |

### D6 — 층화는 day-1 ST 우주에서 직접 계산한다

| | |
| --- | --- |
| **결정** | 종목별 (스프레드 중앙값, ASK1 심도 중앙값, 20틱 RV 중앙값) 3축 분위 버킷으로 층화. `framework/universe.py` 의 MK01~MK07 core 명단은 **교차확인용으로만** 남긴다 |
| **왜 바꿨나** | vendor 클러스터 명단이 **ST + ETF 혼합 우주**에서 나왔다. 실측: MK05 는 ST 1 / EF 49 로 사실상 ETF 클러스터고, MK01 도 ST 44 / EF 38 이다. ETF 를 제외하면 **MK05 가 1종목으로 붕괴**하고 MK01 은 절반이 날아간다. 게다가 core 명단은 508 종목뿐이라 2,570 전체를 덮지 못한다 |
| **look-ahead 여부** | 아니다. 미래 수익·PnL·체결 라벨을 쓰지 않는다 — 프레임워크가 자기 클러스터링에 대해 편 논리와 같고, 여기서는 **모집단과 정의가 일치**한다는 이점이 추가된다 |
| **확정된 층** | 마찰 3분위 × 잔량 2분할 = 6층. 경계는 아래 |
| **되먹임** | 계획서 §3 S2·S6 반영 완료 (§8) |

**확정된 층 경계** (`20260316`, 정규장 유효 호가틱 ≥ 200 인 2,507종목):

| 마찰층 | `spread_bps` 경계 | 종목 수 | 스프레드 중앙값 | ASK1 잔량 중앙값 | 20틱 변화 중앙값 |
| --- | --- | --- | --- | --- | --- |
| **L** 저마찰 | ≤ 18.37 | 837 | 12.99bp | 157 | 5.95bp |
| **M** 중마찰 | 18.37 ~ 32.59 | 834 | 23.61bp | 140 | 10.08bp |
| **H** 고마찰 | > 32.59 | 836 | 43.02bp | 106 | 16.41bp |

각 마찰층 안에서 `ask1_depth` 중앙 분할 → `L-shallow` 419 · `L-deep` 418 · `M-shallow` 418 · `M-deep` 416 · `H-shallow` 419 · `H-deep` 417.

**정본 클러스터와의 교차확인** (라벨이 있는 416종목): MK01·MK07(고유동성) → L 에, MK03·MK06(고비용) → H 에 몰린다. **두 분류가 일치하되 새 층화는 2,507 전부를 덮고 붕괴하지 않는다.**

### D7 — 슬라이스 종목 선택 규칙

| | |
| --- | --- |
| **결정** | 6개 층에서 **각 5종목**씩 30종목. 각 층 안에서 `issue_code` 오름차순 앞에서부터. 무작위 추출을 쓰지 않는다 |
| **근거** | 뽑는 규칙이 결과를 좌우하면 그 run 을 다시 만들 수 없다. 프레임워크의 `universe.members()` 가 같은 이유로 "파일의 순서가 곧 정본" 이라고 못 박았다 |
| **기록** | 선택된 30종목 명단과 각 층의 경계값을 `runs/<id>/universe.json` 에 남긴다 |

### D8 — 임계 격자는 슬라이스에서 3개로 줄인다

| | |
| --- | --- |
| **결정** | 정본 기본 격자 `{0.50, 0.60, 0.70, 0.80, 0.90, 0.95}` 대신 슬라이스에서는 `{0.70, 0.85, 0.95}` |
| **근거** | 계획서 §8 이 이미 경고한 항이다 — 분위 격자가 재생 횟수에 통째로 곱해진다. 슬라이스의 목적은 배관 검증이므로 3개면 전개·재생·집계 경로를 전부 지난다 |
| **주의** | 이 축소는 **슬라이스 전용**이다. 본 실험의 격자는 계획서 P0 사전 등록에서 따로 정한다 |

### D9 — 임계 종류는 정본 기본값을 그대로 쓴다

| | |
| --- | --- |
| **결정** | `rolling_prior_100_ticks_quantile` — 같은 종목·같은 날 직전 100틱, 현재 틱 제외, 보간 없음(`higher`) |
| **근거** | 하루 입력에서 무누수인 유일한 참조다. 전날 분위수를 쓸 수 없고, 하루 전체 분포로 컷을 잡으면 오후 정보로 오전 결정을 내리게 된다 (계획서 §3 S6 "임계 누수") |

### D10 — 원본 parquet 을 직접 읽는다 (슬라이스 한정)

| | |
| --- | --- |
| **결정** | 슬라이스는 `common-stock-cache` 를 만들지 않고 `run_backtest(root=TICK_ROOT)` 로 원본을 직접 읽는다 |
| **근거** | 스파이크에서 그렇게 동작함을 확인했다(§2.2). 30종목에는 캐시 구축 비용이 회수되지 않는다 |
| **언제 뒤집나** | 전종목으로 넓힐 때. 그때는 계획서 P1 대로 `ALL_STOCKS_ASSET_TYPE_ST` 캐시를 만든다 |

### D11 — 컴파일러는 테스트 우선으로 짠다

| | |
| --- | --- |
| **결정** | `sd/compile/` 의 네 단계는 실패 케이스 테스트를 먼저 쓰고 구현한다 |
| **근거** | 컴파일러가 이 실험의 유일한 신규 위험 지점이고(D1), 실패 모드가 조용하다 — 잘못 번역된 진입식은 예외 없이 그럴듯한 원장을 뱉는다. 나머지 모듈은 통상 개발 흐름으로 간다 |

### D14 — 슬라이스의 SR 어휘는 9개뿐이고, 가격 움직임이 없다

| | |
| --- | --- |
| **관측** | Catalog 36개 feature 중 `dimension == "dimensionless"` 인 것은 **9개**다 (Task 6 구현에서 드러남) |
| **남는 것** | `book_imbalance` · `book_imbalance_velocity` · `queue_imbalance_best` · `ask_depth_concentration` · `bid_depth_concentration` · `deep_depth_imbalance_6_10` · `signed_aggr_flow_20` · `signed_aggr_flow_100` · `spread_to_round_trip_cost_ratio` |
| **빠지는 것** | `price` 6 · `quantity` 9 · `return` 6 · `count` 3 · `boolean` 1 |

**빠진 것 중 둘이 아프다.**

| 빠진 축 | 무엇이 없어지나 |
| --- | --- |
| `return` (6개) | `mid_return_5t/20t/100t_bps` · `microprice_dev_bps` · `microprice_velocity` · `spread_bps` — **가격 움직임이 통째로 없다** |
| `quantity` (9개) | `ofi_cks_1` · `ofi_depth_5` · `ofi_depth_10` — **OFI 가 통째로 없다** |

> ⚠️ **그래서 슬라이스의 SR 은 계획서의 대표 가설을 표현조차 할 수 없다.** 계획서 §3 S3 의 무차원 그룹 사전이 앞세운 두 그룹 — `OFI / Q̄`(L2 법칙의 인수)와 `ΔP / s`(정규화 가격변화) — 은 **둘 다 파생량**이고, 슬라이스는 파생을 미뤘다. 남은 9개로 찾을 수 있는 것은 **호가창 상태 규칙**뿐이며 모멘텀·반전 규칙은 원리적으로 나올 수 없다.

**이것은 결함이 아니라 D3 의 결과다.** 슬라이스는 배관을 검증하지 알파를 검증하지 않는다. 다만 **슬라이스 결과를 방법론에 대한 증거로 읽으면 안 된다는 근거가 하나 더 늘었다** — 최소 교사·최소 SR 에 더해, 어휘 자체가 가설을 담지 못한다.

**그리고 9개 중 둘은 같은 것이다.** 실측:

```text
book_imbalance − (2 × queue_imbalance_best − 1)  →  최대 절대오차 1.11e-16
상관계수 r = +1.000000
공분산 행렬 rank 8 / 9  →  특이(singular)
```

`book_imbalance = (Q_b − Q_a)/(Q_b + Q_a)` 이고 `queue_imbalance_best = Q_b/(Q_b + Q_a)` 이므로 정확히 아핀 변환 관계다. **실효 어휘는 9가 아니라 8이다.**

| 어디에 영향이 있나 | 무엇이 일어나나 |
| --- | --- |
| S2 마할라노비스 거리 | 공분산이 특이해 `inv` 가 못 쓰인다. 유사역행렬로 물러나야 한다 |
| S3 SR 탐색 | 중복 차원을 하나 더 뒤진다. 다만 둘의 아핀 결합은 **단조 흡수**(S3)가 임계로 걷어내므로 진입식 수준에서는 새 구조가 되지 않는다 |

> ⚠️ **정정 (최종 전체 리뷰).** 이 문단은 원래 이렇게 적혀 있었다 — *"어느 feature 가 다른 것의 아핀 변환인가는 Catalog 가 말해 주지 않는다. 그것을 여기서 판단하려면 feature 간 관계표를 새로 만들어야 하는데, 그 표가 곧 두 번째 진실 원천이 된다."* **틀렸다. Catalog 는 말해 준다.**

```python
catalog.feature_use_policy('queue_imbalance_best').threshold_kinds
# → ('derived_duplicate',)        duplicate_of = 'book_imbalance'

catalog.condition_problem('queue_imbalance_best')
# → "queue_imbalance_best 는 임계 조건의 대상이 될 수 없다 (derived_duplicate).
#    대신 book_imbalance 를 쓴다. …"

catalog.quantile_state_key('queue_imbalance_best', 'above')
# → ('book_imbalance', 'above')   ← 프레임워크가 이미 같은 축으로 묶는다
```

`FEATURE_USE_POLICIES` 가 바로 그 관계표이고, `duplicate_of="book_imbalance"` 라고 이름까지 적혀 있다. `condition_problem` 의 docstring 은 **우리와 같은 실측치**를 담고 있다 — *"005930 / 20260316 / 193,602틱: queue_imbalance_best 는 book_imbalance 와 Spearman 1.000000"*.

새 표를 만들 필요가 없었다. **함수 호출 하나였다.** "두 번째 진실 원천이 된다" 는 우려는 정확히 거꾸로였다 — 정본이 가진 표를 **안 보는 쪽**이 두 번째 원천을 만든다.

**대가를 실제로 치렀다.** 엔드투엔드 실행에서 컴파일된 9개 진입식 중 **5개가 동일한 원장**을 냈다 (`e000`·`e001`·`e002`·`e005`·`e008`). 그것은 우연이 아니라 **프레임워크가 미리 금지해 둔 축을 다섯 번 판 결과**다. `e001` 과 `e005` 는 `derived_duplicate` 인 `queue_imbalance_best` 를 임계 조건의 직접 대상으로 썼고, 정본은 그것을 `condition_problem` 으로 거부했을 것이다.

**후속 조치 (이 슬라이스 범위 밖):** `sd/compile/check.py` 의 정적 검사에 `catalog.condition_problem()` 호출을 넣는다. 그것은 파이프라인이 받아들이는 것을 바꾸는 **동작 변경**이므로 별도 태스크와 리뷰가 필요하다 — 지금 끼워 넣으면 이 run 의 결과가 조용히 달라진다.

**거리 계산은 그대로 둔다.** `pinv` 전환(S2)은 rank 결손에 견디게 만든 것이고 그 판단은 유효하다.

**다음 단계에서 무엇을 해야 하나.** `ratio` 와 `rolling_zscore` 로 파생 무차원 열을 만든다. Catalog 가 그 연산자를 이미 갖고 있고 타입 검사가 결과를 `dimensionless` 로 추적한다.

```text
ofi_depth_5 / rolling_mean(bid_queue_depth_at_l1)   →  OFI / Q̄
mid_return_20t_bps / spread_bps                      →  ΔP / s
rolling_zscore(어느 feature든)                        →  dimensionless (sigma)
```

`return` 을 무차원으로 승격하지 않는 이유도 여기 적어 둔다. bps 수익률은 물리적으로는 무차원이지만 **종목을 가로질러 비교 가능하지 않다** — 스프레드 5bp 종목의 10bp 이동과 50bp 종목의 10bp 이동은 다른 사건이다. 계획서가 타깃을 `(spread_bps + 23)` 으로 나누는 것과 같은 이유다. 승격이 아니라 **비를 만들어야** 한다.

### D12 — 산출물은 run 단위로 격리한다

| | |
| --- | --- |
| **결정** | `runs/<UTC타임스탬프>-<config해시8>/` 아래에 전부. `runs/` 는 버전관리 제외 |
| **필수 기록** | 정본 프로필 해시 · vendor 매니페스트 해시 · Catalog 해시 · 종목 명단 · 시드 · SR 백엔드 이름 · 임계 격자 · 재생 시도 횟수 |
| **근거** | 계획서 §5 보고 필수 항목이 이 값들을 요구한다. 나중에 모으려 하면 못 모은다 |
### D13 — SR 은 병목이 아니라 **feature 공간**에서 적합한다

| | |
| --- | --- |
| **결정** | SR 의 입력은 무차원 좌표 `X`, 타깃은 **교사의 출력** `ŷ = teacher.predict_path(X)`. 병목 `z` 를 SR 입력으로 쓰지 않는다 |
| **왜** | `z` 는 학습된 병목이라 Catalog feature 가 아니다. `z` 공간에서 나온 수식은 **실행할 수 없다** — 백테스트가 `z` 를 계산할 방법이 없기 때문이다. 계획서 §3 S1 이 `g: z → ŷ` 를 1순위 SR 대상으로 둔 것은 *해석* 관점의 순서이고, **실행 가능한 진입식을 얻는 경로가 아니다** |
| **그럼 병목의 역할은** | 정규화다. 교사를 `d ≤ 3` 병목으로 조이면 `ŷ` 가 `X` 의 **저복잡도 함수**가 되고, 그래야 SR 이 짧은 수식으로 근사할 수 있다. 이것이 계획서 H4(병목 필요성)의 실질적 내용이며, `d ∈ {1,2,3,8,32}` ablation 이 직접 검정한다 |
| **미루는 것** | 2단 분해 — `enc: X → z` 와 `g: z → ŷ` 를 각각 SR 로 풀어 합성하는 경로. 해석에는 그쪽이 낫지만 오차가 두 번 쌓이고 합성 단계가 하나 더 필요하다. **슬라이스 범위 밖** |
| **되먹임** | 계획서 §3 S1 "SR 대상" 표에 실행 가능성 열 추가 — ✅ 반영 |

---

## 2. 실측으로 확인한 것

설계의 근거가 된 관측이다. **추정이 아니라 이 기계에서 실제로 잰 값이다.**

### 2.1 실행 환경

| 항목 | 값 | 설계에 준 영향 |
| --- | --- | --- |
| GPU | RTX PRO 6000 Blackwell ×2 (96GB) — **각 카드 86GB 이미 점유**, 여유 10.7GB, 사용률 0%, compute app 미검출 | D3·D5 — 큰 교사부터 시작하지 않는다 |
| CPU / RAM | 64코어 / 503GB | 재생 워커 32 (절반만 쓴다) |
| 디스크 | `/home/dgu` 811GB 여유 (89% 사용) | 전종목 캐시는 여유 있으나 여유율이 낮아 감시 필요 |
| Python | 3.10.19 (`/opt/conda/envs/lab`) | vendor 프레임워크가 이 인터프리터에서 import 됨을 확인 |
| 패키지 | torch 2.9.0+cu128 · polars 1.40 · pyarrow 23 · sympy 1.14 · optuna 4.8 · numba 0.61 있음. **PySR 없음** | D4 |

### 2.2 정본 백테스트 스파이크 — 성공

`005930`·`000660`·`035420` × `20260316`, 예시 진입식 (`book_imbalance > q80` AND `spread_to_round_trip_cost_ratio < 0.6`):

```text
elapsed 116.3s   errors: []   profile: CANONICAL_QUEUE_V9 / d0b91a03541c0cec
ledger rows 513 · 254 columns
status  UNFILLED 329 · CENSORED 138 · FILLED 46
cohort  PERSISTENT_ADVERSE 34 · COST_INSUFFICIENT 9 · NET_RECOVERY 3
```

컴파일러가 겨냥할 인터페이스가 확정됐다.

```python
template = {
    "entry_program": {"signal": <Catalog AST>, "warmup_ticks": 0},
    "parameter_interface": {
        "theta_x": {"threshold_source": {"kind": "rolling_prior_100_ticks_quantile"}}},
}
canonical.run_backtest(
    contracts={"q80": template},
    members={"q80": symbols},
    dates=["20260316"],
    output=Path(...),
    parameter_table={"q80:005930:20260316": {"theta_x": 0.80}},
    workers=32, root=TICK_ROOT)
```

`parameter_table` 의 키가 `"{contract_id}:{symbol}:{date}"` 라는 점이 중요하다 — **분위 후보를 계약 ID 로 나누면 한 번의 호출로 격자 전체를 재생할 수 있다.**

### 2.2b Catalog 확장이 무엇을 움직이는가 — 실측

계획서 P0 이 요구하는 `sqrt`·`tanh` 추가가 무엇을 바꾸는지 실제로 재 봤다. 두 해시가 **독립적으로 움직인다.**

| 해시 | 원본 값 | `sqrt` 추가 후 | 근거 |
| --- | --- | --- | --- |
| `canonical.profile_hash()` | `d0b91a03541c0cec` | **`d0b91a03541c0cec` (불변)** | `profile()` 이 catalog 를 참조하지 않는다. 실행 계약과 어휘가 분리되어 있다 |
| `catalog.catalog_hash()` | `448d4a81d9341432` | **바뀐다** | 어휘가 바뀌었으므로 당연하다 |

> 이것이 설계에 주는 것: **실행 계약 해시는 어휘 확장과 무관하게 고정**이므로, `test_vendor.py` 가 `d0b91a03541c0cec` 를 상수로 검사해도 된다. 반면 Catalog 해시는 `PATCHES.md` 적용 **후** 값을 기록하고 그것을 고정한다 — 원본 값이 아니라.

### 2.3 비용

| 관측 | 값 |
| --- | --- |
| 3종목 재생 | 116초 (workers=3) |
| 지배 요인 | `005930` **호가틱 193,602** (전종목 최대). 중앙값 종목은 3,118 호가틱이라 60배 가볍다 |
| 추정 — 전종목 1회 | 64워커로 대략 20~40분 |
| 함의 | 후보 × 분위 × 비교군이 곱해지면 재생이 **SR 탐색보다 비싸진다.** 계획서 §8 의 지적이 실측으로 확인됐다 |

### 2.4 데이터

| 항목 | 값 |
| --- | --- |
| 선택 사다리 | 4,236 → ST 2,720 → 6자리숫자 **2,661** → 틱파일 1개 **2,570** → 호가틱≥200 **2,507** |
| 각 단계가 거른 것 | ETF·ETN 등 1,516 / 영문자코드 59 / 틱파일없음 91 (ambiguous 0건) / 호가틱부족 63 |
| 층화 통계 계산 비용 | 2,570종목 × 5컬럼 → **약 10초** (32스레드, 실측 9.8~10.4초). 1스레드는 28.9초 |

> ⚠️ **이 수치는 한 번 틀렸다가 고쳐진 것이다.** 처음 적은 7.7초는 **심볼→경로 맵을 미리
> 만든 벤치마크 스크립트**의 값이었는데, 계획서에 옮긴 코드는 종목마다 `glob.glob` 을
> 돌았다. 파일 3,726개 폴더를 2,570번 훑는 O(N×M) 이라 실측 108~128초가 나왔고,
> 32스레드가 1스레드보다 **느린 역전**까지 생겼다. 경로 맵을 한 번만 만들도록 고친 뒤
> 위 값이 됐다 — 역전도 사라져 32스레드가 2.9배 빠르다.
> 정본 저장소가 `modules/common_stock_cache.py` 주석으로 바로 이 함정을 경고해 두었는데
> 그대로 밟았다. **측정값을 옮길 때는 어느 구현의 값인지 함께 적는다.**
| 총 행 / 용량 | 37,448,102 행 / 2.4 GB. **정규장 유효 호가틱 23,091,003** |
| 구성 | KOSPI 871 · KOSDAQ 1,699 · 스팩 50 · 관리종목 24 · 우선주 89 |
| 검증된 예시 진입식 | `catalog.infer_expression_type(...)` → `TypeInfo(value_type='boolean', dimension='boolean', unit='0/1')` |

### 2.5 클러스터 명단의 오염 (D6 의 근거)

| 클러스터 | ST | EF | DR | 20260316 ST 우주 잔존 |
| --- | --- | --- | --- | --- |
| MK01 | 44 | 38 | — | 44 / 82 |
| MK02 | 75 | 1 | — | 75 / 76 |
| MK03 | 50 | — | — | 50 / 50 |
| MK04 | 100 | — | — | 100 / 100 |
| **MK05** | **1** | **49** | — | **1 / 50** |
| MK06 | 98 | — | 2 | 98 / 100 |
| MK07 | 48 | 2 | — | 48 / 50 |

---

## 3. 디렉터리 구조

```text
0902/
  DESIGN.md                이 문서
  README.md                실행 순서
  requirements.txt
  Makefile                 slice · test · vendor-diff · lint

  vendor/
    framework/             Sandbox_12/framework 의 .py 사본
    VENDOR_MANIFEST.json   원본 경로 · 복사 시각 · 파일별 sha256
    PATCHES.md             원본 대비 변경 내역 (현재: catalog.py 에 sqrt·tanh)

  sd/                      본 패키지 (symbolic distillation)
    config.py              경로 · 상수 · 해시. 단일 진실 원천
    universe.py            S−1①  종목 선택 · 3축 층화 · 슬라이스 30종목
    ticks.py               S−1①  틱 적재 + feature 행렬 (vendor 위임)
    labels.py              S0     y_path · y_exec · 마찰 정규화
    manifold.py            S2     신뢰 반경 · 층화 표집 · 불확실성 게이팅
    dimensionless.py       S3     36 feature → 무차원 좌표
    teacher/
      base.py              Teacher 인터페이스 (.z .g_path .g_fill)
      shallow.py           ShallowMLP — 배관용
      deeplob.py           DeepLOBCompact — 본 실험용. **슬라이스에서는 만들지 않는다**
      gate.py              S1 검증 게이트
    sr/
      base.py              SRBackend 인터페이스 · Candidate
      naive.py             NaiveBackend — 배관용 템플릿 격자
      pysr_backend.py      PySRBackend — 본 실험용. **슬라이스에서는 만들지 않는다**
    compile/               ★ 이 실험의 유일한 신규 위험 지점
      normalize.py         S5①  정규형 환원 (최외곽 단조 껍질 제거)
      to_catalog.py        S5②  sympy → Catalog AST
      check.py             S5③  정적 검사 (타입 · 어휘 · 실행결속). 인과성은 미구현 — §6.2 참조
      threshold.py         S5④  임계 부착 · 분위 격자 전개
    replay.py              S5⑤  run_backtest 래퍼
    select.py              S4    선택 규칙 (분모 하한 포함)
    report.py              원장 → 요약

  tests/
    test_vendor.py         vendor 무결성 — 원본 대비 차이는 PATCHES.md 뿐
    test_compile.py        ★ 컴파일러 계약 (테스트 우선)
    test_select.py         분모 함정 회귀
    test_replay_smoke.py   2종목 소형 재생
  run_slice.py             엔드투엔드
  runs/                    산출물 (버전관리 제외)
```

> 파일이 커지면 그것 자체가 책임이 섞였다는 신호다. `sd/compile/` 을 네 파일로 나눈 것은 각 단계가 **독립적으로 실패해야** 하기 때문이다 — 어느 단계에서 탈락했는지가 계획서 §5 의 보고 항목(컴파일 실패 후보 목록)이다.

---

## 4. 모듈 경계와 인터페이스

각 단위에 대해 셋을 답할 수 있어야 한다 — 무엇을 하는가, 어떻게 쓰는가, 무엇에 의존하는가.

### 4.1 계약

```python
# sd/teacher/base.py
class Teacher(Protocol):
    def z(self, X: np.ndarray) -> np.ndarray: ...              # (n, d) 병목. 진단·H4 용
    def predict_path(self, X: np.ndarray) -> np.ndarray: ...   # (n,) 마찰 정규화 net 예측
    def predict_fill(self, X: np.ndarray) -> np.ndarray: ...   # (n,) 체결 확률

# SR 은 predict_path 를 타깃으로 쓴다 (D13). z 는 병목 ablation 과 게이트 진단에만 쓴다.

# sd/sr/base.py
@dataclass(frozen=True)
class Candidate:
    expr: sympy.Expr
    complexity: int
    in_sample_score: float
    backend: str            # 'naive' | 'pysr' — 산출물에 반드시 기록 (D4)
    seed: int

class SRBackend(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]: ...

# sd/compile/__init__.py
@dataclass(frozen=True)
class EntryExpression:
    ast: dict                    # Catalog AST. 출력 타입 boolean
    thresholds: dict[str, str]   # 'theta_score' -> 'rolling_prior_100_ticks_quantile'
    source: Candidate
    normal_form: str             # 단조 껍질 제거 후 sympy srepr
    shells: tuple[str, ...]      # 흡수된 껍질. 바깥→안 순서. 리포트가 기록한다

@dataclass(frozen=True)
class CompileFailure:
    candidate: Candidate
    stage: str                   # 'normalize' | 'translate' | 'check'
    reason: str                  # 사람이 읽는 이유. 집계해서 보고한다

# `threshold` 는 stage 가 아니다 — `threshold.attach()` 는 예외를 던지지 않는다.
# 실제로 관측되는 것은 'translate' 뿐이고, 나머지 둘은 방어 경로다 (Task 12 리뷰).
```

### 4.2 의존 방향

```text
config ← universe ← ticks ← labels ← manifold ← dimensionless
                                                      ↓
                                                   teacher
                                                      ↓
                                                     sr
                                                      ↓
                                                  compile   ← vendor.framework.catalog
                                                      ↓
                                                   replay   ← vendor.framework.canonical
                                                      ↓
                                                select · report
```

**단방향이다.** `compile` 이 `teacher` 를 모르고, `replay` 가 `sr` 을 모른다. vendor 는 `compile`·`replay`·`ticks` 세 곳에서만 닿는다 — vendor 를 교체하거나 버전을 올릴 때 봐야 할 곳이 셋뿐이라는 뜻이다.

---

## 5. 데이터 흐름

```text
universe.slice_symbols(date='20260316', n=30)
    → tuple[str]  + runs/<id>/universe.json (층 경계 · 명단)

ticks.load(symbol, date)                      # vendor data.load
    → {bid_price, ask_price, bid_qty, ask_qty, time_s, ...}

ticks.features(arrays)                        # vendor catalog compute
    → {36개 feature: np.ndarray}

labels.build(arrays, features)
    → y_path (마찰 정규화 net) · y_fill (체결 여부) · mask (CENSORED 제외)

dimensionless.transform(features)
    → X (무차원 좌표) · meta (어느 feature 가 어떻게 무차원화됐는가)

manifold.select(X, teacher_ensemble=None)
    → idx · w (중요도 가중치) · flags (on_manifold | probe | synthetic)

teacher.fit(X[idx], y_path[idx], y_fill[idx], w)
    → Teacher

sr.fit(X=X_dimensionless[idx], y=teacher.predict_path(X[idx]), w, feature_names)
    → list[Candidate]              # 수식이 Catalog feature 로 쓰인다 (D13)

compile.pipeline(candidates)
    → list[EntryExpression] · list[CompileFailure]

threshold.expand(entry_exprs, grid={0.70, 0.85, 0.95})
    → contracts: dict[contract_id, template] · parameter_table

replay.run(contracts, symbols, date)
    → runs/<id>/ledger.parquet · ledger_manifest.json

select.rank(ledger)   # scorable ≥ 200 하한, total_net_bps 로 부호 판정
report.write(...)     # runs/<id>/report.md
```

### 누수를 막는 지점

| 어디 | 무엇을 막나 |
| --- | --- |
| `labels.build` | CENSORED 를 0 으로 세지 않고 **마스크로 제외**한다. 나쁜 거래만 사라지는 것을 막는다 |
| `dimensionless.transform` | 스케일 상수를 하루 전체에서 계산하지 않는다 — 롤링만 |
| `threshold` | 분위 컷은 vendor 가 계산한다. 우리가 계산하지 않는다 (D9) |
| `compile.check` | 미래 feature · Catalog 밖 어휘 · 실행 결속 변경 |

---

## 6. 오류 처리와 테스트

### 6.1 오류 처리 원칙

| 상황 | 처리 |
| --- | --- |
| 후보 하나가 컴파일 실패 | **예외 아님.** `CompileFailure` 로 기록하고 계속. 실패율이 산출물이다 (계획서 F7) |
| 종목 하나가 적재 실패 (`InsufficientQuoteData`) | 기록하고 건너뛴다. 부재이지 실패가 아니다 |
| 재생에서 `errors` 가 비어 있지 않음 | **중단.** 원장 일부만 가지고 집계하면 조용히 틀린 수가 나온다 |
| vendor 무결성 검사 실패 | **중단.** 코드 정체성이 흔들리면 그 run 은 재현 불가다 |
| `scorable` 이 하한 미만 | 후보 탈락. 성적을 계산하지 않는다 (계획서 §3 S4) |

### 6.2 테스트

| 파일 | 무엇을 고정하나 |
| --- | --- |
| `test_vendor.py` | 원본 대비 차이가 `PATCHES.md` 에 적힌 것뿐인가. `canonical.profile_hash() == 'd0b91a03541c0cec'` 인가. `catalog.catalog_hash()` 가 패치 후 고정값과 같은가 (§2.2b) |
| **`test_compile.py`** | ★ **테스트 우선 (D11).** 아래 표 |
| `test_select.py` | 결정 3건에 net 양수인 후보가 **선택되지 않는가** (분모 함정 회귀) |
| `test_replay_smoke.py` | 2종목 소형 재생이 원장을 뱉고 `errors == []` 인가 |

컴파일러 테스트가 고정할 것:

| 케이스 | 기대 |
| --- | --- |
| 최외곽 `sqrt(u) > τ` | `u > τ²` 로 환원. **구조가 아니라 임계로 흡수** (계획서 S3 단조 흡수) |
| 최외곽 양수 상수배 `3·u > τ` | `u > τ/3` |
| 안쪽 `sqrt` | 환원하지 않는다. `sqrt` 연산자로 번역 |
| `all(mid_price, ...)` | 타입 오류로 탈락. 실제 메시지: `$.args[0]: bool 이 필요한데 numeric/price 이다` |
| Catalog 밖 feature | 어휘 오류로 탈락 |
| 미래를 보는 feature | ⚠️ **미구현** — 아래 정정 참조 |
| 정상 후보 | `infer_expression_type` 이 `boolean` 을 돌려주고, `run_backtest` 가 받는다 |
| 같은 후보의 단조 변형 여러 개 | **하나로 병합**. `N` 회계가 정확해진다 (계획서 §8 완화 ②) |

> ⚠️ **정정 (최종 전체 리뷰).** 위 표에 **인과성 검사**("미래를 보는 feature → 인과성 오류로 탈락")를 적어 두었으나 **구현되지 않았다.** `check.py` 는 타입·어휘·실행결속 셋만 본다. 현재 Catalog 36개 feature 가 전부 `causal=True` 라 실질 위험은 없지만, **보장한다고 문서에 적힌 것이 보장되지 않는 상태**였다. 계획서 §3 S5 ③ 이 요구한 항목이므로 후속으로 남긴다 — 어휘가 넓어지는 순간(파생 열, PySR) 실제 위험이 된다.

> ⚠️ **함께 발견된 것:** 정본 Catalog 의 **feature 사용 정책**(`feature_use_policy` · `condition_problem` · `guard_axes`)도 `check.py` 가 전혀 보지 않는다. 그 결과 `derived_duplicate` 로 금지된 축을 진입식이 임계 조건의 직접 대상으로 쓸 수 있고, 실제로 그렇게 됐다 — D14 정정 참조.

---

## 7. 완료 기준 — 무엇이 되면 슬라이스가 끝났나

체크리스트다. 전부 참이어야 다음 단계로 간다.

- [ ] `make vendor-diff` 가 `PATCHES.md` 외의 차이를 보고하지 않는다
- [ ] `pytest` 전부 통과. 컴파일러 테스트가 위 8케이스를 덮는다
- [ ] `python run_slice.py` 가 끝까지 돌아 `runs/<id>/` 에 다음을 남긴다
  - [ ] `universe.json` — 30종목 명단 + 층 경계
  - [ ] `candidates.json` — SR 후보 전체 (선택된 것만이 아니라)
  - [ ] `compile_report.json` — 성공 목록 + **실패 목록과 단계별 사유**
  - [ ] `entry_expressions/*.json` — Catalog AST. 각각 `infer_expression_type` 통과
  - [ ] `ledger.parquet` + `ledger_manifest.json` — `errors == []`
  - [ ] `report.md` — `total_net_bps` · `bps_per_decision` · `decisions` · `scorable` · `fills` · `censored` · `unfilled` · 코호트 4분류 분포
  - [ ] `provenance.json` — `canonical.profile_hash()` · vendor 매니페스트 해시 · `catalog.catalog_hash()` (패치 후 값) · 시드 · SR 백엔드 이름 · 재생 시도 횟수 `N`
- [ ] 리포트가 SR 백엔드가 `naive` 임을 눈에 띄게 표시한다 (D4)

**명시적으로 완료 기준이 아닌 것:** 진입식이 돈을 버는 것. 슬라이스는 배관을 검증하지 알파를 검증하지 않는다 (D3).

---

## 8. 계획서에 되먹여야 할 것

이 설계 과정에서 계획서의 사실 오류 하나와 미정의 하나가 드러났다. **슬라이스를 짜기 전에 계획서를 고친다.**

| # | 계획서 위치 | 문제 | 수정 | 상태 |
| --- | --- | --- | --- | --- |
| **1** | §3 S2 클러스터 표, §3 S6 횡단면 분할표·재현성, §5 일치율, §3 S0·S3 의 MK 언급 | MK01~MK07 을 ST 우주의 층으로 썼는데, 그 명단은 ST+ETF 혼합에서 나왔다. ETF 제외 시 **MK05 가 1종목으로 붕괴**하고 MK01 은 절반이 날아간다. core 명단은 508종목뿐이라 2,507 을 덮지도 못한다 | 층화를 day-1 ST 우주에서 직접 계산하는 것으로 교체 (D6). 횡단면 분할을 **L 학습 / M 선택 / H 봉인평가** 로 다시 씀. 정본 클러스터는 교차확인용으로 남기고, 왜 안 쓰는지를 `<details>` 로 기록 | ✅ 반영 |
| **2** | §3 S−1 ① | "선택 결과 2,570" 은 맞지만 중간 단계가 안 보인다. `stock_batch_symbols` 단독은 **2,661** 이다 | 5단 선택 사다리로 교체. 각 조건이 몇 개를 거르는지 명시 (4,236 → 2,720 → 2,661 → 2,570 → 2,507) | ✅ 반영 |
| **3** | §8 계산 비용 | 재생 비용이 추정이었다 | 실측으로 대체 — 3종목 116초, 전종목 1회 20~40분, 층화 통계 약 10초 | ✅ 반영 |
| **4** | §3 S3 단조 흡수 | 계획서가 최외곽 `log` 를 흡수 대상으로 열거했다. 그러나 `log` 는 `u ≤ 0` 에서 미정의라, 벗기면 (a) `to_catalog` 의 `log` 거부가 우회되고 (b) 분위수 모집단이 조용히 바뀐다. 실측: 음수 40% 섞인 표본에서 **12% 의 행이 다르게 선택** | `MONOTONE_UNARY` 에서 `log` 제외. `tanh`·`atan` 은 전체 실수에서 순증가라 유지. **구현이 계획서보다 옳은 케이스다** | ⬜ 계획서 미반영 |
| **5** | §3 S5 산출물 예시 | 계획서가 `all([방향 조건 둘, 비용 가드])` 3조각을 예시로 들고 "세 조각이 두 헤드에 대응한다" 고 썼다. 실제 컴파일러는 **항상 단일 `compare` 하나**를 낸다 | 결과적으로 **`teacher.predict_fill` 이 진입식에 한 번도 닿지 않는다** — S1 게이트에서만 쓰인다. 두 헤드 설계의 두 번째 헤드가 배관에서 소비되지 않는다 | ⬜ 계획서·설계 미반영 |
| **6** | §3 S1 게이트 — 큐 헤드 실측 대조 | 계획서는 `p̂_fill` 을 **정본 원장의 실제 `FILLED/(FILLED+UNFILLED)`** 와 대조하라고 했다. 구현은 **우리 자신의 근사 라벨**(`labels._fill_label`, fill 헤드가 BCE 로 학습한 바로 그 타깃)과 대조한다 | 즉 이 검사는 "모델이 자기 학습 라벨을 잘 맞췄는가" 를 인샘플로 묻는다. `max_abs_gap = 0.0341` 은 **자기일관성의 값이지 큐 모델과의 일치도가 아니다.** 계획서가 이 검사를 넣은 목적("큐 낙관" 탐지)은 원장을 참조점으로 써야 달성된다 | ⬜ 미반영 |

---

## 9. 다음 단계

1. ~~이 문서 검토~~ — 완료
2. ~~계획서 §8 항목 세 개 반영~~ — 완료
3. ~~구현 계획 작성~~ — 완료. [`PLAN.md`](./PLAN.md) 태스크 16개
4. 구현 — Task 1(vendor) → Task 2(Catalog 패치) → … → Task 15(엔드투엔드) → Task 16(S1 게이트)

**계획 작성 중 발견해 이 문서에 되먹인 것:** D13 (SR 을 병목이 아니라 feature
공간에서 적합한다). `z` 공간 수식은 실행할 수 없다는 것을 인터페이스를 확정하다
발견했다. 계획서 §3 S1 의 "SR 대상" 표에도 실행 가능성 열을 추가했다.

**슬라이스 이후의 순서** (이 문서의 범위 밖, 기록만):
교사를 `DeepLOBCompact` 로 교체 → SR 을 PySR 로 교체 → E0 게이트 5종 → 전종목 캐시 → 본 실험 비교군 8종.
