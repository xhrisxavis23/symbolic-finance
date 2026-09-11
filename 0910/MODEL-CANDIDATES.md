# DeepLOB 이후 — 심볼릭 증류 교사 후보 조사

조사일: 2026-09-11. 웹 검색(WebSearch)과 논문 원문/추상 확인(WebFetch, arXiv/Tandfonline/PMC/HTML)으로
작성했다. 기억에서 끌어온 수치는 없다 — 확인하지 못한 항목은 전부 "확인 못함"으로 표기했다.

## 0. 우리 상황 재확인 (평가 기준)

`0910/README.md`, `ANALYSIS.md`를 다시 확인한 결과, 지금 교사는 이미 **"DeepLOB 축약판 + 명시적
잠재 병목 `z`(L1 페널티) + 2,570개 종목 풀링 학습"**이다 — 즉 지금도 병목이 있고, 종목별 개별
학습이 아니라 전종목 통합 학습이다. 그런데도 안 본 종목 R²=0.015, 손실의 70~84%가
`PERSISTENT_ADVERSE`(방향 오류)다(`ANALYSIS.md` 91~99행). 그리고 입력은 레벨-정렬 텐서가 아니라
`book_imbalance`, `signed_aggr_flow_20/100`, `spread_to_round_trip_cost_ratio`,
`deep_depth_imbalance_6_10`, `microprice_velocity_over_spread` 같은 **이름 붙은 계산된 특징 벡터**다
(`RESULTS.md` 구조 복원 목록). DeepLOB의 `1×2`/`1×10` 레벨-압축 커널은 이 입력 위에서 아무 의미가
없다 — 이미 스칼라로 압축된 값들을 다시 "레벨"로 취급할 수 없기 때문이다.

즉 **후보를 붙이는 데 드는 진짜 비용은 두 가지 서로 다른 것**이다.
1. **텐서 모양 비용** — CNN·패치 계열은 입력이 `(레벨, 가격/잔량)` 순서를 가진 텐서여야 한다.
2. **정보량 비용** — 계산된 특징 벡터는 레벨별 원시 정보를 이미 버렸다. 아키텍처가 순서에
   무관해도(MLP 등), 논문이 보고한 성능은 원시 레벨 정보를 넣었을 때의 수치다. 우리 특징 벡터로
   그대로 넣으면 아키텍처는 돌아가지만 같은 성능이 나올 근거가 없다.

아래 표의 "입력 표현 요구" 칸에서 이 둘을 구분해서 적었다.

---

## 1. 참고점 — 이미 아는 것 (신규 후보 아님)

### DeepLOB (Zhang, Zohren, Roberts, 2019)
현재 교사의 기반. arXiv:1808.03668. 우리 교사가 이미 이 계열이므로 "교체 후보"에 넣지 않는다.

### TransLOB (Wallbridge, 2020)
- **출처**: "Transformers for Limit Order Books". arXiv:2003.00130 (2020). https://arxiv.org/abs/2003.00130
- **DeepLOB과 차이**: 인과적 팽창 컨볼루션(dilated causal conv)으로 로컬 특징을 뽑고, 그 위에
  마스크된 self-attention을 얹는다. LSTM 대신 attention으로 시간 축을 요약한다.
- **주장하는 성능**: 초록에서 "FI-2010에서 CNN·LSTM 계열을 능가하는 새 SOTA"라고만 주장한다.
  **정확한 F1/정확도 수치는 abstract·초록 수준에서 확인하지 못했다** — 원문 표를 직접 열지 못했다
  (WebFetch가 페이지 요약만 반환했다). 2025년 이후 논문들(TLOB 등)의 비교표에는 TransLOB이
  아예 빠져 있는 경우가 많다 — BiN-CTABL·DeepLOB·TLOB·MLPLOB이 표준 비교군이 된 것으로 보인다.
- **재현 코드**: 공식 https://github.com/jwallbridge/translob . 비교용 재구현 모음
  https://github.com/Jeonghwan-Cheon/lob-deep-learning (DeepLOB·TransLOB·DeepFolio 포함).
- **판단**: 2020년 논문이라 "DeepLOB보다 최신"이라 부르기엔 애매하다(3~6년 전). 아래 §2의
  TLOB/MLPLOB이 TransLOB의 직계 후속이자 사실상의 대체재다. 독자적 후보로 조사를 더 쓰지 않았다.

---

## 2. TLOB & MLPLOB (Berti et al., 2025) — 가장 유력한 후보

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | TLOB (dual-attention transformer) + MLPLOB (같은 논문의 MLP 버전). Berti, Bacciu 등, "TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data", arXiv:2502.15757 (2025). https://arxiv.org/abs/2502.15757 · https://arxiv.org/html/2502.15757v2 |
| 무엇이 DeepLOB과 다른가 | TLOB: CNN 대신 "이중 attention"(시간 축 + 특징/레벨 축을 따로 attention)으로 구조를 잡는다. MLPLOB: attention도 CNN도 없이 GeLU 완전연결층만으로 같은 입력을 처리한다 — "복잡한 구조가 필요한가"를 정면으로 반박하는 실험이다. |
| 주장하는 성능 | **FI-2010, F1**: DeepLOB 71.1/62.4/75.4/77.6 (h=10/20/50/100) 대비 TLOB 81.55/82.68/90.03/92.81, MLPLOB 81.64/84.88/91.39/92.62. BiN-CTABL(2020년대 초 강한 baseline)도 71.5~92.1로 DeepLOB을 이미 능가한다. **NASDAQ 개별종목(Intel/Tesla, LOBSTER)**에서는 격차가 훨씬 작고 장기 호라이즌(h=50,100)에서는 정확도가 40~60%대로 떨어진다 — FI-2010의 우위가 실거래 데이터로 그대로 옮겨지지 않는다는 뜻으로 읽힌다. |
| 병목 가능 여부 | **충족 가능, 단 저자가 설계하지 않음.** MLPLOB은 마지막 완전연결층 차원을 2~8로 좁히기만 하면 바로 병목이 된다 — 우리 문서(`symbolic-distillation-for-market-microstructure.md` 397~423행)의 "패턴 A(명시적 잠재 병목)"를 그대로 적용 가능. TLOB도 최종 분류 헤드 앞에 좁은 층을 끼워 넣으면 된다. 논문 자체에는 병목/증류 관련 언급이 없다. |
| 입력 표현 요구 | **레벨-정렬 원시 벡터가 필요하다.** 입력은 `𝕃(t)∈ℝ^(4L)`(레벨 L개의 매도가·매도잔량·매수가·매수잔량)이다. MLPLOB은 완전연결이라 순서 자체엔 불변이지만, **논문이 보고한 성능은 이 원시 레벨 정보가 다 있을 때의 수치**다. 우리 알파벳정렬 특징 벡터(이미 압축된 OFI류)를 그대로 넣으면 정보가 줄어든 채로 도는 것이고, 같은 성능을 기대할 근거가 없다. → **비용: 레벨별 원시 가격·잔량을 파이프라인에 새로 흘려야 한다.** MLPLOB이라면 텐서 모양 비용은 없지만 정보량 비용은 그대로 남는다. TLOB은 텐서 모양 비용도 추가로 든다. |
| 종목 간 이식 | **검증되지 않음.** FI-2010은 5종목을 섞어 놓고 **시간으로만** train/val/test를 나눈다 — 종목을 아예 나눠서 테스트하지 않는다. NASDAQ 실험(Intel/Tesla)도 "종목별로 따로 학습"(17일 학습/1일 검증/2일 테스트, 종목당 별도 모델)이라고 명시돼 있다. 즉 이 논문은 **우리가 필요로 하는 안 본 종목 이식을 한 번도 테스트한 적이 없다.** |
| 재현 코드 | 있음. https://github.com/LeonardoBerti00/TLOB (TLOB·MLPLOB 둘 다 포함) |
| 구현 비용 | 중간. 레벨별 원시 피처를 추가로 뽑아야 하는 것이 가장 큰 항목(1~2주 추정, 우리 하루치 2,300만 틱 스케일에서 파싱/정규화 재작업 포함). MLPLOB 자체 구현은 가볍다(며칠). TLOB의 dual-attention은 구현+튜닝에 추가로 1주 이상. **핵심 리스크는 코드가 아니라 "종목 풀링 + 병목 + 안 본 종목 평가"를 저자가 한 번도 안 해봤다는 것** — 우리가 그 조합을 처음 시도하는 셈이다. |

---

## 3. LiT: Limit Order Book Transformer (Frontiers in AI, 2025)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | "LiT: limit order book transformer", Frontiers in Artificial Intelligence (2025). https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12555381/ |
| 무엇이 DeepLOB과 다른가 | DeepLOB/TransLOB의 CNN을 완전히 없애고, LOB을 `(레벨, 시간)` 패치로 잘라 transformer self-attention에 바로 태운 뒤 LSTM으로 마무리한다("컨볼루션이 필요 없다"고 명시). |
| 주장하는 성능 | **바이낸스 암호화폐 단일 자산, 밀리초 스냅샷** 기준 정확도. 300–500ms 호라이즌에서 LiT 59.03% vs DeepLOB 57.66%, 300–1000ms에서 68.34% vs 67.76%. **FI-2010이 아니다** — 코인 단일 종목 데이터라 한국 주식 전종목 상황과의 거리가 FI-2010보다 더 멀다. 격차도 1~2%p로 크지 않다. |
| 병목 가능 여부 | 구조적으로는 가능(패치 임베딩 뒤 LSTM 출력을 좁히면 됨)하나, 저자가 시도한 적 없음. |
| 입력 표현 요구 | **레벨-정렬 3D 텐서 필요** — `(호가 깊이 20레벨 × 시간 64스텝 × {가격,잔량} 2채널)`. 우리 특징 벡터로는 대체 불가 — 텐서 모양 자체를 다시 만들어야 한다. |
| 종목 간 이식 | **명시적으로 종목별(자산별) 개별 학습.** 9월 데이터로 학습 후 10~12월로 파인튜닝하는 "시간 전이"만 실험했고, 자산을 바꿔 테스트한 적은 없다. 우리 요구(안 본 종목)와 정반대 설계다. |
| 재현 코드 | **없음.** 논문에 코드 저장소 링크를 찾지 못했다. |
| 구현 비용 | 높음. 텐서 재설계 + 코드 처음부터 작성 + 종목 간 이식 미검증이라는 세 위험이 겹친다. |

---

## 4. LOBFrame / "Deep Limit Order Book Forecasting: A Microstructural Guide" (Briola, Bartolucci, Aste)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | Briola, Bartolucci, Aste, "Deep Limit Order Book Forecasting: A Microstructural Guide" (arXiv:2403.09267, 2024) → 게재본 "Deep limit order book forecasting: a microstructural guide", *Quantitative Finance* 25(7), 2025. https://arxiv.org/abs/2403.09267 · https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/ |
| 무엇이 DeepLOB과 다른가 | **새 아키텍처가 아니다.** 이 논문은 DeepLOB "그 자체"를 케이스 스터디로 써서, 종목의 미시구조 특성(틱 크기)이 예측 가능성을 어떻게 좌우하는지 체계적으로 분석한 **방법론/평가 프레임워크** 논문이다. 우리 문제(안 본 종목 일반화)에 가장 직접적으로 답을 주는 논문이라서 포함했다. |
| 주장하는 성능 | NASDAQ 15개 대형주(AAPL, GOOG, IBM, NVDA 등, 2017–2019, LOBSTER 원자료 — FI-2010의 ≈39.5만 이벤트보다 훨씬 큼). **핵심 발견**: MCC(Matthews 상관계수) 기준 large-tick 종목은 h=10에서 0.29, h=100에서도 0.26 유지. **small-tick 종목은 h=10에서 0.11, h=100에서 0.01 — 거의 무작위.** 즉 **같은 모델, 같은 학습법으로도 종목군에 따라 예측 가능성 자체가 사라진다.** |
| 병목 가능 여부 | 우리 교사와 같은 DeepLOB 기반이므로 우리가 이미 시도한 병목 패턴(A/B)이 그대로 적용된다. 이 논문 자체가 병목/증류를 다루진 않는다. |
| 입력 표현 요구 | DeepLOB 표준 입력(레벨-정렬 텐서)을 그대로 씀. |
| 종목 간 이식 | **이 논문의 중심 주제.** 종목마다 따로 평가하고 "왜 다른가"를 틱 크기로 설명한다 — 우리처럼 2,507개 종목을 하나로 묶어 학습·평가하지는 않지만, **"종목 이질성이 크면 같은 모델도 성능이 무너진다"는 직접적인 경고를 준다.** 우리 R²=0.015가 모델 용량 문제가 아니라 종목 풀 자체의 이질성(틱 크기·유동성 분산) 때문일 수 있다는 가설에 실증적 근거를 더해준다. |
| 재현 코드 | 있음. https://github.com/FinancialComputingUCL/LOBFrame (CC BY-NC-ND 라이선스 — 상업적 재사용 제한 주의). |
| 구현 비용 | 아키텍처 교체 비용은 없음(우리도 이미 DeepLOB 기반). **낮음** — 대신 우리가 얻는 것은 새 모델이 아니라 "종목을 틱 크기·유동성으로 그룹핑해서 따로 평가해봐야 한다"는 진단 방법론이다. 이건 모델 교체보다 먼저 해볼 만한 저비용 진단이다. |

---

## 5. LOBench — LOB 표현학습 벤치마크 (2025)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking", arXiv:2505.02139 (2025). https://arxiv.org/abs/2505.02139 |
| 무엇이 DeepLOB과 다른가 | 특정 아키텍처가 아니라 **"압축·전이 가능한 LOB 표현을 어떻게 뽑을 것인가"를 정면으로 다루는 최초의 체계적 비교 연구**다. 기존 방법들이 표현학습과 다운스트림 과제를 얽어서 end-to-end로 학습해 재사용성이 떨어진다는 문제를 지적하고, 이를 분리해서 평가하는 벤치마크(LOBench)를 만들었다. **우리 제약 1(좁은 병목)과 취지가 가장 가깝다.** |
| 주장하는 성능 | 중국 A주 실거래 데이터. **abstract 수준에서는 구체적 수치(정확도/F1, 표현 차원, 종목 수)를 확인하지 못했다** — 표·본문을 직접 열어보지 못했다(WebFetch가 abstract만 반환). "일반 시계열 표현학습 모델과 task-specific end-to-end 모델을 모두 능가한다"는 주장만 확인했다. |
| 병목 가능 여부 | **가장 유망하지만 차원(2~8)이 실제로 그 정도인지는 확인 못함.** 목적 자체가 "compact"이지만 수치가 몇 차원인지 원문을 못 열어 모른다. 확인이 필요한 최우선 항목. |
| 입력 표현 요구 | 확인 못함(레벨 텐서 여부 abstract에 없음). LOB 원자료 기반일 가능성이 높다(제목상 "표현학습"이므로 원시 입력에서 압축 표현을 뽑는 것이 목적). |
| 종목 간 이식 | 확인 못함. "transferable"이 제목에 들어가지만 이게 종목 간 전이를 뜻하는지 태스크 간 전이(같은 종목에서 여러 다운스트림 과제)를 뜻하는지 abstract만으로는 구분 안 됨. |
| 재현 코드 | 있음(공개 예정/공개됨). https://github.com/financial-simulation-lab/LOBench |
| 구현 비용 | **판단 보류.** 코드와 데이터가 있으니 저비용으로 원문·코드를 먼저 열어보는 것이 다음 조사 단계로 적절하다. 이번 조사에서는 시간상 원문 PDF까지 열지 못했다. |

---

## 6. LOBERT — LOB 메시지용 파운데이션 모델 (2025)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | "LOBERT: Generative AI Foundation Model for Limit Order Book Messages", arXiv:2511.12563 (2025년 11월). https://arxiv.org/abs/2511.12563 |
| 무엇이 DeepLOB과 다른가 | 스냅샷 텐서가 아니라 **개별 주문 메시지(추가/취소/체결)를 토큰으로 취급**하는 BERT류 인코더다. "Masked Message Modeling"으로 사전학습하고 이후 미드프라이스 방향 예측 등으로 파인튜닝한다. DeepLOB 계열과 입력 단위 자체가 다르다(스냅샷이 아니라 이벤트 시퀀스). |
| 주장하는 성능 | "미드프라이스 방향·다음 메시지 예측에서 선두 성능"이라고만 확인했다. **구체적 수치, 비교 baseline, 데이터셋 종목 수는 abstract 수준에서 확인하지 못했다.** |
| 병목 가능 여부 | 확인 못함. BERT 인코더의 `[CLS]`류 풀링 출력을 좁히면 이론적으로는 가능하지만, 임베딩 차원이 보통 수백이라 2~8차원까지 좁히면 성능이 얼마나 남는지는 이 논문에서 다루지 않는다. |
| 입력 표현 요구 | **가장 무겁다.** 스냅샷 텐서도 특징 벡터도 아니고 **개별 메시지 이벤트 스트림**이 필요하다 — 우리 파이프라인은 계산된 특징 벡터 기준이므로, 이걸 쓰려면 메시지 단위 재구성부터 다시 해야 한다(가장 근본적인 재작업). |
| 종목 간 이식 | 확인 못함(per-stock인지 pooled인지 abstract에 없음). "파운데이션 모델"이라는 이름값을 보면 여러 종목/시장으로 사전학습했을 가능성이 높지만 확인된 사실은 아니다. |
| 재현 코드 | 확인 못함(공식 저장소 링크를 찾지 못함). |
| 구현 비용 | **높음.** 입력 표현이 근본적으로 다르므로 우리 파이프라인 전체(특징 계산 이전 단계, 즉 메시지 재구성)를 다시 설계해야 한다. 지금 단계에서는 조사 우선순위가 낮다. |

---

## 7. FastBiNLOB / "The Inference-Compute Frontier..." (2026, 최신)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | C. Evans Hedges, "The Inference-Compute Frontier and a Latency-Efficient Architecture for Limit Order Book Prediction", arXiv:2606.25986 (2026년, 가장 최근 논문). https://arxiv.org/abs/2606.25986 |
| 무엇이 DeepLOB과 다른가 | 이 논문은 우선 "LOB 예측에서 계산량 대비 손실이 거듭제곱 법칙(power law)을 따르는가"를 검증하는 스케일링 연구이고, 그 부산물로 **FastBiNLOB**(하드웨어 친화적 시간축·특징축 분리 mixer, CNN·attention 없음)을 제안한다. |
| 주장하는 성능 | **FI-2010, macro-F1**: "공개된 y₁₀·y₁₀₀ SOTA 타깃을 더 낮은 지연시간으로 초과 달성"(5-seed 실험). 정확한 F1 수치·지연시간(ms/μs)은 abstract에서 확인하지 못했다(PDF 본문에 있으나 이번 조사에서 못 열었다). MLPLOB을 제외한 프론티어로 거듭제곱 법칙을 맞추면 R²=0.941로 MLPLOB 고성능 구간까지 외삽된다는 결과가 흥미롭다 — "MLPLOB류가 이례적으로 계산량 대비 효율이 좋다"는 뜻으로 읽을 수 있다. |
| 병목 가능 여부 | 확인 못함(설계 목적이 지연시간이라 병목/증류는 논문의 관심사가 아닌 것으로 보임). |
| 입력 표현 요구 | 확인 못함. FI-2010을 쓰므로 표준 레벨-정렬 입력을 쓸 가능성이 높다. |
| 종목 간 이식 | FI-2010만 썼다면 §2와 같은 한계(시간 분할, 종목 미분리)를 그대로 물려받을 가능성이 높다 — 확인 못함. |
| 재현 코드 | 있음. https://github.com/evanshedges2/LOB_scaling_and_FastBiNLOB |
| 구현 비용 | 판단 보류. 2026년 6월경 논문이라 커뮤니티 검증이 거의 없다(citation, 재현 시도 사례 없음) — 신선하지만 "아직 아무도 재현 안 해본 위험"이 있다. |

---

## 8. OF-MATNet — 자산 간 주문흐름 attention (2025, ACM ICAIF)

| 항목 | 내용 |
| --- | --- |
| 이름·출처 | "Attention-Based Multi-Asset Order Flow Networks for Enhanced Mid-Price Prediction", 6th ACM International Conference on AI in Finance (2025). https://dl.acm.org/doi/10.1145/3768292.3770430 (검색엔진 요약만 확인 — **원문 abstract를 직접 열지 못했다. ACM이 403을 반환했다.** 아래 내용은 검색 결과 요약에 의존한 것으로, 다른 후보보다 신뢰도가 낮다.) |
| 무엇이 DeepLOB과 다른가 | 종목 하나가 아니라 **나스닥 110개 자산의 OFI를 동시에 입력**해서, 시간·자산·호가레벨 세 축에 대한 attention으로 자산 간 의존성을 모델링한다. 학습 중 롤링 그레인저 인과검정으로 "어떤 자산이 어떤 자산의 선행지표인가"를 골라 쓴다. |
| 주장하는 성능 | "DeepLOB·BiN-CTABL·TLOB 등을 능가"라는 주장만 확인. **정확한 수치는 확인하지 못했다.** |
| 병목 가능 여부 | 확인 못함. |
| 입력 표현 요구 | 레벨별 OFI + **자산 간 관계 행렬**이 필요 — 텐서 모양 비용과 정보량 비용에 더해 **자산 집합 자체를 아키텍처 입력으로 다뤄야 하는 세 번째 비용**이 추가된다. |
| 종목 간 이식 | **오히려 우리 제약 3과 충돌할 위험이 있다.** "어떤 종목이 어떤 종목의 선행지표인가"를 학습 시점의 110개 자산 집합에 맞춰 그레인저 인과검정으로 고정한다 — 이 관계가 학습에 전혀 없던 새 종목(우리는 2,507개 중 매일 다른 홀드아웃)에 그대로 옮겨진다는 보장이 없다. 오히려 "학습 때 본 자산 집합에 의존"하는 구조로 보인다. |
| 재현 코드 | 확인 못함. |
| 구현 비용 | 높음(추정) + 원문을 못 읽어서 확신도가 낮음. **이번 조사에서 가장 근거가 약한 후보이니, 실제로 검토하려면 원문을 반드시 직접 읽어야 한다.** |

---

## 9. 찾아봤지만 없었던 것 (명시적으로 "없다"고 적는다)

지시받은 방향 중 아래는 실제 검색·확인 결과 **해당하는 논문을 찾지 못했다**. 그럴듯한 이름을
지어내지 않기 위해 명시한다.

- **Mamba/S4 기반 "LOB 방향 분류" 논문**: 없음. 찾은 것은 (a) S4를 쓴 **메시지 흐름 생성 모델**
  (Nagy et al., "Generative AI for End-to-End Limit Order Book Modelling", arXiv:2309.00638,
  2023 — 방향 분류가 아니라 다음 메시지를 생성하는 과제), (b) **일봉 단위** 주식가격 예측에 Mamba를
  쓰는 MambaStock(arXiv:2402.18959)·FinMamba(arXiv:2502.06707) — 이들은 OHLCV 일봉을 쓰고 LOB
  틱 데이터를 쓰지 않는다. 둘 다 우리 문제(틱 단위 방향 분류)에 직접 대응하지 않는다.
- **PatchTST를 LOB에 적용한 논문**: 없음. PatchTST(arXiv:2211.14730) 자체는 확인했지만, 이를 LOB
  방향 예측에 적용했다고 명시한 논문은 검색에서 나오지 않았다. LiT·MLPLOB이 "패치"라는 용어를
  쓰지만 PatchTST를 직접 인용/채용한 것은 아니다.
- **시계열 파운데이션 모델(Chronos, TimesFM 등)을 LOB 틱 데이터에 적용한 논문**: 없음. 금융
  파운데이션 모델 서베이(예: arXiv:2511.18578 "Re(Visiting) Time Series Foundation Models in
  Finance")는 LOB을 "향후 적용 가능 영역"으로 언급할 뿐, 실제로 적용한 논문은 찾지 못했다. Kronos
  (arXiv:2508.02739)는 **캔들스틱(K-line, OHLCV 봉)** 파운데이션 모델이라 틱 단위 호가창과는
  입력 단위 자체가 다르다 — 우리 문제에 직접 쓸 수 없다.
- **FI-2010의 상시 유지되는 공개 리더보드**(Papers-with-code류): 찾지 못했다. 대신 각 논문이
  자체적으로 만든 비교표(§2 TLOB 논문의 표가 가장 최근·포괄적)에 의존해야 한다. 즉 "리더보드
  1위"라는 말 자체가 성립하지 않는다 — 논문마다 baseline 재현 방식이 달라 표마다 숫자가 다르다.

---

## 10. 추천 순위와 근거

**전제: 이 순위는 "이 모델을 쓰면 R²가 오른다"는 보장이 아니다.** `ANALYSIS.md`가 이미 지적했듯,
지금 병목이 **모델 용량 부족**인지 **안 본 종목에는 원래 방향 신호가 거의 없는지**(§4 LOBFrame
논문의 small-tick 종목 결과가 이 가능성을 뒷받침한다)는 이번 조사로 가려지지 않는다. 아키텍처를
바꿔도 후자가 맞다면 R²는 그대로일 수 있다.

1. **1순위 — MLPLOB (§2)를 우리 특징 벡터에 먼저 붙여본다.** 텐서 모양 비용이 없다(완전연결이라
   순서 불변). 병목 삽입이 가장 쉽다(층 하나 좁히면 끝). 구현이 가볍다. FI-2010에서 DeepLOB을
   가장 크게 앞선 것도 이 모델이다. **단, 우리 특징 벡터가 이미 압축된 값들이라 논문 수준 성능이
   나올 근거는 없다** — 이게 실패해도 "MLP가 별로다"가 아니라 "우리 특징 벡터의 정보량이 부족하다"
   는 진단이 되어 다음 단계(원시 레벨 정보 추가)로 이어진다는 점에서 저비용 실험으로 가치 있다.
2. **2순위 — TLOB (§2)를, 레벨별 원시 가격·잔량을 파이프라인에 추가한 뒤 시도.** MLPLOB이 정보량
   부족으로 막히면, 같은 데이터에 dual-attention을 얹어 얼마나 회복되는지가 "모델 구조 문제인가
   정보량 문제인가"를 가르는 대조 실험이 된다. 입력 재작업 비용은 어차피 1순위 실험이 실패로
   판명되면 그때 감당한다.
3. **3순위(모델이 아니라 진단) — LOBFrame(§4)식 종목 그룹핑 진단을 먼저 돌린다.** 새 모델을
   붙이기 전에, 지금 교사를 **틱 크기·유동성으로 나눈 종목군별로 R²를 다시 재보는 것**은 코드
   변경이 거의 없다(교사는 그대로, 평가만 나눠서 본다). 만약 특정 종목군에서만 R²가 0에 가깝고
   다른 군은 이미 괜찮다면, 우리 문제는 "이 모델을 못 쓰겠다"가 아니라 "이 종목군은 원래 예측이
   안 된다"는 쪽으로 기울고, 그러면 §2/§3의 모델 교체 시도 자체의 기대 효익을 낮춰 잡아야 한다.
4. **보류 — LOBench(§5), LOBERT(§6), FastBiNLOB(§7), OF-MATNet(§8).** 방향성은 흥미롭지만
   (LOBench는 "압축 표현"이라는 목적이 우리와 가장 가깝고, FastBiNLOB은 가장 최신이라 아직
   검증되지 않았다) 이번 조사에서 원문 본문·수치까지 확인하지 못했다. 위 1~3순위를 먼저 돌리고
   결과가 애매하면 이 중 LOBench를 원문부터 다시 열어보는 것을 다음 조사로 제안한다.
5. **비추천 — LiT(§3), LOBERT(§6), OF-MATNet(§8)을 지금 단계에서 채택.** LiT는 종목 간 이식을
   정반대로 설계했고 코드가 없다. LOBERT는 입력 표현이 근본적으로 다른 메시지 스트림을 요구해
   재작업 범위가 가장 크다. OF-MATNet은 원문을 못 읽었을 뿐 아니라 구조상 종목 간 이식과
   충돌할 위험(그레인저 인과로 고정된 자산 관계)이 있다.

---

## 출처 목록

- DeepLOB — Zhang, Zohren, Roberts (2019). https://arxiv.org/abs/1808.03668
- TransLOB — Wallbridge (2020). https://arxiv.org/abs/2003.00130 · 코드 https://github.com/jwallbridge/translob · 재구현 모음 https://github.com/Jeonghwan-Cheon/lob-deep-learning
- TLOB / MLPLOB — Berti et al. (2025). https://arxiv.org/abs/2502.15757 · https://arxiv.org/html/2502.15757v2 · 코드 https://github.com/LeonardoBerti00/TLOB
- LiT — Frontiers in AI (2025). https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full · https://pmc.ncbi.nlm.nih.gov/articles/PMC12555381/
- Deep Limit Order Book Forecasting: A Microstructural Guide / LOBFrame — Briola, Bartolucci, Aste. https://arxiv.org/abs/2403.09267 · https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/ · 코드 https://github.com/FinancialComputingUCL/LOBFrame
- LOBench (LOB 표현학습) — arXiv:2505.02139 (2025). https://arxiv.org/abs/2505.02139 · 코드 https://github.com/financial-simulation-lab/LOBench
- LOBERT — arXiv:2511.12563 (2025). https://arxiv.org/abs/2511.12563
- FastBiNLOB / Inference-Compute Frontier — Hedges, arXiv:2606.25986 (2026). https://arxiv.org/abs/2606.25986 · 코드 https://github.com/evanshedges2/LOB_scaling_and_FastBiNLOB
- OF-MATNet — ACM ICAIF 2025. https://dl.acm.org/doi/10.1145/3768292.3770430 (원문 접근 실패, 검색 요약만 사용)
- S4 메시지 생성 모델 — Nagy et al., arXiv:2309.00638 (2023). https://arxiv.org/abs/2309.00638
- 생성형 diffusion LOB 시뮬레이션 — Backhouse et al., arXiv:2509.05107 (2025). https://arxiv.org/abs/2509.05107 (방향 예측이 아니라 시뮬레이션 목적이라 후보 표에서 제외, 참고용)
- MambaStock — arXiv:2402.18959. https://arxiv.org/abs/2402.18959
- FinMamba — arXiv:2502.06707. https://arxiv.org/abs/2502.06707
- Kronos (K-line 파운데이션 모델) — arXiv:2508.02739. https://arxiv.org/abs/2508.02739
- Re(Visiting) Time Series Foundation Models in Finance (서베이) — arXiv:2511.18578. https://arxiv.org/abs/2511.18578
- PatchTST — Nie et al., arXiv:2211.14730 (ICLR 2023). https://arxiv.org/abs/2211.14730
- Price predictability in limit order book with deep learning model — Lee, arXiv:2409.14157 (2024). https://arxiv.org/abs/2409.14157 (방향예측이 볼륨 불균형 없이는 약하다는 진단 논문, 후보는 아니지만 우리 진단과 결이 같아 참고)
- An Efficient deep learning model to Predict Stock Price Movement Based on Limit Order Book — arXiv:2505.22678 (2025). https://arxiv.org/abs/2505.22678 (bid/ask 대칭성 활용, Siamese 구조 — 간단히 확인만 하고 표에는 넣지 않음)

## 이번 조사의 한계 (알고 남긴 구멍)

- LOBench(§5)·LOBERT(§6)·FastBiNLOB(§7)·OF-MATNet(§8)은 abstract/검색 요약 수준에서 멈췄다.
  PDF 본문의 표·수치까지 확인하지 못했다 — WebFetch가 PDF를 텍스트로 완전히 풀어주지 못했거나
  (FastBiNLOB, diffusion 논문), 접근이 막혔다(OF-MATNet, ACM 403).
- TransLOB의 정확한 FI-2010 수치를 확인하지 못했다.
- "병목 차원이 실제로 2~8인가"를 명시적으로 확인한 논문은 하나도 없었다 — 전부 우리가 새로
  설계해서 끼워 넣어야 하는 항목이다. 이 조사는 "병목을 끼울 자리가 있는가"까지만 답한다.
