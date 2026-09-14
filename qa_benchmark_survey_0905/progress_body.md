# LLM 설계 QA 벤치마크 조사 — 온체인 → 금융 → 희소 도메인 (심층 24편 + 요약 9편)

**질문.** 2024–26 탑티어 AI 학회(ICML·ICLR·NeurIPS·AAAI·KDD·ACL 계열) 벤치마크 중 **QA셋을 LLM으로 설계한 논문**이 온체인 → 금융 → 희소 도메인 순으로 무엇이 있는가. 그 공개 데이터셋(GitHub/HF)을 실제로 열었을 때 **일반 자연어 QA(A/B)** 인지, 우리 `T124_W26_computed_60`처럼 **특수 상황을 가정하는 계산형 QA(C)** 인지.

**결론.** "LLM 설계 + 상황 특정 계산형(C) + 탑티어"를 동시에 만족하는 벤치마크는 세 도메인 어디에도 없다. 근접 사례는 축 하나씩만 채운다.

- **온체인 + C형 + 탑티어**: Spider 2.0 (ICLR 2025 Oral) BigQuery 블록체인 서브셋 27문항 — 같은 USDC 주소·mint/burn 셀렉터·Optimism 브리지 topic 해시가 질문에 등장. 단 사람 작성(LLM은 paraphrase), 답은 SQL 결과 테이블.
- **LLM 설계 + C형(질문 내 규약·형식) + 탑티어**: InfiAgent-DABench (ICML 2024) — GPT-4가 constraints·format까지 생성, 코드 실행으로 정답. 도메인은 일반적.
- **LLM 개입 + C형 + 탑티어**: CRMArena (NAACL 2025, 템플릿→DB 계산→LLM paraphrase) · FDABench (KDD 2026) · ClaimDB (ACL 2026).
- **C형 + 질문 내 규약 명시**: BIRD-Interact/LiveSQLBench (ICLR 2026 Oral) · FinSearchComp T2/T3 (ICLR 2026) — 전문가 작성.
- **온체인 + LLM 설계**: LATTICE · Intent2Tx · EVM-QuestBench — arXiv만, 정답 없음 또는 action.
- **의미적 쌍둥이**: BizFinBench FNC(종목·구간·순유입) — arXiv·중국어. **DABstep**은 형태상 가장 비슷하나 NeurIPS 2025 D&B reject, 규약은 manual.md.
- 희소 도메인의 LLM 생성 세트(ClimaQA·CTIBench·OceanBench·AgMMU·VRSBench)는 전부 지식형 A/B. 계산형(SportsMetrics·MedCalc-Bench)은 LLM이 설계하지 않음.

**우리 세트만 가진 요소**: 판정 규칙 전문의 질문 내 자기완결성 · `out_of_scope`/`indeterminate` 자료 한계 인식 문항 · 문항별 evidence 레코드 동봉 · 78자리 정수·int256/uint256 같은 원장 고유 함정.

**첨부 HTML 구성**: 분류 기준 → 33편 요약표 → 종합 판정 → 4개 군(온체인 8 · 금융 9 · 희소 8 · 계산형 8)별로 논문마다 [원문·코드/데이터 링크 · venue 검증 · 원본 그림 · 쉽게 말하면 · 핵심 요약 · 데이터셋·벤치마크 깊이 보기(구축 방법·규모·verbatim 문항·등급 판정·평가) · 7대 질문 · 리뷰 · 한계 · 우리 QA셋 시사점] → 용도별 인용 후보 → 검증 한계. 원문 PDF 정독 + GitHub/HF 데이터 실열람 기반, 그림 43장.
