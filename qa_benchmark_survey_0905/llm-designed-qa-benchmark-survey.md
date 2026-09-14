# LLM 설계 QA 벤치마크 조사 (2024–2026): 온체인 · 금융 · 희소 도메인

- 조사일: 2026-09-04
- 방법: 4개 병렬 조사(온체인/크립토 · 금융 · 희소 도메인 · 도메인 무관 "계산형 데이터 QA"). 각 항목은 (1) venue를 OpenReview/proceedings/ACL Anthology 등에서 확인하고 (2) 공개된 데이터(GitHub raw · HuggingFace datasets viewer)를 실제로 열어 원문 질문을 옮겨 적은 뒤 (3) 아래 A/B/C 로 분류했다. 열지 못한 것은 표에 그대로 적었다.
- 비교 기준(참고 세트): `T124_W26_computed_60.csv` — 이더리움 원장 테이블(BigQuery `crypto_ethereum`: token_transfers · logs · transactions · traces · contracts) 1주 구간(W26, block 25,369,414–25,419,597) 위의 60문항. 질문 안에 주소·토큰·블록 구간·이벤트 디코딩 규약(topic0, indexed 파라미터, int256/uint256)·문턱·판정 규칙 전문이 들어 있고, 답은 원장에서 결정론적으로 계산되는 값(integer/record/list/decimal)이며 evidence 레코드가 동봉된다. `out_of_scope`/`indeterminate` 같은 "자료 한계 인식" 문항이 섞여 있다.

## 0. 분류 기준

| 등급 | 뜻 | 예 |
|---|---|---|
| **A** 일반 자연어 QA | 일반인·시험 수준의 지식형 질문, 객관식 포함 | "What does AMM stand for?" |
| **B** 도메인 특화 지식 QA | 전문 지식이 필요하지만 여전히 지식·설명형. 구체적 데이터 인스턴스에 묶이지 않음 | "What threshold actually triggers liquidation on Aave v3?" |
| **C** 상황 특정 계산형 QA | 구체 개체(주소·계정·날짜·블록 구간)와 규칙·문턱·규약이 질문에 명시되고, 답은 제공된 원자료(테이블/DB/로그)에서 결정론적으로 계산되는 값 | 참고 세트 전체 |
| C-lite | 원자료에서 계산은 하지만 소형 단일 테이블이거나 규약·개체 지정이 약함 | TableBench "What is the average number of tropical cyclones per season?" |

핵심 질문은 두 가지다. (i) QA 셋을 **LLM으로 설계**한 탑티어 벤치마크 중 (ii) **C형**이 있는가.

## 1. 온체인 · 크립토 · Web3

| 벤치마크 | venue (검증 방법) | QA 구성 | 공개 (열람 여부) | 원문 질문 예 | 등급 |
|---|---|---|---|---|---|
| **Spider 2.0** — 블록체인 서브셋 | **ICLR 2025 Oral** (OpenReview `XmProj9cPs`, iclr.cc) | 사람 작성 + LLM은 paraphrase만. 547개 lite task 중 27개가 BigQuery 공개 블록체인 데이터(`CRYPTO` 20, `ETHEREUM_BLOCKCHAIN` 3, `GOOG_BLOCKCHAIN` 4). 일부는 `external_knowledge` .md 동봉 | github.com/xlang-ai/Spider2 `spider2-lite.jsonl` (열람, 547행 전수 확인) | sf_bq083: "Can you calculate the daily change in the market value of USDC tokens (address `0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48`) for 2023, based on Ethereum transactions? The change should be computed from minting (input pattern `0x40c10f19%`) and burning (input pattern `0x42966c68%`)… Extract the relevant amount from the 'input' field as a hexadecimal, convert it to millions…" / sf_bq058: "Retrieve all finalized deposits into Optimism at block 29815485 using the Optimism Standard Bridge… Note that, the keccak-256 hash of the Ethereum event signature for DepositFinalized is 0x3303facd…" / sf_bq342: "…token 0x68e54af7… transactions where the address 0x8babf0ba… was the sender or 0xfbd6c6b1… was the receiver, between January 1, 2019, and December 31, 2020…" | **C** |
| **DMind Benchmark** | KDD 2026 (보도자료 + ACM DOI prefix "KDD '26"; ACM 페이지 자체는 차단되어 페이지 수준 미확인) | 전문가 5인 수작업, LLM 생성 아님. 3,150 객관식 + 341 주관식 | HF `DMindAI/DMind_Benchmark` (열람) | "What does AMM stand for?" / "…if the DA layer charges 0.001 ETH per KB, execution layer charges 0.0001 ETH per gas unit… what is the total cost for a 1000-gas transaction worth 5 ETH…?" (가상 숫자 산수) | **A/B** |
| **LATTICE** (Sahara AI) | arXiv 2604.26235 (technical report) | **완전 LLM 생성**: 80 seed → GPT로 240 → Codex로 +960. 정답 없음, GPT-5.2 judge | github.com/SaharaLabsAI/lattice-benchmark (열람) | "What threshold actually triggers liquidation on Aave v3?" / "If Uniswap turns on the fee switch, how does it affect LPs and UNI holders?" | **B** |
| **Intent2Tx** | arXiv 2604.27763 | 실제 메인넷 trace 300일분에서 gemini-2.5-flash가 intent를 역생성 + 수작업 spot-check. QA가 아니라 intent→tx 호출 생성 | anonymous.4open.science (403, 열람 실패) | "Transfer 4.399312393802940721 LPT to 0x0D0707…" → `LivepeerToken.transfer(to=…, value=4399312393802940721)` | QA 아님 (situated action) |
| **EVM-QuestBench** | arXiv 2601.06565 | 사람 task spec + LLM이 NL 템플릿 생성, BSC fork 상태 검증기 | github.com/OpenEdgeHQ/EVM-quest-bench (열람) | "What is the {token_symbol} balance of address {query_address}?" | QA 아님 (situated action) |
| **CryptoBench** (Guo…Mengdi Wang) | arXiv 2512.00417 | 크립토 전문가 위원회 수작성. 월 50문항. 공개 URL 없음 | 미공개 | "From the list of a protocol's top ten holders on Etherscan, identify the two wallets labeled as 'KOL'…" | B / C-lite (라이브 도구 필요, 답이 시간에 따라 변함) |
| **CryptoAnalystBench** (Sentient) | arXiv 2602.11304 | 실제 사용자 쿼리 + 전문가 큐레이션, rubric judge | github.com/sentient-agi/CryptoAnalystBench (열람, 198행) | "Are ETH whales accumulating today?" | **A/B** |
| **TxSum** | EMNLP 2026 (arXiv comments 기준) | 실제 tx 187건 사람 주석 요약. QA 아님 | 데이터 URL 없음 | — | QA 아님 |
| CryptoTrade | EMNLP 2024 | 트레이딩 에이전트, QA 아님 | — | — | — |
| xxcg322/CryptoBench · revflask/blockchain-benchmark | 논문 없음 | 커뮤니티 MCQ | GitHub/HF (열람) | "What is the primary purpose of a 'change' output in a Bitcoin transaction?" | **A** |

존재하지 않거나 QA가 아닌 것: BlockchainBench, Web3Bench, DeFi-QA, ChainQA, OnChain-QA, BlockGPT(2023 이상탐지 모델), DeFiScope(ASE 2025, 분류), CyberChainBench(exploit), ChainBench(Circle, 코드 생성), SolEval(EMNLP 2025, Solidity 코드), CrypQ(DB 벤치마크, NL 없음). ChainReason 저장소는 404.

**온체인 판정.** 탑티어 venue에서 실제 주소·블록·topic 해시를 박은 C형 온체인 질문은 **Spider 2.0(ICLR 2025 Oral)의 블록체인 서브셋뿐**이다. 그러나 (a) 사람이 쓴 질문(LLM은 paraphrase), (b) 답이 SQL 실행 결과 테이블, (c) 데이터가 라이브 BigQuery/Snowflake, (d) 전체의 약 5%다. 전용 Web3 벤치마크(DMind 등)는 전부 A/B다. **LLM 생성 온체인 질문셋은 arXiv에만 있고**(LATTICE·Intent2Tx·EVM-QuestBench) 그중 "제공된 원자료 위의 결정론적 계산 답"을 갖는 것은 없다.

## 2. 금융 · 경제

### 2-1. QA 셋에 LLM이 개입한 것

| 벤치마크 | venue | QA 구성 | 공개 (열람) | 원문 질문 예 | 등급 |
|---|---|---|---|---|---|
| **CRMArena** | NAACL 2025 main (Anthology) | seed 템플릿 14개 → DB에서 정답 계산 → **LLM paraphrase**; DB 자체도 LLM 합성(Salesforce 스키마) | HF `Salesforce/CRMArena` (열람, 1,170행) | "In Q4 of 2021, find the agent who handled over 3 cases and had the minimum handle time. Return only the Id of the agent." → `005Ws000001xYgDIAU` / "Which states closed cases the fastest in May 2021? Return only the two-letter abbreviation…" → `OR` | **C** (기간+문턱+DB 계산; 다만 '핸들 타임' 정의 등 규약은 환경에 숨어 있음; 도메인은 CRM) |
| **Fin-RATE** | KDD 2026 (arXiv abstract 기준, ACM DL 미교차) | DeepSeek-V3.2 생성 → 이중 LLM 검증 → 전문가 확인. SEC 10-K 43사 2020–25 | github.com/jyd777/Fin-RATE (열람) | DR-QA: "What was the total amount spent by VALERO ENERGY CORP/TX on share repurchases during the fourth quarter of 2024, and what was the average price paid per share?" → "$264 million … $127.96" | DR-QA **C-lite** (서술형 답, LLM judge) / EC·LT **B** |
| **FinTMMBench** | ACM MM 2025 (목록 외 venue) | GPT-4o-mini + 템플릿 + CoT 가이드로 생성, 사람 검수 85%+ | github.com/lijunfeng99/FinTMMBench `QA.json` (열람) | "Which company, Charter Communications or Texas Instruments, offers a better return based on their Dividend Yield Ratios, considering Charter Communications has 100000000 shares and…?" / "What is the P/E ratio of Apple on Dec 30, 2022, given 1,000,000 shares?" | **C** (템플릿형; 가정이 질문 안에 있음; 답이 정성적인 경우 많음) |
| **STEER-ME** | NeurIPS 2025 D&B | 사람 템플릿 → gpt-4o style-transfer → 사람 검수. 미시경제 | Streamlit 앱만, 다운로드 미확인 | "If 500 teachers each demand Q = 100 - 2P, what is the aggregate demand? (A)…" | **A/B** (합성 교과서 문제) |
| **FinMathBench** | AAAI 2026 (OJS) | "fully LLM-generated", 공식 DAG 조합 | github.com/ant-research/FinMathBench (xlsx만, 원문 미추출) | — | **B** |
| **PHANTOM** | NeurIPS 2025 D&B | seed 삼중항 사람 검증 후 확장(생성 모델 미확인). 10-K 환각 탐지 | HF `seyled/Phantom_Hallucination_Detection` (열람) | "What percentage of the company's net revenue was generated outside the United States in 2023?" → "41%" | **B / C-lite** |
| **EconLogicQA** | EMNLP 2024 Findings | GPT-4 생성 + 사람 검수 | HF `yinzhu-quan/econ_logic_qa` (열람) | "…Arrange the following events in the logical sequence…" | **A** |
| **BizBench** | ACL 2024 long | FinCode 137 중 91 LLM 생성 후 전문가 검증; 나머지 FinQA/TAT-QA 파생 | HF `kensho/bizbench` (열람) | "What was the percent change in revenue from 2021 to 2022?" | **B / C-lite** |
| **T²-RAGBench** | EACL 2026 | Llama-3.3-70B가 FinQA류 질문을 context-independent 하게 재작성 | 미열람 | "What was the free cash flow of Union Pacific Corporation in 2009?" → $515M | **C-lite** |
| **BizFinBench** | arXiv 2505.19457 (venue 없음) | 실제 사용자 쿼리 + GPT-4o 정제·합성 + 전문가 합의. 중국어 | HF `HiThink-Research/BizFinBench` (열람) | FNC: "同花顺2023年12月1日至2023年12月18日期间的资金净流入是多少亿元？" → 14.8065 (제공된 일별 유입/유출 표에서 순유입 계산) / FTR: "国庆节后平安银行股价首次突破12元的日期是哪一天？" | **C** — 참고 세트와 **의미적으로 가장 닮음**(종목 + 날짜 구간 + 순유입). 단 미출판·중국어·부분 합성 |
| FailSafeQA · UCFE · FinReflectKG · FinMCP-Bench · SEC-QA | arXiv / NAACL 2025 Findings / arXiv / arXiv / FinNLP 워크숍 | LLM 생성이나 A/B 또는 C-lite, 혹은 미공개 | — | SEC-QA 템플릿: "Among {company_names}, what is the {metric2} of the company that has the highest {metric1}?" (규칙 기반 채움, LLM 아님, 미공개) | — |

### 2-2. 사람이 쓴 C형 (대조군)

| 벤치마크 | venue | 원문 질문 예 | 비고 |
|---|---|---|---|
| **FinSearchComp** | **ICLR 2026** (OpenReview PDF 헤더 확인) | T3: "What were the specific dates from January 1, 2020, to December 31, 2024, when London Gold (XAUUSD) dropped by more than $80 in a single day? Please list these dates and the corresponding daily drop in USD (rounded to the nearest integer), presented in a table sorted by date in ascending order." | 문턱·반올림·출력 형식이 질문 안에 있음. **금융에서 가장 C형에 가까움**. 전문가 70인 작성 |
| Finance Agent Benchmark (Vals AI) | arXiv | "How much M&A firepower does Amazon have as of FY2024 end including balance sheet cash, non-restricted cash and other short term investments, and up to 2x GAAP EBITDA leverage?" | 규약 명시, 사람 작성, Zenodo 제한 |
| FinanceQA (AfterQuery) | arXiv | "What is adjusted EBITDA for the year ending in 2024?" → "$11,969 (in millions)" | 규약이 필요하지만 질문에 **없음** |
| BookSQL | NAACL 2024 | "What is the maximum sales for John in the last month?" → SQL | 회계 DB text-to-SQL, 템플릿 치환 |
| FinBen · DocMath-Eval · FinanceMath · FinDER · CFinBench 등 | NeurIPS 2024 D&B · ACL 2024 · ACL 2024 · ICAIF 2025 · NAACL 2025 | — | 사람 주석, B/C-lite |

후보 목록 정정: UCFE는 NAACL 2025 Findings, CFinBench는 NAACL 2025, FinDABench는 COLING 2025, FinDER/FinAgentBench는 ICAIF 2025, CRMArena-Pro는 TMLR. "EconAgentBench"는 존재하지 않음. EDINET-Bench(ICLR 2026)는 분류 태스크, 트레이딩 세트(InvestorBench·DeepFund 등)는 QA가 아님.

**금융 판정.** 목록 venue의 LLM 생성 금융 QA 중 완전한 C형은 없다. 가장 가까운 것은 CRMArena(파이프라인), Fin-RATE DR-QA(개체+기간+값), FinTMMBench(질문 내 가정 명시). 진짜 C형(FinSearchComp T3, Vals FAB)은 전문가 작성이다.

## 3. 희소 · 마이너 도메인

| 벤치마크 | venue | 도메인 | QA 구성 | 공개 (열람) | 원문 질문 예 | 등급 |
|---|---|---|---|---|---|---|
| **ClimaQA** | ICLR 2025 (OpenReview `goFpCuJalN`) | 기후과학 | ClimaGen: gpt-4o-mini가 대학원 교재에서 생성, Gold는 전문가 검증·Silver는 합성 | HF `UCSD-GENIE/ClimaQA` (열람) | "Why does the reflectivity increase between points 3 and 4 in the vertical profile of radar data for stratiform precipitation? a)… b)… c)…" | **B** |
| **CTIBench** | NeurIPS 2024 D&B | 사이버 위협 인텔리전스 | GPT-4o가 ATT&CK/CWE/CAPEC/NIST에서 MCQ 생성, ~3,000개 수동 검증 | HF `AI4Sec/cti-bench` (열람) | "What does mitigation ID M1028 suggest to prevent privilege escalation exploits on a system?" | **B** |
| **OceanBench** (OceanGPT) | ACL 2024 long | 해양학 | DoInstruct 다중 에이전트 LLM 생성 + 전문가 검증 | HF `zjunlp/OceanBench` (열람) | "The pH value of seawater is 8.5, is it acidic or alkaline?" | **B** (A 섞임) |
| **AgMMU** | NeurIPS 2025 D&B | 농업(멀티모달) | GPT-4o가 실제 농부–전문가 대화 116k에서 QA 생성, 사람 검증 | GitHub (열람; HF viewer는 용량 초과) | "What insect is indicated by the image?" | **A/B** |
| **VRSBench** | NeurIPS 2024 D&B | 원격탐사 | GPT-4V 생성 + 사람 검증 | GitHub (열람; 원문 QA는 그림에만) | — | **A** |
| **ChemCoTBench** | NeurIPS 2025 | 화학 | LLM 판단 + 화학자 13인 검토 | HF 카드만 | (SMILES 조작 태스크) | C-ish (규칙 검증형, 표/로그 QA 아님) |
| **DiscoveryBench** DB-SYNTH | ICLR 2025 | 48개 도메인 | 903개 합성 태스크 LLM 생성 | 미열람 | 답이 자유형 가설 | 데이터 기반이나 C 아님 |
| **SportsMetrics** | ACL 2024 long | 스포츠(NBA/NFL) | **프로그램 생성**(LLM 아님): play-by-play 로그 합산, "각 액션 1점" 같은 규칙 변경 변형 | 데이터 저장소 미확인 | play-by-play로 JSON 통계 채우기 | **C-like** (로그 합산 + 규칙 변경) |
| **MedCalc-Bench** | NeurIPS 2024 D&B oral | 임상 계산 | 사람 작성 | HF gated | 환자 노트 + 계산기 질문 | **C** (사람) |
| GeoGrid-Bench | NeurIPS 2025 워크숍 | 기후 격자 데이터 | 전문가 템플릿 8개 + 실제 지역/변수/기간 채움, 정답은 오라클 코드 | arXiv HTML (열람) | "Which region in New York City, NY experienced the largest increase in maximum annual temperature during the historical period?" | C-lite (MCQ) |
| TeleQnA | arXiv/IEEE Network (AI venue 아님) | 통신(3GPP) | GPT-3.5 생성기 + GPT-3.5 검증기 + 전문가 | github.com/netop-team/teleqna (열람) | "What is the maximum number of eigenmodes that the MIMO channel can support?" → min(nt, nr) | **B** |
| LabSafety Bench · MSQA · TeleTables · SecQA · ORAN-Bench · ChemLit-QA · AstroMLab-1 · WeatherQA | NMI / arXiv / arXiv / arXiv / IEEE / IOP / A&C / arXiv | 다양 | 대부분 LLM 생성 | — | — | A/B |

**희소 도메인 판정.** 목록 venue의 LLM 생성 희소 도메인 QA(ClimaQA·CTIBench·OceanBench·AgMMU·VRSBench)는 전부 지식·인식형 A/B다. 결정론적 계산형은 SportsMetrics(프로그램 생성)·MedCalc-Bench(사람)처럼 **LLM이 설계하지 않은** 것뿐이다.

## 4. 도메인 무관: "계산형 데이터 QA" 구조적 이웃

| 벤치마크 | venue | 데이터 | QA 구성 | 공개 (열람) | 원문 질문 예 | 등급 · 참고 세트와의 거리 |
|---|---|---|---|---|---|---|
| **InfiAgent-DABench** | **ICML 2024** (PMLR v235) | GitHub CSV 124개 | **GPT-4가 질문 + `constraints` + `format` 생성 → OpenAI ADA 3회 실행 일치 시 채택 → 사람 검수(85% 적격)** | github.com/InfiAgent `da-dev-questions.jsonl` (열람) | "Generate a new feature called "FamilySize" by summing the "SibSp" and "Parch" columns. Then, calculate the Pearson correlation coefficient (r)…" + constraints "Do not perform any further data cleaning…" + format "rounded to two decimal places" / "Create a new column "AgeGroup" that categorizes passengers into four age groups: 'Child' (0-12…), … calculate the mean fare for each age group." | **C-lite**. **제작 레시피가 가장 가까움**(LLM이 규약·반올림·출력형식까지 질문에 넣고, 코드가 정답 계산). 개체 지정·구간이 없고 도메인 일반적 |
| **DABstep** (Adyen+HF) | NeurIPS 2025 D&B **reject**(OpenReview API) → arXiv 2506.23719 | 결제 데이터 `payments.csv` 138k행 + `fees.json` + `manual.md` | 사람(내부 실제 쿼리 95개 핵심에서 450개 순열) | HF `adyen/DABstep` (열람) | "In January 2023 what delta would Belles_cookbook_store pay if the relative fee of the fee with ID=384 changed to 1?" → `-0.94810300000017` (guideline: 14 decimals) / "For a credit transaction of 1000 euros on SwiftCharge, what would be the most expensive Authorization Characteristics Indicator (ACI)? In the case of a draw… return the ACI with the lowest alphabetical order…" | **C**. 형태상 가장 가까움. 다만 규약은 `manual.md`에, 질문에는 출력형식·동률 규칙만 |
| **Spider 2.0** | ICLR 2025 Oral | BigQuery/Snowflake (블록체인 포함) | 사람 + LLM paraphrase | 열람 | §1 참조 (참고 세트와 **같은 USDC 주소**가 등장) | **C**. 규약은 external .md 또는 암묵 |
| **LiveSQLBench / BIRD-Interact** | BIRD-Interact **ICLR 2026 Oral**; BIRD-Critic NeurIPS 2025 | 신규 PostgreSQL DB + 계층 지식베이스 | 사람 작성, 테스트케이스 채점 | HF `birdsql/*` (열람) | "…warranty status is 'claimed' and have had three or more claims… Just assume they all have 15 years left, produce 500,000 kwh a year, and we sell the power at 12 cents. Give me the grand total." / "…Give me its system unavailability score, just the number, to four decimal points." | **C**. **질문 안에 가정·문턱·반올림 명시**라는 점에서 가장 가까움. 사람 작성 |
| **CRMArena** | NAACL 2025 | LLM 합성 CRM DB | 템플릿 → DB 계산 → LLM paraphrase | 열람 | §2 참조 | **C**. LLM 개입 + 결정론적 답 |
| **FDABench** | KDD 2026 (arXiv comments "Accepted to KDD'26") | 130+ DB + PDF/오디오/비디오 | gold SQL 실행 → **에이전트가 태스크 초안 생성 → 전문가 승인/반려** | HF `FDAbench2026/Fdabench-Lite` (열람) | "…identify the top three customers… what is the ratio of the highest average payment value to the sum of the two São Paulo customers' average payment values, multiplied by 100…?" | **C**(MCQ). 공식이 질문 안에 있으나 객관식 |
| **ClaimDB** | ACL 2026 long | BIRD DB 80개 | BIRD SQL 실행 → **gpt-5가 entailed/contradicted claim 생성** → 다중 judge 필터 | 미열람(논문만) | "In California, the five cities with the lowest K-12 student enrollment are Coulterville, Pinecrest, …" (ENTAILED) | C 인접(검증 태스크). 파이프라인이 참고 세트와 같은 발상 |
| **TableBench** | AAAI 2025 (2차 출처; OJS 403) | 위키형 소형 표 3,681 | 사람 seed → GPT-4 에이전트 생성 → 30% 수동/70% GPT-4 검증, 정답 사람 검토 | HF (열람) | "What is the total GDP (nominal) of all countries with a UN budget greater than 2%?" → 7700143 | **C-lite** |
| **KramaBench** | arXiv 2506.06541 (venue 미확인) | 데이터 레이크 6개(고고학·천문·산불…) | 사람 큐레이션 | GitHub (열람) | "What is the average Potassium in ppm from the first and last time the study recorded people in the Maltese area?" → 8577.5298 | **C**(사람, 규약 암묵) |
| **DSBench** | ICLR 2025 | ModelOff 금융 모델링 + Kaggle | 대회 주최측 작성 | 열람(구조만) | "How many voters are there in the Delta District?" (MC) | C(MC). 규칙은 긴 intro 텍스트에 |
| **QRData** | ACL 2024 Findings | 교재 CSV | 사람 | GitHub (열람) | "Compute the proportion of patients in the treatment group who had a stroke by the end of their first year. Please round to the nearest hundredth." → 0.20 | **C-lite** |
| DataBench / SemEval-2025 T8 | LREC-COLING 2024 | Kaggle 표 65개 | 사람 | HF (열람) | "How many billionaires are there from the 'Technology' category?" → 343 | C-lite |
| AgentBench-DB | ICLR 2024 | WikiSQL류 MySQL | gpt-3.5가 행·SQL 증강 | GitHub (열람) | 조회형 | A / C-lite |
| HG-InsightLog | ACL 2025 Findings | 시스템 로그 | **GPT-4o 생성 8,427 QA** | 원문 미추출 | — | 로그 QA + LLM 생성(참고할 만하나 샘플 미확인) |
| LakeQA | ICML 2026 (Columbia DAPLab 목록) | 9.5TB 데이터 레이크 | 사람 | 미공개 | — | C(사람) |
| TQA-Bench · MMQA · Text2Analysis · Tapilot-Crossing · CORGI · DA-Code · BLADE · ScienceAgentBench · TheAgentCompany · DataSciBench | 철회 / ICLR 2025 / AAAI 2024(미공개) / arXiv / arXiv / EMNLP 2024 / EMNLP 2024 F / ICLR 2025 / NeurIPS 2025 D&B / ACL 2026 F | — | — | — | — | C 아니거나 미확인 |

## 5. 종합 판정

세 축 — (i) QA를 **LLM이 설계**, (ii) **C형**(개체·구간·규약·문턱이 질문 안에, 답은 원자료의 결정론적 값), (iii) **탑티어 venue** — 를 동시에 만족하는 벤치마크는 **온체인·금융·희소 도메인 어디에도 없었다.** 근접 사례는 축 하나씩만 채운다.

| 축 | 채우는 벤치마크 | 못 채우는 축 |
|---|---|---|
| 온체인 도메인 + C형 + 탑티어 | Spider 2.0 블록체인 서브셋 (ICLR 2025 Oral) | 사람 작성(LLM은 paraphrase), 답이 SQL 결과 테이블, 전체의 5% |
| LLM 설계 + C형(질문 내 규약·형식) + 탑티어 | InfiAgent-DABench (ICML 2024) | 도메인 일반적(Titanic 등), 개체·구간 지정 없음 |
| LLM 개입 + C형 + 탑티어 | CRMArena (NAACL 2025), FDABench (KDD 2026, MCQ), ClaimDB (ACL 2026, 검증형) | CRM/일반 DB; 규약이 환경·manual에 있거나 객관식 |
| C형 + 질문 내 규약 명시 + 탑티어 | LiveSQLBench/BIRD-Interact (ICLR 2026 Oral), FinSearchComp T3 (ICLR 2026) | 사람 작성 |
| 온체인 + LLM 설계 | LATTICE, Intent2Tx, EVM-QuestBench | arXiv만; 정답 없음 또는 QA가 아닌 action |
| 참고 세트와 의미적 쌍둥이(종목·구간·순유입) | BizFinBench FNC | arXiv, 중국어, 부분 합성 |
| C형 + 데이터·규약 동봉 | DABstep | NeurIPS 2025 D&B reject; 사람 작성; 규약은 manual.md |

참고 세트만 갖고 있고 위 어디에도 없는 요소: (a) 판정 규칙 **전문**을 질문 안에 넣는 자기완결성(DABstep·Spider 2.0은 외부 manual/.md), (b) `out_of_scope`·`indeterminate`처럼 **자료 한계를 인식해야 맞는 문항**, (c) 답을 만든 **evidence 레코드**를 문항마다 동봉, (d) 78자리 정수·int256/uint256 부호 규약 같은 원장 고유 함정.

## 6. 용도별 인용 후보

- **도메인 선례(온체인 C형)**: Spider 2.0 (ICLR 2025 Oral) `sf_bq083`/`sf_bq058`/`sf_bq444` — 같은 USDC 주소, mint/burn 셀렉터, Optimism 브리지 topic 해시.
- **제작 파이프라인(LLM 생성 → 코드 실행 정답 → 사람 검수)**: InfiAgent-DABench (ICML 2024), CRMArena (NAACL 2025), FDABench (KDD 2026), TableBench (AAAI 2025), ClaimDB (ACL 2026).
- **질문 스타일(규약·문턱·반올림을 질문 안에)**: LiveSQLBench/BIRD-Interact (ICLR 2026 Oral), FinSearchComp T3 (ICLR 2026), DABstep guidelines, Vals Finance Agent Benchmark.
- **A/B형 대조군(도메인 특화지만 지식형)**: DMind (KDD 2026), ClimaQA (ICLR 2025), CTIBench (NeurIPS 2024 D&B), OceanBench (ACL 2024), TeleQnA.
- **결정론 계산 + 규칙 변경 변형**: SportsMetrics (ACL 2024) — "각 액션 1점" 같은 규칙 교체 후 재계산은 참고 세트의 "정의 민감도" 문항(C-T4-11)과 같은 발상.

## 7. 검증 한계

- venue를 2차 출처로만 확인: DMind KDD 2026(보도자료+DOI prefix), TableBench AAAI 2025, Fin-RATE KDD 2026(arXiv 초록), TxSum EMNLP 2026(arXiv comments), KramaBench(미확인). DABstep은 OpenReview API에서 "Rejected Submission" 확인.
- 데이터 미열람: CryptoBench(URL 없음), Intent2Tx(403), FinReflectKG(403), DocMath-Eval·FinanceMath(gated), STEER-ME(Streamlit), FinMathBench(xlsx), HG-InsightLog·VRSBench·ChemCoTBench(원문 추출 실패).
- 금융 조사는 검색 한도 소진으로 WWW/IJCAI 전용 패스를 못 돌렸다. 희소 도메인에서 SportReason(EMNLP 2025 F)·ExpertGenQA(EMNLP 2025 F)는 미확인.

## Sources

Spider2 https://github.com/xlang-ai/Spider2 · https://openreview.net/pdf?id=XmProj9cPs · DMind https://arxiv.org/abs/2504.16116 · https://huggingface.co/datasets/DMindAI/DMind_Benchmark · LATTICE https://github.com/SaharaLabsAI/lattice-benchmark · Intent2Tx https://arxiv.org/abs/2604.27763 · EVM-QuestBench https://github.com/OpenEdgeHQ/EVM-quest-bench · CryptoBench https://arxiv.org/abs/2512.00417 · CryptoAnalystBench https://github.com/sentient-agi/CryptoAnalystBench · TxSum https://arxiv.org/abs/2512.06933 · CRMArena https://aclanthology.org/2025.naacl-long.194/ · https://huggingface.co/datasets/Salesforce/CRMArena · Fin-RATE https://github.com/jyd777/Fin-RATE · FinTMMBench https://github.com/lijunfeng99/FinTMMBench · STEER-ME https://neurips.cc/virtual/2025/poster/121497 · FinMathBench https://github.com/ant-research/FinMathBench · PHANTOM https://neurips.cc/virtual/2025/poster/121830 · EconLogicQA https://aclanthology.org/2024.findings-emnlp.125/ · BizBench https://aclanthology.org/2024.acl-long.452/ · BizFinBench https://huggingface.co/datasets/HiThink-Research/BizFinBench · FinSearchComp https://huggingface.co/datasets/ByteSeedXpert/FinSearchComp · Vals FAB https://arxiv.org/abs/2508.00828 · FinanceQA https://huggingface.co/datasets/AfterQuery/FinanceQA · FinBen https://proceedings.neurips.cc/paper_files/paper/2024/file/adb1d9fa8be4576d28703b396b82ba1b-Paper-Datasets_and_Benchmarks_Track.pdf · ClimaQA https://huggingface.co/datasets/UCSD-GENIE/ClimaQA · CTIBench https://huggingface.co/datasets/AI4Sec/cti-bench · OceanBench https://aclanthology.org/2024.acl-long.184 · AgMMU https://github.com/AgMMU/AgMMU · VRSBench https://github.com/lx709/VRSBench · TeleQnA https://github.com/netop-team/teleqna · SportsMetrics https://aclanthology.org/2024.acl-long.17 · GeoGrid-Bench https://arxiv.org/html/2505.10714v1 · InfiAgent-DABench https://proceedings.mlr.press/v235/hu24s.html · DABstep https://huggingface.co/datasets/adyen/DABstep · https://arxiv.org/abs/2506.23719 · BIRD-Interact https://github.com/bird-bench/BIRD-Interact · LiveSQLBench https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1 · FDABench https://huggingface.co/datasets/FDAbench2026/Fdabench-Lite · https://arxiv.org/abs/2509.02473 · ClaimDB https://aclanthology.org/2026.acl-long.1589/ · TableBench https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench · KramaBench https://github.com/mitdbg/KramaBench · DSBench https://github.com/LiqiangJing/DSBench · QRData https://aclanthology.org/2024.findings-acl.548/ · DataBench https://huggingface.co/datasets/cardiffnlp/databench · HG-InsightLog https://aclanthology.org/2025.findings-acl.1214/ · LakeQA https://daplab.cs.columbia.edu/general/2026/06/29/columbia-daplab-at-icml-2026.html
