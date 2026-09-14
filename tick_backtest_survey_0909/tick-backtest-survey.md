# 틱데이터 백테스트 심층분석 — 틱 원장으로 굴린 22편과 그것을 판정하는 44편

- 조사일: 2026-09-09
- 질문: 틱데이터를 쓴 연구들은 백테스트를 실제로 어떻게 했는가 — 무엇이 체결을 결정했는가
- 항목마다 백테스트가 실제로 굴러간 해상도를 표시했다 (틱·메시지 / 초~분봉 / 일봉 이상 / 합성 / 실집행)
- 분류 축: 백테스트 사실성 사다리 L0~L6 (체결을 무엇이 결정하는가)
- 한국투자증권·NH·키움 등 국내 증권사의 백테스트 코드 공개 여부를 저장소 단위로 확인
- 끝에 /home/dgu/tick 의 최신 백테스트 코드와 대조
- 원문 PDF 전문 확인 + 저장소 소스 직접 열람 + GitHub API 로 star·라이선스 검증

## 분류 축 — 백테스트 사실성 사다리

| 등급 | 뜻 | 대표 사례 |
|---|---|---|
| **L0 시뮬레이션 없음** | 회귀·이벤트 스터디. 전략을 굴리지 않는다 | Kirilenko 외 *Flash Crash* (JF 2017) · Hasbrouck–Saar (JFM 2013) · Cont–Kukanov–Stoikov OFI (JFE 2014) · Almgren–Chriss (2000) · Holden–Jacobsen (JF 2014) · **국내 증권사 API 저장소 전부** |
| **L1 모형 몬테카를로** | 합성 호가창 위 몬테카를로 | **Avellaneda–Stoikov** (QF 2008, 1,000회 몬테카를로 · 실데이터 0줄) · Bertsimas–Lo (1998) · Guilbaud–Pham (QF 2013, 실틱 보정 + 합성 평가) · mbt_gym |
| **L2 무영향 재생** | 과거 틱 재생 · 중간가/종가 즉시 체결 | Guéant–Lehalle–Fernandez-Tapia (한 종목 하루, 전량 체결) · **Cartea–Donnelly–Jaimungal 기본 설정**(체결확률 1) · **KRX RL 최적실행**(5초 격자, 수수료 무시) · **DeepLOB · FI-2010 계보**(체결 없이 F1) · OPD (분봉 평균가) · Ning 외 (중간가) · DeepScalper · **LLM 에이전트 4편**(일봉·비용 0) · Nagel (RFS 2012) · Chinco 외 (JF 2019) · Qlib · Lean · Freqtrade · Hummingbot · **한국투자증권 backtester(일봉)** |
| **L3 재생+실호가·비용** | 실 BID/ASK 체결 + 수수료 (큐 없음) | **Budish–Cramton–Shim** (QJE 2015, 그림자 호가창) · **Cont–Kukanov** (QF 2017, 큐 소진 + 부분체결 + 리베이트) · Spooner 외 (AAMAS 2018) |
| **L4 재생+큐·지연** | 큐 위치를 세고 지연을 모델링 | **hftbacktest** (큐 모델 4종 + 실측 보간 지연) · **NautilusTrader 1.223+** (`queue_position=True`) · **Nevmyvaka 외 ICML 2006** (우선순위 유지 + 테스트에서 가정 해제) · **Schnaubelt EJOR 2022** (큐 앞 물량 + 거래소별 maker/taker + 롤링 전진검증) · **Moallemi–Yuan** (MBO 인공주문) · **Huang–Lehalle–Rosenbaum** (큐 반응 모형) · **Noble–Rosenbaum–Souilmi** (2026, 경주 조건부 체결 + 자기영향 절제) · JAX-LOB · **우리 코드(지연 없음)** |
| **L5 상호작용 시뮬** | 다중 에이전트 · 자기영향 | **ABIDES / ABIDES-Gym** (다중 에이전트, 자기영향) · Balch 외 (2019) 의 IABS 팔 · MarS · 생성형 world agent (Coletta 외 ICAIF 2022) |
| **L6 실집행** | 실거래 체결 기록 | **Aquilina–Budish–O'Neill** (QJE 2022, LSE 전체 메시지) · Brogaard–Hendershott–Riordan (RFS 2014) · Menkveld (JFM 2013) · Kyle–Obizhaeva (Econometrica 2016) · Almgren 외 (2005, Citigroup 실주문) · Frazzini–Israel–Moskowitz ($1.7조 실체결) |

## 종합 요약표

| 군 | 연구·도구 | venue | 해상도 | 데이터 | 체결 가정 | 비용 | 등급 | 공개 코드 |
|---|---|---|---|---|---|---|---|---|
| 전통 | **The High-Frequency Trading Arms Race (Budish · Cramton · Shim)** | Quarterly Journal of Economics 130(4), 1547–1621 (2015) | 틱·메시지 | CME ES + NYSE SPY 직접 피드 · 7년 · 밀리초 | 싼 쪽 ask 매수 · 비싼 쪽 bid 매도 · 표시 잔량까지 호가 소비 | **양쪽 반스프레드만.** 거래소 수수료·리베이트 미반영 | L3 재생+실호가·비용 | [PDF](https://ericbudish.org/files/high_frequency_trading_arms_race.pdf) · [DOI](https://doi.org/10.1093/qje/qjv027) |
| 전통 | **Quantifying the High-Frequency Trading Arms Race** | Quarterly Journal of Economics 137(1), 493–564 (2022) | 틱·메시지 | LSE 전체 메시지 · FTSE 350 · 43일 · 100ns | 실제 체결 기록 (시뮬레이션 없음) | 경주 가격이 반대 호가라 반스프레드가 암묵 반영 · 수수료는 배제하고 이유를 밝힘 | L6 실집행 | [PDF](https://ericbudish.org/wp-content/uploads/2022/02/Quantifying-the-High-Frequency-Trading-Arms-Race.pdf) · [DOI](https://doi.org/10.1093/qje/qjab032) · [GitHub](https://github.com/ericbudish/HFT-Races) · [데이터](https://doi.org/10.7910/DVN/ZFDWDZ) |
| 전통 | **High-Frequency Trading and Price Discovery** | Review of Financial Studies 27(8), 2267–2306 (2014) | 틱·메시지 | NASDAQ HFT 120종목 · 2008–09 · 밀리초 | 실제 체결 (시뮬레이션 없음) | **수수료·리베이트 반영** — 부호가 뒤집힘 | L6 실집행 | [PDF](https://faculty.haas.berkeley.edu/hender/HFT-PD.pdf) · [DOI](https://doi.org/10.1093/rfs/hhu032) |
| 전통 | **Evaporating Liquidity (Nagel)** | Review of Financial Studies 25(7), 2005–2039 (2012) | 일봉 이상 | CRSP 일간 · 1998–2010 | 종가 체결 가정 (체결가 판본 / 중간가 판본) | **의도적으로 미차감** (공급자가 스프레드를 번다) | L2 무영향 재생 | [PDF](https://www.nber.org/papers/w17653) · [DOI](https://doi.org/10.1093/rfs/hhs066) · [데이터](https://voices.uchicago.edu/stefannagel/files/2021/06/allretout.csv) |
| 전통 | **Sparse Signals in the Cross-Section of Returns** | Journal of Finance 74(1), 449–492 (2019) | 초~분봉 | NYSE TAQ 1분 · 2005–2012 | `\|예측\| > 스프레드` 일 때만 진입 | 스프레드 차감 (정의·시점이 본문에 미명시) | L2 무영향 재생 | [PDF](https://alexchinco.com/sparse-signals-in-cross-section.pdf) · [DOI](https://doi.org/10.1111/jofi.12733) |
| 전통 | **Liquidity Measurement Problems in Fast, Competitive Markets** | Journal of Finance 69(4), 1747–1785 (2014) | 틱·메시지 | TAQ 100종목 · 2008 Q2 · 3,375만 체결 | 해당 없음 | 해당 없음 (비용 **측정** 이 주제) | L0 시뮬레이션 없음 | [PDF](https://host.kelley.iu.edu/cholden/Holden%20and%20Jacobsen%20(2014).pdf) · [DOI](https://doi.org/10.1111/jofi.12127) · [데이터](https://www.smu.edu/-/media/site/cox/faculty/classmaterials/holden_jacobsen_code.zip) |
| 전통 | **Zeroing In on the Expected Returns of Anomalies (Chen & Velikov)** | Journal of Financial and Quantitative Analysis 58(3), 968–1004 (2023) | 일봉 이상 | TAQ·ISSM 유효스프레드 + CRSP/Compustat | 월간 재조정 · 유효스프레드 차감 | **틱에서 잰 유효스프레드** (평균 지불 111bp) | L2 무영향 재생 | [PDF](https://www.federalreserve.gov/econres/feds/files/2020039pap.pdf) · [DOI](https://doi.org/10.1017/S0022109022000874) · [GitHub](https://github.com/velikov-mihail/AssayingAnomalies) · [GitHub](https://github.com/velikov-mihail/Chen-Velikov) · [GitHub](https://github.com/chenandrewy/hf-spreads-all) |
| 전통 | **High frequency trading and the new market makers (Menkveld)** | Journal of Financial Markets 16(4), 712–740 (2013) | 틱·메시지 | Chi-X + Euronext · 네덜란드 지수종목 · 2007-01~2008-06 | 실제 체결 (브로커 ID 익명 매칭으로 HFT 한 곳 특정) | **수수료·리베이트·청산비용 전부 반영** | L6 실집행 | [PDF](https://papers.tinbergen.nl/11076.pdf) · [DOI](https://doi.org/10.1016/j.finmar.2013.06.006) |
| 전통 | **The Flash Crash (Kirilenko · Kyle · Samadi · Tuzun)** | Journal of Finance 72(3), 967–998 (2017) | 틱·메시지 | CFTC 감사추적 · E-mini S&P500 2010년 6월물 · 2010-05-03~06 | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [PDF](https://www.repository.cam.ac.uk/handle/1810/270461) · [DOI](https://doi.org/10.1111/jofi.12498) |
| 전통 | **VPIN 논쟁 (Easley–López de Prado–O'Hara vs Andersen–Bondarenko)** | RFS 25(5) 1457–1493 (2012) vs Journal of Financial Markets 17, 1–46 · 47–52 · 53–64 (2014) | 초~분봉 | E-mini·WTI 1분봉 (ELO) / CME BBO 정답 데이터 (A&B) | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [PDF](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf) · [DOI](https://doi.org/10.1016/j.finmar.2013.05.005) |
| 전통 | **Trading Costs (Frazzini · Israel · Moskowitz)** | 미출판 워킹페이퍼 (SSRN 3229719, 2018-08) · 관련 SSRN 2294498 | 실집행 기록 | **$1.7조 실체결** · 1998-08~2016-06 · 선진 21개 시장 · 약 1만 종목 | 실제 체결가 (99.9% 완결, 평균 지평 1일 미만) | Perold 구현부족 · 제곱근 충격모형 | L6 실집행 | [SSRN](https://papers.ssrn.com/abstract=3229719) |
| 최적 | **High-frequency trading in a limit order book (Avellaneda & Stoikov)** | Quantitative Finance 8(3), 217–224 (2008) | 합성·모형 | **없음** (합성) | 베르누이 추첨 `A·e^{−kδ}` | **없음** | L1 모형 몬테카를로 | [PDF](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) · [DOI](https://doi.org/10.1080/14697680701381228) · [GitHub](https://github.com/ragoragino/avellaneda-stoikov) |
| 최적 | **Guéant · Lehalle · Fernandez-Tapia** | SIAM J. Financial Mathematics 3(1), 740–764 (2012) · Mathematics and Financial Economics 7(4), 477–507 (2013) | 틱·메시지 | AXA 2일 · France Télécom 1일 (Euronext) | 체결이 내 호가에 닿으면 **전량** | 없음 (수수료·리베이트 미반영) | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/1106.3279) · [DOI](https://doi.org/10.1137/110850475) |
| 최적 | **Enhancing Trading Strategies with Order Book Signals** | Applied Mathematical Finance 25(1), 1–35 (2018) | 틱·메시지 | NASDAQ 전체 메시지 11종목 × 2014년 | 기본 체결확률 1 · 민감도에서 상태별 확률 | **maker/taker 수수료 미반영** | L2 무영향 재생 | [PDF](https://ora.ox.ac.uk/objects/uuid:006addde-3a03-4d75-89c1-04b59026e1c0) · [DOI](https://doi.org/10.1080/1350486X.2018.1434009) |
| 최적 | **The Queue-Reactive Model** | Journal of the American Statistical Association 110(509), 107–122 (2015) | 틱·메시지 | Euronext Paris 대형틱 2종목 · 2010-01~2012-03 | FIFO (내 주문은 취소 대상에서 제외) | 명시 없음 | L4 재생+큐·지연 | [arXiv](https://arxiv.org/abs/1312.0563) · [DOI](https://doi.org/10.1080/01621459.2014.982278) |
| 최적 | **Optimal Order Placement in Limit Order Markets (Cont & Kukanov)** | Quantitative Finance 17(1), 21–39 (2017) | 틱·메시지 | TAQ MSFT · NASDAQ+BATS · 2012 Q1 보정 / 4월 평가 | 1분 동안 반대편 시장가 물량 > 앞 큐 Q 면 체결 (초과분만큼 부분체결) | 리베이트 0.2~0.25¢ · taker 0.29¢ · 반스프레드 0.50¢ | L3 재생+실호가·비용 | [arXiv](https://arxiv.org/abs/1210.1625) · [DOI](https://doi.org/10.1080/14697688.2016.1190030) |
| 최적 | **A Model for Queue Position Valuation in a Limit Order Book (Moallemi & Yuan)** | 워킹페이퍼 (2016-12, 2017-06 개정) · SSRN 2996221 | 틱·메시지 | NASDAQ ITCH MBO · 9종목 · 2013-08 | 실제 FIFO 우선순위 · 인공주문 삽입 | NASDAQ maker 리베이트 0.3틱 포함 | L4 재생+큐·지연 | [PDF](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) · [SSRN](https://papers.ssrn.com/abstract=2996221) |
| 최적 | **Bridging the Reality Gap in Limit Order Book Simulation (2026)** | arXiv 2603.24137 (2026-03-25) | 틱·메시지 | Databento MBP-10 · 4종목 · 2023-12~2025-12 | **경주 조건부** — 다음 이벤트가 왕복지연보다 빠르면 미체결 | 내생 전이충격 커널 `(1+t/τ)^{−3/2}` | L4 재생+큐·지연 | [arXiv](https://arxiv.org/abs/2603.24137) · [GitHub](https://github.com/SaadSouilmi/Queue-Reactive) |
| 최적 | **How to Evaluate Trading Strategies: Single Agent Market Replay or Multiple Agent Interactive Simulation?** | arXiv 1906.12010 (2019-06-28) | 틱·메시지 | LOBSTER 1일 1시간 (종목·날짜 미명시) + 합성 ZI 100명 | 실제 매칭엔진 FIFO (양쪽 공통) | 다루지 않음 | L5 상호작용 시뮬 | [arXiv](https://arxiv.org/abs/1906.12010) |
| 최적 | **Optimal Execution of Portfolio Transactions (Almgren & Chriss)** | Journal of Risk 3(2), 5–39 (2000/2001) | 합성·모형 | **없음** | 해당 없음 | 가정된 결정론적 함수 두 개 | L0 시뮬레이션 없음 | [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) · [DOI](https://doi.org/10.21314/JOR.2001.041) |
| 최적 | **Direct Estimation of Equity Market Impact (Almgren · Thum · Hauptmann · Li)** | Risk (2005-07) | 실집행 기록 | **Citigroup 실주문 682,562건 → 필터 후 29,509건** · 2001-12~2003-06 | 실제 체결 (중간가는 NYSE TAQ 로 보완) | 영구 `I = γσ(X/V)(Θ/V)^{1/4}` · 일시 `J = I/2 + sgn(X)·η·σ·(X/VT)^{3/5}` | L6 실집행 | [PDF](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf) |
| 최적 | **Optimal High-Frequency Trading with Limit and Market Orders (Guilbaud & Pham)** | Quantitative Finance 13(1), 79–94 (2013) | 합성·모형 | SOGN.PA 2011-04-18 09:30~16:30 · Quanthouse/OneTick 틱 Level-1 | FIFO 우선순위를 반영한 체결 대용치 | **명시적 수수료 — 리베이트 €0.0008/주, 수수료 €0.0012/주** | L1 모형 몬테카를로 | [arXiv](https://arxiv.org/abs/1106.5040) · [DOI](https://doi.org/10.1080/14697688.2012.708779) |
| 최적 | **Anomalous Price Impact and the Critical Nature of Liquidity (Tóth 외)** | Physical Review X 1, 021006 (2011) | 실집행 기록 | **CFM 자사 선물 거래 · 2007-06~2010-12 · 약 50만 건** | 실제 체결 | 충격 = (메타주문 첫 체결가 → 마지막 체결가) / 당일 σ, 물량은 V 로 정규화 | L6 실집행 | [arXiv](https://arxiv.org/abs/1105.1694) · [DOI](https://doi.org/10.1103/PhysRevX.1.021006) |
| 최적 | **Market Microstructure Invariance: Empirical Hypotheses (Kyle & Obizhaeva)** | Econometrica 84(4), 1345–1404 (2016) | 실집행 기록 | **포트폴리오 전환 2,552건 · 2001–2005 · 주문 439,765건**(매수 201,401 · 매도 238,364). 벤더는 NDA 로 비공개 | 실제 체결 | 전일 종가 대비 구현부족(implementation shortfall) | L6 실집행 | [PDF](https://pages.nes.ru/aobizhaeva/Kyle-Obizhaeva-ECTA-2016-Invariance-with-Supplement.pdf) · [DOI](https://doi.org/10.3982/ECTA10486) |
| AI | **DeepLOB · FI-2010 · 그리고 그 위에 쌓인 10년** | IEEE TSP 2019 (DeepLOB) · Artificial Intelligence Review 2024 (Prata) · Quantitative Finance 2025 (Briola) | 틱·메시지 | FI-2010(10일·5종목) → LOBSTER | **체결 없음.** 중간가 방향 F1 로 대체 | **없음** ("no transaction fees" 명시) | L2 무영향 재생 | [GitHub](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) · [데이터](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) · [GitHub](https://github.com/matteoprata/LOBCAST) · [GitHub](https://github.com/FinancialComputingUCL/LOBFrame) · [GitHub](https://github.com/LeonardoBerti00/TLOB) · [GitHub](https://github.com/lorenzolucchese/deepOBs) |
| AI | **Reinforcement Learning for Optimized Trade Execution (ICML 2006)** | ICML 2006, pp. 673–680 | 틱·메시지 | INET ECN 밀리초 1.5년 · 3종목 | 실호가창 + **시간 우선순위 유지** | **0, 그리고 명시적으로 밝힘** | L4 재생+큐·지연 | [PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) · [DOI](https://doi.org/10.1145/1143844.1143929) |
| AI | **Universal Trading for Order Execution with Oracle Policy Distillation (AAAI 2021)** | AAAI 2021, 35(1), 107–115 | 초~분봉 | 중국 A주 **분봉** · 2017-01~2019-06 | **분봉 평균가**(市場 VWAP)에 즉시 체결 | **명시적으로 무시** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2103.10860) · [DOI](https://doi.org/10.1609/aaai.v35i1.16083) · [GitHub](https://github.com/microsoft/qlib) · [웹](https://seqml.github.io/opd/) |
| AI | **Double Deep Q-Learning for Optimal Execution (Ning · Lin · Jaimungal)** | Applied Mathematical Finance 28(4), 361–380 (2021/2022) | 초~분봉 | 9종목 초 단위 중간가 · 2017-01~2018-03 | **중간가**, 스프레드 무시 | 적합된 이차 벌점 a = 0.01 (표본 내 보정) | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/1812.06600) · [DOI](https://doi.org/10.1080/1350486X.2022.2077783) |
| AI | **Deep RL for the optimal placement of cryptocurrency limit orders (Schnaubelt)** | European Journal of Operational Research 296(3), 993–1006 (2022) | 틱·메시지 | 3개 거래소 · 18개월 · 체결 3억 건 | 2단계 가격-시간 우선순위 + **큐 앞 물량 반영** | **거래소별 maker/taker 실제 요율** (0~16 / 20~30 bp) | L4 재생+큐·지연 | [PDF](https://www.econstor.eu/bitstream/10419/216206/1/1696077540.pdf) · [DOI](https://doi.org/10.1016/j.ejor.2021.04.050) |
| AI | **DeepScalper (CIKM 2022)** | CIKM 2022, pp. 1858–1867 | 초~분봉 | 분봉 OHLCV + 5호가 · 6 자산 | **분봉 종가**에 즉시 체결 | δ = 2.3e-5 / 3e-6 + **5배 레버리지** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2201.09058) · [DOI](https://doi.org/10.1145/3511808.3557283) · [GitHub](https://github.com/TradeMaster-NTU/TradeMaster) |
| AI | **MarS (ICLR 2025) · LOB-Bench (ICML 2025)** | ICLR 2025 (MarS) · ICML 2025 (LOB-Bench) | 틱·메시지 | 중국 상위 500종목 주문 토큰 160억 개 | 실제 청산소 매칭 + 생성 흐름 되먹임 | 다루지 않음 | L5 상호작용 시뮬 | [arXiv](https://arxiv.org/abs/2409.07486) · [GitHub](https://github.com/microsoft/MarS) · [웹](https://lobbench.github.io/) · [GitHub](https://github.com/peernagy/lob_bench) |
| AI | **LLM 트레이딩 에이전트 4편 (FinAgent · TradingAgents · FinCon · FinMem)** | KDD 2024 / ICML 2025 워크숍 / NeurIPS 2024 poster / AAAI Symposium Series (3쪽) | 일봉 이상 | **일봉** · 60~400 거래일 · 5~8 대형주 | 일간 가격에 즉시 체결 (사이징 없음/1주) | **전부 0** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2402.18485) · [DOI](https://doi.org/10.1145/3637528.3671801) · [GitHub](https://github.com/TauricResearch/TradingAgents) · [GitHub](https://github.com/The-FinAI/FinCon) · [GitHub](https://github.com/pipiku915/FinMem-LLM-StockTrading) |
| AI | **Hendricks & Wilcox (IEEE CIFEr 2014)** | IEEE CIFEr 2014 | 초~분봉 | TRTH JSE ALSI 166종목 · 2012년 · 5호가 → **5분 집계** | 5분 평균가 | **Almgren-Chriss 일시충격만, 영구충격 ρ = 0** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/1403.2229) · [DOI](https://doi.org/10.1109/CIFEr.2014.6924109) |
| AI | **HALOP (IJCAI 2022)** | IJCAI 2022, pp. 3912–3918 | 초~분봉 | CSI 300 5호가 · 학습 2010-01~2019-12 / 테스트 2020-01~06 | 지연 3초 후 체결 · **총 재고의 1/10 초과 주문은 실행하지 않음** | 수수료·인지세 없음. 대신 취소율 50% 초과 시 −5bp 벌점 | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2207.11152) · [DOI](https://doi.org/10.24963/ijcai.2022/543) |
| AI | **Karpe 외 (ICAIF 2020)** | ICAIF 2020 | 틱·메시지 | NASDAQ → LOBSTER 형식 · CSCO·IBM·INTC·MSFT·YHOO · **2003-01-13~02-06** | ABIDES 매칭엔진 FIFO (배경은 재생) | **언급 없음** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2006.05574) |
| AI | **MASTER (AAAI 2024) · StockFormer (IJCAI 2023)** | AAAI 2024 · IJCAI 2023, pp. 4766–4774 | 일봉 이상 | **일봉** — MASTER: CSI300/CSI800, StockFormer: CSI-300·NASDAQ-100·암호화폐 | 일간 가격 즉시 체결 | MASTER **없음** · StockFormer 있다고만 하고 **수치 미기재** | L2 무영향 재생 | [DOI](https://doi.org/10.24963/ijcai.2023/530) · [GitHub](https://github.com/SJTU-DMTai/MASTER) · [GitHub](https://github.com/gsyyysg/StockFormer) |
| AI | **FinRL · FinRL-Meta · TradeMaster · MacroHFT** | NeurIPS 2022/2023 D&B (FinRL-Meta·TradeMaster) · KDD 2024 (MacroHFT) | 일봉 이상 | 일봉~분봉 (호가창 없음) | **종가 전량 체결** | 고정 비율 `transaction_cost_pct` | L2 무영향 재생 | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) · [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta) · [GitHub](https://github.com/TradeMaster-NTU/TradeMaster) · [GitHub](https://github.com/ZONG0004/MacroHFT) |
| 백테스트 | **Pseudo-Mathematics and Financial Charlatanism / The Probability of Backtest Overfitting** | Notices of the AMS 61(5), 458– (2014) · Journal of Computational Finance (2016-09) | 해당 없음 | 이론 (성과 행렬) | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [PDF](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf) · [DOI](https://doi.org/10.1090/noti1105) · [GitHub](https://github.com/esvhd/pypbo) |
| 백테스트 | **The Deflated Sharpe Ratio** | Journal of Portfolio Management 40(5), 94–107 (2014) | 해당 없음 | 성과 시계열 + 시행 분포 | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [DOI](https://doi.org/10.3905/jpm.2014.40.5.094) |
| 백테스트 | **CPCV 대 워크포워드** | Knowledge-Based Systems 305, 112477 (2024) | 합성·모형 | 합성 (Heston·점프·국면전환) | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [DOI](https://doi.org/10.1016/j.knosys.2024.112477) · [GitHub](https://github.com/RiskLabAI/RiskLabAI.py) |
| 백테스트 | **보정의 한계** | arXiv 2608.27734 (2026) · SSRN 7346738 (미심사) · arXiv 1906.00573 · 2606.01650 | 일봉 이상 | 453종목 PIT + 39 ETF (Gençay) | 해당 없음 | Gençay 는 거래·충격·대차 비용 포함 | L0 시뮬레이션 없음 | [arXiv](https://arxiv.org/abs/2608.27734) · [GitHub](https://github.com/eslazarev/purged-cross-validation) |
| 백테스트 | **How Much Sharpe is Illusory?** | SSRN 프리프린트 (2026) · DOI 10.2139/ssrn.7350238 · 미심사 | 일봉 이상 | Binance USDT-perp 137종목 · 2020–2024 | 팩터 재조정 (체결 모형 미상) | **명시적 거래비용 — 그리고 이것이 격차의 대부분** | L2 무영향 재생 | [DOI](https://doi.org/10.2139/ssrn.7350238) |
| 백테스트 | **다중검정 문턱 논쟁** | Review of Financial Studies 29(1), 5–68 (2016) 외 | 일봉 이상 | 출판 팩터 300+ / 채굴 전략 136,000 | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [PDF](https://academic.oup.com/rfs/article-pdf/29/1/5/24450794/hhv059.pdf) · [DOI](https://doi.org/10.1093/rfs/hhv059) · [GitHub](https://github.com/OpenSourceAP/CrossSection) · [웹](https://people.duke.edu/~charvey/backtesting/Haircut_SR.m) · [GitHub](https://github.com/chenandrewy/high-throughput-ap) · [GitHub](https://github.com/bkelly-lab/ReplicationCrisis) |
| 백테스트 | **The Virtue of Complexity 와 Nagel 의 반박** | Journal of Finance 79(1), 459–503 (2024) vs NBER WP 34104 (2025) | 일봉 이상 | **월간** CRSP 지수 1926–2020 | 월간 비중 조정 | 논쟁의 초점이 아님 | L2 무영향 재생 | [PDF](https://www.nber.org/papers/w30217) · [DOI](https://doi.org/10.1111/jofi.13298) |
| 백테스트 | **Gort 외** | arXiv 2209.05559 (저널·학회 게재 없음) | 초~분봉 | 암호화폐 10종목 **5분봉** · 2022-02-02~06-27 | 시장가 즉시 (슬리피지 무시) | **없음** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2209.05559) |
| 백테스트 | **A Backtesting Protocol in the Era of Machine Learning (Arnott · Harvey · Markowitz)** | Journal of Financial Data Science (2019) | 해당 없음 | 해당 없음 | 해당 없음 | 해당 없음 | L0 시뮬레이션 없음 | [DOI](https://doi.org/10.3905/jfds.2019.1.064) |
| 백테스트 | **기호적 알파 채굴과 다중검정 보정의 교집합은 비어 있다** | arXiv 2502.16789 · 2606.20625 · 2606.29194 (2025–2026) | 일봉 이상 | 일간 팩터 데이터 | 대개 명시 없음 | 대개 없음 | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2502.16789) · [GitHub](https://github.com/jarrettyu/AlphaMemo) · [GitHub](https://github.com/trevorstephens/gplearn) · [GitHub](https://github.com/astroautomata/PySR) · [GitHub](https://github.com/ICT-FinD-Lab/alphagen) |
| 백테스트 | **mlfinlab 은 오픈소스가 아니다** | 상용 구독 소프트웨어 | 해당 없음 | 해당 없음 | 해당 없음 | **£100/월/사용자** (Business 티어) | L0 시뮬레이션 없음 | [GitHub](https://github.com/hudson-and-thames/mlfinlab) · [GitHub](https://github.com/eslazarev/purged-cross-validation) · [GitHub](https://github.com/skfolio/skfolio) · [GitHub](https://github.com/emoen/Machine-Learning-for-Asset-Managers) |
| 오픈소스 | **hftbacktest** | 오픈소스 (MIT) | 틱·메시지 | L2/L3 이벤트 (거래소·Databento·Tardis) | 큐 위치 모델 4종 + 확률 함수 | maker/taker · 방향별 수수료 · 리베이트 | L4 재생+큐·지연 | [GitHub](https://github.com/nkaz001/hftbacktest) · [문서](https://hftbacktest.readthedocs.io/) |
| 오픈소스 | **NautilusTrader** | 오픈소스 (LGPL-3.0) | 틱·메시지 | L3/L2/L1/trade/bar (ns) | queue_position=True (1.223+) · 체결 모델 11종 | maker/taker · 고정·계약당·계단형 | L4 재생+큐·지연 | [GitHub](https://github.com/nautechsystems/nautilus_trader) · [문서](https://nautilustrader.io/docs) |
| 오픈소스 | **ABIDES / ABIDES-Gym** | arXiv 1904.12066 (ABIDES) · arXiv 2110.14771 (ABIDES-Gym, ICAIF 계열) · 오픈소스 BSD-3 | 틱·메시지 | 합성 에이전트 주문흐름 (L3 호가창) | 진짜 FIFO 가격-시간 우선 | **사실상 없음** (학술 시뮬레이터) | L5 상호작용 시뮬 | [arXiv](https://arxiv.org/abs/1904.12066) · [GitHub](https://github.com/abides-sim/abides) · [GitHub](https://github.com/jpmorganchase/abides-jpmc-public) |
| 오픈소스 | **JAX-LOB / AlphaTrade** | arXiv 2308.13289 · ICAIF 2023 | 틱·메시지 | LOBSTER L3 (주문 ID 있음) | 진짜 FIFO (가격→초→나노초) | **없음** (보상은 TWAP 대비 집행품질) | L4 재생+큐·지연 | [arXiv](https://arxiv.org/abs/2308.13289) · [GitHub](https://github.com/KangOxford/jax-lob) · [GitHub](https://github.com/KangOxford/AlphaTrade) |
| 오픈소스 | **Microsoft Qlib** | 오픈소스 (MIT) | 일봉 이상 | 일봉 / 분봉 (호가창 없음) | `get_deal_price`(종가·VWAP 류) · 거래량 문턱으로 클리핑 | 매수 0.15% · 매도 0.25% · 최소 5원 + 이차 충격비용 | L2 무영향 재생 | [GitHub](https://github.com/microsoft/qlib) · [문서](https://qlib.readthedocs.io/) |
| 오픈소스 | **틱 데이터 공급원 지도** | 1차 자료 조사 | 해당 없음 | L3(LOBSTER·Databento) / L2(Tardis) / L1(Binance·Massive·Dukascopy) | — | — | L0 시뮬레이션 없음 | [GitHub](https://github.com/databento/dbn) · [웹](https://lobsterdata.com) · [GitHub](https://github.com/databento/databento-python) · [GitHub](https://github.com/tardis-dev/tardis-machine) · [GitHub](https://github.com/binance/binance-public-data) · [GitHub](https://github.com/massive-com/client-python) · [GitHub](https://github.com/Leo4815162342/dukascopy-node) |
| 오픈소스 | **QuantConnect Lean** | 오픈소스 (Apache-2.0) | 일봉 이상 | 틱·호가봉·분봉·일봉 | 봉의 저가가 지정가를 뚫으면 **전량** 체결 | 거래소별 수수료 모델 약 35종 + 슬리피지 모델 4종 | L2 무영향 재생 | [GitHub](https://github.com/QuantConnect/Lean) |
| 오픈소스 | **Hummingbot** | 오픈소스 (Apache-2.0) | 초~분봉 | 1분 캔들 | **캔들 종가가 내 가격을 넘으면 전량 체결** | 고정 `trade_cost` × 2 | L2 무영향 재생 | [GitHub](https://github.com/hummingbot/hummingbot) |
| 오픈소스 | **mbt_gym** | ICAIF 2023 · 오픈소스 BSD-3 | 합성·모형 | 합성 (포아송·호크스 도착) | 호가 깊이의 함수인 체결 확률 — `ExponentialFillFunction: exp(−k·depth)` | 없음 | L1 모형 몬테카를로 | [GitHub](https://github.com/JJJerome/mbt_gym) |
| 오픈소스 | **봉 기반 백테스터 계열** | 오픈소스 | 일봉 이상 | 봉 (일·분) | **고가/저가 터치 → 전량 체결** | 고정 수수료 ± 슬리피지 상수 | L2 무영향 재생 | [GitHub](https://github.com/kernc/backtesting.py) · [GitHub](https://github.com/mementum/backtrader) · [GitHub](https://github.com/stefan-jansen/zipline-reloaded) · [GitHub](https://github.com/polakowo/vectorbt) · [GitHub](https://github.com/edtechre/pybroker) · [GitHub](https://github.com/freqtrade/freqtrade) |
| 대한민국 | **한국투자증권 open-trading-api / backtester** | 공식 오픈소스 (GitHub, 2026) | 일봉 이상 | KRX 일봉 (KIS API) | Lean 기본 · 일봉 종가 | 0.015% 수수료 + 0.2% 거래세 · 슬리피지 기본 0 | L2 무영향 재생 | [GitHub](https://github.com/koreainvestment/open-trading-api) · [웹](https://apiportal.koreainvestment.com/) · [GitHub](https://github.com/koreainvestment/kis-ai-extensions) |
| 대한민국 | **키움증권 공식 REST API 저장소** | 공식 오픈소스 (GitHub) | 해당 없음 | KRX 틱봉·분봉·일봉 (REST) | 해당 없음 (백테스터 없음) | 해당 없음 | L0 시뮬레이션 없음 | [GitHub](https://github.com/Kiwoom-Securities/Kiwoom-REST-API) · [웹](https://openapi.kiwoom.com/) |
| 대한민국 | **NH투자증권 NAMUH PLUG (PLUG-OpenAPI)** | 공식 오픈소스 (GitHub, 2026-08~) | 해당 없음 | 차트데이터 (해상도 미명시) | 해당 없음 | API 수수료 국내 0.01% · 해외 0.09% | L0 시뮬레이션 없음 | [GitHub](https://github.com/PLUG-OpenAPI) · [웹](https://www.nhplug.com/intro) |
| 대한민국 | **KRX · 코스콤 · 증권사** | 1차 자료 조사 (공식 포털 · 로컬 저장소 실측) | 해당 없음 | KRX 일별 · 코스콤 유료 · 증권사 당일/틱봉 | — | — | L0 시뮬레이션 없음 | [GitHub](https://github.com/sharebook-kr/pykrx) · [문서](https://datamall.koscom.co.kr/) · [웹](https://openapi.krx.co.kr/) |
| 대한민국 | **An Adaptive Dual-level Reinforcement Learning Approach for Optimal Trade Execution** | arXiv 2307.10649 (q-fin.CP, 2023-07-20) · Expert Systems with Applications 투고 | 초~분봉 | KRX 밀리초 체결+호가 4종목 1년 | 5초마다 즉시 전량 체결 | **수수료·거래소비용 무시** | L2 무영향 재생 | [arXiv](https://arxiv.org/abs/2307.10649) · [PDF](https://arxiv.org/pdf/2307.10649) |
| 대한민국 | **토스증권 Open API (2026-05)** | 공식 REST API (개발자 포털) | 해당 없음 | KRX·미국주식 시세 (해상도 미확인) | 해당 없음 | 2026-06 까지 국내주식 수수료 면제 프로모션 | L0 시뮬레이션 없음 | [웹](https://corp.tossinvest.com/ko/open-api) |
| 대한민국 | **LS증권 · DB증권 · 대신증권 · 삼성선물 · 미래에셋** | 공식 API 포털 (공식 GitHub 없음) | 해당 없음 | 각 사 시세 API | 해당 없음 | — | L0 시뮬레이션 없음 | [문서](https://openapi.dbsec.co.kr/testbed-sample) · [웹](https://openapi.ls-sec.co.kr/) |
| 대한민국 | **국내 커뮤니티 오픈소스** | 커뮤니티 오픈소스 | 일봉 이상 | KRX 일별 · 업비트 체결/호가 | 대개 종가 체결 | 대개 고정 bps 또는 없음 | L2 무영향 재생 | [GitHub](https://github.com/sharebook-kr/pykrx) · [GitHub](https://github.com/FinanceData/FinanceDataReader) · [GitHub](https://github.com/sharebook-kr/pyupbit) |

## 종합 판정

사실근거
① 밀리초 데이터를 썼다는 말은 백테스트 사실성을 전혀 보장하지 않는다
KRX 밀리초 체결·호가 원장을 확보한 최적실행 RL 논문(arXiv 2307.10649)이 그 원장을 5초 버킷으로 다운샘플링하고,
가정 셋을 본문에 적는다 — 일시 충격만 있고 다음 스텝에 회복, “Commissions and exchange fees are ignored”,
“traded immediately without order arrival delays”. 사다리로는 L2 다.

② 큐 위치를 무시하면 마켓메이킹 손익이 대략 스프레드 하나만큼 틀린다 — 그리고 부호까지 바뀐다
Moallemi–Yuan: 대형틱 종목에서 큐 가치가 매수매도 스프레드와 같은 자릿수다.
Cartea–Donnelly–Jaimungal §4.4: 같은 전략이 큐 앞이면 샤프 10~33, 큐 뒤면 ORCL 3.98 · CSCO 0.59 · INTC −1.95.
저자들의 결론 — “the true performance should fall somewhere between that of Table 5 and Table 7.”

③ 학술 백테스트에서 지연과 수수료는 거의 언제나 빠져 있다
실행·마켓메이킹 문헌에서 거래소 수수료를 넣은 것은 Cont–Kukanov(리베이트 0.2/0.25¢ · taker 0.29¢)와
Guilbaud–Pham(€0.0008 리베이트 / €0.0012 수수료) 정도다. 지연을 구조적으로 넣은 것은 ABIDES(쌍별 나노초 지연),
데이터에서 보정해 체결 규칙으로 바꾼 것은 Noble–Rosenbaum–Souilmi(2026) 하나뿐이다.
반면 실무 엔진 hftbacktest·NautilusTrader는 둘 다 지연·수수료를 1급 시민으로 다룬다 —
사실성은 학계가 아니라 오픈소스 실무 쪽이 앞선다.

④ 재생 백테스트는 자기영향을 구조적으로 못 다루고, 그 대가가 측정됐다
Balch 외(2019)는 같은 주문을 재생과 다중 에이전트에서 각각 넣어, 재생에서는 충격이 되돌아오고
상호작용에서는 새 수준에서 안정된다는 것을 쌍 실험으로 보였다.
Noble–Rosenbaum–Souilmi(2026)는 자기 체결을 충격 커널에서 빼고/넣고 두 번 돌려 결론을 냈다 —
“A simulator that excludes the strategy's own trades from the impact kernel will systematically overstate profitability,
precisely in the aggressive parameterization regime where a practitioner most needs an accurate assessment.”

⑤ 예측 정확도와 거래 가능성은 거의 직교한다
Chinco 외(JF 2019): AR(3) 이 LASSO 보다 표본 외 R̄² 가 높은데(7.365% vs 2.467%)
순 샤프는 −0.662 vs +1.791. 스프레드 차감 후 수익이 나는 거래는 약 3분의 1뿐이다.
분류 정확도·F1 로 보고하고 끝내는 딥러닝 LOB 문헌 전체에 대한 반례다.

⑥ 국내 대형 증권사 중 백테스트 코드를 공개한 곳은 한국투자증권 하나이고, 그 백테스트는 일봉이다
koreainvestment/open-trading-api 의 backtester/ 는 QuantConnect Lean 을 도커로 돌리는데,
데이터 제공자가 받는 해상도는 Resolution.DAILY 와 Resolution.MINUTE 둘뿐이고
(분봉은 코드 주석부터 “분봉은 단일 날짜”), Lean 코드 생성기는 Resolution.Daily 를 하드코딩한다.
수수료 0.015% · 매도 거래세 0.2% 는 실제 값을 넣었지만 슬리피지 기본값은 0 이다.
키움은 공식 REST 저장소에 337개 API·362개 예제를 열었지만 백테스터가 없고, 틱차트(ka10079)도 호가 없는 봉이다.
NH(PLUG-OpenAPI)는 저장소 3개가 전부 SDK·MCP 다. 그리고 과거 호가+체결 원장은 국내 공개 API 로 살 수 없다 —
KRX Open API 는 전부 일별, 코스콤 데이터몰은 현재 서비스 중단이다.
⑦ 통계적 보정은 누출 탐지가 아니다 — 그리고 숨은 탐색 앞에서는 거꾸로 움직인다
일부러 미래를 보는 샤프 35짜리 오라클이 Deflated Sharpe 와 PBO 를 완전히 통과한다(Gençay 2026).
그리고 보고하지 않은 탐색이 10 → 1,000 으로 넓어지면 실제 선택 손상은 0.066 → 0.141 로 커지는데
보고되는 PBO 는 0.54 → 0.10 으로 떨어진다 — “No statistic computed on the reported grid recovers what is missing.”
방어 가능한 파이프라인은 구조적 가드레일 · 유효 시행 수 보정 · 최대통계량 부트스트랩 셋을 함께 요구한다.

⑧ 기호적 알파 채굴과 다중검정 보정의 교집합은 비어 있다
AlphaAgent · AlphaMemo · Alpha Singularity 같은 2025~2026 LLM·유전 알파 채굴 논문들은 AST 복잡도 벌점과 독창성 정규화로
과적합을 다루는데 DSR · PBO · haircut 어느 것도 보고하지 않는다. 도구도 같다 —
gplearn · PySR · alphagen 모두 다중검정 통제가 없다.
반대편에서 Chen & Dim 은 채굴 전략 136,000개의 경험적 베이즈 상위 1%가 표본 외 연 5.7%(출판 아노말리 200개는 5.9%)를 벌었음을 보였다 —
탐색을 전부 기록하면 그 크기는 부채가 아니라 추정 자원이 된다.


한 줄 결론. 틱 백테스트의 품질을 가르는 것은 데이터 해상도도, 모델의 정교함도 아니라
“무엇이 체결을 결정하는가” 를 얼마나 정직하게 적었는가다. 가장 좋은 작업들(Cartea–Donnelly–Jaimungal의 큐 밴드,
Huang–Lehalle–Rosenbaum의 “체결확률이 약간 과대평가된다”, Cont–Kukanov의 “앞 큐 취소를 안 넣어 보수적이다”,
hftbacktest의 “no market impact is considered”)은 전부 자기 가정의 방향까지 함께 적었다.

## 전통 금융 · 시장미시구조 실증

탑티어 금융저널(JF·JFE·RFS·QJE·JFQA)이 틱·메시지 데이터로 하는 것은 대개 백테스트가 아니라 회귀다. 약 25편 중 전략을 실제로 굴린 것은 Budish–Cramton–Shim(QJE 2015)과 Aquilina–Budish–O'Neill(QJE 2022) 둘뿐이고, 손익이 나오는 나머지는 시뮬레이션이 아니라 실제 체결의 회계(BHR·Menkveld·Kirilenko)다. 이 군의 백테스트 위생은 ML 문헌과 방향이 반대다 — 표본 외 분할 대신 외생 충격으로 식별하고, 재현 대신 인과를 산다. 그리고 진짜 위험은 체결 모형이 아니라 그 앞의 측정 계층에 있다(Holden–Jacobsen).

### The High-Frequency Trading Arms Race (Budish · Cramton · Shim) — 탑티어 금융저널에 실린 몇 안 되는 진짜 주문 단위 반사실 시뮬레이션

- **venue** Quarterly Journal of Economics 130(4), 1547–1621 (2015) · **해상도** 틱·메시지 · **등급** L3 재생+실호가·비용
- **링크** [PDF](https://ericbudish.org/files/high_frequency_trading_arms_race.pdf) · [DOI](https://doi.org/10.1093/qje/qjv027)
- **데이터** CME Globex **DataMine Market Depth** 직접 피드(ES) + NYSE TAQ **ArcaBook** 직접 피드(SPY). 통합 TAQ 가 아니다. 2005-01-01~2011-12-31, 롤 주간·반일 제외 후 **1,560 거래일**, **종목 2개**. 밀리초(각주 10: 2008-11 이전 CME 피드는 밀리초 필드를 채우지 않아 실제로는 센티초). **주문 ID·참가자 ID 없음** — 저자들이 직접 피드 파서를 짜서 메시지 단위로 호가창을 재구성했다.
- **설계** **반사실 시뮬레이션.** ES–SPY 스프레드가 τ\* ms 이동평균을 넘으면 진입. τ\* 는 직전 월 ES–SPY 수익률 상관이 0.99 에 닿는 시계다. 수익 문턱 0.05 지수 포인트.
- **체결** 싼 쪽은 매도호가에서, 비싼 쪽은 매수호가에서 체결. 수익이 나는 한 여러 레벨을 걸어 올라간다. 수량은 표시 잔량으로 제한되고 "in nearly all instances" SPY 잔량이 구속한다.
- **비용** **양쪽 레그에서 반스프레드를 낸다 — 그것이 비용 모형의 전부다.** 각주 13 이 스스로 밝힌다: 실제로는 양쪽 다 반스프레드를 낼 필요가 없을 수 있어 비용이 과대평가되고, 거래소 수수료·리베이트는 빠져 있어 순효과로는 비용이 과소평가된다.
- **지연** 가정이 아니라 **필터**로 쓴다. 광속 미만은 배제. Spread Networks 광케이블 16→13ms(약 $300M), 마이크로웨이브 10→8.1ms 를 배경으로 인용한다.
- **자기영향** 부분적. 자기 체결을 뺀 그림자 호가창을 병행 유지해 같은 유동성을 두 번 먹지 않게 한다 — 재생 백테스트에서 자기영향을 다루는 최소 장치다.
- **검증** **없다.** 'standard error'·'Newey-West'·'clustered'·'bootstrap'·'confidence interval' 이 통계적 의미로 등장하지 않는다. 결과는 평균과 백분위수다. 유일한 회귀는 R²=0.87 만 보고하고 표준오차가 없다. 표본 분리도 없다.
- **지표** 차익 건수/일 · 건당 이익 · 일간 총액 · 연환산.
- **다루지 않는 것** 큐 위치·지연 왕복·수수료·리베이트가 없다. 통계적 불확실성 보고가 전무하다. 그리고 코드와 데이터가 공개되지 않았다.

```
while Figure I depicts bid-ask midpoints, in computing the arbitrage opportunity we assume that the trader buys the cheaper instrument at its ask while selling the more expensive instrument at its bid… That is, the trader pays bid-ask spread costs in both markets.
```

```
not exploitable under any possible technological advances in speed (other than by a god-like arbitrageur who is not bound by special relativity)
```
- **공개 코드** **코드·데이터 미공개.** QJE 의 데이터 제출 의무가 이 논문보다 늦게 도입됐다
- **한 줄** 탑티어 금융저널의 백테스트는 대개 회귀지만, 이 논문은 실제로 굴렸다 — 그리고 결과를 일부러 과소추정으로 만들었다.

### Quantifying the High-Frequency Trading Arms Race — 진 사람까지 보이는 메시지 데이터, 그리고 이 문헌에서 유일하게 공개된 틱 파이프라인

- **venue** Quarterly Journal of Economics 137(1), 493–564 (2022) · **해상도** 틱·메시지 · **등급** L6 실집행
- **링크** [PDF](https://ericbudish.org/wp-content/uploads/2022/02/Quantifying-the-High-Frequency-Trading-Arms-Race.pdf) · [DOI](https://doi.org/10.1093/qje/qjab032) · [GitHub](https://github.com/ericbudish/HFT-Races) · [데이터](https://doi.org/10.7910/DVN/ZFDWDZ)
- **데이터** FCA 가 **Section 165 요청**으로 확보한 LSE 의 **인바운드 + 아웃바운드 전체 메시지**. 2015-08-17~10-16 중 **43 거래일**(9/7 손상), **FTSE 350 전종목**, 약 15,000 종목-일, **약 22억 메시지**. 거래소 방화벽 밖 광탭의 단일 하드웨어 시계로 **100나노초** 스탬프. **UserID·FirmID·매칭엔진 OrderID 존재.**
- **설계** 재구성 + 시가평가. 경주 = 같은 종목·가격·방향에서 2인 이상의 UserID 가 정보 지평 안에 몰려 일부는 성공하고 일부는 실패한 사건.
- **체결** 실제 기록.
- **비용** 각주 27 이 수수료 배제를 정당화한다 — 거래소의 한계처리비용이 0 이면 무시하는 것이 타당하고, 어차피 LSE 수수료는 고빈도 참가자 공격주문 0.15bp, 패시브 0 이라 평균 경주 이익 대비 작다. LSE 에는 리베이트가 없다.
- **지연** 실측. 매칭엔진 지연 중앙값 약 150μs, 최소 반응시간 29μs.
- **자기영향** 해당 없음(시뮬레이션이 아니므로).
- **검증** **주요 결과에 표준오차가 없다.** 회귀는 외삽 모형뿐이고 **n=43일** 의 단순 OLS 에 Student-t p값이다. 저자들이 명시적으로 "설정 불확실성이 표본 불확실성을 압도한다" 며 신뢰구간 대신 민감도 격자에서 나온 **밴드(0.20~0.74bp)** 를 보고한다. '표본 외' 는 다른 연도·국가로의 외삽이지 홀드아웃이 아니다.
- **지표** 경주 건수 · 지속시간 · 참가자 수 · 경주 내 거래 비중 · 지연 차익거래 세금(bp) · 유동성 비용 감소율.
- **다루지 않는 것** LSE 단일 거래소라 다른 곳에서의 청산을 못 본다. 표준오차가 없다. 원자료는 Section 165 로 받은 것이라 절대 공유할 수 없다.

```
As in BCS, we compute profits as the signed difference between the price in the race and the midpoint in the near future, which has the interpretation of the mark-to-market value for the asset in the race.
```

```
Because our data include firm identifiers, it would seem possible to use the actual trades made by participants to realize their profits rather than using mark-to-market profits… First, we only have data from the LSE, so we do not observe when positions are closed by trades on other venues. Second, firms may not unwind positions after each race, but may instead manage inventory risk on a portfolio basis.
```
- **공개 코드** [ericbudish/HFT-Races](https://github.com/ericbudish/HFT-Races) — Python 3 + R, BSD-3/GPL-3. `ArtificialTestData/`, `Code_and_Data_Appendix.pdf`, **범용 메시지 스키마** 포함 / Harvard Dataverse [doi:10.7910/DVN/ZFDWDZ](https://doi.org/10.7910/DVN/ZFDWDZ) 미러
- **한 줄** 데이터가 '실패한 주문' 을 담으면 시뮬레이션 자체가 필요 없어진다 — 그리고 그때 비로소 코드를 공개할 수 있다.

### High-Frequency Trading and Price Discovery — 손익을 '체결 회계' 로 계산하고, 리베이트가 부호를 뒤집은 논문

- **venue** Review of Financial Studies 27(8), 2267–2306 (2014) · **해상도** 틱·메시지 · **등급** L6 실집행
- **링크** [PDF](https://faculty.haas.berkeley.edu/hender/HFT-PD.pdf) · [DOI](https://doi.org/10.1093/rfs/hhu032)
- **데이터** NDA 로 잠긴 **NASDAQ HFT 데이터셋** — 120종목, 2008~2009 전 기간, 밀리초. **주문 ID·기업 ID 없음**. 체결마다 HH/HN/NH/NN 4상태 플래그만 붙는다. TAQ NBBO(시각 비동기)와 NASDAQ BBO(45일, 동기) 로 보완.
- **설계** 가격의 영구/일시 분해는 23,400개 1초 구간 위에서 종목-일별 칼만 필터 MLE. HFT 주문흐름 충격은 10초 시차 VAR 에서.
- **체결** 해당 없음.
- **비용** maker/taker 를 실제 스케줄로 반영. **유효스프레드·실현스프레드는 논문 어디에도 나오지 않는다.**
- **검증** 종목×일 이중 클러스터 표준오차. 표본 외 없음.
- **지표** 종목-일당 거래수익(달러), 영구/일시 가격성분 분해.
- **다루지 않는 것** 체결 회계이므로 '다른 전략이었다면' 을 물을 수 없다. 종료 재고 마크가 소형주에서 결과를 흔든다.

```
We assume that for each stock and each day in our sample, HFTs and nHFTs start and end the day without inventories. […] The first term captures cash flows throughout the day, and the second term values the terminal inventory at the closing midquote.
```
- **공개 코드** **미공개.** 데이터는 NDA
- **한 줄** 탑티어의 '틱데이터 손익' 은 대개 시뮬레이션이 아니라 체결 회계다 — 그리고 리베이트를 넣느냐가 부호를 정한다.

### Evaporating Liquidity (Nagel) — 비용을 안 빼는 것이 옳은 백테스트, 그리고 그 이유

- **venue** Review of Financial Studies 25(7), 2005–2039 (2012) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [PDF](https://www.nber.org/papers/w17653) · [DOI](https://doi.org/10.1093/rfs/hhs066) · [데이터](https://voices.uchicago.edu/stefannagel/files/2021/06/allretout.csv)
- **데이터** **CRSP 일간**. NYSE/AMEX/Nasdaq, 1998-01~2010-12, 주가 ≥ $1. 출판본과 인터넷 부록 어디에도 'TAQ' 라는 단어가 없다 — **틱데이터를 쓰지 않는다.**
- **설계** Lehmann(1990) 가중치 `w ∝ −(R_{i,t−1} − R_{m,t−1})`, 정확히 $1 롱/$1 숏으로 정규화, 매일 종가 재조정. t−1…t−5 다섯 하위전략의 평균으로 보고. 조건부 베타 헤지로 기계적 시장노출 제거.
- **비용** 차감하지 않는다. 대신 체결가·중간가 두 판본의 차이가 바운스의 크기를 보여 준다.
- **검증** Newey-West 20/3 · Stambaugh 편의 점검 · **2007-06 이전 적합 → 위기 구간 예측**.
- **지표** 일간 평균수익 · 표준편차 · 연환산 샤프.
- **다루지 않는 것** "항상 종가에 체결된다" 는 가정이 그대로 남는다. 큐·체결확률·시장충격은 없다.

```
If calculated based on transaction prices, the reversal strategy returns represent the returns of a hypothetical representative liquidity supplier whose limit orders or quotes always get executed at the closing transaction price… the market-making sector does not pay the bid-ask spread, but instead earns the non-adverse selection component of the bid-ask spread.
```
- **공개 코드** 코드 없음. **일간 수익률 시계열은 공개**: [allretout.csv](https://voices.uchicago.edu/stefannagel/files/2021/06/allretout.csv)
- **한 줄** 비용을 안 뺀 샤프 8.44 가 정당할 수 있다 — 단, 내 전략의 주체가 유동성을 공급하는 쪽일 때만.

### Sparse Signals in the Cross-Section of Returns — Journal of Finance 에 실린 유일한 ML 스타일 분 단위 백테스트, 그리고 "적합도 ≠ 거래가능성"

- **venue** Journal of Finance 74(1), 449–492 (2019) · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [PDF](https://alexchinco.com/sparse-signals-in-cross-section.pdf) · [DOI](https://doi.org/10.1111/jofi.12733)
- **데이터** **NYSE TAQ 1분 수익률, 2005-01~2012-12 전 거래일.** 필터는 주가 > $5 뿐. 후보 예측변수는 NYSE 전 종목(**3·N ≈ 6,000 회귀변수**), 매일 무작위로 다시 뽑은 **250종목** 에 대해 예측. 첫 예측 10:04, 마지막 분(종가 단일가) 제외 → 하루 **356 × 250 = 64,000 예측**.
- **설계** 롤링 30분 창 × 3래그 LASSO(glmnet), 창 안 10-fold CV 로 λ 선택.
- **비용** `sprd_{n,t}` 를 차감하지만 **반스프레드인지 전스프레드인지, 어느 시점 값인지 본문에 정의가 없다**(대조변수 목록에서는 '전일 평균 매수매도 스프레드' 로 나온다). 1분 수익률이 체결가 기준인지 중간가 기준인지도 본문에 없다.
- **검증** **일중 롤링 표본 외**. 종목-일 이중 클러스터 표준오차 + Giacomini–White(2006) 조건부 예측력 검정. **다중검정 보정 없음**(창마다 약 6,000 후보인데 White reality check·Romano–Wolf·FDR 어느 것도 없다). 달력 홀드아웃 구간도 없다.
- **지표** 표본 외 R̄² · 총/순 샤프 · 수익 거래 비율.
- **다루지 않는 것** 스프레드 정의 미명시 · 가격 기준(체결가/중간가) 미명시 · 다중검정 미보정 · 큐/체결확률 없음.
- **공개 코드** **미공개.** 저자 사이트에 PDF 와 DOI 만 있다
- **한 줄** 표본 외 R² 가 더 높은 모형이 순 샤프는 음수 — 예측 지표로 백테스트를 대신하지 말라는 가장 좋은 증거.

### Liquidity Measurement Problems in Fast, Competitive Markets — 모든 틱 백테스트가 조용히 기대고 있는 측정 계층

- **venue** Journal of Finance 69(4), 1747–1785 (2014) · **해상도** 틱·메시지 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://host.kelley.iu.edu/cholden/Holden%20and%20Jacobsen%20(2014).pdf) · [DOI](https://doi.org/10.1111/jofi.12127) · [데이터](https://www.smu.edu/-/media/site/cox/faculty/classmaterials/holden_jacobsen_code.zip)
- **데이터** 100개 기업, 2008-04~06, **체결 33,754,779건**. MTAQ(초 단위, NBBO 파일 없음) vs DTAQ(밀리초 + 공식 SIP NBBO) 대조.
- **검증** 기업×일 이중 클러스터 표준오차 + 100,000회 추첨 errors-in-variables 시뮬레이션.
- **다루지 않는 것** 미국 통합시장(NBBO) 전용이다. 단일 거래소 원장(KRX·ITCH)에는 다른 문제가 있다 — Jurkatis(2022)가 같은 보정을 ITCH 에 쓰면 **오히려 나빠진다**(Interpolated Time 정확도 약 73%)고 보고한다.

```
Regarding research that studies 2008 and years thereafter and that is based on NBBO quotes using MTAQ with no adjustments… any estimates of the quoted spread, effective spread, realized spread, price impact, frequency of trades outside the NBBO, frequency of locked and crossed markets, and buy/sell classification are likely to be strongly biased, whereas estimates of depth and absolute order imbalance are likely to be unbiased.
```
- **공개 코드** [holden_jacobsen_code.zip](https://www.smu.edu/-/media/site/cox/faculty/classmaterials/holden_jacobsen_code.zip) — SAS, DTAQ+MTAQ 정리 코드 / 관련: [jktis/Trade-Classification-Algorithms](https://github.com/jktis/Trade-Classification-Algorithms) (Jurkatis 2022, ITCH 정답 기준 체결 부호 판정)
- **한 줄** 틱 백테스트의 가장 큰 위험은 체결 모형이 아니라 그 앞의 측정 계층에 있다 — 그리고 그 편의는 방향까지 뒤집힌다.

### Zeroing In on the Expected Returns of Anomalies (Chen & Velikov) — 틱으로 비용을 재고 진짜 표본 외로 평가한 유일한 백테스트

- **venue** Journal of Financial and Quantitative Analysis 58(3), 968–1004 (2023) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [PDF](https://www.federalreserve.gov/econres/feds/files/2020039pap.pdf) · [DOI](https://doi.org/10.1017/S0022109022000874) · [GitHub](https://github.com/velikov-mihail/AssayingAnomalies) · [GitHub](https://github.com/velikov-mihail/Chen-Velikov) · [GitHub](https://github.com/chenandrewy/hf-spreads-all)
- **데이터** 유효스프레드 `= 2|log(P_k) − log(M_k)|` 를 **Daily TAQ(밀리·나노초, 2003-10~2016)** 에서 Holden–Jacobsen 의 DTAQ 코드로, **MTAQ·ISSM** 에서 HJ 의 월간 코드로 계산(1999년 이전 2초 지연, 1999~2002 는 1ms). 체결 단위에서 40% 초과 스프레드는 버린다. 1983년 이전은 저빈도 대용치(Gibbs·Corwin–Schultz·Abdi–Ranaldo·Fong–Holden–Trzcinka) 평균으로 역추정하며, 그 평균이 TAQ 와 상관 **90%**, ISSM 과 **94%**.
- **검증** **표본 내 최적화 → 출판 이후 & 2005년 이후 평가.** 분위수 정렬 표본 외 + 경험적 베이즈 축소.
- **지표** 총/순 월평균 수익, 회전율, 지불 스프레드.
- **다루지 않는 것** 월간 지평이라 일중 체결 문제는 다루지 않는다.
- **공개 코드** [velikov-mihail/AssayingAnomalies](https://github.com/velikov-mihail/AssayingAnomalies) — MATLAB. WRDS 에서 CRSP/Compustat 을 받아 아노말리 포트폴리오를 만들고 **세 가지 거래비용 측정치**(Hasbrouck Gibbs · Chen–Velikov 저빈도 결합 · **고빈도 TAQ 스프레드 포함 전체 결합**)로 순수익 백테스트를 돌린다 / [velikov-mihail/Chen-Velikov](https://github.com/velikov-mihail/Chen-Velikov) · [chenandrewy/hf-spreads-all](https://github.com/chenandrewy/hf-spreads-all)(WRDS SAS, 1983~ 유효스프레드 생성) / [openassetpricing.com](https://www.openassetpricing.com) — Chen & Zimmermann 시그널 라이브러리
- **한 줄** "비용을 어떻게 재는가" 와 "어디서 평가하는가" 를 동시에 제대로 한 거의 유일한 사례.

### High frequency trading and the new market makers (Menkveld)

- **venue** Journal of Financial Markets 16(4), 712–740 (2013) · **해상도** 틱·메시지 · **등급** L6 실집행
- **링크** [PDF](https://papers.tinbergen.nl/11076.pdf) · [DOI](https://doi.org/10.1016/j.finmar.2013.06.006)
- **설계** Sofianos 식 회계 항등식 — 이익 = 실현 포지셔닝 손익 − 공격주문에서 낸 스프레드 + 패시브 주문에서 번 스프레드. 거래당 **€0.88 = €1.55 순스프레드 − €0.68 포지셔닝 손실**. 포지셔닝 손실이 모든 종목에서 음수라, "낡은 호가를 낚아챈다" 는 서사와 반대다. 샤프 9.35(자본비용 미부과) → 무위험 + 6% 프리미엄을 물리면 5.88.
- **비용** Euronext 거래당 €0.60 + 0.05bp, Chi-X **maker-taker 공격 0.30bp · 패시브 리베이트 0.20bp**. 순스프레드가 Chi-X €2.52 vs Euronext €1.11 로 갈리는데, 그 €1/거래 차이가 거의 전부 리베이트에서 온다.
- **다루지 않는 것** 이 조사에서 가장 이식 가치가 큰 경고 — 공격주문 총스프레드가 Euronext −€1.26 인데 Chi-X 는 +€3.21 이다. 저자는 이를 "an artefact of the accounting that takes the midquote in the incumbent market as a reference price" 라고 설명한다. **분할시장에서 단일 거래소 중간가를 기준으로 삼으면 비용 귀속의 부호가 뒤집힌다.**
- **한 줄** 리베이트와 기준 중간가 선택이 마켓메이킹 손익의 부호를 정한다 — KRX 는 단일 거래소라 후자 문제는 없지만, NXT 출범 뒤에는 생긴다.

### The Flash Crash (Kirilenko · Kyle · Samadi · Tuzun)

- **venue** Journal of Finance 72(3), 967–998 (2017) · **해상도** 틱·메시지 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://www.repository.cam.ac.uk/handle/1810/270461) · [DOI](https://doi.org/10.1111/jofi.12498)
- **설계** **흔히 잘못 알려진 사실 정정: 타임스탬프는 밀리초가 아니라 초 단위**이고, 같은 초 안의 순서는 시퀀스 ID 로만 구분된다. 양쪽 계정 식별자가 있고 15,000개 넘는 계정이 등장한다. 메시지 데이터가 아니어서 "We do not study message-level data and, thus, do not observe activity for orders that did not execute." 분류는 룩어헤드가 없게 설계했다 — HFT/MM 지위를 5/3~5/5 로 고정하고 5/6 에 그대로 쓴다. 추론은 초 단위 재고–가격 강건 OLS 에 White(1980) 표준오차.
- **다루지 않는 것** 재현 불가능하고 저자들이 그렇게 적었다 — "The source data is confidential… even though we have checked and re-checked our results, they are unlikely to be ever independently validated by other researchers."
- **한 줄** 규제기관 데이터로만 가능한 연구는 그 대가로 재현 가능성을 통째로 포기한다 — 저자들이 직접 그렇게 적었다.

### VPIN 논쟁 (Easley–López de Prado–O'Hara vs Andersen–Bondarenko) — 탑티어 금융저널 유일의 백테스트 방법론 논쟁

- **venue** RFS 25(5) 1457–1493 (2012) vs Journal of Financial Markets 17, 1–46 · 47–52 · 53–64 (2014) · **해상도** 초~분봉 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf) · [DOI](https://doi.org/10.1016/j.finmar.2013.05.005)
- **설계** ELO 의 VPIN 은 1분봉의 거래량을 표준정규 CDF 로 나눠(BVC) 매수/매도로 배분하고, 일평균거래량의 1/50 짜리 버킷 50개 이동창으로 불균형을 잰다. **VPIN 은 Tudor 의 상표이고 저자들이 특허를 출원했다**고 논문 1쪽에 공시돼 있다.
- **비용** 없음.
- **검증** **여기가 논쟁의 핵심이다.** ELO 의 문장 — "We use the entire sample period to determine the cumulative distribution function of the estimated parameters." 즉 'CDF(VPIN)이 붕괴 두 시간 전 0.9 를 넘었다' 는 주장이 **붕괴 이후까지 포함한 분포**로 평가된 것이다. 거래를 하지 않아도 이것은 룩어헤드다.
- **다루지 않는 것** A&B 의 다섯 갈래 반박: ① **기계적 인공물** — 체결 분류를 무작위로 지운 위약 지표 U1/U2-VPIN 이 붕괴 때 똑같이 움직인다. ② 고정 거래량 빈에서는 변동성 연관의 **부호가 뒤집힌다**. ③ **순환성** — BVC 가 |Δ가격|에서 불균형을 만들어 |Δ가격|을 예측한다. 동시점 거래량·변동성을 통제하면 증분 설명력이 없고 VIX·거래량이 압도한다. ④ **의사 실시간 재구성** — 붕괴 이전 데이터만 쓰면 경고가 없고 TR-VPIN 은 14:00 이후에 정점을 찍는다. ⑤ **기저율 보정** — 하루 약 50개 관측이 나오므로 '99번째 백분위수' 는 100일에 한 번이 아니다. **일 최댓값의 CDF** 로 고치면 CDF(VPIN) 99.0 은 일 최댓값의 96.9%, 즉 **연 8회 정도**다. 붕괴 시점 수준은 **2주에 한 번** 꼴이다.
- **한 줄** "전 표본으로 임계값을 정하는 것" 은 거래를 안 해도 룩어헤드이고, "99번째 백분위수" 는 관측 빈도를 밝히지 않으면 아무 뜻이 없다.

### Trading Costs (Frazzini · Israel · Moskowitz) — 1.7조 달러 실집행 기록으로 TAQ 기반 비용 추정을 반박

- **venue** 미출판 워킹페이퍼 (SSRN 3229719, 2018-08) · 관련 SSRN 2294498 · **해상도** 실집행 기록 · **등급** L6 실집행
- **링크** [SSRN](https://papers.ssrn.com/abstract=3229719)
- **설계** 레코드마다 매수/매도, 개시 시점 시장가, 수량, 주당 체결가, 그리고 **거래 유형**(신규 롱/롱 청산/신규 숏/숏 청산)이 있다. 대부분 패시브로 체결돼 "the effective bid-ask spread across all trades averages less than 0.015% per year in the U.S." — 비용의 대부분은 가격충격이다. log-log 회귀 계수는 0.35(R²=95%)지만 **과적합을 피하려고 제곱근(0.5)으로 고정**했다.
- **비용** %DTV 의 제곱근. **TAQ 기반 추정과의 격차가 이 논문의 핵심이다** — DTV 2% 에서 자기 모형 13.73bp vs TAQ 제곱근 27.09 vs TAQ 선형(Breen–Hodrick–Korajczyk) 44.89. DTV 10% 에서는 32.34 vs **223.31** — "almost an order of magnitude larger". 이유: "aggregated daily TAQ data captures the average trader, which includes informed insiders, retail traders, liquidity demanders, and impatient traders, who face much higher costs than those of a patient trader."
- **검증** ITG·도이체방크·JP모간·ANcerno 비용 데이터, 그리고 뱅가드 S&P500 펀드와 iShares 러셀2000 ETF 의 실현 비용에 대한 표본 외 검정.
- **다루지 않는 것** 데이터가 AQR 사내 블로터라 재현 불가. 동료심사를 거치지 않았다.
- **한 줄** "TAQ 로 잰 비용" 은 참을성 없는 평균 트레이더의 비용이지 인내심 있는 집행자의 비용이 아니다 — 그 격차가 10배까지 벌어진다.

## 최적 실행 · 마켓메이킹 · 호가창 시뮬레이션

최적실행·마켓메이킹 문헌의 고전은 백테스트를 하지 않는다. 핵심 7편 중 6편에 백테스트가 없고 5편에는 실데이터가 아예 없다 — 최적해가 그 최적해를 낳은 모형 안에서 검증된다. Avellaneda–Stoikov(QF 2008)는 종목명도 날짜도 없는 순수 몬테카를로다. 실데이터로 넘어와도 오랫동안 “내 호가에 체결이 닿으면 전량 체결” 이 표준이었다. 그 가정을 깨는 작업은 넷이다 — 큐 값을 재고(Moallemi–Yuan), 큐 소진으로 체결을 정하고(Cont–Kukanov), 체결확률 밴드를 제시하고(Cartea–Donnelly–Jaimungal), 재생을 버리고 상호작용 시뮬레이터를 만든다(Huang–Lehalle–Rosenbaum). 2026년 Noble–Rosenbaum–Souilmi가 지연·경주·자기영향을 한꺼번에 다루며 현재의 상한을 정했다.

### High-frequency trading in a limit order book (Avellaneda & Stoikov) — 마켓메이킹의 표준 참조, 그런데 실데이터가 한 줄도 없다

- **venue** Quantitative Finance 8(3), 217–224 (2008) · **해상도** 합성·모형 · **등급** L1 모형 몬테카를로
- **링크** [PDF](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) · [DOI](https://doi.org/10.1080/14697680701381228) · [GitHub](https://github.com/ragoragino/avellaneda-stoikov)
- **데이터** **없다.** 종목명·기간·거래소·데이터 벤더가 논문에 등장하지 않는다. 전부 브라운운동 중간가 + 포아송 체결 모형 위의 시뮬레이션이다.
- **설계** 매 dt 마다 호가를 계산하고, 확률 `λ(δ)dt` 로 체결 여부를 추첨한다. 중간가는 `±σ√dt` 로 갱신.
- **체결** 베르누이 추첨. 큐 위치·FIFO 우선순위·부분체결·취소·역선택 채널이 전부 없다. 주문 크기는 1 고정.
- **비용** 없다. 수수료·리베이트 개념이 등장하지 않는다.
- **지연** 없다.
- **자기영향** 없다(모형 안에서 중간가는 내 호가와 무관하게 움직인다).
- **검증** 없다. 표본 외 개념이 성립하지 않는다.
- **지표** 평균 스프레드 · 수익 · 수익 표준편차 · 최종 재고 · 최종 재고 표준편차.
- **다루지 않는 것** 이 논문이 답하는 질문은 "이 모형 안에서 최적 호가는 무엇인가" 이지 "이 전략이 실제 시장에서 통했는가" 가 아니다. 그런데 실무 문헌은 종종 후자로 인용한다.

```
As far as our simulation is concerned, we chose the following parameters: s = 100, T = 1, σ = 2, dt = 0.005, q = 0, γ = 0.1, k = 1.5 and A = 140. The simulation is obtained through the following procedure: at time [t], the agent's quotes a and b are computed, given the state variables. At time t + dt, the state variables are updated. With probability λᵃ(δᵃ)dt, the inventory variable decreases by one and the wealth increases by s + δᵃ… The mid-price is updated by a random increment ±σ√dt.
```
- **공개 코드** 저자 공개 코드 없음. 제3자 재구현 [ragoragino/avellaneda-stoikov](https://github.com/ragoragino/avellaneda-stoikov) 도 **같은 합성 시뮬레이션**을 돌린다
- **한 줄** 마켓메이킹의 언어를 만든 논문이지만 백테스트는 하나도 없다 — 이론과 검증의 분리를 보여 주는 원형.

### Guéant · Lehalle · Fernandez-Tapia — 실제 틱으로 되감은 첫 마켓메이킹 백테스트, 그리고 그 백테스트의 각주

- **venue** SIAM J. Financial Mathematics 3(1), 740–764 (2012) · Mathematics and Financial Economics 7(4), 477–507 (2013) · **해상도** 틱·메시지 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/1106.3279) · [DOI](https://doi.org/10.1137/110850475)
- **데이터** 2012 논문: **AXA (Euronext Paris), 2010-11-05 의 5분 구간 두 개와 2010-11-08 의 2시간 구간 하나.** 2013 논문: **France Télécom, 2012-03-15, 10:00–16:00 — 한 종목 하루.** 데이터 벤더는 명시 없음.
- **설계** 시간순 재생. 주문 크기는 평균 거래 크기(ATS, 1,105주)로 고정하고 Δt 동안 유지한다.
- **체결** 각주가 전부를 말한다 — "In the backtests we do not deal with quantity and priority issues in the order books and supposed that our orders were always entirely filled." 2013 논문도 같다 — "we assumed that our orders were entirely filled when a trade occurred at or above the ask price quoted by the agent."
- **비용** 없다.
- **지연** 없다.
- **자기영향** 없다.
- **검증** **없다.** 표본 외 구간이 없고, 위험회피 계수 γ 는 "재고가 −10~10 사이에 머무는 임의의 값" 으로 골랐다.
- **지표** **집계 통계가 아예 없다.** 시계열 그림 세 장이 결과의 전부다(2012). 2013 논문도 그림 세 장 + 일중 손익 범위.
- **다루지 않는 것** 표본이 하루~이틀이라 성과의 통계적 의미를 말할 수 없다. 그리고 전량 체결 가정은 마켓메이킹 백테스트에서 가장 큰 낙관 요인이다 — 실제로는 큐 뒤에 서서 못 먹는 경우가 대부분이다.

```
In the backtests we do not deal with quantity and priority issues in the order books and supposed that our orders were always entirely filled.
```

```
we assumed that our orders were entirely filled when a trade occurred at or above the ask price quoted by the agent.
```
- **공개 코드** 공개 코드 없음
- **한 줄** "실데이터로 백테스트했다" 는 문장의 실제 내용이 한 종목 하루·전량 체결·그림 세 장일 수 있음을 보여 주는 사례.

### Enhancing Trading Strategies with Order Book Signals — 체결확률 1 로 재생하고, 그 가정을 큐 확률로 되짚은 논문

- **venue** Applied Mathematical Finance 25(1), 1–35 (2018) · **해상도** 틱·메시지 · **등급** L2 무영향 재생
- **링크** [PDF](https://ora.ox.ac.uk/objects/uuid:006addde-3a03-4d75-89c1-04b59026e1c0) · [DOI](https://doi.org/10.1080/1350486X.2018.1434009)
- **데이터** "all messages sent to Nasdaq's LOB during all trading days in 2014" — **249 거래일 × 30분 구간 13개 × 11종목**(AA, AMAT, ARCC, BXS, CSCO, EBAY, FMER, IMGN, INTC, NTAP, ORCL). 논문은 'ITCH'·'LOBSTER' 라는 단어를 쓰지 않는다. **In-sample 1~6월 / Out-of-sample 7~12월, 표본 외 30분 구간 1,375개.**
- **설계** 역사 재생. 상반기 적합 / 하반기 시험.
- **체결** 기본은 전량 체결. §4.4 에서 (첫 큐 전체를 소진하는 시장가 주문 수)/(전체 시장가 주문 수)로 상태별 체결확률을 추정해 큐 앞/뒤 두 시나리오를 만든다.
- **비용** **없다.** 마켓메이킹 샤프를 보고하면서 maker/taker 수수료를 넣지 않은 것은 중대한 누락이다.
- **지연** 0 으로 가정하고 명시한다.
- **자기영향** 없다. 이유도 적었다 — "an empirical estimation of the magnitude of the feedback effect would be very difficult and beyond the scope of this paper."
- **검증** **시간 분할 표본 외.** in-sample 1~6월, out-of-sample 7~12월, 표본 외 구간 1,375개, 11종목 × 재고벌점 8값.
- **지표** 연환산 샤프.
- **다루지 않는 것** 수수료가 없다. 큐 위치는 밴드로만 다루고 실제 큐를 세지 않는다. 지연 0.

```
we do not include a feedback effect of the agent's orders into the dynamics of the LOB… an empirical estimation of the magnitude of the feedback effect would be very difficult and beyond the scope of this paper
```

```
we have assumed that the fill probability of the agent's orders is equal to one
```

```
the true performance should fall somewhere between that of Table 5 and Table 7
```
- **공개 코드** 공개 코드 없음
- **한 줄** 체결확률 1 로 낸 샤프 10~33 이 큐 뒤에서는 −1.95 까지 떨어진다 — 큐 가정이 결론을 뒤집는다는 것을 저자 스스로 보인 논문.

### The Queue-Reactive Model — 재생을 버리고 "내 주문이 큐를 바꾸는" 시뮬레이터를 만든 논문

- **venue** Journal of the American Statistical Association 110(509), 107–122 (2015) · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [arXiv](https://arxiv.org/abs/1312.0563) · [DOI](https://doi.org/10.1080/01621459.2014.982278)
- **데이터** "collected from Cheuvreux's LOB database, from January 2010 to March 2012, on Euronext Paris" — 대형틱 2종목(France Telecom 일 159,250주문 · Alcatel-Lucent). 2년 3개월.
- **설계** **보정된 상호작용 시뮬레이터.** 큐 길이별 도착·취소 강도를 추정해 시장을 다시 생성한다. 시뮬레이션 횟수에 제한이 없다.
- **체결** FIFO. 취소는 내 주문을 뺀 나머지에 균등 배분하고 강도를 재조정한다. 저자들이 이 근사가 체결확률을 약간 과대평가한다고 밝힌다.
- **비용** 논문의 초점이 아니다.
- **지연** 없다.
- **자기영향** **있다.** 이 논문의 핵심 기여다.
- **검증** 보정 구간과 시뮬레이션의 정형 사실 대조. 전략 표본 외 분할은 아니다.
- **지표** 집행 비용, 충격 곡선, 기계적 변동성 대 실증 변동성.
- **다루지 않는 것** 대형틱 2종목 · Euronext 에 특화돼 있다. 지연이 없고, 내생 동역학만으로는 실증 변동성을 재현하지 못한다(5 vs 14 bps).

```
one often needs to rely on so-called market replayers, in which the number of simulations is limited by that of the available trading days… Moreover, the market impact… is often neglected. In contrast, our framework is unlimited in number of simulations.
```

```
Orders with lower priority are actually more likely to be canceled… As a result, execution probabilities might be slightly overestimated.
```
- **공개 코드** 저자 공개 코드 없음. 후속 [SaadSouilmi/Queue-Reactive](https://github.com/SaadSouilmi/Queue-Reactive) (MIT) 가 C++20+Python 구현을 공개
- **한 줄** "재생 백테스트로는 답할 수 없는 두 질문"을 명문화하고 대안을 실제로 만든 논문 — 사다리 L4의 원형.

### Optimal Order Placement in Limit Order Markets (Cont & Kukanov) — 큐를 세고 리베이트를 넣고 시간으로 분할한, 실행 문헌의 드문 정석

- **venue** Quantitative Finance 17(1), 21–39 (2017) · **해상도** 틱·메시지 · **등급** L3 재생+실호가·비용
- **링크** [arXiv](https://arxiv.org/abs/1210.1625) · [DOI](https://doi.org/10.1080/14697688.2016.1190030)
- **데이터** TAQ. **MSFT, NASDAQ 과 BATS Z 두 거래소. 보정 2012-01~03, 평가 2012-04** — 진짜 시간 분할이다.
- **지연** 없다.
- **자기영향** 없다. 순수 재생이다.
- **검증** **시간 분할** (1~3월 보정 → 4월 평가).
- **지표** 중간가 대비 주당 비용(센트).
- **다루지 않는 것** **최우선호가가 변하지 않은 1분 구간만 쓰는 것이 심각한 선택 제약이다** — 가격이 움직이는 순간이 바로 역선택이 일어나는 순간인데 그것을 뺐다. 종목도 MSFT 하나.

```
our limit order fill estimates are conservative as they do not include possible order cancelations from the front of a bid queue… To simplify our simulation we restricted it to one-minute samples where the best quotes did not change up or down.
```
- **공개 코드** 공개 코드 없음
- **한 줄** 큐·부분체결·리베이트·시간 분할을 다 갖춘 드문 백테스트 — 다만 가격이 움직이는 구간을 빼서 역선택을 통째로 피했다.

### A Model for Queue Position Valuation in a Limit Order Book (Moallemi & Yuan) — 큐 위치의 값을 실제로 재고, 그것이 스프레드만큼 크다는 것을 보인 연구

- **venue** 워킹페이퍼 (2016-12, 2017-06 개정) · SSRN 2996221 · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [PDF](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) · [SSRN](https://papers.ssrn.com/abstract=2996221)
- **데이터** **NASDAQ ITCH market-by-order**(주문 ID 있음). 대형틱 9종목 BAC·CSCO·GE·F·INTC·PFE·PBR·EEM·EFA, 2013년 8월(21~22 거래일), 값은 30 거래일 평균.
- **지연** 없다.
- **자기영향** 무한소 가정으로 명시적 배제.
- **검증** 모형 예측 vs 시뮬레이션 교차 확인. 표본 외 시간 분할은 아니다.
- **지표** 주문 가치(틱) · 체결확률 · touch 가치.
- **다루지 않는 것** 대형틱 9종목 · 한 달. 지연 없음. 펀더멘털 대용치가 1분 뒤 중간가라는 선택이 결과에 영향을 준다.

```
the order value cannot be measured by the profitability of the actual historical orders in the limit order book, since actual orders may have private information. In order to bypass this difficulty, instead of actual orders, we will simulate the outcome of randomly placed artificial orders… we will assume that all of the artificial orders are of infinitesimal size and hence have no market impact.
```
- **공개 코드** 공개 코드 없음
- **한 줄** 큐 위치의 값이 스프레드만 하다 — 큐를 안 세는 백테스트는 마켓메이킹 손익을 스프레드 하나만큼 틀린다는 뜻이다.

### Bridging the Reality Gap in Limit Order Book Simulation (2026) — 지연·경주·자기영향을 한 번에 다룬, 이 조사에서 가장 완전한 백테스트 사실성 연구

- **venue** arXiv 2603.24137 (2026-03-25) · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [arXiv](https://arxiv.org/abs/2603.24137) · [GitHub](https://github.com/SaadSouilmi/Queue-Reactive)
- **데이터** **Databento MBP-10**, 대형틱 S&P500 4종목(INTC · VZ · T · PFE), **2023-12 ~ 2025-12**, 10:00~15:30.
- **지연** **실측.** 이벤트 간격 분포의 최빈값에서 왕복 지연 δ 를 읽어 낸다.
- **자기영향** **있다. 그리고 켜고 끄고 비교한다** — 이 조사에서 자기영향을 절제실험으로 정량화한 유일한 작업.
- **검증** 충격 곡선·회복이 실증과 맞는지, 그리고 신호 지평별 예측력 곡선.
- **지표** 전략 손익, 신호 예측력, 충격 곡선.
- **다루지 않는 것** 대형틱 4종목. 경주 승패의 정확한 모형화는 공개 피드로 불가능하다고 저자들이 밝힌다.

```
fill probabilities should not always [be] supposed to be one. When the signal is strong enough to trigger the strategy, it is likely strong enough to trigger competitors as well. Fill probability should decrease with signal strength.
```

```
A simulator that excludes the strategy's own trades from the impact kernel will systematically overstate profitability, precisely in the aggressive parameterization regime where a practitioner most needs an accurate assessment.
```
- **공개 코드** [SaadSouilmi/Queue-Reactive](https://github.com/SaadSouilmi/Queue-Reactive) — MIT, C++20 + Python, Databento 파이프라인 포함
- **한 줄** "신호가 셀수록 체결확률이 낮아야 한다" — 재생 백테스트가 가장 크게 틀리는 지점을 한 문장으로 정리한 논문.

### How to Evaluate Trading Strategies: Single Agent Market Replay or Multiple Agent Interactive Simulation? — 두 방식을 같은 조건에서 맞붙인 실험

- **venue** arXiv 1906.12010 (2019-06-28) · **해상도** 틱·메시지 · **등급** L5 상호작용 시뮬
- **링크** [arXiv](https://arxiv.org/abs/1906.12010)
- **데이터** LOBSTER 메시지 파일 하루, 09:30~10:30 — **86,615 이벤트, 41,844 고유 주문 ID, 매도 신규지정가 22,222 · 매수 19,333**. 신규 지정가 41,554건의 평균 도착 간격 866ms, 취소 38,791건은 928ms. **종목명과 날짜가 논문에 없다.**
- **자기영향** 재생 = 없음 / IABS = 있음. 그 차이가 결과다.
- **검증** 쌍 실험 + 에이전트별 PRNG 로 "다른 에이전트의 행동 변화가 무작위 섭동이 아니라 바뀐 시장 환경에서 온 것" 임을 보장한다.
- **지표** 재생 기준선으로 정규화한 1ms 간격 중간가 궤적.
- **다루지 않는 것** IABS 의 배경 에이전트가 합성이라 '어느 쪽이 진짜 시장에 가까운가' 는 여전히 미해결이다. 종목·날짜 미공개.
- **공개 코드** ABIDES 기반 — [abides-sim/abides](https://github.com/abides-sim/abides) · [jpmorganchase/abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public)
- **한 줄** 재생에서는 충격이 사라지고 상호작용에서는 남는다 — 백테스트 방식이 결론의 성질을 바꾼다는 것을 통제 실험으로 보인 논문.

### Optimal Execution of Portfolio Transactions (Almgren & Chriss)

- **venue** Journal of Risk 3(2), 5–39 (2000/2001) · **해상도** 합성·모형 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) · [DOI](https://doi.org/10.21314/JOR.2001.041)
- **설계** 영구충격 `g(v)=γv`, 일시충격 `h(n/τ)=ε·sgn(n)+(η/τ)n`. 보정은 명시적으로 어림값이다 — "a common rule of thumb is that price effects become significant when we sell 10% of the daily volume" 로 γ = 2.5×10⁻⁷. 지표는 평균–분산 효율경계, 이차효용, L-VaR, 거래 반감기.
- **다루지 않는 것** 체결·큐·지연·부분체결·역선택이 전부 없다. 실행 문헌 전체의 기준선인데 실증 검증은 이 논문에 없다.
- **한 줄** 실행 문헌의 출발점이자, 이 문헌에서 '보정' 이 종종 '지정' 을 뜻한다는 것을 보여 주는 원형.

### Direct Estimation of Equity Market Impact (Almgren · Thum · Hauptmann · Li)

- **venue** Risk (2005-07) · **해상도** 실집행 기록 · **등급** L6 실집행
- **링크** [PDF](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)
- **설계** S&P500 한정, VWAP 주문 제외(약 16%), 16:10 이후 제외(약 10%), 체결 2건 이상·1,000주 이상·ADV 0.25% 이상. 체결 횟수 중앙값 약 5회, 지속 중앙값 약 30분.
- **검증** 적합 지수 α = 0.891 ± 0.10, δ = 0.267 ± 0.22, **β = 0.600 ± 0.038 — 제곱근 법칙 β = 1/2 가 95% 에서 기각된다.** γ = 0.314 ± 0.041(t=7.7), η = 0.142 ± 0.0062(t=23). 그리고 결정적 문장: "The R² values for these regressions are typically less than one percent."
- **다루지 않는 것** 표본 외 백테스트를 약속했으나 — "We hope to provide out-of-sample backtests in a future paper." — **그런 논문은 Almgren 의 발표 목록에 없다.** 데이터는 사내 전용.
- **한 줄** 시장충격 계수를 실집행으로 추정한 표준 참조 — 그런데 R² 가 1% 미만이고, 약속한 표본 외 백테스트는 20년째 나오지 않았다.

### Optimal High-Frequency Trading with Limit and Market Orders (Guilbaud & Pham)

- **venue** Quantitative Finance 13(1), 79–94 (2013) · **해상도** 합성·모형 · **등급** L1 모형 몬테카를로
- **링크** [arXiv](https://arxiv.org/abs/1106.5040) · [DOI](https://doi.org/10.1080/14697688.2012.708779)
- **설계** 마켓메이킹 문헌에서 가장 미시구조적으로 정직한 체결 대용치다. 최우선가를 개선하면 "top priority in the LOB and therefore captures all incoming market order flow", 최우선 큐에 합류하면 매수측 누적 체결량이 V₀ + V^b 를 넘을 때만 체결 카운터가 오른다. 지연은 0.
- **검증** 정보비율 최적 2.117 vs 시장가 미사용 1.999 vs 고정호가 0.472 vs 무작위 0.376. 잉여이익 거래당 €0.056 — "roughly twice the clearing fees".
- **다루지 않는 것** 저자들 스스로 적었다 — "this simulated data backtest must be completed by a backtest on real data."
- **한 줄** 수수료와 FIFO 우선순위를 넣은 드문 마켓메이킹 연구 — 다만 성과는 합성 경로에서 나왔고 저자들도 그렇게 밝힌다.

### Anomalous Price Impact and the Critical Nature of Liquidity (Tóth 외) — 제곱근 법칙의 표준 인용, 그리고 오차막대 없음

- **venue** Physical Review X 1, 021006 (2011) · **해상도** 실집행 기록 · **등급** L6 실집행
- **링크** [arXiv](https://arxiv.org/abs/1105.1694) · [DOI](https://doi.org/10.1103/PhysRevX.1.021006)
- **설계** 그림 1 캡션이 데이터 명세의 전부다 — "The impact of metaorders for CFM proprietary trades on futures markets, in the period June 2007 – December 2010… The data base contains nearly 500,000 trades… For large ticks, the curve can be well fit with δ = 0.6, while for small ticks we find δ = 0.5… We have removed a small positive intercept Δ/σ = 0.0015 for Q = 0."
- **검증** **오차막대도, 신뢰구간도, 적합도 통계도 논문 어디에도 없다.** 계약 목록도 "a variety of futures contracts" 가 전부다.
- **다루지 않는 것** 재현 불가. 그리고 가장 엄밀한 후속 연구(Zarinelli 외 2015, ANcerno 694만 메타주문)에서는 **로그 함수가 잔차 구조 기준으로 제곱근을 분명히 이긴다**(가중 RMSE 2.80 vs 6.70).
- **한 줄** "시장충격은 제곱근" 의 표준 인용인데, 정작 그 논문에는 오차막대가 없고 후속 연구는 로그를 지지한다.

### Market Microstructure Invariance: Empirical Hypotheses (Kyle & Obizhaeva)

- **venue** Econometrica 84(4), 1345–1404 (2016) · **해상도** 실집행 기록 · **등급** L6 실집행
- **링크** [PDF](https://pages.nes.ru/aobizhaeva/Kyle-Obizhaeva-ECTA-2016-Invariance-with-Supplement.pdf) · [DOI](https://doi.org/10.3982/ECTA10486)
- **설계** 이 조사에서 **선택편의 문제를 가장 명료하게 진술한 문장**이 여기 있다 — "When orders are canceled after prices move in an unfavorable direction… implementation shortfall may dramatically underestimate actual transaction costs. Portfolio transitions data are not subject to these concerns." 전환 주문은 반드시 끝까지 집행되므로 취소 선택편의가 없다.
- **검증** 주 단위 클러스터 표준오차. 선형 R² = 0.0991, 제곱근 0.1007, 무제약 0.1016. 기준 비용 10.71bp(선형) vs 14.16bp(제곱근). 주문크기 지수 α̂₀ = −0.62(SE 0.009) vs 예측 −2/3. **통계적으로는 불변성이 기각되지만 경제적 크기는 지지된다.** 저자들의 겸손 — "the R² is equal to 0.0847 in the transaction-cost regressions with market return only… The transaction-cost models improve the R²'s by only one or two percent."
- **다루지 않는 것** 재현 불가(NDA).
- **한 줄** "주문을 중간에 취소할 수 있으면 구현부족이 비용을 극적으로 과소평가한다" — 백테스트에서 미체결을 버리면 안 되는 이유의 정식 진술.

## AI · 딥러닝 · 강화학습 트레이딩

탑티어 AI 학회의 트레이딩 논문에서 가장 흔한 일은 체결을 아예 시뮬레이션하지 않는 것이다. 호가창 딥러닝 계보(DeepLOB → FI-2010 → 수십 편)는 중간가 방향의 F1 을 보고하고 끝내는데, 재현 연구가 시드 간 F1 표준편차가 10~17 포인트(보고되는 개선폭은 1~3 포인트)이고 다른 데이터로 옮기면 성능이 붕괴함을 보였다. 그리고 Briola 외는 거래가능성 지표 p_T 가 0.01~0.15, 신뢰도로 거르면 정확히 0 인데 같은 구간에서 MCC 는 계속 오른다는 것을 보였다. 체결을 흉내 내는 논문들도 대개 봉 종가·분봉 평균가·중간가에 즉시 전량 체결이다. 예외는 셋뿐이다 — Nevmyvaka 외(ICML 2006)가 호가창 우선순위를 유지하고 가정을 학습에서만 쓰고 테스트에서 푼다, Schnaubelt(EJOR 2022)가 큐 앞 물량과 거래소별 maker/taker 를 넣는다, 그리고 MarS·LOB-Bench가 자기영향을 생성모델로 만들고 그것을 채점한다. 가장 널리 회자되는 LLM 에이전트 논문들은 전부 일봉·비용 0·단일 창이고, star 수와 평가 엄밀성이 반비례한다.

### DeepLOB · FI-2010 · 그리고 그 위에 쌓인 10년 — 재현 연구가 밝혀낸 딥러닝 호가창 문헌의 구조적 결함

- **venue** IEEE TSP 2019 (DeepLOB) · Artificial Intelligence Review 2024 (Prata) · Quantitative Finance 2025 (Briola) · **해상도** 틱·메시지 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) · [데이터](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) · [GitHub](https://github.com/matteoprata/LOBCAST) · [GitHub](https://github.com/FinancialComputingUCL/LOBFrame) · [GitHub](https://github.com/LeonardoBerti00/TLOB) · [GitHub](https://github.com/lorenzolucchese/deepOBs)
- **데이터** FI-2010: 핀란드 소형 5종목 **10일**, 10호가, 10 이벤트마다 표본, z-score 정규화, 6-2-2 일 분할, 전역 문턱 θ = 0.002. 후속 연구들은 LOBSTER(NASDAQ)와 자체 수집 데이터로 옮겨 갔다.
- **설계** 지도학습 분류. 향후 k 이벤트 뒤 중간가 이동을 상승/보합/하락 3분류로 라벨링한다.
- **체결** 없다. 있는 경우에도 중간가 또는 다음 시가.
- **비용** 없다. Prata 외가 명시한다 — "we make the assumption of no transaction fees".
- **지연** 없다.
- **자기영향** 없다.
- **검증** **대부분 단일 분할 · 단일 시드.** 라벨 창이 k 이벤트로 겹치는데 **purged CV 나 embargo 를 쓴 논문이 이 조사 전체에 하나도 없다.** 예외적으로 Prata(시드 5 + 표준편차), Lucchese 외(롤링 창 11개 + Model Confidence Set), Zhang & Zohren(JFDS 2020, 5년마다 재학습).
- **지표** 정확도·F1·MCC. Briola 의 p_T 와 Lucchese 의 MCS 만이 예외다.
- **다루지 않는 것** 체결·큐·비용·지연이 전부 없고, 정규화 누출을 감사할 수 없으며, 시드 분산이 보고된 개선폭보다 크다. 그리고 대표 저장소들(DeepLOB·LOBCAST·LOBFrame·TLOB)에 **라이선스 파일이 없다.**

```
we use mid-prices without transaction costs… compare gross profits before fees   — DeepLOB
```

```
we make the assumption of no transaction fees   — Prata 외 (LOBCAST)
```

```
already pre-processed (filtered, normalized, and labelled) so that the original LOB cannot be backtracked   — FI-2010
```
- **공개 코드** [zcakhaa/DeepLOB…](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) — star 606, **라이선스 없음**. FI-2010 데모만 / [matteoprata/LOBCAST](https://github.com/matteoprata/LOBCAST) — Prata 외 재현 벤치마크. star 123, **라이선스 없음** (`v0-LOBCAST` 브랜치 사용) / [FinancialComputingUCL/LOBFrame](https://github.com/FinancialComputingUCL/LOBFrame) — Briola 외. star 258, **LICENSE 404** / [lorenzolucchese/deepOBs](https://github.com/lorenzolucchese/deepOBs) — 롤링 창 + Model Confidence Set. star 67, BSD-3 / [FI-2010 공식 배포](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) — CC BY 4.0
- **한 줄** 10일짜리 데이터셋 위에서 10년간 F1 순위를 다퉜고, 재현해 보니 시드 분산이 개선폭보다 컸으며, 거래가능성은 0이었다.

### Reinforcement Learning for Optimized Trade Execution (ICML 2006) — 2006년 논문이 2024년 논문 대부분보다 잘 설계돼 있다

- **venue** ICML 2006, pp. 673–680 · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) · [DOI](https://doi.org/10.1145/1143844.1143929)
- **데이터** **INET ECN 원장, 밀리초 미시구조 데이터 1년 6개월**, 3종목(AMZN·QCOM·NVDA) — "to examine how the performance and policies found by RL vary with stock properties such as liquidity, volume traded, and volatility". 종목당 수 GB. **학습 12개월 / 테스트 6개월로 분리.** H=2분일 때 학습 데이터에서 약 45,000 에피소드.
- **설계** 3종목 × 주문량 V ∈ {5,000, 10,000} × 지평 H ∈ {2, 8}분. 지평을 T개 결정 시점으로 나누고(H=2분·T=4면 30초마다 갱신), 재고를 I 단계로 이산화. 행동은 현재 매도호가(매도 시 매수호가) 대비 상대 지정가.
- **체결** 실제 호가창에서 우선순위를 유지하며 체결. 종료 시 남은 수량은 **반대편 호가를 아무리 나쁜 가격이라도 먹어 치워** 강제 청산.
- **비용** **없다. 그리고 이유를 밝힌다** — "As our study is primarily aimed at large institutional investors, we assume that commissions and exchange fees are negligible. We also assume direct and fast access to exchanges — i.e. we do not account for possible order arrival delays."
- **지연** 없다(위 문장에서 명시적 배제).
- **자기영향** **학습에서는 없고, 테스트에서는 있다.** 다만 배경 주문 흐름이 반응하지는 않는다.
- **검증** 12개월 학습 / 6개월 테스트, 종목·V·H·상태표현의 모든 조합을 시험. 시드 반복은 없다(뒤에서 앞으로 Q값을 평균하는 결정론적 방식).
- **지표** 에피소드 시작 시점의 **중간가 대비 basis point 거래비용**.
- **다루지 않는 것** 수수료 0, 지연 0, 종목 3개. 배경 흐름이 반응하지 않는다.

```
It executes orders and maintains priorities in order books in the way we have just described. Such a setup allows us to run simulations in historical order books, and capture all costs and uncertainties of trade execution: the bid-ask spread, market impact, and the risk of non-execution.
```

```
all of our results are reported on test data in which this assumption was not made --- all test set order book simulations maintain the impacts of any policy actions.
```
- **공개 코드** 공개 코드 없음(2006)
- **한 줄** 18년이 지난 지금도 대부분의 논문이 이 기준선을 넘지 못한다 — 특히 "가정을 학습에서만 쓰고 테스트에서 푼다" 는 부분.

### Universal Trading for Order Execution with Oracle Policy Distillation (AAAI 2021) — 시드 6개와 유의성 검정은 하는데, 체결은 분봉 평균가

- **venue** AAAI 2021, 35(1), 107–115 · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2103.10860) · [DOI](https://doi.org/10.1609/aaai.v35i1.16083) · [GitHub](https://github.com/microsoft/qlib) · [웹](https://seqml.github.io/opd/)
- **데이터** "The dataset contains (i) **minute-level price-volume market information** and (ii) the order amount of every trading day for each instrument from **Jan. 1, 2017 to June 30, 2019**." 학습 2017-01-01~2019-02-28 / 검증 2019-03-01~"2019-04-31"(원문 오탈자, 4월은 30일까지) / 테스트 2019-05-01~06-30. 검증·테스트는 계산자원 제약으로 CSI 800 만 남긴다.
- **설계** 하루 지평. 행동은 목표 물량 Q 의 비율(이산). 목적은 `arg max Σ (q_{t+1}·p_{t+1})` s.t. `Σ q_{t+1} = Q`.
- **체결** 각주 2 — "We define 'market price' as the averaged transaction price of the whole market at one time which has been widely used in literature." 즉 분봉 평균가를 받는다. 스프레드 지불도, 호가 소비도 없다.
- **비용** **본문이 명시한다** — "(i) the temporary market impact has been adopted as a reward penalty and we assume that the market is resilient and will bounce back to the equilibrium at the next timestep. **(ii) We either ignore the commissions and exchange fees** as these expense is relatively small fractions for the institutional investors that we are mainly aimed at."
- **지연** 없다.
- **자기영향** 일시 충격을 보상 벌점으로만 반영하고 다음 스텝에 회복한다고 본다.
- **검증** **시드 6개 + 유의성 검정.** 다만 시간 분할은 단일이다(워크포워드 아님). 검증 구간 표기에 오탈자가 있다.
- **지표** PA(bp) = 10⁴·평균(전략 평균체결가 / 시장 평균가 − 1) · GLR · 보상.
- **다루지 않는 것** 분봉 평균가 체결은 낙관적이다 — 실제로는 그 평균가를 받으려면 그 분 내내 균등 집행해야 하고 스프레드를 낸다. 비용 없음, 단일 분할.

```
(i) the temporary market impact has been adopted as a reward penalty and we assume that the market is resilient and will bounce back to the equilibrium at the next timestep. (ii) We either ignore the commissions and exchange fees as these expense is relatively small fractions for the institutional investors that we are mainly aimed at.
```
- **공개 코드** 프로젝트 페이지 [seqml.github.io/opd](https://seqml.github.io/opd/) · 파이프라인은 [microsoft/qlib `examples/rl_order_execution`](https://github.com/microsoft/qlib/tree/main/examples/rl_order_execution)
- **한 줄** 시드·유의성·룩어헤드 방지는 모범적인데 체결 가정이 분봉 평균가라 성과 숫자의 층이 다르다.

### Double Deep Q-Learning for Optimal Execution (Ning · Lin · Jaimungal) — 호가창을 가지고 있으면서 중간가로 체결한다

- **venue** Applied Mathematical Finance 28(4), 361–380 (2021/2022) · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/1812.06600) · [DOI](https://doi.org/10.1080/1350486X.2022.2077783)
- **데이터** 9종목(AAPL·AMZN·FB·GOOG·INTC·MSFT·NTAP·SMH·VOD), **2017-01-02 ~ 2018-03-30 전 거래일**. "We use the full limit order book information to extract the **midprice at the end of each second**." 일중 변동이 큰 시간대를 피해 11–12시·12–13시·13–14시를 따로 분석. **데이터 벤더는 논문에 없다.**
- **지연** 없다.
- **자기영향** 없다 — "we assume the trader's actions do not directly effect the price process during training".
- **검증** **본문에서 'training set'·'test set'·'hold-out' 을 찾을 수 없다. 학습/테스트 분할이 문서화돼 있지 않다.** 하이퍼파라미터는 "cannot be tuned with cross-validation" 이라고 적혀 있다. 시드 보고 없음.
- **지표** `ΔP&L = (모델 − TWAP)/TWAP × 10⁴` 를 시간대별로 평균·중앙값·표준편차·GLR·P(ΔP&L>0) 로.
- **다루지 않는 것** 중간가 재생 + 표본 내 적합 벌점의 조합은 집행 문제를 사실상 없애는 것에 가깝다. 실제 반스프레드를 넣으면 AAPL·FB 의 2~5bp 는 남지 않을 가능성이 높다. 시간대도 손으로 골랐고, 표본 외 분할이 문서에 없다.

```
The share prices in our data are 'small' relatively to the tick size, and hence the spread is typically 1 tick – such stocks are called large tick stocks. Therefore, we approximate all execution prices by the mid-price and ignore the spread. We avoid the cost associated with walking the LOB by applying a penalty on the size of any one order
```
- **공개 코드** 공개 코드 없음
- **한 줄** "대형틱이니 스프레드를 무시해도 된다" 는 가정이 결과 대부분을 만든다 — 그리고 그 가정이 깨지는 두 종목에서 성과가 사라진다.

### Deep RL for the optimal placement of cryptocurrency limit orders (Schnaubelt) — 큐와 실제 수수료를 함께 넣은 유일한 집행 논문, 그리고 그 결과가 수수료 차익이었다는 고백

- **venue** European Journal of Operational Research 296(3), 993–1006 (2022) · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [PDF](https://www.econstor.eu/bitstream/10419/216206/1/1696077540.pdf) · [DOI](https://doi.org/10.1016/j.ejor.2021.04.050)
- **데이터** **18개월, 2018-01-01 ~ 2019-06-30, 원자료 약 264GB.** 거래소·시장쌍: BitFinex BTC/USD·ETH/USD·ETH/BTC, Kraken BTC/USD, Coinbase BTC/USD — 세 BTC/USD 거래소가 해당 기간 BTC/USD 지정가 거래대금의 **약 65%**. 체결 8,797만(BitFinex BTC/USD)·6,483만(Kraken)·7,431만(Coinbase)·3,277만(ETH/USD) 등 **총 3억 건 이상 체결과 350만 개 이상 호가창 상태.** 다만 호가창은 **API 폴링으로 1분마다 재구성**된 것이고 메시지 단위가 아니다.
- **설계** T = 4 스텝(각 1분) MDP. 이산 행동은 지정가 가격. 마지막 스텝은 강제 시장가(p_{T−1} = −∞). 알고리즘은 표 기반 Q학습·DDQN·PPO(stable-baselines).
- **체결** 1단계 즉시 매칭(호가를 걸어 올라감) → 2단계 잔여분을 가상 호가창에 넣고 실제 체결 스트림으로 채움. 같은 가격대의 더 좋은 조건 대기 주문을 **내 주문 앞 물량으로 더한다.**
- **비용** 거래소별 실제 maker/taker 요율. **이것이 결과를 지배한다.**
- **지연** **0으로 가정하고 명시한다.** 히든/아이스버그 유동성도 없다(BitFinex 만 지원).
- **자기영향** 없다. 영구 충격도, 타 참가자 반응도 없다. 대신 v₀ 를 분당 평균 거래량 수준으로 **의도적으로 작게** 잡아 그 가정을 지킨다.
- **검증** **롤링 전진검증 4창.** 이 조사의 집행 문헌에서 최상급.
- **지표** 총 구현부족(bp)을 **원(raw) 구현부족과 실현 거래소 수수료로 분해**. 기준선은 submit-and-leave · 즉시 시장가 · 시간가중.
- **다루지 않는 것** **핵심 결과가 미시구조 타이밍이 아니라 maker/taker 수수료 차익이다.** 10 BTC 에서 집행비용의 약 10%만이 원 구현부족이고, 물량이 커지면 수수료가 50% 이상을 차지한다. 그리고 **수수료 이전(raw) 구현부족만 보면 순진한 시간가중 전략이 이긴다.** 호가창이 1분 폴링, 지연 0, 히든 유동성 없음.

```
Same-side orders in the order book at time t with equal or better limit prices are given priority, since these would have been matched before our newly submitted order. Effectively, these orders increase the volume of our own limit order.
```

```
we note that – due to the elevated role of exchange commissions in overall shortfalls – differences in total shortfalls between strategies are driven by differences in realized exchange commissions rather than differences in raw implementation shortfalls.
```
- **공개 코드** **저자 코드 미공개.** 데이터는 사설 API 스크랩
- **한 줄** "비용을 제대로 모델링했다" 는 것이 곧 좋은 결과를 뜻하지 않는다 — 여기서는 그 비용 모델링 자체가 결과 전부였다.

### DeepScalper (CIKM 2022) — 호가창을 상태로 쓰지만 체결은 분봉 종가

- **venue** CIKM 2022, pp. 1858–1867 · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2201.09058) · [DOI](https://doi.org/10.1145/3511808.3557283) · [GitHub](https://github.com/TradeMaster-NTU/TradeMaster)
- **데이터** **분봉 OHLCV + 5호가 스냅샷**, 6개 자산. 지수선물 IC·IF: 학습 2019-05~12 / 테스트 2020-01~04. 국채선물 T01·T02·TF01·TF02: 학습 2017-11-29~2020-04-29 / 테스트 2020-04-30~07-17. **테스트 창이 자산당 약 50 거래일**이고 지수선물 쪽은 코로나 급락 구간이다.
- **설계** "Time is discretized into 1 min interval and we assume that the agent can only long/short a financial future at the end of each minute." 최대 보유 포지션 50.
- **지연** 없다.
- **자기영향** 없다.
- **검증** **시드 5개 + 표준편차 + Wilcoxon 검정.** 다만 단일 시간 분할이고 테스트 창이 매우 짧다.
- **지표** TR · SR · CR · SoR + 순자산 곡선.
- **다루지 않는 것** 테스트 창이 자산당 약 50일이고 하나는 코로나 급락 구간이다. 5배 레버리지가 수익률과 체결 가정 민감도를 함께 증폭한다. 비용 민감도 스윕이 없다.
- **공개 코드** 논문에 코드 URL 없음. 구현은 [TradeMaster-NTU/TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster)(Apache-2.0, star 3,059) 안에
- **한 줄** 호가창을 입력으로 쓰면서 체결은 분봉 종가로 하는 조합 — 이 군에서 가장 흔한 절충이다.

### MarS (ICLR 2025) · LOB-Bench (ICML 2025) — 생성모델로 시장을 만들고, 그 시장이 진짜인지 채점하는 두 축

- **venue** ICLR 2025 (MarS) · ICML 2025 (LOB-Bench) · **해상도** 틱·메시지 · **등급** L5 상호작용 시뮬
- **링크** [arXiv](https://arxiv.org/abs/2409.07486) · [GitHub](https://github.com/microsoft/MarS) · [웹](https://lobbench.github.io/) · [GitHub](https://github.com/peernagy/lob_bench)
- **데이터** MarS: "the **top 500 liquidity stocks in the Chinese stock market**, covering the period from **2017 to 2023** and comprising **16 billion order tokens**". LLaMA2 기반, 32B(주문 모델)·10B(주문배치 모델) 토큰 스케일링 연구. LOB-Bench: LOBSTER 형식의 GOOG·INTC.
- **자기영향** **있다.** 이 시스템의 존재 이유다.
- **검증** 정형 사실 + Wasserstein 거리 + 판별자 ROC + 충격 반응함수, IQM 과 99% 부트스트랩 신뢰구간.
- **지표** MarS: 체결률 + 가격우위. LOB-Bench: 분포 거리와 판별자.
- **다루지 않는 것** **MarS 의 대규모 가중치는 공개되지 않았다** — HuggingFace 에는 2M/5M/10M 파라미터 모델만 있다. 그리고 LOB-Bench 기준으로 현재 생성형 시뮬레이터는 전부 실데이터와 구분된다. Coletta 외(ICAIF 2023)는 한 걸음 더 나가 **적대적 전략이 생성형 시뮬레이터를 악용해 비현실적 초과수익을 낸다**는 것을 보였다.
- **공개 코드** [microsoft/MarS](https://github.com/microsoft/MarS) — star 1,783, MIT (소형 체크포인트만) / [peernagy/lob_bench](https://github.com/peernagy/lob_bench) — star 45, **LICENSE 404** · [리더보드](https://lobbench.github.io/)
- **한 줄** 재생의 자기영향 문제를 생성모델로 풀려는 시도의 최전선 — 그리고 아직 판별자가 전부 잡아낸다.

### LLM 트레이딩 에이전트 4편 (FinAgent · TradingAgents · FinCon · FinMem) — 전부 일봉, 전부 비용 0, 전부 단일 창

- **venue** KDD 2024 / ICML 2025 워크숍 / NeurIPS 2024 poster / AAAI Symposium Series (3쪽) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2402.18485) · [DOI](https://doi.org/10.1145/3637528.3671801) · [GitHub](https://github.com/TauricResearch/TradingAgents) · [GitHub](https://github.com/The-FinAI/FinCon) · [GitHub](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- **데이터** FinAgent: AAPL·AMZN·GOOGL·MSFT·TSLA·ETHUSD, **398 거래일(2022-06-01~2024-01-01), 일봉 OHLC + 차트 이미지 398장 + 뉴스 약 8~10만 건**, 학습 2022-06~2023-06 / 테스트 2023-06~2024-01. TradingAgents: **2024-01-01~03-29, 약 60 거래일**, AAPL·NVDA·MSFT·META·GOOG. FinCon: 학습 2022-01-03~10-04 / 테스트 2022-10-05~2023-06-10, 8종목. FinMem: 일간 수정종가 + 6개월 학습, 5종목.
- **설계** 매일 BUY/SELL/HOLD 하나. FinCon 만 포트폴리오 가중치 최적화를 붙였고 그것도 일간 재조정이다.
- **비용** 네 편 모두 자기 결과에 비용을 적용하지 않는다.
- **지연** 없다.
- **자기영향** 없다.
- **검증** **단일 시간 분할, 시드·온도 반복 없음.** FinAgent 의 테스트 창(2023-06~2024-01)은 대형 기술주 강세장이고, TradingAgents 의 창(2024 Q1)에서 NVDA 는 약 80% 올랐다.
- **지표** 누적수익 CR · 연환산 ARR · 샤프 · Calmar · Sortino · MDD.
- **다루지 않는 것** 체결·비용·지연·사이징·시드·표본 외 어느 것도 없다. 강세장 한 구간의 대형주 5~8개로 얻은 숫자다.

```
Limitation 3: High turnover rate results in high transaction fee. A significant drawback of the MACD strategy is its tendency to generate a high turnover rate… leading to substantial transaction costs.   — FinAgent, 기준선을 비판하며
```
- **공개 코드** [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) — **star 103,491**, Apache-2.0 / [The-FinAI/FinCon](https://github.com/The-FinAI/FinCon) — star 68, **LICENSE 없음** / [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading) — star 956, MIT / **FinAgent 은 저장소 없음** (`Open-Finance-Lab/FinAgent` 404)
- **한 줄** 가장 널리 회자되는 AI 트레이딩 연구들이 이 조사에서 백테스트 사실성이 가장 낮다 — 그리고 인기와 엄밀성이 반비례한다.

### Hendricks & Wilcox (IEEE CIFEr 2014) — "최대 10.3% 개선" 은 12칸 중 최고 칸이다

- **venue** IEEE CIFEr 2014 · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/1403.2229) · [DOI](https://doi.org/10.1109/CIFEr.2014.6924109)
- **설계** 스프레드·거래량 상태에 따라 AC 궤적을 β ∈ [0, 2](0.25 간격)로 스케일하는 표 기반 Q학습. 보고는 SBK·AGL·SAB 3종목. 학습 2012-01~06 / 테스트 2012-07~12. 호가창 참여율 상한 20%.
- **검증** 단일 분할, 시드 없음.
- **다루지 않는 것** **초록의 "up to 10.3% on average" 는 12개 행 중 최고 행이고, 그 행의 8개 시간대 성분은 {23.9, −1.4, 4.7, 13.4, 1.8, 3.3, 1.8, 35.1} 로 두 이상치가 지배한다.** 나머지 11개 행은 1.1~9.8% 이고 개별 칸은 **−26.1%·−25.2%·−24.3%** 까지 내려간다. 저자들도 "This result may be biased due to the assumption of order book resilience" 라고 적었다. 상태 세분화를 늘려도 개선이 없어 "market dynamics may not be fully represented by volume and spread state attributes" 라고 인정한다.
- **한 줄** 격자를 훑고 최고 칸을 초록에 쓰는 것 — Bailey 외가 E[max_N] 으로 형식화한 현상의 교과서적 사례.

### HALOP (IJCAI 2022) — 3초 통신 지연과 취소율 규제를 목적함수에 넣은 드문 사례

- **venue** IJCAI 2022, pp. 3912–3918 · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2207.11152) · [DOI](https://doi.org/10.24963/ijcai.2022/543)
- **설계** 10:00:00~14:30:00 의 180분을 **T = 90 구간**으로. 연속제어 PPO 가 가격변화율로 행동 부분집합을 정하고, 세분 이산 에이전트가 구체적 호가를 고른다. 원문: "**We simulate a three second communication delay so as to make the experiment close to reality. For example, an order sent at 10:02:00 can be executed after 10:02:03.** Moreover, to avoid potential large market impact, our environment **does not execute any large order whose size is larger than 1/10 of the agent's total inventory of the day**."
- **검증** **10년 학습 / 6개월 테스트인데 그 6개월이 2020년 1~6월, 즉 코로나 급락 구간이다.** t값 123.2 는 강하게 상관된 종목-일 에피소드에서 계산돼 유효 표본 수가 명목치와 전혀 다르다 — 유의성 진술로 신뢰할 수 없다. 시드 없음.
- **다루지 않는 것** 지연과 규제 제약을 넣은 것은 이 군에서 예외적으로 좋은데, 검증 구간과 t값이 그 장점을 상쇄한다.
- **한 줄** 지연 3초와 취소율 규제를 넣은 드문 백테스트 — 그런데 검증 구간이 코로나 급락 6개월 하나다.

### Karpe 외 (ICAIF 2020) — 다중 에이전트 시뮬레이터에 RL 을 넣었는데, 정형 사실이 하나도 안 바뀌었다

- **venue** ICAIF 2020 · **해상도** 틱·메시지 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2006.05574)
- **설계** ExchangeAgent + MarketreplayAgent + MomentumAgent 6명 + TWAPExecutionAgent + DDQLExecutionAgent. 학습 9일 / 테스트 9일. 10:00~15:30 안정 구간만.
- **다루지 않는 것** **저자들이 스스로 결정적 증거를 만들었다** — "For all of the stylized facts, we observe that **adding a single new agent to a simulation does not significantly alter the result of the computation**. This means that evaluating the realism of our simulation with a single DDQLExecutionAgent is equivalent to evaluating the realism of the LOB data provided as an input." 이것을 긍정적으로 서술하지만, 실제로는 **배경 흐름이 에이전트에 반응하지 않는다는 직접 증거**다 — 다중 에이전트 껍데기를 써도 재생이면 재생이다. 데이터도 2003년이라 소수점 이전 시대 미시구조다.
- **한 줄** 다중 에이전트 시뮬레이터를 써도 배경이 재생이면 자기영향은 생기지 않는다 — 저자들이 정형 사실로 그것을 증명해 버렸다.

### MASTER (AAAI 2024) · StockFormer (IJCAI 2023) — 일봉·비용 0 으로 초과수익을 보고하는 계열

- **venue** AAAI 2024 · IJCAI 2023, pp. 4766–4774 · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [DOI](https://doi.org/10.24963/ijcai.2023/530) · [GitHub](https://github.com/SJTU-DMTai/MASTER) · [GitHub](https://github.com/gsyyysg/StockFormer)
- **설계** MASTER 는 시장 정보로 종목 간 상관을 조절하는 트랜스포머로 일간 상위 30종목 재조정. StockFormer 는 SAC 로 일간 포트폴리오 가중치. StockFormer 는 학습/테스트만 있고 **검증 분할이 없다**(CSI-300 학습 1,935일 / 테스트 728일).
- **검증** 단일 시간 분할. MASTER 는 5회 실행, StockFormer 는 시드 3개.
- **다루지 않는 것** **일간 상위 30종목 재조정을 비용 0 으로 보고하는 것이 이 조사에서 가장 명백한 무비용 결과다.** 회전율이 높은 전략일수록 비용이 결론을 뒤집는다는 것은 Chen–Velikov 가 정량화했다(월 회전율 30% × 지불 스프레드 111bp = 32bp).
- **한 줄** 탑티어 AI 학회의 일봉 트레이딩 논문은 여전히 비용을 안 넣거나 수치를 안 밝힌다.

### FinRL · FinRL-Meta · TradeMaster · MacroHFT — "HFT" 라는 이름과 실제 해상도

- **venue** NeurIPS 2022/2023 D&B (FinRL-Meta·TradeMaster) · KDD 2024 (MacroHFT) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/AI4Finance-Foundation/FinRL) · [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta) · [GitHub](https://github.com/TradeMaster-NTU/TradeMaster) · [GitHub](https://github.com/ZONG0004/MacroHFT)
- **설계** `finrl/meta/env_stock_trading/env_stocktrading.py` 의 핵심 한 줄: `sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])`. 호가창도, 부분체결도, 지연도, 큐도, 자기영향도 없다. `turbulence_threshold` 로 강제 청산만 있다.
- **다루지 않는 것** **"HFT" 가 상표이지 해상도가 아니다.** MacroHFT(KDD 2024)는 분봉 종가 대 종가이고 √525600 으로 연환산한다. TradeMaster(NeurIPS 2023 D&B)는 마켓메이킹과 HFT 를 6개 과제 중 둘로 내세우면서 1분보다 세밀한 데이터를 제공하지 않는다(12일치 17,113행). **ElegantRL 은 RL 알고리즘 라이브러리이고 시장 시뮬레이터가 아예 없다.**
- **한 줄** 이 계열의 '고빈도' 는 대부분 분봉이고, 체결은 종가 전량이다 — 도구를 고를 때 이름이 아니라 체결 코드를 봐야 한다.

## 백테스트 방법론 · 과최적화 · 다중검정

이 군이 이 리포트의 판정 기준을 제공한다. 핵심 결과는 셋이다. ① 시행 횟수를 세지 않은 샤프는 해석 불가능하다 — 진짜 실력이 0이어도 10번만 시도하면 표본 내 샤프 1.57 이 나온다. ② 보정은 누출 탐지가 아니다 — 일부러 미래를 본 샤프 35짜리 오라클이 DSR 과 PBO 를 완전히 통과한다. ③ 그리고 고빈도에서는 통계보다 마찰이 먼저다 — 틱 빈도에 가장 가까운 실증 감사가 순진한 프로토콜의 샤프 3.6배 부풀림 중 대부분을 거래 마찰에 귀속시켰다. 한편 이 문헌 자체도 다투는 중이다 — CPCV 가 워크포워드를 이겼다는 유일한 통제 비교는 채점표를 만든 쪽이 이긴 것이고, PBO 는 숨은 탐색 앞에서 거꾸로 움직이며, DSR 의 추정량은 James-Stein 축소에 진다. 그리고 기호적 알파 채굴과 다중검정 보정의 교집합은 아직 비어 있다.

### Pseudo-Mathematics and Financial Charlatanism / The Probability of Backtest Overfitting — 시행 횟수를 세지 않은 백테스트는 무의미하다

- **venue** Notices of the AMS 61(5), 458– (2014) · Journal of Computational Finance (2016-09) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf) · [DOI](https://doi.org/10.1090/noti1105) · [GitHub](https://github.com/esvhd/pypbo)
- **데이터** 이론. 적용은 임의의 성과 행렬 M(관측 T′ × 시행 N).
- **검증** CSCV 는 walk-forward(한 시장 국면만 시험)와 K-fold(거짓 IID 가정과 누출)의 문제를 동시에 피한다고 저자들이 주장한다.
- **다루지 않는 것** N 은 **독립** 시행 수여야 하고, 상관된 시행에는 차원 축소(PCA)가 필요하다 — 저자들이 스스로 밝힌다. 그리고 실무에서 N 은 관측되지 않는다. 이것은 결함이 아니라 규범적 요구(시행 횟수를 보고하라)다.
- **공개 코드** [esvhd/pypbo](https://github.com/esvhd/pypbo) — PBO/CSCV 파이썬 구현 / López de Prado 의 코드 모음 quantresearch.org/Software.htm 에 CSCV 포함
- **한 줄** 샤프를 보고할 때 시행 횟수를 함께 적지 않으면, 그 샤프는 해석 불가능한 숫자다.

### The Deflated Sharpe Ratio — 시행 횟수·왜도·첨도·표본 길이를 한꺼번에 깎는다

- **venue** Journal of Portfolio Management 40(5), 94–107 (2014) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [DOI](https://doi.org/10.3905/jpm.2014.40.5.094)
- **데이터** 이론 + 적용 예시
- **다루지 않는 것** **N 하나가 아니라 시행들의 샤프 분산 `Var[{ŜR_n}]` 이 필요하다** — 즉 시도 전체의 분포를 남겨 둬야 한다. 시행 간 샤프 분포의 정규성을 가정하고, N 은 여전히 실무에서 관측되지 않는다. 저자들이 탐색 내역을 공개하지 않으면 이 장치는 작동하지 않는다.
- **공개 코드** [esvhd/pypbo](https://github.com/esvhd/pypbo) 와 저자 코드 모음(quantresearch.org)에 관련 구현
- **한 줄** 샤프를 깎는 데 필요한 것은 시행 '횟수' 만이 아니라 시행 '분포' 다 — 그래서 탐색 내역을 버리면 안 된다.

### CPCV 대 워크포워드 — 유일한 통제 비교, 그리고 그 비교의 심판이 누구인가

- **venue** Knowledge-Based Systems 305, 112477 (2024) · **해상도** 합성·모형 · **등급** L0 시뮬레이션 없음
- **링크** [DOI](https://doi.org/10.1016/j.knosys.2024.112477) · [GitHub](https://github.com/RiskLabAI/RiskLabAI.py)
- **데이터** Heston 확률변동성 · Merton 점프확산 · drift-burst · 국면전환 모형으로 만든 **합성 통제 환경** — 정답을 아는 상태에서 K-Fold · Purged K-Fold · Walk-Forward · CPCV 를 비교한다.
- **검증** 합성 통제 환경에서 정답 대비 PBO·DSR 로 네 방법을 채점.
- **다루지 않는 것** 전문을 열지 못했다. 인용이 적고 독립 재현이 없다. 채점 지표가 제안자와 같은 계열이다. 그리고 틱 빈도에 대한 CPCV 의 블록 불연속 문제는 이 논문이 다루지 않는다.
- **공개 코드** [RiskLabAI/RiskLabAI.py](https://github.com/RiskLabAI/RiskLabAI.py) · `.jl` — BSD-3 (star 4) / 대안 구현: [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) — MIT, 활발, `CombinatorialPurgedCV`·PSR·DSR·PBO·MinTRL·**MinBTL**·`effective_n_trials`·Optuna trial-Sharpe 기록기 포함 / [skfolio](https://github.com/skfolio/skfolio) — BSD-3, star 2,365, `CombinatorialPurgedCV(n_folds, n_test_folds, purged_size, embargo_size)`
- **한 줄** CPCV 가 낫다는 유일한 통제 비교가 존재하고 반박도 없지만, 채점표를 만든 쪽이 이겼다는 점과 틱 빈도의 블록 불연속 문제는 남아 있다.

### 보정의 한계 — 샤프 35짜리 누출 오라클이 DSR 과 PBO 를 그대로 통과한다

- **venue** arXiv 2608.27734 (2026) · SSRN 7346738 (미심사) · arXiv 1906.00573 · 2606.01650 · **해상도** 일봉 이상 · **등급** L0 시뮬레이션 없음
- **링크** [arXiv](https://arxiv.org/abs/2608.27734) · [GitHub](https://github.com/eslazarev/purged-cross-validation)
- **데이터** Gençay: 453종목 point-in-time 미국주식 + 39 ETF 다자산, 실제 거래·충격·대차 비용 포함. Pav: 시뮬레이션. Boutgajouft: 이론 + 시뮬.
- **검증** 누출 오라클 · 숨은 탐색 폭 · 선택적 추론 시뮬레이션.
- **다루지 않는 것** Boutgajouft 는 미심사 프리프린트이고 저자의 다른 색인 저작을 찾지 못했다. Gençay 도 arXiv 단계다.
- **공개 코드** [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) — `effective_n_trials`, Optuna trial-Sharpe 기록기 (MIT)
- **한 줄** DSR·PBO 는 시행 횟수를 깎는 도구이지 누출을 잡는 도구가 아니다 — 그리고 보고하지 않은 탐색 앞에서는 PBO 가 거꾸로 움직인다.

### How Much Sharpe is Illusory? — 틱 빈도에 가장 가까운 유일한 실증 감사, 그리고 "과최적화보다 마찰이 크다"

- **venue** SSRN 프리프린트 (2026) · DOI 10.2139/ssrn.7350238 · 미심사 · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [DOI](https://doi.org/10.2139/ssrn.7350238)
- **데이터** **Binance USDT 무기한 137종목, 2020–2024.** 여섯 팩터(모멘텀·단기반전·변동성·규모/유동성·펀딩캐리·베타).
- **검증** **중첩 워크포워드 선택 + 사전 확정 24개 설정 격자 + DSR.**
- **지표** 연환산 샤프(순진 vs 엄격), 표본 외 샤프.
- **다루지 않는 것** 미심사. 전문 미확인이라 체결 모형과 분해 방법의 세부를 알 수 없다.
- **공개 코드** 없음(확인 실패)
- **한 줄** 순진한 프로토콜은 샤프를 평균 3.6배 부풀리고, 그 격차의 대부분은 과최적화가 아니라 거래 마찰에서 온다.

### 다중검정 문턱 논쟁 — t > 3.0 학파와 그 반론, 그리고 대량 탐색이 자산일 수 있다는 결과

- **venue** Review of Financial Studies 29(1), 5–68 (2016) 외 · **해상도** 일봉 이상 · **등급** L0 시뮬레이션 없음
- **링크** [PDF](https://academic.oup.com/rfs/article-pdf/29/1/5/24450794/hhv059.pdf) · [DOI](https://doi.org/10.1093/rfs/hhv059) · [GitHub](https://github.com/OpenSourceAP/CrossSection) · [웹](https://people.duke.edu/~charvey/backtesting/Haircut_SR.m) · [GitHub](https://github.com/chenandrewy/high-throughput-ap) · [GitHub](https://github.com/bkelly-lab/ReplicationCrisis)
- **데이터** HLZ: 출판된 300여 개 횡단면 팩터. Chen & Dim: **채굴 전략 136,000개**. JKP: 글로벌 팩터 데이터.
- **검증** Bonferroni·Holm·BHY haircut, 최대통계량 부트스트랩, 계층 베이즈, 경험적 베이즈 축소.
- **다루지 않는 것** 이 논쟁의 본질은 **'무작위로 고른 후보가 귀무일 사전확률'** 에 대한 이견이다. 데이터로 결판나지 않는다.
- **공개 코드** [Haircut_SR.m · Profit_Hurdle.m · sample_random_multests.m](https://people.duke.edu/~charvey/backtesting/Haircut_SR.m) — Harvey 의 MATLAB / [Lucky Factors FactorTests_boot.m](https://people.duke.edu/~charvey/Lucky/FactorTests_boot.m) / [OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) — 319 예측변수 + `SignalDoc.csv`. star 1,041, GPL-2.0 · [openassetpricing.com](https://www.openassetpricing.com/) / [chenandrewy/high-throughput-ap](https://github.com/chenandrewy/high-throughput-ap) — 채굴 전략 136,000개 + 경험적 베이즈. CC0 / [bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis) — JKP 계층 베이즈. star 378
- **한 줄** "몇 번 시도했는가" 를 성과에서 깎는 것이 표준이 됐지만, 탐색을 전부 기록하면 그 크기가 오히려 추정 자원이 된다는 반대 결과도 있다.

### The Virtue of Complexity 와 Nagel 의 반박 — 표본 외 분할로는 절대 잡히지 않는 실패 모드

- **venue** Journal of Finance 79(1), 459–503 (2024) vs NBER WP 34104 (2025) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [PDF](https://www.nber.org/papers/w30217) · [DOI](https://doi.org/10.1111/jofi.13298)
- **데이터** **월간** 미국 시장 수익률 — CRSP 가치가중 지수 1926–2020, Goyal–Welch 예측변수 15개(+ 시차 지수수익률)를 Rahimi–Recht Random Fourier Features 로 **P 를 최대 12,000 까지** 확장하고 **T = 12 개월** 같은 짧은 롤링 창으로 학습.
- **검증** 표준 표본 외 분할을 통과했는데도 결과가 인공물이었다 — **그것이 이 항목의 요점이다.**
- **다루지 않는 것** **표본 외 분할이 잡지 못하는 실패 모드가 있다.** 추정기의 기하학이 데이터의 한 실현과 상호작용해 전략을 '제조' 하는 경우다. 진단법은 싸다 — 자기상관 부호를 뒤집은 반사실 데이터에 같은 파이프라인을 돌려 같은 전략이 나오는지 보는 것. **이 조사의 호가창·RL 문헌에서 아무도 이것을 하지 않는다.**
- **공개 코드** 공개 코드 없음(양쪽 다)
- **한 줄** 표본 외에서 잘 나왔는데도 인공물일 수 있다 — 추정기가 데이터와 무관하게 특정 전략을 만들어 내는 경우, 어떤 분할도 그것을 잡지 못한다.

### Gort 외 — 자기 과최적화 확률을 실제로 측정한 유일한 RL 논문

- **venue** arXiv 2209.05559 (저널·학회 게재 없음) · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2209.05559)
- **설계** 학습 02/02~04/30(T=25,055) / 테스트 05/01~06/27(T′=16,704) — 두 번의 급락이 포함된 구간. **가설검정을 명시적으로 세운다** — H₀: p < α (과적합 아님) vs H₁: p ≥ α, α = 10%. PBO 는 CSCV 로 추정(N=5 부분집합, k=2 검증, J=10 분할). 시행 수는 `1 − (1−0.05)^H ≥ 0.9 ⟹ H ≥ 50` 으로 정당화. 하이퍼파라미터 격자 **2,700 = 5×4×5×3×3×3** 중 50개 표본.
- **검증** 결과(누적수익 / 과적합 확률): S&P BDM −50.78% / — · 동일가중 −47.78% / — · PPO-WF −49.39% / **17.5%(기각)** · PPO-KCV −55.54% / 7.9% · **PPO(조합 CV) −34.96% / 8.0%** · TD3 −59.08% / 9.6% · SAC −59.48% / **21.3%(기각)**.
- **다루지 않는 것** **양방향의 교훈이다.** 모든 전략이 돈을 잃었고 '승리' 는 50% 대신 35% 를 잃은 것이다. 그리고 **PPO-KCV 는 과적합 검정을 통과(7.9%)하면서 세 PPO 변형 중 표본 외 수익이 가장 나빴다(−55.54%)** — 낮은 PBO 가 좋은 표본 외 성과를 예측한다는 주장을 정면으로 흔든다. 가설검정도 관행과 반대로 세워져 있어(H₀ 가 바람직한 상태) '기각 실패' 를 무과적합의 증거로 쓰는데 검정력 분석이 없다. 본문 내부 불일치도 있다(부분행렬 14개 vs N=5·k=2·J=10).
- **한 줄** 자기 과최적화 확률을 재려는 시도 자체가 드물고 값지지만, 그 결과가 낮은 PBO 와 좋은 성과가 붙어 있지 않다는 것도 함께 보여 준다.

### A Backtesting Protocol in the Era of Machine Learning (Arnott · Harvey · Markowitz)

- **venue** Journal of Financial Data Science (2019) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [DOI](https://doi.org/10.3905/jfds.2019.1.064)
- **설계** 7개 항목의 연구 프로토콜. 이 리포트에서 특히 중요한 넷 — **#2 시도한 모든 모형을 기록한다(성공·실패 모두)**, **#3 이상치·윈저화 규칙을 사전 선언하고 규칙을 하나만 시도한다**, **#4c 표본 외 분석이 비용과 데이터 수정까지 포함해 실거래를 대표하는가**, **#7 대부분의 검정이 실패할 것을 예상하라.** 그리고 #4a(실거래와의 대조 계획)와 #5b(과밀화 가정 명시).
- **다루지 않는 것** 규범이지 통계량이 아니다. 지키는지 여부는 저자의 자기보고에 달려 있다.
- **한 줄** "시도한 것을 전부 기록하고, 규칙은 사전에 하나만 정하고, 대부분 실패할 것을 예상하라" — 이 리포트의 체크리스트가 여기서 나온다.

### 기호적 알파 채굴과 다중검정 보정의 교집합은 비어 있다

- **venue** arXiv 2502.16789 · 2606.20625 · 2606.29194 (2025–2026) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2502.16789) · [GitHub](https://github.com/jarrettyu/AlphaMemo) · [GitHub](https://github.com/trevorstephens/gplearn) · [GitHub](https://github.com/astroautomata/PySR) · [GitHub](https://github.com/ICT-FinD-Lab/alphagen)
- **설계** **AlphaAgent** — 유전 프로그래밍이 "face rapid alpha decay from overfitting and complexity" 하는 문제를 (i) **AST 기반 유사도로 독창성 강제**, (ii) 가설–팩터 의미 정합, (iii) **AST 구조 제약으로 복잡도 통제** 로 푼다. **AlphaMemo** 는 AST diff 에서 "어떤 편집 모티프가 되고 안 되는지" 를 기억한다. **Alpha Singularity** 가 실패 모드를 가장 날카롭게 진단한다 — "Automated alpha mining holds the scoring function fixed and varies the search algorithm over it. **A search that converges against a fixed scorer overfits whatever the scorer cannot penalize, a primary cause of the out-of-sample generalization gap.**"
- **다루지 않는 것** **세 편 어디에도 DSR·PBO·haircut·다중검정 보정이 없다.** AST 복잡도 벌점·독창성 정규화·봉인 홀드아웃으로 과적합을 다루는데, 이것들은 실재하고 유용하지만 선택편의 기계장치와는 **직교**한다. 도구 쪽도 같다 — `gplearn`·`PySR`·`alphagen` 모두 다중검정 통제가 없다.
- **한 줄** 기호 회귀로 알파를 캐는 문헌과 다중검정을 보정하는 문헌이 아직 만나지 않았다 — 그 교집합이 비어 있다는 것 자체가 기여 지점이다.

### mlfinlab 은 오픈소스가 아니다 — purged CV·CPCV 의 참조 구현으로 인용하면 안 되는 이유

- **venue** 상용 구독 소프트웨어 · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [GitHub](https://github.com/hudson-and-thames/mlfinlab) · [GitHub](https://github.com/eslazarev/purged-cross-validation) · [GitHub](https://github.com/skfolio/skfolio) · [GitHub](https://github.com/emoen/Machine-Learning-for-Asset-Managers)
- **설계** README 원문 — "This repo is public facing and exists for the **sole purpose of providing users with an easy way to raise bugs**, feature requests, and other issues." 실제로 `cross_validation.py` 는 **272줄에 `pass` 문이 10개**로 모든 함수 본문이 스텁이다. `ml_get_train_times` 는 "Advances in Financial Machine Learning, Snippet 7.1, page 106" 를 인용하는 완전한 docstring 뒤에 `pass` 하나뿐이고, `combinatorial.py` 에도 스텁 9개, `PurgedKFold.__init__` 은 인자만 받고 아무것도 안 한다.
- **다루지 않는 것** LICENSE.txt 가 OSI 라이선스가 아니라 **독점 구독 계약**이다(GitHub 표기 `NOASSERTION`). PyPI 에 없고, 마지막 push 가 2023-10-02 로 사실상 방치돼 있으며, **미해결 정확성 이슈 #295 — "PurgedKFold: training events can overlap with testing events"** 가 재현 예제와 함께 열려 있다. 즉 **정작 쓰고 싶은 그 함수에 문서화된 누출 버그가 있다.**
- **한 줄** "purged K-fold 의 참조 구현은 mlfinlab" 이라는 인용은 독자가 얻을 수도 열어 볼 수도 없는 것을 가리킨다 — `purgedcv`·`skfolio`·`RiskLabAI` 를 쓰고 공식은 quantresearch.org 를 인용하라.

## 오픈소스 틱 백테스트 엔진

연구용 틱/호가창 백테스트가 실제로 가능한 오픈소스는 넷뿐이다 — hftbacktest · NautilusTrader(1.223+) · ABIDES · JAX-LOB. 나머지는 “고빈도” 를 표방하더라도 봉에서 체결한다. 이 계층의 공통 환상은 하나다 — 봉의 저가가 내 지정가를 스쳤으면 전량, 지정가 그대로 체결. 패시브 전략의 엣지 전부에 해당하는 크기다. 가장 큰 평판-실체 격차는 Hummingbot 이다 — 가장 널리 쓰이는 마켓메이킹 봇인데 백테스터가 1분 캔들 종가로 체결을 정한다. 그리고 이 모든 것의 상한을 정하는 것은 알고리즘이 아니라 데이터 등급이다.

### hftbacktest — 큐 위치와 지연을 정면으로 모델링하는 오픈소스 HFT 백테스터

- **venue** 오픈소스 (MIT) · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [GitHub](https://github.com/nkaz001/hftbacktest) · [문서](https://hftbacktest.readthedocs.io/)
- **데이터** 이벤트 8필드 구조체 배열(npz/npy): `ev, exch_ts, local_ts, px, qty, order_id, ival, faval`. **`exch_ts`(거래소 시각)와 `local_ts`(내가 받은 시각)를 함께 저장**하는 것이 이 엔진의 토대다 — 피드 지연 모델링이 여기서 나온다. `order_id` 는 L3 MBO 피드에만 채워진다. Databento·Tardis·Binance Futures 변환기를 제공하고, 깊이 백엔드는 `HashMapMarketDepth`·`BTreeMarketDepth`·`ROIVectorMarketDepth`. 나노초 권장.
- **설계** 단일 에이전트 재생. 다만 재생의 정확도를 높이는 데 설계를 다 쓴다 — 로컬/거래소 두 시계, 큐 위치 추정, 지연 왕복.
- **체결** `is_filled` 가 앞 물량이 0 아래로 내려간 양을 lot 단위로 반올림해 체결량으로 돌려준다. 부분체결이 자연스럽게 나온다. 잔량 증가는 앞 물량을 바꾸지 않는다(뒤에 붙은 것으로 본다).
- **비용** `TradingValueFeeModel` 이 체결 금액에 maker/taker 요율을 곱한다. `DirectionalFees` 로 매수/매도에 다른 요율을 더할 수 있어 거래세 같은 편측 비용도 표현 가능하다.
- **지연** **있다.** `ConstantLatency(entry_latency, response_latency)` 또는 실측 지연을 보간하는 `IntpOrderLatency`. 조사한 도구 중 지연을 실측 데이터로 보간하는 것은 이것이 사실상 유일하다.
- **자기영향** **없다.** 문서가 먼저 밝힌다 — "HftBacktest is a market-data replay-based backtesting tool, which means your order cannot make any changes to the simulated market, no market impact is considered."
- **검증** 엔진이지 방법론이 아니다. 표본 분리·다중검정은 사용자 몫이다.
- **지표** equity·position·수수료 누적 등 상태를 노출하고, 지표 계산은 사용자(또는 동봉 노트북)가 한다.
- **다루지 않는 것** 자기영향이 없다. 그리고 큐 모델은 어디까지나 **추정**이다 — L2 데이터만 있으면 취소와 체결을 직접 가를 수 없어서 확률 함수의 형태(`n` 값)가 결과를 흔든다. 문서도 이를 근사라고 명시한다. 또한 확률 큐 갱신은 몬테카를로 추첨이 아니라 **결정론적 기대값 갱신**이라, 한 번 돌리면 분포가 아니라 경로 하나가 나온다.

```
// docs: Regardless of the quantity at the best, liquidity-taking orders will be fully executed
// at the best. Be aware that this may cause unrealistic fill simulations if you attempt to
// execute a large quantity.
```

```
/// Provides a conservative queue position model, where your order's queue position advances only
/// when trades occur at the same price level.
pub struct RiskAdverseQueueModel<MD>(PhantomData<MD>);
```

```
        let front = q.front_q_qty;
        let back = prev_qty - front;

        let mut prob = self.prob.prob(front, back);
        if prob.is_infinite() {
            prob = 1.0;
        }

        let est_front = front - (1.0 - prob) * chg + (back - prob * chg).min(0.0);
        q.front_q_qty = est_front.min(new_qty);
```

```
/// If latency has a negative value, it indicates an order rejection by the exchange and its
/// value represents the latency that the local experiences when receiving the rejection
/// notification.
```
- **공개 코드** [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) — Rust 코어 + `py-hftbacktest`. MIT · star 4,642 · fork 897 · 최근 push 2025-12-23 / 동봉 노트북 `examples/Probability Queue Models.ipynb`, `examples/Queue-Based Market Making in Large Tick Size Assets.ipynb` 가 큐 모델 선택이 결과를 얼마나 바꾸는지 보여 준다
- **한 줄** "틱을 되감았다" 와 "체결을 재현했다" 를 가르는 기준선. 우리 백테스트가 같은 층에 있는지 확인할 때 대는 자 역할을 한다.

### NautilusTrader — 백테스트와 실거래가 같은 코드를 쓰는 엔진, 그리고 1.223.0 에 들어온 큐 위치

- **venue** 오픈소스 (LGPL-3.0) · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [GitHub](https://github.com/nautechsystems/nautilus_trader) · [문서](https://nautilustrader.io/docs)
- **데이터** 데이터 등급을 문서가 명시적으로 서열화한다 — L3 MBO → L2 MBP → L1 quote → trade → bar. 거래소 `book_type` 은 `L1_MBP`(기본)·`L2_MBP`·`L3_MBO`. 나노초. 어댑터 약 20종 + Databento·Tardis.
- **설계** 단일 에이전트 재생 + 매칭 엔진. L2/L3 에서는 시장가가 호가를 갉아먹고 지정가는 교차 가격을 취하거나 대기한다. L1 에서는 결정론적 1틱 잔여 규칙 + 확률적 슬리피지를 쓴다.
- **체결** 체결 틱이 패시브 체결을 일으키고(`trade_execution=True` 기본), `min(order.leaves_qty, trade.size)` 로 제한한다. 별도 스위치 `liquidity_consumption=True` 는 레벨별 소비량을 추적해 "같은 표시 잔량이 한 번의 반복에서 여러 주문을 체결시키는" 문제를 막는다. **문서가 남은 낙관을 스스로 적어 둔다** — `NO_AGGRESSOR` 체결은 양쪽 큐를 모두 줄여서 실제보다 일찍 체결시킬 수 있다.
- **비용** 상품 단위 maker/taker 요율. 옵션 상한·계단형 명목가까지 있다.
- **지연** `StaticLatencyModel(base, insert, update, delete)` — 기본 지연에 연산별 지연이 **가산**된다(base 100ms + insert 200ms = 300ms). 삽입/수정/취소를 분리한 것은 hftbacktest 보다 세밀하다. 반면 **실측 지연을 보간하는 모델이 없고**, 피드 지연 개념도 없다.
- **자기영향** **없다.** 문서가 명시한다 — "Historical data cannot show how a simulated order would have interacted with other market participants."
- **검증** 엔진이다. 표본 분리·다중검정은 사용자 몫.
- **지표** 포트폴리오·체결·수수료 이벤트를 모두 남기고 리포트를 생성한다.
- **다루지 않는 것** L2 에서 내 앞의 **취소**를 확률로 다루지 않는다(잔량 갱신 시 앞 물량을 표시 잔량으로 clamp 할 뿐). 실측 기반 지연 모델이 없다. 자기영향 없음. 그리고 **1.223.0 이전 버전에는 큐 추적이 전혀 없다** — 버전 확인이 필수다.

```
Nautilus cannot generate higher granularity data (L2 or L3) from lower-level data such as quotes, trades, or bars.
If you specify `L2_MBP` or `L3_MBO` as the venue's `book_type`, quotes and bars will not update the book … otherwise orders may appear as though they are never filled.
```

```
1. On acceptance, a LIMIT order snapshots same-side displayed size at its price.
2. Correct-side trades at that price reduce the quantity ahead.
3. The order becomes fill-eligible when the quantity ahead reaches zero.
4. Only trade volume beyond the cleared queue is available to fill on that tick.
```
- **공개 코드** [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) — LGPL-3.0 · star 28,684 · fork 3,745 · v1.231.0 / v2.0.0rc4 · push 2026-09-09
- **한 줄** 다자산·실거래 병행이 필요하면 여기, 암호화폐 L2 마켓메이킹 체결 재현이 목적이면 hftbacktest — 두 도구의 분업이 명확하다.

### ABIDES / ABIDES-Gym — 내 주문이 시장을 바꾸는 유일한 오픈소스 시뮬레이터

- **venue** arXiv 1904.12066 (ABIDES) · arXiv 2110.14771 (ABIDES-Gym, ICAIF 계열) · 오픈소스 BSD-3 · **해상도** 틱·메시지 · **등급** L5 상호작용 시뮬
- **링크** [arXiv](https://arxiv.org/abs/1904.12066) · [GitHub](https://github.com/abides-sim/abides) · [GitHub](https://github.com/jpmorganchase/abides-jpmc-public)
- **데이터** 재생 엔진이 아니다. "데이터"가 곧 **다른 에이전트**다 — zero-intelligence · momentum · value · noise · market maker 에이전트들이 주문을 낸다. 메시지 설계는 "modeled after NASDAQ's published equity trading protocols ITCH and OUCH". 나노초 정수 시계. 과거 데이터로 씨를 뿌리거나 보정할 수 있고 `realism/order_flow_stylized_facts.py` 로 정형 사실 검증을 한다.
- **설계** 이산사건 커널 위의 다중 에이전트. "tens of thousands of trading agents" 를 표방한다.
- **체결** 진짜 호가창 매칭. 부분체결을 최우선 가격부터 순차 소비한다.
- **비용** 수수료 회계가 거의 없다. 필요하면 사용자가 붙여야 한다.
- **지연** 쌍별·방향별 네트워크 지연 + 거래소 파이프라인/연산 지연 + 에이전트별 연산 지연. 이 조사에서 **에이전트 간 지연**을 모델링하는 유일한 도구다(다른 에이전트가 있는 유일한 도구이기 때문이다).
- **자기영향** **있다.** 이 도구의 존재 이유다.
- **검증** 정형 사실 대조(`realism/`)로 시뮬레이터의 사실성을 검증한다. 전략 검증은 사용자 몫.
- **지표** 환경이므로 사용자 정의.
- **다루지 않는 것** 반사실적 시장이 합성이다. 결과의 신뢰도가 **에이전트 보정 품질에 전적으로** 달린다. 순수 파이썬 단일 스레드 커널이라 절대 속도가 느리다. 그리고 두 저장소 모두 정체 상태이며 JPMC 포크는 **archived** 다.
- **공개 코드** [abides-sim/abides](https://github.com/abides-sim/abides) — 원본. BSD-3 · star 564 · push 2023-07-06 / [jpmorganchase/abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public) — ABIDES-Core / Markets / Gym 분리판. BSD-3 · star 174 · **archived** · push 2024-07-22
- **한 줄** 재생 백테스트가 구조적으로 못 푸는 자기영향 문제의 유일한 오픈소스 답 — 대신 시장이 진짜가 아니다.

### JAX-LOB / AlphaTrade — GPU 위에서 수천 개 호가창을 동시에 굴리는 L3 시뮬레이터

- **venue** arXiv 2308.13289 · ICAIF 2023 · **해상도** 틱·메시지 · **등급** L4 재생+큐·지연
- **링크** [arXiv](https://arxiv.org/abs/2308.13289) · [GitHub](https://github.com/KangOxford/jax-lob) · [GitHub](https://github.com/KangOxford/AlphaTrade)
- **데이터** LOBSTER 의 message/orderbook CSV 를 `data/Flow_10/`·`data/Book_10/` 에서 읽고 `message.type.isin([1,2,3,4])` 로 거른다. `nOrdersPerSide=100`, `book_depth=10`. 주문 ID 와 초·나노초 시각 필드를 갖는 **완전한 L3**.
- **설계** 메시지 재생 + 에이전트 주문 삽입. `scan_through_entire_array` 로 스트림을 훑는다.
- **체결** 가격-시간 우선. 내 주문도 같은 배열에서 같은 규칙으로 처리된다.
- **비용** 없다. 보상 함수는 TWAP 대비 집행 품질이다.
- **지연** **없다.** 이 도구의 가장 큰 사실성 공백.
- **자기영향** **부분적.** 유동성은 소비하지만 시장이 반응하지 않는다.
- **검증** RL 학습·평가 분할은 사용자 몫.
- **지표** TWAP 대비 집행 가격 등.
- **다루지 않는 것** 지연 없음 · 시장 무반응 · 고정 용량 배열(측당 100주문) · **두 저장소 모두 라이선스 파일이 없다.** 라이선스 부재는 권리 부여가 없다는 뜻이므로, 저자에게 문의하지 않는 한 읽기 전용 참고로만 다뤄야 한다.
- **공개 코드** [KangOxford/jax-lob](https://github.com/KangOxford/jax-lob) — 논문 저장소. **라이선스 없음** · star 59 · branch `jaxV3` · push 2023-10-22 / [KangOxford/AlphaTrade](https://github.com/KangOxford/AlphaTrade) — 유지되는 후속. **라이선스 없음** · star 145 · push 2026-03-16
- **한 줄** L3 FIFO 를 정확히 하면서 GPU 로 수천 배 빠르게 — 대신 지연과 시장 반응을 통째로 버렸다.

### Microsoft Qlib — "고빈도" 를 표방하지만 바닥이 분봉인 AI 퀀트 플랫폼

- **venue** 오픈소스 (MIT) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/microsoft/qlib) · [문서](https://qlib.readthedocs.io/)
- **데이터** 일봉이 기본이고 `examples/highfreq/` 가 **분봉** 데이터셋이다. `qlib/rl/order_execution/`(TWAP·PPO·OPDS, `simulator_simple.py`/`simulator_qlib.py`)도 **분봉 일중 집행**이다. 호가창은 코드 어디에도 없다.
- **체결** `trade_price = self.get_deal_price(...)` → `order.deal_amount = order.amount` 로 두고 `_clip_amount_by_volume` 으로 잘라 낸다. 큐도 부분체결 확률도 없다.
- **지연** 없다.
- **자기영향** **비용으로만.** 이차항이 내 비용을 올리지만 시장은 바뀌지 않는다.
- **검증** 사용자 몫. 롤링 학습 예제가 있으나 다중검정 장치는 없다.
- **지표** 수익률·정보비율·최대낙폭 등 표준 지표.
- **다루지 않는 것** "high-frequency" 라는 예제 폴더 이름이 오해를 부른다 — 실제로는 **분봉**이다. 틱 백테스트 도구로 인용하면 안 된다.
- **공개 코드** [microsoft/qlib](https://github.com/microsoft/qlib) — MIT · star 48,430 · push 2026-09-02
- **한 줄** AI 퀀트 파이프라인으로는 최고 수준이지만, 그 '고빈도' 는 분봉이다.

### 틱 데이터 공급원 지도 — LOBSTER · Databento · Tardis · Binance · Polygon(Massive) · Kaiko

- **venue** 1차 자료 조사 · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [GitHub](https://github.com/databento/dbn) · [웹](https://lobsterdata.com) · [GitHub](https://github.com/databento/databento-python) · [GitHub](https://github.com/tardis-dev/tardis-machine) · [GitHub](https://github.com/binance/binance-public-data) · [GitHub](https://github.com/massive-com/client-python) · [GitHub](https://github.com/Leo4815162342/dukascopy-node)
- **데이터** L3(주문 ID 있음) · L2(가격대별 집계) · L1(최우선호가) 세 등급으로 갈라 각 공급원이 무엇을 주는지 확인
- **다루지 않는 것** **L4 이상의 백테스트를 하려면 L3 또는 최소한 이중 타임스탬프가 붙은 L2 가 필요하다.** 무료로 구할 수 있는 것은 L1 이거나 분당 스냅샷이라 큐를 세울 수 없다.
- **공개 코드** [databento/dbn](https://github.com/databento/dbn) (Apache-2.0, star 170) · [databento/databento-python](https://github.com/databento/databento-python) (Apache-2.0, star 297) / [tardis-dev/tardis-machine](https://github.com/tardis-dev/tardis-machine) (MPL-2.0, star 310) · [tardis-dev/tardis-python](https://github.com/tardis-dev/tardis-python) (MPL-2.0, star 147) / [binance/binance-public-data](https://github.com/binance/binance-public-data) (star 2,481, 라이선스 없음) · [massive-com/client-python](https://github.com/massive-com/client-python) (MIT, star 1,505) / [Leo4815162342/dukascopy-node](https://github.com/Leo4815162342/dukascopy-node) (MIT, star 899)
- **한 줄** 체결 시뮬레이션의 상한은 알고리즘이 아니라 데이터 등급이 정한다 — 그리고 무료로 얻을 수 있는 것은 대부분 L1 이다.

### QuantConnect Lean

- **venue** 오픈소스 (Apache-2.0) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/QuantConnect/Lean)
- **설계** 이 조사에서 **수수료 생태계가 가장 풍부하다** — InteractiveBrokers·Binance·Bybit·Coinbase·Kraken·Alpaca·Schwab·Zerodha·dYdX 등 거래소별 `FeeModel`. 슬리피지도 `VolumeShareSlippageModel`·`MarketImpactSlippageModel`·`ConstantSlippageModel`·`AlphaStreamsSlippageModel`.
- **다루지 않는 것** 체결 코드가 솔직하다. `FillModel.cs`: `if (prices.Low < limitPrice) { fill.Status = Filled; fill.FillPrice = Math.Min(prices.High, limitPrice); fill.FillQuantity = quantity; }`. 더 정교한 `EquityFillModel.cs` 에는 TODO 가 그대로 남아 있다 — `// assume the order completely filled` / `// TODO: Add separate DepthLimited fill partial order quantities based on tick quantity / bar.Volume available.` **한국투자증권 backtester 가 쓰는 엔진이 바로 이것이다.**
- **한 줄** 다자산 실거래 플랫폼으로는 훌륭하고 수수료 모델은 최고 수준이지만, 지정가 체결은 봉 터치 전량 체결이다.

### Hummingbot — 가장 널리 쓰이는 오픈소스 마켓메이킹 봇, 그런데 백테스터는 1분봉 종가

- **venue** 오픈소스 (Apache-2.0) · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/hummingbot/hummingbot)
- **설계** 포지션 실행기의 진입 조건이 `entry_condition = (df['close'] <= config.entry_price)`(매수). 고가/저가 터치도 아니고 **종가 교차**다 — 봉 기반 중에서도 약한 축이다. 수수료는 `cum_fees_quote = 2 * trade_cost * filled_amount_quote`.
- **다루지 않는 것** **이 조사에서 평판과 실체의 격차가 가장 큰 도구다.** Hummingbot PMM 스프레드를 이 백테스터로 최적화하면 체결에 대해서는 아무것도 측정하지 않는 것이다.
- **한 줄** 마켓메이킹 봇의 백테스터가 캔들 종가로 체결을 정한다 — 스프레드 최적화 결과를 믿으면 안 되는 이유.

### mbt_gym — 모형 기반 마켓메이킹 RL 환경 (재생이 아니다)

- **venue** ICAIF 2023 · 오픈소스 BSD-3 · **해상도** 합성·모형 · **등급** L1 모형 몬테카를로
- **링크** [GitHub](https://github.com/JJJerome/mbt_gym)
- **설계** 체결 함수 `Exponential`·`Triangular`·`Power`·`ExogenousMm`, 도착 모형 `Poisson`·`PoissonNonLinear`·`Hawkes`, 중간가 모형 선택 가능. 벡터화 환경으로 RL 학습이 빠르다.
- **다루지 않는 것** 실제 전략 검증 도구가 **아니다.** 정답을 아는 모형 안에서 최적제어 정책을 시험하는 용도로는 정확한 도구다 — 용도를 혼동하면 안 된다.
- **한 줄** "모형이 옳다면 내 정책이 최적인가" 를 묻는 도구지, "이 전략이 시장에서 통했는가" 를 묻는 도구가 아니다.

### 봉 기반 백테스터 계열 — backtesting.py · backtrader · zipline · vectorbt · PyBroker · bt · Freqtrade

- **venue** 오픈소스 · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/kernc/backtesting.py) · [GitHub](https://github.com/mementum/backtrader) · [GitHub](https://github.com/stefan-jansen/zipline-reloaded) · [GitHub](https://github.com/polakowo/vectorbt) · [GitHub](https://github.com/edtechre/pybroker) · [GitHub](https://github.com/freqtrade/freqtrade)
- **설계** `backtesting.py`(AGPL-3.0, star 8,947)는 `is_limit_hit = low <= order.limit`(롱) — **거래량 확인조차 없다.** `backtrader`(GPL-3.0, star 23,165, 2024-08 이후 사실상 정지)는 `_try_exec_limit` → `elif plimit >= plow: self._execute(order, ago=0, price=plimit)` 이지만 `slip_perc`/`slip_fixed`/`slip_match`/`slip_limit`/`slip_out` 과 거래량 `filler` 로 부분체결을 줄 수 있다. `zipline-reloaded`(Apache-2.0)의 `VolumeShareSlippage` 는 `price * (1 ± price_impact * volume_share²)` 에 **분봉 거래량의 2.5% 상한**(`DEFAULT_EQUITY_VOLUME_SLIPPAGE_BAR_LIMIT`)을 둔다 — 봉 계열에서 가장 쓸모 있는 사실성 장치다. `vectorbt`(star 9,041)는 `adj_price = price * (1 + slippage)`; 틱 인덱스 DataFrame 을 넣어도 결국 넘긴 가격 열에서 체결한다. `PyBroker` 는 **Apache + Commons Clause** 로 OSI 오픈소스가 아니다.
- **다루지 않는 것** **`Freqtrade`(GPL-3.0, star 54,183)의 문서가 이 계열에서 가장 정직하다** — "All orders are filled at the requested price (no slippage) as long as the price is within the candle's high/low range", "Stoploss exits happen exactly at stoploss price, even if low was lower", "Low happens before high for stoploss, protecting capital first". 그리고 `--timeframe-detail`(1시간 신호 + 5분 청산 평가)을 "전략이 백테스트 가정을 악용하고 있지 않은지 확인하는" 마지막 단계로 권한다.
- **한 줄** 이 계열의 공통 환상은 하나다 — "봉의 저가가 내 지정가를 스쳤으면 전량 체결". 패시브 전략의 엣지 전부에 해당하는 크기다.

## 대한민국 — 증권사 공개 코드 · KRX 틱 연구

결론부터: 국내 대형 증권사 중 백테스트 코드를 공개한 곳은 한국투자증권 하나이고, 그 백테스트는 일봉이다. 키움은 공식 REST 저장소를 열었지만 백테스터가 없고, NH 는 2026년 8월 PLUG 로 뒤늦게 합류했으나 저장소 3개가 SDK·MCP 다. LS·대신·미래에셋은 공식 GitHub 조차 없어 생태계를 개인 개발자 래퍼가 떠받친다. 더 근본적인 제약은 데이터다 — KRX Open API 는 31개 서비스가 전부 일별이고, 코스콤 데이터몰은 서비스 중단 상태이며, 한국투자증권의 체결 API 는 당일치뿐이다. 그래서 국내에서 호가창 재생 백테스트를 하려면 실시간을 직접 수집하거나 기관 계약을 맺는 두 길밖에 없다.

### 한국투자증권 open-trading-api / backtester — 국내 대형 증권사가 공개한 유일한 백테스트 엔진

- **venue** 공식 오픈소스 (GitHub, 2026) · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/koreainvestment/open-trading-api) · [웹](https://apiportal.koreainvestment.com/) · [GitHub](https://github.com/koreainvestment/kis-ai-extensions)
- **데이터** KIS Open API `/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice`(TR `FHKST03010100`)로 받은 **일봉**을 CSV 로 떨어뜨려 Lean 데이터 디렉터리 `data/equity/krx/daily/{symbol}.csv` 에 넣는다. `KISDataProvider.get_history()` 가 받는 해상도는 `Resolution.DAILY` 와 `Resolution.MINUTE` 둘뿐이고, 그 외에는 `ValueError(f"지원하지 않는 해상도: {resolution}")`. 분봉은 `_get_minute_bars(symbol, end)` 에 주석으로 "분봉은 단일 날짜" 라고 못 박혀 있다. **틱·호가는 백테스트 경로에 아예 없다.**
- **설계** **과거 일봉 재생**. 사용자가 고른 지표 조건(골든크로스·RSI·MACD+볼린저 등 프리셋 10종 + 지표 80여 종)이 참이 되는 날 매수/매도 신호를 내고 Lean 이 그 신호로 주문을 낸다. 상호작용도, 호가창도 없다.
- **체결** Lean 의 기본 주문 체결 모델을 그대로 쓴다. 저장소 문서·코드 어디에도 지정가 대기·부분체결·체결 확률에 대한 별도 설정이 없다. 즉 **일봉 하나에 한 번, 그날 가격에 원하는 수량이 다 체결된다**고 본다.
- **비용** 직접 정의한 `CustomFeeModel`: 매수는 `value × 0.00015`, 매도는 `value × (0.00015 + 0.002)`. 슬리피지는 `CodeGenConfig.slippage = 0.0` 이 **기본값이라 꺼져 있고**, 0보다 크게 주면 KRX 호가단위 계단으로 반올림한 `KRXSlippageModel` 이 붙는다.
- **지연** 없음. 일봉 단위라 지연이라는 개념이 성립하지 않는다.
- **자기영향** 없음. 단일 에이전트 재생이다.
- **검증** 표본 분리·다중검정 장치가 없다. 제공되는 것은 `Grid/Random Search` **파라미터 최적화**뿐이고, 최적화한 구간과 평가 구간을 나누라는 안내가 문서에 없다 — 오히려 과최적화를 쉽게 만드는 방향이다.
- **지표** 총수익률 · CAGR · 최대낙폭(MDD) · 샤프비율 · 승률 · Profit Factor, 그리고 KOSPI 대비 자산 추이 차트와 매수/매도 마커.
- **다루지 않는 것** 호가·체결(틱) 데이터가 백테스트 경로에 전혀 들어가지 않는다. 큐 위치·부분체결·지연·시장충격·자기영향 중 어느 것도 모델링하지 않고, 표본 외 검증 절차도 없다. 수수료 통화가 `CashAmount(fee, "USD")` 로 하드코딩돼 있는 점도 KRW 시장 백테스트로는 어색하다.

```
# backtester/kis_backtest/codegen/generator.py
@dataclass
class CodeGenConfig:
    """코드 생성 설정"""
    market: str = "krx"  # krx, us
    commission_rate: float = 0.00015  # 0.015%
    tax_rate: float = 0.002  # 0.2% (KRX 매도세)
    slippage: float = 0.0  # 슬리피지 (기본 0%)
    initial_capital: float = 100_000_000  # 1억원
```

```
# backtester/kis_backtest/providers/kis/data.py
        if resolution == Resolution.DAILY:
            return self._get_daily_bars(symbol, start, end)
        elif resolution == Resolution.MINUTE:
            return self._get_minute_bars(symbol, end)  # 분봉은 단일 날짜
        else:
            raise ValueError(f"지원하지 않는 해상도: {resolution}")
```

```
# backtester/scripts/setup_lean_data.sh
mkdir -p "$DATA_DIR/equity/krx/daily"
mkdir -p "$DATA_DIR/equity/usa/daily"
mkdir -p "$DATA_DIR/forex/fxcm/daily"
```
- **공개 코드** [koreainvestment/open-trading-api](https://github.com/koreainvestment/open-trading-api) — `backtester/`(Lean 연동 백테스트), `strategy_builder/`(전략 DSL·YAML), `examples_llm/`·`examples_user/`(API 예제), `postman/`. star 1,596 · fork 827 · **라이선스 파일 없음** / [koreainvestment/kis-ai-extensions](https://github.com/koreainvestment/kis-ai-extensions) — AI 에이전트용 스킬·MCP 확장. star 212 · 라이선스 명시 없음 / 엔진 본체는 [QuantConnect/Lean](https://github.com/QuantConnect/Lean) (Apache-2.0) 도커 이미지 `quantconnect/lean:latest`
- **한 줄** 국내 증권사가 공개한 유일한 백테스트 코드지만, 그 백테스트는 **일봉 종가 재생**이다 — 틱 백테스트의 선례가 아니라 그 부재의 증거다.

### 키움증권 공식 REST API 저장소 — 틱차트는 주지만 백테스터는 없다

- **venue** 공식 오픈소스 (GitHub) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [GitHub](https://github.com/Kiwoom-Securities/Kiwoom-REST-API) · [웹](https://openapi.kiwoom.com/)
- **데이터** REST 337개 API 스펙 + 파이썬 예제 362개(OAuth 2 · 국내주식 226 · 미국주식 134) + WebSocket 31개 + Postman 컬렉션 612 요청. 차트 API 28종 중 **틱차트가 실재한다**: `ka10079` 주식틱차트조회요청, 업종틱차트, 금현물틱차트.
- **설계** 백테스트 설계가 존재하지 않는다. 저장소 README 에 백테스트·시뮬레이션·성과지표에 대한 언급이 한 줄도 없고, 폴더 구조에도 그런 모듈이 없다.
- **검증** 해당 없음.
- **다루지 않는 것** 과거 **호가창** 이력을 주지 않는다. 틱차트도 봉으로 집계된 형태라 매수주도/매도주도 구분, 잔량, 큐 위치를 복원할 수 없다. 즉 이 API 만으로는 L3 이상의 백테스트를 만들 수 없고, 실시간 WebSocket 을 직접 수집해 쌓는 수밖에 없다.
- **공개 코드** [Kiwoom-Securities/Kiwoom-REST-API](https://github.com/Kiwoom-Securities/Kiwoom-REST-API) — 공식 REST/WebSocket 런타임 · 예제 362개 · Postman · MCP. `LICENSE.md` 존재(내용 미확인) / 커뮤니티: [breadum/kiwoom](https://github.com/breadum/kiwoom) (구 OpenAPI+ 파이썬 래퍼), [dongbin300/KiwoomRestApi.Net](https://github.com/dongbin300/KiwoomRestApi.Net) (.NET 래퍼)
- **한 줄** 국내에서 과거 틱을 REST 로 주는 드문 창구지만, 그 틱은 호가 없는 봉이고 백테스터는 애초에 없다.

### NH투자증권 NAMUH PLUG (PLUG-OpenAPI) — 2026년 8월 출시, SDK·MCP 뿐

- **venue** 공식 오픈소스 (GitHub, 2026-08~) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [GitHub](https://github.com/PLUG-OpenAPI) · [웹](https://www.nhplug.com/intro)
- **데이터** 포털이 내세우는 서비스는 실시간 시세 · 통합 계좌관리 · 투자정보 · 주문 · 차트데이터. "정교한 전략 검증과 데이터 시각화를 위한 시계열 데이터" 라고만 적혀 있고 해상도·기간은 인트로 페이지에 명시가 없다.
- **설계** 없음.
- **검증** 없음.
- **다루지 않는 것** 백테스트 엔진, 체결 시뮬레이션, 성과 지표 어느 것도 공개하지 않는다. 과거 호가·체결 데이터 제공 여부도 인트로 페이지에서 확인되지 않는다.
- **공개 코드** [PLUG-OpenAPI/nhplug-sdk](https://github.com/PLUG-OpenAPI/nhplug-sdk) — 파이썬 SDK · 샘플 · 종목마스터 파서 (star 10, 2026-09-08) / [PLUG-OpenAPI/nhplug-mcp](https://github.com/PLUG-OpenAPI/nhplug-mcp) — MCP 서버 (TypeScript, star 11, 2026-09-08) / 커뮤니티(구 QV API): [bekker/qvopenapi-rs](https://github.com/bekker/qvopenapi-rs), [odumag99/pynamuh](https://github.com/odumag99/pynamuh)
- **한 줄** 국내 2위권 증권사의 2026년 신규 OpenAPI 도 백테스트는 공개하지 않았다 — 한투가 예외임을 확인시켜 준다.

### KRX · 코스콤 · 증권사 — 대한민국 틱데이터는 어디서 오는가 (그리고 왜 백테스트가 없는가)

- **venue** 1차 자료 조사 (공식 포털 · 로컬 저장소 실측) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [GitHub](https://github.com/sharebook-kr/pykrx) · [문서](https://datamall.koscom.co.kr/) · [웹](https://openapi.krx.co.kr/)
- **데이터** KRX Open API 31개 서비스 · 코스콤 데이터몰 · KIS Open API 국내주식 160개 · 키움 337개 API 를 대상으로 "과거 체결·호가를 어디까지 주는가" 를 확인
- **설계** 해당 없음
- **다루지 않는 것** "과거 호가창을 되감는다" 는 전제가 국내에서는 공개 경로로 성립하지 않는다. 그래서 국내 틱 백테스트 연구는 (a) 실시간 자가 수집, (b) 기관 내부 데이터, (c) 해외 데이터셋(LOBSTER·FI-2010) 세 갈래로 갈린다.
- **공개 코드** [sharebook-kr/pykrx](https://github.com/sharebook-kr/pykrx) — KRX 웹 스크래핑. **일별** 시세·투자자별 거래·공매도 등. 틱 아님 / [FinanceData/FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) — 일별 시세 통합 리더. 틱 아님 / [sharebook-kr/pyupbit](https://github.com/sharebook-kr/pyupbit) — 업비트(암호화폐) 체결·호가. 국내에서 **틱 원장을 무료로 받을 수 있는 사실상 유일한 시장**
- **한 줄** 국내에서 호가+체결 과거 원장은 공개 시장에 없다 — 그래서 국내 틱 백테스트 연구는 데이터를 직접 모으는 데서 시작한다.

### An Adaptive Dual-level Reinforcement Learning Approach for Optimal Trade Execution — KRX 밀리초 호가 원장으로 돌린 VWAP 추종 RL

- **venue** arXiv 2307.10649 (q-fin.CP, 2023-07-20) · Expert Systems with Applications 투고 · **해상도** 초~분봉 · **등급** L2 무영향 재생
- **링크** [arXiv](https://arxiv.org/abs/2307.10649) · [PDF](https://arxiv.org/pdf/2307.10649)
- **데이터** "Our millisecond trade and limit order book (LOB) tick data is from the Korea Exchange (KRX)." 삼성전자·SK하이닉스·기아·POSCO홀딩스 4종목, 2021-01-01 ~ 2021-12-31. 학습 1/1~9/30, 테스트 10/1~12/31. 원자료를 5초 구간으로 묶어 **구간의 마지막 LOB 스냅샷**으로 상태를 만들고, 5초 VWAP·체결량으로 일간 VWAP 을 계산
- **설계** **무영향 재생**. 에이전트가 5초마다 정해진 수량을 시장에 내보내고, 그 시각의 체결가로 평균 매입단가를 누적한다. 하루가 끝나면 실제 일간 VWAP 과 비교한다.
- **체결** 즉시 전량 체결. 논문이 가정을 세 줄로 명시한다 — "1. We assume that the actions taken by our model only affect a temporary market, and that the market will recover to the equilibrium level at the next time step. 2. Commissions and exchange fees are ignored. 3. Our model's orders are traded immediately without order arrival delays." 지정가 대기·부분체결·큐 위치는 없다.
- **비용** **없다.** 수수료·거래소 수수료를 명시적으로 무시한다. KRX 매도 증권거래세(0.2%)도 당연히 반영되지 않는다. 다만 매수 집행 문제라 거래세는 원래 안 걸린다.
- **지연** 없다(가정 3에서 명시적으로 배제).
- **자기영향** 일시적 충격만 가정하고 다음 스텝에 회복한다고 본다. 영구 충격·타 참가자 반응은 없다. 논문은 이를 "we consider relatively small total orders in comparison to the daily total market volumes" 로 정당화한다.
- **검증** **단일 시간 분할**. 학습 1~9월, 테스트 10~12월. 워크포워드·롤링 재학습·다중 시드 보고는 없다. 종목 4개에 걸친 robustness 는 있다.
- **지표** VAA(VWAP Approximation Accuracy) = (MP_day − VWAP_day) / VWAP_day. 0 에 가까울수록 좋다. 평균·표준편차와 [0, 10bps] 구간 비율을 보고한다.
- **다루지 않는 것** 밀리초 원장을 확보하고도 백테스트는 5초 격자에서 돌아간다. 즉 데이터의 해상도가 백테스트의 사실성으로 이어지지 않았다. 체결 큐·부분체결·수수료·지연이 모두 빠져 있어, VAA 가 10bps 안이라는 결과가 실제 체결 후에도 유지되는지는 이 실험이 답하지 않는다.

```
As in previous works Nevmyvaka et al. (2006), Hendricks & Wilcox (2014), Ning et al. (2021), and Lin & Beling (2020), we make the following assumptions in our experiments.
1. We assume that the actions taken by our model only affect a temporary market, and that the market will recover to the equilibrium level at the next time step.
2. Commissions and exchange fees are ignored.
3. Our model's orders are traded immediately without order arrival delays.
```

```
Our millisecond trade and limit order book (LOB) tick data is from the Korea Exchange (KRX). We use the daily trade and LOB data of Samsung Electronics (SE), SK Hynix (SH), Kia Corporation (KC) and POSCO Holdings Inc (PH) from January 1st, 2021 to December 31st, 2021. … We divide the data into two sets, using January 1st to September 30th as training data and October 1st to December 31st as test data.
```

```
The millisecond raw data is preprocessed by first dividing them into groups of 5 seconds, and extracting data that represents each 5-second interval. We take the last LOB data of the 5-second interval to construct the MDP state and the 5-second VWAP and total traded volume for calculating the daily VWAP.
```
- **공개 코드** **공개 코드 없음.** 논문 본문·arXiv 초록 어디에도 저장소 링크가 없다. KRX 원자료도 재배포 불가
- **한 줄** KRX 밀리초 호가 원장을 쓴 대표 논문이지만, 백테스트는 5초 격자 · 즉시 체결 · 무비용 — 데이터 해상도와 백테스트 사실성은 별개임을 보여 준다.

### 토스증권 Open API (2026-05)

- **venue** 공식 REST API (개발자 포털) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [웹](https://corp.tossinvest.com/ko/open-api)
- **설계** 2026년 5월 사전신청을 시작한 개인 투자자용 REST OpenAPI. 서버 `https://openapi.tossinvest.com`. 심사·승인 대기 없이 설정 화면에서 client_id·client_secret 을 즉시 발급받는 점이 특징이다.
- **한 줄** 발급 문턱을 가장 낮춘 국내 API지만, 백테스트는 여전히 사용자 몫이다.

### LS증권 · DB증권 · 대신증권 · 삼성선물 · 미래에셋

- **venue** 공식 API 포털 (공식 GitHub 없음) · **해상도** 해당 없음 · **등급** L0 시뮬레이션 없음
- **링크** [문서](https://openapi.dbsec.co.kr/testbed-sample) · [웹](https://openapi.ls-sec.co.kr/)
- **설계** LS증권(구 이베스트)은 DLL 기반 XingAPI 에서 HTTP 기반 OPEN API 로 이행 중이며 공식 포털(openapi.ls-sec.co.kr)을 운영한다. 대신증권 CREON Plus 는 여전히 32bit 윈도우 COM 이다. 미래에셋은 HTS Kairos 에 붙이는 Plug-in(AnyLink) 방식이고, 삼성증권은 기관 대상 유료 API 로 알려져 있다. DB증권과 삼성선물은 자체 개발자 포털에 파이썬 샘플을 둔다. **공식 저장소·백테스터는 모두 없고, GitHub 에 있는 것은 개인이 만든 래퍼다** (`xorrhks0216/LsApiHelper`, `teranum/ls-openapi-samples`, `gyusu/Creon-Datareader`, `hspan/creon` 등).
- **한 줄** 국내 중견 증권사 라인은 공식 GitHub 조차 없고, 생태계를 개인 개발자 래퍼가 떠받치고 있다.

### 국내 커뮤니티 오픈소스 — pykrx · FinanceDataReader · pyupbit

- **venue** 커뮤니티 오픈소스 · **해상도** 일봉 이상 · **등급** L2 무영향 재생
- **링크** [GitHub](https://github.com/sharebook-kr/pykrx) · [GitHub](https://github.com/FinanceData/FinanceDataReader) · [GitHub](https://github.com/sharebook-kr/pyupbit)
- **설계** `pykrx` 는 KRX 웹을 긁어 일별 시세·투자자별 거래·공매도를 준다. `FinanceDataReader` 는 국내외 일별 시세를 통합한다. 둘 다 **틱이 아니다.** 국내에서 틱 원장을 무료로 받을 수 있는 사실상 유일한 시장은 암호화폐이고, `pyupbit` 가 그 창구다 — 우리 `tick_collector_pyupbit` 이 여기 있는 이유다.
- **한 줄** 국내 주식 커뮤니티 백테스트의 데이터 상한이 '일봉' 인 이유는 도구가 아니라 데이터 유통 구조에 있다.

## 우리 코드와의 대조

우리 코드와의 대조 — /home/dgu/tick 에서 가장 최근에 돌린 백테스트
    proj_claude_tick_finance/framework · 정본 프로필 CANONICAL_QUEUE_V9 · 마지막 실행 흔적 runs/sandbox12_experiments_20260909 (2026-09-09)

  무엇을 읽었나. framework/fill.py(체결 판정) ·
  framework/canonical.py(정본 실행 프로필) · framework/ledger.py(원장·상태) ·
  framework/portfolio.py(공용 현금 · 부분체결) · framework/data.py(틱 로더) ·
  framework/config.py(상수) · framework/modules/backtest.py(요약 산출) 를 직접 읽고,
  실행 산출물 runs/sandbox12_experiments_20260909/allstock_W1~W4 와 allstock_oos7_20260906 의
  00_research_context.json 에서 실제 날짜·종목·제약을 확인했다.

  우리 백테스트의 실체 — 한 화면 요약
  
    데이터무엇을 되감는가KRX 원장 parquet(~/tick/tickdata_krx/<YYYYMMDD>/<종목>_*.parquet).
      한 파일에 data_type==12(호가 10단계 · 가격/잔량)와 data_type==11(체결 · askbid_type 1=매도주도 / 2=매수주도)가 섞여 있고,
      로더가 호가를 시간 격자로 삼고 체결을 그 격자에 누적한다. 타임스탬프는 HHMMSSffffff 마이크로초.
      세션은 090000000000~152000000000 — 종가 단일가는 연속 체결로 계산하지 않는다.
      전종목 2,656~2,661개, 2026-03-16 ~ 04-23 약 6주.
    결정언제 사는가decision.basis = EVERY_TICK. 간격을 두지 않고 매 틱 조건을 본다.
      포지션 보유 중의 신호 틱은 새 결정으로 세지 않는다(blocked_while_holding) — 성과가 신호 지속시간에 좌우되지 않게.
    진입 체결무엇이 체결을 결정하는가BID1 지정가 + 큐 위치 모델(BEST_QUOTE_QUEUE).
      게시 시점에 remaining = qfrac × 최우선잔량 + 주문수량 만큼 앞에 서 있다고 본다. 이것을 줄이는 것은 두 가지뿐이다 —
      (a) 내 가격에 도달한 반대편 주도 체결(지정가 매수면 sell_volume/sell_min_price 만),
      (b) 체결로 설명되지 않는 잔량 감소 × unexplained_fill_fraction.
      가격이 움직였다는 사실만으로는 체결되지 않는다(price_move_alone_does_not_fill).
      최대 10초 / 100틱 대기, 시계가 우선.
      두 파라미터가 서로 반대 방향으로 작동한다. queue_fraction = 1.0 은 게시 시점 표시 잔량 전부 뒤에 선다는 뜻이라
      가장 보수적이고(FeatureProfile 쪽은 0.20 을 쓴다), unexplained_fill_fraction = 1.0 은 설명되지 않은 잔량 감소를
      전부 내 앞의 취소로 인정한다는 뜻이라 가장 낙관적이다. 순 효과는 측정되지 않았다.
    hold-through호가가 변하면?취소하지 않고 게시가에서 계속 기다린다.
      코드 주석이 이유를 실측으로 적어 뒀다 — 취소 모델은 “가격이 불리하게 움직이는 순간 주문을 빼면 역선택을 공짜로 회피하게 되어 net 이 과대평가된다”.
      취소 모델의 체결률 중앙값(~9%)이 실측 패시브 체결률(~45%)과 5배 어긋났고, hold-through 로 바꾸자 기존 “수익” 61건 중 27건만 살아남았다.
    청산어떻게 나가는가정본 V9: gross −120bps 시장가 손절(BID1 시장가, 무조건 나감) ·
      진입 후 최고 ASK1 대비 −30bps 에서 ASK1 추적 지정가(최우선 호가가 바뀌면 취소 후 재게시) ·
      지정가 60초 대기 후 미체결이면 BID1 시장가 · 최대 보유 900초 후에도 미청산이면 BID1 시장가.
      “안 팔린 거래를 통계에서 빼면 나쁜 거래만 사라진다”가 강제 청산의 이유다.
      손절선을 gross 기준으로 잡는 이유도 명시돼 있다 — net 으로 잡으면 BID1 에 사는 순간 이미 −수수료에서 시작한다.
    비용무엇을 빼는가스프레드는 빼지 않는다(spread_deduction_bps = 0.0).
      체결가가 실제 BID/ASK 에서 나오므로 이미 손익에 들어 있고, 또 빼면 이중 계상이기 때문이다.
      명시비용만 FEE_BPS = 23.0(왕복 수수료·세금)을 뺀다. 그리고 spread_accounting_audit 가
      모든 체결 행에서 gross_bps − net_bps == 23.0 항등식이 성립하는지 매번 검산한다 — 어긋나면 COST_DUPLICATION 으로 실패한다.
    상태무엇을 분모에 넣는가FILLED / UNFILLED / CENSORED(평가 창을 관측할 수 없어 채점 불가) / BLOCKED.
      decisions = FILLED + UNFILLED + CENSORED, scorable = FILLED + UNFILLED.
      생존분석의 중도절단 개념을 그대로 가져온 설계다.
    원장 기록결정 하나에 무엇을 남기는가체결 여부와 손익만 남기지 않는다.
      결정마다 entry_tick(게시) · fill_tick(체결) · exit_tick 을 따로 적고,
      게시–체결 사이의 역선택을 fill_slippage_bps · bid_move_bps · ask_move_bps 로 분해한다.
      여기에 entry_spread_bps · exit_spread_bps · tick_size_bps(bps 로 적힌 손절·비용이 몇 틱인지 읽는 기준) ·
      max_favorable_gross_bps / max_adverse_gross_bps(MFE/MAE) ·
      미체결의 원인을 큐 길이와 시간으로 가르는 큐 상태 · 신호가 체결 시점까지 살아 있었는지(entry_signal_persisted_until_fill)까지 남는다.
    계좌돈이 모자라면?portfolio.py 가 종목 간 공용 현금으로 실제 주문·부분 체결을 시간순(heapq)으로 처리하고,
      호가 갱신마다 (time, cash, reserved, available, equity) 를 쌓아 equity curve 를 만든다.
      별도로 amount.py 가 거래당 100만원 고정·비복리 환산으로 일별 손익과 Sharpe = mean/std × √252 를 낸다.
    검증표본을 어떻게 나누는가롤링 워크포워드. 창마다 발굴 1일 → 검증 1일 → OOS 5일(또는 7일)이고 날짜가 서로 겹치지 않는다.
      W1 0316 / W2 0325 / W3 0403 / W4 0415 네 창 + oos7 세 창.
      제약이 계약으로 박혀 있다 — DECISION_TIME_CAUSAL_INPUTS_ONLY · ACTUAL_BID_ASK_AND_EXPLICIT_23BPS ·
      DISJOINT_DISCOVERY_VALIDATION_FINAL_DATES · REFINEMENT_USES_DISCOVERY_ONLY ·
      FULL_STRATEGY_FROZEN_BEFORE_VALIDATION · LONG_ONLY · CANONICAL_ENTRY_AND_EXIT_EXECUTION_FIXED.
      탐색은 전략당 Optuna 18 trial, 시드 고정.
    재현같은 결과가 다시 나오는가실행마다 runtime_identity 에 git HEAD · 브랜치 · dirty 여부 ·
      status_sha256 · 파이썬/numpy/optuna/pandas/pyarrow 버전 · 소스 파일별 sha256 을 박는다.
      프로필도 profile_sha256, 파라미터 락도 plan_sha256, 수익 목표도 profit_target_sha256 로 지문을 남긴다.
      여기에 사전등록(prereg.sh freeze → 주장 직전 audit)이 붙는다.
    지표무엇으로 좋다고 하는가분모가 다른 수를 나란히 낸다 — net_bps_total ·
      net_bps_per_decision(scorable) · net_bps_per_fill · net_bps_per_day · fill_rate.
      그리고 (종목, 날) 블록 분해로 총합 하나에 가려지는 쏠림을 드러낸다.
      분모가 이름에 안 드러나는 bps_per_attempt 류는 FORBIDDEN_METRIC_NAMES 로 금지돼 있다.
  

  
그림 2 · 큐 모델의 핵심 분기 — “설명되지 않은 잔량 감소” 를 누구 앞의 취소로 볼 것인가

  
    
      .h{font:800 13px system-ui,-apple-system,"Noto Sans KR",sans-serif;fill:#0f172a}
      .t{font:400 11.5px system-ui,-apple-system,"Noto Sans KR",sans-serif;fill:#475569}
      .m{font:600 11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#0f172a}
      .k{font:700 11px system-ui,-apple-system,"Noto Sans KR",sans-serif;fill:#fff}
      .cap{font:700 11.5px system-ui,-apple-system,"Noto Sans KR",sans-serif;fill:#334155}
    
    
      
    
  

  
  상황 — BID1 에 지정가 매수를 걸었다
  
  
  
  내 앞 물량 (ahead)
  내
  뒤 물량
  게시 시점 표시 잔량 전체 = prev_depth
  우리 설정 queue_fraction = 1.0 → 표시 잔량 전부 뒤에 선다 (가장 보수적)

  
  한 틱 뒤 — 잔량이 줄었다. 그런데 그중 일부는 체결로 설명되지 않는다
  
  
  
  
  남은 잔량
  체결
  ???
  체결(eligible) = 내 가격에 도달한 반대편 주도 체결 — 직접 관측된다
  ??? = prev_depth − depth − eligible — 취소인지 미관측 체결인지 가릴 수 없다

  
  이 “???” 를 어디에 배분하느냐가 큐 모델을 가른다

  
  
    
    
    전부 내 뒤에서 취소됐다 (보수적)
    hftbacktest RiskAdverseQueueModel
    front = min(front, new_qty)
    내 앞 물량은 체결로만 줄어든다.
    전체 잔량이 내 앞 물량보다 작아질 때만
    clamp 된다 → 체결이 가장 늦게 난다
  

  
    
    
    앞/뒤 물량 비율로 나눈다 (중간)
    hftbacktest ProbQueueModel
    prob = f(back)/(f(back)+f(front))
    front −= (1−prob)·chg
    f 는 x^n 또는 log(1+x).
    내가 앞쪽일수록 취소가 앞에서 났다고 본다
  

  
    
    
    전부 내 앞에서 취소됐다 (낙관적)
    우리 코드 framework/fill.py
    ahead −= unexplained · chg
    unexplained = 1.0
    = ProbQueueModel 에서 1−prob ≡ 1,
    즉 이 축에서 가장 낙관적인 끝
  

  
  보수 ←— 같은 데이터, 같은 상황. 배분 가정 하나가 체결률을 정한다 —→ 낙관

주문 ID 가 없는 L2 원장에서는 잔량 감소를 체결과 취소로 직접 가를 수 없다. 그래서 모든 큐 모델은 이 배분을 가정으로 정한다.
우리 코드는 이 축에서 가장 낙관적인 끝(unexplained_fill_fraction = 1.0)에 있고, 동시에 대기 위치 축에서는 가장 보수적인 끝(queue_fraction = 1.0,
표시 잔량 전부 뒤)에 있다 — 두 가정의 순 효과는 아직 측정되지 않았다.


  실제 산출물 한 조각 — 마지막 워크포워드 창의 OOS 유닛 하나
  runs/sandbox12_experiments_20260909/allstock_W4_0415/06_final/final_artifact.json 의 유닛
  S1_FLOW_LIQ__D1(OOS 5일, 전종목). 이 리포트가 다루는 개념들이 실제로 어떤 숫자로 나오는지 보여 주는 예다.
  항목값읽는 법
    신호 틱 / 차단 틱36,401 / 12,660포지션 보유 중의 신호는 새 결정으로 세지 않는다
    결정 (decisions)23,741= FILLED 4,020 + UNFILLED 19,539 + CENSORED 182
    채점 가능 (scorable)23,559CENSORED 182 를 뺀 값 — 분모가 둘로 갈린다
    체결률16.93%BID1 지정가 + hold-through + 큐 모델의 결과. 전량 체결을 가정한 문헌과 비교할 지점
    gross 합92,259.96 bps비용 이전. 체결 4,020건
    체결당 평균 gross22.95 bps수수료 문턱 23.0 bps 에 0.05bps 모자란다
    net 합−200.04 bpsgross 전부가 사실상 수수료였다는 뜻이다
    비용 회계 감사gross_minus_net_max_gap_bps = 0.0, 검사한 체결 4,020건4,020건 전부에서 gross − net == 23.0 이 정확히 성립 — 비용 이중계상 없음
    블록 분해(종목,날) 블록 2,240개 · 양수 비율 40.9% · 중앙값 −14.70bps · 상위 3블록 제외 net −3,952bps · 최대 기여 종목 073570 (절대 기여 0.97%)총합 하나에 가려지는 쏠림을 드러낸다. 여기서는 쏠림이 없다
    수익 목표 판정success: false실패를 실패로 기록한다
    경로 정합PARTIAL (6개 중 3개 적중) · 낙관 편향: 예측 유리 3 vs 실제 유리 2사전등록한 실행 경로 예측(체결률·슬리피지 부호 등)의 적중을 따로 기록. 선택에는 쓰지 않는다
    지문strategy_sha256 e23bf94d… · contract_sha256 482c80b2… · frozen_strategy_sha256 de4dbc57… · execution_profile CANONICAL_QUEUE_V9 / d0b91a03 · strategy_changed_after_freeze: false · validation_used_for_selection: false동결 이후 전략이 바뀌지 않았고 검증 구간을 선택에 쓰지 않았음을 산출물이 스스로 증명한다
  
  이 한 유닛이 말해 주는 것. 체결당 평균 gross 22.95bps 가 왕복 수수료 23bps 에 0.05bps 모자라 전체가 −200bps 로 끝났다.
  조사한 문헌 상당수가 이 지점을 아예 만들지 않는다 — 수수료를 넣지 않거나(KRX RL 최적실행 · Cartea–Donnelly–Jaimungal · JAX-LOB),
  전량 체결을 가정해 체결률 16.93%가 아니라 100%를 쓰거나(Guéant 계열), 봉 종가에 전량 체결시킨다(한투 backtester · Hummingbot).
  그 셋 중 하나만 해도 이 유닛은 “성공”으로 보고됐을 것이다.

  같은 것 — 조사한 최고 수준 백테스트와 우리가 공유하는 것
  
    항목문헌·도구가 하는 것우리 코드

    큐 위치 모델
    hftbacktest 의 ProbQueueModel 계열 — 내 앞 물량을 추정하고, 그 물량이 줄어야 체결시킨다.
      실무 HFT 백테스트가 L2 와 갈라지는 지점.
    있다. 게시 시점 최우선 잔량 뒤에 선다고 가정하고, 반대편 주도 체결과 설명되지 않은 잔량 감소로만 큐를 깎는다.

    매수/매도 주도 구분
    Lee–Ready 이후 미시구조 실증의 기본. 지정가 매수의 큐를 소진시키는 것은 매도 주도 체결뿐이다.
    있다. 로더가 askbid_type 으로 sell_volume/sell_min_price 를 따로 모으고,
      체결 방향과 가격 도달을 함께 본다.

    실호가 체결 · 이중계상 금지
    실행 문헌의 원칙 — 스프레드는 체결가에 이미 들어 있으므로 따로 빼면 두 번 문다.
    있다. 그것도 서술이 아니라 감사 코드로: gross − net == FEE_BPS 항등식을 매 실행 검산한다.

    강제 청산 · 생존편향 차단
    “열린 포지션을 통계에서 빼면 나쁜 거래만 사라진다”는 백테스트 위생의 기본.
    있다. 900초 뒤 무조건 시장가 청산 + CENSORED 상태로 관측 불가를 따로 표시.

    표본 외 분리
    Chen & Velikov(JFQA 2023) "표본 내에서만 최적화 → 출판 이후·2005년 이후에서만 평가",
      Cartea–Donnelly–Jaimungal(2018) 상반기 적합/하반기 시험, Cont–Kukanov(2017) 1~3월 보정/4월 평가.
    있다. 롤링 워크포워드 4창(+7일 OOS 3창), 발굴 1일 → 검증 1일 → OOS 5~7일,
      그리고 날짜 분리가 서술이 아니라 계약이다(DISJOINT_DISCOVERY_VALIDATION_FINAL_DATES).

    사전등록 문서
    임상·심리학에서는 표준이지만, 조사한 금융·실행·AI 문헌 어디에서도 사전등록 문서를 본 적이 없다.
      Chen & Velikov(JFQA 2023)의 "표본 내에서만 최적화한다"가 가장 가까운데, 그것도 논문 본문의 서술이다.
    우리만 한다. ChatReport/07_executable_spec/*/preregistration.md에
      freeze 커밋 해시 · 코드 sha256 · H0/H1 · 문턱 격자 · 종목 목록 · 발굴/검증/OOS 날짜 · 시드까지 포함한 실행 명령이 결과를 보기 전에 고정된다.
      게다가 "이 판정규칙은 20260330~0401 결과를 본 뒤에 정했다. 그래서 그 날짜는 여기서 쓰지 않는다"처럼
      발굴 구간에서 본 것을 스스로 공개한다. 실행 경로에 대한 예측(fill_rate·fill_slippage_bps 부호 등)까지 미리 적는다.

    재현 지문
    ACM/NeurIPS 재현성 체크리스트가 요구하는 환경 고정.
    있다(오히려 더 강함). git HEAD·dirty·소스 파일별 sha256·의존성 버전까지 산출물에 박는다.
  

  다른 것 — 우리가 안 하는 것과, 우리만 하는 것
  
    항목문헌·도구가 하는 것우리 코드

    지연(latency)
    hftbacktest 는 피드 지연과 주문 지연을 분리해 모델링하고(IntpOrderLatency 로 실측 왕복 지연을 보간),
      NautilusTrader 도 LatencyModel 로 insert/update/delete 지연을 따로 준다.
    없다. ledger.py가 post = int(record.entry_tick) — 신호가 뜬 그 틱에 즉시 게시한다.
      왕복 지연이 수십 ms 만 돼도 BID1 큐 순번이 달라지므로 우리 체결률은 낙관 쪽으로 치우쳐 있을 수 있다.
      게다가 우리 원장의 local_time은 HHMMSSffffff 단일 시각이라
      (Databento 의 ts_recv/ts_in_delta, Tardis 의 timestamp/local_timestamp 같은
      거래소-수신 시각 쌍이 없다) 지연을 데이터에서 뽑을 수 없다 — 상수 가정으로 시작할 수밖에 없다.

    자기영향(self-impact)
    ABIDES 계열은 다중 에이전트 상호작용으로 내 주문이 남의 행동을 바꾸게 한다. 재생 기반 백테스트의 구조적 한계에 대한 답이다.
    없다. 순수 재생이다. 다만 1주 단위 소액(거래당 100만원) 가정이라 영향이 작다는 변론은 가능하다.
      기관 규모로 키우면 이 가정이 먼저 깨진다.

    주문 단위 원장(L3)
    LOBSTER·NASDAQ ITCH 는 주문 ID 가 붙어 추가·취소·체결을 직접 구분할 수 있다.
    없다. 프로필이 스스로 한계로 적어 뒀다 — “주문 추가·취소·수정이 이 데이터에 없다. 잔량 감소를 체결과 취소로 직접 가를 수 없다”.
      그래서 unexplained_fill_fraction 이라는 튜닝 가능한 가정이 남는다. 이 값의 민감도를 보고하지 않으면 약점이다.

    다중검정 보정
    Deflated Sharpe Ratio·PBO(CSCV)·Romano–Wolf 는 “몇 번 시도했는가”를 성과에서 깎아 낸다.
    없다. Optuna 18 trial × 전략 수 × 정제 라운드 × 창 4개 — 실제 시행 횟수는 세고 있지만
      그 수를 성과 판정에 반영하는 통계량이 없다. 날짜 분리와 사전등록으로 막고 있으나, 시행 횟수를 넣은 DSR 은 추가 여지다.

    목적함수
    대개 Sharpe·IS(implementation shortfall)·VWAP 편차처럼 비율이거나 벤치마크 대비 값.
    다르다. net_bps_total 은 합이라 거래를 줄이면 좋아진다 —
      research-log 도 이 문제를 열린 항목으로 적어 뒀다. 분모가 다른 지표를 나란히 내는 것으로 완화하지만, 선택 기준 자체는 합이다.

    중도절단(CENSORED)
    조사한 어떤 백테스트 엔진·논문에서도 “평가 창을 관측할 수 없어 채점 불가” 를 별도 상태로 분리하는 것을 보지 못했다.
      보통은 조용히 버리거나 마지막 값으로 채운다.
    우리만 한다. 생존분석의 중도절단을 백테스트 상태 기계에 넣어 decisions 와 scorable 의 분모를 갈랐다.

    체결 가정의 실측 교정
    큐 모델의 파라미터를 실측 체결률로 맞추는 절차를 명시한 문헌은 드물다.
    우리만 한다. 취소 모델의 체결률 중앙값(~9%) vs 실측 패시브 체결률(~45%) 을 비교해
      모델을 실측에 맞춰 갈아엎은 기록이 코드 주석에 남아 있다.

    게시–체결 역선택 분해
    실행 문헌은 이 개념(adverse selection / fill slippage)을 정의하지만, 백테스트 엔진이 결정마다 이것을 열로 남기는 경우는 못 봤다.
      보통 체결가와 손익까지만 남는다.
    우리만 한다. fill_slippage_bps · bid_move_bps · ask_move_bps 로
      “게시한 뒤 시장이 어디로 갔길래 체결됐는가” 를 결정 단위로 되짚을 수 있다.
      research-log 가 “BID1 지정가는 시장이 내려와야 체결되므로 fill_slippage_bps 의 POSITIVE 가 구조적으로 없다(162건 중 0건)” 를
      찾아낸 것도 이 열 덕분이다.

    실행 프로필의 지문화
    대부분 설정 파일로 둔다.
    우리만 한다. 실행 프로필 자체를 sha256 으로 굳히고, 프로필이 바뀌면 기존 실험을 이어 돌리지 않는다.
      hidden_constants() 로 “결과를 바꾸는데 계약 밖에 있던 상수” 목록을 명시적으로 관리한다.
  

  한 줄 대조. 우리 백테스트는 L4(재생 + 큐 위치) 에 서 있다 —
  조사한 학술 논문 대부분(L0~L2)보다 위이고, hftbacktest·NautilusTrader 같은 실무 엔진과 같은 층이되
  지연 모델이 없다는 점에서 그 아래다. 그리고 국내 증권사가 공개한 유일한 백테스트(한투 backtester, L2 일봉)와는 층이 두 개 다르다.
  우리가 문헌보다 확실히 앞서는 것은 체결 모델이 아니라 검증 위생(날짜 분리 계약 · 사전등록 · 소스 해시 · 중도절단 · 비용 이중계상 감사)이다.

  ✍︎ 이 조사가 우리 코드에 남기는 세 가지 숙제
    지연을 넣어 보고, 넣기 전과 후의 체결률을 같이 보고한다. 조사한 실무 엔진이 전부 하는 것이고, 우리에게만 없다.
      상수 하나(왕복 ms)를 프로필에 넣고 LimitOrder.post 시각을 미루는 것으로 시작할 수 있다.
    unexplained_fill_fraction 의 민감도 곡선을 낸다. 지금 1.0 인 이 값은
      hftbacktest 의 확률 큐 모델에서 1 − prob ≡ 1, 즉 설명되지 않은 잔량 감소를 전부 내 앞의 취소로 보는 가장 낙관적인 설정이다
      (hftbacktest 의 RiskAdverseQueueModel 은 정반대로 전부 내 뒤로 보고, ProbQueueModel 은 앞/뒤 물량 비율로 그 사이를 정한다).
      반대로 queue_fraction = 1.0 은 가장 보수적이다. 두 낙관·보수가 상쇄되는지 재지 않으면
      주문 ID 없는 L2 원장을 쓰는 한 이것이 우리 백테스트의 가장 큰 미확인 가정으로 남는다.
    강한 신호에서 체결률이 떨어지는지 먼저 데이터로 확인한다. Noble–Rosenbaum–Souilmi(2026)의
      “신호가 셀수록 경쟁자도 몰리므로 체결확률이 낮아야 한다” 는 가정을 넣기 전에,
      우리 원장의 결정 시점 신호 값 × fill_rate 교차로 실제로 그런지 바로 잴 수 있다.
    시행 횟수를 성과에 반영한다. 이미 Optuna trial 수·정제 라운드·창 수를 전부 기록하고 있으므로,
      Deflated Sharpe Ratio 의 입력(N, 왜도, 첨도, 표본 길이)을 채우는 데 추가 실험이 필요 없다.

## 검증 한계

검증 한계 — 확인하지 못한 것과 확인 방법
  
    웹 검색 예산. 이 세션의 WebSearch 한도(200회)를 조사 중간에 소진했다. 그 뒤의 확인은 전부 WebFetch(직접 URL 열람)로만 했다.
      그래서 “검색해서 더 찾았어야 할 것”이 남아 있을 수 있고, 특히 국내 증권사 쪽은 검색이 아니라 포털·저장소를 직접 여는 방식으로만 좁혔다.
    GitHub API 호출 한도. 비인증 호출 한도에 걸려 일부 저장소의 star·라이선스·최근 커밋을 숫자로 확정하지 못했다.
      확정한 것만 숫자를 적었고(예: koreainvestment/open-trading-api star 1,596 · fork 827 · push 2026-08-26 · 라이선스 필드 null,
      nkaz001/hftbacktest star 4,642 · MIT), 나머지는 “미확인”으로 남겼다.
    유료·폐쇄 원문. JF·JFE·RFS·QJE 계열은 출판본이 유료라 NBER·SSRN·저자 홈페이지의 워킹페이퍼 판본으로 확인한 항목이 있다.
      워킹페이퍼와 출판본의 표·수치가 다를 수 있으므로, 인용할 때는 출판본을 다시 확인해야 한다.
    ACM/IEEE. dl.acm.org 는 종종 Cloudflare 로 막히고 IEEE Xplore 도 유료다. venue 는 arXiv comments · Crossref 메타데이터 ·
      학회 프로그램 페이지로 교차 확인했고, 확인하지 못한 것은 그대로 적었다.
    국내 증권사에서 확인 못 한 것. 코스콤 데이터몰이 “서비스 잠정 중단” 상태라 틱 상품의 라인업·가격을 확인하지 못했다.
      KRX 가 기관에 별도 판매하는 시장데이터의 해상도, 키움 ka10079 틱차트의 조회 가능 과거 기간,
      NH PLUG 의 차트데이터 해상도(로그인 뒤 문서)도 미확인이다.
    수치까지 확인하지 못한 것. Holden & Jacobsen(2014)의 편의 크기(유효스프레드 +54% 등)는
      Wiley·SSRN 403, Unpaywall is_oa: false, 저자 페이지 삭제로 모든 공개 경로에서 전문을 얻지 못했다 —
      초록에 있는 세 메커니즘과 Interpolated Time 기법만 인용 가능하고 숫자는 2차 출처 기준이다.
      Sullivan·Timmermann·White(1999)의 "7,846개 거래규칙" 도 원문에서 확인하지 못했다(DOI·venue·초록은 확인).
      Barndorff-Nielsen 외의 정제규칙 표(P1–P3/T1–T4/Q1–Q4)도 표 자체는 열지 못했다.
    미심사 프리프린트를 명시한다. 본문에서 인용한 것 중
      Nefedov(샤프 3.6배 부풀림, SSRN 7350238) · Boutgajouft(PBO 역방향, SSRN 7346738) ·
      Gençay(누출 오라클, arXiv 2608.27734) · Bunčić(VoC 반박, SSRN 5239006) 은
      동료심사를 거치지 않았고, SSRN 403 때문에 Crossref 등록 초록 또는 저자 홈페이지 요약만 읽었다.
      내용이 이 리포트의 논지와 정확히 맞물리지만 그만큼 가중치를 낮춰 읽어야 한다.
      Berk 의 VoC 코멘트는 초록조차 색인돼 있지 않아 논지를 옮기지 않았다.
    venue 표기를 바로잡은 것들. 조사 과정에서 흔한 오인용을 여럿 확인했다 —
      TransLOB 은 동료심사를 거친 적이 없고, DeepLOB-Seq2Seq/multi-horizon 도 게재 venue 가 없으며,
      TradingAgents 는 ICML 2025 워크숍 포스터이고 FinMem 은 AAAI Symposium Series 3쪽이다.
      Gort 외의 PDF 에 있는 "Copyright © 2022, AAAI" 는 LaTeX 클래스 보일러플레이트이지 채택 증거가 아니다.
      PBO 논문은 2017 이 아니라 2016년 발행이고 Crossref 가 권·호·페이지를 null 로 반환한다.
      IMM 은 IJCAI 2023 이 아니라 IEEE TETCI 2025 다.
    도구에 대한 경고. mlfinlab 은 오픈소스가 아니다 — 공개 저장소의
      cross_validation.py 는 272줄에 pass 문 10개로 모든 함수 본문이 스텁이고,
      LICENSE 는 독점 구독 계약(£100/월/사용자)이며, PyPI 에 없고, 2023-10 이후 방치돼 있으며,
      정작 쓰려는 PurgedKFold 에 미해결 누출 이슈 #295 가 재현 예제와 함께 열려 있다.
      purged CV·CPCV 의 참조 구현으로 인용하면 안 된다.
    백테스트 등급 판정의 한계. 사다리 등급은 논문·문서가 명시한 것만 근거로 매겼다.
      “안 적혀 있다 = 안 했다” 로 읽은 곳이 있고(특히 지연·큐), 부록이나 저장소 코드에는 있는데 본문에 없는 경우 과소평가됐을 수 있다.
      코드가 공개된 항목은 코드를 열어 확인했고, 그렇지 않은 항목은 “문서 기준”이라고 적었다.
    우리 코드 쪽. 소스를 읽어 확인한 것이지 재실행해 수치를 재현한 것은 아니다.
      본문에 인용한 실측 수치(취소 모델 체결률 중앙값 ~9% vs 실측 패시브 ~45%, hold-through 전환 뒤 61건 중 27건 생존)는
      framework/fill.py 의 모듈 주석에 적힌 기록을 그대로 옮긴 것이다.