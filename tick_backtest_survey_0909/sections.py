#!/usr/bin/env python3
"""manifest.json 의 텍스트 섹션(제목·리드·사다리 예시·판정·군 소개)을 갱신한다."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, "manifest.json")
m = json.load(open(M, encoding="utf-8"))

S = m["sections"]
S["title"] = "틱데이터 백테스트 심층분석 — 틱 원장으로 굴린 22편과 그것을 판정하는 44편"
S["h1"] = ("틱데이터 백테스트 심층분석 — 무엇이 체결을 결정하는가 "
           "(틱 원장으로 굴린 22편 · 해상도를 낮춘 대조군 30편 · 판정 도구 10편 · 실집행 기준선 4편)")
S["submeta"] = [
    "질문: 틱데이터를 쓴 연구들은 백테스트를 실제로 어떻게 했는가 — 무엇이 체결을 결정했는가",
    "항목마다 백테스트가 실제로 굴러간 해상도를 표시했다 (틱·메시지 / 초~분봉 / 일봉 이상 / 합성 / 실집행)",
    "분류 축: 백테스트 사실성 사다리 L0~L6 (체결을 무엇이 결정하는가)",
    "한국투자증권·NH·키움 등 국내 증권사의 백테스트 코드 공개 여부를 저장소 단위로 확인",
    "끝에 /home/dgu/tick 의 최신 백테스트 코드와 대조",
    "원문 PDF 전문 확인 + 저장소 소스 직접 열람 + GitHub API 로 star·라이선스 검증",
]
S["lead"] = """
<p class="lead"><b>질문은 하나다. “틱데이터로 백테스트했다” 는 문장이 실제로 무엇을 뜻하는가.</b>
조사해 보니 이 문장은 서로 두 층 이상 떨어진 작업들을 같은 이름으로 부르고 있었다.
한쪽 끝에는 밀리초 호가 원장을 확보하고도 5초 격자에서 즉시·무비용 체결을 가정한 논문이 있고,
다른 쪽 끝에는 내 주문 앞에 남은 물량을 세고 실측 왕복 지연으로 경주에서 지는 경우까지 넣은 엔진이 있다.
<b>데이터의 해상도와 백테스트의 사실성은 별개다.</b></p>
<div class="grid">
  <div class="card"><h4>조사 범위</h4><p>6갈래 병렬 조사로 150편 남짓한 논문·도구를 훑고, 이 질문에 직접 답하는
  <b>{deep}개 항목을 원문 정독 · 코드 실열람</b>으로 심층 분석하고 {brief}개는 요약 카드로 붙였다(항목 하나가 계보 전체를 묶은 경우도 있다).
  전통 금융(JF·JFE·RFS·QJE·JFQA) · 최적실행/마켓메이킹(QF·SIAM·JASA) · AI 학회(NeurIPS·ICML·ICLR·AAAI·KDD·ICAIF) ·
  백테스트 방법론 · 오픈소스 엔진 · 대한민국 증권사와 KRX 를 각각 따로 팠다.</p></div>
  <div class="card"><h4>검증 방식</h4><p>venue 는 Crossref API·OpenReview·PMLR·학회 프로시딩으로 교차 확인했고,
  코드는 GitHub REST API 로 star·라이선스·최근 커밋을 받아 적었다. 체결 로직은 홍보 문구가 아니라
  <b>소스를 열어</b> 확인했고(hftbacktest 의 <code>queue.rs</code>, Nautilus 의 <code>fill.rs</code>, Lean 의 <code>FillModel.cs</code>,
  한국투자증권의 <code>generator.py</code>), 인용은 전부 원문 그대로다.
  확인하지 못한 것은 <a href="#caveats">검증 한계</a>에 그대로 남겼다.</p></div>
  <div class="card"><h4>비교 대상</h4><p><code>/home/dgu/tick/proj_claude_tick_finance/framework</code> —
  KRX 전종목 2,661개 × 6주 틱 원장(10호가 + 체결, 마이크로초) 위에서 BID1 지정가를 큐 위치 모델로 체결시키는
  정본 프로필 <code>CANONICAL_QUEUE_V9</code>. 마지막 실행 흔적은 <code>runs/sandbox12_experiments_20260909</code>(2026-09-09).
  맨 아래 <a href="#local">우리 코드 비교</a>에서 무엇이 같고 무엇이 다른지 항목별로 대조한다.</p></div>
</div>

<div class="callout"><b>이 조사에 틱이 아닌 것이 왜 들어 있는가 — 66항목의 실제 해상도.</b>
항목마다 <b>백테스트가 실제로 굴러간 단위</b>로 해상도를 매겼다(데이터를 어디까지 확보했는지가 아니라).
결과는 이렇다 —
<span class="pill resTick">틱·메시지</span> <b>22</b> ·
<span class="pill resSub">초~분봉</span> <b>10</b> ·
<span class="pill resDay">일봉 이상</span> <b>15</b> ·
<span class="pill resSyn">합성·모형</span> <b>5</b> ·
<span class="pill resExec">실집행 기록</span> <b>4</b> ·
<span class="pill resNA">해당 없음</span> <b>10</b>.
<b>즉 틱 원장 위에서 실제로 굴린 것은 3분의 1(22편)뿐이다.</b> 나머지를 넣은 이유는 넷이다 —
<b>(가) 대조군</b>: 밀리초 원장을 확보하고도 5초 격자에서 굴린 논문(KRX RL 최적실행), 호가창을 상태로 쓰면서 분봉 종가에 체결시킨 논문(DeepScalper),
일봉·비용 0으로 초과수익을 보고하는 LLM 에이전트 — <b>이것들이 없으면 사다리의 층 차이를 보일 수 없다.</b>
<b>(나) 판정 도구</b>: DSR·PBO·CPCV 같은 방법론과 백테스트 엔진은 데이터를 쓰지 않지만 위 항목들을 채점하는 기준이다.
<b>(다) 실집행 기준선</b>: LSE 전체 메시지·AQR $1.7조 실체결처럼 시뮬레이션이 아닌 기록이 사다리의 상한을 정한다.
<b>(라) 국내 현황</b>: 한국투자증권 backtester 가 일봉이라는 사실 자체가 질문의 답이므로, 일봉이어도 넣어야 한다.
<b>틱 항목만 보려면</b> 요약표의 <span class="pill resTick">틱·메시지</span> 배지가 붙은 행을 따라가면 된다.</div>
"""

S["ladder_examples"] = {
    "L0": "Kirilenko 외 *Flash Crash* (JF 2017) · Hasbrouck–Saar (JFM 2013) · Cont–Kukanov–Stoikov OFI (JFE 2014) · Almgren–Chriss (2000) · Holden–Jacobsen (JF 2014) · **국내 증권사 API 저장소 전부**",
    "L1": "**Avellaneda–Stoikov** (QF 2008, 1,000회 몬테카를로 · 실데이터 0줄) · Bertsimas–Lo (1998) · Guilbaud–Pham (QF 2013, 실틱 보정 + 합성 평가) · mbt_gym",
    "L2": "Guéant–Lehalle–Fernandez-Tapia (한 종목 하루, 전량 체결) · **Cartea–Donnelly–Jaimungal 기본 설정**(체결확률 1) · **KRX RL 최적실행**(5초 격자, 수수료 무시) · **DeepLOB · FI-2010 계보**(체결 없이 F1) · OPD (분봉 평균가) · Ning 외 (중간가) · DeepScalper · **LLM 에이전트 4편**(일봉·비용 0) · Nagel (RFS 2012) · Chinco 외 (JF 2019) · Qlib · Lean · Freqtrade · Hummingbot · **한국투자증권 backtester(일봉)**",
    "L3": "**Budish–Cramton–Shim** (QJE 2015, 그림자 호가창) · **Cont–Kukanov** (QF 2017, 큐 소진 + 부분체결 + 리베이트) · Spooner 외 (AAMAS 2018)",
    "L4": "**hftbacktest** (큐 모델 4종 + 실측 보간 지연) · **NautilusTrader 1.223+** (`queue_position=True`) · **Nevmyvaka 외 ICML 2006** (우선순위 유지 + 테스트에서 가정 해제) · **Schnaubelt EJOR 2022** (큐 앞 물량 + 거래소별 maker/taker + 롤링 전진검증) · **Moallemi–Yuan** (MBO 인공주문) · **Huang–Lehalle–Rosenbaum** (큐 반응 모형) · **Noble–Rosenbaum–Souilmi** (2026, 경주 조건부 체결 + 자기영향 절제) · JAX-LOB · **우리 코드(지연 없음)**",
    "L5": "**ABIDES / ABIDES-Gym** (다중 에이전트, 자기영향) · Balch 외 (2019) 의 IABS 팔 · MarS · 생성형 world agent (Coletta 외 ICAIF 2022)",
    "L6": "**Aquilina–Budish–O'Neill** (QJE 2022, LSE 전체 메시지) · Brogaard–Hendershott–Riordan (RFS 2014) · Menkveld (JFM 2013) · Kyle–Obizhaeva (Econometrica 2016) · Almgren 외 (2005, Citigroup 실주문) · Frazzini–Israel–Moskowitz ($1.7조 실체결)",
}

S["ladder_note"] = open(os.path.join(ROOT, "fig_ladder.html"), encoding="utf-8").read() + """
<div class="callout"><b>사다리를 읽는 법.</b> 등급은 <b>데이터의 해상도가 아니라 체결을 결정하는 규칙</b>으로 매겼다.
밀리초 원장을 쓰고도 “5초마다 원하는 수량이 즉시 다 체결된다” 고 하면 L2 이고,
1분 캔들만 있어도 큐 소진 물량으로 체결을 정하면 L3 다.
그리고 <b>위로 갈수록 좋은 것이 아니다</b> — L1(모형 몬테카를로)은 “모형이 옳다면 이 정책이 최적인가” 를 묻는 정확한 도구이고,
L6(실집행)은 가장 사실적이지만 반사실 질문을 못 던지고 재현도 불가능하다.
문제에 맞는 층을 고르되, <b>어느 층에서 나온 숫자인지 밝히지 않는 것</b>이 이 조사에서 가장 흔한 결함이다.</div>
"""

S["verdict_title"] = "종합 판정 — 여덟 가지 사실"

S["verdict"] = """
<div class="table-scroll"><table><thead><tr><th style="width:250px">사실</th><th>근거</th></tr></thead><tbody>
<tr><th>① 밀리초 데이터를 썼다는 말은 백테스트 사실성을 전혀 보장하지 않는다</th>
<td><b>KRX 밀리초 체결·호가 원장</b>을 확보한 최적실행 RL 논문(arXiv 2307.10649)이 그 원장을 <b>5초 버킷</b>으로 다운샘플링하고,
가정 셋을 본문에 적는다 — 일시 충격만 있고 다음 스텝에 회복, <b>“Commissions and exchange fees are ignored”</b>,
<b>“traded immediately without order arrival delays”</b>. 사다리로는 L2 다.</td></tr>

<tr><th>② 큐 위치를 무시하면 마켓메이킹 손익이 대략 스프레드 하나만큼 틀린다 — 그리고 부호까지 바뀐다</th>
<td><b>Moallemi–Yuan</b>: 대형틱 종목에서 큐 가치가 매수매도 스프레드와 같은 자릿수다.
<b>Cartea–Donnelly–Jaimungal §4.4</b>: 같은 전략이 큐 앞이면 샤프 10~33, 큐 뒤면 ORCL 3.98 · CSCO 0.59 · <b>INTC −1.95</b>.
저자들의 결론 — “the true performance should fall somewhere between that of Table 5 and Table 7.”</td></tr>

<tr><th>③ 학술 백테스트에서 지연과 수수료는 거의 언제나 빠져 있다</th>
<td>실행·마켓메이킹 문헌에서 거래소 수수료를 넣은 것은 <b>Cont–Kukanov</b>(리베이트 0.2/0.25¢ · taker 0.29¢)와
<b>Guilbaud–Pham</b>(€0.0008 리베이트 / €0.0012 수수료) 정도다. 지연을 구조적으로 넣은 것은 <b>ABIDES</b>(쌍별 나노초 지연),
데이터에서 보정해 체결 규칙으로 바꾼 것은 <b>Noble–Rosenbaum–Souilmi(2026)</b> 하나뿐이다.
반면 실무 엔진 <b>hftbacktest</b>·<b>NautilusTrader</b>는 둘 다 지연·수수료를 1급 시민으로 다룬다 —
<b>사실성은 학계가 아니라 오픈소스 실무 쪽이 앞선다.</b></td></tr>

<tr><th>④ 재생 백테스트는 자기영향을 구조적으로 못 다루고, 그 대가가 측정됐다</th>
<td><b>Balch 외(2019)</b>는 같은 주문을 재생과 다중 에이전트에서 각각 넣어, 재생에서는 충격이 되돌아오고
상호작용에서는 <b>새 수준에서 안정된다</b>는 것을 쌍 실험으로 보였다.
<b>Noble–Rosenbaum–Souilmi(2026)</b>는 자기 체결을 충격 커널에서 빼고/넣고 두 번 돌려 결론을 냈다 —
“A simulator that excludes the strategy's own trades from the impact kernel will <b>systematically overstate profitability</b>,
precisely in the aggressive parameterization regime where a practitioner most needs an accurate assessment.”</td></tr>

<tr><th>⑤ 예측 정확도와 거래 가능성은 거의 직교한다</th>
<td><b>Chinco 외(JF 2019)</b>: AR(3) 이 LASSO 보다 표본 외 R̄² 가 <b>높은데</b>(7.365% vs 2.467%)
순 샤프는 <b>−0.662 vs +1.791</b>. 스프레드 차감 후 수익이 나는 거래는 약 3분의 1뿐이다.
분류 정확도·F1 로 보고하고 끝내는 딥러닝 LOB 문헌 전체에 대한 반례다.</td></tr>

<tr><th>⑥ 국내 대형 증권사 중 백테스트 코드를 공개한 곳은 <b>한국투자증권 하나</b>이고, 그 백테스트는 <b>일봉</b>이다</th>
<td><code>koreainvestment/open-trading-api</code> 의 <code>backtester/</code> 는 QuantConnect Lean 을 도커로 돌리는데,
데이터 제공자가 받는 해상도는 <code>Resolution.DAILY</code> 와 <code>Resolution.MINUTE</code> 둘뿐이고
(분봉은 코드 주석부터 “분봉은 단일 날짜”), Lean 코드 생성기는 <code>Resolution.Daily</code> 를 하드코딩한다.
수수료 0.015% · 매도 거래세 0.2% 는 실제 값을 넣었지만 <b>슬리피지 기본값은 0</b> 이다.
<b>키움</b>은 공식 REST 저장소에 337개 API·362개 예제를 열었지만 백테스터가 없고, 틱차트(<code>ka10079</code>)도 호가 없는 봉이다.
<b>NH</b>(PLUG-OpenAPI)는 저장소 3개가 전부 SDK·MCP 다. 그리고 <b>과거 호가+체결 원장은 국내 공개 API 로 살 수 없다</b> —
KRX Open API 는 전부 일별, 코스콤 데이터몰은 현재 서비스 중단이다.</td></tr>
<tr><th>⑦ 통계적 보정은 누출 탐지가 아니다 — 그리고 숨은 탐색 앞에서는 거꾸로 움직인다</th>
<td>일부러 미래를 보는 <b>샤프 35짜리 오라클이 Deflated Sharpe 와 PBO 를 완전히 통과한다</b>(Gençay 2026).
그리고 보고하지 않은 탐색이 10 → 1,000 으로 넓어지면 실제 선택 손상은 0.066 → 0.141 로 커지는데
<b>보고되는 PBO 는 0.54 → 0.10 으로 떨어진다</b> — “No statistic computed on the reported grid recovers what is missing.”
방어 가능한 파이프라인은 <b>구조적 가드레일 · 유효 시행 수 보정 · 최대통계량 부트스트랩</b> 셋을 함께 요구한다.</td></tr>

<tr><th>⑧ 기호적 알파 채굴과 다중검정 보정의 교집합은 비어 있다</th>
<td>AlphaAgent · AlphaMemo · Alpha Singularity 같은 2025~2026 LLM·유전 알파 채굴 논문들은 AST 복잡도 벌점과 독창성 정규화로
과적합을 다루는데 <b>DSR · PBO · haircut 어느 것도 보고하지 않는다.</b> 도구도 같다 —
<code>gplearn</code> · <code>PySR</code> · <code>alphagen</code> 모두 다중검정 통제가 없다.
반대편에서 <b>Chen &amp; Dim</b> 은 채굴 전략 136,000개의 경험적 베이즈 상위 1%가 표본 외 연 5.7%(출판 아노말리 200개는 5.9%)를 벌었음을 보였다 —
<b>탐색을 전부 기록하면 그 크기는 부채가 아니라 추정 자원이 된다.</b></td></tr>
</tbody></table></div>

<div class="oneline"><b>한 줄 결론.</b> 틱 백테스트의 품질을 가르는 것은 데이터 해상도도, 모델의 정교함도 아니라
<b>“무엇이 체결을 결정하는가” 를 얼마나 정직하게 적었는가</b>다. 가장 좋은 작업들(Cartea–Donnelly–Jaimungal의 큐 밴드,
Huang–Lehalle–Rosenbaum의 “체결확률이 약간 과대평가된다”, Cont–Kukanov의 “앞 큐 취소를 안 넣어 보수적이다”,
hftbacktest의 “no market impact is considered”)은 전부 자기 가정의 방향까지 함께 적었다.</div>
"""

_local = open(os.path.join(ROOT, "local_section.html"), encoding="utf-8").read()
_figq = open(os.path.join(ROOT, "fig_queue.html"), encoding="utf-8").read()
_local = _local.replace('<div class="deepbar">실제 산출물 한 조각', _figq + '\n  <div class="deepbar">실제 산출물 한 조각')
S["local_section"] = _local
S["checklist_section"] = open(os.path.join(ROOT, "checklist_section.html"), encoding="utf-8").read()
S["caveats_section"] = open(os.path.join(ROOT, "caveats_section.html"), encoding="utf-8").read()

G = m["groups"]
G["g1"]["intro"] = ("탑티어 금융저널(JF·JFE·RFS·QJE·JFQA)이 틱·메시지 데이터로 하는 것은 대개 <b>백테스트가 아니라 회귀</b>다. "
 "약 25편 중 전략을 실제로 굴린 것은 <b>Budish–Cramton–Shim(QJE 2015)</b>과 <b>Aquilina–Budish–O'Neill(QJE 2022)</b> 둘뿐이고, "
 "손익이 나오는 나머지는 시뮬레이션이 아니라 <b>실제 체결의 회계</b>(BHR·Menkveld·Kirilenko)다. "
 "이 군의 백테스트 위생은 ML 문헌과 방향이 반대다 — 표본 외 분할 대신 <b>외생 충격으로 식별</b>하고, "
 "재현 대신 <b>인과</b>를 산다. 그리고 진짜 위험은 체결 모형이 아니라 그 앞의 <b>측정 계층</b>에 있다(Holden–Jacobsen).")

G["g2"]["intro"] = ("최적실행·마켓메이킹 문헌의 고전은 <b>백테스트를 하지 않는다.</b> 핵심 7편 중 6편에 백테스트가 없고 5편에는 실데이터가 아예 없다 — "
 "최적해가 그 최적해를 낳은 모형 안에서 검증된다. <b>Avellaneda–Stoikov(QF 2008)</b>는 종목명도 날짜도 없는 순수 몬테카를로다. "
 "실데이터로 넘어와도 오랫동안 <b>“내 호가에 체결이 닿으면 전량 체결”</b> 이 표준이었다. "
 "그 가정을 깨는 작업은 넷이다 — 큐 값을 재고(<b>Moallemi–Yuan</b>), 큐 소진으로 체결을 정하고(<b>Cont–Kukanov</b>), "
 "체결확률 밴드를 제시하고(<b>Cartea–Donnelly–Jaimungal</b>), 재생을 버리고 상호작용 시뮬레이터를 만든다(<b>Huang–Lehalle–Rosenbaum</b>). "
 "2026년 <b>Noble–Rosenbaum–Souilmi</b>가 지연·경주·자기영향을 한꺼번에 다루며 현재의 상한을 정했다.")

G["g3"]["intro"] = ("탑티어 AI 학회의 트레이딩 논문에서 가장 흔한 일은 <b>체결을 아예 시뮬레이션하지 않는 것</b>이다. "
 "호가창 딥러닝 계보(DeepLOB → FI-2010 → 수십 편)는 중간가 방향의 F1 을 보고하고 끝내는데, 재현 연구가 "
 "<b>시드 간 F1 표준편차가 10~17 포인트</b>(보고되는 개선폭은 1~3 포인트)이고 다른 데이터로 옮기면 성능이 붕괴함을 보였다. "
 "그리고 Briola 외는 거래가능성 지표 p_T 가 <b>0.01~0.15, 신뢰도로 거르면 정확히 0</b> 인데 같은 구간에서 MCC 는 계속 오른다는 것을 보였다. "
 "체결을 흉내 내는 논문들도 대개 <b>봉 종가·분봉 평균가·중간가</b>에 즉시 전량 체결이다. "
 "예외는 셋뿐이다 — <b>Nevmyvaka 외(ICML 2006)</b>가 호가창 우선순위를 유지하고 <b>가정을 학습에서만 쓰고 테스트에서 푼다</b>, "
 "<b>Schnaubelt(EJOR 2022)</b>가 큐 앞 물량과 거래소별 maker/taker 를 넣는다, 그리고 <b>MarS·LOB-Bench</b>가 자기영향을 생성모델로 만들고 그것을 채점한다. "
 "가장 널리 회자되는 LLM 에이전트 논문들은 전부 <b>일봉·비용 0·단일 창</b>이고, star 수와 평가 엄밀성이 반비례한다.")

G["g4"]["intro"] = ("이 군이 이 리포트의 판정 기준을 제공한다. 핵심 결과는 셋이다. "
 "① <b>시행 횟수를 세지 않은 샤프는 해석 불가능하다</b> — 진짜 실력이 0이어도 <b>10번만 시도하면 표본 내 샤프 1.57</b> 이 나온다. "
 "② <b>보정은 누출 탐지가 아니다</b> — 일부러 미래를 본 샤프 35짜리 오라클이 DSR 과 PBO 를 <b>완전히 통과한다</b>. "
 "③ 그리고 고빈도에서는 <b>통계보다 마찰이 먼저다</b> — 틱 빈도에 가장 가까운 실증 감사가 순진한 프로토콜의 샤프 3.6배 부풀림 중 "
 "<b>대부분을 거래 마찰에 귀속</b>시켰다. "
 "한편 이 문헌 자체도 다투는 중이다 — CPCV 가 워크포워드를 이겼다는 유일한 통제 비교는 채점표를 만든 쪽이 이긴 것이고, "
 "PBO 는 숨은 탐색 앞에서 거꾸로 움직이며, DSR 의 추정량은 James-Stein 축소에 진다. "
 "그리고 <b>기호적 알파 채굴과 다중검정 보정의 교집합은 아직 비어 있다.</b>")

G["g5"]["intro"] = ("연구용 틱/호가창 백테스트가 실제로 가능한 오픈소스는 <b>넷뿐</b>이다 — hftbacktest · NautilusTrader(1.223+) · ABIDES · JAX-LOB. "
 "나머지는 “고빈도” 를 표방하더라도 <b>봉에서 체결한다.</b> 이 계층의 공통 환상은 하나다 — "
 "<b>봉의 저가가 내 지정가를 스쳤으면 전량, 지정가 그대로 체결.</b> 패시브 전략의 엣지 전부에 해당하는 크기다. "
 "가장 큰 평판-실체 격차는 <b>Hummingbot</b> 이다 — 가장 널리 쓰이는 마켓메이킹 봇인데 백테스터가 <b>1분 캔들 종가</b>로 체결을 정한다. "
 "그리고 이 모든 것의 상한을 정하는 것은 알고리즘이 아니라 <b>데이터 등급</b>이다.")

G["g6"]["intro"] = ("<b>결론부터: 국내 대형 증권사 중 백테스트 코드를 공개한 곳은 한국투자증권 하나이고, 그 백테스트는 일봉이다.</b> "
 "키움은 공식 REST 저장소를 열었지만 백테스터가 없고, NH 는 2026년 8월 PLUG 로 뒤늦게 합류했으나 저장소 3개가 SDK·MCP 다. "
 "LS·대신·미래에셋은 공식 GitHub 조차 없어 생태계를 개인 개발자 래퍼가 떠받친다. "
 "더 근본적인 제약은 <b>데이터</b>다 — KRX Open API 는 31개 서비스가 전부 일별이고, 코스콤 데이터몰은 서비스 중단 상태이며, "
 "한국투자증권의 체결 API 는 <b>당일치</b>뿐이다. 그래서 국내에서 호가창 재생 백테스트를 하려면 "
 "실시간을 직접 수집하거나 기관 계약을 맺는 두 길밖에 없다.")

json.dump(m, open(M, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

deep = sum(len(g["deep"]) for g in G.values())
brief = sum(len(g["brief"]) for g in G.values())
S["lead"] = S["lead"].replace("{deep}", str(deep)).replace("{brief}", str(brief))
json.dump(m, open(M, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("sections updated. deep:", deep, "brief:", brief)
