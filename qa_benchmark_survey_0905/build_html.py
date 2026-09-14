#!/usr/bin/env python3
# notes/*.json + figs/*.png -> 사이트 양식 HTML 리포트
import json, os, re, io, base64, html, datetime, sys
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CSS = open(os.path.join(ROOT, "site", "ex3.css"), encoding="utf-8").read()
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "LLM설계_QA벤치마크_심층분석_0905.html")
DATE = "2026-09-05"

GROUPS = [
  {"id":"g1","cls":"g1","name":"온체인 · 크립토 · Web3","short":"온체인",
   "deep":["spider2","dmind","lattice","intent2tx","evmquest","cryptobench"],"brief":["cryptoanalystbench","txsum"],
   "intro":"온체인 데이터를 다룬 QA·에이전트 벤치마크 8편. 탑티어 venue에서 실제 주소·블록·topic 해시를 질문에 박은 <b>C형</b>은 Spider 2.0(ICLR 2025 Oral)의 BigQuery 블록체인 서브셋뿐이고, 전용 Web3 벤치마크(DMind)는 전문가 수작성 객관식(A/B)이다. <b>LLM이 질문을 생성한 온체인 세트는 arXiv에만</b> 있으며(LATTICE·Intent2Tx·EVM-QuestBench), 그중 \"제공된 원자료 위의 결정론적 계산 답\"을 갖는 것은 없다."},
  {"id":"g2","cls":"g2","name":"금융 · 경제 · 비즈니스","short":"금융",
   "deep":["crmarena","finrate","fintmmbench","finsearchcomp","bizfinbench","steerme"],"brief":["finmathbench","econlogicqa","bizbench"],
   "intro":"LLM이 QA 구성에 개입한 금융·비즈니스 벤치마크 9편. <b>CRMArena</b>(NAACL 2025)의 \"템플릿 → DB에서 정답 계산 → LLM paraphrase\" 파이프라인이 우리 세트와 구조적으로 가장 가깝고, <b>FinSearchComp</b>(ICLR 2026) T3는 문턱·반올림·출력 형식을 질문 안에 명시하는 진짜 C형이지만 전문가가 썼다. <b>BizFinBench</b> FNC(\"종목 X의 12/1~12/18 순유입\")는 의미적으로 가장 닮았으나 arXiv·중국어다."},
  {"id":"g3","cls":"g3","name":"희소 · 마이너 도메인","short":"희소 도메인",
   "deep":["climaqa","ctibench","oceanbench","agmmu","sportsmetrics","medcalc"],"brief":["teleqna","vrsbench"],
   "intro":"기후·위협인텔리전스·해양·농업·스포츠·임상 등 희소 도메인 8편. 탑티어 venue의 LLM 생성 세트(ClimaQA·CTIBench·OceanBench·AgMMU·VRSBench)는 전부 <b>지식·인식형 A/B</b>이고, 결정론적 계산형(SportsMetrics·MedCalc-Bench)은 <b>LLM이 설계하지 않은</b> 것뿐이다."},
  {"id":"g4","cls":"g4","name":"도메인 무관 · 계산형 데이터 QA (구조적 이웃)","short":"계산형 QA",
   "deep":["infiagent","dabstep","birdinteract","fdabench","tablebench","claimdb"],"brief":["qrdata","kramabench"],
   "intro":"도메인과 무관하게 \"제공된 표·DB·로그에서 결정론적 값을 계산하는 QA\" 8편. <b>InfiAgent-DABench</b>(ICML 2024)는 LLM이 constraints·format까지 질문에 넣고 코드 실행으로 정답을 만드는 <b>제작 레시피</b>가 가장 가깝고, <b>BIRD-Interact/LiveSQLBench</b>(ICLR 2026 Oral)는 가정·문턱·반올림을 <b>질문 안에 명시</b>하는 스타일이 가장 가깝다. <b>DABstep</b>은 형태상 가장 비슷하지만 NeurIPS 2025 D&amp;B reject였고 규약이 manual.md에 있다."},
]


VENUE_SHORT = {
 "spider2":"ICLR 2025 Oral","dmind":"KDD 2026 (D&B)","lattice":"arXiv 2026","intent2tx":"arXiv 2026","evmquest":"arXiv 2026","cryptobench":"arXiv 2025/26",
 "cryptoanalystbench":"arXiv 2026","txsum":"EMNLP 2026 (self-claim)",
 "crmarena":"NAACL 2025","finrate":"KDD 2026","fintmmbench":"ACM MM 2025","finsearchcomp":"ICLR 2026","bizfinbench":"arXiv 2025","steerme":"NeurIPS 2025 D&B",
 "finmathbench":"AAAI 2026","econlogicqa":"EMNLP 2024 Findings","bizbench":"ACL 2024",
 "climaqa":"ICLR 2025","ctibench":"NeurIPS 2024 D&B Spotlight","oceanbench":"ACL 2024","agmmu":"NeurIPS 2025 D&B","sportsmetrics":"ACL 2024","medcalc":"NeurIPS 2024 D&B Oral",
 "teleqna":"arXiv / IEEE Network","vrsbench":"NeurIPS 2024 D&B",
 "infiagent":"ICML 2024","dabstep":"arXiv (NeurIPS'25 D&B reject)","birdinteract":"ICLR 2026 Oral","fdabench":"KDD 2026","tablebench":"AAAI 2025","claimdb":"ACL 2026",
 "qrdata":"ACL 2024 Findings","kramabench":"ICLR 2026",
}
QA_SHORT = {
 "spider2":"사람 작성(CS 저자 8인) · LLM은 paraphrase만 · 정답 = SQL 실행 결과",
 "dmind":"전문가 5인 수작성 MCQ+주관식 · LLM 생성 아님(에이전트 검색·NLI 검증만)",
 "lattice":"완전 LLM 생성(80 seed→240→1,200) · 정답 없음, LLM judge",
 "intent2tx":"실제 tx trace에서 Gemini가 intent 역생성 + 수동 spot-check · QA 아닌 intent→tx",
 "evmquest":"사람 task 명세 + LLM이 NL 템플릿 생성 · 포크 체인 상태 검증",
 "cryptobench":"크립토 전문가 위원회 수작성 · 3단계 검증 · 데이터 미공개",
 "cryptoanalystbench":"실제 사용자 쿼리 5단계 큐레이션(LLM 보조 중복제거) · 루브릭 judge",
 "txsum":"사람 주석 요약(187 tx) · QA 아님",
 "crmarena":"seed 템플릿 → DB에서 정답 계산 → LLM paraphrase · DB 자체 gpt-4o 합성",
 "finrate":"DeepSeek-V3.2 생성 → 이중 LLM 검증 → 전문가 확인 · SEC 공시",
 "fintmmbench":"전문가 템플릿 + GPT-4o-mini 생성 · 사람 검수",
 "finsearchcomp":"금융 전문가 70인 수작성 · T1은 API로 정답 갱신",
 "bizfinbench":"실제 사용자 쿼리 + GPT-4o 정제·합성 + 전문가 합의",
 "steerme":"사람 템플릿 → gpt-4o style-transfer → 사람 검수 · 합성 문제",
 "finmathbench":"공식 DAG 기반 완전 LLM 생성",
 "econlogicqa":"GPT-4 생성 + 사람 검수 · 사건 순서 MCQ",
 "bizbench":"FinCode 137 중 91 LLM 생성 + 전문가 검증 · 나머지 기존셋 파생",
 "climaqa":"ClimaGen: GPT-3.5/4o-mini가 교재에서 생성 · Gold 전문가 검증/Silver 합성",
 "ctibench":"GPT-4o가 ATT&CK·CWE 등에서 MCQ 생성 · 수동 검증",
 "oceanbench":"전문가 seed → LLM 다중 에이전트 생성 · 전문가 검증",
 "agmmu":"실제 농부–전문가 대화에서 LLaMA/GPT-4o가 QA 생성 · 사람 검증",
 "sportsmetrics":"프로그램 생성(play-by-play 합산) · LLM 아님",
 "medcalc":"사람+GPT 보조 노트 선별 · 계산기 구현으로 정답 산출",
 "teleqna":"GPT-3.5 생성기+검증기 · 전문가 검토",
 "vrsbench":"GPT-4V 생성 · 사람 검증",
 "infiagent":"GPT-4가 질문+constraints+format 생성 → 코드 실행 정답 → 사람 검수",
 "dabstep":"Adyen 내부 실제 쿼리 95개 핵심에서 450개 순열 · 사람 작성 · 규약은 manual.md",
 "birdinteract":"사람 작성(선발 주석자 12명) · 테스트케이스 채점 · 모호성 주입",
 "fdabench":"gold SQL 실행 → 에이전트(PUDDING)가 초안 생성 → 전문가 승인/반려",
 "tablebench":"사람 seed → GPT-4 에이전트 생성 → 30% 수동/70% GPT-4 검증 · 정답 사람 검토",
 "claimdb":"BIRD SQL 실행 → gpt-5가 claim 생성 → 다중 judge 필터",
 "qrdata":"교재·논문 연습문제 그대로 + 저자 주석 · 사람",
 "kramabench":"출판 연구 재현 · 기여자 교차 검증 · 사람",
}

def esc(s): return html.escape(str(s if s is not None else ""), quote=False)
def md(s):
    """아주 가벼운 인라인 마크다운: **b**, `code`"""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s

def load_notes():
    notes = {}
    d = os.path.join(ROOT, "notes")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            try:
                notes[f[:-5]] = json.load(open(os.path.join(d, f), encoding="utf-8"))
            except Exception as e:
                print("!! bad json", f, e)
    return notes

def img_b64(path, max_w=1000):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p): return None
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    png = buf.getvalue()
    if len(png) > 140_000:   # 큰 그림은 JPEG로
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=82, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(png).decode()

CLASS_PILL = {"A":"tagA","B":"tagB","C":"tagC","C-lite":"tagCl"}
def class_pill(c):
    c = (c or "").strip()
    key = "C-lite" if c.lower().startswith("c-lite") else c[:1].upper() if c else ""
    label = {"A":"A 일반 QA","B":"B 도메인 지식 QA","C":"C 상황특정 계산형","C-lite":"C-lite"}.get(key, esc(c))
    return f'<span class="pill {CLASS_PILL.get(key,"")}">{label}</span>'

def link(u, text=None):
    if not u: return ""
    return f'<a href="{esc(u)}" target="_blank" rel="noopener">{esc(text or u)}</a>'

def links_line(L):
    parts = []
    if L.get("arxiv"): parts.append(link(L["arxiv"], "arXiv " + L["arxiv"].rsplit("/",1)[-1]))
    if L.get("pdf"): parts.append(link(L["pdf"], "PDF"))
    if L.get("proceedings"): parts.append(link(L["proceedings"], "Proceedings/OpenReview"))
    return " · ".join(parts) or "—"

def data_line(L):
    parts = []
    if L.get("github"): parts.append(link(L["github"], "GitHub " + re.sub(r"^https?://github.com/","",L["github"]).rstrip("/")))
    if L.get("huggingface"): parts.append(link(L["huggingface"], "HF " + re.sub(r"^https?://huggingface.co/(datasets/)?","",L["huggingface"]).rstrip("/")))
    if L.get("website"): parts.append(link(L["website"], "웹사이트"))
    return " · ".join(parts) or '<span class="pill">미공개 / 링크 없음</span>'

def figure_html(fg):
    src = img_b64(fg["file"])
    if not src: return f'<!-- missing figure {esc(fg["file"])} -->'
    return (f'<figure class="paperfig"><span class="figsrc">원본 {esc(fg.get("label",""))} · p.{esc(fg.get("page",""))}</span>'
            f'<img src="{src}" alt="{esc(fg.get("alt",""))}">'
            f'<figcaption>논문 {esc(fg.get("label",""))} — {md(fg.get("caption_ko",""))}</figcaption></figure>')

def paper_section(n, key, N, gcls):
    L = N.get("links", {}) or {}
    ds = N.get("dataset", {}) or {}
    sq = N.get("seven_questions", {}) or {}
    rv = N.get("review")
    tags = "".join(f'<span class="pill tag{gcls[-1]}">{esc(t)}</span> ' for t in (N.get("tags") or [])[:4])
    figs = N.get("figures") or []
    h = [f'<section id="p{n}" class="paper {gcls}">']
    h.append(f'<h2>{esc(N.get("title",key))}<span class="vtag">{esc(N.get("authors_short",""))} · {esc(N.get("venue",""))}</span></h2>')
    h.append('<div class="meta-box">')
    h.append(f'<div><b>유형</b><span>{esc(N.get("type",""))} · {esc(N.get("domain",""))}</span></div>')
    h.append(f'<div><b>원문</b><span>{links_line(L)}</span></div>')
    h.append(f'<div><b>코드/데이터</b><span>{data_line(L)} <span class="small">({esc(L.get("verified",""))})</span></span></div>')
    h.append(f'<div><b>venue 검증</b><span>{esc(N.get("venue_verified",""))}</span></div>')
    h.append(f'<div><b>데이터 범위</b><span>{esc(N.get("data_scope",""))}</span></div>')
    h.append(f'<div><b>핵심 태그</b><span>{tags}</span></div>')
    h.append(f'<div><b>QA 등급</b><span>{class_pill(ds.get("class"))} <span class="small">{md(ds.get("class_reason",""))}</span></span></div>')
    h.append('</div>')
    # 그림 1
    if figs: h.append(figure_html(figs[0]))
    # 쉽게 말하면
    ez = N.get("easy") or {}
    h.append('<div class="easy"><div class="ezhead">쉽게 말하면</div>')
    h.append(f'<p><b>무슨 문제?</b> {md(ez.get("problem",""))}</p><p><b>어떻게?</b> {md(ez.get("how",""))}</p>')
    if ez.get("analogy"): h.append(f'<div class="analogy"><b>비유.</b> {md(ez["analogy"])}</div>')
    h.append('</div>')
    # 핵심 요약
    h.append('<div class="deepbar">핵심 요약 · 문제의식과 기여</div><ol class="contrib">')
    for c in N.get("contributions") or []: h.append(f'<li>{md(c)}</li>')
    h.append('</ol>')
    # 그림 2+
    for fg in figs[1:]: h.append(figure_html(fg))
    # 데이터셋 · 벤치마크
    h.append('<div class="deepbar">데이터셋 · 벤치마크 깊이 보기</div>')
    h.append(f'<h4><span class="spec">1</span>문항·정답은 어떻게 만들었나 (LLM 관여 여부)</h4><p>{md(ds.get("construction",""))}</p>')
    h.append(f'<h4><span class="spec">2</span>규모 · 형식 · 공개</h4><p>{md(ds.get("size_format",""))}</p><p><b>공개.</b> {md(ds.get("release",""))}'
             + (f' <b>오염 방어.</b> {md(ds["contamination"])}' if ds.get("contamination") else "") + '</p>')
    if ds.get("samples"):
        h.append('<h4><span class="spec">3</span>실제 문항 (verbatim)</h4>')
        for s in ds["samples"]: h.append(f'<div class="qbox">{esc(s)}</div>')
    h.append(f'<h4><span class="spec">4</span>QA 등급 판정</h4><p>{class_pill(ds.get("class"))} {md(ds.get("class_reason",""))}</p>')
    if ds.get("evaluation"): h.append(f'<h4><span class="spec">5</span>평가 · 주요 결과</h4><p>{md(ds["evaluation"])}</p>')
    # 7대 질문
    if sq:
        h.append('<h4>데이터셋 7대 질문</h4><div class="table-scroll"><table><thead><tr><th style="width:240px">질문</th><th>답</th></tr></thead><tbody>')
        for k_, lab in [("q1_source","① 기존 공개 수집 vs 신규 제작"),("q2_method","② 수집 방법 · 출처 분명성"),("q3_license","③ 오픈소스 / 제휴 / 유료 · 타기관 협의"),
                        ("q4_relabel","④ 공개데이터 그대로 vs task 맞춤 재라벨링"),("q5_industry","⑤ 자체구성 vs 기업협업 · 데이터 공개 여부"),
                        ("q6_constraints","⑥ 저작권·보안·이익 제약 · 그럼에도 공개?"),("q7_accept","⑦ 리뷰어·AC가 높이 산 기여")]:
            if sq.get(k_): h.append(f'<tr><th>{lab}</th><td>{md(sq[k_])}</td></tr>')
        h.append('</tbody></table></div>')
    # 리뷰
    if rv and (rv.get("decision") or rv.get("why_accept")):
        h.append('<div class="review"><div class="rvhead">동료 리뷰 — 어떻게 받아들여졌나</div>')
        sc = ""
        if rv.get("scores"):
            sc = '<div class="rvscore">' + "".join(f'<span class="s">{esc(x.strip())}</span>' for x in str(rv["scores"]).replace("/", ",").split(",") if x.strip()) + '</div>'
        h.append(f'<div class="rvdec"><b>결과:</b> {md(rv.get("decision",""))}{sc}</div>')
        h.append('<div class="rvgrid"><div class="rvcol ok"><h5>✓ 인정된 기여</h5><ul class="tight">' + "".join(f'<li>{md(x)}</li>' for x in rv.get("praised") or []) + '</ul></div>')
        h.append('<div class="rvcol con"><h5>⚠ 제기된 우려</h5><ul class="tight">' + "".join(f'<li>{md(x)}</li>' for x in rv.get("concerns") or []) + '</ul></div></div>')
        if rv.get("why_accept"): h.append(f'<div class="rvwhy"><b>accept의 결정적 이유.</b> {md(rv["why_accept"])}</div>')
        h.append('</div>')
    if N.get("limitations"): h.append(f'<div class="warn"><b>한계.</b> {md(N["limitations"])}</div>')
    if N.get("oneline"): h.append(f'<div class="oneline"><b>한 줄 요약.</b> {md(N["oneline"])}</div>')
    if N.get("implications"):
        h.append('<div class="mynote"><div class="mnhead">✍︎ 우리 QA셋(T124 계산형 60문항)에 주는 시사점</div><ul>')
        for x in N["implications"]: h.append(f'<li>{md(x)}</li>')
        h.append('</ul></div>')
    h.append('</section>')
    return "\n".join(h)

def brief_card(key, N):
    L = N.get("links", {}) or {}; ds = N.get("dataset", {}) or {}
    s = f'<div class="card"><h4>{esc(N.get("title",key))} <span class="small">{esc(N.get("venue",""))}</span></h4>'
    s += f'<p class="small">{esc(N.get("authors_short",""))} · {links_line(L)} · {data_line(L)}</p>'
    s += f'<p>{md(ds.get("construction",""))}</p>'
    if ds.get("samples"): s += f'<div class="qbox">{esc(ds["samples"][0])}</div>'
    s += f'<p>{class_pill(ds.get("class"))} {md(ds.get("class_reason",""))}</p>'
    if N.get("oneline"): s += f'<p><b>한 줄.</b> {md(N["oneline"])}</p>'
    return s + '</div>'

def build():
    notes = load_notes()
    order = []  # (n, key, gcls)
    n = 0
    for g in GROUPS:
        for k in g["deep"]:
            n += 1; order.append((n, k, g["cls"]))
    num = {k: i for i, k, _ in order}

    # ---------- 요약표 ----------
    rows = []
    for g in GROUPS:
        for k in g["deep"] + g["brief"]:
            N = notes.get(k)
            if not N: rows.append(f'<tr><td colspan="6"><i>{esc(k)}: 노트 없음</i></td></tr>'); continue
            ds = N.get("dataset", {}) or {}; L = N.get("links", {}) or {}
            name = f'<a href="#p{num[k]}">{esc(N.get("title",k))}</a>' if k in num else esc(N.get("title",k))
            cons_short = QA_SHORT.get(k) or ((ds.get("construction","")[:110] + "…"))
            rows.append(f'<tr><td><span class="pill tag{g["cls"][-1]}">{esc(g["short"])}</span></td><td><b>{name}</b></td><td>{esc(VENUE_SHORT.get(k, N.get("venue","")))}</td>'
                        f'<td>{esc(N.get("domain",""))}</td><td>{esc(cons_short)}</td><td>{class_pill(ds.get("class"))}</td><td>{data_line(L)}</td></tr>')
    summary_table = ('<div class="table-scroll"><table><thead><tr><th>군</th><th>벤치마크</th><th>venue</th><th>도메인</th><th>QA 구성(요약)</th><th>등급</th><th>데이터 공개</th></tr></thead><tbody>'
                     + "\n".join(rows) + '</tbody></table></div>')

    # ---------- 등급 분포 ----------
    from collections import Counter
    cnt = Counter(); names = {}
    for g in GROUPS:
        for k in g["deep"]:
            c = ((notes.get(k, {}).get("dataset", {}) or {}).get("class") or "").strip()
            key = "C-lite" if c.lower().startswith("c-lite") else c[:1].upper()
            cnt[key] += 1; names.setdefault(key, []).append(notes[k].get("title","").split(":")[0] if k in notes else k)
    dist = " · ".join(f"{k} {cnt[k]}편" + (f"({'·'.join(names[k])})" if k == "C" else "") for k in ["C","C-lite","B","A"] if cnt[k]) 
    # ---------- nav ----------
    nav = ['<a href="#start">시작</a>', '<a href="#criteria" class="hl">분류 기준</a>', '<a href="#summary" class="hl">종합 요약표</a>', '<a href="#verdict" class="hl">종합 판정</a>']
    for g in GROUPS: nav.append(f'<a href="#{g["id"]}">{esc(g["short"])}</a>')
    nav.append('<a href="#cite">인용 후보</a><a href="#caveats">검증 한계</a>')
    navgrid = []
    for g in GROUPS:
        navgrid.append(f'<span class="navhd">{esc(g["name"])}</span>')
        for k in g["deep"]:
            N = notes.get(k, {}); t = N.get("title", k)
            navgrid.append(f'<a href="#p{num[k]}">{num[k]} {esc(t[:38])}</a>')

    # ---------- 본문 ----------
    body = []
    body.append(f'''
<header class="topbar"><div class="wrap">
  <h1>LLM 설계 QA 벤치마크 심층분석 — 온체인 · 금융 · 희소 도메인 24편 + 요약 9편 (웹 원문 PDF 정독 · 공개 데이터셋 실열람)</h1>
  <div class="submeta">
    <span class="b">질문: 2024–26 탑티어 벤치마크 중 QA셋을 LLM으로 설계한 논문은? 그중 일반 QA vs 도메인 특화(상황 특정 계산형) 구분</span>
    <span class="b">비교 기준: T124_W26_computed_60 (온체인 계산형 60문항)</span>
    <span class="b">원문 PDF 정독 + GitHub/HF 데이터 실열람 + verbatim 문항 인용 + 원본 그림</span>
    <span class="b">{DATE}</span>
  </div>
  <nav class="nav">{"".join(nav)}
    <details class="navdetails"><summary>&#9656; 논문 24편 목차</summary><div class="navgrid">{"".join(navgrid)}</div></details>
  </nav>
</div></header>
<main class="wrap">
<section id="start">
  <div class="how"><b>읽는 법.</b> 각 논문은 <b>메타(원문·코드/데이터 링크·venue 검증·QA 등급) → 원본 그림 → 쉽게 말하면 → 핵심 요약(기여) → 데이터셋·벤치마크 깊이 보기(구축 방법·규모·verbatim 문항·등급 판정·평가) → 7대 질문 → 리뷰 → 한계 → 한 줄 → 우리 QA셋 시사점</b> 순서다. 링크는 모두 새 탭에서 열린다. 모든 수치는 PDF 쪽수를 달았고, 열지 못한 링크·확인 못 한 venue는 그대로 "미확인"으로 남겼다.</div>
  <p class="lead">질문은 두 겹이다. (i) 온체인 → 없으면 금융 → 없으면 희소 도메인 순으로, 2024–2026 탑티어 AI 학회(ICML·ICLR·NeurIPS·AAAI·KDD·ACL 계열) 벤치마크 중 <b>QA셋을 LLM으로 설계한 논문</b>이 무엇인가. (ii) 그 공개 데이터셋을 실제로 열어 봤을 때 <b>일반 자연어 QA</b>인지, 우리 T124 세트처럼 <b>아주 특수한 상황을 가정하는 디테일한 계산형 QA</b>인지. 결론부터: <b>"LLM 설계 + 상황 특정 계산형(C) + 탑티어"를 동시에 만족하는 벤치마크는 세 도메인 어디에도 없었다.</b> 근접 사례들은 세 조건 중 하나씩만 채운다(<a href="#verdict">종합 판정</a>).</p>
  <div class="grid">
    <div class="card"><h4>조사 범위</h4><p>1차 광역 조사(4갈래 병렬, 약 60편 후보)에서 venue·구축 방식·공개 여부를 확인한 뒤, 우리 질문에 직접 답하는 <b>24편을 원문 PDF 정독 + 데이터 실열람</b>으로 심층 분석하고 9편은 요약 카드로 붙였다. 트레이딩 환경(CryptoTrade·InvestorBench 등)과 코드 생성(SolEval·ChainBench)은 QA가 아니라 제외했다.</p></div>
    <div class="card"><h4>비교 기준 세트</h4><p><code>T124_W26_computed_60.csv</code>: 이더리움 원장 테이블(token_transfers·logs·transactions·traces·contracts) 1주 구간(block 25,369,414–25,419,597) 위 60문항(T1 순보유량 변화 20 · T2 거래소 귀속 판정 20 · T4 DEX run 폐쇄성 20). 질문 안에 주소·토큰·블록 구간·디코딩 규약·문턱·판정 규칙 전문이 있고, 답은 원장에서 결정론적으로 계산되는 값이며 evidence 레코드가 동봉된다.</p></div>
  </div>
</section>

<section id="criteria">
  <h2>분류 기준 — 일반 QA(A) · 도메인 지식 QA(B) · 상황 특정 계산형 QA(C)<span class="vtag">모든 논문의 등급은 공개 데이터의 실제 문항을 열어 본 뒤 매겼다</span></h2>
  <div class="table-scroll"><table><thead><tr><th style="width:170px">등급</th><th>뜻</th><th>예</th></tr></thead><tbody>
    <tr><th><span class="pill tagA">A 일반 QA</span></th><td>일반인·시험 수준의 지식형 질문. 객관식 포함</td><td>"What does AMM stand for?" (DMind)</td></tr>
    <tr><th><span class="pill tagB">B 도메인 지식 QA</span></th><td>전문 지식이 필요하지만 지식·설명형. 구체적 데이터 인스턴스에 묶이지 않음</td><td>"What threshold actually triggers liquidation on Aave v3?" (LATTICE)</td></tr>
    <tr><th><span class="pill tagC">C 상황특정 계산형</span></th><td>구체 개체(주소·계정·날짜·블록 구간)와 규칙·문턱·규약이 질문에 명시되고, 답은 제공된 원자료(테이블/DB/로그)에서 결정론적으로 계산되는 값</td><td>"주소 0x1ab4… 가 block 25369414–25369513 사이에 토큰 0xa0b8… 을 얼마나 받고 보냈는지, 순변화는? 셋 다 최소 단위 정수로" (T124)</td></tr>
    <tr><th><span class="pill tagCl">C-lite</span></th><td>원자료에서 계산은 하지만 소형 단일 테이블이거나 규약·개체 지정이 약함</td><td>"What is the average number of tropical cyclones per season?" (TableBench)</td></tr>
  </tbody></table></div>
  <div class="callout"><b>우리 세트만 갖고 있고 24편 어디에도 없는 요소.</b> (a) 판정 규칙 <b>전문</b>을 질문 안에 넣는 자기완결성(DABstep·Spider 2.0은 외부 manual/.md에 둠), (b) <code>out_of_scope</code>·<code>indeterminate</code>처럼 <b>자료 한계를 인식해야 맞는 문항</b>, (c) 답을 만든 <b>evidence 레코드</b>를 문항마다 동봉, (d) 78자리 정수·int256/uint256 부호 규약 같은 <b>원장 고유 함정</b>.</div>
</section>

<section id="summary">
  <h2>종합 요약표 — 33편<span class="vtag">군 · venue · 도메인 · QA 구성 · 등급 · 데이터 공개 링크</span></h2>
  {summary_table}
</section>

<section id="verdict">
  <h2>종합 판정 — 세 축(LLM 설계 × C형 × 탑티어)을 동시에 채우는 벤치마크는 없다</h2>
  <div class="table-scroll"><table><thead><tr><th style="width:280px">채우는 축</th><th>벤치마크</th><th>못 채우는 축</th></tr></thead><tbody>
    <tr><th>온체인 도메인 + C형 + 탑티어</th><td>Spider 2.0 블록체인 서브셋 (ICLR 2025 Oral)</td><td>사람 작성(LLM은 paraphrase), 답이 SQL 결과 테이블, 전체의 약 5%</td></tr>
    <tr><th>LLM 설계 + C형(질문 내 규약·형식) + 탑티어</th><td>InfiAgent-DABench (ICML 2024)</td><td>도메인 일반적(Titanic 등), 개체·구간 지정 없음</td></tr>
    <tr><th>LLM 개입 + C형 + 탑티어</th><td>CRMArena (NAACL 2025, C) · FDABench (KDD 2026, C-lite 객관식) · ClaimDB (ACL 2026, C-lite 검증형)</td><td>CRM/일반 DB. 규약이 환경 metadata·manual에 있거나, 답이 선택지·라벨</td></tr>
    <tr><th>C형 + 질문 내 규약 명시 + 탑티어</th><td>BIRD-Interact/LiveSQLBench (ICLR 2026 Oral, C) · FinSearchComp T2/T3 (ICLR 2026, C-lite: 원자료 미동봉·오픈도메인 검색)</td><td>사람(전문가) 작성</td></tr>
    <tr><th>희소 도메인 + C형 + 탑티어</th><td>SportsMetrics (ACL 2024, C) · MedCalc-Bench (NeurIPS 2024 D&amp;B Oral, C-lite)</td><td>프로그램 생성 또는 사람 작성. LLM이 문항을 설계하지 않음</td></tr>
    <tr><th>온체인 + LLM 설계</th><td>LATTICE · Intent2Tx · EVM-QuestBench</td><td>arXiv만. 정답 없음 또는 QA가 아닌 action</td></tr>
    <tr><th>참고 세트와 의미적 쌍둥이(종목·구간·순유입)</th><td>BizFinBench FNC/FTR (C-lite)</td><td>arXiv, 중국어, 부분 합성, 순유입 정의·단위 규약이 질문 밖</td></tr>
    <tr><th>C형 + 데이터·규약 동봉</th><td>DABstep</td><td>NeurIPS 2025 D&amp;B reject, 사람 작성, 규약은 manual.md</td></tr>
  </tbody></table></div>
  <div class="oneline"><b>한 줄 결론.</b> 온체인·금융·희소 도메인 모두에서 <b>"LLM이 설계한 상황 특정 계산형 QA"는 탑티어에 비어 있다.</b> 도메인 선례는 Spider 2.0, 제작 파이프라인 선례는 InfiAgent-DABench·CRMArena·FDABench·ClaimDB, 질문 스타일 선례는 BIRD-Interact/LiveSQLBench·FinSearchComp·DABstep에서 각각 가져오면 된다. 심층 24편의 등급 분포: {dist} — "도메인 특화 = 계산형"이 아님이 분명하다.</div>
</section>
''')

    for g in GROUPS:
        body.append(f'<section id="{g["id"]}" class="grouphead {g["cls"]}"><h2>{esc(g["name"])}<span class="vtag">심층 {len(g["deep"])}편 · 요약 {len(g["brief"])}편</span></h2><p class="lead">{g["intro"]}</p></section>')
        for k in g["deep"]:
            N = notes.get(k)
            if not N:
                body.append(f'<section id="p{num[k]}" class="paper {g["cls"]}"><h2>{esc(k)}<span class="vtag">노트 없음 — 작성 대기</span></h2></section>'); continue
            body.append(paper_section(num[k], k, N, g["cls"]))
        briefs = [notes[k] for k in g["brief"] if k in notes]
        if briefs:
            body.append(f'<section class="{g["cls"]}"><h2>{esc(g["short"])} · 요약 카드<span class="vtag">심층 분석 대상에서 제외했지만 지형을 채우는 논문</span></h2><div class="grid">' + "".join(brief_card(k, notes[k]) for k in g["brief"] if k in notes) + '</div></section>')

    body.append('''
<section id="cite">
  <h2>용도별 인용 후보 — 우리 논문에서 어디에 무엇을 인용할까</h2>
  <div class="grid">
    <div class="card"><h4>도메인 선례 (온체인 C형)</h4><p>Spider 2.0 (ICLR 2025 Oral) 블록체인 task <code>sf_bq083</code>·<code>sf_bq058</code>·<code>sf_bq444</code> — 같은 USDC 주소, mint/burn 셀렉터, Optimism 브리지 topic 해시가 질문에 등장. "탑티어에서 온체인 계산형 질문이 통과된 적이 있다"의 근거.</p></div>
    <div class="card"><h4>제작 파이프라인 (LLM 생성 → 코드 실행 정답 → 사람 검수)</h4><p>InfiAgent-DABench (ICML 2024) · CRMArena (NAACL 2025) · FDABench (KDD 2026) · TableBench (AAAI 2025) · ClaimDB (ACL 2026). 우리 "LLM이 문항을 설계하고 notebook이 정답을 재계산하고 독립 verifier가 검산"하는 구조의 직접 선례.</p></div>
    <div class="card"><h4>질문 스타일 (규약·문턱·반올림을 질문 안에)</h4><p>BIRD-Interact / LiveSQLBench (ICLR 2026 Oral) · FinSearchComp T3 (ICLR 2026) · DABstep guidelines · Vals Finance Agent Benchmark. 우리 문항의 "[정의 고정: …]" 블록이 이 계보에 있음을 보이는 용도.</p></div>
    <div class="card"><h4>A/B형 대조군 (도메인 특화지만 지식형)</h4><p>DMind (KDD 2026) · ClimaQA (ICLR 2025) · CTIBench (NeurIPS 2024 D&amp;B) · OceanBench (ACL 2024) · TeleQnA. "도메인 특화 = 계산형"이 아님을 보이는 대조.</p></div>
    <div class="card"><h4>결정론 계산 + 규칙 변경 변형</h4><p>SportsMetrics (ACL 2024) — "각 액션 1점" 같은 규칙 교체 후 재계산은 우리 "정의 민감도" 문항(C-T4-11)과 같은 발상. MedCalc-Bench (NeurIPS 2024 D&amp;B oral) — 계산기 규칙이 정답을 결정하는 임상 계산형.</p></div>
    <div class="card"><h4>가장 닮은 의미적 쌍둥이</h4><p>BizFinBench FNC/FTR (arXiv) — "종목·날짜 구간·순유입". 탑티어가 아니라 본문 인용보다는 related work 각주용.</p></div>
  </div>
</section>

<section id="caveats">
  <h2>검증 한계 — 확인하지 못한 것과 확인 방법</h2>
  <ul class="tight">
    <li><b>venue 확인 경로.</b> OpenReview 사이트·API는 이 세션에서 브라우저 검증(403)에 막혀, 결정·점수는 OpenReview 검색 API(notes/search)·papercopilot 집계·HF 리뷰 미러로 확인했다(Spider 2.0 'ICLR 2025 Oral', BIRD-Interact 'ICLR 2026 Oral' 8/8/8/6, FinSearchComp 8/6/6/4, ClimaQA 8/6/6/8/6, CTIBench Spotlight 6/8/9, MedCalc Oral 7/8/8, AgMMU 5/5/5/5, STEER-ME 3/6/4, DABstep reject 5/5/4/4, KramaBench poster 4/4/6/6). 리뷰 본문까지 읽은 것은 ClimaQA·BIRD-Interact·FinSearchComp뿐이다.</li>
    <li><b>ACM 계열(KDD·ACM MM).</b> dl.acm.org 는 Cloudflare 403이라 DOI는 Crossref 메타데이터로 확인했다(DMind 10.1145/3770855.3817512, Fin-RATE …3817528, FDABench …3817454, FinTMMBench 10.1145/3746027.3755723). TxSum의 EMNLP 2026은 GitHub README self-claim뿐이다.</li>
    <li><b>데이터를 열지 못한 것.</b> CryptoBench(논문에 공개 URL 없음, 동명 GitHub 저장소는 다른 프로젝트), STEER-ME 웹앱(Streamlit; 대신 HF narunraman/steer_me 표본), Intent2Tx 익명 저장소(루트 401; README 경유 HF 데이터셋은 열람), MedCalc-Bench HF(gated; GitHub test_data.csv 1,100행으로 대체), DABstep hidden test 정답(450행 answer 공란; dev 10행만 정답), FinMathBench(PDF·arXiv 없음, GitHub 생성기·샘플 xlsx만).</li>
    <li><b>논문 내부 불일치·라이선스 불일치</b>는 각 섹션에 적었다(예: ClimaQA Gold 문항 수 Table 2 vs p.8, DMind 3,543 vs HF 3,538, OceanBench Table 1 합 11,426 vs HF 10,344, BizFinBench HF cc-by-4.0 vs GitHub CC BY-NC).</li>
    <li><b>범위.</b> 1차 광역 조사는 WebSearch 한도 안에서 수행했고, 심층 정독은 arXiv·Anthology·PMLR·GitHub raw·HF datasets-server 로 직접 내려받아 진행했다. WWW·IJCAI 전용 패스는 돌리지 못했다.</li>
  </ul>
</section>
</main>''')

    extra_css = '''
  section.paper.g1{border-top-color:var(--p1)}section.paper.g2{border-top-color:var(--p2)}section.paper.g3{border-top-color:var(--p3)}section.paper.g4{border-top-color:var(--p4)}
  .g1 .spec,.g1 .contrib li::before{background:var(--p1)}.g2 .spec,.g2 .contrib li::before{background:var(--p2)}
  .g3 .spec,.g3 .contrib li::before{background:var(--p3)}.g4 .spec,.g4 .contrib li::before{background:var(--p4)}
  section.grouphead{border-left:8px solid var(--accent);background:#f8fafc}
  section.grouphead.g1{border-left-color:var(--p1)}section.grouphead.g2{border-left-color:var(--p2)}section.grouphead.g3{border-left-color:var(--p3)}section.grouphead.g4{border-left-color:var(--p4)}
  .pill.tagA{color:#475569;background:#f1f5f9;border-color:#cbd5e1}.pill.tagB{color:#0e7490;background:#e6f6fa;border-color:#bae6e6}
  .pill.tagC{color:#b91c1c;background:#fdecec;border-color:#fbcaca;font-weight:800}.pill.tagCl{color:#b45309;background:#fef3e2;border-color:#fad9a5}
  .meta-box{display:grid;grid-template-columns:1fr;gap:6px 18px;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:10px 0 6px;font-size:13.4px}
  .meta-box>div{display:grid;grid-template-columns:96px minmax(0,1fr);gap:8px}.meta-box>div>span{min-width:0;overflow-wrap:anywhere}.meta-box b{color:#475569;font-weight:700}
  .qbox{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Noto Sans KR",monospace;font-size:12.6px}
  .mynote{margin:16px 0 4px;background:#fffaf0;border:1px solid #f0d79c;border-left:5px solid #c79a3a;border-radius:10px;padding:4px 18px 15px}
  .mynote .mnhead{display:inline-flex;align-items:center;gap:6px;background:#c79a3a;color:#fff;font-size:12px;font-weight:800;border-radius:0 0 9px 9px;padding:4px 13px;margin-bottom:9px}
  .mynote ul{margin:6px 0 2px 4px;list-style:none;padding:0}.mynote li{margin:8px 0;font-size:13.9px;padding-left:22px;position:relative;line-height:1.66}
  .mynote li::before{content:"";position:absolute;left:4px;top:9px;width:8px;height:8px;border-radius:2px;background:#c79a3a;transform:rotate(45deg)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.3px;background:#f1f5f9;border-radius:4px;padding:1px 5px}
  ul.tight{margin:6px 0 0 18px}ul.tight li{margin:5px 0}
'''
    page = f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=1200, initial-scale=1">
<title>LLM 설계 QA 벤치마크 심층분석 — 온체인·금융·희소 도메인 24편 (PDF 정독 + 데이터셋 실열람)</title>
<style>{CSS}{extra_css}</style>
</head>
<body>
{"".join(body)}
</body></html>'''
    open(OUT, "w", encoding="utf-8").write(page)
    print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB", "notes:", len(notes))

if __name__ == "__main__":
    build()
