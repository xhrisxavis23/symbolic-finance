#!/usr/bin/env python3
# notes/*.json -> 사이트 양식 HTML 리포트 (틱데이터 백테스트 조사)
import json, os, re, html, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CSS = open(os.path.join(ROOT, "site", "ex3.css"), encoding="utf-8").read()
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "틱데이터_백테스트_심층분석_0909.html")
DATE = "2026-09-09"

# ---- 백테스트 사실성 사다리 (이 리포트의 분류 축) ---------------------------------
LADDER = {
    "L0": ("L0 시뮬레이션 없음", "회귀·이벤트 스터디. 전략을 굴리지 않고 계수·상관·인과만 본다"),
    "L1": ("L1 모형 몬테카를로", "합성 호가창(포아송·해석해) 위에서 정책을 굴린다. 실데이터 체결 없음"),
    "L2": ("L2 무영향 재생", "과거 틱을 되감으며 중간가·종가에 즉시 체결됐다고 가정. 큐·지연 없음"),
    "L3": ("L3 재생 + 실호가·비용", "실제 BID/ASK 로 체결하고 수수료·세금을 뺀다. 큐 위치는 없음"),
    "L4": ("L4 재생 + 큐·지연", "내 주문 앞 물량을 세고, 그 물량이 줄어야 체결된다. 지연을 모델링"),
    "L5": ("L5 상호작용 시뮬", "다중 에이전트. 내 주문이 남의 행동을 바꾼다 (자기영향 포함)"),
    "L6": ("L6 실집행", "실거래 체결 기록 자체가 데이터"),
}
LADDER_CLASS = {"L0": "tagL0", "L1": "tagL1", "L2": "tagL2", "L3": "tagL3",
                "L4": "tagL4", "L5": "tagL5", "L6": "tagL6"}

# ---- 데이터 해상도 (백테스트가 실제로 굴러간 단위) ----------------------------------
RESOLUTION = {
    "틱·메시지": ("resTick", "호가창 메시지 또는 체결 단위 원장 위에서 굴렸다"),
    "초~분봉": ("resSub", "틱을 확보했더라도 백테스트는 초·분 격자에서 굴렸다"),
    "일봉 이상": ("resDay", "일봉 또는 그보다 성긴 단위"),
    "합성·모형": ("resSyn", "실데이터가 아니라 모형이 만든 경로"),
    "실집행 기록": ("resExec", "시뮬레이션이 아니라 실제로 체결된 주문 기록"),
    "해당 없음": ("resNA", "백테스트를 굴리지 않는 항목 — 방법론·도구·데이터 유통"),
}


def res_pill(r):
    r = (r or "").strip()
    if r not in RESOLUTION:
        return ""
    cls, tip = RESOLUTION[r]
    return f'<span class="pill {cls}" title="{esc(tip)}">{esc(r)}</span>' 

GROUPS = [
    {"id": "g1", "cls": "g1", "name": "전통 금융 · 시장미시구조 실증", "short": "전통 금융",
     "deep": [], "brief": [], "intro": ""},
    {"id": "g2", "cls": "g2", "name": "최적 실행 · 마켓메이킹 · 호가창 시뮬레이션", "short": "실행·MM",
     "deep": [], "brief": [], "intro": ""},
    {"id": "g3", "cls": "g3", "name": "AI · 딥러닝 · 강화학습 트레이딩", "short": "AI 트레이딩",
     "deep": [], "brief": [], "intro": ""},
    {"id": "g4", "cls": "g4", "name": "백테스트 방법론 · 과최적화 · 다중검정", "short": "방법론",
     "deep": [], "brief": [], "intro": ""},
    {"id": "g5", "cls": "g5", "name": "오픈소스 틱 백테스트 엔진", "short": "엔진",
     "deep": [], "brief": [], "intro": ""},
    {"id": "g6", "cls": "g6", "name": "대한민국 — 증권사 공개 코드 · KRX 틱 연구", "short": "한국",
     "deep": [], "brief": [], "intro": ""},
]

MANIFEST = os.path.join(ROOT, "manifest.json")
if os.path.exists(MANIFEST):
    _m = json.load(open(MANIFEST, encoding="utf-8"))
    _by_id = {g["id"]: g for g in GROUPS}
    for gid, spec in _m.get("groups", {}).items():
        _by_id[gid].update(spec)
    EXTRA = _m.get("sections", {})
else:
    EXTRA = {}


def esc(s):
    return html.escape(str(s if s is not None else ""), quote=False)


def md(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
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


def rung_pill(r):
    r = (r or "").strip().upper()[:2]
    if r not in LADDER:
        return ""
    return f'<span class="pill {LADDER_CLASS[r]}" title="{esc(LADDER[r][1])}">{esc(LADDER[r][0])}</span>'


def link(u, text=None):
    if not u:
        return ""
    return f'<a href="{esc(u)}" target="_blank" rel="noopener">{esc(text or u)}</a>'


def links_line(L):
    parts = []
    if L.get("arxiv"):
        parts.append(link(L["arxiv"], "arXiv " + L["arxiv"].rstrip("/").rsplit("/", 1)[-1]))
    if L.get("pdf"):
        parts.append(link(L["pdf"], "PDF"))
    if L.get("proceedings"):
        parts.append(link(L["proceedings"], "Proceedings/DOI"))
    if L.get("ssrn"):
        parts.append(link(L["ssrn"], "SSRN"))
    return " · ".join(parts) or "—"


def data_line(L):
    parts = []
    if L.get("github"):
        parts.append(link(L["github"], "GitHub " + re.sub(r"^https?://github\.com/", "", L["github"]).rstrip("/")))
    for extra in L.get("github_extra") or []:
        parts.append(link(extra, re.sub(r"^https?://github\.com/", "GitHub ", extra).rstrip("/")))
    if L.get("docs"):
        parts.append(link(L["docs"], "문서"))
    if L.get("website"):
        parts.append(link(L["website"], "웹사이트"))
    if L.get("data"):
        parts.append(link(L["data"], "데이터"))
    return " · ".join(parts) or '<span class="pill">공개 코드 없음</span>'


BT_FIELDS = [
    ("data", "데이터", "어느 거래소·기간·해상도의 틱을 썼는가"),
    ("design", "설계", "회귀인가, 재생인가, 상호작용 시뮬인가"),
    ("fill", "체결 가정", "무엇이 체결을 결정하는가 (큐 위치·부분체결 포함)"),
    ("cost", "비용", "스프레드·수수료·세금·리베이트·시장충격"),
    ("latency", "지연", "피드 지연 · 주문 지연"),
    ("selfimpact", "자기영향", "내 주문이 시장을 바꾸는가"),
    ("validation", "검증", "표본 분리 · 다중검정 · 재현"),
    ("metrics", "지표", "무엇으로 좋다고 말하는가"),
]


def paper_section(n, key, N, gcls):
    L = N.get("links", {}) or {}
    bt = N.get("backtest", {}) or {}
    tags = "".join(f'<span class="pill tag{gcls[-1]}">{esc(t)}</span> ' for t in (N.get("tags") or [])[:5])
    h = [f'<section id="p{n}" class="paper {gcls}">']
    h.append(f'<h2>{esc(N.get("title", key))}'
             f'<span class="vtag">{esc(N.get("authors_short", ""))} · {esc(N.get("venue", ""))}</span></h2>')
    h.append('<div class="meta-box">')
    h.append(f'<div><b>유형</b><span>{esc(N.get("type", ""))} · {esc(N.get("domain", ""))}</span></div>')
    h.append(f'<div><b>원문</b><span>{links_line(L)}</span></div>')
    h.append(f'<div><b>코드/데이터</b><span>{data_line(L)} '
             f'<span class="small">({esc(L.get("verified", ""))})</span></span></div>')
    if N.get("venue_verified"):
        h.append(f'<div><b>venue 검증</b><span>{esc(N["venue_verified"])}</span></div>')
    h.append(f'<div><b>데이터 범위</b><span>{md(N.get("data_scope", ""))}</span></div>')
    if tags:
        h.append(f'<div><b>핵심 태그</b><span>{tags}</span></div>')
    h.append(f'<div><b>데이터 해상도</b><span>{res_pill(N.get("resolution"))} '
             f'<span class="small">{md(N.get("resolution_note", ""))}</span></span></div>')
    h.append(f'<div><b>백테스트 등급</b><span>{rung_pill(bt.get("rung"))} '
             f'<span class="small">{md(bt.get("rung_reason", ""))}</span></span></div>')
    h.append('</div>')

    ez = N.get("easy") or {}
    if ez:
        h.append('<div class="easy"><div class="ezhead">쉽게 말하면</div>')
        if ez.get("problem"):
            h.append(f'<p><b>무슨 문제?</b> {md(ez["problem"])}</p>')
        if ez.get("how"):
            h.append(f'<p><b>어떻게?</b> {md(ez["how"])}</p>')
        if ez.get("analogy"):
            h.append(f'<div class="analogy"><b>비유.</b> {md(ez["analogy"])}</div>')
        h.append('</div>')

    if N.get("contributions"):
        h.append('<div class="deepbar">핵심 요약 · 문제의식과 기여</div><ol class="contrib">')
        for c in N["contributions"]:
            h.append(f'<li>{md(c)}</li>')
        h.append('</ol>')

    # ---- 백테스트 집중 ----
    h.append('<div class="btbar">백테스트 깊이 보기 — 이 논문은 무엇을 어떻게 굴렸나</div>')
    h.append('<div class="table-scroll"><table class="bt"><tbody>')
    for k, label, hint in BT_FIELDS:
        if bt.get(k):
            h.append(f'<tr><th>{esc(label)}<span class="hint">{esc(hint)}</span></th>'
                     f'<td>{md(bt[k])}</td></tr>')
    h.append('</tbody></table></div>')
    for q in bt.get("quotes") or []:
        h.append(f'<div class="qbox">{esc(q)}</div>')
    if bt.get("gap"):
        h.append(f'<div class="warn"><b>백테스트가 다루지 않는 것.</b> {md(bt["gap"])}</div>')

    if N.get("code"):
        h.append('<div class="deepbar">공개된 코드</div><ul class="tight">')
        for c in N["code"]:
            h.append(f'<li>{md(c)}</li>')
        h.append('</ul>')

    if N.get("limitations"):
        h.append(f'<div class="warn"><b>한계.</b> {md(N["limitations"])}</div>')
    if N.get("oneline"):
        h.append(f'<div class="oneline"><b>한 줄 요약.</b> {md(N["oneline"])}</div>')
    if N.get("implications"):
        h.append('<div class="mynote"><div class="mnhead">✍︎ 우리 틱 백테스트에 주는 시사점</div><ul>')
        for x in N["implications"]:
            h.append(f'<li>{md(x)}</li>')
        h.append('</ul></div>')
    h.append('</section>')
    return "\n".join(h)


def brief_card(key, N):
    L = N.get("links", {}) or {}
    bt = N.get("backtest", {}) or {}
    s = f'<div class="card"><h4>{esc(N.get("title", key))} <span class="small">{esc(N.get("venue", ""))}</span></h4>'
    s += f'<p class="small">{res_pill(N.get("resolution"))} {esc(N.get("authors_short", ""))} · {links_line(L)} · {data_line(L)}</p>'
    if bt.get("design"):
        s += f'<p>{md(bt["design"])}</p>'
    if bt.get("fill"):
        s += f'<p><b>체결.</b> {md(bt["fill"])}</p>'
    s += f'<p>{rung_pill(bt.get("rung"))} {md(bt.get("rung_reason", ""))}</p>'
    if N.get("oneline"):
        s += f'<p><b>한 줄.</b> {md(N["oneline"])}</p>'
    return s + '</div>'


def build():
    notes = load_notes()
    order, n = [], 0
    for g in GROUPS:
        for k in g["deep"]:
            n += 1
            order.append((n, k, g["cls"]))
    num = {k: i for i, k, _ in order}

    # ---------- 요약표 ----------
    rows = []
    for g in GROUPS:
        for k in g["deep"] + g["brief"]:
            N = notes.get(k)
            if not N:
                rows.append(f'<tr><td colspan="9"><i>{esc(k)}: 노트 없음</i></td></tr>')
                continue
            bt = N.get("backtest", {}) or {}
            L = N.get("links", {}) or {}
            name = (f'<a href="#p{num[k]}">{esc(N.get("title", k))}</a>' if k in num
                    else esc(N.get("title", k)))
            rows.append(
                f'<tr><td><span class="pill tag{g["cls"][-1]}">{esc(g["short"])}</span></td>'
                f'<td><b>{name}</b></td><td>{esc(N.get("venue", ""))}</td>'
                f'<td>{res_pill(N.get("resolution"))}</td>'
                f'<td>{md(bt.get("data_short", ""))}</td>'
                f'<td>{md(bt.get("fill_short", ""))}</td>'
                f'<td>{md(bt.get("cost_short", ""))}</td>'
                f'<td>{rung_pill(bt.get("rung"))}</td><td>{data_line(L)}</td></tr>')
    summary_table = ('<div class="table-scroll"><table><thead><tr><th>군</th><th>연구·도구</th>'
                     '<th>venue</th><th>해상도</th><th>데이터</th><th>체결 가정</th><th>비용</th>'
                     '<th>등급</th><th>공개 코드</th></tr></thead><tbody>'
                     + "\n".join(rows) + '</tbody></table></div>')

    # ---------- 등급 분포 ----------
    from collections import Counter
    cnt = Counter()
    for g in GROUPS:
        for k in g["deep"] + g["brief"]:
            r = ((notes.get(k, {}).get("backtest", {}) or {}).get("rung") or "").strip().upper()[:2]
            if r in LADDER:
                cnt[r] += 1
    dist = " · ".join(f"{LADDER[r][0]} {cnt[r]}편" for r in ["L0", "L1", "L2", "L3", "L4", "L5", "L6"] if cnt[r])

    # ---------- nav ----------
    nav = ['<a href="#start">시작</a>', '<a href="#ladder" class="hl">백테스트 사다리</a>',
           '<a href="#summary" class="hl">종합 요약표</a>', '<a href="#verdict" class="hl">종합 판정</a>']
    for g in GROUPS:
        nav.append(f'<a href="#{g["id"]}">{esc(g["short"])}</a>')
    nav.append('<a href="#checklist" class="hl">체크리스트</a><a href="#local" class="hl">우리 코드 비교</a><a href="#caveats">검증 한계</a>')
    navgrid = []
    for g in GROUPS:
        navgrid.append(f'<span class="navhd">{esc(g["name"])}</span>')
        for k in g["deep"]:
            t = notes.get(k, {}).get("title", k)
            navgrid.append(f'<a href="#p{num[k]}">{num[k]} {esc(t[:40])}</a>')

    ladder_rows = "".join(
        f'<tr><th>{rung_pill(r)}</th><td>{esc(LADDER[r][1])}</td>'
        f'<td>{md(EXTRA.get("ladder_examples", {}).get(r, ""))}</td></tr>'
        for r in ["L0", "L1", "L2", "L3", "L4", "L5", "L6"])

    body = []
    body.append(f'''
<header class="topbar"><div class="wrap">
  <h1>{esc(EXTRA.get("h1", "틱데이터 백테스트 심층분석"))}</h1>
  <div class="submeta">
    {"".join(f'<span class="b">{esc(x)}</span>' for x in EXTRA.get("submeta", []))}
    <span class="b">{DATE}</span>
  </div>
  <nav class="nav">{"".join(nav)}
    <details class="navdetails"><summary>&#9656; 전체 목차</summary><div class="navgrid">{"".join(navgrid)}</div></details>
  </nav>
</div></header>
<main class="wrap">
<section id="start">
  <div class="how"><b>읽는 법.</b> 각 항목은 <b>메타(원문·코드 링크·백테스트 등급) → 쉽게 말하면 → 핵심 기여 →
  백테스트 깊이 보기(데이터·설계·체결·비용·지연·자기영향·검증·지표) → 공개 코드 → 한계 → 한 줄 → 우리 코드에 주는 시사점</b>
  순서다. 링크는 모두 새 탭에서 열린다. 확인하지 못한 것은 "미확인"으로 남겼다.</div>
  {EXTRA.get("lead", "")}
</section>

<section id="ladder">
  <h2>분류 축 — 백테스트 사실성 사다리<span class="vtag">"틱데이터를 썼다"는 말은 등급을 말해주지 않는다. 무엇이 체결을 결정하는가가 등급이다</span></h2>
  <div class="table-scroll"><table><thead><tr><th style="width:190px">등급</th><th style="width:340px">뜻</th><th>대표 사례</th></tr></thead><tbody>
  {ladder_rows}
  </tbody></table></div>
  {EXTRA.get("ladder_note", "")}
</section>

<section id="summary">
  <h2>종합 요약표<span class="vtag">군 · venue · 데이터 · 체결 가정 · 비용 · 등급 · 공개 코드</span></h2>
  {summary_table}
</section>

<section id="verdict">
  <h2>{esc(EXTRA.get("verdict_title", "종합 판정"))}</h2>
  {EXTRA.get("verdict", "")}
  <div class="oneline"><b>등급 분포.</b> {dist}</div>
</section>
''')

    for g in GROUPS:
        body.append(f'<section id="{g["id"]}" class="grouphead {g["cls"]}">'
                    f'<h2>{esc(g["name"])}<span class="vtag">심층 {len(g["deep"])}편 · 요약 {len(g["brief"])}편</span></h2>'
                    f'<p class="lead">{g["intro"]}</p></section>')
        for k in g["deep"]:
            N = notes.get(k)
            if not N:
                body.append(f'<section id="p{num[k]}" class="paper {g["cls"]}">'
                            f'<h2>{esc(k)}<span class="vtag">노트 없음</span></h2></section>')
                continue
            body.append(paper_section(num[k], k, N, g["cls"]))
        briefs = [k for k in g["brief"] if k in notes]
        if briefs:
            body.append(f'<section class="{g["cls"]}"><h2>{esc(g["short"])} · 요약 카드'
                        f'<span class="vtag">심층 대상에서 뺐지만 지형을 채우는 것들</span></h2><div class="grid">'
                        + "".join(brief_card(k, notes[k]) for k in briefs) + '</div></section>')

    body.append(EXTRA.get("checklist_section", ""))
    body.append(EXTRA.get("local_section", ""))
    body.append(EXTRA.get("caveats_section", ""))
    body.append('</main>')

    extra_css = '''
  section.paper.g1{border-top-color:var(--p1)}section.paper.g2{border-top-color:var(--p2)}
  section.paper.g3{border-top-color:var(--p3)}section.paper.g4{border-top-color:var(--p4)}
  section.paper.g5{border-top-color:#0369a1}section.paper.g6{border-top-color:#be123c}
  section.grouphead{border-left:8px solid var(--accent);background:#f8fafc}
  section.grouphead.g1{border-left-color:var(--p1)}section.grouphead.g2{border-left-color:var(--p2)}
  section.grouphead.g3{border-left-color:var(--p3)}section.grouphead.g4{border-left-color:var(--p4)}
  section.grouphead.g5{border-left-color:#0369a1}section.grouphead.g6{border-left-color:#be123c}
  .g1 .spec,.g1 .contrib li::before{background:var(--p1)}.g2 .spec,.g2 .contrib li::before{background:var(--p2)}
  .g3 .spec,.g3 .contrib li::before{background:var(--p3)}.g4 .spec,.g4 .contrib li::before{background:var(--p4)}
  .g5 .spec,.g5 .contrib li::before{background:#0369a1}.g6 .spec,.g6 .contrib li::before{background:#be123c}
  .pill.tag1{color:#5b21b6;background:#f3eeff;border-color:#ddd0ff}
  .pill.tag2{color:#0e7490;background:#e6f6fa;border-color:#bae6e6}
  .pill.tag3{color:#b91c1c;background:#fdecec;border-color:#fbcaca}
  .pill.tag4{color:#b45309;background:#fef3e2;border-color:#fad9a5}
  .pill.tag5{color:#0369a1;background:#e0f2fe;border-color:#bae6fd}
  .pill.tag6{color:#be123c;background:#ffe4e6;border-color:#fecdd3}
  .pill.resTick{color:#065f46;background:#d1fae5;border-color:#6ee7b7;font-weight:800}
  .pill.resSub{color:#92400e;background:#fef3c7;border-color:#fcd34d;font-weight:700}
  .pill.resDay{color:#9f1239;background:#ffe4e6;border-color:#fda4af;font-weight:700}
  .pill.resSyn{color:#5b21b6;background:#ede9fe;border-color:#c4b5fd;font-weight:700}
  .pill.resExec{color:#1e293b;background:#e2e8f0;border-color:#94a3b8;font-weight:700}
  .pill.resNA{color:#64748b;background:#f8fafc;border-color:#e2e8f0}
  .pill.tagL0{color:#64748b;background:#f1f5f9;border-color:#cbd5e1}
  .pill.tagL1{color:#7c3aed;background:#f5f3ff;border-color:#ddd6fe}
  .pill.tagL2{color:#b45309;background:#fffbeb;border-color:#fde68a}
  .pill.tagL3{color:#0e7490;background:#ecfeff;border-color:#a5f3fc}
  .pill.tagL4{color:#15803d;background:#f0fdf4;border-color:#bbf7d0;font-weight:800}
  .pill.tagL5{color:#b91c1c;background:#fef2f2;border-color:#fecaca;font-weight:800}
  .pill.tagL6{color:#1e293b;background:#e2e8f0;border-color:#94a3b8;font-weight:800}
  .meta-box{display:grid;grid-template-columns:1fr;gap:6px 18px;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:10px 0 6px;font-size:13.4px}
  .meta-box>div{display:grid;grid-template-columns:104px minmax(0,1fr);gap:8px}
  .meta-box>div>span{min-width:0;overflow-wrap:anywhere}.meta-box b{color:#475569;font-weight:700}
  .btbar{margin:18px 0 8px;background:#0f766e;color:#fff;font-size:12.5px;font-weight:800;
         border-radius:7px;padding:6px 14px;display:inline-block;letter-spacing:.02em}
  table.bt th{width:132px;vertical-align:top;text-align:left;background:#f6f9fc;color:#334155;font-size:12.6px}
  table.bt th .hint{display:block;font-weight:500;color:#94a3b8;font-size:11px;line-height:1.45;margin-top:2px}
  table.bt td{font-size:13.6px;line-height:1.7}
  .qbox{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Noto Sans KR",monospace;font-size:12.6px}
  .mynote{margin:16px 0 4px;background:#fffaf0;border:1px solid #f0d79c;border-left:5px solid #c79a3a;border-radius:10px;padding:4px 18px 15px}
  .mynote .mnhead{display:inline-flex;align-items:center;gap:6px;background:#c79a3a;color:#fff;font-size:12px;font-weight:800;border-radius:0 0 9px 9px;padding:4px 13px;margin-bottom:9px}
  .mynote ul{margin:6px 0 2px 4px;list-style:none;padding:0}
  .mynote li{margin:8px 0;font-size:13.9px;padding-left:22px;position:relative;line-height:1.66}
  .mynote li::before{content:"";position:absolute;left:4px;top:9px;width:8px;height:8px;border-radius:2px;background:#c79a3a;transform:rotate(45deg)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.3px;background:#f1f5f9;border-radius:4px;padding:1px 5px}
  ul.tight{margin:6px 0 0 18px}ul.tight li{margin:5px 0}
  .navdetails{width:100%;margin-top:6px}
  .navdetails summary{cursor:pointer;color:#93c5fd;font-size:11.5px;list-style:none}
  .navdetails summary::-webkit-details-marker{display:none}
  .navgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px 14px;margin-top:7px;
           padding:9px 11px;background:rgba(255,255,255,.05);border-radius:8px}
  .navgrid a{color:#cbd5e1;font-size:11px;border:0;padding:1px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .navgrid .navhd{grid-column:1/-1;color:#f8fafc;font-size:11px;font-weight:800;margin-top:6px;
                  border-bottom:1px solid #334155;padding-bottom:2px}
  .diff{display:grid;grid-template-columns:150px 1fr 1fr;gap:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;font-size:13.4px}
  .diff>div{padding:10px 14px;border-bottom:1px solid var(--line);min-width:0;overflow-wrap:anywhere}
  .diff>div:nth-child(3n+1){background:#f6f9fc;font-weight:700;color:#334155}
  .diff>div:nth-child(3n+2){border-left:1px solid var(--line)}
  .diff>div:nth-child(3n+3){border-left:1px solid var(--line);background:#fffdf7}
  .diff .hd{background:#0f172a !important;color:#fff;font-weight:800}
  .same{color:#15803d;font-weight:700}.notsame{color:#b42318;font-weight:700}
'''
    page = f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=1200, initial-scale=1">
<title>{esc(EXTRA.get("title", "틱데이터 백테스트 심층분석"))}</title>
<style>{CSS}{extra_css}</style>
</head>
<body>
{"".join(body)}
</body></html>'''
    open(OUT, "w", encoding="utf-8").write(page)
    print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB", "notes:", len(notes))


if __name__ == "__main__":
    build()
