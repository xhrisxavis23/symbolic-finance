#!/usr/bin/env python3
"""notes/*.json + manifest.json -> 요약 마크다운 (HTML 의 동반 문서)"""
import json, os, re, html

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "tick-backtest-survey.md")
m = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
notes = {f[:-5]: json.load(open(os.path.join(ROOT, "notes", f), encoding="utf-8"))
         for f in sorted(os.listdir(os.path.join(ROOT, "notes"))) if f.endswith(".json")}

GROUP_NAMES = {
    "g1": "전통 금융 · 시장미시구조 실증",
    "g2": "최적 실행 · 마켓메이킹 · 호가창 시뮬레이션",
    "g3": "AI · 딥러닝 · 강화학습 트레이딩",
    "g4": "백테스트 방법론 · 과최적화 · 다중검정",
    "g5": "오픈소스 틱 백테스트 엔진",
    "g6": "대한민국 — 증권사 공개 코드 · KRX 틱 연구",
}
LADDER = {
    "L0": "L0 시뮬레이션 없음", "L1": "L1 모형 몬테카를로", "L2": "L2 무영향 재생",
    "L3": "L3 재생+실호가·비용", "L4": "L4 재생+큐·지연", "L5": "L5 상호작용 시뮬", "L6": "L6 실집행",
}


def strip_html(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    return html.unescape(s).strip()


def cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def links(N):
    L = N.get("links", {}) or {}
    out = []
    for k, label in [("arxiv", "arXiv"), ("pdf", "PDF"), ("proceedings", "DOI"), ("ssrn", "SSRN"),
                     ("github", "GitHub"), ("docs", "문서"), ("website", "웹"), ("data", "데이터")]:
        if L.get(k):
            out.append(f"[{label}]({L[k]})")
    for x in L.get("github_extra") or []:
        out.append(f"[GitHub]({x})")
    return " · ".join(out) or "—"


lines = []
lines.append("# " + m["sections"]["title"])
lines.append("")
lines.append("- 조사일: 2026-09-09")
for s in m["sections"]["submeta"]:
    lines.append(f"- {s}")
lines.append("")
lines.append("## 분류 축 — 백테스트 사실성 사다리")
lines.append("")
lines.append("| 등급 | 뜻 | 대표 사례 |")
lines.append("|---|---|---|")
LD = {"L0": "회귀·이벤트 스터디. 전략을 굴리지 않는다", "L1": "합성 호가창 위 몬테카를로",
      "L2": "과거 틱 재생 · 중간가/종가 즉시 체결", "L3": "실 BID/ASK 체결 + 수수료 (큐 없음)",
      "L4": "큐 위치를 세고 지연을 모델링", "L5": "다중 에이전트 · 자기영향", "L6": "실거래 체결 기록"}
for r in ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]:
    ex = m["sections"]["ladder_examples"].get(r, "")
    lines.append(f"| **{LADDER[r]}** | {LD[r]} | {cell(ex)} |")
lines.append("")

lines.append("## 종합 요약표")
lines.append("")
lines.append("| 군 | 연구·도구 | venue | 해상도 | 데이터 | 체결 가정 | 비용 | 등급 | 공개 코드 |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for gid, g in m["groups"].items():
    for k in g["deep"] + g["brief"]:
        N = notes.get(k)
        if not N:
            continue
        bt = N.get("backtest", {}) or {}
        rung = (bt.get("rung") or "")[:2].upper()
        lines.append("| {} | **{}** | {} | {} | {} | {} | {} | {} | {} |".format(
            GROUP_NAMES[gid].split(" ")[0], cell(N.get("title", k).split(" — ")[0]), cell(N.get("venue", "")),
            cell(N.get("resolution", "")),
            cell(bt.get("data_short", "")), cell(bt.get("fill_short", "")), cell(bt.get("cost_short", "")),
            LADDER.get(rung, ""), links(N)))
lines.append("")

lines.append("## 종합 판정")
lines.append("")
lines.append(strip_html(m["sections"]["verdict"]))
lines.append("")

for gid, g in m["groups"].items():
    if not (g["deep"] or g["brief"]):
        continue
    lines.append(f"## {GROUP_NAMES[gid]}")
    lines.append("")
    lines.append(strip_html(g["intro"]))
    lines.append("")
    for k in g["deep"] + g["brief"]:
        N = notes.get(k)
        if not N:
            continue
        bt = N.get("backtest", {}) or {}
        rung = (bt.get("rung") or "")[:2].upper()
        lines.append(f"### {N.get('title', k)}")
        lines.append("")
        lines.append(f"- **venue** {N.get('venue','')} · **해상도** {N.get('resolution','')} · **등급** {LADDER.get(rung,'')}")
        lines.append(f"- **링크** {links(N)}")
        if N.get("data_scope"):
            lines.append(f"- **데이터** {N['data_scope']}")
        for fk, label in [("design", "설계"), ("fill", "체결"), ("cost", "비용"),
                          ("latency", "지연"), ("selfimpact", "자기영향"),
                          ("validation", "검증"), ("metrics", "지표"), ("gap", "다루지 않는 것")]:
            if bt.get(fk):
                lines.append(f"- **{label}** {bt[fk]}")
        for q in bt.get("quotes") or []:
            lines.append("")
            lines.append("```")
            lines.append(q)
            lines.append("```")
        if N.get("code"):
            lines.append("- **공개 코드** " + " / ".join(N["code"]))
        if N.get("oneline"):
            lines.append(f"- **한 줄** {N['oneline']}")
        lines.append("")

lines.append("## 우리 코드와의 대조")
lines.append("")
lines.append(strip_html(m["sections"]["local_section"]))
lines.append("")
lines.append("## 검증 한계")
lines.append("")
lines.append(strip_html(m["sections"]["caveats_section"]))

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
