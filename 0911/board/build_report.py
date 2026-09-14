#!/usr/bin/env python3
"""진행보드 첨부용 HTML 을 만든다 — 진행보드의 조사 보고서 양식(QA 벤치마크 0905 · 틱 백테스트 0909 와 같은 CSS)을 그대로 쓴다.

  CSS    : template.css (없으면 ../../tick_backtest_survey_0909/틱데이터_백테스트_심층분석_0909.html 에서 한 번 뽑아 저장)
  본문   : report_body.html
  출력   : ../symbolic_distillation_0911_report.html

만들고 나서 점검한다: 목차 링크가 전부 실제 id 로 이어지는가, 태그 짝이 맞는가, 남은 자리표시자가 없는가.
"""
import html.parser
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_CSS = os.path.join(HERE, "template.css")
SOURCE_HTML = os.path.join(HERE, "..", "..", "tick_backtest_survey_0909", "틱데이터_백테스트_심층분석_0909.html")
BODY = os.path.join(HERE, "report_body.html")
OUT = os.path.join(HERE, "..", "symbolic_distillation_0911_report.html")
TITLE = "시장 미시구조 심볼릭 증류 — 교사 자체 경제성 · 3일 전체 학습 · 틱 크기별 종목군 · 데이터 양 곡선 (0909~0914)"

if not os.path.exists(TEMPLATE_CSS):
    src = open(SOURCE_HTML, encoding="utf-8").read()
    css = re.search(r"<style[^>]*>(.*?)</style>", src, re.S).group(1)
    open(TEMPLATE_CSS, "w", encoding="utf-8").write(css)
css = open(TEMPLATE_CSS, encoding="utf-8").read()
body = open(BODY, encoding="utf-8").read()

# 양식에 없는 규칙은 이것 하나다: 숨어 있는 용어 설명 상자(.term::after)가 오른쪽 끝에서 문서 폭을 늘려
# 가로 스크롤이 생기는 것을 막는다(양식 원본에도 있는 현상). clip 은 스크롤 컨테이너를 만들지 않아 상단 고정 바가 그대로 동작한다.
EXTRA_CSS = "\n  /* 0911 추가: 용어 설명 상자로 인한 가로 스크롤 방지 */\n  html,body{overflow-x:clip}\n"

doc = ("<!DOCTYPE html>\n<html lang=\"ko\"><head><meta charset=\"UTF-8\">\n"
       "<meta name=\"viewport\" content=\"width=1200, initial-scale=1\">\n"
       f"<title>{TITLE}</title>\n<style>{css}{EXTRA_CSS}</style>\n</head>\n<body>\n{body}\n</body></html>\n")


class Check(html.parser.HTMLParser):
    VOID = {"meta", "link", "br", "hr", "img", "input", "source", "col", "wbr"}

    def __init__(self):
        super().__init__()
        self.ids, self.hrefs, self.stack, self.errors = set(), [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"):
            self.hrefs.append(a["href"][1:])
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append(f"닫는 태그 불일치 </{tag}> at {self.getpos()} (열린 것: {self.stack[-1] if self.stack else None})")
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break
        else:
            self.stack.pop()


c = Check()
c.feed(doc)
problems = list(c.errors)
problems += [f"목차 링크 대상 없음: #{h}" for h in c.hrefs if h not in c.ids]
problems += [f"닫히지 않은 태그: {t}" for t in c.stack]
if re.search(r"__[A-Z]+__|TODO|TBD", body):
    problems.append("자리표시자가 남아 있다")
if problems:
    print("점검 실패:")
    for p in problems:
        print(" -", p)
    sys.exit(1)

open(OUT, "w", encoding="utf-8").write(doc)
print(f"작성 {os.path.normpath(OUT)} ({len(doc.encode())/1024:.0f} KB) · 절 {body.count('<section')}개 · "
      f"목차 링크 {len(c.hrefs)}개 전부 연결 · 태그 짝 정상")
