#!/usr/bin/env python3
"""진행보드 첨부용 0914 보고서 HTML 을 만든다 — 0911 보고서와 같은 양식(CSS)을 쓴다.

  CSS    : ../../0911/board/template.css (읽기 전용)
  본문   : report_body.html
  출력   : ../symbolic_distillation_0914_report.html

만들고 나서 점검한다: 목차 링크가 전부 실제 id 로 이어지는가, 태그 짝이 맞는가, 남은 자리표시자가 없는가.
"""
import html.parser
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_CSS = os.path.normpath(os.path.join(HERE, "..", "..", "0911", "board", "template.css"))
BODY = os.path.join(HERE, "report_body.html")
OUT = os.path.normpath(os.path.join(HERE, "..", "symbolic_distillation_0914_report.html"))
TITLE = "교사 자체는 돈을 버는가 — R² 가 아니라 손익으로 잰 딥러닝 교사 (시장 미시구조 심볼릭 증류 0914)"

css = open(TEMPLATE_CSS, encoding="utf-8").read()
body = open(BODY, encoding="utf-8").read()
# 0911 과 같은 추가 규칙: 용어 설명 상자로 인한 가로 스크롤 방지
EXTRA_CSS = "\n  /* 0911 추가: 용어 설명 상자로 인한 가로 스크롤 방지 */\n  html,body{overflow-x:clip}\n"

doc = ("<!DOCTYPE html>\n<html lang=\"ko\"><head><meta charset=\"UTF-8\">\n"
       "<meta name=\"viewport\" content=\"width=1200, initial-scale=1\">\n"
       f"<title>{TITLE}</title>\n<style>{css}{EXTRA_CSS}</style>\n</head>\n<body>\n{body}\n</body></html>\n")


class Check(html.parser.HTMLParser):
    VOID = {"meta", "link", "br", "hr", "img", "input", "source", "col", "wbr"}

    def __init__(self):
        super().__init__()
        self.ids, self.hrefs, self.stack, self.errors, self.sections = set(), [], [], [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "section":
            self.sections += 1
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
print(f"작성 {OUT} ({len(doc.encode()) // 1024} KB) · 절 {c.sections}개 · 목차 링크 {len(c.hrefs)}개 전부 연결 · 태그 짝 정상")
