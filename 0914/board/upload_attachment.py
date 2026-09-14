#!/usr/bin/env python3
"""진행보드 항목 a2adef9ecfc4 에 0914 보고서 HTML 을 첨부로 **추가**한다.

제목 · 날짜 · 상태 · 본문 · 기존 첨부는 건드리지 않는다. 같은 이름의 첨부가 이미 있으면 멈춘다
(`--replace` 를 주면 그 파일 하나만 지우고 다시 올린다).

    ONCHAIN_EDIT_KEY='편집비밀번호' python3 upload_attachment.py [--replace]

올린 뒤 대조: 제목·날짜·상태·본문 그대로 · 기존 첨부 이름·크기 그대로 · 새 첨부 기록 크기 = 로컬 ·
서버에서 내려받은 파일이 Cloudflare 방문 통계 스크립트 줄을 빼면 로컬과 같다.
"""
import hashlib
import json
import mimetypes
import os
import sys
import urllib.parse
import urllib.request
import uuid

BASE, ITEM = "https://onchain.damilab.cc", "a2adef9ecfc4"
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.normpath(os.path.join(HERE, "..", "symbolic_distillation_0914_report.html"))
key = os.environ.get("ONCHAIN_EDIT_KEY") or sys.exit("ONCHAIN_EDIT_KEY 환경변수가 없다")
replace = "--replace" in sys.argv[1:]


def req(method, path, data=None, headers=None, raw=False):
    h = {"User-Agent": "Mozilla/5.0 (onchain-upload-script)"}
    if method != "GET":
        h["X-Edit-Key"] = key
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(BASE + path, data=data, method=method, headers=h), timeout=180) as r:
        b = r.read()
        return b if raw else json.loads(b.decode())


def item():
    return next(i for i in req("GET", "/api/progress") if i["id"] == ITEM)


cur = item()
name = os.path.basename(HTML)
before = {a["name"]: a["size"] for a in cur["attachments"]}
print("현재:", cur["title"], cur["date"], cur["status"], "| 첨부", list(before.items()))
if name in before:
    if not replace:
        sys.exit(f"같은 이름의 첨부 {name} 가 이미 있다 — 덮어쓰지 않는다 (--replace)")
    upd = req("DELETE", f"/api/progress/{ITEM}/files/{urllib.parse.quote(name)}")
    print("같은 이름 첨부 삭제 후:", [a["name"] for a in upd.get("attachments", [])])
    before.pop(name)

local = open(HTML, "rb").read()
boundary = uuid.uuid4().hex
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
        f'Content-Type: {mimetypes.guess_type(HTML)[0]}\r\n\r\n').encode() + local + f"\r\n--{boundary}--\r\n".encode()
res = req("POST", f"/api/progress/{ITEM}/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("올린 뒤 첨부:", [(a["name"], a["size"]) for a in res.get("attachments", [])])

after = item()
att = next(a for a in after["attachments"] if a["name"] == name)
served = req("GET", urllib.parse.quote(att["url"]), raw=True)
stripped = b"\n".join(l for l in served.split(b"\n") if b"static.cloudflareinsights.com/beacon" not in l)
checks = {
    "제목·날짜·상태 유지": all(after[k] == cur[k] for k in ("title", "date", "status")),
    "본문 유지": after["body"] == cur["body"],
    "기존 첨부 유지": {a["name"]: a["size"] for a in after["attachments"] if a["name"] != name} == before,
    "기록 크기 = 로컬": att["size"] == len(local),
    "내려받은 내용 = 로컬(통계 스크립트 줄 제외)": hashlib.md5(stripped).hexdigest() == hashlib.md5(local).hexdigest(),
}
print("검증:", " · ".join(f"{k} {v}" for k, v in checks.items()), f"· 크기 {att['size']} · 첨부 순서 {[a['name'] for a in after['attachments']]}")
print(f"파일: {BASE}{att['url']}")
print(f"항목: {BASE}/progress.html#item-{ITEM}")
if not all(checks.values()):
    sys.exit(1)
