#!/usr/bin/env python3
"""onchain.damilab.cc 진행보드에 새 항목을 만들고 HTML을 첨부한다.
사용: ONCHAIN_EDIT_KEY='편집비밀번호' python3 upload_progress.py
     (또는 python3 upload_progress.py 실행 후 프롬프트에 입력)"""
import os, sys, json, getpass, urllib.request, mimetypes, uuid
BASE = "https://onchain.damilab.cc"
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "틱데이터_백테스트_심층분석_0909.html")
MD   = os.path.join(HERE, "tick-backtest-survey.md")
BODY = open(os.path.join(HERE, "progress_body.md"), encoding="utf-8").read()
key = os.environ.get("ONCHAIN_EDIT_KEY") or getpass.getpass("편집 비밀번호: ")

def req(method, path, data=None, headers=None):
    h = {"X-Edit-Key": key, "User-Agent": "Mozilla/5.0 (onchain-upload-script)"}; h.update(headers or {})
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode())

item = req("POST", "/api/progress", json.dumps({
    "title": "틱데이터 백테스트 심층분석 — 전통 금융·실행·AI·방법론·엔진·한국",
    "date": "2026-09-09", "status": "DONE", "body": BODY}).encode(),
    {"Content-Type": "application/json"})
print("created item:", item.get("id"))

boundary = uuid.uuid4().hex; parts = []
for fp in [HTML, MD]:
    name = os.path.basename(fp); ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                  f'Content-Type: {ctype}\r\n\r\n').encode() + open(fp, "rb").read() + b"\r\n")
body = b"".join(parts) + f"--{boundary}--\r\n".encode()
res = req("POST", f"/api/progress/{item['id']}/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("attachments:", [a["name"] for a in res.get("attachments", [])])
print(f"확인: {BASE}/progress.html#item-{item['id']}")
