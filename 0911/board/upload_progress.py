#!/usr/bin/env python3
"""onchain.damilab.cc 진행보드에 새 항목을 만들고 HTML·md 를 첨부한다 (0905·0909 업로드 스크립트와 같은 API).
사용: ONCHAIN_EDIT_KEY='편집비밀번호' python3 upload_progress.py
비밀번호는 파일에 적지 않는다. 같은 제목의 항목이 이미 있으면 새로 만들지 않고 멈춘다(재실행 중복 방지)."""
import json, mimetypes, os, sys, urllib.request, uuid

BASE = "https://onchain.damilab.cc"
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.normpath(os.path.join(HERE, "..", "symbolic_distillation_0911_report.html"))
MD = os.path.normpath(os.path.join(HERE, "..", "..", "EXPERIMENT-RECORD.md"))
BODY = open(os.path.join(HERE, "progress_body.md"), encoding="utf-8").read()
TITLE = "시장 미시구조 심볼릭 증류 — 3일 전체 학습·틱 크기별 종목군·데이터 양 곡선, 데이터 증가 효과 있음(작음) (0909~0911)"
DATE, STATUS = "2026-09-14", "DONE"

key = os.environ.get("ONCHAIN_EDIT_KEY")
if not key:
    sys.exit("ONCHAIN_EDIT_KEY 환경변수가 없다")


def req(method, path, data=None, headers=None):
    h = {"User-Agent": "Mozilla/5.0 (onchain-upload-script)"}
    if method != "GET":
        h["X-Edit-Key"] = key
    h.update(headers or {})
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.loads(resp.read().decode())


for fp in (HTML, MD):
    if not os.path.exists(fp):
        sys.exit(f"첨부 파일이 없다: {fp}")

existing = [it for it in req("GET", "/api/progress") if it.get("title") == TITLE]
if existing:
    sys.exit(f"같은 제목의 항목이 이미 있다: id={existing[0]['id']} — 새로 만들지 않는다")

item = req("POST", "/api/progress", json.dumps({"title": TITLE, "date": DATE, "status": STATUS, "body": BODY}).encode(),
           {"Content-Type": "application/json"})
print("created item:", item.get("id"))

boundary = uuid.uuid4().hex
parts = []
for fp in (HTML, MD):
    name = os.path.basename(fp)
    ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                  f'Content-Type: {ctype}\r\n\r\n').encode() + open(fp, "rb").read() + b"\r\n")
body = b"".join(parts) + f"--{boundary}--\r\n".encode()
res = req("POST", f"/api/progress/{item['id']}/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("attachments:", [a["name"] for a in res.get("attachments", [])])
print(f"확인: {BASE}/progress.html#item-{item['id']}")
