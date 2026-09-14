#!/usr/bin/env python3
"""진행보드 항목 a2adef9ecfc4 갱신 — 본문(PUT)과 보고서 HTML 첨부 교체(DELETE 후 POST). 보드 화면의 저장과 같은 API·같은 필드.
사용: ONCHAIN_EDIT_KEY='편집비밀번호' python3 update_progress_item.py"""
import hashlib, json, mimetypes, os, sys, urllib.parse, urllib.request, uuid

BASE, ITEM = "https://onchain.damilab.cc", "a2adef9ecfc4"
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.normpath(os.path.join(HERE, "..", "symbolic_distillation_0911_report.html"))
BODY = open(os.path.join(HERE, "progress_body.md"), encoding="utf-8").read()
key = os.environ.get("ONCHAIN_EDIT_KEY") or sys.exit("ONCHAIN_EDIT_KEY 환경변수가 없다")


def req(method, path, data=None, headers=None, raw=False):
    h = {"User-Agent": "Mozilla/5.0 (onchain-upload-script)"}
    if method != "GET":
        h["X-Edit-Key"] = key
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(BASE + path, data=data, method=method, headers=h), timeout=180) as r:
        b = r.read()
        return b if raw else json.loads(b.decode())


cur = next(i for i in req("GET", "/api/progress") if i["id"] == ITEM)
print("현재:", cur["title"], cur["date"], cur["status"], "| 첨부", [(a["name"], a["size"]) for a in cur["attachments"]])
saved = req("PUT", f"/api/progress/{ITEM}", json.dumps({"title": cur["title"], "date": cur["date"], "status": cur["status"],
                                                        "body": BODY}).encode(), {"Content-Type": "application/json"})
print("본문 갱신:", len(saved.get("body", "")), "자")
name = os.path.basename(HTML)
if any(a["name"] == name for a in cur["attachments"]):
    upd = req("DELETE", f"/api/progress/{ITEM}/files/{urllib.parse.quote(name)}")
    print("옛 첨부 삭제 후:", [a["name"] for a in upd.get("attachments", [])])
boundary = uuid.uuid4().hex
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
        f'Content-Type: {mimetypes.guess_type(HTML)[0]}\r\n\r\n').encode() + open(HTML, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
res = req("POST", f"/api/progress/{ITEM}/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("새 첨부:", [(a["name"], a["size"]) for a in res.get("attachments", [])])

after = next(i for i in req("GET", "/api/progress") if i["id"] == ITEM)
local = open(HTML, "rb").read()
att = next(a for a in after["attachments"] if a["name"] == name)
served = req("GET", urllib.parse.quote(att["url"]), raw=True)
stripped = b"\n".join(l for l in served.split(b"\n") if b"static.cloudflareinsights.com/beacon" not in l)
print(f"검증: 제목·날짜·상태 유지 {after['title']==cur['title'] and after['date']==cur['date'] and after['status']==cur['status']} · "
      f"본문 일치 {after['body']==BODY} · 기록 크기 {att['size']} = 로컬 {len(local)} · "
      f"통계 스크립트 줄 제외 내용 동일 {hashlib.md5(stripped).hexdigest()==hashlib.md5(local).hexdigest()} · "
      f"첨부 순서 {[a['name'] for a in after['attachments']]}")
print(f"확인: {BASE}/progress.html#item-{ITEM}")
