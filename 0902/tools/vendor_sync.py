"""정본 프레임워크의 .py 를 vendor 로 복사하고 매니페스트를 쓴다.

`--check` 는 복사하지 않고 현재 vendor 가 원본과 어떻게 다른지만 보고한다.
PATCHES.md 에 적힌 파일만 달라야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("/home/dgu/tick/proj_claude_tick_finance/Sandbox_12/framework")
DEST = Path(__file__).resolve().parents[1] / "vendor" / "framework"
MANIFEST = DEST.parent / "VENDOR_MANIFEST.json"
PATCHES = DEST.parent / "PATCHES.md"
SKIP_DIRS = {"__pycache__", "Experiments", ".pytest_cache", ".ruff_cache"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> list[Path]:
    out = []
    for path in sorted(SOURCE.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(SOURCE).parts):
            continue
        if path.is_file() and path.suffix in {".py", ".json"}:
            out.append(path)
    return out


def patched_files() -> set[str]:
    """PATCHES.md 의 `- path` 목록. 원본과 달라도 되는 파일."""
    if not PATCHES.is_file():
        return set()
    return set(re.findall(r"^- `([^`]+)`", PATCHES.read_text(encoding="utf-8"), re.M))


def copy() -> dict:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    files = {}
    for path in source_files():
        rel = path.relative_to(SOURCE)
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(rel)] = sha256(path)
    manifest = {
        "schema": "vendor_manifest.v1",
        "source": str(SOURCE),
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "source_sha256": files,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return manifest


def check() -> list[str]:
    """원본과 다른 파일의 상대경로. PATCHES.md 에 적힌 것은 제외한다."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    allowed = patched_files()
    problems = []
    for rel, original in manifest["source_sha256"].items():
        local = DEST / rel
        if rel in allowed:
            continue
        if not local.is_file():
            problems.append(f"없음: {rel}")
        elif sha256(local) != original:
            problems.append(f"원본과 다름: {rel}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        found = check()
        print("\n".join(found) if found else "vendor 무결 — PATCHES.md 외 차이 없음")
        raise SystemExit(1 if found else 0)
    result = copy()
    print(f"복사 {result['file_count']}개 파일 → {DEST}")
