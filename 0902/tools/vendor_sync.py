"""정본 프레임워크의 .py 를 vendor 로 복사하고 매니페스트를 쓴다.

`--check` 는 복사하지 않고 현재 vendor 가 원본과 어떻게 다른지만 보고한다.
PATCHES.md 에 적힌 파일만 달라야 한다. `check()` 는 세 가지를 본다:

1. 매니페스트에 없는 파일이 vendor 에 새로 생겼는가 (미등재 파일).
2. PATCHES.md 에 등재된 파일이라도 실제로 존재하는가 (삭제 탐지).
3. 등재된 파일 중 매니페스트의 `patched_sha256` 에 동결된 해시가 있는 파일은
   그 해시와 일치하는가 (패치 후 내용 변조 탐지). 항목이 없으면 존재만 본다.
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
SUFFIXES = {".py", ".json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_paths(root: Path) -> list[Path]:
    """`root` 아래 `.py`/`.json` 파일 목록. SKIP_DIRS 는 건너뛴다.

    `source_files()` 와 `check()` 의 vendor 스캔이 같은 규칙을 쓰도록 공유한다.
    """
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix in SUFFIXES:
            out.append(path)
    return out


def source_files() -> list[Path]:
    return _candidate_paths(SOURCE)


def dest_files(dest: Path) -> set[str]:
    """`dest` 안의 `.py`/`.json` 상대경로 집합. 매니페스트 미등재 탐지에 쓴다."""
    return {str(path.relative_to(dest)) for path in _candidate_paths(dest)}


def patched_files(patches_path: Path = PATCHES) -> set[str]:
    """PATCHES.md 의 `- path` 목록. 원본과 달라도 되는 파일."""
    if not patches_path.is_file():
        return set()
    return set(re.findall(r"^- `([^`]+)`", patches_path.read_text(encoding="utf-8"), re.M))


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
        # Task 2+ 가 패치를 적용한 뒤, 그 패치가 고정한 해시를 여기 채운다.
        # 채워진 항목은 check() 가 "패치 후에도 이 내용으로 고정" 으로 강제한다.
        "patched_sha256": {},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return manifest


def check(dest: Path = DEST, manifest_path: Path = MANIFEST, patches_path: Path = PATCHES) -> list[str]:
    """vendor 가 원본·PATCHES.md·patched_sha256 과 어긋나는 지점의 상대경로 목록."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha256 = manifest["source_sha256"]
    patched_sha256 = manifest.get("patched_sha256", {})
    allowed = patched_files(patches_path)
    problems = []

    for rel, original in source_sha256.items():
        local = dest / rel
        if rel in allowed:
            if not local.is_file():
                problems.append(f"없음: {rel}")
                continue
            frozen = patched_sha256.get(rel)
            if frozen is not None and sha256(local) != frozen:
                problems.append(f"패치 후 내용이 바뀜: {rel}")
            continue
        if not local.is_file():
            problems.append(f"없음: {rel}")
        elif sha256(local) != original:
            problems.append(f"원본과 다름: {rel}")

    unregistered = dest_files(dest) - set(source_sha256)
    for rel in sorted(unregistered):
        problems.append(f"매니페스트에 없는 파일: {rel}")

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
