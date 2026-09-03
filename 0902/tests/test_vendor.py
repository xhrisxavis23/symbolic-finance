import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sd import config  # noqa: E402
from tools import vendor_sync  # noqa: E402


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_fake_tree(tmp_path: Path, *, files: dict[str, str]) -> Path:
    """`files` (rel -> content) 를 실제로 써 넣은 가짜 vendor 트리를 만든다."""
    dest = tmp_path / "framework"
    dest.mkdir()
    for rel, content in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return dest


def test_vendor_manifest_exists_and_is_nonempty():
    manifest = json.loads((config.VENDOR_ROOT / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "vendor_manifest.v1"
    assert manifest["file_count"] > 50


def test_vendor_matches_source_except_patches():
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "vendor_sync.py"), "--check"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_framework_imports_from_vendor_not_upstream():
    framework = config.load_framework()
    loaded = Path(framework.__file__).resolve()
    assert config.VENDOR_ROOT in loaded.parents


def test_canonical_profile_hash_is_frozen():
    config.load_framework()
    from framework import canonical
    assert canonical.profile_hash() == config.PROFILE_HASH
    assert canonical.PROFILE_ID == "CANONICAL_QUEUE_V9"


def test_check_detects_unregistered_file(tmp_path):
    """F1: 매니페스트에 없는 신규 파일은 미등재로 보고돼야 한다."""
    dest = _make_fake_tree(tmp_path, files={
        "a.py": "A\n",
        "_injected_backdoor.py": "import os\nos.system('echo pwned')\n",
    })
    manifest_path = tmp_path / "VENDOR_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema": "vendor_manifest.v1",
        "source_sha256": {"a.py": _sha256_text("A\n")},
        "patched_sha256": {},
    }), encoding="utf-8")
    patches_path = tmp_path / "PATCHES.md"
    patches_path.write_text("현재 없음.\n", encoding="utf-8")

    problems = vendor_sync.check(dest=dest, manifest_path=manifest_path, patches_path=patches_path)

    assert any("_injected_backdoor.py" in p and "매니페스트에 없는 파일" in p for p in problems)


def test_check_detects_deleted_patched_file(tmp_path):
    """F2: PATCHES.md 에 등재된 파일이 사라지면 존재 확인에서 걸려야 한다."""
    dest = _make_fake_tree(tmp_path, files={"a.py": "A\n"})  # canonical.py 는 없음
    manifest_path = tmp_path / "VENDOR_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema": "vendor_manifest.v1",
        "source_sha256": {
            "a.py": _sha256_text("A\n"),
            "canonical.py": _sha256_text("ORIGINAL\n"),
        },
        "patched_sha256": {},
    }), encoding="utf-8")
    patches_path = tmp_path / "PATCHES.md"
    patches_path.write_text("- `canonical.py`\n", encoding="utf-8")

    problems = vendor_sync.check(dest=dest, manifest_path=manifest_path, patches_path=patches_path)

    assert any("없음: canonical.py" == p for p in problems)


def test_check_enforces_patched_sha256_when_present_and_skips_when_absent(tmp_path):
    """등재 파일이라도 patched_sha256 에 항목이 있으면 그 해시로 고정, 없으면 통과."""
    dest = _make_fake_tree(tmp_path, files={
        "canonical.py": "PATCHED AND TAMPERED\n",
        "contract.py": "PATCHED FREELY\n",
    })
    manifest_path = tmp_path / "VENDOR_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema": "vendor_manifest.v1",
        "source_sha256": {
            "canonical.py": _sha256_text("ORIGINAL\n"),
            "contract.py": _sha256_text("ORIGINAL\n"),
        },
        # canonical.py 는 패치 후 해시가 동결돼 있다; contract.py 는 동결 안 됨.
        "patched_sha256": {"canonical.py": _sha256_text("PATCHED AND FROZEN\n")},
    }), encoding="utf-8")
    patches_path = tmp_path / "PATCHES.md"
    patches_path.write_text("- `canonical.py`\n- `contract.py`\n", encoding="utf-8")

    problems = vendor_sync.check(dest=dest, manifest_path=manifest_path, patches_path=patches_path)

    assert any("패치 후 내용이 바뀜: canonical.py" == p for p in problems)
    assert not any("contract.py" in p for p in problems)


def test_check_passes_when_patched_sha256_matches(tmp_path):
    """동결된 patched_sha256 과 실제 내용이 일치하면 문제로 보고하지 않는다."""
    dest = _make_fake_tree(tmp_path, files={"canonical.py": "PATCHED AND FROZEN\n"})
    manifest_path = tmp_path / "VENDOR_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "schema": "vendor_manifest.v1",
        "source_sha256": {"canonical.py": _sha256_text("ORIGINAL\n")},
        "patched_sha256": {"canonical.py": _sha256_text("PATCHED AND FROZEN\n")},
    }), encoding="utf-8")
    patches_path = tmp_path / "PATCHES.md"
    patches_path.write_text("- `canonical.py`\n", encoding="utf-8")

    problems = vendor_sync.check(dest=dest, manifest_path=manifest_path, patches_path=patches_path)

    assert problems == []
