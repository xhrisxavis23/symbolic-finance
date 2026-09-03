import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sd import config  # noqa: E402


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
