# 심볼릭 증류 얇은 수직 슬라이스 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `20260316` KRX 주식 30종목으로 데이터 적재부터 정본 백테스트 원장까지 심볼릭 증류 파이프라인 전 단계를 관통시킨다.

**Architecture:** 정본 백테스트 프레임워크를 `vendor/framework/` 로 복사해 코드 정체성을 동결하고, 그 위에 `sd/` 패키지를 얹는다. 교사와 SR은 인터페이스 뒤에 최소 구현으로 두고, 이 실험의 유일한 신규 위험 지점인 **컴파일러**(SR 수식 → Catalog AST)를 테스트 우선으로 짠다. 최종 산출은 LONG 진입식 하나와 그것이 만든 원장이며, 청산·큐·비용은 `CANONICAL_QUEUE_V9` 가 상수로 건다.

**Tech Stack:** Python 3.10.19 (`/opt/conda/envs/lab`) · numpy · pandas · pyarrow · sympy · torch 2.9 · pytest

**Spec:** [`0902/DESIGN.md`](./DESIGN.md) — 결정 기록 D1~D13과 실측치가 여기 있다. 상위 근거는 [`../symbolic-distillation-for-market-microstructure.md`](../symbolic-distillation-for-market-microstructure.md)

## Global Constraints

모든 태스크의 요구사항에 아래가 암묵적으로 포함된다. 값은 스펙에서 그대로 옮긴 것이다.

- **작업 범위:** `/home/dgu/tick/symbolic/0902/` 안에서만 쓴다. `~/tick/proj_claude_tick_finance/` 는 **읽기 전용** — 어떤 이유로도 수정하지 않는다 (D2)
- **Python:** `/opt/conda/envs/lab/bin/python3`, 버전 3.10.19
- **틱 루트:** `/home/dgu/tick/tickdata_krx`
- **날짜:** `20260316` (증류 입력). `20260317` 이후는 이 슬라이스에서 **열지 않는다**
- **정본 실행 프로필:** `CANONICAL_QUEUE_V9`, 해시 `d0b91a03541c0cec` — Catalog 확장 후에도 불변임을 실측 확인
- **Catalog 원본 해시:** `448d4a81d9341432` — `sqrt`·`tanh` 추가 후 값이 바뀌며, 바뀐 값을 상수로 동결한다
- **종목 선택:** `asset_type == "ST"` AND `issue_code` 6자리 숫자 AND 틱 parquet 정확히 1개 AND 정규장 유효 호가틱 ≥ 200
- **임계 종류:** `rolling_prior_100_ticks_quantile` 고정 (D9)
- **분위 격자:** `(0.70, 0.85, 0.95)` — 슬라이스 전용 (D8)
- **왕복 비용:** 23bp. **스프레드는 체결가에 이미 포함 — 다시 빼지 않는다**
- **재생 워커:** 32
- **탐색 금지:** 청산·큐·비용·결정 규칙·주문 유지시간. 진입식만이 미지수다
- **커밋:** 태스크마다 마지막 스텝에서 커밋한다

---

### Task 1: 저장소 초기화 · vendor 복사 · 무결성 잠금

정본 프레임워크 사본을 만들고, 그것이 원본과 정확히 무엇이 다른지 기계가 검사하게 만든다.

**Files:**
- Create: `0902/.gitignore`
- Create: `0902/sd/__init__.py`
- Create: `0902/sd/config.py`
- Create: `0902/tools/vendor_sync.py`
- Create: `0902/vendor/framework/**` (복사본)
- Create: `0902/vendor/VENDOR_MANIFEST.json` (스크립트가 생성)
- Create: `0902/vendor/PATCHES.md`
- Create: `0902/vendor/__init__.py`
- Test: `0902/tests/test_vendor.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `sd.config` 모듈 — 상수 `TICK_ROOT: Path`, `DATE: str`, `REPO: Path`, `VENDOR_ROOT: Path`, `RUNS_ROOT: Path`, `PROFILE_HASH: str`, `QUANTILE_GRID: tuple[float, ...]`, `REPLAY_WORKERS: int`, `MIN_QUOTE_TICKS: int`, `SLICE_PER_STRATUM: int`, `SEED: int`. 함수 `sd.config.load_framework() -> ModuleType` 이 vendor 를 import 가능하게 만들고 `framework` 패키지를 돌려준다

- [ ] **Step 1: 저장소 초기화와 기본 파일**

```bash
cd /home/dgu/tick/symbolic/0902
git init
mkdir -p sd tools tests vendor runs
touch sd/__init__.py vendor/__init__.py
```

`0902/.gitignore`:

```text
runs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: `sd/config.py` 작성**

```python
"""경로·상수·해시의 단일 진실 원천. 다른 모듈은 값을 자기 안에 두지 않는다."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO / "vendor"
RUNS_ROOT = REPO / "runs"

TICK_ROOT = Path("/home/dgu/tick/tickdata_krx")
DATE = "20260316"

# 정본 실행 프로필. Catalog 확장으로 바뀌지 않음을 실측 확인했다.
PROFILE_HASH = "d0b91a03541c0cec"

# 슬라이스 전용 값 (DESIGN.md D7·D8).
QUANTILE_GRID = (0.70, 0.85, 0.95)
SLICE_PER_STRATUM = 5          # 6개 층 × 5 = 30종목
MIN_QUOTE_TICKS = 200          # 층화 대상 최소 정규장 호가틱
REPLAY_WORKERS = 32
SEED = 0


def load_framework() -> ModuleType:
    """vendor 사본을 import 가능하게 만들고 `framework` 패키지를 돌려준다.

    상위 저장소를 import 하지 않는다. `sys.path` 맨 앞에 vendor 를 넣어,
    같은 이름의 다른 `framework` 가 있어도 사본이 이긴다.
    """
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import framework                      # noqa: E402
    loaded = Path(framework.__file__).resolve()
    if VENDOR_ROOT not in loaded.parents:
        raise RuntimeError(f"vendor 가 아닌 framework 를 잡았다: {loaded}")
    return framework
```

- [ ] **Step 3: `tools/vendor_sync.py` 작성**

```python
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
```

- [ ] **Step 4: 복사 실행**

```bash
cd /home/dgu/tick/symbolic/0902
printf '# vendor 패치 내역\n\n원본과 달라도 되는 파일은 여기 `- `경로`` 형식으로 적는다.\n현재 없음.\n' > vendor/PATCHES.md
python3 tools/vendor_sync.py
```

Expected: `복사 N개 파일 → .../vendor/framework`

- [ ] **Step 5: 무결성 테스트를 쓴다**

`0902/tests/test_vendor.py`:

```python
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
```

- [ ] **Step 6: 테스트 실행**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_vendor.py -v`
Expected: 4 passed

- [ ] **Step 7: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add .gitignore sd/ tools/ tests/ vendor/
git commit -m "feat: vendor 정본 프레임워크 사본과 무결성 검사

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 2: Catalog 에 `sqrt`·`tanh` 추가 (계획서 P0 패치)

계획서 SR 사양이 요구하는 두 단항 연산자를 vendor Catalog 에 더한다. **실험 시작 전 1회, 이후 동결.**

**Files:**
- Modify: `0902/vendor/framework/catalog.py` (`OPERATORS` 정의에 항목 2개 추가)
- Modify: `0902/vendor/framework/contract.py` (`ExpressionRuntime` 에 메서드 2개 추가)
- Modify: `0902/vendor/PATCHES.md`
- Modify: `0902/sd/config.py` (`CATALOG_HASH` 추가)
- Test: `0902/tests/test_catalog_patch.py`

**Interfaces:**
- Consumes: `sd.config.load_framework()`
- Produces: Catalog 연산자 `sqrt`·`tanh` — 둘 다 `{"op": "sqrt", "input": <numeric expr>}` 형태이고 **무차원 입력만 받는다**. 출력은 `TypeInfo("numeric", "dimensionless", ...)`. `sd.config.CATALOG_HASH: str` 상수

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다**

`0902/tests/test_catalog_patch.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402

config.load_framework()
from framework import catalog, contract  # noqa: E402


def test_sqrt_and_tanh_are_registered():
    assert "sqrt" in catalog.OPERATORS
    assert "tanh" in catalog.OPERATORS


def test_sqrt_accepts_dimensionless_input():
    expr = {"op": "sqrt", "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    info = catalog.infer_expression_type(expr)
    assert info.value_type == "numeric"
    assert info.dimension == "dimensionless"


def test_sqrt_rejects_dimensionful_input():
    expr = {"op": "sqrt", "input": {"op": "primitive", "primitive_id": "mid_price"}}
    with pytest.raises(catalog.ExpressionError) as excinfo:
        catalog.infer_expression_type(expr)
    assert "무차원" in str(excinfo.value)


def test_tanh_rejects_dimensionful_input():
    expr = {"op": "tanh", "input": {"op": "primitive", "primitive_id": "bid_depth_total_5"}}
    with pytest.raises(catalog.ExpressionError):
        catalog.infer_expression_type(expr)


def test_sqrt_of_negative_is_nan_not_exception():
    n = 8
    book = _fixture_book(n)
    runtime = contract.ExpressionRuntime(book.data)
    expr = {"op": "sqrt",
            "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    out = runtime.evaluate(expr)
    assert out.shape == (n,)
    assert np.all(np.isnan(out) | (out >= 0.0))


def test_tanh_is_bounded():
    n = 8
    book = _fixture_book(n)
    runtime = contract.ExpressionRuntime(book.data)
    expr = {"op": "tanh",
            "input": {"op": "primitive", "primitive_id": "book_imbalance"}}
    out = runtime.evaluate(expr)
    finite = out[np.isfinite(out)]
    assert np.all(np.abs(finite) <= 1.0)


def test_catalog_hash_is_frozen():
    assert catalog.catalog_hash() == config.CATALOG_HASH
    assert catalog.catalog_hash() != "448d4a81d9341432"   # 원본 값에서 바뀌었다


def _fixture_book(n: int):
    """매수 잔량이 매도보다 많다가 뒤집히는 최소 호가창."""
    levels = 10
    bid_price = np.tile(np.arange(1000, 1000 - levels, -1, dtype=float), (n, 1))
    ask_price = np.tile(np.arange(1001, 1001 + levels, dtype=float), (n, 1))
    bid_qty = np.tile(np.full(levels, 100.0), (n, 1))
    ask_qty = np.tile(np.full(levels, 100.0), (n, 1))
    bid_qty[: n // 2, 0] = 300.0        # 앞 절반은 book_imbalance > 0
    ask_qty[n // 2 :, 0] = 300.0        # 뒤 절반은 < 0
    data = {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": bid_qty, "ask_qty": ask_qty,
            "time_s": np.arange(n, dtype=float),
            "local_time": np.arange(n, dtype=np.int64)}
    return catalog.Book(data)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_catalog_patch.py -v`
Expected: FAIL — `test_sqrt_and_tanh_are_registered` 에서 `assert "sqrt" in catalog.OPERATORS` 실패, 그리고 `config` 에 `CATALOG_HASH` 가 없어 `AttributeError`

- [ ] **Step 3: `catalog.py` 에 연산자 두 개 추가**

`0902/vendor/framework/catalog.py` 에서 `OPERATORS` 정의 안, `Operator("log1p", ...)` 항목 **바로 다음 줄**에 아래를 넣는다.

```python
    Operator("sqrt", ("input",), inputs={"input": "numeric"},
             output=_sqrt_type, output_doc="dimensionless",
             doc="제곱근. **무차원 입력만** 받는다 — 차원 있는 값은 ratio 나 "
                 "rolling_zscore 를 먼저 거쳐야 한다"),
    Operator("tanh", ("input",), inputs={"input": "numeric"},
             output=_tanh_type, output_doc="dimensionless [-1, 1]",
             doc="포화 함수. **무차원 입력만** 받는다. 유동성 유한성의 표현"),
```

그리고 `OPERATORS = {...}` 정의보다 **앞에**, `_same_or_derived` 함수 다음에 타입 함수 둘을 넣는다.

```python
def _sqrt_type(kids: Mapping[str, TypeInfo], node: Mapping[str, Any]) -> TypeInfo:
    """제곱근은 차원을 반으로 나눈다. 이 차원계에는 반차원이 없으므로 무차원만 받는다."""
    value = kids["input"]
    if value.dimension != "dimensionless":
        raise ValueError(f"sqrt 는 무차원 입력만 받는다. 받은 차원: {value.dimension}")
    return TypeInfo("numeric", "dimensionless", "sqrt")


def _tanh_type(kids: Mapping[str, TypeInfo], node: Mapping[str, Any]) -> TypeInfo:
    """포화 함수의 인수는 무차원이어야 한다. 물리와 같은 규칙이다."""
    value = kids["input"]
    if value.dimension != "dimensionless":
        raise ValueError(f"tanh 는 무차원 입력만 받는다. 받은 차원: {value.dimension}")
    return TypeInfo("numeric", "dimensionless", "ratio")
```

- [ ] **Step 4: `contract.py` 에 평가 메서드 두 개 추가**

`0902/vendor/framework/contract.py` 의 `ExpressionRuntime` 클래스에서 `_op_log1p` 메서드 **바로 다음**에 넣는다.

```python
    def _op_sqrt(self, node):
        values = np.asarray(self.evaluate(node["input"]), dtype=float)
        out = np.full(self.n, np.nan)
        valid = np.isfinite(values) & (values >= 0.0)
        out[valid] = np.sqrt(values[valid])
        return out

    def _op_tanh(self, node):
        values = np.asarray(self.evaluate(node["input"]), dtype=float)
        out = np.full(self.n, np.nan)
        valid = np.isfinite(values)
        out[valid] = np.tanh(values[valid])
        return out
```

- [ ] **Step 5: 새 Catalog 해시를 읽어 `config.py` 에 못 박는다**

```bash
cd /home/dgu/tick/symbolic/0902
python3 -c "
import sys; sys.path.insert(0, 'vendor')
from framework import catalog
print(catalog.catalog_hash())
"
```

출력된 16자 해시를 `sd/config.py` 의 `PROFILE_HASH` 줄 아래에 넣는다.

```python
# Catalog 어휘 해시. sqrt·tanh 추가 후 값이며 실험 종료까지 동결한다.
# 원본(패치 전) 값은 448d4a81d9341432 였다.
CATALOG_HASH = "<위 명령이 출력한 값>"
```

- [ ] **Step 6: `PATCHES.md` 기록**

`0902/vendor/PATCHES.md` 를 아래로 덮어쓴다.

```markdown
# vendor 패치 내역

원본과 달라도 되는 파일은 `- ` + 백틱 경로 형식으로 적는다.
`tools/vendor_sync.py --check` 가 이 목록만 예외로 인정한다.

## 계획서 P0 — SR 연산자 확장

- `catalog.py`
- `contract.py`

**무엇을:** 단항 연산자 `sqrt` · `tanh` 를 `OPERATORS` 에 추가하고,
`ExpressionRuntime` 에 `_op_sqrt` · `_op_tanh` 를 더했다.

**왜:** 계획서 §3 S3 이 두 연산자를 SR 사양에 필수로 요구한다.
`sqrt` 없이는 L3(제곱근 법칙) 복원 여부를 물을 수 없고, `tanh` 없이는
유동성 포화 구조를 표현할 수 없다.

**설계:** 둘 다 **무차원 입력만** 받는다. 이것은 제약이 아니라 계획서 §3 S3 의
규칙("차원 있는 feature 는 비나 z-score 를 거치지 않고는 SR 입력이 될 수 없다")을
타입 수준에서 강제하는 장치다.

**언제 다시 건드리나:** 안 건드린다. 실험 종료까지 동결.
```

- [ ] **Step 7: 테스트 실행**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/ -v`
Expected: `test_vendor.py` 4 passed + `test_catalog_patch.py` 7 passed

- [ ] **Step 8: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add vendor/ sd/config.py tests/test_catalog_patch.py
git commit -m "feat: Catalog 에 sqrt/tanh 추가 (무차원 입력 강제)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 3: 종목 우주와 층화

`20260316` ST 우주를 6개 층으로 나누고 슬라이스 30종목을 결정론적으로 고른다.

**Files:**
- Create: `0902/sd/universe.py`
- Test: `0902/tests/test_universe.py`

**Interfaces:**
- Consumes: `sd.config` (TICK_ROOT, DATE, MIN_QUOTE_TICKS, SLICE_PER_STRATUM)
- Produces:
  - `sd.universe.stock_symbols(date: str) -> tuple[str, ...]` — 선택 사다리 3단계까지 (ST ∩ 6자리 ∩ 틱파일 1개). `20260316` 에서 2570개
  - `sd.universe.liquidity_stats(symbols, date, workers=32) -> pd.DataFrame` — index=symbol, columns=`n_quote, spread_bps, ask1_depth, rv20_bps`
  - `sd.universe.assign_strata(stats: pd.DataFrame) -> pd.DataFrame` — 위 컬럼 + `friction` (`L`/`M`/`H`) + `depth` (`shallow`/`deep`) + `stratum` (`L-shallow` 등 6종)
  - `sd.universe.slice_symbols(strata: pd.DataFrame, per_stratum: int) -> tuple[str, ...]` — 층마다 `issue_code` 오름차순 앞에서 `per_stratum` 개
  - `sd.universe.STRATA = ("L-shallow","L-deep","M-shallow","M-deep","H-shallow","H-deep")`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_universe.py`:

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, universe  # noqa: E402


def test_stock_symbols_matches_measured_count():
    symbols = universe.stock_symbols(config.DATE)
    assert len(symbols) == 2570
    assert symbols == tuple(sorted(symbols))
    assert all(len(s) == 6 and s.isdigit() for s in symbols)
    assert "005930" in symbols


def test_etf_codes_are_excluded():
    symbols = set(universe.stock_symbols(config.DATE))
    assert "0000D0" not in symbols          # TIGER ETF
    assert "33637K" not in symbols          # 영문자 포함 우선주 코드


def test_assign_strata_makes_six_balanced_groups():
    stats = pd.DataFrame({
        "n_quote":    [1000] * 12,
        "spread_bps": [5, 6, 7, 8, 20, 21, 22, 23, 50, 51, 52, 53],
        "ask1_depth": [10, 20, 300, 400] * 3,
        "rv20_bps":   [1.0] * 12,
    }, index=[f"{i:06d}" for i in range(12)])
    out = universe.assign_strata(stats)
    assert set(out.stratum.unique()) == set(universe.STRATA)
    assert out.stratum.value_counts().tolist() == [2] * 6


def test_slice_symbols_is_deterministic_and_balanced():
    stats = pd.DataFrame({
        "n_quote":    [1000] * 60,
        "spread_bps": ([5.0] * 20) + ([20.0] * 20) + ([50.0] * 20),
        "ask1_depth": ([10.0] * 10 + [900.0] * 10) * 3,
        "rv20_bps":   [1.0] * 60,
    }, index=[f"{i:06d}" for i in range(60)])
    strata = universe.assign_strata(stats)
    picked = universe.slice_symbols(strata, per_stratum=2)
    assert len(picked) == 12
    assert picked == universe.slice_symbols(strata, per_stratum=2)   # 재현
    assert picked == tuple(sorted(picked))
    counts = strata.loc[list(picked)].stratum.value_counts()
    assert counts.tolist() == [2] * 6


@pytest.mark.slow
def test_real_strata_boundaries_match_design():
    symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(symbols, config.DATE)
    assert len(stats) == 2507                      # 호가틱 200 미만 63개 제외
    strata = universe.assign_strata(stats)
    friction = strata.groupby("friction").size()
    assert friction["L"] == 837 and friction["M"] == 834 and friction["H"] == 836
    lo = strata.loc[strata.friction == "L", "spread_bps"].max()
    assert 18.3 < lo < 18.4
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_universe.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.universe'`

- [ ] **Step 3: `sd/universe.py` 구현**

```python
"""종목 우주와 층화. 계획서 §3 S−1 ① · §3 S2.

층은 이 날짜 이 우주에서 직접 계산한다. 정본 `framework/universe.py` 의
MK01~MK07 을 쓰지 않는 이유는 DESIGN.md D6 에 있다 — 그 명단은 ST+ETF 혼합
우주에서 나왔고, ETF 를 빼면 MK05 가 1종목으로 붕괴한다.
"""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

QUOTE_ROW = 12
SESSION_START = 90000000000        # 09:00:00.000000
SESSION_END = 153000000000         # 15:30:00.000000
STAT_COLUMNS = ("data_type", "local_time", "bid1_price", "ask1_price", "ask1_quantity")
RV_LAG = 20

STRATA = ("L-shallow", "L-deep", "M-shallow", "M-deep", "H-shallow", "H-deep")
FRICTIONS = ("L", "M", "H")


def _date_dir(date: str) -> Path:
    return config.TICK_ROOT / str(date)


def stock_symbols(date: str) -> tuple[str, ...]:
    """선택 사다리 3단계까지. ST ∩ 6자리 숫자 ∩ 틱 parquet 정확히 1개.

    정본 `modules/common_stock_cache.py` 와 같은 규칙이다. 증류가 학습한 종목과
    백테스트가 실행하는 종목이 어긋나면 비교가 성립하지 않는다.
    """
    folder = _date_dir(date)
    batch = pd.read_parquet(folder / f"stock_batch_{date}.parquet",
                            columns=["issue_code", "asset_type"])
    selected = {
        str(code) for code, asset in zip(batch.issue_code, batch.asset_type)
        if str(asset) == "ST" and re.fullmatch(r"[0-9]{6}", str(code))
    }
    counts: dict[str, int] = {}
    for path in folder.glob("*.parquet"):
        match = re.fullmatch(r"([0-9]{6})_.*\.parquet", path.name)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    available = {symbol for symbol, count in counts.items() if count == 1}
    return tuple(sorted(selected & available))


def _one_symbol_stats(symbol: str, date: str) -> dict | None:
    """한 종목의 3축 통계. 유효 호가틱이 모자라면 None — 부재이지 실패가 아니다."""
    matches = glob.glob(str(_date_dir(date) / f"{symbol}_*.parquet"))
    if len(matches) != 1:
        return None
    table = pq.read_table(matches[0], columns=list(STAT_COLUMNS))
    data_type = table["data_type"].combine_chunks().to_numpy()
    local_time = table["local_time"].combine_chunks().to_numpy()
    keep = (data_type == QUOTE_ROW) & (local_time >= SESSION_START) & (local_time <= SESSION_END)
    if int(keep.sum()) < config.MIN_QUOTE_TICKS:
        return None
    bid = table["bid1_price"].combine_chunks().to_numpy()[keep].astype(float)
    ask = table["ask1_price"].combine_chunks().to_numpy()[keep].astype(float)
    ask_qty = table["ask1_quantity"].combine_chunks().to_numpy()[keep].astype(float)
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    bid, ask, ask_qty = bid[valid], ask[valid], ask_qty[valid]
    if len(bid) < config.MIN_QUOTE_TICKS:
        return None
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid * 1e4
    moves = (np.abs(mid[RV_LAG:] / mid[:-RV_LAG] - 1.0) * 1e4
             if len(mid) > RV_LAG else np.zeros(1))
    return {"symbol": symbol, "n_quote": int(len(bid)),
            "spread_bps": float(np.median(spread)),
            "ask1_depth": float(np.median(ask_qty)),
            "rv20_bps": float(np.median(moves))}


def liquidity_stats(symbols: Sequence[str], date: str, workers: int = 32) -> pd.DataFrame:
    """종목별 (스프레드, ASK1 잔량, 20틱 변화) 중앙값. 미래 라벨을 쓰지 않는다."""
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        rows = list(pool.map(lambda s: _one_symbol_stats(s, date), symbols))
    frame = pd.DataFrame([r for r in rows if r is not None])
    return frame.set_index("symbol").sort_index()


def assign_strata(stats: pd.DataFrame) -> pd.DataFrame:
    """마찰 3분위 × 잔량 2분할 = 6층. 경계는 이 표본에서 나온다."""
    out = stats.copy()
    low, high = out.spread_bps.quantile([1 / 3, 2 / 3])
    out["friction"] = np.where(out.spread_bps <= low, "L",
                               np.where(out.spread_bps <= high, "M", "H"))
    out["depth"] = out.groupby("friction").ask1_depth.transform(
        lambda column: np.where(column <= column.median(), "shallow", "deep"))
    out["stratum"] = out.friction + "-" + out.depth
    return out


def slice_symbols(strata: pd.DataFrame, per_stratum: int) -> tuple[str, ...]:
    """층마다 코드 오름차순 앞에서 n 개. 무작위 추출을 쓰지 않는다 — 재현성."""
    picked: list[str] = []
    for name in STRATA:
        members = sorted(strata.index[strata.stratum == name])
        picked.extend(members[: int(per_stratum)])
    return tuple(sorted(picked))
```

- [ ] **Step 4: 빠른 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_universe.py -v -m "not slow"`
Expected: 4 passed

- [ ] **Step 5: 실데이터 테스트 확인**

`0902/pytest.ini` 생성:

```ini
[pytest]
markers =
    slow: 실데이터를 읽는 테스트 (수 초 ~ 수 분)
```

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_universe.py -v -m slow`
Expected: 2 passed (약 10초). 층 크기 L 837 · M 834 · H 836 이 DESIGN.md §D6 과 일치

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/universe.py tests/test_universe.py pytest.ini
git commit -m "feat: day-1 ST 우주 층화와 결정론적 슬라이스 선택

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 4: 틱 적재와 feature 행렬

vendor 의 적재·feature 계산을 감싸 하나의 종목-일을 학습 가능한 행렬로 만든다.

**Files:**
- Create: `0902/sd/ticks.py`
- Test: `0902/tests/test_ticks.py`

**Interfaces:**
- Consumes: `sd.config.load_framework()`
- Produces:
  - `sd.ticks.load_arrays(symbol: str, date: str) -> dict[str, np.ndarray]` — vendor `data.load` 위임. 키는 `bid_price, ask_price, bid_qty, ask_qty, time_s, local_time` (+체결이 있으면 `buy_volume, sell_volume, buy_max_price, sell_min_price`)
  - `sd.ticks.feature_matrix(arrays) -> tuple[np.ndarray, tuple[str, ...]]` — `(n, k)` 실수 행렬과 열 이름. 이 데이터로 못 만드는 feature 는 열에서 빠진다
  - `sd.ticks.FEATURE_ORDER: tuple[str, ...]` — 계산을 시도할 feature 이름을 **정렬 고정**한 것. 종목마다 열 순서가 달라지지 않게 한다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_ticks.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, ticks  # noqa: E402


def test_feature_order_is_sorted_and_covers_catalog():
    config.load_framework()
    from framework import catalog
    assert ticks.FEATURE_ORDER == tuple(sorted(ticks.FEATURE_ORDER))
    assert set(ticks.FEATURE_ORDER) <= set(catalog.FEATURES)
    assert "book_imbalance" in ticks.FEATURE_ORDER
    assert "spread_to_round_trip_cost_ratio" in ticks.FEATURE_ORDER


@pytest.mark.slow
def test_load_arrays_returns_expected_keys():
    arrays = ticks.load_arrays("005930", config.DATE)
    for key in ("bid_price", "ask_price", "bid_qty", "ask_qty", "time_s", "local_time"):
        assert key in arrays
    assert arrays["bid_price"].ndim == 2 and arrays["bid_price"].shape[1] == 10
    assert len(arrays["time_s"]) > 100_000


@pytest.mark.slow
def test_feature_matrix_shape_and_names_agree():
    arrays = ticks.load_arrays("005930", config.DATE)
    matrix, names = ticks.feature_matrix(arrays)
    assert matrix.shape == (len(arrays["time_s"]), len(names))
    assert matrix.dtype == np.float64
    assert names == tuple(sorted(names))
    assert "book_imbalance" in names
    column = matrix[:, names.index("book_imbalance")]
    finite = column[np.isfinite(column)]
    assert np.all((finite >= -1.0) & (finite <= 1.0))
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_ticks.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.ticks'`

- [ ] **Step 3: `sd/ticks.py` 구현**

```python
"""틱 적재와 feature 행렬. 계산은 vendor Catalog 가 한다 — 여기서 다시 짜지 않는다.

feature 정의를 우리 쪽에서 구현하면 백테스트가 쓰는 정의와 조용히 갈라진다.
그러면 진입식이 학습 때와 재생 때 다른 값을 본다.
"""

from __future__ import annotations

import numpy as np

from . import config

_framework = config.load_framework()
from framework import catalog as _catalog        # noqa: E402
from framework import data as _data              # noqa: E402

# 열 순서를 이름 정렬로 고정한다. 종목마다 순서가 달라지면 행렬이 뜻을 잃는다.
FEATURE_ORDER: tuple[str, ...] = tuple(sorted(_catalog.FEATURES))


def load_arrays(symbol: str, date: str) -> dict[str, np.ndarray]:
    """한 종목-일의 원본 배열. 정규장 구간만. 캐시는 쓰지 않는다 (DESIGN.md D10)."""
    arrays, _source = _data.load(symbol, date, root=config.TICK_ROOT, cache_root=None)
    return arrays


def feature_matrix(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    """`(n, k)` 행렬과 열 이름.

    체결 행이 없는 종목-일에서는 체결 의존 feature 가 계산되지 않는다. 0 으로
    채우지 않고 **열에서 뺀다** — 0 은 "매도 압력이 없었다" 는 관측을 지어내는 것이다.
    """
    book = _catalog.Book(arrays)
    computed = _catalog.compute_features(FEATURE_ORDER, book)
    names = tuple(sorted(computed))
    if not names:
        raise ValueError("계산된 feature 가 하나도 없다")
    matrix = np.column_stack([np.asarray(computed[name], dtype=float) for name in names])
    return matrix, names
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_ticks.py -v`
Expected: 3 passed (slow 포함, 약 1분)

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/ticks.py tests/test_ticks.py
git commit -m "feat: 틱 적재와 Catalog feature 행렬 래퍼

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 5: 라벨 — 고정 청산 아래의 진입시점 결과

계획서 §3 S0. 각 호가틱을 진입 후보로 보고 30초 경로 net 과 체결 여부를 만든다.

**Files:**
- Create: `0902/sd/labels.py`
- Test: `0902/tests/test_labels.py`

**Interfaces:**
- Consumes: `sd.ticks.load_arrays`, `sd.config`
- Produces: `sd.labels.build(arrays: dict[str, np.ndarray]) -> Labels` where

```python
@dataclass(frozen=True)
class Labels:
    y_path: np.ndarray      # (n,) 마찰 정규화 net. 관측 불가 지점은 NaN
    y_fill: np.ndarray      # (n,) 0/1. 10초·100틱 안에 BID1 지정가가 체결됐는가의 근사
    mask: np.ndarray        # (n,) bool. y_path 가 유한하고 학습에 쓸 수 있는 지점
    raw_net_bps: np.ndarray # (n,) 정규화 전 net. 진단용
```

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_labels.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, labels  # noqa: E402


def _ramp_arrays(n=400, drift_bps=50.0):
    """가격이 완만히 오르는 합성 호가창. 30초 뒤 net 이 양수여야 한다."""
    levels = 10
    base = 10_000.0
    step = base * (drift_bps / 1e4) / n
    mid = base + step * np.arange(n)
    bid1 = np.round(mid - 5.0)
    ask1 = np.round(mid + 5.0)
    bid_price = bid1[:, None] - np.arange(levels)[None, :]
    ask_price = ask1[:, None] + np.arange(levels)[None, :]
    qty = np.full((n, levels), 100.0)
    return {"bid_price": bid_price, "ask_price": ask_price,
            "bid_qty": qty, "ask_qty": qty,
            "time_s": np.arange(n, dtype=float) * 0.5,
            "local_time": (90000000000 + np.arange(n) * 500_000).astype(np.int64)}


def test_normalisation_divides_by_round_trip_friction():
    arrays = _ramp_arrays()
    out = labels.build(arrays)
    idx = np.flatnonzero(out.mask)[0]
    spread_bps = 10.0 / 10_000.0 * 1e4          # 10 KRW / 10,000 KRW
    expected = out.raw_net_bps[idx] / (spread_bps + 23.0)
    assert out.y_path[idx] == pytest.approx(expected, rel=1e-6)


def test_tail_of_day_is_masked_not_zeroed():
    arrays = _ramp_arrays(n=400)
    out = labels.build(arrays)
    assert not out.mask[-1]                      # 30초 창이 데이터 끝을 넘는다
    assert np.isnan(out.y_path[-1])              # 0 으로 채우지 않는다


def test_rising_market_gives_positive_labels_somewhere():
    arrays = _ramp_arrays(n=400, drift_bps=400.0)
    out = labels.build(arrays)
    assert out.mask.sum() > 0
    assert np.nanmax(out.y_path[out.mask]) > 0.0


def test_shapes_and_fill_label_are_binary():
    arrays = _ramp_arrays()
    out = labels.build(arrays)
    n = len(arrays["time_s"])
    assert out.y_path.shape == out.y_fill.shape == out.mask.shape == (n,)
    assert set(np.unique(out.y_fill)) <= {0.0, 1.0}
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.labels'`

- [ ] **Step 3: `sd/labels.py` 구현**

```python
"""라벨. 계획서 §3 S0 — 청산이 타깃을 정의한다.

두 가지를 만든다.

  y_path   진입 후 30초 BID1 경로의 net. **청산 규칙을 타지 않는다.**
           교사의 주 학습 신호다.
  y_fill   BID1 지정가가 10초/100틱 안에 체결됐을지의 근사.

`y_exec`(원장 net)는 여기서 만들지 않는다. 그것은 재생이 만들고 판정에만 쓴다.
교사를 `y_exec` 로 학습시키면 청산 규칙의 특이점을 외운다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

_framework = config.load_framework()
from framework.config import ENTRY_ORDER_MAX_SECONDS, ENTRY_ORDER_MAX_TICKS  # noqa: E402
from framework.config import FEE_BPS, PRIMARY_HORIZON_SECONDS                # noqa: E402


@dataclass(frozen=True)
class Labels:
    y_path: np.ndarray
    y_fill: np.ndarray
    mask: np.ndarray
    raw_net_bps: np.ndarray


def build(arrays: dict[str, np.ndarray]) -> Labels:
    time_s = np.asarray(arrays["time_s"], dtype=float)
    bid1 = np.asarray(arrays["bid_price"], dtype=float)[:, 0]
    ask1 = np.asarray(arrays["ask_price"], dtype=float)[:, 0]
    n = len(time_s)

    # 평가 창 끝 틱. 데이터 끝을 넘으면 관측 불가 = 마스크 (CENSORED 와 같은 뜻).
    horizon = np.searchsorted(time_s, time_s + float(PRIMARY_HORIZON_SECONDS), side="left")
    observable = horizon < n

    # 체결 가능한 호가만. 0 이나 비유한은 "그 가격에 팔 수 있다" 가 아니다.
    sellable = np.where(np.isfinite(bid1) & (bid1 > 0.0), bid1, np.nan)
    buyable = np.where(np.isfinite(ask1) & (ask1 > 0.0), ask1, np.nan)

    exit_index = np.where(observable, np.clip(horizon, 0, n - 1), 0)
    raw = (sellable[exit_index] / buyable - 1.0) * 1e4 - FEE_BPS
    raw = np.where(observable, raw, np.nan)

    mid = (bid1 + ask1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_bps = (ask1 - bid1) / mid * 1e4
    friction = spread_bps + FEE_BPS
    y_path = np.where(friction > 0.0, raw / friction, np.nan)

    mask = observable & np.isfinite(y_path)
    return Labels(y_path=y_path, y_fill=_fill_label(arrays, time_s, bid1),
                  mask=mask, raw_net_bps=raw)


def _fill_label(arrays: dict[str, np.ndarray], time_s: np.ndarray,
                bid1: np.ndarray) -> np.ndarray:
    """BID1 지정가가 대기 창 안에 체결됐을지의 보수적 근사.

    정확한 큐 순번은 이 데이터에 없다. 재생이 쓰는 정본 큐 모델이 정답이고,
    이것은 **교사에게 줄 학습 신호**일 뿐이다. 판정에는 절대 쓰지 않는다.

    규칙: 게시 시점 BID1 가격 이하로 매도 주도 체결이 실제로 내려왔는가.
    체결 데이터가 없으면 전부 0 — 없는 관측을 지어내지 않는다.
    """
    n = len(time_s)
    out = np.zeros(n, dtype=float)
    sell_min = arrays.get("sell_min_price")
    if sell_min is None:
        return out
    sell_min = np.asarray(sell_min, dtype=float)
    deadline = time_s + float(ENTRY_ORDER_MAX_SECONDS)
    last_by_clock = np.searchsorted(time_s, deadline, side="right")
    last_by_ticks = np.arange(n) + int(ENTRY_ORDER_MAX_TICKS) + 1
    last = np.minimum(np.minimum(last_by_clock, last_by_ticks), n)
    for i in range(n):
        window = sell_min[i + 1: last[i]]
        touched = window[np.isfinite(window) & (window > 0.0)]
        if touched.size and float(np.min(touched)) <= bid1[i]:
            out[i] = 1.0
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_labels.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/labels.py tests/test_labels.py
git commit -m "feat: 고정 청산 아래의 진입시점 라벨 (y_path, y_fill)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 6: 무차원 좌표

계획서 §3 S3. 차원 있는 feature 를 SR 입력에서 배제하고, 어떤 것이 어떻게 무차원이 됐는지 기록한다.

**Files:**
- Create: `0902/sd/dimensionless.py`
- Test: `0902/tests/test_dimensionless.py`

**Interfaces:**
- Consumes: `sd.config.load_framework()`, `sd.ticks.feature_matrix` 의 출력
- Produces:
  - `sd.dimensionless.DIMENSIONLESS_FEATURES: tuple[str, ...]` — Catalog 에서 `dimension == "dimensionless"` 인 feature 이름 정렬본
  - `sd.dimensionless.transform(matrix, names) -> tuple[np.ndarray, tuple[str, ...], dict]` — 무차원 열만 남긴 행렬, 그 열 이름, 그리고 `{"kept": [...], "dropped": {name: dimension}}` 메타

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_dimensionless.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, dimensionless  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402


def test_only_dimensionless_features_are_kept():
    for name in dimensionless.DIMENSIONLESS_FEATURES:
        assert catalog.FEATURES[name].type_info.dimension == "dimensionless"
    assert "book_imbalance" in dimensionless.DIMENSIONLESS_FEATURES
    assert "mid_price" not in dimensionless.DIMENSIONLESS_FEATURES
    assert "bid_depth_total_5" not in dimensionless.DIMENSIONLESS_FEATURES


def test_transform_drops_dimensionful_columns_and_records_why():
    names = ("book_imbalance", "mid_price", "queue_imbalance_best")
    matrix = np.arange(12, dtype=float).reshape(4, 3)
    kept_matrix, kept_names, meta = dimensionless.transform(matrix, names)
    assert kept_names == ("book_imbalance", "queue_imbalance_best")
    assert kept_matrix.shape == (4, 2)
    assert meta["dropped"] == {"mid_price": "price"}
    np.testing.assert_array_equal(kept_matrix[:, 0], matrix[:, 0])


def test_transform_is_order_stable():
    names = ("queue_imbalance_best", "book_imbalance")
    matrix = np.zeros((3, 2))
    _kept, kept_names, _meta = dimensionless.transform(matrix, names)
    assert kept_names == tuple(sorted(kept_names))
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_dimensionless.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.dimensionless'`

- [ ] **Step 3: `sd/dimensionless.py` 구현**

```python
"""무차원 좌표. 계획서 §3 S3.

규칙 하나다 — **차원 있는 feature 는 SR 입력이 될 수 없다.** 종목 스케일이
수식에 들어오면 SR 은 유동성 등급을 설명하는 항을 붙이고 그것을 알파로 착각한다.

Catalog 가 이미 모든 feature 에 `dimension` 을 붙여 두었으므로 여기서 다시
분류하지 않는다. Task 2 에서 `sqrt`·`tanh` 를 무차원 입력만 받게 만든 것도
같은 규칙을 타입 수준에서 강제한 것이다.

슬라이스에서는 **이미 무차원인 feature 만** 쓴다. 차원 있는 feature 를
`ratio` 나 `rolling_zscore` 로 무차원화해 파생 열을 만드는 것은 다음 단계다.
"""

from __future__ import annotations

import numpy as np

from . import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402

DIMENSIONLESS_FEATURES: tuple[str, ...] = tuple(sorted(
    name for name, spec in _catalog.FEATURES.items()
    if spec.type_info.dimension == "dimensionless"
))


def transform(matrix: np.ndarray,
              names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """무차원 열만 남긴다. 무엇을 왜 뺐는지 함께 돌려준다."""
    keep = tuple(sorted(n for n in names if n in set(DIMENSIONLESS_FEATURES)))
    dropped = {n: _catalog.FEATURES[n].type_info.dimension
               for n in names if n not in set(DIMENSIONLESS_FEATURES)}
    if not keep:
        raise ValueError("무차원 feature 가 하나도 남지 않았다")
    index = [names.index(n) for n in keep]
    meta = {"kept": list(keep), "dropped": dropped,
            "rule": "Catalog dimension == 'dimensionless' 인 feature 만 SR 입력"}
    return matrix[:, index], keep, meta
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_dimensionless.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/dimensionless.py tests/test_dimensionless.py
git commit -m "feat: 무차원 좌표 변환 — 차원 있는 feature 를 SR 입력에서 배제

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 7: On-manifold 표집

계획서 §3 S2. 실존하는 시장 상태만 SR 학습에 쓰고, 꼬리 상태에 가중치를 준다.

**Files:**
- Create: `0902/sd/manifold.py`
- Test: `0902/tests/test_manifold.py`

**Interfaces:**
- Consumes: 무차원 행렬 `X`, 라벨 마스크
- Produces: `sd.manifold.select(X, mask, max_samples, seed) -> Selection` where

```python
@dataclass(frozen=True)
class Selection:
    index: np.ndarray      # (m,) int. 학습에 쓸 행
    weight: np.ndarray     # (m,) float. 중요도 가중치. 합이 m 이 되게 정규화
    on_manifold: np.ndarray  # (m,) bool. 신뢰 반경 안인가
```

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_manifold.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import manifold  # noqa: E402


def _blob(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def test_selection_only_returns_masked_rows():
    X = _blob()
    mask = np.zeros(len(X), dtype=bool)
    mask[::3] = True
    sel = manifold.select(X, mask, max_samples=100, seed=0)
    assert np.all(mask[sel.index])


def test_far_outliers_are_flagged_as_probe():
    X = _blob(n=500)
    X = np.vstack([X, np.full((5, 3), 50.0)])         # 명백한 off-manifold
    mask = np.ones(len(X), dtype=bool)
    sel = manifold.select(X, mask, max_samples=len(X), seed=0)
    tail = sel.index >= 500
    assert tail.sum() > 0
    assert not sel.on_manifold[tail].any()


def test_weights_are_normalised_and_positive():
    X = _blob()
    mask = np.ones(len(X), dtype=bool)
    sel = manifold.select(X, mask, max_samples=200, seed=0)
    assert len(sel.weight) == len(sel.index)
    assert np.all(sel.weight > 0)
    assert sel.weight.sum() == pytest_approx(len(sel.index))


def test_selection_is_reproducible_under_same_seed():
    X = _blob()
    mask = np.ones(len(X), dtype=bool)
    a = manifold.select(X, mask, max_samples=200, seed=7)
    b = manifold.select(X, mask, max_samples=200, seed=7)
    np.testing.assert_array_equal(a.index, b.index)
    np.testing.assert_array_equal(a.weight, b.weight)


def pytest_approx(value):
    import pytest
    return pytest.approx(value, rel=1e-9)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_manifold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.manifold'`

- [ ] **Step 3: `sd/manifold.py` 구현**

```python
"""On-manifold 표집. 계획서 §3 S2.

원칙 한 줄 — **넷이 모르는 곳의 답을 수식으로 만들지 않는다.**

슬라이스에서는 세 선택지 중 가장 안전한 것을 쓴다 (계획서 §3 S2 ①):
관측을 재사용하고 합성 상태를 만들지 않는다. 관측이 2,860만 호가틱이므로
밀도가 부족한 것은 전체가 아니라 꼬리뿐이고, 꼬리는 가중치로 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRUST_QUANTILE = 0.99      # 마할라노비스 거리 이 분위 안이면 on-manifold
WEIGHT_CLIP = 10.0         # 꼬리 가중치 상한. 한 점이 손실을 지배하지 않게


@dataclass(frozen=True)
class Selection:
    index: np.ndarray
    weight: np.ndarray
    on_manifold: np.ndarray


def _mahalanobis(X: np.ndarray) -> np.ndarray:
    """공분산 구조를 반영한 중심 거리. 특이행렬이면 유사역행렬로 물러난다."""
    centered = X - X.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    covariance = np.atleast_2d(covariance)
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0.0))


def select(X: np.ndarray, mask: np.ndarray, max_samples: int, seed: int) -> Selection:
    """학습에 쓸 행과 가중치.

    `probe`(신뢰 반경 밖)는 버리지 않고 플래그만 단다 — H2 검증에 통제된
    외삽 질의가 필요하기 때문이다. 다만 **SR 적합에는 쓰지 않는다.**
    """
    X = np.asarray(X, dtype=float)
    usable = np.flatnonzero(mask & np.all(np.isfinite(X), axis=1))
    if usable.size == 0:
        raise ValueError("사용 가능한 행이 없다")

    distance = _mahalanobis(X[usable])
    radius = float(np.quantile(distance, TRUST_QUANTILE))
    on_manifold = distance <= radius

    # 층화 가중치: 밀도가 낮은 곳(거리 상위)에 더 큰 가중치. 상한으로 자른다.
    ranks = distance.argsort().argsort() / max(len(distance) - 1, 1)
    weight = np.clip(1.0 + 4.0 * ranks, 1.0, WEIGHT_CLIP)

    if usable.size > max_samples:
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(usable.size, size=int(max_samples), replace=False))
    else:
        chosen = np.arange(usable.size)

    picked_weight = weight[chosen]
    picked_weight = picked_weight / picked_weight.sum() * len(chosen)
    return Selection(index=usable[chosen], weight=picked_weight,
                     on_manifold=on_manifold[chosen])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_manifold.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/manifold.py tests/test_manifold.py
git commit -m "feat: on-manifold 표집 — 신뢰 반경과 꼬리 가중치

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 8: 교사 인터페이스와 얕은 구현

DESIGN.md D5·D13. 병목을 가진 교사를 두되, SR 은 **feature 공간**에서 교사 출력을 근사한다.

**Files:**
- Create: `0902/sd/teacher/__init__.py`
- Create: `0902/sd/teacher/base.py`
- Create: `0902/sd/teacher/shallow.py`
- Test: `0902/tests/test_teacher.py`

> **유예:** DESIGN.md §3 이 나열한 `sd/teacher/deeplob.py` 는 **이 계획에서 만들지 않는다.**
> 슬라이스는 배관 검증이 목적이고(D3), GPU 여유가 카드당 10.7GB 뿐이다. `Teacher`
> 프로토콜만 지키면 나중에 파일 하나로 꽂힌다. 빠뜨린 것이 아니라 미룬 것이다.

**Interfaces:**
- Consumes: 무차원 행렬 `X`, `sd.labels.Labels`, `sd.manifold.Selection`
- Produces:
  - `sd.teacher.base.Teacher` (Protocol) — `z(X) -> (n, d)`, `predict_path(X) -> (n,)`, `predict_fill(X) -> (n,)`
  - `sd.teacher.shallow.ShallowMLP(n_features: int, bottleneck: int, seed: int)` — `fit(X, y_path, y_fill, weight, epochs=200) -> ShallowMLP`, 위 세 메서드 구현
  - `sd.teacher.shallow.ShallowMLP.bottleneck: int`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_teacher.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher.shallow import ShallowMLP  # noqa: E402


def _linear_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y_path = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    y_fill = (X[:, 2] > 0.0).astype(float)
    return X, y_path, y_fill


def test_bottleneck_shape_is_respected():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=50)
    assert model.z(X).shape == (len(X), 2)
    assert model.bottleneck == 2


def test_predictions_have_right_shape_and_range():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=50)
    path = model.predict_path(X)
    fill = model.predict_fill(X)
    assert path.shape == fill.shape == (len(X),)
    assert np.all((fill >= 0.0) & (fill <= 1.0))


def test_teacher_learns_a_linear_signal():
    X, y_path, y_fill = _linear_problem()
    model = ShallowMLP(n_features=4, bottleneck=2, seed=0).fit(
        X, y_path, y_fill, np.ones(len(X)), epochs=400)
    predicted = model.predict_path(X)
    correlation = float(np.corrcoef(predicted, y_path)[0, 1])
    assert correlation > 0.9


def test_same_seed_gives_same_model():
    X, y_path, y_fill = _linear_problem()
    kwargs = dict(n_features=4, bottleneck=2, seed=3)
    a = ShallowMLP(**kwargs).fit(X, y_path, y_fill, np.ones(len(X)), epochs=30)
    b = ShallowMLP(**kwargs).fit(X, y_path, y_fill, np.ones(len(X)), epochs=30)
    np.testing.assert_allclose(a.predict_path(X), b.predict_path(X), rtol=1e-6)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_teacher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.teacher'`

- [ ] **Step 3: `sd/teacher/base.py` 구현**

```python
"""교사 계약. 내부는 교체 가능하되 이 세 메서드는 고정이다 (DESIGN.md D5).

`z` 는 병목이고 진단·H4 ablation 에만 쓴다. **SR 의 입력이 아니다** — `z` 는
Catalog feature 가 아니라 백테스트가 계산할 수 없기 때문이다 (D13).
SR 은 `predict_path` 를 타깃으로 삼아 feature 공간에서 적합한다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Teacher(Protocol):
    bottleneck: int

    def z(self, X: np.ndarray) -> np.ndarray: ...
    def predict_path(self, X: np.ndarray) -> np.ndarray: ...
    def predict_fill(self, X: np.ndarray) -> np.ndarray: ...
```

- [ ] **Step 4: `sd/teacher/shallow.py` 구현**

```python
"""배관 검증용 얕은 교사. 본 실험에서는 DeepLOBCompact 로 교체한다 (DESIGN.md D3).

이 모델의 성능은 의미가 없다. 확인하는 것은 **병목을 가진 교사가 두 헤드를 내고
그 출력이 SR 로 흘러간다**는 계약뿐이다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ShallowMLP:
    def __init__(self, n_features: int, bottleneck: int, seed: int = 0,
                 hidden: int = 32, l1: float = 1e-4) -> None:
        self.bottleneck = int(bottleneck)
        self.l1 = float(l1)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(int(n_features), hidden), nn.ReLU(),
            nn.Linear(hidden, self.bottleneck))
        self._head_path = nn.Linear(self.bottleneck, 1)
        self._head_fill = nn.Linear(self.bottleneck, 1)
        self._mean = np.zeros(int(n_features))
        self._scale = np.ones(int(n_features))

    # -- 학습 ---------------------------------------------------------------
    def fit(self, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
            weight: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> "ShallowMLP":
        torch.manual_seed(self._seed)
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

        inputs = torch.tensor(self._standardise(X), dtype=torch.float32)
        target_path = torch.tensor(np.asarray(y_path, dtype=float), dtype=torch.float32)
        target_fill = torch.tensor(np.asarray(y_fill, dtype=float), dtype=torch.float32)
        weights = torch.tensor(np.asarray(weight, dtype=float), dtype=torch.float32)

        parameters = (list(self._encoder.parameters())
                      + list(self._head_path.parameters())
                      + list(self._head_fill.parameters()))
        optimiser = torch.optim.Adam(parameters, lr=lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for _ in range(int(epochs)):
            optimiser.zero_grad()
            latent = self._encoder(inputs)
            path = self._head_path(latent).squeeze(-1)
            fill = self._head_fill(latent).squeeze(-1)
            loss_path = (weights * (path - target_path) ** 2).mean()
            loss_fill = (weights * bce(fill, target_fill)).mean()
            # 병목에 L1. Cranmer 방법의 핵심이 이 한 줄이다.
            loss = loss_path + loss_fill + self.l1 * latent.abs().mean()
            loss.backward()
            optimiser.step()
        return self

    # -- 추론 ---------------------------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def _latent(self, X: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self._encoder(torch.tensor(self._standardise(X), dtype=torch.float32))

    def z(self, X: np.ndarray) -> np.ndarray:
        return self._latent(X).numpy()

    def predict_path(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._head_path(self._latent(X)).squeeze(-1).numpy()

    def predict_fill(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self._head_fill(self._latent(X))).squeeze(-1).numpy()
```

`0902/sd/teacher/__init__.py`:

```python
from .base import Teacher
from .shallow import ShallowMLP

__all__ = ["Teacher", "ShallowMLP"]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_teacher.py -v`
Expected: 4 passed

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/teacher/ tests/test_teacher.py
git commit -m "feat: 병목 교사 인터페이스와 얕은 구현

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 9: SR 백엔드 인터페이스와 최소 구현

DESIGN.md D4. PySR 설치 성패와 배관 진척을 분리한다.

**Files:**
- Create: `0902/sd/sr/__init__.py`
- Create: `0902/sd/sr/base.py`
- Create: `0902/sd/sr/naive.py`
- Test: `0902/tests/test_sr.py`

> **유예:** DESIGN.md §3 이 나열한 `sd/sr/pysr_backend.py` 는 **이 계획에서 만들지 않는다.**
> PySR 은 Julia 백엔드라 설치가 무겁고 실패할 수 있는데, 배관이 거기 막히면 안 된다(D4).
> `SRBackend` 프로토콜만 지키면 나중에 어댑터 하나로 꽂힌다. 빠뜨린 것이 아니라 미룬 것이다.

**Interfaces:**
- Consumes: 무차원 행렬 `X`, 교사 출력 `y`, 가중치 `w`, 열 이름
- Produces:
  - `sd.sr.base.Candidate` — frozen dataclass: `expr: sympy.Expr`, `complexity: int`, `in_sample_score: float`, `backend: str`, `seed: int`
  - `sd.sr.base.SRBackend` (Protocol) — `fit(X, y, w, feature_names) -> list[Candidate]`
  - `sd.sr.naive.NaiveBackend(seed: int, max_candidates: int = 12)` — 템플릿 격자 탐색. `backend` 필드에 `"naive"` 를 넣는다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_sr.py`:

```python
import sys
from pathlib import Path

import numpy as np
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.sr.naive import NaiveBackend  # noqa: E402


def _problem(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    names = ("book_imbalance", "queue_imbalance_best", "signed_aggr_flow_20")
    y = 3.0 * X[:, 0] - 2.0 * X[:, 2]
    return X, y, np.ones(n), names


def test_candidates_are_marked_with_backend_name():
    X, y, w, names = _problem()
    for candidate in NaiveBackend(seed=0).fit(X, y, w, names):
        assert candidate.backend == "naive"
        assert candidate.seed == 0


def test_expressions_use_only_given_feature_symbols():
    X, y, w, names = _problem()
    allowed = set(names)
    for candidate in NaiveBackend(seed=0).fit(X, y, w, names):
        used = {str(s) for s in candidate.expr.free_symbols}
        assert used <= allowed


def test_recovers_a_linear_signal_with_high_score():
    X, y, w, names = _problem()
    best = max(NaiveBackend(seed=0).fit(X, y, w, names),
               key=lambda c: c.in_sample_score)
    assert best.in_sample_score > 0.95


def test_complexity_is_positive_and_pareto_is_sorted():
    X, y, w, names = _problem()
    candidates = NaiveBackend(seed=0).fit(X, y, w, names)
    assert candidates
    assert all(c.complexity > 0 for c in candidates)
    assert [c.complexity for c in candidates] == sorted(c.complexity for c in candidates)


def test_same_seed_is_reproducible():
    X, y, w, names = _problem()
    first = [sympy.srepr(c.expr) for c in NaiveBackend(seed=1).fit(X, y, w, names)]
    second = [sympy.srepr(c.expr) for c in NaiveBackend(seed=1).fit(X, y, w, names)]
    assert first == second
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_sr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.sr'`

- [ ] **Step 3: `sd/sr/base.py` 구현**

```python
"""SR 백엔드 계약 (DESIGN.md D4).

`backend` 필드가 반드시 채워져야 한다. `naive` 로 나온 진입식은 **어떤 가설
판정에도 쓰지 않는다** — 배관 검증용이다. 리포트가 그것을 눈에 띄게 표시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import sympy


@dataclass(frozen=True)
class Candidate:
    expr: sympy.Expr
    complexity: int
    in_sample_score: float
    backend: str
    seed: int


class SRBackend(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]: ...


def complexity_of(expr: sympy.Expr) -> int:
    """AST 노드 수. Pareto 의 x 축이다."""
    return sum(1 for _ in sympy.preorder_traversal(expr))


def weighted_r2(y: np.ndarray, prediction: np.ndarray, w: np.ndarray) -> float:
    """가중 R². 가중치를 손실에 명시적으로 넘긴다 (계획서 §3 S2 ③)."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(w)
    if valid.sum() < 2:
        return float("-inf")
    y, prediction, w = y[valid], prediction[valid], w[valid]
    mean = float(np.average(y, weights=w))
    total = float(np.average((y - mean) ** 2, weights=w))
    residual = float(np.average((y - prediction) ** 2, weights=w))
    return 1.0 - residual / total if total > 0 else float("-inf")
```

- [ ] **Step 4: `sd/sr/naive.py` 구현**

```python
"""배관 검증용 최소 SR 백엔드 (DESIGN.md D4).

템플릿 격자를 훑고 상수만 최소제곱으로 맞춘다. 진짜 탐색이 아니다 —
컴파일러가 다뤄야 할 **수식의 모양**(선형결합·비·sqrt·tanh·곱)을 전부
내보내는 것이 목적이다. 본 실험에서는 PySRBackend 로 교체한다.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import sympy

from .base import Candidate, complexity_of, weighted_r2

# (이름, 심볼 몇 개를 쓰는가, 수식 만드는 함수)
TEMPLATES: tuple[tuple[str, int, Callable[..., sympy.Expr]], ...] = (
    ("linear1", 1, lambda a: a),
    ("neg1", 1, lambda a: -a),
    ("abs1", 1, lambda a: sympy.Abs(a)),
    ("linear2", 2, lambda a, b: a + b),
    ("diff2", 2, lambda a, b: a - b),
    ("product2", 2, lambda a, b: a * b),
    ("ratio2", 2, lambda a, b: a / (1 + sympy.Abs(b))),
    ("sqrt_abs", 1, lambda a: sympy.sqrt(sympy.Abs(a))),
    ("tanh1", 1, lambda a: sympy.tanh(a)),
    ("linear3", 3, lambda a, b, c: a + b + c),
    ("mixed3", 3, lambda a, b, c: a * b + c),
    ("saturated2", 2, lambda a, b: sympy.tanh(a) * b),
)


class NaiveBackend:
    name = "naive"

    def __init__(self, seed: int = 0, max_candidates: int = 12,
                 top_features: int = 4) -> None:
        self.seed = int(seed)
        self.max_candidates = int(max_candidates)
        self.top_features = int(top_features)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            feature_names: Sequence[str]) -> list[Candidate]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        names = tuple(str(n) for n in feature_names)

        ranked = self._rank_features(X, y, w, names)
        symbols = {name: sympy.Symbol(name) for name in names}
        found: list[Candidate] = []

        for _label, arity, build in TEMPLATES:
            for combo in self._combinations(ranked, arity):
                expr = build(*[symbols[name] for name in combo])
                scored = self._score(expr, X, y, w, names)
                if scored is not None:
                    found.append(scored)

        # 같은 정규형은 하나만 남긴다 — 중복은 N 회계를 부풀린다.
        unique: dict[str, Candidate] = {}
        for candidate in found:
            key = sympy.srepr(sympy.simplify(candidate.expr))
            best = unique.get(key)
            if best is None or candidate.in_sample_score > best.in_sample_score:
                unique[key] = candidate

        ordered = sorted(unique.values(),
                         key=lambda c: (-c.in_sample_score, c.complexity))
        top = ordered[: self.max_candidates]
        return sorted(top, key=lambda c: c.complexity)

    # -- 내부 ---------------------------------------------------------------
    def _rank_features(self, X, y, w, names) -> tuple[str, ...]:
        """단변량 가중 상관 절댓값 상위. 결정론적이다."""
        scores = []
        for j, name in enumerate(names):
            column = X[:, j]
            valid = np.isfinite(column) & np.isfinite(y)
            if valid.sum() < 10 or np.std(column[valid]) == 0:
                scores.append((0.0, name))
                continue
            scores.append((abs(float(np.corrcoef(column[valid], y[valid])[0, 1])), name))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return tuple(name for _score, name in scores[: self.top_features])

    @staticmethod
    def _combinations(ranked: tuple[str, ...], arity: int) -> list[tuple[str, ...]]:
        import itertools
        if arity > len(ranked):
            return []
        return [tuple(c) for c in itertools.combinations(ranked, arity)]

    def _score(self, expr, X, y, w, names) -> Candidate | None:
        """상수 스케일·절편만 최소제곱으로 맞춘다. 구조는 템플릿이 정한다."""
        try:
            function = sympy.lambdify([sympy.Symbol(n) for n in names], expr, "numpy")
            raw = np.asarray(function(*[X[:, j] for j in range(len(names))]), dtype=float)
        except Exception:
            return None
        raw = np.broadcast_to(raw, y.shape).astype(float)
        valid = np.isfinite(raw) & np.isfinite(y) & np.isfinite(w)
        if valid.sum() < 10 or np.std(raw[valid]) == 0:
            return None
        design = np.column_stack([raw[valid], np.ones(valid.sum())])
        sqrt_w = np.sqrt(w[valid])
        coefficients, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y[valid] * sqrt_w,
                                           rcond=None)
        slope, intercept = float(coefficients[0]), float(coefficients[1])
        prediction = np.full_like(y, np.nan)
        prediction[valid] = slope * raw[valid] + intercept
        score = weighted_r2(y, prediction, w)
        if not np.isfinite(score):
            return None
        # 상수는 진입식에서 임계로 흡수되므로 구조에 남기지 않는다 (계획서 S3 단조 흡수).
        return Candidate(expr=expr, complexity=complexity_of(expr),
                         in_sample_score=float(score), backend=self.name, seed=self.seed)
```

`0902/sd/sr/__init__.py`:

```python
from .base import Candidate, SRBackend, complexity_of, weighted_r2
from .naive import NaiveBackend

__all__ = ["Candidate", "SRBackend", "NaiveBackend", "complexity_of", "weighted_r2"]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_sr.py -v`
Expected: 5 passed

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/sr/ tests/test_sr.py
git commit -m "feat: SR 백엔드 인터페이스와 최소 템플릿 구현

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 10: 컴파일러 ① — 정규형 환원 (단조 흡수)

계획서 §3 S3. 진입식은 `s(x) > τ` 이므로 최외곽 순증가 껍질은 임계로 흡수된다.

**Files:**
- Create: `0902/sd/compile/__init__.py`
- Create: `0902/sd/compile/normalize.py`
- Test: `0902/tests/test_compile_normalize.py`

**Interfaces:**
- Consumes: `sympy.Expr`
- Produces: `sd.compile.normalize.strip_monotone(expr: sympy.Expr) -> tuple[sympy.Expr, list[str]]` — 껍질을 벗긴 수식과 벗긴 껍질 이름 목록 (바깥→안 순서). `sd.compile.normalize.normal_form(expr) -> str` — 중복 병합용 정규형 키

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다 (D11 — 테스트 우선)**

`0902/tests/test_compile_normalize.py`:

```python
import sys
from pathlib import Path

import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.compile import normalize  # noqa: E402

a, b = sympy.symbols("book_imbalance ofi_depth_5")


def test_outer_sqrt_is_absorbed():
    stripped, shells = normalize.strip_monotone(sympy.sqrt(sympy.Abs(a)))
    assert stripped == sympy.Abs(a)
    assert shells == ["sqrt"]


def test_outer_tanh_is_absorbed():
    stripped, shells = normalize.strip_monotone(sympy.tanh(a + b))
    assert stripped == a + b
    assert shells == ["tanh"]


def test_positive_scale_and_offset_are_absorbed():
    stripped, shells = normalize.strip_monotone(3 * a + 5)
    assert stripped == a
    assert set(shells) == {"scale", "offset"}


def test_negative_scale_is_not_absorbed():
    """부호가 뒤집히면 부등호 방향이 바뀐다. 임계로 흡수할 수 없다."""
    stripped, shells = normalize.strip_monotone(-3 * a)
    assert stripped == -3 * a
    assert shells == []


def test_inner_sqrt_is_kept():
    expr = sympy.sqrt(sympy.Abs(a)) + b
    stripped, shells = normalize.strip_monotone(expr)
    assert stripped == expr
    assert shells == []


def test_nested_shells_are_stripped_outermost_first():
    stripped, shells = normalize.strip_monotone(sympy.tanh(2 * sympy.sqrt(sympy.Abs(a))))
    assert stripped == sympy.Abs(a)
    assert shells[0] == "tanh"
    assert "sqrt" in shells


def test_monotone_variants_share_one_normal_form():
    assert normalize.normal_form(a) == normalize.normal_form(3 * a + 5)
    assert normalize.normal_form(a) == normalize.normal_form(sympy.tanh(a))
    assert normalize.normal_form(a) != normalize.normal_form(-a)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.compile'`

- [ ] **Step 3: `sd/compile/normalize.py` 구현**

```python
"""정규형 환원 — 단조 흡수. 계획서 §3 S3.

    entry(x) = [ s(x) > τ ]
    s = g(u), g 가 순증가면  [g(u) > τ] ≡ [u > g⁻¹(τ)]

최외곽 순증가 껍질은 **구조가 아니라 임계**다. 벗겨서 중복을 병합하면
Pareto 후보 수 K 가 줄고, 그만큼 다중 검정 부담(계획서 §3 S4)이 줄어든다.

부호를 뒤집는 변환은 벗기지 않는다 — 부등호 방향이 바뀌기 때문이다.
"""

from __future__ import annotations

import sympy

# 정의역 전체에서 순증가인 단항 **클래스**만. `sympy.sqrt` 는 클래스가 아니라
# 함수라서 `isinstance` 에 넣을 수 없다 — 제곱근은 `Pow(x, 1/2)` 로 잡는다.
MONOTONE_UNARY = {sympy.tanh: "tanh", sympy.atan: "atan", sympy.log: "log"}


def strip_monotone(expr: sympy.Expr) -> tuple[sympy.Expr, list[str]]:
    """최외곽 순증가 껍질을 벗긴다. 바깥에서 안으로 반복한다."""
    shells: list[str] = []
    current = sympy.sympify(expr)
    while True:
        peeled, name = _peel_once(current)
        if name is None:
            return current, shells
        shells.append(name)
        current = peeled


def _peel_once(expr: sympy.Expr) -> tuple[sympy.Expr, str | None]:
    if isinstance(expr, sympy.Add):
        constants = [t for t in expr.args if t.is_number]
        rest = [t for t in expr.args if not t.is_number]
        if constants and rest:
            return sympy.Add(*rest), "offset"
    if isinstance(expr, sympy.Mul):
        constants = [t for t in expr.args if t.is_number]
        rest = [t for t in expr.args if not t.is_number]
        # 양수 상수배만. 음수면 부등호가 뒤집힌다.
        if constants and rest and sympy.Mul(*constants).is_positive:
            return sympy.Mul(*rest), "scale"
    if isinstance(expr, sympy.Pow) and expr.exp == sympy.Rational(1, 2):
        return expr.base, "sqrt"
    for function, name in MONOTONE_UNARY.items():
        if isinstance(expr, function):
            return expr.args[0], name
    return expr, None


def normal_form(expr: sympy.Expr) -> str:
    """중복 병합용 키. 단조 변형끼리는 같은 키를 낸다."""
    stripped, _shells = strip_monotone(expr)
    return sympy.srepr(sympy.simplify(stripped))
```

`0902/sd/compile/__init__.py`:

```python
from .normalize import normal_form, strip_monotone

__all__ = ["normal_form", "strip_monotone"]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_normalize.py -v`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/compile/ tests/test_compile_normalize.py
git commit -m "feat: 컴파일러 정규형 환원 — 최외곽 단조 껍질 흡수

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 11: 컴파일러 ② — sympy → Catalog AST 번역과 정적 검사

이 실험의 유일한 신규 위험 지점이다. 실패는 조용하므로 테스트가 먼저다.

**Files:**
- Create: `0902/sd/compile/to_catalog.py`
- Create: `0902/sd/compile/check.py`
- Test: `0902/tests/test_compile_translate.py`

**Interfaces:**
- Consumes: `sd.compile.normalize.strip_monotone`, vendor `catalog`
- Produces:
  - `sd.compile.to_catalog.translate(expr: sympy.Expr) -> dict` — Catalog AST (numeric). 번역 불가면 `TranslationError`
  - `sd.compile.to_catalog.TranslationError(ValueError)`
  - `sd.compile.check.static_check(ast: dict, *, allow_unresolved: bool) -> None` — 통과하면 조용, 실패하면 `CheckError` (사유 문자열 포함)
  - `sd.compile.check.CheckError(ValueError)`

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다**

`0902/tests/test_compile_translate.py`:

```python
import sys
from pathlib import Path

import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402
from sd.compile import check, to_catalog  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402

bi = sympy.Symbol("book_imbalance")
qi = sympy.Symbol("queue_imbalance_best")


def test_symbol_becomes_primitive_node():
    assert to_catalog.translate(bi) == {"op": "primitive", "primitive_id": "book_imbalance"}


def test_sum_becomes_add_nodes():
    node = to_catalog.translate(bi + qi)
    assert node["op"] == "add"
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_product_becomes_multiply_node():
    node = to_catalog.translate(bi * qi)
    assert node["op"] == "multiply"


def test_abs_becomes_absolute_node():
    node = to_catalog.translate(sympy.Abs(bi))
    assert node == {"op": "absolute", "input": {"op": "primitive", "primitive_id": "book_imbalance"}}


def test_inner_sqrt_becomes_sqrt_node():
    node = to_catalog.translate(sympy.sqrt(sympy.Abs(bi)) + qi)
    assert node["op"] == "add"
    assert catalog.infer_expression_type(node).value_type == "numeric"


def test_tanh_becomes_tanh_node():
    node = to_catalog.translate(sympy.tanh(bi))
    assert node["op"] == "tanh"


def test_unknown_function_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.exp(bi))
    assert "exp" in str(excinfo.value)


def test_unknown_symbol_is_rejected():
    with pytest.raises(to_catalog.TranslationError) as excinfo:
        to_catalog.translate(sympy.Symbol("not_a_feature"))
    assert "not_a_feature" in str(excinfo.value)


def test_static_check_rejects_type_error():
    bad = {"op": "all", "args": [{"op": "primitive", "primitive_id": "mid_price"}]}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "bool" in str(excinfo.value)


def test_static_check_rejects_vocabulary_outside_catalog():
    bad = {"op": "primitive", "primitive_id": "invented_feature"}
    with pytest.raises(check.CheckError):
        check.static_check(bad, allow_unresolved=False)


def test_static_check_accepts_valid_boolean_entry_expression():
    good = {"op": "all", "args": [
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "book_imbalance"},
         "comparator": ">", "value": "UNRESOLVED:theta_book_imbalance"},
        {"op": "compare",
         "input": {"op": "primitive", "primitive_id": "spread_to_round_trip_cost_ratio"},
         "comparator": "<", "value": 0.6},
    ]}
    check.static_check(good, allow_unresolved=True)          # 예외가 없어야 한다
    assert catalog.infer_expression_type(good, allow_unresolved=True).value_type == "boolean"


def test_static_check_rejects_execution_binding_override():
    bad = {"op": "compare",
           "input": {"op": "primitive", "primitive_id": "book_imbalance"},
           "comparator": ">", "value": 0.5,
           "execution_binding": {"entry_execution": "MARKET"}}
    with pytest.raises(check.CheckError) as excinfo:
        check.static_check(bad, allow_unresolved=False)
    assert "실행" in str(excinfo.value)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_translate.py -v`
Expected: FAIL — `ImportError: cannot import name 'to_catalog' from 'sd.compile'`

- [ ] **Step 3: `sd/compile/to_catalog.py` 구현**

```python
"""sympy 수식을 Catalog AST 로 옮긴다. 계획서 §3 S5 ②.

번역할 수 없는 후보는 **예외로 죽이지 않고 상위에서 기록**한다. 탈락률 자체가
산출물이기 때문이다 (계획서 F7 — 컴파일 성공률 20% 미만이면 중단).
"""

from __future__ import annotations

import sympy

from .. import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402


class TranslationError(ValueError):
    """Catalog 문법으로 옮길 수 없는 수식."""


def translate(expr: sympy.Expr) -> dict:
    """numeric 을 내는 Catalog AST. boolean 으로 감싸는 것은 threshold 단계가 한다."""
    return _walk(sympy.sympify(expr))


def _walk(node: sympy.Expr) -> dict:
    if isinstance(node, sympy.Symbol):
        name = _catalog.resolve(str(node))
        if name not in _catalog.FEATURES:
            raise TranslationError(f"Catalog 에 없는 feature: {node}")
        return {"op": "primitive", "primitive_id": name}

    if node.is_number:
        raise TranslationError(
            "상수만으로 된 가지는 진입식이 될 수 없다. 상수는 임계로 흡수한다")

    if isinstance(node, sympy.Add):
        return _fold("add", "left", "right", [_walk(a) for a in node.args])

    if isinstance(node, sympy.Mul):
        constants = [a for a in node.args if a.is_number]
        rest = [a for a in node.args if not a.is_number]
        if not rest:
            raise TranslationError("상수 곱만 남았다")
        inner = _fold("multiply", "left", "right", [_walk(a) for a in rest])
        if not constants:
            return inner
        # 상수 배율의 **크기**는 분위 임계 아래에서 뜻이 없다.
        #   q 분위(c·u) = c · q 분위(u)  (c > 0) 이므로 [c·u > q(c·u)] ≡ [u > q(u)]
        # 따라서 크기는 버리고 **부호만** 남긴다. 부호는 부등호 방향을 바꾼다.
        if sympy.Mul(*constants).is_negative:
            return {"op": "negate", "input": inner}
        return inner

    if isinstance(node, sympy.Pow):
        if node.exp == sympy.Rational(1, 2):
            return {"op": "sqrt", "input": _walk(node.base)}
        if node.exp == 2:
            child = _walk(node.base)
            return {"op": "multiply", "left": child, "right": child}
        raise TranslationError(
            f"지원하지 않는 지수: {node.exp}. 역수는 ratio/divide 로 명시해야 "
            "분모 0 정책이 붙는다")

    if isinstance(node, sympy.Abs):
        return {"op": "absolute", "input": _walk(node.args[0])}
    if isinstance(node, sympy.tanh):
        return {"op": "tanh", "input": _walk(node.args[0])}
    if isinstance(node, sympy.log):
        raise TranslationError("log 는 log1p 로 바꿔 쓴다. 인수 양수성이 보장되지 않는다")

    raise TranslationError(f"번역할 수 없는 노드: {type(node).__name__} ({node})")


def _fold(op: str, left_key: str, right_key: str, children: list[dict]) -> dict:
    """이항 연산자를 왼쪽으로 접는다. Catalog 에는 n항 산술이 없다."""
    if not children:
        raise TranslationError(f"{op} 에 인자가 없다")
    folded = children[0]
    for child in children[1:]:
        folded = {"op": op, left_key: folded, right_key: child}
    return folded
```

- [ ] **Step 4: `sd/compile/check.py` 구현**

```python
"""정적 검사. 계획서 §3 S5 ③.

이것이 파이프라인의 진짜 게이트다. S1 게이트가 교사 오염을 막는다면 여기는
**수식이 실행 계약을 몰래 바꾸는 것**을 막는다. 후자가 더 은밀하다 — 청산을
살짝 유리하게 바꾼 수식은 백테스트에서 아름답게 나온다.
"""

from __future__ import annotations

from typing import Any, Mapping

from .. import config

_framework = config.load_framework()
from framework import catalog as _catalog  # noqa: E402

# 진입식이 선언해서는 안 되는 키. 실행은 정본이 정한다.
FORBIDDEN_KEYS = ("execution_binding", "exit", "exit_rule", "stop_gross_bps",
                  "trailing_drawdown_gross_bps", "horizon_seconds", "fee_bps")


class CheckError(ValueError):
    """정적 검사 탈락. 사유가 메시지에 들어 있다."""


def static_check(ast: Mapping[str, Any], *, allow_unresolved: bool) -> None:
    _reject_execution_override(ast, "$")
    try:
        _catalog.infer_expression_type(ast, allow_unresolved=allow_unresolved)
    except _catalog.ExpressionError as error:
        raise CheckError(str(error)) from error
    except (KeyError, ValueError) as error:
        raise CheckError(f"어휘 또는 값 오류: {error}") from error


def _reject_execution_override(node: Any, path: str) -> None:
    """진입식이 실행 계약을 정하려 들면 조용히 넘어가지 않는다."""
    if isinstance(node, Mapping):
        for key in FORBIDDEN_KEYS:
            if key in node:
                raise CheckError(
                    f"{path}: 진입식이 실행 결속 {key!r} 를 선언했다. "
                    "청산·큐·비용은 정본이 정하고 진입식은 정하지 못한다")
        for key, value in node.items():
            _reject_execution_override(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _reject_execution_override(value, f"{path}[{i}]")
```

`0902/sd/compile/__init__.py` 를 덮어쓴다.

```python
from .check import CheckError, static_check
from .normalize import normal_form, strip_monotone
from .to_catalog import TranslationError, translate

__all__ = ["normal_form", "strip_monotone", "translate", "TranslationError",
           "static_check", "CheckError"]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_translate.py -v`
Expected: 12 passed

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/compile/ tests/test_compile_translate.py
git commit -m "feat: sympy → Catalog AST 번역과 정적 검사

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 12: 컴파일러 ③ — 임계 부착과 파이프라인

numeric AST 를 boolean 진입식으로 바꾸고 분위 격자로 전개한다.

**Files:**
- Create: `0902/sd/compile/threshold.py`
- Create: `0902/sd/compile/pipeline.py`
- Test: `0902/tests/test_compile_pipeline.py`

**Interfaces:**
- Consumes: `sd.sr.base.Candidate`, `sd.compile.{normalize,to_catalog,check}`
- Produces:
  - `sd.compile.threshold.THETA_PREFIX = "UNRESOLVED:theta_"`
  - `sd.compile.threshold.attach(numeric_ast: dict, parameter: str) -> dict` — `compare(numeric_ast, ">", "UNRESOLVED:theta_<parameter>")`
  - `sd.compile.threshold.contract_template(entry_ast: dict, parameter: str) -> dict` — `run_backtest` 가 받는 계약
  - `sd.compile.threshold.parameter_table(contract_ids, symbols, date, parameter, grid) -> dict[str, dict[str, float]]`
  - `sd.compile.pipeline.EntryExpression` / `CompileFailure` (DESIGN.md §4.1 그대로)
  - `sd.compile.pipeline.compile_candidates(candidates) -> tuple[list[EntryExpression], list[CompileFailure]]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_compile_pipeline.py`:

```python
import sys
from pathlib import Path

import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config  # noqa: E402
from sd.compile import pipeline, threshold  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402

config.load_framework()
from framework import catalog  # noqa: E402

bi = sympy.Symbol("book_imbalance")


def _candidate(expr, score=0.5):
    return Candidate(expr=expr, complexity=3, in_sample_score=score,
                     backend="naive", seed=0)


def test_attach_makes_a_boolean_expression():
    numeric = {"op": "primitive", "primitive_id": "book_imbalance"}
    entry = threshold.attach(numeric, "score")
    assert catalog.infer_expression_type(entry, allow_unresolved=True).value_type == "boolean"
    assert entry["value"] == "UNRESOLVED:theta_score"


def test_contract_template_matches_backtest_interface():
    entry = threshold.attach({"op": "primitive", "primitive_id": "book_imbalance"}, "score")
    template = threshold.contract_template(entry, "score")
    assert template["entry_program"]["signal"] == entry
    assert template["entry_program"]["warmup_ticks"] == 0
    source = template["parameter_interface"]["theta_score"]["threshold_source"]
    assert source["kind"] == "rolling_prior_100_ticks_quantile"


def test_parameter_table_keys_are_contract_symbol_date():
    table = threshold.parameter_table(
        contract_ids=("q0.70",), symbols=("005930",), date="20260316",
        parameter="score", grid=(0.70,))
    assert table == {"q0.70:005930:20260316": {"theta_score": 0.70}}


def test_compile_succeeds_for_translatable_candidate():
    ok, failed = pipeline.compile_candidates([_candidate(3 * bi + 5)])
    assert len(ok) == 1 and not failed
    assert catalog.infer_expression_type(ok[0].ast, allow_unresolved=True).value_type == "boolean"
    assert ok[0].source.backend == "naive"


def test_compile_records_failure_instead_of_raising():
    ok, failed = pipeline.compile_candidates([_candidate(sympy.exp(bi))])
    assert not ok and len(failed) == 1
    assert failed[0].stage == "translate"
    assert "exp" in failed[0].reason


def test_monotone_variants_collapse_to_one_expression():
    candidates = [_candidate(bi, 0.5), _candidate(3 * bi + 5, 0.6), _candidate(sympy.tanh(bi), 0.4)]
    ok, _failed = pipeline.compile_candidates(candidates)
    assert len(ok) == 1
    assert ok[0].source.in_sample_score == 0.6      # 같은 정규형 중 최고 점수가 남는다
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'pipeline' from 'sd.compile'`

- [ ] **Step 3: `sd/compile/threshold.py` 구현**

```python
"""임계 부착과 분위 격자 전개. 계획서 §3 S5 ④ · DESIGN.md D8·D9.

임계는 절대값이 아니라 **분위수**다. 스프레드 중앙값이 13bp 인 저마찰층과
43bp 인 고마찰층에 같은 절대 임계를 걸면 전혀 다른 사건을 고르게 된다.
"""

from __future__ import annotations

from typing import Sequence

THETA_PREFIX = "UNRESOLVED:theta_"
THRESHOLD_KIND = "rolling_prior_100_ticks_quantile"


def attach(numeric_ast: dict, parameter: str) -> dict:
    """numeric AST 를 boolean 진입식으로. 임계는 미결로 남긴다."""
    return {"op": "compare", "input": numeric_ast, "comparator": ">",
            "value": f"{THETA_PREFIX}{parameter}"}


def contract_template(entry_ast: dict, parameter: str) -> dict:
    """`canonical.run_backtest` 가 받는 계약. 실행 결속을 선언하지 않는다."""
    return {
        "entry_program": {"signal": entry_ast, "warmup_ticks": 0},
        "parameter_interface": {
            f"theta_{parameter}": {"threshold_source": {"kind": THRESHOLD_KIND}}},
    }


def contract_id(prefix: str, quantile: float) -> str:
    return f"{prefix}:q{float(quantile):.2f}"


def parameter_table(contract_ids: Sequence[str], symbols: Sequence[str], date: str,
                    parameter: str, grid: Sequence[float]) -> dict[str, dict[str, float]]:
    """`"{contract_id}:{symbol}:{date}" -> {theta: quantile}`.

    계약 ID 로 분위를 나누면 한 번의 재생 호출로 격자 전체를 돈다.
    """
    if len(contract_ids) != len(grid):
        raise ValueError("계약 ID 와 분위 격자의 길이가 다르다")
    table: dict[str, dict[str, float]] = {}
    for identifier, quantile in zip(contract_ids, grid):
        for symbol in symbols:
            table[f"{identifier}:{symbol}:{date}"] = {f"theta_{parameter}": float(quantile)}
    return table
```

- [ ] **Step 4: `sd/compile/pipeline.py` 구현**

```python
"""컴파일 파이프라인. 계획서 §3 S5 ①~④.

후보 하나가 실패해도 예외로 죽지 않는다. **실패율 자체가 산출물**이다
(계획서 F7 — Pareto 후보의 컴파일 성공률이 20% 미만이면 중단).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from ..sr.base import Candidate
from . import check, normalize, threshold, to_catalog

PARAMETER = "score"


@dataclass(frozen=True)
class EntryExpression:
    ast: dict
    thresholds: dict[str, str]
    source: Candidate
    normal_form: str
    shells: tuple[str, ...]


@dataclass(frozen=True)
class CompileFailure:
    candidate: Candidate
    stage: str
    reason: str


def compile_candidates(
        candidates: list[Candidate]) -> tuple[list[EntryExpression], list[CompileFailure]]:
    succeeded: dict[str, EntryExpression] = {}
    failed: list[CompileFailure] = []

    for candidate in candidates:
        try:
            stripped, shells = normalize.strip_monotone(candidate.expr)
        except Exception as error:                       # pragma: no cover - 방어
            failed.append(CompileFailure(candidate, "normalize", str(error)))
            continue

        try:
            numeric = to_catalog.translate(stripped)
        except to_catalog.TranslationError as error:
            failed.append(CompileFailure(candidate, "translate", str(error)))
            continue

        entry = threshold.attach(numeric, PARAMETER)
        try:
            check.static_check(entry, allow_unresolved=True)
        except check.CheckError as error:
            failed.append(CompileFailure(candidate, "check", str(error)))
            continue

        key = sympy.srepr(sympy.simplify(stripped))
        existing = succeeded.get(key)
        if existing is None or candidate.in_sample_score > existing.source.in_sample_score:
            succeeded[key] = EntryExpression(
                ast=entry,
                thresholds={f"theta_{PARAMETER}": threshold.THRESHOLD_KIND},
                source=candidate, normal_form=key, shells=tuple(shells))

    ordered = sorted(succeeded.values(), key=lambda e: (e.source.complexity, e.normal_form))
    return ordered, failed
```

`0902/sd/compile/__init__.py` 를 덮어쓴다.

```python
from .check import CheckError, static_check
from .normalize import normal_form, strip_monotone
from .pipeline import CompileFailure, EntryExpression, compile_candidates
from .threshold import attach, contract_id, contract_template, parameter_table
from .to_catalog import TranslationError, translate

__all__ = ["normal_form", "strip_monotone", "translate", "TranslationError",
           "static_check", "CheckError", "EntryExpression", "CompileFailure",
           "compile_candidates", "attach", "contract_template", "contract_id",
           "parameter_table"]
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_compile_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/compile/ tests/test_compile_pipeline.py
git commit -m "feat: 임계 부착과 컴파일 파이프라인 — 실패를 기록하고 계속한다

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 13: 정본 재생 래퍼

`canonical.run_backtest` 를 감싸고 원장이 성립하는지 검사한다.

**Files:**
- Create: `0902/sd/replay.py`
- Test: `0902/tests/test_replay.py`

**Interfaces:**
- Consumes: `sd.compile.EntryExpression`, `sd.config`
- Produces: `sd.replay.run(entries, symbols, date, output_dir, grid, workers) -> ReplayResult` where

```python
@dataclass(frozen=True)
class ReplayResult:
    ledger_path: Path
    manifest: dict
    contract_ids: tuple[str, ...]
    attempts: int          # = len(entries) * len(grid). 계획서 §4 의 N 회계
```

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_replay.py`:

```python
import sys
from pathlib import Path

import pandas as pd
import pytest
import sympy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import config, replay  # noqa: E402
from sd.compile import compile_candidates  # noqa: E402
from sd.sr.base import Candidate  # noqa: E402


def _entries():
    expr = sympy.Symbol("book_imbalance")
    candidate = Candidate(expr=expr, complexity=1, in_sample_score=0.5,
                          backend="naive", seed=0)
    ok, _failed = compile_candidates([candidate])
    return ok


def test_attempts_counts_entries_times_grid(tmp_path):
    entries = _entries()
    assert replay.attempt_count(entries, config.QUANTILE_GRID) == len(config.QUANTILE_GRID)


@pytest.mark.slow
def test_replay_produces_a_ledger_with_no_errors(tmp_path):
    result = replay.run(_entries(), symbols=("000660", "035420"), date=config.DATE,
                        output_dir=tmp_path, grid=(0.85,), workers=2)
    assert result.manifest["errors"] == []
    assert result.manifest["canonical_backtest_profile_hash"] == config.PROFILE_HASH
    frame = pd.read_parquet(result.ledger_path)
    assert len(frame) > 0
    assert set(frame["status"].unique()) <= {"FILLED", "UNFILLED", "CENSORED"}
    assert result.attempts == 1
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_replay.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.replay'`

- [ ] **Step 3: `sd/replay.py` 구현**

```python
"""정본 재생. 계획서 §3 S5 ⑤.

실행 설정을 인자로 받지 않는다 — 정본 프로필이 정한다. 여기서 하는 일은
계약을 모아 넘기고, 돌아온 원장이 성립하는지 확인하는 것뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config
from .compile import EntryExpression, contract_id, contract_template, parameter_table
from .compile.pipeline import PARAMETER

_framework = config.load_framework()
from framework import canonical as _canonical  # noqa: E402


@dataclass(frozen=True)
class ReplayResult:
    ledger_path: Path
    manifest: dict
    contract_ids: tuple[str, ...]
    attempts: int


def attempt_count(entries: Sequence[EntryExpression], grid: Sequence[float]) -> int:
    """계획서 §4 가 보고를 요구하는 시도 횟수 N."""
    return len(entries) * len(grid)


def run(entries: Sequence[EntryExpression], symbols: Sequence[str], date: str,
        output_dir: Path, grid: Sequence[float] = config.QUANTILE_GRID,
        workers: int = config.REPLAY_WORKERS) -> ReplayResult:
    if not entries:
        raise ValueError("재생할 진입식이 없다")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contracts: dict[str, dict] = {}
    members: dict[str, tuple[str, ...]] = {}
    table: dict[str, dict[str, float]] = {}

    for index, entry in enumerate(entries):
        prefix = f"e{index:03d}"
        identifiers = tuple(contract_id(prefix, q) for q in grid)
        template = contract_template(entry.ast, PARAMETER)
        for identifier in identifiers:
            contracts[identifier] = template
            members[identifier] = tuple(symbols)
        table.update(parameter_table(identifiers, symbols, date, PARAMETER, grid))

    ledger_path = output_dir / "ledger.parquet"
    manifest = _canonical.run_backtest(
        contracts=contracts, members=members, dates=[str(date)],
        output=ledger_path, parameter_table=table,
        workers=int(workers), root=config.TICK_ROOT)

    if manifest.get("errors"):
        raise RuntimeError(
            "재생에 오류가 있다. 원장 일부만 집계하면 조용히 틀린 수가 나온다:\n  "
            + "\n  ".join(str(e) for e in manifest["errors"]))

    return ReplayResult(ledger_path=ledger_path, manifest=manifest,
                        contract_ids=tuple(sorted(contracts)),
                        attempts=attempt_count(entries, grid))
```

- [ ] **Step 4: 빠른 테스트 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_replay.py -v -m "not slow"`
Expected: 1 passed

- [ ] **Step 5: 실데이터 재생 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_replay.py -v -m slow`
Expected: 1 passed (약 1~2분)

- [ ] **Step 6: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/replay.py tests/test_replay.py
git commit -m "feat: 정본 백테스트 재생 래퍼

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 14: 선택 규칙 — 분모 함정 방어

계획서 §3 S4. 거래를 거의 안 하는 후보가 이기는 것을 막는다.

**Files:**
- Create: `0902/sd/select.py`
- Test: `0902/tests/test_select.py`

**Interfaces:**
- Consumes: 원장 `pd.DataFrame`
- Produces:
  - `sd.select.MIN_SCORABLE = 200`
  - `sd.select.summarise(ledger) -> pd.DataFrame` — index=`contract_id`, columns=`decisions, scorable, fills, censored, unfilled, total_net_bps, bps_per_decision, positive` + 코호트 4열
  - `sd.select.rank(summary) -> pd.DataFrame` — `scorable >= MIN_SCORABLE` 만 남기고 `total_net_bps` 내림차순. 동률은 `bps_per_decision` → `contract_id`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`0902/tests/test_select.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import select  # noqa: E402


def _ledger(rows):
    return pd.DataFrame(rows)


def _row(contract, status, net=0.0, cohort="NET_RECOVERY"):
    return {"contract_id": contract, "status": status, "net_bps": net,
            "cohort": cohort if status == "FILLED" else None}


def test_summarise_counts_each_status_separately():
    ledger = _ledger([_row("a", "FILLED", 10.0), _row("a", "UNFILLED"),
                      _row("a", "CENSORED")])
    summary = select.summarise(ledger)
    assert summary.loc["a", "decisions"] == 3
    assert summary.loc["a", "fills"] == 1
    assert summary.loc["a", "unfilled"] == 1
    assert summary.loc["a", "censored"] == 1
    assert summary.loc["a", "scorable"] == 2       # CENSORED 는 분모에서 뺀다


def test_censored_is_excluded_from_denominator_not_zeroed():
    ledger = _ledger([_row("a", "FILLED", 20.0)] + [_row("a", "CENSORED")] * 99)
    summary = select.summarise(ledger)
    assert summary.loc["a", "scorable"] == 1
    assert summary.loc["a", "bps_per_decision"] == 20.0


def test_a_tiny_but_profitable_candidate_is_rejected():
    """계획서 §3 S4 의 분모 함정. 결정 3건에 양수인 후보는 발견이 아니다."""
    tiny = [_row("tiny", "FILLED", 500.0)] * 3
    wide = [_row("wide", "FILLED", 1.0)] * 300
    summary = select.summarise(_ledger(tiny + wide))
    ranked = select.rank(summary)
    assert "tiny" not in ranked.index
    assert ranked.index[0] == "wide"


def test_ranking_uses_total_net_not_ratio():
    """비율이 좋아도 총액이 음수면 이기지 못한다."""
    thin = [_row("thin", "FILLED", -1.0)] * 200 + [_row("thin", "UNFILLED")] * 100
    fat = [_row("fat", "FILLED", 2.0)] * 250 + [_row("fat", "UNFILLED")] * 50
    ranked = select.rank(select.summarise(_ledger(thin + fat)))
    assert ranked.index[0] == "fat"
    assert ranked.loc["fat", "total_net_bps"] == 500.0


def test_cohort_counts_are_reported():
    ledger = _ledger([_row("a", "FILLED", -5.0, "PERSISTENT_ADVERSE"),
                      _row("a", "FILLED", -1.0, "COST_INSUFFICIENT"),
                      _row("a", "FILLED", 3.0, "NET_RECOVERY")])
    summary = select.summarise(ledger)
    assert summary.loc["a", "cohort_PERSISTENT_ADVERSE"] == 1
    assert summary.loc["a", "cohort_COST_INSUFFICIENT"] == 1
    assert summary.loc["a", "cohort_NET_RECOVERY"] == 1
    assert summary.loc["a", "cohort_EARLY_RECOVERY_LATE_REVERSAL"] == 0
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_select.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.select'`

- [ ] **Step 3: `sd/select.py` 구현**

```python
"""후보 선택. 계획서 §3 S4 — 분모 함정.

정본 저장소가 계약 1,025개에서 실측한 것: 전부 손실 구간이었고, 분자가 음수면
`net/분모` 는 분모가 클수록 0 에 가까워져 좋아 보인다. 그래서 **더 많이 잃은
후보가 신호를 촘촘히 켰다는 이유로** 이겼다. 가드는 정반대로 분모를 줄여서 이긴다.

그래서 두 가지를 건다 — `scorable` 하한, 그리고 부호 판정은 `total_net_bps` 로만.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SCORABLE = 200
COHORTS = ("NET_RECOVERY", "EARLY_RECOVERY_LATE_REVERSAL",
           "COST_INSUFFICIENT", "PERSISTENT_ADVERSE")


def summarise(ledger: pd.DataFrame) -> pd.DataFrame:
    """계약별 성과. 분모가 다른 수를 나란히 들고 다닌다."""
    rows = []
    for contract, group in ledger.groupby("contract_id", sort=True):
        status = group["status"]
        fills = int((status == "FILLED").sum())
        unfilled = int((status == "UNFILLED").sum())
        censored = int((status == "CENSORED").sum())
        scorable = fills + unfilled          # CENSORED 는 관측 불가라 분모에서 뺀다
        net = float(group.loc[status == "FILLED", "net_bps"].sum())
        record = {
            "contract_id": contract,
            "decisions": int(len(group)),
            "scorable": scorable,
            "fills": fills,
            "unfilled": unfilled,
            "censored": censored,
            "total_net_bps": net,
            "bps_per_decision": (net / scorable) if scorable else np.nan,
            "positive": bool(scorable and net > 0.0),
        }
        cohorts = group.loc[status == "FILLED", "cohort"] if "cohort" in group else pd.Series(dtype=object)
        for name in COHORTS:
            record[f"cohort_{name}"] = int((cohorts == name).sum())
        rows.append(record)
    return pd.DataFrame(rows).set_index("contract_id")


def rank(summary: pd.DataFrame, min_scorable: int = MIN_SCORABLE) -> pd.DataFrame:
    """자격을 통과한 후보만, 총액 기준으로.

    비율 지표는 동률 처리에만 쓴다. 손실 구간에서 비율만 보면 방향을 잃는다.
    """
    eligible = summary[summary["scorable"] >= int(min_scorable)].copy()
    return eligible.sort_values(
        by=["total_net_bps", "bps_per_decision"],
        ascending=[False, False], kind="mergesort")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_select.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/select.py tests/test_select.py
git commit -m "feat: 선택 규칙과 분모 함정 방어

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 15: 엔드투엔드 슬라이스 · 산출물 · README

전 단계를 잇고 DESIGN.md §7 완료 기준의 산출물을 전부 쓴다.

**Files:**
- Create: `0902/sd/report.py`
- Create: `0902/run_slice.py`
- Create: `0902/README.md`
- Create: `0902/Makefile`
- Test: `0902/tests/test_report.py`

**Interfaces:**
- Consumes: 앞의 모든 모듈
- Produces:
  - `sd.report.provenance(...) -> dict` — 계획서 §5 가 요구하는 기록 전부
  - `sd.report.write(run_dir, universe, candidates, compiled, failures, ranked, provenance) -> Path`
  - `run_slice.py` CLI — `--per-stratum` `--max-samples` `--epochs` `--seed` `--bottleneck`

- [ ] **Step 1: 리포트 테스트를 쓴다**

`0902/tests/test_report.py`:

```python
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd import report  # noqa: E402


def test_provenance_records_every_required_field():
    record = report.provenance(symbols=("005930",), seed=0, sr_backend="naive",
                               grid=(0.7, 0.85, 0.95), attempts=9, bottleneck=2)
    for key in ("profile_hash", "catalog_hash", "vendor_manifest_sha256",
                "symbols", "seed", "sr_backend", "quantile_grid", "attempts",
                "bottleneck", "date", "python"):
        assert key in record, key
    assert record["attempts"] == 9


def test_report_flags_naive_backend_loudly(tmp_path):
    ranked = pd.DataFrame(
        {"decisions": [10], "scorable": [8], "fills": [3], "unfilled": [5],
         "censored": [2], "total_net_bps": [1.5], "bps_per_decision": [0.19],
         "positive": [True], "cohort_NET_RECOVERY": [1],
         "cohort_EARLY_RECOVERY_LATE_REVERSAL": [0],
         "cohort_COST_INSUFFICIENT": [1], "cohort_PERSISTENT_ADVERSE": [1]},
        index=pd.Index(["e000:q0.85"], name="contract_id"))
    path = report.write(
        tmp_path, universe={"symbols": ["005930"], "strata": {}},
        candidates=[], compiled=[], failures=[], ranked=ranked,
        prov=report.provenance(symbols=("005930",), seed=0, sr_backend="naive",
                               grid=(0.85,), attempts=1, bottleneck=2))
    text = path.read_text(encoding="utf-8")
    assert "naive" in text
    assert "배관 검증용" in text
    assert json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sd.report'`

- [ ] **Step 3: `sd/report.py` 구현**

```python
"""산출물. 계획서 §5 보고 필수 항목 · DESIGN.md §7 완료 기준.

기록은 run 이 끝난 뒤에 모으려 하면 못 모은다. 여기서 한 번에 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import config

_framework = config.load_framework()
from framework import canonical as _canonical  # noqa: E402
from framework import catalog as _catalog      # noqa: E402


def provenance(symbols: Sequence[str], seed: int, sr_backend: str,
               grid: Sequence[float], attempts: int, bottleneck: int) -> dict:
    manifest_path = config.VENDOR_ROOT / "VENDOR_MANIFEST.json"
    return {
        "schema": "sd_provenance.v1",
        "date": config.DATE,
        "python": platform.python_version(),
        "profile_id": _canonical.PROFILE_ID,
        "profile_hash": _canonical.profile_hash(),
        "catalog_hash": _catalog.catalog_hash(),
        "vendor_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest()[:16],
        "symbols": list(symbols),
        "symbol_count": len(symbols),
        "seed": int(seed),
        "sr_backend": str(sr_backend),
        "bottleneck": int(bottleneck),
        "quantile_grid": [float(q) for q in grid],
        "attempts": int(attempts),
    }


def write(run_dir: Path, universe: dict, candidates: list, compiled: list,
          failures: list, ranked: pd.DataFrame, prov: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _json(run_dir / "universe.json", universe)
    _json(run_dir / "provenance.json", prov)
    _json(run_dir / "candidates.json", [
        {"expr": str(c.expr), "complexity": c.complexity,
         "in_sample_score": c.in_sample_score, "backend": c.backend, "seed": c.seed}
        for c in candidates])
    _json(run_dir / "compile_report.json", {
        "attempted": len(candidates),
        "succeeded": len(compiled),
        "failed": len(failures),
        "success_rate": (len(compiled) / len(candidates)) if candidates else 0.0,
        "failures": [{"expr": str(f.candidate.expr), "stage": f.stage, "reason": f.reason}
                     for f in failures],
    })

    expression_dir = run_dir / "entry_expressions"
    expression_dir.mkdir(exist_ok=True)
    for index, entry in enumerate(compiled):
        _json(expression_dir / f"e{index:03d}.json", {
            "ast": entry.ast, "thresholds": entry.thresholds,
            "absorbed_shells": list(entry.shells),
            "source_expr": str(entry.source.expr),
            "source_backend": entry.source.backend,
        })

    path = run_dir / "report.md"
    path.write_text(_markdown(universe, candidates, compiled, failures, ranked, prov),
                    encoding="utf-8")
    return path


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _markdown(universe, candidates, compiled, failures, ranked, prov) -> str:
    warning = ""
    if prov["sr_backend"] == "naive":
        warning = (
            "> ⚠️ **이 run 의 SR 백엔드는 `naive` 다 — 배관 검증용 템플릿 격자이지 "
            "심볼릭 회귀가 아니다.**\n"
            "> 아래 성과 숫자는 **어떤 가설 판정에도 쓸 수 없다.** 이 run 이 검증하는 "
            "것은 배관이지 알파가 아니다 (DESIGN.md D3·D4).\n\n")

    success_rate = (len(compiled) / len(candidates)) if candidates else 0.0
    lines = [
        "# 슬라이스 실행 리포트", "",
        warning,
        "## 실행 정체성", "",
        "| 항목 | 값 |", "| --- | --- |",
        f"| 날짜 | `{prov['date']}` |",
        f"| 종목 수 | {prov['symbol_count']} |",
        f"| 정본 프로필 | `{prov['profile_id']}` / `{prov['profile_hash']}` |",
        f"| Catalog 해시 | `{prov['catalog_hash']}` |",
        f"| vendor 매니페스트 | `{prov['vendor_manifest_sha256']}` |",
        f"| SR 백엔드 | **`{prov['sr_backend']}`** |",
        f"| 병목 차원 | {prov['bottleneck']} |",
        f"| 분위 격자 | {prov['quantile_grid']} |",
        f"| **재생 시도 횟수 N** | **{prov['attempts']}** |",
        f"| 시드 | {prov['seed']} |", "",
        "## 컴파일", "",
        f"- 후보 {len(candidates)}개 중 **{len(compiled)}개 성공** "
        f"(성공률 {success_rate:.0%}), {len(failures)}개 탈락",
        "",
    ]
    if failures:
        lines += ["| 단계 | 사유 | 수식 |", "| --- | --- | --- |"]
        for failure in failures[:20]:
            reason = failure.reason.replace("\n", " ")[:80]
            lines.append(f"| `{failure.stage}` | {reason} | `{failure.candidate.expr}` |")
        lines.append("")

    lines += ["## 성과", ""]
    if ranked.empty:
        lines.append("자격(`scorable ≥ 200`)을 통과한 계약이 없다. "
                     "**분모가 모자라면 성적을 계산하지 않는다** (계획서 §3 S4).")
    else:
        columns = ["decisions", "scorable", "fills", "unfilled", "censored",
                   "total_net_bps", "bps_per_decision"]
        lines.append(ranked[columns].to_markdown())
        lines += ["", "### 손실 코호트", ""]
        cohort_columns = [c for c in ranked.columns if c.startswith("cohort_")]
        lines.append(ranked[cohort_columns].to_markdown())
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_report.py -v`
Expected: 2 passed

- [ ] **Step 5: `run_slice.py` 구현**

```python
#!/usr/bin/env python3
"""얇은 수직 슬라이스. DESIGN.md §5 데이터 흐름 그대로.

    python3 run_slice.py --per-stratum 5 --max-samples 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sd import config, dimensionless, labels, manifold, replay, report, select, ticks, universe
from sd.compile import compile_candidates
from sd.sr.naive import NaiveBackend
from sd.teacher.shallow import ShallowMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=config.SLICE_PER_STRATUM)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bottleneck", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--workers", type=int, default=config.REPLAY_WORKERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    settings = {k: v for k, v in vars(args).items()}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.RUNS_ROOT / f"{stamp}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    # S−1 우주와 층화
    all_symbols = universe.stock_symbols(config.DATE)
    stats = universe.liquidity_stats(all_symbols, config.DATE)
    strata = universe.assign_strata(stats)
    symbols = universe.slice_symbols(strata, args.per_stratum)
    print(f"[S-1] 전종목 {len(all_symbols)} → 층화 {len(strata)} → 슬라이스 {len(symbols)}")

    boundaries = strata.groupby("friction").spread_bps.agg(["min", "max", "size"])
    universe_record = {
        "date": config.DATE,
        "total_symbols": len(all_symbols),
        "stratified_symbols": len(strata),
        "symbols": list(symbols),
        "per_stratum": args.per_stratum,
        "friction_boundaries": json.loads(boundaries.to_json(orient="index")),
        "stratum_of": {s: str(strata.loc[s, "stratum"]) for s in symbols},
    }

    # S0~S2 종목별 적재 → 라벨 → 무차원 → 표집
    #
    # 체결 행이 없는 종목-일은 체결 의존 feature 가 계산되지 않아 열이 줄어든다.
    # 그런 종목을 버리면 유동성 꼬리가 통째로 사라지므로, **열의 교집합**을 쓴다.
    per_symbol: list[tuple[dict[str, np.ndarray], object]] = []
    for symbol in symbols:
        try:
            arrays = ticks.load_arrays(symbol, config.DATE)
        except (FileNotFoundError, ValueError) as error:
            print(f"[skip] {symbol}: {type(error).__name__}: {error}")
            continue
        matrix, names = ticks.feature_matrix(arrays)
        X_symbol, kept, _meta = dimensionless.transform(matrix, names)
        columns = {name: X_symbol[:, j] for j, name in enumerate(kept)}
        per_symbol.append((columns, labels.build(arrays)))

    if not per_symbol:
        raise SystemExit("적재된 종목이 없다")

    common = set(per_symbol[0][0])
    for columns, _label in per_symbol[1:]:
        common &= set(columns)
    feature_names = tuple(sorted(common))
    if not feature_names:
        raise SystemExit("모든 종목에 공통인 무차원 feature 가 없다")
    print(f"[S0] 공통 무차원 feature {len(feature_names)}: {', '.join(feature_names)}")

    X = np.vstack([np.column_stack([columns[name] for name in feature_names])
                   for columns, _label in per_symbol])
    y_path = np.concatenate([label.y_path for _columns, label in per_symbol])
    y_fill = np.concatenate([label.y_fill for _columns, label in per_symbol])
    mask = np.concatenate([label.mask for _columns, label in per_symbol])
    print(f"[S0] 행 {len(X):,} · 학습 가능 {int(mask.sum()):,}")

    selection = manifold.select(X, mask, max_samples=args.max_samples, seed=args.seed)
    print(f"[S2] 표집 {len(selection.index):,} · on-manifold "
          f"{int(selection.on_manifold.sum()):,}")

    # S1 교사 (on-manifold 만 적합에 쓴다)
    fit_rows = selection.index[selection.on_manifold]
    fit_weight = selection.weight[selection.on_manifold]
    teacher = ShallowMLP(n_features=X.shape[1], bottleneck=args.bottleneck,
                         seed=args.seed).fit(
        X[fit_rows], y_path[fit_rows], y_fill[fit_rows], fit_weight, epochs=args.epochs)
    print(f"[S1] 교사 학습 완료 (병목 {args.bottleneck})")

    # S3 SR — feature 공간에서 교사 출력을 근사한다 (DESIGN.md D13)
    target = teacher.predict_path(X[fit_rows])
    candidates = NaiveBackend(seed=args.seed).fit(
        X[fit_rows], target, fit_weight, feature_names)
    print(f"[S3] SR 후보 {len(candidates)}")

    # S5 컴파일
    compiled, failures = compile_candidates(candidates)
    rate = (len(compiled) / len(candidates)) if candidates else 0.0
    print(f"[S5] 컴파일 성공 {len(compiled)} / {len(candidates)} ({rate:.0%}), 탈락 {len(failures)}")
    if not compiled:
        raise SystemExit("컴파일된 진입식이 없다. 계획서 F7 에 해당한다")

    # S5⑤ 정본 재생
    result = replay.run(compiled, symbols=symbols, date=config.DATE,
                        output_dir=run_dir, grid=config.QUANTILE_GRID,
                        workers=args.workers)
    ledger = pd.read_parquet(result.ledger_path)
    print(f"[S5] 원장 {len(ledger):,}행 · 시도 N={result.attempts}")

    # S4 선택
    summary = select.summarise(ledger)
    ranked = select.rank(summary)
    print(f"[S4] 자격 통과 {len(ranked)} / {len(summary)}")

    prov = report.provenance(symbols=symbols, seed=args.seed,
                             sr_backend=NaiveBackend.name, grid=config.QUANTILE_GRID,
                             attempts=result.attempts, bottleneck=args.bottleneck)
    summary.to_parquet(run_dir / "summary.parquet")
    path = report.write(run_dir, universe_record, candidates, compiled, failures,
                        ranked, prov)
    print(f"[done] {time.time() - started:.1f}s → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: `Makefile` 과 `README.md` 작성**

`0902/Makefile`:

```makefile
PY := python3

.PHONY: test test-fast slice vendor-diff

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -v -m "not slow"

slice:
	$(PY) run_slice.py

vendor-diff:
	$(PY) tools/vendor_sync.py --check
```

`0902/README.md`:

```markdown
# 0902 — 심볼릭 증류 얇은 수직 슬라이스

`20260316` KRX 주식 30종목으로 데이터 적재부터 정본 백테스트 원장까지
파이프라인 전 단계를 관통시킨다.

**최종 산출은 LONG 진입 조건 하나다. 청산은 탐색하지 않는다** —
`CANONICAL_QUEUE_V9` 가 gross −120bp 손절 · 최고 ASK1 대비 −30bp 추적 ·
최대 900초 · 왕복 23bp 를 상수로 건다.

## 문서

| 문서 | 내용 |
| --- | --- |
| [`../symbolic-distillation-for-market-microstructure.md`](../symbolic-distillation-for-market-microstructure.md) | 계획서 — 무엇을 왜 |
| [`DESIGN.md`](./DESIGN.md) | 설계와 결정 기록 D1~D13 |
| [`PLAN.md`](./PLAN.md) | 구현 계획 |
| [`vendor/PATCHES.md`](./vendor/PATCHES.md) | 정본 프레임워크 사본의 변경 내역 |

## 실행

```bash
make vendor-diff     # vendor 가 원본과 PATCHES.md 외에 다르지 않은지
make test-fast       # 합성 데이터 테스트만 (수 초)
make test            # 실데이터 테스트 포함 (수 분)
make slice           # 엔드투엔드
```

산출물은 `runs/<UTC타임스탬프>-<설정해시>/` 아래에 쌓인다.

| 파일 | 내용 |
| --- | --- |
| `universe.json` | 30종목 명단과 층 경계 |
| `candidates.json` | SR 후보 전체 (선택된 것만이 아니라) |
| `compile_report.json` | 성공 목록 + **실패 목록과 단계별 사유** |
| `entry_expressions/*.json` | Catalog AST 진입식 |
| `ledger.parquet` | 정본 재생 원장 |
| `summary.parquet` | 계약별 성과 |
| `provenance.json` | 프로필·Catalog·vendor 해시, 시드, SR 백엔드, 시도 횟수 N |
| `report.md` | 사람이 읽는 요약 |

## 지금 이 슬라이스가 검증하는 것

**배관이다. 알파가 아니다.** 교사는 얕은 MLP 이고 SR 백엔드는 템플릿 격자다.
성과 숫자는 어떤 가설 판정에도 쓸 수 없다 (DESIGN.md D3·D4).
리포트가 이것을 맨 위에 표시한다.

## 다음

교사를 `DeepLOBCompact` 로 교체 → SR 을 PySR 로 교체 → E0 게이트 5종 →
전종목 캐시 → 본 실험 비교군 8종.
```

- [ ] **Step 7: 전체 테스트 실행**

Run: `cd /home/dgu/tick/symbolic/0902 && make test`
Expected: 모든 테스트 통과 (slow 포함)

- [ ] **Step 8: 엔드투엔드 실행**

Run: `cd /home/dgu/tick/symbolic/0902 && make slice`
Expected: `[done]` 으로 끝나고 `runs/<id>/report.md` 생성. DESIGN.md §7 체크리스트의 파일이 전부 있어야 한다

- [ ] **Step 9: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/report.py run_slice.py README.md Makefile tests/test_report.py
git commit -m "feat: 엔드투엔드 슬라이스와 산출물

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```

---

### Task 16: S1 교사 검증 게이트

계획서 §3 S1. **가장 위험한 실패 양식은 "Teacher 오염"이고, 그것은 결과가 좋게 나온다.** 형식적으로 통과시키지 않도록 기계가 검사한다.

**Files:**
- Create: `0902/sd/teacher/gate.py`
- Modify: `0902/sd/teacher/__init__.py`
- Modify: `0902/run_slice.py` (S1 뒤 · S3 앞에 게이트 호출을 끼운다)
- Test: `0902/tests/test_teacher_gate.py`

**Interfaces:**
- Consumes: `sd.teacher.base.Teacher`, 무차원 행렬 `X`, `sd.labels.Labels`
- Produces:
  - `sd.teacher.gate.GateResult` — frozen dataclass: `passed: bool`, `checks: dict[str, dict]`
  - `sd.teacher.gate.evaluate(teacher, X, y_path, y_fill, mask, deciles=10) -> GateResult`
  - `sd.teacher.gate.ADVERSE_SELECTION = "adverse_selection_sign"`, `FILL_CALIBRATION = "fill_calibration"`, `BEATS_CONSTANT = "beats_constant"`

- [ ] **Step 1: 실패하는 테스트를 먼저 쓴다**

`0902/tests/test_teacher_gate.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sd.teacher import gate  # noqa: E402


class _FakeTeacher:
    """게이트가 무엇을 보는지 고정하기 위한 대역. 학습하지 않는다."""

    bottleneck = 1

    def __init__(self, path_fn, fill_fn):
        self._path_fn, self._fill_fn = path_fn, fill_fn

    def z(self, X):
        return X[:, :1]

    def predict_path(self, X):
        return self._path_fn(X)

    def predict_fill(self, X):
        return self._fill_fn(X)


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    # 역선택: 체결이 잘 되는 상태일수록 실제 결과가 나쁘다
    y_fill = (X[:, 0] > 0).astype(float)
    y_path = -1.0 * X[:, 0] + 0.2 * rng.normal(size=n)
    return X, y_path, y_fill, np.ones(n, dtype=bool)


def test_healthy_teacher_passes_all_checks():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert result.passed
    assert result.checks[gate.ADVERSE_SELECTION]["passed"]


def test_positive_outcome_in_high_fill_bucket_fails_the_gate():
    """체결 잘 되는 구간에서 결과가 양수면 큐 모델이나 라벨을 의심한다."""
    X, _y_path, y_fill, mask = _data()
    y_path = +1.0 * X[:, 0]                      # 부호를 뒤집어 오염을 흉내낸다
    teacher = _FakeTeacher(lambda x: x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.passed
    assert not result.checks[gate.ADVERSE_SELECTION]["passed"]
    assert result.checks[gate.ADVERSE_SELECTION]["top_decile_mean_y_path"] > 0


def test_constant_predictor_fails_beats_constant():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: np.zeros(len(x)), lambda x: np.full(len(x), 0.5))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.checks[gate.BEATS_CONSTANT]["passed"]


def test_miscalibrated_fill_head_is_caught():
    X, y_path, y_fill, mask = _data()
    # 실제 체결률과 무관하게 항상 0.99 를 말하는 헤드
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: np.full(len(x), 0.99))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert not result.checks[gate.FILL_CALIBRATION]["passed"]


def test_checks_report_numbers_not_just_booleans():
    X, y_path, y_fill, mask = _data()
    teacher = _FakeTeacher(lambda x: -1.0 * x[:, 0], lambda x: (x[:, 0] > 0).astype(float))
    result = gate.evaluate(teacher, X, y_path, y_fill, mask)
    assert "correlation" in result.checks[gate.BEATS_CONSTANT]
    assert "max_abs_gap" in result.checks[gate.FILL_CALIBRATION]
    assert len(result.checks[gate.FILL_CALIBRATION]["curve"]) == 10
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_teacher_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'gate' from 'sd.teacher'`

- [ ] **Step 3: `sd/teacher/gate.py` 구현**

```python
"""S1 교사 검증 게이트. 계획서 §3 S1.

> 오염된 교사에서 증류한 수식은 인샘플에서 훌륭하고 해석도 그럴듯하다.
> 이 게이트를 형식적으로 통과시키지 말 것.

세 가지를 본다. 계획서가 열거한 여섯 검사 중, **이 슬라이스에서 실제로 계산되는
것**만 넣었다. 대칭성·스케일 준불변·무시 변수 확인은 DeepLOB 교사가 들어올 때
추가한다 — 얕은 MLP 에 걸어 봐야 뜻이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ADVERSE_SELECTION = "adverse_selection_sign"
FILL_CALIBRATION = "fill_calibration"
BEATS_CONSTANT = "beats_constant"

MIN_CORRELATION = 0.05        # 상수 예측기보다 나은가
MAX_CALIBRATION_GAP = 0.25    # 신뢰도 곡선이 대각선에서 얼마나 벗어나도 되는가


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, dict]


def evaluate(teacher, X: np.ndarray, y_path: np.ndarray, y_fill: np.ndarray,
             mask: np.ndarray, deciles: int = 10) -> GateResult:
    X = np.asarray(X, dtype=float)
    usable = np.asarray(mask, dtype=bool) & np.all(np.isfinite(X), axis=1)
    usable &= np.isfinite(y_path) & np.isfinite(y_fill)
    if usable.sum() < deciles * 10:
        raise ValueError("게이트를 재기에 표본이 모자란다")

    Xs = X[usable]
    truth_path = np.asarray(y_path, dtype=float)[usable]
    truth_fill = np.asarray(y_fill, dtype=float)[usable]
    predicted_path = np.asarray(teacher.predict_path(Xs), dtype=float)
    predicted_fill = np.asarray(teacher.predict_fill(Xs), dtype=float)

    checks = {
        BEATS_CONSTANT: _beats_constant(predicted_path, truth_path),
        FILL_CALIBRATION: _calibration(predicted_fill, truth_fill, deciles),
        ADVERSE_SELECTION: _adverse_selection(predicted_fill, truth_path, deciles),
    }
    return GateResult(passed=all(c["passed"] for c in checks.values()), checks=checks)


def _beats_constant(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """상수를 말하는 교사는 증류할 것이 없다."""
    if np.std(prediction) == 0.0:
        return {"passed": False, "correlation": 0.0,
                "why": "교사가 상수를 예측한다. 증류할 함수가 없다"}
    correlation = float(np.corrcoef(prediction, truth)[0, 1])
    return {"passed": bool(correlation > MIN_CORRELATION),
            "correlation": correlation, "threshold": MIN_CORRELATION}


def _calibration(predicted: np.ndarray, truth: np.ndarray, deciles: int) -> dict:
    """예측 체결확률 십분위별 실제 체결률. 대각선 부근이어야 한다."""
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, deciles + 1))
    curve = []
    for i in range(deciles):
        lower, upper = edges[i], edges[i + 1]
        inside = (predicted >= lower) & (predicted <= upper if i == deciles - 1
                                         else predicted < upper)
        if inside.sum() == 0:
            curve.append({"bucket": i, "predicted": float("nan"), "actual": float("nan")})
            continue
        curve.append({"bucket": i,
                      "predicted": float(np.mean(predicted[inside])),
                      "actual": float(np.mean(truth[inside]))})
    gaps = [abs(p["predicted"] - p["actual"]) for p in curve
            if np.isfinite(p["predicted"]) and np.isfinite(p["actual"])]
    worst = float(max(gaps)) if gaps else float("inf")
    return {"passed": bool(worst <= MAX_CALIBRATION_GAP),
            "max_abs_gap": worst, "threshold": MAX_CALIBRATION_GAP, "curve": curve}


def _adverse_selection(predicted_fill: np.ndarray, truth_path: np.ndarray,
                       deciles: int) -> dict:
    """체결이 잘 되는 구간의 실제 결과는 **음수여야 정상**이다.

    지정가 매수가 잘 체결되는 순간은 대체로 파는 쪽이 급한 순간이다. 이 구간에서
    결과가 양수로 나오면 그것은 발견이 아니라 **큐 모델이 낙관적이거나 라벨에
    미래가 새고 있다는 신호**다. 정본 큐는 이미 보수적이므로 라벨을 먼저 의심한다.
    """
    cut = float(np.quantile(predicted_fill, 1.0 - 1.0 / deciles))
    top = predicted_fill >= cut
    if top.sum() == 0:
        return {"passed": False, "top_decile_mean_y_path": float("nan"),
                "why": "상위 십분위가 비었다"}
    mean = float(np.mean(truth_path[top]))
    return {"passed": bool(mean <= 0.0), "top_decile_mean_y_path": mean,
            "top_decile_size": int(top.sum()),
            "why": "양수면 큐 모델 낙관 또는 라벨 누수를 의심한다"}
```

`0902/sd/teacher/__init__.py` 를 덮어쓴다.

```python
from .base import Teacher
from .gate import GateResult, evaluate
from .shallow import ShallowMLP

__all__ = ["Teacher", "ShallowMLP", "GateResult", "evaluate"]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/dgu/tick/symbolic/0902 && python3 -m pytest tests/test_teacher_gate.py -v`
Expected: 5 passed

- [ ] **Step 5: `run_slice.py` 에 게이트를 끼운다**

`run_slice.py` 의 import 줄을 바꾼다.

```python
from sd.teacher.gate import evaluate as evaluate_gate
from sd.teacher.shallow import ShallowMLP
```

그리고 `print(f"[S1] 교사 학습 완료 (병목 {args.bottleneck})")` **바로 다음**에 넣는다.

```python
    gate_result = evaluate_gate(teacher, X[fit_rows], y_path[fit_rows],
                                y_fill[fit_rows], np.ones(len(fit_rows), dtype=bool))
    for name, check in gate_result.checks.items():
        print(f"[S1] {'통과' if check['passed'] else '실패'}  {name}: "
              + ", ".join(f"{k}={v}" for k, v in check.items()
                          if k not in {"passed", "curve", "why"}))
    if not gate_result.passed and not args.ignore_gate:
        raise SystemExit(
            "S1 교사 검증 게이트 불통과. 오염된 교사에서 증류하면 그 오염을 수식으로 "
            "고정할 뿐이다. 무시하려면 --ignore-gate 를 준다 (그 사실이 산출물에 남는다)")
```

`parse_args()` 에 인자 하나를 더한다.

```python
    parser.add_argument("--ignore-gate", action="store_true",
                        help="S1 게이트 불통과를 무시한다. provenance 에 기록된다")
```

`report.provenance(...)` 호출에 두 인자를 더한다.

```python
    prov = report.provenance(symbols=symbols, seed=args.seed,
                             sr_backend=NaiveBackend.name, grid=config.QUANTILE_GRID,
                             attempts=result.attempts, bottleneck=args.bottleneck,
                             gate_passed=gate_result.passed,
                             gate_ignored=bool(args.ignore_gate))
```

- [ ] **Step 6: `sd/report.py` 에 게이트 기록을 더한다**

`provenance` 의 시그니처와 반환에 두 항목을 넣는다.

```python
def provenance(symbols: Sequence[str], seed: int, sr_backend: str,
               grid: Sequence[float], attempts: int, bottleneck: int,
               gate_passed: bool = True, gate_ignored: bool = False) -> dict:
```

반환 dict 의 `"attempts"` 줄 다음에 넣는다.

```python
        "s1_gate_passed": bool(gate_passed),
        "s1_gate_ignored": bool(gate_ignored),
```

`_markdown` 의 `warning` 을 만드는 곳 다음에 한 줄을 더한다.

```python
    if prov.get("s1_gate_ignored"):
        warning += ("> ⚠️ **S1 교사 검증 게이트를 무시하고 돌린 run 이다** "
                    "(`--ignore-gate`). 교사 오염이 수식으로 고정됐을 수 있다.\n\n")
```

그리고 실행 정체성 표에 한 줄을 더한다.

```python
        f"| S1 게이트 | {'통과' if prov.get('s1_gate_passed') else '**불통과**'}"
        f"{' · **무시하고 진행**' if prov.get('s1_gate_ignored') else ''} |",
```

- [ ] **Step 7: 전체 테스트와 엔드투엔드 재실행**

Run: `cd /home/dgu/tick/symbolic/0902 && make test && make slice`
Expected: 모든 테스트 통과. 슬라이스가 `[S1] 통과/실패 ...` 세 줄을 찍고 끝까지 돈다

- [ ] **Step 8: 커밋**

```bash
cd /home/dgu/tick/symbolic/0902
git add sd/teacher/ sd/report.py run_slice.py tests/test_teacher_gate.py
git commit -m "feat: S1 교사 검증 게이트 — 역선택 부호와 체결 헤드 캘리브레이션

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JVLoKpNP3EW1ya1hXnfX9e"
```
