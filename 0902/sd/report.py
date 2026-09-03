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

# 성과 지표가 이 열들에서 완전히 같으면 "지표가 동일한 계약"으로 묶는다.
# 같다는 것은 재생이 같은 사건 집합을 골랐다는 **증거**이지 증명은 아니다 —
# 우연히 같을 수도 있다. 그래서 결과를 "같은 진입식"이 아니라 "지표가 동일한
# 계약"이라고만 부른다 (F2 리뷰).
DUPLICATE_METRIC_COLUMNS: tuple[str, ...] = (
    "total_net_bps", "scorable", "fills", "unfilled", "censored",
    "cohort_NET_RECOVERY", "cohort_EARLY_RECOVERY_LATE_REVERSAL",
    "cohort_COST_INSUFFICIENT", "cohort_PERSISTENT_ADVERSE",
)


def provenance(symbols: Sequence[str], seed: int, sr_backend: str,
               grid: Sequence[float], attempts: int, bottleneck: int,
               gate_passed: bool = True, gate_ignored: bool = False) -> dict:
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
        "s1_gate_passed": bool(gate_passed),
        "s1_gate_ignored": bool(gate_ignored),
    }


def duplicate_signal_groups(ranked: pd.DataFrame) -> list[list[str]]:
    """`DUPLICATE_METRIC_COLUMNS` 이 완전히 같은 계약 ID 를 묶는다.

    빈 그룹은 없다 — 계약마다 정확히 하나의 그룹에 속한다. 그룹 수가 곧
    "지표 기준 실질 독립 계약 수"다 (F2 리뷰). 크기 1인 그룹은 중복이 없다는
    뜻이고, 크기 2 이상인 그룹이 중복이다.
    """
    if ranked.empty:
        return []
    columns = [c for c in DUPLICATE_METRIC_COLUMNS if c in ranked.columns]
    if not columns:
        return [[cid] for cid in ranked.index]
    groups: dict[tuple, list[str]] = {}
    for contract_id, row in ranked[columns].iterrows():
        key = tuple(row.tolist())
        groups.setdefault(key, []).append(str(contract_id))
    return [sorted(ids) for ids in groups.values()]


def write(run_dir: Path, universe: dict, candidates: list, compiled: list,
          failures: list, ranked: pd.DataFrame, prov: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    groups = duplicate_signal_groups(ranked)
    independent_contract_count = len(groups)
    duplicate_groups = [g for g in groups if len(g) > 1]

    prov_written = dict(prov)
    prov_written["independent_contract_count"] = independent_contract_count
    prov_written["independent_contract_count_of"] = len(ranked)
    prov_written["duplicate_metric_groups"] = duplicate_groups

    _json(run_dir / "universe.json", universe)
    _json(run_dir / "provenance.json", prov_written)
    _json(run_dir / "candidates.json", [
        {"expr": str(c.expr), "complexity": c.complexity,
         "in_sample_score": c.in_sample_score, "backend": c.backend, "seed": c.seed}
        for c in candidates])

    merged_duplicates = len(candidates) - len(compiled) - len(failures)
    _json(run_dir / "compile_report.json", {
        "attempted": len(candidates),
        "succeeded": len(compiled),
        "failed": len(failures),
        "merged_duplicates": merged_duplicates,
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
    path.write_text(_markdown(universe, candidates, compiled, failures, ranked, prov,
                              merged_duplicates, independent_contract_count,
                              duplicate_groups),
                    encoding="utf-8")
    return path


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _markdown(universe, candidates, compiled, failures, ranked, prov,
             merged_duplicates: int, independent_contract_count: int,
             duplicate_groups: list[list[str]]) -> str:
    warning = ""
    if prov["sr_backend"] == "naive":
        warning = (
            "> ⚠️ **이 run 의 SR 백엔드는 `naive` 다 — 배관 검증용 템플릿 격자이지 "
            "심볼릭 회귀가 아니다.**\n"
            "> 아래 성과 숫자는 **어떤 가설 판정에도 쓸 수 없다.** 이 run 이 검증하는 "
            "것은 배관이지 알파가 아니다 (DESIGN.md D3·D4).\n\n")
    if prov.get("s1_gate_ignored"):
        warning += ("> ⚠️ **S1 교사 검증 게이트를 무시하고 돌린 run 이다** "
                    "(`--ignore-gate`). 교사 오염이 수식으로 고정됐을 수 있다.\n\n")

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
        f"| 지표 기준 실질 독립 계약 수 | {independent_contract_count} / {len(ranked)} |",
        f"| S1 게이트 | {'통과' if prov.get('s1_gate_passed') else '**불통과**'}"
        f"{' · **무시하고 진행**' if prov.get('s1_gate_ignored') else ''} |",
        f"| 시드 | {prov['seed']} |", "",
        "## 컴파일", "",
        f"- 후보 {len(candidates)}개 중 **{len(compiled)}개 성공** "
        f"(성공률 {success_rate:.0%}), {len(failures)}개 탈락, **{merged_duplicates}개 병합**",
        f"  ({len(candidates)} = {len(compiled)} + {len(failures)} + {merged_duplicates}) "
        "— 병합은 실패가 아니다: 단조 껍질을 벗긴 정규형이 같은 후보끼리는 "
        "점수가 가장 높은 것만 남고 나머지는 `succeeded` 에도 `failed` 에도 들어가지 "
        "않는다 (계획서 §3 단조 흡수 · `compile/pipeline.py`).",
        "",
    ]
    if merged_duplicates < 0:
        lines.append(
            "> ⚠️ **`merged_duplicates` 가 음수다 — `attempted = succeeded + failed + "
            "merged_duplicates` 산술이 깨졌다.** 컴파일 파이프라인 버그 신호이니 "
            "이 리포트의 다른 숫자도 재검증해야 한다.\n")
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

        lines += ["", "### 지표 중복 계약", ""]
        lines.append(
            f"`{', '.join(DUPLICATE_METRIC_COLUMNS)}` 이 완전히 같은 계약을 묶으면 "
            f"{len(ranked)}개 계약이 **{independent_contract_count}개 그룹**으로 나뉜다. "
            "지표가 같다는 것은 재생이 같은 사건 집합을 골랐다는 **증거**이지 증명은 "
            "아니다 — \"지표가 동일한 계약\"이지 \"같은 진입식\"이 아니다.")
        if duplicate_groups:
            lines.append("")
            for group in duplicate_groups:
                lines.append(f"- {', '.join(group)}")
        else:
            lines.append("- 지표가 동일한 계약 쌍 없음 — 모든 계약이 서로 다른 지표를 낸다.")
    lines.append("")
    return "\n".join(lines)
