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
