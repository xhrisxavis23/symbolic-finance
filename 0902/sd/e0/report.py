"""E0 run 산출물. 계획서 §5 필수 항목 — "Pareto front 전체(선택된 것만이
아니라)", "시드별 추출 수식 전체 목록" — 을 그대로 지킨다: SR 이 낸 후보
전원과 그 판정 근거 숫자를 JSON 으로 남긴다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import LAW_TITLES, VETO_LAWS
from .runner import LawResult, TrackResult


def _candidate_dict(candidate, judged: dict) -> dict:
    return {"expr": str(candidate.expr), "complexity": candidate.complexity,
            "in_sample_score": candidate.in_sample_score, "backend": candidate.backend,
            "seed": candidate.seed, "judged": judged}


def _track_dict(track: TrackResult) -> dict:
    return {
        "name": track.name, "n_rows": track.n_rows, "fit_seconds": track.fit_seconds,
        "n_candidates": len(track.candidates),
        "primary_index": track.primary_index,
        "primary_verdict": track.primary_verdict,
        "recovered": track.recovered,
        "any_candidate_recovered": track.any_candidate_recovered,
        "fraction_of_front_recovered": track.fraction_of_front_recovered,
        "diagnostics": {k: v for k, v in track.diagnostics.items()
                        if k not in {"discarded", "complexity_mismatches"}},
        "n_discarded": len(track.diagnostics.get("discarded", [])),
        "candidates": [_candidate_dict(c, j) for c, j in zip(track.candidates, track.judged)],
    }


def law_result_dict(result: LawResult) -> dict:
    return {
        "law": result.law, "title": LAW_TITLES.get(result.law, ""),
        "has_veto": result.law in VETO_LAWS,
        "n_rows_total": result.n_rows_total,
        "symbols_used": list(result.symbols_used),
        "n_symbols_used": len(result.symbols_used),
        "symbols_skipped": result.symbols_skipped,
        "teacher_dimless": result.teacher_dimless_diag,
        "teacher_raw": result.teacher_raw_diag,
        "tracks": {name: _track_dict(track) for name, track in result.tracks.items()},
        "recovered": result.recovered,
    }


def write(run_dir: Path, results: dict[str, LawResult], *, symbols, args: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {law: law_result_dict(result) for law, result in results.items()}
    (run_dir / "e0_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    # `_track_dict` 는 `e0_results.json` 에 넣을 `diagnostics` 에서 `discarded`·
    # `complexity_mismatches`(표가 커진다)를 뺀다 — 그래서 그 두 필드의 전체
    # 목록은 지금까지 어떤 산출물에도 안 남았다. `PySRBackend.fit()` 이 만드는
    # `diagnostics`(fit_seconds·complexity_mismatches·discarded·deterministic/
    # parallelism 등) 전체를 법칙×트랙별로 그대로 남긴다 — `provenance.json` 에
    # `s1_gate_checks` 를 남긴 것과 같은 방식. 순수 추가다: `e0_results.json`·
    # `gate.json`·`report.md` 의 생성 로직은 아래에서 손대지 않는다.
    diagnostics_payload = {
        law: {name: dict(track.diagnostics) for name, track in result.tracks.items()}
        for law, result in results.items()
    }
    (run_dir / "sr_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")

    n_recovered = sum(1 for r in results.values() if r.recovered)
    # 계획서 §4 게이트 규칙 — 두 조건은 별개 축이다. 카운트 규칙은 5개 중
    # {0,1,2}(중단) / {3,4,5}(진행) 로 완전히 나뉘어 "중간" 값이 없다 — 세
    # 번째 분기("borderline")는 도달 불가능한 죽은 코드였다(전부 5개 법칙을
    # 돌렸을 때 n_recovered<3 이면 항상 n_recovered<=2 이다). L1·L2 동시
    # 실패는 카운트와 무관한 **별도의 거부권**이라 카운트가 3 이상이어도
    # "경고"로 따로 표시해야 한다(계획서: "원인 규명 전 E1 보류").
    veto_failed = [law for law in VETO_LAWS if law in results and not results[law].recovered]
    veto_triggered = len(veto_failed) == len(VETO_LAWS) and len(VETO_LAWS) > 0
    if n_recovered <= 2:
        verdict = "stop"
    elif veto_triggered:
        verdict = "hold_veto"
    else:
        verdict = "proceed"
    gate = {
        "n_laws": len(results), "n_recovered": n_recovered,
        "recovered_laws": [law for law, r in results.items() if r.recovered],
        "failed_laws": [law for law, r in results.items() if not r.recovered],
        "veto_triggered": veto_triggered,
        "veto_laws_failed": veto_failed,
        "verdict": verdict,
        "symbols": list(symbols), "args": args,
    }
    (run_dir / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = ["# E0 run", "", f"복원: {n_recovered} / {len(results)}", ""]
    lines.append("| 법칙 | 복원(main) | 후보수(main) | ambiguous 있음 | fit_seconds(main) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for law, result in results.items():
        main = result.tracks.get("main")
        if main is None:
            lines.append(f"| {law} | - | - | - | - |")
            continue
        has_ambiguous = any(j.get("ambiguous") for j in main.judged)
        lines.append(f"| {law} | {main.recovered} | {len(main.candidates)} | "
                     f"{has_ambiguous} | {main.fit_seconds:.1f} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run_dir / "e0_results.json"
