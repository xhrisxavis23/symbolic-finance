#!/usr/bin/env python3
"""state/ 의 JSON 산출물을 사람이 읽는 `0910/RESULTS.md` 로 정리한다.

여러 번 다시 실행해도 안전하다(항상 최신 state/ 로 덮어쓴다) — 이 파일
자체는 "완료 마커"로 쓰지 않는다.
"""
from __future__ import annotations

import json

from . import locked_params as P
from .common import STATE_DIR, git_commit, log

OUT = "/home/dgu/tick/symbolic/0910/RESULTS.md"


def _fmt_row(name: str, row: dict | None) -> str:
    if row is None:
        return f"| {name} | — | — | — |"
    return (f"| {name} | {row['total_net_bps']:.2f} | "
           f"{row['bps_per_decision']:.4f} | {row['scorable']} |")


def main() -> int:
    select_path = STATE_DIR / "select.json"
    curve_path = STATE_DIR / "curve.json"
    if not select_path.exists() or not curve_path.exists():
        log("results", "select.json 또는 curve.json 이 아직 없다 — 아직 못 쓴다")
        return 0

    select_data = json.loads(select_path.read_text())
    curve_data = json.loads(curve_path.read_text())

    seed_diag = {}
    for seed in P.SEEDS:
        p = STATE_DIR / f"seed_{seed}.json"
        if p.exists():
            seed_diag[seed] = json.loads(p.read_text())

    lines = []
    lines.append("# RESULTS — E1 본 실험 실행 결과\n")
    lines.append(f"`PREREG.md` 에 잠근 파라미터로 실행했다. 시드 {list(P.SEEDS)}, "
                 f"분위격자 {list(P.QUANTILE_GRID)}, niterations={P.SR_NITERATIONS}, "
                 f"maxsize={P.SR_MAXSIZE}, max_samples={P.MAX_SAMPLES:,}.\n")

    lines.append("## 최종 판정\n")
    lines.append(f"**{curve_data['final_claim']}**\n")
    lines.append("| 시드 | 게이트 통과 | H total_net_bps>0 | H scorable≥200 | "
                 "H 에서 영점(h) 대비 우위 | 종합 판정 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label, v in curve_data["seed_verdicts"].items():
        lines.append(f"| {label} | {v.get('gate_passed')} | "
                     f"{v.get('H_total_net_bps_positive')} | "
                     f"{v.get('H_scorable_sufficient')} | "
                     f"{v.get('beats_null_on_H')} | "
                     f"{'통과' if v.get('pass') else '실패'} |")
    lines.append("")

    lines.append("## L·M·H 저하 곡선 (R37-2)\n")
    lines.append("L 은 학습에 쓴 층이라 성과 근거가 아니라 저하 곡선의 "
                 "기준점이다. M 은 선택에, H 는 최종 평가에 썼다(봉인 층).\n")
    for label, curve in curve_data["curve_L_M_H"].items():
        lines.append(f"### {label}\n")
        lines.append("| 층 | total_net_bps | bps_per_decision | scorable |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(_fmt_row("L(학습, 기준점)", curve.get("L")))
        lines.append(_fmt_row("M(선택)", curve.get("M")))
        lines.append(_fmt_row("H(최종, 봉인)", curve.get("H")))
        lines.append("")

    if curve_data.get("H_intraday_after_1300"):
        lines.append("## H2② 일중 외삽 — H, 13:00 이후만\n")
        lines.append("| label | total_net_bps | bps_per_decision | scorable |")
        lines.append("| --- | --- | --- | --- |")
        for r in curve_data["H_intraday_after_1300"]:
            lines.append(f"| {r['label']} | {r['total_net_bps']:.2f} | "
                         f"{r['bps_per_decision']:.4f} | {r['scorable']} |")
        lines.append("")

    lines.append("## 구조 복원율\n")
    sr = select_data["structural_recovery"]
    lines.append(f"전체 SR 후보(전 시드 합) {sr['n_total_candidates']}개 중 "
                 f"서로 다른 구조(normal_form) {sr['n_unique_structures']}개.\n")
    lines.append("가장 흔한 구조 (구조, 등장 횟수):\n")
    for structure, count in sr["top_structures"]:
        lines.append(f"- `{structure}` — {count}회")
    lines.append("")

    lines.append("## 시드별 진단\n")
    for seed, data in seed_diag.items():
        lines.append(f"### seed={seed}\n")
        lines.append(f"- 게이트 통과: {data['gate_passed']}")
        lines.append(f"- 게이트 상세: {json.dumps(data['gate_checks'], ensure_ascii=False)}")
        lines.append(f"- 교사 학습 {data['teacher_fit_seconds']:.0f}초, "
                     f"조기종료 stopped_epoch={data['teacher_early_stop']['stopped_epoch']} "
                     f"best_epoch={data['teacher_early_stop']['best_epoch']}")
        lines.append(f"- SR 적합 {data['sr_fit_seconds']:.0f}초, "
                     f"후보 {data['n_candidates']}개, 컴파일 성공 "
                     f"{data['n_compiled']}/{data['n_candidates']}")
        primary = select_data["primaries"].get(str(seed), {})
        if primary.get("eligible"):
            lines.append(f"- 1순위 후보: `{primary['expr_str']}` "
                         f"(임계 분위 {primary['contract_id'].split(':q')[1]})")
        else:
            lines.append("- 1순위 후보: 자격 미달(scorable<200 이거나 후보 없음)")
        lines.append("")

    naive_p = select_data.get("naive_primary", {})
    null_p = select_data.get("null_primary", {})
    lines.append("## 비교군\n")
    lines.append(f"- (f) 나이브(`{P.NAIVE_FEATURE}`): "
                 f"{'`'+naive_p['expr_str']+'`' if naive_p.get('eligible') else '자격 미달'}")
    lines.append(f"- (h) 영점 근사(`{P.NULL_FEATURE}`): "
                 f"{'`'+null_p['expr_str']+'`' if null_p.get('eligible') else '자격 미달'}")
    lines.append("- (a)(c)(g) 및 ablation 트랙·병목 ablation은 이번 실행 범위 밖"
                 "(`run/locked_params.py::NOT_IN_SCOPE` 참조)\n")

    n_eligible = sum(1 for p in select_data["primaries"].values() if p.get("eligible"))
    lines.append("## 시도 횟수 N\n")
    lines.append(f"- SR 적합: 시드 {len(P.SEEDS)} × 트랙 1(main) = {len(P.SEEDS)}회")
    lines.append(f"- M 선택 재생 attempts: {select_data['attempts']} "
                 f"(전 시드 컴파일 후보 합 × 분위격자 {len(P.QUANTILE_GRID)} "
                 f"+ 비교군 2 × 분위격자)")
    lines.append(f"- M 선택에서 자격을 얻은 시드: {n_eligible}/{len(P.SEEDS)}")
    lines.append(f"- L/M/H 곡선 재생 attempts: 동결 후보 수 × 분위격자 × 2개 층(L,H)\n")

    lines.append("## 봉인 홀드아웃(`20260317`)\n")
    lines.append("**이 실행은 봉인 홀드아웃을 열지 않았다.** DESIGN.md §5.1 의 "
                 "조건(H 통과 + 3시드 전부 통과)을 만족하는지는 위 표로 판단한다. "
                 "만족하면 별도 단계로 연다 — 이 파일이 그 결정을 자동으로 "
                 "내리지 않는다.\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("results", f"RESULTS.md 작성 완료 -> {OUT}")

    git_commit(["RESULTS.md"], f"E1 실행: RESULTS.md 갱신 — {curve_data['final_claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
