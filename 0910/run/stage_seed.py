#!/usr/bin/env python3
"""Stage A — 시드별 교사 학습 + SR 적합 + 컴파일. PREREG.md 잠금 파라미터를 쓴다.

    python -m run.stage_seed --seed 0

`0910/run/state/seed_{N}.json` 이 이미 있으면 스킵한다(재개 가능 — 세션이
끊겨도 이미 끝난 시드는 다시 돌리지 않는다). 끝나면 그 파일을 커밋한다.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import common, locked_params as P
from .common import STATE_DIR, log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=list(P.SEEDS))
    args = parser.parse_args()
    seed = args.seed

    out_path = STATE_DIR / f"seed_{seed}.json"
    if out_path.exists():
        log("seed", f"seed={seed} 이미 완료 — 스킵(재개)")
        return 0

    t0 = time.time()
    log("seed", f"seed={seed} 시작")

    strata_df = common.stratified_universe()
    groups = common.friction_group_symbols(strata_df)
    L, M = groups["L"], groups["M"]
    log("seed", f"L={len(L)} M={len(M)} H={len(groups['H'])}")

    X_L, y_path_L, y_fill_L, mask_L, feat_L, sess_L = common.load_windowed(
        L, P.L_TRAIN_WINDOW)
    X_M, y_path_M, y_fill_M, mask_M, feat_M, sess_M = common.load_windowed(
        M, P.M_SELECT_WINDOW)

    common_features = tuple(sorted(set(feat_L) & set(feat_M)))
    if common_features != feat_L or common_features != feat_M:
        log("seed", f"경고: L·M feature 교집합({len(common_features)})이 "
                    f"L({len(feat_L)})·M({len(feat_M)})과 다르다 — 교집합으로 좁힌다")
        idx_L = [feat_L.index(n) for n in common_features]
        idx_M = [feat_M.index(n) for n in common_features]
        X_L = X_L[:, idx_L]
        X_M = X_M[:, idx_M]
    feature_names = common_features

    from sd.manifold import select as manifold_select
    from sd.teacher.window import make_causal_windows

    sel_L = manifold_select(X_L, mask_L, max_samples=P.MAX_SAMPLES, seed=seed)
    fit_rows = sel_L.index[sel_L.on_manifold]
    fit_weight = sel_L.weight[sel_L.on_manifold]
    log("seed", f"L on-manifold fit rows={len(fit_rows):,} (usable={mask_L.sum():,})")

    sel_M = manifold_select(X_M, mask_M, max_samples=P.MAX_SAMPLES, seed=seed)
    select_rows = sel_M.index[sel_M.on_manifold]
    select_weight = sel_M.weight[sel_M.on_manifold]
    log("seed", f"M on-manifold select rows={len(select_rows):,} (usable={mask_M.sum():,})")

    Xw_L = make_causal_windows(X_L, window=P.TEACHER_WINDOW, session_ids=sess_L)
    Xw_M = make_causal_windows(X_M, window=P.TEACHER_WINDOW, session_ids=sess_M)
    log("seed", f"윈도우 생성 완료 L={Xw_L.shape} M={Xw_M.shape}")

    from sd.teacher.deeplob import DeepLOBCompact

    t_teacher0 = time.time()
    teacher = DeepLOBCompact(
        n_features=len(feature_names), bottleneck=P.BOTTLENECK, seed=seed,
        window=P.TEACHER_WINDOW).fit(
        Xw_L[fit_rows], y_path_L[fit_rows], y_fill_L[fit_rows], fit_weight,
        epochs=P.TEACHER_EPOCHS,
        X_path_select=Xw_M[select_rows], y_path_select=y_path_M[select_rows],
        weight_select=select_weight,
        eval_every=P.TEACHER_EVAL_EVERY, patience=P.TEACHER_PATIENCE)
    t_teacher1 = time.time()
    early_stop = getattr(teacher, "early_stop_history_", None)
    log("seed", f"교사 학습 완료 {t_teacher1-t_teacher0:.1f}s "
                f"stopped_epoch={getattr(early_stop, 'stopped_epoch', None)} "
                f"best_epoch={getattr(early_stop, 'best_epoch', None)}")

    from sd.teacher.gate import evaluate as evaluate_gate

    gate_result = evaluate_gate(
        teacher, Xw_M[select_rows], y_path_M[select_rows], y_fill_M[select_rows],
        np.ones(len(select_rows), dtype=bool))
    for name, check in gate_result.checks.items():
        log("gate", f"{'통과' if check['passed'] else '실패'} {name}: "
                    + ", ".join(f"{k}={v}" for k, v in check.items()
                                if k not in {"passed", "curve"}))
    log("gate", f"게이트 전체 판정: {'통과' if gate_result.passed else '실패'}")

    target = teacher.predict_path(Xw_L[fit_rows])

    from sd.sr.pysr_backend import PySRBackend
    from sd.compile import compile_candidates

    backend = PySRBackend(seed=seed, deterministic=P.SR_DETERMINISTIC,
                          niterations=P.SR_NITERATIONS, maxsize=P.SR_MAXSIZE)
    t_sr0 = time.time()
    candidates = backend.fit(X_L[fit_rows], target, fit_weight, feature_names)
    t_sr1 = time.time()
    log("seed", f"SR 적합 완료 {t_sr1-t_sr0:.1f}s 후보={len(candidates)}")

    compiled, failures = compile_candidates(candidates)
    rate = (len(compiled) / len(candidates)) if candidates else 0.0
    log("seed", f"컴파일 성공 {len(compiled)}/{len(candidates)} ({rate:.0%}) "
                f"탈락 {len(failures)}")

    result = {
        "schema": "e1_stage_seed.v1",
        "seed": seed,
        "n_symbols_L": len(L), "n_symbols_M": len(M),
        "n_fit_rows": int(len(fit_rows)), "n_select_rows": int(len(select_rows)),
        "n_usable_L": int(mask_L.sum()), "n_usable_M": int(mask_M.sum()),
        "feature_names": list(feature_names),
        "teacher_fit_seconds": t_teacher1 - t_teacher0,
        "teacher_early_stop": {
            "enabled": getattr(early_stop, "early_stopping_enabled", None),
            "stopped_epoch": getattr(early_stop, "stopped_epoch", None),
            "best_epoch": getattr(early_stop, "best_epoch", None),
            "best_select_r2": getattr(early_stop, "best_score", None),
            "triggered": getattr(early_stop, "triggered", None),
        },
        "gate_passed": bool(gate_result.passed),
        "gate_checks": {
            name: {k: v for k, v in check.items() if k != "curve"}
            for name, check in gate_result.checks.items()},
        "sr_fit_seconds": t_sr1 - t_sr0,
        "sr_diagnostics": {k: v for k, v in backend.diagnostics.items()
                          if k not in ("discarded", "complexity_mismatches")},
        "n_candidates": len(candidates),
        "n_compiled": len(compiled),
        "n_compile_failed": len(failures),
        "compile_failure_reasons": [
            {"stage": f.stage, "reason": f.reason[:300]} for f in failures],
        "compiled_entries": [
            {
                "ast": e.ast, "thresholds": e.thresholds,
                "normal_form": e.normal_form, "shells": list(e.shells),
                "complexity": e.source.complexity,
                "in_sample_score": e.source.in_sample_score,
                "expr_str": str(e.source.expr),
            }
            for e in compiled
        ],
        "total_elapsed_seconds": time.time() - t0,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    log("seed", f"seed={seed} 완료 {time.time()-t0:.1f}s -> {out_path}")

    common.git_commit(
        [f"run/state/seed_{seed}.json"],
        f"E1 실행: 시드 {seed} 교사+SR+컴파일 완료 "
        f"(게이트={'통과' if gate_result.passed else '실패'}, "
        f"컴파일 {len(compiled)}/{len(candidates)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
