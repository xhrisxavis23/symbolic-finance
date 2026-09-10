#!/usr/bin/env python3
"""G3 — 1단계(대칭 천장·정직한 지표) + 2단계(음성 대조군 재실행).
`PREREG-G3.md` §2 원칙1·2, §2 원칙3의 순서(수정 → 음성대조군 → 합성검산 →
그다음 L1)를 그대로 따른다. **이 스크립트는 L1 이 통과하는지 보지 않는다**
— 참조 천장(밴드)과 음성 대조군만 계산·저장한다. 후보 재채점(L1 대조군
포함)은 `g3_rescore_and_verdict.py`(별도 실행, 이 스크립트와 합성 검산이
끝난 뒤에만 돈다)로 분리했다.

기존 `results/g2_full_run/00_split.json` 을 그대로 재사용한다 — 참조/증류
분할은 시드로 결정되므로 다시 계산해도 같은 값이 나오지만, 이미 쓰인
증류 결과(`03_distill_L*.json`)와 정확히 같은 분할이라는 것을 파일로
보증하기 위해 재계산 대신 읽는다.

산출물: `results/g3_symmetric_run/01_reference_ceilings.json`,
`results/g3_symmetric_run/02_negative_control.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd import config, manifold          # noqa: E402
from sd.e0 import targets                 # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
OLD_OUT_DIR = REPO_0909 / "results" / "g2_full_run"
OUT_DIR = REPO_0909 / "results" / "g3_symmetric_run"

import g2_reference   # noqa: E402

DATE = config.DATE
assert DATE == "20260316", f"허용되지 않은 날짜: {DATE}"

LAWS = ("L1", "L2", "L3", "L4", "L5")
VETO_LAW = "L1"
MAX_MANIFOLD_SAMPLES_REFERENCE = 20000   # G2 와 동일값 — 이번 수정은 R 표집 방식이 아니라
                                          # "R 을 언제 어떻게 훈련에 쓰는가"를 고친다.
SEED = config.SEED
N_PERMUTATIONS = 5
SCOPES = ("within_symbol", "global")
N_BOOT_CEILING = 1000
N_BOOT_NEGCONTROL = 300


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _load_split() -> SimpleNamespace:
    d = json.loads((OLD_OUT_DIR / "00_split.json").read_text())
    return SimpleNamespace(reference=tuple(d["reference"]),
                           distill_fit=tuple(d["distill_fit"]),
                           distill_select=tuple(d["distill_select"]))


def _reference_arrays(law: str, ref_symbols: tuple) -> dict:
    ref_ds = targets.assemble(law, ref_symbols, DATE)
    selection = manifold.select(ref_ds.X_dimless, ref_ds.mask,
                                max_samples=MAX_MANIFOLD_SAMPLES_REFERENCE, seed=SEED)
    on_rows = selection.index[selection.on_manifold]
    return {"names": ref_ds.names_dimless, "X": ref_ds.X_dimless[on_rows],
            "y": ref_ds.y_dimless[on_rows], "sid": ref_ds.symbol_ids[on_rows],
            "w": selection.weight[selection.on_manifold],
            "n_symbols_used": len(ref_ds.symbols_used), "n_rows_total": ref_ds.n_rows_total}


def _distill_pool_arrays(law: str, fit_symbols: tuple, select_symbols: tuple) -> dict:
    """`PREREG-G2.md` §3.5 "단순회귀 기준선"과 정확히 같은 정의 —
    D_fit∪D_select 전체(마스크만 적용, on-manifold 선별 없음)를 참조모델
    적합에 쓴다. `g2_full_run.phase4_rescore` 의 `X_D/y_D/mask_D/sid_D`
    조립과 동일하다 — 이번 수정으로 천장이 기준선과 **같은 D** 를 보게
    됐으므로(원칙1), 둘이 다른 D 조립 규칙을 쓰면 그 자체가 새 비대칭이
    된다."""
    fit_ds = targets.assemble(law, fit_symbols, DATE)
    select_ds = targets.assemble(law, select_symbols, DATE)
    X_D = np.concatenate([fit_ds.X_dimless, select_ds.X_dimless], axis=0)
    y_D = np.concatenate([fit_ds.y_dimless, select_ds.y_dimless], axis=0)
    mask_D = np.concatenate([fit_ds.mask, select_ds.mask], axis=0)
    sid_D = np.concatenate([fit_ds.symbol_ids, select_ds.symbol_ids], axis=0)
    return {"X": X_D, "y": y_D, "mask": mask_D, "sid": sid_D}


def phase1_reference_ceilings(split) -> dict:
    _log("=== G3 단계 1: 대칭 참조 천장 (D 에서 적합 → R 에서 채점) ===")
    ceilings = {}
    ref_arrays = {}
    d_arrays = {}
    for law in LAWS:
        t0 = time.time()
        arrs = _reference_arrays(law, split.reference)
        darrs = _distill_pool_arrays(law, split.distill_fit, split.distill_select)
        ref_arrays[law] = arrs
        d_arrays[law] = darrs
        ceiling = g2_reference.ceiling_fit_on_D_score_on_R(
            law, darrs["X"], darrs["y"], darrs["mask"], darrs["sid"],
            arrs["X"], arrs["y"], arrs["w"], arrs["sid"],
            n_boot=N_BOOT_CEILING, seed=SEED)
        eligible = bool(ceiling.get("ok") and ceiling.get("row_r2_ci_low", -1.0) > 0.0)
        ceilings[law] = {"ceiling": ceiling, "eligible": eligible,
                         "n_symbols_reference": arrs["n_symbols_used"],
                         "n_rows_reference_total": arrs["n_rows_total"],
                         "n_on_manifold_reference": int(len(arrs["y"])),
                         "n_rows_D_pool": int(len(darrs["y"]))}
        _log(f"  {law}: row_r2 point={ceiling.get('row_r2_point')} "
             f"CI=[{ceiling.get('row_r2_ci_low')},{ceiling.get('row_r2_ci_high')}] "
             f"binned_r2 point={ceiling.get('binned_r2_point')} "
             f"amplification={ceiling.get('amplification_factor')} "
             f"eligible={eligible} ({time.time()-t0:.1f}s)")
    _write_json(OUT_DIR / "01_reference_ceilings.json", ceilings)
    _log("대칭 천장(밴드) 확정 — 이 시점 이후 손대지 않는다.")
    return {"ceilings": ceilings, "ref_arrays": ref_arrays, "d_arrays": d_arrays}


def _negative_control_for_law(law: str, arrs: dict, darrs: dict) -> dict:
    out = {}
    for scope in SCOPES:
        results = []
        for perm_i in range(N_PERMUTATIONS):
            seed = 1000 * perm_i + 7
            ceiling = g2_reference.negative_control_fit_on_D_score_on_R(
                law, darrs["X"], darrs["y"], darrs["mask"], darrs["sid"],
                arrs["X"], arrs["y"], arrs["w"], arrs["sid"],
                scope=scope, seed=seed, n_boot=N_BOOT_NEGCONTROL)
            eligible = bool(ceiling.get("ok") and ceiling.get("row_r2_ci_low", -1.0) > 0.0)
            results.append({"perm_index": perm_i, "eligible": eligible,
                            "row_r2_point": ceiling.get("row_r2_point"),
                            "row_r2_ci_low": ceiling.get("row_r2_ci_low"),
                            "row_r2_ci_high": ceiling.get("row_r2_ci_high"),
                            "reason": ceiling.get("reason")})
        out[scope] = results
    return out


def phase2_negative_control(phase1: dict) -> dict:
    _log("=== G3 단계 2: 음성 대조군 (D·R 양쪽에서 y 섞기) ===")
    ref_arrays, d_arrays = phase1["ref_arrays"], phase1["d_arrays"]
    neg_controls = {}
    for law in LAWS:
        t0 = time.time()
        neg_controls[law] = _negative_control_for_law(law, ref_arrays[law], d_arrays[law])
        n_within = sum(r["eligible"] for r in neg_controls[law]["within_symbol"])
        n_global = sum(r["eligible"] for r in neg_controls[law]["global"])
        _log(f"  {law} 음성대조군: within_symbol {n_within}/{N_PERMUTATIONS}, "
             f"global {n_global}/{N_PERMUTATIONS} ({time.time()-t0:.1f}s)")

    l1_within_clean = sum(r["eligible"] for r in neg_controls[VETO_LAW]["within_symbol"]) == 0
    untrustworthy = [law for law in LAWS
                     if sum(r["eligible"] for r in neg_controls[law]["within_symbol"]) > 0]
    result = {"negative_controls": neg_controls, "l1_within_symbol_clean": l1_within_clean,
             "untrustworthy_laws_within_symbol": untrustworthy}
    _write_json(OUT_DIR / "02_negative_control.json", result)
    if l1_within_clean:
        _log("L1 음성 대조군 정상(5/5 자격없음).")
    else:
        _log("!!! L1 음성 대조군에서 허위 양성 — 선결 조건 위반. 측정 코드부터 재검토해야 한다.")
    if untrustworthy:
        _log(f"주의: {untrustworthy} 는 음성 대조군에서 허위 양성 — 신뢰 불가로 표시할 것.")
    return result


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = _load_split()
    _log(f"참조 {len(split.reference)} / 증류-적합 {len(split.distill_fit)} / "
         f"증류-선택 {len(split.distill_select)} (00_split.json 재사용)")
    phase1 = phase1_reference_ceilings(split)
    phase2 = phase2_negative_control(phase1)
    total_s = time.time() - t_start
    _write_json(OUT_DIR / "DONE_controls.json", {
        "total_seconds": total_s,
        "l1_within_symbol_clean": phase2["l1_within_symbol_clean"],
        "untrustworthy_laws_within_symbol": phase2["untrustworthy_laws_within_symbol"]})
    _log(f"=== 1·2단계 완료. 총 {total_s:.1f}s. "
         f"L1 음성대조군 정상={phase2['l1_within_symbol_clean']} ===")
    _log(">>> 다음 단계는 합성 검산(g3_synthetic_check.py)이다 — "
         "아직 어떤 후보(L1 포함)도 재채점하지 않았다.")


if __name__ == "__main__":
    main()
