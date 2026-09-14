#!/usr/bin/env python3
"""실행 전 점검 C3(주입 경로 동등성) · C4(신호 독립 대조). PREREG-TEACHER-ECON.md §3.4.

C4: 평가 종목 앞 5개에서 `contract.entry_signal` 이 낸 신호가 점수 배열로 직접 계산한 신호와 같은가.
    단일 경로(`_rolling_quantile_bindings`)와 공유 경로(`rolling_quantile_binding_grid`) 둘 다 본다.
C3: 평가 종목 앞 20개에서 점수를 book_imbalance 로 채운 가짜 교사 계약의 정본 원장이 나이브 계약 원장과
    행 단위로 같은가 — 주입 경로 전체(워커 fork · 적재 감싸기 · 공유 분위 · 원장)가 원래 경로와 같은 답을 내는지.

성과 수치는 보지 않는다. 신호 수·원장 행 수·일치 여부만 남긴다.

    python -m stage2.checks
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import common as C
from . import inject

from framework import canonical  # noqa: E402
from framework import contract as fcontract  # noqa: E402
from framework import data as fdata  # noqa: E402
from sd import config  # noqa: E402
from sd.compile import contract_template  # noqa: E402


def c4(symbols: list[str]) -> dict:
    rows, ok = [], True
    for sym in symbols:
        arrays, _path = fdata.load(sym, C.DATE, config.TICK_ROOT)    # 재생 워커와 같은 호출(기본 캐시)
        for m in C.MODELS:
            score = np.load(C.score_path(m, sym))["score"]
            sig = np.asarray(fcontract.entry_signal(arrays, C.abs_template(m)), dtype=bool)
            ind = np.isfinite(score) & (score > 0.0)
            same = bool(np.array_equal(sig, ind))
            ok &= same
            rows.append({"symbol": sym, "model": m, "rule": "ABS", "path": "single", "same": same,
                         "n_signal": int(sig.sum()), "n_ticks": int(len(sig))})
            tpl = C.q_template(C.MARKERS[m])
            thresholds, cache = fcontract.rolling_quantile_binding_grid(
                arrays, tpl["entry_program"], [{C.THETA: q} for q in C.QS])
            for i, q in enumerate(C.QS):
                cut = pd.Series(score).shift(1).rolling(100, min_periods=100).quantile(
                    q, interpolation="higher").to_numpy()
                ind = np.isfinite(score) & np.isfinite(cut) & (score > cut)
                single = np.asarray(fcontract.entry_signal(arrays, tpl, rolling_quantiles={C.THETA: q}), dtype=bool)
                shared = np.asarray(fcontract.entry_signal(
                    arrays, tpl, rolling_quantiles={C.THETA: q}, rolling_thresholds=thresholds[i],
                    precomputed_expressions=cache), dtype=bool)
                for path, s in (("single", single), ("shared", shared)):
                    same = bool(np.array_equal(s, ind))
                    ok &= same
                    rows.append({"symbol": sym, "model": m, "rule": f"Q{int(round(q * 100))}", "path": path,
                                 "same": same, "n_signal": int(s.sum()), "n_ticks": int(len(s))})
    return {"pass": bool(ok), "rows": rows}


def c3(symbols: list[str]) -> dict:
    select = json.loads(C.SELECT_0910.read_text())
    contracts, members, table = {}, {}, {}
    for q in C.QS:
        for label, tpl in (("FAKE", C.q_template(C.MARKERS["FAKE"])),
                           ("naive", contract_template(select["naive_primary"]["ast"], C.PARAM))):
            cid = f"{label}:q{q:.2f}"
            contracts[cid], members[cid] = tpl, list(symbols)
            table.update({f"{cid}:{s}:{C.DATE}": {C.THETA: q} for s in symbols})
    out = C.OUT / "c3_ledger.parquet"
    manifest = canonical.run_backtest(contracts=contracts, members=members, dates=[C.DATE], output=out,
                                      parameter_table=table, workers=8, root=config.TICK_ROOT)
    errors = manifest.get("errors") or []
    led = pd.read_parquet(out)
    per_q, ok = {}, not errors
    for q in C.QS:
        a = led[led.contract_id == f"FAKE:q{q:.2f}"].drop(columns=["contract_id"]).reset_index(drop=True)
        b = led[led.contract_id == f"naive:q{q:.2f}"].drop(columns=["contract_id"]).reset_index(drop=True)
        same = bool(len(a) == len(b) and len(a) > 0 and a.equals(b))
        rec = {"rows_fake": int(len(a)), "rows_naive": int(len(b)), "identical": same}
        if not same and len(a) == len(b):
            rec["differing_columns"] = [c for c in a.columns if not a[c].equals(b[c])]
        per_q[f"{q:.2f}"] = rec
        ok &= same
    missing = manifest.get("missing") or []
    return {"pass": bool(ok), "per_q": per_q, "errors": [str(e) for e in errors][:20],
            "n_missing": len(missing), "missing_sample": [str(x) for x in missing[:5]]}


def main() -> None:
    t0 = time.time()
    scores = json.loads((C.OUT / "scores_manifest.json").read_text())
    if not scores.get("c1_pass"):
        raise SystemExit("C1 을 통과하지 않았다 — C3·C4 를 돌리지 않는다")
    inject.install()
    symbols = C.eval_symbols()
    r4 = c4(symbols[:C.C4_SYMBOLS])
    C.log("checks", f"C4 {'통과' if r4['pass'] else '실패'} ({sum(r['same'] for r in r4['rows'])}/{len(r4['rows'])})")
    r3 = c3(symbols[:C.C3_SYMBOLS])
    C.log("checks", f"C3 {'통과' if r3['pass'] else '실패'} {r3['per_q']} errors={len(r3['errors'])}")
    result = {"schema": "0914_stage2_checks.v1", "c3_pass": r3["pass"], "c4_pass": r4["pass"],
              "c3": r3, "c4": r4, "seconds": time.time() - t0}
    (C.OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    C.git_commit(["stage2/out/checks.json"],
                 f"run(0914): 실행 전 점검 C3 {'통과' if r3['pass'] else '실패'} · C4 {'통과' if r4['pass'] else '실패'}")
    if not (r3["pass"] and r4["pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
