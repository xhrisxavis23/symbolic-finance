#!/usr/bin/env python3
"""G3 추가 점검 — **후보 수식 자체**에 대한 음성 대조군.

`g3_reference_and_controls.py` 의 음성 대조군은 **참조모델(단순회귀/구간
평균) 기계**만 검사했다 — 증류가 만든 실제 PySR 수식은 한 번도 permuted y
로 검사되지 않았다. 그런데 L4 의 참조모델은 R31 에서 **종목 내 중심화**로
재설계됐다(종목수준 생태학적 상관 때문에 5/5 허위양성을 냈던 전력이 있다) —
반면 PySR 로 증류된 L4 main 수식은 원본 특성(rv5/rv20/rv100_over_spread)을
그대로 쓰는 **풀링 수식**이라 종목 내 중심화가 없다. 그렇다면 L4(그리고
같은 이유로 다른 법칙)의 증류 수식 자체가 R31 이 잡았던 것과 같은
생태학적 상관을 타고 있을 가능성을 배제할 수 없다 — 참조 기계는 고쳤지만
후보 수식은 그 병을 갖고 있을 수 있다는 뜻이다.

**검사:** 참조 집합 R 의 `y` 를 (permute_y, within_symbol/global) 섞은 뒤,
**이미 고정된** main 후보 수식(재적합 없음)을 그 permuted y 에 대해
`score_candidate`(행단위 R², 원칙2)로 채점한다. 진짜 신호를 못 배웠다면
permuted y 에서는 R² 가 0 근처(또는 음수)여야 한다. 높은 양수가 나오면
그 수식이 종목 수준 구조(또는 다른 비인과적 상관)를 타고 있다는 뜻이고,
그러면 §4.2 의 "recovered"/"exceeds_ceiling" 판정 자체가 의심스러워진다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy

REPO_0902 = Path("/home/dgu/tick/symbolic/0902")
if str(REPO_0902) not in sys.path:
    sys.path.insert(0, str(REPO_0902))

from sd.sr.base import Candidate, score_candidate   # noqa: E402

REPO_0909 = Path(__file__).resolve().parent
OLD_OUT_DIR = REPO_0909 / "results" / "g2_full_run"
OUT_DIR = REPO_0909 / "results" / "g3_symmetric_run"

import g2_reference                                  # noqa: E402
from g3_reference_and_controls import _reference_arrays, _load_split, SEED, LAWS  # noqa: E402

N_PERMUTATIONS = 5
SCOPES = ("within_symbol", "global")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    split = _load_split()
    distilled = {law: json.loads((OLD_OUT_DIR / f"03_distill_{law}.json").read_text()) for law in LAWS}
    results = {}
    for law in LAWS:
        d = distilled[law]
        if "error" in d or d.get("main") is None:
            results[law] = {"skipped": "증류 실패 또는 main 트랙 없음"}
            continue
        arrs = _reference_arrays(law, split.reference)
        names, X_R, y_R, w_R, sid_R = arrs["names"], arrs["X"], arrs["y"], arrs["w"], arrs["sid"]
        expr = sympy.sympify(d["main"]["expr"])
        cand = Candidate(expr=expr, complexity=d["main"]["complexity"],
                         in_sample_score=0.0, backend="pysr", seed=SEED)
        real_score = float(score_candidate(cand, names, X_R, y_R, w_R))

        law_out = {"expr": d["main"]["expr"], "real_score_row": real_score, "permuted": {}}
        for scope in SCOPES:
            scores = []
            for perm_i in range(N_PERMUTATIONS):
                y_perm = g2_reference.permute_y(y_R, sid_R, scope=scope, seed=1000 * perm_i + 7)
                s = score_candidate(cand, names, X_R, y_perm, w_R)
                scores.append(float(s) if np.isfinite(s) else None)
            law_out["permuted"][scope] = scores
        results[law] = law_out
        finite_within = [s for s in law_out["permuted"]["within_symbol"] if s is not None]
        finite_global = [s for s in law_out["permuted"]["global"] if s is not None]
        _log(f"{law}: real={real_score:.4f} | permuted(within_symbol)={finite_within} "
             f"| permuted(global)={finite_global}")

    out_path = OUT_DIR / "05_candidate_negative_control.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    _log(f"저장: {out_path}")


if __name__ == "__main__":
    main()
