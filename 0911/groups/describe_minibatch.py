"""세 교사(전체 배치 200만 · 미니배치 200만 · 미니배치 3일) 서술 비교 — **사전등록 판정이 아니다.**

평가 ① 전체 변동의 48% 를 차지한 종목(261780)이 차이를 만드는지 본다: 변동 몫 상위 종목을 뺀 전체 R², 종목별 R² 중앙값,
그리고 상위 1종목을 뺀 짝지은 차이(3일 − 전체 배치 200만).
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_analysis as ga
import paired_analysis as paired

OUT = ga.ROOT / "out"
files = {"전체배치200만": OUT / "predictions.parquet", "미니배치200만": OUT / "mb_gate" / "predictions.parquet",
         "미니배치3일": OUT / "mb_full" / "predictions.parquet"}
res = {"note": "서술 — 사전등록 판정 아님"}
for es in ("eval1", "eval2"):
    base = pd.read_parquet(files["전체배치200만"]); base = base[base.eval_set == es].reset_index(drop=True)
    base["symbol"] = base.symbol.astype(str).str.zfill(6)
    cat = pd.Categorical(base.symbol); codes, syms = cat.codes.astype(np.int64), np.asarray(cat.categories); S = len(syms)
    y = base.y_path.to_numpy(float)
    contrib = np.bincount(codes, weights=(y - y.mean()) ** 2, minlength=S); order = np.argsort(-contrib)
    preds = {}
    for name, f in files.items():
        d = pd.read_parquet(f, columns=["symbol", "eval_set", "y_path", "pred_deeplob"]); d = d[d.eval_set == es].reset_index(drop=True)
        assert len(d) == len(base) and np.array_equal(d.y_path.to_numpy(float), y), f"{name} 행 불일치"
        preds[name] = d.pred_deeplob.to_numpy(float)
    r = {"top_sst_symbols": [syms[i] for i in order[:5]]}
    for name, pr in preds.items():
        st = ga.symbol_stats(codes, y, pr, S)
        sst_i = st[:, 2] - st[:, 1] ** 2 / np.maximum(st[:, 0], 1)
        r2_i = 1 - st[:, 3] / np.where(sst_i > 0, sst_i, np.nan)
        q = {"pooled": ga.r2_from_stats(st), "median_symbol_r2": float(np.nanmedian(r2_i)), "without_top": {}}
        for k in (1, 5, 10):
            keep = np.ones(S, bool); keep[order[:k]] = False
            q["without_top"][k] = ga.r2_from_stats(st[keep])
        r[name] = q
    drop = np.isin(codes, order[:1], invert=True)
    d1 = paired.paired_delta(syms[codes[drop]], y[drop], preds["전체배치200만"][drop], preds["미니배치3일"][drop]); d1.pop("verdict")
    r["delta_3day_minus_fullbatch_without_top1"] = d1
    res[es] = r
(OUT / "mb_full" / "describe_minibatch.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
for es in ("eval1", "eval2"):
    r = res[es]; print(f"=== {es} (변동 상위 {r['top_sst_symbols'][:3]})")
    for name in files:
        q = r[name]
        print(f"  {name:8s} 전체 {q['pooled']:.5f} | 상위 1·5·10 제외 {q['without_top'][1]:.5f} {q['without_top'][5]:.5f} {q['without_top'][10]:.5f} | 종목별 중앙값 {q['median_symbol_r2']:.5f}")
    d = r["delta_3day_minus_fullbatch_without_top1"]
    print(f"  상위 1종목 제외 짝지은 차이(3일 − 전체배치200만): {d['delta']:+.5f} [{d['delta_lo']:+.5f}, {d['delta_hi']:+.5f}]")
