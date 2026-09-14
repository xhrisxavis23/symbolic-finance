"""평가 결과 서술 — **사전등록 판정이 아니다.** 1차 판정(group_results.json)을 해석하기 위한 보조 수치.

1. 전체 R² 가 어느 종목에 좌우되는가: 종목별로 전체 변동(전체 평균 기준 제곱합)에서 차지하는 몫,
   변동 몫 상위 종목을 빼면 전체 R² 가 어떻게 되는가, 종목별 R² 의 중앙값.
2. DeepLOB 대 릿지: 같은 평가 행에서 종목 단위 짝지은 부트스트랩 차이 (paired_analysis 그대로).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgu/tick/symbolic/0911/groups")
import group_analysis as ga  # noqa: E402
import paired_analysis as paired  # noqa: E402

df = pd.read_parquet(ga.PRED)
df["symbol"] = df.symbol.astype(str).str.zfill(6)
assign = pd.read_csv(ga.ASSIGN, index_col=0)
assign.index = assign.index.astype(str).str.zfill(6)
out = {"note": "서술 — 사전등록 판정 아님"}

for es in ("eval1", "eval2"):
    part = df[df.eval_set == es]
    cat = pd.Categorical(part.symbol)
    codes, syms = cat.codes.astype(np.int64), np.asarray(cat.categories)
    S, y = len(syms), part.y_path.to_numpy(float)
    ybar = y.mean()
    contrib = np.bincount(codes, weights=(y - ybar) ** 2, minlength=S)      # 전체 SST 에 대한 종목별 몫
    share = contrib / contrib.sum()
    res = {"n_symbols": int(S), "n_rows": int(len(y)), "y_std": float(y.std())}
    order = np.argsort(-share)
    res["sst_share_top"] = {k: float(share[order[:k]].sum()) for k in (1, 5, 10, 20, 50)}
    grp = assign.reindex(syms)
    res["sst_share_by_tick"] = {g: float(share[grp.tick_group.to_numpy() == g].sum()) for g in ga.TICK_ORDER}
    res["rows_share_by_tick"] = {g: float(np.isin(codes, np.flatnonzero(grp.tick_group.to_numpy() == g)).mean())
                                 for g in ga.TICK_ORDER}
    for m in ("deeplob", "ridge"):
        pred = part[f"pred_{m}"].to_numpy(float)
        st = ga.symbol_stats(codes, y, pred, S)
        sst_i = st[:, 2] - st[:, 1] ** 2 / np.maximum(st[:, 0], 1)
        r2_i = np.where(sst_i > 0, 1 - st[:, 3] / np.where(sst_i > 0, sst_i, 1), np.nan)
        sse = np.bincount(codes, weights=(y - pred) ** 2, minlength=S)
        r = {"pooled_r2": ga.r2_from_stats(st),
             "median_symbol_r2": float(np.nanmedian(r2_i)),
             "frac_symbols_r2_pos": float(np.nanmean(r2_i > 0)),
             "pooled_r2_without_top_sst": {}}
        for k in (1, 5, 10, 20):
            keep = np.ones(S, bool); keep[order[:k]] = False
            r["pooled_r2_without_top_sst"][k] = ga.r2_from_stats(st[keep])
        r["top10"] = [{"symbol": syms[i], "tick": grp.tick_group.iloc[i], "liq": grp.liq_group.iloc[i],
                       "rows": int(st[i, 0]), "sst_share": float(share[i]), "symbol_r2": float(r2_i[i]),
                       "sse_share": float(sse[i] / sse.sum())} for i in order[:10]]
        res[m] = r
    sym, yy, a, b, nd = paired.load_aligned(ga.PRED, ga.PRED, es, "pred_ridge")
    _, _, d_, _, _ = paired.load_aligned(ga.PRED, ga.PRED, es, "pred_deeplob")
    res["deeplob_minus_ridge"] = paired.paired_delta(sym, yy, a, d_)
    res["deeplob_minus_ridge"].pop("verdict")
    res["pred_corr_deeplob_ridge"] = float(np.corrcoef(a, d_)[0, 1])
    out[es] = res

(ga.ROOT / "out" / "eval_description.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for es in ("eval1", "eval2"):
    r = out[es]
    print(f"=== {es}: 종목 {r['n_symbols']} 행 {r['n_rows']:,} y 표준편차 {r['y_std']:.4g}")
    print("  전체 변동 몫 상위 k 종목:", {k: round(v, 3) for k, v in r["sst_share_top"].items()})
    print("  틱군별 변동 몫:", {k: round(v, 3) for k, v in r["sst_share_by_tick"].items()},
          " 행 몫:", {k: round(v, 3) for k, v in r["rows_share_by_tick"].items()})
    for m in ("deeplob", "ridge"):
        q = r[m]
        print(f"  [{m}] 전체 {q['pooled_r2']:+.5f} | 종목별 R² 중앙값 {q['median_symbol_r2']:+.5f}, 양수 비율 {q['frac_symbols_r2_pos']:.2f}"
              f" | 변동 상위 제외 {{{', '.join(f'{k}: {v:+.5f}' for k, v in q['pooled_r2_without_top_sst'].items())}}}")
        for t in q["top10"][:5]:
            print(f"      {t['symbol']} {t['tick']}/{t['liq']} 행 {t['rows']:,} 변동몫 {t['sst_share']:.3f} 오차몫 {t['sse_share']:.3f} 종목 R² {t['symbol_r2']:+.4f}")
    d = r["deeplob_minus_ridge"]
    print(f"  DeepLOB − 릿지: {d['delta']:+.5f} [{d['delta_lo']:+.5f}, {d['delta_hi']:+.5f}], 예측 상관 {r['pred_corr_deeplob_ridge']:.3f}")
