"""2,507개 주식 종목의 종목군 기준값. 종목군 경계는 이 분포로 정하고, 종목군별 R² 는 그 뒤에 본다.

틱은 표를 박지 않고 종목별 1~5호가 인접 간격의 최빈값으로 정한다(PROGRESS.md 발견 1).
종목 집합·마찰 층은 0902 의 sd.universe 를 그대로 쓴다 — 두 번째 진실 원천을 만들지 않는다.
"""
import sys, json, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

sys.path.insert(0, "/home/dgu/tick/symbolic/0902")
from sd import universe  # noqa: E402

DATE = "20260316"
OUT = "/home/dgu/tick/symbolic/0911/groups/data/symbol_covariates.csv"
COLS = (["data_type", "time", "cumulative_trading_amount"]
        + [f"{s}{i}_price" for i in range(1, 6) for s in ("ask", "bid")])
PATHS = {}


def one(sym):
    t = pq.read_table(PATHS[sym], columns=COLS).to_pandas()
    q = t[(t.data_type == 12) & (t.time >= "09:00") & (t.time < "15:30")
          & (t.bid1_price > 0) & (t.ask1_price > t.bid1_price)]
    if len(q) < 50:
        return {"symbol": sym, "ok": False}
    gaps = []
    for s in ("ask", "bid"):
        for lv in range(1, 5):
            a = q[f"{s}{lv}_price"].to_numpy(); b = q[f"{s}{lv+1}_price"].to_numpy()
            ok = (a > 0) & (b > 0); d = np.abs(b[ok] - a[ok]); gaps.append(d[d > 0])
    g = np.concatenate(gaps)
    if g.size == 0:
        return {"symbol": sym, "ok": False}
    vals, cnt = np.unique(g, return_counts=True)
    tick = float(vals[cnt.argmax()])
    sp = (q.ask1_price - q.bid1_price).to_numpy()
    mid = ((q.ask1_price + q.bid1_price) / 2).to_numpy()
    return {"symbol": sym, "ok": True, "tick": tick, "tick_share": float(cnt.max() / cnt.sum()),
            "median_price": float(np.median(mid)), "rel_tick_bps": float(tick / np.median(mid) * 1e4),
            "mean_spread_ticks": float(np.mean(sp / tick)), "frac_one_tick": float(np.mean(sp == tick)),
            "n_quotes": int(len(q)), "traded_value": float(t.cumulative_trading_amount.max())}


if __name__ == "__main__":
    t0 = time.time()
    syms = universe.stock_symbols(DATE)
    PATHS.update(universe._scan_symbol_paths(DATE))
    strata = universe.assign_strata(universe.liquidity_stats(syms, DATE, workers=16))
    print(f"종목 집합 {len(syms)} → 층화 {len(strata)}", flush=True)
    assert len(strata) == 2507, f"층화 종목 수가 기록(2,507)과 다르다: {len(strata)}"
    with Pool(16) as pool:
        rows = list(pool.imap_unordered(one, list(strata.index), chunksize=8))
    cov = pd.DataFrame(rows).set_index("symbol")
    df = strata.join(cov, how="left")
    df.to_csv(OUT)
    ok = df[df.ok == True]  # noqa: E712
    print(f"기준값 계산 성공 {len(ok)}/{len(df)}  ({time.time()-t0:.0f}s)")
    print("\n틱 판정 확신도(tick_share) 분위:", ok.tick_share.quantile([.05, .25, .5]).round(3).to_dict())
    print("스프레드 평균 틱 수 분위:", ok.mean_spread_ticks.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
    print("1틱 고정 시간 비율 분위:", ok.frac_one_tick.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print("상대 틱(bps) 분위:", ok.rel_tick_bps.quantile([.1, .5, .9]).round(1).to_dict())
    print("\n기존 마찰층 × 스프레드 틱 수 중앙값:")
    print(ok.groupby("friction").agg(n=("tick", "size"), spread_ticks_med=("mean_spread_ticks", "median"),
                                     one_tick_med=("frac_one_tick", "median"),
                                     rel_tick_bps_med=("rel_tick_bps", "median")).round(3).to_string())
    print("\n스프레드(bps) 와 상대 틱(bps) 순위상관:",
          round(ok[["spread_bps", "rel_tick_bps"]].corr(method="spearman").iloc[0, 1], 3))
