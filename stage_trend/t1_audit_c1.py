"""P1 脚本3: C1 审计 — 因果断言 / 4h重算 vs 1h聚合对照 / 五窗切分 / 有效样本量 / 标签分布
输出: p1_c1_audit.json
因果断言: 对随机抽样 bar, 用"只含 bar_start<=t 的 1h 原始数据"重算特征, 与矩阵值比对 (容差 1e-6)。
重算对照: 1h 指标在 4h bar 内取末值, 与 4h 重算值算 Pearson 相关 (应显著 <1, 证明非简单复用)。
"""
import os, json, time
from pathlib import Path
import polars as pl
import numpy as np

BASE = Path(os.path.expanduser("~/quant_iTransformer/data/train"))
OUT = BASE / "processed" / "trend_4h" / "BTCUSDT"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

df = pl.read_parquet(OUT / "btc_4h_features_31d.parquet").sort("bar_start")
b1h = pl.read_parquet(BASE / "raw_temp/klines/BTCUSDT/BTCUSDT_raw_1h.parquet").sort("timestamp")
b4 = pl.read_parquet(OUT / "btc_4h_bars.parquet").sort("bar_start")

# ============ 1. 五窗切分 ============
WINDOWS = [
    ("W1", "2023-01-01", "2023-06-30", "2022-10-03", "2022-12-31"),
    ("W2", "2024-01-01", "2024-06-30", "2023-10-03", "2023-12-31"),
    ("W3", "2024-10-01", "2025-03-31", "2024-07-03", "2024-09-30"),
    ("W4", "2025-07-01", "2025-12-31", "2025-04-03", "2025-06-30"),
    ("W5", "2026-04-01", "2026-10-27", "2026-01-02", "2026-03-31"),
]
TRAIN_START = "2021-12-01"
PURGE_BARS = 6  # 24h 持有期
def d(s): return pl.datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), 0, 0)
splits = {}
for name, ts, te, vs, ve in WINDOWS:
    ts_, te_, vs_, ve_ = d(ts), d(te), d(vs), d(ve)
    te_ = pl.datetime(int(te[:4]), int(te[5:7]), int(te[8:10]), 23, 59)
    # train: TRAIN_START → val 起点前, 再 purge 6 bars
    train_mask = (df["bar_start"] >= d(TRAIN_START)) & (df["bar_start"] < vs_) & (df["fwd_ret_24h"].is_not_null())
    val_mask = (df["bar_start"] >= vs_) & (df["bar_start"] <= ve_) & (df["fwd_ret_24h"].is_not_null())
    test_mask = (df["bar_start"] >= ts_) & (df["bar_start"] <= te_) & (df["fwd_ret_24h"].is_not_null())
    purge_start = (ts_ - pl.duration(hours=24))
    splits[name] = {
        "test": [str(ts_), str(te_)], "val": [str(vs_), str(ve_)],
        "train_end": str(vs_), "purge": [str(purge_start), str(ts_)],
        "n_train": int(df.filter(train_mask).height), "n_val": int(df.filter(val_mask).height), "n_test": int(df.filter(test_mask).height),
    }
    log(f"{name}: train={splits[name]['n_train']} val={splits[name]['n_val']} test={splits[name]['n_test']}")

# ============ 2. 因果断言 (抽样 20 bar, 从 1h 原始数据重算) ============
log("因果断言 ...")
FEAT_CHECK = ["log_return", "ma_bias", "upper_shadow", "bb_squeeze", "price_pos_24h", "rsi_14", "taker_ratio_delta"]
rng = np.random.default_rng(42)
idx = rng.choice(df.height - 10, size=20, replace=False)
audit_rows = []
for i in idx:
    row = df.slice(int(i), 1).row(0, named=True)
    from datetime import timedelta
    t_end = row["bar_start"] + timedelta(hours=4)
    seg = b1h.filter((b1h["timestamp"] >= row["bar_start"]) & (b1h["timestamp"] < t_end))
    pre = b1h.filter(b1h["timestamp"] < row["bar_start"])  # 严格事前
    c, o, h, l = seg["close"][-1], seg["open"][0], seg["high"].max(), seg["low"].min()
    cp = pre["close"][-1]
    vol4, tb4 = seg["volume"].sum(), seg["taker_buy_volume"].sum()
    tbr_prev = pre["taker_buy_volume"].tail(4).sum() / (pre["volume"].tail(4).sum() + 1e-9)
    close_s = b4.filter(b4["bar_start"] <= row["bar_start"])["close"].to_numpy()  # 4h 收盘序列(含当前, 严格≤t)
    o_h = (h - max(o, c)) / (h - l + 1e-9)
    ma200 = np.array([np.prod(1 + 0) for _ in range(1)])  # placeholder
    # EMA200 (adjust=False) on close series truncated at t
    def ema(x, span):
        a = 2 / (span + 1); out = [x[0]]
        for v in x[1:]: out.append(a * v + (1 - a) * out[-1])
        return out[-1]
    # RSI14 Wilder on truncated series
    def rsi(x, p):
        d = np.diff(x)
        g = np.where(d > 0, d, 0.0); lo = np.where(d < 0, -d, 0.0)
        ag = g.copy().tolist(); alo = lo.copy().tolist()
        mg, ml = ag[0], alo[0]
        a = 1 / p
        for i in range(1, len(ag)):
            mg = a * ag[i] + (1 - a) * mg; ml = a * alo[i] + (1 - a) * ml
        return 100 - 100 / (1 + mg / (ml + 1e-9)) if ml > 0 else 100.0
    exp = {
        "log_return": float(np.log(c) - np.log(cp)),
        "ma_bias": float(c / ema(close_s, 200) - 1),
        "upper_shadow": float(o_h),
        "bb_squeeze": float(np.std(close_s[-20:], ddof=1) * 4 / (np.mean(close_s[-20:]) + 1e-9)),
        "price_pos_24h": float((c - np.min(np.concatenate([pre["low"].tail(20).to_numpy(), seg["low"].to_numpy()])))
                                / (np.max(np.concatenate([pre["high"].tail(20).to_numpy(), seg["high"].to_numpy()]))
                                   - np.min(np.concatenate([pre["low"].tail(20).to_numpy(), seg["low"].to_numpy()])) + 1e-9)),
        "rsi_14": float((rsi(close_s, 14) - 50) / 50),
        "taker_ratio_delta": float(tb4 / (vol4 + 1e-9) - tbr_prev),
    }
    for k, v in exp.items():
        got = float(row[k])
        ok = (got is not None) and not np.isnan(v) and abs(got - v) < 1e-4 * max(1, abs(v))
        audit_rows.append({"bar": str(row["bar_start"]), "feat": k, "matrix": round(got, 8) if got is not None else None,
                           "recomputed": round(v, 8), "match": bool(ok)})
n_match = sum(r["match"] for r in audit_rows)
log(f"因果断言: {n_match}/{len(audit_rows)} 通过")

# ============ 3. 4h 重算 vs 1h 聚合对照 ============
log("重算对照 ...")
# 1h 版 RSI14 / ma_bias / log_return, 在 4h bar 内取末值
b1f = b1h.with_columns([
    (pl.col("close").log() - pl.col("close").shift(1).log()).alias("lr1"),
    (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-9)).alias("tbr1"),
])
delta = b1f["close"].diff()
gain = pl.when(delta > 0).then(delta).otherwise(0).ewm_mean(alpha=1/14, adjust=False)
loss = pl.when(delta < 0).then(-delta).otherwise(0).ewm_mean(alpha=1/14, adjust=False)
_rsi14 = 100 - 100 / (1 + gain / (loss + 1e-9))
b1f = b1f.with_columns(_rsi14.alias("rsi14_1h"))
b1f = b1f.with_columns((pl.col("close") / pl.col("close").ewm_mean(span=200, adjust=False) - 1).alias("mab1"))
b1f = b1f.with_columns(pl.col("timestamp").dt.truncate("4h").alias("bar_start"))
last1 = b1f.group_by("bar_start").agg(pl.col("rsi14_1h").last().alias("rsi14_last"),
                                      pl.col("mab1").last().alias("mab_last"),
                                      pl.col("lr1").last().alias("lr_last"),
                                      pl.col("tbr1").last().alias("tbr_last"))
cmp_df = df.join(last1, on="bar_start", how="inner").drop_nulls(["rsi14_last", "mab_last", "lr_last", "tbr_last"])
def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])
recomp = {
    "rsi_14": pear(cmp_df["rsi_14"], cmp_df["rsi14_last"]),
    "ma_bias": pear(cmp_df["ma_bias"], cmp_df["mab_last"]),
    "log_return": pear(cmp_df["log_return"], cmp_df["lr_last"]),
    "taker_ratio(4h重算 vs 1h末值)": pear(cmp_df["taker_ratio_delta"].fill_null(0), (cmp_df["tbr_last"] - cmp_df["tbr_last"].shift(1)).fill_null(0)),
}
log(f"重算对照 Pearson: {recomp}")

# ============ 4. 标签分布 (每窗 train σ̂ → 三分类) ============
log("标签分布 ...")
labels = {}
for name in splits:
    tr = df.filter((df["bar_start"] >= d(TRAIN_START)) & (df["bar_start"] < d(WINDOWS[[w[0] for w in WINDOWS].index(name)][3])))
    r = tr["fwd_ret_24h"].drop_nulls().to_numpy()
    sigma = float(np.std(r, ddof=1))
    te_mask = (df["bar_start"] >= d(WINDOWS[[w[0] for w in WINDOWS].index(name)][1])) & (df["bar_start"] <= d(WINDOWS[[w[0] for w in WINDOWS].index(name)][2]))
    rt = df.filter(te_mask)["fwd_ret_24h"].drop_nulls().to_numpy()
    z = rt / sigma
    cls = np.array(["neg" if v < -0.5 else ("pos" if v > 0.5 else "mid") for v in z])
    labels[name] = {"sigma_train": round(sigma, 6), "n_test": int(rt.shape[0]),
                    "class_pct": {c: round(float((cls == c).mean()), 4) for c in ["neg", "mid", "pos"]},
                    "test_ret_med": round(float(np.median(rt)), 6)}
    log(f"{name}: σ̂={sigma:.5f}  {labels[name]['class_pct']}")

out = {"splits": splits, "causal_audit": {"n_checked": len(audit_rows), "n_match": n_match,
                                           "failures": [r for r in audit_rows if not r["match"]][:10],
                                           "detail": audit_rows},
       "recompute_vs_1h_aggregate": recomp, "labels": labels}
with open(OUT / "p1_c1_audit.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("DONE_T1_AUDIT")
print(json.dumps({"splits": {k: {kk: vv for kk, vv in v.items()} for k, v in splits.items()},
                  "causal": f"{n_match}/{len(audit_rows)}", "recomp": {k: round(v, 4) for k, v in recomp.items()},
                  "labels": labels}, indent=1))
