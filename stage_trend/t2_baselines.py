"""P2: 6 条无模型基线 × 5 窗（1x 全成本，Block Bootstrap CI，窗口相关矩阵，H1 判定）
决策口径: day t 收盘决策（只用 ≤t 的数据 + t 日 00:00 已可用的 regime_t），位置在 t+1 生效。
成本: 0.0007/边（换边按 |Δpos|），funding 按持仓逐 8h 计（long 付/收 = −pos×rate）。
H1: 每笔毛期望(价格,未扣费)/0.0014 的中位比值 vs 1h 的 0.4。
"""
import os, json, time
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np

BASE = Path(os.path.expanduser("~/quant_iTransformer/data/train"))
OUT = BASE / "processed" / "trend_4h" / "BTCUSDT"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

COST_SIDE = 0.0007
WINDOWS = [("W1","2023-01-01","2023-06-30"),("W2","2024-01-01","2024-06-30"),
           ("W3","2024-10-01","2025-03-31"),("W4","2025-07-01","2025-12-31"),
           ("W5","2026-04-01","2026-10-27")]

# ============ 数据 ============
d = pl.read_parquet(OUT / "btc_1d_bars.parquet").sort("bar_start").with_columns(
    (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret1d")).drop_nulls("ret1d")
# 每日 funding (3 个 8h 事件求和)
fund4 = pl.read_parquet(OUT / "funding_4h.parquet")
fund1d = fund4.with_columns(pl.col("bar_start").dt.truncate("1d").alias("day")).group_by("day").agg(
    pl.col("funding_cum_4h").sum().alias("funding_1d")).sort("day")
d = d.with_columns(pl.col("bar_start").dt.truncate("1d").alias("day")).join(fund1d.rename({"day": "bar_start"}), on="bar_start", how="left")
d = d.with_columns(pl.col("funding_1d").fill_null(0.0))
# regime: 日最后一小时 → 次日可用 (shift 1)
reg = pl.read_csv(BASE / "regime/regime_series.csv")
reg = reg.with_columns(pl.from_epoch(pl.col("ts_sec"), time_unit="s").alias("ts")).sort("ts")
reg1d = reg.group_by(pl.col("ts").dt.truncate("1d")).agg(pl.col("regime").last().alias("regime_t")).sort("ts")
d = d.join(reg1d.rename({"ts": "bar_start"}), on="bar_start", how="left")
d = d.with_columns(pl.col("regime_t").fill_null("NA"))
log(f"1d 数据: {d.height} 天, {d['bar_start'][0]} → {d['bar_start'][-1]}")

# ============ 策略位置 (day t 收盘决策 → pos 序列, t+1 生效由 r 计算时 shift) ============
close = d["close"].to_numpy()
high = d["high"].to_numpy()
low = d["low"].to_numpy()
regime = d["regime_t"].to_list()
ema50 = d.with_columns(pl.col("close").ewm_mean(span=50, adjust=False).alias("e50"))["e50"].to_numpy()
ema200 = d.with_columns(pl.col("close").ewm_mean(span=200, adjust=False).alias("e200"))["e200"].to_numpy()
sma200 = np.array([np.mean(close[max(0,i-199):i+1]) if i >= 199 else np.nan for i in range(len(close))])

def donchian_pos():
    pos, cur = np.zeros(len(close)), 0.0
    for i in range(20, len(close)):
        hh = high[i-20:i].max(); ll = low[i-10:i].min()
        if cur == 0 and close[i] > hh: cur = 1
        elif cur == 1 and close[i] < ll: cur = 0
        pos[i] = cur
    return pos

P = {
    "bh": np.ones(len(close)),
    "det_long": np.array([1.0 if r == "BULL" else 0.0 for r in regime]),
    "det_ls": np.array([1.0 if r == "BULL" else (-1.0 if r == "BEAR" else 0.0) for r in regime]),
    "sma200": np.array([1.0 if (not np.isnan(sma200[i]) and close[i] > sma200[i]) else 0.0 for i in range(len(close))]),
    "donchian20_10": donchian_pos(),
    "ema50x200": np.array([1.0 if (not np.isnan(ema200[i]) and ema50[i] > ema200[i]) else 0.0 for i in range(len(close))]),
}
log("策略位置构建完成")

# ============ 评估 ============
ret = d["ret1d"].to_numpy()
fund = d["funding_1d"].to_numpy()
bar_starts = d["bar_start"].to_list()

def block_bootstrap_ci(vals, B, n_boot=2000, seed=42):
    """平稳 bootstrap: 随机起点 + 几何块长(均值 B), 环绕取样"""
    v = np.asarray(vals, float)
    n = len(v)
    if n < 8: return [float(np.mean(v))] * 2
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        take, s = 0, 0.0
        while take < n:
            pos_i = int(rng.integers(0, n))  # 每块独立随机起点 (连续游走=循环移位, 和不变)
            # 手工几何分布(均值 B): 本 venv 的 rng.geometric 恒返回 1/p (实测 bug), 不用
            L = int(np.floor(np.log(rng.random()) / np.log(1.0 - 1.0 / B))) + 1
            L = max(1, min(L, n - take))
            seg = v[pos_i:pos_i + L] if pos_i + L <= n else np.concatenate([v[pos_i:], v[:pos_i + L - n]])
            s += seg.sum()
            take += L
        means.append(s / n)
    means = np.array(means)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

results = {}
bs = d["bar_start"].to_list()
for name, pos_dec in P.items():
    results[name] = {}
    pos = np.concatenate([[0.0], pos_dec[:-1]])  # 全序列生效位置
    for wname, ws, we in WINDOWS:
        ws_dt, we_dt = datetime.fromisoformat(ws), datetime.fromisoformat(we + "T23:59:59")
        i0 = next(i for i, b in enumerate(bs) if b >= ws_dt)
        i1 = max(i for i, b in enumerate(bs) if b <= we_dt)
        rnet_w = pos[i0:i1+1] * ret[i0:i1+1] + pos[i0:i1+1] * fund[i0:i1+1] \
                 - np.abs(np.diff(pos[i0-1:i1+1])) * COST_SIDE
        eq = np.cumprod(1 + rnet_w)
        peak = np.maximum.accumulate(eq)
        mdd = float(np.max((peak - eq) / peak)) if peak.max() > 0 else 0.0
        pos_w = pos[i0:i1+1]
        trades, cur, t_entry = [], 0.0, 0
        for t in range(len(pos_w)):
            if pos_w[t] != cur:
                if cur != 0: trades.append((t_entry, t-1, cur))
                cur, t_entry = pos_w[t], t
        if cur != 0: trades.append((t_entry, len(pos_w)-1, cur))
        tr_gross, tr_net = [], []
        for (s, e, p) in trades:
            g = float(np.sum(ret[i0+s:i0+e+1] * p))
            tr_gross.append(g); tr_net.append(g - 0.0014 - p * float(np.sum(fund[i0+s:i0+e+1])))
        tr_gross = np.array(tr_gross); tr_net = np.array(tr_net)
        pf = float(tr_net[tr_net > 0].sum() / abs(tr_net[tr_net < 0].sum())) if (tr_net < 0).any() and (tr_net > 0).any() else (99.0 if (tr_net > 0).any() else 0.0)
        rec = {"ret_total": round(float(np.prod(1 + rnet_w) - 1), 5), "mdd": round(mdd, 5), "n_trades": len(tr_gross),
               "median_E_net": round(float(np.median(tr_net)), 6) if len(tr_net) else None,
               "median_E_gross": round(float(np.median(tr_gross)), 6) if len(tr_gross) else None,
               "h1_ratio": round(float(np.median(tr_gross) / 0.0014), 3) if len(tr_gross) else None,
               "pf": round(pf, 3), "n_days": len(pos_w)}
        for B in (3, 7, 14):
            ci = block_bootstrap_ci(tr_net, B)
            rec[f"ci_trades_B{B}"] = [round(x, 5) for x in ci]
            ci_d = block_bootstrap_ci(rnet_w, B)  # 日收益级 (180 obs, 主口径)
            rec[f"ci_daily_B{B}"] = [round(x, 5) for x in ci_d]
            rec[f"ci_cumret_B{B}"] = [round(x * len(rnet_w), 5) for x in ci_d]  # 窗累计净收益 CI
        if len(rnet_w) > 10:
            r1d = float(np.corrcoef(rnet_w[1:], rnet_w[:-1])[0, 1])
            rec["n_eff_daily"] = round(len(rnet_w) / (1 + 2 * max(0.0, r1d)), 1)
        if len(tr_net) > 4:
            r1 = np.corrcoef(tr_net[1:], tr_net[:-1])[0, 1]
            rec["n_eff_block"] = round(len(tr_net) / (1 + 2 * max(0.0, float(r1))), 1)
        else:
            rec["n_eff_block"] = len(tr_net)
        results[name][wname] = rec
        results[name][wname]["_rnet"] = rnet_w
        log(f"{name} {wname}: ret={rec['ret_total']:+.2%} mdd={rec['mdd']:.2%} n={rec['n_trades']} E_net中位={rec['median_E_net']} h1={rec['h1_ratio']}")

# ============ 5x5 窗口相关矩阵 + n_eff (每策略) ============
log("窗口相关矩阵 ...")
corr = {}
for name in P:
    M = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            a = results[name][WINDOWS[i][0]]["_rnet"]; b = results[name][WINDOWS[j][0]]["_rnet"]
            n = min(len(a), len(b))
            M[i, j] = np.corrcoef(a[:n], b[:n])[0, 1]
    off = M[~np.eye(5, dtype=bool)]
    corr[name] = {"matrix": [[round(float(x), 4) for x in row] for row in M],
                  "n_eff_window": round(5 / (1 + 2 * max(0.0, float(off.mean()))), 3)}
    log(f"{name}: n_eff_window={corr[name]['n_eff_window']}")

# ============ H1 判定 ============
log("H1 ...")
h1 = {}
for name in P:
    ratios = [results[name][w]["h1_ratio"] for w, _, _ in WINDOWS if results[name].get(w, {}).get("h1_ratio") is not None]
    h1[name] = {"median_h1_ratio": round(float(np.median(ratios)), 3) if ratios else None, "n_windows": len(ratios),
                "verdict_vs_1h_0.4": "高于1h" if (ratios and np.median(ratios) > 0.4) else "不高于1h"}
for k, v in h1.items(): log(f"H1 {k}: {v}")

clean = {name: {w: {k2: v2 for k2, v2 in ww.items() if not k2.startswith("_")} for w, ww in ws_.items() if not w.startswith("_")} for name, ws_ in results.items()}
out = {"cost_side": COST_SIDE, "windows": {w: [a, b] for w, a, b in WINDOWS},
       "strategies": clean, "window_corr": corr, "h1": h1,
       "bh_benchmark": {w: clean["bh"][w]["ret_total"] for w, _, _ in WINDOWS}}
with open(OUT / "p2_results.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("DONE_T2_BASELINES")
print(json.dumps({"h1": h1, "corr_neff": {k: v["n_eff_window"] for k, v in corr.items()}}, indent=1))
