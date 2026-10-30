"""N2: C 单边交易映射（39C，零训练，事后映射）
策略（参数全部在严格 val 定/文献标准，未对 OOS 拟合）:
  C1_top{k}: 纯多头 — p_up ≥ val_top{k}(p_up) 才做多，不做空，其余空仓（k ∈ 10/5/1，主口径 top5 预声明）
  C2_top5 : 趋势过滤 — 多: p_up ≥ val_top5 且 return_24h>0 且 ER_24h≥0.5
            空: p_down ≥ val_top5 且 return_24h<0 且 ER_24h≥0.5（Kaufman regime 标准参数，未拟合）
口径: 块起点 12h 网格（同 P0）；固定 12h 出场；成本 0.07%/0.14% 单边；log 收益 r12
MFE/MAE: 1m 路径 [T+1:00, T+13:00)，entry = 首根 1m open
硬约束检查: 交易数 ≤ 当前 C(~135/窗口) 的 1/3
"""
import os, sys, json
import numpy as np, pandas as pd, polars as pl

BASE = os.path.expanduser("~/quant_iTransformer/data/train/processed/main_features_1h/main_features_46d/BTCUSDT")
M46L = os.path.join(BASE, "BTCUSDT_Master_Training_Matrix_12h_46d_L_20261030.parquet")
SIGDIR = os.path.expanduser("~/quant_iTransformer/data/train/n1_signals")
RH = os.path.expanduser("~/quant_iTransformer/data/train/raw_temp/klines/BTCUSDT")
WINDOWS = {
    "W1": ("2024-03-21 00:00:00", "2024-06-19 00:00:00"),
    "W2": ("2024-09-17 00:00:00", "2024-12-16 00:00:00"),
    "W3": ("2025-09-12 00:00:00", "2025-12-11 00:00:00"),
    "W4": ("2026-07-27 00:00:00", "2026-10-27 12:00:00"),
}
SEEDS = [42, 1, 7, 123, 1729, 999]
COST = 0.0007
COST_HI = 0.0014

p0 = json.load(open("/tmp/p0_eval_results.json"))
per_run = p0["per_run"]

print("读取矩阵与 K 线 ...")
m = pl.read_parquet(M46L)
ts = m["hour_timestamp"].cast(pl.Datetime("us")).to_numpy().astype("datetime64[us]").astype("int64") // 10**6
R12 = m["target_12h_return"].to_numpy().astype(float)
ret24 = m["return_24h"].to_numpy().astype(float)
er24 = m["trend_efficiency_ratio"].to_numpy().astype(float)

rm = pd.read_parquet(f"{RH}/BTCUSDT_raw_1m.parquet")
ts_1m = rm["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy() // 10**9
open_1m = rm["open"].to_numpy()
high_1m = rm["high"].to_numpy()
low_1m = rm["low"].to_numpy()

def ts_of(s): return int(pd.Timestamp(s).timestamp())

def mfe_mae(t0, d):
    i0 = int(np.searchsorted(ts_1m, t0 + 3600))
    i1 = int(np.searchsorted(ts_1m, t0 + 13 * 3600))
    if i1 - i0 < 60:
        return np.nan, np.nan
    e = open_1m[i0]
    if d == 1:
        return float((high_1m[i0:i1].max() - e) / e), float((low_1m[i0:i1].min() - e) / e)
    else:
        return float((e - low_1m[i0:i1].min()) / e), float((e - high_1m[i0:i1].max()) / e)

def trade_stats(pos, r12, cost_side):
    cost = np.where(pos != 0, 2 * cost_side, 0.0)
    ret = pos * r12 - cost
    traded = pos != 0
    out = {"n_trades": int(traded.sum()), "exposure": float(traded.mean()) if len(traded) else 0.0,
           "ret_total": float(np.prod(1 + ret) - 1), "win_rate": 0.0, "breakeven_E": 0.0,
           "long_ret": 0.0, "short_ret": 0.0,
           "long_n": int((pos == 1).sum()), "short_n": int((pos == -1).sum())}
    if traded.sum() == 0:
        return out, ret
    tr = ret[traded]
    wins = tr[tr > 0]; losses = tr[tr <= 0]
    W = wins.mean() if len(wins) else 0.0
    L = abs(losses.mean()) if len(losses) else 0.0
    p = len(wins) / len(tr)
    out.update({
        "ret_total": float(np.prod(1 + ret) - 1),
        "win_rate": float(p),
        "breakeven_E": float(p * W - (1 - p) * L - 2 * cost_side),
        "long_ret": float(ret[pos == 1].sum()), "short_ret": float(ret[pos == -1].sum()),
        "long_n": int((pos == 1).sum()), "short_n": int((pos == -1).sum()),
    })
    return out, ret

STRATS = ["C1_top10", "C1_top5", "C1_top1", "C2_top5"]
results = {}
for w, (t0, t1) in WINDOWS.items():
    a, b = ts_of(t0), ts_of(t1)
    tidx = np.where((ts >= a) & (ts < b))[0]
    first = tidx[0]
    dec = tidx[(tidx - first) % 12 == 0]
    r12d = R12[dec]
    for seed in SEEDS:
        sig = pd.read_csv(f"{SIGDIR}/sig_39C_{w}_{seed}.csv")
        p_up_dec = sig["p_up"].to_numpy()[dec - tidx[0]]
        p_down_dec = sig["p_down"].to_numpy()[dec - tidx[0]]
        r24_d = ret24[dec]
        er_d = er24[dec]
        run = per_run[f"{w}/39C"][str(seed)]
        th_u, th_d = run["val_th_up"], run["val_th_down"]
        t0s = ts[dec]
        for cname in STRATS:
            if cname.startswith("C1"):
                k = cname.split("_")[-1]
                pos = np.where((p_up_dec >= th_u[k]), 1, 0).astype(int)
            else:  # C2_top5
                pos = np.zeros(len(dec), dtype=int)
                pos[(p_up_dec >= th_u["top5"]) & (r24_d > 0) & (er_d >= 0.5)] = 1
                pos[(p_down_dec >= th_d["top5"]) & (r24_d < 0) & (er_d >= 0.5)] = -1
            st, ret = trade_stats(pos, r12d, COST)
            st_hi, _ = trade_stats(pos, r12d, COST_HI)
            mfe_l, mae_l = [], []
            for k2 in np.where(pos != 0)[0]:
                mf, ma = mfe_mae(int(t0s[k2]), int(pos[k2]))
                if np.isfinite(mf): mfe_l.append(mf); mae_l.append(ma)
            st.update({
                "ret_total_hi": st_hi["ret_total"],
                "mfe_median": float(np.median(mfe_l)) if mfe_l else None,
                "mae_median": float(np.median(mae_l)) if mae_l else None,
            })
            results[f"{w}|{seed}|{cname}"] = st
        print(f"[{w}|{seed}] done")

# 聚合: 每 (窗口, 策略) 6-seed 中位数 + 全局
print("\n=== N2 汇总（6-seed 中位数，扣费 0.07% 单边）===")
agg = {}
for w in WINDOWS:
    for cname in STRATS:
        rows = [results[f"{w}|{s}|{cname}"] for s in SEEDS]
        def med(k):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k))]
            return float(np.median(vals)) if vals else None
        agg[f"{w}|{cname}"] = {
            "n_trades": med("n_trades"), "ret_total": med("ret_total"), "ret_total_hi": med("ret_total_hi"),
            "win_rate": med("win_rate"), "breakeven_E": med("breakeven_E"),
            "long_ret": med("long_ret"), "short_ret": med("short_ret"),
            "exposure": med("exposure"), "mfe_median": med("mfe_median"), "mae_median": med("mae_median"),
        }
        a = agg[f"{w}|{cname}"]
        print(f"{w} {cname:<9}: 笔={a['n_trades']:.0f} 累计={a['ret_total']*100:+.2f}% (hi {a['ret_total_hi']*100:+.2f}%) "
              f"WR={a['win_rate']*100:.0f}% E/笔={a['breakeven_E']*100:+.3f}% 多={a['long_ret']*100:+.2f}% 空={a['short_ret']*100:+.2f}% "
              f"MFE={a['mfe_median']*100:+.2f}% MAE={a['mae_median']*100:+.2f}%")
# 全局中位数（4 窗口 × 6 seeds）
print("\n全局（4 窗口 × 6 seeds 中位数）:")
for cname in STRATS:
    rows = [results[f"{w}|{s}|{cname}"] for w in WINDOWS for s in SEEDS]
    rt = [r["ret_total"] for r in rows if np.isfinite(r["ret_total"])]
    nh = [r["n_trades"] for r in rows]
    pos_wins = sum(1 for r in rows if r["ret_total"] > 0)
    print(f"{cname:<9}: 累计中位={np.median(rt)*100:+.2f}% | 笔中位={np.median(nh):.0f} | 正收益组合={pos_wins}/24")

with open(os.path.expanduser("~/quant_iTransformer/data/train/n1_runs/n2_results.json"), "w") as f:
    json.dump({"per_run": results, "agg": agg}, f, ensure_ascii=False, indent=1, default=float)
print("\nSAVED n2_results.json")
print("DONE_N2")
