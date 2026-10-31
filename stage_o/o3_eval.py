"""Stage O3: 条件化策略评估（6 基准 × 4 窗口，全指标 + CI + per-regime 分解）
输入: o3_json/O3_{strat}_{w}.json（prod/long/short/regime @1x）+ regime_series.csv + B&H（raw 1h close）
口径: 1x 杠杆，引擎完整生产出场逻辑（prod 为原档位压平到 1x；regime 含 BT_REGIME_FLAT 强平）
CI: 逐笔 pnl_frac bootstrap B=2000（95%）；trades 非重叠近似独立
"""
import json, glob, os
import numpy as np, pandas as pd

DIR = os.path.expanduser("~/.hermes/cache/scratch/o3_json")
REGIME_CSV = os.path.expanduser("~/.hermes/cache/scratch/o3_json/regime_series.csv")
WINDOWS = ["W1", "W2", "W3", "W4"]
STRATS = ["prod", "long", "short", "regime"]
RNG = np.random.default_rng(42)
B = 2000

# regime 序列
reg = pd.read_csv(REGIME_CSV)
r_ts = reg["ts_sec"].to_numpy(); r_vals = reg["regime"].to_numpy()

def regime_at(t):
    i = int(np.searchsorted(r_ts, t, side="right")) - 1
    return r_vals[i] if i >= 0 else "NA"

def mdd(curve):
    peak = np.maximum.accumulate(curve)
    return float((peak - np.array(curve)).max())

def sharpe_sortino(curve):
    r = np.diff(curve) / np.array(curve[:-1])
    mu, sd = r.mean(), r.std()
    sh = (mu / sd * np.sqrt(8760)) if sd > 0 else 0.0
    dn = r[r < 0]
    ds = dn.std() if len(dn) else 0.0
    so = (mu / ds * np.sqrt(8760)) if ds > 0 else 0.0
    return float(sh), float(so)

def boot_ci(pnls, b=B):
    if len(pnls) < 3:
        return [float(np.sum(pnls)) - 9e9, float(np.sum(pnls)) + 9e9] if False else [None, None]
    idx = RNG.integers(0, len(pnls), size=(b, len(pnls)))
    totals = np.sum(pnls[idx], axis=1)
    return [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))]

# B&H: 引擎窗口 = 同一组；用 O3 JSON 的 benchmark？——直接由 raw 1h 计算（脚本外传入 bh dict）
import sys
bh = {}
if len(sys.argv) > 1:
    bh = json.load(open(sys.argv[1]))

results = {}
for s in STRATS:
    results[s] = {}
    for w in WINDOWS:
        f = f"{DIR}/O3_{s}_{w}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        t = pd.DataFrame(d["trades"]) if d["trades"] else pd.DataFrame()
        eq = np.array(d["equity_curve"])
        sh, so = sharpe_sortino(eq)
        row = {
            "net": float(eq[-1] - 1), "mdd": mdd(eq), "sharpe": sh, "sortino": so,
            "n_trades": d["n_trades"], "win_rate": d["win_rate"], "pf": d["profit_factor"],
            "exit_reasons": {k: v for k, v in d["exit_reasons"].items() if v},
        }
        if len(t):
            row["long_ret"] = float(t.loc[t.dir == 1, "pnl_frac"].sum())
            row["short_ret"] = float(t.loc[t.dir == -1, "pnl_frac"].sum())
            gross = float(t["gross_pnl"].sum()); fee = float((t["open_cost"] + t["close_cost"]).sum())
            fund = float(t["funding"].sum())
            row["cost_gross"] = {"gross": gross, "fee": fee, "funding": fund,
                                 "fee_pct_abs_gross": fee / abs(gross) if gross else None,
                                 "cost_drag_pct": (fee + fund)}
            row["ci_net_95"] = boot_ci(t["pnl_frac"].to_numpy())
            row["top5_conc"] = float(t["pnl_frac"].nlargest(5).sum())
            # per-regime（按开仓时点）
            t["regime_at_entry"] = t["entry_time"].map(regime_at)
            grp = t.groupby("regime_at_entry")["pnl_frac"].agg(["sum", "count"])
            row["by_regime"] = {k: {"ret": float(v["sum"]), "n": int(v["count"])} for k, v in grp.iterrows()}
            # per-exit-reason
            row["by_exit"] = {k: float(v) for k, v in t.groupby("reason")["pnl_frac"].sum().items()}
        results[s][w] = row

results["bh"] = {w: {"net": bh.get(w, None)} for w in WINDOWS}
results["cash"] = {w: {"net": 0.0} for w in WINDOWS}

json.dump(results, open(f"{DIR}/o3_results.json", "w"), ensure_ascii=False, indent=1, default=float)

# 打印主表
print("=== 主表（1x，生产阈值，窗口净收益）===")
print(f"{'策略':8s}" + "".join(f"{w:>10s}" for w in WINDOWS))
for s in STRATS + ["bh", "cash"]:
    line = f"{s:8s}"
    for w in WINDOWS:
        v = results[s].get(w, {})
        line += f"{v.get('net', 0):>+10.2%}" if v.get("net") is not None else f"{'—':>10s}"
    print(line)
print()
for s in STRATS:
    print(f"=== {s} 明细 ===")
    for w in WINDOWS:
        v = results[s].get(w)
        if not v:
            continue
        ci = v.get("ci_net_95", [None, None])
        cis = f"[{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci[0] is not None else "n<3"
        print(f"  {w}: net={v['net']:+.2%} MDD={v['mdd']:.2%} PF={v.get('pf') or 0:.2f} Sharpe={v['sharpe']:.2f} Sortino={v['sortino']:.2f} n={v['n_trades']} CI95={cis}")
        if "long_ret" in v:
            cg = v["cost_gross"]
            print(f"       多={v['long_ret']:+.2%} 空={v['short_ret']:+.2%} | gross={cg['gross']:+.1%} 费={cg['fee']:.1%} 资金费={cg['funding']:.2%} (费/|gross|={cg['fee_pct_abs_gross'] if cg['fee_pct_abs_gross'] is None else format(cg['fee_pct_abs_gross'], '.0%')}) top5集中={v['top5_conc']:+.2%}")
        if "by_regime" in v:
            print(f"       按开仓regime: " + " | ".join(f"{k}:{x['ret']:+.2%}({x['n']})" for k, x in v["by_regime"].items()))
        print(f"       按平仓原因: " + " | ".join(f"{k}:{x:+.2%}" for k, x in v["by_exit"].items()))
print("SAVED", f"{DIR}/o3_results.json")
