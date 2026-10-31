"""Stage O1: W3/W4 生产策略逐笔归因（读 O1_*_trades.json，回答任务书 5 问）"""
import json, glob, os
import numpy as np, pandas as pd

DIR = os.path.expanduser("~/.hermes/cache/scratch/o1_trades")
files = {
    "W3_orig": f"{DIR}/O1_W3_trades.json", "W3_strict": f"{DIR}/O1_W3s_trades.json",
    "W4_orig": f"{DIR}/O1_W4_trades.json", "W4_strict": f"{DIR}/O1_W4s_trades.json",
}

def leg_stats(df):
    if len(df) == 0:
        return None
    pnl = df["pnl_frac"].to_numpy()
    wins = df[df.pnl_frac > 0].pnl_frac; losses = df[df.pnl_frac <= 0].pnl_frac
    eq = np.cumsum(pnl); mdd = float((np.maximum.accumulate(eq) - eq).max())
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return {"n": int(len(df)), "ret": float(pnl.sum()), "win_rate": float((pnl > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
            "payoff": (float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None),
            "profit_factor": pf, "mdd": float(mdd), "best": float(pnl.max()), "worst": float(pnl.min())}

out = {}
for tag, f in files.items():
    d = json.load(open(f))
    t = pd.DataFrame(d["trades"])
    gross = t["gross_pnl"].sum(); fee = (t["open_cost"] + t["close_cost"]).sum(); fund = t["funding"].sum()
    net = t["pnl_frac"].sum() * 1.0  # pnl_frac 已扣全部成本（对账：net = gross - fee - fund，按初始权益=1 口径近似）
    # 对账（绝对额，初始权益 1）
    net_abs = t["pnl_frac"].sum()
    recon = gross - fee - fund
    long_t = t[t.dir == 1]; short_t = t[t.dir == -1]
    # 集中度
    pnl_sorted = t.sort_values("pnl_frac", ascending=False)
    top5 = pnl_sorted.head(5).pnl_frac.sum(); top1 = pnl_sorted.head(1).pnl_frac.sum()
    # 按 max_tier 拆分（8x=3 / 5x=2 / 2x=1）
    by_tier = {}
    for tier in [1, 2, 3]:
        sub = t[t.max_tier == tier]
        if len(sub):
            by_tier[f"tier{tier}({'2x' if tier==1 else '5x' if tier==2 else '8x'})"] = {
                "n": int(len(sub)), "ret": float(sub.pnl_frac.sum()), "win_rate": float((sub.pnl_frac > 0).mean())}
    # 开仓档位 × 平仓原因
    pivot = t.groupby(["entry_tier", "reason"]).pnl_frac.agg(["sum", "count"]).reset_index()
    # MFE/MAE/持仓
    o = {
        "total_return": d["total_return"], "n_trades": d["n_trades"],
        "leg_long": leg_stats(long_t), "leg_short": leg_stats(short_t),
        "cost_split": {"gross": float(gross), "fee_slippage": float(fee), "funding": float(fund),
                       "net_abs_check": float(recon), "net_actual": float(net_abs),
                       "fee_pct_of_gross": float(fee / gross) if gross > 0 else None,
                       "fee_pct_of_abs_gross": float(fee / abs(gross))},
        "concentration": {"top1_trade": float(top1), "top5_trades": float(top5),
                          "top5_pct_of_net": float(top5 / net_abs) if net_abs else None,
                          "top5_trades_list": pnl_sorted.head(5)[["entry_time", "dir", "leverage", "pnl_frac", "reason"]].to_dict("records")},
        "by_max_tier": by_tier,
        "exit_reason_pnl": t.groupby("reason").pnl_frac.agg(["sum", "count"]).to_dict("index"),
        "entry_tier_x_exit": pivot.to_dict("records"),
        "mfe_median": d.get("mfe_median"), "mae_median": d.get("mae_median"),
        "hold_median_min": float(t["hold_minutes"].median()),
        "hold_dist_min": {q: float(x) for q, x in t["hold_minutes"].quantile([0.25, 0.5, 0.75, 0.9]).items()},
        "mfe_dist": {q: float(x) for q, x in t["mfe_phys"].quantile([0.25, 0.5, 0.75, 0.9]).items()},
        "mae_dist": {q: float(x) for q, x in t["mae_phys"].quantile([0.1, 0.25, 0.5, 0.75]).items()},
        "cum_pnl_by_trade": [round(float(x), 5) for x in np.cumsum(t.sort_values("exit_time")["pnl_frac"])],
        "lev_hist": t["leverage"].value_counts().to_dict(),
    }
    out[tag] = o
    print(f"=== {tag} (ret {d['total_return']:+.2%}, {d['n_trades']} 笔) ===")
    for side in ["leg_long", "leg_short"]:
        s = o[side]
        if s:
            print(f"  {side:10s}: n={s['n']:3d} ret={s['ret']:+.4f} wr={s['win_rate']:.2f} payoff={s['payoff'] or 0:.2f} PF={s['profit_factor']:.2f} MDD={s['mdd']:.4f} best/worst={s['best']:+.4f}/{s['worst']:+.4f}")
    cs = o["cost_split"]
    print(f"  成本: gross={cs['gross']:+.4f} fee={cs['fee_slippage']:.4f} funding={cs['funding']:.4f} → 对账net={cs['net_abs_check']:+.4f} vs 实际={cs['net_actual']:+.4f}")
    print(f"  集中度: top1={o['concentration']['top1_trade']:+.4f} top5={o['concentration']['top5_trades']:+.4f} ({o['concentration']['top5_pct_of_net']:.0%} of net)")
    print(f"  按最高档位: " + " | ".join(f"{k}: n={v['n']} {v['ret']:+.4f}" for k, v in o["by_max_tier"].items()))
    print(f"  平仓原因PnL: " + " | ".join(f"{k}:{v['sum']:+.4f}({int(v['count'])})" for k, v in o["exit_reason_pnl"].items()))
    print(f"  持仓中位={o['hold_median_min']:.0f}min MFE中位={o['mfe_median']:+.4f} MAE中位={o['mae_median']:+.4f}")

json.dump(out, open(f"{DIR}/o1_attribution.json", "w"), ensure_ascii=False, indent=1, default=float)
print("SAVED", f"{DIR}/o1_attribution.json")
