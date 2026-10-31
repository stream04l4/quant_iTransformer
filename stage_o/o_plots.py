"""Stage O 报告图：O1 逐笔累计 PnL（W3/W4 orig/strict）+ O3 六策略净值曲线"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = os.path.expanduser("~/.hermes/cache/scratch")
o1 = json.load(open(f"{S}/o1_trades/o1_attribution.json"))
o3 = json.load(open(f"{S}/o3_json/o3_results.json"))
outdir = f"{S}/plots"
os.makedirs(outdir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
for ax, tag in zip(axes.flat, ["W3_orig", "W3_strict", "W4_orig", "W4_strict"]):
    cum = np.array(o1[tag]["cum_pnl_by_trade"])
    ax.plot(np.arange(len(cum) + 1), np.concatenate([[0], cum]), lw=1.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{tag}: per-trade cum PnL (native 2/5/8x), net={o1[tag]['total_return']:+.1%}")
    ax.set_xlabel("trade #")
    ax.grid(alpha=0.3)
fig.suptitle("Stage O1: production strategy per-trade cumulative PnL (W3/W4)")
fig.tight_layout()
fig.savefig(f"{outdir}/o1_cum_pnl.png", dpi=110)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
strats = ["prod", "long", "short", "regime"]
colors = {"prod": "black", "long": "tab:blue", "short": "tab:red", "regime": "tab:green"}
for ax, w in zip(axes.flat, ["W1", "W2", "W3", "W4"]):
    for s in strats:
        d = json.load(open(f"{S}/o3_json/O3_{s}_{w}.json"))
        eq = np.array(d["equity_curve"])
        ax.plot(eq, lw=1.3, label=s, color=colors[s])
    ax.axhline(1.0, color="gray", lw=0.5)
    bh = o3["bh"][w]["net"]
    ax.axhline(1 + bh, color="gray", ls="--", lw=0.8)
    ax.text(0.01, 1 + bh + 0.005, f"B&H {bh:+.1%}", fontsize=8, color="gray")
    ax.set_title(f"{w}  (1x, production exits)")
    ax.set_ylim(0.7, 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Stage O3: 1x strategies vs B&H (per window)")
fig.tight_layout()
fig.savefig(f"{outdir}/o3_equity_curves.png", dpi=110)
print("SAVED", outdir)
