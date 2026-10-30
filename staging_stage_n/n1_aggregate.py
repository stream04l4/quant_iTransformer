"""N1 S1 出场消融聚合（纯 stdlib+numpy，无 pandas 依赖）
决策门: G1 S1-B/C 显著差于 S1-A → 出场机制确认瓶颈; G2 三种全 ≤0 → 升级拍板
"""
import json, glob, os, statistics as st
import numpy as np

D = os.path.expanduser("~/.hermes/joint_project/btc-feature-v2/n1_runs")
rows = []
for f in sorted(glob.glob(os.path.join(D, "N1_*.json"))):
    base = os.path.basename(f)[:-5]
    v, w, s, mode = base.split("_")[1], base.split("_")[2], base.split("_")[3], "_".join(base.split("_")[4:])
    r = json.load(open(f))
    rows.append({"variant": v, "window": w, "seed": int(s), "exit_mode": mode, **{
        "ret": r["total_return"], "n": r["n_trades"], "wr": r["win_rate"],
        "pf": r["profit_factor"] or 0.0, "exp": r["expectancy"], "maxdd": r["max_dd"],
        "mfe": r["mfe_median"] or 0.0, "mae": r["mae_median"] or 0.0,
        "reasons": r["exit_reasons"], "long_n": r["long_trades"], "short_n": r["short_trades"]}})
print(f"共 {len(rows)} 路（预期 144）")
modes = ["fixed12h", "no_sl", "production"]
mname = {"fixed12h": "S1-A 固定12h", "no_sl": "S1-C 无SL", "production": "S1-B 生产规则"}

def med(vals): return st.median(vals) if vals else float("nan")

print("\n=== 39A 主信号：累计收益（6-seed 中位数）===")
print(f"{'窗口':<4} " + " ".join(f"{mname[m]:>12}" for m in modes))
for w in ["W1", "W2", "W3", "W4"]:
    line = f"{w:<4} "
    for m in modes:
        vals = [r["ret"] for r in rows if r["variant"] == "39A" and r["window"] == w and r["exit_mode"] == m]
        line += f"{med(vals)*100:>11.2f}% "
    print(line)
gline = "全局 "
for m in modes:
    vals = [r["ret"] for r in rows if r["variant"] == "39A" and r["exit_mode"] == m]
    gline += f"{med(vals)*100:>11.2f}% "
print(gline)

print("\n=== 39A 全局指标（24 路中位数）===")
print(f"{'指标':<10} " + " ".join(f"{mname[m]:>12}" for m in modes))
for key, name, pct in [("n", "笔数", 0), ("wr", "胜率", 1), ("exp", "E/笔", 1), ("maxdd", "最大回撤", 1), ("mfe", "MFE中位", 1), ("mae", "MAE中位", 1)]:
    line = f"{name:<10} "
    for m in modes:
        vals = [r[key] for r in rows if r["variant"] == "39A" and r["exit_mode"] == m]
        line += f"{med(vals)*100 if pct else med(vals):>11.2f}{'%' if pct else ''} "
    print(line)

print("\n=== 31A '无优势'对照组（三模式都应 ≈ -成本 ≈ -0.2~-0.5%/窗口）===")
for m in modes:
    vals = [r["ret"] for r in rows if r["variant"] == "31A" and r["exit_mode"] == m]
    nmed = [r["n"] for r in rows if r["variant"] == "31A" and r["exit_mode"] == m]
    print(f"  {mname[m]:<14} 累计中位 {med(vals)*100:+.2f}%  笔数中位 {med(nmed):.0f}  正收益路数 {sum(1 for x in vals if x > 0)}/24")

prod = [r for r in rows if r["variant"] == "39A" and r["exit_mode"] == "production"]
reasons = {}
for r in prod:
    for k, v2 in r["reasons"].items():
        reasons[k] = reasons.get(k, 0) + v2
tot = sum(reasons.values()) or 1
print("\n=== 39A S1-B(生产规则) 退出原因分布 ===")
for k in ["TIME_LIMIT_EXIT", "TRAILING_EXIT", "HARD_SL_EXIT", "REVERSAL_EXIT", "LIQUIDATION_EXIT"]:
    print(f"  {k:<18} {reasons.get(k,0):6d}  {reasons.get(k,0)/tot*100:5.1f}%")

f12 = [r for r in rows if r["variant"] == "39A" and r["exit_mode"] == "fixed12h"]
ns = [r for r in rows if r["variant"] == "39A" and r["exit_mode"] == "no_sl"]
pr = prod
print(f"\n=== 决策门 ===")
a, b, c = med([r["ret"] for r in f12]), med([r["ret"] for r in pr]), med([r["ret"] for r in ns])
print(f"39A: S1-A 固定12h {a*100:+.2f}% | S1-B 生产规则 {b*100:+.2f}% | S1-C 无SL {c*100:+.2f}% (均为 24 路中位数, 0.07% 单边)")
allneg = all(x < 0 for x in (a, b, c))
print(f"G2 判定（三种全≤0）: {'触发' if allneg else '未触发'}")
if a > 0 and (b < a - 0.01 or c < a - 0.01):
    print(f"G1 判定（B/C 显著差于 A）: 触发（A 正且 {'B' if b < a-0.01 else ''}{'C' if c < a-0.01 else ''} 落后 >1pp）")
else:
    print("G1 判定: 未触发")
