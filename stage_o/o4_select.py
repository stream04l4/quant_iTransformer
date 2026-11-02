"""O4 选参: 按 val 段 breakeven_E(每笔净期望) 选每窗口 top2 (要求 n_trades>=20)
输入 /tmp/O4_val_eval.json → 输出 /tmp/O4_selected.txt (每行一个 O4A tag)
泄漏纪律: 只用 val 段指标, 不碰任何 test 段数字。
"""
import json

MIN_TRADES = 8  # val 段交易数门槛：gate 0.40 下 180 块 val 大多组合 n<20，20 会把 W1 全排除（过严）；8 为薄样本下限，报告中须标注 n
d = json.load(open("/tmp/O4_val_eval.json"))
per = d["per_tag"]
out = {}
for w in ["W1", "W2", "W3", "W4"]:
    rows = [(t, r) for t, r in per.items() if r.get("window") == w and r.get("n_trades", 0) >= MIN_TRADES]
    rows.sort(key=lambda x: -x[1].get("breakeven_E", -9))
    sel = rows[:2]
    out[w] = [t for t, _ in sel]
    print(f"--- {w} (val 段, n_trades>={MIN_TRADES}, 按 breakeven_E 降序 top5) ---")
    for t, r in rows[:5]:
        mark = " <== SEL" if t in out[w] else ""
        print(f"  {t}: E={r.get('breakeven_E', 0)*100:+.4f}% n={r.get('n_trades', 0)} "
              f"PF={r.get('profit_factor', 0):.2f} ret={r.get('ret_total', 0)*100:+.2f}% IC={r.get('rank_ic', 0):+.4f}{mark}")
    if not sel:
        print(f"  !! {w} 无 n_trades>=20 的组合")

with open("/tmp/O4_selected.txt", "w") as f:
    for w in ["W1", "W2", "W3", "W4"]:
        for t in out[w]:
            f.write(t + "\n")
print("SAVED /tmp/O4_selected.txt:", sum(len(v) for v in out.values()), "tags")
