"""O4 严格逐窗口寻优评估（改造自 p0_eval.py，口径完全一致：gate 0.40 / 12h 非重叠块 / 单边 0.07%）
用法:
  o4_eval.py --mode val  --tags AUTO:W1,W2,W3,W4 --out /tmp/O4_val_eval.json
  o4_eval.py --mode test --tags AUTO:W1,W2,W3,W4 --out /tmp/O4_test_eval.json
AUTO:W1 = model/ 下所有 itransformer_BTCUSDT_best_O4*_W1_*.pth 的 exp id
tag 内嵌窗口: O4A_W1_c03 / O4B_W1_c03_1 → window=W1
泄漏纪律: --mode val 只读窗口 val 段（窗口前 90 天，与训练数据 12h purge 隔离）→ 用于选参；
          --mode test 只读窗口 test 段，仅在 val 选参后运行一次。
"""
import os, sys, json, argparse, glob
import numpy as np, pandas as pd, polars as pl
import torch
sys.path.insert(0, os.path.expanduser("~/quant_iTransformer"))
from historical_pipeline.train_itransformer import InvertedTransformer

BASE = os.path.expanduser("~/quant_iTransformer/data/train/processed/main_features_1h/main_features_46d/BTCUSDT")
MODEL_DIR = os.path.expanduser("~/quant_iTransformer/model")
M46L = os.path.join(BASE, "BTCUSDT_Master_Training_Matrix_12h_46d_L_20261030.parquet")
LABS = os.path.join(BASE, "BTCUSDT_L_labels_46d.parquet")
SEQ_LEN = 672
COST_SIDE = 0.0007
COST_SIDE_HI = 0.0014
GATE = 0.40

WINDOWS = {
    "W1": ("2024-03-21 00:00:00", "2024-06-19 00:00:00", "2023-12-21 12:00:00", "2024-03-20 12:00:00"),
    "W2": ("2024-09-17 00:00:00", "2024-12-16 00:00:00", "2024-06-19 12:00:00", "2024-09-16 12:00:00"),
    "W3": ("2025-09-12 00:00:00", "2025-12-11 00:00:00", "2025-06-14 12:00:00", "2025-09-11 12:00:00"),
    "W4": ("2026-07-27 00:00:00", "2026-10-27 12:00:00", "2026-04-28 12:00:00", "2026-07-26 12:00:00"),
}
LAB = "label_A"

print("读取 L 矩阵 ...")
m = pl.read_parquet(M46L)
ts = (m["hour_timestamp"].cast(pl.Datetime("us")).to_numpy().astype("datetime64[us]").astype("int64") // 10**6)
R12 = m["target_12h_return"].to_numpy().astype(float)
label_df = pl.read_parquet(LABS)
labelA = label_df[LAB].to_numpy().astype(float)

def ts_of(s):
    return int(pd.Timestamp(s).timestamp())

# 每窗口预计算 test/val 的 12h 非重叠块网格 + 12h 前向收益
WPRE = {}
for w, (t0, t1, v0, v1) in WINDOWS.items():
    a, b = ts_of(t0), ts_of(t1)
    tidx = np.where((ts >= a) & (ts < b))[0]
    vidx = np.where((ts >= ts_of(v0)) & (ts < ts_of(v1)))[0]
    dec = tidx[(tidx - tidx[0]) % 12 == 0]
    vgrid = vidx[(vidx - vidx[0]) % 12 == 0]
    WPRE[w] = {
        "test_grid": dec, "test_r12": R12[dec], "test_y": labelA[dec],
        "val_grid": vgrid, "val_r12": R12[vgrid], "val_y": labelA[vgrid],
        "bh_test": float(np.expm1(np.sum(R12[dec]))),
    }
    print(f"[{w}] test 块={len(dec)} val 块={len(vgrid)} B&H={WPRE[w]['bh_test']*100:+.2f}%")

def infer(ckpt_path, dec_idx):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = InvertedTransformer(num_variates=ck["num_variates"], seq_len=ck["seq_len"],
                                d_model=ck["d_model"], n_heads=ck["n_heads"], d_ff=ck["d_ff"],
                                e_layers=ck["e_layers"], dropout=ck["dropout"]).eval()
    model.load_state_dict(ck["model_state_dict"])
    cols = ck["feature_cols"]
    Xall = m.select(cols).to_numpy().astype(np.float32)
    out = np.zeros((len(dec_idx), 3), dtype=np.float32)
    B = 256
    with torch.no_grad():
        for s in range(0, len(dec_idx), B):
            chunk = dec_idx[s:s+B]
            win = np.stack([Xall[i-SEQ_LEN+1:i+1] for i in chunk])
            X = torch.from_numpy(win)
            for s2 in range(0, len(X), 512):
                out[s+s2:s+s2+min(512, len(X)-s2)] = torch.softmax(model(X[s2:s2+512]), dim=1).cpu().numpy()
    return out

def trade_stats(pos, r12, cost_side):
    cost = np.where(pos != 0, 2*cost_side, 0.0)
    ret = pos * r12 - cost
    traded = pos != 0
    out = {"n_blocks": int(len(ret)), "n_trades": int(traded.sum())}
    if traded.sum() == 0:
        return out, ret
    tr = ret[traded]
    wins = tr[tr > 0]; losses = tr[tr <= 0]
    W = wins.mean() if len(wins) else 0.0
    L = abs(losses.mean()) if len(losses) else 0.0
    C = 2*cost_side
    p = len(wins)/len(tr)
    eq = np.cumprod(1+ret)
    total = float(eq[-1] - 1)
    years = (len(ret) * 12) / (365*24)
    ann = (1+total)**(1/years) - 1 if years > 0.05 and total > -1 else float("nan")
    sharpe = (ret.mean()/ret.std()*np.sqrt(730)) if ret.std() > 0 else 0.0
    dn = ret[ret < 0]
    sortino = (ret.mean()/(dn.std()*np.sqrt(len(dn)/len(ret)))*np.sqrt(730)) if len(dn) > 1 else float("nan")
    gp = tr[tr > 0].sum(); gl = abs(tr[tr <= 0].sum())
    out.update({
        "ret_total": total, "ret_ann": float(ann) if np.isfinite(ann) else None,
        "win_rate": float(p),
        "avg_win": float(W), "avg_loss": float(L),
        "payoff_ratio": float(W/L) if L > 0 else float("inf"),
        "profit_factor": float(gp/gl) if gl > 0 else float("inf"),
        "breakeven_E": float(p*W - (1-p)*L - C),
        "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd": float(np.max(1 - eq/np.maximum.accumulate(eq))),
        "long_ret": float(ret[pos == 1].sum()), "short_ret": float(ret[pos == -1].sum()),
        "long_n": int((pos == 1).sum()), "short_n": int((pos == -1).sum()),
        "exposure": float(traded.mean()),
    })
    return out, ret

def bootstrap_ci(ret, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ret), size=(n, len(ret)))
    tot = ret[idx].sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["val", "test"])
ap.add_argument("--tags", required=True, help="逗号分隔 tag，或 AUTO:W1,W2 自动发现")
ap.add_argument("--out", required=True)
args = ap.parse_args()

tags = []
spec = args.tags.strip()
if spec.startswith("AUTO:"):
    for w in spec[5:].split(","):
        w = w.strip()
        if not w:
            continue
        fs = glob.glob(os.path.join(MODEL_DIR, f"itransformer_BTCUSDT_best_O4*_{w}_*.pth"))
        for f in sorted(fs):
            t = os.path.basename(f)[len("itransformer_BTCUSDT_best_"):-len(".pth")]
            if t not in tags:
                tags.append(t)
else:
    for part in spec.split(","):
        if part.strip():
            tags.append(part.strip())

results = {}
for tag in tags:
    w = tag.split("_")[1]
    if w not in WPRE:
        print(f"!! 未知窗口: {tag}"); continue
    ckpt = os.path.join(MODEL_DIR, f"itransformer_BTCUSDT_best_{tag}.pth")
    if not os.path.exists(ckpt):
        print(f"!! 缺 {tag}"); continue
    if args.mode == "val":
        grid, r12, y = WPRE[w]["val_grid"], WPRE[w]["val_r12"], WPRE[w]["val_y"]
    else:
        grid, r12, y = WPRE[w]["test_grid"], WPRE[w]["test_r12"], WPRE[w]["test_y"]
    probs = infer(ckpt, grid)
    valid = ~np.isnan(y)
    ic = pd.Series(probs[valid,2]-probs[valid,0]).corr(pd.Series(r12[valid]), method="spearman") if valid.sum() > 10 else float("nan")
    pos = np.zeros(len(grid), dtype=int)
    pos[(probs[:,2] >= GATE) & (probs[:,2] > probs[:,0])] = 1
    pos[(probs[:,0] >= GATE) & (probs[:,0] > probs[:,2])] = -1
    st, ret = trade_stats(pos, r12, COST_SIDE)
    st_hi, ret_hi = trade_stats(pos, r12, COST_SIDE_HI)
    ci_lo, ci_hi = bootstrap_ci(ret)
    results[tag] = {"window": w, "mode": args.mode, "rank_ic": float(ic),
                    **{k: v for k, v in st.items()},
                    "ret_total_hi_cost": float(np.prod(1+ret_hi)-1), "breakeven_E_hi": st_hi.get("breakeven_E"),
                    "boot_ci": [ci_lo, ci_hi]}
    print(f"[{args.mode}][{w}] {tag}: n={st.get('n_trades',0)} ret={st.get('ret_total',0)*100:+.2f}% "
          f"PF={st.get('profit_factor',0):.2f} E={st.get('breakeven_E',0)*100:+.4f}% IC={ic:+.4f}")

with open(args.out, "w") as f:
    json.dump({"mode": args.mode, "per_tag": results,
               "bh": {w: WPRE[w]["bh_test"] for w in WPRE}}, f, ensure_ascii=False, indent=1, default=float)
print(f"SAVED {args.out}")
print(f"DONE_O4_EVAL_{args.mode.upper()}")
