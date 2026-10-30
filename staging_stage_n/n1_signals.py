"""Stage N 信号提取：P0 checkpoint → 窗口内逐小时概率 + 严格 val 阈值
推理口径与 p0_eval.py 完全一致（Xall[i-SEQ_LEN+1:i+1]，含行 i；CPU softmax）
阈值来源 p0_eval_results.json（S5 严格 val 段分位，top10 供 N1 引擎反转/刷新检查）
输出: ~/quant_iTransformer/data/train/n1_signals/sig_{variant}_{window}_{seed}.csv
     列: hour_timestamp, p_down, p_up, th_up_top10, th_down_top10
"""
import os, sys, json
import numpy as np, pandas as pd, polars as pl, torch
sys.path.insert(0, os.path.expanduser("~/quant_iTransformer"))
from historical_pipeline.train_itransformer import InvertedTransformer

BASE = os.path.expanduser("~/quant_iTransformer/data/train/processed/main_features_1h/main_features_46d/BTCUSDT")
M46L = os.path.join(BASE, "BTCUSDT_Master_Training_Matrix_12h_46d_L_20261030.parquet")
MODEL_DIR = os.path.expanduser("~/quant_iTransformer/model")
SEQ_LEN = 672
WINDOWS = {
    "W1": ("2024-03-21 00:00:00", "2024-06-19 00:00:00"),
    "W2": ("2024-09-17 00:00:00", "2024-12-16 00:00:00"),
    "W3": ("2025-09-12 00:00:00", "2025-12-11 00:00:00"),
    "W4": ("2026-07-27 00:00:00", "2026-10-27 12:00:00"),
}
SEEDS = [42, 1, 7, 123, 1729, 999]
VARIANTS = ["39A", "31A", "39C"]

p0 = json.load(open("/tmp/p0_eval_results.json"))
per_run = p0["per_run"]

print("读取 46d_L 矩阵 ...")
m = pl.read_parquet(M46L)
ts = m["hour_timestamp"].cast(pl.Datetime("us")).to_numpy().astype("datetime64[us]").astype("int64") // 10**6
Xall_cache = {}

def infer(ckpt_path, idx):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = InvertedTransformer(num_variates=ck["num_variates"], seq_len=ck["seq_len"],
                                d_model=ck["d_model"], n_heads=ck["n_heads"], d_ff=ck["d_ff"],
                                e_layers=ck["e_layers"], dropout=ck["dropout"]).eval()
    model.load_state_dict(ck["model_state_dict"])
    cols = tuple(ck["feature_cols"])
    if cols not in Xall_cache:
        Xall_cache[cols] = m.select(list(cols)).to_numpy().astype(np.float32)
    Xall = Xall_cache[cols]
    out = np.zeros((len(idx), 3), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            chunk = idx[s:s+256]
            win = np.stack([Xall[i-SEQ_LEN+1:i+1] for i in chunk])
            X = torch.from_numpy(win)
            for s2 in range(0, len(X), 512):
                out[s+s2:s+s2+min(512, len(X)-s2)] = torch.softmax(model(X[s2:s2+512]), dim=1).cpu().numpy()
    return out

OUT = os.path.expanduser("~/quant_iTransformer/data/train/n1_signals")
os.makedirs(OUT, exist_ok=True)
done = 0
for w, (t0, t1) in WINDOWS.items():
    a, b = int(pd.Timestamp(t0).timestamp()), int(pd.Timestamp(t1).timestamp())
    tidx = np.where((ts >= a) & (ts < b))[0]
    for v in VARIANTS:
        for seed in SEEDS:
            tag = f"P0{w}_{v}_{seed}"
            ckpt = os.path.join(MODEL_DIR, f"itransformer_BTCUSDT_best_{tag}.pth")
            if not os.path.exists(ckpt):
                print(f"!! 缺 {tag}")
                continue
            run = per_run[f"{w}/{v}"][str(seed)]
            probs = infer(ckpt, tidx)
            df = pd.DataFrame({
                "hour_timestamp": pd.to_datetime(ts[tidx], unit="s"),
                "p_down": probs[:, 0].astype(float),
                "p_up": probs[:, 2].astype(float),
                "th_up_top10": float(run["val_th_up"]["top10"]),
                "th_down_top10": float(run["val_th_down"]["top10"]),
            })
            path = os.path.join(OUT, f"sig_{v}_{w}_{seed}.csv")
            df.to_csv(path, index=False)
            done += 1
            print(f"{tag}: {len(df)} rows")
print(f"DONE_N1_SIGNALS ({done}/72)")
