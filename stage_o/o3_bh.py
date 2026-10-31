"""O3 基准: 每窗口 B&H（raw 1h close, 窗口起点 close → 终点 close）"""
import json
import numpy as np, polars as pl, pandas as pd

df = pl.read_parquet("/home/ubuntu/quant_iTransformer/data/train/raw_temp/klines/BTCUSDT/BTCUSDT_raw_1h.parquet").sort("timestamp")
ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
c = df["close"].to_numpy().astype(float)
windows = {
    "W1": ("2024-03-21 00:00:00", "2024-06-19 00:00:00"),
    "W2": ("2024-09-17 00:00:00", "2024-12-16 00:00:00"),
    "W3": ("2025-09-12 00:00:00", "2025-12-11 00:00:00"),
    "W4": ("2026-07-27 00:00:00", "2026-10-27 12:00:00"),
}
out = {}
for w, (a, b) in windows.items():
    ta, tb = int(pd.Timestamp(a).timestamp()), int(pd.Timestamp(b).timestamp())
    i0 = int(np.searchsorted(ts, ta)); i1 = int(np.searchsorted(ts, tb, side="right")) - 1
    out[w] = float(c[i1] / c[i0] - 1.0)
    print(w, f"{pd.Timestamp(ts[i0], unit='s')} → {pd.Timestamp(ts[i1], unit='s')}", f"{out[w]:+.4%}")
json.dump(out, open("/tmp/o3_bh.json", "w"))
print("SAVED /tmp/o3_bh.json")
