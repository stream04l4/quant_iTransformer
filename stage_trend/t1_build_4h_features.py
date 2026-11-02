"""P1 脚本2: 4h 31d 特征矩阵（按周期重算，因果对齐）
周期重算约定（C1 报告 §因果审计 将逐条列出）:
  - bar 计数型指标（RSI 14/48, EMA 12/26/9/200, BB 20）→ 4h bar 计数不变
  - 小时窗口型特征（24h/72h/168h）→ 保持小时: 24h=6bars, 72h=18bars, 168h=42bars
  - robust z 窗口 168(120) → 42(30) bars
  - RV 保留 sub-bar 精度: RV_4h = sqrt(Σ ret_1m²)（1m 数据，对齐 1h 管道定义）
  - flow 为可加量: 4h = 4×1h 原始流量求和; retail_ratio/whale_ls_ratio 在 4h 重算
  - funding: 8h 事件 ffill 覆盖 2 个 4h bar; momentum = mean(6) − mean(18)
因果: 特征[t] 只依赖 bar_start ≤ t 的数据（rolling 均含当前已收盘 bar, 无 shift 的未来项）。
"""
import os, glob, json, time
from pathlib import Path
import polars as pl
import numpy as np

BASE = Path(os.path.expanduser("~/quant_iTransformer/data/train"))
OUT = BASE / "processed" / "trend_4h" / "BTCUSDT"
t0 = time.time()

def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def rsi_expr(period: int) -> pl.Expr:
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0).ewm_mean(alpha=1.0/period, adjust=False)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).ewm_mean(alpha=1.0/period, adjust=False)
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))

# 数据
log("读 bars ...")
b = pl.read_parquet(OUT / "btc_4h_bars.parquet")
eth4 = pl.read_parquet(OUT / "eth_4h_bars.parquet").select("bar_start", pl.col("close").alias("eth_close"))
flow = pl.read_parquet(OUT / "flow_4h.parquet")
oi4 = pl.read_parquet(OUT / "oi_4h.parquet").select(["bar_start", "oi_delta_pct_4h"])
fund = pl.read_parquet(OUT / "funding_4h.parquet")
prem = pl.read_parquet(OUT / "premium_4h.parquet").select(["bar_start", "premium_delta_4h"])

df = b.join(eth4, on="bar_start", how="left") \
      .join(flow.select(["bar_start", "cvd_4h", "whale_net_flow_4h", "whale_ls_ratio_4h", "retail_ratio_4h"]), on="bar_start", how="left") \
      .join(oi4, on="bar_start", how="left") \
      .join(fund, on="bar_start", how="left") \
      .join(prem, on="bar_start", how="left").sort("bar_start")

# RV_4h (1m sub-bar)
log("读 1m 数据算 RV_4h ...")
p1m = BASE / "raw_temp/klines/BTCUSDT/BTCUSDT_raw_1m.parquet"
if p1m.exists():
    m1 = pl.read_parquet(p1m)
else:
    parts = []
    for f in sorted(glob.glob(str(BASE / "raw_temp/klines/BTCUSDT/1m/*/*.parquet"))) + \
            sorted(glob.glob(str(BASE / "raw_temp/klines/BTCUSDT/1m/*/*.csv"))):
        if f.endswith(".parquet"):
            d = pl.read_parquet(f)
        else:
            d = pl.read_csv(f, has_header=False)
            d = d.rename({d.columns[i]: n for i, n in enumerate(
                ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])})
            _ot = pl.col("open_time").cast(pl.Utf8).str.to_integer(strict=False)
            d = d.with_columns(pl.when(_ot >= 1e14).then(_ot // 1000).otherwise(_ot).alias("_ms"))
            d = d.select(pl.from_epoch(pl.col("_ms"), time_unit="ms").alias("timestamp"), pl.col("close").cast(pl.Float64, strict=False))
        parts.append(d)
    m1 = pl.concat(parts, how="diagonal_relaxed").drop_nulls(["timestamp", "close"]).unique("timestamp").sort("timestamp")
rv = m1.with_columns(
    (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1m"),
    pl.col("timestamp").dt.truncate("4h").alias("bar_start"),
).group_by("bar_start").agg(pl.col("ret_1m").pow(2).sum().sqrt().alias("realized_vol_4h"))
df = df.join(rv, on="bar_start", how="left")
log(f"joined: {df.height}")

# funding ffill: 8h 事件 → 覆盖 2 个 4h bar
df = df.with_columns(pl.col("funding_cum_4h").fill_null(strategy="forward", limit=2))

# ============ 特征 ============
log("算 31d 特征 ...")
H = {"h24": 6, "h72": 18, "h168": 42}
df = df.with_columns(
    (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
)
df = df.with_columns([
    pl.col("log_return").rolling_max(window_size=H["h24"]).alias("max_ret_24h"),
    (pl.col("close") / pl.col("close").ewm_mean(span=200, adjust=False) - 1.0).alias("ma_bias"),
    ((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("upper_shadow"),
    ((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("lower_shadow"),
    (pl.col("close").rolling_std(window_size=20) * 4 / (pl.col("close").rolling_mean(window_size=20) + 1e-9)).alias("bb_squeeze"),
    (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-9)).diff(1).alias("taker_ratio_delta"),
])
# 标签: 24h 前向 log 收益 (6 bars) + 48h/72h 诊断
df = df.with_columns([
    (pl.col("close").shift(-6).log() - pl.col("close").log()).alias("fwd_ret_24h"),
    (pl.col("close").shift(-12).log() - pl.col("close").log()).alias("fwd_ret_48h"),
    (pl.col("close").shift(-18).log() - pl.col("close").log()).alias("fwd_ret_72h"),
])
ppo = (pl.col("close").ewm_mean(span=12, adjust=False) - pl.col("close").ewm_mean(span=26, adjust=False)) / (pl.col("close").ewm_mean(span=26, adjust=False) + 1e-9)
m1x = pl.col("log_return").rolling_mean(window_size=H["h24"])
m2x = pl.col("log_return").pow(2).rolling_mean(window_size=H["h24"])
m3x = pl.col("log_return").pow(3).rolling_mean(window_size=H["h24"])
var = (m2x - m1x.pow(2)).abs()
skew = (m3x - 3 * m1x * m2x + 2 * m1x.pow(3)) / (var.pow(1.5) + 1e-15)
df = df.with_columns([
    (ppo - ppo.ewm_mean(span=9, adjust=False)).alias("ppo_hist"),
    ((rsi_expr(14) - 50) / 50.0).alias("rsi_14"),
    ((rsi_expr(48) - 50) / 50.0).alias("rsi_48"),
    skew.alias("vol_skew_24h"),
    (pl.col("realized_vol_4h") / (pl.col("realized_vol_4h").rolling_mean(window_size=H["h24"]) + 1e-9)).clip(0.0, 50.0).alias("rv_term_structure"),
    (pl.col("funding_cum_4h").rolling_mean(window_size=H["h24"]) - pl.col("funding_cum_4h").rolling_mean(window_size=H["h72"])).alias("funding_rate_momentum"),
    (pl.col("eth_close") / (pl.col("close") + 1e-9)).alias("_xratio"),
])
df = df.with_columns([
    (pl.col("_xratio").log() - pl.col("_xratio").log().shift(1)).alias("cross_asset_return"),
    ((pl.col("close") - pl.col("low").rolling_min(window_size=H["h24"])) / (pl.col("high").rolling_max(window_size=H["h24"]) - pl.col("low").rolling_min(window_size=H["h24"]) + 1e-9)).alias("price_pos_24h"),
    ((pl.col("close") - pl.col("low").rolling_min(window_size=H["h72"])) / (pl.col("high").rolling_max(window_size=H["h72"]) - pl.col("low").rolling_min(window_size=H["h72"]) + 1e-9)).alias("price_pos_72h"),
    ((pl.col("close") - pl.col("low").rolling_min(window_size=H["h168"])) / (pl.col("high").rolling_max(window_size=H["h168"]) - pl.col("low").rolling_min(window_size=H["h168"]) + 1e-9)).alias("price_pos_168h"),
    (pl.col("bar_start").dt.hour() * (2 * np.pi / 24)).sin().alias("hour_sin"),
    (pl.col("bar_start").dt.hour() * (2 * np.pi / 24)).cos().alias("hour_cos"),
    (pl.col("bar_start").dt.weekday() * (2 * np.pi / 7)).sin().alias("week_sin"),
    (pl.col("bar_start").dt.weekday() * (2 * np.pi / 7)).cos().alias("week_cos"),
    (pl.col("log_return").abs() / (pl.col("quote_volume") + 1e-9) + 1e-12).log().clip(-50.0, 50.0).alias("_amihud"),
]).drop("_xratio")

# robust z (168h=42 bars, min 120h=30), 对齐 1h 管道 (shift(1) 严格事前)
def rz(src, alias, W=42, min_n=30):
    med = pl.col(src).rolling_median(W, min_samples=min_n).shift(1).alias(f"__m_{alias}")
    mad = (pl.col(src) - pl.col(f"__m_{alias}")).abs().rolling_median(W, min_samples=min_n).shift(1).alias(f"__d_{alias}")
    return med, mad, (0.6745 * (pl.col(src) - pl.col(f"__m_{alias}")) / (pl.col(f"__d_{alias}") + 1e-9)).clip(-4.0, 4.0).alias(alias)

jobs = [("realized_vol_4h", "realized_vol_4h_zscore"), ("_amihud", "amihud_illiquidity_zscore"),
        ("oi_delta_pct_4h", "oi_delta_pct_zscore"), ("cvd_4h", "cvd_zscore"),
        ("whale_net_flow_4h", "whale_net_flow_zscore"), ("whale_ls_ratio_4h", "whale_ls_ratio_zscore")]
W, min_n = 42, 30  # 168h=42bars, 120h=30bars
meds, mads, zexprs = [], [], []
for src, al in jobs:
    meds.append(pl.col(src).rolling_median(W, min_samples=min_n).shift(1).alias(f"__m_{al}"))
    mads.append((pl.col(src) - pl.col(f"__m_{al}")).abs().rolling_median(W, min_samples=min_n).shift(1).alias(f"__d_{al}"))
    zexprs.append((0.6745 * (pl.col(src) - pl.col(f"__m_{al}")) / (pl.col(f"__d_{al}") + 1e-9)).clip(-4.0, 4.0).alias(al))
df = df.with_columns(meds)
df = df.with_columns(mads)
df = df.with_columns(zexprs)
df = df.with_columns((pl.col("cvd_zscore") - pl.col("oi_delta_pct_zscore")).alias("cvd_oi_divergence"))
tmp_names = [f"__m_{al}" for _, al in jobs] + [f"__d_{al}" for _, al in jobs]
df = df.drop(tmp_names + ["_amihud", "oi_delta_pct_zscore"])

final_cols = ["bar_start", "is_synthetic_flag",
              "log_return", "max_ret_24h", "ma_bias", "upper_shadow", "lower_shadow",
              "bb_squeeze", "ppo_hist", "rsi_14", "rsi_48", "taker_ratio_delta",
              "amihud_illiquidity_zscore", "realized_vol_4h_zscore", "vol_skew_24h",
              "oi_delta_pct_4h", "price_pos_24h", "price_pos_72h", "price_pos_168h",
              "rv_term_structure", "funding_rate_momentum", "premium_delta_4h", "cross_asset_return",
              "hour_sin", "hour_cos", "week_sin", "week_cos",
              "cvd_zscore", "whale_net_flow_zscore", "whale_ls_ratio_zscore", "retail_ratio_4h",
              "cvd_oi_divergence", "fwd_ret_24h", "fwd_ret_48h", "fwd_ret_72h"]
df = df.select(final_cols).rename({"is_synthetic_flag": "is_synthetic", "oi_delta_pct_4h": "oi_delta_pct",
                                    "premium_delta_4h": "premium_delta", "retail_ratio_4h": "retail_ratio"})
# 清洗: NaN/Inf → null
float_cols = [c for c, dt in df.schema.items() if dt == pl.Float64]
df = df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
df = df.with_columns([pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in float_cols])

null_report = df.null_count()
null_report = {}
for _r in df.null_count().to_dicts():
    null_report.update(_r)
n_feat = df.shape[0]
qc = {
    "n_rows": int(df.height),
    "range": [str(df["bar_start"][0]), str(df["bar_start"][-1])],
    "nulls": {k: int(v) for k, v in null_report.items() if v > 0},
    "null_pct": {k: round(int(v) / df.height, 4) for k, v in null_report.items() if v > 0},
}
df.write_parquet(OUT / "btc_4h_features_31d.parquet")
qc["elapsed_s"] = round(time.time() - t0, 1)
with open(OUT / "p1_feature_qc.json", "w") as f:
    json.dump(qc, f, indent=1, default=str)
print("DONE_T1_FEATURES")
print(json.dumps(qc, indent=1, default=str))
