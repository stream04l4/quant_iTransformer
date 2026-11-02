"""P1 脚本1: 4h/1d bar 构建（1h 原始源 → 4h/1d，因果对齐）
输入（otcca ~/quant_iTransformer/data/train）:
  raw_temp/klines/BTCUSDT/BTCUSDT_raw_1h.parquet, ETHUSDT 同
  processed/aggtrades_1h/BTCUSDT/BTCUSDT_aggTrades_features_YYYY_MM.parquet (+daily 兜底)
  raw_temp/metrics/BTCUSDT/<day>/BTCUSDT-metrics-<day>.csv (5min OI/toptrader)
  raw_temp/fundingRate/BTCUSDT/YYYY_MM/*.csv (8h)
  raw_temp/premiumIndexKlines/BTCUSDT/YYYY_MM/*.csv (1h)
输出 data/train/processed/trend_4h/BTCUSDT/:
  btc_4h_bars.parquet, btc_1d_bars.parquet, eth_4h_bars.parquet,
  flow_4h.parquet, oi_4h.parquet, funding_4h.parquet, premium_4h.parquet,
  p1_qc.json
因果规则: 4h bar(bar_start=t) 只含 [t, t+4h) 的 1h bar；信息可用时点 = t+4h。
flow 为可加量: 4h 值 = 4 个 1h 原始流量求和; retail_ratio/whale_ls_ratio 在 4h 上重算(不聚合 1h 比值)。
"""
import os, glob, json, time
from pathlib import Path
import polars as pl
import numpy as np

BASE = Path(os.path.expanduser("~/quant_iTransformer/data/train"))
OUT = BASE / "processed" / "trend_4h" / "BTCUSDT"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
qc = {}

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ============ 1. BTC/ETH 4h & 1d bars ============
def resample_bars(p1h: pl.DataFrame, factor: int, name: str) -> pl.DataFrame:
    df = p1h.with_columns(
        pl.col("timestamp").dt.truncate(f"{factor}h").alias("bar_start"),
        pl.col("timestamp").alias("h_ts"),
    )
    g = df.group_by("bar_start").agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("quote_volume").sum().alias("quote_volume"),
        pl.col("taker_buy_volume").sum().alias("taker_buy_volume"),
        pl.col("h_ts").count().alias("n_1h"),
        pl.col("h_ts").min().alias("first_1h"),
        pl.col("h_ts").max().alias("last_1h"),
    ).sort("bar_start")
    # 完整性: bar_start + (factor-1)h 应有 1h bar
    g = g.with_columns(((pl.col("last_1h") - pl.col("bar_start")).dt.total_hours() == factor - 1).alias("complete"))
    # 时间连续性: bar_start 间隔应为 factor h
    g = g.with_columns((pl.col("bar_start").diff().dt.total_hours().fill_null(0) == factor).alias("contiguous"))
    g = g.with_columns(((~pl.col("complete")) | (~pl.col("contiguous"))).cast(pl.Int8).alias("is_synthetic_flag"))
    qc[name] = {
        "n_bars": int(g.height), "n_incomplete": int((~g["complete"]).sum()),
        "n_gap": int((~g["contiguous"]).sum()),
        "range": [str(g["bar_start"][0]), str(g["bar_start"][-1])],
    }
    return g

log("读取 BTC 1h K线 ...")
btc1h = pl.read_parquet(BASE / "raw_temp/klines/BTCUSDT/BTCUSDT_raw_1h.parquet")
log("读取 ETH 1h K线 (月度 CSV) ...")
eth_parts = []
for f in sorted(glob.glob(str(BASE / "raw_temp/klines/ETHUSDT/1h/*/*.csv"))):
    d = pl.read_csv(f, has_header=False)
    _names = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume"]
    d = d.rename({d.columns[i]: _names[i] for i in range(min(len(_names), len(d.columns)))})
    _ot = pl.col("open_time").cast(pl.Utf8).str.to_integer(strict=False)
    # 数据源 2025-01 起 epoch 单位 ms→µs 漂移: 按量纲自动归一 (strict=False 容表头行, 后丢 null)
    d = d.with_columns(pl.when(_ot >= 1e14).then(_ot // 1000).otherwise(_ot).alias("_ot_ms"))
    eth_parts.append(d.select(pl.from_epoch(pl.col("_ot_ms"), time_unit="ms").alias("timestamp"),
                              pl.col("open", "high", "low", "close"),
                              pl.col("volume", "quote_volume"), pl.col("count"),
                              pl.col("taker_buy_volume").cast(pl.Float64)))
eth1h = pl.concat(eth_parts, how="diagonal_relaxed").drop_nulls("timestamp").unique("timestamp").sort("timestamp")
btc4h = resample_bars(btc1h, 4, "btc_4h")
btc1d = resample_bars(btc1h, 24, "btc_1d")
eth4h = resample_bars(eth1h, 4, "eth_4h")
btc4h.write_parquet(OUT / "btc_4h_bars.parquet")
btc1d.write_parquet(OUT / "btc_1d_bars.parquet")
eth4h.write_parquet(OUT / "eth_4h_bars.parquet")
log(f"bars done: btc4h={btc4h.height} btc1d={btc1d.height} eth4h={eth4h.height}")

# ============ 2. 订单流 (可加量求和; 比值在 4h 重算) ============
log("读取 aggtrades 1h flow ...")
flow_files = sorted(glob.glob(str(BASE / "processed/aggtrades_1h/BTCUSDT/BTCUSDT_aggTrades_features_*.parquet")))
fd = pl.concat([pl.read_parquet(f) for f in flow_files], how="diagonal_relaxed")
fd = fd.unique("hour_timestamp", keep="first").sort("hour_timestamp")
fd = fd.with_columns(pl.col("hour_timestamp").dt.truncate("4h").alias("bar_start"))
f4 = fd.group_by("bar_start").agg(
    pl.col("cvd").sum().alias("cvd_4h"),
    pl.col("whale_buy").sum().alias("whale_buy_4h"),
    pl.col("whale_sell").sum().alias("whale_sell_4h"),
    pl.col("whale_net_flow").sum().alias("whale_net_flow_4h"),
    pl.col("retail_vol").sum().alias("retail_vol_4h"),
    pl.col("hour_timestamp").count().alias("n_1h"),
)
f4 = f4.sort("bar_start").join(btc4h.select(["bar_start", "volume"]).rename({"volume": "total_vol_4h"}), on="bar_start", how="left")
f4 = f4.with_columns(
    (pl.col("retail_vol_4h") / pl.col("total_vol_4h")).alias("retail_ratio_4h"),
    (pl.col("whale_buy_4h") / pl.col("whale_sell_4h")).alias("whale_ls_ratio_4h"),
)
qc["flow_4h"] = {"n_bars": int(f4.height), "n_missing_join": int(f4["total_vol_4h"].is_null().sum())}
f4.write_parquet(OUT / "flow_4h.parquet")
log(f"flow done: {f4.height}")

# ============ 3. OI (5min → bar 边界值) ============
log("读取 5min OI metrics ...")
oi_rows = []
for day in sorted(os.listdir(BASE / "raw_temp/metrics/BTCUSDT"))[:4000]:
    p = BASE / "raw_temp/metrics/BTCUSDT" / day / f"BTCUSDT-metrics-{day}.csv"
    if p.exists():
        try:
            oi_rows.append(pl.read_csv(p).select(["create_time", "sum_open_interest", "sum_taker_long_short_vol_ratio"]))
        except Exception:
            pass
oi = pl.concat(oi_rows, how="diagonal_relaxed")
oi = oi.with_columns(pl.col("create_time").cast(pl.Utf8).str.to_datetime(strict=False).alias("ts")).drop("create_time").drop_nulls("ts")
oi = oi.sort("ts").with_columns(pl.col("ts").dt.truncate("4h").alias("bar_start"))
oi4 = oi.group_by("bar_start").agg(
    pl.col("sum_open_interest").last().alias("oi_4h"),
    pl.col("sum_taker_long_short_vol_ratio").last().alias("taker_ls_ratio_4h"),
    pl.col("ts").count().alias("n_5min"),
).sort("bar_start")
oi4 = oi4.with_columns((pl.col("oi_4h").diff() / pl.col("oi_4h").shift(1)).clip(-1.0, 1.0).alias("oi_delta_pct_4h"))  # 对齐 1h: 带符号, clip ±1
qc["oi_4h"] = {"n_bars": int(oi4.height), "n_gap5min": int((oi4["n_5min"] < 12).sum())}
oi4.write_parquet(OUT / "oi_4h.parquet")
log(f"oi done: {oi4.height}")

# ============ 4. funding (8h 事件 → 4h bar 内累计) ============
log("读取 funding ...")
fr = []
for f in sorted(glob.glob(str(BASE / "raw_temp/fundingRate/BTCUSDT/*/*.csv"))):
    d = pl.read_csv(f)
    c = d.select(pl.col("calc_time").cast(pl.Utf8))
    digits = c["calc_time"].str.contains(r"^\d+$")
    if digits.all():  # epoch ms/µs
        _v = c["calc_time"].str.to_integer()
        fr.append(d.with_columns(pl.when(_v >= 1e14).then(_v // 1000).otherwise(_v).alias("_ms"))
                      .with_columns(pl.from_epoch(pl.col("_ms"), time_unit="ms").alias("ts"))
                      .select(["ts", pl.col("last_funding_rate").alias("rate")]))
    else:  # 日期字符串
        fr.append(d.select(pl.col("calc_time").cast(pl.Utf8).str.to_datetime().alias("ts"),
                           pl.col("last_funding_rate").alias("rate")))
fr = pl.concat(fr).drop_nulls("ts").unique("ts").sort("ts")
fr = fr.with_columns(pl.col("ts").dt.truncate("4h").alias("bar_start"))
funding4 = fr.group_by("bar_start").agg(pl.col("rate").sum().alias("funding_cum_4h"), pl.col("rate").count().alias("n_events")).sort("bar_start")
qc["funding_4h"] = {"n_bars": int(funding4.height)}
funding4.write_parquet(OUT / "funding_4h.parquet")
log(f"funding done: {funding4.height}")

# ============ 5. premium (1h → 4h bar 边界差) ============
log("读取 premium index ...")
pr = []
for f in sorted(glob.glob(str(BASE / "raw_temp/premiumIndexKlines/BTCUSDT/*/*.csv"))):
    d = pl.read_csv(f, has_header=False)
# 列索引按 1h 管道 KLINE_COLS 映射: 0=open_time 1=mark 2=index 3=premium 4=est_settle("close"列=premium_close)
    d = d.rename({d.columns[0]: "open_ms", d.columns[4]: "premium"})
    _ot = pl.col("open_ms").cast(pl.Utf8).str.to_integer(strict=False)
    d = d.with_columns(pl.when(_ot >= 1e14).then(_ot // 1000).otherwise(_ot).alias("_ot_ms"))
    pr.append(d.select(pl.from_epoch(pl.col("_ot_ms"), time_unit="ms").alias("ts"), pl.col("premium").cast(pl.Float64, strict=False)))
pr = pl.concat(pr).drop_nulls(["ts", "premium"])
pr = pr.unique("ts").sort("ts").with_columns(pl.col("ts").dt.truncate("4h").alias("bar_start"))
pr4 = pr.group_by("bar_start").agg(pl.col("premium").last().alias("premium_4h"), pl.col("premium").first().alias("premium_start_4h"))
pr4 = pr4.with_columns(pl.col("premium_4h").diff(1).alias("premium_delta_4h")).sort("bar_start")  # 对齐 1h: bar 间 diff
qc["premium_4h"] = {"n_bars": int(pr4.height)}
pr4.write_parquet(OUT / "premium_4h.parquet")
log(f"premium done: {pr4.height}")

qc["elapsed_s"] = round(time.time() - t0, 1)
with open(OUT / "p1_qc.json", "w") as f:
    json.dump(qc, f, indent=1, default=str)
print("DONE_T1_BARS")
print(json.dumps(qc, indent=1, default=str))
