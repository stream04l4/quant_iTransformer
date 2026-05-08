import os
import sys
import requests
import zipfile
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path

# ==========================================
# 0. 环境初始化
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from quant_config.config import (
    TEMP_DIR, PROCESSED_MAIN_DIR, PROCESSED_AGGTRADES_DIR, SYMBOL, AUX_SYMBOLS,
    DATA_START_MONTH, DATA_END_MONTH, FORECAST_HORIZON,
    BINANCE_BASE_URL_FUTURES, BINANCE_BASE_URL_SPOT,
    KLINE_COLS, FUNDING_COLS
)

PROCESSED_MAIN_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. 强化版数据下载引擎 (带详细日志与防错)
# ==========================================
def generate_month_range(start_ym, end_ym):
    curr_y, curr_m = start_ym
    end_y, end_m = end_ym
    while (curr_y, curr_m) <= (end_y, end_m):
        yield curr_y, curr_m
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1

def fetch_binance_data(market, data_type, symbol, year, month):
    month_str = f"{month:02d}"
    base = BINANCE_BASE_URL_FUTURES if market == "futures" else BINANCE_BASE_URL_SPOT
    
    # 动态构建 URL
    if data_type == "klines":
        url = f"{base}/klines/{symbol}/1h/{symbol}-1h-{year}-{month_str}.zip"
    elif data_type == "premiumIndexKlines":
        url = f"{base}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{year}-{month_str}.zip"
    elif data_type == "fundingRate":
        url = f"{base}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month_str}.zip"
    else: return None

    type_temp_dir = TEMP_DIR / data_type / symbol
    type_temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = type_temp_dir / f"temp_{year}_{month_str}.zip"
    
    print(f"⬇️ 尝试下载: {data_type} | {symbol} | {year}-{month_str}")
    try:
        # 设置超时，防止死锁
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_filename = z.namelist()[0]
                z.extract(csv_filename, type_temp_dir)
            
            os.remove(zip_path)
            print("   ✅ 下载并解压成功")
            return type_temp_dir / csv_filename
        else:
            print(f"   ⚠️ 跳过 (HTTP {r.status_code}) - 可能是该月数据暂未生成或 URL 不存在")
            return None
    except Exception as e:
        print(f"   ❌ 网络异常: {e}")
        return None

def load_and_concat_months(market, data_type, symbol):
    dfs = []
    for year, month in generate_month_range(DATA_START_MONTH, DATA_END_MONTH):
        csv_path = fetch_binance_data(market, data_type, symbol, year, month)
        if csv_path and csv_path.exists():
            cols = FUNDING_COLS if data_type == "fundingRate" else KLINE_COLS
            df = pl.read_csv(csv_path, has_header=(market=="futures"), new_columns=cols)
            time_col = "calc_time" if data_type == "fundingRate" else "open_time"
            df = df.cast({col: pl.Float64 for col in cols if col != time_col})
            dfs.append(df)
            os.remove(csv_path)
            
    if not dfs: 
        print(f"🚨 严重警告: {data_type} ({symbol}) 未获取到任何数据！")
        return None
        
    full_df = pl.concat(dfs)
    time_col = "calc_time" if data_type == "fundingRate" else "open_time"
    return full_df.with_columns(
        pl.from_epoch(pl.col(time_col), time_unit="ms").dt.truncate("1h").alias("hour_timestamp")
    ).sort("hour_timestamp")

def build_rsi_expr(price_col: str, period: int) -> pl.Expr:
    delta = pl.col(price_col).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0).ewm_mean(span=period, adjust=False)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).ewm_mean(span=period, adjust=False)
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))

# ==========================================
# 2. 核心特征构建与聚合逻辑
# ==========================================
def build_features():
    print(f"\n🚀 [第一阶段] 正在拉取并拼接主维度数据...")
    btc_klines = load_and_concat_months("futures", "klines", SYMBOL)
    btc_funding = load_and_concat_months("futures", "fundingRate", SYMBOL)
    btc_premium = load_and_concat_months("futures", "premiumIndexKlines", SYMBOL)
    
    if btc_klines is None:
        print("❌ 核心 K 线数据为空，程序中止。")
        return

    print(f"\n🚀 [第二阶段] 正在拉取横截面辅助数据...")
    eth_usdt = load_and_concat_months("futures", "klines", AUX_SYMBOLS[0])
    eth_btc = load_and_concat_months("spot", "klines", AUX_SYMBOLS[1])

    print("\n🧩 [第三阶段] 执行全局时间轴对齐 (Join)...")
    btc_premium = btc_premium.select(["hour_timestamp", pl.col("close").alias("premium_close")]) if btc_premium is not None else None
    btc_funding = btc_funding.select(["hour_timestamp", "last_funding_rate"]) if btc_funding is not None else None
    eth_usdt = eth_usdt.select(["hour_timestamp", pl.col("close").alias("eth_usdt_close")]) if eth_usdt is not None else None
    eth_btc = eth_btc.select(["hour_timestamp", pl.col("close").alias("eth_btc_close")]) if eth_btc is not None else None

    # 动态 Join 机制：哪怕某个外围数据挂了，也不影响主 K 线对齐
    master_df = btc_klines
    if btc_funding is not None: master_df = master_df.join(btc_funding, on="hour_timestamp", how="left")
    if btc_premium is not None: master_df = master_df.join(btc_premium, on="hour_timestamp", how="left")
    if eth_usdt is not None: master_df = master_df.join(eth_usdt, on="hour_timestamp", how="left")
    if eth_btc is not None: master_df = master_df.join(eth_btc, on="hour_timestamp", how="left")
    
    # 🚨 双向空值填充：防止部分时序数据早期没上线导致的头部 NaN 断层
    master_df = master_df.fill_null(strategy="forward").fill_null(strategy="backward")

    print("⚙️ [第四阶段] 计算 23 维特征矩阵...")
    master_df = master_df.cast({col: pl.Float64 for col in ["open", "high", "low", "close", "volume"]})

    # 处理潜在的缺失列（如果下载失败，赋予一个常量列防止报错）
    required_cols = ["last_funding_rate", "premium_close", "eth_usdt_close", "eth_btc_close"]
    for col in required_cols:
        if col not in master_df.columns:
            print(f"⚠️ 缺失辅助列 {col}，使用 0 占位。")
            master_df = master_df.with_columns(pl.lit(0.0).alias(col))

    features = master_df.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("body_strength"),
        ((pl.col("close") - pl.col("close").ewm_mean(span=200, adjust=False)) / pl.col("close").ewm_mean(span=200, adjust=False)).alias("ma_bias"),
        ((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("upper_shadow"),
        ((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("lower_shadow"),
        ((pl.col("close").rolling_std(window_size=20) * 4) / (pl.col("close").rolling_mean(window_size=20) + 1e-9)).alias("bb_squeeze"),
        (pl.col("close").ewm_mean(span=12, adjust=False) - pl.col("close").ewm_mean(span=26, adjust=False)).alias("macd_line"),
        build_rsi_expr("close", 14).alias("rsi_14"),
        build_rsi_expr("close", 48).alias("rsi_48"),
        (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-9)).alias("taker_ratio"),
        (pl.col("quote_volume") / (pl.col("count") + 1e-9)).alias("vpt"),
        (pl.col("taker_buy_volume") - (pl.col("volume") - pl.col("taker_buy_volume"))).alias("net_taker_vol"),
        pl.col("last_funding_rate").alias("funding_rate"),
        (pl.col("premium_close") - pl.col("premium_close").shift(1)).alias("premium_delta"),
        (pl.col("eth_usdt_close").log() - pl.col("eth_usdt_close").shift(1).log()).alias("eth_log_return"),
        (pl.col("eth_btc_close").log() - pl.col("eth_btc_close").shift(1).log()).alias("eth_btc_return"),
        (pl.col("hour_timestamp").dt.hour() * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (pl.col("hour_timestamp").dt.hour() * (2 * np.pi / 24)).cos().alias("hour_cos"),
        (pl.col("hour_timestamp").dt.weekday() * (2 * np.pi / 7)).sin().alias("week_sin"),
        (pl.col("hour_timestamp").dt.weekday() * (2 * np.pi / 7)).cos().alias("week_cos"),
        (pl.col("close").shift(-FORECAST_HORIZON).log() - pl.col("close").log()).alias("target_24h_return")
    ])

    features = features.with_columns((pl.col("macd_line") - pl.col("macd_line").ewm_mean(span=9, adjust=False)).alias("macd_hist"))

    print("\n📊 [第五阶段] 聚合 aggTrades 历史订单流数据...")
    if list(PROCESSED_AGGTRADES_DIR.glob("*.parquet")):
        df_agg = pl.scan_parquet(PROCESSED_AGGTRADES_DIR / "*.parquet").collect().sort("hour_timestamp")
        features = features.join(df_agg, on="hour_timestamp", how="left").with_columns([
            pl.col("cvd").fill_null(0),
            pl.col("whale_net_flow").fill_null(0),
            # 强化双向填充
            pl.col("retail_ratio").fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.1)
        ])
    else:
        print("⚠️ 警告：未找到 aggTrades Parquet 文件，将跳过订单流维度。")

    # 确立最终保留的干净列
    final_cols = [
        "hour_timestamp", "log_return", "body_strength", "ma_bias", "upper_shadow", "lower_shadow",
        "bb_squeeze", "macd_hist", "rsi_14", "rsi_48", "taker_ratio", "vpt", "net_taker_vol",
        "funding_rate", "premium_delta", "eth_log_return", "eth_btc_return",
        "hour_sin", "hour_cos", "week_sin", "week_cos", "target_24h_return"
    ]
    if "cvd" in features.columns:
        final_cols.extend(["cvd", "whale_net_flow", "retail_ratio"])

    # 只选取需要的列
    features = features.select(final_cols)

    # 🔍 核心诊断：在 drop_nulls 前查验空值
    print("\n🩺 [空值诊断] 正在扫描异常的空列...")
    null_report = features.null_count()
    with pl.Config(tbl_cols=-1):
        print(null_report)

    # 最后只剥离正常的尾部空值 (如 target_24h_return 产生的末端空位)
    features = features.drop_nulls()
    
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"{SYMBOL}_Master_Training_Matrix_{current_date}.parquet"
    output_path = PROCESSED_MAIN_DIR / file_name
    features.write_parquet(output_path)
    print(f"\n✅ 终极宽表构建成功！路径: {output_path} | 最终规模: {features.shape}")

if __name__ == "__main__":
    build_features()