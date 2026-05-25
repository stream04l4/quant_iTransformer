import os
import sys
import requests
import zipfile
import numpy as np
import time
import polars as pl
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. 环境初始化
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from quant_config.config import (
    TEMP_DIR, PROCESSED_MAIN_DIR, PROCESSED_AGGTRADES_DIR, SYMBOL, AUX_SYMBOLS,
    DATA_START_MONTH, DATA_END_MONTH, FORECAST_HORIZON,
    BINANCE_BASE_URL_FUTURES, BINANCE_BASE_URL_SPOT, BINANCE_BASE_URL_METRICS_DAILY,
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

def timestamp_ms_expr(time_col: str) -> pl.Expr:
    raw_ts = pl.col(time_col).cast(pl.Int64, strict=False)
    return pl.when(raw_ts > 10_000_000_000_000).then(raw_ts // 1000).otherwise(raw_ts)

def csv_has_header(csv_path: Path) -> bool:
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_token = f.readline().split(",", 1)[0].strip()
    try:
        float(first_token)
        return False
    except ValueError:
        return True

def csv_schema_overrides(data_type: str) -> dict[str, pl.DataType]:
    if data_type == "fundingRate":
        return {
            "calc_time": pl.Int64,
            "funding_interval_hours": pl.Float64,
            "last_funding_rate": pl.Float64,
        }

    return {
        "open_time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "close_time": pl.Int64,
        "quote_volume": pl.Float64,
        "count": pl.Int64,
        "taker_buy_volume": pl.Float64,
        "taker_buy_quote_volume": pl.Float64,
        "ignore": pl.Float64,
    }

def fetch_binance_data(market, data_type, symbol, year, month, timeframe="1h"):
    month_str = f"{month:02d}"
    base = BINANCE_BASE_URL_FUTURES if market == "futures" else BINANCE_BASE_URL_SPOT
    
    if data_type == "klines": url = f"{base}/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
    elif data_type == "premiumIndexKlines": url = f"{base}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{year}-{month_str}.zip"
    elif data_type == "fundingRate": url = f"{base}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month_str}.zip"
    else: return None

    # 🚨 核心修复 1：线程绝对隔离！为每个年月创建独立的沙盒文件夹
    if data_type == "klines":
        sandbox_dir = TEMP_DIR / data_type / symbol / timeframe / f"{year}_{month_str}"
        legacy_sandbox_dir = TEMP_DIR / data_type / symbol / f"{year}_{month_str}"
    else:
        sandbox_dir = TEMP_DIR / data_type / symbol / f"{year}_{month_str}"
        legacy_sandbox_dir = None
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    zip_path = sandbox_dir / "temp.zip"

    existing_csvs = [p for p in sandbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not existing_csvs and timeframe == "1h" and legacy_sandbox_dir and legacy_sandbox_dir.exists():
        existing_csvs = [p for p in legacy_sandbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if existing_csvs:
        print(f"♻️ 复用本地缓存: {data_type} ({timeframe}) | {symbol} | {year}-{month_str}")
        return existing_csvs[0]
    
    print(f"⬇️ 正在下载: {data_type} ({timeframe}) | {symbol} | {year}-{month_str}")
    
    # 🚨 核心修复 2：加入防 DDoS 的退避重试机制
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=15)
            if r.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    csv_filename = z.namelist()[0]
                    z.extract(csv_filename, sandbox_dir)
                
                os.remove(zip_path)
                print("   ✅ 下载并解压成功")
                return sandbox_dir / csv_filename
            elif r.status_code in [403, 429]:
                # 触发限流，休眠避让后重试
                print(f"   ⚠️ 跳过 (HTTP {r.status_code})")
                time.sleep(2 * (attempt + 1))
            else:
                return None
        except Exception as e:
            print(f"   ❌ 网络异常: {e}")
            time.sleep(2)
    return None

def load_and_concat_months(market, data_type, symbol, timeframe="1h", save_raw_parquet=False):
    months_list = list(generate_month_range(DATA_START_MONTH, DATA_END_MONTH))
    
    # 定义单个任务的闭包函数，包含下载与解析
    def _download_and_read(ym):
        year, month = ym
        csv_path = fetch_binance_data(market, data_type, symbol, year, month, timeframe)
        if csv_path and csv_path.exists():
            cols = FUNDING_COLS if data_type == "fundingRate" else KLINE_COLS
            df = pl.read_csv(
                csv_path,
                has_header=csv_has_header(csv_path),
                new_columns=cols,
                schema_overrides=csv_schema_overrides(data_type),
            )
            time_col = "calc_time" if data_type == "fundingRate" else "open_time"
            df = df.cast({col: pl.Float64 for col in cols if col != time_col})
            return df
        return None

    dfs = []
    print(f"   ⚡ 开启多线程并发管道处理: {data_type} ({timeframe}) ...")
    
    # 启用 8 个线程并发下载和解析
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_download_and_read, ym): ym for ym in months_list}
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                dfs.append(df)
            
    if not dfs: 
        print(f"🚨 严重警告: {data_type} ({symbol}) 未获取到任何数据！")
        return None
        
    full_df = pl.concat(dfs)
    time_col = "calc_time" if data_type == "fundingRate" else "open_time"
    
    # 提取极其精确的毫秒时间戳
    full_df = full_df.with_columns(
        pl.from_epoch(timestamp_ms_expr(time_col), time_unit="ms").alias("exact_timestamp")
    ).sort("exact_timestamp")

    if save_raw_parquet:
        raw_dir = TEMP_DIR / "klines" / symbol
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_path = raw_dir / f"{symbol}_raw_{timeframe}.parquet"
        raw_cols = ["exact_timestamp", "open", "high", "low", "close", "volume"]
        optional_cols = ["quote_volume", "count", "taker_buy_volume"]
        raw_cols.extend([c for c in optional_cols if c in full_df.columns])
        full_df.select(raw_cols).rename({"exact_timestamp": "timestamp"}).write_parquet(output_path)
        print(f"   💾 {timeframe} 物理 K 线库合并完成并生成缓存: {output_path.name}")
        
    return full_df.with_columns(
        pl.col("exact_timestamp").dt.truncate("1h").alias("hour_timestamp")
    )

def fetch_daily_metrics_concurrent(symbol, start_ym, end_ym):
    print(f"\n🐋 启动多线程下载 Daily Metrics 订单流 (包含 Open Interest)...")
    start_date = date(start_ym[0], start_ym[1], 1)
    # 处理结束日期：推导到 end_ym 的最后一天
    if end_ym[1] == 12: end_date = date(end_ym[0] + 1, 1, 1) - timedelta(days=1)
    else: end_date = date(end_ym[0], end_ym[1] + 1, 1) - timedelta(days=1)
    
    # 限制不要超过今天，因为今天的历史包还没生成
    today = date.today()
    if end_date >= today: end_date = today - timedelta(days=1)

    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    base_url = BINANCE_BASE_URL_METRICS_DAILY
    
    type_temp_dir = TEMP_DIR / "metrics" / symbol
    type_temp_dir.mkdir(parents=True, exist_ok=True)
    
    def download_day(d):
        date_str = d.strftime("%Y-%m-%d")
        url = f"{base_url}/{symbol}/{symbol}-metrics-{date_str}.zip"
        
        # 🚨 每日隔离沙盒
        sandbox_dir = type_temp_dir / date_str
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        zip_path = sandbox_dir / "temp.zip"

        existing_csvs = [p for p in sandbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        if existing_csvs:
            return existing_csvs[0]
        
        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        csv_filename = z.namelist()[0]
                        z.extract(csv_filename, sandbox_dir)
                    os.remove(zip_path)
                    return sandbox_dir / csv_filename
                time.sleep(1)
            except Exception:
                time.sleep(1)
        return None

    csv_paths = []
    # 启用 16 个线程极速并发下载
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_day, d): d for d in date_list}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: csv_paths.append(res)
            if (i+1) % 100 == 0:
                print(f"   已处理 {i+1}/{len(date_list)} 天的数据...")

    if not csv_paths:
        return None

    print(f"   💾 正在合并 {len(csv_paths)} 天的 Metrics 数据...")
    dfs = []
    for p in csv_paths:
        try:
            # 宽容读取并强制转换为 Float64，遇到无法解析的乱码字符串 (strict=False) 自动转为空值
            df = pl.read_csv(p, has_header=True, infer_schema_length=0).select([
                "create_time", "sum_open_interest"
            ]).with_columns([
                pl.col("sum_open_interest").cast(pl.Float64, strict=False)
            ])
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 跳过异常 Metrics 文件 {p.name}: {e}")

    if not dfs:
        print("   ⚠️ 所有 Metrics 文件均读取失败，将保留缺失并在最终清洗阶段剔除。")
        return None
    
    metrics_df = pl.concat(dfs)
    
    # 解析时间并按小时采样，取该小时结束时最后一条状态
    # 截取前 19 位应对 "2026-05-10 00:05:00" 这种格式，并转化为 timestamp
    metrics_df = metrics_df.with_columns(
        pl.col("create_time").str.slice(0, 19).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("exact_timestamp")
    ).with_columns(
        pl.col("exact_timestamp").dt.truncate("1h").alias("hour_timestamp")
    )
    
    # 按小时聚合，提取最末状态
    metrics_hourly = metrics_df.sort("exact_timestamp").group_by("hour_timestamp").last()
    return metrics_hourly.select(["hour_timestamp", "sum_open_interest"])

def build_rsi_expr(price_col: str, period: int) -> pl.Expr:
    delta = pl.col(price_col).diff()
    # 使用 Wilder's Smoothing 标准的 alpha=1/period，而非 EMA 的 2/(N+1)
    gain = pl.when(delta > 0).then(delta).otherwise(0).ewm_mean(alpha=1.0/period, adjust=False)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).ewm_mean(alpha=1.0/period, adjust=False)
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))

# ==========================================
# 2. 核心特征构建与聚合逻辑
# ==========================================
def build_features(horizon):
    print(f"\n🚀 [第一阶段] 正在拉取并拼接主维度数据 (预测窗口: {horizon}h)...")
    
    # 1. 拉取 1h K线，并顺手保存一份未被对数和归一化污染的 raw_1h.parquet 给回测用
    main_klines = load_and_concat_months("futures", "klines", SYMBOL, timeframe="1h", save_raw_parquet=True)
    
    # 2. 独家补充：拉取 1m K线，只保存 raw_1m.parquet 给回测用，随后丢弃，不干扰训练特征
    print(f"\n🔬 [特殊补充] 正在拉取 1m K线，专供回测显微镜碰撞使用 (单线程下载可能会花 1-3 分钟)...")
    raw_1m_loaded = load_and_concat_months("futures", "klines", SYMBOL, timeframe="1m", save_raw_parquet=True) is not None
    main_funding = load_and_concat_months("futures", "fundingRate", SYMBOL)
    main_premium = load_and_concat_months("futures", "premiumIndexKlines", SYMBOL)
    
    if main_klines is None:
        print("❌ 核心 K 线数据为空，程序中止。")
        return

    print(f"\n🚀 [第二阶段] 正在拉取横截面辅助数据...")
    aux_reference = load_and_concat_months("spot", "klines", AUX_SYMBOLS[0]) if len(AUX_SYMBOLS) >= 1 else None
    direct_cross = load_and_concat_months("spot", "klines", AUX_SYMBOLS[1]) if len(AUX_SYMBOLS) >= 2 else None

    # 🚨 [新增拉取] 获取 Metrics (Open Interest)
    main_metrics = fetch_daily_metrics_concurrent(SYMBOL, DATA_START_MONTH, DATA_END_MONTH)

    print("\n🧩 [第三阶段] 执行全局时间轴对齐 (Join)...")
    main_premium = main_premium.select(["hour_timestamp", pl.col("close").alias("premium_close")]) if main_premium is not None else None
    main_funding = main_funding.select(["hour_timestamp", "last_funding_rate"]) if main_funding is not None else None
    aux_reference = aux_reference.select(["hour_timestamp", pl.col("close").alias("aux_reference_close")]) if aux_reference is not None else None
    direct_cross = direct_cross.select(["hour_timestamp", pl.col("close").alias("cross_pair_close")]) if direct_cross is not None else None

    # 动态 Join 机制：哪怕某个外围数据挂了，也不影响主 K 线对齐
    master_df = main_klines
    if main_funding is not None: master_df = master_df.join(main_funding, on="hour_timestamp", how="left")
    if main_premium is not None: master_df = master_df.join(main_premium, on="hour_timestamp", how="left")
    if aux_reference is not None: master_df = master_df.join(aux_reference, on="hour_timestamp", how="left")
    if direct_cross is not None: master_df = master_df.join(direct_cross, on="hour_timestamp", how="left")
    if main_metrics is not None: master_df = master_df.join(main_metrics, on="hour_timestamp", how="left")

    # 训练集不使用中性值伪造缺失信号；可选数据源缺席时只补 Null 列，后续统一 drop_nulls 剔除。
    optional_null_cols = {
        "last_funding_rate": pl.Float64,
        "premium_close": pl.Float64,
        "aux_reference_close": pl.Float64,
        "cross_pair_close": pl.Float64,
        "sum_open_interest": pl.Float64,
    }
    missing_optional_cols = [
        pl.lit(None).cast(dtype).alias(col)
        for col, dtype in optional_null_cols.items()
        if col not in master_df.columns
    ]
    if missing_optional_cols:
        master_df = master_df.with_columns(missing_optional_cols)

    # ==========================================
    # 🚨 核心修复：强制时间轴连续化，消除断网/维护造成的 shift/rolling 未来信息错位
    # ==========================================
    master_df = master_df.upsample(
        time_column="hour_timestamp", 
        every="1h"
    ).with_columns([
        # 🚨 深度修复：全数据源断档监控。只要 K线、持仓量或溢价中任意一个断网缺失，即判定该行为人工合成
        pl.any_horizontal([
            pl.col("close").is_null(),
            pl.col("sum_open_interest").is_null(),
            pl.col("premium_close").is_null()
        ]).cast(pl.Int8).alias("is_synthetic")
    ])

    # 优先使用直接交叉对；若缺失，则回退到“辅助现货/主币”合成比值。
    master_df = master_df.with_columns(
        pl.coalesce([
            pl.col("cross_pair_close"),
            pl.when(pl.col("aux_reference_close").is_not_null())
              .then(pl.col("aux_reference_close") / (pl.col("close") + 1e-9))
              .otherwise(None)
        ]).alias("cross_asset_close")
    )
    
    # 🚨 [新增高频 Alpha] 计算 1 小时级别 Realized Volatility (RV)
    print("🔬 正在基于 1m 数据计算每小时高频已实现波动率 (RV)...")
    raw_1m_path = TEMP_DIR / "klines" / SYMBOL / f"{SYMBOL}_raw_1m.parquet"
    if raw_1m_loaded and raw_1m_path.exists():
        df_1m = pl.read_parquet(raw_1m_path)
        rv_df = df_1m.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1m"),
            pl.col("timestamp").dt.truncate("1h").alias("hour_timestamp")
        ]).group_by("hour_timestamp").agg(
            pl.col("ret_1m").pow(2).sum().sqrt().alias("realized_vol_1h")
        )
        master_df = master_df.join(rv_df, on="hour_timestamp", how="left")
    else:
        master_df = master_df.with_columns(pl.lit(None).cast(pl.Float64).alias("realized_vol_1h"))

    # 🚨 [核心修复] 分级前向填充：依据各个特征在物理世界的更新频率
    fill_exprs = []
    
    # 1. 资金费率是 8 小时一更，必须填满整个 8 小时周期
    if "last_funding_rate" in master_df.columns: 
        fill_exprs.append(pl.col("last_funding_rate").fill_null(strategy="forward", limit=8))
        
    if fill_exprs:
        master_df = master_df.with_columns(fill_exprs)

    print("⚙️ [第四阶段] 计算 31 维特征矩阵...")
    master_df = master_df.cast({col: pl.Float64 for col in ["open", "high", "low", "close", "volume"]})
            
    # 合成完毕后，物理销毁用于计算的冗余母体数据
    drop_aux_cols = [c for c in ["aux_reference_close", "cross_pair_close"] if c in master_df.columns]
    if drop_aux_cols:
        master_df = master_df.drop(drop_aux_cols)
    
    # 🚨 兼容性降维打击：通过前3阶原生中心矩手动展开计算滚动偏度 (Skewness)
    m1 = pl.col("log_return").rolling_mean(window_size=24)
    m2 = pl.col("log_return").pow(2).rolling_mean(window_size=24)
    m3 = pl.col("log_return").pow(3).rolling_mean(window_size=24)
    
    # 方差计算 (使用 .abs() 抹平极小浮点数精度带来的负数误差)
    var = (m2 - m1.pow(2)).abs() 
    
    # 偏度标准公式: E[(X-mu)^3] / sigma^3 = (m3 - 3*m1*m2 + 2*m1^3) / sigma^3
    skew_expr = ((m3 - 3 * m1 * m2 + 2 * m1.pow(3)) / (var.pow(1.5) + 1e-15)).alias("vol_skew_24h")

    # 🚨 [核心修正] 在计算任何位移特征之前，先计算核心对数收益率
    master_df = master_df.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    ])

    # 🚨 [核心修正] 现在 log_return 存在了，直接用它来计算波动偏度 (Skewness)
    features = master_df.with_columns([
        ((pl.col("close") - pl.col("close").ewm_mean(span=200, adjust=False)) / pl.col("close").ewm_mean(span=200, adjust=False)).alias("ma_bias"),
        ((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("upper_shadow"),
        ((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("lower_shadow"),
        ((pl.col("close").rolling_std(window_size=20) * 4) / (pl.col("close").rolling_mean(window_size=20) + 1e-9)).alias("bb_squeeze"),
        
        (
            (pl.col("close").ewm_mean(span=12, adjust=False) - pl.col("close").ewm_mean(span=26, adjust=False)) 
            / (pl.col("close").ewm_mean(span=26, adjust=False) + 1e-9)
        ).alias("ppo_line"),

        # 🚨 [新增] MAX 因子 (彩票效应)：捕捉过去24小时内的最大单根K线涨幅，识别非理性拉升
        (pl.col("log_return").rolling_max(window_size=24)).alias("max_ret_24h"),

        ((build_rsi_expr("close", 14) - 50) / 50.0).alias("rsi_14"),
        ((build_rsi_expr("close", 48) - 50) / 50.0).alias("rsi_48"),
        
        # 🚨 [修改] TBRD: 主动买入占比的差分(加速度)，捕捉资金突袭
        (pl.col("taker_buy_volume") / (pl.col("volume") + 1e-9)).diff().alias("taker_ratio_delta"),
        (pl.col("quote_volume") / (pl.col("count") + 1e-9)).alias("avg_trade_size"), 
        
        # ==========================================================
        # 🚨 终极护城河修改区 (开始)
        # ==========================================================
        
        # 🚨 Alpha 1: Open Interest (修复 91万亿溢出，使用 Null 丢弃宕机恢复行)
        pl.when(pl.col("sum_open_interest").shift(1).abs() > 1e-6)
          .then(((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)) / pl.col("sum_open_interest").shift(1)).clip(-1.0, 1.0))
          .otherwise(None)
          .alias("oi_delta_pct"),
          
        # 🚨 Alpha 3: Amihud 流动性消耗率 (增加 clip 保护，防止极端缩量暴涨)
        (
            (pl.col("log_return").abs() / (pl.col("quote_volume") + 1e-9)) + 1e-12
        ).log().clip(-50.0, 50.0).alias("amihud_illiquidity"),
        
        # 🚨 [新增] RVTS: 波动率期限结构 (增加 clip 保护，防止死水行情方差趋近于0引发爆发)
        (pl.col("realized_vol_1h") / (pl.col("realized_vol_1h").rolling_mean(window_size=24) + 1e-9)).clip(0.0, 50.0).alias("rv_term_structure"),
        
        # ==========================================================
        # 🚨 终极护城河修改区 (结束)
        # ==========================================================

        # 🚨 [修改] FRM: 资金费率动量 (24h均线 - 72h均线)，过滤长期极化噪音
        (pl.col("last_funding_rate").rolling_mean(window_size=24) - pl.col("last_funding_rate").rolling_mean(window_size=72)).alias("funding_rate_momentum"),
        (pl.col("premium_close") - pl.col("premium_close").shift(1)).alias("premium_delta"),
        (pl.col("cross_asset_close").log() - pl.col("cross_asset_close").shift(1).log()).alias("cross_asset_return"),
        (pl.col("hour_timestamp").dt.hour() * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (pl.col("hour_timestamp").dt.hour() * (2 * np.pi / 24)).cos().alias("hour_cos"),
        (pl.col("hour_timestamp").dt.weekday() * (2 * np.pi / 7)).sin().alias("week_sin"),
        (pl.col("hour_timestamp").dt.weekday() * (2 * np.pi / 7)).cos().alias("week_cos"),
        (pl.col("close").shift(-horizon).log() - pl.col("close").log()).alias(f"target_{horizon}h_return")
    ])

    features = features.with_columns(
        (pl.col("ppo_line") - pl.col("ppo_line").ewm_mean(span=9, adjust=False)).alias("ppo_hist"),
        (pl.col("log_return").sign() * pl.col("oi_delta_pct")).alias("oi_price_cross"), # 多头放量建仓 vs 空头被动踩踏
        skew_expr, # 🚨 Alpha 4: 波动偏度 (原生中心矩计算版)
        # 🚨 产业界级突破特征：多时间级别 (MTF) 结构分位数
        # 24h 级别 (微观结构，极度适配未来 6h 预测)
        ((pl.col("close") - pl.col("low").rolling_min(window_size=24)) / (pl.col("high").rolling_max(window_size=24) - pl.col("low").rolling_min(window_size=24) + 1e-9)).alias("price_pos_24h"),
        # 72h 级别 (中短线波段中枢)
        ((pl.col("close") - pl.col("low").rolling_min(window_size=72)) / (pl.col("high").rolling_max(window_size=72) - pl.col("low").rolling_min(window_size=72) + 1e-9)).alias("price_pos_72h"),
        # 168h 级别 (宏观压制/支撑位置)
        ((pl.col("close") - pl.col("low").rolling_min(window_size=168)) / (pl.col("high").rolling_max(window_size=168) - pl.col("low").rolling_min(window_size=168) + 1e-9)).alias("price_pos_168h")
    )

    print("\n📊 [第五阶段] 聚合 aggTrades 历史订单流数据...")
    if list(PROCESSED_AGGTRADES_DIR.glob("*.parquet")):
        # 🚨 致命缺陷修复：强行按时间去重，防范月度包与日线增量包交叉覆盖导致的行数爆炸
        # 日线增量通常比月度包更新，显式排序与优先级保证同一小时保留结果可复现。
        parquet_files = sorted(PROCESSED_AGGTRADES_DIR.glob("*.parquet"))
        required_agg_cols = {
            "hour_timestamp", "cvd", "whale_buy", "whale_sell",
            "whale_net_flow", "retail_ratio", "whale_ls_ratio"
        }
        invalid_agg_files = []
        for f in parquet_files:
            missing_cols = required_agg_cols - set(pl.read_parquet_schema(f).keys())
            if missing_cols:
                invalid_agg_files.append(f"{f.name} 缺少 {sorted(missing_cols)}")
        if invalid_agg_files:
            preview = "\n".join(f"   - {item}" for item in invalid_agg_files[:10])
            extra = f"\n   ... 另有 {len(invalid_agg_files) - 10} 个文件" if len(invalid_agg_files) > 10 else ""
            raise ValueError(
                "aggTrades 特征文件 schema 已过期，请先删除/重建这些 parquet 后再构建主特征：\n"
                f"{preview}{extra}"
            )

        dfs_agg = [
            pl.read_parquet(f).with_columns(
                pl.lit(1 if "_daily_" in f.name else 0).alias("_source_priority")
            )
            for f in parquet_files
        ]
        df_agg = (
            pl.concat(dfs_agg, how="diagonal")
            .sort(["hour_timestamp", "_source_priority"])
            .unique(subset=["hour_timestamp"], keep="last")
            .drop("_source_priority")
        )
        features = features.join(df_agg, on="hour_timestamp", how="left").with_columns([
            pl.col("cvd").fill_nan(None),
            pl.col("whale_net_flow").fill_nan(None),
            pl.col("retail_ratio").fill_nan(None),
            pl.col("whale_ls_ratio").fill_nan(None)
        ])
    else:
        print("⚠️ 警告：未找到 aggTrades Parquet 文件，将跳过订单流维度。")

    # 🚨 时序平稳化：去除高度共线的 net_taker_vol，保留 vpt 和 cvd
    print("🌊 [时序平稳化] 正在对高维度绝对量特征进行 Robust Z-Score (MAD) 降维与截断...")
    # 🚨 [剔除] vpt，因其逻辑已被更敏锐的 cvd_zscore 覆盖
    rolling_cols = ["realized_vol_1h", "amihud_illiquidity", "oi_delta_pct"]
    if "cvd" in features.columns:
        rolling_cols.extend(["cvd", "whale_net_flow", "whale_ls_ratio"])

    # 步骤 1: 计算 rolling median (严格事前：使用 t-1 往前的 168 个窗口作为环境基准)
    features = features.with_columns([
        pl.col(c).rolling_median(window_size=168, min_samples=120).shift(1).alias(f"{c}_median") for c in rolling_cols
    ])
    # 步骤 2: 计算 rolling MAD (Median Absolute Deviation)，同样 shift(1)
    features = features.with_columns([
        (pl.col(c) - pl.col(f"{c}_median")).abs().rolling_median(window_size=168, min_samples=120).shift(1).alias(f"{c}_mad") for c in rolling_cols
    ])
    # 步骤 3: 计算 Robust Z-Score 并截断到 [-4, 4]
    zscore_exprs = [
        (0.6745 * (pl.col(c) - pl.col(f"{c}_median")) / (pl.col(f"{c}_mad") + 1e-9)).clip(-4.0, 4.0).alias(f"{c}_zscore") for c in rolling_cols
    ]
    features = features.with_columns(zscore_exprs)
    
    # 步骤 4: 丢弃中间临时变量
    features = features.drop([f"{c}_median" for c in rolling_cols] + [f"{c}_mad" for c in rolling_cols])

    # 🚨 [新增] COID: 高阶订单流背离 (CVD净买单偏离度减去总持仓偏离度)
    if "cvd_zscore" in features.columns and "oi_delta_pct_zscore" in features.columns:
        features = features.with_columns(
            (pl.col("cvd_zscore") - pl.col("oi_delta_pct_zscore")).alias("cvd_oi_divergence")
        )

    # 🚨 终极核对：新增 max_ret_24h，剔除 oi_price_cross 与 vpt_zscore
    final_cols = [
        "hour_timestamp", "is_synthetic", "log_return", "max_ret_24h", "ma_bias", "upper_shadow", "lower_shadow",
        "bb_squeeze", "ppo_hist", "rsi_14", "rsi_48", "taker_ratio_delta",
        "amihud_illiquidity_zscore", "realized_vol_1h_zscore", "vol_skew_24h",
        "oi_delta_pct",
        "price_pos_24h", "price_pos_72h", "price_pos_168h", 
        "rv_term_structure", 
        "funding_rate_momentum", "premium_delta", "cross_asset_return",
        "hour_sin", "hour_cos", "week_sin", "week_cos", f"target_{horizon}h_return"
    ]
    # 🌟 新增: COID 被放进含有深度订单流的分支中
    if "cvd" in features.columns:
        final_cols.extend(["cvd_zscore", "whale_net_flow_zscore", "whale_ls_ratio_zscore", "retail_ratio", "cvd_oi_divergence"])

    features = features.select(final_cols)

    print("\n🩺 [极值诊断] 启动全局 NaN / Inf 强制清洗网...")
    
    # 获取所有浮点数列
    float_cols = [c for c, dtype in features.schema.items() if dtype in (pl.Float32, pl.Float64)]
    
    # 1. 绞杀所有 NaN 转化为 Null
    features = features.with_columns([
        pl.col(c).fill_nan(None).alias(c) for c in float_cols
    ])
    
    # 2. 绞杀所有 Infinity 转化为 Null
    features = features.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in float_cols
    ])

    print("\n🩺 [空值诊断] 正在扫描异常空列 (脏数据已被全部转化为 Null)...")
    null_report = features.null_count()
    with pl.Config(tbl_cols=-1):
        print(null_report)

    # 无漏网之鱼，安全执行空值剥离
    before_rows = features.height
    features = features.drop_nulls()
    after_rows = features.height
    drop_ratio = ((before_rows - after_rows) / before_rows) if before_rows else 1.0
    print(f"🧹 drop_nulls 清洗完成: 删除 {before_rows - after_rows} 行，剩余 {after_rows} 行，删除比例 {drop_ratio:.2%}")
    if drop_ratio > 0.2:
        print("⚠️ 警告：缺失清洗删除比例超过 20%，建议检查数据源完整性后再训练。")
    
    current_date = datetime.now().strftime("%Y%m%d")
    # 👇 文件名加上了 {horizon}h 的标志
    file_name = f"{SYMBOL}_Master_Training_Matrix_{horizon}h_{current_date}.parquet"
    output_path = PROCESSED_MAIN_DIR / file_name
    features.write_parquet(output_path)
    print(f"\n✅ 终极宽表构建成功！路径: {output_path} | 最终规模: {features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建加密货币量化特征宽表')
    parser.add_argument('--horizon', type=int, default=FORECAST_HORIZON, help='预测未来的时间窗口 (小时)')
    args = parser.parse_args()
    
    build_features(horizon=args.horizon)
