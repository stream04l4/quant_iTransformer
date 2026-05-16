import os
import sys
import requests
import zipfile
import numpy as np
import time
import shutil
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

def fetch_binance_data(market, data_type, symbol, year, month, timeframe="1h"):
    month_str = f"{month:02d}"
    base = BINANCE_BASE_URL_FUTURES if market == "futures" else BINANCE_BASE_URL_SPOT
    
    if data_type == "klines": url = f"{base}/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
    elif data_type == "premiumIndexKlines": url = f"{base}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{year}-{month_str}.zip"
    elif data_type == "fundingRate": url = f"{base}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month_str}.zip"
    else: return None

    # 🚨 核心修复 1：线程绝对隔离！为每个年月创建独立的沙盒文件夹
    sandbox_dir = TEMP_DIR / data_type / symbol / f"{year}_{month_str}"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    zip_path = sandbox_dir / "temp.zip"
    
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
            df = pl.read_csv(csv_path, has_header=(market=="futures"), new_columns=cols)
            time_col = "calc_time" if data_type == "fundingRate" else "open_time"
            df = df.cast({col: pl.Float64 for col in cols if col != time_col})
            shutil.rmtree(csv_path.parent, ignore_errors=True) # 🚨 摧毁整个独立沙盒，回收硬盘
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
        pl.from_epoch(pl.col(time_col), time_unit="ms").alias("exact_timestamp")
    ).sort("exact_timestamp")

    if save_raw_parquet:
        raw_dir = TEMP_DIR / "klines" / symbol
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_path = raw_dir / f"{symbol}_raw_{timeframe}.parquet"
        full_df.select(["exact_timestamp", "open", "high", "low", "close"]).rename({"exact_timestamp": "timestamp"}).write_parquet(output_path)
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
                "create_time", "sum_open_interest", "sum_toptrader_long_short_ratio"
            ]).with_columns([
                pl.col("sum_open_interest").cast(pl.Float64, strict=False),
                pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64, strict=False)
            ])
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 跳过异常 Metrics 文件 {p.name}: {e}")
        finally:
            shutil.rmtree(p.parent, ignore_errors=True)
    
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
    return metrics_hourly.select(["hour_timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio"])

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
    btc_klines = load_and_concat_months("futures", "klines", SYMBOL, timeframe="1h", save_raw_parquet=True)
    
    # 2. 独家补充：拉取 1m K线，只保存 raw_1m.parquet 给回测用，随后丢弃，不干扰训练特征
    print(f"\n🔬 [特殊补充] 正在拉取 1m K线，专供回测显微镜碰撞使用 (单线程下载可能会花 1-3 分钟)...")
    _ = load_and_concat_months("futures", "klines", SYMBOL, timeframe="1m", save_raw_parquet=True)
    btc_funding = load_and_concat_months("futures", "fundingRate", SYMBOL)
    btc_premium = load_and_concat_months("futures", "premiumIndexKlines", SYMBOL)
    
    if btc_klines is None:
        print("❌ 核心 K 线数据为空，程序中止。")
        return

    print(f"\n🚀 [第二阶段] 正在拉取横截面辅助数据...")
    # 🚨 改为拉取数据极度完整的 ETHUSDT 合约，用于数学合成
    eth_usdt = load_and_concat_months("futures", "klines", AUX_SYMBOLS[0])

    # 🚨 [新增拉取] 获取 Metrics (Open Interest)
    btc_metrics = fetch_daily_metrics_concurrent(SYMBOL, DATA_START_MONTH, DATA_END_MONTH)

    print("\n🧩 [第三阶段] 执行全局时间轴对齐 (Join)...")
    btc_premium = btc_premium.select(["hour_timestamp", pl.col("close").alias("premium_close")]) if btc_premium is not None else None
    btc_funding = btc_funding.select(["hour_timestamp", "last_funding_rate"]) if btc_funding is not None else None
    eth_usdt = eth_usdt.select(["hour_timestamp", pl.col("close").alias("eth_usdt_close")]) if eth_usdt is not None else None

    # 动态 Join 机制：哪怕某个外围数据挂了，也不影响主 K 线对齐
    master_df = btc_klines
    if btc_funding is not None: master_df = master_df.join(btc_funding, on="hour_timestamp", how="left")
    if btc_premium is not None: master_df = master_df.join(btc_premium, on="hour_timestamp", how="left")
    if eth_usdt is not None: master_df = master_df.join(eth_usdt, on="hour_timestamp", how="left")
    if btc_metrics is not None: master_df = master_df.join(btc_metrics, on="hour_timestamp", how="left")

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
    ]).with_columns([
        # 第一步：只对 close 和其他宏观状态类特征做前向填充
        pl.col("close").fill_null(strategy="forward"),
        pl.col("eth_usdt_close").fill_null(strategy="forward"),
        pl.col("last_funding_rate").fill_null(strategy="forward"),
        pl.col("premium_close").fill_null(strategy="forward"),
        pl.col("sum_open_interest").fill_null(strategy="forward"),
        pl.col("sum_toptrader_long_short_ratio").fill_null(strategy="forward")
    ]).with_columns([
        # 第二步：缺失的 open, high, low 必须对齐已经被填补的 close，表示这段时间价格彻底冻结，防止制造幽灵波动率
        pl.col("open").fill_null(pl.col("close")),
        pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")),
        # 交易量维度数据若缺失，视为该小时 0 成交，绝不能 forward fill
        pl.col("volume").fill_null(0.0),
        pl.col("quote_volume").fill_null(0.0),
        pl.col("count").fill_null(0),
        pl.col("taker_buy_volume").fill_null(0.0)
    ])

    # 🚨 产业界三角合成法：用 ETH/USDT 除以 BTC/USDT (即当前的 close) 生成 ETH/BTC
    if "eth_usdt_close" in master_df.columns:
        master_df = master_df.with_columns(
            (pl.col("eth_usdt_close") / pl.col("close")).alias("eth_btc_close")
        )
    
    # 🚨 [新增高频 Alpha] 计算 1 小时级别 Realized Volatility (RV)
    print("🔬 正在基于 1m 数据计算每小时高频已实现波动率 (RV)...")
    raw_1m_path = TEMP_DIR / "klines" / SYMBOL / f"{SYMBOL}_raw_1m.parquet"
    if raw_1m_path.exists():
        df_1m = pl.read_parquet(raw_1m_path)
        rv_df = df_1m.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("ret_1m"),
            pl.col("timestamp").dt.truncate("1h").alias("hour_timestamp")
        ]).group_by("hour_timestamp").agg(
            pl.col("ret_1m").pow(2).sum().sqrt().alias("realized_vol_1h")
        )
        master_df = master_df.join(rv_df, on="hour_timestamp", how="left")
    else:
        master_df = master_df.with_columns(pl.lit(0.0).alias("realized_vol_1h"))

    # 🚨 [核心修复] 分级前向填充：依据各个特征在物理世界的更新频率
    fill_exprs = []
    
    # 1. 资金费率是 8 小时一更，必须填满整个 8 小时周期
    if "last_funding_rate" in master_df.columns: 
        fill_exprs.append(pl.col("last_funding_rate").fill_null(strategy="forward", limit=8))
        
    if fill_exprs:
        master_df = master_df.with_columns(fill_exprs)

    # 3. 对于其他基础微观 K 线衍生的特征，依然保持 limit=3 防死水
    # 必须排除已经具有专属限流规则的列，防止二次溢出
    exclude_fill_cols = ["last_funding_rate"]
    target_fill_cols = [c for c in master_df.columns if c not in exclude_fill_cols]
    master_df = master_df.with_columns([
        pl.col(c).fill_null(strategy="forward", limit=3) for c in target_fill_cols
    ])

    print("⚙️ [第四阶段] 计算 31 维特征矩阵...")
    master_df = master_df.cast({col: pl.Float64 for col in ["open", "high", "low", "close", "volume"]})

    required_cols = ["last_funding_rate", "premium_close", "eth_btc_close"]
    for col in required_cols:
        if col not in master_df.columns:
            print(f"⚠️ 缺失辅助列 {col}，正在进行安全占位。")
            # 价格类数据必须使用 1.0 占位以防止 log(0) 产生负无穷，其它指标用 0.0
            fallback_val = 1.0 if "close" in col else 0.0
            master_df = master_df.with_columns(pl.lit(fallback_val).alias(col))
            
    # 合成完毕后，物理销毁用于计算的冗余母体数据
    if "eth_usdt_close" in master_df.columns:
        master_df = master_df.drop("eth_usdt_close")
    
    # 🚨 兼容性降维打击：通过前3阶原生中心矩手动展开计算滚动偏度 (Skewness)
    m1 = pl.col("log_return").rolling_mean(window_size=24)
    m2 = pl.col("log_return").pow(2).rolling_mean(window_size=24)
    m3 = pl.col("log_return").pow(3).rolling_mean(window_size=24)
    
    # 方差计算 (使用 .abs() 抹平极小浮点数精度带来的负数误差)
    var = (m2 - m1.pow(2)).abs() 
    
    # 偏度标准公式: E[(X-mu)^3] / sigma^3 = (m3 - 3*m1*m2 + 2*m1^3) / sigma^3
    skew_expr = ((m3 - 3 * m1 * m2 + 2 * m1.pow(3)) / (var.pow(1.5) + 1e-15)).fill_null(0).alias("vol_skew_24h")

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
        (pl.col("quote_volume") / (pl.col("count") + 1e-9)).alias("avg_trade_size"), # 修正命名逻辑，真实含义为平均交易粒度
        
        # 🚨 Alpha 1: Open Interest 持仓动能与价格背离交叉乘积
        ((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)) / (pl.col("sum_open_interest").shift(1) + 1e-9)).alias("oi_delta_pct"),
        pl.col("sum_toptrader_long_short_ratio").alias("ls_ratio"),
        
        # 🚨 Alpha 3: Amihud 流动性消耗率 (对数尺度)
        (
            (pl.col("log_return").abs() / (pl.col("quote_volume") + 1e-9)) + 1e-12
        ).log().alias("amihud_illiquidity"),
        
        # 🚨 [新增] RVTS: 波动率期限结构 (1h极短期波动 vs 24h基准波动)
        (pl.col("realized_vol_1h") / (pl.col("realized_vol_1h").rolling_mean(window_size=24) + 1e-9)).alias("rv_term_structure"),

        # 🚨 [修改] FRM: 资金费率动量 (24h均线 - 72h均线)，过滤长期极化噪音
        (pl.col("last_funding_rate").rolling_mean(window_size=24) - pl.col("last_funding_rate").rolling_mean(window_size=72)).alias("funding_rate_momentum"),
        (pl.col("premium_close") - pl.col("premium_close").shift(1)).alias("premium_delta"),
        (pl.col("eth_btc_close").log() - pl.col("eth_btc_close").shift(1).log()).alias("eth_btc_return"),
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
        df_agg = (
            pl.scan_parquet(PROCESSED_AGGTRADES_DIR / "*.parquet")
            .collect()
            .sort("hour_timestamp")
            .unique(subset=["hour_timestamp"], keep="last")
        )
        features = features.join(df_agg, on="hour_timestamp", how="left").with_columns([
            pl.col("cvd").fill_null(0),
            pl.col("whale_net_flow").fill_null(0),
            # 剔除 backward 填充，防止开局缺失时使用未来行倒填，直接使用 0.1(10%) 兜底
            pl.col("retail_ratio").fill_null(strategy="forward").fill_null(0.1)
        ])
    else:
        print("⚠️ 警告：未找到 aggTrades Parquet 文件，将跳过订单流维度。")

    # 🚨 时序平稳化：去除高度共线的 net_taker_vol，保留 vpt 和 cvd
    print("🌊 [时序平稳化] 正在对高维度绝对量特征进行 Robust Z-Score (MAD) 降维与截断...")
    # 🚨 [剔除] vpt，因其逻辑已被更敏锐的 cvd_zscore 覆盖
    rolling_cols = ["realized_vol_1h", "amihud_illiquidity", "oi_delta_pct"]
    if "cvd" in features.columns:
        rolling_cols.extend(["cvd", "whale_net_flow"])

    # 步骤 1: 计算 rolling median (严格事前：使用 t-1 往前的 168 个窗口作为环境基准)
    features = features.with_columns([
        pl.col(c).rolling_median(window_size=168).shift(1).alias(f"{c}_median") for c in rolling_cols
    ])
    # 步骤 2: 计算 rolling MAD (Median Absolute Deviation)，同样 shift(1)
    features = features.with_columns([
        (pl.col(c) - pl.col(f"{c}_median")).abs().rolling_median(window_size=168).shift(1).alias(f"{c}_mad") for c in rolling_cols
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
        "oi_delta_pct", "ls_ratio",
        "price_pos_24h", "price_pos_72h", "price_pos_168h", 
        "rv_term_structure", 
        "funding_rate_momentum", "premium_delta", "eth_btc_return", 
        "hour_sin", "hour_cos", "week_sin", "week_cos", f"target_{horizon}h_return"
    ]
    # 🌟 新增: COID 被放进含有深度订单流的分支中
    if "cvd" in features.columns:
        final_cols.extend(["cvd_zscore", "whale_net_flow_zscore", "retail_ratio", "cvd_oi_divergence"])

    features = features.select(final_cols)

    # 🔍 核心诊断：在 drop_nulls 前查验空值
    print("\n🩺 [空值诊断] 正在扫描异常的空列...")
    null_report = features.null_count()
    with pl.Config(tbl_cols=-1):
        print(null_report)

    # 最后只剥离正常的尾部空值 (如 target_24h_return 产生的末端空位)
    features = features.drop_nulls()
    
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