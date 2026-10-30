import os
import sys
import json
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
    TEMP_DIR, PROCESSED_MAIN_DIR, PROCESSED_AGGTRADES_DIR,
    PROCESSED_AGGTRADES_ROOT, SYMBOL, AUX_SYMBOLS,
    DATA_START_MONTH, DATA_END_MONTH, FORECAST_HORIZON,
    BINANCE_BASE_URL_FUTURES, BINANCE_BASE_URL_FUTURES_DAILY, BINANCE_BASE_URL_SPOT,
    BINANCE_BASE_URL_SPOT_DAILY, BINANCE_BASE_URL_METRICS_DAILY,
    KLINE_COLS, FUNDING_COLS
)

PROCESSED_MAIN_DIR.mkdir(parents=True, exist_ok=True)


def find_symbol_aggtrade_files(symbol):
    pattern = f"{symbol}_aggTrades_*.parquet"
    candidate_dirs = [PROCESSED_AGGTRADES_DIR, PROCESSED_AGGTRADES_ROOT]
    files = []
    for directory in dict.fromkeys(candidate_dirs):
        if directory.exists():
            files.extend(directory.glob(pattern))
    return sorted(set(files))

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
        save_raw_kline_parquet(full_df, timeframe)

    return full_df.with_columns(
        pl.col("exact_timestamp").dt.truncate("1h").alias("hour_timestamp")
    )

def save_raw_kline_parquet(full_df, timeframe):
    """保存未被污染的 raw K 线库 (给回测/RV 计算用)，当前月 daily 补齐后也会刷新。"""
    raw_dir = TEMP_DIR / "klines" / SYMBOL
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{SYMBOL}_raw_{timeframe}.parquet"
    raw_cols = ["exact_timestamp", "open", "high", "low", "close", "volume"]
    optional_cols = ["quote_volume", "count", "taker_buy_volume"]
    raw_cols.extend([c for c in optional_cols if c in full_df.columns])
    full_df.select(raw_cols).rename({"exact_timestamp": "timestamp"}).write_parquet(output_path)
    print(f"   💾 {timeframe} 物理 K 线库合并完成并生成缓存: {output_path.name}")

# ==========================================
# 1b. 当前月 daily 增量补齐 (数据 dump 有 T+1~2 天延迟, 未发布日自动跳过)
# ==========================================
def fetch_daily_fill(market, data_type, symbol, timeframe, start_date, end_date):
    """当前月月度文件要到月末才生成，用 daily 包把数据桥接到最近可用日。
    404 (当天 dump 未发布) 跳过继续，保证幂等可重跑。"""
    base_daily = BINANCE_BASE_URL_FUTURES_DAILY if market == "futures" else BINANCE_BASE_URL_SPOT_DAILY
    sandbox_root = TEMP_DIR / "daily" / data_type / symbol / timeframe
    dfs = []
    d = start_date
    while d <= end_date:
        date_str = d.strftime("%Y-%m-%d")
        if data_type == "premiumIndexKlines":
            url = f"{base_daily}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{date_str}.zip"
        else:
            url = f"{base_daily}/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{date_str}.zip"
        sandbox_dir = sandbox_root / date_str
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        csv_path = next((p for p in sandbox_dir.iterdir() if p.suffix.lower() == ".csv"), None)
        if csv_path is None:
            try:
                with requests.get(url, stream=True, timeout=15) as r:
                    if r.status_code == 200:
                        zip_path = sandbox_dir / "temp.zip"
                        with open(zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                        with zipfile.ZipFile(zip_path) as z:
                            z.extract(z.namelist()[0], sandbox_dir)
                        os.remove(zip_path)
                    else:
                        print(f"   ⚠️ {date_str} 尚未发布 (HTTP {r.status_code})，跳过")
                csv_path = next((p for p in sandbox_dir.iterdir() if p.suffix.lower() == ".csv"), None)
            except Exception as e:
                print(f"   ❌ {date_str} 下载异常: {e}")
        if csv_path is not None:
            try:
                df = pl.read_csv(csv_path, has_header=csv_has_header(csv_path),
                                 new_columns=KLINE_COLS, schema_overrides=csv_schema_overrides("klines"))
                df = df.cast({col: pl.Float64 for col in KLINE_COLS if col != "open_time"})
                dfs.append(df)
                print(f"   ✅ daily {data_type} ({timeframe}) | {symbol} | {date_str} | {df.height} 行")
            except Exception as e:
                print(f"   ❌ {date_str} 解析失败: {e}")
        d += timedelta(days=1)
    if not dfs:
        return None
    full_df = pl.concat(dfs).with_columns(
        pl.from_epoch(timestamp_ms_expr("open_time"), time_unit="ms").alias("exact_timestamp")
    ).sort("exact_timestamp")
    return full_df.with_columns(pl.col("exact_timestamp").dt.truncate("1h").alias("hour_timestamp"))

def fetch_funding_api(symbol, start_date):
    """fundingRate 在 data.binance.vision 上没有 daily dump (只有月度)，
    当前月从实时 API 补 (8 小时间隔，单页 limit=1000 足够覆盖当月)。"""
    start_ms = int(datetime(start_date.year, start_date.month, 1).timestamp() * 1000)
    try:
        r = requests.get(
            f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={start_ms}&limit=1000",
            timeout=15,
        )
        r.raise_for_status()
        rows = [{"calc_time": int(it["fundingTime"]), "funding_interval_hours": 8.0,
                 "last_funding_rate": float(it["fundingRate"])} for it in r.json()]
    except Exception as e:
        print(f"   ❌ funding API 补齐失败: {e}")
        return None
    if not rows:
        return None
    return pl.DataFrame(rows).with_columns([
        pl.from_epoch(pl.col("calc_time"), time_unit="ms").alias("exact_timestamp"),
        pl.from_epoch(pl.col("calc_time"), time_unit="ms").dt.truncate("1h").alias("hour_timestamp"),
    ])

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
    hourly_cache_path = type_temp_dir / f"{symbol}_metrics_hourly.parquet"
    manifest_path = type_temp_dir / f"{symbol}_metrics_manifest.json"

    def read_metrics_manifest():
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️ Metrics manifest 读取失败，将重新校验缓存: {e}")
            return None

    def read_hourly_cache():
        if not hourly_cache_path.exists():
            return None
        try:
            df = pl.read_parquet(hourly_cache_path)
            required_cols = {"hour_timestamp", "sum_open_interest"}
            if not required_cols.issubset(set(df.columns)):
                print(f"   ⚠️ Metrics 小时级缓存 schema 过期，将重建: {hourly_cache_path.name}")
                return None
            return df.select(["hour_timestamp", "sum_open_interest"]).sort("hour_timestamp")
        except Exception as e:
            print(f"   ⚠️ Metrics 小时级缓存读取失败，将重建: {e}")
            return None

    def filter_metrics_range(df):
        end_exclusive = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
        start_inclusive = datetime.combine(start_date, datetime.min.time())
        return df.filter(
            (pl.col("hour_timestamp") >= start_inclusive)
            & (pl.col("hour_timestamp") < end_exclusive)
        )

    cached_hourly = read_hourly_cache()
    manifest = read_metrics_manifest()
    download_start_date = start_date

    if cached_hourly is not None and manifest:
        try:
            manifest_end = date.fromisoformat(manifest["requested_end"])
            if manifest_end >= end_date:
                cached_slice = filter_metrics_range(cached_hourly)
                if not cached_slice.is_empty():
                    print(f"   ♻️ 复用 Metrics 小时级缓存: {hourly_cache_path.name} | {cached_slice.height} 行")
                    return cached_slice
            if manifest_end < end_date:
                download_start_date = manifest_end + timedelta(days=1)
                print(f"   ♻️ Metrics 已缓存至 {manifest_end}，仅补齐 {download_start_date} -> {end_date}")
        except Exception as e:
            print(f"   ⚠️ Metrics manifest 内容异常，将重新扫描日线缓存: {e}")

    if download_start_date > end_date:
        cached_slice = filter_metrics_range(cached_hourly) if cached_hourly is not None else None
        return cached_slice if cached_slice is not None and not cached_slice.is_empty() else None

    date_list = [
        download_start_date + timedelta(days=x)
        for x in range((end_date - download_start_date).days + 1)
    ]

    def day_sandbox_dir(d):
        return type_temp_dir / d.strftime("%Y-%m-%d")

    def is_day_resolved(d):
        sandbox_dir = day_sandbox_dir(d)
        if not sandbox_dir.exists():
            return False
        has_csv = any(p.is_file() and p.suffix.lower() == ".csv" for p in sandbox_dir.iterdir())
        return has_csv or (sandbox_dir / "missing.marker").exists()
    
    def download_day(d):
        date_str = d.strftime("%Y-%m-%d")
        url = f"{base_url}/{symbol}/{symbol}-metrics-{date_str}.zip"
        
        # 🚨 每日隔离沙盒
        sandbox_dir = day_sandbox_dir(d)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        zip_path = sandbox_dir / "temp.zip"
        missing_marker = sandbox_dir / "missing.marker"

        existing_csvs = [p for p in sandbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        if existing_csvs:
            return existing_csvs[0]
        if missing_marker.exists():
            return None
        
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
                if r.status_code == 404:
                    missing_marker.write_text(f"{url}\n", encoding="utf-8")
                    return None
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
        cached_slice = filter_metrics_range(cached_hourly) if cached_hourly is not None else None
        return cached_slice if cached_slice is not None and not cached_slice.is_empty() else None

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
    metrics_hourly = metrics_hourly.select(["hour_timestamp", "sum_open_interest"])

    if cached_hourly is not None and not cached_hourly.is_empty():
        metrics_hourly = pl.concat([cached_hourly, metrics_hourly], how="vertical_relaxed")

    metrics_hourly = (
        metrics_hourly
        .sort("hour_timestamp")
        .unique(subset=["hour_timestamp"], keep="last")
        .select(["hour_timestamp", "sum_open_interest"])
    )
    metrics_hourly.write_parquet(hourly_cache_path)

    if all(is_day_resolved(d) for d in date_list):
        old_start = date.fromisoformat(manifest["requested_start"]) if manifest and "requested_start" in manifest else start_date
        manifest_payload = {
            "symbol": symbol,
            "requested_start": min(old_start, start_date).isoformat(),
            "requested_end": end_date.isoformat(),
            "hourly_cache": hourly_cache_path.name,
            "updated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, ensure_ascii=False, indent=2)
        print(f"   💾 Metrics 小时级缓存已更新: {hourly_cache_path.name}")

    return filter_metrics_range(metrics_hourly)

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

    # ==========================================
    # 📅 [当前月 daily 增量补齐] 月度 dump 到上月为止，
    #    当前月从 1 号补到昨天 (T+1~2 天延迟，未发布日自动跳过)。
    # ==========================================
    today = date.today()
    if tuple(DATA_END_MONTH) >= (today.year, today.month):
        d_start = date(today.year, today.month, 1)
        d_end = today - timedelta(days=1)
        if d_end >= d_start:
            print(f"\n📅 [当前月 daily 增量补齐] {d_start} → {d_end} ...")
            extra = fetch_daily_fill("futures", "klines", SYMBOL, "1h", d_start, d_end)
            if extra is not None:
                main_klines = pl.concat([main_klines, extra]).unique(subset=["exact_timestamp"], keep="last").sort("exact_timestamp")
                save_raw_kline_parquet(main_klines, "1h")  # 刷新 raw_1h (含当前月)
            extra_1m = fetch_daily_fill("futures", "klines", SYMBOL, "1m", d_start, d_end)
            if extra_1m is not None:
                # 1m 只用于 realized_vol (读 raw_1m.parquet)，与月度缓存按 timestamp 去重合并后刷新
                raw_1m_path = TEMP_DIR / "klines" / SYMBOL / f"{SYMBOL}_raw_1m.parquet"
                raw_cols = ["timestamp", "open", "high", "low", "close", "volume",
                            "quote_volume", "count", "taker_buy_volume"]
                extra_raw = extra_1m.rename({"exact_timestamp": "timestamp"})
                base = pl.read_parquet(raw_1m_path) if raw_1m_path.exists() else None
                common = [c for c in raw_cols if c in extra_raw.columns and (base is None or c in base.columns)]
                merged = (pl.concat([base.select(common), extra_raw.select(common)]) if base is not None
                          else extra_raw.select(common))
                merged = merged.unique(subset=["timestamp"], keep="last").sort("timestamp")
                raw_1m_path.parent.mkdir(parents=True, exist_ok=True)
                merged.select(common).write_parquet(raw_1m_path)
                print(f"   💾 raw_1m 已刷新 (含当前月 daily): {merged.height} 行")
            extra_f = fetch_funding_api(SYMBOL, d_start)
            if extra_f is not None:
                main_funding = extra_f if main_funding is None else pl.concat([main_funding, extra_f]).unique(subset=["calc_time"], keep="last")
            extra = fetch_daily_fill("futures", "premiumIndexKlines", SYMBOL, "1h", d_start, d_end)
            if extra is not None:
                main_premium = extra if main_premium is None else pl.concat([main_premium, extra]).unique(subset=["exact_timestamp"], keep="last")
            if len(AUX_SYMBOLS) >= 1:
                extra = fetch_daily_fill("spot", "klines", AUX_SYMBOLS[0], "1h", d_start, d_end)
                if extra is not None:
                    aux_reference = extra if aux_reference is None else pl.concat([aux_reference, extra]).unique(subset=["exact_timestamp"], keep="last")
            if len(AUX_SYMBOLS) >= 2:
                extra = fetch_daily_fill("spot", "klines", AUX_SYMBOLS[1], "1h", d_start, d_end)
                if extra is not None:
                    direct_cross = extra if direct_cross is None else pl.concat([direct_cross, extra]).unique(subset=["exact_timestamp"], keep="last")

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

    print("⚙️ [第四阶段] 计算 39 维特征矩阵 (31 旧 + 8 V2 新, FES v1)...")
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

    # ==========================================================
    # 🚨 [特征 V2 · FES v1] 8 个新特征（Joel 批准 2026-10-30，零新数据源）
    # 因果约定（FES §0.1）：滚动统计 shift(1) 严格事前；当根 bar 字段在决策时刻已收盘可用
    # ==========================================================
    features = features.with_columns([
        # 32 taker_imbalance: 主动买/卖量原始失衡 (V_buy−V_sell)/(V_buy+V_sell+ε)，天然 [-1,1] 不 clip
        (
            (pl.col("taker_buy_volume") - (pl.col("volume") - pl.col("taker_buy_volume")))
            / (pl.col("volume") + 1e-9)
        ).alias("taker_imbalance"),
        # 33 volume_zscore 前缀列: log(1+V)，下方 24h MAD 滚动 z（FES: min_periods=16）
        (pl.col("volume") + 1.0).log().alias("volume_log1p"),
        # 34 trend_efficiency_ratio: Kaufman ER 24h，1=趋势 0=震荡，天然 [0,1] 不 clip
        (
            (pl.col("close") - pl.col("close").shift(24)).abs()
            / (pl.col("close").diff(1).abs().rolling_sum(window_size=24) + 1e-9)
        ).alias("trend_efficiency_ratio"),
        # 35 downside_volatility 前缀列: 24h 下行波动率 sqrt(mean(min(r,0)²))，下方 168h MAD z
        (
            pl.col("log_return").clip(upper_bound=0.0).pow(2).rolling_mean(window_size=24).sqrt()
        ).alias("downside_vol_24h"),
        # 38 volatility_of_volatility 前缀列: RV_1h 的 24h 标准差，下方 168h MAD z
        pl.col("realized_vol_1h").rolling_std(window_size=24).alias("vov_24h"),
        # 39 return_24h: 24h 累计对数收益（原始值入模，与 log_return 口径一致）
        (pl.col("close").log() - pl.col("close").shift(24).log()).alias("return_24h"),
    ])

    # ==========================================================
    # 🚨 [特征 V3 · 阶段 3] 7 个扩展特征（Joel 批准 2026-11-01 升 V3，零新数据源）
    # 数据源核对：count=1h K线自带交易笔数；premium_close/funding=现有管道；ETH=aux_reference_close（现货 ETHUSDT）
    # ==========================================================
    features = features.with_columns([
        # 40 volume_price_divergence 前缀×2: 24h ΔP z 与 24h Δlog(1+V) z，下方相减
        pl.col("log_return").alias("vpd_pret_24h"),
        (pl.col("volume") + 1.0).log().diff(1).alias("vpd_vret_24h"),
        # 41 trade_intensity_zscore 前缀列: log(1+N_trades)，下方 24h MAD z
        (pl.col("count") + 1.0).log().alias("count_log1p"),
        # 42 drawdown_168h: 距 168h 高点回撤 P_t/HH − 1（shift(1) 严格事前，天然 (−1, 0]）
        (
            pl.col("close") / pl.col("high").rolling_max(window_size=168).shift(1) - 1.0
        ).alias("drawdown_168h"),
        # 44 return_4h: 4h 对数收益（原始值）
        (pl.col("close").log() - pl.col("close").shift(4).log()).alias("return_4h"),
        # 45 atr_pct_24h: ATR_24h/P_t（相对值，天然 ~1-3% 量级，不 z）
        (
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            ).rolling_mean(window_size=24) / (pl.col("close") + 1e-9)
        ).alias("atr_pct_24h"),
        # 46 btc_eth_corr_24h: BTC/ETH 1h 对数收益 24h 滚动相关（当根 bar 已收盘，口径同 return 系）
        # 本 polars 版本无 rolling_corr，用协方差恒等式手工展开：cov=mean(xy)−mean(x)mean(y)，除以 std 积
        (
            (
                pl.col("log_return") * (pl.col("aux_reference_close").log().diff(1))
            ).rolling_mean(window_size=24)
            - pl.col("log_return").rolling_mean(window_size=24) * (pl.col("aux_reference_close").log().diff(1)).rolling_mean(window_size=24)
        ) / (
            pl.col("log_return").rolling_std(window_size=24)
            * (pl.col("aux_reference_close").log().diff(1)).rolling_std(window_size=24) + 1e-12
        ).alias("btc_eth_corr_24h"),
    ])
    # 40 volume_price_divergence 的派生（vpd_pzscore − vpd_vzscore）在 z 块之后执行（见下方 COID 邻块）

    print("\n📊 [第五阶段] 聚合 aggTrades 历史订单流数据...")
    symbol_agg_files = find_symbol_aggtrade_files(SYMBOL)
    if symbol_agg_files:
        # 🚨 致命缺陷修复：强行按时间去重，防范月度包与日线增量包交叉覆盖导致的行数爆炸
        # 日线增量通常比月度包更新，显式排序与优先级保证同一小时保留结果可复现。
        parquet_files = symbol_agg_files
        required_agg_cols = {
            "hour_timestamp", "cvd", "whale_buy", "whale_sell",
            "whale_net_flow", "retail_ratio", "whale_ls_ratio"
        }
        agg_float_cols = [
            "cvd", "whale_buy", "whale_sell", "whale_net_flow",
            "retail_vol", "retail_ratio", "whale_ls_ratio"
        ]
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

        def read_agg_parquet(f: Path) -> pl.DataFrame:
            df = pl.read_parquet(f)
            casts = [
                pl.col(col).cast(pl.Float64, strict=False)
                for col in agg_float_cols
                if col in df.columns
            ]
            return df.with_columns(
                casts + [pl.lit(1 if "_daily_" in f.name else 0).alias("_source_priority")]
            )

        dfs_agg = [read_agg_parquet(f) for f in parquet_files]
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
        print(f"⚠️ 警告：未找到 {SYMBOL} 的 aggTrades Parquet 文件，将跳过订单流维度。")

    # 🚨 时序平稳化：去除高度共线的 net_taker_vol，保留 vpt 和 cvd
    print("🌊 [时序平稳化] 正在对高维度绝对量特征进行 Robust Z-Score (MAD) 降维与截断...")
    # 🚨 [剔除] vpt，因其逻辑已被更敏锐的 cvd_zscore 覆盖
    # FES v1（Joel 批准 2026-10-30）：原 :779-795 的 168h MAD 滚动 z 抽成参数化作业表，
    # 旧 6 列保持 (168, 120) 原参数——计算序列与原逻辑位级一致（回归验证 max|Δ|<1e-9 兜底）
    rz_jobs = [
        # (源列, 窗口, min_samples, 输出列名)
        ("realized_vol_1h", 168, 120, "realized_vol_1h_zscore"),
        ("amihud_illiquidity", 168, 120, "amihud_illiquidity_zscore"),
        ("oi_delta_pct", 168, 120, "oi_delta_pct_zscore"),
        # V2 中间列: log_return 的 168h z，仅用于 oi_price_divergence（FES §2-37），算完即弃
        ("log_return", 168, 120, "log_return_zscore"),
    ]
    if "cvd" in features.columns:
        rz_jobs.extend([
            ("cvd", 168, 120, "cvd_zscore"),
            ("whale_net_flow", 168, 120, "whale_net_flow_zscore"),
            ("whale_ls_ratio", 168, 120, "whale_ls_ratio_zscore"),
        ])
    # V2 新 z 列（FES §2）: volume 24h 窗口(min 16)；downside/vov/funding 168h 窗口
    rz_jobs.extend([
        ("volume_log1p", 24, 16, "volume_zscore"),
        ("downside_vol_24h", 168, 120, "downside_volatility"),
        ("last_funding_rate", 168, 120, "funding_rate_zscore"),
        ("vov_24h", 168, 120, "volatility_of_volatility"),
    ])
    # V3 新 z 列（阶段 3）: vpd 双前缀 24h；count 24h；premium 168h
    rz_jobs.extend([
        ("vpd_pret_24h", 24, 16, "vpd_pzscore"),
        ("vpd_vret_24h", 24, 16, "vpd_vzscore"),
        ("count_log1p", 24, 16, "trade_intensity_zscore"),
        ("premium_close", 168, 120, "basis_zscore"),
    ])

    # 步骤 1: 计算 rolling median (严格事前：shift(1) 后窗口只覆盖 t-1 及以前)
    rz_medians = [f"__rz_median_{alias}" for _, _, _, alias in rz_jobs]
    rz_mads = [f"__rz_mad_{alias}" for _, _, _, alias in rz_jobs]
    features = features.with_columns([
        pl.col(src).rolling_median(window, min_samples=min_samples).shift(1).alias(f"__rz_median_{alias}")
        for src, window, min_samples, alias in rz_jobs
    ])
    # 步骤 2: 计算 rolling MAD (Median Absolute Deviation)，同样 shift(1)
    features = features.with_columns([
        (pl.col(src) - pl.col(f"__rz_median_{alias}")).abs()
        .rolling_median(window, min_samples=min_samples).shift(1).alias(f"__rz_mad_{alias}")
        for src, window, min_samples, alias in rz_jobs
    ])
    # 步骤 3: 计算 Robust Z-Score 并截断到 [-4, 4]
    zscore_exprs = [
        (0.6745 * (pl.col(src) - pl.col(f"__rz_median_{alias}")) / (pl.col(f"__rz_mad_{alias}") + 1e-9)).clip(-4.0, 4.0).alias(alias)
        for src, window, min_samples, alias in rz_jobs
    ]
    features = features.with_columns(zscore_exprs)

    # 步骤 4: 丢弃中间临时变量（vpd_*zscore 留到 volume_price_divergence 算完后丢弃）
    features = features.drop(rz_medians + rz_mads + ["volume_log1p", "downside_vol_24h", "vov_24h", "count_log1p"])

    # 🚨 [新增] COID: 高阶订单流背离 (CVD净买单偏离度减去总持仓偏离度)
    if "cvd_zscore" in features.columns and "oi_delta_pct_zscore" in features.columns:
        features = features.with_columns(
            (pl.col("cvd_zscore") - pl.col("oi_delta_pct_zscore")).alias("cvd_oi_divergence")
        )

    # 🚨 [特征 V2 · FES v1] 37 oi_price_divergence: 价格收益 z − OI 变化 z（各自 168h MAD z 后相减，合成值不 clip）
    if "log_return_zscore" in features.columns and "oi_delta_pct_zscore" in features.columns:
        features = features.with_columns(
            (pl.col("log_return_zscore") - pl.col("oi_delta_pct_zscore")).alias("oi_price_divergence")
        ).drop("log_return_zscore")  # 中间列用完即弃，不进最终矩阵

    # 🚨 [特征 V3 · 阶段 3] 40 volume_price_divergence: 价格 z − 量能 z（各自 24h MAD z 后相减，合成值不 clip）
    if "vpd_pzscore" in features.columns and "vpd_vzscore" in features.columns:
        features = features.with_columns(
            (pl.col("vpd_pzscore") - pl.col("vpd_vzscore")).alias("volume_price_divergence")
        ).drop(["vpd_pzscore", "vpd_vzscore", "vpd_pret_24h", "vpd_vret_24h"])

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

    # 🚨 [特征 V2 · FES v1] 8 个新列 (32-39) 按 FES §2 固定顺序追加在尾部
    final_cols.extend([
        "taker_imbalance", "volume_zscore", "trend_efficiency_ratio", "downside_volatility",
        "funding_rate_zscore", "oi_price_divergence", "volatility_of_volatility", "return_24h",
    ])
    # 🚨 [特征 V3 · 阶段 3] 7 个扩展列 (40-46)
    final_cols.extend([
        "volume_price_divergence", "trade_intensity_zscore", "drawdown_168h", "basis_zscore",
        "return_4h", "atr_pct_24h", "btc_eth_corr_24h",
    ])

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

    # 空值区间诊断：对有空值的列输出空值所在时间区间 (区分头部预热 vs 尾部数据缺口)
    ts_sorted = features.sort("hour_timestamp")
    for col in [c for c in features.columns if ts_sorted[c].null_count() > 0]:
        ts = ts_sorted.filter(pl.col(col).is_null())["hour_timestamp"].to_list()
        if not ts:
            continue
        ranges = []
        start = prev = ts[0]
        for t in ts[1:]:
            if (t - prev).total_seconds() <= 3600 * 2:
                prev = t
            else:
                ranges.append((start, prev))
                start = prev = t
        ranges.append((start, prev))
        desc = ", ".join(f"{a}~{b}" for a, b in ranges[:3])
        more = f" (+{len(ranges) - 3} 个区间)" if len(ranges) > 3 else ""
        print(f"   ⚠️ {col} 空值 {len(ts)} 处 | {desc}{more}")

    # 无漏网之鱼，安全执行空值剥离
    before_rows = features.height
    features = features.drop_nulls()
    after_rows = features.height
    drop_ratio = ((before_rows - after_rows) / before_rows) if before_rows else 1.0
    print(f"🧹 drop_nulls 清洗完成: 删除 {before_rows - after_rows} 行，剩余 {after_rows} 行，删除比例 {drop_ratio:.2%}")
    if drop_ratio > 0.2:
        print("⚠️ 警告：缺失清洗删除比例超过 20%，建议检查数据源完整性后再训练。")
    
    current_date = datetime.now().strftime("%Y%m%d")
    # 👇 [特征 V3 · 阶段 3] 46d 矩阵写入独立目录 main_features_46d/，
    # 39d/31d 矩阵文件保留不动（消融分组在训练端按列选取，共用 46d 矩阵）
    out_dir = PROCESSED_MAIN_DIR.parent / "main_features_46d" / SYMBOL
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{SYMBOL}_Master_Training_Matrix_{horizon}h_46d_{current_date}.parquet"
    output_path = out_dir / file_name
    features.write_parquet(output_path)
    print(f"\n✅ 终极宽表构建成功！路径: {output_path} | 最终规模: {features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='构建加密货币量化特征宽表')
    parser.add_argument('--horizon', type=int, default=FORECAST_HORIZON, help='预测未来的时间窗口 (小时)')
    args = parser.parse_args()
    
    build_features(horizon=args.horizon)
