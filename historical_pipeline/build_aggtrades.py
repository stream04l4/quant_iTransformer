import os
import sys
import requests
import zipfile
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta, date

# ==========================================
# 0. 环境初始化与配置导入
# ==========================================
# 确保能够正确导入根目录下的 config 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入公共配置
from quant_config.config import (
    TEMP_DIR, PROCESSED_AGGTRADES_DIR, SYMBOL, WHALE_THRESHOLD, 
    RETAIL_THRESHOLD, DATA_START_MONTH, DATA_END_MONTH, 
    BINANCE_BASE_URL_FUTURES, BINANCE_BASE_URL_FUTURES_DAILY, AGGTRADE_COLS
)

# 脚本启动时，创建所需的文件夹 (副作用操作放在执行脚本中)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_AGGTRADES_DIR.mkdir(parents=True, exist_ok=True)

def generate_month_range(start_ym, end_ym):
    """
    输入起止年月元组，生成连续的年月。
    例如: (2025, 11) 到 (2026, 2)
    产出: (2025,11), (2025,12), (2026,1), (2026,2)
    """
    curr_y, curr_m = start_ym
    end_y, end_m = end_ym
    
    # Python 中元组是可以直接比大小的，(2022, 12) < (2023, 1) 会返回 True
    while (curr_y, curr_m) <= (end_y, end_m):
        yield curr_y, curr_m
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1

# ==========================================
# 1. 核心计算引擎 (基于 Polars)
# ==========================================
def process_csv_to_features(csv_path: Path, output_parquet_path: Path):
    """
    使用 Polars 读取超大 CSV，降维至 1 小时，提取微观订单流特征
    """
    print(f"🔄 正在使用 Polars 并发处理数据: {csv_path.name}")
    
    df = pl.read_csv(csv_path, has_header=True, skip_rows=0, new_columns=AGGTRADE_COLS)

    # 特征工程：时间戳截断与买卖量分离
    df = df.with_columns([
        pl.from_epoch(pl.col("transact_time"), time_unit="ms").dt.truncate("1h").alias("hour_timestamp"),
        pl.when(pl.col("is_buyer_maker") == False).then(pl.col("quantity")).otherwise(0).alias("buy_vol"),
        pl.when(pl.col("is_buyer_maker") == True).then(pl.col("quantity")).otherwise(0).alias("sell_vol"),
    ])

    # 按小时聚合计算
    hourly_features = df.group_by("hour_timestamp").agg([
        pl.col("quantity").sum().alias("total_volume"),
        (pl.col("buy_vol").sum() - pl.col("sell_vol").sum()).alias("cvd"),
        pl.col("buy_vol").filter(pl.col("quantity") >= WHALE_THRESHOLD).sum().alias("whale_buy"),
        pl.col("sell_vol").filter(pl.col("quantity") >= WHALE_THRESHOLD).sum().alias("whale_sell"),
        pl.col("quantity").filter(pl.col("quantity") <= RETAIL_THRESHOLD).sum().alias("retail_vol"),
    ]).with_columns([
        (pl.col("whale_buy") - pl.col("whale_sell")).alias("whale_net_flow"),
        pl.when(pl.col("total_volume") > 0)
            .then(pl.col("retail_vol") / pl.col("total_volume"))
            .otherwise(None)
            .alias("retail_ratio")
    ]).sort("hour_timestamp")

    final_df = hourly_features.select([
        "hour_timestamp", "cvd", "whale_buy", "whale_sell", 
        "whale_net_flow", "retail_vol", "retail_ratio"
    ])

    final_df.write_parquet(output_parquet_path)
    print(f"✅ 特征提取完成并落盘: {output_parquet_path.name}")


# ==========================================
# 2. 下载与文件管理逻辑
# ==========================================
def download_file(url: str, dest_path: Path) -> bool:
    print(f"⬇️ 开始下载: {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败 (可能该月数据暂未生成): {e}")
        return False

def pipeline_for_month(year: int, month: int):
    """处理单个自然月的数据流水线"""
    month_str = f"{month:02d}"
    file_name = f"{SYMBOL}-aggTrades-{year}-{month_str}.zip"
    download_url = f"{BINANCE_BASE_URL_FUTURES}/aggTrades/{SYMBOL}/{file_name}"
    
    zip_path = TEMP_DIR / file_name
    csv_name = f"{SYMBOL}-aggTrades-{year}-{month_str}.csv"
    csv_path = TEMP_DIR / csv_name
    output_parquet = PROCESSED_AGGTRADES_DIR / f"{SYMBOL}_aggTrades_features_{year}_{month_str}.parquet"

    if output_parquet.exists():
        print(f"⏭️ {output_parquet.name} 已存在，跳过处理。")
        return

    if not download_file(download_url, zip_path):
        return

    print(f"📦 正在解压: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
    except zipfile.BadZipFile:
        print(f"❌ 压缩包损坏: {zip_path.name}")
        os.remove(zip_path)
        return

    os.remove(zip_path)
    print(f"🗑️ 已删除压缩包: {zip_path.name}")

    if csv_path.exists():
        process_csv_to_features(csv_path, output_parquet)
        os.remove(csv_path)
        print(f"🗑️ 已删除原始 CSV: {csv_path.name}")
        print("-" * 50)
    else:
        print(f"❌ 未找到解压后的 CSV 文件: {csv_path.name}")

def pipeline_for_day(d: date):
    date_str = d.strftime("%Y-%m-%d")
    file_name = f"{SYMBOL}-aggTrades-{date_str}.zip"
    download_url = f"{BINANCE_BASE_URL_FUTURES_DAILY}/aggTrades/{SYMBOL}/{file_name}"
    
    zip_path = TEMP_DIR / file_name
    csv_name = f"{SYMBOL}-aggTrades-{date_str}.csv"
    csv_path = TEMP_DIR / csv_name
    output_parquet = PROCESSED_AGGTRADES_DIR / f"{SYMBOL}_aggTrades_daily_{date_str}.parquet"

    if output_parquet.exists():
        return
    if not download_file(download_url, zip_path):
        return
    
    print(f"📦 正在解压日线数据: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        os.remove(zip_path)
        if csv_path.exists():
            process_csv_to_features(csv_path, output_parquet)
            os.remove(csv_path)
    except Exception as e:
        print(f"❌ 处理日线压缩包异常: {e}")
        if zip_path.exists(): os.remove(zip_path)
# ==========================================
# 3. 主函数执行口
# ==========================================
if __name__ == "__main__":
    print(f"🚀 开始为 {SYMBOL} 执行历史 aggTrades 订单流特征构建管道...")
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # 1. 拉取历史月度底座数据 (遇到当前月自动跳过)
    for year, month in generate_month_range(DATA_START_MONTH, DATA_END_MONTH):
        if year > today.year or (year == today.year and month >= today.month):
            break 
        pipeline_for_month(year, month)

    # 2. 拉取当前月份的每日增量数据 (从 1 号拉取到昨天)
    if tuple(DATA_END_MONTH) >= (today.year, today.month):
        current_date = date(today.year, today.month, 1)
        while current_date <= yesterday:
            pipeline_for_day(current_date)
            current_date += timedelta(days=1)

    print(f"🎉 所有处理完毕！检查目录: {PROCESSED_AGGTRADES_DIR}")