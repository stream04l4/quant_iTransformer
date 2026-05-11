import sys
import polars as pl
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# 🚨 引入 TRAIN_DATA_DIR 以便定位到 data/train 文件夹
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, TRAIN_DATA_DIR

def inspect_master_matrix():
    # 自动定位最新生成的文件
    file_pattern = f"{SYMBOL}_Master_Training_Matrix_*.parquet"
    files = sorted(list(PROCESSED_MAIN_DIR.glob(file_pattern)))
    
    if not files:
        print("❌ 错误：终极宽表不存在，请先运行 build_main_features.py")
        return
        
    file_path = files[-1]
    df = pl.read_parquet(file_path)
    total_cols = len(df.columns)
    
    print(f"🔍 检查矩阵: {file_path.name}")
    print(f"   | 行数: {len(df)}")
    
    # 🚨 新增：提取并打印首尾时间戳
    start_time = df.select(pl.col("hour_timestamp").min()).item()
    end_time = df.select(pl.col("hour_timestamp").max()).item()
    print(f"   | 📅 数据跨度: {start_time}  至  {end_time}")
    
    if total_cols == 31:
        print(f"   | 列数: {total_cols} (✅ 满载状态：包含全部 Alpha 与订单流)")
    elif total_cols == 28:
        print(f"   | 列数: {total_cols} (⚠️ 降级状态：缺少 aggTrades 微观订单流特征)")
    else:
        print(f"   | 列数: {total_cols} (❌ 异常状态：维度不符合预期的 25 或 28！)")
    
    # 1. 检查是否存在 NaN
    null_counts = df.null_count().sum_horizontal()[0]
    if null_counts > 0: 
        print(f"❌ 警告：发现 {null_counts} 个空值，请检查前置填充逻辑！")
    else: 
        print("✅ 数据完整性：100% (无空值)")

    # 2. 统计审查计算
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    stats_df = df.select(numeric_cols).describe()
    
    with pl.Config(tbl_cols=-1, tbl_width_chars=180, float_precision=4):
        print("\n📊 统计审查结论已计算完毕，正在终端预览...")
        print(stats_df)
    
    # ==========================================
    # 🚨 新增：导出至 data/train 文件夹
    # ==========================================
    current_date = datetime.now().strftime("%Y%m%d")
    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True) # 确保目录存在
    
    # 导出 1：统计特征报告 CSV
    stats_csv_path = TRAIN_DATA_DIR / f"{SYMBOL}_Feature_Stats_{current_date}.csv"
    stats_df.write_csv(stats_csv_path)
    
    # 导出 2：头尾真实数据切片 CSV (前100行 + 后100行)
    sample_csv_path = TRAIN_DATA_DIR / f"{SYMBOL}_Feature_Sample_{current_date}.csv"
    if len(df) > 200:
        sample_df = pl.concat([df.head(100), df.tail(100)])
    else:
        sample_df = df
    
    # 特别处理时间戳，方便在 Excel 中阅读
    sample_df = sample_df.with_columns(
        pl.col("hour_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S").alias("hour_timestamp")
    )
    sample_df.write_csv(sample_csv_path)

    print("\n" + "="*80)
    print("💾 【本地审计文件已生成】")
    print(f"📄 1. 维度统计审查表: {stats_csv_path}")
    print(f"📄 2. 真实数据抽样表 (200行): {sample_csv_path}")
    print("="*80)
    
    print("\n💡 去 Excel 审查建议:")
    print("1. 打开 Stats 表，找到所有 `_zscore` 列，看均值是否卡在 0 附近，max 是否在 4.0，min 是否在 -4.0。")
    print("2. 打开 Sample 表，拉到最右侧看 `target_24h_return`，正负分布是否均匀，有没有出现单边极端的现象。")

if __name__ == "__main__":
    inspect_master_matrix()