import sys
import polars as pl
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
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
    
    start_time = df.select(pl.col("hour_timestamp").min()).item()
    end_time = df.select(pl.col("hour_timestamp").max()).item()
    print(f"   | 📅 数据跨度: {start_time}  至  {end_time}")
    
    # 动态匹配实际的 Target 字段名称
    target_return_col = next((c for c in df.columns if c.startswith("target_") and c.endswith("_return")), "未知 target_return")
    target_zscore_col = next((c for c in df.columns if c.startswith("target_") and c.endswith("_zscore")), "未知 target_zscore")
    
    if total_cols >= 31:
        print(f"   | 列数: {total_cols} (✅ 满载状态：包含全部 Alpha 与订单流)")
    elif total_cols >= 28:
        print(f"   | 列数: {total_cols} (⚠️ 降级状态：缺少 aggTrades 微观订单流特征)")
    else:
        print(f"   | 列数: {total_cols} (❌ 异常状态：维度不符合预期的模型要求！)")
    
    # 1. 检查是否存在 NaN
    null_counts = df.null_count().sum_horizontal()[0]
    if null_counts > 0: 
        print(f"❌ 警告：发现 {null_counts} 个空值，请检查前置 drop_nulls 或填充逻辑！")
    else: 
        print("✅ 数据完整性：100% (无空值)")

    # 2. 统计审查计算
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    stats_df = df.select(numeric_cols).describe()
    
    with pl.Config(tbl_cols=-1, tbl_width_chars=180, float_precision=4):
        print("\n📊 统计审查结论已计算完毕，正在终端预览...")
        print(stats_df)
        
    # 🚨 新增：工业界自动化特征坍塌与失活预警
    print("\n🩺 [自动化特征异常诊断]")
    abnormal_features = False
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        range_val = col_max - col_min
        
        # 1. 检测方差/极差为 0 的绝对死特征
        if range_val == 0:
            print(f"  ❌ 致命异常: `{col}` 极差为 0 (全列均为 {col_min})，特征已完全失活！")
            abnormal_features = True
            
        # 2. 检测 Epsilon 吞噬现象 (对于非收益率的 Z-score 特征，若绝对波动极小则报警)
        elif "zscore" in col and range_val < 0.1:
            print(f"  ⚠️ 严重警告: `{col}` 作为 Z-Score 特征，极差仅为 {range_val:.6f}。可能发生 Epsilon 吞噬，模型无法提取有效梯度！")
            abnormal_features = True
            
        # 3. 统计高度稀疏的死水特征 (如 Funding Rate)
        else:
            zero_ratio = (df[col] == 0.0).sum() / len(df)
            if zero_ratio > 0.3:
                print(f"  ℹ️ 弱警告: `{col}` 包含 {zero_ratio*100:.1f}% 的绝对 0 值，在震荡市中可能存在表征失效。")

    if not abnormal_features:
        print("  ✅ 未检测到明显的特征极值坍塌或死水现象。")

    # ==========================================
    # 导出至 data/train 文件夹
    # ==========================================
    current_date = datetime.now().strftime("%Y%m%d")
    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True) 
    
    stats_csv_path = TRAIN_DATA_DIR / f"{SYMBOL}_Feature_Stats_{current_date}.csv"
    stats_df.write_csv(stats_csv_path)
    
    sample_csv_path = TRAIN_DATA_DIR / f"{SYMBOL}_Feature_Sample_{current_date}.csv"
    if len(df) > 200:
        sample_df = pl.concat([df.head(100), df.tail(100)])
    else:
        sample_df = df
    
    sample_df = sample_df.with_columns(
        pl.col("hour_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S").alias("hour_timestamp")
    )
    sample_df.write_csv(sample_csv_path)

    print("\n" + "="*80)
    print("💾 【本地审计文件已生成】")
    print(f"📄 1. 维度统计审查表: {stats_csv_path}")
    print(f"📄 2. 真实数据抽样表 (200行): {sample_csv_path}")
    print("="*80)
    
    # 🚨 修正：使用动态提取的 target 列名，取代硬编码
    print("\n💡 去 Excel 审查建议:")
    print("1. 打开 Stats 表，找到所有 `_zscore` 列，看均值是否卡在 0 附近，max 是否在 4.0，min 是否在 -4.0。")
    print(f"2. 打开 Sample 表，拉到最右侧看 `{target_return_col}` 和 `{target_zscore_col}`，正负分布是否均匀，有无异常极值。")

if __name__ == "__main__":
    inspect_master_matrix()