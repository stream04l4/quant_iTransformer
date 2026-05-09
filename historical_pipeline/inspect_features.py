import sys
import polars as pl
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR

def inspect_master_matrix():
    current_date = datetime.now().strftime("%Y%m%d")
    file_path = PROCESSED_MAIN_DIR / f"{SYMBOL}_Master_Training_Matrix_{current_date}.parquet"
    if not file_path.exists():
        print("❌ 错误：终极宽表不存在，请先运行 build_main_features.py")
        return

    df = pl.read_parquet(file_path)
    total_cols = len(df.columns)
    
    print(f"🔍 检查矩阵: {file_path.name}")
    print(f"   | 行数: {len(df)}")
    # 客观判定维度是否达到预期的 25 维
    if total_cols == 25:
        print(f"   | 列数: {total_cols} (✅ 满载状态：包含基础特征与订单流)")
    elif total_cols == 22:
        print(f"   | 列数: {total_cols} (⚠️ 降级状态：缺少 aggTrades 订单流特征)")
    else:
        print(f"   | 列数: {total_cols} (❌ 异常状态：维度不符合预期的 22 或 25，请排查特征生成逻辑)")
    
    # 1. 检查是否存在 NaN (模型杀手)
    null_counts = df.null_count().sum_horizontal()[0]
    if null_counts > 0: 
        print(f"❌ 警告：发现 {null_counts} 个空值！")
    else: 
        print("✅ 数据完整性：100% (无空值)")

    # 2. 检查横截面相关性
    if "log_return" in df.columns and "eth_log_return" in df.columns:
        corr = df.select(pl.corr("log_return", "eth_log_return")).to_series()[0]
        print(f"🔗 市场一致性检查 (BTC/ETH Correlation): {corr:.4f}")

    # 3. 打印数据片段 (Data Snippet) - 动态展示前 5 个和后 5 个特征
    display_cols = df.columns[:5] + ["..."] + df.columns[-5:] if total_cols > 10 else df.columns
    print("\n" + "="*80)
    print("📄 数据片段预览 (全量维度)")
    print("="*80)
    print("--- 头部前 5 行 (Head) ---")
    print(df.head(5))
    
    print("\n--- 尾部后 5 行 (Tail) ---")
    print(df.tail(5))

    # 4. 统计审查结论 (Statistical Review)
    print("\n" + "="*80)
    print("📊 统计审查结论 (Descriptive Statistics) - 针对所有数值特征")
    print("="*80)
    
    # 自动过滤掉非数值列（如时间戳）以生成统计表
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    stats_df = df.select(numeric_cols).describe()
    
    # 由于 25 维太宽，设置 polars 打印配置使其不折叠，方便在终端滑动查看
    with pl.Config(tbl_cols=-1, tbl_width_chars=150):
        print(stats_df)
    
    print("\n💡 审查要点提示:")
    print("- 【异常值】检查 `min` 和 `max`，如 `net_taker_vol` 或 `cvd` 是否出现物理上不可能的极值。")
    print("- 【标度统一】注意各特征量级差异。如果使用线性模型或 NN，需要确认后续流程是否有 Scaling (归一化/标准化) 机制，基于树的模型 (XGBoost/LightGBM) 则无需过度担心。")
    print("- 【周期特征】`hour_sin`, `hour_cos`, `week_sin`, `week_cos` 的范围必须严格在 -1 到 1 之间。")
    print("\n✅ 终端数据诊断完毕。")

if __name__ == "__main__":
    inspect_master_matrix()