import sys
import polars as pl

# 🚨 核心修复：必须在导入 matplotlib.pyplot 之前，强制指定无头后端 'Agg'
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, TRAIN_DATA_DIR

def inspect_master_matrix():
    current_date = datetime.now().strftime("%Y%m%d")
    file_path = PROCESSED_MAIN_DIR / f"{SYMBOL}_Master_Training_Matrix_{current_date}.parquet"
    if not file_path.exists():
        print("❌ 错误：终极宽表不存在，请先运行 build_main_features.py")
        return

    df = pl.read_parquet(file_path)
    print(f"🔍 检查矩阵: {file_path.name} | 行数: {len(df)} | 列数: {len(df.columns)}")
    
    # 1. 检查是否存在 NaN (模型杀手)
    null_counts = df.null_count().sum_horizontal()[0]
    if null_counts > 0: print(f"❌ 警告：发现 {null_counts} 个空值！")
    else: print("✅ 数据完整性：100% (无空值)")

    # 2. 检查横截面相关性
    corr = df.select(pl.corr("log_return", "eth_log_return")).to_series()[0]
    print(f"🔗 市场一致性检查 (BTC/ETH Correlation): {corr:.4f}")

    # 3. 统计学分布可视化
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Ultimate Feature Matrix Distribution Check", fontsize=16)

    # A. 收益率 (Log Return) - 应该正态分布
    sns.histplot(df['log_return'], bins=100, kde=True, ax=axes[0,0], color='blue')
    axes[0,0].set_title("1H Log Return")

    # B. 订单流 (CVD) - 反映微观博弈
    sns.histplot(df['cvd'], bins=100, kde=True, ax=axes[0,1], color='orange')
    axes[0,1].set_title("1H CVD (AggTrades)")

    # C. 巨鲸流 (Whale Flow)
    sns.histplot(df['whale_net_flow'], bins=100, kde=True, ax=axes[0,2], color='red')
    axes[0,2].set_title("Whale Net Flow")

    # D. 散户率 (Retail Ratio) - 0.05-0.3 之间
    sns.histplot(df['retail_ratio'], bins=50, kde=True, ax=axes[1,0], color='green')
    axes[1,0].set_title("Retail Ratio")

    # E. 标签 (Target Return) - 模型要学的东西
    sns.histplot(df['target_24h_return'], bins=100, kde=True, ax=axes[1,1], color='purple')
    axes[1,1].set_title("Target 24H Future Return")

    # F. 资金费率 (Funding Rate)
    sns.histplot(df['funding_rate'], bins=50, kde=True, ax=axes[1,2], color='brown')
    axes[1,2].set_title("Funding Rate")

    plt.tight_layout()
    
    # 🚨 核心修复：保存为高分辨率的 PNG 图片，而不是调用 plt.show()
    save_path = TRAIN_DATA_DIR / "feature_inspection.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n📈 完美！诊断分布图已渲染完毕，已存入服务器本地：\n   {save_path}")
    print("   请使用 SSH 工具 (如 MobaXterm, Xshell 或 VSCode) 将该图片拉取到你自己的电脑上查看。")

if __name__ == "__main__":
    inspect_master_matrix()