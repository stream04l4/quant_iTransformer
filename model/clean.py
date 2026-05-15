import os
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import MODEL_DIR

# 这是我们要保留的 5 个精英模型的 ID
SAFE_IDS = [
    "07373A9A",
    "1626B0D2",
    "B84CD094",
    "55D81945",
    "94D14540"
]
# ===========================================

def cleanup_models(directory, safe_ids):
    # 构建匹配模式：匹配所有以 itransformer_BTCUSDT_best_EXP_ 开头且以 .pth 结尾的文件
    pattern = os.path.join(directory, "itransformer_BTCUSDT_best_EXP_*.pth")
    #pattern = os.path.join(directory, "dynamic_thresholds_EXP_*.json")
    #pattern = os.path.join(directory, "backtest_probabilistic_EXP_*.png")
    all_files = glob.glob(pattern)
    
    deleted_count = 0
    kept_count = 0
    
    print(f"开始清理目录: {directory}")
    print(f"计划保留的 ID: {safe_ids}")
    print("-" * 50)

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # 检查当前文件名是否包含任何一个保留的 ID
        should_keep = any(safe_id in file_name for safe_id in safe_ids)
        
        if should_keep:
            print(f"[保留] {file_name}")
            kept_count += 1
        else:
            try:
                os.remove(file_path)
                # print(f"[已删除] {file_name}") # 如果想看详细删除列表可以取消注释
                deleted_count += 1
            except Exception as e:
                print(f"[错误] 无法删除 {file_name}: {e}")

    print("-" * 50)
    print(f"清理完成！")
    print(f"保留模型数量: {kept_count}")
    print(f"删除模型数量: {deleted_count}")

if __name__ == "__main__":
    # 确保文件夹存在
    if os.path.exists(MODEL_DIR):
        cleanup_models(MODEL_DIR, SAFE_IDS)
    else:
        print(f"错误：找不到目录 {MODEL_DIR}，请检查路径设置是否正确。")