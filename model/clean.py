import os
import glob
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import MODEL_DIR

# 这是我们要保留的 5 个精英模型的 ID
SAFE_IDS = [
    "65FE6D05",
    "88265D82",
    "501A8B93",
    "7F8A7E79",
    "F76B3A97",
    "C948E9A7",
    "D726B98C",
    "2C2F9681",
    "80F2A0DE",
    "1A86D81B",
    "E8A6060C"
]
DELETE_PATTERNS = [
    "dynamic_thresholds_EXP_*.json",
    "itransformer_BTCUSDT_best_EXP_*.pth",
    "backtest_probabilistic_EXP_*.png",
]


def extract_exp_id(file_name):
    match = re.search(r"EXP_([A-Za-z0-9]+)", file_name)
    return match.group(1) if match else None


def cleanup_models(directory, safe_ids):
    all_files = []
    for pattern in DELETE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(directory, pattern)))

    all_files = sorted(set(all_files))
    
    deleted_count = 0
    kept_count = 0
    deleted_groups = {}
    kept_groups = {}
    
    print(f"开始清理目录: {directory}")
    print(f"计划保留的 ID: {safe_ids}")
    print(f"清理文件类型: {', '.join(DELETE_PATTERNS)}")
    print("-" * 50)

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        exp_id = extract_exp_id(file_name)
        
        # 检查当前文件名是否属于任何一个保留的 EXP ID
        should_keep = exp_id in safe_ids
        
        if should_keep:
            print(f"[保留] {file_name}")
            kept_count += 1
            kept_groups.setdefault(exp_id, 0)
            kept_groups[exp_id] += 1
        else:
            try:
                os.remove(file_path)
                # print(f"[已删除] {file_name}") # 如果想看详细删除列表可以取消注释
                deleted_count += 1
                deleted_groups.setdefault(exp_id or "UNKNOWN", 0)
                deleted_groups[exp_id or "UNKNOWN"] += 1
            except Exception as e:
                print(f"[错误] 无法删除 {file_name}: {e}")

    print("-" * 50)
    print(f"清理完成！")
    print(f"保留文件数量: {kept_count} ({len(kept_groups)} 个 EXP)")
    print(f"删除文件数量: {deleted_count} ({len(deleted_groups)} 个 EXP)")

if __name__ == "__main__":
    # 确保文件夹存在
    if os.path.exists(MODEL_DIR):
        cleanup_models(MODEL_DIR, SAFE_IDS)
    else:
        print(f"错误：找不到目录 {MODEL_DIR}，请检查路径设置是否正确。")
