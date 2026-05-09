import os
import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path
import json
# 引入之前的模型架构和配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, SEQ_LEN, D_MODEL, E_LAYERS, DROPOUT, CLASS_THRESHOLD, BATCH_SIZE, EPOCHS, EXPERIMENT_ID
from historical_pipeline.train_itransformer import InvertedTransformer

def check_distribution():
    data_files = sorted(list(PROCESSED_MAIN_DIR.glob("*_Master_Training_Matrix_*.parquet")))
    model_files = sorted(list(MODEL_DIR.glob(f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth")))
    
    if not data_files or not model_files:
        print("❌ 找不到数据或模型文件！")
        return
        
    data_path = data_files[-1]
    model_path = model_files[-1]
    print(f"🔍 正在读取模型: {model_path.name}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    feature_cols = checkpoint['feature_cols']
    
    dynamic_n_heads = max(2, D_MODEL // 16) 
    dynamic_d_ff = D_MODEL * 4

    model = InvertedTransformer(
        num_variates=len(feature_cols), 
        seq_len=SEQ_LEN,
        d_model=D_MODEL,     
        n_heads=dynamic_n_heads, # 🚨 自适应：比如 d_model=64 时为 4，d_model=32 时为 2
        d_ff=dynamic_d_ff,       # 🚨 自适应：比如 d_model=64 时为 256，d_model=32 时为 128
        e_layers=E_LAYERS,   
        dropout=DROPOUT      
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pl.read_parquet(data_path)
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    
    # 严格取最后 20% 的样本外数据进行统计，防止拿训练集作弊
    split_idx = int(len(X_raw) * 0.8)
    X_test_raw = X_raw[split_idx:]
    
    print("\n⚙️ 正在 H100 上全量推断样本外概率分布...")
    p_down_list, p_flat_list, p_up_list = [], [], []
    
    with torch.no_grad():
        for i in range(len(X_test_raw) - SEQ_LEN):
            seq_x = X_test_raw[i : i + SEQ_LEN]
            tensor_x = torch.from_numpy(seq_x).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
            p_down_list.append(probs[0])
            p_flat_list.append(probs[1])
            p_up_list.append(probs[2])

    p_down = np.array(p_down_list)
    p_flat = np.array(p_flat_list)
    p_up = np.array(p_up_list)

    print("\n📊 模型输出概率分位数统计 (OOS):")
    print("=" * 60)
    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    
    print(f"{'分位数 (Top %)':<15} | {'看跌概率 (p_down)':<15} | {'震荡概率 (p_flat)':<15} | {'看涨概率 (p_up)':<15}")
    print("-" * 60)
    
    for p in percentiles:
        # 注意：分位数越高，代表这个概率值越大 (例如 Top 1% 的极值)
        v_down = np.percentile(p_down, p)
        v_flat = np.percentile(p_flat, p)
        v_up = np.percentile(p_up, p)
        
        # 换算为 Top % 的概念，例如 99 percentile 就是 Top 1%
        top_str = f"Top {100 - p:.1f}%" if p != 100 else "Max (最高峰值)"
        if p == 50: top_str = "Median (中位数)"
        
        print(f"{top_str:<15} | {v_down:<15.4f} | {v_flat:<15.4f} | {v_up:<15.4f}")
    
    print("=" * 60)
    # 🚨 自动化对接：将核心分位数落盘，供回测引擎调用
    thresholds = {
        "up_top_1": float(np.percentile(p_up, 99)),
        "up_top_5": float(np.percentile(p_up, 95)),
        "up_top_10": float(np.percentile(p_up, 90)),
        "down_top_1": float(np.percentile(p_down, 99)),
        "down_top_5": float(np.percentile(p_down, 95)),
        "down_top_10": float(np.percentile(p_down, 90))
    }
    json_path = TRAIN_DATA_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f)
    print(f"💾 动态分位数阈值已生成并保存至: {json_path.name}")
    print("\n💡 解读指南:")
    print("1. 如果 '看涨概率' 的 Top 1% 只有 0.45，说明模型极度保守，根本给不出 0.8 的置信度。")
    print("2. 如果 '看跌概率' 的 Top 5% 达到了 0.85，说明模型对下跌行情的捕捉非常有自信。")
    print("3. '震荡概率' 的中位数通常会在 0.6 以上，这是正常现象。")

if __name__ == "__main__":
    check_distribution()