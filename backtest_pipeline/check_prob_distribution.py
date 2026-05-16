import os
import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 🚨 不再导入模型拓扑参数，全部从 checkpoint 获取
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, EXPERIMENT_ID, FORECAST_HORIZON

def check_distribution():
    # 1. 优先定位并加载模型，一切以模型本身的元数据为准
    model_files = sorted(list(MODEL_DIR.glob(f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth")))
    if not model_files:
        print("❌ 找不到对应的模型权重文件！")
        return
        
    model_path = model_files[-1]
    print(f"🔍 正在读取模型: {model_path.name}")

    DEVICE = torch.device("cuda") # 明确纯 GPU 环境
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # 提取模型元数据
    ckpt_horizon = checkpoint.get('forecast_horizon', FORECAST_HORIZON)
    ckpt_seq_len = checkpoint.get('seq_len')
    ckpt_d_model = checkpoint.get('d_model')
    ckpt_n_heads = checkpoint.get('n_heads')
    ckpt_d_ff = checkpoint.get('d_ff')
    ckpt_e_layers = checkpoint.get('e_layers')
    ckpt_dropout = checkpoint.get('dropout')
    feature_cols = checkpoint['feature_cols']

    # 2. 🚨 核心修复 1：基于模型真实的 horizon，去精准匹配对应的特征数据表
    data_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"{SYMBOL}_Master_Training_Matrix_{ckpt_horizon}h_*.parquet")))
    if not data_files:
        print(f"❌ 找不到对应 Horizon ({ckpt_horizon}h) 的数据文件！")
        return
    data_path = data_files[-1]
    print(f"🔍 精准匹配到数据源: {data_path.name}")

    # 3. 动态构建模型架构
    from historical_pipeline.train_itransformer import InvertedTransformer
    model = InvertedTransformer(
        num_variates=len(feature_cols), 
        seq_len=ckpt_seq_len,      
        d_model=ckpt_d_model,     
        n_heads=ckpt_n_heads,
        d_ff=ckpt_d_ff,       
        e_layers=ckpt_e_layers,   
        dropout=ckpt_dropout      
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ 模型组装完毕 | 预测窗口={ckpt_horizon}h | 感受野={ckpt_seq_len}")

    # 4. 加载并切分数据
    df = pl.read_parquet(data_path)
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    
    # 🚨 核心修复 2：严格对齐训练集的 Purged Split，绝不多吃一口未来数据
    split_idx = int(len(X_raw) * 0.8)
    baseline_end = split_idx - ckpt_horizon
    X_baseline = X_raw[:baseline_end]
    
    print(f"\n⚙️ H100 极速推断启动 (基准集大小: {len(X_baseline)} 行)...")
    
    X_tensor = torch.tensor(X_baseline, dtype=torch.float32, device=DEVICE)
    # 保持 .contiguous() 确保 GPU 内存寻址效率
    X_windows = X_tensor.unfold(0, ckpt_seq_len, 1).transpose(1, 2).contiguous()
    
    predictions_probs_chunks = []
    INFERENCE_BATCH_SIZE = 4096
    
    with torch.no_grad():
        for i in range(0, len(X_windows), INFERENCE_BATCH_SIZE):
            batch_x = X_windows[i : i + INFERENCE_BATCH_SIZE]
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions_probs_chunks.append(probs)

    predictions_probs_arr = np.vstack(predictions_probs_chunks)
    p_down, p_flat, p_up = predictions_probs_arr[:, 0], predictions_probs_arr[:, 1], predictions_probs_arr[:, 2]

    print("\n📊 模型输出概率分位数统计 (基于严格纯净历史区间):")
    print("=" * 60)
    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    print(f"{'分位数 (Top %)':<15} | {'看跌概率 (p_down)':<15} | {'震荡概率 (p_flat)':<15} | {'看涨概率 (p_up)':<15}")
    print("-" * 60)
    
    for p in percentiles:
        v_down, v_flat, v_up = np.percentile(p_down, p), np.percentile(p_flat, p), np.percentile(p_up, p)
        top_str = f"Top {100 - p:.1f}%" if p != 100 else "Max (峰值)"
        if p == 50: top_str = "Median (中位数)"
        print(f"{top_str:<15} | {v_down:<15.4f} | {v_flat:<15.4f} | {v_up:<15.4f}")
    
    print("=" * 60)
    
    # 🚨 核心修复 3：打包元数据，生成可追溯的工程级 JSON
    output_payload = {
        "metadata": {
            "symbol": SYMBOL,
            "experiment_id": EXPERIMENT_ID,
            "model_file": model_path.name,
            "data_file": data_path.name,
            "forecast_horizon": ckpt_horizon,
            "seq_len": ckpt_seq_len,
            "baseline_samples": len(X_baseline),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "thresholds": {
            "up_top_0.1": float(np.percentile(p_up, 99.9)),
            "up_top_1": float(np.percentile(p_up, 99)),
            "up_top_5": float(np.percentile(p_up, 95)),
            "up_top_10": float(np.percentile(p_up, 90)),
            "down_top_0.1": float(np.percentile(p_down, 99.9)),
            "down_top_1": float(np.percentile(p_down, 99)),
            "down_top_5": float(np.percentile(p_down, 95)),
            "down_top_10": float(np.percentile(p_down, 90))
        }
    }
    
    json_path = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4) # indent=4 保证极佳的可读性
    print(f"💾 实盘阈值配置 (含元数据) 已生成: {json_path.name}")

if __name__ == "__main__":
    check_distribution()