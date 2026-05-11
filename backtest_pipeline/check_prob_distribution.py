import os
import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, SEQ_LEN, D_MODEL, E_LAYERS, DROPOUT, EXPERIMENT_ID

# 构造一个极简的数据集用于高速批量推断
class FastSeqDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len]

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
    
    # 动态构建模型架构以对齐训练时的情况
    from historical_pipeline.train_itransformer import InvertedTransformer
    feature_cols = checkpoint['feature_cols']
    dynamic_n_heads = max(2, D_MODEL // 16) 
    dynamic_d_ff = D_MODEL * 4

    model = InvertedTransformer(
        num_variates=len(feature_cols), 
        seq_len=SEQ_LEN,
        d_model=D_MODEL,     
        n_heads=dynamic_n_heads,
        d_ff=dynamic_d_ff,       
        e_layers=E_LAYERS,   
        dropout=DROPOUT      
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pl.read_parquet(data_path)
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    
    # 🚨 核心修复 1：消灭未来函数！使用前 80% (训练验证集) 的历史数据来确立阈值标准
    split_idx = int(len(X_raw) * 0.8)
    X_baseline = X_raw[:split_idx]
    
    print(f"\n⚙️ 正在 H100 上执行批量高速推断 (历史基准集大小: {len(X_baseline)})...")
    
    # 🚨 核心修复 2：使用 DataLoader 进行大批量并行推断
    dataset = FastSeqDataset(X_baseline, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    
    p_down_list, p_flat_list, p_up_list = [], [], []
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
            p_down_list.extend(probs[:, 0])
            p_flat_list.extend(probs[:, 1])
            p_up_list.extend(probs[:, 2])

    p_down = np.array(p_down_list)
    p_flat = np.array(p_flat_list)
    p_up = np.array(p_up_list)

    print("\n📊 模型输出概率分位数统计 (基于历史基准防止未来函数):")
    print("=" * 60)
    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    
    print(f"{'分位数 (Top %)':<15} | {'看跌概率 (p_down)':<15} | {'震荡概率 (p_flat)':<15} | {'看涨概率 (p_up)':<15}")
    print("-" * 60)
    
    for p in percentiles:
        v_down = np.percentile(p_down, p)
        v_flat = np.percentile(p_flat, p)
        v_up = np.percentile(p_up, p)
        
        top_str = f"Top {100 - p:.1f}%" if p != 100 else "Max (最高峰值)"
        if p == 50: top_str = "Median (中位数)"
        
        print(f"{top_str:<15} | {v_down:<15.4f} | {v_flat:<15.4f} | {v_up:<15.4f}")
    
    print("=" * 60)
    
    thresholds = {
        "up_top_0.1": float(np.percentile(p_up, 99.9)),
        "up_top_1": float(np.percentile(p_up, 99)),
        "up_top_5": float(np.percentile(p_up, 95)),
        "up_top_10": float(np.percentile(p_up, 90)),
        "down_top_0.1": float(np.percentile(p_down, 99.9)),
        "down_top_1": float(np.percentile(p_down, 99)),
        "down_top_5": float(np.percentile(p_down, 95)),
        "down_top_10": float(np.percentile(p_down, 90))
    }
    
    json_path = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f)
    print(f"💾 历史防作弊动态分位数阈值已生成并保存至: {json_path.name}")

if __name__ == "__main__":
    check_distribution()