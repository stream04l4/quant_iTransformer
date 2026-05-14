import os
import sys
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import (
    SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, EXPERIMENT_ID, 
    SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    D_MODEL, E_LAYERS, DROPOUT, CLASS_THRESHOLD, FORECAST_HORIZON
)
# ==========================================
# 1. iTransformer 模型核心架构 (维度倒置)
# ==========================================
class InvertedTransformer(nn.Module):
    def __init__(self, num_variates, seq_len, d_model, n_heads, d_ff, e_layers, dropout):
        super(InvertedTransformer, self).__init__()
        
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # 🚨 修改 1：引入 Flatten 层，保留所有特征的独立表征
        self.flatten = nn.Flatten(start_dim=1)
        
        # 🚨 修改 2：废弃单层 Linear，替换为产业界标准的 MLP Projector，实现特征的高阶非线性交叉
        self.projector = nn.Sequential(
            nn.Linear(num_variates * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, x):
        seq_means = x.mean(dim=1, keepdim=True)
        seq_stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-8)
        x = (x - seq_means) / seq_stdev
        
        x = x.transpose(1, 2)  
        
        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        
        # 🚨 修改 3：移除 mean(dim=1)，直接展平
        # enc_out 原始形状: [Batch, Num_Variates, d_model]
        # 展平后形状: [Batch, Num_Variates * d_model]
        global_rep = self.flatten(enc_out) 
        
        # 通过多层感知机进行非线性降维与分类预测
        dec_out = self.projector(global_rep) 
        
        return dec_out

# ==========================================
# 2. PyTorch 数据加载器 (Dataset) - 3分类版本
# ==========================================
class CryptoDataset(Dataset):
    def __init__(self, parquet_path, seq_len, is_train=True, train_ratio=0.8):
        print(f"📦 正在加载数据集 ({'训练' if is_train else '验证'}): {Path(parquet_path).name}")
        df = pl.read_parquet(parquet_path)
        
        target_col = f"target_{FORECAST_HORIZON}h_return"
        exclude_cols = ["hour_timestamp", target_col]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df.select(self.feature_cols).to_numpy().astype('float32')
        Y_raw = df.select(target_col).to_numpy().astype('float32').reshape(-1)
        
        THRESHOLD = CLASS_THRESHOLD
        Y_class = np.where(Y_raw > THRESHOLD, 2, 
                           np.where(Y_raw < -THRESHOLD, 0, 1))
        
        split_idx = int(len(X) * train_ratio)
        
        if is_train:
            self.X, self.Y = X[:split_idx], Y_class[:split_idx]
        else:
            # 🚨 核心修复：强行隔离！验证集起点必须向后退让 FORECAST_HORIZON，切断特征与训练集标签的未来交叉
            val_start_idx = split_idx + FORECAST_HORIZON
            self.X, self.Y = X[val_start_idx:], Y_class[val_start_idx:]
            
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        seq_x = self.X[idx : idx + self.seq_len]
        seq_y = self.Y[idx + self.seq_len - 1]
        
        # 🚨 极其重要：CrossEntropyLoss 要求目标标签必须是 long 类型 (int64)，且不能是二维矩阵
        return torch.from_numpy(seq_x), torch.tensor(seq_y, dtype=torch.long)

# ==========================================
# 3. H100 训练主引擎
# ==========================================
def train_model():
    # 路径配置
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 自动定位最新生成的特征宽表
    parquet_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"*_Master_Training_Matrix_{FORECAST_HORIZON}h_*.parquet")))
    if not parquet_files:
        print("❌ 未找到特征宽表，请先运行 build_main_features.py")
        return
    DATA_PATH = parquet_files[-1]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🚀 启动 iTransformer 极速训练管道...")
    print(f"🖥️ 硬件设备: {DEVICE} (H100 就绪)")
    
    train_set = CryptoDataset(DATA_PATH, seq_len=SEQ_LEN, is_train=True)
    val_set = CryptoDataset(DATA_PATH, seq_len=SEQ_LEN, is_train=False)
    
    # 🚨 并发寻优模式下，严禁开启内部子进程，防止 CPU 踩踏
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    num_variates = len(train_set.feature_cols)
    print(f"📊 特征维度数: {num_variates} 维")
    
    # === 模型架构与极致正则化优化 ===
    dynamic_n_heads = max(2, D_MODEL // 16) 
    dynamic_d_ff = D_MODEL * 4

    model = InvertedTransformer(
        num_variates=num_variates, 
        seq_len=SEQ_LEN,
        d_model=D_MODEL,     
        n_heads=dynamic_n_heads, # 🚨 自适应：比如 d_model=64 时为 4，d_model=32 时为 2
        d_ff=dynamic_d_ff,       # 🚨 自适应：比如 d_model=64 时为 256，d_model=32 时为 128
        e_layers=E_LAYERS,   
        dropout=DROPOUT      
    ).to(DEVICE)
    
    # 🚨 提升 Weight Decay 10倍 (从 1e-3 到 1e-2)，严厉惩罚“死记硬背”的参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 动态类别权重 (保持不变)
    train_labels = train_set.Y
    unique_classes = np.unique(train_labels)
    from sklearn.utils.class_weight import compute_class_weight
    computed_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
    class_weights = torch.tensor(computed_weights, dtype=torch.float).to(DEVICE)
    print(f"\n⚖️ 基于真实数据分布，动态计算的类别权重: {computed_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 🚨 产业界前沿：换用余弦退火调度器，强迫模型多走几十轮
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # 🚨 终极修复：废弃 F1，回归 Val Loss 置信度监控，并引入容忍度极低的早停机制
    best_val_loss = float('inf')
    patience = 16                  # 容忍度：连续 15 个 Epoch 验证集 Loss 不降则强行停止 (对于高噪金融数据，5已足够)
    patience_counter = 0          # 早停计数器
    
    print("\n🔥 开始炼丹 (引入置信度监控与防过拟合机制)...")
    for epoch in range(EPOCHS):
        # --- 训练阶段 (保持你原有的训练逻辑不变) ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    output = model(batch_x)
                    v_loss = criterion(output, batch_y)
                    
                val_loss += v_loss.item()
                preds = torch.argmax(output, dim=1)
                
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100
        
        scheduler.step(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.2f}%) | "
              f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%)")
        
        # --- 置信度选优与早停保存机制 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 创下置信度新高，重置早停计数器
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth"
            
            torch.save({
                'seq_len': SEQ_LEN,              # 固化感受野，确保推理/评估时不依赖全局 config
                'class_threshold': CLASS_THRESHOLD,  # 固化分类阈值，便于复现
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_cols': train_set.feature_cols,
                'val_loss': best_val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"    ⭐ 泛化能力最强的权重已保存! 当前最低 Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    ⚠️ Val Loss 出现抬头迹象，未改善 ({patience_counter}/{patience})")

        # 🚨 熔断判定：强行切断死记硬背
        if patience_counter >= patience:
            print("\n🛑 触发早停机制 (Early Stopping)：验证集置信度已连续恶化，强行终止训练！")
            print(f"🎯 最终采用的是第 {epoch - patience_counter + 1} 轮保存的最优泛化权重。")
            break  # 砸碎炼丹炉，跳出训练

    print(f"\n🎉 训练全部完成！最终模型路径: {save_path}")

if __name__ == "__main__":
    train_model()