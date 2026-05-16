import os
import sys
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import (
    SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, EXPERIMENT_ID, 
    SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    D_MODEL, E_LAYERS, DROPOUT, CLASS_THRESHOLD, FORECAST_HORIZON
)

# ==========================================
# 0. 安全辅助函数 (防止调参期崩溃)
# ==========================================
def get_safe_n_heads(d_model: int) -> int:
    """确保 d_model 能被 n_heads 完美整除，防止 Transformer 报错"""
    for h in [16, 8, 4, 2, 1]:
        if d_model % h == 0 and h <= d_model:
            return h
    return 1

# ==========================================
# 1. iTransformer 模型核心架构
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
        self.flatten = nn.Flatten(start_dim=1)
        self.projector = nn.Sequential(
            nn.Linear(num_variates * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, x):
        # 🚨 终极裁决：坚决剔除局部的 (x - mean)/std。
        # 你的特征矩阵已经是严格的平稳因子，局部归一化会抹平宏观级别的极值特征！
        x = x.transpose(1, 2)  
        
        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        
        global_rep = self.flatten(enc_out) 
        dec_out = self.projector(global_rep) 
        return dec_out

# ==========================================
# 2. H100 极速 GPU 数据集 (0次拷贝，瞬间生成)
# ==========================================
class CryptoDatasetGPU(Dataset):
    def __init__(self, parquet_path, seq_len, device, is_train=True, train_ratio=0.8):
        print(f"📦 加载数据集 ({'训练' if is_train else '验证'}): {Path(parquet_path).name}")
        df = pl.read_parquet(parquet_path)
        
        target_col = f"target_{FORECAST_HORIZON}h_return"
        exclude_cols = ["hour_timestamp", target_col]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df.select(self.feature_cols).to_numpy().astype('float32')
        Y_raw = df.select(target_col).to_numpy().astype('float32').reshape(-1)
        
        # 3 分类逻辑
        Y_class = np.where(Y_raw > CLASS_THRESHOLD, 2, 
                           np.where(Y_raw < -CLASS_THRESHOLD, 0, 1))
        
        split_idx = int(len(X) * train_ratio)
        
        # 🚨 修复：严格 Purged Split，阻断标签与特征的微小交叉泄露
        train_end = split_idx - FORECAST_HORIZON
        
        if is_train:
            X_data, Y_data = X[:train_end], Y_class[:train_end]
        else:
            val_start_idx = split_idx + FORECAST_HORIZON
            X_data, Y_data = X[val_start_idx:], Y_class[val_start_idx:]
            
        # 直接推送至 GPU
        X_tensor = torch.tensor(X_data, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_data, dtype=torch.long, device=device)
        
        # 核心：使用 unfold 直接在显存中生成滑动窗口视图，完美避开 CPU Dataloader 瓶颈
        self.X = X_tensor.unfold(dimension=0, size=seq_len, step=1).transpose(1, 2).contiguous()
        self.Y = Y_tensor[seq_len - 1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. 训练主引擎
# ==========================================
def train_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"*_Master_Training_Matrix_{FORECAST_HORIZON}h_*.parquet")))
    if not parquet_files:
        print("❌ 未找到特征宽表！")
        return
    DATA_PATH = parquet_files[-1]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 启动 iTransformer 训练 | 硬件: {DEVICE} (H100 极速模式)")
    
    train_set = CryptoDatasetGPU(DATA_PATH, seq_len=SEQ_LEN, device=DEVICE, is_train=True)
    val_set = CryptoDatasetGPU(DATA_PATH, seq_len=SEQ_LEN, device=DEVICE, is_train=False)
    
    # 无需 workers，数据已在 GPU 显存内
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    num_variates = len(train_set.feature_cols)
    dynamic_n_heads = get_safe_n_heads(D_MODEL)
    dynamic_d_ff = D_MODEL * 4
    
    print(f"📊 特征维数: {num_variates} | D_MODEL: {D_MODEL} | Heads: {dynamic_n_heads}")

    model = InvertedTransformer(
        num_variates=num_variates, seq_len=SEQ_LEN, d_model=D_MODEL,     
        n_heads=dynamic_n_heads, d_ff=dynamic_d_ff, e_layers=E_LAYERS, dropout=DROPOUT      
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 🚨 修复：安全生成类别权重，严格锁定长度为 3，防御缺类崩溃
    train_labels = train_set.Y
    class_counts = torch.bincount(train_labels, minlength=3).float()
    total_samples = len(train_labels)
    num_valid_classes = (class_counts > 0).sum().item()
    
    weights = torch.ones(3, dtype=torch.float32, device=DEVICE)
    valid_mask = class_counts > 0
    # sklearn 'balanced' 逻辑的原生 PyTorch 实现
    weights[valid_mask] = total_samples / (num_valid_classes * class_counts[valid_mask])
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience, patience_counter = 16, 0
    
    # 🚨 新增：提取验证集标签分布，防范 Regime Shift (环境突变)
    val_labels = val_set.Y
    val_class_counts = torch.bincount(val_labels, minlength=3).float()

    print(f"\n⚖️ 训练集样本分布: {class_counts.cpu().numpy()}")
    print(f"⚖️ 验证集样本分布: {val_class_counts.cpu().numpy()}")
    print(f"⚖️ 自动补偿权重 (基于训练集): {weights.cpu().numpy()}\n")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            
            # 🚨 护城河：执行梯度裁剪，防止偶尔的数值尖峰炸毁 Transformer 权重
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_correct += (torch.argmax(output, dim=1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                with torch.amp.autocast('cuda'):
                    output = model(batch_x)
                    v_loss = criterion(output, batch_y)
                    
                val_loss += v_loss.item()
                preds = torch.argmax(output, dim=1)
                val_preds_list.append(preds)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100
        scheduler.step(avg_val_loss)
        
        # 🚨 诊断打印：监控模型是否摆烂（只预测震荡类1）
        all_val_preds = torch.cat(val_preds_list)
        pred_dist = torch.bincount(all_val_preds, minlength=3).cpu().numpy()
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%) "
              f"| 预测分布[空, 平, 多]: {pred_dist}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 🚨 修复：保存极其详尽的模型元数据，防止实盘加载时架构碎裂
            save_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth"
            torch.save({
                'symbol': SYMBOL,
                'experiment_id': EXPERIMENT_ID,
                'forecast_horizon': FORECAST_HORIZON,
                'seq_len': SEQ_LEN,
                'class_threshold': CLASS_THRESHOLD,
                'num_variates': num_variates,
                'd_model': D_MODEL,
                'n_heads': dynamic_n_heads,
                'd_ff': dynamic_d_ff,
                'e_layers': E_LAYERS,
                'dropout': DROPOUT,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_cols': train_set.feature_cols,
                # 🚨 补回：性能基准元数据，用于后续模型对比与实盘心跳校验
                'val_loss': best_val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"    ⭐ 新权重及完整元数据已保存! (V_Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n🛑 触发早停机制，防止过拟合！")
                break

if __name__ == "__main__":
    train_model()