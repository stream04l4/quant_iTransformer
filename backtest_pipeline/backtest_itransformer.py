import os
import sys
import torch
import torch.nn as nn
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, SEQ_LEN, D_MODEL, E_LAYERS, DROPOUT, CLASS_THRESHOLD, EXPERIMENT_ID


# ==========================================
# 1. 必须对齐的三分类架构
# ==========================================
class InvertedTransformer(nn.Module):
    def __init__(self, num_variates, seq_len, d_model=64, n_heads=4, d_ff=256, e_layers=1, dropout=0.4):
        super(InvertedTransformer, self).__init__()
        self.enc_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        # 🚨 必须是 3 分类输出
        self.projector = nn.Linear(d_model, 3) 

    def forward(self, x):
        x = x.transpose(1, 2)
        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        dec_out = self.projector(enc_out)
        return dec_out.mean(dim=1)

def run_backtest():
    data_files = sorted(list(PROCESSED_MAIN_DIR.glob("*_Master_Training_Matrix_*.parquet")))
    model_files = sorted(list(MODEL_DIR.glob(f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth")))
    
    if not data_files or not model_files:
        print("❌ 找不到数据或模型文件！")
        return
        
    data_path = data_files[-1]
    model_path = model_files[-1]
    print(f"🚀 启动终极回测 V4 (Softmax 概率引擎 + 动态杠杆)...")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']
    
    # 显式对齐架构参数
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

    # 读取数据
    df = pl.read_parquet(data_path)
    actuals_1h = df.select("log_return").to_numpy().astype('float32').reshape(-1)
    prices_full = np.exp(np.cumsum(np.insert(actuals_1h, 0, 0)))
    
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    timestamps = df.select("hour_timestamp").to_series().to_list()
    
    split_idx = int(len(X_raw) * 0.8)
    X_test_scaled = scaler.transform(X_raw[split_idx:])
    
    prices = prices_full[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    print("\n⚙️ 正在 H100 上生成概率矩阵 (Softmax 置信度)...")
    predictions_probs = []
    valid_timestamps = []
    
    with torch.no_grad():
        for i in range(len(X_test_scaled) - SEQ_LEN):
            seq_x = X_test_scaled[i : i + SEQ_LEN]
            tensor_x = torch.from_numpy(seq_x).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(tensor_x)
                # 转化为概率向量: [看跌概率, 震荡概率, 看涨概率]
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            predictions_probs.append(probs)
            valid_timestamps.append(test_timestamps[i + SEQ_LEN])

    predictions_probs = np.array(predictions_probs)
    
    # ==========================================
    # 🏦 基于置信度的动态状态机
    # ==========================================
    # 自动加载 check_prob_distribution 生成的分位数阈值
    json_path = TRAIN_DATA_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    if not json_path.exists():
        print("❌ 找不到 dynamic_thresholds.json，请先运行 check_prob_distribution.py！")
        return
    with open(json_path, "r") as f:
        th = json.load(f)
    print("📥 已成功加载动态分位数阈值。")

    HOLD_PERIOD = 24         
    FEE_RATE = 0.0005        
    HARD_TP = 0.04           # 绝对止盈线 4% (防止过山车)
    
    equity = 1.0             
    equity_curve = []        
    
    current_pos = 0          
    hold_timer = 0           
    entry_price = 0.0        
    trade_start_equity = 1.0 
    current_leverage = 0.0   # 记录当前在场单子的杠杆倍数
    
    trade_count = 0
    win_count = 0
    liquidated = False       
    
    print("\n⚔️ 正在执行概率风控撮合...")
    
    for i in range(len(predictions_probs)):
        if liquidated:
            equity_curve.append(0)
            continue
            
        p_down, p_flat, p_up = predictions_probs[i]
        curr_price = prices[i]       
        next_price = prices[i+1]     
        
        # 1. 寻找开仓机会 (基于真实分位数的动态阶梯杠杆)
        if current_pos == 0:
            target_pos = 0
            dynamic_leverage = 0.0

            if p_up > th["up_top_1"]:      
                target_pos = 1; dynamic_leverage = 3.0
            elif p_up > th["up_top_5"]:    
                target_pos = 1; dynamic_leverage = 2.0
            elif p_up > th["up_top_10"]:   
                target_pos = 1; dynamic_leverage = 1.0

            elif p_down > th["down_top_1"]:  
                target_pos = -1; dynamic_leverage = 3.0
            elif p_down > th["down_top_5"]:  
                target_pos = -1; dynamic_leverage = 2.0
            elif p_down > th["down_top_10"]: 
                target_pos = -1; dynamic_leverage = 1.0

            if target_pos != 0:
                current_pos = target_pos
                current_leverage = dynamic_leverage
                hold_timer = HOLD_PERIOD
                entry_price = curr_price
                trade_start_equity = equity
                equity -= trade_start_equity * current_leverage * FEE_RATE

        # 2. 持仓监控与结算
        if current_pos != 0:
            underlying_return = (next_price - entry_price) / entry_price if current_pos == 1 else (entry_price - next_price) / entry_price
            floating_equity = trade_start_equity * (1 - current_leverage * FEE_RATE + current_leverage * underlying_return)
            
            if floating_equity <= 0:
                liquidated = True
                equity = 0
                current_pos = 0
                equity_curve.append(0)
                print(f"💀 爆仓！发生于第 {i} 小时！")
                trade_count += 1
                continue
                
            hold_timer -= 1
            close_position = False
            
            # 🎯 离场逻辑一：概率反转 (对齐新的 Top 10% 警戒线)
            if current_pos == 1 and p_down > th["down_top_10"]:
                close_position = True
            elif current_pos == -1 and p_up > th["up_top_10"]:
                close_position = True
            
            # 🎯 离场逻辑二：触及绝对止盈线
            elif underlying_return >= HARD_TP:
                close_position = True
                
            # ⏳ 离场逻辑三：时间锁到期
            elif hold_timer == 0:
                close_position = True
                
            if close_position:
                floating_equity -= trade_start_equity * current_leverage * FEE_RATE
                equity = floating_equity
                if underlying_return > 0: win_count += 1
                trade_count += 1
                current_pos = 0 
                current_leverage = 0.0
            else:
                equity = floating_equity
                
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    benchmark_curve = prices[1:len(predictions_probs)+1] / prices[0]

    if trade_count > 0:
        win_rate = (win_count / trade_count) * 100
    else:
        win_rate = 0.0
        
    returns_series = np.diff(equity_curve, prepend=1.0) / np.roll(equity_curve, 1)
    returns_series[0] = 0 
    if np.std(returns_series) != 0:
        sharpe = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(8760)
    else:
        sharpe = 0.0

    print("\n📊 终极 3分类动态杠杆 回测结果:")
    print("-" * 45)
    print(f"总回测时长: {len(predictions_probs)} 小时")
    print(f"共计交易: {trade_count} 笔")
    print(f"胜率 (Win Rate): {win_rate:.2f}%")
    if liquidated:
        print(f"⚠️ 状态: 已爆仓 (账户归零)")
    print(f"期末策略净值: {equity_curve[-1]:.4f}")
    print(f"期末基准净值: {benchmark_curve[-1]:.4f}")
    print(f"估算年化夏普: {sharpe:.2f}")
    print("-" * 45)

    plt.figure(figsize=(14, 7))
    plt.plot(valid_timestamps, benchmark_curve, label='BTC Buy & Hold', color='gray', alpha=0.5)
    plt.plot(valid_timestamps, equity_curve, label='iTransformer Probabilistic Engine', color='#00ff88', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    plt.title('True OOS Equity: 3-Class Probabilistic Strategy with Dynamic Leverage', fontsize=16, color='white')
    
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#121212')
    plt.ylabel('Equity Multiplier (1.0 = 100%)', color='white')
    plt.tick_params(colors='white')
    plt.legend(facecolor='#1e1e1e', labelcolor='white')
    plt.grid(True, alpha=0.1, color='white')
    
    plot_path = TRAIN_DATA_DIR / "backtest_probabilistic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    print(f"\n📈 最终资金曲线已生成: {plot_path}")

if __name__ == "__main__":
    run_backtest()