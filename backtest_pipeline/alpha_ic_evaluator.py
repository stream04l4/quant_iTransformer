import os
import sys
import torch
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 0. 环境与路径导入
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import (
    SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, TEMP_DIR,
    SEQ_LEN, D_MODEL, E_LAYERS, DROPOUT, FORECAST_HORIZON, EXPERIMENT_ID
)
from historical_pipeline.train_itransformer import InvertedTransformer


# ==========================================
# 1. 模型纯净评估主引擎
# ==========================================
def evaluate_pure_alpha():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 定位最新数据矩阵
    data_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"*_Master_Training_Matrix_{FORECAST_HORIZON}h_*.parquet")))
    if not data_files:
        print("❌ 找不到特征矩阵文件！")
        return
    data_path = data_files[-1]
    
    # 2. 定位模型权重文件
    target_model_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth"
    if not target_model_path.exists():
        print(f"❌ 找不到当前批次的模型权重文件: {target_model_path.name}！")
        return
    model_files = [target_model_path]

    # 3. 加载特征空间与时间轴
    print(f"📦 正在加载特征宽表: {data_path.name}")
    df_matrix = pl.read_parquet(data_path)
    timestamps = df_matrix.select("hour_timestamp").to_series().to_numpy()
    
    # 🚨 严格对齐回测逻辑：只截取最后的 20% 作为 Out-of-Sample 测试集
    split_idx = int(len(df_matrix) * 0.8) + FORECAST_HORIZON
    test_timestamps = timestamps[split_idx:]
    
    # 4. 加载物理价格空间 (用于计算未来真实收益率)
    raw_kline_dir = TEMP_DIR / "klines" / SYMBOL
    try:
        df_1h = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1h.parquet")
        # 构建一个以毫秒/微秒级时间戳为索引的价格映射表，实现极速查询
        price_series = pd.Series(
            data=df_1h["close"].to_numpy(), 
            index=df_1h["timestamp"].to_numpy().astype('datetime64[ns]')
        )
    except Exception as e:
        print(f"❌ 找不到原始 1h K 线，请检查路径: {e}")
        return

    horizons = [6, 12, 24, 48]
    all_reports = []

    print(f"\n🚀 开始对 {len(model_files)} 个模型进行工业级 IC/Rank IC 扫雷测试...\n")
    
    for model_path in model_files:
        model_name = model_path.name
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        feature_cols = checkpoint['feature_cols']
        
        # 构建并加载模型（从 checkpoint 读取 seq_len，防止修改 config 后 shape mismatch）
        ckpt_seq_len = checkpoint.get('seq_len', SEQ_LEN)  # 兼容旧权重文件
        dynamic_n_heads = max(2, D_MODEL // 16) 
        dynamic_d_ff = D_MODEL * 4
        model = InvertedTransformer(
            num_variates=len(feature_cols), seq_len=ckpt_seq_len, d_model=D_MODEL,
            n_heads=dynamic_n_heads, d_ff=dynamic_d_ff, e_layers=E_LAYERS, dropout=DROPOUT
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  ✅ 模型加载成功 | 感受野 seq_len={ckpt_seq_len} | 特征数={len(feature_cols)}")

        # 提取当前模型对应的特征数据并进入测试集
        X_raw = df_matrix.select(feature_cols).to_numpy().astype('float32')
        X_test_raw = X_raw[split_idx:]
        
        # 批量推理（Zero-Copy 模式，使用 checkpoint 中的 seq_len）
        X_tensor = torch.tensor(X_test_raw, dtype=torch.float32, device=DEVICE)
        X_windows = X_tensor.unfold(0, ckpt_seq_len, 1).transpose(1, 2)
        
        predictions_probs = []
        INFERENCE_BATCH_SIZE = 4096
        
        with torch.no_grad():
            for i in range(0, len(X_windows), INFERENCE_BATCH_SIZE):
                batch_x = X_windows[i : i + INFERENCE_BATCH_SIZE]
                with torch.amp.autocast('cuda'):
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                predictions_probs.extend(probs)
                
        # 生成时间戳对齐（扣除 ckpt_seq_len 造成的窗口位移）
        valid_timestamps = test_timestamps[ckpt_seq_len - 1 :].tolist()
        valid_timestamps_pd = pd.to_datetime(valid_timestamps)

        # 构建评测数据框
        eval_df = pd.DataFrame({'timestamp': valid_timestamps_pd})
        probs_array = np.array(predictions_probs)
        
        eval_df['p_down'] = probs_array[:, 0]
        eval_df['p_flat'] = probs_array[:, 1]
        eval_df['p_up'] = probs_array[:, 2]
        
        # 🌟 核心量价衍生：计算连续的 Alpha Score
        eval_df['alpha_score'] = eval_df['p_up'] - eval_df['p_down']
        # 离散信号：取 argmax 对应的方向。0类->-1, 1类->0, 2类->1
        eval_df['signal'] = np.argmax(probs_array, axis=1)
        eval_df['signal'] = eval_df['signal'].map({0: -1, 1: 0, 2: 1})
        
        # 提取当前时间点 (T) 的收盘价
        eval_df['close_T'] = eval_df['timestamp'].map(price_series)
        
        metrics = {}
        for N in horizons:
            # 提取未来时间点 (T+N) 的收盘价
            future_timestamps = eval_df['timestamp'] + pd.Timedelta(hours=N)
            eval_df[f'close_T+{N}'] = future_timestamps.map(price_series)
            
            # 计算绝对物理收益率
            fwd_col = f'fwd_return_{N}h'
            eval_df[fwd_col] = (eval_df[f'close_T+{N}'] - eval_df['close_T']) / eval_df['close_T']
            
            # 剔除由于超尾造成的 NaN 数据
            clean_df = eval_df.dropna(subset=['alpha_score', fwd_col]).copy()
            
            if len(clean_df) < 100:
                continue
                
            # 计算纯方向带来的真实收益
            clean_df['trade_pnl'] = clean_df['signal'] * clean_df[fwd_col]
            trade_df = clean_df[clean_df['signal'] != 0]
            
            accuracy = (trade_df['trade_pnl'] > 0).mean() if len(trade_df) > 0 else 0
            mean_return = trade_df['trade_pnl'].mean() if len(trade_df) > 0 else 0
            
            # 🌟 核心指标：计算连续因子的 IC 与 Rank IC
            ic_val = clean_df['alpha_score'].corr(clean_df[fwd_col], method='pearson')
            rank_ic_val = clean_df['alpha_score'].corr(clean_df[fwd_col], method='spearman')
            
            metrics[N] = {
                'Acc': accuracy,
                'Mean_Ret': mean_return,
                'IC': ic_val,
                'Rank_IC': rank_ic_val
            }
            
        print(f"🔬 模型: {model_name}")
        row = {'Model': model_name}
        for N in horizons:
            res = metrics.get(N, {'Acc':0, 'Mean_Ret':0, 'IC':0, 'Rank_IC':0})
            print(f"   [N={N:02d}h] 胜率:{res['Acc']*100:5.2f}% | 单笔期望:{res['Mean_Ret']*100:+6.3f}% | IC:{res['IC']:+6.4f} | Rank_IC:{res['Rank_IC']:+6.4f}")
            row[f'Rank_IC_{N}h'] = res['Rank_IC']
            row[f'Acc_{N}h'] = res['Acc']
        print("-" * 60)
        all_reports.append(row)

    # 5. 输出排行榜
    print("\n🏆 ================== 核心预测力排行榜 (按 12h Rank IC 降序) ================== 🏆")
    report_df = pd.DataFrame(all_reports)
    if not report_df.empty:
        report_df = report_df.sort_values(by='Rank_IC_12h', ascending=False).reset_index(drop=True)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(report_df.to_string(index=False))
        
        out_file = TRAIN_DATA_DIR / "alpha_pure_evaluation.csv"
        report_df.to_csv(out_file, index=False)
        print(f"\n💾 扫雷报告已保存至 {out_file}")

if __name__ == "__main__":
    evaluate_pure_alpha()