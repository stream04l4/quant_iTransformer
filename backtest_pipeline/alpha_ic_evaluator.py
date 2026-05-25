import os
import sys
import torch
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path

# ==========================================
# 0. 环境与路径导入
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# 🚨 剔除网络拓扑和预测周期的硬编码依赖，只保留路径
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, TEMP_DIR, EXPERIMENT_ID
from historical_pipeline.train_itransformer import InvertedTransformer


def calculate_daily_rank_ic_stats(clean_df, fwd_col):
    """基于每日 Rank IC 序列计算稳定性指标。"""
    daily_rank_ic = (
        clean_df
        .assign(trade_date=clean_df["timestamp"].dt.date)
        .groupby("trade_date")[["alpha_score", fwd_col]]
        .apply(lambda x: x["alpha_score"].corr(x[fwd_col], method="spearman") if len(x) >= 3 else np.nan)
        .dropna()
    )

    if daily_rank_ic.empty:
        return {
            "Rank_IC_IR": 0.0,
            "Rank_IC_Win_Rate": 0.0,
            "Rank_IC_MDD": 0.0,
            "Daily_Rank_IC_Count": 0,
        }

    ic_mean = daily_rank_ic.mean()
    ic_std = daily_rank_ic.std(ddof=1)
    rank_ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else 0.0
    rank_ic_win_rate = (daily_rank_ic > 0).mean()

    cumulative_ic = daily_rank_ic.cumsum()
    running_peak = cumulative_ic.cummax()
    rank_ic_mdd = (running_peak - cumulative_ic).max()

    return {
        "Rank_IC_IR": float(rank_ic_ir),
        "Rank_IC_Win_Rate": float(rank_ic_win_rate),
        "Rank_IC_MDD": float(rank_ic_mdd),
        "Daily_Rank_IC_Count": int(len(daily_rank_ic)),
    }


# ==========================================
# 1. 模型纯净评估主引擎
# ==========================================
def evaluate_pure_alpha():
    DEVICE = torch.device("cuda") # 纯 GPU 环境
    
    # 1. 优先定位模型权重文件
    target_model_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth"
    if not target_model_path.exists():
        print(f"❌ 找不到当前批次的模型权重文件: {target_model_path.name}！")
        return
    model_files = [target_model_path]
    all_reports = []

    # 提取物理价格空间
    raw_kline_dir = TEMP_DIR / "klines" / SYMBOL
    try:
        df_1h = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1h.parquet")
        price_series = pd.Series(
            data=df_1h["close"].to_numpy(), 
            index=df_1h["timestamp"].to_numpy().astype('datetime64[ns]')
        )
    except Exception as e:
        print(f"❌ 找不到原始 1h K 线，请检查路径: {e}")
        return

    print(f"\n🚀 开始对 {len(model_files)} 个模型进行工业级 IC/Rank IC 扫雷测试...\n")
    
    for model_path in model_files:
        model_name = model_path.name
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        feature_cols = checkpoint['feature_cols']
        
        # 🚨 修复 1：全量读取元数据，切断污染
        ckpt_horizon = checkpoint.get('forecast_horizon', 12)
        ckpt_seq_len = checkpoint.get('seq_len')
        ckpt_d_model = checkpoint.get('d_model')
        ckpt_n_heads = checkpoint.get('n_heads')
        ckpt_d_ff = checkpoint.get('d_ff')
        ckpt_e_layers = checkpoint.get('e_layers')
        ckpt_dropout = checkpoint.get('dropout')

        # 🚨 修复 2：动态将该模型的真实预测目标时间加入验证列表
        base_horizons = [6, 12, 24, 48]
        horizons = sorted(list(set(base_horizons + [ckpt_horizon])))

        model = InvertedTransformer(
            num_variates=len(feature_cols), seq_len=ckpt_seq_len, d_model=ckpt_d_model,
            n_heads=ckpt_n_heads, d_ff=ckpt_d_ff, e_layers=ckpt_e_layers, dropout=ckpt_dropout
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  ✅ 模型加载成功 | 感受野={ckpt_seq_len} | 专属目标预测={ckpt_horizon}h")

        # 基于真实 horizon 匹配数据表
        data_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"{SYMBOL}_Master_Training_Matrix_{ckpt_horizon}h_*.parquet")))
        if not data_files:
            print(f"❌ 找不到对应 Horizon ({ckpt_horizon}h) 的数据文件！跳过。")
            continue
        data_path = data_files[-1]

        df_matrix = pl.read_parquet(data_path)
        timestamps = df_matrix.select("hour_timestamp").to_series().to_numpy()
        X_raw = df_matrix.select(feature_cols).to_numpy().astype('float32')

        # 解决预热损耗：测试集往回多拿 seq_len-1 行
        split_idx = int(len(df_matrix) * 0.8) + ckpt_horizon
        warmup_start_idx = split_idx - ckpt_seq_len + 1
        
        if warmup_start_idx < 0:
            print(f"⚠️ 数据量不足以支撑 seq_len={ckpt_seq_len} 的预热，跳过！")
            continue

        X_test_raw = X_raw[warmup_start_idx:]
        test_timestamps = timestamps[split_idx:]

        X_tensor = torch.tensor(X_test_raw, dtype=torch.float32, device=DEVICE)
        X_windows = X_tensor.unfold(0, ckpt_seq_len, 1).transpose(1, 2).contiguous()
        
        predictions_probs = []
        INFERENCE_BATCH_SIZE = 4096
        
        with torch.no_grad():
            for i in range(0, len(X_windows), INFERENCE_BATCH_SIZE):
                batch_x = X_windows[i : i + INFERENCE_BATCH_SIZE]
                with torch.amp.autocast('cuda'):
                    logits = model(batch_x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                predictions_probs.extend(probs)
                
        valid_timestamps_pd = pd.to_datetime(test_timestamps.tolist())

        eval_df = pd.DataFrame({'timestamp': valid_timestamps_pd})
        probs_array = np.array(predictions_probs)
        
        eval_df['p_down'] = probs_array[:, 0]
        eval_df['p_flat'] = probs_array[:, 1]
        eval_df['p_up'] = probs_array[:, 2]
        
        eval_df['alpha_score'] = eval_df['p_up'] - eval_df['p_down']
        eval_df['signal'] = np.argmax(probs_array, axis=1)
        eval_df['signal'] = eval_df['signal'].map({0: -1, 1: 0, 2: 1})
        eval_df['close_T'] = eval_df['timestamp'].map(price_series)
        
        metrics = {}
        for N in horizons:
            future_timestamps = eval_df['timestamp'] + pd.Timedelta(hours=N)
            eval_df[f'close_T+{N}'] = future_timestamps.map(price_series)
            
            fwd_col = f'fwd_return_{N}h'
            eval_df[fwd_col] = (eval_df[f'close_T+{N}'] - eval_df['close_T']) / eval_df['close_T']
            
            clean_df = eval_df.dropna(subset=['alpha_score', fwd_col]).copy()
            if len(clean_df) < 100: continue
                
            clean_df['trade_pnl'] = clean_df['signal'] * clean_df[fwd_col]
            trade_df = clean_df[clean_df['signal'] != 0]
            trade_count = len(trade_df)
            sample_count = len(clean_df)
            signal_coverage = trade_count / sample_count if sample_count > 0 else 0.0
            
            accuracy = (trade_df['trade_pnl'] > 0).mean() if trade_count > 0 else 0
            mean_return = trade_df['trade_pnl'].mean() if trade_count > 0 else 0
            
            ic_val = clean_df['alpha_score'].corr(clean_df[fwd_col], method='pearson')
            rank_ic_val = clean_df['alpha_score'].corr(clean_df[fwd_col], method='spearman')
            daily_rank_ic_stats = calculate_daily_rank_ic_stats(clean_df, fwd_col)
            
            metrics[N] = {
                'Acc': accuracy,
                'Mean_Ret': mean_return,
                'IC': ic_val,
                'Rank_IC': rank_ic_val,
                'Trade_Count': trade_count,
                'Sample_Count': sample_count,
                'Signal_Coverage': signal_coverage,
                **daily_rank_ic_stats,
            }
            
        print(f"🔬 模型: {model_name}")
        row = {'Model': model_name}
        for N in horizons:
            res = metrics.get(N, {
                'Acc': 0, 'Mean_Ret': 0, 'IC': 0, 'Rank_IC': 0,
                'Trade_Count': 0, 'Sample_Count': 0, 'Signal_Coverage': 0,
                'Rank_IC_IR': 0, 'Rank_IC_Win_Rate': 0, 'Rank_IC_MDD': 0,
                'Daily_Rank_IC_Count': 0,
            })
            marker = "🎯" if N == ckpt_horizon else "  " # 高亮主要评估指标
            print(
                f" {marker} [N={N:02d}h] 胜率:{res['Acc']*100:5.2f}% | "
                f"单笔期望:{res['Mean_Ret']*100:+6.3f}% | IC:{res['IC']:+6.4f} | "
                f"Rank_IC:{res['Rank_IC']:+6.4f} | IC_IR:{res['Rank_IC_IR']:+6.3f} | "
                f"IC胜率:{res['Rank_IC_Win_Rate']*100:5.2f}% | IC_MDD:{res['Rank_IC_MDD']:.4f} | "
                f"信号:{res['Trade_Count']}/{res['Sample_Count']} ({res['Signal_Coverage']*100:.2f}%)"
            )
            row[f'Rank_IC_{N}h'] = res['Rank_IC']
            row[f'Rank_IC_IR_{N}h'] = res['Rank_IC_IR']
            row[f'Rank_IC_Win_Rate_{N}h'] = res['Rank_IC_Win_Rate']
            row[f'Rank_IC_MDD_{N}h'] = res['Rank_IC_MDD']
            row[f'Daily_Rank_IC_Count_{N}h'] = res['Daily_Rank_IC_Count']
            row[f'Acc_{N}h'] = res['Acc']
            row[f'Trade_Count_{N}h'] = res['Trade_Count']
            row[f'Sample_Count_{N}h'] = res['Sample_Count']
            row[f'Signal_Coverage_{N}h'] = res['Signal_Coverage']
        print("-" * 60)
        all_reports.append(row)

    print("\n🏆 ================== 核心预测力评估完毕 ================== 🏆")
    report_df = pd.DataFrame(all_reports)
    if not report_df.empty:
        # 使用真实的模型目标周期进行排序
        primary_sort_col = f'Rank_IC_{ckpt_horizon}h'
        if primary_sort_col in report_df.columns:
            report_df = report_df.sort_values(by=primary_sort_col, ascending=False).reset_index(drop=True)
            
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(report_df.to_string(index=False))
        
        out_file = TRAIN_DATA_DIR / "alpha_pure_evaluation.csv"
        report_df.to_csv(out_file, mode='a', header=not out_file.exists(), index=False)
        print(f"\n💾 扫雷报告已追加保存至 {out_file}")

if __name__ == "__main__":
    evaluate_pure_alpha()
