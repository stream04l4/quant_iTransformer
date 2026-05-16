import os
import sys
import torch
import torch.nn as nn
import numpy as np
import polars as pl
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import (
    SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, TEMP_DIR,
    EXPERIMENT_ID, FEE_RATE, HOLD_PERIOD, HARD_TP, HARD_SL, 
    LEVERAGE_1, LEVERAGE_2, LEVERAGE_3, FORECAST_HORIZON
)
from historical_pipeline.train_itransformer import InvertedTransformer


def run_backtest():
    DEVICE = torch.device("cuda")  # 明确 H100 环境
    
    # 1. 优先定位模型权重文件，一切以它保存的元数据为准
    model_files = sorted(list(MODEL_DIR.glob(f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth")))
    if not model_files:
        print("❌ 找不到对应的模型权重文件！")
        return
        
    model_path = model_files[-1]
    print(f"🚀 启动终极回测 V5 (绝对严谨财务核算 + 动态杠杆)...")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # 🚨 防呆机制 1：全面从 checkpoint 读取真实的超参，切断隐性错配
    ckpt_horizon = checkpoint.get('forecast_horizon', FORECAST_HORIZON)
    ckpt_seq_len = checkpoint.get('seq_len')
    ckpt_d_model = checkpoint.get('d_model')
    ckpt_n_heads = checkpoint.get('n_heads')
    ckpt_d_ff = checkpoint.get('d_ff')
    ckpt_e_layers = checkpoint.get('e_layers')
    ckpt_dropout = checkpoint.get('dropout')
    feature_cols = checkpoint['feature_cols']

    # 基于真实 horizon 精准匹配特征宽表
    data_files = sorted(list(PROCESSED_MAIN_DIR.glob(f"{SYMBOL}_Master_Training_Matrix_{ckpt_horizon}h_*.parquet")))
    if not data_files:
        print(f"❌ 找不到对应 Horizon ({ckpt_horizon}h) 的数据文件！")
        return
    data_path = data_files[-1]

    # 显式对齐架构参数并加载模型
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

    # === 1. 独立加载特征空间与执行空间 ===
    df = pl.read_parquet(data_path)
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    timestamps = df.select("hour_timestamp").to_series().to_numpy()
    
    # 🚨 防呆机制 2：彻底消灭时空错位。使用 seq_len 预热对齐测试集原点
    split_idx = int(len(X_raw) * 0.8) + ckpt_horizon
    warmup_start_idx = split_idx - ckpt_seq_len + 1
    
    if warmup_start_idx < 0:
        print(f"❌ 数据量不足以支撑 seq_len={ckpt_seq_len} 的预热！")
        return

    X_test_raw = X_raw[warmup_start_idx:]
    test_timestamps = timestamps[split_idx:]
    valid_timestamps = test_timestamps.tolist() 
    
    print("\n⚙️ 正在 H100 上生成概率矩阵 (全显存 Zero-Copy 极速推理)...")
    
    X_tensor = torch.tensor(X_test_raw, dtype=torch.float32, device=DEVICE)
    # 加速修复：补齐 .contiguous() 消除显存碎片
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

    # 极速加载 1h 和 1m 的真实物理价格 (执行空间)
    raw_kline_dir = TEMP_DIR / "klines" / SYMBOL
    try:
        df_1h = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1h.parquet")
        df_1m = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1m.parquet")
    except Exception as e:
        print(f"❌ 找不到原始 K 线，请先运行 build_main_features.py 完成下载！错误: {e}")
        return

    # 全部强制转换为 int64 的秒级 Unix 绝对时间，防止 Numpy 精度错乱
    ts_1h = df_1h["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    opens_1h = df_1h["open"].to_numpy()
    closes_1h = df_1h["close"].to_numpy()
    
    ts_1m = df_1m["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    highs_1m = df_1m["high"].to_numpy()
    lows_1m = df_1m["low"].to_numpy()
    closes_1m = df_1m["close"].to_numpy()

    valid_timestamps_sec = np.array(valid_timestamps).astype('datetime64[s]').astype(np.int64)

    # ==========================================
    # 🏦 基于 1h 决策 + 1m 显微镜追踪的双轨状态机
    # ==========================================
    json_path = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    if not json_path.exists():
        print("❌ 找不到 dynamic_thresholds.json，请先运行 check_prob_distribution.py！")
        return
        
    with open(json_path, "r") as f:
        config_data = json.load(f)
        # 🚨 防呆机制 3：兼容带有 metadata 的新结构 JSON 和老版 Flat JSON
        th = config_data.get("thresholds", config_data)
        
    print("📥 已成功加载动态分位数阈值。")
    
    equity = 1.0             
    equity_curve = []
    benchmark_prices = []
    
    current_pos = 0          
    current_tier = 0         # 👈 新增：记录当前阶梯水位
    hold_timer_mins = 0      
    entry_price = 0.0     
    trade_start_equity = 1.0 
    current_leverage = 0.0   
    entry_time = None
    
    # 🚨 引入严格的财务账本变量
    cumulative_funding = 0.0 
    open_cost = 0.0
    
    max_unrealized_phys_pnl = 0.0 
    trailing_active = False
    dynamic_sl_phys = -abs(HARD_SL)
    
    trade_count = 0
    win_count = 0
    liquidated = False       
    
    long_trades = 0
    short_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    print("\n⚔️ 正在执行工业级撮合 (1h 决策开仓 + 1m 盘中悲观碰撞)...")
    
    SLIPPAGE = 0.0002 
    HOURLY_FUNDING_COST = 0.0001 / 8 

    exit_reasons = {
        "REVERSAL_EXIT": 0,    
        "HARD_SL_EXIT": 0,     
        "TRAILING_EXIT": 0,    
        "TIME_LIMIT_EXIT": 0   
    }
    pnl_list = [] 

    for i in range(len(predictions_probs)):
        if i > 0 and i % 1000 == 0:
            print(f"   ⏳ 已撮合 {i} / {len(predictions_probs)} 小时...")
            
        if liquidated:
            equity_curve.append(0)
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            continue
            
        p_down, p_flat, p_up = predictions_probs[i]
        signal_time = valid_timestamps_sec[i] 
        exec_time = signal_time + 3600
        exec_idx_1h = np.searchsorted(ts_1h, exec_time)
        
        if exec_idx_1h < len(closes_1h):
            benchmark_prices.append(closes_1h[exec_idx_1h])
        else:
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            break

        # ------------------------------------------
        # 1. 寻找开仓与【阶梯加仓】机会 (1h 级别)
        # ------------------------------------------
        target_pos = 0
        target_tier = 0
        target_leverage = 0.0

        if p_up > th["up_top_1"]:      target_pos = 1; target_tier = 3; target_leverage = LEVERAGE_3
        elif p_up > th["up_top_5"]:    target_pos = 1; target_tier = 2; target_leverage = LEVERAGE_2
        elif p_up > th["up_top_10"]:   target_pos = 1; target_tier = 1; target_leverage = LEVERAGE_1
        elif p_down > th["down_top_1"]:  target_pos = -1; target_tier = 3; target_leverage = LEVERAGE_3
        elif p_down > th["down_top_5"]:  target_pos = -1; target_tier = 2; target_leverage = LEVERAGE_2
        elif p_down > th["down_top_10"]: target_pos = -1; target_tier = 1; target_leverage = LEVERAGE_1

        if target_pos != 0 and exec_idx_1h < len(opens_1h):
            current_hour_open = opens_1h[exec_idx_1h]
            
            # 💡 情景 A: 绝对空仓，首次建仓
            if current_pos == 0:
                current_pos = target_pos
                current_tier = target_tier
                current_leverage = target_leverage
                hold_timer_mins = HOLD_PERIOD * 60 
                
                entry_price = current_hour_open 
                entry_time = ts_1h[exec_idx_1h]
                
                trade_start_equity = equity 
                open_cost = trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                equity = trade_start_equity - open_cost 
                cumulative_funding = 0.0  
                
                if target_pos == 1: long_trades += 1
                if target_pos == -1: short_trades += 1
                
                max_unrealized_phys_pnl = 0.0   
                trailing_active = False         
                dynamic_sl_phys = -abs(HARD_SL) 

            # 💡 情景 B: 同向持仓，且触发了更高的 Tier (执行金字塔加仓)
            elif current_pos == target_pos and target_tier > current_tier:
                # [核心修正]: 必须记住加仓前的"绝对止损价"，防止均价上移导致瞬间爆仓
                if current_pos == 1:
                    absolute_sl_price = entry_price * (1 + dynamic_sl_phys)
                else:
                    absolute_sl_price = entry_price * (1 - dynamic_sl_phys)
                    
                # 【对齐 OMS 物理逻辑】：直接累加新杠杆
                added_leverage = target_leverage
                old_notional = trade_start_equity * current_leverage
                new_notional = trade_start_equity * added_leverage
                
                old_size = old_notional / entry_price
                new_size = new_notional / current_hour_open
                total_size = old_size + new_size
                
                # 重新计算加仓后的加权持仓均价
                avg_entry_price = (old_notional + new_notional) / total_size
                
                # 扣除新增仓位的摩擦成本
                added_open_cost = trade_start_equity * added_leverage * (FEE_RATE + SLIPPAGE)
                open_cost += added_open_cost
                equity -= added_open_cost 
                
                # 更新全局状态机
                entry_price = avg_entry_price
                current_leverage += added_leverage
                current_tier = target_tier
                
                # 重新锚定防线：用旧的绝对止损价反推距离新均价的百分比距离
                if current_pos == 1:
                    new_dyn_sl = (absolute_sl_price - entry_price) / entry_price
                    max_unrealized_phys_pnl = (current_hour_open - entry_price) / entry_price
                else:
                    new_dyn_sl = (entry_price - absolute_sl_price) / entry_price
                    max_unrealized_phys_pnl = (entry_price - current_hour_open) / entry_price
                    
                # 兜底：如果均价恶化太多导致防线变得离谱，用硬止损兜底
                dynamic_sl_phys = max(new_dyn_sl, -abs(HARD_SL))

        # ------------------------------------------
        # 2. 1分钟级别显微镜追踪持仓
        # ------------------------------------------
        if current_pos != 0:
            start_1m_idx = np.searchsorted(ts_1m, entry_time)
            chunk_end_time = exec_time + 3600
            end_1m_idx = np.searchsorted(ts_1m, chunk_end_time)
            
            close_position = False
            actual_exit_pct = 0.0
            close_pnl_pct = 0.0 
            close_return = 0.0
            
            is_reversal = False
            if current_pos == 1 and p_down > th["down_top_1"]: is_reversal = True
            elif current_pos == -1 and p_up > th["up_top_1"]: is_reversal = True
                
            if is_reversal:
                current_hour_open = opens_1h[exec_idx_1h]
                current_phys_profit = (current_hour_open - entry_price) / entry_price if current_pos == 1 else (entry_price - current_hour_open) / entry_price
                
                if current_phys_profit < 0.005:
                    close_position = True
                    actual_exit_pct = current_phys_profit * current_leverage
                    exit_reasons["REVERSAL_EXIT"] += 1 
                else:
                    trailing_active = True
                    dynamic_sl_phys = max(dynamic_sl_phys, 0.005)
            
            if not close_position:
                for m_idx in range(start_1m_idx, end_1m_idx):
                    if m_idx >= len(closes_1m): break
                    
                    m_high = highs_1m[m_idx]
                    m_low = lows_1m[m_idx]
                    m_close = closes_1m[m_idx]
                    
                    if current_pos == 1:
                        worst_intra = (m_low - entry_price) / entry_price
                        best_intra = (m_high - entry_price) / entry_price
                        close_return = (m_close - entry_price) / entry_price
                    else:
                        worst_intra = (entry_price - m_high) / entry_price
                        best_intra = (entry_price - m_low) / entry_price
                        close_return = (entry_price - m_close) / entry_price

                    hold_timer_mins -= 1

                    worst_pnl_pct = current_leverage * worst_intra
                    # 爆仓判定 (资金毁灭，使用带上双边滑点预期最恶劣的情况)
                    if trade_start_equity - open_cost - (trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)) + (trade_start_equity * worst_pnl_pct) <= 0:
                        liquidated = True
                        break
                        
                    if worst_intra <= dynamic_sl_phys:
                        close_position = True
                        actual_exit_pct = dynamic_sl_phys * current_leverage
                        if trailing_active: exit_reasons["TRAILING_EXIT"] += 1
                        else: exit_reasons["HARD_SL_EXIT"] += 1
                        break

                    if best_intra > max_unrealized_phys_pnl:
                        max_unrealized_phys_pnl = best_intra
                        
                    if max_unrealized_phys_pnl >= abs(HARD_TP) + 0.005:
                        trailing_active = True
                        dynamic_sl_phys = max(dynamic_sl_phys, abs(HARD_TP))
                        
                    if hold_timer_mins <= 0 and not trailing_active:
                        if close_return < 0.005:
                            close_position = True
                            actual_exit_pct = close_return * current_leverage
                            exit_reasons["TIME_LIMIT_EXIT"] += 1
                            break
                        else:
                            trailing_active = True
                            dynamic_sl_phys = max(dynamic_sl_phys, 0.005)
                            
                    if trailing_active:
                        dynamic_sl_phys = max(dynamic_sl_phys, max_unrealized_phys_pnl - 0.01)

                    if worst_intra <= dynamic_sl_phys:
                        close_position = True
                        actual_exit_pct = dynamic_sl_phys * current_leverage
                        if trailing_active: exit_reasons["TRAILING_EXIT"] += 1
                        else: exit_reasons["HARD_SL_EXIT"] += 1
                        break

                close_pnl_pct = close_return * current_leverage

            if liquidated:
                equity = 0
                current_pos = 0
                current_tier = 0
                equity_curve.append(0)
                print(f"💀 盘中分钟级扎针爆仓！发生于 {signal_time}")
                trade_count += 1
                gross_loss += trade_start_equity
                continue
                
            if close_position:
                # 🚨 财务修复 2：统一严谨结算（本金 + 收益 - 开仓摩擦 - 平仓摩擦 - 累计资金费）
                close_cost = trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                final_pnl = trade_start_equity * actual_exit_pct
                
                equity = trade_start_equity + final_pnl - open_cost - close_cost - cumulative_funding
                
                trade_pnl = equity - trade_start_equity
                if trade_pnl > 0: 
                    win_count += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)
                
                trade_count += 1
                current_pos = 0 
                current_tier = 0
                current_leverage = 0.0
                
                pnl_list.append(trade_pnl / trade_start_equity)
            else:
                # 🚨 财务修复 3：跨小时浮动结算，同样严格遵守完整公式，防止资金幽灵恢复
                cumulative_funding += trade_start_equity * current_leverage * HOURLY_FUNDING_COST
                floating_pnl = trade_start_equity * close_pnl_pct
                equity = trade_start_equity + floating_pnl - open_cost - cumulative_funding
                entry_time = chunk_end_time
                
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    valid_timestamps = valid_timestamps[:len(equity_curve)]
    benchmark_prices = np.array(benchmark_prices)

    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    ls_ratio = (long_trades / short_trades) if short_trades > 0 else float('inf')

    total_return = equity_curve[-1] - 1.0
    years = len(predictions_probs) / 8760
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = (annualized_return / (abs(max_drawdown)/100)) if max_drawdown < 0 else float('inf')
    
    if len(benchmark_prices) > 0: benchmark_curve = benchmark_prices / benchmark_prices[0]
    else: benchmark_curve = np.ones(len(predictions_probs))

    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0.0
        
    returns_series = pd.Series(equity_curve).pct_change().fillna(0).to_numpy()
    returns_series[0] = 0 
    sharpe = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(8760) if np.std(returns_series) != 0 else 0.0

    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([p for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    recovery_factor = total_return / (abs(max_drawdown)/100) if max_drawdown != 0 else 0

    print("\n" + "="*50)
    print(f" 📜 TimesFM 引擎终极回测审计报告 | {SYMBOL}")
    print("="*50)
    
    print(f"【1. 基础战况】")
    print(f" ▪ 总回测时长 : {len(predictions_probs)} 小时")
    print(f" ▪ 撮合交易数 : {trade_count} 笔 (多:{long_trades} | 空:{short_trades})")
    print(f" ▪ 最终净值   : {equity_curve[-1]:.4f}")
    
    print(f"\n【2. 核心绩效】")
    print(f" ▪ 胜率 (WR)  : {win_rate:.2f}%")
    print(f" ▪ 盈亏比 (PF): {profit_factor:.2f}")
    print(f" ▪ 夏普比率   : {sharpe:.2f}")
    print(f" ▪ 卡玛比率   : {calmar:.2f}")
    print(f" ▪ 最大回撤   : {max_drawdown:+.2f}%")
    print(f" ▪ 恢复因子   : {recovery_factor:.2f}")
    print(f" ▪ 交易期望值 : {expectancy * 100:+.3f}% (每次发车平均净赚本金比例)")

    print(f"\n【3. 行为诊断 (出场分布)】")
    for reason, count in exit_reasons.items():
        ratio = (count / trade_count * 100) if trade_count > 0 else 0
        print(f" ▪ {reason:<16} : {count} 次 ({ratio:.1f}%)")

    if liquidated:
        print(f"\n💀 警报：该参数组合在回测期间触发了强制平仓(爆仓)！")
    print("="*50 + "\n")

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
    
    plot_path = MODEL_DIR / f"backtest_probabilistic_{EXPERIMENT_ID}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    print(f"\n📈 最终资金曲线已生成: {plot_path}")
    plt.close('all')

if __name__ == "__main__":
    run_backtest()