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
from quant_config.config import SYMBOL, PROCESSED_MAIN_DIR, MODEL_DIR, TRAIN_DATA_DIR, TEMP_DIR, SEQ_LEN, D_MODEL, E_LAYERS, DROPOUT, CLASS_THRESHOLD, EXPERIMENT_ID,FEE_RATE,HOLD_PERIOD,HARD_TP,HARD_SL,LEVERAGE_1,LEVERAGE_2,LEVERAGE_3
from historical_pipeline.train_itransformer import InvertedTransformer


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

    # === 1. 独立加载特征空间与执行空间 ===
    df = pl.read_parquet(data_path)
    X_raw = df.select(feature_cols).to_numpy().astype('float32')
    timestamps = df.select("hour_timestamp").to_series().to_numpy()
    
    split_idx = int(len(X_raw) * 0.8)
    X_test_raw = X_raw[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    print("\n⚙️ 正在 H100 上生成概率矩阵 (Softmax 置信度)...")
    predictions_probs = []
    valid_timestamps = []
    
    class FastSeqDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.seq_len = seq_len
        def __len__(self):
            return len(self.data) - self.seq_len
        def __getitem__(self, idx):
            return self.data[idx : idx + self.seq_len]

    dataset = FastSeqDataset(X_test_raw, SEQ_LEN)
    # 🚨 关闭锁页内存
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions_probs.extend(probs)
            
    valid_timestamps = test_timestamps[SEQ_LEN - 1 : len(X_test_raw) - 1].tolist()

    # 极速加载 1h 和 1m 的真实物理价格 (执行空间)
    raw_kline_dir = TEMP_DIR / "klines" / SYMBOL
    try:
        df_1h = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1h.parquet")
        df_1m = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1m.parquet")
    except Exception as e:
        print(f"❌ 找不到原始 K 线，请先运行 build_main_features.py 完成下载！错误: {e}")
        return

    # 🚨 降维打击：全部强制转换为 int64 的秒级 Unix 绝对时间，彻底消除 NumPy 精度错乱
    ts_1h = df_1h["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    opens_1h = df_1h["open"].to_numpy()
    closes_1h = df_1h["close"].to_numpy()
    
    ts_1m = df_1m["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    highs_1m = df_1m["high"].to_numpy()
    lows_1m = df_1m["low"].to_numpy()
    closes_1m = df_1m["close"].to_numpy()

    # 专门为撮合引擎生成一份秒级信号时间戳 (画图依然保留原来的 datetime 格式)
    valid_timestamps_sec = np.array(valid_timestamps).astype('datetime64[s]').astype(np.int64)

    # ==========================================
    # 🏦 基于 1h 决策 + 1m 显微镜追踪的双轨状态机
    # ==========================================
    json_path = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"
    if not json_path.exists():
        print("❌ 找不到 dynamic_thresholds.json，请先运行 check_prob_distribution.py！")
        return
    with open(json_path, "r") as f:
        th = json.load(f)
    print("📥 已成功加载动态分位数阈值。")
    
    equity = 1.0             
    equity_curve = []
    benchmark_prices = []
    
    current_pos = 0          
    hold_timer_mins = 0           
    entry_price = 0.0        
    trade_start_equity = 1.0 
    current_leverage = 0.0   
    entry_time = None
    
    max_unrealized_pnl = 0.0 
    is_trailing = False      
    trade_count = 0
    win_count = 0
    liquidated = False       
    
    long_trades = 0
    short_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    print("\n⚔️ 正在执行工业级撮合 (1h 决策开仓 + 1m 盘中悲观碰撞)...")
    
    # 🚨 补充实盘摩擦力参数
    SLIPPAGE = 0.0002 # 单边滑点万分之二 (应对市价单冲击成本)
    HOURLY_FUNDING_COST = 0.0001 / 8 # 假设平均资金费率为万分之一/8小时，均摊到每小时

    for i in range(len(predictions_probs)):
        # 🚨 增加进度回显，打破静默卡死错觉
        if i > 0 and i % 1000 == 0:
            print(f"   ⏳ 已撮合 {i} / {len(predictions_probs)} 小时...")
            
        if liquidated:
            equity_curve.append(0)
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            continue
            
        p_down, p_flat, p_up = predictions_probs[i]
        signal_time = valid_timestamps_sec[i] 
        exec_time = signal_time + 3600  # 🚨 1小时 = 3600秒，极速纯整数加法
        exec_idx_1h = np.searchsorted(ts_1h, exec_time)
        
        if exec_idx_1h < len(closes_1h):
            benchmark_prices.append(closes_1h[exec_idx_1h])
        else:
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            break

        # ------------------------------------------
        # 1. 寻找开仓机会 (1h 级别)
        # ------------------------------------------
        if current_pos == 0:
            target_pos = 0
            dynamic_leverage = 0.0

            if p_up > th["up_top_1"]:      target_pos = 1; dynamic_leverage = LEVERAGE_3
            elif p_up > th["up_top_5"]:    target_pos = 1; dynamic_leverage = LEVERAGE_2
            elif p_up > th["up_top_10"]:   target_pos = 1; dynamic_leverage = LEVERAGE_1

            elif p_down > th["down_top_1"]:  target_pos = -1; dynamic_leverage = LEVERAGE_3
            elif p_down > th["down_top_5"]:  target_pos = -1; dynamic_leverage = LEVERAGE_2
            elif p_down > th["down_top_10"]: target_pos = -1; dynamic_leverage = LEVERAGE_1

            if target_pos != 0 and exec_idx_1h < len(opens_1h):
                current_pos = target_pos
                current_leverage = dynamic_leverage
                hold_timer_mins = HOLD_PERIOD * 60  # 时间锁转换为分钟
                
                # 💡 核心升级：按 exec_time 对应 K 线的开盘价直接进场，消除旧版的 1 小时滑点延迟
                entry_price = opens_1h[exec_idx_1h] 
                entry_time = ts_1h[exec_idx_1h]
                
                trade_start_equity = equity 
                equity -= trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                
                if target_pos == 1: long_trades += 1
                if target_pos == -1: short_trades += 1
                
                max_unrealized_pnl = 0.0
                is_trailing = False

        # ------------------------------------------
        # 2. 1分钟级别显微镜追踪持仓 (盘中碰撞)
        # ------------------------------------------
        if current_pos != 0:
            start_1m_idx = np.searchsorted(ts_1m, entry_time)
            chunk_end_time = exec_time + 3600 # 🚨 1小时 = 3600秒，极速纯整数加法
            end_1m_idx = np.searchsorted(ts_1m, chunk_end_time)
            
            close_position = False
            actual_exit_pct = 0.0
            close_pnl_pct = 0.0 # 初始化，防止当小时无数据
            
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

                worst_pnl_pct = current_leverage * worst_intra
                best_pnl_pct = current_leverage * best_intra
                close_pnl_pct = current_leverage * close_return
                
                if best_pnl_pct > max_unrealized_pnl:
                    max_unrealized_pnl = best_pnl_pct
                    
                hold_timer_mins -= 1
                
                # 分钟级极限爆仓检测
                if trade_start_equity * (1 - current_leverage * FEE_RATE + worst_pnl_pct) <= 0:
                    liquidated = True
                    break
                    
                active_sl_pct = -abs(HARD_SL)
                if is_trailing:
                    # 💡 修正：正常的阶梯追踪止损，跟随最高浮盈上移。
                    # 当且仅当浮盈减去回撤容忍度(5%)后依然大于0时，止损线才会变成正数锁定利润。
                    # 否则，允许止损线在硬止损 (HARD_SL) 和成本线之间活动，给予行情呼吸空间。
                    trailing_sl = max_unrealized_pnl - 0.05
                    active_sl_pct = max(-abs(HARD_SL), trailing_sl)
                
                # 🚨 极其严苛的回测规则：同一分钟内若既触发止损又触发止盈，【优先判定止损】！
                # 这能彻底消灭“天地针”带来的回测虚假乐观。
                
                # 🎯 盘中悲观离场：跌破止损线 / 跌破保本线
                if worst_pnl_pct <= active_sl_pct:
                    close_position = True
                    actual_exit_pct = min(active_sl_pct, close_pnl_pct)
                    break 

                # 🎯 盘中绝对止盈离场：瞬间碰触 50% 止盈线
                if best_pnl_pct >= abs(HARD_TP):
                    close_position = True
                    actual_exit_pct = abs(HARD_TP)
                    break
                
                # 🎯 时间锁到期判定
                if hold_timer_mins <= 0:
                    if close_pnl_pct <= 0:
                        close_position = True
                        actual_exit_pct = close_pnl_pct
                        break
                    else:
                        is_trailing = True

            # 如果这 1 小时走完了都没触发平仓，检查 AI 是否发生了概率反转
            if not close_position and not liquidated:
                if current_pos == 1 and p_down > th["down_top_1"]:
                    close_position = True
                    actual_exit_pct = close_pnl_pct
                elif current_pos == -1 and p_up > th["up_top_1"]:
                    close_position = True
                    actual_exit_pct = close_pnl_pct

            # 统一交割结算
            if liquidated:
                equity = 0
                current_pos = 0
                equity_curve.append(0)
                print(f"💀 盘中分钟级扎针爆仓！发生于 {signal_time}")
                trade_count += 1
                gross_loss += trade_start_equity
                continue
                
            if close_position:
                floating_equity = trade_start_equity * (1 - current_leverage * FEE_RATE + actual_exit_pct)
                floating_equity -= trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE) # 扣除平仓手续费
                equity = floating_equity
                
                trade_pnl = equity - trade_start_equity
                if trade_pnl > 0: 
                    win_count += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)
                
                trade_count += 1
                current_pos = 0 
                current_leverage = 0.0
            else:
                # 若持仓跨越当前小时，账面浮盈以本小时最后一分钟收盘价计
                equity = trade_start_equity * (1 - current_leverage * FEE_RATE + close_pnl_pct)
                equity -= trade_start_equity * current_leverage * HOURLY_FUNDING_COST
                # 将追踪起点推移至下一个 1 小时的开端
                entry_time = chunk_end_time
                
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    valid_timestamps = valid_timestamps[:len(equity_curve)]
    benchmark_prices = np.array(benchmark_prices)

    # 计算最大回撤 (Max Drawdown)
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    # 计算盈亏比 (Profit Factor)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # 计算多空比
    ls_ratio = (long_trades / short_trades) if short_trades > 0 else float('inf')

    # 计算卡玛比率 (Calmar Ratio)
    total_return = equity_curve[-1] - 1.0
    years = len(predictions_probs) / 8760
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = (annualized_return / (abs(max_drawdown)/100)) if max_drawdown < 0 else float('inf')
    
    # 修复基准对齐：动态构建的 benchmark_prices 已经完全对齐了每次开仓时刻
    if len(benchmark_prices) > 0:
        benchmark_curve = benchmark_prices / benchmark_prices[0]
    else:
        benchmark_curve = np.ones(len(predictions_probs))

    if trade_count > 0:
        win_rate = (win_count / trade_count) * 100
    else:
        win_rate = 0.0
        
    returns_series = pd.Series(equity_curve).pct_change().fillna(0).to_numpy()
    returns_series[0] = 0 
    if np.std(returns_series) != 0:
        sharpe = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(8760)
    else:
        sharpe = 0.0

    print("\n📊 终极 3分类动态杠杆 回测结果:")
    print("-" * 45)
    print(f"总回测时长: {len(predictions_probs)} 小时")
    print(f"共计交易: {trade_count} 笔 (多头: {long_trades} | 空头: {short_trades} | 多空比: {ls_ratio:.2f})")
    print(f"胜率 (Win Rate): {win_rate:.2f}% | 盈亏比 (Profit Factor): {profit_factor:.2f}")
    if liquidated:
        print(f"⚠️ 状态: 已爆仓 (账户归零)")
    print(f"期末策略净值: {equity_curve[-1]:.4f}")
    print(f"最大回撤 (Max Drawdown): {max_drawdown:.2f}%")
    print(f"估算年化夏普: {sharpe:.2f} | 卡玛比率: {calmar:.2f}")
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
    
    plot_path = MODEL_DIR / f"backtest_probabilistic_{EXPERIMENT_ID}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    print(f"\n📈 最终资金曲线已生成: {plot_path}")
    plt.close('all')

if __name__ == "__main__":
    run_backtest()