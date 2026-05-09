import time
import requests
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from historical_pipeline.model_itransformer import InvertedTransformer
from execution_oms import execute_gate_trade, log_system_alert
from risk_manager import bailout_worker
import threading
from state_manager import state
import sqlite3
import os
from quant_config.config import DATA_DIR

# ==========================================
# 1. 加载实盘配置 (对齐 1296 次寻优跑出的终极神级参数)
# ==========================================
SYMBOL = "BTC/USDT"
GATEWAY_URL = "http://127.0.0.1:8000/api/snapshot"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 336
D_MODEL = 64
E_LAYERS = 1
# 注意：自适应架构参数 (N_HEADS=4, D_FF=256) 会在初始化时自己计算

# 🚨 极其重要：把我们在网格搜索中跑出的神级模型的绝对路径填好！
# 如果你没改文件名，大概率长这样，请替换为你本地真实的路径
MODEL_PATH = "model/itransformer_BTCUSDT_best_EXP_xxxxx.pth" 
# 对应的概率门槛文件
JSON_PATH = "data/train/dynamic_thresholds_EXP_xxxxx.json"

# ==========================================
# 🧠 宏观情报读取器 (DeepSeek 接口)
# ==========================================
def get_macro_sentiment_score():
    """实时读取最新的宏观情绪得分"""
    db_path = os.path.join(DATA_DIR, "macro_intelligence.db")
    if not os.path.exists(db_path):
        return 0.0 # 找不到数据库，默认宏观中性

    try:
        # 使用超时机制，防止高频引擎被数据库文件锁卡死
        with sqlite3.connect(db_path, timeout=0.5) as conn:
            cursor = conn.cursor()
            # 永远只取最新的一条宏观决策
            cursor.execute("SELECT score FROM llm_decisions ORDER BY decision_ts DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return float(row[0])
    except Exception as e:
        print(f"⚠️ [宏观雷达] 读取情绪得分失败: {e}")
        
    return 0.0
# ==========================================
# 2. 特征工程构建器 (复刻回测逻辑)
# ==========================================
def compute_features(df_1h, snapshot):
    """根据历史K线和网关快照，构建与训练完全对齐的 23 维宽表"""
    df = df_1h.copy()
    
    # === 1. 基础价格与容量 ===
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_12h'] = df['log_return'].rolling(window=12).std()
    
    # === 2. 技术指标 (复刻) ===
    # MA & EMA
    df['ma_12'] = df['close'].rolling(window=12).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # === 3. 微观特征拼接 (关键融合点！) ===
    # 我们将网关传来的实时衍生品特征，"强行"拼接到最后一行
    # 训练时的衍生品特征可能比较粗糙，实盘中我们直接用 Binance 最新的真实数据覆盖
    df['funding_rate'] = snapshot['funding_rate']
    df['open_interest'] = snapshot['open_interest']
    df['liquidation_vol_15m'] = snapshot['liquidation_vol_15m']
    df['taker_buy_ratio_15m'] = snapshot['taker_buy_ratio_15m']
    
    # 填充前面的空值 (为了能够顺利通过 scaler)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    # === 🚨 确保特征顺序与训练时绝对一致！===
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'return', 'log_return', 'volatility_12h',
        'ma_12', 'ema_12', 'macd', 'macd_signal', 'macd_hist',
        'rsi_14', 'bb_mid', 'bb_upper', 'bb_lower', 'bb_width',
        'funding_rate', 'open_interest', 'liquidation_vol_15m', 'taker_buy_ratio_15m'
    ]
    
    return df[feature_cols]

# ==========================================
# 3. 实盘主循环
# ==========================================
def run_live():
    print("🚀 极速实盘预测引擎启动...")
    threading.Thread(target=bailout_worker, daemon=True).start()
    # 1. 加载神级模型
    print(f"📥 加载最优权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    
    dynamic_n_heads = max(2, D_MODEL // 16) 
    dynamic_d_ff = D_MODEL * 4
    
    model = InvertedTransformer(
        num_variates=len(feature_cols), 
        seq_len=SEQ_LEN, 
        d_model=D_MODEL, 
        n_heads=dynamic_n_heads, 
        d_ff=dynamic_d_ff, 
        e_layers=E_LAYERS
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. 加载对应的风控门槛
    with open(JSON_PATH, 'r') as f:
        thresholds = json.load(f)
    print(f"📥 加载分位数风控: {thresholds}")
    
    while True:
        try:
            # 1. 抓取网关切片
            resp = requests.get(f"{GATEWAY_URL}/{SYMBOL.replace('/', '_')}")
            if resp.status_code != 200:
                print(f"等待网关预热... {resp.text}")
                time.sleep(10)
                continue
                
            snapshot = resp.json()
            ohlcv = snapshot['ohlcv_1h']
            
            # 转为 DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 2. 算子提取与拼接
            features_df = compute_features(df, snapshot)
            
            # 3. 截取最近的 336 根 K 线
            if len(features_df) < SEQ_LEN:
                print(f"数据不足，当前 K 线数: {len(features_df)}")
                time.sleep(10)
                continue
                
            input_window = features_df.iloc[-SEQ_LEN:].values
            
            # 4. Z-Score 缩放 (必须用训练时的 Scaler)
            input_scaled = scaler.transform(input_window)
            
            # 5. 张量化送入大脑
            x_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            p_down, p_flat, p_up = probs[0], probs[1], probs[2]
            
            # ==========================================
            # 6. 风控撮合与动态微调 (微观主导，宏观辅助)
            # ==========================================
            # 1. 获取最新宏观情绪得分 (范围 [-1, 1])
            macro_score = get_macro_sentiment_score() 
            
            # 2. 💡 核心算法：微调 Softmax 置信度
            # MAX_MACRO_IMPACT 定义了宏观情绪的“最大辅助干预权限”。
            # 设置为 0.015 (1.5%) 意味着：即使 DeepSeek 给出最极端的 +1.0 狂热得分，
            # 它也只能给做多方向额外加上 1.5% 的概率辅助。
            # 由于模型的原生概率（主导力量）通常跨度高达 30%~60%，这 1.5% 只会在“临门一脚”时发挥作用！
            MAX_MACRO_IMPACT = 0.015 
            
            # 计算宏观情绪带来的概率 Buff (为正则利多，为负则利空)
            macro_buff = macro_score * MAX_MACRO_IMPACT
            
            # 将 Buff 叠加到 AI 的原生微观置信度上
            adj_p_up = p_up + macro_buff
            adj_p_down = p_down - macro_buff

            print(f"\n[{snapshot['server_time']}] 💰 {SYMBOL} 最新价: {snapshot['current_price']}")
            print(f"🧠 [宏观辅助] 情绪得分: {macro_score:+.2f} | 施加置信度偏移: {macro_buff:+.4f}")
            print(f"🎯 [原生微观] 📉 看跌: {p_down:.4f} | ⚪ 震荡: {p_flat:.4f} | 📈 看涨: {p_up:.4f}")
            print(f"⚖️ [综合裁决] 📉 修正看跌: {adj_p_down:.4f} | 📈 修正看涨: {adj_p_up:.4f}")
            
            signal = 0
            # 依然使用网格搜索跑出来的、不容篡改的物理阈值门槛！
            if adj_p_up > thresholds['up_top_5']:
                signal = 1
                print("🔥 [开仓指令] 综合置信度突破看涨红线！启动多头 OMS...")
            elif adj_p_down > thresholds['down_top_5']:
                signal = -1
                print("🔥 [开仓指令] 综合置信度突破看跌红线！启动空头 OMS...")
            else:
                print("⏳ 综合置信度被阈值拦截，持币观望...")
            
            # 👇 将信号发送给执行系统 (OMS)
            if signal != 0:
                try:
                    free_usdt = float(state.portfolio_memory.get('balance', {}).get('USDT', {}).get('free', 100.0))
                except Exception:
                    free_usdt = 100.0 
                    
                trade_margin = max(free_usdt * 0.05, 10.0) 
                
                # OMS 扣动扳机
                success, actual_margin = execute_gate_trade(
                    symbol=SYMBOL,
                    signal=signal,
                    leverage=2,
                    current_price=snapshot['current_price'],
                    tp_pct=0.08,
                    sl_pct=0.04,
                    risk_margin_usd=trade_margin
                )
                
                if success:
                    print(f"✅ 交易指令已成功送达并成交，动用保证金: ${actual_margin:.2f}")
                else:
                    print("❌ 交易指令执行失败，已被 OMS 拦截。")
                
        except Exception as e:
            print(f"❌ 预测引擎异常: {e}")
            
        # 每隔 15 秒预测一次 (因为订单薄和成交是秒级变的，即便 K 线没走完，特征也会变)
        time.sleep(15)

if __name__ == "__main__":
    run_live()