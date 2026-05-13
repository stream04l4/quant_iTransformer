import time
import requests
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from historical_pipeline.train_itransformer import InvertedTransformer
from execution_oms import execute_gate_trade, log_system_alert
from risk_manager import bailout_worker
import threading
from state_manager import state
import sqlite3
import os
from datetime import datetime
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR, TEMP_DIR, MODEL_DIR, SYMBOL, AUX_SYMBOLS,
    LIVE_SEQ_LEN, LIVE_D_MODEL, LIVE_E_LAYERS, LIVE_CLASS_THRESHOLD, LIVE_DROPOUT,
    LEVERAGE_1, LEVERAGE_2, LEVERAGE_3, HARD_TP, HARD_SL, LIVE_RISK_MARGIN_USD, 
    TRADE_COOLDOWN_SEC, MAX_MACRO_IMPACT, MACRO_CHECK_INTERVAL, HEARTBEAT_INTERVAL
)
# ==========================================
# 1. 加载实盘配置 (固定单模型路径)
# ==========================================
GATEWAY_URL = "http://127.0.0.1:8000/api/snapshot"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🚨 对齐用户指定的文件名
MODEL_PATH = MODEL_DIR / "itransformer_BTCUSDT_best_24h.pth" 
JSON_PATH = MODEL_DIR / "dynamic_thresholds_24h.json"

# ==========================================
# 🧠 宏观情报读取器 (DeepSeek 接口)
# ==========================================
def get_macro_sentiment_score():
    """实时读取最新的宏观情绪得分"""
    db_path = os.path.join(LIVE_DATA_DIR, "macro_intelligence.db")
    if not os.path.exists(db_path):
        return 0.0 # 找不到数据库，默认宏观中性

    try:
        # 使用超时机制，防止高频引擎被数据库文件锁卡死
        with sqlite3.connect(db_path, timeout=1.0) as conn:
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
# 2. 特征工程构建器 (严禁补零，严格对齐)
# ==========================================
def fetch_gateway_snapshot(sym):
    """从本地网关拉取指定币种快照"""
    # 格式化请求符号，兼容网关的 REST 解析
    request_sym = sym.replace("/", "") 
    try:
        resp = requests.get(f"{GATEWAY_URL}/{request_sym}", timeout=2)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

def compute_features(main_snap, eth_usdt_snap, eth_btc_snap):
    """纯粹的引擎映射层：100% 像素级复刻离线 Polars 特征构建逻辑，杜绝无限回填"""
    
    # 1. 基础 K 线体系
    df = pd.DataFrame(main_snap['ohlcv_1h'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    current_hour_ts = int(datetime.utcnow().timestamp() / 3600) * 3600000
    if df['timestamp'].iloc[-1] >= current_hour_ts:
        df = df.iloc[:-1]
    df['hour_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if len(df) < (LIVE_SEQ_LEN + 168):
        raise ValueError(f"主K线历史长度不足 (需>={LIVE_SEQ_LEN + 168}，当前{len(df)})")

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. 横截面辅助缝合
    df_ethbtc = pd.DataFrame(eth_btc_snap['ohlcv_1h'], columns=['timestamp', 'open_eb', 'high_eb', 'low_eb', 'eth_btc_close', 'volume_eb'])
    df = df.merge(df_ethbtc[['timestamp', 'eth_btc_close']], on='timestamp', how='left')

    # 3. 衍生品历史缝合
    if main_snap.get('premium_history'):
        df_prem = pd.DataFrame(main_snap['premium_history'], columns=['ts', 'premium_close'])
        df_prem['hour_timestamp'] = pd.to_datetime(df_prem['ts'], unit='ms')
        df = df.merge(df_prem[['hour_timestamp', 'premium_close']], on='hour_timestamp', how='left')
    else: df['premium_close'] = np.nan

    if main_snap.get('funding_history'):
        df_fund = pd.DataFrame(main_snap['funding_history'], columns=['ts', 'funding_rate'])
        df_fund['hour_timestamp'] = pd.to_datetime(df_fund['ts'], unit='ms').dt.floor('h')
        df = df.merge(df_fund[['hour_timestamp', 'funding_rate']], on='hour_timestamp', how='left')
    else: df['funding_rate'] = np.nan

    if main_snap.get('oi_history'):
        df_oi = pd.DataFrame(main_snap['oi_history'], columns=['ts', 'sum_open_interest'])
        df_oi['hour_timestamp'] = pd.to_datetime(df_oi['ts'], unit='ms')
        df = df.merge(df_oi[['hour_timestamp', 'sum_open_interest']], on='hour_timestamp', how='left')
    else: df['sum_open_interest'] = np.nan

    if main_snap.get('ls_history'):
        df_ls = pd.DataFrame(main_snap['ls_history'], columns=['ts', 'ls_ratio'])
        df_ls['hour_timestamp'] = pd.to_datetime(df_ls['ts'], unit='ms')
        df = df.merge(df_ls[['hour_timestamp', 'ls_ratio']], on='hour_timestamp', how='left')
    else: df['ls_ratio'] = np.nan

    # 🚨 修复 1：严格限制填充范围，杜绝无限回填。对齐离线 limit=8 与 limit=3 的物理设定
    df['funding_rate'] = df['funding_rate'].ffill(limit=8)
    state_cols = ['eth_btc_close', 'premium_close', 'sum_open_interest', 'ls_ratio']
    df[state_cols] = df[state_cols].ffill(limit=3)

    # 🚨 修复 2：彻底斩断 Parquet 依赖，全量使用网关推来的 35000 根纯净 1m K线
    if 'ohlcv_1m' in main_snap and len(main_snap['ohlcv_1m']) > 0:
        live_1m = pd.DataFrame(main_snap['ohlcv_1m'], columns=['ts', 'o', 'h', 'l', 'close', 'v'])
        live_1m['timestamp'] = pd.to_datetime(live_1m['ts'], unit='ms')
        live_1m['log_ret'] = np.log(live_1m['close'] / live_1m['close'].shift(1))
        live_1m['hour_timestamp'] = live_1m['timestamp'].dt.floor('h')
        
        rv_df = live_1m.groupby('hour_timestamp')['log_ret'].apply(lambda x: np.sqrt((x**2).sum())).reset_index(name='realized_vol_1h')
        df = df.merge(rv_df, on='hour_timestamp', how='left')
    else:
        df['realized_vol_1h'] = np.nan

    # 5. 订单流特征
    taker_stats = main_snap.get('taker_stats_1h', {})
    if taker_stats:
        taker_df = pd.DataFrame.from_dict(taker_stats, orient='index')
        taker_df.index = pd.to_datetime(taker_df.index.astype(int), unit='ms')
        df = df.merge(taker_df, left_on='hour_timestamp', right_index=True, how='left')
        
        df['taker_ratio_delta'] = (df['taker_buy_volume'] / (df['volume'] + 1e-9)).diff()
        df['cvd'] = (df['taker_buy_volume'] - df['taker_sell_volume']).fillna(0)
        df['whale_net_flow'] = (df['whale_buy'] - df['whale_sell']).fillna(0)
        df['retail_ratio'] = (df['retail_vol'] / (df['taker_buy_volume'] + df['taker_sell_volume'] + 1e-9)).ffill().fillna(0.1)
        df['amihud_illiquidity'] = df['log_return'].abs() / (df['quote_volume'] + 1e-9)
    else:
        for col in ['taker_ratio_delta', 'cvd', 'whale_net_flow', 'retail_ratio', 'amihud_illiquidity']:
            df[col] = np.nan

    # 6. 基础技术与 MTF
    ma_200 = df['close'].ewm(span=200, adjust=False).mean()
    df['ma_bias'] = (df['close'] - ma_200) / ma_200
    
    hl_range = df['high'] - df['low'] + 1e-9
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / hl_range
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / hl_range
    
    std_20 = df['close'].rolling(window=20).std()
    df['bb_squeeze'] = (std_20 * 4) / (df['close'].rolling(window=20).mean() + 1e-9)
    
    # 🚨 修复 3：补齐 adjust=False，彻底对齐离线 Polars 的平滑算法
    ppo = (df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()) / (df['close'].ewm(span=26, adjust=False).mean() + 1e-9)
    df['ppo_hist'] = ppo - ppo.ewm(span=9, adjust=False).mean()
    
    for p in [14, 48]:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=p, adjust=False).mean()
        df[f'rsi_{p}'] = ((100 - (100 / (1 + gain / (loss + 1e-9)))) - 50) / 50.0

    for w in [24, 72, 168]:
        low_min = df['low'].rolling(window=w).min()
        high_max = df['high'].rolling(window=w).max()
        df[f'price_pos_{w}h'] = (df['close'] - low_min) / (high_max - low_min + 1e-9)

    # 🚨 修复 4：还原原生三阶中心矩偏度公式，消除 Pandas 样本偏度带来的 6.7% 黑盒误差
    m1 = df['log_return'].rolling(window=24, min_periods=1).mean()
    m2 = (df['log_return']**2).rolling(window=24, min_periods=1).mean()
    m3 = (df['log_return']**3).rolling(window=24, min_periods=1).mean()
    var = (m2 - m1**2).abs()
    df['vol_skew_24h'] = (m3 - 3 * m1 * m2 + 2 * m1**3) / (var**1.5 + 1e-9)
    
    df['oi_delta_pct'] = (df['sum_open_interest'] - df['sum_open_interest'].shift(1)) / (df['sum_open_interest'].shift(1) + 1e-9)
    
    # 🚨 新增：补齐缺失的高阶特征
    df['max_ret_24h'] = df['log_return'].rolling(window=24).max()
    df['rv_term_structure'] = df['realized_vol_1h'] / (df['realized_vol_1h'].rolling(window=24).mean() + 1e-9)
    df['funding_rate_momentum'] = df['funding_rate'].rolling(window=24).mean() - df['funding_rate'].rolling(window=72).mean()

    # 7. 免疫断层的 Z-Score (🚨 修正：剔除 vpt，加入 oi_delta_pct)
    rolling_cols = ["realized_vol_1h", "amihud_illiquidity", "oi_delta_pct", "cvd", "whale_net_flow"]
    for c in rolling_cols:
        if c in df.columns:
            median = df[c].rolling(window=168, min_periods=1).median()
            mad = (df[c] - median).abs().rolling(window=168, min_periods=1).median()
            df[f"{c}_zscore"] = (0.6745 * (df[c] - median) / (mad + 1e-9)).clip(-4.0, 4.0)
        else:
            df[f"{c}_zscore"] = np.nan
            
    # 🚨 新增：基于 Z-Score 计算的高阶背离特征
    df['cvd_oi_divergence'] = df['cvd_zscore'] - df['oi_delta_pct_zscore']

    df['eth_btc_return'] = np.log(df['eth_btc_close'] / df['eth_btc_close'].shift(1))
    df['premium_delta'] = df['premium_close'].diff()
    df['hour_sin'] = np.sin(df['hour_timestamp'].dt.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour_timestamp'].dt.hour * (2 * np.pi / 24))
    df['week_sin'] = np.sin(df['hour_timestamp'].dt.weekday * (2 * np.pi / 7))
    df['week_cos'] = np.cos(df['hour_timestamp'].dt.weekday * (2 * np.pi / 7))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 8. 终极截取与熔断防线 (🚨 像素级对齐训练集特征与顺序)
    final_cols = [
        "log_return", "max_ret_24h", "ma_bias", "upper_shadow", "lower_shadow",
        "bb_squeeze", "ppo_hist", "rsi_14", "rsi_48", "taker_ratio_delta", 
        "amihud_illiquidity_zscore", "realized_vol_1h_zscore", "vol_skew_24h",
        "oi_delta_pct", "ls_ratio",
        "price_pos_24h", "price_pos_72h", "price_pos_168h", 
        "rv_term_structure", 
        "funding_rate_momentum", "premium_delta", "eth_btc_return", 
        "hour_sin", "hour_cos", "week_sin", "week_cos", 
        "cvd_zscore", "whale_net_flow_zscore", "retail_ratio", "cvd_oi_divergence"
    ]
    
    for col in final_cols:
        if col not in df.columns: df[col] = np.nan

    inference_window = df[final_cols].iloc[-LIVE_SEQ_LEN:]
    
    if inference_window.isnull().any().any():
        missing_cols = inference_window.columns[inference_window.isnull().any()].tolist()
        raise ValueError(f"🚨 引擎熔断！网关传入数据在以下特征存在断层: {missing_cols}。拒绝输出预测信号！")
        
    return inference_window

# ==========================================
# 3. 实盘主循环
# ==========================================
def run_live():
    print(f"🚀 极速实盘预测引擎启动 (单模型模式: {MODEL_PATH.name})...")
    threading.Thread(target=bailout_worker, daemon=True).start()
    
    # --- 1. 加载单一神级模型 ---
    print(f"📥 正在加载模型权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    
    # 动态构建网络拓扑
    dynamic_n_heads = max(2, LIVE_D_MODEL // 16) 
    dynamic_d_ff = LIVE_D_MODEL * 4
    
    model = InvertedTransformer(
        num_variates=len(feature_cols), 
        seq_len=LIVE_SEQ_LEN, 
        d_model=LIVE_D_MODEL, 
        n_heads=dynamic_n_heads, 
        d_ff=dynamic_d_ff, 
        e_layers=LIVE_E_LAYERS,
        dropout=LIVE_DROPOUT
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(JSON_PATH, 'r') as f:
        th = json.load(f)
            
    print(f"✅ 模型与阈值加载成功。")
    
    # 频段隔离器变量
    last_inferred_hour = 0
    cached_p_down, cached_p_flat, cached_p_up = 0.0, 0.0, 0.0
    
    last_macro_check = 0.0
    cached_macro_score = 0.0
    last_trade_hour = 0  
    last_trade_signal = 0
    latest_closed_ts = 0

    while True:
        try:
            # ---------------------------------------------------------
            # ⚡ [10秒级 Tick] 高频抓取与风控心跳
            # ---------------------------------------------------------
            resp = requests.get(f"{GATEWAY_URL}/{SYMBOL}", timeout=2)
            if resp.status_code != 200:
                time.sleep(2)
                continue
            snapshot = resp.json()
            current_time = time.time()
            current_price = snapshot['current_price']
            
            # --- 读入前端 UI 设置并应用 ---
            ui_config = {}
            ui_path = os.path.join(LIVE_DATA_DIR, "ui_config.json")
            if os.path.exists(ui_path):
                try:
                    with open(ui_path, "r") as f:
                        ui_config = json.load(f)
                except:
                    pass
            
            # 动态更新给右脑风控熔断器的参数
            state.dynamic_settings["bailout_atr_multi"] = float(ui_config.get("bailout_atr_multi", 2.5))
            state.dynamic_settings["bailout_vol_multi"] = float(ui_config.get("bailout_vol_multi", 3.0))
            state.dynamic_settings["excursion"] = float(ui_config.get("excursion", 1.5))
            
            # 动态计算仓位保证金与止盈止损参数
            initial_cap = float(ui_config.get("initial_capital", 10000.0))
            base_risk = float(ui_config.get("base_risk_budget", 0.02))
            dynamic_margin_usd = initial_cap * base_risk if initial_cap * base_risk > 0 else LIVE_RISK_MARGIN_USD
            
            # 从 UI 读取 6.0 并转为 0.06。现在的语义是：标的资产物理涨跌 6.0%
            dynamic_sl = float(ui_config.get("target_risk", HARD_SL * 100.0)) / 100.0
            dynamic_tp = float(ui_config.get("target_profit", HARD_TP * 100.0)) / 100.0

            # 👇 【关键补充】将最新价格更新到全局状态机，供后端的 bailout_worker (阶梯止损) 随时读取
            try:
                state.update_price(SYMBOL, current_price)
            except Exception:
                pass # 兼容你可能的 state_manager 写法

            # ---------------------------------------------------------
            # ⏱️ [30分钟级 Tick] 宏观情报低频轮询
            # ---------------------------------------------------------
            needs_evaluation_log = False
            trigger_reason = ""
            
            # 超过 MACRO_CHECK_INTERVAL 才去读一次数据库更新宏观得分
            if current_time - last_macro_check > MACRO_CHECK_INTERVAL:
                cached_macro_score = get_macro_sentiment_score()
                last_macro_check = current_time
                current_time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                print(f"📡 [{current_time_str} UTC] [宏观雷达] 已刷新宏观情绪得分: {cached_macro_score:+.2f}")
                needs_evaluation_log = True
                trigger_reason = "宏观得分刷新"

            # ---------------------------------------------------------
            # ⏳ [1小时级 Tick] 深度学习推理 (单模型)
            # ---------------------------------------------------------
            if len(snapshot['ohlcv_1h']) >= 2:
                latest_closed_ts = snapshot['ohlcv_1h'][-2][0] 
                
                if latest_closed_ts != last_inferred_hour:
                    print(f"\n🔄 [换线时刻] 开始执行单模型推理 (TS: {latest_closed_ts})...")
                    
                    eth_usdt_snap = requests.get(f"{GATEWAY_URL}/{AUX_SYMBOLS[0]}").json()
                    eth_btc_snap = requests.get(f"{GATEWAY_URL}/{AUX_SYMBOLS[1]}").json()
                    
                    features_df = compute_features(snapshot, eth_usdt_snap, eth_btc_snap)
                    input_raw = features_df[feature_cols].iloc[-LIVE_SEQ_LEN:].values.astype('float32')
                    x_tensor = torch.tensor(input_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                    cached_p_down, cached_p_flat, cached_p_up = probs[0], probs[1], probs[2]
                    last_inferred_hour = latest_closed_ts
                    needs_evaluation_log = True
                    trigger_reason = "新K线闭合推理"

            # ---------------------------------------------------------
            # ⚔️ [10秒级 Tick] 实时高频信号撮合
            # ---------------------------------------------------------
            dynamic_macro_impact = float(ui_config.get("max_macro_impact", MAX_MACRO_IMPACT * 100.0)) / 100.0
            macro_buff = cached_macro_score * dynamic_macro_impact
            
            adj_p_up = cached_p_up + macro_buff
            adj_p_down = cached_p_down - macro_buff

            # --- 单模型信号决策 ---
            signal = 0
            dynamic_leverage = 0.0
            
            if adj_p_up > th['up_top_1']:      signal = 1; dynamic_leverage = LEVERAGE_3
            elif adj_p_up > th['up_top_5']:    signal = 1; dynamic_leverage = LEVERAGE_2
            elif adj_p_up > th['up_top_10']:   signal = 1; dynamic_leverage = LEVERAGE_1
            elif adj_p_down > th['down_top_1']:  signal = -1; dynamic_leverage = LEVERAGE_3
            elif adj_p_down > th['down_top_5']:  signal = -1; dynamic_leverage = LEVERAGE_2
            elif adj_p_down > th['down_top_10']: signal = -1; dynamic_leverage = LEVERAGE_1

            # 👇 优化日志：移除“联席会议”字样，改为单模型汇报
            if needs_evaluation_log:
                action_str = "按兵不动"
                if signal == 1: action_str = f"🚀 触发做多 (杠杆: {dynamic_leverage}x)"
                elif signal == -1: action_str = f"🩸 触发做空 (杠杆: {dynamic_leverage}x)"
                    
                eval_log = (
                    f"【触发原因】{trigger_reason}\n"
                    f"【原始概率】📈多:{cached_p_up:.4f} | ⚪平:{cached_p_flat:.4f} | 📉空:{cached_p_down:.4f}\n"
                    f"【修正概率】📈多:{adj_p_up:.4f} | 📉空:{adj_p_down:.4f} (宏观修正: {macro_buff:+.4f})\n"
                    f"【引擎决策】{action_str}"
                )
                print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🧠 单模型评估汇报:\n{eval_log}\n")
                log_system_alert("🧠 决策评估", SYMBOL, eval_log)


            if signal != 0:
                if latest_closed_ts == last_trade_hour and signal == last_trade_signal:
                    pass # 同一根 K 线内同方向静默跳过
                else:
                    print(f"\n[{snapshot['server_time']}] 🔥 突破红线！最新价: {current_price} | 发送 OMS 指令 (杠杆: {dynamic_leverage}x)...")
                    success, actual_margin = execute_gate_trade(...)
                    if success:
                        print(f"✅ 成交！价格: {current_price} | 动用保证金: ${actual_margin:.2f}")
                        last_trade_hour = latest_closed_ts
                        last_trade_signal = signal
            
            # 👇 新增：引擎遥测状态原子级持久化 (Telemetry State Persistence)
            live_state_payload = {
                "timestamp": current_time,
                "server_time": snapshot.get('server_time', int(current_time * 1000)),
                "symbol": SYMBOL,
                "current_price": current_price,
                "macro_score": cached_macro_score,
                "model_inference": {
                    "raw_prob_up": float(cached_p_up),
                    "raw_prob_down": float(cached_p_down),
                    "adj_prob_up": float(adj_p_up),
                    "adj_prob_down": float(adj_p_down),
                    "macro_buff": float(macro_buff)
                },
                "execution_state": {
                    "current_signal": int(signal),
                    "dynamic_leverage": float(dynamic_leverage) if signal != 0 else 0.0,
                    "last_inferred_hour": int(last_inferred_hour),
                    "last_trade_hour": int(last_trade_hour)
                }
            }

            tmp_state_path = os.path.join(LIVE_DATA_DIR, "live_status.json.tmp")
            target_state_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
            
            try:
                # 1. 写入临时文件
                with open(tmp_state_path, "w", encoding="utf-8") as f:
                    json.dump(live_state_payload, f, ensure_ascii=False, indent=2)
                # 2. 操作系统级原子替换，杜绝外部进程读到不完整的 JSON
                os.replace(tmp_state_path, target_state_path)
            except IOError as e:
                print(f"⚠️ 状态持久化写入失败: {e}")
            # 👆 新增结束

        except Exception as e:
            print(f"❌ 主循环异常: {e}")
            
        # 稳定维持 10 秒的心跳，给 bailout_worker 提供充足的反应时间
        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    run_live()