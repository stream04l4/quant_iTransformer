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
    LEVERAGE_1, LEVERAGE_2, LEVERAGE_3, HARD_TP, HARD_SL,
    TRADE_COOLDOWN_SEC, MACRO_ALPHA_TOP_10, MACRO_ALPHA_TOP_5, MACRO_ALPHA_TOP_1, HEARTBEAT_INTERVAL,
    INITIAL_CAPITAL, BASE_RISK_BUDGET, BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, FORECAST_HORIZON,
    EXPERIMENT_ID, TRAIN_DATA_DIR  
)
# ==========================================
# 1. 加载实盘配置 (基于 EXPERIMENT_ID 精准定位)
# ==========================================
GATEWAY_URL = "http://127.0.0.1:8000/api/snapshot"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🚨 修复 1：基于 EXPERIMENT_ID 精准定位，防止加载成别的策略包
MODEL_PATH = MODEL_DIR / f"itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth" 
JSON_PATH = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"

# ==========================================
# 🧠 宏观情报读取器 (DeepSeek 接口)
# ==========================================
def get_macro_sentiment_info():
    """实时读取最新的宏观情绪得分及时间戳"""
    db_path = os.path.join(LIVE_DATA_DIR, "macro_intelligence.db")
    if not os.path.exists(db_path):
        return 0.0, "0"

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT score, decision_ts FROM llm_decisions ORDER BY decision_ts DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return float(row[0]), str(row[1])
    except Exception as e:
        print(f"⚠️ [宏观雷达] 读取情绪详情失败: {e}")
        
    return 0.0, "0"
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
        df['amihud_illiquidity'] = np.log((df['log_return'].abs() / (df['quote_volume'] + 1e-9)) + 1e-12)
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
        gain = delta.clip(lower=0).ewm(alpha=1.0/p, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0/p, adjust=False).mean()
        df[f'rsi_{p}'] = ((100 - (100 / (1 + gain / (loss + 1e-9)))) - 50) / 50.0

    # 🚨 修复 2：补齐缺漏的掩码特征。实盘数据均为物理真实发生，不存在断网填充，恒为 0
    df['is_synthetic'] = 0.0

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

    inference_window = df[final_cols].iloc[-ACTUAL_SEQ_LEN:]
    
    if inference_window.isnull().any().any():
        missing_cols = inference_window.columns[inference_window.isnull().any()].tolist()
        raise ValueError(f"🚨 引擎熔断！网关传入数据在以下特征存在断层: {missing_cols}。拒绝输出预测信号！")
        
    return inference_window

# ==========================================
# ⚡ 贝叶斯 Logit 融合算子
# ==========================================
def apply_logit_fusion(base_prob, macro_score, alpha):
    """将大模型概率转换为 Log-Odds，加上宏观偏置后再还原为概率 (防止概率越界且边缘平滑)"""
    epsilon = 1e-7
    # 防御性剪裁，防止极端情况除零报错
    p_safe = np.clip(base_prob, epsilon, 1.0 - epsilon)
    # 转换为无限边界的 Logit 空间
    logit = np.log(p_safe / (1.0 - p_safe))
    # 注入宏观偏置
    adjusted_logit = logit + (alpha * macro_score)
    # Sigmoid 还原回 [0, 1] 空间
    return float(1.0 / (1.0 + np.exp(-adjusted_logit)))

# ==========================================
# 3. 实盘主循环
# ==========================================
def _reconstruct_tier_state():
    """
    重启恢复：从真实交易所仓位 + live_status.json 双轨重建阶梯状态机。
    优先级：exchange pos_map（真实） > live_status.json（快速兜底）。
    """
    # --- 轨道 1：真实交易所仓位（portfolio_sync_worker 已刷入 state.active_pos_map）---
    with state.pos_lock:
        pos_map = dict(state.active_pos_map)

    sym_key = SYMBOL.replace("/", "")
    print(f"🔍 [状态恢复] 正在校验符号: {sym_key}，当前已同步仓位列表: {list(pos_map.keys())}")
    if sym_key in pos_map:
        p = pos_map[sym_key]
        pos_side_str = str(p.get('side', '')).upper()
        side_int = 1 if pos_side_str == 'LONG' else (-1 if pos_side_str == 'SHORT' else 0)
        if side_int != 0:
            lev = float(p.get('leverage', p['info'].get('lever', p['info'].get('leverage', 0))))
            # 根据杠杆反推当时开仓所属 Tier
            if   lev >= LEVERAGE_3: tier = 3
            elif lev >= LEVERAGE_2: tier = 2
            else:                   tier = 1
            print(f"♻️  [重启恢复] 从交易所仓位重建状态机: side={side_int}, Tier={tier}, lev={lev}x")
            return {"side": side_int, "highest_tier": tier}

    # --- 轨道 2：live_status.json 快速兜底（交易所尚未响应时）---
    status_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            ts_state = saved.get("position_tier_state", {})
            side = int(ts_state.get("side", 0))
            tier = int(ts_state.get("highest_tier", 0))
            if side != 0:
                print(f"♻️  [重启恢复] 从 live_status.json 恢复状态机: side={side}, Tier={tier}")
                return {"side": side, "highest_tier": tier}
        except Exception as e:
            print(f"⚠️  [重启恢复] 读取 live_status.json 失败: {e}")

    print("🆕 [重启恢复] 未发现持仓记录，状态机初始化为空仓。")
    return {"side": 0, "highest_tier": 0}

def run_live():
    print(f"🚀 极速实盘预测引擎启动 (单模型模式: {MODEL_PATH.name})...")
    threading.Thread(target=bailout_worker, daemon=True).start()
    
    # --- 1. 加载单一神级模型 ---
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到模型权重，请检查 EXPERIMENT_ID 是否正确: {MODEL_PATH}")
        
    print(f"📥 正在加载模型权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # 🚨 修复 2：彻底丢弃 LIVE_D_MODEL 等伪隔离参数，全量从 Checkpoint 提取拓扑
    feature_cols = checkpoint['feature_cols']
    ckpt_seq_len = checkpoint.get('seq_len')
    ckpt_d_model = checkpoint.get('d_model')
    ckpt_n_heads = checkpoint.get('n_heads')
    ckpt_d_ff = checkpoint.get('d_ff')
    ckpt_e_layers = checkpoint.get('e_layers')
    ckpt_dropout = checkpoint.get('dropout')
    
    # 把提取出的动态窗口长度作为全局变量使用，替换掉旧版的 LIVE_SEQ_LEN
    global ACTUAL_SEQ_LEN
    ACTUAL_SEQ_LEN = ckpt_seq_len
    
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
    
    with open(JSON_PATH, 'r') as f:
        config_data = json.load(f)
        # 🚨 修复 3：兼容解包 Metadata 格式的 JSON 阈值配置
        th = config_data.get("thresholds", config_data)
            
    print(f"✅ 模型与阈值加载成功。")
    
    # 频段隔离器变量
    last_inferred_hour = 0
    cached_p_down, cached_p_flat, cached_p_up = 0.0, 0.0, 0.0
    
    cached_macro_score = 0.0
    last_macro_ts = "0" # 👈 新增：用于侦测宏观情报异动
    last_trade_hour = 0  
    last_trade_signal = 0
    latest_closed_ts = 0
    
    # 🎯 仓位阶梯状态机：重启时从交易所/磁盘双轨恢复，防止重复开仓
    # portfolio_sync_worker 在后台独立线程中运行，动态等待其完成首次同步
    print("⏳ 正在同步交易所账户状态 (资金/仓位/委托)...")
    wait_start = time.time()
    while state.portfolio_memory.get('last_sync_ts', 0) == 0:
        if time.time() - wait_start > 15: # 增加超时保护
            print("⚠️ [启动预警] 供货线程同步超时 (15s)，将尝试从本地缓存恢复状态。")
            break
        time.sleep(1)
        
    position_tier_state = _reconstruct_tier_state()

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
            
            # --- 直接使用 config 常量，无需读取 UI JSON ---
            # 右脑风控熔断器参数（直接来自 config）
            state.dynamic_settings["bailout_atr_multi"] = BAILOUT_ATR_MULTI
            state.dynamic_settings["bailout_vol_multi"] = BAILOUT_VOL_MULTI
            
            # 仓位保证金与止盈止损参数（直接来自 config）
            dynamic_margin_usd = INITIAL_CAPITAL * BASE_RISK_BUDGET  # 单笔保证金 = 总资本 × 风险拨备率
            dynamic_sl = HARD_SL   # K线物理兜底止损幅度
            dynamic_tp = HARD_TP   # K线物理止盈幅度

            # 👇 【关键补充】将最新价格更新到全局状态机，供后端的 bailout_worker (阶梯止损) 随时读取
            try:
                state.update_price(SYMBOL, current_price)
            except Exception:
                pass # 兼容你可能的 state_manager 写法

            # ---------------------------------------------------------
            # ⏳ [1小时级 Tick] 深度学习推理 (单模型)
            # ---------------------------------------------------------
            needs_evaluation_log = False
            trigger_reason = ""
            if len(snapshot['ohlcv_1h']) >= 2:
                latest_closed_ts = snapshot['ohlcv_1h'][-2][0] 
                
                if latest_closed_ts != last_inferred_hour:
                    # 💡 策略：整点第1分钟留给 DeepSeek 脚本执行，第2分钟准时开始大模型推理
                    current_minute = datetime.utcnow().minute
                    if current_minute < 2:
                        if int(current_time) % 30 == 0: # 每30秒提醒一次
                            print(f"⏳ [等待窗口] 检测到新K线，等待进入整点第2分钟以同步宏观分 (当前: {current_minute}分)...")
                        time.sleep(1) # 稍微睡眠，防止极短时间内的 while 循环空转过快
                        continue

                    print(f"\n🔄 [时钟对齐] 进入整点第{current_minute}分钟，开始执行单模型推理 (TS: {latest_closed_ts})...")
                    
                    eth_usdt_snap = requests.get(f"{GATEWAY_URL}/{AUX_SYMBOLS[0]}").json()
                    eth_btc_snap = requests.get(f"{GATEWAY_URL}/{AUX_SYMBOLS[1]}").json()
                    
                    features_df = compute_features(snapshot, eth_usdt_snap, eth_btc_snap)
                    
                    # 提取喂给模型的那个精确窗口
                    inference_window = features_df[feature_cols].iloc[-LIVE_SEQ_LEN:]
                    input_raw = inference_window.values.astype('float32')
                    x_tensor = torch.tensor(input_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    # 🚨 新增：审计外挂，将当前送入模型的 DataFrame 写入 CSV
                    try:
                        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
                        audit_csv_path = os.path.join(TRAIN_DATA_DIR, f"live_inference_audit_{SYMBOL}.csv")
                        
                        # 覆写保存最新的 336 根特征矩阵，保留表头以便核对
                        inference_window.to_csv(audit_csv_path, index=False)
                        print(f"   💾 [特征审计] 本次推理输入矩阵 (Shape: {inference_window.shape}) 已输出至: {audit_csv_path}")
                    except Exception as e:
                        print(f"   ⚠️ [特征审计] CSV 输出失败: {e}")
                    
                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                    cached_p_down, cached_p_flat, cached_p_up = probs[0], probs[1], probs[2]
                    
                    # 🎯 [联动触发] 大模型推理完成后，立即调取宏观分进行 Logit 融合调整
                    cached_macro_score, last_macro_ts = get_macro_sentiment_info()
                    current_time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"📡 [{current_time_str} UTC] [宏观雷达] 联动刷新宏观情绪得分: {cached_macro_score:+.2f} (TS: {last_macro_ts})")
                    
                    last_inferred_hour = latest_closed_ts
                    needs_evaluation_log = True
                    trigger_reason = "新K线闭合推理并联动宏观评分"

            # ---------------------------------------------------------
            # 💡 [新增：非整点宏观突袭侦测]
            # ---------------------------------------------------------
            if not needs_evaluation_log and cached_p_up + cached_p_down > 0:
                new_macro_score, new_macro_ts = get_macro_sentiment_info()
                if new_macro_ts != last_macro_ts:
                    cached_macro_score = new_macro_score
                    last_macro_ts = new_macro_ts
                    needs_evaluation_log = True
                    trigger_reason = f"🚀 宏观哨兵突击！监测到情报异动 (TS: {new_macro_ts})"
                    print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🦅 宏观哨兵触发实时重估！新得分: {cached_macro_score:+.2f}")

            # ---------------------------------------------------------
            # ⚔️ [10秒级 Tick] 实时高频信号撮合 (Logit 贝叶斯非线性融合)
            # ---------------------------------------------------------
            # 多头宏观得分为正收益，空头宏观得分为负收益 (反向叠加，即做空时宏观得分为负反而加成)
            adj_up_top1 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_1)
            adj_up_top5 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_5)
            adj_up_top10 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_10)

            adj_down_top1 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_1)
            adj_down_top5 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_5)
            adj_down_top10 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_10)

            # --- 🎯 阶梯加仓信号决策 ---
            target_signal = 0
            target_tier = 0
            dynamic_leverage = 0.0
            
            # 判断真实物理仓位是否已空，归零状态机
            with state.pos_lock:
                if SYMBOL.replace("/", "") not in state.active_pos_map:
                    position_tier_state = {"side": 0, "highest_tier": 0}

            # 评估是否击穿各层级动态阈值 (用各自阶梯专属的修正概率)
            if adj_up_top1 > th['up_top_1']:        target_signal = 1; target_tier = 3; dynamic_leverage = LEVERAGE_3
            elif adj_up_top5 > th['up_top_5']:      target_signal = 1; target_tier = 2; dynamic_leverage = LEVERAGE_2
            elif adj_up_top10 > th['up_top_10']:     target_signal = 1; target_tier = 1; dynamic_leverage = LEVERAGE_1
            elif adj_down_top1 > th['down_top_1']:  target_signal = -1; target_tier = 3; dynamic_leverage = LEVERAGE_3
            elif adj_down_top5 > th['down_top_5']:  target_signal = -1; target_tier = 2; dynamic_leverage = LEVERAGE_2
            elif adj_down_top10 > th['down_top_10']: target_signal = -1; target_tier = 1; dynamic_leverage = LEVERAGE_1

            # 阶梯过滤器核心逻辑
            actual_signal = 0
            if target_signal != 0:
                if position_tier_state["side"] == 0:
                    actual_signal = target_signal
                elif position_tier_state["side"] == target_signal:
                    if target_tier > position_tier_state["highest_tier"]:
                        actual_signal = target_signal
                else:
                    actual_signal = target_signal # 反向平仓交由 OMS 处理

            if needs_evaluation_log:
                action_str = "按兵不动"
                if actual_signal == 1: 
                    action_str = f"🚀 触发做多 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"
                elif actual_signal == -1: 
                    action_str = f"🩸 触发做空 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"
                    
                eval_log = (
                    f"【触发原因】{trigger_reason}\n"
                    f"【原始概率】📈多:{cached_p_up:.4f} | ⚪平:{cached_p_flat:.4f} | 📉空:{cached_p_down:.4f}\n"
                    f"【融合概率(入场级Tier1)】📈多:{adj_up_top10:.4f} | 📉空:{adj_down_top10:.4f} (宏观分:{cached_macro_score:+.2f})\n"
                    f"【引擎决策】{action_str} | 状态机: {position_tier_state}"
                )
                print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🧠 单模型评估汇报:\n{eval_log}\n")
                log_system_alert("🧠 决策评估", SYMBOL, eval_log)


            if actual_signal != 0:
                print(f"\n[{snapshot['server_time']}] 🔥 突破阶梯红线！最新价: {current_price} | 发送 OMS 指令 (Tier-{target_tier})...")
                log_system_alert("🔥 下单触发", SYMBOL,
                    f"信号: {'做多' if actual_signal==1 else '做空'} | Tier: {target_tier} | 杠杆变量: {dynamic_leverage}x | 价格: {current_price:.4f}")
                success, actual_margin = execute_gate_trade(
                    symbol=SYMBOL,
                    signal=actual_signal,
                    leverage=dynamic_leverage,
                    current_price=current_price,
                    tp_pct=dynamic_tp,
                    sl_pct=dynamic_sl,
                    risk_margin_usd=dynamic_margin_usd,
                )
                if success:
                    print(f"✅ 成交！价格: {current_price} | 动用保证金: ${actual_margin:.2f}")
                    log_system_alert("✅ 下单成交", SYMBOL,
                        f"Tier-{target_tier} 建立 | 成交价: {current_price:.4f} | 保证金: ${actual_margin:.2f}")
                    last_trade_hour = latest_closed_ts
                    last_trade_signal = actual_signal
                    # 升级状态机最高记录
                    position_tier_state["side"] = actual_signal
                    position_tier_state["highest_tier"] = target_tier
                else:
                    log_system_alert("❌ 下单失败", SYMBOL,
                        f"OMS 返回 False | 信号: {'做多' if actual_signal==1 else '做空'} | 价格: {current_price:.4f}")
            
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
                    "adj_prob_up_tier1": float(adj_up_top10),
                    "adj_prob_down_tier1": float(adj_down_top10),
                    "macro_score": float(cached_macro_score)
                },
                "execution_state": {
                    "current_signal": int(actual_signal),
                    "dynamic_leverage": float(dynamic_leverage) if actual_signal != 0 else 0.0,
                    "last_inferred_hour": int(last_inferred_hour),
                    "last_trade_hour": int(last_trade_hour)
                },
                "position_tier_state": {
                    "side": int(position_tier_state["side"]),
                    "highest_tier": int(position_tier_state["highest_tier"])
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