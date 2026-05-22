import time
import requests
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from historical_pipeline.train_itransformer import InvertedTransformer
from execution_oms import execute_gate_trade, get_gate_stop_coverage_size, log_system_alert
from risk_manager import bailout_worker
import threading
from state_manager import state
import sqlite3
import os
from datetime import datetime
from quant_config.config import (
    LIVE_DATA_DIR, MODEL_DIR, SYMBOL, AUX_SYMBOLS,
    LEVERAGE_1, LEVERAGE_2, LEVERAGE_3, HARD_TP, HARD_SL,
    MACRO_ALPHA_TOP_10, MACRO_ALPHA_TOP_5, MACRO_ALPHA_TOP_1, HEARTBEAT_INTERVAL,
    INITIAL_CAPITAL, BASE_RISK_BUDGET, BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, FORECAST_HORIZON,
    EXPERIMENT_ID, TRAIN_DATA_DIR  
)
from dotenv import load_dotenv
load_dotenv()
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

def assert_hourly_continuity(df):
    """离线端会 upsample 到连续 1h 轴；实盘端遇到断档直接熔断，避免跨空洞 rolling。"""
    gaps = df['timestamp'].diff().dropna()
    bad_gaps = gaps[gaps != 3600000]
    if not bad_gaps.empty:
        gap_idx = int(bad_gaps.index[0])
        prev_ts = int(df.loc[gap_idx - 1, 'timestamp']) if gap_idx - 1 in df.index else None
        curr_ts = int(df.loc[gap_idx, 'timestamp'])
        raise ValueError(
            "🚨 引擎熔断！主K线时间轴存在断层，离线训练端会将其标记为 synthetic 并剔除，"
            f"实盘拒绝跨断层构建特征: prev={prev_ts}, curr={curr_ts}, gap_ms={int(bad_gaps.iloc[0])}"
        )

def compute_features(main_snap, aux_reference_snap=None, direct_cross_snap=None):
    """纯粹的引擎映射层：100% 像素级复刻离线 Polars 特征构建逻辑，杜绝无限回填"""
    
    # 1. 基础 K 线体系
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(main_snap['ohlcv_1h'])
    if df.shape[1] < len(ohlcv_cols):
        raise ValueError(f"主K线字段不足，至少需要 {len(ohlcv_cols)} 列，当前 {df.shape[1]} 列")
    df = df.iloc[:, :len(ohlcv_cols)]
    df.columns = ohlcv_cols
    current_hour_ts = int(datetime.utcnow().timestamp() / 3600) * 3600000
    if df['timestamp'].iloc[-1] >= current_hour_ts:
        df = df.iloc[:-1]
    df = df.reset_index(drop=True)
    assert_hourly_continuity(df)
    df['hour_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    required_history = ACTUAL_SEQ_LEN + 336
    if len(df) < required_history:
        raise ValueError(f"主K线历史长度不足 (需>={required_history}，当前{len(df)})")

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. 横截面辅助缝合：优先使用直接交叉对，否则回退到“辅助现货/主币”合成。
    if aux_reference_snap and aux_reference_snap.get('ohlcv_1h'):
        df_aux = pd.DataFrame(
            aux_reference_snap['ohlcv_1h'],
            columns=['timestamp', 'open_aux', 'high_aux', 'low_aux', 'aux_reference_close', 'volume_aux']
        )
        df = df.merge(df_aux[['timestamp', 'aux_reference_close']], on='timestamp', how='left')
    else:
        df['aux_reference_close'] = np.nan

    if direct_cross_snap and direct_cross_snap.get('ohlcv_1h'):
        df_cross = pd.DataFrame(
            direct_cross_snap['ohlcv_1h'],
            columns=['timestamp', 'open_cross', 'high_cross', 'low_cross', 'cross_pair_close', 'volume_cross']
        )
        df = df.merge(df_cross[['timestamp', 'cross_pair_close']], on='timestamp', how='left')
    else:
        df['cross_pair_close'] = np.nan

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

    # 训练端只允许资金费率按物理周期有限延续；其它缺口保留 NaN，交给最终熔断处理。
    df['funding_rate'] = df['funding_rate'].ffill(limit=8)
    df['cross_asset_close'] = df['cross_pair_close'].combine_first(df['aux_reference_close'] / (df['close'] + 1e-9))

    # 🚨 修复 2：彻底斩断 Parquet 依赖，全量使用网关推来的完整 1m 纯净 K线
    if 'ohlcv_1m' in main_snap and len(main_snap['ohlcv_1m']) > 0:
        live_1m = pd.DataFrame(main_snap['ohlcv_1m'], columns=['ts', 'o', 'h', 'l', 'close', 'v'])
        live_1m['timestamp'] = pd.to_datetime(live_1m['ts'], unit='ms')
        live_1m['log_ret'] = np.log(live_1m['close'] / live_1m['close'].shift(1))
        live_1m['hour_timestamp'] = live_1m['timestamp'].dt.floor('h')
        
        rv_df = live_1m.groupby('hour_timestamp')['log_ret'].apply(lambda x: np.sqrt((x**2).sum())).reset_index(name='realized_vol_1h')
        df = df.merge(rv_df, on='hour_timestamp', how='left')
        rv_missing = df['realized_vol_1h'].isna().sum()
        if rv_missing:
            first_rv = rv_df['hour_timestamp'].min() if not rv_df.empty else None
            last_rv = rv_df['hour_timestamp'].max() if not rv_df.empty else None
            print(
                f"   ⚠️ [RV诊断] 1m覆盖 {len(live_1m)} 根/{rv_df.shape[0]} 小时，"
                f"RV范围 {first_rv} -> {last_rv}，主窗口缺失 {rv_missing} 小时。"
            )
    else:
        df['realized_vol_1h'] = np.nan
        
    # 5. 订单流特征
    taker_stats = main_snap.get('taker_stats_1h', {})
    if taker_stats:
        taker_df = pd.DataFrame.from_dict(taker_stats, orient='index')
        taker_df.index = pd.to_datetime(taker_df.index.astype(int), unit='ms')
        df = df.merge(taker_df, left_on='hour_timestamp', right_index=True, how='left')
        
        df['taker_ratio_delta'] = (df['taker_buy_volume'] / (df['volume'] + 1e-9)).diff()
        df['cvd'] = df['taker_buy_volume'] - df['taker_sell_volume']
        df['whale_net_flow'] = df['whale_buy'] - df['whale_sell']
        df['whale_ls_ratio'] = (df['whale_buy'] + 1e-9) / (df['whale_sell'] + 1e-9)
        df['retail_ratio'] = df['retail_vol'] / (df['taker_buy_volume'] + df['taker_sell_volume'] + 1e-9)
        df['amihud_illiquidity'] = np.log((df['log_return'].abs() / (df['quote_volume'] + 1e-9)) + 1e-12).clip(-50.0, 50.0)
    else:
        for col in ['taker_ratio_delta', 'cvd', 'whale_net_flow', 'whale_ls_ratio', 'retail_ratio', 'amihud_illiquidity']:
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

    # 与离线 upsample 后的 synthetic 规则保持一致；若进入推理窗口，后续空值检查会熔断。
    df['is_synthetic'] = (
        df['close'].isna() | df['sum_open_interest'].isna() | df['premium_close'].isna()
    ).astype(float)

    for w in [24, 72, 168]:
        low_min = df['low'].rolling(window=w).min()
        high_max = df['high'].rolling(window=w).max()
        df[f'price_pos_{w}h'] = (df['close'] - low_min) / (high_max - low_min + 1e-9)

    # 🚨 修复 4：还原原生三阶中心矩偏度公式，消除 Pandas 样本偏度带来的 6.7% 黑盒误差
    m1 = df['log_return'].rolling(window=24, min_periods=24).mean()
    m2 = (df['log_return']**2).rolling(window=24, min_periods=24).mean()
    m3 = (df['log_return']**3).rolling(window=24, min_periods=24).mean()
    var = (m2 - m1**2).abs()
    df['vol_skew_24h'] = (m3 - 3 * m1 * m2 + 2 * m1**3) / (var**1.5 + 1e-15)
    
    prev_oi = df['sum_open_interest'].shift(1)
    
    df['oi_delta_pct'] = np.where(
        prev_oi.abs() > 1e-6,
        ((df['sum_open_interest'] - prev_oi) / prev_oi).clip(-1.0, 1.0),
        np.nan
    )
    
    # 🚨 新增：补齐缺失的高阶特征
    df['max_ret_24h'] = df['log_return'].rolling(window=24).max()
    df['rv_term_structure'] = (
        df['realized_vol_1h'] / (df['realized_vol_1h'].rolling(window=24).mean() + 1e-9)
    ).clip(0.0, 50.0)
    df['funding_rate_momentum'] = df['funding_rate'].rolling(window=24).mean() - df['funding_rate'].rolling(window=72).mean()

    # 7. 免疫断层的 Z-Score (🚨 修正：剔除 vpt，加入 oi_delta_pct)
    rolling_cols = ["realized_vol_1h", "amihud_illiquidity", "oi_delta_pct", "cvd", "whale_net_flow", "whale_ls_ratio"]
    for c in rolling_cols:
        if c in df.columns:
            # 与离线训练完全对齐：环境基准必须只使用 t-1 以前的数据。
            median = df[c].rolling(window=168, min_periods=120).median().shift(1)
            mad = (df[c] - median).abs().rolling(window=168, min_periods=120).median().shift(1)
            df[f"{c}_zscore"] = (0.6745 * (df[c] - median) / (mad + 1e-9)).clip(-4.0, 4.0)
        else:
            df[f"{c}_zscore"] = np.nan
            
    # 🚨 新增：基于 Z-Score 计算的高阶背离特征
    df['cvd_oi_divergence'] = df['cvd_zscore'] - df['oi_delta_pct_zscore']

    df['cross_asset_return'] = np.log(df['cross_asset_close'] / df['cross_asset_close'].shift(1))
    df['premium_delta'] = df['premium_close'].diff()
    df['hour_sin'] = np.sin(df['hour_timestamp'].dt.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour_timestamp'].dt.hour * (2 * np.pi / 24))
    # Polars dt.weekday() 是 ISO weekday: Monday=1..Sunday=7；Pandas weekday 是 0..6。
    iso_weekday = df['hour_timestamp'].dt.weekday + 1
    df['week_sin'] = np.sin(iso_weekday * (2 * np.pi / 7))
    df['week_cos'] = np.cos(iso_weekday * (2 * np.pi / 7))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    final_cols = [
        "is_synthetic", "log_return", "max_ret_24h", "ma_bias", "upper_shadow", "lower_shadow",
        "bb_squeeze", "ppo_hist", "rsi_14", "rsi_48", "taker_ratio_delta", 
        "amihud_illiquidity_zscore", "realized_vol_1h_zscore", "vol_skew_24h",
        "oi_delta_pct",
        "price_pos_24h", "price_pos_72h", "price_pos_168h", 
        "rv_term_structure", 
        "funding_rate_momentum", "premium_delta", "cross_asset_return",
        "hour_sin", "hour_cos", "week_sin", "week_cos", 
        "cvd_zscore", "whale_net_flow_zscore", "whale_ls_ratio_zscore", "retail_ratio", "cvd_oi_divergence"
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
def _sanitize_tier_state(raw_state):
    """统一清洗状态机结构，防止脏值把决策闸门带偏。"""
    raw_state = raw_state or {}
    try:
        side = int(raw_state.get("side", 0))
    except (TypeError, ValueError):
        side = 0
    if side not in (-1, 0, 1):
        side = 0

    try:
        highest_tier = int(raw_state.get("highest_tier", 0))
    except (TypeError, ValueError):
        highest_tier = 0
    highest_tier = max(0, min(3, highest_tier))

    if side == 0:
        highest_tier = 0

    return {"side": side, "highest_tier": highest_tier}


def _tier_from_leverage(leverage):
    """根据交易所当前杠杆推导最低可信的阶梯层级。"""
    if leverage >= LEVERAGE_3:
        return 3
    if leverage >= LEVERAGE_2:
        return 2
    return 1


def _read_persisted_tier_state():
    """读取上一次落盘的状态机，作为交易所快照未就绪时的兜底。"""
    status_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if not os.path.exists(status_path):
        return {"side": 0, "highest_tier": 0}

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return _sanitize_tier_state(saved.get("position_tier_state", {}))
    except Exception as e:
        print(f"⚠️  [状态恢复] 读取 live_status.json 失败: {e}")
        return {"side": 0, "highest_tier": 0}


def _get_exchange_position_state():
    """从供货线程同步过来的真实仓位快照推导当前状态。"""
    with state.pos_lock:
        pos_map = dict(state.active_pos_map)

    sym_key = SYMBOL.replace("/", "")
    last_sync_ts = float(state.portfolio_memory.get("last_sync_ts", 0) or 0)
    sync_age = float("inf") if last_sync_ts <= 0 else max(0.0, time.time() - last_sync_ts)
    sync_fresh = sync_age <= max(HEARTBEAT_INTERVAL * 3, 30)

    result = {
        "side": 0,
        "highest_tier": 0,
        "exchange_tier": 0,
        "has_position": False,
        "sync_fresh": sync_fresh,
        "sync_age_sec": sync_age,
        "source": "exchange"
    }

    p = pos_map.get(sym_key)
    if not p:
        return result

    pos_side_str = str(p.get('side', '')).upper()
    side_int = 1 if pos_side_str == 'LONG' else (-1 if pos_side_str == 'SHORT' else 0)
    if side_int == 0:
        return result

    info = p.get("info", {}) or {}
    try:
        lev = float(p.get('leverage', info.get('lever', info.get('leverage', 0))) or 0)
    except (TypeError, ValueError):
        lev = 0.0

    tier = _tier_from_leverage(lev) if lev > 0 else 1
    result.update({
        "side": side_int,
        "highest_tier": tier,
        "exchange_tier": tier,
        "has_position": True
    })
    return result


def _reconcile_position_tier_state(cached_state):
    """
    每个 tick 都重新对账：
    1. 真实仓位决定 side。
    2. highest_tier 取 exchange / 内存 / 落盘三者中同向的最高值，避免初始化错一次后持续漂移。
    3. 若供货线程快照陈旧，则退化到缓存/磁盘兜底，但不会永久锁死。
    """
    cached_state = _sanitize_tier_state(cached_state)
    persisted_state = _read_persisted_tier_state()
    exchange_state = _get_exchange_position_state()

    if exchange_state["has_position"]:
        candidates = [exchange_state["exchange_tier"]]
        for fallback_state in (cached_state, persisted_state):
            if fallback_state["side"] == exchange_state["side"]:
                candidates.append(fallback_state["highest_tier"])

        reconciled = {
            "side": exchange_state["side"],
            "highest_tier": max(candidates)
        }
        meta = {
            "source": "exchange_authoritative",
            "sync_fresh": exchange_state["sync_fresh"],
            "sync_age_sec": exchange_state["sync_age_sec"]
        }
        return reconciled, meta

    if exchange_state["sync_fresh"]:
        return {"side": 0, "highest_tier": 0}, {
            "source": "exchange_flat",
            "sync_fresh": True,
            "sync_age_sec": exchange_state["sync_age_sec"]
        }

    fallback_state = cached_state if cached_state["side"] != 0 else persisted_state
    fallback_source = "cached_memory_fallback" if cached_state["side"] != 0 else "disk_fallback"
    return fallback_state, {
        "source": fallback_source,
        "sync_fresh": False,
        "sync_age_sec": exchange_state["sync_age_sec"]
    }


def _reconstruct_tier_state():
    """启动时先做一次对账恢复，后续每个 tick 会继续自动校正。"""
    with state.pos_lock:
        pos_keys = list(state.active_pos_map.keys())
    print(f"🔍 [状态恢复] 正在校验符号: {SYMBOL.replace('/', '')}，当前已同步仓位列表: {pos_keys}")

    reconstructed, meta = _reconcile_position_tier_state({"side": 0, "highest_tier": 0})
    if reconstructed["side"] != 0:
        print(
            f"♻️  [重启恢复] 状态机已恢复: side={reconstructed['side']}, "
            f"Tier={reconstructed['highest_tier']} | source={meta['source']}"
        )
    else:
        print(f"🆕 [重启恢复] 初始化为空仓 | source={meta['source']}")
    return reconstructed


def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _load_recent_local_open_trades(limit=20):
    """读取本地开仓流水，用于判断当前库存是否仍在策略可解释范围内。"""
    db_path = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{_normalize_symbol_key(SYMBOL)}.db")
    if not os.path.exists(db_path):
        return []

    try:
        with sqlite3.connect(db_path, timeout=3.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT exec_ts, side, amount, fill_price, margin_used, leverage
                FROM live_trades
                WHERE side IN ('buy', 'sell')
                ORDER BY exec_ts DESC
                LIMIT ?
                """,
                (int(limit),)
            )
            rows = cursor.fetchall()
    except Exception as e:
        print(f"⚠️ [库存守卫] 读取本地交易流水失败: {e}")
        return []

    records = []
    for exec_ts, side, amount, fill_price, margin_used, leverage in rows:
        try:
            parsed_amount = float(amount or 0.0)
        except (TypeError, ValueError):
            parsed_amount = 0.0
        try:
            parsed_fill = float(fill_price or 0.0)
        except (TypeError, ValueError):
            parsed_fill = 0.0
        try:
            parsed_margin = float(margin_used or 0.0)
        except (TypeError, ValueError):
            parsed_margin = 0.0
        try:
            parsed_leverage = float(leverage or 0.0)
        except (TypeError, ValueError):
            parsed_leverage = 0.0

        records.append({
            "exec_ts": str(exec_ts),
            "side": str(side or "").lower(),
            "amount": parsed_amount,
            "fill_price": parsed_fill,
            "margin_used": parsed_margin,
            "leverage": parsed_leverage
        })

    return records


def _get_forced_exit_lock(symbol, latest_closed_ts):
    """
    若本小时内发生过强制退出，则锁住当前这根已闭合 K 线，直到下一根 K 线闭合后才允许重新评估入场。
    """
    sym_key = _normalize_symbol_key(symbol)
    try:
        current_closed_ts = int(latest_closed_ts or 0)
    except (TypeError, ValueError):
        current_closed_ts = 0

    with state.event_lock:
        lock_info = dict(state.forced_exit_locks.get(sym_key, {}) or {})
        if not lock_info:
            return {"active": False}

        anchor_closed_ts = int(lock_info.get("anchor_closed_ts", 0) or 0)
        if current_closed_ts > 0 and current_closed_ts > anchor_closed_ts:
            state.forced_exit_locks.pop(sym_key, None)
            return {
                "active": False,
                "released": True,
                "released_anchor_closed_ts": anchor_closed_ts,
            }

    lock_info["active"] = True
    return lock_info


def _inspect_live_stop_coverage(position):
    """
    校验当前持仓是否被交易所止损单完整覆盖。
    这里只做告警与隔离判定，不擅自修改挂单单位口径。
    """
    info = position.get("info", {}) or {}
    try:
        expected_contracts = abs(float(info.get("size", position.get("contracts", 0)) or 0.0))
    except (TypeError, ValueError):
        expected_contracts = 0.0

    snapshot_stops = list(state.portfolio_memory.get("open_stops", []) or [])
    sym_key = _normalize_symbol_key(position.get("symbol", SYMBOL))
    matched_stops = [o for o in snapshot_stops if _normalize_symbol_key(o.get("symbol")) == sym_key]

    if expected_contracts <= 0:
        return {
            "status": "no_position",
            "expected_contracts": 0.0,
            "matched_stop_count": len(matched_stops),
            "max_stop_size": 0.0,
            "coverage_ratio": 1.0,
            "reason": ""
        }

    if not matched_stops:
        return {
            "status": "missing_stop",
            "expected_contracts": expected_contracts,
            "matched_stop_count": 0,
            "max_stop_size": 0.0,
            "coverage_ratio": 0.0,
            "reason": "交易所未查询到任何活动止损单"
        }

    parsed_sizes = []
    for stop_order in matched_stops:
        parsed_size = get_gate_stop_coverage_size(stop_order, expected_contracts)
        if parsed_size is not None:
            parsed_sizes.append(parsed_size)

    if not parsed_sizes:
        return {
            "status": "unparseable_stop",
            "expected_contracts": expected_contracts,
            "matched_stop_count": len(matched_stops),
            "max_stop_size": 0.0,
            "coverage_ratio": 0.0,
            "reason": "止损单存在，但无法解析其原始 size"
        }

    max_stop_size = max(parsed_sizes)
    coverage_ratio = max_stop_size / (expected_contracts + 1e-9)

    if coverage_ratio < 0.9:
        return {
            "status": "undercovered_stop",
            "expected_contracts": expected_contracts,
            "matched_stop_count": len(matched_stops),
            "max_stop_size": max_stop_size,
            "coverage_ratio": coverage_ratio,
            "reason": f"止损单覆盖不足，raw size={max_stop_size:.4f} < 持仓={expected_contracts:.4f}"
        }

    if coverage_ratio > 1.1:
        return {
            "status": "oversized_stop",
            "expected_contracts": expected_contracts,
            "matched_stop_count": len(matched_stops),
            "max_stop_size": max_stop_size,
            "coverage_ratio": coverage_ratio,
            "reason": f"止损单数量偏大，raw size={max_stop_size:.4f} > 持仓={expected_contracts:.4f}"
        }

    return {
        "status": "ok",
        "expected_contracts": expected_contracts,
        "matched_stop_count": len(matched_stops),
        "max_stop_size": max_stop_size,
        "coverage_ratio": coverage_ratio,
        "reason": ""
    }


def _build_inventory_guard(position_tier_state, position_state_meta):
    """
    识别“策略自有库存”与“异常外部库存”：
    一旦进入 external_inventory，主引擎停止继续下单，只保留右脑风控托管。
    """
    sym_key = _normalize_symbol_key(SYMBOL)
    with state.pos_lock:
        position = dict(state.active_pos_map).get(sym_key)

    if not position:
        return {
            "mode": "flat",
            "block_new_orders": False,
            "hard_reasons": [],
            "soft_reasons": [],
            "current_leverage": 0.0,
            "configured_leverages": sorted(set(float(x) for x in [LEVERAGE_1, LEVERAGE_2, LEVERAGE_3] if float(x) > 0)),
            "stop_coverage": {
                "status": "no_position",
                "expected_contracts": 0.0,
                "matched_stop_count": 0,
                "max_stop_size": 0.0,
                "coverage_ratio": 1.0,
                "reason": ""
            },
            "matching_open_record_count": 0,
            "state_source": position_state_meta["source"]
        }

    info = position.get("info", {}) or {}
    pos_side = str(position.get("side", "")).upper()
    local_side = "buy" if pos_side == "LONG" else "sell"
    configured_leverages = sorted(set(float(x) for x in [LEVERAGE_1, LEVERAGE_2, LEVERAGE_3] if float(x) > 0))

    try:
        current_leverage = float(position.get("leverage", info.get("lever", info.get("leverage", 0))) or 0.0)
    except (TypeError, ValueError):
        current_leverage = 0.0

    hard_reasons = []
    soft_reasons = []

    if configured_leverages and not any(abs(current_leverage - expected) <= 0.05 for expected in configured_leverages):
        hard_reasons.append(
            f"交易所杠杆 {current_leverage:.2f}x 不在策略分层 {configured_leverages} 内"
        )

    recent_open_trades = _load_recent_local_open_trades(limit=20)
    matching_open_records = [r for r in recent_open_trades if r["side"] == local_side]
    if not matching_open_records:
        soft_reasons.append("本地成交流水缺少与当前仓位同向的开仓记录")

    stop_coverage = _inspect_live_stop_coverage(position)
    if stop_coverage["status"] in {"missing_stop", "unparseable_stop", "undercovered_stop"}:
        hard_reasons.append(stop_coverage["reason"])
    elif stop_coverage["status"] == "oversized_stop":
        soft_reasons.append(stop_coverage["reason"])

    if not position_state_meta["sync_fresh"] and position_tier_state["side"] != 0:
        soft_reasons.append("仓位同步快照陈旧，当前库存来源仍处于 fallback 模式")

    block_new_orders = bool(hard_reasons) or (len(soft_reasons) >= 2)
    mode = "external_inventory" if block_new_orders else "strategy_managed"

    return {
        "mode": mode,
        "block_new_orders": block_new_orders,
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "current_leverage": current_leverage,
        "configured_leverages": configured_leverages,
        "stop_coverage": stop_coverage,
        "matching_open_record_count": len(matching_open_records),
        "state_source": position_state_meta["source"]
    }

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
    last_inventory_guard_mode = "flat"
    last_forced_exit_anchor_ts = 0
    
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
                    
                    aux_reference_snap = fetch_gateway_snapshot(AUX_SYMBOLS[0]) if len(AUX_SYMBOLS) >= 1 else None
                    direct_cross_snap = fetch_gateway_snapshot(AUX_SYMBOLS[1]) if len(AUX_SYMBOLS) >= 2 else None

                    features_df = compute_features(snapshot, aux_reference_snap, direct_cross_snap)
                    
                    # 提取喂给模型的那个精确窗口
                    inference_window = features_df[feature_cols].iloc[-ACTUAL_SEQ_LEN:]
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

            # --- 🎯 阶梯升档信号决策（Tier 对应目标总杠杆） ---
            position_tier_state, position_state_meta = _reconcile_position_tier_state(position_tier_state)
            inventory_guard = _build_inventory_guard(position_tier_state, position_state_meta)
            if inventory_guard["mode"] != last_inventory_guard_mode:
                reason_text = " | ".join(inventory_guard["hard_reasons"] + inventory_guard["soft_reasons"]) or "无"
                guard_msg = (
                    f"库存守卫模式切换为 {inventory_guard['mode']} | "
                    f"杠杆 {inventory_guard['current_leverage']:.2f}x | 原因: {reason_text}"
                )
                print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🧯 {guard_msg}")
                log_system_alert("🧯 库存守卫", SYMBOL, guard_msg)
                last_inventory_guard_mode = inventory_guard["mode"]
            forced_exit_lock = _get_forced_exit_lock(SYMBOL, latest_closed_ts)
            if forced_exit_lock.get("released"):
                released_anchor = forced_exit_lock.get("released_anchor_closed_ts", 0)
                print(
                    f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🔓 "
                    f"强制退出冷却解除，检测到新闭合K线 (prev_ts: {released_anchor} -> now: {latest_closed_ts})"
                )
                last_forced_exit_anchor_ts = 0
            elif forced_exit_lock.get("active"):
                anchor_closed_ts = int(forced_exit_lock.get("anchor_closed_ts", 0) or 0)
                if anchor_closed_ts != last_forced_exit_anchor_ts:
                    reason = forced_exit_lock.get("reason", "强制退出")
                    print(
                        f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🧊 "
                        f"强制退出冷却生效：{reason} | 当前闭合K线TS={latest_closed_ts} | 锁定基准TS={anchor_closed_ts}"
                    )
                    last_forced_exit_anchor_ts = anchor_closed_ts
            target_signal = 0
            target_tier = 0
            dynamic_leverage = 0.0

            # 评估是否击穿各层级动态阈值 (用各自阶梯专属的修正概率)
            if adj_up_top1 > th['up_top_1']:        target_signal = 1; target_tier = 3; dynamic_leverage = LEVERAGE_3
            elif adj_up_top5 > th['up_top_5']:      target_signal = 1; target_tier = 2; dynamic_leverage = LEVERAGE_2
            elif adj_up_top10 > th['up_top_10']:     target_signal = 1; target_tier = 1; dynamic_leverage = LEVERAGE_1
            elif adj_down_top1 > th['down_top_1']:  target_signal = -1; target_tier = 3; dynamic_leverage = LEVERAGE_3
            elif adj_down_top5 > th['down_top_5']:  target_signal = -1; target_tier = 2; dynamic_leverage = LEVERAGE_2
            elif adj_down_top10 > th['down_top_10']: target_signal = -1; target_tier = 1; dynamic_leverage = LEVERAGE_1

            # 阶梯过滤器核心逻辑
            actual_signal = 0
            if forced_exit_lock.get("active"):
                actual_signal = 0
            elif inventory_guard["block_new_orders"]:
                actual_signal = 0
            elif target_signal != 0:
                if position_tier_state["side"] == 0:
                    actual_signal = target_signal
                elif position_tier_state["side"] == target_signal:
                    if target_tier > position_tier_state["highest_tier"]:
                        actual_signal = target_signal
                else:
                    actual_signal = target_signal # 反向平仓交由 OMS 处理

            if needs_evaluation_log:
                action_str = "按兵不动"
                if forced_exit_lock.get("active"):
                    action_str = "🧊 强制退出冷却中，等待下一根1h K线闭合后再评估"
                elif inventory_guard["block_new_orders"]:
                    action_str = "🧯 异常库存隔离中，主引擎禁止继续下单"
                elif actual_signal == 1: 
                    action_str = f"🚀 触发做多 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"
                elif actual_signal == -1: 
                    action_str = f"🩸 触发做空 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"

                inventory_reason_text = " | ".join(inventory_guard["hard_reasons"] + inventory_guard["soft_reasons"]) or "无"
                forced_exit_text = "无"
                if forced_exit_lock.get("active"):
                    forced_exit_text = (
                        f"{forced_exit_lock.get('reason', '强制退出')} | "
                        f"锁定至下一根闭合K线 (> {forced_exit_lock.get('anchor_closed_ts', 0)})"
                    )
                eval_log = (
                    f"【触发原因】{trigger_reason}\n"
                    f"【原始概率】📈多:{cached_p_up:.4f} | ⚪平:{cached_p_flat:.4f} | 📉空:{cached_p_down:.4f}\n"
                    f"【融合概率(入场级Tier1)】📈多:{adj_up_top10:.4f} | 📉空:{adj_down_top10:.4f} (宏观分:{cached_macro_score:+.2f})\n"
                    f"【引擎决策】{action_str} | 状态机: {position_tier_state} | 来源: {position_state_meta['source']} | "
                    f"仓位同步新鲜: {'是' if position_state_meta['sync_fresh'] else '否'}\n"
                    f"【库存守卫】模式: {inventory_guard['mode']} | 当前杠杆: {inventory_guard['current_leverage']:.2f}x | "
                    f"止损覆盖: {inventory_guard['stop_coverage']['status']} | 原因: {inventory_reason_text}"
                    f"\n【强制退出冷却】{forced_exit_text}"
                )
                print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 🧠 单模型评估汇报:\n{eval_log}\n")
                log_system_alert("🧠 决策评估", SYMBOL, eval_log)


            if actual_signal != 0:
                print(f"\n[{snapshot['server_time']}] 🔥 突破阶梯红线！最新价: {current_price} | 发送 OMS 指令 (Tier-{target_tier})...")
                log_system_alert("🔥 下单触发", SYMBOL,
                    f"信号: {'做多' if actual_signal==1 else '做空'} | Tier: {target_tier} | 目标杠杆: {dynamic_leverage}x | 价格: {current_price:.4f}")
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
                },
                "position_state_meta": {
                    "source": position_state_meta["source"],
                    "sync_fresh": bool(position_state_meta["sync_fresh"]),
                    "sync_age_sec": None if np.isinf(position_state_meta["sync_age_sec"]) else round(float(position_state_meta["sync_age_sec"]), 3)
                },
                "inventory_guard": {
                    "mode": inventory_guard["mode"],
                    "block_new_orders": bool(inventory_guard["block_new_orders"]),
                    "current_leverage": float(inventory_guard["current_leverage"]),
                    "configured_leverages": [float(x) for x in inventory_guard["configured_leverages"]],
                    "matching_open_record_count": int(inventory_guard["matching_open_record_count"]),
                    "hard_reasons": inventory_guard["hard_reasons"],
                    "soft_reasons": inventory_guard["soft_reasons"],
                    "stop_coverage": {
                        "status": inventory_guard["stop_coverage"]["status"],
                        "expected_contracts": float(inventory_guard["stop_coverage"]["expected_contracts"]),
                        "matched_stop_count": int(inventory_guard["stop_coverage"]["matched_stop_count"]),
                        "max_stop_size": float(inventory_guard["stop_coverage"]["max_stop_size"]),
                        "coverage_ratio": float(inventory_guard["stop_coverage"]["coverage_ratio"]),
                        "reason": inventory_guard["stop_coverage"]["reason"]
                    }
                },
                "forced_exit_lock": {
                    "active": bool(forced_exit_lock.get("active", False)),
                    "anchor_closed_ts": int(forced_exit_lock.get("anchor_closed_ts", 0) or 0),
                    "reason": forced_exit_lock.get("reason", ""),
                    "trigger_ts": float(forced_exit_lock.get("trigger_ts", 0.0) or 0.0)
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
