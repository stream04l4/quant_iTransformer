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
    MACRO_LOGIT_DEAD_ZONE, MACRO_LOGIT_MID_SCORE, MACRO_LOGIT_TAIL_SCORE,
    MACRO_LOGIT_MID_DELTA, MACRO_LOGIT_TAIL_DELTA, MACRO_LOGIT_MAX_DELTA, MACRO_MAX_PROB_SHIFT,
    MACRO_LOGIT_MID_POWER, MACRO_LOGIT_TAIL_POWER, MACRO_LOGIT_EXTREME_POWER,
    INITIAL_CAPITAL, BASE_RISK_BUDGET, BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, FORECAST_HORIZON,
    EXPERIMENT_ID, TRAIN_DATA_DIR, TRAILING_ATR_PERIOD, TRAILING_ATR_MULTI,
    LIVE_SYMBOL_CONFIGS
)
from dotenv import load_dotenv
load_dotenv()
# ==========================================
# 1. 加载实盘配置 (基于 EXPERIMENT_ID 精准定位)
# ==========================================
GATEWAY_URL = "http://127.0.0.1:8000/api/snapshot"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GATEWAY_CONNECT_TIMEOUT = 1.0
GATEWAY_READ_TIMEOUT = 5.0
GATEWAY_DEGRADED_AFTER = 2
GATEWAY_ALERT_COOLDOWN = 60.0
GATEWAY_SESSION = requests.Session()
gateway_request_state = {}

def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _normalize_model_specs(cfg):
    raw_models = cfg.get("models")
    if not raw_models:
        raw_models = [{"experiment_id": cfg.get("experiment_id", EXPERIMENT_ID), "weight": cfg.get("weight", 1.0)}]

    normalized = []
    for i, raw_model in enumerate(raw_models):
        model_cfg = raw_model or {}
        exp_id = str(model_cfg.get("experiment_id", cfg.get("experiment_id", EXPERIMENT_ID)))
        weight = float(model_cfg.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"模型权重必须为正数: experiment_id={exp_id}, weight={weight}")
        normalized.append({
            "name": str(model_cfg.get("name", exp_id)),
            "experiment_id": exp_id,
            "weight": weight,
        })

    total_weight = sum(m["weight"] for m in normalized)
    for model_cfg in normalized:
        model_cfg["normalized_weight"] = model_cfg["weight"] / total_weight
    return normalized


def _normalize_live_configs():
    raw_configs = LIVE_SYMBOL_CONFIGS if isinstance(LIVE_SYMBOL_CONFIGS, dict) and LIVE_SYMBOL_CONFIGS else {
        SYMBOL: {"experiment_id": EXPERIMENT_ID, "aux_symbols": AUX_SYMBOLS}
    }
    normalized = {}
    for raw_symbol, cfg in raw_configs.items():
        sym_key = _normalize_symbol_key(raw_symbol)
        cfg = cfg or {}
        normalized[sym_key] = {
            "experiment_id": str(cfg.get("experiment_id", EXPERIMENT_ID)),
            "aux_symbols": [_normalize_symbol_key(s) for s in cfg.get("aux_symbols", [])],
            "models": _normalize_model_specs(cfg),
        }
    return normalized


LIVE_CONFIGS = _normalize_live_configs()
LIVE_MAIN_SYMBOLS = list(LIVE_CONFIGS.keys())
LIVE_MAIN_SYMBOL_SET = set(LIVE_MAIN_SYMBOLS)


def _resolve_aux_request_symbol(sym):
    sym_key = _normalize_symbol_key(sym)
    return f"{sym_key}_SPOT" if sym_key in LIVE_MAIN_SYMBOL_SET else sym_key


def _gateway_request_state(sym):
    sym_key = _normalize_symbol_key(sym)
    return gateway_request_state.setdefault(sym_key, {
        "consecutive_failures": 0,
        "last_alert_ts": 0.0,
        "last_error": "",
    })


def _mark_gateway_success(sym, latency_ms):
    state_info = _gateway_request_state(sym)
    previous_failures = int(state_info.get("consecutive_failures", 0) or 0)
    if previous_failures >= GATEWAY_DEGRADED_AFTER:
        msg = f"本地行情网关已恢复，连续失败清零 | latency={latency_ms:.0f}ms"
        print(f"✅ [{_normalize_symbol_key(sym)}] {msg}")
        log_system_alert("✅ 行情网关恢复", _normalize_symbol_key(sym), msg)
    state_info["consecutive_failures"] = 0
    state_info["last_error"] = ""


def _mark_gateway_failure(sym, reason):
    sym_key = _normalize_symbol_key(sym)
    state_info = _gateway_request_state(sym_key)
    state_info["consecutive_failures"] = int(state_info.get("consecutive_failures", 0) or 0) + 1
    state_info["last_error"] = reason
    fail_count = state_info["consecutive_failures"]
    now = time.time()

    if fail_count == 1:
        print(f"⚠️ [{sym_key}] 行情网关快照暂不可用: {reason}")
        return

    if fail_count >= GATEWAY_DEGRADED_AFTER and now - float(state_info.get("last_alert_ts", 0.0) or 0.0) >= GATEWAY_ALERT_COOLDOWN:
        msg = (
            f"本地行情网关连续 {fail_count} 次快照失败，主引擎将跳过本轮交易评估 | "
            f"原因: {reason}"
        )
        print(f"❌ [{sym_key}] {msg}")
        log_system_alert("❌ 行情网关超时", sym_key, msg)
        state_info["last_alert_ts"] = now

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
    start_ts = time.time()
    try:
        resp = GATEWAY_SESSION.get(
            f"{GATEWAY_URL}/{request_sym}",
            timeout=(GATEWAY_CONNECT_TIMEOUT, GATEWAY_READ_TIMEOUT),
        )
        latency_ms = (time.time() - start_ts) * 1000
        if resp.status_code != 200:
            _mark_gateway_failure(request_sym, f"HTTP {resp.status_code}: {resp.text[:180]}")
            return None
        payload = resp.json()
        payload["_gateway_latency_ms"] = latency_ms
        _mark_gateway_success(request_sym, latency_ms)
        return payload
    except requests.exceptions.Timeout:
        _mark_gateway_failure(
            request_sym,
            f"读取超时 > {GATEWAY_READ_TIMEOUT:.1f}s；网关进程可能仍存活但 API 响应阻塞",
        )
        return None
    except requests.exceptions.RequestException as e:
        _mark_gateway_failure(request_sym, f"{type(e).__name__}: {e}")
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

def dedupe_hourly_history(df, value_col):
    """衍生品历史可能来自 API/缓存合并；按小时去重，避免 merge 扩行主时间轴。"""
    return (
        df.sort_values('hour_timestamp')
          .dropna(subset=['hour_timestamp', value_col])
          .drop_duplicates(subset=['hour_timestamp'], keep='last')
    )

def compute_features(main_snap, aux_reference_snap=None, direct_cross_snap=None, seq_len=None):
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
    
    if seq_len is None:
        raise ValueError("缺少模型 seq_len，无法构建实盘推理窗口。")
    required_history = seq_len + 336
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
        df_prem = dedupe_hourly_history(df_prem, 'premium_close')
        df = df.merge(df_prem[['hour_timestamp', 'premium_close']], on='hour_timestamp', how='left')
    else: df['premium_close'] = np.nan

    if main_snap.get('funding_history'):
        df_fund = pd.DataFrame(main_snap['funding_history'], columns=['ts', 'funding_rate'])
        df_fund['hour_timestamp'] = pd.to_datetime(df_fund['ts'], unit='ms').dt.floor('h')
        df_fund = dedupe_hourly_history(df_fund, 'funding_rate')
        df = df.merge(df_fund[['hour_timestamp', 'funding_rate']], on='hour_timestamp', how='left')
    else: df['funding_rate'] = np.nan

    if main_snap.get('oi_history'):
        df_oi = pd.DataFrame(main_snap['oi_history'], columns=['ts', 'sum_open_interest'])
        df_oi['hour_timestamp'] = pd.to_datetime(df_oi['ts'], unit='ms')
        df_oi = dedupe_hourly_history(df_oi, 'sum_open_interest')
        df = df.merge(df_oi[['hour_timestamp', 'sum_open_interest']], on='hour_timestamp', how='left')
    else: df['sum_open_interest'] = np.nan

    # 训练端只允许资金费率按物理周期有限延续；其它缺口保留 NaN，交给最终熔断处理。
    df['funding_rate'] = df['funding_rate'].ffill(limit=8)
    df['cross_asset_close'] = df['cross_pair_close'].combine_first(df['aux_reference_close'] / (df['close'] + 1e-9))
    df = df.reset_index(drop=True)
    assert_hourly_continuity(df)

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

    inference_window = df[final_cols].iloc[-seq_len:]
    
    if inference_window.isnull().any().any():
        missing_cols = inference_window.columns[inference_window.isnull().any()].tolist()
        raise ValueError(f"🚨 引擎熔断！网关传入数据在以下特征存在断层: {missing_cols}。拒绝输出预测信号！")
        
    return inference_window

# ==========================================
# ⚡ 贝叶斯 Logit 融合算子
# ==========================================
def apply_logit_fusion(base_prob, macro_score, alpha):
    """
    将模型概率转换为 Log-Odds 后注入宏观偏置。

    宏观分数采用非线性强度曲线：
    - 低于 config 中的 dead zone 基本无影响。
    - 中段对 Tier-1 仅产生小幅概率扰动。
    - 黑天鹅/大波动区间凸性放大，Tier-1 可直接推动信号越线。
    - alpha 仍表示档位权重：Tier-1(top10) 最大，Tier-3(top1) 最小。
    - 最终概率位移被限制在 0%~5% 区间：|macro_score|=1 时 Tier-1 最多移动 5 个百分点。
    """
    epsilon = 1e-7
    p_safe = np.clip(float(base_prob or 0.0), epsilon, 1.0 - epsilon)
    logit = np.log(p_safe / (1.0 - p_safe))

    score = float(np.clip(float(macro_score or 0.0), -1.0, 1.0))
    abs_score = abs(score)
    sign = 1.0 if score >= 0 else -1.0
    tier_scale = float(np.clip(float(alpha or 0.0) / max(abs(MACRO_ALPHA_TOP_10), 1e-9), 0.0, 1.0))

    dead_zone = min(max(MACRO_LOGIT_DEAD_ZONE, 0.0), 1.0)
    mid_score = min(max(MACRO_LOGIT_MID_SCORE, dead_zone + 1e-9), 1.0)
    tail_score = min(max(MACRO_LOGIT_TAIL_SCORE, mid_score + 1e-9), 1.0)

    if abs_score <= dead_zone:
        macro_delta = 0.0
    elif abs_score <= mid_score:
        x = (abs_score - dead_zone) / (mid_score - dead_zone)
        macro_delta = MACRO_LOGIT_MID_DELTA * (x ** MACRO_LOGIT_MID_POWER)
    elif abs_score <= tail_score:
        x = (abs_score - mid_score) / (tail_score - mid_score)
        macro_delta = MACRO_LOGIT_MID_DELTA + (
            MACRO_LOGIT_TAIL_DELTA - MACRO_LOGIT_MID_DELTA
        ) * (x ** MACRO_LOGIT_TAIL_POWER)
    else:
        x = (abs_score - tail_score) / (1.0 - tail_score + 1e-9)
        macro_delta = MACRO_LOGIT_TAIL_DELTA + (
            MACRO_LOGIT_MAX_DELTA - MACRO_LOGIT_TAIL_DELTA
        ) * (x ** MACRO_LOGIT_EXTREME_POWER)

    adjusted_logit = logit + (sign * macro_delta * tier_scale)
    adjusted_prob = float(1.0 / (1.0 + np.exp(-adjusted_logit)))
    max_shift = max(0.0, float(MACRO_MAX_PROB_SHIFT)) * abs_score * tier_scale
    return float(np.clip(adjusted_prob, p_safe - max_shift, p_safe + max_shift))

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


def _read_persisted_tier_state(symbol):
    """读取上一次落盘的状态机，作为交易所快照未就绪时的兜底。"""
    status_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if not os.path.exists(status_path):
        return {"side": 0, "highest_tier": 0}

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        sym_key = _normalize_symbol_key(symbol)
        symbol_state = (saved.get("symbols", {}) or {}).get(sym_key, {})
        if symbol_state:
            return _sanitize_tier_state(symbol_state.get("position_tier_state", {}))
        if _normalize_symbol_key(saved.get("symbol")) == sym_key:
            return _sanitize_tier_state(saved.get("position_tier_state", {}))
        return {"side": 0, "highest_tier": 0}
    except Exception as e:
        print(f"⚠️  [状态恢复] 读取 live_status.json 失败: {e}")
        return {"side": 0, "highest_tier": 0}


def _get_exchange_position_state(symbol):
    """从供货线程同步过来的真实仓位快照推导当前状态。"""
    with state.pos_lock:
        pos_map = dict(state.active_pos_map)

    sym_key = _normalize_symbol_key(symbol)
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


def _reconcile_position_tier_state(symbol, cached_state):
    """
    每个 tick 都重新对账：
    1. 真实仓位决定 side。
    2. highest_tier 取 exchange / 内存 / 落盘三者中同向的最高值，避免初始化错一次后持续漂移。
    3. 若供货线程快照陈旧，则退化到缓存/磁盘兜底，但不会永久锁死。
    """
    cached_state = _sanitize_tier_state(cached_state)
    persisted_state = _read_persisted_tier_state(symbol)
    exchange_state = _get_exchange_position_state(symbol)

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


def _reconstruct_tier_state(symbol):
    """启动时先做一次对账恢复，后续每个 tick 会继续自动校正。"""
    with state.pos_lock:
        pos_keys = list(state.active_pos_map.keys())
    sym_key = _normalize_symbol_key(symbol)
    print(f"🔍 [状态恢复] 正在校验符号: {sym_key}，当前已同步仓位列表: {pos_keys}")

    reconstructed, meta = _reconcile_position_tier_state(sym_key, {"side": 0, "highest_tier": 0})
    if reconstructed["side"] != 0:
        print(
            f"♻️  [重启恢复] 状态机已恢复: side={reconstructed['side']}, "
            f"Tier={reconstructed['highest_tier']} | source={meta['source']}"
        )
    else:
        print(f"🆕 [重启恢复] 初始化为空仓 | source={meta['source']}")
    return reconstructed


def _load_recent_local_open_trades(symbol, limit=20):
    """读取本地开仓流水，用于判断当前库存是否仍在策略可解释范围内。"""
    db_path = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{_normalize_symbol_key(symbol)}.db")
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


def _inspect_live_stop_coverage(position, symbol=None):
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
    sym_key = _normalize_symbol_key(position.get("symbol", symbol or SYMBOL))
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

    if len(parsed_sizes) > 1:
        return {
            "status": "duplicate_stop",
            "expected_contracts": expected_contracts,
            "matched_stop_count": len(matched_stops),
            "max_stop_size": max_stop_size,
            "coverage_ratio": coverage_ratio,
            "reason": f"检测到 {len(parsed_sizes)} 张可解析止损单，右脑将收敛为单一统一防线"
        }

    return {
        "status": "ok",
        "expected_contracts": expected_contracts,
        "matched_stop_count": len(matched_stops),
        "max_stop_size": max_stop_size,
        "coverage_ratio": coverage_ratio,
        "reason": ""
    }


def _build_inventory_guard(symbol, position_tier_state, position_state_meta):
    """
    识别“策略自有库存”与“异常外部库存”：
    一旦进入 external_inventory，主引擎停止继续下单，只保留右脑风控托管。
    """
    sym_key = _normalize_symbol_key(symbol)
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

    recent_open_trades = _load_recent_local_open_trades(sym_key, limit=20)
    matching_open_records = [r for r in recent_open_trades if r["side"] == local_side]
    if not matching_open_records:
        soft_reasons.append("本地成交流水缺少与当前仓位同向的开仓记录")

    stop_coverage = _inspect_live_stop_coverage(position, sym_key)
    if stop_coverage["status"] in {"missing_stop", "unparseable_stop", "undercovered_stop"}:
        hard_reasons.append(stop_coverage["reason"])
    elif stop_coverage["status"] in {"oversized_stop", "duplicate_stop"}:
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

def _load_model_component(symbol, model_cfg):
    exp_id = str(model_cfg.get("experiment_id", EXPERIMENT_ID))
    model_path = MODEL_DIR / f"itransformer_{symbol}_best_{exp_id}.pth"
    json_path = MODEL_DIR / f"dynamic_thresholds_{exp_id}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"❌ [{symbol}] 找不到模型权重: {model_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"❌ [{symbol}] 找不到动态阈值文件: {json_path}")

    print(f"📥 [{symbol}] 加载模型权重: {model_path.name} | weight={model_cfg['normalized_weight']:.3f}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    ckpt_seq_len = checkpoint.get('seq_len')
    model = InvertedTransformer(
        num_variates=len(feature_cols),
        seq_len=ckpt_seq_len,
        d_model=checkpoint.get('d_model'),
        n_heads=checkpoint.get('n_heads'),
        d_ff=checkpoint.get('d_ff'),
        e_layers=checkpoint.get('e_layers'),
        dropout=checkpoint.get('dropout')
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(json_path, 'r') as f:
        config_data = json.load(f)
        thresholds = config_data.get("thresholds", config_data)

    return {
        "name": model_cfg.get("name", exp_id),
        "exp_id": exp_id,
        "weight": float(model_cfg["normalized_weight"]),
        "raw_weight": float(model_cfg["weight"]),
        "model_path": model_path,
        "json_path": json_path,
        "model": model,
        "feature_cols": feature_cols,
        "seq_len": ckpt_seq_len,
        "thresholds": thresholds,
    }


def _weighted_thresholds(model_components):
    threshold_keys = ["up_top_1", "up_top_5", "up_top_10", "down_top_1", "down_top_5", "down_top_10"]
    return {
        key: sum(float(component["thresholds"][key]) * component["weight"] for component in model_components)
        for key in threshold_keys
    }


def _shadow_audit_db_path():
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_DATA_DIR / "live_model_shadow_audit.db"


def _ensure_shadow_audit_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solo_model_inference (
            symbol TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            decision_ts INTEGER NOT NULL,
            server_time TEXT,
            forecast_horizon INTEGER,
            live_weight REAL,
            entry_price REAL,
            macro_score REAL,
            p_down REAL,
            p_flat REAL,
            p_up REAL,
            adj_p_down REAL,
            adj_p_up REAL,
            raw_signal INTEGER,
            raw_tier INTEGER,
            raw_leverage REAL,
            adj_signal INTEGER,
            adj_tier INTEGER,
            adj_leverage REAL,
            settled INTEGER DEFAULT 0,
            settle_ts INTEGER,
            settle_price REAL,
            raw_pnl REAL,
            adj_pnl REAL,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (symbol, experiment_id, decision_ts)
        )
        """
    )


def _solo_decision_from_probs(p_down, p_up, thresholds):
    if p_up > thresholds.get("up_top_1", 1.0):
        return 1, 3, LEVERAGE_3
    if p_up > thresholds.get("up_top_5", 1.0):
        return 1, 2, LEVERAGE_2
    if p_up > thresholds.get("up_top_10", 1.0):
        return 1, 1, LEVERAGE_1
    if p_down > thresholds.get("down_top_1", 1.0):
        return -1, 3, LEVERAGE_3
    if p_down > thresholds.get("down_top_5", 1.0):
        return -1, 2, LEVERAGE_2
    if p_down > thresholds.get("down_top_10", 1.0):
        return -1, 1, LEVERAGE_1
    return 0, 0, 0.0


def _record_solo_model_shadow_decisions(symbol, rt, model_details, latest_closed_ts, current_price, macro_score, server_time):
    """记录每次实盘推理下，各子模型单独服役时会做出的影子决策。"""
    if not model_details:
        return

    component_by_exp = {component["exp_id"]: component for component in rt["models"]}
    now_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for detail in model_details:
        exp_id = detail["exp_id"]
        component = component_by_exp.get(exp_id)
        if not component:
            continue
        thresholds = component["thresholds"]

        p_down = float(detail["prob_down"])
        p_flat = float(detail["prob_flat"])
        p_up = float(detail["prob_up"])
        raw_signal, raw_tier, raw_leverage = _solo_decision_from_probs(p_down, p_up, thresholds)

        adj_p_up = apply_logit_fusion(p_up, macro_score, MACRO_ALPHA_TOP_10)
        adj_p_down = apply_logit_fusion(p_down, -macro_score, MACRO_ALPHA_TOP_10)
        adj_signal, adj_tier, adj_leverage = _solo_decision_from_probs(adj_p_down, adj_p_up, thresholds)

        rows.append((
            symbol,
            exp_id,
            int(latest_closed_ts),
            str(server_time or ""),
            int(FORECAST_HORIZON),
            float(detail["weight"]),
            float(current_price),
            float(macro_score),
            p_down,
            p_flat,
            p_up,
            float(adj_p_down),
            float(adj_p_up),
            int(raw_signal),
            int(raw_tier),
            float(raw_leverage),
            int(adj_signal),
            int(adj_tier),
            float(adj_leverage),
            now_text,
            now_text,
        ))

    if not rows:
        return

    try:
        with sqlite3.connect(_shadow_audit_db_path(), timeout=5.0) as conn:
            _ensure_shadow_audit_schema(conn)
            conn.executemany(
                """
                INSERT INTO solo_model_inference (
                    symbol, experiment_id, decision_ts, server_time, forecast_horizon,
                    live_weight, entry_price, macro_score,
                    p_down, p_flat, p_up, adj_p_down, adj_p_up,
                    raw_signal, raw_tier, raw_leverage,
                    adj_signal, adj_tier, adj_leverage,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, experiment_id, decision_ts) DO UPDATE SET
                    server_time=excluded.server_time,
                    forecast_horizon=excluded.forecast_horizon,
                    live_weight=excluded.live_weight,
                    entry_price=excluded.entry_price,
                    macro_score=excluded.macro_score,
                    p_down=excluded.p_down,
                    p_flat=excluded.p_flat,
                    p_up=excluded.p_up,
                    adj_p_down=excluded.adj_p_down,
                    adj_p_up=excluded.adj_p_up,
                    raw_signal=excluded.raw_signal,
                    raw_tier=excluded.raw_tier,
                    raw_leverage=excluded.raw_leverage,
                    adj_signal=excluded.adj_signal,
                    adj_tier=excluded.adj_tier,
                    adj_leverage=excluded.adj_leverage,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
            conn.commit()
    except Exception as e:
        print(f"   ⚠️ [{symbol}] 子模型影子决策落库失败: {e}")


def _settle_solo_model_shadow_decisions(symbol, ohlcv_1h):
    """用真实后续 1h close 给影子推理打标。不是完整 OMS 回放，而是 live horizon 方向盈亏审计。"""
    if not ohlcv_1h:
        return

    close_by_ts = {}
    for candle in ohlcv_1h:
        try:
            close_by_ts[int(candle[0])] = float(candle[4])
        except (TypeError, ValueError, IndexError):
            continue
    if not close_by_ts:
        return

    try:
        with sqlite3.connect(_shadow_audit_db_path(), timeout=5.0) as conn:
            _ensure_shadow_audit_schema(conn)
            rows = conn.execute(
                """
                SELECT symbol, experiment_id, decision_ts, forecast_horizon, entry_price,
                       raw_signal, raw_leverage, adj_signal, adj_leverage
                FROM solo_model_inference
                WHERE symbol = ? AND settled = 0
                """,
                (symbol,),
            ).fetchall()

            updates = []
            now_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            for row in rows:
                _, exp_id, decision_ts, horizon, entry_price, raw_signal, raw_lev, adj_signal, adj_lev = row
                settle_ts = int(decision_ts) + int(horizon or FORECAST_HORIZON) * 3600000
                settle_price = close_by_ts.get(settle_ts)
                if settle_price is None or not entry_price:
                    continue

                raw_pnl = float(raw_signal or 0) * ((settle_price / float(entry_price)) - 1.0) * float(raw_lev or 0.0)
                adj_pnl = float(adj_signal or 0) * ((settle_price / float(entry_price)) - 1.0) * float(adj_lev or 0.0)
                updates.append((int(settle_ts), float(settle_price), raw_pnl, adj_pnl, now_text, symbol, exp_id, int(decision_ts)))

            if updates:
                conn.executemany(
                    """
                    UPDATE solo_model_inference
                    SET settled = 1,
                        settle_ts = ?,
                        settle_price = ?,
                        raw_pnl = ?,
                        adj_pnl = ?,
                        updated_at = ?
                    WHERE symbol = ? AND experiment_id = ? AND decision_ts = ?
                    """,
                    updates,
                )
                conn.commit()
    except Exception as e:
        print(f"   ⚠️ [{symbol}] 子模型影子审计结算失败: {e}")


def _run_weighted_model_ensemble(rt, features_df):
    ensemble_probs = np.zeros(3, dtype=np.float64)
    model_details = []

    with torch.no_grad():
        for component in rt["models"]:
            missing_cols = [c for c in component["feature_cols"] if c not in features_df.columns]
            if missing_cols:
                raise ValueError(
                    f"[{rt['symbol']}] 模型 {component['exp_id']} 所需特征缺失: {missing_cols}"
                )

            inference_window = features_df[component["feature_cols"]].iloc[-component["seq_len"]:]
            input_raw = inference_window.values.astype('float32')
            x_tensor = torch.tensor(input_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            logits = component["model"](x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            ensemble_probs += probs * component["weight"]
            model_details.append({
                "name": component["name"],
                "exp_id": component["exp_id"],
                "weight": component["weight"],
                "prob_down": float(probs[0]),
                "prob_flat": float(probs[1]),
                "prob_up": float(probs[2]),
            })

    return ensemble_probs, model_details


def _load_symbol_runtime(symbol, cfg):
    model_components = [_load_model_component(symbol, model_cfg) for model_cfg in cfg.get("models", [])]
    if not model_components:
        raise ValueError(f"❌ [{symbol}] 未配置任何实盘模型。")
    max_seq_len = max(component["seq_len"] for component in model_components)
    ensemble_thresholds = _weighted_thresholds(model_components)
    primary_component = model_components[0]

    return {
        "symbol": symbol,
        "exp_id": primary_component["exp_id"],
        "model_path": primary_component["model_path"],
        "json_path": primary_component["json_path"],
        "models": model_components,
        "model_count": len(model_components),
        "seq_len": max_seq_len,
        "thresholds": ensemble_thresholds,
        "aux_symbols": cfg.get("aux_symbols", []),
        "last_inferred_hour": 0,
        "cached_p_down": 0.0,
        "cached_p_flat": 0.0,
        "cached_p_up": 0.0,
        "cached_model_details": [],
        "cached_macro_score": 0.0,
        "last_macro_ts": "0",
        "last_signal_evaluated_ts": 0.0,
        "last_trade_hour": 0,
        "latest_closed_ts": 0,
        "last_inventory_guard_mode": "flat",
        "last_forced_exit_anchor_ts": 0,
        "position_tier_state": {"side": 0, "highest_tier": 0},
    }


def _compact_stop_coverage(stop_coverage):
    return {
        "status": stop_coverage["status"],
        "expected_contracts": float(stop_coverage["expected_contracts"]),
        "matched_stop_count": int(stop_coverage["matched_stop_count"]),
        "max_stop_size": float(stop_coverage["max_stop_size"]),
        "coverage_ratio": float(stop_coverage["coverage_ratio"]),
        "reason": stop_coverage["reason"]
    }


def run_live():
    print(f"🚀 极速实盘预测引擎启动 (多模型模式: {', '.join(LIVE_MAIN_SYMBOLS)})...")
    threading.Thread(target=bailout_worker, daemon=True).start()

    runtimes = {}
    for symbol, cfg in LIVE_CONFIGS.items():
        runtimes[symbol] = _load_symbol_runtime(symbol, cfg)
    print("✅ 多币种模型与阈值加载成功。")

    print("⏳ 正在同步交易所账户状态 (资金/仓位/委托)...")
    wait_start = time.time()
    while state.portfolio_memory.get('last_sync_ts', 0) == 0:
        if time.time() - wait_start > 15:
            print("⚠️ [启动预警] 供货线程同步超时 (15s)，将尝试从本地缓存恢复状态。")
            break
        time.sleep(1)

    for symbol, rt in runtimes.items():
        rt["position_tier_state"] = _reconstruct_tier_state(symbol)

    while True:
        cycle_start = time.time()
        symbol_payloads = {}

        state.dynamic_settings["bailout_atr_multi"] = BAILOUT_ATR_MULTI
        state.dynamic_settings["bailout_vol_multi"] = BAILOUT_VOL_MULTI
        state.dynamic_settings["trailing_atr_period"] = TRAILING_ATR_PERIOD
        state.dynamic_settings["trailing_atr_multi"] = TRAILING_ATR_MULTI

        dynamic_margin_usd = INITIAL_CAPITAL * BASE_RISK_BUDGET
        dynamic_sl = HARD_SL
        dynamic_tp = HARD_TP

        for symbol, rt in runtimes.items():
            try:
                snapshot = fetch_gateway_snapshot(symbol)
                if not snapshot:
                    continue

                current_time = time.time()
                current_price = snapshot['current_price']
                try:
                    state.update_price(symbol, current_price)
                except Exception:
                    pass

                needs_evaluation_log = False
                trigger_reason = ""

                if len(snapshot['ohlcv_1h']) >= 2:
                    latest_closed_ts = snapshot['ohlcv_1h'][-2][0]
                    rt["latest_closed_ts"] = latest_closed_ts

                    if latest_closed_ts != rt["last_inferred_hour"]:
                        current_minute = datetime.utcnow().minute
                        if current_minute < 2:
                            if int(current_time) % 30 == 0:
                                print(f"⏳ [{symbol}] 检测到新K线，等待整点第2分钟同步宏观分 (当前: {current_minute}分)...")
                        else:
                            print(f"\n🔄 [{symbol}] 进入整点第{current_minute}分钟，开始模型推理 (TS: {latest_closed_ts})...")
                            aux_symbols = rt["aux_symbols"]
                            aux_reference_snap = fetch_gateway_snapshot(_resolve_aux_request_symbol(aux_symbols[0])) if len(aux_symbols) >= 1 else None
                            direct_cross_snap = fetch_gateway_snapshot(_resolve_aux_request_symbol(aux_symbols[1])) if len(aux_symbols) >= 2 else None

                            features_df = compute_features(
                                snapshot,
                                aux_reference_snap,
                                direct_cross_snap,
                                seq_len=rt["seq_len"]
                            )

                            try:
                                os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
                                audit_csv_path = os.path.join(TRAIN_DATA_DIR, f"live_inference_audit_{symbol}.csv")
                                features_df.to_csv(audit_csv_path, index=False)
                                print(f"   💾 [{symbol}] 特征审计矩阵 {features_df.shape} 已输出至: {audit_csv_path}")
                            except Exception as e:
                                print(f"   ⚠️ [{symbol}] 特征审计 CSV 输出失败: {e}")

                            probs, model_details = _run_weighted_model_ensemble(rt, features_df)

                            rt["cached_p_down"], rt["cached_p_flat"], rt["cached_p_up"] = probs[0], probs[1], probs[2]
                            rt["cached_model_details"] = model_details
                            rt["cached_macro_score"], rt["last_macro_ts"] = get_macro_sentiment_info()
                            _record_solo_model_shadow_decisions(
                                symbol,
                                rt,
                                model_details,
                                latest_closed_ts,
                                current_price,
                                rt["cached_macro_score"],
                                snapshot.get("server_time"),
                            )
                            _settle_solo_model_shadow_decisions(symbol, snapshot.get('ohlcv_1h', []))
                            detail_text = " | ".join(
                                f"{m['exp_id']}@{m['weight']:.2f}: 多{m['prob_up']:.3f}/平{m['prob_flat']:.3f}/空{m['prob_down']:.3f}"
                                for m in model_details
                            )
                            print(f"🧠 [{symbol}] Ensemble({rt['model_count']}): 多{probs[2]:.4f} | 平{probs[1]:.4f} | 空{probs[0]:.4f}")
                            if detail_text:
                                print(f"   └─ 子模型: {detail_text}")
                            print(
                                f"📡 [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                                f"[{symbol}] 宏观情绪得分: {rt['cached_macro_score']:+.2f} (TS: {rt['last_macro_ts']})"
                            )

                            rt["last_inferred_hour"] = latest_closed_ts
                            needs_evaluation_log = True
                            trigger_reason = "新K线闭合推理并联动宏观评分"

                if not needs_evaluation_log and rt["cached_p_up"] + rt["cached_p_down"] > 0:
                    new_macro_score, new_macro_ts = get_macro_sentiment_info()
                    if new_macro_ts != rt["last_macro_ts"]:
                        rt["cached_macro_score"] = new_macro_score
                        rt["last_macro_ts"] = new_macro_ts
                        needs_evaluation_log = True
                        trigger_reason = f"宏观哨兵突击，监测到情报异动 (TS: {new_macro_ts})"
                        print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] [{symbol}] 宏观突袭重估: {new_macro_score:+.2f}")

                _settle_solo_model_shadow_decisions(symbol, snapshot.get('ohlcv_1h', []))

                cached_p_up = rt["cached_p_up"]
                cached_p_down = rt["cached_p_down"]
                cached_p_flat = rt["cached_p_flat"]
                cached_macro_score = rt["cached_macro_score"]
                th = rt["thresholds"]

                adj_up_top1 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_1)
                adj_up_top5 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_5)
                adj_up_top10 = apply_logit_fusion(cached_p_up, cached_macro_score, MACRO_ALPHA_TOP_10)
                adj_down_top1 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_1)
                adj_down_top5 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_5)
                adj_down_top10 = apply_logit_fusion(cached_p_down, -cached_macro_score, MACRO_ALPHA_TOP_10)

                rt["position_tier_state"], position_state_meta = _reconcile_position_tier_state(symbol, rt["position_tier_state"])
                position_tier_state = rt["position_tier_state"]
                inventory_guard = _build_inventory_guard(symbol, position_tier_state, position_state_meta)
                if inventory_guard["mode"] != rt["last_inventory_guard_mode"]:
                    reason_text = " | ".join(inventory_guard["hard_reasons"] + inventory_guard["soft_reasons"]) or "无"
                    guard_msg = (
                        f"库存守卫模式切换为 {inventory_guard['mode']} | "
                        f"杠杆 {inventory_guard['current_leverage']:.2f}x | 原因: {reason_text}"
                    )
                    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] [{symbol}] 🧯 {guard_msg}")
                    log_system_alert("🧯 库存守卫", symbol, guard_msg)
                    rt["last_inventory_guard_mode"] = inventory_guard["mode"]

                forced_exit_lock = _get_forced_exit_lock(symbol, rt["latest_closed_ts"])
                if forced_exit_lock.get("released"):
                    released_anchor = forced_exit_lock.get("released_anchor_closed_ts", 0)
                    print(
                        f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] [{symbol}] "
                        f"强制退出冷却解除 (prev_ts: {released_anchor} -> now: {rt['latest_closed_ts']})"
                    )
                    rt["last_forced_exit_anchor_ts"] = 0
                elif forced_exit_lock.get("active"):
                    anchor_closed_ts = int(forced_exit_lock.get("anchor_closed_ts", 0) or 0)
                    if anchor_closed_ts != rt["last_forced_exit_anchor_ts"]:
                        reason = forced_exit_lock.get("reason", "强制退出")
                        print(
                            f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] [{symbol}] "
                            f"强制退出冷却生效：{reason} | 当前闭合K线TS={rt['latest_closed_ts']} | 锁定TS={anchor_closed_ts}"
                        )
                        rt["last_forced_exit_anchor_ts"] = anchor_closed_ts

                target_signal = 0
                target_tier = 0
                dynamic_leverage = 0.0
                reverse_signal_ignored = False
                if adj_up_top1 > th['up_top_1']:
                    target_signal = 1; target_tier = 3; dynamic_leverage = LEVERAGE_3
                elif adj_up_top5 > th['up_top_5']:
                    target_signal = 1; target_tier = 2; dynamic_leverage = LEVERAGE_2
                elif adj_up_top10 > th['up_top_10']:
                    target_signal = 1; target_tier = 1; dynamic_leverage = LEVERAGE_1
                elif adj_down_top1 > th['down_top_1']:
                    target_signal = -1; target_tier = 3; dynamic_leverage = LEVERAGE_3
                elif adj_down_top5 > th['down_top_5']:
                    target_signal = -1; target_tier = 2; dynamic_leverage = LEVERAGE_2
                elif adj_down_top10 > th['down_top_10']:
                    target_signal = -1; target_tier = 1; dynamic_leverage = LEVERAGE_1

                if needs_evaluation_log:
                    rt["last_signal_evaluated_ts"] = current_time

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
                        reverse_signal_ignored = True

                if needs_evaluation_log:
                    action_str = "按兵不动"
                    if forced_exit_lock.get("active"):
                        action_str = "强制退出冷却中，等待下一根1h K线闭合后再评估"
                    elif inventory_guard["block_new_orders"]:
                        action_str = "异常库存隔离中，主引擎禁止继续下单"
                    elif reverse_signal_ignored:
                        action_str = "反向入场信号忽略，交由右脑按原始概率 Top-1 规则评估退出"
                    elif actual_signal == 1:
                        action_str = f"触发做多 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"
                    elif actual_signal == -1:
                        action_str = f"触发做空 (Tier-{target_tier}, 杠杆: {dynamic_leverage}x)"

                    inventory_reason_text = " | ".join(inventory_guard["hard_reasons"] + inventory_guard["soft_reasons"]) or "无"
                    forced_exit_text = "无"
                    if forced_exit_lock.get("active"):
                        forced_exit_text = (
                            f"{forced_exit_lock.get('reason', '强制退出')} | "
                            f"锁定至下一根闭合K线 (> {forced_exit_lock.get('anchor_closed_ts', 0)})"
                        )
                    model_stack_text = ", ".join(
                        f"{m['exp_id']}@{m['weight']:.2f}" for m in rt["cached_model_details"]
                    ) or "尚无推理明细"
                    eval_log = (
                        f"【触发原因】{trigger_reason}\n"
                        f"【模型架构】加权Ensemble x{rt['model_count']} | {model_stack_text}\n"
                        f"【原始概率】多:{cached_p_up:.4f} | 平:{cached_p_flat:.4f} | 空:{cached_p_down:.4f}\n"
                        f"【融合概率(入场级Tier1)】多:{adj_up_top10:.4f} | 空:{adj_down_top10:.4f} (宏观分:{cached_macro_score:+.2f})\n"
                        f"【引擎决策】{action_str} | 状态机: {position_tier_state} | 来源: {position_state_meta['source']} | "
                        f"仓位同步新鲜: {'是' if position_state_meta['sync_fresh'] else '否'}\n"
                        f"【库存守卫】模式: {inventory_guard['mode']} | 当前杠杆: {inventory_guard['current_leverage']:.2f}x | "
                        f"止损覆盖: {inventory_guard['stop_coverage']['status']} | 原因: {inventory_reason_text}"
                        f"\n【强制退出冷却】{forced_exit_text}"
                    )
                    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] [{symbol}] 多模型评估汇报:\n{eval_log}\n")
                    log_system_alert("🧠 决策评估", symbol, eval_log)

                if actual_signal != 0:
                    print(f"\n[{snapshot['server_time']}] [{symbol}] 突破阶梯红线，最新价: {current_price} | 发送 OMS 指令 (Tier-{target_tier})...")
                    log_system_alert(
                        "🔥 下单触发",
                        symbol,
                        f"信号: {'做多' if actual_signal==1 else '做空'} | Tier: {target_tier} | 目标杠杆: {dynamic_leverage}x | 价格: {current_price:.4f}"
                    )
                    success, actual_margin = execute_gate_trade(
                        symbol=symbol,
                        signal=actual_signal,
                        leverage=dynamic_leverage,
                        current_price=current_price,
                        tp_pct=dynamic_tp,
                        sl_pct=dynamic_sl,
                        risk_margin_usd=dynamic_margin_usd,
                    )
                    if success:
                        print(f"✅ [{symbol}] 成交！价格: {current_price} | 动用保证金: ${actual_margin:.2f}")
                        log_system_alert("✅ 下单成交", symbol, f"Tier-{target_tier} 建立 | 成交价: {current_price:.4f} | 保证金: ${actual_margin:.2f}")
                        rt["last_trade_hour"] = rt["latest_closed_ts"]
                        position_tier_state["side"] = actual_signal
                        position_tier_state["highest_tier"] = target_tier
                    else:
                        log_system_alert("❌ 下单失败", symbol, f"OMS 返回 False | 信号: {'做多' if actual_signal==1 else '做空'} | 价格: {current_price:.4f}")

                symbol_payloads[symbol] = {
                    "timestamp": current_time,
                    "server_time": snapshot.get('server_time', int(current_time * 1000)),
                    "symbol": symbol,
                    "current_price": current_price,
                    "macro_score": cached_macro_score,
                    "model_inference": {
                        "raw_prob_up": float(cached_p_up),
                        "raw_prob_flat": float(cached_p_flat),
                        "raw_prob_down": float(cached_p_down),
                        "adj_prob_up_tier1": float(adj_up_top10),
                        "adj_prob_down_tier1": float(adj_down_top10),
                        "macro_score": float(cached_macro_score),
                        "model_count": int(rt["model_count"]),
                        "ensemble_models": [
                            {
                                "name": m["name"],
                                "experiment_id": m["exp_id"],
                                "weight": float(m["weight"]),
                                "prob_up": float(m["prob_up"]),
                                "prob_flat": float(m["prob_flat"]),
                                "prob_down": float(m["prob_down"]),
                            }
                            for m in rt["cached_model_details"]
                        ],
                    },
                    "execution_state": {
                        "current_signal": int(actual_signal),
                        "target_signal": int(target_signal),
                        "target_tier": int(target_tier),
                        "dynamic_leverage": float(dynamic_leverage) if actual_signal != 0 else 0.0,
                        "last_inferred_hour": int(rt["last_inferred_hour"]),
                        "last_signal_evaluated_ts": float(rt["last_signal_evaluated_ts"]),
                        "last_trade_hour": int(rt["last_trade_hour"])
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
                        "stop_coverage": _compact_stop_coverage(inventory_guard["stop_coverage"])
                    },
                    "forced_exit_lock": {
                        "active": bool(forced_exit_lock.get("active", False)),
                        "anchor_closed_ts": int(forced_exit_lock.get("anchor_closed_ts", 0) or 0),
                        "reason": forced_exit_lock.get("reason", ""),
                        "trigger_ts": float(forced_exit_lock.get("trigger_ts", 0.0) or 0.0)
                    }
                }

            except Exception as e:
                print(f"❌ [{symbol}] 主循环单币异常: {e}")

        live_state_payload = {
            "timestamp": time.time(),
            "mode": "multi_symbol",
            "active_symbols": list(runtimes.keys()),
            "symbols": symbol_payloads,
        }
        if symbol_payloads:
            primary_symbol = next(iter(symbol_payloads))
            live_state_payload["primary_symbol"] = primary_symbol
            # 兼容旧 UI / 右脑读取路径：顶层仍放第一个币种的状态。
            live_state_payload.update(symbol_payloads[primary_symbol])

        tmp_state_path = os.path.join(LIVE_DATA_DIR, "live_status.json.tmp")
        target_state_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
        try:
            with open(tmp_state_path, "w", encoding="utf-8") as f:
                json.dump(live_state_payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_state_path, target_state_path)
        except IOError as e:
            print(f"⚠️ 状态持久化写入失败: {e}")

        elapsed = time.time() - cycle_start
        time.sleep(max(1.0, HEARTBEAT_INTERVAL - elapsed))

if __name__ == "__main__":
    run_live()
