import numpy as np  
import json
import os
import sys
import pynvml
import psutil
import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import time  # 新增：引入时间模块控制刷新频率
import requests  # 新增：用于发送 Webhook 请求
# 👇 新增这两行，激活 .env 读取
from dotenv import load_dotenv
load_dotenv()
from quant_config.config import (
    SYMBOL, DATA_DIR, MODEL_DIR, LIVE_DATA_DIR, AUX_SYMBOLS,
    HARD_TP, HARD_SL, INITIAL_CAPITAL, BASE_RISK_BUDGET,
    BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, TRAILING_ATR_PERIOD, TRAILING_ATR_MULTI,
    FORECAST_HORIZON, EXPERIMENT_ID, LIVE_SYMBOL_CONFIGS
)

# --- 核心配置路径 ---
def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _normalize_model_specs(cfg):
    raw_models = cfg.get("models")
    if not raw_models:
        raw_models = [{"experiment_id": cfg.get("experiment_id", EXPERIMENT_ID), "weight": cfg.get("weight", 1.0)}]

    normalized = []
    for raw_model in raw_models:
        model_cfg = raw_model or {}
        exp_id = str(model_cfg.get("experiment_id", cfg.get("experiment_id", EXPERIMENT_ID)))
        weight = float(model_cfg.get("weight", 1.0))
        if weight <= 0:
            continue
        normalized.append({
            "name": str(model_cfg.get("name", exp_id)),
            "experiment_id": exp_id,
            "weight": weight,
        })

    if not normalized:
        normalized = [{"name": EXPERIMENT_ID, "experiment_id": EXPERIMENT_ID, "weight": 1.0}]

    total_weight = sum(m["weight"] for m in normalized)
    for model_cfg in normalized:
        model_cfg["normalized_weight"] = model_cfg["weight"] / total_weight
    return normalized


def _normalize_live_configs():
    if isinstance(LIVE_SYMBOL_CONFIGS, dict) and LIVE_SYMBOL_CONFIGS:
        raw_configs = LIVE_SYMBOL_CONFIGS
    else:
        fallback_symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
        raw_configs = {
            sym: {"experiment_id": EXPERIMENT_ID, "aux_symbols": AUX_SYMBOLS}
            for sym in fallback_symbols
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
ALL_SYMBOLS = list(LIVE_CONFIGS.keys())

# --- UI 页面配置与高频轮询开关 ---
st.set_page_config(page_title="iTransformer 预测引擎", layout="wide")
 
# --- 🧠 按实盘配置加载概率门槛 ---
def _load_thresholds_for_experiment(exp_id):
    json_path = os.path.join(MODEL_DIR, f"dynamic_thresholds_{exp_id}.json")
    if not os.path.exists(json_path):
        return {}, json_path
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    return raw_data.get("thresholds", raw_data), json_path


def _load_weighted_thresholds(model_specs):
    threshold_keys = ["up_top_1", "up_top_5", "up_top_10", "down_top_1", "down_top_5", "down_top_10"]
    weighted = {key: 0.0 for key in threshold_keys}
    model_meta = []
    loaded_any = False

    for spec in model_specs:
        exp_id = spec["experiment_id"]
        try:
            thresholds, path = _load_thresholds_for_experiment(exp_id)
        except Exception:
            thresholds, path = {}, os.path.join(MODEL_DIR, f"dynamic_thresholds_{exp_id}.json")

        loaded = bool(thresholds)
        loaded_any = loaded_any or loaded
        if loaded:
            for key in threshold_keys:
                weighted[key] += float(thresholds.get(key, 0.0)) * spec["normalized_weight"]

        model_meta.append({
            "name": spec.get("name", exp_id),
            "experiment_id": exp_id,
            "weight": spec["normalized_weight"],
            "path": path,
            "loaded": loaded,
        })

    label = ", ".join(f"{m['experiment_id']}@{m['weight']:.2f}" for m in model_meta) or str(EXPERIMENT_ID)
    return (weighted if loaded_any else {}), {
        "experiment_id": label,
        "mode": "weighted_ensemble" if len(model_meta) > 1 else "single_model",
        "models": model_meta,
    }


THRESHOLDS_BY_SYMBOL = {}
THRESHOLD_SOURCE_BY_SYMBOL = {}
for _sym, _cfg in LIVE_CONFIGS.items():
    _th, _meta = _load_weighted_thresholds(_cfg.get("models", []))
    THRESHOLDS_BY_SYMBOL[_sym] = _th
    THRESHOLD_SOURCE_BY_SYMBOL[_sym] = _meta

# ==========================================
# ⚡ 极简系统总控台 (合并刷新控制与硬件监控)
# ==========================================
st.sidebar.markdown("### ⚡ 系统总控台")

# 使用列布局，把开关和频率数字框挤在一行，极致节省空间
col_tog, col_val = st.sidebar.columns([1.2, 1])
with col_tog:
    enable_refresh = st.toggle("🔄 自动刷新", value=True)
with col_val:
    if enable_refresh:
        # 用紧凑的 number_input 替代占地方的 slider，并隐藏 label
        refresh_interval = st.number_input("频率", min_value=3, max_value=180, value=3, step=1, label_visibility="collapsed")
    else:
        refresh_interval = 0

st.sidebar.caption("👇 核心微服务与服务器负载实时状态")

# 💡 核心魔法变量：传给 Fragment 的刷新频率
_run_every = f"{refresh_interval}s" if enable_refresh and refresh_interval > 0 else None

@st.fragment(run_every=_run_every)
def render_status_dashboard():
    # 彻底删除了内部冗余的大标题，直接渲染监控灯泡，让视觉高度统一
    
    # --- 1. 核心进程存活检查 ---
    engine_alive = False
    engine_status_file = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if os.path.exists(engine_status_file):
        if (time.time() - os.path.getmtime(engine_status_file)) < 60:
            engine_alive = True

    # 👇 【新增】：WebSocket 极速网关存活检查
    ws_alive = False
    try:
        # 极速探测本地 8000 端口，只要没有抛出 ConnectionError 就说明网关健在
        test_sym = SYMBOL.replace("/", "_")
        requests.get(f"http://127.0.0.1:8000/api/snapshot/{test_sym}", timeout=1.0)
        ws_alive = True
    except Exception:
        pass

    # --- 2. 硬件资源监控 (GPU / CPU / RAM) ---
    gpu_mem_text = "N/A"
    gpu_pct = 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        current_used = info.used / 1024**3
        total = info.total / 1024**3
        
        # 10 分钟滚动峰值记录器
        peak_file = os.path.join(LIVE_DATA_DIR, "gpu_peak_mem.json")
        mem_history = []
        now_ts = time.time()
        
        if os.path.exists(peak_file):
            try:
                with open(peak_file, 'r') as f:
                    mem_history = json.load(f)
            except: pass
            
        mem_history = [item for item in mem_history if (now_ts - item[0]) <= 600]
        mem_history.append([now_ts, current_used])
        peak_used = max([item[1] for item in mem_history])
        
        try:
            with open(f"{peak_file}.tmp", 'w') as f:
                json.dump(mem_history, f)
            os.replace(f"{peak_file}.tmp", peak_file)
        except: pass

        gpu_mem_text = f"{peak_used:.1f} / {total:.1f} GB (10m峰值)"
        gpu_pct = peak_used / total
    except: pass

    # 👇 【新增】：CPU 与 系统内存监控
    cpu_pct = 0.0
    ram_pct = 0.0
    ram_text = "N/A"
    try:
        # interval=None 会返回自上次调用以来的瞬时 CPU 利用率
        cpu_pct = psutil.cpu_percent(interval=None) / 100.0
        mem_info = psutil.virtual_memory()
        ram_pct = mem_info.percent / 100.0
        ram_text = f"{mem_info.used / 1024**3:.1f} / {mem_info.total / 1024**3:.1f} GB"
    except Exception as e:
        ram_text = "依赖缺失 (需 pip install psutil)"

    # --- 3. UI 渲染 ---
    c1, c2 = st.columns(2)
    with c1:
        st.status("🟢 引擎" if engine_alive else "🔴 引擎", state="complete" if engine_alive else "error")
    with c2:
        st.status("🟢 网关" if ws_alive else "🔴 网关", state="complete" if ws_alive else "error")
    
    # 纵向堆叠三大硬件进度条
    st.caption(f"💾 L4 显存占用: {gpu_mem_text}") 
    st.progress(min(gpu_pct, 1.0))              
    
    st.caption(f"⚙️ CPU 核心负载: {cpu_pct * 100:.1f}%")
    st.progress(min(max(cpu_pct, 0.0), 1.0))
    
    st.caption(f"🧠 系统内存 (RAM): {ram_text}")
    st.progress(min(ram_pct, 1.0))
    
    st.divider()

# 👇 【核心修复】：在调用时，从外部将其推入侧边栏
with st.sidebar:
    render_status_dashboard()


# --- 资产组合监控模块 (Portfolio Monitor) ---
PORTFOLIO_SNAPSHOT_GRACE_SEC = 12

def get_gate_portfolio():
    """架构解耦：读取本地快照，并对瞬时空快照做短暂抗抖处理。"""
    state_file = os.path.join(LIVE_DATA_DIR, "portfolio_state.json")
    cache_key = "portfolio_snapshot_cache"

    def load_cached_snapshot():
        cached = st.session_state.get(cache_key)
        if not cached:
            return None, [], []
        cache_age = time.time() - cached.get('cached_at', 0)
        if cache_age > PORTFOLIO_SNAPSHOT_GRACE_SEC:
            return None, [], []
        return cached['balance'], cached['positions'], cached['open_stops']

    if not os.path.exists(state_file):
        return load_cached_snapshot()
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        balance = state.get('balance', {})
        positions = state.get('positions', [])
        open_stops = state.get('open_stops', [])
        updated_at = float(state.get('updated_at', 0) or 0)
        
        active_positions = [p for p in positions if abs(float(p.get('contracts', 0) or 0)) > 1e-8]
        cached = st.session_state.get(cache_key)
        cached_positions = cached.get('positions', []) if cached else []

        is_suspicious_empty = (
            not active_positions
            and bool(cached_positions)
            and (time.time() - cached.get('cached_at', 0) <= PORTFOLIO_SNAPSHOT_GRACE_SEC)
        ) if cached else False

        if is_suspicious_empty:
            return cached['balance'], cached['positions'], cached['open_stops']

        has_meaningful_snapshot = bool(balance) or bool(active_positions) or bool(open_stops)
        is_fresh = updated_at > 0 and (time.time() - updated_at) <= PORTFOLIO_SNAPSHOT_GRACE_SEC

        if has_meaningful_snapshot or is_fresh:
            st.session_state[cache_key] = {
                'balance': balance,
                'positions': active_positions,
                'open_stops': open_stops,
                'cached_at': time.time(),
            }
            return balance, active_positions, open_stops

        return load_cached_snapshot()
    except Exception:
        return load_cached_snapshot()


def load_trade_history_for_symbol(symbol):
    """稳健加载单币种交易流水，并用最近预测快照补齐杠杆、胜率和版本血统。"""
    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol.replace('/', '_')}.db")
    if not os.path.exists(db_name):
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_name, timeout=5.0, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_trades'")
            if cursor.fetchone() is None:
                return pd.DataFrame()

            # 自动升级旧表结构，确保所有列存在
            for col in ['leverage', 'margin_used', 'order_id']:
                try: 
                    cursor.execute(f"ALTER TABLE live_trades ADD COLUMN {col} REAL")
                except sqlite3.OperationalError: 
                    pass

            # 读取时使用 fillna 填充缺失值，防止计算崩溃
            df_live = pd.read_sql("SELECT * FROM live_trades ORDER BY exec_ts ASC", conn)
            if df_live.empty:
                return pd.DataFrame()

            try:
                df_pred = pd.read_sql(
                    """
                    SELECT predict_ts, leverage_used, kelly_prob, data_version, signal
                    FROM predictions
                    WHERE signal IN (1, -1)
                    ORDER BY predict_ts ASC
                    """,
                    conn
                )
            except Exception:
                df_pred = pd.DataFrame()

        if not df_pred.empty:
            df_live['exec_ts'] = pd.to_datetime(df_live['exec_ts'], format='mixed', errors='coerce')
            df_pred['predict_ts'] = pd.to_datetime(df_pred['predict_ts'], format='mixed', errors='coerce')
            df_pred = df_pred.dropna(subset=['predict_ts']).sort_values('predict_ts')
            df_pred = df_pred.rename(columns={
                'leverage_used': 'pred_leverage',
                'signal': 'original_signal_direction'
            })
            df_live = pd.merge_asof(
                df_live.sort_values('exec_ts'),
                df_pred[['predict_ts', 'pred_leverage', 'kelly_prob', 'data_version', 'original_signal_direction']].sort_values('predict_ts'),
                left_on='exec_ts',
                right_on='predict_ts',
                direction='backward'
            )
        else:
            df_live['pred_leverage'] = np.nan
            df_live['kelly_prob'] = np.nan
            df_live['data_version'] = np.nan
            df_live['original_signal_direction'] = np.nan

        live_lev = pd.to_numeric(df_live.get('leverage', pd.Series(np.nan, index=df_live.index)), errors='coerce')
        pred_lev = pd.to_numeric(df_live.get('pred_leverage', pd.Series(np.nan, index=df_live.index)), errors='coerce')
        df_live['effective_leverage'] = np.where(live_lev > 0, live_lev, np.where(pred_lev > 0, pred_lev, 10.0))
        df_live['xgb_prob'] = pd.to_numeric(df_live.get('kelly_prob', pd.Series(np.nan, index=df_live.index)), errors='coerce')
        df_live['trade_version'] = df_live.get('data_version', pd.Series(np.nan, index=df_live.index)).fillna("unknown")
        df_live.insert(0, '交易对', symbol)
        return df_live
    except Exception as e:
        st.error(f"❌ 加载 {symbol} 交易历史失败: {e}")
        return pd.DataFrame()


def resolve_current_serving_model(symbol):
    """统一解析当前服役模型身份，兼容单模型与加权 ensemble。"""
    sym_key = _normalize_symbol_key(symbol)
    model_specs = LIVE_CONFIGS.get(sym_key, {}).get("models", [])
    primary_exp_id = str(model_specs[0]["experiment_id"]) if model_specs else str(EXPERIMENT_ID or "").strip()
    current_inf_version = primary_exp_id or "unknown"
    curr_m_data_version = current_inf_version
    curr_m_name = "未命名"
    curr_m_created_at = pd.NaT
    model_stack = [
        {
            "experiment_id": m.get("experiment_id", "unknown"),
            "weight": float(m.get("normalized_weight", m.get("weight", 1.0))),
        }
        for m in model_specs
    ]

    registry_file = os.path.join(MODEL_DIR, "model_registry.json")
    if os.path.exists(registry_file):
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                reg_data = json.load(f)
            reg_item = reg_data.get(primary_exp_id, {})
            if reg_item:
                curr_m_data_version = reg_item.get('data_version', curr_m_data_version) or curr_m_data_version
                curr_m_name = reg_item.get('name', curr_m_name) or curr_m_name
                curr_m_created_at = pd.to_datetime(reg_item.get('created_at'), errors='coerce')
        except Exception:
            pass

    model_file = os.path.join(MODEL_DIR, f"itransformer_{symbol}_best_{primary_exp_id}.pth")
    if os.path.exists(model_file):
        if curr_m_name == "未命名":
            curr_m_name = os.path.splitext(os.path.basename(model_file))[0]
        if pd.isna(curr_m_created_at):
            curr_m_created_at = pd.to_datetime(os.path.getmtime(model_file), unit='s')
    else:
        latest_model_file = None
        latest_model_mtime = None
        prefix = f"itransformer_{symbol}_best_"
        try:
            for fname in os.listdir(MODEL_DIR):
                if not (fname.startswith(prefix) and fname.endswith(".pth")):
                    continue
                full_path = os.path.join(MODEL_DIR, fname)
                file_mtime = os.path.getmtime(full_path)
                if latest_model_mtime is None or file_mtime > latest_model_mtime:
                    latest_model_file = full_path
                    latest_model_mtime = file_mtime
        except Exception:
            latest_model_file = None

        if latest_model_file:
            inferred_version = os.path.basename(latest_model_file)[len(prefix):-4]
            if current_inf_version in {"unknown", "default_run", "v1.0"}:
                current_inf_version = inferred_version
                curr_m_data_version = inferred_version
            if curr_m_name == "未命名":
                curr_m_name = os.path.splitext(os.path.basename(latest_model_file))[0]
            if pd.isna(curr_m_created_at):
                curr_m_created_at = pd.to_datetime(latest_model_mtime, unit='s')

    if pd.isna(curr_m_created_at):
        curr_m_created_at = pd.to_datetime('2000-01-01')

    if len(model_stack) > 1:
        current_inf_version = " + ".join(f"{m['experiment_id']}@{m['weight']:.2f}" for m in model_stack)
        curr_m_name = f"Ensemble x{len(model_stack)}"

    return {
        'inference_version': current_inf_version,
        'data_version': curr_m_data_version,
        'name': curr_m_name,
        'created_at': curr_m_created_at,
        'model_stack': model_stack,
    }


def _shadow_audit_db_path():
    return os.path.join(LIVE_DATA_DIR, "live_model_shadow_audit.db")


def load_live_shadow_model_audit(symbol):
    db_path = _shadow_audit_db_path()
    if not os.path.exists(db_path):
        return pd.DataFrame(), db_path
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            df = pd.read_sql(
                """
                SELECT *
                FROM solo_model_inference
                WHERE symbol = ?
                ORDER BY decision_ts DESC
                """,
                conn,
                params=(_normalize_symbol_key(symbol),),
            )
        return df, db_path
    except Exception:
        return pd.DataFrame(), db_path


def _calc_shadow_max_drawdown(pnl_series):
    pnl = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if pnl.empty:
        return np.nan
    equity = (1.0 + pnl).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def summarize_live_shadow_model_audit(audit_df, model_stack):
    if audit_df.empty:
        return pd.DataFrame()

    rows = []
    weight_by_exp = {
        str(m.get("experiment_id", "")): float(m.get("weight", 0.0) or 0.0)
        for m in model_stack
    }

    for exp_id, g in audit_df.groupby("experiment_id"):
        settled = g[pd.to_numeric(g.get("settled", 0), errors="coerce").fillna(0).astype(int) == 1].copy()
        adj_trades = g[pd.to_numeric(g.get("adj_signal", 0), errors="coerce").fillna(0).astype(int) != 0]
        settled_adj = settled[pd.to_numeric(settled.get("adj_signal", 0), errors="coerce").fillna(0).astype(int) != 0].copy()
        settled_raw = settled[pd.to_numeric(settled.get("raw_signal", 0), errors="coerce").fillna(0).astype(int) != 0].copy()

        adj_pnl = pd.to_numeric(settled_adj.get("adj_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        raw_pnl = pd.to_numeric(settled_raw.get("raw_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        settled_adj_chrono = settled_adj.sort_values("decision_ts")
        settled_raw_chrono = settled_raw.sort_values("decision_ts")
        last = g.sort_values("decision_ts").iloc[-1]

        rows.append({
            "experiment_id": exp_id,
            "live_weight": weight_by_exp.get(str(exp_id), float(g["live_weight"].iloc[0]) if "live_weight" in g else 0.0),
            "inference_count": len(g),
            "shadow_trade_count": len(adj_trades),
            "settled_trade_count": len(adj_pnl),
            "pending_trade_count": max(0, len(adj_trades) - len(adj_pnl)),
            "win_rate": float((adj_pnl > 0).mean()) if len(adj_pnl) else np.nan,
            "avg_pnl": float(adj_pnl.mean()) if len(adj_pnl) else np.nan,
            "total_pnl": float(adj_pnl.sum()) if len(adj_pnl) else 0.0,
            "max_drawdown": _calc_shadow_max_drawdown(settled_adj_chrono.get("adj_pnl", pd.Series(dtype=float))),
            "raw_win_rate": float((raw_pnl > 0).mean()) if len(raw_pnl) else np.nan,
            "raw_total_pnl": float(raw_pnl.sum()) if len(raw_pnl) else 0.0,
            "raw_max_drawdown": _calc_shadow_max_drawdown(settled_raw_chrono.get("raw_pnl", pd.Series(dtype=float))),
            "long_count": int((pd.to_numeric(g.get("adj_signal", 0), errors="coerce") == 1).sum()),
            "short_count": int((pd.to_numeric(g.get("adj_signal", 0), errors="coerce") == -1).sum()),
            "last_signal": int(last.get("adj_signal", 0) or 0),
            "last_tier": int(last.get("adj_tier", 0) or 0),
            "last_p_up": float(last.get("p_up", 0.0) or 0.0),
            "last_p_down": float(last.get("p_down", 0.0) or 0.0),
        })

    return pd.DataFrame(rows).sort_values(
        by=["total_pnl", "win_rate", "settled_trade_count"],
        ascending=[False, False, False],
        na_position="last",
    )


st.title("iTransformer 预测引擎")

# ==========================================
# 🧠 V3 宏观情报横幅 (DeepSeek Marquee)
# ==========================================
try:
    if os.path.exists(os.path.join(LIVE_DATA_DIR, "macro_intelligence.db")):
        with sqlite3.connect(os.path.join(LIVE_DATA_DIR, "macro_intelligence.db"), timeout=2.0) as m_conn:
            m_cur = m_conn.cursor()
            m_cur.execute("SELECT score, marquee_text, urgency, decision_ts FROM llm_decisions ORDER BY decision_ts DESC LIMIT 1")
            m_row = m_cur.fetchone()
            if m_row:
                m_score, m_text, m_urgency, m_ts = m_row
                
                # 根据紧急程度和得分决定渐变背景色
                if m_urgency == "high":
                    bg_color = "linear-gradient(90deg, #4b0000 0%, #800000 50%, #4b0000 100%)" if m_score < 0 else "linear-gradient(90deg, #004b00 0%, #008000 50%, #004b00 100%)"
                    border_color = "#ff4b4b" if m_score < 0 else "#21c354"
                    icon = "🚨 [高危预警]" if m_score < 0 else "🚀 [史诗逼空]"
                else:
                    bg_color = "linear-gradient(90deg, #1E1E1E 0%, #2A2A2A 50%, #1E1E1E 100%)"
                    border_color = "#444"
                    icon = "📻 [DeepSeek 宏观播报]"
                
                # 注入 HTML 跑马灯
                st.markdown(
                    f"""
                    <div style="background: {bg_color}; border: 1px solid {border_color}; padding: 12px 15px; border-radius: 8px; margin-bottom: 25px; display: flex; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
                        <div style="font-size: 1.3rem; margin-right: 15px; font-weight: bold; color: {border_color};">{icon}</div>
                        <div style="flex-grow: 1; overflow: hidden; white-space: nowrap;">
                            <marquee scrollamount="6" style="color: #E0E0E0; font-size: 1.15rem; font-weight: 500; letter-spacing: 1px;">
                                (快照时间: {m_ts}) &nbsp;&nbsp;|&nbsp;&nbsp; {m_text} &nbsp;&nbsp;|&nbsp;&nbsp; 情绪量化得分: <b>{m_score:+.2f}</b>
                            </marquee>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
except Exception:
    pass
# ====================================================================
# 🏦 动态切片 1：实时仓位与资金监控中心 (15秒一刷)
# ====================================================================
@st.fragment(run_every=_run_every)
def render_realtime_portfolio():
    balance, positions, all_open_stops = get_gate_portfolio()
    if balance is None:
        st.warning("⏳ 正在等待后台量化引擎启动并同步资产快照...")
        return
        
    with st.expander("🏦 Gate.io 仓位监控中心", expanded=True):
        try:
            # 1. 提前读取吊灯内存，用于算全局保底
            chandelier_mem = {}
            if os.path.exists(os.path.join(LIVE_DATA_DIR, "chandelier_state.json")):
                try:
                    with open(os.path.join(LIVE_DATA_DIR, "chandelier_state.json"), "r") as f:
                        chandelier_mem = json.load(f)
                except Exception: 
                    pass

            # 适配 CCXT 解析 Gate.io 余额的标准结构
            wallet_balance = float(balance.get('total', {}).get('USDT', 0))
            avail_margin = float(balance.get('free', {}).get('USDT', 0))
            
            unrealized_pnl = 0.0
            total_margin_used = 0.0
            # 👇 合并为一个综合累加器
            total_unified_min_pnl = 0.0  # 综合守底
            
            for p in positions:
                p_info = p['info']
                symbol_raw = p['symbol']
                base_symbol = symbol_raw.split(':')[0]
                
                # 累加未实现盈亏
                unrealized_pnl += float(p_info.get('unrealised_pnl', 0))
                
                # 算占用保证金
                notional = float(p.get('notional', p_info.get('value', 0)))
                if notional == 0:
                    contracts = abs(float(p_info.get('pos', p_info.get('contracts', 0))))
                    mark_px = float(p_info.get('markPx', p_info.get('mark_price', 0)))
                    contract_size = float(p.get('contractSize', 1))
                    notional = contracts * contract_size * mark_px
                    
                leverage_used = float(p_info.get('lever', p_info.get('leverage', '1')))
                if leverage_used <= 0: leverage_used = 1.0
                actual_margin = notional / leverage_used
                total_margin_used += actual_margin
                
                # ==========================================
                # 👇 新增核心：计算单仓保底盈亏金额 (最严防线距离 * 保证金杠杆)
                # ==========================================
                pos_side = str(p.get('side', '')).upper()
                entry_px = float(p_info.get('avgPx', p_info.get('entry_price', 0)))
                mark_px = float(p_info.get('markPx', p_info.get('mark_price', 0)))
                
                # 从统一拉取的内存列表中，过滤出该币种的物理止损
                physical_sl_px = 0.0
                symbol_stops = [o for o in all_open_stops if o['symbol'] == symbol_raw]
                for o in symbol_stops:
                    stop_px = float(o.get('stopPrice', o['info'].get('stopPrice', o['info'].get('stop_price', 0))))
                    if stop_px > 0 and mark_px > 0:
                        if pos_side == 'LONG' and stop_px < mark_px: physical_sl_px = stop_px
                        elif pos_side == 'SHORT' and stop_px > mark_px: physical_sl_px = stop_px
                
                # 提取内存中的虚拟吊灯
                virtual_sl_px = 0.0
                if base_symbol in chandelier_mem and 'current_stop' in chandelier_mem[base_symbol]:
                    virtual_sl_px = float(chandelier_mem[base_symbol]['current_stop'])
                
                # 👇 【UI重构：合并双防线，计算单仓综合守底金额并累加】
                if entry_px > 0 and mark_px > 0:
                    coin_amount = notional / mark_px
                    
                    unified_sl_px = 0.0
                    if pos_side == 'LONG':
                        # 做多：取两者最高价（离现价最近的防线）
                        unified_sl_px = max(physical_sl_px, virtual_sl_px)
                    elif pos_side == 'SHORT':
                        # 做空：过滤掉 0 之后，取两者最低价
                        valid_stops = [s for s in [physical_sl_px, virtual_sl_px] if s > 0]
                        unified_sl_px = min(valid_stops) if valid_stops else 0.0
                        
                    if unified_sl_px > 0:
                        if pos_side == 'LONG':
                            unified_pnl = (unified_sl_px - entry_px) * coin_amount
                        elif pos_side == 'SHORT':
                            unified_pnl = (entry_px - unified_sl_px) * coin_amount
                        else:
                            unified_pnl = 0.0
                        total_unified_min_pnl += unified_pnl

            # 账户总权益 = 真实钱包余额 + 浮动盈亏
            total_equity = wallet_balance + unrealized_pnl

        except Exception as e:
            # 异常时所有变量归零
            total_equity, unrealized_pnl, avail_margin, wallet_balance, total_margin_used, total_unified_min_pnl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            st.warning(f"⚠️ 解析 Gate 资产列表出现异常: {e}")

        # 计算已实现盈亏与历史累计收益率
        initial_cap = float(INITIAL_CAPITAL)  # 直接来自 config
        historical_roi = (total_equity - initial_cap) / initial_cap
        realized_pnl = wallet_balance - initial_cap

        margin_utilization = (total_margin_used / total_equity) if total_equity > 0 else 0.0

        # UI 重构：增加进度条刻度
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("账户总权益", f"${total_equity:,.2f}", delta=f"累计收益率: {historical_roi:+.2%}", delta_color="normal" if historical_roi >= 0 else "inverse")
        m_col2.metric("已实现盈亏", f"${realized_pnl:,.2f}", delta="已落袋" if realized_pnl >= 0 else "已亏损", delta_color="normal" if realized_pnl >= 0 else "inverse")
        
        # 👇 【核心呈现：综合保底收益金额展示】
        delta_label = f"综合保底收益 {total_unified_min_pnl:+.2f}" if positions else "浮动中"
        m_col3.metric("未实现盈亏", f"${unrealized_pnl:,.2f}", delta=delta_label, delta_color="normal" if unrealized_pnl >= 0 else "inverse")
        
        # 保持第四个卡片为战略预备队雷达
        m_col4.metric("资金使用率", f"{margin_utilization:.1%}")
        
        # 👇 【UI 修复】：让进度条回归真实的 0~100% 绝对物理刻度
        st.progress(
            min(margin_utilization, 1.0), 
            text=f"🛡️ 场内已用资金: {total_margin_used:,.2f} USDT | 可用保证金: {avail_margin:,.2f} USDT"
        )
        pass
        active_pos_count = len(positions) if positions else 0
        st.markdown(f"##### 📍 当前活跃持仓({active_pos_count})")
        if not positions:
            st.info("📭 当前无任何持仓。引擎正在等待完美击球点...")
        else:
            pos_data = []
            
            # 在循环前，一次性读取后台硬盘里的吊灯黑匣子
            chandelier_mem = {}
            if os.path.exists(os.path.join(DATA_DIR, "chandelier_state.json")):
                try:
                    with open(os.path.join(DATA_DIR, "chandelier_state.json"), "r") as f:
                        chandelier_mem = json.load(f)
                except Exception:
                    pass
                    
            for p in positions:
                info = p['info']
                symbol_raw = p['symbol'] 
                base_symbol = symbol_raw.split(':')[0]
                
                real_tp, real_sl = "未挂载 ⚠️", "未挂载 ⚠️"
                physical_sl_px = 0.0 # 👇 新增：用于数学计算的物理止损价
                
                try:
                    # 👇 从 engine.py 传过来的全量订单列表中，瞬间过滤出当前币种的条件单
                    open_orders = [o for o in all_open_stops if o['symbol'] == symbol_raw]
                    mark_px = float(info.get('markPx', info.get('mark_price', 0)))
                    pos_side = str(p.get('side', '')).upper()
                    entry_px = float(info.get('avgPx', info.get('entry_price', 0)))
                    
                    for o in open_orders:
                        stop_px = float(o.get('stopPrice', o['info'].get('stopPrice', o['info'].get('stop_price', 0))))
                        if stop_px > 0 and mark_px > 0:
                            if pos_side == 'LONG':
                                if stop_px > mark_px: 
                                    real_tp = f"${stop_px:,.4f}"
                                else: 
                                    real_sl = f"${stop_px:,.4f}"
                                    physical_sl_px = stop_px
                            elif pos_side == 'SHORT':
                                if stop_px < mark_px: 
                                    real_tp = f"${stop_px:,.4f}"
                                else: 
                                    real_sl = f"${stop_px:,.4f}"
                                    physical_sl_px = stop_px
                except Exception:
                    real_tp, real_sl = "查询异常", "查询异常"

                upl_raw = info.get('unrealised_pnl')
                upl = float(upl_raw) if upl_raw is not None else 0.0
                
                pct_raw = p.get('percentage')
                if pct_raw is None: pct_raw = info.get('uplRatio')
                upl_ratio = float(pct_raw) if pct_raw is not None else 0.0
                if -1.0 < upl_ratio < 1.0 and upl_ratio != 0.0: upl_ratio *= 100

                notional = p.get('notional', float(info.get('value', 0)))
                if notional == 0:
                    contracts = abs(float(info.get('pos', info.get('contracts', 0))))
                    mark_px = float(info.get('markPx', info.get('mark_price', 0)))
                    contract_size = p.get('contractSize', 1) 
                    notional = contracts * contract_size * mark_px

                leverage_used = float(info.get('lever', info.get('leverage', '1')))
                if leverage_used == 0: leverage_used = 1.0
                actual_margin = notional / leverage_used

                if upl_ratio == 0.0 and actual_margin > 0:
                    upl_ratio = (upl / actual_margin) * 100
                
                # 从内存中提取该币种的实时吊灯线
                virtual_sl = "初始化中 ⏳"
                virtual_sl_px = 0.0 
                if base_symbol in chandelier_mem and 'current_stop' in chandelier_mem[base_symbol]:
                    virtual_sl_px = float(chandelier_mem[base_symbol]['current_stop'])
                    virtual_sl = f"${virtual_sl_px:,.4f}"
                
                # ==========================================
                # 👇 【核心重构】：分别计算双防线的独立保底收益率
                # ==========================================
                # 1. 计算底层物理 OCO 止损的保底 ROE
                oco_min_roe_str = "未挂载 ⚠️"
                if physical_sl_px > 0 and entry_px > 0:
                    if pos_side == 'LONG':
                        oco_roe = (physical_sl_px - entry_px) / entry_px * leverage_used * 100
                    else:
                        oco_roe = (entry_px - physical_sl_px) / entry_px * leverage_used * 100
                    oco_min_roe_str = f"{oco_roe:+.2f}%"

                pos_data.append({
                    "交易对": symbol_raw,
                    "方向": "🟢 做多" if pos_side == 'LONG' else "🔴 做空",
                    "杠杆": f"{leverage_used:.0f}x",
                    "占用保证金": f"${actual_margin:,.2f}",
                    "开仓均价": f"${entry_px:,.4f}",                    
                    "防守止损 (SL)": real_sl,
                    "当前标记价": f"${float(info.get('markPx', info.get('mark_price', 0))):,.4f}",
                    "浮动盈亏 (USDT)": f"{upl:,.2f}",
                    "收益率 (ROE)": f"{upl_ratio:+.2f}%", 
                    # 👇 重组字段名，并将双防线保底收益移到表格最末端，方便横向对比
                    "保底收益 (OCO)": oco_min_roe_str
                })
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, width='stretch')
            pass
render_realtime_portfolio()

# ====================================================================
# 📊 静态切片：策略分析驾驶舱与历史本纪 (无动作绝对不刷新，解放 CPU)
# ====================================================================
initial_cap = float(INITIAL_CAPITAL)  # 直接来自 config
actual_track_list = ALL_SYMBOLS
hist_data = []
total_ml_samples = 0

for sym in actual_track_list:
    df_hist = load_trade_history_for_symbol(sym)
    if not df_hist.empty:
        total_ml_samples += int(df_hist['side'].isin(['buy', 'sell']).sum())
        hist_data.append(df_hist)

# ==========================================
# 👇 核心修复：初始化兜底的 DataFrame
# ==========================================
combined_hist = pd.DataFrame()
df_closed_trades = pd.DataFrame()

if hist_data:
    combined_hist = pd.concat(hist_data, ignore_index=True)
    combined_hist.sort_values(by='exec_ts', ascending=True, inplace=True)
    
    # 预处理 ROE 和持仓时间
    closed_records = []
    target_info_list = []
    entry_memory = {} 
    rev_flags = [] 
    is_closing_list = [] # 👈 新增：标记是否为平仓动作
    
    for idx, row in combined_hist.iterrows():
        sym = row['交易对']
        side = row['side']
        px = float(row['fill_price'])
        ts = pd.to_datetime(row['exec_ts'])
        
        # 智能识别逻辑：如果当前有持仓且动作相反，则判定为平仓
        mem = entry_memory.get(sym)
        is_opening = side in ['buy', 'sell']
        is_closing_action = side in ['close_long', 'close_short', 'exchange_close', 'bailout_close', 'bailout_panic_sell']
        
        real_is_closing = False
        if is_closing_action:
            real_is_closing = True
        elif is_opening and mem and mem.get('px', 0) > 0:
            # 如果动作是 buy 且之前是 sell，或者动作是 sell 且之前是 buy -> 判定为平仓
            if (side == 'buy' and mem['side'] == 'sell') or (side == 'sell' and mem['side'] == 'buy'):
                real_is_closing = True

        if real_is_closing:
            # 执行平仓逻辑
            is_closing_list.append(True)
            if mem and mem.get('px', 0) > 0:
                entry_px = mem['px']
                entry_side = mem['side']
                if entry_side == 'buy':
                    raw_pnl = (px - entry_px) / entry_px
                else:
                    raw_pnl = (entry_px - px) / entry_px
                
                real_roe = raw_pnl * mem['lev']
                closed_records.append({
                    'exec_ts': ts,
                    'symbol': sym,
                    'real_roe': real_roe,
                    'account_impact': (real_roe * mem['margin']) / initial_cap,
                    'hold_hours': (ts - mem['ts']).total_seconds() / 3600,
                    'version': mem.get('version', 'unknown'), # 👈 继承入场时的血统
                    'xgb_prob': mem.get('xgb_prob', 0.5)    # 👈 继承入场时的概率
                })
                color = "🟢" if real_roe > 0 else "🔴"
                target_info_list.append(f"{color} 实际: {real_roe:+.2%}")
                rev_flags.append(False)
                entry_memory[sym] = {'px': 0}
            else:
                target_info_list.append("💰 历史遗留单 (匹配失败)")
                rev_flags.append(False)
        elif is_opening:
            # 执行开仓逻辑
            is_closing_list.append(False)
            entry_memory[sym] = {
                'px': px, 
                'side': side, 
                'lev': float(row.get('leverage', 1.0)), 
                'ts': ts, 
                'margin': float(row.get('margin_used', 100.0)),
                'version': row.get('trade_version', 'v1.0'),
                'xgb_prob': row.get('xgb_prob', 0.5)
            }
            tp_move = abs(HARD_TP)
            sl_move = abs(HARD_SL)
            target_info_list.append(f"⏳ 预期: +{tp_move*100:.1f}% / -{sl_move*100:.1f}%")
            rev_flags.append(row.get('is_ai_rev', False))
        else:
            is_closing_list.append(False)
            target_info_list.append("-")
            rev_flags.append(False)
            
    if not combined_hist.empty:
        # 补齐长度
        while len(target_info_list) < len(combined_hist): target_info_list.append("-")
        while len(rev_flags) < len(combined_hist): rev_flags.append(False)
        while len(is_closing_list) < len(combined_hist): is_closing_list.append(False)
        
        combined_hist['预期/实际收益'] = target_info_list[:len(combined_hist)]
        combined_hist['is_ai_rev'] = rev_flags[:len(combined_hist)]
        combined_hist['is_closing_trade'] = is_closing_list[:len(combined_hist)] # 👈 写入标记
        df_closed_trades = pd.DataFrame(closed_records)
# ==========================================
# 📍 历史本纪
# ==========================================
st.markdown("##### 📜 历史本纪")
with st.expander("查看由机器人自动执行的历史订单记录", expanded=False):
    if not combined_hist.empty:
        # 恢复倒序展示（最新鲜的订单在最上面）
        combined_hist.sort_values(by='exec_ts', ascending=False, inplace=True)

        display_hist = combined_hist.rename(columns={
            'exec_ts': '执行时间',
            'side': '方向',
            'fill_price': '执行时成交价',
            'slippage': '滑点损耗'
        })
            
        # 兼容旧数据表：如果字段存在则重命名并格式化，不存在则补空
        if 'margin_used' in display_hist.columns:
            display_hist.rename(columns={'margin_used': '开仓保证金'}, inplace=True)
            display_hist['开仓保证金'] = display_hist['开仓保证金'].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) and float(x) > 0 else "-")
        else:
            display_hist['开仓保证金'] = "-"
            
        def map_side(row):
            x = row['方向']
            is_closing = row.get('is_closing_trade', False)
            is_rev = row.get('is_ai_rev', False)
            flag = " 🔄(AI反杀)" if is_rev else ""

            if x == 'buy':
                return ("✔️ 智能平仓" if is_closing else "🟢 智能开多") + flag
            if x == 'sell':
                return ("✔️ 智能平仓" if is_closing else "🔴 智能开空") + flag
            if x == 'bailout_close': return "🎯 状态机止盈/损"
            if x == 'bailout_panic_sell': return "🆘 右脑紧急熔断"
            if x == 'exchange_close': return "🏁 交易所 OCO 触发"
            return x
            
        # 👇 【终极修复：先翻译中文方向，再根据中文方向决定是否显示胜率】
        display_hist['方向'] = display_hist.apply(map_side, axis=1)
        
        display_hist['滑点损耗'] = display_hist['滑点损耗'].apply(lambda x: f"{x:.3%}")
        display_hist['执行时成交价'] = display_hist['执行时成交价'].apply(lambda x: f"${float(x):,.4f}" if pd.notnull(x) else "N/A")
        st.dataframe(display_hist[['执行时间', '交易对', '方向', '开仓保证金', '执行时成交价', '预期/实际收益', '滑点损耗']], hide_index=True, width='stretch')
    else:
        st.info("📭 本地数据库暂无机器人的自动开仓记录，请耐心等待首单信号。")  
# ==========================================
# 📊 策略分析驾驶舱 (Analytics Dashboard)
# ==========================================
st.markdown("---")
st.markdown("### 📊 策略分析驾驶舱")

# 👇 【新增：全局时间区间选择器】
if not df_closed_trades.empty:
    # 提取历史数据中的最早和最晚日期作为日历边界
    min_date = df_closed_trades['exec_ts'].min().date()
    max_date = df_closed_trades['exec_ts'].max().date()
    
    # 使用两列布局，让日期选择器在左侧，留出呼吸空间
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        selected_dates = st.date_input(
            "📅 选择复盘时间区间 (起 - 止)",
            value=(min_date, max_date), # 默认全选
            min_value=min_date,
            max_value=max_date,
            key="ui_date_filter"
        )
    
    # 只有当用户确实选取了两个完整的日期时，才执行切片
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
        # 将日期格式转换为 datetime，并确保结束日期包含当天到 23:59:59
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # 🔪 核心动作：在送给图表渲染之前，直接切断底层 DataFrame
        df_closed_trades = df_closed_trades[(df_closed_trades['exec_ts'] >= start_dt) & (df_closed_trades['exec_ts'] <= end_dt)]

dash_tab1, dash_tab2, dash_tab3, dash_tab4 = st.tabs(["📈 资金与回撤", "🗓️ 日度盈亏热力", "⏳ 盈亏比散点", "🤖 AI 实盘绩效"])

with dash_tab1:
    if not df_closed_trades.empty:
        df_closed_trades = df_closed_trades.sort_values('exec_ts')
        df_closed_trades['cum_roe'] = df_closed_trades['account_impact'].cumsum()
        df_closed_trades['equity'] = 1 + df_closed_trades['cum_roe'] 
        df_closed_trades['max_equity'] = df_closed_trades['equity'].cummax()
        df_closed_trades['drawdown'] = (df_closed_trades['equity'] / df_closed_trades['max_equity']) - 1
        
        from plotly.subplots import make_subplots
        fig_eq = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 资金主线 (平滑曲线 + 幽灵透明填充)
        fig_eq.add_trace(go.Scatter(
            x=df_closed_trades['exec_ts'], y=df_closed_trades['equity'],
            name="账户净值", mode='lines', 
            line=dict(color='#00E5FF', width=2.5, shape='spline'), # spline 平滑过渡
            fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.05)', # 极微弱的赛博蓝底色
            hovertemplate="时间: %{x}<br>净值: <b>%{y:.4f}</b><extra></extra>"
        ), secondary_y=False)
        
        # 水下回撤面积图 (去除边框，只留血红色的深渊填充)
        fig_eq.add_trace(go.Scatter(
            x=df_closed_trades['exec_ts'], y=df_closed_trades['drawdown'],
            name="水下回撤", mode='lines', fill='tozeroy', 
            line=dict(color='rgba(255, 51, 102, 0)'), # 隐藏线条本身
            fillcolor='rgba(255, 51, 102, 0.2)', # 机构级半透明血红色
            hovertemplate="回撤: <b>%{y:.2%}</b><extra></extra>"
        ), secondary_y=True)
        
        fig_eq.update_layout(
            height=400, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', # 彻底透明化背景
            hovermode="x unified", # 彭博终端标配的全局十字准星悬浮窗
            hoverlabel=dict(bgcolor="#1E1E1E", font_size=13, font_family="Courier New"),
            showlegend=False,
            spikedistance=-1 # 十字准星全局贯穿
        )
        
        # 计算自适应的 Y 轴动态范围 (上下预留 10% 的视觉呼吸空间)
        eq_min = df_closed_trades['equity'].min()
        eq_max = df_closed_trades['equity'].max()
        y_padding = (eq_max - eq_min) * 0.1 if eq_max > eq_min else 0.05

        # 坐标轴美化 (去除杂乱网格线，保留极弱参考线，强制自适应 Y 轴视野)
        fig_eq.update_xaxes(showgrid=False, showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot", spikecolor="#666666", spikethickness=1)
        fig_eq.update_yaxes(title_text="账户净值 (Equity)", secondary_y=False, tickformat=".4f", showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, range=[eq_min - y_padding, eq_max + y_padding])
        fig_eq.update_yaxes(title_text="历史回撤幅度", secondary_y=True, tickformat=".0%", range=[-0.6, 0], showgrid=False, zeroline=False)
        
        st.plotly_chart(fig_eq, width='stretch')
    else:
        st.info("💡 暂无历史平仓数据，等待第一笔交易完结后在此生成全景资金曲线。")
        
with dash_tab2:
    if not df_closed_trades.empty:
        # 1. 按天聚合盈亏
        df_daily = df_closed_trades.resample('D', on='exec_ts')['account_impact'].sum().reset_index()
        
        # 2. 提取日历坐标：X轴为绝对日期，Y轴为星期几
        df_daily['weekday'] = df_daily['exec_ts'].dt.weekday # 0=周一, 6=周日
        df_daily['date_str'] = df_daily['exec_ts'].dt.strftime('%Y-%m-%d')
        
        # 👇 【核心修复】：计算该日期所在周的“周一”作为统一的 X 轴坐标
        # 这样同一周的 7 天就会被完美挤在同一列里！
        df_daily['week_start'] = df_daily['exec_ts'] - pd.to_timedelta(df_daily['weekday'], unit='d')
        
        # 3. 绘制 GitHub 风格的方块热力图
        import plotly.graph_objects as go
        fig_cal = go.Figure(data=go.Scatter(
            x=df_daily['week_start'], # 👈 修改：X 轴使用该周的起点
            y=df_daily['weekday'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=22, # 方块大小
                color=df_daily['account_impact'],
                colorscale=['#FF3030', '#222222', '#00FF7F'], # 红(亏) -> 黑(平) -> 绿(赚)
                showscale=True,
                cmid=0, # 强制将 0% 锚定在黑色中间态
                colorbar=dict(title="ROE", tickformat=".1%")
            ),
            text=df_daily.apply(lambda row: f"日期: {row['date_str']}<br>全账户净收益: {row['account_impact']:+.2%}", axis=1),
            hoverinfo='text'
        ))
        
        # 4. 优化坐标轴视觉
        fig_cal.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5, 6],
                ticktext=['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                autorange="reversed", # 星期一在最上面
                showgrid=False
            ),
            # 👇 新增：强制 X 轴按时间刻度显示，并且隐藏网格线
            xaxis=dict(showgrid=False, type='date', dtick="604800000", tickformat="%m-%d"), # 604800000毫秒=1周
            height=350,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=20),
            title="📅 GitHub 风格日度盈亏热力阵列"
        )
        
        st.plotly_chart(fig_cal, width='stretch')
    else:
        st.info("💡 暂无数据生成日度盈亏日历。")
        
with dash_tab3:
    if not df_closed_trades.empty and len(df_closed_trades) >= 2:
        # 颜色映射：盈利为绿，亏损为红
        colors = np.where(df_closed_trades['real_roe'] > 0, '#2ECC71', '#E74C3C')
        
        fig_scatter = go.Figure()
        
        # 绘制散点图
        fig_scatter.add_trace(go.Scatter(
            x=df_closed_trades['hold_hours'],
            y=df_closed_trades['real_roe'],
            mode='markers',
            marker=dict(
                size=10, # 统一大小，保持盘面干净
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='rgba(255,255,255,0.2)') # 给点增加一点反光质感边框
            ),
            text=df_closed_trades['symbol'], # 悬浮提示显示的币种名
            customdata=np.stack((df_closed_trades['exec_ts'].dt.strftime('%Y-%m-%d %H:%M'),), axis=-1),
            hovertemplate="<b>%{text}</b><br>平仓时间: %{customdata[0]}<br>持仓时间: <b>%{x:.2f} 小时</b><br>最终盈亏: <b>%{y:.2%}</b><extra></extra>"
        ))
        
        fig_scatter.update_layout(
            height=380, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode="closest", # 散点图使用 closest 悬浮模式最精确
            hoverlabel=dict(bgcolor="#1E1E1E", font_size=13, font_family="Courier New")
        )
        
        # 坐标轴美化 (强化零轴，形成明确的“四大象限”)
        fig_scatter.update_xaxes(
            title_text="持仓时间 (Hours)", 
            showgrid=True, gridcolor='rgba(255,255,255,0.05)', 
            zeroline=False
        )
        fig_scatter.update_yaxes(
            title_text="最终收益率 (ROE)", tickformat=".1%", 
            showgrid=True, gridcolor='rgba(255,255,255,0.05)', 
            zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=1.5 # 粗体零轴分界
        )
        
        st.plotly_chart(fig_scatter, width='stretch')
        
        # 在图表下方追加一个极简的系统级诊断统计
        win_rate = (df_closed_trades['real_roe'] > 0).mean()
        avg_hold = df_closed_trades['hold_hours'].mean()
        st.caption(f"💡 **风控雷达洞察**: 当前实盘胜率 **{win_rate:.1%}** | 平均持仓 **{avg_hold:.1f}** 小时。<br>*(注：完美的系统应当呈现『左侧红点密集 - 迅速止损』，『右侧绿点发散 - 利润奔跑』的象限分布。)*", unsafe_allow_html=True)
        # ==========================================
        # 👇 【新增功能】：各币种独立表现考核排行榜
        # ==========================================
        st.markdown("##### 🗄️ 各币种独立表现考核排行榜")
        
        # 使用 Pandas 的 groupby 进行多维聚合统计
        coin_stats = df_closed_trades.groupby('symbol').agg(
            出手次数=('symbol', 'count'),
            胜率=('real_roe', lambda x: (x > 0).mean()),
            累计净值贡献=('account_impact', 'sum'),
            单次平均ROE=('real_roe', 'mean'),
            平均持仓小时=('hold_hours', 'mean')
        ).reset_index()

        # 按照“累计净值贡献”降序排列，谁赚钱最多谁排第一
        coin_stats = coin_stats.sort_values(by='累计净值贡献', ascending=False)

        # 格式化数据，供前端漂亮地渲染
        coin_stats['胜率'] = coin_stats['胜率'].apply(lambda x: f"{x:.1%}")
        coin_stats['累计净值贡献'] = coin_stats['累计净值贡献'].apply(lambda x: f"{x:+.2%}")
        coin_stats['单次平均ROE'] = coin_stats['单次平均ROE'].apply(lambda x: f"{x:+.2%}")
        coin_stats['平均持仓小时'] = coin_stats['平均持仓小时'].apply(lambda x: f"{x:.1f} h")

        # 渲染出漂亮的无边框 DataFrame
        st.dataframe(
            coin_stats,
            hide_index=True,
            width='stretch',
            column_config={
                "symbol": "交易标的"
            }
        )
    else:
        st.info("💡 数据量不足，等待更多平仓记录...")

with dash_tab4:
    st.markdown("##### 🤖 当前服役主模型 (Live) 实盘绩效监控")
    st.caption("此处仅对当前实盘正在加载的 AI 模型进行单独剥离分析，排除历史其他老旧型号的混合数据污染，用于直观评估【大模型当下的实战准确度与诚实度】。")

    serving_symbol = _normalize_symbol_key(st.session_state.get("ui_radar_target") or SYMBOL)
    if serving_symbol not in ALL_SYMBOLS and ALL_SYMBOLS:
        serving_symbol = ALL_SYMBOLS[0]
    model_meta = resolve_current_serving_model(serving_symbol)
    current_inf_version = model_meta['inference_version']
    curr_m_data_version = model_meta['data_version']
    curr_m_created_at = model_meta['created_at']
    curr_m_name = model_meta['name']
    model_stack = model_meta.get('model_stack', [])

    st.info(f"🛡️ **{serving_symbol} 正在服役的模型组**: `[{curr_m_data_version}] {curr_m_name}` (指纹: `{current_inf_version}`)")
    if model_stack:
        st.caption("权重编队: " + " | ".join(f"`{m['experiment_id']}` × {m['weight']:.1%}" for m in model_stack))

    st.markdown("###### 🧪 Live 子模型影子账本")
    st.caption("从 live_pipeline 真实推理流落库：每次实盘推理时，逐个记录子模型如果单独服役会发出的信号，并在预测周期结束后用真实后续价格打标盈亏。旧推理若此前没落库，无法完整补算。")
    shadow_df, shadow_db_path = load_live_shadow_model_audit(serving_symbol)
    if os.path.exists(shadow_db_path):
        shadow_mtime = datetime.fromtimestamp(os.path.getmtime(shadow_db_path)).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"影子账本: `{os.path.basename(shadow_db_path)}` | 更新时间: `{shadow_mtime}`")

    if shadow_df.empty:
        st.info("暂无 Live 子模型影子账本。重启 live_pipeline 后，将从下一次整点推理开始累积。")
    else:
        shadow_summary = summarize_live_shadow_model_audit(shadow_df, model_stack)
        if shadow_summary.empty:
            st.info("影子账本已有原始记录，但尚无可汇总的子模型数据。")
        else:
            display_shadow = shadow_summary.copy()
            display_shadow["Live权重"] = display_shadow["live_weight"].apply(lambda x: f"{x:.1%}")
            display_shadow["影子胜率"] = display_shadow["win_rate"].apply(lambda x: "待结算" if pd.isna(x) else f"{x:.1%}")
            display_shadow["累计影子ROE"] = display_shadow["total_pnl"].apply(lambda x: f"{x:+.2%}")
            display_shadow["平均单次ROE"] = display_shadow["avg_pnl"].apply(lambda x: "待结算" if pd.isna(x) else f"{x:+.2%}")
            display_shadow["最大回撤"] = display_shadow["max_drawdown"].apply(lambda x: "待结算" if pd.isna(x) else f"{x:.2%}")
            display_shadow["未加宏观胜率"] = display_shadow["raw_win_rate"].apply(lambda x: "待结算" if pd.isna(x) else f"{x:.1%}")
            display_shadow["未加宏观累计"] = display_shadow["raw_total_pnl"].apply(lambda x: f"{x:+.2%}")
            display_shadow["未加宏观回撤"] = display_shadow["raw_max_drawdown"].apply(lambda x: "待结算" if pd.isna(x) else f"{x:.2%}")
            display_shadow["最近信号"] = display_shadow.apply(
                lambda r: "多" if int(r["last_signal"]) == 1 else ("空" if int(r["last_signal"]) == -1 else "无"),
                axis=1,
            )
            display_shadow["最近概率"] = display_shadow.apply(
                lambda r: f"多{float(r['last_p_up']):.2%} / 空{float(r['last_p_down']):.2%}",
                axis=1,
            )
            st.dataframe(
                display_shadow[[
                    "experiment_id", "Live权重", "inference_count", "shadow_trade_count",
                    "settled_trade_count", "pending_trade_count", "影子胜率",
                    "累计影子ROE", "平均单次ROE", "最大回撤", "未加宏观胜率", "未加宏观累计", "未加宏观回撤",
                    "long_count", "short_count", "最近信号", "last_tier", "最近概率"
                ]],
                hide_index=True,
                width='stretch',
                column_config={
                    "experiment_id": "模型",
                    "inference_count": "推理数",
                    "shadow_trade_count": "影子发车",
                    "settled_trade_count": "已结算",
                    "pending_trade_count": "待结算",
                    "long_count": "多",
                    "short_count": "空",
                    "last_tier": "最近Tier",
                }
            )

            with st.expander("查看最近 100 条子模型影子推理明细", expanded=False):
                recent_shadow = shadow_df.head(100).copy()
                if not recent_shadow.empty:
                    recent_shadow["decision_time"] = pd.to_datetime(recent_shadow["decision_ts"], unit="ms", errors="coerce")
                    recent_shadow["方向"] = recent_shadow["adj_signal"].map({1: "多", -1: "空", 0: "无"})
                    recent_shadow["影子ROE"] = pd.to_numeric(recent_shadow["adj_pnl"], errors="coerce").apply(
                        lambda x: "待结算" if pd.isna(x) else f"{x:+.2%}"
                    )
                    st.dataframe(
                        recent_shadow[[
                            "decision_time", "experiment_id", "方向", "adj_tier",
                            "p_up", "p_down", "entry_price", "settle_price", "影子ROE"
                        ]],
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "decision_time": "推理时间",
                            "experiment_id": "模型",
                            "adj_tier": "Tier",
                            "p_up": st.column_config.NumberColumn("p_up", format="%.4f"),
                            "p_down": st.column_config.NumberColumn("p_down", format="%.4f"),
                            "entry_price": st.column_config.NumberColumn("入场参考价", format="$%.4f"),
                            "settle_price": st.column_config.NumberColumn("结算价", format="$%.4f"),
                        }
                    )

    if df_closed_trades.empty:
        st.info("💡 暂无平仓流水。")
    else:
        serving_closed_trades = df_closed_trades.copy()
        if 'symbol' in serving_closed_trades.columns:
            serving_closed_trades = serving_closed_trades[serving_closed_trades['symbol'] == serving_symbol].copy()

        version_series = serving_closed_trades.get('version', pd.Series(index=serving_closed_trades.index, dtype='object')).fillna("unknown")
        version_matched_df = serving_closed_trades[version_series == curr_m_data_version].copy()
        v_df_curr = version_matched_df[version_matched_df['exec_ts'] >= curr_m_created_at].copy()

        fallback_reason = ""
        if v_df_curr.empty:
            v_df_curr = version_matched_df.copy()
            if not v_df_curr.empty:
                fallback_reason = "当前主模型上线后的闭环样本为空，已回退为同血统历史样本。"
        if v_df_curr.empty:
            known_version_df = serving_closed_trades[version_series != "unknown"].copy()
            if not known_version_df.empty:
                latest_version = known_version_df.sort_values('exec_ts').iloc[-1]['version']
                v_df_curr = known_version_df[known_version_df['version'] == latest_version].copy()
                fallback_reason = f"当前主模型血统暂无闭环样本，已临时展示最近一个有实盘样本的模型血统 `{latest_version}`。"
        if v_df_curr.empty and (version_series == "unknown").any():
            v_df_curr = serving_closed_trades[version_series == "unknown"].copy()
            fallback_reason = "历史流水缺少模型血统字段，已退化展示未标注版本的实盘样本。"
        if v_df_curr.empty:
            version_mix = serving_closed_trades['version'].value_counts(dropna=False).to_dict() if 'version' in serving_closed_trades.columns else {}
            st.caption(f"⏳ 当前主模型血统 `{curr_m_data_version}` 暂无已完结的【实盘发车】平仓样本。现有历史样本分布: `{version_mix}`")
        else:
            if fallback_reason:
                st.caption(f"ℹ️ {fallback_reason}")
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown("###### 🎯 核心作战指标")
                v_win_rate_curr = (v_df_curr['real_roe'] > 0).mean()
                v_total_pnl_curr = v_df_curr['account_impact'].sum()
                v_total_usdt_curr = v_total_pnl_curr * initial_cap
                
                span_days = (v_df_curr['exec_ts'].max() - v_df_curr['exec_ts'].min()).total_seconds() / 86400.0
                safe_span = max(1.0, span_days)
                annual_roi = v_total_pnl_curr * (365.0 / safe_span)
                
                m1, m2 = st.columns(2)
                m1.metric("服役期间实盘胜率", f"{v_win_rate_curr:.1%}", delta=f"总发车 {len(v_df_curr)} 次")
                m2.metric("累计创造利润", f"{v_total_usdt_curr:+.2f} USDT", delta=f"账户净值折算 {v_total_pnl_curr:+.2%}", delta_color="normal" if v_total_usdt_curr > 0 else "inverse")
                
                m3, m4 = st.columns(2)
                m3.metric("平均持仓时间", f"{v_df_curr['hold_hours'].mean():.1f} 小时")
                if span_days > 0:
                    m4.metric("实盘理论年化 (预估)", f"{annual_roi:+.2%}", delta=f"基于 {span_days:.1f} 天外推", delta_color="normal" if annual_roi > 0 else "inverse")
                else:
                    m4.metric("实盘理论年化 (预估)", "计算中 ⏳", delta="跨度不足 24h")
                    
            with col2:
                st.markdown("###### ⚖️ AI 诚实度校准 (Confidence Calibration)")
                bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                v_df_cal = v_df_curr[pd.to_numeric(v_df_curr['xgb_prob'], errors='coerce').between(0.5, 1.0, inclusive='both')].copy()
                if v_df_cal.empty:
                    st.info("⏳ 当前批次平仓样本未记录有效置信度，暂时无法绘制校准诚实度柱状图。")
                else:
                    v_df_cal.loc[:, 'prob_bin'] = pd.cut(v_df_cal['xgb_prob'], bins=bins, include_lowest=True)
                    cal_stats = v_df_cal.groupby('prob_bin', observed=False)['real_roe'].apply(
                        lambda x: (x > 0).mean() if len(x) > 0 else 0
                    ).fillna(0)
                    
                    fig_cal = go.Figure(data=[
                        go.Bar(
                            x=[f"{b.left*100:.0f}%-{b.right*100:.0f}%" for b in cal_stats.index],
                            y=cal_stats.values, marker_color='#00FFAA',
                            text=[f"{v*100:.1f}%" for v in cal_stats.values], textposition='auto'
                        )
                    ])
                    fig_cal.update_layout(
                        height=240, margin=dict(l=0,r=0,t=10,b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(tickformat=".0%", range=[0, 1.1])
                    )
                    st.plotly_chart(fig_cal, width='stretch')
                    st.caption("*(图解：完美的 AI 应该在预期胜率更高的区间，展现出对应的高物理胜率，即**极佳的校准诚实度**)*")
# ==========================================
# 🚨 引擎底层警报与异常状态监控器 (挪至页末展示)
# ==========================================
@st.fragment(run_every=_run_every)
def render_bottom_alerts():
    st.markdown("### 🚨 引擎实时警报")
    # 👇 剥离 sidebar 定位，改为普通 st.expander
    with st.expander("查看底层巡逻兵异常日志", expanded=True):
        log_db_path = os.path.join(LIVE_DATA_DIR, "logs", "engine_system_logs.db")
        if os.path.exists(log_db_path):
            try:
                with sqlite3.connect(log_db_path, timeout=5.0) as conn:
                    # 👇 修改 1：把底层提取池扩大到 800 条，确保去重后还能剩下足够的日志
                    df_alerts = pd.read_sql("SELECT log_ts, level, symbol, message FROM system_alerts ORDER BY log_ts DESC LIMIT 500", conn)
                    
                    if not df_alerts.empty:
                        df_alerts['log_ts'] = pd.to_datetime(df_alerts['log_ts'], format='mixed')
                        
                        filtered_logs = []
                        seen_signatures = {}
                        
                        # 【改动2】：智能去重算法 (时间窗口：30分钟 / 1800秒)
                        for _, r in df_alerts.iterrows():
                            # 定义“重复”的指纹：同一个币，同一种级别，同一个报错内容
                            sig = f"{r['symbol']}_{r['level']}_{r['message']}"
                            current_ts = r['log_ts']
                            
                            if sig in seen_signatures:
                                # 因为是按时间倒序遍历，seen_signatures 存的是最新一次发生的时间
                                time_diff = seen_signatures[sig] - current_ts
                                if time_diff.total_seconds() < 1800:
                                    # 距离该错误最新一次发生不到半小时，视为“噪音刷屏”，直接丢弃旧的
                                    continue 
                                    
                            seen_signatures[sig] = current_ts
                            filtered_logs.append(r)
                            
                        # 【改动3】：紧凑型固定高度滚动框 (剔除臃肿的 st.error/warning)
                        if filtered_logs:
                            # 👇 修改 2：放开展示上限，展示前 200 条精华日志
                            display_logs = filtered_logs[:200] 
                            
                            # 👇 修改 3：稍微拉长一点容器的高度（比如从 350 改为 450），方便你在 200 条数据里舒展地滚动查看
                            with st.container(height=450):
                                for r in display_logs:
                                    ts_str = r['log_ts'].strftime('%m-%d %H:%M:%S')
                                    sym = r['symbol']
                                    lvl = r['level']
                                    msg = r['message']
                                    
                                    # 智能颜色映射，保留视觉警示力
                                    if "🚨" in lvl or "❌" in lvl:
                                        color = "#ff4b4b" # 血红色
                                    elif "⚠️" in lvl:
                                        color = "#ffa421" # 警告橙
                                    else:
                                        color = "#21c354" # 成功绿
                                        
                                    # 使用原生 HTML 渲染极简小字体，极大提升同屏信息密度
                                    st.markdown(
                                        f"<div style='font-size: 0.82em; margin-bottom: 8px; line-height: 1.35;'>"
                                        f"<span style='color: #888;'>[{ts_str}]</span> "
                                        f"<b>{sym}</b> "
                                        f"<span style='color: {color};'><b>{lvl}</b> {msg}</span>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.info("🟢 系统运行平稳，无警报。")
                    else:
                        st.info("🟢 系统运行平稳，无警报。")
            except Exception as e:
                st.info(f"🟢 警报数据库读取中: {e}")
        else:
            st.info("🟢 等待底层引擎生成警报库...")

# ---------------------------------------------------------
# 📋 核心风控参数来自 config.py（只读展示，如需修改请编辑 quant_config/config.py 并重启）
threshold_symbol = _normalize_symbol_key(
    st.session_state.get("ui_radar_target") or (ALL_SYMBOLS[0] if ALL_SYMBOLS else SYMBOL)
)
if threshold_symbol not in THRESHOLDS_BY_SYMBOL and ALL_SYMBOLS:
    threshold_symbol = ALL_SYMBOLS[0]
th = THRESHOLDS_BY_SYMBOL.get(threshold_symbol, {})
threshold_meta = THRESHOLD_SOURCE_BY_SYMBOL.get(threshold_symbol, {"experiment_id": EXPERIMENT_ID})
threshold_model_count = len(threshold_meta.get("models", [])) if isinstance(threshold_meta, dict) else 1
threshold_mode_label = "加权Ensemble" if threshold_model_count > 1 else "单模型"

st.sidebar.markdown(f"""
| 参数 | 当前值 |
|------|--------|
| 🎯 门槛标的 | `{threshold_symbol}` |
| 🧬 门槛模型 | `{threshold_mode_label} x{threshold_model_count}` |
| 🏦 初始资本 | `{INITIAL_CAPITAL:,.0f} USDT` |
|  单笔风险预算 | `{BASE_RISK_BUDGET:.1%}` |
| 🟢 K线物理止盈 | `{HARD_TP*100:.1f}%` |
| 🔴 K线物理止损 | `{HARD_SL*100:.1f}%` |
| 📍 追踪ATR回撤 | `ATR({TRAILING_ATR_PERIOD}) × {TRAILING_ATR_MULTI}` |
| ⚡ 瀑布ATR乘数 | `{BAILOUT_ATR_MULTI}x` |
| 📊 爆量乘数 | `{BAILOUT_VOL_MULTI}x` |
| **📈 做多门槛 (Tier 1/2/3)** | `{th.get('up_top_10', 0):.2%} / {th.get('up_top_5', 0):.2%} / {th.get('up_top_1', 0):.2%}` |
| **📉 做空门槛 (Tier 1/2/3)** | `{th.get('down_top_10', 0):.2%} / {th.get('down_top_5', 0):.2%} / {th.get('down_top_1', 0):.2%}` |
""")
if not th:
    st.sidebar.warning(f"未加载到 {threshold_symbol} 的概率门槛文件，请检查 config.py 中的 experiment_id。")
with st.sidebar.expander("🧬 模型权重/门槛来源", expanded=False):
    threshold_models = threshold_meta.get("models", []) if isinstance(threshold_meta, dict) else []
    if threshold_models:
        st.dataframe(
            pd.DataFrame([
                {
                    "模型": m.get("experiment_id", "unknown"),
                    "权重": f"{float(m.get('weight', 0.0)):.2%}",
                    "阈值": "已加载" if m.get("loaded") else "缺失",
                }
                for m in threshold_models
            ]),
            hide_index=True,
            width='stretch',
        )
    else:
        st.caption(f"`{threshold_meta.get('experiment_id', EXPERIMENT_ID)}`")

st.sidebar.divider()

# ==========================================
# 📡 实时全市场扫描雷达 (读写分离：数据同步自后台引擎)
# ==========================================
st.markdown("---")
st.markdown("### 📡 实时全市场扫描雷达")

@st.fragment(run_every=_run_every)
def render_market_radar():
    # 👇 从 session_state 动态读取当前存活的币种池
    active_radar_symbols = ALL_SYMBOLS
    if isinstance(active_radar_symbols, str):
        active_radar_symbols = [active_radar_symbols]
    
    if not active_radar_symbols:
        st.warning("暂无激活的监控币种。")
        return

    # ==========================================
    # 🎯 【核心重构：弃用 st.tabs，改用下拉狙击模式】
    # ==========================================
    col_sel, col_blank = st.columns([1, 3])
    with col_sel:
        target_symbol = st.selectbox(
            "🎯 选择要深度透视的雷达目标", 
            options=active_radar_symbols,
            key="ui_radar_target"
        )

    try:
        # 🎯 策略：优先从极速网关 API 获取最新的内存切片 (最快)
        snap_data = {}
        try:
            # 去除符号中的斜杠以匹配网关可能的格式 (如 BTC/USDT -> BTCUSDT)
            api_sym = target_symbol.replace("/", "")
            api_url = f"http://127.0.0.1:8000/api/snapshot/{api_sym}"
            resp = requests.get(api_url, timeout=1.0)
            if resp.status_code == 200:
                snap_data = resp.json()
        except:
            pass

        current_price = snap_data.get("current_price", 0.0)
        ohlcv_1h = snap_data.get("ohlcv_1h", [])
        
        # 🛡️ 容错处理：如果 API 挂了或没数据，直接穿透到本地 SQLite 硬盘存储 (兜底)
        if not ohlcv_1h or current_price == 0:
            db_path = os.path.join(LIVE_DATA_DIR, "gateway_state.db")
            if os.path.exists(db_path):
                try:
                    with sqlite3.connect(db_path, timeout=5.0) as conn:
                        # 查找该币种最近的 168 根 1h K 线 (一周)
                        df_db = pd.read_sql(
                            f"SELECT timestamp as ts, open, high, low, close, volume FROM memory_1h WHERE symbol = '{target_symbol}' ORDER BY timestamp DESC LIMIT 168", 
                            conn
                        )
                        if not df_db.empty:
                            df_db = df_db.sort_values("ts")
                            ohlcv_1h = df_db.values.tolist()
                            current_price = df_db['close'].iloc[-1]
                except:
                    pass

        # 转换 1h K 线为 DataFrame
        df_actual = pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"])
        if not df_actual.empty:
            # 兼容性处理：API 返回的是 list，SQLite 返回的列名已对齐
            df_actual['ts'] = pd.to_datetime(df_actual['ts'], unit='ms')
            # 🎯 仅保留最近一周 (168 小时) 的数据，防止图表过于拥挤
            df_actual = df_actual.tail(168)
            
        # 2. 从 live_status.json 读取 AI 推理状态 (适配全新的嵌套结构树)
        status_file = os.path.join(LIVE_DATA_DIR, "live_status.json")
        status_data = {}
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    status_data = json.load(f)
            except: pass
        status_data = (status_data.get("symbols", {}) or {}).get(target_symbol, status_data)
            
        # 填充展示指标 (优先使用根节点的 timestamp，适配新 Payload)
        last_update_ts = status_data.get("timestamp", time.time())
        last_update = datetime.fromtimestamp(last_update_ts).strftime("%Y-%m-%d %H:%M:%S")
        
        macro_score = status_data.get("macro_score", 0.0)
        
        # 安全提取嵌套的 model_inference 数据
        inference_data = status_data.get("model_inference", {})
        configured_model_stack = resolve_current_serving_model(target_symbol).get("model_stack", [])
        
        p_up = inference_data.get("raw_prob_up", 0.33)
        p_down = inference_data.get("raw_prob_down", 0.33)
        p_flat = inference_data.get("raw_prob_flat", max(0.0, 1.0 - p_up - p_down))
        configured_model_count = len(configured_model_stack)
        model_count = max(int(inference_data.get("model_count", 1) or 1), configured_model_count or 1)
        ensemble_models = inference_data.get("ensemble_models", []) or []
        if not ensemble_models and configured_model_stack:
            ensemble_models = [
                {
                    "experiment_id": m.get("experiment_id", "unknown"),
                    "weight": float(m.get("weight", 0.0)),
                    "prob_up": np.nan,
                    "prob_flat": np.nan,
                    "prob_down": np.nan,
                    "status": "等待推理",
                }
                for m in configured_model_stack
            ]
        
        adj_p_up = inference_data.get("adj_prob_up_tier1", p_up)
        adj_p_down = inference_data.get("adj_prob_down_tier1", p_down)
        
        is_uptrend_proxy = current_price > (df_actual['close'].mean() if not df_actual.empty else current_price)
        ema_macro_slope = 0.001 if is_uptrend_proxy else -0.001
        
        up_prob = adj_p_up if adj_p_up > adj_p_down else (1.0 - adj_p_down)
        auto_leverage = 8.0  # 默认保底杠杆率 8x
        tp_pct = 0.08
        sl_pct = 0.04
        
        st.caption(f"⏱️ 引擎状态最后同步: `{last_update}` (基于 {target_symbol} 极速快照)")
        
        # ==========================================
        # 🧠 第一行：iTransformer 大模型预测概率
        # ==========================================
        model_title = f"iTransformer 加权多模型概率分布 x{model_count}" if model_count > 1 else "iTransformer 大模型概率分布"
        st.markdown(f"##### {model_title}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.metric("当前标的价格", f"${current_price:,.4f}")
        c2.metric("宏观情绪加成", f"{macro_score:+.2f}", delta_color="normal" if macro_score != 0 else "off")
        c3.metric("看涨概率 (p_up)", f"{adj_p_up:.2%}", f"原始: {p_up:.2%}")
        c4.metric("震荡概率 (p_flat)", f"{p_flat:.2%}")
        c5.metric("看跌概率 (p_down)", f"{adj_p_down:.2%}", f"原始: {p_down:.2%}")
        c6.metric("模型票仓", f"{model_count} 个", "加权平均" if model_count > 1 else "单模型")

        if ensemble_models:
            with st.expander("查看子模型概率与权重", expanded=model_count > 1):
                model_df = pd.DataFrame([
                    {
                        "模型": m.get("experiment_id", m.get("name", "unknown")),
                        "权重": float(m.get("weight", 0.0)) * 100,
                        "看涨": np.nan if pd.isna(m.get("prob_up", np.nan)) else float(m.get("prob_up", 0.0)) * 100,
                        "震荡": np.nan if pd.isna(m.get("prob_flat", np.nan)) else float(m.get("prob_flat", 0.0)) * 100,
                        "看跌": np.nan if pd.isna(m.get("prob_down", np.nan)) else float(m.get("prob_down", 0.0)) * 100,
                        "状态": m.get("status", "已推理"),
                    }
                    for m in ensemble_models
                ])
                st.dataframe(
                    model_df,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "权重": st.column_config.ProgressColumn("权重", format="%.1f%%", min_value=0.0, max_value=100.0),
                        "看涨": st.column_config.ProgressColumn("看涨", format="%.1f%%", min_value=0.0, max_value=100.0),
                        "震荡": st.column_config.ProgressColumn("震荡", format="%.1f%%", min_value=0.0, max_value=100.0),
                        "看跌": st.column_config.ProgressColumn("看跌", format="%.1f%%", min_value=0.0, max_value=100.0),
                    }
                )

        # --- 绘制极简趋势预测图 ---
        fig = go.Figure()
        if not df_actual.empty and 'open' in df_actual.columns:
            fig.add_trace(go.Candlestick(x=df_actual['ts'], open=df_actual['open'], high=df_actual['high'], low=df_actual['low'], close=df_actual['close'], name="实时1h K 线"))
        
        fig.update_layout(title=f"{target_symbol} 实时盘口K线与推演", yaxis_title="Price (USDT)", 
                        template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"渲染 {target_symbol} 雷达时出错: {e}")
render_market_radar()

# ==========================================
# 🚨 引擎底层警报与异常状态监控器 (挪至页末展示)
# ==========================================
st.markdown("---")
render_bottom_alerts()
