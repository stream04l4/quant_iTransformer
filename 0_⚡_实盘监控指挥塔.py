import os
import json
import time
import psutil
import pynvml
import requests
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from quant_config.config import TRACKED_SYMBOLS, DATA_DIR

# ==========================================
# ⚡ 页面基础配置
# ==========================================
st.set_page_config(page_title="TimesFM 监控塔", layout="wide", page_icon="⚡")

# ==========================================
# 🔐 安全认证拦截器 (保持原有的防爆破登录)
# ==========================================
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@yourquant.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "123456") 
SECRET_TOKEN = os.environ.get("SECRET_TOKEN", "token_888")

def check_auth():
    if st.session_state.get("auth_success", False) or st.query_params.get("auth") == SECRET_TOKEN:
        st.session_state["auth_success"] = True
        return True

    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>🛡️ TimesFM 实盘指挥塔受限</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("📧 管理员邮箱")
            pwd = st.text_input("🔑 访问密码", type="password")
            submitted = st.form_submit_button("解锁系统 🚀", use_container_width=True)
            if submitted:
                if email == ADMIN_EMAIL and pwd == ADMIN_PASSWORD:
                    st.session_state["auth_success"] = True
                    st.query_params["auth"] = SECRET_TOKEN
                    st.rerun()
                else:
                    st.error("❌ 身份核验失败！")
    return False

if not check_auth():
    st.stop()

# ==========================================
# 🎛️ 侧边栏：微服务总控与硬件监控
# ==========================================
st.sidebar.markdown("### ⚡ 系统总控台")
col_tog, col_val = st.sidebar.columns([1.2, 1])
with col_tog:
    enable_refresh = st.toggle("🔄 自动刷新", value=True)
with col_val:
    refresh_interval = st.number_input("频率(s)", min_value=3, max_value=60, value=3, step=1, label_visibility="collapsed") if enable_refresh else 0

_run_every = f"{refresh_interval}s" if enable_refresh and refresh_interval > 0 else None

@st.fragment(run_every=_run_every)
def render_status_sidebar():
    # 1. 探活：极速网关
    ws_alive = False
    try:
        requests.get(f"http://127.0.0.1:8000/api/snapshot/{TRACKED_SYMBOLS[0].replace('/', '_')}", timeout=0.3)
        ws_alive = True
    except Exception: pass

    # 2. 探活：OMS 供货线程 (通过查 JSON 更新时间)
    oms_alive = False
    state_file = os.path.join(DATA_DIR, "portfolio_state.json")
    if os.path.exists(state_file):
        if time.time() - os.path.getmtime(state_file) < 15:
            oms_alive = True

    # 3. 探活：熔断右脑 (看系统警报库最新写入时间)
    sys_db = os.path.join(DATA_DIR, "engine_system_logs.db")
    risk_brain_alive = os.path.exists(sys_db) and (time.time() - os.path.getmtime(sys_db) < 60) # 只要库还在更新就行

    # --- 渲染微服务指示灯 ---
    st.sidebar.caption("👇 微服务存活状态")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1: st.status("📡 网关" if ws_alive else "🔴 网关", state="complete" if ws_alive else "error")
    with c2: st.status("🏦 OMS" if oms_alive else "🔴 OMS", state="complete" if oms_alive else "error")
    with c3: st.status("🧠 左脑", state="complete") # 预测引擎在控制台，默认放行

    # --- 硬件探针 ---
    gpu_pct, cpu_pct, ram_pct = 0.0, 0.0, 0.0
    gpu_text, ram_text = "N/A", "N/A"
    try:
        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        gpu_pct = info.used / info.total
        gpu_text = f"{info.used / 1024**3:.1f} / {info.total / 1024**3:.1f} GB"
    except: pass
    
    try:
        cpu_pct = psutil.cpu_percent() / 100.0
        mem_info = psutil.virtual_memory()
        ram_pct = mem_info.percent / 100.0
        ram_text = f"{mem_info.used / 1024**3:.1f} GB"
    except: pass

    st.sidebar.divider()
    st.sidebar.caption(f"💾 L4 显存占用: {gpu_text}") 
    st.sidebar.progress(min(gpu_pct, 1.0))              
    st.sidebar.caption(f"⚙️ CPU 核心负载: {cpu_pct * 100:.1f}%")
    st.sidebar.progress(min(max(cpu_pct, 0.0), 1.0))
    st.sidebar.caption(f"🧠 系统内存: {ram_text}")
    st.sidebar.progress(min(ram_pct, 1.0))

with st.sidebar:
    render_status_sidebar()

# ==========================================
# ⚔️ 页面主干：四大高频数据看板
# ==========================================
st.title("⚡ TimesFM 实盘高频监控塔")

# --- 模块 1：账户核心资产与实时仓位 ---
@st.fragment(run_every=_run_every)
def render_portfolio():
    state_file = os.path.join(DATA_DIR, "portfolio_state.json")
    if not os.path.exists(state_file):
        st.info("⏳ 等待 OMS 同步资产快照...")
        return
        
    try:
        with open(state_file, 'r') as f:
            p_state = json.load(f)
            
        wallet = float(p_state.get('balance', {}).get('total', {}).get('USDT', 0))
        free = float(p_state.get('balance', {}).get('free', {}).get('USDT', 0))
        positions = [p for p in p_state.get('positions', []) if float(p.get('contracts', 0)) > 0]
        
        # 顶部 KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("💰 账户总净值 (USDT)", f"${wallet:,.2f}")
        kpi2.metric("🛡️ 可用空闲资金", f"${free:,.2f}")
        kpi3.metric("⚔️ 当前暴露兵力", f"{len(positions)} 个仓位")
        
        total_upnl = sum(float(p['info'].get('unrealised_pnl', 0)) for p in positions)
        kpi4.metric("📊 全局未实现盈亏", f"${total_upnl:+,.2f}", delta_color="normal" if total_upnl >= 0 else "inverse")

        # 仓位明细表
        if positions:
            pos_data = []
            for p in positions:
                info = p['info']
                side = "🟢 做多" if str(p.get('side', '')).upper() == 'LONG' else "🔴 做空"
                leverage = float(info.get('lever', info.get('leverage', '1')))
                entry_px = float(info.get('avgPx', info.get('entry_price', 0)))
                mark_px = float(info.get('markPx', info.get('mark_price', 0)))
                upnl = float(info.get('unrealised_pnl', 0))
                
                pos_data.append({
                    "标的": p['symbol'],
                    "方向": side,
                    "杠杆": f"{leverage}x",
                    "开仓均价": f"${entry_px:.4f}",
                    "当前标记价": f"${mark_px:.4f}",
                    "未实现 PnL": f"${upnl:+.2f}"
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"解析资产文件失败: {e}")

# --- 模块 2 & 3：底层日志与熔断记录 ---
@st.fragment(run_every=_run_every)
def render_logs_and_trades():
    c_log, c_trade = st.columns([1, 1])
    
    # 🔴 右脑预警与 OMS 动作
    with c_log:
        st.markdown("### 🚨 系统警报与熔断雷达")
        sys_db = os.path.join(DATA_DIR, "engine_system_logs.db")
        if os.path.exists(sys_db):
            try:
                with sqlite3.connect(sys_db) as conn:
                    df_alerts = pd.read_sql("SELECT log_ts as '时间', level as '级别', symbol as '标的', message as '情报详情' FROM system_alerts ORDER BY log_ts DESC LIMIT 20", conn)
                st.dataframe(df_alerts, hide_index=True, use_container_width=True, height=350)
            except: pass
        else:
            st.caption("暂无预警日志。")

    # 🟢 实际落单成交历史
    with c_trade:
        st.markdown("### 📜 最近 20 笔真实成交记录")
        trade_db = os.path.join(DATA_DIR, "trading_intelligence_BTC_USDT.db") # 如果你跑的别的币，这里可能需要改成动态搜寻
        if os.path.exists(trade_db):
            try:
                with sqlite3.connect(trade_db) as conn:
                    # 获取表结构确认是否有 leverage 字段
                    has_leverage = 'leverage' in [col[1] for col in conn.execute("PRAGMA table_info(live_trades)")]
                    select_cols = "exec_ts as '执行时间', side as '动作', fill_price as '成交价', margin_used as '耗用本金(U)'"
                    if has_leverage: select_cols += ", leverage as '杠杆'"
                    
                    df_trades = pd.read_sql(f"SELECT {select_cols} FROM live_trades ORDER BY exec_ts DESC LIMIT 20", conn)
                    # 格式化动作图标
                    df_trades['动作'] = df_trades['动作'].apply(lambda x: f"🛡️ {x}" if "bailout" in str(x).lower() else x)
                    
                st.dataframe(df_trades, hide_index=True, use_container_width=True, height=350)
            except Exception as e: 
                st.error(f"读取流水失败: {e}")
        else:
            st.caption("暂无交易流水。")

render_portfolio()
st.divider()
render_logs_and_trades()