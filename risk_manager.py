import os
import time
import sqlite3
import math
import pandas as pd
import requests
import json
from datetime import datetime
import ccxt
from state_manager import state
from execution_oms import build_gate_close_stop_initial, get_gate_stop_coverage_size, log_system_alert
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR, MODEL_DIR,
    HARD_TP, HARD_SL, HOLD_PERIOD, 
    BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, FORECAST_HORIZON,
    BAILOUT_MIN_PHYSICAL_DROP, BAILOUT_OBI_THRESH, EXPERIMENT_ID,
    MACRO_ALPHA_TOP_10, MACRO_ALPHA_TOP_5, MACRO_ALPHA_TOP_1,
    TRAILING_ATR_PERIOD, TRAILING_ATR_MULTI
)
from dotenv import load_dotenv
load_dotenv()

MY_GATE_API_KEY  = os.environ.get("GATE_DEMO_API_KEY", "")
MY_GATE_SECRET   = os.environ.get("GATE_DEMO_SECRET", "")
JSON_PATH = MODEL_DIR / f"dynamic_thresholds_{EXPERIMENT_ID}.json"


def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _inspect_snapshot_stop_coverage(position):
    """基于供货线程快照检查当前仓位的止损覆盖情况。"""
    info = position.get("info", {}) or {}
    try:
        expected_contracts = abs(float(info.get("size", position.get("contracts", 0)) or 0.0))
    except (TypeError, ValueError):
        expected_contracts = 0.0

    snapshot_stops = list(state.portfolio_memory.get("open_stops", []) or [])
    sym_key = _normalize_symbol_key(position.get("symbol"))
    matched_stops = [o for o in snapshot_stops if _normalize_symbol_key(o.get("symbol")) == sym_key]

    if expected_contracts <= 0:
        return {
            "status": "no_position",
            "expected_contracts": 0.0,
            "max_stop_size": 0.0,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": 1.0,
            "reason": "",
        }

    if not matched_stops:
        return {
            "status": "missing_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": 0.0,
            "matched_stop_count": 0,
            "coverage_ratio": 0.0,
            "reason": "交易所快照中未发现该仓位的活动止损单",
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
            "max_stop_size": 0.0,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": 0.0,
            "reason": "止损单存在，但无法解析其覆盖张数",
        }

    max_stop_size = max(parsed_sizes)
    coverage_ratio = max_stop_size / (expected_contracts + 1e-9)
    if coverage_ratio < 0.9:
        return {
            "status": "undercovered_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": max_stop_size,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": coverage_ratio,
            "reason": f"止损单覆盖不足，size={max_stop_size:.4f} < 持仓={expected_contracts:.4f}",
        }

    return {
        "status": "ok",
        "expected_contracts": expected_contracts,
        "max_stop_size": max_stop_size,
        "matched_stop_count": len(matched_stops),
        "coverage_ratio": coverage_ratio,
        "reason": "",
    }


def _calc_recent_atr(ohlcv, period, current_close=None):
    if not ohlcv or len(ohlcv) < 2:
        return 0.0

    recent = ohlcv[-(period + 1):]
    tr_list = []
    for i in range(1, len(recent)):
        prev_close = float(recent[i - 1][4])
        curr_h = float(recent[i][2])
        curr_l = float(recent[i][3])
        if current_close is not None and i == len(recent) - 1:
            curr_h = max(curr_h, float(current_close))
            curr_l = min(curr_l, float(current_close))
        tr = max(curr_h - curr_l, abs(curr_h - prev_close), abs(curr_l - prev_close))
        tr_list.append(tr)

    return sum(tr_list) / len(tr_list) if tr_list else 0.0


def _safe_parse_epoch(raw_value):
    """兼容秒/毫秒时间戳，解析失败返回 None。"""
    try:
        ts = float(raw_value)
    except (TypeError, ValueError):
        return None

    if ts <= 0:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return ts


def _resolve_position_entry_time(position, base_sym):
    """
    优先使用交易所仓位对象里的 open_time/timestamp，避免新仓尚未写库时误拿到上一笔老仓时间。
    仅当交易所字段缺失时，才退回本地最新同向开仓流水。
    """
    info = position.get("info", {}) or {}
    candidates = [
        info.get("open_time"),
        position.get("timestamp"),
        position.get("lastUpdateTimestamp"),
        info.get("update_time"),
    ]
    for raw_ts in candidates:
        parsed_ts = _safe_parse_epoch(raw_ts)
        if parsed_ts is not None:
            return parsed_ts

    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
    pos_side = str(position.get("side", "")).upper()
    local_open_side = "buy" if pos_side == "LONG" else "sell"
    try:
        with sqlite3.connect(db_name, timeout=1.0) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT exec_ts
                FROM live_trades
                WHERE side = ?
                ORDER BY exec_ts DESC
                LIMIT 1
                """,
                (local_open_side,),
            )
            row = cur.fetchone()
            if row:
                return datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()
    except Exception:
        pass

    return time.time()


def _apply_logit_fusion(base_prob, macro_score, alpha):
    """复刻主引擎的概率融合逻辑，供右脑识别同向信号刷新持仓计时。"""
    epsilon = 1e-7
    p_safe = min(max(float(base_prob or 0.0), epsilon), 1.0 - epsilon)
    logit = math.log(p_safe / (1.0 - p_safe))
    adjusted_logit = logit + (float(alpha) * float(macro_score or 0.0))
    return 1.0 / (1.0 + math.exp(-adjusted_logit))


def _resolve_target_signal_from_status(live_status, thresholds, base_sym):
    """
    基于 live_status 里的最新模型输出，复算主引擎本小时的目标信号方向。
    这里返回 target_signal，而不是 actual_signal，因为同向仓位下被 OMS 放弃的信号
    仍然应该刷新 12 小时持仓计时。
    """
    if _normalize_symbol_key(live_status.get("symbol")) != base_sym:
        return 0

    inference_data = live_status.get("model_inference", {}) or {}
    raw_p_up = float(inference_data.get("raw_prob_up", 0.0) or 0.0)
    raw_p_down = float(inference_data.get("raw_prob_down", 0.0) or 0.0)
    macro_score = float(
        inference_data.get("macro_score", live_status.get("macro_score", 0.0)) or 0.0
    )

    adj_up_top1 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_1)
    adj_up_top5 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_5)
    adj_up_top10 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_10)
    adj_down_top1 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_1)
    adj_down_top5 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_5)
    adj_down_top10 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_10)

    if adj_up_top1 > thresholds.get('up_top_1', 1.0):
        return 1
    if adj_up_top5 > thresholds.get('up_top_5', 1.0):
        return 1
    if adj_up_top10 > thresholds.get('up_top_10', 1.0):
        return 1
    if adj_down_top1 > thresholds.get('down_top_1', 1.0):
        return -1
    if adj_down_top5 > thresholds.get('down_top_5', 1.0):
        return -1
    if adj_down_top10 > thresholds.get('down_top_10', 1.0):
        return -1
    return 0


def _refresh_entry_time_on_same_direction_signal(mem, base_sym, pos_side, live_status, thresholds):
    """
    若本小时出现与当前仓位同向的 target_signal，即使 OMS 因已有仓位未再次下单，
    也把 entry_time 锚点刷新到该小时，避免 HOLD_PERIOD 仍从最初建仓时刻累计。
    """
    execution_state = live_status.get("execution_state", {}) or {}
    inferred_hour_ts = _safe_parse_epoch(execution_state.get("last_inferred_hour"))
    if inferred_hour_ts is None:
        return

    target_signal = _resolve_target_signal_from_status(live_status, thresholds, base_sym)
    current_side = 1 if str(pos_side).upper() == "LONG" else -1
    if target_signal != current_side:
        return

    last_refresh_ts = float(mem.get('last_same_signal_refresh_ts', 0.0) or 0.0)
    if inferred_hour_ts <= last_refresh_ts:
        return
    if inferred_hour_ts <= float(mem.get('entry_time', 0.0) or 0.0):
        return

    mem['entry_time'] = inferred_hour_ts
    mem['last_same_signal_refresh_ts'] = inferred_hour_ts
    print(
        f"🕒 [{base_sym}] 侦测到同向信号延续，持仓计时已刷新至 "
        f"{datetime.fromtimestamp(inferred_hour_ts).strftime('%Y-%m-%d %H:%M:%S')}。"
    )


def _read_last_inferred_hour():
    """读取主引擎最近一次使用的闭合 1h K 线时间戳，供强制退出后做本K线冷却。"""
    status_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if not os.path.exists(status_path):
        return 0

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return int(payload.get("execution_state", {}).get("last_inferred_hour", 0) or 0)
    except Exception:
        return 0


def _register_forced_exit_lock(base_sym, reason, anchor_closed_ts=None):
    """
    强制退出后锁住当前这根已闭合 1h K 线。
    只有等到下一根新 K 线闭合，主引擎才允许再次入场。
    """
    if anchor_closed_ts is None:
        anchor_closed_ts = _read_last_inferred_hour()
    try:
        anchor_closed_ts = int(anchor_closed_ts or 0)
    except (TypeError, ValueError):
        anchor_closed_ts = 0

    if anchor_closed_ts <= 0:
        current_closed_hour = max(int(time.time() // 3600) - 1, 0)
        anchor_closed_ts = current_closed_hour * 3600000

    with state.event_lock:
        state.forced_exit_locks[base_sym] = {
            "anchor_closed_ts": anchor_closed_ts,
            "trigger_ts": time.time(),
            "reason": str(reason),
        }


def execute_bailout(ex, base_sym, p, reason, current_price, anchor_closed_ts=None):
    """右脑专用：执行紧急市价逃生"""
    gate_sym = p['symbol']
    pos_side = str(p.get('side', '')).upper()
    contracts = float(p.get('contracts', 0))
    entry_px = float(p.get('entryPrice', p['info'].get('entry_price', p['info'].get('avgPx', 0))))
    
    print(f"\n🚨 [右脑介入] 侦测到 {base_sym} {reason}！执行市价核弹清仓 (数量: {contracts})...")
    try:
        # 1. 越权下发市价单
        cl_side = 'sell' if pos_side == 'LONG' else 'buy'
        try:
            ex.create_order(symbol=gate_sym, type='market', side=cl_side, amount=contracts, params={'reduceOnly': True})
            print(f"   ✅ 交易所市价平仓指令已下发 ({gate_sym})")
        except Exception as e:
            print(f"   ⚠️ 交易所下单接口报错: {e}")
            # 即便下单报错也尝试继续执行落库，可能仓位已经由于 OCO 先一步平掉了
        
        # 2. 扫清残余 OCO
        try:
            for o in ex.fetch_open_orders(gate_sym, params={'stop': True}):
                try: ex.cancel_order(o['id'], gate_sym, params={'stop': True})
                except: pass
        except: pass
        
        # 3. 落库：历史本纪 (增加写入保障)
        db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
        with sqlite3.connect(db_name, timeout=30.0, check_same_thread=False) as conn_log:
            cursor = conn_log.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                              (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                               amount REAL, fill_price REAL, slippage REAL,
                               tp_price REAL, sl_price REAL, margin_used REAL, leverage REAL)''')
            
            # 记录一笔平仓流水
            cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'bailout_panic_sell', 'bailout_close',
                            float(contracts), current_price, 0.0, 0.0, 0.0, 0.0, 0.0))
            conn_log.commit()
            print(f"   💾 平仓记录已同步至本地数据库: {db_name}")

        _register_forced_exit_lock(base_sym, reason, anchor_closed_ts=anchor_closed_ts)
                    
        # 5. 警报与推送
        log_system_alert("🆘 熔断跳伞", base_sym, f"微观崩塌 ({reason})，已市价斩仓逃生。")
        
        pos_lev = float(p.get('leverage', p['info'].get('leverage', p['info'].get('lever', 1))))
        if pos_lev <= 0: pos_lev = 10.0
        
        raw_pnl = (current_price - entry_px) / entry_px if pos_side == 'LONG' else (entry_px - current_price) / entry_px
        realized_roe = raw_pnl * pos_lev
        contract_size = float(p.get('contractSize', 1))
        abs_profit = raw_pnl * (contracts * contract_size * entry_px)
        
        PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")
        push_title = f"🆘 [高频熔断] {base_sym} 微观结构雪崩！"
        dir_str = "做多 🟢" if pos_side == 'LONG' else "做空 🔴"
        pnl_icon = "🟢 逃生结算" if abs_profit > 0 else "🔴 截断亏损"
        
        push_content = (
            f"<b>🔹 交易标的：</b>{base_sym} ({dir_str})<br>"
            f"<b>🔸 触发警报：</b>{reason} (右脑紧急接管)<br>"
            f"<b>🎯 逃生均价：</b>${current_price:.4f}<br>"
            f"<b>{pnl_icon}：</b>{abs_profit:+.2f} USDT ({realized_roe:+.2%})"
        )
        state.push_queue.put({"token": PUSHPLUS_TOKEN, "title": push_title, "content": push_content, "template": "html"})
        state.recent_pushes[base_sym] = time.time()
        
    except Exception as e:
        print(f"   ⚠️ 右脑熔断清仓失败: {e}")
        log_system_alert("❌ 熔断异常", base_sym, f"跳伞执行失败: {e}")

def update_exchange_sl(ex, gate_symbol, pos_side, new_sl_price, pos_size):
    """向交易所发送 API：先挂载新追踪止损，成功后再撤销旧追踪，杜绝裸奔窗口"""
    exact_size = int(round(float(pos_size)))
    if exact_size <= 0:
        return False
        
    try:
        # 1. Gate.io 官方 API 规则对齐：多单止损需跌破触发(rule 2)
        rule = 2 if pos_side == 'LONG' else 1
        if not getattr(ex, "markets", None) or gate_symbol not in ex.markets:
            ex.load_markets(True)
        native_symbol = ex.market_id(gate_symbol)
        
        # 🚨 修复 3：生成带有精确时间戳的唯一标识符
        unique_sl_text = f"tsl_{int(time.time()*1000)}"
        
        # 2. 先建立新防线
        ex.private_futures_post_settle_price_orders({
            "settle": "usdt",
            "initial": build_gate_close_stop_initial(native_symbol, unique_sl_text, pos_side),
            "trigger": {
                "strategy_type": 0, 
                "price_type": 1, 
                "price": str(ex.price_to_precision(gate_symbol, new_sl_price)), 
                "rule": rule
            }
        })

        # 3. 新单挂载成功后，立刻拉取现有止损单，撤除所有不是该 unique_sl_text 的单子
        time.sleep(0.5) 
        stops = ex.fetch_open_orders(gate_symbol, params={'stop': True})
        for o in stops:
            t = str(o.get('clientOrderId', '') or o['info'].get('initial', {}).get('text', '')).lower()
            if t != unique_sl_text:
                try: 
                    ex.cancel_order(o['id'], gate_symbol, params={'stop': True})
                except Exception: 
                    pass
                    
        return True
    except Exception as e:
        print(f"   ⚠️ 交易所追踪止损单上移请求失败 (保留原防线兜底): {e}")
        return False

def bailout_worker():
    """右脑主循环：双轨制 (物理高水位状态机 + ATR 主动熔断)"""
    print("🛡️ [右脑] 高频双轨制风控已启动 (追踪止盈 OCO + ATR 主动熔断)...")
    bailout_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    bailout_ex.set_sandbox_mode(True)
    try:
        bailout_ex.load_markets(True)
        print("🔄 [右脑] Gate 市场静态规则已预加载。")
    except Exception as e:
        print(f"⚠️ [右脑] 启动预加载市场规则失败，将在补挂时重试: {e}")
    
    # 🧠 物理水位状态机内存池
    tracking_mem = {}
    bailout_cooldown_locks = {}
    last_worker_error_ts = 0.0
    
    # 加载回测同款概率阈值
    th = {}
    try:
        with open(JSON_PATH, 'r') as f:
            config_data = json.load(f)
            # 🚨 修复 2：解包带有 metadata 信息的工业级 JSON 格式
            th = config_data.get("thresholds", config_data)
    except Exception as e:
        print(f"❌ 右脑加载阈值文件失败，请检查模型编号: {e}")

    while True:
        try:
            with state.pos_lock:
                current_positions = list(state.active_pos_map.values())
                
            if not current_positions:
                tracking_mem.clear()
                # 清理 UI 显示文件
                if os.path.exists(os.path.join(LIVE_DATA_DIR, "chandelier_state.json")):
                    try: os.remove(os.path.join(LIVE_DATA_DIR, "chandelier_state.json"))
                    except: pass
                time.sleep(1)
                continue
            
            # 读取最新 AI 概率与前端硬参数
            live_status = {}
            if os.path.exists(os.path.join(LIVE_DATA_DIR, "live_status.json")):
                try:
                    with open(os.path.join(LIVE_DATA_DIR, "live_status.json"), "r") as f:
                        live_status = json.load(f)
                except: pass
                
            # 🚨 修复 1：读取模型最原始的输出概率，切断字段命名错位
            inference_data = live_status.get("model_inference", {})
            raw_p_up = inference_data.get("raw_prob_up", 0.0)
            raw_p_down = inference_data.get("raw_prob_down", 0.0)

            hard_tp = HARD_TP   # K线物理止盈幅度（直接来自 config）
            hard_sl = HARD_SL   # K线物理底部止损幅度（直接来自 config）
            
            export_ui_mem = {}
            
            import concurrent.futures
            def fetch_snap(p):
                base_sym = p['symbol'].split(':')[0].replace('/', '')
                try:
                    resp = requests.get(f"http://127.0.0.1:8000/api/snapshot/{base_sym}", timeout=1.0)
                    if resp.status_code == 200:
                        return p, base_sym, resp.json()
                except: pass
                return p, base_sym, None
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(5, len(current_positions))) as executor:
                results = list(executor.map(fetch_snap, current_positions))
                
            for p, base_sym, data in results:
                try:
                    gate_sym = p['symbol']
                    pos_side = str(p.get('side', '')).upper()
                    info = p.get('info', {}) or {}
                    entry_px = float(p.get('entryPrice', info.get('entry_price', info.get('avgPx', 0))) or 0.0)
                    if entry_px <= 0:
                        continue

                    mark_px = float(p.get('markPrice', info.get('mark_price', entry_px)) or entry_px)
                    current_close = float(data.get('current_price', mark_px) if data else mark_px)
                    current_close = current_close if current_close > 0 else entry_px
                    trailing_atr_price = (
                        _calc_recent_atr(data.get('ohlcv_1m'), TRAILING_ATR_PERIOD, current_close)
                        if data and data.get('ohlcv_1m')
                        else 0.0
                    )

                    # 冷却检测 (防重复触发)
                    last_fired_time = bailout_cooldown_locks.get(base_sym, 0)
                    if time.time() - last_fired_time < 5.0:
                        continue

                    # 初始化该币种的内存追踪器
                    if base_sym not in tracking_mem:
                        entry_time = _resolve_position_entry_time(p, base_sym)
                        
                        tracking_mem[base_sym] = {
                            'max_phys_pnl': 0.0,
                            'dynamic_sl_phys': -abs(hard_sl),
                            'trailing_active': False,
                            'entry_time': entry_time,
                            'last_same_signal_refresh_ts': 0.0,
                            'last_sl_update_px': 0.0,
                            'last_entry_px': entry_px,  # 👈 补丁 1/2：必须记录首次进入右脑时的均价
                            'last_stop_repair_ts': 0.0,
                            'last_snapshot_warn_ts': 0.0,
                        }
                        
                    mem = tracking_mem[base_sym]
                    # ---------------------------------------------------------
                    # 🚨 补丁 2/2：侦测阶梯加仓造成的均价偏移，重置水位防线防误杀
                    # ---------------------------------------------------------
                    if 'last_entry_px' not in mem: 
                        mem['last_entry_px'] = entry_px
                        
                    # 如果交易所最新返回的均价和我们内存中的均价偏差超过 0.01%，说明触发了实盘加仓！
                    if abs(entry_px - mem['last_entry_px']) / max(mem['last_entry_px'], 1e-9) > 0.0001:
                        # 记住当时的物理绝对止损价格
                        if pos_side == 'LONG':
                            abs_sl = mem['last_entry_px'] * (1 + mem['dynamic_sl_phys'])
                            mem['dynamic_sl_phys'] = max((abs_sl - entry_px) / entry_px, -abs(hard_sl))
                            mem['max_phys_pnl'] = (current_close - entry_px) / entry_px
                        else:
                            abs_sl = mem['last_entry_px'] * (1 - mem['dynamic_sl_phys'])
                            mem['dynamic_sl_phys'] = max((entry_px - abs_sl) / entry_px, -abs(hard_sl))
                            mem['max_phys_pnl'] = (entry_px - current_close) / entry_px
                            
                        mem['last_entry_px'] = entry_px
                        print(f"⚖️ [{base_sym}] 右脑侦测到阶梯加仓行为！新均价 ${entry_px:.4f}，已平移止损防线防止误杀。")
                    # ---------------------------------------------------------
                    # 📈 轨道 1：动态物理状态机 (更新底层 OCO 止损单)
                    # ---------------------------------------------------------
                    current_phys_pnl = (current_close - entry_px) / entry_px if pos_side == 'LONG' else (entry_px - current_close) / entry_px
                    mem['max_phys_pnl'] = max(mem['max_phys_pnl'], current_phys_pnl)
                    _refresh_entry_time_on_same_direction_signal(mem, base_sym, pos_side, live_status, th)
                    hold_hours = (time.time() - mem['entry_time']) / 3600.0
                    
                    # 🚨 修复 2：为了和回测 (backtest_itransformer.py) 保持 100% 物理一致，
                    # 必须使用模型【最原始的概率】来判断反转斩仓，绝不能用宏观融合后的变异概率。
                    is_reversal = False
                    if pos_side == 'LONG' and raw_p_down > th.get('down_top_1', 1.0): is_reversal = True
                    if pos_side == 'SHORT' and raw_p_up > th.get('up_top_1', 1.0): is_reversal = True
                    
                    need_market_close = False
                    close_reason = ""
                    
                    # 触发器 1 & 3：反转 或 超过配置中的持仓时长上限
                    if is_reversal or hold_hours >= HOLD_PERIOD:
                        if current_phys_pnl < 0.005 and not mem['trailing_active']:
                            need_market_close = True
                            close_reason = "AI反转" if is_reversal else f"{HOLD_PERIOD}小时持仓超时"
                            close_reason += " (利润不足0.5%立即斩仓)"
                        else:
                            mem['trailing_active'] = True
                            mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], 0.005)
                            
                    # 触发器 2：触及止盈线
                    if mem['max_phys_pnl'] >= hard_tp + 0.005:
                        mem['trailing_active'] = True
                        mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], hard_tp)
                        
                    # 追踪算法应用：ATR 自适应回撤限制
                    if mem['trailing_active'] and trailing_atr_price > 0:
                        trailing_retrace_phys = TRAILING_ATR_MULTI * trailing_atr_price / entry_px
                        mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], mem['max_phys_pnl'] - trailing_retrace_phys)

                    if need_market_close:
                        execute_bailout(bailout_ex, base_sym, p, close_reason, current_close)
                        bailout_cooldown_locks[base_sym] = time.time()
                        del tracking_mem[base_sym]
                        continue
                        
                    target_sl_px = entry_px * (1 + mem['dynamic_sl_phys']) if pos_side == 'LONG' else entry_px * (1 - mem['dynamic_sl_phys'])
                    export_ui_mem[base_sym] = {'current_stop': target_sl_px}

                    if data is None:
                        snapshot_warn_cooldown = 60.0
                        since_last_warn = time.time() - float(mem.get('last_snapshot_warn_ts', 0.0) or 0.0)
                        if since_last_warn >= snapshot_warn_cooldown:
                            msg = f"[{base_sym}] 右脑未取到本地行情快照，已退化使用交易所 mark_price=${mark_px:.4f} 维持止损托管。"
                            print(f"⚠️ {msg}")
                            log_system_alert("⚠️ 右脑快照缺失", base_sym, msg)
                            mem['last_snapshot_warn_ts'] = time.time()

                    # ---------------------------------------------------------
                    # 🩹 轨道 1.5：若交易所止损单丢失/覆盖不足，立即按当前防线补挂
                    # ---------------------------------------------------------
                    stop_coverage = _inspect_snapshot_stop_coverage(p)
                    repairable_statuses = {"missing_stop", "undercovered_stop", "unparseable_stop"}
                    if stop_coverage["status"] in repairable_statuses:
                        repair_cooldown = 8.0
                        since_last_repair = time.time() - float(mem.get('last_stop_repair_ts', 0.0) or 0.0)
                        if since_last_repair >= repair_cooldown:
                            pos_size = abs(float(info.get('size', p.get('contracts', 0)) or 0.0))
                            repaired = update_exchange_sl(bailout_ex, gate_sym, pos_side, target_sl_px, pos_size)
                            mem['last_stop_repair_ts'] = time.time()
                            if repaired:
                                mem['last_sl_update_px'] = target_sl_px
                                repair_msg = (
                                    f"[{base_sym}] 检测到止损缺口，已按当前防线补挂止损 | "
                                    f"status={stop_coverage['status']} | stop=${target_sl_px:.4f}"
                                )
                                print(f"🩹 {repair_msg}")
                                log_system_alert("🩹 补挂止损", base_sym, repair_msg)
                            else:
                                fail_msg = (
                                    f"[{base_sym}] 检测到止损缺口，但补挂失败 | "
                                    f"status={stop_coverage['status']} | reason={stop_coverage['reason']}"
                                )
                                print(f"⚠️ {fail_msg}")
                                log_system_alert("⚠️ 止损补挂失败", base_sym, fail_msg)
                    
                    if mem['trailing_active']:
                        diff_pct = abs(target_sl_px - mem['last_sl_update_px']) / (entry_px + 1e-9)
                        if diff_pct > 0.001: 
                            pos_size = abs(float(info.get('size', p.get('contracts', 0)) or 0.0))
                            success = update_exchange_sl(bailout_ex, gate_sym, pos_side, target_sl_px, pos_size)
                            if success:
                                mem['last_sl_update_px'] = target_sl_px
                                print(f"📍 [{base_sym}] {pos_side} 追踪止损已上移至 ${target_sl_px:.4f} (物理保底: {mem['dynamic_sl_phys']*100:+.2f}%)")

                    # ---------------------------------------------------------
                    # 🆘 轨道 2：高频瀑布插针熔断 (主动雷达逃生)
                    # ---------------------------------------------------------
                    if data and 'ohlcv_1m' in data and len(data['ohlcv_1m']) >= 14:
                        ohlcv_1m = data['ohlcv_1m']
                        prev_minute_close = float(ohlcv_1m[-2][4])
                        current_open = float(ohlcv_1m[-1][1])
                        current_high = max(float(ohlcv_1m[-1][2]), current_close)
                        current_low = min(float(ohlcv_1m[-1][3]), current_close)
                        current_vol = float(ohlcv_1m[-1][5])
                        
                        past_vols = [x[5] for x in ohlcv_1m[:-1]]
                        avg_vol = sum(past_vols) / len(past_vols) if past_vols else 1e-8
                        
                        recent_15 = ohlcv_1m[-15:]
                        tr_list = []
                        for i in range(1, len(recent_15)):
                            prev_close = float(recent_15[i-1][4])
                            curr_h = float(recent_15[i][2])
                            curr_l = float(recent_15[i][3])
                            tr = max(curr_h - curr_l, abs(curr_h - prev_close), abs(curr_l - prev_close))
                            tr_list.append(tr)
                            
                        atr_1m = sum(tr_list) / len(tr_list) if tr_list else 0
                        
                        if atr_1m > 0:
                            dyn_atr_multi = BAILOUT_ATR_MULTI
                            dyn_vol_multi = BAILOUT_VOL_MULTI
                            
                            is_flash_crash = False
                            is_flash_surge = False
                            min_physical_space = current_close * BAILOUT_MIN_PHYSICAL_DROP
                            adverse_for_long = current_close < min(prev_minute_close, current_open)
                            adverse_for_short = current_close > max(prev_minute_close, current_open)
                            
                            if pos_side == 'LONG':
                                drop_amt = current_high - current_close
                                if adverse_for_long and drop_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                                    is_flash_crash = True
                            elif pos_side == 'SHORT':
                                surge_amt = current_close - current_low
                                if adverse_for_short and surge_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                                    is_flash_surge = True
                                    
                            if (is_flash_crash or is_flash_surge) and ('orderbook' in data):
                                actual_vol_multi = current_vol / (avg_vol + 1e-8)
                                if actual_vol_multi >= dyn_vol_multi:
                                    ob = data['orderbook']
                                    bid_vol = sum([float(o[0]) * float(o[1]) for o in ob['bids'][:20]])
                                    ask_vol = sum([float(o[0]) * float(o[1]) for o in ob['asks'][:20]])
                                    
                                    obi_thresh = BAILOUT_OBI_THRESH
                                    
                                    if pos_side == 'LONG' and is_flash_crash:
                                        obi = bid_vol / (ask_vol + 1e-8)
                                        if obi < obi_thresh:
                                            execute_bailout(bailout_ex, base_sym, p, f"真瀑布崩塌 (量能放大 {actual_vol_multi:.1f}x)", current_close)
                                            bailout_cooldown_locks[base_sym] = time.time()
                                            del tracking_mem[base_sym]
                                            continue
                                            
                                    elif pos_side == 'SHORT' and is_flash_surge:
                                        obi = ask_vol / (bid_vol + 1e-8)
                                        if obi < obi_thresh:
                                            execute_bailout(bailout_ex, base_sym, p, f"真逼空爆发 (量能放大 {actual_vol_multi:.1f}x)", current_close)
                                            bailout_cooldown_locks[base_sym] = time.time()
                                            del tracking_mem[base_sym]
                                            continue
                except Exception as pos_e:
                    msg = f"[{base_sym}] 右脑处理单仓位失败: {pos_e}"
                    print(f"⚠️ {msg}")
                    log_system_alert("⚠️ 右脑单仓异常", base_sym, msg)

            # 🔄 同步写入文件，供前端 UI 读取展示虚线保底
            try:
                tmp_path = os.path.join(LIVE_DATA_DIR, "chandelier_state.json.tmp")
                with open(tmp_path, "w") as f:
                    json.dump(export_ui_mem, f)
                os.replace(tmp_path, os.path.join(LIVE_DATA_DIR, "chandelier_state.json"))
            except: pass
                
        except Exception as loop_e:
            if time.time() - last_worker_error_ts >= 30.0:
                msg = f"右脑主循环异常: {loop_e}"
                print(f"⚠️ {msg}")
                log_system_alert("⚠️ 右脑线程异常", "SYSTEM", msg)
                last_worker_error_ts = time.time()
        time.sleep(1)
