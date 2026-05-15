import os
import time
import sqlite3
import pandas as pd
import requests
import json
from datetime import datetime
import ccxt
from state_manager import state
from execution_oms import log_system_alert
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR, MODEL_DIR,
    HARD_TP, HARD_SL,
    BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI,
    BAILOUT_MIN_PHYSICAL_DROP, BAILOUT_OBI_THRESH
)
from dotenv import load_dotenv
load_dotenv()

MY_GATE_API_KEY  = os.environ.get("GATE_DEMO_API_KEY", "")
MY_GATE_SECRET   = os.environ.get("GATE_DEMO_SECRET", "")

def execute_bailout(ex, base_sym, p, reason, current_price):
    """右脑专用：执行紧急市价逃生"""
    gate_sym = p['symbol']
    pos_side = str(p.get('side', '')).upper()
    contracts = float(p.get('contracts', 0))
    entry_px = float(p.get('entryPrice', p['info'].get('entry_price', p['info'].get('avgPx', 0))))
    
    print(f"\n🚨 [右脑介入] 侦测到 {base_sym} {reason}！执行市价核弹清仓！")
    try:
        # 1. 越权下发市价单
        cl_side = 'sell' if pos_side == 'LONG' else 'buy'
        ex.create_order(symbol=gate_sym, type='market', side=cl_side, amount=contracts, params={'reduceOnly': True})
        
        # 2. 扫清残余 OCO
        try:
            for o in ex.fetch_open_orders(gate_sym, params={'stop': True}):
                try: ex.cancel_order(o['id'], gate_sym, params={'stop': True})
                except: pass
        except: pass
        
        # 3. 落库：历史本纪
        db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
        with sqlite3.connect(db_name, timeout=10.0, check_same_thread=False) as conn_log:
            cursor = conn_log.cursor()
            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
            except: pass
            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
            except: pass
            
            cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'bailout_panic_sell', 'bailout_close',
                            float(contracts), current_price, 0.0, 0.0, 0.0, 0.0, 0.0))
            conn_log.commit()
                    
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
        
    except Exception as e:
        print(f"   ⚠️ 右脑熔断清仓失败: {e}")
        log_system_alert("❌ 熔断异常", base_sym, f"跳伞执行失败: {e}")

def update_exchange_sl(ex, gate_symbol, pos_side, new_sl_price):
    """向交易所发送 API：撤销旧止损，挂载新追踪止损"""
    try:
        # 1. 撤销旧单
        for o in ex.fetch_open_orders(gate_symbol, params={'stop': True}):
            try: ex.cancel_order(o['id'], gate_symbol, params={'stop': True})
            except: pass
        
        # 2. 挂载新单
        rule = 2 if pos_side == 'LONG' else 1
        native_symbol = ex.market_id(gate_symbol)
        ex.private_futures_post_settle_price_orders({
            "settle": "usdt",
            "initial": {"contract": native_symbol, "size": 0, "close": True, "price": "0", "tif": "ioc", "text": "trailing_sl"},
            "trigger": {"strategy_type": 0, "price_type": 1, "price": str(ex.price_to_precision(gate_symbol, new_sl_price)), "rule": rule}
        })
        return True
    except Exception as e:
        print(f"   ⚠️ 交易所止损单上移请求失败 (等待下秒重试): {e}")
        return False

def bailout_worker():
    """右脑主循环：双轨制 (物理高水位状态机 + ATR 主动熔断)"""
    print("🛡️ [右脑] 高频双轨制风控已启动 (追踪止盈 OCO + ATR 主动熔断)...")
    bailout_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    bailout_ex.set_sandbox_mode(True)
    
    # 🧠 物理水位状态机内存池
    tracking_mem = {}
    bailout_cooldown_locks = {}
    
    # 加载回测同款概率阈值
    th = {}
    try:
        with open(MODEL_DIR / "dynamic_thresholds_24h.json", 'r') as f:
            th = json.load(f)
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
                
            inference_data = live_status.get("model_inference", {})
            adj_p_up = inference_data.get("adj_prob_up", 0.0)
            adj_p_down = inference_data.get("adj_prob_down", 0.0)

            hard_tp = HARD_TP   # K线物理止盈幅度（直接来自 config）
            hard_sl = HARD_SL   # K线物理底部止损幅度（直接来自 config）
            
            export_ui_mem = {}
            
            import concurrent.futures
            def fetch_snap(p):
                base_sym = p['symbol'].split(':')[0].replace('/', '')
                try:
                    resp = requests.get(f"http://127.0.0.1:8000/api/snapshot/{base_sym}", timeout=0.2)
                    if resp.status_code == 200:
                        return p, base_sym, resp.json()
                except: pass
                return p, base_sym, None
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(5, len(current_positions))) as executor:
                results = list(executor.map(fetch_snap, current_positions))
                
            for p, base_sym, data in results:
                if not data: continue
                gate_sym = p['symbol']
                pos_side = str(p.get('side', '')).upper()
                current_close = float(data['current_price'])
                entry_px = float(p.get('entryPrice', p['info'].get('entry_price', p['info'].get('avgPx', 0))))
                if entry_px <= 0: continue
                
                # 冷却检测 (防重复触发)
                last_fired_time = bailout_cooldown_locks.get(base_sym, 0)
                if time.time() - last_fired_time < 5.0:
                    continue

                # 初始化该币种的内存追踪器
                if base_sym not in tracking_mem:
                    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
                    entry_time = time.time()
                    try:
                        with sqlite3.connect(db_name, timeout=1.0) as conn:
                            cur = conn.cursor()
                            cur.execute("SELECT exec_ts FROM live_trades WHERE side IN ('buy', 'sell') ORDER BY exec_ts DESC LIMIT 1")
                            row = cur.fetchone()
                            if row: entry_time = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()
                    except: pass
                    
                    tracking_mem[base_sym] = {
                        'max_phys_pnl': 0.0,
                        'dynamic_sl_phys': -abs(hard_sl),
                        'trailing_active': False,
                        'entry_time': entry_time,
                        'last_sl_update_px': 0.0
                    }
                    
                mem = tracking_mem[base_sym]
                
                # ---------------------------------------------------------
                # 📈 轨道 1：动态物理状态机 (更新底层 OCO 止损单)
                # ---------------------------------------------------------
                current_phys_pnl = (current_close - entry_px) / entry_px if pos_side == 'LONG' else (entry_px - current_close) / entry_px
                mem['max_phys_pnl'] = max(mem['max_phys_pnl'], current_phys_pnl)
                hold_hours = (time.time() - mem['entry_time']) / 3600.0
                
                is_reversal = False
                if pos_side == 'LONG' and adj_p_down > th.get('down_top_1', 1.0): is_reversal = True
                if pos_side == 'SHORT' and adj_p_up > th.get('up_top_1', 1.0): is_reversal = True
                
                need_market_close = False
                close_reason = ""
                
                # 触发器 1 & 3：反转 或 超过 24 小时
                if is_reversal or hold_hours >= 24.0:
                    if current_phys_pnl < 0.005 and not mem['trailing_active']:
                        need_market_close = True
                        close_reason = "AI反转" if is_reversal else "24小时持仓超时"
                        close_reason += " (利润不足0.5%立即斩仓)"
                    else:
                        mem['trailing_active'] = True
                        mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], 0.005)
                        
                # 触发器 2：触及止盈线
                if mem['max_phys_pnl'] >= hard_tp + 0.005:
                    mem['trailing_active'] = True
                    mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], hard_tp)
                    
                # 追踪算法应用：1% 回撤限制
                if mem['trailing_active']:
                    mem['dynamic_sl_phys'] = max(mem['dynamic_sl_phys'], mem['max_phys_pnl'] - 0.01)

                if need_market_close:
                    execute_bailout(bailout_ex, base_sym, p, close_reason, current_close)
                    bailout_cooldown_locks[base_sym] = time.time()
                    del tracking_mem[base_sym]
                    continue
                    
                target_sl_px = entry_px * (1 + mem['dynamic_sl_phys']) if pos_side == 'LONG' else entry_px * (1 - mem['dynamic_sl_phys'])
                export_ui_mem[base_sym] = {'current_stop': target_sl_px}
                
                if mem['trailing_active']:
                    diff_pct = abs(target_sl_px - mem['last_sl_update_px']) / (entry_px + 1e-9)
                    if diff_pct > 0.001: 
                        success = update_exchange_sl(bailout_ex, gate_sym, pos_side, target_sl_px)
                        if success:
                            mem['last_sl_update_px'] = target_sl_px
                            print(f"📍 [{base_sym}] {pos_side} 追踪止损已上移至 ${target_sl_px:.4f} (物理保底: {mem['dynamic_sl_phys']*100:+.2f}%)")

                # ---------------------------------------------------------
                # 🆘 轨道 2：高频瀑布插针熔断 (主动雷达逃生)
                # ---------------------------------------------------------
                if 'ohlcv_1m' in data and len(data['ohlcv_1m']) >= 14:
                    ohlcv_1m = data['ohlcv_1m']
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
                        dyn_atr_multi = BAILOUT_ATR_MULTI   # 直接来自 config
                        dyn_vol_multi = BAILOUT_VOL_MULTI   # 直接来自 config
                        
                        is_flash_crash = False
                        is_flash_surge = False
                        
                        # 设置一个物理底限，防止波动极小时的误触（直接使用 config 中定义的 BAILOUT_MIN_PHYSICAL_DROP）
                        min_physical_space = current_close * BAILOUT_MIN_PHYSICAL_DROP
                        
                        if pos_side == 'LONG':
                            drop_amt = current_high - current_close
                            if drop_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                                is_flash_crash = True
                        elif pos_side == 'SHORT':
                            surge_amt = current_close - current_low
                            if surge_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                                is_flash_surge = True
                                
                        if (is_flash_crash or is_flash_surge) and ('orderbook' in data):
                            actual_vol_multi = current_vol / (avg_vol + 1e-8)
                            if actual_vol_multi >= dyn_vol_multi:
                                ob = data['orderbook']
                                bid_vol = sum([float(o[0]) * float(o[1]) for o in ob['bids'][:20]])
                                ask_vol = sum([float(o[0]) * float(o[1]) for o in ob['asks'][:20]])
                                
                                obi_thresh = BAILOUT_OBI_THRESH  # 直接使用 config 中定义的订单簿失衡阈值
                                
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

            # 🔄 同步写入文件，供前端 UI 读取展示虚线保底
            try:
                tmp_path = os.path.join(LIVE_DATA_DIR, "chandelier_state.json.tmp")
                with open(tmp_path, "w") as f:
                    json.dump(export_ui_mem, f)
                os.replace(tmp_path, os.path.join(LIVE_DATA_DIR, "chandelier_state.json"))
            except: pass
                
        except Exception: pass
        time.sleep(1)