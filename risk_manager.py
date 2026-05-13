import os
import time
import sqlite3
import pandas as pd
import requests
from datetime import datetime
import ccxt
from state_manager import state
from execution_oms import log_system_alert
from quant_config.config import *


from quant_config.config import *
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
        db_name = os.path.join(DATA_DIR, f"trading_intelligence_{base_sym}.db")
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
def bailout_worker():
    """右脑主循环：每 1 秒高频巡视活体仓位 (融合防插针熔断与 ROE 动态追踪锁润)"""
    print("🛡️ [右脑] 高频微观熔断与动态锁润器已启动 (1秒巡视，微秒级本地感知)")
    bailout_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    bailout_ex.set_sandbox_mode(True)
    bailout_cooldown_locks = {}
    
    # 🧠 虚拟吊灯内存池 (记录每个币种的历史最高收益率)
    chandelier_mem = {}
    
    while True:
        try:
            with state.pos_lock:
                current_positions = list(state.active_pos_map.values())
                
            if not current_positions:
                time.sleep(1)
                continue
            
            # 清理已经平仓的遗留状态
            active_syms = [p['symbol'].split(':')[0].replace('/', '') for p in current_positions]
            for k in list(chandelier_mem.keys()):
                if k not in active_syms:
                    del chandelier_mem[k]
                    
            import concurrent.futures
            
            def fetch_snap(p):
                base_sym = p['symbol'].split(':')[0].replace('/', '')
                try:
                    resp = requests.get(f"http://127.0.0.1:8000/api/snapshot/{base_sym}", timeout=0.2)
                    if resp.status_code == 200:
                        return p, base_sym, resp.json()
                except Exception:
                    pass
                return p, base_sym, None
                
            worker_count = max(30, len(current_positions) + 5)
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(fetch_snap, current_positions))
                
            for p, base_sym, data in results:
                pos_side = str(p.get('side', '')).upper()
                
                last_fired_time = bailout_cooldown_locks.get(base_sym, 0)
                if time.time() - last_fired_time < 5.0:
                    continue
                
                if not data: continue
                
                try:
                    current_close = float(data['current_price'])
                    entry_px = float(p.get('entryPrice', p['info'].get('entry_price', p['info'].get('avgPx', 0))))
                    pos_lev = float(p.get('leverage', p['info'].get('leverage', p['info'].get('lever', 1))))
                    if pos_lev <= 0: pos_lev = 10.0
                    
                    # ==========================================
                    # 📈 1. 动态 ROE 追踪锁润 (Trailing Profit Lock)
                    # ==========================================
                    # 读取前端 UI 动态设定的目标参数 (默认赚 5% 开始追踪，回撤 3% 也就是保底 2% 就平仓)
                    dyn_trailing_trigger = float(state.dynamic_settings.get("trailing_trigger_roe", 5.0)) / 100.0
                    dyn_trailing_pullback = float(state.dynamic_settings.get("trailing_pullback_roe", 3.0)) / 100.0
                    
                    if entry_px > 0:
                        # 计算物理涨跌并放大为本金 ROE
                        if pos_side == 'LONG':
                            physical_move_pct = (current_close - entry_px) / entry_px
                        else:
                            physical_move_pct = (entry_px - current_close) / entry_px
                            
                        current_roe = physical_move_pct * pos_lev
                        
                        # 初始化或更新最高水位线 (High-Water Mark)
                        if base_sym not in chandelier_mem:
                            chandelier_mem[base_sym] = {'max_roe': 0.0, 'current_stop': 0.0}
                            
                        ch_state = chandelier_mem[base_sym]
                        if current_roe > ch_state['max_roe']:
                            ch_state['max_roe'] = current_roe
                            
                        # 如果最高收益曾经跨过起征点，激活锁润机制！
                        if ch_state['max_roe'] >= dyn_trailing_trigger:
                            # 计算需要硬性保底的利润率
                            locked_roe = ch_state['max_roe'] - dyn_trailing_pullback
                            required_physical_move = locked_roe / pos_lev
                            
                            # 将保底利润率反推回物理止盈价
                            if pos_side == 'LONG':
                                virtual_sl_px = entry_px * (1 + required_physical_move)
                                ch_state['current_stop'] = max(ch_state['current_stop'], virtual_sl_px) if ch_state['current_stop'] > 0 else virtual_sl_px
                            else:
                                virtual_sl_px = entry_px * (1 - required_physical_move)
                                ch_state['current_stop'] = min(ch_state['current_stop'], virtual_sl_px) if ch_state['current_stop'] > 0 else virtual_sl_px
                                
                            # 🎯 物理触发判定
                            if ch_state['current_stop'] > 0:
                                if (pos_side == 'LONG' and current_close <= ch_state['current_stop']) or \
                                   (pos_side == 'SHORT' and current_close >= ch_state['current_stop']):
                                    
                                    execute_bailout(bailout_ex, base_sym, p, f"🎯 动态追踪止盈 (已锁定 ROE: {locked_roe*100:+.1f}%)", current_close)
                                    bailout_cooldown_locks[base_sym] = time.time()
                                    del chandelier_mem[base_sym]
                                    continue # 已经平仓，跳过下面的瀑布防插针检查

                    # ==========================================
                    # 🆘 2. 原始高频瀑布插针熔断 (原封不动保留)
                    # ==========================================
                    ohlcv_1m = data['ohlcv_1m']
                    if len(ohlcv_1m) < 14: continue
                    
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
                    if atr_1m <= 0: continue
                    
                    dyn_atr_multi = float(state.dynamic_settings.get("bailout_atr_multi", BAILOUT_ATR_MULTI))
                    dyn_vol_multi = float(state.dynamic_settings.get("bailout_vol_multi", BAILOUT_VOL_MULTI))
                    
                    is_flash_crash = False
                    is_flash_surge = False
                    
                    min_physical_space = current_close * BAILOUT_MIN_PHYSICAL_DROP
                    
                    if pos_side == 'LONG':
                        drop_amt = current_high - current_close
                        if drop_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                            is_flash_crash = True
                    elif pos_side == 'SHORT':
                        surge_amt = current_close - current_low
                        if surge_amt > max((dyn_atr_multi * atr_1m), min_physical_space):
                            is_flash_surge = True
                            
                    if not (is_flash_crash or is_flash_surge): continue
                    
                    actual_vol_multi = current_vol / (avg_vol + 1e-8)
                    if actual_vol_multi < dyn_vol_multi: continue
                        
                    ob = data['orderbook']
                    bid_vol = sum([float(o[0]) * float(o[1]) for o in ob['bids'][:20]])
                    ask_vol = sum([float(o[0]) * float(o[1]) for o in ob['asks'][:20]])
                    
                    if pos_side == 'LONG' and is_flash_crash:
                        obi = bid_vol / (ask_vol + 1e-8)
                        if obi < BAILOUT_OBI_THRESH:
                            execute_bailout(bailout_ex, base_sym, p, f"真瀑布崩塌 (量能放大 {actual_vol_multi:.1f}x)", current_close)
                            bailout_cooldown_locks[base_sym] = time.time()
                            
                    elif pos_side == 'SHORT' and is_flash_surge:
                        obi = ask_vol / (bid_vol + 1e-8)
                        if obi < BAILOUT_OBI_THRESH:
                            execute_bailout(bailout_ex, base_sym, p, f"真逼空爆发 (量能放大 {actual_vol_multi:.1f}x)", current_close)
                            bailout_cooldown_locks[base_sym] = time.time()
                            
                except Exception: pass
                
            # 🔄 每秒将最新的虚拟吊灯状态写入硬盘，供前端 UI 页面读取展示
            try:
                tmp_path = os.path.join(LIVE_DATA_DIR, "chandelier_state.json.tmp")
                target_path = os.path.join(LIVE_DATA_DIR, "chandelier_state.json")
                with open(tmp_path, "w") as f:
                    json.dump(chandelier_mem, f)
                os.replace(tmp_path, target_path)
            except Exception: pass
                
        except Exception: pass
        
        time.sleep(1)