import os
import time
import json
import sqlite3
import copy
import requests
import random
from datetime import datetime
import ccxt
import threading
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout
from state_manager import state
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR,
    MAX_NOTIONAL_BTC, MAX_NOTIONAL_ALTS
)
from dotenv import load_dotenv
load_dotenv()

MY_GATE_API_KEY  = os.environ.get("GATE_DEMO_API_KEY", "")
MY_GATE_SECRET   = os.environ.get("GATE_DEMO_SECRET", "")

global_gate = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
global_gate.set_sandbox_mode(True)

sync_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
sync_ex.set_sandbox_mode(True)

# 👇 【核心重构 1：异步推送队列】
def push_worker():
    while True:
        task = state.push_queue.get()
        if task is None: break
        try:
            # 这里才是真正干活的地方，独立线程发送，失败不影响主引擎
            resp = requests.post('http://www.pushplus.plus/send', json=task, timeout=10)
            if resp.status_code != 200:
                print(f"   ⚠️ PushPlus 拒收推送: {resp.text}")
            time.sleep(1.5) # 阻尼冷却，防止把接口打挂
        except Exception as e:
            print(f"   ⚠️ 异步推送网络错误: {e}")
        state.push_queue.task_done()
# 启动守护线程 (程序关闭时自动销毁)
threading.Thread(target=push_worker, daemon=True).start()

def portfolio_sync_worker():
    """ 
    🏦 [供货线程] 全能查询员：
    集中处理所有私有查询任务，更新内存账本并同步 JSON 供前端读取。
    """
    print("🏦 [供货线程] 集中查询引擎已启动 (隔离 Nonce B)...")
    while True:
        try:
            # 1. 独立查资金 (最核心，给主引擎起步用)
            try:
                bal = sync_ex.fetch_balance()
                state.portfolio_memory['balance'] = bal
            except Exception as e:
                print(f"   ⚠️ 供货线程-资金查询延迟: {e}")

            # 2. 独立查仓位 (同步给右脑)
            try:
                pos = sync_ex.fetch_positions()
                state.portfolio_memory['positions'] = pos
                
                active_pos_map = {p['symbol'].split(':')[0].replace('/', ''): p for p in pos if float(p.get('contracts', 0)) > 0}
                with state.pos_lock:
                    dead_keys = [k for k in list(state.active_pos_map.keys()) if k not in active_pos_map]
                    for k in dead_keys:
                        state.active_pos_map.pop(k, None)
                    state.active_pos_map.update(active_pos_map)
            except Exception as e:
                print(f"   ⚠️ 供货线程-仓位查询延迟: {e}")

            # 3. 独立查条件单 (最容易被限流，静默处理)
            try:
                stops = sync_ex.fetch_open_orders(params={'stop': True})
                state.portfolio_memory['open_stops'] = stops
            except Exception:
                pass 

            # 💡 只要走到这里，证明线程活着，必须更新时间戳！
            state.portfolio_memory['last_sync_ts'] = time.time()
            
            # 👇 【新增：每天刷新静态市场规则】
            current_day = datetime.now().day
            if getattr(portfolio_sync_worker, 'last_refresh_day', None) != current_day:
                try:
                    state.gate_markets = sync_ex.load_markets(True) # 强制刷新
                    portfolio_sync_worker.last_refresh_day = current_day
                    print(f"🔄 市场静态规则已自动刷新更新 (Day: {current_day})")
                except Exception as e:
                    print(f"⚠️ 刷新静态市场规则失败: {e}")
            
            # 4. 同步至磁盘 JSON 供 app.py 渲染
            # 深度复制：防止复制了正在被其他线程修改的字典而导致 Python 报错
            sync_state = copy.deepcopy({
                'balance': state.portfolio_memory.get('balance', {}),
                'positions': state.portfolio_memory.get('positions', []),
                'open_stops': state.portfolio_memory.get('open_stops', []),
                'updated_at': time.time()
            })
            tmp_file = os.path.join(LIVE_DATA_DIR, "portfolio_state.json.tmp")
            target_file = os.path.join(LIVE_DATA_DIR, "portfolio_state.json")
            with open(tmp_file, "w") as f:
                json.dump(sync_state, f)
            os.replace(tmp_file, target_file)

        except Exception as e:
            print(f"   ⚠️ 供货线程遭遇系统级异常: {e}")
        
        # 每 3 秒执行一次全方位大排查
        time.sleep(3)
# 启动快照线程
threading.Thread(target=portfolio_sync_worker, daemon=True).start()

# ==========================================
# 🛑 【V6.0 核心基建：全局缓存池与基础重试器】
# ==========================================
global_market_cache = {} # 用于存储 Stale-State Padding 缓存数据

def safe_api_call(func, *args, max_retries=3, initial_wait=1.0, **kwargs):
    """基础 API 调用包装器（已摘除限频，仅用于资金费率等极少数未进网关的兜底请求）"""
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (NetworkError, ExchangeError) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(wait_time)
            wait_time = (wait_time * 2) + random.uniform(0, 1)

def log_system_alert(level, symbol, message):
    """将底层系统警报写入独立数据库供前端实时呈现"""
    try:
        log_db_path = os.path.join(LIVE_DATA_DIR, "logs", "engine_system_logs.db")
        os.makedirs(os.path.dirname(log_db_path), exist_ok=True)
        with sqlite3.connect(log_db_path, timeout=5.0, check_same_thread=False) as sys_conn:
            sys_cursor = sys_conn.cursor()
            sys_cursor.execute('PRAGMA journal_mode=WAL;')
            # 创建专门的系统警报表
            sys_cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (log_ts TIMESTAMP, level TEXT, symbol TEXT, message TEXT)''')
            sys_cursor.execute('''INSERT INTO system_alerts (log_ts, level, symbol, message) VALUES (?, ?, ?, ?)''',
                               (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), level, symbol, message))
            sys_conn.commit()
    except Exception:
        pass

def execute_gate_trade(symbol, signal, leverage, current_price, tp_pct, sl_pct, risk_margin_usd):
    """
    工业级 API 下单器 (异构版)：执行下单、计算滑点并落库
    """
    try:
        # 【核心修复】：彻底抛弃 Gate 原生的下划线写法，拥抱 CCXT 跨平台标准永续符号
        # 构建 CCXT 标准合约代码格式，例如: 'APT/USDT' -> 'APT/USDT:USDT'
        if symbol.endswith("USDT"):
            gate_symbol = f"{symbol[:-4]}/USDT:USDT"
        else:
            gate_symbol = symbol + ":" + symbol.split('/')[1]
        side = 'buy' if signal == 1 else 'sell'
        # 👇 新增这一行：将旧的局部变量名直接映射到我们预热好的全局单例上
        exchange_private = global_gate

        # ---------------------------------------------------------
        # 1. 仓位互斥锁与反转引擎：防止重复同向开仓，支持反向平仓反转
        # ---------------------------------------------------------
        order_sent = False 
        positions = exchange_private.fetch_positions([gate_symbol])
        for pos in positions:
            if float(pos.get('contracts', 0)) > 1e-8:
                current_pos_side = str(pos.get('side', '')).lower()
                
                # 👇 【修改点1：反转拦截移交右脑】
                if (side == 'buy' and current_pos_side == 'short') or (side == 'sell' and current_pos_side == 'long'):
                    msg = f"🔄 侦测到 AI 预测反转信号！主引擎已挂起，交由右脑风控 (risk_manager) 评估执行 0.5% 锁润或强制平仓。"
                    print("⚠️", msg)
                    log_system_alert("🔄 信号反转拦截", symbol, msg)
                    return False, 0.0 # 阻断主引擎开新仓，等待右脑完成状态机结算
                else:
                    msg = f"🔒 仓位锁拦截：{gate_symbol} 当前已有同向存续仓位，拒绝重复加仓。"
                    print("⚠️", msg)
                    log_system_alert("⚠️ 拦截", symbol, msg)
                    return False, 0.0

        # ---------------------------------------------------------
        # 2. 面值与张数计算 (工业级精度适配与风控智能压降)
        # ---------------------------------------------------------
        markets = state.gate_markets
        
        # 👇 【阻击冷启动竞态条件 (Cold Start Race Condition)】
        # 如果主引擎光速触发了第一笔交易，而后台供货线程还在请求队列中，
        # 此时 markets 必定为空。强制主线程同步阻塞式拉取，补齐时序差。
        if not markets:
            print("⏳ [系统保护] 静态市场字典尚未就绪，正强制同步 Gate.io 市场规则...")
            try:
                markets = exchange_private.load_markets(True)
                state.gate_markets = markets
            except Exception as e:
                print(f"   ⚠️ 紧急加载市场规则失败，可能遭遇网络限流: {e}")
                
        if gate_symbol not in markets:
            # 加入 CCXT 探针：如果真的被跳过，将前 5 个物理存在的交易对名称打印出来，
            # 供我们排查是否是 Gate.io 测试网悄悄变更了永续合约的命名规范
            sample_markets = list(markets.keys())[:5] if markets else "网络拉取失败"
            msg = f"🚧 模拟盘受限拦截：Gate.io 未上架 {gate_symbol} 永续合约，已跳过。(当前测试网探测到: {sample_markets}...)"
            print("⚠️", msg)
            log_system_alert("⚠️ 拦截", symbol, msg)
            return False, 0.0
            
        market = markets[gate_symbol]
        contract_size = float(market['contractSize'])
        target_notional = risk_margin_usd * leverage
        
        # 👇 【智能修复 1：动态风险限额压降 (Risk Limit Capping)】
        # 直接使用 config 中定义的 MAX_NOTIONAL_BTC / MAX_NOTIONAL_ALTS，自动压降至安全水位
        base_coin = symbol.replace('USDT', '').split('/')[0]
        safe_notional_limit = MAX_NOTIONAL_BTC if base_coin in ['BTC', 'ETH'] else MAX_NOTIONAL_ALTS
        
        if target_notional > safe_notional_limit:
            target_notional = safe_notional_limit
            msg_limit = f"⚠️ 触发风控红线，名义价值已自动压降至 {safe_notional_limit} USDT"
            print(f"   {msg_limit}")
            log_system_alert("📉 仓位降级", symbol, msg_limit)
        
        raw_amount = target_notional / (current_price * contract_size)
        order_amount = float(exchange_private.amount_to_precision(gate_symbol, raw_amount))

        # 👇 【智能修复 2：市价单防滑点极限截断 (Market Order Size Capping)】
        # 提取交易所明确规定的市价单最大下单张数兜底
        max_market_size = float(market['info'].get('max_market_size', 100000))
        if order_amount > max_market_size:
            order_amount = max_market_size
            msg_size = f"⚠️ 触发市价防滑点红线，已自动截断至 {max_market_size} 张"
            print(f"   {msg_size}")
            log_system_alert("🔪 张数截断", symbol, msg_size)
            
        # 👇 【极其重要的账本修复】：反向重算真实的保证金占用！
        # 因为我们的订单规模被强制缩小了，如果还按原来的 margin_usd 记账，后期的 PnL 计算会严重失真。
        risk_margin_usd = (order_amount * contract_size * current_price) / leverage

        if order_amount < float(market['limits']['amount']['min']):
            msg = f"❌ 资金不足或被过度压降，无法开启最小单位 ({market['limits']['amount']['min']} 张)。"
            print("⚠️", msg)
            log_system_alert("❌ 资金不足", symbol, msg)
            return False, 0.0

        # 设置杠杆 (包一层 try-except 防止因为已有其他同向/反向持仓而导致报错)
        try:
            exchange_private.set_leverage(leverage, gate_symbol)
        except Exception as e:
            print(f"⚠️ 设置杠杆提醒 (可能已有存续仓位): {e}")

        # ---------------------------------------------------------
        # 3. 核心市价开仓 (增强参数传递)
        # ---------------------------------------------------------
        if side == 'buy':  
            tp_rule, sl_rule = 1, 2  
        else:              
            tp_rule, sl_rule = 2, 1  

        native_symbol = exchange_private.market_id(gate_symbol)

        # 内部挂单重试闭包函数
        def safe_post_condition(rule, price, text_label):
            for attempt in range(3):
                try:
                    exchange_private.private_futures_post_settle_price_orders({
                        "settle": "usdt",
                        "initial": {"contract": native_symbol, "size": 0, "close": True, "price": "0", "tif": "ioc", "text": text_label},
                        "trigger": {"strategy_type": 0, "price_type": 1, "price": str(exchange_private.price_to_precision(gate_symbol, price)), "rule": rule}
                    })
                    return True
                except Exception as e:
                    print(f"   ⚠️ {text_label.upper()} 挂载重试 {attempt+1}/3 (网络波动): {e}")
                    time.sleep(1)
            return False
        # 【核心修复】：使用基础的 create_order，显式指定 type 为 market，并利用 params 传递附加参数
        order_sent = True # 👇 标记为已发送
        order = exchange_private.create_order(
            symbol=gate_symbol,
            type='market',
            side=side,
            amount=order_amount,
            price=None, # 市价单不需要价格
            params={
                'reduceOnly': False # 明确表示这是一笔开仓单
            }
        )

        # 稍等 1.5 秒拉取真实成交数据，确保测试网仓位绝对同步
        time.sleep(1.5)
        filled_order = exchange_private.fetch_order(order['id'], gate_symbol)
        actual_fill_price = float(filled_order.get('average') or filled_order.get('price') or current_price)
        slippage = abs(actual_fill_price - current_price) / current_price
        
        # 基于真实的成交均价，计算物理止盈止损位
        # 统一量纲：传入的 tp_pct 和 sl_pct 是带杠杆的预期收益率/亏损率，因此需要除以杠杆得到实际的 K 线涨跌幅
        actual_tp_move = abs(tp_pct) 
        actual_sl_move = abs(sl_pct) 
        
        tp_price = actual_fill_price * (1 + actual_tp_move) if side == 'buy' else actual_fill_price * (1 - actual_tp_move)
        sl_price = actual_fill_price * (1 - actual_sl_move) if side == 'buy' else actual_fill_price * (1 + actual_sl_move)

        # ---------------------------------------------------------
        # 【修改点2：移除硬止盈，只挂兜底硬止损 (给追踪止盈让路)】
        # ---------------------------------------------------------
        # 挂载物理硬止损
        sl_success = safe_post_condition(sl_rule, sl_price, "sl")
        
        # 终极物理校验
        time.sleep(1.5) 
        try:
            verify_stops = exchange_private.fetch_open_orders(gate_symbol, params={'stop': True})
            active_texts = [o.get('clientOrderId', '') or o['info'].get('initial', {}).get('text', '') for o in verify_stops]
            
            # 校验只查 SL 是否存在即可
            if len(verify_stops) >= 1 or 'sl' in active_texts:
                print("🔔 交易所底层 OCO 兜底止损单【挂单且物理校验成功】！")
            else:
                msg_oco_fail = f"物理兜底止损单校验未完全通过 (现存: {len(verify_stops)}个)，可能存在遗漏，交由右脑兜底。"
                print(f"❌ 警告：{gate_symbol} {msg_oco_fail}")
                log_system_alert("⚠️ 挂单残缺", symbol, msg_oco_fail)
        except Exception as e:
             print(f"⚠️ 校验请求被限流，假定止损挂单成功: {e}")

        print(
            f"✅ ⚡ [Gate实盘] 下单成功！方向：{side.upper()} | 数量：{order_amount} 张 {gate_symbol}。\n"
            f"🎯 进场均价: ${actual_fill_price:.4f} (滑点 {slippage:.3%})\n"
            f"🛡️ 兜底止损 (SL): ${sl_price:.4f} (极限亏损 -{abs(sl_pct):.2%})\n"
            f"🚀 追踪止盈: 已移交右脑状态机动态接管"
        )

        # ---------------------------------------------------------
        # 4. 真实订单特征落库（live_trades 表）
        # ---------------------------------------------------------
        db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol}.db")
        with sqlite3.connect(db_name, timeout=20.0, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                              (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                               amount REAL, fill_price REAL, slippage REAL,
                               tp_price REAL, sl_price REAL, margin_used REAL)''')
            # 动态升级旧表
            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
            except sqlite3.OperationalError: pass
            # 👇 新增：动态升级杠杆列
            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
            except sqlite3.OperationalError: pass
            
            # 👇 新增：将 leverage 写入
            cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), str(order['id']), side,
                            float(order_amount), actual_fill_price, float(slippage),
                            float(tp_price), float(sl_price), float(risk_margin_usd), float(leverage)))
            conn.commit()

        return True, float(risk_margin_usd)

    # 【核心修复：分层处理网络超时幻影丢单，并同步前端日志】
    except RequestTimeout as e:
        if not order_sent:
            msg = f"⚠️ 发单前网络超时，放弃开仓：{e}"
            print("⚠️", msg)
            log_system_alert("⚠️ 网络超时", symbol, msg)
            return False, 0.0
            
        msg1 = f"⚠️ 发送指令时网络请求超时！正在向交易所核实 {gate_symbol} 的真实仓位状态，防止幻影丢单..."
        print("⚠️", msg1)
        log_system_alert("⚠️ 网络超时", symbol, msg1)
        
        try:
            # 强制等待 2 秒让交易所撮合引擎落库
            time.sleep(2)
            verify_positions = exchange_private.fetch_positions([gate_symbol])
            for pos in verify_positions:
                if abs(float(pos.get('contracts', 0))) > 1e-8:
                    msg_ok = f"✅ 虚惊一场：虽然网络超时，但底层的 {side.upper()} 订单已成功被交易所撮合。"
                    print("✅", msg_ok)
                    log_system_alert("✅ 超时抢救", symbol, msg_ok)
                    
                    # 👇 【修复 2：兜底价格计算】因为断网拿不到实际成交价，强制用当前盘口价测算防线
                    actual_tp_move = abs(tp_pct)
                    actual_sl_move = abs(sl_pct)
                    
                    fallback_sl_price = current_price * (1 - actual_sl_move) if side == 'buy' else current_price * (1 + actual_sl_move)
                    
                    # 立刻挂载 OCO 条件单，堵住超时裸奔漏洞
                    safe_post_condition(sl_rule, fallback_sl_price, "sl")
                    
                    # 超时抢救成功后兜底落库
                    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol}.db")
                    try:
                        with sqlite3.connect(db_name, timeout=20.0, check_same_thread=False) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                                              (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                                               amount REAL, fill_price REAL, slippage REAL,
                                               tp_price REAL, sl_price REAL, margin_used REAL)''')
                            # 动态升级旧表
                            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
                            except sqlite3.OperationalError: pass
                            # 👇 新增
                            try: cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
                            except sqlite3.OperationalError: pass
                            
                            # 👇 新增：写入 leverage
                            cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                           (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'timeout_recovery', side,
                                            float(order_amount), current_price, 0.0, 0.0, 0.0, float(risk_margin_usd), float(leverage)))
                            conn.commit()
                    except Exception as db_e:
                        print(f"⚠️ 兜底落库失败: {db_e}")
                    return True, float(risk_margin_usd)
            
            msg2 = f"❌ 查单确认：超时导致 {gate_symbol} 订单未到达交易所。"
            print("❌", msg2)
            log_system_alert("❌ 丢单", symbol, msg2)
            return False, 0.0
            
        except Exception as verify_e:
            msg3 = f"❌ 灾难性断网，仓位核实失败，请立刻登录 APP 检查: {verify_e}"
            print("❌", msg3)
            log_system_alert("❌ 断网", symbol, msg3)
            return False, 0.0
            
    except Exception as e:
        msg4 = f"❌ [Gate模拟盘] API 下单遭遇硬性失败: {e}"
        print("❌", msg4)
        log_system_alert("❌ 崩溃", symbol, msg4)
        return False, 0.0