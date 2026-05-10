import asyncio
import ccxt.pro as ccxtpro
from collections import deque
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import random
from datetime import datetime
import time
import sqlite3
from quant_config.config import SYMBOL, AUX_SYMBOLS, WHALE_THRESHOLD, RETAIL_THRESHOLD, LIVE_DATA_DIR

app = FastAPI(title="Binance链币种: {} 极速行情网关".format(SYMBOL))

# ==========================================
# 🛑 全局并发限流阀 (防止冷启动惊群效应)
# ==========================================
# 强制规定：同一时刻最多只允许 3 个 REST 请求发往币安，保护本地 Socket 与币安 WAF
rest_semaphore = asyncio.Semaphore(3)

# ==========================================
# 🧠 内存状态机：行情数据的物理插槽
# ==========================================
# 替换原有的 market_data_store 初始化逻辑
market_data_store = {}

# 将主币种和辅助币种合并初始化
all_symbols = [SYMBOL] + AUX_SYMBOLS

for sym in all_symbols:
    market_data_store[sym] = {
        "1m": deque(maxlen=120),
        "1h": deque(maxlen=500), 
        "orderbook": {"bids": [], "asks": []},
        "ticker": {"last": None},
        "liquidations": deque(maxlen=1000), 
        "status": "initializing"
    }
    
    # 只有主币种才需要复杂的订单流和衍生品特征
    if sym == SYMBOL:
        market_data_store[sym].update({
            # 将 trades 升级，记录 quote_volume, count, 及大中小单
            "taker_stats_1h": {}, 
            "funding_rate": 0.0,                
            "open_interest": 0.0,
            "premium_index": deque(maxlen=500), # 新增：溢价指数
        })
# ==========================================
# 🗄️ 本地记忆库初始化 (SQLite)
# ==========================================
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = LIVE_DATA_DIR / "gateway_state.db"

def init_sqlite_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_1h (
                symbol TEXT,
                timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                quote_volume REAL, count INTEGER,
                taker_buy_volume REAL, taker_sell_volume REAL,
                whale_buy REAL, whale_sell REAL, retail_vol REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
init_sqlite_db()
# ==========================================
# 🛰️ WebSocket 搬运工：死循环监听流
# ==========================================
async def ohlcv_loop(exchange, symbol, swap_symbol, timeframe, engine_type, startup_delay):
    """ 纯粹的 K 线 WebSocket 监听流 (历史预热已交由中台全局引擎处理) """
    # 严格的错峰启动，杜绝 Binance 并发订阅静默丢包
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} ({timeframe}) {engine_type}K线订阅...")
    
    was_dead = False
    while True:
        try:
            # 🛡️ 【看门狗防线】：90秒收不到任何 K 线跳动，强制视为假死并抛出异常
            ohlcvs = await asyncio.wait_for(
                exchange.watch_ohlcv(swap_symbol, timeframe),
                timeout=90.0
            )
            
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} ({timeframe}) {engine_type}K线流已成功重新连接并收到数据！")
                was_dead = False
                
            for candle in ohlcvs:
                q = market_data_store[symbol][timeframe]
                
                # 同频更新去重 (Binance 的 WS 会每秒推送同一根 K 线的最新状态)
                if len(q) > 0 and q[-1][0] == candle[0]:
                    q[-1] = candle
                else:
                    q.append(candle)
                    
        except asyncio.TimeoutError:
            # 🚨 触发重连：由于我们有底层的 SQLite 兜底，这里只需简单的 WS 重连即可
            print(f"   ⚠️ [WS 看门狗] {symbol} ({timeframe}) 超过 90 秒假死，触发 WebSocket 重连！")
            # 强制清理底层死锁的 WS 客户端，促使下次调用重新发 SUBSCRIBE 请求
            for url, client in list(exchange.clients.items()):
                for f in list(client.futures.values()):
                    if isinstance(f, list):
                        for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                    else:
                        f.add_done_callback(lambda x: x.exception())
                client.reject(Exception(f"{symbol} ({timeframe}) K线流假死重连"))
                exchange.clients.pop(url, None)
            was_dead = True
            await asyncio.sleep(2)
        except Exception as e:
            print(f"   ❌ [WS 异常] {symbol} {timeframe}: {e}")
            await asyncio.sleep(5)

async def orderbook_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听盘口流 (增量自动合并) """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}盘口深度订阅...")
    
    was_dead = False
    while True:
        try:
            # 🛡️ 【看门狗防线】：90秒收不到盘口变动，强制视为假死
            orderbook = await asyncio.wait_for(
                exchange.watch_order_book(swap_symbol, limit=100),
                timeout=90.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}盘口深度流已成功恢复！")
                was_dead = False
                
            market_data_store[symbol]["orderbook"] = {
                "bids": orderbook['bids'],
                "asks": orderbook['asks']
            }
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 盘口深度超过 90 秒无变动，触发假死重连！")
            for url, client in list(exchange.clients.items()):
                for f in list(client.futures.values()):
                    if isinstance(f, list):
                        for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                    else:
                        f.add_done_callback(lambda x: x.exception())
                client.reject(Exception(f"{symbol} 盘口深度流假死重连"))
                exchange.clients.pop(url, None)
            was_dead = True
            await asyncio.sleep(2)
        except Exception as e:
            print(f"   ❌ [WS 盘口异常] {symbol}: {e}")
            await asyncio.sleep(5)

async def ticker_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听最新价流 """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}最新价流订阅...")
    
    was_dead = False
    while True:
        try:
            # 🛡️ 【看门狗防线】：90秒收不到最新价跳动，强制视为假死
            ticker = await asyncio.wait_for(
                exchange.watch_ticker(swap_symbol),
                timeout=90.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}最新价流已成功恢复！")
                was_dead = False
                
            market_data_store[symbol]["ticker"]["last"] = ticker['last']
            market_data_store[symbol]["status"] = "live"
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 最新价超过 90 秒无跳动，触发假死重连！")
            for url, client in list(exchange.clients.items()):
                for f in list(client.futures.values()):
                    if isinstance(f, list):
                        for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                    else:
                        f.add_done_callback(lambda x: x.exception())
                client.reject(Exception(f"{symbol} 最新价流假死重连"))
                exchange.clients.pop(url, None)
            was_dead = True
            await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(5)
async def liquidation_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听币安实时爆仓流 (CCXT 强力防呆兼容版) """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}爆仓流订阅...")
    
    while True:
        try:
            liquidations = await exchange.watch_liquidations(swap_symbol)
            for liq in liquidations:
                # 👇 核心防呆设计：CCXT 某些版本的字段映射不全，我们使用 .get() 穿透到底层 info
                
                # 1. 提取价格 (优先标准 price，缺失则去 info 里找 p)
                raw_price = liq.get('price') or liq.get('info', {}).get('p', 0)
                price = float(raw_price)
                
                # 2. 提取数量 (优先标准 amount，缺失则去 info 里找 q)
                raw_amount = liq.get('amount') or liq.get('info', {}).get('q', 0)
                amount = float(raw_amount)
                
                # 3. 提取方向 (优先标准 side，缺失则去 info 里找 S)
                side = str(liq.get('side') or liq.get('info', {}).get('S', 'buy')).lower()
                
                usd_val = price * amount
                
                # 写入内存队列
                market_data_store[symbol]["liquidations"].append({
                    "ts": liq.get('timestamp', int(datetime.utcnow().timestamp() * 1000)),
                    "side": side,
                    "usd_value": usd_val
                })
                
                # 💡 高频探针：打印出来确认系统在正常干活
                # dir_icon = "🩸多军被爆" if side == 'buy' else "🚀空军被爆"
                # print(f"   💥 [CCXT 爆仓侦测] {symbol} {dir_icon} | 金额: ${usd_val:,.0f}")
                
        except Exception as e:
            # 捕获异常，休眠后重试
            print(f"   ❌ [WS 爆仓流致命异常] {symbol}: {type(e).__name__} -> {e}")
            await asyncio.sleep(5)

async def trade_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听逐笔成交流，降维聚合为 1h 级别的高级量价与订单流特征 """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}逐笔成交流订阅 (包含大单拆解)...")
    
    was_dead = False
    while True:
        try:
            trades = await asyncio.wait_for(
                exchange.watch_trades(swap_symbol),
                timeout=90.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}逐笔成交流已成功恢复！")
                was_dead = False
                
            for t in trades:
                # 按 1h 降维 (对齐离线特征的 truncate("1h"))
                hour_ts = int(t['timestamp'] / 3600000) * 3600000
                price = float(t['price'])
                amount = float(t['amount'])
                cost = price * amount  # quote_volume
                
                bucket = market_data_store[symbol]["taker_stats_1h"]
                if hour_ts not in bucket:
                    bucket[hour_ts] = {
                        "taker_buy_volume": 0.0, "taker_sell_volume": 0.0,
                        "quote_volume": 0.0, "count": 0,
                        "whale_buy": 0.0, "whale_sell": 0.0, "retail_vol": 0.0
                    }
                
                stats = bucket[hour_ts]
                stats["quote_volume"] += cost
                stats["count"] += 1
                
                is_buy = t['side'] == 'buy'
                if is_buy:
                    stats["taker_buy_volume"] += amount
                    if amount >= WHALE_THRESHOLD:
                        stats["whale_buy"] += amount
                else:
                    stats["taker_sell_volume"] += amount
                    if amount >= WHALE_THRESHOLD:
                        stats["whale_sell"] += amount
                        
                if amount <= RETAIL_THRESHOLD:
                    stats["retail_vol"] += amount
            
            # 定期垃圾回收：保留最近 3 小时的字典即可
            current_hour = int(datetime.utcnow().timestamp() / 3600) * 3600000
            old_keys = [k for k in list(bucket.keys()) if k < current_hour - 504 * 3600000]
            for k in old_keys:
                del bucket[k]
                
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 逐笔成交超时 90 秒，触发重连！")
            for url, client in list(exchange.clients.items()):
                for f in list(client.futures.values()):
                    if isinstance(f, list):
                        for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                    else:
                        f.add_done_callback(lambda x: x.exception())
                client.reject(Exception(f"{symbol} 逐笔成交流假死重连"))
                exchange.clients.pop(url, None)
            was_dead = True
            await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(5)

async def mark_price_loop(exchange, symbol, swap_symbol):
    """ 监听标记价格流，提取溢价指数 (Premium Index) """
    print(f"📡 [WS] 启动 {symbol} 溢价指数订阅...")
    while True:
        try:
            # 币安 U 本位 WebSocket 原生包含 markPriceUpdate，CCXT 封装为 watch_mark_price
            mark_data = await asyncio.wait_for(
                exchange.watch_mark_price(swap_symbol),
                timeout=90.0
            )
            # info 中通常包含 'p' (Mark Price) 和 'i' (Index Price) 或者原生溢价率
            # 为了严谨对齐你离线里的 premium_close，我们保存为时序队列
            if 'info' in mark_data and 'r' in mark_data['info']:
                # 'r' 在 Binance raw payload 中代表 funding rate, 但我们也可以自己算溢价或者取历史API
                pass 
                
            # 退而求其次，直接用 REST 轮询 1h 的 premiumIndexKlines
            # 这里推荐直接放入之前的 derivatives_loop 进行低频轮询，见下一步合并
        except Exception as e:
            await asyncio.sleep(5)

async def derivatives_loop(exchange, symbol, swap_symbol, engine_type):
    print(f"📡 [REST] 启动 {symbol} {engine_type} 衍生品数据轮询...")
    await asyncio.sleep(random.uniform(5.0, 25.0))
    
    while True:
        try:
            async with rest_semaphore:
                fr_data = await exchange.fetch_funding_rate(swap_symbol)
            market_data_store[symbol]["funding_rate"] = float(fr_data.get('fundingRate', 0.0))
            
            async with rest_semaphore:
                oi_data = await exchange.fetch_open_interest(swap_symbol)
            market_data_store[symbol]["open_interest"] = float(oi_data.get('openInterestAmount', oi_data.get('baseVolume', 0.0)))
            
            # 新增：拉取溢价指数 K 线 (最近 2 根 1h)
            async with rest_semaphore:
                # CCXT 底层透传 Binance 专属 API
                premium_klines = await exchange.fapiPublicGetPremiumIndexKlines({
                    'symbol': exchange.market_id(swap_symbol),
                    'interval': '1h',
                    'limit': 2
                })
                # 将最后收盘价压入队列
                if premium_klines and len(premium_klines) > 0:
                    latest_premium_close = float(premium_klines[-1][4]) # index 4 is close
                    market_data_store[symbol]["premium_index"].append(latest_premium_close)
            
        except Exception as e:
            pass 
        await asyncio.sleep(60 + random.uniform(0, 5.0))

async def smart_warmup_and_recover(exchange, symbol, swap_symbol, is_spot):
    """ 智能补隙与预热引擎：查验本地库 -> 计算断层时间差 -> 增量拉取 -> 灌入内存 """
    print(f"🔍 [冷启动引擎] 正在校验 {symbol} 本地记忆库...")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM memory_1h WHERE symbol=?", (symbol,))
        last_ts = cursor.fetchone()[0]

    limit = 504 # 默认最大拉取 21 天 (覆盖 Z-score 168 + 模型 336)
    params = {}
    
    if last_ts:
        # 计算距离现在断层了多少个小时
        gap_hours = int((datetime.utcnow().timestamp() * 1000 - last_ts) / 3600000)
        if gap_hours <= 0:
            print(f"   ✅ {symbol} 本地数据已是最新，无缝启动。")
            limit = 0
        else:
            print(f"   ⚠️ {symbol} 发现数据断层，正在从 {datetime.fromtimestamp(last_ts/1000)} 追赶 {gap_hours} 小时数据...")
            params['startTime'] = last_ts + 1  # 严格从上一根之后开始接力
            limit = min(gap_hours + 1, 1000)   # 防止单次拉取击穿交易所上限

    if limit > 0:
        try:
            async with rest_semaphore:
                if not is_spot:
                    # 💡 主币种 (合约)：绕开 CCXT 标准壳，直击 Binance 底层 API，抠出 12 维量价密码
                    klines = await exchange.fapiPublicGetKlines({'symbol': exchange.market_id(swap_symbol), 'interval': '1h', 'limit': limit, **params})
                else:
                    # 辅币种 (现货)：只需基础 K 线，用标准 CCXT 足矣
                    klines = await exchange.fetch_ohlcv(swap_symbol, '1h', limit=limit, params=params)

            new_records = []
            for k in klines:
                ts = int(k[0])
                vol = float(k[5])
                
                # 现货没这些数据，统统 0 兜底；合约则精准剥离
                q_vol = float(k[7]) if not is_spot and len(k) > 7 else 0.0
                cnt = int(k[8]) if not is_spot and len(k) > 8 else 0
                tb_vol = float(k[9]) if not is_spot and len(k) > 9 else 0.0

                new_records.append((
                    symbol, ts,
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), vol,
                    q_vol, cnt, tb_vol, vol - tb_vol,
                    0.0, 0.0, vol * 0.1 # 预热阶段的大户/散户使用中性值过渡
                ))

            if new_records:
                with sqlite3.connect(DB_PATH) as conn:
                    # INSERT OR REPLACE 完美处理时间轴重叠问题
                    conn.executemany("INSERT OR REPLACE INTO memory_1h VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", new_records)
                    conn.commit()
                print(f"   💾 {symbol} 追赶完毕，已将 {len(new_records)} 根断层特征安全落库。")
                
        except Exception as e:
            print(f"   ❌ {symbol} 追赶拉取失败: {e}")

    # ================= 终极步骤：将本地无瑕疵的 504 根记录全量灌入内存 =================
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_1h WHERE symbol=? ORDER BY timestamp DESC LIMIT 504", (symbol,))
        rows = cursor.fetchall()

    rows.reverse() # 倒序，按时间先后顺序推入 deque

    for r in rows:
        ts = r[1]
        market_data_store[symbol]["1h"].append([ts, r[2], r[3], r[4], r[5], r[6]])
        if not is_spot:
            market_data_store[symbol]["taker_stats_1h"][ts] = {
                "quote_volume": r[7], "count": r[8],
                "taker_buy_volume": r[9], "taker_sell_volume": r[10],
                "whale_buy": r[11], "whale_sell": r[12], "retail_vol": r[13]
            }

async def db_persist_loop():
    """ 自动存档守护进程：将 WebSocket 即时揉出的微观流，每分钟刷入硬盘 """
    print("🛡️ 启动本地数据库持久化守护进程 (60秒自动存档)...")
    while True:
        await asyncio.sleep(60)
        try:
            records = []
            for sym, data in market_data_store.items():
                if len(data["1h"]) == 0: continue
                # 只抓取内存中最新的那一根（或正在聚合的那一根）
                k = data["1h"][-1]
                ts = k[0]
                stats = data.get("taker_stats_1h", {}).get(ts, {})

                records.append((
                    sym, ts,
                    k[1], k[2], k[3], k[4], k[5],
                    stats.get("quote_volume", 0.0), stats.get("count", 0),
                    stats.get("taker_buy_volume", 0.0), stats.get("taker_sell_volume", 0.0),
                    stats.get("whale_buy", 0.0), stats.get("whale_sell", 0.0), stats.get("retail_vol", 0.0)
                ))
            if records:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.executemany("INSERT OR REPLACE INTO memory_1h VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
                    conn.commit()
        except Exception:
            pass # 硬盘写入失败默默忽略，下个周期重试，不阻塞主线程
# ==========================================
# 🚑 5分钟健康看门狗巡检
# ==========================================
async def health_monitor_loop(all_symbols):
    print("🛡️ 启动全局 WebSocket 健康巡检看门狗 (每 5 分钟汇报一次)...")
    while True:
        await asyncio.sleep(300)
        print("\n" + "="*50)
        print(f"🕒 [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 极速网关健康巡检报告:")
        all_healthy = True
        
        for sym in all_symbols:
            data = market_data_store.get(sym)
            if not data or data.get("status") == "unsupported": continue
            
            # 以 Ticker 是否在跳动作为最直观的存活指标 (超过 120 秒未跳动视为异常)
            # 或者以 K 线的时间戳衡量
            latest_kline_ts = data["1h"][-1][0] if len(data["1h"]) > 0 else 0
            gap_seconds = (time.time() * 1000 - latest_kline_ts) / 1000
            
            # K 线至少应该在 2 小时以内 (1小时周期)，我们放宽到 7200 秒
            if gap_seconds > 7200:
                print(f"   ❌ {sym} 状态异常：数据严重滞后或断流 (滞后 {gap_seconds/3600:.1f} 小时)！")
                all_healthy = False
            else:
                last_price = data['ticker'].get('last', '等待中')
                print(f"   ✅ {sym} 全局数据流同步正常 | 实时监控价: {last_price}")
                
        if all_healthy:
            print("   🟢 极好！所有核心/辅助币种的 WebSocket 订阅队列均运行完美，无假死断流。")
        print("="*50 + "\n")

# ==========================================
# 🚀 网关初始化与多任务编排
# ==========================================
async def start_gateways():
    futures_ex = ccxtpro.binanceusdm()
    spot_ex = ccxtpro.binance()
    
    try:
        print("⏳ 正在拉取 Binance 市场列表，进行合约与现货合法性校验...")
        try:
            await asyncio.gather(futures_ex.load_markets(), spot_ex.load_markets())
        except Exception as e:
            print(f"🛑 无法连接到交易所获取市场列表: {e}")
            return
            
        tasks = []
        tasks.append(db_persist_loop())
        tasks.append(health_monitor_loop(all_symbols))
        
        # 强制串行延迟，根治 Binance 5条/秒 订阅速率限制导致的并发静默丢包
        startup_delay = 0.0 
        
        for sym in all_symbols:
            # 简化解析逻辑：直接加斜杠
            if "/" not in sym:
                quote = "USDT" if sym.endswith("USDT") else "BTC"
                sym_standard = f"{sym[:-len(quote)]}/{quote}"
            else:
                sym_standard = sym

            # 自动分配引擎
            is_spot = (sym in AUX_SYMBOLS)
            ex = spot_ex if is_spot else futures_ex
            
            # --- 核心修改：定义并传递引擎类型用于日志打印 ---
            engine_type = "现货" if is_spot else "合约"
            swap_sym = f"{sym_standard}:{sym_standard.split('/')[1]}" if not is_spot else sym_standard
            
            # 市场存在性强校验
            if swap_sym not in ex.markets:
                print(f"   ⚠️ [物理拦截] 交易所未上架 {swap_sym}，已自动跳过！")
                market_data_store[sym]["status"] = "unsupported"
                continue
            await smart_warmup_and_recover(ex, sym, swap_sym, is_spot)    
            
            # 主辅币种都需要基础 K 线和 Ticker，带上 engine_type 和严格间隔的 startup_delay
            tasks.append(ohlcv_loop(ex, sym, swap_sym, "1h", engine_type, startup_delay))
            startup_delay += 0.5
            tasks.append(ticker_loop(ex, sym, swap_sym, engine_type, startup_delay))
            startup_delay += 0.5
            
            # 只有主币种才订阅深度、爆仓、逐笔成交和衍生品
            if not is_spot:
                tasks.append(ohlcv_loop(ex, sym, swap_sym, "1m", engine_type, startup_delay))
                startup_delay += 0.5
                tasks.append(orderbook_loop(ex, sym, swap_sym, engine_type, startup_delay))
                startup_delay += 0.5
                tasks.append(liquidation_loop(ex, sym, swap_sym, engine_type, startup_delay))  
                startup_delay += 0.5
                tasks.append(trade_loop(ex, sym, swap_sym, engine_type, startup_delay))        
                startup_delay += 0.5
                tasks.append(derivatives_loop(ex, sym, swap_sym, engine_type))
            
            # 不要额外的 asyncio.sleep 了，已经通过 startup_delay 排好队了
        
        # 统计所有成功存活的币种
        active_count = len([s for s in all_symbols if market_data_store[s]["status"] != "unsupported"])
        print(f"🔥 行情网关校验完毕，正在维护 {active_count} 个存活币种的全量切片...")
        
        if tasks:
            # 并发执行所有监听协程
            await asyncio.gather(*tasks)
            
    finally:
        # 全局资源兜底释放，优雅切断 aiohttp 连接
        await asyncio.gather(futures_ex.close(), spot_ex.close())

# ==========================================
# 🔌 本地微服务接口：供 live_pipeline.py 秒级调用
# ==========================================
@app.get("/api/snapshot/{symbol}")
async def get_market_snapshot(symbol: str):
    """ 核心接口：返回该币种在内存中缓存的所有特征切片 """
    # 优先检查原样，兼容全字母（如 BTCUSDT）
    if symbol in market_data_store:
        clean_sym = symbol
    else:
        # 处理网关格式，兼容前端传来的格式转换
        clean_sym = symbol if "/" in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
        if clean_sym not in market_data_store:
            # 尝试匹配 BTC 等 3 字母结尾的计价币
            clean_sym = symbol if "/" in symbol else f"{symbol[:-3]}/{symbol[-3:]}"
            if clean_sym not in market_data_store:
                raise HTTPException(status_code=404, detail=f"币种不在监控池内: {symbol}")
    
    data = market_data_store[clean_sym]
    
    if data["status"] == "unsupported":
         raise HTTPException(status_code=400, detail="交易所不支持该币种")
         
    # 物理销毁拦截：如果 K 线还没推过来，拒绝返回脏数据
    if not data["1h"] or data["ticker"]["last"] is None:
        raise HTTPException(status_code=503, detail="数据尚未预热完成")

    # 返回给推理端进行二次对齐的原始纯净时序
    response = {
        "symbol": clean_sym,
        "current_price": data["ticker"]["last"],
        "ohlcv_1h": list(data["1h"]),
        "server_time": datetime.utcnow().isoformat()
    }
    
    # 只有主币种附加深度微观特征
    if clean_sym == SYMBOL or clean_sym.replace("/", "") == SYMBOL:
        response.update({
            "taker_stats_1h": data.get("taker_stats_1h", {}),
            "premium_index": list(data.get("premium_index", [])),
            "funding_rate": data.get("funding_rate", 0.0),
        })

    return response

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    try:
        asyncio.run(start_gateways())
    except KeyboardInterrupt:
        print("🛑 网关已手动关闭")