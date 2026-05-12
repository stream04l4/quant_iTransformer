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
        "1m": deque(maxlen=35000), # 增加长度以支撑 1m RV 计算
        "1h": deque(maxlen=1000), # 从 500 提升到 1000，确保满足 504 需求
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
                timeout=180.0
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
                    
        except Exception as e:
            is_timeout = isinstance(e, asyncio.TimeoutError)
            err_str = str(e).lower()
            
            if "假死重连" in str(e):
                # 兄弟线程已经把共享的 websocket 砸了重建，我们只需要默默跟随重连即可，不要大声喧哗
                was_dead = True
                await asyncio.sleep(1)
                continue
                
            if is_timeout or "timeout" in err_str or "network" in err_str or "closed" in err_str:
                if is_timeout:
                    print(f"   ⚠️ [WS 看门狗] {symbol} ({timeframe}) 超过 180 秒无变动，触发 WebSocket 重连！")
                else:
                    print(f"   ⚠️ [WS 断线] {symbol} ({timeframe}) 连接异常: {e}，触发底舱清理与重连！")
                
                # 强制清理底层死锁的 WS 客户端，促使下次调用重新发 SUBSCRIBE 请求
                for url, client in list(exchange.clients.items()):
                    for f in list(client.futures.values()):
                        if isinstance(f, list):
                            for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                        else:
                            f.add_done_callback(lambda x: x.exception())
                    client.reject(Exception(f"底层引擎触发假死重连"))
                    exchange.clients.pop(url, None)
                was_dead = True
                await asyncio.sleep(2)
            else:
                print(f"   ❌ [WS 未知异常] {symbol} {timeframe}: {type(e).__name__} -> {e}")
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
                timeout=180.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}盘口深度流已成功恢复！")
                was_dead = False
                
            market_data_store[symbol]["orderbook"] = {
                "bids": orderbook['bids'],
                "asks": orderbook['asks']
            }
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 盘口深度超过 180 秒无变动，触发假死重连！")
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
                timeout=180.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}最新价流已成功恢复！")
                was_dead = False
                
            market_data_store[symbol]["ticker"]["last"] = ticker['last']
            market_data_store[symbol]["status"] = "live"
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 最新价超过 180 秒无跳动，触发假死重连！")
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
                timeout=180.0
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
            print(f"   ⚠️ [WS 看门狗] {symbol} 逐笔成交超时 180 秒，触发重连！")
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
                timeout=180.0
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
    print(f"📡 [REST] 启动 {symbol} {engine_type} 衍生品数据轮询 (维护底层时序数组)...")
    await asyncio.sleep(random.uniform(5.0, 25.0))
    
    raw_sym = exchange.market_id(swap_symbol)

    while True:
        try:
            # 1. 维护 1h 溢价指数时序 (Premium Index)
            async with rest_semaphore:
                prem_klines = await exchange.fapiPublicGetPremiumIndexKlines({
                    'symbol': raw_sym, 'interval': '1h', 'limit': 2
                })
            if prem_klines:
                hist = market_data_store[symbol].setdefault("premium_history", [])
                for k in prem_klines:
                    ts, val = int(k[0]), float(k[4]) # 4 is close
                    if not hist or ts > hist[-1][0]:
                        hist.append([ts, val])
                    elif ts == hist[-1][0]:
                        hist[-1] = [ts, val]

            # 2. 维护 1h 未平仓合约量时序 (Open Interest)
            async with rest_semaphore:
                oi_hist_data = await exchange.fapiDataGetOpenInterestHist({
                    'symbol': raw_sym, 'period': '1h', 'limit': 2
                })
            if oi_hist_data:
                hist = market_data_store[symbol].setdefault("oi_history", [])
                for x in oi_hist_data:
                    ts, val = int(x['timestamp']), float(x['sumOpenInterest'])
                    if not hist or ts > hist[-1][0]:
                        hist.append([ts, val])
                    elif ts == hist[-1][0]:
                        hist[-1] = [ts, val]

            # 3. 维护 1h 大户多空比时序 (Long/Short Ratio)
            async with rest_semaphore:
                ls_hist_data = await exchange.fapiDataGetTopLongShortPositionRatio({
                    'symbol': raw_sym, 'period': '1h', 'limit': 2
                })
            if ls_hist_data:
                hist = market_data_store[symbol].setdefault("ls_history", [])
                for x in ls_hist_data:
                    ts, val = int(x['timestamp']), float(x['longShortRatio'])
                    if not hist or ts > hist[-1][0]:
                        hist.append([ts, val])
                    elif ts == hist[-1][0]:
                        hist[-1] = [ts, val]

            # 4. 维护资金费率时序 (Funding Rate)
            async with rest_semaphore:
                fr_hist_data = await exchange.fapiPublicGetFundingRate({
                    'symbol': raw_sym, 'limit': 2
                })
            if fr_hist_data:
                hist = market_data_store[symbol].setdefault("funding_history", [])
                for x in fr_hist_data:
                    ts, val = int(x['fundingTime']), float(x['fundingRate'])
                    if not hist or ts > hist[-1][0]:
                        hist.append([ts, val])
                    elif ts == hist[-1][0]:
                        hist[-1] = [ts, val]

            # ==========================================
            # 🧹 内存泄露防护：控制历史数组的最大长度 (对齐历史预热的 500)
            # ==========================================
            for key in ["premium_history", "oi_history", "ls_history", "funding_history"]:
                if key in market_data_store[symbol] and len(market_data_store[symbol][key]) > 600:
                    # 裁剪头部冗余数据，保留最新 600 个时间节点
                    market_data_store[symbol][key] = market_data_store[symbol][key][-600:]

        except Exception as e:
            # 去除之前的 pass，将网络抖动和底层 API 拒载显式抛出
            print(f"   ⚠️ [REST 衍生品轮询异常] {symbol} 发生级联错误: {type(e).__name__} -> {e}")
            
        # 轮询抖动设计：防止触发 Binance 每分钟请求权重超限
        await asyncio.sleep(60 + random.uniform(0, 5.0))

async def smart_warmup_and_recover(exchange, symbol, swap_symbol, is_spot):
    """
    自愈式预热：检查本地库是否存在时间空隙 (Gaps)，发现即重拉，绝不补0。
    """
    print(f"🔍 [网关自愈] 正在校验 {symbol} 历史数据的连续性...")
    
    with sqlite3.connect(DB_PATH) as conn:
        # 获取最近 1000 根 K 线的时间戳
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM memory_1h WHERE symbol=? ORDER BY timestamp DESC LIMIT 1000", (symbol,))
        ts_list = [r[0] for r in cursor.fetchall()]

    if len(ts_list) < 2:
        # 初次启动，拉取全量
        print(f"   🆕 {symbol} 本地无数据，执行全量预热...")
        await fetch_and_store_range(exchange, symbol, swap_symbol, is_spot, limit=1000)
    else:
        # 检查时间轴空隙
        ts_list.reverse()
        gaps = []
        for i in range(len(ts_list) - 1):
            if ts_list[i+1] - ts_list[i] != 3600000: # 1h 差值应为 3600000ms
                gaps.append((ts_list[i], ts_list[i+1]))
        
        if gaps:
            print(f"   ⚠️ {symbol} 侦测到 {len(gaps)} 处数据空隙，正在强制回填物理数据...")
            for start, end in gaps:
                # 针对每一处空隙，精准拉取
                await fetch_and_store_range(exchange, symbol, swap_symbol, is_spot, startTime=start+1, limit=500)
        else:
            print(f"   ✅ {symbol} 时间轴完美连续。")

# ================= 终极步骤：将完美的 1000 根数据灌入内存 =================
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_1h WHERE symbol=? ORDER BY timestamp DESC LIMIT 1000", (symbol,))
        rows = cursor.fetchall()

    rows.reverse()
    for r in rows:
        ts = r[1]
        market_data_store[symbol]["1h"].append([ts, r[2], r[3], r[4], r[5], r[6]])
        if not is_spot:
            market_data_store[symbol]["taker_stats_1h"][ts] = {
                "quote_volume": r[7], "count": r[8],
                "taker_buy_volume": r[9], "taker_sell_volume": r[10],
                "whale_buy": r[11], "whale_sell": r[12], "retail_vol": r[13]
            }
    # === 新增：网关包揽 1m K线与所有衍生品的历史时序拉取 ===
    if not is_spot:
        print(f"🔍 [网关自愈] 正在拉取 {symbol} 衍生品时序与 1m 显微镜底座...")
        try:
            print(f"🔍 [网关自愈] 正在全量拉取 {symbol} 过去 504 小时的 1m 显微数据 (约3万根)...")
            end_time = int(time.time() * 1000)
            start_time = end_time - 505 * 3600 * 1000 # 多拉 1 小时容错
            all_1m = []
            curr_ts = start_time

            while curr_ts < end_time:
                batch = await exchange.fetch_ohlcv(swap_symbol, '1m', since=curr_ts, limit=1500)
                if not batch:
                    break
                all_1m.extend(batch)
                curr_ts = batch[-1][0] + 60000 # 步进 1 分钟
                await asyncio.sleep(0.1) # 防护 API 速率限制

            market_data_store[symbol]["1m"].clear()
            market_data_store[symbol]["1m"].extend(all_1m)
            print(f"   ✅ {symbol} 1m 高频数据装填完毕，共 {len(all_1m)} 根。")

            raw_sym = exchange.market_id(swap_symbol)

            # 2. 补齐溢价指数时序 (500小时)
            prem_klines = await exchange.fapiPublicGetPremiumIndexKlines({'symbol': raw_sym, 'interval': '1h', 'limit': 500})
            market_data_store[symbol]["premium_history"] = [[int(k[0]), float(k[4])] for k in prem_klines]

            # 3. 补齐资金费率时序 (500个节点)
            fr_hist = await exchange.fapiPublicGetFundingRate({'symbol': raw_sym, 'limit': 500})
            market_data_store[symbol]["funding_history"] = [[int(x['fundingTime']), float(x['fundingRate'])] for x in fr_hist]

            # 4. 补齐 Open Interest 时序 (500小时)
            oi_hist = await exchange.fapiDataGetOpenInterestHist({'symbol': raw_sym, 'period': '1h', 'limit': 500})
            market_data_store[symbol]["oi_history"] = [[int(x['timestamp']), float(x['sumOpenInterest'])] for x in oi_hist]

            # 5. 补齐大户多空比时序 (500小时)
            ls_hist = await exchange.fapiDataGetTopLongShortPositionRatio({'symbol': raw_sym, 'period': '1h', 'limit': 500})
            market_data_store[symbol]["ls_history"] = [[int(x['timestamp']), float(x['longShortRatio'])] for x in ls_hist]
            
        except Exception as e:
            print(f"⚠️ {symbol} 衍生品时序拉取异常: {e}")
async def fetch_and_store_range(exchange, symbol, swap_symbol, is_spot, startTime=None, limit=500):
    """通用拉取并落库逻辑，直接从交易所获取物理真实 K 线"""
    params = {'startTime': startTime} if startTime else {}
    try:
        async with rest_semaphore:
            if not is_spot:
                # 合约使用 fapi 获取全维度数据
                klines = await exchange.fapiPublicGetKlines({'symbol': exchange.market_id(swap_symbol), 'interval': '1h', 'limit': limit, **params})
            else:
                klines = await exchange.fetch_ohlcv(swap_symbol, '1h', params=params)
        
        new_records = []
        for k in klines:
            # 这里的 index 严格对应你的 memory_1h 表结构
            # 绝不手动填充 0，直接使用交易所返回的物理值
            vol = float(k[5])
            new_records.append((
                symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), vol,
                float(k[7]) if len(k)>7 else 0.0, # quote_volume
                int(k[8]) if len(k)>8 else 0,     # count
                float(k[9]) if len(k)>9 else 0.0, # taker_buy
                vol - float(k[9]) if len(k)>9 else 0.0,
                0.0, 0.0, vol * 0.1 # 权重过渡
            ))
        
        if new_records:
            with sqlite3.connect(DB_PATH) as conn:
                conn.executemany("INSERT OR REPLACE INTO memory_1h VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", new_records)
                conn.commit()
    except Exception as e:
        print(f"   ❌ {symbol} 物理回填失败: {e}")

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
            "ohlcv_1m": list(data.get("1m", [])),
            "premium_history": data.get("premium_history", []),
            "funding_history": data.get("funding_history", []),
            "oi_history": data.get("oi_history", []),
            "ls_history": data.get("ls_history", [])
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