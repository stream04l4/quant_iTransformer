import asyncio
import ccxt.pro as ccxtpro
from collections import deque
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import random
from datetime import datetime

from quant_config.config import TRACKED_SYMBOLS

app = FastAPI(title="TimesFM 极速行情网关")

# ==========================================
# 🛑 全局并发限流阀 (防止冷启动惊群效应)
# ==========================================
# 强制规定：同一时刻最多只允许 3 个 REST 请求发往币安，保护本地 Socket 与币安 WAF
rest_semaphore = asyncio.Semaphore(3)

# ==========================================
# 🧠 内存状态机：行情数据的物理插槽
# ==========================================
market_data_store = {}

for sym in TRACKED_SYMBOLS:
    market_data_store[sym] = {
        # 🚨 极简重构：只保留模型需要的 1h 级别 K 线，长度至少覆盖 SEQ_LEN + 生成指标所需缓冲
        # 336 (序列长度) + 100 (计算 RSI/MACD 等需要的历史缓冲) = 450 即可，我们稳妥给 500
        "1m": deque(maxlen=120),
        "1h": deque(maxlen=500), 
        
        # 订单薄、爆仓、主动吃单依然保留，因为我们可以用它们构造 1h 内的高频特征
        "orderbook": {"bids": [], "asks": []},
        "ticker": {"last": None},
        "liquidations": deque(maxlen=1000), 
        "taker_vols": {},                   
        "funding_rate": 0.0,                
        "open_interest": 0.0,               
        "status": "initializing"
    }

# ==========================================
# 🛰️ WebSocket 搬运工：死循环监听流
# ==========================================
async def ohlcv_loop(exchange, symbol, swap_symbol, timeframe):
    """ 监听 K 线流并维护内存队列 (具备历史预热与同频去重功能) """
    print(f"📡 [WS] 启动 {symbol} ({timeframe}) K线订阅...")
    
    # 👇 【核心修复 1：冷启动历史预热 (加入防爆破错峰与重试机制)】
    limit_map = {"1m": 120, "1h": 600}
    limit = limit_map.get(timeframe, 100)
    
    # 👇 [修复 1：扩大初始错峰范围，从 3.0 秒放大到 15.0 秒，让大军缓慢进场]
    await asyncio.sleep(random.uniform(0.1, 15.0))
    
    # 移入外层循环：支持假死后强制极速预热重连，完美缝合时间断层 (Time Hole)
    while True:
        for attempt in range(3):
            try:
                # 👇 [修复 2：核心！用限流阀包裹 CCXT 的 REST 请求]
                async with rest_semaphore:
                    historical_data = await exchange.fetch_ohlcv(swap_symbol, timeframe, limit=limit)
                for candle in historical_data:
                    market_data_store[symbol][timeframe].append(candle)
                break # 成功拉取，跳出重试循环
            except Exception as e:
                print(f"   ⚠️ [预热重试 {attempt+1}/3] {symbol} {timeframe} 拉取异常: {e}")
                # 指数退避：失败后分别等待 2秒、4秒、8秒
                await asyncio.sleep(2 ** (attempt + 1))

        # 👇 预热完毕后，正式进入 WebSocket 实时监听内层心跳流
        while True:
            try:
                # 🛡️ 【看门狗防线】：90秒收不到任何 K 线跳动，强制视为假死并抛出异常
                ohlcvs = await asyncio.wait_for(
                    exchange.watch_ohlcv(swap_symbol, timeframe),
                    timeout=90.0
                )
                for candle in ohlcvs:
                    q = market_data_store[symbol][timeframe]
                    
                    # 同频更新去重
                    if len(q) > 0 and q[-1][0] == candle[0]:
                        q[-1] = candle
                    else:
                        q.append(candle)
                        
            except asyncio.TimeoutError:
                # 🚨 打破内圈死循环跳到外圈：先同步缝合所有的落后数据再监听流
                print(f"   ⚠️ [WS 看门狗] {symbol} ({timeframe}) 超过 90 秒假死，触发硬核全序列补偿同步！")
                await asyncio.sleep(2)
                break
            except Exception as e:
                print(f"   ❌ [WS 异常] {symbol} {timeframe}: {e}")
                await asyncio.sleep(5)
                break

async def orderbook_loop(exchange, symbol, swap_symbol):
    """ 监听盘口流 (增量自动合并) """
    print(f"📡 [WS] 启动 {symbol} 盘口深度订阅...")
    while True:
        try:
            # 🛡️ 【看门狗防线】：90秒收不到盘口变动，强制视为假死
            orderbook = await asyncio.wait_for(
                exchange.watch_order_book(swap_symbol, limit=100),
                timeout=90.0
            )
            market_data_store[symbol]["orderbook"] = {
                "bids": orderbook['bids'],
                "asks": orderbook['asks']
            }
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 盘口深度超过 90 秒无变动，触发假死重连！")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"   ❌ [WS 盘口异常] {symbol}: {e}")
            await asyncio.sleep(5)

async def ticker_loop(exchange, symbol, swap_symbol):
    """ 监听最新价流 """
    while True:
        try:
            # 🛡️ 【看门狗防线】：90秒收不到最新价跳动，强制视为假死
            ticker = await asyncio.wait_for(
                exchange.watch_ticker(swap_symbol),
                timeout=90.0
            )
            market_data_store[symbol]["ticker"]["last"] = ticker['last']
            market_data_store[symbol]["status"] = "live"
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 最新价超过 90 秒无跳动，触发假死重连！")
            await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(5)
async def liquidation_loop(exchange, symbol, swap_symbol):
    """ 监听币安实时爆仓流 (CCXT 强力防呆兼容版) """
    print(f"📡 [WS] 启动 {symbol} 爆仓流订阅...")
    
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

async def trade_loop(exchange, symbol, swap_symbol):
    """ 监听逐笔成交流并执行“时间桶”轻量聚合 (防内存爆炸) """
    print(f"📡 [WS] 启动 {symbol} 逐笔成交流订阅...")
    while True:
        try:
            # 🛡️ 逐笔成交非常频繁，设置 90 秒看门狗
            trades = await asyncio.wait_for(
                exchange.watch_trades(swap_symbol),
                timeout=90.0
            )
            for t in trades:
                # 毫秒级降维：将时间戳向下取整到“分钟桶”
                minute_ts = int(t['timestamp'] / 60000) * 60000
                cost = float(t['price']) * float(t['amount'])
                
                bucket = market_data_store[symbol]["taker_vols"]
                if minute_ts not in bucket:
                    bucket[minute_ts] = {"buy": 0.0, "sell": 0.0}
                
                # CCXT 的 side 就是吃单方 (Taker) 的方向
                if t['side'] == 'buy':
                    bucket[minute_ts]["buy"] += cost
                elif t['side'] == 'sell':
                    bucket[minute_ts]["sell"] += cost
            
            # 定期垃圾回收：只保留最近 20 分钟的数据桶，其余无情销毁
            current_min = int(datetime.utcnow().timestamp() / 60) * 60000
            old_keys = [k for k in list(bucket.keys()) if k < current_min - 20 * 60000]
            for k in old_keys:
                del bucket[k]
                
        except asyncio.TimeoutError:
            print(f"   ⚠️ [WS 看门狗] {symbol} 逐笔成交超时 90 秒，触发重连！")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"   ❌ [WS 逐笔异常] {symbol}: {e}")
            await asyncio.sleep(5)

async def derivatives_loop(exchange, symbol, swap_symbol):
    """ 独立轮询衍生品数据 (资金费率 & 持仓量) """
    print(f"📡 [REST] 启动 {symbol} 衍生品数据轮询...")
    
    # 👇 [修复 3：扩大衍生品启动错峰]
    await asyncio.sleep(random.uniform(5.0, 25.0))
    
    while True:
        try:
            # 👇 [修复 4：核心！用限流阀包裹请求]
            async with rest_semaphore:
                fr_data = await exchange.fetch_funding_rate(swap_symbol)
            market_data_store[symbol]["funding_rate"] = float(fr_data.get('fundingRate', 0.0))
            
            # 👇 [修复 5：独立限流获取持仓量]
            async with rest_semaphore:
                oi_data = await exchange.fetch_open_interest(swap_symbol)
            market_data_store[symbol]["open_interest"] = float(oi_data.get('openInterestAmount', oi_data.get('baseVolume', 0.0)))
            
        except Exception as e:
            pass # 静默处理，下一轮重试
            
        await asyncio.sleep(60 + random.uniform(0, 5.0))
# ==========================================
# 🚀 网关初始化与多任务编排
# ==========================================
async def start_gateways():
    # 👇 【核心升级 1】：改用币安 U 本位合约 (binanceusdm) 作为主数据源，覆盖所有热门 Meme
    exchange = ccxtpro.binanceusdm()
    
    print("⏳ 正在拉取 Binance 市场列表，进行合约合法性校验...")
    try:
        await exchange.load_markets()
    except Exception as e:
        print(f"🛑 无法连接到交易所获取市场列表: {e}")
        return
        
    tasks = []
    for sym in TRACKED_SYMBOLS:
        # 👇 【核心升级 2】：强制转换为 CCXT 永续合约标准格式 (如 BTC/USDT:USDT)
        swap_sym = f"{sym}:{sym.split('/')[1]}"
        
        # 👇 【核心升级 3】：市场存在性强校验
        if swap_sym not in exchange.markets:
            print(f"   ⚠️ [物理拦截] 交易所未上架 {swap_sym} 合约，已自动跳过！")
            market_data_store[sym]["status"] = "unsupported"
            continue
            
        # 🚨 极简重构：移除所有多余的时间框架，只死死盯住 1 小时级别
        tasks.append(ohlcv_loop(exchange, sym, swap_sym, "1m"))
        tasks.append(ohlcv_loop(exchange, sym, swap_sym, "1h"))
        
        tasks.append(orderbook_loop(exchange, sym, swap_sym))
        tasks.append(ticker_loop(exchange, sym, swap_sym))
        tasks.append(liquidation_loop(exchange, sym, swap_sym))  
        tasks.append(trade_loop(exchange, sym, swap_sym))        
        tasks.append(derivatives_loop(exchange, sym, swap_sym))
        
        # 💡 【并发防抖】：每分配完一个币种的 5 个频道，强制微小休眠 0.1 秒
        # 防止 120 个连接瞬间砸向币安导致 IP 被短暂拉黑
        await asyncio.sleep(0.1) 
    
    active_count = len([s for s in TRACKED_SYMBOLS if market_data_store[s]["status"] != "unsupported"])
    print(f"🔥 行情网关校验完毕，正在维护 {active_count} 个存活币种的全量切片...")
    
    if tasks:
        # 并发执行所有监听协程
        await asyncio.gather(*tasks)

# ==========================================
# 🔌 本地微服务接口：供 engine.py 秒级调用
# ==========================================
@app.get("/api/snapshot/{symbol}")
async def get_market_snapshot(symbol: str):
    """ 核心接口：返回该币种在内存中缓存的所有特征切片 """
    clean_sym = symbol.replace("_", "/")
    if clean_sym not in market_data_store:
        raise HTTPException(status_code=404, detail="币种不在监控池内")
    
    data = market_data_store[clean_sym]
    
    if data["status"] == "unsupported":
         raise HTTPException(status_code=400, detail="交易所不支持该币种")
         
    # 物理销毁拦截：如果 K 线还没推过来，拒绝返回脏数据
    if not data["1h"] or data["ticker"]["last"] is None:
        raise HTTPException(status_code=503, detail="数据尚未预热完成")

    # 👇 计算 15 分钟滚动视角的微观数据
    cutoff_ts = int(datetime.utcnow().timestamp() * 1000) - 900000
    
    # 1. 净爆仓计算
    net_liq_vol = 0.0
    for liq in list(data["liquidations"]): # list() 确保线程安全
        if liq["ts"] >= cutoff_ts:
            # 🚨 修复：在币安中，爆仓推送的 side 是“强平订单的方向”
            # side == 'sell' 代表系统强制挂卖单，说明是【多头被爆 (杀多)】，应记为负的抛压
            # side == 'buy'  代表系统强制挂买单，说明是【空头被爆 (逼空)】，应记为正的买盘推力
            if liq["side"] == "sell": net_liq_vol -= liq["usd_value"]  # 杀多，记负
            elif liq["side"] == "buy": net_liq_vol += liq["usd_value"] # 逼空，记正
            
    # 2. 主动买卖比计算
    buy_vol, sell_vol = 0.0, 0.0
    for ts_key, vols in list(data["taker_vols"].items()):
        if ts_key >= cutoff_ts:
            buy_vol += vols["buy"]
            sell_vol += vols["sell"]
    # 🚨 修复：防止极端情况下 (只有买单没有卖单) 导致买卖比退化为 1.0
    taker_ratio = buy_vol / max(sell_vol, 1.0) if (buy_vol + sell_vol) > 0 else 1.0

    # 返回纯净的切片快照
    return {
        "symbol": clean_sym,
        "current_price": data["ticker"]["last"],
        "ohlcv_1m": list(data["1m"]),
        "ohlcv_1h": list(data["1h"]),           # 模型唯一依赖的主轴数据
        "orderbook": data["orderbook"],
        "taker_buy_ratio_15m": taker_ratio,   
        "liquidation_vol_15m": net_liq_vol,   
        "funding_rate": data["funding_rate"],   
        "open_interest": data["open_interest"], 
        "server_time": datetime.utcnow().isoformat()
    }

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    try:
        asyncio.run(start_gateways())
    except KeyboardInterrupt:
        print("🛑 网关已手动关闭")