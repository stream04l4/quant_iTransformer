import os
import time
import json
import sqlite3
import logging
import requests
import re
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from quant_config.config import DATA_DIR, LOG_DIR

# 👇 激活 .env 文件读取
load_dotenv()

# ==========================================
# ⚙️ 哨兵全局配置与日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Sentinel] %(message)s',
    handlers=[logging.StreamHandler()]
)

DB_NAME = os.path.join(DATA_DIR, "macro_intelligence.db")

def init_db():
    """初始化宏观情报数据库"""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_ts TIMESTAMP,
                context_json TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_ts TIMESTAMP,
                score REAL,
                marquee_text TEXT,
                urgency TEXT,
                raw_context_id INTEGER
            )
        ''')
        conn.commit()

# ==========================================
# 📡 维度一：主流加密新闻聚合 (纯开源 RSS)
# ==========================================
import xml.etree.ElementTree as ET

class NewsScraper:
    def __init__(self):
        self.rss_urls = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]

    def fetch(self):
        logging.info("正在通过开源 RSS 直连抓取全网加密新闻...")
        news_items = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        for url in self.rss_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    for item in root.findall('./channel/item')[:5]: 
                        title = item.find('title')
                        pub_date = item.find('pubDate')
                        if title is not None:
                            news_items.append({
                                "title": title.text,
                                "source": "CoinTelegraph" if "cointelegraph" in url else "CoinDesk",
                                "time": pub_date.text if pub_date is not None else ""
                            })
                else:
                    logging.warning(f"RSS 源 {url} 抓取失败，状态码: {response.status_code}")
            except Exception as e:
                logging.error(f"RSS 解析异常 ({url}): {e}")

        return news_items

# ==========================================
# 📡 维度二：大盘散户情绪 (Alternative.me)
# ==========================================
class SentimentIndexScraper:
    def __init__(self):
        self.url = "https://api.alternative.me/fng/"

    def fetch(self):
        logging.info("正在抓取贪婪与恐慌指数...")
        try:
            response = requests.get(self.url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                index_data = data.get('data', [{}])[0]
                return {
                    "fetch_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), # 👈 新增时间戳
                    "value": int(index_data.get('value', 50)),
                    "classification": index_data.get('value_classification', 'Neutral')
                }
            return None
        except Exception as e:
            logging.error(f"SentimentIndexScraper 异常: {e}")
            return None

# ==========================================
# 📡 维度三：真金白银多空比 (Binance)
# ==========================================
class MarketPositionScraper:
    def __init__(self):
        self.url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"

    def fetch(self, symbol="BTCUSDT", period="1h"):
        logging.info(f"正在抓取 {symbol} 合约大户多空比...")
        try:
            params = {"symbol": symbol, "period": period, "limit": 1}
            response = requests.get(self.url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "fetch_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), # 👈 新增时间戳
                        "symbol": symbol,
                        "long_short_ratio": float(data[0].get('longShortRatio', 1.0))
                    }
            return None
        except Exception as e:
            logging.error(f"MarketPositionScraper 异常: {e}")
            return None

# ==========================================
# 📡 维度四：传统宏观金融 (Yahoo Finance)
# ==========================================
class TradFiScraper:
    def __init__(self):
        # NQ=F (纳指100), DX-Y.NYB (美元指数), ^TNX (10年期美债收益率)
        self.tickers = {
            "Nasdaq_Futures": "NQ=F", 
            "US_Dollar_Index": "DX-Y.NYB",
            "US_10Y_Treasury": "^TNX"
        }

    def fetch(self):
        logging.info("正在抓取传统宏观资产实时序列 (启动高频缝合机制)...")
        macro_data = {}
        try:
            for name, symbol in self.tickers.items():
                ticker = yf.Ticker(symbol)
                
                # 👇 【核心提速 1】：将基础序列从 1h 降维到 15m，大幅缩小 K 线闭合等待期
                hist = ticker.history(period="5d", interval="15m")
                
                if len(hist) >= 2:
                    # 👇 【核心提速 2】：强制拉取当前最新 1 分钟的 Tick（现价）
                    try:
                        live_tick = ticker.history(period="1d", interval="1m").tail(1)
                        # 如果最新 1 分钟的时间戳比 15 分钟的末尾还要新，直接将它追加到末尾缝合！
                        if not live_tick.empty and live_tick.index[0] > hist.index[-1]:
                            import pandas as pd
                            hist = pd.concat([hist, live_tick])
                    except Exception as e:
                        logging.warning(f"  ⚠️ {symbol} 极速 Tick 缝合失败，回退至基础 K 线: {e}")

                    # 截取最后 8 根 K 线 (由于是 15m，这刚好代表了最近 2 小时的密集微观轨迹)
                    recent_bars = hist.tail(8)
                    trend_data = []
                    
                    for idx, row in recent_bars.iterrows():
                        trend_data.append({
                            "time": (idx.tz_convert('UTC') if getattr(idx, 'tzinfo', None) else idx).strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "close": round(float(row['Close']), 3)
                        })
                    
                    # 计算这 2 小时跨度内的总体动能变化率
                    first_close = float(recent_bars['Close'].iloc[0])
                    last_close = float(recent_bars['Close'].iloc[-1])
                    overall_delta_pct = (last_close - first_close) / first_close
                    
                    macro_data[name] = {
                        # 👇 这里的 update_time 将精确到最近的 1 分钟！
                        "latest_update_time": (recent_bars.index[-1].tz_convert('UTC') if getattr(recent_bars.index[-1], 'tzinfo', None) else recent_bars.index[-1]).strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "current_price": round(last_close, 3),
                        "recent_trend_delta_pct": round(overall_delta_pct * 100, 3), 
                        "recent_k_lines": trend_data 
                    }
            return macro_data
        except Exception as e:
            logging.error(f"TradFiScraper 异常: {e}")
            return None

# ==========================================
# 🧠 V3 大脑：DeepSeek 宏观情绪裁判
# ==========================================
class DeepSeekMacroAgent:
    """
    接入 DeepSeek API，负责消化异构数据并输出冷血量化得分。
    """
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/chat/completions"
        
        if not self.api_key:
            logging.warning("⚠️ 未检测到 DEEPSEEK_API_KEY！智能体将返回中性兜底数据。")

    def clean_json_response(self, text):
        """防止大模型输出 markdown 代码块包裹的 JSON"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def analyze(self, current_context, previous_decision=None, decision_t_10=None, decision_t_50=None):
        if not self.api_key:
            return {"score": 0.00, "marquee_text": "未配置 API 密钥，大模型休眠中。", "urgency": "low"}

        logging.info("🧠 正在呼叫 DeepSeek 宏观裁判进行推理...")
        
        # 提取系统当前的绝对时间
        system_current_time = current_context.get("timestamp", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))
        
        system_prompt = f"""
        你是一个顶级的华尔街 Crypto 宏观量化对冲基金经理。你的任务是对当前市场的宏观情绪进行量化打分，输出一个 [-1.00, 1.00] 的浮点数。
        -1.0 代表末日崩盘级别的恐慌，1.00 代表史诗级的狂热逼空，0.00 代表中性震荡。

        【当前系统绝对时间】：{system_current_time}
        (你必须以此时间为基准，衡量所有传入数据的“保鲜度”)

        【跨市场流动性与权重打分基准 (Strict Rubric)】
        1. 权重 40%：美国宏观推力与跨市场溢出 (US Macro & Equities)
           - 数据源：`us_macro_equities` 中的 K 线序列与最后更新时间。包含纳指 (Nasdaq_Futures)、美元指数 (US_Dollar_Index)、10年期美债 (US_10Y_Treasury)。
           - ⚠️ 宏观逻辑铁律：
             * 10年期美债收益率 (^TNX) 飙升代表无风险利率抬升与流动性收紧，对加密市场是重大【利空】。
             * 美元指数 (DXY) 走强代表全球流动性回流美元及避险情绪升温 (Risk-off)，对加密市场是【利空】；美元走弱则是【利好】。
             * 纳指上涨代表宏观风险偏好上升，是【利好】。
           - ⚠️ 数据时效与休市逻辑（严格执行）：【唯一】判断休市的标准是美股数据的 `latest_update_time` 距离【当前系统绝对时间】明确超过 2 小时（此时降权至 10%）。如果时间差在 2 小时以内（如仅相差几分钟到1小时），代表美股**正在正常交易中**，你**绝对不可**以任何理由判定为“休市”，必须维持 40% 的满额权重！

        2. 权重 30%：Crypto 核心基本面与机构资金 (Institutional Crypto Fundamentals)
           - 数据源：`hot_news`。
           - ⚠️ 时效逻辑：阅读新闻发布时间，距离当前时间越久，情绪影响越弱。

        3. 权重 20%：衍生品盘面与散户博弈 (Derivatives & Retail Sentiment)
           - 数据源：`market_position` 中的多空比 (long_short_ratio)。
           - ⚠️ 量化反向指标指南 (极度重要)：币安的多空比代表散户账户比例，必须作为【反向指标】使用！
             * 大于 1.5：散户极度贪婪看多，暗示主力在做空，应计为【负面情绪】。
             * 小于 0.8：散户极度恐慌看空，暗示底部可能形成，应计为【正面情绪】。
             * 0.8 到 1.5：中性博弈状态。

        4. 权重 10%：大盘基础情绪 (Fear & Greed)
           - 数据源：`fear_and_greed`。

        【打分情景锚点（请严格参考此标尺校准你的输出分数，防止分数漂移）】
        +0.80 ~ +1.00：纳指暴涨 + 10年期美债收益率与美元指数大跌 + 散户极度看空（多空比<0.8）+ 极度贪婪。
        +0.30 ~ +0.50：美股温和上涨，无重大新闻，多空比处于中性健康水平。
        -0.10 ~ +0.10：周末休市期，或者多空数据互相矛盾，缺乏明确宏观方向。
        -0.50 ~ -0.80：纳指大跌 + 10年期美债收益率与美元指数飙升 + 散户极度看多（多空比>1.5）。
        -0.80 ~ -1.00：发生黑天鹅事件（如顶级交易所被盗、监管毁灭性打击、宏观流动性极度枯竭）。

        【物理限速锁与历史多重锚点防漂移机制 (State Drift Prevention)】：
        如果用户提供了【历史打分基线】（包含 T-1、T-10、T-50），你必须按以下严格逻辑使用它们：
        1. T-1 (上一次)：作为短期平滑锚点。除非你在极鲜数据中读到末日/史诗级事件，否则本次打分与 T-1 的绝对差值（Delta）不允许超过 ±0.3。如果数据无显著变化，允许打同分。
        2. T-10 (往前第10次)：作为中期趋势校准锚点。评估当前情绪相比中期周期是处于升温还是降温阶段。
        3. T-50 (往前第50次)：作为长期均值回归 (Mean Reversion) 锚点。防止在连续的微小 Delta 累加下产生长期的“分数漂移”。如果当前宏观基本面与 T-50 时期相似，你的分数趋势应当向 T-50 靠拢收敛。
        
        你必须，且只能输出一段严格的 JSON 数据。请绝对不要输出任何其他解释性文字。
        JSON 的结构必须如下：
        {{
            "score": <float>, // 情绪得分：-1.00 到 1.00，必须严格保留两位小数精度。
            "marquee_text": "<string>", // 一句话中文播报，不超过50个字。如果因为美股周末休市而平滑了分数，请在播报中简要说明。
            "urgency": "<string>" // 紧急程度："low", "medium", "high"。
        }}
        """

        user_content = f"【当前市场最新切片数据】\n{json.dumps(current_context, ensure_ascii=False, indent=2)}"
        
        # 动态拼接历史锚点数据
        historical_context_added = False
        if previous_decision:
            user_content += f"\n\n【T-1：上一次的宏观打分与结论】\n{json.dumps(previous_decision, ensure_ascii=False, indent=2)}"
            historical_context_added = True
        if decision_t_10:  
            user_content += f"\n\n【T-10：往前第10次的打分锚点】\n{json.dumps(decision_t_10, ensure_ascii=False, indent=2)}"
            historical_context_added = True
        if decision_t_50: 
            user_content += f"\n\n【T-50：往前第50次的打分锚点】\n{json.dumps(decision_t_50, ensure_ascii=False, indent=2)}"
            historical_context_added = True

        if historical_context_added:
            user_content += "\n\n请结合上述最新数据的时间戳与新鲜度，并以这三个历史打分为多重基准。严格评估情绪的边际变化（Delta）并参照 T-10 和 T-50 防止分数长期漂移，给出平滑过渡后的最终打分。"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "response_format": {"type": "json_object"}, 
            "temperature": 0.1 
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                
                result = response.json()
                raw_content = result['choices'][0]['message']['content']
                
                clean_str = self.clean_json_response(raw_content)
                decision = json.loads(clean_str)
                
                return {
                    "score": round(float(decision.get("score", 0.00)), 2), 
                    "marquee_text": str(decision.get("marquee_text", "API 返回解析异常，使用默认兜底。")),
                    "urgency": str(decision.get("urgency", "low")).lower()
                }
                
            except Exception as e:
                logging.error(f"DeepSeek 分析第 {attempt} 次尝试崩溃: {e}")
                if attempt < max_retries:
                    logging.info(f"⏳ 等待 3 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(3)
                else:
                    logging.error("❌ 达到最大重试次数，分析彻底失败。")
                    return {"score": 0.00, "marquee_text": "🧠 大模型连接异常，多次重试失败，情绪判断已熔断归零。", "urgency": "low"}
def compress_historical_context(raw_json_str):
    """【降维提取】仅提取关键标量特征，剔除冗长的K线数组和新闻，防止Token爆炸"""
    if not raw_json_str: 
        return {}
    try:
        data = json.loads(raw_json_str)
        compressed = {}
        
        # 1. 提取宏观资产现价
        macro = data.get("us_macro_equities", {})
        if macro:
            compressed["US_10Y_Treasury"] = macro.get("US_10Y_Treasury", {}).get("current_price")
            compressed["US_Dollar_Index"] = macro.get("US_Dollar_Index", {}).get("current_price")
            compressed["Nasdaq_Futures"] = macro.get("Nasdaq_Futures", {}).get("current_price")
            
        # 2. 提取多空比和贪婪指数
        pos = data.get("market_position", {})
        if pos:
            compressed["Long_Short_Ratio"] = pos.get("long_short_ratio")
            
        fng = data.get("fear_and_greed", {})
        if fng:
            compressed["Fear_Greed_Value"] = fng.get("value")
            
        return compressed
    except Exception as e:
        logging.error(f"历史数据降维压缩失败: {e}")
        return {}
# ==========================================
# 🔄 中央调度器 (Orchestrator)
# ==========================================
def run_sentinel_pipeline():
    init_db()
    
    news_scraper = NewsScraper()
    fng_scraper = SentimentIndexScraper()
    pos_scraper = MarketPositionScraper()
    tradfi_scraper = TradFiScraper() # 👈 实例化美股刮削器
    
    current_context = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        "us_macro_equities": tradfi_scraper.fetch(), # 👈 将美股真实数据喂入大模型嘴里
        "fear_and_greed": fng_scraper.fetch(),
        "market_position": pos_scraper.fetch(),
        "hot_news": news_scraper.fetch()
    }
    
    temp_log_file = os.path.join(LOG_DIR, "temp_raw_context.json")
    try:
        with open(temp_log_file, "w", encoding="utf-8") as f:
            json.dump(current_context, f, ensure_ascii=False, indent=4)
        logging.info(f"📄 原始切片数据已输出至 -> {temp_log_file}")
    except: pass

    context_json_str = json.dumps(current_context, ensure_ascii=False)
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO raw_context (fetch_ts, context_json) VALUES (?, ?)", 
                       (current_context["timestamp"], context_json_str))
        raw_id = cursor.lastrowid
        
        # 👇 【新增记忆提取】：联表查询，将过去的打分与当时的宏观底层核心数据一并拉取
        previous_decision = None
        decision_t_10 = None
        decision_t_50 = None
        
        try:
            # 联表查询：获取决策及对应的原始数据上下文
            cursor.execute("""
                SELECT d.score, d.marquee_text, d.urgency, r.context_json 
                FROM llm_decisions d
                LEFT JOIN raw_context r ON d.raw_context_id = r.id
                ORDER BY d.decision_ts DESC LIMIT 50
            """)
            historical_rows = cursor.fetchall()
            row_count = len(historical_rows)
            
            def build_anchor(row):
                """组装包含底层数据的锚点对象"""
                anchor = {
                    "score": row[0],
                    "marquee_text": row[1],
                    "urgency": row[2]
                }
                # 将当时的环境变量压缩后塞入参考系中
                env_data = compress_historical_context(row[3])
                if env_data:
                    anchor["macro_environment_at_that_time"] = env_data
                return anchor

            # T-1: 提取最近一次打分及数据
            if row_count > 0:
                previous_decision = build_anchor(historical_rows[0])
                logging.info(f"🔄 成功提取 T-1 基线及底层环境变量")
                
            # T-10: 提取往前第 10 次的打分 (索引为 9)
            if row_count >= 10:
                decision_t_10 = build_anchor(historical_rows[9])
                logging.info(f"🔄 成功提取 T-10 基线及底层环境变量")
                
            # T-50: 提取往前第 50 次的打分 (索引为 49)
            if row_count >= 50:
                decision_t_50 = build_anchor(historical_rows[49])
                logging.info(f"🔄 成功提取 T-50 均值回归基线及底层环境变量")
                
        except Exception as e:
            logging.warning(f"⚠️ 提取历史多重锚点及其底层数据失败: {e}")
            
        conn.commit()

    # 4. 召唤 DeepSeek 进行降维分析 (传入历史多重记忆)
    ai_agent = DeepSeekMacroAgent()
    decision = ai_agent.analyze(
        current_context, 
        previous_decision=previous_decision, 
        decision_t_10=decision_t_10, 
        decision_t_50=decision_t_50
    )
    
    # 5. 判定落库
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO llm_decisions (decision_ts, score, marquee_text, urgency, raw_context_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (current_context["timestamp"], decision["score"], decision["marquee_text"], decision["urgency"], raw_id))
        conn.commit()
        
    logging.info(f"✅ DeepSeek 裁决完成！\n   [情绪打分]: {decision['score']}\n   [大局观]: {decision['marquee_text']}\n   [紧急度]: {decision['urgency'].upper()}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🦅 启动 TimesFM 宏观情报哨兵 (V3 架构 - DeepSeek 核心)")
    print("="*50 + "\n")
    
    run_sentinel_pipeline()