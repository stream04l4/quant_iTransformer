import os
import time
import json
import sqlite3
import logging
import requests
import re
import yfinance as yf
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from quant_config.config import LIVE_DATA_DIR

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

DB_NAME = os.path.join(LIVE_DATA_DIR, "macro_intelligence.db")

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
        self.freshness_threshold_hours = 2.0
        self.delta_tolerance_hours = 6.0

    def _to_utc_timestamp(self, idx):
        if getattr(idx, "tzinfo", None):
            return idx.tz_convert("UTC")
        return idx.tz_localize("UTC")

    def _compute_24h_delta(self, hist, latest_ts):
        """按真实时间差回溯 24 小时，避免周末/休市把 96 根交易 K 线误当成自然 24h。"""
        import pandas as pd

        target_ts = latest_ts - pd.Timedelta(hours=24)
        anchor_hist = hist[hist.index <= target_ts]
        if anchor_hist.empty:
            return None, None

        anchor_ts = anchor_hist.index[-1]
        elapsed_hours = (latest_ts - anchor_ts).total_seconds() / 3600.0
        if abs(elapsed_hours - 24.0) > self.delta_tolerance_hours:
            return None, round(elapsed_hours, 2)

        latest_close = float(hist['Close'].iloc[-1])
        anchor_close = float(anchor_hist['Close'].iloc[-1])
        if anchor_close == 0:
            return None, round(elapsed_hours, 2)
        return ((latest_close - anchor_close) / anchor_close), round(elapsed_hours, 2)

    def fetch(self):
        logging.info("正在抓取传统宏观资产实时序列 (启动高频缝合机制)...")
        macro_data = {}
        try:
            for name, symbol in self.tickers.items():
                ticker = yf.Ticker(symbol)
                
                # 👇 【核心提速 1】：将基础序列从 1h 降维到 15m，大幅缩小 K 线闭合等待期
                hist = ticker.history(period="5d", interval="15m")
                
                if len(hist) >= 2:
                    import pandas as pd

                    if getattr(hist.index, "tz", None) is None:
                        hist.index = hist.index.tz_localize("UTC")
                    else:
                        hist.index = hist.index.tz_convert("UTC")

                    # 👇 【核心提速 2】：强制拉取当前最新 1 分钟的 Tick（现价）
                    try:
                        live_tick = ticker.history(period="1d", interval="1m").tail(1)
                        # 如果最新 1 分钟的时间戳比 15 分钟的末尾还要新，直接将它追加到末尾缝合！
                        if not live_tick.empty:
                            if getattr(live_tick.index, "tz", None) is None:
                                live_tick.index = live_tick.index.tz_localize("UTC")
                            else:
                                live_tick.index = live_tick.index.tz_convert("UTC")
                        if not live_tick.empty and live_tick.index[0] > hist.index[-1]:
                            hist = pd.concat([hist, live_tick])
                    except Exception as e:
                        logging.warning(f"  ⚠️ {symbol} 极速 Tick 缝合失败，回退至基础 K 线: {e}")

                    # === 截取最后 8 根 K 线保留微观轨迹 (保留原逻辑) ===
                    recent_bars = hist.tail(8)
                    trend_data = []
                    
                    for idx, row in recent_bars.iterrows():
                        trend_data.append({
                            "time": self._to_utc_timestamp(idx).strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "close": round(float(row['Close']), 3)
                        })
                    
                    # === 以下为新增与修改的核心逻辑 ===
                    latest_ts = self._to_utc_timestamp(recent_bars.index[-1])
                    now_utc = datetime.now(timezone.utc)
                    hours_since_update = (now_utc - latest_ts.to_pydatetime()).total_seconds() / 3600.0
                    is_fresh = hours_since_update <= self.freshness_threshold_hours
                    market_status = "live_or_recent" if is_fresh else "stale_or_closed"

                    last_close = float(recent_bars['Close'].iloc[-1])
                    
                    # 1. 精确提取 Rolling 24H 变化率：按真实时间回溯，跨周末/休市则直接标记不可用
                    daily_delta, delta_window_hours = self._compute_24h_delta(hist, latest_ts)
                    
                    # 2. 跨资产波动率标准化 (Price Z-Score)
                    # 将现价放置在 5 日的统计分布中计算偏离度，抹平美债(波动极小)和纳指(波动极大)的底层差异
                    period_mean = hist['Close'].mean()
                    period_std = hist['Close'].std()
                    # 防御性除法，防止除以 0
                    z_score = (last_close - period_mean) / period_std if period_std > 0 else 0.0
                    
                    macro_data[name] = {
                        "latest_update_time": latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "hours_since_update": round(hours_since_update, 2),
                        "is_fresh": is_fresh,
                        "market_status": market_status,
                        "current_price": round(last_close, 3),
                        "daily_24h_delta_pct": None if daily_delta is None else round(daily_delta * 100, 3),
                        "delta_window_hours": delta_window_hours,
                        "price_z_score": round(z_score, 2),
                        "recent_k_lines": trend_data 
                    }
            return macro_data
        except Exception as e:
            logging.error(f"TradFiScraper 异常: {e}")
            return None

# ==========================================
# 🧠 V3 大脑：DeepSeek 宏观情绪裁判
# ==========================================
def _is_weekend_utc(timestamp_text):
    try:
        dt = datetime.strptime(str(timestamp_text or ""), "%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        dt = datetime.utcnow()
    return dt.weekday() >= 5


def _all_macro_assets_stale(current_context):
    macro = (current_context or {}).get("us_macro_equities", {}) or {}
    if not macro:
        return False
    return all(not bool(asset_data.get("is_fresh")) for asset_data in macro.values())


def calibrate_macro_score(raw_score, marquee_text="", urgency="low", current_context=None):
    """把大模型的自然语言判断压回交易系统可解释的强度标尺。"""
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(-1.0, min(1.0, score))
    text = str(marquee_text or "")
    urgency = str(urgency or "low").lower()

    neutral_terms = [
        "震荡", "横盘", "观望", "维持", "休市", "数据陈旧",
        "缺乏方向", "缺乏明确方向", "中性", "互相矛盾"
    ]
    extreme_terms = [
        "黑天鹅", "崩盘", "暴跌", "暴涨", "大跌", "大涨", "急跌", "急涨",
        "史诗", "逼空", "毁灭性", "交易所被盗", "流动性枯竭", "极端", "高危"
    ]

    is_neutral_text = any(term in text for term in neutral_terms)
    has_extreme_text = any(term in text for term in extreme_terms)

    if is_neutral_text and not has_extreme_text:
        cap = 0.40 if urgency == "medium" else 0.30
        score = max(-cap, min(cap, score))

    current_context = current_context or {}
    is_weekend = _is_weekend_utc(current_context.get("timestamp"))
    all_macro_stale = _all_macro_assets_stale(current_context)

    if all_macro_stale:
        cap = 0.55 if urgency == "high" and has_extreme_text else 0.40
        score = max(-cap, min(cap, score))

    if is_weekend:
        cap = 0.60 if urgency == "high" and has_extreme_text else 0.45
        score = max(-cap, min(cap, score))

    return round(score, 2)


class DeepSeekMacroAgent:
    """
    接入 DeepSeek API，负责消化异构数据并输出冷血量化得分。
    """
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/chat/completions"
        self.connect_timeout = float(os.environ.get("DEEPSEEK_CONNECT_TIMEOUT", "10"))
        self.read_timeout = float(os.environ.get("DEEPSEEK_READ_TIMEOUT", "90"))
        self.max_completion_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "1200"))
        self.thinking_mode = os.environ.get("DEEPSEEK_THINKING_MODE", "disabled").lower()
        
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

    def extract_json_object(self, text):
        """从混杂文本中提取第一个 JSON 对象，兼容模型偶发前后缀说明。"""
        text = (text or "").strip()
        if not text:
            return ""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return ""
        return text[start:end + 1]

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
           - 数据源：`us_macro_equities`。包含纳指、美元指数、10年期美债。
           - 你必须逐个资产先检查 `is_fresh`、`hours_since_update`、`market_status`、`delta_window_hours`：
             * 若 `is_fresh=false` 或 `hours_since_update > 2`，该资产视为【陈旧/休市数据】，只能作为背景参考，单资产权重降到接近 0，不得描述为“刚刚暴涨/暴跌”。
             * 若 `daily_24h_delta_pct` 为 null，或者 `delta_window_hours` 明显偏离 24 小时，说明该 24H 变化率因周末/休市断档而无效，必须忽略这条涨跌幅信息。
           - ⚠️ 宏观逻辑铁律与 Z-Score 判定标准 (极为关键)：
             你必须优先查看【新鲜资产】的 `price_z_score` (价格的标准差偏离度) 与 `daily_24h_delta_pct`，不要受各资产绝对波动百分比的干扰。
             * Z-Score 的统计学意义：0 附近为均值回归/横盘。|Z-Score| > 1.0 视为趋势形成，|Z-Score| > 2.0 视为极端单边暴涨/暴跌 (即 2 个标准差事件)。
             * 10年期美债收益率 (^TNX) 与 美元指数 (DXY)：仅当该资产【数据新鲜】且 Z-Score > +1.5 且 24H 涨幅为正，代表全球流动性剧烈收紧，对 Crypto 构成【毁灭性打击】；反之为【利好】。
             * 纳斯达克 (NQ=F)：仅当该资产【数据新鲜】且 Z-Score < -1.5 且 24H 涨幅显著为负（例如跌幅超过 1%），代表风险资产崩盘溢出，对 Crypto 是【重大利空】；反之为【利好】。
           - ⚠️ 数据时效与休市逻辑（严格执行）：40% 的满额权重只给【新鲜资产】。某个资产陈旧，不代表整个宏观模块完全失效；你必须逐资产降权，而不是整组一刀切。
           - ⚠️ 周末/休市低流动性约束：若当前是周六或周日 UTC，或 `us_macro_equities` 全部 `is_fresh=false`，不得把 TradFi 的 `price_z_score` 当作实时极端冲击；除非新闻中存在明确黑天鹅，否则 |score| 不应超过 0.60。

        2. 权重 30%：Crypto 核心基本面与机构资金 (Institutional Crypto Fundamentals)
           - 数据源：`hot_news`。
           - ⚠️ 时效逻辑：阅读新闻发布时间，距离当前时间越久，情绪影响越弱。
           - ⚠️ 合规约束：本任务禁止把 BTC 现价/点位 作为打分依据。若新闻标题中的 BTC 价格数字被脱敏，不得自行脑补具体点位。
           - ⚠️ 输出约束：`marquee_text` 禁止出现 BTC 的具体价格、点位、支撑阻力位、百分位报价描述，只能描述方向和宏观事件，例如“BTC承压”“市场风险偏好下降”。

        3. 权重 30%：大盘基础情绪 (Fear & Greed)
           - 数据源：`fear_and_greed`。

        【打分情景锚点（请严格参考此标尺校准你的输出分数，防止分数漂移）】
        +0.00 ~ +0.20：中性震荡、休市/数据陈旧、消息互相矛盾、缺乏明确方向。
        +0.20 ~ +0.35：轻微风险偏好修复；只允许用于“温和修复/偏暖震荡”，不得称作强趋势。
        +0.40 ~ +0.60：明确方向性风险偏好改善；必须同时有【新鲜资产】趋势证据或重要利好新闻支撑。普通震荡不得进入本区间。
        +0.70 ~ +1.00：大波动/极端风险偏好释放；需要 Z-Score > 2 或重大行业/宏观事件共振。
        -0.00 ~ -0.20：中性偏弱震荡、休市/数据陈旧、消息互相矛盾、缺乏明确方向。
        -0.20 ~ -0.35：轻微风险偏好降温；只允许用于“弱势震荡/偏冷震荡”，不得称作强趋势。
        -0.40 ~ -0.60：明确方向性风险偏好恶化；必须同时有【新鲜资产】趋势证据或重要利空新闻支撑。普通震荡不得进入本区间。
        -0.70 ~ -1.00：大波动/极端恐慌；需要 Z-Score < -2、宏观流动性剧烈收紧或黑天鹅事件。

        【震荡区间硬约束】
        如果 `marquee_text` 的语义包含“震荡、横盘、观望、维持、休市、数据陈旧、缺乏方向”，且没有明确的“大跌/暴跌/暴涨/黑天鹅/监管毁灭性打击/交易所被盗/流动性枯竭”等极端事件：
        - `urgency` 必须为 low 或 medium。
        - low 紧急度时，|score| 不得超过 0.30。
        - medium 紧急度时，|score| 不得超过 0.40。
        - 0.40 以上代表较大方向性波动，0.70 以上代表大波动/极端事件，禁止用于普通震荡描述。

        【物理限速锁与历史多重锚点防漂移机制 (State Drift Prevention)】：
        如果用户提供了【历史打分基线】（包含 T-1、T-10、T-50），你必须按以下严格逻辑使用它们：
        1. T-1 (上一次)：作为短期平滑锚点。在常规震荡市中，本次打分与 T-1 的绝对差值（Delta）尽量控制在 ±0.3 以内。但【例外铁律】：如果【新鲜资产】的 `daily_24h_delta_pct` 与 `price_z_score` 同时显示强劲单边趋势，或者 `hot_news` 包含行业巨震事件，你必须打破 ±0.3 的限制，直接给出符合当前极端情况的跳空打分，不可被 T-1 拖累。
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

        llm_context = build_llm_ready_context(current_context)
        user_content = f"【当前市场最新切片数据】\n{json.dumps(llm_context, ensure_ascii=False, indent=2)}"
        
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
            "temperature": 0.1,
            "max_tokens": self.max_completion_tokens,
            "thinking": {"type": self.thinking_mode}
        }

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                response.raise_for_status()

                response_text = response.text or ""
                if not response_text.strip():
                    raise ValueError("DeepSeek HTTP 响应体为空")

                try:
                    result = response.json()
                except Exception as json_resp_err:
                    preview = response_text[:500].replace("\n", "\\n")
                    raise ValueError(f"DeepSeek HTTP 响应不是合法 JSON: {json_resp_err} | preview={preview}")

                choices = result.get("choices") or []
                if not choices:
                    preview = response_text[:500].replace("\n", "\\n")
                    raise ValueError(f"DeepSeek 响应缺少 choices | preview={preview}")

                choice0 = choices[0] or {}
                finish_reason = choice0.get("finish_reason")
                message = choice0.get("message", {}) or {}
                raw_content = message.get("content", "")
                if not str(raw_content or "").strip():
                    reasoning_preview = str(message.get("reasoning_content", "") or "")[:500].replace("\n", "\\n")
                    preview = response_text[:500].replace("\n", "\\n")
                    raise ValueError(
                        f"DeepSeek message.content 为空 | finish_reason={finish_reason} "
                        f"| reasoning_preview={reasoning_preview} | preview={preview}"
                    )

                clean_str = self.clean_json_response(str(raw_content))
                if not clean_str.startswith("{"):
                    extracted = self.extract_json_object(clean_str)
                    clean_str = extracted or clean_str
                if not clean_str:
                    preview = str(raw_content)[:500].replace("\n", "\\n")
                    raise ValueError(f"DeepSeek content 中未提取到 JSON 对象 | preview={preview}")

                try:
                    decision = json.loads(clean_str)
                except Exception as json_content_err:
                    preview = clean_str[:500].replace("\n", "\\n")
                    raise ValueError(f"DeepSeek content 不是合法 JSON: {json_content_err} | preview={preview}")
                
                marquee_text = sanitize_marquee_text(str(decision.get("marquee_text", "API 返回解析异常，使用默认兜底。")))
                urgency = str(decision.get("urgency", "low")).lower()
                calibrated_score = calibrate_macro_score(
                    float(decision.get("score", 0.00)),
                    marquee_text,
                    urgency,
                    current_context,
                )

                return {
                    "score": calibrated_score,
                    "marquee_text": marquee_text,
                    "urgency": urgency
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
        
        macro = data.get("us_macro_equities", {})
        if macro:
            for asset in ["US_10Y_Treasury", "US_Dollar_Index", "Nasdaq_Futures"]:
                asset_data = macro.get(asset, {})
                if asset_data:
                    compressed[asset] = {
                        "z_score": asset_data.get("price_z_score"),
                        "daily_delta": asset_data.get("daily_24h_delta_pct"),
                        "is_fresh": asset_data.get("is_fresh"),
                        "hours_since_update": asset_data.get("hours_since_update")
                    }   
        fng = data.get("fear_and_greed", {})
        if fng:
            compressed["Fear_Greed_Value"] = fng.get("value")
            
        return compressed
    except Exception as e:
        logging.error(f"历史数据降维压缩失败: {e}")
        return {}


def _redact_btc_price_mentions(text):
    """去掉 BTC/Bitcoin 标题里的价格数字，防止把现价泄漏给情绪裁判。"""
    if not text:
        return text

    if not re.search(r"(?i)(?:BTC|Bitcoin|比特币)", text):
        return text

    price_expr = r"(?:[$€¥£]?\s?\d[\d,]*(?:\.\d+)?(?:\s*[kmb])?|\b\d[\d,]*(?:\.\d+)?\s*(?:usd|usdt|dollars?|k|m|b|万|亿)\b)"
    btc_expr = r"(?:BTC|Bitcoin|比特币)"

    redacted = re.sub(
        rf"(?i)({btc_expr}[^\d$€¥£，。,；;]{{0,24}}){price_expr}",
        r"\1[PRICE_REDACTED]",
        text,
    )
    redacted = re.sub(
        rf"(?i){price_expr}([^\d$€¥£，。,；;]{{0,24}}{btc_expr})",
        r"[PRICE_REDACTED]\1",
        redacted,
    )
    redacted = re.sub(r"(?i)\bprice\b", "price[REDACTED]", redacted)
    redacted = re.sub(r"(?i)\b(?:to|at|near|below|above|under|over)\s+\[PRICE_REDACTED\]", "", redacted)
    redacted = re.sub(r"(?i)(?:跌至|升至|涨至|降至|触及|站上|跌破|突破)\s*\[PRICE_REDACTED\]", "走势变化", redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted


def _sanitize_hot_news_for_btc_prediction(news_items):
    """保留新闻事件本身，但移除 BTC 现价/点位描述。"""
    sanitized_items = []
    for item in news_items or []:
        sanitized_items.append({
            "title": _redact_btc_price_mentions(item.get("title", "")),
            "source": item.get("source", ""),
            "time": item.get("time", ""),
        })
    return sanitized_items


def sanitize_marquee_text(text):
    """兜底清洗模型输出，禁止播报 BTC 价格与点位。"""
    text = _redact_btc_price_mentions(text or "")
    text = re.sub(
        r"(?i)(?:多空比|巨鲸|散户|订单流|资金费率|持仓量|whale|retail|order flow|funding rate|open interest|long[-\s]?short ratio)[^，。,；;]*",
        "市场微观噪声已过滤",
        text,
    )
    text = re.sub(r"(?i)\b(?:support|resistance)\b", "level", text)
    text = re.sub(r"(?i)(?:BTC|Bitcoin|比特币)[^，。,；;]*\[PRICE_REDACTED\][^，。,；;]*", "BTC走势承压", text)
    text = re.sub(r"(?i)(Z[-\s]?Score)\s*[-:：]?\s*\[PRICE_REDACTED\]", r"\1", text)
    text = text.replace("[PRICE_REDACTED]", "价格点位已脱敏")
    text = re.sub(r"(?:市场微观噪声已过滤[，。,；;\s]*)+", "市场微观噪声已过滤，", text)
    text = re.sub(r"\s+", " ", text).strip(" ，。,；;")
    return text or "宏观情绪中性，未见可广播的价格无关结论。"


def build_llm_ready_context(current_context):
    """压缩当前切片，明确排除 BTC 现价与任何绝对价格轨迹。"""
    compact = {
        "timestamp": current_context.get("timestamp"),
        "fear_and_greed": current_context.get("fear_and_greed"),
        "hot_news": _sanitize_hot_news_for_btc_prediction((current_context.get("hot_news", []) or [])[:5]),
        "us_macro_equities": {},
    }

    macro = current_context.get("us_macro_equities", {}) or {}
    for asset, asset_data in macro.items():
        compact["us_macro_equities"][asset] = {
            "latest_update_time": asset_data.get("latest_update_time"),
            "hours_since_update": asset_data.get("hours_since_update"),
            "is_fresh": asset_data.get("is_fresh"),
            "market_status": asset_data.get("market_status"),
            "daily_24h_delta_pct": asset_data.get("daily_24h_delta_pct"),
            "delta_window_hours": asset_data.get("delta_window_hours"),
            "price_z_score": asset_data.get("price_z_score"),
        }
    return compact
# ==========================================
# 🔄 中央调度器 (Orchestrator)
# ==========================================
def run_sentinel_pipeline():
    init_db()
    
    news_scraper = NewsScraper()
    fng_scraper = SentimentIndexScraper()
    tradfi_scraper = TradFiScraper() # 👈 实例化美股刮削器
    
    current_context = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        "us_macro_equities": tradfi_scraper.fetch(), # 👈 将美股真实数据喂入大模型嘴里
        "fear_and_greed": fng_scraper.fetch(),
        "hot_news": news_scraper.fetch()
    }
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
                    "marquee_text": sanitize_marquee_text(row[1]),
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
    logging.info(f"✅ DeepSeek 裁决完成！\n   [情绪打分]: {decision['score']}\n   [大局观]: {decision['marquee_text']}\n   [紧急度]: {decision['urgency'].upper()}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🦅 启动 TimesFM 宏观情报哨兵 (V3 架构 - DeepSeek 核心)")
    print("="*50 + "\n")
    
    run_sentinel_pipeline()
