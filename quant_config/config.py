import os
from pathlib import Path

# ==========================================
# 1. 目录结构配置 (Directory Paths)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 主要模块目录
BACKTEST_DIR = PROJECT_ROOT / "backtest_pipeline"
HISTORICAL_DIR = PROJECT_ROOT / "historical_pipeline"
CONFIG_DIR = PROJECT_ROOT / "quant_config"

# 数据与模型目录
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
LIVE_DATA_DIR = DATA_DIR / "live"

TEMP_DIR = TRAIN_DATA_DIR / "raw_temp"
PROCESSED_AGGTRADES_DIR = TRAIN_DATA_DIR / "processed" / "aggtrades_1h"
PROCESSED_MAIN_DIR = TRAIN_DATA_DIR / "processed" / "main_features_1h"
MODEL_DIR = PROJECT_ROOT / "model"
LOG_DIR = LIVE_DATA_DIR / "logs"

# ==========================================
# 2. 交易标的与业务参数 (Trading & Features)
# ==========================================
SYMBOL = "BTCUSDT"
AUX_SYMBOLS = ["ETHUSDT", "ETHBTC"]  

WHALE_THRESHOLD = 5.0    
RETAIL_THRESHOLD = 0.1   
# 👇 改为动态获取，供 Mode 1 寻优
FORECAST_HORIZON = int(os.environ.get("OPT_FORECAST_HORIZON", 12))

# ==========================================
# 3. 历史数据获取区间 (Data Range)
# ==========================================
DATA_START_MONTH = (2023, 1)   
DATA_END_MONTH = (2026, 5)     

# ==========================================
# 4. 币安接口与表结构路由 (Endpoints & Schema)
# ==========================================
BINANCE_BASE_URL_FUTURES = "https://data.binance.vision/data/futures/um/monthly"
BINANCE_BASE_URL_FUTURES_DAILY = "https://data.binance.vision/data/futures/um/daily"
BINANCE_BASE_URL_SPOT = "https://data.binance.vision/data/spot/monthly"
BINANCE_BASE_URL_METRICS_DAILY = "https://data.binance.vision/data/futures/um/daily/metrics"
# 强制规范无表头 CSV 的字段名防错
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", 
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
              
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

AGGTRADE_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", 
                 "last_trade_id", "transact_time", "is_buyer_maker"]

METRICS_COLS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value", 
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", 
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
]
# ==========================================
# 5. 回测与实盘交易参数 (Trading Strategy)
# ==========================================
# 👇 全部改为动态获取，供 Mode 2 寻优
LEVERAGE_1 = float(os.environ.get("OPT_LEVERAGE_1", 2.0))
LEVERAGE_2 = float(os.environ.get("OPT_LEVERAGE_2", 2.0))
LEVERAGE_3 = float(os.environ.get("OPT_LEVERAGE_3", 3.0))

FEE_RATE = 0.0005        

HOLD_PERIOD = int(os.environ.get("OPT_HOLD_PERIOD", 12))         
HARD_TP = float(os.environ.get("OPT_HARD_TP", 0.05))           
HARD_SL = float(os.environ.get("OPT_HARD_SL", 0.015))

# --- 实盘专用风控与仓位参数 ---
TRADE_COOLDOWN_SEC = 3500            # 开仓后防连续发车冷却时间(秒)
# 宏观情绪 Logit 贝叶斯融合分层系数 (Alpha)
MACRO_ALPHA_TOP_10 = 1.20            # Tier 1 (入场门槛): 宏观权重极高，主导尾部风险与大趋势
MACRO_ALPHA_TOP_5 = 0.50             # Tier 2 (波段加仓): 宏观权重中等
MACRO_ALPHA_TOP_1 = 0.15             # Tier 3 (激进杠杆): 宏观权重极低，尊重微观量价动能

HEARTBEAT_INTERVAL = 10              # 实盘高频心跳间隔(秒)
MAX_NOTIONAL_ALTS = 10000            # 实盘非BTC/ETH最大notional(USDT)
MAX_NOTIONAL_BTC = 100000            # 实盘BTC/ETH最大notional(USDT)

# --- 账户资金管理参数 ---
INITIAL_CAPITAL = 5000.0            # 账户起始资本(USDT)，用于统计累计收益率
BASE_RISK_BUDGET = 0.2              # 单笔风险预算(拨备率)，保证金 = INITIAL_CAPITAL × BASE_RISK_BUDGET
MAX_OPEN_POSITIONS = 1               # 最大允许同时持仓数量
MAX_MARGIN_UTILIZATION = 1         # 最大资金使用率红线(70%)
MIN_ORDER_NOTIONAL = 1.0             # 最小下单名义价值(USDT)

# --- 右脑高频熔断参数 ---
BAILOUT_ATR_MULTI = 2.5              # 价格瀑布触发阈值(ATR倍数)：1分钟内跌幅超过此倍数ATR才触发
BAILOUT_VOL_MULTI = 3.0              # 异常爆量触发阈值(Volume倍数)：成交量必须超过均量的此倍数
BAILOUT_MIN_PHYSICAL_DROP = 0.005    # 熔断最小物理跌幅过滤(0.5%)，防止微小波动误触
BAILOUT_OBI_THRESH = 0.2             # 订单簿失衡阈值(OBI < 0.2时才确认为真瀑布)

# ==========================================
# 6. 模型架构与训练超参数 (全参环境变量动态注入区)
# ==========================================
EXPERIMENT_ID = os.environ.get("OPT_EXP_ID", "EXP_D47B1607")

SEQ_LEN = int(os.environ.get("OPT_SEQ_LEN", 672))
BATCH_SIZE = int(os.environ.get("OPT_BATCH_SIZE", 128))
EPOCHS = int(os.environ.get("OPT_EPOCHS", 50))
LEARNING_RATE = float(os.environ.get("OPT_LEARNING_RATE", 5e-6))

D_MODEL = int(os.environ.get("OPT_D_MODEL", 128))
E_LAYERS = int(os.environ.get("OPT_E_LAYERS", 2))
DROPOUT = float(os.environ.get("OPT_DROPOUT", 0.2))

CLASS_THRESHOLD = float(os.environ.get("OPT_CLASS_THRESHOLD", 0.015))
