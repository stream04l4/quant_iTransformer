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
AUX_SYMBOLS = ["ETHUSDT", "ETHBTC"]  # 横截面辅助标的

WHALE_THRESHOLD = 5.0    # 巨鲸大单阈值 (>=5 BTC)
RETAIL_THRESHOLD = 0.1   # 散户碎单阈值 (<=0.1 BTC)
FORECAST_HORIZON = 24    # 模型预测未来 24 小时

# ==========================================
# 3. 历史数据获取区间 (Data Range)
# ==========================================
DATA_START_MONTH = (2023, 1)   
DATA_END_MONTH = (2026, 4)     

# ==========================================
# 4. 币安接口与表结构路由 (Endpoints & Schema)
# ==========================================
BINANCE_BASE_URL_FUTURES = "https://data.binance.vision/data/futures/um/monthly"
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
# 这些参数在调优时需要频繁变动，集中管理以防止回测与实盘错配
LEVERAGE_1 = 1.0           # 杠杆倍数
LEVERAGE_2 = 2.0           # 杠杆倍数
LEVERAGE_3 = 3.0           # 杠杆倍数

FEE_RATE = 0.0005        # 交易所单边手续费 (币安市价单 Taker 万五)

HOLD_PERIOD = 24         
HARD_TP = 0.6           # 绝对止盈线 50% 
HARD_SL = 0.3           # 绝对止损线 50% 

# --- 实盘专用风控与仓位参数 ---
LIVE_RISK_MARGIN_USD = 100.0         # 每次发车动用的固定保证金金额(U)
TRADE_COOLDOWN_SEC = 3600            # 开仓后防连续发车冷却时间(秒)
MAX_MACRO_IMPACT = 0.015            # 宏观情绪对模型输出概率的最大干预幅度
MACRO_CHECK_INTERVAL = 1800         # 宏观情报拉取间隔(秒)
HEARTBEAT_INTERVAL = 10             # 实盘高频心跳间隔(秒)


# ==========================================
# 6. 模型架构与训练超参数 (全参环境变量动态注入区)
# ==========================================
EXPERIMENT_ID = os.environ.get("OPT_EXP_ID", "default_run")

SEQ_LEN = int(os.environ.get("OPT_SEQ_LEN", 336))
BATCH_SIZE = int(os.environ.get("OPT_BATCH_SIZE", 256))
EPOCHS = int(os.environ.get("OPT_EPOCHS", 50))
LEARNING_RATE = float(os.environ.get("OPT_LEARNING_RATE", 5e-5))

D_MODEL = int(os.environ.get("OPT_D_MODEL", 512))
E_LAYERS = int(os.environ.get("OPT_E_LAYERS", 2))
DROPOUT = float(os.environ.get("OPT_DROPOUT", 0.4))

CLASS_THRESHOLD = float(os.environ.get("OPT_CLASS_THRESHOLD", 0.026))