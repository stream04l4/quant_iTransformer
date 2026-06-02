import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded before configuration parsing
load_dotenv()

# ==========================================
# 1. 目录结构配置 (Directory Paths)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 主要模块目录
BACKTEST_DIR = PROJECT_ROOT / "backtest_pipeline"
HISTORICAL_DIR = PROJECT_ROOT / "historical_pipeline"

# 数据与模型目录
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
LIVE_DATA_DIR = DATA_DIR / "live"

TEMP_DIR = TRAIN_DATA_DIR / "raw_temp"
PROCESSED_DIR = TRAIN_DATA_DIR / "processed"
PROCESSED_AGGTRADES_ROOT = PROCESSED_DIR / "aggtrades_1h"
PROCESSED_MAIN_ROOT = PROCESSED_DIR / "main_features_1h"
MODEL_DIR = PROJECT_ROOT / "model"

# ==========================================
# 2. 交易标的与业务参数 (Trading & Features)
# ==========================================
SYMBOL = "BTCUSDT"
AUX_SYMBOLS = ["ETHUSDT", "ETHBTC"]  

def normalize_symbol(raw_symbol):
    return str(raw_symbol or "").split(":")[0].replace("/", "").upper()

def processed_aggtrades_dir(symbol=None):
    return PROCESSED_AGGTRADES_ROOT / normalize_symbol(symbol or SYMBOL)

def processed_main_dir(symbol=None):
    return PROCESSED_MAIN_ROOT / normalize_symbol(symbol or SYMBOL)

PROCESSED_AGGTRADES_DIR = processed_aggtrades_dir(SYMBOL)
PROCESSED_MAIN_DIR = processed_main_dir(SYMBOL)

WHALE_NOTIONAL_THRESHOLD_USDT = 200000.0
RETAIL_NOTIONAL_THRESHOLD_USDT = 4000.0
# 👇 改为动态获取，供 Mode 1 寻优
FORECAST_HORIZON = int(os.environ.get("OPT_FORECAST_HORIZON", 12))

# ==========================================
# 3. 历史数据获取区间 (Data Range)
# ==========================================
DATA_START_MONTH = (2021, 12)   
DATA_END_MONTH = (2026, 6)     

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
# ==========================================
# 5. 回测与实盘交易参数 (Trading Strategy)
# ==========================================
# 👇 全部改为动态获取，供 Mode 2 寻优
LEVERAGE_1 = float(os.environ.get("OPT_LEVERAGE_1", 2.0))
LEVERAGE_2 = float(os.environ.get("OPT_LEVERAGE_2", 5.0))
LEVERAGE_3 = float(os.environ.get("OPT_LEVERAGE_3", 8.0))

FEE_RATE = 0.0005        

HOLD_PERIOD = int(os.environ.get("OPT_HOLD_PERIOD", 12))         
HARD_TP = float(os.environ.get("OPT_HARD_TP", 0.025))           
HARD_SL = float(os.environ.get("OPT_HARD_SL", 0.025))
# 追踪止盈回撤宽度：ATR(period) × multi / entry_price，替代固定 1% 回撤。
TRAILING_ATR_PERIOD = int(os.environ.get("OPT_TRAILING_ATR_PERIOD", 14))
TRAILING_ATR_MULTI = float(os.environ.get("OPT_TRAILING_ATR_MULTI", 3.5))

# --- 实盘专用风控与仓位参数 ---
# 宏观情绪非线性 Logit 融合参数。
# MACRO_ALPHA_TOP_* 只表示档位相对权重；曲线强度由 MACRO_LOGIT_* 控制。
MACRO_ALPHA_TOP_10 = 0.10            # Tier 1 (入场门槛): 模型信心不足，外部宏观输入权重最高
MACRO_ALPHA_TOP_5 = 0.05             # Tier 2 (波段加仓): 宏观权重中等
MACRO_ALPHA_TOP_1 = 0.02             # Tier 3 (激进杠杆): 模型信号很强，宏观只做轻微确认
MACRO_LOGIT_DEAD_ZONE = float(os.environ.get("OPT_MACRO_LOGIT_DEAD_ZONE", 0.20))
MACRO_LOGIT_MID_SCORE = float(os.environ.get("OPT_MACRO_LOGIT_MID_SCORE", 0.50))
MACRO_LOGIT_TAIL_SCORE = float(os.environ.get("OPT_MACRO_LOGIT_TAIL_SCORE", 0.80))
MACRO_LOGIT_MID_DELTA = float(os.environ.get("OPT_MACRO_LOGIT_MID_DELTA", 0.075))
MACRO_LOGIT_TAIL_DELTA = float(os.environ.get("OPT_MACRO_LOGIT_TAIL_DELTA", 0.420))
MACRO_LOGIT_MAX_DELTA = float(os.environ.get("OPT_MACRO_LOGIT_MAX_DELTA", 0.550))
MACRO_LOGIT_MID_POWER = float(os.environ.get("OPT_MACRO_LOGIT_MID_POWER", 1.35))
MACRO_LOGIT_TAIL_POWER = float(os.environ.get("OPT_MACRO_LOGIT_TAIL_POWER", 3.0))
MACRO_LOGIT_EXTREME_POWER = float(os.environ.get("OPT_MACRO_LOGIT_EXTREME_POWER", 1.5))

HEARTBEAT_INTERVAL = 10              # 实盘高频心跳间隔(秒)
MAX_NOTIONAL_ALTS = 10000            # 实盘非BTC/ETH最大notional(USDT)
MAX_NOTIONAL_BTC = 100000            # 实盘BTC/ETH最大notional(USDT)

# --- 账户资金管理参数 ---
INITIAL_CAPITAL = 5000.0            # 账户起始资本(USDT)，用于统计累计收益率
BASE_RISK_BUDGET = 0.2              # 单笔风险预算(拨备率)，保证金 = INITIAL_CAPITAL × BASE_RISK_BUDGET

# --- 右脑高频熔断参数 ---
BAILOUT_ATR_MULTI = 3.0              # 价格瀑布触发阈值(ATR倍数)：1分钟内跌幅超过此倍数ATR才触发
BAILOUT_VOL_MULTI = 4.0              # 异常爆量触发阈值(Volume倍数)：成交量必须超过均量的此倍数
BAILOUT_MIN_PHYSICAL_DROP = 0.005    # 熔断最小物理跌幅过滤(0.5%)，防止微小波动误触
BAILOUT_OBI_THRESH = 0.2             # 订单簿失衡阈值(OBI < 0.2时才确认为真瀑布)

# ==========================================
# 6. 模型架构与训练超参数 (全参环境变量动态注入区)
# ==========================================
EXPERIMENT_ID = os.environ.get("OPT_EXP_ID", "default_run")

# 实盘组合配置：训练脚本仍使用上面的 SYMBOL/AUX_SYMBOLS 单币种配置；
# live_pipeline/market_gateway/risk_manager 会读取这里，同时加载多个已训练好的单币模型。
# 每个交易标的可配置任意数量模型；单模型时保留旧写法也兼容。
BTCUSDT_LIVE_MODELS = [
    {"experiment_id": "EXP_6ADB71DE", "weight": 0.4},
    {"experiment_id": "EXP_A49047F0", "weight": 0.25},
    {"experiment_id": "EXP_AA8D8236", "weight": 0.2},
    {"experiment_id": "EXP_321B10D8", "weight": 0.15},
]

LIVE_SYMBOL_CONFIGS = {
    SYMBOL: {
        "experiment_id": "EXP_6ADB71DE" if normalize_symbol(SYMBOL) == "BTCUSDT" else EXPERIMENT_ID,
        "aux_symbols": AUX_SYMBOLS,
        "models": BTCUSDT_LIVE_MODELS if normalize_symbol(SYMBOL) == "BTCUSDT" else [
            {"experiment_id": EXPERIMENT_ID, "weight": 1.0},
        ],
    },
}

if normalize_symbol(SYMBOL) != "BTCUSDT":
    LIVE_SYMBOL_CONFIGS["BTCUSDT"] = {
        "aux_symbols": ["ETHUSDT", "ETHBTC"],
        "models": BTCUSDT_LIVE_MODELS,
    }

# 继续加其它币种时按下面结构追加即可：
# LIVE_SYMBOL_CONFIGS["ETHUSDT"] = {
#     "aux_symbols": ["BTCUSDT", "ETHBTC"],
#     "models": [
#         {"experiment_id": "EXP_ETH_A", "weight": 0.7},
#         {"experiment_id": "EXP_ETH_B", "weight": 0.3},
#     ],
# }

SEQ_LEN = int(os.environ.get("OPT_SEQ_LEN", 672))
BATCH_SIZE = int(os.environ.get("OPT_BATCH_SIZE", 128))
EPOCHS = int(os.environ.get("OPT_EPOCHS", 100))
LEARNING_RATE = float(os.environ.get("OPT_LEARNING_RATE", 5e-6))

D_MODEL = int(os.environ.get("OPT_D_MODEL", 64))
E_LAYERS = int(os.environ.get("OPT_E_LAYERS", 1))
DROPOUT = float(os.environ.get("OPT_DROPOUT", 0.3))

CLASS_THRESHOLD = float(os.environ.get("OPT_CLASS_THRESHOLD", 0.035))
