import asyncio
import ccxt.pro as ccxtpro
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading
import random
import math
from datetime import datetime
import time
import sqlite3
import polars as pl
from quant_config.config import (
    SYMBOL, AUX_SYMBOLS, WHALE_NOTIONAL_THRESHOLD_USDT, RETAIL_NOTIONAL_THRESHOLD_USDT,
    LIVE_DATA_DIR, TEMP_DIR, PROCESSED_AGGTRADES_DIR, PROCESSED_MAIN_DIR,
    PROCESSED_AGGTRADES_ROOT, PROCESSED_MAIN_ROOT,
    processed_aggtrades_dir, processed_main_dir,
    KLINE_COLS, FUNDING_COLS, LIVE_SYMBOL_CONFIGS
)

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
DERIVATIVE_LOOKBACK_HOURS = 1250
ONE_MINUTE_HISTORY_LIMIT = DERIVATIVE_LOOKBACK_HOURS * 60 + 120
RECENT_AGGTRADE_REST_BACKFILL_HOURS = 48

def _normalize_symbol_key(raw_symbol):
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _normalize_live_configs():
    raw_configs = LIVE_SYMBOL_CONFIGS if isinstance(LIVE_SYMBOL_CONFIGS, dict) and LIVE_SYMBOL_CONFIGS else {
        SYMBOL: {"aux_symbols": AUX_SYMBOLS}
    }
    normalized = {}
    for raw_symbol, cfg in raw_configs.items():
        sym_key = _normalize_symbol_key(raw_symbol)
        cfg = cfg or {}
        normalized[sym_key] = {
            "aux_symbols": [_normalize_symbol_key(s) for s in cfg.get("aux_symbols", [])]
        }
    return normalized


LIVE_CONFIGS = _normalize_live_configs()
LIVE_MAIN_SYMBOLS = list(LIVE_CONFIGS.keys())
LIVE_MAIN_SYMBOL_SET = set(LIVE_MAIN_SYMBOLS)


def _build_subscription_specs():
    specs = {}
    for sym in LIVE_MAIN_SYMBOLS:
        specs[sym] = {"market_symbol": sym, "is_spot": False}

    for cfg in LIVE_CONFIGS.values():
        for aux_sym in cfg.get("aux_symbols", []):
            # 如果辅助参考资产本身也是实盘主合约，需要额外订阅一份现货，避免覆盖合约数据。
            store_key = f"{aux_sym}_SPOT" if aux_sym in LIVE_MAIN_SYMBOL_SET else aux_sym
            specs.setdefault(store_key, {"market_symbol": aux_sym, "is_spot": True})
    return specs


SUBSCRIPTION_SPECS = _build_subscription_specs()
all_symbols = list(SUBSCRIPTION_SPECS.keys())

for sym in all_symbols:
    spec = SUBSCRIPTION_SPECS[sym]
    market_data_store[sym] = {
        "1m": deque(maxlen=ONE_MINUTE_HISTORY_LIMIT), # 覆盖完整 1h 推理窗口的 RV 预热
        "1h": deque(maxlen=1250), # 从 500 提升到 1000，确保满足 504 需求
        "orderbook": {"bids": [], "asks": []},
        "ticker": {"last": None},
        "liquidations": deque(maxlen=1000), 
        "status": "initializing"
    }
    
    # 只有主币种才需要复杂的订单流和衍生品特征
    if not spec["is_spot"]:
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
DERIVATIVE_HISTORY_WINDOW_MS = DERIVATIVE_LOOKBACK_HOURS * 3600000
PROCESSED_DERIVATIVE_ROOT_DIRS = [PROCESSED_AGGTRADES_ROOT, PROCESSED_MAIN_ROOT]
DERIVATIVE_METRIC_KEYS = {
    "premium_history": "premium_close",
    "funding_history": "funding_rate",
    "oi_history": "sum_open_interest",
}


def print_gateway_data_chain():
    print("🧭 [网关链路] 启动恢复顺序: SQLite 自身记忆库 -> 训练集缓存/processed -> Binance API 缺口补齐")
    print("🧭 [网关链路] 实时落库范围: 1h K线+订单流、1m 显微镜、premium/funding/OI 衍生品时序")


def _is_temporal_dtype(dtype):
    """兼容 Polars 1.0+，避免使用已弃用的 TEMPORAL_DTYPES。"""
    if dtype is None:
        return False
    base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype
    return base_type in (pl.Datetime, pl.Date, pl.Time, pl.Duration)


def _load_cached_1m_history(symbol, cutoff_ts_ms):
    raw_1m_path = TEMP_DIR / "klines" / symbol / f"{symbol}_raw_1m.parquet"
    if not raw_1m_path.exists():
        return []

    try:
        schema = pl.read_parquet_schema(raw_1m_path)
        select_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in schema]
        if not {"timestamp", "open", "high", "low", "close"}.issubset(set(select_cols)):
            return []

        df = pl.read_parquet(raw_1m_path, columns=select_cols)
        if "volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("volume"))

        ts_col = "timestamp"
        ts_dtype = df.schema.get(ts_col)
        if _is_temporal_dtype(ts_dtype):
            df = df.with_columns(pl.col(ts_col).dt.epoch(time_unit="ms").alias("ts_ms"))
        else:
            df = df.with_columns(pl.col(ts_col).cast(pl.Int64, strict=False).alias("ts_ms"))

        df = df.filter(pl.col("ts_ms") >= cutoff_ts_ms).sort("ts_ms").tail(ONE_MINUTE_HISTORY_LIMIT)
        if df.is_empty():
            return []

        records = [
            [
                int(row["ts_ms"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ]
            for row in df.iter_rows(named=True)
        ]
        print(f"   ♻️ [1m本地缓存] {symbol} 复用 raw_1m.parquet {len(records)} 根。")
        return records
    except Exception as e:
        print(f"   ⚠️ [1m本地缓存] {symbol} 读取失败: {e}")
        return []


def seed_sqlite_from_training_1h(symbol):
    """用训练集 raw_1h.parquet 补 SQLite 缺口，保留自身数据库已有记录。"""
    raw_1h_path = TEMP_DIR / "klines" / symbol / f"{symbol}_raw_1h.parquet"
    if not raw_1h_path.exists():
        return 0

    try:
        schema = pl.read_parquet_schema(raw_1h_path)
        select_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"] if c in schema]
        if not {"timestamp", "open", "high", "low", "close"}.issubset(set(select_cols)):
            return 0

        df = pl.read_parquet(raw_1h_path, columns=select_cols)
        if "volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("volume"))
        if "quote_volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("quote_volume"))
        if "count" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("count"))
        if "taker_buy_volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("taker_buy_volume"))

        ts_dtype = df.schema.get("timestamp")
        if _is_temporal_dtype(ts_dtype):
            df = df.with_columns(pl.col("timestamp").dt.epoch(time_unit="ms").alias("ts_ms"))
        else:
            df = df.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False).alias("ts_ms"))

        df = df.sort("ts_ms").tail(DERIVATIVE_LOOKBACK_HOURS)
        if df.is_empty():
            return 0

        records = []
        for row in df.iter_rows(named=True):
            volume = float(row["volume"])
            taker_buy_volume = float(row["taker_buy_volume"])
            records.append((
                symbol,
                int(row["ts_ms"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                volume,
                float(row["quote_volume"]),
                int(row["count"]),
                taker_buy_volume,
                max(0.0, volume - taker_buy_volume),
                None,
                None,
                None,
            ))

        insert_ignore_memory_1h_records(records)

        print(f"   ♻️ [1h训练集补齐] {symbol} 已扫描 raw_1h.parquet {len(records)} 根，缺口已写入 SQLite。")
        return len(records)
    except Exception as e:
        print(f"   ⚠️ [1h训练集补齐] {symbol} 注入 SQLite 失败: {e}")
        return 0


def _read_raw_kline_csv(csv_path):
    for kwargs in (
        {"has_header": True, "infer_schema_length": 200},
        {"has_header": False, "new_columns": KLINE_COLS, "infer_schema_length": 200},
    ):
        try:
            df = pl.read_csv(csv_path, **kwargs)
            if {"open_time", "close"}.issubset(set(df.columns)):
                return df
        except Exception:
            continue
    return None


def _read_raw_funding_csv(csv_path):
    for kwargs in (
        {"has_header": True, "infer_schema_length": 200},
        {"has_header": False, "new_columns": FUNDING_COLS, "infer_schema_length": 200},
    ):
        try:
            df = pl.read_csv(csv_path, **kwargs)
            if {"calc_time", "last_funding_rate"}.issubset(set(df.columns)):
                return df
        except Exception:
            continue
    return None


def load_tempdir_derivative_history(symbol, cutoff_ts_ms):
    """优先读取训练阶段保留下来的 TEMP_DIR 原始缓存。"""
    cutoff_dt = datetime.utcfromtimestamp(cutoff_ts_ms / 1000.0)
    result = {}

    premium_files = sorted((TEMP_DIR / "premiumIndexKlines" / symbol).glob("*/*.csv"))
    if premium_files:
        frames = []
        for file_path in premium_files:
            df = _read_raw_kline_csv(file_path)
            if df is None:
                continue
            frames.append(
                df.select([
                    pl.from_epoch(pl.col("open_time").cast(pl.Int64, strict=False), time_unit="ms").dt.truncate("1h").alias("hour_timestamp"),
                    pl.col("close").cast(pl.Float64, strict=False).alias("premium_close")
                ]).drop_nulls(["hour_timestamp", "premium_close"])
            )
        if frames:
            merged = pl.concat(frames, how="vertical_relaxed").filter(pl.col("hour_timestamp") >= cutoff_dt).sort("hour_timestamp")
            merged = merged.group_by("hour_timestamp").agg(pl.col("premium_close").drop_nulls().last().alias("premium_close")).sort("hour_timestamp")
            if not merged.is_empty():
                result["premium_history"] = [
                    [int(row["hour_timestamp"].timestamp() * 1000), float(row["premium_close"])]
                    for row in merged.iter_rows(named=True)
                ]

    funding_files = sorted((TEMP_DIR / "fundingRate" / symbol).glob("*/*.csv"))
    if funding_files:
        frames = []
        for file_path in funding_files:
            df = _read_raw_funding_csv(file_path)
            if df is None:
                continue
            frames.append(
                df.select([
                    pl.from_epoch(pl.col("calc_time").cast(pl.Int64, strict=False), time_unit="ms").dt.truncate("1h").alias("hour_timestamp"),
                    pl.col("last_funding_rate").cast(pl.Float64, strict=False).alias("funding_rate")
                ]).drop_nulls(["hour_timestamp", "funding_rate"])
            )
        if frames:
            merged = pl.concat(frames, how="vertical_relaxed").filter(pl.col("hour_timestamp") >= cutoff_dt).sort("hour_timestamp")
            merged = merged.group_by("hour_timestamp").agg(pl.col("funding_rate").drop_nulls().last().alias("funding_rate")).sort("hour_timestamp")
            if not merged.is_empty():
                result["funding_history"] = [
                    [int(row["hour_timestamp"].timestamp() * 1000), float(row["funding_rate"])]
                    for row in merged.iter_rows(named=True)
                ]

    metrics_cache_path = TEMP_DIR / "metrics" / symbol / f"{symbol}_metrics_hourly.parquet"
    if metrics_cache_path.exists():
        try:
            merged = (
                pl.read_parquet(metrics_cache_path, columns=["hour_timestamp", "sum_open_interest"])
                .filter(pl.col("hour_timestamp") >= cutoff_dt)
                .sort("hour_timestamp")
                .drop_nulls(["hour_timestamp", "sum_open_interest"])
            )
            if not merged.is_empty():
                result["oi_history"] = [
                    [int(row["hour_timestamp"].timestamp() * 1000), float(row["sum_open_interest"])]
                    for row in merged.iter_rows(named=True)
                ]
        except Exception as e:
            print(f"   ⚠️ [TEMP_DIR衍生品缓存] {symbol} Metrics 小时级缓存读取失败: {e}")

    metrics_files = sorted((TEMP_DIR / "metrics" / symbol).glob("*/*.csv"))
    if "oi_history" not in result and metrics_files:
        frames = []
        for file_path in metrics_files:
            try:
                df = pl.read_csv(file_path, has_header=True, infer_schema_length=0).select([
                    "create_time", "sum_open_interest"
                ]).with_columns([
                    pl.col("create_time").str.slice(0, 19).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("exact_timestamp"),
                    pl.col("sum_open_interest").cast(pl.Float64, strict=False),
                ])
                frames.append(df)
            except Exception:
                continue
        if frames:
            merged = pl.concat(frames, how="vertical_relaxed").with_columns(
                pl.col("exact_timestamp").dt.truncate("1h").alias("hour_timestamp")
            ).filter(pl.col("hour_timestamp") >= cutoff_dt).sort("exact_timestamp").group_by("hour_timestamp").last().sort("hour_timestamp")
            if not merged.is_empty():
                oi_df = merged.select(["hour_timestamp", "sum_open_interest"]).drop_nulls()
                if not oi_df.is_empty():
                    result["oi_history"] = [
                        [int(row["hour_timestamp"].timestamp() * 1000), float(row["sum_open_interest"])]
                        for row in oi_df.iter_rows(named=True)
                    ]

    if result:
        print(f"   ♻️ [TEMP_DIR衍生品缓存] {symbol} 命中: {list(result.keys())}")
    return result


def _load_single_offline_derivative_file(file_path):
    """从单个离线文件中抽取可用于实盘的衍生品时序字段。"""
    suffix = file_path.suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        return None

    try:
        if suffix == ".parquet":
            schema = pl.read_parquet_schema(file_path)
        else:
            schema = pl.read_csv(file_path, n_rows=5, infer_schema_length=50).schema
    except Exception:
        return None

    col_names = set(schema.keys())
    time_col = next((c for c in ["hour_timestamp", "timestamp", "exact_timestamp", "open_time"] if c in col_names), None)
    if time_col is None:
        return None

    normalized_cols = {}
    for normalized_name, aliases in {
        "premium_close": ["premium_close", "premium_index", "premium_index_close"],
        "funding_rate": ["last_funding_rate", "funding_rate"],
        "sum_open_interest": ["sum_open_interest", "open_interest"],
    }.items():
        match = next((alias for alias in aliases if alias in col_names), None)
        if match:
            normalized_cols[normalized_name] = match

    if not normalized_cols:
        return None

    select_cols = [time_col] + list(dict.fromkeys(normalized_cols.values()))
    try:
        if suffix == ".parquet":
            df = pl.read_parquet(file_path, columns=select_cols)
        else:
            df = pl.read_csv(file_path, columns=select_cols, infer_schema_length=200)
    except Exception:
        return None

    ts_dtype = df.schema.get(time_col)
    if _is_temporal_dtype(ts_dtype):
        ts_expr = pl.col(time_col).dt.truncate("1h")
    elif ts_dtype == pl.Utf8:
        ts_expr = pl.col(time_col).str.strptime(pl.Datetime, strict=False).dt.truncate("1h")
    else:
        ts_expr = pl.from_epoch(pl.col(time_col).cast(pl.Int64, strict=False), time_unit="ms").dt.truncate("1h")

    exprs = [ts_expr.alias("hour_timestamp")]
    for normalized_name, source_col in normalized_cols.items():
        exprs.append(pl.col(source_col).cast(pl.Float64, strict=False).alias(normalized_name))

    return df.select(exprs).drop_nulls(["hour_timestamp"])


def load_processed_derivative_history(symbol, cutoff_ts_ms):
    """最后一层兜底：扫描 processed 目录中的宽表或聚合特征文件。"""
    frames = []
    scanned_files = 0

    candidate_dirs = [
        processed_aggtrades_dir(symbol),
        processed_main_dir(symbol),
        *PROCESSED_DERIVATIVE_ROOT_DIRS,
    ]
    for directory in dict.fromkeys(candidate_dirs):
        if not directory.exists():
            continue

        symbol_files = sorted(directory.glob(f"{symbol}*.parquet")) + sorted(directory.glob(f"{symbol}*.csv"))
        candidate_files = symbol_files

        for file_path in candidate_files:
            scanned_files += 1
            df = _load_single_offline_derivative_file(file_path)
            if df is not None and not df.is_empty():
                frames.append(df)

    if not frames:
        return {}

    cutoff_dt = datetime.utcfromtimestamp(cutoff_ts_ms / 1000.0)
    merged = pl.concat(frames, how="diagonal_relaxed").filter(
        pl.col("hour_timestamp") >= cutoff_dt
    ).sort("hour_timestamp")

    if merged.is_empty():
        return {}

    value_cols = [c for c in ["premium_close", "funding_rate", "sum_open_interest"] if c in merged.columns]
    agg_exprs = [pl.col(c).drop_nulls().last().alias(c) for c in value_cols]
    if not agg_exprs:
        return {}

    merged = merged.group_by("hour_timestamp").agg(agg_exprs).sort("hour_timestamp")
    if merged.is_empty():
        return {}

    result = {}
    for source_col, target_key in {
        "premium_close": "premium_history",
        "funding_rate": "funding_history",
        "sum_open_interest": "oi_history",
    }.items():
        if source_col not in merged.columns:
            continue
        sub_df = merged.select(["hour_timestamp", source_col]).drop_nulls()
        if sub_df.is_empty():
            continue
        result[target_key] = [
            [int(row["hour_timestamp"].timestamp() * 1000), float(row[source_col])]
            for row in sub_df.iter_rows(named=True)
        ]

    if result:
        print(f"   📚 [processed衍生品兜底] {symbol} 已从本地扫描 {scanned_files} 个文件并装载历史兜底。")
    return result


def merge_derivative_history(primary_history, offline_history, cutoff_ts_ms):
    """用离线历史回补早期缺口，但始终让 API 最新值覆盖旧值。"""
    merged = {}

    for source in (offline_history or []):
        if len(source) != 2 or source[1] is None:
            continue
        merged[int(source[0])] = float(source[1])

    for source in (primary_history or []):
        if len(source) != 2 or source[1] is None:
            continue
        merged[int(source[0])] = float(source[1])

    if not merged:
        return []

    return [[ts, merged[ts]] for ts in sorted(merged) if ts >= cutoff_ts_ms]


def upsert_memory_1h_records(records):
    if not records:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO memory_1h (
                symbol, timestamp, open, high, low, close, volume,
                quote_volume, count, taker_buy_volume, taker_sell_volume,
                whale_buy, whale_sell, retail_vol
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                quote_volume=COALESCE(excluded.quote_volume, memory_1h.quote_volume),
                count=COALESCE(excluded.count, memory_1h.count),
                taker_buy_volume=COALESCE(excluded.taker_buy_volume, memory_1h.taker_buy_volume),
                taker_sell_volume=COALESCE(excluded.taker_sell_volume, memory_1h.taker_sell_volume),
                whale_buy=COALESCE(excluded.whale_buy, memory_1h.whale_buy),
                whale_sell=COALESCE(excluded.whale_sell, memory_1h.whale_sell),
                retail_vol=COALESCE(excluded.retail_vol, memory_1h.retail_vol)
            """,
            records,
        )
        conn.commit()


def insert_ignore_memory_1h_records(records):
    if not records:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO memory_1h VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()


def upsert_memory_1m_records(symbol, candles):
    records = [
        (symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        for k in (candles or [])
        if len(k) >= 6
    ]
    if not records:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO memory_1m (symbol, timestamp, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            records,
        )
        conn.commit()


def insert_ignore_memory_1m_records(symbol, candles):
    records = [
        (symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        for k in (candles or [])
        if len(k) >= 6
    ]
    if not records:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO memory_1m VALUES (?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()


def upsert_derivative_history_records(symbol, hist_key, history):
    metric = DERIVATIVE_METRIC_KEYS.get(hist_key)
    if not metric:
        return

    records = [
        (symbol, metric, int(row[0]), float(row[1]))
        for row in (history or [])
        if len(row) == 2 and row[1] is not None
    ]
    if not records:
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO derivative_history (symbol, metric, timestamp, value)
            VALUES (?,?,?,?)
            ON CONFLICT(symbol, metric, timestamp) DO UPDATE SET value=excluded.value
            """,
            records,
        )
        conn.commit()


def load_memory_1h_rows(symbol, limit=DERIVATIVE_LOOKBACK_HOURS):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM memory_1h WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit),
        )
        rows = cursor.fetchall()
    rows.reverse()
    return rows


def load_memory_1m_history(symbol, cutoff_ts_ms):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM memory_1m
            WHERE symbol=? AND timestamp>=?
            ORDER BY timestamp ASC
            """,
            (symbol, cutoff_ts_ms),
        )
        return [
            [int(ts), float(o), float(h), float(l), float(c), float(v)]
            for ts, o, h, l, c, v in cursor.fetchall()
        ][-ONE_MINUTE_HISTORY_LIMIT:]


def load_derivative_history_from_db(symbol, hist_key, cutoff_ts_ms):
    metric = DERIVATIVE_METRIC_KEYS[hist_key]
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, value
            FROM derivative_history
            WHERE symbol=? AND metric=? AND timestamp>=?
            ORDER BY timestamp ASC
            """,
            (symbol, metric, cutoff_ts_ms),
        )
        return [[int(ts), float(value)] for ts, value in cursor.fetchall()]


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_1m (
                symbol TEXT,
                timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS derivative_history (
                symbol TEXT,
                metric TEXT,
                timestamp INTEGER,
                value REAL,
                PRIMARY KEY (symbol, metric, timestamp)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_1m_symbol_ts ON memory_1m(symbol, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derivative_symbol_metric_ts ON derivative_history(symbol, metric, timestamp)")
init_sqlite_db()


def build_exchange(is_spot):
    """按 symbol 隔离 exchange，避免一个交易对断线时把同引擎下其他 symbol 一起清掉。"""
    exchange_cls = ccxtpro.binance if is_spot else ccxtpro.binanceusdm
    return exchange_cls({
        "enableRateLimit": True,
    })


def should_log_derivative_error(error_state, endpoint, e, throttle_seconds=300):
    """同类 REST 错误节流，避免地区/网络不可用时每分钟刷屏。"""
    now = time.time()
    signature = f"{type(e).__name__}:{str(e)[:180]}"
    previous = error_state.get(endpoint)
    if previous and previous["signature"] == signature and now - previous["last_log_ts"] < throttle_seconds:
        return False
    error_state[endpoint] = {"signature": signature, "last_log_ts": now}
    return True


async def safe_derivative_request(symbol, endpoint, request_factory, error_state):
    """单端点容错：一个 Binance 衍生品接口失败时，不阻断其它接口更新。"""
    try:
        async with rest_semaphore:
            return await request_factory()
    except Exception as e:
        if should_log_derivative_error(error_state, endpoint, e):
            print(f"   ⚠️ [REST 衍生品{endpoint}失败] {symbol}: {type(e).__name__} -> {e}；保留上一轮/本地兜底数据。")
        return None

# ==========================================
# 🛰️ WebSocket 搬运工：死循环监听流
# ==========================================
async def handle_ws_exception(exchange, symbol, stream_name, e, was_dead, retry_delay):
    is_timeout = isinstance(e, asyncio.TimeoutError)
    err_str = str(e).lower()
    
    if "假死重连" in err_str:
        return True, retry_delay, 1
        
    if is_timeout or "timeout" in err_str or "network" in err_str or "closed" in err_str or "client connection dropped" in err_str:
        if is_timeout:
            print(f"   ⚠️ [WS 看门狗] {symbol} {stream_name} 超过 180 秒无变动，触发重连！")
        else:
            print(f"   ⚠️ [WS 断线] {symbol} {stream_name} 连接异常: {e}，清理连接并休眠 {retry_delay}s...")
            
        for url, client in list(exchange.clients.items()):
            for f in list(client.futures.values()):
                if isinstance(f, list):
                    for sub_f in f: sub_f.add_done_callback(lambda x: x.exception())
                else:
                    f.add_done_callback(lambda x: x.exception())
            client.reject(Exception(f"{symbol} {stream_name}触发假死重连"))
            exchange.clients.pop(url, None)
            
        sleep_time = retry_delay
        new_retry_delay = min(60, retry_delay * 2)
        return True, new_retry_delay, sleep_time
        
    if stream_name != "最新价流": # 最新价流报错太多，屏蔽未知异常打印
        print(f"   ❌ [WS 未知异常] {symbol} {stream_name}: {type(e).__name__} -> {e}")
    return was_dead, retry_delay, 5

async def ohlcv_loop(exchange, symbol, swap_symbol, timeframe, engine_type, startup_delay):
    """ 纯粹的 K 线 WebSocket 监听流 (历史预热已交由中台全局引擎处理) """
    # 严格的错峰启动，杜绝 Binance 并发订阅静默丢包
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} ({timeframe}) {engine_type}K线订阅...")
    
    was_dead = False
    retry_delay = 2
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
            retry_delay = 2
                
            for candle in ohlcvs:
                q = market_data_store[symbol][timeframe]
                
                # 同频更新去重 (Binance 的 WS 会每秒推送同一根 K 线的最新状态)
                if len(q) > 0 and q[-1][0] == candle[0]:
                    q[-1] = candle
                else:
                    q.append(candle)

                if timeframe == "1h":
                    stats = market_data_store[symbol].get("taker_stats_1h", {}).get(int(candle[0]), {})
                    record = (
                        symbol, int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5]),
                        stats.get("quote_volume"), stats.get("count"),
                        stats.get("taker_buy_volume"), stats.get("taker_sell_volume"),
                        stats.get("whale_buy"), stats.get("whale_sell"), stats.get("retail_vol"),
                    )
                    await asyncio.to_thread(upsert_memory_1h_records, [record])
                elif timeframe == "1m":
                    await asyncio.to_thread(upsert_memory_1m_records, symbol, [candle])
                    
        except Exception as e:
            was_dead, retry_delay, sleep_time = await handle_ws_exception(
                exchange, symbol, f"({timeframe}) K线流", e, was_dead, retry_delay
            )
            await asyncio.sleep(sleep_time)

async def orderbook_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听盘口流 (增量自动合并) """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}盘口深度订阅...")
    
    was_dead = False
    retry_delay = 2
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
            retry_delay = 2
                
            market_data_store[symbol]["orderbook"] = {
                "bids": orderbook['bids'],
                "asks": orderbook['asks']
            }
        except Exception as e:
            was_dead, retry_delay, sleep_time = await handle_ws_exception(
                exchange, symbol, "盘口深度流", e, was_dead, retry_delay
            )
            await asyncio.sleep(sleep_time)

async def ticker_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听最新价流 """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}最新价流订阅...")
    
    was_dead = False
    retry_delay = 2
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
            retry_delay = 2
                
            market_data_store[symbol]["ticker"]["last"] = ticker['last']
            market_data_store[symbol]["status"] = "live"
        except Exception as e:
            was_dead, retry_delay, sleep_time = await handle_ws_exception(
                exchange, symbol, "最新价流", e, was_dead, retry_delay
            )
            await asyncio.sleep(sleep_time)
async def liquidation_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听币安实时爆仓流 (CCXT 强力防呆兼容版) """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}爆仓流订阅...")
    
    was_dead = False
    retry_delay = 2
    while True:
        try:
            liquidations = await exchange.watch_liquidations(swap_symbol)
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}爆仓流已成功恢复！")
                was_dead = False
            retry_delay = 2
            
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
            was_dead, retry_delay, sleep_time = await handle_ws_exception(
                exchange, symbol, "爆仓流", e, was_dead, retry_delay
            )
            await asyncio.sleep(sleep_time)

def normalize_live_taker_stats(stats):
    """实时逐笔成交桶允许从 SQLite 的 NULL 底座恢复成可累加状态。"""
    numeric_defaults = {
        "taker_buy_volume": 0.0,
        "taker_sell_volume": 0.0,
        "quote_volume": 0.0,
        "count": 0,
        "whale_buy": 0.0,
        "whale_sell": 0.0,
        "retail_vol": 0.0,
    }
    for key, default in numeric_defaults.items():
        if stats.get(key) is None:
            stats[key] = default
    return stats

async def trade_loop(exchange, symbol, swap_symbol, engine_type, startup_delay):
    """ 监听逐笔成交流，降维聚合为 1h 级别的高级量价与订单流特征 """
    await asyncio.sleep(startup_delay)
    print(f"📡 [WS] 启动 {symbol} {engine_type}逐笔成交流订阅 (包含大单拆解)...")
    
    was_dead = False
    retry_delay = 2
    while True:
        try:
            trades = await asyncio.wait_for(
                exchange.watch_trades(swap_symbol),
                timeout=180.0
            )
            if was_dead:
                print(f"   ✅ [WS 恢复] {symbol} {engine_type}逐笔成交流已成功恢复！")
                was_dead = False
            retry_delay = 2
                
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
                
                stats = normalize_live_taker_stats(bucket[hour_ts])
                stats["quote_volume"] += cost
                stats["count"] += 1
                
                is_buy = t['side'] == 'buy'
                if is_buy:
                    stats["taker_buy_volume"] += amount
                    if cost >= WHALE_NOTIONAL_THRESHOLD_USDT:
                        stats["whale_buy"] += amount
                else:
                    stats["taker_sell_volume"] += amount
                    if cost >= WHALE_NOTIONAL_THRESHOLD_USDT:
                        stats["whale_sell"] += amount
                        
                if cost <= RETAIL_NOTIONAL_THRESHOLD_USDT:
                    stats["retail_vol"] += amount

            latest_candle = market_data_store[symbol]["1h"][-1] if market_data_store[symbol]["1h"] else None
            if latest_candle:
                ts = int(latest_candle[0])
                stats = market_data_store[symbol]["taker_stats_1h"].get(ts, {})
                record = (
                    symbol, ts,
                    float(latest_candle[1]), float(latest_candle[2]), float(latest_candle[3]), float(latest_candle[4]), float(latest_candle[5]),
                    stats.get("quote_volume"), stats.get("count"),
                    stats.get("taker_buy_volume"), stats.get("taker_sell_volume"),
                    stats.get("whale_buy"), stats.get("whale_sell"), stats.get("retail_vol"),
                )
                await asyncio.to_thread(upsert_memory_1h_records, [record])
            
            # 定期垃圾回收：配合 1250 小时的回溯需求，延长保留期
            current_hour = int(datetime.utcnow().timestamp() / 3600) * 3600000
            old_keys = [k for k in list(bucket.keys()) if k < current_hour - 1250 * 3600000]
            for k in old_keys:
                del bucket[k]
                
        except Exception as e:
            was_dead, retry_delay, sleep_time = await handle_ws_exception(
                exchange, symbol, "逐笔成交流", e, was_dead, retry_delay
            )
            await asyncio.sleep(sleep_time)

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
    error_state = {}

    while True:
        # 1. 维护 1h 溢价指数时序 (Premium Index)
        prem_klines = await safe_derivative_request(
            symbol,
            "PremiumIndexKlines",
            lambda: exchange.fapiPublicGetPremiumIndexKlines({
                    'symbol': raw_sym, 'interval': '1h', 'limit': 2
            }),
            error_state,
        )
        if prem_klines:
            hist = market_data_store[symbol].setdefault("premium_history", [])
            changed_rows = []
            for k in prem_klines:
                ts, val = int(k[0]), float(k[4]) # 4 is close
                if not hist or ts > hist[-1][0]:
                    hist.append([ts, val])
                    changed_rows.append([ts, val])
                elif ts == hist[-1][0]:
                    hist[-1] = [ts, val]
                    changed_rows.append([ts, val])
            if changed_rows:
                await asyncio.to_thread(upsert_derivative_history_records, symbol, "premium_history", changed_rows)

        # 2. 维护 1h 未平仓合约量时序 (Open Interest)
        oi_hist_data = await safe_derivative_request(
            symbol,
            "OpenInterestHist",
            lambda: exchange.fapiDataGetOpenInterestHist({
                    'symbol': raw_sym, 'period': '1h', 'limit': 2
            }),
            error_state,
        )
        if oi_hist_data:
            hist = market_data_store[symbol].setdefault("oi_history", [])
            changed_rows = []
            for x in oi_hist_data:
                ts, val = int(x['timestamp']), float(x['sumOpenInterest'])
                if not hist or ts > hist[-1][0]:
                    hist.append([ts, val])
                    changed_rows.append([ts, val])
                elif ts == hist[-1][0]:
                    hist[-1] = [ts, val]
                    changed_rows.append([ts, val])
            if changed_rows:
                await asyncio.to_thread(upsert_derivative_history_records, symbol, "oi_history", changed_rows)

        # 3. 维护资金费率时序 (Funding Rate)
        fr_hist_data = await safe_derivative_request(
            symbol,
            "FundingRate",
            lambda: exchange.fapiPublicGetFundingRate({
                    'symbol': raw_sym, 'limit': 2
            }),
            error_state,
        )
        if fr_hist_data:
            hist = market_data_store[symbol].setdefault("funding_history", [])
            changed_rows = []
            for x in fr_hist_data:
                ts, val = int(x['fundingTime']), float(x['fundingRate'])
                if not hist or ts > hist[-1][0]:
                    hist.append([ts, val])
                    changed_rows.append([ts, val])
                elif ts == hist[-1][0]:
                    hist[-1] = [ts, val]
                    changed_rows.append([ts, val])
            if changed_rows:
                await asyncio.to_thread(upsert_derivative_history_records, symbol, "funding_history", changed_rows)

        # ==========================================
        # 🧹 内存泄露防护：控制历史数组的最大长度，但必须保留完整 1250h 推理窗口
        # ==========================================
        for key in ["premium_history", "oi_history", "funding_history"]:
            if key in market_data_store[symbol] and len(market_data_store[symbol][key]) > DERIVATIVE_LOOKBACK_HOURS + 64:
                market_data_store[symbol][key] = market_data_store[symbol][key][-(DERIVATIVE_LOOKBACK_HOURS + 64):]
            
        # 轮询抖动设计：防止触发 Binance 每分钟请求权重超限
        await asyncio.sleep(60 + random.uniform(0, 5.0))


def hydrate_1h_memory_from_rows(symbol, rows, is_spot):
    market_data_store[symbol]["1h"].clear()
    if not is_spot:
        market_data_store[symbol]["taker_stats_1h"].clear()

    for r in rows:
        ts = r[1]
        market_data_store[symbol]["1h"].append([ts, r[2], r[3], r[4], r[5], r[6]])
        if not is_spot:
            market_data_store[symbol]["taker_stats_1h"][ts] = {
                "quote_volume": r[7], "count": r[8],
                "taker_buy_volume": r[9], "taker_sell_volume": r[10],
                "whale_buy": r[11], "whale_sell": r[12], "retail_vol": r[13]
            }


def find_1h_gaps(timestamps):
    return [
        (timestamps[i], timestamps[i + 1])
        for i in range(len(timestamps) - 1)
        if timestamps[i + 1] - timestamps[i] != 3600000
    ]


async def backfill_1h_from_api(exchange, symbol, swap_symbol, is_spot, start_ts, end_ts):
    cursor = start_ts
    while cursor <= end_ts:
        await fetch_and_store_range(exchange, symbol, swap_symbol, is_spot, startTime=cursor, limit=1000)
        cursor += 1000 * 3600000
        await asyncio.sleep(0.2)


async def load_1m_sqlite_training_api(exchange, symbol, swap_symbol):
    end_time = int(time.time() * 1000)
    cutoff_ts_ms = end_time - DERIVATIVE_LOOKBACK_HOURS * 3600000

    print(f"🔍 [网关自愈] 1m 链路: SQLite -> 训练集 raw_1m -> Binance API 缺口补齐...")
    all_1m = await asyncio.to_thread(load_memory_1m_history, symbol, cutoff_ts_ms)
    if all_1m:
        print(f"   ✅ [1m SQLite] {symbol} 已恢复 {len(all_1m)} 根。")
    else:
        print(f"   ⏳ [1m SQLite] {symbol} 暂无本地记忆，尝试训练集补齐。")

    training_1m = await asyncio.to_thread(_load_cached_1m_history, symbol, cutoff_ts_ms)
    if training_1m:
        await asyncio.to_thread(insert_ignore_memory_1m_records, symbol, training_1m)
        all_1m = await asyncio.to_thread(load_memory_1m_history, symbol, cutoff_ts_ms)

    curr_ts = all_1m[-1][0] + 60000 if all_1m else cutoff_ts_ms
    if curr_ts < end_time:
        print(f"   🔎 [1m API补洞] {symbol} 从 {datetime.utcfromtimestamp(curr_ts / 1000):%Y-%m-%d %H:%M} UTC 补到当前。")

    while curr_ts < end_time:
        async with rest_semaphore:
            batch = await exchange.fetch_ohlcv(swap_symbol, '1m', since=curr_ts, limit=1500)
        if not batch:
            break
        await asyncio.to_thread(upsert_memory_1m_records, symbol, batch)
        curr_ts = int(batch[-1][0]) + 60000
        await asyncio.sleep(0.1)

    all_1m = await asyncio.to_thread(load_memory_1m_history, symbol, cutoff_ts_ms)
    market_data_store[symbol]["1m"].clear()
    market_data_store[symbol]["1m"].extend(all_1m)
    print(f"   ✅ {symbol} 1m 高频数据装填完毕，共 {len(all_1m)} 根。")


async def fetch_derivatives_from_api(exchange, symbol, swap_symbol, cutoff_ts_ms):
    raw_sym = exchange.market_id(swap_symbol)
    end_time = int(time.time() * 1000)
    result = {}

    try:
        async with rest_semaphore:
            prem_klines = await exchange.fapiPublicGetPremiumIndexKlines({'symbol': raw_sym, 'interval': '1h', 'limit': 1500})
        result["premium_history"] = [[int(k[0]), float(k[4])] for k in prem_klines if int(k[0]) >= cutoff_ts_ms]
    except Exception as e:
        print(f"   ⚠️ {symbol} premium API 补齐失败: {e}")

    try:
        async with rest_semaphore:
            fr_hist = await exchange.fapiPublicGetFundingRate({'symbol': raw_sym, 'limit': 500})
        result["funding_history"] = [[int(x['fundingTime']), float(x['fundingRate'])] for x in fr_hist if int(x['fundingTime']) >= cutoff_ts_ms]
    except Exception as e:
        print(f"   ⚠️ {symbol} funding API 补齐失败: {e}")

    try:
        oi_hist = []
        curr_end = end_time
        while len(oi_hist) < DERIVATIVE_LOOKBACK_HOURS:
            async with rest_semaphore:
                batch = await exchange.fapiDataGetOpenInterestHist({'symbol': raw_sym, 'period': '1h', 'limit': 500, 'endTime': curr_end})
            if not batch:
                break
            oi_hist = batch + oi_hist
            curr_end = int(batch[0]['timestamp']) - 1
            await asyncio.sleep(0.2)
        result["oi_history"] = [[int(x['timestamp']), float(x['sumOpenInterest'])] for x in oi_hist if int(x['timestamp']) >= cutoff_ts_ms]
    except Exception as e:
        print(f"   ⚠️ {symbol} open interest API 补齐失败: {e}")

    return result


async def smart_warmup_and_recover(exchange, symbol, swap_symbol, is_spot):
    """
    自愈式预热：SQLite 自身库优先，训练集补缺，其余缺口再请求 API。
    """
    print(f"🔍 [网关自愈] {symbol} 1h 链路: SQLite -> 训练集 raw_1h -> Binance API 缺口补齐...")

    rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)
    print(f"   ✅ [1h SQLite] {symbol} 已恢复 {len(rows)} 根。")

    if len(rows) < DERIVATIVE_LOOKBACK_HOURS:
        print(f"   ⏳ [1h训练集补齐] {symbol} SQLite 不足 {DERIVATIVE_LOOKBACK_HOURS} 根，扫描训练缓存。")
        await asyncio.to_thread(seed_sqlite_from_training_1h, symbol)
        rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)

    timestamps = [int(r[1]) for r in rows]
    target_latest_hour = int(time.time() / 3600) * 3600000

    if timestamps:
        newest_ts = timestamps[-1]
        if newest_ts + 3600000 < target_latest_hour:
            print(f"   🔎 [1h API补洞] {symbol} 本地最新到 {datetime.utcfromtimestamp(newest_ts / 1000):%Y-%m-%d %H:%M} UTC，补齐到当前小时。")
            await backfill_1h_from_api(exchange, symbol, swap_symbol, is_spot, newest_ts + 3600000, target_latest_hour)
            rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)
            timestamps = [int(r[1]) for r in rows]
    else:
        print(f"   🔎 [1h API补洞] {symbol} SQLite/训练集均无数据，直接从 API 回填最近窗口。")
        await backfill_1h_from_api(
            exchange,
            symbol,
            swap_symbol,
            is_spot,
            target_latest_hour - (DERIVATIVE_LOOKBACK_HOURS - 1) * 3600000,
            target_latest_hour,
        )
        rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)
        timestamps = [int(r[1]) for r in rows]

    if len(rows) < DERIVATIVE_LOOKBACK_HOURS and timestamps:
        missing = DERIVATIVE_LOOKBACK_HOURS - len(rows)
        oldest_ts = timestamps[0]
        print(f"   🔎 [1h API回溯] {symbol} 仍缺 {missing} 根，向过去回溯补齐。")
        await backfill_1h_from_api(
            exchange,
            symbol,
            swap_symbol,
            is_spot,
            oldest_ts - missing * 3600000,
            oldest_ts - 3600000,
        )
        rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)
        timestamps = [int(r[1]) for r in rows]

    gaps = find_1h_gaps(timestamps)
    if gaps:
        print(f"   ⚠️ {symbol} 侦测到 {len(gaps)} 处 1h 空隙，使用 API 精准回填。")
        for start, end in gaps:
            await backfill_1h_from_api(exchange, symbol, swap_symbol, is_spot, start + 3600000, end - 3600000)
        rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)
        timestamps = [int(r[1]) for r in rows]

    if not find_1h_gaps(timestamps):
        print(f"   ✅ {symbol} 1h 时间轴连续。")

    # 🚨 触发一次性回填
    if not is_spot:
        await sync_local_aggtrades_to_db(symbol)
        await backfill_recent_aggtrades_from_rest(exchange, symbol, swap_symbol)
        rows = await asyncio.to_thread(load_memory_1h_rows, symbol, DERIVATIVE_LOOKBACK_HOURS)

    hydrate_1h_memory_from_rows(symbol, rows, is_spot)

    if not is_spot:
        await load_1m_sqlite_training_api(exchange, symbol, swap_symbol)

        print(f"🔍 [网关自愈] 衍生品链路: SQLite -> 训练集缓存/processed -> Binance API 最新补齐...")
        cutoff_ts_ms = int(time.time() * 1000) - DERIVATIVE_HISTORY_WINDOW_MS
        db_derivatives = {
            hist_key: await asyncio.to_thread(load_derivative_history_from_db, symbol, hist_key, cutoff_ts_ms)
            for hist_key in DERIVATIVE_METRIC_KEYS
        }
        temp_derivatives = await asyncio.to_thread(load_tempdir_derivative_history, symbol, cutoff_ts_ms)
        processed_derivatives = await asyncio.to_thread(load_processed_derivative_history, symbol, cutoff_ts_ms)
        api_derivatives = await fetch_derivatives_from_api(exchange, symbol, swap_symbol, cutoff_ts_ms)

        for hist_key in DERIVATIVE_METRIC_KEYS:
            merged_hist = merge_derivative_history(db_derivatives.get(hist_key, []), temp_derivatives.get(hist_key, []), cutoff_ts_ms)
            merged_hist = merge_derivative_history(merged_hist, processed_derivatives.get(hist_key, []), cutoff_ts_ms)
            merged_hist = merge_derivative_history(api_derivatives.get(hist_key, []), merged_hist, cutoff_ts_ms)
            if merged_hist:
                market_data_store[symbol][hist_key] = merged_hist[-(DERIVATIVE_LOOKBACK_HOURS + 64):]
                await asyncio.to_thread(upsert_derivative_history_records, symbol, hist_key, market_data_store[symbol][hist_key])
                print(f"   ✅ [衍生品装填] {symbol} {hist_key} 共 {len(market_data_store[symbol][hist_key])} 点。")
async def fetch_and_store_range(exchange, symbol, swap_symbol, is_spot, startTime=None, limit=500):
    """通用拉取并落库逻辑，直接从交易所获取物理真实 K 线"""
    params = {'startTime': startTime} if startTime else {}
    try:
        async with rest_semaphore:
            if not is_spot:
                klines = await exchange.fapiPublicGetKlines({'symbol': exchange.market_id(swap_symbol), 'interval': '1h', 'limit': limit, **params})
            else:
                klines = await exchange.fetch_ohlcv(swap_symbol, '1h', since=startTime, limit=limit)
        
        new_records = []
        for k in klines:
            vol = float(k[5])
            new_records.append((
                symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), vol,
                float(k[7]) if len(k)>7 else 0.0, 
                int(k[8]) if len(k)>8 else 0,     
                float(k[9]) if len(k)>9 else 0.0, 
                vol - float(k[9]) if len(k)>9 else 0.0,
                # 🚨 修正：对于缺失的高频拆单数据，使用 NaN (或 None 存入 DB 时转 Null)，杜绝注入虚假分布
                None, None, None 
            ))
        
        if new_records:
            await asyncio.to_thread(upsert_memory_1h_records, new_records)
            
    except Exception as e:
        print(f"   ❌ {symbol} 物理回填失败: {e}")

async def sync_local_aggtrades_to_db(symbol):
    """一次性动作：从本地 Parquet 抽取大户/散户拆单特征覆盖到 SQLite"""
    print(f"📂 [订单流预热] 尝试从本地 Parquet 回填 {symbol} 的微观特征...")
    try:
        agg_dir = processed_aggtrades_dir(symbol)
        parquet_files = sorted(agg_dir.glob(f"{symbol}_aggTrades_*.parquet"))
        scan_pattern = agg_dir / f"{symbol}_aggTrades_*.parquet"
        if not parquet_files:
            parquet_files = sorted(PROCESSED_AGGTRADES_ROOT.glob(f"{symbol}_aggTrades_*.parquet"))
            scan_pattern = PROCESSED_AGGTRADES_ROOT / f"{symbol}_aggTrades_*.parquet"
        if not parquet_files:
            return

        df = (
            pl.scan_parquet(scan_pattern)
            .collect()
            .sort("hour_timestamp")
            .unique(subset=["hour_timestamp"], keep="last")
            .with_columns(
                pl.col("hour_timestamp").dt.epoch(time_unit="ms").alias("timestamp_ms")
            )
        )
        records = list(df.select(["whale_buy", "whale_sell", "retail_vol", "timestamp_ms"]).iter_rows())
        
        def _update_db():
            with sqlite3.connect(DB_PATH) as conn:
                conn.executemany(
                    "UPDATE memory_1h SET whale_buy=?, whale_sell=?, retail_vol=? WHERE symbol=? AND timestamp=?",
                    [(wb, ws, rv, symbol, int(ts_ms)) for wb, ws, rv, ts_ms in records]
                )
                conn.commit()
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM memory_1h WHERE symbol=? AND retail_vol IS NOT NULL",
                    (symbol,)
                )
                updated_rows = int(cur.fetchone()[0])
                return updated_rows

        updated_rows = await asyncio.to_thread(_update_db)
        print(f"   ✅ 历史订单流特征已成功挂载入网关底座，当前命中 {updated_rows} 个小时桶。")
    except Exception as e:
        print(f"   ❌ 回填订单流特征异常: {e}")

async def backfill_recent_aggtrades_from_rest(exchange, symbol, swap_symbol):
    """用 Binance aggTrades REST 补齐本地日包尚未覆盖的近期 whale/retail 缺口。"""
    current_hour = int(time.time() / 3600) * 3600000
    cutoff_ts = current_hour - RECENT_AGGTRADE_REST_BACKFILL_HOURS * 3600000

    def _find_missing_hours():
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT timestamp FROM memory_1h
                WHERE symbol=?
                  AND timestamp>=?
                  AND timestamp<?
                  AND (whale_buy IS NULL OR whale_sell IS NULL OR retail_vol IS NULL)
                ORDER BY timestamp ASC
                """,
                (symbol, cutoff_ts, current_hour),
            )
            return [int(r[0]) for r in cur.fetchall()]

    missing_hours = await asyncio.to_thread(_find_missing_hours)
    if not missing_hours:
        print(f"   ✅ [近期订单流补洞] {symbol} 最近 {RECENT_AGGTRADE_REST_BACKFILL_HOURS}h whale/retail 已完整。")
        return

    start_time = min(missing_hours)
    end_time = current_hour - 1
    raw_sym = exchange.market_id(swap_symbol)
    buckets = {}
    cursor_time = start_time
    request_count = 0

    print(
        f"   🔎 [近期订单流补洞] {symbol} 发现 {len(missing_hours)} 个小时缺少 whale/retail，"
        f"正在拉取 aggTrades: {datetime.utcfromtimestamp(start_time / 1000):%Y-%m-%d %H:%M} -> "
        f"{datetime.utcfromtimestamp(end_time / 1000):%Y-%m-%d %H:%M} UTC"
    )

    try:
        while cursor_time <= end_time:
            async with rest_semaphore:
                batch = await exchange.fapiPublicGetAggTrades({
                    "symbol": raw_sym,
                    "startTime": cursor_time,
                    "endTime": end_time,
                    "limit": 1000,
                })
            request_count += 1
            if not batch:
                break

            last_trade_ts = cursor_time
            for trade in batch:
                trade_ts = int(trade.get("T") or trade.get("time") or 0)
                if trade_ts <= 0:
                    continue
                last_trade_ts = max(last_trade_ts, trade_ts)
                hour_ts = int(trade_ts / 3600000) * 3600000
                if hour_ts not in buckets:
                    buckets[hour_ts] = {"whale_buy": 0.0, "whale_sell": 0.0, "retail_vol": 0.0}

                price = float(trade.get("p") or trade.get("price") or 0.0)
                qty = float(trade.get("q") or trade.get("amount") or 0.0)
                notional = price * qty
                raw_buyer_maker = trade.get("m", trade.get("isBuyerMaker", False))
                is_buyer_maker = (
                    raw_buyer_maker
                    if isinstance(raw_buyer_maker, bool)
                    else str(raw_buyer_maker).lower() == "true"
                )

                if notional >= WHALE_NOTIONAL_THRESHOLD_USDT:
                    if is_buyer_maker:
                        buckets[hour_ts]["whale_sell"] += qty
                    else:
                        buckets[hour_ts]["whale_buy"] += qty
                if notional <= RETAIL_NOTIONAL_THRESHOLD_USDT:
                    buckets[hour_ts]["retail_vol"] += qty

            if len(batch) < 1000:
                break
            next_cursor = last_trade_ts + 1
            if next_cursor <= cursor_time:
                break
            cursor_time = next_cursor
            await asyncio.sleep(0.08)

        update_rows = [
            (
                buckets[ts]["whale_buy"],
                buckets[ts]["whale_sell"],
                buckets[ts]["retail_vol"],
                symbol,
                ts,
            )
            for ts in missing_hours
            if ts in buckets
        ]

        if not update_rows:
            print(f"   ⚠️ [近期订单流补洞] {symbol} REST 未取得可覆盖的 aggTrades，保留 NULL 让推理端熔断。")
            return

        def _write_updates():
            with sqlite3.connect(DB_PATH) as conn:
                conn.executemany(
                    "UPDATE memory_1h SET whale_buy=?, whale_sell=?, retail_vol=? WHERE symbol=? AND timestamp=?",
                    update_rows,
                )
                conn.commit()

        await asyncio.to_thread(_write_updates)
        print(
            f"   ✅ [近期订单流补洞] {symbol} 已用 aggTrades REST 补齐 {len(update_rows)}/{len(missing_hours)} 个小时桶 "
            f"(REST请求 {request_count} 次)。"
        )
    except Exception as e:
        print(f"   ❌ [近期订单流补洞] {symbol} aggTrades REST 回填失败: {type(e).__name__} -> {e}")

async def db_persist_loop():
    """ 自动存档守护进程：将 WebSocket 即时揉出的微观流，每分钟刷入硬盘 """
    print("🛡️ 启动本地数据库持久化守护进程 (60秒自动存档)...")
    while True:
        await asyncio.sleep(60)
        try:
            records = []
            for sym, data in market_data_store.items():
                if len(data["1h"]) == 0: continue
                k = data["1h"][-1]
                ts = k[0]
                stats = data.get("taker_stats_1h", {}).get(ts, {})

                records.append((
                    sym, ts,
                    k[1], k[2], k[3], k[4], k[5],
                    stats.get("quote_volume", 0.0), stats.get("count", 0),
                    stats.get("taker_buy_volume", 0.0), stats.get("taker_sell_volume", 0.0),
                    stats.get("whale_buy", None), stats.get("whale_sell", None), stats.get("retail_vol", None)
                ))
            if records:
                await asyncio.to_thread(upsert_memory_1h_records, records)

            for main_symbol in LIVE_MAIN_SYMBOLS:
                main_data = market_data_store.get(main_symbol, {})
                if main_data.get("1m"):
                    await asyncio.to_thread(upsert_memory_1m_records, main_symbol, list(main_data["1m"])[-5:])
                for hist_key in DERIVATIVE_METRIC_KEYS:
                    hist = main_data.get(hist_key, [])
                    if hist:
                        await asyncio.to_thread(upsert_derivative_history_records, main_symbol, hist_key, hist[-5:])
        except Exception:
            pass
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
    exchange_registry = {}

    try:
        print_gateway_data_chain()
        print("⏳ 正在按交易对拉起独立 Binance 连接，并进行市场合法性校验...")
            
        tasks = []
        tasks.append(db_persist_loop())
        tasks.append(health_monitor_loop(all_symbols))
        
        # 强制串行延迟，根治 Binance 5条/秒 订阅速率限制导致的并发静默丢包
        startup_delay = 0.0 
        
        for sym in all_symbols:
            spec = SUBSCRIPTION_SPECS[sym]
            # 简化解析逻辑：直接加斜杠
            market_symbol = spec["market_symbol"]
            if "/" not in market_symbol:
                quote = "USDT" if market_symbol.endswith("USDT") else "BTC"
                sym_standard = f"{market_symbol[:-len(quote)]}/{quote}"
            else:
                sym_standard = market_symbol

            # 自动分配引擎
            is_spot = spec["is_spot"]
            ex = build_exchange(is_spot)
            exchange_registry[sym] = ex
            
            # --- 核心修改：定义并传递引擎类型用于日志打印 ---
            engine_type = "现货" if is_spot else "合约"
            swap_sym = f"{sym_standard}:{sym_standard.split('/')[1]}" if not is_spot else sym_standard

            try:
                await ex.load_markets()
            except Exception as e:
                print(f"   🛑 [{sym}] 无法连接交易所获取市场列表: {e}")
                market_data_store[sym]["status"] = "unsupported"
                continue
            
            # 市场存在性强校验
            if swap_sym not in ex.markets:
                print(f"   ⚠️ [物理拦截] 交易所未上架 {swap_sym}，已自动跳过！")
                market_data_store[sym]["status"] = "unsupported"
                await ex.close()
                exchange_registry.pop(sym, None)
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
        if exchange_registry:
            await asyncio.gather(
                *(ex.close() for ex in exchange_registry.values()),
                return_exceptions=True
            )

# ==========================================
# 🔌 本地微服务接口：供 live_pipeline.py 秒级调用
# ==========================================
@app.get("/api/snapshot/{symbol}")
async def get_market_snapshot(symbol: str):
    """ 核心接口：返回该币种在内存中缓存的所有特征切片 """
    request_start = time.time()
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
    if clean_sym in LIVE_MAIN_SYMBOL_SET or clean_sym.replace("/", "") in LIVE_MAIN_SYMBOL_SET:
        # 🚨 修正：彻底剔除未闭合的 1m K 线，防止 RV 污染
        safe_1m = list(data.get("1m", []))
        current_minute_ts = int(time.time() / 60) * 60000
        if safe_1m and safe_1m[-1][0] >= current_minute_ts:
            safe_1m = safe_1m[:-1]

        response.update({
            "taker_stats_1h": data.get("taker_stats_1h", {}),
            "ohlcv_1m": safe_1m,
            "premium_history": data.get("premium_history", []),
            "funding_history": data.get("funding_history", []),
            "oi_history": data.get("oi_history", [])
        })

    api_response = JSONResponse(content=response)
    latency_ms = (time.time() - request_start) * 1000
    if latency_ms > 750:
        print(
            f"   ⚠️ [API 慢响应] /api/snapshot/{symbol} -> {clean_sym} "
            f"耗时 {latency_ms:.0f}ms | "
            f"1h={len(response.get('ohlcv_1h', []))} "
            f"1m={len(response.get('ohlcv_1m', []))} "
            f"taker={len(response.get('taker_stats_1h', {}))}"
        )
    return api_response

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    try:
        asyncio.run(start_gateways())
    except KeyboardInterrupt:
        print("🛑 网关已手动关闭")
