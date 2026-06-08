import os
import time
import sqlite3
import math
import pandas as pd
import requests
import json
from datetime import datetime
import ccxt
from state_manager import state
from execution_oms import (
    build_gate_close_stop_initial,
    cancel_obsolete_gate_close_stops,
    get_gate_stop_coverage_size,
    log_system_alert,
)
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR, MODEL_DIR, SYMBOL, AUX_SYMBOLS,
    HARD_TP, HARD_SL, HOLD_PERIOD,
    BAILOUT_ATR_MULTI, BAILOUT_VOL_MULTI, FORECAST_HORIZON,
    BAILOUT_MIN_PHYSICAL_DROP, BAILOUT_OBI_THRESH, EXPERIMENT_ID,
    MACRO_ALPHA_TOP_10, MACRO_ALPHA_TOP_5, MACRO_ALPHA_TOP_1,
    MACRO_LOGIT_DEAD_ZONE, MACRO_LOGIT_MID_SCORE, MACRO_LOGIT_TAIL_SCORE,
    MACRO_LOGIT_MID_DELTA, MACRO_LOGIT_TAIL_DELTA, MACRO_LOGIT_MAX_DELTA,
    MACRO_LOGIT_MID_POWER, MACRO_LOGIT_TAIL_POWER, MACRO_LOGIT_EXTREME_POWER,
    TRAILING_ATR_PERIOD, TRAILING_ATR_MULTI, LIVE_SYMBOL_CONFIGS,
    GATE_SANDBOX_MODE, RIGHT_BRAIN_SNAPSHOT_TIMEOUT,
    RIGHT_BRAIN_SNAPSHOT_WARN_COOLDOWN
)
from dotenv import load_dotenv
load_dotenv()

MY_GATE_API_KEY  = os.environ.get("GATE_DEMO_API_KEY", "")
MY_GATE_SECRET   = os.environ.get("GATE_DEMO_SECRET", "")

THRESHOLD_KEYS = ["up_top_1", "up_top_5", "up_top_10", "down_top_1", "down_top_5", "down_top_10"]
PHYSICAL_TRAIL_STEP_PHYS = 0.005


def _calc_bailout_adverse_move(prev_minute_close, current_open, current_close, pos_side):
    """Return adverse move against the position, anchored to real minute entry points."""
    prev_minute_close = float(prev_minute_close)
    current_open = float(current_open)
    current_close = float(current_close)
    pos_side = str(pos_side or "").upper()

    if pos_side == "LONG":
        reference = min(prev_minute_close, current_open)
        return max(0.0, reference - current_close), reference
    if pos_side == "SHORT":
        reference = max(prev_minute_close, current_open)
        return max(0.0, current_close - reference), reference
    return 0.0, current_close


def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _calc_physical_ratchet_phys(max_phys_pnl, hard_tp):
    """交易所物理保底棘轮：价格每多走 0.5%，保底同步抬高 0.5%，始终约保留 2% 空间。"""
    hard_tp = abs(float(hard_tp))
    if max_phys_pnl < hard_tp:
        return None
    step = PHYSICAL_TRAIL_STEP_PHYS
    gap = max(hard_tp - step, 0.0)
    raw_floor = max_phys_pnl - gap
    stepped_floor = math.floor((raw_floor + 1e-12) / step) * step
    return max(step, stepped_floor)


def _stop_price_from_phys(entry_px, pos_side, stop_phys):
    if stop_phys is None:
        return None
    return entry_px * (1 + stop_phys) if pos_side == 'LONG' else entry_px * (1 - stop_phys)


def _normalize_model_specs(cfg):
    raw_models = cfg.get("models")
    if not raw_models:
        raw_models = [{"experiment_id": cfg.get("experiment_id", EXPERIMENT_ID), "weight": cfg.get("weight", 1.0)}]

    normalized = []
    for raw_model in raw_models:
        model_cfg = raw_model or {}
        exp_id = str(model_cfg.get("experiment_id", cfg.get("experiment_id", EXPERIMENT_ID)))
        weight = float(model_cfg.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"模型权重必须为正数: experiment_id={exp_id}, weight={weight}")
        normalized.append({
            "name": str(model_cfg.get("name", exp_id)),
            "experiment_id": exp_id,
            "weight": weight,
        })

    total_weight = sum(m["weight"] for m in normalized)
    for model_cfg in normalized:
        model_cfg["normalized_weight"] = model_cfg["weight"] / total_weight
    return normalized


def _normalize_live_configs():
    raw_configs = LIVE_SYMBOL_CONFIGS if isinstance(LIVE_SYMBOL_CONFIGS, dict) and LIVE_SYMBOL_CONFIGS else {
        SYMBOL: {"experiment_id": EXPERIMENT_ID, "aux_symbols": AUX_SYMBOLS}
    }
    normalized = {}
    for raw_symbol, cfg in raw_configs.items():
        sym_key = _normalize_symbol_key(raw_symbol)
        cfg = cfg or {}
        normalized[sym_key] = {
            "experiment_id": str(cfg.get("experiment_id", EXPERIMENT_ID)),
            "models": _normalize_model_specs(cfg),
        }
    return normalized


LIVE_CONFIGS = _normalize_live_configs()


def _load_thresholds_for_exp(exp_id):
    json_path = MODEL_DIR / f"dynamic_thresholds_{exp_id}.json"
    with open(json_path, "r") as f:
        config_data = json.load(f)
    thresholds = config_data.get("thresholds", config_data)
    missing_keys = [key for key in THRESHOLD_KEYS if key not in thresholds]
    if missing_keys:
        raise ValueError(f"{json_path.name} 缺少阈值字段: {missing_keys}")
    return thresholds, json_path


def _weighted_thresholds_for_models(model_specs):
    weighted = {key: 0.0 for key in THRESHOLD_KEYS}
    for model_cfg in model_specs:
        model_thresholds, _ = _load_thresholds_for_exp(model_cfg["experiment_id"])
        weight = float(model_cfg["normalized_weight"])
        for key in THRESHOLD_KEYS:
            weighted[key] += float(model_thresholds[key]) * weight
    return weighted


def _load_thresholds_by_symbol():
    thresholds = {}
    for sym_key, cfg in LIVE_CONFIGS.items():
        try:
            model_specs = cfg.get("models") or [
                {"experiment_id": cfg.get("experiment_id", EXPERIMENT_ID), "normalized_weight": 1.0}
            ]
            thresholds[sym_key] = _weighted_thresholds_for_models(model_specs)
            model_text = ", ".join(
                f"{m['experiment_id']}@{float(m.get('normalized_weight', 1.0)):.2f}"
                for m in model_specs
            )
            print(f"✅ 右脑加载 {sym_key} 加权阈值: {model_text}")
        except Exception as e:
            print(f"❌ 右脑加载 {sym_key} 阈值文件失败: {e}")
            thresholds[sym_key] = {}
    return thresholds


def _status_for_symbol(live_status, base_sym):
    base_sym = _normalize_symbol_key(base_sym)
    symbol_status = (live_status.get("symbols", {}) or {}).get(base_sym)
    if symbol_status:
        return symbol_status
    if _normalize_symbol_key(live_status.get("symbol")) == base_sym:
        return live_status
    return {}


def _inspect_snapshot_stop_coverage(position):
    """基于供货线程快照检查当前仓位的止损覆盖情况。"""
    info = position.get("info", {}) or {}
    try:
        expected_contracts = abs(float(info.get("size", position.get("contracts", 0)) or 0.0))
    except (TypeError, ValueError):
        expected_contracts = 0.0

    snapshot_stops = list(state.portfolio_memory.get("open_stops", []) or [])
    sym_key = _normalize_symbol_key(position.get("symbol"))
    matched_stops = [o for o in snapshot_stops if _normalize_symbol_key(o.get("symbol")) == sym_key]

    if expected_contracts <= 0:
        return {
            "status": "no_position",
            "expected_contracts": 0.0,
            "max_stop_size": 0.0,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": 1.0,
            "reason": "",
        }

    if not matched_stops:
        return {
            "status": "missing_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": 0.0,
            "matched_stop_count": 0,
            "coverage_ratio": 0.0,
            "reason": "交易所快照中未发现该仓位的活动止损单",
        }

    parsed_sizes = []
    for stop_order in matched_stops:
        parsed_size = get_gate_stop_coverage_size(stop_order, expected_contracts)
        if parsed_size is not None:
            parsed_sizes.append(parsed_size)

    if not parsed_sizes:
        return {
            "status": "unparseable_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": 0.0,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": 0.0,
            "reason": "止损单存在，但无法解析其覆盖张数",
        }

    max_stop_size = max(parsed_sizes)
    coverage_ratio = max_stop_size / (expected_contracts + 1e-9)
    if coverage_ratio < 0.9:
        return {
            "status": "undercovered_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": max_stop_size,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": coverage_ratio,
            "reason": f"止损单覆盖不足，size={max_stop_size:.4f} < 持仓={expected_contracts:.4f}",
        }

    if len(parsed_sizes) > 1:
        return {
            "status": "duplicate_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": max_stop_size,
            "matched_stop_count": len(matched_stops),
            "coverage_ratio": coverage_ratio,
            "reason": f"检测到 {len(parsed_sizes)} 张可解析止损单，需收敛为单一统一防线",
        }

    return {
        "status": "ok",
        "expected_contracts": expected_contracts,
        "max_stop_size": max_stop_size,
        "matched_stop_count": len(matched_stops),
        "coverage_ratio": coverage_ratio,
        "reason": "",
    }


def _calc_recent_atr(ohlcv, period, current_close=None):
    if not ohlcv or len(ohlcv) < 2:
        return 0.0

    recent = ohlcv[-(period + 1):]
    tr_list = []
    for i in range(1, len(recent)):
        prev_close = float(recent[i - 1][4])
        curr_h = float(recent[i][2])
        curr_l = float(recent[i][3])
        if current_close is not None and i == len(recent) - 1:
            curr_h = max(curr_h, float(current_close))
            curr_l = min(curr_l, float(current_close))
        tr = max(curr_h - curr_l, abs(curr_h - prev_close), abs(curr_l - prev_close))
        tr_list.append(tr)

    return sum(tr_list) / len(tr_list) if tr_list else 0.0


def _safe_parse_epoch(raw_value):
    """兼容秒/毫秒时间戳，解析失败返回 None。"""
    try:
        ts = float(raw_value)
    except (TypeError, ValueError):
        return None

    if ts <= 0:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return ts


def _parse_local_exec_ts(raw_value):
    """解析 live_trades 中的本地执行时间。"""
    if not raw_value:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(str(raw_value), fmt).timestamp()
        except (TypeError, ValueError):
            continue
    try:
        return pd.to_datetime(raw_value, errors="coerce").timestamp()
    except Exception:
        return None


def _resolve_latest_same_side_trade_time(position, base_sym):
    """读取本地最近一笔同向开仓/补仓时间，作为持仓超时计时锚点。"""
    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
    pos_side = str(position.get("side", "")).upper()
    local_open_side = "buy" if pos_side == "LONG" else "sell"
    try:
        with sqlite3.connect(db_name, timeout=1.0) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT exec_ts
                FROM live_trades
                WHERE side = ?
                ORDER BY exec_ts DESC
                LIMIT 1
                """,
                (local_open_side,),
            )
            row = cur.fetchone()
            if row:
                return _parse_local_exec_ts(row[0])
    except Exception:
        pass
    return None


def _resolve_position_entry_time(position, base_sym):
    """
    以最近一笔同向开仓/补仓作为持仓超时计时锚点。
    交易所 open_time 往往只代表首次开仓；加仓后需要优先采用本地最新下单流水。
    """
    info = position.get("info", {}) or {}
    exchange_entry_ts = None
    candidates = [
        info.get("open_time"),
        position.get("timestamp"),
        position.get("lastUpdateTimestamp"),
        info.get("update_time"),
    ]
    for raw_ts in candidates:
        parsed_ts = _safe_parse_epoch(raw_ts)
        if parsed_ts is not None:
            exchange_entry_ts = parsed_ts
            break

    local_trade_ts = _resolve_latest_same_side_trade_time(position, base_sym)
    if local_trade_ts is not None and (
        exchange_entry_ts is None or local_trade_ts >= exchange_entry_ts - 1.0
    ):
        return local_trade_ts
    if exchange_entry_ts is not None:
        return exchange_entry_ts

    return time.time()


def _apply_logit_fusion(base_prob, macro_score, alpha):
    """复刻主引擎的非线性宏观概率融合逻辑，供右脑识别同向信号刷新持仓计时。"""
    epsilon = 1e-7
    p_safe = min(max(float(base_prob or 0.0), epsilon), 1.0 - epsilon)
    logit = math.log(p_safe / (1.0 - p_safe))

    score = min(max(float(macro_score or 0.0), -1.0), 1.0)
    abs_score = abs(score)
    sign = 1.0 if score >= 0 else -1.0
    tier_scale = min(max(float(alpha or 0.0) / max(abs(MACRO_ALPHA_TOP_10), 1e-9), 0.0), 1.0)

    dead_zone = min(max(MACRO_LOGIT_DEAD_ZONE, 0.0), 1.0)
    mid_score = min(max(MACRO_LOGIT_MID_SCORE, dead_zone + 1e-9), 1.0)
    tail_score = min(max(MACRO_LOGIT_TAIL_SCORE, mid_score + 1e-9), 1.0)

    if abs_score <= dead_zone:
        macro_delta = 0.0
    elif abs_score <= mid_score:
        x = (abs_score - dead_zone) / (mid_score - dead_zone)
        macro_delta = MACRO_LOGIT_MID_DELTA * (x ** MACRO_LOGIT_MID_POWER)
    elif abs_score <= tail_score:
        x = (abs_score - mid_score) / (tail_score - mid_score)
        macro_delta = MACRO_LOGIT_MID_DELTA + (
            MACRO_LOGIT_TAIL_DELTA - MACRO_LOGIT_MID_DELTA
        ) * (x ** MACRO_LOGIT_TAIL_POWER)
    else:
        x = (abs_score - tail_score) / (1.0 - tail_score + 1e-9)
        macro_delta = MACRO_LOGIT_TAIL_DELTA + (
            MACRO_LOGIT_MAX_DELTA - MACRO_LOGIT_TAIL_DELTA
        ) * (x ** MACRO_LOGIT_EXTREME_POWER)

    adjusted_logit = logit + (sign * macro_delta * tier_scale)
    return 1.0 / (1.0 + math.exp(-adjusted_logit))


def _resolve_target_signal_tier_from_status(live_status, thresholds, base_sym):
    """
    优先读取主引擎已经算出的 target_signal/target_tier；旧状态文件缺字段时再复算。
    这里返回 target_signal，而不是 actual_signal，因为同向仓位下被 OMS 放弃的信号
    仍然应该刷新 12 小时持仓计时。
    """
    if _normalize_symbol_key(live_status.get("symbol")) != base_sym:
        return 0, 0

    execution_state = live_status.get("execution_state", {}) or {}
    try:
        target_signal = int(execution_state.get("target_signal", 0) or 0)
    except (TypeError, ValueError):
        target_signal = 0
    try:
        target_tier = int(execution_state.get("target_tier", 0) or 0)
    except (TypeError, ValueError):
        target_tier = 0
    if target_signal in (-1, 1) and target_tier in (1, 2, 3):
        return target_signal, target_tier

    inference_data = live_status.get("model_inference", {}) or {}
    raw_p_up = float(inference_data.get("raw_prob_up", 0.0) or 0.0)
    raw_p_down = float(inference_data.get("raw_prob_down", 0.0) or 0.0)
    macro_score = float(
        inference_data.get("macro_score", live_status.get("macro_score", 0.0)) or 0.0
    )

    adj_up_top1 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_1)
    adj_up_top5 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_5)
    adj_up_top10 = _apply_logit_fusion(raw_p_up, macro_score, MACRO_ALPHA_TOP_10)
    adj_down_top1 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_1)
    adj_down_top5 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_5)
    adj_down_top10 = _apply_logit_fusion(raw_p_down, -macro_score, MACRO_ALPHA_TOP_10)

    if adj_up_top1 > thresholds.get('up_top_1', 1.0):
        return 1, 3
    if adj_up_top5 > thresholds.get('up_top_5', 1.0):
        return 1, 2
    if adj_up_top10 > thresholds.get('up_top_10', 1.0):
        return 1, 1
    if adj_down_top1 > thresholds.get('down_top_1', 1.0):
        return -1, 3
    if adj_down_top5 > thresholds.get('down_top_5', 1.0):
        return -1, 2
    if adj_down_top10 > thresholds.get('down_top_10', 1.0):
        return -1, 1
    return 0, 0


def _resolve_current_position_tier_from_status(live_status, current_side):
    position_tier_state = live_status.get("position_tier_state", {}) or {}
    try:
        state_side = int(position_tier_state.get("side", 0) or 0)
    except (TypeError, ValueError):
        state_side = 0
    try:
        current_tier = int(position_tier_state.get("highest_tier", 0) or 0)
    except (TypeError, ValueError):
        current_tier = 0

    if state_side != current_side:
        return 0
    return current_tier if current_tier in (1, 2, 3) else 0


def _refresh_entry_time_on_same_direction_signal(mem, base_sym, pos_side, live_status, thresholds):
    """
    若最新信号评估出现与当前仓位同向且同 Tier 的 target_signal，即使 OMS 因已有仓位未再次下单，
    也把 entry_time 锚点刷新到信号触发时刻，避免 HOLD_PERIOD 仍从最初建仓时刻累计。
    低 Tier 信号代表概率走弱，不能给高 Tier 持仓续命。
    """
    execution_state = live_status.get("execution_state", {}) or {}
    signal_refresh_ts = _safe_parse_epoch(execution_state.get("last_signal_evaluated_ts"))
    if signal_refresh_ts is None:
        signal_refresh_ts = _safe_parse_epoch(execution_state.get("last_inferred_hour"))
    if signal_refresh_ts is None:
        return

    target_signal, target_tier = _resolve_target_signal_tier_from_status(live_status, thresholds, base_sym)
    current_side = 1 if str(pos_side).upper() == "LONG" else -1
    if target_signal != current_side:
        return

    current_tier = _resolve_current_position_tier_from_status(live_status, current_side)
    if target_tier != current_tier:
        return

    last_refresh_ts = float(mem.get('last_same_signal_refresh_ts', 0.0) or 0.0)
    if signal_refresh_ts <= last_refresh_ts:
        return
    if signal_refresh_ts <= float(mem.get('entry_time', 0.0) or 0.0):
        return

    mem['entry_time'] = signal_refresh_ts
    mem['last_same_signal_refresh_ts'] = signal_refresh_ts
    print(
        f"🕒 [{base_sym}] 侦测到同向同Tier-{current_tier}信号延续，持仓计时已刷新至 "
        f"{datetime.fromtimestamp(signal_refresh_ts).strftime('%Y-%m-%d %H:%M:%S')}。"
    )


def _read_last_inferred_hour(base_sym=None):
    """读取主引擎最近一次使用的闭合 1h K 线时间戳，供强制退出后做本K线冷却。"""
    status_path = os.path.join(LIVE_DATA_DIR, "live_status.json")
    if not os.path.exists(status_path):
        return 0

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if base_sym is not None:
            symbol_status = _status_for_symbol(payload, base_sym)
            return int(symbol_status.get("execution_state", {}).get("last_inferred_hour", 0) or 0)
        return int(payload.get("execution_state", {}).get("last_inferred_hour", 0) or 0)
    except Exception:
        return 0


def _register_forced_exit_lock(base_sym, reason, anchor_closed_ts=None):
    """
    强制退出后锁住当前这根已闭合 1h K 线。
    只有等到下一根新 K 线闭合，主引擎才允许再次入场。
    """
    if anchor_closed_ts is None:
        anchor_closed_ts = _read_last_inferred_hour(base_sym)
    try:
        anchor_closed_ts = int(anchor_closed_ts or 0)
    except (TypeError, ValueError):
        anchor_closed_ts = 0

    if anchor_closed_ts <= 0:
        current_closed_hour = max(int(time.time() // 3600) - 1, 0)
        anchor_closed_ts = current_closed_hour * 3600000

    with state.event_lock:
        state.forced_exit_locks[base_sym] = {
            "anchor_closed_ts": anchor_closed_ts,
            "trigger_ts": time.time(),
            "reason": str(reason),
        }


def execute_bailout(ex, base_sym, p, reason, current_price, anchor_closed_ts=None):
    """右脑专用：执行紧急市价逃生"""
    gate_sym = p['symbol']
    pos_side = str(p.get('side', '')).upper()
    contracts = float(p.get('contracts', 0))
    entry_px = float(p.get('entryPrice', p['info'].get('entry_price', p['info'].get('avgPx', 0))))

    print(f"\n🚨 [右脑介入] 侦测到 {base_sym} {reason}！执行市价核弹清仓 (数量: {contracts})...")
    try:
        # 1. 越权下发市价单
        cl_side = 'sell' if pos_side == 'LONG' else 'buy'
        try:
            ex.create_order(symbol=gate_sym, type='market', side=cl_side, amount=contracts, params={'reduceOnly': True})
            print(f"   ✅ 交易所市价平仓指令已下发 ({gate_sym})")
        except Exception as e:
            print(f"   ⚠️ 交易所下单接口报错: {e}")
            # 即便下单报错也尝试继续执行落库，可能仓位已经由于 OCO 先一步平掉了

        # 2. 扫清残余 OCO
        try:
            for o in ex.fetch_open_orders(gate_sym, params={'stop': True}):
                try: ex.cancel_order(o['id'], gate_sym, params={'stop': True})
                except: pass
        except: pass

        # 3. 落库：历史本纪 (增加写入保障)
        db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{base_sym}.db")
        with sqlite3.connect(db_name, timeout=30.0, check_same_thread=False) as conn_log:
            cursor = conn_log.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                              (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                               amount REAL, fill_price REAL, slippage REAL,
                               tp_price REAL, sl_price REAL, margin_used REAL, leverage REAL)''')

            # 记录一笔平仓流水
            cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'bailout_panic_sell', 'bailout_close',
                            float(contracts), current_price, 0.0, 0.0, 0.0, 0.0, 0.0))
            conn_log.commit()
            print(f"   💾 平仓记录已同步至本地数据库: {db_name}")

        _register_forced_exit_lock(base_sym, reason, anchor_closed_ts=anchor_closed_ts)

        pos_lev = float(p.get('leverage', p['info'].get('leverage', p['info'].get('lever', 1))))
        if pos_lev <= 0: pos_lev = 10.0

        raw_pnl = (current_price - entry_px) / entry_px if pos_side == 'LONG' else (entry_px - current_price) / entry_px
        realized_roe = raw_pnl * pos_lev
        contract_size = float(p.get('contractSize', 1))
        abs_profit = raw_pnl * (contracts * contract_size * entry_px)

        dir_str = "LONG" if pos_side == "LONG" else "SHORT" if pos_side == "SHORT" else pos_side or "UNKNOWN"
        log_system_alert(
            "🏁 主动出场",
            base_sym,
            (
                f"{dir_str} 市价平仓已下发 | 原因: {reason} | "
                f"估算平仓价: ${current_price:.4f} | 数量: {contracts:g} 张 | "
                f"估算盈亏: {abs_profit:+.2f} USDT | ROE {realized_roe:+.2%}"
            ),
        )

        PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")
        push_title = f"🧯 [Gate实盘] {base_sym} 主动风控平仓"
        dir_str = "做多 🟢" if pos_side == 'LONG' else "做空 🔴"
        if abs_profit > 0:
            pnl_label = "🟢 估算盈利"
        elif abs_profit < 0:
            pnl_label = "🔴 估算亏损"
        else:
            pnl_label = "🟡 近似保本"

        push_content = (
            f"<b>🔹 交易标的：</b>{base_sym} ({dir_str})<br>"
            f"<b>🔸 出场原因：</b>{reason}<br>"
            f"<b>⚙️ 执行动作：</b>右脑风控市价减仓/平仓<br>"
            f"<b>🎯 估算平仓价：</b>${current_price:.4f}<br>"
            f"<b>{pnl_label}：</b>{abs_profit:+.2f} USDT ({realized_roe:+.2%})<br>"
            f"<b>⏰ 时间：</b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        state.push_queue.put({"token": PUSHPLUS_TOKEN, "title": push_title, "content": push_content, "template": "html"})
        state.recent_pushes[base_sym] = time.time()

    except Exception as e:
        print(f"   ⚠️ 右脑熔断清仓失败: {e}")
        log_system_alert("❌ 熔断异常", base_sym, f"跳伞执行失败: {e}")


def _gate_stop_text(stop_order):
    initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
    return str(stop_order.get("clientOrderId", "") or initial.get("text", "") or "")


def _gate_stop_trigger_price(stop_order):
    info = stop_order.get("info", {}) or {}
    trigger = info.get("trigger", {}) or {}
    raw_price = stop_order.get("stopPrice") or trigger.get("price", 0.0)
    try:
        return float(raw_price or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fetch_live_same_side_position(ex, gate_symbol, pos_side):
    """直接查询交易所仓位，区分快照延迟导致的假缺口与真实裸仓。"""
    try:
        positions = ex.fetch_positions([gate_symbol])
    except Exception as e:
        print(f"   ⚠️ 右脑实时仓位复核失败，暂按旧快照继续保护: {e}")
        return "unknown", None

    gate_key = _normalize_symbol_key(gate_symbol)
    for position in positions:
        info = position.get("info", {}) or {}
        contracts = abs(float(info.get("size", position.get("contracts", 0)) or 0.0))
        if contracts <= 1e-8:
            continue
        if _normalize_symbol_key(position.get("symbol")) != gate_key:
            continue
        if str(position.get("side", "")).upper() == str(pos_side).upper():
            return "active", position
    return "flat", None


def _is_stop_line_crossed(pos_side, stop_price, current_price):
    """判断当前价格是否已经越过追踪止损线，此时不能再补挂落后于市场的条件单。"""
    try:
        stop_price = float(stop_price)
        current_price = float(current_price)
    except (TypeError, ValueError):
        return False
    if stop_price <= 0 or current_price <= 0:
        return False
    if str(pos_side).upper() == "LONG":
        return current_price <= stop_price
    if str(pos_side).upper() == "SHORT":
        return current_price >= stop_price
    return False


def _converge_duplicate_exchange_stops(ex, gate_symbol, pos_side, target_sl_px):
    """直接收敛重复止损，不再通过额外新增条件单制造更多短暂重复。"""
    stops = ex.fetch_open_orders(gate_symbol, params={'stop': True})
    gate_key = _normalize_symbol_key(gate_symbol)
    expected_auto_size = "close_long" if str(pos_side).upper() == "LONG" else "close_short"
    candidates = []
    for stop_order in stops:
        if _normalize_symbol_key(stop_order.get("symbol")) != gate_key:
            continue
        text = _gate_stop_text(stop_order).lower()
        if "sl" not in text and "tsl" not in text:
            continue
        initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
        auto_size = str(initial.get("auto_size", "") or "").lower()
        if auto_size and auto_size != expected_auto_size:
            continue
        candidates.append(stop_order)

    if len(candidates) <= 1:
        return 0

    def keep_rank(stop_order):
        stop_px = _gate_stop_trigger_price(stop_order)
        distance = abs(stop_px - target_sl_px) if stop_px > 0 else float("inf")
        return distance, _gate_stop_text(stop_order)

    keep_order = min(candidates, key=keep_rank)
    return cancel_obsolete_gate_close_stops(
        ex,
        gate_symbol,
        _gate_stop_text(keep_order),
        pos_side,
        stops,
    )


def update_exchange_sl(ex, gate_symbol, pos_side, new_sl_price, pos_size):
    """向交易所发送 API：先挂载新追踪止损，成功后再撤销旧追踪，杜绝裸奔窗口。"""
    exact_size = int(round(float(pos_size)))
    if exact_size <= 0:
        return False

    try:
        # 1. Gate.io 官方 API 规则对齐：多单止损需跌破触发(rule 2)
        rule = 2 if pos_side == 'LONG' else 1
        if not getattr(ex, "markets", None) or gate_symbol not in ex.markets:
            ex.load_markets(True)
        native_symbol = ex.market_id(gate_symbol)

        # 🚨 修复 3：生成带有精确时间戳的唯一标识符
        unique_sl_text = f"tsl_{int(time.time()*1000)}"

        # 2. 先建立新防线
        ex.private_futures_post_settle_price_orders({
            "settle": "usdt",
            "initial": build_gate_close_stop_initial(native_symbol, unique_sl_text, pos_side),
            "trigger": {
                "strategy_type": 0,
                "price_type": 1,
                "price": str(ex.price_to_precision(gate_symbol, new_sl_price)),
                "rule": rule
            }
        })

        # 3. 新单挂载成功后，确认 Gate 已经能查到新单，再撤除同方向旧防线。
        time.sleep(0.5)
        stops = ex.fetch_open_orders(gate_symbol, params={'stop': True})
        if not any(_gate_stop_text(o).lower() == unique_sl_text for o in stops):
            print("   ⚠️ 新追踪止损已提交但尚未出现在查询结果中，保留旧防线等待下一轮收敛。")
            return False
        cancel_obsolete_gate_close_stops(
            ex,
            gate_symbol,
            unique_sl_text,
            pos_side,
            stops,
        )

        return True
    except Exception as e:
        print(f"   ⚠️ 交易所追踪止损单刷新请求失败 (保留原防线兜底): {e}")
        return False

def bailout_worker():
    """右脑主循环：双轨制 (物理高水位状态机 + ATR 主动熔断)"""
    print("🛡️ [右脑] 高频双轨制风控已启动 (追踪止盈 OCO + ATR 主动熔断)...")
    bailout_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    bailout_ex.set_sandbox_mode(GATE_SANDBOX_MODE)
    try:
        bailout_ex.load_markets(True)
        print("🔄 [右脑] Gate 市场静态规则已预加载。")
    except Exception as e:
        print(f"⚠️ [右脑] 启动预加载市场规则失败，将在补挂时重试: {e}")

    # 🧠 物理水位状态机内存池
    tracking_mem = {}
    bailout_cooldown_locks = {}
    last_worker_error_ts = 0.0

    thresholds_by_symbol = _load_thresholds_by_symbol()

    while True:
        try:
            with state.pos_lock:
                current_positions = list(state.active_pos_map.values())

            if not current_positions:
                tracking_mem.clear()
                # 清理 UI 显示文件
                if os.path.exists(os.path.join(LIVE_DATA_DIR, "chandelier_state.json")):
                    try: os.remove(os.path.join(LIVE_DATA_DIR, "chandelier_state.json"))
                    except: pass
                time.sleep(1)
                continue

            # 读取最新 AI 概率与前端硬参数
            live_status = {}
            if os.path.exists(os.path.join(LIVE_DATA_DIR, "live_status.json")):
                try:
                    with open(os.path.join(LIVE_DATA_DIR, "live_status.json"), "r") as f:
                        live_status = json.load(f)
                except: pass

            hard_tp = HARD_TP   # K线物理止盈幅度（直接来自 config）
            hard_sl = HARD_SL   # K线物理底部止损幅度（直接来自 config）

            export_ui_mem = {}

            import concurrent.futures
            def fetch_snap(p):
                base_sym = p['symbol'].split(':')[0].replace('/', '')
                try:
                    resp = requests.get(
                        f"http://127.0.0.1:8000/api/snapshot/{base_sym}",
                        timeout=RIGHT_BRAIN_SNAPSHOT_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        return p, base_sym, resp.json(), ""
                    return p, base_sym, None, f"HTTP {resp.status_code}: {resp.text[:160]}"
                except Exception as e:
                    return p, base_sym, None, f"{type(e).__name__}: {e}"

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(5, len(current_positions))) as executor:
                results = list(executor.map(fetch_snap, current_positions))

            for p, base_sym, data, snapshot_error in results:
                try:
                    symbol_status = _status_for_symbol(live_status, base_sym)
                    inference_data = symbol_status.get("model_inference", {}) or {}
                    raw_p_up = inference_data.get("raw_prob_up", 0.0)
                    raw_p_down = inference_data.get("raw_prob_down", 0.0)
                    th = thresholds_by_symbol.get(base_sym, {})

                    gate_sym = p['symbol']
                    pos_side = str(p.get('side', '')).upper()
                    info = p.get('info', {}) or {}
                    entry_px = float(p.get('entryPrice', info.get('entry_price', info.get('avgPx', 0))) or 0.0)
                    if entry_px <= 0:
                        continue
                    position_contracts = abs(float(info.get('size', p.get('contracts', 0)) or 0.0))

                    mark_px = float(p.get('markPrice', info.get('mark_price', entry_px)) or entry_px)
                    current_close = float(data.get('current_price', mark_px) if data else mark_px)
                    current_close = current_close if current_close > 0 else entry_px
                    trailing_atr_price = (
                        _calc_recent_atr(data.get('ohlcv_1m'), TRAILING_ATR_PERIOD, current_close)
                        if data and data.get('ohlcv_1m')
                        else 0.0
                    )

                    # 冷却检测 (防重复触发)
                    last_fired_time = bailout_cooldown_locks.get(base_sym, 0)
                    if time.time() - last_fired_time < 5.0:
                        continue

                    # 初始化该币种的内存追踪器
                    if base_sym not in tracking_mem:
                        entry_time = _resolve_position_entry_time(p, base_sym)
                        tracking_mem[base_sym] = {
                            'max_phys_pnl': 0.0,
                            'physical_sl_phys': -abs(hard_sl),
                            'atr_sl_phys': None,
                            'trailing_active': False,
                            'entry_time': entry_time,
                            'last_same_signal_refresh_ts': 0.0,
                            'last_sl_update_px': 0.0,
                            'last_entry_px': entry_px,
                            'last_contracts': position_contracts,
                            'last_stop_repair_ts': 0.0,
                            'last_snapshot_warn_ts': 0.0,
                        }

                    mem = tracking_mem[base_sym]
                    if 'physical_sl_phys' not in mem:
                        mem['physical_sl_phys'] = float(mem.get('dynamic_sl_phys', -abs(hard_sl)) or -abs(hard_sl))
                    if 'atr_sl_phys' not in mem:
                        mem['atr_sl_phys'] = None
                    # ---------------------------------------------------------
                    # 🚨 补丁 2/2：侦测阶梯加仓造成的均价偏移，重置水位防线防误杀
                    # ---------------------------------------------------------
                    if 'last_entry_px' not in mem:
                        mem['last_entry_px'] = entry_px
                    if 'last_contracts' not in mem:
                        mem['last_contracts'] = position_contracts

                    entry_px_shifted = abs(entry_px - mem['last_entry_px']) / max(mem['last_entry_px'], 1e-9) > 0.0001
                    contracts_increased = position_contracts > float(mem.get('last_contracts', 0.0) or 0.0) * 1.0001
                    add_position_detected = contracts_increased or (
                        entry_px_shifted
                        and position_contracts >= float(mem.get('last_contracts', 0.0) or 0.0) * 0.9999
                    )

                    # 如果交易所最新返回的均价/张数发生同向增加，说明触发了实盘加仓。
                    if add_position_detected:
                        # 记住当时的物理绝对止损价格
                        if pos_side == 'LONG':
                            abs_sl = mem['last_entry_px'] * (1 + mem['physical_sl_phys'])
                            mem['physical_sl_phys'] = max((abs_sl - entry_px) / entry_px, -abs(hard_sl))
                            if mem.get('atr_sl_phys') is not None:
                                abs_atr_sl = mem['last_entry_px'] * (1 + mem['atr_sl_phys'])
                                mem['atr_sl_phys'] = (abs_atr_sl - entry_px) / entry_px
                            mem['max_phys_pnl'] = (current_close - entry_px) / entry_px
                        else:
                            abs_sl = mem['last_entry_px'] * (1 - mem['physical_sl_phys'])
                            mem['physical_sl_phys'] = max((entry_px - abs_sl) / entry_px, -abs(hard_sl))
                            if mem.get('atr_sl_phys') is not None:
                                abs_atr_sl = mem['last_entry_px'] * (1 - mem['atr_sl_phys'])
                                mem['atr_sl_phys'] = (entry_px - abs_atr_sl) / entry_px
                            mem['max_phys_pnl'] = (entry_px - current_close) / entry_px

                        mem['last_entry_px'] = entry_px
                        mem['last_contracts'] = position_contracts
                        latest_trade_ts = _resolve_latest_same_side_trade_time(p, base_sym)
                        mem['entry_time'] = latest_trade_ts or time.time()
                        mem['last_same_signal_refresh_ts'] = max(
                            float(mem.get('last_same_signal_refresh_ts', 0.0) or 0.0),
                            mem['entry_time'],
                        )
                        print(
                            f"⚖️ [{base_sym}] 右脑侦测到阶梯加仓行为！新均价 ${entry_px:.4f}，"
                            f"持仓计时已重置至 {datetime.fromtimestamp(mem['entry_time']).strftime('%Y-%m-%d %H:%M:%S')}，"
                            "并已平移止损防线防止误杀。"
                        )
                    else:
                        mem['last_contracts'] = position_contracts

                    _refresh_entry_time_on_same_direction_signal(
                        mem,
                        base_sym,
                        pos_side,
                        symbol_status,
                        th,
                    )
                    # ---------------------------------------------------------
                    # 📈 轨道 1：双轨追踪状态机
                    # ---------------------------------------------------------
                    current_phys_pnl = (current_close - entry_px) / entry_px if pos_side == 'LONG' else (entry_px - current_close) / entry_px
                    mem['max_phys_pnl'] = max(mem['max_phys_pnl'], current_phys_pnl)
                    hold_hours = (time.time() - mem['entry_time']) / 3600.0

                    # 🚨 修复 2：为了和回测 (backtest_itransformer.py) 保持 100% 物理一致，
                    # 必须使用模型【最原始的概率】来判断反转斩仓，绝不能用宏观融合后的变异概率。
                    is_reversal = False
                    if pos_side == 'LONG' and raw_p_down > th.get('down_top_1', 1.0): is_reversal = True
                    if pos_side == 'SHORT' and raw_p_up > th.get('up_top_1', 1.0): is_reversal = True

                    need_market_close = False
                    close_reason = ""

                    # 触发器 1 & 3：反转 或 超过配置中的持仓时长上限
                    if is_reversal or hold_hours >= HOLD_PERIOD:
                        if current_phys_pnl < 0.005 and not mem['trailing_active']:
                            need_market_close = True
                            close_reason = "AI反转" if is_reversal else f"{HOLD_PERIOD}小时持仓超时"
                            close_reason += " (利润不足0.5%立即斩仓)"
                        else:
                            mem['trailing_active'] = True
                            mem['physical_sl_phys'] = max(mem['physical_sl_phys'], PHYSICAL_TRAIL_STEP_PHYS)

                    # 触发器 2：触及止盈线
                    if mem['max_phys_pnl'] >= hard_tp:
                        mem['trailing_active'] = True
                        ratchet_floor = _calc_physical_ratchet_phys(mem['max_phys_pnl'], hard_tp)
                        if ratchet_floor is not None:
                            mem['physical_sl_phys'] = max(mem['physical_sl_phys'], ratchet_floor)

                    # ATR 是内存状态机，不刷新交易所物理单；它可随 ATR 扩张/收缩上下移动。
                    trailing_retrace_phys = 0.0
                    if mem['trailing_active'] and trailing_atr_price > 0:
                        trailing_retrace_phys = TRAILING_ATR_MULTI * trailing_atr_price / entry_px
                        mem['atr_sl_phys'] = mem['max_phys_pnl'] - trailing_retrace_phys

                    if need_market_close:
                        execute_bailout(bailout_ex, base_sym, p, close_reason, current_close)
                        bailout_cooldown_locks[base_sym] = time.time()
                        del tracking_mem[base_sym]
                        continue

                    physical_sl_px = _stop_price_from_phys(entry_px, pos_side, mem['physical_sl_phys'])
                    atr_target_sl_px = (
                        _stop_price_from_phys(entry_px, pos_side, mem.get('atr_sl_phys'))
                        if mem['trailing_active'] and mem.get('atr_sl_phys') is not None
                        else None
                    )
                    export_ui_mem[base_sym] = {
                        'current_stop': atr_target_sl_px,
                        'exchange_stop': physical_sl_px,
                        'trailing_active': bool(mem['trailing_active']),
                        'profit_floor': physical_sl_px if mem['trailing_active'] else None,
                        'atr_retrace_pct': trailing_retrace_phys,
                        'max_phys_pnl': mem['max_phys_pnl'],
                        'physical_sl_phys': mem['physical_sl_phys'],
                        'atr_sl_phys': mem.get('atr_sl_phys'),
                    }

                    if atr_target_sl_px is not None and _is_stop_line_crossed(pos_side, atr_target_sl_px, current_close):
                        live_position_status, live_position = _fetch_live_same_side_position(
                            bailout_ex,
                            gate_sym,
                            pos_side,
                        )
                        if live_position_status == "active":
                            execute_bailout(
                                bailout_ex,
                                base_sym,
                                live_position,
                                "ATR内存追踪防线已触发",
                                current_close,
                            )
                            bailout_cooldown_locks[base_sym] = time.time()
                            del tracking_mem[base_sym]
                        elif live_position_status == "flat":
                            print(f"ℹ️ [{base_sym}] ATR 内存追踪防线已触发，交易所仓位已消失，等待供货线程确认平仓。")
                        else:
                            print(f"⚠️ [{base_sym}] ATR 内存追踪防线已触发，但实时仓位复核失败，保留交易所现有 OCO 并等待下一轮。")
                        continue

                    if data is None:
                        since_last_warn = time.time() - float(mem.get('last_snapshot_warn_ts', 0.0) or 0.0)
                        if since_last_warn >= RIGHT_BRAIN_SNAPSHOT_WARN_COOLDOWN:
                            reason_text = snapshot_error or "未知原因"
                            msg = (
                                f"[{base_sym}] 右脑未取到本地行情快照，已退化使用交易所 mark_price=${mark_px:.4f} "
                                f"维持止损托管。原因: {reason_text}"
                            )
                            print(f"⚠️ {msg}")
                            log_system_alert("⚠️ 右脑快照缺失", base_sym, msg)
                            mem['last_snapshot_warn_ts'] = time.time()

                    # ---------------------------------------------------------
                    # 🩹 轨道 1.5：若交易所止损单丢失/覆盖不足，立即按当前防线补挂
                    # ---------------------------------------------------------
                    stop_coverage = _inspect_snapshot_stop_coverage(p)
                    stop_repair_attempted = False
                    if stop_coverage["status"] == "duplicate_stop":
                        repair_cooldown = 8.0
                        since_last_repair = time.time() - float(mem.get('last_stop_repair_ts', 0.0) or 0.0)
                        if since_last_repair >= repair_cooldown:
                            mem['last_stop_repair_ts'] = time.time()
                            stop_repair_attempted = True
                            try:
                                cancelled_count = _converge_duplicate_exchange_stops(
                                    bailout_ex,
                                    gate_sym,
                                    pos_side,
                                    physical_sl_px,
                                )
                                if cancelled_count:
                                    repair_msg = f"[{base_sym}] 检测到重复止损，已撤销 {cancelled_count} 张旧单并保留最接近当前物理保底的条件单。"
                                    print(f"🩹 {repair_msg}")
                                    log_system_alert("🩹 收敛重复止损", base_sym, repair_msg)
                            except Exception as e:
                                print(f"⚠️ [{base_sym}] 重复止损收敛失败，保留现有防线等待下一轮: {e}")

                    repairable_statuses = {"missing_stop", "undercovered_stop", "unparseable_stop"}
                    if stop_coverage["status"] in repairable_statuses:
                        repair_cooldown = 8.0
                        since_last_repair = time.time() - float(mem.get('last_stop_repair_ts', 0.0) or 0.0)
                        if since_last_repair >= repair_cooldown:
                            mem['last_stop_repair_ts'] = time.time()
                            stop_repair_attempted = True
                            live_position_status, live_position = _fetch_live_same_side_position(
                                bailout_ex,
                                gate_sym,
                                pos_side,
                            )
                            if live_position_status == "flat":
                                print(f"ℹ️ [{base_sym}] 止损缺口出现时交易所已无活动仓位，跳过补挂并等待仓位快照确认。")
                                continue

                            live_info = (live_position or {}).get("info", {}) or {}
                            pos_size = abs(float(live_info.get("size", (live_position or {}).get("contracts", position_contracts)) or position_contracts))
                            repaired = update_exchange_sl(bailout_ex, gate_sym, pos_side, physical_sl_px, pos_size)
                            if repaired:
                                mem['last_sl_update_px'] = physical_sl_px
                                repair_msg = (
                                    f"[{base_sym}] 检测到止损缺口，已按当前物理保底补挂止损 | "
                                    f"status={stop_coverage['status']} | stop=${physical_sl_px:.4f}"
                                )
                                print(f"🩹 {repair_msg}")
                                log_system_alert("🩹 补挂止损", base_sym, repair_msg)
                            else:
                                fail_msg = (
                                    f"[{base_sym}] 检测到止损缺口，但补挂失败 | "
                                    f"status={stop_coverage['status']} | reason={stop_coverage['reason']}"
                                )
                                print(f"⚠️ {fail_msg}")
                                log_system_alert("⚠️ 止损补挂失败", base_sym, fail_msg)

                    if mem['trailing_active'] and not stop_repair_attempted:
                        diff_pct = abs(physical_sl_px - mem['last_sl_update_px']) / (entry_px + 1e-9)
                        if diff_pct > 0.001:
                            pos_size = abs(float(info.get('size', p.get('contracts', 0)) or 0.0))
                            success = update_exchange_sl(bailout_ex, gate_sym, pos_side, physical_sl_px, pos_size)
                            if success:
                                previous_sl_px = mem['last_sl_update_px']
                                mem['last_sl_update_px'] = physical_sl_px
                                is_tightening = (
                                    physical_sl_px > previous_sl_px
                                    if pos_side == 'LONG'
                                    else previous_sl_px <= 0 or physical_sl_px < previous_sl_px
                                )
                                action = "收紧" if is_tightening else "初始化"
                                ratchet_msg = (
                                    f"{pos_side} 物理保底止损已{action}至 ${physical_sl_px:.4f} | "
                                    f"保底: {mem['physical_sl_phys']*100:+.2f}% | "
                                    f"ATR内存线: {(mem.get('atr_sl_phys') or 0.0)*100:+.2f}% | "
                                    f"最高浮盈: {mem['max_phys_pnl']*100:+.2f}%"
                                )
                                print(f"📍 [{base_sym}] {ratchet_msg}")
                                log_system_alert("📍 棘轮调整", base_sym, ratchet_msg)

                    # ---------------------------------------------------------
                    # 🆘 轨道 2：高频瀑布插针熔断 (主动雷达逃生)
                    # ---------------------------------------------------------
                    if data and 'ohlcv_1m' in data and len(data['ohlcv_1m']) >= 14:
                        ohlcv_1m = data['ohlcv_1m']
                        prev_minute_close = float(ohlcv_1m[-2][4])
                        current_open = float(ohlcv_1m[-1][1])
                        current_vol = float(ohlcv_1m[-1][5])

                        past_vols = [x[5] for x in ohlcv_1m[:-1]]
                        avg_vol = sum(past_vols) / len(past_vols) if past_vols else 1e-8

                        recent_15 = ohlcv_1m[-15:]
                        tr_list = []
                        for i in range(1, len(recent_15)):
                            prev_close = float(recent_15[i-1][4])
                            curr_h = float(recent_15[i][2])
                            curr_l = float(recent_15[i][3])
                            tr = max(curr_h - curr_l, abs(curr_h - prev_close), abs(curr_l - prev_close))
                            tr_list.append(tr)

                        atr_1m = sum(tr_list) / len(tr_list) if tr_list else 0

                        if atr_1m > 0:
                            dyn_atr_multi = BAILOUT_ATR_MULTI
                            dyn_vol_multi = BAILOUT_VOL_MULTI

                            is_flash_crash = False
                            is_flash_surge = False
                            adverse_move, adverse_reference = _calc_bailout_adverse_move(
                                prev_minute_close,
                                current_open,
                                current_close,
                                pos_side,
                            )
                            adverse_move_pct = adverse_move / (adverse_reference + 1e-8)
                            min_physical_space = adverse_reference * BAILOUT_MIN_PHYSICAL_DROP
                            price_break_threshold = max((dyn_atr_multi * atr_1m), min_physical_space)

                            if pos_side == 'LONG':
                                if adverse_move > price_break_threshold:
                                    is_flash_crash = True
                            elif pos_side == 'SHORT':
                                if adverse_move > price_break_threshold:
                                    is_flash_surge = True

                            if (is_flash_crash or is_flash_surge) and ('orderbook' in data):
                                actual_vol_multi = current_vol / (avg_vol + 1e-8)
                                if actual_vol_multi >= dyn_vol_multi:
                                    ob = data['orderbook']
                                    bid_vol = sum([float(o[0]) * float(o[1]) for o in ob['bids'][:20]])
                                    ask_vol = sum([float(o[0]) * float(o[1]) for o in ob['asks'][:20]])

                                    obi_thresh = BAILOUT_OBI_THRESH

                                    if pos_side == 'LONG' and is_flash_crash:
                                        obi = bid_vol / (ask_vol + 1e-8)
                                        if obi < obi_thresh:
                                            execute_bailout(
                                                bailout_ex,
                                                base_sym,
                                                p,
                                                f"真瀑布崩塌 (跌幅 {adverse_move_pct*100:.2f}%, 量能放大 {actual_vol_multi:.1f}x)",
                                                current_close,
                                            )
                                            bailout_cooldown_locks[base_sym] = time.time()
                                            del tracking_mem[base_sym]
                                            continue

                                    elif pos_side == 'SHORT' and is_flash_surge:
                                        obi = ask_vol / (bid_vol + 1e-8)
                                        if obi < obi_thresh:
                                            execute_bailout(
                                                bailout_ex,
                                                base_sym,
                                                p,
                                                f"真逼空爆发 (涨幅 {adverse_move_pct*100:.2f}%, 量能放大 {actual_vol_multi:.1f}x)",
                                                current_close,
                                            )
                                            bailout_cooldown_locks[base_sym] = time.time()
                                            del tracking_mem[base_sym]
                                            continue
                except Exception as pos_e:
                    msg = f"[{base_sym}] 右脑处理单仓位失败: {pos_e}"
                    print(f"⚠️ {msg}")
                    log_system_alert("⚠️ 右脑单仓异常", base_sym, msg)

            # 🔄 同步写入文件，供前端 UI 读取展示虚线保底
            try:
                tmp_path = os.path.join(LIVE_DATA_DIR, "chandelier_state.json.tmp")
                with open(tmp_path, "w") as f:
                    json.dump(export_ui_mem, f)
                os.replace(tmp_path, os.path.join(LIVE_DATA_DIR, "chandelier_state.json"))
            except: pass

        except Exception as loop_e:
            if time.time() - last_worker_error_ts >= 30.0:
                msg = f"右脑主循环异常: {loop_e}"
                print(f"⚠️ {msg}")
                log_system_alert("⚠️ 右脑线程异常", "SYSTEM", msg)
                last_worker_error_ts = time.time()
        time.sleep(1)
