import os
import time
import json
import sqlite3
import copy
import requests
import random
from datetime import datetime
import ccxt
import threading
import tempfile
import fcntl
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout
from state_manager import state
from quant_config.config import (
    DATA_DIR, LIVE_DATA_DIR,
    MAX_NOTIONAL_BTC, MAX_NOTIONAL_ALTS,
    GATE_TRADING_MODE, LIVE_TRADING_ARMED, GATE_SANDBOX_MODE,
    GATE_REQUIRE_ISOLATED_MARGIN, MAX_MARGIN_PER_TRADE_USDT,
    MAX_BALANCE_RISK_PCT, MIN_FREE_USDT_AFTER_ORDER,
    DAILY_MAX_LOSS_USDT,
)
from dotenv import load_dotenv
load_dotenv()

MY_GATE_API_KEY  = os.environ.get("GATE_DEMO_API_KEY", "")
MY_GATE_SECRET   = os.environ.get("GATE_DEMO_SECRET", "")
PUSHPLUS_TOKEN   = os.environ.get("PUSHPLUS_TOKEN", "")

global_gate = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
global_gate.set_sandbox_mode(GATE_SANDBOX_MODE)

sync_ex = ccxt.gate({'apiKey': MY_GATE_API_KEY, 'secret': MY_GATE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
sync_ex.set_sandbox_mode(GATE_SANDBOX_MODE)

# 👇 【核心重构 1：异步推送队列】
def push_worker():
    while True:
        task = state.push_queue.get()
        if task is None: break
        try:
            # 这里才是真正干活的地方，独立线程发送，失败不影响主引擎
            resp = requests.post('http://www.pushplus.plus/send', json=task, timeout=10)
            if resp.status_code != 200:
                print(f"   ⚠️ PushPlus 拒收推送: {resp.text}")
            time.sleep(1.5) # 阻尼冷却，防止把接口打挂
        except Exception as e:
            print(f"   ⚠️ 异步推送网络错误: {e}")
        state.push_queue.task_done()
# 启动守护线程 (程序关闭时自动销毁)
threading.Thread(target=push_worker, daemon=True).start()


def atomic_write_json(payload, target_file):
    """Write JSON via a process-unique temp file to avoid cross-process rename races."""
    target_dir = os.path.dirname(target_file)
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_file = tempfile.mkstemp(
        prefix=os.path.basename(target_file) + ".",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, target_file)
    except Exception:
        try:
            os.unlink(tmp_file)
        except FileNotFoundError:
            pass
        raise


def acquire_process_lock(lock_name):
    """Acquire a non-blocking cross-process lock; return fd on success, else None."""
    lock_dir = os.path.join(LIVE_DATA_DIR, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, lock_name)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        return lock_fd
    except OSError:
        os.close(lock_fd)
        return None


def _balance_number(balance, bucket, currency="USDT"):
    try:
        return float((balance.get(bucket, {}) or {}).get(currency, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _extract_usdt_balance(balance):
    """Return free and total USDT from a CCXT balance payload."""
    free_usdt = _balance_number(balance, "free")
    total_usdt = _balance_number(balance, "total")
    if total_usdt <= 0:
        total_usdt = free_usdt
    return free_usdt, total_usdt


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _check_daily_loss_guard(balance):
    """UTC-day equity drawdown guard; blocks new entries after configured loss."""
    if DAILY_MAX_LOSS_USDT <= 0:
        if LIVE_TRADING_ARMED:
            return False, "live 模式未配置 DAILY_MAX_LOSS_USDT，拒绝真实资金开仓"
        return True, "日内亏损熔断未启用"

    _, total_usdt = _extract_usdt_balance(balance)
    if total_usdt <= 0:
        return False, "无法读取有效 USDT 总权益，日内亏损熔断拒绝开仓"

    today = datetime.utcnow().strftime("%Y-%m-%d")
    state_path = os.path.join(LIVE_DATA_DIR, "live_risk_guard_state.json")
    guard_state = _read_json_file(state_path)
    start_equity = _safe_float(guard_state.get("start_equity_usdt"), 0.0)

    if guard_state.get("utc_day") != today or start_equity <= 0:
        guard_state = {
            "utc_day": today,
            "start_equity_usdt": float(total_usdt),
            "last_equity_usdt": float(total_usdt),
            "daily_loss_usdt": 0.0,
            "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(guard_state, state_path)
        return True, f"日内权益基准已初始化: {total_usdt:.4f} USDT"

    daily_loss = max(0.0, start_equity - total_usdt)
    guard_state.update({
        "last_equity_usdt": float(total_usdt),
        "daily_loss_usdt": float(daily_loss),
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    })
    atomic_write_json(guard_state, state_path)

    if daily_loss >= DAILY_MAX_LOSS_USDT:
        return (
            False,
            f"日内亏损熔断触发: loss={daily_loss:.4f} >= limit={DAILY_MAX_LOSS_USDT:.4f} USDT",
        )
    return True, f"日内亏损检查通过: loss={daily_loss:.4f}/{DAILY_MAX_LOSS_USDT:.4f} USDT"


def _apply_pretrade_cap(exchange, symbol, requested_margin):
    """Apply live confirmation, balance, per-order cap, and daily loss guard."""
    try:
        requested_margin = float(requested_margin or 0.0)
    except (TypeError, ValueError):
        requested_margin = 0.0

    if requested_margin <= 0:
        return False, 0.0, "保证金预算无效，拒绝开仓"

    if GATE_TRADING_MODE == "live" and not LIVE_TRADING_ARMED:
        return (
            False,
            0.0,
            "GATE_TRADING_MODE=live 但 LIVE_TRADING_ENABLED 未设置为 I_UNDERSTAND_REAL_MONEY",
        )

    if not LIVE_TRADING_ARMED:
        return True, requested_margin, "sandbox 模式跳过真实余额硬闸"

    if MAX_MARGIN_PER_TRADE_USDT <= 0:
        return False, 0.0, "live 模式未配置 MAX_MARGIN_PER_TRADE_USDT，拒绝真实资金开仓"
    if MAX_BALANCE_RISK_PCT <= 0:
        return False, 0.0, "MAX_BALANCE_RISK_PCT 必须为正数"

    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        return False, 0.0, f"真实余额读取失败，拒绝开仓: {e}"

    free_usdt, total_usdt = _extract_usdt_balance(balance)
    if free_usdt <= 0:
        return False, 0.0, f"可用 USDT 余额不足: free={free_usdt:.4f}"

    daily_ok, daily_reason = _check_daily_loss_guard(balance)
    if not daily_ok:
        return False, 0.0, daily_reason

    free_after_reserve = max(0.0, free_usdt - max(0.0, MIN_FREE_USDT_AFTER_ORDER))
    balance_cap = free_usdt * MAX_BALANCE_RISK_PCT
    hard_cap = min(MAX_MARGIN_PER_TRADE_USDT, balance_cap, free_after_reserve)
    if hard_cap <= 0:
        return (
            False,
            0.0,
            f"余额保护后无可用保证金: free={free_usdt:.4f}, reserve={MIN_FREE_USDT_AFTER_ORDER:.4f}",
        )

    capped_margin = min(requested_margin, hard_cap)
    if capped_margin < requested_margin:
        msg = (
            f"{symbol} 保证金预算由 {requested_margin:.4f} 压降至 {capped_margin:.4f} USDT | "
            f"free={free_usdt:.4f}, total={total_usdt:.4f}, {daily_reason}"
        )
        return True, capped_margin, msg

    return True, capped_margin, f"余额硬闸通过 | free={free_usdt:.4f}, total={total_usdt:.4f}, {daily_reason}"


def _margin_mode_text(position):
    if not position:
        return ""
    info = position.get("info", {}) or {}
    candidates = [
        position.get("marginMode"),
        position.get("margin_mode"),
        info.get("margin_mode"),
        info.get("marginMode"),
        info.get("margin_type"),
        info.get("marginType"),
    ]
    for raw in candidates:
        text = str(raw or "").strip().lower()
        if text:
            return text
    return ""


def _ensure_isolated_margin(exchange, gate_symbol, target_leverage=None):
    if not GATE_REQUIRE_ISOLATED_MARGIN:
        return True, "逐仓校验未启用"

    leverage_value = float(target_leverage or 1.0)
    try:
        # Gate does not implement ccxt.set_margin_mode(); for futures, isolated
        # mode is expressed by setting a non-zero leverage on the position.
        exchange.set_leverage(leverage_value, gate_symbol, {"marginMode": "isolated"})
        set_ok = True
        set_msg = f"已请求交易所切换/保持逐仓，并预设杠杆 {leverage_value:g}x"
    except Exception as e:
        err_text = str(e).lower()
        already_isolated = any(
            token in err_text
            for token in ("no_change", "not change", "already", "same", "isolated")
        )
        set_ok = already_isolated
        set_msg = "交易所返回已处于逐仓/无需修改" if already_isolated else f"逐仓设置失败: {e}"
        if LIVE_TRADING_ARMED and not already_isolated:
            return False, set_msg

    try:
        positions = exchange.fetch_positions([gate_symbol])
    except Exception as e:
        if LIVE_TRADING_ARMED:
            return False, f"逐仓校验读取仓位失败: {e}"
        return True, f"{set_msg}；sandbox 仓位校验失败但不阻断: {e}"

    active_positions = [p for p in positions if _position_contracts(p) > 1e-8]
    if not active_positions:
        return True, set_msg if set_ok else f"{set_msg}；当前无仓位可二次校验"

    for pos in active_positions:
        mode_text = _margin_mode_text(pos)
        if "cross" in mode_text:
            return False, f"{gate_symbol} 当前仓位为全仓模式 ({mode_text})，拒绝继续加仓"
        if "isolated" in mode_text:
            return True, f"{gate_symbol} 逐仓校验通过 ({mode_text})"

    if LIVE_TRADING_ARMED:
        return False, f"{gate_symbol} 无法从交易所返回字段确认逐仓模式，拒绝真实资金加仓"
    return True, f"{gate_symbol} 未能明确校验逐仓；sandbox 模式放行"

def portfolio_sync_worker():
    """ 
    🏦 [供货线程] 全能查询员：
    集中处理所有私有查询任务，更新内存账本并同步 JSON 供前端读取。
    """
    lock_fd = acquire_process_lock("portfolio_sync_worker.lock")
    if lock_fd is None:
        print("⚠️ [供货线程] 检测到其他进程已接管 portfolio 快照写入，本进程跳过该线程。")
        return

    portfolio_sync_worker.lock_fd = lock_fd
    if not hasattr(portfolio_sync_worker, 'empty_positions_streak'):
        portfolio_sync_worker.empty_positions_streak = 0

    print("🏦 [供货线程] 集中查询引擎已启动 (隔离 Nonce B)...")
    while True:
        try:
            # 1. 独立查资金 (最核心，给主引擎起步用)
            try:
                bal = sync_ex.fetch_balance()
                state.portfolio_memory['balance'] = bal
            except Exception as e:
                print(f"   ⚠️ 供货线程-资金查询延迟: {e}")

            # 2. 独立查仓位 (同步给右脑)
            try:
                pos = sync_ex.fetch_positions()
                new_active_pos_map = {
                    p['symbol'].split(':')[0].replace('/', ''): p
                    for p in pos
                    if abs(float(p.get('contracts', 0) or 0)) > 1e-8
                }

                had_active_positions = bool(state.active_pos_map)
                suspicious_empty_positions = not new_active_pos_map and had_active_positions

                if suspicious_empty_positions:
                    portfolio_sync_worker.empty_positions_streak += 1
                    if portfolio_sync_worker.empty_positions_streak < 2:
                        raise RuntimeError("仓位快照瞬时返回空列表，已保留上一帧等待二次确认")
                else:
                    portfolio_sync_worker.empty_positions_streak = 0

                state.portfolio_memory['positions'] = pos

                with state.pos_lock:
                    dead_keys = [k for k in list(state.active_pos_map.keys()) if k not in new_active_pos_map]
                    for k in dead_keys:
                        old_p = state.active_pos_map.get(k)
                        state.active_pos_map.pop(k, None)
                        
                        # 👇 [新增：被动平仓推送逻辑]
                        # 如果该币种在 10 秒内没有触发过主动熔断推送，说明是交易所 OCO 触发的被动平仓
                        last_push_ts = state.recent_pushes.get(k, 0)
                        is_passive_close = time.time() - last_push_ts > 10
                        if is_passive_close:
                            try:
                                _record_passive_close_to_db(k, old_p, close_side="exchange_close")
                            except Exception as e:
                                print(f"   ⚠️ 被动平仓流水补记失败: {e}")

                        if is_passive_close and PUSHPLUS_TOKEN:
                            try:
                                dir_str = "多单 🟢" if str(old_p.get('side', '')).lower() == 'long' else "空单 🔴"
                                contracts = old_p.get('contracts', 0)
                                abs_profit, realized_roe = _estimate_close_pnl_from_position(old_p)
                                pnl_line = _format_close_pnl_line(abs_profit, realized_roe, estimated=True)
                                exit_reason = _infer_passive_exit_reason(old_p, abs_profit, realized_roe)
                                info = old_p.get("info", {}) or {}
                                exit_px = _safe_float(old_p.get("markPrice", info.get("mark_price", info.get("markPx", 0))), 0.0)
                                exit_px_line = f"<b>🎯 估算平仓价：</b>${exit_px:.4f}<br>" if exit_px > 0 else ""

                                push_title = f"🏁 [Gate实盘] {k} 仓位已平仓"
                                push_content = (
                                    f"<b>🔹 交易标的：</b>{k} ({dir_str})<br>"
                                    f"<b>🔸 出场原因：</b>{exit_reason}<br>"
                                    f"<b>📦 结算数量：</b>{contracts} 张<br>"
                                    f"{exit_px_line}"
                                    f"{pnl_line}"
                                    f"<b>⏰ 时间：</b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                state.push_queue.put({"token": PUSHPLUS_TOKEN, "title": push_title, "content": push_content, "template": "html"})
                                state.recent_pushes[k] = time.time()
                            except Exception as e:
                                print(f"   ⚠️ 被动平仓推送入队失败: {e}")

                    state.active_pos_map.update(new_active_pos_map)
            except Exception as e:
                print(f"   ⚠️ 供货线程-仓位查询延迟: {e}")

            # 3. 独立查条件单 (最容易被限流，静默处理)
            try:
                stops = sync_ex.fetch_open_orders(params={'stop': True})
                state.portfolio_memory['open_stops'] = stops
            except Exception:
                pass 

            # 💡 只要走到这里，证明线程活着，必须更新时间戳！
            state.portfolio_memory['last_sync_ts'] = time.time()
            
            # 👇 【新增：每天刷新静态市场规则】
            current_day = datetime.now().day
            if getattr(portfolio_sync_worker, 'last_refresh_day', None) != current_day:
                try:
                    state.gate_markets = sync_ex.load_markets(True) # 强制刷新
                    portfolio_sync_worker.last_refresh_day = current_day
                    print(f"🔄 市场静态规则已自动刷新更新 (Day: {current_day})")
                except Exception as e:
                    print(f"⚠️ 刷新静态市场规则失败: {e}")
            
            # 4. 同步至磁盘 JSON 供 app.py 渲染
            # 深度复制：防止复制了正在被其他线程修改的字典而导致 Python 报错
            sync_state = copy.deepcopy({
                'balance': state.portfolio_memory.get('balance', {}),
                'positions': state.portfolio_memory.get('positions', []),
                'open_stops': state.portfolio_memory.get('open_stops', []),
                'updated_at': time.time(),
                'writer_pid': os.getpid(),
            })

            has_any_payload = (
                bool(sync_state['balance'])
                or bool(sync_state['positions'])
                or bool(sync_state['open_stops'])
            )
            last_snapshot = getattr(portfolio_sync_worker, 'last_good_snapshot', None)
            if not has_any_payload and last_snapshot:
                sync_state = copy.deepcopy(last_snapshot)
                sync_state['updated_at'] = time.time()
                sync_state['writer_pid'] = os.getpid()
            elif has_any_payload:
                portfolio_sync_worker.last_good_snapshot = copy.deepcopy(sync_state)

            target_file = os.path.join(LIVE_DATA_DIR, "portfolio_state.json")
            atomic_write_json(sync_state, target_file)

        except Exception as e:
            print(f"   ⚠️ 供货线程遭遇系统级异常: {e}")
        
        # 每 3 秒执行一次全方位大排查
        time.sleep(3)
# 启动快照线程
threading.Thread(target=portfolio_sync_worker, daemon=True).start()

# ==========================================
# 🛑 【V6.0 核心基建：全局缓存池与基础重试器】
# ==========================================
global_market_cache = {} # 用于存储 Stale-State Padding 缓存数据

def safe_api_call(func, *args, max_retries=3, initial_wait=1.0, **kwargs):
    """基础 API 调用包装器（已摘除限频，仅用于资金费率等极少数未进网关的兜底请求）"""
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (NetworkError, ExchangeError) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(wait_time)
            wait_time = (wait_time * 2) + random.uniform(0, 1)

def log_system_alert(level, symbol, message):
    """将底层系统警报写入独立数据库供前端实时呈现"""
    try:
        log_db_path = os.path.join(LIVE_DATA_DIR, "logs", "engine_system_logs.db")
        os.makedirs(os.path.dirname(log_db_path), exist_ok=True)
        with sqlite3.connect(log_db_path, timeout=5.0, check_same_thread=False) as sys_conn:
            sys_cursor = sys_conn.cursor()
            sys_cursor.execute('PRAGMA journal_mode=WAL;')
            # 创建专门的系统警报表
            sys_cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (log_ts TIMESTAMP, level TEXT, symbol TEXT, message TEXT)''')
            sys_cursor.execute('''INSERT INTO system_alerts (log_ts, level, symbol, message) VALUES (?, ?, ?, ?)''',
                               (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), level, symbol, message))
            sys_conn.commit()
    except Exception:
        pass


def _normalize_symbol_key(raw_symbol):
    """统一符号格式，兼容 BTCUSDT / BTC/USDT:USDT / BTC/USDT。"""
    return str(raw_symbol or "").split(":")[0].replace("/", "")


def _safe_float(value, default=0.0):
    """容错解析数值，避免快照字段为空或格式异常。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_close_pnl_from_position(position, exit_price=None):
    """基于最近一帧仓位快照，估算平仓盈亏与 ROE。"""
    if not position:
        return None, None

    info = position.get("info", {}) or {}
    pos_side = str(position.get("side", info.get("side", "")) or "").upper()
    contracts = abs(_safe_float(position.get("contracts", info.get("size", info.get("contracts", 0))), 0.0))
    entry_px = _safe_float(position.get("entryPrice", info.get("entry_price", info.get("avgPx", 0))), 0.0)
    mark_px = _safe_float(
        exit_price if exit_price is not None else position.get("markPrice", info.get("mark_price", info.get("markPx", 0))),
        0.0,
    )
    leverage = _safe_float(position.get("leverage", info.get("leverage", info.get("lever", 1))), 1.0)
    contract_size = _safe_float(position.get("contractSize", info.get("contract_size", 1)), 1.0)
    if leverage <= 0:
        leverage = 1.0

    raw_pnl = None
    if entry_px > 0 and mark_px > 0 and contracts > 0:
        if pos_side == "LONG":
            raw_pnl = (mark_px - entry_px) / entry_px
        elif pos_side == "SHORT":
            raw_pnl = (entry_px - mark_px) / entry_px

    if raw_pnl is not None:
        abs_profit = raw_pnl * (contracts * contract_size * entry_px)
        realized_roe = raw_pnl * leverage
        return abs_profit, realized_roe

    upl_raw = info.get("unrealised_pnl", info.get("unrealized_pnl"))
    abs_profit = _safe_float(upl_raw, None) if upl_raw is not None else None

    pct_raw = position.get("percentage")
    if pct_raw is None:
        pct_raw = info.get("uplRatio")
    realized_roe = _safe_float(pct_raw, None) if pct_raw is not None else None
    if realized_roe is not None and -1.0 < realized_roe < 1.0 and realized_roe != 0.0:
        realized_roe *= 100.0
    if realized_roe is not None:
        realized_roe /= 100.0

    return abs_profit, realized_roe


def _format_close_pnl_line(abs_profit, realized_roe, estimated=True):
    """生成平仓推送中的客观盈亏行。"""
    prefix = "估算" if estimated else "结算"
    if abs_profit is None and realized_roe is None:
        return f"<b>💰 {prefix}盈亏：</b>快照不足，暂无法估算<br>"

    pnl_probe = abs_profit if abs_profit is not None else realized_roe
    if pnl_probe > 0:
        pnl_label = f"🟢 {prefix}盈利"
    elif pnl_probe < 0:
        pnl_label = f"🔴 {prefix}亏损"
    else:
        pnl_label = f"🟡 {prefix}保本"

    pnl_parts = []
    if abs_profit is not None:
        pnl_parts.append(f"{abs_profit:+.2f} USDT")
    if realized_roe is not None:
        pnl_parts.append(f"{realized_roe:+.2%}")
    return f"<b>{pnl_label}：</b>{' '.join(pnl_parts)}<br>"


def _infer_passive_exit_reason(position, abs_profit, realized_roe):
    """
    交易所条件单/手动平仓没有回传明确触发源。
    这里只按最后一帧仓位快照给出客观推断，避免把止盈、止损、手动误报成确定事实。
    """
    pnl_probe = abs_profit if abs_profit is not None else realized_roe
    if pnl_probe is None:
        return "交易所仓位消失，可能为条件单触发或手动平仓；快照不足，无法判断盈亏方向"
    if pnl_probe > 0:
        return "交易所仓位消失；按最后快照看为盈利出场，倾向止盈/追踪止盈或手动止盈"
    if pnl_probe < 0:
        return "交易所仓位消失；按最后快照看为亏损出场，倾向兜底止损或手动止损"
    return "交易所仓位消失；按最后快照看接近保本，可能为手动平仓或条件单贴近成本触发"


def _record_passive_close_to_db(symbol_key, position, close_side="exchange_close"):
    """为交易所被动平仓补记一笔 close 流水，供历史本纪识别。"""
    if not symbol_key or not position:
        return

    info = position.get("info", {}) or {}
    contracts = abs(_safe_float(position.get("contracts", info.get("size", info.get("contracts", 0))), 0.0))
    contract_size = _safe_float(position.get("contractSize", info.get("contract_size", 1)), 1.0)
    exit_price = _safe_float(
        position.get("markPrice", info.get("mark_price", info.get("markPx", 0))),
        0.0,
    )
    leverage = _safe_float(position.get("leverage", info.get("leverage", info.get("lever", 1))), 1.0)
    if leverage <= 0:
        leverage = 1.0

    notional = _safe_float(position.get("notional", info.get("value", 0)), 0.0)
    if notional <= 0 and contracts > 0 and contract_size > 0 and exit_price > 0:
        notional = contracts * contract_size * exit_price
    margin_used = (notional / leverage) if leverage > 0 and notional > 0 else 0.0

    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol_key}.db")
    with sqlite3.connect(db_name, timeout=20.0, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS live_trades
               (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                amount REAL, fill_price REAL, slippage REAL,
                tp_price REAL, sl_price REAL, margin_used REAL, leverage REAL)'''
        )
        try:
            cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
        except sqlite3.OperationalError:
            pass

        cursor.execute(
            '''INSERT INTO live_trades
               (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'passive_close_sync',
                close_side,
                float(contracts),
                float(exit_price),
                0.0,
                0.0,
                0.0,
                float(margin_used),
                float(leverage),
            ),
        )
        conn.commit()

    abs_profit, realized_roe = _estimate_close_pnl_from_position(position, exit_price=exit_price)
    exit_reason = _infer_passive_exit_reason(position, abs_profit, realized_roe)
    pos_side = str(position.get("side", info.get("side", "")) or "").upper()
    dir_str = "LONG" if pos_side == "LONG" else "SHORT" if pos_side == "SHORT" else pos_side or "UNKNOWN"
    pnl_parts = []
    if abs_profit is not None:
        pnl_parts.append(f"{abs_profit:+.2f} USDT")
    if realized_roe is not None:
        pnl_parts.append(f"ROE {realized_roe:+.2%}")
    pnl_text = " | ".join(pnl_parts) if pnl_parts else "盈亏快照不足"
    price_text = f"${exit_price:.4f}" if exit_price > 0 else "未知"
    log_system_alert(
        "🏁 被动出场",
        symbol_key,
        (
            f"{dir_str} 仓位已从交易所快照消失 | "
            f"估算平仓价: {price_text} | 数量: {contracts:g} 张 | {pnl_text} | {exit_reason}"
        ),
    )


def is_gate_close_position_stop(stop_order):
    """识别 Gate 的“平全仓”条件单。"""
    initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
    auto_size = str(initial.get("auto_size", "") or "").lower()
    close_flag = initial.get("close")
    if close_flag is None:
        close_flag = initial.get("is_close")
    return bool(close_flag) or auto_size in {"close_long", "close_short"}


def get_gate_stop_coverage_size(stop_order, expected_contracts):
    """
    统一解析止损单覆盖张数。
    Gate 的平仓条件单要求 size=0，此时用 expected_contracts 代表其覆盖全仓。
    """
    if is_gate_close_position_stop(stop_order):
        return float(expected_contracts or 0.0)

    initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
    raw_size = initial.get("size", stop_order.get("amount", 0.0))
    try:
        return abs(float(raw_size or 0.0))
    except (TypeError, ValueError):
        return None


def _gate_stop_text(stop_order):
    initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
    return str(stop_order.get("clientOrderId", "") or initial.get("text", "") or "")


def _gate_stop_auto_size(stop_order):
    initial = (stop_order.get("info", {}) or {}).get("initial", {}) or {}
    return str(initial.get("auto_size", "") or "").lower()


def _position_contracts(position):
    if not position:
        return 0.0
    info = position.get("info", {}) or {}
    return abs(_safe_float(position.get("contracts", info.get("size", info.get("contracts", 0))), 0.0))


def _position_entry_price(position):
    if not position:
        return 0.0
    info = position.get("info", {}) or {}
    return _safe_float(position.get("entryPrice", info.get("entry_price", info.get("avgPx", 0))), 0.0)


def _is_same_side_close_stop(stop_order, close_side):
    expected_auto_size = "close_long" if str(close_side).upper() == "LONG" else "close_short"
    auto_size = _gate_stop_auto_size(stop_order)
    if auto_size:
        return auto_size == expected_auto_size
    return is_gate_close_position_stop(stop_order)


def cancel_obsolete_gate_close_stops(exchange, gate_symbol, keep_text, close_side, stop_orders=None):
    """
    新止损确认挂上后，撤掉同币种同方向旧的平仓止损。
    只清理带 sl/tsl 标识的平仓条件单，避免误伤其它人工挂单。
    """
    keep_text = str(keep_text or "").lower()
    gate_key = _normalize_symbol_key(gate_symbol)
    stops = stop_orders
    if stops is None:
        stops = exchange.fetch_open_orders(gate_symbol, params={'stop': True})

    cancelled = 0
    for stop_order in stops:
        if _normalize_symbol_key(stop_order.get("symbol")) != gate_key:
            continue

        text = _gate_stop_text(stop_order).lower()
        if text == keep_text:
            continue
        if "sl" not in text and "tsl" not in text:
            continue
        if not _is_same_side_close_stop(stop_order, close_side):
            continue

        try:
            exchange.cancel_order(stop_order["id"], gate_symbol, params={'stop': True})
            cancelled += 1
        except Exception as e:
            print(f"   ⚠️ 旧止损单撤销失败，保留由右脑后续校验: id={stop_order.get('id')} err={e}")

    return cancelled


def build_gate_close_stop_initial(native_symbol, text_label, close_side):
    """构造 Gate 平全仓条件止损单的 initial 字段。"""
    normalized_side = str(close_side or "").upper()
    auto_size = "close_long" if normalized_side == "LONG" else "close_short"
    return {
        "contract": native_symbol,
        "size": 0,
        "close": False,
        "price": "0",
        "tif": "ioc",
        "text": text_label,
        "reduce_only": True,
        "auto_size": auto_size,
    }


def _inspect_stop_order_coverage(position, stop_orders):
    """核对交易所止损挂单是否覆盖了当前整笔仓位。"""
    if not position:
        return {
            "status": "no_position",
            "expected_contracts": 0.0,
            "max_stop_size": 0.0,
            "coverage_ratio": 1.0
        }

    info = position.get("info", {}) or {}
    try:
        expected_contracts = abs(float(info.get("size", position.get("contracts", 0)) or 0.0))
    except (TypeError, ValueError):
        expected_contracts = 0.0

    if expected_contracts <= 0:
        return {
            "status": "no_position",
            "expected_contracts": 0.0,
            "max_stop_size": 0.0,
            "coverage_ratio": 1.0
        }

    parsed_sizes = []
    pos_key = _normalize_symbol_key(position.get("symbol"))
    for stop_order in stop_orders:
        if _normalize_symbol_key(stop_order.get("symbol")) != pos_key:
            continue

        parsed_size = get_gate_stop_coverage_size(stop_order, expected_contracts)
        if parsed_size is not None:
            parsed_sizes.append(parsed_size)

    if not parsed_sizes:
        return {
            "status": "missing_stop",
            "expected_contracts": expected_contracts,
            "max_stop_size": 0.0,
            "coverage_ratio": 0.0
        }

    max_stop_size = max(parsed_sizes)
    coverage_ratio = max_stop_size / (expected_contracts + 1e-9)
    if coverage_ratio < 0.9:
        status = "undercovered_stop"
    elif coverage_ratio > 1.1:
        status = "oversized_stop"
    else:
        status = "ok"

    return {
        "status": status,
        "expected_contracts": expected_contracts,
        "max_stop_size": max_stop_size,
        "coverage_ratio": coverage_ratio
    }

def execute_gate_trade(symbol, signal, leverage, current_price, tp_pct, sl_pct, risk_margin_usd):
    """
    工业级 API 下单器 (异构版)：执行下单、计算滑点并落库
    """
    try:
        # 【核心修复】：彻底抛弃 Gate 原生的下划线写法，拥抱 CCXT 跨平台标准永续符号
        # 构建 CCXT 标准合约代码格式，例如: 'APT/USDT' -> 'APT/USDT:USDT'
        if symbol.endswith("USDT"):
            gate_symbol = f"{symbol[:-4]}/USDT:USDT"
        else:
            gate_symbol = symbol + ":" + symbol.split('/')[1]
        side = 'buy' if signal == 1 else 'sell'
        # 👇 新增这一行：将旧的局部变量名直接映射到我们预热好的全局单例上
        exchange_private = global_gate

        # ---------------------------------------------------------
        # 1. 仓位互斥锁：支持同向升档到目标绝对杠杆，反向信号统一交由右脑评估
        # ---------------------------------------------------------
        order_sent = False 
        has_existing_same_side = False
        existing_contracts = 0.0
        current_exchange_leverage = 0.0
        positions = exchange_private.fetch_positions([gate_symbol])
        for pos in positions:
            pos_contracts = abs(float(pos.get('contracts', 0) or 0))
            if pos_contracts > 1e-8:
                current_pos_side = str(pos.get('side', '')).lower()
                
                if (side == 'buy' and current_pos_side == 'short') or (side == 'sell' and current_pos_side == 'long'):
                    msg = "检测到反向入场请求，OMS 已忽略；由右脑按原始概率 Top-1 规则决定是否退出。"
                    print("ℹ️", msg)
                    log_system_alert("ℹ️ 反向信号忽略", symbol, msg)
                    return False, 0.0
                else:
                    has_existing_same_side = True
                    existing_contracts = pos_contracts
                    info = pos.get("info", {}) or {}
                    try:
                        current_exchange_leverage = float(
                            pos.get("leverage", info.get("lever", info.get("leverage", 0))) or 0.0
                        )
                    except (TypeError, ValueError):
                        current_exchange_leverage = 0.0
                    msg = f"📈 侦测到 {gate_symbol} 同向存续仓位，准备升档到目标绝对杠杆 {leverage}x。"
                    print("ℹ️", msg)

        # ---------------------------------------------------------
        # 2. 面值与张数计算 (工业级精度适配与风控智能压降)
        # ---------------------------------------------------------
        markets = state.gate_markets
        
        # 👇 【阻击冷启动竞态条件 (Cold Start Race Condition)】
        # 如果主引擎光速触发了第一笔交易，而后台供货线程还在请求队列中，
        # 此时 markets 必定为空。强制主线程同步阻塞式拉取，补齐时序差。
        if not markets:
            print("⏳ [系统保护] 静态市场字典尚未就绪，正强制同步 Gate.io 市场规则...")
            try:
                markets = exchange_private.load_markets(True)
                state.gate_markets = markets
            except Exception as e:
                print(f"   ⚠️ 紧急加载市场规则失败，可能遭遇网络限流: {e}")
                
        if gate_symbol not in markets:
            # 加入 CCXT 探针：如果真的被跳过，将前 5 个物理存在的交易对名称打印出来，
            # 供我们排查是否是 Gate.io 测试网悄悄变更了永续合约的命名规范
            sample_markets = list(markets.keys())[:5] if markets else "网络拉取失败"
            msg = f"🚧 模拟盘受限拦截：Gate.io 未上架 {gate_symbol} 永续合约，已跳过。(当前测试网探测到: {sample_markets}...)"
            print("⚠️", msg)
            log_system_alert("⚠️ 拦截", symbol, msg)
            return False, 0.0
            
        market = markets[gate_symbol]
        contract_size = float(market['contractSize'])

        guard_ok, risk_margin_usd, guard_reason = _apply_pretrade_cap(
            exchange_private,
            symbol,
            risk_margin_usd,
        )
        if not guard_ok:
            msg = f"❌ 实盘安全闸拒绝开仓: {guard_reason}"
            print("⚠️", msg)
            log_system_alert("❌ 实盘安全闸", symbol, msg)
            return False, 0.0
        if guard_reason:
            print(f"🧯 [实盘安全闸] {guard_reason}")
            log_system_alert("🧯 实盘安全闸", symbol, guard_reason)

        target_notional = risk_margin_usd * leverage
        
        # 👇 【智能修复 1：动态风险限额压降 (Risk Limit Capping)】
        # 直接使用 config 中定义的 MAX_NOTIONAL_BTC / MAX_NOTIONAL_ALTS，自动压降至安全水位
        base_coin = symbol.replace('USDT', '').split('/')[0]
        safe_notional_limit = MAX_NOTIONAL_BTC if base_coin in ['BTC', 'ETH'] else MAX_NOTIONAL_ALTS
        
        if target_notional > safe_notional_limit:
            target_notional = safe_notional_limit
            msg_limit = f"⚠️ 触发风控红线，名义价值已自动压降至 {safe_notional_limit} USDT"
            print(f"   {msg_limit}")
            log_system_alert("📉 仓位降级", symbol, msg_limit)

        existing_notional = existing_contracts * contract_size * current_price if has_existing_same_side else 0.0
        order_notional = max(0.0, target_notional - existing_notional)
        if has_existing_same_side and order_notional <= max(target_notional * 1e-6, 1e-8):
            msg = (
                f"ℹ️ {gate_symbol} 当前同向仓位已达到或超过目标杠杆 {leverage}x，"
                f"无需继续补仓 (existing={existing_notional:.2f} / target={target_notional:.2f} USDT)。"
            )
            print(msg)
            log_system_alert("ℹ️ 升档跳过", symbol, msg)
            return False, 0.0

        raw_amount = order_notional / (current_price * contract_size)
        order_amount = float(exchange_private.amount_to_precision(gate_symbol, raw_amount))

        # 👇 【智能修复 2：市价单防滑点极限截断 (Market Order Size Capping)】
        # 提取交易所明确规定的市价单最大下单张数兜底
        max_market_size = float(market['info'].get('max_market_size', 100000))
        if order_amount > max_market_size:
            order_amount = max_market_size
            msg_size = f"⚠️ 触发市价防滑点红线，已自动截断至 {max_market_size} 张"
            print(f"   {msg_size}")
            log_system_alert("🔪 张数截断", symbol, msg_size)
            
        # 👇 【极其重要的账本修复】：按本次补单的真实目标杠杆反推增量保证金占用。
        risk_margin_usd = (order_amount * contract_size * current_price) / leverage

        if order_amount < float(market['limits']['amount']['min']):
            msg = f"❌ 资金不足或被过度压降，无法开启最小单位 ({market['limits']['amount']['min']} 张)。"
            print("⚠️", msg)
            log_system_alert("❌ 资金不足", symbol, msg)
            return False, 0.0

        margin_ok, margin_reason = _ensure_isolated_margin(exchange_private, gate_symbol, leverage)
        if not margin_ok:
            msg = f"❌ 逐仓保护拒绝开仓: {margin_reason}"
            print("⚠️", msg)
            log_system_alert("❌ 逐仓保护", symbol, msg)
            return False, 0.0
        if margin_reason:
            print(f"🧯 [逐仓保护] {margin_reason}")
            log_system_alert("🧯 逐仓保护", symbol, margin_reason)

        # 统一绝对杠杆语义：无论首次开仓还是同向升档，都先把交易所杠杆对齐到 Tier 目标值。
        margin_preselected_leverage = (
            margin_ok
            and GATE_REQUIRE_ISOLATED_MARGIN
            and "预设杠杆" in str(margin_reason or "")
        )
        if not margin_preselected_leverage and abs(current_exchange_leverage - leverage) > 0.05:
            try:
                exchange_private.set_leverage(leverage, gate_symbol, {"marginMode": "isolated"})
                time.sleep(0.5)
            except Exception as e:
                msg = f"⚠️ 无法将 {gate_symbol} 杠杆对齐到目标值 {leverage}x: {e}"
                print(msg)
                log_system_alert("⚠️ 杠杆对齐失败", symbol, msg)
                return False, 0.0

        # ---------------------------------------------------------
        # 3. 核心市价开仓 (增强参数传递)
        # ---------------------------------------------------------
        # ✅ Gate.io 官方 API：rule=1 是 >=，rule=2 是 <=
        # 多单 (buy) 止损在下方，需跌破触发，故 rule=2。你的原版逻辑是正确的！
        if side == 'buy':  
            tp_rule, sl_rule = 1, 2  
        else:              
            tp_rule, sl_rule = 2, 1  

        native_symbol = exchange_private.market_id(gate_symbol)

        # 🚨 修复 1：安全的四舍五入防截断 + close=True 防止仓位翻倍
        def safe_post_condition(rule, price, text_label, stop_size, close_side):
            exact_size = int(round(float(stop_size)))
            if exact_size <= 0:
                msg_err = f"⚠️ 计算所得止损张数 ({exact_size}) <= 0，放弃挂载止损单。"
                print(msg_err)
                log_system_alert("⚠️ 止损挂单异常", symbol, msg_err)
                return False
                
            for attempt in range(3):
                try:
                    exchange_private.private_futures_post_settle_price_orders({
                        "settle": "usdt",
                        "initial": build_gate_close_stop_initial(native_symbol, text_label, close_side),
                        "trigger": {
                            "strategy_type": 0, 
                            "price_type": 1, 
                            "price": str(exchange_private.price_to_precision(gate_symbol, price)), 
                            "rule": rule
                        }
                    })
                    return True
                except Exception as e:
                    print(f"   ⚠️ {text_label.upper()} 挂载重试 {attempt+1}/3 (网络波动): {e}")
                    time.sleep(1)
            return False
        # 【核心修复】：使用基础的 create_order，显式指定 type 为 market，并利用 params 传递附加参数
        order_sent = True # 👇 标记为已发送
        order = exchange_private.create_order(
            symbol=gate_symbol,
            type='market',
            side=side,
            amount=order_amount,
            price=None, # 市价单不需要价格
            params={
                'reduceOnly': False # 明确表示这是一笔开仓单
            }
        )

        # 稍等 1.5 秒拉取真实成交数据，确保测试网仓位绝对同步
        time.sleep(1.5)
        filled_order = exchange_private.fetch_order(order['id'], gate_symbol)
        actual_fill_price = float(filled_order.get('average') or filled_order.get('price') or current_price)
        slippage = abs(actual_fill_price - current_price) / current_price

        try:
            post_fill_positions = exchange_private.fetch_positions([gate_symbol])
        except Exception as e:
            print(f"   ⚠️ 成交后仓位均价读取失败，临时使用本次成交价计算止损: {e}")
            post_fill_positions = []
        active_position = next(
            (
                p for p in post_fill_positions
                if _position_contracts(p) > 1e-8
                and str(p.get('side', '')).lower() == ('long' if side == 'buy' else 'short')
            ),
            None
        )
        position_entry_price = _position_entry_price(active_position) or actual_fill_price
        position_contracts = _position_contracts(active_position) or order_amount
        
        # 基于真实的成交均价，计算物理止盈止损位
        # 统一量纲：传入的 tp_pct 和 sl_pct 是带杠杆的预期收益率/亏损率，因此需要除以杠杆得到实际的 K 线涨跌幅
        actual_tp_move = abs(tp_pct) 
        actual_sl_move = abs(sl_pct) 
        
        tp_price = position_entry_price * (1 + actual_tp_move) if side == 'buy' else position_entry_price * (1 - actual_tp_move)
        sl_price = position_entry_price * (1 - actual_sl_move) if side == 'buy' else position_entry_price * (1 + actual_sl_move)

        # ---------------------------------------------------------
        # 【修改点2：移除硬止盈，只挂兜底硬止损 (给追踪止盈让路)】
        # ---------------------------------------------------------
        # 挂载平全仓硬止损；同向升档时用交易所最新整体均价重算统一防线。
        close_side = "LONG" if side == "buy" else "SHORT"
        sl_label = f"sl_tier_{int(time.time() * 1000)}"
        sl_success = safe_post_condition(sl_rule, sl_price, sl_label, position_contracts, close_side)
        
        # 终极物理校验
        time.sleep(1.5) 
        try:
            verify_stops = exchange_private.fetch_open_orders(gate_symbol, params={'stop': True})
            active_texts = [_gate_stop_text(o).lower() for o in verify_stops]
            
            # 🚨 修复 2：只要含有 sl 字样即算挂单成功，防止因 'sl_tier' != 'sl' 造成的 Python 语法误判
            if sl_success and any(t == sl_label.lower() for t in active_texts):
                print("🔔 交易所底层 OCO 兜底止损单【挂单且物理校验成功】！")
                cancelled_count = cancel_obsolete_gate_close_stops(
                    exchange_private,
                    gate_symbol,
                    sl_label,
                    close_side,
                    verify_stops,
                )
                if cancelled_count:
                    print(f"🧹 已撤销 {cancelled_count} 张同币种旧止损单，仅保留最新统一防线。")
                    verify_stops = exchange_private.fetch_open_orders(gate_symbol, params={'stop': True})
            else:
                msg_oco_fail = f"物理兜底止损单校验未完全通过 (现存: {len(verify_stops)}个)，可能存在遗漏，交由右脑兜底。"
                print(f"❌ 警告：{gate_symbol} {msg_oco_fail}")
                log_system_alert("⚠️ 挂单残缺", symbol, msg_oco_fail)

            verify_positions = post_fill_positions
            active_position = next(
                (p for p in verify_positions if _position_contracts(p) > 1e-8),
                None
            )
            coverage_check = _inspect_stop_order_coverage(active_position, verify_stops)
            if coverage_check["status"] != "ok":
                msg_cov = (
                    f"{gate_symbol} 止损覆盖异常: status={coverage_check['status']} | "
                    f"stop_size={coverage_check['max_stop_size']:.4f} | "
                    f"position_size={coverage_check['expected_contracts']:.4f} | "
                    f"coverage={coverage_check['coverage_ratio']:.2%}"
                )
                print(f"⚠️ {msg_cov}")
                log_system_alert("⚠️ 止损覆盖异常", symbol, msg_cov)
        except Exception as e:
             print(f"⚠️ 校验请求被限流，假定止损挂单成功: {e}")

        print(
            f"✅ ⚡ [Gate实盘] 下单成功！方向：{side.upper()} | 数量：{order_amount} 张 {gate_symbol}。\n"
            f"🎛️ 目标总杠杆: {leverage:.2f}x | 本次补仓名义价值: ${order_amount * contract_size * actual_fill_price:,.2f}\n"
            f"🎯 本次成交均价: ${actual_fill_price:.4f} (滑点 {slippage:.3%}) | 仓位整体均价: ${position_entry_price:.4f}\n"
            f"🛡️ 兜底止损 (SL): ${sl_price:.4f} (极限亏损 -{abs(sl_pct):.2%})\n"
            f"🚀 追踪止盈: 已移交右脑状态机动态接管"
        )

        # ---------------------------------------------------------
        # 4. 真实订单特征落库（live_trades 表）
        # ---------------------------------------------------------
        db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol}.db")
        try:
            with sqlite3.connect(db_name, timeout=30.0, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                                (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                                 amount REAL, fill_price REAL, slippage REAL,
                                 tp_price REAL, sl_price REAL, margin_used REAL)''')
                # 动态升级旧表
                try: cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
                except sqlite3.OperationalError: pass
                # 👇 新增：动态升级杠杆列
                try: cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
                except sqlite3.OperationalError: pass
                
                # 👇 新增：将 leverage 写入
                cursor.execute('''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                               (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), str(order['id']), side,
                                float(order_amount), actual_fill_price, float(slippage),
                                float(tp_price), float(sl_price), float(risk_margin_usd), float(leverage)))
                conn.commit()
                print(f"   💾 交易记录已同步至本地数据库: {db_name}")
        except Exception as e:
            print(f"   ⚠️ 交易记录入库失败: {e}")

        # ---------------------------------------------------------
        # 5. 异步推送：进场成功通知
        # ---------------------------------------------------------
        if PUSHPLUS_TOKEN:
            try:
                push_title = f"🚀 [Gate实盘] {symbol} 开仓成功！"
                dir_str = "做多 🟢" if side == 'buy' else "做空 🔴"
                push_content = (
                    f"<b>🔹 交易标的：</b>{symbol} ({dir_str})<br>"
                    f"<b>🎯 进场均价：</b>${actual_fill_price:.4f}<br>"
                    f"<b>📦 下单张数：</b>{order_amount} 张<br>"
                    f"<b>🛡️ 初始止损：</b>${sl_price:.4f}<br>"
                    f"<b>💰 保证金占用：</b>{risk_margin_usd:.2f} USDT"
                )
                state.push_queue.put({
                    "token": PUSHPLUS_TOKEN,
                    "title": push_title,
                    "content": push_content,
                    "template": "html"
                })
                # 记录推送时间，防止与同步线程冲突（虽然这是开仓，同步线程只管平仓，但保持习惯）
                base_sym = symbol.split(':')[0].replace('/', '')
                state.recent_pushes[base_sym] = time.time()
            except Exception as e:
                print(f"   ⚠️ 推送任务入队失败: {e}")

        return True, float(risk_margin_usd)

    # 【核心修复：分层处理网络超时幻影丢单，并同步前端日志】
    except RequestTimeout as e:
        if not order_sent:
            msg = f"⚠️ 发单前网络超时，放弃开仓：{e}"
            print("⚠️", msg)
            log_system_alert("⚠️ 网络超时", symbol, msg)
            return False, 0.0
            
        msg1 = f"⚠️ 发送指令时网络请求超时！正在向交易所核实 {gate_symbol} 的真实仓位状态，防止幻影丢单..."
        print("⚠️", msg1)
        log_system_alert("⚠️ 网络超时", symbol, msg1)
        
        try:
            # 强制等待 2 秒让交易所撮合引擎落库
            time.sleep(2)
            verify_positions = exchange_private.fetch_positions([gate_symbol])
            for pos in verify_positions:
                if abs(float(pos.get('contracts', 0))) > 1e-8:
                    msg_ok = f"✅ 虚惊一场：虽然网络超时，但底层的 {side.upper()} 订单已成功被交易所撮合。"
                    print("✅", msg_ok)
                    log_system_alert("✅ 超时抢救", symbol, msg_ok)
                    
                    # 👇 【修复 2：兜底价格计算】因为断网拿不到实际成交价，强制用当前盘口价测算防线
                    actual_tp_move = abs(tp_pct)
                    actual_sl_move = abs(sl_pct)
                    position_entry_price = _position_entry_price(pos) or current_price
                    position_contracts = _position_contracts(pos) or order_amount
                    
                    fallback_sl_price = position_entry_price * (1 - actual_sl_move) if side == 'buy' else position_entry_price * (1 + actual_sl_move)
                    
                    # 立刻挂载条件单，堵住超时裸奔漏洞
                    close_side = "LONG" if side == 'buy' else "SHORT"
                    sl_label = f"sl_tier_{int(time.time() * 1000)}"
                    if safe_post_condition(sl_rule, fallback_sl_price, sl_label, position_contracts, close_side):
                        try:
                            time.sleep(0.5)
                            verify_stops = exchange_private.fetch_open_orders(gate_symbol, params={'stop': True})
                            cancelled_count = cancel_obsolete_gate_close_stops(
                                exchange_private,
                                gate_symbol,
                                sl_label,
                                close_side,
                                verify_stops,
                            )
                            if cancelled_count:
                                print(f"🧹 [超时抢救] 已撤销 {cancelled_count} 张旧止损单，仅保留最新统一防线。")
                        except Exception as stop_cleanup_e:
                            print(f"   ⚠️ [超时抢救] 旧止损清理失败，交由右脑后续校验: {stop_cleanup_e}")
                    
                    # 超时抢救成功后兜底落库
                    db_name = os.path.join(LIVE_DATA_DIR, f"trading_intelligence_{symbol}.db")
                    try:
                        with sqlite3.connect(db_name, timeout=20.0, check_same_thread=False) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''CREATE TABLE IF NOT EXISTS live_trades
                                              (exec_ts TIMESTAMP PRIMARY KEY, order_id TEXT, side TEXT,
                                               amount REAL, fill_price REAL, slippage REAL,
                                               tp_price REAL, sl_price REAL, margin_used REAL)''')
                            try:
                                cursor.execute("ALTER TABLE live_trades ADD COLUMN margin_used REAL")
                            except sqlite3.OperationalError:
                                pass
                            try:
                                cursor.execute("ALTER TABLE live_trades ADD COLUMN leverage REAL")
                            except sqlite3.OperationalError:
                                pass

                            cursor.execute(
                                '''INSERT INTO live_trades (exec_ts, order_id, side, amount, fill_price, slippage, tp_price, sl_price, margin_used, leverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                    'timeout_recovery',
                                    side,
                                    float(order_amount),
                                    current_price,
                                    0.0,
                                    0.0,
                                    0.0,
                                    float(risk_margin_usd),
                                    float(leverage),
                                ),
                            )
                            conn.commit()

                        # 🚨 推送：超时抢救成功通知
                        if PUSHPLUS_TOKEN:
                            try:
                                push_title = f"⚠️ [Gate超时抢救] {symbol} 开仓成功！"
                                dir_str = "做多 🟢" if side == 'buy' else "做空 🔴"
                                push_content = (
                                    f"<b>🚨 预警：</b>网络超时，但已核实开仓成功。"
                                    f"<br><b>🔹 交易标的：</b>{symbol} ({dir_str})"
                                    f"<br><b>🎯 参考进场价：</b>${current_price:.4f}"
                                )
                                state.push_queue.put(
                                    {
                                        "token": PUSHPLUS_TOKEN,
                                        "title": push_title,
                                        "content": push_content,
                                        "template": "html",
                                    }
                                )
                                base_sym = symbol.split(':')[0].replace('/', '')
                                state.recent_pushes[base_sym] = time.time()
                            except Exception:
                                pass
                    except Exception as db_e:
                        print(f"⚠️ 兜底落库失败: {db_e}")
                    return True, float(risk_margin_usd)
            
            msg2 = f"❌ 查单确认：超时导致 {gate_symbol} 订单未到达交易所。"
            print("❌", msg2)
            log_system_alert("❌ 丢单", symbol, msg2)
            return False, 0.0
            
        except Exception as verify_e:
            msg3 = f"❌ 灾难性断网，仓位核实失败，请立刻登录 APP 检查: {verify_e}"
            print("❌", msg3)
            log_system_alert("❌ 断网", symbol, msg3)
            return False, 0.0
            
    except Exception as e:
        msg4 = f"❌ [Gate模拟盘] API 下单遭遇硬性失败: {e}"
        print("❌", msg4)
        log_system_alert("❌ 崩溃", symbol, msg4)
        return False, 0.0
