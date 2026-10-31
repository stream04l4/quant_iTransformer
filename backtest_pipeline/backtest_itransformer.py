import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import polars as pl
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import (
    SYMBOL, PROCESSED_MAIN_DIR, PROCESSED_MAIN_ROOT, MODEL_DIR, TRAIN_DATA_DIR, TEMP_DIR,
    EXPERIMENT_ID, FEE_RATE, HOLD_PERIOD, HOLD_EXPIRY_GRACE_SECONDS, HARD_TP, HARD_SL,
    LEVERAGE_1, LEVERAGE_2, LEVERAGE_3, FORECAST_HORIZON,
    TRAILING_ATR_PERIOD, TRAILING_ATR_MULTI, TRAILING_PHYSICAL_GAP, LIVE_SYMBOL_CONFIGS,
    normalize_symbol
)
from historical_pipeline.train_itransformer import InvertedTransformer


THRESHOLD_KEYS = ["up_top_1", "up_top_5", "up_top_10", "down_top_1", "down_top_5", "down_top_10"]
PHYSICAL_TRAIL_STEP_PHYS = 0.005


def _find_training_matrix_files(horizon):
    pattern = f"{SYMBOL}_Master_Training_Matrix_{horizon}h_*.parquet"
    candidate_dirs = [PROCESSED_MAIN_DIR, PROCESSED_MAIN_ROOT]
    files = []
    for directory in dict.fromkeys(candidate_dirs):
        if directory.exists():
            files.extend(directory.glob(pattern))
    return sorted(set(files))


def _calc_physical_ratchet_phys(max_phys_pnl, hard_tp, physical_gap):
    hard_tp = abs(float(hard_tp))
    if max_phys_pnl < hard_tp:
        return None
    step = PHYSICAL_TRAIL_STEP_PHYS
    gap = max(float(physical_gap), 0.0)
    raw_floor = max_phys_pnl - gap
    stepped_floor = np.floor((raw_floor + 1e-12) / step) * step
    return max(step, stepped_floor)


def _calc_tight_ratchet_phys(max_phys_pnl):
    step = PHYSICAL_TRAIL_STEP_PHYS
    if max_phys_pnl < step:
        return None
    raw_floor = max_phys_pnl - step
    stepped_floor = np.floor((raw_floor + 1e-12) / step) * step
    return max(step, stepped_floor)


def _active_stop_phys(physical_sl_phys, atr_sl_phys, trailing_mode):
    if trailing_mode == "hard_tp" and atr_sl_phys is not None:
        return max(physical_sl_phys, atr_sl_phys)
    return physical_sl_phys


def _parse_args():
    parser = argparse.ArgumentParser(description="iTransformer probabilistic backtest")
    parser.add_argument(
        "--models",
        default="",
        help=(
            "多模型回测配置，格式: EXP_A:0.4,EXP_B:0.6。"
            "不传则沿用 EXPERIMENT_ID 单模型，保障 auto_optimize 调用链不变。"
        ),
    )
    parser.add_argument(
        "--use-live-models",
        action="store_true",
        help="使用 quant_config.config.LIVE_SYMBOL_CONFIGS 中当前 SYMBOL 的实盘模型组合。",
    )
    return parser.parse_args()


def _normalize_model_specs(raw_specs):
    if not raw_specs:
        raw_specs = [{"experiment_id": EXPERIMENT_ID, "weight": 1.0}]

    normalized = []
    for raw in raw_specs:
        exp_id = str(raw.get("experiment_id", EXPERIMENT_ID))
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"模型权重必须为正数: experiment_id={exp_id}, weight={weight}")
        normalized.append({
            "name": str(raw.get("name", exp_id)),
            "experiment_id": exp_id,
            "weight": weight,
        })

    total_weight = sum(item["weight"] for item in normalized)
    for item in normalized:
        item["normalized_weight"] = item["weight"] / total_weight
    return normalized


def _parse_model_specs_arg(models_arg):
    if not models_arg:
        return []

    specs = []
    for chunk in models_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            exp_id, weight = chunk.split(":", 1)
        elif "@" in chunk:
            exp_id, weight = chunk.split("@", 1)
        else:
            exp_id, weight = chunk, "1.0"
        specs.append({"experiment_id": exp_id.strip(), "weight": float(weight)})
    return specs


def _model_specs_from_live_config():
    symbol_key = normalize_symbol(SYMBOL)
    live_cfg = None
    for raw_symbol, cfg in (LIVE_SYMBOL_CONFIGS or {}).items():
        if normalize_symbol(raw_symbol) == symbol_key:
            live_cfg = cfg or {}
            break

    if not live_cfg:
        raise ValueError(f"LIVE_SYMBOL_CONFIGS 中找不到 {SYMBOL} 的实盘模型配置。")
    return live_cfg.get("models") or [
        {"experiment_id": live_cfg.get("experiment_id", EXPERIMENT_ID), "weight": 1.0}
    ]


def _resolve_model_specs(args):
    if args.use_live_models:
        raw_specs = _model_specs_from_live_config()
    else:
        raw_specs = _parse_model_specs_arg(args.models)
    return _normalize_model_specs(raw_specs)


def _load_thresholds(exp_id):
    # [E2] BT_THRESH_SUFFIX: 阈值文件后缀（如 _P0S=严格 val 版），默认空=原文件名
    suffix = os.environ.get("BT_THRESH_SUFFIX", "").strip()
    json_path = MODEL_DIR / f"dynamic_thresholds_{exp_id}{suffix}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"❌ 找不到动态阈值文件: {json_path}")
    with open(json_path, "r") as f:
        config_data = json.load(f)
    return config_data.get("thresholds", config_data), json_path


def _load_model_component(model_cfg, device):
    exp_id = str(model_cfg["experiment_id"])
    model_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {model_path}")

    print(f"📥 加载回测模型: {model_path.name} | weight={model_cfg['normalized_weight']:.3f}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    ckpt_seq_len = checkpoint.get('seq_len')
    thresholds, json_path = _load_thresholds(exp_id)

    model = InvertedTransformer(
        num_variates=len(feature_cols),
        seq_len=ckpt_seq_len,
        d_model=checkpoint.get('d_model'),
        n_heads=checkpoint.get('n_heads'),
        d_ff=checkpoint.get('d_ff'),
        e_layers=checkpoint.get('e_layers'),
        dropout=checkpoint.get('dropout')
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return {
        "name": model_cfg.get("name", exp_id),
        "exp_id": exp_id,
        "weight": float(model_cfg["normalized_weight"]),
        "raw_weight": float(model_cfg["weight"]),
        "model_path": model_path,
        "json_path": json_path,
        "model": model,
        "feature_cols": feature_cols,
        "seq_len": ckpt_seq_len,
        "forecast_horizon": checkpoint.get('forecast_horizon', FORECAST_HORIZON),
        "d_model": checkpoint.get('d_model'),
        "n_heads": checkpoint.get('n_heads'),
        "d_ff": checkpoint.get('d_ff'),
        "e_layers": checkpoint.get('e_layers'),
        "dropout": checkpoint.get('dropout'),
        "thresholds": thresholds,
    }


def _weighted_thresholds(model_components):
    return {
        key: sum(float(component["thresholds"][key]) * component["weight"] for component in model_components)
        for key in THRESHOLD_KEYS
    }


def _validate_ensemble_components(model_components):
    if not model_components:
        raise ValueError("未加载任何模型。")

    horizons = {int(component["forecast_horizon"]) for component in model_components}
    if len(horizons) != 1:
        raise ValueError(f"多模型回测要求 forecast_horizon 一致，当前: {sorted(horizons)}")

    missing_thresholds = [
        component["exp_id"]
        for component in model_components
        if any(key not in component["thresholds"] for key in THRESHOLD_KEYS)
    ]
    if missing_thresholds:
        raise ValueError(f"以下模型动态阈值字段不完整: {missing_thresholds}")


def _generate_weighted_probabilities(df, model_components, split_idx, device):
    sample_count = len(df) - split_idx
    if sample_count <= 0:
        return np.empty((0, 3), dtype=np.float64)

    ensemble_probs = np.zeros((sample_count, 3), dtype=np.float64)
    inference_batch_size = 4096

    with torch.no_grad():
        for component in model_components:
            missing_cols = [c for c in component["feature_cols"] if c not in df.columns]
            if missing_cols:
                raise ValueError(f"模型 {component['exp_id']} 所需特征缺失: {missing_cols}")

            seq_len = int(component["seq_len"])
            warmup_start_idx = split_idx - seq_len + 1
            if warmup_start_idx < 0:
                raise ValueError(f"数据量不足以支撑模型 {component['exp_id']} 的 seq_len={seq_len} 预热。")

            x_raw = df.select(component["feature_cols"]).to_numpy().astype('float32')
            x_test_raw = x_raw[warmup_start_idx:]
            x_tensor = torch.tensor(x_test_raw, dtype=torch.float32, device=device)
            x_windows = x_tensor.unfold(0, seq_len, 1).transpose(1, 2).contiguous()

            component_probs = []
            print(
                f"   🧠 推理模型 {component['exp_id']} | seq_len={seq_len} | "
                f"样本={len(x_windows)} | weight={component['weight']:.3f}"
            )
            for i in range(0, len(x_windows), inference_batch_size):
                batch_x = x_windows[i : i + inference_batch_size]
                with torch.amp.autocast('cuda'):
                    logits = component["model"](batch_x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                component_probs.append(probs)

            component_probs = np.concatenate(component_probs, axis=0) if component_probs else np.empty((0, 3))
            if len(component_probs) != sample_count:
                raise ValueError(
                    f"模型 {component['exp_id']} 推理样本数不一致: "
                    f"{len(component_probs)} != {sample_count}"
                )
            ensemble_probs += component_probs * component["weight"]

    return ensemble_probs


def _compute_atr(highs, lows, closes, period):
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]
    true_ranges = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_closes),
        np.abs(lows - prev_closes),
    ])
    return pd.Series(true_ranges).rolling(period, min_periods=1).mean().to_numpy()


def run_backtest(args=None):
    if args is None:
        args = _parse_args()

    DEVICE = torch.device("cuda")  # 明确 H100 环境

    # ============ N1 模式（Stage N S1 出场消融，2026-10-30 Joel 批准） ============
    # 全部开关默认关闭 = 原行为；仅在提供 N1_SIGNAL_FILE 时启用。
    # N1_SIGNAL_FILE : 逐小时信号文件 (hour_timestamp,p_down,p_up,th_up_top10,th_down_top10)
    # N1_EXIT_MODE   : production(默认) / no_sl(去硬止损,保留棘轮等) / fixed12h(12h 硬到期)
    # N1_WINDOW_START/END : 评估段 ISO 时间戳；N1_GATE 入场门槛(默认 0.40)；N1_LEVERAGE 固定杠杆(默认 1.0)
    # N1_JSON_OUT / N1_TRADES_OUT : 机器可读输出
    n1_signal_file = os.environ.get("N1_SIGNAL_FILE", "").strip()
    n1_mode = bool(n1_signal_file)
    n1_exit_mode = os.environ.get("N1_EXIT_MODE", "production").strip()
    n1_window_start = os.environ.get("N1_WINDOW_START", "").strip()
    n1_window_end = os.environ.get("N1_WINDOW_END", "").strip()
    n1_gate = float(os.environ.get("N1_GATE", "0.40"))
    n1_leverage = float(os.environ.get("N1_LEVERAGE", "1.0"))
    n1_json_out = os.environ.get("N1_JSON_OUT", "").strip()
    n1_trades_out = os.environ.get("N1_TRADES_OUT", "").strip()
    n1_fixed12h = n1_mode and n1_exit_mode == "fixed12h"
    n1_no_sl = n1_mode and n1_exit_mode == "no_sl"
    if n1_mode and n1_exit_mode not in ("production", "no_sl", "fixed12h"):
        raise ValueError(f"❌ N1_EXIT_MODE 非法: {n1_exit_mode}")
    if n1_mode and not (n1_window_start and n1_window_end):
        raise ValueError("❌ N1 模式需要 N1_WINDOW_START / N1_WINDOW_END")
    n1_ws_sec = int(pd.Timestamp(n1_window_start).timestamp()) if n1_mode else 0
    th = None  # 由 N1 信号段（严格 val 阈值）或下方 _weighted_thresholds 赋值

    # ============ Stage O 开关（regime 归因/条件化策略，2026-10-31 Joel 任务书；全部默认关闭=原行为零改动） ============
    # BT_TRADES_OUT  : 非 N1 模式也写逐笔 JSON（N1 schema 超集，含成本拆分/档位字段）
    # BT_LEV_T1/T2/T3: 三档杠杆覆盖（默认 config 2/5/8）；设 1 即全 1x 仿真
    # BT_LONG_ONLY / BT_SHORT_ONLY : 仅允许单方向开新仓（存量仓仍按生产规则出场）
    # BT_REGIME_FILE : 因果 regime 序列 CSV (ts_sec,regime∈{BULL,BEAR,SIDEWAYS,TRANSITION})；
    #                  多头仅 BULL 可开、空头仅 BEAR 可开，其余状态不开新仓（信号时点取最后一个 ts<=T 的状态）
    bt_trades_out = os.environ.get("BT_TRADES_OUT", "").strip()
    bt_lev_t1 = float(os.environ.get("BT_LEV_T1", "0") or 0) or LEVERAGE_1
    bt_lev_t2 = float(os.environ.get("BT_LEV_T2", "0") or 0) or LEVERAGE_2
    bt_lev_t3 = float(os.environ.get("BT_LEV_T3", "0") or 0) or LEVERAGE_3
    bt_long_only = os.environ.get("BT_LONG_ONLY", "0") == "1"
    bt_short_only = os.environ.get("BT_SHORT_ONLY", "0") == "1"
    # BT_REGIME_FLAT=1: regime 切换到不允许持仓方向时强制市价平存量仓（"默认空仓"语义；默认关闭=只禁新仓）
    bt_regime_flat = os.environ.get("BT_REGIME_FLAT", "0") == "1"
    bt_regime_file = os.environ.get("BT_REGIME_FILE", "").strip()
    bt_regime_ts = None
    bt_regime_vals = None
    if bt_regime_file:
        _rdf = pd.read_csv(bt_regime_file)
        bt_regime_ts = _rdf["ts_sec"].to_numpy(dtype=np.int64)
        bt_regime_vals = _rdf["regime"].to_numpy()

    # 1. 优先定位模型权重文件，一切以它保存的元数据为准。
    # 默认仍是 EXPERIMENT_ID 单模型；只有显式 --models / --use-live-models 才进入 ensemble。
    # N1 模式：外部信号文件，跳过模型加载/推理（Stage N S1 消融）。
    if n1_mode:
        model_components = []
        ckpt_horizon = 12  # N1 口径固定 12h（与 P0 一致）
        _n1_matrix_env = os.environ.get("N1_MATRIX", "").strip()
        if not _n1_matrix_env or not Path(_n1_matrix_env).exists():
            print(f"❌ N1 模式需要 N1_MATRIX 指向特征宽表 parquet（当前: {_n1_matrix_env}）")
            return
        data_path = Path(_n1_matrix_env)
        print(f"🚀 N1 模式 | exit={n1_exit_mode} | gate={n1_gate} | lev={n1_leverage} | matrix={data_path.name}")
    else:
        try:
            model_specs = _resolve_model_specs(args)
            model_components = [_load_model_component(spec, DEVICE) for spec in model_specs]
            _validate_ensemble_components(model_components)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return

        primary_component = model_components[0]
        ckpt_horizon = int(primary_component["forecast_horizon"])
        model_count = len(model_components)
        model_stack = ", ".join(
            f"{component['exp_id']}@{component['weight']:.2f}" for component in model_components
        )
        print(f"🚀 启动终极回测 V5 (绝对严谨财务核算 + 动态杠杆)...")
        print(f"🧩 模型模式: {'Ensemble' if model_count > 1 else 'Single'} x{model_count} | {model_stack}")

        # 基于真实 horizon 精准匹配特征宽表
        data_files = _find_training_matrix_files(ckpt_horizon)
        if not data_files:
            print(f"❌ 找不到对应 Horizon ({ckpt_horizon}h) 的数据文件！")
            return
        data_path = data_files[-1]

    # === 1. 独立加载特征空间与执行空间 ===
    df = pl.read_parquet(data_path)
    timestamps = df.select("hour_timestamp").to_series().to_numpy()

    if n1_mode:
        # N1：评估段 = [N1_WINDOW_START, N1_WINDOW_END)（无推理，无需预热）
        _ts64 = timestamps.astype("datetime64[s]").astype(np.int64)
        _we_sec = int(pd.Timestamp(n1_window_end).timestamp())
        split_idx = int(np.searchsorted(_ts64, n1_ws_sec, side="left"))
        end_idx = int(np.searchsorted(_ts64, _we_sec, side="left"))
        if end_idx <= split_idx:
            print("❌ N1 窗口内没有矩阵行！")
            return
        test_timestamps = timestamps[split_idx:end_idx]
        valid_timestamps = test_timestamps.tolist()

        sig = pd.read_csv(n1_signal_file)
        sig_ts = sig["hour_timestamp"].astype("datetime64[s]").astype(np.int64).to_numpy()
        matrix_ts = _ts64[split_idx:end_idx]
        if len(sig_ts) != len(matrix_ts) or not np.array_equal(sig_ts, matrix_ts):
            raise ValueError(
                f"❌ N1 信号文件与矩阵行不对齐: signal={len(sig_ts)} matrix={len(matrix_ts)}"
            )
        p_down_arr = sig["p_down"].to_numpy().astype(float)
        p_up_arr = sig["p_up"].to_numpy().astype(float)
        p_flat_arr = np.clip(1.0 - p_up_arr - p_down_arr, 0.0, 1.0)
        predictions_probs = np.stack([p_down_arr, p_flat_arr, p_up_arr], axis=1)
        _thu = float(sig["th_up_top10"].iloc[0])
        _thd = float(sig["th_down_top10"].iloc[0])
        th = {"up_top_1": _thu, "up_top_5": _thu, "up_top_10": _thu,
              "down_top_1": _thd, "down_top_5": _thd, "down_top_10": _thd}
        print(f"⚙️ N1 信号加载: {len(predictions_probs)} 行 | 严格 val 阈值 up_top10={_thu:.4f} down_top10={_thd:.4f}")
    else:
        # [E2 严格窗口] (2026-11-03, Joel 预批): env 覆盖评估窗口, 默认不启用=原行为零改动
        #   BT_TEST_END_TS  : 数据在窗口末端物理截断 (hour_timestamp < end)
        #   BT_TEST_START_TS: test 起点, split=首个 ts>=start 的行号 (替代 80% 比例切分)
        import datetime as _dt_mod
        _bt_end = os.environ.get("BT_TEST_END_TS", "").strip()
        if _bt_end:
            _end_lit = pl.lit(_dt_mod.datetime.fromisoformat(_bt_end)).cast(df["hour_timestamp"].dtype)
            _h0 = df.height
            df = df.filter(df["hour_timestamp"] < _end_lit)
            timestamps = df.select("hour_timestamp").to_series().to_numpy()
            print(f"🔒 [E2窗口] 数据截断于 {_bt_end}: {_h0} -> {df.height} 行")
        _bt_start = os.environ.get("BT_TEST_START_TS", "").strip()
        if _bt_start:
            _start_lit = pl.lit(_dt_mod.datetime.fromisoformat(_bt_start)).cast(df["hour_timestamp"].dtype)
            split_idx = df.filter(df["hour_timestamp"] < _start_lit).height
            print(f"🔒 [E2窗口] test 起点 {_bt_start}: split_idx={split_idx} (test 侧 {df.height - split_idx} 行)")
        else:
            split_idx = int(df.height * 0.8) + ckpt_horizon
        max_seq_len = max(int(component["seq_len"]) for component in model_components)
        warmup_start_idx = split_idx - max_seq_len + 1

        if warmup_start_idx < 0:
            print(f"❌ 数据量不足以支撑最大 seq_len={max_seq_len} 的预热！")
            return

        test_timestamps = timestamps[split_idx:]
        valid_timestamps = test_timestamps.tolist()

        print("\n⚙️ 正在 H100 上生成概率矩阵 (全显存 Zero-Copy 极速推理)...")
        predictions_probs = _generate_weighted_probabilities(df, model_components, split_idx, DEVICE)

    # 极速加载 1h 和 1m 的真实物理价格 (执行空间)
    raw_kline_dir = TEMP_DIR / "klines" / SYMBOL
    try:
        df_1h = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1h.parquet")
        df_1m = pl.read_parquet(raw_kline_dir / f"{SYMBOL}_raw_1m.parquet")
    except Exception as e:
        print(f"❌ 找不到原始 K 线，请先运行 build_main_features.py 完成下载！错误: {e}")
        return

    # 全部强制转换为 int64 的秒级 Unix 绝对时间，防止 Numpy 精度错乱
    ts_1h = df_1h["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    opens_1h = df_1h["open"].to_numpy()
    closes_1h = df_1h["close"].to_numpy()

    ts_1m = df_1m["timestamp"].to_numpy().astype('datetime64[s]').astype(np.int64)
    highs_1m = df_1m["high"].to_numpy()
    lows_1m = df_1m["low"].to_numpy()
    closes_1m = df_1m["close"].to_numpy()
    atr_1m = _compute_atr(highs_1m, lows_1m, closes_1m, TRAILING_ATR_PERIOD)

    valid_timestamps_sec = np.array(valid_timestamps).astype('datetime64[s]').astype(np.int64)

    # ==========================================
    # 🏦 基于 1h 决策 + 1m 显微镜追踪的双轨状态机
    # ==========================================
    # 🚨 防呆机制 3：多模型时动态分位数阈值跟随概率 ensemble 做同权重融合。
    # （N1 模式的 th 已在信号加载段由严格 val 阈值给出）
    if not n1_mode:
        th = _weighted_thresholds(model_components)
    if th is None:
        raise RuntimeError("❌ 阈值未就绪（N1 信号段 / _weighted_thresholds 均未赋值）")

    print("📥 已成功加载动态分位数阈值。")

    equity = 1.0
    equity_curve = []
    benchmark_prices = []

    current_pos = 0
    current_tier = 0         # 👈 新增：记录当前阶梯水位
    hold_timer_mins = 0
    entry_price = 0.0
    trade_start_equity = 1.0
    current_leverage = 0.0
    entry_time = None

    # 🚨 引入严格的财务账本变量
    cumulative_funding = 0.0
    open_cost = 0.0

    max_unrealized_phys_pnl = 0.0
    min_adverse_phys_pnl = 0.0   # N1: 全程最大不利波动（MAE，物理口径）
    n1_trades = []               # N1: 逐笔记录（仅 N1 模式写出）
    entry_tier = 0               # Stage O: 开仓档位（首开）
    max_tier = 0                 # Stage O: 持仓期间达到的最高档位
    trailing_active = False
    trailing_mode = None
    physical_sl_phys = -abs(HARD_SL)
    atr_sl_phys = None

    trade_count = 0
    win_count = 0
    liquidated = False

    long_trades = 0
    short_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0

    print("\n⚔️ 正在执行工业级撮合 (1h 决策开仓 + 1m 盘中悲观碰撞)...")

    SLIPPAGE = 0.0002
    HOURLY_FUNDING_COST = 0.0001 / 8

    exit_reasons = {
        "REVERSAL_EXIT": 0,
        "HARD_SL_EXIT": 0,
        "TRAILING_EXIT": 0,
        "TIME_LIMIT_EXIT": 0,
        "LIQUIDATION_EXIT": 0,
        "REGIME_FLAT_EXIT": 0,
    }
    pnl_list = []

    for i in range(len(predictions_probs)):
        if i > 0 and i % 1000 == 0:
            print(f"   ⏳ 已撮合 {i} / {len(predictions_probs)} 小时...")

        if liquidated:
            equity_curve.append(0)
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            continue

        p_down, p_flat, p_up = predictions_probs[i]
        signal_time = valid_timestamps_sec[i]
        exec_time = signal_time + 3600
        exec_idx_1h = np.searchsorted(ts_1h, exec_time)

        if exec_idx_1h < len(closes_1h):
            benchmark_prices.append(closes_1h[exec_idx_1h])
        else:
            if benchmark_prices: benchmark_prices.append(benchmark_prices[-1])
            break

        # Stage O: regime 强制平仓（BT_REGIME_FLAT=1；默认关闭=原行为）。
        # 语义：regime 切换到不允许该持仓方向（多=BULL / 空=BEAR）时，执行时点开盘价市价全平。
        # 完整对账：gross - 开仓摩擦(已扣) - 平仓摩擦 - 累计资金费，与生产平仓公式一致。
        if bt_regime_flat and bt_regime_ts is not None and current_pos != 0 and not liquidated:
            _ri = int(np.searchsorted(bt_regime_ts, signal_time, side="right")) - 1
            _r = bt_regime_vals[_ri] if _ri >= 0 else "SIDEWAYS"
            _allowed = (_r == "BULL") if current_pos == 1 else (_r == "BEAR")
            if not _allowed and exec_idx_1h < len(opens_1h):
                _fc_px = opens_1h[exec_idx_1h]
                _fc_ret_phys = (_fc_px / entry_price - 1.0) if current_pos == 1 else (1.0 - _fc_px / entry_price)
                _fc_close_cost = trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                _fc_gross = trade_start_equity * _fc_ret_phys * current_leverage
                equity = trade_start_equity + _fc_gross - open_cost - _fc_close_cost - cumulative_funding
                _fc_pnl = equity - trade_start_equity
                if _fc_pnl > 0:
                    win_count += 1; gross_profit += _fc_pnl
                else:
                    gross_loss += abs(_fc_pnl)
                trade_count += 1
                pnl_list.append(_fc_pnl / trade_start_equity)
                exit_reasons["REGIME_FLAT_EXIT"] += 1
                if n1_mode or bt_trades_out:
                    n1_trades.append({
                        "entry_time": int(entry_time), "exit_time": int(signal_time + 3600),
                        "dir": int(current_pos), "leverage": float(current_leverage),
                        "entry_price": float(entry_price),
                        "exit_ret_phys": float(_fc_ret_phys),
                        "pnl_frac": float(_fc_pnl / trade_start_equity),
                        "mfe_phys": float(max_unrealized_phys_pnl),
                        "mae_phys": float(min_adverse_phys_pnl),
                        "hold_minutes": int((signal_time + 3600 - entry_time) / 60),
                        "reason": "REGIME_FLAT_EXIT",
                        "entry_tier": int(entry_tier), "max_tier": int(max_tier), "start_equity": float(trade_start_equity),
                        "open_cost": float(open_cost), "close_cost": float(_fc_close_cost),
                        "funding": float(cumulative_funding),
                        "gross_pnl": float(_fc_gross),
                    })
                if current_pos == 1:
                    pass  # long_trades/short_trades 已在开仓时计数，平仓不重复
                current_pos = 0
                current_tier = 0
                current_leverage = 0.0
                trailing_active = False
                trailing_mode = None
                atr_sl_phys = None

        # ------------------------------------------
        # 1. 寻找开仓与【阶梯升档】机会 (1h 级别)
        # ------------------------------------------
        target_pos = 0
        target_tier = 0
        target_leverage = 0.0

        if n1_mode:
            # N1：仅在 12h 块起点（相对窗口起点的 12h 网格）允许开仓；固定 gate，单档 N1_LEVERAGE。
            # 与 P0 口径一致；同向信号刷新逻辑对 fixed12h 在情景 B 关闭（见下）。
            if (signal_time - n1_ws_sec) % (12 * 3600) == 0:
                if p_up >= n1_gate and p_up > p_down:
                    target_pos = 1; target_tier = 1; target_leverage = n1_leverage
                elif p_down >= n1_gate and p_down > p_up:
                    target_pos = -1; target_tier = 1; target_leverage = n1_leverage
        elif p_up > th["up_top_1"]:      target_pos = 1; target_tier = 3; target_leverage = bt_lev_t3
        elif p_up > th["up_top_5"]:    target_pos = 1; target_tier = 2; target_leverage = bt_lev_t2
        elif p_up > th["up_top_10"]:   target_pos = 1; target_tier = 1; target_leverage = bt_lev_t1
        elif p_down > th["down_top_1"]:  target_pos = -1; target_tier = 3; target_leverage = bt_lev_t3
        elif p_down > th["down_top_5"]:  target_pos = -1; target_tier = 2; target_leverage = bt_lev_t2
        elif p_down > th["down_top_10"]: target_pos = -1; target_tier = 1; target_leverage = bt_lev_t1

        # Stage O: 方向/regime 门控（默认关闭=原行为）
        if target_pos != 0 and not n1_mode:
            if (bt_long_only and target_pos == -1) or (bt_short_only and target_pos == 1):
                target_pos = 0
            elif bt_regime_ts is not None:
                _ri = int(np.searchsorted(bt_regime_ts, signal_time, side="right")) - 1
                _r = bt_regime_vals[_ri] if _ri >= 0 else "SIDEWAYS"
                if (target_pos == 1 and _r != "BULL") or (target_pos == -1 and _r != "BEAR"):
                    target_pos = 0

        if target_pos != 0 and exec_idx_1h < len(opens_1h):
            current_hour_open = opens_1h[exec_idx_1h]

            # 💡 情景 A: 绝对空仓，首次建仓
            if current_pos == 0:
                current_pos = target_pos
                current_tier = target_tier
                current_leverage = target_leverage
                hold_timer_mins = HOLD_PERIOD * 60

                entry_price = current_hour_open
                entry_time = ts_1h[exec_idx_1h]

                trade_start_equity = equity
                open_cost = trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                equity = trade_start_equity - open_cost
                cumulative_funding = 0.0

                if target_pos == 1: long_trades += 1
                if target_pos == -1: short_trades += 1

                max_unrealized_phys_pnl = 0.0
                min_adverse_phys_pnl = 0.0
                entry_tier = target_tier
                max_tier = target_tier
                _n1_reason = ""
                trailing_active = False
                trailing_mode = None
                physical_sl_phys = -1.0 if n1_no_sl else -abs(HARD_SL)
                atr_sl_phys = None

            # 💡 情景 B: 同 Tier 信号才刷新持仓计时；更高 Tier 额外补仓并重置计时。
            # 低 Tier 信号代表概率走弱，不能给高 Tier 持仓续命。
            # N1 fixed12h：无刷新/补仓（持仓固定 12h，与 P0 口径一致）。
            elif current_pos == target_pos and not n1_fixed12h:
                hold_near_expiry = hold_timer_mins * 60 <= HOLD_EXPIRY_GRACE_SECONDS
                if target_tier > current_tier or (target_tier == current_tier and not hold_near_expiry):
                    hold_timer_mins = HOLD_PERIOD * 60
                    if trailing_mode == "tight":
                        trailing_active = False
                        trailing_mode = None
                        atr_sl_phys = None

                if target_tier > current_tier:
                    # [核心修正]: 必须记住加仓前的"绝对止损价"，防止均价上移导致瞬间爆仓
                    if current_pos == 1:
                        absolute_sl_price = entry_price * (1 + physical_sl_phys)
                        absolute_atr_sl_price = (
                            entry_price * (1 + atr_sl_phys)
                            if atr_sl_phys is not None
                            else None
                        )
                    else:
                        absolute_sl_price = entry_price * (1 - physical_sl_phys)
                        absolute_atr_sl_price = (
                            entry_price * (1 - atr_sl_phys)
                            if atr_sl_phys is not None
                            else None
                        )

                    # 【对齐实盘 OMS】：Tier 杠杆表示目标总杠杆，仅补足当前仓位与目标仓位之间的差额
                    old_notional = trade_start_equity * current_leverage
                    target_notional = trade_start_equity * target_leverage
                    added_notional = max(0.0, target_notional - old_notional)
                    if added_notional <= 1e-12:
                        current_tier = target_tier
                    else:
                        old_size = old_notional / entry_price
                        new_size = added_notional / current_hour_open
                        total_size = old_size + new_size

                        # 重新计算加仓后的加权持仓均价
                        avg_entry_price = target_notional / total_size

                        # 扣除新增仓位的摩擦成本
                        added_open_cost = added_notional * (FEE_RATE + SLIPPAGE)
                        open_cost += added_open_cost
                        equity -= added_open_cost

                        # 更新全局状态机
                        entry_price = avg_entry_price
                        current_leverage = target_leverage
                        current_tier = target_tier
                        max_tier = max(max_tier, target_tier)

                        # 重新锚定防线：用旧的绝对止损价反推距离新均价的百分比距离
                        if current_pos == 1:
                            new_physical_sl = (absolute_sl_price - entry_price) / entry_price
                            if absolute_atr_sl_price is not None:
                                atr_sl_phys = (absolute_atr_sl_price - entry_price) / entry_price
                            max_unrealized_phys_pnl = (current_hour_open - entry_price) / entry_price
                        else:
                            new_physical_sl = (entry_price - absolute_sl_price) / entry_price
                            if absolute_atr_sl_price is not None:
                                atr_sl_phys = (entry_price - absolute_atr_sl_price) / entry_price
                            max_unrealized_phys_pnl = (entry_price - current_hour_open) / entry_price

                        # 兜底：如果均价恶化太多导致防线变得离谱，用硬止损兜底
                        physical_sl_phys = max(new_physical_sl, -abs(HARD_SL))

        # ------------------------------------------
        # 2. 1分钟级别显微镜追踪持仓
        # ------------------------------------------
        if current_pos != 0:
            start_1m_idx = np.searchsorted(ts_1m, entry_time)
            chunk_end_time = exec_time + 3600
            end_1m_idx = np.searchsorted(ts_1m, chunk_end_time)

            close_position = False
            actual_exit_pct = 0.0
            close_pnl_pct = 0.0
            close_return = 0.0

            is_reversal = False
            if n1_fixed12h:
                pass  # N1 fixed12h：无反向退出（仅 12h 到期）
            elif current_pos == 1 and p_down > th["down_top_10"]: is_reversal = True
            elif current_pos == -1 and p_up > th["up_top_10"]: is_reversal = True
            is_favorable_signal = False
            if current_pos == 1 and p_up > th["up_top_10"]: is_favorable_signal = True
            elif current_pos == -1 and p_down > th["down_top_10"]: is_favorable_signal = True

            tight_cancelled_this_hour = False
            if trailing_mode == "tight" and is_favorable_signal:
                trailing_active = False
                trailing_mode = None
                atr_sl_phys = None
                hold_timer_mins = HOLD_PERIOD * 60
                tight_cancelled_this_hour = True

            if is_reversal and not tight_cancelled_this_hour:
                current_hour_open = opens_1h[exec_idx_1h]
                current_phys_profit = (current_hour_open - entry_price) / entry_price if current_pos == 1 else (entry_price - current_hour_open) / entry_price

                if current_phys_profit < 0.005 and not trailing_active:
                    close_position = True
                    actual_exit_pct = current_phys_profit * current_leverage
                    _n1_reason = "REVERSAL_EXIT"
                    exit_reasons["REVERSAL_EXIT"] += 1
                else:
                    trailing_active = True
                    trailing_mode = "tight"
                    atr_sl_phys = None
                    physical_sl_phys = max(physical_sl_phys, PHYSICAL_TRAIL_STEP_PHYS)
                    tight_floor = _calc_tight_ratchet_phys(max_unrealized_phys_pnl)
                    if tight_floor is not None:
                        physical_sl_phys = max(physical_sl_phys, tight_floor)

            if not close_position:
                for m_idx in range(start_1m_idx, end_1m_idx):
                    if m_idx >= len(closes_1m): break

                    m_high = highs_1m[m_idx]
                    m_low = lows_1m[m_idx]
                    m_close = closes_1m[m_idx]

                    if current_pos == 1:
                        worst_intra = (m_low - entry_price) / entry_price
                        best_intra = (m_high - entry_price) / entry_price
                        close_return = (m_close - entry_price) / entry_price
                    else:
                        worst_intra = (entry_price - m_high) / entry_price
                        best_intra = (entry_price - m_low) / entry_price
                        close_return = (entry_price - m_close) / entry_price

                    if n1_fixed12h:
                        # N1 fixed12h：无 TP/SL/棘轮/反向，仅计时；到期无条件平仓
                        # （对齐 P0 固定 12h 口径：入场=信号 bar 次根 open，出场=12h 后的 1m close）
                        hold_timer_mins -= 1
                        if best_intra > max_unrealized_phys_pnl:
                            max_unrealized_phys_pnl = best_intra
                        if worst_intra < min_adverse_phys_pnl:
                            min_adverse_phys_pnl = worst_intra
                        if hold_timer_mins <= 0:
                            close_position = True
                            actual_exit_pct = close_return * current_leverage
                            _n1_reason = "TIME_LIMIT_EXIT"
                            exit_reasons["TIME_LIMIT_EXIT"] += 1
                            break
                        continue

                    atr_price = atr_1m[m_idx] if m_idx < len(atr_1m) else 0.0
                    trailing_retrace_phys = (
                        TRAILING_ATR_MULTI * atr_price / entry_price
                        if entry_price > 0 and atr_price > 0
                        else 0.0
                    )

                    hold_timer_mins -= 1

                    worst_pnl_pct = current_leverage * worst_intra
                    # 爆仓判定 (资金毁灭，使用带上双边滑点预期最恶劣的情况)
                    if trade_start_equity - open_cost - (trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)) + (trade_start_equity * worst_pnl_pct) <= 0:
                        liquidated = True
                        break

                    active_stop_phys = _active_stop_phys(physical_sl_phys, atr_sl_phys, trailing_mode)
                    if worst_intra <= active_stop_phys:
                        close_position = True
                        actual_exit_pct = active_stop_phys * current_leverage
                        _n1_reason = "TRAILING_EXIT" if trailing_active else "HARD_SL_EXIT"
                        if trailing_active: exit_reasons["TRAILING_EXIT"] += 1
                        else: exit_reasons["HARD_SL_EXIT"] += 1
                        break

                    if best_intra > max_unrealized_phys_pnl:
                        max_unrealized_phys_pnl = best_intra
                    if worst_intra < min_adverse_phys_pnl:
                        min_adverse_phys_pnl = worst_intra

                    if max_unrealized_phys_pnl >= abs(HARD_TP):
                        if trailing_mode != "tight":
                            trailing_active = True
                            trailing_mode = "hard_tp"
                            ratchet_floor = _calc_physical_ratchet_phys(
                                max_unrealized_phys_pnl,
                                HARD_TP,
                                TRAILING_PHYSICAL_GAP,
                            )
                            if ratchet_floor is not None:
                                physical_sl_phys = max(physical_sl_phys, ratchet_floor)
                    if trailing_mode == "tight":
                        tight_floor = _calc_tight_ratchet_phys(max_unrealized_phys_pnl)
                        if tight_floor is not None:
                            physical_sl_phys = max(physical_sl_phys, tight_floor)

                    if hold_timer_mins <= 0 and not tight_cancelled_this_hour:
                        if close_return < 0.005 and not trailing_active:
                            close_position = True
                            actual_exit_pct = close_return * current_leverage
                            _n1_reason = "TIME_LIMIT_EXIT"
                            exit_reasons["TIME_LIMIT_EXIT"] += 1
                            break
                        else:
                            trailing_active = True
                            trailing_mode = "tight"
                            atr_sl_phys = None
                            physical_sl_phys = max(physical_sl_phys, PHYSICAL_TRAIL_STEP_PHYS)
                            tight_floor = _calc_tight_ratchet_phys(max_unrealized_phys_pnl)
                            if tight_floor is not None:
                                physical_sl_phys = max(physical_sl_phys, tight_floor)

                    if trailing_mode == "hard_tp" and trailing_retrace_phys > 0:
                        atr_sl_phys = max_unrealized_phys_pnl - trailing_retrace_phys

                    active_stop_phys = _active_stop_phys(physical_sl_phys, atr_sl_phys, trailing_mode)
                    if worst_intra <= active_stop_phys:
                        close_position = True
                        actual_exit_pct = active_stop_phys * current_leverage
                        _n1_reason = "TRAILING_EXIT" if trailing_active else "HARD_SL_EXIT"
                        if trailing_active: exit_reasons["TRAILING_EXIT"] += 1
                        else: exit_reasons["HARD_SL_EXIT"] += 1
                        break

                close_pnl_pct = close_return * current_leverage

            if liquidated:
                _n1_dir = current_pos
                _n1_lev = current_leverage
                equity = 0
                current_pos = 0
                current_tier = 0
                equity_curve.append(0)
                print(f"💀 盘中分钟级扎针爆仓！发生于 {signal_time}")
                trade_count += 1
                gross_loss += trade_start_equity
                exit_reasons["LIQUIDATION_EXIT"] += 1
                pnl_list.append(-1.0)
                if n1_mode or bt_trades_out:
                    n1_trades.append({
                        "entry_time": int(entry_time), "exit_time": int(chunk_end_time),
                        "dir": int(_n1_dir), "leverage": float(_n1_lev),
                        "entry_price": float(entry_price), "exit_ret_phys": -1.0,
                        "pnl_frac": -1.0, "mfe_phys": float(max_unrealized_phys_pnl),
                        "mae_phys": float(min_adverse_phys_pnl),
                        "hold_minutes": int(HOLD_PERIOD * 60 - hold_timer_mins),
                        "reason": "LIQUIDATION_EXIT",
                        "entry_tier": int(entry_tier), "max_tier": int(max_tier), "start_equity": float(trade_start_equity),
                        "open_cost": float(open_cost), "close_cost": 0.0,
                        "funding": float(cumulative_funding),
                        "gross_pnl": -float(trade_start_equity),
                    })
                trailing_active = False
                trailing_mode = None
                atr_sl_phys = None
                continue

            if close_position:
                # 🚨 财务修复 2：统一严谨结算（本金 + 收益 - 开仓摩擦 - 平仓摩擦 - 累计资金费）
                close_cost = trade_start_equity * current_leverage * (FEE_RATE + SLIPPAGE)
                final_pnl = trade_start_equity * actual_exit_pct

                equity = trade_start_equity + final_pnl - open_cost - close_cost - cumulative_funding

                trade_pnl = equity - trade_start_equity
                if trade_pnl > 0:
                    win_count += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)

                _n1_dir = current_pos
                _n1_lev = current_leverage
                trade_count += 1
                current_pos = 0
                current_tier = 0
                current_leverage = 0.0
                trailing_active = False
                trailing_mode = None
                atr_sl_phys = None

                pnl_list.append(trade_pnl / trade_start_equity)
                if n1_mode or bt_trades_out:
                    n1_trades.append({
                        "entry_time": int(entry_time),
                        "exit_time": int(chunk_end_time),
                        "dir": int(_n1_dir),
                        "leverage": float(_n1_lev),
                        "entry_price": float(entry_price),
                        "exit_ret_phys": float(actual_exit_pct / _n1_lev) if _n1_lev else 0.0,
                        "pnl_frac": float(trade_pnl / trade_start_equity),
                        "mfe_phys": float(max_unrealized_phys_pnl),
                        "mae_phys": float(min_adverse_phys_pnl),
                        "hold_minutes": int(HOLD_PERIOD * 60 - hold_timer_mins),
                        "reason": _n1_reason,
                        "entry_tier": int(entry_tier), "max_tier": int(max_tier), "start_equity": float(trade_start_equity),
                        "open_cost": float(open_cost), "close_cost": float(close_cost),
                        "funding": float(cumulative_funding),
                        "gross_pnl": float(final_pnl),
                    })
            else:
                # 🚨 财务修复 3：跨小时浮动结算，同样严格遵守完整公式，防止资金幽灵恢复
                # N1：口径与 P0/阶段 L 一致（仅 0.07% 单边成本，不计 funding），保证三模式归因干净
                if not n1_mode:
                    cumulative_funding += trade_start_equity * current_leverage * HOURLY_FUNDING_COST
                floating_pnl = trade_start_equity * close_pnl_pct
                equity = trade_start_equity + floating_pnl - open_cost - cumulative_funding
                entry_time = chunk_end_time

        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    benchmark_prices = np.array(benchmark_prices)

    # 统一所有逐小时序列长度，避免临近数据尾部或提前终止时绘图/指标错位。
    curve_len = min(len(equity_curve), len(benchmark_prices), len(valid_timestamps))
    equity_curve = equity_curve[:curve_len]
    benchmark_prices = benchmark_prices[:curve_len]
    valid_timestamps = valid_timestamps[:curve_len]

    if curve_len == 0:
        print("❌ 没有可用于统计的有效回测样本。")
        return

    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    ls_ratio = (long_trades / short_trades) if short_trades > 0 else float('inf')

    total_return = equity_curve[-1] - 1.0
    years = curve_len / 8760
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = (annualized_return / (abs(max_drawdown)/100)) if max_drawdown < 0 else float('inf')

    if len(benchmark_prices) > 0: benchmark_curve = benchmark_prices / benchmark_prices[0]
    else: benchmark_curve = np.ones(curve_len)

    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0.0

    returns_series = pd.Series(equity_curve).pct_change().fillna(0).to_numpy()
    returns_series[0] = 0
    sharpe = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(8760) if np.std(returns_series) != 0 else 0.0

    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([p for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0
    expectancy = float(np.mean(pnl_list)) if pnl_list else 0.0
    recovery_factor = total_return / (abs(max_drawdown)/100) if max_drawdown != 0 else 0

    # ============ N1 模式：写出机器可读结果并提前返回（跳过绘图） ============
    if n1_mode:
        _n1_pf = float(profit_factor) if np.isfinite(profit_factor) else None
        n1_summary = {
            "exit_mode": n1_exit_mode, "gate": n1_gate, "leverage": n1_leverage,
            "window": [n1_window_start, n1_window_end],
            "total_return": float(total_return),
            "n_trades": int(trade_count),
            "win_rate": float(win_rate) / 100.0,
            "profit_factor": _n1_pf,
            "expectancy": float(expectancy),
            "max_dd": float(max_drawdown) / 100.0,
            "long_trades": int(long_trades), "short_trades": int(short_trades),
            "exit_reasons": {k: int(v) for k, v in exit_reasons.items()},
            "mfe_median": float(np.median([t["mfe_phys"] for t in n1_trades])) if n1_trades else None,
            "mae_median": float(np.median([t["mae_phys"] for t in n1_trades])) if n1_trades else None,
            "trades": n1_trades,
        }
        if n1_json_out:
            with open(n1_json_out, "w") as f:
                json.dump(n1_summary, f, ensure_ascii=False, indent=1)
        if n1_trades_out:
            if n1_trades:
                pd.DataFrame(n1_trades).to_csv(n1_trades_out, index=False)
            else:
                with open(n1_trades_out, "w") as f:
                    f.write("entry_time,exit_time,dir,leverage,entry_price,exit_ret_phys,pnl_frac,mfe_phys,mae_phys,hold_minutes,reason\n")
        print(f"💾 N1 结果已写出: {n1_json_out or '(无 json)'} / {n1_trades_out or '(无 csv)'}")
        return

    # ============ Stage O: 非 N1 模式逐笔 JSON（不跳过绘图，只追加机器可读输出；默认关闭） ============
    if bt_trades_out:
        _o_pf = float(profit_factor) if np.isfinite(profit_factor) else None
        o_summary = {
            "window": [os.environ.get("BT_TEST_START_TS"), os.environ.get("BT_TEST_END_TS")],
            "leverage_tiers": [bt_lev_t1, bt_lev_t2, bt_lev_t3],
            "long_only": bt_long_only, "short_only": bt_short_only,
            "regime_file": bt_regime_file or None,
            "total_return": float(total_return),
            "n_trades": int(trade_count),
            "win_rate": float(win_rate) / 100.0,
            "profit_factor": _o_pf,
            "expectancy": float(expectancy),
            "max_dd": float(max_drawdown) / 100.0,
            "sharpe": float(sharpe),
            "long_trades": int(long_trades), "short_trades": int(short_trades),
            "exit_reasons": {k: int(v) for k, v in exit_reasons.items()},
            "mfe_median": float(np.median([t["mfe_phys"] for t in n1_trades])) if n1_trades else None,
            "mae_median": float(np.median([t["mae_phys"] for t in n1_trades])) if n1_trades else None,
            "equity_curve": [round(float(x), 6) for x in equity_curve],
            "trades": n1_trades,
        }
        with open(bt_trades_out, "w") as f:
            json.dump(o_summary, f, ensure_ascii=False, indent=1)
        print(f"💾 Stage O trades 已写出: {bt_trades_out} ({len(n1_trades)} 笔)")

    print("\n" + "="*50)
    print(f" 📜 TimesFM 引擎终极回测审计报告 | {SYMBOL}")
    print("="*50)

    print(f"【1. 基础战况】")
    print(f" ▪ 总回测时长 : {curve_len} 小时")
    print(f" ▪ 撮合交易数 : {trade_count} 笔 (多:{long_trades} | 空:{short_trades})")
    print(f" ▪ 最终净值   : {equity_curve[-1]:.4f}")

    print(f"\n【2. 核心绩效】")
    print(f" ▪ 胜率 (WR)  : {win_rate:.2f}%")
    print(f" ▪ 盈亏比 (PF): {profit_factor:.2f}")
    print(f" ▪ 夏普比率   : {sharpe:.2f}")
    print(f" ▪ 卡玛比率   : {calmar:.2f}")
    print(f" ▪ 最大回撤   : {max_drawdown:+.2f}%")
    print(f" ▪ 恢复因子   : {recovery_factor:.2f}")
    print(f" ▪ 交易期望值 : {expectancy * 100:+.3f}% (每次发车平均净赚本金比例)")

    print(f"\n【3. 行为诊断 (出场分布)】")
    for reason, count in exit_reasons.items():
        ratio = (count / trade_count * 100) if trade_count > 0 else 0
        print(f" ▪ {reason:<16} : {count} 次 ({ratio:.1f}%)")

    if liquidated:
        print(f"\n💀 警报：该参数组合在回测期间触发了强制平仓(爆仓)！")
    print("="*50 + "\n")

    plt.figure(figsize=(14, 7))
    plt.plot(valid_timestamps, benchmark_curve, label='BTC Buy & Hold', color='gray', alpha=0.5)
    plt.plot(valid_timestamps, equity_curve, label='iTransformer Probabilistic Engine', color='#00ff88', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    plt.title('True OOS Equity: 3-Class Probabilistic Strategy with Dynamic Leverage', fontsize=16, color='white')

    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#121212')
    plt.ylabel('Equity Multiplier (1.0 = 100%)', color='white')
    plt.tick_params(colors='white')
    plt.legend(facecolor='#1e1e1e', labelcolor='white')
    plt.grid(True, alpha=0.1, color='white')

    plot_path = MODEL_DIR / f"backtest_probabilistic_{os.environ.get('BT_PLOT_TAG', EXPERIMENT_ID)}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    print(f"\n📈 最终资金曲线已生成: {plot_path}")
    plt.close('all')

if __name__ == "__main__":
    run_backtest()
