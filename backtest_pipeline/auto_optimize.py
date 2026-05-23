import os
import subprocess
import re
import json
import pandas as pd
import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import sys
import argparse
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import TRAIN_DATA_DIR, MODEL_DIR, HISTORICAL_DIR, BACKTEST_DIR, SYMBOL

ALPHA_SEARCH_SPACE = {
    'FORECAST_HORIZON': [12],
    'SEQ_LEN': [336],
    'CLASS_THRESHOLD': [0.015],
    'E_LAYERS': [1],
    'D_MODEL': [32],
    'DROPOUT': [0.3],
    'BATCH_SIZE': [64],
    'LEARNING_RATE': [1e-5],
    'EPOCHS': [50]
}

BACKTEST_SEARCH_SPACE = {
    'LEVERAGE_1': [2.0],
    'LEVERAGE_2': [5.0],
    'LEVERAGE_3': [8.0],
    'HOLD_PERIOD': [12],
    'HARD_TP': [0.015],
    'HARD_SL': [0.015],
    'TRAILING_ATR_MULTI': [3.5]
}

# 🚨 修复：传入 target_horizon，精准提取目标周期的核心指标
def extract_alpha_metrics(output, target_horizon):
    metrics = {}
    num = r"([\+\-]?(?:\d+(?:\.\d*)?|\.\d+|inf))"
    pattern = (
        rf"\[N=(\d+)h\]\s+胜率:\s*{num}%\s*\|\s*"
        rf"单笔期望:\s*{num}%\s*\|\s*IC:\s*{num}\s*\|\s*Rank_IC:\s*{num}"
        rf"(?:\s*\|\s*IC_IR:\s*{num}\s*\|\s*IC胜率:\s*{num}%\s*\|\s*IC_MDD:\s*{num})?"
    )
    matches = re.findall(pattern, output)
    
    for m in matches:
        n = int(m[0])
        metrics[f'acc_{n}h'] = float(m[1])
        metrics[f'mean_ret_{n}h'] = float(m[2])
        metrics[f'ic_{n}h'] = float(m[3])
        metrics[f'rank_ic_{n}h'] = float(m[4])
        metrics[f'ic_ir_{n}h'] = float(m[5]) if m[5] else 0.0
        metrics[f'ic_win_rate_{n}h'] = float(m[6]) if m[6] else 0.0
        metrics[f'ic_mdd_{n}h'] = float(m[7]) if m[7] else 0.0
        
    # 精准锚定主要排序指标
    metrics['primary_rank_ic'] = metrics.get(f'rank_ic_{target_horizon}h', 0.0)
    metrics['primary_acc'] = metrics.get(f'acc_{target_horizon}h', 0.0)
    metrics['primary_ic_ir'] = metrics.get(f'ic_ir_{target_horizon}h', 0.0)
    metrics['primary_ic_win_rate'] = metrics.get(f'ic_win_rate_{target_horizon}h', 0.0)
    metrics['primary_ic_mdd'] = metrics.get(f'ic_mdd_{target_horizon}h', 0.0)
    return metrics

def extract_backtest_metrics(output):
    metrics = {
        'win_rate': 0.0, 'final_equity': 0.0, 'sharpe': 0.0, 'trades': 0,
        'max_drawdown': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'ls_ratio': 0.0,
        'recovery_factor': 0.0, 'expectancy_pct': 0.0,
        'exit_reversal': 0, 'exit_hard_sl': 0, 'exit_trailing': 0, 'exit_timeout': 0
    }
    try:
        if match := re.search(r"▪ 撮合交易数 : (\d+) 笔 \(多:(\d+) \| 空:(\d+)\)", output):
            metrics['trades'] = int(match.group(1))
            shorts = int(match.group(3))
            metrics['ls_ratio'] = (int(match.group(2)) / shorts) if shorts > 0 else 999.0
            
        if match := re.search(r"▪ 最终净值\s+: ([\-\+\d\.]+)", output): metrics['final_equity'] = float(match.group(1))
        if match := re.search(r"▪ 胜率 \(WR\)\s+: ([\-\+\d\.]+)%", output): metrics['win_rate'] = float(match.group(1))
        if match := re.search(r"▪ 盈亏比 \(PF\):\s*([\-\+\d\.inf]+)", output): metrics['profit_factor'] = float(match.group(1)) if match.group(1) != 'inf' else 99.0
        if match := re.search(r"▪ 夏普比率\s+: ([\-\+\d\.]+)", output): metrics['sharpe'] = float(match.group(1))
        if match := re.search(r"▪ 卡玛比率\s*:\s*([\-\+\d\.inf]+)", output): metrics['calmar'] = float(match.group(1)) if match.group(1) != 'inf' else 99.0
        if match := re.search(r"▪ 最大回撤\s+: ([\-\+\d\.]+)%", output): metrics['max_drawdown'] = float(match.group(1))
        if match := re.search(r"▪ 恢复因子\s+: ([\-\+\d\.inf]+)", output): metrics['recovery_factor'] = float(match.group(1)) if match.group(1) != 'inf' else 99.0
        if match := re.search(r"▪ 交易期望值 : ([\-\+\d\.]+)%", output): metrics['expectancy_pct'] = float(match.group(1))
        
        if match := re.search(r"▪ REVERSAL_EXIT\s*:\s*(\d+)", output): metrics['exit_reversal'] = int(match.group(1))
        if match := re.search(r"▪ HARD_SL_EXIT\s*:\s*(\d+)", output): metrics['exit_hard_sl'] = int(match.group(1))
        if match := re.search(r"▪ TRAILING_EXIT\s*:\s*(\d+)", output): metrics['exit_trailing'] = int(match.group(1))
        if match := re.search(r"▪ TIME_LIMIT_EXIT\s*:\s*(\d+)", output): metrics['exit_timeout'] = int(match.group(1))
    except Exception: pass
    return metrics

def prefix_metrics(metrics, prefix):
    return {f"{prefix}{k}": v for k, v in metrics.items()}

def calculate_hybrid_score(metrics):
    """
    给 alpha 模式一个粗排分数，方便大量重复实验后先筛一遍。
    具体入选仍建议看 bt_final_equity / bt_expectancy_pct / bt_max_drawdown / primary_rank_ic 的组合。
    """
    bt_final_equity = float(metrics.get("bt_final_equity", 0.0) or 0.0)
    bt_expectancy_pct = float(metrics.get("bt_expectancy_pct", 0.0) or 0.0)
    bt_max_drawdown = float(metrics.get("bt_max_drawdown", 0.0) or 0.0)
    bt_trades = int(metrics.get("bt_trades", 0) or 0)
    primary_rank_ic = float(metrics.get("primary_rank_ic", 0.0) or 0.0)
    primary_ic_ir = float(metrics.get("primary_ic_ir", 0.0) or 0.0)

    trade_count_penalty = 0.0 if bt_trades >= 30 else -1.0
    drawdown_penalty = abs(min(bt_max_drawdown, 0.0)) / 100.0
    return (
        (bt_final_equity - 1.0)
        + (bt_expectancy_pct / 10.0)
        + (2.0 * primary_rank_ic)
        + (0.05 * primary_ic_ir)
        - (0.25 * drawdown_penalty)
        + trade_count_penalty
    )

def run_experiment(params, mode, base_model, save_models):
    exp_id = f"EXP_{uuid.uuid4().hex[:8].upper()}"
    env = os.environ.copy()
    env["OPT_EXP_ID"] = exp_id
    
    for k, v in params.items():
        env[f"OPT_{k}"] = str(v)
    
    env["OMP_NUM_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    stdout_log = ""
    res = {}
    
    try:
        if mode == 'alpha':
            scripts = ['train_itransformer.py', 'check_prob_distribution.py', 'alpha_ic_evaluator.py']
            for script in scripts:
                script_path = HISTORICAL_DIR / script if script == 'train_itransformer.py' else BACKTEST_DIR / script
                result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    return {"error": f"💥 {script} 报错:\n{result.stderr}", **params, 'exp_id': exp_id}
                if script == 'alpha_ic_evaluator.py':
                    stdout_log = result.stdout
            # 获取当前实验对应的 Horizon
            target_horizon = params.get('FORECAST_HORIZON', 12)
            res = extract_alpha_metrics(stdout_log, target_horizon)

            # 立刻用同一模型和同一动态概率阈值跑完整撮合，补齐“可交易性”维度。
            backtest_path = BACKTEST_DIR / 'backtest_itransformer.py'
            bt_result = subprocess.run(['python3', str(backtest_path)], capture_output=True, text=True, env=env)
            if bt_result.returncode != 0:
                res.update(prefix_metrics(extract_backtest_metrics(""), "bt_"))
                res["bt_error"] = bt_result.stderr
            else:
                bt_metrics = extract_backtest_metrics(bt_result.stdout)
                res.update(prefix_metrics(bt_metrics, "bt_"))
                res["bt_error"] = ""

            res["hybrid_score"] = calculate_hybrid_score(res)
            
        elif mode == 'backtest':
            src_model = MODEL_DIR / f"itransformer_{SYMBOL}_best_{base_model}.pth"
            src_json = MODEL_DIR / f"dynamic_thresholds_{base_model}.json"
            tmp_model = MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth"
            tmp_json = MODEL_DIR / f"dynamic_thresholds_{exp_id}.json"
            
            try:
                shutil.copy(src_model, tmp_model)
                shutil.copy(src_json, tmp_json)
            except Exception as e:
                return {"error": f"💥 基座模型复制失败: {e}", **params, 'exp_id': exp_id}
                
            script_path = BACKTEST_DIR / 'backtest_itransformer.py'
            result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                res = {"error": f"💥 回测报错:\n{result.stderr}", **params}
            else:
                stdout_log = result.stdout
                res = extract_backtest_metrics(stdout_log)

        res.update(params)
        res['exp_id'] = exp_id
        
        if not save_models:
            res['model_file'] = "Deleted"
        else:
            res['model_file'] = f"itransformer_{SYMBOL}_best_{exp_id}.pth"

        return res

    finally:
        if not save_models:
            for f in [
                MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth",
                MODEL_DIR / f"dynamic_thresholds_{exp_id}.json",
                MODEL_DIR / f"backtest_probabilistic_{exp_id}.png"
            ]:
                try: 
                    if f.exists(): os.remove(f)
                except: pass

def main():
    parser = argparse.ArgumentParser(description='Auto Optimize TimesFM')
    parser.add_argument('--mode', type=str, required=True, choices=['alpha', 'backtest'], help='运行模式: alpha 或 backtest')
    parser.add_argument('--base-model', type=str, default="", help='(仅 backtest 需要) 基座模型的 EXPERIMENT_ID')
    parser.add_argument('--repeats', type=int, default=1, help='每组参数的重复运行次数')
    parser.add_argument('--save-models', action='store_true', help='是否保存模型和图片 (建议：搜超参时不加，抽随机种子时加上)')
    args = parser.parse_args()

    if args.mode == 'backtest' and not args.base_model:
        print("❌ 错误：在 backtest 模式下，必须使用 --base-model 指定基座模型！")
        sys.exit(1)

    RESULTS_CSV = TRAIN_DATA_DIR / f"optimization_results_{args.mode}.csv"
    search_space = ALPHA_SEARCH_SPACE if args.mode == 'alpha' else BACKTEST_SEARCH_SPACE

    keys, values = zip(*search_space.items())
    base_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    combinations = []
    for combo in base_combinations:
        for _ in range(args.repeats):
            combinations.append(combo.copy())
            
    print(f"🔥 启动双轨引擎 | 当前模式: 【{args.mode.upper()}】")
    print(f"   参数组合数: {len(base_combinations)} (总任务数: {len(combinations)})")
    print(f"   保存模型开关: {'开启 💾' if args.save_models else '关闭 🗑️'}")
    if args.mode == 'backtest': print(f"   基座 Alpha 模型: {args.base_model}")
    print("=" * 80)

    # 🚨 并发控制：alpha 模式 8 并发；backtest 模式 16 并发
    MAX_WORKERS = 8 if args.mode == 'alpha' else 16
    batches = [combinations[i:i + MAX_WORKERS] for i in range(0, len(combinations), MAX_WORKERS)]
    
    all_results = []
    start_time = time.time()
    completed_tasks = 0
    
    for batch_idx, batch in enumerate(batches, 1):
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_experiment, params, args.mode, args.base_model, args.save_models): params for params in batch}
            
            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                completed_tasks += 1
                
                progress_str = f"[{completed_tasks}/{len(combinations)}]"
                if "error" in res:
                    print(f"   {progress_str} ❌ 失败:\n{res['error']}")
                else:
                    if args.mode == 'alpha':
                        target_hz = res.get('FORECAST_HORIZON', 12)
                        rank_ic = res.get('primary_rank_ic', 0)
                        acc_val = res.get('primary_acc', 0)
                        ic_ir = res.get('primary_ic_ir', 0)
                        ic_win = res.get('primary_ic_win_rate', 0)
                        ic_mdd = res.get('primary_ic_mdd', 0)
                        bt_equity = res.get('bt_final_equity', 0)
                        bt_expectancy = res.get('bt_expectancy_pct', 0)
                        bt_mdd = res.get('bt_max_drawdown', 0)
                        bt_trades = res.get('bt_trades', 0)
                        hybrid_score = res.get('hybrid_score', 0)
                        bt_error_note = " | BT_ERR" if res.get("bt_error") else ""
                        print(
                            f"   {progress_str} 🧠 [Alpha] 核心 {target_hz}h Rank_IC: {rank_ic:>+6.4f} | "
                            f"IC_IR: {ic_ir:>+6.3f} | IC胜率: {ic_win:>5.2f}% | IC_MDD: {ic_mdd:>6.4f} | "
                            f"胜率: {acc_val:>5.2f}% || BT净值:{bt_equity:>7.4f} | "
                            f"BT期望:{bt_expectancy:>+6.3f}% | BT回撤:{bt_mdd:>+7.2f}% | "
                            f"交易:{bt_trades:>4} | 综合:{hybrid_score:>+7.4f}{bt_error_note} | ID: {res['exp_id']}"
                        )
                    else:
                        print(f"   {progress_str} 🏦 [回测] 期望值: {res['expectancy_pct']:>+6.3f}% | 恢复因子: {res['recovery_factor']:>5.2f} | ID: {res['exp_id']}")
        
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    print("=" * 80)
    print(f"🎉 任务完成！耗时: {(time.time() - start_time) / 60:.2f} 分钟。报告已存至 {RESULTS_CSV.name}")

if __name__ == "__main__":
    main()
