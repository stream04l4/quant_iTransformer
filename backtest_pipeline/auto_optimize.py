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

# ==========================================
# 1. 双轨寻优空间定义
# ==========================================
# 模式一：纯粹的 Alpha 预测力寻优 (不带任何止盈止损)
ALPHA_SEARCH_SPACE = {
    'FORECAST_HORIZON': [6, 12, 24],
    'SEQ_LEN': [168, 336],
    'CLASS_THRESHOLD': [0.024, 0.025, 0.026],
    'E_LAYERS': [1, 2],
    'D_MODEL': [64, 128, 256],
    'DROPOUT': [0.3, 0.4, 0.5],
    'BATCH_SIZE': [128, 256],
    'LEARNING_RATE': [1e-6],
    'EPOCHS': [50]
}

# 模式二：基于某一个绝佳 Alpha 模型的出场参数寻优
BACKTEST_SEARCH_SPACE = {
    'LEVERAGE_1': [1.0, 2.0],
    'LEVERAGE_2': [3.0, 4.0],
    'LEVERAGE_3': [5.0, 7.0],
    'HOLD_PERIOD': [12, 24, 48],
    'HARD_TP': [0.04, 0.06, 0.08],
    'HARD_SL': [0.03, 0.04, 0.06]
}

def extract_alpha_metrics(output):
    """解析 alpha_ic_evaluator.py 的输出"""
    metrics = {}
    pattern = r"\[N=(\d+)h\]\s+胜率:\s*([\d\.]+)%\s*\|\s*单笔期望:\s*([\+\-\d\.]+)%\s*\|\s*IC:\s*([\+\-\d\.]+)\s*\|\s*Rank_IC:\s*([\+\-\d\.]+)"
    matches = re.findall(pattern, output)
    
    for m in matches:
        n = m[0]
        metrics[f'acc_{n}h'] = float(m[1])
        metrics[f'mean_ret_{n}h'] = float(m[2])
        metrics[f'ic_{n}h'] = float(m[3])
        metrics[f'rank_ic_{n}h'] = float(m[4])
        metrics['primary_rank_ic'] = float(m[4]) # 默认最后一个为主要排序指标
    return metrics

def extract_backtest_metrics(output):
    """解析 backtest_itransformer.py 的输出"""
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
        if match := re.search(r"▪ 夏普比率\s+: ([\-\+\d\.]+)", output): metrics['sharpe'] = float(match.group(1))
        if match := re.search(r"▪ 最大回撤\s+: ([\-\+\d\.]+)%", output): metrics['max_drawdown'] = float(match.group(1))
        if match := re.search(r"▪ 恢复因子\s+: ([\-\+\d\.inf]+)", output): metrics['recovery_factor'] = float(match.group(1)) if match.group(1) != 'inf' else 99.0
        if match := re.search(r"▪ 交易期望值 : ([\-\+\d\.]+)%", output): metrics['expectancy_pct'] = float(match.group(1))
    except Exception: pass
    return metrics

def run_experiment(params, mode, base_model, save_models):
    exp_id = f"EXP_{uuid.uuid4().hex[:8].upper()}"
    env = os.environ.copy()
    env["OPT_EXP_ID"] = exp_id
    
    for k, v in params.items():
        env[f"OPT_{k}"] = str(v)
    
    env["OMP_NUM_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "1"
    
    stdout_log = ""
    res = {}
    
    if mode == 'alpha':
        scripts = ['train_itransformer.py', 'check_prob_distribution.py', 'alpha_ic_evaluator.py']
        for script in scripts:
            script_path = HISTORICAL_DIR / script if script != 'alpha_ic_evaluator.py' else BACKTEST_DIR / script
            result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
            if result.returncode != 0:
                return {"error": f"💥 {script} 报错:\n{result.stderr}", **params}
            if script == 'alpha_ic_evaluator.py':
                stdout_log = result.stdout
        res = extract_alpha_metrics(stdout_log)
        
    elif mode == 'backtest':
        src_model = MODEL_DIR / f"itransformer_{SYMBOL}_best_{base_model}.pth"
        src_json = MODEL_DIR / f"dynamic_thresholds_{base_model}.json"
        
        tmp_model = MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth"
        tmp_json = MODEL_DIR / f"dynamic_thresholds_{exp_id}.json"
        
        try:
            shutil.copy(src_model, tmp_model)
            shutil.copy(src_json, tmp_json)
        except Exception as e:
            return {"error": f"💥 基座模型复制失败: {e}", **params}
            
        script_path = BACKTEST_DIR / 'backtest_itransformer.py'
        result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            res = {"error": f"💥 回测报错:\n{result.stderr}", **params}
        else:
            stdout_log = result.stdout
            res = extract_backtest_metrics(stdout_log)
            
        # 回测打完立刻销毁临时分身
        try:
            os.remove(tmp_model)
            os.remove(tmp_json)
        except: pass

    res.update(params)
    res['exp_id'] = exp_id
    
    # 【第一道防线】：根据 --save-models 决定是否清理当前进程产物
    if not save_models:
        try: os.remove(MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth") 
        except: pass
        try: os.remove(MODEL_DIR / f"dynamic_thresholds_{exp_id}.json") 
        except: pass
        try: os.remove(MODEL_DIR / f"backtest_probabilistic_{exp_id}.png")
        except: pass
        res['model_file'] = "Deleted"
    else:
        res['model_file'] = f"itransformer_{SYMBOL}_best_{exp_id}.pth"

    return res

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
    print(f"   保存模型开关: {'开启 💾 (将占用磁盘空间)' if args.save_models else '关闭 🗑️ (跑完即焚)'}")
    if args.mode == 'backtest': print(f"   基座 Alpha 模型: {args.base_model}")
    print("=" * 80)

    MAX_WORKERS = 2 if args.mode == 'alpha' else 8 
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
                    print(f"   {progress_str} ❌ 失败: {res['error'].splitlines()[0]}")
                else:
                    if args.mode == 'alpha':
                        rank_ic = res.get('primary_rank_ic', 0)
                        acc_12h = res.get('acc_12h', 0)
                        print(f"   {progress_str} 🧠 [Alpha] 核心 Rank_IC: {rank_ic:>+6.4f} | 12h 胜率: {acc_12h:>5.2f}% | ID: {res['exp_id']}")
                    else:
                        print(f"   {progress_str} 🏦 [回测] 期望值: {res['expectancy_pct']:>+6.3f}% | 恢复因子: {res['recovery_factor']:>5.2f} | ID: {res['exp_id']}")
        
        # 每批次保存 CSV 
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

        # 【第二道防线】：如果未开启 save-models，进行批次级别的物理兜底清理
        if not args.save_models:
            swept_files = 0
            for f in MODEL_DIR.glob("dynamic_thresholds_EXP_*.json"):
                try: os.remove(f); swept_files += 1 
                except Exception: pass
            for f in MODEL_DIR.glob("backtest_probabilistic_EXP_*.png"):
                try: os.remove(f); swept_files += 1
                except Exception: pass
            for f in MODEL_DIR.glob("itransformer_*_best_EXP_*.pth"):
                try: os.remove(f); swept_files += 1 
                except Exception: pass
            if swept_files > 0:
                print(f"   🧹 第 {batch_idx} 批次收尾 | 强制清理了 {swept_files} 个僵尸残余文件。")

    print("=" * 80)
    print(f"🎉 任务完成！耗时: {(time.time() - start_time) / 60:.2f} 分钟。报告已存至 {RESULTS_CSV.name}")

if __name__ == "__main__":
    main()