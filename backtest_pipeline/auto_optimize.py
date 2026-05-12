import os
import subprocess
import re
import json
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import sys
import argparse
import uuid
import gc
import torch

# ==========================================
# 1. 通宵级全维寻优空间 
# ==========================================
SEARCH_SPACE = {
    'SEQ_LEN': [336],
    'CLASS_THRESHOLD': [0.026],
    'E_LAYERS': [1],
    'D_MODEL': [256],
    'DROPOUT': [0.4],
    'LEARNING_RATE': [5e-5],
    'BATCH_SIZE': [512],
    'EPOCHS': [50] 
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import TRAIN_DATA_DIR, MODEL_DIR, HISTORICAL_DIR, BACKTEST_DIR, SYMBOL

RESULTS_CSV = TRAIN_DATA_DIR / "optimization_results.csv"

def extract_metrics(output):
    metrics = {
        'win_rate': 0.0, 'final_equity': 0.0, 'sharpe': 0.0, 'trades': 0,
        'max_drawdown': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'ls_ratio': 0.0
    }
    try:
        if match := re.search(r"胜率 \(Win Rate\): ([\d\.]+)% \| 盈亏比.*: ([\d\.inf]+)", output): 
            metrics['win_rate'] = float(match.group(1))
            metrics['profit_factor'] = float(match.group(2)) if match.group(2) != 'inf' else 999.0
        if match := re.search(r"期末策略净值: ([\d\.]+)", output): metrics['final_equity'] = float(match.group(1))
        if match := re.search(r"估算年化夏普: ([\-\d\.]+) \| 卡玛比率: ([\-\d\.inf]+)", output): 
            metrics['sharpe'] = float(match.group(1))
            metrics['calmar'] = float(match.group(2)) if match.group(2) != 'inf' else 999.0
        if match := re.search(r"最大回撤.*: ([\-\d\.]+)%", output): metrics['max_drawdown'] = float(match.group(1))
        if match := re.search(r"共计交易: (\d+) 笔 .*多空比: ([\d\.inf]+)", output): 
            metrics['trades'] = int(match.group(1))
            metrics['ls_ratio'] = float(match.group(2)) if match.group(2) != 'inf' else 999.0
    except Exception: 
        pass
    return metrics

def run_experiment(params, save_models):
    exp_id = f"EXP_{uuid.uuid4().hex[:8].upper()}"
    
    env = os.environ.copy()
    env["OPT_EXP_ID"] = exp_id
    for k, v in params.items():
        env[f"OPT_{k}"] = str(v)
    
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "1"
    
    scripts = ['train_itransformer.py', 'check_prob_distribution.py', 'backtest_itransformer.py']
    backtest_out = ""
    
    for script in scripts:
        script_path = HISTORICAL_DIR / script if script == 'train_itransformer.py' else BACKTEST_DIR / script
        result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            return {"error": f"💥 {script} 报错:\n{result.stderr}", **params}
        if script == 'backtest_itransformer.py':
            backtest_out = result.stdout
            
    res = extract_metrics(backtest_out)
    res.update(params)
    res['exp_id'] = exp_id
    
    # 子进程内的即时清理 (防积压第一道防线)
    model_file_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth"
    if not save_models:
        try: os.remove(model_file_path) 
        except FileNotFoundError: pass
        res['model_file'] = "Deleted"
    else:
        res['model_file'] = model_file_path.name

    if not save_models:
        try: os.remove(MODEL_DIR / f"dynamic_thresholds_{exp_id}.json") 
        except FileNotFoundError: pass
        try: os.remove(MODEL_DIR / f"backtest_probabilistic_{exp_id}.png")
        except FileNotFoundError: pass
    
    return res

def main():
    parser = argparse.ArgumentParser(description='Auto Optimize iTransformer')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeats for each parameter combination')
    parser.add_argument('--save-models', action='store_true', help='Whether to save models and dynamic thresholds')
    args = parser.parse_args()

    keys, values = zip(*SEARCH_SPACE.items())
    base_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    combinations = []
    for combo in base_combinations:
        for _ in range(args.repeats):
            combinations.append(combo.copy())
            
    print(f"🔥 模式切换：进入【统一寻优模式】...")
    print(f"   参数组合数: {len(base_combinations)}")
    print(f"   每组重复次数: {args.repeats}")
    print(f"   是否保存模型: {'是' if args.save_models else '否'}")

    MAX_WORKERS = 8 
    total_tasks = len(combinations)
    
    # 🚨 核心逻辑：将任务切分为多个批次 (Chunking)
    batches = [combinations[i:i + MAX_WORKERS] for i in range(0, total_tasks, MAX_WORKERS)]
    
    print(f"📦 任务已被切分为 {len(batches)} 个批次，每批最大 {MAX_WORKERS} 组。将执行严格的批次清理。")
    print("=" * 70)

    all_results = []
    start_time = time.time()
    completed_tasks = 0
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n🚀 正在拉起第 {batch_idx}/{len(batches)} 批次 (包含 {len(batch)} 组模型) ...")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_experiment, params, args.save_models): params for params in batch}
            
            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                completed_tasks += 1
                
                # 动态进度条计算
                progress_pct = (completed_tasks / total_tasks) * 100
                progress_str = f"[{completed_tasks}/{total_tasks} - {progress_pct:.1f}%]"
                
                if "error" in res:
                    print(f"   {progress_str} ❌ 失败: {res['error'].splitlines()[0]}") # 只打首行防刷屏
                else:
                    print(f"   {progress_str} ✅ 夏普: {res['sharpe']:>5.2f} | 净值: {res['final_equity']:>5.2f} | 回撤: {res['max_drawdown']:>6.2f}%")
        
        # 🚨 每个批次结束后的“第二道防线”：物理级强制扫盘清场
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
        
        swept_files = 0
        if not args.save_models:
            for f in MODEL_DIR.glob("dynamic_thresholds_EXP_*.json"):
                try: os.remove(f); swept_files += 1 
                except Exception: pass
            for f in TRAIN_DATA_DIR.glob("backtest_probabilistic_EXP_*.png"):
                try: os.remove(f); swept_files += 1
                except Exception: pass
            for f in MODEL_DIR.glob("itransformer_*_best_EXP_*.pth"):
                try: os.remove(f); swept_files += 1 
                except Exception: pass
            
        print(f"🧹 第 {batch_idx} 批次收尾完毕 | CSV已更新 | 强制清场扫描清理了 {swept_files} 个僵尸残余。")

    print("=" * 70)
    print(f"🎉 寻优任务大功告成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟。")
    print(f"💾 所有结果已安全保存至 {RESULTS_CSV.name}")

if __name__ == "__main__":
    main()