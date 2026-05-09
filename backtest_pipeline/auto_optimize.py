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

# 新增一个配置：当检测到参数唯一时，重复训练几次来找寻最优模型？
SINGLE_MODE_REPEATS = 5
# ==========================================
# 1. 通宵级全维寻优空间 (总计 1296 组)
# ==========================================
SEARCH_SPACE = {
    # 🎯 观测视野 (Alpha 的核心来源)：
    # 168 (7天), 336 (14天), 504 (21天), 720 (30天)
    # 测试模型是更擅长短期动量，还是长期均值回归
    'SEQ_LEN': [336],

    # 🧬 梯度噪音控制：
    'BATCH_SIZE': [256],

    # 脑容量与深度：
    # 直接锁死 E_LAYERS=1，不浪费算力。
    # 增加 D_MODEL=16 探底，看看更极端的微型网络会不会泛化更好
    'E_LAYERS': [1],
    'D_MODEL': [64],

    # 🛡️ 极致防过拟合：
    'DROPOUT': [0.4],

    # 步伐大小：
    'LEARNING_RATE': [1e-4],

    # ⚖️ 赔率与胜率的天平 (极其精细地扫描 0.025 附近的黄金坑)：
    'CLASS_THRESHOLD': [0.027],
    
    'EPOCHS': [100] # 固定 100 轮交由余弦退火和早停控制
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from quant_config.config import TRAIN_DATA_DIR, MODEL_DIR, HISTORICAL_DIR, BACKTEST_DIR

RESULTS_CSV = TRAIN_DATA_DIR / "optimization_results.csv"

def extract_metrics(output):
    # 增加新的维度捕获
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
    except: pass
    return metrics

def run_experiment(params, is_grid_search):
    exp_id = f"EXP_{hash(frozenset(params.items()) if is_grid_search else str(time.time())) % 1000000}"
    
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
            return {"error": f"\n💥 {script} 崩溃了！真实报错日志:\n{result.stderr}", **params}
        if script == 'backtest_itransformer.py':
            backtest_out = result.stdout
            
    res = extract_metrics(backtest_out)
    res.update(params)
    res['exp_id'] = exp_id
    
    # 🚨 双模式清理逻辑
    model_file_path = MODEL_DIR / f"itransformer_{SYMBOL}_best_{exp_id}.pth"
    if is_grid_search:
        # 网格寻优模式：跑完就删，节约硬盘
        try: os.remove(model_file_path) except: pass
        res['model_file'] = "Deleted (Grid Search)"
    else:
        # 单一复训模式：保留模型，并在表格中记录名字
        res['model_file'] = model_file_path.name

    try: os.remove(TRAIN_DATA_DIR / f"dynamic_thresholds_{exp_id}.json") except: pass
    
    return res

def main():
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    is_grid_search = len(combinations) > 1
    
    if not is_grid_search:
        print(f"🔍 模式切换：检测到参数唯一，进入【稳健性盲盒模式】...")
        print(f"   将使用相同参数运行 {SINGLE_MODE_REPEATS} 次，并保留所有过程模型，以便你挑选夏普最高的种子。")
        combinations = [combinations[0] for _ in range(SINGLE_MODE_REPEATS)]
    else:
        print(f"🔥 模式切换：检测到多种参数组合，进入【海量并发寻优模式】...")
        print(f"   共 {len(combinations)} 组矩阵，过程中产生的冗余模型将会被自动销毁。")

    MAX_WORKERS = 16 
    all_results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_experiment, params, is_grid_search): params for params in combinations}
        
        for count, future in enumerate(as_completed(futures), 1):
            res = future.result()
            all_results.append(res)
            
            if "error" in res:
                print(f"[{count}/{len(combinations)}] ❌ 失败: {res['error']}")
            else:
                # 打印出带有回撤的新日志
                print(f"[{count}/{len(combinations)}] ✅ 夏普: {res['sharpe']:.2f} | 净值: {res['final_equity']:.2f} | 回撤: {res['max_drawdown']:.2f}% | 模型: {res['model_file']}")
            
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    print(f"\n🎉 任务完毕！总耗时: {(time.time() - start_time) / 60:.2f} 分钟。结果已保存至 {RESULTS_CSV.name}")