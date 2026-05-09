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
    metrics = {'win_rate': 0.0, 'final_equity': 0.0, 'sharpe': 0.0, 'trades': 0}
    try:
        if match := re.search(r"胜率 \(Win Rate\): ([\d\.]+)%", output): metrics['win_rate'] = float(match.group(1))
        if match := re.search(r"期末策略净值: ([\d\.]+)", output): metrics['final_equity'] = float(match.group(1))
        if match := re.search(r"估算年化夏普: ([\-\d\.]+)", output): metrics['sharpe'] = float(match.group(1))
        if match := re.search(r"共计交易: (\d+)", output): metrics['trades'] = int(match.group(1))
    except: pass
    return metrics

def run_experiment(params):
    # 1. 生成唯一实验 ID (用哈希或简单拼接防止名字过长)
    exp_id = f"EXP_{hash(frozenset(params.items())) % 1000000}"
    
    # 2. 注入全量环境变量
    env = os.environ.copy()
    env["OPT_EXP_ID"] = exp_id
    for k, v in params.items():
        env[f"OPT_{k}"] = str(v)
    # 🚨 极度重要：强行限制所有底层 C 库只能单线程运行，防止 CPU 踩踏
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["POLARS_MAX_THREADS"] = "1"
    
    scripts = ['train_itransformer.py', 'check_prob_distribution.py', 'backtest_itransformer.py']
    backtest_out = ""
    
    for script in scripts:
        if script == 'train_itransformer.py':
            script_path = HISTORICAL_DIR / script
        else:
            script_path = BACKTEST_DIR / script
        result = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            return {"error": f"\n💥 {script} 崩溃了！真实报错日志如下:\n{result.stderr}", **params}
        if script == 'backtest_itransformer.py':
            backtest_out = result.stdout
            
    res = extract_metrics(backtest_out)
    res.update(params)
    res['exp_id'] = exp_id
    
    # 并发产生的文件极多，回测完务必删掉模型和JSON释放硬盘！
    try:
        #os.remove(MODEL_DIR / f"itransformer_BTCUSDT_best_{exp_id}.pth")
        os.remove(TRAIN_DATA_DIR / f"dynamic_thresholds_{exp_id}.json")
    except: pass
    
    return res

def main():
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔥 启动并发寻优，共 {len(combinations)} 组参数矩阵...")
    
    # H100 拥有强大的并发能力，设置 16-24 个 Worker 塞满显存
    MAX_WORKERS = 16 
    all_results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_experiment, params): params for params in combinations}
        
        for count, future in enumerate(as_completed(futures), 1):
            res = future.result()
            all_results.append(res)
            
            if "error" in res:
                print(f"[{count}/{len(combinations)}] ❌ 失败: {res['error']}")
            else:
                print(f"[{count}/{len(combinations)}] ✅ 夏普 {res['sharpe']:.2f} | 净值 {res['final_equity']:.2f} | 交易 {res['trades']}")
            
            # 存盘
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    print(f"\n🎉 并发寻优完毕！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

if __name__ == "__main__":
    main()