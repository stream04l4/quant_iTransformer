#!/bin/bash

# ==========================================
# 🚀 Quant iTransformer 项目环境一键初始化
# ==========================================

echo "🔄 正在更新系统依赖与安装基础工具..."
sudo apt-get update && sudo apt-get install -y build-essential curl git

echo "📦 正在安装核心数据科学与量化交易依赖..."
# 1. 基础数据处理与可视化
pip install pandas polars numpy matplotlib seaborn plotly pyarrow

# 2. 机器学习与深度学习 (Torch 默认安装 CPU/CUDA 兼容版)
pip install torch torchvision torchaudio
pip install scikit-learn

# 3. 量化交易与接口
pip install ccxt requests yfinance

# 4. 实时监控与 Web 框架 (Streamlit + FastAPI)
pip install streamlit fastapi uvicorn psutil python-dotenv

# 5. 显卡监控与运维工具
echo "🛠️ 正在修复 pynvml 环境..."
sudo pip uninstall pynvml -y
pip install nvidia-ml-py

echo "✅ 所有依赖安装完毕！"
echo "💡 提示：运行实盘监控请执行: streamlit run 0_⚡_实盘监控指挥塔.py"