# quant_iTransformer

`quant_iTransformer` 是一套面向加密货币永续合约的量化交易系统。系统以 Binance 历史数据和实时行情为数据源，使用 iTransformer 结构训练方向分类模型，并通过回测、概率阈值、宏观情绪融合、风控和 Gate.io 模拟盘执行链路完成从研究到实盘的闭环。

> 说明：当前配置默认标的为 `BTCUSDT`，辅助标的为 `ETHUSDT`、`ETHBTC`。所有实盘交易代码默认使用 Gate.io sandbox 模式。

## 目录结构

```text
quant_iTransformer/
├── quant_config/
│   └── config.py                 # 全局路径、标的、模型、训练、回测、实盘配置
├── historical_pipeline/
│   ├── build_aggtrades.py         # 下载并聚合 aggTrades 订单流特征
│   ├── build_main_features.py     # 构建主训练宽表
│   ├── inspect_features.py        # 特征检查工具
│   └── train_itransformer.py      # iTransformer 训练入口与模型定义
├── backtest_pipeline/
│   ├── check_prob_distribution.py # 生成动态概率阈值 JSON
│   ├── alpha_ic_evaluator.py      # Alpha/IC/Rank IC 评估
│   ├── backtest_itransformer.py   # 概率阈值驱动的撮合回测
│   └── auto_optimize.py           # Alpha 与回测双轨超参搜索
├── market_gateway.py              # 本地 FastAPI 行情网关，维护实时行情内存状态
├── live_pipeline.py               # 实盘推理、宏观融合、信号分层与下单调度
├── execution_oms.py               # Gate.io 下单、止盈止损、账本与推送
├── risk_manager.py                # 高频熔断、止损覆盖、右脑风控
├── sentiment_scraper.py           # DeepSeek 宏观情绪哨兵
├── state_manager.py               # 线程共享状态
├── 0_⚡_实盘监控指挥塔.py          # Streamlit 实盘监控面板
├── data/
│   ├── train/                     # 训练数据、缓存、优化结果、实盘审计 CSV
│   └── live/                      # 实盘状态、SQLite、JSON 快照
└── model/                         # 模型权重、动态阈值、回测图表
```

## 整体架构

```text
Binance 历史数据
  ├─ klines / premium / funding / open interest
  └─ aggTrades
        ↓
historical_pipeline
  ├─ build_aggtrades.py      订单流聚合到 1h
  └─ build_main_features.py  合成训练宽表与目标收益
        ↓
train_itransformer.py
  └─ 训练 3 分类 iTransformer：看跌 / 震荡 / 看涨
        ↓
backtest_pipeline
  ├─ check_prob_distribution.py  生成动态概率阈值
  ├─ alpha_ic_evaluator.py       检查 Alpha 排序能力
  └─ backtest_itransformer.py    撮合回测与策略评估
        ↓
实盘链路
  ├─ market_gateway.py      WebSocket/REST 行情网关与本地缓存
  ├─ sentiment_scraper.py   宏观情绪评分
  ├─ live_pipeline.py       多模型加权推理、宏观 logit 融合、信号分层
  ├─ execution_oms.py       Gate.io sandbox 执行与止损单管理
  └─ risk_manager.py        高频熔断、追踪止盈、仓位保护
        ↓
Streamlit 监控面板
```

### 离线训练链路

1. `build_aggtrades.py` 从 Binance futures monthly/daily 目录下载 `aggTrades`，按小时聚合 `cvd`、大单买卖、散户占比、鲸鱼多空比等微观订单流特征。
2. `build_main_features.py` 下载并合并 K 线、资金费率、溢价指数、open interest、辅助标的等数据，生成主训练宽表。
3. `train_itransformer.py` 读取训练宽表，将未来 `FORECAST_HORIZON` 小时收益按 `CLASS_THRESHOLD` 划为三类：看跌、震荡、看涨。
4. 训练采用 purged split，训练集和验证集之间按预测周期隔离，降低标签泄露风险。
5. 最优模型按 `selection_score` 保存到 `model/itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth`，权重文件内包含模型结构参数、特征列、预测周期、验证指标等元数据。

### 模型结构

核心模型定义在 `historical_pipeline/train_itransformer.py` 的 `InvertedTransformer`：

- 输入形状：`[batch, seq_len, num_features]`
- 反转维度：转为 `[batch, num_features, seq_len]`，让每个变量作为 token，时间窗口作为 token embedding 的输入。
- 编码层：`nn.TransformerEncoder`，`n_heads` 由 `D_MODEL` 自动选择可整除值。
- 输出层：flatten 后通过 MLP 输出 3 个 logits，对应 `down / flat / up`。
- 训练损失：`CrossEntropyLoss`，带类别权重和 label smoothing。
- 模型选择指标：`Rank IC + Top precision edge + tail return spread` 组合成 `selection_score`。

### 回测与阈值链路

- `check_prob_distribution.py` 在纯净历史区间上统计 `p_up`、`p_down` 分位数，生成 `model/dynamic_thresholds_{EXPERIMENT_ID}.json`。
- `alpha_ic_evaluator.py` 评估模型输出 `alpha_score = p_up - p_down` 与未来收益的 IC、Rank IC、IC IR、IC 胜率、信号覆盖率。
- `backtest_itransformer.py` 使用动态阈值执行概率分层回测，支持单模型和多模型加权：
  - `--models EXP_A:0.4,EXP_B:0.6`
  - `--use-live-models`
- `auto_optimize.py` 支持两种调参模式：
  - `alpha`：训练模型、生成阈值、评估 Alpha，再补跑完整回测。
  - `backtest`：复制指定基座模型，仅搜索交易参数。

### 实盘链路

1. `market_gateway.py` 负责实时行情网关，启动 FastAPI 服务 `127.0.0.1:8000`，提供 `/api/snapshot/{symbol}`。
2. 网关同时维护 1h、1m、订单簿、成交流、资金费率、溢价指数、open interest 等内存状态，并通过 SQLite 做本地恢复。
3. `live_pipeline.py` 每个心跳周期读取网关快照，整点后等待闭合 K 线稳定，再构建与离线一致的实时特征矩阵。
4. 实盘支持多模型 ensemble，按配置权重加权 `p_down / p_flat / p_up`。
5. 宏观情绪来自 `sentiment_scraper.py` 写入的 `data/live/macro_intelligence.db`，实盘通过非线性 logit 融合调整概率。
6. 信号按动态阈值分为 Top 10 / Top 5 / Top 1 三档，对应不同杠杆。
7. `execution_oms.py` 负责下单、止损单、交易流水、PushPlus 推送；`risk_manager.py` 负责异常爆量、价格瀑布、止损覆盖和追踪止盈。

## 配置说明

所有核心配置集中在 `quant_config/config.py`。脚本启动时会读取 `.env`，并允许使用 `OPT_` 前缀的环境变量覆盖部分训练、回测和策略参数。

### 路径配置

| 配置 | 默认位置 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `data/` | 项目数据根目录 |
| `TRAIN_DATA_DIR` | `data/train/` | 训练数据、优化结果、实盘特征审计 |
| `LIVE_DATA_DIR` | `data/live/` | 实盘状态 JSON 与 SQLite |
| `TEMP_DIR` | `data/train/raw_temp/` | 历史数据下载与缓存 |
| `PROCESSED_AGGTRADES_DIR` | `data/train/processed/aggtrades_1h/{SYMBOL}` | 订单流小时特征 |
| `PROCESSED_MAIN_DIR` | `data/train/processed/main_features_1h/{SYMBOL}` | 主训练宽表 |
| `MODEL_DIR` | `model/` | 模型权重、阈值、回测图 |

### 标的与数据区间

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `SYMBOL` | `BTCUSDT` | 主交易标的 |
| `AUX_SYMBOLS` | `["ETHUSDT", "ETHBTC"]` | 辅助参考标的 |
| `DATA_START_MONTH` | `(2021, 12)` | 历史数据起始年月 |
| `DATA_END_MONTH` | `(2026, 5)` | 历史数据结束年月 |
| `WHALE_NOTIONAL_THRESHOLD_USDT` | `200000.0` | 大单名义金额阈值 |
| `RETAIL_NOTIONAL_THRESHOLD_USDT` | `4000.0` | 小单名义金额阈值 |

### 训练与模型配置

| 配置 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `EXPERIMENT_ID` | `OPT_EXP_ID` | `default_run` | 实验 ID，决定模型和阈值文件名 |
| `FORECAST_HORIZON` | `OPT_FORECAST_HORIZON` | `12` | 预测未来 N 小时收益 |
| `SEQ_LEN` | `OPT_SEQ_LEN` | `672` | 输入时间窗口长度 |
| `BATCH_SIZE` | `OPT_BATCH_SIZE` | `128` | 训练 batch size |
| `EPOCHS` | `OPT_EPOCHS` | `100` | 最大训练轮数 |
| `LEARNING_RATE` | `OPT_LEARNING_RATE` | `5e-6` | AdamW 学习率 |
| `D_MODEL` | `OPT_D_MODEL` | `64` | Transformer 隐层维度 |
| `E_LAYERS` | `OPT_E_LAYERS` | `1` | Encoder 层数 |
| `DROPOUT` | `OPT_DROPOUT` | `0.3` | Dropout |
| `CLASS_THRESHOLD` | `OPT_CLASS_THRESHOLD` | `0.035` | 未来收益分类阈值 |

### 回测与交易配置

| 配置 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `LEVERAGE_1` | `OPT_LEVERAGE_1` | `2.0` | Tier 1 杠杆 |
| `LEVERAGE_2` | `OPT_LEVERAGE_2` | `5.0` | Tier 2 杠杆 |
| `LEVERAGE_3` | `OPT_LEVERAGE_3` | `8.0` | Tier 3 杠杆 |
| `FEE_RATE` | 无 | `0.0005` | 手续费率 |
| `HOLD_PERIOD` | `OPT_HOLD_PERIOD` | `12` | 最大持仓小时数 |
| `HARD_TP` | `OPT_HARD_TP` | `0.025` | 固定止盈 |
| `HARD_SL` | `OPT_HARD_SL` | `0.025` | 固定止损 |
| `TRAILING_ATR_PERIOD` | `OPT_TRAILING_ATR_PERIOD` | `14` | 追踪止盈 ATR 周期 |
| `TRAILING_ATR_MULTI` | `OPT_TRAILING_ATR_MULTI` | `3.5` | 追踪止盈 ATR 倍数 |

### 实盘与风控配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `HEARTBEAT_INTERVAL` | `10` | 实盘主循环心跳秒数 |
| `INITIAL_CAPITAL` | `5000.0` | 统计和仓位预算使用的账户初始资金 |
| `BASE_RISK_BUDGET` | `0.2` | 单笔保证金预算比例 |
| `MAX_NOTIONAL_BTC` | `100000` | BTC/ETH 最大 notional |
| `MAX_NOTIONAL_ALTS` | `10000` | 其他币种最大 notional |
| `BAILOUT_ATR_MULTI` | `3.5` | 高频熔断 ATR 跌幅阈值 |
| `BAILOUT_VOL_MULTI` | `4.0` | 高频熔断爆量阈值 |
| `BAILOUT_MIN_PHYSICAL_DROP` | `0.005` | 熔断最小物理跌幅 |
| `BAILOUT_OBI_THRESH` | `0.2` | 订单簿失衡确认阈值 |

### 宏观情绪融合配置

| 配置 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MACRO_ALPHA_TOP_10` | 无 | `0.10` | Tier 1 宏观权重 |
| `MACRO_ALPHA_TOP_5` | 无 | `0.05` | Tier 2 宏观权重 |
| `MACRO_ALPHA_TOP_1` | 无 | `0.02` | Tier 3 宏观权重 |
| `MACRO_LOGIT_DEAD_ZONE` | `OPT_MACRO_LOGIT_DEAD_ZONE` | `0.20` | 情绪死区 |
| `MACRO_LOGIT_MID_SCORE` | `OPT_MACRO_LOGIT_MID_SCORE` | `0.50` | 中段情绪锚点 |
| `MACRO_LOGIT_TAIL_SCORE` | `OPT_MACRO_LOGIT_TAIL_SCORE` | `0.80` | 尾部情绪锚点 |
| `MACRO_LOGIT_MID_DELTA` | `OPT_MACRO_LOGIT_MID_DELTA` | `0.075` | 中段 logit 调整 |
| `MACRO_LOGIT_TAIL_DELTA` | `OPT_MACRO_LOGIT_TAIL_DELTA` | `0.420` | 尾部 logit 调整 |
| `MACRO_LOGIT_MAX_DELTA` | `OPT_MACRO_LOGIT_MAX_DELTA` | `0.550` | 最大 logit 调整 |

### 实盘模型组合

`LIVE_SYMBOL_CONFIGS` 定义实盘加载哪些标的和模型。当前 `BTCUSDT` 默认使用四个模型做加权 ensemble：

```python
BTCUSDT_LIVE_MODELS = [
    {"experiment_id": "EXP_6ADB71DE", "weight": 0.4},
    {"experiment_id": "EXP_A49047F0", "weight": 0.25},
    {"experiment_id": "EXP_AA8D8236", "weight": 0.2},
    {"experiment_id": "EXP_321B10D8", "weight": 0.15},
]
```

每个模型需要同时存在：

```text
model/itransformer_{SYMBOL}_best_{EXPERIMENT_ID}.pth
model/dynamic_thresholds_{EXPERIMENT_ID}.json
```

新增标的时，按以下结构追加：

```python
LIVE_SYMBOL_CONFIGS["ETHUSDT"] = {
    "aux_symbols": ["BTCUSDT", "ETHBTC"],
    "models": [
        {"experiment_id": "EXP_ETH_A", "weight": 0.7},
        {"experiment_id": "EXP_ETH_B", "weight": 0.3},
    ],
}
```

## 环境变量

`.env` 用于配置外部服务密钥，不要提交到仓库。

```bash
GATE_DEMO_API_KEY=your_gate_sandbox_key
GATE_DEMO_SECRET=your_gate_sandbox_secret
PUSHPLUS_TOKEN=your_pushplus_token
DEEPSEEK_API_KEY=your_deepseek_key

DEEPSEEK_CONNECT_TIMEOUT=10
DEEPSEEK_READ_TIMEOUT=90
DEEPSEEK_MAX_TOKENS=1200
DEEPSEEK_THINKING_MODE=disabled
```

临时覆盖实验参数可以直接使用环境变量：

```bash
OPT_EXP_ID=EXP_TEST_001 OPT_D_MODEL=128 OPT_CLASS_THRESHOLD=0.03 python3 historical_pipeline/train_itransformer.py
```

## 常用命令

### 1. 初始化环境

```bash
bash setup.sh
```

### 2. 构建历史数据

```bash
python3 historical_pipeline/build_aggtrades.py
python3 historical_pipeline/build_main_features.py
```

### 3. 训练模型

```bash
OPT_EXP_ID=EXP_MY_RUN python3 historical_pipeline/train_itransformer.py
```

### 4. 生成动态阈值

```bash
OPT_EXP_ID=EXP_MY_RUN python3 backtest_pipeline/check_prob_distribution.py
```

### 5. Alpha 评估

```bash
OPT_EXP_ID=EXP_MY_RUN python3 backtest_pipeline/alpha_ic_evaluator.py
```

### 6. 单模型回测

```bash
OPT_EXP_ID=EXP_MY_RUN python3 backtest_pipeline/backtest_itransformer.py
```

### 7. 多模型回测

```bash
python3 backtest_pipeline/backtest_itransformer.py --models EXP_A:0.4,EXP_B:0.6
python3 backtest_pipeline/backtest_itransformer.py --use-live-models
```

### 8. 自动调参

```bash
python3 backtest_pipeline/auto_optimize.py --mode alpha --repeats 1
python3 backtest_pipeline/auto_optimize.py --mode alpha --repeats 3 --save-models
python3 backtest_pipeline/auto_optimize.py --mode backtest --base-model EXP_MY_RUN
```

### 9. 启动实盘链路

建议分终端启动：

```bash
python3 market_gateway.py
python3 live_pipeline.py
streamlit run 0_⚡_实盘监控指挥塔.py
```

如需要宏观情绪融合，定时或手动运行：

```bash
python3 sentiment_scraper.py
```

## 产物约定

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `data/train/processed/aggtrades_1h/{SYMBOL}/*.parquet` | `build_aggtrades.py` | 订单流小时特征 |
| `data/train/processed/main_features_1h/{SYMBOL}/*.parquet` | `build_main_features.py` | 训练宽表 |
| `model/itransformer_{SYMBOL}_best_{EXP}.pth` | `train_itransformer.py` | 模型权重与元数据 |
| `model/dynamic_thresholds_{EXP}.json` | `check_prob_distribution.py` | 分位数概率阈值 |
| `data/train/optimization_results_alpha.csv` | `auto_optimize.py --mode alpha` | Alpha 搜索结果 |
| `data/train/optimization_results_backtest.csv` | `auto_optimize.py --mode backtest` | 回测参数搜索结果 |
| `data/live/live_status.json` | `live_pipeline.py` | 实盘监控面板状态 |
| `data/live/gateway_state.db` | `market_gateway.py` | 行情网关本地记忆库 |
| `data/live/trading_intelligence_{SYMBOL}.db` | `execution_oms.py` | 交易流水与实盘审计 |
| `data/live/macro_intelligence.db` | `sentiment_scraper.py` | 宏观情绪评分 |

## 注意事项

- 历史数据和实盘特征必须保持同构。`live_pipeline.py` 中的实时特征构建逻辑需要与 `build_main_features.py` 的离线逻辑同步维护。
- 训练和验证使用 purged split；修改 `FORECAST_HORIZON` 后需要重新生成训练宽表、重新训练模型并重新生成动态阈值。
- 多模型 ensemble 要求各模型的 `forecast_horizon` 一致，否则回测会拒绝加载。
- `check_prob_distribution.py` 和 `alpha_ic_evaluator.py` 当前明确使用 `cuda`，没有 GPU 时需要改成自动设备选择或使用有 GPU 的环境。
- 实盘执行当前使用 `global_gate.set_sandbox_mode(True)`。切换真实账户前必须重新检查 API 权限、下单数量、止损覆盖、推送与熔断逻辑。
- 本项目是研究和工程框架，不构成投资建议。上线前需要额外进行样本外验证、极端行情回放、交易所异常演练和资金上限保护。
