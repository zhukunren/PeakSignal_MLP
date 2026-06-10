# 机器学习简化版

这是一个面向 A 股/指数行情的峰谷识别、组合筛选和回测实验项目。采用模块化架构，按职责分离数据、特征、模型、交易和可视化。

核心流程：
1. 读取本地 `完整数据.csv` 或通过 Tushare 补齐行情
2. 使用 62+ 技术指标和形态识别生成特征和峰谷标签
3. 训练 Peak/Trough 两类模型（MLP 或 Transformer）
4. 将峰谷预测转换为交易信号
5. 回测评估：收益、超额收益、回撤、胜率、交易次数
6. 多轮训练组合筛选最佳模型

## 项目结构

```
├── ml_trader/                   # 核心包
│   ├── config.py               # 全局配置
│   ├── data/                   # 数据模块
│   │   ├── loader.py          # Tushare 数据加载
│   │   └── preprocessor.py    # 数据预处理
│   ├── features/               # 特征工程
│   │   ├── indicators.py      # 技术指标计算（62个函数）
│   │   ├── patterns.py        # 峰谷识别
│   │   ├── engineering.py     # 扩展特征生成
│   │   └── selector.py        # 特征过滤选择
│   ├── models/                 # 模型模块
│   │   ├── architectures.py   # 模型定义 (Transformer, MLP)
│   │   ├── trainer.py         # 训练逻辑
│   │   └── predictor.py       # 预测和信号生成
│   ├── trading/                # 交易模块
│   │   └── backtest.py        # 回测引擎
│   └── visualization/          # 可视化
│       └── plots.py           # K线图绘制
├── scripts/                    # 执行脚本
│   ├── run_cached_combo_training_check.py
│   ├── run_fixed_feature_training_check.py
│   ├── train_combo_round_worker.py
│   └── ...
├── app.py                      # Streamlit Web 界面
├── requirements.txt            # Python 依赖
└── .env.example               # 环境变量模板
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 或当前 shell 中设置 `TUSHARE_TOKEN` 后运行：

```powershell
streamlit run app.py
```

组合筛选脚本示例：

```powershell
$env:COMBO_NUM_ROUNDS="20"
$env:COMBO_RUN_TAG="dev"
python scripts/run_cached_combo_training_check.py
```

## 数据文件

项目默认从根目录读取 `完整数据.csv`。该文件、派生训练集、模型、日志、HTML 图表和缓存目录都属于本地运行产物，已在 `.gitignore` 中排除。

`ml_trader/models/trainer.py` 默认不再写出 `简化版训练集.csv`。如需调试导出，可设置：

```powershell
$env:EXPORT_TRAINING_DATASET="1"
```
