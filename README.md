# 机器学习简化版

这是一个面向 A 股/指数行情的峰谷识别、组合筛选和回测实验项目。核心流程是：

1. 读取本地 `完整数据.csv` 或通过 Tushare 补齐行情。
2. 使用 `src/preprocess.py` 和 `src/feature_expanded.py` 生成技术指标、形态特征和峰谷标签。
3. 使用 `src/train.py` 训练 Peak/Trough 两类模型。
4. 使用 `src/predict.py` 将峰谷预测转换为交易信号。
5. 使用 `src/backtest.py` 评估组合收益、超额收益、回撤、胜率和交易次数。
6. 使用 `scripts/run_cached_combo_training_check.py` 多轮训练并缓存每轮预测结果，再交叉组合筛选最佳峰/谷模型。

## 项目结构

```
├── src/                    # 核心模块
│   ├── models.py          # 模型定义 (Transformer, MLP)
│   ├── preprocess.py      # 数据预处理和特征工程
│   ├── feature_expanded.py # 扩展技术指标
│   ├── function.py        # 基础技术指标函数
│   ├── train.py           # 模型训练
│   ├── predict.py         # 预测和信号生成
│   ├── backtest.py        # 回测引擎
│   ├── tushare_function.py # Tushare 数据获取
│   ├── plot_candlestick.py # K线图绘制
│   └── filter_feature.py  # 特征过滤
├── scripts/               # 执行脚本
│   ├── run_cached_combo_training_check.py  # 主要组合筛选脚本
│   ├── run_fixed_feature_training_check.py # 固定特征训练检查
│   ├── run_training_combo_check.py         # 早期组合训练检查
│   ├── train_combo_round_worker.py         # 单轮训练 worker
│   ├── save_base_round008_model.py         # 模型导出工具
│   ├── generate_best_combo_chart.py        # 最佳组合图表生成
│   ├── generate_base_best_cached_chart.py  # 基准图表生成
│   └── batchtraining.py                    # 批量训练脚本
├── app.py                 # Streamlit Web 界面
├── requirements.txt       # Python 依赖
├── .env.example          # 环境变量模板
└── README.md             # 项目文档
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

`src/train.py` 默认不再写出 `简化版训练集.csv`。如需调试导出，可设置：

```powershell
$env:EXPORT_TRAINING_DATASET="1"
```
