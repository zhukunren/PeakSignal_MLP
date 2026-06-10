# 机器学习简化版

这是一个面向 A 股/指数行情的峰谷识别、组合筛选和回测实验项目。当前代码仍采用单目录脚本结构，核心流程是：

1. 读取本地 `完整数据.csv` 或通过 Tushare 补齐行情。
2. 使用 `preprocess.py` 和 `feature_expanded.py` 生成技术指标、形态特征和峰谷标签。
3. 使用 `train.py` 训练 Peak/Trough 两类模型。
4. 使用 `predict.py` 将峰谷预测转换为交易信号。
5. 使用 `backtest.py` 评估组合收益、超额收益、回撤、胜率和交易次数。
6. 使用 `run_cached_combo_training_check.py` 多轮训练并缓存每轮预测结果，再交叉组合筛选最佳峰/谷模型。

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
python run_cached_combo_training_check.py
```

## 数据文件

项目默认从根目录读取 `完整数据.csv`。该文件、派生训练集、模型、日志、HTML 图表和缓存目录都属于本地运行产物，已在 `.gitignore` 中排除。

`train.py` 默认不再写出 `简化版训练集.csv`。如需调试导出，可设置：

```powershell
$env:EXPORT_TRAINING_DATASET="1"
```

## 主要入口

- `app.py`：Streamlit 前端，支持行情读取、训练、预测、回测和模型导入导出。
- `run_cached_combo_training_check.py`：当前主要组合筛选脚本，支持缓存特征矩阵和每轮预测。
- `train_combo_round_worker.py`：组合筛选的单轮训练 worker。
- `run_fixed_feature_training_check.py`：固定特征版本的训练/组合检查脚本。
- `run_training_combo_check.py`：早期组合训练检查脚本。
- `save_base_round008_model.py`：从缓存中复现并导出指定轮次模型。

更详细的模块职责见 `docs/ARCHITECTURE.md`。
