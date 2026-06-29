# 快速使用指南

## 1. 安装环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写：

```text
TUSHARE_TOKEN=your_token_here
```

也可以临时在当前 PowerShell 会话中设置：

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

## 2. 启动应用

```powershell
streamlit run app.py
```

当前 Streamlit 入口是 `app.py`。它只负责页面路由，实际页面在 `app/pages/`：

- `training_page.py`：训练模型
- `prediction_page.py`：预测和策略回测
- `finetune_page.py`：模型微调
- `upload_page.py`：上传模型预测

## 3. 运行测试

```powershell
python -m pytest -q
```

常用测试命令：

```powershell
python -m pytest tests/unit/test_features/
python -m pytest tests/unit/test_trading/
python -m pytest --cov=ml_trader --cov-report=html
```

`htmlcov/`、`.coverage`、`.pytest_cache/` 都是运行产物，可以随时删除。

## 4. 数据约定

默认数据文件：

```text
完整数据.csv
```

该文件不进入版本管理。默认流程会优先复用本地数据，再通过 Tushare 补齐。

必要列：

```text
TradeDate, Open, High, Low, Close, Volume, Amount
```

其中 `Amount` 可缺失，核心流程至少需要日期和 OHLC。

## 5. 开发入口

核心模块：

- `ml_trader/data/preprocessor.py`：特征工程和峰谷标签
- `ml_trader/models/trainer.py`：模型训练
- `ml_trader/models/predictor.py`：预测和信号生成
- `ml_trader/trading/backtest.py`：交易构造和回测指标

应用层：

- `app/services/training_service.py`：多轮训练服务
- `app/services/prediction_service.py`：组合筛选和预测服务
- `app/utils/session_state.py`：Session state 初始化和访问
- `app/ui_helpers.py`：页面共享辅助函数

## 6. 当前验证状态

最近一次验证：

```text
23 passed
```

覆盖率仍偏低，主要测试集中在技术指标和回测模块。后续补测优先级：

1. `ml_trader/data/preprocessor.py`
2. `ml_trader/models/predictor.py`
3. `app/services/`
4. 训练到预测的最小集成测试
