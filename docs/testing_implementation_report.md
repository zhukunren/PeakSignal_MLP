# 测试实施状态

**日期**: 2026-06-29  
**当前状态**: 基础单元测试已建立，现有测试全部通过。

## 当前结果

最近一次运行：

```powershell
python -m pytest -q
```

结果：

```text
23 passed
```

现有测试集中在两个区域：

```text
tests/
├── conftest.py
├── unit/
│   ├── test_features/
│   │   └── test_indicators.py
│   └── test_trading/
│       └── test_backtest.py
└── __init__.py
```

## 覆盖情况

当前整体覆盖率约为 14%。覆盖率低的原因是训练、预测、预处理和应用服务层尚未充分测试。

已覆盖较多的模块：

- `ml_trader/trading/backtest.py`
- `ml_trader/features/indicators.py`

待补测模块：

- `ml_trader/data/preprocessor.py`
- `ml_trader/models/predictor.py`
- `ml_trader/models/trainer.py`
- `app/services/training_service.py`
- `app/services/prediction_service.py`

## 常用命令

```powershell
python -m pytest -q
python -m pytest tests/unit/test_features/
python -m pytest tests/unit/test_trading/
python -m pytest --cov=ml_trader --cov-report=html
```

`htmlcov/`、`.coverage` 和 `.pytest_cache/` 是测试产物，不应提交。

## 下一步

优先补四类测试：

1. 预处理最小样本测试：验证 `preprocess_data` 能生成核心特征和标签。
2. 预测服务测试：用轻量 mock 模型覆盖组合搜索失败和成功路径。
3. Session state 测试：验证默认键初始化完整。
4. 最小端到端测试：用小样本或 mock 模型跑通预测到回测。

目标不是立即追求高覆盖率，而是先覆盖最容易回归的主流程边界。
