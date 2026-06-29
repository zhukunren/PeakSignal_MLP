# 机器学习简化版

面向 A 股/指数行情的峰谷识别、模型组合筛选和策略回测实验项目。

核心流程：

1. 读取本地 `完整数据.csv`，必要时通过 Tushare 补齐行情。
2. 生成技术指标、扩展特征和 Peak/Trough 峰谷标签。
3. 训练 Peak/Trough 两类模型，支持 MLP 和 Transformer。
4. 对多组峰/谷模型做组合筛选。
5. 将峰谷预测转换为交易信号。
6. 回测评估累计收益、超额收益、回撤、胜率和交易次数。

## 当前结构

```text
├── app.py                       # Streamlit 入口和页面路由
├── app/                         # Web 应用层
│   ├── pages/                   # 四个页面：训练、预测、微调、上传模型预测
│   ├── services/                # 训练和预测服务
│   ├── components/              # Streamlit 展示组件
│   ├── utils/                   # Session state 等工具
│   └── ui_helpers.py            # 页面共享辅助函数
├── config/
│   └── default.yaml             # 默认配置
├── docs/                        # 项目文档，入口见 docs/README.md
├── ml_trader/                   # 核心算法包
│   ├── data/                    # 数据加载和预处理
│   ├── features/                # 技术指标、形态识别、特征筛选
│   ├── models/                  # 模型结构、训练、预测
│   ├── trading/                 # 回测引擎
│   └── visualization/           # K 线和结果图表
├── tests/                       # Pytest 单元测试
├── saved_models/                # 可选模型产物说明
├── requirements.txt
└── .env.example
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

## 测试

```powershell
python -m pytest -q
```

当前基础测试覆盖技术指标和回测逻辑。覆盖率报告由 `pytest.ini` 配置生成，`htmlcov/` 和 `.coverage` 属于运行产物，不进入版本管理。

## 数据和产物

项目默认从根目录读取 `完整数据.csv`。该文件、派生训练集、模型、日志、HTML 图表和缓存目录属于本地运行产物，已在 `.gitignore` 中排除。

`ml_trader/models/trainer.py` 默认不再写出 `简化版训练集.csv`。如需调试导出，可设置：

```powershell
$env:EXPORT_TRAINING_DATASET="1"
```

## 文档

文档入口：[docs/README.md](docs/README.md)

建议阅读顺序：

1. [快速使用指南](docs/quick_start_guide.md)
2. [架构说明](docs/architecture.md)
3. [重构进度](docs/refactoring_progress_report.md)
4. [测试实施状态](docs/testing_implementation_report.md)
5. [配置管理](docs/configuration_management.md)
