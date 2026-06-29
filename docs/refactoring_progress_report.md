# 重构进度

**日期**: 2026-06-29  
**状态**: 页面拆分、服务层接入和基础去重已完成。

## 已完成

### 1. 应用入口精简

`app.py` 已精简为 Streamlit 入口和路由文件，主要职责：

- 设置 Streamlit 页面配置。
- 渲染侧栏参数。
- 创建四个 Tab。
- 调用 `app/pages/` 下的页面模块。

### 2. 页面层拆分

```text
app/pages/
├── training_page.py
├── prediction_page.py
├── finetune_page.py
└── upload_page.py
```

页面层负责 UI 编排，核心训练和预测逻辑逐步下沉到 service 层。

### 3. 服务层接入

```text
app/services/
├── training_service.py
└── prediction_service.py
```

已接入：

- 训练页调用 `TrainingService.train_multiple_rounds`。
- 预测页的多组合搜索调用 `PredictionService.search_best_combination`。

### 4. 状态管理集中

`SessionState` 默认值已集中到：

```text
app/utils/session_state.py
```

`app.py` 调用 `SessionStateManager.initialize()` 初始化状态。

### 5. 删除旧注释实现

已清理不会影响运行行为的旧代码块：

- `ml_trader/models/predictor.py` 中注释保留的旧版 `predict_new_data`。
- `ml_trader/models/architectures.py` 中注释保留的旧版模型构造函数。
- `ml_trader/trading/backtest.py` 中注释保留的旧版交易构造函数。

## 当前结构

```text
app.py
app/
├── pages/
├── services/
├── components/
├── utils/
└── ui_helpers.py

ml_trader/
├── data/
├── features/
├── models/
├── trading/
└── visualization/
```

## 验证结果

已完成：

- Python 语法编译通过。
- 当前单元测试通过：`23 passed`。
- Streamlit 健康检查通过：`/_stcore/health` 返回 `200 ok`。

## 剩余风险

- `app/ui_helpers.py` 仍偏大，后续可继续拆到 `app/components/` 和 `app/services/`。
- 微调页仍包含较多业务逻辑，建议后续提取 `FinetuneService`。
- 上传模型预测页和普通预测页仍有重复策略控件，可抽出复用组件。
- 训练、预测和预处理测试覆盖率仍偏低。

## 下一步

建议按以下顺序继续：

1. 为 `TrainingService` 和 `PredictionService` 增加单元测试。
2. 将预测页和上传页的策略控件抽为组件。
3. 将微调流程提取到 `app/services/finetune_service.py`。
4. 拆分 `app/ui_helpers.py`，保留少量跨页面辅助函数。
5. 为 `preprocess_data` 和 `predict_new_data` 建立最小集成测试。
