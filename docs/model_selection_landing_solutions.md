# 模型组合筛选落地问题和解决方案

## 目标

本文档用于指导后续改造训练、模型组合筛选和逐日预测流程。当前系统的核心使用方式是：

```text
2000-2020 训练 Peak/Trough 候选模型
2021 起逐日用已发生行情评估候选组合
选出截至当日表现最优的 Peak/Trough 组合
用当日收盘后已知数据生成信号
下一交易日开盘执行买入或卖出
```

这个方向可以继续保留。需要落地解决的是以下三类问题：

1. 高点模型和低点模型可能需要不同特征体系。
2. 组合筛选不应只依赖回测后的超额收益率，应更贴近“高低点附近有信号、中段误报少”的实盘目标。
3. 2000-2020 训练样本距今较远，需要让模型吸收近年市场变化，同时不明显损失既有性能。

## 当前实现概况

相关代码位置：

- `app/services/training_service.py`：多 seed 训练 Peak/Trough 候选模型。
- `ml_trader/models/trainer.py`：单标签模型训练、特征过滤、过采样、阈值优化。
- `app/services/prediction_service.py`：10x10 组合搜索和最佳组合选择。
- `ml_trader/models/predictor.py`：模型预测、信号后处理、回测入口。
- `ml_trader/trading/backtest.py`：交易构造和回测指标。
- `ml_trader/data/preprocessor.py`：特征工程和 Peak/Trough 标签生成。

目前系统已经分别保存 `peak_selected_features` 和 `trough_selected_features`，因此数据结构上支持 Peak/Trough 使用不同特征。主要不足在于筛选逻辑仍偏通用，组合评分也主要以 `超额收益率` 为核心。

## 问题一：Peak/Trough 特征体系不应强制一致

### 现状

训练时 Peak 和 Trough 会分别调用 `train_model_for_label`。但当使用自动特征时，当前逻辑基本保留相关性过滤后的特征集合，Peak/Trough 的特征差异不够明确。

这会带来两个问题：

- 顶部信号更依赖超涨、放量滞涨、新高衰竭、上影线、趋势末端等信息。
- 底部信号更依赖超跌、回撤、低位反转、恐慌放量、下影线、跌幅衰竭等信息。

两类信号共享同一套筛选逻辑，可能导致模型学习目标被稀释。

### 解决方案

新增标签专属特征筛选配置，至少拆成：

```text
Peak 特征池
Trough 特征池
通用特征池
```

建议先不删除现有特征，而是在筛选阶段加入标签偏好：

```python
FEATURE_GROUPS = {
    "peak": [
        "Price_Position_20",
        "Price_Position_60",
        "New_High_20",
        "New_High_60",
        "High_From_Close_20",
        "Upper_Shadow_Pct",
        "RSI_Signal",
        "Bollinger_Position",
        "Volume_Ratio_20",
        "ATR_14_Pct",
    ],
    "trough": [
        "Drawdown_20",
        "Drawdown_60",
        "New_Low_20",
        "New_Low_60",
        "Close_From_Low_20",
        "Lower_Shadow_Pct",
        "RSI_Signal",
        "Bollinger_Position",
        "Volume_Ratio_20",
        "ATR_14_Pct",
    ],
}
```

落地时不要一次性把特征池写死为唯一特征集合，而是作为优先候选集。这样可以保留原模型能力，降低回归风险。

### 改造步骤

1. 在 `config/default.yaml` 增加 Peak/Trough 独立特征选择配置：

```yaml
features:
  selection:
    peak:
      mode: "hybrid"
      max_features: 40
      preferred_groups: ["peak", "common"]
    trough:
      mode: "hybrid"
      max_features: 40
      preferred_groups: ["trough", "common"]
```

2. 新增 `ml_trader/features/feature_groups.py`，定义特征组。

3. 修改 `train_model_for_label`：

- 根据 `label_column` 读取对应特征池。
- 对候选特征先做可用性过滤。
- 再做现有相关性过滤、方差过滤、目标相关性排序。
- Peak/Trough 分别得到 `selected_features`。

4. UI 层后续可增加：

```text
Peak 特征数量
Trough 特征数量
Peak 特征策略
Trough 特征策略
```

第一阶段可以不改 UI，只走配置默认值。

### 验收标准

- 训练产物中 `peak_selected_features` 和 `trough_selected_features` 可以不同。
- Peak/Trough 特征数量、特征名称在训练日志中清晰输出。
- 旧流程仍可运行，不传新配置时使用当前逻辑。
- 单元测试覆盖：给定一份模拟特征表，Peak/Trough 能按不同候选池筛出不同特征。

## 问题二：组合筛选指标不应只看超额收益率

### 现状

`PredictionService.search_best_combination` 当前主要用 `bt_result['超额收益率']` 选择最佳组合。

这会带来偏差：

- 少数大收益交易可能主导选择结果。
- 模型不一定真的稳定识别高低点附近。
- 上涨或下跌中段的错误信号没有被直接惩罚。
- 同一个高低点附近重复触发多个信号没有被单独约束。

实盘目标不是精确买在最低点、卖在最高点，而是：

```text
低点附近尽量有买入信号
高点附近尽量有卖出信号
趋势中段错误反向信号尽量少
信号不要过度密集
回测收益和回撤仍要可接受
```

### 解决方案

新增信号质量评分，与回测评分一起组成综合评分。

建议新增模块：

```text
ml_trader/evaluation/signal_quality.py
```

核心函数：

```python
evaluate_peak_signal_quality(result_df, tolerance_days=5)
evaluate_trough_signal_quality(result_df, tolerance_days=5)
evaluate_combo_quality(result_df, bt_result, weights=None)
```

### 信号质量定义

以 `tolerance_days = 5` 为例：

- 真实 Peak 前后 5 个交易日内出现 `Peak_Prediction=1`，视为 Peak 附近命中。
- 真实 Trough 前后 5 个交易日内出现 `Trough_Prediction=1`，视为 Trough 附近命中。
- 预测信号落在任意真实事件窗口内，视为有效信号。
- 预测信号不在任何真实 Peak/Trough 附近，视为中段误报。
- 同一事件窗口内多次触发，只记一次命中，多余信号计入重复惩罚。

建议输出指标：

```python
{
    "peak_event_recall": 0.0,
    "trough_event_recall": 0.0,
    "peak_signal_precision": 0.0,
    "trough_signal_precision": 0.0,
    "mid_zone_false_signal_rate": 0.0,
    "duplicate_signal_penalty": 0.0,
    "avg_peak_distance": None,
    "avg_trough_distance": None,
}
```

### 综合评分建议

第一版可使用可解释的线性评分：

```text
SignalScore =
    0.30 * trough_event_recall
  + 0.25 * peak_event_recall
  + 0.20 * trough_signal_precision
  + 0.15 * peak_signal_precision
  - 0.25 * mid_zone_false_signal_rate
  - 0.10 * duplicate_signal_penalty
```

再结合回测：

```text
FinalScore =
    0.60 * SignalScore
  + 0.25 * normalized_excess_return
  + 0.10 * normalized_sharpe
  - 0.20 * normalized_max_drawdown
```

第一阶段也可以更保守：

```text
先过滤，再排序
```

过滤条件示例：

```text
trough_event_recall >= 0.45
peak_event_recall >= 0.35
mid_zone_false_signal_rate <= 0.40
交易笔数 >= 3
最大回撤 >= -0.35
```

过滤后再按超额收益率或综合评分排序。这样比直接重写评分体系更稳。

### 改造步骤

1. 新增 `ml_trader/evaluation/signal_quality.py`。

2. 在 `predict_new_data(..., eval_mode=True)` 的组合搜索中保留真实标签：

- 当前 `eval_mode=True` 会在预处理阶段生成 `Peak` 和 `Trough`。
- 这正好可用于计算附近命中率。

3. 修改 `PredictionService.search_best_combination`：

- 每个组合先得到 `result_df`、`bt_result`。
- 调用 `evaluate_combo_quality(result_df, bt_result)`。
- 用 `final_score` 选择最佳组合。
- 同时保存 `quality_metrics` 到 `best_models` 或返回值中。

4. UI 展示新增：

```text
低点附近命中率
高点附近命中率
低点信号精确率
高点信号精确率
中段误报率
重复信号惩罚
综合评分
```

### 验收标准

- 组合搜索不再只能按 `超额收益率` 排序。
- 组合搜索结果能解释为什么某个组合被选中。
- 对构造样本测试：
  - 信号落在真实低点附近时，低点命中率上升。
  - 信号落在非事件区间时，中段误报率上升。
  - 同一事件窗口重复信号会被惩罚。

## 问题三：训练样本需要吸收近年行情

### 现状

当前主流程默认使用 2000-2020 训练候选模型。这个设计保留了长期样本，但距当前市场已经有多年距离。

近年市场变化可能影响模型表现：

- 注册制和交易结构变化。
- 行业风格切换速度加快。
- 指数结构、ETF、量化交易影响增强。
- 2021 以后部分行情特征与 2000-2020 不完全一致。

直接把旧模型替换为近年模型风险较高，因为旧模型可能仍在部分市场状态下有效。

### 解决方案

不要用单一训练窗口替换现有模型。改为建立多训练窗口候选池：

```text
A组：2000-2020 基础模型
B组：2000-最近可训练日 长周期扩展模型
C组：2010-最近可训练日 中周期模型
D组：2016-最近可训练日 近期模型
E组：2000-2020 基础模型 + 近年低学习率微调模型
```

组合搜索从原来的：

```text
10 Peak x 10 Trough
```

扩展为：

```text
训练窗口 x seed x Peak/Trough
```

最终由组合评分机制决定当前市场更适合哪一类模型。

### 近年数据加入方式

优先级从低风险到高风险：

1. 新增近年训练窗口模型，不删除旧模型。
2. 对阈值做滚动校准，而不是频繁重训权重。
3. 对旧模型做低学习率微调，并混入旧数据。
4. 使用样本时间权重，让近年样本权重更高。
5. 定期滚动重训，例如每月或每季度，而不是每天重训。

### 推荐训练窗口

第一阶段建议先做 3 类：

```text
base_2000_2020
mid_2010_latest_confirmed
recent_2016_latest_confirmed
```

其中 `latest_confirmed` 要考虑峰谷标签需要后验确认。如果 `N=20`，最新训练标签至少应回退 20 个交易日，避免最近一段标签尚未确认。

### 模型产物结构建议

当前模型 tuple 结构较简洁，但扩展多窗口后需要增加元数据：

```python
{
    "model": model,
    "scaler": scaler,
    "selector": selector,
    "selected_features": selected_features,
    "threshold": threshold,
    "label": "Peak",
    "train_window": "2016-latest_confirmed",
    "seed": 7308,
    "feature_policy": "hybrid_peak_v1",
    "model_family": "MLP",
}
```

短期可以继续兼容 tuple，但下载和组合搜索建议逐步迁移到字典结构。

### 改造步骤

1. 增加训练窗口配置：

```yaml
training:
  windows:
    - name: "base_2000_2020"
      start: "2000-01-01"
      end: "2020-12-31"
    - name: "mid_2010_latest"
      start: "2010-01-01"
      end: "latest_confirmed"
    - name: "recent_2016_latest"
      start: "2016-01-01"
      end: "latest_confirmed"
```

2. 扩展 `TrainingService`：

- 支持按多个训练窗口循环训练。
- 每个模型保存 `train_window` 元数据。
- 保留现有单窗口训练作为默认兼容路径。

3. 扩展 `PredictionService`：

- 支持跨训练窗口候选模型组合搜索。
- 组合评分返回所选模型来自哪个训练窗口。

4. 增加模型池下载：

- 保存全部训练窗口候选模型。
- 保存当前选中的最佳组合。
- 保存组合评分和选择依据。

### 验收标准

- 旧的 2000-2020 单窗口流程仍可运行。
- 新流程能训练至少两个窗口的候选模型。
- 组合搜索结果能展示所选 Peak/Trough 模型来源窗口。
- 在同一测试区间中，旧模型、新模型、混合候选池三者能并排比较。

## 推荐实施顺序

### 阶段一：先改组合评分

优先级最高。原因是它直接决定系统选什么模型，也最贴近实盘目标。

交付内容：

- 新增 `ml_trader/evaluation/signal_quality.py`。
- `PredictionService.search_best_combination` 支持综合评分。
- UI 展示信号质量指标。
- 增加信号质量单元测试。

### 阶段二：拆分 Peak/Trough 特征选择

交付内容：

- 新增 `feature_groups.py`。
- Peak/Trough 独立候选特征池。
- 训练日志输出两类模型的实际特征。
- 增加特征筛选单元测试。

### 阶段三：引入多训练窗口模型池

交付内容：

- 配置化训练窗口。
- 训练服务支持多窗口训练。
- 组合搜索支持跨窗口候选模型。
- 模型下载文件包含训练窗口和评分元数据。

### 阶段四：近年微调和阈值滚动校准

交付内容：

- 按月或按季度校准阈值。
- 低学习率微调旧模型。
- 新旧模型池并行验证。

## 风险和控制

### 风险一：综合评分过度复杂

控制方式：

- 第一版使用简单线性评分。
- 所有子指标都单独展示。
- 保留按超额收益率排序的对照结果。

### 风险二：多训练窗口导致候选组合过多

控制方式：

- 每个窗口只保留 Top K Peak 和 Top K Trough。
- 先按单模型信号质量预筛，再做组合回测。
- 限制第一阶段窗口数量为 3 个。

### 风险三：近年训练导致模型遗忘长期规律

控制方式：

- 不删除 2000-2020 基础模型。
- 近年模型只是候选池成员。
- 微调时混入旧数据，且使用较低学习率。

### 风险四：信号质量指标与收益不一致

控制方式：

- 不完全放弃回测指标。
- 使用信号质量 + 回测表现的综合评分。
- 展示两个维度，避免只看单一总分。

## 最小可落地版本

如果只做一个最小版本，建议范围如下：

1. 新增信号质量评分模块。
2. 组合搜索时返回：

```text
超额收益率
高点附近命中率
低点附近命中率
中段误报率
综合评分
```

3. 默认按综合评分选组合。
4. UI 中保留“按超额收益率选组合”和“按综合评分选组合”两个选项。

这样改动最小，但能显著降低“只按收益选组合”的偏差，并为后续 Peak/Trough 特征拆分和多训练窗口模型池打基础。
