# Event Regime Model 训练脚本详解

## 文件：`scripts/train_event_regime_model.py`

### 一、核心作用

这是一个**高级策略模型训练脚本**，用于训练**事件驱动 + 市场状态（Regime）**组合模型，目的是**提升基础峰谷预测模型的交易表现**。

---

## 二、设计理念

### 2.1 双层预测架构

```
┌─────────────────────────────────────────────────────────┐
│                   组合预测系统                            │
│                                                           │
│  ┌──────────────────┐        ┌──────────────────┐      │
│  │  基础模型层       │        │  事件模型层       │      │
│  │  (Base Model)    │        │  (Event Model)   │      │
│  ├──────────────────┤        ├──────────────────┤      │
│  │ Peak Model       │        │ Event Buy Model  │      │
│  │ Trough Model     │        │ Event Sell Model │      │
│  │                  │        │                  │      │
│  │ 训练数据:        │        │ 训练数据:        │      │
│  │ 2000-2020        │        │ 2000-2020        │      │
│  │                  │        │                  │      │
│  │ 预测方法:        │        │ 预测方法:        │      │
│  │ 分类概率         │        │ 回归评分         │      │
│  │ (0-1之间)        │        │ (连续值)         │      │
│  └──────────────────┘        └──────────────────┘      │
│           │                            │                 │
│           └────────────┬───────────────┘                 │
│                        ▼                                 │
│              ┌──────────────────┐                        │
│              │   信号融合层      │                        │
│              │ (Signal Fusion)  │                        │
│              ├──────────────────┤                        │
│              │ • 基础信号        │                        │
│              │ • 事件信号        │                        │
│              │ • Regime过滤     │                        │
│              │ • 信号抑制       │                        │
│              └──────────────────┘                        │
│                        │                                 │
│                        ▼                                 │
│              ┌──────────────────┐                        │
│              │  最终交易信号     │                        │
│              │ (Final Signals)  │                        │
│              └──────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 为什么需要事件模型？

基础的 Peak/Trough 分类模型存在局限：
- **标签稀疏**：只在局部极值点才标为1，大部分时间为0
- **前瞻性不足**：只看当前是否是峰谷，未考虑"未来N天能赚多少"
- **信号过于保守**：高阈值下交易次数少，错失机会

事件模型的创新：
- **前瞻性目标**：预测"未来10天的上涨潜力"和"下跌风险"
- **连续评分**：不是0/1二分类，而是回归评分（更灵活）
- **市场状态感知**：加入 Regime Gate（趋势过滤）

---

## 三、关键参数配置

### 3.1 时间分割（严格OOS验证）

```python
TRAIN_START = "20000101"  # 训练起始：2000年1月1日
TRAIN_END   = "20201231"  # 训练截止：2020年12月31日
PRED_START  = "20210101"  # 预测起始：2021年1月1日
PRED_END    = "20260608"  # 预测截止：2026年6月8日
```

**关键设计**：
- ✅ **严格的时间分割**：训练集与预测集完全不重叠
- ✅ **前瞻窗口保护**：训练样本的未来10天标签必须 ≤ 2020-12-31
- ✅ **避免未来泄露**：排除 `Ichimoku_Chikou`（一云后行线，使用未来数据）

### 3.2 事件目标定义

```python
EVENT_HORIZON = 10  # 前瞻10个交易日
```

#### 买入评分（Buy Target）

```python
buy_target = future_upside - 0.35 * future_drawdown + 0.20 * future_close_return
```

- `future_upside`：未来10天内的最高涨幅（相对次日开盘价）
- `future_drawdown`：未来10天内的最大回撤（相对次日开盘价）
- `future_close_return`：第10天收盘价的收益率

**含义**：在当前时刻买入，未来10天能获得多少**风险调整后收益**

#### 卖出评分（Sell Target）

```python
sell_target = future_drawdown - 0.25 * future_upside - 0.20 * future_close_return
```

**含义**：在当前时刻持有，未来10天会遭遇多少**潜在损失**

### 3.3 信号生成阈值

| 信号类型 | 阈值 | 窗口抑制 | 说明 |
|---------|------|----------|------|
| **基础买入** | `BASE_TROUGH_THRESHOLD = 0.54` | 20天 | 基础谷底概率 > 54% |
| **基础卖出** | `BASE_PEAK_THRESHOLD = 0.94` | 20天 | 基础峰顶概率 > 94% |
| **事件买入** | `EVENT_BUY_THRESHOLD = 0.038` | 40天 | 事件买入评分 > 0.038 |
| **事件卖出** | `EVENT_SELL_THRESHOLD = 0.039` | 40天 | 事件卖出评分 > 0.039 |

**窗口抑制机制**：
- 当产生一个信号后，在接下来的N天内屏蔽相同信号
- 避免频繁交易，降低交易成本

---

## 四、核心函数解析

### 4.1 `add_sequence_features()` - 时序特征扩展

```python
def add_sequence_features(df, base_features):
    """
    在基础特征上添加时序特征：
    - Lag特征：1, 3, 5, 10, 20天前的值
    - 统计特征：5, 10, 20, 60天的均值/最大值/最小值
    """
```

**扩展的核心特征**：
- `Return_5`, `Return_20`, `Return_60` → 多周期收益率
- `Drawdown_20`, `Drawdown_60` → 多周期回撤
- `Price_Position_20`, `Price_Position_60` → 价格相对位置
- `RSI_Signal`, `MACD_Diff_Pct`, `Bollinger_Position` → 技术指标
- `Volume_Ratio_20`, `ATR_14_Pct` → 成交量和波动率

**生成特征示例**：
```
Return_5_lag1, Return_5_lag3, Return_5_lag5, Return_5_lag10, Return_5_lag20
Return_5_mean5, Return_5_mean10, Return_5_mean20, Return_5_mean60
Return_5_min5, Return_5_min10, Return_5_min20, Return_5_min60
Return_5_max5, Return_5_max10, Return_5_max20, Return_5_max60
```

**特征总量**：基础特征 + 时序扩展 ≈ **几百个特征**

### 4.2 `build_event_targets()` - 事件目标构造

```python
def build_event_targets(df, horizon):
    """
    为每个交易日构造未来N天的事件目标
    
    步骤：
    1. 计算未来10天内的最高价和最低价
    2. 相对次日开盘价计算涨幅和回撤
    3. 加权组合成风险调整后的收益评分
    """
    next_open = df["Open"].shift(-1)
    future_high = pd.concat(
        [df["High"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    ).max(axis=1)
    future_low = pd.concat(
        [df["Low"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    ).min(axis=1)
    future_close = df["Close"].shift(-horizon)
    
    future_upside = future_high / next_open - 1
    future_drawdown = 1 - future_low / next_open
    future_close_return = future_close / next_open - 1
    
    buy_target = future_upside - 0.35 * future_drawdown + 0.20 * future_close_return
    sell_target = future_drawdown - 0.25 * future_upside - 0.20 * future_close_return
    
    return buy_target, sell_target
```

**示例**：
```
日期       | Open  | 未来10天High | 未来10天Low | 次日Open | Buy Target
---------|-------|-------------|------------|----------|------------
20200102 | 3000  | 3200        | 2950       | 3010     | 0.063 - 0.35*0.020 + 0.20*0.033 = 0.063
20200103 | 3020  | 3050        | 2900       | 3025     | 0.008 - 0.35*0.041 + 0.20*(-0.008) = -0.008
```

### 4.3 `train_event_models()` - 事件模型训练

```python
def train_event_models(df, features, horizon):
    """
    使用 HistGradientBoostingRegressor 训练两个独立的回归模型：
    - buy_model：预测买入评分
    - sell_model：预测卖出评分
    
    模型参数：
    - max_iter=160：最大迭代次数
    - learning_rate=0.035：学习率
    - max_leaf_nodes=7：每棵树最大叶子节点数（防止过拟合）
    - l2_regularization=0.5：L2正则化系数
    """
```

**训练集构造规则**：
```python
train_mask = (
    (df.index >= "2000-01-01")
    & (df.index <= "2020-12-31")
    & (future_end_dates <= "2020-12-31")  # 关键：未来标签不能超出训练期
    & buy_target.notna()
    & sell_target.notna()
)
```

### 4.4 `build_combined_predictions()` - 信号融合

```python
def build_combined_predictions(base_model, data_cache, event_df, event_features, buy_model, sell_model):
    """
    融合基础模型和事件模型的预测，生成最终交易信号
    
    融合逻辑：
    1. 基础信号 = 基础模型概率 > 阈值 + 窗口抑制
    2. 事件信号 = 事件模型评分 > 阈值 + Regime过滤 + 窗口抑制
    3. 最终信号 = 基础信号 OR 事件信号（取最大值）
    """
```

#### 关键代码解析

```python
# 1. 基础信号生成
base_buy = suppress_repeated_signals(
    base_trough_probability > BASE_TROUGH_THRESHOLD, 
    BASE_SIGNAL_WINDOW  # 20天抑制
)
base_sell = suppress_repeated_signals(
    base_peak_probability > BASE_PEAK_THRESHOLD, 
    BASE_SIGNAL_WINDOW
)

# 2. 事件信号生成（带Regime Gate）
regime_gate = event_df.loc[test_mask, "Close_MA200_Diff"].to_numpy(dtype=float) > 0
event_buy = suppress_repeated_signals(
    (event_buy_score >= EVENT_BUY_THRESHOLD) & regime_gate,  # 必须在牛市状态
    EVENT_SIGNAL_WINDOW  # 40天抑制
)
event_sell = suppress_repeated_signals(
    event_sell_score >= EVENT_SELL_THRESHOLD, 
    EVENT_SIGNAL_WINDOW
)

# 3. 信号融合（OR逻辑）
combined_buy = np.maximum(base_buy, event_buy)
combined_sell = np.maximum(base_sell, event_sell)
```

**Regime Gate 解释**：
```python
regime_gate = Close_MA200_Diff > 0
```
- `Close_MA200_Diff = Close / MA200 - 1`
- 当收盘价 > 200日均线时，`regime_gate = True`（牛市）
- **只在牛市状态下允许事件买入信号**，熊市禁止

### 4.5 `suppress_repeated_signals()` - 信号抑制

```python
def suppress_repeated_signals(signal, window):
    """
    信号抑制机制：当产生一个信号后，在接下来的N天内屏蔽相同信号
    
    示例：
    输入：[0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1]，window=3
    输出：[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
           ↑     ↑           ↑           ↑
          保留  抑制         保留         保留
    """
```

**作用**：
- 避免连续多天重复买入/卖出
- 降低交易频率和成本
- 让每次交易有充足的持仓时间

---

## 五、模型输出

### 5.1 保存的模型文件

```python
model_payload = {
    "model_type": "event_regime_hgbr_combo",
    "base_model": base_model,                    # 基础峰谷模型
    "event_buy_model": buy_model,                # 事件买入模型
    "event_sell_model": sell_model,              # 事件卖出模型
    "event_features": event_features,            # 时序扩展特征列表
    "params": {...},                             # 所有超参数
    "train_metadata": {...},                     # 训练集统计信息
    "bt_result": {...},                          # 回测结果
}
```

保存为 `saved_models/event_regime_hgbr_2021_present_model.pkl`

### 5.2 验证报告

```json
{
  "method": "HistGradientBoostingRegressor event-score model + MA200 regime gate + baseline high-confidence signals",
  "no_2021_to_2026_samples_in_training": true,
  "training_rule": "fit rows must be between 2000-01-01 and 2020-12-31, and every 10-day future label window must end no later than 2020-12-31",
  "excluded_future_like_features": ["Ichimoku_Chikou"],
  "bt_result": {
    "累计收益率": 0.xx,
    "超额收益率": 0.xx,
    "胜率": 0.xx,
    "交易笔数": xx,
    "最大回撤": -0.xx,
    "年化夏普比率": x.xx
  },
  "trades": [...]
}
```

保存为 `saved_models/event_regime_hgbr_2021_present_model_report.json`

---

## 六、使用场景

### 6.1 在 app.py 中使用

```python
# Tab4: 上传模型预测
if best_models.get("model_type") == "event_regime_hgbr_combo":
    base_result_up, base_bt_up, _ = predict_event_regime_model_data(
        raw_data_up,
        best_models,
        eval_mode=False,
        N_buy=1,
        N_sell=1,
        N_newhigh=60,
        enable_chase=False,
        enable_stop_loss=False,
        enable_change_signal=False,
        backtest_start_date=pred_start_up_str,
        backtest_end_date=pred_end_up_str,
    )
```

### 6.2 独立验证脚本

`scripts/strict_oos_event_validation.py`（严格OOS验证）

---

## 七、关键优势

### 7.1 相比基础模型的改进

| 维度 | 基础模型 | 事件模型 |
|------|---------|---------|
| **预测目标** | 当前是否是峰谷（0/1） | 未来10天能赚多少（连续值） |
| **信号密度** | 稀疏（只在极值点） | 更密集（基于风险收益评分） |
| **前瞻性** | 无 | 有（10天窗口） |
| **市场状态** | 无感知 | 有Regime Gate（牛熊过滤） |
| **交易频率** | 可能过低 | 通过融合增加有效信号 |

### 7.2 严格的OOS验证

```
训练集：2000-2020（含未来标签窗口保护）
测试集：2021-2026（完全未见过的数据）
```

- ✅ **时间顺序严格**：避免未来泄露
- ✅ **标签窗口保护**：训练样本的未来10天标签必须在训练期内
- ✅ **排除前瞻特征**：去除 `Ichimoku_Chikou`（一云后行线）

### 7.3 双重信号融合

```
最终信号 = 基础高置信信号 OR 事件机会信号
```

- **基础信号**：高阈值（94%/54%），保守但准确
- **事件信号**：捕捉基础模型遗漏的机会
- **OR融合**：互补而非冲突，增加交易机会

---

## 八、训练命令

### 8.1 使用默认参数

```powershell
python scripts/train_event_regime_model.py
```

### 8.2 自定义参数

```powershell
python scripts/train_event_regime_model.py `
    --base-model "my_base_model.pkl" `
    --data-cache "my_cache/prepared_data.pkl" `
    --output-model "saved_models/my_event_model.pkl" `
    --output-report "saved_models/my_event_report.json"
```

### 8.3 依赖文件

需要提前准备：
- `base_98pct_round008_model.pkl`：基础峰谷预测模型
- `fixed_feature_combo_cache/prepared_data.pkl`：预处理后的数据缓存

---

## 九、技术亮点

1. **HistGradientBoostingRegressor**：
   - 原生支持缺失值
   - 梯度提升速度快
   - 适合高维特征

2. **时序特征扩展**：
   - Lag特征：捕捉历史惯性
   - 统计特征：捕捉波动和趋势

3. **风险调整目标**：
   - 不只看涨幅，也考虑回撤
   - 符合真实交易的风险收益权衡

4. **信号抑制机制**：
   - 避免过度交易
   - 降低交易成本
   - 提升信号质量

5. **市场状态感知**：
   - Regime Gate 过滤熊市买入
   - 提升信号的环境适应性

---

## 十、总结

`train_event_regime_model.py` 是一个**生产级策略模型训练脚本**，通过以下创新提升量化交易表现：

✅ **双层架构**：基础模型（峰谷分类）+ 事件模型（风险收益回归）  
✅ **前瞻性目标**：预测未来10天的交易机会，而非当前状态  
✅ **时序特征**：Lag + 统计特征，捕捉市场动态  
✅ **信号融合**：OR逻辑组合，增加有效交易机会  
✅ **严格OOS**：训练测试完全分离，避免过拟合  
✅ **市场状态过滤**：Regime Gate 防止熊市买入  

这是一个将**机器学习**与**量化交易领域知识**深度结合的范例。

---

**生成时间**：2026-06-29  
**项目版本**：机器学习简化版 - 副本
