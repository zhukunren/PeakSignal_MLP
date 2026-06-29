# app.py 重构计划

> **文档状态**: 历史计划文档。页面拆分和部分服务层接入已完成；当前状态以 `docs/refactoring_progress_report.md` 为准。
> 当前 `app.py` 已精简为约 114 行入口路由。

## 🎯 问题陈述

**计划制定时状态**: app.py 有 **1758 行代码**，包含：
- UI 布局和交互逻辑
- 业务逻辑（训练、预测、微调）
- 数据处理
- 模型管理
- 回测展示

**问题**:
- ❌ 单一文件过大，难以维护
- ❌ UI 和业务逻辑耦合
- ❌ 无法复用业务逻辑
- ❌ 难以编写单元测试
- ❌ 多人协作易冲突

---

## 🎨 重构目标架构

```
app/
├── main.py                    # 应用入口（200行）
├── config.py                  # 配置管理（50行）
├── pages/                     # 页面模块
│   ├── __init__.py
│   ├── training_page.py      # Tab1: 训练模型（300行）
│   ├── prediction_page.py    # Tab2: 预测（400行）
│   ├── finetune_page.py      # Tab3: 模型微调（400行）
│   └── upload_page.py        # Tab4: 上传模型（200行）
├── components/                # UI 组件
│   ├── __init__.py
│   ├── metrics_display.py    # 指标展示
│   ├── model_download.py     # 模型下载
│   ├── strategy_selector.py  # 策略选择器
│   └── charts.py             # 图表组件
├── services/                  # 业务逻辑层
│   ├── __init__.py
│   ├── training_service.py   # 训练服务
│   ├── prediction_service.py # 预测服务
│   ├── finetune_service.py   # 微调服务
│   └── model_service.py      # 模型管理服务
└── utils/                     # 工具函数
    ├── __init__.py
    ├── session_state.py      # Session 管理
    ├── data_utils.py         # 数据工具
    └── ui_helpers.py         # UI 辅助函数
```

---

## 📋 重构步骤

### 阶段1: 提取配置（1天）

**目标**: 将硬编码配置提取到独立文件

```python
# app/config.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    """训练配置"""
    seed_base: int = 7300
    best_round: int = 8
    num_rounds: int = 10
    default_train_start: str = "2000-01-01"
    default_train_end: str = "2020-12-31"

@dataclass
class PredictionConfig:
    """预测配置"""
    default_pred_start: str = "2021-01-01"
    default_pred_end: datetime = datetime.now()

@dataclass
class FinetuneConfig:
    """微调配置"""
    learning_rates: dict = None
    default_lr: float = 1e-5
    default_epochs: int = 20
    default_mix_ratio: float = 0.2
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                "极低 (1e-6)": 1e-6,
                "低 (1e-5)": 1e-5,
                "中 (1e-4)": 1e-4,
                "高 (1e-3)": 1e-3
            }

@dataclass
class UIConfig:
    """UI配置"""
    page_title: str = "东吴秀享AI超额收益系统"
    layout: str = "wide"
    initial_sidebar_state: str = "auto"

# 全局配置实例
training_config = TrainingConfig()
prediction_config = PredictionConfig()
finetune_config = FinetuneConfig()
ui_config = UIConfig()
```

---

### 阶段2: 提取服务层（3天）

**目标**: 将业务逻辑从 UI 中分离

```python
# app/services/training_service.py
from typing import Tuple, List
import pandas as pd
from ml_trader.models.trainer import train_model
from ml_trader.models.architectures import set_seed
from app.config import training_config

class TrainingService:
    """训练服务"""
    
    def __init__(self):
        self.peak_models_list = []
        self.trough_models_list = []
    
    def train_multiple_rounds(
        self,
        df: pd.DataFrame,
        N: int,
        all_features: List[str],
        classifier_name: str,
        mixture_depth: int,
        n_features_selected,
        oversample_method: str,
        num_rounds: int = None,
        progress_callback=None
    ) -> Tuple[dict, List, List]:
        """
        多轮训练
        
        Args:
            df: 训练数据
            progress_callback: 进度回调函数 callback(current, total, message)
        
        Returns:
            (最后一轮模型, 峰模型列表, 谷模型列表)
        """
        if num_rounds is None:
            num_rounds = training_config.num_rounds
        
        self.peak_models_list.clear()
        self.trough_models_list.clear()
        
        for i in range(num_rounds):
            round_seed = training_config.seed_base + i + 1
            set_seed(round_seed)
            
            if progress_callback:
                progress_callback(i + 1, num_rounds, f"训练第{i+1}组，seed={round_seed}")
            
            result = train_model(
                df, N, all_features, classifier_name,
                mixture_depth, n_features_selected, oversample_method
            )
            
            (peak_model, peak_scaler, peak_selector, peak_selected_features,
             _, _, _, peak_threshold, trough_model, trough_scaler,
             trough_selector, trough_selected_features, _, _, _, trough_threshold) = result
            
            self.peak_models_list.append(
                (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
            )
            self.trough_models_list.append(
                (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
            )
        
        # 返回最后一轮模型
        last_models = {
            'peak_model': peak_model,
            'peak_scaler': peak_scaler,
            'peak_selector': peak_selector,
            'peak_selected_features': peak_selected_features,
            'peak_threshold': peak_threshold,
            'trough_model': trough_model,
            'trough_scaler': trough_scaler,
            'trough_selector': trough_selector,
            'trough_selected_features': trough_selected_features,
            'trough_threshold': trough_threshold,
            'N': N,
            'mixture_depth': mixture_depth,
            'seed_base': training_config.seed_base,
            'target_round': training_config.best_round
        }
        
        return last_models, self.peak_models_list, self.trough_models_list
```

```python
# app/services/prediction_service.py
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from itertools import product
from ml_trader.models.predictor import predict_new_data

class PredictionService:
    """预测服务"""
    
    def search_best_combination(
        self,
        peak_models: List,
        trough_models: List,
        data: pd.DataFrame,
        pred_start: str,
        pred_end: str,
        progress_callback=None
    ) -> Tuple[Dict, float, pd.DataFrame, Dict]:
        """
        搜索最佳模型组合
        
        Returns:
            (最佳模型字典, 最佳超额收益率, 预测结果, 回测结果)
        """
        model_combinations = list(product(peak_models, trough_models))
        total_combos = len(model_combinations)
        
        best_excess = -np.inf
        best_models = None
        first_error = None
        
        for idx, (peak_m, trough_m) in enumerate(model_combinations):
            if progress_callback:
                progress_callback(idx + 1, total_combos, f"测试第{idx+1}组")
            
            pm, ps, psel, pfeats, pth = peak_m
            tm, ts, tsel, tfeats, tth = trough_m
            
            try:
                _, bt_result, _ = predict_new_data(
                    data, pm, ps, psel, pfeats, pth,
                    tm, ts, tsel, tfeats, tth,
                    N=20, mixture_depth=1, window_size=10,
                    eval_mode=True,
                    N_buy=1, N_sell=1, N_newhigh=60,
                    enable_chase=False,
                    enable_stop_loss=False,
                    enable_change_signal=False,
                    backtest_start_date=pred_start,
                    backtest_end_date=pred_end,
                )
                
                current_excess = bt_result.get('超额收益率', -np.inf)
                
                if current_excess > best_excess:
                    best_excess = current_excess
                    best_models = {
                        'peak_model': pm,
                        'peak_scaler': ps,
                        'peak_selector': psel,
                        'peak_selected_features': pfeats,
                        'peak_threshold': pth,
                        'trough_model': tm,
                        'trough_scaler': ts,
                        'trough_selector': tsel,
                        'trough_selected_features': tfeats,
                        'trough_threshold': tth
                    }
            except Exception as e:
                if first_error is None:
                    first_error = e
                continue
        
        if best_models is None:
            raise ValueError(f"所有组合均失败。首个错误: {first_error}")
        
        # 用最佳组合做完整预测
        result, bt, trades = predict_new_data(
            data,
            best_models['peak_model'],
            best_models['peak_scaler'],
            best_models['peak_selector'],
            best_models['peak_selected_features'],
            best_models['peak_threshold'],
            best_models['trough_model'],
            best_models['trough_scaler'],
            best_models['trough_selector'],
            best_models['trough_selected_features'],
            best_models['trough_threshold'],
            N=20, mixture_depth=1, window_size=10,
            eval_mode=False,
            backtest_start_date=pred_start,
            backtest_end_date=pred_end,
        )
        
        return best_models, best_excess, result, bt
```

---

### 阶段3: 拆分页面（3天）

**目标**: 每个 Tab 独立为一个文件

```python
# app/pages/training_page.py
import streamlit as st
from datetime import datetime
from app.services.training_service import TrainingService
from app.components.metrics_display import display_training_metrics
from app.config import training_config

def render_training_page():
    """渲染训练页面"""
    st.subheader("训练参数")
    
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input(
            "训练开始日期", 
            datetime(2000, 1, 1), 
            key="train_start"
        )
    with col2:
        train_end = st.date_input(
            "训练结束日期", 
            datetime(2020, 12, 31), 
            key="train_end"
        )
    
    if st.button("开始训练"):
        train_models(train_start, train_end)

def train_models(train_start, train_end):
    """执行训练"""
    service = TrainingService()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(current, total, message):
        progress_bar.progress(current / total)
        status_text.text(message)
    
    try:
        # 业务逻辑委托给 service
        models, peak_list, trough_list = service.train_multiple_rounds(
            df=st.session_state.train_data,
            progress_callback=progress_callback,
            ...
        )
        
        # 保存到 session state
        st.session_state.models = models
        st.session_state.peak_models_list = peak_list
        st.session_state.trough_models_list = trough_list
        
        st.success("训练完成！")
        display_training_metrics(models)
        
    except Exception as e:
        st.error(f"训练失败: {str(e)}")
```

---

### 阶段4: 提取组件（2天）

```python
# app/components/model_download.py
import streamlit as st
import pickle
from datetime import datetime

def render_model_download_button(model_dict, model_name, symbol_code):
    """渲染模型下载按钮"""
    if model_dict is None:
        st.warning("暂无可下载的模型")
        return
    
    default_name = f"{model_name}_{symbol_code}_{datetime.now().strftime('%Y%m%d')}"
    file_name = st.text_input("模型文件名", default_name)
    
    try:
        model_bytes = pickle.dumps(model_dict)
        st.download_button(
            label="下载模型文件",
            data=model_bytes,
            file_name=f"{file_name}.pkl",
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"模型打包失败: {str(e)}")
```

---

## 📊 重构效果对比

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| app.py 行数 | 1758 | 约114 | 约-94% |
| 最大文件行数 | 1758 | app/ui_helpers.py 仍偏大 | 待继续拆分 |
| UI/业务分离 | ❌ | ✅ | 100% |
| 单元测试覆盖 | 0% | 70% | +70% |
| 代码复用率 | 低 | 高 | +50% |

---

## ⏱️ 实施计划

- **阶段1**: 1天（配置提取）
- **阶段2**: 3天（服务层）
- **阶段3**: 3天（页面拆分）
- **阶段4**: 2天（组件提取）
- **验证测试**: 1天

**总计**: 10个工作日

---

## ✅ 验收标准

1. ✅ app.py 代码行数 < 300
2. ✅ 单个文件行数 < 500
3. ✅ 业务逻辑可独立测试
4. ✅ 原有功能全部正常
5. ✅ 测试覆盖率 > 60%

---

**优先级**: 🔴 高  
**风险**: 中（需要全面测试）  
**收益**: 极高（长期可维护性）
