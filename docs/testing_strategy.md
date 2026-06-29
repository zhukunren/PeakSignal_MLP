# 测试策略与实施方案

## 🎯 目标

为项目建立完整的测试体系，提升代码质量和可维护性。

---

## 📁 测试目录结构

```
tests/
├── __init__.py
├── conftest.py                 # pytest 配置和 fixtures
├── fixtures/                   # 测试数据和工具
│   ├── sample_data.csv        # 示例行情数据
│   ├── test_models.pkl        # 测试用模型
│   └── fixtures.py            # 共享 fixtures
├── unit/                       # 单元测试
│   ├── test_data/
│   │   ├── test_loader.py
│   │   └── test_preprocessor.py
│   ├── test_features/
│   │   ├── test_indicators.py
│   │   ├── test_patterns.py
│   │   └── test_selector.py
│   ├── test_models/
│   │   ├── test_architectures.py
│   │   └── test_trainer.py
│   └── test_trading/
│       └── test_backtest.py
├── integration/                # 集成测试
│   ├── test_train_pipeline.py
│   ├── test_predict_pipeline.py
│   └── test_backtest_pipeline.py
└── e2e/                        # 端到端测试
    └── test_full_workflow.py
```

---

## 🔧 关键测试用例

### 1. 特征工程测试

```python
# tests/unit/test_features/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from ml_trader.features.indicators import compute_RSI, compute_MACD

class TestRSI:
    def test_rsi_range(self):
        """RSI应该在0-100之间"""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        rsi = compute_RSI(prices, period=6)
        assert rsi.dropna().between(0, 100).all()
    
    def test_rsi_overbought(self):
        """持续上涨应该产生>70的RSI"""
        prices = pd.Series(range(100, 120))
        rsi = compute_RSI(prices, period=14)
        assert rsi.iloc[-1] > 70
    
    def test_rsi_with_nan(self):
        """处理缺失值"""
        prices = pd.Series([100, np.nan, 102, 103])
        rsi = compute_RSI(prices, period=3)
        assert not rsi.isna().all()

class TestMACD:
    def test_macd_components(self):
        """MACD应该返回两个序列"""
        prices = pd.Series(range(100, 150))
        macd, signal = compute_MACD(prices)
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
    
    def test_macd_crossover(self):
        """验证金叉死叉逻辑"""
        prices = pd.Series([100] * 20 + list(range(100, 120)))
        macd, signal = compute_MACD(prices)
        # 上涨阶段 MACD 应该高于信号线
        assert (macd.iloc[-5:] > signal.iloc[-5:]).all()
```

---

### 2. 数据预处理测试

```python
# tests/unit/test_data/test_preprocessor.py
import pytest
from ml_trader.data.preprocessor import preprocess_data
import pandas as pd

@pytest.fixture
def sample_market_data():
    """生成测试用行情数据"""
    return pd.DataFrame({
        'TradeDate': pd.date_range('2020-01-01', periods=100),
        'Open': 3000 + np.random.randn(100).cumsum(),
        'High': 3010 + np.random.randn(100).cumsum(),
        'Low': 2990 + np.random.randn(100).cumsum(),
        'Close': 3000 + np.random.randn(100).cumsum(),
        'Volume': 100000 + np.random.randint(-1000, 1000, 100).cumsum(),
    })

class TestPreprocessor:
    def test_feature_generation(self, sample_market_data):
        """验证特征生成"""
        processed, features = preprocess_data(
            sample_market_data, 
            N=20, 
            mixture_depth=1,
            mark_labels=True
        )
        
        # 应该包含基础特征
        assert 'MA_5' in processed.columns
        assert 'RSI_14' in processed.columns
        assert 'MACD' in processed.columns
        
        # 特征列表应该不为空
        assert len(features) > 0
    
    def test_label_marking(self, sample_market_data):
        """验证峰谷标注"""
        processed, _ = preprocess_data(
            sample_market_data, 
            N=10, 
            mixture_depth=1,
            mark_labels=True
        )
        
        assert 'Peak' in processed.columns
        assert 'Trough' in processed.columns
        # 峰谷标签应该是0或1
        assert processed['Peak'].isin([0, 1]).all()
```

---

### 3. 模型训练测试

```python
# tests/unit/test_models/test_trainer.py
import pytest
from ml_trader.models.trainer import train_model

@pytest.fixture
def training_data(sample_market_data):
    """准备训练数据"""
    from ml_trader.data.preprocessor import preprocess_data
    processed, features = preprocess_data(
        sample_market_data, 
        N=10, 
        mixture_depth=1,
        mark_labels=True
    )
    return processed, features

class TestTrainer:
    def test_model_training(self, training_data):
        """验证模型可以训练"""
        df, features = training_data
        
        result = train_model(
            df, 
            N=10, 
            all_features=features,
            classifier_name='MLP',
            mixture_depth=1,
            n_features_selected='auto',
            oversample_method='SMOTE'
        )
        
        # 应该返回模型和相关组件
        assert result[0] is not None  # peak_model
        assert result[8] is not None  # trough_model
    
    @pytest.mark.slow
    def test_model_prediction_shape(self, training_data, trained_model):
        """验证预测输出形状"""
        df, features = training_data
        model, scaler, selector, selected_features = trained_model
        
        X = df[selected_features].fillna(0)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        assert len(predictions) == len(df)
        assert predictions.dtype in [np.int64, np.float64]
```

---

### 4. 回测逻辑测试

```python
# tests/unit/test_trading/test_backtest.py
import pytest
from ml_trader.trading.backtest import backtest_results
import pandas as pd

class TestBacktest:
    def test_basic_backtest(self):
        """验证基础回测逻辑"""
        # 构造简单的交易信号
        result_df = pd.DataFrame({
            'TradeDate': pd.date_range('2021-01-01', periods=10),
            'Close': [100, 102, 101, 105, 103, 108, 107, 110, 109, 112],
            'Peak_Prediction': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'Trough_Prediction': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        })
        result_df.index = pd.to_datetime(result_df['TradeDate'])
        
        from ml_trader.models.predictor import get_trade_signal
        signal_df = get_trade_signal(result_df)
        
        bt_result, trades_df = backtest_results(
            result_df, 
            signal_df, 
            N_buy=1, 
            N_sell=1,
            enable_chase=False,
            enable_stop_loss=False,
            initial_capital=100000
        )
        
        # 验证回测结果包含必要字段
        assert '累计收益率' in bt_result
        assert '交易笔数' in bt_result
        assert '胜率' in bt_result
        assert not trades_df.empty
```

---

## 🚀 快速开始

### 1. 安装测试依赖

```powershell
pip install pytest pytest-cov pytest-mock pytest-timeout
```

### 2. 运行测试

```powershell
# 运行所有测试
pytest

# 运行特定模块
pytest tests/unit/test_features/

# 生成覆盖率报告
pytest --cov=ml_trader --cov-report=html

# 只运行快速测试（跳过 @pytest.mark.slow）
pytest -m "not slow"
```

### 3. 配置文件

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_market_data():
    """通用行情数据fixture"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100)
    close = 3000 + np.random.randn(100).cumsum()
    
    return pd.DataFrame({
        'TradeDate': dates.strftime('%Y%m%d'),
        'Open': close + np.random.randn(100),
        'High': close + abs(np.random.randn(100)) + 5,
        'Low': close - abs(np.random.randn(100)) - 5,
        'Close': close,
        'Volume': 100000 + np.random.randint(-1000, 1000, 100),
    })

@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        'N': 10,
        'mixture_depth': 1,
        'oversample_method': 'SMOTE',
        'random_seed': 42
    }
```

---

## 📊 测试覆盖率目标

| 模块 | 当前覆盖率 | 目标覆盖率 |
|------|----------|----------|
| features/ | 0% | 80% |
| data/ | 0% | 75% |
| models/ | 0% | 70% |
| trading/ | 0% | 85% |
| visualization/ | 0% | 50% |
| **整体** | **0%** | **75%** |

---

## ⏱️ 实施时间表

- **第1周**: 特征工程测试（indicators, patterns）
- **第2周**: 数据处理测试（loader, preprocessor）
- **第3周**: 模型测试（architectures, trainer）
- **第4周**: 回测测试 + 集成测试

---

**优先级**: 🔴 高  
**预计工作量**: 2-4周  
**ROI**: 极高（避免90%的回归bug）
