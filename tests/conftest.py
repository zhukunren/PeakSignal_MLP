"""
Pytest 配置文件
"""
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_market_data():
    """
    生成测试用的市场数据

    包含100天的模拟行情数据
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100)

    # 生成模拟价格（随机游走）
    close_prices = 3000 + np.random.randn(100).cumsum() * 10

    return pd.DataFrame({
        'TradeDate': dates.strftime('%Y%m%d'),
        'Open': close_prices + np.random.randn(100) * 5,
        'High': close_prices + np.abs(np.random.randn(100)) * 10 + 5,
        'Low': close_prices - np.abs(np.random.randn(100)) * 10 - 5,
        'Close': close_prices,
        'Volume': 100000 + np.random.randint(-5000, 5000, 100),
        'Amount': None  # 可选字段
    })


@pytest.fixture
def sample_prices():
    """简单的价格序列用于指标测试"""
    np.random.seed(42)
    return pd.Series(3000 + np.random.randn(50).cumsum() * 10)


@pytest.fixture
def test_config():
    """测试配置"""
    return {
        'N': 10,
        'mixture_depth': 1,
        'oversample_method': 'SMOTE',
        'random_seed': 42
    }
