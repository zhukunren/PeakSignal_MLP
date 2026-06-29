"""
特征工程 - 技术指标测试

测试 ml_trader.features.indicators 模块中的技术指标计算
"""
import pytest
import pandas as pd
import numpy as np
from ml_trader.features.indicators import (
    compute_RSI,
    compute_MACD,
    compute_KD,
    compute_Bollinger_Bands,
    compute_ATR
)


class TestRSI:
    """RSI (相对强弱指数) 测试"""

    def test_rsi_range(self, sample_prices):
        """RSI 应该在 0-100 之间"""
        rsi = compute_RSI(sample_prices, period=14)
        # 删除前面的 NaN 值
        valid_rsi = rsi.dropna()
        assert valid_rsi.between(0, 100).all(), "RSI 值应该在 0-100 之间"

    def test_rsi_overbought(self):
        """持续上涨应该产生 > 70 的 RSI"""
        # 创建持续上涨的价格序列
        prices = pd.Series(range(100, 120))
        rsi = compute_RSI(prices, period=14)
        assert rsi.iloc[-1] > 70, "持续上涨应该产生超买信号 (RSI > 70)"

    def test_rsi_oversold(self):
        """持续下跌应该产生 < 30 的 RSI"""
        # 创建持续下跌的价格序列
        prices = pd.Series(range(120, 100, -1))
        rsi = compute_RSI(prices, period=14)
        assert rsi.iloc[-1] < 30, "持续下跌应该产生超卖信号 (RSI < 30)"

    def test_rsi_with_constant_prices(self):
        """价格不变时 RSI 应该是 50"""
        prices = pd.Series([100] * 30)
        rsi = compute_RSI(prices, period=14)
        # 价格不变时，RSI 可能是 NaN（因为没有变化）
        # 跳过 NaN 值进行检查
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # 如果有有效值，应该接近 50
            assert 40 <= valid_rsi.iloc[-1] <= 60, "价格不变时 RSI 应该接近 50"
        # 如果全是 NaN，也是合理的（0/0 情况）

    def test_rsi_length(self, sample_prices):
        """RSI 序列长度应该与输入相同"""
        rsi = compute_RSI(sample_prices, period=14)
        assert len(rsi) == len(sample_prices), "RSI 序列长度应该与输入价格序列相同"


class TestMACD:
    """MACD 指标测试"""

    def test_macd_returns_two_series(self, sample_prices):
        """MACD 应该返回两个序列（MACD 和信号线）"""
        macd, signal = compute_MACD(sample_prices)
        assert isinstance(macd, pd.Series), "MACD 应该是 pandas Series"
        assert isinstance(signal, pd.Series), "信号线应该是 pandas Series"
        assert len(macd) == len(sample_prices), "MACD 长度应该与输入相同"
        assert len(signal) == len(sample_prices), "信号线长度应该与输入相同"

    def test_macd_uptrend(self):
        """上涨趋势中 MACD 应该在信号线上方"""
        prices = pd.Series([100] * 20 + list(range(100, 150)))
        macd, signal = compute_MACD(prices)
        # 在明显上涨阶段，MACD 应该大于信号线
        assert (macd.iloc[-10:] > signal.iloc[-10:]).sum() >= 7, \
            "上涨趋势中 MACD 应该多数时间在信号线上方"

    def test_macd_downtrend(self):
        """下跌趋势中 MACD 应该在信号线下方"""
        prices = pd.Series([100] * 20 + list(range(100, 50, -1)))
        macd, signal = compute_MACD(prices)
        # 在明显下跌阶段，MACD 应该小于信号线
        assert (macd.iloc[-10:] < signal.iloc[-10:]).sum() >= 7, \
            "下跌趋势中 MACD 应该多数时间在信号线下方"


class TestKD:
    """KD 指标测试"""

    def test_kd_range(self, sample_market_data):
        """K 和 D 值应该在 0-100 之间"""
        high = sample_market_data['High']
        low = sample_market_data['Low']
        close = sample_market_data['Close']

        K, D = compute_KD(high, low, close, period=14)

        valid_k = K.dropna()
        valid_d = D.dropna()

        assert valid_k.between(0, 100).all(), "K 值应该在 0-100 之间"
        assert valid_d.between(0, 100).all(), "D 值应该在 0-100 之间"

    def test_kd_length(self, sample_market_data):
        """K 和 D 序列长度应该与输入相同"""
        high = sample_market_data['High']
        low = sample_market_data['Low']
        close = sample_market_data['Close']

        K, D = compute_KD(high, low, close, period=14)

        assert len(K) == len(high), "K 序列长度应该与输入相同"
        assert len(D) == len(high), "D 序列长度应该与输入相同"


class TestBollingerBands:
    """布林带测试"""

    def test_bollinger_bands_structure(self, sample_prices):
        """布林带应该返回上轨、中轨、下轨三个序列"""
        upper, middle, lower = compute_Bollinger_Bands(sample_prices, period=20)

        assert isinstance(upper, pd.Series), "上轨应该是 pandas Series"
        assert isinstance(middle, pd.Series), "中轨应该是 pandas Series"
        assert isinstance(lower, pd.Series), "下轨应该是 pandas Series"

    def test_bollinger_bands_order(self, sample_prices):
        """上轨 > 中轨 > 下轨"""
        upper, middle, lower = compute_Bollinger_Bands(sample_prices, period=20)

        # 删除 NaN 后检查
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all(), "上轨应该 >= 中轨"
        assert (middle[valid_idx] >= lower[valid_idx]).all(), "中轨应该 >= 下轨"

    def test_bollinger_bands_middle_is_sma(self, sample_prices):
        """中轨应该等于移动平均线"""
        upper, middle, lower = compute_Bollinger_Bands(sample_prices, period=20)
        sma = sample_prices.rolling(window=20).mean()

        # 比较中轨和 SMA（允许微小误差）
        valid_idx = ~(middle.isna() | sma.isna())
        diff = (middle[valid_idx] - sma[valid_idx]).abs()
        assert (diff < 1e-10).all(), "中轨应该等于移动平均线"


class TestATR:
    """ATR (平均真实波幅) 测试"""

    def test_atr_positive(self, sample_market_data):
        """ATR 应该始终为正"""
        high = sample_market_data['High']
        low = sample_market_data['Low']
        close = sample_market_data['Close']

        atr = compute_ATR(high, low, close, period=14)
        valid_atr = atr.dropna()

        assert (valid_atr >= 0).all(), "ATR 应该始终为正"

    def test_atr_length(self, sample_market_data):
        """ATR 序列长度应该与输入相同"""
        high = sample_market_data['High']
        low = sample_market_data['Low']
        close = sample_market_data['Close']

        atr = compute_ATR(high, low, close, period=14)

        assert len(atr) == len(high), "ATR 序列长度应该与输入相同"

    def test_atr_reflects_volatility(self):
        """高波动时 ATR 应该更大"""
        # 低波动数据
        low_vol_data = pd.DataFrame({
            'High': [101, 102, 101, 102, 101] * 10,
            'Low': [99, 98, 99, 98, 99] * 10,
            'Close': [100, 100, 100, 100, 100] * 10
        })

        # 高波动数据
        high_vol_data = pd.DataFrame({
            'High': [110, 120, 105, 125, 95] * 10,
            'Low': [90, 80, 95, 75, 85] * 10,
            'Close': [100, 100, 100, 100, 100] * 10
        })

        atr_low = compute_ATR(low_vol_data['High'], low_vol_data['Low'],
                             low_vol_data['Close'], period=14)
        atr_high = compute_ATR(high_vol_data['High'], high_vol_data['Low'],
                              high_vol_data['Close'], period=14)

        assert atr_high.iloc[-1] > atr_low.iloc[-1], \
            "高波动时 ATR 应该比低波动时大"


# 性能测试（可选，用 @pytest.mark.slow 标记）
@pytest.mark.slow
class TestPerformance:
    """性能测试"""

    def test_rsi_large_dataset(self):
        """测试 RSI 在大数据集上的性能"""
        import time

        # 生成大数据集（10000 个数据点）
        large_prices = pd.Series(3000 + np.random.randn(10000).cumsum())

        start_time = time.time()
        rsi = compute_RSI(large_prices, period=14)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, "RSI 计算应该在 1 秒内完成"
        assert len(rsi) == len(large_prices)
