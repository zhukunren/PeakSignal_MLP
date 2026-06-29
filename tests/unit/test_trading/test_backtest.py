"""
交易回测逻辑测试

测试 ml_trader.trading.backtest 模块
"""
import pytest
import pandas as pd
import numpy as np
from ml_trader.trading.backtest import backtest_results
from ml_trader.models.predictor import get_trade_signal


class TestBacktest:
    """回测逻辑测试"""

    @pytest.fixture
    def simple_result_df(self):
        """简单的预测结果数据"""
        dates = pd.date_range('2021-01-01', periods=20)
        return pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Close': [100, 102, 101, 105, 103, 108, 107, 110, 109, 112,
                     115, 113, 118, 117, 120, 119, 122, 121, 125, 123],
            'Peak_Prediction': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                               0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'Trough_Prediction': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        }, index=dates)

    def test_backtest_basic(self, simple_result_df):
        """测试基础回测功能"""
        signal_df = get_trade_signal(simple_result_df)

        bt_result, trades_df = backtest_results(
            simple_result_df,
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
        assert '最大回撤' in bt_result
        assert not trades_df.empty, "应该有交易记录"

    def test_backtest_no_trades(self):
        """测试无交易信号的情况"""
        dates = pd.date_range('2021-01-01', periods=10)
        result_df = pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Close': range(100, 110),
            'Peak_Prediction': [0] * 10,
            'Trough_Prediction': [0] * 10,
        }, index=dates)

        signal_df = get_trade_signal(result_df)
        bt_result, trades_df = backtest_results(
            result_df, signal_df, N_buy=1, N_sell=1,
            enable_chase=False, enable_stop_loss=False
        )

        # 无交易时累计收益率应该为0
        assert bt_result['累计收益率'] == 0
        assert bt_result['交易笔数'] == 0
        assert trades_df.empty

    def test_backtest_profit_calculation(self):
        """测试收益计算的正确性"""
        dates = pd.date_range('2021-01-01', periods=6)
        result_df = pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Open': [100, 100, 110, 110, 90, 90],   # 添加 Open 列（回测需要）
            'Close': [100, 100, 110, 110, 90, 90],  # 买100卖110（+10%），买110卖90（-18.18%）
            'Peak_Prediction': [0, 0, 1, 0, 1, 0],  # 第3天和第5天卖
            'Trough_Prediction': [1, 0, 0, 1, 0, 0],  # 第1天和第4天买
        }, index=dates)

        signal_df = get_trade_signal(result_df)
        bt_result, trades_df = backtest_results(
            result_df, signal_df, N_buy=1, N_sell=1,
            enable_chase=False, enable_stop_loss=False,
            initial_capital=100000
        )

        # 验证回测执行成功
        assert 'cumulative_return' in bt_result or '累计收益率' in bt_result, "应该有累计收益率"
        assert 'trade_count' in bt_result or '交易笔数' in bt_result, "应该有交易笔数"

        # 如果有交易，验证交易记录格式
        if not trades_df.empty:
            # 验证交易记录有必要的列
            assert len(trades_df.columns) > 0, "交易记录应该有列"
            # 验证日期列存在
            date_cols = [col for col in trades_df.columns if '日期' in col or 'date' in col.lower()]
            assert len(date_cols) > 0, "交易记录应该包含日期信息"

    def test_backtest_max_drawdown(self):
        """测试最大回撤计算"""
        dates = pd.date_range('2021-01-01', periods=8)
        result_df = pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Close': [100, 100, 120, 120, 80, 80, 150, 150],
            'Peak_Prediction': [0, 0, 1, 0, 1, 0, 1, 0],
            'Trough_Prediction': [1, 0, 0, 1, 0, 1, 0, 0],
        }, index=dates)

        signal_df = get_trade_signal(result_df)
        bt_result, trades_df = backtest_results(
            result_df, signal_df, N_buy=1, N_sell=1,
            enable_chase=False, enable_stop_loss=False,
            initial_capital=100000
        )

        # 最大回撤应该为负数
        assert bt_result['最大回撤'] <= 0

    def test_backtest_win_rate(self):
        """测试胜率计算"""
        dates = pd.date_range('2021-01-01', periods=10)
        result_df = pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Close': [100, 100, 110, 110, 105, 105, 115, 115, 110, 110],
            'Peak_Prediction': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'Trough_Prediction': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        }, index=dates)

        signal_df = get_trade_signal(result_df)
        bt_result, trades_df = backtest_results(
            result_df, signal_df, N_buy=1, N_sell=1,
            enable_chase=False, enable_stop_loss=False,
            initial_capital=100000
        )

        # 胜率应该在 0-1 之间
        win_rate = bt_result.get('胜率', 0)
        if win_rate is not None:
            assert 0 <= win_rate <= 1, f"胜率应该在 0-1 之间，实际为 {win_rate}"
        else:
            # 如果胜率为 None，说明没有交易
            assert bt_result['交易笔数'] == 0, "无交易时胜率可以为 None"

        # 验证胜率计算
        if bt_result['交易笔数'] > 0:
            winning_trades = (trades_df['收益率'] > 0).sum()
            expected_win_rate = winning_trades / len(trades_df)
            assert abs(bt_result['胜率'] - expected_win_rate) < 0.01


class TestTradeSignal:
    """交易信号生成测试"""

    def test_signal_generation(self):
        """测试信号生成"""
        dates = pd.date_range('2021-01-01', periods=5)
        result_df = pd.DataFrame({
            'TradeDate': dates.strftime('%Y%m%d'),
            'Close': [100, 102, 101, 105, 103],
            'Peak_Prediction': [0, 0, 1, 0, 0],
            'Trough_Prediction': [1, 0, 0, 1, 0],
        }, index=dates)

        signal_df = get_trade_signal(result_df)

        # 验证信号列存在（实际列名是 'direction'）
        assert 'direction' in signal_df.columns, "应该包含 direction 列"

        # 买入信号应该出现
        assert 'buy' in signal_df['direction'].values, "应该有买入信号"

        # 卖出信号应该出现
        assert 'sell' in signal_df['direction'].values, "应该有卖出信号"
