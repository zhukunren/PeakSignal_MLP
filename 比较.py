import pandas as pd
import numpy as np

# 1. 读取简化版预测集并处理列名
data_a = pd.read_csv(
    r"D:\项目\机器学习简化版\简化版预测集.csv",
    parse_dates=['tradedate']
)
data_a.columns = data_a.columns.astype(str).str.lower()
data_a = data_a[[
    'tradedate', 'close_ma5_diff', 'ma5_ma20_diff', 'rsi_signal', 'macd_diff',
    'bollinger_position', 'k_d_diff', 'consecutiveup', 'consecutivedown',
    'cross_ma5_count', 'volume_spike_count', 'one', 'close', 'pch', 'cci_20',
    'williams_%r_14', 'obv', 'vwap', 'zscore_20', 'plus_di', 'minus_di',
    'adx_14', 'bollinger_width', 'slope_ma5', 'volume_change',
    'price_mean_diff', 'high_mean_diff', 'low_mean_diff',
    'ma_5', 'ma_20', 'ma_50', 'ma_200', 'ema_5', 'ema_20',
    'mfi_14', 'cmf_20', 'trix_15', 'ultimate_osc', 'chaikin_osc', 'ppo',
    'dpo_20', 'kst', 'kst_signal', 'kama_10'
]]
data_a.set_index('tradedate', inplace=True)

# 2. 读取完整版预测集并统一列名小写
data_b = pd.read_csv(
    r"D:\项目\机器学习\完整版预测集.csv",
    parse_dates=['tradedate']
)
data_b.columns = data_b.columns.astype(str).str.lower()
data_b = data_b[[
    'tradedate', 'close_ma5_diff', 'ma5_ma20_diff', 'rsi_signal', 'macd_diff',
    'bollinger_position', 'k_d_diff', 'consecutiveup', 'consecutivedown',
    'cross_ma5_count', 'volume_spike_count', 'one', 'close', 'pch', 'cci_20',
    'williams_%r_14', 'obv', 'vwap', 'zscore_20', 'plus_di', 'minus_di',
    'adx_14', 'bollinger_width', 'slope_ma5', 'volume_change',
    'price_mean_diff', 'high_mean_diff', 'low_mean_diff',
    'ma_5', 'ma_20', 'ma_50', 'ma_200', 'ema_5', 'ema_20',
    'mfi_14', 'cmf_20', 'trix_15', 'ultimate_osc', 'chaikin_osc', 'ppo',
    'dpo_20', 'kst', 'kst_signal', 'kama_10'
]]
data_b.set_index('tradedate', inplace=True)

# 3. 计算绝对差值并筛选 > 1 的位置
abs_diff = (data_a - data_b).abs()
mask = abs_diff > 0.01

# 4. 展开成 (tradedate, column, abs_diff) 格式
diff_positions = (
    abs_diff[mask]
    .stack()
    .reset_index()
    .rename(columns={
        'level_0': 'tradedate',
        'level_1': 'column',
        0: 'abs_diff'
    })
)

# 5. 用列表推导取出原始值
diff_positions['data_a'] = [
    data_a.at[dt, col]
    for dt, col in zip(diff_positions['tradedate'], diff_positions['column'])
]
diff_positions['data_b'] = [
    data_b.at[dt, col]
    for dt, col in zip(diff_positions['tradedate'], diff_positions['column'])
]

# 6. 打印结果
print(f"共找到 {len(diff_positions)} 个差值绝对值大于1的单元格：")
print(diff_positions[['tradedate', 'column', 'data_a', 'data_b', 'abs_diff']])
filtered_data = diff_positions[diff_positions['tradedate'] == '2024-08-14']
print(filtered_data)
