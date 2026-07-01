# preprocess.py
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 从外部 function.py 导入技术指标计算函数
# 请确保你的 function.py 文件中包含 compute_RSI, compute_MACD, compute_KD, compute_momentum, compute_ROC, compute_Bollinger_Bands,
# compute_ATR, compute_volatility, compute_OBV, compute_VWAP, compute_MFI, compute_CMF, compute_chaikin_oscillator,
# compute_CCI, compute_williams_r, compute_zscore, compute_ADX, compute_TRIX, compute_ultimate_oscillator, compute_PPO,
# compute_DPO, compute_KST, compute_KAMA, compute_EMA, compute_MoneyFlowIndex, identify_low_troughs, identify_high_peaks,
# compute_SMA, compute_PercentageB, compute_AccumulationDistribution, compute_HighLow_Spread, compute_PriceChannel, compute_RenkoSlope
from ml_trader.features.indicators import (
    compute_RSI, compute_MACD, compute_KD, compute_momentum, compute_ROC,
    compute_Bollinger_Bands, compute_ATR, compute_volatility, compute_OBV,
    compute_VWAP, compute_MFI, compute_CMF, compute_chaikin_oscillator,
    compute_CCI, compute_williams_r, compute_zscore, compute_ADX, compute_TRIX,
    compute_ultimate_oscillator, compute_PPO, compute_DPO, compute_KST,
    compute_KAMA, compute_EMA, compute_MoneyFlowIndex, compute_SMA,
    compute_PercentageB, compute_AccumulationDistribution, compute_HighLow_Spread,
    compute_PriceChannel, compute_RenkoSlope
)
from ml_trader.features.patterns import (
    compute_MACD_histogram,
    compute_ichimoku,
    compute_coppock_curve,
    compute_chaikin_volatility,
    compute_ease_of_movement,
    compute_vortex_indicator,
    compute_annualized_volatility,
    compute_fisher_transform,
    compute_CMO,
    identify_low_troughs,
    identify_high_peaks,
)
from ml_trader.features.engineering import generate_features
from ml_trader.logging_config import get_logger
#import streamlit as st


logger = get_logger(__name__)


def _feature_frame(feature_map, index):
    normalized = {}
    for name, value in feature_map.items():
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                raise ValueError(f"特征 {name} 必须是一维序列，实际 DataFrame 形状为 {value.shape}")
            value = value.iloc[:, 0]
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            if value.shape[1] != 1:
                raise ValueError(f"特征 {name} 必须是一维序列，实际 ndarray 形状为 {value.shape}")
            value = value[:, 0]
        normalized[name] = value
    return pd.DataFrame(normalized, index=index)

# 封装相关性过滤函数
def correlation_filtering(data, features, threshold=0.95):
    """
    根据相关性阈值过滤特征，移除高相关性特征。
    
    参数:
        data: 包含特征数据的DataFrame
        features: 待过滤的特征列表
        threshold: 相关性阈值（默认0.95）
        
    返回:
        过滤后的特征列表
    """
    corr_matrix = data[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    filtered_features = [f for f in features if f not in to_drop]
    logger.info("Correlation filtering completed: remaining_features=%s", len(filtered_features))
    return filtered_features

# 封装 PCA 降维函数
def pca_reduction(data, features, max_components=100):
    """
    对给定特征进行 PCA 降维，并将降维后的特征添加到 data 中。
    
    参数:
        data: 包含特征数据的 DataFrame
        features: 待降维的特征列表
        max_components: 最大降维维度（默认100）
        
    返回:
        PCA 后生成的特征名称列表
    """
    X = data[features].fillna(0).values
    n_components = min(max_components, len(features))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_feature_names = [f'PCA_{i}' for i in range(n_components)]
    for i, name in enumerate(pca_feature_names):
        data[name] = X_pca[:, i]
    logger.info("PCA reduction completed: components=%s", n_components)
    return pca_feature_names


def preprocess_data(
    data: pd.DataFrame,
    N: int,
    mixture_depth: int,
    mark_labels: bool = True,
    min_features_to_select: int = 10,
    selected_func_names=None,
    selected_system=None,
):
    import torch
    """
    完整的特征工程示例:
      1) 数据排序 & 设置索引
      2) 原有手动计算的一些基础特征
      3) 调用 generate_features(data) 扩充更多特征
      4) (可选) 打标签 Peak/Trough
      5) 添加计数指标、衍生因子
      6) 整理 base_features, 并做方差过滤 & 相关性过滤
      7) mixture_depth>1 时生成混合因子, 并用 PCA 压缩
      8) 删除 NaN, 返回 data 与最终 all_features

    参数:
      data: 原始数据，至少包含 'TradeDate','Open','High','Low','Close' 等
      N: 用于打标签的窗口大小
      mixture_depth: 混合因子深度 (1 表示不做混合，>1 则做多层组合)
      mark_labels: 是否标注局部高/低点
      min_features_to_select, max_features_for_mixture: 预留的可选参数，目前未用

    返回:
      data, all_features
      - data: 处理后的 DataFrame（含新特征、滤除缺失值后）
      - all_features: 最终可用于建模的特征列名
    """

    logger.info(
        "Preprocess data started: rows=%s N=%s mixture_depth=%s mark_labels=%s",
        len(data),
        N,
        mixture_depth,
        mark_labels,
    )
    # (A) 对数据做排序、索引
    selected_func_names = [] if selected_func_names is None else list(selected_func_names)
    selected_system = [] if selected_system is None else list(selected_system)

    data = data.sort_values('TradeDate').copy()
    data.index = pd.to_datetime(data['TradeDate'], format='%Y%m%d')
    
    # ----------------- 原有基本特征计算 -----------------
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_60'] = data['Close'].rolling(window=60).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Price_MA20_Diff'] = (data['Close'] - data['MA_20']) / data['MA_20']
    data['MA5_MA20_Cross'] = np.where(data['MA_5'] > data['MA_20'], 1, 0)
    data['MA5_MA20_Cross_Diff'] = data['MA5_MA20_Cross'].diff()
    data['Slope_MA5'] = data['MA_5'].diff()
    data['RSI_14'] = compute_RSI(data['Close'], period=14)
    data['MACD'], data['MACD_signal'] = compute_MACD(data['Close'])
    data['MACD_Cross'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)
    data['MACD_Cross_Diff'] = data['MACD_Cross'].diff()
    data['K'], data['D'] = compute_KD(data['High'], data['Low'], data['Close'], period=14)
    data['Momentum_10'] = compute_momentum(data['Close'], period=10)
    data['ROC_10'] = compute_ROC(data['Close'], period=10)
    data['RSI_Reversal'] = (data['RSI_14'] > 70).astype(int) - (data['RSI_14'] < 30).astype(int)
    data['Reversal_Signal'] = (data['Close'] > data['High'].rolling(window=10).max()).astype(int) - (data['Close'] < data['Low'].rolling(window=10).min()).astype(int)
    data['UpperBand'], data['MiddleBand'], data['LowerBand'] = compute_Bollinger_Bands(data['Close'], period=20)
    data['ATR_14'] = compute_ATR(data['High'], data['Low'], data['Close'], period=14)
    data['Volatility_10'] = compute_volatility(data['Close'], period=10)
    data['Bollinger_Width'] = (data['UpperBand'] - data['LowerBand']) / data['MiddleBand']
    
    if 'Volume' in data.columns:
        data['OBV'] = compute_OBV(data['Close'], data['Volume'])
        data['Volume_Change'] = data['Volume'].pct_change()
        data['VWAP'] = compute_VWAP(data['High'], data['Low'], data['Close'], data['Volume'])
        data['MFI_14'] = compute_MFI(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
        data['CMF_20'] = compute_CMF(data['High'], data['Low'], data['Close'], data['Volume'], period=20)
        data['Chaikin_Osc'] = compute_chaikin_oscillator(data['High'], data['Low'], data['Close'], data['Volume'], short_period=3, long_period=10)
    else:
        data['OBV'] = np.nan
        data['Volume_Change'] = np.nan
        data['VWAP'] = np.nan
        data['MFI_14'] = np.nan
        data['CMF_20'] = np.nan
        data['Chaikin_Osc'] = np.nan

    # 累计量/价格水平类指标用滚动或比例表达，降低起始点和市场点位漂移影响。
    data['Close_MA200_Diff'] = data['Close'] / data['MA_200'].replace(0, np.nan) - 1
    data['Close_VWAP_Diff'] = data['Close'] / data['VWAP'].replace(0, np.nan) - 1
    data['Close_EMA20_Diff'] = data['Close'] / data['EMA_20'].replace(0, np.nan) - 1
    data['MA20_MA50_Diff_Pct'] = data['MA_20'] / data['MA_50'].replace(0, np.nan) - 1
    data['MA20_MA60_Diff_Pct'] = data['MA_20'] / data['MA_60'].replace(0, np.nan) - 1
    data['MA50_MA200_Diff_Pct'] = data['MA_50'] / data['MA_200'].replace(0, np.nan) - 1
    data['MA60_MA200_Diff_Pct'] = data['MA_60'] / data['MA_200'].replace(0, np.nan) - 1
    data['Close_MA60_Diff'] = data['Close'] / data['MA_60'].replace(0, np.nan) - 1
    data['Trend_Bull_20_60_200'] = ((data['MA_20'] > data['MA_60']) & (data['MA_60'] > data['MA_200'])).astype(int)
    data['Trend_Bear_20_60_200'] = ((data['MA_20'] < data['MA_60']) & (data['MA_60'] < data['MA_200'])).astype(int)
    data['Trend_Regime_Score'] = (
        np.sign(data['Close_MA200_Diff'].fillna(0))
        + np.sign(data['MA20_MA60_Diff_Pct'].fillna(0))
        + np.sign(data['MA60_MA200_Diff_Pct'].fillna(0))
    )
    data['Trend_Regime_Change'] = data['Trend_Regime_Score'].diff()
    if 'Volume' in data.columns:
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        rolling_volume_250 = data['Volume'].rolling(window=250, min_periods=20).sum()
        data['Rolling_VWAP_250'] = (
            (typical_price * data['Volume']).rolling(window=250, min_periods=20).sum()
            / rolling_volume_250.replace(0, np.nan)
        )
        data['Close_RollingVWAP250_Diff'] = data['Close'] / data['Rolling_VWAP_250'].replace(0, np.nan) - 1
        for window in (20, 60, 250):
            volume_sum = data['Volume'].rolling(window=window, min_periods=max(5, window // 5)).sum()
            obv_change = data['OBV'].diff(window)
            data[f'OBV_Change_{window}'] = obv_change
            data[f'OBV_Ratio_{window}'] = obv_change / volume_sum.replace(0, np.nan)
        data['Volume_Ratio_20'] = data['Volume'] / data['Volume'].rolling(window=20, min_periods=5).mean().replace(0, np.nan) - 1
        data['Volume_Ratio_60'] = data['Volume'] / data['Volume'].rolling(window=60, min_periods=10).mean().replace(0, np.nan) - 1
        data['Volume_Ratio_250'] = data['Volume'] / data['Volume'].rolling(window=250, min_periods=20).mean().replace(0, np.nan) - 1
        data['Chaikin_Osc_Ratio'] = data['Chaikin_Osc'] / data['Volume'].rolling(window=20, min_periods=5).mean().replace(0, np.nan)
    else:
        data['Rolling_VWAP_250'] = np.nan
        data['Close_RollingVWAP250_Diff'] = np.nan
        for window in (20, 60, 250):
            data[f'OBV_Change_{window}'] = np.nan
            data[f'OBV_Ratio_{window}'] = np.nan
        data['Volume_Ratio_20'] = np.nan
        data['Volume_Ratio_60'] = np.nan
        data['Volume_Ratio_250'] = np.nan
        data['Chaikin_Osc_Ratio'] = np.nan

    if 'Amount' in data.columns:
        data['Amount_Ratio_20'] = data['Amount'] / data['Amount'].rolling(window=20, min_periods=5).mean().replace(0, np.nan) - 1
        data['Amount_Ratio_60'] = data['Amount'] / data['Amount'].rolling(window=60, min_periods=10).mean().replace(0, np.nan) - 1
    else:
        data['Amount_Ratio_20'] = np.nan
        data['Amount_Ratio_60'] = np.nan
        
    data['CCI_20'] = compute_CCI(data['High'], data['Low'], data['Close'], period=20)
    data['Williams_%R_14'] = compute_williams_r(data['High'], data['Low'], data['Close'], period=14)
    data['ZScore_20'] = compute_zscore(data['Close'], period=20)
    data['Price_Mean_Diff'] = (data['Close'] - data['Close'].rolling(window=10).mean()) / data['Close'].rolling(window=10).mean()
    data['High_Mean_Diff'] = (data['High'] - data['High'].rolling(window=10).mean()) / data['High'].rolling(window=10).mean()
    data['Low_Mean_Diff'] = (data['Low'] - data['Low'].rolling(window=10).mean()) / data['Low'].rolling(window=10).mean()
    data['Plus_DI'], data['Minus_DI'], data['ADX_14'] = compute_ADX(data['High'], data['Low'], data['Close'], period=14)
    data['TRIX_15'] = compute_TRIX(data['Close'], period=15)
    data['Ultimate_Osc'] = compute_ultimate_oscillator(data['High'], data['Low'], data['Close'], short_period=7, medium_period=14, long_period=28)
    data['PPO'] = compute_PPO(data['Close'], fast_period=12, slow_period=26)
    data['DPO_20'] = compute_DPO(data['Close'], period=20)
    data['DPO_20_Pct'] = data['DPO_20'] / data['Close'].replace(0, np.nan)
    data['KST'], data['KST_signal'] = compute_KST(data['Close'], r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15)
    data['KAMA_10'] = compute_KAMA(data['Close'], n=10, pow1=2, pow2=30)
    data['Seasonality'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['one'] = 1

    data = data.copy()
    for window in (5, 20, 60, 120, 250):
        data[f'Return_{window}'] = data['Close'].pct_change(window)
        rolling_high = data['High'].rolling(window=window, min_periods=max(5, window // 5)).max()
        rolling_low = data['Low'].rolling(window=window, min_periods=max(5, window // 5)).min()
        data[f'Price_Position_{window}'] = (
            (data['Close'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        )
        data[f'Drawdown_{window}'] = data['Close'] / rolling_high.replace(0, np.nan) - 1

        if window in (20, 60):
            prev_high = rolling_high.shift(1)
            prev_low = rolling_low.shift(1)
            data[f'New_High_{window}'] = (data['High'] >= prev_high).astype(int)
            data[f'New_Low_{window}'] = (data['Low'] <= prev_low).astype(int)
            data[f'Close_From_Low_{window}'] = data['Close'] / rolling_low.replace(0, np.nan) - 1
            data[f'High_From_Close_{window}'] = data['High'] / data['Close'].replace(0, np.nan) - 1
            data[f'Low_From_Close_{window}'] = data['Low'] / data['Close'].replace(0, np.nan) - 1
            data[f'Bars_Since_High_{window}'] = (
                window - 1 - data['High'].rolling(window=window, min_periods=max(5, window // 5))
                .apply(np.argmax, raw=True)
            ) / window
            data[f'Bars_Since_Low_{window}'] = (
                window - 1 - data['Low'].rolling(window=window, min_periods=max(5, window // 5))
                .apply(np.argmin, raw=True)
            ) / window

    def rolling_zscore(series, window, min_periods):
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)

    data = data.copy()
    volatility_20 = compute_volatility(data['Close'], period=20)
    volatility_60 = compute_volatility(data['Close'], period=60)
    atr_14_pct = data['ATR_14'] / data['Close'].replace(0, np.nan)
    candle_range = (data['High'] - data['Low']).replace(0, np.nan)
    derived_features = {
        'Volatility_20': volatility_20,
        'Volatility_60': volatility_60,
        'Volatility_Ratio_20_60': volatility_20 / volatility_60.replace(0, np.nan),
        'ATR_14_Pct': atr_14_pct,
        'HL_Range_Pct': (data['High'] - data['Low']) / data['Close'].replace(0, np.nan),
        'Body_Pct': (data['Close'] - data['Open']) / data['Open'].replace(0, np.nan),
        'Upper_Shadow_Pct': (
            data['High'] - data[['Open', 'Close']].max(axis=1)
        ) / data['Close'].replace(0, np.nan),
        'Lower_Shadow_Pct': (
            data[['Open', 'Close']].min(axis=1) - data['Low']
        ) / data['Close'].replace(0, np.nan),
        'Gap_Pct': data['Open'] / data['Close'].shift(1).replace(0, np.nan) - 1,
        'Candle_Close_Position': (data['Close'] - data['Low']) / candle_range,
        'Return_5_20_Diff': data['Return_5'] - data['Return_20'],
        'Return_20_60_Diff': data['Return_20'] - data['Return_60'],
        'Return_20_Z_250': rolling_zscore(data['Return_20'], 250, 60),
        'Return_60_Z_250': rolling_zscore(data['Return_60'], 250, 60),
        'Drawdown_60_Z_250': rolling_zscore(data['Drawdown_60'], 250, 60),
        'Volatility_20_Z_250': rolling_zscore(volatility_20, 250, 60),
        'Bollinger_Width_Ratio_120': (
            data['Bollinger_Width']
            / data['Bollinger_Width'].rolling(window=120, min_periods=20).mean().replace(0, np.nan)
            - 1
        ),
        'ATR_14_Ratio_60': (
            atr_14_pct / atr_14_pct.rolling(window=60, min_periods=10).mean().replace(0, np.nan) - 1
        ),
        'RSI_14_Slope_5': data['RSI_14'].diff(5),
        'RSI_14_Z_120': rolling_zscore(data['RSI_14'], 120, 20),
        'Price_Position_20_60_Diff': data['Price_Position_20'] - data['Price_Position_60'],
        'Price_Position_60_250_Diff': data['Price_Position_60'] - data['Price_Position_250'],
        'Drawdown_20_60_Diff': data['Drawdown_20'] - data['Drawdown_60'],
    }

    # ----------------- 新增更多样化特征 -----------------
    derived_features['SMA_10'] = compute_SMA(data['Close'], window=10)
    derived_features['SMA_30'] = compute_SMA(data['Close'], window=30)
    derived_features['EMA_10'] = compute_EMA(data['Close'], span=10)
    derived_features['EMA_30'] = compute_EMA(data['Close'], span=30)
    derived_features['PercentB'] = compute_PercentageB(data['Close'], data['UpperBand'], data['LowerBand'])
    if 'Volume' in data.columns:
        derived_features['AccumDist'] = compute_AccumulationDistribution(data['High'], data['Low'], data['Close'], data['Volume'])
    else:
        derived_features['AccumDist'] = np.nan
    if 'Volume' in data.columns:
        derived_features['MFI_New'] = compute_MoneyFlowIndex(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
    else:
        derived_features['MFI_New'] = np.nan
    derived_features['HL_Spread'] = compute_HighLow_Spread(data['High'], data['Low'])
    price_channel = compute_PriceChannel(data['High'], data['Low'], data['Close'], window=20)
    derived_features['PriceChannel_Mid'] = price_channel['middle_channel']
    derived_features['RenkoSlope'] = compute_RenkoSlope(data['Close'], bricks=3)
    data = pd.concat([data, _feature_frame(derived_features, data.index)], axis=1).copy()

    # ------------------ 3) 调用 generate_features 扩充特征 ------------------
    logger.info("Generating additional features")
    pre_cols = set(data.columns)
    data = generate_features(data)  # 这行里会生成额外的列
    data = data.loc[:, ~data.columns.duplicated(keep='first')].copy()
    post_cols = set(data.columns)
    new_cols = post_cols - pre_cols
    logger.info("Additional feature generation completed: new_columns=%s", len(new_cols))

    # ------------------ 4) 打标签 (可选) ------------------
    if mark_labels:
        logger.info("Identifying local peaks and troughs")
        N = int(N)
        data = identify_low_troughs(data, N)
        data = identify_high_peaks(data, N)
    else:
        # 若不需要，则保证 Peak/Trough 不存在或置为0
        if 'Peak' in data.columns:
            data.drop(columns=['Peak'], inplace=True)
        if 'Trough' in data.columns:
            data.drop(columns=['Trough'], inplace=True)
        data['Peak'] = 0
        data['Trough'] = 0

    # ------------------ 5) 添加计数指标 ------------------
    logger.info("Adding count-based features")
    post_label_features = {}
    price_change = data['Close'].diff()
    up = pd.Series(np.where(price_change > 0, 1, 0), index=data.index)
    down = pd.Series(np.where(price_change < 0, 1, 0), index=data.index)
    post_label_features['PriceChange'] = price_change
    post_label_features['Up'] = up
    post_label_features['Down'] = down
    post_label_features['ConsecutiveUp'] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    post_label_features['ConsecutiveDown'] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    window_size = 10
    cross_ma5 = pd.Series(np.where(data['Close'] > data['MA_5'], 1, 0), index=data.index)
    post_label_features['Cross_MA5'] = cross_ma5
    post_label_features['Cross_MA5_Count'] = cross_ma5.rolling(window=window_size).sum()
    if 'Volume' in data.columns:
        volume_ma_5 = data['Volume'].rolling(window=5).mean()
        volume_spike = pd.Series(np.where(data['Volume'] > volume_ma_5 * 1.5, 1, 0), index=data.index)
        post_label_features['Volume_MA_5'] = volume_ma_5
        post_label_features['Volume_Spike'] = volume_spike
        post_label_features['Volume_Spike_Count'] = volume_spike.rolling(window=10).sum()
    else:
        post_label_features['Volume_Spike_Count'] = np.nan
    
    logger.info("Building derived base factors")
    macd_diff = data['MACD'] - data['MACD_signal']
    macd_diff_pct = macd_diff / data['Close'].replace(0, np.nan)
    post_label_features['Close_MA5_Diff'] = data['Close'] - data['MA_5']
    post_label_features['Close_MA5_Diff_Pct'] = data['Close'] / data['MA_5'].replace(0, np.nan) - 1
    post_label_features['Pch'] = data['Close'] / data['Close'].shift(1) - 1
    post_label_features['MA5_MA20_Diff'] = data['MA_5'] - data['MA_20']
    post_label_features['MA5_MA20_Diff_Pct'] = data['MA_5'] / data['MA_20'].replace(0, np.nan) - 1
    post_label_features['Slope_MA5_Pct'] = data['MA_5'].pct_change()
    post_label_features['RSI_Signal'] = data['RSI_14'] - 50
    post_label_features['MACD_Diff'] = macd_diff
    post_label_features['MACD_Diff_Pct'] = macd_diff_pct
    post_label_features['MACD_Diff_Pct_Change'] = macd_diff_pct.diff()
    band_range = (data['UpperBand'] - data['LowerBand']).replace(0, np.nan)
    bollinger_position = ((data['Close'] - data['MiddleBand']) / band_range).fillna(0)
    k_d_diff = data['K'] - data['D']
    post_label_features['Bollinger_Position'] = bollinger_position
    post_label_features['K_D_Diff'] = k_d_diff
    post_label_features['K_D_Diff_Change'] = k_d_diff.diff()

    # ------------- 新增扩展指标（新增的指标函数调用） -------------
    post_label_features['MACD_Hist'] = compute_MACD_histogram(data['Close'])
    ichimoku = compute_ichimoku(data['High'], data['Low'], data['Close'])
    post_label_features['Ichimoku_Tenkan'] = ichimoku['tenkan_sen']
    post_label_features['Ichimoku_Kijun'] = ichimoku['kijun_sen']
    post_label_features['Ichimoku_SpanA'] = ichimoku['senkou_span_a']
    post_label_features['Ichimoku_SpanB'] = ichimoku['senkou_span_b']
    post_label_features['Ichimoku_Chikou'] = ichimoku['chikou_span']
    post_label_features['Coppock'] = compute_coppock_curve(data['Close'])
    post_label_features['Chaikin_Vol'] = compute_chaikin_volatility(data['High'], data['Low'], period=10, ma_period=10)
    if 'Volume' in data.columns:
        post_label_features['EOM'] = compute_ease_of_movement(data['High'], data['Low'], data['Volume'], period=14)
    else:
        post_label_features['EOM'] = np.nan
    vortex_pos, vortex_neg = compute_vortex_indicator(data['High'], data['Low'], data['Close'], period=14)
    post_label_features['Vortex_Pos'] = vortex_pos
    post_label_features['Vortex_Neg'] = vortex_neg
    post_label_features['Annualized_Vol'] = compute_annualized_volatility(data['Close'], period=10, trading_days=252)
    post_label_features['Fisher'] = compute_fisher_transform(data['Close'], period=10)
    post_label_features['CMO_14'] = compute_CMO(data['Close'], period=14)
    data = pd.concat([data, _feature_frame(post_label_features, data.index)], axis=1).copy()

    # ------------------ 6) 检查关键列 ------------------
    required_cols = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff'
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"列 {col} 未被创建，请检查数据和计算步骤。")
    # ------------------ 6) 构建基础因子 base_features 列表 ------------------
    logger.info("Building base feature list")
    base_features = [
        'Close_MA5_Diff_Pct', 'MA5_MA20_Diff_Pct', 'RSI_Signal', 'MACD_Diff_Pct',
        'Bollinger_Position', 'K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown',
        'Cross_MA5_Count', 'Volume_Spike_Count', 'one', 'Pch','CCI_20',
        'Williams_%R_14', 'ZScore_20', 'Plus_DI', 'Minus_DI',
        'ADX_14','Bollinger_Width', 'Slope_MA5_Pct', 'Volume_Change',
        'Price_Mean_Diff','High_Mean_Diff','Low_Mean_Diff',
        'Price_MA20_Diff', 'Close_MA200_Diff', 'Close_VWAP_Diff',
        'Close_EMA20_Diff', 'MA20_MA50_Diff_Pct', 'MA50_MA200_Diff_Pct',
        'Close_RollingVWAP250_Diff', 'OBV_Ratio_20', 'OBV_Ratio_60',
        'OBV_Ratio_250', 'Volume_Ratio_20', 'Volume_Ratio_60',
        'Chaikin_Osc_Ratio', 'MFI_14','CMF_20','TRIX_15','Ultimate_Osc','PPO',
        'DPO_20_Pct','KST','KST_signal',
        'Return_5', 'Return_20', 'Return_60', 'Return_120', 'Return_250',
        'Return_20_Z_250', 'Return_60_Z_250',
        'Price_Position_5', 'Price_Position_20', 'Price_Position_60',
        'Price_Position_120', 'Price_Position_250',
        'Drawdown_5', 'Drawdown_20', 'Drawdown_60', 'Drawdown_120', 'Drawdown_250',
        'Drawdown_60_Z_250', 'New_High_20', 'New_High_60', 'New_Low_20',
        'New_Low_60', 'Close_From_Low_20', 'Close_From_Low_60',
        'High_From_Close_20', 'High_From_Close_60',
        'Volatility_Ratio_20_60', 'ATR_14_Pct', 'HL_Range_Pct',
        'ATR_14_Ratio_60', 'RSI_14_Z_120', 'Body_Pct', 'Upper_Shadow_Pct',
        'Lower_Shadow_Pct', 'Gap_Pct'
    ]

    # ★ 将 generate_features 里新增的列也并入 base_features
    #   这样后面方差过滤 & 相关性过滤也会考虑它们
    #base_features = list(set(base_features).union(new_cols))

    logger.info("Initial base feature count: %s", len(base_features))

    # ------------------ 6) 更新特征列表 ------------------
    selected_features = list(selected_func_names) + list(selected_system)
    if selected_features:
        missing_selected_features = [f for f in selected_features if f not in data.columns]
        if missing_selected_features:
            logger.warning("Selected features are missing and ignored: %s", missing_selected_features)
        base_features = [f for f in selected_features if f in data.columns]

    base_features = [f for f in base_features if f in data.columns]
    if base_features:
        data[base_features] = data[base_features].replace([np.inf, -np.inf], np.nan)
  
    # ------------------ 9) 方差过滤 ------------------
    logger.info("Applying variance filter")
    try:
        X_base = data[base_features].fillna(0)
        scaler_for_variance = StandardScaler()
        X_base_scaled = scaler_for_variance.fit_transform(X_base)
        selector = VarianceThreshold(threshold=1e-8)
        selector.fit(X_base_scaled)
        filtered_features = [f for f, s in zip(base_features, selector.get_support()) if s]
        logger.info(
            "Variance filter completed: remaining_features=%s original_features=%s",
            len(filtered_features),
            len(base_features),
        )
        base_features = filtered_features
    except Exception as e:
        logger.exception("Variance filtering failed: %s", e)

    # ------------------ 10) 相关性过滤 ------------------
    logger.info("Applying correlation filter")
    corr_matrix = data[base_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    base_features = [f for f in base_features if f not in to_drop]
    logger.info("Correlation filter completed: remaining_features=%s", len(base_features))

    # ------------------ 11) 若 mixture_depth > 1, 生成混合因子 ------------------
    logger.info("Generating mixed features: mixture_depth=%s", mixture_depth)
    if mixture_depth > 1:
        operators = ['+', '-', '*', '/']
        mixed_features = base_features.copy()
        current_depth_features = base_features.copy()

        for depth in range(2, mixture_depth + 1):
            logger.info("Generating mixed features for depth=%s", depth)
            new_features = []
            feature_pairs = combinations(current_depth_features, 2)
            for f1, f2 in feature_pairs:
                for op in operators:
                    new_feature_name = f'({f1}){op}({f2})_d{depth}'
                    try:
                        if op == '+':
                            data[new_feature_name] = data[f1] + data[f2]
                        elif op == '-':
                            data[new_feature_name] = data[f1] - data[f2]
                        elif op == '*':
                            data[new_feature_name] = data[f1] * data[f2]
                        elif op == '/':
                            denom = data[f2].replace(0, np.nan)
                            data[new_feature_name] = data[f1] / denom
                        data[new_feature_name] = data[new_feature_name].replace([np.inf, -np.inf], np.nan).fillna(0)
                        new_features.append(new_feature_name)
                    except Exception as e:
                        logger.exception("Failed to compute mixed feature %s: %s", new_feature_name, e)

            # 对新因子先做一次方差过滤 & 高相关过滤
            if new_features:
                X_new = data[new_features].fillna(0)
                sel_new = VarianceThreshold(threshold=0.0001)
                sel_new.fit(X_new)
                new_features = [nf for nf, s in zip(new_features, sel_new.get_support()) if s]
                if len(new_features) > 1:
                    corr_matrix_new = data[new_features].corr().abs()
                    upper_new = corr_matrix_new.where(np.triu(np.ones(corr_matrix_new.shape), k=1).astype(bool))
                    to_drop_new = [col for col in upper_new.columns if any(upper_new[col] > 0.95)]
                    new_features = [f for f in new_features if f not in to_drop_new]

            mixed_features.extend(new_features)
            current_depth_features = new_features.copy()

        # 现在 all_features = 基础 + 混合
        all_features = mixed_features.copy()
        if all_features:
            data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan)

        # 最后做 PCA 降维
        logger.info("Applying PCA reduction to mixed features")
        pca_components = min(100, len(all_features))
        pca = PCA(n_components=pca_components)
        X_mixed = data[all_features].fillna(0).values
        X_mixed_pca = pca.fit_transform(X_mixed)

        pca_feature_names = [f'PCA_{i}' for i in range(pca_components)]
        for i in range(pca_components):
            data[pca_feature_names[i]] = X_mixed_pca[:, i]

        all_features = pca_feature_names
    else:
        all_features = base_features.copy()

    if all_features:
        data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ------------------ 11) 删除缺失值 & 返回 ------------------
    data.index.name = 'date_index'
    #print(f"数据预处理前长度: {initial_length}, 数据预处理后长度: {final_length}")
    #all_features = selected_func_names+selected_system
    logger.info("Preprocess data completed: final_feature_count=%s rows=%s", len(all_features), len(data))
    return data, all_features

#时间序列强化采样
#@st.cache_data
import numpy as np

def create_pos_neg_sequences_by_consecutive_labels(
    X, y, negative_ratio=1.0, adjacent_steps=5, 
    bidirectional_neg=True,  # 负样本可在正段前后寻找
    random_fallback=True,    # 兜底随机采样负段
    seed=42,                 # 随机种子
    shuffle=True,            # 返回前打乱
    return_indices=False     # 是否返回对应段的索引
):
    """
    将相邻为 1 的 y 位置合并为正段，对每个正段取 X 的均值作为一个正样本。
    再按 negative_ratio 采样负样本段（长度为 adjacent_steps），取均值作为负样本。
    负段优先选在正段的相邻区域（不与任何正标签重叠），不足再全局兜底。
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    assert X.shape[0] == y.shape[0], "X 与 y 的长度必须一致"
    n = len(y)
    rng = np.random.default_rng(seed)

    # -------- 1) 找正段 --------
    pos_idx = np.where(y == 1)[0]
    pos_segments = []
    if pos_idx.size > 0:
        start = pos_idx[0]
        for i in range(1, len(pos_idx)):
            if pos_idx[i] != pos_idx[i-1] + 1:
                pos_segments.append(np.arange(start, pos_idx[i-1] + 1))
                start = pos_idx[i]
        pos_segments.append(np.arange(start, pos_idx[-1] + 1))

    # 正样本特征
    if len(pos_segments) > 0:
        pos_features = np.array([X[seg].mean(axis=0) for seg in pos_segments])
        pos_labels = np.ones(len(pos_features), dtype=np.int64)
    else:
        pos_features = np.zeros((0, X.shape[1]), dtype=X.dtype)
        pos_labels = np.zeros((0,), dtype=np.int64)

    # -------- 2) 负样本目标数量 --------
    neg_target = int(np.ceil(len(pos_features) * negative_ratio))

    # -------- 3) 构造可用的负段候选（长度 = adjacent_steps）--------
    def slice_window(start, length):
        end = start + length  # Python 切片右开区间
        if start < 0 or end > n:
            return None
        return np.arange(start, end)

    # 建立所有合法负段（全局），但先不加入，先尝试“相邻”
    all_neg_segments = []
    if adjacent_steps > 0:
        for s in range(0, n - adjacent_steps + 1):
            seg = np.arange(s, s + adjacent_steps)
            if np.all(y[seg] == 0):
                all_neg_segments.append(seg)

    # 标记每个正段相邻的候选负段（优先）
    neg_segments = []
    if adjacent_steps > 0 and len(pos_segments) > 0:
        used = set()  # 避免重复

        for seg in pos_segments:
            # after：正段后面紧邻
            if bidirectional_neg:
                # before：正段前面紧邻
                before = slice_window(seg[0] - adjacent_steps, adjacent_steps)
                if before is not None and np.all(y[before] == 0):
                    t = tuple(before.tolist())
                    if t not in used:
                        neg_segments.append(before)
                        used.add(t)
                        if len(neg_segments) >= neg_target:
                            break

            after = slice_window(seg[-1] + 1, adjacent_steps)
            if after is not None and np.all(y[after] == 0):
                t = tuple(after.tolist())
                if t not in used:
                    neg_segments.append(after)
                    used.add(t)
                    if len(neg_segments) >= neg_target:
                        break

    # -------- 4) 兜底：从全局候选里补足（随机或顺序）--------
    if len(neg_segments) < neg_target and len(all_neg_segments) > 0:
        remaining = neg_target - len(neg_segments)
        # 去掉已选
        chosen = {tuple(seg.tolist()) for seg in neg_segments}
        pool = [seg for seg in all_neg_segments if tuple(seg.tolist()) not in chosen]

        if random_fallback:
            if len(pool) > 0:
                take = min(remaining, len(pool))
                pick_idx = rng.choice(len(pool), size=take, replace=False)
                neg_segments.extend([pool[i] for i in pick_idx])
        else:
            neg_segments.extend(pool[:remaining])

    # 截断到目标数
    neg_segments = neg_segments[:neg_target]

    # -------- 5) 负样本特征与标签 --------
    if len(neg_segments) > 0:
        neg_features = np.array([X[seg].mean(axis=0) for seg in neg_segments])
        neg_labels = np.zeros(len(neg_features), dtype=np.int64)
    else:
        neg_features = np.zeros((0, X.shape[1]), dtype=X.dtype)
        neg_labels = np.zeros((0,), dtype=np.int64)

    # -------- 6) 拼接并可选打乱 --------
    features = np.concatenate([pos_features, neg_features], axis=0)
    labels = np.concatenate([pos_labels, neg_labels], axis=0)

    if shuffle and features.shape[0] > 0:
        idx = rng.permutation(features.shape[0])
        features = features[idx]
        labels = labels[idx]
        if return_indices:
            # 也要同步打乱返回的段索引
            seg_idx = (pos_segments + neg_segments)
            seg_idx = [seg_idx[i] for i in idx]
            return features, labels, seg_idx

    if return_indices:
        return features, labels, (pos_segments + neg_segments)
    return features, labels


#L正则化进行特征选择
def feature_selection(X, y, method="lasso", threshold=0.01):
    if method == "lasso":
        # 使用Lasso进行特征选择
        lasso = LogisticRegression(penalty='l1', solver='saga')
        lasso.fit(X, y)
        selected_features = [f for i, f in enumerate(X.columns) if abs(lasso.coef_[0][i]) > threshold]
    elif method == "random_forest":
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        feature_importances = rf.feature_importances_
        selected_features = [X.columns[i] for i in range(len(feature_importances)) if feature_importances[i] > threshold]
    else:
        raise ValueError("Unsupported feature selection method: Choose 'lasso' or 'random_forest'.")
    
    return selected_features
