import numpy as np
import pandas as pd
import os
import tushare as ts


def get_tushare_pro():
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置 TUSHARE_TOKEN 环境变量，再调用 Tushare 数据接口。")
    ts.set_token(token)
    return ts.pro_api()

# ---------- 技术指标计算函数 ----------

def compute_RSI(series, period=14):
    """
    RSI (Relative Strength Index) - 相对强弱指数
    衡量价格上涨和下跌的速度和幅度，用于判断超买或超卖状态。
    参数:
    - series: 序列 (如收盘价)
    - period: 计算周期
    返回:
    - RSI值序列
    
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD (Moving Average Convergence Divergence) - 指数平滑异同移动平均线
    衡量短期和长期价格趋势之间的差异。
    参数:
    - series: 序列 (如收盘价)
    - fast_period: 快速均线周期
    - slow_period: 慢速均线周期
    - signal_period: 信号线周期
    返回:
    - MACD值, 信号线值
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_Bollinger_Bands(series, period=20, num_std=2):
    """
    Bollinger Bands - 布林带
    基于移动平均和标准差构造的价格波动区间。
    参数:
    - series: 序列 (如收盘价)
    - period: 移动平均周期
    - num_std: 标准差倍数
    返回:
    - 上轨, 中轨, 下轨
    """
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def compute_KD(high, low, close, period=14):
    """
    KD指标 (KDJ的基础)
    衡量当前价格相对于过去高点和低点的位置。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - K值, D值
    """
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    K = rsv.ewm(com=2).mean()
    D = K.ewm(com=2).mean()
    return K, D

def compute_ATR(high, low, close, period=14):
    """
    ATR (Average True Range) - 平均真实波幅
    衡量价格波动范围的指标。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - ATR值序列
    """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_ADX(high, low, close, period=14):
    """
    ADX (Average Directional Index) - 平均趋向指数
    衡量趋势强度的指标。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - +DI, -DI, ADX值
    """
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * (-down_move)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    tr_sum = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_sum)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return plus_di, minus_di, adx

def compute_CCI(high, low, close, period=20):
    """
    CCI (Commodity Channel Index) - 商品通道指标
    衡量价格偏离其统计均值的程度。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期
    返回:
    - CCI值序列
    """
    tp = (high + low + close) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

def compute_momentum(series, period=10):
    """
    Momentum - 动量指标
    衡量当前价格相对于过去N天价格的变化幅度，反映价格变化的速度和方向。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认10)
    返回:
    - 动量值序列
    """
    return series.diff(period)

def compute_ROC(series, period=10):
    """
    ROC (Rate of Change) - 变化率指标
    衡量当前价格相对于过去N天价格的变化百分比，用于反映趋势的强弱。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认10)
    返回:
    - ROC值序列（百分比）
    """
    return series.pct_change(period) * 100

def compute_volume_change(volume, period=10):
    """
    Volume Change - 成交量变化率
    衡量当前成交量相对于过去N天成交量的变化比例，用于捕捉市场活跃度的变化。
    参数:
    - volume: 成交量序列
    - period: 计算周期 (默认10)
    返回:
    - 成交量变化率序列
    """
    return volume.diff(period) / volume.shift(period)

def compute_VWAP(high, low, close, volume):
    """
    VWAP (Volume Weighted Average Price) - 成交量加权平均价
    衡量市场的平均成交成本，常用于判断价格的合理区间。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - volume: 成交量序列
    返回:
    - VWAP值序列
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def compute_zscore(series, period=20):
    """
    Z-Score - 标准分数
    衡量当前值相对于过去N天均值的标准化偏差，反映价格的异常程度。
    参数:
    - series: 时间序列 (如收盘价)
    - period: 计算周期 (默认20)
    返回:
    - Z-Score值序列
    """
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return (series - mean) / std

def compute_volatility(series, period=10):
    """
    Volatility - 波动率
    衡量价格在过去N天的波动幅度，通常以标准差为度量。
    参数:
    - series: 时间序列 (如收盘价的收益率)
    - period: 计算周期 (默认10)
    返回:
    - 波动率序列
    """
    return series.pct_change().rolling(window=period).std()

def compute_OBV(close, volume):
    """
    OBV (On-Balance Volume) - 平衡成交量
    通过成交量的累积变化来衡量买卖力量，从而判断价格趋势的强弱。
    参数:
    - close: 收盘价序列
    - volume: 成交量序列
    返回:
    - OBV值序列
    用法:
    - OBV值随时间上升表示资金流入市场，可能预示价格上涨。
    - OBV值下降表示资金流出市场，可能预示价格下跌。
    """
    # 计算价格变化方向 (+1, 0, -1)
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  # 第一天无法计算变化方向，设为0
    # 根据方向累积成交量
    obv = (volume * direction).fillna(0).cumsum()
    return obv

def compute_williams_r(high, low, close, period=14):
    """
    Williams %R - 威廉指标
    衡量当前收盘价相对于过去N天的高点和低点的位置，常用于超买和超卖状态的判断。
    参数:
    - high: 最高价序列
    - low: 最低价序列
    - close: 收盘价序列
    - period: 计算周期 (默认14)
    返回:
    - Williams %R值序列
    用法:
    - %R接近-100: 表示超卖区域，可能出现反弹。
    - %R接近0: 表示超买区域，可能出现回调。
    """
    # 计算过去N天的最高点和最低点
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    # 计算威廉指标
    wr = -100 * ((hh - close) / (hh - ll))
    return wr

def compute_MFI(high, low, close, volume, period=14):
    """
    MFI (Money Flow Index)
    类似于RSI，但考虑成交量。
    """
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_flow = mf.where(tp > tp.shift(), 0)
    negative_flow = mf.where(tp < tp.shift(), 0)
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    mfi = 100 * (positive_sum / (positive_sum + negative_sum))
    return mfi

def compute_CMF(high, low, close, volume, period=20):
    """
    CMF (Chaikin Money Flow)
    衡量资金流入流出强度。
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def compute_TRIX(series, period=15):
    """
    TRIX (Triple Exponential Average)
    衡量价格变化的速度，三重平滑的EMA变化率。
    """
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = (ema3.diff() / ema3.shift()) * 100
    return trix

def compute_ultimate_oscillator(high, low, close, short_period=7, medium_period=14, long_period=28):
    """
    Ultimate Oscillator (UO)
    综合不同周期的摆动值衡量市场动能。
    """
    bp = close - np.minimum(low.shift(1), close.shift(1))
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    avg7 = bp.rolling(short_period).sum() / tr.rolling(short_period).sum()
    avg14 = bp.rolling(medium_period).sum() / tr.rolling(medium_period).sum()
    avg28 = bp.rolling(long_period).sum() / tr.rolling(long_period).sum()

    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    return uo

def compute_chaikin_oscillator(high, low, close, volume, short_period=3, long_period=10):
    """
    Chaikin Oscillator
    基于ADL(累积/派发线)的MACD式指标。
    """
    adl = compute_ADL_line(high, low, close, volume)
    short_ema = adl.ewm(span=short_period, adjust=False).mean()
    long_ema = adl.ewm(span=long_period, adjust=False).mean()
    cho = short_ema - long_ema
    return cho

def compute_ADL_line(high, low, close, volume):
    """
    ADL (Accumulation/Distribution Line)
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], np.nan).fillna(0)
    mf_volume = mf_multiplier * volume
    adl = mf_volume.cumsum()
    return adl

def compute_PPO(series, fast_period=12, slow_period=26):
    """
    PPO (Percentage Price Oscillator)
    与MACD类似，只是输出为百分比。
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    ppo = (fast_ema - slow_ema) / slow_ema * 100
    return ppo

def compute_DPO(series, period=20):
    """
    DPO (Detrended Price Oscillator)
    去趋势价格振荡指标。
    """
    shifted = series.shift(int((period/2)+1))
    sma = series.rolling(window=period).mean()
    dpo = series - sma.shift(int((period/2)+1))
    return dpo

def compute_KST(series, r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15):
    """
    KST (Know Sure Thing)
    基于ROC的综合动量指标。
    """
    roc1 = series.pct_change(r1)*100
    roc2 = series.pct_change(r2)*100
    roc3 = series.pct_change(r3)*100
    roc4 = series.pct_change(r4)*100

    sma_roc1 = roc1.rolling(sma1).mean()
    sma_roc2 = roc2.rolling(sma2).mean()
    sma_roc3 = roc3.rolling(sma3).mean()
    sma_roc4 = roc4.rolling(sma4).mean()

    kst = sma_roc1 + 2*sma_roc2 + 3*sma_roc3 + 4*sma_roc4
    signal = kst.rolling(9).mean()
    return kst, signal

def compute_KAMA(series, n=10, pow1=2, pow2=30):
    """
    KAMA (Kaufman's Adaptive Moving Average)
    自适应移动平均
    """
    change = series.diff(n).abs()
    volatility = series.diff(1).abs().rolling(window=n).sum()
    er = change / volatility
    sc = (er * (2/(pow1+1)-2/(pow2+1)) + 2/(pow2+1))**2

    kama = series.copy()
    for i in range(n, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(series.iloc[i]-kama.iloc[i-1])
    return kama

import numpy as np
import pandas as pd

def compute_SMA(series, window):
    """
    简单移动平均线 (Simple Moving Average)
    
    参数:
      - series: 数值序列（例如收盘价）
      - window: 窗口长度
    
    返回:
      - 简单移动平均序列
    """
    return series.rolling(window=window, min_periods=1).mean()

def compute_EMA(series, span):
    """
    指数移动平均线 (Exponential Moving Average)
    
    参数:
      - series: 数值序列（例如收盘价）
      - span: EMA的窗口跨度
    
    返回:
      - 指数移动平均序列
    """
    return series.ewm(span=span, adjust=False).mean()

def compute_PercentageB(close, upper_band, lower_band):
    """
    计算Bollinger %B指标
    %B 指标反映收盘价在布林带中的位置，取值范围通常在0到1之间。
    
    参数:
      - close: 收盘价序列
      - upper_band: 上轨序列
      - lower_band: 下轨序列
      
    返回:
      - %B 值序列
    """
    band_range = upper_band - lower_band
    # 防止除零操作
    band_range = band_range.replace(0, np.nan)
    percent_b = (close - lower_band) / band_range
    return percent_b.fillna(0)

def compute_AccumulationDistribution(high, low, close, volume):
    """
    累积/派发线 (Accumulation/Distribution Line)
    A/D线综合考虑价格和成交量信息，用于反映资金流入/流出情况。
    
    参数:
      - high: 最高价序列
      - low: 最低价序列
      - close: 收盘价序列
      - volume: 成交量序列
      
    返回:
      - A/D线序列
    """
    # 计算资金流向因子
    denominator = (high - low)
    denominator = denominator.replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denominator  # Money Flow Multiplier
    mfm = mfm.fillna(0)
    mfv = mfm * volume  # Money Flow Volume
    return mfv.cumsum()

def compute_MoneyFlowIndex(high, low, close, volume, period=14):
    """
    资金流量指标 (Money Flow Index, MFI)
    MFI 综合价格与成交量信息，反映市场的资金流入和流出情况。
    
    参数:
      - high: 最高价序列
      - low: 最低价序列
      - close: 收盘价序列
      - volume: 成交量序列
      - period: 计算周期 (默认为14)
      
    返回:
      - MFI 序列
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    
    positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
    
    mfi = 100 * positive_mf / (positive_mf + negative_mf)
    return mfi.fillna(50)  # 缺失值填充为50，中性水平

def compute_HighLow_Spread(high, low):
    """
    计算日内价差（High-Low Spread）
    
    参数:
      - high: 最高价序列
      - low: 最低价序列
      
    返回:
      - 价差序列（高点减去低点）
    """
    return high - low

def compute_PriceChannel(high, low, close, window=20):
    """
    价格通道 (Price Channel)
    价格通道通常由一定周期内的最高价和最低价构成，可用于捕捉价格突破情况。
    
    参数:
      - high: 最高价序列
      - low: 最低价序列
      - close: 收盘价序列
      - window: 周期 (默认为20)
      
    返回:
      - 一个DataFrame，包含通道上轨、下轨及中轨（中轨为均值）
    """
    upper_channel = high.rolling(window=window, min_periods=1).max()
    lower_channel = low.rolling(window=window, min_periods=1).min()
    middle_channel = (upper_channel + lower_channel) / 2
    return pd.DataFrame({
        'upper_channel': upper_channel,
        'middle_channel': middle_channel,
        'lower_channel': lower_channel
    })

def compute_RenkoSlope(close, bricks=3):
    """
    Renko 块趋势指标（简化版）
    根据价格区间构建Renko图中每个块的斜率，用于反映趋势力度。
    
    参数:
      - close: 收盘价序列
      - bricks: Renko块的价格差（默认为3）
      
    返回:
      - Renko斜率序列
    """
    price_diff = close.diff()
    # 当价格涨跌超过设定的砖块值时，记录为1或-1，否则为0
    renko = price_diff.apply(lambda x: 1 if x >= bricks else (-1 if x <= -bricks else 0))
    # 对renko序列进行累积或平滑处理，作为趋势力度指标
    return renko.rolling(window=5, min_periods=1).sum()

def compute_MACD_histogram(macd, signal):
    """
    MACD Histogram
    用于显示MACD与其信号线之间的差异。
    """
    return macd - signal

def compute_ema_crossover(series, short_period=12, long_period=26):
    """
    计算EMA交叉
    """
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    crossover = short_ema > long_ema
    return crossover

def compute_average_gain_loss(series, period=14):
    """
    计算平均涨幅和跌幅
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    return avg_gain, avg_loss

def compute_mfm(high, low, close):
    """
    计算资金流动乘数（MFM）
    """
    return ((close - low) - (high - close)) / (high - low)

def compute_RVI(series, period=14):
    """
    计算相对波动率指数（RVI）
    """
    log_returns = np.log(series / series.shift(1))
    rolling_mean = log_returns.rolling(window=period).mean()
    rolling_std = log_returns.rolling(window=period).std()
    rvi = rolling_mean / rolling_std
    return rvi

def compute_force_index(close, volume, period=1):
    """
    计算强势指数（Force Index）
    """
    force = close.diff(period) * volume
    return force

def compute_parabolic_sar(high, low, close, acceleration=0.02, maximum=0.2):
    """
    计算抛物线SAR
    """
    sar = close.copy()
    ep = high.max()
    af = acceleration
    for i in range(1, len(sar)):
        sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
        if close.iloc[i] > sar.iloc[i]:
            ep = high.iloc[i]
            af = min(af + acceleration, maximum)
        else:
            ep = low.iloc[i]
            af = min(af + acceleration, maximum)
    return sar

def compute_DMI(high, low, close, period=14):
    """
    计算方向性运动指数（DMI）
    """
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * (-down_move)
    tr = high - low
    tr_sum = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_sum)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_sum)
    return plus_di, minus_di

def compute_smoothed_RSI(series, period=14):
    """
    计算平滑RSI
    """
    avg_gain, avg_loss = compute_average_gain_loss(series, period)
    rs = avg_gain / avg_loss
    smoothed_rsi = 100 - (100 / (1 + rs))
    return smoothed_rsi

def compute_std(series, period=20):
    """
    计算标准差
    """
    return series.rolling(window=period).std()

def compute_ema_trend(series, period=14):
    """
    计算基于EMA的趋势
    """
    ema = series.ewm(span=period, adjust=False).mean()
    trend = series - ema
    return trend

# ---------- 高低点识别函数 ----------

def identify_high_peaks(df, window=3):
    df = df.copy()
    # 定义滚动窗口大小
    win = 2 * window + 1

    # 使用 NumPy 快速计算滚动最大值
    rolling_max = df['High'].rolling(window=win, center=True).max()

    # 标记潜在高点（等于滚动窗口最大值）
    df['PotentialPeak'] = (df['High'] == rolling_max).astype(int)

    # 计算窗口内最大值出现的次数
    # 使用 NumPy 的布尔操作替代 apply 函数
    rolling_max_counts = (
        df['High']
        .rolling(window=win, center=True)
        .apply(lambda x: np.sum(x == np.max(x)), raw=True)
    )

    # 标记最终的高点：既是潜在高点，又是窗口中唯一最大值
    df['Peak'] = ((df['PotentialPeak'] == 1) & (rolling_max_counts == 1)).astype(int)

    # 清理临时列
    df.drop(columns=['PotentialPeak'], inplace=True)

    return df


def identify_low_troughs(df, window=3):
    df = df.copy()
    # 定义滚动窗口大小
    win = 2 * window + 1

    # 使用 NumPy 快速计算滚动最小值
    rolling_min = df['Low'].rolling(window=win, center=True).min()

    # 标记潜在低点（等于滚动窗口最小值）
    df['PotentialTrough'] = (df['Low'] == rolling_min).astype(int)

    # 计算窗口内最小值出现的次数
    rolling_min_counts = (
        df['Low']
        .rolling(window=win, center=True)
        .apply(lambda x: np.sum(x == np.min(x)), raw=True)
    )

    # 标记最终的低点：既是潜在低点，又是窗口中唯一最小值
    df['Trough'] = ((df['PotentialTrough'] == 1) & (rolling_min_counts == 1)).astype(int)

    # 清理临时列
    df.drop(columns=['PotentialTrough'], inplace=True)

    return df



# ---------- 数据读取与处理函数 ----------

def read_day_fromtdx(file_path, stock_code_tdx):
    """
    从通达信DAY文件中读取股票日线数据。
    参数:
    - file_path: 文件目录路径
    - stock_code_tdx: 股票代码 (如 "sh600000")
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    file_full_path = os.path.join(file_path, 'vipdoc', stock_code_tdx[:2].lower(), 'lday', f"{stock_code_tdx}.day")
    print(f"尝试读取文件: {file_full_path}")
    dtype = np.dtype([
        ('date', '<i4'),
        ('open', '<i4'),
        ('high', '<i4'),
        ('low', '<i4'),
        ('close', '<i4'),
        ('amount', '<f4'),
        ('volume', '<i4'),
        ('reserved', '<i4')
    ])
    if not os.path.exists(file_full_path):
        print(f"文件 {file_full_path} 不存在。")
        return pd.DataFrame()
    try:
        data = np.fromfile(file_full_path, dtype=dtype)
        print(f"读取了 {len(data)} 条记录。")
    except Exception as e:
        print(f"读取文件失败：{e}")
        return pd.DataFrame()
    if data.size == 0:
        print("文件为空。")
        return pd.DataFrame()
    df = pd.DataFrame({
        'date': pd.to_datetime(data['date'].astype(str), format='%Y%m%d', errors='coerce'),
        'Open': data['open'] / 100.0,
        'High': data['high'] / 100.0,
        'Low': data['low'] / 100.0,
        'Close': data['close'] / 100.0,
        'Amount': data['amount'],
        'Volume': data['volume'],
    })
    df = df.dropna(subset=['date'])
    df['TradeDate'] = df['date'].dt.strftime('%Y%m%d')
    df.set_index('date', inplace=True)
    print(f"创建了包含 {len(df)} 条记录的DataFrame。")
    return df

def select_time(df, start_time='20230101', end_time='20240910'):
    """
    根据指定的时间范围筛选数据。
    参数:
    - df: 包含日期索引的DataFrame
    - start_time: 起始时间 (字符串, 格式 'YYYYMMDD')
    - end_time: 截止时间 (字符串, 格式 'YYYYMMDD')
    返回:
    - 筛选后的DataFrame
    """
    print(f"筛选日期范围: {start_time} 至 {end_time}")
    try:
        start_time = pd.to_datetime(start_time, format='%Y%m%d')
        end_time = pd.to_datetime(end_time, format='%Y%m%d')
    except Exception as e:
        print(f"日期转换错误：{e}")
        return pd.DataFrame()
    df_filtered = df.loc[start_time:end_time]
    print(f"筛选后数据长度: {len(df_filtered)}")
    return df_filtered

def compute_MACD_histogram(series, fast_period=12, slow_period=26, signal_period=9):
    """MACD直方图：返回 MACD 与 Signal 的差值"""
    macd, signal = compute_MACD(series, fast_period, slow_period, signal_period)
    return macd - signal

def compute_ichimoku(high, low, close, conversion_period=9, base_period=26, span_b_period=52, displacement=26):
    """
    Ichimoku云图指标，返回一个字典包含各条线：
      - tenkan_sen: 转换线 (Conversion Line)
      - kijun_sen: 基准线 (Base Line)
      - senkou_span_a: 领先A线 (Leading Span A)，向未来平移 displacement 个周期
      - senkou_span_b: 领先B线 (Leading Span B)，向未来平移 displacement 个周期
      - chikou_span: 滞后线 (Lagging Span)，向过去平移 displacement 个周期
    """
    tenkan_sen = (high.rolling(window=conversion_period).max() + low.rolling(window=conversion_period).min()) / 2
    kijun_sen = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(window=span_b_period).max() + low.rolling(window=span_b_period).min()) / 2).shift(displacement)
    chikou_span = close.shift(-displacement)
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def compute_coppock_curve(series, roc_period1=14, roc_period2=11, sma_period=10):
    """
    Coppock Curve 指标：常用于长周期底部确认
      计算方法：先计算两个不同周期的收益率，再取其和后平滑
    """
    roc1 = series.pct_change(roc_period1) * 100
    roc2 = series.pct_change(roc_period2) * 100
    coppock = (roc1 + roc2).rolling(window=sma_period).mean()
    return coppock

def compute_chaikin_volatility(high, low, period=10, ma_period=10):
    """
    Chaikin Volatility 指标：基于高低价差的移动平均变化率，反映波动性变化
      计算方法：先计算高低价差的简单移动平均，再取其周期性变化率（百分比）
    """
    hl_range = high - low
    sma_hl = hl_range.rolling(window=period).mean()
    volatility = sma_hl.pct_change(periods=ma_period) * 100
    return volatility

def compute_ease_of_movement(high, low, volume, period=14):
    """
    Ease of Movement (EOM) 指标：衡量价格变动与成交量之间的关系
      计算方法：先求中点价格的变化乘以价格幅度，再除以成交量，最后平滑处理
    """
    midpoint_diff = ((high + low) / 2).diff()
    price_range = high - low
    eom = midpoint_diff * price_range / volume.replace(0, 1e-9)
    return eom.rolling(window=period).mean()

def compute_vortex_indicator(high, low, close, period=14):
    """
    Vortex Indicator (VI) 指标：反映趋势的强度和方向
      返回两个系列：(VI+, VI-)
      其中：
        VI+ = rolling sum(|High - PrevLow|) / rolling sum(TR)
        VI- = rolling sum(|Low - PrevHigh|) / rolling sum(TR)
      TR 为真实波幅
    """
    tr = pd.concat([
        high - low, 
        (high - close.shift()).abs(), 
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    vm_plus = (high - low.shift()).abs()
    vm_minus = (low - high.shift()).abs()
    sum_tr = tr.rolling(window=period).sum()
    vi_plus = vm_plus.rolling(window=period).sum() / sum_tr.replace(0, 1e-9)
    vi_minus = vm_minus.rolling(window=period).sum() / sum_tr.replace(0, 1e-9)
    return vi_plus, vi_minus

def compute_annualized_volatility(series, period=10, trading_days=252):
    """
    年化波动率：基于滚动收益率标准差转换
      通常年化波动率 = rolling volatility * sqrt(交易日数)
    """
    vol = compute_volatility(series, period)
    return vol * np.sqrt(trading_days)

def compute_fisher_transform(series, period=10):
    """
    Fisher Transform 指标：将数据转换为近似正态分布
      计算方法：先归一化到[-1,1]区间，再应用 Fisher 变换公式
    """
    min_val = series.rolling(window=period).min()
    max_val = series.rolling(window=period).max()
    x = 2 * ((series - min_val) / (max_val - min_val + 1e-9)) - 1
    x = x.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    return fisher

def compute_CMO(series, period=14):
    """
    Chande Momentum Oscillator (CMO) 指标：
      计算方法：((正收益之和 - 负收益之和) / (正收益之和 + 负收益之和)) * 100
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    sum_gain = gain.rolling(window=period).sum()
    sum_loss = loss.rolling(window=period).sum()
    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss + 1e-9)
    return cmo

def compute_AD(high, low, close, volume):
    """
    累积分布线 (Accumulation/Distribution Line, A/D)
    公式：Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
         Money Flow Volume = Money Flow Multiplier * Volume
         A/D Line = 累积 Money Flow Volume
    """
    # 避免除零
    denominator = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / denominator
    mf_volume = mf_multiplier * volume
    ad = mf_volume.fillna(0).cumsum()
    return ad

def compute_chaikin_oscillator(high, low, close, volume, short_period=3, long_period=10):
    """
    查金振荡器 (Chaikin Oscillator)
    计算方法：使用累积分布线的短期EMA与长期EMA之差
    """
    ad = compute_AD(high, low, close, volume)
    ema_short = ad.ewm(span=short_period, adjust=False).mean()
    ema_long = ad.ewm(span=long_period, adjust=False).mean()
    chaikin = ema_short - ema_long
    return chaikin

def compute_MFI(high, low, close, volume, period=14):
    """
    资金流指数 (Money Flow Index, MFI)
    计算方法：
      1. 计算典型价格: (High + Low + Close) / 3
      2. 计算 Money Flow = 典型价格 * Volume
      3. 分别累计正向和负向 Money Flow
      4. MFI = 100 - 100 / (1 + (正向 Money Flow / 负向 Money Flow))
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    tp_diff = typical_price.diff()
    pos_mf = money_flow.where(tp_diff > 0, 0)
    neg_mf = money_flow.where(tp_diff < 0, 0)
    # 为避免除零，可加一个小数 epsilon
    epsilon = 1e-10
    mfi_ratio = pos_mf.rolling(window=period).sum() / (neg_mf.rolling(window=period).sum() + epsilon)
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi

def compute_VPT(close, volume):
    """
    成交量价格趋势 (Volume Price Trend, VPT)
    公式：VPT = 前一期 VPT + ((当前收盘价 - 前一期收盘价) / 前一期收盘价) * 当前成交量
    """
    vpt = ((close - close.shift(1)) / close.shift(1)) * volume
    vpt = vpt.fillna(0).cumsum()
    return vpt

def compute_EOM(high, low, volume, period=14):
    """
    易动指标 (Ease of Movement, EOM)
    常用计算公式：
      EOM = ( (High + Low)/2 - (前一日(High + Low)/2) ) / (Volume / 100000000)
      返回值通常再取一个移动平均以平滑噪音。
    """
    midpoint = (high + low) / 2
    dm = midpoint.diff()
    box_ratio = volume / 100000000  # 调整成交量的尺度
    eom = dm / (box_ratio.replace(0, np.nan))
    eom_ma = eom.rolling(window=period).mean()
    return eom_ma

def compute_VWAP_deviation(high, low, close, volume):
    """
    VWAP偏离指标：计算收盘价与VWAP的相对偏离比例
    """
    vwap = compute_VWAP(high, low, close, volume)
    deviation = (close - vwap) / vwap
    return deviation

def compute_OBV_MA(close, volume, ma_period=20):
    """
    计算 OBV 及其移动平均线 (MA)，用于捕捉 OBV 均线交叉信号
    """
    obv = compute_OBV(close, volume)
    obv_ma = obv.rolling(window=ma_period).mean()
    return obv, obv_ma

def compute_OBV_cross_signal(close, volume, ma_period=20):
    """
    生成 OBV 均线交叉信号：
      当 OBV 上穿 OBV MA 时，可能视为买入信号；下穿时视为卖出信号。
    此处返回一个简单的0/1信号数组（1 表示 OBV 高于均线）。
    """
    obv, obv_ma = compute_OBV_MA(close, volume, ma_period)
    signal = np.where(obv > obv_ma, 1, 0)
    return signal

def compute_OBV_divergence(close, volume, window=20):
    """
    计算 OBV 与收盘价之间的滚动相关系数，用以判断二者是否出现背离。
    若相关系数持续为负值，则可能预示价格走势与成交量背离。
    """
    obv = compute_OBV(close, volume)
    corr = close.rolling(window=window).corr(obv)
    return corr

def read_day_from_tushare(symbol_code, symbol_type='stock'):
    """
    使用 Tushare API 获取股票或指数的全部日线行情数据。
    参数:
    - symbol_code: 股票或指数代码 (如 "000001.SZ" 或 "000300.SH")
    - symbol_type: 'stock' 或 'index' (不区分大小写)
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    symbol_type = symbol_type.lower()
    print(f"传递给 read_day_from_tushare 的 symbol_type: {symbol_type} (类型: {type(symbol_type)})")  # 调试输出
    print(f"尝试通过 Tushare 获取{symbol_type}数据: {symbol_code}")
    
    # 添加断言，确保 symbol_type 是 'stock' 或 'index'
    assert symbol_type in ['stock', 'index'], "symbol_type 必须是 'stock' 或 'index'"
    
    try:
        pro = get_tushare_pro()
        if symbol_type == 'stock':
            # 获取股票日线数据
            df = pro.daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的股票数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        elif symbol_type == 'index':
            # 获取指数日线数据，使用 index_daily 接口
            df = pro.index_daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的指数数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列，处理可能缺失的字段
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        print(f"通过 Tushare 获取了 {len(df)} 条记录。")
        print(f"数据框的列：{df.columns.tolist()}")
        print(f"数据框前5行：\n{df.head()}")
        return df
    except AssertionError as ae:
        print(f"断言错误：{ae}")
        return pd.DataFrame()
    except Exception as e:
        print(f"通过 Tushare 获取数据失败：{e}")
        return pd.DataFrame()


def select_time(df, start_time='20230101', end_time='20240910'):
    """
    根据指定的时间范围筛选数据。
    参数:
    - df: 包含日期索引的DataFrame
    - start_time: 起始时间 (字符串, 格式 'YYYYMMDD')
    - end_time: 截止时间 (字符串, 格式 'YYYYMMDD')
    返回:
    - 筛选后的DataFrame
    """
    print(f"筛选日期范围: {start_time} 至 {end_time}")
    try:
        start_time = pd.to_datetime(start_time, format='%Y%m%d')
        end_time = pd.to_datetime(end_time, format='%Y%m%d')
    except Exception as e:
        print(f"日期转换错误：{e}")
        return pd.DataFrame()
    df_filtered = df.loc[start_time:end_time]
    print(f"筛选后数据长度: {len(df_filtered)}")
    return df_filtered
