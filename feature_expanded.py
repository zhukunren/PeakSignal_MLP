import numpy as np
import pandas as pd

#############################################################################
#                          1) 各类技术指标计算函数                           #
#############################################################################

def compute_RSI(series, period=14):
    """相对强弱指数 (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """MACD 指标, 返回 (macd, signal)"""
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_Bollinger_Bands(series, period=20, num_std=2):
    """布林带: 返回(上轨, 中轨, 下轨)"""
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def compute_KD(high, low, close, period=14):
    """KD 指标(用于KDJ)"""
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-9) * 100
    K = rsv.ewm(com=2).mean()
    D = K.ewm(com=2).mean()
    return K, D

def compute_ATR(high, low, close, period=14):
    """ATR 平均真实波幅"""
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_ADX(high, low, close, period=14):
    """ADX (平均趋向指数), 返回 (+DI, -DI, ADX)"""
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * (-down_move)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    tr_sum = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_sum.replace(0, 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_sum.replace(0, 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(window=period).mean()
    return plus_di, minus_di, adx

def compute_CCI(high, low, close, period=20):
    """CCI 商品通道指标"""
    tp = (high + low + close) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    cci = (tp - ma) / (0.015 * md.replace(0,1e-9))
    return cci

def compute_momentum(series, period=10):
    """动量"""
    return series.diff(period)

def compute_ROC(series, period=10):
    """ROC 变化率(%)"""
    return series.pct_change(period) * 100

def compute_volume_change(volume, period=10):
    """成交量变化率"""
    return volume.diff(period) / volume.shift(period)

def compute_VWAP(high, low, close, volume):
    """成交量加权平均价"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def compute_zscore(series, period=20):
    """Z-Score"""
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return (series - mean) / std.replace(0, 1e-9)

def compute_volatility(series, period=10):
    """波动率: rolling std of returns"""
    ret = series.pct_change()
    return ret.rolling(window=period).std()

def compute_OBV(close, volume):
    """OBV 平衡成交量"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  
    obv = (volume * direction).cumsum()
    return obv

def compute_williams_r(high, low, close, period=14):
    """Williams %R"""
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    wr = -100 * ((hh - close) / (hh - ll + 1e-9))
    return wr

def compute_MFI(high, low, close, volume, period=14):
    """MFI 资金流量指标"""
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_flow = mf.where(tp > tp.shift(), 0)
    negative_flow = mf.where(tp < tp.shift(), 0)
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    mfi = 100 * positive_sum / (positive_sum + negative_sum + 1e-9)
    return mfi

def compute_CMF(high, low, close, volume, period=20):
    """CMF (Chaikin Money Flow)"""
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def compute_TRIX(series, period=15):
    """TRIX 三重指数平滑平均"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = (ema3.diff() / ema3.shift().replace(0,1e-9)) * 100
    return trix

def compute_ultimate_oscillator(high, low, close, short_period=7, medium_period=14, long_period=28):
    """UO 终极震荡指标"""
    bp = close - np.minimum(low.shift(1), close.shift(1))
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    avg_s = bp.rolling(short_period).sum() / tr.rolling(short_period).sum().replace(0,1e-9)
    avg_m = bp.rolling(medium_period).sum() / tr.rolling(medium_period).sum().replace(0,1e-9)
    avg_l = bp.rolling(long_period).sum() / tr.rolling(long_period).sum().replace(0,1e-9)
    uo = 100 * ((4 * avg_s) + (2 * avg_m) + avg_l) / (4 + 2 + 1)
    return uo

def compute_chaikin_oscillator(high, low, close, volume, short_period=3, long_period=10):
    """Chaikin Osc (基于ADL的MACD式指标)"""
    adl_line = compute_ADL_line(high, low, close, volume)
    short_ema = adl_line.ewm(span=short_period, adjust=False).mean()
    long_ema = adl_line.ewm(span=long_period, adjust=False).mean()
    return short_ema - long_ema

def compute_ADL_line(high, low, close, volume):
    """ADL 累积/派发线"""
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    adl = mf_volume.cumsum()
    return adl

def compute_PPO(series, fast_period=12, slow_period=26):
    """PPO (百分比价格振荡器)"""
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    ppo = (fast_ema - slow_ema) / slow_ema.replace(0,1e-9) * 100
    return ppo

def compute_DPO(series, period=20):
    """DPO 去趋势价格振荡器"""
    sma = series.rolling(window=period).mean()
    dpo = series.shift(int((period/2)+1)) - sma
    return dpo

def compute_KST(series, r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15):
    """KST (Know Sure Thing)"""
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
    """KAMA 自适应移动平均"""
    change = (series - series.shift(n)).abs()
    volatility = series.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0,1e-9)
    sc = (er * (2/(pow1+1) - 2/(pow2+1)) + 2/(pow2+1))**2

    kama = series.copy()
    for i in range(n, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(series.iloc[i]-kama.iloc[i-1])
    return kama

def compute_SMA(series, window):
    """简单移动平均 (SMA)"""
    return series.rolling(window=window, min_periods=1).mean()

def compute_EMA(series, span):
    """指数移动平均 (EMA)"""
    return series.ewm(span=span, adjust=False).mean()

def compute_PercentageB(close, upper_band, lower_band):
    """Bollinger %B"""
    band_range = upper_band - lower_band
    band_range = band_range.replace(0, np.nan)
    percent_b = (close - lower_band) / band_range
    return percent_b.fillna(0)

def compute_AccumulationDistribution(high, low, close, volume):
    """A/D 累积派发线"""
    denominator = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denominator
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

def compute_MoneyFlowIndex(high, low, close, volume, period=14):
    """MFI资金流量指标 (另一种实现)"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfi = 100 * positive_mf / (positive_mf + negative_mf + 1e-9)
    return mfi.fillna(50)

def compute_HighLow_Spread(high, low):
    """日内价差 (High - Low)"""
    return high - low

def compute_PriceChannel(high, low, close, window=20):
    """价格通道(上轨,下轨,中轨)"""
    upper_channel = high.rolling(window=window, min_periods=1).max()
    lower_channel = low.rolling(window=window, min_periods=1).min()
    middle_channel = (upper_channel + lower_channel) / 2
    df_channel = pd.DataFrame({
        'upper_channel': upper_channel,
        'middle_channel': middle_channel,
        'lower_channel': lower_channel
    })
    return df_channel

def compute_RenkoSlope(close, bricks=3):
    """Renko块简易趋势指标"""
    price_diff = close.diff()
    renko = price_diff.apply(lambda x: 1 if x >= bricks else (-1 if x <= -bricks else 0))
    slope = renko.rolling(window=5, min_periods=1).sum()
    return slope

#############################################################################
#                          2) 生成统一的 100+ 特征                          #
#############################################################################

def generate_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    生成并返回含 100+ 列的特征 DataFrame。
    要求原始 df_input 至少包含:
      - 'Open','High','Low','Close'
      - 'Volume' (如果没有, volume相关指标会返回 NaN)

    返回:
      df_feat: 扩增后含有各类技术指标、衍生特征的 DataFrame
    """
    df = df_input.copy()

    # -----------------------------------------------------------------
    #  (A) 先保留并简单派生一些基础列
    # -----------------------------------------------------------------
    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    # 日收益率 (简单涨跌幅)
    df['Return_1d'] = df['Close'].pct_change()
    # 日对数收益
    df['LogReturn_1d'] = np.log(df['Close'] / df['Close'].shift(1))

    # -----------------------------------------------------------------
    #  (B) 多周期 SMA / EMA
    # -----------------------------------------------------------------
    sma_windows = [5, 10, 20, 30, 60]
    for w in sma_windows:
        df[f'SMA_{w}'] = compute_SMA(df['Close'], w)

    ema_windows = [5, 10, 20, 30, 60]
    for w in ema_windows:
        df[f'EMA_{w}'] = compute_EMA(df['Close'], w)

    # -----------------------------------------------------------------
    #  (C) 多周期 RSI
    # -----------------------------------------------------------------
    rsi_periods = [6, 14, 21]
    for p in rsi_periods:
        df[f'RSI_{p}'] = compute_RSI(df['Close'], period=p)

    # -----------------------------------------------------------------
    #  (D) 多种 MACD 组合
    # -----------------------------------------------------------------
    macd_sets = [(12,26,9), (8,17,9), (16,32,9)]
    for (f,s,sg) in macd_sets:
        macd, signal = compute_MACD(df['Close'], f, s, sg)
        df[f'MACD_{f}_{s}'] = macd
        df[f'MACDsig_{f}_{s}'] = signal

    # -----------------------------------------------------------------
    #  (E) 多周期 Bollinger
    # -----------------------------------------------------------------
    boll_sets = [(20,2), (20,3)]
    for (period, std) in boll_sets:
        up, mid, low = compute_Bollinger_Bands(df['Close'], period=period, num_std=std)
        df[f'BollUp_{period}_{std}'] = up
        df[f'BollMid_{period}_{std}'] = mid
        df[f'BollLow_{period}_{std}'] = low

    # -----------------------------------------------------------------
    #  (F) KD / KDJ
    # -----------------------------------------------------------------
    kd_periods = [9, 14]
    for p in kd_periods:
        k, d = compute_KD(df['High'], df['Low'], df['Close'], period=p)
        df[f'K_{p}'] = k
        df[f'D_{p}'] = d
        df[f'J_{p}'] = 3 * k - 2 * d

    # -----------------------------------------------------------------
    #  (G) ATR, ADX, CCI, MFI 等
    # -----------------------------------------------------------------
    atr_periods = [14, 21]
    for p in atr_periods:
        df[f'ATR_{p}'] = compute_ATR(df['High'], df['Low'], df['Close'], p)

    adx_periods = [14, 21]
    for p in adx_periods:
        plus_di, minus_di, adx_val = compute_ADX(df['High'], df['Low'], df['Close'], p)
        df[f'plusDI_{p}'] = plus_di
        df[f'minusDI_{p}'] = minus_di
        df[f'ADX_{p}'] = adx_val

    cci_periods = [14, 20, 30]
    for p in cci_periods:
        df[f'CCI_{p}'] = compute_CCI(df['High'], df['Low'], df['Close'], period=p)

    # 资金流量指标 MFI
    mfi_periods = [14, 21]
    if 'Volume' in df.columns:
        for p in mfi_periods:
            df[f'MFI_{p}'] = compute_MFI(df['High'], df['Low'], df['Close'], df['Volume'], p)
    else:
        for p in mfi_periods:
            df[f'MFI_{p}'] = np.nan

    # CMF
    if 'Volume' in df.columns:
        df['CMF_20'] = compute_CMF(df['High'], df['Low'], df['Close'], df['Volume'], 20)
    else:
        df['CMF_20'] = np.nan

    # -----------------------------------------------------------------
    #  (H) Momentum, ROC, Volatility, OBV, WR
    # -----------------------------------------------------------------
    momentum_periods = [3,7,14]
    for p in momentum_periods:
        df[f'Momentum_{p}'] = compute_momentum(df['Close'], p)

    roc_periods = [5,10,20]
    for p in roc_periods:
        df[f'ROC_{p}'] = compute_ROC(df['Close'], p)

    vol_periods = [5,10,20]
    for p in vol_periods:
        df[f'Volatility_{p}'] = compute_volatility(df['Close'], p)

    if 'Volume' in df.columns:
        df['OBV'] = compute_OBV(df['Close'], df['Volume'])
    else:
        df['OBV'] = np.nan

    wr_periods = [14, 21]
    for p in wr_periods:
        df[f'WilliamsR_{p}'] = compute_williams_r(df['High'], df['Low'], df['Close'], period=p)

    # -----------------------------------------------------------------
    #  (I) VWAP, VolumeChange
    # -----------------------------------------------------------------
    if 'Volume' in df.columns:
        df['VWAP'] = compute_VWAP(df['High'], df['Low'], df['Close'], df['Volume'])
        for p in [5,10,20]:
            df[f'VolumeChange_{p}'] = compute_volume_change(df['Volume'], period=p)
    else:
        df['VWAP'] = np.nan
        for p in [5,10,20]:
            df[f'VolumeChange_{p}'] = np.nan

    # -----------------------------------------------------------------
    #  (J) zscore, TRIX, UO, ChaikinOsc, PPO, DPO, KST, KAMA
    # -----------------------------------------------------------------
    zscore_periods = [10, 20]
    for p in zscore_periods:
        df[f'ZScore_{p}'] = compute_zscore(df['Close'], period=p)

    trix_periods = [15, 30]
    for p in trix_periods:
        df[f'TRIX_{p}'] = compute_TRIX(df['Close'], p)

    df['UO'] = compute_ultimate_oscillator(df['High'], df['Low'], df['Close'], 7, 14, 28)
    
    if 'Volume' in df.columns:
        df['Chaikin_Osc'] = compute_chaikin_oscillator(df['High'], df['Low'], df['Close'], df['Volume'], 3,10)
    else:
        df['Chaikin_Osc'] = np.nan

    df['PPO'] = compute_PPO(df['Close'], fast_period=12, slow_period=26)

    dpo_windows = [20,30]
    for w in dpo_windows:
        df[f'DPO_{w}'] = compute_DPO(df['Close'], w)

    kst, kst_signal = compute_KST(df['Close'], 10,15,20,30, 10,10,10,15)
    df['KST'] = kst
    df['KST_signal'] = kst_signal

    df['KAMA_10'] = compute_KAMA(df['Close'], n=10, pow1=2, pow2=30)
    df['KAMA_30'] = compute_KAMA(df['Close'], n=30, pow1=2, pow2=30)

    # -----------------------------------------------------------------
    #  (K) 其它价量衍生
    # -----------------------------------------------------------------
    df['HighLow_Spread'] = compute_HighLow_Spread(df['High'], df['Low'])
    pc = compute_PriceChannel(df['High'], df['Low'], df['Close'], window=20)
    df['PC_Upper_20'] = pc['upper_channel']
    df['PC_Mid_20'] = pc['middle_channel']
    df['PC_Lower_20'] = pc['lower_channel']

    df['RenkoSlope_3'] = compute_RenkoSlope(df['Close'], bricks=3)

    # 用布林带 20,2 做 %B
    up_boll, mid_boll, low_boll = compute_Bollinger_Bands(df['Close'], 20, 2)
    df['PercentB_20_2'] = compute_PercentageB(df['Close'], up_boll, low_boll)

    # -----------------------------------------------------------------
    #  (L) 一些额外的交叉/衍生特征 (示例)
    # -----------------------------------------------------------------
    df['Close_div_EMA20'] = df['Close'] / (df['EMA_20']+1e-9)
    df['High_minus_Low'] = df['High'] - df['Low']
    df['MACD_12_26_minus_RSI_14'] = df['MACD_12_26'] - df['RSI_14']
    df['ATR_14_x_Volume'] = df['ATR_14'] * df['Volume']
    df['Return_5d'] = df['Close'].pct_change(5)

    # 还可以多加一些 Lag/Lead
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Volume_lag1'] = df['Volume'].shift(1)

    # -----------------------------------------------------------------
    # 在返回前，对缺失值进行简单填充（或您可用其他方式处理）
    # -----------------------------------------------------------------
    df.fillna(0, inplace=True)

    # 打印一下最终特征数量(可选)
    # print("Final Feature Count:", len(df.columns))

    return df