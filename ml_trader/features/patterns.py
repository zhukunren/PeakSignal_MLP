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
