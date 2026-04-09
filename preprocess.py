# preprocess.py
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 从外部 function.py 导入技术指标计算函数
# 请确保你的 function.py 文件中包含 compute_RSI, compute_MACD, compute_KD, compute_momentum, compute_ROC, compute_Bollinger_Bands,
# compute_ATR, compute_volatility, compute_OBV, compute_VWAP, compute_MFI, compute_CMF, compute_chaikin_oscillator,
# compute_CCI, compute_williams_r, compute_zscore, compute_ADX, compute_TRIX, compute_ultimate_oscillator, compute_PPO,
# compute_DPO, compute_KST, compute_KAMA, compute_EMA, compute_MoneyFlowIndex, identify_low_troughs, identify_high_peaks,
# compute_SMA, compute_PercentageB, compute_AccumulationDistribution, compute_HighLow_Spread, compute_PriceChannel, compute_RenkoSlope
from function import *
from feature_expanded import generate_features
#import streamlit as st

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
    print(f"相关性过滤后剩余特征数：{len(filtered_features)}")
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
    print(f"PCA降维后生成 {n_components} 个特征。")
    return pca_feature_names

def preprocess_data(
    data: pd.DataFrame,
    N: int,
    mixture_depth: int,
    mark_labels: bool = True,
    min_features_to_select: int = 10,
    max_features_for_mixture: int = 50
):
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

    print("开始预处理数据...")
    # (A) 对数据做排序、索引
    print("开始预处理数据...")
    data = data.sort_values('TradeDate').copy()
    data.index = pd.to_datetime(data['TradeDate'], format='%Y%m%d')
    
    # ----------------- 原有基本特征计算 -----------------
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
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
    data['KST'], data['KST_signal'] = compute_KST(data['Close'], r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15)
    data['KAMA_10'] = compute_KAMA(data['Close'], n=10, pow1=2, pow2=30)
    data['Seasonality'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['one'] = 1

    # ----------------- 新增更多样化特征 -----------------
    data['SMA_10'] = compute_SMA(data['Close'], window=10)
    data['SMA_30'] = compute_SMA(data['Close'], window=30)
    data['EMA_10'] = compute_EMA(data['Close'], span=10)
    data['EMA_30'] = compute_EMA(data['Close'], span=30)
    data['PercentB'] = compute_PercentageB(data['Close'], data['UpperBand'], data['LowerBand'])
    if 'Volume' in data.columns:
        data['AccumDist'] = compute_AccumulationDistribution(data['High'], data['Low'], data['Close'], data['Volume'])
    else:
        data['AccumDist'] = np.nan
    if 'Volume' in data.columns:
        data['MFI_New'] = compute_MoneyFlowIndex(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
    else:
        data['MFI_New'] = np.nan
    data['HL_Spread'] = compute_HighLow_Spread(data['High'], data['Low'])
    price_channel = compute_PriceChannel(data['High'], data['Low'], data['Close'], window=20)
    data['PriceChannel_Mid'] = price_channel['middle_channel']
    data['RenkoSlope'] = compute_RenkoSlope(data['Close'], bricks=3)

    # ------------------ 3) 调用 generate_features 扩充特征 ------------------
    print("[preprocess_data] 调用 generate_features 生成更多特征...")
    pre_cols = set(data.columns)
    data = generate_features(data)  # 这行里会生成额外的列
    post_cols = set(data.columns)
    new_cols = post_cols - pre_cols
    print(f"generate_features 新增特征列数: {len(new_cols)}")

    # ------------------ 4) 打标签 (可选) ------------------
    if mark_labels:
        print("寻找局部高点和低点(仅训练阶段)...")
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
    print("添加计数指标...")
    data['PriceChange'] = data['Close'].diff()
    data['Up'] = np.where(data['PriceChange'] > 0, 1, 0)
    data['Down'] = np.where(data['PriceChange'] < 0, 1, 0)
    data['ConsecutiveUp'] = data['Up'] * (data['Up'].groupby((data['Up'] != data['Up'].shift()).cumsum()).cumcount() + 1)
    data['ConsecutiveDown'] = data['Down'] * (data['Down'].groupby((data['Down'] != data['Down'].shift()).cumsum()).cumcount() + 1)
    window_size = 10
    data['Cross_MA5'] = np.where(data['Close'] > data['MA_5'], 1, 0)
    data['Cross_MA5_Count'] = data['Cross_MA5'].rolling(window=window_size).sum()
    if 'Volume' in data.columns:
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Spike'] = np.where(data['Volume'] > data['Volume_MA_5'] * 1.5, 1, 0)
        data['Volume_Spike_Count'] = data['Volume_Spike'].rolling(window=10).sum()
    else:
        data['Volume_Spike_Count'] = np.nan
    
    print("构建基础因子...")
    data['Close_MA5_Diff'] = data['Close'] - data['MA_5']
    data['Pch'] = data['Close'] / data['Close'].shift(1) - 1
    data['MA5_MA20_Diff'] = data['MA_5'] - data['MA_20']
    data['RSI_Signal'] = data['RSI_14'] - 50
    data['MACD_Diff'] = data['MACD'] - data['MACD_signal']
    band_range = (data['UpperBand'] - data['LowerBand']).replace(0, np.nan)
    data['Bollinger_Position'] = (data['Close'] - data['MiddleBand']) / band_range
    data['Bollinger_Position'] = data['Bollinger_Position'].fillna(0)
    data['K_D_Diff'] = data['K'] - data['D']

    # ------------- 新增扩展指标（新增的指标函数调用） -------------
    data['MACD_Hist'] = compute_MACD_histogram(data['Close'])
    ichimoku = compute_ichimoku(data['High'], data['Low'], data['Close'])
    data['Ichimoku_Tenkan'] = ichimoku['tenkan_sen']
    data['Ichimoku_Kijun'] = ichimoku['kijun_sen']
    data['Ichimoku_SpanA'] = ichimoku['senkou_span_a']
    data['Ichimoku_SpanB'] = ichimoku['senkou_span_b']
    data['Ichimoku_Chikou'] = ichimoku['chikou_span']
    data['Coppock'] = compute_coppock_curve(data['Close'])
    data['Chaikin_Vol'] = compute_chaikin_volatility(data['High'], data['Low'], period=10, ma_period=10)
    if 'Volume' in data.columns:
        data['EOM'] = compute_ease_of_movement(data['High'], data['Low'], data['Volume'], period=14)
    else:
        data['EOM'] = np.nan
    data['Vortex_Pos'], data['Vortex_Neg'] = compute_vortex_indicator(data['High'], data['Low'], data['Close'], period=14)
    data['Annualized_Vol'] = compute_annualized_volatility(data['Close'], period=10, trading_days=252)
    data['Fisher'] = compute_fisher_transform(data['Close'], period=10)
    data['CMO_14'] = compute_CMO(data['Close'], period=14)

    # 计算 VWAP 和 VWAP 偏离指标
    #data['VWAP'] = compute_VWAP(data['High'], data['Low'], data['Close'], data['Volume'])
    data['VWAP_deviation'] = compute_VWAP_deviation(data['High'], data['Low'], data['Close'], data['Volume'])

    # 计算 OBV 指标
    #data['OBV'] = compute_OBV(data['Close'], data['Volume'])
    # 计算 OBV 的移动平均（例如20日均线）和 OBV 交叉信号
    obv, obv_ma = compute_OBV_MA(data['Close'], data['Volume'], ma_period=20)
    data['OBV_MA'] = obv_ma
    data['OBV_cross'] = compute_OBV_cross_signal(data['Close'], data['Volume'], ma_period=20)
    # 计算 OBV 与收盘价的滚动相关系数（背离指标）
    data['OBV_divergence'] = compute_OBV_divergence(data['Close'], data['Volume'], window=20)

    # 计算 累积分布线（A/D Line）
    data['AD'] = compute_AD(data['High'], data['Low'], data['Close'], data['Volume'])

    # 计算 查金振荡器（Chaikin Oscillator）
    data['Chaikin_Osc'] = compute_chaikin_oscillator(data['High'], data['Low'], data['Close'], data['Volume'], short_period=3, long_period=10)

    # 计算 资金流指数（MFI），这里取14日周期
    data['MFI_14'] = compute_MFI(data['High'], data['Low'], data['Close'], data['Volume'], period=14)

    # 计算 成交量价格趋势 (VPT)
    data['VPT'] = compute_VPT(data['Close'], data['Volume'])

    # 计算 易动指标 (EOM)，这里取14日周期
    data['EOM'] = compute_EOM(data['High'], data['Low'], data['Volume'], period=14)

    # ------------------ 6) 构建基础因子 base_features 列表 ------------------
    print("构建基础因子列表 base_features...")
 
    base_features = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown',
        'Cross_MA5_Count', 'Volume_Spike_Count', 'one', 'Close', 'Pch','CCI_20',
        'Williams_%R_14',  'OBV','VWAP', 'ZScore_20', 'Plus_DI', 'Minus_DI',
        'ADX_14','Bollinger_Width', 'Slope_MA5', 'Volume_Change',
        'Price_Mean_Diff','High_Mean_Diff','Low_Mean_Diff',
        #'MA_5','MA_20','MA_50','MA_200','EMA_5','EMA_20',
        'MFI_14','CMF_20','TRIX_15','Ultimate_Osc','Chaikin_Osc','PPO',
        'DPO_20','KST','KST_signal','KAMA_10'
    ]

    print(f"初始 base_features 数量: {len(base_features)}")

    # ------------------ 7) 方差过滤 ------------------

    print("对基础特征进行方差过滤...")
    X_base = data[base_features].fillna(0)
    selector = VarianceThreshold(threshold=0.0001)
    selector.fit(X_base)
    filtered_features = [f for f, s in zip(base_features, selector.get_support()) if s]
    print(f"方差过滤后剩余特征数：{len(filtered_features)}（从{len(base_features)}减少）")
    base_features = filtered_features

    # ------------------ 8) 相关性过滤 ------------------
    print("对基础特征进行相关性过滤...")
    corr_matrix = data[base_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    base_features = [f for f in base_features if f not in to_drop]
    print(f"相关性过滤后剩余特征数：{len(base_features)}")
  
    # ------------------ 9) 若 mixture_depth > 1, 生成混合因子 ------------------
    print(f"生成混合因子, mixture_depth = {mixture_depth}")
    if mixture_depth > 1:
        operators = ['+', '-', '*', '/']
        mixed_features = base_features.copy()
        current_depth_features = base_features.copy()

        for depth in range(2, mixture_depth + 1):
            print(f"生成深度 {depth} 的混合因子...")
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
                        print(f"无法计算特征 {new_feature_name}，错误：{e}")

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

        # 最后做 PCA 降维
        print("进行 PCA 降维...")
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

    print(f"最终特征数量：{len(all_features)}")

    # ------------------ 10) 检查关键列 ------------------

    required_cols = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff'
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"列 {col} 未被创建，请检查数据和计算步骤。")

    # ------------------ 11) 删除缺失值 & 返回 ------------------
    print("删除缺失值...")
    initial_length = len(data)
    #data = data.dropna().copy()
    final_length = len(data)
    data.index.name = 'date_index'
    #print(f"数据预处理前长度: {initial_length}, 数据预处理后长度: {final_length}")
    print("all_features:",all_features)
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