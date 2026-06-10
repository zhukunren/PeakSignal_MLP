
import numpy as np
from ml_trader.features.indicators import *
import pandas as pd
import plotly.graph_objects as go

'''
def build_trades_from_signals(df, signal_df):

    # 修改列名转换方式：先复制，再修改列名
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    
    signal_df = signal_df.copy()
    signal_df.columns = signal_df.columns.astype(str).str.lower()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if not isinstance(signal_df.index, pd.DatetimeIndex):
        signal_df.index = pd.to_datetime(signal_df.index)

    signal_df = signal_df.reindex(df.index).fillna('')

    trades = []
    holding = False
    entry_date = None
    entry_price = None
    dates = df.index.to_list()

    for i in range(len(dates) - 1):  
        today = dates[i]
        next_day = dates[i + 1]
        direction_signal = signal_df.loc[today, 'direction'] if 'direction' in signal_df.columns else ''

        if not holding:
            if direction_signal == 'buy':

                # 下一交易日开盘买入
                holding = True
                entry_date = next_day
                entry_price = df.loc[next_day, 'open']
        else:
            if direction_signal == 'sell':
                # 下一交易日开盘卖出
                holding = False
                exit_date = next_day
                exit_price = df.loc[next_day, 'open']
                trade_return = exit_price / entry_price - 1 if entry_price else None
                hold_days = (df.index.get_loc(exit_date) - df.index.get_loc(entry_date))
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'hold_days': hold_days,
                    'return': trade_return
                })
                entry_date = None
                entry_price = None
    # 在for循环结束后，检查是否还在持仓
    if holding:
        holding = False
        exit_date = dates[-1]  # 最后一天
        exit_price = df.loc[exit_date, 'close']  # 或者 open, 或者您想要的价格
        trade_return = exit_price / entry_price - 1 if entry_price else None
        hold_days = (df.index.get_loc(exit_date) - df.index.get_loc(entry_date))
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'hold_days': hold_days,
            'return': trade_return
        })
        entry_date = None
        entry_price = None

    #print(pd.DataFrame(trades))
    return pd.DataFrame(trades)
'''

def build_trades_from_signals(df, signal_df, N_buy, N_sell, enable_chase=False, enable_stop_loss=False):
    # 复制并统一列名为小写
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    
    signal_df = signal_df.copy()
    signal_df.columns = signal_df.columns.astype(str).str.lower()

    # 确保索引是 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if not isinstance(signal_df.index, pd.DatetimeIndex):
        signal_df.index = pd.to_datetime(signal_df.index)
    
    # 让 signal_df 与 df 的日期对齐，空缺部分填充为空字符
    signal_df = signal_df.reindex(df.index).fillna('')

    # 初始化变量
    trades = []
    holding = False
    entry_date = None
    entry_price = None
    exit_date = None
    exit_price = None
    holding_days = 0
    signal_type_buy = None
    signal_type_sell = None
    wait_days = 0

    dates = df.index.to_list()  # 将日期索引转换为列表

    # 主循环
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i + 1]

        # 获取当日信号
        direction_signal = ''
        if 'direction' in signal_df.columns:
            direction_signal = signal_df.loc[today, 'direction']

        if not holding:
            # 如果启用追涨，先判断是否满足追涨条件
            if enable_chase and exit_date is not None:
                wait_days = df.index.get_loc(today) - df.index.get_loc(exit_date)
                if wait_days >= N_buy and (df.loc[today, 'high'] / exit_price) >= 1:
                    holding = True
                    entry_date = today
                    holding_days += 1
                    signal_type_buy = '追涨'
                    try:
                        if df.loc[today, 'open'] > exit_price:
                            entry_price = df.loc[today, 'open']
                        else:
                            entry_price = exit_price
                    except Exception as e:
                        print('处理追涨信号出错：', e)
                    # 进入交易后跳过后续逻辑
                    continue
            
            # 如果没有追涨信号或者未启用追涨，则按照买入信号简单逻辑：隔日开盘价买入
            if direction_signal == 'buy':
                try:
                    holding = True
                    entry_date = next_day
                    entry_price = df.loc[next_day, 'open']
                    holding_days = 0
                    signal_type_buy = '信号'
                except Exception as e:
                    print("处理buy信号报错：", e)
        else:
            # 持仓中
            holding_days += 1
            try:
                if enable_stop_loss:
                    # 1) 止损逻辑：持仓 N_sell 天后，当日低价跌破买入价则止损
                    if holding_days >= N_sell and (df.loc[today, 'low'] / entry_price) < 1:
                        holding = False
                        # 当日开盘价小于买入价，则以开盘价平仓，否则按买入价平仓
                        if df.loc[today, 'open'] < entry_price:
                            exit_date = today
                            exit_price = df.loc[today, 'open']
                        else:
                            exit_date = today
                            exit_price = entry_price
                        signal_type_sell = '止损'
                        trade_return = (exit_price / entry_price - 1) if entry_price else None
                        hold_days = df.index.get_loc(exit_date) - df.index.get_loc(entry_date)
                        trades.append({
                            'entry_date': entry_date,
                            'signal_type_buy': signal_type_buy,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'signal_type_sell': signal_type_sell,
                            'exit_price': exit_price,
                            'hold_days': hold_days,
                            'return': trade_return,
                            'signal_loc': df.index.get_loc(today),
                        })
                        entry_date = None
                        entry_price = None
                        holding_days = 0
                    # 2) 正常卖出逻辑：满足持仓天数条件且当日低价不跌破买入价，同时出现卖出信号，则隔日以开盘价卖出
                    elif direction_signal == 'sell' and holding_days >= N_sell and (df.loc[today, 'low'] / entry_price) >= 1:
                        holding = False
                        exit_date = next_day
                        exit_price = df.loc[next_day, 'open']
                        signal_type_sell = '信号'
                        trade_return = (exit_price / entry_price - 1) if entry_price else None
                        hold_days = df.index.get_loc(exit_date) - df.index.get_loc(entry_date)
                        trades.append({
                            'entry_date': entry_date,
                            'signal_type_buy': signal_type_buy,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'signal_type_sell': signal_type_sell,
                            'exit_price': exit_price,
                            'hold_days': hold_days,
                            'return': trade_return,
                            'signal_loc': df.index.get_loc(today),
                        })
                        entry_date = None
                        entry_price = None
                        holding_days = 0
                else:
                    # 未启用止损，则按照简单逻辑：一旦出现卖出信号，则隔日以开盘价卖出
                    if direction_signal == 'sell':
                        holding = False
                        exit_date = next_day
                        exit_price = df.loc[next_day, 'open']
                        signal_type_sell = '信号'
                        trade_return = (exit_price / entry_price - 1) if entry_price else None
                        hold_days = df.index.get_loc(exit_date) - df.index.get_loc(entry_date)
                        trades.append({
                            'entry_date': entry_date,
                            'signal_type_buy': signal_type_buy,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'signal_type_sell': signal_type_sell,
                            'exit_price': exit_price,
                            'hold_days': hold_days,
                            'return': trade_return,
                            'signal_loc': df.index.get_loc(today),
                        })
                        entry_date = None
                        entry_price = None
                        holding_days = 0
            except Exception as e:
                print("处理持仓逻辑错误：", e)

    # 回测最后一天，如果还在持仓，则在最后一天收盘价平仓
    if holding:
        try:
            exit_date = dates[-1]
            exit_price = df.loc[exit_date, 'close']
            trade_return = (exit_price / entry_price - 1) if entry_price else None
            hold_days = df.index.get_loc(exit_date) - df.index.get_loc(entry_date)
            trades.append({
                'entry_date': entry_date,
                'signal_type_buy': signal_type_buy,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'signal_type_sell': '最后平仓',
                'exit_price': exit_price,
                'hold_days': hold_days,
                'return': trade_return,
                'signal_loc': df.index.get_loc(today),
            })
        except Exception as e:
            print("处理最后持仓错误：", e)

    trades_df = pd.DataFrame(trades)
    return trades_df

def build_daily_equity_curve(df, trades_df, initial_capital=1_000_000):
    # 正确复制 DataFrame 并转换列名小写
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    
    trades_df = trades_df.copy()
    trades_df.columns = trades_df.columns.astype(str).str.lower()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    equity_curve = pd.DataFrame(index=df.index, columns=['equity'], data=np.nan)
    equity_curve.iloc[0, 0] = initial_capital

    # 标记每个交易日是否持仓
    position = pd.Series(data=0, index=df.index)  # 0=空仓,1=持仓
    for idx, row in trades_df.iterrows():
        ed = row['entry_date']
        xd = row['exit_date']
        if pd.isna(ed) or ed not in df.index:
            continue
        if pd.isna(xd) or xd not in df.index:
            # 如果exit_date为空或不在index里，就持仓到最后
            position.loc[ed:] = 1
        else:
            ed_loc = df.index.get_loc(ed)
            xd_loc = df.index.get_loc(xd)
            position.iloc[ed_loc:xd_loc] = 1

    # 计算每日涨跌幅
    df['daily_ret'] = df['close'].pct_change().fillna(0.0)

    # 逐日生成净值
    for i in range(1, len(df)):
        equity_curve.iloc[i, 0] = equity_curve.iloc[i-1, 0]
        if position.iloc[i-1] == 1:
            daily_ret = df['daily_ret'].iloc[i]
            equity_curve.iloc[i, 0] *= (1 + daily_ret)

    return equity_curve

def backtest_results(df, signal_df, N_buy,N_sell,enable_chase, enable_stop_loss,initial_capital=1_000_000):
    """
    返回一个 dict，包含所有回测结果的指标，并返回 trades_df、equity_curve、以及净值曲线图。
    
    参数
    -------
    df: 行情数据（index=交易日，含 'open','close' 列）
    signal_df: 含有 'direction' ('buy','sell') 列的信号
    N_backtest: 回测中自定义的持仓天数/强制平仓参考长度
    initial_capital: 初始资金
    
    返回
    -------
    result: dict，包含以下字段：
      '同期标的涨跌幅'、'"累计收益率'、'超额收益率'、
      '单笔交易最大收益'、'单笔交易最低收益'、'单笔平均收益率'、
      '收益率为正的交易笔数'、'收益率为负的交易笔数'、'持仓天数'、
      '空仓天数'、'交易笔数'、'胜率'、'最大回撤'、'年化夏普比率'
    trades_df: DataFrame，所有交易的明细（开仓日、平仓日、持仓天数、每笔收益等）
    equity_curve: DataFrame，按日记录的净值序列
    equity_fig: Plotly Figure，绘制的净值曲线图
    """
    # ---------- (1) 构建交易表 ----------
    trades_df = build_trades_from_signals(df, signal_df, N_buy,N_sell,enable_chase,enable_stop_loss)
    
    # ---------- (2) 构建日度净值 ----------
    equity_curve = build_daily_equity_curve(df, trades_df, initial_capital=initial_capital)

    # ---------- (3) 计算所有需要的指标 ----------

    # ========== 3.1 同期标的涨跌幅 ==========
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    signal_df = signal_df.copy()
    signal_df.columns = signal_df.columns.astype(str).str.lower()

    first_day = df.index[0]
    last_day = df.index[-1]
    if 'open' in df.columns and 'close' in df.columns:
        start_price = df.loc[first_day, 'open']
        end_price   = df.loc[last_day, 'close']
        if start_price != 0:
            benchmark_return = end_price / start_price - 1
        else:
            benchmark_return = np.nan
    else:
        benchmark_return = np.nan

    # ========== 3.2 策略累计收益率 ==========
    start_equity = equity_curve['equity'].iloc[0]
    end_equity   = equity_curve['equity'].iloc[-1]
    strategy_return = end_equity / start_equity - 1  # 波段盈

    # ========== 3.3 超额收益 ==========
    if not np.isnan(benchmark_return):
        excess_return = strategy_return - benchmark_return
    else:
        excess_return = np.nan

    # ========== 3.4 单笔交易相关 ==========
    if trades_df.empty:
        max_trade = None
        min_trade = None
        avg_trade = None
        pos_trades = 0
        neg_trades = 0
        num_trades = 0
        win_rate   = None
    else:
        max_trade = trades_df['return'].max()
        min_trade = trades_df['return'].min()
        avg_trade = trades_df['return'].mean()
        num_trades = len(trades_df)
        pos_trades = (trades_df['return'] > 0).sum()
        neg_trades = (trades_df['return'] < 0).sum()
        win_rate   = pos_trades / num_trades if num_trades > 0 else None

    # ========== 3.5 持仓天数、空仓天数 ==========
    position = pd.Series(data=0, index=df.index)
    for idx, row in trades_df.iterrows():
        ed = row['entry_date']
        xd = row['exit_date']
        if pd.isna(ed) or ed not in df.index:
            continue
        if pd.isna(xd) or xd not in df.index:
            # 如果exit_date为空或不在index中，则持仓到最后
            position.loc[ed:] = 1
        else:
            ed_loc = df.index.get_loc(ed)
            xd_loc = df.index.get_loc(xd)
            position.iloc[ed_loc:xd_loc] = 1

    holding_days = (position == 1).sum()
    noholding_days = (position == 0).sum()

    # ========== 3.6 计算最大回撤、夏普比率 ==========
    # 1) 最大回撤
    rolling_max = equity_curve['equity'].cummax()
    drawdown = equity_curve['equity'] / rolling_max - 1
    max_drawdown = drawdown.min()  # 取最小值即最大回撤
    
    # 2) 夏普比率（年化）
    daily_returns = equity_curve['equity'].pct_change().fillna(0)
    if daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    # ========== 3.7 整理输出 ========== 
    result = {
        '同期标的涨跌幅': benchmark_return,
        '累计收益率': strategy_return,
        '超额收益率': excess_return,
        '单笔交易最大收益': max_trade,
        '单笔交易最低收益': min_trade,
        '单笔平均收益率': avg_trade,
        '收益率为正的交易笔数': pos_trades,
        '收益率为负的交易笔数': neg_trades,
        '持仓天数': holding_days,
        '空仓天数': noholding_days,
        '交易笔数': num_trades,
        '胜率': win_rate,
        '最大回撤': max_drawdown,
        '年化夏普比率': sharpe_ratio
    }

    # ---------- (4) 绘制净值曲线图（Plotly） ----------
    equity_fig = go.Figure()
    equity_fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='净值曲线'
        )
    )
    equity_fig.update_layout(
        title='策略净值曲线',
        xaxis_title='日期',
        yaxis_title='净值',
        template='plotly_white',
        hovermode='x unified'
    )

    return result, trades_df
