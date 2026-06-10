import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_candlestick(data, stock_code, start_date, end_date, 
                     peaks=None, troughs=None, 
                     prediction=False, selected_classifiers=None, 
                     bt_result=None):
    # 确保索引是日期类型并过滤有效日期范围
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # 过滤数据只保留在 start_date 到 end_date 之间的数据
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    print(data.head(5))
    # 建立双子图：上方K线，下方成交量
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,              # 共享 X 轴，放大 K 线时成交量的 X 也一起缩放
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],         # 上下子图高度比例
        specs=[[{"type": "candlestick"}],
               [{"type": "bar"}]]
    )

    # (1) 绘制 K 线图
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=stock_code,
        increasing=dict(line=dict(color='red')),
        decreasing=dict(line=dict(color='green')),
        hoverinfo='x+y+text'
    ), row=1, col=1)

    # (2) 绘制成交量
    if 'Volume' in data.columns:
        volume_colors = [
            'red' if row['Close'] > row['Open'] else 'green'
            for _, row in data.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=volume_colors,
            name='成交量',
            hoverinfo='x+y'
        ), row=2, col=1)

    # (3) 标注高点、低点
    if peaks is not None and not peaks.empty:
        marker_y_peaks = peaks['High'] * 1.02
        marker_x_peaks = peaks.index
        color_peak = 'green'
        label_peak = '局部高点' if not prediction else '预测高点'
        fig.add_trace(go.Scatter(
            x=marker_x_peaks,
            y=marker_y_peaks,
            mode='text',
            text='W',
            textfont=dict(color=color_peak, size=20),
            name=label_peak
        ), row=1, col=1)

    if troughs is not None and not troughs.empty:
        marker_y_troughs = troughs['Low'] * 0.98
        marker_x_troughs = troughs.index
        color_trough = 'red'
        label_trough = '局部低点' if not prediction else '预测低点'
        fig.add_trace(go.Scatter(
            x=marker_x_troughs,
            y=marker_y_troughs,
            mode='text',
            text='D',
            textfont=dict(color=color_trough, size=20),
            name=label_trough
        ), row=1, col=1)

    # (4) 如果含有 trade 列，则标注买/卖点
    if 'trade' in data.columns:
        for idx, row in data.iterrows():
            if row['trade'] == 'buy':
                fig.add_trace(go.Scatter(
                    x=[idx],
                    y=[row['Low'] * 0.98],
                    mode='markers+text',
                    text='b',
                    textfont=dict(color='white', size=16),
                    marker=dict(
                        color='green',
                        size=20,
                        line=dict(color='green', width=2)
                    ),
                    name='Buy',
                    showlegend=False,
                    legendgroup='trade',
                    textposition='middle center',
                ), row=1, col=1)
            elif row['trade'] == 'sell':
                fig.add_trace(go.Scatter(
                    x=[idx],
                    y=[row['High'] * 1.02],
                    mode='markers+text',
                    text='s',
                    textfont=dict(color='white', size=16),
                    marker=dict(
                        color='red',
                        size=20,
                        line=dict(color='red', width=2)
                    ),
                    name='Sell',
                    showlegend=False,
                    legendgroup='trade',
                    textposition='middle center',
                ), row=1, col=1)

        # 在图例上添加 Buy/Sell 标识
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='green', size=20, line=dict(color='green', width=2)),
            name='Buy',
            showlegend=True,
            legendgroup='trade',
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='red', size=20, line=dict(color='red', width=2)),
            name='Sell',
            showlegend=True,
            legendgroup='trade',
        ))

    # (5) 如果有回测结果，注释到图上
    if bt_result:
        annotations = []
        y_pos = 0.95
        for key, value in bt_result.items():
            if isinstance(value, float):
                # 某些字段以百分比显示
                if key in {"同期标的涨跌幅", '"波段盈"累计收益率', "超额收益率", 
                           "单笔交易最大收益", "单笔交易最低收益", "单笔平均收益率", "胜率"}:
                    value = f"{value*100:.2f}%"
                else:
                    value = f"{value:.2f}"
                annotations.append(dict(
                    xref='paper', yref='paper',
                    x=0.05, y=1-y_pos,
                    text=f"{key}: {value}",
                    showarrow=False,
                    align='left'
                ))
                y_pos -= 0.05

        fig.update_layout(annotations=annotations)

    # (6) 全局布局设置：鼠标十字光标 + 框选放大
    fig.update_layout(
        title=f"{stock_code} {start_date} 至 {end_date}",
        height=800,
        hovermode='x unified',  # 统一十字光标
        template='plotly_white',
        dragmode='zoom'         # 用户可框选放大
    )

    # (7) X轴：中文日期格式，不显示 rangeslider，启用 spike
    fig.update_xaxes(
        rangeslider_visible=False,
        tickformat="%Y年%m月%d日",  # 中文日期格式
        showspikes=True,
        spikedash='solid',
        spikemode='across',
        spikesnap='cursor'
    )

    # (8) Y轴分别自动范围、显示光标线
    #    row=1, col=1 对应上方K线；row=2, col=1 对应下方成交量
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        showspikes=True,
        spikedash='solid',
        spikemode='across',
        spikesnap='cursor',
        row=1, col=1
    )
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        showspikes=True,
        spikedash='solid',
        spikemode='across',
        spikesnap='cursor',
        row=2, col=1
    )

    return fig