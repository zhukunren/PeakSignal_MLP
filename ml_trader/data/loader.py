import tushare as ts
import pandas as pd
import numpy as np
import os
from datetime import datetime

from ml_trader.logging_config import get_logger


logger = get_logger(__name__)


def get_tushare_pro():
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置 TUSHARE_TOKEN 环境变量，再调用 Tushare 数据接口。")
    ts.set_token(token)
    return ts.pro_api()

def read_day_from_tushare(symbol_code, symbol_type='index', start_date='19920101', end_date=None):
    """
    使用 Tushare API 获取股票或指数的全部日线行情数据。
    参数:
    - symbol_code: 股票或指数代码 (如 "000001.SZ" 或 "000300.SH")
    - symbol_type: 'stock' 或 'index' (不区分大小写)
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    symbol_type = symbol_type.lower()
    end_date = datetime.now().strftime("%Y%m%d") if end_date is None else str(end_date)
    logger.info(
        "Fetching Tushare data: symbol=%s type=%s start=%s end=%s",
        symbol_code,
        symbol_type,
        start_date,
        end_date,
    )
    
    # 添加断言，确保 symbol_type 是 'stock' 或 'index'
    assert symbol_type in ['stock', 'index'], "symbol_type 必须是 'stock' 或 'index'"
    
    try:
        pro = get_tushare_pro()
        if symbol_type == 'stock':
            # 获取股票日线数据
            df = pro.daily(ts_code=symbol_code, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("Tushare returned empty stock data: symbol=%s", symbol_code)
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
            df = pro.index_daily(ts_code=symbol_code, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("Tushare returned empty index data: symbol=%s", symbol_code)
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
        logger.info(
            "Fetched Tushare data: symbol=%s rows=%s columns=%s",
            symbol_code,
            len(df),
            df.columns.tolist(),
        )
        logger.debug("Tushare head:\n%s", df.head())
        return df
    except AssertionError as ae:
        logger.exception("Invalid Tushare request: %s", ae)
        return pd.DataFrame()
    except Exception as e:
        logger.exception("Failed to fetch Tushare data: %s", e)
        return pd.DataFrame()
    

def select_time(df, start_time='20230101', end_time='20240910'):
    start_time = pd.to_datetime(start_time, format='%Y%m%d')
    end_time = pd.to_datetime(end_time, format='%Y%m%d')

    # 不设置索引，只转列到 Datetime
    if 'TradeDate' in df.columns:
        df['TradeDate'] = pd.to_datetime(df['TradeDate'], format='%Y%m%d')
    else:
        # 如果真没有 'TradeDate' 列，就当索引是 Datetime
        df.index = pd.to_datetime(df.index)

    # 仍可排序（可根据 TradeDate 列或索引）
    df.sort_values('TradeDate', inplace=True)
    
    # 用 boolean mask 筛选
    mask = (df['TradeDate'] >= start_time) & (df['TradeDate'] <= end_time)
    df_filtered = df.loc[mask].copy()

    logger.info(
        "Selected time range: start=%s end=%s rows=%s",
        df_filtered['TradeDate'].min(),
        df_filtered['TradeDate'].max(),
        len(df_filtered),
    )
    logger.debug("Selected data head:\n%s", df_filtered.head())
    return df_filtered




