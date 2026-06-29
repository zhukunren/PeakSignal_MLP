"""
预测服务模块

处理模型预测和组合搜索相关的业务逻辑
"""
from typing import Tuple, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from itertools import product
from ml_trader.models.predictor import predict_new_data


class PredictionService:
    """预测服务类"""

    def search_best_combination(
        self,
        peak_models: List,
        trough_models: List,
        data: pd.DataFrame,
        N: int,
        mixture_depth: int,
        pred_start: str,
        pred_end: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict, float, pd.DataFrame, Dict]:
        """
        搜索最佳模型组合

        Args:
            peak_models: 峰模型列表
            trough_models: 谷模型列表
            data: 预测数据
            N: 峰谷识别窗口
            mixture_depth: 混合因子深度
            pred_start: 预测开始日期
            pred_end: 预测结束日期
            progress_callback: 进度回调函数

        Returns:
            (最佳模型字典, 最佳超额收益率, 预测结果DataFrame, 回测结果字典)
        """
        model_combinations = list(product(peak_models, trough_models))
        total_combos = len(model_combinations)

        best_excess = -np.inf
        best_models = None
        first_error = None

        for idx, (peak_m, trough_m) in enumerate(model_combinations):
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_combos,
                    f"正在测试第 {idx+1}/{total_combos} 组模型..."
                )

            pm, ps, psel, pfeats, pth = peak_m
            tm, ts, tsel, tfeats, tth = trough_m

            try:
                _, bt_result, _ = predict_new_data(
                    data,
                    pm, ps, psel, pfeats, pth,
                    tm, ts, tsel, tfeats, tth,
                    N=N,
                    mixture_depth=mixture_depth,
                    window_size=10,
                    eval_mode=True,
                    N_buy=1,
                    N_sell=1,
                    N_newhigh=60,
                    enable_chase=False,
                    enable_stop_loss=False,
                    enable_change_signal=False,
                    backtest_start_date=pred_start,
                    backtest_end_date=pred_end,
                )

                current_excess = bt_result.get('超额收益率', -np.inf)

                if current_excess > best_excess:
                    best_excess = current_excess
                    best_models = {
                        'peak_model': pm,
                        'peak_scaler': ps,
                        'peak_selector': psel,
                        'peak_selected_features': pfeats,
                        'peak_threshold': pth,
                        'trough_model': tm,
                        'trough_scaler': ts,
                        'trough_selector': tsel,
                        'trough_selected_features': tfeats,
                        'trough_threshold': tth,
                        'N': N,
                        'mixture_depth': mixture_depth
                    }

            except Exception as e:
                if first_error is None:
                    first_error = e
                continue

        if best_models is None:
            raise ValueError(f"所有组合均失败。首个错误: {first_error}")

        # 用最佳组合做完整预测
        result_df, bt_result, trades_df = predict_new_data(
            data,
            best_models['peak_model'],
            best_models['peak_scaler'],
            best_models['peak_selector'],
            best_models['peak_selected_features'],
            best_models['peak_threshold'],
            best_models['trough_model'],
            best_models['trough_scaler'],
            best_models['trough_selector'],
            best_models['trough_selected_features'],
            best_models['trough_threshold'],
            N=N,
            mixture_depth=mixture_depth,
            window_size=10,
            eval_mode=False,
            N_buy=1,
            N_sell=1,
            N_newhigh=60,
            enable_chase=False,
            enable_stop_loss=False,
            enable_change_signal=False,
            backtest_start_date=pred_start,
            backtest_end_date=pred_end,
        )

        return best_models, best_excess, result_df, bt_result

    def predict_with_strategy(
        self,
        data: pd.DataFrame,
        models: Dict,
        pred_start: str,
        pred_end: str,
        N_buy: int = 10,
        N_sell: int = 10,
        N_newhigh: int = 60,
        enable_chase: bool = False,
        enable_stop_loss: bool = False,
        enable_change_signal: bool = False
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        使用指定策略参数进行预测

        Args:
            data: 预测数据
            models: 模型字典
            pred_start: 预测开始日期
            pred_end: 预测结束日期
            N_buy: 追涨天数
            N_sell: 止损天数
            N_newhigh: 新高天数
            enable_chase: 是否启用追涨
            enable_stop_loss: 是否启用止损
            enable_change_signal: 是否启用信号调整

        Returns:
            (预测结果DataFrame, 回测结果字典, 交易记录DataFrame)
        """
        return predict_new_data(
            data,
            models['peak_model'],
            models['peak_scaler'],
            models['peak_selector'],
            models['peak_selected_features'],
            models['peak_threshold'],
            models['trough_model'],
            models['trough_scaler'],
            models['trough_selector'],
            models['trough_selected_features'],
            models['trough_threshold'],
            N=models.get('N', 20),
            mixture_depth=models.get('mixture_depth', 1),
            window_size=10,
            eval_mode=False,
            N_buy=N_buy,
            N_sell=N_sell,
            N_newhigh=N_newhigh,
            enable_chase=enable_chase,
            enable_stop_loss=enable_stop_loss,
            enable_change_signal=enable_change_signal,
            backtest_start_date=pred_start,
            backtest_end_date=pred_end,
        )
