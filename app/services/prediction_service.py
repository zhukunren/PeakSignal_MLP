"""
预测服务模块

处理模型预测和组合搜索相关的业务逻辑
"""
from typing import Tuple, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from itertools import product
from ml_trader.config_loader import get_config
from ml_trader.evaluation.signal_quality import evaluate_combo_quality
from ml_trader.logging_config import get_logger
from ml_trader.models.predictor import predict_new_data


logger = get_logger(__name__)


class PredictionService:
    """预测服务类"""

    @staticmethod
    def _unpack_model_candidate(candidate):
        """Support both legacy tuples and metadata-rich candidate dictionaries."""
        if isinstance(candidate, dict):
            return (
                candidate["model"],
                candidate["scaler"],
                candidate.get("selector"),
                candidate["selected_features"],
                candidate["threshold"],
                {k: v for k, v in candidate.items() if k not in {
                    "model", "scaler", "selector", "selected_features", "threshold"
                }},
            )

        model, scaler, selector, selected_features, threshold = candidate[:5]
        return model, scaler, selector, selected_features, threshold, {}

    def search_best_combination(
        self,
        peak_models: List,
        trough_models: List,
        data: pd.DataFrame,
        N: int,
        mixture_depth: int,
        pred_start: str,
        pred_end: str,
        progress_callback: Optional[Callable] = None,
        selection_metric: Optional[str] = None,
        signal_tolerance_days: Optional[int] = None,
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
        if selection_metric is None:
            selection_metric = get_config("prediction.combo_search.selection_metric", "quality")
        if signal_tolerance_days is None:
            signal_tolerance_days = int(get_config("prediction.combo_search.signal_tolerance_days", 5))

        model_combinations = list(product(peak_models, trough_models))
        total_combos = len(model_combinations)
        logger.info(
            "Searching best model combination: peak=%s trough=%s total=%s N=%s mixture_depth=%s "
            "pred_start=%s pred_end=%s selection_metric=%s tolerance_days=%s",
            len(peak_models),
            len(trough_models),
            total_combos,
            N,
            mixture_depth,
            pred_start,
            pred_end,
            selection_metric,
            signal_tolerance_days,
        )

        best_excess = -np.inf
        best_score = -np.inf
        best_models = None
        best_quality_metrics = {}
        first_error = None

        for idx, (peak_m, trough_m) in enumerate(model_combinations):
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_combos,
                    f"正在测试第 {idx+1}/{total_combos} 组模型..."
                )

            pm, ps, psel, pfeats, pth, peak_meta = self._unpack_model_candidate(peak_m)
            tm, ts, tsel, tfeats, tth, trough_meta = self._unpack_model_candidate(trough_m)

            try:
                eval_result_df, bt_result, _ = predict_new_data(
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
                quality_metrics = evaluate_combo_quality(
                    eval_result_df,
                    bt_result,
                    tolerance_days=signal_tolerance_days,
                )
                if selection_metric == "excess_return":
                    current_score = current_excess
                else:
                    current_score = quality_metrics.get("final_score", -np.inf)
                if current_score is None or not np.isfinite(current_score):
                    current_score = -np.inf

                if current_score > best_score:
                    best_score = current_score
                    best_excess = current_excess
                    best_quality_metrics = quality_metrics
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
                        'mixture_depth': mixture_depth,
                        'selection_metric': selection_metric,
                        'selection_score': current_score,
                        'selection_quality_metrics': quality_metrics,
                        'peak_candidate_meta': peak_meta,
                        'trough_candidate_meta': trough_meta,
                        'peak_train_window': peak_meta.get("train_window"),
                        'peak_train_start': peak_meta.get("train_start"),
                        'peak_train_end': peak_meta.get("train_end"),
                        'peak_seed': peak_meta.get("seed"),
                        'peak_round': peak_meta.get("round"),
                        'trough_train_window': trough_meta.get("train_window"),
                        'trough_train_start': trough_meta.get("train_start"),
                        'trough_train_end': trough_meta.get("train_end"),
                        'trough_seed': trough_meta.get("seed"),
                        'trough_round': trough_meta.get("round"),
                    }

            except Exception as e:
                if first_error is None:
                    first_error = e
                logger.debug(
                    "Model combination failed: index=%s total=%s peak_meta=%s trough_meta=%s",
                    idx + 1,
                    total_combos,
                    peak_meta,
                    trough_meta,
                    exc_info=True,
                )
                continue

        if best_models is None:
            logger.error("All model combinations failed: first_error=%s", first_error)
            raise ValueError(f"所有组合均失败。首个错误: {first_error}")

        logger.info(
            "Best model combination selected: score=%s excess=%s peak_meta=%s trough_meta=%s",
            best_score,
            best_excess,
            best_models.get("peak_candidate_meta"),
            best_models.get("trough_candidate_meta"),
        )

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
        bt_result = dict(bt_result)
        bt_result["组合筛选指标"] = selection_metric
        bt_result["组合评分"] = best_score
        bt_result["信号质量评分"] = best_quality_metrics.get("signal_score")
        bt_result["低点附近命中率"] = best_quality_metrics.get("trough_event_recall")
        bt_result["高点附近命中率"] = best_quality_metrics.get("peak_event_recall")
        bt_result["中段误报率"] = best_quality_metrics.get("mid_zone_false_signal_rate")
        bt_result["Peak模型训练窗口"] = best_models.get("peak_train_window")
        bt_result["Peak模型训练起始"] = best_models.get("peak_train_start")
        bt_result["Peak模型训练结束"] = best_models.get("peak_train_end")
        bt_result["Peak模型seed"] = best_models.get("peak_seed")
        bt_result["Peak模型轮次"] = best_models.get("peak_round")
        bt_result["Trough模型训练窗口"] = best_models.get("trough_train_window")
        bt_result["Trough模型训练起始"] = best_models.get("trough_train_start")
        bt_result["Trough模型训练结束"] = best_models.get("trough_train_end")
        bt_result["Trough模型seed"] = best_models.get("trough_seed")
        bt_result["Trough模型轮次"] = best_models.get("trough_round")
        logger.info(
            "Best combination prediction completed: score=%s excess=%s trades=%s",
            best_score,
            best_excess,
            bt_result.get("交易笔数"),
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
