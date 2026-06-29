"""
训练服务模块

处理模型训练相关的业务逻辑
"""
from typing import Tuple, List, Dict, Optional, Callable
import pandas as pd
from ml_trader.models.trainer import train_model
from ml_trader.models.architectures import set_seed
from app.config import AppConfig


class TrainingService:
    """训练服务类"""

    def __init__(self):
        self.peak_models_list = []
        self.trough_models_list = []

    def train_multiple_rounds(
        self,
        df: pd.DataFrame,
        N: int,
        all_features: List[str],
        classifier_name: str,
        mixture_depth: int,
        n_features_selected,
        oversample_method: str,
        num_rounds: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict, List, List]:
        """
        执行多轮训练

        Args:
            df: 训练数据
            N: 峰谷识别窗口
            all_features: 所有特征列表
            classifier_name: 分类器名称（MLP/Transformer）
            mixture_depth: 混合因子深度
            n_features_selected: 选择的特征数量（或'auto'）
            oversample_method: 过采样方法
            num_rounds: 训练轮数（默认从配置读取）
            progress_callback: 进度回调函数 callback(current, total, message)

        Returns:
            (最后一轮模型字典, 峰模型列表, 谷模型列表)
        """
        if num_rounds is None:
            num_rounds = AppConfig.NUM_ROUNDS

        self.peak_models_list.clear()
        self.trough_models_list.clear()

        last_round_models = {}

        for i in range(num_rounds):
            round_seed = AppConfig.SEED_BASE + i + 1
            set_seed(round_seed)

            if progress_callback:
                progress_callback(
                    i + 1,
                    num_rounds,
                    f"正在训练第 {i+1}/{num_rounds} 组模型，seed={round_seed}..."
                )

            # 训练模型
            result = train_model(
                df,
                N,
                all_features,
                classifier_name,
                mixture_depth,
                n_features_selected,
                oversample_method
            )

            (peak_model, peak_scaler, peak_selector, peak_selected_features,
             all_features_peak, peak_best_score, peak_metrics, peak_threshold,
             trough_model, trough_scaler, trough_selector, trough_selected_features,
             all_features_trough, trough_best_score, trough_metrics, trough_threshold) = result

            # 保存到列表
            self.peak_models_list.append(
                (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
            )
            self.trough_models_list.append(
                (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
            )

            # 保存最后一轮模型
            if i == num_rounds - 1:
                last_round_models = {
                    'peak_model': peak_model,
                    'peak_scaler': peak_scaler,
                    'peak_selector': peak_selector,
                    'peak_selected_features': peak_selected_features,
                    'peak_threshold': peak_threshold,
                    'trough_model': trough_model,
                    'trough_scaler': trough_scaler,
                    'trough_selector': trough_selector,
                    'trough_selected_features': trough_selected_features,
                    'trough_threshold': trough_threshold,
                    'N': N,
                    'mixture_depth': mixture_depth,
                    'seed_base': AppConfig.SEED_BASE,
                    'target_round': AppConfig.BEST_ROUND,
                    'classifier_name': classifier_name,
                    'oversample_method': oversample_method
                }

        return last_round_models, self.peak_models_list, self.trough_models_list

    def get_model_summary(self, models: Dict) -> Dict[str, any]:
        """
        获取模型摘要信息

        Args:
            models: 模型字典

        Returns:
            摘要信息字典
        """
        return {
            'classifier': models.get('classifier_name', 'Unknown'),
            'N': models.get('N', 0),
            'mixture_depth': models.get('mixture_depth', 0),
            'seed_base': models.get('seed_base', 0),
            'target_round': models.get('target_round', 0),
            'peak_features_count': len(models.get('peak_selected_features', [])),
            'trough_features_count': len(models.get('trough_selected_features', []))
        }
