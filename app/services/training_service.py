"""
训练服务模块

处理模型训练相关的业务逻辑
"""
from typing import Tuple, List, Dict, Optional, Callable
import pandas as pd
from ml_trader.models.trainer import train_model
from ml_trader.models.architectures import set_seed
from ml_trader.logging_config import get_logger
from app.config import AppConfig


logger = get_logger(__name__)


class TrainingService:
    """训练服务类"""

    def __init__(self):
        self.peak_models_list = []
        self.trough_models_list = []

    @staticmethod
    def _build_model_candidate(
        label: str,
        model,
        scaler,
        selector,
        selected_features,
        threshold,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        metadata = metadata or {}
        return {
            "label": label,
            "model": model,
            "scaler": scaler,
            "selector": selector,
            "selected_features": selected_features,
            "threshold": threshold,
            **metadata,
        }

    @staticmethod
    def _trade_dates(df: pd.DataFrame) -> pd.Series:
        if "TradeDate" in df.columns:
            return pd.to_datetime(
                df["TradeDate"].astype(str).str.replace("-", "", regex=False),
                format="%Y%m%d",
                errors="coerce",
            )
        return pd.to_datetime(df.index, errors="coerce")

    @classmethod
    def filter_training_window(
        cls,
        df: pd.DataFrame,
        start,
        end,
        label_confirmation_window: int,
    ) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Filter a training window. ``latest_confirmed`` excludes the last N rows."""
        if df is None or df.empty:
            return pd.DataFrame(), None, None

        dates = cls._trade_dates(df)
        valid_dates = dates.dropna().sort_values().unique()
        if len(valid_dates) == 0:
            return pd.DataFrame(), None, None

        start_ts = pd.to_datetime(start, errors="coerce") if start else pd.Timestamp(valid_dates[0])
        if isinstance(end, str) and end == "latest_confirmed":
            offset = max(int(label_confirmation_window), 0)
            end_pos = max(0, len(valid_dates) - offset - 1)
            end_ts = pd.Timestamp(valid_dates[end_pos])
        elif end:
            end_ts = pd.to_datetime(end, errors="coerce")
        else:
            end_ts = pd.Timestamp(valid_dates[-1])

        mask = (dates >= start_ts) & (dates <= end_ts)
        return df.loc[mask].copy(), start_ts, end_ts

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
        progress_callback: Optional[Callable] = None,
        model_metadata: Optional[Dict] = None,
        append_to_existing: bool = False,
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

        if not append_to_existing:
            self.peak_models_list.clear()
            self.trough_models_list.clear()

        last_round_models = {}
        model_metadata = model_metadata or {}
        logger.info(
            "Starting multi-round training: rounds=%s N=%s classifier=%s mixture_depth=%s oversample=%s metadata=%s",
            num_rounds,
            N,
            classifier_name,
            mixture_depth,
            oversample_method,
            model_metadata,
        )

        for i in range(num_rounds):
            round_seed = AppConfig.SEED_BASE + i + 1
            set_seed(round_seed)
            logger.info("Training round started: round=%s seed=%s", i + 1, round_seed)
            round_metadata = {
                **model_metadata,
                "seed": round_seed,
                "round": i + 1,
                "classifier_name": classifier_name,
                "oversample_method": oversample_method,
                "N": N,
                "mixture_depth": mixture_depth,
            }

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
            logger.info(
                "Training round completed: round=%s seed=%s peak_score=%.4f peak_threshold=%.4f "
                "trough_score=%.4f trough_threshold=%.4f",
                i + 1,
                round_seed,
                float(peak_best_score),
                float(peak_threshold),
                float(trough_best_score),
                float(trough_threshold),
            )

            # 保存到列表
            self.peak_models_list.append(
                self._build_model_candidate(
                    "Peak",
                    peak_model,
                    peak_scaler,
                    peak_selector,
                    peak_selected_features,
                    peak_threshold,
                    round_metadata,
                )
            )
            self.trough_models_list.append(
                self._build_model_candidate(
                    "Trough",
                    trough_model,
                    trough_scaler,
                    trough_selector,
                    trough_selected_features,
                    trough_threshold,
                    round_metadata,
                )
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
                    'oversample_method': oversample_method,
                    'train_window': model_metadata.get("train_window"),
                    'train_start': model_metadata.get("train_start"),
                    'train_end': model_metadata.get("train_end"),
                }

        logger.info(
            "Multi-round training completed: peak_candidates=%s trough_candidates=%s",
            len(self.peak_models_list),
            len(self.trough_models_list),
        )
        return last_round_models, self.peak_models_list, self.trough_models_list

    def train_multiple_windows(
        self,
        df: pd.DataFrame,
        N: int,
        all_features: List[str],
        classifier_name: str,
        mixture_depth: int,
        n_features_selected,
        oversample_method: str,
        training_windows: Optional[List[Dict]] = None,
        num_rounds: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Dict, List, List]:
        """Train a candidate pool across configured training windows."""
        if num_rounds is None:
            num_rounds = AppConfig.NUM_ROUNDS
        training_windows = training_windows or AppConfig.TRAINING_WINDOWS
        if not training_windows:
            return self.train_multiple_rounds(
                df,
                N,
                all_features,
                classifier_name,
                mixture_depth,
                n_features_selected,
                oversample_method,
                num_rounds=num_rounds,
                progress_callback=progress_callback,
            )

        self.peak_models_list.clear()
        self.trough_models_list.clear()
        last_round_models = {}
        total_steps = len(training_windows) * num_rounds

        for window_index, window in enumerate(training_windows):
            window_name = window.get("name", f"window_{window_index + 1}")
            df_window, start_ts, end_ts = self.filter_training_window(
                df,
                window.get("start"),
                window.get("end"),
                label_confirmation_window=N,
            )
            if df_window.empty:
                logger.error("Training window has no data: window=%s start=%s end=%s", window_name, window.get("start"), window.get("end"))
                raise ValueError(f"训练窗口 {window_name} 没有可用数据。")

            metadata = {
                "train_window": window_name,
                "train_start": start_ts.strftime("%Y-%m-%d") if start_ts is not None else None,
                "train_end": end_ts.strftime("%Y-%m-%d") if end_ts is not None else None,
            }
            logger.info(
                "Training window selected: window=%s start=%s end=%s rows=%s",
                window_name,
                metadata["train_start"],
                metadata["train_end"],
                len(df_window),
            )

            def window_progress(current, total, message, window_index=window_index, window_name=window_name):
                if progress_callback:
                    global_current = window_index * num_rounds + current
                    progress_callback(
                        global_current,
                        total_steps,
                        f"{window_name}: {message}",
                    )

            last_round_models, _, _ = self.train_multiple_rounds(
                df_window,
                N,
                all_features,
                classifier_name,
                mixture_depth,
                n_features_selected,
                oversample_method,
                num_rounds=num_rounds,
                progress_callback=window_progress,
                model_metadata=metadata,
                append_to_existing=True,
            )

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
            'trough_features_count': len(models.get('trough_selected_features', [])),
            'train_window': models.get('train_window')
        }
