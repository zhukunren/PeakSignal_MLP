"""
应用配置模块

从 YAML 配置文件加载应用配置
"""
from ml_trader.config_loader import config


class AppConfig:
    """应用配置类"""

    # 训练配置
    SEED_BASE = config.get("training.seed_base", 7300)
    BEST_ROUND = config.get("training.best_round", 8)
    NUM_ROUNDS = config.get("training.num_rounds", 10)
    DEFAULT_TRAIN_START = config.get("training.default_train_start", "2000-01-01")
    DEFAULT_TRAIN_END = config.get("training.default_train_end", "2020-12-31")
    ENABLE_MULTI_WINDOW_POOL = config.get("training.enable_multi_window_pool", False)
    TRAINING_WINDOWS = config.get("training.windows", [])

    # 预测配置
    DEFAULT_PRED_START = config.get("prediction.default_pred_start", "2021-01-01")
    USE_BEST_COMBO = config.get("prediction.use_best_combo", True)

    # 特征工程配置
    DEFAULT_N = config.get("features.N", 20)
    DEFAULT_MIXTURE_DEPTH = config.get("features.mixture_depth", 1)

    # 回测配置
    INITIAL_CAPITAL = config.get("backtest.initial_capital", 1000000)
    COMMISSION_RATE = config.get("backtest.commission_rate", 0.0003)

    # UI配置
    PAGE_TITLE = config.get("app.page_title", "东吴秀享AI超额收益系统")
    PAGE_LAYOUT = config.get("app.layout", "wide")
    SIDEBAR_STATE = config.get("app.initial_sidebar_state", "auto")

    # 数据源选项
    DATA_SOURCES = config.get("ui.data_sources", ["指数", "股票"])
    CLASSIFIERS = config.get("ui.classifiers", ["MLP", "Transformer"])
    OVERSAMPLE_OPTIONS = config.get("ui.oversample_options", [
        "SMOTE", "ADASYN", "BorderlineSMOTE", "时间感知过采样"
    ])

    @classmethod
    def get_training_config(cls):
        """获取训练配置字典"""
        return {
            "seed_base": cls.SEED_BASE,
            "best_round": cls.BEST_ROUND,
            "num_rounds": cls.NUM_ROUNDS,
            "default_train_start": cls.DEFAULT_TRAIN_START,
            "default_train_end": cls.DEFAULT_TRAIN_END,
            "enable_multi_window_pool": cls.ENABLE_MULTI_WINDOW_POOL,
            "windows": cls.TRAINING_WINDOWS,
        }

    @classmethod
    def get_prediction_config(cls):
        """获取预测配置字典"""
        return {
            "default_pred_start": cls.DEFAULT_PRED_START,
            "use_best_combo": cls.USE_BEST_COMBO,
        }
