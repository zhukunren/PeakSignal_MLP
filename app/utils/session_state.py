"""
Session State 管理工具

封装 Streamlit session_state 的访问和初始化
"""
import streamlit as st
from typing import Any, Optional
import pandas as pd


class SessionStateManager:
    """Session State 管理器"""

    # 键名常量
    MODELS = 'models'
    PEAK_MODELS_LIST = 'peak_models_list'
    TROUGH_MODELS_LIST = 'trough_models_list'
    TRAIN_DF = 'train_df_preprocessed'
    TRAIN_FEATURES = 'train_all_features'
    NEW_DF_RAW = 'new_df_raw'
    NEW_DF_DISPLAY = 'new_df_display'

    @staticmethod
    def initialize():
        """初始化 session state"""
        defaults = {
            'trained': False,
            SessionStateManager.MODELS: {},
            'best_models': None,
            SessionStateManager.PEAK_MODELS_LIST: [],
            SessionStateManager.TROUGH_MODELS_LIST: [],
            SessionStateManager.TRAIN_DF: None,
            SessionStateManager.TRAIN_FEATURES: None,
            SessionStateManager.NEW_DF_RAW: None,
            SessionStateManager.NEW_DF_DISPLAY: None,
            'final_result': None,
            'final_bt': {},
            'final_trades_df': pd.DataFrame(),
            'base_prediction_result': None,
            'selected_prediction_models': None,
            'base_selection_bt': {},
            'inc_final_result': None,
            'inc_final_bt': {},
            'finetune_params': {},
            'peak_models_finetuned_list': [],
            'trough_models_finetuned_list': [],
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """获取 session state 值"""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """设置 session state 值"""
        st.session_state[key] = value

    @staticmethod
    def save_models(models: dict):
        """保存模型到 session state"""
        st.session_state[SessionStateManager.MODELS] = models

    @staticmethod
    def get_models() -> Optional[dict]:
        """获取保存的模型"""
        return st.session_state.get(SessionStateManager.MODELS)

    @staticmethod
    def save_training_data(df, features):
        """保存训练数据"""
        st.session_state[SessionStateManager.TRAIN_DF] = df
        st.session_state[SessionStateManager.TRAIN_FEATURES] = features

    @staticmethod
    def get_training_data():
        """获取训练数据"""
        return (
            st.session_state.get(SessionStateManager.TRAIN_DF),
            st.session_state.get(SessionStateManager.TRAIN_FEATURES)
        )

    @staticmethod
    def save_model_lists(peak_models, trough_models):
        """保存模型列表"""
        st.session_state[SessionStateManager.PEAK_MODELS_LIST] = peak_models
        st.session_state[SessionStateManager.TROUGH_MODELS_LIST] = trough_models

    @staticmethod
    def get_model_lists():
        """获取模型列表"""
        return (
            st.session_state.get(SessionStateManager.PEAK_MODELS_LIST, []),
            st.session_state.get(SessionStateManager.TROUGH_MODELS_LIST, [])
        )

    @staticmethod
    def clear():
        """清空所有 session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
