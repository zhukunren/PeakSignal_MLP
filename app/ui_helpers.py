import logging

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
import pickle
import io
from itertools import product
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import copy  # 新增，用于克隆初始模型
import time
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError
from ml_trader.models.architectures import set_seed
from ml_trader.data.preprocessor import preprocess_data, create_pos_neg_sequences_by_consecutive_labels
from ml_trader.models.trainer import train_model
from ml_trader.models.predictor import (
    predict_event_regime_model_data,
    predict_new_data,
    get_trade_signal,
    change_trough_and_peak,
)
from ml_trader.trading.backtest import backtest_results
from ml_trader.data.loader import read_day_from_tushare, select_time
from ml_trader.visualization.plots import plot_candlestick
from ml_trader.models.architectures import time_aware_oversampling
from ml_trader.logging_config import get_logger

TARGET_REPRO_SEED_BASE = 7300
TARGET_REPRO_BEST_ROUND = 8
TARGET_PRED_END = datetime.now()
logger = get_logger(__name__)

# 设置随机种子
set_seed(42)


class _ClosedWebSocketLogFilter(logging.Filter):
    """Hide noisy asyncio logs when Streamlit writes to a disconnected browser."""

    marker = "_closed_websocket_log_filter"

    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, (StreamClosedError, WebSocketClosedError)):
            message = record.getMessage()
            if "Task exception was never retrieved" in message:
                return False
        return True


def install_closed_websocket_log_filter():
    logger = logging.getLogger("asyncio")
    for existing_filter in logger.filters:
        if getattr(existing_filter, _ClosedWebSocketLogFilter.marker, False):
            return

    log_filter = _ClosedWebSocketLogFilter()
    setattr(log_filter, _ClosedWebSocketLogFilter.marker, True)
    logger.addFilter(log_filter)


class StreamlitProgressReporter:
    def __init__(self, min_interval=0.25, min_fraction_delta=0.01):
        self.progress_bar = st.progress(0.0)
        self.status_text = st.empty()
        self.min_interval = min_interval
        self.min_fraction_delta = min_fraction_delta
        self._last_update_at = 0.0
        self._last_fraction = None
        self._last_message = None
        self._closed = False

    def _should_update(self, fraction, force):
        if force or self._last_fraction is None or fraction >= 1.0:
            return True
        if abs(fraction - self._last_fraction) >= self.min_fraction_delta:
            return True
        return time.monotonic() - self._last_update_at >= self.min_interval

    def update(self, current, total, message=None, force=False):
        if self._closed:
            return

        total = max(float(total), 1.0)
        fraction = min(max(float(current) / total, 0.0), 1.0)
        if not self._should_update(fraction, force):
            return

        try:
            if message is not None and message != self._last_message:
                self.status_text.text(message)
                self._last_message = message
            self.progress_bar.progress(fraction)
            self._last_fraction = fraction
            self._last_update_at = time.monotonic()
        except Exception:
            self._closed = True

    def finish(self, message=None):
        self.update(1, 1, message, force=True)

    def clear(self):
        if self._closed:
            return
        try:
            self.progress_bar.empty()
            self.status_text.empty()
        except Exception:
            pass
        finally:
            self._closed = True


def inject_orientation_script():
    orientation_script = """
    <style>
    #rotate-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        z-index: 9999;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 24px;
    }
    </style>
    <div id="rotate-overlay">
      请旋转手机至横屏模式使用
    </div>
    <script>
    function checkOrientation() {
        if (window.innerHeight > window.innerWidth) {
            document.getElementById('rotate-overlay').style.display = 'flex';
        } else {
            document.getElementById('rotate-overlay').style.display = 'none';
        }
    }
    window.addEventListener('resize', checkOrientation);
    checkOrientation();
    </script>
    """
    components.html(orientation_script, height=0)

def load_custom_css():
    custom_css = """
    <style>
    .strategy-row {
        margin-bottom: 8px;
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    .strategy-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }
    @media only screen and (max-width: 768px) {
        .strategy-row {
            flex-direction: column;
            align-items: flex-start;
        }
        .strategy-label {
            justify-content: flex-start;
            margin-bottom: 4px;
        }
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
            overflow-x: auto;
        }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def normalize_market_data(df):
    base_cols = ["Open", "High", "Low", "Close", "Volume", "Amount", "TradeDate"]
    keep_cols = [col for col in base_cols if col in df.columns]
    out = df[keep_cols].copy()
    if "TradeDate" not in out.columns:
        return pd.DataFrame(columns=base_cols)
    out["TradeDate"] = out["TradeDate"].astype(str).str.replace("-", "", regex=False)
    for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["TradeDate", "Open", "High", "Low", "Close"])


def read_front_market_data(symbol_code, symbol_type, end_date):
    """
    对齐组合训练脚本的数据口径：默认指数 000001.SH 优先复用本地完整数据.csv，
    再用 Tushare 补齐到 end_date；其他标的仍直接从 Tushare 获取。
    """
    end_date = str(end_date)
    logger.info(
        "Reading market data: symbol=%s type=%s end_date=%s",
        symbol_code,
        symbol_type,
        end_date,
    )

    if symbol_type != "index" or symbol_code != "000001.SH":
        raw = read_day_from_tushare(symbol_code, symbol_type, end_date=end_date)
        raw = normalize_market_data(raw.reset_index(drop=True)) if not raw.empty else raw
        if not raw.empty and raw["TradeDate"].max() < end_date:
            logger.warning(
                "Tushare data ended before requested date: actual_end=%s requested_end=%s",
                raw["TradeDate"].max(),
                end_date,
            )
            st.warning(f"行情数据实际截止到 {raw['TradeDate'].max()}，早于你设置的预测截止日期 {end_date}。")
        return raw

    local_raw = pd.DataFrame()
    try:
        local_raw = normalize_market_data(pd.read_csv("完整数据.csv"))
    except Exception:
        logger.exception("Failed to read local 完整数据.csv")
        local_raw = pd.DataFrame()

    if local_raw.empty:
        logger.info("Local market data is unavailable; falling back to Tushare")
        ts_df = read_day_from_tushare(symbol_code, symbol_type, end_date=end_date)
        raw = normalize_market_data(ts_df.reset_index(drop=True)) if not ts_df.empty else pd.DataFrame()
        if not raw.empty and raw["TradeDate"].max() < end_date:
            logger.warning(
                "Tushare data ended before requested date: actual_end=%s requested_end=%s",
                raw["TradeDate"].max(),
                end_date,
            )
            st.warning(f"行情数据实际截止到 {raw['TradeDate'].max()}，早于你设置的预测截止日期 {end_date}。")
        return raw

    raw = local_raw.copy()
    if raw["TradeDate"].max() < end_date:
        logger.info(
            "Local market data requires Tushare补齐: local_end=%s requested_end=%s",
            raw["TradeDate"].max(),
            end_date,
        )
        ts_df = read_day_from_tushare(symbol_code, symbol_type, end_date=end_date)
        ts_raw = normalize_market_data(ts_df.reset_index(drop=True)) if not ts_df.empty else pd.DataFrame()
        if not ts_raw.empty:
            raw = (
                pd.concat([raw, ts_raw], ignore_index=True)
                .drop_duplicates(subset=["TradeDate"], keep="last")
                .sort_values("TradeDate")
                .reset_index(drop=True)
            )

    raw = raw[raw["TradeDate"] <= end_date].copy()
    if not raw.empty and raw["TradeDate"].max() < end_date:
        logger.warning(
            "Market data ended before requested date: actual_end=%s requested_end=%s",
            raw["TradeDate"].max(),
            end_date,
        )
        st.warning(f"行情数据实际截止到 {raw['TradeDate'].max()}，早于你设置的预测截止日期 {end_date}。")
    return raw.reset_index(drop=True)

# ========== 模型微调的辅助函数 ========== #
def incremental_train_for_label(model, scaler, selected_features, df_new, label_column, classifier_name,
                                window_size=10, oversample_method=None, new_lr=None, new_epochs=5,
                                freeze_option="none", old_df=None, mix_ratio=1.0, progress_bar=None,
                                early_stopping=True, val_size=0.2, patience=3):
    """
    使用新数据对已有模型进行微调训练（partial_fit），支持多种冻结策略和验证集监控：
      - 如果提供了 old_df，则从 old_df 中随机抽取 mix_ratio 倍于新数据样本数的旧数据，与新数据混合训练。
      - new_lr: 微调阶段使用的学习率
      - new_epochs: 对混合数据进行微调的 epoch 数
      - freeze_option: 冻结策略选项 ["none", "first_layer", "second_layer", "all", "partial"] (MLP)
                       或 ["none", "first_layer", "encoder_layers", "output_layer", "all"] (Transformer)
      - early_stopping: 是否启用早停
      - val_size: 验证集比例
      - patience: 早停耐心值，连续多少轮验证集性能未提升则停止
      - progress_bar: 可选，streamlit 的进度条控件，用于显示训练进度
    
    Returns:
        model: 微调后的模型
        best_val_acc: 最佳验证集准确率
        epoch_stopped: 实际训练的轮数
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 1) 提取新数据
    X_new = df_new[selected_features].fillna(0)
    X_new_scaled = scaler.transform(X_new).astype(np.float32)
    y_new = df_new[label_column].astype(int).values

    # 2) 如果提供了旧数据，则进行混合训练
    if old_df is not None:
        sample_size = int(len(X_new) * mix_ratio)
        sample_size = min(sample_size, len(old_df))
        X_old_sample = old_df[selected_features].fillna(0).sample(n=sample_size, random_state=42)
        y_old_sample = old_df[label_column].astype(int).loc[X_old_sample.index].values
        X_old_scaled = scaler.transform(X_old_sample).astype(np.float32)
        X_combined = np.concatenate([X_new_scaled, X_old_scaled], axis=0)
        y_combined = np.concatenate([y_new, y_old_sample], axis=0)
    else:
        X_combined = X_new_scaled
        y_combined = y_new

    # 3) 对于非 Transformer 模型，若需要过采样
    if classifier_name != 'Transformer' and oversample_method is not None and oversample_method not in ["Class Weights", "None"]:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.combine import SMOTEENN, SMOTETomek
        sampler = None
        if oversample_method == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif oversample_method == "ADASYN":
            sampler = ADASYN(random_state=42)
        elif oversample_method == "Borderline-SMOTE":
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif oversample_method == "SMOTEENN":
            sampler = SMOTEENN(random_state=42)
        elif oversample_method == "SMOTETomek":
            sampler = SMOTETomek(random_state=42)
        elif oversample_method == "Time-Aware":
            X_combined, y_combined = time_aware_oversampling(X_combined, y_combined, recency_weight=0.9, sequence_length=60)
            sampler = None
        
        if sampler is not None:
            X_combined, y_combined = sampler.fit_resample(X_combined, y_combined)

    # 4) 如果启用早停，则划分验证集
    if early_stopping:
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined, test_size=val_size, random_state=42, stratify=y_combined
        )
    else:
        X_train, y_train = X_combined, y_combined
        X_val, y_val = None, None

    # 5) 对于 Transformer 模型，将数据转换为时序数据
    if classifier_name == 'Transformer':
        if X_val is not None:
            X_seq_train, y_seq_train = create_pos_neg_sequences_by_consecutive_labels(X_train, y_train)
            X_seq_val, y_seq_val = create_pos_neg_sequences_by_consecutive_labels(X_val, y_val)
            X_input_train, y_input_train = X_seq_train, y_seq_train
            X_input_val, y_input_val = X_seq_val, y_seq_val
        else:
            X_seq, y_seq = create_pos_neg_sequences_by_consecutive_labels(X_train, y_train)
            X_input_train, y_input_train = X_seq, y_seq
            X_input_val, y_input_val = None, None
    else:
        X_input_train, y_input_train = X_train, y_train
        X_input_val, y_input_val = X_val, y_val

    # 6) 调整微调学习率
    if new_lr is not None and hasattr(model, 'optimizer_') and model.optimizer_ is not None:
        for param_group in model.optimizer_.param_groups:
            param_group['lr'] = new_lr

    # 7) 根据选择的冻结策略冻结不同层
    if classifier_name == 'MLP':
        # 解冻所有层（重置）
        for param in model.module_.parameters():
            param.requires_grad = True
            
        if freeze_option == "first_layer":
            # 只冻结第一层
            for param in model.module_.fc1.parameters():
                param.requires_grad = False
        elif freeze_option == "second_layer":
            # 只冻结第二层
            for param in model.module_.fc2.parameters():
                param.requires_grad = False
        elif freeze_option == "all":
            # 冻结所有层
            for param in model.module_.parameters():
                param.requires_grad = False
        elif freeze_option == "partial":
            # 对第一层做部分冻结
            fc1_size = model.module_.fc1.weight.shape[0]
            half_size = fc1_size // 2
            
            weight_mask = torch.ones_like(model.module_.fc1.weight)
            weight_mask[:half_size] = 0  # 冻结前半部分
            def weight_hook(grad):
                return grad * weight_mask
            model.module_.fc1.weight.register_hook(weight_hook)
            
            if model.module_.fc1.bias is not None:
                bias_mask = torch.ones_like(model.module_.fc1.bias)
                bias_mask[:half_size] = 0
                def bias_hook(grad):
                    return grad * bias_mask
                model.module_.fc1.bias.register_hook(bias_hook)

    elif classifier_name == 'Transformer':
        # 解冻所有层（重置）
        for param in model.module_.parameters():
            param.requires_grad = True
            
        if freeze_option == "first_layer":
            # 冻结输入线性层
            for param in model.module_.input_linear.parameters():
                param.requires_grad = False
        elif freeze_option == "encoder_layers":
            # 冻结Transformer编码器层（除最后一层）
            num_layers = len(model.module_.transformer_encoder.layers)
            for i in range(num_layers - 1):
                for param in model.module_.transformer_encoder.layers[i].parameters():
                    param.requires_grad = False
        elif freeze_option == "output_layer":
            # 冻结输出层
            for param in model.module_.fc.parameters():
                param.requires_grad = False
        elif freeze_option == "all":
            # 冻结所有层
            for param in model.module_.parameters():
                param.requires_grad = False

    # 8) 多 epoch 微调，同时更新进度条（如果提供）
    best_val_acc = 0.0
    early_stop_counter = 0
    epoch_stopped = new_epochs
    
    # 检查是否所有参数都被冻结了
    all_frozen = True
    for param in model.module_.parameters():
        if param.requires_grad:
            all_frozen = False
            break
    
    # 如果所有参数都被冻结，则跳过训练过程
    if all_frozen:
        if progress_bar is not None:
            progress_bar.progress(1.0)
        if early_stopping and X_input_val is not None:
            from sklearn.metrics import accuracy_score
            y_val_pred = model.predict(X_input_val)
            best_val_acc = accuracy_score(y_input_val, y_val_pred)
        epoch_stopped = 0
    else:
        from sklearn.metrics import accuracy_score
        for epoch in range(new_epochs):
            model.partial_fit(X_input_train, y_input_train, classes=np.array([0, 1]))
            
            # 在验证集上评估（如果启用早停）
            if early_stopping and X_input_val is not None:
                y_val_pred = model.predict(X_input_val)
                val_acc = accuracy_score(y_input_val, y_val_pred)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= patience:
                    epoch_stopped = epoch + 1
                    break
            
            if progress_bar is not None:
                progress_bar.progress((epoch + 1) / new_epochs)
    
    if early_stopping and best_val_acc == 0 and X_input_val is not None:
        # 若整个循环中都没有提升，最后再评一次
        y_val_pred = model.predict(X_input_val)
        best_val_acc = accuracy_score(y_input_val, y_val_pred)
    
    return model, best_val_acc, epoch_stopped


# ========== 新增：自动阈值调整功能（示例，可根据需要自行修改/调用） ========== #
def auto_adjust_thresholds(model_dict, inc_final_result, target_trade_count=None):
    """
    根据微调后的预测结果，自动调整阈值以达到目标交易次数（简单示例）。
    """
    adjusted_model_dict = model_dict.copy()
    
    # 如果没有设置目标交易次数，就使用原始交易次数（若有）
    if target_trade_count is None and 'final_bt' in st.session_state:
        target_trade_count = st.session_state.final_bt.get('交易笔数', 10)
    elif target_trade_count is None:
        target_trade_count = 10  # 默认目标交易次数
    
    peak_probs = inc_final_result['Peak_Probability'].dropna()
    trough_probs = inc_final_result['Trough_Probability'].dropna()
    
    if not peak_probs.empty and not trough_probs.empty:
        def binary_search_threshold(probs, orig_threshold, min_th=0.5, max_th=0.95):
            left, right = min_th, max_th
            best_threshold = orig_threshold
            best_count_diff = float('inf')
            
            for _ in range(20):  # 最多尝试10轮
                mid = (left + right) / 2
                count = sum(probs >= mid)
                
                # 假设峰信号和谷信号各占一半
                count_diff = abs(count - target_trade_count / 2)  
                
                if count_diff < best_count_diff:
                    best_threshold = mid
                    best_count_diff = count_diff
                
                if count < target_trade_count / 2:
                    right = mid
                else:
                    left = mid
            
            return best_threshold
        
        new_peak_threshold = binary_search_threshold(
            peak_probs,
            adjusted_model_dict['peak_threshold']
        )
        new_trough_threshold = binary_search_threshold(
            trough_probs,
            adjusted_model_dict['trough_threshold']
        )
        
        adjusted_model_dict['peak_threshold'] = new_peak_threshold
        adjusted_model_dict['trough_threshold'] = new_trough_threshold
        
        return adjusted_model_dict, new_peak_threshold, new_trough_threshold
    else:
        return adjusted_model_dict, adjusted_model_dict['peak_threshold'], adjusted_model_dict['trough_threshold']


# ========== 新增：微调效果评估功能 ========== #
def evaluate_finetune_effect(freeze_option):
    """评估模型微调效果并给出改进建议。"""
    st.subheader("微调效果评估")
    
    if 'final_bt' not in st.session_state or 'inc_final_bt' not in st.session_state:
        st.warning("无法评估微调效果，缺少微调前后的回测结果。")
        return
    
    orig_bt = st.session_state.final_bt
    inc_bt = st.session_state.inc_final_bt
    
    # 计算关键指标的变化
    return_change = (inc_bt.get('累计收益率', 0) - orig_bt.get('累计收益率', 0))
    excess_change = (inc_bt.get('超额收益率', 0) - orig_bt.get('超额收益率', 0))
    win_rate_change = (inc_bt.get('胜率', 0) - orig_bt.get('胜率', 0))
    drawdown_change = (inc_bt.get('最大回撤', 0) - orig_bt.get('最大回撤', 0))
    
    # 指标权重（简单示例）
    weights = {
        '收益率': 0.3,
        '超额收益': 0.3,
        '胜率': 0.2,
        '回撤': 0.2
    }
    # 计算综合评分
    def safe_div(num, denom):
        if abs(denom) < 1e-8:
            return 0
        return num / denom

    score = (
        weights['收益率'] * safe_div(return_change, orig_bt.get('累计收益率', 0.01)) +
        weights['超额收益'] * safe_div(excess_change, orig_bt.get('超额收益率', 0.01)) +
        weights['胜率'] * safe_div(win_rate_change, orig_bt.get('胜率', 0.01)) -
        weights['回撤'] * safe_div(drawdown_change, orig_bt.get('最大回撤', 0.01))
    )
    
    if score > 0.1:
        st.success(f"微调效果显著 (评分: {score:.2f})")
    elif score > 0:
        st.info(f"微调效果轻微改善 (评分: {score:.2f})")
    elif score > -0.1:
        st.warning(f"微调效果轻微下降 (评分: {score:.2f})")
    else:
        st.error(f"微调效果显著恶化 (评分: {score:.2f})")
    
    # 给出具体建议
    st.markdown("### 改进建议")
    suggestions = []
    
    if return_change < 0:
        suggestions.append("- **累计收益率下降**: 可能过拟合新数据或学习率过高；可尝试降低学习率或增加旧数据混合比例。")
    
    if win_rate_change < 0:
        suggestions.append("- **胜率下降**: 可能模型变得过于激进或保守；可考虑部分冻结或调整阈值。")
    
    if drawdown_change > 0:
        suggestions.append("- **最大回撤增加**: 说明风险控制变差；可考虑调整止损逻辑或加大风控特征。")
    
    ft_params = st.session_state.finetune_params
    
    if ft_params.get('lr', 1e-4) > 1e-4 and score < 0:
        suggestions.append("- **学习率可能过高**: 建议尝试更低学习率 (1e-5 或更低)。")
    
    if ft_params.get('mix_ratio', 1.0) < 0.5 and score < 0:
        suggestions.append("- **混合旧数据比例过低**: 导致对新数据过拟合；可尝试提高至 0.5-1.0。")
    
    if freeze_option == "all":
        suggestions.append("- **所有层被冻结**: 无法学习新特征；可尝试部分冻结或不冻结。")
    
    if ft_params.get('peak_epochs', 0) < ft_params.get('epochs', 0) * 0.5 and \
       ft_params.get('trough_epochs', 0) < ft_params.get('epochs', 0) * 0.5:
        suggestions.append("- **早停过早**: 验证集准确率停滞；可调整学习率、冻结策略或耐心值。")
    
    if not suggestions:
        if score > 0:
            suggestions.append("- **微调效果良好**：可以考虑加长训练或多批次微调，进一步提升模型。")
        else:
            suggestions.append("- 可尝试不同微调参数组合，如降低学习率、调整冻结策略、增大旧数据比例等。")
    
    for s in suggestions:
        st.markdown(s)


MODEL_EXPORT_REQUIRED_KEYS = (
    "peak_model",
    "peak_scaler",
    "peak_selector",
    "peak_selected_features",
    "peak_threshold",
    "trough_model",
    "trough_scaler",
    "trough_selector",
    "trough_selected_features",
    "trough_threshold",
)


def is_downloadable_model_dict(model_dict):
    return isinstance(model_dict, dict) and all(key in model_dict for key in MODEL_EXPORT_REQUIRED_KEYS)


def with_export_metadata(payload, saved_from, symbol_code):
    export_payload = dict(payload)
    export_payload.update({
        "saved_from": saved_from,
        "symbol_code": symbol_code,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return export_payload


def summarize_candidate_pool_by_window(peak_models, trough_models):
    """Summarize metadata-rich candidate pools by training window."""
    rows_by_window = {}

    def add_candidates(candidates, count_column):
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            key = (
                candidate.get("train_window") or "未标记",
                candidate.get("train_start"),
                candidate.get("train_end"),
            )
            row = rows_by_window.setdefault(
                key,
                {
                    "训练窗口": key[0],
                    "训练起始": key[1],
                    "训练结束": key[2],
                    "峰模型数": 0,
                    "谷模型数": 0,
                },
            )
            row[count_column] += 1

    add_candidates(peak_models, "峰模型数")
    add_candidates(trough_models, "谷模型数")
    rows = list(rows_by_window.values())
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["训练起始", "训练窗口"], na_position="last")
        .reset_index(drop=True)
    )


def render_model_download_options(symbol_code, key_prefix="model_download"):
    """Render model export choices for the model artifacts currently held in session_state."""
    options = []

    current_models = st.session_state.get("models")
    if is_downloadable_model_dict(current_models):
        options.append({
            "label": "当前模型（最新训练/微调）",
            "file_prefix": "current_model",
            "payload": with_export_metadata(current_models, "current_session_model", symbol_code),
        })

    selected_models = st.session_state.get("selected_prediction_models")
    if is_downloadable_model_dict(selected_models):
        selected_payload = with_export_metadata(selected_models, "prediction_best_combo", symbol_code)
        base_selection_bt = st.session_state.get("base_selection_bt", {})
        selection_metric = (
            selected_models.get("selection_metric")
            or base_selection_bt.get("组合筛选指标")
        )
        if selection_metric is not None:
            selected_payload["selection_metric"] = selection_metric
        if base_selection_bt.get("组合评分") is not None:
            selected_payload["selection_score"] = base_selection_bt.get("组合评分")
        selected_payload["strategy_applied_after_selection"] = True
        if st.session_state.get("prediction_cache_key"):
            selected_payload["prediction_cache_key"] = st.session_state.prediction_cache_key
        if base_selection_bt:
            selected_payload["base_selection_bt"] = base_selection_bt
        options.append({
            "label": "预测选中的最佳组合模型",
            "file_prefix": "best_combo_model",
            "payload": selected_payload,
        })

    loaded_models = st.session_state.get("best_models")
    if is_downloadable_model_dict(loaded_models):
        options.append({
            "label": "已加载/缓存模型",
            "file_prefix": "loaded_model",
            "payload": with_export_metadata(loaded_models, "loaded_or_cached_model", symbol_code),
        })

    if st.session_state.get("peak_models_list") and st.session_state.get("trough_models_list"):
        training_payload = dict(current_models) if is_downloadable_model_dict(current_models) else {}
        training_payload.update({
            "peak_models_list": st.session_state.peak_models_list,
            "trough_models_list": st.session_state.trough_models_list,
            "N": st.session_state.models.get("N"),
            "mixture_depth": st.session_state.models.get("mixture_depth"),
            "seed_base": st.session_state.models.get("seed_base", TARGET_REPRO_SEED_BASE),
            "target_round": st.session_state.models.get("target_round", TARGET_REPRO_BEST_ROUND),
        })
        options.append({
            "label": "全部训练候选模型",
            "file_prefix": "all_training_candidates",
            "payload": with_export_metadata(training_payload, "all_training_candidates", symbol_code),
        })

    if st.session_state.get("peak_models_finetuned_list") and st.session_state.get("trough_models_finetuned_list"):
        finetuned_payload = dict(current_models) if is_downloadable_model_dict(current_models) else {}
        finetuned_payload.update({
            "peak_models_finetuned_list": st.session_state.peak_models_finetuned_list,
            "trough_models_finetuned_list": st.session_state.trough_models_finetuned_list,
            "base_model_config": dict(st.session_state.models),
            "finetune_params": st.session_state.get("finetune_params", {}),
        })
        options.append({
            "label": "全部微调候选模型",
            "file_prefix": "all_finetuned_candidates",
            "payload": with_export_metadata(finetuned_payload, "all_finetuned_candidates", symbol_code),
        })

    if not options:
        return

    st.subheader("模型下载")
    labels = [option["label"] for option in options]
    col_choice, col_name = st.columns([2, 3])
    with col_choice:
        selected_label = st.selectbox("下载内容", labels, key=f"{key_prefix}_choice")

    selected_option = options[labels.index(selected_label)]
    default_name = f"{selected_option['file_prefix']}_{symbol_code}_{datetime.now().strftime('%Y%m%d')}"
    with col_name:
        model_name = st.text_input("模型文件名", default_name, key=f"{key_prefix}_name")

    try:
        model_bytes = pickle.dumps(selected_option["payload"])
    except Exception as e:
        logger.exception("Failed to package model for download")
        st.error(f"模型打包失败: {str(e)}")
        return

    st.download_button(
        label="下载模型文件",
        data=model_bytes,
        file_name=f"{model_name}.pkl",
        mime="application/octet-stream",
        key=f"{key_prefix}_button",
    )


# ========== 新增：模型保存功能 ========== #
def add_model_save_functionality(symbol_code):
    """Render persistent model download controls after finetuning."""
    render_model_download_options(symbol_code, key_prefix="finetune_model_download")


def render_prediction_model_download(symbol_code, model_state_key="selected_prediction_models"):
    render_model_download_options(symbol_code, key_prefix=f"download_{model_state_key}")


def apply_strategy_to_prediction(
    base_result,
    n_buy,
    n_sell,
    n_newhigh,
    enable_chase,
    enable_stop_loss,
    enable_change_signal,
):
    """Use cached model predictions to rebuild trades and metrics with strategy settings."""
    if base_result is None or base_result.empty:
        raise ValueError("没有可用的预测缓存，请先完成一次预测。")

    result = base_result.copy()
    for col in ["entry_date", "exit_date", "trade", "date"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    if enable_change_signal:
        result = change_trough_and_peak(result, n_newhigh)

    signal_df = get_trade_signal(result)
    bt_result, trades_df = backtest_results(
        result,
        signal_df,
        n_buy,
        n_sell,
        enable_chase,
        enable_stop_loss,
        initial_capital=1_000_000,
    )
    result = mark_trade_points(result, trades_df)
    return result, bt_result, trades_df


def mark_trade_points(result_df, trades_df):
    result_df = result_df.copy()
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.to_datetime(result_df.index, errors="coerce")
    result_df["trade"] = None
    if trades_df is None or trades_df.empty:
        return result_df

    for _, trade in trades_df.iterrows():
        entry_date = pd.to_datetime(trade.get("entry_date"), errors="coerce")
        exit_date = pd.to_datetime(trade.get("exit_date"), errors="coerce")
        if pd.notna(exit_date) and exit_date in result_df.index:
            result_df.loc[exit_date, "trade"] = "sell"
        if pd.notna(entry_date) and entry_date in result_df.index:
            result_df.loc[entry_date, "trade"] = "buy"
    return result_df


def render_backtest_outputs(
    result_df,
    bt_result,
    trades_df,
    symbol_code,
    pred_start,
    pred_end,
    chart_key,
):
    st.subheader("回测结果")
    metrics = [
        ("累计收益率", bt_result.get("累计收益率", 0)),
        ("超额收益率", bt_result.get("超额收益率", 0)),
        ("胜率", bt_result.get("胜率", 0)),
        ("交易笔数", bt_result.get("交易笔数", 0)),
        ("最大回撤", bt_result.get("最大回撤", 0)),
        ("夏普比率", bt_result.get("年化夏普比率", 0)),
    ]
    cols_1 = st.columns(3)
    for col, (name, value) in zip(cols_1, metrics[:3]):
        col.metric(name, format_metric_value(name, value))
    cols_2 = st.columns(3)
    for col, (name, value) in zip(cols_2, metrics[3:]):
        col.metric(name, format_metric_value(name, value))

    model_origin = []
    peak_window = bt_result.get("Peak模型训练窗口")
    trough_window = bt_result.get("Trough模型训练窗口")
    if peak_window is not None:
        peak_seed = bt_result.get("Peak模型seed")
        peak_text = f"Peak窗口: {peak_window}"
        if peak_seed is not None:
            peak_text += f" / seed={peak_seed}"
        model_origin.append(peak_text)
    if trough_window is not None:
        trough_seed = bt_result.get("Trough模型seed")
        trough_text = f"Trough窗口: {trough_window}"
        if trough_seed is not None:
            trough_text += f" / seed={trough_seed}"
        model_origin.append(trough_text)
    if model_origin:
        st.caption("模型来源：" + "；".join(model_origin))

    quality_metrics = [
        ("组合评分", bt_result.get("组合评分")),
        ("信号质量评分", bt_result.get("信号质量评分")),
        ("低点附近命中率", bt_result.get("低点附近命中率")),
        ("高点附近命中率", bt_result.get("高点附近命中率")),
        ("中段误报率", bt_result.get("中段误报率")),
    ]
    if any(value is not None for _, value in quality_metrics):
        quality_cols = st.columns(len(quality_metrics))
        for col, (name, value) in zip(quality_cols, quality_metrics):
            col.metric(name, format_metric_value(name, value))

    peaks_pred = result_df[result_df["Peak_Prediction"] == 1]
    troughs_pred = result_df[result_df["Trough_Prediction"] == 1]
    fig = plot_candlestick(
        result_df.copy(),
        symbol_code,
        pred_start.strftime("%Y%m%d"),
        pred_end.strftime("%Y%m%d"),
        peaks_pred,
        troughs_pred,
        prediction=True,
        bt_result=bt_result,
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    col_left, col_right = st.columns(2)
    display_result = result_df.rename(columns={
        "TradeDate": "交易日期",
        "Peak_Prediction": "高点标注",
        "Peak_Probability": "高点概率",
        "Trough_Prediction": "低点标注",
        "Trough_Probability": "低点概率",
    })
    with col_left:
        st.subheader("预测明细")
        st.dataframe(display_result[["交易日期", "高点标注", "高点概率", "低点标注", "低点概率"]])

    display_trades = trades_df.rename(columns={
        "entry_date": "买入日",
        "signal_type_buy": "买入原因",
        "entry_price": "买入价",
        "exit_date": "卖出日",
        "signal_type_sell": "卖出原因",
        "exit_price": "卖出价",
        "hold_days": "持仓日",
        "return": "盈亏",
    })
    if not display_trades.empty:
        display_trades["盈亏"] = display_trades["盈亏"] * 100
        display_trades["买入日"] = pd.to_datetime(display_trades["买入日"]).dt.strftime("%Y-%m-%d")
        display_trades["卖出日"] = pd.to_datetime(display_trades["卖出日"]).dt.strftime("%Y-%m-%d")

    with col_right:
        st.subheader("交易记录")
        if not display_trades.empty:
            st.dataframe(display_trades[[
                "买入日", "买入原因", "买入价",
                "卖出日", "卖出原因", "卖出价",
                "持仓日", "盈亏",
            ]].style.format({"盈亏": "{:.2f}%"}))
        else:
            st.write("暂无交易记录")


def format_metric_value(name, value):
    if value is None:
        return "暂无"
    if isinstance(value, (int, np.integer)) and "笔数" in name:
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if name in {"夏普比率", "组合评分", "信号质量评分"}:
            return f"{float(value):.4f}"
        return f"{float(value) * 100:.2f}%"
    return f"{value}"


