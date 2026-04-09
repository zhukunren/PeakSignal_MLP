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
from models import set_seed
from preprocess import preprocess_data, create_pos_neg_sequences_by_consecutive_labels
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick
from models import time_aware_oversampling

# 设置随机种子
set_seed(42)

# 修改页面配置
st.set_page_config(
    page_title="东吴秀享AI超额收益系统",
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- 初始化 session_state -------------------- #
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = None

if 'peak_models_list' not in st.session_state:
    st.session_state.peak_models_list = []
if 'trough_models_list' not in st.session_state:
    st.session_state.trough_models_list = []

if 'train_df_preprocessed' not in st.session_state:
    st.session_state.train_df_preprocessed = None
if 'train_all_features' not in st.session_state:
    st.session_state.train_all_features = None

# 预测 / 回测 结果（未模型微调）
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'final_bt' not in st.session_state:
    st.session_state.final_bt = {}

# ★ 新增：模型微调后的预测 / 回测结果，用于对比
if 'inc_final_result' not in st.session_state:
    st.session_state.inc_final_result = None
if 'inc_final_bt' not in st.session_state:
    st.session_state.inc_final_bt = {}

# ★ 新增：存储预测集原始 DataFrame（模型微调后需要再次预测）
if 'new_df_raw' not in st.session_state:
    st.session_state.new_df_raw = None

# ★ 新增：微调参数记录
if 'finetune_params' not in st.session_state:
    st.session_state.finetune_params = {}

# ★ 新增：用于存储“微调后的多个峰模型”和“微调后的多个谷模型”
if 'peak_models_finetuned_list' not in st.session_state:
    st.session_state.peak_models_finetuned_list = []
if 'trough_models_finetuned_list' not in st.session_state:
    st.session_state.trough_models_finetuned_list = []

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


# ========== 新增：模型保存功能 ========== #
def add_model_save_functionality(symbol_code):
    """添加微调后模型保存功能。"""
    st.subheader("保存微调后的模型")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        model_name = st.text_input("模型文件名", f"finetune_model_{symbol_code}_{datetime.now().strftime('%Y%m%d')}")
    
    with col2:
        if st.button("保存模型"):
            try:
                # 组织要保存的模型字典
                models_to_save = {
                    'peak_model': st.session_state.models['peak_model'],
                    'peak_scaler': st.session_state.models['peak_scaler'],
                    'peak_selector': st.session_state.models['peak_selector'],
                    'peak_selected_features': st.session_state.models['peak_selected_features'],
                    'peak_threshold': st.session_state.models['peak_threshold'],
                    'trough_model': st.session_state.models['trough_model'],
                    'trough_scaler': st.session_state.models['trough_scaler'],
                    'trough_selector': st.session_state.models['trough_selector'],
                    'trough_selected_features': st.session_state.models['trough_selected_features'],
                    'trough_threshold': st.session_state.models['trough_threshold'],
                    'N': st.session_state.models['N'],
                    'mixture_depth': st.session_state.models['mixture_depth'],
                    'finetune_params': st.session_state.finetune_params,
                    'finetune_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                model_bytes = pickle.dumps(models_to_save)
                
                st.download_button(
                    label="点击下载模型文件",
                    data=model_bytes,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"
                )
                st.success(f"模型已打包完成！点击上方按钮可下载 `{model_name}.pkl` 文件")
            except Exception as e:
                st.error(f"保存模型失败: {str(e)}")


def main_product():
    inject_orientation_script()
    st.title("东吴秀享AI超额收益系统")

    # ========== 侧边栏参数设置 ========== 
    with st.sidebar:
        st.header("参数设置")
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)
        with st.expander("模型设置", expanded=True):
            classifier_name_display = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            classifier_name = "MLP" if classifier_name_display == "深度学习" else "Transformer"
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_display = st.selectbox(
                "类别不均衡处理",
                ["过采样", "类别权重", "ADASYN", "Borderline-SMOTE", "SMOTEENN", "SMOTETomek", "时间感知过采样"]
            )
            if oversample_display == "过采样":
                oversample_method = "SMOTE"
            elif oversample_display == "类别权重":
                oversample_method = "Class Weights"
            elif oversample_display == "时间感知过采样":
                oversample_method = "Time-Aware"
            else:
                oversample_method = oversample_display
            use_best_combo = True
        with st.expander("特征设置", expanded=True):
            auto_feature = st.checkbox("自动特征选择", True)
            n_features_selected = st.number_input(
                "选择特征数量",
                min_value=5, max_value=100, value=20,
                disabled=auto_feature
            )

    load_custom_css()

    # ========== 四个选项卡 ========== 
    tab1, tab2, tab3, tab4 = st.tabs(["训练模型", "预测", "模型微调", "上传模型预测"])

    # =======================================
    #    Tab1: 训练模型
    # =======================================
    with tab1:
        st.subheader("训练参数")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("训练开始日期", datetime(2000, 1, 1))
        with col2:
            train_end = st.date_input("训练结束日期", datetime(2020, 12, 31))

        num_rounds = 10  # 固定多轮训练次数
        if st.button("开始训练"):
            begin_time = time.time()
            try:
                with st.spinner("数据预处理中..."):
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data, all_features_train = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=True
                    )
                    raw_data.to_csv("完整数据.csv", index=False, encoding="utf-8")
                    df_preprocessed_train = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
                
                with st.spinner(f"开始多轮训练，共 {num_rounds} 次..."):
                    st.session_state.peak_models_list.clear()
                    st.session_state.trough_models_list.clear()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(num_rounds):
                        progress_val = (i + 1) / num_rounds
                        status_text.text(f"正在训练第 {i+1}/{num_rounds} 组模型...")
                        progress_bar.progress(progress_val)

                        (peak_model, peak_scaler, peak_selector, peak_selected_features,
                         all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                         trough_model, trough_scaler, trough_selector, trough_selected_features,
                         all_features_trough, trough_best_score, trough_metrics, trough_threshold
                        ) = train_model(
                            df_preprocessed_train,
                            N,
                            all_features_train,
                            classifier_name,
                            mixture_depth,
                            n_features_selected if not auto_feature else 'auto',
                            oversample_method
                        )
                        st.session_state.peak_models_list.append(
                            (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
                        )
                        st.session_state.trough_models_list.append(
                            (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
                        )

                    progress_bar.progress(1.0)
                    status_text.text("多轮训练完成！")

                # 记录最后一次训练的模型到 session_state
                st.session_state.models = {
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
                    'mixture_depth': mixture_depth
                }
                st.session_state.train_df_preprocessed = df_preprocessed_train
                st.session_state.train_all_features = all_features_train
                st.session_state.trained = True

                st.success(f"多轮训练完成，共训练 {num_rounds} 组峰/谷模型。")

                # 训练可视化
                peaks = df_preprocessed_train[df_preprocessed_train['Peak'] == 1]
                troughs = df_preprocessed_train[df_preprocessed_train['Trough'] == 1]
                fig = plot_candlestick(
                    df_preprocessed_train,
                    symbol_code,
                    train_start.strftime("%Y%m%d"),
                    train_end.strftime("%Y%m%d"),
                    peaks=peaks,
                    troughs=troughs
                )
                st.plotly_chart(fig, use_container_width=True, key="chart1")
                # 训练完成后立即计算耗时
                end_time = time.time()
                elapsed_time = end_time - begin_time
                st.success(f'训练完成，总耗时：{elapsed_time:.2f}秒')  # 显示在训练区块内
            except Exception as e:
                st.error(f"训练失败: {str(e)}")

        # 训练集可视化（仅展示，不进行训练）
        try:
            st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
            symbol_type = 'index' if data_source == '指数' else 'stock'
            raw_data = read_day_from_tushare(symbol_code, symbol_type)
            
            raw_data, _ = preprocess_data(
                raw_data, N, mixture_depth, mark_labels=True
            )
            df_preprocessed_vis = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
            peaks_vis = df_preprocessed_vis[df_preprocessed_vis['Peak'] == 1]
            troughs_vis = df_preprocessed_vis[df_preprocessed_vis['Trough'] == 1]
            fig_vis = plot_candlestick(
                df_preprocessed_vis,
                symbol_code,
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                peaks=peaks_vis,
                troughs=troughs_vis
            )
            st.plotly_chart(fig_vis, use_container_width=True, key="chart2")
        except Exception as e:
            st.warning(f"可视化失败: {e}")


    # =======================================
    #   Tab2: 预测 + 回测
    # =======================================
    with tab2:
        if not st.session_state.get('trained', False):
            st.warning("请先完成模型训练")
        else:
            st.subheader("预测参数")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2:
                pred_end = st.date_input("预测结束日期", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab2"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab2")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab2"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal = st.checkbox("调整买卖信号", value=False, help="阳线买，阴线卖，高点需创X日新高", key="enable_change_signal_tab2")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh = st.number_input(
                        "",
                        min_value=0,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab2"
                    )

            if st.button("开始预测"):
                #记时
                
                try:
                    if st.session_state.train_df_preprocessed is None or st.session_state.train_all_features is None:
                        st.error("无法获取训练数据，请先在 [训练模型] 完成训练。")
                        return

                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data, _ = preprocess_data(raw_data, N, mixture_depth, mark_labels=False)
                    new_df_raw = select_time(raw_data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))

                    # 存到 session_state，供模型微调使用
                    st.session_state.new_df_raw = new_df_raw

                    # 策略参数
                    enable_chase_val = enable_chase
                    enable_stop_loss_val = enable_stop_loss
                    enable_change_signal_val = enable_change_signal
                    n_buy_val = n_buy
                    n_sell_val = n_sell
                    n_newhigh_val = n_newhigh

                    peak_models = st.session_state.peak_models_list
                    trough_models = st.session_state.trough_models_list

                    best_excess = -np.inf
                    best_models = None
                    final_result, final_bt, final_trades_df = None, {}, pd.DataFrame()

                    # 多组合搜索
                    if use_best_combo:
                        model_combinations = list(product(peak_models, trough_models))
                        total_combos = len(model_combinations)
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, (peak_m, trough_m) in enumerate(model_combinations):
                            combo_progress = (idx + 1) / total_combos
                            status_text.text(f"正在测试第 {idx+1}/{total_combos} 组模型...")
                            progress_bar.progress(combo_progress)

                            pm, ps, psel, pfeats, pth = peak_m
                            tm, ts, tsel, tfeats, tth = trough_m
                            try:
                                _, bt_result, _ = predict_new_data(
                                    new_df_raw,
                                    pm, ps, psel, pfeats, pth,
                                    tm, ts, tsel, tfeats, tth,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=10,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
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
                                        'trough_threshold': tth
                                    }
                            except:
                                continue

                        progress_bar.empty()
                        status_text.empty()

                        if best_models is None:
                            raise ValueError("所有组合均测试失败，无法完成预测。")

                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
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
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        
                        st.success(f"预测完成！(多组合) 最佳超额收益率: {best_excess * 100:.2f}%")
                    
                    else:
                        # 单模型预测
                        single_models = st.session_state.models
                        _, bt_result_temp, _ = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=True,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        best_excess = bt_result_temp.get('超额收益率', -np.inf)
                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！(单模型) 超额收益率: {best_excess*100:.2f}%")

                    # 显示回测结果
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',     '{:.4f}'.format(final_bt.get('年化夏普比率', 0)))
                    ]
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{float(value)*100:.2f}%" if "比率" not in name else f"{value}")
                        else:
                            col.metric(name, f"{value}")
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{float(value)*100:.2f}%" if "比率" not in name else f"{value}")
                        else:
                            col.metric(name, f"{value}")

                    # 蜡烛图
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart3")

                    # 预测明细 & 交易记录
                    col_left, col_right = st.columns(2)
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(final_trades_df[[
                                '买入日', '买入原因', '买入价',
                                '卖出日', '卖出原因', '卖出价',
                                '持仓日', '盈亏'
                            ]].style.format({'盈亏': '{:.2f}%'}))
                        else:
                            st.write("暂无交易记录")

                    # 保存到 session_state
                    st.session_state.final_result = final_result
                    st.session_state.final_bt = final_bt
                    st.session_state.pred_start = pred_start
                    st.session_state.pred_end = pred_end
                    st.session_state.n_buy_val = n_buy_val
                    st.session_state.n_sell_val = n_sell_val
                    st.session_state.n_newhigh_val = n_newhigh_val
                    st.session_state.enable_chase_val = enable_chase_val
                    st.session_state.enable_stop_loss_val = enable_stop_loss_val
                    st.session_state.enable_change_signal_val = enable_change_signal_val

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")


    # =======================================
    #   Tab3: 模型微调（核心改动）
    # =======================================
    with tab3:
        st.subheader("模型微调（微调已有模型）")
        if st.session_state.final_result is None or st.session_state.new_df_raw is None:
            st.warning("请先在 [预测] 页完成一次预测，才能进行模型微调。")
        else:
            inc_col1, inc_col2 = st.columns(2)
            with inc_col1:
                inc_start_date = st.date_input(
                    "模型微调起始日期",
                    st.session_state.get('pred_start', datetime(2021, 1, 1))
                )
            with inc_col2:
                inc_end_date = st.date_input(
                    "模型微调结束日期",
                    st.session_state.get('pred_end', datetime.now())
                )

            # 学习率
            lr_dict = {"极低 (1e-6)": 1e-6, "低 (1e-5)": 1e-5, "中 (1e-4)": 1e-4, "高 (1e-3)": 1e-3}
            lr_choice = st.selectbox("学习率", list(lr_dict.keys()), index=1)
            inc_lr = lr_dict[lr_choice]

            # 训练轮数
            inc_epochs = st.slider("最大训练轮数", 5, 100, 20)

            # 冻结层策略
            if classifier_name == "MLP":
                freeze_options = {
                    "不冻结任何层": "none",
                    "只冻结第一层 (fc1)": "first_layer",
                    "只冻结第二层 (fc2)": "second_layer",
                    "冻结所有层": "all",
                    "部分冻结第一层": "partial"
                }
            else:
                freeze_options = {
                    "不冻结任何层": "none",
                    "冻结输入层": "first_layer",
                    "冻结编码器层 (除最后一层)": "encoder_layers",
                    "冻结输出层": "output_layer",
                    "冻结所有层": "all"
                }
            freeze_choice = st.selectbox("冻结策略", list(freeze_options.keys()), index=0)
            freeze_option = freeze_options[freeze_choice]

            # 混合训练
            mix_enabled = st.checkbox("启用混合训练", value=True)
            inc_mix_ratio = 0.2
            if mix_enabled:
                inc_mix_ratio = st.slider("旧数据与新数据比例", 0.1, 2.0, 0.2, step=0.1)

            # 早停
            early_stopping = st.checkbox("启用早停", value=True)
            col_val1, col_val2 = st.columns(2)
            with col_val1:
                val_size = st.slider("验证集比例", 0.1, 0.5, 0.2, step=0.05, disabled=not early_stopping)
            with col_val2:
                patience = st.slider("早停耐心值", 1, 10, 3, step=1, disabled=not early_stopping)

            # 开始微调
            if st.button("执行模型微调"):
                try:
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_full = read_day_from_tushare(symbol_code, symbol_type)

                    # ① 获取全量数据 + 自动打标签
                    df_preprocessed_all, _ = preprocess_data(
                        raw_data_full,
                        N,
                        mixture_depth,
                        mark_labels=True
                    )

                    # ② 截取微调区间（这里也可以用同样区间做回测）
                    add_df = select_time(
                        df_preprocessed_all,
                        inc_start_date.strftime("%Y%m%d"),
                        inc_end_date.strftime("%Y%m%d")
                    )

                    # =============== ③ 核心改动：对峰/谷模型各进行10次“独立”微调 ============== #
                    st.session_state.peak_models_finetuned_list.clear()
                    st.session_state.trough_models_finetuned_list.clear()

                    # ---- 3.1 对峰模型进行 10 次微调 ----
                    st.write("正在对峰模型进行 10 轮微调训练...")
                    peak_progress_bar = st.progress(0)
                    for i in range(10):
                        round_text = st.empty()
                        round_text.text(f"峰模型 - 第 {i+1}/10 轮微调...")
                        # 每次都从“原模型”克隆一份，避免上一轮修改带来的影响
                        cloned_peak_model = copy.deepcopy(st.session_state.models['peak_model'])

                        updated_peak_model, peak_val_acc, peak_epochs = incremental_train_for_label(
                            model=cloned_peak_model,
                            scaler=st.session_state.models['peak_scaler'],
                            selected_features=st.session_state.models['peak_selected_features'],
                            df_new=add_df,
                            label_column='Peak',
                            classifier_name=classifier_name,
                            window_size=10,
                            oversample_method=oversample_method,
                            new_lr=inc_lr,
                            new_epochs=inc_epochs,
                            freeze_option=freeze_option,
                            old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                            mix_ratio=inc_mix_ratio,
                            progress_bar=None,  # 不用单次进度条了
                            early_stopping=early_stopping,
                            val_size=val_size,
                            patience=patience
                        )
                        # 将这次微调所得的“峰模型”存起来
                        st.session_state.peak_models_finetuned_list.append(
                            (updated_peak_model, peak_val_acc, peak_epochs)
                        )
                        peak_progress_bar.progress((i+1)/10)
                    
                    st.success("峰模型 10 轮微调全部完成！")

                    # ---- 3.2 对谷模型进行 10 次微调 ----
                    st.write("正在对谷模型进行 10 轮微调训练...")
                    trough_progress_bar = st.progress(0)
                    for i in range(10):
                        round_text = st.empty()
                        round_text.text(f"谷模型 - 第 {i+1}/10 轮微调...")
                        # 同理，克隆一份
                        cloned_trough_model = copy.deepcopy(st.session_state.models['trough_model'])

                        updated_trough_model, trough_val_acc, trough_epochs = incremental_train_for_label(
                            model=cloned_trough_model,
                            scaler=st.session_state.models['trough_scaler'],
                            selected_features=st.session_state.models['trough_selected_features'],
                            df_new=add_df,
                            label_column='Trough',
                            classifier_name=classifier_name,
                            window_size=10,
                            oversample_method=oversample_method,
                            new_lr=inc_lr,
                            new_epochs=inc_epochs,
                            freeze_option=freeze_option,
                            old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                            mix_ratio=inc_mix_ratio,
                            progress_bar=None,
                            early_stopping=early_stopping,
                            val_size=val_size,
                            patience=patience
                        )
                        # 将这次微调所得的“谷模型”存起来
                        st.session_state.trough_models_finetuned_list.append(
                            (updated_trough_model, trough_val_acc, trough_epochs)
                        )
                        trough_progress_bar.progress((i+1)/10)

                    st.success("谷模型 10 轮微调全部完成！")
                    
                    # 将一些微调参数存到 session_state
                    st.session_state.finetune_params = {
                        'lr': inc_lr,
                        'epochs': inc_epochs,
                        'freeze_option': freeze_option,
                        'mix_ratio': inc_mix_ratio if mix_enabled else 0,
                        'early_stopping': early_stopping,
                        'val_size': val_size,
                        'patience': patience
                    }

                    # ============ ④ 现在我们有 10 个微调峰模型 × 10 个微调谷模型 = 100 组合 ============
                    #     依次回测，找出“超额收益”最高的一组
                    st.write("正在对 10×10=100 种 微调后模型组合 进行回测，筛选最佳超额收益...")
                    best_excess_finetune = -np.inf
                    best_combo_finetune = None

                    # 以最新的 new_df_raw 区间做“验证回测”，也可以用 add_df 区间，根据需要自由调整
                    eval_df = st.session_state.new_df_raw
                    if eval_df is None or eval_df.empty:
                        eval_df = add_df  # 如果 new_df_raw 没数据，就用 add_df
                    total_combos = 100
                    combo_progress_bar = st.progress(0)
                    combo_text = st.empty()

                    for idx, (peak_tuple, trough_tuple) in enumerate(product(
                        st.session_state.peak_models_finetuned_list,
                        st.session_state.trough_models_finetuned_list
                    )):
                        i_progress = (idx+1)/total_combos
                        combo_text.text(f"第 {idx+1}/{total_combos} 组合...")
                        combo_progress_bar.progress(i_progress)

                        (fined_peak_model, peak_val_acc, peak_epochs) = peak_tuple
                        (fined_trough_model, trough_val_acc, trough_epochs) = trough_tuple

                        try:
                            # 注意：peak_scaler/selector/selected_features/threshold 还是沿用原先的
                            # 因为微调只更新模型参数，不更新 scaler/特征选择器/阈值
                            _, bt_result_temp, _ = predict_new_data(
                                eval_df,
                                fined_peak_model,
                                st.session_state.models['peak_scaler'],
                                st.session_state.models['peak_selector'],
                                st.session_state.models['peak_selected_features'],
                                st.session_state.models['peak_threshold'],
                                fined_trough_model,
                                st.session_state.models['trough_scaler'],
                                st.session_state.models['trough_selector'],
                                st.session_state.models['trough_selected_features'],
                                st.session_state.models['trough_threshold'],
                                st.session_state.models['N'],
                                st.session_state.models['mixture_depth'],
                                window_size=10,
                                eval_mode=True,  # 只做回测，不要存最终结果
                                N_buy=st.session_state.get('n_buy_val', 10),
                                N_sell=st.session_state.get('n_sell_val', 10),
                                N_newhigh=st.session_state.get('n_newhigh_val', 60),
                                enable_chase=st.session_state.get('enable_chase_val', False),
                                enable_stop_loss=st.session_state.get('enable_stop_loss_val', False),
                                enable_change_signal=st.session_state.get('enable_change_signal_val', False),
                            )
                            current_excess = bt_result_temp.get('超额收益率', -np.inf)
                            if current_excess > best_excess_finetune:
                                best_excess_finetune = current_excess
                                best_combo_finetune = (peak_tuple, trough_tuple)
                        except Exception as e:
                            # 某些组合可能因为数据极端/过采样导致报错，忽略
                            pass

                    combo_progress_bar.empty()
                    combo_text.empty()

                    if best_combo_finetune is None:
                        st.error("在 100 组合中，全部回测都失败，请检查数据或微调参数。")
                        return

                    (final_peak_model, final_peak_val_acc, _) = best_combo_finetune[0]
                    (final_trough_model, final_trough_val_acc, _) = best_combo_finetune[1]
                    st.success(f"微调后最佳组合已找到！ 超额收益率 = {best_excess_finetune*100:.2f}%")
                    
                    # ============ ⑤ 用这套最佳微调模型做最终预测 + 回测，生成前后对比 ============
                    # 更新 session_state.models 中的 “peak_model”/“trough_model”
                    st.session_state.models['peak_model'] = final_peak_model
                    st.session_state.models['trough_model'] = final_trough_model

                    # 用之前的预测区间 `[pred_start, pred_end]` 来回测对比
                    refreshed_new_df = st.session_state.new_df_raw
                    if refreshed_new_df is None or refreshed_new_df.empty:
                        st.warning("未发现可用的预测集数据，将使用微调数据区间进行回测展示。")
                        refreshed_new_df = add_df

                    inc_final_result, inc_final_bt, inc_final_trades_df = predict_new_data(
                        refreshed_new_df,
                        st.session_state.models['peak_model'],
                        st.session_state.models['peak_scaler'],
                        st.session_state.models['peak_selector'],
                        st.session_state.models['peak_selected_features'],
                        st.session_state.models['peak_threshold'],
                        st.session_state.models['trough_model'],
                        st.session_state.models['trough_scaler'],
                        st.session_state.models['trough_selector'],
                        st.session_state.models['trough_selected_features'],
                        st.session_state.models['trough_threshold'],
                        st.session_state.models['N'],
                        st.session_state.models['mixture_depth'],
                        window_size=10,
                        eval_mode=False,
                        N_buy=st.session_state.get('n_buy_val', 10),
                        N_sell=st.session_state.get('n_sell_val', 10),
                        N_newhigh=st.session_state.get('n_newhigh_val', 60),
                        enable_chase=st.session_state.get('enable_chase_val', False),
                        enable_stop_loss=st.session_state.get('enable_stop_loss_val', False),
                        enable_change_signal=st.session_state.get('enable_change_signal_val', False),
                    )

                    # 保存微调后的结果
                    st.session_state.inc_final_result = inc_final_result
                    st.session_state.inc_final_bt = inc_final_bt

                    # ---- 对比：微调前后的回测 ----
                    st.markdown("### 对比：未模型微调 vs 模型微调后")
                    orig_bt = st.session_state.final_bt
                    if orig_bt is None:
                        st.warning("无法找到微调前的回测结果，可能尚未做过预测。请先在 [预测] 标签页完成一次预测。")
                    else:
                        inc_bt = st.session_state.inc_final_bt
                        col_before, col_after, col_diff = st.columns(3)
                        with col_before:
                            st.write("**微调前**")
                            st.metric("累计收益率", f"{orig_bt.get('累计收益率', 0)*100:.2f}%")
                            st.metric("超额收益率", f"{orig_bt.get('超额收益率', 0)*100:.2f}%")
                            st.metric("胜率", f"{orig_bt.get('胜率', 0)*100:.2f}%")
                            st.metric("最大回撤", f"{orig_bt.get('最大回撤', 0)*100:.2f}%")
                            st.metric("交易笔数", f"{orig_bt.get('交易笔数', 0)}")

                        with col_after:
                            st.write("**微调后**")
                            st.metric("累计收益率", f"{inc_bt.get('累计收益率', 0)*100:.2f}%")
                            st.metric("超额收益率", f"{inc_bt.get('超额收益率', 0)*100:.2f}%")
                            st.metric("胜率", f"{inc_bt.get('胜率', 0)*100:.2f}%")
                            st.metric("最大回撤", f"{inc_bt.get('最大回撤', 0)*100:.2f}%")
                            st.metric("交易笔数", f"{inc_bt.get('交易笔数', 0)}")

                        with col_diff:
                            st.write("**变化量**")
                            st.metric("累计收益率变化",
                                      f"{(inc_bt.get('累计收益率', 0) - orig_bt.get('累计收益率', 0))*100:.2f}%",
                                      delta_color="normal")
                            st.metric("超额收益率变化",
                                      f"{(inc_bt.get('超额收益率', 0) - orig_bt.get('超额收益率', 0))*100:.2f}%",
                                      delta_color="normal")
                            st.metric("胜率变化",
                                      f"{(inc_bt.get('胜率', 0) - orig_bt.get('胜率', 0))*100:.2f}%",
                                      delta_color="normal")
                            st.metric("最大回撤变化",
                                      f"{(inc_bt.get('最大回撤', 0) - orig_bt.get('最大回撤', 0))*100:.2f}%",
                                      delta_color="inverse")
                            st.metric("交易笔数变化",
                                      f"{inc_bt.get('交易笔数', 0) - orig_bt.get('交易笔数', 0)}",
                                      delta_color="normal")

                    # ---- 微调前后图表对比 ----
                    st.subheader("微调前后图表对比")
                    col_before_chart, col_after_chart = st.columns(2)

                    with col_before_chart:
                        st.markdown("**微调前预测**")
                        if st.session_state.final_result is not None:
                            orig_result = st.session_state.final_result.copy()
                            peaks_pred_orig = orig_result[orig_result['高点标注'] == 1]
                            troughs_pred_orig = orig_result[orig_result['低点标注'] == 1]
                            
                            fig_before = plot_candlestick(
                                orig_result,
                                symbol_code,
                                st.session_state.pred_start.strftime("%Y%m%d"),
                                st.session_state.pred_end.strftime("%Y%m%d"),
                                peaks_pred_orig,
                                troughs_pred_orig,
                                prediction=True
                            )
                            fig_before.update_layout(height=400)
                            st.plotly_chart(fig_before, use_container_width=True, key="chart_before")
                            st.markdown(f"高点预测: **{len(peaks_pred_orig)}** 个")
                            st.markdown(f"低点预测: **{len(troughs_pred_orig)}** 个")
                            
                            if 'final_bt' in st.session_state and st.session_state.final_bt:
                                st.markdown(f"交易次数: **{st.session_state.final_bt.get('交易笔数', 0)}** 笔")
                                st.markdown(f"交易胜率: **{st.session_state.final_bt.get('胜率', 0)*100:.2f}%**")
                        else:
                            st.warning("暂无微调前预测数据")

                    with col_after_chart:
                        st.markdown("**微调后预测**")
                        if inc_final_result is not None and not inc_final_result.empty:
                            peaks_pred_inc = inc_final_result[inc_final_result['Peak_Prediction'] == 1]
                            troughs_pred_inc = inc_final_result[inc_final_result['Trough_Prediction'] == 1]
                            fig_after = plot_candlestick(
                                inc_final_result,
                                symbol_code,
                                st.session_state.pred_start.strftime("%Y%m%d"),
                                st.session_state.pred_end.strftime("%Y%m%d"),
                                peaks_pred_inc,
                                troughs_pred_inc,
                                prediction=True
                            )
                            fig_after.update_layout(height=400)
                            st.plotly_chart(fig_after, use_container_width=True, key="chart_after")
                            st.markdown(f"高点预测: **{len(peaks_pred_inc)}** 个")
                            st.markdown(f"低点预测: **{len(troughs_pred_inc)}** 个")
                            if st.session_state.inc_final_bt:
                                st.markdown(f"交易次数: **{st.session_state.inc_final_bt.get('交易笔数', 0)}** 笔")
                                st.markdown(f"交易胜率: **{st.session_state.inc_final_bt.get('胜率', 0)*100:.2f}%**")
                        else:
                            st.warning("暂无微调后预测数据")

                    # ---- 评估微调效果 ----
                    evaluate_finetune_effect(freeze_option)

                    # ---- 添加模型保存功能 ----
                    add_model_save_functionality(symbol_code)

                except Exception as e:
                    st.error(f"模型微调过程出现错误: {str(e)}")
                    st.exception(e)


    # =======================================
    #   Tab4: 上传模型文件，独立预测
    # =======================================
    with tab4:
        st.subheader("上传模型文件（.pkl）并预测")
        st.markdown("在此页面可以上传之前已保存的最佳模型或单模型文件，直接进行预测。")
        uploaded_file = st.file_uploader("选择本地模型文件：", type=["pkl"])
        if uploaded_file is not None:
            with st.spinner("正在加载模型..."):
                best_models_loaded = pickle.load(uploaded_file)
                st.session_state.best_models = best_models_loaded
                st.session_state.trained = True
            st.success("模型文件已加载，可进行预测！")

        if not st.session_state.trained or (st.session_state.best_models is None):
            st.warning("请先上传模型文件，或前往 [训练模型] 页面进行训练并保存。")
        else:
            st.markdown("### 预测参数（使用上传模型）")
            col_date1_up, col_date2_up = st.columns(2)
            with col_date1_up:
                pred_start_up = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2_up:
                pred_end_up = st.date_input("预测结束日期", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab4")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab4"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab4")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab4"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="阳线买，阴线卖，高点需创X日新高", key="enable_change_signal_tab4")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab4"
                    )

            if st.button("开始预测(上传模型Tab)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_up = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data_up, _ = preprocess_data(
                        raw_data_up, N, mixture_depth, mark_labels=False
                    )
                    new_df_up = select_time(raw_data_up, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

                    # 如果模型文件里保存了N、mixture_depth，则优先使用
                    N_val = best_models.get('N', N)
                    mixture_val = best_models.get('mixture_depth', mixture_depth)

                    final_result_up, final_bt_up, final_trades_df_up = predict_new_data(
                        new_df_up,
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
                        N_val,
                        mixture_val,
                        window_size=10,
                        eval_mode=False,
                        N_buy=n_buy_up,
                        N_sell=n_sell_up,
                        N_newhigh=n_newhigh_up,
                        enable_chase=enable_chase_up,
                        enable_stop_loss=enable_stop_loss_up,
                        enable_change_signal=enable_change_signal_up,
                    )
                    st.success("预测完成！（使用已上传模型）")

                    st.subheader("回测结果")
                    metrics_up = [
                        ('累计收益率',   final_bt_up.get('累计收益率', 0)),
                        ('超额收益率',   final_bt_up.get('超额收益率', 0)),
                        ('胜率',         final_bt_up.get('胜率', 0)),
                        ('交易笔数',     final_bt_up.get('交易笔数', 0)),
                        ('最大回撤',     final_bt_up.get('最大回撤', 0)),
                        ('夏普比率',     final_bt_up.get('年化夏普比率', 0)),
                    ]
                    first_line_up = metrics_up[:3]
                    cols_1_up = st.columns(3)
                    for col, (name, value) in zip(cols_1_up, first_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line_up = metrics_up[3:]
                    cols_2_up = st.columns(3)
                    for col, (name, value) in zip(cols_2_up, second_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # 蜡烛图
                    peaks_pred_up = final_result_up[final_result_up['Peak_Prediction'] == 1]
                    troughs_pred_up = final_result_up[final_result_up['Trough_Prediction'] == 1]
                    fig_up = plot_candlestick(
                        final_result_up,
                        symbol_code,
                        pred_start_up.strftime("%Y%m%d"),
                        pred_end_up.strftime("%Y%m%d"),
                        peaks_pred_up,
                        troughs_pred_up,
                        prediction=True
                    )
                    st.plotly_chart(fig_up, use_container_width=True, key="chart_upload_tab")

                    # 预测明细
                    col_left_up, col_right_up = st.columns(2)
                    final_result_up = final_result_up.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left_up:
                        st.subheader("预测明细")
                        st.dataframe(final_result_up[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    # 交易记录
                    final_trades_df_up = final_trades_df_up.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df_up.empty:
                        final_trades_df_up['盈亏'] = final_trades_df_up['盈亏'] * 100
                        final_trades_df_up['买入日'] = final_trades_df_up['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df_up['卖出日'] = final_trades_df_up['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right_up:
                        st.subheader("交易记录")
                        if not final_trades_df_up.empty:
                            st.dataframe(
                                final_trades_df_up[[
                                    '买入日', '买入原因', '买入价',
                                    '卖出日', '卖出原因', '卖出价',
                                    '持仓日', '盈亏'
                                ]].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")


if __name__ == "__main__":
    main_product()
