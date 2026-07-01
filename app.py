import streamlit as st

from ml_trader.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

from app.pages import finetune_page, prediction_page, training_page, upload_page
from app.ui_helpers import (
    install_closed_websocket_log_filter,
    inject_orientation_script,
    load_custom_css,
)
from app.utils.session_state import SessionStateManager


install_closed_websocket_log_filter()


st.set_page_config(
    page_title="东吴秀享AI超额收益系统",
    layout="wide",
    initial_sidebar_state="auto",
)


def render_sidebar():
    with st.sidebar:
        st.header("参数设置")
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=20)
        with st.expander("模型设置", expanded=True):
            classifier_name_display = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            classifier_name = "MLP" if classifier_name_display == "深度学习" else "Transformer"
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_display = st.selectbox(
                "类别不均衡处理",
                ["过采样", "类别权重", "ADASYN", "Borderline-SMOTE", "SMOTEENN", "SMOTETomek", "时间感知过采样"],
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
                min_value=5,
                max_value=100,
                value=20,
                disabled=auto_feature,
            )

    return {
        "data_source": data_source,
        "symbol_code": symbol_code,
        "N": N,
        "classifier_name": classifier_name,
        "mixture_depth": mixture_depth,
        "oversample_method": oversample_method,
        "use_best_combo": use_best_combo,
        "auto_feature": auto_feature,
        "n_features_selected": n_features_selected,
    }


def main_product():
    logger.info("Starting Streamlit application")
    SessionStateManager.initialize()
    inject_orientation_script()
    st.title("东吴秀享AI超额收益系统")

    params = render_sidebar()
    load_custom_css()

    tab1, tab2, tab3, tab4 = st.tabs(["训练模型", "预测", "模型微调", "上传模型预测"])

    with tab1:
        training_page.render(
            params["data_source"],
            params["symbol_code"],
            params["N"],
            params["classifier_name"],
            params["mixture_depth"],
            params["oversample_method"],
            params["auto_feature"],
            params["n_features_selected"],
        )

    with tab2:
        prediction_page.render(
            params["data_source"],
            params["symbol_code"],
            params["use_best_combo"],
        )

    with tab3:
        finetune_page.render(
            params["data_source"],
            params["symbol_code"],
            params["classifier_name"],
            params["N"],
            params["mixture_depth"],
            params["oversample_method"],
        )

    with tab4:
        upload_page.render(
            params["data_source"],
            params["symbol_code"],
            params["N"],
            params["mixture_depth"],
        )


if __name__ == "__main__":
    main_product()
