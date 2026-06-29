"""
指标展示组件

用于展示回测结果、训练指标等
"""
import streamlit as st
from typing import Dict


def display_backtest_metrics(bt_result: Dict, title: str = "回测结果"):
    """
    展示回测指标

    Args:
        bt_result: 回测结果字典
        title: 标题
    """
    st.subheader(title)

    # 使用列布局展示关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cumulative_return = bt_result.get('累计收益率', 0)
        st.metric(
            "累计收益率",
            f"{cumulative_return:.2%}",
            delta=None
        )

    with col2:
        excess_return = bt_result.get('超额收益率', 0)
        st.metric(
            "超额收益率",
            f"{excess_return:.2%}",
            delta=None
        )

    with col3:
        win_rate = bt_result.get('胜率', 0)
        if win_rate is not None:
            st.metric("胜率", f"{win_rate:.2%}")
        else:
            st.metric("胜率", "N/A")

    with col4:
        trade_count = bt_result.get('交易笔数', 0)
        st.metric("交易笔数", f"{trade_count}")

    # 第二行指标
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        max_drawdown = bt_result.get('最大回撤', 0)
        st.metric(
            "最大回撤",
            f"{max_drawdown:.2%}",
            delta=None,
            delta_color="inverse"
        )

    with col6:
        sharpe = bt_result.get('年化夏普比率', 0)
        if sharpe:
            st.metric("夏普比率", f"{sharpe:.2f}")
        else:
            st.metric("夏普比率", "N/A")

    with col7:
        avg_return = bt_result.get('单笔平均收益率', 0)
        if avg_return:
            st.metric("单笔平均", f"{avg_return:.2%}")

    with col8:
        benchmark_return = bt_result.get('同期标的涨跌幅', 0)
        if benchmark_return is not None:
            st.metric("基准收益", f"{benchmark_return:.2%}")


def display_training_summary(models: Dict):
    """
    展示训练摘要信息

    Args:
        models: 模型字典
    """
    st.success("✅ 训练完成！")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**分类器**: {models.get('classifier_name', 'Unknown')}")

    with col2:
        st.info(f"**N窗口**: {models.get('N', 0)}")

    with col3:
        st.info(f"**混合深度**: {models.get('mixture_depth', 0)}")

    # 特征数量
    col4, col5, col6 = st.columns(3)

    with col4:
        peak_features = len(models.get('peak_selected_features', []))
        st.info(f"**峰模型特征**: {peak_features}")

    with col5:
        trough_features = len(models.get('trough_selected_features', []))
        st.info(f"**谷模型特征**: {trough_features}")

    with col6:
        st.info(f"**种子基准**: {models.get('seed_base', 0)}")


def display_prediction_summary(best_excess: float, combo_count: int):
    """
    展示预测摘要信息

    Args:
        best_excess: 最佳超额收益率
        combo_count: 测试的组合数量
    """
    st.success(f"✅ 预测完成！测试了 {combo_count} 种组合")

    st.metric(
        "最佳超额收益率",
        f"{best_excess:.2%}",
        delta=None
    )


def display_error(error_message: str, title: str = "错误"):
    """
    展示错误信息

    Args:
        error_message: 错误消息
        title: 标题
    """
    st.error(f"**{title}**: {error_message}")


def display_warning(message: str):
    """展示警告信息"""
    st.warning(message)


def display_info(message: str):
    """展示提示信息"""
    st.info(message)
