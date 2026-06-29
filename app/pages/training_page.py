from app.ui_helpers import *
from app.services.training_service import TrainingService


def render(data_source, symbol_code, N, classifier_name, mixture_depth, oversample_method, auto_feature, n_features_selected):
    st.subheader("训练参数")
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input("训练开始日期", datetime(2000, 1, 1), key="train_start_tab1")
    with col2:
        train_end = st.date_input("训练结束日期", datetime(2020, 12, 31), key="train_end_tab1")

    num_rounds = 10  # 固定多轮训练次数，默认包含 seed=7308 的第8轮目标模型
    if st.button("开始训练"):
        begin_time = time.time()
        try:
            with st.spinner("数据预处理中..."):
                symbol_type = 'index' if data_source == '指数' else 'stock'
                raw_data = read_front_market_data(
                    symbol_code,
                    symbol_type,
                    TARGET_PRED_END.strftime("%Y%m%d")
                )
                raw_data, all_features_train = preprocess_data(
                    raw_data, N, mixture_depth, mark_labels=True
                )
                raw_data.to_csv("完整数据.csv", index=False, encoding="utf-8")
                df_preprocessed_train = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
            
            with st.spinner(f"开始多轮训练，共 {num_rounds} 次..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total, message):
                    status_text.text(message)
                    progress_bar.progress(current / total)

                training_service = TrainingService()
                last_round_models, peak_models_list, trough_models_list = training_service.train_multiple_rounds(
                    df_preprocessed_train,
                    N,
                    all_features_train,
                    classifier_name,
                    mixture_depth,
                    n_features_selected if not auto_feature else 'auto',
                    oversample_method,
                    num_rounds=num_rounds,
                    progress_callback=progress_callback,
                )

                progress_bar.progress(1.0)
                status_text.text("多轮训练完成！")

                # 记录最后一次训练的模型到 session_state
                st.session_state.models = last_round_models
                st.session_state.peak_models_list = peak_models_list
                st.session_state.trough_models_list = trough_models_list
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

    if st.session_state.get('trained') and is_downloadable_model_dict(st.session_state.get('models')):
        render_model_download_options(symbol_code, key_prefix="tab1_model_download")

    # 训练集可视化（仅展示，不进行训练）
    try:
        st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
        symbol_type = 'index' if data_source == '指数' else 'stock'
        raw_data = read_front_market_data(
            symbol_code,
            symbol_type,
            TARGET_PRED_END.strftime("%Y%m%d")
        )
        
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
