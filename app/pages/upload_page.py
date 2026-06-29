from app.ui_helpers import *


def render(data_source, symbol_code, N, mixture_depth):
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
        render_model_download_options(symbol_code, key_prefix="tab4_model_download")
        col_date1_up, col_date2_up = st.columns(2)
        with col_date1_up:
            pred_start_up = st.date_input("预测开始日期", datetime(2021, 1, 1), key="pred_start_tab4")
        with col_date2_up:
            pred_end_up = st.date_input("预测结束日期", TARGET_PRED_END, key="pred_end_tab4")

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
                # 如果模型文件里保存了N、mixture_depth，则优先使用
                N_val = best_models.get('N', N)
                mixture_val = best_models.get('mixture_depth', mixture_depth)
                pred_start_up_str = pred_start_up.strftime("%Y%m%d")
                pred_end_up_str = pred_end_up.strftime("%Y%m%d")
                raw_data_up = read_front_market_data(
                    symbol_code,
                    symbol_type,
                    end_date=pred_end_up_str
                )

                if best_models.get("model_type") == "event_regime_hgbr_combo":
                    base_result_up, base_bt_up, _ = predict_event_regime_model_data(
                        raw_data_up,
                        best_models,
                        eval_mode=False,
                        N_buy=1,
                        N_sell=1,
                        N_newhigh=60,
                        enable_chase=False,
                        enable_stop_loss=False,
                        enable_change_signal=False,
                        backtest_start_date=pred_start_up_str,
                        backtest_end_date=pred_end_up_str,
                    )
                else:
                    base_result_up, base_bt_up, _ = predict_new_data(
                        raw_data_up,
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
                        N_buy=1,
                        N_sell=1,
                        N_newhigh=60,
                        enable_chase=False,
                        enable_stop_loss=False,
                        enable_change_signal=False,
                        backtest_start_date=pred_start_up_str,
                        backtest_end_date=pred_end_up_str,
                    )
                st.session_state.upload_base_prediction_result = base_result_up.copy()
                st.session_state.upload_base_selection_bt = base_bt_up
                st.session_state.upload_prediction_cache_key = {
                    'data_source': data_source,
                    'symbol_code': symbol_code,
                    'pred_start': pred_start_up_str,
                    'pred_end': pred_end_up_str,
                }
                st.success("预测完成！（使用已上传模型，未叠加策略）")
            except Exception as e:
                st.error(f"预测失败: {str(e)}")

        upload_cache_key = {
            'data_source': data_source,
            'symbol_code': symbol_code,
            'pred_start': pred_start_up.strftime("%Y%m%d"),
            'pred_end': pred_end_up.strftime("%Y%m%d"),
        }
        if (
            st.session_state.get('upload_base_prediction_result') is not None
            and st.session_state.get('upload_prediction_cache_key') == upload_cache_key
        ):
            try:
                final_result_up, final_bt_up, final_trades_df_up = apply_strategy_to_prediction(
                    st.session_state.upload_base_prediction_result,
                    n_buy_up,
                    n_sell_up,
                    n_newhigh_up,
                    enable_chase_up,
                    enable_stop_loss_up,
                    enable_change_signal_up,
                )
                render_backtest_outputs(
                    final_result_up,
                    final_bt_up,
                    final_trades_df_up,
                    symbol_code,
                    pred_start_up,
                    pred_end_up,
                    chart_key="chart_upload_tab_strategy",
                )
            except Exception as e:
                st.error(f"上传模型策略回测刷新失败: {str(e)}")
        elif st.session_state.get('upload_base_prediction_result') is not None:
            st.info("上传模型预测参数已变化，请点击“开始预测(上传模型Tab)”生成新的模型预测缓存。")
