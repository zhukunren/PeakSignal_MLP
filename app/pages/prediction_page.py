from app.ui_helpers import *
from app.services.prediction_service import PredictionService
from ml_trader.logging_config import get_logger


logger = get_logger(__name__)


def render(data_source, symbol_code, use_best_combo):
    if not st.session_state.get('trained', False):
        st.warning("请先完成模型训练")
    else:
        st.subheader("预测参数")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1), key="pred_start_tab2")
        with col_date2:
            pred_end = st.date_input("预测结束日期", TARGET_PRED_END, key="pred_end_tab2")

        compare_candidate_pools = False
        if use_best_combo:
            compare_candidate_pools = st.checkbox(
                "比较旧/近年/混合候选池",
                value=False,
                help="开启后会额外搜索基础窗口、近年窗口和全量混合候选池，耗时更长。",
                key="compare_candidate_pools_tab2",
            )

        with st.expander("策略选择", expanded=False):
            load_custom_css()
            strategy_row1 = st.columns([2, 2, 5])
            with strategy_row1[0]:
                enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
            with strategy_row1[1]:
                st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
            with strategy_row1[2]:
                n_buy = st.number_input(
                    "追涨长度",
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
                    "止损长度",
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
                    "高点需创多少日新高",
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
                raw_data = read_front_market_data(
                    symbol_code,
                    symbol_type,
                    end_date=pred_end.strftime("%Y%m%d")
                )
                new_df_raw = raw_data.copy()
                new_df_for_display = select_time(
                    raw_data.copy(),
                    pred_start.strftime("%Y%m%d"),
                    pred_end.strftime("%Y%m%d")
                )

                # 存到 session_state，供模型微调使用
                st.session_state.new_df_raw = new_df_raw
                st.session_state.new_df_display = new_df_for_display

                # 策略参数
                enable_chase_val = enable_chase
                enable_stop_loss_val = enable_stop_loss
                enable_change_signal_val = enable_change_signal
                n_buy_val = n_buy
                n_sell_val = n_sell
                n_newhigh_val = n_newhigh

                peak_models = st.session_state.peak_models_list
                trough_models = st.session_state.trough_models_list

                def candidate_window(candidate):
                    return candidate.get("train_window") if isinstance(candidate, dict) else None

                def comparison_row(pool_name, models, excess, bt_result):
                    return {
                        "候选池": pool_name,
                        "组合评分": bt_result.get("组合评分"),
                        "超额收益率": excess,
                        "信号质量评分": bt_result.get("信号质量评分"),
                        "低点附近命中率": bt_result.get("低点附近命中率"),
                        "高点附近命中率": bt_result.get("高点附近命中率"),
                        "中段误报率": bt_result.get("中段误报率"),
                        "Peak窗口": bt_result.get("Peak模型训练窗口") or models.get("peak_train_window"),
                        "Trough窗口": bt_result.get("Trough模型训练窗口") or models.get("trough_train_window"),
                    }

                best_excess = -np.inf
                best_models = None
                base_result, base_bt = None, {}
                pool_comparison_rows = []

                # 多组合搜索
                if use_best_combo:
                    progress_reporter = StreamlitProgressReporter()

                    def progress_callback(current, total, message):
                        progress_reporter.update(current, total, message)

                    prediction_service = PredictionService()
                    best_models, best_excess, base_result, base_bt = prediction_service.search_best_combination(
                        peak_models,
                        trough_models,
                        new_df_raw,
                        st.session_state.models['N'],
                        st.session_state.models['mixture_depth'],
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        progress_callback=progress_callback,
                    )
                    progress_reporter.clear()
                    
                    selection_score = base_bt.get("组合评分")
                    if selection_score is not None:
                        st.success(
                            f"预测完成！(多组合，未叠加策略筛选) "
                            f"组合评分: {selection_score:.4f}，超额收益率: {best_excess * 100:.2f}%"
                        )
                    else:
                        st.success(f"预测完成！(多组合，未叠加策略筛选) 超额收益率: {best_excess * 100:.2f}%")

                    pool_comparison_rows.append(
                        comparison_row("混合候选池", best_models, best_excess, base_bt)
                    )
                    if compare_candidate_pools:
                        comparison_groups = [
                            (
                                "旧模型池",
                                [m for m in peak_models if candidate_window(m) == "base_2000_2020"],
                                [m for m in trough_models if candidate_window(m) == "base_2000_2020"],
                            ),
                            (
                                "近年模型池",
                                [
                                    m for m in peak_models
                                    if candidate_window(m) and candidate_window(m) != "base_2000_2020"
                                ],
                                [
                                    m for m in trough_models
                                    if candidate_window(m) and candidate_window(m) != "base_2000_2020"
                                ],
                            ),
                        ]
                        for pool_name, pool_peak_models, pool_trough_models in comparison_groups:
                            if not pool_peak_models or not pool_trough_models:
                                pool_comparison_rows.append({
                                    "候选池": pool_name,
                                    "状态": "候选不足",
                                    "Peak候选数": len(pool_peak_models),
                                    "Trough候选数": len(pool_trough_models),
                                })
                                continue
                            with st.spinner(f"候选池对照：{pool_name}"):
                                cmp_models, cmp_excess, _, cmp_bt = prediction_service.search_best_combination(
                                    pool_peak_models,
                                    pool_trough_models,
                                    new_df_raw,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    pred_start.strftime("%Y%m%d"),
                                    pred_end.strftime("%Y%m%d"),
                                )
                                pool_comparison_rows.append(
                                    comparison_row(pool_name, cmp_models, cmp_excess, cmp_bt)
                                )
                
                else:
                    # 单模型预测
                    single_models = st.session_state.models
                    best_models = {
                        'peak_model': single_models['peak_model'],
                        'peak_scaler': single_models['peak_scaler'],
                        'peak_selector': single_models['peak_selector'],
                        'peak_selected_features': single_models['peak_selected_features'],
                        'peak_threshold': single_models['peak_threshold'],
                        'trough_model': single_models['trough_model'],
                        'trough_scaler': single_models['trough_scaler'],
                        'trough_selector': single_models['trough_selector'],
                        'trough_selected_features': single_models['trough_selected_features'],
                        'trough_threshold': single_models['trough_threshold'],
                    }
                    base_result, base_bt, _ = predict_new_data(
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
                        N_buy=1,
                        N_sell=1,
                        N_newhigh=60,
                        enable_chase=False,
                        enable_stop_loss=False,
                        enable_change_signal=False,
                        backtest_start_date=pred_start.strftime("%Y%m%d"),
                        backtest_end_date=pred_end.strftime("%Y%m%d"),
                    )
                    best_excess = base_bt.get('超额收益率', -np.inf)
                    st.success(f"预测完成！(单模型，未叠加策略筛选) 超额收益率: {best_excess*100:.2f}%")

                cached_models = {
                    **best_models,
                    'N': st.session_state.models['N'],
                    'mixture_depth': st.session_state.models['mixture_depth'],
                }
                st.session_state.selected_prediction_models = cached_models
                st.session_state.best_models = cached_models
                st.session_state.models.update(cached_models)
                st.session_state.base_prediction_result = base_result.copy()
                st.session_state.base_selection_bt = base_bt
                st.session_state.candidate_pool_comparison = (
                    pd.DataFrame(pool_comparison_rows)
                    if compare_candidate_pools and pool_comparison_rows
                    else None
                )
                st.session_state.prediction_cache_key = {
                    'data_source': data_source,
                    'symbol_code': symbol_code,
                    'pred_start': pred_start.strftime("%Y%m%d"),
                    'pred_end': pred_end.strftime("%Y%m%d"),
                }
                st.session_state.pred_start = pred_start
                st.session_state.pred_end = pred_end
                st.session_state.n_buy_val = n_buy_val
                st.session_state.n_sell_val = n_sell_val
                st.session_state.n_newhigh_val = n_newhigh_val
                st.session_state.enable_chase_val = enable_chase_val
                st.session_state.enable_stop_loss_val = enable_stop_loss_val
                st.session_state.enable_change_signal_val = enable_change_signal_val

            except Exception as e:
                logger.exception("Prediction page failed")
                st.error(f"预测失败: {str(e)}")

        current_cache_key = {
            'data_source': data_source,
            'symbol_code': symbol_code,
            'pred_start': pred_start.strftime("%Y%m%d"),
            'pred_end': pred_end.strftime("%Y%m%d"),
        }
        if (
            st.session_state.get('base_prediction_result') is not None
            and st.session_state.get('prediction_cache_key') == current_cache_key
        ):
            try:
                final_result, final_bt, final_trades_df = apply_strategy_to_prediction(
                    st.session_state.base_prediction_result,
                    n_buy,
                    n_sell,
                    n_newhigh,
                    enable_chase,
                    enable_stop_loss,
                    enable_change_signal,
                )
                for metric_key in [
                    "组合筛选指标",
                    "组合评分",
                    "信号质量评分",
                    "低点附近命中率",
                    "高点附近命中率",
                    "中段误报率",
                    "Peak模型训练窗口",
                    "Peak模型训练起始",
                    "Peak模型训练结束",
                    "Peak模型seed",
                    "Peak模型轮次",
                    "Trough模型训练窗口",
                    "Trough模型训练起始",
                    "Trough模型训练结束",
                    "Trough模型seed",
                    "Trough模型轮次",
                ]:
                    if metric_key in st.session_state.get("base_selection_bt", {}):
                        final_bt[metric_key] = st.session_state.base_selection_bt[metric_key]
                st.session_state.final_result = final_result
                st.session_state.final_bt = final_bt
                st.session_state.final_trades_df = final_trades_df
                st.session_state.n_buy_val = n_buy
                st.session_state.n_sell_val = n_sell
                st.session_state.n_newhigh_val = n_newhigh
                st.session_state.enable_chase_val = enable_chase
                st.session_state.enable_stop_loss_val = enable_stop_loss
                st.session_state.enable_change_signal_val = enable_change_signal
                render_backtest_outputs(
                    final_result,
                    final_bt,
                    final_trades_df,
                    symbol_code,
                    pred_start,
                    pred_end,
                    chart_key="chart3_strategy",
                )
                candidate_pool_comparison = st.session_state.get("candidate_pool_comparison")
                if isinstance(candidate_pool_comparison, pd.DataFrame) and not candidate_pool_comparison.empty:
                    st.subheader("候选池对照")
                    st.dataframe(candidate_pool_comparison, use_container_width=True, hide_index=True)
                render_prediction_model_download(symbol_code)
            except Exception as e:
                logger.exception("Strategy refresh failed on prediction page")
                st.error(f"策略回测刷新失败: {str(e)}")
        elif st.session_state.get('base_prediction_result') is not None:
            st.info("预测参数已变化，请点击“开始预测”生成新的模型预测缓存。")
