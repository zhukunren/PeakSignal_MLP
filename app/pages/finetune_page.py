from app.ui_helpers import *
from ml_trader.logging_config import get_logger


logger = get_logger(__name__)


def render(data_source, symbol_code, classifier_name, N, mixture_depth, oversample_method):
    st.subheader("模型微调（微调已有模型）")
    if st.session_state.final_result is None or st.session_state.new_df_raw is None:
        st.warning("请先在 [预测] 页完成一次预测，才能进行模型微调。")
    else:
        inc_col1, inc_col2 = st.columns(2)
        with inc_col1:
            inc_start_date = st.date_input(
                "模型微调起始日期",
                st.session_state.get('pred_start', datetime(2021, 1, 1)),
                key="inc_start_tab3"
            )
        with inc_col2:
            inc_end_date = st.date_input(
                "模型微调结束日期",
                st.session_state.get('pred_end', datetime.now()),
                key="inc_end_tab3"
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
                peak_progress = StreamlitProgressReporter()
                for i in range(10):
                    peak_progress.update(i, 10, f"峰模型 - 第 {i+1}/10 轮微调...", force=True)
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
                    peak_progress.update(i + 1, 10, f"峰模型 - 第 {i+1}/10 轮完成", force=True)
                peak_progress.finish("峰模型 10 轮微调全部完成！")
                
                st.success("峰模型 10 轮微调全部完成！")

                # ---- 3.2 对谷模型进行 10 次微调 ----
                st.write("正在对谷模型进行 10 轮微调训练...")
                trough_progress = StreamlitProgressReporter()
                for i in range(10):
                    trough_progress.update(i, 10, f"谷模型 - 第 {i+1}/10 轮微调...", force=True)
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
                    trough_progress.update(i + 1, 10, f"谷模型 - 第 {i+1}/10 轮完成", force=True)
                trough_progress.finish("谷模型 10 轮微调全部完成！")

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
                combo_progress = StreamlitProgressReporter()

                for idx, (peak_tuple, trough_tuple) in enumerate(product(
                    st.session_state.peak_models_finetuned_list,
                    st.session_state.trough_models_finetuned_list
                )):
                    combo_progress.update(idx + 1, total_combos, f"第 {idx+1}/{total_combos} 组合...")

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
                            N_buy=1,
                            N_sell=1,
                            N_newhigh=60,
                            enable_chase=False,
                            enable_stop_loss=False,
                            enable_change_signal=False,
                        )
                        current_excess = bt_result_temp.get('超额收益率', -np.inf)
                        if current_excess > best_excess_finetune:
                            best_excess_finetune = current_excess
                            best_combo_finetune = (peak_tuple, trough_tuple)
                    except Exception as e:
                        # 某些组合可能因为数据极端/过采样导致报错，忽略
                        logger.debug("Finetune combo failed", exc_info=True)
                        pass

                combo_progress.clear()

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

                inc_base_result, inc_base_bt, _ = predict_new_data(
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
                    N_buy=1,
                    N_sell=1,
                    N_newhigh=60,
                    enable_chase=False,
                    enable_stop_loss=False,
                    enable_change_signal=False,
                )
                inc_final_result, inc_final_bt, inc_final_trades_df = apply_strategy_to_prediction(
                    inc_base_result,
                    st.session_state.get('n_buy_val', 10),
                    st.session_state.get('n_sell_val', 10),
                    st.session_state.get('n_newhigh_val', 60),
                    st.session_state.get('enable_chase_val', False),
                    st.session_state.get('enable_stop_loss_val', False),
                    st.session_state.get('enable_change_signal_val', False),
                )

                # 保存微调后的结果
                st.session_state.inc_base_prediction_result = inc_base_result
                st.session_state.inc_base_selection_bt = inc_base_bt
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
                        peaks_pred_orig = orig_result[orig_result['Peak_Prediction'] == 1]
                        troughs_pred_orig = orig_result[orig_result['Trough_Prediction'] == 1]
                        
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

            except Exception as e:
                logger.exception("Model finetune failed")
                st.error(f"模型微调过程出现错误: {str(e)}")
                st.exception(e)

        if is_downloadable_model_dict(st.session_state.get('models')):
            add_model_save_functionality(symbol_code)
