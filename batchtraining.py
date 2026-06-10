#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
说明：
1. 该程序对每个股票代码（列表中的每个代码）执行以下流程：
   - 从 tushare 获取原始数据（股票类型）。
   - 对数据进行预处理（默认窗口长度 N=30，因子混合深度默认为1，并自动打标签）。
   - 根据指定训练日期区间（例如 2000-01-01 至 2020-12-31）截取训练数据。
   - 进行多轮（例如 10 轮）模型训练，每轮调用 train_model 函数获得一组峰/谷模型，
     并将每轮模型保存到列表中。
   - 使用预测日期区间（例如 2021-01-01 至当前日期）获取预测数据。
   - 对所有训练得到的峰/谷模型组合进行回测（调用 predict_new_data 函数，设置 eval_mode=True），
     按“超额收益率”筛选出最佳模型组合。
   - 用最佳模型组合进行最终预测回测，获得回测指标。
   - 将各股票的回测指标（累计收益率、超额收益率、胜率、交易笔数、最大回撤、年化夏普比率）保存到列表中。
2. 除了保存回测结果外，每只股票最终选定的模型也会保存为 pickle 文件，方便以后上传到原始程序中复现。
3. 程序中使用的默认参数（如 N、mixture_depth、训练/预测日期、模型参数等）均可根据需要进行修改。
"""

import os
import copy
import pickle
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch

# 从各模块导入所需函数（请确保相应模块在 PYTHONPATH 中）
from models import set_seed, time_aware_oversampling
from preprocess import preprocess_data, create_pos_neg_sequences_by_consecutive_labels
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick

from WindPy import w
w.start()
stock_codes = w.wset("sectorconstituent","date=2025-03-16;sectorid=1000008491000000;field=wind_code").Data[0]

def main():
    # 设置随机种子
    set_seed(42)
    
    # --------------------- 参数设置 --------------------- #
    # 股票代码列表（示例，用户可自行修改）
    #stock_codes = ["000001.SZ", "601555.SH"]
    
    # 默认参数设置
    N = 30                              # 窗口长度
    mixture_depth = 1                   # 因子混合深度
    n_features_selected = "auto"        # 自动特征选择（如果不是自动，则指定特征数量）
    
    # 训练日期区间
    train_start_date = datetime(2000, 1, 1)
    train_end_date   = datetime(2020, 12, 31)
    
    # 预测/回测日期区间
    pred_start_date = datetime(2021, 1, 1)
    pred_end_date   = datetime.now()
    
    num_rounds = 10                     # 训练轮数（多轮训练以增加模型多样性）
    use_best_combo = True               # 是否使用最佳模型组合进行预测（默认True）
    window_size = 10                    # 预测时的滑动窗口长度
    
    # 策略参数（此处均使用默认值，可根据需要调整）
    N_buy = 10
    N_sell = 10
    N_newhigh = 60
    enable_chase = False
    enable_stop_loss = False
    enable_change_signal = False
    
    # 存放所有股票的回测结果
    results_list = []
    
    # 遍历每个股票代码进行处理
    for stock_code in stock_codes:
        print(f"========== 正在处理股票：{stock_code} ==========")
        try:
            # --------------------- 数据读取与预处理 --------------------- #
            # 获取原始数据（这里 symbol_type 设为 'stock'，如需处理指数请修改）
            raw_data = read_day_from_tushare(stock_code, 'stock')
            
            # 预处理数据并自动打标签（返回预处理后的数据和所有特征列表）
            raw_data_processed, all_features_train = preprocess_data(raw_data, N, mixture_depth, mark_labels=True)
            
            # 截取训练数据：根据指定的训练日期区间（格式为 "YYYYMMDD"）
            df_preprocessed_train = select_time(
                raw_data_processed,
                train_start_date.strftime("%Y%m%d"),
                train_end_date.strftime("%Y%m%d")
            )
            
            # --------------------- 模型训练 --------------------- #
            peak_models_list = []
            trough_models_list = []
            
            print("开始多轮模型训练...")
            for i in range(num_rounds):
                print(f"【{stock_code}】第 {i+1}/{num_rounds} 轮训练...")
                # 调用 train_model 进行训练，选择的模型类型默认为 MLP（即“深度学习”）
                train_output = train_model(
                    df_preprocessed_train,
                    N,
                    all_features_train,
                    classifier_name="MLP",           # 此处使用默认深度学习模型
                    mixture_depth=mixture_depth,
                    n_features_selected=n_features_selected,
                    oversample_method="SMOTE"         # 默认采用 SMOTE 过采样
                )
                # train_model 返回：
                # (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                #  trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_best_score, trough_metrics, trough_threshold)
                (peak_model, peak_scaler, peak_selector, peak_selected_features,
                 all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                 trough_model, trough_scaler, trough_selector, trough_selected_features,
                 all_features_trough, trough_best_score, trough_metrics, trough_threshold) = train_output
                
                peak_models_list.append((peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold))
                trough_models_list.append((trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold))
            
            # --------------------- 模型筛选与预测回测 --------------------- #
            # 获取预测数据（注意，此处依然调用 tushare 获取数据，然后预处理，不打标签）
            raw_data_pred = read_day_from_tushare(stock_code, 'stock')
            raw_data_pred_processed, _ = preprocess_data(raw_data_pred, N, mixture_depth, mark_labels=False)
            new_df_raw = select_time(
                raw_data_pred_processed,
                pred_start_date.strftime("%Y%m%d"),
                pred_end_date.strftime("%Y%m%d")
            )
            
            # 采用多组合测试筛选最佳模型组合（基于超额收益率）
            best_excess = -np.inf
            best_models = None
            if use_best_combo:
                print("开始组合筛选，测试所有峰/谷模型组合...")
                model_combinations = list(product(peak_models_list, trough_models_list))
                total_combos = len(model_combinations)
                for idx, (peak_m, trough_m) in enumerate(model_combinations):
                    # 解包模型参数
                    pm, ps, psel, pfeats, pth = peak_m
                    tm, ts, tsel, tfeats, tth = trough_m
                    try:
                        # 调用 predict_new_data 进行回测（eval_mode=True 表示只回测，不保存最终结果）
                        _, bt_result_temp, _ = predict_new_data(
                            new_df_raw,
                            pm, ps, psel, pfeats, pth,
                            tm, ts, tsel, tfeats, tth,
                            N,
                            mixture_depth,
                            window_size=window_size,
                            eval_mode=True,
                            N_buy=1,
                            N_sell=1,
                            N_newhigh=60,
                            enable_chase=False,
                            enable_stop_loss=False,
                            enable_change_signal=False,
                        )
                        current_excess = bt_result_temp.get('超额收益率', -np.inf)
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
                    except Exception as e:
                        print(f"组合 {idx+1}/{total_combos} 测试失败：{e}")
                        continue
                if best_models is None:
                    print(f"股票 {stock_code} 的所有模型组合均回测失败，跳过该股票。")
                    continue
                print(f"股票 {stock_code} 最佳组合超额收益率：{best_excess*100:.2f}%")
            else:
                # 若不采用组合筛选，则使用最后一轮训练的模型
                best_models = {
                    'peak_model': peak_models_list[-1][0],
                    'peak_scaler': peak_models_list[-1][1],
                    'peak_selector': peak_models_list[-1][2],
                    'peak_selected_features': peak_models_list[-1][3],
                    'peak_threshold': peak_models_list[-1][4],
                    'trough_model': trough_models_list[-1][0],
                    'trough_scaler': trough_models_list[-1][1],
                    'trough_selector': trough_models_list[-1][2],
                    'trough_selected_features': trough_models_list[-1][3],
                    'trough_threshold': trough_models_list[-1][4]
                }
            
            # 用最佳模型组合进行最终预测与回测（eval_mode=False 表示保存最终预测结果）
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
                N,
                mixture_depth,
                window_size=window_size,
                eval_mode=False,
                N_buy=N_buy,
                N_sell=N_sell,
                N_newhigh=N_newhigh,
                enable_chase=enable_chase,
                enable_stop_loss=enable_stop_loss,
                enable_change_signal=enable_change_signal,
            )
            
            # 提取关键回测指标
            metrics = {
                '股票代码': stock_code,
                '累计收益率': final_bt.get('累计收益率', 0),
                '超额收益率': final_bt.get('超额收益率', 0),
                '胜率': final_bt.get('胜率', 0),
                '交易笔数': final_bt.get('交易笔数', 0),
                '最大回撤': final_bt.get('最大回撤', 0),
                '夏普比率': final_bt.get('年化夏普比率', 0)
            }
            results_list.append(metrics)
            print(f"股票 {stock_code} 回测结果：{metrics}\n")
            
            # --------------------- 保存最终模型 --------------------- #
            # 将最终选定的模型（峰/谷模型及其参数）保存为 pickle 文件，
            # 方便以后上传到原始程序中复现
            # 指定 CSV 和模型保存的目录（可根据需要修改路径）
            csv_save_dir = "csv_results"      # 保存回测结果 CSV 的目录
            model_save_dir = "saved_models"   # 保存模型文件的目录

            # 如果目录不存在则创建
            if not os.path.exists(csv_save_dir):
                os.makedirs(csv_save_dir)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            

            final_model_dict = {
                'peak_model': best_models['peak_model'],
                'peak_scaler': best_models['peak_scaler'],
                'peak_selector': best_models['peak_selector'],
                'peak_selected_features': best_models['peak_selected_features'],
                'peak_threshold': best_models['peak_threshold'],
                'trough_model': best_models['trough_model'],
                'trough_scaler': best_models['trough_scaler'],
                'trough_selector': best_models['trough_selector'],
                'trough_selected_features': best_models['trough_selected_features'],
                'trough_threshold': best_models['trough_threshold'],
                'N': N,
                'mixture_depth': mixture_depth,
                'train_date_range': (train_start_date.strftime("%Y%m%d"), train_end_date.strftime("%Y%m%d")),
                'pred_date_range': (pred_start_date.strftime("%Y%m%d"), pred_end_date.strftime("%Y%m%d")),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            final_model_file = os.path.join(model_save_dir, f"{stock_code}.pkl")
            with open(final_model_file, "wb") as f:
                pickle.dump(final_model_dict, f)
            print(f"股票 {stock_code} 的最终模型已保存到 {final_model_file}")
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时出现错误：{e}")
    
    # --------------------- 汇总保存结果 --------------------- #
    if results_list:
        results_df = pd.DataFrame(results_list)
        output_csv = os.path.join(csv_save_dir, "all_stocks_backtest_results.csv")
        results_df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n所有股票回测结果已保存至 {output_csv}")
    else:
        print("未获得任何股票的回测结果。")

if __name__ == "__main__":
    main()
