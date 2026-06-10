import json
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd

from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import get_trade_signal
from backtest import backtest_results
from tushare_function import read_day_from_tushare, select_time

set_seed(42)

N = 20
mixture_depth = 1
classifier_name = 'MLP'
oversample_method = 'SMOTE'
num_rounds = 60
train_start = '20000101'
train_end = '20201231'
pred_start = '20210101'
pred_end = '20260608'
N_buy = 10
N_sell = 10
N_newhigh = 60
enable_chase = False
enable_stop_loss = False
enable_change_signal = False

log_path = 'fixed_feature_training_check.log'
result_path = 'fixed_feature_training_check_result.json'


def normalize_market_data(df):
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
    keep_cols = [c for c in base_cols if c in df.columns]
    out = df[keep_cols].copy()
    out['TradeDate'] = out['TradeDate'].astype(str).str.replace('-', '', regex=False)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out.dropna(subset=['TradeDate', 'Open', 'High', 'Low', 'Close'])

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def predict_label_arrays(data_preprocessed, model, scaler, selector, selected_features, threshold):
    missing_features = [f for f in selected_features if f not in data_preprocessed.columns]
    if missing_features:
        for feature in missing_features:
            data_preprocessed[feature] = 0

    X_new = data_preprocessed[selected_features].fillna(0)
    X_scaled = scaler.transform(X_new).astype(np.float32)
    X_model = selector.transform(X_scaled) if selector is not None else X_scaled

    if hasattr(model, "predict_proba"):
        logits = model.predict_proba(X_model)
        if getattr(logits, "ndim", 1) == 2:
            probas = logits[:, 1]
        else:
            probas = 1 / (1 + np.exp(-logits))
    else:
        probas = model.predict(X_model).astype(float)

    preds = (probas > threshold).astype(int)
    return probas, preds


def suppress_repeated_signals(df):
    df = df.copy()
    df.index = df.index.astype(str)
    for idx, index in enumerate(df.index):
        if df.loc[index, 'Peak_Prediction'] == 1:
            start = idx + 1
            end = min(idx + 20, len(df))
            df.iloc[start:end, df.columns.get_loc('Peak_Prediction')] = 0
        if df.loc[index, 'Trough_Prediction'] == 1:
            start = idx + 1
            end = min(idx + 20, len(df))
            df.iloc[start:end, df.columns.get_loc('Trough_Prediction')] = 0
    return df

open(log_path, 'w', encoding='utf-8').close()
start_time = time.time()
try:
    log('读取本地完整数据.csv')
    raw = normalize_market_data(pd.read_csv('完整数据.csv'))
    log(f'本地数据 shape={raw.shape}, date={raw.TradeDate.min()}~{raw.TradeDate.max()}')
    if raw['TradeDate'].max() < pred_end:
        log(f'本地数据不足，使用 Tushare 获取 000001.SH 指数行情至 {pred_end}')
        ts_df = read_day_from_tushare('000001.SH', symbol_type='index', start_date='19920101', end_date=pred_end)
        if ts_df.empty:
            raise RuntimeError(f'Tushare 未返回行情，无法覆盖回测截止日 {pred_end}')
        ts_raw = normalize_market_data(ts_df.reset_index(drop=True))
        raw = (
            pd.concat([raw, ts_raw], ignore_index=True)
            .drop_duplicates(subset=['TradeDate'], keep='last')
            .sort_values('TradeDate')
            .reset_index(drop=True)
        )
        log(f'合并后数据 shape={raw.shape}, date={raw.TradeDate.min()}~{raw.TradeDate.max()}')
    if raw['TradeDate'].max() < pred_end:
        raise RuntimeError(f'行情截止 {raw.TradeDate.max()}，仍早于回测截止 {pred_end}')

    log('执行 preprocess_data(mark_labels=True)')
    processed, all_features = preprocess_data(raw.copy(), N, mixture_depth, mark_labels=True)
    processed_for_select = processed.copy()
    processed_for_select['TradeDate'] = pd.to_datetime(processed_for_select['TradeDate']).dt.strftime('%Y%m%d')
    train_df = select_time(processed_for_select, train_start, train_end)
    log(f'训练集 shape={train_df.shape}, features={len(all_features)}')
    log(f"训练标签 Peak={int(train_df['Peak'].sum())}, Trough={int(train_df['Trough'].sum())}")

    log('准备预测区间原始数据')
    pred_raw = raw.copy()
    log(f'预测特征上下文 shape={pred_raw.shape}，回测区间={pred_start}~{pred_end}')
    log('执行预测集 preprocess_data(mark_labels=True)，训练过程中将增量复用同一特征矩阵')
    pred_preprocessed, _ = preprocess_data(pred_raw.copy(), N, mixture_depth=mixture_depth, mark_labels=True)

    trade_dates = pd.to_datetime(pred_preprocessed['TradeDate'], errors='coerce')
    backtest_mask = pd.Series(True, index=pred_preprocessed.index)
    backtest_mask &= trade_dates >= pd.to_datetime(pred_start, format='%Y%m%d')
    backtest_mask &= trade_dates <= pd.to_datetime(pred_end, format='%Y%m%d')
    mask_values = backtest_mask.to_numpy()
    base_backtest_df = pred_preprocessed.loc[backtest_mask].copy()
    log(f'预测集预处理完成 shape={pred_preprocessed.shape}, 回测切片 shape={base_backtest_df.shape}')

    peak_models = []
    trough_models = []
    peak_predictions = []
    trough_predictions = []
    best_by_excess_checkpoint = None
    best_by_cumulative_checkpoint = None
    checkpoint_failures = []
    evaluated_combos = set()

    def write_checkpoint(completed_rounds, is_complete=False):
        checkpoint = {
            'params': {
                'N': N,
                'mixture_depth': mixture_depth,
                'classifier_name': classifier_name,
                'oversample_method': oversample_method,
                'num_rounds': num_rounds,
                'completed_rounds': completed_rounds,
                'train_start': train_start,
                'train_end': train_end,
                'pred_start': pred_start,
                'pred_end': pred_end,
                'selection_metric': '超额收益率',
                'change_type': 'feature_engineering_and_training_params_only',
                'checkpoint_complete': is_complete,
            },
            'best_by_excess': best_by_excess_checkpoint,
            'best_by_cumulative': best_by_cumulative_checkpoint,
            'best_excess_gt_100pct': (
                best_by_excess_checkpoint is not None
                and best_by_excess_checkpoint['超额收益率'] > 1.0
            ),
            'best_excess_gt_previous_46_46pct': (
                best_by_excess_checkpoint is not None
                and best_by_excess_checkpoint['超额收益率'] > 0.464648347023231
            ),
            'failures_count': len(checkpoint_failures),
            'first_failures': checkpoint_failures[:5],
            'elapsed_seconds': time.time() - start_time,
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def evaluate_checkpoint_combo(pi, ti):
        global best_by_excess_checkpoint, best_by_cumulative_checkpoint
        combo_key = (pi, ti)
        if combo_key in evaluated_combos:
            return
        evaluated_combos.add(combo_key)
        combo_index = pi * num_rounds + ti + 1
        try:
            peak_probas, peak_preds = peak_predictions[pi]
            trough_probas, trough_preds = trough_predictions[ti]
            combo_df = base_backtest_df.copy()
            combo_df['Peak_Probability'] = peak_probas[mask_values]
            combo_df['Peak_Prediction'] = peak_preds[mask_values]
            combo_df['Trough_Probability'] = trough_probas[mask_values]
            combo_df['Trough_Prediction'] = trough_preds[mask_values]
            combo_df = suppress_repeated_signals(combo_df)
            signal_df = get_trade_signal(combo_df)
            bt, trades = backtest_results(
                combo_df,
                signal_df,
                1,
                1,
                False,
                False,
                initial_capital=1_000_000,
            )
            record = {
                'combo_index': combo_index,
                'peak_model_index': pi + 1,
                'trough_model_index': ti + 1,
                '累计收益率': float(bt.get('累计收益率', float('-inf'))),
                '超额收益率': float(bt.get('超额收益率', float('-inf'))),
                '胜率': float(bt.get('胜率', 0)),
                '最大回撤': float(bt.get('最大回撤', 0)),
                '交易笔数': int(bt.get('交易笔数', 0)),
                '年化夏普比率': float(bt.get('年化夏普比率', 0)),
            }
            if best_by_excess_checkpoint is None or record['超额收益率'] > best_by_excess_checkpoint['超额收益率']:
                best_by_excess_checkpoint = record
            if best_by_cumulative_checkpoint is None or record['累计收益率'] > best_by_cumulative_checkpoint['累计收益率']:
                best_by_cumulative_checkpoint = record
        except Exception as e:
            checkpoint_failures.append({
                'combo_index': combo_index,
                'peak_model_index': pi + 1,
                'trough_model_index': ti + 1,
                'error': str(e),
            })
            log(f'checkpoint 组合 {combo_index} 失败: {e}')

    for i in range(num_rounds):
        log(f'开始训练第 {i+1}/{num_rounds} 组模型')
        t0 = time.time()
        (peak_model, peak_scaler, peak_selector, peak_selected_features,
         all_features_peak, peak_best_score, peak_metrics, peak_threshold,
         trough_model, trough_scaler, trough_selector, trough_selected_features,
         all_features_trough, trough_best_score, trough_metrics, trough_threshold) = train_model(
            train_df,
            N,
            all_features,
            classifier_name,
            mixture_depth,
            'auto',
            oversample_method,
        )
        peak_models.append((peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold))
        trough_models.append((trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold))
        log(
            f'完成第 {i+1}/{num_rounds} 组，用时 {time.time()-t0:.1f}s, '
            f'peak_score={peak_best_score:.4f}, peak_th={peak_threshold:.4f}, '
            f'trough_score={trough_best_score:.4f}, trough_th={trough_threshold:.4f}'
        )

        peak_probas, peak_preds = predict_label_arrays(pred_preprocessed, peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
        peak_predictions.append((peak_probas, peak_preds))
        log(f'Peak模型 {i+1}/{num_rounds} 增量预测完成，信号数={int(peak_preds[mask_values].sum())}')
        trough_probas, trough_preds = predict_label_arrays(pred_preprocessed, trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
        trough_predictions.append((trough_probas, trough_preds))
        log(f'Trough模型 {i+1}/{num_rounds} 增量预测完成，信号数={int(trough_preds[mask_values].sum())}')

        new_pi = len(peak_predictions) - 1
        new_ti = len(trough_predictions) - 1
        for ti in range(len(trough_predictions)):
            evaluate_checkpoint_combo(new_pi, ti)
        for pi in range(len(peak_predictions) - 1):
            evaluate_checkpoint_combo(pi, new_ti)
        write_checkpoint(i + 1, is_complete=False)
        if best_by_excess_checkpoint is not None:
            log(
                f'checkpoint 已完成 {i+1}/{num_rounds} 组，'
                f'最佳超额={best_by_excess_checkpoint["超额收益率"]*100:.2f}%, '
                f'对应累计={best_by_excess_checkpoint["累计收益率"]*100:.2f}%'
            )

    log('准备预测区间原始数据')
    pred_raw = raw.copy()
    log(f'预测特征上下文 shape={pred_raw.shape}，回测区间={pred_start}~{pred_end}')
    log('执行预测集 preprocess_data(mark_labels=True)，后续组合复用同一特征矩阵')
    pred_preprocessed, _ = preprocess_data(pred_raw.copy(), N, mixture_depth=mixture_depth, mark_labels=True)

    trade_dates = pd.to_datetime(pred_preprocessed['TradeDate'], errors='coerce')
    backtest_mask = pd.Series(True, index=pred_preprocessed.index)
    backtest_mask &= trade_dates >= pd.to_datetime(pred_start, format='%Y%m%d')
    backtest_mask &= trade_dates <= pd.to_datetime(pred_end, format='%Y%m%d')
    mask_values = backtest_mask.to_numpy()
    base_backtest_df = pred_preprocessed.loc[backtest_mask].copy()
    log(f'预测集预处理完成 shape={pred_preprocessed.shape}, 回测切片 shape={base_backtest_df.shape}')

    peak_predictions = []
    trough_predictions = []
    log('开始单模型预测缓存')
    for pi, (pm, ps, psel, pfeats, pth) in enumerate(peak_models, start=1):
        probas, preds = predict_label_arrays(pred_preprocessed, pm, ps, psel, pfeats, pth)
        peak_predictions.append((probas, preds))
        log(f'Peak模型 {pi}/{len(peak_models)} 预测完成，信号数={int(preds[mask_values].sum())}')
    for ti, (tm, ts, tsel, tfeats, tth) in enumerate(trough_models, start=1):
        probas, preds = predict_label_arrays(pred_preprocessed, tm, ts, tsel, tfeats, tth)
        trough_predictions.append((probas, preds))
        log(f'Trough模型 {ti}/{len(trough_models)} 预测完成，信号数={int(preds[mask_values].sum())}')

    best_by_excess = None
    best_by_cumulative = None
    failures = []
    combos = list(product(range(len(peak_models)), range(len(trough_models))))
    log(f'开始组合筛选 total={len(combos)}，筛选指标=超额收益率')
    for idx, (pi, ti) in enumerate(combos, start=1):
        try:
            peak_probas, peak_preds = peak_predictions[pi]
            trough_probas, trough_preds = trough_predictions[ti]
            combo_df = base_backtest_df.copy()
            combo_df['Peak_Probability'] = peak_probas[mask_values]
            combo_df['Peak_Prediction'] = peak_preds[mask_values]
            combo_df['Trough_Probability'] = trough_probas[mask_values]
            combo_df['Trough_Prediction'] = trough_preds[mask_values]
            combo_df = suppress_repeated_signals(combo_df)
            signal_df = get_trade_signal(combo_df)
            bt, trades = backtest_results(
                combo_df,
                signal_df,
                1,
                1,
                False,
                False,
                initial_capital=1_000_000,
            )
            record = {
                'combo_index': idx,
                'peak_model_index': pi + 1,
                'trough_model_index': ti + 1,
                '累计收益率': float(bt.get('累计收益率', float('-inf'))),
                '超额收益率': float(bt.get('超额收益率', float('-inf'))),
                '胜率': float(bt.get('胜率', 0)),
                '最大回撤': float(bt.get('最大回撤', 0)),
                '交易笔数': int(bt.get('交易笔数', 0)),
                '年化夏普比率': float(bt.get('年化夏普比率', 0)),
            }
            if best_by_excess is None or record['超额收益率'] > best_by_excess['超额收益率']:
                best_by_excess = record
            if best_by_cumulative is None or record['累计收益率'] > best_by_cumulative['累计收益率']:
                best_by_cumulative = record
            if idx % 10 == 0 or idx == 1:
                log(
                    f'组合进度 {idx}/{len(combos)}, '
                    f'最佳超额={best_by_excess["超额收益率"]*100:.2f}%, '
                    f'对应累计={best_by_excess["累计收益率"]*100:.2f}%'
                )
        except Exception as e:
            failures.append({'combo_index': idx, 'peak_model_index': pi + 1, 'trough_model_index': ti + 1, 'error': str(e)})
            log(f'组合 {idx}/{len(combos)} 失败: {e}')

    if best_by_excess is None:
        raise RuntimeError(f'所有组合失败，失败数={len(failures)}，首个错误={failures[0] if failures else None}')

    result = {
        'params': {
            'N': N,
            'mixture_depth': mixture_depth,
            'classifier_name': classifier_name,
            'oversample_method': oversample_method,
            'num_rounds': num_rounds,
            'train_start': train_start,
            'train_end': train_end,
            'pred_start': pred_start,
            'pred_end': pred_end,
            'selection_metric': '超额收益率',
            'change_type': 'feature_engineering_and_training_params_only',
        },
        'best_by_excess': best_by_excess,
        'best_by_cumulative': best_by_cumulative,
        'best_excess_gt_100pct': best_by_excess['超额收益率'] > 1.0,
        'best_excess_gt_previous_46_46pct': best_by_excess['超额收益率'] > 0.464648347023231,
        'failures_count': len(failures),
        'first_failures': failures[:5],
        'elapsed_seconds': time.time() - start_time,
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log('完成组合筛选')
    log(json.dumps(result, ensure_ascii=False))
except Exception:
    err = traceback.format_exc()
    log('任务失败')
    log(err)
    raise
