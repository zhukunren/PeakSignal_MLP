import json
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd

from ml_trader.models.architectures import set_seed
from ml_trader.data.preprocessor import preprocess_data
from ml_trader.models.trainer import train_model
from ml_trader.models.predictor import predict_new_data
from ml_trader.data.loader import select_time

set_seed(42)

N = 30
mixture_depth = 1
classifier_name = 'MLP'
oversample_method = 'SMOTE'
num_rounds = 10
train_start = '20000101'
train_end = '20201231'
pred_start = '20210101'
pred_end = '20251231'
N_buy = 10
N_sell = 10
N_newhigh = 60
enable_chase = False
enable_stop_loss = False
enable_change_signal = False

log_path = 'training_combo_check.log'
result_path = 'training_combo_check_result.json'

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

open(log_path, 'w', encoding='utf-8').close()
start_time = time.time()
try:
    log('读取本地完整数据.csv')
    raw = pd.read_csv('完整数据.csv')
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
    keep_cols = [c for c in base_cols if c in raw.columns]
    raw = raw[keep_cols].copy()
    raw['TradeDate'] = raw['TradeDate'].astype(str).str.replace('-', '', regex=False)
    log(f'原始数据 shape={raw.shape}, date={raw.TradeDate.min()}~{raw.TradeDate.max()}')

    log('执行 preprocess_data(mark_labels=True)')
    processed, all_features = preprocess_data(raw.copy(), N, mixture_depth, mark_labels=True)
    processed_for_select = processed.copy()
    processed_for_select['TradeDate'] = pd.to_datetime(processed_for_select['TradeDate']).dt.strftime('%Y%m%d')
    train_df = select_time(processed_for_select, train_start, train_end)
    log(f'训练集 shape={train_df.shape}, features={len(all_features)}')
    log(f"训练标签 Peak={int(train_df['Peak'].sum())}, Trough={int(train_df['Trough'].sum())}")

    peak_models = []
    trough_models = []
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

    log('准备预测区间原始数据')
    pred_raw = select_time(raw.copy(), pred_start, pred_end)
    log(f'预测集 shape={pred_raw.shape}')

    best = None
    failures = []
    combos = list(product(range(len(peak_models)), range(len(trough_models))))
    log(f'开始组合筛选 total={len(combos)}')
    for idx, (pi, ti) in enumerate(combos, start=1):
        pm, ps, psel, pfeats, pth = peak_models[pi]
        tm, ts, tsel, tfeats, tth = trough_models[ti]
        try:
            _, bt, trades = predict_new_data(
                pred_raw.copy(),
                pm, ps, psel, pfeats, pth,
                tm, ts, tsel, tfeats, tth,
                N,
                mixture_depth,
                window_size=10,
                eval_mode=True,
                N_buy=1,
                N_sell=1,
                N_newhigh=60,
                enable_chase=False,
                enable_stop_loss=False,
                enable_change_signal=False,
            )
            cumulative_return = float(bt.get('累计收益率', float('-inf')))
            excess_return = float(bt.get('超额收益率', float('-inf')))
            trade_count = int(bt.get('交易笔数', 0))
            if best is None or cumulative_return > best['累计收益率']:
                best = {
                    'combo_index': idx,
                    'peak_model_index': pi + 1,
                    'trough_model_index': ti + 1,
                    '累计收益率': cumulative_return,
                    '超额收益率': excess_return,
                    '胜率': float(bt.get('胜率', 0)),
                    '最大回撤': float(bt.get('最大回撤', 0)),
                    '交易笔数': trade_count,
                    '年化夏普比率': float(bt.get('年化夏普比率', 0)),
                }
            if idx % 10 == 0 or idx == 1:
                log(f'组合进度 {idx}/{len(combos)}, 当前最佳累计收益率={best["累计收益率"]*100:.2f}%')
        except Exception as e:
            failures.append({'combo_index': idx, 'peak_model_index': pi + 1, 'trough_model_index': ti + 1, 'error': str(e)})
            log(f'组合 {idx}/{len(combos)} 失败: {e}')

    if best is None:
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
            'selection_metric': '累计收益率',
        },
        'best': best,
        'best_cumulative_return_gt_100pct': best['累计收益率'] > 1.0,
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
