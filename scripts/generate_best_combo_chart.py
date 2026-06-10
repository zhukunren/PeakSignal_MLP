import json
import time
import traceback

import pandas as pd

from src.models import set_seed
from src.preprocess import preprocess_data
from src.train import train_model
from src.predict import predict_new_data
from src.tushare_function import select_time
from src.plot_candlestick import plot_candlestick

set_seed(42)

N = 30
mixture_depth = 1
classifier_name = 'MLP'
oversample_method = 'SMOTE'
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
best_peak_idx = 3
best_trough_idx = 9

log_path = 'best_combo_chart.log'
html_path = 'best_combo_prediction_kline.html'
json_path = 'best_combo_prediction_result.json'

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

    peak_choice = None
    trough_choice = None
    max_round = max(best_peak_idx, best_trough_idx)
    for i in range(max_round):
        round_no = i + 1
        log(f'开始训练第 {round_no}/{max_round} 组模型')
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
        if round_no == best_peak_idx:
            peak_choice = (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
            log(f'已保存峰模型 {best_peak_idx}: score={peak_best_score:.4f}, threshold={peak_threshold:.4f}')
        if round_no == best_trough_idx:
            trough_choice = (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
            log(f'已保存谷模型 {best_trough_idx}: score={trough_best_score:.4f}, threshold={trough_threshold:.4f}')
        log(f'完成第 {round_no}/{max_round} 组，用时 {time.time()-t0:.1f}s')

    if peak_choice is None or trough_choice is None:
        raise RuntimeError('未能取得最佳组合对应模型')

    log('准备预测区间原始数据')
    pred_raw = select_time(raw.copy(), pred_start, pred_end)
    pm, ps, psel, pfeats, pth = peak_choice
    tm, ts, tsel, tfeats, tth = trough_choice
    final_result, final_bt, final_trades_df = predict_new_data(
        pred_raw.copy(),
        pm, ps, psel, pfeats, pth,
        tm, ts, tsel, tfeats, tth,
        N,
        mixture_depth,
        window_size=10,
        eval_mode=False,
        N_buy=N_buy,
        N_sell=N_sell,
        N_newhigh=N_newhigh,
        enable_chase=enable_chase,
        enable_stop_loss=enable_stop_loss,
        enable_change_signal=enable_change_signal,
    )
    log(f'预测完成，累计收益率={final_bt.get("累计收益率", 0)*100:.2f}%, 交易笔数={final_bt.get("交易笔数", 0)}')

    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
    fig = plot_candlestick(
        final_result.copy(),
        '000001.SH',
        pd.to_datetime(pred_start, format='%Y%m%d'),
        pd.to_datetime(pred_end, format='%Y%m%d'),
        peaks_pred,
        troughs_pred,
        prediction=True,
        bt_result=final_bt,
    )
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)

    out = {
        'chart_html': html_path,
        'best_combo': {
            'peak_model_index': best_peak_idx,
            'trough_model_index': best_trough_idx,
        },
        'bt_result': final_bt,
        'predicted_peaks': int(len(peaks_pred)),
        'predicted_troughs': int(len(troughs_pred)),
        'trades_count': int(len(final_trades_df)),
        'elapsed_seconds': time.time() - start_time,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f'K线图已导出: {html_path}')
    log(json.dumps(out, ensure_ascii=False, default=str))
except Exception:
    err = traceback.format_exc()
    log('任务失败')
    log(err)
    raise
