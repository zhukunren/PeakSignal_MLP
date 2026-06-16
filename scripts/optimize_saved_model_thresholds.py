import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_trader.models.predictor import get_trade_signal
from ml_trader.trading.backtest import backtest_results


PRED_START = "20210101"
PRED_END = "20260608"
N_BUY = 10
N_SELL = 10
ENABLE_CHASE = False
ENABLE_STOP_LOSS = False


def predict_label_arrays(data_preprocessed, model, scaler, selector, selected_features, threshold):
    missing_features = [feature for feature in selected_features if feature not in data_preprocessed.columns]
    if missing_features:
        data_preprocessed = data_preprocessed.copy()
        for feature in missing_features:
            data_preprocessed[feature] = 0

    x_new = data_preprocessed[selected_features].fillna(0)
    x_scaled = scaler.transform(x_new).astype(np.float32)
    x_model = selector.transform(x_scaled) if selector is not None else x_scaled

    if hasattr(model, "predict_proba"):
        logits = model.predict_proba(x_model)
        if getattr(logits, "ndim", 1) == 2:
            probas = logits[:, 1]
        else:
            probas = 1 / (1 + np.exp(-logits))
    else:
        probas = model.predict(x_model).astype(float)

    preds = (probas > threshold).astype(np.int8)
    return probas.astype(np.float32), preds


def suppress_prediction_arrays(peak_preds, trough_preds):
    peak = peak_preds.copy()
    trough = trough_preds.copy()
    n_rows = len(peak)
    for idx in range(n_rows):
        if peak[idx] == 1:
            peak[idx + 1 : min(idx + 20, n_rows)] = 0
        if trough[idx] == 1:
            trough[idx + 1 : min(idx + 20, n_rows)] = 0
    return peak, trough


def suppress_repeated_signals(df):
    df = df.copy()
    df.index = df.index.astype(str)
    for idx, index in enumerate(df.index):
        if df.loc[index, "Peak_Prediction"] == 1:
            start = idx + 1
            end = min(idx + 20, len(df))
            df.iloc[start:end, df.columns.get_loc("Peak_Prediction")] = 0
        if df.loc[index, "Trough_Prediction"] == 1:
            start = idx + 1
            end = min(idx + 20, len(df))
            df.iloc[start:end, df.columns.get_loc("Trough_Prediction")] = 0
    return df


def prepare_market_arrays(base_backtest_df):
    df = base_backtest_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    daily_ret = np.zeros(len(df), dtype=float)
    daily_ret[1:] = closes[1:] / closes[:-1] - 1
    benchmark_return = closes[-1] / opens[0] - 1
    return df.index.to_numpy(), opens, closes, daily_ret, benchmark_return


def evaluate_fast(peak_probas, trough_probas, market_arrays, peak_threshold, trough_threshold):
    _, opens, closes, daily_ret, benchmark_return = market_arrays
    peak = (peak_probas > peak_threshold).astype(np.int8)
    trough = (trough_probas > trough_threshold).astype(np.int8)
    peak, trough = suppress_prediction_arrays(peak, trough)

    # Same precedence as get_trade_signal: Trough_Prediction overwrites Peak_Prediction.
    direction = np.zeros(len(peak), dtype=np.int8)
    direction[peak == 1] = -1
    direction[trough == 1] = 1

    holding = False
    entry_index = None
    entry_price = None
    trades = []
    for idx in range(len(direction) - 1):
        if not holding:
            if direction[idx] == 1:
                holding = True
                entry_index = idx + 1
                entry_price = opens[idx + 1]
        elif direction[idx] == -1:
            exit_index = idx + 1
            exit_price = opens[idx + 1]
            trades.append((entry_index, entry_price, exit_index, exit_price, exit_price / entry_price - 1))
            holding = False
            entry_index = None
            entry_price = None

    if holding:
        exit_index = len(direction) - 1
        exit_price = closes[-1]
        trades.append((entry_index, entry_price, exit_index, exit_price, exit_price / entry_price - 1))

    position = np.zeros(len(direction), dtype=np.int8)
    for entry_index, _, exit_index, _, _ in trades:
        if entry_index is not None and entry_index < len(position):
            position[entry_index:exit_index] = 1

    equity = 1.0
    curve = np.ones(len(direction), dtype=float)
    for idx in range(1, len(direction)):
        if position[idx - 1] == 1:
            equity *= 1 + daily_ret[idx]
        curve[idx] = equity

    strategy_return = equity - 1
    returns = np.array([trade[4] for trade in trades], dtype=float)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve / running_max - 1
    daily_equity_returns = np.zeros_like(curve)
    daily_equity_returns[1:] = curve[1:] / curve[:-1] - 1
    sharpe = (
        np.nan
        if daily_equity_returns.std() == 0
        else daily_equity_returns.mean() / daily_equity_returns.std() * np.sqrt(252)
    )

    return {
        "peak_threshold": float(peak_threshold),
        "trough_threshold": float(trough_threshold),
        "累计收益率": float(strategy_return),
        "超额收益率": float(strategy_return - benchmark_return),
        "交易笔数": int(len(trades)),
        "胜率": None if len(returns) == 0 else float((returns > 0).sum() / len(returns)),
        "最大回撤": float(drawdown.min()),
        "年化夏普比率": None if np.isnan(sharpe) else float(sharpe),
        "peak_signals": int(peak.sum()),
        "trough_signals": int(trough.sum()),
    }


def evaluate_official(data_cache, peak_probas, trough_probas, peak_threshold, trough_threshold):
    mask_values = data_cache["mask_values"]
    combo_df = data_cache["base_backtest_df"].copy()
    combo_df["Peak_Probability"] = peak_probas[mask_values]
    combo_df["Peak_Prediction"] = (peak_probas[mask_values] > peak_threshold).astype(np.int8)
    combo_df["Trough_Probability"] = trough_probas[mask_values]
    combo_df["Trough_Prediction"] = (trough_probas[mask_values] > trough_threshold).astype(np.int8)
    combo_df = suppress_repeated_signals(combo_df)
    signal_df = get_trade_signal(combo_df)
    bt_result, trades_df = backtest_results(
        combo_df,
        signal_df,
        N_BUY,
        N_SELL,
        ENABLE_CHASE,
        ENABLE_STOP_LOSS,
        initial_capital=1_000_000,
    )
    return bt_result, trades_df, combo_df


def search_thresholds(peak_slice, trough_slice, market_arrays):
    best = None
    evaluated = 0

    def consider(record):
        nonlocal best
        if best is None or record["累计收益率"] > best["累计收益率"]:
            best = record

    coarse_thresholds = np.round(np.arange(0.50, 0.991, 0.01), 2)
    for peak_threshold in coarse_thresholds:
        for trough_threshold in coarse_thresholds:
            evaluated += 1
            consider(evaluate_fast(peak_slice, trough_slice, market_arrays, peak_threshold, trough_threshold))

    peak_refine = np.round(
        np.arange(max(0.0, best["peak_threshold"] - 0.03), min(0.999, best["peak_threshold"] + 0.0301), 0.001),
        3,
    )
    trough_refine = np.round(
        np.arange(max(0.0, best["trough_threshold"] - 0.03), min(0.999, best["trough_threshold"] + 0.0301), 0.001),
        3,
    )
    for peak_threshold in peak_refine:
        for trough_threshold in trough_refine:
            evaluated += 1
            consider(evaluate_fast(peak_slice, trough_slice, market_arrays, peak_threshold, trough_threshold))

    return best, evaluated


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="base_98pct_round008_model.pkl")
    parser.add_argument("--data-cache", default="fixed_feature_combo_cache/prepared_data.pkl")
    parser.add_argument("--output-model", default="saved_models/optimized_2021_present_threshold_model.pkl")
    parser.add_argument("--output-report", default="saved_models/optimized_2021_present_threshold_model_report.json")
    args = parser.parse_args()

    start = time.time()
    base_model_path = ROOT_DIR / args.base_model
    data_cache_path = ROOT_DIR / args.data_cache
    output_model_path = ROOT_DIR / args.output_model
    output_report_path = ROOT_DIR / args.output_report

    with base_model_path.open("rb") as f:
        base_model = pickle.load(f)
    with data_cache_path.open("rb") as f:
        data_cache = pickle.load(f)

    pred_preprocessed = data_cache["pred_preprocessed"]
    mask_values = data_cache["mask_values"]
    base_backtest_df = data_cache["base_backtest_df"]

    peak_probas, _ = predict_label_arrays(
        pred_preprocessed,
        base_model["peak_model"],
        base_model["peak_scaler"],
        base_model["peak_selector"],
        base_model["peak_selected_features"],
        0.5,
    )
    trough_probas, _ = predict_label_arrays(
        pred_preprocessed,
        base_model["trough_model"],
        base_model["trough_scaler"],
        base_model["trough_selector"],
        base_model["trough_selected_features"],
        0.5,
    )

    market_arrays = prepare_market_arrays(base_backtest_df)
    baseline_fast = evaluate_fast(
        peak_probas[mask_values],
        trough_probas[mask_values],
        market_arrays,
        base_model["peak_threshold"],
        base_model["trough_threshold"],
    )
    best_fast, evaluated = search_thresholds(peak_probas[mask_values], trough_probas[mask_values], market_arrays)

    baseline_bt, baseline_trades, _ = evaluate_official(
        data_cache,
        peak_probas,
        trough_probas,
        base_model["peak_threshold"],
        base_model["trough_threshold"],
    )
    best_bt, best_trades, _ = evaluate_official(
        data_cache,
        peak_probas,
        trough_probas,
        best_fast["peak_threshold"],
        best_fast["trough_threshold"],
    )

    optimized_model = copy.copy(base_model)
    optimized_model["peak_threshold"] = best_fast["peak_threshold"]
    optimized_model["trough_threshold"] = best_fast["trough_threshold"]
    optimized_model["bt_result"] = best_bt
    optimized_model["optimized_from"] = str(base_model_path)
    optimized_model["optimization"] = {
        "type": "threshold_grid_search",
        "pred_start": PRED_START,
        "pred_end": PRED_END,
        "selection_metric": "累计收益率",
        "evaluated_threshold_pairs": evaluated,
        "baseline_thresholds": {
            "peak_threshold": float(base_model["peak_threshold"]),
            "trough_threshold": float(base_model["trough_threshold"]),
        },
        "optimized_thresholds": {
            "peak_threshold": best_fast["peak_threshold"],
            "trough_threshold": best_fast["trough_threshold"],
        },
        "baseline_bt_result": baseline_bt,
        "optimized_bt_result": best_bt,
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with output_model_path.open("wb") as f:
        pickle.dump(optimized_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "model_path": str(output_model_path),
        "base_model_path": str(base_model_path),
        "data_cache_path": str(data_cache_path),
        "pred_start": PRED_START,
        "pred_end": PRED_END,
        "selection_metric": "累计收益率",
        "evaluated_threshold_pairs": evaluated,
        "baseline_fast": baseline_fast,
        "best_fast": best_fast,
        "baseline_bt_result": baseline_bt,
        "best_bt_result": best_bt,
        "baseline_trade_count": int(len(baseline_trades)),
        "best_trade_count": int(len(best_trades)),
        "best_trades": best_trades.to_dict(orient="records"),
        "improvement": {
            "累计收益率": float(best_bt["累计收益率"] - baseline_bt["累计收益率"]),
            "超额收益率": float(best_bt["超额收益率"] - baseline_bt["超额收益率"]),
        },
        "elapsed_seconds": time.time() - start,
    }
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
