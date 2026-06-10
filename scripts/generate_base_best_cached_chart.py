import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml_trader.trading.backtest import backtest_results
from ml_trader.visualization.plots import plot_candlestick
from ml_trader.models.predictor import get_trade_signal


N_BUY = 10
N_SELL = 10
ENABLE_CHASE = False
ENABLE_STOP_LOSS = False
PRED_START = "20210101"
PRED_END = "20260608"

CACHE_DIR = Path("fixed_feature_combo_cache")
DATA_CACHE_PATH = CACHE_DIR / "prepared_data.pkl"
PRED_DIR = CACHE_DIR / "predictions"
RESULT_PATH = Path("fixed_feature_cached_combo_result.json")
HTML_PATH = Path("best_combo_prediction_kline.html")
JSON_PATH = Path("best_combo_prediction_result.json")


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


def load_round_prediction(round_no):
    path = PRED_DIR / f"round_{round_no:03d}.npz"
    with np.load(path) as arr:
        return {
            "peak_probas": arr["peak_probas"],
            "peak_preds": arr["peak_preds"],
            "trough_probas": arr["trough_probas"],
            "trough_preds": arr["trough_preds"],
        }


def mark_trades(combo_df, trades_df):
    combo_df = combo_df.copy()
    combo_df["trade"] = None
    if trades_df.empty:
        return combo_df
    if not isinstance(combo_df.index, pd.DatetimeIndex):
        combo_df.index = pd.to_datetime(combo_df.index)
    for _, trade in trades_df.iterrows():
        entry_date = pd.to_datetime(trade.get("entry_date"))
        exit_date = pd.to_datetime(trade.get("exit_date"))
        if entry_date in combo_df.index:
            combo_df.loc[entry_date, "trade"] = "buy"
        if exit_date in combo_df.index:
            combo_df.loc[exit_date, "trade"] = "sell"
    return combo_df


def main():
    with RESULT_PATH.open("r", encoding="utf-8") as f:
        result = json.load(f)
    best = result["best_by_excess"]
    peak_round = int(best["peak_model_index"])
    trough_round = int(best["trough_model_index"])

    with DATA_CACHE_PATH.open("rb") as f:
        data_cache = pickle.load(f)

    peak_pred = load_round_prediction(peak_round)
    trough_pred = load_round_prediction(trough_round)
    mask_values = data_cache["mask_values"]
    combo_df = data_cache["base_backtest_df"].copy()
    combo_df["Peak_Probability"] = peak_pred["peak_probas"][mask_values]
    combo_df["Peak_Prediction"] = peak_pred["peak_preds"][mask_values]
    combo_df["Trough_Probability"] = trough_pred["trough_probas"][mask_values]
    combo_df["Trough_Prediction"] = trough_pred["trough_preds"][mask_values]
    combo_df = suppress_repeated_signals(combo_df)

    signal_df = get_trade_signal(combo_df)
    bt, trades_df = backtest_results(
        combo_df,
        signal_df,
        N_BUY,
        N_SELL,
        ENABLE_CHASE,
        ENABLE_STOP_LOSS,
        initial_capital=1_000_000,
    )
    combo_df = mark_trades(combo_df, trades_df)
    if not isinstance(combo_df.index, pd.DatetimeIndex):
        combo_df.index = pd.to_datetime(combo_df.index)

    peaks_pred = combo_df[combo_df["Peak_Prediction"] == 1]
    troughs_pred = combo_df[combo_df["Trough_Prediction"] == 1]
    fig = plot_candlestick(
        combo_df.copy(),
        "000001.SH",
        pd.to_datetime(PRED_START, format="%Y%m%d"),
        pd.to_datetime(PRED_END, format="%Y%m%d"),
        peaks_pred,
        troughs_pred,
        prediction=True,
        bt_result=bt,
    )
    fig.write_html(HTML_PATH, include_plotlyjs="cdn", full_html=True)

    out = {
        "chart_html": str(HTML_PATH),
        "best_combo": best,
        "bt_result": bt,
        "predicted_peaks": int(len(peaks_pred)),
        "predicted_troughs": int(len(troughs_pred)),
        "trades_count": int(len(trades_df)),
        "trades": trades_df.to_dict(orient="records"),
    }
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
