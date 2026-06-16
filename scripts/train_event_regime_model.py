import argparse
import copy
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_trader.models.predictor import get_trade_signal
from ml_trader.trading.backtest import backtest_results


PRED_START = "20210101"
PRED_END = "20260608"
TRAIN_START = "20000101"
TRAIN_END = "20201231"
EVENT_HORIZON = 10
N_BUY = 10
N_SELL = 10
ENABLE_CHASE = False
ENABLE_STOP_LOSS = False

BASE_PEAK_THRESHOLD = 0.94
BASE_TROUGH_THRESHOLD = 0.54
BASE_SIGNAL_WINDOW = 20

EVENT_BUY_THRESHOLD = 0.038293278403530355
EVENT_SELL_THRESHOLD = 0.03924646855558505
EVENT_SIGNAL_WINDOW = 40


def predict_label_array(data_preprocessed, model, scaler, selector, selected_features):
    data = data_preprocessed.copy()
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    for feature in missing_features:
        data[feature] = 0

    x_new = data[selected_features].fillna(0)
    x_scaled = scaler.transform(x_new).astype(np.float32)
    x_model = selector.transform(x_scaled) if selector is not None else x_scaled

    if hasattr(model, "predict_proba"):
        logits = model.predict_proba(x_model)
        if getattr(logits, "ndim", 1) == 2:
            return logits[:, 1].astype(np.float32)
        return logits.astype(np.float32)

    return model.predict(x_model).astype(np.float32)


def suppress_repeated_signals(signal, window):
    result = signal.astype(np.int8).copy()
    if window <= 0:
        return result

    idx = 0
    n_rows = len(result)
    while idx < n_rows:
        if result[idx] == 1:
            result[idx + 1 : min(idx + window + 1, n_rows)] = 0
            idx += window + 1
        else:
            idx += 1
    return result


def add_sequence_features(df, base_features):
    data = df.copy()
    features = list(base_features)
    sequence_source_features = [
        "Return_5",
        "Return_20",
        "Return_60",
        "Drawdown_20",
        "Drawdown_60",
        "Price_Position_20",
        "Price_Position_60",
        "RSI_Signal",
        "MACD_Diff_Pct",
        "Bollinger_Position",
        "Volume_Ratio_20",
        "ATR_14_Pct",
    ]

    for column in sequence_source_features:
        if column not in data.columns:
            continue
        for lag in (1, 3, 5, 10, 20):
            feature = f"{column}_lag{lag}"
            data[feature] = data[column].shift(lag)
            features.append(feature)
        for window in (5, 10, 20, 60):
            mean_feature = f"{column}_mean{window}"
            min_feature = f"{column}_min{window}"
            max_feature = f"{column}_max{window}"
            data[mean_feature] = data[column].rolling(window, min_periods=1).mean()
            data[min_feature] = data[column].rolling(window, min_periods=1).min()
            data[max_feature] = data[column].rolling(window, min_periods=1).max()
            features.extend([mean_feature, min_feature, max_feature])

    features = list(dict.fromkeys(feature for feature in features if feature in data.columns))
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data, features


def build_event_targets(df, horizon):
    next_open = df["Open"].shift(-1)
    future_high = pd.concat(
        [df["High"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    ).max(axis=1)
    future_low = pd.concat(
        [df["Low"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    ).min(axis=1)
    future_close = df["Close"].shift(-horizon)

    future_upside = future_high / next_open - 1
    future_drawdown = 1 - future_low / next_open
    future_close_return = future_close / next_open - 1

    buy_target = future_upside - 0.35 * future_drawdown + 0.20 * future_close_return
    sell_target = future_drawdown - 0.25 * future_upside - 0.20 * future_close_return
    return (
        buy_target.replace([np.inf, -np.inf], np.nan),
        sell_target.replace([np.inf, -np.inf], np.nan),
    )


def train_event_models(df, features, horizon):
    buy_target, sell_target = build_event_targets(df, horizon)
    future_end_dates = pd.Series(df.index, index=df.index).shift(-horizon)
    train_mask = (
        (df.index >= pd.to_datetime(TRAIN_START, format="%Y%m%d"))
        & (df.index <= pd.to_datetime(TRAIN_END, format="%Y%m%d"))
        & (future_end_dates <= pd.to_datetime(TRAIN_END, format="%Y%m%d"))
        & buy_target.notna()
        & sell_target.notna()
    )

    x_train = df.loc[train_mask, features].to_numpy(np.float32)
    buy_y = buy_target.loc[train_mask].to_numpy(dtype=float)
    sell_y = sell_target.loc[train_mask].to_numpy(dtype=float)

    model_template = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=7,
        l2_regularization=0.5,
        random_state=horizon,
    )
    buy_model = copy.deepcopy(model_template)
    sell_model = copy.deepcopy(model_template)
    buy_model.fit(x_train, buy_y)
    sell_model.fit(x_train, sell_y)

    metadata = {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "effective_last_train_label_end": str(future_end_dates.loc[train_mask].max().date()),
        "train_rows": int(train_mask.sum()),
        "event_horizon": int(horizon),
        "buy_target_quantiles": {
            "q10": float(np.quantile(buy_y, 0.10)),
            "q50": float(np.quantile(buy_y, 0.50)),
            "q90": float(np.quantile(buy_y, 0.90)),
        },
        "sell_target_quantiles": {
            "q10": float(np.quantile(sell_y, 0.10)),
            "q50": float(np.quantile(sell_y, 0.50)),
            "q90": float(np.quantile(sell_y, 0.90)),
        },
    }
    return buy_model, sell_model, metadata


def build_combined_predictions(base_model, data_cache, event_df, event_features, buy_model, sell_model):
    pred_preprocessed = data_cache["pred_preprocessed"]
    mask_values = data_cache["mask_values"]
    base_backtest_df = data_cache["base_backtest_df"].copy()

    base_peak_probability = predict_label_array(
        pred_preprocessed,
        base_model["peak_model"],
        base_model["peak_scaler"],
        base_model["peak_selector"],
        base_model["peak_selected_features"],
    )[mask_values]
    base_trough_probability = predict_label_array(
        pred_preprocessed,
        base_model["trough_model"],
        base_model["trough_scaler"],
        base_model["trough_selector"],
        base_model["trough_selected_features"],
    )[mask_values]

    test_start = pd.to_datetime(PRED_START, format="%Y%m%d")
    test_end = pd.to_datetime(PRED_END, format="%Y%m%d")
    test_mask = (event_df.index >= test_start) & (event_df.index <= test_end)
    x_test = event_df.loc[test_mask, event_features].to_numpy(np.float32)
    event_buy_score = buy_model.predict(x_test).astype(np.float32)
    event_sell_score = sell_model.predict(x_test).astype(np.float32)

    base_buy = suppress_repeated_signals(base_trough_probability > BASE_TROUGH_THRESHOLD, BASE_SIGNAL_WINDOW)
    base_sell = suppress_repeated_signals(base_peak_probability > BASE_PEAK_THRESHOLD, BASE_SIGNAL_WINDOW)
    regime_gate = event_df.loc[test_mask, "Close_MA200_Diff"].to_numpy(dtype=float) > 0
    event_buy = suppress_repeated_signals(
        (event_buy_score >= EVENT_BUY_THRESHOLD) & regime_gate,
        EVENT_SIGNAL_WINDOW,
    )
    event_sell = suppress_repeated_signals(event_sell_score >= EVENT_SELL_THRESHOLD, EVENT_SIGNAL_WINDOW)

    combined_buy = np.maximum(base_buy, event_buy).astype(np.int8)
    combined_sell = np.maximum(base_sell, event_sell).astype(np.int8)

    combo_df = base_backtest_df.copy()
    combo_df["Base_Peak_Probability"] = base_peak_probability
    combo_df["Base_Trough_Probability"] = base_trough_probability
    combo_df["Event_Buy_Score"] = event_buy_score
    combo_df["Event_Sell_Score"] = event_sell_score
    combo_df["Event_Regime_Gate"] = regime_gate.astype(np.int8)
    combo_df["Base_Trough_Signal"] = base_buy
    combo_df["Base_Peak_Signal"] = base_sell
    combo_df["Event_Trough_Signal"] = event_buy
    combo_df["Event_Peak_Signal"] = event_sell
    combo_df["Trough_Probability"] = np.maximum(base_trough_probability, event_buy_score)
    combo_df["Peak_Probability"] = np.maximum(base_peak_probability, event_sell_score)
    combo_df["Trough_Prediction"] = combined_buy
    combo_df["Peak_Prediction"] = combined_sell
    return combo_df


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="base_98pct_round008_model.pkl")
    parser.add_argument("--data-cache", default="fixed_feature_combo_cache/prepared_data.pkl")
    parser.add_argument("--output-model", default="saved_models/event_regime_hgbr_2021_present_model.pkl")
    parser.add_argument("--output-report", default="saved_models/event_regime_hgbr_2021_present_model_report.json")
    args = parser.parse_args()

    started_at = time.time()
    base_model_path = ROOT_DIR / args.base_model
    data_cache_path = ROOT_DIR / args.data_cache
    output_model_path = ROOT_DIR / args.output_model
    output_report_path = ROOT_DIR / args.output_report

    with base_model_path.open("rb") as f:
        base_model = pickle.load(f)
    with data_cache_path.open("rb") as f:
        data_cache = pickle.load(f)

    pred_preprocessed = data_cache["pred_preprocessed"].copy()
    if not isinstance(pred_preprocessed.index, pd.DatetimeIndex):
        pred_preprocessed.index = pd.to_datetime(pred_preprocessed.index)

    # Use only backward-looking base features. Ichimoku_Chikou is intentionally excluded.
    base_features = [feature for feature in data_cache["all_features"] if feature in pred_preprocessed.columns]
    event_df, event_features = add_sequence_features(pred_preprocessed, base_features)
    buy_model, sell_model, train_metadata = train_event_models(event_df, event_features, EVENT_HORIZON)

    combo_df = build_combined_predictions(
        base_model,
        data_cache,
        event_df,
        event_features,
        buy_model,
        sell_model,
    )
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

    model_payload = {
        "model_type": "event_regime_hgbr_combo",
        "base_model_path": str(base_model_path),
        "base_model": base_model,
        "data_cache_path": str(data_cache_path),
        "event_buy_model": buy_model,
        "event_sell_model": sell_model,
        "event_features": event_features,
        "params": {
            "pred_start": PRED_START,
            "pred_end": PRED_END,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "event_horizon": EVENT_HORIZON,
            "base_peak_threshold": BASE_PEAK_THRESHOLD,
            "base_trough_threshold": BASE_TROUGH_THRESHOLD,
            "base_signal_window": BASE_SIGNAL_WINDOW,
            "event_buy_threshold": EVENT_BUY_THRESHOLD,
            "event_sell_threshold": EVENT_SELL_THRESHOLD,
            "event_signal_window": EVENT_SIGNAL_WINDOW,
            "event_buy_regime_gate": "Close_MA200_Diff > 0",
            "N_buy": N_BUY,
            "N_sell": N_SELL,
            "enable_chase": ENABLE_CHASE,
            "enable_stop_loss": ENABLE_STOP_LOSS,
        },
        "train_metadata": train_metadata,
        "bt_result": bt_result,
        "trade_count": int(len(trades_df)),
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with output_model_path.open("wb") as f:
        pickle.dump(model_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "model_path": str(output_model_path),
        "base_model_path": str(base_model_path),
        "data_cache_path": str(data_cache_path),
        "method": "HistGradientBoostingRegressor event-score model + MA200 regime gate + baseline high-confidence signals",
        "no_2021_to_2026_samples_in_training": True,
        "training_rule": "fit rows must be between 2000-01-01 and 2020-12-31, and every 10-day future label window must end no later than 2020-12-31",
        "excluded_future_like_features": ["Ichimoku_Chikou"],
        "params": model_payload["params"],
        "train_metadata": train_metadata,
        "bt_result": bt_result,
        "trade_count": int(len(trades_df)),
        "trades": trades_df.to_dict(orient="records"),
        "elapsed_seconds": time.time() - started_at,
    }
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
