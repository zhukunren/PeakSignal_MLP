import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.backtest import backtest_results
from src.models import set_seed
from src.predict import get_trade_signal
from src.train import train_model


N = 20
MIXTURE_DEPTH = 1
CLASSIFIER_NAME = "MLP"
OVERSAMPLE_METHOD = "SMOTE"
SEED_BASE = 7300
ROUND_NO = 8
SEED = SEED_BASE + ROUND_NO

N_BUY = 10
N_SELL = 10
ENABLE_CHASE = False
ENABLE_STOP_LOSS = False

DATA_CACHE_PATH = Path("fixed_feature_combo_cache") / "prepared_data.pkl"
TARGET_RESULT_PATH = Path("fixed_feature_cached_combo_result.json")
MODEL_PATH = Path("base_98pct_round008_model.pkl")
REPORT_PATH = Path("base_98pct_round008_model_report.json")


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


def predict_label_arrays(data_preprocessed, model, scaler, selector, selected_features, threshold):
    missing_features = [f for f in selected_features if f not in data_preprocessed.columns]
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


def evaluate_model_combo(data_cache, peak_bundle, trough_bundle):
    (
        peak_model,
        peak_scaler,
        peak_selector,
        peak_selected_features,
        peak_threshold,
    ) = peak_bundle
    (
        trough_model,
        trough_scaler,
        trough_selector,
        trough_selected_features,
        trough_threshold,
    ) = trough_bundle

    pred_preprocessed = data_cache["pred_preprocessed"]
    mask_values = data_cache["mask_values"]
    combo_df = data_cache["base_backtest_df"].copy()

    peak_probas, peak_preds = predict_label_arrays(
        pred_preprocessed,
        peak_model,
        peak_scaler,
        peak_selector,
        peak_selected_features,
        peak_threshold,
    )
    trough_probas, trough_preds = predict_label_arrays(
        pred_preprocessed,
        trough_model,
        trough_scaler,
        trough_selector,
        trough_selected_features,
        trough_threshold,
    )

    combo_df["Peak_Probability"] = peak_probas[mask_values]
    combo_df["Peak_Prediction"] = peak_preds[mask_values]
    combo_df["Trough_Probability"] = trough_probas[mask_values]
    combo_df["Trough_Prediction"] = trough_preds[mask_values]
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
    return bt_result, trades_df


def main():
    start = time.time()
    if not DATA_CACHE_PATH.exists():
        raise FileNotFoundError(f"缺少训练缓存: {DATA_CACHE_PATH}")
    if not TARGET_RESULT_PATH.exists():
        raise FileNotFoundError(f"缺少目标结果文件: {TARGET_RESULT_PATH}")

    with TARGET_RESULT_PATH.open("r", encoding="utf-8") as f:
        target_result = json.load(f)
    target = target_result["best_by_excess"]

    with DATA_CACHE_PATH.open("rb") as f:
        data_cache = pickle.load(f)

    set_seed(SEED)
    train_df = data_cache["train_df"]
    all_features = data_cache["all_features"]

    (
        peak_model,
        peak_scaler,
        peak_selector,
        peak_selected_features,
        all_features_peak,
        peak_best_score,
        peak_metrics,
        peak_threshold,
        trough_model,
        trough_scaler,
        trough_selector,
        trough_selected_features,
        all_features_trough,
        trough_best_score,
        trough_metrics,
        trough_threshold,
    ) = train_model(
        train_df,
        N,
        all_features,
        CLASSIFIER_NAME,
        MIXTURE_DEPTH,
        "auto",
        OVERSAMPLE_METHOD,
    )

    bt_result, trades_df = evaluate_model_combo(
        data_cache,
        (
            peak_model,
            peak_scaler,
            peak_selector,
            peak_selected_features,
            peak_threshold,
        ),
        (
            trough_model,
            trough_scaler,
            trough_selector,
            trough_selected_features,
            trough_threshold,
        ),
    )

    achieved_cumulative = float(bt_result.get("累计收益率", float("-inf")))
    achieved_excess = float(bt_result.get("超额收益率", float("-inf")))
    target_cumulative = float(target["累计收益率"])
    target_excess = float(target["超额收益率"])
    reached_cumulative = achieved_cumulative >= target_cumulative
    reached_excess = achieved_excess >= target_excess

    model_package = {
        "peak_model": peak_model,
        "peak_scaler": peak_scaler,
        "peak_selector": peak_selector,
        "peak_selected_features": peak_selected_features,
        "peak_threshold": peak_threshold,
        "trough_model": trough_model,
        "trough_scaler": trough_scaler,
        "trough_selector": trough_selector,
        "trough_selected_features": trough_selected_features,
        "trough_threshold": trough_threshold,
        "N": N,
        "mixture_depth": MIXTURE_DEPTH,
        "classifier_name": CLASSIFIER_NAME,
        "oversample_method": OVERSAMPLE_METHOD,
        "round_no": ROUND_NO,
        "seed": SEED,
        "source_cache": str(DATA_CACHE_PATH),
        "target_result": target,
        "bt_result": bt_result,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "model_path": str(MODEL_PATH),
        "round_no": ROUND_NO,
        "seed": SEED,
        "target_cumulative_return": target_cumulative,
        "target_excess_return": target_excess,
        "achieved_cumulative_return": achieved_cumulative,
        "achieved_excess_return": achieved_excess,
        "reached_target_cumulative": reached_cumulative,
        "reached_target_excess": reached_excess,
        "bt_result": bt_result,
        "trade_count": int(len(trades_df)),
        "elapsed_seconds": time.time() - start,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
