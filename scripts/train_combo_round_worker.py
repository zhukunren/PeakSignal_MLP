import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ml_trader.models.architectures import set_seed
from ml_trader.models.trainer import train_model


N = 20
MIXTURE_DEPTH = 1
CLASSIFIER_NAME = "MLP"
OVERSAMPLE_METHOD = "SMOTE"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    t0 = time.time()
    set_seed(args.seed)

    data_cache_path = Path(args.data_cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with data_cache_path.open("rb") as f:
        data_cache = pickle.load(f)

    train_df = data_cache["train_df"]
    all_features = data_cache["all_features"]
    pred_preprocessed = data_cache["pred_preprocessed"]
    mask_values = data_cache["mask_values"]

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

    round_name = f"round_{args.round:03d}"
    npz_path = output_dir / f"{round_name}.npz"
    meta_path = output_dir / f"{round_name}.json"
    tmp_npz_path = output_dir / f"{round_name}.tmp.npz"
    tmp_meta_path = output_dir / f"{round_name}.tmp.json"

    np.savez_compressed(
        tmp_npz_path,
        peak_probas=peak_probas,
        peak_preds=peak_preds,
        trough_probas=trough_probas,
        trough_preds=trough_preds,
    )

    meta = {
        "round": args.round,
        "seed": args.seed,
        "peak_score": float(peak_best_score),
        "peak_threshold": float(peak_threshold),
        "peak_signals": int(peak_preds[mask_values].sum()),
        "trough_score": float(trough_best_score),
        "trough_threshold": float(trough_threshold),
        "trough_signals": int(trough_preds[mask_values].sum()),
        "elapsed_seconds": time.time() - t0,
    }
    with tmp_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    tmp_npz_path.replace(npz_path)
    tmp_meta_path.replace(meta_path)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
