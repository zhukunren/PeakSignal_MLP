"""Signal quality metrics for Peak/Trough predictions."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "trough_event_recall": 0.30,
    "peak_event_recall": 0.25,
    "trough_signal_precision": 0.20,
    "peak_signal_precision": 0.15,
    "mid_zone_false_signal_rate": -0.25,
    "duplicate_signal_penalty": -0.10,
}


def _as_binary_array(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(len(df), dtype=np.int8)
    return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int).to_numpy()


def _indices(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(values == 1)


def _nearest_distance(index: int, event_indices: np.ndarray) -> int | None:
    if event_indices.size == 0:
        return None
    return int(np.min(np.abs(event_indices - index)))


def _event_windows(event_indices: Iterable[int], tolerance_days: int, n_rows: int) -> list[Tuple[int, int]]:
    windows = []
    for event_index in event_indices:
        start = max(0, int(event_index) - tolerance_days)
        end = min(n_rows - 1, int(event_index) + tolerance_days)
        windows.append((start, end))
    return windows


def _in_any_window(index: int, windows: list[Tuple[int, int]]) -> bool:
    return any(start <= index <= end for start, end in windows)


def evaluate_label_signal_quality(
    result_df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    tolerance_days: int = 5,
) -> Dict[str, float | None]:
    """Evaluate one label's event recall, signal precision, distance and duplicates."""
    if result_df is None or result_df.empty:
        return {
            "event_count": 0,
            "signal_count": 0,
            "event_recall": 0.0,
            "signal_precision": 0.0,
            "false_signal_rate": 0.0,
            "duplicate_signal_penalty": 0.0,
            "avg_distance": None,
        }

    true_values = _as_binary_array(result_df, true_col)
    pred_values = _as_binary_array(result_df, pred_col)
    event_indices = _indices(true_values)
    signal_indices = _indices(pred_values)
    windows = _event_windows(event_indices, tolerance_days, len(result_df))

    hit_distances = []
    duplicate_signals = 0
    for event_index, (start, end) in zip(event_indices, windows):
        signals_in_window = signal_indices[(signal_indices >= start) & (signal_indices <= end)]
        if signals_in_window.size > 0:
            hit_distances.append(int(np.min(np.abs(signals_in_window - event_index))))
            duplicate_signals += max(0, int(signals_in_window.size) - 1)

    valid_signals = sum(1 for idx in signal_indices if _in_any_window(int(idx), windows))
    false_signals = max(0, int(signal_indices.size) - valid_signals)

    event_count = int(event_indices.size)
    signal_count = int(signal_indices.size)
    event_recall = len(hit_distances) / event_count if event_count else 0.0
    signal_precision = valid_signals / signal_count if signal_count else 0.0
    false_signal_rate = false_signals / signal_count if signal_count else 0.0
    duplicate_signal_penalty = duplicate_signals / signal_count if signal_count else 0.0
    avg_distance = float(np.mean(hit_distances)) if hit_distances else None

    return {
        "event_count": event_count,
        "signal_count": signal_count,
        "event_recall": event_recall,
        "signal_precision": signal_precision,
        "false_signal_rate": false_signal_rate,
        "duplicate_signal_penalty": duplicate_signal_penalty,
        "avg_distance": avg_distance,
    }


def evaluate_peak_signal_quality(result_df: pd.DataFrame, tolerance_days: int = 5) -> Dict[str, float | None]:
    return evaluate_label_signal_quality(result_df, "Peak", "Peak_Prediction", tolerance_days)


def evaluate_trough_signal_quality(result_df: pd.DataFrame, tolerance_days: int = 5) -> Dict[str, float | None]:
    return evaluate_label_signal_quality(result_df, "Trough", "Trough_Prediction", tolerance_days)


def _bounded_return_score(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return (math.tanh(numeric * 3.0) + 1.0) / 2.0


def _bounded_sharpe_score(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return (math.tanh(numeric / 2.0) + 1.0) / 2.0


def _drawdown_penalty(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return min(abs(numeric), 1.0)


def evaluate_combo_quality(
    result_df: pd.DataFrame,
    bt_result: Dict,
    tolerance_days: int = 5,
    weights: Dict[str, float] | None = None,
) -> Dict[str, float | None]:
    """Return signal and backtest-aware quality metrics for a model combination."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    peak = evaluate_peak_signal_quality(result_df, tolerance_days)
    trough = evaluate_trough_signal_quality(result_df, tolerance_days)

    mid_zone_false_signal_rate = (
        peak["false_signal_rate"] * peak["signal_count"]
        + trough["false_signal_rate"] * trough["signal_count"]
    )
    total_signals = peak["signal_count"] + trough["signal_count"]
    mid_zone_false_signal_rate = mid_zone_false_signal_rate / total_signals if total_signals else 0.0

    duplicate_signal_penalty = (
        peak["duplicate_signal_penalty"] * peak["signal_count"]
        + trough["duplicate_signal_penalty"] * trough["signal_count"]
    )
    duplicate_signal_penalty = duplicate_signal_penalty / total_signals if total_signals else 0.0

    raw_metrics = {
        "peak_event_recall": peak["event_recall"],
        "trough_event_recall": trough["event_recall"],
        "peak_signal_precision": peak["signal_precision"],
        "trough_signal_precision": trough["signal_precision"],
        "mid_zone_false_signal_rate": mid_zone_false_signal_rate,
        "duplicate_signal_penalty": duplicate_signal_penalty,
    }
    signal_score = sum(float(raw_metrics[key]) * weight for key, weight in weights.items())

    excess_score = _bounded_return_score(bt_result.get("超额收益率") if bt_result else None)
    sharpe_score = _bounded_sharpe_score(bt_result.get("年化夏普比率") if bt_result else None)
    drawdown_penalty = _drawdown_penalty(bt_result.get("最大回撤") if bt_result else None)
    final_score = 0.60 * signal_score + 0.25 * excess_score + 0.10 * sharpe_score - 0.20 * drawdown_penalty

    return {
        **raw_metrics,
        "peak_event_count": peak["event_count"],
        "trough_event_count": trough["event_count"],
        "peak_signal_count": peak["signal_count"],
        "trough_signal_count": trough["signal_count"],
        "avg_peak_distance": peak["avg_distance"],
        "avg_trough_distance": trough["avg_distance"],
        "signal_score": signal_score,
        "excess_score": excess_score,
        "sharpe_score": sharpe_score,
        "drawdown_penalty": drawdown_penalty,
        "final_score": final_score,
    }
