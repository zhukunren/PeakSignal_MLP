import pandas as pd

from ml_trader.evaluation.signal_quality import (
    evaluate_combo_quality,
    evaluate_peak_signal_quality,
    evaluate_trough_signal_quality,
)


def test_peak_signal_quality_counts_nearby_hit_and_duplicate_penalty():
    df = pd.DataFrame({
        "Peak": [0, 0, 0, 0, 0, 1, 0, 0],
        "Peak_Prediction": [0, 0, 0, 0, 1, 1, 1, 0],
    })

    metrics = evaluate_peak_signal_quality(df, tolerance_days=1)

    assert metrics["event_recall"] == 1.0
    assert metrics["signal_precision"] == 1.0
    assert metrics["duplicate_signal_penalty"] > 0
    assert metrics["avg_distance"] == 0.0


def test_trough_signal_quality_penalizes_mid_zone_false_signal():
    df = pd.DataFrame({
        "Trough": [0, 0, 1, 0, 0, 0, 0, 0],
        "Trough_Prediction": [0, 0, 0, 0, 0, 0, 1, 0],
    })

    metrics = evaluate_trough_signal_quality(df, tolerance_days=1)

    assert metrics["event_recall"] == 0.0
    assert metrics["signal_precision"] == 0.0
    assert metrics["false_signal_rate"] == 1.0


def test_combo_quality_prefers_nearby_signals_over_false_signals():
    good = pd.DataFrame({
        "Peak": [0, 0, 0, 0, 1, 0],
        "Trough": [0, 1, 0, 0, 0, 0],
        "Peak_Prediction": [0, 0, 0, 0, 1, 0],
        "Trough_Prediction": [0, 1, 0, 0, 0, 0],
    })
    bad = pd.DataFrame({
        "Peak": [0, 0, 0, 0, 1, 0],
        "Trough": [0, 1, 0, 0, 0, 0],
        "Peak_Prediction": [1, 0, 0, 0, 0, 0],
        "Trough_Prediction": [0, 0, 0, 0, 0, 1],
    })
    bt = {"超额收益率": 0.1, "年化夏普比率": 1.0, "最大回撤": -0.1}

    good_metrics = evaluate_combo_quality(good, bt, tolerance_days=1)
    bad_metrics = evaluate_combo_quality(bad, bt, tolerance_days=1)

    assert good_metrics["signal_score"] > bad_metrics["signal_score"]
    assert good_metrics["final_score"] > bad_metrics["final_score"]
