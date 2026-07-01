import pandas as pd

from ml_trader.features.feature_groups import get_label_feature_candidates
from ml_trader.models.trainer import select_features_for_label


def test_label_feature_candidates_are_different():
    available = [
        "Price_Position_20",
        "Upper_Shadow_Pct",
        "Drawdown_20",
        "Lower_Shadow_Pct",
        "RSI_Signal",
    ]

    peak_candidates = get_label_feature_candidates("Peak", available)
    trough_candidates = get_label_feature_candidates("Trough", available)

    assert "Price_Position_20" in peak_candidates
    assert "Upper_Shadow_Pct" in peak_candidates
    assert "Drawdown_20" in trough_candidates
    assert "Lower_Shadow_Pct" in trough_candidates
    assert peak_candidates != trough_candidates


def test_select_features_for_label_prefers_label_specific_groups():
    X = pd.DataFrame({
        "Price_Position_20": [0.1, 0.2, 0.9, 0.8, 0.2, 0.85],
        "Upper_Shadow_Pct": [0.0, 0.1, 0.5, 0.4, 0.1, 0.45],
        "Drawdown_20": [-0.8, -0.1, -0.2, -0.7, -0.1, -0.2],
        "Lower_Shadow_Pct": [0.6, 0.1, 0.1, 0.5, 0.1, 0.1],
        "Return_20": [-0.5, 0.1, 0.2, -0.4, 0.1, 0.2],
        "Volume_Ratio_20": [1.2, 0.8, 1.0, 1.1, 0.9, 1.0],
    })
    y_peak = pd.Series([0, 0, 1, 1, 0, 1])
    y_trough = pd.Series([1, 0, 0, 1, 0, 0])
    config = {
        "default_mode": "hybrid",
        "method": "pearson",
        "peak": {
            "mode": "hybrid",
            "max_features": 4,
            "min_preferred_features": 2,
            "preferred_groups": ["peak", "common"],
        },
        "trough": {
            "mode": "hybrid",
            "max_features": 4,
            "min_preferred_features": 2,
            "preferred_groups": ["trough", "common"],
        },
    }

    peak_features = select_features_for_label(
        X, y_peak, "Peak", list(X.columns), "auto", selection_config=config
    )
    trough_features = select_features_for_label(
        X, y_trough, "Trough", list(X.columns), "auto", selection_config=config
    )

    assert len(peak_features) == 4
    assert len(trough_features) == 4
    assert any(feature in peak_features for feature in ["Price_Position_20", "Upper_Shadow_Pct"])
    assert any(feature in trough_features for feature in ["Drawdown_20", "Lower_Shadow_Pct"])
    assert peak_features != trough_features
