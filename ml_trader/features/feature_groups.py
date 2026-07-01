"""Label-aware feature groups for Peak/Trough models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


FEATURE_GROUPS: Dict[str, List[str]] = {
    "common": [
        "Return_5",
        "Return_20",
        "Return_60",
        "Price_Position_20",
        "Price_Position_60",
        "Drawdown_20",
        "Drawdown_60",
        "Volatility_Ratio_20_60",
        "ATR_14_Pct",
        "RSI_Signal",
        "MACD_Diff_Pct",
        "Bollinger_Position",
        "Volume_Ratio_20",
        "Volume_Ratio_60",
        "Body_Pct",
        "HL_Range_Pct",
    ],
    "peak": [
        "Price_Position_20",
        "Price_Position_60",
        "Price_Position_120",
        "Price_Position_250",
        "New_High_20",
        "New_High_60",
        "High_From_Close_20",
        "High_From_Close_60",
        "Upper_Shadow_Pct",
        "Bollinger_Position",
        "RSI_Signal",
        "RSI_14_Z_120",
        "Return_20_Z_250",
        "Return_60_Z_250",
        "Volume_Ratio_20",
        "ATR_14_Ratio_60",
    ],
    "trough": [
        "Drawdown_20",
        "Drawdown_60",
        "Drawdown_120",
        "Drawdown_250",
        "Drawdown_60_Z_250",
        "New_Low_20",
        "New_Low_60",
        "Close_From_Low_20",
        "Close_From_Low_60",
        "Lower_Shadow_Pct",
        "Bollinger_Position",
        "RSI_Signal",
        "RSI_14_Z_120",
        "Volume_Ratio_20",
        "ATR_14_Ratio_60",
    ],
}


DEFAULT_LABEL_GROUPS = {
    "Peak": ["peak", "common"],
    "Trough": ["trough", "common"],
}


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_label_feature_candidates(
    label_column: str,
    available_features: Sequence[str],
    selection_config: Mapping | None = None,
) -> List[str]:
    """Return label-preferred features that are present in ``available_features``."""
    available = set(available_features)
    selection_config = selection_config or {}
    label_key = str(label_column).lower()
    label_config = selection_config.get(label_key, {})
    preferred_groups = label_config.get(
        "preferred_groups",
        DEFAULT_LABEL_GROUPS.get(label_column, ["common"]),
    )

    candidates = []
    for group_name in preferred_groups:
        candidates.extend(FEATURE_GROUPS.get(group_name, []))

    return [feature for feature in dedupe_preserve_order(candidates) if feature in available]


def get_label_selection_config(selection_config: Mapping | None, label_column: str) -> Mapping:
    selection_config = selection_config or {}
    label_key = str(label_column).lower()
    return selection_config.get(label_key, {})
