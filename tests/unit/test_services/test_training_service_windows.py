import pandas as pd

from app.services.training_service import TrainingService


def test_filter_training_window_resolves_latest_confirmed_by_rows():
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "TradeDate": dates.strftime("%Y%m%d"),
        "Close": range(10),
    })

    filtered, start_ts, end_ts = TrainingService.filter_training_window(
        df,
        "2021-01-03",
        "latest_confirmed",
        label_confirmation_window=2,
    )

    assert start_ts == pd.Timestamp("2021-01-03")
    assert end_ts == pd.Timestamp("2021-01-08")
    assert filtered["TradeDate"].min() == "20210103"
    assert filtered["TradeDate"].max() == "20210108"
