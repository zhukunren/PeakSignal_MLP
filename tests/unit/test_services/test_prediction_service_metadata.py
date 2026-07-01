import pandas as pd

from app.services import prediction_service as prediction_service_module
from app.services.prediction_service import PredictionService


def _candidate(label, train_window, seed):
    return {
        "label": label,
        "model": object(),
        "scaler": object(),
        "selector": None,
        "selected_features": [],
        "threshold": 0.5,
        "train_window": train_window,
        "train_start": "2016-01-01",
        "train_end": "2026-05-29",
        "seed": seed,
        "round": 8,
    }


def test_search_best_combination_returns_candidate_origin_metadata(monkeypatch):
    eval_modes = []

    def fake_predict_new_data(*args, **kwargs):
        eval_modes.append(kwargs.get("eval_mode"))
        result_df = pd.DataFrame({
            "Peak": [0, 0, 1, 0],
            "Trough": [0, 1, 0, 0],
            "Peak_Prediction": [0, 0, 1, 0],
            "Trough_Prediction": [0, 1, 0, 0],
        })
        bt_result = {
            "超额收益率": 0.12,
            "年化夏普比率": 1.1,
            "最大回撤": -0.08,
        }
        return result_df, bt_result, pd.DataFrame()

    monkeypatch.setattr(prediction_service_module, "predict_new_data", fake_predict_new_data)

    service = PredictionService()
    best_models, best_excess, _, bt_result = service.search_best_combination(
        [_candidate("Peak", "recent_2016_latest", 7308)],
        [_candidate("Trough", "mid_2010_latest", 7309)],
        pd.DataFrame({"Close": [1, 2, 3, 4]}),
        N=20,
        mixture_depth=1,
        pred_start="20210101",
        pred_end="20210104",
        selection_metric="quality",
    )

    assert eval_modes == [True, False]
    assert best_excess == 0.12
    assert best_models["peak_train_window"] == "recent_2016_latest"
    assert best_models["trough_train_window"] == "mid_2010_latest"
    assert bt_result["组合筛选指标"] == "quality"
    assert bt_result["Peak模型训练窗口"] == "recent_2016_latest"
    assert bt_result["Peak模型seed"] == 7308
    assert bt_result["Trough模型训练窗口"] == "mid_2010_latest"
    assert bt_result["Trough模型seed"] == 7309
