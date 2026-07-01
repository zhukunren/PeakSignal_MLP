# predict.py
import numpy as np
import torch
from ml_trader.data.preprocessor import preprocess_data
from skorch import NeuralNetClassifier
from ml_trader.trading.backtest import backtest_results
from ml_trader.models.architectures import  TransformerClassifier
import pandas as pd
import pickle
from pathlib import Path
from ml_trader.logging_config import get_logger


logger = get_logger(__name__)

#绘图函数

# ============== 预测新数据的函数 (修改后返回数据与回测结果) ==============
def merge_trades(data_preprocessed, trades_df):
    """
    合并交易数据并保持原始索引，确保在合并后日期列 'date' 与交易信号一致。
    """
    # 保存原始索引
    original_index = data_preprocessed.index

    # 合并卖出日期，确保 exit_date 对齐到 data_preprocessed['date']
    data_preprocessed = pd.merge(
        data_preprocessed, 
        trades_df[['exit_date']],  # 选择 trades_df 中的 'exit_date'
        left_on='date',            # 使用 data_preprocessed 中的 'date' 列进行合并
        right_on='exit_date',      # 使用 trades_df 中的 'exit_date' 列进行合并
        how='left'                 # 使用左连接，保留 data_preprocessed 中所有行
    )

    # 设置 trade 为 'sell' 当 exit_date 非空时
    data_preprocessed['trade'] = np.where(data_preprocessed['exit_date'].notna(), 'sell', data_preprocessed['trade'])

    # 合并 entry_date
    data_preprocessed = pd.merge(
        data_preprocessed, 
        trades_df[['entry_date']],  # 选择 trades_df 中的 'entry_date'
        left_on='date',                           # 使用 data_preprocessed 中的 'date' 列进行合并
        right_on='entry_date',                    # 使用 trades_df 中的 'entry_date' 列进行合并
        how='left'                                # 使用左连接，保留 data_preprocessed 中所有行
    )

    # 设置 trade 为 'buy' 当 entry_date 非空时
    data_preprocessed['trade'] = np.where(data_preprocessed['entry_date'].notna(), 'buy', data_preprocessed['trade'])

    # 删除重复日期
    data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
    logger.debug("Merged trades:\n%s", data_preprocessed['trade'])
    
    # 恢复原始索引
    data_preprocessed.index = original_index

    return data_preprocessed


def _predict_probability_array(data_preprocessed, model, scaler, selector, selected_features):
    data = data_preprocessed.copy()
    for feature in selected_features:
        if feature not in data.columns:
            data[feature] = 0

    x_new = data[selected_features].fillna(0)
    x_scaled = scaler.transform(x_new).astype(np.float32) if scaler is not None else x_new.to_numpy(np.float32)
    x_model = selector.transform(x_scaled) if selector is not None else x_scaled

    if hasattr(model, "predict_proba"):
        logits = model.predict_proba(x_model)
        if getattr(logits, "ndim", 1) == 2:
            return logits[:, 1].astype(np.float32)
        return logits.astype(np.float32)

    return model.predict(x_model).astype(np.float32)


def _suppress_repeated_signal_array(signal, window):
    result = np.asarray(signal).astype(np.int8).copy()
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


def _add_event_sequence_features(df, feature_names):
    data = df.copy()
    features = list(feature_names)
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

    generated = {}
    for column in sequence_source_features:
        if column not in data.columns:
            continue
        for lag in (1, 3, 5, 10, 20):
            feature = f"{column}_lag{lag}"
            generated[feature] = data[column].shift(lag)
            features.append(feature)
        for window in (5, 10, 20, 60):
            mean_feature = f"{column}_mean{window}"
            min_feature = f"{column}_min{window}"
            max_feature = f"{column}_max{window}"
            rolling = data[column].rolling(window, min_periods=1)
            generated[mean_feature] = rolling.mean()
            generated[min_feature] = rolling.min()
            generated[max_feature] = rolling.max()
            features.extend([mean_feature, min_feature, max_feature])

    if generated:
        data = pd.concat([data, pd.DataFrame(generated, index=data.index)], axis=1)

    features = list(dict.fromkeys(feature for feature in features if feature in data.columns))
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data, features


def _load_event_base_model(event_model_package):
    if "base_model" in event_model_package:
        return event_model_package["base_model"]

    candidates = []
    base_model_path = event_model_package.get("base_model_path")
    if base_model_path:
        candidates.append(Path(base_model_path))
        candidates.append(Path.cwd() / base_model_path)
    candidates.append(Path.cwd() / "base_98pct_round008_model.pkl")

    for candidate in candidates:
        try:
            if candidate.exists():
                with candidate.open("rb") as f:
                    return pickle.load(f)
        except OSError:
            continue

    raise FileNotFoundError("事件组合模型需要 base_98pct_round008_model.pkl，当前路径未找到。")


def _parse_trade_dates(trade_date_series):
    return pd.to_datetime(
        trade_date_series.astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    )


def predict_event_regime_model_data(
    new_df,
    event_model_package,
    eval_mode=False,
    N_buy=None,
    N_sell=None,
    enable_chase=False,
    enable_stop_loss=False,
    enable_change_signal=False,
    N_newhigh=60,
    backtest_start_date=None,
    backtest_end_date=None,
):
    """
    Predict with the event-regime combo model saved by scripts/train_event_regime_model.py.

    The uploaded model is not a legacy Peak/Trough model. It combines:
    - the saved base Peak/Trough model probabilities;
    - event buy/sell regressors;
    - a regime gate for event buys.
    """
    if event_model_package.get("model_type") != "event_regime_hgbr_combo":
        raise ValueError("不是 event_regime_hgbr_combo 模型。")

    base_model = _load_event_base_model(event_model_package)
    params = event_model_package.get("params", {})
    N = base_model.get("N", 20)
    mixture_depth = base_model.get("mixture_depth", 1)

    data_preprocessed, all_features = preprocess_data(
        new_df,
        N,
        mixture_depth=mixture_depth,
        mark_labels=eval_mode,
    )
    if not isinstance(data_preprocessed.index, pd.DatetimeIndex):
        data_preprocessed.index = pd.to_datetime(data_preprocessed.index)

    base_peak_probability = _predict_probability_array(
        data_preprocessed,
        base_model["peak_model"],
        base_model["peak_scaler"],
        base_model["peak_selector"],
        base_model["peak_selected_features"],
    )
    base_trough_probability = _predict_probability_array(
        data_preprocessed,
        base_model["trough_model"],
        base_model["trough_scaler"],
        base_model["trough_selector"],
        base_model["trough_selected_features"],
    )

    event_buy_model = event_model_package.get("event_buy_model")
    event_sell_model = event_model_package.get("event_sell_model")
    event_features = event_model_package.get("event_features") or all_features
    event_df, _ = _add_event_sequence_features(data_preprocessed, event_features)
    if event_buy_model is not None and event_sell_model is not None:
        for feature in event_features:
            if feature not in event_df.columns:
                event_df[feature] = 0
        x_event = event_df[event_features].fillna(0).to_numpy(np.float32)
        event_buy_score = event_buy_model.predict(x_event).astype(np.float32)
        event_sell_score = event_sell_model.predict(x_event).astype(np.float32)
    else:
        event_buy_score = np.zeros(len(data_preprocessed), dtype=np.float32)
        event_sell_score = np.zeros(len(data_preprocessed), dtype=np.float32)

    data_preprocessed["Base_Peak_Probability"] = base_peak_probability
    data_preprocessed["Base_Trough_Probability"] = base_trough_probability
    data_preprocessed["Event_Buy_Score"] = event_buy_score
    data_preprocessed["Event_Sell_Score"] = event_sell_score

    if backtest_start_date is not None or backtest_end_date is not None:
        trade_dates = _parse_trade_dates(data_preprocessed["TradeDate"])
        mask = pd.Series(True, index=data_preprocessed.index)
        if backtest_start_date is not None:
            mask &= trade_dates >= pd.to_datetime(str(backtest_start_date), format="%Y%m%d", errors="coerce")
        if backtest_end_date is not None:
            mask &= trade_dates <= pd.to_datetime(str(backtest_end_date), format="%Y%m%d", errors="coerce")
        data_preprocessed = data_preprocessed.loc[mask].copy()

    base_peak_probability = data_preprocessed["Base_Peak_Probability"].to_numpy(dtype=float)
    base_trough_probability = data_preprocessed["Base_Trough_Probability"].to_numpy(dtype=float)
    event_buy_score = data_preprocessed["Event_Buy_Score"].to_numpy(dtype=float)
    event_sell_score = data_preprocessed["Event_Sell_Score"].to_numpy(dtype=float)

    base_peak_threshold = params.get("base_peak_threshold", 0.94)
    base_trough_threshold = params.get("base_trough_threshold", 0.54)
    base_signal_window = params.get("base_signal_window", 20)
    event_buy_threshold = params.get("event_buy_threshold", 0.038293278403530355)
    event_sell_threshold = params.get("event_sell_threshold", 0.03924646855558505)
    event_signal_window = params.get("event_signal_window", 40)

    base_buy = _suppress_repeated_signal_array(base_trough_probability > base_trough_threshold, base_signal_window)
    base_sell = _suppress_repeated_signal_array(base_peak_probability > base_peak_threshold, base_signal_window)
    gate_rule = params.get("event_buy_regime_gate", "Close_MA200_Diff > 0")
    if gate_rule == "Close_MA200_Diff > 0" and "Close_MA200_Diff" in data_preprocessed.columns:
        regime_gate = data_preprocessed["Close_MA200_Diff"].to_numpy(dtype=float) > 0
    else:
        regime_gate = np.ones(len(data_preprocessed), dtype=bool)
    event_buy = _suppress_repeated_signal_array(
        (event_buy_score >= event_buy_threshold) & regime_gate,
        event_signal_window,
    )
    event_sell = _suppress_repeated_signal_array(event_sell_score >= event_sell_threshold, event_signal_window)

    data_preprocessed["Event_Regime_Gate"] = regime_gate.astype(np.int8)
    data_preprocessed["Base_Trough_Signal"] = base_buy
    data_preprocessed["Base_Peak_Signal"] = base_sell
    data_preprocessed["Event_Trough_Signal"] = event_buy
    data_preprocessed["Event_Peak_Signal"] = event_sell
    data_preprocessed["Trough_Probability"] = np.maximum(base_trough_probability, event_buy_score)
    data_preprocessed["Peak_Probability"] = np.maximum(base_peak_probability, event_sell_score)
    strict_mode = params.get("strict_oos_mode")
    if strict_mode == "base_only":
        trough_prediction = base_buy
        peak_prediction = base_sell
    elif strict_mode == "event_only":
        trough_prediction = event_buy
        peak_prediction = event_sell
    elif strict_mode == "event_buy_base_sell":
        trough_prediction = event_buy
        peak_prediction = base_sell
    elif strict_mode == "base_buy_event_sell":
        trough_prediction = base_buy
        peak_prediction = event_sell
    else:
        trough_prediction = np.maximum(base_buy, event_buy)
        peak_prediction = np.maximum(base_sell, event_sell)
    data_preprocessed["Trough_Prediction"] = trough_prediction.astype(int)
    data_preprocessed["Peak_Prediction"] = peak_prediction.astype(int)

    if enable_change_signal:
        data_preprocessed = change_trough_and_peak(data_preprocessed, N_newhigh)

    signal_df = get_trade_signal(data_preprocessed)
    bt_result, trades_df = backtest_results(
        data_preprocessed,
        signal_df,
        N_buy,
        N_sell,
        enable_chase,
        enable_stop_loss,
        initial_capital=1_000_000,
    )

    if trades_df.empty:
        data_preprocessed["trade"] = None
    else:
        data_preprocessed["date"] = _parse_trade_dates(data_preprocessed["TradeDate"])
        data_preprocessed["trade"] = None
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[["exit_date"]],
            left_on="date",
            right_on="exit_date",
            how="left",
        )
        data_preprocessed["trade"] = np.where(
            data_preprocessed["exit_date"].notna(),
            "sell",
            data_preprocessed["trade"],
        )
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[["entry_date"]],
            left_on="date",
            right_on="entry_date",
            how="left",
        )
        data_preprocessed["trade"] = np.where(
            data_preprocessed["entry_date"].notna(),
            "buy",
            data_preprocessed["trade"],
        )
        data_preprocessed = data_preprocessed.drop_duplicates(subset=["date"])
        data_preprocessed.set_index("date", inplace=True)

    return data_preprocessed, bt_result, trades_df


def predict_new_data(
    new_df,
    peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold,
    trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold,
    N, mixture_depth=3, window_size=300, eval_mode=False, 
    N_buy=None, N_sell=None,  # 追涨、止损窗口
    enable_chase=True, 
    enable_stop_loss=True,
    enable_change_signal=False,
    N_newhigh=60,
    backtest_start_date=None,
    backtest_end_date=None,
):
    """
    使用训练好的模型(峰/谷)对 new_df 做预测，并可选做回测。
    注意：peak_selected_features/trough_selected_features 是模型真正见过的特征列表。
    """
    logger.info(
        "Predicting new data: rows=%s N=%s mixture_depth=%s eval_mode=%s backtest_start=%s backtest_end=%s",
        len(new_df),
        N,
        mixture_depth,
        eval_mode,
        backtest_start_date,
        backtest_end_date,
    )
    try:
        # 首先做预处理
        data_preprocessed, _ = preprocess_data(
            new_df, 
            N, 
            mixture_depth=mixture_depth, 
            mark_labels=eval_mode
        )
        # ========== 预测 Peak ==========
        logger.info("Predicting Peak probabilities")

        # 补全新数据中缺失的特征
        missing_peak = [f for f in peak_selected_features if f not in data_preprocessed.columns]
        if missing_peak:
            logger.warning("Filling missing Peak features: %s", missing_peak)
            for feature in missing_peak:
                data_preprocessed[feature] = 0
        
        # 只取模型实际使用的特征
        X_new_peak = data_preprocessed[peak_selected_features].fillna(0)
        
        # 调用训练时的 scaler
        X_new_peak_scaled = peak_scaler.transform(X_new_peak).astype(np.float32)
        logger.info("Peak feature matrix shape: %s", X_new_peak_scaled.shape)

        # 如果是 Transformer 模型，需要构造序列数据
        from skorch import NeuralNetClassifier
        
        if (isinstance(peak_model, NeuralNetClassifier) and
            isinstance(peak_model.module_, TransformerClassifier)):
            logger.info("Building Peak sequence data")
            X_seq_list = []
            for i in range(window_size, len(X_new_peak_scaled) + 1):
                seq_x = X_new_peak_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_peak = np.array(X_seq_list, dtype=np.float32)
            logger.info("Peak sequence matrix shape: %s", X_new_seq_peak.shape)

            batch_size = 64
            predictions = []
            peak_model.module_.eval()

            import torch
            with torch.no_grad():
                for i in range(0, len(X_new_seq_peak), batch_size):
                    batch = torch.from_numpy(X_new_seq_peak[i : i + batch_size]).float()
                    batch = batch.to(peak_model.device)
                    outputs = peak_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            
            all_probas = np.concatenate(predictions)
            peak_probas = np.zeros(len(data_preprocessed))
            peak_probas[window_size-1:] = all_probas
        else:
            # 传统模型或 MLP 模型
            if hasattr(peak_model, "predict_proba"):
                if peak_selector is not None:
                    X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
                    logits = peak_model.predict_proba(X_new_peak_selected)
                else:
                    logits = peak_model.predict_proba(X_new_peak_scaled)
                
                if logits.ndim == 2:
                    peak_probas = logits[:, 1]
                else:
                    import torch
                    peak_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                peak_probas = peak_model.predict(X_new_peak_scaled).astype(float)

        peak_preds = (peak_probas > peak_threshold).astype(int)
        data_preprocessed['Peak_Probability'] = peak_probas
        data_preprocessed['Peak_Prediction'] = peak_preds

        # ========== 预测 Trough ==========
        logger.info("Predicting Trough probabilities")

        missing_trough = [f for f in trough_selected_features if f not in data_preprocessed.columns]
        if missing_trough:
            logger.warning("Filling missing Trough features: %s", missing_trough)
            for feature in missing_trough:
                data_preprocessed[feature] = 0

        X_new_trough = data_preprocessed[trough_selected_features].fillna(0)
        X_new_trough_scaled = trough_scaler.transform(X_new_trough).astype(np.float32)
        logger.info("Trough feature matrix shape: %s", X_new_trough_scaled.shape)

        if (isinstance(trough_model, NeuralNetClassifier) and
            isinstance(trough_model.module_, TransformerClassifier)):
            logger.info("Building Trough sequence data")
            X_seq_list = []
            for i in range(window_size, len(X_new_trough_scaled) + 1):
                seq_x = X_new_trough_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_trough = np.array(X_seq_list, dtype=np.float32)
            logger.info("Trough sequence matrix shape: %s", X_new_seq_trough.shape)

            batch_size = 64
            predictions = []
            trough_model.module_.eval()

            import torch  # 补充缺失的 torch 导入
            with torch.no_grad():
                for i in range(0, len(X_new_seq_trough), batch_size):
                    batch = torch.from_numpy(X_new_seq_trough[i : i + batch_size]).float()
                    batch = batch.to(trough_model.device)
                    outputs = trough_model.module_(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predictions.append(probs.cpu().numpy())
            
            all_probas = np.concatenate(predictions)
            trough_probas = np.zeros(len(data_preprocessed))
            trough_probas[window_size-1:] = all_probas
        else:
            if hasattr(trough_model, "predict_proba"):
                if trough_selector is not None:
                    X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
                    logits = trough_model.predict_proba(X_new_trough_selected)
                else:
                    logits = trough_model.predict_proba(X_new_trough_scaled)
                
                if logits.ndim == 2:
                    trough_probas = logits[:, 1]
                else:
                    import torch
                    trough_probas = torch.sigmoid(torch.tensor(logits)).numpy()
            else:
                trough_probas = trough_model.predict(X_new_trough_scaled).astype(float)

        trough_preds = (trough_probas > trough_threshold).astype(int)
        data_preprocessed['Trough_Probability'] = trough_probas
        data_preprocessed['Trough_Prediction'] = trough_preds

        if backtest_start_date is not None or backtest_end_date is not None:
            trade_dates = pd.to_datetime(data_preprocessed['TradeDate'], errors='coerce')
            mask = pd.Series(True, index=data_preprocessed.index)
            if backtest_start_date is not None:
                mask &= trade_dates >= pd.to_datetime(str(backtest_start_date), format='%Y%m%d', errors='coerce')
            if backtest_end_date is not None:
                mask &= trade_dates <= pd.to_datetime(str(backtest_end_date), format='%Y%m%d', errors='coerce')
            data_preprocessed = data_preprocessed.loc[mask].copy()

        # ====== 后处理：20日内不重复预测 (根据原逻辑) ======
        logger.info("Applying prediction post-processing")
        data_preprocessed.index = data_preprocessed.index.astype(str)
        for idx, index in enumerate(data_preprocessed.index):
            if data_preprocessed.loc[index, 'Peak_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Peak_Prediction')] = 0
            if data_preprocessed.loc[index, 'Trough_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Trough_Prediction')] = 0

        # 若启用其他信号修改
        if enable_change_signal:
            data_preprocessed = change_trough_and_peak(data_preprocessed, N_newhigh)

        # ====== 回测部分 ======
        signal_df = get_trade_signal(data_preprocessed)
        bt_result, trades_df = backtest_results(
            data_preprocessed, 
            signal_df,
            N_buy,           # 追涨窗口
            N_sell,          # 止损窗口
            enable_chase,    # 是否启用追涨
            enable_stop_loss,# 是否启用止损
            initial_capital=1_000_000
        )

        # 若交易记录为空，则直接返回默认回测结果，并跳过后续交易日期的合并
        if trades_df.empty:
            logger.info("No trades generated; returning default backtest result")
            data_preprocessed['trade'] = None
        else:
            # 用 'TradeDate' 或索引做时间列
            if 'TradeDate' in data_preprocessed.columns:
                data_preprocessed['date'] = pd.to_datetime(data_preprocessed['TradeDate'], errors='coerce')
            else:
                data_preprocessed['date'] = pd.to_datetime(data_preprocessed.index, errors='coerce')

            data_preprocessed['trade'] = None
            # 合并卖出日期
            data_preprocessed = pd.merge(
                data_preprocessed,
                trades_df[['exit_date']],
                left_on='date',
                right_on='exit_date',
                how='left'
            )
            data_preprocessed['trade'] = np.where(
                data_preprocessed['exit_date'].notna(), 
                'sell', 
                data_preprocessed['trade']
            )

            # 合并买入日期
            data_preprocessed = pd.merge(
                data_preprocessed,
                trades_df[['entry_date']],
                left_on='date',
                right_on='entry_date',
                how='left'
            )
            data_preprocessed['trade'] = np.where(
                data_preprocessed['entry_date'].notna(),
                'buy',
                data_preprocessed['trade']
            )

            # 删除重复并将日期设为索引
            data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
            data_preprocessed.set_index('date', inplace=True)

    except Exception as e:
        logger.exception("predict_new_data failed: %s", e)
        if 'trades_df' in locals():
            logger.debug("Backtest trades before failure:\n%s", trades_df)
        else:
            logger.debug("No trades were generated before failure")
        raise e

    return data_preprocessed, bt_result, trades_df


def predict_new_data_with_ensemble(
    new_df,
    original_peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold,
    finetuned_peak_model,
    original_trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold,
    finetuned_trough_model,
    N, mixture_depth=3, window_size=300, eval_mode=False,
    ensemble_weight=0.5,
    N_buy=None, N_sell=None,  # 追涨、止损窗口
    enable_chase=True,
    enable_stop_loss=True,
    enable_change_signal=False,
    N_newhigh=60
):
    """
    使用硬混合模型对 new_df 进行预测及回测：
    对峰/谷预测分别使用原始模型与微调模型的预测按 ensemble_weight 加权融合。
    
    参数:
      new_df: 待预测数据
      original_peak_model: 原始峰模型
      peak_scaler, peak_selector, peak_selected_features, peak_threshold: 峰模型相关组件及阈值
      finetuned_peak_model: 微调后的峰模型
      original_trough_model: 原始谷模型
      trough_scaler, trough_selector, trough_selected_features, trough_threshold: 谷模型相关组件及阈值
      finetuned_trough_model: 微调后的谷模型
      N, mixture_depth, window_size, eval_mode: 数据预处理和预测相关参数
      ensemble_weight: 原始模型权重（0～1之间），微调模型权重即为 (1-ensemble_weight)
      N_buy, N_sell: 回测参数（追涨、止损窗口）
      enable_chase, enable_stop_loss, enable_change_signal, N_newhigh: 策略相关参数
      
    返回:
      data_preprocessed: 包含预测结果的 DataFrame
      bt_result: 回测结果字典
      trades_df: 交易记录 DataFrame
    """
    import numpy as np
    import torch
    from skorch import NeuralNetClassifier
    # 预处理数据
    try:
        logger.info(
            "Predicting with ensemble: rows=%s N=%s mixture_depth=%s eval_mode=%s ensemble_weight=%s",
            len(new_df),
            N,
            mixture_depth,
            eval_mode,
            ensemble_weight,
        )
        data_preprocessed, _ = preprocess_data(
            new_df, 
            N, 
            mixture_depth=mixture_depth, 
            mark_labels=eval_mode
        )
        
        # ---------------- Peak 预测 ----------------
        # 补全缺失的峰特征
        missing_peak = [f for f in peak_selected_features if f not in data_preprocessed.columns]
        if missing_peak:
            logger.warning("Filling missing Peak features for ensemble: %s", missing_peak)
            for feature in missing_peak:
                data_preprocessed[feature] = 0

        X_new_peak = data_preprocessed[peak_selected_features].fillna(0)
        X_new_peak_scaled = peak_scaler.transform(X_new_peak).astype(np.float32)
        logger.info("Ensemble Peak feature matrix shape: %s", X_new_peak_scaled.shape)
        
        # 判断是否为 Transformer 模型
        if (isinstance(original_peak_model, NeuralNetClassifier) and
            isinstance(original_peak_model.module_, TransformerClassifier)):
            logger.info("Building ensemble Peak sequence data")
            X_seq_list = []
            for i in range(window_size, len(X_new_peak_scaled) + 1):
                seq_x = X_new_peak_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_peak = np.array(X_seq_list, dtype=np.float32)
            logger.info("Ensemble Peak sequence matrix shape: %s", X_new_seq_peak.shape)
            
            batch_size = 64
            predictions_list = []
            original_peak_model.module_.eval()
            finetuned_peak_model.module_.eval()
            device = original_peak_model.device if hasattr(original_peak_model, 'device') else torch.device("cpu")
            
            for i in range(0, len(X_new_seq_peak), batch_size):
                batch = torch.from_numpy(X_new_seq_peak[i : i + batch_size]).float().to(device)
                # 分别计算两个模型的输出
                outputs_orig = original_peak_model.module_(batch)
                outputs_finetune = finetuned_peak_model.module_(batch)
                probs_orig = torch.softmax(outputs_orig, dim=1)[:, 1]
                probs_finetune = torch.softmax(outputs_finetune, dim=1)[:, 1]
                combined_probs = ensemble_weight * probs_orig + (1 - ensemble_weight) * probs_finetune
                predictions_list.append(combined_probs.cpu().numpy())
            all_probas_seq = np.concatenate(predictions_list)
            # 对应原始数据长度，前 window_size-1 个位置无法生成序列，补 0
            peak_probas = np.zeros(len(data_preprocessed))
            peak_probas[window_size-1:] = all_probas_seq
        else:
            # 若使用传统模型或 MLP，则先根据 peak_selector（若有）做特征选择
            if peak_selector is not None:
                X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
            else:
                X_new_peak_selected = X_new_peak_scaled
            # 调用 ensemble 融合函数（需要提前定义 predict_with_model_ensemble）
            peak_pred, peak_probas = predict_with_model_ensemble(
                X_new_peak_selected, 
                original_peak_model, 
                finetuned_peak_model, 
                peak_threshold, 
                ensemble_weight
            )
        
        data_preprocessed['Peak_Probability'] = peak_probas
        # 此处可直接采用融合后的概率生成标记
        data_preprocessed['Peak_Prediction'] = (peak_probas > peak_threshold).astype(int)
        
        # ---------------- Trough 预测 ----------------
        missing_trough = [f for f in trough_selected_features if f not in data_preprocessed.columns]
        if missing_trough:
            logger.warning("Filling missing Trough features for ensemble: %s", missing_trough)
            for feature in missing_trough:
                data_preprocessed[feature] = 0

        X_new_trough = data_preprocessed[trough_selected_features].fillna(0)
        X_new_trough_scaled = trough_scaler.transform(X_new_trough).astype(np.float32)
        logger.info("Ensemble Trough feature matrix shape: %s", X_new_trough_scaled.shape)
        
        if (isinstance(original_trough_model, NeuralNetClassifier) and
            isinstance(original_trough_model.module_, TransformerClassifier)):
            logger.info("Building ensemble Trough sequence data")
            X_seq_list = []
            for i in range(window_size, len(X_new_trough_scaled) + 1):
                seq_x = X_new_trough_scaled[i - window_size:i]
                X_seq_list.append(seq_x)
            X_new_seq_trough = np.array(X_seq_list, dtype=np.float32)
            logger.info("Ensemble Trough sequence matrix shape: %s", X_new_seq_trough.shape)
            
            batch_size = 64
            predictions_list = []
            original_trough_model.module_.eval()
            finetuned_trough_model.module_.eval()
            device = original_trough_model.device if hasattr(original_trough_model, 'device') else torch.device("cpu")
            
            for i in range(0, len(X_new_seq_trough), batch_size):
                batch = torch.from_numpy(X_new_seq_trough[i : i + batch_size]).float().to(device)
                outputs_orig = original_trough_model.module_(batch)
                outputs_finetune = finetuned_trough_model.module_(batch)
                probs_orig = torch.softmax(outputs_orig, dim=1)[:, 1]
                probs_finetune = torch.softmax(outputs_finetune, dim=1)[:, 1]
                combined_probs = ensemble_weight * probs_orig + (1 - ensemble_weight) * probs_finetune
                predictions_list.append(combined_probs.cpu().numpy())
            all_probas_seq = np.concatenate(predictions_list)
            trough_probas = np.zeros(len(data_preprocessed))
            trough_probas[window_size-1:] = all_probas_seq
        else:
            if trough_selector is not None:
                X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
            else:
                X_new_trough_selected = X_new_trough_scaled
            trough_pred, trough_probas = predict_with_model_ensemble(
                X_new_trough_selected, 
                original_trough_model, 
                finetuned_trough_model, 
                trough_threshold, 
                ensemble_weight
            )
        
        data_preprocessed['Trough_Probability'] = trough_probas
        data_preprocessed['Trough_Prediction'] = (trough_probas > trough_threshold).astype(int)

        # ---------------- 后处理：避免短期内重复信号 ----------------
        logger.info("Applying ensemble prediction post-processing")
        data_preprocessed.index = data_preprocessed.index.astype(str)
        for idx, index in enumerate(data_preprocessed.index):
            if data_preprocessed.loc[index, 'Peak_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Peak_Prediction')] = 0
            if data_preprocessed.loc[index, 'Trough_Prediction'] == 1:
                start = idx + 1
                end = min(idx + 20, len(data_preprocessed))
                data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Trough_Prediction')] = 0
        
        if enable_change_signal:
            data_preprocessed = change_trough_and_peak(data_preprocessed, N_newhigh)
        
        # ---------------- 回测 ----------------
        signal_df = get_trade_signal(data_preprocessed)
        bt_result, trades_df = backtest_results(
            data_preprocessed, 
            signal_df,
            N_buy,
            N_sell,
            enable_chase,
            enable_stop_loss,
            initial_capital=1_000_000
        )
        
        # 处理时间序列和交易标记
        if 'TradeDate' in data_preprocessed.columns:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed['TradeDate'], errors='coerce')
        else:
            data_preprocessed['date'] = pd.to_datetime(data_preprocessed.index, errors='coerce')
        
        data_preprocessed['trade'] = None
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['exit_date']],
            left_on='date',
            right_on='exit_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(data_preprocessed['exit_date'].notna(), 'sell', data_preprocessed['trade'])
        data_preprocessed = pd.merge(
            data_preprocessed,
            trades_df[['entry_date']],
            left_on='date',
            right_on='entry_date',
            how='left'
        )
        data_preprocessed['trade'] = np.where(data_preprocessed['entry_date'].notna(), 'buy', data_preprocessed['trade'])
        
        data_preprocessed = data_preprocessed.drop_duplicates(subset=['date'])
        data_preprocessed.set_index('date', inplace=True)
        
    except Exception as e:
        logger.exception("predict_new_data_with_ensemble failed: %s", e)
        if 'trades_df' in locals():
            logger.debug("Backtest trades before ensemble failure:\n%s", trades_df)
        else:
            logger.debug("No trades were generated before ensemble failure")
        raise e
    
    return data_preprocessed, bt_result, trades_df


def predict_with_model_ensemble(X, original_model, finetuned_model, threshold, ensemble_weight=0.5):
    """
    将原始模型和微调模型的预测进行加权混合
    
    参数:
      X: 特征数据
      original_model: 原始训练好的模型
      finetuned_model: 微调后的模型
      threshold: 预测阈值
      ensemble_weight: 原始模型的权重 (0-1之间)
    
    返回:
      predictions: 二分类预测结果
      probabilities: 融合后的预测概率
    """
    orig_proba = original_model.predict_proba(X)
    finetuned_proba = finetuned_model.predict_proba(X)
    combined_proba = ensemble_weight * orig_proba + (1 - ensemble_weight) * finetuned_proba
    return (combined_proba[:, 1] >= threshold).astype(int), combined_proba[:, 1]

#W出现于阴线，D出现于阳线，且盘中要创60日新高
def change_trough_and_peak(df, N_newhigh):
    
    def update_peak_or_trough(df, prediction_col, opposite_col, condition):
        for i, date in enumerate(df.index):
            # 只处理预测值为1的情况
            if df.loc[date, prediction_col] == 1:
                if condition(df, i, date):
                    df.loc[date, prediction_col] = 1  # 保持当前预测
                else:
                    df.loc[date, prediction_col] = 0  # 移除预测
                    # 寻找下一个符合条件的日期，将预测信号转移过去
                    for j in range(i + 1, len(df)):
                        next_date = df.index[j]
                        if condition(df, j, next_date):
                            df.loc[next_date, prediction_col] = 1
                            break
        return df

    # 高点处理
    if N_newhigh > 0:
        # 当N_newhigh>0时，执行完整逻辑（阴线且创新高）
        peak_condition = lambda df, i, date: (
            i >= N_newhigh and 
            df.loc[date, 'High'] > df.loc[df.index[i-N_newhigh:i], 'Close'].max() and 
            df.loc[date, 'Close'] < df.loc[date, 'Open']
        )
    else:
        # 当N_newhigh=0时，仅检查阴线条件
        peak_condition = lambda df, i, date: df.loc[date, 'Close'] < df.loc[date, 'Open']
    
    df = update_peak_or_trough(
        df, 
        'Peak_Prediction', 
        'Trough_Prediction', 
        peak_condition
    )

    # 低点处理保持不变（仅阳线条件）
    df = update_peak_or_trough(
        df, 
        'Trough_Prediction', 
        'Peak_Prediction', 
        lambda df, i, date: df.loc[date, 'Close'] > df.loc[date, 'Open']
    )

    return df





def adjust_probabilities_in_range(df, start_date, end_date):
    """
    将 DataFrame 中指定日期范围内的 'Peak_Probability' 和 'Trough_Probability' 列的值设为 0。

    参数:
      df: 包含预测结果的 DataFrame，其索引为日期。
      start_date: 起始日期（字符串，格式 'YYYY-MM-DD'）。
      end_date: 截止日期（字符串，格式 'YYYY-MM-DD'）。

    返回:
      修改后的 DataFrame。
    """
    # 如果索引不是 datetime 类型，则转换为 datetime 类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    
    if "Peak_Probability" in df.columns:
        df.loc[mask, "Peak_Prediction"] = 0
        df.loc[mask, "Peak"] = 0
        df.loc[mask, "Peak_Probability"] = 0
    if "Trough_Probability" in df.columns:
        df.loc[mask, "Trough_Prediction"] = 0
        df.loc[mask, "Trough"] = 0
        df.loc[mask, "Trough_Probability"] = 0
    return df


def get_trade_signal(data_preprocessed):
    # 复制数据以避免修改原始 DataFrame
    data_preprocessed = data_preprocessed.copy()

    # 筛选出存在高点或低点预测的行
    signal_mask = (
        (data_preprocessed['Peak_Prediction'] == 1)
        | (data_preprocessed['Trough_Prediction'] == 1)
    )
    signal_df = data_preprocessed.loc[signal_mask, ['Peak_Prediction', 'Trough_Prediction']].copy()
    signal_df['direction'] = ''
    
    # 对于高点预测的行，设定方向为 'sell'
    signal_df.loc[signal_df['Peak_Prediction'] == 1, 'direction'] = 'sell'
    
    # 对于低点预测的行，设定方向为 'buy'
    signal_df.loc[signal_df['Trough_Prediction'] == 1, 'direction'] = 'buy'
    
    # 仅返回交易方向这一列
    signal_df = signal_df[['direction']]
    

    return signal_df
