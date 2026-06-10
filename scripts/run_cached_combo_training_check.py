import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import backtest_results
from src.plot_candlestick import plot_candlestick
from src.predict import get_trade_signal
from src.preprocess import preprocess_data
from src.tushare_function import read_day_from_tushare, select_time


N = 20
MIXTURE_DEPTH = 1
CLASSIFIER_NAME = "MLP"
OVERSAMPLE_METHOD = "SMOTE"
NUM_ROUNDS = int(os.environ.get("COMBO_NUM_ROUNDS", "90"))
SEED_BASE = int(os.environ.get("COMBO_SEED_BASE", "7300"))
ROUND_TIMEOUT_SECONDS = int(os.environ.get("COMBO_ROUND_TIMEOUT_SECONDS", "420"))
RUN_TAG = os.environ.get("COMBO_RUN_TAG", "cached")
FEATURE_VERSION = os.environ.get("FEATURE_VERSION", "base")

TRAIN_START = "20000101"
TRAIN_END = "20201231"
PRED_START = "20210101"
PRED_END = "20260608"

N_BUY = 10
N_SELL = 10
ENABLE_CHASE = False
ENABLE_STOP_LOSS = False

if RUN_TAG == "cached":
    CACHE_DIR = Path("fixed_feature_combo_cache")
    RESULT_PATH = Path("fixed_feature_cached_combo_result.json")
    LOG_PATH = Path("fixed_feature_cached_combo.log")
    CHART_HTML_PATH = Path("best_combo_prediction_kline.html")
    CHART_JSON_PATH = Path("best_combo_prediction_result.json")
else:
    CACHE_DIR = Path(f"fixed_feature_{RUN_TAG}_combo_cache")
    RESULT_PATH = Path(f"fixed_feature_{RUN_TAG}_combo_result.json")
    LOG_PATH = Path(f"fixed_feature_{RUN_TAG}_combo.log")
    CHART_HTML_PATH = Path(f"best_combo_{RUN_TAG}_prediction_kline.html")
    CHART_JSON_PATH = Path(f"best_combo_{RUN_TAG}_prediction_result.json")

PRED_DIR = CACHE_DIR / "predictions"
WORKER_LOG_DIR = CACHE_DIR / "worker_logs"
DATA_CACHE_PATH = CACHE_DIR / "prepared_data.pkl"


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def normalize_market_data(df):
    base_cols = ["Open", "High", "Low", "Close", "Volume", "Amount", "TradeDate"]
    keep_cols = [c for c in base_cols if c in df.columns]
    out = df[keep_cols].copy()
    out["TradeDate"] = out["TradeDate"].astype(str).str.replace("-", "", regex=False)
    for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["TradeDate", "Open", "High", "Low", "Close"])


def prepare_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_CACHE_PATH.exists():
        with DATA_CACHE_PATH.open("rb") as f:
            cached = pickle.load(f)
        params = cached.get("params", {})
        if (
            params.get("N") == N
            and params.get("mixture_depth") == MIXTURE_DEPTH
            and params.get("train_start") == TRAIN_START
            and params.get("train_end") == TRAIN_END
            and params.get("pred_start") == PRED_START
            and params.get("pred_end") == PRED_END
            and params.get("feature_version") == FEATURE_VERSION
        ):
            log("复用已准备好的全历史特征缓存")
            return cached

    log("读取本地完整数据.csv")
    t_read = time.time()
    local_df = pd.read_csv("完整数据.csv")
    log(f"本地CSV读取完成，用时 {time.time()-t_read:.2f}s，shape={local_df.shape}")
    raw = normalize_market_data(local_df)
    log(f"本地数据 shape={raw.shape}, date={raw.TradeDate.min()}~{raw.TradeDate.max()}")

    if raw["TradeDate"].max() < PRED_END:
        log(f"本地数据不足，使用 Tushare 获取 000001.SH 指数行情至 {PRED_END}")
        t_ts = time.time()
        ts_df = read_day_from_tushare(
            "000001.SH",
            symbol_type="index",
            start_date="19920101",
            end_date=PRED_END,
        )
        log(f"Tushare 请求完成，用时 {time.time()-t_ts:.2f}s，shape={ts_df.shape}")
        if ts_df.empty:
            raise RuntimeError(f"Tushare 未返回行情，无法覆盖回测截止日 {PRED_END}")
        ts_raw = normalize_market_data(ts_df.reset_index(drop=True))
        raw = (
            pd.concat([raw, ts_raw], ignore_index=True)
            .drop_duplicates(subset=["TradeDate"], keep="last")
            .sort_values("TradeDate")
            .reset_index(drop=True)
        )
        log(f"合并后数据 shape={raw.shape}, date={raw.TradeDate.min()}~{raw.TradeDate.max()}")

    if raw["TradeDate"].max() < PRED_END:
        raise RuntimeError(f"行情截止 {raw.TradeDate.max()}，仍早于回测截止 {PRED_END}")

    log("执行 preprocess_data(mark_labels=True)")
    processed, all_features = preprocess_data(raw.copy(), N, MIXTURE_DEPTH, mark_labels=True)
    processed_for_select = processed.copy()
    processed_for_select["TradeDate"] = pd.to_datetime(processed_for_select["TradeDate"]).dt.strftime("%Y%m%d")
    train_df = select_time(processed_for_select, TRAIN_START, TRAIN_END)
    log(
        f"训练集 shape={train_df.shape}, features={len(all_features)}, "
        f"Peak={int(train_df['Peak'].sum())}, Trough={int(train_df['Trough'].sum())}"
    )

    log("执行预测集 preprocess_data(mark_labels=True)，后续所有模型复用同一特征矩阵")
    pred_preprocessed, _ = preprocess_data(raw.copy(), N, mixture_depth=MIXTURE_DEPTH, mark_labels=True)
    trade_dates = pd.to_datetime(pred_preprocessed["TradeDate"], errors="coerce")
    backtest_mask = pd.Series(True, index=pred_preprocessed.index)
    backtest_mask &= trade_dates >= pd.to_datetime(PRED_START, format="%Y%m%d")
    backtest_mask &= trade_dates <= pd.to_datetime(PRED_END, format="%Y%m%d")
    mask_values = backtest_mask.to_numpy()
    base_backtest_df = pred_preprocessed.loc[backtest_mask].copy()
    log(f"预测特征 shape={pred_preprocessed.shape}, 回测切片 shape={base_backtest_df.shape}")

    cached = {
        "params": {
            "N": N,
            "mixture_depth": MIXTURE_DEPTH,
            "classifier_name": CLASSIFIER_NAME,
            "oversample_method": OVERSAMPLE_METHOD,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "pred_start": PRED_START,
            "pred_end": PRED_END,
            "feature_version": FEATURE_VERSION,
        },
        "train_df": train_df,
        "all_features": all_features,
        "pred_preprocessed": pred_preprocessed,
        "mask_values": mask_values,
        "base_backtest_df": base_backtest_df,
    }
    tmp_path = DATA_CACHE_PATH.with_suffix(".tmp.pkl")
    with tmp_path.open("wb") as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(DATA_CACHE_PATH)
    return cached


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


def prediction_path(round_no):
    return PRED_DIR / f"round_{round_no:03d}.npz"


def metadata_path(round_no):
    return PRED_DIR / f"round_{round_no:03d}.json"


def completed_rounds():
    rounds = []
    for path in sorted(PRED_DIR.glob("round_*.npz")):
        try:
            rounds.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return rounds


def run_worker(round_no, seed):
    if prediction_path(round_no).exists() and metadata_path(round_no).exists():
        log(f"第 {round_no}/{NUM_ROUNDS} 轮已有预测缓存，跳过训练")
        return True

    log(f"开始训练第 {round_no}/{NUM_ROUNDS} 轮模型，seed={seed}")
    worker_log_path = WORKER_LOG_DIR / f"round_{round_no:03d}.log"
    cmd = [
        sys.executable,
        "train_combo_round_worker.py",
        "--round",
        str(round_no),
        "--seed",
        str(seed),
        "--data-cache",
        str(DATA_CACHE_PATH),
        "--output-dir",
        str(PRED_DIR),
    ]
    t0 = time.time()
    with worker_log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=ROUND_TIMEOUT_SECONDS,
                check=True,
            )
        except subprocess.TimeoutExpired:
            log(f"第 {round_no} 轮超过 {ROUND_TIMEOUT_SECONDS}s，已跳过")
            return False
        except subprocess.CalledProcessError as exc:
            log(f"第 {round_no} 轮训练失败，退出码={exc.returncode}，详见 {worker_log_path}")
            return False

    if not prediction_path(round_no).exists():
        log(f"第 {round_no} 轮结束但未生成预测缓存，详见 {worker_log_path}")
        return False

    with metadata_path(round_no).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    log(
        f"完成第 {round_no}/{NUM_ROUNDS} 轮，用时 {time.time()-t0:.1f}s, "
        f"peak_score={meta['peak_score']:.4f}, peak_th={meta['peak_threshold']:.4f}, "
        f"peak信号={meta['peak_signals']}, trough_score={meta['trough_score']:.4f}, "
        f"trough_th={meta['trough_threshold']:.4f}, trough信号={meta['trough_signals']}"
    )
    return True


def load_round_prediction(round_no):
    with np.load(prediction_path(round_no)) as arr:
        return {
            "round": round_no,
            "peak_probas": arr["peak_probas"],
            "peak_preds": arr["peak_preds"],
            "trough_probas": arr["trough_probas"],
            "trough_preds": arr["trough_preds"],
        }


def evaluate_combo(data_cache, peak_pred, trough_pred, combo_index):
    mask_values = data_cache["mask_values"]
    combo_df = data_cache["base_backtest_df"].copy()
    combo_df["Peak_Probability"] = peak_pred["peak_probas"][mask_values]
    combo_df["Peak_Prediction"] = peak_pred["peak_preds"][mask_values]
    combo_df["Trough_Probability"] = trough_pred["trough_probas"][mask_values]
    combo_df["Trough_Prediction"] = trough_pred["trough_preds"][mask_values]
    combo_df = suppress_repeated_signals(combo_df)
    signal_df = get_trade_signal(combo_df)
    bt, trades = backtest_results(
        combo_df,
        signal_df,
        1,
        1,
        False,
        False,
        initial_capital=1_000_000,
    )
    return {
        "combo_index": combo_index,
        "peak_model_index": peak_pred["round"],
        "trough_model_index": trough_pred["round"],
        "累计收益率": float(bt.get("累计收益率", float("-inf"))),
        "超额收益率": float(bt.get("超额收益率", float("-inf"))),
        "胜率": None if bt.get("胜率") is None else float(bt.get("胜率")),
        "最大回撤": float(bt.get("最大回撤", 0)),
        "交易笔数": int(bt.get("交易笔数", 0)),
        "年化夏普比率": float(bt.get("年化夏普比率", 0)),
    }


def load_previous_state():
    if not RESULT_PATH.exists():
        return None
    with RESULT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_result(best_by_excess, best_by_cumulative, evaluated_count, skipped_rounds, elapsed_seconds, complete):
    done_rounds = completed_rounds()
    result = {
        "params": {
            "N": N,
            "mixture_depth": MIXTURE_DEPTH,
            "classifier_name": CLASSIFIER_NAME,
            "oversample_method": OVERSAMPLE_METHOD,
            "num_rounds": NUM_ROUNDS,
            "completed_rounds": len(done_rounds),
            "completed_round_ids": done_rounds,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "pred_start": PRED_START,
            "pred_end": PRED_END,
            "selection_metric": "超额收益率",
            "change_type": "cached_seeded_model_pool",
            "feature_version": FEATURE_VERSION,
            "checkpoint_complete": complete,
        },
        "best_by_excess": best_by_excess,
        "best_by_cumulative": best_by_cumulative,
        "best_excess_gt_100pct": (
            best_by_excess is not None and best_by_excess["超额收益率"] > 1.0
        ),
        "evaluated_combos": evaluated_count,
        "skipped_rounds": skipped_rounds,
        "elapsed_seconds": elapsed_seconds,
    }
    tmp_path = RESULT_PATH.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    tmp_path.replace(RESULT_PATH)
    return result


def update_best_records(record, best_by_excess, best_by_cumulative):
    if best_by_excess is None or record["超额收益率"] > best_by_excess["超额收益率"]:
        best_by_excess = record
    if best_by_cumulative is None or record["累计收益率"] > best_by_cumulative["累计收益率"]:
        best_by_cumulative = record
    return best_by_excess, best_by_cumulative


def evaluate_new_round(data_cache, predictions, evaluated_pairs, new_round, best_by_excess, best_by_cumulative):
    evaluated_now = 0
    round_ids = sorted(predictions)
    for trough_round in round_ids:
        pair = (new_round, trough_round)
        if pair not in evaluated_pairs:
            evaluated_pairs.add(pair)
            evaluated_now += 1
            record = evaluate_combo(
                data_cache,
                predictions[new_round],
                predictions[trough_round],
                len(evaluated_pairs),
            )
            best_by_excess, best_by_cumulative = update_best_records(
                record,
                best_by_excess,
                best_by_cumulative,
            )

    for peak_round in round_ids:
        pair = (peak_round, new_round)
        if pair not in evaluated_pairs:
            evaluated_pairs.add(pair)
            evaluated_now += 1
            record = evaluate_combo(
                data_cache,
                predictions[peak_round],
                predictions[new_round],
                len(evaluated_pairs),
            )
            best_by_excess, best_by_cumulative = update_best_records(
                record,
                best_by_excess,
                best_by_cumulative,
            )

    return best_by_excess, best_by_cumulative, evaluated_now


def evaluate_all_cached(data_cache):
    rounds = completed_rounds()
    predictions = {round_no: load_round_prediction(round_no) for round_no in rounds}
    best_by_excess = None
    best_by_cumulative = None
    evaluated_count = 0
    total = len(rounds) * len(rounds)
    t0 = time.time()

    for pi, peak_round in enumerate(rounds):
        for ti, trough_round in enumerate(rounds):
            evaluated_count += 1
            combo_index = pi * len(rounds) + ti + 1
            record = evaluate_combo(
                data_cache,
                predictions[peak_round],
                predictions[trough_round],
                combo_index,
            )
            if best_by_excess is None or record["超额收益率"] > best_by_excess["超额收益率"]:
                best_by_excess = record
            if best_by_cumulative is None or record["累计收益率"] > best_by_cumulative["累计收益率"]:
                best_by_cumulative = record
        if (pi + 1) % 5 == 0 or pi == 0 or pi + 1 == len(rounds):
            log(
                f"组合评估进度 {pi+1}/{len(rounds)} 行，已评估 {evaluated_count}/{total}，"
                f"当前最佳超额={best_by_excess['超额收益率']*100:.2f}%"
            )

    log(f"组合评估完成，用时 {time.time()-t0:.1f}s")
    return best_by_excess, best_by_cumulative, evaluated_count


def mark_trades(combo_df, trades_df):
    combo_df = combo_df.copy()
    combo_df["trade"] = None
    if trades_df.empty:
        return combo_df

    if not isinstance(combo_df.index, pd.DatetimeIndex):
        combo_df.index = pd.to_datetime(combo_df.index)

    for _, trade in trades_df.iterrows():
        entry_date = pd.to_datetime(trade.get("entry_date"))
        exit_date = pd.to_datetime(trade.get("exit_date"))
        if entry_date in combo_df.index:
            combo_df.loc[entry_date, "trade"] = "buy"
        if exit_date in combo_df.index:
            combo_df.loc[exit_date, "trade"] = "sell"
    return combo_df


def write_best_chart(data_cache, best_record):
    if best_record is None:
        return None

    peak_pred = load_round_prediction(best_record["peak_model_index"])
    trough_pred = load_round_prediction(best_record["trough_model_index"])
    mask_values = data_cache["mask_values"]
    combo_df = data_cache["base_backtest_df"].copy()
    combo_df["Peak_Probability"] = peak_pred["peak_probas"][mask_values]
    combo_df["Peak_Prediction"] = peak_pred["peak_preds"][mask_values]
    combo_df["Trough_Probability"] = trough_pred["trough_probas"][mask_values]
    combo_df["Trough_Prediction"] = trough_pred["trough_preds"][mask_values]
    combo_df = suppress_repeated_signals(combo_df)
    signal_df = get_trade_signal(combo_df)
    bt, trades_df = backtest_results(
        combo_df,
        signal_df,
        N_BUY,
        N_SELL,
        ENABLE_CHASE,
        ENABLE_STOP_LOSS,
        initial_capital=1_000_000,
    )
    combo_df = mark_trades(combo_df, trades_df)
    if not isinstance(combo_df.index, pd.DatetimeIndex):
        combo_df.index = pd.to_datetime(combo_df.index)

    peaks_pred = combo_df[combo_df["Peak_Prediction"] == 1]
    troughs_pred = combo_df[combo_df["Trough_Prediction"] == 1]
    fig = plot_candlestick(
        combo_df.copy(),
        "000001.SH",
        pd.to_datetime(PRED_START, format="%Y%m%d"),
        pd.to_datetime(PRED_END, format="%Y%m%d"),
        peaks_pred,
        troughs_pred,
        prediction=True,
        bt_result=bt,
    )
    fig.write_html(CHART_HTML_PATH, include_plotlyjs="cdn", full_html=True)

    out = {
        "chart_html": str(CHART_HTML_PATH),
        "best_combo": best_record,
        "bt_result": bt,
        "predicted_peaks": int(len(peaks_pred)),
        "predicted_troughs": int(len(troughs_pred)),
        "trades_count": int(len(trades_df)),
        "trades": trades_df.to_dict(orient="records"),
    }
    with CHART_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"最佳组合K线图已导出: {CHART_HTML_PATH}")
    return out


def main():
    start_time = time.time()
    LOG_PATH.write_text("", encoding="utf-8")
    skipped_rounds = []
    try:
        data_cache = prepare_data()
        predictions = {}
        evaluated_pairs = set()
        best_by_excess = None
        best_by_cumulative = None

        existing_rounds = completed_rounds()
        if existing_rounds:
            log(f"启动时发现已有 {len(existing_rounds)} 轮预测缓存，先加载并评估已有组合")
            for existing_round in existing_rounds:
                predictions[existing_round] = load_round_prediction(existing_round)
                best_by_excess, best_by_cumulative, evaluated_now = evaluate_new_round(
                    data_cache,
                    predictions,
                    evaluated_pairs,
                    existing_round,
                    best_by_excess,
                    best_by_cumulative,
                )
                log(
                    f"已载入第 {existing_round} 轮缓存，新增评估 {evaluated_now} 个组合，"
                    f"累计评估 {len(evaluated_pairs)} 个，最佳超额={best_by_excess['超额收益率']*100:.4f}%"
                )
            write_result(
                best_by_excess,
                best_by_cumulative,
                len(evaluated_pairs),
                skipped_rounds,
                time.time() - start_time,
                complete=False,
            )

        for round_no in range(1, NUM_ROUNDS + 1):
            seed = SEED_BASE + round_no
            if round_no in predictions:
                continue
            ok = run_worker(round_no, seed)
            if not ok:
                skipped_rounds.append(round_no)
                continue

            if round_no not in predictions:
                predictions[round_no] = load_round_prediction(round_no)

            t_eval = time.time()
            best_by_excess, best_by_cumulative, evaluated_now = evaluate_new_round(
                data_cache,
                predictions,
                evaluated_pairs,
                round_no,
                best_by_excess,
                best_by_cumulative,
            )
            log(
                f"第 {round_no} 轮新增评估 {evaluated_now} 个组合，用时 {time.time()-t_eval:.1f}s，"
                f"累计评估 {len(evaluated_pairs)} 个，最佳超额={best_by_excess['超额收益率']*100:.4f}%"
            )
            result = write_result(
                best_by_excess,
                best_by_cumulative,
                len(evaluated_pairs),
                skipped_rounds,
                time.time() - start_time,
                complete=False,
            )
            if result["best_excess_gt_100pct"]:
                log("最佳新超额收益已超过100%，停止继续扩池并生成图表")
                break

        result = write_result(
            best_by_excess,
            best_by_cumulative,
            len(evaluated_pairs),
            skipped_rounds,
            time.time() - start_time,
            complete=True,
        )
        write_best_chart(data_cache, best_by_excess)
        log("完成缓存组合筛选")
        log(json.dumps(result, ensure_ascii=False))
    except Exception:
        err = traceback.format_exc()
        log("任务失败")
        log(err)
        raise


if __name__ == "__main__":
    main()
