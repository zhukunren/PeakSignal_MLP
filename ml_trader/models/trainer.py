import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score, recall_score, average_precision_score,
    matthews_corrcoef, roc_auc_score, f1_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from joblib import Parallel, delayed, parallel_backend

from ml_trader.data.preprocessor import create_pos_neg_sequences_by_consecutive_labels
from ml_trader.models.architectures import get_transformer_classifier, get_mlp_classifier
from ml_trader.models.architectures import time_aware_oversampling

from ml_trader.config_loader import get_config
from ml_trader.features.feature_groups import (
    dedupe_preserve_order,
    get_label_feature_candidates,
    get_label_selection_config,
)
from ml_trader.features.selector import filter_features
from ml_trader.logging_config import get_logger


logger = get_logger(__name__)

def identity_transform(x):
    return x


def optimize_threshold(y_true, y_proba, metric='precision'):
    best_thresh = 0.5
    best_score = -1
    for thresh in np.linspace(0, 1, 101):
        y_pred_temp = (y_proba > thresh).astype(int)
        if metric == 'precision':
            score = precision_score(y_true, y_pred_temp)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred_temp)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_temp)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred_temp)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred_temp)
        else:
            raise ValueError("metric must be one of 'precision', 'f1', 'recall', 'accuracy', 'mcc'")
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh


def expand_labels_to_next_turn(y: pd.Series) -> pd.Series:
    """
    训练目标从“今天就是峰/谷”改为“今天或下个交易日会出现峰/谷”。
    回测仍使用模型预测信号，不直接使用真实标签；这里仅扩大可学习的拐点前沿样本。
    """
    y_int = y.astype(np.int64)
    return ((y_int == 1) | (y_int.shift(-1, fill_value=0) == 1)).astype(np.int64)


def _rank_features(X: pd.DataFrame, y: pd.Series, features: list, method: str, n_features=None) -> list:
    if not features:
        return []

    n_final = min(n_features, len(features)) if isinstance(n_features, int) else None
    try:
        ranked = filter_features(X[features].fillna(0), y, method=method, n_features=n_final)
        ranked = [feature for feature in ranked if feature in features]
    except Exception as exc:
        logger.exception("Feature ranking failed; keeping candidate order: %s", exc)
        ranked = []

    if not ranked:
        ranked = features.copy()

    return ranked[:n_final] if n_final is not None else ranked


def select_features_for_label(
    X: pd.DataFrame,
    y: pd.Series,
    label_column: str,
    available_features: list,
    n_features_selected,
    selection_config=None,
) -> list:
    """
    Select features with label-aware preferred groups.

    ``legacy`` mode preserves the old behavior. ``hybrid`` keeps a minimum
    number of label-preferred features and fills the rest from global ranking.
    """
    selection_config = get_config("features.selection", {}) if selection_config is None else selection_config
    label_config = get_label_selection_config(selection_config, label_column)
    mode = label_config.get("mode", selection_config.get("default_mode", "legacy"))
    method = label_config.get("method", selection_config.get("method", "pearson"))

    if isinstance(n_features_selected, int):
        target_count = min(n_features_selected, len(available_features))
    elif n_features_selected == "auto":
        target_count = min(int(label_config.get("max_features", len(available_features))), len(available_features))
    else:
        target_count = len(available_features)

    if target_count <= 0:
        target_count = len(available_features)

    if mode == "legacy":
        if n_features_selected == "auto":
            return available_features.copy()
        if isinstance(n_features_selected, int):
            return _rank_features(X, y, available_features, method, target_count)
        return available_features.copy()

    preferred_features = get_label_feature_candidates(label_column, available_features, selection_config)
    if not preferred_features:
        preferred_features = available_features.copy()

    if mode == "preferred_only":
        selected = _rank_features(X, y, preferred_features, method, target_count)
        return selected or preferred_features[:target_count]

    if mode != "hybrid":
        logger.warning("Unknown feature selection mode %s; falling back to legacy", mode)
        return select_features_for_label(
            X,
            y,
            label_column,
            available_features,
            n_features_selected,
            selection_config={"default_mode": "legacy", "method": method},
        )

    min_preferred = int(label_config.get("min_preferred_features", min(10, target_count)))
    min_preferred = min(min_preferred, target_count, len(preferred_features))
    preferred_ranked = _rank_features(X, y, preferred_features, method, min_preferred)
    global_ranked = _rank_features(X, y, available_features, method, target_count)
    selected = dedupe_preserve_order(preferred_ranked + global_ranked)

    if len(selected) < target_count:
        selected = dedupe_preserve_order(selected + available_features)

    return selected[:target_count]


def train_model_for_label(
    df: pd.DataFrame,
    N: int,
    label_column: str,
    all_features: list,
    classifier_name: str,
    n_features_selected,
    window_size: int = 10,
    oversample_method: str = 'SMOTE',
    class_weight=None,
    feature_selection_config=None,
):
    logger.info("Training %s model started", label_column)
    data = df.copy()
    if os.getenv("EXPORT_TRAINING_DATASET") == "1":
        data.to_csv("简化版训练集.csv", index=False)
    X = data[all_features]
    y = data[label_column].astype(np.int64)
    logger.info("%s positive samples: %s", label_column, int(y.sum()))

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        logger.info("Dropping highly correlated features: label=%s count=%s", label_column, len(to_drop))
    else:
        logger.info("No highly correlated features detected: label=%s", label_column)

    all_features_filtered = [f for f in all_features if f not in to_drop]
    X = X[all_features_filtered].fillna(0)
    logger.info("Feature count after correlation filter: label=%s count=%s", label_column, len(all_features_filtered))

    selected_features = select_features_for_label(
        X,
        y,
        label_column,
        all_features_filtered,
        n_features_selected,
        selection_config=feature_selection_config,
    )
    logger.info("%s selected features: count=%s first10=%s", label_column, len(selected_features), selected_features[:10])

    X = X[selected_features].fillna(0)
    logger.info("Scaling training data: label=%s", label_column)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if classifier_name == 'Transformer':
        logger.info("Building Transformer sequence dataset: label=%s", label_column)
        X_seq, y_seq = create_pos_neg_sequences_by_consecutive_labels(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        model_name, model = get_transformer_classifier(
            num_features=X_train.shape[-1],
            window_size=window_size,
            class_weights=None
        )
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [10]
        }
        scoring_used = 'precision'
    elif classifier_name in ['XGBoost', 'XGB']:
        logger.info("Training fixed-parameter XGBoost model: label=%s", label_column)
        from xgboost import XGBClassifier

        y_array = y.to_numpy(dtype=np.int64)
        split_index = int(len(X_scaled) * 0.8)
        if y_array[:split_index].sum() == 0 or y_array[split_index:].sum() == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_array, test_size=0.2, random_state=42, stratify=y_array
            )
        else:
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y_array[:split_index], y_array[split_index:]

        neg_count = max(int((y_train == 0).sum()), 1)
        pos_count = max(int((y_train == 1).sum()), 1)
        scale_pos_weight = neg_count / pos_count
        random_state = int(np.random.randint(0, 1_000_000))

        best_estimator = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=3.0,
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            random_state=random_state,
            n_jobs=1,
        )
        best_estimator.fit(X_train, y_train)
        scoring_used = 'f1'
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        best_thresh = optimize_threshold(y_test, y_proba, metric=scoring_used)
        y_pred = (y_proba > best_thresh).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_value = roc_auc_score(y_test, y_proba)

        logger.info("XGBoost random_state=%s scale_pos_weight=%.2f", random_state, scale_pos_weight)
        logger.info("Classification report for %s:\n%s", label_column, classification_report(y_test, y_pred, zero_division=0))
        logger.info("Confusion matrix for %s:\n%s", label_column, confusion_matrix(y_test, y_pred))
        logger.info("Metrics for %s: ROC AUC=%.4f PR AUC=%.4f MCC=%.4f", label_column, roc_value, pr_auc, mcc)

        metrics = {
            'ROC AUC': roc_value,
            'PR AUC': pr_auc,
            'Precision': precision,
            'Recall': recall,
            'MCC': mcc
        }

        return (
            best_estimator, scaler, FunctionTransformer(func=identity_transform), selected_features,
            all_features_filtered, f1_score(y_test, y_pred, zero_division=0), metrics, best_thresh
        )
    else:
        logger.info("Preparing oversampling for MLP: label=%s method=%s", label_column, oversample_method)
        sampler = None
        if oversample_method == 'SMOTE':
            sampler = SMOTE(random_state=42, k_neighbors=3)
        elif oversample_method == 'ADASYN':
            sampler = ADASYN(random_state=42, n_neighbors=3)
        elif oversample_method == 'Borderline-SMOTE':
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=3)
        elif oversample_method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=3))
        elif oversample_method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=3))
        elif oversample_method == 'Time-Aware':
            X_os, y_os = time_aware_oversampling(X_scaled, y, recency_weight=0.7, sequence_length=60)
        elif oversample_method in ['Class Weights', 'None']:
            sampler = None
        else:
            raise ValueError(f"未知过采样: {oversample_method}")

        if sampler is not None:
            X_os, y_os = sampler.fit_resample(X_scaled, y)
        else:
            X_os, y_os = X_scaled, y

        X_train, X_test, y_train, y_test = train_test_split(
            X_os, y_os, test_size=0.5, random_state=42, stratify=y_os
        )

        model_name, model = get_mlp_classifier(
            input_dim=X_train.shape[-1],
            class_weights=None
        )
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [20]
        }
        scoring_used = 'f1'

    logger.info("Starting grid search: label=%s scoring=%s", label_column, scoring_used)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=1,
        scoring=scoring_used,
        verbose=0,
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    logger.info(
        "Grid search completed: label=%s best_params=%s best_score=%.4f",
        label_column,
        grid_search.best_params_,
        grid_search.best_score_,
    )

    y_proba = best_estimator.predict_proba(X_test)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    else:
        y_proba = F.softmax(torch.tensor(y_proba), dim=1)[:, 1].numpy()

    best_thresh = optimize_threshold(y_test, y_proba, metric=scoring_used)
    y_pred = (y_proba > best_thresh).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_value = roc_auc_score(y_test, y_proba)

    logger.info("Classification report for %s:\n%s", label_column, classification_report(y_test, y_pred))
    logger.info("Confusion matrix for %s:\n%s", label_column, confusion_matrix(y_test, y_pred))
    logger.info("Metrics for %s: ROC AUC=%.4f PR AUC=%.4f MCC=%.4f", label_column, roc_value, pr_auc, mcc)

    metrics = {
        'ROC AUC': roc_value,
        'PR AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }

    return (
        best_estimator, scaler, FunctionTransformer(func=identity_transform), selected_features,
        all_features_filtered, grid_search.best_score_, metrics, best_thresh
    )

def train_model(
    df_preprocessed: pd.DataFrame,
    N: int,
    all_features: list,
    classifier_name: str,
    mixture_depth: int,
    n_features_selected,
    oversample_method: str,
    window_size: int = 10,
    feature_selection_config=None,
):
    logger.info(
        "Training peak/trough models started: rows=%s features=%s N=%s classifier=%s mixture_depth=%s oversample=%s",
        len(df_preprocessed),
        len(all_features),
        N,
        classifier_name,
        mixture_depth,
        oversample_method,
    )
    data = df_preprocessed.copy()
    labels = ['Peak', 'Trough']

    # 训练第一个模型 (Peak)
    logger.info("Training Peak model")
    peak_results = train_model_for_label(
        data, N, 'Peak', all_features, classifier_name, n_features_selected, window_size, oversample_method,
        class_weight='balanced' if oversample_method == 'Class Weights' else None,
        feature_selection_config=feature_selection_config,
    )

    # 训练第二个模型 (Trough)
    logger.info("Training Trough model")
    trough_results = train_model_for_label(
        data, N, 'Trough', all_features, classifier_name, n_features_selected, window_size, oversample_method,
        class_weight='balanced' if oversample_method == 'Class Weights' else None,
        feature_selection_config=feature_selection_config,
    )

    # 解包结果
    (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak,
     peak_best_score, peak_metrics, peak_threshold) = peak_results

    (trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough,
     trough_best_score, trough_metrics, trough_threshold) = trough_results

    return (
        peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score, peak_metrics, peak_threshold,
        trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_best_score, trough_metrics, trough_threshold
    )

