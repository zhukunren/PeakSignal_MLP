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

from preprocess import create_pos_neg_sequences_by_consecutive_labels
from models import get_transformer_classifier, get_mlp_classifier
from models import time_aware_oversampling

from filter_feature import filter_features

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


def train_model_for_label(
    df: pd.DataFrame,
    N: int,
    label_column: str,
    all_features: list,
    classifier_name: str,
    n_features_selected,
    window_size: int = 10,
    oversample_method: str = 'SMOTE',
    class_weight=None
):
    print(f"\n=== 开始训练 {label_column} 模型 ===")
    data = df.copy()
    if os.getenv("EXPORT_TRAINING_DATASET") == "1":
        data.to_csv("简化版训练集.csv", index=False)
    X = data[all_features]
    y = data[label_column].astype(np.int64)
    print(f"{label_column} 训练正样本: {int(y.sum())}")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"检测到高相关特征 {len(to_drop)} 个，将进行剔除。")
    else:
        print("未检测到高相关特征。")

    all_features_filtered = [f for f in all_features if f not in to_drop]
    X = X[all_features_filtered].fillna(0)
    print(f"相关性过滤后特征数量: {len(all_features_filtered)}")

    if n_features_selected == 'auto':
        selected_features = all_features_filtered
        print(f"[自动筛选] 实际保留 {len(selected_features)} 个特征。")
    elif isinstance(n_features_selected, int):
        print(f"[手动指定] 使用随机森林保留前 {n_features_selected} 个特征...")
        top_feats = filter_features(X, y, method='pearson', n_features=n_features_selected)
        selected_features = top_feats
        print(f"[手动指定] 保留特征数量: {len(selected_features)}")
    else:
        selected_features = all_features_filtered
        print("n_features_selected 参数无效，直接使用相关性过滤后的特征。")

    X = X[selected_features].fillna(0)
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if classifier_name == 'Transformer':
        print("构造 Transformer 时序数据集...")
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
        print("准备训练固定参数 XGBoost 模型...")
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

        print(f"XGBoost random_state={random_state}, scale_pos_weight={scale_pos_weight:.2f}")
        print("\n=== 评估结果 ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(confusion_matrix(y_test, y_pred))
        print(f"ROC AUC: {roc_value:.4f}, PR AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

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
        print("准备对 MLP 做过采样处理...")
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

    print(f"开始网格搜索... scoring={scoring_used}")
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
    print(f"最佳参数: {grid_search.best_params_}, 最佳得分: {grid_search.best_score_:.4f}")

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

    print("\n=== 评估结果 ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_value:.4f}, PR AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

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
    window_size: int = 10
):
    print("开始训练模型...")
    data = df_preprocessed.copy()
    labels = ['Peak', 'Trough']

    # 训练第一个模型 (Peak)
    print("训练 Peak 模型...")
    peak_results = train_model_for_label(
        data, N, 'Peak', all_features, classifier_name, n_features_selected, window_size, oversample_method,
        class_weight='balanced' if oversample_method == 'Class Weights' else None
    )

    # 训练第二个模型 (Trough)
    print("训练 Trough 模型...")
    trough_results = train_model_for_label(
        data, N, 'Trough', all_features, classifier_name, n_features_selected, window_size, oversample_method,
        class_weight='balanced' if oversample_method == 'Class Weights' else None
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

