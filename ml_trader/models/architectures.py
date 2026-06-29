# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
torch.set_num_threads(8)
#import streamlit as st

def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
# 自定义损失函数
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true.long(), num_classes=2).float()
        return self.loss_fn(y_pred, y_true)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, X):
        X = X + self.pe[:, :X.size(1)]
        return self.dropout(X)

# Transformer 模型定义
class TransformerClassifier(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes=2, 
                 hidden_dim=2048,
                 nhead=20, 
                 num_encoder_layers=10,
                 dropout=0.1,
                 window_size=30):
        """
        改进后的 Transformer 模型，在编码器后增加额外的多头自注意力层，并加入残差连接和层归一化。
        """
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(num_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dropout=dropout,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.attn_layernorm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.float()
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_layernorm(x + attn_out)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# MLP 模型定义
import torch.nn as nn

class MLPClassifierModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5, activation=nn.ReLU):
        """
        一个可配置激活函数的 MLP 分类器模块。

        Parameters
        ----------
        input_dim : int
            输入特征维度。
        hidden_dim : int, optional (default=64)
            隐藏层神经元数量。
        output_dim : int, optional (default=2)
            输出类别数。
        dropout : float, optional (default=0.5)
            Dropout 概率。
        activation : class from torch.nn, optional (default=nn.ReLU)
            激活函数类，例如 nn.ReLU, nn.Tanh, nn.LeakyReLU。
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        mid_dim = max(hidden_dim // 2, output_dim)
        self.fc_mid = nn.Linear(hidden_dim, mid_dim)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, X):
        X = self.fc1(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc_mid(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X


# 计算类别权重
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)

def get_transformer_classifier(num_features, window_size, class_weights=None):
    if class_weights is not None:
        loss = WeightedCrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    net = NeuralNetClassifier(
        module=TransformerClassifier,
        module__num_features=num_features,
        module__window_size=window_size,
        module__hidden_dim=512,
        module__nhead=8,
        module__num_encoder_layers=3,
        module__dropout=0.1,
        max_epochs=100,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        criterion=loss,
        batch_size=128,
        train_split=None,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        warm_start=True  # 允许后续调用 partial_fit 增量训练
    )
    return ('transformer', net)


def get_mlp_classifier(input_dim, class_weights=None):
    """
    构建并返回一个基于 MLP 的二分类器（使用 skorch 封装的 PyTorch 模型）。

    Parameters
    ----------
    input_dim : int
        输入特征的维度大小，例如样本的特征数。
    class_weights : torch.Tensor or list or None, optional (default=None)
        类别权重，用于解决类别不平衡问题。
        - 如果为 None，则使用默认的均等权重。
        - 如果提供，需为 1D Tensor 或 list，对应每个类别的权重。

    Returns
    -------
    tuple
        ('mlp', net)
        - 'mlp' : str
            模型的名称标签，便于与其他模型区分。
        - net : skorch.NeuralNetClassifier
            已配置好的 MLP 分类器对象，可直接调用 `.fit(X, y)` 进行训练。

    Notes
    -----
    - 模型基于 `MLPClassifierModule`（需自行定义）。
    - 训练相关参数：
        * 隐藏层维度：128
        * Dropout 概率：0.5
        * 优化器：Adam
        * 学习率：1e-3
        * 批大小：128
        * 最大 epoch：100
    - 设备优先选择 GPU（cuda），否则使用 CPU。
    - warm_start=True 表示在多次调用 fit 时，模型不会重新初始化参数。
    """
    if class_weights is not None:
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.float()
        loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()

    net = NeuralNetClassifier(
        module=MLPClassifierModule,
        module__input_dim=input_dim,
        module__hidden_dim=128,#隐藏层维度
        module__output_dim=2,#输出类别数
        module__dropout=0.35,#dropout概率
        module__activation=nn.LeakyReLU,#激活函数，可选ReLU，Tanh，LeakyReLU
        criterion=loss,
        optimizer=torch.optim.AdamW,
        max_epochs=100,#最大迭代次数
        lr=1e-3,
        batch_size=128,#批大小
        train_split=None,
        verbose=0,#输出信息
        device='cuda' if torch.cuda.is_available() else 'cpu',
        warm_start=True
    )
    return ('mlp', net)


# 仅支持 Transformer 与 MLP，其他模型不再提供
def get_classifier(classifier_name, num_features=None, window_size=10, class_weight=None):
    if classifier_name == 'Transformer':
        if num_features is None:
            raise ValueError("num_features必须为Transformer模型指定")
        return get_transformer_classifier(num_features, window_size, class_weights=class_weight)
    elif classifier_name == 'MLP':
        if num_features is None:
            raise ValueError("num_features必须为MLP模型指定")
        return get_mlp_classifier(num_features, class_weights=class_weight)
    else:
        raise ValueError(f"未知的分类器名称: {classifier_name}. 目前仅支持 Transformer 和 MLP。")




def time_aware_oversampling(X, y, recency_weight=0.9, sequence_length=60):
    """
    时间感知的过采样方法：
      - 针对少数类（假设类别不平衡），采样出连续序列，使少数类样本数达到多数类样本数。
      - X: numpy 数组，要求已按时间顺序排列（旧数据在前，新数据在后）
      - y: 标签数组
      - recency_weight: 越接近近期数据的权重越大（建议取值 0~1）
      - sequence_length: 每次连续采样的最大长度
    返回经过过采样后的 X_os, y_os
    """
    import numpy as np

    classes, counts = np.unique(y, return_counts=True)
    # 找出少数类和多数类
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    
    X_min = X[y == minority_class]
    y_min = y[y == minority_class]
    X_maj = X[y == majority_class]
    y_maj = y[y == majority_class]

    target_size = len(y_maj)
    current_size = len(y_min)
    additional_size = target_size - current_size

    # 如果少数类样本已经足够，则直接返回原数据
    if additional_size <= 0:
        return X, y

    n = current_size
    # 构造时间权重，假设 X_min 按时间正序排列（较新的数据在后面）
    weights = np.linspace(1 - recency_weight, 1, n)

    sampled_indices = []
    while len(sampled_indices) < additional_size:
        max_start = n - sequence_length
        if max_start <= 0:
            break
        # 依据时间权重选取起始点
        start_probs = weights[:max_start] / np.sum(weights[:max_start])
        start_idx = np.random.choice(max_start, p=start_probs)
        seq_len = min(sequence_length, additional_size - len(sampled_indices))
        seq_indices = list(range(start_idx, start_idx + seq_len))
        sampled_indices.extend(seq_indices)

    sampled_indices = sampled_indices[:additional_size]
    X_sampled = X_min[sampled_indices]
    y_sampled = y_min[sampled_indices]

    # 合并补充后的少数类数据和多数类数据
    X_min_new = np.concatenate([X_min, X_sampled], axis=0)
    y_min_new = np.concatenate([y_min, y_sampled], axis=0)
    X_os = np.concatenate([X_maj, X_min_new], axis=0)
    y_os = np.concatenate([y_maj, y_min_new], axis=0)

    return X_os, y_os
