import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import (
    mutual_info_classif, chi2, f_classif
)
from sklearn.preprocessing import MinMaxScaler

def filter_features_by_pearson(X: pd.DataFrame, 
                               y: pd.Series, 
                               threshold: float = 0.1, 
                               drop_abs_below: bool = True) -> list:
    """
    使用皮尔逊相关系数来筛选特征。
    
    参数：
    -------
    X : pd.DataFrame
        特征矩阵，行对应样本，列对应特征
    y : pd.Series
        目标向量，需可被视作连续数值或 0/1 整数
    threshold : float
        阈值，默认 0.1；可根据需要调整
    drop_abs_below : bool
        如果为 True，则只保留 |corr| >= threshold 的特征；即绝对值小于 threshold 的特征会被剔除。
        如果为 False，则保留 |corr| <= threshold 的特征，效果相反。
        
    返回：
    -------
    selected_features : list
        筛选后保留的特征名称
    """
    # 将目标 y 也拼接进来，方便直接用 corr() 一次性算出相关系数
    df_tmp = X.copy()
    df_tmp['__TARGET__'] = y.values
    
    # 计算每个特征与目标的皮尔逊相关系数
    corr_with_target = df_tmp.corr(method='pearson')['__TARGET__'].drop(labels=['__TARGET__'])  # 只保留特征列
    
    # 根据阈值进行过滤
    if drop_abs_below:
        # 保留绝对值 >= threshold 的特征
        selected = corr_with_target[abs(corr_with_target) >= threshold].index
    else:
        # 保留绝对值 <= threshold 的特征
        selected = corr_with_target[abs(corr_with_target) <= threshold].index
        
    return list(selected)

def filter_features_by_spearman(X: pd.DataFrame, 
                                y: pd.Series, 
                                threshold: float = 0.1, 
                                drop_abs_below: bool = True) -> list:
    """
    使用斯皮尔曼相关系数来筛选特征。
    
    参数同上，不再赘述。
    """
    df_tmp = X.copy()
    df_tmp['__TARGET__'] = y.values
    
    corr_with_target = df_tmp.corr(method='spearman')['__TARGET__'].drop(labels=['__TARGET__'])
    
    if drop_abs_below:
        selected = corr_with_target[abs(corr_with_target) >= threshold].index
    else:
        selected = corr_with_target[abs(corr_with_target) <= threshold].index
    return list(selected)

from sklearn.feature_selection import mutual_info_classif

def filter_features_by_mutual_info(X: pd.DataFrame, 
                                   y: pd.Series, 
                                   top_k: int = None,
                                   threshold: float = 0.0) -> list:
    """
    使用互信息 (Mutual Information) 来对特征进行评分并筛选。
    默认用于分类场景 (可以改用 mutual_info_regression 用于回归)。
    
    参数：
    -------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标向量（分类）
    top_k : int
        如果指定，则只保留互信息评分最高的 top_k 个特征
    threshold : float
        如果指定一个互信息下限，只保留互信息评分 >= threshold 的特征。
        注意互信息计算结果通常比较小（例如 0~0.5），需要结合经验设置。
        
    返回：
    -------
    selected_features : list
        保留的特征名称
    """
    # 计算互信息
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_mi = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    if top_k is not None:
        # 先根据互信息得分排序，取前 top_k
        selected = feature_mi.head(top_k).index
        return list(selected)
    else:
        # 根据阈值
        selected = feature_mi[feature_mi >= threshold].index
        return list(selected)

from sklearn.feature_selection import chi2

def filter_features_by_chi2(X: pd.DataFrame, 
                            y: pd.Series, 
                            top_k: int = None, 
                            pval_threshold: float = None) -> list:
    """
    使用卡方检验来对特征进行筛选。
    
    参数：
    -------
    X : pd.DataFrame
        特征矩阵（要求非负，如计数/频次/独热编码后的数据）
    y : pd.Series
        目标向量（分类）
    top_k : int
        若指定，则保留 chi2 检验统计量最高的前 top_k 个特征
    pval_threshold : float
        若指定 p 值阈值，则保留卡方检验 p 值 < pval_threshold 的特征
        
    返回：
    -------
    selected_features : list
        保留的特征名称
    """
    chi2_vals, p_vals = chi2(X, y)
    df_chi = pd.DataFrame({"feature": X.columns, "chi2": chi2_vals, "pval": p_vals})
    df_chi.sort_values(by="chi2", ascending=False, inplace=True)
    
    if top_k is not None:
        selected = df_chi.head(top_k)['feature']
        return list(selected)
    elif pval_threshold is not None:
        selected = df_chi[df_chi["pval"] < pval_threshold]['feature']
        return list(selected)
    else:
        # 如果不指定 top_k 或 pval_threshold，就直接返回原顺序
        return list(df_chi['feature'])

from sklearn.feature_selection import f_classif, SelectKBest

def filter_features_by_f_classif(X: pd.DataFrame, 
                                 y: pd.Series, 
                                 k: int = 10) -> list:
    """
    使用单变量 F 检验 (ANOVA) 来筛选特征，用于分类任务。
    
    参数：
    -------
    X : pd.DataFrame
        连续型特征矩阵
    y : pd.Series
        分类目标
    k : int
        保留的特征数量
        
    返回：
    -------
    selected_features : list
        保留的特征名称
    """
    # SelectKBest 会筛选出得分最高的 k 个特征
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    support_mask = selector.get_support()  # 布尔数组
    selected = X.columns[support_mask]
    
    return list(selected)


def select_top_n_features_tree(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
    """
    使用随机森林进行特征重要度排序，并选出重要度最高的 n_features 个特征。
    如果 n_features 大于当前特征总数，则返回全部特征。
    """
    rf_for_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_for_fs.fit(X, y)

    importances = rf_for_fs.feature_importances_  # 长度与 X.shape[1] 相同
    feature_names = X.columns                     # 保证 X 还是 DataFrame

    # 根据重要度从大到小排序
    indices = np.argsort(importances)[::-1]

    # 防止 n_features 超过实际特征数
    n_features = min(n_features, len(feature_names))

    top_indices = indices[:n_features]
    top_features = feature_names[top_indices]
    return list(top_features)



def auto_select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_candidates=[5, 10, 15, 20, 30], 
    scoring='f1'
) -> list:
    """
    在 n_candidates 里自动搜索最优特征数 n，并返回排名前 n 的最重要特征。
    使用随机森林对所有特征做初步特征重要度排序，然后对每个候选 n 做一次 3 折交叉验证，
    并将每次 fold 的得分、以及平均得分都打印出来。
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score

    #print("[auto_select_features] >> Start searching best n from", n_candidates)

    # 第一步：用一个固定的随机森林获取所有特征的重要度
    rf_for_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_for_fs.fit(X, y)
    importances = rf_for_fs.feature_importances_
    feature_names = X.columns

    # 从大到小排序
    indices = np.argsort(importances)[::-1]

    # 准备交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_n = None
    best_score = -1.0

    # 遍历 [5, 10, 15, 20, 30]
    for n in n_candidates:
        n_final = min(n, len(feature_names))
        top_indices = indices[:n_final]
        top_feats = feature_names[top_indices]

        cv_scores = []
        # 对 n 做 3 折交叉验证
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_cv = X.iloc[train_idx][top_feats]
            X_test_cv  = X.iloc[test_idx][top_feats]
            y_train_cv = y.iloc[train_idx]
            y_test_cv  = y.iloc[test_idx]

            rf_cv = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = rf_cv.predict(X_test_cv)

            if scoring == 'f1':
                score_val = f1_score(y_test_cv, y_pred_cv)
            elif scoring == 'precision':
                score_val = precision_score(y_test_cv, y_pred_cv)
            elif scoring == 'recall':
                score_val = recall_score(y_test_cv, y_pred_cv)
            else:
                # 缺省使用 f1
                score_val = f1_score(y_test_cv, y_pred_cv)

            #print(f"[auto_select_features] n={n}, fold={fold_index+1}, score={score_val:.4f}")
            cv_scores.append(score_val)

        mean_score = np.mean(cv_scores)
        #print(f"[auto_select_features] n={n}, mean_score={mean_score:.4f}\n")

        # 若当前 mean_score 严格大于已有 best_score，则更新
        if mean_score > best_score:
            best_score = mean_score
            best_n = n

    # 如果所有 n 都得分 0，best_score 依旧会是 0，best_n 则是第一个达到 0 的 n
    if best_n is None:
        #print("[auto_select_features] 未找到最优 n, 返回全部特征")
        return list(feature_names)

    # 最终保留前 best_n 个
    top_indices_final = indices[:best_n]
    top_features_final = feature_names[top_indices_final]

    #print(f"[auto_select_features] 最优 n = {best_n}, CV({scoring}) = {best_score:.4f}")
    return list(top_features_final)


def filter_features(
    X: pd.DataFrame, 
    y: pd.Series,
    method: str = "pearson",
    n_features: int = None,
    pval_threshold: float = None
) -> list:
    """
    统一的特征筛选函数，返回按“重要性从高到低”排列的特征名称列表。
    
    可选 method:
      1) 'pearson'       => 皮尔逊相关系数（绝对值越大越重要）
      2) 'spearman'      => 斯皮尔曼相关系数（绝对值越大越重要）
      3) 'mutual_info'   => 互信息 (分类)
      4) 'chi2'          => 卡方检验 (分类, X需要非负)
      5) 'anova_f'       => 单因素方差分析 F检验 (分类)
      6) 'rftree'        => 随机森林特征重要度 (分类)

    参数:
    -------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标向量
    method : str
        筛选方法
    n_features : int
        最终想要保留的特征个数。如果不指定 (或为 None)，则返回全部已排序的特征。
    pval_threshold : float
        仅对 'chi2' 生效。若指定则先保留 p值 < pval_threshold 的特征，再做降序截断。
    
    返回:
    -------
    selected_features : list
        最终保留的特征名称，按重要性/相关性从高到低排列
    """

    # 统一转小写，防止大小写混用
    method = method.lower()

    score_series = None  # "特征 => 分数"

    if method in ["pearson", "spearman"]:
        # 将 y 合并进临时 df，计算相关系数
        df_tmp = X.copy()
        df_tmp['__TARGET__'] = y.values
        corr_with_target = df_tmp.corr(
            method=('pearson' if method == 'pearson' else 'spearman')
        )['__TARGET__'].drop(labels=['__TARGET__'])
        
        # 将绝对值相关系数作为“得分”，进行降序
        score_series = corr_with_target.abs().sort_values(ascending=False)

        # 如果指定了 n_features，则截断
        if n_features is not None:
            score_series = score_series.iloc[:n_features]

    elif method == "mutual_info":
        mi_scores = mutual_info_classif(X, y, random_state=42)
        score_series = pd.Series(mi_scores, index=X.columns)
        # 从大到小排序
        score_series.sort_values(ascending=False, inplace=True)

        if n_features is not None:
            score_series = score_series.iloc[:n_features]

    elif method == "chi2":
        chi2_vals, p_vals = chi2(X, y)
        df_chi = pd.DataFrame({
            "feature": X.columns, 
            "chi2": chi2_vals, 
            "pval": p_vals
        })
        # 先按 chi2 值降序
        df_chi.sort_values(by="chi2", ascending=False, inplace=True)

        # 若给定 pval_threshold，则先用 p 值筛掉不显著的特征
        if pval_threshold is not None:
            df_chi = df_chi[df_chi["pval"] < pval_threshold]

        # 如果指定了 n_features，则对剩余特征进行截断
        if n_features is not None:
            df_chi = df_chi.head(n_features)

        # 最终将 chi2 值作为“分数”
        score_series = pd.Series(df_chi["chi2"].values, index=df_chi["feature"])

    elif method == "anova_f":
        # ANOVA F检验
        fvals, pvals = f_classif(X, y)
        df_f = pd.DataFrame({
            "feature": X.columns,
            "fval": fvals,
            "pval": pvals
        })
        df_f.sort_values(by="fval", ascending=False, inplace=True)

        if n_features is not None:
            df_f = df_f.head(n_features)

        score_series = pd.Series(df_f["fval"].values, index=df_f["feature"])

    elif method == "rftree":
        # 随机森林
        rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_fs.fit(X, y)
        importances = rf_fs.feature_importances_

        score_series = pd.Series(importances, index=X.columns)
        score_series.sort_values(ascending=False, inplace=True)

        if n_features is not None:
            score_series = score_series.iloc[:n_features]

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Please choose from ['pearson','spearman','mutual_info','chi2','anova_f','rftree']."
        )

    # 如果没有得分或都被过滤没了，直接返回空列表
    if score_series is None or score_series.empty:
        return []

    # 确保最后返回时是按分数降序排列
    score_series = score_series.sort_values(ascending=False)

    return list(score_series.index)