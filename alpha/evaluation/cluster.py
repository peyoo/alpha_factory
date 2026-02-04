import polars as pl
import polars.selectors as cs
import numpy as np
import fastcluster
from loguru import logger
from typing import Union, List
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


def batch_clustering(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        threshold: float = 0.8,
        method: str = "average"
) -> dict:
    """
    因子聚类分析核心函数
    :param df: 输入数据 (LazyFrame 或 DataFrame)
    :param factors: 因子列选择器
    :param threshold: 相关性阈值 (0-1)，越高表示聚类要求越苛刻
    :param method: 联动算法 ('average' 推荐用于因子分析, 'complete' 较严苛)
    """
    # 1. 提取因子列并进入内存
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)

    # 仅 collect 必要的因子列
    factor_df = lf.select(selector).collect()

    if factor_df.width < 1:
        return {}
    if factor_df.width == 1:
        return {factor_df.columns[0]: 1}

    # 2. 数据清洗：处理常数因子和空值
    # 剔除方差为 0 的因子
    stats = factor_df.std()
    valid_cols = [col for col in factor_df.columns if (stats.get_column(col)[0] or 0) > 1e-9]

    if not valid_cols:
        return {col: 0 for col in factor_df.columns}

    # 【关键】剔除 Null 值后再计算相关性。
    # 均值算子在头部有 null，不剔除会导致 corr 结果为 0 或 NaN
    cleaned_df = factor_df.select(valid_cols).drop_nulls()

    if cleaned_df.height < 5:
        logger.warning("有效数据行数过少（可能由于因子回看窗口过长），无法有效聚类")
        return {col: i + 1 for i, col in enumerate(factor_df.columns)}

    # 3. 计算相关性矩阵 (采用 Rank 变换以计算 Spearman 相关性，对异常值更稳健)
    # 这一步能解决 Pearson 对换手率等偏态分布数据不敏感的问题
    corr_matrix = cleaned_df.select([pl.col(c).rank() for c in valid_cols]).corr().to_numpy()
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 调试信息
    if corr_matrix.shape[0] >= 2:
        logger.debug(f"Factor_1与Factor_2相关性(秩): {corr_matrix[0, 1]:.4f}")

    # 4. 构建距离矩阵
    # 使用 1 - abs(rho) 作为基础。若想更严谨可用 sqrt(2*(1-rho))
    dist_matrix = np.clip(1 - np.abs(corr_matrix), 0, 1)
    np.fill_diagonal(dist_matrix, 0)

    # 5. 执行分层聚类
    dist_vec = squareform(dist_matrix)
    Z = fastcluster.linkage(dist_vec, method=method)

    # 截断阈值：距离 t = 1 - threshold
    labels = fcluster(Z, t=1 - threshold, criterion='distance')

    # 6. 结果映射
    result = {col: 0 for col in factor_df.columns}
    for col, label in zip(valid_cols, labels):
        result[col] = int(label)

    return result
