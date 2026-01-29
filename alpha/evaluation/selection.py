"""
实现对批量因子的选取与剔除功能。
基于 Polars 实现的高相关性因子剔除函数。

"""


import polars as pl
import numpy as np
from typing import List, Tuple


def drop_above_corr_thresh_polars(df: pl.DataFrame, thresh: float = 0.85) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    基于 Polars 实现的高相关性因子剔除函数。
    对于任意一对相关系数超过阈值的因子，
    保留全局相关性较低的那个，剔除另一个。
    :param df:
    :param thresh:
    :return:
    """

    # 1. 计算相关性矩阵并转为 NumPy
    cols = df.columns
    # df.corr() 在 Polars 中返回一个各列相互关联的 DataFrame
    corr_mat = df.corr().to_numpy()

    # 2. 计算每个因子与其他因子的绝对相关性之和 (向量化运算)
    # axis=0 即对每一列求和，得到每个因子的“冗余度得分”
    abs_corr_sums = np.sum(np.abs(corr_mat), axis=0)

    # 3. 提取上三角部分（剔除对角线），寻找超过阈值的坐标
    # k=1 表示对角线向上偏移 1，避开自己与自己相关(1.0)
    upper_tri = np.triu(corr_mat, k=1)
    rows, g_cols = np.where(np.abs(upper_tri) > thresh)

    cols_to_drop = set()
    above_thresh_pairs = []

    # 4. 只针对超过阈值的对进行逻辑判断
    for r, c in zip(rows, g_cols):
        col_a, col_b = cols[r], cols[c]
        val = corr_mat[r, c]
        above_thresh_pairs.append((col_a, col_b, val))

        # 核心逻辑：谁的全局相关性更高（更冗余），就删谁
        if abs_corr_sums[r] > abs_corr_sums[c]:
            cols_to_drop.add(col_a)
        else:
            cols_to_drop.add(col_b)

    return list(cols_to_drop), above_thresh_pairs
