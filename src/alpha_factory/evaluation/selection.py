"""
实现对批量因子的选取与剔除功能。
基于 Polars 实现的高相关性因子剔除函数。

"""


import polars as pl
import numpy as np
from typing import List, Tuple


def batch_filter_variance_refined(
        df: pl.DataFrame,
        factor_pattern: str = r"^factor_.*",
        quantile_limit: float = 0.01,
        var_thresh: float = 1e-6
) -> pl.DataFrame:
    """
    带缩尾处理的高性能方差过滤
    1. 对各因子进行双端百分位缩尾 (Winsorize)
    2. 计算缩尾后的方差
    3. 剔除低于阈值的因子
    """

    # 1. 自动识别因子列
    factor_cols = df.select(pl.col(factor_pattern)).columns

    # 2. 构造缩尾后的方差计算表达式
    # 使用 qcut 或 quantile 找到边界，然后用 clip 限制范围
    exprs = []
    for f in factor_cols:
        # 计算该因子的上下分位数
        lower = pl.col(f).quantile(quantile_limit)
        upper = pl.col(f).quantile(1 - quantile_limit)

        # 链式操作：缩尾 -> 算方差
        exprs.append(
            pl.col(f).clip(lower, upper).var().alias(f)
        )

    # 3. 执行并行计算 (一次扫描完成所有因子)
    var_summary = df.select(exprs)

    # 4. 筛选因子：保留方差大于阈值的列
    # 将结果转为字典方便过滤
    var_dict = var_summary.to_dicts()[0]
    kept_factors = [f for f, v in var_dict.items() if v > var_thresh]
    dropped_factors = list(set(factor_cols) - set(kept_factors))

    print(f"✅ 处理完成。原始因子: {len(factor_cols)}，剔除低方差因子: {len(dropped_factors)}，剩余: {len(kept_factors)}")

    # 返回剔除后的 DataFrame (包含非因子列)
    non_factor_cols = [c for c in df.columns if c not in factor_cols]
    return df.select(non_factor_cols + kept_factors)


def filter_by_ic_metrics(df: pl.DataFrame,
                         ic_summary: pl.DataFrame,
                         min_ic: float = 0.02,
                         min_ir: float = 0.5) -> List[str]:
    """
    根据 IC 指标筛选优质因子
    :param df: 原始数据
    :param ic_summary: 由 batch_get_ic_summary 计算出的统计表
    :param min_ic: IC 均值的最小绝对值
    :param min_ir: ICIR 的最小值
    :return: 保留的因子列表
    """

    # 筛选逻辑：绝对值 IC 够大 且 稳定性够好
    qualified_factors = ic_summary.filter(
        (pl.col("ic_mean").abs() >= min_ic) &
        (pl.col("ic_ir") >= min_ir)
    ).get_column("factor").to_list()

    print(f"筛选前因子数: {ic_summary.height}")
    print(f"筛选后因子数: {len(qualified_factors)}")

    return qualified_factors


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
