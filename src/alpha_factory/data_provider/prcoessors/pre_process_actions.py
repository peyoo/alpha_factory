import polars as pl
from polars_ta.wq import (
    cs_mad_zscore_resid,
    cs_mad_zscore_resid_zscore,
    cs_resid,
)


def cs_resid_log_mv(x):
    if x.name == "TOTAL_MV":
        return x
    return cs_resid(x, pl.col("LOG_MV"))


def my_cs_mad_zscore_resid(x):
    return cs_mad_zscore_resid(x, pl.col("LOG_MV"))


def my_cs_mad_zscore_resid_zscore(x):
    return cs_mad_zscore_resid_zscore(x, pl.col("LOG_MV"))


def cs_rank_norm(x: pl.Expr) -> pl.Expr:
    """
    高阶正态化变换：Rank -> Uniform -> Normal
    1. 计算百分比排名 (0, 1)
    2. 使用近似公式将 (0, 1) 映射到标准正态分布的 Z 值
    """
    # 计算秩 (0.5 / count 是为了避免 0 和 1 导致分位数函数爆炸)
    count = x.count()
    rank_pct = (x.rank() - 0.5) / count

    # 使用近似公式实现正态分布的反函数 (Inverse CDF)
    # 或者如果你有 scipy，可以用 pl.map_batches 配合 norm.ppf
    # 简单的替代方案：对 rank_pct 做 cs_zscore
    return (rank_pct - rank_pct.mean()) / rank_pct.std()


def cs_rank_zscore(x: pl.Expr) -> pl.Expr:
    """
    简单组合：先 Rank 化，再 Z-Score 化
    """
    # 1. 变成 [-0.5, 0.5] 的均匀分布
    r = (x.rank() - 1) / (x.count() - 1) - 0.5
    # 2. 变成 均值0, 方差1 的分布
    return (r - r.mean()) / r.std()
