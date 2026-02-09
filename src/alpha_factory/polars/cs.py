import polars as pl
from polars import when

# Use new src layout package paths
from alpha_factory.utils.schema import F


def cs_mad_zscore_mask(x: pl.Expr, mask: pl.Expr = pl.col(F.POOL_MASK),fill_null=0.0) -> pl.Expr:
    """
    在指定的 mask 范围内进行 MAD 去极值和标准化
    """

    # 1. 过滤池外数据，使其不参与统计量的计算
    # 使用 .val() 或直接传入的 Expr
    if mask is None:
        mask = pl.lit(True)
    x_in_pool = when(mask).then(x).otherwise(None)

    # 2. 计算基于池内统计量的标准化值
    # 由于 codegen 会自动补 over("DATE")，这里保持逻辑简洁
    a = x_in_pool.median()
    # MAD 计算
    mad = (x_in_pool - a).abs().median()
    # 3. sigma 裁剪逻辑 (3 * 1.4826)
    x_clipped = x.clip(lower_bound=a - 4.4478 * mad, upper_bound=a + 4.4478 * mad)

    # 4. 二次标准化并填充
    res = (x_clipped - x_in_pool.mean()) / x_in_pool.std(ddof=0)

    # 5. 重点：填充 0。这样外层算子 (+-*/) 遇到池外数据时不会崩溃
    return res.fill_null(fill_null)


def cs_rank_mask(x: pl.Expr, mask: pl.Expr = pl.col(F.POOL_MASK), fill_null=0.5) -> pl.Expr:
    """
    在指定的 mask 范围内进行横截面排名标准化
    """

    # 1. 过滤池外数据，使其不参与排名计算
    # 池外数据设为 None，计算 rank 时 polars 默认会将 null 排在最后或忽略（取决于版本）
    if mask is None:
        mask = pl.lit(True)
    x_in_pool = when(mask).then(x).otherwise(None)

    # 2. 计算排名 (这里使用百分比排名)
    # .rank() 之后除以有效样本数，将值域缩放到 [0, 1]
    # 注意：count() 必须也是基于 mask 的有效计数
    rank_val = x_in_pool.rank(method="average")
    valid_count = mask.cast(pl.UInt32).sum()

    # 缩放排名：(rank - 1) / (n - 1) 使其严格落在 [0, 1]
    # 或者简单使用 rank / n，取决于你对 0 和 1 边界的处理偏好
    res = (rank_val - 1) / (valid_count - 1 + 1e-6)

    # 3. 填充中性值
    # 对于 Rank 来说，中性值是 0.5（代表截面最中间）
    return when(mask).then(res).otherwise(fill_null)


def cs_demean_mask(x: pl.Expr, mask: pl.Expr = pl.col(F.POOL_MASK), fill_null=0.0) -> pl.Expr:
    """截面去均值：x - mean(x_in_pool)"""
    if mask is None:
        mask = pl.lit(True)
    x_in_pool = when(mask).then(x).otherwise(None)

    # 计算池内均值
    mean_val = x_in_pool.mean()

    # 结果：池内标的减去均值，池外标的赋 0
    res = x - mean_val
    return when(mask).then(res).otherwise(fill_null)


def cs_qcut_mask(x: pl.Expr, n_bins: int = 10, mask: pl.Expr = pl.col(F.POOL_MASK), fill_null=0.0) -> pl.Expr:
    """截面等频分箱：将池内数据映射到 [0, n_bins-1]"""
    # 逻辑同 rank，因为 qcut 本质上是对 rank 的切分
    if mask is None:
        mask = pl.lit(True)
    x_in_pool = when(mask).then(x).otherwise(None)

    # 使用 qcut 处理池内数据
    # 注意：Polars 的 qcut 在截面中使用时，需确保处理好边界
    rank_pct = x_in_pool.rank(method="average") / (mask.cast(pl.UInt32).sum() + 1e-6)

    # 映射到分箱编号 (0 to n_bins-1)
    res = (rank_pct * n_bins).floor().clip(0, n_bins - 1)

    # 池外通常填充中位分箱，如 10 分箱填 5.0（或 4.5）
    return when(mask).then(res).otherwise(fill_null)
