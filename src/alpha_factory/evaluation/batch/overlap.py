"""evaluation/batch/overlap.py — 因子 Top-N 持仓重合度（Hit Rate）计算

对任意两个因子 (A, B) 计算其每日截面内排名最高的 Top-N 只持仓资产的重合率：

    Hit Rate(A, B) = (1/T) Σ_t  |TopN_A(t) ∩ TopN_B(t)| / N

其中 TopN_A(t) 为第 t 天因子 A 值排名最高的 N 只资产集合。
"""

from __future__ import annotations

from typing import List, Union

import polars as pl
from loguru import logger

from alpha_factory.utils.schema import F


def batch_topn_overlap(
    df: Union[pl.DataFrame, pl.LazyFrame],
    factor_names: List[str],
    topn: int = 50,
    date_col: str = F.DATE,
    asset_col: str = F.ASSET,
    pool_mask_col: str = F.POOL_MASK,
) -> pl.DataFrame:
    """计算任意两因子 Top-N 持仓资产的日均重合率（Hit Rate）。

    Args:
        df: 包含因子列、日期列、资产列及股票池掩码列的数据。
        factor_names: 因子列名列表（列须已存在于 df 中）。
        topn: 每日每因子取排名最高的 N 只资产作为持仓。
        date_col: 日期列名。
        asset_col: 资产列名。
        pool_mask_col: 股票池掩码列名（True = 在池内）。

    Returns:
        pl.DataFrame，列：factor_a, factor_b, hit_rate（已按 hit_rate 降序排列）。
        hit_rate = (1/T) Σ_t |TopN_A(t) ∩ TopN_B(t)| / N。
        当因子数 < 2 时返回空 DataFrame。
    """
    if len(factor_names) < 2:
        logger.warning("batch_topn_overlap: 至少需要 2 个因子，返回空 DataFrame")
        return pl.DataFrame({"factor_a": [], "factor_b": [], "hit_rate": []})

    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # ── Step 1: 应用股票池掩码过滤（与 full_metrics Step 2 一致）
    schema_names = lf.collect_schema().names()
    if pool_mask_col in schema_names:
        lf = lf.filter(pl.col(pool_mask_col))
        logger.debug(f"  ✓ 已应用股票池掩码: '{pool_mask_col}'")
    else:
        logger.warning(
            f"  ⚠️ 未找到池掩码列 '{pool_mask_col}'，将使用全量数据计算重合度"
        )

    # ── Step 2: 仅保留计算所需列，过滤不存在的因子列
    available = [f for f in factor_names if f in schema_names]
    if len(available) < 2:
        logger.warning(
            f"  ✗ 有效因子列不足 2 个（请求: {factor_names}，可用: {available}）"
        )
        return pl.DataFrame({"factor_a": [], "factor_b": [], "hit_rate": []})
    if len(available) < len(factor_names):
        missing = set(factor_names) - set(available)
        logger.warning(f"  ⚠️ 以下因子列不存在，将跳过: {missing}")

    lf = lf.select([date_col, asset_col] + available)

    # ── Step 3: 每日截面内对每个因子升序排名（rank 最大 = 因子值最大）
    lf = lf.with_columns(
        [pl.col(f).rank().over(date_col).alias(f"__rank_{f}") for f in available]
    )

    # ── Step 4: 生成 Top-N 布尔标志
    #   rank > (pool_size - topn).clip(0) 表示该资产在当日 Top-N 内
    #   pool_size = 每日当前截面的股票数量
    daily_pool_size = pl.len().over(date_col)
    threshold = (daily_pool_size - topn).clip(lower_bound=0)
    lf = lf.with_columns(
        [(pl.col(f"__rank_{f}") > threshold).alias(f"__topN_{f}") for f in available]
    )

    # ── Step 5: 单次 group_by 计算所有因子对的日度交集数量
    n = len(available)
    pairs = [
        (i, j, available[i], available[j]) for i in range(n) for j in range(i + 1, n)
    ]

    daily_intersection = (
        lf.group_by(date_col)
        .agg(
            [
                (pl.col(f"__topN_{fi}") & pl.col(f"__topN_{fj}"))
                .sum()
                .cast(pl.Float64)
                .alias(f"__pair_{i}_{j}")
                for i, j, fi, fj in pairs
            ]
        )
        .collect()
    )

    total_days = daily_intersection.height
    logger.info(
        f"  ✓ 完成日度交集聚合 | {len(pairs)} 对因子 | {total_days} 个交易日 | topn={topn}"
    )

    # ── Step 6: 逐对计算 Hit Rate = mean_over_days(count(t) / N)
    rows = []
    for i, j, fi, fj in pairs:
        col = f"__pair_{i}_{j}"
        # 先除以 topn 得到日度 hit_rate，再取算术平均
        daily_hit = daily_intersection[col] / topn
        hit_rate = float(daily_hit.mean()) if total_days > 0 else 0.0
        rows.append({"factor_a": fi, "factor_b": fj, "hit_rate": hit_rate})

    result = pl.DataFrame(rows).sort("hit_rate", descending=True)
    logger.info(
        f"  ✅ batch_topn_overlap 完成 | 最高重合对: "
        f"{result['factor_a'][0]} vs {result['factor_b'][0]} "
        f"(hit_rate={result['hit_rate'][0]:.4f})"
    )
    return result


__all__ = ["batch_topn_overlap"]
