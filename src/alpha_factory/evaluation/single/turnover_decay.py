from typing import Union

import numpy as np
import polars as pl

from alpha_factory.utils.schema import F


def single_calc_decay_turnover(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factor_col: str,
        ret_col: str,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        max_lag: int = 10
) -> dict:
    """
    计算单因子的 IC 衰减与换手率估计
    :param df:
    :param factor_col:
    :param ret_col:
    :param date_col:
    :param asset_col:
    :param pool_mask_col:
    :param max_lag:
    :return:
    {
        "ic_lags": List[float],  # 各滞后期的 IC 均值
        "autocorr": float,      # 因子自身的滞后相关系数
        "est_daily_turnover": float  # 估计的日换手率
    }
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # 1. 在“完整时序”上计算位移列（不要先 filter！）
    # 这样 shift(1).over(asset) 才能找到物理上的前一个交易日
    shift_exprs = [
        pl.col(ret_col).shift(-i).over(asset_col).alias(f"_ret_lag_{i}")
        for i in range(max_lag)
    ]
    shift_exprs.append(pl.col(factor_col).shift(1).over(asset_col).alias("_factor_pre"))

    # 2. 预计算位移并应用过滤
    # 在这里 filter，保证 corr 计算时只使用 POOL_MASK=True 且位移成功的行
    filtered_lf = (
        lf.with_columns(shift_exprs)
        .filter(pl.col(pool_mask_col)) # 计算完位移再过滤
        .select([date_col, factor_col, "_factor_pre"] + [f"_ret_lag_{i}" for i in range(max_lag)])
    )

    # 3. 计算聚合指标
    daily_res = (
        filtered_lf.group_by(date_col)
        .agg([
            pl.corr(factor_col, f"_ret_lag_{i}", method="spearman").alias(f"ic_{i}")
            for i in range(max_lag)
        ] + [
            pl.corr(factor_col, "_factor_pre", method="spearman").alias("ac")
        ])
        .collect()
    )

    # 4. 提取均值并处理空
    # 使用 drop_nans().mean() 保证稳健性
    lags = [daily_res.get_column(f"ic_{i}").drop_nans().mean() or 0.0 for i in range(max_lag)]
    autocorr_val = daily_res.get_column("ac").drop_nans().mean() or 0.0

    # 5. 换手率计算逻辑保护
    # 如果 autocorr 还是 nan，给定一个保守的极低值 0.0 (代表 100% 换手)
    safe_ac = autocorr_val if not np.isnan(autocorr_val) else 0.0
    est_daily_turnover = (1 - max(0, safe_ac)) * 0.85

    return {
        "ic_lags": lags,
        "autocorr": autocorr_val,
        "est_daily_turnover": est_daily_turnover
    }
