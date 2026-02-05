"""
大批量因子评估相关计算函数集

"""

from typing import List

import polars as pl


def batch_calc_factor_full_metrics(
        df: pl.DataFrame,
        factor_pattern: List[str],
        ret_col: str,
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        max_lag: int = 10
) -> pl.DataFrame:
    """
    【合并版】一次性计算基础 IC 指标 + IC/IR 衰减图谱
    大批量计算因子 IC 衰减图谱及基础统计指标
    结果返回每个因子在不同滞后期的 IC Mean、IR 以及基础指标（t-stat、win rate）
    计算逻辑：
    1. 构造滞后收益率列并 Rank
    2. 长表化处理所有因子，方便统一计算
    3. 按日期和因子分组，计算各滞后期的 IC 序列
    4. 最终统计各因子在不同滞后期的 IC Mean、IR 以及基础指标
    说明：
    这种方法避免了多次循环计算，极大提升了效率。
    适用于大规模因子评估场景。
    """

    # 1. 统一选取因子列名
    # 如果传入的是字符串，则视为正则匹配；如果已经是 List，则直接使用
    if isinstance(factor_pattern, str):
        factor_cols = df.select(pl.col(factor_pattern)).columns
    else:
        factor_cols = factor_pattern

    # 2. 构造滞后收益率并 Rank
    target_lags = [f"target_lag_{i}" for i in range(max_lag)]
    q = df.lazy().with_columns([
        pl.col(ret_col).shift(-i).over(asset_col).rank().over(date_col).alias(f"target_lag_{i}")
        for i in range(max_lag)
    ])

    # 2. 长表化处理
    q_long = q.unpivot(
        index=[date_col, asset_col] + target_lags,
        on=factor_cols,
        variable_name="factor",
        value_name="factor_value"
    ).with_columns(
        pl.col("factor_value").rank().over([date_col, "factor"])
    )

    # 3. 核心聚合：计算各 Lag 的每日 IC 序列
    ic_series = q_long.group_by([date_col, "factor"]).agg([
        pl.corr("factor_value", pl.col(f"target_lag_{i}"), method="pearson").alias(f"lag_{i}")
        for i in range(max_lag)
    ])

    # 4. 终极聚合：合并衰减与基础指标
    # 我们以 lag_0 作为基础 IC (即传统的 IC 统计)
    full_stats = ic_series.group_by("factor").agg([
        # --- 衰减部分 ---
        *[pl.col(f"lag_{i}").mean().alias(f"IC_Mean_Lag_{i}") for i in range(max_lag)],
        *[(pl.col(f"lag_{i}").mean() / pl.col(f"lag_{i}").std()).alias(f"IR_Lag_{i}") for i in range(max_lag)],

        # --- 基础指标补全 (基于 lag_0) ---
        (pl.col("lag_0").mean() / pl.col("lag_0").std() * pl.count().sqrt()).alias("t_stat"),
        (pl.col("lag_0").filter(pl.col("lag_0") > 0).count() / pl.count()).alias("win_rate")
    ]).collect()

    return full_stats


#
# def batch_factor_alpha_lens(
#         df: pl.DataFrame,
#         factors: str = r"^factor_.*",
#         label_for_ret: str = "target_ret",
#         date_col: str = "DATE",
#         asset_col: str = "ASSET",
#         n_bins: int = 5,
#         max_lag: int = 5
# ) -> pl.DataFrame:
#     """
#     【终极全能版】大批量因子体检引擎：IC/IR + 衰减 + 换手 + 分层收益
#     """
#     # lf = df.lazy() if isinstance(df, pl.DataFrame) else df
#
#     factor_cols = df.select(pl.col(factors)).columns
#
#     # --- 第一部分：基础指标与衰减 (IC/IR/t-stat/WinRate) ---
#     full_metrics = batch_calc_factor_full_metrics(
#         df, factor_cols, label_for_ret, date_col, asset_col, max_lag
#     )
#
#     # --- 第二部分：稳定性指标 (Turnover/Autocorr) ---
#     turnover_stats = batch_calc_factor_turnover(
#         df, factors, date_col, asset_col
#     )
#
#     # --- 第三部分：实战收益指标 (Quantile Returns) ---
#     # 我们从中提取多空年化收益和多空夏普
#     q_rets = batch_calc_quantile_returns(
#         df, factors, label_for_ret, date_col, n_bins
#     )
#
#     ls_metrics = (
#         q_rets.pivot(index=[date_col, "factor"], on="quantile", values="daily_ret")
#         .with_columns((pl.col("Q1") - pl.col(f"Q{n_bins}")).alias("ls_ret"))
#         .group_by("factor")
#         .agg([
#             (pl.col("ls_ret").mean() * 242).alias("annual_ls_ret"),
#             (pl.col("ls_ret").mean() / pl.col("ls_ret").std() * (242 ** 0.5)).alias("ls_sharpe")
#         ])
#     )
#
#     # --- 最终合并所有维度 ---
#     master_table = (
#         full_metrics
#         .join(turnover_stats, on="factor")
#         .join(ls_metrics, on="factor")
#     )
#
#     return master_table
