# from typing import Union, Literal
#
# import numpy as np
# import polars as pl
#
#
# def single_calc_quantile_metrics(
#         df: Union[pl.DataFrame, pl.LazyFrame],  # 修改支持 LazyFrame
#         factor_col: str,
#         ret_col: str,
#         date_col: str = "DATE",
#         asset_col: str = "ASSET",
#         pool_mask_col: str = 'POOL',
#         n_bins: int = 5,
#         mode: Literal['long_only', 'long_short', 'active'] = 'active',
#         period: int = 1,
#         cost: float = 0.0,
#         est_turnover: float = 0.2,
#         annual_days: int = 251
# ) -> dict:
#     # --- 0. 统一转为 LazyFrame 以便利用下压优化 ---
#     lf = df.lazy() if isinstance(df, pl.DataFrame) else df
#
#     # --- 1. 模拟调仓周期逻辑 (这里必须 collect，因为 Python 需要日期列表来做循环) ---
#     all_dates = (
#         lf.select(date_col)
#         .unique()
#         .sort(date_col)
#         .collect()
#         .get_column(date_col)
#         .to_list()
#     )
#     rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), period)]
#
#     # --- 2. 动态股票池过滤与分层 ---
#     working_lf = lf.filter(pl.col(pool_mask_col)) if pool_mask_col else lf
#
#     # 分组分位数计算
#     df_with_q = (
#         working_lf.with_columns(
#             pl.when(pl.col(date_col).is_in(rebalance_dates))
#             .then(
#                 pl.col(factor_col)
#                 .rank(method="random")
#                 .over(date_col)
#                 .qcut(n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])
#             )
#             .otherwise(None).alias("quantile")
#         )
#         .sort([asset_col, date_col])
#         .with_columns(pl.col("quantile").forward_fill().over(asset_col))
#         .filter(pl.col("quantile").is_not_null())
#     )
#
#     # --- 3. 聚合收益 ---
#     q_rets_lf = (
#         df_with_q.group_by([date_col, "quantile"])
#         .agg([
#             pl.col(ret_col).mean().alias("ret"),
#             pl.len().alias("count")  # 增加 count 用于后续统计
#         ])
#     )
#
#     # Pivot 操作在 Polars Lazy 中是阻塞的，会自动触发部分 collect
#     res_series = q_rets_lf.collect().pivot(
#         index=date_col, on="quantile", values="ret"
#     ).sort(date_col)
#
#     # --- 4. 扣除成本 ---
#     reb_cost = est_turnover * period * cost
#     all_q_cols = [f"Q{i + 1}" for i in range(n_bins)]
#
#     if mode == "long_only":
#         res_series = res_series.with_columns(pl.col("Q1").alias("raw_ret"))
#     elif mode == "long_short":
#         res_series = res_series.with_columns((pl.col("Q1") - pl.col(f"Q{n_bins}")).alias("raw_ret"))
#         reb_cost = reb_cost * 2
#     elif mode == "active":
#         res_series = res_series.with_columns((pl.col("Q1") - pl.mean_horizontal(all_q_cols)).alias("raw_ret"))
#
#     res_series = res_series.with_columns(
#         pl.when(pl.col(date_col).is_in(rebalance_dates))
#         .then(pl.col("raw_ret") - reb_cost)
#         .otherwise(pl.col("raw_ret"))
#         .alias("target_ret")
#     ).with_columns(
#         (pl.col("target_ret").fill_null(0) + 1).cum_prod().alias("nav")
#     )
#
#     # --- 5. 计算评价指标 ---
#     total_days = len(all_dates)
#     if total_days <= 1:
#         return {"error": "Insufficient data"}
#
#     # 使用 get_column 替代 [col]
#     nav_arr = res_series.get_column("nav").to_numpy()
#     target_ret_arr = res_series.get_column("target_ret")
#
#     total_ret = nav_arr[-1] - 1 if len(nav_arr) > 0 else 0.0
#     annual_ret = (1 + total_ret) ** (annual_days / total_days) - 1
#     annual_vol = target_ret_arr.std() * (annual_days ** 0.5)
#     sharpe_ratio = annual_ret / (annual_vol + 1e-9)
#
#     # 最大回撤
#     running_max = np.maximum.accumulate(nav_arr)
#     max_drawdown = np.min((nav_arr - running_max) / (running_max + 1e-9))
#
#     # 稳定性分析
#     # 注意：q_rets 此时需要 collect
#     q_rets_df = q_rets_lf.collect()
#     smoothness = _check_factor_smoothness(q_rets_df, n_bins)
#
#     return {
#         "quantile_daily_ret": q_rets_df,
#         "series": res_series,
#         "mode": mode,
#         "metrics": {
#             "total_return": total_ret,
#             "annual_return": annual_ret,
#             "annual_volatility": annual_vol,
#             "sharpe_ratio": sharpe_ratio,
#             "max_drawdown": max_drawdown,
#             "win_rate": res_series.filter(pl.col("target_ret") > 0).height / total_days,
#             "monotonicity": smoothness["monotonicity_score"],
#             "smoothness_index": smoothness["gap_stability"],
#             "avg_count_per_bin": q_rets_df.get_column("count").mean(),
#             "total_obs": q_rets_df.get_column("count").sum(),
#             "rebalance_period": period,
#             "avg_daily_turnover": est_turnover
#         }
#     }
