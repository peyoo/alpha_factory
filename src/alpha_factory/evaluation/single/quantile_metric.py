from typing import Union, Literal

import numpy as np
import polars as pl

from alpha_factory.utils.schema import F


def single_calc_quantile_metrics(
    df: Union[pl.DataFrame, pl.LazyFrame],  # 修改支持 LazyFrame
    factor_col: str,
    ret_col: str,
    date_col: str = F.DATE,
    asset_col: str = F.ASSET,
    pool_mask_col: str = F.POOL_MASK,
    n_bins: int = 10,
    mode: Literal["long_only", "long_short", "active"] = "active",
    period: int = 1,
    cost: float = 0.0025,
    est_turnover: float = 0.2,
    annual_days: int = 251,
    direction: Literal[1, -1] = 1,  # 🆕 新增方向参数
) -> dict:
    """
    计算单因子的分层收益表现及综合评价指标
    1. 根据因子值将股票分为 n_bins 个分层（分位数）
    2. 模拟按指定周期调仓，计算各分层的平均收益率
    3. 根据评估模式计算组合收益（多头、多空、相对收益）
    4. 扣除交易成本，计算净收益曲线
    5. 计算总收益、年化收益、夏普比率、最大回撤等指标

    Args:
        df: 输入数据框，包含因子列和收益率列
        factor_col: 因子列名称
        ret_col: 收益率列名称
        date_col: 日期列名称
        asset_col: 资产列名称
        pool_mask_col: 股票池掩码列名称（True = 在池内）
        n_bins: 分层数量（分位数）
        mode: 评估模式
            - 'long_only': 仅多头头寸
            - 'long_short': 多空头寸
            - 'active': 相对于市场平均收益
        period: 调仓周期（单位：交易日）
        cost: 单边交易费用率（用于扣除成本）
        est_turnover: 估计的日均换手率（用于计算调仓成本）
        annual_days: 年化交易日天数（用于年化收益计算）
        direction: 因子方向 (1=正向, -1=反向)，决定多头和空头的分层选择
    :returns
        dict: 包含分层收益数据和综合评价指标的字典
    结构如下：
        {
            "quantile_daily_ret": pl.DataFrame,  # 各分层每日收益
            "series": pl.DataFrame,               # 净收益曲线数据
            "mode": str,                          # 评估模式
            "metrics": {                         # 综合评价指标
                "total_return": float,
                "annual_return": float,
                "annual_volatility": float,
                "sharpe_ratio": float,
                "max_drawdown": float,
                "win_rate": float,
                "monotonicity": float,
                "smoothness_index": float,
                "avg_count_per_bin": float,
                "total_obs": int,
                "rebalance_period": int,
                "avg_daily_turnover": float
            }
        }
    说明：
    该函数实现了基于因子分层的收益评估，考虑了动态股票池和交易成本，能够全面反映因子的实际投资价值。
    """
    # --- 0. 统一转为 LazyFrame 以便利用下压优化 ---
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # --- 1. 模拟调仓周期逻辑 (这里必须 collect，因为 Python 需要日期列表来做循环) ---
    all_dates = (
        lf.select(date_col)
        .unique()
        .sort(date_col)
        .collect()
        .get_column(date_col)
        .to_list()
    )
    rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), period)]

    # --- 2. 动态股票池过滤与分层 ---
    working_lf = lf.filter(pl.col(pool_mask_col)) if pool_mask_col else lf

    # 分组分位数计算
    df_with_q = (
        working_lf.with_columns(
            pl.when(pl.col(date_col).is_in(rebalance_dates))
            .then(
                pl.col(factor_col)
                .rank(method="random")
                .over(date_col)
                .qcut(n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])
            )
            .otherwise(None)
            .alias("quantile")
        )
        .sort([asset_col, date_col])
        .with_columns(pl.col("quantile").forward_fill().over(asset_col))
        .filter(pl.col("quantile").is_not_null())
    )

    # --- 3. 聚合收益 ---
    q_rets_lf = df_with_q.group_by([date_col, "quantile"]).agg(
        [
            pl.col(ret_col).mean().alias("ret"),
            pl.len().alias("count"),  # 增加 count 用于后续统计
        ]
    )

    # Pivot 操作在 Polars Lazy 中是阻塞的，会自动触发部分 collect
    res_series = (
        q_rets_lf.collect()
        .pivot(index=date_col, on="quantile", values="ret")
        .sort(date_col)
    )

    # --- 4. 扣除成本 ---
    reb_cost = est_turnover * period * cost
    # --- 根据方向确定多头和空头桶 ---
    if direction == 1:
        long_col = f"Q{n_bins}"  # 因子值最大为多头
        short_col = "Q1"
    else:
        long_col = "Q1"  # 因子值最小为多头
        short_col = f"Q{n_bins}"
    all_q_cols = [f"Q{i + 1}" for i in range(n_bins)]

    if mode == "long_only":
        res_series = res_series.with_columns(pl.col(long_col).alias("raw_ret"))
    elif mode == "long_short":
        # 此时如果是 direction=-1，会自动变成 Q1 - Q10
        res_series = res_series.with_columns(
            (pl.col(long_col) - pl.col(short_col)).alias("raw_ret")
        )
        reb_cost = reb_cost * 2
    elif mode == "active":
        # 使用 long_col 减去截面平均
        res_series = res_series.with_columns(
            (pl.col(long_col) - pl.mean_horizontal(all_q_cols)).alias("raw_ret")
        )

    res_series = res_series.with_columns(
        pl.when(pl.col(date_col).is_in(rebalance_dates))
        .then(pl.col("raw_ret") - reb_cost)
        .otherwise(pl.col("raw_ret"))
        .alias("target_ret")
    ).with_columns((pl.col("target_ret").fill_null(0) + 1).cum_prod().alias("nav"))

    # --- 5. 计算评价指标 ---
    total_days = len(all_dates)
    if total_days <= 1:
        return {"error": "Insufficient data"}

    # 使用 get_column 替代 [col]
    nav_arr = res_series.get_column("nav").to_numpy()
    target_ret_arr = res_series.get_column("target_ret")

    total_ret = nav_arr[-1] - 1 if len(nav_arr) > 0 else 0.0
    annual_ret = (1 + total_ret) ** (annual_days / total_days) - 1
    annual_vol = target_ret_arr.std() * (annual_days**0.5)
    sharpe_ratio = annual_ret / (annual_vol + 1e-9)

    # 最大回撤
    running_max = np.maximum.accumulate(nav_arr)
    max_drawdown = np.min((nav_arr - running_max) / (running_max + 1e-9))

    # 稳定性分析
    # 注意：q_rets 此时需要 collect
    q_rets_df = q_rets_lf.collect()
    smoothness = _check_factor_smoothness(q_rets_df, n_bins)

    return {
        "quantile_daily_ret": q_rets_df,
        "series": res_series,
        "mode": mode,
        "metrics": {
            "total_return": total_ret,
            "annual_return": annual_ret,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": res_series.filter(pl.col("target_ret") > 0).height / total_days,
            "monotonicity": smoothness["monotonicity_score"],
            "smoothness_index": smoothness["gap_stability"],
            "avg_count_per_bin": q_rets_df.get_column("count").mean(),
            "total_obs": q_rets_df.get_column("count").sum(),
            "rebalance_period": period,
            "avg_daily_turnover": est_turnover,
        },
    }


def _check_factor_smoothness(q_rets: pl.DataFrame, n_bins: int) -> dict:
    """
    判断分层收益的平滑度
    """
    # 1. 计算各分层的全周期平均收益
    mean_rets = q_rets.group_by("quantile").agg(pl.col("ret").mean()).sort("quantile")

    # 2. 计算单调性得分 (Spearman Rank Correlation)
    # 理想值是 1 (严格单调递增) 或 -1 (严格单调递减)
    quantile_idx = np.arange(1, n_bins + 1)
    return_values = mean_rets["ret"].to_numpy()

    # 使用简单相关系数衡量单调性
    monotonicity = np.corrcoef(quantile_idx, return_values)[0, 1]

    # 3. 计算收益间距的稳定性 (Gap Deviation)
    # 如果 Q1-Q2, Q2-Q3... 的间距均匀，说明因子对各分段的区分度都很平滑
    gaps = np.diff(return_values)
    gap_cv = np.std(gaps) / (np.abs(np.mean(gaps)) + 1e-9)  # 间距变异系数，越小越平滑

    return {
        "monotonicity_score": monotonicity,
        "gap_stability": 1 / (1 + gap_cv),  # 归一化，越接近 1 越平滑
    }
