"""
单因子深度分析工具集
包含 IC 计算、分层收益分析、衰减与换手率等功能

"""
from typing import Literal, Union

import numpy as np
import polars as pl
from loguru import logger

from alpha.evaluation.batch import batch_get_ic_summary
from alpha.utils.schema import F


def single_calc_ic_analysis(
        df: pl.DataFrame,
        factor_col: str,
        ret_col: str,
        date_col: str = F.DATE,
        rolling_window: int = 20
) -> pl.DataFrame:
    """
    计算单因子的每日 IC 序列及滚动 ICIR
    """
    # 1. 计算每日 Rank IC
    ic_series = (
        df.group_by(date_col)
        .agg(pl.corr(pl.col(factor_col).rank(), pl.col(ret_col).rank(), method="pearson").alias("ic"))
        .sort(date_col)
    )

    # 2. 计算滚动指标 (识别因子最近是否失效)
    ic_analysis = ic_series.with_columns([
        pl.col("ic").rolling_mean(rolling_window).alias("rolling_ic_mean"),
        (pl.col("ic").rolling_mean(rolling_window) / pl.col("ic").rolling_std(rolling_window)).alias("rolling_ir"),
        pl.col("ic").cum_sum().alias("cum_ic")
    ])

    return ic_analysis


def _check_factor_smoothness(q_rets: pl.DataFrame, n_bins: int) -> dict:
    """
    判断分层收益的平滑度
    """
    # 1. 计算各分层的全周期平均收益
    mean_rets = (
        q_rets.group_by("quantile")
        .agg(pl.col("ret").mean())
        .sort("quantile")
    )

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
        "gap_stability": 1 / (1 + gap_cv)  # 归一化，越接近 1 越平滑
    }


def single_calc_quantile_metrics(
        df: Union[pl.DataFrame, pl.LazyFrame],  # 修改支持 LazyFrame
        factor_col: str,
        ret_col: str,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'active',
        period: int = 1,
        cost: float = 0.0,
        est_turnover: float = 0.2,
        annual_days: int = 251,
        direction: Literal[1, -1] = 1,  # 🆕 新增方向参数
) -> dict:
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
            .otherwise(None).alias("quantile")
        )
        .sort([asset_col, date_col])
        .with_columns(pl.col("quantile").forward_fill().over(asset_col))
        .filter(pl.col("quantile").is_not_null())
    )

    # --- 3. 聚合收益 ---
    q_rets_lf = (
        df_with_q.group_by([date_col, "quantile"])
        .agg([
            pl.col(ret_col).mean().alias("ret"),
            pl.len().alias("count")  # 增加 count 用于后续统计
        ])
    )

    # Pivot 操作在 Polars Lazy 中是阻塞的，会自动触发部分 collect
    res_series = q_rets_lf.collect().pivot(
        index=date_col, on="quantile", values="ret"
    ).sort(date_col)

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
        res_series = res_series.with_columns((pl.col(long_col) - pl.col(short_col)).alias("raw_ret"))
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
    ).with_columns(
        (pl.col("target_ret").fill_null(0) + 1).cum_prod().alias("nav")
    )

    # --- 5. 计算评价指标 ---
    total_days = len(all_dates)
    if total_days <= 1:
        return {"error": "Insufficient data"}

    # 使用 get_column 替代 [col]
    nav_arr = res_series.get_column("nav").to_numpy()
    target_ret_arr = res_series.get_column("target_ret")

    total_ret = nav_arr[-1] - 1 if len(nav_arr) > 0 else 0.0
    annual_ret = (1 + total_ret) ** (annual_days / total_days) - 1
    annual_vol = target_ret_arr.std() * (annual_days ** 0.5)
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
            "avg_daily_turnover": est_turnover
        }
    }

def single_calc_decay_turnover(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factor_col: str,
        ret_col: str,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        max_lag: int = 10
) -> dict:
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

def single_factor_alpha_analysis(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factor_col: str,
        ret_col: str,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        mode: Literal['long_only', 'long_short', 'active'] = 'active',
        n_bins: int = 5,
        period: int = 1,
        cost: float = 0.0015  # 默认单边费率（如印花税+佣金）
) -> dict:
    """
    【工业级】单因子全能体检报告：
    集成信号衰减、自相关性换手估算、扣费分层回测。
    """

    # 1. 信号衰减与换手率估算 (核心：先算稳定性)
    # 返回包含 ic_lags, autocorr, est_daily_turnover 的字典
    logger.info("🔍 正在计算因子信号衰减与换手率估算...")
    decay_stats = single_calc_decay_turnover(
        df, factor_col, ret_col, date_col, asset_col
    )
    logger.info(f"    > 估算日均换手率: {decay_stats['est_daily_turnover']:.2%} (自相关: {decay_stats['autocorr']:.3f})")
    est_turnover = decay_stats['est_daily_turnover']


    logger.info("🔍 正在计算因子预测效力指标 (IC Summary)...")
    # 2. 基础 IC 统计 (预测效力)
    ic_summary = batch_get_ic_summary(
        df,
        factor_pattern=f"^{factor_col}$",
        ret_col=ret_col,
        date_col=date_col
    )
    ic_mean = ic_summary['ic_mean'][0]
    logger.info(f"    > IC 均值: {ic_mean:.4f}, ICIR: {ic_summary['ic_ir'][0]:.4f}")

    # 3. 分层收益与实盘风险指标 (传入估算的 est_turnover 进行扣费)
    quantile_res = single_calc_quantile_metrics(
        df, factor_col, ret_col,
        date_col=date_col,
        asset_col=asset_col,
        pool_mask_col=pool_mask_col,
        mode=mode,
        n_bins=n_bins,
        period=period,
        cost=cost,
        est_turnover=est_turnover,  # 自动关联换手
        direction= 1 if ic_mean > 0 else -1  # 根据信号方向调整多空逻辑
    )

    m = quantile_res['metrics']
    nav_series = quantile_res['series']

    # --- 开始打印全量解释报告 ---
    print(f"\n{'#' * 30} 因子体检报告: {factor_col} {'#' * 30}")

    # --- 第一部分：预测效力 ---
    print("\n【1. 预测效力 - 衡量因子捕捉收益的相关性】")
    ic_val = ic_summary['ic_mean'][0]
    icir_val = ic_summary['ic_ir'][0]
    print(f"  > IC 均值: {ic_val:.4f}")
    print("    [解释]: 因子值与下期收益的相关系数。>0.02代表有预测力，值越大方向越准。")
    print(f"  > ICIR: {icir_val:.4f}")
    print("    [解释]: IC均值/IC标准差。衡量稳定性，>0.5代表信号稳健。")

    # --- 第二部分：实盘表现 ---
    print("\n【2. 实盘表现 - 模拟真实交易扣费后的收益】")
    print(f"  > 净年化收益: {m['annual_return']:.2%}")
    print("    [解释]: 考虑调仓周期和基于自相关性估算的换手扣费后的年化。")
    print(f"  > 净夏普比率: {m['sharpe_ratio']:.2f}")
    print(f"  > 最大回撤: {m['max_drawdown']:.2%}")

    # --- 第三部分：执行成本 ---
    print("\n【3. 执行成本 - 衡量因子在实盘中落地的难易度】")
    print(f"  > 估算日均换手率: {est_turnover:.2%}")
    print(f"    [解释]: 基于因子秩自相关性(AC={decay_stats['autocorr']:.3f})推导出的每日头寸变动。")
    print(f"  > 调仓周期: {period} 交易日")
    print(f"  > 摩擦成本系数: {cost * 10000:.1f} bps (基点)")

    # --- 第四部分：逻辑健壮性 ---
    print("\n【4. 逻辑健壮性 - 检验因子赚钱的底层逻辑】")
    print(f"  > 收益单调性: {m['monotonicity']:.2f}")
    print(f"  > 分层平滑度: {m['smoothness_index']:.2f}")

    # --- 第五部分：信号衰减 ---
    print("\n【5. 信号衰减 - 衡量因子的“保鲜期”】")
    lags = decay_stats['ic_lags']
    # 避免除以 0，且 lag0 通常是当期 IC
    lag1_val = lags[1] if len(lags) > 1 else 1e-9
    lag5_val = lags[5] if len(lags) > 5 else 0.0
    retention = (lag5_val / lag1_val) if lag1_val != 0 else 0.0
    print(f"  > 信号留存率 (Lag5/Lag1): {retention:.1%}")
    print("    [解释]: 5天后预测能力剩下的比例。若<20%，说明该因子必须高频调仓。")

    # --- 样本统计 ---
    print("\n【6. 样本统计】")
    print(f"  > 每层平均样本数: {m['avg_count_per_bin']:.1f}")

    print(f"\n{'#' * 78}\n")
    logger.info("✅ 因子体检报告生成完毕。")

    return {
        "summary": ic_summary,
        "metrics": m,
        "decay": decay_stats,
        "nav": nav_series
    }
