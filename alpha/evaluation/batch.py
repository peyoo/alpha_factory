"""
大批量因子评估相关计算函数集

"""


from typing import List

import polars as pl


def batch_calc_rank_ic(df: pl.DataFrame,
                       factor_pattern: str = r"^factor_.*",
                       ret_col: str = "target_ret",
                       date_col: str = "DATE") -> pl.DataFrame:
    """
    大批量并行计算 Rank IC
    factor_pattern: 匹配所有因子的正则表达式
    ret_col: 目标收益列名
    date_col: 日期列名
    结果返回每日各因子的 IC 宽表

    """
    # 1. 获取所有因子列名
    factor_cols = df.select(pl.col(factor_pattern)).columns

    # 2. 核心计算：一次 group_by 完成所有因子的 Spearman 相关性计算
    # 这里直接对因子和收益率进行 rank 预处理，然后算 pearson，在数学上等价于 spearman 且更快
    ic_series = (
        df.group_by(date_col)
        .agg([
            pl.corr(
                pl.col(f).rank(),
                pl.col(ret_col).rank(),
                method="pearson"
            ).alias(f)
            for f in factor_cols
        ])
        .sort(date_col)
    )
    return ic_series


def batch_calc_ic_metrics(ic_series: pl.DataFrame, date_col: str = "DATE") -> pl.DataFrame:
    """
    输入是每日各因子的 IC 宽表
    输出是每个因子的 IC Mean, ICIR, t-stat 等
    结果返回因子 IC 汇总统计表

    """
    # 剔除日期列，剩下全是因子列
    factor_cols = [c for c in ic_series.columns if c != date_col]

    # 使用 unpivot (melt) 将宽表转回长表，方便统一聚合统计
    metrics = (
        ic_series.unpivot(
            index=date_col,
            on=factor_cols,
            variable_name="factor",
            value_name="ic"
        )
        .group_by("factor")
        .agg([
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            (pl.col("ic").mean() / pl.col("ic").std()).alias("ic_ir"),
            (pl.col("ic").mean() / pl.col("ic").std() * pl.count().sqrt()).alias("t_stat"),
            # 胜率：IC > 0 的比例
            (pl.col("ic").filter(pl.col("ic") > 0).count() / pl.count()).alias("win_rate")
        ])
    )
    return metrics


def batch_get_ic_summary(df: pl.DataFrame,
                         factor_pattern: str = r"^factor_.*",
                         ret_col: str = "ret_real_trade", # 建议这里默认值与你常用的保持一致
                         date_col: str = "DATE") -> pl.DataFrame:
    # 1. 自动获取因子列
    factor_cols = df.select(pl.col(factor_pattern)).collect_schema().names()
    if not factor_cols:
        raise ValueError(f"未找到匹配 {factor_pattern} 的因子列")

    # 2. 构造计算链路
    ic_summary = (
        df.lazy()
        .select([date_col, ret_col] + factor_cols)
        .drop_nulls()  # 【关键修复】确保参与计算的行没有空值
        .group_by(date_col)
        .agg([
            # 使用 Spearman (Rank) 相关性通常比 Pearson 更稳健
            pl.corr(pl.col(f), pl.col(ret_col), method="spearman").alias(f) for f in factor_cols
        ])
        .unpivot(index=date_col, on=factor_cols, variable_name="factor", value_name="ic")
        .filter(pl.col("ic").is_not_nan() & pl.col("ic").is_not_null())
        .group_by("factor")
        .agg([
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            # 增加 fill_nan(0) 防止除以 0 的情况
            (pl.col("ic").mean() / pl.col("ic").std().fill_nan(1e-9)).alias("ic_ir"),
            (pl.col("ic").mean() / pl.col("ic").std().fill_nan(1e-9) * pl.count().sqrt()).alias("t_stat"),
            (pl.col("ic").filter(pl.col("ic") > 0).count() / pl.count()).alias("win_rate")
        ])
        .collect()
    )
    return ic_summary

def batch_calc_factor_decay_stats(
        df: pl.DataFrame,
        factor_pattern: List[str],
        ret_col: str,
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        max_lag: int = 10
) -> pl.DataFrame:
    """
    大批量计算因子 IC 衰减图谱
    结果返回每个因子在不同滞后期的 IC Mean 和 IR
    计算逻辑：
    1. 构造滞后收益率列并 Rank
    2. 长表化处理所有因子，方便统一计算
    3. 按日期和因子分组，计算各滞后期的 IC 序列
    4. 最终统计各因子在不同滞后期的 IC Mean 和 IR
    说明：
    这种方法避免了多次循环计算，极大提升了效率。
    适用于大规模因子评估场景。

    :param df:
    :param factor_pattern:
    :param ret_col:
    :param date_col:
    :param asset_col:
    :param max_lag:
    :return:
    """

    # 1. 统一选取因子列名
    # 如果传入的是字符串，则视为正则匹配；如果已经是 List，则直接使用
    if isinstance(factor_pattern, str):
        factor_cols = df.select(pl.col(factor_pattern)).columns
    else:
        factor_cols = factor_pattern

    # 1. 预处理：构造滞后收益率并 Rank
    # 这一步保持不变，是最高效的对齐方式
    target_lags = [f"target_lag_{i}" for i in range(max_lag)]
    q = df.lazy().with_columns([
        pl.col(ret_col).shift(-i).over(asset_col).rank().over(date_col).alias(f"target_lag_{i}")
        for i in range(max_lag)
    ])

    # 2. 长表化处理：将所有因子转为一列，方便统一计算
    # 结果格式：[date, asset, target_lag_0...target_lag_n, factor_name, factor_value]
    q_long = q.unpivot(
        index=[date_col, asset_col] + target_lags,
        on=factor_cols,
        variable_name="factor",
        value_name="factor_value"
    ).with_columns(
        pl.col("factor_value").rank().over([date_col, "factor"])
    )

    # 3. 核心聚合：按日期和因子分组，一次性计算所有相关性
    # 得到的是 IC 的时间序列
    ic_series = q_long.group_by([date_col, "factor"]).agg([
        pl.corr("factor_value", pl.col(f"target_lag_{i}"), method="pearson").alias(f"lag_{i}")
        for i in range(max_lag)
    ])

    # 4. 最终统计：计算 Mean 和 IR
    # 这里不需要 collect 两次，直接在链式调用中完成
    decay_stats = ic_series.group_by("factor").agg([
        *[pl.col(f"lag_{i}").mean().alias(f"IC_Mean_Lag_{i}") for i in range(max_lag)],
        *[(pl.col(f"lag_{i}").mean() / pl.col(f"lag_{i}").std()).alias(f"IR_Lag_{i}") for i in range(max_lag)]
    ]).collect()

    return decay_stats


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


def batch_calc_factor_turnover(
        df: pl.DataFrame,
        factor_pattern: str = r"^factor_.*",
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        lag: int = 1
) -> pl.DataFrame:
    """
    大批量计算因子的自相关性 (衡量换手率)
    lag: 滞后天数，1代表日换手，5代表周换手
    """
    # 1. 自动获取因子列
    factor_cols = df.select(pl.col(factor_pattern)).columns

    # 2. 计算各因子自身的滞后相关性
    # 结果反映了因子的稳定性，值越大换手越低
    turnover_stats = (
        df.lazy()
        .with_columns([
            pl.col(f).shift(lag).over(asset_col).alias(f"{f}_lag")
            for f in factor_cols
        ])
        .group_by(date_col)
        .agg([
            pl.corr(pl.col(f).rank(), pl.col(f"{f}_lag").rank(), method="pearson").alias(f)
            for f in factor_cols
        ])
        .unpivot(index=date_col, on=factor_cols, variable_name="factor", value_name="autocorr")
        .group_by("factor")
        .agg([
            pl.col("autocorr").mean().alias("avg_autocorr"),
            # 换手率估算：1 - 平均自相关系数
            (1 - pl.col("autocorr").mean()).alias("turnover_estimate")
        ])
        .collect()
    )
    return turnover_stats


def batch_calc_quantile_returns(
        df: pl.DataFrame,
        factor_pattern: str = r"^factor_.*",
        ret_col: str = "target_ret",
        date_col: str = "DATE",
        n_bins: int = 5  # 分为5层
) -> pl.DataFrame:
    """
    大批量并行计算因子的分层收益
    """
    factor_cols = df.select(pl.col(factor_pattern)).columns

    # 1. 长表化并计算分层
    # 结果：[date, factor, factor_value, target_ret]
    q_long = df.lazy().unpivot(
        index=[date_col, "ASSET", ret_col],
        on=factor_cols,
        variable_name="factor",
        value_name="value"
    ).filter(pl.col("value").is_not_null())

    # 2. 截面分层 (关键步骤)
    # 对每一个日期、每一个因子，独立进行分位切割
    q_returns = (
        q_long.with_columns(
            pl.col("value")
            .qcut(n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])
            .over([date_col, "factor"])
            .alias("quantile")
        )
        # 3. 计算每日每层平均收益
        .group_by([date_col, "factor", "quantile"])
        .agg(pl.col(ret_col).mean().alias("daily_ret"))

        # 4. 计算累计收益 (累加或累乘)
        .sort(["factor", "quantile", date_col])
        .with_columns(
            (pl.col("daily_ret") + 1).product().over(["factor", "quantile"]).alias("cum_ret")
        )
        .collect()
    )
    return q_returns


def batch_factor_alpha_lens(
        df: pl.DataFrame,
        factor_pattern: str = r"^factor_.*",
        ret_col: str = "target_ret",
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        n_bins: int = 5,
        max_lag: int = 5
) -> pl.DataFrame:
    """
    【终极全能版】大批量因子体检引擎：IC/IR + 衰减 + 换手 + 分层收益
    """
    factor_cols = df.select(pl.col(factor_pattern)).columns

    # --- 第一部分：基础指标与衰减 (IC/IR/t-stat/WinRate) ---
    full_metrics = batch_calc_factor_full_metrics(
        df, factor_cols, ret_col, date_col, asset_col, max_lag
    )

    # --- 第二部分：稳定性指标 (Turnover/Autocorr) ---
    turnover_stats = batch_calc_factor_turnover(
        df, factor_pattern, date_col, asset_col
    )

    # --- 第三部分：实战收益指标 (Quantile Returns) ---
    # 我们从中提取多空年化收益和多空夏普
    q_rets = batch_calc_quantile_returns(
        df, factor_pattern, ret_col, date_col, n_bins
    )

    ls_metrics = (
        q_rets.pivot(index=[date_col, "factor"], on="quantile", values="daily_ret")
        .with_columns((pl.col("Q1") - pl.col(f"Q{n_bins}")).alias("ls_ret"))
        .group_by("factor")
        .agg([
            (pl.col("ls_ret").mean() * 242).alias("annual_ls_ret"),
            (pl.col("ls_ret").mean() / pl.col("ls_ret").std() * (242 ** 0.5)).alias("ls_sharpe")
        ])
    )

    # --- 最终合并所有维度 ---
    master_table = (
        full_metrics
        .join(turnover_stats, on="factor")
        .join(ls_metrics, on="factor")
    )

    return master_table
