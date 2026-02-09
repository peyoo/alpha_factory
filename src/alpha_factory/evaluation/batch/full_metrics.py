from typing import Union, List, Literal
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

from alpha_factory.utils.schema import F

def batch_full_metrics(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ic_col: str = F.LABEL_FOR_IC,
        label_ret_col: str = F.LABEL_FOR_RET,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252,
        fee: float = 0.0025
) -> pl.DataFrame:
    """
    批量因子评估（基于目标股票池）

    Args:
        df: 输入数据框，包含因子列、标签列及股票池掩码列
        pool_mask_col: 动态股票池掩码列（True = 在池内）
            示例：_POOL_MASK_（包含流通市值过滤 + 停牌过滤）
            每日动态变化，反映实际可投资范围
        factors: 因子列选择器（正则表达式或名称列表）
        label_ic_col: 用于计算 IC 的标签列
        label_ret_col: 用于计算收益率的标签列
        date_col: 日期列名称
        asset_col: 资产列名称
        n_bins: 分桶数量（用于多分位信号生成）
        mode: 评估模式
            - 'long_only': 仅多头头寸
            - 'long_short': 多空头寸
            - 'active': 相对于市场平均收益
        annual_days: 年化交易日天数（用于年化收益计算）
        fee: 单边交易费用率（用于换手成本估计）
    Returns:
        pl.DataFrame: 因子评估结果，Schema 如下：
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 (e.g., 'factor_0')
        | ic_mean | Float64 | 每日 IC 的算术平均值 |
        | ic_mean_abs | Float64 | IC 均值的绝对值 (常用于进化目标) |
        | ic_ir | Float64 | IC 信息比率 (ic_mean / ic_std) |
        | ic_ir_abs | Float64 | IC IR 的绝对
        | ann_ret | Float64 | 年化收益率 |
        | sharpe | Float64 | 夏普比率 |
        | turnover_est | Float64 | 换手率估计 |
        | direction | Int32 | 因子方向 (1=正向, -1=反向) |
    说明：
    该函数实现了基于目标股票池的批量因子评估，确保评估结果真实反映因子在可交易范围内的表现。


    关键改动：
        ✓ pool_mask_col 过滤在最前面（Step 1）
        ✓ 之后的 Rank/分桶 基于池内股票，不是全市场
        ✓ 因子评估的所有指标都是"相对于目标池"的真实表现
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = lf.sort([asset_col, date_col])

    # ⭐ Step 1: 最前面过滤到目标股票池（小微盘 + 可交易）
    schema_names = lf.collect_schema().names()
    if pool_mask_col in schema_names:
        lf = lf.filter(pl.col(pool_mask_col))
        logger.info(f"✓ 应用股票池掩码: {pool_mask_col}（已过滤到目标池）")
    else:
        logger.warning(f"⚠️ 未找到池掩码列 {pool_mask_col}，使用全市场数据")

    # Step 2: 获取因子列
    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    factor_cols = lf.select(f_selector).collect_schema().names()
    if not factor_cols:
        return pl.DataFrame()

    lf = lf.with_columns([
        pl.col(f).rank().over(date_col).alias(f"{f}_rank")
        for f in factor_cols
    ]).with_columns([
        (pl.col(f"{f}_rank") > (pl.len().over(date_col) * (n_bins - 1) / n_bins))
        .alias(f"{f}_is_top") for f in factor_cols
    ] + [
        (pl.col(f"{f}_rank") <= (pl.len().over(date_col) / n_bins))
        .alias(f"{f}_is_btm") for f in factor_cols
    ])

    # Step 4: 生成信号（shift、换手检测）
    lf = lf.with_columns([
        pl.col(f"{f}_is_top").shift(1).over(asset_col).fill_null(False)
        .alias(f"{f}_sig_top") for f in factor_cols
    ] + [
        pl.col(f"{f}_is_btm").shift(1).over(asset_col).fill_null(False)
        .alias(f"{f}_sig_btm") for f in factor_cols
    ]).with_columns([
        (pl.col(f"{f}_is_top") & ~pl.col(f"{f}_sig_top"))
        .alias(f"{f}_buy_top") for f in factor_cols
    ] + [
        (pl.col(f"{f}_is_btm") & ~pl.col(f"{f}_sig_btm"))
        .alias(f"{f}_buy_btm") for f in factor_cols
    ])

    # Step 5: 日度聚合
    daily_stats = (
        lf.group_by(date_col)
        .agg([
            pl.col(label_ret_col).mean().alias("market_avg"),

            *[pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],

            *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_top")).mean().alias(f"{f}_top_ret") for f in factor_cols],

            *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_btm")).mean().alias(f"{f}_btm_ret") for f in factor_cols],

            *[(pl.col(f"{f}_buy_top").sum() / (pl.len() / n_bins)).alias(f"{f}_to_top") for f in factor_cols],

            *[(pl.col(f"{f}_buy_btm").sum() / (pl.len() / n_bins)).alias(f"{f}_to_btm") for f in factor_cols],
        ])
        .sort(date_col)
        .collect()
    )

    # Step 6: 汇总统计
    final_results = []
    total_days = daily_stats.height

    for f in factor_cols:
        ic_series = daily_stats.get_column(f"{f}_ic")
        ic_mean = ic_series.mean() or 0.0
        ic_mean_abs = abs(ic_mean)
        ic_std = ic_series.std() or 1e-9
        direction = 1 if ic_mean >= 0 else -1

        if direction == 1:
            raw_ret = daily_stats.get_column(f"{f}_top_ret").fill_null(0.0).to_numpy()
            turnover_val = daily_stats.get_column(f"{f}_to_top").mean() or 0.0
        else:
            raw_ret = daily_stats.get_column(f"{f}_btm_ret").fill_null(0.0).to_numpy()
            turnover_val = daily_stats.get_column(f"{f}_to_btm").mean() or 0.0

        if mode == 'active':
            raw_ret = raw_ret - daily_stats.get_column("market_avg").fill_null(0.0).to_numpy()

        net_daily_ret = raw_ret - (turnover_val * fee * 2)
        nav = np.cumprod(1.0 + net_daily_ret)

        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        vol = np.std(net_daily_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (vol + 1e-9)

        final_results.append({
            "factor": f,
            "ic_mean": ic_mean,
            'ic_mean_abs': ic_mean_abs,
            "ic_ir": ic_mean / ic_std,
            "ic_ir_abs": ic_mean_abs / ic_std,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_est": turnover_val,
            "direction": direction
        })

    return pl.DataFrame(final_results).sort("sharpe", descending=True)
