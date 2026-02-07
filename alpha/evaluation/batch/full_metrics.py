from typing import Union, List, Literal
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

from alpha.utils.schema import F

def batch_full_metrics(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ic_col: str = F.LABEL_FOR_IC,
        label_ret_col: str = F.LABEL_FOR_RET,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,  # ← 小微盘动态掩码
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252,
        fee: float = 0.0025
) -> pl.DataFrame:
    """
    批量因子评估（基于目标股票池）

    Args:
        pool_mask_col: 动态股票池掩码列（True = 在池内）
            示例：_POOL_MASK_（包含流通市值过滤 + 停牌过滤）
            每日动态变化，反映实际可投资范围


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

    # ✅ Step 3: 在【池内股票】上计算 Rank（核心改动）
    #           现在 Rank 范围是 1-N（N = 池内股票数），不是 1-5797
    lf = lf.with_columns([
        pl.col(f).rank().over(date_col).alias(f"{f}_rank")
        for f in factor_cols# ↑ 重要：rank() 现在基于过滤后的池内股票
        #        如果池内有 200 只，rank 范围是 1-200
        #        不再是全市场的 1-5797
    ]).with_columns([
        (pl.col(f"{f}_rank") > (pl.len().over(date_col) * (n_bins - 1) / n_bins))
        .alias(f"{f}_is_top")
        for f in factor_cols# ↑ Top 10% 现在基于池内分位数
        #   如果池内 200 只，Top 10% = rank > 180（20 只）
    ] + [
        (pl.col(f"{f}_rank") <= (pl.len().over(date_col) / n_bins))
        .alias(f"{f}_is_btm")
        for f in factor_cols
    ])

    # Step 4: 生成信号（shift、换手检测）
    lf = lf.with_columns([
        pl.col(f"{f}_is_top").shift(1).over(asset_col).fill_null(False)
        .alias(f"{f}_sig_top")
        for f in factor_cols
    ] + [
        pl.col(f"{f}_is_btm").shift(1).over(asset_col).fill_null(False)
        .alias(f"{f}_sig_btm")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_is_top").cast(pl.Int8).diff().over(asset_col) == 1)
        .fill_null(False).alias(f"{f}_buy_top")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_is_btm").cast(pl.Int8).diff().over(asset_col) == 1)
        .fill_null(False).alias(f"{f}_buy_btm")
        for f in factor_cols
    ])

    # Step 5: 日度聚合
    daily_stats = (
        lf.group_by(date_col)
        .agg([
            pl.col(label_ret_col).mean().alias("market_avg"),
            *[pl.corr(f, label_ic_col, method="spearman")
              .alias(f"{f}_ic") for f in factor_cols],
            *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_top"))
              .mean().alias(f"{f}_top_ret") for f in factor_cols],
            *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_btm"))
              .mean().alias(f"{f}_btm_ret") for f in factor_cols],
            *[(pl.col(f"{f}_buy_top").sum() / (pl.len() / n_bins))
              .alias(f"{f}_to_top") for f in factor_cols],
            *[(pl.col(f"{f}_buy_btm").sum() / (pl.len() / n_bins))
              .alias(f"{f}_to_btm") for f in factor_cols],
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
            "ic_ir": ic_mean / ic_std,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_est": turnover_val,
            "direction": direction
        })

    return pl.DataFrame(final_results).sort("sharpe", descending=True)
