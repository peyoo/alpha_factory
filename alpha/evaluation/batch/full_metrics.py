from typing import Union, List, Literal
import numpy as np
import polars as pl
import polars.selectors as cs
from alpha.utils.schema import F

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
    # 1. 落地并排序
    df = (df.collect() if isinstance(df, pl.LazyFrame) else df).sort([asset_col, date_col])

    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    factor_cols = df.select(f_selector).columns
    if not factor_cols:
        return pl.DataFrame()

    # 2. 预计算 Rank 并标记 Top/Btm 状态
    df_scored = df.with_columns([
        pl.col(f).rank().over(date_col).alias(f"{f}_rank") for f in factor_cols
    ]).with_columns([
        (pl.col(f"{f}_rank") > (pl.count().over(date_col) * (n_bins - 1) / n_bins)).alias(f"{f}_is_top")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_rank") <= (pl.count().over(date_col) / n_bins)).alias(f"{f}_is_btm")
        for f in factor_cols
    ])

    # 3. 生成下移信号与换手检测
    df_scored = df_scored.with_columns([
        pl.col(f"{f}_is_top").shift(1).over(asset_col).fill_null(False).alias(f"{f}_sig_top")
        for f in factor_cols
    ] + [
        pl.col(f"{f}_is_btm").shift(1).over(asset_col).fill_null(False).alias(f"{f}_sig_btm")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_is_top").cast(pl.Int8).diff().over(asset_col) == 1).fill_null(False).alias(f"{f}_buy_top")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_is_btm").cast(pl.Int8).diff().over(asset_col) == 1).fill_null(False).alias(f"{f}_buy_btm")
        for f in factor_cols
    ])

    # 4. 过滤股票池并聚合指标
    if pool_mask_col in df_scored.columns:
        df_scored = df_scored.filter(pl.col(pool_mask_col))

    daily_stats = df_scored.group_by(date_col).agg([
        pl.col(label_ret_col).mean().alias("market_avg"),
        *[pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],
        *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_top")).mean().alias(f"{f}_top_ret") for f in factor_cols],
        *[pl.col(label_ret_col).filter(pl.col(f"{f}_sig_btm")).mean().alias(f"{f}_btm_ret") for f in factor_cols],
        *[(pl.col(f"{f}_buy_top").sum() / (pl.count() / n_bins)).alias(f"{f}_to_top") for f in factor_cols],
        *[(pl.col(f"{f}_buy_btm").sum() / (pl.count() / n_bins)).alias(f"{f}_to_btm") for f in factor_cols],
    ]).sort(date_col)

    # 5. 汇总
    final_results = []
    total_days = daily_stats.height

    for f in factor_cols:
        ic_series = daily_stats.get_column(f"{f}_ic")
        # 直接计算 ic_mean
        ic_mean = ic_series.mean() or 0.0
        ic_std = ic_series.std() or 1e-9
        direction = 1 if ic_mean >= 0 else -1

        # 选桶逻辑
        if direction == 1:
            raw_ret = daily_stats.get_column(f"{f}_top_ret").fill_null(0.0).to_numpy()
            turnover_val = daily_stats.get_column(f"{f}_to_top").mean() or 0.0
        else:
            raw_ret = daily_stats.get_column(f"{f}_btm_ret").fill_null(0.0).to_numpy()
            turnover_val = daily_stats.get_column(f"{f}_to_btm").mean() or 0.0

        if mode == 'active':
            raw_ret = raw_ret - daily_stats.get_column("market_avg").fill_null(0.0).to_numpy()

        # 扣费并算净值
        net_daily_ret = raw_ret - (turnover_val * fee * 2)
        nav = np.cumprod(1.0 + net_daily_ret)

        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        vol = np.std(net_daily_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (vol + 1e-9)

        final_results.append({
            "factor": f,
            "ic_mean": ic_mean,  # 新增
            "ic_ir": ic_mean / ic_std,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_est": turnover_val,
            "direction": direction
        })

    return pl.DataFrame(final_results).sort("sharpe", descending=True)
