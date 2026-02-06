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
    # ✅ 1. 保持 LazyFrame（不立即 collect），直到 group_by 前
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = lf.sort([asset_col, date_col])

    # ✅ 2. 获取因子列（用 collect_schema 无需物化）
    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    factor_cols = lf.select(f_selector).collect_schema().names()
    if not factor_cols:
        return pl.DataFrame()

    # ✅ 3. 在 LazyFrame 中构建完整查询链（无内存开销）
    # 预计算 Rank 并标记 Top/Btm 状态
    lf = lf.with_columns([
        pl.col(f).rank().over(date_col).alias(f"{f}_rank") for f in factor_cols
    ]).with_columns([
        (pl.col(f"{f}_rank") > (pl.len().over(date_col) * (n_bins - 1) / n_bins)).alias(f"{f}_is_top")
        for f in factor_cols
    ] + [
        (pl.col(f"{f}_rank") <= (pl.len().over(date_col) / n_bins)).alias(f"{f}_is_btm")
        for f in factor_cols
    ])

    # 生成下移信号与换手检测
    lf = lf.with_columns([
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

    # ✅ 4. 检查 schema（无需 collect）
    schema_names = lf.collect_schema().names()
    if pool_mask_col in schema_names:
        lf = lf.filter(pl.col(pool_mask_col))

    # ✅ 5. 在 group_by 时执行 collect（最后才物化）
    # 此时 Polars 已经优化了整个查询计划
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
        .collect()  # ← 只在这里 collect
    )

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
