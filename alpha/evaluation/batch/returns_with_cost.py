from typing import Union, Literal, List

import numpy as np
import polars as pl
import polars.selectors as cs

from alpha.utils.schema import F


def batch_quantile_returns_with_cost(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ret_col: str = "LABEL_FOR_RET",
        date_col: str = "DATE",
        pool_mask_col: str = "POOL_MASK",
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252,
        fee: float = 0.0015  # 新增：单边交易成本（含佣金、印花税、冲击成本）
) -> pl.DataFrame:
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    if pool_mask_col in lf.collect_schema().names():
        lf = lf.filter(pl.col(pool_mask_col))

    factor_cols = lf.select(
        cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)).collect_schema().names()

    # --- 核心改进：计算因子自相关性 (用于估算换手) ---
    # 利用 shift 计算因子值与其昨日的秩相关
    autocorr_exprs = [
        pl.corr(f, pl.col(f).shift(1), method="spearman").over(F.ASSET).alias(f"{f}_autocorr")
        for f in factor_cols
    ]

    daily_raw = (
        lf.with_columns(autocorr_exprs)
        .group_by(date_col)
        .agg([
            pl.col(label_ret_col).mean().alias("market_avg"),
            *[pl.corr(f, label_ret_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],
            # 计算全市场平均因子自相关性，用于换手率估算
            *[pl.col(f"{f}_autocorr").mean().alias(f"{f}_turnover_idx") for f in factor_cols],
            *[pl.col(label_ret_col).filter(
                pl.col(f).rank(descending=True, method="random") <= (pl.count() / n_bins)).mean().alias(f"{f}_top") for
              f in factor_cols],
            *[pl.col(label_ret_col).filter(pl.col(f).rank(descending=False) <= (pl.count() / n_bins)).mean().alias(
                f"{f}_btm") for f in factor_cols]
        ])
        .collect()
    )

    results = []
    total_days = daily_raw.height
    mkt_avg = daily_raw.get_column("market_avg").to_numpy()

    for f in factor_cols:
        avg_ic = daily_raw.get_column(f"{f}_ic").mean()
        direction = 1 if (avg_ic is not None and avg_ic >= 0) else -1

        top_ret = daily_raw.get_column(f"{f}_top").to_numpy()
        btm_ret = daily_raw.get_column(f"{f}_btm").to_numpy()

        # --- 核心改进：换手率扣费逻辑 ---
        # 估算换手率公式: 1 - Autocorr (这是一个简化的线性映射)
        # 真实的换手率与因子自相关性高度负相关
        avg_autocorr = daily_raw.get_column(f"{f}_turnover_idx").mean()
        estimated_turnover = max(0, 1 - (avg_autocorr if avg_autocorr is not None else 0))
        daily_cost = estimated_turnover * fee * 2  # 双边交易成本

        if mode == 'long_only':
            raw_ret = top_ret if direction == 1 else btm_ret
        elif mode == 'long_short':
            raw_ret = (top_ret - btm_ret) * direction
        else:
            target_ret = top_ret if direction == 1 else btm_ret
            raw_ret = target_ret - mkt_avg

        # 预扣费：将账面收益减去交易成本
        raw_ret = np.nan_to_num(raw_ret, nan=0.0)
        net_ret = raw_ret - daily_cost  # 扣除成本后的净收益

        nav = np.cumprod(1.0 + net_ret)
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        ann_vol = np.std(net_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (ann_vol + 1e-9)

        results.append({
            "factor": f,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_estimate": estimated_turnover,  # 输出换手率方便监控
            "direction": direction
        })

    return pl.DataFrame(results)
