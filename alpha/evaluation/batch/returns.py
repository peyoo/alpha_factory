import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Union, List, Literal


def batch_quantile_returns(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ret_col: str = "LABEL_FOR_RET",
        date_col: str = "DATE",
        pool_mask_col: str = "POOL_MASK",
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252
) -> pl.DataFrame:
    """
    支持多模式(Long-Only, Long-Short, Active)的 GP 适应度评估函数
    没有考虑交易成本与滑点，仅用于快速评估因子收益能力。
    对于高频换手率因子误差较大，建议结合 batch_full_metrics 进行综合评估。
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    if pool_mask_col in lf.collect_schema().names():
        lf = lf.filter(pl.col(pool_mask_col))

    factor_cols = lf.select(
        cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)).collect_schema().names()
    if not factor_cols:
        return pl.DataFrame()

    # 1. 截面聚合计算
    daily_raw = (
        lf.group_by(date_col)
        .agg([
            pl.col(label_ret_col).mean().alias("market_avg"),
            *[pl.corr(f, label_ret_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],
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
        # 2. 方向判定
        avg_ic = daily_raw.get_column(f"{f}_ic").mean()
        direction = 1 if (avg_ic is not None and avg_ic >= 0) else -1

        # 3. 根据 mode 确定每日收益逻辑
        top_ret = daily_raw.get_column(f"{f}_top").to_numpy()
        btm_ret = daily_raw.get_column(f"{f}_btm").to_numpy()

        # 核心逻辑切换
        if mode == 'long_only':
            # 仅做多因子最强的一端 (Top if dir=1, Btm if dir=-1)
            raw_ret = top_ret if direction == 1 else btm_ret
        elif mode == 'long_short':
            # 多空对冲：多头桶 - 空头桶
            # 如果 dir=1, 则 Top-Btm; 如果 dir=-1, 则 Btm-Top
            raw_ret = (top_ret - btm_ret) * direction
        else:  # active 模式
            # 超额收益：多头桶 - 市场平均
            target_ret = top_ret if direction == 1 else btm_ret
            raw_ret = target_ret - mkt_avg

        # 4. 复利与指标计算
        raw_ret = np.nan_to_num(raw_ret, nan=0.0)
        # 注意：多空模式下，由于是价差收益，计算 nav 时逻辑略有不同，但为了统一评价体系仍采用 (1+r) 累乘
        nav = np.cumprod(1.0 + raw_ret)

        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        ann_vol = np.std(raw_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (ann_vol + 1e-9)

        results.append({
            "factor": f,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "direction": direction,
            "mode": mode
        })

    return pl.DataFrame(results)
