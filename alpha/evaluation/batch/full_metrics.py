import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Union, List, Literal

from alpha.utils.schema import F


def batch_full_metrics(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ic_col: str = F.LABEL_FOR_RET,
        label_ret_col: str = F.LABEL_FOR_RET,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,  # 确保传入 asset 列名
        pool_mask_col: str = F.POOL_MASK,
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252,
        fee: float = 0.0015  # 单边交易费用
) -> pl.DataFrame:
    """
    全维度因子评估：集成 IC、净值收益、以及基于自相关性估算的换手成本。
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # 1. 预过滤股票池
    if pool_mask_col in lf.collect_schema().names():
        lf = lf.filter(pl.col(pool_mask_col))

    # 选择因子列
    factor_cols = lf.select(
        cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    ).collect_schema().names()

    if not factor_cols:
        return pl.DataFrame()

    # 2. 计算因子自相关性表达式 (用于换手成本估算)
    # 修复：确保使用传入的 asset_col 进行分组计算位移相关性
    autocorr_exprs = [
        pl.corr(f, pl.col(f).shift(1), method="spearman")
        .over(asset_col)
        .alias(f"{f}_stab")
        for f in factor_cols
    ]

    # 3. 截面聚合计算 (IC + Bin Returns + Stability)
    daily_raw = (
        lf.with_columns(autocorr_exprs)
        .group_by(date_col)
        .agg([
            pl.col(label_ret_col).mean().alias("market_avg"),
            # 计算截面 IC
            *[pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],
            # 修正：移除多余的逗号，确保聚合逻辑正确
            *[pl.col(f"{f}_stab").mean().alias(f"{f}_stab_avg") for f in factor_cols],
            # 分桶收益 (修正：btm 增加 method="random" 保持一致)
            *[pl.col(label_ret_col).filter(
                pl.col(f).rank(descending=True, method="random") <= (pl.count() / n_bins)
            ).mean().alias(f"{f}_top") for f in factor_cols],
            *[pl.col(label_ret_col).filter(
                pl.col(f).rank(descending=False, method="random") <= (pl.count() / n_bins)
            ).mean().alias(f"{f}_btm") for f in factor_cols]
        ])
        .collect()
    )

    results = []
    total_days = daily_raw.height
    mkt_avg = daily_raw.get_column("market_avg").to_numpy()

    for f in factor_cols:
        # --- 维度 1: 统计学指标 ---
        ic_series = daily_raw.get_column(f"{f}_ic")
        ic_mean = ic_series.mean() or 0.0
        ic_ir = ic_mean / (ic_series.std() + 1e-9)
        direction = 1 if ic_mean >= 0 else -1

        # --- 维度 2: 换手成本估算 ---
        # 换手率 ≈ 1 - 因子自相关性
        # 使用你设定的阈值 0.8 [cite: 2026-02-04] 作为参考
        avg_stab = daily_raw.get_column(f"{f}_stab_avg").mean() or 0.0
        est_turnover = max(0.0, 1.0 - avg_stab)
        daily_fee_deduction = est_turnover * fee * 2  # 双边扣费

        # --- 维度 3: 收益逻辑 ---
        top_ret = daily_raw.get_column(f"{f}_top").to_numpy()
        btm_ret = daily_raw.get_column(f"{f}_btm").to_numpy()

        if mode == 'long_only':
            raw_ret = top_ret if direction == 1 else btm_ret
        elif mode == 'long_short':
            raw_ret = (top_ret - btm_ret) * direction
        else:  # active
            target_ret = top_ret if direction == 1 else btm_ret
            raw_ret = target_ret - mkt_avg

        # 计算扣费后的净收益
        net_daily_ret = np.nan_to_num(raw_ret, nan=0.0) - daily_fee_deduction

        # --- 指标汇总 ---
        # 使用 (1 + r) 计算复利 NAV
        nav = np.cumprod(1.0 + net_daily_ret)
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        # 年化收益计算
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        ann_vol = np.std(net_daily_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (ann_vol + 1e-9)

        results.append({
            "factor": f,
            "ic_mean": ic_mean,
            "ic_ir": ic_ir,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_est": est_turnover,
            "direction": direction
        })

    return pl.DataFrame(results)
