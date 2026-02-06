import re
from typing import Union, Literal, List
import numpy as np
import polars as pl


def batch_quantile_returns_with_cost(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_ret_col: str = "LABEL_FOR_RET",
        label_ic_col: str = "LABEL_FOR_IC",
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        pool_mask_col: str = "POOL_MASK",
        n_bins: int = 10,
        mode: Literal['long_only', 'long_short', 'active'] = 'long_only',
        annual_days: int = 252,
        fee: float = 0.0015
) -> pl.DataFrame:
    """
    高性能批量因子回测逻辑
    1. 使用截面自相关性 (1 - Cross-sectional Autocorr) 稳定估算换手率。
    2. 严格执行信号滞后 (Shift 1)，避免未来函数。
    3. 向量化计算，支持 LazyFrame 接入。
    """

    # --- 1. 数据准备与物理排序 ---
    # 必须先 collect 并排序，确保 shift(1).over(asset) 的物理意义正确
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # 自动识别因子列
    all_cols = lf.collect_schema().names()
    if isinstance(factors, str):
        factor_cols = [c for c in all_cols if re.match(factors, c)]
    else:
        factor_cols = [c for c in factors if c in all_cols]

    # 预过滤掉不需要的列，减少内存占用
    essential_cols = [date_col, asset_col, label_ret_col,label_ic_col]
    if pool_mask_col in all_cols:
        essential_cols.append(pool_mask_col)

    # 核心预处理：物理排序 -> 信号滞后
    df_sorted = (
        lf.select(essential_cols + factor_cols)
        .sort([asset_col, date_col])
        .with_columns([
            pl.col(f).shift(1).over(asset_col).alias(f"{f}_sig")
            for f in factor_cols
        ])
    )

    # --- 2. 截面聚合计算 ---
    # 应用股票池过滤
    if pool_mask_col in all_cols:
        df_sorted = df_sorted.filter(pl.col(pool_mask_col))

    # 构建聚合表达式
    # 我们同时计算：
    # - IC: Corr(Factor_T, Ret_T)
    # - Autocorr: Corr(Factor_T, Factor_T_Lag) -> 换手估算代理
    # - Returns: 基于 Factor_T_Lag 分箱的 Ret_T
    agg_exprs = [
        pl.col(label_ret_col).mean().alias("market_avg")
    ]

    count_expr = pl.count()  # 缓存计数表达式

    for f in factor_cols:
        # 稳定性：截面自相关 (Spearman)
        agg_exprs.append(pl.corr(f, f"{f}_sig", method="spearman").alias(f"{f}_autocorr"))
        # 预测力：IC
        agg_exprs.append(pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic"))

        # 分箱收益 (使用 sig 即 T-1 期的因子值)
        limit = count_expr / n_bins
        agg_exprs.append(
            pl.col(label_ret_col).filter(pl.col(f"{f}_sig").rank(descending=True) <= limit).mean().alias(f"{f}_top")
        )
        agg_exprs.append(
            pl.col(label_ret_col).filter(pl.col(f"{f}_sig").rank(descending=False) <= limit).mean().alias(f"{f}_btm")
        )

    # 执行计算
    daily_raw = df_sorted.group_by(date_col).agg(agg_exprs).collect().sort(date_col)

    # --- 3. 结果汇总与扣费 ---
    results = []
    total_days = daily_raw.height
    if total_days == 0:
        return pl.DataFrame()

    mkt_avg = daily_raw.get_column("market_avg").fill_null(0.0).to_numpy()

    for f in factor_cols:
        # IC 均值与方向
        avg_ic = daily_raw.get_column(f"{f}_ic").mean() or 0.0
        direction = 1 if avg_ic >= 0 else -1

        # 换手率估算：基于截面自相关性的均值
        # 这种方式极难产生 NaN，且对因子稳定性有极好的刻画
        avg_autocorr = daily_raw.get_column(f"{f}_autocorr").fill_nan(None).mean() or 0.0
        estimated_turnover = np.clip(1.0 - avg_autocorr, 0.0, 1.0)
        daily_cost = estimated_turnover * fee * 5  # 估算的双边成本

        # 收益序列处理
        top_ret = daily_raw.get_column(f"{f}_top").fill_null(0.0).to_numpy()
        btm_ret = daily_raw.get_column(f"{f}_btm").fill_null(0.0).to_numpy()

        if mode == 'long_only':
            raw_ret = top_ret if direction == 1 else btm_ret
        elif mode == 'long_short':
            raw_ret = (top_ret - btm_ret) * direction
        else:  # active 模式
            target_ret = top_ret if direction == 1 else btm_ret
            raw_ret = target_ret - mkt_avg

        # 计算净收益与指标
        # 增加 nan_to_num 防御，确保 cumprod 不会失效
        raw_ret = np.nan_to_num(raw_ret, nan=0.0)
        net_ret = raw_ret - daily_cost

        nav = np.cumprod(1.0 + net_ret)
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        ann_ret = (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        ann_vol = np.std(net_ret) * np.sqrt(annual_days)
        sharpe = ann_ret / (ann_vol + 1e-9)

        results.append({
            "factor": f,
            "ic_mean": avg_ic,
            "ann_ret": ann_ret,
            "sharpe": sharpe,
            "turnover_estimate": estimated_turnover,
            "direction": direction
        })

    return pl.DataFrame(results).sort("sharpe", descending=True)
