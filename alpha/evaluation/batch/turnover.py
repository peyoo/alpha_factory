import polars.selectors as cs
from typing import List, Union
import polars as pl
from loguru import logger
import time
from alpha.utils.schema import F

def batch_calc_factor_turnover(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        lag: int = 1
) -> pl.DataFrame:
    """
    大批量计算因子的截面自相关性，用于估算因子换手率和逻辑稳定性。

    计算逻辑：
    1. **滞后对齐**：按资产（ASSET）对因子值进行位移（shift），获取前 T 期的因子值。
    2. **每日计算**：在每个交易日（DATE）截面上，计算当前因子值与滞后因子值的 Rank 相关性（Spearman）。
    3. **聚合统计**：计算全时段自相关性的均值。自相关性越接近 1，因子越稳定，换手越低。

    Returns:
        pl.DataFrame: 因子稳定性统计表。
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | avg_autocorr | Float64 | 因子值序列的平均截面自相关系数 (越接近 1 越稳定) |
        | turnover_estimate | Float64 | 换手率估算值 (1 - avg_autocorr)，用于惩罚高频因子 |
    """
    start_time = time.perf_counter()
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # --- 1. 自动获取因子列 ---
    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    try:
        factor_cols = lf.select(f_selector).collect_schema().names()
    except Exception as e:
        logger.error(f"❌ 因子选择器匹配失败: {e}")
        return pl.DataFrame()

    if not factor_cols:
        logger.warning(f"⚠️ 无法匹配到任何因子 (模式: {factors})，返回空结果。")
        return pl.DataFrame()

    logger.info(f"🔄 开始计算 {len(factor_cols)} 个因子的自相关性 (Lag={lag})")

    # --- 2. 核心计算链路 ---
    # 利用 Polars 的 over() 窗口函数实现高效的个股滞后对齐
    try:
        turnover_stats = (
            lf.select([date_col, asset_col] + factor_cols)
            .with_columns([
                pl.col(f).shift(lag).over(asset_col).alias(f"{f}_lag")
                for f in factor_cols
            ])
            .group_by(date_col)
            .agg([
                # 计算 Spearman 相关性 (通过对 Rank 后的值计算 Pearson 实现)
                pl.corr(pl.col(f).rank(), pl.col(f"{f}_lag").rank(), method="pearson").alias(f)
                for f in factor_cols
            ])
            # 将宽表转为长表：[DATE, factor, autocorr]
            .unpivot(index=date_col, on=factor_cols, variable_name="factor", value_name="autocorr")
            .group_by("factor")
            .agg([
                # 过滤掉无法计算自相关的日期（如全停牌或初始几日）
                pl.col("autocorr").filter(pl.col("autocorr").is_not_nan()).mean().alias("avg_autocorr")
            ])
            .with_columns([
                # 换手率估算：1 - 平均自相关。
                # 注意：这只是一个线性估算，实盘换手还受持仓权重的非线性映射影响。
                (1 - pl.col("avg_autocorr")).alias("turnover_estimate")
            ])
            .collect()
        )

        duration = time.perf_counter() - start_time
        logger.success(f"✅ 因子换手率估算完成 | 耗时: {duration:.2f}s | 因子数: {len(factor_cols)}")
        return turnover_stats

    except Exception as e:
        logger.exception(f"❌ 计算因子自相关时崩溃: {e}")
        return pl.DataFrame()
