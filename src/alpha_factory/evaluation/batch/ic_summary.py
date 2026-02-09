import polars.selectors as cs
from typing import List, Union
import polars as pl
from loguru import logger
import time
from alpha_factory.utils.schema import F


def batch_ic_summary(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_for_ic: str = F.LABEL_FOR_IC,
        date_col: str = F.DATE,
        pool_mask_col: str = F.POOL_MASK
) -> pl.DataFrame:
    """
    批量计算因子 IC (Information Coefficient) 指标摘要。

    计算逻辑：
    1. 筛选因子：根据正则或名称列表定位因子列。
    2. 过滤：应用股票池掩码并剔除 Label 为空的行。
    3. 时序计算：计算每日 Spearman Rank IC。
    4. 统计聚合：计算均值、标准差、IR、T统计量及胜率。

    Returns:
        pl.DataFrame: 结果数据框，每行代表一个因子，Schema 如下：
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 (e.g., 'factor_0') |
        | ic_mean | Float64 | 每日 IC 的算术平均值 |
        | ic_std | Float64 | 每日 IC 的标准差 |
        | ic_ir | Float64 | IC 信息比率 (ic_mean / ic_std) |
        | t_stat | Float64 | IC 序列的 T 统计量 (显著性指标) |
        | win_rate | Float64 | 胜率 (IC > 0 的天数占比) |
        | ic_mean_abs | Float64 | IC 均值的绝对值 (常用于进化目标) |
        | ic_ir_abs | Float64 | IC IR 的绝对值 (常用于进化目标) |
    """
    start_time = time.perf_counter()
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # --- 1. 因子列识别 ---
    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    try:
        current_schema = lf.collect_schema()
        factor_cols = lf.select(f_selector).collect_schema().names()
    except Exception as e:
        logger.error(f"❌ 因子选择器匹配失败: {e}")
        return pl.DataFrame()

    if not factor_cols:
        logger.warning(f"⚠️ 无法匹配到任何因子 (模式: {factors})，返回空结果。")
        return pl.DataFrame()

    # --- 2. 预过滤：股票池与有效 Label ---
    if pool_mask_col in current_schema.names():
        lf = lf.filter(pl.col(pool_mask_col))
        logger.debug(f"ℹ️ 已过滤股票池: {pool_mask_col}")

    # 必须确保 Label 列存在
    if label_for_ic not in current_schema.names():
        logger.error(f"❌ 关键列 '{label_for_ic}' 缺失，无法继续计算。")
        return pl.DataFrame()

    # --- 3. 执行聚合计算 ---
    logger.info(f"📊 启动 IC 聚合计算 | 因子数: {len(factor_cols)} | 计算模式: Spearman")

    try:
        ic_summary = (
            lf.select([date_col, label_for_ic] + factor_cols)
            # 过滤 Label 无效的行，防止对相关性产生噪音
            .drop_nulls(subset=[label_for_ic])
            .group_by(date_col)
            .agg([
                pl.corr(pl.col(f), pl.col(label_for_ic), method="spearman").alias(f)
                for f in factor_cols
            ])
            # 将宽表旋转为长表，方便后续按因子聚合
            .unpivot(index=date_col, on=factor_cols, variable_name="factor", value_name="ic")
            # 过滤无法计算 IC 的日期（如全停牌）
            .filter(pl.col("ic").is_not_nan() & pl.col("ic").is_not_null())
            .group_by("factor")
            .agg([
                pl.col("ic").mean().alias("ic_mean"),
                pl.col("ic").std().alias("ic_std"),
                # 数值稳定性修复：防止 std 为 0 导致的除以零错误
                (pl.col("ic").mean() / pl.col("ic").std().fill_nan(1e-9)).alias("ic_ir"),
                # T-Stat = Mean / Std * sqrt(N)
                (pl.col("ic").mean() / pl.col("ic").std().fill_nan(1e-9) * pl.count().sqrt()).alias("t_stat"),
                # WinRate = Count(IC > 0) / TotalCount
                (pl.col("ic").filter(pl.col("ic") > 0).count() / pl.count()).alias("win_rate")
            ])
            # 生成进化算法所需的绝对值指标
            .with_columns([
                pl.col("ic_mean").abs().alias("ic_mean_abs"),
                pl.col('ic_ir').abs().alias('ic_ir_abs')
            ])
            .collect()
        )

        duration = time.perf_counter() - start_time
        logger.success(
            f"✅ IC 摘要计算完成 | 耗时: {duration:.3f}s | 因子有效性: {ic_summary.height}/{len(factor_cols)}")
        return ic_summary

    except Exception as e:
        logger.exception(f"❌ 聚合计算链条崩溃: {e}")
        return pl.DataFrame()
