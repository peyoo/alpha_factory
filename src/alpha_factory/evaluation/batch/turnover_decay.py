import polars.selectors as cs
from typing import List, Union
import polars as pl
from loguru import logger
import time
from alpha_factory.utils.schema import F


def batch_calc_factor_turnover(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        label_col=F.LABEL_FOR_RET,
        n_bins: int = 10,
        lag: int = 1,
        descending = False
) -> pl.DataFrame:
    """
    基于 Rank 变化比例法批量计算因子换手率（Top 桶）。

    ⚠️ **重要说明**：本函数仅计算 **Top 桶**（因子值最大的前 1/n_bins）的换手率，
    不区分因子方向（IC 正/负）。若需根据 IC 自动选择 Top/Btm 桶，请使用
    `batch_calc_factor_turnover_with_direction()` 函数。

    计算逻辑：
    1. **Rank 计算**：每日截面对因子值进行降序排名，值大排名靠前。
    2. **Top 桶标记**：标记排名在前 1/n_bins 的资产为 Top 持仓。
    3. **信号滞后**：将 Top 标记向后位移 lag 期，获取前期持仓状态。
    4. **换手计算**：换手率 = 新进入 Top 的数量 / Top 总数量。

    Args:
        df: 输入数据，包含因子列
        factors: 因子列名正则表达式或列表
        date_col: 日期列名
        asset_col: 资产列名
        n_bins: 分桶数量，Top 桶为前 1/n_bins
        lag: 滞后期数，默认 1
        descending: 排名是否降序，默认 False（值大排名靠前）

    Returns:
        pl.DataFrame: 因子换手率统计表。
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | avg_turnover | Float64 | 平均换手率 (新进入 Top / Top 总数) |
        | turnover_std | Float64 | 换手率标准差 |

    Example:
        >>> df = pl.DataFrame({
        ...     "_DATE_": [20240101, 20240101, 20240102],
        ...     "_ASSET_": ["A", "B", "A"],
        ...     "factor_1": [0.5, 0.7, 0.6]
        ... })
        >>> batch_calc_factor_turnover(df, "factor_1", n_bins=2)
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

    logger.info(f"🔄 开始计算 {len(factor_cols)} 个因子的换手率 (Rank变化法, n_bins={n_bins}, lag={lag})")

    try:
        # --- 2. 计算 Rank 并标记 Top 桶 ---
        lf_ranked = (
            lf.select([date_col, asset_col] + factor_cols)
            .sort([asset_col, date_col])  # 确保 over() 分组内有序
            .with_columns(
                pl.count().over(date_col).alias("_daily_count_")
            )
            .with_columns([
                # 降序排名：值大 = 排名靠前 = Top
                (pl.col(f).rank(descending=descending).over(date_col)
                 <= (pl.col("_daily_count_") / n_bins)).alias(f"{f}_is_top")
                for f in factor_cols
            ]))

        # --- 3. 计算滞后信号 ---
        lf_with_lag = lf_ranked.with_columns([
            pl.col(f"{f}_is_top").shift(lag).over(asset_col).fill_null(False).alias(f"{f}_was_top")
            for f in factor_cols
        ])

        # --- 4. 按日聚合计算换手 ---
        daily_turnover = (
            lf_with_lag
            .group_by(date_col)
            .agg([
                # 新进入 = 当前是 Top 且之前不是
                (pl.col(f"{f}_is_top") & ~pl.col(f"{f}_was_top")).sum().alias(f"{f}_new_in")
                for f in factor_cols
            ] + [
                pl.col(f"{f}_is_top").sum().alias(f"{f}_top_count")
                for f in factor_cols
            ])
        )

        # --- 5. 计算换手率并转为长表 ---
        # 先计算每日换手率
        daily_turnover = daily_turnover.with_columns([
            (pl.col(f"{f}_new_in") / pl.col(f"{f}_top_count")).fill_null(0.0).alias(f"{f}_turnover")
            for f in factor_cols
        ])

        # 转为长表并聚合
        turnover_cols = [f"{f}_turnover" for f in factor_cols]
        turnover_stats = (
            daily_turnover
            .select([date_col] + turnover_cols)
            .unpivot(
                index=date_col,
                on=turnover_cols,
                variable_name="factor_raw",
                value_name="turnover"
            )
            # 还原因子名（去掉 _turnover 后缀）
            .with_columns(
                pl.col("factor_raw").str.replace("_turnover$", "").alias("factor")
            )
            .group_by("factor")
            .agg([
                pl.col("turnover").filter(pl.col("turnover").is_finite()).mean().alias("avg_turnover"),
                pl.col("turnover").filter(pl.col("turnover").is_finite()).std().alias("turnover_std"),
            ])
            .sort("avg_turnover")
            .collect()
        )

        duration = time.perf_counter() - start_time
        logger.success(f"✅ 因子换手率估算完成 | 耗时: {duration:.2f}s | 因子数: {len(factor_cols)}")
        return turnover_stats

    except Exception as e:
        logger.exception(f"❌ 计算因子换手率时崩溃: {e}")
        return pl.DataFrame()


def batch_calc_factor_turnover_with_direction(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_col: str = F.LABEL_FOR_RET,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        n_bins: int = 10,
        lag: int = 1
) -> pl.DataFrame:
    """
    基于 Rank 变化比例法计算因子换手率，根据 IC 自动选择 Top/Btm 桶。

    **核心差异**：本函数根据因子与标签的相关性（IC）判断因子方向：
    - IC ≥ 0：因子值大 = 好 → 计算 **Top 桶**的换手率
    - IC < 0：因子值小 = 好 → 计算 **Btm 桶**的换手率

    这确保计算的换手率与实际持仓策略相对应。

    Args:
        df: 输入数据，包含因子列和标签列
        factors: 因子列名正则表达式或列表
        label_col: 用于计算 IC 的标签列（如日收益率）
        date_col: 日期列名
        asset_col: 资产列名
        n_bins: 分桶数量，Top/Btm 桶各占 1/n_bins
        lag: 信号滞后期数，默认 1

    Returns:
        pl.DataFrame: 因子换手率统计表，额外包含方向指示。
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | ic_mean | Float64 | 平均 IC |
        | direction | Int32 | 方向（1=看涨，-1=看跌） |
        | side | String | 实际持仓桶（top 或 btm） |
        | avg_turnover | Float64 | 平均换手率 |
        | turnover_std | Float64 | 换手率标准差 |
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

    logger.info(f"🔄 开始计算 {len(factor_cols)} 个因子的换手率（含 IC 方向判断）")

    try:
        # --- 2. 计算 IC 判断因子方向 ---
        # 修复：先选择有效的数据（排除 null 和无穷值）
        lf_valid = (
            lf.select([date_col] + factor_cols + [label_col])
            .with_columns([
                pl.col(f).fill_null(strategy="forward").alias(f)
                for f in factor_cols + [label_col]
            ])
        )

        ic_daily = (
            lf_valid
            .group_by(date_col)
            .agg([
                pl.corr(f, label_col, method="spearman").alias(f"{f}_ic")
                for f in factor_cols
            ])
            .collect()
        )

        logger.debug("IC 计算完成，结果摘要:")
        for f in factor_cols:
            ic_col = ic_daily.get_column(f"{f}_ic")
            logger.debug(f"  {f}: null={ic_col.is_null().sum()}, nan={(~ic_col.is_finite()).sum()}, mean={ic_col.mean()}")

        # 提取各因子平均 IC 并判断方向
        factor_directions = {}
        factor_ics = {}
        for f in factor_cols:
            ic_series = ic_daily.get_column(f"{f}_ic").drop_nulls()

            # 过滤有限值（排除 NaN, Inf）
            ic_finite = ic_series.filter(ic_series.is_finite())

            if ic_finite.len() > 0:
                avg_ic = ic_finite.mean()
            else:
                logger.warning(f"⚠️ {f} 的 IC 全部为 NaN/Inf，使用默认方向 'top'")
                avg_ic = 0.0

            factor_directions[f] = "top" if avg_ic >= 0 else "btm"
            factor_ics[f] = avg_ic

        logger.debug("IC 方向判断完成")
        for f in factor_cols:
            logger.debug(f"  {f}: IC={factor_ics[f]:.6f} → {factor_directions[f]}")

        # --- 3. 计算 Rank 并标记 Top/Btm 桶 ---
        lf_ranked = (
            lf.select([date_col, asset_col] + factor_cols)
            .sort([asset_col, date_col])  # 确保 over() 分组内有序
            .with_columns(
                pl.len().over(date_col).alias("_daily_count_")
            )
            .with_columns([
                (pl.col(f).rank(descending=True).over(date_col)
                 <= (pl.col("_daily_count_") / n_bins)).alias(f"{f}_is_top")
                for f in factor_cols
            ] + [
                (pl.col(f).rank(descending=True).over(date_col)
                 > (pl.col("_daily_count_") * (n_bins - 1) / n_bins)).alias(f"{f}_is_btm")
                for f in factor_cols
            ])
        )

        # --- 4. 计算滞后信号 ---
        lf_with_lag = lf_ranked.with_columns([
            pl.col(f"{f}_is_top").shift(lag).over(asset_col).fill_null(False).alias(f"{f}_was_top")
            for f in factor_cols
        ] + [
            pl.col(f"{f}_is_btm").shift(lag).over(asset_col).fill_null(False).alias(f"{f}_was_btm")
            for f in factor_cols
        ])

        # --- 5. 按日聚合计算换手 ---
        daily_turnover = (
            lf_with_lag
            .group_by(date_col)
            .agg([
                (pl.col(f"{f}_is_top") & ~pl.col(f"{f}_was_top")).sum().alias(f"{f}_new_in_top")
                for f in factor_cols
            ] + [
                (pl.col(f"{f}_is_btm") & ~pl.col(f"{f}_was_btm")).sum().alias(f"{f}_new_in_btm")
                for f in factor_cols
            ] + [
                pl.col(f"{f}_is_top").sum().alias(f"{f}_top_count")
                for f in factor_cols
            ] + [
                pl.col(f"{f}_is_btm").sum().alias(f"{f}_btm_count")
                for f in factor_cols
            ])
            .with_columns([
                (pl.col(f"{f}_new_in_top") / pl.col(f"{f}_top_count")).fill_null(0.0).alias(f"{f}_turnover_top")
                for f in factor_cols
            ] + [
                (pl.col(f"{f}_new_in_btm") / pl.col(f"{f}_btm_count")).fill_null(0.0).alias(f"{f}_turnover_btm")
                for f in factor_cols
            ])
            .collect()
        )

        # --- 6. 根据 IC 方向选择对应桶的换手率 ---
        results = []
        for f in factor_cols:
            side = factor_directions[f]
            turnover_col = f"{f}_turnover_{side}"
            turnover_series = daily_turnover.get_column(turnover_col).filter(
                daily_turnover.get_column(turnover_col).is_finite()
            )
            results.append({
                "factor": f,
                "ic_mean": factor_ics[f],
                "direction": 1 if side == "top" else -1,
                "side": side,
                "avg_turnover": turnover_series.mean() or 0.0,
                "turnover_std": turnover_series.std() or 0.0,
            })

        result_df = pl.DataFrame(results).sort("avg_turnover")

        duration = time.perf_counter() - start_time
        logger.success(f"✅ 因子换手率（含方向判断）计算完成 | 耗时: {duration:.2f}s | 因子数: {len(factor_cols)}")
        return result_df

    except Exception as e:
        logger.exception(f"❌ 计算因子换手率时崩溃: {e}")
        return pl.DataFrame()


def batch_calc_factor_turnover_single_agg(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        label_col=F.LABEL_FOR_RET,
        n_bins: int = 10,
        lag: int = 1
) -> pl.DataFrame:
    """
    单行聚合方式计算因子换手率（高效版）。

    **优势**：在一个聚合步骤中计算所有统计量，减少中间步骤，性能更优。

    计算逻辑同 batch_calc_factor_turnover()，但使用单行 agg() 实现：
    1. 计算 Rank 并标记 Top 桶
    2. 计算滞后信号
    3. 按日期聚合并计算换手率统计

    Args:
        df: 输入数据，包含因子列
        factors: 因子列名正则表达式或列表
        date_col: 日期列名
        asset_col: 资产列名
        n_bins: 分桶数量，Top 桶为前 1/n_bins
        lag: 滞后期数，默认 1

    Returns:
        pl.DataFrame: 因子换手率统计表（与 batch_calc_factor_turnover() 相同格式）
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | avg_turnover | Float64 | 平均换手率 |
        | turnover_std | Float64 | 换手率标准差 |

    Example:
        >>> df = pl.DataFrame({
        ...     "DATE": [20240101, 20240101, 20240102],
        ...     "ASSET": ["A", "B", "A"],
        ...     "factor_1": [0.5, 0.7, 0.6],
        ...     "factor_2": [0.3, 0.8, 0.4]
        ... })
        >>> batch_calc_factor_turnover_single_agg(df, n_bins=2)
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

    logger.info(f"🔄 开始计算 {len(factor_cols)} 个因子的换手率 (单行聚合, n_bins={n_bins}, lag={lag})")

    try:
        # --- 2. 数据预处理：计算 Rank + 滞后信号 ---
        lf_prep = (
            lf.select([date_col, asset_col] + factor_cols)
            .sort([asset_col, date_col])
            .with_columns(
                pl.len().over(date_col).alias("_daily_count_")
            )
            .with_columns([
                # 当前是否在 Top 桶
                (pl.col(f).rank(descending=True).over(date_col)
                 <= (pl.col("_daily_count_") / n_bins)).alias(f"{f}_is_top")
                for f in factor_cols
            ])
            .with_columns([
                # 昨日是否在 Top 桶（滞后 lag 期）
                pl.col(f"{f}_is_top").shift(lag).over(asset_col)
                .fill_null(False).alias(f"{f}_was_top")
                for f in factor_cols
            ])
        )

        # --- 3. 单行聚合：一次性计算所有换手率统计 ---
        daily_stats = (
            lf_prep
            .group_by(date_col)
            .agg([
                # 对每个因子，计算 Top 桶大小、新进入数量和换手率
                *[
                    (
                        (pl.col(f"{f}_is_top") & ~pl.col(f"{f}_was_top")).sum()
                        / pl.col(f"{f}_is_top").sum()
                    ).fill_null(0.0).alias(f"{f}_turnover")
                    for f in factor_cols
                ]
            ])
            .collect()
        )

        # --- 4. 转为长表并最终聚合 ---
        turnover_cols = [f"{f}_turnover" for f in factor_cols]
        result_df = (
            daily_stats
            .select([date_col] + turnover_cols)
            .unpivot(
                index=date_col,
                on=turnover_cols,
                variable_name="factor_raw",
                value_name="turnover"
            )
            .with_columns(
                pl.col("factor_raw").str.replace("_turnover$", "").alias("factor")
            )
            .group_by("factor")
            .agg([
                pl.col("turnover").filter(pl.col("turnover").is_finite()).mean().alias("avg_turnover"),
                pl.col("turnover").filter(pl.col("turnover").is_finite()).std().alias("turnover_std"),
            ])
            .sort("avg_turnover")
        )

        duration = time.perf_counter() - start_time
        logger.success(f"✅ 因子换手率（单行聚合）计算完成 | 耗时: {duration:.2f}s | 因子数: {len(factor_cols)}")
        return result_df

    except Exception as e:
        logger.exception(f"❌ 计算因子换手率时崩溃: {e}")
        return pl.DataFrame()


def batch_calc_factor_turnover_by_autocorr(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        lag: int = 1,
        method: str = "spearman"
) -> pl.DataFrame:
    """
    基于截面自相关法计算因子换手率（轻量级版本）。

    **原理**：因子的截面自相关性反映其排序的稳定性。
    - 自相关性越高 → 排序越稳定 → 换手率越低
    - 自相关性越低 → 排序变化越大 → 换手率越高

    换手率估算公式：**estimated_turnover ≈ 1 - autocorr(Factor_T, Factor_{T-lag})**

    **优势**：
    1. 无需分桶，计算简单快速
    2. 无需指定 n_bins 参数
    3. 对因子的排序稳定性有直观理解
    4. 性能最优（仅需日度相关系数计算）

    **局限性**：
    1. 这是换手率的代理指标，不是精确值
    2. 需要足够的交叉截面样本（至少 20+ 只资产）
    3. 假设线性关系

    Args:
        df: 输入数据，包含因子列
        factors: 因子列名正则表达式或列表
        date_col: 日期列名
        asset_col: 资产列名
        lag: 滞后期数（计算相关性时的间隔），默认 1
        method: 相关性方法，'spearman' 或 'pearson'，默认 'spearman'

    Returns:
        pl.DataFrame: 因子自相关性统计表
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | avg_autocorr | Float64 | 平均自相关性 (-1 ~ 1) |
        | autocorr_std | Float64 | 自相关性标准差 |
        | estimated_turnover | Float64 | 估计换手率 (1 - avg_autocorr) |

    Example:
        >>> df = pl.DataFrame({
        ...     "DATE": [20240101, 20240102, 20240103],
        ...     "ASSET": ["A", "B", "A"],
        ...     "factor_1": [0.5, 0.7, 0.6]
        ... })
        >>> batch_calc_factor_turnover_by_autocorr(df)
        # 输出:
        # factor   | avg_autocorr | autocorr_std | estimated_turnover
        # factor_1 | 0.85         | 0.12         | 0.15
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

    logger.info(f"🔄 开始计算 {len(factor_cols)} 个因子的自相关性 (截面法, lag={lag}, method={method})")

    try:
        # --- 2. 计算截面自相关性 ---
        # 核心思想：对每个因子，计算 T 期与 T-lag 期的截面排序相关性
        lf_autocorr = (
            lf.select([date_col, asset_col] + factor_cols)
            .sort([asset_col, date_col])
            .with_columns([
                # 计算滞后值（同一资产的 lag 期前的值）
                pl.col(f).shift(lag).over(asset_col).alias(f"{f}_lag")
                for f in factor_cols
            ])
            .group_by(date_col)
            .agg([
                # 计算截面相关性（同一日期内，不同资产之间的相关性）
                pl.corr(f, f"{f}_lag", method=method).alias(f"{f}_autocorr")
                for f in factor_cols
            ])
            .collect()
        )

        logger.debug("自相关性计算完成，结果摘要:")
        for f in factor_cols:
            col = lf_autocorr.get_column(f"{f}_autocorr")
            valid_count = col.is_not_null().sum()
            avg_val = col.drop_nulls().mean() if valid_count > 0
