import polars.selectors as cs
from typing import List, Union
import polars as pl
from loguru import logger
import time
from alpha.utils.schema import F

def batch_calc_factor_ic_decay(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factors: Union[str, List[str]] = r"^factor_.*",
        label_for_ret: str = F.LABEL_FOR_RET,
        max_lag: int = 5,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
) -> pl.DataFrame:
    """
    大批量计算因子 IC 衰减图谱，评估因子预测能力的持续性。

    计算逻辑：
    1. **收益率预处理**：对 label_for_ret 进行 0 到 max_lag 的滞后处理，并进行截面 Rank 变换。
    2. **长表化 (Melting)**：将宽表形式的因子列转换为长表，使得所有因子共享一套计算逻辑。
    3. **批处理聚合**：按日期和因子分组，利用向量化运算一次性计算所有滞后期的 Spearman IC。
    4. **指标生成**：计算各因子在不同滞后期的 IC 均值和 ICIR。

    Returns:
        pl.DataFrame: 因子衰减统计表，Schema 如下：
        | 列名 | 类型 | 说明 |
        | :--- | :--- | :--- |
        | factor | String | 因子名称 |
        | IC_Mean_Lag_n | Float64 | 滞后 n 期的 IC 均值 (n=0..max_lag-1) |
        | IR_Lag_n | Float64 | 滞后 n 期的 ICIR (均值/标准差) |
    """
    start_time = time.perf_counter()
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # --- 1. 因子列名提取 ---
    f_selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    factor_cols = lf.select(f_selector).collect_schema().names()

    if not factor_cols:
        logger.error(f"⚠️ 无法匹配到任何因子 (模式: {factors})，返回空结果。")
        return pl.DataFrame()

    logger.info(f"🧬 开始计算 {len(factor_cols)} 个因子的衰减图谱 | 最大滞后: {max_lag} 天")

    # --- 2. 构造收益率滞后序列并 Rank (Spearman 准备) ---
    # 我们预先对收益率做 Rank，后续直接计算 Pearson 即可等价于 Spearman
    target_lags = [f"target_lag_{i}" for i in range(max_lag)]
    q = lf.with_columns([
        pl.col(label_for_ret).shift(-i).over(asset_col).rank().over(date_col).alias(f"target_lag_{i}")
        for i in range(max_lag)
    ])

    # --- 3. 长表化处理：将因子维度打散 ---
    # 这一步是为了避免写 Python 循环，充分利用 Polars 的并行聚合能力
    q_long = q.unpivot(
        index=[date_col, asset_col] + target_lags,
        on=factor_cols,
        variable_name="factor",
        value_name="factor_value"
    ).with_columns(
        # 因子值截面 Rank
        pl.col("factor_value").rank().over([date_col, "factor"])
    )

    # --- 4. 核心聚合计算 IC 时间序列 ---
    logger.debug("🔄 正在执行跨因子、跨滞后期的批量相关性并行计算...")
    ic_series = q_long.group_by([date_col, "factor"]).agg([
        pl.corr("factor_value", pl.col(f"target_lag_{i}"), method="pearson").alias(f"lag_{i}")
        for i in range(max_lag)
    ])

    # --- 5. 统计 Mean 和 IR ---
    decay_stats = ic_series.group_by("factor").agg([
        *[pl.col(f"lag_{i}").mean().alias(f"IC_Mean_Lag_{i}") for i in range(max_lag)],
        *[(pl.col(f"lag_{i}").mean() / pl.col(f"lag_{i}").std().fill_nan(1e-9)).alias(f"IR_Lag_{i}") for i in
          range(max_lag)]
    ]).collect()

    duration = time.perf_counter() - start_time
    logger.success(f"✅ 衰减图谱计算完成 | 耗时: {duration:.2f}s | 生成数据: {decay_stats.height} 行")
    return decay_stats
