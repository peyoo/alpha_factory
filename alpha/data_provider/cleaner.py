"""
数据清洗流水线 (Data Cleaner)。

本模块实现从 Pandas DataFrame (Tushare 返回格式) 到 Polars LazyFrame (系统标准格式) 的
完整转换过程，包括：
- 类型转换
- 字段重命名
- 单位标准化
- 后复权价格计算
"""

import polars as pl
from loguru import logger

from alpha.data_provider.schema import (
    DAILY_BARS_MAPPING,
    DAILY_BARS_SCHEMA,
    CALENDAR_MAPPING,
    CALENDAR_SCHEMA,
    ADJ_FACTOR_MAPPING,
    ADJ_FACTOR_SCHEMA,
    DAILY_BASIC_MAPPING,
    DAILY_BASIC_SCHEMA,
    STOCK_BASIC_MAPPING,
    STOCK_BASIC_SCHEMA,
    TUSHARE_VOLUME_UNIT,
    TUSHARE_AMOUNT_UNIT,
    validate_schema,
)


# ============================================================================
# 日线行情清洗
# ============================================================================


def clean_daily_bars(
    df_pandas,  # pd.DataFrame
    with_adjustment: bool = True,
    adj_factor_df=None,  # Optional[pd.DataFrame]
) -> pl.LazyFrame:
    """
    清洗日线行情数据。

    数据流程：
    1. Pandas DataFrame -> Polars DataFrame
    2. 字段重命名（Tushare 格式 -> 系统标准）
    3. 类型转换（所有数值转为 Float32）
    4. 单位转换（手 -> 股，千元 -> 元）
    5. 日期格式标准化
    6. 计算后复权价格 (可选，需要 adj_factor)
    7. 转换为 LazyFrame

    Args:
        df_pandas: 来自 Tushare 的 Pandas DataFrame
        with_adjustment: 是否计算后复权价格
        adj_factor_df: 复权因子 Pandas DataFrame (可选)

    Returns:
        符合系统标准的 Polars LazyFrame

    Raises:
        AssertionError: 如果数据验证失败
    """
    logger.info(f"开始清洗日线行情数据，行数: {len(df_pandas)}")

    # Step 1: Pandas -> Polars，同时进行字段重命名
    df: pl.DataFrame = pl.from_pandas(df_pandas)
    logger.debug(f"转换后 shape: {df.shape}, 列: {df.columns}")

    # Step 2: 字段重命名
    rename_mapping = {v: k for k, v in DAILY_BARS_MAPPING.items()}
    df = df.rename(rename_mapping)

    # Step 3: 日期格式标准化 (_DATE_ 转为 Date 类型)
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_")
    )

    # Step 4: 单位转换 (手 -> 股，千元 -> 元)
    df = df.with_columns(
        (pl.col("VOLUME") * TUSHARE_VOLUME_UNIT).cast(pl.Float32).alias("VOLUME"),
        (pl.col("AMOUNT") * TUSHARE_AMOUNT_UNIT).cast(pl.Float32).alias("AMOUNT"),
    )

    # Step 5: 原始价格列转为 Float32
    raw_price_cols = ["RAW_OPEN", "RAW_HIGH", "RAW_LOW", "RAW_CLOSE"]
    for col in raw_price_cols:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

    # Step 6: 计算后复权价格 (RAW_* * adj_factor)
    if with_adjustment and adj_factor_df is not None:
        logger.info("计算后复权价格...")
        df_adj = clean_adj_factors(adj_factor_df).collect()

        # Join 复权因子
        df = df.join(
            df_adj.select(["_DATE_", "_ASSET_", "adj_factor"]),
            on=["_DATE_", "_ASSET_"],
            how="left",
        )

        # 计算后复权价格
        for raw_col, adj_col in zip(
            ["RAW_OPEN", "RAW_HIGH", "RAW_LOW", "RAW_CLOSE"],
            ["OPEN", "HIGH", "LOW", "CLOSE"],
        ):
            df = df.with_columns(
                (pl.col(raw_col) * pl.col("adj_factor"))
                .cast(pl.Float32)
                .alias(adj_col)
            )

        # 删除临时的 adj_factor 列
        df = df.drop("adj_factor")
    else:
        # 无复权因子时，直接使用原始价格作为后复权价格
        logger.warning("未计算后复权价格，使用原始价格")
        df = df.with_columns(
            pl.col("RAW_OPEN").alias("OPEN"),
            pl.col("RAW_HIGH").alias("HIGH"),
            pl.col("RAW_LOW").alias("LOW"),
            pl.col("RAW_CLOSE").alias("CLOSE"),
        )

    # Step 7: 类型强制转换（确保完全符合 Schema）
    df = df.select(
        [
            pl.col("_DATE_"),
            pl.col("_ASSET_"),
            *[pl.col(c).cast(pl.Float32) for c in raw_price_cols],
            *[pl.col(c).cast(pl.Float32) for c in ["OPEN", "HIGH", "LOW", "CLOSE"]],
            pl.col("VOLUME").cast(pl.Float32),
            pl.col("AMOUNT").cast(pl.Float32),
        ]
    )

    # Step 8: 验证 Schema (collect 验证)
    df_collected = df.collect()
    try:
        validate_schema(df_collected, DAILY_BARS_SCHEMA)
        logger.info(f"✓ Schema 验证通过，最终 shape: {df_collected.shape}")
    except AssertionError as e:
        logger.error(f"✗ Schema 验证失败: {e}")
        raise

    return df.lazy()


# ============================================================================
# 交易日历清洗
# ============================================================================


def clean_calendar(df_pandas) -> pl.LazyFrame:
    """
    清洗交易日历数据。

    Args:
        df_pandas: 来自 Tushare 的 Pandas DataFrame

    Returns:
        符合系统标准的 Polars LazyFrame
    """
    logger.info(f"开始清洗交易日历数据，行数: {len(df_pandas)}")

    df: pl.DataFrame = pl.from_pandas(df_pandas)

    # 字段重命名
    rename_mapping = {v: k for k, v in CALENDAR_MAPPING.items()}
    df = df.rename(rename_mapping)

    # 日期格式标准化
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_")
    )

    # 类型转换
    df = df.with_columns(pl.col("is_open").cast(pl.Boolean))

    # 验证
    df_collected = df.collect()
    validate_schema(df_collected, CALENDAR_SCHEMA)
    logger.info(f"✓ 交易日历验证通过，行数: {len(df_collected)}")

    return df.lazy()


# ============================================================================
# 复权因子清洗
# ============================================================================


def clean_adj_factors(df_pandas) -> pl.LazyFrame:
    """
    清洗复权因子数据。

    Args:
        df_pandas: 来自 Tushare 的 Pandas DataFrame

    Returns:
        符合系统标准的 Polars LazyFrame
    """
    logger.info(f"开始清洗复权因子数据，行数: {len(df_pandas)}")

    df: pl.DataFrame = pl.from_pandas(df_pandas)

    # 字段重命名
    rename_mapping = {v: k for k, v in ADJ_FACTOR_MAPPING.items()}
    df = df.rename(rename_mapping)

    # 日期格式标准化
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_")
    )

    # 类型转换
    df = df.with_columns(pl.col("adj_factor").cast(pl.Float32))

    # 验证
    df_collected = df.collect()
    validate_schema(df_collected, ADJ_FACTOR_SCHEMA)
    logger.info(f"✓ 复权因子验证通过，行数: {len(df_collected)}")

    return df.lazy()


# ============================================================================
# 每日基础指标清洗
# ============================================================================


def clean_daily_basic(df_pandas) -> pl.LazyFrame:
    """
    清洗每日基础指标数据。

    Args:
        df_pandas: 来自 Tushare 的 Pandas DataFrame

    Returns:
        符合系统标准的 Polars LazyFrame
    """
    logger.info(f"开始清洗每日基础指标数据，行数: {len(df_pandas)}")

    df: pl.DataFrame = pl.from_pandas(df_pandas)

    # 字段重命名
    rename_mapping = {v: k for k, v in DAILY_BASIC_MAPPING.items()}
    df = df.rename(rename_mapping)

    # 日期格式标准化
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_")
    )

    # 所有数值列转为 Float32
    float_cols = [
        c for c in df.columns
        if c not in ["_DATE_", "_ASSET_"]
    ]
    for col in float_cols:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

    # 验证
    df_collected = df.collect()
    validate_schema(df_collected, DAILY_BASIC_SCHEMA)
    logger.info(f"✓ 每日基础指标验证通过，行数: {len(df_collected)}")

    return df.lazy()


# ============================================================================
# 市场状态清洗（综合多个来源的数据）
# ============================================================================


def clean_market_status(
    stk_limit_df=None,
    stock_st_df=None,
    suspend_d_df=None,
) -> pl.LazyFrame:
    """
    清洗市场状态数据（涨跌停、ST、停牌）。

    将来自多个 Tushare 接口的数据合并为一个综合表。

    Args:
        stk_limit_df: 涨跌停价 Pandas DataFrame (可选)
        stock_st_df: ST 状态 Pandas DataFrame (可选)
        suspend_d_df: 停牌记录 Pandas DataFrame (可选)

    Returns:
        符合系统标准的 Polars LazyFrame
    """
    logger.info("开始清洗市场状态数据...")


    # 涨跌停价
    if stk_limit_df is not None:
        df_stk: pl.DataFrame = pl.from_pandas(stk_limit_df)
        rename_map = {v: k for k, v in {"trade_date": "_DATE_", "ts_code": "_ASSET_", "up_limit": "up_limit", "down_limit": "down_limit"}.items()}
        df_stk = df_stk.rename(rename_map)
        df_stk = df_stk.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_"),
            pl.col("up_limit").cast(pl.Float32),
            pl.col("down_limit").cast(pl.Float32),
        )
        logger.debug(f"涨跌停数据：{df_stk.shape}")
    else:
        df_stk = None

    # ST 状态 (需要生成衍生的 is_st 列)
    if stock_st_df is not None:
        df_st: pl.DataFrame = pl.from_pandas(stock_st_df)
        df_st = df_st.rename({"trade_date": "_DATE_", "ts_code": "_ASSET_"})
        df_st = df_st.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_"),
            pl.lit(True).alias("is_st"),  # stock_st 返回的行都是 ST 股票
        )
        logger.debug(f"ST 状态数据：{df_st.shape}")
    else:
        df_st = None

    # 停牌状态 (需要生成衍生的 is_suspended 列)
    if suspend_d_df is not None:
        df_suspend: pl.DataFrame = pl.from_pandas(suspend_d_df)
        df_suspend = df_suspend.rename(
            {"suspend_date": "_DATE_", "ts_code": "_ASSET_"}
        )
        df_suspend = df_suspend.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d").alias("_DATE_"),
            pl.lit(True).alias("is_suspended"),  # 标记为停牌
        )
        logger.debug(f"停牌状态数据：{df_suspend.shape}")
    else:
        df_suspend = None

    # 合并所有表 (假设都有相同的 _DATE_, _ASSET_ 列)
    result = None
    for df_temp in [df_stk, df_st, df_suspend]:
        if df_temp is not None:
            if result is None:
                result = df_temp
            else:
                result = result.join(df_temp, on=["_DATE_", "_ASSET_"], how="outer")

    if result is None:
        logger.warning("市场状态数据全为空，返回空 LazyFrame")
        return pl.LazyFrame({"_DATE_": [], "_ASSET_": []}).lazy()

    # 填充缺失值
    result = result.with_columns(
        pl.col("is_st").fill_null(False),
        pl.col("is_suspended").fill_null(False),
    )

    logger.info(f"✓ 市场状态数据合并完成，行数: {result.collect().shape[0]}")
    return result.lazy()


# ============================================================================
# 股票基础信息清洗
# ============================================================================


def clean_stock_basic(df_pandas) -> pl.LazyFrame:
    """
    清洗股票基础信息数据。

    Args:
        df_pandas: 来自 Tushare 的 Pandas DataFrame

    Returns:
        符合系统标准的 Polars LazyFrame
    """
    logger.info(f"开始清洗股票基础信息，行数: {len(df_pandas)}")

    df: pl.DataFrame = pl.from_pandas(df_pandas)

    # 字段重命名
    rename_mapping = {v: k for k, v in STOCK_BASIC_MAPPING.items()}
    df = df.rename(rename_mapping)

    # list_date 转为 Date 类型
    df = df.with_columns(
        pl.col("list_date").str.strptime(pl.Date, "%Y%m%d").alias("list_date")
    )

    # 验证
    df_collected = df.collect()
    validate_schema(df_collected, STOCK_BASIC_SCHEMA)
    logger.info(f"✓ 股票基础信息验证通过，行数: {len(df_collected)}")

    return df.lazy()
