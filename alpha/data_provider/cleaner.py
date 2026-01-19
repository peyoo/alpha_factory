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
    """清洗日线行情数据。"""
    logger.info(f"开始清洗日线行情数据，行数: {len(df_pandas)}")

    # 处理空 DataFrame
    if len(df_pandas) == 0:
        logger.warning("输入数据为空，返回空 LazyFrame")
        empty_df = pl.DataFrame(schema=DAILY_BARS_SCHEMA)
        return empty_df.lazy()

    # Pandas -> Polars
    df: pl.DataFrame = pl.from_pandas(df_pandas)
    logger.debug(f"转换后 shape: {df.shape}, 列: {df.columns}")

    # 字段重命名
    df = df.rename(DAILY_BARS_MAPPING)

    # 日期格式标准化
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_")
    )

    # 单位转换
    df = df.with_columns(
        (pl.col("VOLUME") * TUSHARE_VOLUME_UNIT).cast(pl.Float32).alias("VOLUME"),
        (pl.col("AMOUNT") * TUSHARE_AMOUNT_UNIT).cast(pl.Float32).alias("AMOUNT"),
    )

    # 原始价格列转为 Float32
    raw_price_cols = ["RAW_OPEN", "RAW_HIGH", "RAW_LOW", "RAW_CLOSE"]
    for col in raw_price_cols:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

    # 计算后复权价格
    if with_adjustment and adj_factor_df is not None:
        logger.info("计算后复权价格...")
        df_adj = clean_adj_factors(adj_factor_df).collect()
        df = df.join(
            df_adj.select(["_DATE_", "_ASSET_", "adj_factor"]),
            on=["_DATE_", "_ASSET_"],
            how="left",
        )
        for raw_col, adj_col in zip(
            ["RAW_OPEN", "RAW_HIGH", "RAW_LOW", "RAW_CLOSE"],
            ["OPEN", "HIGH", "LOW", "CLOSE"],
        ):
            df = df.with_columns(
                (pl.col(raw_col) * pl.col("adj_factor"))
                .cast(pl.Float32)
                .alias(adj_col)
            )
        df = df.drop("adj_factor")
    else:
        logger.warning("未计算后复权价格，使用原始价格")
        df = df.with_columns(
            pl.col("RAW_OPEN").alias("OPEN"),
            pl.col("RAW_HIGH").alias("HIGH"),
            pl.col("RAW_LOW").alias("LOW"),
            pl.col("RAW_CLOSE").alias("CLOSE"),
        )

    # 类型强制转换
    df = df.select([
        pl.col("_DATE_"),
        pl.col("_ASSET_"),
        *[pl.col(c).cast(pl.Float32) for c in raw_price_cols],
        *[pl.col(c).cast(pl.Float32) for c in ["OPEN", "HIGH", "LOW", "CLOSE"]],
        pl.col("VOLUME").cast(pl.Float32),
        pl.col("AMOUNT").cast(pl.Float32),
    ])

    # 验证 Schema
    df_collected = df.collect() if hasattr(df, 'collect') else df
    try:
        validate_schema(df_collected, DAILY_BARS_SCHEMA)
        logger.info(f"✓ Schema 验证通过，最终 shape: {df_collected.shape}")
    except AssertionError as e:
        logger.error(f"✗ Schema 验证失败: {e}")
        raise

    return df.lazy() if hasattr(df, 'lazy') else df.lazy()


# ============================================================================
# 交易日历清洗
# ============================================================================


def clean_calendar(df_pandas) -> pl.LazyFrame:
    """清洗交易日历数据。"""
    logger.info(f"开始清洗交易日历数据，行数: {len(df_pandas)}")

    if len(df_pandas) == 0:
        logger.warning("输入数据为空，返回空 LazyFrame")
        empty_df = pl.DataFrame(schema=CALENDAR_SCHEMA)
        return empty_df.lazy()

    df: pl.DataFrame = pl.from_pandas(df_pandas)
    df = df.rename(CALENDAR_MAPPING)
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_")
    )
    df = df.with_columns(pl.col("is_open").cast(pl.Boolean))

    df_collected = df.collect() if hasattr(df, 'collect') else df
    validate_schema(df_collected, CALENDAR_SCHEMA)
    logger.info(f"✓ 交易日历验证通过，行数: {len(df_collected)}")

    return df.lazy() if hasattr(df, 'lazy') else df.lazy()


# ============================================================================
# 复权因子清洗
# ============================================================================


def clean_adj_factors(df_pandas) -> pl.LazyFrame:
    """清洗复权因子数据。"""
    logger.info(f"开始清洗复权因子数据，行数: {len(df_pandas)}")

    if len(df_pandas) == 0:
        logger.warning("输入数据为空，返回空 LazyFrame")
        empty_df = pl.DataFrame(schema=ADJ_FACTOR_SCHEMA)
        return empty_df.lazy()

    df: pl.DataFrame = pl.from_pandas(df_pandas)
    df = df.rename(ADJ_FACTOR_MAPPING)
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_")
    )
    df = df.with_columns(pl.col("adj_factor").cast(pl.Float32))

    df_collected = df.collect() if hasattr(df, 'collect') else df
    validate_schema(df_collected, ADJ_FACTOR_SCHEMA)
    logger.info(f"✓ 复权因子验证通过，行数: {len(df_collected)}")

    return df.lazy() if hasattr(df, 'lazy') else df.lazy()


# ============================================================================
# 每日基础指标清洗
# ============================================================================


def clean_daily_basic(df_pandas) -> pl.LazyFrame:
    """清洗每日基础指标数据。"""
    logger.info(f"开始清洗每日基础指标数据，行数: {len(df_pandas)}")

    if len(df_pandas) == 0:
        logger.warning("输入数据为空，返回空 LazyFrame")
        empty_df = pl.DataFrame(schema=DAILY_BASIC_SCHEMA)
        return empty_df.lazy()

    df: pl.DataFrame = pl.from_pandas(df_pandas)
    df = df.rename(DAILY_BASIC_MAPPING)
    df = df.with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_")
    )

    float_cols = [c for c in df.columns if c not in ["_DATE_", "_ASSET_"]]
    for col in float_cols:
        df = df.with_columns(pl.col(col).cast(pl.Float32))

    df_collected = df.collect() if hasattr(df, 'collect') else df
    validate_schema(df_collected, DAILY_BASIC_SCHEMA)
    logger.info(f"✓ 每日基础指标验证通过，行数: {len(df_collected)}")

    return df.lazy() if hasattr(df, 'lazy') else df.lazy()


# ============================================================================
# 市场状态清洗
# ============================================================================


def clean_market_status(
    stk_limit_df=None,
    stock_st_df=None,
    suspend_d_df=None,
) -> pl.LazyFrame:
    """清洗市场状态数据（涨跌停、ST、停牌）。"""
    logger.info("开始清洗市场状态数据...")

    dfs = []

    # 涨跌停价
    if stk_limit_df is not None and len(stk_limit_df) > 0:
        df_stk: pl.DataFrame = pl.from_pandas(stk_limit_df)
        df_stk = df_stk.rename({"trade_date": "_DATE_", "ts_code": "_ASSET_"})
        df_stk = df_stk.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_"),
            pl.col("up_limit").cast(pl.Float32),
            pl.col("down_limit").cast(pl.Float32),
        )
        dfs.append(df_stk)
        logger.debug(f"涨跌停数据：{df_stk.shape}")

    # ST 状态
    if stock_st_df is not None and len(stock_st_df) > 0:
        df_st: pl.DataFrame = pl.from_pandas(stock_st_df)
        df_st = df_st.rename({"trade_date": "_DATE_", "ts_code": "_ASSET_"})
        df_st = df_st.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_"),
            pl.lit(True).alias("is_st"),
        )
        dfs.append(df_st)
        logger.debug(f"ST 状态数据：{df_st.shape}")

    # 停牌状态
    if suspend_d_df is not None and len(suspend_d_df) > 0:
        df_suspend: pl.DataFrame = pl.from_pandas(suspend_d_df)
        df_suspend = df_suspend.rename({"suspend_date": "_DATE_", "ts_code": "_ASSET_"})
        df_suspend = df_suspend.with_columns(
            pl.col("_DATE_").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_DATE_"),
            pl.lit(True).alias("is_suspended"),
        )
        dfs.append(df_suspend)
        logger.debug(f"停牌状态数据：{df_suspend.shape}")

    if not dfs:
        logger.warning("市场状态数据全为空，返回空 LazyFrame")
        return pl.LazyFrame({"_DATE_": [], "_ASSET_": []}).lazy()

    result = dfs[0]
    for df_temp in dfs[1:]:
        result = result.join(df_temp, on=["_DATE_", "_ASSET_"], how="outer")

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
    """清洗股票基础信息数据。"""
    logger.info(f"开始清洗股票基础信息，行数: {len(df_pandas)}")

    if len(df_pandas) == 0:
        logger.warning("输入数据为空，返回空 LazyFrame")
        empty_df = pl.DataFrame(schema=STOCK_BASIC_SCHEMA)
        return empty_df.lazy()

    df: pl.DataFrame = pl.from_pandas(df_pandas)
    df = df.rename(STOCK_BASIC_MAPPING)
    df = df.with_columns(
        pl.col("list_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("list_date")
    )

    df_collected = df.collect() if hasattr(df, 'collect') else df
    validate_schema(df_collected, STOCK_BASIC_SCHEMA)
    logger.info(f"✓ 股票基础信息验证通过，行数: {len(df_collected)}")

    return df.lazy() if hasattr(df, 'lazy') else df.lazy()
