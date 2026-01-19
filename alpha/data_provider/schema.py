"""
数据接入层 Schema 定义与字段映射。

本模块定义 Tushare 原始字段到系统标准字段的映射规则，包括：
- 字段重命名
- 类型转换
- 单位标准化
"""

from typing import Any, Dict
import polars as pl

# ============================================================================
# 日线行情 (Daily Bars) Schema 映射
# ============================================================================

# Tushare 字段 -> 系统标准字段的映射
DAILY_BARS_MAPPING: Dict[str, str] = {
    "trade_date": "_DATE_",
    "ts_code": "_ASSET_",
    "open": "RAW_OPEN",           # 原始不复权开盘价
    "high": "RAW_HIGH",           # 原始不复权最高价
    "low": "RAW_LOW",             # 原始不复权最低价
    "close": "RAW_CLOSE",         # 原始不复权收盘价
    "vol": "VOLUME",              # 交易量（Tushare 默认单位：手 -> 转换为股）
    "amount": "AMOUNT",           # 交易额（Tushare 默认单位：千元 -> 转换为元）
}

# 后复权价格字段（衍生计算）
# 在清洗阶段由 RAW_* 和 adj_factor 计算得出
ADJUSTED_PRICE_FIELDS = ["OPEN", "HIGH", "LOW", "CLOSE"]

# 日线行情的完整 Schema 定义
DAILY_BARS_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,          # 存储态为 String，计算态转为 Categorical
    "RAW_OPEN": pl.Float32,
    "RAW_HIGH": pl.Float32,
    "RAW_LOW": pl.Float32,
    "RAW_CLOSE": pl.Float32,
    "OPEN": pl.Float32,            # 后复权
    "HIGH": pl.Float32,            # 后复权
    "LOW": pl.Float32,             # 后复权
    "CLOSE": pl.Float32,           # 后复权
    "VOLUME": pl.Float32,          # 单位：股
    "AMOUNT": pl.Float32,          # 单位：元
}

# ============================================================================
# 交易日历 (Trade Calendar) Schema 映射
# ============================================================================

CALENDAR_MAPPING: Dict[str, str] = {
    "cal_date": "_DATE_",
    "is_open": "is_open",
}

CALENDAR_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "is_open": pl.Boolean,
}

# ============================================================================
# 复权因子 (Adjustment Factors) Schema 映射
# ============================================================================

ADJ_FACTOR_MAPPING: Dict[str, str] = {
    "trade_date": "_DATE_",
    "ts_code": "_ASSET_",
    "adj_factor": "adj_factor",
}

ADJ_FACTOR_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "adj_factor": pl.Float32,
}

# ============================================================================
# 每日基础指标 (Daily Basic) Schema 映射
# ============================================================================

DAILY_BASIC_MAPPING: Dict[str, str] = {
    "trade_date": "_DATE_",
    "ts_code": "_ASSET_",
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio",
    "pe": "pe",
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps": "ps",
    "ps_ttm": "ps_ttm",
    "dv_ratio": "dv_ratio",
    "dv_ttm": "dv_ttm",
    "total_share": "total_share",
    "float_share": "float_share",
    "free_share": "free_share",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
}

DAILY_BASIC_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "turnover_rate": pl.Float32,
    "turnover_rate_f": pl.Float32,
    "volume_ratio": pl.Float32,
    "pe": pl.Float32,
    "pe_ttm": pl.Float32,
    "pb": pl.Float32,
    "ps": pl.Float32,
    "ps_ttm": pl.Float32,
    "dv_ratio": pl.Float32,
    "dv_ttm": pl.Float32,
    "total_share": pl.Float32,      # 单位：万股
    "float_share": pl.Float32,      # 单位：万股
    "free_share": pl.Float32,       # 单位：万股
    "total_mv": pl.Float32,         # 单位：万元
    "circ_mv": pl.Float32,          # 单位：万元
}

# ============================================================================
# 市场状态因子 (Market Status) Schema 映射
# ============================================================================

# 涨跌停价 (stk_limit)
STK_LIMIT_MAPPING: Dict[str, str] = {
    "trade_date": "_DATE_",
    "ts_code": "_ASSET_",
    "up_limit": "up_limit",
    "down_limit": "down_limit",
}

STK_LIMIT_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "up_limit": pl.Float32,
    "down_limit": pl.Float32,
}

# ST 状态 (stock_st)
STOCK_ST_MAPPING: Dict[str, str] = {
    "trade_date": "_DATE_",
    "ts_code": "_ASSET_",
}

STOCK_ST_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "is_st": pl.Boolean,  # 衍生字段，表示该股在该日期是否为 ST
}

# 停牌状态 (suspend_d)
SUSPEND_D_MAPPING: Dict[str, str] = {
    "suspend_date": "_DATE_",
    "ts_code": "_ASSET_",
    "suspend_type": "suspend_type",
}

SUSPEND_D_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "suspend_type": pl.String,
    "is_suspended": pl.Boolean,  # 衍生字段，表示该股在该日期是否停牌
}

# 市场状态综合表
MARKET_STATUS_SCHEMA: Dict[str, Any] = {
    "_DATE_": pl.Date,
    "_ASSET_": pl.String,
    "up_limit": pl.Float32,
    "down_limit": pl.Float32,
    "is_st": pl.Boolean,
    "is_suspended": pl.Boolean,
}

# ============================================================================
# 股票基础信息 (Stock Basic) Schema 映射
# ============================================================================

STOCK_BASIC_MAPPING: Dict[str, str] = {
    "ts_code": "_ASSET_",
    "symbol": "symbol",
    "name": "name",
    "area": "area",
    "industry": "industry",
    "market": "market",
    "list_date": "list_date",
}

STOCK_BASIC_SCHEMA: Dict[str, Any] = {
    "_ASSET_": pl.String,
    "symbol": pl.String,
    "name": pl.String,
    "area": pl.String,
    "industry": pl.String,
    "market": pl.String,
    "list_date": pl.Date,
}

# ============================================================================
# 单位转换常量
# ============================================================================

# Tushare 原始数据的单位
TUSHARE_VOLUME_UNIT = 100  # 手 -> 股，1 手 = 100 股
TUSHARE_AMOUNT_UNIT = 1000  # 千元 -> 元

# ============================================================================
# 辅助函数
# ============================================================================


def get_schema_by_type(data_type: str) -> Dict[str, pl.DataType]:
    """
    根据数据类型返回对应的 Schema。

    Args:
        data_type: 数据类型标识，如 'daily_bars', 'calendar', 'adj_factor' 等

    Returns:
        对应的 Schema 字典
    """
    schema_map = {
        "daily_bars": DAILY_BARS_SCHEMA,
        "calendar": CALENDAR_SCHEMA,
        "adj_factor": ADJ_FACTOR_SCHEMA,
        "daily_basic": DAILY_BASIC_SCHEMA,
        "market_status": MARKET_STATUS_SCHEMA,
        "stock_basic": STOCK_BASIC_SCHEMA,
    }
    return schema_map.get(data_type, {})


def get_mapping_by_type(data_type: str) -> Dict[str, str]:
    """
    根据数据类型返回对应的字段映射。

    Args:
        data_type: 数据类型标识

    Returns:
        Tushare 字段 -> 系统标准字段的映射字典
    """
    mapping_map = {
        "daily_bars": DAILY_BARS_MAPPING,
        "calendar": CALENDAR_MAPPING,
        "adj_factor": ADJ_FACTOR_MAPPING,
        "daily_basic": DAILY_BASIC_MAPPING,
        "stk_limit": STK_LIMIT_MAPPING,
        "stock_st": STOCK_ST_MAPPING,
        "suspend_d": SUSPEND_D_MAPPING,
        "stock_basic": STOCK_BASIC_MAPPING,
    }
    return mapping_map.get(data_type, {})


# ============================================================================
# 验证函数
# ============================================================================


def validate_schema(
    df: pl.DataFrame,
    expected_schema: Dict[str, Any]
) -> bool:
    """
    验证 DataFrame 的 Schema 是否符合预期。

    Args:
        df: 待验证的 Polars DataFrame
        expected_schema: 预期的 Schema 字典

    Returns:
        True 如果 Schema 匹配，否则 False

    Raises:
        AssertionError: 如果 Schema 不匹配，输出详细信息
    """
    for col_name, expected_type in expected_schema.items():
        assert col_name in df.columns, f"缺少列: {col_name}"
        actual_type = df.schema[col_name]
        assert actual_type == expected_type, (
            f"列 {col_name} 类型不匹配: "
            f"期望 {expected_type}, 实际 {actual_type}"
        )
    return True


# ============================================================================
# 单位转换函数
# ============================================================================


def convert_volume_unit(lf: pl.LazyFrame, column_name: str = "VOLUME") -> pl.LazyFrame:
    """
    将交易量从手转换为股。

    Args:
        lf: 输入 LazyFrame
        column_name: 交易量列名

    Returns:
        转换后的 LazyFrame
    """
    return lf.with_columns(
        (pl.col(column_name) * TUSHARE_VOLUME_UNIT)
        .cast(pl.Float32)
        .alias(column_name)
    )


def convert_amount_unit(lf: pl.LazyFrame, column_name: str = "AMOUNT") -> pl.LazyFrame:
    """
    将交易额从千元转换为元。

    Args:
        lf: 输入 LazyFrame
        column_name: 交易额列名

    Returns:
        转换后的 LazyFrame
    """
    return lf.with_columns(
        (pl.col(column_name) * TUSHARE_AMOUNT_UNIT)
        .cast(pl.Float32)
        .alias(column_name)
    )
