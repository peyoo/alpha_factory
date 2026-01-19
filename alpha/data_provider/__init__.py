"""
数据接入层 (Data Provider)

核心模块：
- schema: 字段映射与数据契约定义
- cleaner: Polars 数据清洗流水线
- tushare_source: Tushare API 交互
"""

from alpha.data_provider.schema import (
    DAILY_BARS_MAPPING,
    DAILY_BARS_SCHEMA,
    CALENDAR_MAPPING,
    CALENDAR_SCHEMA,
    ADJ_FACTOR_MAPPING,
    ADJ_FACTOR_SCHEMA,
    DAILY_BASIC_MAPPING,
    DAILY_BASIC_SCHEMA,
    MARKET_STATUS_SCHEMA,
    STOCK_BASIC_MAPPING,
    STOCK_BASIC_SCHEMA,
    get_schema_by_type,
    get_mapping_by_type,
    validate_schema,
    convert_volume_unit,
    convert_amount_unit,
)

__all__ = [
    # Schema 映射
    "DAILY_BARS_MAPPING",
    "DAILY_BARS_SCHEMA",
    "CALENDAR_MAPPING",
    "CALENDAR_SCHEMA",
    "ADJ_FACTOR_MAPPING",
    "ADJ_FACTOR_SCHEMA",
    "DAILY_BASIC_MAPPING",
    "DAILY_BASIC_SCHEMA",
    "MARKET_STATUS_SCHEMA",
    "STOCK_BASIC_MAPPING",
    "STOCK_BASIC_SCHEMA",
    # 工具函数
    "get_schema_by_type",
    "get_mapping_by_type",
    "validate_schema",
    "convert_volume_unit",
    "convert_amount_unit",
]
