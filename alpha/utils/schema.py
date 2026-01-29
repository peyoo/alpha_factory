from typing import Dict
import polars as pl

from enum import StrEnum


class F(StrEnum):
    """
    Field Constants: 强类型字段库
    好处：1. IDE 自动补全 2. 修改一处，全域生效 3. 静态检查报错
    """
    # 基础标识
    DATE = "DATE"
    ASSET = "ASSET"

    # 行情
    OPEN = "OPEN"
    HIGH = "HIGH"
    LOW = "LOW"
    CLOSE = "CLOSE"
    VOLUME = "VOLUME"
    TOTAL_MV = "TOTAL_MV"

    # 交易限制 (容易拼错的)
    UP_LIMIT = "UP_LIMIT"
    DOWN_LIMIT = "DOWN_LIMIT"
    IS_ST = "IS_ST"
    IS_SUSPENDED = "IS_SUSPENDED"

    # 内部衍生列
    POOL_TRADABLE = "POOL_TRADABLE"
    LIST_DAYS = "LIST_DAYS"
    IS_UP_LIMIT = "is_up_limit"
    IS_DOWN_LIMIT = "is_down_limit"

class DataSchema:
    # 1. 唯一标识列 (Key Columns)
    IDS: Dict[str, pl.DataType] = {
        "DATE": pl.Date,
        "ASSET": pl.Utf8,
    }

    # 2. 静态/低频属性 (Metadata)
    ATTRS: Dict[str, pl.DataType] = {
        "name": pl.Utf8,
        "industry": pl.Categorical,
        "list_date": pl.Date,
        "delist_date": pl.Date,
        "exchange": pl.Categorical,
    }

    # 3. 基础行情 (Market Data) - 强制 Float32
    QUOTES: Dict[str, pl.DataType] = {
        "OPEN": pl.Float32,
        "HIGH": pl.Float32,
        "LOW": pl.Float32,
        "CLOSE": pl.Float32,
        "VOLUME": pl.Float64,  # 成交量较大，保留 Float64
        "AMOUNT": pl.Float64,
        "TOTAL_MV": pl.Float32,
    }

    # 4. 交易环境 (Trade Environment)
    ENV: Dict[str, pl.DataType] = {
        "POOL_TRADABLE": pl.Boolean,
        "IS_ST": pl.Boolean,
        "is_up_limit": pl.Boolean,
        "is_down_limit": pl.Boolean,
    }

    @classmethod
    def all_base_columns(cls) -> Dict[str, pl.DataType]:
        """合并所有非因子的基础列"""
        return {**cls.IDS, **cls.ATTRS, **cls.QUOTES, **cls.ENV}
