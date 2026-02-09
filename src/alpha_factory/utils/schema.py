from typing import Dict
import polars as pl

from enum import Enum


class F(str, Enum):
    """
    Field Constants: 强类型字段库
    这些绝大部分都是data_provider输出数据框架中的标准列名。
    使用枚举类的方式定义字段名称，避免字符串拼写错误。
    好处：1. IDE 自动补全 2. 修改一处，全域生效 3. 静态检查报错
    """
    # 基础标识
    DATE = "DATE"
    ASSET = "ASSET"

    # 行情
    OPEN = "OPEN" #后复权开盘价
    HIGH = "HIGH" #后复权最高价
    LOW = "LOW" #后复权最低价
    CLOSE = "CLOSE" #后复权收盘价
    VOLUME = "VOLUME" #成交量
    VWAP = "VWAP"  # 后复权成交均价
    AMOUNT = "AMOUNT" #成交额
    TOTAL_MV = "TOTAL_MV" #总市值
    CIRC_MV = "CIRC_MV" #流通市值
    ADJ_FACTOR = "ADJ_FACTOR" #后复权因子

    CLOSE_RAW = "CLOSE_RAW" #未复权收盘价
    OPEN_RAW = "OPEN_RAW" #未复权开盘价
    HIGH_RAW = "HIGH_RAW" #未复权最高价
    LOW_RAW = "LOW_RAW" #未复权最低价
    VWAP_RAW = "VWAP_RAW"   #未复权成交均价

    PE = "PE" #市盈率
    PB = "PB" #市净率
    PS = "PS" #市销率
    TURNOVER_RATE = "TURNOVER_RATE" #换手率(基于总市值)



    # 交易限制 (容易拼错的)
    UP_LIMIT = "UP_LIMIT" #涨停价
    DOWN_LIMIT = "DOWN_LIMIT" #跌停价
    IS_ST = "IS_ST" #是否ST股
    IS_SUSPENDED = "IS_SUSPENDED" #是否停牌

    POOL_MASK = "POOL_MASK"    # 股票池掩码


    # 内部衍生列
    LIST_DAYS = "LIST_DAYS"  #上市天数（不是交易天数）
    IS_UP_LIMIT = "IS_UP_LIMIT" #涨停
    IS_DOWN_LIMIT = "IS_DOWN_LIMIT" #跌停

    # INDUSTRY = "INDUSTRY" #行业

    EXCHANGE = "EXCHANGE" #交易所
    MARKET_TYPE = "MARKET_TYPE" #市场类型

    NOT_BUYABLE = "not_buyable"
    NOT_SELLABLE = "not_sellable"

    LABEL_FOR_IC = "LABEL_FOR_IC"
    LABEL_FOR_RET = "LABEL_FOR_RET"

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
