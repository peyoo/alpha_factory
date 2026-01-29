import pandas as pd
import polars as pl
from typing import List, Union, Optional
from pathlib import Path

from loguru import logger


def extract_expressions_from_csv(
        file_path: Union[str, Path],
        formula_col: str = "expression",
        name_col: Optional[str] = 'factor_name',
) -> List[str]:
    """
    从 CSV 中提取符合 expr_codegen 格式的表达式列表。

    CSV 预期格式:
    | name     | formula                  | is_active |
    |----------|--------------------------|-----------|
    | alpha_01 | close / delay(close, 1)  | 1         |
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到表达式配置文件: {path}")

    # 1. 加载数据
    df = pd.read_csv(path)

    # 3. 构造表达式字符串
    expressions = []
    for _, row in df.iterrows():
        formula = str(row[formula_col]).strip()

        # 如果提供了 name 列，构造 "name=formula" 格式
        if name_col and name_col in df.columns:
            name = str(row[name_col]).strip()
            expressions.append(f"{name}={formula}")
        else:
            # 如果没有 name 列，假设 CSV 直接就是公式行
            expressions.append(formula)

    logger.info(f"🚀 从 CSV 成功提取 {len(expressions)} 条表达式")
    return expressions


def small_static_universe(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    小微盘静态股票池
    自定义选股器：动态市值最小前 1000 名股票池
    :param lf:
    :return:
    """
    # 找出曾进入前 1000 名的股票（Semi-Join 模式，不取数）
    pl.col('TOTAL_MV').rank().over("DATE").alias("mv_rank"),
    target_assets = (
        lf.with_columns(mv_rank=pl.col("TOTAL_MV").rank("ordinal").over("DATE"))
        .filter(pl.col("mv_rank") <= 1000)
        .select("ASSET")
        .unique()
    )
    return lf.join(target_assets, on="ASSET", how="semi")

def tradable_pool(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    小微盘掩码
    :param lf:
    :return:
    """
    return lf.with_columns([
        # 截面市值排名
        pl.col("TOTAL_MV").rank().over("DATE").alias("mv_rank"),
        pl.col("ASSET").count().over("DATE").alias("total_count"),

        # 精准判断封板逻辑：收盘价 == 涨/跌停价 且 成交额不为0（排除全天停牌）
        (pl.col("CLOSE_RAW") >= pl.col("UP_LIMIT")).alias("is_locked_up"),
        (pl.col("CLOSE_RAW") <= pl.col("DOWN_LIMIT")).alias("is_locked_down")
    ]).with_columns([
        (
                (pl.col("mv_rank") / pl.col("total_count") <= 0.2) &
                (pl.col("IS_ST") is False) &
                (pl.col("IS_SUSPENDED") is False) &
                (pl.col("LIST_DAYS") >= 242) &
                (pl.col("AMOUNT") >= 1e7) &
                # 过滤掉无法买入的封死涨停股 和 无法卖出的封死跌停股
                (pl.col("is_locked_up") is False) &
                (pl.col("is_locked_down") is False)
        ).alias("POOL_TRADABLE")
    ])
