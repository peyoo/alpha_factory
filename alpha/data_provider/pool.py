"""
预定义的一些股票池

每一个股票池必须有一个POOL_MASK列，表示该股票是否在池内,True表示在池内，False表示不在池内
为了减轻后续的计算负担，可以预先过滤掉部分池外的股票，静态过滤。
一般的池子会有两个阶段：
1. 静态过滤：在数据加载阶段，过滤掉明显不符合条件的股票，比如交易所、板块等
2. 动态过滤：在因子计算阶段，根据每日的市值、流动性等动态条件，打上POOL_MASK标记

"""

import polars as pl

from alpha.utils.schema import F

def main_small_pool(lf: pl.LazyFrame, small_num: int = 800,production = False) -> pl.LazyFrame:
    """
    定义一个“小市值股票池”，用于捕捉小盘股效应,不可用于生产环境，仅供研究参考。
    逻辑：
    1. 静态过滤：仅保留主板和创业板的股票
    2. 动态过滤：每日市值排名在前 small_num 名的股票

    参数：
    lf: 输入的 LazyFrame，必须包含必要的列
    small_num: 每日市值排名前多少名的股票纳入池内，默认800
    返回：
    包含POOL_MASK列的LazyFrame
    """
    # --- 第一阶段：静态硬过滤 ---
    # 1. 限制交易所
    # 2. 限制板块：仅保留主板和创业板 (科创板通常为 "科创板")
    # 3. 补充：通过代码前缀排除 688 (科创板) 和 8/4 (北交所)

    lf = lf.with_columns([
        pl.col("EXCHANGE").cast(pl.String),
        pl.col("MARKET_TYPE").cast(pl.String),
        pl.col("ASSET").cast(pl.String)
    ])
    lf = lf.filter(
        (pl.col("EXCHANGE").is_in(["SSE", "SZSE"])) &
        (pl.col("MARKET_TYPE").is_in(["主板", "创业板"])) &
        ~pl.col("ASSET").str.starts_with("688") &
        ~pl.col("ASSET").str.starts_with("8") &
        ~pl.col("ASSET").str.starts_with("4")
    )
    if production is False:
        # --- 第二阶段：全局候选池缩容 (Semi-Join 优化) ---
        # 找到在回测期间“至少有一次”进入过市值前1000名的标的
        # 这一步是为了过滤掉那些永远不可能入选的“巨头”或“僵尸股”，从而加速后续计算
        candidate_assets = (
            lf.with_columns(
                _tmp_rank=pl.col("TOTAL_MV").rank("ordinal").over(F.DATE)
            )
            .filter(pl.col("_tmp_rank") <= 1200)
            .select("ASSET")
            .unique()
        )
        lf = lf.join(candidate_assets, on="ASSET", how="semi")

    # --- 第三阶段：动态掩码 (含动态排名) ---
    return lf.with_columns([
        # 计算每一天的实时排名
        pl.col("TOTAL_MV").rank("ordinal").over(F.DATE).alias("mv_rank")
    ]).with_columns([
        # 预定义可交易池：核心逻辑
        (
            ~pl.col("IS_ST") &
            ~pl.col("IS_SUSPENDED") &
            (pl.col("LIST_DAYS") >= 180) &
            ~pl.col("IS_UP_LIMIT") &
            ~pl.col("IS_DOWN_LIMIT") &
            (pl.col("mv_rank") <= small_num) # 动态选择当日的小市值标的
        ).alias(F.POOL_MASK)
    ]).drop(["mv_rank"])
