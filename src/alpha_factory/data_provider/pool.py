"""
预定义的一些股票池

每一个股票池必须有一个POOL_MASK列，表示该股票是否在池内,True表示在池内，False表示不在池内
为了减轻后续的计算负担，可以预先过滤掉部分池外的股票，静态过滤。
一般的池子会有两个阶段：
1. 静态过滤：在数据加载阶段，过滤掉明显不符合条件的股票，比如交易所、板块等
2. 动态过滤：在因子计算阶段，根据每日的市值、流动性等动态条件，打上POOL_MASK标记

"""

import polars as pl

from alpha_factory.utils.schema import F


def main_small_pool(
    lf: pl.LazyFrame, small_num: int = 800, production=False
) -> pl.LazyFrame:
    """
    定义一个"小市值股票池"，用于捕捉小盘股效应，不可用于生产环境，仅供研究参考。

    ✨ 改进版本：先过滤坏数据，后排名

    逻辑：
    1. 静态过滤：仅保留主板和创业板的股票
    2. 全局候选池缩容：找出曾经进入过前1200名市值的标的
    3. 动态过滤（新增/前移）：删除停牌、涨跌停、上市不足180天的股票 ← 关键改进点
    4. 排名筛选：对"干净数据"进行市值排名，取前 small_num 名

    关键改进原因：
    - 改进前：先排名全部股票 → 再过滤停牌 → 导致"稀疏日期"下排名分布严重扭曲
    - 改进后：先过滤停牌 → 再排名有效股票 → 确保排名分布始终稳定
    - 这解决了嵌套因子（如 ts_rank(cs_rank_mask(...))）在特定日期失效的问题

    参数：
    lf: 输入的 LazyFrame，必须包含必要的列
    small_num: 每日市值排名前多少名的股票纳入池内，默认800
    production: 是否为生产模式（仅供研究参考，不可用于生产）

    返回：
    包含 POOL_MASK 列的 LazyFrame

    Schema:
        输入：必须包含列 _DATE_, _ASSET_, EXCHANGE, MARKET_TYPE, TOTAL_MV,
              IS_ST, IS_SUSPENDED, LIST_DAYS, IS_UP_LIMIT, IS_DOWN_LIMIT
        输出：同输入 schema，添加 POOL_MASK: bool
    """
    # --- 第一阶段：静态硬过滤 ---
    # 1. 限制交易所：仅保留 SSE (上交所) 和 SZSE (深交所)
    # 2. 限制板块：仅保留主板和创业板（科创板通常为 "科创板"）
    # 3. 补充：通过代码前缀排除 688 (科创板) 和 8/4 (北交所)

    lf = lf.with_columns(
        [
            pl.col("EXCHANGE").cast(pl.String),
            pl.col("MARKET_TYPE").cast(pl.String),
            pl.col("ASSET").cast(pl.String),
        ]
    )
    lf = lf.filter(
        (pl.col("EXCHANGE").is_in(["SSE", "SZSE"]))
        & (pl.col("MARKET_TYPE").is_in(["主板", "创业板"]))
        & ~pl.col("ASSET").str.starts_with("688")
        & ~pl.col("ASSET").str.starts_with("8")
        & ~pl.col("ASSET").str.starts_with("4")
    )

    if production is False:
        # --- 第二阶段：全局候选池缩容 (Semi-Join 优化) ---
        # 找到在回测期间"至少有一次"进入过市值前1200名的标的
        # 这一步是为了过滤掉那些永远不可能入选的"巨头"或"僵尸股"，从而加速后续计算
        candidate_assets = (
            lf.with_columns(_tmp_rank=pl.col("TOTAL_MV").rank("ordinal").over(F.DATE))
            .filter(pl.col("_tmp_rank") <= 1200)
            .select("ASSET")
            .unique()
        )
        lf = lf.join(candidate_assets, on="ASSET", how="semi")

    # ✨ --- 第三阶段改进：先动态过滤"坏数据"，再排名（关键改动） ---
    # 将停牌、涨跌停等过滤提前到排名之前
    # 这确保排名是基于"有效可交易"的股票，避免了"稀疏日期"问题
    # 导致嵌套因子（如 ts_rank(cs_rank_mask(...))）在特定日期失效
    lf = lf.filter(
        ~pl.col("IS_ST")
        & ~pl.col("IS_SUSPENDED")
        & (pl.col("LIST_DAYS") >= 180)
        & ~pl.col("IS_UP_LIMIT")
        & ~pl.col("IS_DOWN_LIMIT")
    )

    # 现在对"干净数据"进行排名
    return (
        lf.with_columns(
            [
                # 计算每一天的实时排名（仅基于有效交易的股票）
                pl.col("TOTAL_MV").rank("ordinal").over(F.DATE).alias("mv_rank")
            ]
        )
        .with_columns(
            [
                # 预定义可交易池：简化的核心逻辑（因为坏数据已提前过滤）
                (pl.col("mv_rank") <= small_num).alias(F.POOL_MASK)
            ]
        )
        .drop(["mv_rank"])
    )
