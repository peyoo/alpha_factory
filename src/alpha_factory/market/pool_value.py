import polars as pl


def calculate_correct_market_pc(all_factors: pl.DataFrame, pool_num: int = 40):
    # 1. 确定每日的持仓名单（基于当日 rank）
    # 我们将这个名单“推后一天”，表示这是下一交易日的持仓
    holds = (
        all_factors.filter(pl.col("rank") <= pool_num)
        .select(["date", "asset"])
        .with_columns(
            # 将日期向后平移一个交易单位，表示这些票在下一个 date 产生收益
            pl.col("date").shift(1).over("asset")
        )
        .drop_nulls()  # 去掉第一天，因为第一天没有“昨日持仓”
    )

    # 2. 将“昨日名单”与“今日涨幅”对齐
    # 这样 join 出来的 pc 就是 T-1 日选出的票在 T 日的真实涨幅
    market_pc_df = (
        holds.join(
            all_factors.select(["date", "asset", "pc"]),
            on=["date", "asset"],
            how="inner",
        )
        .group_by("date")
        .agg(pl.col("pc").mean().alias("market_pc"))
        .sort("date")
    )

    # 3. 计算累计净值
    final_df = market_pc_df.with_columns(
        ((pl.col("market_pc") + 1) * 0.9996).cum_prod().alias("market_pool_value")
    )

    return final_df
