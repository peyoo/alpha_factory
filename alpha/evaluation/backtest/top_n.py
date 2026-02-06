import polars as pl
import numpy as np
from typing import List, Callable
from loguru import logger

from alpha.data_provider import DataProvider
from alpha.utils.schema import F


def not_buyable_sellable(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 移除 sort，利用之前的排序结果
    return lf.with_columns([
        (pl.col('IS_UP_LIMIT') | pl.col('IS_SUSPENDED')).alias(F.NOT_BUYABLE),
        (pl.col("IS_DOWN_LIMIT") | pl.col('IS_SUSPENDED')).alias(F.NOT_SELLABLE)
    ])


def backtest_top_n(
        start: str,
        end: str,
        factor_col: str,
        ascending: bool = False,
        funcs: List[Callable[[pl.LazyFrame],pl.LazyFrame]] = None,
        n_buy: int = 10,  # 持仓总数，当前也是最大买入rank排名
        n_sell: int = 30,
        cost_rate: float = 0.002,
        price_col: str = F.CLOSE,
        date_col: str = F.DATE,
        asset_col: str = F.ASSET,
        pool_mask_col: str = F.POOL_MASK,
        not_buyable_col: str = F.NOT_BUYABLE,
        not_sellable_col: str = F.NOT_SELLABLE,
        annual_days: int = 252
) -> pl.DataFrame:
    # 1. 数据准备
    factor_col_is_expr = ("=" in factor_col)
    lf = DataProvider().load_data(start, end, funcs=funcs,
                                  column_exprs=([factor_col] if factor_col_is_expr else None))

    target_factor = factor_col.split('=')[0] if factor_col_is_expr else factor_col

    # 预计算：收益率对齐与截面排名
    processed_lf = (
        lf.sort([asset_col, date_col])
        .with_columns([
            (pl.col(price_col).shift(-1).over(asset_col) / pl.col(price_col) - 1).alias("next_ret")
        ])
        .with_columns([
            pl.col(target_factor).rank(descending=not ascending, method="random").over(date_col).alias("rank")
        ])
        .select([
            date_col, asset_col, pool_mask_col, "rank", not_buyable_col, not_sellable_col, "next_ret"
        ])
        .filter(pl.col("next_ret").is_not_null())
    )

    data = processed_lf.collect()
    all_dates = data.get_column(date_col).unique().sort().to_list()
    grouped = data.partition_by(date_col, as_dict=True)

    # --- 2. 优化后的核心迭代逻辑 ---
    current_holdings = set()
    records = []

    logger.info(f"🚀 开始回测: {target_factor}, 周期: {len(all_dates)} 天")

    for dt in all_dates:
        key = (dt,)
        if key not in grouped:
            prev_h = records[-1]["holdings"] if records else []
            records.append({"DATE": dt, "raw_ret": 0.0, "turnover": 0.0, "count": len(prev_h), "holdings": prev_h})
            continue

        day_df = grouped[key]

        # 【优化点：Fast Dictionary 构建】
        # 1. 预先通过 POOL 过滤候选集
        active_day_df = day_df.filter(pl.col(pool_mask_col) | pl.col(asset_col).is_in(current_holdings))

        # 2. 批量转换为 Numpy 数组进行 Zip 组合，避开 to_dicts()
        assets = active_day_df.get_column(asset_col).to_numpy()
        ranks = active_day_df.get_column("rank").to_numpy()
        nb = active_day_df.get_column(not_buyable_col).to_numpy()
        ns = active_day_df.get_column(not_sellable_col).to_numpy()
        rets = active_day_df.get_column("next_ret").to_numpy()
        # 构造字典：{Asset: (rank, not_buyable, not_sellable, next_ret)}
        day_info = dict(zip(assets, zip(ranks, nb, ns, rets)))

        # 下面是原始的 to_dicts() 版本，供参考
        # day_info = {
        #     row[asset_col]: row
        #     for row in active_day_df.select([
        #         asset_col, "rank", not_buyable_col, not_sellable_col, "next_ret"
        #     ]).to_dicts()
        # }

        # A. 卖出逻辑
        to_keep = set()
        for asset in current_holdings:
            info = day_info.get(asset)
            if info:
                # info[0] 是 rank, info[2] 是 not_sellable
                if info[0] <= n_sell or info[2]:
                    to_keep.add(asset)

        # B. 买入逻辑
        current_holdings = to_keep
        num_to_buy = n_buy - len(current_holdings)

        if num_to_buy > 0:
            # 筛选符合买入条件的候选股
            candidates = (
                active_day_df.filter(
                    (pl.col("rank") <= n_buy) &
                    (~pl.col(asset_col).is_in(current_holdings)) &
                    (~pl.col(not_buyable_col))
                )
                .sort("rank")
                .head(num_to_buy)
                .get_column(asset_col)
                .to_list()
            )
            current_holdings.update(candidates)

        # C. 计算当日表现
        if current_holdings:
            day_rets = [day_info[a][3] for a in current_holdings if a in day_info]
            raw_ret = np.sum(day_rets) / n_buy
        else:
            raw_ret = 0.0

        # D. 换手率
        prev_h = set(records[-1]["holdings"]) if records else set()
        turnover = len(current_holdings - prev_h) / n_buy if n_buy > 0 else 0

        records.append({
            "DATE": dt, "raw_ret": raw_ret, "turnover": turnover,
            "count": len(current_holdings), "holdings": list(current_holdings)
        })

    # --- 3. 结果汇总与指标计算 ---
    res_df = pl.DataFrame(records).with_columns([
        (pl.col("raw_ret") - pl.col("turnover") * cost_rate * 2).alias("NET_RET"),
        pl.col("DATE").cast(pl.String).str.slice(0, 4).alias("Year")
    ]).with_columns((pl.col("net_ret") + 1).cum_prod().alias("nav"))

    return res_df
