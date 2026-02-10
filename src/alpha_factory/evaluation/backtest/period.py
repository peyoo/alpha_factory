from typing import Union, Dict

import polars as pl
from loguru import logger

from alpha_factory.utils.schema import F


def backtest_periodic_rebalance(
    df_input: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    hold_num: int = 10,
    rebalance_period: int = 20,
    cost_rate: float = 0.0025,
    exec_price: str = F.OPEN,
    ascending: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    按时间周期换股的回测框架

    参数：
        df_input: 输入数据
        factor_col: 因子列名
        hold_num: 持股数量（每个周期持有排名 <= hold_num 的股票）
        rebalance_period: 换股周期（天数），如 5 表示每 5 天换股一次
        cost_rate: 交易费率
        exec_price: 执行价格列
        ascending: 因子排序方向（False=降序，True=升序）

    返回：
        dict，包含日度结果和交易明细
    """

    # --- 1. 数据预处理 ---
    lf = df_input.lazy() if isinstance(df_input, pl.LazyFrame) else df_input.lazy()

    # 预计算排名
    lf = lf.with_columns(
        [
            pl.when(pl.col(F.POOL_MASK))
            .then(pl.col(factor_col))
            .otherwise(None)
            .rank(descending=not ascending, method="random")
            .over(F.DATE)
            .fill_null(999999)
            .alias("RANK")
        ]
    )

    final_cols = [
        F.DATE,
        F.ASSET,
        exec_price,
        F.CLOSE,
        "RANK",
        F.IS_UP_LIMIT,
        F.IS_DOWN_LIMIT,
        F.IS_SUSPENDED,
    ]
    df = lf.select(final_cols).collect()

    all_dates = df.get_column(F.DATE).unique().sort().to_list()
    grouped = df.partition_by(F.DATE, as_dict=True)

    # --- 2. 状态维护 ---
    current_holdings = {}
    target_holdings = set()
    daily_records = []
    trades_records = []
    last_rebalance_idx = -rebalance_period  # 强制第一次交易

    logger.info(
        f"🚀 周期换股回测 | 因子: {factor_col} | 持股数: {hold_num} | 周期: {rebalance_period} 天 | 费率: {cost_rate:.4f}"
    )

    # --- 3. 逐日演进 ---
    for i, curr_dt in enumerate(all_dates):
        day_df = grouped.get((curr_dt,))
        if day_df is None:
            daily_records.append(
                {
                    F.DATE: curr_dt,
                    "RAW_RET": 0.0,
                    "TURNOVER": 0.0,
                    "COUNT": len(current_holdings),
                }
            )
            continue

        day_info = {row[F.ASSET]: row for row in day_df.to_dicts()}

        new_holdings = {}
        num_bought = 0
        num_sold = 0
        day_raw_ret = 0.0

        # A. 判断是否需要换股
        need_rebalance = (i - last_rebalance_idx) >= rebalance_period

        # B. 更新目标持仓
        if need_rebalance:
            target_holdings = {
                a for a, info in day_info.items() if info["RANK"] <= hold_num
            }
            last_rebalance_idx = i

        # C. 处理现有持仓
        for asset, hold_info in current_holdings.items():
            info = day_info.get(asset)
            if not info:
                new_holdings[asset] = hold_info
                continue

            price_today_exec = info[exec_price]
            price_yesterday_close = hold_info["last_price"]
            day_raw_ret += ((price_today_exec / price_yesterday_close) - 1) / hold_num

            # 判定卖出
            if asset not in target_holdings:
                if info[F.IS_DOWN_LIMIT] or info[F.IS_SUSPENDED]:
                    hold_info["last_price"] = info[F.CLOSE]
                    day_raw_ret += ((info[F.CLOSE] / price_today_exec) - 1) / hold_num
                    new_holdings[asset] = hold_info
                else:
                    trades_records.append(
                        {
                            F.ASSET: asset,
                            "entry_date": hold_info["entry_date"],
                            "exit_date": curr_dt,
                            "entry_price": hold_info["entry_price"],
                            "exit_price": price_today_exec,
                            "pnl_ret": price_today_exec / hold_info["entry_price"] - 1,
                            "holding_periods": i - hold_info["entry_idx"],
                        }
                    )
                    num_sold += 1
            else:
                day_raw_ret += ((info[F.CLOSE] / price_today_exec) - 1) / hold_num
                hold_info["last_price"] = info[F.CLOSE]
                new_holdings[asset] = hold_info

        # D. 处理新股票买入
        potential_buys = [a for a in target_holdings if a not in new_holdings]
        potential_buys.sort(
            key=lambda x: day_info[x]["RANK"] if x in day_info else 999999
        )

        for asset in potential_buys:
            if len(new_holdings) < hold_num:
                info = day_info.get(asset)
                if info and not (info[F.IS_UP_LIMIT] or info[F.IS_SUSPENDED]):
                    price_buy_exec = info[exec_price]
                    day_raw_ret += ((info[F.CLOSE] / price_buy_exec) - 1) / hold_num

                    new_holdings[asset] = {
                        "entry_date": curr_dt,
                        "entry_price": price_buy_exec,
                        "last_price": info[F.CLOSE],
                        "entry_idx": i,
                    }
                    num_bought += 1

        # E. 记录流水
        turnover = (num_bought + num_sold) / hold_num if hold_num > 0 else 0.0
        current_holdings = new_holdings
        daily_records.append(
            {
                F.DATE: curr_dt,
                "RAW_RET": day_raw_ret,
                "TURNOVER": turnover,
                "COUNT": len(current_holdings),
            }
        )

    # --- 4. 结算 ---
    res_daily = (
        pl.DataFrame(daily_records)
        .with_columns(
            [(pl.col("RAW_RET") - pl.col("TURNOVER") * cost_rate).alias("NET_RET")]
        )
        .with_columns([(pl.col("NET_RET") + 1).cum_prod().alias("NAV")])
    )

    return {"daily_results": res_daily, "trade_details": pl.DataFrame(trades_records)}
