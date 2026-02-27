from typing import Union, Dict
import polars as pl
from loguru import logger
from alpha_factory.utils.schema import F


def backtest_daily_evolving(
    df_input: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    n_buy: int = 10,
    sell_rank: int = 30,
    cost_rate: float = 0.003,
    exec_price: str = F.OPEN,
    ascending: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    逐日演进回测框架 (精准收益捕捉版)
    1. 收益闭环：捕捉了买入当天 (Exec -> Close) 和 卖出当天 (Prev_Close -> Exec) 的收益。
    2. 信号对齐：确保 T 日信号在 T+1 日准时执行。
    """

    # --- 1. 数据预处理 ---
    lf = df_input.lazy() if isinstance(df_input, pl.LazyFrame) else df_input.lazy()

    # 预计算排名 (T日收盘产生信号)
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

    # 采集核心字段：增加 F.CLOSE 用于计算买入当日收盘后的估值
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
    current_holdings = {}  # {Asset: {"entry_price":..., "last_price":..., "entry_date":..., "entry_idx":...}}
    must_hold_list = set()
    can_hold_list = set()

    daily_records = []
    trades_records = []

    logger.info(
        f"🚀 启动回测演进 | 因子: {factor_col} | 买入/卖出线: {n_buy}/{sell_rank} | 费率: {cost_rate:.4f}"
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

        # A. 交易执行与收益计算 (T+1日)

        # 1. 先处理原有持仓的卖出与持仓收益
        for asset, hold_info in current_holdings.items():
            info = day_info.get(asset)
            if not info:
                # 停牌无数据，收益为0，保留状态
                new_holdings[asset] = hold_info
                continue

            price_today_exec = info[exec_price]
            price_yesterday_close = hold_info["last_price"]

            # 价格为 null（真实停牌/数据缺失）：跳过收益计算，保留持仓
            if price_today_exec is None or price_yesterday_close is None:
                new_holdings[asset] = hold_info
                continue

            # 无论卖不卖，都要计算从“昨收”到“今日执行价”的收益贡献
            day_raw_ret += ((price_today_exec / price_yesterday_close) - 1) / n_buy

            # 判定卖出信号
            if asset not in must_hold_list and asset not in can_hold_list:
                if info[F.IS_DOWN_LIMIT] or info[F.IS_SUSPENDED]:
                    # 卖不掉，更新价格继续持有（防止 close 为 null）
                    new_close = info[F.CLOSE]
                    if new_close is not None:
                        day_raw_ret += ((new_close / price_today_exec) - 1) / n_buy
                        hold_info["last_price"] = new_close
                    else:
                        hold_info["last_price"] = price_today_exec
                    new_holdings[asset] = hold_info
                else:
                    # 成功卖出：记录闭环交易
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
                # 继续持有：累加“执行价到今日收盘”的收益（防止 close 为 null）
                new_close = info[F.CLOSE]
                if new_close is not None:
                    day_raw_ret += ((new_close / price_today_exec) - 1) / n_buy
                    hold_info["last_price"] = new_close
                else:
                    hold_info["last_price"] = price_today_exec
                new_holdings[asset] = hold_info

        # 2. 处理新标的买入
        potential_buys = [a for a in must_hold_list if a not in new_holdings]
        potential_buys.sort(
            key=lambda x: day_info[x]["RANK"] if x in day_info else 999999
        )

        for asset in potential_buys:
            if len(new_holdings) < n_buy:
                info = day_info.get(asset)
                if info and not (info[F.IS_UP_LIMIT] or info[F.IS_SUSPENDED]):
                    price_buy_exec = info[exec_price]
                    # 计算买入后到收盘的收益贡献
                    day_raw_ret += ((info[F.CLOSE] / price_buy_exec) - 1) / n_buy

                    new_holdings[asset] = {
                        "entry_date": curr_dt,
                        "entry_price": price_buy_exec,
                        "last_price": info[F.CLOSE],
                        "entry_idx": i,
                    }
                    num_bought += 1

        # B. 信号更新 (为明天 T+2 做准备)
        must_hold_list = {a for a, info in day_info.items() if info["RANK"] <= n_buy}
        can_hold_list = {
            a for a, info in day_info.items() if n_buy < info["RANK"] <= sell_rank
        }

        # C. 记录流水
        turnover = (num_bought + num_sold) / n_buy
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
