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
    exec_price: str = F.VWAP,
    ascending: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    逐日演进回测框架 (基金净值版 · 无限可拆分)

    执行时序 (每个交易日 T+1)：
      1. 执行前日生成的 sell_orders（持仓中排名超出 sell_rank 的标的）
      2. 执行前日生成的 buy_orders（排名在 n_buy 内的新标的）
      3. 以当日收盘价对全部持仓估值，计算组合净值
      4. 根据当日排名为明日生成新的 sell_orders / buy_orders

    基金净值计算（无限可拆分假设）：
      - 不追踪股数 units，直接以 pos_val（持仓现值）管理每只标的的仓位
      - 建仓：pos_val = pf_val_prev / n_buy，last_price = exec_price
      - 每日估值：pos_val *= close_p / last_price，并更新 last_price = close_p
      - 卖出：proceeds = pos_val * exec_price / last_price，回收为现金
      - 总市值 pf_val = Σ pos_val_i + cash
      - 交易费用 = turnover × cost_rate × 前日净值，从现金中扣除
    """

    # --- 1. 数据预处理 ---
    lf = df_input if isinstance(df_input, pl.LazyFrame) else df_input.lazy()

    # 预计算排名（T 日收盘产生信号，T+1 日执行）
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

    # --- 2. 组合状态 ---
    # holdings: {asset: {pos_val, last_price, entry_price, entry_date, entry_idx}}
    #   pos_val    — 持仓现值（基金货币单位），随价格每日更新
    #   last_price — 最近一次已知价格，用于计算涨跌幅
    holdings: dict = {}
    cash: float = 1.0  # 闲置资金（初始全部为现金，fund_value = 1.0）
    pf_val: float = 1.0  # 组合总市值 = 基金净值，从 1.0 起步

    # 前日生成、当日执行的订单（T 日信号 → T+1 日执行）
    sell_orders: set = set()  # 平仓标的集合
    buy_orders: list = []  # 建仓标的列表（已按排名升序排列）

    daily_records = []
    trades_records = []

    logger.info(
        f"🚀 启动回测演进 | 因子: {factor_col} | 买入/卖出线: {n_buy}/{sell_rank} | 费率: {cost_rate:.4f}"
    )

    # --- 3. 逐日演进 ---
    for i, curr_dt in enumerate(all_dates):
        day_df = grouped.get((curr_dt,))
        if day_df is None:
            # 当日无数据，持仓与净值不变
            daily_records.append(
                {
                    F.DATE: curr_dt,
                    "RAW_RET": 0.0,
                    "NET_RET": 0.0,
                    "TURNOVER": 0.0,
                    "COUNT": len(holdings),
                    "NAV": pf_val,
                }
            )
            continue

        day_info = {row[F.ASSET]: row for row in day_df.to_dicts()}
        pf_val_prev = pf_val
        num_bought = 0
        num_sold = 0

        # ================================================================
        # STEP 1: 执行卖出订单（来自前日信号）
        # ================================================================
        for asset in list(sell_orders):
            hold = holdings.get(asset)
            if hold is None:
                logger.error(f"卖出标的 {asset} 在 {curr_dt} 不在持仓中，跳过")
                continue

            info = day_info.get(asset)
            if info is None:
                logger.warning(
                    f"卖出标的 {asset} 在 {curr_dt} 无数据（停牌），继续持有"
                )
                continue

            exec_p = info.get(exec_price)
            if exec_p is None:
                logger.warning(f"卖出标的 {asset} 在 {curr_dt} 执行价为 null，继续持有")
                continue

            if info[F.IS_SUSPENDED]:
                logger.info(f"标的 {asset} 在 {curr_dt} 停牌，无法卖出，顺延至下一日")
                continue
            if info[F.IS_DOWN_LIMIT]:
                logger.info(f"标的 {asset} 在 {curr_dt} 跌停，无法卖出，顺延至下一日")
                continue

            # 成功卖出：以执行价折算当前持仓现值，回收为现金
            proceeds = hold["pos_val"] * exec_p / hold["last_price"]
            cash += proceeds
            trades_records.append(
                {
                    F.ASSET: asset,
                    "entry_date": hold["entry_date"],
                    "exit_date": curr_dt,
                    "entry_price": hold["entry_price"],
                    "exit_price": exec_p,
                    "pnl_ret": exec_p / hold["entry_price"] - 1,
                    "holding_periods": i - hold["entry_idx"],
                }
            )
            del holdings[asset]
            sell_orders.discard(asset)
            num_sold += 1

        # ================================================================
        # STEP 2: 执行买入订单（来自前日信号）
        # ================================================================
        for asset in buy_orders:
            if len(holdings) >= n_buy:
                break
            if asset in holdings:
                logger.error(f"买入标的 {asset} 在 {curr_dt} 已在持仓中，跳过")
                continue

            info = day_info.get(asset)
            if info is None:
                logger.error(f"买入标的 {asset} 在 {curr_dt} 无数据，无法建仓，跳过")
                continue

            exec_p = info.get(exec_price)
            if exec_p is None:
                logger.error(
                    f"买入标的 {asset} 在 {curr_dt} 执行价为 null，无法建仓，跳过"
                )
                continue

            if info[F.IS_UP_LIMIT] or info[F.IS_SUSPENDED]:
                logger.info(
                    f"标的 {asset} 在 {curr_dt} 涨停或停牌，无法买入，顺延至下一日"
                )
                continue

            # 以前日净值的 1/n_buy 建仓，仓位直接以资金价值记录
            alloc = pf_val_prev / n_buy
            cash -= alloc

            holdings[asset] = {
                "pos_val": alloc,  # 持仓现值（随后每日随价格更新）
                "last_price": exec_p,  # 建仓参考价，用于后续折算涨跌幅
                "entry_price": exec_p,
                "entry_date": curr_dt,
                "entry_idx": i,
            }
            num_bought += 1

        # ================================================================
        # STEP 3: 以收盘价更新持仓估值，计算当日组合净值
        # ================================================================
        invested_value = 0.0
        for asset, hold in holdings.items():
            info = day_info.get(asset)
            if info is None:
                # 停牌无数据，pos_val 不变，沿用昨日价格
                invested_value += hold["pos_val"]
                logger.warning(f"持仓 {asset} 在 {curr_dt} 无数据，沿用昨日估值")
                continue

            close_p = info.get(F.CLOSE)
            if close_p is None:
                invested_value += hold["pos_val"]
                logger.warning(f"持仓 {asset} 在 {curr_dt} 收盘价为 null，沿用昨日估值")
                continue

            # 以价格涨跌幅更新持仓价值
            hold["pos_val"] = hold["pos_val"] * close_p / hold["last_price"]
            hold["last_price"] = close_p
            invested_value += hold["pos_val"]

        # 组合毛市值 = 已投资部分 + 现金
        pf_val_gross = invested_value + cash

        # 扣除交易费用（按前日净值的换手比例计量）
        turnover = (num_bought + num_sold) / n_buy
        cost_total = turnover * cost_rate * pf_val_prev
        pf_val = pf_val_gross - cost_total
        cash -= cost_total  # 费用从现金中扣除

        raw_ret = (pf_val_gross / pf_val_prev - 1) if pf_val_prev > 0 else 0.0
        net_ret = raw_ret - turnover * cost_rate

        # ================================================================
        # STEP 4: 根据当日排名，为明日生成 sell_orders / buy_orders
        # ================================================================
        # 持仓中排名超出 sell_rank 的标的，明日平仓
        sell_orders = {
            a for a in holdings if day_info.get(a, {}).get("RANK", 999999) >= sell_rank
        }

        # 排名在 n_buy 以内、且未持有的标的，明日按排名顺序建仓
        buy_candidates = [
            a for a in day_info if day_info[a]["RANK"] <= n_buy and a not in holdings
        ]
        buy_orders = sorted(buy_candidates, key=lambda a: day_info[a]["RANK"])

        daily_records.append(
            {
                F.DATE: curr_dt,
                "RAW_RET": raw_ret,
                "NET_RET": net_ret,
                "TURNOVER": turnover,
                "COUNT": len(holdings),
                "NAV": pf_val,
            }
        )

    # --- 4. 输出结果 ---
    res_daily = pl.DataFrame(daily_records)

    return {"daily_results": res_daily, "trade_details": pl.DataFrame(trades_records)}
