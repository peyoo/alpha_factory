import numpy as np
import polars as pl
from numba import njit
from typing import Union, Dict
from loguru import logger

from alpha_factory.utils.schema import F


@njit(cache=True)
def _core_evolution_engine(
    rank_mat,
    price_mat,
    exec_p_mat,
    is_suspended_mat,
    is_up_limit_mat,
    is_down_limit_mat,
    trades_buf,
    n_buy,
    sell_rank,
    cost_rate,
):
    """纯数值计算核心，不涉及任何 Python 对象。

    trades_buf: 预分配的交易记录缓冲区，形状 (T * n_buy * 2, 5)
                列顺序: [col_j, entry_t, exit_t, entry_price, exit_price]
    返回: (nav_history, turnover_history, raw_ret_history, count_history,
           trades_buf, n_trades, is_holding, holdings_entry_price, holdings_entry_t,
           holdings_last_price)

    买入执行顺序与 daily_evolving 保持一致：按 rank 升序（rank=1 最优先）。
    信号 T 日生成，T+1 日执行。
    """
    T, N = rank_mat.shape

    # 状态变量
    holdings_pos_val = np.zeros(N)
    holdings_last_price = np.zeros(N)
    holdings_entry_price = np.zeros(N)
    holdings_entry_t = np.zeros(N, dtype=np.int64)
    is_holding = np.zeros(N, dtype=np.bool_)

    cash = 1.0
    pf_val = 1.0
    n_trades = 0

    # 结果数组
    nav_history = np.empty(T)
    turnover_history = np.empty(T)
    raw_ret_history = np.empty(T)
    count_history = np.empty(T, dtype=np.int64)

    # 卖出订单（布尔掩码，与 daily_evolving 逻辑相同）
    sell_orders = np.zeros(N, dtype=np.bool_)

    # 买入候选：以 rank 升序排列的列索引序列（替代原布尔掩码，保证选股优先级一致）
    buy_seq = np.empty(N, dtype=np.int64)  # 候选列索引（按 rank 升序）
    buy_seq_r = np.empty(N, dtype=np.float64)  # 对应 rank 值（用于插入排序）
    n_buy_seq = 0

    for t in range(T):
        pf_val_prev = pf_val
        num_bought = 0
        num_sold = 0

        # 1. 执行卖出 (T+1)
        for j in range(N):
            if sell_orders[j]:
                # 停牌或跌停无法卖出，顺延至下一日
                if is_suspended_mat[t, j] or is_down_limit_mat[t, j]:
                    continue

                exec_p = exec_p_mat[t, j]
                if np.isnan(exec_p):
                    continue

                proceeds = holdings_pos_val[j] * exec_p / holdings_last_price[j]
                cash += proceeds

                # 记录交易: [col_j, entry_t, exit_t, entry_price, exit_price]
                trades_buf[n_trades, 0] = j
                trades_buf[n_trades, 1] = holdings_entry_t[j]
                trades_buf[n_trades, 2] = t
                trades_buf[n_trades, 3] = holdings_entry_price[j]
                trades_buf[n_trades, 4] = exec_p
                n_trades += 1

                holdings_pos_val[j] = 0.0
                is_holding[j] = False
                sell_orders[j] = False
                num_sold += 1

        # 2. 执行买入 (T+1)，按前日生成的 rank 升序依次尝试
        current_count = 0
        for j in range(N):
            if is_holding[j]:
                current_count += 1

        for ci in range(n_buy_seq):
            if current_count >= n_buy:
                break
            j = buy_seq[ci]

            # 安全检查：信号生成后可能已被买入（理论上不应出现）
            if is_holding[j]:
                continue

            if is_suspended_mat[t, j] or is_up_limit_mat[t, j]:
                # 涨停/停牌顺延：下一次信号重新评估，不需要特殊处理
                continue

            exec_p = exec_p_mat[t, j]
            if np.isnan(exec_p):
                continue

            alloc = pf_val_prev / n_buy
            cash -= alloc
            holdings_pos_val[j] = alloc
            holdings_last_price[j] = exec_p
            holdings_entry_price[j] = exec_p
            holdings_entry_t[j] = t
            is_holding[j] = True
            num_bought += 1
            current_count += 1

        # 3. 估值 (T 日收盘)
        invested_value = 0.0
        holding_count = 0
        for j in range(N):
            if is_holding[j]:
                close_p = price_mat[t, j]
                if not np.isnan(close_p):
                    holdings_pos_val[j] *= close_p / holdings_last_price[j]
                    holdings_last_price[j] = close_p
                invested_value += holdings_pos_val[j]
                holding_count += 1

        pf_val_gross = invested_value + cash
        turnover = (num_bought + num_sold) / n_buy
        cost_total = turnover * cost_rate * pf_val_prev
        pf_val = pf_val_gross - cost_total
        cash -= cost_total

        raw_ret = (pf_val_gross / pf_val_prev - 1.0) if pf_val_prev > 0.0 else 0.0

        nav_history[t] = pf_val
        turnover_history[t] = turnover
        raw_ret_history[t] = raw_ret
        count_history[t] = holding_count

        # 4. 生成明日信号
        #    卖出：持仓中 rank >= sell_rank 的标的
        #    买入：rank <= n_buy 且未持有的标的，按 rank 升序插入排序存入 buy_seq
        sell_orders[:] = False
        n_buy_seq = 0

        for j in range(N):
            r = rank_mat[t, j]

            if is_holding[j] and r >= sell_rank:
                sell_orders[j] = True

            if (not is_holding[j]) and r <= n_buy:
                # 插入排序：将 (r, j) 插入 buy_seq，保持 rank 升序
                pos = n_buy_seq
                while pos > 0 and buy_seq_r[pos - 1] > r:
                    buy_seq_r[pos] = buy_seq_r[pos - 1]
                    buy_seq[pos] = buy_seq[pos - 1]
                    pos -= 1
                buy_seq_r[pos] = r
                buy_seq[pos] = j
                n_buy_seq += 1

    return (
        nav_history,
        turnover_history,
        raw_ret_history,
        count_history,
        trades_buf,
        n_trades,
        is_holding,
        holdings_entry_price,
        holdings_entry_t,
        holdings_last_price,
    )


def backtest_quick_daily(
    df_input: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    n_buy: int = 10,
    sell_rank: int = 30,
    cost_rate: float = 0.003,
    exec_price: str = F.VWAP,
    ascending: bool = False,
) -> Dict[str, pl.DataFrame]:
    """对外接口与 backtest_daily_evolving 保持一致，内部调用 numba JIT 加速。"""

    # logger.info(
    #     f"🚀 准备 JIT 回测数据 | 因子: {factor_col} | 买入/卖出线: {n_buy}/{sell_rank} | 费率: {cost_rate:.4f}"
    # )

    # --- 1. 预处理与排名（对齐 daily_evolving：POOL_MASK 过滤 + fill_null(999999)）---
    lf = df_input if isinstance(df_input, pl.LazyFrame) else df_input.lazy()
    lf = lf.with_columns(
        pl.when(pl.col(F.POOL_MASK))
        .then(pl.col(factor_col))
        .otherwise(None)
        .rank(descending=not ascending, method="random")
        .over(F.DATE)
        .fill_null(999999)
        .alias("RANK")
    ).select(
        [
            F.DATE,
            F.ASSET,
            "RANK",
            F.CLOSE,
            exec_price,
            F.IS_SUSPENDED,
            F.IS_UP_LIMIT,
            F.IS_DOWN_LIMIT,
        ]
    )

    df = lf.collect()
    date_list = df[F.DATE].unique().sort().to_list()
    T = len(date_list)

    # --- 2. 矩阵化（单次双 Join 替代 6 次 Pivot，避免重复全表扫描）---
    asset_names = df[F.ASSET].unique().sort().to_list()
    N = len(asset_names)

    date_idx_df = pl.DataFrame({F.DATE: date_list, "_t": list(range(T))})
    asset_idx_df = pl.DataFrame({F.ASSET: asset_names, "_j": list(range(N))})
    df_idx = df.join(date_idx_df, on=F.DATE).join(asset_idx_df, on=F.ASSET)
    ti = df_idx["_t"].to_numpy()
    ji = df_idx["_j"].to_numpy()

    rank_mat = np.full((T, N), 999999.0, dtype=np.float64)
    price_mat = np.full((T, N), np.nan, dtype=np.float64)
    exec_p_mat = np.full((T, N), np.nan, dtype=np.float64)
    is_suspended_mat = np.zeros((T, N), dtype=np.bool_)
    is_up_limit_mat = np.zeros((T, N), dtype=np.bool_)
    is_down_limit_mat = np.zeros((T, N), dtype=np.bool_)

    rank_mat[ti, ji] = df_idx["RANK"].cast(pl.Float64).fill_null(999999).to_numpy()
    price_mat[ti, ji] = df_idx[F.CLOSE].cast(pl.Float64).to_numpy()
    exec_p_mat[ti, ji] = df_idx[exec_price].cast(pl.Float64).to_numpy()
    is_suspended_mat[ti, ji] = df_idx[F.IS_SUSPENDED].fill_null(False).to_numpy()
    is_up_limit_mat[ti, ji] = df_idx[F.IS_UP_LIMIT].fill_null(False).to_numpy()
    is_down_limit_mat[ti, ji] = df_idx[F.IS_DOWN_LIMIT].fill_null(False).to_numpy()

    # --- 3. 预分配 trades_buf 并调用 JIT 核心 ---
    max_trades = T * n_buy * 2
    trades_buf = np.zeros((max_trades, 5), dtype=np.float64)

    (
        nav,
        turnover,
        raw_ret,
        count,
        trades_buf,
        n_trades,
        is_holding,
        holdings_entry_price,
        holdings_entry_t,
        holdings_last_price,
    ) = _core_evolution_engine(
        rank_mat,
        price_mat,
        exec_p_mat,
        is_suspended_mat,
        is_up_limit_mat,
        is_down_limit_mat,
        trades_buf,
        n_buy,
        sell_rank,
        cost_rate,
    )

    # --- 4. 构建 daily_results ---
    net_ret = raw_ret - turnover * cost_rate
    res_daily = pl.DataFrame(
        {
            F.DATE: date_list,
            "NAV": nav,
            "TURNOVER": turnover,
            "RAW_RET": raw_ret,
            "NET_RET": net_ret,
            "COUNT": count,
        }
    )

    # --- 5. 构建 trade_details ---
    trade_rows = trades_buf[:n_trades]

    # 补充未平仓持仓（以最后交易日价格收盘）
    open_j = np.where(is_holding)[0]
    if len(open_j) > 0:
        last_t = T - 1
        extra = np.column_stack(
            [
                open_j.astype(np.float64),
                holdings_entry_t[open_j].astype(np.float64),
                np.full(len(open_j), last_t, dtype=np.float64),
                holdings_entry_price[open_j],
                holdings_last_price[open_j],
            ]
        )
        trade_rows = np.vstack([trade_rows, extra])

    if len(trade_rows) > 0:
        col_j = trade_rows[:, 0].astype(int)
        entry_t = trade_rows[:, 1].astype(int)
        exit_t = trade_rows[:, 2].astype(int)
        entry_prices = trade_rows[:, 3]
        exit_prices = trade_rows[:, 4]

        trade_details = pl.DataFrame(
            {
                F.ASSET: [asset_names[j] for j in col_j],
                "entry_date": [date_list[t] for t in entry_t],
                "exit_date": [date_list[t] for t in exit_t],
                "entry_price": entry_prices,
                "exit_price": exit_prices,
                "pnl_ret": exit_prices / entry_prices - 1.0,
                "holding_periods": exit_t - entry_t,
            }
        )
    else:
        trade_details = pl.DataFrame(
            {
                F.ASSET: pl.Series([], dtype=pl.Utf8),
                "entry_date": pl.Series([], dtype=pl.Date),
                "exit_date": pl.Series([], dtype=pl.Date),
                "entry_price": pl.Series([], dtype=pl.Float64),
                "exit_price": pl.Series([], dtype=pl.Float64),
                "pnl_ret": pl.Series([], dtype=pl.Float64),
                "holding_periods": pl.Series([], dtype=pl.Int64),
            }
        )

    logger.info(
        f"✅ JIT 回测完成 | 交易笔数: {len(trade_details)} | 最终净值: {nav[-1]:.4f}"
    )

    return {"daily_results": res_daily, "trade_details": trade_details}
