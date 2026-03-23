import numpy as np
import polars as pl
from numba import njit
from typing import Union, Dict
from loguru import logger

# 假设 F 是你的 Schema 类，如果不在同一个文件请导入
from alpha_factory.utils.schema import F


@njit
def _core_evolution_engine(
    rank_mat,
    price_mat,
    exec_p_mat,
    is_suspended_mat,
    is_up_limit_mat,
    is_down_limit_mat,
    n_buy,
    sell_rank,
    cost_rate,
):
    """纯数值计算核心，不涉及任何 Python 对象"""
    T, N = rank_mat.shape

    # 状态变量
    holdings_pos_val = np.zeros(N)
    holdings_last_price = np.zeros(N)
    is_holding = np.zeros(N, dtype=np.bool_)

    cash = 1.0
    pf_val = 1.0

    # 结果数组
    nav_history = np.empty(T)
    turnover_history = np.empty(T)

    # 订单缓存
    sell_orders = np.zeros(N, dtype=np.bool_)
    buy_orders = np.zeros(N, dtype=np.bool_)

    for t in range(T):
        pf_val_prev = pf_val
        num_bought = 0
        num_sold = 0

        # 1. 执行卖出 (T+1)
        for j in range(N):
            if sell_orders[j]:
                # 停牌或跌停无法卖出
                if is_suspended_mat[t, j] or is_down_limit_mat[t, j]:
                    continue

                exec_p = exec_p_mat[t, j]
                # 防止 exec_p 为 nan
                if np.isnan(exec_p):
                    continue

                proceeds = holdings_pos_val[j] * exec_p / holdings_last_price[j]
                cash += proceeds
                holdings_pos_val[j] = 0
                is_holding[j] = False
                sell_orders[j] = False
                num_sold += 1

        # 2. 执行买入 (T+1)
        # 统计当前持仓数
        current_count = 0
        for j in range(N):
            if is_holding[j]:
                current_count += 1

        for j in range(N):
            if buy_orders[j] and current_count < n_buy:
                if is_suspended_mat[t, j] or is_up_limit_mat[t, j]:
                    continue

                exec_p = exec_p_mat[t, j]
                if np.isnan(exec_p):
                    continue

                alloc = pf_val_prev / n_buy
                cash -= alloc
                holdings_pos_val[j] = alloc
                holdings_last_price[j] = exec_p
                is_holding[j] = True
                buy_orders[j] = False
                num_bought += 1
                current_count += 1

        # 3. 估值 (T 日收盘)
        invested_value = 0.0
        for j in range(N):
            if is_holding[j]:
                close_p = price_mat[t, j]
                # 若当日无价格（停牌），保持估值不变
                if not np.isnan(close_p):
                    holdings_pos_val[j] *= close_p / holdings_last_price[j]
                    holdings_last_price[j] = close_p
                invested_value += holdings_pos_val[j]

        pf_val_gross = invested_value + cash
        turnover = (num_bought + num_sold) / n_buy
        pf_val = pf_val_gross - (turnover * cost_rate * pf_val_prev)
        cash -= turnover * cost_rate * pf_val_prev

        nav_history[t] = pf_val
        turnover_history[t] = turnover

        # 4. 生成明日信号
        sell_orders[:] = False
        buy_orders[:] = False
        for j in range(N):
            r = rank_mat[t, j]
            if is_holding[j] and r >= sell_rank:
                sell_orders[j] = True
            if (not is_holding[j]) and r <= n_buy:
                buy_orders[j] = True

    return nav_history, turnover_history


def backtest_quick_daily(
    df_input: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    n_buy: int = 10,
    sell_rank: int = 30,
    cost_rate: float = 0.003,
    exec_price: str = F.VWAP,  # 示例，根据实际 F.VWAP 修改
    ascending: bool = False,
) -> Dict[str, pl.DataFrame]:
    """对外接口保持一致，内部调用 JIT"""

    logger.info(f"🚀 准备 JIT 回测数据 | 因子: {factor_col}")

    # 1. 预处理与排名
    lf = df_input if isinstance(df_input, pl.LazyFrame) else df_input.lazy()
    lf = lf.with_columns(
        RANK=pl.col(factor_col)
        .rank(descending=not ascending, method="random")
        .over("date")
    ).select(
        [
            "date",
            "asset",
            "RANK",
            "close",
            exec_price,
            "is_suspended",
            "is_up_limit",
            "is_down_limit",
        ]
    )

    df = lf.collect()
    dates = df["date"].unique().sort()

    # 2. 矩阵化 (Pivot) —— 这是最消耗内存的一步
    def to_mat(col):
        # 使用 pivot 将长表转宽表，确保所有矩阵行列索引完全对齐
        return (
            df.pivot(index="date", on="asset", values=col)
            .sort("date")
            .drop("date")
            .to_numpy()
        )

    rank_mat = np.nan_to_num(to_mat("RANK"), nan=999999)
    price_mat = to_mat("close")
    exec_p_mat = to_mat(exec_price)

    # 限制状态转为布尔矩阵
    is_suspended_mat = np.nan_to_num(to_mat("is_suspended"), nan=0).astype(np.bool_)
    is_up_limit_mat = np.nan_to_num(to_mat("is_up_limit"), nan=0).astype(np.bool_)
    is_down_limit_mat = np.nan_to_num(to_mat("is_down_limit"), nan=0).astype(np.bool_)

    # 3. 调用 JIT 核心
    nav, turnover = _core_evolution_engine(
        rank_mat,
        price_mat,
        exec_p_mat,
        is_suspended_mat,
        is_up_limit_mat,
        is_down_limit_mat,
        n_buy,
        sell_rank,
        cost_rate,
    )

    # 4. 封装结果
    res_daily = pl.DataFrame(
        {
            "date": dates,
            "NAV": nav,
            "TURNOVER": turnover,
            "NET_RET": pl.Series(nav).pct_change().fill_null(0),  # 简易计算
        }
    )

    return {"daily_results": res_daily}
