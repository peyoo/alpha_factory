"""
诊断：找出两引擎第一个发散点，并对比前 N 天的买入选股差异。
用法: uv run python scripts/debug_bt_divergence.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from rich.console import Console

from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.utils.schema import F
from alpha_factory.cli.opt import _resolve_pool

console = Console()

YAML_PATH = Path("output/strategies/s2.yaml")
START_DATE = "20190101"
END_DATE = None
_EXEC_PRICE_MAP = {"open": F.OPEN, "close": F.CLOSE, "vwap": F.VWAP}


def simulate_evolving_nav(
    df: pl.DataFrame,
    factor_col: str,
    n_buy: int,
    sell_rank: int,
    cost_rate: float,
    exec_price: str,
):
    """Stripped-down version of daily_evolving that records daily buy selections."""
    lf = df.lazy().with_columns(
        pl.when(pl.col(F.POOL_MASK))
        .then(pl.col(factor_col))
        .otherwise(None)
        .rank(descending=True, method="ordinal")  # ← ordinal for determinism
        .over(F.DATE)
        .fill_null(999999)
        .alias("RANK")
    )
    df2 = lf.select(
        [
            F.DATE,
            F.ASSET,
            exec_price,
            F.CLOSE,
            "RANK",
            F.IS_UP_LIMIT,
            F.IS_DOWN_LIMIT,
            F.IS_SUSPENDED,
        ]
    ).collect()

    all_dates = df2[F.DATE].unique().sort().to_list()
    grouped = {k: v for k, v in df2.partition_by(F.DATE, as_dict=True).items()}

    holdings: dict = {}
    cash = 1.0
    pf_val = 1.0
    sell_orders: set = set()
    buy_orders: list = []
    nav_list = []
    buy_log = []  # (date, bought_assets)

    for i, curr_dt in enumerate(all_dates):
        day_df = grouped.get((curr_dt,))
        if day_df is None:
            nav_list.append(pf_val)
            buy_log.append((curr_dt, []))
            continue
        day_info = {row[F.ASSET]: row for row in day_df.to_dicts()}
        pf_val_prev = pf_val
        num_bought = 0
        num_sold = 0

        for asset in list(sell_orders):
            hold = holdings.get(asset)
            if hold is None:
                continue
            info = day_info.get(asset)
            if info is None:
                continue
            exec_p = info.get(exec_price)
            if exec_p is None:
                continue
            if info[F.IS_SUSPENDED] or info[F.IS_DOWN_LIMIT]:
                continue
            proceeds = hold["pos_val"] * exec_p / hold["last_price"]
            cash += proceeds
            del holdings[asset]
            sell_orders.discard(asset)
            num_sold += 1

        bought_today = []
        for asset in buy_orders:
            if len(holdings) >= n_buy:
                break
            if asset in holdings:
                continue
            info = day_info.get(asset)
            if info is None:
                continue
            exec_p = info.get(exec_price)
            if exec_p is None:
                continue
            if info[F.IS_UP_LIMIT] or info[F.IS_SUSPENDED]:
                continue
            alloc = pf_val_prev / n_buy
            cash -= alloc
            holdings[asset] = {"pos_val": alloc, "last_price": exec_p}
            num_bought += 1
            bought_today.append(f"{asset}(r={day_info[asset]['RANK']})")

        buy_log.append((curr_dt, bought_today))

        invested_value = 0.0
        for asset, hold in holdings.items():
            info = day_info.get(asset)
            if info is None:
                invested_value += hold["pos_val"]
                continue
            close_p = info.get(F.CLOSE)
            if close_p is None:
                invested_value += hold["pos_val"]
                continue
            hold["pos_val"] = hold["pos_val"] * close_p / hold["last_price"]
            hold["last_price"] = close_p
            invested_value += hold["pos_val"]

        pf_val_gross = invested_value + cash
        turnover = (num_bought + num_sold) / n_buy
        pf_val = pf_val_gross - turnover * cost_rate * pf_val_prev
        cash -= turnover * cost_rate * pf_val_prev
        nav_list.append(pf_val)

        sell_orders = {
            a for a in holdings if day_info.get(a, {}).get("RANK", 999999) >= sell_rank
        }
        buy_candidates = [
            a for a in day_info if day_info[a]["RANK"] <= n_buy and a not in holdings
        ]
        buy_orders = sorted(buy_candidates, key=lambda a: day_info[a]["RANK"])

    return np.array(nav_list), all_dates, buy_log


def simulate_quick_colorder(
    df: pl.DataFrame,
    factor_col: str,
    n_buy: int,
    sell_rank: int,
    cost_rate: float,
    exec_price: str,
):
    """Quick engine but with pivot column-index buy order (to reproduce the bug)."""
    lf = (
        df.lazy()
        .with_columns(
            pl.when(pl.col(F.POOL_MASK))
            .then(pl.col(factor_col))
            .otherwise(None)
            .rank(descending=True, method="ordinal")  # ← ordinal for determinism
            .over(F.DATE)
            .fill_null(999999)
            .alias("RANK")
        )
        .select(
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
    )

    df2 = lf.collect()
    pivot_base = df2.pivot(index=F.DATE, on=F.ASSET, values="RANK").sort(F.DATE)
    asset_names = pivot_base.drop(F.DATE).columns

    rank_mat = np.nan_to_num(pivot_base.drop(F.DATE).to_numpy(), nan=999999).astype(
        np.float64
    )

    def to_mat(col):
        return (
            df2.pivot(index=F.DATE, on=F.ASSET, values=col)
            .sort(F.DATE)
            .drop(F.DATE)
            .to_numpy()
            .astype(np.float64)
        )

    price_mat = to_mat(F.CLOSE)
    exec_p_mat = to_mat(exec_price)
    is_susp = np.nan_to_num(to_mat(F.IS_SUSPENDED), nan=0).astype(bool)
    is_up = np.nan_to_num(to_mat(F.IS_UP_LIMIT), nan=0).astype(bool)
    is_dn = np.nan_to_num(to_mat(F.IS_DOWN_LIMIT), nan=0).astype(bool)

    dates = pivot_base[F.DATE].to_list()
    T, N = rank_mat.shape

    pos_val = np.zeros(N)
    last_p = np.zeros(N)
    is_hold = np.zeros(N, dtype=bool)
    entry_p = np.zeros(N)

    cash = 1.0
    pf_val = 1.0

    sell_ord = np.zeros(N, dtype=bool)
    buy_ord = np.zeros(N, dtype=bool)

    nav_list = []
    buy_log = []

    for t in range(T):
        pf_prev = pf_val
        num_b = num_s = 0

        # sell
        for j in range(N):
            if sell_ord[j]:
                if is_susp[t, j] or is_dn[t, j]:
                    continue
                ep = exec_p_mat[t, j]
                if np.isnan(ep):
                    continue
                cash += pos_val[j] * ep / last_p[j]
                pos_val[j] = 0
                is_hold[j] = False
                sell_ord[j] = False
                num_s += 1

        # buy (column-index order ← BUG)
        cc = int(is_hold.sum())
        bought_today = []
        for j in range(N):
            if buy_ord[j] and cc < n_buy:
                if is_susp[t, j] or is_up[t, j]:
                    continue
                ep = exec_p_mat[t, j]
                if np.isnan(ep):
                    continue
                alloc = pf_prev / n_buy
                cash -= alloc
                pos_val[j] = alloc
                last_p[j] = ep
                entry_p[j] = ep
                is_hold[j] = True
                buy_ord[j] = False
                num_b += 1
                cc += 1
                bought_today.append(f"{asset_names[j]}(r={int(rank_mat[t, j])})")
        buy_log.append((dates[t], bought_today))

        # value
        inv = 0.0
        for j in range(N):
            if is_hold[j]:
                cp = price_mat[t, j]
                if not np.isnan(cp):
                    pos_val[j] = pos_val[j] * cp / last_p[j]
                    last_p[j] = cp
                inv += pos_val[j]

        pf_gross = inv + cash
        to = (num_b + num_s) / n_buy
        pf_val = pf_gross - to * cost_rate * pf_prev
        cash -= to * cost_rate * pf_prev
        nav_list.append(pf_val)

        # signals
        sell_ord[:] = False
        buy_ord[:] = False
        for j in range(N):
            r = rank_mat[t, j]
            if is_hold[j] and r >= sell_rank:
                sell_ord[j] = True
            if (not is_hold[j]) and r <= n_buy:
                buy_ord[j] = True

    return np.array(nav_list), dates, buy_log


def main():
    cfg = StrategyConfig.from_yaml(YAML_PATH)
    pool_instance = _resolve_pool(cfg.pool)
    exec_price_col = _EXEC_PRICE_MAP.get(cfg.exe_price.strip().lower(), F.VWAP)
    factor_col = cfg.ranks[0].name

    dp = DataProvider()
    all_exprs = cfg.ranked_factor_exprs + cfg.get_condition_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        START_DATE,
        END_DATE,
        exprs=all_exprs,
        actions=cfg.build_actions(),
    )
    df = lf.collect()

    console.print("[bold green]▶ 运行 evolving (ordinal rank) ...[/bold green]")
    nav_e, dates_e, buy_log_e = simulate_evolving_nav(
        df, factor_col, cfg.hold_num, cfg.sell_rank, cfg.cost, exec_price_col
    )

    console.print(
        "[bold yellow]▶ 运行 quick col-order (ordinal rank) ...[/bold yellow]"
    )
    nav_q, dates_q, buy_log_q = simulate_quick_colorder(
        df, factor_col, cfg.hold_num, cfg.sell_rank, cfg.cost, exec_price_col
    )

    n = min(len(nav_e), len(nav_q))
    diff = np.abs(nav_e[:n] - nav_q[:n])

    # 找第一个差异 > 1e-6 的日期
    first_div = np.argmax(diff > 1e-6)
    if diff[first_div] > 1e-6:
        console.print(
            f"\n[bold red]第一个发散点: {dates_e[first_div]} "
            f"diff={diff[first_div]:.8f}[/bold red]"
        )

        # 展示前后 5 天的买入选股差异
        start = max(0, first_div - 2)
        end = min(n, first_div + 8)
        console.print("\n[bold]买入选股对比 (evolving vs col-order quick)[/bold]")
        for i in range(start, end):
            marker = " ←" if i == first_div else ""
            bought_e = buy_log_e[i][1]
            bought_q = buy_log_q[i][1]
            if bought_e != bought_q:
                console.print(f"[red]{dates_e[i]}{marker}[/red]")
                console.print(f"  evolving: {bought_e}")
                console.print(f"  quick:    {bought_q}")
            else:
                console.print(f"[dim]{dates_e[i]}: 买入一致 {bought_e}[/dim]")
    else:
        console.print("[bold green]两引擎完全一致（ordinal rank）[/bold green]")

    console.print(
        f"\n最终 NAV — evolving: {nav_e[-1]:.6f} | quick: {nav_q[-1]:.6f} | diff: {nav_e[-1] - nav_q[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
