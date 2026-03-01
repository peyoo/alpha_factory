from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.evaluation.backtest.period import backtest_periodic_rebalance
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.utils.schema import F

console = Console()

# 执行价字段映射：CLI 友好字符串 → schema 常量
_EXEC_PRICE_MAP: dict[str, str] = {
    "open": F.OPEN,
    "close": F.CLOSE,
    "vwap": F.VWAP,
}


def quant_bt(
    start_date: str = typer.Option(
        "20190101", "-s", "--start-date", help="开始日期 YYYYMMDD"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "--end-date", help="结束日期 YYYYMMDD（默认至最新）"
    ),
    expr: str = typer.Option(
        ...,
        "-e",
        "--expr",
        help="因子表达式，格式: FACTOR_NAME = <polars-ta 表达式>，例如 F1 = CLOSE.rolling_mean(5)",
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small, "--pool", help="股票池"
    ),
    mode: str = typer.Option(
        "daily",
        "--mode",
        help="回测模式: daily（逐日演进，默认）| period（周期换股）",
    ),
    n_buy: int = typer.Option(
        10, "--n-buy", help="最大持仓数（daily: 买入线；period: 持股数量）"
    ),
    sell_rank: int = typer.Option(
        30, "--sell-rank", help="卖出线：排名超过此值时触发卖出（仅 daily 模式）"
    ),
    rebalance_period: int = typer.Option(
        5,
        "--period",
        help="换股周期（天数），如 5 表示每 5 天换股一次（仅 period 模式）",
    ),
    cost: float = typer.Option(0.003, "--cost", help="单边交易成本率"),
    exec_price: str = typer.Option(
        "vwap",
        "--exec-price",
        help="执行价字段: open（默认，后复权开盘价）| close（后复权收盘价）",
    ),
    ascending: bool = typer.Option(
        False,
        "--ascending/--descending",
        help="因子排序方向：--ascending 表示小值买入，--descending（默认）表示大值买入",
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="是否生成 HTML 报告并打开"
    ),
    save_trades: Optional[Path] = typer.Option(
        None,
        "--save-trades",
        help="保存交易明细到文件，按扩展名自动选格式（.csv 或 .parquet），"
        "例如 --save-trades output/trades.csv",
        show_default=False,
    ),
):
    """
    因子回测框架，支持两种模式。

    \b
    模式说明:
      daily   逐日演进回测（T+1 精准收益闭环），支持动态调仓
      period  周期换股回测，按固定周期整体换仓

    \b
    示例:
      quant bt -s 20200101 -e 20231231 --expr "F1 = -ts_mean(AMOUNT, 30)"
      quant bt -s 20200101 --expr "MOM = CLOSE / CLOSE.shift(20) - 1" --n-buy 20 --sell-rank 50
      quant bt --mode period -s 20200101 --expr "F1 = -CLOSE.shift(1)" --n-buy 10 --period 5
    """
    # --- 1. 参数校验 ---
    mode_lower = mode.strip().lower()
    if mode_lower not in ("daily", "period"):
        typer.echo(
            f"❌ --mode 无效值 '{mode}'，仅支持: daily | period",
            err=True,
        )
        raise typer.Exit(code=1)

    exec_price_lower = exec_price.strip().lower()
    if exec_price_lower not in _EXEC_PRICE_MAP:
        typer.echo(
            f"❌ --exec-price 无效值 '{exec_price}'，仅支持: {list(_EXEC_PRICE_MAP.keys())}",
            err=True,
        )
        raise typer.Exit(code=1)
    exec_price_col = _EXEC_PRICE_MAP[exec_price_lower]

    if mode_lower == "daily" and sell_rank < n_buy:
        typer.echo(
            f"❌ --sell-rank ({sell_rank}) 必须 >= --n-buy ({n_buy})",
            err=True,
        )
        raise typer.Exit(code=1)

    if mode_lower == "period" and rebalance_period < 1:
        typer.echo(
            f"❌ --period ({rebalance_period}) 必须 >= 1",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- 2. 数据加载 ---
    # 若表达式中不含 '='，视为纯表达式，自动添加默认因子名 f1
    if "=" not in expr:
        expr = f"f1 = {expr.strip()}"
    factor_col = expr.split("=")[0].strip()
    console.print(
        f"[bold cyan]📦 加载数据[/bold cyan] pool={pool.name} | "
        f"{start_date} ~ {end_date or '最新'} | expr={expr!r}"
    )

    dp = DataProvider()
    lf = dp.load_pool_data(pool.value(), start_date, end_date, exprs=[expr])

    # --- 3. 回测执行 ---
    if mode_lower == "period":
        console.print(
            f"[bold cyan]🚀 启动周期换股回测[/bold cyan] | "
            f"因子={factor_col} | 持股={n_buy} | 换仓周期={rebalance_period}天 | "
            f"费率={cost:.4f} | 执行价={exec_price_col}"
        )
        result = backtest_periodic_rebalance(
            df_input=lf,
            factor_col=factor_col,
            hold_num=n_buy,
            rebalance_period=rebalance_period,
            cost_rate=cost,
            exec_price=exec_price_col,
            ascending=ascending,
        )
    else:
        console.print(
            f"[bold cyan]🚀 启动逐日演进回测[/bold cyan] | "
            f"因子={factor_col} | 持仓={n_buy} | 卖出线={sell_rank} | "
            f"费率={cost:.4f} | 执行价={exec_price_col}"
        )
        result = backtest_daily_evolving(
            df_input=lf,
            factor_col=factor_col,
            n_buy=n_buy,
            sell_rank=sell_rank,
            cost_rate=cost,
            exec_price=exec_price_col,
            ascending=ascending,
        )

    daily_df = result["daily_results"]
    trade_df = result["trade_details"]

    # --- 4. 摘要统计 ---
    _print_summary(daily_df, trade_df, factor_col)

    # --- 4b. 保存交易明细 ---
    if save_trades is not None:
        _save_trades(trade_df, save_trades)

    # --- 5. HTML 报告 ---
    if report:
        # generate_and_open_report 期望 result["series"] 且列名小写
        series = daily_df.rename(
            {
                "RAW_RET": "raw_ret",
                "NET_RET": "net_ret",
                "TURNOVER": "turnover",
                "COUNT": "count",
                "NAV": "nav",
                F.DATE: "DATE",
            }
        )
        try:
            generate_and_open_report({"series": series}, factor_col)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]⚠ 报告生成失败: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _print_summary(daily_df, trade_df, factor_col: str) -> None:
    """在终端打印回测关键指标表格。"""
    nav_series = daily_df["NAV"]
    net_ret_series = daily_df["NET_RET"]

    total_ret = float(nav_series[-1]) - 1.0

    # 年化收益率：假设每年 252 个交易日
    n_days = len(daily_df)
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0.0

    # 最大回撤
    rolling_max = nav_series.cum_max()
    drawdown = (nav_series - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # 年化波动率
    ann_vol = float(net_ret_series.std() or 0.0) * (252**0.5)

    # 夏普比率（无风险利率取 0）
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    # Calmar 比率
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # 年化换手率 = 平均日换手率 × 252
    avg_turnover = float(daily_df["TURNOVER"].mean() or 0.0)
    ann_turnover = avg_turnover * 252

    # ---- 交易层面统计 ----
    n_trades = len(trade_df)
    if n_trades > 0:
        pnl = trade_df["pnl_ret"]

        avg_pnl = float(pnl.mean() or 0.0)
        avg_hold = float(trade_df["holding_periods"].mean() or 0.0)

        wins = pnl.filter(pnl > 0)
        losses = pnl.filter(pnl <= 0)

        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        # 盈亏比：平均盈利 / 平均亏损绝对值
        profit_factor = (
            avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else float("inf")
        )
    else:
        avg_pnl = avg_hold = win_rate = avg_win = avg_loss = profit_factor = 0.0

    # ---- 构建表格（分组显示） ----
    table = Table(
        title=f"回测摘要 · {factor_col}",
        show_header=True,
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("指标", style="cyan", no_wrap=True, min_width=18)
    table.add_column("数值", justify="right", min_width=12)

    # 整体表现
    table.add_row("[bold]── 整体表现 ──[/bold]", "")
    table.add_row("交易天数", str(n_days))
    table.add_row("总收益率", f"{total_ret:+.2%}")
    table.add_row("年化收益率", f"{ann_ret:+.2%}")
    table.add_row("年化波动率", f"{ann_vol:.2%}")
    table.add_row("夏普比率", f"{sharpe:.3f}")
    table.add_row("Calmar 比率", f"{calmar:.3f}")
    table.add_row("最大回撤", f"{max_dd:.2%}")

    # 换手
    table.add_row("[bold]── 换手 ──[/bold]", "")
    table.add_row("平均日换手率", f"{avg_turnover:.2%}")
    table.add_row("年化换手率", f"{ann_turnover:.1f}x")

    # 交易明细
    table.add_row("[bold]── 交易明细 ──[/bold]", "")
    table.add_row("成交笔数", str(n_trades))
    if n_trades > 0:
        table.add_row("平均持有天数", f"{avg_hold:.1f} 天")
        table.add_row("胜率", f"{win_rate:.2%}")
        table.add_row("平均单笔 PnL", f"{avg_pnl:+.2%}")
        table.add_row("正收益平均", f"{avg_win:+.2%}")
        table.add_row("负收益平均", f"{avg_loss:+.2%}")
        pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
        table.add_row("盈亏比", pf_str)
    else:
        table.add_row("平均持有天数", "N/A")
        table.add_row("胜率", "N/A")
        table.add_row("平均单笔 PnL", "N/A")
        table.add_row("正收益平均", "N/A")
        table.add_row("负收益平均", "N/A")
        table.add_row("盈亏比", "N/A")

    console.print(table)


def _save_trades(trade_df, path: Path) -> None:
    """将交易明细保存到文件，按扩展名自动选择格式（csv / parquet）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        trade_df.write_parquet(path)
        fmt = "Parquet"
    else:
        # 默认 CSV（包括 .csv 或无扩展名等情况）
        if suffix not in (".csv",):
            path = path.with_suffix(".csv")
        trade_df.write_csv(path)
        fmt = "CSV"

    n = len(trade_df)
    console.print(
        f"[bold green]💾 交易明细已保存[/bold green] → {path}  ({fmt}, {n} 条记录)"
    )
