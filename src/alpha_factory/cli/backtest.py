from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli.eval import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.utils.schema import F

console = Console()

# 执行价字段映射：CLI 友好字符串 → schema 常量
_EXEC_PRICE_MAP: dict[str, str] = {
    "open": F.OPEN,
    "close": F.CLOSE,
}


def quant_bt(
    start_date: str = typer.Option(..., "-s", "--start-date", help="开始日期 YYYYMMDD"),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="结束日期 YYYYMMDD（默认至最新）"
    ),
    expr: str = typer.Option(
        ...,
        "--expr",
        help="因子表达式，格式: FACTOR_NAME = <polars-ta 表达式>，例如 F1 = CLOSE.rolling_mean(5)",
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small, "--pool", help="股票池"
    ),
    n_buy: int = typer.Option(10, "--n-buy", help="最大持仓数（买入线）"),
    sell_rank: int = typer.Option(
        30, "--sell-rank", help="卖出线：排名超过此值时触发卖出"
    ),
    cost: float = typer.Option(0.003, "--cost", help="单边交易成本率"),
    exec_price: str = typer.Option(
        "open",
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
):
    """
    逐日演进回测 (T+1 精准收益闭环)。

    \b
    示例:
      quant bt -s 20200101 -e 20231231 --expr "F1 = -ts_mean(AMOUNT, 10)"
      quant bt -s 20200101 --expr "MOM = CLOSE / CLOSE.shift(20) - 1" --n-buy 20 --sell-rank 50
    """
    # --- 1. 参数校验 ---
    exec_price_lower = exec_price.strip().lower()
    if exec_price_lower not in _EXEC_PRICE_MAP:
        typer.echo(
            f"❌ --exec-price 无效值 '{exec_price}'，仅支持: {list(_EXEC_PRICE_MAP.keys())}",
            err=True,
        )
        raise typer.Exit(code=1)
    exec_price_col = _EXEC_PRICE_MAP[exec_price_lower]

    if sell_rank < n_buy:
        typer.echo(
            f"❌ --sell-rank ({sell_rank}) 必须 >= --n-buy ({n_buy})",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- 2. 数据加载 ---
    factor_col = expr.split("=")[0].strip() if "=" in expr else expr.strip()
    console.print(
        f"[bold cyan]📦 加载数据[/bold cyan] pool={pool.name} | "
        f"{start_date} ~ {end_date or '最新'} | expr={expr!r}"
    )

    dp = DataProvider()
    lf = dp.load_pool_data(pool.value(), start_date, end_date, exprs=[expr])

    # --- 3. 回测执行 ---
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

    # 平均每日换手率
    avg_turnover = float(daily_df["TURNOVER"].mean() or 0.0)

    # 交易笔数
    n_trades = len(trade_df)
    avg_pnl = float(trade_df["pnl_ret"].mean() or 0.0) if n_trades > 0 else 0.0

    table = Table(
        title=f"回测摘要 · {factor_col}", show_header=True, header_style="bold magenta"
    )
    table.add_column("指标", style="cyan", no_wrap=True)
    table.add_column("数值", justify="right")

    table.add_row("交易天数", str(n_days))
    table.add_row("总收益率", f"{total_ret:+.2%}")
    table.add_row("年化收益率", f"{ann_ret:+.2%}")
    table.add_row("年化波动率", f"{ann_vol:.2%}")
    table.add_row("夏普比率", f"{sharpe:.3f}")
    table.add_row("最大回撤", f"{max_dd:.2%}")
    table.add_row("平均日换手率", f"{avg_turnover:.2%}")
    table.add_row("成交笔数", str(n_trades))
    table.add_row("平均单笔 PnL", f"{avg_pnl:+.2%}" if n_trades > 0 else "N/A")

    console.print(table)
