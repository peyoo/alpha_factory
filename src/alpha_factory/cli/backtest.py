from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.cli.utils import resolve_yaml_path
from alpha_factory.utils.schema import F

console = Console()

# 执行价字段映射：CLI 友好字符串 → schema 常量
_EXEC_PRICE_MAP: dict[str, str] = {
    "open": F.OPEN,
    "close": F.CLOSE,
    "vwap": F.VWAP,
}


def quant_bt(
    yaml_file: Path = typer.Option(
        ...,
        "-y",
        "--yaml",
        help="策略 YAML 文件路径（StrategyConfig 格式）",
    ),
    start_date: str = typer.Option(
        "20190101", "-s", "--start-date", help="开始日期 YYYYMMDD"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "--end-date", help="结束日期 YYYYMMDD（默认至最新）"
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="是否生成 HTML 报告并打开"
    ),
    save_trades: Optional[Path] = typer.Option(
        None,
        "--save-trades",
        help="保存交易明细到文件，按扩展名自动选格式（.csv 或 .parquet）",
        show_default=False,
    ),
):
    """
    逐日演进因子回测（T+1 精准收益闭环）。
    所有回测参数（因子/股票池/持仓数/卖出线/成本/执行价）均来自 YAML 文件。

    \b
    示例:
      quant bt -y output/main_small_pool/s1.yaml
      quant bt -y output/main_small_pool/s1.yaml -s 20210101 --end 20241231
    """
    _run_bt_from_yaml(yaml_file, start_date, end_date, report, save_trades)


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


# ---------------------------------------------------------------------------
# YAML 策略回测
# ---------------------------------------------------------------------------


def _run_bt_from_yaml(
    yaml_file: Path,
    start_date: str,
    end_date: Optional[str],
    report: bool,
    save_trades: Optional[Path],
) -> None:
    """从 StrategyConfig YAML 加载策略并执行逐日演进回测。

    单因子：直接加载原始表达式，direction 控制 ascending 。
    多因子：截面 rank 预计算 + softmax 加权合成。
    """
    from alpha_factory.cli.opt import _resolve_pool

    yaml_file = resolve_yaml_path(yaml_file)

    if not yaml_file.exists():
        typer.echo(f"❌ YAML 文件不存在: {yaml_file}", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = StrategyConfig.from_yaml(yaml_file)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"❌ 加载策略配置失败: {exc}", err=True)
        raise typer.Exit(code=1)

    if not cfg.ranks:
        typer.echo("❌ YAML ranks 列表为空，无法执行回测", err=True)
        raise typer.Exit(code=1)

    try:
        pool_instance = _resolve_pool(cfg.pool)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1)

    exec_price_col = _EXEC_PRICE_MAP.get(cfg.exe_price.strip().lower(), F.VWAP)

    console.print(
        f"[bold cyan]📦 加载策略[/bold cyan] "
        f"name={cfg.name!r} | pool={cfg.pool} | 因子={len(cfg.ranks)} | "
        f"hold={cfg.hold_num} | sell_rank={cfg.sell_rank} | "
        f"{start_date} ~ {end_date or '最新'}"
    )

    dp = DataProvider()

    # 统一的数据加载 - 合并因子表达式和过滤表达式，一起计算
    all_exprs = cfg.factor_exprs + cfg.get_filter_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        start_date,
        end_date,
        exprs=all_exprs,
        actions=cfg.build_actions(),
    )
    df = lf.collect()

    # 确定因子列名和方向（单/多因子自动判断）
    if len(cfg.ranks) == 1:
        from alpha_factory.cli.opt import _COMPOSITE_COL

        factor_col = cfg.ranks[0].name
        mode_label = "单因子"
    else:
        from alpha_factory.cli.opt import _COMPOSITE_COL

        factor_col = _COMPOSITE_COL
        mode_label = f"多因子 ({len(cfg.ranks)} 因子)"

    console.print(
        f"[bold cyan]🚀 逐日演进回测 ({mode_label})[/bold cyan] | "
        f"因子={factor_col} | 持仓={cfg.hold_num} | 卖出线={cfg.sell_rank} | "
        f"费率={cfg.cost:.4f} | 执行价={cfg.exe_price}"
    )

    result = backtest_quick_daily(
        df_input=df,
        factor_col=factor_col,
        n_buy=cfg.hold_num,
        sell_rank=cfg.sell_rank,
        cost_rate=cfg.cost,
        exec_price=exec_price_col,
        # 方向已在表达式生成时统一处理，backtest 层级统一使用 ascending=False
        ascending=False,
    )

    daily_df = result["daily_results"]
    trade_df = result["trade_details"]

    factor_label = cfg.name if len(cfg.ranks) > 1 else cfg.ranks[0].name
    _print_summary(daily_df, trade_df, factor_label)

    if save_trades is not None:
        _save_trades(trade_df, save_trades)

    if report:
        series = daily_df.rename(
            {
                "NET_RET": "net_ret",
                "TURNOVER": "turnover",
                "COUNT": "count",
                "NAV": "nav",
                F.DATE: "DATE",
            }
        )
        try:
            generate_and_open_report({"series": series}, factor_label)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]⚠ 报告生成失败: {exc}[/yellow]")


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
