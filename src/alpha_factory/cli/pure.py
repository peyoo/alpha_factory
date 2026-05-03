"""pure.py — 纯净因子回测命令

计算给定因子相对于策略配置中 rank factors 的横截面残差（纯净因子），
然后以该残差作为唯一因子，使用与 `quant bt` 完全相同的回测引擎计算收益。

示例:
  quant pure -y output/main_small_pool/s1.yaml --expr CLOSE/OPEN
  quant pure -y output/main_small_pool/s1.yaml --expr my_f=ts_mean(AMOUNT,40)
  quant pure -y output/main_small_pool/s1.yaml --expr "my_f=ts_mean(AMOUNT,40)" -s 20210101
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import polars_ols as pls
import typer
from polars_ols.least_squares import OLSKwargs
from rich.console import Console

from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.cli.utils import resolve_yaml_path
from alpha_factory.utils.schema import F

console = Console()

_ols_kwargs = OLSKwargs(null_policy="drop", solve_method="svd")

# 执行价字段映射（与 backtest.py 保持一致）
_EXEC_PRICE_MAP: dict[str, str] = {
    "open": F.OPEN,
    "close": F.CLOSE,
    "vwap": F.VWAP,
}


def _parse_expr(expr_str: str) -> tuple[str, str]:
    """解析 --expr 参数。

    含 `=` 时：左侧为因子名，右侧为表达式。
    不含 `=` 时：因子名默认为 ``pure_f1``，整个字符串为表达式。
    """
    if "=" in expr_str:
        name, _, expression = expr_str.partition("=")
        return name.strip(), expression.strip()
    return "pure_f1", expr_str.strip()


def _compute_pure_factor(
    df: pl.DataFrame,
    target_col: str,
    regressor_cols: list[str],
    pure_col: str,
) -> pl.DataFrame:
    """横截面 OLS 回归，求 target 相对于 regressors 的残差。

    只在 POOL_MASK=True 的截面内参与回归（与 pre_processor.py 保持一致）；
    池外股票残差置为 None。

    参数
    ----
    df:             原始 DataFrame，已含 target_col / regressor_cols / POOL_MASK / DATE
    target_col:     待纯净化的因子列名
    regressor_cols: 用于回归的控制因子列名列表（cfg.ranks 的列名）
    pure_col:       残差列输出名
    """
    regressor_exprs = [pl.col(c) for c in regressor_cols]

    residual_expr = (
        pl.when(pl.col(F.POOL_MASK))
        .then(
            pls.compute_least_squares(
                pl.col(target_col),
                *regressor_exprs,
                mode="residuals",
                ols_kwargs=_ols_kwargs,
            )
        )
        .otherwise(None)
        .over(F.DATE)
        .alias(pure_col)
    )

    return df.with_columns(residual_expr)


def quant_pure(
    yaml_file: Path = typer.Option(
        ...,
        "-y",
        "--yaml",
        help="策略 YAML 文件路径（StrategyConfig 格式），提供股票池/持仓数/成本等参数",
    ),
    expr: str = typer.Option(
        ...,
        "--expr",
        help=(
            "因子表达式，格式为 `name=expression` 或纯表达式。"
            "含 `=` 时左侧为列名，否则默认命名为 `pure_f1`。"
        ),
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
        help="保存交易明细到文件（.csv 或 .parquet）",
        show_default=False,
    ),
):
    """
    纯净因子回测：计算 `--expr` 因子相对于策略 rank factors 的残差，作为唯一因子回测。

    使用横截面 OLS 逐日去除策略中已有 rank factors 的影响，
    评估剩余收益能力（与 `quant bt` 使用相同的 T+1 精准收益引擎）。

    \b
    示例:
      quant pure -y s1.yaml --expr CLOSE/OPEN
      quant pure -y s1.yaml --expr "my_f=ts_mean(AMOUNT,40)" -s 20210101
    """
    from alpha_factory.cli._loader import resolve_pool

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
        typer.echo(
            "❌ YAML ranks 列表为空，纯净因子需要至少一个 rank factor 进行残差回归",
            err=True,
        )
        raise typer.Exit(code=1)

    factor_name, factor_expression = _parse_expr(expr)
    pure_col = factor_name

    try:
        pool_instance = resolve_pool(cfg.pool)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1)

    exec_price_col = _EXEC_PRICE_MAP.get(cfg.exe_price.strip().lower(), F.VWAP)

    rank_names = [r.name for r in cfg.ranks]

    console.print(
        f"[bold cyan]📦 纯净因子回测[/bold cyan] "
        f"name={cfg.name!r} | pool={cfg.pool} | "
        f"控制因子={rank_names} | 目标因子={factor_name!r} | "
        f"{start_date} ~ {end_date or '最新'}"
    )

    # 构建所有需要计算的表达式：rank factors + conditions + 目标因子
    all_exprs = (
        cfg.ranked_factor_exprs
        + cfg.get_condition_exprs()
        + [f"{factor_name} = {factor_expression}"]
    )

    dp = DataProvider()
    # 不传 actions，保留 rank factor 原始列用于 OLS 回归
    lf = dp.load_pool_data(pool_instance, start_date, end_date, exprs=all_exprs)
    df = lf.collect()

    console.print(
        f"[bold cyan]🔬 计算残差[/bold cyan] {factor_name!r} ~ {' + '.join(rank_names)}"
    )

    df = _compute_pure_factor(df, factor_name, rank_names, pure_col)

    console.print(
        f"[bold cyan]🚀 逐日演进回测（纯净因子）[/bold cyan] | "
        f"因子={pure_col!r} | 持仓={cfg.hold_num} | 卖出线={cfg.sell_rank} | "
        f"费率={cfg.cost:.4f} | 执行价={cfg.exe_price}"
    )

    result = backtest_quick_daily(
        df_input=df,
        factor_col=pure_col,
        n_buy=cfg.hold_num,
        sell_rank=cfg.sell_rank,
        cost_rate=cfg.cost,
        exec_price=exec_price_col,
        ascending=False,
    )

    daily_df = result["daily_results"]
    trade_df = result["trade_details"]

    from alpha_factory.cli.backtest import _print_summary, _save_trades

    _print_summary(daily_df, trade_df, pure_col)

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
            generate_and_open_report({"series": series}, pure_col)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]⚠ 报告生成失败: {exc}[/yellow]")
