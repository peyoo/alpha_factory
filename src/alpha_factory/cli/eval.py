from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.single.single import single_factor_alpha_analysis
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.utils.schema import F

console = Console()


def quant_eval(
    start_date: str = typer.Option(..., "-s", "--start-date", help="开始日期 YYYYMMDD"),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="结束日期 YYYYMMDD"
    ),
    expr: str = typer.Option(
        ...,
        "--expr",
        help="用于生成或选择的因子表达式，格式示例：FACTOR = CLOSE.rolling_mean(5)",
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="是否生成并打开 HTML 报告"
    ),
    mode: str = typer.Option(
        "long_only", "--mode", help="评估模式: long_only|long_short|active"
    ),
    n_bins: int = typer.Option(
        10, "--n-bins", help="分层数量（分位数，默认10与evals一致）"
    ),
    period: int = typer.Option(1, "--period", help="调仓周期（交易日）"),
    cost: float = typer.Option(
        0.0025, "--cost", help="单边交易成本率（默认0.0025与evals一致）"
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small, "--pool", help="股票池"
    ),
):
    """
    单因子评估（简化入口）。目前仅支持 MainSmallPool。
    数据通过 DataProvider.load_pool_data 获取。
    """
    typer.echo(
        f"准备加载池数据: pool={pool}, start_date={start_date}, end_date={end_date}"
    )
    dp = DataProvider()
    # 从 expr 推断因子列名：
    #   - 有 "=" 时，左侧为列名（如 "MY_FACTOR = CLOSE/VWAP"）
    #   - 无 "=" 时，自动命名为 factor_0，并改写为 "factor_0=expr" 传给 DataProvider
    #     这与 evals 的行为一致：DataProvider 需要 "name=expression" 格式才能生成命名列
    if "=" in expr:
        factor_col = expr.split("=")[0].strip()
        expr_for_loader = expr
    else:
        factor_col = "factor_0"
        expr_for_loader = f"factor_0={expr}"

    lf = dp.load_pool_data(pool.value(), start_date, end_date, exprs=[expr_for_loader])

    typer.echo("✅ 数据加载完成 — 开始评估")

    try:
        # 传入 LazyFrame（函数支持 LazyFrame）以便下压优化
        result = single_factor_alpha_analysis(
            lf,
            factor_col=factor_col,
            ret_col=F.LABEL_FOR_RET_CC,
            date_col=F.DATE,
            asset_col=F.ASSET,
            pool_mask_col=F.POOL_MASK,
            n_bins=n_bins,
            mode=mode,
            period=period,
            cost=cost,
        )
        if report:
            try:
                # 兼容旧的 report 接口，期待 key 为 'series'
                report_result = result
                if "series" not in report_result and "nav" in report_result:
                    report_result = dict(report_result)
                    report_result["series"] = report_result.get("nav")

                generate_and_open_report(report_result, factor_col)
            except Exception as e:
                console.print(f"[yellow]报告生成失败: {e}[/yellow]")

    except Exception as e:
        from rich.markup import escape

        console.print(f"[red]评估执行失败:[/red] {escape(str(e))}")
        raise typer.Exit(code=1)

    typer.echo("✅ 评估完成")


__all__ = ["quant_eval"]
