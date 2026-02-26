from __future__ import annotations

from enum import Enum
from typing import Optional

import typer
from rich.console import Console

from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.evaluation.single.quantile_metric import single_calc_quantile_metrics
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.utils.schema import F

console = Console()


class PoolUniverseEnum(Enum):
    main_small = MainSmallPool


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
    report: bool = typer.Option(False, "--report", help="是否生成并打开 HTML 报告"),
    mode: str = typer.Option(
        "long_only", "--mode", help="评估模式: long_only|long_short|active"
    ),
    n_bins: int = typer.Option(5, "--n-bins", help="分层数量（分位数）"),
    period: int = typer.Option(1, "--period", help="调仓周期（交易日）"),
    cost: float = typer.Option(0.0015, "--cost", help="单边交易成本率"),
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
    # 通过 expr 让 DataProvider 生成因子列（若 expr 为赋值语句，左侧为列名）
    lf = dp.load_pool_data(pool.value(), start_date, end_date, exprs=[expr])

    try:
        # 仅做快速预览以验证加载成功
        df_preview = lf.select(["DATE"]).limit(3).collect()
        typer.echo(f"加载预览成功，示例行数: {df_preview.shape[0]}")
    except Exception as e:
        console.print(f"[red]加载数据失败: {e}[/red]")
        raise typer.Exit(code=1)

    typer.echo("✅ 数据加载完成 — 开始评估")

    # 从 expr 中推断因子列名（若为赋值形式）
    factor_col = expr.split("=")[0].strip() if "=" in expr else expr.strip()

    try:
        # 传入 LazyFrame（函数支持 LazyFrame）以便下压优化
        result = single_calc_quantile_metrics(
            lf,
            factor_col=factor_col,
            ret_col=F.LABEL_FOR_RET,
            date_col=F.DATE,
            asset_col=F.ASSET,
            pool_mask_col=F.POOL_MASK,
            n_bins=n_bins,
            mode=mode,
            period=period,
            cost=cost,
        )

        metrics = result.get("metrics") or {}

        console.print("**评估摘要**")
        console.print(
            f"年化收益: {metrics.get('annual_return'):.4f}"
            if metrics.get("annual_return") is not None
            else "年化收益: -"
        )
        console.print(
            f"夏普比率: {metrics.get('sharpe_ratio'):.4f}"
            if metrics.get("sharpe_ratio") is not None
            else "夏普比率: -"
        )
        console.print(
            f"最大回撤: {metrics.get('max_drawdown'):.4f}"
            if metrics.get("max_drawdown") is not None
            else "最大回撤: -"
        )
        console.print(
            f"估算日均换手率: {metrics.get('avg_daily_turnover'):.4f}"
            if metrics.get("avg_daily_turnover") is not None
            else "估算日均换手率: -"
        )
        console.print("IC 均值 / ICIR: (暂未计算在此处) ")

        if report:
            try:
                generate_and_open_report(result, factor_col)
            except Exception as e:
                console.print(f"[yellow]报告生成失败: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]评估执行失败: {e}[/red]")
        raise typer.Exit(code=1)

    typer.echo("✅ 评估完成")


__all__ = ["quant_eval"]
