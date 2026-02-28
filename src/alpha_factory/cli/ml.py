"""
quant ml - 因子筛选 CLI 命令（Lasso / Elastic Net）
====================================================

基于面板数据的正则化回归因子筛选：
  读取 evals CSV 中的因子表达式 -> DataProvider 加载面板数据 ->
  以因子值为特征、前瞻收益为目标进行 LassoCV/ElasticNetCV 回归 ->
  输出各因子系数，剔除系数为 0 的因子。

用法示例::

    quant ml --pool main_small --start-date 20240101
    quant ml --method elastic-net --l1-ratio 0.5 --start-date 20240101
    quant ml --csv /path/to/factors.csv --start-date 20230101 --end-date 20231231 --no-report
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.config.base import settings
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.ml.dim_reduction import (
    FactorSelectionResult,
    generate_selection_html_report,
    load_factor_csv,
    run_regularized_selection,
)

console = Console()

_METHOD_LASSO = "lasso"
_METHOD_ELASTIC_NET = "elastic-net"
_DEFAULT_TARGET = "LABEL_FOR_IC"


def quant_ml(
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        help=(
            "输入因子评估 CSV 文件路径（含 factor/expression 列）。"
            "未提供时自动推断为 --pool 对应目录下的同名 CSV。"
        ),
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small,
        "--pool",
        help="股票池。",
    ),
    method: str = typer.Option(
        _METHOD_ELASTIC_NET,
        "--method",
        help="筛选方法：elastic-net（默认）或 lasso。",
    ),
    target: str = typer.Option(
        _DEFAULT_TARGET,
        "--target",
        help="面板数据中的目标列名（默认 LABEL_FOR_IC，即前瞻收益）。",
    ),
    l1_ratio: float = typer.Option(
        0.5,
        "--l1-ratio",
        min=0.0,
        max=1.0,
        help="Elastic Net L1 正则化比例（0=Ridge, 1=Lasso；仅 elastic-net 时生效）。",
    ),
    start_date: str = typer.Option(
        "20190101",
        "--start-date",
        "-s",
        help="数据起始日期（YYYYMMDD，默认 20190101）。",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        "-e",
        help="数据结束日期（YYYYMMDD，默认今天）。",
    ),
    cv: int = typer.Option(
        5,
        "--cv",
        min=2,
        help="交叉验证折数。",
    ),
    max_samples: int = typer.Option(
        200_000,
        "--max-samples",
        help="回归最大样本行数（超过则随机抽样，0 = 不限制）。",
    ),
    report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="是否生成 HTML 可视化报告。",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="是否保存筛选结果到 CSV。",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="结果 CSV 的输出目录；默认为 <pool_dir>/ml/。",
    ),
):
    """
    [bold cyan]因子筛选（Lasso / Elastic Net）[/bold cyan]

    从 evals CSV 读取因子表达式，通过 DataProvider 加载面板数据，
    以因子值为特征、前瞻收益为目标进行正则化回归，筛选有效因子。

    使用 [cyan]--method elastic-net[/cyan] 启用 Elastic Net，
    通过 [cyan]--l1-ratio[/cyan] 控制 L1/L2 混合比例。
    """
    # -- 校验 method
    method = method.lower().strip()
    if method not in (_METHOD_LASSO, _METHOD_ELASTIC_NET):
        console.print(
            f"[red]--method={method!r} 不支持，可选：lasso / elastic-net[/red]"
        )
        raise typer.Exit(1)

    # -- 推断 CSV 路径
    pool_inst = pool.value()
    _default_out_base: Path
    if csv is None:
        csv = pool_inst.pool_dir / f"{pool_inst.name}.csv"
        _default_out_base = pool_inst.pool_dir
        console.print(f"[dim]--csv 未指定，自动推断路径：{csv}[/dim]")
    else:
        _default_out_base = Path(csv).parent

    csv = Path(csv)
    if not csv.exists():
        console.print(
            f"[red]CSV 文件不存在：{csv}\n"
            "    请先运行 quant evals 生成评估结果，"
            "或通过 --csv 指定正确路径。[/red]"
        )
        raise typer.Exit(1)

    # -- 输出目录
    if output_dir is None:
        output_dir = _default_out_base / "ml"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- 加载 evals CSV -> 提取因子表达式
    console.print(f"[cyan]加载 CSV：{csv}[/cyan]")
    evals_df = load_factor_csv(csv)
    console.print(
        f"[dim]   -> 读取到 {len(evals_df)} 行 x {len(evals_df.columns)} 列[/dim]"
    )

    if "factor" not in evals_df.columns or "expression" not in evals_df.columns:
        console.print("[red]CSV 必须包含 factor 和 expression 列。[/red]")
        raise typer.Exit(1)

    factor_pairs: list[tuple[str, str]] = [
        (row["factor"], row["expression"])
        for row in evals_df.select(["factor", "expression"]).to_dicts()
    ]
    factor_names = [name for name, _ in factor_pairs]
    expr_map = {name: expr for name, expr in factor_pairs}
    console.print(f"[dim]   -> 共 {len(factor_pairs)} 个因子表达式[/dim]")

    # -- DataProvider 加载面板数据
    console.print(
        f"[cyan]加载面板数据[/cyan]（pool={pool_inst.name}，"
        f"start={start_date}，end={end_date or '今天'}）"
    )
    exprs_for_loader = [f"{name}={expr}" for name, expr in factor_pairs]
    dp = DataProvider()
    try:
        lf = dp.load_pool_data(
            pool_inst,
            start_date,
            end_date,
            exprs=exprs_for_loader,
        )
        panel_df = lf.collect()
    except Exception as exc:
        console.print(f"[red]面板数据加载失败：{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[dim]   -> 面板数据：{len(panel_df):,} 行 x {len(panel_df.columns)} 列[/dim]"
    )

    # -- 执行正则化回归
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_label = "Lasso" if method == _METHOD_LASSO else "Elastic Net"
    console.print(
        f"\n[bold]执行 {method_label} 因子筛选[/bold]"
        f"（target={target}，factors={len(factor_names)}）"
    )

    try:
        result = run_regularized_selection(
            panel_df=panel_df,
            factor_cols=factor_names,
            target_col=target,
            method=method,  # type: ignore[arg-type]
            l1_ratio=l1_ratio,
            cv=cv,
            max_samples=max_samples,
        )
    except Exception as exc:
        console.print(f"[red]{method_label} 筛选失败：{exc}[/red]")
        raise typer.Exit(1)

    # 注入表达式映射
    result.expressions = expr_map

    # -- 终端打印
    _print_selection_summary(result)

    # -- 保存 CSV
    if save:
        method_tag = method.replace("-", "_")
        save_path = output_dir / f"ml_{method_tag}_{target}_{ts}.csv"
        rows = []
        for fname in result.factor_names:
            coef = result.factor_coefs[fname]
            rows.append(
                {
                    "factor": fname,
                    "expression": expr_map.get(fname, ""),
                    "coefficient": coef,
                    "selected": abs(coef) > 1e-10,
                }
            )
        pl.DataFrame(rows).write_csv(save_path)
        console.print(f"[green]筛选结果已保存：{save_path}[/green]")

    # -- HTML 报告
    if report:
        report_dir = Path(settings.OUTPUT_DIR) / "html_reports"
        method_tag = "Lasso" if method == _METHOD_LASSO else "ElasticNet"
        report_path = report_dir / f"ML_{method_tag}_{target}_{ts}.html"
        try:
            generate_selection_html_report(
                result,
                report_path,
                open_browser=True,
            )
            console.print(f"[green]HTML 报告已生成：{report_path}[/green]")
        except Exception as exc:
            console.print(f"[yellow]报告生成失败（不影响结果）：{exc}[/yellow]")


# ──────────────────────────────────────────────────────────────────────────────
# 终端打印辅助
# ──────────────────────────────────────────────────────────────────────────────


def _print_selection_summary(result: FactorSelectionResult) -> None:
    """在终端打印因子筛选结果摘要。"""
    method_label = "Lasso" if result.method == "lasso" else "Elastic Net"
    console.print()

    # -- 统计信息
    info_line = (
        f"  R2 = [yellow]{result.r2:.6f}[/yellow]，"
        f"alpha = [dim]{result.alpha:.6f}[/dim]"
    )
    if result.method == "elastic-net":
        info_line += f"，l1_ratio = [dim]{result.l1_ratio:.4f}[/dim]"
    info_line += f"，样本数 = [dim]{result.n_samples:,}[/dim]"
    console.print(info_line)

    n_sel = len(result.selected_factors)
    n_total = len(result.factor_names)
    console.print(
        f"[bold]{method_label} 筛选结果[/bold]：保留 {n_sel} / {n_total} 个因子"
        f"（剔除 {n_total - n_sel} 个）"
    )

    # -- 因子系数表格
    tbl = Table(
        title=f"{method_label} 因子系数（非零，按 |coef| 降序）",
        show_lines=True,
    )
    tbl.add_column("因子", style="cyan", width=12)
    tbl.add_column("系数", justify="right", width=14)
    tbl.add_column("表达式", style="dim", overflow="fold", min_width=30)

    for fname in result.selected_factors:
        coef = result.factor_coefs[fname]
        expr = result.expressions.get(fname, "")
        tbl.add_row(fname, f"{coef:+.6f}", expr)
    console.print(tbl)

    # -- 被剔除因子
    if result.eliminated_factors:
        elim_str = ", ".join(result.eliminated_factors[:20])
        if len(result.eliminated_factors) > 20:
            elim_str += f"... 等共 {len(result.eliminated_factors)} 个"
        console.print(f"\n[dim]被剔除因子：{elim_str}[/dim]")
