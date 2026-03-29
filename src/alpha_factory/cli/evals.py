"""quant evals —— 批量因子评估命令

支持直接指定 --expr 表达式、--csv-file 文件，或自动扫描股票池目录。

示例用法:

    # 直接传入表达式评估
    quant evals -s 20220101 --expr "f1=ts_mean(AMOUNT,40)"

    # 从 CSV 文件批量评估
    quant evals -s 20220101 --csv-file factors.csv

    # 自动扫描 pool 目录，评估所有 CSV 因子
    quant evals -s 20220101
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.evaluation.batch.full_metrics import batch_full_metrics

console = Console()

_DEFAULT_N_BINS = 10
_DEFAULT_LS_MODE = "long_only"
_DEFAULT_FEE = 0.003
_DEFAULT_START_DATE = "20190101"


def _parse_expr(expr_str: str, idx: int) -> tuple[str, str]:
    """解析 'name=expression' 或 'expression'（自动命名为 factor_{idx}）。"""
    if "=" in expr_str:
        name, _, expr = expr_str.partition("=")
        return name.strip(), expr.strip()
    return f"factor_{idx}", expr_str.strip()


def _load_csv_factors(
    csv_path: Path,
    name_col: str = "factor",
    expr_col: str = "expression",
) -> list[tuple[str, str]]:
    """从 CSV 文件加载因子列表。"""
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        console.print(f"[red]❌ 读取 CSV 文件失败: {e}[/red]")
        raise typer.Exit(code=1)

    # 自动回退列名搜索（兼容历史 CSV）
    resolved_name_col = name_col
    if resolved_name_col not in df.columns:
        for alias in ["name", "factor_name", "因子名"]:
            if alias in df.columns:
                resolved_name_col = alias
                break

    if resolved_name_col not in df.columns:
        console.print(
            f"[red]❌ CSV 中未找到列 '{name_col}'，可用列: {df.columns}[/red]"
        )
        raise typer.Exit(code=1)

    if expr_col not in df.columns:
        console.print(
            f"[red]❌ CSV 中未找到列 '{expr_col}'，可用列: {df.columns}[/red]"
        )
        raise typer.Exit(code=1)

    factors = [
        (row[resolved_name_col], row[expr_col])
        for row in df.select([resolved_name_col, expr_col]).to_dicts()
    ]
    console.print(f"[dim]从 {csv_path.name} 加载 {len(factors)} 个因子[/dim]")
    return factors


def _scan_pool_dir(
    pool_dir: Path,
    name_col: str = "factor",
    expr_col: str = "expression",
) -> list[tuple[str, str]]:
    """扫描池目录中的所有 CSV，按因子名去重后返回列表。"""
    csv_files = sorted(pool_dir.glob("*.csv"))
    seen: dict[str, str] = {}
    for csv_file in csv_files:
        try:
            df = pl.read_csv(csv_file)
            actual_name = (
                name_col
                if name_col in df.columns
                else next(
                    (c for c in ["name", "factor_name", "因子名"] if c in df.columns),
                    None,
                )
            )
            actual_expr = (
                expr_col
                if expr_col in df.columns
                else next(
                    (c for c in ["expression", "expr", "公式"] if c in df.columns), None
                )
            )
            if actual_name and actual_expr:
                for row in df.select([actual_name, actual_expr]).to_dicts():
                    n, e = row[actual_name], row[actual_expr]
                    if n not in seen:
                        seen[n] = e
        except Exception:
            pass
    return list(seen.items())


def _print_result_table(result_df: pl.DataFrame, top_n: int = 20) -> None:
    """使用 Rich 打印评估结果表格。"""
    display_cols = [
        "factor",
        "ic_mean",
        "ic_ir",
        "ann_ret",
        "sharpe",
        "turnover_est",
        "direction",
    ]
    cols_to_show = [c for c in display_cols if c in result_df.columns]

    table = Table(title=f"批量因子评估结果（Top {min(top_n, len(result_df))}）")
    for col in cols_to_show:
        if col == "factor":
            table.add_column(col, style="cyan", no_wrap=True)
        else:
            table.add_column(col, style="white")

    for row in result_df.head(top_n).to_dicts():
        table.add_row(
            *[
                f"{row[col]:.4f}"
                if isinstance(row[col], float)
                else str(int(row[col]))
                if isinstance(row[col], int)
                else str(row[col])
                for col in cols_to_show
            ]
        )
    console.print(table)


def quant_evals(
    start_date: Optional[str] = typer.Option(
        None, "-s", "--start-date", help="开始日期（YYYYMMDD）"
    ),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="结束日期（YYYYMMDD）"
    ),
    expr: Optional[List[str]] = typer.Option(
        None, "--expr", help="因子表达式，格式 'name=expr' 或 'expr'，可重复"
    ),
    csv_file: Optional[Path] = typer.Option(None, "--csv-file", help="CSV 因子文件"),
    name_col: str = typer.Option("factor", "--name-col", help="CSV 中因子名列"),
    expr_col: str = typer.Option("expression", "--expr-col", help="CSV 中表达式列"),
    batch_size: int = typer.Option(100, "--batch-size", min=1, help="评估批大小"),
    top_n: int = typer.Option(20, "--top-n", help="显示前 N 条结果"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="输出 CSV 路径"),
    min_sharpe: float = typer.Option(1.0, "--min-sharpe", help="质量过滤：最低 Sharpe"),
    min_ann_ret: float = typer.Option(
        0.20, "--min-ann-ret", help="质量过滤：最低年化收益"
    ),
):
    """
    批量因子评估。

    支持 --expr 直接指定表达式、--csv-file 文件批量输入，或自动扫描股票池目录。

    [bold]示例 1[/bold] — 直接指定表达式：

      quant evals -s 20220101 --expr "f1=ts_mean(AMOUNT,40)"

    [bold]示例 2[/bold] — 从 CSV 批量输入：

      quant evals -s 20220101 --csv-file factors.csv

    [bold]示例 3[/bold] — 启用质量过滤，输出结果：

      quant evals --expr "f1=ts_mean(AMOUNT,40)" --min-sharpe 0.5 -o results.csv
    """
    t0 = perf_counter()

    # ── 1. 收集因子对 ──────────────────────────────────────────────────────
    factor_pairs: list[tuple[str, str]] = []
    auto_pool_mode = False

    if expr:
        for i, e in enumerate(expr):
            factor_pairs.append(_parse_expr(e, i))

    if csv_file:
        csv_factors = _load_csv_factors(csv_file, name_col, expr_col)
        factor_pairs.extend(csv_factors)

    if not factor_pairs:
        # 自动扫描池目录
        pool_dir = MainSmallPool().pool_dir
        if pool_dir.exists():
            console.print(f"[dim]扫描池目录: {pool_dir}[/dim]")
            factor_pairs = _scan_pool_dir(pool_dir, name_col, expr_col)
            auto_pool_mode = True

    if not factor_pairs:
        console.print(
            "[red]❌ 至少提供 --expr 或 --csv-file，或在 pool 目录放置 CSV 文件。[/red]"
        )
        raise typer.Exit(code=1)

    # 按因子名去重
    seen: dict[str, str] = {}
    for n, e in factor_pairs:
        if n not in seen:
            seen[n] = e
    factor_pairs = list(seen.items())

    console.print(f"[bold]准备评估 {len(factor_pairs)} 个因子[/bold]")

    # ── 2. 加载数据 ────────────────────────────────────────────────────────
    actual_start = start_date or _DEFAULT_START_DATE
    pool = MainSmallPool()
    dp = DataProvider()
    exprs_for_loader = [f"{n}={e}" for n, e in factor_pairs]
    lf = dp.load_pool_data(pool, actual_start, end_date, exprs=exprs_for_loader)

    # ── 3. 批量评估 ────────────────────────────────────────────────────────
    factor_names = [n for n, _ in factor_pairs]
    batches = [
        factor_names[i : i + batch_size]
        for i in range(0, len(factor_names), batch_size)
    ]
    result_parts: list[pl.DataFrame] = []
    for batch_factors in batches:
        part = batch_full_metrics(
            lf,
            factors=batch_factors,
            n_bins=_DEFAULT_N_BINS,
            mode=_DEFAULT_LS_MODE,
            fee=_DEFAULT_FEE,
        )
        if not part.is_empty():
            result_parts.append(part)

    result_df = (
        pl.concat(result_parts, how="vertical_relaxed")
        if result_parts
        else pl.DataFrame()
    )

    # ── 4. 质量过滤 ────────────────────────────────────────────────────────
    if not result_df.is_empty() and "sharpe" in result_df.columns:
        filtered = result_df.filter(
            (pl.col("sharpe") >= min_sharpe) & (pl.col("ann_ret") >= min_ann_ret)
        )
        if filtered.is_empty():
            console.print(
                f"[yellow]⚠️ 过滤后无可用因子（min_sharpe={min_sharpe}, "
                f"min_ann_ret={min_ann_ret}）[/yellow]"
            )
        result_df = filtered

    total_seconds = perf_counter() - t0
    console.print(f"[dim]时间统计: 总耗时 {total_seconds:.3f}s[/dim]")

    # ── 5. 展示结果 ────────────────────────────────────────────────────────
    if not result_df.is_empty():
        _print_result_table(result_df, top_n)

    # ── 6. 保存 --output ──────────────────────────────────────────────────
    if output and not result_df.is_empty():
        output.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_csv(output)
        console.print(f"[green]✅ 完整结果已写入: {output}[/green]")

    # ── 7. 自动池模式：落盘到 pool_dir ────────────────────────────────────
    if auto_pool_mode and not result_df.is_empty():
        pool_out = MainSmallPool().pool_dir / "main_small_pool.csv"
        result_df.write_csv(pool_out)
        console.print(f"[green]✅ 自动保存至: {pool_out}[/green]")

    console.print(f"[bold cyan]批量评估完成[/bold cyan] | 评估因子 {len(result_df)} 个")


__all__ = ["quant_evals"]
