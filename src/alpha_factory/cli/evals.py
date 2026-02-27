"""quant evals —— 批量因子评估命令。

支持两种输入方式：
  1. --expr "factor1=ts_mean(AMOUNT,40)" --expr "factor2=rank(CLOSE)"  （可重复传入）
  2. --csv-file factors.csv --name-col name --expr-col expression       （CSV 文件）
两种方式可同时使用，结果合并去重（以因子名为 key）。

输出：Rich 终端表格（前 --top-n 行） + 可选 --output CSV 落盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.batch.full_metrics import batch_full_metrics

console = Console()


def _parse_exprs(raw: List[str]) -> list[tuple[str, str]]:
    """将 'name=expression' 字符串列表解析为 (name, expression) 对。

    若某条不含 '='，自动以 factor_{i} 命名。
    """
    result: list[tuple[str, str]] = []
    for i, item in enumerate(raw):
        if "=" in item:
            name, expr = item.split("=", 1)
            result.append((name.strip(), expr.strip()))
        else:
            result.append((f"factor_{i}", item.strip()))
    return result


def _load_from_csv(
    csv_path: Path,
    name_col: str,
    expr_col: str,
) -> list[tuple[str, str]]:
    """从 CSV 文件读取因子名与表达式列表。"""
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        console.print(f"[red]❌ 读取 CSV 文件失败: {e}[/red]")
        raise typer.Exit(code=1)

    if name_col not in df.columns:
        console.print(
            f"[red]❌ CSV 中未找到列 '{name_col}'，可用列: {df.columns}[/red]"
        )
        raise typer.Exit(code=1)
    if expr_col not in df.columns:
        console.print(
            f"[red]❌ CSV 中未找到列 '{expr_col}'，可用列: {df.columns}[/red]"
        )
        raise typer.Exit(code=1)

    return [
        (row[name_col], row[expr_col])
        for row in df.select([name_col, expr_col]).to_dicts()
    ]


def _print_rich_table(result_df: pl.DataFrame, top_n: int) -> None:
    """用 Rich 打印批量评估结果表格。"""
    display_cols = [
        "factor",
        "expression",
        "ic_mean",
        "ic_ir",
        "ann_ret",
        "sharpe",
        "turnover_est",
        "direction",
    ]
    cols = [c for c in display_cols if c in result_df.columns]
    subset = result_df.head(top_n).select(cols)

    table = Table(
        title=f"批量因子评估结果（Top {min(top_n, len(result_df))}，按 sharpe 降序）"
    )
    for col in cols:
        if col == "factor":
            table.add_column(col, style="cyan", no_wrap=True)
        elif col == "expression":
            table.add_column(col, style="dim", overflow="fold", min_width=20)
        else:
            table.add_column(col, style="white")

    for row in subset.to_dicts():
        table.add_row(
            *[
                (
                    f"{v:.4f}"
                    if isinstance(v, float)
                    else (str(int(v)) if isinstance(v, int) else str(v))
                )
                for v in row.values()
            ]
        )
    console.print(table)


def _diagnose_exprs(
    exprs: list[str],
    start_date: str,
    end_date: Optional[str],
    pool: PoolUniverseEnum,
) -> None:
    """逐条测试表达式，识别并打印失败的条目，方便用户定位语法错误。"""
    if len(exprs) <= 1:
        return  # 只有一条时原始错误已经够清楚

    console.print("[yellow]正在逐条检验表达式，定位错误...[/yellow]")
    dp = DataProvider()
    # 使用时间窗口较小的日期范围加速检验
    diag_end = end_date
    bad_exprs: list[str] = []
    for expr_str in exprs:
        try:
            dp.load_pool_data(pool.value(), start_date, diag_end, exprs=[expr_str])
        except Exception as err:
            name = expr_str.split("=")[0].strip() if "=" in expr_str else expr_str
            console.print(f"  [red]✗ {name!r} 表达式无效: {err}[/red]")
            bad_exprs.append(expr_str)
        else:
            name = expr_str.split("=")[0].strip() if "=" in expr_str else expr_str
            console.print(f"  [green]✓ {name!r} 表达式有效[/green]")

    if bad_exprs:
        console.print(
            f"\n[bold red]共 {len(bad_exprs)} 条表达式无效，请修正后重试。[/bold red]"
        )
        console.print(
            "[dim]提示：常用函数包括 "
            "ts_mean / ts_sum / ts_std_dev / ts_rank / ts_delta / ts_delay / "
            "ts_min / ts_max / ts_corr / ts_arg_max / cs_rank / "
            "if_else / sign / abs_ / log / signed_power。\n"
            "  ✗ rank(CLOSE)      → 不支持裸 rank，改用 cs_rank(CLOSE) 或 ts_rank(CLOSE, n)\n"
            "  ✓ cs_rank(CLOSE)   → 截面排名\n"
            "  ✓ ts_rank(CLOSE,5) → 时序排名（最近5日）[/dim]"
        )


def quant_evals(
    start_date: str = typer.Option(..., "-s", "--start-date", help="开始日期 YYYYMMDD"),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="结束日期 YYYYMMDD"
    ),
    expr: Optional[List[str]] = typer.Option(
        None,
        "--expr",
        help="因子表达式，格式: 'factor1=ts_mean(AMOUNT,40)'，可重复传入多次",
    ),
    csv_file: Optional[Path] = typer.Option(
        None, "--csv-file", help="CSV 文件路径，包含因子名与表达式列"
    ),
    name_col: str = typer.Option("name", "--name-col", help="CSV 中因子名所在列名"),
    expr_col: str = typer.Option(
        "expression", "--expr-col", help="CSV 中表达式所在列名"
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small, "--pool", help="股票池"
    ),
    mode: str = typer.Option(
        "long_only", "--mode", help="评估模式: long_only|long_short|active"
    ),
    n_bins: int = typer.Option(10, "--n-bins", help="分层数量"),
    fee: float = typer.Option(0.0025, "--fee", help="单边交易费率"),
    top_n: int = typer.Option(20, "--top-n", help="终端展示前 N 条"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="将完整结果写入 CSV 文件"
    ),
):
    """
    批量因子评估（多因子并行）。

    示例（命令行表达式）：

      quant evals -s 20220101 \\
        --expr "factor1=ts_mean(AMOUNT,40)" \\
        --expr "factor2=rank(CLOSE)"

    示例（CSV 文件）：

      quant evals -s 20220101 --csv-file factors.csv
    """
    # ── 1. 收集所有因子对 ────────────────────────────────────────────────────
    factor_pairs: list[tuple[str, str]] = []

    if expr:
        factor_pairs.extend(_parse_exprs(list(expr)))

    if csv_file:
        factor_pairs.extend(_load_from_csv(csv_file, name_col, expr_col))

    if not factor_pairs:
        console.print(
            "[red]❌ 请通过 --expr 或 --csv-file 至少提供一个因子表达式。[/red]"
        )
        raise typer.Exit(code=1)

    # 去重（保留最后出现的）
    seen: dict[str, str] = {}
    for name, expression in factor_pairs:
        seen[name] = expression
    factor_pairs = list(seen.items())

    factor_names = [name for name, _ in factor_pairs]
    # DataProvider 的 exprs 接受 'name=expr' 格式字符串
    exprs_for_loader = [f"{name}={expression}" for name, expression in factor_pairs]

    console.print(
        f"[bold]准备评估 {len(factor_pairs)} 个因子: {', '.join(factor_names)}[/bold]"
    )
    console.print(
        f"  pool={pool.name}  start={start_date}  end={end_date or '最新'}  mode={mode}"
    )

    # ── 2. 加载数据 ──────────────────────────────────────────────────────────
    try:
        dp = DataProvider()
        lf = dp.load_pool_data(
            pool.value(), start_date, end_date, exprs=exprs_for_loader
        )
    except Exception as e:
        console.print(f"[red]❌ 数据加载失败: {e}[/red]")
        # 逐条检验，找出哪条表达式有问题
        _diagnose_exprs(exprs_for_loader, start_date, end_date, pool)
        raise typer.Exit(code=1)

    console.print("✅ 数据加载完成 — 开始批量评估")

    # ── 3. 批量评估 ──────────────────────────────────────────────────────────
    try:
        result_df = batch_full_metrics(
            lf,
            factors=factor_names,
            n_bins=n_bins,
            mode=mode,
            fee=fee,
        )
    except Exception as e:
        console.print(f"[red]❌ 评估执行失败: {e}[/red]")
        raise typer.Exit(code=1)

    if result_df.is_empty():
        console.print("[yellow]⚠️ 评估结果为空，请检查因子表达式或数据范围。[/yellow]")
        raise typer.Exit(code=0)

    # 将表达式附加到结果中，方便对照
    expr_map = pl.DataFrame(
        {"factor": factor_names, "expression": [e for _, e in factor_pairs]}
    )
    result_df = result_df.join(expr_map, on="factor", how="left").select(
        ["factor", "expression", *[c for c in result_df.columns if c != "factor"]]
    )

    # ── 4. 输出 ──────────────────────────────────────────────────────────────
    _print_rich_table(result_df, top_n)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_csv(output)
        console.print(f"[green]✅ 完整结果已写入: {output}[/green]")

    console.print(f"✅ 批量评估完成，共 {len(result_df)} 个因子")


__all__ = ["quant_evals"]
