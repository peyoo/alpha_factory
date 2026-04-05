"""quant evals —— 批量因子评估命令

支持通过 --expr 或 --csv-file 直接提供因子表达式，
或在无输入时自动扫描 pool 目录中的 CSV 文件。

示例用法:

    # 直接评估表达式
    quant evals -s 20220101 --expr "factor1=ts_mean(AMOUNT,40)"

    # 从 CSV 文件加载因子
    quant evals -s 20220101 --csv-file factors.csv

    # 自动扫描 pool 目录，输出到指定文件
    quant evals -s 20220101 -o results.csv
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
from alpha_factory.evaluation.batch.overlap import batch_topn_overlap

console = Console()


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


def _parse_expr_args(exprs: list[str]) -> list[tuple[str, str]]:
    """解析 --expr 参数列表，支持 'name=expression' 或无名格式。"""
    result = []
    for i, raw in enumerate(exprs):
        if "=" in raw:
            idx = raw.index("=")
            name = raw[:idx].strip()
            expression = raw[idx + 1 :].strip()
        else:
            name = f"factor_{i}"
            expression = raw.strip()
        result.append((name, expression))
    return result


def _print_result_table(result_df: pl.DataFrame, top_n: int = 20) -> None:
    """使用 Rich 打印评估结果表格。

    Args:
        result_df: 评估结果 DataFrame
        top_n: 显示前 N 条结果
    """
    # 选择要显示的列
    # 主表全列展示（含 cluster 列）
    display_cols = [
        "factor",
        "expression",
        "ic_mean",
        "ic_ir",
        "ann_ret",
        "sharpe",
        "turnover_est",
        "direction",
        "cluster",
    ]
    cols_to_show = [c for c in display_cols if c in result_df.columns]

    # 检查是否有 IC Decay 列
    decay_cols = sorted(
        [c for c in result_df.columns if c.startswith(("IC_Mean_Lag", "IR_Lag"))]
    )
    decay_cols_with_data = [
        c for c in decay_cols if result_df[c].is_not_null().sum() > 0
    ]

    # 检查是否有 Turnover Decay 列
    turnover_decay_cols = [
        c for c in result_df.columns if c in ["avg_turnover", "turnover_std", "side"]
    ]

    # 如果有 IC Decay 或 Turnover Decay 列，使用简化显示
    if decay_cols_with_data or turnover_decay_cols:
        # 仅显示关键列，避免表格过宽
        cols_to_show = [
            c
            for c in cols_to_show
            if c in ["factor", "expression", "sharpe", "ic_ir", "ann_ret"]
        ]
    else:
        cols_to_show = [c for c in display_cols if c in result_df.columns]

    table = Table(title=f"批量因子评估结果（Top {min(top_n, len(result_df))}）")

    for col in cols_to_show:
        if col == "factor":
            table.add_column(col, style="cyan", no_wrap=True)
        elif col == "expression":
            table.add_column(col, style="dim", overflow="fold", min_width=20)
        else:
            table.add_column(col, style="white")

    for row in result_df.head(top_n).to_dicts():
        table.add_row(
            *[
                (
                    f"{row[col]:.4f}"
                    if isinstance(row[col], float)
                    else (
                        str(int(row[col]))
                        if isinstance(row[col], int)
                        else str(row[col])
                    )
                )
                for col in cols_to_show
            ]
        )

    console.print(table)

    # 如果有 IC Decay 列，拆成 IC 表和 IR 表单独显示
    if decay_cols_with_data:
        ic_cols = [c for c in decay_cols_with_data if c.startswith("IC_Mean_Lag")]
        ir_cols = [c for c in decay_cols_with_data if c.startswith("IR_Lag")]

        def _decay_row(row: dict, cols: list[str]) -> list[str]:
            values = [row["factor"]]
            for col in cols:
                val = row[col]
                if isinstance(val, float) and val == val:
                    values.append(f"{val:.4f}")
                else:
                    values.append("-")
            return values

        if ic_cols:
            console.print("\n[bold]IC Decay（IC Mean）：[/bold]")
            ic_table = Table(title="各因子在不同滞后期的 IC Mean")
            ic_table.add_column("factor", style="cyan", no_wrap=True, min_width=8)
            for col in ic_cols:
                lag_num = col.replace("IC_Mean_Lag_", "")
                ic_table.add_column(
                    f"IC_lag{lag_num}", style="white", min_width=8, no_wrap=True
                )
            for row in result_df.head(top_n).to_dicts():
                ic_table.add_row(*_decay_row(row, ic_cols))
            console.print(ic_table)

        if ir_cols:
            console.print("\n[bold]IC Decay（IR = IC Mean / IC Std）：[/bold]")
            ir_table = Table(title="各因子在不同滞后期的 IR")
            ir_table.add_column("factor", style="cyan", no_wrap=True, min_width=8)
            for col in ir_cols:
                lag_num = col.replace("IR_Lag_", "")
                ir_table.add_column(
                    f"IR_lag{lag_num}", style="white", min_width=8, no_wrap=True
                )
            for row in result_df.head(top_n).to_dicts():
                ir_table.add_row(*_decay_row(row, ir_cols))
            console.print(ir_table)

    # 如果有 Turnover Decay 列，单独显示
    if turnover_decay_cols:
        console.print("\n[bold]Turnover Decay 分析：[/bold]")
        turnover_table = Table(title="因子换手率与方向")
        turnover_table.add_column("factor", style="cyan", no_wrap=True)

        # 添加 turnover 相关列
        for col in ["side", "direction", "avg_turnover", "turnover_std"]:
            if col in result_df.columns:
                turnover_table.add_column(col, style="white")

        for row in result_df.head(top_n).to_dicts():
            values = [row["factor"]]
            for col in ["side", "direction", "avg_turnover", "turnover_std"]:
                if col in result_df.columns:
                    val = row[col]
                    if isinstance(val, float):
                        if val == val:  # 检查 NaN
                            values.append(f"{val:.4f}")
                        else:
                            values.append("-")
                    else:
                        values.append(str(val))
            turnover_table.add_row(*values)

        console.print(turnover_table)

    # 如果有聚类结果，显示簇摘要
    if "cluster" in result_df.columns:
        console.print("\n[bold]因子聚类结果：[/bold]")
        # 按 cluster 分组
        cluster_groups: dict[int, list[str]] = {}
        for row in result_df.to_dicts():
            cid = int(row["cluster"])
            cluster_groups.setdefault(cid, []).append(row["factor"])

        cluster_table = Table(title=f"共 {len(cluster_groups)} 个簇")
        cluster_table.add_column("簇 ID", style="cyan", no_wrap=True)
        cluster_table.add_column("因子数", style="white")
        cluster_table.add_column("因子列表", style="dim")
        for cid in sorted(cluster_groups.keys()):
            names = cluster_groups[cid]
            style = "bold yellow" if len(names) > 1 else "white"
            cluster_table.add_row(
                str(cid),
                str(len(names)),
                ", ".join(names),
                style=style,
            )
        console.print(cluster_table)


def _print_overlap_table(overlap_df: pl.DataFrame, topn: int) -> None:
    """使用 Rich 打印因子 Top-N 持仓重合度（Hit Rate）表格。"""
    if overlap_df.is_empty():
        console.print("[yellow]⚠️ 无可用重合度数据。[/yellow]")
        return

    table = Table(title=f"因子 Top-{topn} 持仓重合度（Hit Rate，按降序）")
    table.add_column("因子 A", style="cyan", no_wrap=True)
    table.add_column("因子 B", style="cyan", no_wrap=True)
    table.add_column("Hit Rate", style="bold")

    for row in overlap_df.to_dicts():
        rate = row["hit_rate"]
        if rate >= 0.5:
            rate_str = f"[bold red]{rate:.1%}[/bold red]"
        elif rate >= 0.3:
            rate_str = f"[yellow]{rate:.1%}[/yellow]"
        else:
            rate_str = f"[green]{rate:.1%}[/green]"
        table.add_row(row["factor_a"], row["factor_b"], rate_str)

    console.print(table)
    console.print(
        "[dim]颜色说明: [bold red]红色[/bold red] ≥50% 高重合 "
        "| [yellow]黄色[/yellow] 30~50% 中等 "
        "| [green]绿色[/green] <30% 低重合[/dim]"
    )


def _run_batched_eval(
    lf: pl.LazyFrame,
    factor_names: list[str],
    batch_size: int,
) -> pl.DataFrame:
    """分批调用 batch_full_metrics 并合并结果。"""
    batches = [
        factor_names[i : i + batch_size]
        for i in range(0, len(factor_names), batch_size)
    ]
    parts: list[pl.DataFrame] = []
    for batch in batches:
        part = batch_full_metrics(lf, factors=batch)
        if not part.is_empty():
            parts.append(part)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed")


def quant_evals(
    yaml_file: Optional[Path] = typer.Option(
        None,
        "-y",
        "--yaml",
        help="StrategyConfig YAML 文件路径（基于配置的完整评估模式）",
    ),
    expr: Optional[List[str]] = typer.Option(
        None, "--expr", help="因子表达式，支持 'name=expression' 格式，可重复"
    ),
    start_date: Optional[str] = typer.Option(
        None, "-s", "--start-date", help="开始日期（YYYYMMDD）"
    ),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="结束日期（YYYYMMDD）"
    ),
    csv_file: Optional[Path] = typer.Option(
        None, "--csv-file", help="CSV 因子文件路径（--yaml 模式下作为补充因子）"
    ),
    name_col: str = typer.Option("factor", "--name-col", help="CSV 中因子名所在列"),
    expr_col: str = typer.Option("expression", "--expr-col", help="CSV 中表达式所在列"),
    ic_decay: bool = typer.Option(
        False, "--ic-decay", help="计算 IC Decay（仅 --yaml 模式）"
    ),
    turnover_decay: bool = typer.Option(
        False, "--turnover-decay", help="计算 Turnover Decay（仅 --yaml 模式）"
    ),
    cluster: bool = typer.Option(
        True, "--cluster", help="因子聚类分析（仅 --yaml 模式）"
    ),
    relevance_threshold: Optional[float] = typer.Option(
        0.8,
        "--relevance-threshold",
        min=0.0,
        max=1.0,
        help="聚类相关性阈值，0~1（仅 --yaml 模式）",
    ),
    batch_size: int = typer.Option(100, "--batch-size", min=1, help="评估批大小"),
    top_n: int = typer.Option(20, "--top-n", help="终端显示前 N 条结果"),
    min_sharpe: float = typer.Option(0.3, "--min-sharpe", help="最低 Sharpe 过滤阈值"),
    min_ann_ret: float = typer.Option(
        0.1, "--min-ann-ret", help="最低年化收益过滤阈值"
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="输出 CSV 文件路径"
    ),
    overlap_topn: Optional[int] = typer.Option(
        50,
        "--overlap-topn",
        min=1,
        help="计算任意两因子 Top-N 持仓重合度（Hit Rate），设置 N 即启用，默认不计算",
    ),
) -> None:
    """批量因子评估。

    [bold]模式 1[/bold] — 基于 YAML 配置（支持 IC Decay / Turnover Decay / 聚类）：

      quant evals -y s1.yaml -s 20220101 --ic-decay --cluster

    [bold]模式 2[/bold] — 直接指定表达式：

      quant evals --expr "f1=ts_mean(AMOUNT,40)" --expr "f2=rank(CLOSE)"

    [bold]模式 3[/bold] — 从 CSV 文件加载：

      quant evals --csv-file factors.csv

    [bold]模式 4[/bold] — 自动扫描 pool 目录：

      quant evals -s 20220101

    [bold]持仓重合度[/bold] — 任意模式下追加 --overlap-topn N 即可计算：

      quant evals --expr "f1=..." --expr "f2=..." --overlap-topn 50
    """
    # ── YAML 模式 ────────────────────────────────────────────────────────────
    if yaml_file is not None:
        from alpha_factory.cli.eval_core import run_eval_pipeline
        from alpha_factory.cli.utils import resolve_yaml_path
        from alpha_factory.config.strategy import StrategyConfig

        yaml_file = resolve_yaml_path(yaml_file)
        console.print(f"[cyan]加载 YAML 配置: {yaml_file}[/cyan]")
        try:
            config = StrategyConfig.from_yaml(yaml_file)
        except Exception as e:
            console.print(f"[red]❌ 加载 YAML 失败: {e}[/red]")
            raise typer.Exit(code=1)

        if start_date:
            config.start_date = start_date
        if end_date:
            config.end_date = end_date
        if ic_decay:
            config.ic_decay = True
        if turnover_decay:
            config.turnover_decay = True
        if cluster:
            config.cluster = True
        if relevance_threshold is not None:
            config.relevance_threshold = relevance_threshold

        console.print(f"[dim]策略: {config.name} | 池: {config.pool}[/dim]")

        csv_factors = None
        if csv_file:
            csv_factors = _load_csv_factors(csv_file, name_col, expr_col)

        eval_start_ts = perf_counter()
        try:
            result_df = run_eval_pipeline(
                config=config,
                csv_factors=csv_factors,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
            )
        except Exception as e:
            console.print(f"[red]❌ 评估执行失败: {e}[/red]")
            raise typer.Exit(code=1)
        total_eval_seconds = perf_counter() - eval_start_ts

        if result_df.is_empty():
            console.print(
                "[yellow]⚠️ 评估结果为空，请检查因子表达式或数据范围。[/yellow]"
            )
            raise typer.Exit(code=0)

        filtered_df = result_df.filter(
            (pl.col("sharpe") > min_sharpe) & (pl.col("ann_ret") > min_ann_ret)
        )
        if filtered_df.is_empty():
            console.print(
                f"[yellow]⚠️ 过滤后无可用因子"
                f"（min_sharpe={min_sharpe}, min_ann_ret={min_ann_ret}）。[/yellow]"
            )
            raise typer.Exit(code=0)

        _print_result_table(filtered_df, top_n)
        console.print(f"[dim]时间统计: 耗时 {total_eval_seconds:.3f}s[/dim]")

        # ── Top-N 持仓重合度（YAML 模式）
        if overlap_topn is not None:
            filtered_names = filtered_df["factor"].to_list()
            if len(filtered_names) < 2:
                console.print(
                    "[yellow]⚠️ 至少需要 2 个因子才能计算持仓重合度。[/yellow]"
                )
            else:
                console.print(
                    f"\n[bold]计算 Top-{overlap_topn} 持仓重合度"
                    f"（{len(filtered_names)} 个因子，"
                    f"{len(filtered_names) * (len(filtered_names) - 1) // 2} 对）...[/bold]"
                )
                from alpha_factory.cli.eval_core import _get_pool_universe, _load_data

                _pool = _get_pool_universe(config.pool)
                _factor_pairs_all = [
                    (row["factor"], row["expression"])
                    for row in filtered_df.select(["factor", "expression"]).to_dicts()
                    if "expression" in filtered_df.columns
                ]
                if not _factor_pairs_all:
                    console.print(
                        "[dim]（YAML 模式下需要 expression 列以重建数据，跳过重合度计算）[/dim]"
                    )
                else:
                    _overlap_lf = _load_data(
                        config, _factor_pairs_all, start_date, end_date
                    )
                    _overlap_df = batch_topn_overlap(
                        _overlap_lf, filtered_names, overlap_topn
                    )
                    _print_overlap_table(_overlap_df, overlap_topn)

        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            filtered_df.write_csv(out_path)
            console.print(f"[green]✅ 完整结果已写入: {out_path}[/green]")
        console.print(
            f"[bold cyan]批量评估完成[/bold cyan] | {len(filtered_df)} 个因子通过筛选"
        )
        return

    # ── 表达式 / CSV / 自动扫描模式 ──────────────────────────────────────────
    auto_pool_mode = False
    pool = MainSmallPool()
    factor_pairs: list[tuple[str, str]] = []
    if expr:
        factor_pairs = _parse_expr_args(list(expr))
    elif csv_file:
        factor_pairs = _load_csv_factors(csv_file, name_col, expr_col)
    else:
        pool_dir = pool.pool_dir
        csv_files = sorted(
            f for f in pool_dir.glob("*.csv") if f.name != "main_small_pool.csv"
        )
        if not csv_files:
            console.print(
                "[red]❌ 至少提供 --expr 或 --csv-file，"
                "或在 pool 目录中放置 CSV 文件。[/red]"
            )
            console.print(
                "[dim]使用 --expr 指定因子表达式，"
                "例如：--expr 'factor1=ts_mean(AMOUNT,40)'[/dim]"
            )
            raise typer.Exit(code=1)
        auto_pool_mode = True
        console.print(
            f"[dim]扫描池目录 {pool_dir}，发现 {len(csv_files)} 个 CSV 文件[/dim]"
        )
        seen_names: set[str] = set()
        for csv_path in csv_files:
            try:
                batch = _load_csv_factors(csv_path, "factor", "expression")
            except SystemExit:
                continue
            for name, expression in batch:
                if name not in seen_names:
                    seen_names.add(name)
                    factor_pairs.append((name, expression))
        if not factor_pairs:
            console.print(
                "[red]❌ 至少提供 --expr 或 --csv-file，"
                "或确保 pool 目录中 CSV 包含有效因子。[/red]"
            )
            raise typer.Exit(code=1)
    console.print(f"[bold]准备评估 {len(factor_pairs)} 个因子[/bold]")
    actual_start = start_date or "20190101"
    exprs_for_loader = [f"{name}={expression}" for name, expression in factor_pairs]
    factor_names = [name for name, _ in factor_pairs]
    dp = DataProvider()
    lf = dp.load_pool_data(pool, actual_start, end_date, exprs=exprs_for_loader)
    eval_start_ts = perf_counter()
    result_df = _run_batched_eval(lf, factor_names, batch_size)
    total_eval_seconds = perf_counter() - eval_start_ts
    if result_df.is_empty():
        console.print("[yellow]⚠️ 评估结果为空，请检查因子表达式或数据范围。[/yellow]")
        raise typer.Exit(code=0)
    filtered_df = result_df.filter(
        (pl.col("sharpe") > min_sharpe) & (pl.col("ann_ret") > min_ann_ret)
    )
    if filtered_df.is_empty():
        console.print(
            f"[yellow]⚠️ 过滤后无可用因子"
            f"（min_sharpe={min_sharpe}, min_ann_ret={min_ann_ret}）。[/yellow]"
        )
        raise typer.Exit(code=0)
    _print_result_table(filtered_df, top_n)
    console.print(
        f"[dim]时间统计: 评估 {len(factor_pairs)} 个因子耗时 {total_eval_seconds:.3f}s[/dim]"
    )

    # ── Top-N 持仓重合度（非 YAML 模式，lf 直接可用）
    if overlap_topn is not None:
        filtered_names = filtered_df["factor"].to_list()
        if len(filtered_names) < 2:
            console.print("[yellow]⚠️ 至少需要 2 个因子才能计算持仓重合度。[/yellow]")
        else:
            console.print(
                f"\n[bold]计算 Top-{overlap_topn} 持仓重合度"
                f"（{len(filtered_names)} 个因子，"
                f"{len(filtered_names) * (len(filtered_names) - 1) // 2} 对）...[/bold]"
            )
            overlap_df = batch_topn_overlap(lf, filtered_names, overlap_topn)
            _print_overlap_table(overlap_df, overlap_topn)

    if auto_pool_mode and output is None:
        output = pool.pool_dir / "main_small_pool.csv"
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.write_csv(out_path)
        console.print(f"[green]✅ 完整结果已写入: {out_path}[/green]")
    console.print(
        f"[bold cyan]批量评估完成[/bold cyan] | {len(filtered_df)} 个因子通过筛选"
    )


__all__ = ["quant_evals"]
