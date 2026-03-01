"""quant group —— 通过因子时序相关性聚类，对因子库进行去同质化分组。

流程：
  1. 读取含 factor / expression 列的指标 CSV（默认为股票池同名 CSV）。
  2. 用 DataProvider 计算全部因子的时序值，构建宽格式矩阵
     （行 = 日期×标的，列 = 因子）。
  3. 调用 batch_clustering 计算斯皮尔曼秩相关距离并层次聚类。
  4. 将聚类结果（cluster_id）写回 CSV 的 group 列。
  5. 若指定 --select N，则在每个聚类内按综合评分（sharpe/ic_ir/ann_ret）
     取前 N 名并输出。

路径规则：
  - --csv  未指定 → 自动使用 <pool_dir>/<pool_name>.csv
  - --output 未指定 → 原地覆盖输入 CSV

示例：
  quant group
  quant group --select 2
  quant group --threshold 0.7 --select 1
  quant group --csv my_factors.csv --select 2 -s 20220101
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.batch.cluster import batch_clustering

console = Console()

# 综合评分权重（用于 --select 时在簇内排序）
_METRIC_WEIGHTS: dict[str, float] = {
    "sharpe": 0.35,
    "ic_ir_abs": 0.30,
    "ann_ret": 0.25,
    "ic_mean_abs": 0.10,
}
_METRIC_FALLBACK: dict[str, str] = {
    "ic_ir_abs": "ic_ir",
    "ic_mean_abs": "ic_mean",
}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _compute_score(df: pl.DataFrame, available_metrics: list[str]) -> pl.DataFrame:
    """对指标做 min-max 归一化后加权求和，返回含 score 列的 DataFrame。"""
    score_expr = pl.lit(0.0)
    total_weight = 0.0

    for col_name, weight in _METRIC_WEIGHTS.items():
        actual = (
            col_name
            if col_name in available_metrics
            else _METRIC_FALLBACK.get(col_name)
        )
        if actual is None or actual not in available_metrics:
            continue
        col_min = float(df[actual].abs().min() or 0.0)
        col_max = float(df[actual].abs().max() or 0.0)
        denom = col_max - col_min if col_max != col_min else 1.0
        score_expr = score_expr + ((pl.col(actual).abs() - col_min) / denom) * weight
        total_weight += weight

    if total_weight == 0:
        return df.with_columns(pl.lit(0.0).alias("score"))
    score_expr = score_expr / total_weight
    return df.with_columns(score_expr.alias("score"))


def _load_factor_values(
    factor_pairs: list[tuple[str, str]],
    pool: PoolUniverseEnum,
    start_date: str,
    end_date: Optional[str],
) -> pl.DataFrame:
    """加载所有因子的时序值，返回宽格式 DataFrame（列为各因子）。

    DataProvider 内置批量处理（CODEGEN_BATCH_SIZE），无需外部分批。
    """
    exprs_for_loader = [f"{name}={expr}" for name, expr in factor_pairs]
    names = [name for name, _ in factor_pairs]
    dp = DataProvider()
    try:
        lf = dp.load_pool_data(
            pool.value(), start_date, end_date, exprs=exprs_for_loader
        )
        return lf.select(names).collect()
    except Exception as e:
        console.print(f"[bold red]❌ 因子数据加载失败: {e}[/bold red]")
        raise typer.Exit(code=1)


def _print_cluster_summary(df: pl.DataFrame) -> None:
    """打印每个聚类的因子数量与均分摘要。"""
    summary = (
        df.group_by("group")
        .agg(
            pl.len().alias("count"),
            pl.col("score").mean().round(4).alias("avg_score"),
            pl.col("score").max().round(4).alias("max_score"),
            pl.col("score").min().round(4).alias("min_score"),
        )
        .sort("group")
    )
    table = Table(title="📊 聚类分组摘要", show_lines=True)
    table.add_column("cluster_id", style="bold yellow", justify="center")
    table.add_column("因子数", justify="right")
    table.add_column("均分", justify="right")
    table.add_column("最高分", justify="right")
    table.add_column("最低分", justify="right")

    for row in summary.iter_rows(named=True):
        table.add_row(
            str(row["group"]),
            str(row["count"]),
            f"{row['avg_score']:.4f}",
            f"{row['max_score']:.4f}",
            f"{row['min_score']:.4f}",
        )
    console.print(table)


def _print_factor_table(df: pl.DataFrame, available_metrics: list[str]) -> None:
    """打印因子详情表（factor / group / score 及可用指标列）。"""
    priority = ["factor", "group", "score"] + [
        c for c in available_metrics if c in df.columns
    ]
    display_cols = [c for c in priority if c in df.columns]
    sub = df.select(display_cols)

    col_styles = {"factor": "cyan", "group": "bold yellow", "score": "bold green"}
    table = Table(show_lines=True)
    for col in sub.columns:
        table.add_column(
            col,
            style=col_styles.get(col, ""),
            justify="left" if col == "factor" else "right",
        )
    for row in sub.iter_rows():
        table.add_row(
            *[
                f"{v:.4f}"
                if isinstance(v, float)
                else (str(v) if v is not None else "")
                for v in row
            ]
        )
    console.print(table)


# ---------------------------------------------------------------------------
# CLI 命令
# ---------------------------------------------------------------------------


def quant_group(
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="因子指标 CSV（需含 factor / expression 列）。默认：股票池同名 CSV",
        show_default=False,
    ),
    pool: PoolUniverseEnum = typer.Option(
        PoolUniverseEnum.main_small,
        "--pool",
        help="股票池（用于定位默认 CSV 及加载因子数据）",
    ),
    start_date: str = typer.Option(
        "20190101", "-s", "--start-date", help="因子时序数据开始日期 YYYYMMDD"
    ),
    end_date: Optional[str] = typer.Option(
        None, "-e", "--end-date", help="因子时序数据结束日期 YYYYMMDD（默认至最新）"
    ),
    threshold: float = typer.Option(
        0.8,
        "--threshold",
        help="聚类相关性阈值（0~1）。越高阈值越严格，簇数越多",
    ),
    method: str = typer.Option(
        "average",
        "--method",
        help="层次聚类联动算法: average（默认）| complete | single",
    ),
    sample_n: Optional[int] = typer.Option(
        50000,
        "--sample-n",
        help="聚类时随机采样行数（加速计算，不指定则全量）",
        show_default=False,
    ),
    select: Optional[int] = typer.Option(
        None,
        "--select",
        help="每个聚类取综合评分前 N 名输出（不指定则输出全部）",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="结果写入路径（默认覆盖输入 CSV）",
        show_default=False,
    ),
):
    """
    基于因子时序相关性聚类，对因子库进行去同质化分组。

    \b
    示例:
      quant group
      quant group --select 2
      quant group --threshold 0.7 --select 1
      quant group --csv my_factors.csv -s 20220101 --select 2
    """
    pool_inst = pool.value()
    t0 = perf_counter()

    # --- 1. 确定输入 CSV ---
    if csv is None:
        csv = pool_inst.pool_dir / f"{pool_inst.name}.csv"
    if not csv.is_absolute():
        csv = pool_inst.pool_dir / csv
    if not csv.exists():
        console.print(f"[bold red]❌ 找不到 CSV 文件: {csv}[/bold red]")
        raise typer.Exit(code=1)

    df_meta = pl.read_csv(csv)
    console.print(
        f"[bold cyan]📂 读取 CSV[/bold cyan] {csv} | 共 {df_meta.height} 条因子"
    )

    # --- 2. 检查必需列 ---
    for required in ("factor", "expression"):
        if required not in df_meta.columns:
            console.print(
                f"[bold red]❌ CSV 缺少必需列 '{required}'，"
                f"可用列: {df_meta.columns}[/bold red]"
            )
            raise typer.Exit(code=1)

    factor_pairs: list[tuple[str, str]] = [
        (row["factor"], row["expression"])
        for row in df_meta.select(["factor", "expression"]).to_dicts()
    ]
    factor_names = [p[0] for p in factor_pairs]

    # --- 3. 检测可用评分指标 ---
    candidate_metrics = list(_METRIC_WEIGHTS.keys()) + list(_METRIC_FALLBACK.values())
    available_metrics = [c for c in candidate_metrics if c in df_meta.columns]
    if not available_metrics:
        console.print(
            "[yellow]⚠️  CSV 中未找到评分指标列（sharpe/ic_ir/ann_ret 等），"
            "score 将以 0 填充，--select 结果仅按聚类顺序排列。[/yellow]"
        )

    # --- 4. 加载因子时序值 ---
    console.print(
        f"[bold cyan]📡 加载因子时序数据[/bold cyan] "
        f"{start_date} ~ {end_date or '最新'} | {len(factor_pairs)} 个因子"
    )
    factor_wide = _load_factor_values(factor_pairs, pool, start_date, end_date)

    loaded_names = [n for n in factor_names if n in factor_wide.columns]
    if len(loaded_names) < 2:
        console.print("[bold red]❌ 有效因子列不足 2 个，无法聚类。[/bold red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]✅ 成功加载 {len(loaded_names)} / {len(factor_names)} 个因子列[/green]"
    )

    # --- 5. 调用 batch_clustering ---
    console.print(
        f"[bold cyan]🔬 开始聚类[/bold cyan] "
        f"threshold={threshold} | method={method} | "
        f"sample_n={sample_n or '全量'}"
    )
    name_to_cluster, cluster_to_names = batch_clustering(
        df=factor_wide,
        factors=loaded_names,
        threshold=threshold,
        method=method,
        sample_n=sample_n,
    )
    n_clusters = len(cluster_to_names)
    console.print(f"[green]✅ 聚类完成，共 {n_clusters} 个簇[/green]")

    # --- 6. 计算综合评分 ---
    df_scored = df_meta.clone()
    cols_to_drop = [c for c in ("score", "group") if c in df_scored.columns]
    if cols_to_drop:
        df_scored = df_scored.drop(cols_to_drop)

    if available_metrics:
        df_scored = _compute_score(df_scored, available_metrics)
    else:
        df_scored = df_scored.with_columns(pl.lit(0.0).alias("score"))

    # --- 7. 写回 group 列（cluster_id） ---
    group_col = pl.Series(
        "group",
        [name_to_cluster.get(n, 0) for n in df_scored["factor"].to_list()],
    )
    df_scored = df_scored.with_columns(group_col)

    # --- 8. 保存到 CSV ---
    effective_output = output or csv
    if effective_output is not None and not effective_output.is_absolute():
        effective_output = pool_inst.pool_dir / effective_output
    effective_output.parent.mkdir(parents=True, exist_ok=True)
    df_scored.write_csv(effective_output)
    console.print(f"[bold green]💾 已保存分组结果[/bold green] → {effective_output}")

    # --- 9. 打印摘要 ---
    _print_cluster_summary(df_scored)

    # --- 10. --select：每簇取前 N 名 ---
    elapsed = perf_counter() - t0
    if select is not None and select > 0:
        selected = (
            df_scored.sort("score", descending=True)
            .group_by("group", maintain_order=False)
            .head(select)
            .sort(["group", "score"], descending=[False, True])
        )
        console.print(
            f"\n[bold cyan]🏆 每簇 Top {select} 因子（共 {selected.height} 条）[/bold cyan]"
        )
        _print_factor_table(selected, available_metrics)
    else:
        df_display = df_scored.sort(["group", "score"], descending=[False, True])
        _print_factor_table(df_display, available_metrics)

    console.print(f"[dim]总耗时 {elapsed:.2f}s[/dim]")
