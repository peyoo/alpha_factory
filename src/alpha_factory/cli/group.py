"""quant group —— 通过因子时序相关性聚类，对因子库进行去同质化分组。

流程：
  1. 读取含 factor / expression 列的指标 CSV（默认为股票池同名 CSV）。
  2. 用 DataProvider 计算全部因子的时序值，构建宽格式矩阵
     （行 = 日期×标的，列 = 因子）。
  3. 调用 batch_clustering 计算斯皮尔曼秩相关距离并层次聚类。
  4. 将聚类结果（cluster_id）写回 CSV 的 group 列。
  5. 若指定 --select N，则在每个聚类内按各指标分别取前 N 名，取并集输出。
     - ic / ic_ir  取绝对值最大的前 N 名
     - ann_ret / sharpe 取原值最大的前 N 名

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

# 每个指标的配置：(优先列名, 备用列名, 是否取绝对值排序)
# ic/ic_ir 取绝对值最大；ann_ret/sharpe 取原值最大
_SELECT_METRIC_CONFIGS: list[tuple[str, str | None, bool]] = [
    ("ic_mean_abs", "ic_mean", True),
    ("ic_ir_abs", "ic_ir", True),
    ("ann_ret", None, False),
    ("sharpe", None, False),
]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _select_top_n_per_cluster(
    df: pl.DataFrame,
    n: int,
) -> pl.DataFrame:
    """每个聚类内，按各指标分别取前 N 名，返回并集（去重）。

    - ic / ic_ir  系列：按绝对值降序取 Top N
    - ann_ret / sharpe：按原值降序取 Top N
    """
    selected_factors: set[str] = set()

    for preferred, fallback, use_abs in _SELECT_METRIC_CONFIGS:
        col = (
            preferred
            if preferred in df.columns
            else (fallback if fallback and fallback in df.columns else None)
        )
        if col is None:
            continue

        sort_col = f"__abs_{col}" if use_abs else col
        work = df.with_columns(pl.col(col).abs().alias(sort_col)) if use_abs else df

        top_factors = (
            work.sort(sort_col, descending=True)
            .group_by("group", maintain_order=False)
            .head(n)["factor"]
            .to_list()
        )
        selected_factors.update(top_factors)

    return df.filter(pl.col("factor").is_in(selected_factors)).sort(
        ["group", "sharpe" if "sharpe" in df.columns else "factor"],
        descending=[False, True],
    )


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
    """打印每个聚类的因子数量摘要。"""
    summary = df.group_by("group").agg(pl.len().alias("count")).sort("group")
    table = Table(title="📊 聚类分组摘要", show_lines=True)
    table.add_column("cluster_id", style="bold yellow", justify="center")
    table.add_column("因子数", justify="right")

    for row in summary.iter_rows(named=True):
        table.add_row(str(row["group"]), str(row["count"]))
    console.print(table)


def _print_factor_table(df: pl.DataFrame) -> None:
    """打印因子详情表（factor / group 及可用指标列）。"""
    metric_cols = [c for _, c, _ in _SELECT_METRIC_CONFIGS if c and c in df.columns]
    preferred_cols = [p for p, _, _ in _SELECT_METRIC_CONFIGS if p in df.columns]
    metric_display = list(dict.fromkeys(preferred_cols + metric_cols))  # 去重保序

    display_cols = [c for c in ["factor", "group"] + metric_display if c in df.columns]
    sub = df.select(display_cols)

    col_styles = {"factor": "cyan", "group": "bold yellow"}
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
        0.9,
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
        2,
        "--select",
        help=(
            "每簇内各指标（ic/ic_ir 取绝对值，ann_ret/sharpe 取原值）"
            "分别取前 N 名，输出并集"
        ),
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

    # --- 3. 检测可用指标列 ---
    all_metric_cols = [
        col
        for preferred, fallback, _ in _SELECT_METRIC_CONFIGS
        for col in (preferred, fallback)
        if col and col in df_meta.columns
    ]
    available_metrics = list(dict.fromkeys(all_metric_cols))  # 去重保序
    if not available_metrics:
        console.print(
            "[yellow]⚠️  CSV 中未找到指标列（sharpe/ic_ir/ann_ret 等），"
            "--select 将无法筛选。[/yellow]"
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

    # --- 6. 写回 group 列（cluster_id），移除旧的 score/group 列 ---
    df_result = df_meta.clone()
    cols_to_drop = [c for c in ("group",) if c in df_result.columns]
    if cols_to_drop:
        df_result = df_result.drop(cols_to_drop)

    group_col = pl.Series(
        "group",
        [name_to_cluster.get(n, 0) for n in df_result["factor"].to_list()],
    )
    df_result = df_result.with_columns(group_col)

    # --- 7. 打印聚类摘要 ---
    _print_cluster_summary(df_result)

    # --- 8. --select：每簇按各指标分别取前 N 名，取并集 ---
    elapsed = perf_counter() - t0
    if select is not None and select > 0:
        if not available_metrics:
            console.print("[yellow]⚠️  无指标列，--select 无法筛选。[/yellow]")
            df_to_save = df_result
        else:
            df_to_save = _select_top_n_per_cluster(df_result, select)
            console.print(
                f"\n[bold cyan]🏆 每簇各指标 Top {select} 并集（共 {df_to_save.height} 条）[/bold cyan]"
            )
            _print_factor_table(df_to_save)
    else:
        df_to_save = df_result.sort(["group", "factor"])
        _print_factor_table(df_to_save)

    # --- 9. 保存到 CSV（含 group 列；--select 时仅保存筛选后的因子）---
    effective_output = output or csv
    if effective_output is not None and not effective_output.is_absolute():
        effective_output = pool_inst.pool_dir / effective_output
    effective_output.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.write_csv(effective_output)
    console.print(
        f"[bold green]💾 已保存结果[/bold green] → {effective_output}（{df_to_save.height} 条）"
    )

    console.print(f"[dim]总耗时 {elapsed:.2f}s[/dim]")
