"""quant gp —— 遗传编程因子挖掘命令。

使用遗传算法（DEAP）在指定股票池和时间段内搜索量化因子表达式。
进化结果自动写入 output/<pool_name>/gp_best_factors.csv。

示例：
  quant gp                          # 使用默认参数（20代，1000种群）
  quant gp --n-gen 50 --n-pop 2000  # 更大规模搜索
  quant gp -s 20200101 -e 20231231  # 自定义时间范围
"""

from __future__ import annotations

import traceback
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def quant_gp(
    n_gen: int = typer.Option(20, "--n-gen", help="进化代数（迭代轮次）"),
    n_pop: int = typer.Option(1000, "--n-pop", help="初始种群大小"),
    start_date: str = typer.Option(
        "20190101", "-s", "--start-date", help="数据开始日期 YYYYMMDD"
    ),
    end_date: str = typer.Option(
        "20241231", "-e", "--end-date", help="数据结束日期 YYYYMMDD"
    ),
    max_height: int = typer.Option(4, "--max-height", help="表达式树最大高度"),
    hof_size: int = typer.Option(100, "--hof-size", help="名人堂（Hall of Fame）大小"),
    top_n: int = typer.Option(1000, "--top-n", help="名人堂筛选保留因子数"),
    cluster_threshold: float = typer.Option(
        0.7, "--cluster-threshold", help="因子独立性聚类相关系数阈值"
    ),
    cxpb: float = typer.Option(0.5, "--cxpb", help="交叉概率 [0, 1]"),
    mutpb: float = typer.Option(0.3, "--mutpb", help="变异概率 [0, 1]"),
) -> None:
    """[遗传编程] 使用 GP 算法自动挖掘 Alpha 因子。

    在 [bold cyan]MainSmallPool[/bold cyan] 股票池中运行遗传算法，
    按年化收益（ann_ret）为优化目标，搜索最优因子表达式。
    结果保存至 output/main_small_pool/gp_best_factors.csv。
    """
    from alpha_factory.gp.small_cs_generator import SmallCSGenerator

    config: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "max_height": max_height,
        "hof_size": hof_size,
        "top_n": top_n,
        "cluster_threshold": cluster_threshold,
        "cxpb": cxpb,
        "mutpb": mutpb,
    }

    console.print(
        f"\n[bold green]🧬 GP 因子挖掘启动[/bold green]\n"
        f"  进化代数:   [cyan]{n_gen}[/cyan]\n"
        f"  种群大小:   [cyan]{n_pop}[/cyan]\n"
        f"  时间范围:   [cyan]{start_date}[/cyan] ~ [cyan]{end_date}[/cyan]\n"
        f"  树高上限:   [cyan]{max_height}[/cyan]\n"
        f"  名人堂大小: [cyan]{hof_size}[/cyan]\n"
        f"  聚类阈值:   [cyan]{cluster_threshold}[/cyan]\n"
    )

    try:
        generator = SmallCSGenerator(config=config)
        _pop, _logbook, hof = generator.run(n_gen, n_pop=n_pop)
    except Exception:
        console.print("[bold red]✗ 遗传算法运行失败：[/bold red]")
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)

    # ── 打印名人堂摘要 ────────────────────────────────────────────────────
    hof_list = list(hof)
    console.print(
        f"\n[bold green]✓ 进化完成[/bold green]，名人堂收录因子数：[cyan]{len(hof_list)}[/cyan]"
    )

    if hof_list:
        table = Table(title="Top-10 候选因子", show_lines=True)
        table.add_column("排名", style="bold", width=6)
        table.add_column("表达式", style="cyan")
        table.add_column("适应度", style="yellow")
        for i, ind in enumerate(hof_list[:10], start=1):
            fitness_vals = ind.fitness.values if hasattr(ind, "fitness") else ("—",)
            fitness_str = (
                ", ".join(f"{v:.4f}" for v in fitness_vals) if fitness_vals else "—"
            )
            table.add_row(str(i), str(ind), fitness_str)
        console.print(table)

    save_dir = getattr(generator, "save_dir", None)
    if save_dir:
        console.print(f"\n[dim]结果已保存至：{save_dir}[/dim]\n")


__all__ = ["quant_gp"]
