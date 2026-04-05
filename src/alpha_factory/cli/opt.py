"""
quant opt — 多因子权重 Optuna 超参数优化命令

流程：
  1. 读取 YAML 策略文件（扁平格式：name/type/ranks 直接在根节点）。
  2. 一次性预计算所有因子的截面 rank 值，整个优化过程共享同一份数据。
  3. 用 Optuna TPE 采样器搜索各因子在 [-3, 3] 空间的 logit 权重，
     每个 trial 仅对预计算列做加权求和 + 逐日演进回测，无重复 I/O。
  4. 优化目标：最大化年化收益率。
  5. 将最优权重写回原 YAML 文件的 weight 字段（ruamel.yaml 保留格式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from loguru import logger

from alpha_factory.cli.utils import PoolUniverseEnum, resolve_yaml_path
from alpha_factory.config.strategy import FactorRank, StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.evaluation.batch.ic_summary import batch_ic_summary
from alpha_factory.utils.schema import F

console = Console()

_COMPOSITE_COL = "COMPOSITE_OPT"


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _resolve_pool(pool_name: str) -> PoolUniverse:
    """将股票池名称字符串解析为 PoolUniverse 实例。

    遍历 PoolUniverseEnum，找到 name 匹配的成员并实例化返回。
    若未找到则抛出 ValueError。
    """
    for member in PoolUniverseEnum:
        instance = member.value()
        if instance.name == pool_name:
            return instance
    raise ValueError(
        f"未知股票池 {pool_name!r}，可选值: "
        + ", ".join(m.value().name for m in PoolUniverseEnum)
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def softmax_weights(x: np.ndarray) -> np.ndarray:
    """将任意实数向量映射为正权重，且 sum=1（softmax）。"""
    e = np.exp(x - x.max())
    return e / e.sum()


def auto_fill_directions(
    ranks: list[FactorRank],
    dp: DataProvider,
    pool: PoolUniverse,
    start_date: str,
    end_date: Optional[str],
) -> None:
    """对 direction=None 的因子，通过 batch_ic_summary 自动推断排序方向。

    ic_mean >= 0 → direction=1（因子值越大越好）；
    ic_mean < 0  → direction=-1（因子值越小越好）。
    """
    pending = [r for r in ranks if r.direction is None]
    if not pending:
        return

    exprs = [r.expr_str for r in pending]
    factor_cols = [r.name for r in pending]

    console.print(
        f"[bold cyan]⚙ 自动推断因子方向[/bold cyan]  "
        f"共 {len(pending)} 个未指定方向的因子，正在加载数据计算 IC…"
    )

    lf = dp.load_pool_data(pool, start_date, end_date, exprs=exprs)
    ic_df = batch_ic_summary(lf, factors=factor_cols)

    if ic_df.is_empty():
        console.print(
            "[yellow]⚠ IC 计算结果为空，所有未指定方向的因子默认使用 direction=1[/yellow]"
        )
        for r in pending:
            r.direction = 1
        return

    ic_map: dict[str, float] = dict(
        zip(ic_df["factor"].to_list(), ic_df["ic_mean"].to_list())
    )

    from rich.table import Table as RichTable

    t = RichTable(show_header=True, header_style="bold magenta")
    t.add_column("因子", style="cyan")
    t.add_column("ic_mean", justify="right")
    t.add_column("direction", justify="center", style="bold")
    for rank in pending:
        ic_mean = ic_map.get(rank.name)
        direction: int = 1 if (ic_mean is None or ic_mean >= 0) else -1
        rank.direction = direction  # type: ignore[assignment]
        ic_str = f"{ic_mean:.4f}" if ic_mean is not None else "N/A"
        color = "green" if direction == 1 else "red"
        t.add_row(rank.name, ic_str, f"[{color}]{direction}[/{color}]")
    console.print(t)


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------


def compute_ann_ret(nav_series: pl.Series) -> float:
    """从 NAV 序列计算年化收益率。"""
    n = len(nav_series)
    if n < 2:
        return 0.0
    total = float(nav_series[-1]) - 1.0
    return float((1.0 + total) ** (252.0 / n) - 1.0)


# ---------------------------------------------------------------------------
# 预处理器定义
# ---------------------------------------------------------------------------
# 预处理函数现在从 StrategyConfig.preprocess 字段读取
# 示例在 YAML 中: preprocess: [my_cs_mad_zscore_resid, ...]


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def quant_opt(
    yaml_file: Path = typer.Option(
        ...,
        "--yaml",
        "-y",
        help="多因子策略 YAML 文件路径（如 output/main_small_pool/s1.yaml）",
    ),
    start_date: str = typer.Option(
        "20190101",
        "-s",
        "--start-date",
        help="回测开始日期 YYYYMMDD",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end",
        "--end-date",
        help="回测结束日期 YYYYMMDD（默认取仓库最新日期）",
    ),
    n_trials: int = typer.Option(100, "--n-trials", help="Optuna 试验次数"),
    seed: int = typer.Option(42, "--seed", help="随机种子，确保结果可复现"),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="是否显示优化进度条"
    ),
):
    """
    [优化指令] 使用 Optuna 对 YAML 多因子策略的权重系数进行超参数优化。

    \b
    策略参数（pool / hold_num / sell_rank / cost 等）均从 YAML 文件读取，
    无需在命令行重复指定。YAML 需符合 StrategyConfig 格式，例如：

        name: "s1"
        pool: "main_small_pool"
        hold_num: 10
        sell_rank: 30
        cost: 0.003
        ranks:
          - name: "f1"
            weight: 1.0
            expression: "ts_mean(AMOUNT, 60)"
            direction: -1

    \b
    优化目标：最大化年化收益率（逐日演进回测）
    权重约束：所有权重 ≥ 0，且归一化（sum=1，通过 softmax 实现）
    因子预计算：所有因子截面 rank 在优化前一次性计算完毕，trial 间共享
    结果写回：最优权重自动更新到原 YAML 文件的 weight 字段

    \b
    示例:
      quant opt --yaml output/main_small_pool/s1.yaml \\
                --start-date 20210101 --n-trials 50
    """
    # ---- 依赖检查 ----
    try:
        import optuna
    except ImportError:
        typer.echo("❌ 未安装 optuna，请运行 `uv sync` 安装依赖", err=True)
        raise typer.Exit(code=1)

    # 静默 optuna 内部日志，由 rich 接管显示
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- 1. 通过 StrategyConfig 加载并校验 YAML ----
    yaml_file = resolve_yaml_path(yaml_file)

    if not yaml_file.exists():
        typer.echo(f"❌ YAML 文件不存在: {yaml_file}", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = StrategyConfig.from_yaml(yaml_file)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"❌ 加载策略配置失败: {exc}", err=True)
        raise typer.Exit(code=1)

    # 开启优化模式：build_actions() 返回空，由 run_opt_trial() 接管所有计算
    cfg.opt_mode = True

    if len(cfg.ranks) < 2:
        typer.echo(
            f"❌ 策略 {cfg.name!r} 的 ranks 数量 < 2，无需优化",
            err=True,
        )
        raise typer.Exit(code=1)

    strat_name = cfg.name
    ranks = cfg.ranks
    n_factors = len(ranks)
    factor_names = cfg.factor_names
    directions = cfg.factor_directions
    original_weights = cfg.factor_weights

    # 将 pool 名称字符串解析为 PoolUniverse 实例
    try:
        pool_instance = _resolve_pool(cfg.pool)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1)

    console.rule("[bold cyan]Optuna 多因子权重优化[/bold cyan]")
    console.print(
        f"  策略: [bold]{strat_name}[/bold] | 因子数: {n_factors} | "
        f"股票池: {cfg.pool} | {start_date} ~ {end_date or '最新'}"
    )
    console.print(f"  因子: {factor_names}")
    console.print(
        f"  持仓: hold_num={cfg.hold_num}, sell_rank={cfg.sell_rank} | "
        f"成本: {cfg.cost:.4f} | 试验次数: {n_trials}\n"
    )

    # ---- 2. 一次性预计算所有因子（PreProcess: 市值中性化 + rank + z-normalize + 正交化）----
    dp = DataProvider()

    # 对 direction=None 的因子，通过 IC 自动推断排序方向
    auto_fill_directions(ranks, dp, pool_instance, start_date, end_date)

    console.print(
        f"[bold cyan]⚙ 预计算因子数据[/bold cyan]  "
        f"共 {len(ranks)} 个因子，时间范围 {start_date} ~ {end_date or '最新'}"
    )

    # 构建预计算 actions：因子预处理 + 无动态参数的条件 And/Or 聚合（所有 trial 共享）
    from alpha_factory.data_provider.factorsprocessor import FactorsPreProcessor
    from alpha_factory.data_provider.prcoessors import And, Or

    def _g_names(g):
        return [c.name for c in g if c.expression.strip()]

    def _has_dyn(g):
        return any(c.has_opt_params for c in g)

    precompute_actions = []
    if cfg.preprocess and cfg.factor_names:
        precompute_actions.append(
            FactorsPreProcessor(factors=cfg.factor_names, actions=cfg.preprocess)
        )
    if (ns := _g_names(cfg.pool_mask)) and not _has_dyn(cfg.pool_mask):
        precompute_actions.append(And(factors=ns, name=F.POOL_MASK))
    if (ns := _g_names(cfg.buy_able)) and not _has_dyn(cfg.buy_able):
        precompute_actions.append(And(factors=ns, name="buy_able"))
    if (ns := _g_names(cfg.not_buy_able)) and not _has_dyn(cfg.not_buy_able):
        precompute_actions.append(Or(factors=ns, name="not_buy_able"))
    if (ns := _g_names(cfg.sell_able)) and not _has_dyn(cfg.sell_able):
        precompute_actions.append(And(factors=ns, name="sell_able"))
    if (ns := _g_names(cfg.not_sell_able)) and not _has_dyn(cfg.not_sell_able):
        precompute_actions.append(Or(factors=ns, name="not_sell_able"))

    # precompute_actions.append(SymmetricOrtho(factors=cfg.factor_names))

    all_exprs = cfg.ranked_factor_exprs + cfg.get_static_condition_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        start_date,
        end_date,
        exprs=all_exprs,
        actions=precompute_actions or None,
    )
    base_df = lf.collect()
    console.print(
        f"[green]✓ 预计算完成[/green]"
        f"行数: {base_df.height:,}  日期: {base_df[F.DATE].min()} ~ {base_df[F.DATE].max()}"
    )

    # ---- 3. 定义 Optuna 目标函数（仅加权合成 + 回测，无 I/O） ----
    def objective(trial: "optuna.Trial") -> float:
        try:
            df_trial = cfg.run_opt_trial(trial, dp, base_df)
            result = backtest_quick_daily(
                df_input=df_trial,
                factor_col=_COMPOSITE_COL,
                n_buy=cfg.hold_num,
                sell_rank=cfg.sell_rank,
                cost_rate=cfg.cost,
                ascending=False,
            )
            return compute_ann_ret(result["daily_results"]["NAV"])
        except Exception:  # noqa: BLE001
            return -1.0

    # ---- 4. 运行 Optuna 优化 ----
    sampler = optuna.samplers.TPESampler(seed=seed)
    # sampler = optuna.samplers.CmaEsSampler(seed=seed)  # CMA-ES 适合连续优化
    study = optuna.create_study(direction="maximize", sampler=sampler)

    _DE_MODULE = "alpha_factory.evaluation.backtest.daily_evolving"
    logger.disable("alpha_factory.evaluation.backtest.daily_evolving")
    logger.disable("alpha_factory.evaluation.backtest.quick_daily")

    try:
        if show_progress:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("最优年化: [bold green]{task.fields[best]:.2%}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("优化中...", total=n_trials, best=0.0)

                def _cb(s: optuna.Study, t: optuna.Trial) -> None:
                    best_val = s.best_value if s.best_trial is not None else 0.0
                    progress.update(task, advance=1, best=best_val)

                study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        else:
            study.optimize(objective, n_trials=n_trials)
    finally:
        logger.enable(_DE_MODULE)

    # ---- 5. 提取最优权重 ----
    best_raw = np.array([study.best_params[f"w_{name}"] for name in factor_names])
    best_weights = softmax_weights(best_raw)
    best_ann_ret = study.best_value  # ---- 6. 打印对比表 ----
    table = Table(
        title=f"优化结果 — 策略: {strat_name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("因子", style="cyan", min_width=12)
    table.add_column("原始权重（归一）", justify="right")
    table.add_column("优化权重", justify="right", style="bold green")
    table.add_column("方向", justify="center")
    table.add_column("变化", justify="right")

    orig_sum = sum(original_weights) or 1.0
    for name, orig, opt_w, d in zip(
        factor_names, original_weights, best_weights, directions
    ):
        orig_norm = orig / orig_sum
        delta = opt_w - orig_norm
        delta_str = (
            f"[green]+{delta:.4f}[/green]" if delta >= 0 else f"[red]{delta:.4f}[/red]"
        )
        table.add_row(name, f"{orig_norm:.4f}", f"{opt_w:.4f}", str(d), delta_str)

    console.print(table)
    console.print(
        f"\n[bold yellow]📈 最优年化收益率: {best_ann_ret * 100:.2f}%[/bold yellow]"
        f"  (第 {study.best_trial.number + 1} 次 / 共 {n_trials} 次试验)"
    )

    # 打印条件参数优化结果（若存在）
    if cfg.has_condition_opt_params():
        cond_table = Table(
            title="条件参数优化结果",
            show_header=True,
            header_style="bold magenta",
        )
        cond_table.add_column("条件名", style="cyan")
        cond_table.add_column("参数", style="yellow")
        cond_table.add_column("原始 default", justify="right")
        cond_table.add_column("最优值", justify="right", style="bold green")
        for group in (
            cfg.pool_mask,
            cfg.buy_able,
            cfg.not_buy_able,
            cfg.sell_able,
            cfg.not_sell_able,
        ):
            for cond in group:
                if not cond.has_opt_params:
                    continue
                for param_name, spec in cond.opt.items():
                    key = f"{cond.name}__{param_name}"
                    orig_default = spec.get("default", "N/A")
                    best_val = study.best_params.get(key, "N/A")
                    cond_table.add_row(
                        cond.name, param_name, str(orig_default), str(best_val)
                    )
        console.print(cond_table)

    # ---- 7. 写回原 YAML（通过 StrategyConfig.to_yaml，保留注释） ----
    for rank_item, opt_w in zip(cfg.ranks, best_weights):
        rank_item.weight = round(float(opt_w), 6)

    # 将条件参数的最优值写回 opt.default —— 下次非优化模式直接使用最优值
    if cfg.has_condition_opt_params():
        cfg.apply_best_condition_params(study.best_params)
        console.print("[cyan]✓ 条件参数最优值已写回 opt.default 字段[/cyan]")

    cfg.to_yaml(yaml_file)

    console.print(f"[bold green]💾 最优权重已写回[/bold green] → {yaml_file}")
