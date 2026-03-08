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

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.config.strategy import FactorRank, StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.utils.schema import F

console = Console()

_COMPOSITE_COL = "COMPOSITE_OPT"
_RANK_PREFIX = "_RANK_"  # 预计算截面 rank 列的命名前缀


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


def precompute_factor_ranks(
    ranks: list[FactorRank],
    dp: DataProvider,
    pool: PoolUniverse,
    start_date: str,
    end_date: Optional[str],
) -> pl.DataFrame:
    """一次性加载所有因子原始值，并计算截面百分比 rank，返回 DataFrame。

    对每个因子生成列 ``_RANK_{name}``：
    - 仅在 ``POOL_MASK=True`` 的标的中排名（与回测逻辑对齐）
    - 停牌/不在池的标的排名为 ``null``

    同时保留回测所需的基础列（DATE / ASSET / POOL_MASK / VWAP / CLOSE /
    IS_UP_LIMIT / IS_DOWN_LIMIT / IS_SUSPENDED）。
    """
    factor_names = [r.name for r in ranks]

    raw_col_names = [f"RAW_{name}" for name in factor_names]
    exprs = [
        f"{raw_col} = {rank_def.expression.strip()}"
        for raw_col, rank_def in zip(raw_col_names, ranks)
    ]

    console.print(
        f"[bold cyan]⚙ 预计算因子数据[/bold cyan]  "
        f"共 {len(ranks)} 个因子，时间范围 {start_date} ~ {end_date or '最新'}"
    )

    lf = dp.load_pool_data(pool, start_date, end_date, exprs=exprs)

    # 截面百分比 rank（用 direction 控制排序方向）
    # direction=1：大値好，descending=False → 最大値得最高秩（与回测中 ascending=False 一致）
    # direction=-1：小値好，descending=True  → 最小値得最高秩
    rank_exprs = [
        pl.when(pl.col(F.POOL_MASK))
        .then(pl.col(raw_col))
        .otherwise(None)
        .rank(method="average", descending=(rank_def.direction < 0))
        .over(F.DATE)
        .alias(f"{_RANK_PREFIX}{name}")
        for raw_col, name, rank_def in zip(raw_col_names, factor_names, ranks)
    ]
    lf = lf.with_columns(rank_exprs)

    keep_cols = [
        F.DATE,
        F.ASSET,
        F.POOL_MASK,
        F.VWAP,
        F.CLOSE,
        F.IS_UP_LIMIT,
        F.IS_DOWN_LIMIT,
        F.IS_SUSPENDED,
    ] + [f"{_RANK_PREFIX}{name}" for name in factor_names]

    available = set(lf.collect_schema().names())
    keep_cols = [c for c in keep_cols if c in available]

    df = lf.select(keep_cols).collect()

    console.print(
        f"[green]✓ 预计算完成[/green]  "
        f"行数: {df.height:,}  日期: {df[F.DATE].min()} ~ {df[F.DATE].max()}"
    )
    return df


def make_composite(
    base_df: pl.DataFrame,
    factor_names: list[str],
    weights: np.ndarray,
) -> pl.DataFrame:
    """在 base_df（含预计算 rank 列）上构造加权合成因子列并返回新 DataFrame。

    合成公式::

        COMPOSITE = Σ (weight_i × _RANK_{name_i})

    direction 已在预计算阶段通过 rank 的 descending 参数吸收，
    此处仅做纯加权求和。
    """
    terms = [
        pl.col(f"{_RANK_PREFIX}{name}").cast(pl.Float64) * float(w)
        for name, w in zip(factor_names, weights)
    ]
    composite_expr = terms[0]
    for t in terms[1:]:
        composite_expr = composite_expr + t

    return base_df.with_columns(composite_expr.alias(_COMPOSITE_COL))


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
    if not yaml_file.is_absolute():
        yaml_file = Path.cwd() / yaml_file

    if not yaml_file.exists():
        typer.echo(f"❌ YAML 文件不存在: {yaml_file}", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = StrategyConfig.from_yaml(yaml_file)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"❌ 加载策略配置失败: {exc}", err=True)
        raise typer.Exit(code=1)

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

    # ---- 2. 一次性预计算所有因子截面 rank（整个优化过程共享） ----
    dp = DataProvider()
    base_df = precompute_factor_ranks(ranks, dp, pool_instance, start_date, end_date)

    # ---- 3. 定义 Optuna 目标函数（仅加权合成 + 回测，无 I/O） ----
    def objective(trial: "optuna.Trial") -> float:
        raw = np.array(
            [trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in factor_names]
        )
        weights = softmax_weights(raw)
        try:
            df_trial = make_composite(base_df, factor_names, weights)
            result = backtest_daily_evolving(
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
    study = optuna.create_study(direction="maximize", sampler=sampler)

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

    # ---- 5. 提取最优权重 ----
    best_raw = np.array([study.best_params[f"w_{name}"] for name in factor_names])
    best_weights = softmax_weights(best_raw)
    best_ann_ret = study.best_value

    # ---- 6. 打印对比表 ----
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

    # ---- 7. 写回原 YAML（通过 StrategyConfig.to_yaml，保留注释） ----
    for rank_item, opt_w in zip(cfg.ranks, best_weights):
        rank_item.weight = round(float(opt_w), 6)

    cfg.to_yaml(yaml_file)

    console.print(f"[bold green]💾 最优权重已写回[/bold green] → {yaml_file}")
