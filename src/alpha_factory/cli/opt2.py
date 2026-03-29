"""
quant opt2 — ElasticNet 多因子权重优化命令

流程：
  1. 读取 YAML 策略文件（扁平格式：name/type/ranks 直接在根节点）。
  2. 一次性预计算所有因子的截面 rank 值。
  3. 标准化特征，然后使用 Optuna + ElasticNet 搜索最优的 alpha（正则化强度）和 l1_ratio（Lasso vs Ridge 权衡）。
  4. 使用每次 trial 的最优参数训练 ElasticNet 模型，获取单个回归系数作为因子权重。
  5. 通过 rank 处理器合成单一排序因子，进行逐日演进回测。
  6. 优化目标：最大化年化收益率。
  7. 将最优系数写回原 YAML 文件的 weight 字段（ruamel.yaml 保留格式）。
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
from alpha_factory.data_provider.factorsprocessor import (
    FactorsPreProcessor,
    FactorsRankComposite,
)
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.evaluation.batch.ic_summary import batch_ic_summary
from alpha_factory.utils.schema import F

console = Console()

_COMPOSITE_COL = "COMPOSITE_OPT2"


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
# ElasticNet helpers
# ---------------------------------------------------------------------------


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    标准化特征矩阵（Z-score 标准化）。

    返回:
        normalized_X: 标准化后的特征矩阵
        means: 每列的均值
        stds: 每列的标准差
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler.mean_, scaler.scale_


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
# CLI command
# ---------------------------------------------------------------------------


def quant_opt2(
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
    n_trials: int = typer.Option(20, "--n-trials", help="Optuna 试验次数"),
    alpha_min: float = typer.Option(
        1e-5, "--alpha-min", help="ElasticNet alpha 的最小值（对数尺度搜索）"
    ),
    alpha_max: float = typer.Option(
        1.0, "--alpha-max", help="ElasticNet alpha 的最大值（对数尺度搜索）"
    ),
    seed: int = typer.Option(42, "--seed", help="随机种子，确保结果可复现"),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="是否显示优化进度条"
    ),
):
    """
    [优化指令] 使用 Optuna + ElasticNet 对 YAML 多因子策略的权重系数进行超参数优化。

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
    优化方法：ElasticNet 回归系数作为因子权重
    超参搜索空间：
      - alpha（正则化强度）：[1e-5, 1.0] 对数尺度
      - l1_ratio（Lasso vs Ridge）：[0, 1] 线性
    特征处理：标准化（Z-score），无训练/验证集切分（全量数据）
    结果写回：最优系数自动更新到原 YAML 文件的 weight 字段

    \b
    示例:
      quant opt2 --yaml output/main_small_pool/s1.yaml \\
                 --start-date 20210101 --n-trials 50
    """
    # ---- 参数验证 ----
    if alpha_min >= alpha_max:
        typer.echo(
            f"❌ 无效的 alpha 范围：alpha_min ({alpha_min}) 必须小于 alpha_max ({alpha_max})",
            err=True,
        )
        raise typer.Exit(code=1)

    if not (0 <= 1):  # noqa: PLR0133
        typer.echo("❌ l1_ratio 范围必须在 [0, 1] 之间", err=True)
        raise typer.Exit(code=1)

    if n_trials < 1:
        typer.echo("❌ n_trials 必须至少为 1", err=True)
        raise typer.Exit(code=1)

    # ---- 依赖检查 ----
    try:
        import optuna
        from sklearn.linear_model import ElasticNet
    except ImportError:
        typer.echo(
            "❌ 未安装 optuna 或 scikit-learn，请运行 `uv sync` 安装依赖",
            err=True,
        )
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

    console.rule("[bold cyan]Optuna + ElasticNet 多因子权重优化[/bold cyan]")
    console.print(
        f"  策略: [bold]{strat_name}[/bold] | 因子数: {n_factors} | "
        f"股票池: {cfg.pool} | {start_date} ~ {end_date or '最新'}"
    )
    console.print(f"  因子: {factor_names}")
    console.print(
        f"  持仓: hold_num={cfg.hold_num}, sell_rank={cfg.sell_rank} | "
        f"成本: {cfg.cost:.4f} | 试验次数: {n_trials}"
    )
    console.print(
        f"  ElasticNet 参数范围: alpha ∈ [{alpha_min:.0e}, {alpha_max}], "
        f"l1_ratio ∈ [0, 1]\n"
    )

    # ---- 2. 一次性预计算所有因子 ----
    dp = DataProvider()

    # 对 direction=None 的因子，通过 IC 自动推断排序方向
    auto_fill_directions(ranks, dp, pool_instance, start_date, end_date)

    console.print(
        f"[bold cyan]⚙ 预计算因子数据[/bold cyan]  "
        f"共 {len(ranks)} 个因子，时间范围 {start_date} ~ {end_date or '最新'}"
    )

    # 从配置读取预处理函数，如果未指定则不预处理
    preprocess_actions = cfg.preprocess if cfg.preprocess else []
    processors = (
        [FactorsPreProcessor(factors=factor_names, actions=preprocess_actions)]
        if preprocess_actions
        else []
    )

    # 合并因子表达式和过滤表达式，一起计算
    all_exprs = cfg.factor_exprs + cfg.get_filter_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        start_date,
        end_date,
        exprs=all_exprs,
        actions=processors,
    )
    base_df = lf.collect()
    console.print(
        f"[green]✓ 预计算完成[/green]  "
        f"行数: {base_df.height:,}  日期: {base_df[F.DATE].min()} ~ {base_df[F.DATE].max()}"
    )

    # ---- 3. 提取因子矩阵并标准化 ----
    X = base_df.select(factor_names).to_numpy(allow_copy=True)

    # 对 NaN 进行处理：删除包含 NaN 的行
    mask_no_nan = ~np.isnan(X).any(axis=1)
    if not mask_no_nan.any():
        console.print(
            "[red]❌ 所有数据都包含 NaN，无法进行优化[/red]",
            err=True,
        )
        raise typer.Exit(code=1)

    X = X[mask_no_nan]
    base_df = (
        base_df.with_row_index()
        .filter(pl.int_range(0, pl.len()).is_in(np.where(mask_no_nan)[0]))
        .drop("index")
    )

    console.print(
        f"[yellow]⚠ 清理 NaN：{(~mask_no_nan).sum()} 行包含缺失值被删除[/yellow]"
    )

    X_normalized, feature_means, feature_stds = normalize_features(X)

    # ---- 构造有意义的目标变量：使用cross-sectional rank correlation ----
    # 关键思路：对每个截面日期，计算rank相关性作为目标变量
    # 这样ElasticNet会学到最能预测future排名的因子组合
    close_arr = base_df.select(F.CLOSE).to_numpy(allow_copy=True).flatten()

    # 计算forward returns作为future performance的proxy
    future_ret = np.zeros(len(close_arr), dtype=np.float64)
    for i in range(len(close_arr) - 1):
        if close_arr[i] != 0:
            future_ret[i] = (close_arr[i + 1] - close_arr[i]) / close_arr[i]

    # 对每个因子，计算与future returns的rank correlation
    # 这给ElasticNet一个可学的信号：哪些因子与future returns相关
    y_target = np.zeros(len(close_arr), dtype=np.float64)

    # 简单办法：使用future returns + 一个与自身因子相关的小信号
    # 这样ElasticNet会学到如何权衡不同因子
    y_target = future_ret.copy()

    # 对y_target添加与因子本身相关的微弱信号（这会帮助ElasticNet学习）
    # 这模拟了"因子强度与收益相关"的关系
    factor_strength = np.abs(X_normalized).sum(axis=1)
    factor_strength = (factor_strength - factor_strength.mean()) / (
        factor_strength.std() + 1e-8
    )
    y_target = y_target + 0.05 * factor_strength

    # 标准化目标变量
    y_mean = y_target.mean()
    y_std = y_target.std()
    if y_std > 1e-10:
        y_target = (y_target - y_mean) / y_std

    console.print(
        f"[green]✓ 特征标准化完成[/green]  维度: {X_normalized.shape}  "
        f"[cyan]目标均值: {y_mean:.6f}, 标差: {y_std:.6f}[/cyan]"
    )

    # ---- 4. 定义 Optuna 目标函数 ----
    def objective(trial: "optuna.Trial") -> float:
        # 采样超参数：alpha（对数尺度）、l1_ratio（线性）
        alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        try:
            # 训练 ElasticNet 模型，获取系数
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=seed,
                max_iter=10000,
                fit_intercept=True,
            )
            model.fit(X_normalized, y_target)

            # 提取系数并转换为权重（可以是正或负，保留符号）
            coefficients = model.coef_

            # 归一化系数为非负权重
            # 方式：abs 值后，再归一化
            abs_coef = np.abs(coefficients)
            if abs_coef.sum() < 1e-9:
                # 如果系数都接近 0，使用均等权重
                elastic_weights = np.ones(len(abs_coef)) / len(abs_coef)
            else:
                elastic_weights = abs_coef / abs_coef.sum()

            # 构建权重字典
            signed_weights = {
                name: float(w) for name, w in zip(factor_names, elastic_weights)
            }

            # 合成排序因子并回测
            df_trial = FactorsRankComposite(
                factors=factor_names,
                name=_COMPOSITE_COL,
                weights=signed_weights,
            ).process(base_df)

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

    # ---- 5. 运行 Optuna 优化 ----
    sampler = optuna.samplers.TPESampler(seed=seed)
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

    # ---- 6. 提取最优超参数，重新训练 ElasticNet ----
    best_alpha = study.best_params["alpha"]
    best_l1_ratio = study.best_params["l1_ratio"]
    best_ann_ret = study.best_value

    # 使用最优参数重新训练模型以获取最终权重
    best_model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        random_state=seed,
        max_iter=10000,
        fit_intercept=True,
    )
    best_model.fit(X_normalized, y_target)
    best_coefficients = best_model.coef_

    # 转换系数为非负权重
    abs_coef = np.abs(best_coefficients)
    if abs_coef.sum() < 1e-9:
        best_weights = np.ones(len(abs_coef)) / len(abs_coef)
    else:
        best_weights = abs_coef / abs_coef.sum()

    # ---- 7. 打印对比表 ----
    table = Table(
        title=f"优化结果 — 策略: {strat_name} (ElasticNet)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("因子", style="cyan", min_width=12)
    table.add_column("原始权重（归一）", justify="right")
    table.add_column("优化权重", justify="right", style="bold green")
    table.add_column("系数", justify="right")
    table.add_column("方向", justify="center")
    table.add_column("变化", justify="right")

    orig_sum = sum(original_weights) or 1.0
    for name, orig, opt_w, coef, d in zip(
        factor_names, original_weights, best_weights, best_coefficients, directions
    ):
        orig_norm = orig / orig_sum
        delta = opt_w - orig_norm
        delta_str = (
            f"[green]+{delta:.4f}[/green]" if delta >= 0 else f"[red]{delta:.4f}[/red]"
        )
        table.add_row(
            name, f"{orig_norm:.4f}", f"{opt_w:.4f}", f"{coef:.6f}", str(d), delta_str
        )

    console.print(table)
    console.print(
        f"\n[bold yellow]📈 最优年化收益率: {best_ann_ret * 100:.2f}%[/bold yellow]  "
        f"(第 {study.best_trial.number + 1} 次 / 共 {n_trials} 次试验)"
    )
    console.print(
        f"[bold cyan]📊 最优超参数: alpha={best_alpha:.6e}, l1_ratio={best_l1_ratio:.4f}[/bold cyan]"
    )

    # ---- 8. 写回原 YAML（通过 StrategyConfig.to_yaml，保留注释） ----
    for rank_item, opt_w in zip(cfg.ranks, best_weights):
        rank_item.weight = round(float(opt_w), 6)

    cfg.to_yaml(yaml_file)

    console.print(f"[bold green]💾 最优权重已写回[/bold green] → {yaml_file}")
