"""eval_core.py — 批量因子评估的核心流程（与 CLI 独立）

提供独立的评估管道，将 YAML 配置（StrategyConfig）与可选的 CSV 因子文件
转换为评估结果。支持 IC Decay 和 Turnover Decay 计算。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Optional

import polars as pl
from loguru import logger
from rich.console import Console

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.evaluation.batch.cluster import batch_clustering
from alpha_factory.evaluation.batch.full_metrics import batch_full_metrics
from alpha_factory.evaluation.batch.ic_decay import batch_calc_factor_ic_decay
from alpha_factory.evaluation.batch.turnover_decay import (
    batch_calc_factor_turnover_with_direction,
)

console = Console()


def _get_pool_universe(pool_name: str) -> PoolUniverse:
    """从配置中的池名称获取 PoolUniverse 对象。

    Args:
        pool_name: 池名称（如 "main_small_pool"）

    Returns:
        PoolUniverse 对象

    Raises:
        ValueError: 如果池名称无法识别
    """
    # 尝试从 PoolUniverseEnum 中找到匹配的枚举值
    for enum_member in PoolUniverseEnum:
        # enum_member.value 返回 PoolUniverse 对象
        pool_obj = enum_member.value()
        if pool_obj.name == pool_name:
            return pool_obj

    # 默认返回 main_small 池
    logger.warning(f"无法识别池名称 '{pool_name}'，使用默认池 'main_small_pool'")
    return PoolUniverseEnum.main_small.value()


def _collect_factors(
    config: StrategyConfig,
    csv_factors: Optional[List[tuple[str, str]]] = None,
) -> List[tuple[str, str]]:
    """从 StrategyConfig 的 ranks 和可选的 CSV 因子文件收集因子列表。

    Args:
        config: StrategyConfig 对象
        csv_factors: 可选的 (name, expression) 对列表，通常来自 CSV 文件

    Returns:
        (name, expression) 对的列表，CSV 因子优先于 config.ranks
    """
    # 先收集 YAML 中的 ranks
    yaml_factors = [
        (rank.name or f"rank_f{i}", rank.expression)
        for i, rank in enumerate(config.ranks, start=1)
    ]

    # 如果提供了 CSV 因子，使用 CSV 因子覆盖或补充 YAML 因子
    if csv_factors:
        # 创建 CSV 因子的字典（name -> expression）
        csv_dict = {name: expr for name, expr in csv_factors}

        # 保留 YAML 中不在 CSV 中的因子
        result = [(name, expr) for name, expr in yaml_factors if name not in csv_dict]

        # 添加 CSV 因子
        result.extend(csv_factors)
        return result
    else:
        return yaml_factors


def _load_data(
    config: StrategyConfig,
    factor_pairs: List[tuple[str, str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.LazyFrame:
    """加载评估所需的数据。

    Args:
        config: StrategyConfig 对象
        factor_pairs: (name, expression) 对列表
        start_date: 可选的开始日期（优先级高于 config.start_date）
        end_date: 可选的结束日期（优先级高于 config.end_date）

    Returns:
        pl.LazyFrame: 包含因子列和标签列的惰性数据框
    """
    # 使用命令行参数覆盖配置文件中的日期
    actual_start = start_date or config.start_date or "20190101"
    actual_end = end_date or config.end_date

    logger.info(f"加载数据: pool={config.pool}, start={actual_start}, end={actual_end}")

    # 将池名称转换为 PoolUniverse 对象
    pool_universe = _get_pool_universe(config.pool)

    dp = DataProvider()
    # 将 (name, expression) 对转换为 "name=expression" 格式
    exprs_for_loader = [f"{name}={expression}" for name, expression in factor_pairs]

    lf = dp.load_pool_data(
        pool_universe,
        actual_start,
        actual_end,
        exprs=exprs_for_loader,
    )
    return lf


def _run_batch_eval(
    lf: pl.LazyFrame,
    factors: List[str],
    config: StrategyConfig,
    batch_size: int = 100,
) -> pl.DataFrame:
    """执行批量评估。

    Args:
        lf: 包含因子数据的 LazyFrame
        factors: 要评估的因子名称列表
        config: StrategyConfig 对象（提供 ls_mode, cost, n_bins 等参数）
        batch_size: 单次评估的最大因子数

    Returns:
        pl.DataFrame: 评估结果
    """
    logger.info(f"开始批量评估 {len(factors)} 个因子，batch_size={batch_size}")

    batches = [factors[i : i + batch_size] for i in range(0, len(factors), batch_size)]
    result_parts: list[pl.DataFrame] = []

    for idx, batch_factors in enumerate(batches, start=1):
        batch_start_ts = perf_counter()
        logger.info(f"批次 {idx}/{len(batches)}: 评估 {len(batch_factors)} 个因子")

        part_df = batch_full_metrics(
            lf,
            factors=batch_factors,
            n_bins=config.n_bins,
            mode=config.ls_mode,
            fee=config.cost,
        )

        batch_elapsed = perf_counter() - batch_start_ts
        logger.info(
            f"  完成，耗时 {batch_elapsed:.3f}s，"
            f"单因子 {batch_elapsed / max(len(batch_factors), 1):.3f}s"
        )

        if not part_df.is_empty():
            result_parts.append(part_df)

    result_df = (
        pl.concat(result_parts, how="vertical_relaxed")
        if result_parts
        else pl.DataFrame()
    )

    return result_df


def _compute_ic_decay(
    lf: pl.LazyFrame,
    factors: List[str],
    max_lag: int = 5,
) -> pl.DataFrame:
    """计算因子的 IC Decay。

    Args:
        lf: 包含因子数据的 LazyFrame
        factors: 要计算 IC Decay 的因子名称列表
        max_lag: 最大滞后期数

    Returns:
        pl.DataFrame: IC Decay 结果
    """
    logger.info(f"计算 IC Decay (max_lag={max_lag}) 对 {len(factors)} 个因子...")
    # 注意：batch_calc_factor_ic_decay 需要显式的因子列表，不能使用默认的 factor_* 模式
    decay_df = batch_calc_factor_ic_decay(lf, factors=factors, max_lag=max_lag)
    logger.info("IC Decay 计算完成")
    return decay_df


def _compute_clustering(
    lf: pl.LazyFrame,
    factors: List[str],
    threshold: float = 0.7,
) -> dict[str, int]:
    """计算因子层次聚类，返回各因子所属簇 ID。

    Args:
        lf: 包含因子数据的 LazyFrame
        factors: 要聚类的因子名称列表
        threshold: 相关性聚类阈值（0~1），越高聚类越严格

    Returns:
        name_to_cluster: {因子名: 簇ID} 映射字典
    """
    logger.info(f"计算因子聚类 (threshold={threshold}) 对 {len(factors)} 个因子...")
    name_to_cluster, _ = batch_clustering(lf, factors=factors, threshold=threshold)
    n_clusters = len(set(name_to_cluster.values()))
    logger.info(f"聚类完成，共 {n_clusters} 个簇")
    return name_to_cluster


def _compute_turnover_decay(
    lf: pl.LazyFrame,
    factors: List[str],
    n_bins: int = 10,
    lag: int = 1,
) -> pl.DataFrame:
    """计算因子的 Turnover Decay（根据 IC 方向选择 Top/Btm）。

    Args:
        lf: 包含因子数据的 LazyFrame
        factors: 要计算 Turnover Decay 的因子名称列表
        n_bins: 分桶数量
        lag: 滞后期数

    Returns:
        pl.DataFrame: Turnover Decay 结果
    """
    logger.info(
        f"计算 Turnover Decay (n_bins={n_bins}, lag={lag}) 对 {len(factors)} 个因子..."
    )
    from alpha_factory.utils.schema import F

    # 在 LazyFrame 阶段过滤列，避免提前 collect 全量数据
    required_cols = [F.DATE, F.ASSET, F.LABEL_FOR_RET] + factors
    schema_names = lf.collect_schema().names()
    existing_cols = [c for c in required_cols if c in schema_names]
    lf_subset = lf.select(existing_cols)

    turnover_df = batch_calc_factor_turnover_with_direction(
        lf_subset, factors=factors, n_bins=n_bins, lag=lag
    )
    logger.info("Turnover Decay 计算完成")
    return turnover_df


def _merge_results(
    main_df: pl.DataFrame,
    factor_pairs: List[tuple[str, str]],
    decay_df: Optional[pl.DataFrame] = None,
    turnover_df: Optional[pl.DataFrame] = None,
    cluster_map: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """合并评估结果与表达式、IC Decay、Turnover Decay、聚类标签等。

    Args:
        main_df: 主评估结果
        factor_pairs: (name, expression) 对列表
        decay_df: 可选的 IC Decay 结果
        turnover_df: 可选的 Turnover Decay 结果
        cluster_map: 可选的 {因子名: 簇ID} 映射字典

    Returns:
        合并后的结果 DataFrame
    """
    # 添加表达式列
    expr_map = pl.DataFrame(
        {
            "factor": [name for name, _ in factor_pairs],
            "expression": [expr for _, expr in factor_pairs],
        }
    )
    result_df = main_df.join(expr_map, on="factor", how="left").select(
        ["factor", "expression", *[c for c in main_df.columns if c != "factor"]]
    )

    # 如果有 IC Decay 结果，进行联接
    if decay_df is not None and not decay_df.is_empty():
        # 从 decay_df 选择需要的列，避免列并集超出预期
        decay_cols = [c for c in decay_df.columns if c not in ["factor"]]
        if decay_cols:
            decay_subset = decay_df.select(["factor"] + decay_cols)
            result_df = result_df.join(decay_subset, on="factor", how="left")

    # 如果有 Turnover Decay 结果，进行联接
    if turnover_df is not None and not turnover_df.is_empty():
        # 排除所有已存在于 result_df 的列（避免列名冲突），只保留 turnover 专属列
        existing = set(result_df.columns)
        turnover_cols = [
            c for c in turnover_df.columns if c == "factor" or c not in existing
        ]
        if len(turnover_cols) > 1:  # 至少有 factor + 一列有效数据
            turnover_subset = turnover_df.select(turnover_cols)
            result_df = result_df.join(turnover_subset, on="factor", how="left")

    # 如果有聚类结果，附加 cluster 列
    if cluster_map:
        cluster_series = pl.Series(
            "cluster",
            [cluster_map.get(name, 0) for name in result_df["factor"].to_list()],
        )
        result_df = result_df.with_columns(cluster_series)

    return result_df


def run_eval_pipeline(
    config: StrategyConfig,
    csv_factors: Optional[List[tuple[str, str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    batch_size: int = 100,
    output_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """完整的评估管道。

    Args:
        config: StrategyConfig 对象
        csv_factors: 可选的 CSV 因子列表，格式为 (name, expression) 对
        start_date: 可选的开始日期，优先级高于 config.start_date
        end_date: 可选的结束日期，优先级高于 config.end_date
        batch_size: 评估批大小
        output_dir: 可选的输出目录

    Returns:
        pl.DataFrame: 完整的评估结果（已排序）
    """
    eval_start_ts = perf_counter()
    logger.info("开始因子评估管道")

    # 1. 收集因子
    logger.info("步骤 1：收集因子列表")
    factor_pairs = _collect_factors(config, csv_factors)
    factor_names = [name for name, _ in factor_pairs]

    if not factor_pairs:
        logger.error("未找到任何因子，请检查配置")
        raise ValueError("No factors found in config or CSV")

    logger.info(f"总共 {len(factor_pairs)} 个因子: {', '.join(factor_names)}")

    # 2. 加载数据
    logger.info("步骤 2：加载评估数据")
    lf = _load_data(config, factor_pairs, start_date, end_date)

    # 3. 运行批量评估
    logger.info("步骤 3：执行批量评估")
    result_df = _run_batch_eval(lf, factor_names, config, batch_size)

    if result_df.is_empty():
        logger.warning("评估结果为空，请检查数据范围或因子表达式")
        raise ValueError("Evaluation result is empty")

    # 4. 可选的 IC Decay 计算
    decay_df = None
    if config.ic_decay:
        logger.info("步骤 4a：计算 IC Decay")
        try:
            decay_df = _compute_ic_decay(lf, factor_names)
            if decay_df.is_empty():
                logger.warning("IC Decay 计算返回空结果，将跳过")
                decay_df = None
        except Exception as e:
            logger.warning(f"IC Decay 计算出错: {e}，将跳过")
            decay_df = None

    # 4b. 可选的 Turnover Decay 计算
    turnover_df = None
    if config.turnover_decay:
        logger.info("步骤 4b：计算 Turnover Decay")
        try:
            turnover_df = _compute_turnover_decay(
                lf, factor_names, n_bins=config.n_bins, lag=1
            )
            if turnover_df.is_empty():
                logger.warning("Turnover Decay 计算返回空结果，将跳过")
                turnover_df = None
        except Exception as e:
            logger.warning(f"Turnover Decay 计算出错: {e}，将跳过")
            turnover_df = None

    # 4c. 可选的聚类计算
    cluster_map = None
    if config.cluster:
        logger.info("步骤 4c：计算因子聚类")
        try:
            cluster_map = _compute_clustering(
                lf, factor_names, config.relevance_threshold
            )
        except Exception as e:
            logger.warning(f"聚类计算出错: {e}，将跳过")
            cluster_map = None

    # 5. 合并结果
    logger.info("步骤 5：合并结果")
    result_df = _merge_results(
        result_df, factor_pairs, decay_df, turnover_df, cluster_map
    )

    # 6. 排序（按 sharpe 降序）
    if "sharpe" in result_df.columns:
        result_df = result_df.sort("sharpe", descending=True)

    total_eval_seconds = perf_counter() - eval_start_ts
    logger.info(
        f"管道完成，总耗时 {total_eval_seconds:.3f}s, 评估因子数 {len(factor_names)}"
    )

    # 可选的输出保存
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir / f"eval_result_{int(perf_counter() * 1000) % 100000}.csv"
        )
        result_df.write_csv(output_file)
        logger.info(f"结果已保存到: {output_file}")

    return result_df


__all__ = [
    "run_eval_pipeline",
    "_collect_factors",
    "_load_data",
    "_run_batch_eval",
    "_compute_ic_decay",
    "_compute_clustering",
    "_merge_results",
]
