import polars as pl
from loguru import logger

from alpha_factory.data_provider import DataProvider
from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.data_provider.utils import extract_expressions_from_csv
from alpha_factory.evaluation.batch.cluster import batch_clustering
from alpha_factory.evaluation.batch.full_metrics import batch_full_metrics
from alpha_factory.gp.extra_terminal import add_extra_terminals
from alpha_factory.gp.label import label_OO_for_tradable, label_OO_for_IC
from alpha_factory.patch.expr_codegen_patch import apply_expr_codegen_patches
from alpha_factory.utils.config import settings
from alpha_factory.utils.schema import F


def main():
    # 1. 路径与参数设置
    input_path = settings.OUTPUT_DIR / 'main_small_pool' / 'best_factors.csv'
    output_path = settings.OUTPUT_DIR / 'main_small_pool' / 'factors_full_report.csv'
    refined_output_path = settings.OUTPUT_DIR / 'main_small_pool' / 'refined_top_factors.csv'

    # 2. 提取表达式
    exprs = extract_expressions_from_csv(input_path)
    if not exprs: return

    factor_expr_map = {f"factor_{i + 1}": expr for i, expr in enumerate(exprs)}
    factor_names = list(factor_expr_map.keys())
    # 构建加载列，包含表达式定义
    exprs_with_names = [f"{name}={expr}" for name, expr in factor_expr_map.items()]

    logger.info(f"🚀 启动增强型全维度评估，共 {len(exprs)} 条因子")
    apply_expr_codegen_patches()

    # 3. 加载数据
    # 注意：增强版函数需要 LABEL_FOR_RET 和 POOL_MASK
    needed_columns = [*exprs_with_names, F.LABEL_FOR_IC, F.LABEL_FOR_RET, F.POOL_MASK]
    lf = DataProvider().load_data(
        start_date="20190101",
        end_date="20251231",
        funcs=[main_small_pool, add_extra_terminals, label_OO_for_IC, label_OO_for_tradable],
        column_exprs=needed_columns,
        lookback_window=200
    )

    # 4. 执行计算 (Collect)
    logger.info("📡 执行并行计算与数据采集...")
    df_calculated = lf.collect()

    # 5. 因子聚类分析 (去重基石)
    # 基于你设定的阈值 0.8 [cite: 2026-02-04]
    logger.info("🌿 正在计算因子聚类 (Threshold=0.8)...")
    cluster_mapping = batch_clustering(
        df=df_calculated,
        factors=factor_names,
        threshold=0.8,
        method="average"
    )
    if isinstance(cluster_mapping, tuple):
        cluster_mapping = cluster_mapping[0]

    # 6. 一站式增强评估 (取代旧的 IC/Turnover/Returns 三个函数)
    # 这里直接集成了换手扣费 (15bps)
    logger.info("📊 执行增强型收益评估 (集成 IC + 换手扣费)...")
    report_data = batch_full_metrics(
        df=df_calculated,
        factors=factor_names,
        label_ret_col=F.LABEL_FOR_RET,
        fee=0.0015,  # 设置单边 15bps 的交易摩擦
        mode='long_only'
    )

    # 7. 报表合并与初步格式化
    final_report = (
        report_data
        .with_columns([
            # 注入聚类 ID 和 原始公式
            pl.col("factor").replace(cluster_mapping).cast(pl.Int32).alias("cluster_id"),
            pl.col("factor").replace(factor_expr_map).alias("expression"),
            pl.col(pl.Float64).round(4)
        ])
        .select([
            "cluster_id", "factor", "ic_ir", "ann_ret", "sharpe", "turnover_est", "expression"
        ])
        .sort(by=["cluster_id", "sharpe"], descending=[False, True])
    )

    # 8. 自动化精选：每簇取前两个“优等生”
    # 逻辑：在每个逻辑簇内，选择扣费后 Sharpe 最高的前 2 名
    refined_report = (
        final_report
        .filter(
            (pl.col("sharpe") > 0.3) &  # 扣费后 Sharpe 至少要为正且具备基本意义
            (pl.col("ic_ir").abs() > 0.05)
        )
        .group_by("cluster_id")
        .head(2)
        .sort(by="sharpe", descending=True)
    )

    # 9. 保存结果与日志输出
    final_report.write_csv(output_path)
    refined_report.write_csv(refined_output_path)

    logger.success(f"🎊 增强型分析完成！结果已写入: {output_path}")
    logger.info(f"原样本: {len(final_report)} | 扣费并每簇选二后剩余: {len(refined_report)}")

    # 10. 展示精选名单预览
    with pl.Config(fmt_str_lengths=50, tbl_rows=20, tbl_width_chars=160):
        print("\n" + "=" * 140)
        print("💎 REFINED TOP ALPHA (Top 2 per Cluster | Fee: 15bps | Threshold: 0.8)")
        print("-" * 140)
        print(refined_report)
        print("=" * 140 + "\n")


if __name__ == "__main__":
    main()
