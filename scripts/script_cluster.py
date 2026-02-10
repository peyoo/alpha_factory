import numpy as np
import polars as pl
from loguru import logger
from typing import Dict

from alpha_factory.data_provider import DataProvider
from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.data_provider.utils import extract_expressions_from_csv
from alpha_factory.evaluation.batch.cluster import batch_clustering
from alpha_factory.evaluation.batch.returns import batch_quantile_returns
from alpha_factory.gp.extra_terminal import add_extra_terminals
from alpha_factory.gp.label import label_OO_for_tradable
from alpha_factory.utils.config import settings
from alpha_factory.utils.schema import F


def main():
    # 1. 设置路径与提取表达式
    path = settings.OUTPUT_DIR / "gp" / "SmallCSGenerator" / "best_factors.csv"
    exprs = extract_expressions_from_csv(path)

    # 创建因子名与表达式的映射字典
    factor_expr_map: Dict[str, str] = {
        f"factor_{i + 1}": expr for i, expr in enumerate(exprs)
    }
    # 用于添加到数据管道
    exprs_with_names = [f"factor_{i + 1}={expr}" for i, expr in enumerate(exprs)]

    logger.info(f"🚀 从 CSV 提取并准备计算 {len(exprs)} 条因子表达式")

    # 2. 加载数据并计算因子
    lf = DataProvider().load_data(
        start_date="20190101",
        end_date="20251231",
        funcs=[main_small_pool, add_extra_terminals, label_OO_for_tradable],
        column_exprs=[f"{F.LABEL_FOR_RET}=OPEN[-2] / OPEN[-1] - 1", *exprs_with_names],
        lookback_window=200,
    )

    # 3. 批量绩效评估
    logger.info("📊 正在进行因子绩效评估...")
    df_result = batch_quantile_returns(lf)

    # 后处理：添加 expression 列并排序列
    df_result = df_result.with_columns(
        pl.col("factor")
        .map_elements(
            lambda f: factor_expr_map.get(f, "unknown"), return_dtype=pl.String
        )
        .alias("expression")
    ).select(
        [
            "factor",
            "expression",
            *[col for col in df_result.columns if col not in ["factor", "expression"]],
        ]
    )

    # 4. 逻辑聚类
    logger.info("🔍 正在进行因子逻辑聚类 (采样 50,000 行)...")
    cluster_dict = batch_clustering(lf, sample_n=50000)

    # 5. Cluster 分组统计信息
    logger.info("📈 正在生成 Cluster 分组统计...")

    # 转换聚类字典为 DataFrame
    df_clusters = pl.DataFrame(
        {"factor": list(cluster_dict.keys()), "cluster_id": list(cluster_dict.values())}
    )

    # 合并绩效与聚类 ID
    df_merged = df_result.join(df_clusters, on="factor")

    # 按 Cluster 分组：统计因子数、最高夏普、并选出最强因子的 ID
    df_cluster_stats = (
        df_merged.group_by("cluster_id")
        .agg(
            [
                pl.count("factor").alias("因子数量"),
                pl.max("sharpe").alias("最高夏普"),
                # 找到夏普最高的那个因子的 ID
                pl.col("factor").sort_by("sharpe").last().alias("最强因子ID"),
                # 找到夏普最高的那个因子的 表达式
                pl.col("expression").sort_by("sharpe").last().alias("最强因子逻辑"),
            ]
        )
        .sort("最高夏普", descending=True)
    )

    # 打印 Cluster 统计表
    print("\n" + "=" * 60 + " Cluster 分组统计信息 " + "=" * 60)
    with pl.Config(fmt_str_lengths=100, tbl_rows=20):
        print(df_cluster_stats)
    print("=" * 140 + "\n")

    # 6. 相关性分析
    logger.info("🧪 正在计算各 Cluster 最强代表因子之间的相关性...")

    # 获取每个簇最强因子的 ID 列表
    best_factor_ids = df_cluster_stats["最强因子ID"].to_list()

    # 从 LazyFrame 中提取这些因子的数据并计算相关性
    # 采样 20000 行足以代表截面相关性
    df_corr_data = (
        lf.select(best_factor_ids).collect().sample(n=min(20000, 50000)).to_pandas()
    )
    corr_matrix = df_corr_data.corr()

    # 打印相关性矩阵
    print(
        "\n" + "=" * 50 + " 各 Cluster 族长相关性矩阵 (Cross-Correlation) " + "=" * 50
    )
    print(corr_matrix.round(2))

    # 计算整体相关性指标
    n = len(corr_matrix)
    if n > 1:
        # 使用 np.triu_indices 提取上三角（不含对角线）
        upper_indices = np.triu_indices(n, k=1)
        corr_values = corr_matrix.values[upper_indices]
        mean_corr = corr_values.mean()
        max_corr = corr_values.max()

        print("-" * 118)
        logger.info(f"💡 簇间代表因子平均相关性: {mean_corr:.4f}")
        logger.info(f"🔥 最大簇间相关性: {max_corr:.4f}")

        if mean_corr < 0.3:
            logger.success("🚀 结论：因子池逻辑分散度极高，具备极强的组合潜力！")
        else:
            logger.warning("⚠️ 结论：部分 Cluster 之间仍存在一定相关性。")
    print("=" * 118 + "\n")


if __name__ == "__main__":
    main()
