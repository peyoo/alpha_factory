import polars as pl
import polars.selectors as cs
import numpy as np
import fastcluster
from loguru import logger
from typing import Union, List, Optional, Any
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


def batch_clustering(
    df: Union[pl.DataFrame, pl.LazyFrame],
    factors: Union[str, List[str]] = r"^factor_.*",
    threshold: float = 0.8,
    method: str = "average",
    sample_n: Optional[int] = 50000,
) -> dict[str, int] | dict[Any, Any] | tuple[dict[str, int], dict[int, list]]:
    """
    因子聚类分析核心函数：识别并归类逻辑冗余的因子。

    该函数通过计算因子间的斯皮尔曼秩相关性 (Spearman Rank Correlation) 构建距离矩阵，
    并利用层次聚类算法将表现相似的因子划分为同一个簇。

    Args:
        df (Union[pl.DataFrame, pl.LazyFrame]): 包含因子数据的 Polars 对象。
        factors (Union[str, List[str]]): 因子列选择器，支持正则或名称列表。默认为 "factor_*"。
        threshold (float): 相关性聚类阈值 (0~1)。
            - 越高(如0.95)表示聚类越严格，只有极度相似的因子才会被合并。
            - 越低(如0.7)表示聚类越宽松。
        method (str): 层次聚类联动算法。
            - 'average': 组平均法 (UPGMA)，对金融数据噪声较稳健。
            - 'complete': 全联动，倾向于产生紧凑且直径小的簇。
        sample_n (Optional[int]): 采样行数。
            - 若为 None，则使用全量数据计算，或表示输入数据已被采样。
            - 若为 int，则在 collect 后随机采样 N 行，以加速相关性矩阵计算。

    Returns:
        name_to_cluster: 因子名与聚类标签的映射字典 {因子名: 簇ID}。簇ID相同的因子逻辑同质化较高。
        cluster_to_names: 聚类标签与因子名列表的映射字典 {簇ID: [因子名列表]}。

    """

    # 1. 统一转换为 LazyFrame 以复用执行计划优化器
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    selector = cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)

    # 2. 提取因子列并执行采样 (仅针对 LazyFrame 优化的内存路径)
    try:
        if sample_n is not None:
            # 仅提取选中的因子列，避免 collect 无关列导致 OOM
            factor_df = lf.select(selector).collect()
            actual_n = min(sample_n, factor_df.height)
            factor_df = factor_df.sample(n=actual_n, seed=42)
            logger.info(
                f"📊 因子聚类采样：已从 {lf.select(pl.len()).collect().item()} 行中抽取 {actual_n} 行"
            )
        else:
            factor_df = lf.select(selector).collect()
            logger.info(f"📊 因子聚类全量计算：共计 {factor_df.height} 行数据")
    except Exception as e:
        logger.error(f"❌ 数据提取失败: {e}")
        return {}, {}

    # 3. 边界条件检查
    if factor_df.width < 1:
        logger.warning("⚠️ 未匹配到任何因子列，请检查 selector 参数")
        return {}, {}

    if factor_df.width == 1:
        logger.info("ℹ️ 仅有一个因子，跳过聚类")
        return {factor_df.columns[0]: 1}, {1: [factor_df.columns[0]]}

    # 4. 数据预处理：剔除方差过小的常数因子
    stats = factor_df.std()
    valid_cols = [
        col for col in factor_df.columns if (stats.get_column(col)[0] or 0) > 1e-9
    ]
    invalid_count = factor_df.width - len(valid_cols)
    if invalid_count > 0:
        logger.warning(f"🚫 已剔除 {invalid_count} 个常数因子 (方差 ≈ 0)")

    if not valid_cols:
        return {col: 0 for col in factor_df.columns}

    # 5. 清理 Null 值 (Spearman Rank 对 Null 敏感)
    cleaned_df = factor_df.select(valid_cols).drop_nulls()
    if cleaned_df.height < 10:
        logger.error(f"❌ 有效行数不足 ({cleaned_df.height})，无法计算相关性")
        return {col: i + 1 for i, col in enumerate(factor_df.columns)}

    # 6. 计算斯皮尔曼秩相关性 (Spearman Rank Correlation)
    # 使用秩变换 (rank) 后计算皮尔逊相关系数，即为斯皮尔曼相关系数
    logger.debug(f"🔍 正在计算 {len(valid_cols)} 个因子的相关性矩阵...")
    corr_matrix = (
        cleaned_df.select([pl.col(c).rank() for c in valid_cols]).corr().to_numpy()
    )
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 7. 构建距离矩阵 (Distance Matrix)
    # 距离定义：d = 1 - |rho|。高度正相关或负相关的因子距离都接近 0
    dist_matrix = np.clip(1 - np.abs(corr_matrix), 0, 1)

    # 关键步骤：强制对称化以消除浮点数精度引起的 Scipy 报错
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    # 8. 执行层次聚类
    try:
        # 将对称方阵压缩为 1D 距离向量（squareform 要求的输入格式）
        dist_vec = squareform(dist_matrix)

        # 使用 fastcluster 进行计算（比 scipy 原生快）
        Z = fastcluster.linkage(dist_vec, method=method)

        # 切割聚类树，得到标签。t = 1 - threshold 为切割高度
        labels = fcluster(Z, t=1 - threshold, criterion="distance")
    except Exception as e:
        logger.error(f"❌ 层次聚类算法崩溃: {e}")
        return {col: i + 1 for i, col in enumerate(factor_df.columns)}

    # --- 9. 构造最终双向映射结果 ---
    from collections import defaultdict

    # 映射 A: {因子名: 簇ID}
    name_to_cluster = {col: 0 for col in factor_df.columns}
    # 映射 B: {簇ID: List[因子名]}
    cluster_to_names = defaultdict(list)

    # 首先处理参与聚类的有效因子
    for col, label in zip(valid_cols, labels):
        cid = int(label)
        name_to_cluster[col] = cid
        cluster_to_names[cid].append(col)

    # 处理被剔除的常数因子（统一归为簇 0）
    invalid_cols = [c for c in factor_df.columns if c not in valid_cols]
    if invalid_cols:
        cluster_to_names[0] = invalid_cols
        # name_to_cluster 默认值已为 0，无需重复赋值

    # --- 10. 打印聚类摘要日志 ---
    num_clusters = len(set(labels))
    logger.success(
        f"✅ 因子聚类完成 | 总数: {len(factor_df.columns)} | 逻辑簇: {num_clusters} | 常数因子: {len(invalid_cols)}"
    )

    return name_to_cluster, dict(cluster_to_names)
