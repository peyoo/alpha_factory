"""
供以后研究使用
通过聚类动态生成股票池（POOL_MASK）
"""

import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from alpha_factory.utils.schema import F


def main_cluster_pool(
    lf: pl.LazyFrame, n_clusters: int = 5, target_cluster_idx: int = 0
) -> pl.LazyFrame:
    """
    通过聚类动态生成 POOL_MASK
    1. 特征选取：如市值(MV)、换手率(Turnover)、波动率(Volatility)
    2. 聚类：在每个截面(DATE)上寻找特征相似的股票集群
    """

    def apply_clustering(df: pl.DataFrame) -> pl.DataFrame:
        # 1. 提取聚类特征
        features = ["TOTAL_MV", "TURNOVER", "VOLATILITY"]
        X = df.select(features).to_numpy()

        # 2. 标准化（sklearn 聚类对量纲敏感）
        X_scaled = StandardScaler().fit_transform(X)

        # 3. 聚类执行
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        return df.with_columns(pl.Series("cluster_id", clusters))

    # 在 LazyFrame 中按日期分组应用聚类逻辑
    # 注意：这将触发 Eager 计算，性能开销比原生 Polars 算子大
    lf_with_clusters = lf.group_by("DATE").map_groups(apply_clustering)

    # 4. 生成 MASK：选取你感兴趣的那个簇
    # 比如我们发现 cluster_id == 0 总是代表小市值、高活跃标的
    return lf_with_clusters.with_columns(
        (pl.col("cluster_id") == target_cluster_idx).alias(F.POOL_MASK)
    )
