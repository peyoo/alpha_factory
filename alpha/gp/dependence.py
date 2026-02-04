import numpy as np
import polars as pl
import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from loguru import logger
from typing import List, Dict, Optional, Tuple
from deap import tools


class DependenceManager:
    def __init__(self,
                 opt_names: Tuple[str, ...],
                 opt_weights: Tuple[float, ...],
                 cluster_threshold: float = 0.95,
                 penalty_factor: float = 0.8):
        self.threshold = cluster_threshold
        self.penalty_factor = penalty_factor
        self.opt_names = opt_names
        self.opt_weights = opt_weights

        self.perf_indices = [i for i, name in enumerate(opt_names)
                             if name not in ["complexity", "independence"]]
        self.perf_weights = [opt_weights[i] for i in self.perf_indices]

        self.anchor_df: Optional[pl.DataFrame] = None

        # 【统一字典】存储 表达式 -> 指纹Series
        # 在 evaluate 期间它会迅速膨胀，在 update_and_prune 时会被剪枝
        self.fingerprints_dict: Dict[str, pl.Series] = {}
        # 记录哪些 Key 是真正的精英，用于在聚类时区分“谁是雕像，谁是考生”
        self.elite_keys: List[str] = []
        self.elite_power_scores: Dict[str, float] = {}

    def _init_anchor_if_needed(self, df_output: pl.DataFrame):
        if self.anchor_df is not None:
            return
        self.anchor_df = df_output.select(["DATE", "ASSET"]).unique().sample(
            n=min(50000, df_output.height), seed=42
        ).sort(["DATE", "ASSET"])

    def _get_power_score(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for idx, weight in zip(self.perf_indices, self.perf_weights):
            name = self.opt_names[idx]
            score += metrics.get(name, 0.0) * weight
        return score

    # --- 调用点 1: 打分并寄存 ---
    def evaluate_independence(self, df_output: pl.DataFrame, exprs_list: List[Tuple], new_results: Dict) -> Dict:
        if not exprs_list:
            return {}
        self._init_anchor_if_needed(df_output)

        # 1. 采样并存入统一字典
        df_sampled = self.anchor_df.join(df_output, on=["DATE", "ASSET"], how="inner")

        current_batch_keys = []
        batch_power = {}
        for col_name, expr_obj, _ in exprs_list:
            expr_str = str(expr_obj)
            self.fingerprints_dict[expr_str] = df_sampled[col_name]  # 存入/覆盖
            current_batch_keys.append(expr_str)
            batch_power[expr_str] = self._get_power_score(new_results.get(expr_str, {}))

        try:
            # 2. 构造聚类矩阵 (只取 精英 + 本批次)
            # 这样既能保证新因子与历史精英比，又能保证新因子之间不重复
            compare_keys = self.elite_keys + current_batch_keys
            # 转换为计算矩阵
            matrix_df = pl.DataFrame({k: self.fingerprints_dict[k] for k in compare_keys})

            full_power_map = {**self.elite_power_scores, **batch_power}
            cluster_labels = self._run_fast_clustering(matrix_df, self.threshold)

            # 3. 簇内比武
            cluster_max_power = {}
            for col, label in cluster_labels.items():
                p = full_power_map.get(col, -np.inf)
                if label not in cluster_max_power or p > cluster_max_power[label]:
                    cluster_max_power[label] = p

            # 4. 打分
            scores = {}
            for expr_str in current_batch_keys:
                label = cluster_labels[expr_str]
                if batch_power[expr_str] >= cluster_max_power[label] - 1e-9:
                    scores[expr_str] = 1.0
                else:
                    scores[expr_str] = self.penalty_factor
            return scores
        except Exception as e:
            logger.error(f"聚类异常: {e}")
            return {str(item[1]): 1.0 for item in exprs_list}

    def _run_fast_clustering(self, df: pl.DataFrame, threshold: float):
        rank_df = df.fill_null(0).select([pl.col(c).rank() for c in df.columns])
        corr_array = np.nan_to_num(rank_df.corr().to_numpy(), nan=0.0)
        dist_matrix = np.sqrt(np.clip(2 * (1 - np.abs(corr_array)), 0, None))
        Z = fastcluster.linkage(squareform(dist_matrix), method='complete')
        return dict(zip(df.columns, fcluster(Z, t=np.sqrt(2 * (1 - threshold)), criterion='distance')))

    # --- 调用点 2: 剪枝更新 (核心变化) ---
    def update_and_prune(self, halloffame: tools.HallOfFame):
        """
        代末剪枝：签名保持不变。
        直接从个体身上提取 expr_str，将指纹从旧字典迁移至新字典。
        """
        before_count = len(self.fingerprints_dict)
        new_fingerprints = {}
        self.elite_keys = []
        self.elite_power_scores = {}

        for ind in halloffame:
            # 尝试获取挂载的标识符，若无（理论上不应发生）则回退
            expr_str = getattr(ind, 'expr_str', str(ind))

            if expr_str in self.fingerprints_dict:
                new_fingerprints[expr_str] = self.fingerprints_dict[expr_str]
                self.elite_keys.append(expr_str)

                # 同步精英的武力值以便下一代对比
                metrics = {name: val for name, val in zip(self.opt_names, ind.fitness.values)}
                self.elite_power_scores[expr_str] = self._get_power_score(metrics)

        self.fingerprints_dict = new_fingerprints
        logger.info(f"🧹 字典剪枝完成 | 原规模: {before_count} -> 精英数: {len(self.elite_keys)}")
