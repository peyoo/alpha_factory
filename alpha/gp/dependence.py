import numpy as np
import polars as pl
import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from loguru import logger
from typing import List, Dict, Optional, Tuple
from deap import tools


class DependenceManager:
    """
    GP 因子独立性管理器 (DependenceManager)

    核心机制 - 属性挂载协议 (Attribute Tagging Protocol):
    --------------------------------------------------
    本类高度依赖于 DEAP 个体 (Individual) 对象上挂载的 `expr_str` 属性。
    1. 协议要求：在进化流程的评估阶段（通常在 fill_fitness 函数中），必须执行:
       `ind.expr_str = str(simplified_expression)`
    2. 作用：该属性作为“唯一标识符”，将 DEAP 树对象与本管理器缓存的“因子指纹 (fingerprints_dict)”
       以及“简化表达式字符串”强行绑定。
    3. 优势：避免了代末剪枝时重复进行 Sympy 表达式简化计算，极大提升了多代进化下的运行效率。

    主要任务:
    1. 采样寄存：将批次计算出的因子值采样并存储在 fingerprints_dict。
    2. 独立打分：基于簇内武力值对比，对逻辑重复的因子进行降分处理。
    3. 代末剪枝：依据 HoF 成员的 `expr_str` 属性清理无效指纹，维持内存健康。
    """

    def __init__(self,
                 opt_names: Tuple[str, ...],
                 opt_weights: Tuple[float, ...],
                 cluster_threshold: float = 0.5,
                 penalty_factor: float = 0.5):
        """
        初始化配置。

        Args:
            opt_names: 优化目标名称列表 (需包含 'independence' 才能激活打分逻辑)。
            opt_weights: 优化目标权重，正数代表最大化，负数代表最小化。
            cluster_threshold: 聚类阈值 (0.95 代表相关性 > 0.95 的因子会被划分为同一逻辑簇)。
            penalty_factor: 惩罚因子，簇内表现不如冠军的因子将获得的独立性评价得分。
        """
        self.threshold = cluster_threshold
        self.penalty_factor = penalty_factor
        self.opt_names = opt_names
        self.opt_weights = opt_weights

        # 预先筛选绩效指标索引（排除复杂度、独立性等辅助指标），用于计算因子的综合“武力值”
        self.perf_indices = [i for i, name in enumerate(opt_names)
                             if name not in ["complexity", "independence"]]
        self.perf_weights = [opt_weights[i] for i in self.perf_indices]

        # 锚点数据框：用于确保所有因子的采样坐标（DATE, ASSET）完全一致，保证相关性计算的有效性
        self.anchor_df: Optional[pl.DataFrame] = None

        # 核心缓存字典: 简化表达式字符串 -> 采样后的因子值 Series (指纹)
        self.fingerprints_dict: Dict[str, pl.Series] = {}

        # 精英库维护：记录名人堂成员的标识符及其历史综合得分
        self.elite_keys: List[str] = []
        self.elite_power_scores: Dict[str, float] = {}

        logger.info(f"🚀 DependenceManager 初始化 | 阈值: {self.threshold} | 惩罚分: {self.penalty_factor}")

    def _init_anchor_if_needed(self, df_output: pl.DataFrame):
        """
        初始化采样锚点。固定 50,000 个随机样本点，在保证统计有效性的同时最大化计算速度。
        """
        if self.anchor_df is not None:
            return

        full_coords = df_output.select(["DATE", "ASSET"]).unique()
        sample_n = min(50000, full_coords.height)

        # 固定 seed=42 保证进化过程中不同批次的指纹提取具有严格可比性
        self.anchor_df = full_coords.sample(n=sample_n, seed=42).sort(["DATE", "ASSET"])
        logger.success(f"⚓ 独立性采样锚点已固定，样本规模: {sample_n} 行")

    def _get_power_score(self, metrics: Dict[str, float]) -> float:
        """根据配置权重计算因子的综合绩效得分（武力值）"""
        score = 0.0
        for idx, weight in zip(self.perf_indices, self.perf_weights):
            name = self.opt_names[idx]
            score += metrics.get(name, 0.0) * weight
        return score

    def evaluate_independence(self, df_output: pl.DataFrame, exprs_list: List[Tuple], new_results: Dict) -> Dict:
        """
        [调用点: 因子评估批处理阶段]
        对当前批次的因子进行独立性打分，并将其指纹寄存。

        接口说明:
            - df_output: 包含因子计算结果的 Polars DataFrame。
            - exprs_list: 包含 (因子名, 表达式对象, _) 的元组列表。
            - new_results: 绩效计算结果字典 {表达式字符串: {指标名: 指标值}}。

        注意:
            调用此函数后，外部必须确保 Individual 对象执行了属性挂载:
            `ind.expr_str = str(simplified_expression)`。
        """
        if not exprs_list:
            return {}

        self._init_anchor_if_needed(df_output)

        # 1. 采样并寄存指纹
        df_sampled = self.anchor_df.join(df_output, on=["DATE", "ASSET"], how="inner")

        current_batch_keys = []
        batch_power = {}
        for col_name, expr_obj, _ in exprs_list:
            expr_str = str(expr_obj)
            self.fingerprints_dict[expr_str] = df_sampled[col_name]
            current_batch_keys.append(expr_str)
            batch_power[expr_str] = self._get_power_score(new_results.get(expr_str, {}))

        try:
            # 2. 构造计算矩阵 (历史精英 + 当前批次)
            compare_keys = self.elite_keys + current_batch_keys
            matrix_df = pl.DataFrame({k: self.fingerprints_dict[k] for k in compare_keys})
            full_power_map = {**self.elite_power_scores, **batch_power}

            # 3. 执行快速层次聚类
            cluster_labels = self._run_fast_clustering(matrix_df, self.threshold)
            n_clusters = len(set(cluster_labels.values()))
            logger.debug(f"🔍 聚类完成 | 因子总数: {len(compare_keys)} | 逻辑簇数: {n_clusters}")

            # 4. 簇内比武：锁定每个逻辑类别的最高得分
            cluster_max_power = {}
            for col, label in cluster_labels.items():
                p = full_power_map.get(col, -np.inf)
                if label not in cluster_max_power or p > cluster_max_power[label]:
                    cluster_max_power[label] = p

            # 5. 计算最终独立性分数
            scores = {}
            for expr_str in current_batch_keys:
                label = cluster_labels[expr_str]
                # 只有簇内表现最好的因子能获得 1.0 (满分独立性)
                if batch_power[expr_str] >= cluster_max_power[label] - 1e-9:
                    scores[expr_str] = 1.0
                else:
                    scores[expr_str] = self.penalty_factor

            logger.debug(f"📊 独立性评估完成 | 批次因子: {len(current_batch_keys)} | 精英库规模: {len(self.elite_keys)}")
            return scores

        except Exception as e:
            logger.error(f"❌ 聚类过程异常: {e}")
            return {str(item[1]): 1.0 for item in exprs_list}

    def _run_fast_clustering(self, df: pl.DataFrame, threshold: float):
        """核心聚类逻辑：带数值稳定性修复"""
        # 使用秩变换计算相关性
        rank_df = df.fill_null(0).select([pl.col(c).rank() for c in df.columns])
        corr_array = np.nan_to_num(rank_df.corr().to_numpy(), nan=0.0)

        # 1. 距离转换: d = sqrt(2 * (1 - |rho|))
        dist_matrix = np.sqrt(np.clip(2 * (1 - np.abs(corr_array)), 0, None))

        # --- 核心修复：处理浮点数非对称问题 ---
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        # ------------------------------------

        # 使用 fastcluster 提升 linkage 效率
        Z = fastcluster.linkage(squareform(dist_matrix), method='complete')

        # 切割聚类树
        t_val = np.sqrt(2 * (1 - threshold))
        labels = fcluster(Z, t=t_val, criterion='distance')

        return dict(zip(df.columns, labels))

    def update_and_prune(self, halloffame: tools.HallOfFame):
        """
        [调用点: 进化循环代末]
        代末剪枝。根据 HoF 成员携带的 `expr_str` 属性进行指纹清理，防止内存爆炸。

        依赖说明:
            - halloffame 内部的个体对象必须已经过 fill_fitness 阶段的属性挂载。
        """
        before_count = len(self.fingerprints_dict)
        new_fingerprints = {}
        self.elite_keys = []
        self.elite_power_scores = {}

        for ind in halloffame:
            # 协议读取：尝试获取挂载的标识符
            expr_str = getattr(ind, 'expr_str', str(ind))

            if expr_str in self.fingerprints_dict:
                # 迁移精英指纹，未在 HoF 中的指纹将被 GC 自动回收
                new_fingerprints[expr_str] = self.fingerprints_dict[expr_str]
                self.elite_keys.append(expr_str)

                # 同步精英适应度，供下一代 evaluate 时进行簇内对比
                metrics = {name: val for name, val in zip(self.opt_names, ind.fitness.values)}
                self.elite_power_scores[expr_str] = self._get_power_score(metrics)

        self.fingerprints_dict = new_fingerprints
        logger.info(f"🧹 指纹字典剪枝完成: {before_count} -> {len(self.elite_keys)} (保留精英)")
