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

    重构重点:
    1. 采集与评估解耦: register_fingerprints (计算层) vs calculate_contextual_independence (评估层)。
    2. 动态打分机制: 每一代全量重算独立性，彻底解决由于缓存命中导致的“克隆体霸榜”问题。
    3. 聚类引擎剥离: _run_fast_clustering 负责纯粹的数学计算。
    """

    def __init__(self,
                 opt_names: Tuple[str, ...],
                 opt_weights: Tuple[float, ...],
                 cluster_threshold: float = 0.9,
                 penalty_factor: float = -0.1):
        # 按照用户记忆，threshold 默认为 0.8
        self.threshold = cluster_threshold
        self.penalty_factor = penalty_factor
        self.opt_names = opt_names
        self.opt_weights = opt_weights

        # 筛选绩效指标索引，用于计算综合“武力值” (如 IC, Returns)
        self.perf_indices = [i for i, name in enumerate(opt_names)
                             if name not in ["complexity", "independence"]]
        self.perf_weights = [opt_weights[i] for i in self.perf_indices]

        # 采样锚点数据
        self.anchor_df: Optional[pl.DataFrame] = None

        # 指纹缓存: 表达式字符串 -> 采样后的 numpy 数组
        self.fingerprints_dict: Dict[str, np.ndarray] = {}

        # 精英库：记录上一代 HoF 的成员标识及其历史综合得分
        self.elite_keys: List[str] = []
        self.elite_power_scores: Dict[str, float] = {}

        logger.info(f"🚀 DependenceManager 初始化 | 阈值: {self.threshold} | 惩罚: {self.penalty_factor}")

    def _init_anchor_if_needed(self, df_output: pl.DataFrame):
        """初始化采样锚点 (固定 50,000 点)"""
        if self.anchor_df is not None:
            return
        full_coords = df_output.select(["DATE", "ASSET"]).unique()
        sample_n = min(50000, full_coords.height)
        self.anchor_df = full_coords.sample(n=sample_n, seed=42).sort(["DATE", "ASSET"])
        logger.success(f"⚓ 独立性采样锚点已固定: {sample_n} 行")

    def _get_power_score(self, metrics: Dict[str, float]) -> float:
        """根据配置权重计算综合绩效得分"""
        score = 0.0
        for idx, weight in zip(self.perf_indices, self.perf_weights):
            name = self.opt_names[idx]
            score += metrics.get(name, 0.0) * weight
        return score

    # --- 阶段 1: 指纹采集 (在 batched_exprs 计算层调用) ---

    def register_fingerprints(self, df_output: pl.DataFrame, expr_batch_info: List[Tuple]):
        """
        Args:
            df_output: 包含计算结果的数据框 (列名为临时的因子名)
            expr_batch_info: 传入 [(因子名, 表达式对象, ...), ...]
        """
        self._init_anchor_if_needed(df_output)

        # 1. 采样对齐
        df_sampled = self.anchor_df.join(df_output, on=["DATE", "ASSET"], how="inner")

        # 2. 建立映射并存储
        for col_name, expr_obj, _ in expr_batch_info:
            # 【关键】生成全局唯一的 Key
            expr_str = str(expr_obj)

            # 如果这个表达式已经在指纹库了（比如不同个体变异出了相同表达式），就不用再存一遍
            if expr_str not in self.fingerprints_dict:
                try:
                    # 1. 提取 Series
                    series = df_sampled.select(pl.col(col_name)).to_series()

                    # 2. 处理极端值 (关键步骤)
                    # 使用 np.nan_to_num 将 inf 转为 float32 的最大值，nan 转为 0
                    arr = series.to_numpy()
                    arr = np.nan_to_num(arr, nan=0.0, posinf=3e38, neginf=-3e38)

                    # 3. 【关键】增加一步剪切，限制在 float32 的安全范围内
                    # 使用 np.clip 确保数值不会在 cast 时溢出
                    f32_max = np.finfo(np.float32).max
                    f32_min = np.finfo(np.float32).min
                    arr = np.clip(arr, f32_min, f32_max)

                    # 4. 转换为 float32 并存储
                    self.fingerprints_dict[expr_str] = arr.astype(np.float32)

                except Exception as e:
                    logger.error(f"提取指纹失败: {col_name} | {e}")

    # --- 阶段 2: 动态评价 (在 fill_fitness 评估层调用) ---
    def calculate_contextual_independence(self, exprs_list: List[str], current_results: Dict) -> List[float]:
        """
        [终极进攻版]
        1. 物理层：全员默认 0.1，彻底封杀克隆体。
        2. 参赛权：仅限‘有指纹’的个体（当前新因子 + 上榜老精英）。
        3. 规则：簇内武力值冠军拿 1.0，不看资历，只看强弱。
        4. 档案：缓存中无指纹的因子直接维持 0.1，不给翻身机会。
        """
        # 1. 初始化全员惩罚
        scores_list = [self.penalty_factor] * len(exprs_list)

        # 2. 建立位置映射，锁定每个逻辑的“首发代理人”
        expr_to_indices = {}
        for idx, expr in enumerate(exprs_list):
            if expr not in expr_to_indices:
                expr_to_indices[expr] = []
            expr_to_indices[expr].append(idx)

        all_to_cluster = []  # 真正进入聚类竞技场的名单
        batch_power = {}

        # 3. 筛选参赛者
        for expr_str, indices in expr_to_indices.items():
            first_idx = indices[0]  # 只取第一个位置代表该逻辑

            # 【关键】只有具备指纹的因子才有资格争夺 1.0
            # 这自动包含了：
            #   - 本代新计算出来的因子 (在 register_fingerprints 中录入)
            #   - 上一代留存的精英 (在 update_and_prune 中保留)
            if expr_str in self.fingerprints_dict:
                all_to_cluster.append((expr_str, first_idx))
                # 记录该因子的纯武力值
                batch_power[expr_str] = self._get_power_score(current_results.get(expr_str, {}))
            else:
                # 凡是没指纹的（即：既没进榜，本代也没被变异出来的老因子）
                # 哪怕你是首发，也维持 scores_list[first_idx] = 0.1
                pass

                # 4. 簇内大乱斗：新老同台，强者胜出
        if all_to_cluster:
            try:
                # 提取参与竞争的所有因子指纹
                keys_to_calc = [x[0] for x in all_to_cluster]
                matrix = np.column_stack([self.fingerprints_dict[k] for k in keys_to_calc])

                # 聚类：把逻辑相似（>0.8）的划分为一簇
                labels = self._run_fast_clustering(matrix, self.threshold)
                key_to_label = dict(zip(keys_to_calc, labels))

                # 簇内排序：按武力值绝对高低排列
                # 如果武力值完全一样，我们可以保留一个稳定的次序（如 expr_str 字典序）
                sorted_candidates = sorted(
                    all_to_cluster,
                    key=lambda x: (batch_power.get(x[0], 0), x[0]),
                    reverse=True
                )

                cluster_occupied = set()
                for expr_str, first_idx in sorted_candidates:
                    label = key_to_label[expr_str]

                    # 只有每个逻辑领地的“最强者”能拿到 1.0 独立性
                    if label not in cluster_occupied:
                        scores_list[first_idx] = 1.0
                        cluster_occupied.add(label)
                    else:
                        # 你虽然是某个表达式的首发，但你这一代遇到了更强的同族竞争者
                        scores_list[first_idx] = self.penalty_factor

            except Exception as e:
                logger.error(f"聚类失败: {e}")
                # 容错处理：至少让首发代表们活下去
                for _, f_idx in all_to_cluster:
                    scores_list[f_idx] = 1.0

        return scores_list

    def _run_fast_clustering(self, matrix: np.ndarray, threshold: float) -> np.ndarray:
        """
        [核心计算逻辑] 基于 Spearman 相关性的快速聚类实现
        """
        # 1. 秩变换 (Spearman 相关性基础)
        matrix_rank = np.apply_along_axis(lambda x: x.argsort().argsort(), 0, matrix).astype(np.float32)

        # 2. 相关性矩阵
        corr_matrix = np.nan_to_num(np.corrcoef(matrix_rank, rowvar=False), nan=0.0)

        # 3. 距离矩阵: d = sqrt(2 * (1 - |rho|))
        dist_matrix = np.sqrt(np.clip(2 * (1 - np.abs(corr_matrix)), 0, None))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

        # 4. 快速聚类
        Z = fastcluster.linkage(squareform(dist_matrix), method='complete')

        # 5. 切割
        t_val = np.sqrt(2 * (1 - threshold))
        return fcluster(Z, t=t_val, criterion='distance')

    # --- 阶段 3: 内存与精英状态同步 ---

    def update_and_prune(self, halloffame: tools.HallOfFame):
        """代末清理，同步精英指纹并释放无效内存"""
        # hof_size 按照记忆应为 100
        before_count = len(self.fingerprints_dict)
        new_fingerprints = {}
        self.elite_keys = []
        self.elite_power_scores = {}

        for ind in halloffame:
            expr_str = getattr(ind, 'expr_str',None)
            if expr_str is None:
                 logger.error(f"无法从个体获取 expr_str 属性: {ind}")
                 continue

            if expr_str in self.fingerprints_dict:
                new_fingerprints[expr_str] = self.fingerprints_dict[expr_str]
                if expr_str not in self.elite_keys:
                    self.elite_keys.append(expr_str)
                    metrics = {name: val for name, val in zip(self.opt_names, ind.fitness.values)}
                    self.elite_power_scores[expr_str] = self._get_power_score(metrics)

        self.fingerprints_dict = new_fingerprints
        logger.info(f"🧹 独立性管理器剪枝: {before_count} -> {len(self.elite_keys)} (保留精英)")
