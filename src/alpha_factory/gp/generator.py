"""
GP 因子生成器主类

使用 DEAP 库进行遗传编程，自动化因子挖掘。

主要功能：
1. 基于遗传编程自动生成因子表达式
2. 批量计算和评估因子适应度
3. 支持进化过程的断点恢复
4. 自动缓存中间结果

典型用法：
    config = {
        "split_date": datetime(2021, 1, 1),
        "batch_size": 50,
        "mu": 100,
        "lambda": 100,
        "hof_size": 100
    }
    generator = GPDeapGenerator(config)
    pop, logbook, hof = generator.run(input_data, n_gen=10, n_pop=500)
"""

import operator
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np
import polars as pl
from deap import base, creator, gp, tools
from deap.gp import PrimitiveTree
from expr_codegen.tool import ExprTool
from loguru import logger
import more_itertools

from alpha_factory.data_provider import DataProvider

# 导入打过补丁的组件和基础工具
from alpha_factory.gp.base import (
    population_to_exprs,
    filter_exprs,
    print_population,
    strings_to_sympy,
)
from alpha_factory.gp.base import RET_TYPE, Expr
from alpha_factory.gp.dependence import DependenceManager
from alpha_factory.gp.ea import eaMuPlusLambda_NSGA2
from alpha_factory.data_provider.label import label_OO_for_IC, label_OO_for_tradable
from alpha_factory.patch.deap_patch import apply_deap_patches
from alpha_factory.patch.expr_codegen_patch import apply_expr_codegen_patches
from alpha_factory.polars.utils import CUSTOM_OPERATORS
from alpha_factory.config.base import settings
from alpha_factory.utils.schema import F

from typing import TypeVar
from polars import DataFrame as _pl_DataFrame
from polars import LazyFrame as _pl_LazyFrame


DataFrame = TypeVar("DataFrame", _pl_LazyFrame, _pl_DataFrame)

# 在脚本最上方或 __init__ 中调用一次即可
apply_expr_codegen_patches()
apply_deap_patches()


class GPDeapGenerator(object):
    """
    遗传编程因子生成器

    使用 DEAP 框架实现的自动化因子挖掘引擎。

    Attributes:
        config (Dict): 配置参数字典
        split_date (datetime): 训练/测试集分割日期
        batch_size (int): 批量计算大小
        save_dir (Path): 结果保存目录
        mu (int): 种群保留规模
        lambda_ (int): 每代生成后代规模
        hof_size (int): 名人堂大小
    """

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        """
        初始化 GP 因子生成器

        Args:
            config: 配置字典，支持的键：
                - split_date (datetime): 分割日期，训练集于验证集的分割日期，默认None
                - batch_size (int): 批处理大小，默认 50
                - mu (int): 进化算法的 mu 参数，默认 100
                - lambda (int): 进化算法的 lambda 参数，默认 100
                - hof_size (int): 名人堂大小，默认 100

        Raises:
            ValueError: 如果配置参数无效
        """
        # --- 1. 基础信息配置 ---
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

        # --- 2. 数据与日期配置 ---
        self.start_date = config.get("start_date", "20190101")
        self.end_date = config.get("end_date", "20241231")
        # 分割日期，训练集与验证集的分割日期，默认None
        self.split_date = config.get("split_date", None)
        # 多目标优化名称及权重
        # 这里的名称要和fitness_population_func 输出指标保持一致(要包含对应的列)
        # complexity，表示因子复杂度，可以不包含在fitness_population_func输出指标中
        # independence，表示因子独立性分数，可以不包含在fitness_population_func输出指标中
        # 多目标示例
        # self.opt_names = config.get("opt_names",("ic_mean_abs", "ic_ir_abs",'complexity','independence'))  #
        # self.opt_weights = config.get("opt_weights",(1.0, 1.0,-0.01,1.0))  # 多目标优化权重
        self.opt_names = config.get("opt_names", ("ic_mean_abs",))  #
        self.opt_weights = config.get("opt_weights", (1.0,))
        # 整体种群fitness函数,
        # 输入参数为:df,factors（所有的因子列名）,split_date(可以没有，训练集与验证集的分割日期),其它参数采用默认值
        # 输出数据格式为: pl.DataFrame，必须包含列factor,以及opt_names所包含的列
        self.fitness_population_func = config.get("fitness_population_func", None)

        self.pool_func = config.get("pool_func", None)  # 股票池函数
        # 标签计算函数，提供fitness_population_func计算所需的标签列，
        # 生成的标签列名必须和函数所需列名一致，一般为 F.LABEL_FOR_IC 和 F.LABEL_FOR_RET
        self.label_funcs = config.get(
            "label_funcs", [label_OO_for_IC, label_OO_for_tradable]
        )
        self.extra_terminal_func = config.get(
            "extra_terminal_func", []
        )  # 额外终端因子计算函数

        self.terminals = config.get("terminals", [])  # 终端因子列表
        self.random_window_func = config.get("random_window_func", None)  # 随机窗口函数

        # --- 4. 进化算法超参数 ---
        self.mu = config.get("mu", 400)  # 种群保留规模
        self.lambda_ = config.get("lambda", 400)  # 每代生成后代规模
        self.cxpb = config.get("cxpb", 0.3)  # 交叉概率
        self.mutpb = config.get("mutpb", 0.5)  # 变异概率
        self.hof_size = config.get("hof_size", 100)  # 名人堂大小
        self.batch_size = config.get("batch_size", 200)  # 批处理大小
        self.max_height = config.get("max_height", 4)  # 最大树高限制
        # 路径设置
        self._save_dir = None
        self.dep_manager = None  # 因子独立性管理器，稍后初始化
        self.cluster_threshold = config.get(
            "cluster_threshold", 0.7
        )  # 因子独立性聚类阈值
        self.penalty_factor = config.get("penalty_factor", -0.1)  # 因子独立性惩罚因子

        self.seed_file = config.get(
            "seed_file", "best_factors.csv"
        )  # 种子文件路径，用于断点恢复
        self.expression_formula = config.get(
            "expression_formula", "expression"
        )  # 预定义种子公式列表
        self.max_seed = config.get("max_seed", 0)  # 最大种子注入数量，默认0不注入

        # 缓存本轮实验的适应度结果，避免重复计算
        self.fitness_cache = {}

        logger.info(f"✓ GP 生成器初始化完成 | 批大小: {self.batch_size}")

    @property
    def save_dir(self):
        """获取结果保存目录"""
        if self._save_dir is None:
            self._save_dir = Path(settings.OUTPUT_DIR) / self.pool_func.__name__
            self._save_dir.mkdir(parents=True, exist_ok=True)
        return self._save_dir

    def seep_file_path(self):
        """获取种子文件路径"""
        return self.save_dir / self.seed_file

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        精简版算子集：专为 LightGBM/ElasticNet 特征工程设计
        所有算子均返回数值类型，移除了导致 SchemaError 的逻辑算子
        """
        # 直接使用 Expr 作为标识，默认即为浮点数序列
        pset = gp.PrimitiveSetTyped("MAIN", [], Expr)
        return pset

    def build_toolbox(self, input_data: pl.LazyFrame) -> base.Toolbox:
        """
        构建进化工具箱

        Args:
            input_data: 输入数据，用于适应度评估

        Returns:
            base.Toolbox: DEAP 工具箱实例
        """
        creator.create("FitnessMulti", base.Fitness, weights=self.opt_weights)
        creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # 树生成算法: 半数半萌法 (Half and Half)
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 遗传算子: 锦标赛选择、交叉、变异
        if len(self.opt_weights) == 1:
            toolbox.register(
                "select", tools.selTournament, tournsize=3
            )  # 单目标优化选择
        else:
            toolbox.register("select", tools.selNSGA2)  # 多目标优化选择

        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        # 限制树高，防止膨胀 (Bloat)
        toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.max_height
            ),
        )
        toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.max_height
            ),
        )

        # 核心：批量评估映射
        toolbox.register(
            "evaluate", lambda x: (np.nan, np.nan)
        )  # 实际评分在 map 中完成
        toolbox.register(
            "map",
            self.map_exprs,
            gen=count(),
            split_date=self.split_date,
            input_data=input_data,
        )

        logger.debug("✓ Toolbox 构建完成")
        return toolbox

    def _extra_toolbox_settings(self, toolbox):
        """额外的 Toolbox 设置（可选扩展）,会覆盖默认的设置"""
        pass

    def build_statistics(self) -> tools.Statistics:
        """
        定义进化过程中的统计监控指标

        Returns:
            tools.Statistics: DEAP 统计对象
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean, axis=0)
        stats.register("max", np.nanmax, axis=0)
        stats.register("min", np.nanmin, axis=0)
        stats.register("std", np.nanstd, axis=0)
        return stats

    def seed_load(self, pop, seed_exprs: List[str], max_seeds: int = 20) -> List:
        """
        将预定义的种子公式注入种群。去掉了聚类逻辑，直接进行强类型转换。

        Args:
            pop: 初始化的 DEAP 种群列表
            seed_exprs: 候选公式字符串列表
            max_seeds: 最大注入数量

        Returns:
            pop: 注入种子后的种群
        """
        if not seed_exprs:
            return pop

        logger.info(f"🌱 正在加载种子选手 (上限: {max_seeds})...")

        # 1. 限制种子数量，防止其完全占据初始种群
        actual_seeds_to_load = seed_exprs[:max_seeds]

        # 2. 利用 Sympy 进行预处理（主要为了格式规范化和简化）
        try:
            # strings_to_sympy 返回 [(name, expr_obj, complexity), ...]
            processed = strings_to_sympy(actual_seeds_to_load, globals().copy())
        except Exception as e:
            logger.error(f"❌ Sympy 预处理种子失败: {e}")
            return pop

        seeds_count = 0
        for i, (name, expr_obj, _) in enumerate(processed):
            if i >= len(pop):
                break

            try:
                # 3. 将 Sympy 表达式转回字符串并处理潜在的类型不匹配
                # 针对 PrimitiveSetTyped 的常见问题：将 "20.0" 替换回 "20"
                expr_str = str(expr_obj).replace(".0)", ")").replace(".0,", ",")

                # 4. 强类型转换并替换种群中的个体
                ind = creator.Individual.from_string(expr_str, self.pset)
                pop[i] = ind
                seeds_count += 1
                logger.debug(f"✅ 种子注入成功 [{seeds_count}]: {expr_str}")

            except Exception as e:
                # 如果转换失败（通常是算子名不匹配或参数类型不对），跳过该种子
                logger.warning(f"⚠️ 种子转换跳过: {expr_obj} | 错误: {e}")
                continue

        logger.success(f"✨ 种子注入流程结束，成功注入 {seeds_count} 个个体")
        return pop

    def run(
        self, n_gen: int = 10, n_pop: int = 1000
    ) -> Tuple[List, Any, tools.HallOfFame]:
        """
        启动进化流程

        Args:
            input_data: 输入数据，必须包含标签列
            n_gen: 进化代数，默认 10
            n_pop: 初始种群大小，默认 1000

        Returns:
            Tuple: (最终种群, 进化日志, 名人堂)

        Raises:
            ValueError: 如果输入数据无效
        """
        # 初始化管理器, 用于因子独立性评估
        self.dep_manager = DependenceManager(
            opt_names=self.opt_names,
            opt_weights=self.opt_weights,
            cluster_threshold=self.cluster_threshold,
            penalty_factor=self.penalty_factor,
        )

        # 2. 载入原始数据
        # 挖掘因子通常需要 OHLCV，计算 OO 收益率需要 OPEN
        input_data = DataProvider().load_data(
            start_date=self.start_date,
            end_date=self.end_date,
            funcs=[self.pool_func, *self.label_funcs, self.extra_terminal_func],
            select_cols=[F.POOL_MASK, F.LABEL_FOR_IC, F.LABEL_FOR_RET, *self.terminals],
            cache_path=self.save_dir / f"{self.pool_func.__name__}.parquet",
        )
        logger.info("💾 标签数据已就绪")

        logger.info(f"🚀 启动 GP 进化 | 代数: {n_gen} | 种群: {n_pop}")
        self.pset = self._build_pset()
        toolbox = self.build_toolbox(input_data)
        stats = self.build_statistics()
        if len(self.opt_weights) == 1:
            # 单目标 selTournament
            hof = tools.HallOfFame(
                self.hof_size, similar=lambda ind1, ind2: str(ind1) == str(ind2)
            )
        else:
            # 多目标 selNSGA2
            hof = tools.ParetoFront(similar=lambda ind1, ind2: str(ind1) == str(ind2))

        # 初始化种群
        pop = toolbox.population(n=n_pop)
        logger.info(f"✓ 初始种群已生成 | 大小: {len(pop)}")

        if self.seep_file_path().exists() and self.max_seed > 0:
            try:
                seed_df = pl.read_csv(self.seep_file_path())
                seed_exprs = seed_df[self.expression_formula].to_list()
                pop = self.seed_load(pop, seed_exprs, self.max_seed)
            except Exception as e:
                logger.error(f"❌ 种子文件加载失败: {e}")

        # 执行进化
        logger.info("▶️ 开始遗传编程进化...")
        pop, logbook = eaMuPlusLambda_NSGA2(
            pop,
            toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.cxpb,  # 交叉概率
            mutpb=self.mutpb,  # 变异概率
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True,
            generator=self,
        )

        logger.info(f"✨ GP 进化完成 | 最终种群: {len(pop)} | 名人堂: {len(hof)}")

        print("=" * 60)
        print(logbook)

        print("=" * 60)
        print_population(hof, globals().copy())
        self.export_hof_to_csv(hof, globals().copy())
        return pop, logbook, hof

    def map_exprs(
        self,
        evaluate_func: Any,
        individuals: List,
        gen,
        split_date: datetime,
        input_data: pl.DataFrame,
    ) -> List[Tuple]:
        """
        批量计算种群适应度的核心方法

        处理流程：
        1. 备份当前代的表达式
        2. 加载历史适应度缓存
        3. 提取并过滤表达式
        4. 批量计算新表达式的适应度
        5. 更新缓存并返回结果

        Args:
            evaluate_func: 评估函数（未使用，由 map 调用要求）
            individuals: 当前代的个体列表
            gen: 代数迭代器
            split_date: 训练/测试分割日期
            input_data: 输入数据

        Returns:
            List[Tuple]: 每个个体的适应度元组列表
        """
        g = next(gen)
        logger.info(f">>> 第 {g} 代 | 种群大小: {len(individuals)}")

        # 3. 表达式清洗与过滤
        logger.debug("🔄 转换 DEAP 树 -> Sympy 表达式...")
        exprs_list = population_to_exprs(individuals, globals().copy())
        exprs_to_calc = filter_exprs(
            exprs_list, self.pset, RET_TYPE, self.fitness_cache
        )

        logger.info(f"📊 需计算: {len(exprs_to_calc)} / {len(exprs_list)} 个表达式")

        # 4. 批量计算
        if len(exprs_to_calc) > 0:
            for batch_id, batch in enumerate(
                more_itertools.batched(exprs_to_calc, self.batch_size)
            ):
                logger.debug(f"  批次 {batch_id + 1} | 大小: {len(list(batch))}")
                new_scores = self.batched_exprs(
                    batch_id, list(batch), g, split_date, input_data
                )
                self.fitness_cache.update(new_scores)

        # 5. 回填适应度（可加入惩罚）
        fitness_values = self.fill_fitness(individuals, exprs_list, self.fitness_cache)
        logger.info(f"✓ 第 {g} 代评估完成")
        return fitness_values

    def _calc_exprs(self, exprs_list, df_input):
        lf = df_input.lazy() if isinstance(df_input, pl.DataFrame) else df_input

        tool = ExprTool()
        codes, G = tool.all(
            exprs_list,
            style="polars",
            template_file="template.py.j2",
            replace=False,
            regroup=True,
            format=True,
            date="DATE",
            asset="ASSET",
            over_null=None,
            skip_simplify=True,
        )

        globals_ = {**CUSTOM_OPERATORS}
        exec(codes, globals_)

        df_output = globals_["main"](lf, ge_date_idx=0).collect()

        return df_output

    def batched_exprs(self, batch_id, exprs_list, gen, split_date, df_input):
        """每代种群分批计算，包含详细性能日志及平均用时"""
        if len(exprs_list) == 0:
            return {}

        # --- 阶段 A: 因子值计算 ---
        cnt = len(exprs_list)
        logger.info("第{}代-第{}批：开始计算因子值 (共 {} 条)", gen, batch_id, cnt)
        tic_calc = time.perf_counter()

        df_output = self._calc_exprs(exprs_list, df_input)

        toc_calc = time.perf_counter()
        calc_duration = toc_calc - tic_calc

        # 日志输出：添加速度和平均耗时
        logger.info(
            "第{}代-第{}批：计算完成。总耗时: {:.3f}s | 速度: {:.2f} 条/s | 平均: {:.4f}s/条",
            gen,
            batch_id,
            calc_duration,
            cnt / calc_duration,
            calc_duration / cnt,
        )

        # --- 阶段 B: 适应度计算 ---
        logger.info("第{}代-第{}批：开始聚合计算 IC/RET 适应度指标", gen, batch_id)
        tic_fit = time.perf_counter()

        factor_columns = [k for k, v, c in exprs_list]
        import inspect

        if (
            split_date
            and "split_date"
            in inspect.signature(self.fitness_population_func).parameters
        ):
            fitness_df = self.fitness_population_func(
                df_output, factors=factor_columns, split_date=split_date
            )
        else:
            fitness_df = self.fitness_population_func(df_output, factors=factor_columns)

        toc_fit = time.perf_counter()
        fit_duration = toc_fit - tic_fit

        logger.info(
            "第{}代-第{}批：聚合完成。耗时: {:.3f}s | 平均: {:.4f}s/条",
            gen,
            batch_id,
            fit_duration,
            fit_duration / cnt,
        )

        # 3. 结果转换
        key_to_expr = {k: str(v) for k, v, c in exprs_list}
        new_results = {}
        for row in fitness_df.to_dicts():
            f_name = row.pop("factor")
            # 获取对应的表达式字符串作为 Key
            expr_str = key_to_expr[f_name]
            new_results[expr_str] = row

        if "independence" in self.opt_names:
            # 这里的 exprs_list 包含了 (因子名, 表达式对象, _)
            self.dep_manager.register_fingerprints(df_output, exprs_list)

        # 4. 汇总
        total_dur = calc_duration + fit_duration
        logger.info(
            "第{}代-第{}批：流程结束。总计: {:.3f}s | 总平均: {:.4f}s/条 (算值:{:.1%}, 指标:{:.1%})",
            gen,
            batch_id,
            total_dur,
            total_dur / cnt,
            calc_duration / total_dur,
            fit_duration / total_dur,
        )

        return new_results

    def fill_fitness(self, individuals, exprs_old, fitness_results):
        """
        重构版：完全适配按位置评分的独立性接口
        """
        if len(individuals) != len(exprs_old):
            raise ValueError(
                f"数据对齐失败: individuals({len(individuals)}) != exprs_old({len(exprs_old)})"
            )

        # 1. 预计算惩罚向量
        penalty_values = tuple(0.0 if w > 0 else 999.0 for w in self.opt_weights)

        # 2. 挂载 expr_str 并收集全量表达式
        all_expr_strs = []
        for ind, (_, v, _) in zip(individuals, exprs_old):
            search_key = str(v)
            ind.expr_str = search_key  # 方便后续 update_and_prune 识别
            all_expr_strs.append(search_key)

        # 3. 【核心修改】获取按位置对应的独立性分数列表
        # 现在的 indep_scores_list 是一个 List[float]，长度与 individuals 一致
        indep_scores_list = self.dep_manager.calculate_contextual_independence(
            all_expr_strs, fitness_results
        )

        fit_tuples_list = []

        # 4. 遍历填充：利用 zip(individuals, all_expr_strs, indep_scores_list) 实现物理对齐
        for ind, search_key, current_indep_score in zip(
            individuals, all_expr_strs, indep_scores_list
        ):
            score_dict = fitness_results.get(search_key)

            # 情况 A: 匹配失败或触发惩罚
            if score_dict is None or self.is_penalty(score_dict):
                fit_tuples_list.append(penalty_values)
                ind.stats = None
                continue

            # 情况 B: 正常评估
            try:
                current_fit = []
                for i, name in enumerate(self.opt_names):
                    # 逻辑分支 1: 复杂度 (静态)
                    if name == "complexity":
                        val = float(len(ind))

                    # 逻辑分支 2: 独立性 (使用 DM 实时计算的、基于位置的分数)
                    elif name == "independence":
                        # 【关键点】这里不再查表，直接用 current_indep_score
                        val = float(current_indep_score)

                    # 逻辑分支 3: 其他绩效指标
                    else:
                        raw_val = score_dict.get(name)
                        if raw_val is None or not np.isfinite(raw_val):
                            val = penalty_values[i]
                        else:
                            val = float(raw_val)

                    current_fit.append(val)

                fit_tuple = tuple(current_fit)
                fit_tuples_list.append(fit_tuple)

                # 更新 stats，确保 stats 里的独立性也是“这个个体”专属的实时分数
                final_stats = score_dict.copy()
                if "independence" in self.opt_names:
                    final_stats["independence"] = current_indep_score
                ind.stats = final_stats

            except Exception as e:
                logger.error(f"处理适应度异常: {e} | {search_key}")
                fit_tuples_list.append(penalty_values)
                ind.stats = None

        return fit_tuples_list

    def is_penalty(self, score_dict):
        """判断某个计算结果是否为惩罚值"""
        if "ic_ir" in score_dict:
            val = score_dict["ic_ir"]
            if np.isnan(val) or val < 0.0001:
                return True
        return False

    def export_hof_to_csv(self, hof, globals_, filename="gp_best_factors.csv"):
        """
        将名人堂内容导出到 CSV

        Args:
            hof: 名人堂对象
            globals_: 全局命名空间 globals()
            filename: 输出文件名
        """
        import pandas as pd

        # exprs_list 得到的是 (简化名 k, 表达式文本 v, 复杂度 c)
        exprs_list = population_to_exprs(hof, globals_)
        data = []
        for (k, v, c), ind in zip(exprs_list, hof):
            kvs = {
                "factor_name": k,  # 因子简化名
                "expression": v,  # 简化后的表达式文本 (v)
                "complexity": c,  # 复杂度 (c)
                "raw_tree": str(ind),  # 原始 DEAP 树结构
            }
            # 提取名人堂个体的适应度值并存储到字典中
            for name, value in zip(self.opt_names, ind.fitness.values):
                kvs[name] = value
            data.append(kvs)

        # 2. 转换为 DataFrame 并保存
        df = pd.DataFrame(data)
        output_path = self.save_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 名人堂因子已导出至 CSV: {output_path}")
        return df
