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
import pickle
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

from alpha.data_provider import DataProvider
# 导入打过补丁的组件和基础工具
from alpha.gp.base import population_to_exprs, filter_exprs, print_population
from alpha.gp.base import RET_TYPE, Expr
from alpha.gp.dependence import DependenceManager
from alpha.gp.ea import eaMuPlusLambda_NSGA2
from alpha.gp.label import label_OO_for_IC, label_OO_for_tradable
from alpha.patch.expr_codegen_patch import apply_expr_codegen_patches
from alpha.polars.utils import CUSTOM_OPERATORS
from alpha.utils.config import settings

from typing import TypeVar
from polars import DataFrame as _pl_DataFrame
from polars import LazyFrame as _pl_LazyFrame

from alpha.utils.schema import F

DataFrame = TypeVar("DataFrame", _pl_LazyFrame, _pl_DataFrame)


# 在脚本最上方或 __init__ 中调用一次即可
apply_expr_codegen_patches()

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
        # 这里的名称和权重要和fitness_population_func 输出指标保持一致
        # complexity，表示因子复杂度，可以不包含在fitness_population_func输出指标中
        self.opt_names = config.get("opt_names",("ic_mean_abs", "ic_ir_abs",'complexity'))  #
        self.opt_weights = config.get("opt_weights",(1.0, 1.0,-0.01))  # 多目标优化权重
        # 整体种群fitness函数,
        # 输入参数为:df,factors（所有的因子列名）,split_date(可以没有，训练集与验证集的分割日期),其它参数采用默认值
        # 输出数据格式为: pl.DataFrame，必须包含列factor,以及opt_names所包含的列
        self.fitness_population_func = config.get("fitness_population_func", None)

        self.pool_func = config.get("pool_func", None)  # 股票池函数
        # 标签计算函数，提供fitness_population_func计算所需的标签列，
        # 生成的标签列名必须和函数所需列名一致，一般为 F.LABEL_FOR_IC 和 F.LABEL_FOR_RET
        self.label_funcs = config.get("label_funcs", [label_OO_for_IC,label_OO_for_tradable])
        self.extra_terminal_func = config.get("extra_terminal_func", [])  # 额外终端因子计算函数

        self.terminals = config.get('terminals', [])  # 终端因子列表
        self.random_window_func = config.get("random_window_func", None)  # 随机窗口函数

        # --- 4. 进化算法超参数 ---
        self.mu = config.get("mu", 400) # 种群保留规模
        self.lambda_ = config.get("lambda", 400)  # 每代生成后代规模
        self.cxpb = config.get("cxpb", 0.4)  # 交叉概率
        self.mutpb = config.get("mutpb", 0.3)  # 变异概率
        self.hof_size = config.get("hof_size", 1000) # 名人堂大小
        self.batch_size = config.get("batch_size", 200) # 批处理大小
        self.max_height = config.get("max_height", 3) # 最大树高限制
        # 路径设置
        self.save_dir = Path(settings.GP_DEAP_DIR)/ self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dep_manager = None  # 因子独立性管理器，稍后初始化

        logger.info(f"✓ GP 生成器初始化完成 | 批大小: {self.batch_size}")


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
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 遗传算子: 锦标赛选择、交叉、变异
        # toolbox.register("select", tools.selTournament, tournsize=3) # 单目标优化选择
        toolbox.register("select", tools.selNSGA2)  # 多目标优化选择

        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        # 限制树高，防止膨胀 (Bloat)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_height))

        # 核心：批量评估映射
        toolbox.register("evaluate", lambda x: (np.nan, np.nan))  # 实际评分在 map 中完成
        toolbox.register(
            "map",
            self.map_exprs,
            gen=count(),
            split_date=self.split_date,
            input_data=input_data
        )

        logger.debug("✓ Toolbox 构建完成")
        return toolbox


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

    def run(
        self,
        n_gen: int = 10,
        n_pop: int = 1000
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
            cluster_threshold=0.7
        )

        # 2. 载入原始数据
        # 挖掘因子通常需要 OHLCV，计算 OO 收益率需要 OPEN
        input_data = DataProvider().load_data(
            start_date=self.start_date,
            end_date=self.end_date,
            funcs=[self.pool_func, *self.label_funcs, self.extra_terminal_func],
            select_cols=[F.POOL_MASK, F.LABEL_FOR_IC,F.LABEL_FOR_RET, *self.terminals],
            cache_path=self.save_dir / f"{self.pool_func.__name__}.parquet"
        )
        logger.info("💾 标签数据已就绪")

        logger.info(f"🚀 启动 GP 进化 | 代数: {n_gen} | 种群: {n_pop}")
        self.pset = self._build_pset()
        toolbox = self.build_toolbox(input_data)
        stats = self.build_statistics()
        hof = tools.HallOfFame(self.hof_size)
        # 多目标 selNSGA2
        # hof = tools.ParetoFront(similar=operator.eq)

        # 初始化种群
        pop = toolbox.population(n=n_pop)
        logger.info(f"✓ 初始种群已生成 | 大小: {len(pop)}")

        # 执行进化
        logger.info("▶️ 开始遗传编程进化...")
        pop, logbook = eaMuPlusLambda_NSGA2(
            pop, toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb= self.cxpb,  # 交叉概率
            mutpb= self.mutpb,  # 变异概率
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True,
            generator = self
        )

        # 保存名人堂
        hof_path = self.save_dir / 'best_hof.pkl'
        try:
            with open(hof_path, 'wb') as f:
                pickle.dump(hof, f)
            logger.info(f"💾 名人堂已保存至: {hof_path}")
        except Exception as e:
            logger.error(f"❌ 名人堂保存失败: {e}")

        logger.info(f"✨ GP 进化完成 | 最终种群: {len(pop)} | 名人堂: {len(hof)}")

        print('=' * 60)
        print(logbook)

        print('=' * 60)
        print_population(hof, globals().copy())
        self.export_hof_to_csv(hof, globals().copy())
        return pop, logbook, hof


    def map_exprs(
        self,
        evaluate_func: Any,
        individuals: List,
        gen,
        split_date: datetime,
        input_data: pl.DataFrame
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

        # 2. 缓存管理
        cache_path = self.save_dir / 'fitness_cache.pkl'
        fitness_results: Dict = {} # 表达式字符串 -> 适应度元组
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    fitness_results = pickle.load(f)
                logger.debug(f"✓ 加载历史缓存 | 已有结果: {len(fitness_results)}")
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")

        # 3. 表达式清洗与过滤
        logger.debug("🔄 转换 DEAP 树 -> Sympy 表达式...")
        exprs_list = population_to_exprs(individuals, globals().copy())
        exprs_to_calc = filter_exprs(exprs_list, self.pset, RET_TYPE, fitness_results)

        logger.info(f"📊 需计算: {len(exprs_to_calc)} / {len(exprs_list)} 个表达式")

        # 4. 批量计算
        if len(exprs_to_calc) > 0:
            for batch_id, batch in enumerate(more_itertools.batched(exprs_to_calc, self.batch_size)):
                logger.debug(f"  批次 {batch_id + 1} | 大小: {len(list(batch))}")
                new_scores = self.batched_exprs(batch_id, list(batch), g, split_date, input_data)
                fitness_results.update(new_scores)

            # 更新全局缓存
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(fitness_results, f)
                logger.debug(f"✓ 缓存已更新 | 总结果数: {len(fitness_results)}")
            except Exception as e:
                logger.warning(f"⚠️ 缓存保存失败: {e}")

        # 5. 回填适应度（可加入惩罚）
        fitness_values = self.fill_fitness(individuals,exprs_list, fitness_results)
        logger.info(f"✓ 第 {g} 代评估完成")
        return fitness_values

    def batched_exprs(self, batch_id, exprs_list, gen, split_date, df_input):
        """每代种群分批计算，包含详细性能日志及平均用时"""
        if len(exprs_list) == 0:
            return {}

        lf = df_input.lazy() if isinstance(df_input, pl.DataFrame) else df_input

        tool = ExprTool()
        codes, G = tool.all(exprs_list, style='polars', template_file='template.py.j2',
                            replace=False, regroup=True, format=True,
                            date='DATE', asset='ASSET', over_null=None,
                            skip_simplify=True)

        cnt = len(exprs_list)
        globals_ = {**CUSTOM_OPERATORS}
        exec(codes, globals_)

        # --- 阶段 A: 因子值计算 ---
        logger.info("第{}代-第{}批：开始计算因子值 (共 {} 条)", gen, batch_id, cnt)
        tic_calc = time.perf_counter()

        df_output = globals_['main'](lf, ge_date_idx=0).collect()

        toc_calc = time.perf_counter()
        calc_duration = toc_calc - tic_calc

        # 日志输出：添加速度和平均耗时
        logger.info(
            "第{}代-第{}批：计算完成。总耗时: {:.3f}s | 速度: {:.2f} 条/s | 平均: {:.4f}s/条",
            gen, batch_id, calc_duration, cnt / calc_duration, calc_duration / cnt
        )



        # --- 阶段 B: 适应度计算 ---
        logger.info("第{}代-第{}批：开始聚合计算 IC/RET 适应度指标", gen, batch_id)
        tic_fit = time.perf_counter()

        factor_columns = [k for k, v, c in exprs_list]
        import inspect
        if split_date and 'split_date' in inspect.signature(self.fitness_population_func).parameters:
            fitness_df = self.fitness_population_func(df_output, factors=factor_columns, split_date=split_date)
        else:
            fitness_df = self.fitness_population_func(df_output, factors=factor_columns)

        toc_fit = time.perf_counter()
        fit_duration = toc_fit - tic_fit

        logger.info(
            "第{}代-第{}批：聚合完成。耗时: {:.3f}s | 平均: {:.4f}s/条",
            gen, batch_id, fit_duration, fit_duration / cnt
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
            # 这里的 exprs_list 包含了 (因子名, 表达式对象, 复杂度)
            indep_map = self.dep_manager.evaluate_independence(df_output, exprs_list,new_results)
            for expr_str, indep_score in indep_map.items():
                if expr_str in new_results:
                    new_results[expr_str]['independence'] = indep_score
        # 4. 汇总
        total_dur = calc_duration + fit_duration
        logger.info(
            "第{}代-第{}批：流程结束。总计: {:.3f}s | 总平均: {:.4f}s/条 (算值:{:.1%}, 指标:{:.1%})",
            gen, batch_id, total_dur, total_dur / cnt, calc_duration / total_dur, fit_duration / total_dur
        )

        return new_results

    def fill_fitness(self, individuals, exprs_old, fitness_results):
        """
        根据惯例处理并返回 Fitness 元组列表。
        同时原地更新个体的 stats 属性。

        Args:
            individuals: DEAP 个体列表 [ind1, ind2, ...]
            exprs_old: 辅助信息 [(k, v, c), ...]，其中 v 是表达式对象或字符串
            fitness_results: 计算结果字典 {str(v): {metrics_dict}}

        Returns:
            List[Tuple]: 对应每个个体的适应度元组列表，例如 [(ic, ir, comp), ...]
        """
        # 0. 长度安全性检查
        if len(individuals) != len(exprs_old):
            raise ValueError(f"数据对齐失败: individuals({len(individuals)}) != exprs_old({len(exprs_old)})")

        # 1. 预计算惩罚向量 (根据 opt_weights 符号确定惩罚方向)
        # 若权重为正(求最大)，惩罚值为 0.0；若权重为负(求最小)，惩罚值为 999.0
        penalty_values = tuple(0.0 if w > 0 else 999.0 for w in self.opt_weights)

        fit_tuples_list = []

        # 2. 遍历个体与对应的表达式描述
        # v 表示因子表达式字符串
        for ind, (_, v, _) in zip(individuals, exprs_old):
            # 统一使用字符串键匹配结果字典
            search_key = str(v)
            ind.expr_str = search_key  # 原地挂载表达式字符串，便于后续查询
            score_dict = fitness_results.get(search_key)

            # 情况 A: 匹配失败 (该因子因非法、重复被过滤，或计算模块报错)
            if score_dict is None:
                fit_tuples_list.append(penalty_values)
                ind.stats = None  # 清空或初始化 stats
                continue

            if self.is_penalty(score_dict):
                fit_tuples_list.append(penalty_values)
                ind.stats = None  # 清空或初始化 stats
                continue

            # 情况 B: 匹配成功，根据 self.opt_names 提取指标
            try:
                current_fit = []
                for i, name in enumerate(self.opt_names):
                    if name == "complexity":
                        # 直接获取 DEAP 树的节点数作为复杂度
                        val = float(len(ind))
                    else:
                        # 从结果字典提取指标，若 Key 不存在则直接触发 KeyError (配置错误)
                        try:
                            raw_val = score_dict[name]

                            # 核心防御：处理计算结果中的 NaN 或 Inf，防止 Logbook 崩溃
                            if raw_val is None or not np.isfinite(raw_val):
                                val = penalty_values[i]
                            else:
                                val = float(raw_val)
                        except KeyError:
                            logger.error(f"❌ 指标配置错误: '{name}' 不在计算结果中!")
                            logger.error(f"当前可用指标: {list(score_dict.keys())}")
                            raise KeyError(f"Metric '{name}' missing in fitness_results.")

                    current_fit.append(val)

                # 转换为元组并存入结果列表
                fit_tuple = tuple(current_fit)
                fit_tuples_list.append(fit_tuple)

                # 原地挂载全量指标字典，方便后验分析
                ind.stats = score_dict

            except Exception as e:
                logger.error(f"处理个体适应度异常: {e} | 表达式: {search_key}")
                fit_tuples_list.append(penalty_values)
                ind.stats = None

        return fit_tuples_list

    def is_penalty(self,score_dict):
        """判断某个计算结果是否为惩罚值"""
        if 'ic_ir' in score_dict:
            val = score_dict['ic_ir']
            if np.isnan(val) or val < 0.0001:
                return True
        return False

    def export_hof_to_csv(self, hof, globals_, filename="best_factors.csv"):
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
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 名人堂因子已导出至 CSV: {output_path}")
        return df
