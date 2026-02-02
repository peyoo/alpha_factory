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
        "label_y": "RETURN_OO_1",
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
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from deap import base, creator, gp, tools
from deap.gp import PrimitiveTree
from loguru import logger
import more_itertools
import polars.selectors as cs

from alpha.data_provider import DataProvider
# 导入打过补丁的组件和基础工具
from alpha.gp.deap_patch import eaMuPlusLambda  # 核心进化算法
from alpha.gp.base import population_to_exprs, filter_exprs, print_population
# from alpha.gp.cs.helper import batched_exprs, fill_fitness
from alpha.gp.base import RET_TYPE, Expr
from alpha.utils.config import settings

from typing import TypeVar
from polars import DataFrame as _pl_DataFrame
from polars import LazyFrame as _pl_LazyFrame

from alpha.utils.schema import F

DataFrame = TypeVar("DataFrame", _pl_LazyFrame, _pl_DataFrame)



class GPDeapGenerator(object):
    """
    遗传编程因子生成器

    使用 DEAP 框架实现的自动化因子挖掘引擎。

    Attributes:
        config (Dict): 配置参数字典
        label_y (str): 目标标签列名
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
                - label_y (str): 目标列名，默认 "RETURN_OO_1"
                - split_date (datetime): 分割日期，默认 2021-01-01
                - batch_size (int): 批处理大小，默认 50
                - mu (int): 进化算法的 mu 参数，默认 100
                - lambda (int): 进化算法的 lambda 参数，默认 100
                - hof_size (int): 名人堂大小，默认 100

        Raises:
            ValueError: 如果配置参数无效
        """
        """
                初始化配置，将所有 config 取值集中在此
                """
        # --- 1. 基础信息配置 ---
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

        # --- 2. 数据与日期配置 ---
        self.start_date = config.get("start_date", "20190101")
        self.end_date = config.get("end_date", "20241231")
        self.split_date = config.get("split_date", datetime(2022, 1, 1))
        self.opt_names = config.get("opt_names",("ic_mean", "ic_ir", "complexity"))  # 多目标优化名称及权重
        self.opt_weights = config.get("opt_weights",(1.0, 1.0, -0.01))  # 多目标优化名称及权重
        # 整体种群fitness函数,输入参数为:df,factors,split_date
        # 其它参数采用默认名
        self.fitness_population_func = config.get("fitness_population_func", None)

        self.pool_func = config.get("pool_func", None)  # 股票池函数
        self.label_func = config.get("label_func", None)  # 标签计算函数
        self.random_window_func = config.get("random_window_func", None)  # 随机窗口函数
        self.extra_terminal_func = config.get("extra_terminal_func", [])  # 额外终端因子计算函数

        self.terminals = config.get('terminals', [])  # 终端因子列表


        # --- 3. 标签计算配置 ---
        self.label_window = config.get("label_window", 1) # 计算标签的未来窗口大小
        self.label_y = config.get("label_y", f"LABEL_OO_{self.label_window}")  # 目标标签列名,当前仅支持 OPEN-OPEN 收益率

        # --- 4. 进化算法超参数 ---
        self.mu = config.get("mu", 300) # 种群保留规模
        self.lambda_ = config.get("lambda", 400)  # 每代生成后代规模
        self.cxpb = config.get("cxpb", 0.6)  # 交叉概率
        self.mutpb = config.get("mutpb", 0.2)  # 变异概率
        self.hof_size = config.get("hof_size", 1000) # 名人堂大小
        self.batch_size = config.get("batch_size", 200) # 批处理大小
        self.max_height = config.get("max_height", 6) # 最大树高限制


        # 路径设置
        self.save_dir = Path(settings.GP_DEAP_DIR)/ self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ GP 生成器初始化完成 | 标签: {self.label_y} | 批大小: {self.batch_size}")


    def _prepare_labeled_data(self) -> DataFrame:
        """
        从 DataProvider 获取数据并计算挖掘标签
        计算逻辑：未来 N 日的 Open-to-Open 收益率
        """

        cache_file = self.save_dir / f"labeled_{self.label_y}.parquet"
        logger.info(f"📡 正在计算标签 '{self.label_y}'...")

        data_provider = DataProvider()

        # 2. 载入原始数据
        # 挖掘因子通常需要 OHLCV，计算 OO 收益率需要 OPEN
        lf = data_provider.load_data(
            start_date= self.start_date,
            end_date=self.end_date,
            funcs=[self.pool_func,self.label_func,self.extra_terminal_func],
            select_cols=[F.POOL_MASK,self.label_y,*self.terminals],
            cache_path= cache_file
        )
        # 排序
        lf = lf.sort(['ASSET', 'DATE']).with_columns([
            pl.col("ASSET").set_sorted(),
            # 强制将所有数值列转为 Float64，避免 GP 运行时 SchemaError
            cs.numeric().cast(pl.Float64)
        ])
        logger.info("💾 标签数据已就绪")

        return lf


    def run_workflow(self, n_gen: int = 10) -> Tuple[List, Any, Any]:
        """
        全流程一键启动：数据准备 -> 进化挖掘

        Args:
            n_gen: 进化代数，默认 10
        Returns:
            Tuple[List, logbook, HallOfFame]: (最终种群, 进化日志, 名人堂)
        """
        logger.info("=" * 60)
        logger.info("GP 因子挖掘全流程启动")
        logger.info("=" * 60)

        # 步骤 1: 数据准备
        logger.info(">>> 步骤 1/2: 数据准备")
        input_df = self._prepare_labeled_data()

        # 步骤 2: 启动进化
        logger.info(">>> 步骤 2/2: 启动遗传编程进化")
        result = self.run(input_df, n_gen=n_gen)

        logger.info("=" * 60)
        logger.info("✅ 全流程执行完成")
        logger.info("=" * 60)

        return result

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        精简版算子集：专为 LightGBM/ElasticNet 特征工程设计
        所有算子均返回数值类型，移除了导致 SchemaError 的逻辑算子
        """
        # 直接使用 Expr 作为标识，默认即为浮点数序列
        pset = gp.PrimitiveSetTyped("MAIN", [], Expr)
        return pset

    def build_toolbox(self, input_data: pl.DataFrame) -> base.Toolbox:
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
        toolbox.register("select", tools.selTournament, tournsize=3)
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
            label=self.label_y,
            split_date=self.split_date,
            input_data=input_data
        )

        logger.debug("✓ Toolbox 构建完成")
        return toolbox

    def map_exprs(
        self,
        evaluate_func: Any,
        individuals: List,
        gen,
        label: str,
        split_date: datetime,
        input_data: pl.DataFrame
    ) -> List[Tuple[float, float]]:
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
            label: 标签列名
            split_date: 训练/测试分割日期
            input_data: 输入数据

        Returns:
            List[Tuple[float, float]]: 每个个体的适应度元组列表
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
                new_scores = self.batched_exprs(batch_id, list(batch), g, label, split_date, input_data)
                fitness_results.update(new_scores)

            # 更新全局缓存
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(fitness_results, f)
                logger.debug(f"✓ 缓存已更新 | 总结果数: {len(fitness_results)}")
            except Exception as e:
                logger.warning(f"⚠️ 缓存保存失败: {e}")

        # 5. 回填适应度
        fitness_values = self.fill_fitness(exprs_list, fitness_results)
        logger.info(f"✓ 第 {g} 代评估完成")
        return fitness_values

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
        input_data: pl.DataFrame,
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
        # 验证输入数据
        if self.label_y not in input_data.collect_schema().names():
            raise ValueError(f"输入数据缺少标签列: {self.label_y}")

        logger.info(f"🚀 启动 GP 进化 | 代数: {n_gen} | 种群: {n_pop}")
        self.pset = self._build_pset()
        toolbox = self.build_toolbox(input_data)
        stats = self.build_statistics()
        hof = tools.HallOfFame(self.hof_size)

        # 初始化种群
        pop = toolbox.population(n=n_pop)
        logger.info(f"✓ 初始种群已生成 | 大小: {len(pop)}")

        # 执行进化
        logger.info("▶️ 开始遗传编程进化...")
        pop, logbook = eaMuPlusLambda(
            pop, toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb= self.cxpb,  # 交叉概率
            mutpb= self.mutpb,  # 变异概率
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True
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

    def export_hof_to_csv(self, hof, globals_, filename="best_factors.csv"):
        """
        将名人堂内容导出到 CSV

        Args:
            hof: 名人堂对象
            globals_: 全局命名空间 globals()
            filename: 输出文件名
        """
        import pandas as pd

        # 1. 模仿你的逻辑解析表达式
        # exprs_list 得到的是 (简化名 k, 表达式文本 v, 复杂度 c)
        exprs_list = population_to_exprs(hof, globals_)

        data = []
        for (k, v, c), ind in zip(exprs_list, hof):
            # 提取适应度（处理多目标情况）
            fitness_values = ind.fitness.values
            train_ic = fitness_values[0] if len(fitness_values) > 0 else None
            test_ic = fitness_values[1] if len(fitness_values) > 1 else None

            data.append({
                "factor_name": k,  # 因子简化名
                "fitness_train": train_ic,  # 训练集适应度/IC
                "fitness_test": test_ic,  # 测试集适应度/IC
                "expression": v,  # 简化后的表达式文本 (v)
                "complexity": c,  # 复杂度 (c)
                "raw_tree": str(ind)  # 原始 DEAP 树结构
            })

        # 2. 转换为 DataFrame 并保存
        df = pd.DataFrame(data)
        output_path = self.save_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info(f"✅ 名人堂因子已导出至 CSV: {output_path}")
        return df

    def fitness_individual(self,a: str, b: str) -> pl.Expr:
        """个体fitness函数"""
        return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)

    def batched_exprs(self,batch_id, exprs_list, gen, label, split_date, df_input):
        return {}

    def fill_fitness(self,exprs_old, fitness_results):
        return []
