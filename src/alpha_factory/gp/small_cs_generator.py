import random
from typing import Dict, Any

from deap import gp

from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.evaluation.batch.full_metrics import batch_full_metrics
from alpha_factory.gp.base import Expr, dummy
from alpha_factory.gp.extra_terminal import add_extra_terminals
from alpha_factory.gp.generator import GPDeapGenerator
from alpha_factory.gp.label import label_OO_for_IC, label_OO_for_tradable


def _random_window_int():
    # 偏向短周期：5, 10, 21, 42, 63
    return random.choice([5, 10, 15, 20, 30, 40, 60, 80])


class SmallCSGenerator(GPDeapGenerator):
    """
    针对小微盘进行横截面因子挖掘GP因子生成器
    """

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        super().__init__(config)
        self.top_n = config.get("top_n", 1000)
        self.max_height = config.get("max_height", 4)  # 最大树高限制
        self.cxpb = config.get("cxpb", 0.5)  # 交叉概率
        self.mutpb = config.get("mutpb", 0.3)  # 变异概率

        self.pool_func = config.get("pool_func", main_small_pool)  # 小微盘股票池函数
        self.label_funcs = config.get(
            "label_funcs", [label_OO_for_IC, label_OO_for_tradable]
        )
        self.random_window_func = config.get(
            "random_window_func", _random_window_int
        )  # 随机窗口函数
        self.extra_terminal_func = config.get(
            "extra_terminal_func", add_extra_terminals
        )  # 额外终端因子计算函数
        self.terminals = config.get(
            "terminals",
            [
                "OPEN",
                "HIGH",
                "LOW",
                "CLOSE",
                "TURNOVER_RATE",
                "VWAP",
                "RET",
                "VWAP_RET",
            ],
        )

        self.fitness_population_func = config.get(
            "fitness_population_func", batch_full_metrics
        )
        self.opt_names = config.get("opt_names", ("ann_ret",))  # 多目标优化因子名称
        self.opt_weights = config.get("opt_weights", (1.0,))  # 多目标优化权重

        self.hof_size = config.get("hof_size", 100)  # 名人堂大小
        self.cluster_threshold = config.get(
            "cluster_threshold", 0.7
        )  # 因子独立性聚类阈值
        self.penalty_factor = config.get("penalty_factor", -1.0)  # 因子独立性惩罚因子

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        精简版算子集：专为 LightGBM/ElasticNet 特征工程设计
        所有算子均返回数值类型，移除了导致 SchemaError 的逻辑算子
        """
        # 直接使用 Expr 作为标识，默认即为浮点数序列
        pset = gp.PrimitiveSetTyped("MAIN", [], Expr)

        # 1. 终端 (Terminals)
        # AMOUNT有极强的市值效应，不适合截面因子挖掘
        for factor in self.terminals:
            pset.addTerminal(1, Expr, name=factor)

        # 2. 窗口参数 (Constants)
        pset.addEphemeralConstant("rand_int", _random_window_int, int)

        # 3. 基础算术算子 (线性模型 ElasticNet 无法自学除法和乘法交互)
        for name in ["add", "sub", "mul", "div"]:
            pset.addPrimitive(dummy, [Expr, Expr], Expr, name=f"oo_{name}")

        # 4. 时序统计算子 (LightGBM 的盲区：无法跨行感知历史)
        # ts_mean:最基础的趋势中枢（均线）。它提供了价格的“锚点”。
        # ts_std_dev:波动率是金融市场中信息含量最高的特征之一。树模型对原始收益率敏感，但无法直接推算当前是否处于“高波动环境”。
        # ts_max / ts_min:极值点往往对应支撑阻力位，帮助模型捕捉反转信号。
        # ts_delta:衡量当前价格与过去价格的差异，捕捉动量效应。
        # ts_returns:直接提供收益率信息，帮助模型理解价格变动。
        # ts_rank:提供当前价格在过去一段时间内的相对位置，辅助模型识别超买超卖状态。核心特征。它将价格转化成 0-1 的比例
        # ts_skewness:衡量收益率分布的偏态，帮助模型理解极端事件的风险。
        # 可选：ts_sum, ts_kurtosis,ts_delay,ts_arg_max,ts_arg_min
        ts_ops = [
            "ts_mean",
            "ts_std_dev",
            "ts_max",
            "ts_min",
            "ts_delta",
            "ts_returns",
            "ts_skewness",
            "ts_decay_linear",
            "ts_BIAS",
        ]
        for op in ts_ops:
            pset.addPrimitive(dummy, [Expr, int], Expr, name=op)
        # 时序相关性与协方差 (增强模型对多因子联动的感知),极其强大。
        # 例如成交量和价格的相关性（价升量增 vs 价升量减）。这是 ML 模型绝对无法通过原始特征自己算出来的交互信息。
        pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name="ts_corr")
        # 量化两个序列的同步运动程度，是 ts_corr 的未归一化版本，能保留数值规模信息。
        # pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_covariance')

        # 5. 截面归一化算子 (机器学习模型的关键：提供样本间的相对坐标)
        # 所有的界面算子都需要针对股票池POOL_MASK进行计算
        # 精细化建议：'cs_demean_mask','cs_qcut_mask'
        cs_ops = ["cs_rank_mask", "cs_mad_zscore_mask"]
        for op in cs_ops:
            pset.addPrimitive(dummy, [Expr], Expr, name=op)

        # 6. 数值变换 (改善线性模型的特征分布)
        # 可选:sigmoid / tanh (软截断)
        # for op in ['abs_', 'log','sqrt']:
        #     pset.addPrimitive(dummy, [Expr], Expr, name=op)

        return pset
