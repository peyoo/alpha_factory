import random

from deap import gp

from alpha.gp.base import Expr, dummy


from alpha.gp.generator import GPDeapGenerator


def _random_window_int_timeseries() -> int:
    """
    针对时序因子挖掘优化的窗口采样逻辑：
    - 超短期 (2-5d): 20% 概率 -> 捕捉日间反转、高频噪音过滤
    - 核心趋势区 (10-60d): 60% 概率 -> 时序因子的均值回归、动量主要发生在该区间
    - 长期宏观 (120-500d): 20% 概率 -> 捕捉长牛/长熊的跨年趋势，识别生命周期
    """
    p = random.random()

    if p < 0.20:
        # 时序上的极短线常用于 ts_delta(CLOSE, 2) 等，识别瞬时脉冲
        return random.choice([2, 3, 5, 8])

    elif p < 0.80:
        # 核心区：时序信号需要更“丝滑”的窗口步进，涵盖 10-60 的细分区间
        # 注意这里增加了 30, 45 等非标窗口，用于增加时序信号的平滑度
        return random.choice([10, 15, 20, 22, 30, 45, 60])

    else:
        # 时序长窗甚至可以扩展到 500（两年线），用于识别极端估值回归
        return random.choice([120, 200, 250, 500])

class TSGPGenerator(GPDeapGenerator):
    """基于时序因子挖掘的GP因子生成器"""

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        时序因子挖掘专用算子集
        重点：序列压缩、权重衰减、形态识别
        """
        pset = gp.PrimitiveSetTyped("MAIN", [], Expr)

        # 1. 终端 (Terminals)
        for factor in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME','AMOUNT']:
            pset.addTerminal(1, Expr, name=factor)

        # 2. 窗口参数 (Constants)
        pset.addEphemeralConstant("rand_int", _random_window_int_timeseries, int)

        # 3. 基础算术 (用于构建复合指标)
        for name in ['add', 'sub', 'mul', 'div', 'max', 'min']:
            pset.addPrimitive(dummy, [Expr, Expr], Expr, name=f'oo_{name}')

        # 4. 核心时序统计算子 (分布特征)
        # 相比截面，增加了 skewness 和 kurtosis 来捕捉肥尾风险
        ts_stats_ops = [
            'ts_mean', 'ts_std_dev', 'ts_max', 'ts_min','ts_skewness', 'ts_kurtosis'
        ]
        for op in ts_stats_ops:
            pset.addPrimitive(dummy, [Expr, int], Expr, name=op)

        # 5. 权重衰减与平滑算子 (时序挖掘的灵魂)
        # 强调“近高远低”的逻辑，这是捕捉时序动量的关键
        ts_smooth_ops = [
            'ts_ema',  # 指数移动平均
            'ts_wma',  # 加权移动平均
            'ts_decay_linear'  # 线性衰减权重
        ]
        for op in ts_smooth_ops:
            pset.addPrimitive(dummy, [Expr, int], Expr, name=op)

        # 6. 动量与位置算子 (捕捉形态)
        # ts_arg_max/min 能告诉模型“高点在多久前”，是极强的择时特征
        ts_morph_ops = [
            'ts_delta', 'ts_returns', 'ts_rank','ts_arg_max', 'ts_arg_min'
        ]
        for op in ts_morph_ops:
            pset.addPrimitive(dummy, [Expr, int], Expr, name=op)

        # 7. 时序关联算子
        pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_corr')
        pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_covariance')

        # 8. 数值变换 (非线性增强)
        for op in ['abs_', 'log', 'sqrt', 'sign']:
            pset.addPrimitive(dummy, [Expr], Expr, name=op)

        return pset

    # def root_operator(self,df: pl.DataFrame):
    #     """强插一个根算子
    #
    #     比如挖掘出的因子是
    #     ts_SMA(CLOSE, 10)
    #     ts_returns(ts_SMA(CLOSE, 20),1)
    #
    #     这一步相当于在外层套一个ts_zscore，变成
    #     ts_zscore(ts_SMA(CLOSE, 10),120)
    #     ts_zscore(ts_returns(ts_SMA(CLOSE, 20),1),120)
    #
    #     注意，复制到其它工具验证时，一定要记得要带上根算子
    #
    #     这里只对GP_开头的因子添加根算子
    #
    #     """
    #     from polars_ta.prefix.wq import ts_zscore  # noqa
    #     from polars_ta.prefix.wq import cs_mad, cs_zscore  # noqa
    #
    #     def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    #         df = df.sort(by=['DATE'])
    #         # ========================================
    #         df = df.with_columns(
    #             ts_zscore(pl.col(r'^GP_\d+$'), 120),
    #         )
    #         return df
    #
    #     df = df.group_by('ASSET').map_groups(func_0_ts__asset)
    #     logger.warning("启用了根算子，复制到其它平台时记得手工添加")
    #
    #     return df
    #
    # def fitness_population(self,df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
    #     """种群fitness函数"""
    #     if df is None:
    #         return {}, {}
    #
    #
    #
    #     # 将所有数值列转换为 Float64，避免类型不匹配
    #     df = df.with_columns(cs.numeric().cast(pl.Float64))
    #
    #     # 将IC划分成训练集与测试集
    #     df_train = df.filter(pl.col('DATE') < split_date)
    #     df_valid = df.filter(pl.col('DATE') >= split_date)
    #
    #     # 时序相关性，没有IR
    #     ic_train = df_train.group_by('ASSET').agg([self.fitness_individual(X, label) for X in columns]).fill_nan(None)
    #     ic_valid = df_valid.group_by('ASSET').agg([self.fitness_individual(X, label) for X in columns]).fill_nan(None)
    #
    #     # 时序IC的多资产平均。可用来挖掘在多品种上适应的因子
    #     ic_train = ic_train.select(
    #         pl.when(cs.numeric().is_not_null().mean() >= 0.8).then(cs.numeric().mean()).otherwise(None))
    #     ic_valid = ic_valid.select(cs.numeric().mean())
    #
    #     ic_train = ic_train.to_dicts()[0]
    #     ic_valid = ic_valid.to_dicts()[0]
    #
    #     return ic_train, ic_valid
    #
    # def batched_exprs(self,batch_id, exprs_list, gen, label, split_date, df_input):
    #     """每代种群分批计算
    #
    #     由于种群数大，一次性计算可能内存不足，所以提供分批计算功能，同时也为分布式计算做准备
    #     """
    #     if len(exprs_list) == 0:
    #         return {}
    #
    #     tool = ExprTool()
    #     # 表达式转脚本
    #     codes, G = tool.all(exprs_list, style='polars', template_file='template.py.j2',
    #                         replace=False, regroup=True, format=True,
    #                         date='DATE', asset='ASSET', over_null="partition_by",
    #                         skip_simplify=True)
    #
    #     cnt = len(exprs_list)
    #     logger.info("{}代{}批 代码 开始执行。共 {} 条 表达式", gen, batch_id, cnt)
    #     tic = time.perf_counter()
    #
    #     globals_ = {}
    #     exec(codes, globals_)
    #     df_output = globals_['main'](df_input, ge_date_idx=0)
    #
    #     elapsed_time = time.perf_counter() - tic
    #     logger.info("{}代{}批 因子 计算完成。共用时 {:.3f} 秒，平均 {:.3f} 秒/条，或 {:.3f} 条/秒", gen, batch_id,
    #                 elapsed_time, elapsed_time / cnt, cnt / elapsed_time)
    #
    #     # 计算种群适应度
    #     ic_train, ic_valid = self.fitness_population(df_output, [k for k, v, c in exprs_list], label=label,
    #                                             split_date=split_date)
    #     logger.info("{}代{}批 适应度 计算完成", gen, batch_id)
    #
    #     # 样本内外适应度提取
    #     new_results = {}
    #     for k, v, c in exprs_list:
    #         v = str(v)
    #         new_results[v] = {'ic_train': get_fitness(k, ic_train),
    #                           'ic_valid': get_fitness(k, ic_valid),
    #                           }
    #     return new_results
    #
    # def fill_fitness(self,exprs_old, fitness_results):
    #     """填充fitness"""
    #     results = []
    #     for k, v, c in exprs_old:
    #         v = str(v)
    #         d = fitness_results.get(v, None)
    #         if d is None:
    #             logger.debug('{} 不合法/无意义/重复 等原因，在计算前就被剔除了', v)
    #         else:
    #             s0, s1 = d['ic_train'], d['ic_valid']
    #             # ic要看绝对值
    #             s0, s1 = abs(s0), abs(s1)
    #             # TODO 这地方要按自己需求定制，过滤太多可能无法输出有效表达式
    #             if s0 == s0:  # 非空
    #                 if s0 > 0.001:  # 样本内打分要大
    #                     if s0 * 0.6 < s1:  # 样本外打分大于样本内打分的70%
    #                         # 可以向fitness添加多个值，但长度要与weight完全一样
    #                         results.append((s0, s1))
    #                         continue
    #         # 可以向fitness添加多个值，但长度要与weight完全一样
    #         results.append((np.nan, np.nan))
    #
    #     return results
