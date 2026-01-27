import random
import time
from datetime import datetime
from typing import Sequence

import expr_codegen.polars.code
import numpy as np
import polars as pl
from expr_codegen.expr import TS, CS, GP
from expr_codegen.tool import ExprTool
from loguru import logger
from polars import selectors as cs
from deap import gp
from alpha.gp.base import Expr, dummy
from alpha.gp.base import get_fitness

from alpha.gp.generator import GPDeapGenerator

def get_groupby_from_tuple(tup, func_name, drop_cols):
    """从传入的元组中生成分组运行代码"""
    prefix2, *_ = tup

    if len(drop_cols)>0:
        drop_str = f'.drop(*{drop_cols})'
    else:
        drop_str = ""

    if prefix2 == TS:
        # 组内需要按时间进行排序，需要维持顺序
        prefix2, asset = tup
        return f'df = {func_name}(df){drop_str}'
    if prefix2 == CS:
        prefix2, date = tup
        return f'df = {func_name}(df){drop_str}'
    if prefix2 == GP:
        prefix2, date, group = tup
        return f'df = {func_name}(df){drop_str}'

    return f'df = {func_name}(df){drop_str}'

# 打个补丁，取消排序
expr_codegen.polars.code.get_groupby_from_tuple = get_groupby_from_tuple

# 1. 添加时序窗口常量
def _random_window_int() -> int:
    """
    生成随机窗口大小常量
    这些窗口大小覆盖了常见的短中长期趋势分析需求。
    全周期窗口采样：
    - 极短线 (3, 5): 25% 概率
    - 短中线 (10, 20, 40): 50% 概率 (核心交易频率)
    - 长线 (60, 120, 250): 25% 概率 (季/半年/年线)
    """
    p = random.random()
    if p < 0.25:
        return random.choice([3, 5])
    elif p < 0.75:
        return random.choice([10, 20, 40])  # 对应 2周, 1月, 2月
    else:
        return random.choice([60, 120, 250])  # 对应 1季, 半年, 1年


class CSGPGenerator(GPDeapGenerator):
    """基于截面因子挖掘的GP因子生成器"""

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        精简版算子集：专为 LightGBM/ElasticNet 特征工程设计
        所有算子均返回数值类型，移除了导致 SchemaError 的逻辑算子
        """
        # 直接使用 Expr 作为标识，默认即为浮点数序列
        pset = gp.PrimitiveSetTyped("MAIN", [], Expr)

        # 1. 终端 (Terminals)
        # AMOUNT有极强的市值效应，不适合截面因子挖掘
        for factor in ['OPEN', 'HIGH', 'LOW', 'CLOSE','TURNOVER_RATE']:
            pset.addTerminal(1, Expr, name=factor)

        # 2. 窗口参数 (Constants)
        pset.addEphemeralConstant("rand_int", _random_window_int, int)

        # 3. 基础算术算子 (线性模型 ElasticNet 无法自学除法和乘法交互)
        for name in ['add', 'sub', 'mul', 'div', 'max', 'min']:
            pset.addPrimitive(dummy, [Expr, Expr], Expr, name=f'oo_{name}')

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
            'ts_mean', 'ts_std_dev', 'ts_max', 'ts_min','ts_delta', 'ts_returns', 'ts_rank', 'ts_skewness'
        ]
        for op in ts_ops:
            pset.addPrimitive(dummy, [Expr, int], Expr, name=op)
        # 时序相关性与协方差 (增强模型对多因子联动的感知),极其强大。
        # 例如成交量和价格的相关性（价升量增 vs 价升量减）。这是 ML 模型绝对无法通过原始特征自己算出来的交互信息。
        pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_corr')
        # 量化两个序列的同步运动程度，是 ts_corr 的未归一化版本，能保留数值规模信息。
        # pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_covariance')

        # 5. 截面归一化算子 (机器学习模型的关键：提供样本间的相对坐标)
        # 精细化建议：'cs_demean','cs_qcut'
        cs_ops = ['cs_rank', 'cs_zscore']
        for op in cs_ops:
            pset.addPrimitive(dummy, [Expr], Expr, name=op)

        # 6. 数值变换 (改善线性模型的特征分布)
        # 可选:sigmoid / tanh (软截断)
        for op in ['abs_', 'log', 'sqrt', 'sign']:
            pset.addPrimitive(dummy, [Expr], Expr, name=op)

        return pset


    def fitness_population(self,df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
        """种群fitness函数"""
        if df is None:
            return {}, {}, {}, {}

        # 将所有数值列转换为 Float64，避免类型不匹配
        df = df.with_columns(cs.numeric().cast(pl.Float64))

        df = df.group_by('DATE').agg(
            [self.fitness_individual(X, label) for X in columns]
        ).sort(by=['DATE']).fill_nan(None)
        # 将IC划分成训练集与测试集
        df_train = df.filter(pl.col('DATE') < split_date)
        df_valid = df.filter(pl.col('DATE') >= split_date)

        # TODO 有效数不足，生成的意义不大，返回null, 而适应度第0位是nan时不加入名人堂
        # cs.numeric().count() / cs.numeric().len() >= 0.5
        # cs.numeric().count() >= 30
        ic_train = df_train.select(
            pl.when(cs.numeric().is_not_null().mean() >= 0.5).then(cs.numeric().mean()).otherwise(None))
        ic_valid = df_valid.select(cs.numeric().mean())
        ir_train = df_train.select(cs.numeric().mean() / cs.numeric().std(ddof=0))
        ir_valid = df_valid.select(cs.numeric().mean() / cs.numeric().std(ddof=0))

        ic_train = ic_train.to_dicts()[0]
        ic_valid = ic_valid.to_dicts()[0]
        ir_train = ir_train.to_dicts()[0]
        ir_valid = ir_valid.to_dicts()[0]

        return ic_train, ic_valid, ir_train, ir_valid

    def batched_exprs(self,batch_id, exprs_list, gen, label, split_date, df_input):
        """每代种群分批计算

        由于种群数大，一次性计算可能内存不足，所以提供分批计算功能，同时也为分布式计算做准备
        """
        if len(exprs_list) == 0:
            return {}

        tool = ExprTool()
        # 表达式转脚本
        codes, G = tool.all(exprs_list, style='polars', template_file='template.py.j2',
                            replace=False, regroup=True, format=True,
                            date='DATE', asset='ASSET', over_null=None,
                            skip_simplify=True)


        cnt = len(exprs_list)
        logger.info("{}代{}批 代码 开始执行。共 {} 条 表达式", gen, batch_id, cnt)
        tic = time.perf_counter()

        globals_ = {}
        exec(codes, globals_)
        df_output = globals_['main'](df_input.lazy(), ge_date_idx=0).collect()

        elapsed_time = time.perf_counter() - tic
        logger.info("{}代{}批 因子 计算完成。共用时 {:.3f} 秒，平均 {:.3f} 秒/条，或 {:.3f} 条/秒", gen, batch_id,
                    elapsed_time, elapsed_time / cnt, cnt / elapsed_time)

        # 计算种群适应度
        ic_train, ic_valid, ir_train, ir_valid = self.fitness_population(df_output, [k for k, v, c in exprs_list],
                                                                    label=label, split_date=split_date)
        logger.info("{}代{}批 适应度 计算完成", gen, batch_id)

        # 样本内外适应度提取
        new_results = {}
        for k, v, c in exprs_list:
            v = str(v)
            new_results[v] = {'ic_train': get_fitness(k, ic_train),
                              'ic_valid': get_fitness(k, ic_valid),
                              'ir_train': get_fitness(k, ir_train),
                              'ir_valid': get_fitness(k, ir_valid),
                              }
        return new_results

    def fill_fitness(self,exprs_old, fitness_results):
        """填充fitness"""
        results = []
        for k, v, c in exprs_old:
            v = str(v)
            d = fitness_results.get(v, None)
            if d is None:
                logger.debug('{} 不合法/无意义/重复 等原因，在计算前就被剔除了', v)
            else:
                s0, s1, s2, s3 = d['ic_train'], d['ic_valid'], d['ir_train'], d['ir_valid']
                # ic要看绝对值
                s0, s1, s2, s3 = abs(s0), abs(s1), s2, s3
                # TODO 这地方要按自己需求定制，过滤太多可能无法输出有效表达式
                if s0 == s0:  # 非空
                    if s0 > 0.001:  # 样本内打分要大
                        if s0 * 0.6 < s1:  # 样本外打分大于样本内打分的60%
                            # 可以向fitness添加多个值，但长度要与weight完全一样
                            results.append((s0, s1))
                            continue
            # 可以向fitness添加多个值，但长度要与weight完全一样
            results.append((np.nan, np.nan))

        return results

    def export_hof_to_csv(self, hof, globals_, filename="cs_best_factors.csv"):
        """
        导出截面挖掘的名人堂因子，包含 IC 和 IR 指标
        """
        import pandas as pd
        from alpha.gp.base import population_to_exprs

        # 1. 解析表达式
        exprs_list = population_to_exprs(hof, globals_)

        # 2. 从缓存中获取详细的指标
        # 注意：batched_exprs 计算时会将结果存入 fitness_cache.pkl
        # 这里我们直接从个体对象的 fitness 属性和计算逻辑中还原
        data = []
        for (k, v, c), ind in zip(exprs_list, hof):
            # 获取 DEAP 存储的绝对值 IC (因为 fill_fitness 做了 abs())
            ic_train_abs, ic_valid_abs = ind.fitness.values

            data.append({
                "factor_name": k,
                "abs_ic_train": round(ic_train_abs, 5),
                "abs_ic_valid": round(ic_valid_abs, 5),
                "complexity": c,
                "expression": v,
                "raw_tree": str(ind),
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

        df = pd.DataFrame(data)
        # 按照样本内 IC 排序
        df = df.sort_values("abs_ic_train", ascending=False)

        output_path = self.save_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info(f"📊 截面因子报告已生成: {output_path}")
        return df
