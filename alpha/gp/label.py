"""
添加打标签相关的功能
"""
from functools import partial

import polars as pl
from polars import when

from alpha.utils.schema import F


def label_OO_for_IC(lf: pl.LazyFrame, label_window=1, mask_col=F.POOL_MASK) -> pl.LazyFrame:
    """
    计算未来开盘收益率标签，并进行基于池内标的的局部中性化处理
    主要针对用于 IC 计算的标签处理逻辑：
    1. 计算未来开盘收益率
    2. 过滤掉无效标签行
    3. 基于 Mask 的局部中性化
    参数：
    - lf: 输入的 LazyFrame，必须包含 OPEN 列和 ASSET 列
    - label_window: 未来收益率的窗口大小，默认为 1（即计算下一个交易日的开盘收益率）
    - mask_col: 用于局部中性化的池掩码列名，默认为 F.POOL_MASK
    返回值：
    - 处理后的 LazyFrame，包含新的标签列 LABEL_OO_{label_window}
    备注：
    - 局部中性化是指仅使用池内标的的数据来计算均值和标准差，从而避免池外异常值对标签分布的影响。
    - 标签值被裁剪在 [-3, 3] 范围内，以减少极端值的影响。

    """

    label_y = F.LABEL_FOR_IC

    # 1. 计算原始收益率
    lf = lf.with_columns([
        ((pl.col("OPEN").shift(-(label_window + 1)).over(F.ASSET) /
          pl.col("OPEN").shift(-1).over(F.ASSET)) - 1).alias(label_y)
    ])

    # 2. 过滤掉无效标签行
    lf = lf.filter(pl.col(label_y).is_not_null())

    # 3. 核心：基于 Mask 的局部中性化
    # 只让池内标的参与均值和标准差的计算，防止池外异常值干扰标签分布
    lf = lf.with_columns([
        when(pl.col(mask_col))
        .then(pl.col(label_y))
        .otherwise(None)
        .alias("_tmp_label_in_pool")
    ])

    lf = lf.with_columns([
        (
                (pl.col(label_y) - pl.col("_tmp_label_in_pool").mean().over(F.DATE)) /
                (pl.col("_tmp_label_in_pool").std().over(F.DATE) + 1e-6)
        )
        .clip(-3, 3)
        .alias(label_y)
    ])

    return lf.drop("_tmp_label_in_pool")

label_OO_1 = partial(label_OO_for_IC, label_window=1)


def label_OO_for_tradable(lf: pl.LazyFrame,
                          label_window=1,
                          mask_col=F.POOL_MASK) -> pl.LazyFrame:
    """
    针对收益率类适应度的标签函数
    1. 计算 Open-to-Open 收益率
    2. 考虑一字涨停买不入、一字跌停卖不出
    3. 扣除交易税费
    """
    # 定义列名
    label_y = F.LABEL_FOR_RET

    # 1. 基础收益率计算 (未来 T+1 开盘买入, T+1+window 开盘卖出)
    # 注意：shift(-1) 是明天的开盘价，shift(-(1+window)) 是卖出时的开盘价
    lf = lf.with_columns([
        ((pl.col("OPEN").shift(-(label_window + 1)).over("ASSET") /
          pl.col("OPEN").shift(-1).over("ASSET")) - 1).alias(label_y),
        # 获取 T+1 日（买入日）是否一字涨停
        (pl.col("LOW").shift(-1).over("ASSET") == pl.col("HIGH").shift(-1).over("ASSET")).alias("_is_limit_T1"),
        # 获取 T+1 日（买入日）是否触达涨停 (简单逻辑：收盘涨幅 > 9.8%)
        ((pl.col("CLOSE").shift(-1).over("ASSET") / pl.col("CLOSE").over("ASSET") - 1) > 0.098).alias("_at_limit_T1")
    ])

    # 2. 核心交易逻辑过滤
    lf = lf.with_columns([
        pl.when(
            # 情况 A: 不在股票池内 -> 收益为 0
            (~pl.col(mask_col)) |
            # 情况 B: T+1日一字涨停或开盘即涨停 -> 买不入，收益为 0
            (pl.col("_is_limit_T1") & pl.col("_at_limit_T1"))
        )
        .then(0.0)
        # 情况 C: 可成交 -> 原始收益 - 双边费用 (买入+卖出)
        .otherwise(pl.col(label_y))
        .alias(label_y)
    ])

    # 3. 极端值平滑 (可选，收益率类不建议 Z-Score，建议直接 Percentile Clip)
    # 比如：单票单期收益最高只计 20%，防止个别极端值误导 GA
    lf = lf.with_columns(
        pl.col(label_y).clip(-0.2, 0.2).alias(label_y)
    )

    return lf.drop(["_is_limit_T1", "_at_limit_T1"])
