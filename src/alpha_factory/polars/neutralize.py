import polars as pl
from polars import when
import polars_ols as pls
from polars_ols.least_squares import OLSKwargs

from alpha_factory.utils.schema import F

# 配置 OLS 逻辑：丢弃空值（只在池内算），使用 SVD 增加数值稳定性
_ols_kwargs = OLSKwargs(null_policy='drop', solve_method='svd')


def cs_neutralize_mask(y: pl.Expr,
                       industry: pl.Expr = pl.col(F.INDUSTRY),
                       mask: pl.Expr = pl.col(F.POOL_MASK),
                       fill_null: float = 0.0) -> pl.Expr:
    """
    横截面行业中性化：回归取残差
    y: 因子Expr
    industry: 行业分类Expr (通常是 L1/L2 编码)
    mask: 股票池掩码
    """
    if mask is None:
        mask = pl.lit(True)

    # 1. 核心：只保留池内数据参与回归，池外设为 null
    # polars_ols 会根据 null_policy='drop' 自动忽略这些行
    y_in_pool = when(mask).then(y).otherwise(None)

    # 2. 执行回归取残差
    # 这里的 industry 会被自动识别为 Categorical/Int 并作为分类变量处理（取决于库版本）
    # 或者我们可以将其传入作为 one-hot (pls 处理多列 x)
    # 我们这里假设 industry 是分类特征，或者使用 pls.compute_least_squares 的多列模式
    res = pls.compute_least_squares(
        y_in_pool,
        industry,
        mode='residuals',
        ols_kwargs=_ols_kwargs
    )

    # 3. 填充缺失值：中性化残差均值为 0，所以缺失填 0.0 是最合理的
    return when(mask).then(res).otherwise(fill_null)
