from loguru import logger
from typing import Dict
import sys
import polars as pl

# 添加项目路径
sys.path.insert(0, '/Users/yongpeng/Documents/github/alpha_factory')

from alpha.data_provider import DataProvider
from alpha.data_provider.pool import main_small_pool
from alpha.evaluation.batch.turnover_decay import batch_calc_factor_turnover_by_autocorr
from alpha.gp.extra_terminal import add_extra_terminals
from alpha.gp.label import label_OO_for_tradable, label_OO_for_IC
from alpha.utils.schema import F

# ✓ 保留原始表达式与因子的映射
exprs = [
    "ts_mean(TURNOVER_RATE, 10)",
    "ts_mean(TURNOVER_RATE, 40)",
    "ts_mean(TURNOVER_RATE, 60)",
    "ts_mean(TURNOVER_RATE, 120)",
    "ts_rank(ts_mean(TURNOVER_RATE, 40),10)",
    "ts_rank(cs_mad_zscore_mask(VWAP), 80)",
    "ts_std_dev(cs_mad_zscore_mask(VWAP), 30)",
    "ts_rank(cs_rank_mask(cs_rank_mask(VWAP)), 15)"
]

# 创建因子名与表达式的映射字典
factor_expr_map: Dict[str, str] = {
    f"factor_{i+1}": expr for i, expr in enumerate(exprs)
}

# 用于后续添加到数据管道中
exprs_with_names = [f"factor_{i+1}={expr}" for i, expr in enumerate(exprs)]

logger.info(f"使用因子表达式: {exprs_with_names}")

lf = DataProvider().load_data(
    start_date="20190101",
    end_date="20251231",
    funcs=[main_small_pool, add_extra_terminals,label_OO_for_IC,label_OO_for_tradable],
    column_exprs=[f'{F.LABEL_FOR_RET}=OPEN[-2] / OPEN[-1] - 1', *exprs_with_names],
    lookback_window=200
)
lf = lf.collect()
logger.info("使用 batch_quantile_returns 进行多因子评估")

# 调用函数，明确传递 label_col 参数
df_result = batch_calc_factor_turnover_by_autocorr(
    lf,
    factors=r"^factor_.*",
    # label_col=F.LABEL_FOR_RET,  # ← 显式传递标签列
    # descending= True
)

# ✓ 后处理：添加 expression 列到第二列
df_result = (
    df_result
    .with_columns(
        # 使用 factor 列进行 left join 获取 expression
        pl.col("factor").map_elements(
            lambda f: factor_expr_map.get(f, "unknown"),
            return_dtype=pl.String
        ).alias("expression")
    )
    # 重新排序列：factor → expression → 其他列
    .select([
        "factor",
        "expression",
        *[col for col in df_result.columns if col not in ["factor", "expression"]]
    ])
)

logger.info(f"评估结果:\n{df_result}")
