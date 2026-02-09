from loguru import logger

from alpha_factory.data_provider import DataProvider

from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.evaluation.single.single import single_factor_alpha_analysis
from alpha_factory.gp.extra_terminal import add_extra_terminals
from alpha_factory.gp.label import label_OO_for_tradable, label_OO_for_IC
from alpha_factory.utils.schema import F

# expr = 'ts_corr(cs_rank_mask(CLOSE), cs_rank_mask(VOLUME), 20)'
# 中性120日换手率因子表达式
# expr = "ts_max(cs_mad_zscore_mask(ts_skewness(VWAP, 15)), 10)"
expr = 'cs_mad_zscore_mask(ts_skewness(HIGH, 20))'

logger.info(f"使用因子表达式: {expr}")
lf = DataProvider().load_data(
    start_date="20190101",
    end_date="20251231",
    funcs=[main_small_pool, add_extra_terminals,label_OO_for_IC, label_OO_for_tradable],
    column_exprs= [f"factor_1={expr}"],
    lookback_window=200
)

single_factor_alpha_analysis(lf, "factor_1", F.LABEL_FOR_RET,
                             period=1,n_bins=5,mode='long_only',
                             )
