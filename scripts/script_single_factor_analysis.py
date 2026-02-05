from loguru import logger

from alpha.data_provider import DataProvider

from alpha.data_provider.pool import main_small_pool
from alpha.evaluation.batch.returns import batch_quantile_returns
from alpha.evaluation.single import single_factor_alpha_analysis
from alpha.gp.extra_terminal import add_extra_terminals
from alpha.gp.label import label_OO_for_tradable
from alpha.utils.schema import F

# expr = 'ts_corr(cs_rank_mask(CLOSE), cs_rank_mask(VOLUME), 20)'
# 中性120日换手率因子表达式
expr = "ts_mean(TURNOVER_RATE, 60)"
expr = 'cs_mad_zscore_mask(ts_corr(cs_rank_mask(HIGH), ts_std_dev(OPEN, 80), 10))'
logger.info(f"使用因子表达式: {expr}")
lf = DataProvider().load_data(
    start_date="20190101",
    end_date="20251231",
    funcs=[main_small_pool, add_extra_terminals,label_OO_for_tradable],
    column_exprs= ['ret_real_trade=OPEN[-2] / OPEN[-1] - 1',f"factor_1={expr}"],
    lookback_window=200
)

single_factor_alpha_analysis(lf, "factor_1", F.LABEL_FOR_RET,
                             period=1,n_bins=5,mode='long_only',
                             )

logger.info("使用 batch_quantile_returns 进行多因子评估")
df = batch_quantile_returns(lf, "factor_1",label_ret_col=F.LABEL_FOR_RET)
print(df)
