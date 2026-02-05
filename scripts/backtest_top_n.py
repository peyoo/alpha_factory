from loguru import logger

from alpha.evaluation.backtest.top_n import backtest_top_n, show_report, not_buyable_sellable
from alpha.data_provider.pool import main_small_pool
from alpha.gp.extra_terminal import add_extra_terminals

# 因子表达式：流动性加权动量减去波动率惩罚
# 这里的逻辑是：值越大越好，所以 ascending=False
expr = "-(cs_mad_zscore_mask(ts_mean(RET, 20)) + cs_mad_zscore_mask(ts_mean(TURNOVER_RATE, 20)))"
# expr = 'ts_decay_linear(max_(cs_mad_zscore_mask(RET), max_(cs_mad_zscore_mask(RET), log(RET)/ts_decay_linear(TURNOVER_RATE, 5))), 5)'
result = backtest_top_n(
    start="20190101",
    end="20251231",
    factor_col=f"IE_Factor={expr}",
    funcs=[main_small_pool, add_extra_terminals, not_buyable_sellable],
    ascending=False,  # 选因子值最大的前 N 名
    n_buy=30,         # 小微盘建议持仓稍微分散一点，增加稳定性
    n_sell=65,
    cost_rate=0.0015  # 适当调低费率看看因子的理论上限
)

logger.info("IE-Factor 回测摘要：")
print(result['metrics'])
print(result['yearly_summary'])

show_report(result, factor_name='CIRC_MV')
