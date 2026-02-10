from loguru import logger

from alpha_factory.data_provider import DataProvider
from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.evaluation.analysis.returns import show_report
from alpha_factory.evaluation.analysis.trades import analysis_trades
from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.gp.extra_terminal import add_extra_terminals

# 导入你的配置、字段库和回测函数
from alpha_factory.utils.schema import F


def run_factor_test():
    # --- 1. 数据加载 ---
    exprs = [
        "ts_mean(TURNOVER_RATE, 60)",
        "ts_std_dev(TURNOVER_RATE,60)",
        "ts_decay_linear(TURNOVER_RATE,60)ts_std_dev(AMOUNT,60)",
        "ts_mean(AMOUNT,60)",
        "ts_decay_linear(AMOUNT,60)",
        "ts_max(AMOUNT,60)",
        "-ts_corr(cs_rank(VOLUME), cs_rank(ts_returns(CLOSE, 1)), 10)",
    ]
    expr = exprs[len(exprs) - 1]  # 选择第7个表达式进行测试
    # expr = 'ts_std_dev(cs_mad_zscore_mask(VWAP), 30)'
    # expr = 'ts_rank(cs_rank_mask(cs_rank_mask(VWAP)), 15)'
    # 中性30日换手率波动率因子表达式
    # expr = "ts_std_dev(TURNOVER_RATE,40)"
    # # ILLIQ20 非流动性因子表达
    # expr = "ts_mean(abs(ts_returns(CLOSE,1))/AMOUNT,20)"

    factor_name = "factor_1"

    logger.info(f"正在加载数据并计算因子: {expr}")

    # 假设 DataProvider 已经配置好
    # 注意：lookback_window=200 确保了 ts_skewness(20) 有足够的历史数据预热
    lf = DataProvider().load_data(
        start_date="20190101",
        end_date="20251231",
        funcs=[main_small_pool, add_extra_terminals],
        column_exprs=[f"{factor_name}={expr}"],
        lookback_window=200,
        cache_path="md5",
    )

    # --- 2. 运行回测 ---
    # 我们使用之前定义的 backtest_daily_evolving
    # 设置：买入前100，卖出阈值120，费率0.15%
    results = backtest_daily_evolving(
        df_input=lf,
        factor_col=factor_name,
        n_buy=10,
        sell_rank=30,
        cost_rate=0.0025,
        exec_price=F.OPEN,  # 也可以根据你的 funcs 结果使用 VWAP
        ascending=True,  # 偏度因子通常测试“高偏度”或“低偏度”的 Alpha 性
    )

    # --- 3. 结果提取与展示 ---
    daily_df = results["daily_results"]
    trade_df = results["trade_details"]

    # 打印核心绩效摘要
    analysis_trades(trade_df)
    show_report(daily_df, factor=factor_name, ret_col="NET_RET")


if __name__ == "__main__":
    run_factor_test()
