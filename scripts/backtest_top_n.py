from loguru import logger
import importlib

# 可用的替代/相关工具（确保这些模块在仓库中存在）
from alpha_factory.evaluation.backtest import daily_evolving as backtest_mod
from alpha_factory.evaluation.analysis.returns import show_report
from alpha_factory.data_provider.pool import main_small_pool
from alpha_factory.gp.extra_terminal import add_extra_terminals

# 尝试动态导入可能不存在的历史模块 top_n，避免在 import 时因 ImportError 中断
backtest_top_n = None
not_buyable_sellable = None
try:
    _mod = importlib.import_module("alpha_factory.evaluation.backtest.top_n")
    backtest_top_n = getattr(_mod, "backtest_top_n", None)
    not_buyable_sellable = getattr(_mod, "not_buyable_sellable", None)
except Exception:
    logger.debug(
        "alpha_factory.evaluation.backtest.top_n 模块在当前环境中不可用，已采用安全回退策略。"
    )

# 因子表达式：流动性加权动量减去波动率惩罚
# 这里的逻辑是：值越大越好，所以 ascending=False
expr = "-(cs_mad_zscore_mask(ts_mean(RET, 20)) + cs_mad_zscore_mask(ts_mean(TURNOVER_RATE, 20)))"


def main():
    """主运行入口：在命令行运行时才会执行回测调用，导入该模块不会触发回测逻辑。"""
    logger.info("准备运行 IE-Factor 回测（脚本运行）")

    # ----- 调用回测 -----
    if callable(backtest_top_n):
        # 历史接口可用时按原逻辑调用（保持向后兼容）
        result = backtest_top_n(
            start="20190101",
            end="20251231",
            factor_col=f"IE_Factor={expr}",
            funcs=[main_small_pool, add_extra_terminals, not_buyable_sellable],
            ascending=False,  # 选因子值最大的前 N 名
            n_buy=30,  # 小微盘建议持仓稍微分散一点，增加稳定性
            n_sell=65,
            cost_rate=0.0015,  # 适当调低费率看看因子的理论上限
        )
    else:
        # 若不存在历史 top_n 实现，则给出明确的提示并退出。
        # 在提示中引用 backtest_mod.__name__（daily_evolving）以展示可替代接口并避免静态检查警告
        raise RuntimeError(
            "alpha_factory.evaluation.backtest.top_n 模块不可用。\n"
            f"请准备并传入 df_input（pl.DataFrame / pl.LazyFrame），然后调用 {backtest_mod.__name__}.backtest_daily_evolving(df_input, factor_col=..., ...) 。\n"
            "或者在你的环境中安装/提供 top_n 模块以恢复原先行为。"
        )

    logger.info("IE-Factor 回测摘要：")

    # 兼容多种返回结构：如果 result 是包含 daily_results 的 dict，则向 show_report 传入 daily DataFrame；
    # 如果 result 包含 metrics/yearly_summary（历史格式），则尽量打印这些字段。
    if isinstance(result, dict):
        if "daily_results" in result:
            # daily_results 通常是 pl.DataFrame
            df_daily = result["daily_results"]
            # show_report 在 show=True 时通常会生成并打开 HTML，返回值可能为空 dict
            show_report(df_daily, factor="CIRC_MV", show=True)
            # 额外打印兼容字段（如果存在）
            if "metrics" in result:
                print(result.get("metrics"))
            if "yearly_summary" in result:
                print(result.get("yearly_summary"))
        else:
            # 兼容旧返回值（metrics/yearly_summary）
            if "metrics" in result:
                print(result["metrics"])
            if "yearly_summary" in result:
                print(result["yearly_summary"])
    else:
        # 未知返回类型，直接打印 repr
        print(repr(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("回测执行失败: {}", e)
        raise
