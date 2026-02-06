import polars as pl
from loguru import logger

def analysis_trades(trades: pl.DataFrame) -> dict:
    """
    分析交易明细数据，提供胜率、盈亏比、持仓周期等核心指标。

    参数:
        trades (pl.DataFrame): 包含字段 [F.ASSET, 'entry_date', 'exit_date',
                                     'entry_price', 'exit_price', 'pnl_ret', 'holding_periods']
    返回:
        dict: 包含多维度统计结果的字典
    """
    if trades.is_empty():
        logger.warning("交易明细为空，无法分析。")
        return {}

    # 1. 基础盈亏分类
    profits = trades.filter(pl.col("pnl_ret") > 0)
    losses = trades.filter(pl.col("pnl_ret") <= 0)

    # 2. 核心指标计算
    total_count = len(trades)
    win_count = len(profits)
    win_rate = win_count / total_count if total_count > 0 else 0

    avg_profit = profits["pnl_ret"].mean() if not profits.is_empty() else 0
    avg_loss = losses["pnl_ret"].mean() if not losses.is_empty() else 0 # 注意此处为负数

    # 盈亏比 (Profit/Loss Ratio)
    pnl_ratio = (avg_profit / abs(avg_loss)) if avg_loss != 0 else float('inf')

    # 3. 持仓周期统计
    avg_holding = trades["holding_periods"].mean()
    max_holding = trades["holding_periods"].max()

    # 4. 极端交易捕捉
    best_trade = trades.sort("pnl_ret", descending=True).head(1).to_dicts()[0]
    worst_trade = trades.sort("pnl_ret", descending=False).head(1).to_dicts()[0]

    # 5. 结果汇总
    metrics = {
        "count": total_count,                   # 总交易次数
        "win_rate": win_rate,                   # 胜率
        "pnl_ratio": pnl_ratio,                 # 盈亏比
        "avg_ret": trades["pnl_ret"].mean(),     # 笔均收益
        "avg_profit": avg_profit,               # 平均盈利单收益
        "avg_loss": avg_loss,                   # 平均亏损单收益
        "avg_holding_days": avg_holding,        # 平均持仓天数
        "max_holding_days": max_holding,        # 最长持仓天数
        "best_pnl": best_trade["pnl_ret"],      # 最大单笔盈利
        "worst_pnl": worst_trade["pnl_ret"],    # 最大单笔亏损
        "best_asset": best_trade.get("ASSET"),  # 最佳标的
        "worst_asset": worst_trade.get("ASSET") # 最差标的
    }

    # 打印格式化输出
    print("\n" + "🔍 交易明细深度透视" + " " + "="*30)
    print(f"📊 样本规模: {metrics['count']} 笔交易")
    print(f"📈 胜率/盈亏比: {metrics['win_rate']:.2%} | {metrics['pnl_ratio']:.2f}")
    print(f"⏱️ 平均持仓: {metrics['avg_holding_days']:.1f} 天 (最大 {metrics['max_holding_days']} 天)")
    print(f"💰 单笔均益: {metrics['avg_ret']:.2%}")
    print(f"✅ 平均盈利: {metrics['avg_profit']:.2%} | ❌ 平均亏损: {metrics['avg_loss']:.2%}")
    print(f"🚀 最佳单笔: {metrics['best_pnl']:.2%} ({metrics['best_asset']})")
    print(f"💀 最差单笔: {metrics['worst_pnl']:.2%} ({metrics['worst_asset']})")
    print("="*50 + "\n")

    return metrics
