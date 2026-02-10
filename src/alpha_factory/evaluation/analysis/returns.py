import os
import webbrowser
import polars as pl
import pandas as pd
import quantstats as qs
from loguru import logger

from alpha_factory.config.base import settings
from alpha_factory.utils.schema import F


def show_report(
    df_daily: pl.DataFrame, factor="", ret_col="NET_RET", show=True
) -> dict:
    """
    分析收益率数据，生成专业可视化 HTML 报告并自动打开。

    参数:
        df_daily: 包含 [F.DATE, 'NET_RET'] 的每日回测结果表
        factor: 因子名称，用于报告命名
        show: 是否生成 HTML 报告并自动打开
    """
    if df_daily.is_empty():
        logger.error("❌ 每日收益数据为空，无法生成报告。")
        return {}

    # --- 1. 数据转换: Polars -> Pandas (quantstats 兼容型) ---
    df_pd = df_daily.select([pl.col(F.DATE), pl.col(ret_col)]).to_pandas()
    # 转换为 Series 并处理索引
    returns = df_pd.set_index(F.DATE)[ret_col]
    returns.index = pd.to_datetime(returns.index)

    # --- 2. 核心指标打印 ---
    # 使用 qs 计算几个关键值用于日志输出
    sharpe = qs.stats.sharpe(returns)
    cagr = qs.stats.cagr(returns)
    max_dd = qs.stats.max_drawdown(returns)

    logger.info(
        f"📈 策略初评 | Sharpe: {sharpe:.2f} | CAGR: {cagr:.2%} | MaxDD: {max_dd:.2%}"
    )

    if show:
        # --- 3. 生成报告与展示 ---
        # 这里的 settings.OUTPUT_DIR 建议根据你的项目实际配置
        # 临时演示使用当前路径下的 output/html_reports
        report_dir = settings.OUTPUT_DIR / "html_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Report_{factor}_{timestamp}.html"
        output_path = report_dir / filename

        # 生成全量 HTML 报告
        qs.reports.html(
            returns, title=f"Factor Strategy: {factor}", output=str(output_path)
        )
        logger.info(f"📊 报告已成功生成: {output_path}")

        # 自动在浏览器打开
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")

        # 显示了，为了节省时间就不返回详细字典了
        return {}

    # 提取详细指标字典供后续保存
    return qs.reports.metrics(returns, display=False).to_dict()
