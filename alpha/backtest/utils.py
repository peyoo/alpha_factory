from datetime import datetime
from typing import Dict, Any

import quantstats as qs
import pandas as pd
import webbrowser
import os


def generate_and_open_report(result: Dict[str, Any], factor_name: str):
    """
    使用 QuantStats 生成 HTML 报告并自动在浏览器打开
    """
    # 1. 数据转换：Polars -> Pandas
    series_df = result['series'].to_pandas()

    # 2. 准备收益率序列 (QuantStats 必须以 DatetimeIndex 作为索引)
    returns = series_df.set_index('DATE')['net_ret']
    returns.index = pd.to_datetime(returns.index)
    returns.name = "Strategy"

    # 3. 定义输出路径
    report_filename = f"Report_{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.abspath(report_filename)

    # 4. 生成报告
    # 如果你有基准数据，可以将 benchmark 替换为基准收益率序列
    qs.reports.html(
        returns,
        title=f"Factor Backtest Report: {factor_name}",
        output=report_path,
        show_sharpe_ratio=True
    )

    # 5. 自动在默认浏览器中打开
    print(f"✅ 报告已生成: {report_path}")
    webbrowser.open(f"file://{report_path}")
