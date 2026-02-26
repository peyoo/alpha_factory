from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import quantstats as qs
import pandas as pd
import webbrowser

from alpha_factory.config.base import settings


def generate_and_open_report(result: Dict[str, Any], factor_name: str):
    """
    使用 QuantStats 生成 HTML 报告并自动在浏览器打开
    """
    # 1. 数据转换：Polars -> Pandas
    series_df = result["series"].to_pandas()

    # 2. 准备收益率序列 (QuantStats 必须以 DatetimeIndex 作为索引)
    # 支持不同函数返回的列名：优先使用 'net_ret'，其次尝试 'target_ret'/'raw_ret'/'ret'，
    # 若无显式日收益列但有 'nav'，则从 'nav' 计算日收益率。
    returns_col_candidates = ["net_ret", "target_ret", "raw_ret", "ret"]
    returns = None
    for col in returns_col_candidates:
        if col in series_df.columns:
            returns = series_df.set_index("DATE")[col]
            break

    if returns is None and "nav" in series_df.columns:
        nav = series_df.set_index("DATE")["nav"]
        nav.index = pd.to_datetime(nav.index)
        returns = nav.pct_change().fillna(0)

    if returns is None:
        raise KeyError(
            "No suitable returns column found in result['series']; expected one of 'net_ret','target_ret','raw_ret','ret' or 'nav' to compute returns."
        )

    returns.index = pd.to_datetime(returns.index)
    returns.name = "Strategy"

    # 3. 定义输出路径：放到 OUTPUT_DIR/html_reports
    report_dir = Path(settings.OUTPUT_DIR) / "html_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_filename = (
        f"Report_{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path = (report_dir / report_filename).resolve()

    # 4. 生成报告
    # 如果你有基准数据，可以将 benchmark 替换为基准收益率序列
    qs.reports.html(
        returns,
        title=f"Factor Backtest Report: {factor_name}",
        output=str(report_path),
        show_sharpe_ratio=True,
    )

    # 5. 自动在默认浏览器中打开
    print(f"✅ 报告已生成: {report_path}")
    webbrowser.open(f"file://{report_path}")
