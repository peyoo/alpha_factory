from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

import quantstats as qs
import pandas as pd
import webbrowser

from alpha_factory.config.base import settings

if TYPE_CHECKING:
    from alpha_factory.data_provider.benchmark import Benchmark


def generate_and_open_report(
    result: Dict[str, Any],
    factor_name: str,
    benchmark: Optional["Benchmark"] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    使用 QuantStats 生成 HTML 报告并自动在浏览器打开

    Args:
        result: 回测结果字典，包含 'series' 键
        factor_name: 因子名称
        benchmark: 可选的基准对象，用于对比分析
        start_date: 回测启始日期（YYYYMMDD），用于加载benchmark数据
        end_date: 回测结束日期（YYYYMMDD），用于加载benchmark数据
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

    # 4. 准备基准数据（可选）
    benchmark_series = None
    if benchmark is not None and start_date is not None and end_date is not None:
        from datetime import datetime as dt

        start_dt = dt.strptime(start_date, "%Y%m%d").date()
        end_dt = dt.strptime(end_date, "%Y%m%d").date()
        bench_df = benchmark.load_returns(start_dt, end_dt).collect().to_pandas()
        if not bench_df.empty:
            benchmark_series = bench_df.set_index("DATE")["ret"]
            benchmark_series.index = pd.to_datetime(benchmark_series.index)
            benchmark_series.name = f"Benchmark ({benchmark.name})"
            # 对齐日期：只保留策略和基准都有的日期
            common_dates = returns.index.intersection(benchmark_series.index)
            returns = returns.loc[common_dates]
            benchmark_series = benchmark_series.loc[common_dates]

    # 5. 生成报告
    qs.reports.html(
        returns,
        benchmark=benchmark_series,
        title=f"Factor Backtest Report: {factor_name}",
        output=str(report_path),
        show_sharpe_ratio=True,
    )

    # 6. 自动在默认浏览器中打开
    print(f"✅ 报告已生成: {report_path}")
    webbrowser.open(f"file://{report_path}")
