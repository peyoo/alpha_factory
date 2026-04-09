"""沪深300指数基准实现

时间序列：从 Tushare 拉取 000300.SH（沪深300指数）
字段：日期 + 日收益率（百分比转小数）
"""

from datetime import date

import polars as pl
import tushare as ts
from loguru import logger

from alpha_factory.config.base import settings
from alpha_factory.data_provider.benchmark import Benchmark
from alpha_factory.data_provider.tushare_service import RateLimiter


class HS300Benchmark(Benchmark):
    """沪深300指数基准

    数据源：Tushare Pro / index_daily
    代码：000300.SH
    """

    def __init__(self):
        super().__init__(name="HS300")
        self._token = getattr(settings, "TUSHARE_TOKEN", None) or None
        self._is_vip = getattr(settings, "IS_VIP", True)
        self._rate_limiter = RateLimiter(is_vip=self._is_vip)

    def _init_tushare(self):
        """初始化 Tushare API"""
        if not self._token:
            raise ValueError("❌ TUSHARE_TOKEN 未配置")
        return ts.pro_api(self._token)

    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """从 Tushare 拉取指数日线数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pl.DataFrame with columns [DATE, ret]
        """
        try:
            pro = self._init_tushare()
        except ValueError as e:
            logger.error(str(e))
            raise

        logger.info(
            f"📡 从 Tushare 拉取 HS300 "
            f"({start_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})"
        )

        self._rate_limiter.wait()
        df = pro.index_daily(
            ts_code="000300.SH",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            fields="trade_date,pct_chg",
        )

        if df is None or df.empty:
            logger.warning(f"⚠️ Tushare 返回空数据 {start_date} ~ {end_date}")
            return pl.DataFrame(schema={"DATE": pl.Date, "ret": pl.Float32})

        # 转换为 Polars 并计算日收益率
        result = (
            pl.from_pandas(df)
            .with_columns(
                pl.col("trade_date").str.to_date("%Y%m%d").alias("DATE"),
                (pl.col("pct_chg") / 100.0).cast(pl.Float32).alias("ret"),
            )
            .select(["DATE", "ret"])
            .sort("DATE")
        )

        logger.info(f"✓ 成功获取 {result.height} 条记录")
        return result
