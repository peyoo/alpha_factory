from abc import abstractmethod, ABC
from datetime import date, timedelta
from typing import Optional

import polars as pl
from loguru import logger

from alpha_factory.config.base import settings


class Benchmark(ABC):
    """基准时间序列管理器

    存储结构：Parquet (DATE: Date, ret: Float32)
    支持增量更新、lazy加载、因子表达式中的join以及评估层的择时计算。
    """

    def __init__(self, name: str, default_start_date: date = date(2010, 1, 1)):
        self.name = name
        self._cache_path = settings.BENCHMARKS_DIR / f"{self.name}.parquet"
        self._default_start_date = default_start_date
        self._data = None

    def _get_last_date(self) -> Optional[date]:
        """读取本地已有数据的最大日期，文件不存在则返回 None"""
        if not self._cache_path.exists():
            return None
        try:
            result = (
                pl.scan_parquet(self._cache_path)
                .select(pl.col("DATE").max())
                .collect()
                .item(0, 0)
            )
            return result
        except Exception as e:
            logger.debug(f"读取 {self.name} 最大日期失败: {e}")
            return None

    def update(self, end_date: Optional[date] = None) -> None:
        """增量同步：只拉取 max_date 之后的数据

        Args:
            end_date: 结束日期，默认为今天
        """
        last_date = self._get_last_date()
        start_date = (
            (last_date + timedelta(days=1)) if last_date else self._default_start_date
        )
        end_date = end_date or date.today()

        if start_date > end_date:
            logger.info(f"💡 {self.name} 已是最新数据 (至 {last_date})，无需更新")
            return

        logger.info(f"📡 正在更新 {self.name}，日期范围 {start_date} ~ {end_date}")
        new_df = self._fetch(start_date, end_date)

        if new_df.is_empty():
            logger.warning(f"⚠️ 获取 {self.name} 数据为空")
            return

        # 增量合并：保留已有数据 + 追加新数据，去重
        if self._cache_path.exists():
            existing = pl.read_parquet(self._cache_path)
            combined = pl.concat([existing, new_df]).unique("DATE").sort("DATE")
        else:
            combined = new_df.sort("DATE")

        # 确保目录存在
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self._cache_path)

        max_date = combined.select(pl.col("DATE").max()).item(0, 0)
        logger.info(f"✅ {self.name} 更新完成，数据至 {max_date}")

    def load_returns(self, start_date: date, end_date: date) -> pl.LazyFrame:
        """加载指定日期范围的基准收益率序列 (LazyFrame)

        返回结构: (DATE, ret)
        用途：
        1. 因子计算时作为虚拟资产join
        2. 评估层择时和基准对比

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pl.LazyFrame with columns [DATE, ret]
        """
        if not self._cache_path.exists():
            logger.warning(f"⚠️ {self.name} 数据文件不存在，请先调用 update()")
            return pl.LazyFrame(schema={"DATE": pl.Date, "ret": pl.Float32})

        return (
            pl.scan_parquet(self._cache_path)
            .filter((pl.col("DATE") >= start_date) & (pl.col("DATE") <= end_date))
            .select([pl.col("DATE"), pl.col("ret")])
        )

    def load_statistics(self, start_date: date, end_date: date) -> pl.LazyFrame:
        """加载指定日期范围的完整统计指标 (LazyFrame)

        基类默认只返回 (DATE, ret)。子类可Override以返回更多统计列。

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pl.LazyFrame with all available columns
        """
        if not self._cache_path.exists():
            logger.warning(f"⚠️ {self.name} 数据文件不存在，请先调用 update()")
            return pl.LazyFrame(schema={"DATE": pl.Date, "ret": pl.Float32})

        return pl.scan_parquet(self._cache_path).filter(
            (pl.col("DATE") >= start_date) & (pl.col("DATE") <= end_date)
        )

    @abstractmethod
    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """子类实现：从数据源拉取 [start_date, end_date] 范围的数据

        Returns:
            pl.DataFrame with columns [DATE: Date, ret: Float32]
            ret 为日收益率（百分比形式，例 0.01 代表 1%）
        """
        pass

    def load(self):
        """向后兼容的懒加载方法"""
        if self._cache_path.exists():
            self._data = pl.scan_parquet(self._cache_path)
        return self
