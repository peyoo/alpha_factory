from __future__ import annotations
import datetime
import threading
from pathlib import Path
from typing import Optional, Union, List

import polars as pl

from alpha.utils.config import settings
from alpha.utils.logger import logger


class TradeCalendarManager:
    """
    交易日历管理器 (Registry)。

    设计逻辑：
    - 接入层友好：输入输出均支持 Python 原生 datetime.date。
    - 存储层标准：使用 Parquet 落地 pl.Date，格式统一。
    - 计算层高性能：利用 Polars Series.search_sorted 实现毫秒级的日期偏移检索。
    """

    def __init__(self, path: Optional[Path] = None):
        self._lock = threading.Lock()
        # 依赖注入：优先使用 settings 定义的协议
        self.path = path or (settings.WAREHOUSE_DIR / settings.CALENDAR_FILENAME)
        self.start_date_str = settings.SYSTEM_START_DATE
        self.schema = settings.CALENDAR_SCHEMA

        # 内部缓存：仅保存 is_open=1 的日期，用于高速偏移计算
        self._df = pl.DataFrame(schema=self.schema)
        self._trade_days_cache: Optional[pl.Series] = None

        with self._lock:
            self._load_locked()

    def _load_locked(self):
        """内部加载，并强制根据 START_DATE 进行数据截断。"""
        if self.path.exists():
            try:
                start_dt = datetime.datetime.strptime(self.start_date_str, "%Y%m%d").date()
                self._df = (
                    pl.read_parquet(self.path)
                    .cast(self.schema)
                    .filter(pl.col("date") >= start_dt)
                    .sort("date")
                )
                self._refresh_cache_locked()
                if self._df.height > 0:
                    logger.debug(f"✓ 交易日历加载成功 (起始: {self.start_date_str}, 截止: {self._df['date'].max()})")
            except Exception as e:
                logger.error(f"加载本地交易日历失败: {e}")

    def _refresh_cache_locked(self):
        """刷新交易日缓存 Series。"""
        if self._df.height > 0:
            self._trade_days_cache = (
                self._df.filter(pl.col("is_open") == 1)
                .get_column("date")
                .sort()
            )

    def sync_from_tushare(self, force: bool = False, days_buffer: int = 10) -> None:
        """
        同步 Tushare 交易日历。
        策略：一次性同步到远期 (2035年)，只要未来覆盖足够，就不再触碰 API 积分。
        """
        today = datetime.date.today()

        if not force and self.path.exists() and self._df.height > 0:
            max_date = self._df["date"].max()
            if max_date >= (today + datetime.timedelta(days=days_buffer)):
                logger.debug(f"本地日历覆盖充足 (至 {max_date})，跳过同步。")
                return

        try:
            import tushare as ts
            token = getattr(settings, "TUSHARE_TOKEN", None)
            if not token:
                logger.error("未配置 TUSHARE_TOKEN")
                return

            pro = ts.pro_api(token)
            logger.info(f"正在从 Tushare 同步全量交易日历 (起始: {self.start_date_str})...")

            df_raw = pro.trade_cal(
                exchange='SSE',
                start_date=self.start_date_str,
                end_date='20351231'
            )

            if df_raw is None or df_raw.empty:
                return

            snapshot = (
                pl.from_pandas(df_raw)
                .select([
                    pl.col("cal_date").str.to_date("%Y%m%d").alias("date"),
                    pl.col("is_open").cast(pl.Int8),
                    pl.col("exchange")
                ])
                .sort("date")
            )

            with self._lock:
                self._df = snapshot
                self._refresh_cache_locked()
                self._save_locked()
            logger.info(f"✓ 交易日历同步完成，覆盖至: {self._df['date'].max()}")

        except Exception as e:
            logger.exception(f"同步交易日历异常: {e}")

    def is_trade_day(self, date_val: Union[datetime.date, str]) -> bool:
        """判断日期是否为交易日。"""
        if isinstance(date_val, str):
            date_val = datetime.date.fromisoformat(date_val.replace('/', '-'))
        if self._trade_days_cache is None:
            return False
        # search_sorted 是 O(log N) 操作
        idx = self._trade_days_cache.search_sorted(date_val)
        if idx >= self._trade_days_cache.len():
            return False
        return self._trade_days_cache[idx] == date_val

    def offset(self, date_val: datetime.date, n: int) -> Optional[datetime.date]:
        """
        核心功能：基于交易日的快速偏移计算。
        例如：offset(T, -1) 即寻找 T 日前的一个交易日。
        """
        if self._trade_days_cache is None or self._trade_days_cache.len() == 0:
            return None

        with self._lock:
            # search_sorted 默认返回第一个 >= date_val 的索引
            idx = self._trade_days_cache.search_sorted(date_val)

            if n >= 0:
                target_idx = idx + n
            else:
                # 向前偏移：如果当前是非交易日，search_sorted 指向其后的第一个
                # 需要减 1 回到其前的第一个交易日作为基准
                if idx >= self._trade_days_cache.len() or self._trade_days_cache[idx] != date_val:
                    idx -= 1
                target_idx = idx + n + 1

            if 0 <= target_idx < self._trade_days_cache.len():
                return self._trade_days_cache[target_idx]
            return None

    def get_trade_days(self, start: datetime.date, end: datetime.date) -> List[datetime.date]:
        """
        获取时间段内的所有有序交易日。
        返回 List[datetime.date] 以完美适配 TushareDataService 的同步循环。
        """
        if self._trade_days_cache is None:
            return []

        filtered = self._trade_days_cache.filter(
            (self._trade_days_cache >= start) & (self._trade_days_cache <= end)
        )
        return filtered.to_list()

    def _save_locked(self):
        """持久化数据。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(self.path, compression="snappy")