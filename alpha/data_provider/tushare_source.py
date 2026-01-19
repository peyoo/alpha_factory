"""
Tushare 数据源模块 (Tushare Source)。

本模块负责与 Tushare API 的交互，包括：
- 限流管理
- API 调用与重试
- 数据下载与整理
- 增量更新支持
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
import tushare as ts
from loguru import logger

from alpha.data_provider.cleaner import (
    clean_daily_bars,
    clean_calendar,
    clean_adj_factors,
    clean_daily_basic,
    clean_market_status,
    clean_stock_basic,
)
from alpha.utils.config import settings


# ============================================================================
# 限流器 (Rate Limiter)
# ============================================================================


class RateLimiter:
    """
    限流器，用于控制 API 请求频率。

    支持两种模式：
    - VIP 模式：极低延迟 (仅做错误重试)
    - 普通模式：严格限流 (每分钟几百次)
    """

    def __init__(self, is_vip: bool = True):
        """
        初始化限流器。

        Args:
            is_vip: 是否为 VIP 账户
        """
        self.is_vip = is_vip
        self.last_request_time = 0

        if is_vip:
            # VIP 模式：最小延迟 (Tushare VIP 基本无限流)
            self.min_interval = 0.01  # 10ms
        else:
            # 普通模式：严格限流 (200 请求/分钟 = 300ms/请求)
            self.min_interval = 0.3

    def wait(self):
        """等待直到满足最小请求间隔。"""
        elapsed = time.time() - self.last_request_time
        wait_time = self.min_interval - elapsed

        if wait_time > 0:
            time.sleep(wait_time)

        self.last_request_time = time.time()


# ============================================================================
# Tushare 数据服务
# ============================================================================


class TushareDataService:
    """
    Tushare 数据接入服务。

    负责：
    1. 初始化 Tushare 连接
    2. 管理 API 调用与限流
    3. 数据下载与清洗
    4. 数据持久化 (Parquet)
    """

    def __init__(self, token: Optional[str] = None, is_vip: bool = True):
        """
        初始化服务。

        Args:
            token: Tushare API Token (默认从环境变量读取)
            is_vip: 是否为 VIP 账户 (默认 True)
        """
        if token is None:
            token = settings.TUSHARE_TOKEN

        self.pro = ts.pro_api(token)
        self.is_vip = is_vip
        self.rate_limiter = RateLimiter(is_vip=is_vip)

        logger.info(f"✓ Tushare 服务初始化完成 (VIP模式: {is_vip})")

    # ========================================================================
    # 基础数据同步
    # ========================================================================

    def sync_basic_info(self) -> pl.LazyFrame:
        """
        同步全市场股票基础列表。

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        logger.info("同步股票基础信息...")
        self.rate_limiter.wait()

        try:
            df_pandas = self.pro.stock_basic(exchange="", list_status="L")
            logger.info(f"✓ 获取股票基础信息: {len(df_pandas)} 行")

            # 清洗与存储
            lf = clean_stock_basic(df_pandas)
            df_collected = lf.collect()

            # 存储到 Parquet
            stock_basic_path = settings.WAREHOUSE_DIR / "stock_basic.parquet"
            df_collected.write_parquet(stock_basic_path)
            logger.info(f"✓ 股票基础信息已存储: {stock_basic_path}")

            return lf
        except Exception as e:
            logger.error(f"✗ 同步股票基础信息失败: {e}")
            raise

    def sync_calendar(self, start_year: str = "2010", end_year: Optional[str] = None) -> pl.LazyFrame:
        """
        同步交易日历。

        Args:
            start_year: 起始年份 (YYYY 格式)
            end_year: 结束年份 (默认当前年)

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        if end_year is None:
            end_year = str(datetime.now().year)

        logger.info(f"同步交易日历 ({start_year}-{end_year})...")
        self.rate_limiter.wait()

        try:
            df_pandas = self.pro.trade_cal(
                exchange="",
                start_date=f"{start_year}0101",
                end_date=f"{end_year}1231",
            )
            logger.info(f"✓ 获取交易日历: {len(df_pandas)} 行")

            # 清洗与存储
            lf = clean_calendar(df_pandas)
            df_collected = lf.collect()

            # 存储到 Parquet
            calendar_path = settings.WAREHOUSE_DIR / "meta" / "calendar.parquet"
            calendar_path.parent.mkdir(parents=True, exist_ok=True)
            df_collected.write_parquet(calendar_path)
            logger.info(f"✓ 交易日历已存储: {calendar_path}")

            return lf
        except Exception as e:
            logger.error(f"✗ 同步交易日历失败: {e}")
            raise

    # ========================================================================
    # 行情数据同步
    # ========================================================================

    def sync_daily_bars(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 5,
    ) -> pl.LazyFrame:
        """
        同步日线行情。

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            max_workers: 并发线程数

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        logger.info(f"同步日线行情 ({start_date}-{end_date})...")

        # Step 1: 获取股票列表
        stock_basic_path = settings.WAREHOUSE_DIR / "stock_basic.parquet"
        if stock_basic_path.exists():
            df_stocks = pl.read_parquet(stock_basic_path)
            stock_codes = df_stocks["_ASSET_"].to_list()
            logger.info(f"从缓存加载 {len(stock_codes)} 只股票")
        else:
            logger.warning("股票列表不存在，先同步基础信息")
            self.sync_basic_info()
            df_stocks = pl.read_parquet(stock_basic_path)
            stock_codes = df_stocks["_ASSET_"].to_list()

        # Step 2: 并发下载日线数据
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_daily_bars_single, ts_code, start_date, end_date): ts_code
                for ts_code in stock_codes
            }

            for future in as_completed(futures):
                ts_code = futures[future]
                try:
                    df_daily = future.result()
                    if df_daily is not None:
                        results.append(df_daily)
                except Exception as e:
                    logger.error(f"✗ 下载 {ts_code} 失败: {e}")

        if not results:
            logger.warning("未获取任何日线数据")
            return pl.LazyFrame({})

        # Step 3: 合并并清洗
        df_combined = pl.concat(results, how="vertical")
        logger.info(f"✓ 合并 {len(results)} 只股票的日线数据: {df_combined.shape}")

        # 获取复权因子（用于计算后复权价格）
        adj_factor_path = settings.WAREHOUSE_DIR / "daily_adj.parquet"
        adj_factor_df = None
        if adj_factor_path.exists():
            adj_factor_df = pl.read_parquet(adj_factor_path).to_pandas()
            logger.info(f"加载复权因子: {adj_factor_df.shape}")

        lf = clean_daily_bars(df_combined.to_pandas(), with_adjustment=True, adj_factor_df=adj_factor_df)

        # Step 4: 按年份分区存储
        df_collected = lf.collect()
        df_collected = df_collected.with_columns(
            pl.col("_DATE_").dt.year().alias("year")
        )

        for year in df_collected["year"].unique().to_list():
            df_year = df_collected.filter(pl.col("year") == year).drop("year")
            year_path = settings.WAREHOUSE_DIR / "daily" / f"{year}.parquet"
            year_path.parent.mkdir(parents=True, exist_ok=True)
            df_year.write_parquet(year_path)
            logger.info(f"✓ {year} 年日线数据已存储: {year_path}")

        return lf

    def _download_daily_bars_single(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        下载单只股票的日线行情。

        Args:
            ts_code: 股票代码
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            Polars DataFrame 或 None
        """
        self.rate_limiter.wait()

        try:
            df_pandas = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )

            if df_pandas is None or len(df_pandas) == 0:
                logger.debug(f"- {ts_code}: 无数据")
                return None

            df = pl.from_pandas(df_pandas)
            logger.debug(f"✓ {ts_code}: {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"✗ 下载 {ts_code} 异常: {e}")
            return None

    # ========================================================================
    # 辅助数据同步
    # ========================================================================

    def sync_adj_factors(self, start_date: str, end_date: str) -> pl.LazyFrame:
        """
        同步复权因子。

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        logger.info(f"同步复权因子 ({start_date}-{end_date})...")
        self.rate_limiter.wait()

        try:
            df_pandas = self.pro.adj_factor(
                ts_code="",
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"✓ 获取复权因子: {len(df_pandas)} 行")

            lf = clean_adj_factors(df_pandas)
            df_collected = lf.collect()

            # 存储到 Parquet
            adj_path = settings.WAREHOUSE_DIR / "daily_adj.parquet"
            df_collected.write_parquet(adj_path)
            logger.info(f"✓ 复权因子已存储: {adj_path}")

            return lf
        except Exception as e:
            logger.error(f"✗ 同步复权因子失败: {e}")
            raise

    def sync_daily_basic(self, start_date: str, end_date: str) -> pl.LazyFrame:
        """
        同步每日基础指标。

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        logger.info(f"同步每日基础指标 ({start_date}-{end_date})...")
        self.rate_limiter.wait()

        try:
            df_pandas = self.pro.daily_basic(
                ts_code="",
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"✓ 获取每日基础指标: {len(df_pandas)} 行")

            lf = clean_daily_basic(df_pandas)
            df_collected = lf.collect()

            # 存储到 Parquet
            basic_path = settings.WAREHOUSE_DIR / "daily_basic.parquet"
            df_collected.write_parquet(basic_path)
            logger.info(f"✓ 每日基础指标已存储: {basic_path}")

            return lf
        except Exception as e:
            logger.error(f"✗ 同步每日基础指标失败: {e}")
            raise

    def sync_market_status(self, start_date: str, end_date: str) -> pl.LazyFrame:
        """
        同步市场状态数据（涨跌停、ST、停牌）。

        Args:
            start_date: 起始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            符合系统标准的 Polars LazyFrame
        """
        logger.info(f"同步市场状态 ({start_date}-{end_date})...")

        # 1. 涨跌停价
        logger.info("获取涨跌停价...")
        self.rate_limiter.wait()
        try:
            df_stk_limit = self.pro.stk_limit(
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"✓ 获取涨跌停价: {len(df_stk_limit)} 行")
        except Exception as e:
            logger.error(f"✗ 获取涨跌停价失败: {e}")
            df_stk_limit = None

        # 2. ST 状态 (按日期逐日获取)
        logger.info("获取 ST 状态...")
        df_st_list = []
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        while current_date <= end:
            date_str = current_date.strftime("%Y%m%d")
            self.rate_limiter.wait()

            try:
                df_st = self.pro.stock_st(trade_date=date_str)
                if df_st is not None and len(df_st) > 0:
                    df_st_list.append(df_st)
                    logger.debug(f"✓ {date_str}: {len(df_st)} 只 ST 股")
            except Exception as e:
                logger.debug(f"✗ {date_str} 获取 ST 状态失败: {e}")

            current_date += timedelta(days=1)

        df_stock_st = pl.concat([pl.from_pandas(df) for df in df_st_list]).to_pandas() if df_st_list else None
        logger.info(f"✓ ST 状态总计: {len(df_st_list)} 天")

        # 3. 停牌状态
        logger.info("获取停牌状态...")
        self.rate_limiter.wait()
        try:
            df_suspend = self.pro.suspend_d(
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"✓ 获取停牌状态: {len(df_suspend)} 行")
        except Exception as e:
            logger.error(f"✗ 获取停牌状态失败: {e}")
            df_suspend = None

        # 4. 合并
        lf = clean_market_status(
            stk_limit_df=df_stk_limit,
            stock_st_df=df_stock_st,
            suspend_d_df=df_suspend,
        )

        df_collected = lf.collect()
        market_status_path = settings.WAREHOUSE_DIR / "daily_status.parquet"
        df_collected.write_parquet(market_status_path)
        logger.info(f"✓ 市场状态已存储: {market_status_path}")

        return lf

    # ========================================================================
    # 增量更新
    # ========================================================================

    def daily_update(self) -> None:
        """
        每日增量更新。

        自动检测最新日期，更新所有相关表。
        """
        logger.info("开始每日增量更新...")

        # 扫描现有数据的最大日期
        max_date = self._get_max_date()
        if max_date is None:
            logger.warning("未找到现有数据，建议先运行历史数据同步")
            return

        start_date = (max_date + timedelta(days=1)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        logger.info(f"增量更新范围: {start_date} - {end_date}")

        # 同步各类数据
        try:
            self.sync_basic_info()
            self.sync_daily_bars(start_date, end_date)
            self.sync_adj_factors(start_date, end_date)
            self.sync_daily_basic(start_date, end_date)
            self.sync_market_status(start_date, end_date)
            logger.info("✓ 增量更新完成")
        except Exception as e:
            logger.error(f"✗ 增量更新失败: {e}")
            raise

    def _get_max_date(self) -> Optional[datetime]:
        """
        扫描现有数据，获取最大日期。

        Returns:
            最大日期或 None
        """
        daily_dir = settings.WAREHOUSE_DIR / "daily"
        if not daily_dir.exists():
            return None

        max_date = None
        for parquet_file in daily_dir.glob("*.parquet"):
            try:
                df = pl.read_parquet(parquet_file).select("_DATE_")
                file_max = df["_DATE_"].max()
                if file_max is not None:
                    file_max_date = file_max.item()
                    if max_date is None or file_max_date > max_date:
                        max_date = file_max_date
            except Exception as e:
                logger.debug(f"扫描 {parquet_file} 失败: {e}")

        return max_date
