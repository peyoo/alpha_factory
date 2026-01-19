"""
数据读取接口 (Data Reader)。

本模块为下游模块（因子计算、回测、机器学习）提供统一的数据访问接口。

核心功能：
- 跨多个 Parquet 文件的智能关联
- 时间范围过滤
- 列选择与缓存
- Lazy 执行保持优化链完整
"""

from datetime import datetime, date
from typing import List, Optional
import polars as pl
from loguru import logger

from alpha.utils.config import settings


# ============================================================================
# 数据读取服务
# ============================================================================


class DataProvider:
    """
    本地数据仓库读取服务。

    负责从 Parquet 数据库中读取数据，自动进行表关联、时间过滤和列选择。
    所有返回值均为 LazyFrame，支持 Polars 的查询优化。
    """

    def __init__(self):
        """初始化数据提供者。"""
        logger.info("✓ DataProvider 初始化完成")

    # ========================================================================
    # 核心接口：load_data
    # ========================================================================

    def load_data(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        asset_filter: Optional[List[str]] = None,
    ) -> pl.LazyFrame:
        """
        加载本地清洗后的数据 (Lazy Mode)。

        自动关联行情、因子、状态等所有数据表。

        Args:
            start_date: 起始日期 (YYYYMMDD 格式，闭区间)
            end_date: 结束日期 (YYYYMMDD 格式，闭区间)
            columns: 需要加载的列名列表。
                     如 ['CLOSE', 'pe', 'is_st']。
                     None 表示加载所有列。
                     如不显式包含 _DATE_ 和 _ASSET_，会自动添加。
            asset_filter: 股票代码过滤列表，如 ['000001.SZ', '000002.SZ']。
                         None 表示加载所有股票。

        Returns:
            pl.LazyFrame: 已过滤时间范围、选定列的 LazyFrame。

        Raises:
            ValueError: 如果日期格式不合法或数据不存在
        """
        logger.info(f"加载数据: {start_date}-{end_date}, columns={columns}")

        # Step 1: 验证日期格式
        try:
            start_dt = self._parse_date(start_date)
            end_dt = self._parse_date(end_date)
            if start_dt > end_dt:
                raise ValueError(f"起始日期 {start_date} 晚于结束日期 {end_date}")
        except ValueError as e:
            logger.error(f"日期解析失败: {e}")
            raise

        # Step 2: 确保必需列存在
        if columns is not None:
            if "_DATE_" not in columns:
                columns = ["_DATE_"] + columns
            if "_ASSET_" not in columns:
                columns = ["_ASSET_"] + columns
            # 去重
            columns = list(dict.fromkeys(columns))
        else:
            columns = None

        # Step 3: 加载日线行情数据（核心表）
        lf_daily = self._load_daily_bars(start_date, end_date, columns, asset_filter)
        if lf_daily is None:
            logger.warning("未找到日线行情数据")
            return pl.LazyFrame({})

        # Step 4: 关联其他辅助表（如果需要）
        lf = lf_daily

        # 检查是否需要加载其他表
        requested_cols = set(columns) if columns else set()

        # 关联每日基础指标
        if self._should_load_table("daily_basic", requested_cols):
            lf_basic = self._load_daily_basic(start_date, end_date, asset_filter)
            if lf_basic is not None:
                lf = lf.join(
                    lf_basic,
                    on=["_DATE_", "_ASSET_"],
                    how="left",
                )
                logger.debug("✓ 关联每日基础指标")

        # 关联市场状态
        if self._should_load_table("market_status", requested_cols):
            lf_status = self._load_market_status(start_date, end_date, asset_filter)
            if lf_status is not None:
                lf = lf.join(
                    lf_status,
                    on=["_DATE_", "_ASSET_"],
                    how="left",
                )
                logger.debug("✓ 关联市场状态")

        # Step 5: 最终列选择
        if columns:
            # 只保留请求的列（且存在于结果中）
            available_cols = set(lf.collect_schema().names())
            final_cols = [c for c in columns if c in available_cols]
            lf = lf.select(final_cols)
            logger.debug(f"最终列数: {len(final_cols)}")

        logger.info(f"✓ 数据加载完成，预期行数: {self._estimate_row_count(start_date, end_date)}")
        return lf

    # ========================================================================
    # 私有方法：表加载
    # ========================================================================

    def _load_daily_bars(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        asset_filter: Optional[List[str]] = None,
    ) -> Optional[pl.LazyFrame]:
        """
        加载日线行情数据。

        按年份分区读取，自动合并多个年份文件。

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            columns: 列过滤
            asset_filter: 股票过滤

        Returns:
            pl.LazyFrame 或 None
        """
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)
        start_year = start_dt.year
        end_year = end_dt.year

        daily_dir = settings.WAREHOUSE_DIR / "daily"
        if not daily_dir.exists():
            logger.warning(f"日线行情目录不存在: {daily_dir}")
            return None

        dfs = []
        for year in range(start_year, end_year + 1):
            year_file = daily_dir / f"{year}.parquet"
            if year_file.exists():
                try:
                    df = pl.scan_parquet(year_file)
                    dfs.append(df)
                    logger.debug(f"✓ 加载 {year} 年日线数据: {year_file}")
                except Exception as e:
                    logger.error(f"✗ 加载 {year_file} 失败: {e}")
            else:
                logger.debug(f"- {year} 年数据文件不存在")

        if not dfs:
            return None

        # 合并多年数据
        lf = pl.concat(dfs, how="vertical")

        # 时间过滤
        lf = lf.filter(
            (pl.col("_DATE_") >= pl.lit(start_dt))
            & (pl.col("_DATE_") <= pl.lit(end_dt))
        )

        # 股票过滤
        if asset_filter:
            lf = lf.filter(pl.col("_ASSET_").is_in(asset_filter))

        # 列选择
        if columns:
            available_cols = set(lf.collect_schema().names())
            final_cols = [c for c in columns if c in available_cols]
            lf = lf.select(final_cols)

        logger.debug(f"✓ 日线行情已加载，预期行数: {self._estimate_row_count(start_date, end_date)}")
        return lf

    def _load_daily_basic(
        self,
        start_date: str,
        end_date: str,
        asset_filter: Optional[List[str]] = None,
    ) -> Optional[pl.LazyFrame]:
        """
        加载每日基础指标数据。

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            asset_filter: 股票过滤

        Returns:
            pl.LazyFrame 或 None
        """
        basic_path = settings.WAREHOUSE_DIR / "daily_basic.parquet"
        if not basic_path.exists():
            logger.debug(f"每日基础指标文件不存在: {basic_path}")
            return None

        try:
            start_dt = self._parse_date(start_date)
            end_dt = self._parse_date(end_date)

            lf = pl.scan_parquet(basic_path)

            # 时间过滤
            lf = lf.filter(
                (pl.col("_DATE_") >= pl.lit(start_dt))
                & (pl.col("_DATE_") <= pl.lit(end_dt))
            )

            # 股票过滤
            if asset_filter:
                lf = lf.filter(pl.col("_ASSET_").is_in(asset_filter))

            logger.debug("✓ 每日基础指标已加载")
            return lf
        except Exception as e:
            logger.error(f"✗ 加载每日基础指标失败: {e}")
            return None

    def _load_market_status(
        self,
        start_date: str,
        end_date: str,
        asset_filter: Optional[List[str]] = None,
    ) -> Optional[pl.LazyFrame]:
        """
        加载市场状态数据（涨跌停、ST、停牌）。

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            asset_filter: 股票过滤

        Returns:
            pl.LazyFrame 或 None
        """
        status_path = settings.WAREHOUSE_DIR / "daily_status.parquet"
        if not status_path.exists():
            logger.debug(f"市场状态文件不存在: {status_path}")
            return None

        try:
            start_dt = self._parse_date(start_date)
            end_dt = self._parse_date(end_date)

            lf = pl.scan_parquet(status_path)

            # 时间过滤
            lf = lf.filter(
                (pl.col("_DATE_") >= pl.lit(start_dt))
                & (pl.col("_DATE_") <= pl.lit(end_dt))
            )

            # 股票过滤
            if asset_filter:
                lf = lf.filter(pl.col("_ASSET_").is_in(asset_filter))

            logger.debug("✓ 市场状态已加载")
            return lf
        except Exception as e:
            logger.error(f"✗ 加载市场状态失败: {e}")
            return None

    # ========================================================================
    # 工具方法
    # ========================================================================

    @staticmethod
    def _parse_date(date_str: str) -> date:
        """
        解析 YYYYMMDD 格式的日期字符串。

        Args:
            date_str: 日期字符串，如 "20230101"

        Returns:
            datetime.date 对象

        Raises:
            ValueError: 如果格式不合法
        """
        try:
            return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError as e:
            raise ValueError(f"日期格式错误: {date_str} (期望 YYYYMMDD)") from e

    @staticmethod
    def _should_load_table(table_name: str, requested_cols: set) -> bool:
        """
        判断是否需要加载某个表。

        基于请求的列名决定。

        Args:
            table_name: 表名，如 "daily_basic"
            requested_cols: 请求的列集合

        Returns:
            bool: 是否需要加载
        """
        # 如果没有特定列请求，则加载所有表
        if not requested_cols:
            return True

        # 定义每个表的特有列
        table_columns = {
            "daily_basic": {
                "turnover_rate", "turnover_rate_f", "volume_ratio",
                "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                "dv_ratio", "dv_ttm",
                "total_share", "float_share", "free_share",
                "total_mv", "circ_mv",
            },
            "market_status": {
                "up_limit", "down_limit", "is_st", "is_suspended",
            },
        }

        # 检查是否有表的特有列被请求
        if table_name in table_columns:
            return bool(requested_cols & table_columns[table_name])

        return False

    @staticmethod
    def _estimate_row_count(start_date: str, end_date: str) -> int:
        """
        估计行数（粗略）。

        基于交易天数 * 平均股票数。

        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD

        Returns:
            int: 估计行数
        """
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        trading_days = (end_dt - start_dt).days * 0.7  # 交易日率假设为 70%
        avg_stocks = 5000  # 平均股票数
        return int(trading_days * avg_stocks)


# ============================================================================
# 便利函数
# ============================================================================


def get_data(
    start_date: str,
    end_date: str,
    columns: Optional[List[str]] = None,
    asset_filter: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """
    全局便利函数，快速加载数据。

    Args:
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        columns: 列过滤
        asset_filter: 股票过滤

    Returns:
        pl.LazyFrame
    """
    provider = DataProvider()
    return provider.load_data(start_date, end_date, columns, asset_filter)
