"""
Tushare 数据同步服务 (L0-L1 接入层)

【核心策略】按日期全市场批量获取
- 每个交易日调用一次 API: daily(trade_date=date)
- 返回该日全市场数据 (通常 4500-5000 行)
- 禁止按股票循环 (ts_code 参数)
- 无数据量超限风险
"""

import os
import time
from loguru import logger
from datetime import datetime, date
from typing import Optional
import pandas as pd
import polars as pl

from alpha_factory.config.base import settings
from alpha_factory.data_provider.cache_manager import HDF5CacheManager
from alpha_factory.data_provider.unified_factor_builder import UnifiedFactorBuilder
from alpha_factory.data_provider.trade_calendar_manager import TradeCalendarManager
from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager
from alpha_factory.utils.schema import F


class DataSyncError(RuntimeError):
    """致命性同步错误：当关键分片缺失或写入失败时抛出。"""

    pass


class RateLimiter:
    """API 限流控制器（基于 Tushare 官方限流策略）"""

    def __init__(self, is_vip: bool = True):
        self.is_vip = is_vip
        # VIP: 800次/分 ≈ 75ms; 普通: 200次/分 ≈ 300ms
        self.min_interval = 0.075 if is_vip else 0.3
        self.last_request_time = 0

    def wait(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class TushareDataService:
    """
    Tushare 数据同步服务 (L0-L1)

    【核心架构】
    - 接入层：Python datetime.date 对象 (通用性、可读性)
    - 缓存层：HDF5 (保护 API 积分，写前必查 is_cached)
    - 计算层：Polars (极致性能，Date 映射为 Int32)
    """

    def __init__(self):
        # 1. Token 获取逻辑
        self.token = getattr(settings, "TUSHARE_TOKEN", None) or os.getenv(
            "TUSHARE_TOKEN"
        )
        if not self.token:
            raise ValueError("❌ TUSHARE_TOKEN 未设置，请在 settings 或环境变量中配置")

        # NOTE: 使用 Settings 中声明的 IS_VIP 字段（全大写）
        is_vip = settings.IS_VIP
        self.rate_limiter = RateLimiter(is_vip)
        self.pro = self._init_tushare()

        # 2. 初始化核心管理器
        self.cache_manager = HDF5CacheManager(settings.RAW_DATA_DIR)
        self.calendar = TradeCalendarManager()
        self.assets_mgr = StockAssetsManager()

        # 3. 初始化因子构建器 (修正点：匹配最新的 __init__ 签名)
        # UnifiedFactorBuilder 期望位置参数：assets_mgr, calendar_mgr
        self.factor_builder = UnifiedFactorBuilder(self.assets_mgr, self.calendar)

        logger.info(f"✓ TushareService 初始化完成 (VIP={is_vip})")

    def _init_tushare(self):
        import tushare as ts

        return ts.pro_api(self.token)

    # ---------------------------------------------------------------------
    # 核心同步流程
    # ---------------------------------------------------------------------

    def sync_data(self, start_date: str, end_date: Optional[str] = None) -> None:
        """
        全量同步主入口：按天打包同步所有分片（已适配长连接优化）
        """
        # 1. 前置元数据同步
        try:
            self.calendar.sync_from_tushare()
            self.assets_mgr.sync_from_tushare()
        except Exception as e:
            logger.warning(f"元数据同步告警: {e}")

        # 2. 确定 end_date：如果为 None，智能查找最新可用数据
        if end_date is None:
            end_date = self._find_latest_available_date()
            logger.info(
                f"⏰ end_date 自动设置为: {end_date} (daily_basic 最新可用数据)"
            )

        # 3. 获取交易日列表
        start_dt = datetime.strptime(start_date, "%Y%m%d").date()
        end_dt = datetime.strptime(end_date, "%Y%m%d").date()
        trade_days = self.calendar.get_trade_days(start_dt, end_dt)

        if not trade_days:
            logger.warning(f"⚠️ {start_date} ~ {end_date} 之间无交易日")
            return

        total = len(trade_days)
        logger.info(f"🚀 开始同步任务，共计 {total} 个交易日...")

        # 4. 【核心修改】使用 try...finally 维护 HDF5 长连接
        try:
            for i, current_date in enumerate(trade_days, 1):
                # 此时内部调用的 is_cached 和 save_to_hdf5 会自动复用已打开的句柄
                self._sync_single_day_bundle(current_date, i, total)

            logger.success("✨ 所有数据分片同步已完成并刷入磁盘")

        except Exception as e:
            logger.error(f"❌ 同步过程中发生致命错误: {e}")
            raise  # 向上抛出以防后续因子构建在错误基础上运行

        finally:
            # 💡 无论任务成功还是报错中断，必须显式释放文件句柄
            self.cache_manager.close_all()

        # 4. 同步完成后触发 L2 构建
        logger.info("⚙️ 启动年度 Parquet 因子库构建...")
        self.factor_builder.build_unified_factors(start_dt, end_dt)

    def _sync_single_day_bundle(self, trade_date: date, idx: int, total: int) -> None:
        date_str = trade_date.strftime("%Y%m%d")

        # 1. 定义标准任务表 (数据源, API函数, 预期的 Schema)
        # 统一使用 dict 存储列名和 Dtype，既能用于 fields 参数，也能用于 astype
        tasks = [
            (
                "daily",
                self.pro.daily,
                {
                    "ts_code": "string",
                    "open": "float32",
                    "high": "float32",
                    "low": "float32",
                    "close": "float32",
                    "vol": "float32",
                    "amount": "float64",
                },
            ),
            (
                "adj_factor",
                self.pro.adj_factor,
                {"ts_code": "string", "adj_factor": "float32"},
            ),
            (
                "daily_basic",
                self.pro.daily_basic,
                {
                    "ts_code": "string",
                    "turnover_rate": "float32",
                    "pe": "float32",
                    "pb": "float32",
                    "ps": "float32",
                    "total_mv": "float64",
                    "circ_mv": "float64",
                },
            ),
            (
                "stk_limit",
                self.pro.stk_limit,
                {"ts_code": "string", "up_limit": "float32", "down_limit": "float32"},
            ),
            (
                "suspend_d",
                self.pro.suspend_d,
                {"ts_code": "string", "suspend_type": "string"},
            ),
            ("st", self.pro.stock_st, {"ts_code": "string", "is_st": "string"}),
            (
                "daily_names",
                self.pro.bak_daily,
                {"ts_code": "string", "name": "string"},
            ),
        ]

        for source, api_func, fields_schema in tasks:
            if self.cache_manager.is_cached(source, trade_date):
                continue

            try:
                self.rate_limiter.wait()

                # 💡 1. 精准获取：只拿 fields_schema 中定义的业务字段
                fetch_fields = list(fields_schema.keys())
                df = api_func(trade_date=date_str, fields=fetch_fields)

                if df is None or df.empty:
                    continue

                if source == "daily_names" and "name" in df.columns:
                    name_clean = df["name"].fillna("").astype(str).str.strip()
                    name_lower = name_clean.str.lower()
                    is_st = (
                        name_lower.str.contains("st", regex=False)
                        | name_clean.str.endswith("退")
                        | name_clean.str.endswith("退市")
                    )
                    df = pd.DataFrame(
                        {
                            "ts_code": df["ts_code"],
                            "is_st": is_st.fillna(False).astype(bool),
                        }
                    )

                # 💡 2. 强转类型：仅为兼容 Fixed 模式和内存优化
                # 此时 df 已经没有冗余日期列了
                for col, dtype in fields_schema.items():
                    if col in df.columns:
                        if dtype == "string":
                            df[col] = df[col].fillna("").astype(str)
                            if col == "ts_code":
                                df[col] = df[col].str.slice(0, 12).astype("S12")
                        else:
                            df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                                dtype
                            )

                # 💡 3. 直接落盘
                self.cache_manager.save_to_hdf5(source, trade_date, df)
                logger.info(f"[{idx}/{total}] ✓ 已持久化: {source} ({date_str})")

            except Exception as e:
                logger.error(f"❌ {date_str} {source} 异常: {e}")
                raise DataSyncError(f"API 中断: {source}")

    def _find_latest_available_date(self, lookback_days: int = 10) -> str:
        """
        智能查找 Tushare 上最新可用数据的交易日

        【使用 daily_basic 接口判断数据可用性】
        daily_basic 包含 PE、PB、PS 等估值数据，数据完整性更好。
        Tushare 网站上的数据通常有 1-2 个交易日的延迟。

        参数:
            lookback_days: 最多往前查找多少个交易日 (默认 10)

        返回:
            有数据的最新交易日 'YYYYMMDD' 格式
        """
        today = date.today()

        # 获取过去的交易日列表
        lookback_start = today - pd.Timedelta(days=lookback_days * 2)
        trade_days_back = self.calendar.get_trade_days(lookback_start, today)

        if not trade_days_back:
            logger.error(f"⚠️ 无法获取交易日历，返回今天: {today.strftime('%Y%m%d')}")
            return today.strftime("%Y%m%d")

        # 反向遍历（从最近往前），最多查找 lookback_days 个
        checked_count = 0
        for check_date in reversed(trade_days_back):
            if checked_count >= lookback_days:
                break

            date_str = check_date.strftime("%Y%m%d")
            checked_count += 1

            try:
                # 使用 daily_basic 接口，只获取 1 条记录检查数据可用性
                self.rate_limiter.wait()
                df = self.pro.bak_basic(trade_date=date_str, limit=1)

                # 如果返回不为空，说明该日有数据
                if df is not None and not df.empty:
                    logger.info(
                        f"✓ 找到最新可用数据 (daily_basic): {date_str} (检查了 {checked_count} 个交易日)"
                    )
                    return date_str
                else:
                    logger.debug(f"⏭️  {date_str} 无数据，继续查找")

            except Exception as e:
                logger.debug(f"❌ 检查 {date_str} 时异常: {e}，继续查找")
                continue

        # 如果找不到任何有数据的日期，返回今天
        logger.warning(
            f"⚠️ 向前查找 {lookback_days} 个交易日都无数据，使用今天作为 end_date: {today.strftime('%Y%m%d')}"
        )
        return today.strftime("%Y%m%d")

    # ---------------------------------------------------------------------
    # 增量更新逻辑
    # ---------------------------------------------------------------------

    def daily_update(self) -> None:
        """日频自动增量同步"""
        # 从 L2 仓库探测最新日期
        last_date_str = self.get_latest_date_from_warehouse()
        if not last_date_str:
            logger.error("无法获取仓库日期，请先进行全量同步")
            return

        last_date = datetime.strptime(last_date_str, "%Y%m%d").date()
        next_date = self.calendar.offset(last_date, 1)

        if next_date > date.today():
            logger.info("✅ 数据已是最新")
            return

        self.sync_data(next_date.strftime("%Y%m%d"), end_date=None)

    def get_latest_date_from_warehouse(self) -> Optional[str]:
        """利用 Polars 快速探测 Parquet 仓库的最大日期"""
        path = self.factor_builder.warehouse_dir / "unified_factors/*.parquet"
        try:
            # 极致性能：只扫描不加载，获取最大值
            # 注意：统一因子库的日期列是 DATE（而非 trade_date）
            max_date = (
                pl.scan_parquet(str(path)).select(pl.col(F.DATE).max()).collect().item()
            )
            # 兼容多种返回类型：None / datetime.date / datetime.datetime / str / pandas.Timestamp / numpy datetime
            if max_date is None:
                return None

            # 优先处理 Python 原生 date
            if isinstance(max_date, date):
                return max_date.strftime("%Y%m%d")

            # 处理 datetime.datetime
            if isinstance(max_date, datetime):
                return max_date.strftime("%Y%m%d")

            # 处理字符串（可能为 'YYYY-MM-DD' 或 'YYYYMMDD'）
            if isinstance(max_date, str):
                try:
                    if "-" in max_date:
                        dt = datetime.fromisoformat(max_date)
                    else:
                        dt = datetime.strptime(max_date, "%Y%m%d")
                    return dt.strftime("%Y%m%d")
                except Exception:
                    # 如果是长度为8的纯数字字符串，认为已是 YYYYMMDD
                    if len(max_date) == 8 and max_date.isdigit():
                        return max_date
                    return None

            # 兜底：尝试用 pandas.Timestamp 解析（适配 numpy.datetime64 等）
            try:
                import pandas as _pd

                ts = _pd.Timestamp(max_date)
                return ts.strftime("%Y%m%d")
            except Exception:
                return None

        except Exception as e:
            logger.error(f"获取仓库最大日期失败: {e}")
        return None
