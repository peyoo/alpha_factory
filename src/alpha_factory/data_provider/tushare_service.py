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

        # 5. 同步完成后触发 L2 构建
        logger.info("⚙️ 启动年度 Parquet 因子库构建...")
        self.factor_builder.build_unified_factors(start_dt, end_dt)

    def _disclosure(
        self, trade_date: str, fields: Optional[list] = None
    ) -> pd.DataFrame:
        """
        报告期披露计划因子

        【信号定义】当前交易日是否满足以下三个条件（全部满足则 flag = True）：
        1. 预计披露日期(pre_date) 与当前交易日(trade_date) 的差异 <= 3 天
        2. 预计披露日期位于4月下旬（4月21-30日）
        3. 当前交易日 < 实际披露日期(actual_date)

        参数:
            trade_date: YYYYMMDD 格式的交易日期字符串
            fields: 忽略（兼容 API 调用签名）

        返回：DataFrame with columns {ts_code, flag}
        """
        trade_date_obj = datetime.strptime(trade_date, "%Y%m%d").date()

        if trade_date_obj.month == 4 and trade_date_obj.day <= 15:
            return pd.DataFrame(columns=["ts_code", "flag"])

        report_year = trade_date_obj.year - 1  # 报告年度通常是前一年
        report_end = date(report_year, 12, 31)

        # 1. 直接从 API 获取该年度的披露计划（不使用缓存）
        try:
            self.rate_limiter.wait()
            df_disclosure = self.pro.disclosure_date(
                end_date=report_end.strftime("%Y%m%d"),
                fields=["ts_code", "pre_date", "actual_date"],
            )

            if df_disclosure is None or df_disclosure.empty:
                # 返回空的结果集
                return pd.DataFrame(columns=["ts_code", "flag"])
        except Exception as e:
            logger.warning(f"⚠️ 无法获取披露计划数据 ({report_end}): {e}")
            return pd.DataFrame(columns=["ts_code", "flag"])

        # 2. 数据预处理：转换日期列为 datetime
        try:
            if "pre_date" in df_disclosure.columns:
                df_disclosure["pre_date"] = pd.to_datetime(df_disclosure["pre_date"])
            if "actual_date" in df_disclosure.columns:
                df_disclosure["actual_date"] = pd.to_datetime(
                    df_disclosure["actual_date"]
                )
        except Exception as e:
            logger.warning(f"⚠️ 披露日期列转换异常: {e}")
            return pd.DataFrame(columns=["ts_code", "flag"])

        # 3. 应用三个条件筛选
        result_data = []

        for _, row in df_disclosure.iterrows():
            ts_code = row["ts_code"]
            pre_date = row.get("pre_date")
            actual_date = row.get("actual_date")

            flag = False

            # 仅当必要字段都存在且不为 NaT 时才进行判断
            if pd.notna(pre_date) and pd.notna(actual_date):
                pre_date_obj = (
                    pre_date.date() if hasattr(pre_date, "date") else pre_date
                )
                actual_date_obj = (
                    actual_date.date() if hasattr(actual_date, "date") else actual_date
                )

                # 【条件1】计划披露日期与当前交易日的差异 <= 4 天
                # 这里至少需要4天，周末两天，再加上买卖各一天。
                date_diff = abs((pre_date_obj - trade_date_obj).days)
                cond1 = date_diff <= 4

                # 【条件2】计划披露日期位于4月下旬（4月21-30日）
                cond2 = pre_date_obj.month == 4 and 21 <= pre_date_obj.day <= 30

                # 【条件3】当前交易日 < 实际发布日期
                cond3 = trade_date_obj < actual_date_obj

                # 全部条件都满足则置为 True
                flag = cond1 and cond2 and cond3

            result_data.append({"ts_code": ts_code, "flag": flag})

        # 4. 构造返回 DataFrame
        result_df = pd.DataFrame(result_data)
        if not result_df.empty:
            # 类型转换（兼容 HDF5 Fixed 模式）
            result_df["ts_code"] = result_df["ts_code"].astype(str).str.slice(0, 12)
            result_df["flag"] = result_df["flag"]
        else:
            # 保证列类型一致
            result_df["ts_code"] = result_df["ts_code"]
            result_df["flag"] = result_df["flag"]

        return result_df

    def _st_data(self, trade_date: str, fields: Optional[list] = None) -> pd.DataFrame:
        """
        融合 namechange 和 stock_st 生成可靠的 ST 标记

        【核心策略】
        - 查询 [prev_trade_date, trade_date] 内的 namechange 记录
        - 名称規則：包含"st"(忽视大小写) | 以"退" | 以"退市" → is_st=True
        - 融合优先级：namechange 规则结果 > stock_st 的 is_st 字段
        - 处理缓存和日志由 _sync_single_day_bundle 统一管理

        参数:
            trade_date: YYYYMMDD 格式的交易日期字符串
            fields: 忽略（兼容 API 调用签名）

        返回：DataFrame with columns {ts_code, is_st}
        """
        # 1. 转换日期格式并获取前一个交易日
        trade_date_obj = datetime.strptime(trade_date, "%Y%m%d").date()
        prev_trade_date = self.calendar.offset(trade_date_obj, -1)
        prev_date_str = prev_trade_date.strftime("%Y%m%d")

        # 2. 查询 namechange [prev_date_str, trade_date] 的名称变更记录
        df_namechange = None
        try:
            self.rate_limiter.wait()
            df_namechange = self.pro.namechange(
                start_date=prev_date_str,
                end_date=trade_date,
                fields=["ts_code", "name", "change_reason"],
            )
        except Exception as e:
            logger.warning(f"⚠️ namechange 查询异常 ({prev_date_str}~{trade_date}): {e}")

        # 3. 查询 stock_st (trade_date) 的官方 ST 股票列表
        # stock_st 返回的列表本身就代表 ST 股票，无需 is_st 字段
        df_stock_st = None
        try:
            self.rate_limiter.wait()
            df_stock_st = self.pro.stock_st(trade_date=trade_date, fields=["ts_code"])
        except Exception as e:
            logger.warning(f"⚠️ stock_st 查询异常 ({trade_date}): {e}")

        # 4. 融合两个数据源
        return self._merge_st_sources(df_namechange, df_stock_st)

    def _extract_st_from_names(self, df: pd.DataFrame) -> dict:
        """
        从名称字段和变更原因提取 ST 标记

        规则：
        1. 名称包含"st"(任意大小写) | 以"退" | 以"退市" → True
        2. change_reason == "终止上市" → True (退市标记)

        返回：{ts_code: is_st} 字典
        """
        if df is None or df.empty or "name" not in df.columns:
            return {}

        result = {}
        for _, row in df.iterrows():
            ts_code = row["ts_code"]
            name = str(row.get("name", "")).strip()
            change_reason = str(row.get("change_reason", "")).strip()

            # 名称规则检查
            is_st_from_name = (
                "st" in name.lower()
                or name.endswith("退")
                or name.endswith("退市")
                or name.startswith("退市")
            )

            # 退市原因检查
            is_delisted = change_reason == "终止上市"

            is_st = is_st_from_name or is_delisted
            result[ts_code] = is_st

        return result

    def _merge_st_sources(
        self, df_namechange: Optional[pd.DataFrame], df_stock_st: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        融合 namechange 和 stock_st 的 ST 标记

        融合策略：取并集
        - 如果 namechange 中 ts_code 满足 ST 规则 → is_st=True
        - 或者 ts_code 出现在 stock_st 列表中 → is_st=True
        - 否则 → is_st=False

        注意：stock_st API 返回的列表本身就代表 ST 股票，无需 is_st 字段

        返回：DataFrame with columns {ts_code, is_st}
        """
        # 提取两个数据源中识别为 ST 的 ts_code
        st_codes_from_namechange = set()
        st_dict = self._extract_st_from_names(df_namechange)
        st_codes_from_namechange = {code for code, is_st in st_dict.items() if is_st}

        # stock_st 返回的所有 ts_code 都代表 ST 股票
        st_codes_from_stock_st = set()
        if df_stock_st is not None and not df_stock_st.empty:
            st_codes_from_stock_st = set(df_stock_st["ts_code"].values)

        # 并集：两个数据源中满足 ST 条件的所有 ts_code
        all_st_codes = st_codes_from_namechange | st_codes_from_stock_st

        # 获取所有涉及的 ts_code（包括两个数据源中的所有股票）
        all_codes = st_dict.keys() | st_codes_from_stock_st

        result_data = []
        for ts_code in sorted(all_codes):
            # 并集策略：任一来源认为是 ST 就标记为 True
            is_st = ts_code in all_st_codes

            result_data.append({"ts_code": ts_code, "is_st": is_st})

        result_df = pd.DataFrame(result_data)

        # 类型转换（兼容 HDF5 Fixed 模式）
        if not result_df.empty:
            result_df["ts_code"] = result_df["ts_code"].str.slice(0, 12).astype("S12")
            result_df["is_st"] = result_df["is_st"].astype(bool)

        return result_df

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
            (
                "st",
                self._st_data,
                {"ts_code": "string", "is_st": "boolean"},
            ),
            (
                "disclosure",
                self._disclosure,
                {"ts_code": "string", "flag": "boolean"},
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

                # 💡 2. 强转类型：仅为兼容 Fixed 模式和内存优化
                for col, dtype in fields_schema.items():
                    if col in df.columns:
                        if dtype == "string":
                            df[col] = df[col].fillna("").astype(str)
                            if col == "ts_code":
                                df[col] = df[col].str.slice(0, 12).astype("S12")
                        elif dtype == "boolean":
                            df[col] = df[col].astype(bool)
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
