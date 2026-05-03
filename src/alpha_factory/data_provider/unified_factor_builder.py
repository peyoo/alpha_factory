from __future__ import annotations

import datetime
from datetime import date
from typing import List, Optional
import polars as pl
from loguru import logger

from alpha_factory.config.base import settings
from alpha_factory.data_provider._data_cleaner import DataCleanerMixin
from alpha_factory.data_provider.cache_manager import HDF5CacheManager
from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager
from alpha_factory.data_provider.trade_calendar_manager import TradeCalendarManager
from alpha_factory.utils.schema import F


class UnifiedFactorBuilder(DataCleanerMixin):
    """
        统一因子库构建器 (L2/L3 ETL 引擎)

        职责:
        1. 骨架填充：生成 (Date x Asset) 矩阵，确保停牌及存续期数据连续。
        2. 单位对齐：金额(元), 成交量(股), 市值(元)。
        3. 类型锁定：强制使用 StockAssetsManager 的全局 Enum 确保跨表计算性能。
        4. 指标分类：集成坐标轴、原始价格、复权价格、量价指标、状态标记。

    类别,字段名,类型,单位,业务含义与逻辑
    坐标轴,DATE,Date,-,交易日期（已根据交易日历对齐）
    ,ASSET,Enum,-,股票唯一代码（类型锁定，跨表计算不丢索引）
    状态,IS_ST,Bool,-,是否风险警示：基于证券名称规则判定并前向传递。
    ,IS_SUSPENDED,Bool,-,是否全天停牌：(显式停牌接口 == True) OR (价格缺失)。
    复权价格,OPEN,F32,元,后复权开盘价：用于计算收益率（已处理停牌填充）。
    ,HIGH,F32,元,后复权最高价：用于计算波动率及技术指标。
    ,LOW,F32,元,后复权最低价：用于计算波动率及技术指标。
    ,CLOSE,F32,元,后复权收盘价：最核心的价格计算基准。
    原始行情,CLOSE_RAW,F32,元,交易所原始价格：用于判断是否触及涨跌停。
    ,UP_LIMIT,F32,元,当日涨停价：用于计算封板强度。
    ,DOWN_LIMIT,F32,元,当日跌停价：用于判断极端流动性风险。
    ,ADJ_FACTOR,F32,-,Tushare 原始复权因子。
    量价指标,VOLUME,F64,股,当日成交股数（已由"手"换算为"股"，停牌日为 0）。
    ,AMOUNT,F64,元,当日成交金额（已由"千元"换算为"元"，停牌日为 0）。
    ,TURNOVER_RATE,F32,%,当日成交量占总流通股比例（用于流动性分析）。
    ,VWAP,F32,元,成交量加权平均价（AMOUNT/VOLUME），停牌日由前一日填充。
    基本面,TOTAL_MV,F64,元,当日总市值（已换算为"元"，用于市值加权）。
    ,CIRC_MV,F64,元,当日流通市值（已换算为"元"，用于成分股筛选）。
    ,PE,F32,倍,市盈率（TTM/最近），停牌日由前一日填充。
    ,PB,F32,倍,市净率（最近），停牌日由前一日填充。
    ,PS,F32,倍,市销率（最近），停牌日由前一日填充。
    """

    def __init__(
        self, assets_mgr: StockAssetsManager, calendar_mgr: TradeCalendarManager
    ):
        self.assets_mgr = assets_mgr
        self.calendar_mgr = calendar_mgr
        self.cache_manager = HDF5CacheManager(settings.RAW_DATA_DIR)
        self.warehouse_dir = settings.WAREHOUSE_DIR
        # ✅ 缓存有效资产代码于初始化时，避免后续重复调用 get_all_codes()
        self._valid_asset_codes = assets_mgr.get_all_codes()

    def build_unified_factors(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> None:
        """
        构建 L2 统一因子库：
        内部自动按年拆分时间段，逐年执行 ETL 并独立保存，确保内存安全。
        (输入已限定为 date 类型)
        """
        # --- 1. 跨度解析与年份切分 ---
        all_years = list(range(start_date.year, end_date.year + 1))
        logger.info(
            f"🚀 开始任务：跨度 {start_date} -> {end_date}，拆分为 {len(all_years)} 个年度任务"
        )

        for year in all_years:
            # 动态计算年度区间
            cur_start = max(start_date, date(year, 1, 1))
            cur_end = min(end_date, date(year, 12, 31))
            self._execute_single_year_build(cur_start, cur_end, year)

        logger.success("✨ 所有年度任务已处理完毕。")

    def _resolve_query_end_date(self, end: Optional[date]) -> date:
        """解析查询结束日期，None 时返回统一因子库最新可用日期。"""
        if end is not None:
            return end

        factor_dir = self.warehouse_dir / "unified_factors"
        parquet_files = sorted(factor_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"未找到统一因子文件目录: {factor_dir}")

        max_dates: list[date] = []
        for file_path in parquet_files:
            max_dt = (
                pl.scan_parquet(file_path)
                .select(pl.col(F.DATE).max().alias("max_date"))
                .collect()
                .item(0, 0)
            )
            if max_dt is not None:
                max_dates.append(max_dt)

        if not max_dates:
            raise ValueError("统一因子库中未找到有效 DATE 数据")
        return max(max_dates)

    def datas(
        self,
        start: date,
        end: Optional[date] = None,
        assets: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """内部查询接口，用于数据校验。

        参数:
            start: 起始日期（含）。
            end: 结束日期（含），None 表示查询到最新日期。
            assets: 资产代码列表，None 表示不过滤。
            cols: 额外字段列表，返回结果始终包含 DATE 和 ASSET。
        """
        end_date = self._resolve_query_end_date(end)
        if start > end_date:
            raise ValueError(f"start({start}) 不得晚于 end({end_date})")

        factor_dir = self.warehouse_dir / "unified_factors"
        scans: list[pl.LazyFrame] = []
        for year in range(start.year, end_date.year + 1):
            file_path = factor_dir / f"{year}.parquet"
            if file_path.exists():
                scans.append(
                    pl.scan_parquet(file_path).with_columns(
                        pl.col(F.ASSET).cast(pl.String)
                    )
                )

        if not scans:
            raise FileNotFoundError(f"数据区间 {start} - {end_date} 无可用统一因子文件")

        lf = pl.concat(scans).filter(
            (pl.col(F.DATE) >= start) & (pl.col(F.DATE) <= end_date)
        )

        if assets:
            lf = lf.filter(pl.col(F.ASSET).is_in(assets))

        if cols:
            selected_cols = [
                F.DATE,
                F.ASSET,
                *[c for c in cols if c not in {F.DATE, F.ASSET}],
            ]
            lf = lf.select(selected_cols)

        return lf.collect()

    def _execute_single_year_build(
        self, start_dt: date, end_dt: date, year: int
    ) -> None:
        """
        [私有方法] 执行单一年度片段的 ETL 逻辑

        跨年连续性策略：在调用 _op_process_indicators 之前，将上一年最后一个
        交易日的已处理数据拼接在面板最前面，作为 forward_fill 的锚点种子行。
        上年数据经过完整 ETL，ffill_cols 必然非空，从而结构性保证本年第一天
        的 forward_fill 总有值可继承，无需依赖任意天数的 buffer。
        """
        logger.info(f"📂 正在处理 {year} 年度数据片段: {start_dt} -> {end_dt}")
        try:
            # --- 1. 获取当年交易日（不再需要前置 Buffer）---
            all_dates = self.calendar_mgr.get_trade_days(start_dt, end_dt)

            if not all_dates:
                logger.warning(f"⚠️ {year} 年在指定区间内无交易日，跳过。")
                return

            # --- 2. 算子流水线 (Lazy) ---
            skeleton = self._generate_skeleton_lf(all_dates)

            # 批量加载 L1 碎片
            daily_lf = self._op_clean_daily(all_dates)
            adj_lf = self._op_clean_adj(all_dates)
            basic_lf = self._op_clean_basic(all_dates)
            limit_lf = self._op_clean_limit(all_dates)
            st_lf = self._op_clean_st(all_dates)
            suspend_lf = self._op_clean_suspend(all_dates)
            disclosure_lf = self._op_clean_disclosure(all_dates)

            # 多路左连接
            panel = (
                skeleton.join(daily_lf, on=[F.DATE, F.ASSET], how="left")
                .join(adj_lf, on=[F.DATE, F.ASSET], how="left")
                .join(basic_lf, on=[F.DATE, F.ASSET], how="left")
                .join(limit_lf, on=[F.DATE, F.ASSET], how="left")
                .join(st_lf, on=[F.DATE, F.ASSET], how="left")
                .join(suspend_lf, on=[F.DATE, F.ASSET], how="left")
                .join(disclosure_lf, on=[F.DATE, F.ASSET], how="left")
            )

            # 💡 跨年锚点：将上一年最后一天已处理数据拼在最前，
            # 确保 forward_fill 在本年第一天总有非空种子
            anchor_lf = self._load_prev_year_anchor(year)
            if anchor_lf is not None:
                panel = pl.concat([anchor_lf, panel], how="diagonal_relaxed")

            # 核心指标处理 (包含 ST 填充、价格补全、复权计算)
            panel = self._op_process_indicators(panel)

            # --- 3. 落地存储 ---
            output_path = self.warehouse_dir / "unified_factors" / f"{year}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 触发计算（不尝试转为 Enum，保留字符串格式以支持新资产）
            df_full = panel.collect()

            # 过滤掉锚点行（上一年日期 < start_dt），只保留当年数据落盘
            df_year = df_full.filter(
                (pl.col(F.DATE) >= start_dt) & (pl.col(F.DATE) <= end_dt)
            )

            if df_year.is_empty():
                logger.warning(f"⚠️ {year} 年过滤后数据为空，不进行保存。")
                return

            # 💡 增量合并逻辑：检查文件是否存在，若存在则合并而非覆盖
            # 这解决了分次同步时数据被覆盖的问题
            # ⚠️ 关键修复：仅以 CLOSE_RAW 非空行作为"已填充"基准，
            # 避免旧的空壳行（CLOSE_RAW=null）占位导致无法写入正确价格数据
            if output_path.exists():
                # ✅ ASSET 保持为 String，无需往返转换
                df_existing = pl.read_parquet(output_path).filter(
                    pl.col(F.ASSET).is_not_null()
                )
                # 只保留有实际价格数据的行作为"已填充"基准
                df_populated = df_existing.filter(pl.col(F.CLOSE_RAW).is_not_null())
                existing_populated = df_populated.select([F.DATE, F.ASSET]).unique()
                # 新数据去除已有实际价格的行（避免重复写入），空壳行允许被覆盖
                new_data = df_year.join(
                    existing_populated, on=[F.DATE, F.ASSET], how="anti"
                )
                # 合并：保留现有有效数据 + 新数据（包含替换原空壳行）
                if not new_data.is_empty():
                    df_final = pl.concat([df_populated, new_data])
                else:
                    df_final = df_populated if not df_populated.is_empty() else df_year
                    logger.info(f"ℹ️ {year}.parquet 中该时间段数据已存在，无需重复写入")
            else:
                df_final = df_year

            # 写入 Parquet（保留 ASSET 为 String 类型以支持动态资产和跨年兼容性）
            df_final.write_parquet(output_path, compression="snappy")

            logger.info(
                f"💾 {year}.parquet 已保存 | 包含日期: {df_final['DATE'].min()} ~ {df_final['DATE'].max()} | 行数: {df_final.height}"
            )
        finally:
            # 💡 每次年度任务完成后手动清理一下 HDF5 句柄
            # 避免多年度连续同步时，同时打开过多的 .h5 文件
            self.cache_manager.close_all()

    # ================= 内部算子 (Lazy Operations) =================

    def _load_prev_year_anchor(self, year: int) -> pl.LazyFrame | None:
        """
        加载上一年度最后一个交易日的已处理数据，作为 forward_fill 的跨年锚点。

        返回 LazyFrame（含上年最后一天所有行）；若上年 parquet 不存在（首次构建），
        则返回 None，调用方跳过 concat 即可，保持全量首次构建的兼容性。

        ✅ 简化逻辑：ASSET 保存为 String，后续在 join 前统一转换类型。
        """
        prev_path = self.warehouse_dir / "unified_factors" / f"{year - 1}.parquet"
        if not prev_path.exists():
            return None

        # 直接读取，ASSET 保持为 String（避免 Categorical 类别集不兼容）
        lf = pl.scan_parquet(prev_path).filter(pl.col(F.ASSET).is_not_null())

        # 取最后一个交易日（仅需一次小 collect 获取单个日期标量）
        max_date = lf.select(pl.col(F.DATE).max()).collect().item(0, 0)
        if max_date is None:
            return None

        return lf.filter(pl.col(F.DATE) == max_date)

    def _generate_skeleton_lf(self, trading_dates: List[date]) -> pl.LazyFrame:
        """生成基于资产存续期的标准坐标轴"""
        date_df = pl.DataFrame({F.DATE: trading_dates}).select(
            pl.col(F.DATE).cast(pl.Date)
        )
        properties = self.assets_mgr.get_properties()

        return (
            date_df.join(
                properties.select(
                    [
                        pl.col(F.ASSET).cast(
                            pl.String
                        ),  # ✅ 保证 ASSET 为 String以支持后续 join
                        "list_date",
                        "delist_date",
                    ]
                ),
                how="cross",
            )
            .filter(
                (pl.col(F.DATE) >= pl.col("list_date"))
                & (
                    pl.col("delist_date").is_null()
                    | (pl.col(F.DATE) <= pl.col("delist_date"))
                )
            )
            .drop(["list_date", "delist_date"])
            .lazy()
        )

    def _validate_unified_factors(self, lf: pl.LazyFrame) -> None:
        """数据质量验证"""
        # 示例验证：检查关键主键是否包含 Null
        check = lf.select(
            [
                pl.col(F.DATE).null_count().alias("null_date"),
                pl.col(F.ASSET).null_count().alias("null_asset"),
            ]
        ).collect()

        if check["null_date"][0] > 0 or check["null_asset"][0] > 0:
            raise ValueError(f"✗ 关键坐标轴包含 Null 值: {check}")
        logger.debug("✓ 坐标轴完整性验证通过")
