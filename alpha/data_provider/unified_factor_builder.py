from __future__ import annotations

import datetime
from datetime import date
from typing import List
import pandas as pd
import polars as pl
import polars.selectors as cs
from loguru import logger

from alpha.utils.config import settings
from alpha.data_provider.cache_manager import HDF5CacheManager
from alpha.data_provider.stock_assets_manager import StockAssetsManager
from alpha.data_provider.trade_calendar_manager import TradeCalendarManager
from alpha.utils.schema import F


class UnifiedFactorBuilder:
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
状态,IS_ST,Bool,-,是否风险警示：基于 st 接口标记并前向填充。
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

    def __init__(self, assets_mgr: StockAssetsManager, calendar_mgr: TradeCalendarManager):
        self.assets_mgr = assets_mgr
        self.calendar_mgr = calendar_mgr
        self.cache_manager = HDF5CacheManager(settings.RAW_DATA_DIR)
        self.warehouse_dir = settings.WAREHOUSE_DIR

    def build_unified_factors(self, start_date: datetime.date, end_date: datetime.date) -> None:
        """
        构建 L2 统一因子库：
        内部自动按年拆分时间段，逐年执行 ETL 并独立保存，确保内存安全。
        (输入已限定为 date 类型)
        """
        # --- 1. 跨度解析与年份切分 ---
        all_years = list(range(start_date.year, end_date.year + 1))
        logger.info(f"🚀 开始任务：跨度 {start_date} -> {end_date}，拆分为 {len(all_years)} 个年度任务")

        for year in all_years:
            # 动态计算年度区间
            cur_start = max(start_date, date(year, 1, 1))
            cur_end = min(end_date, date(year, 12, 31))
            self._execute_single_year_build(cur_start, cur_end, year)

        logger.success("✨ 所有年度任务已处理完毕。")

    def _execute_single_year_build(self, start_dt: date, end_dt: date, year: int) -> None:
        """
        [私有方法] 执行单一年度片段的 ETL 逻辑
        加入了 30 天的前置 Buffer 机制，确保跨年数据填充的连续性。
        """
        logger.info(f"📂 正在处理 {year} 年度数据片段: {start_dt} -> {end_dt}")
        try:


            # --- 1. 获取带 Buffer 的交易日 ---
            # 向前多取 30 天，确保 1 月初的 forward_fill 有初始值
            buffer_start = start_dt - pd.Timedelta(days=30)
            all_dates = self.calendar_mgr.get_trade_days(buffer_start, end_dt)

            if not all_dates:
                logger.warning(f"⚠️ {year} 年在指定区间内无交易日，跳过。")
                return

            # --- 2. 算子流水线 (Lazy) ---
            # 基于 all_dates 生成骨架，确保 Buffer 期间的资产也在对齐范围内
            skeleton = self._generate_skeleton_lf(all_dates)

            # 批量加载 L1 碎片 (此时加载的是含 Buffer 的年度数据)
            daily_lf = self._op_clean_daily(all_dates)
            adj_lf = self._op_clean_adj(all_dates)
            basic_lf = self._op_clean_basic(all_dates)
            limit_lf = self._op_clean_limit(all_dates)
            st_lf = self._op_clean_st(all_dates)
            suspend_lf = self._op_clean_suspend(all_dates)

            # 多路左连接
            panel = (
                skeleton
                .join(daily_lf, on=[F.DATE, F.ASSET], how="left")
                .join(adj_lf, on=[F.DATE, F.ASSET], how="left")
                .join(basic_lf, on=[F.DATE, F.ASSET], how="left")
                .join(limit_lf, on=[F.DATE, F.ASSET], how="left")
                .join(st_lf, on=[F.DATE, F.ASSET], how="left")
                .join(suspend_lf, on=[F.DATE, F.ASSET], how="left")
            )

            # 核心指标处理 (包含 ST 填充、价格补全、复权计算)
            panel = self._op_process_indicators(panel)

            # --- 3. 落地存储前过滤 Buffer ---
            output_path = self.warehouse_dir / "unified_factors" / f"{year}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 触发计算（不尝试转为 Enum，保留字符串格式以支持新资产）
            df_full = panel.collect()

            # 💡 关键：过滤掉 Buffer 天数，只保留当前年度的数据落盘
            # 但此时 1 月 1 日的数据已经通过 Buffer 完成了前向填充
            df_year = df_full.filter(
                (pl.col(F.DATE) >= start_dt) & (pl.col(F.DATE) <= end_dt)
            )

            if df_year.is_empty():
                logger.warning(f"⚠️ {year} 年过滤后数据为空，不进行保存。")
                return

            # 写入 Parquet（保留 ASSET 为 String 类型以支持动态资产）
            df_year.write_parquet(output_path, compression="snappy")

            logger.info(
                f"💾 {year}.parquet 已保存 | 包含日期: {df_year['DATE'].min()} ~ {df_year['DATE'].max()} | 行数: {df_year.height}")
        finally:
            # 💡 每次年度任务完成后手动清理一下 HDF5 句柄
            # 避免多年度连续同步时，同时打开过多的 .h5 文件
            self.cache_manager.close_all()

    # ================= 内部算子 (Lazy Operations) =================

    def _generate_skeleton_lf(self, trading_dates: List[date]) -> pl.LazyFrame:
        """生成基于资产存续期的标准坐标轴"""
        date_df = pl.DataFrame({F.DATE: trading_dates}).select(pl.col(F.DATE).cast(pl.Date))
        properties = self.assets_mgr.get_properties()

        return (
            date_df.join(properties.select([F.ASSET, "list_date", "delist_date"]), how="cross")
            .filter(
                (pl.col(F.DATE) >= pl.col("list_date")) &
                (pl.col("delist_date").is_null() | (pl.col(F.DATE) <= pl.col("delist_date")))
            )
            .drop(["list_date", "delist_date"])
            .lazy()
        )

    # ================= 内部算子 (Lazy Operations) =================

    def _ensure_valid_assets(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """防火墙：剔除名录外代码并强制转换 Enum"""
        valid_codes = self.assets_mgr.get_all_codes()
        return (
            lf.filter(pl.col("ASSET").is_in(valid_codes))
            .with_columns(pl.col("ASSET").cast(self.assets_mgr.stock_type))
        )

    def _op_clean_daily(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗原始行情：使用 load_as_polars 获取数据"""
        # 1. 直接获取已经转好 Date 类型的 Polars DataFrame
        df_pl = self.cache_manager.load_as_polars("daily", trading_dates)
        if df_pl is None:
            return pl.LazyFrame(schema={F.DATE: pl.Date, F.ASSET: self.assets_mgr.stock_type})

        # 2. 这里的 DATE 和 ASSET 已经是正确类型，保留字符串以支持新资产
        return (self._ensure_valid_assets(df_pl.lazy())
        .select([
            pl.col(F.DATE),
            pl.col(F.ASSET).cast(self.assets_mgr.stock_type),  # 保留为字符串而非 Enum
            pl.col("open").cast(pl.Float32).alias("OPEN_RAW"),
            pl.col("high").cast(pl.Float32).alias("HIGH_RAW"),
            pl.col("low").cast(pl.Float32).alias("LOW_RAW"),
            pl.col("close").cast(pl.Float32).alias("CLOSE_RAW"),
            (pl.col("vol") * 100).cast(pl.Float32).alias("VOLUME"),  # 手 -> 股
            (pl.col("amount") * 1000).cast(pl.Float32).alias("AMOUNT"),  # 千元 -> 元
            # 💡 成交量加权平均价：AMOUNT / VOLUME（停牌日为 None，前向填充）
            pl.when(pl.col("vol") > 0)
                .then((pl.col("amount") * 1000 / (pl.col("vol") * 100)).cast(pl.Float32))
                .otherwise(None)
                .alias("VWAP"),
        ]))

    def _op_clean_adj(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("adj_factor", trading_dates)
        if df_pl is None:
            return pl.LazyFrame()

        return self._ensure_valid_assets(df_pl.lazy()).select([
            pl.col(F.DATE),
            pl.col(F.ASSET).cast(self.assets_mgr.stock_type),  # 保留为字符串
            pl.col("adj_factor").cast(pl.Float32).alias("ADJ_FACTOR"),
        ])

    def _op_clean_basic(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("daily_basic", trading_dates)
        if df_pl is None:
            return pl.LazyFrame(schema={
                F.DATE: pl.Date,
                F.ASSET: self.assets_mgr.stock_type,
                "PE": pl.Float32,
                "PB": pl.Float32,
                "PS": pl.Float32,
                "TURNOVER_RATE": pl.Float32,
                "TOTAL_MV": pl.Float64,
                "CIRC_MV": pl.Float64
            })

        return self._ensure_valid_assets(df_pl.lazy()).select([
            pl.col(F.DATE),
            pl.col(F.ASSET),  # 已经在 load_as_polars 重命名过，且在 _ensure_valid_assets 转了 Enum
            pl.col("pe").cast(pl.Float32).alias("PE"),
            pl.col("pb").cast(pl.Float32).alias("PB"),
            pl.col("ps").cast(pl.Float32).alias("PS"),
            pl.col("turnover_rate").cast(pl.Float32).alias("TURNOVER_RATE"),
            # 💡 这里一定要补齐 circ_mv，且金额换算为"元"
            (pl.col("total_mv") * 10000).cast(pl.Float64).alias("TOTAL_MV"),
            (pl.col("circ_mv") * 10000).cast(pl.Float64).alias("CIRC_MV"),
        ])

    def _op_clean_limit(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("stk_limit", trading_dates)
        if df_pl is None:
            return pl.LazyFrame()
        return self._ensure_valid_assets(df_pl.lazy()).select([
            pl.col(F.DATE),
            pl.col(F.ASSET).cast(self.assets_mgr.stock_type),
            pl.col("up_limit").cast(pl.Float32).alias("UP_LIMIT"),
            pl.col("down_limit").cast(pl.Float32).alias("DOWN_LIMIT"),
        ])

    def _op_clean_suspend(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗显式停牌数据"""
        df_pl = self.cache_manager.load_as_polars("suspend_d", trading_dates)

        if df_pl is None:
            return pl.LazyFrame(schema={
                F.DATE: pl.Date,
                F.ASSET: self.assets_mgr.stock_type,
                "_TMP_SUSPEND_": pl.Boolean # 💡 使用临时前缀，方便批量剔除
            })

        return self._ensure_valid_assets(df_pl.lazy()).select([
            pl.col(F.DATE),
            pl.col(F.ASSET),
            pl.lit(True).alias("_TMP_SUSPEND_")
        ])

    def _op_clean_st(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗 ST 标记数据"""
        # 注意：这里的 source 需与你 TushareDataService 同步时的名称一致
        df_pl = self.cache_manager.load_as_polars("st", trading_dates)

        # 如果没有 ST 数据（可能该年度无 ST 股票或未同步），返回带 Schema 的空表
        if df_pl is None:
            return pl.LazyFrame(schema={
                F.DATE: pl.Date,
                F.ASSET: self.assets_mgr.stock_type,
                "IS_ST": pl.Boolean
            })

        return self._ensure_valid_assets(df_pl.lazy()).select([
            pl.col(F.DATE),
            pl.col(F.ASSET).cast(self.assets_mgr.stock_type),
            pl.lit(True).alias("IS_ST")
        ])

    def _op_process_indicators(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """核心业务逻辑：填充、状态判定、复权计算"""
        ffill_cols = ["CLOSE_RAW", "ADJ_FACTOR", "TOTAL_MV", "CIRC_MV", "PE", "PB", "PS", "TURNOVER_RATE", "VWAP", "UP_LIMIT", "DOWN_LIMIT"]

        return (
            lf.sort([F.ASSET, F.DATE])
            .with_columns([
                # 1. 综合停牌判定：显式标记 OR 价格缺失
                (
                        pl.col("_TMP_SUSPEND_").fill_null(False) | pl.col("CLOSE_RAW").is_null()
                ).alias("IS_SUSPENDED"),

                # ST 状态填充
                pl.col("IS_ST").fill_null(False).forward_fill().over(F.ASSET),

                # 时序填充
                pl.col(ffill_cols).forward_fill().over(F.ASSET),
                pl.col(["VOLUME", "AMOUNT"]).fill_null(0.0),
            ])
            .with_columns([
                # 2. 停牌日价格补全
                pl.col("OPEN_RAW").fill_null(pl.col("CLOSE_RAW")),
                pl.col("HIGH_RAW").fill_null(pl.col("CLOSE_RAW")),
                pl.col("LOW_RAW").fill_null(pl.col("CLOSE_RAW")),
            ])
            .with_columns([
                # 3. 复权计算
                (pl.col("OPEN_RAW") * pl.col("ADJ_FACTOR")).cast(pl.Float32).alias("OPEN"),
                (pl.col("HIGH_RAW") * pl.col("ADJ_FACTOR")).cast(pl.Float32).alias("HIGH"),
                (pl.col("LOW_RAW") * pl.col("ADJ_FACTOR")).cast(pl.Float32).alias("LOW"),
                (pl.col("CLOSE_RAW") * pl.col("ADJ_FACTOR")).cast(pl.Float32).alias("CLOSE"),
            ])
            # 💡 4. 仅删除临时列，保留 _RAW 原始价格列和 ADJ_FACTOR 供后续分析使用
            .drop([
                cs.starts_with("_TMP_")
            ])
        )

    def _validate_unified_factors(self, lf: pl.LazyFrame) -> None:
        """数据质量验证"""
        # 示例验证：检查关键主键是否包含 Null
        check = lf.select([
            pl.col(F.DATE).null_count().alias("null_date"),
            pl.col(F.ASSET).null_count().alias("null_asset")
        ]).collect()

        if check["null_date"][0] > 0 or check["null_asset"][0] > 0:
            raise ValueError(f"✗ 关键坐标轴包含 Null 值: {check}")
        logger.debug("✓ 坐标轴完整性验证通过")
