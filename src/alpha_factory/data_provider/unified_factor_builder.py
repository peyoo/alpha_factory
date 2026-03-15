from __future__ import annotations

import datetime
from datetime import date
from typing import List, Optional
import polars as pl
import polars.selectors as cs
from loguru import logger

from alpha_factory.config.base import settings
from alpha_factory.data_provider.cache_manager import HDF5CacheManager
from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager
from alpha_factory.data_provider.trade_calendar_manager import TradeCalendarManager
from alpha_factory.utils.schema import F


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
            if output_path.exists():
                # 读取现有数据，将 ASSET 经由 String 重新 cast 到当前 session 的 Enum 类型。
                # 原因：不同批次写入时 Enum 类别集合可能不同，直接 join 会报类型不匹配；
                # strict=False 保证极少数已退市且已从名录移除的代码变为 null 而非抛错。
                df_existing = (
                    pl.read_parquet(output_path)
                    .with_columns(
                        pl.col(F.ASSET)
                        .cast(pl.String)
                        .cast(self.assets_mgr.stock_type, strict=False)
                    )
                    .filter(pl.col(F.ASSET).is_not_null())
                )
                # 删除新数据中已存在的 (DATE, ASSET) 组合，避免重复
                existing_dates = df_existing.select([F.DATE, F.ASSET]).unique()
                new_data = df_year.join(
                    existing_dates, on=[F.DATE, F.ASSET], how="anti"
                )
                # 合并：保留现有数据 + 新数据
                if not new_data.is_empty():
                    df_final = pl.concat([df_existing, new_data])
                else:
                    df_final = df_existing
                    logger.info(f"ℹ️ {year}.parquet 中该时间段数据已存在，无需重复写入")
            else:
                df_final = df_year

            # 写入 Parquet（保留 ASSET 为 String 类型以支持动态资产）
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
        """
        prev_path = self.warehouse_dir / "unified_factors" / f"{year - 1}.parquet"
        if not prev_path.exists():
            return None

        # 读取上一年 parquet，将 ASSET 重新绑定到当前 session 的 Enum 类型
        lf = (
            pl.scan_parquet(prev_path)
            .with_columns(
                pl.col(F.ASSET)
                .cast(pl.String)
                .cast(self.assets_mgr.stock_type, strict=False)
            )
            .filter(pl.col(F.ASSET).is_not_null())
        )

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
                properties.select([F.ASSET, "list_date", "delist_date"]), how="cross"
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

    # ================= 内部算子 (Lazy Operations) =================

    def _ensure_valid_assets(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """防火墙：剔除名录外代码并强制转换 Enum"""
        valid_codes = self.assets_mgr.get_all_codes()
        return lf.filter(pl.col(F.ASSET).is_in(valid_codes)).with_columns(
            pl.col(F.ASSET).cast(self.assets_mgr.stock_type)
        )

    def _op_clean_daily(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗原始行情：使用 load_as_polars 获取数据"""
        # 1. 直接获取已经转好 Date 类型的 Polars DataFrame
        df_pl = self.cache_manager.load_as_polars("daily", trading_dates)
        if df_pl is None:
            return pl.LazyFrame(
                schema={F.DATE: pl.Date, F.ASSET: self.assets_mgr.stock_type}
            )

        # 2. 这里的 DATE 和 ASSET 已经是正确类型，保留字符串以支持新资产
        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET).cast(
                    self.assets_mgr.stock_type
                ),  # 保留为字符串而非 Enum
                pl.col("open").cast(pl.Float32).alias(F.OPEN_RAW),
                pl.col("high").cast(pl.Float32).alias(F.HIGH_RAW),
                pl.col("low").cast(pl.Float32).alias(F.LOW_RAW),
                pl.col("close").cast(pl.Float32).alias(F.CLOSE_RAW),
                (pl.col("vol") * 100).cast(pl.Float32).alias(F.VOLUME),  # 手 -> 股
                (pl.col("amount") * 1000)
                .cast(pl.Float32)
                .alias(F.AMOUNT),  # 千元 -> 元
                # 💡 成交量加权平均价：AMOUNT / VOLUME（停牌日为 None，前向填充）
                pl.when(pl.col("vol") > 0)
                .then(
                    (pl.col("amount") * 1000 / (pl.col("vol") * 100)).cast(pl.Float32)
                )
                .otherwise(None)
                .alias(F.VWAP_RAW),
            ]
        )

    def _op_clean_adj(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("adj_factor", trading_dates)
        if df_pl is None:
            return pl.LazyFrame()

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET).cast(self.assets_mgr.stock_type),  # 保留为字符串
                pl.col("adj_factor").cast(pl.Float32).alias("ADJ_FACTOR"),
            ]
        )

    def _op_clean_basic(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("daily_basic", trading_dates)
        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    "PE": pl.Float32,
                    "PB": pl.Float32,
                    "PS": pl.Float32,
                    "TURNOVER_RATE": pl.Float32,
                    "TOTAL_MV": pl.Float64,
                    "CIRC_MV": pl.Float64,
                }
            )

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(
                    F.ASSET
                ),  # 已经在 load_as_polars 重命名过，且在 _ensure_valid_assets 转了 Enum
                pl.col("pe").cast(pl.Float32).alias(F.PE),
                pl.col("pb").cast(pl.Float32).alias(F.PB),
                pl.col("ps").cast(pl.Float32).alias(F.PS),
                pl.col("turnover_rate").cast(pl.Float32).alias(F.TURNOVER_RATE),
                # 💡 这里一定要补齐 circ_mv，且金额换算为"元"
                (pl.col("total_mv") * 10000).cast(pl.Float64).alias(F.TOTAL_MV),
                (pl.col("circ_mv") * 10000).cast(pl.Float64).alias(F.CIRC_MV),
            ]
        )

    def _op_clean_limit(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("stk_limit", trading_dates)
        if df_pl is None:
            return pl.LazyFrame()
        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET).cast(self.assets_mgr.stock_type),
                pl.col("up_limit").cast(pl.Float32).alias(F.UP_LIMIT),
                pl.col("down_limit").cast(pl.Float32).alias(F.DOWN_LIMIT),
            ]
        )

    def _op_clean_suspend(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗显式停牌数据"""
        df_pl = self.cache_manager.load_as_polars("suspend_d", trading_dates)

        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    "_TMP_SUSPEND_": pl.Boolean,  # 💡 使用临时前缀，方便批量剔除
                }
            )

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [pl.col(F.DATE), pl.col(F.ASSET), pl.lit(True).alias("_TMP_SUSPEND_")]
        )

    def _op_clean_st(self, trading_dates: List[date]) -> pl.LazyFrame:
        """
        加载融合的 ST 标记因子

        【数据源】
        "st" - 融合数据源（namechange + stock_st 并集）
        - namechange 名称规则 OR stock_st 列表中存在 → is_st=True
        - 否则 → is_st=False

        【填充策略】
        - 缺失值采用 forward_fill 填充（按 ASSET 分组，按 DATE 排序）
        """
        df_st = self.cache_manager.load_as_polars("st", trading_dates)

        # 如果没有数据，返回带 Schema 的空表
        if df_st is None or "is_st" not in df_st.columns:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    F.IS_ST: pl.Boolean,
                }
            )

        # 使用融合的 ST 标记，采用 forward_fill 填充缺失值
        lf = self._ensure_valid_assets(df_st.lazy())
        return (
            lf.sort([F.ASSET, F.DATE])
            .with_columns(
                pl.col("is_st")
                .forward_fill()
                .over(F.ASSET)
                .fill_null(False)
                .cast(pl.Boolean)
                .alias(F.IS_ST)
            )
            .select(
                [
                    pl.col(F.DATE),
                    pl.col(F.ASSET).cast(self.assets_mgr.stock_type),
                    pl.col(F.IS_ST),
                ]
            )
        )

    def _op_clean_disclosure(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗披露数据：提取每个资产每年的最新 ann_date，并展开到 4 月交易日。"""
        if not trading_dates:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    "_TMP_APRIL_ANN_DATE": pl.Date,
                }
            )

        report_end_dates = sorted({date(d.year - 1, 12, 31) for d in trading_dates})
        df_pl = self.cache_manager.load_as_polars("disclosure_date", report_end_dates)
        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    "_TMP_APRIL_ANN_DATE": pl.Date,
                }
            )

        valid_codes = set(self.assets_mgr.get_all_codes())
        april_plan = (
            df_pl.lazy()
            .filter(pl.col(F.ASSET).is_in(valid_codes))
            .with_columns(
                [
                    # 💡 优先使用 actual_date（实际披露日期），其次 ann_date（公告日期）
                    pl.when(pl.col("actual_date").is_not_null())
                    .then(pl.col("actual_date").cast(pl.Utf8))
                    .when(pl.col("ann_date").is_not_null())
                    .then(pl.col("ann_date").cast(pl.Utf8))
                    .otherwise(None)
                    .str.strptime(pl.Date, "%Y%m%d", strict=False)
                    .alias("_ACTUAL_DISCLOSURE_DATE"),
                ]
            )
            .filter(pl.col("_ACTUAL_DISCLOSURE_DATE").is_not_null())
            .with_columns(
                pl.col("_ACTUAL_DISCLOSURE_DATE").dt.year().alias("_PLAN_YEAR")
            )
            .group_by([F.ASSET, "_PLAN_YEAR"])
            .agg(pl.col("_ACTUAL_DISCLOSURE_DATE").max().alias("_TMP_APRIL_ANN_DATE"))
        )

        april_dates = [d for d in trading_dates if d.month == 4]
        if not april_dates:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: self.assets_mgr.stock_type,
                    "_TMP_APRIL_ANN_DATE": pl.Date,
                }
            )

        april_dates_lf = (
            pl.DataFrame({F.DATE: april_dates})
            .lazy()
            .with_columns(pl.col(F.DATE).dt.year().alias("_PLAN_YEAR"))
        )

        return (
            april_dates_lf.join(april_plan, on="_PLAN_YEAR", how="inner")
            .select(
                [
                    pl.col(F.DATE),
                    pl.col(F.ASSET).cast(self.assets_mgr.stock_type),
                    pl.col("_TMP_APRIL_ANN_DATE"),
                ]
            )
            .unique([F.DATE, F.ASSET], keep="last")
        )

    def _op_process_indicators(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """核心业务逻辑：填充、状态判定、复权计算"""
        schema_names = set(lf.collect_schema().names())
        additional_cols: list[pl.Expr] = []
        if "_TMP_APRIL_ANN_DATE" not in schema_names:
            additional_cols.append(
                pl.lit(None).cast(pl.Date).alias("_TMP_APRIL_ANN_DATE")
            )

        ffill_cols = [
            F.CLOSE_RAW,
            F.ADJ_FACTOR,
            F.TOTAL_MV,
            F.CIRC_MV,
            F.PE,
            F.PB,
            F.PS,
            F.TURNOVER_RATE,
            F.VWAP_RAW,
            F.UP_LIMIT,
            F.DOWN_LIMIT,
        ]

        return (
            lf.sort([F.ASSET, F.DATE])
            .with_columns(additional_cols)
            .with_columns(
                [
                    # 1. 综合停牌判定：显式标记 OR 价格缺失
                    (
                        pl.col("_TMP_SUSPEND_").fill_null(False)
                        | pl.col(F.CLOSE_RAW).is_null()
                    ).alias(F.IS_SUSPENDED),
                    # ST 状态传递：若上一交易日为 True，则当前交易日保持 True
                    pl.col(F.IS_ST)
                    .fill_null(False)
                    .cast(pl.Int8)
                    .cum_max()
                    .over(F.ASSET)
                    .cast(pl.Boolean)
                    .alias(F.IS_ST),
                    pl.when(pl.col(F.DATE).dt.month() < 4)
                    .then(pl.lit(True))
                    .when(pl.col(F.DATE).dt.month() == 4)
                    .then(
                        pl.col("_TMP_APRIL_ANN_DATE").is_not_null()
                        & (pl.col(F.DATE) > pl.col("_TMP_APRIL_ANN_DATE"))
                    )
                    .otherwise(None)
                    .alias(F.APRIL_DISCLOSURE_SIGNAL),
                    pl.col([F.VOLUME, F.AMOUNT]).fill_null(0.0),
                ]
            )
            .with_columns(
                [
                    # 时序填充
                    pl.col(ffill_cols).forward_fill().over(F.ASSET),
                    pl.col(F.APRIL_DISCLOSURE_SIGNAL).forward_fill().over(F.ASSET),
                ]
            )
            .with_columns(
                [
                    # 2. 停牌日价格补全
                    pl.col(F.OPEN_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.HIGH_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.LOW_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.VWAP_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                ]
            )
            .with_columns(
                [
                    # 3. 复权计算
                    (pl.col(F.OPEN_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.OPEN),
                    (pl.col(F.HIGH_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.HIGH),
                    (pl.col(F.LOW_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.LOW),
                    (pl.col(F.CLOSE_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.CLOSE),
                    (pl.col(F.VWAP_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.VWAP),
                ]
            )
            # 💡 4. 仅删除临时列，保留 _RAW 原始价格列和 ADJ_FACTOR 供后续分析使用
            .drop([cs.starts_with("_TMP_")])
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
