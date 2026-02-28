import hashlib

import polars.selectors as cs

import polars as pl
from pathlib import Path

from loguru import logger
from typing import Optional, List, Union, Callable, Literal
from datetime import datetime, timedelta

from expr_codegen import codegen_exec


from alpha_factory.data_provider import TushareDataService
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager
from alpha_factory.config.base import settings
from alpha_factory.utils.schema import F


class DataProvider:
    """
    工业级声明式数据中枢 (L4 层)

    核心特性：
    1. 声明式架构：解耦“列生成”与“行过滤”逻辑。
    2. 表达式计算：集成 expr_codegen，支持 Batch 处理，自动清理中间变量。
    3. 冷启动支持：自动向前追溯（Lookback）以解决时序算子（MA/STD）的空值问题。
    4. 性能压榨：支持类型智能压缩 (shrink_dtype) 与 投影下压优化。
    """

    def __init__(self, asset_manager: Optional[StockAssetsManager] = None):
        self.warehouse_dir = Path(settings.WAREHOUSE_DIR)
        self.factor_dir = self.warehouse_dir / "unified_factors"
        self.asset_manager = asset_manager or StockAssetsManager()
        self.tushare_service = TushareDataService()

        # 预加载静态元数据 LazyFrame
        # 提示：确保 asset 列在管理器中已设为 Categorical 或 Enum
        self._static_props = self.asset_manager.get_properties().lazy()
        logger.debug("✓ DataProvider (Enhanced) 初始化完成")

    def load_pool_data(
        self,
        pool: PoolUniverse,
        start_date: str,
        end_date: Optional[str] = None,
        exprs: Optional[List] = None,
        cache: Optional[Union[str, Path]] = "md5",
    ) -> pl.LazyFrame:
        return self.load_data(
            start_date,
            end_date,
            column_exprs=exprs,
            funcs=[pool.pool, pool.extra_cols, *pool.label_col_funcs],
            select_cols=pool.needed_cols(),
            cache_path=cache,
        )

    def load_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        column_blocks: Optional[List] = None,
        column_exprs: Optional[List[str]] = None,
        funcs: Optional[List[Callable[[pl.LazyFrame], pl.LazyFrame]]] = None,
        lookback_window: int = 0,
        select_cols: Optional[List] = None,
        cache_path: Optional[Union[str, Path]] = None,  # 🆕 新增缓存路径参数
        codegen_over_null: Literal["partition_by", "order_by", None] = None,
    ) -> pl.LazyFrame:
        """
        统一数据集构建管线（带持久化缓存支持）\n
        :param start_date: 起始日期 (YYYYMMDD)
        :param end_date: 结束日期 (YYYYMMDD)
        :param column_blocks: 列生成函数块列表 (func block 型)
        :param column_exprs: 列生成表达式列表 (expr 型)
        :param funcs: 自定义函数列表 (每个函数接受并返回 LazyFrame)
        :param lookback_window: 向前预热天数 (解决时序算子空值问题)
        :param select_cols: 最终投影列列表 (None 表示之后自行选择)，这里不包括表达式自动生产的列
        :param cache_path: 可选的缓存文件路径 (Parquet 格式)，命中则直接加载
        :param codegen_over_null: expr_codegen 的 over_null 参数
        :return: 构建完成的 LazyFrame
        说明：
        1. 优先检查 cache_path 指定的缓存文件是否存在，若存在则直接加载返回。
        2. 若缓存未命中，则执行完整的计算流水线：
           - 物理层扫描（支持 lookback）
           - 基础上下文增强（注入静态属性列）
           - 列生成（支持 func block 和 expr 两种模式）
           - 自定义函数处理
           - 时间切片与行过滤
           - 最终投影下压
        3. 若指定了 cache_path，则在计算完成后将结果持久化为 Parquet 文件以供后续加载。
        4. 返回的始终是 LazyFrame，确保后续处理链路一致性。
        5. 通过合理使用缓存，可大幅提升重复查询的性能。
        6. 注意：缓存文件的管理（如清理过期缓存）需由调用方负责。
        7. 示例用法：
           dp = DataProvider()
           lf = dp.load_data(
               start_date="20220101",
               end_date="20221231",
               column_exprs=["MA_20 = CLOSE.rolling_mean(20)"],
               lookback_window=20,
               select_cols=["DATE", "ASSET", "CLOSE", "MA_20"],
               cache_path="cache/2022_factors.parquet"
           )
        说明：上述示例会尝试加载指定的缓存文件，若不存在则计算所需列并将结果缓存。
        备注：合理设置 lookback_window 可确保时序算子（如移动平均）在起始日期处有足够的数据支持，避免空值问题。
        进阶：结合 expr_codegen 的批量处理能力，可高效生成大量衍生列，提升数据处理效率。
        适用场景：
        - 高频查询同一时间区间的数据时，缓存机制能显著减少重复计算开销。
        - 复杂列生成逻辑通过声明式表达式和函数块实现，提升代码可维护性和复用性。
        - 适用于量化研究、因子开发等需要灵活数据处理的场景。
        设计目标：
        - 提供一个高性能、易用且灵活的数据加载与处理框架。
        - 通过缓存机制优化重复查询的性能，提升用户体验。
        - 支持多种列生成方式，满足不同用户的需求。
        """

        # 1. 🔍 检查缓存命中
        if cache_path:
            # 由start_date和end_date ，column_blocks 等参数形成md5作为缓存文件名的一部分更好
            if cache_path == "md5":
                # 1. 准备需要哈希的内容字符串
                # 建议将列表/字典等对象先 str() 化
                hash_content = f"{start_date}_{end_date}_{lookback_window}_{funcs}_{column_blocks}_{column_exprs}_{select_cols}"

                # 2. 使用标准 hashlib 计算 MD5
                # hex digest 返回的是标准的 32 位 16 进制字符串
                md5_hash = hashlib.md5(hash_content.encode("utf-8")).hexdigest()

                # 3. 构建路径 (假设默认前缀为 'cached_data')
                # 注意：不要对字符串使用 .stem，直接构建文件名
                file_name = f"factor_data_{md5_hash}.parquet"
                cache_path = Path(settings.OUTPUT_DIR) / "tmp_data" / file_name

                # 确保目录存在，防止写入时报错
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"🆕 缓存路径已生成: {cache_path}")

            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info(f"✨ 发现缓存，直接加载: {cache_path}")
                # 使用 scan_parquet 保持 Lazy 特性
                lf = pl.scan_parquet(cache_path)
                return lf.with_columns(
                    # 转换成float64，避免后续计算中类型不匹配的问题
                    pl.col(pl.NUMERIC_DTYPES).cast(pl.Float64)
                )

        end_date = (
            self.tushare_service.get_latest_date_from_warehouse()
            if end_date is None
            else end_date.strip()
        )

        # 2. 🏗️ 执行完整计算流水线 (如果缓存未命中或未设置)
        logger.info(
            f"⚙️ 缓存未命中或未设置，开始计算数据 [{start_date} -> {end_date}]..."
        )

        # A. 物理层扫描
        lf = self._scan_with_lookback(start_date, end_date, lookback_window)

        # B. 基础上下文增强（与assets join）
        lf = self._enrich_context(lf)

        # C. 函数型，这里既可以生成新列，也可以用来过滤行,以及其他任何操作
        if funcs:
            for i, func in enumerate(funcs):
                try:
                    lf = func(lf)
                except Exception as e:
                    logger.error(f"❌ 自定义函数 #{i} 执行失败: {e}")
                    raise

        template_path = settings.template_path_str
        # D. 列生成：func block 型，expr_codegen 支持批量处理
        if column_blocks:
            # 因为使用了POOL_MASK ,函数映射必须加上自定义操作符
            lf = codegen_exec(
                lf,
                *column_blocks,
                over_null=codegen_over_null,
                template_file=template_path,
                date="DATE",  # 这里不能使用F.DATE，因为 codegen_exec 因为 codegen_exec 内部是通过模版渲染的
                asset="ASSET",  # 同上，保持字符串一致
            )

        # E. 列生成：表达式型
        generated_expr_cols = []
        if column_exprs:
            for expr_str in column_exprs:
                if "=" in expr_str:
                    generated_expr_cols.append(expr_str.split("=")[0].strip())

            batch_size = getattr(settings, "CODEGEN_BATCH_SIZE", 100)
            for i in range(0, len(column_exprs), batch_size):
                batch = column_exprs[i : i + batch_size]
                lf = codegen_exec(
                    lf,
                    *batch,
                    over_null=codegen_over_null,
                    template_file=template_path,
                    date="DATE",  # 这里不能使用F.DATE，因为 codegen_exec 内部是通过模版渲染的
                    asset="ASSET",  # 同上，保持字符串一致
                )

        # F. 时间切片 & 行过滤
        s_dt = datetime.strptime(start_date, "%Y%m%d").date()
        lf = lf.filter(pl.col("DATE") >= s_dt)

        # G. 投影
        if select_cols:
            lf = self._finalize_projection(lf, select_cols, generated_expr_cols)

        # lf.sort('ASSET', 'DATE').with_columns(
        #     pl.col("ASSET").set_sorted(True)
        # )

        # 3. 💾 持久化缓存 (如果指定了 cache_path)
        if cache_path:
            # 注意：LazyFrame 必须 collect 之后才能 write_parquet
            # 或者使用 sink_parquet (如果是流式支持的操作)
            # 为了稳健性，这里先 collect
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📥 正在将计算结果写入缓存: {cache_path}")

            # 执行计算并保存
            df = lf.collect()
            # df.shrink_dtype()  # 类型压缩
            df.write_parquet(cache_path, compression="zstd")

            # 返回保存后的 Lazy 视图，确保后续链路统一
            lf = pl.scan_parquet(cache_path)

        return lf.with_columns(
            # 全部转换成float64，避免后续计算中类型不匹配的问题
            cs.numeric().cast(pl.Float64)
        )

    def clean_old_caches(self, days=7):
        tmp_path = Path(settings.OUTPUT_DIR) / "tmp_data"
        now = datetime.now().timestamp()
        for f in tmp_path.glob("factor_data_*.parquet"):
            if f.stat().st_mtime < (now - days * 86400):
                f.unlink()

    # --- 内部核心组件 ---

    def _scan_with_lookback(
        self, start_date: str, end_date: str, lookback: int
    ) -> pl.LazyFrame:
        """根据 lookback 天数自动向前扩充扫描年份"""
        s_dt = datetime.strptime(start_date, "%Y%m%d").date()
        e_dt = datetime.strptime(end_date, "%Y%m%d").date()

        # 预估预热所需的起始日期（交易日天数 * 1.5 倍近似自然日）
        effective_start = s_dt - timedelta(days=int(lookback * 1.5) + 7)

        scans = []
        for year in range(effective_start.year, e_dt.year + 1):
            file_path = self.factor_dir / f"{year}.parquet"
            if file_path.exists():
                scans.append(pl.scan_parquet(file_path))

        if not scans:
            raise FileNotFoundError(f"数据区间 {start_date}-{end_date} 无可用文件")

        # 此时不过滤 start_date，只过滤 end_date，保留预热空间
        return pl.concat(scans).filter(pl.col("DATE") <= e_dt)

    def _enrich_context(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        注入物理环境列,以及一些常用的列
        LIST_DAYS: 上市天数
        IS_UP_LIMIT: 是否涨停
        IS_DOWN_LIMIT: 是否跌停
        TOTAL_MV_PCT: 截面市值百分位
        EXCHANGE: 交易所(主板/创业板/科创板)
        MARKET_TYPE: 市场类型（SZSE/SSE/BSE）

        """
        return lf.join(
            self._static_props, left_on=F.ASSET, right_on=F.ASSET, how="left"
        ).with_columns(
            [
                # 计算上市天数
                (pl.col(F.DATE).cast(pl.Date) - pl.col("list_date"))
                .dt.total_days()
                .fill_null(0)
                .alias("LIST_DAYS"),
                # 识别基础交易限制
                (pl.col("CLOSE_RAW") >= pl.col("UP_LIMIT") - 0.001).alias(
                    "IS_UP_LIMIT"
                ),
                (pl.col("CLOSE_RAW") <= pl.col("DOWN_LIMIT") + 0.001).alias(
                    "IS_DOWN_LIMIT"
                ),
                # 计算截面市值百分位
                (
                    pl.col("TOTAL_MV").rank().over(F.DATE)
                    / pl.col(F.ASSET).count().over(F.DATE)
                ).alias("TOTAL_MV_PCT"),
                # 关键修复：在此处转换，避免后续 filter 中的严格类型检查
                pl.col("exchange").alias("EXCHANGE"),
                pl.col("market").alias("MARKET_TYPE"),
            ]
        )

    def _finalize_projection(
        self, lf: pl.LazyFrame, base_cols: List[str], generated_cols: List[str]
    ) -> pl.LazyFrame:
        """动态感知列空间并执行投影下压"""
        # 默认始终保留的 ID 和状态列
        essential = [
            F.DATE,
            F.ASSET,
        ]

        # 汇总所有请求的列
        requested = essential + (base_cols or []) + generated_cols
        requested = list(dict.fromkeys(requested))

        # # 动态获取当前 LazyFrame 的 Schema，防止 select 不存在的列
        available_cols = set(lf.collect_schema().names())
        final_selection = [c for c in requested if c in available_cols]

        return lf.select(final_selection)
