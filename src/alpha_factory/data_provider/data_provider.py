import hashlib
import inspect

import polars.selectors as cs

import polars as pl
from pathlib import Path

from loguru import logger
from typing import Optional, List, Union, Literal
from datetime import date, datetime, timedelta

from expr_codegen import codegen_exec


from alpha_factory.data_provider import TushareDataService
from alpha_factory.data_provider.pool import PoolUniverse
from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager
from alpha_factory.config.base import settings
from alpha_factory.utils.schema import F

# str（YYYYMMDD）或 datetime.date 均可接受
DateLike = Union[str, date]


class DataProvider:
    """
    工业级声明式数据中枢 (L4 层)

    核心特性：
    1. 声明式架构：解耦“列生成”与“行过滤”逻辑。
    2. 表达式计算：集成 expr_codegen，支持 Batch 处理，自动清理中间变量。
    3. 冷启动支持：自动向前追溯（Lookback）以解决时序算子（MA/STD）的空值问题。
    4. 性能压榨：支持类型智能压缩 (shrink_dtype) 与 投影下压优化。
    """

    def __init__(
        self,
        asset_manager: Optional[StockAssetsManager] = None,
        look_back_window: int = 200,
        pool_data_in_memory: bool = False,
    ):
        """初始化 DataProvider。

        参数:
            asset_manager: 可选的资产元数据管理器。未提供时使用默认实现。
            look_back_window: 时序算子预热窗口（交易日），默认 200。
                用于向前追溯以解决 MA/STD 等时序算子的冷启动空值问题。
            pool_data_in_memory: 是否将 pool 基础数据缓存到内存。
                为 True 时，同一 pool+日期区间的基础数据将常驻内存，
                后续调用 load_pool_data / build_factors_view 时可直接复用，
                避免重复磁盘 I/O。适合在同一进程中对同一 pool 计算多组表达式的场景。

        说明:
            - 初始化仓库路径与因子目录。
            - 预加载静态属性表为 LazyFrame，供后续 join 复用。
            - 创建 Tushare 服务实例用于日期边界推断。
        """
        self.warehouse_dir = Path(settings.WAREHOUSE_DIR)
        self.factor_dir = self.warehouse_dir / "unified_factors"
        self.asset_manager = asset_manager or StockAssetsManager()
        self.tushare_service = TushareDataService()
        self.look_back_window: int = look_back_window
        self.pool_data_in_memory: bool = pool_data_in_memory

        # pool_data_in_memory=True 时缓存 (lf, pool_cache_path)（一个 DataProvider 绑定一个 pool，
        # 同一运行会话内 pool_cache_path 不变，一并存储避免重复哈希计算）
        self._pool_memory_cache: Optional[tuple[pl.LazyFrame, Path]] = None

        # 预加载静态元数据 LazyFrame
        # 提示：确保 asset 列在管理器中已设为 Categorical 或 Enum
        self._static_props = self.asset_manager.get_properties().lazy()
        logger.debug("✓ DataProvider (Enhanced) 初始化完成")

    def load_pool_data(
        self,
        pool: PoolUniverse,
        start_date: DateLike,
        end_date: Optional[DateLike] = None,
        exprs: Optional[List] = None,
        cache: Optional[Union[str, Path]] = None,
    ) -> pl.LazyFrame:
        """按股票池加载数据，并支持两阶段缓存策略。

        参数:
            pool: 股票池对象，定义过滤逻辑与基础列需求。
            start_date: 起始日期，接受 str（YYYYMMDD）或 datetime.date。
            end_date: 结束日期，None 时自动推断仓库最新日期。
            exprs: 因子表达式列表，支持 `name = expr` 形式。
            cache: 缓存策略。
                - `"md5"`: pool 基础数据缓存 + factors 结果按 MD5 自动合成路径缓存。
                - `Path/str`: 显式 factors 缓存路径（相对路径转绝对路径）。
                - `None`: 不缓存 factors 结果（pool 基础数据仍缓存）。

        返回:
            pl.LazyFrame，包含基础列与（可选）表达式生成列。
        """
        # 入口统一规范化为 date，内部全程使用 date 类型
        start_dt: date = self._to_date(start_date)
        end_dt: date = self._resolve_end_date(end_date)

        pool_data, pool_cache_path = self._build_pool_base_data(pool, start_dt, end_dt)
        factors_cache_path = self._build_factors_cache_path(
            pool_cache_path, exprs, cache
        )

        lf = self._build_factors_view(pool, pool_data, exprs, factors_cache_path)

        # 在最后阶段过滤到 start_dt 及以后（表达式计算已完成，可以安全过滤）
        return lf.filter(pl.col("DATE") >= start_dt)

    def _build_pool_base_data(
        self,
        pool: PoolUniverse,
        start_date: date,
        end_date: date,
    ) -> tuple[pl.LazyFrame, Path]:
        """构建并缓存 pool 基础数据。

        返回:
            (pool_data_lf, pool_cache_path) — pool 级缓存始终生效。
        """
        # 优先查询内存缓存（pool_data_in_memory=True 时生效，同时跳过哈希计算）
        if self.pool_data_in_memory and self._pool_memory_cache is not None:
            logger.info("⚡ 命中内存缓存，直接复用 pool 基础数据")
            return self._pool_memory_cache

        pool_cache_path = self._build_pool_cache_path(pool, start_date, end_date)

        cached_lf = self._load_cached_lazyframe(pool_cache_path)
        if cached_lf is not None:
            if self.pool_data_in_memory:
                self._pool_memory_cache = (cached_lf, pool_cache_path)
            return cached_lf, pool_cache_path

        logger.info(
            f"⚙️ 构建股票池基础数据 [{start_date.strftime('%Y%m%d')} -> {end_date.strftime('%Y%m%d')}]..."
        )

        funcs = [pool.extra_cols, pool.pool, *pool.label_col_funcs]
        select_cols = pool.needed_cols()

        lf = self._scan_with_lookback(
            start_date, end_date, lookback=self.look_back_window
        )
        lf = self._enrich_context(lf)

        for i, func in enumerate(funcs):
            try:
                lf = func(lf)
            except Exception as e:
                logger.error(f"❌ 自定义函数 #{i} 执行失败: {e}")
                raise

        # 注意：不在这里过滤时间，保留 lookback 数据用于表达式计算（如 ts_mean）
        # 时间过滤延后到 load_pool_data 最后阶段

        if select_cols:
            lf = self._finalize_projection(lf, select_cols, generated_cols=[])

        lf = self._persist_cache_and_reload(lf, pool_cache_path)
        if self.pool_data_in_memory:
            self._pool_memory_cache = (lf, pool_cache_path)
        return lf, pool_cache_path

    def clean_old_caches(self, days=1):
        """清理旧缓存文件。

        参数:
            days: 仅保留最近 N 天缓存，默认 7 天。
        """
        tmp_path = Path(settings.OUTPUT_DIR) / "tmp_data"
        now = datetime.now().timestamp()
        for pattern in ("factor_data_*.parquet", "pool_base_*.parquet"):
            for f in tmp_path.glob(pattern):
                if f.stat().st_mtime < (now - days * 86400):
                    f.unlink()

    @staticmethod
    def _to_date(d: DateLike) -> date:
        """将 str（YYYYMMDD）或 date 统一转换为 date 对象。"""
        if isinstance(d, date):
            return d
        return datetime.strptime(d.strip(), "%Y%m%d").date()

    def _resolve_end_date(self, end_date: Optional[DateLike]) -> date:
        """解析结束日期，统一返回 date 对象。

        - `None` 时返回仓库最新可用交易日。
        - 非空时规范化为 date。
        """
        if end_date is None:
            return self._to_date(self.tushare_service.get_latest_date_from_warehouse())
        return self._to_date(end_date)

    def _build_pool_cache_path(
        self, pool: PoolUniverse, start_date: date, end_date: date
    ) -> Path:
        """生成 pool 基础数据的缓存文件路径（哈希键内聚到此方法）。

        key 来源: pool 类源码 + start_date + end_date
        """
        try:
            pool_source = inspect.getsource(pool.__class__)
        except (TypeError, OSError):
            class_obj = pool.__class__
            class_signature = sorted(class_obj.__dict__.keys())
            pool_source = (
                f"{class_obj.__module__}.{class_obj.__qualname__}:{class_signature}"
            )

        pool_key = hashlib.md5(
            f"{pool_source}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}".encode(
                "utf-8"
            )
        ).hexdigest()
        return Path(settings.OUTPUT_DIR) / "tmp_data" / f"pool_base_{pool_key}.parquet"

    def _build_factors_cache_path(
        self,
        pool_cache_path: Path,
        exprs: Optional[List],
        cache: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        """根据 pool_cache_path、exprs 和 cache 策略合成 factors 缓存路径。

        规则:
            - exprs 为空时，始终返回 None。
            - cache is None:      返回 None（不缓存 factors）。
            - cache == "md5":     以 pool_cache_path 路径字符串 + sorted exprs
                                  联合哈希，存入默认 tmp_data 目录。
            - 其他（显式路径）:   直接使用该路径（相对路径转绝对路径）。
        """
        if not exprs or cache is None:
            return None

        if cache == "md5":
            normalized_exprs = self._normalize_exprs(exprs)
            factors_key = hashlib.md5(
                f"{pool_cache_path}_{'|'.join(normalized_exprs)}".encode("utf-8")
            ).hexdigest()
            return (
                Path(settings.OUTPUT_DIR)
                / "tmp_data"
                / f"factor_data_{factors_key}.parquet"
            )

        return Path(cache).resolve()

    def _normalize_exprs(self, exprs: Optional[List]) -> List[str]:
        """规范化表达式列表（去空白、去空值、去重、排序）。"""
        if not exprs:
            return []
        normalized = [str(expr).strip() for expr in exprs if str(expr).strip()]
        return sorted(set(normalized))

    def _load_cached_lazyframe(
        self, cache_path: Optional[Path]
    ) -> Optional[pl.LazyFrame]:
        """尝试加载缓存文件并返回 LazyFrame。

        返回:
            - 命中缓存: LazyFrame（数值列统一 cast 到 Float64）
            - 未命中: None
        """
        if not cache_path:
            return None
        if not cache_path.exists():
            return None

        logger.info(f"✨ 发现缓存，直接加载: {cache_path}")
        lf = pl.scan_parquet(cache_path)
        return self._cast_numeric_float64(lf)

    def _persist_cache_and_reload(
        self, lf: pl.LazyFrame, cache_path: Path
    ) -> pl.LazyFrame:
        """持久化 LazyFrame 到 Parquet 并回读为 LazyFrame。

        说明:
            - 写入前统一数值列为 Float64，规避类型混合带来的执行异常。
            - 使用 `collect(no_optimization=True)` 兼容历史 Polars 场景。
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📥 正在将计算结果写入缓存: {cache_path}")

        lf = self._cast_numeric_float64(lf)
        df = lf.collect(no_optimization=True)
        df.write_parquet(cache_path, compression="zstd")
        return pl.scan_parquet(cache_path)

    def _build_factors_view(
        self,
        pool: PoolUniverse,
        base_lf: pl.LazyFrame,
        exprs: Optional[List],
        final_cache_path: Optional[Path] = None,
    ) -> pl.LazyFrame:
        """在基础层数据上生成表达式列，并按需缓存最终结果。

        缓存命中时直接返回缓存数据，跳过计算。
        """
        if cached_lf := self._load_cached_lazyframe(final_cache_path):
            return cached_lf

        if not exprs:
            return base_lf
        select_cols: List[str] = pool.needed_cols()
        lf, generated_expr_cols = self._apply_column_exprs(base_lf, exprs)
        lf = self._finalize_projection(lf, select_cols, generated_expr_cols)

        # lf = pool.preprocessor(lf, generated_expr_cols)

        if final_cache_path:
            return self._persist_cache_and_reload(lf, final_cache_path)

        return self._cast_numeric_float64(lf)

    def _cast_numeric_float64(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """统一将数值列转换为 Float64，减少后续类型不一致问题。"""
        return lf.with_columns(cs.numeric().cast(pl.Float64))

    def _apply_column_exprs(
        self,
        lf: pl.LazyFrame,
        column_exprs: Optional[List],
        codegen_over_null: Literal["partition_by", "order_by", None] = None,
    ) -> tuple[pl.LazyFrame, List[str]]:
        """批量执行表达式并返回生成后的列名列表。

        返回:
            (new_lf, generated_expr_cols)
        """
        generated_expr_cols: List[str] = []
        if not column_exprs:
            return lf, generated_expr_cols

        normalized_exprs = [
            str(expr).strip() for expr in column_exprs if str(expr).strip()
        ]
        for expr_str in normalized_exprs:
            if "=" in expr_str:
                generated_expr_cols.append(expr_str.split("=")[0].strip())

        template_path = settings.template_path_str
        batch_size = getattr(settings, "CODEGEN_BATCH_SIZE", 200)
        for i in range(0, len(normalized_exprs), batch_size):
            batch = normalized_exprs[i : i + batch_size]
            lf = codegen_exec(
                lf,
                *batch,
                over_null=codegen_over_null,
                template_file=template_path,
                date="DATE",
                asset="ASSET",
            )

        return lf, generated_expr_cols

    # --- 内部核心组件 ---

    def _scan_with_lookback(
        self, start_date: date, end_date: date, lookback: int
    ) -> pl.LazyFrame:
        """按年份扫描因子库，并基于 lookback 预热历史窗口。

        参数:
            start_date: 起始日期（date 对象）。
            end_date: 结束日期（date 对象）。
            lookback: 预热窗口（交易日近似转换为自然日）。
        """
        # 预估预热所需的起始日期（交易日天数 * 1.5 倍近似自然日）
        effective_start = start_date - timedelta(days=int(lookback * 1.5) + 7)

        scans = []
        for year in range(effective_start.year, end_date.year + 1):
            file_path = self.factor_dir / f"{year}.parquet"
            if file_path.exists():
                # ✅ ASSET 保存为 String，concat 后统一转为 Categorical
                scans.append(pl.scan_parquet(file_path))

        if not scans:
            raise FileNotFoundError(
                f"数据区间 {start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')} 无可用文件"
            )

        # concat 后统一 cast 为当前 session 的 Categorical，与 _static_props join 保持一致
        return (
            pl.concat(scans)
            .with_columns(
                pl.col(F.ASSET).cast(self.asset_manager.stock_type, strict=False)
            )
            .filter(pl.col(F.ASSET).is_not_null())
            .filter(pl.col("DATE") <= end_date)
        )

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
        """动态感知列空间并执行投影下压。

        说明:
            - 自动保留 `DATE`、`ASSET`。
            - 自动忽略不存在列，避免 select 抛错。
        """
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
