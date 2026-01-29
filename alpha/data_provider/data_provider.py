import polars as pl
from pathlib import Path

from loguru import logger
from typing import Optional, List, Union, Callable
from datetime import date, datetime, timedelta

from expr_codegen import codegen_exec
from alpha.data_provider.stock_assets_manager import StockAssetsManager
from alpha.utils.config import settings


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

        # 预加载静态元数据 LazyFrame
        # 提示：确保 asset 列在管理器中已设为 Categorical 或 Enum
        self._static_props = self.asset_manager.get_properties().lazy()
        logger.debug("✓ DataProvider (Enhanced) 初始化完成")

    def load_data(
            self,
            start_date: str,
            end_date: str,
            column_blocks: Optional[List] = None,
            column_exprs: Optional[List[str]] = None,
            funcs: Optional[List[Callable[[pl.LazyFrame], pl.LazyFrame]]] = None,
            lookback_window: int = 0,
            optimize_memory: bool = True,
            cache_path: Optional[Union[str, Path]] = None,  # 🆕 新增缓存路径参数
    ) -> pl.LazyFrame:
        """
        统一数据集构建管线（带持久化缓存支持）
        """

        # 1. 🔍 检查缓存命中
        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info(f"✨ 发现缓存，直接加载: {cache_path}")
                # 使用 scan_parquet 保持 Lazy 特性
                return pl.scan_parquet(cache_path)

        # 2. 🏗️ 执行完整计算流水线 (如果缓存未命中或未设置)
        logger.info(f"⚙️ 缓存未命中或未设置，开始计算数据 [{start_date} -> {end_date}]...")

        # A. 物理层扫描
        lf = self._scan_with_lookback(start_date, end_date, lookback_window)

        # B. 基础上下文增强
        lf = self._enrich_context(lf)

        # C. 列生成：func block 型，expr_codegen 支持批量处理
        if column_blocks:
            lf = codegen_exec(lf, *column_blocks, style='polars', over_null=None, date='DATE', asset='ASSET')

        # D. 列生成：表达式型
        generated_expr_cols = []
        if column_exprs:
            for expr_str in column_exprs:
                if "=" in expr_str:
                    generated_expr_cols.append(expr_str.split("=")[0].strip())

            batch_size = settings.get("CODEGEN_BATCH_SIZE", 100)
            for i in range(0, len(column_exprs), batch_size):
                batch = column_exprs[i: i + batch_size]
                lf = codegen_exec(lf, *batch, style='polars', over_null=None, date='DATE', asset='ASSET')

        # E. 函数型，这里既可以生成新列，也可以用来过滤行
        if funcs:
            for func in funcs:
                lf = func(lf)

        # F. 时间切片 & 行过滤
        s_dt = datetime.strptime(start_date, "%Y%m%d").date()
        lf = lf.filter(pl.col("DATE") >= s_dt)


        # G. 投影与类型压缩
        # lf = self._finalize_projection(lf, base_columns, generated_expr_cols)
        if optimize_memory:
            lf = lf.with_columns(pl.all().shrink_dtype())

        # 3. 💾 持久化缓存 (如果指定了 cache_path)
        if cache_path:
            # 注意：LazyFrame 必须 collect 之后才能 write_parquet
            # 或者使用 sink_parquet (如果是流式支持的操作)
            # 为了稳健性，这里先 collect
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📥 正在将计算结果写入缓存: {cache_path}")

            # 执行计算并保存
            df = lf.collect()
            df.write_parquet(cache_path)

            # 返回保存后的 Lazy 视图，确保后续链路统一
            return pl.scan_parquet(cache_path)

        return lf

    # --- 内部核心组件 ---

    def _scan_with_lookback(self, start_date: str, end_date: str, lookback: int) -> pl.LazyFrame:
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
        """注入物理环境列及生存者偏差修正"""
        return (
            lf.join(self._static_props, left_on="ASSET", right_on="asset", how="left")
            .with_columns([
                # 计算上市天数
                (pl.col("DATE").cast(pl.Date) - pl.col("list_date")).dt.total_days().fill_null(0).alias("LIST_DAYS"),
                # 识别基础交易限制
                (pl.col("CLOSE") >= pl.col("UP_LIMIT") - 0.001).alias("is_up_limit"),
                (pl.col("CLOSE") <= pl.col("DOWN_LIMIT") + 0.001).alias("is_down_limit"),
                # 计算截面市值百分位
                (pl.col("TOTAL_MV").rank().over("DATE") / pl.col("ASSET").count().over("DATE")).alias("mv_pct")
            ])
            .with_columns([
                # 预定义可交易池：排除 ST、停牌、新股、退市期、封板
                (
                        (pl.col("IS_ST") is False) &
                        (pl.col("IS_SUSPENDED") is False) &
                        (pl.col("LIST_DAYS") >= 242) &
                        (pl.col("is_up_limit") is False) &
                        (pl.col("is_down_limit") is False) &
                        (pl.col("DATE").cast(pl.Date) < pl.col("delist_date").fill_null(date(2099, 12, 31)))
                ).alias("POOL_TRADABLE")
            ])
        )

    def _finalize_projection(self, lf: pl.LazyFrame, base_cols: List[str], generated_cols: List[str]) -> pl.LazyFrame:
        """动态感知列空间并执行投影下压"""
        # 默认始终保留的 ID 和状态列
        essential = ["DATE", "ASSET", "POOL_TRADABLE", "LIST_DAYS"]

        # 汇总所有请求的列
        requested = essential + (base_cols or []) + generated_cols
        requested = list(dict.fromkeys(requested))

        # 动态获取当前 LazyFrame 的 Schema，防止 select 不存在的列
        current_schema = lf.collect_schema()
        final_selection = [c for c in requested if c in current_schema]

        return lf.select(final_selection)
