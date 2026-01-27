import polars as pl
import polars.selectors as cs
from pathlib import Path
from loguru import logger
from typing import Optional, List
from datetime import date, datetime

from alpha.utils.config import settings


class DataProvider:
    """
    数据读取接口 (L4 层)

    职责:
    - 统一入口: 对外隐藏按年存储的物理细节，支持跨年数据无缝拼接。
    - 性能优化: 深度集成 Polars 的延迟加载机制，实现磁盘到内存的最小化传输。
    - 资产筛选: 支持在加载阶段通过 is_in 算子直接下压过滤资产池。
    """

    def __init__(self):
        self.warehouse_dir = Path(settings.WAREHOUSE_DIR)
        self.factor_dir = self.warehouse_dir / "unified_factors"
        # 确保目录存在
        self.factor_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"✓ DataProvider 初始化完成 | 仓库路径: {self.factor_dir}")


    def _prepare_labeled_data(self) -> pl.DataFrame:
        """
        从 DataProvider 获取数据并计算挖掘标签
        计算逻辑：未来 N 日的 Open-to-Open 收益率
        """

        cache_file = self.save_dir / f"labeled_{self.label_y}.parquet"

        data_provider = DataProvider()

        if not self.overwrite and cache_file.exists():
            logger.info(f"📂 发现标签数据缓存，直接加载: {cache_file}")
            return pl.read_parquet(cache_file)

        logger.info(f"📡 正在计算标签 '{self.label_y}'...")


        # 2. 载入原始数据
        # 挖掘因子通常需要 OHLCV，计算 OO 收益率需要 OPEN
        lf = data_provider.load_data(
            start_date= self.start_date,
            end_date=self.end_date,
            columns=["DATE", "ASSET", "OPEN", "CLOSE", "HIGH", "LOW", "VOLUME",'AMOUNT']
        )

        # 3. 计算未来收益率 (Label Generation)
        # 计算公式: (未来第 N+1 日开盘价 / 未来第 1 日开盘价) - 1
        # 注意：OO_1 实际上需要 shift(-2) 和 shift(-1)
        lf = lf.sort(["ASSET", "DATE"]).with_columns([
            (
                    (pl.col("OPEN").shift(-(self.label_window  + 1)).over("ASSET") /
                     pl.col("OPEN").shift(-1).over("ASSET")) - 1
            ).alias(self.label_y)
        ])

        # 4. 截面中性化与去极值 (Z-Score)
        # 这一步将收益率转化为“该股票在当日全市场中的相对强度”
        date_col = "DATE"
        lf = lf.filter(pl.col(self.label_y).is_not_null()).with_columns([
            (
                    (pl.col(self.label_y) - pl.col(self.label_y).mean().over(date_col))
                    / (pl.col(self.label_y).std().over(date_col) + 1e-6)
            )
            .clip(-3, 3)  # 限制在正负 3 个标准差内，消除妖股噪声
            .fill_nan(0.0)
            .alias(self.label_y)
        ])
        lf = lf.sort(['ASSET', 'DATE']).with_columns([
            pl.col("ASSET").set_sorted(),
            # 强制将所有数值列转为 Float64，避免 GP 运行时 SchemaError
            cs.numeric().cast(pl.Float64)
        ])
        # 5. 执行计算并持久化
        df = lf.collect()

        if df.height == 0:
            raise ValueError("计算后数据为空，请检查 DataProvider 返回结果或日期范围")

        df.write_parquet(cache_file, compression="snappy")
        logger.info(f"💾 标签数据已就绪 | 行数: {df.height:,} | 均值: {df[self.label_y].mean():.4f}")

        return df


    def load_data(
            self,
            start_date: str,
            end_date: str,
            assets: Optional[List[str]] = None,
            columns: Optional[List[str]] = None,
            exclude_suspended: bool = False,
            exclude_st: bool = False,
    ) -> pl.LazyFrame:
        """
        加载统一因子库 (Lazy Mode)

        Args:
            start_date: 起始日期 'YYYYMMDD'
            end_date: 截止日期 'YYYYMMDD'
            assets: 资产列表 (ts_code 列表)，不传则加载全市场
            columns: 需要加载的特征列名，不传则加载所有列
            exclude_suspended: 是否过滤掉停牌日数据
            exclude_st: 是否过滤掉 ST 状态数据

        Returns:
            pl.LazyFrame: 包含计算图的延迟对象
        """
        try:
            s_dt = datetime.strptime(start_date, "%Y%m%d").date()
            e_dt = datetime.strptime(end_date, "%Y%m%d").date()
        except ValueError as e:
            raise ValueError(f"❌ 日期格式错误 (需 YYYYMMDD): {e}")

        # 1. 动态路由年度 Parquet 文件
        scans = []
        for year in range(s_dt.year, e_dt.year + 1):
            file_path = self.factor_dir / f"{year}.parquet"
            if file_path.exists():
                # 💡 scan_parquet 会自动进行 Row Group 级别的谓词下压优化
                scans.append(pl.scan_parquet(file_path))
            else:
                logger.warning(f"⚠️ 因子库缺少年度数据: {year}")

        if not scans:
            raise FileNotFoundError(f"❌ 在 {self.factor_dir} 中未找到 [{start_date} ~ {end_date}] 范围内的任何数据")

        # 2. 垂直拼接年度分片 (Lazy 级别)
        lf = pl.concat(scans)

        # 3. 谓词下压优化 (Predicate Pushdown)
        # 过滤日期范围
        lf = lf.filter(pl.col("DATE").is_between(s_dt, e_dt))

        # 过滤资产池
        if assets:
            lf = lf.filter(pl.col("ASSET").is_in(assets))

        # 状态过滤
        if exclude_suspended:
            lf = lf.filter(pl.col("IS_SUSPENDED") is False)

        if exclude_st:
            lf = lf.filter(pl.col("IS_ST") is False)

        # 4. 列投影优化 (Projection Pushdown)
        if columns:
            # 自动保留主键列（去重处理）
            final_cols = list(dict.fromkeys(["DATE", "ASSET"] + columns))
            lf = lf.select(final_cols)

        return lf

    def get_available_dates(self) -> List[date]:
        """获取仓库中已有的所有交易日清单"""
        try:
            # 仅扫描 DATE 列，且利用通配符扫描全库，极其高效
            return (
                pl.scan_parquet(self.factor_dir / "*.parquet")
                .select("DATE")
                .collect()
                .unique()
                .sort("DATE")
                .get_column("DATE")
                .to_list()
            )
        except Exception as e:
            logger.error(f"获取可用日期清单失败: {e}")
            return []

    def validate_schema(self, lf: pl.LazyFrame) -> bool:
        """
        验证 LazyFrame 的 Schema 是否符合 L2 标准契约。
        无需真正 collect() 数据，仅在元数据层面进行静态检查。
        """
        expected = {
            "DATE": pl.Date,
            "ASSET": [pl.Categorical, pl.Enum, pl.String],
            "CLOSE": [pl.Float32, pl.Float64],
            "IS_SUSPENDED": pl.Boolean,
            "VOLUME": [pl.Float32, pl.Float64, pl.Int64],
        }

        actual_schema = lf.schema

        for col, expected_types in expected.items():
            if col not in actual_schema:
                logger.error(f"❌ Schema 验证失败: 缺失关键列 '{col}'")
                return False

            actual_type = actual_schema[col]
            # 支持多种兼容类型（如 Enum 和 String 在查询端通用）
            if isinstance(expected_types, list):
                if not any(actual_type == t for t in expected_types):
                    logger.error(f"❌ 类型不匹配: '{col}' 实际为 {actual_type}, 期望 {expected_types}")
                    return False
            else:
                if actual_type != expected_types:
                    logger.error(f"❌ 类型不匹配: '{col}' 实际为 {actual_type}, 期望 {expected_types}")
                    return False

        logger.info("✅ 因子库 Schema 契约验证通过")
        return True

    def get_data_summary(self, lf: pl.LazyFrame) -> None:
        """打印数据摘要（会触发一次轻量计算）"""
        summary = lf.select([
            pl.col("DATE").min().alias("start"),
            pl.col("DATE").max().alias("end"),
            pl.col("ASSET").n_unique().alias("assets_count"),
            pl.len().alias("total_rows")
        ]).collect()

        logger.info(f"📊 数据载入成功: {summary['start'][0]} ~ {summary['end'][0]} | "
                    f"标的数量: {summary['assets_count'][0]} | 总行数: {summary['total_rows'][0]}")
