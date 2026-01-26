import polars as pl
from alpha.utils.config import settings

# 1. 路径配置
parquet_path = settings.WAREHOUSE_DIR / "unified_factors" / "2019.parquet"

# 2. 读取 Shape
if parquet_path.exists():
    # 使用 scan_parquet 不会把文件载入内存，仅读取元数据，速度极快
    shape = pl.scan_parquet(parquet_path).collect().shape

    print(f"📊 宽表规模详情 (2019.parquet):")
    print(f" - 总行数: {shape[0]:,}")
    print(f" - 总列数: {shape[1]}")

    # 算一下平均每天有多少只股票
    # 2019年1月有22个交易日
    avg_stocks = shape[0] // 22
    print(f" - 1月平均每日标数: ~{avg_stocks}")
else:
    print("❌ 2019.parquet 文件不存在。")