import polars as pl
from alpha.utils.config import settings
from pathlib import Path

# 1. 路径配置
parquet_path = settings.WAREHOUSE_DIR / "unified_factors" / "2019.parquet"
output_path = settings.WAREHOUSE_DIR / "unified_factors" / "check_head_50.csv"

# 2. 执行提取并保存
if parquet_path.exists():
    # 使用 scan_parquet 极速读取头部
    df_50 = pl.scan_parquet(parquet_path).head(50).collect()

    # 导出为 CSV
    df_50.write_csv(output_path)

    print(f"✅ 已成功提取前 50 行数据至: {output_path}")
    print(f"📊 数据包含 {len(df_50.columns)} 个因子列。")
else:
    print(f"❌ 未找到文件: {parquet_path}，请确认 2019 年度任务已运行成功。")