#!/usr/bin/env python3
"""
快速测试脚本 - 验证标签列计算功能

测试流程：
1. 检查 2024 年数据是否包含 RETURN_OO_1 列
2. 如果不包含，自动计算
3. 验证计算结果
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from alpha.utils.config import settings
from alpha.utils.logger import setup_logger
from loguru import logger

setup_logger()

def test_label_column():
    """测试标签列计算"""

    logger.info("=" * 70)
    logger.info("📋 测试标签列计算功能")
    logger.info("=" * 70)

    # 1. 加载 2024 年数据
    warehouse_dir = Path(settings.WAREHOUSE_DIR) / "unified_factors"
    parquet_file = warehouse_dir / "2024.parquet"

    if not parquet_file.exists():
        logger.error(f"❌ 数据文件不存在: {parquet_file}")
        return False

    logger.info(f"✓ 数据文件存在: {parquet_file}")

    # 2. 读取数据
    df = pl.read_parquet(parquet_file)
    logger.info(f"✓ 数据加载成功: {df.shape[0]:,} 行 × {len(df.columns)} 列")

    # 3. 检查所有列
    logger.info(f"📝 当前列表: {df.columns}")

    # 4. 检查 RETURN_OO_1 是否存在
    if "RETURN_OO_1" in df.columns:
        logger.info("✅ RETURN_OO_1 列已存在")
        return True

    # 5. 如果不存在，计算
    logger.info("⚠️ RETURN_OO_1 列不存在，尝试计算...")

    if "OPEN" not in df.columns:
        logger.error("❌ 缺少 OPEN 列，无法计算 RETURN_OO_1")
        return False

    logger.info("📊 计算 RETURN_OO_1 = (next_OPEN - OPEN) / OPEN")

    try:
        df_with_label = df.with_columns([
            (
                (pl.col("OPEN").shift(-1).over("ASSET") - pl.col("OPEN"))
                / pl.col("OPEN")
            ).alias("RETURN_OO_1")
        ])

        logger.info(f"✓ 计算成功")

        # 6. 验证计算结果
        label_col = df_with_label["RETURN_OO_1"]
        null_count = label_col.null_count()
        non_null_count = label_col.height - null_count

        logger.info(f"✓ 标签列统计:")
        logger.info(f"  - 总数: {label_col.height:,}")
        logger.info(f"  - 非空: {non_null_count:,}")
        logger.info(f"  - 空值: {null_count:,}")
        logger.info(f"  - 范围: [{label_col.min():.4f}, {label_col.max():.4f}]")
        logger.info(f"  - 均值: {label_col.mean():.6f}")

        # 7. 保存回文件
        df_with_label.write_parquet(parquet_file, compression="snappy")
        logger.info(f"✅ 标签列已保存: {parquet_file}")

        return True

    except Exception as e:
        logger.error(f"❌ 计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_label_column()
    logger.info("=" * 70)
    if success:
        logger.info("✅ 测试通过！可以继续运行因子挖掘脚本")
        sys.exit(0)
    else:
        logger.error("❌ 测试失败")
        sys.exit(1)
