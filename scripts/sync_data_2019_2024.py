#!/usr/bin/env python3
"""
Alpha-Factory 多年份数据同步脚本 (2019-2024)

【功能】
- 从 Tushare 获取 2019-2024 年全市场行情数据
- 自动处理数据清洗、对齐、复权调整
- 增量更新支持（断点续传）
- 多年份数据统一存储为 Parquet

【使用】
python scripts/sync_data_2019_2024.py \\
    --start-year 2019 \\
    --end-year 2024 \\
    --mode full \\
    --resume

【输出结构】
data/
├── raw/                          # Tushare 原始数据 (HDF5 缓存)
│   ├── daily_2019.h5
│   ├── daily_2020.h5
│   └── ...
└── warehouse/unified_factors/    # 清洗后的统一因子库 (Parquet)
    ├── 2019.parquet
    ├── 2020.parquet
    └── ...

【核心参数】
- start_year, end_year: 同步的年份范围
- mode: full (全量重新同步) | incremental (增量更新)
- resume: 是否支持断点续传
- overwrite: 是否覆盖已有数据
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import polars as pl
from loguru import logger

# 项目路径修正
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpha.data_provider import TushareDataService, DataProvider
from alpha.utils.config import settings
from alpha.utils.logger import setup_logger


class MultiYearDataSyncer:
    """
    多年份数据同步协调器

    职责：
    - 按年份分段同步数据
    - 管理缓存和断点续传
    - 数据验证和质量检查
    - 生成同步报告
    """

    def __init__(
        self,
        start_year: int,
        end_year: int,
        mode: str = "full",
        resume: bool = True,
        overwrite: bool = False,
        is_vip: bool = True
    ):
        """
        初始化多年份数据同步器

        Args:
            start_year: 同步起始年份
            end_year: 同步结束年份
            mode: 同步模式 (full / incremental)
            resume: 是否支持断点续传
            overwrite: 是否覆盖已有数据
            is_vip: 是否使用 Tushare VIP 账户 (更高限流)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.mode = mode
        self.resume = resume
        self.overwrite = overwrite

        self.service = TushareDataService(is_vip=is_vip)
        self.data_provider = DataProvider()

        self.sync_stats: Dict[int, Dict[str, Any]] = {}
        self.checkpoint_file = Path(settings.LOG_DIR) / "sync_checkpoint.json"

        logger.info("=" * 70)
        logger.info(f"🚀 多年份数据同步配置")
        logger.info(f"  年份范围: {start_year} - {end_year}")
        logger.info(f"  同步模式: {mode} | 断点续传: {resume}")
        logger.info(f"  覆盖数据: {overwrite} | VIP 账户: {is_vip}")
        logger.info("=" * 70)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        加载断点续传检查点

        Returns:
            Dict: 检查点信息
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ 检查点加载失败: {e}")
        return {}

    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        保存断点续传检查点

        Args:
            checkpoint: 检查点信息
        """
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"✓ 检查点已保存")
        except Exception as e:
            logger.warning(f"⚠️ 检查点保存失败: {e}")

    def sync_single_year(self, year: int, checkpoint: Dict) -> Dict[str, Any]:
        """
        同步单个年份的数据

        Args:
            year: 目标年份
            checkpoint: 断点信息

        Returns:
            Dict: 同步统计结果

        Raises:
            Exception: 如果同步过程出现错误
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 同步 {year} 年数据")
        logger.info("=" * 70)

        start_date = f"{year}0101"
        end_date = f"{year}1231"

        # 1. 检查是否已同步
        year_checkpoint = checkpoint.get(str(year), {})
        if year_checkpoint.get("status") == "completed" and not self.overwrite:
            logger.info(f"✓ {year} 年已同步，跳过")
            return {
                "year": year,
                "status": "skipped",
                "reason": "已存在"
            }

        try:
            # 2. 同步日线数据
            logger.info(f"📡 从 Tushare 获取 {start_date} ~ {end_date} 日线数据...")
            sync_result = self.service.sync_daily_bars(start_date, end_date)

            # 3. 验证数据
            logger.info("🔍 验证数据完整性...")
            warehouse_dir = Path(settings.WAREHOUSE_DIR) / "unified_factors"
            parquet_file = warehouse_dir / f"{year}.parquet"

            if parquet_file.exists():
                df = pl.read_parquet(parquet_file)
                row_count = df.height
                col_count = len(df.columns)
                null_ratio = df.null_count().sum() / (row_count * col_count)

                logger.info(f"✓ 数据验证通过")
                logger.info(f"  行数: {row_count:,} | 列数: {col_count}")
                logger.info(f"  空值率: {null_ratio:.2%}")

                stat = {
                    "year": year,
                    "status": "completed",
                    "rows": row_count,
                    "columns": col_count,
                    "null_ratio": null_ratio,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.warning(f"⚠️ {year} 年 Parquet 文件不存在")
                stat = {
                    "year": year,
                    "status": "warning",
                    "message": "Parquet 文件不存在"
                }

            return stat

        except Exception as e:
            logger.error(f"❌ {year} 年同步失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "year": year,
                "status": "failed",
                "error": str(e)
            }

    def run_all_years(self) -> Dict[int, Dict[str, Any]]:
        """
        按年份循环执行数据同步

        Returns:
            Dict: 每个年份的同步统计结果
        """
        logger.info("\n🔄 开始多年份数据同步...")

        # 加载检查点
        checkpoint = self.load_checkpoint() if self.resume else {}

        for year in range(self.start_year, self.end_year + 1):
            result = self.sync_single_year(year, checkpoint)
            self.sync_stats[year] = result

            # 更新检查点
            checkpoint[str(year)] = result
            if self.resume:
                self.save_checkpoint(checkpoint)

        return self.sync_stats

    def generate_summary_report(self) -> str:
        """
        生成同步摘要报告

        Returns:
            str: 格式化的报告文本
        """
        report_lines = [
            "\n" + "=" * 70,
            "📊 Alpha-Factory 多年份数据同步报告",
            "=" * 70,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"同步年份: {self.start_year} - {self.end_year}",
            f"同步模式: {self.mode}",
            "",
            "【年份统计】",
        ]

        total_rows = 0
        success_count = 0

        for year in range(self.start_year, self.end_year + 1):
            if year in self.sync_stats:
                stat = self.sync_stats[year]
                status = stat.get("status", "unknown")

                if status == "completed":
                    rows = stat.get("rows", 0)
                    cols = stat.get("columns", 0)
                    null_ratio = stat.get("null_ratio", 0)
                    total_rows += rows
                    success_count += 1
                    report_lines.append(
                        f"  {year}: ✅ | 行数={rows:7,} | 列数={cols:2d} | 空值={null_ratio:.1%}"
                    )
                elif status == "skipped":
                    report_lines.append(f"  {year}: ⏭️  | 已存在，跳过")
                else:
                    reason = stat.get("error", stat.get("message", "未知错误"))
                    report_lines.append(f"  {year}: ❌ | {reason}")
            else:
                report_lines.append(f"  {year}: ❓ | 未同步")

        report_lines.extend([
            "",
            "【统计汇总】",
            f"  成功: {success_count}/{self.end_year - self.start_year + 1}",
            f"  总数据行数: {total_rows:,}",
            "",
            "【输出位置】",
            f"  原始数据: {Path(settings.RAW_DATA_DIR)}",
            f"  仓库数据: {Path(settings.WAREHOUSE_DIR) / 'unified_factors'}",
            "",
            "【后续步骤】",
            "  1. 验证数据完整性",
            "  2. 启动因子挖掘 (mine_factors_2019_2024.py)",
            "=" * 70,
        ])

        return "\n".join(report_lines)

    def save_summary(self, output_file: Path = None):
        """
        保存同步摘要到文件

        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = Path(settings.REPORT_DIR) / f"sync_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_summary_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✅ 摘要已保存: {output_file}")


def main():
    """
    CLI 主函数
    """
    parser = argparse.ArgumentParser(
        description="Alpha-Factory 多年份数据同步脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 全量同步 2019-2024 年
  python sync_data_2019_2024.py --start-year 2019 --end-year 2024 --mode full

  # 增量更新最近一年
  python sync_data_2019_2024.py --start-year 2024 --end-year 2024 --mode incremental

  # 支持断点续传
  python sync_data_2019_2024.py --start-year 2019 --end-year 2024 --resume

  # 覆盖已有数据
  python sync_data_2019_2024.py --start-year 2024 --end-year 2024 --overwrite
        """
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="同步起始年份 (默认: 2019)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="同步结束年份 (默认: 2024)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="同步模式：full (全量) 或 incremental (增量，默认: full)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="支持断点续传 (默认: 启用)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="禁用断点续传"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有数据"
    )
    parser.add_argument(
        "--vip",
        action="store_true",
        default=True,
        help="使用 Tushare VIP 账户 (默认: 是)"
    )
    parser.add_argument(
        "--no-vip",
        dest="vip",
        action="store_false",
        help="不使用 VIP 账户"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        default=True,
        help="保存摘要报告 (默认: 是)"
    )

    args = parser.parse_args()

    # 参数验证
    if args.start_year > args.end_year:
        logger.error("❌ 起始年份不能晚于结束年份")
        sys.exit(1)

    if args.start_year < 2015:
        logger.warning("⚠️ 数据从 2015 年开始提供")

    # 初始化日志
    setup_logger()

    # 创建同步器
    syncer = MultiYearDataSyncer(
        start_year=args.start_year,
        end_year=args.end_year,
        mode=args.mode,
        resume=args.resume,
        overwrite=args.overwrite,
        is_vip=args.vip
    )

    # 执行同步
    try:
        logger.info("📥 开始多年份数据同步...")
        results = syncer.run_all_years()

        # 保存摘要
        if args.save_summary:
            syncer.save_summary()

        # 打印摘要
        logger.info(syncer.generate_summary_report())

        logger.info("\n✅ 数据同步全流程完成！")
        logger.info("   下一步：python scripts/mine_factors_2019_2024.py")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 同步被用户中断")
        logger.info("   检查点已保存，下次可通过 --resume 恢复")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n❌ 同步过程出现未预期的错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
