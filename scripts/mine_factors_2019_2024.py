#!/usr/bin/env python3
"""
Alpha-Factory 因子挖掘脚本 (2019-2024)

【功能】
- 支持跨年份多轮因子挖掘
- 每个年份独立进化周期，避免数据混淆
- 自动管理缓存和中间结果
- 支持断点恢复

【使用】
python scripts/mine_factors_2019_2024.py \\
    --start-year 2019 \\
    --end-year 2024 \\
    --n-gen 20 \\
    --n-pop 500 \\
    --label RETURN_OO_1 \\
    --overwrite-data

【输出结构】
output/gp/
├── 2019_exprs_*.pkl          # 每代种群
├── 2019_best_hof.pkl          # 最佳因子名人堂
├── 2020_exprs_*.pkl
├── ...
└── 2024_best_hof.pkl

【核心参数】
- start_year, end_year: 挖掘的年份范围
- n_gen: 每个年份的进化代数 (推荐 15-30)
- n_pop: 初始种群大小 (推荐 300-500)
- label: 目标标签列名 (RETURN_OO_1, target_1d_return 等)
- overwrite_data: 是否覆盖缓存数据
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle

import polars as pl
from loguru import logger

# 项目路径修正
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpha.data_provider import DataProvider
from alpha.gp.generator import GPDeapGenerator
from alpha.utils.config import settings
from alpha.utils.logger import setup_logger


class MultiYearMiner:
    """
    多年份因子挖掘协调器

    职责：
    - 按年份分段组织数据
    - 为每个年份创建独立的 GP 生成器
    - 管理进化过程和结果汇总
    - 生成进化报告
    """

    def __init__(
        self,
        start_year: int,
        end_year: int,
        label_y: str = "RETURN_OO_1",
        n_gen: int = 20,
        n_pop: int = 500,
        batch_size: int = 50,
        overwrite_data: bool = False
    ):
        """
        初始化多年份挖掘器

        Args:
            start_year: 起始年份 (2019)
            end_year: 结束年份 (2024)
            label_y: 目标标签列名
            n_gen: 每个年份的进化代数
            n_pop: 初始种群大小
            batch_size: 批处理大小
            overwrite_data: 是否覆盖缓存数据
        """
        self.start_year = start_year
        self.end_year = end_year
        self.label_y = label_y
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.batch_size = batch_size
        self.overwrite_data = overwrite_data

        self.data_provider = DataProvider()
        self.results_by_year: Dict[int, Dict[str, Any]] = {}

        logger.info("=" * 70)
        logger.info(f"🚀 多年份因子挖掘配置")
        logger.info(f"  年份范围: {start_year} - {end_year}")
        logger.info(f"  目标标签: {label_y}")
        logger.info(f"  进化代数: {n_gen} | 种群大小: {n_pop}")
        logger.info(f"  批处理: {batch_size} | 覆盖数据: {overwrite_data}")
        logger.info("=" * 70)

    def _ensure_label_column(self, year: int) -> bool:
        """
        确保数据中存在目标标签列，如不存在则计算

        Args:
            year: 目标年份

        Returns:
            bool: 成功返回 True

        Raises:
            ValueError: 如果无法计算标签列
        """
        logger.info(f"🔍 检查标签列 '{self.label_y}' 是否存在...")

        # 读取该年份数据
        warehouse_dir = Path(settings.WAREHOUSE_DIR) / "unified_factors"
        parquet_file = warehouse_dir / f"{year}.parquet"

        if not parquet_file.exists():
            logger.warning(f"⚠️ {year} 年数据不存在: {parquet_file}")
            return False

        # 读取数据
        df = pl.read_parquet(parquet_file)
        available_cols = df.columns

        logger.info(f"✓ 数据包含 {len(available_cols)} 列")

        # 检查标签列是否存在
        if self.label_y in available_cols:
            logger.info(f"✓ 标签列 '{self.label_y}' 已存在")
            return True

        # 如果不存在，尝试计算
        logger.info(f"⚠️ 标签列 '{self.label_y}' 不存在，尝试计算...")

        # RETURN_OO_1 = 开盘到开盘 1 天收益率
        if self.label_y == "RETURN_OO_1":
            if "OPEN" not in available_cols:
                raise ValueError(f"无法计算 {self.label_y}：缺少 OPEN 列")

            logger.info(f"📊 计算 {self.label_y} = (next_OPEN - OPEN) / OPEN")

            df_with_label = df.with_columns([
                (
                    (pl.col("OPEN").shift(-1).over("ASSET") - pl.col("OPEN"))
                    / pl.col("OPEN")
                ).alias(self.label_y)
            ])

            df_with_label.write_parquet(parquet_file, compression="snappy")
            logger.info(f"✅ 标签列已计算并保存")
            return True

        elif self.label_y == "target_1d_return":
            if "CLOSE" not in available_cols:
                raise ValueError(f"无法计算 {self.label_y}：缺少 CLOSE 列")

            logger.info(f"📊 计算 {self.label_y} = (next_CLOSE - CLOSE) / CLOSE")

            df_with_label = df.with_columns([
                (
                    (pl.col("CLOSE").shift(-1).over("ASSET") - pl.col("CLOSE"))
                    / pl.col("CLOSE")
                ).alias(self.label_y)
            ])

            df_with_label.write_parquet(parquet_file, compression="snappy")
            logger.info(f"✅ 标签列已计算并保存")
            return True

        elif self.label_y == "target_5d_return":
            if "CLOSE" not in available_cols:
                raise ValueError(f"无法计算 {self.label_y}：缺少 CLOSE 列")

            logger.info(f"📊 计算 {self.label_y} = 5 天后收益率")

            df_with_label = df.with_columns([
                (
                    (pl.col("CLOSE").shift(-5).over("ASSET") - pl.col("CLOSE"))
                    / pl.col("CLOSE")
                ).alias(self.label_y)
            ])

            df_with_label.write_parquet(parquet_file, compression="snappy")
            logger.info(f"✅ 标签列已计算并保存")
            return True

        else:
            raise ValueError(
                f"不支持的标签列: {self.label_y}\n"
                f"支持的选项: RETURN_OO_1, target_1d_return, target_5d_return"
            )

    def mine_single_year(self, year: int) -> Dict[str, Any]:
        """
        为单个年份执行因子挖掘

        Args:
            year: 目标年份

        Returns:
            Dict: 挖掘结果 (种群、进化日志、名人堂)

        Raises:
            ValueError: 如果数据无法加载或配置错误
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 开始挖掘 {year} 年数据")
        logger.info("=" * 70)

        # 1. 配置参数
        config = {
            "label_y": self.label_y,
            "split_date": datetime(year - 1, 12, 31),  # 前一年最后一天作为分割点
            "batch_size": self.batch_size,
            "mu": self.n_pop // 2,
            "lambda": self.n_pop // 2,
            "hof_size": max(100, self.n_pop // 5)
        }

        # 2. 创建 GP 生成器
        try:
            generator = GPDeapGenerator(config)
        except Exception as e:
            logger.error(f"❌ 创建 GP 生成器失败: {e}")
            raise

        # 3. 构建年份特定的输出目录
        year_output_dir = Path(settings.GP_DEAP_DIR) / str(year)
        year_output_dir.mkdir(parents=True, exist_ok=True)

        # 临时修改 generator 的输出目录
        generator.save_dir = year_output_dir
        generator.data_cache_dir = year_output_dir / "data_cache"
        generator.data_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ 输出目录: {year_output_dir}")

        # 4. 确保标签列存在
        try:
            self._ensure_label_column(year)
        except Exception as e:
            logger.error(f"❌ {year} 年标签列检查/计算失败: {e}")
            raise

        # 5. 执行全流程挖掘
        try:
            result = generator.run_workflow(
                data_provider=self.data_provider,
                n_gen=self.n_gen,
                overwrite_data=self.overwrite_data
            )
            pop, logbook, hof = result

            # 6. 保存结果
            result_dict = {
                "year": year,
                "population": pop,
                "logbook": logbook,
                "halloffame": hof,
                "config": config,
                "timestamp": datetime.now().isoformat()
            }

            # 保存到 pickle
            result_path = year_output_dir / f"{year}_result.pkl"
            try:
                with open(result_path, 'wb') as f:
                    pickle.dump(result_dict, f)
                logger.info(f"💾 挖掘结果已保存: {result_path}")
            except Exception as e:
                logger.warning(f"⚠️ 结果保存失败: {e}")

            logger.info(f"✅ {year} 年挖掘完成")
            logger.info(f"   最终种群: {len(pop)} | 名人堂: {len(hof)}")

            return result_dict

        except Exception as e:
            logger.error(f"❌ {year} 年挖掘失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def run_all_years(self) -> Dict[int, Dict[str, Any]]:
        """
        按年份循环执行因子挖掘

        Returns:
            Dict: 每个年份的挖掘结果汇总

        Raises:
            RuntimeError: 如果任何年份挖掘失败
        """
        logger.info("\n🔄 开始多年份因子挖掘循环...")

        failed_years = []
        for year in range(self.start_year, self.end_year + 1):
            try:
                result = self.mine_single_year(year)
                self.results_by_year[year] = result
            except Exception as e:
                logger.error(f"❌ {year} 年挖掘失败，跳过")
                failed_years.append(year)
                continue

        # 7. 总结报告
        logger.info("\n" + "=" * 70)
        logger.info("📈 多年份挖掘完成总结")
        logger.info("=" * 70)

        success_years = [y for y in range(self.start_year, self.end_year + 1) if y not in failed_years]
        logger.info(f"✅ 成功年份: {success_years}")
        if failed_years:
            logger.warning(f"⚠️ 失败年份: {failed_years}")

        for year in success_years:
            result = self.results_by_year[year]
            logger.info(f"  {year}: 种群={len(result['population'])} | 名人堂={len(result['halloffame'])}")

        return self.results_by_year

    def generate_summary_report(self) -> str:
        """
        生成进化摘要报告

        Returns:
            str: 格式化的报告文本
        """
        report_lines = [
            "\n" + "=" * 70,
            "📊 Alpha-Factory 多年份因子挖掘报告",
            "=" * 70,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"挖掘年份: {self.start_year} - {self.end_year}",
            f"目标标签: {self.label_y}",
            f"进化代数: {self.n_gen}",
            f"种群大小: {self.n_pop}",
            "",
            "【年份统计】",
        ]

        for year in range(self.start_year, self.end_year + 1):
            if year in self.results_by_year:
                result = self.results_by_year[year]
                pop_size = len(result['population'])
                hof_size = len(result['halloffame'])
                report_lines.append(
                    f"  {year}: ✅ 完成 | 种群={pop_size:3d} | 名人堂={hof_size:3d}"
                )
            else:
                report_lines.append(f"  {year}: ❌ 未完成")

        report_lines.extend([
            "",
            "【输出目录】",
            f"  {Path(settings.GP_DEAP_DIR)}",
            "",
            "【后续步骤】",
            "  1. 检查各年份的名人堂因子",
            "  2. 对最优因子进行效果评估",
            "  3. 进行多因子合成与回测",
            "=" * 70,
        ])

        return "\n".join(report_lines)

    def save_summary(self, output_file: Path = None):
        """
        保存挖掘摘要到文件

        Args:
            output_file: 输出文件路径，默认为 reports/mining_summary.txt
        """
        if output_file is None:
            output_file = Path(settings.REPORT_DIR) / f"mining_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
        description="Alpha-Factory 多年份因子挖掘脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 挖掘 2019-2024 年，20 代，500 种群
  python mine_factors_2019_2024.py --start-year 2019 --end-year 2024 --n-gen 20 --n-pop 500

  # 快速测试 (2023 年，5 代，100 种群)
  python mine_factors_2019_2024.py --start-year 2023 --end-year 2023 --n-gen 5 --n-pop 100

  # 覆盖缓存重新挖掘
  python mine_factors_2019_2024.py --start-year 2024 --end-year 2024 --n-gen 15 --overwrite-data
        """
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="挖掘起始年份 (默认: 2019)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="挖掘结束年份 (默认: 2024)"
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=20,
        help="每个年份的进化代数 (默认: 20)"
    )
    parser.add_argument(
        "--n-pop",
        type=int,
        default=500,
        help="初始种群大小 (默认: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="批处理大小 (默认: 50)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="RETURN_OO_1",
        help="目标标签列名 (默认: RETURN_OO_1)"
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="是否覆盖已有缓存数据"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        default=True,
        help="是否保存摘要报告 (默认: True)"
    )

    args = parser.parse_args()

    # 参数验证
    if args.start_year > args.end_year:
        logger.error("❌ 起始年份不能晚于结束年份")
        sys.exit(1)

    if args.start_year < 2015 or args.end_year > 2025:
        logger.warning(f"⚠️ 建议年份范围在 2015-2025 之间")

    if args.n_gen < 5:
        logger.warning("⚠️ 进化代数过少，可能难以收敛")

    if args.n_pop < 100:
        logger.warning("⚠️ 种群过小，可能难以探索")

    # 初始化日志
    setup_logger()

    # 创建挖掘器
    miner = MultiYearMiner(
        start_year=args.start_year,
        end_year=args.end_year,
        label_y=args.label,
        n_gen=args.n_gen,
        n_pop=args.n_pop,
        batch_size=args.batch_size,
        overwrite_data=args.overwrite_data
    )

    # 执行挖掘
    try:
        results = miner.run_all_years()

        # 保存摘要
        if args.save_summary:
            miner.save_summary()

        # 打印摘要
        logger.info(miner.generate_summary_report())

        logger.info("\n✅ 因子挖掘全流程完成！")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 挖掘被用户中断")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n❌ 挖掘过程出现未预期的错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
