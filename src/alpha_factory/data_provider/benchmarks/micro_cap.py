"""微盘股基准实现

时间序列：从 DataProvider 计算生成，T-1 市值排名 <= 400，T 日计算相关指标
字段：日期 + 等权平均收益率 + 多维统计指标（中位数、累计收益、平均市值、成交额中位数、换手率）
"""

from datetime import date, timedelta

import polars as pl
from loguru import logger

from alpha_factory.data_provider.benchmark import Benchmark
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.utils.schema import F


class MicroCapBenchmark(Benchmark):
    """微盘股基准 (Top 400 by Market Cap)

    定义：$T-1$ 日市值排名 <= 400 的主板+创业板股票
    计算：$T$ 日这些股票的多维统计指标

    数据源：DataProvider + MainSmallPool
    """

    def __init__(self):
        super().__init__(name="MicroCap")
        self._pool = MainSmallPool()

    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """计算微盘股基准的多指标统计

        逻辑：
        1. 向前多加数个交易日（用于计算 T-1 的排名）
        2. 使用 DataProvider 加载 MainSmallPool 数据
        3. 按 ASSET 排序，计算 T-1 的市值排名（shift 前向）
        4. 过滤：市值排名 <= 400（这些股票在 T-1 被选中）
        5. 按 DATE 分组，计算 7 个统计指标

        Args:
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            pl.DataFrame with columns:
                DATE, ret, ret_mean, ret_median, cum_ret,
                avg_mcap, median_amt, turnover_rate_avg
        """
        # 延迟导入以避免循环导入
        from alpha_factory.data_provider.data_provider import DataProvider

        dp = DataProvider()

        logger.info(
            f"📊 计算微盘股基准 "
            f"({start_date.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})"
        )

        # 向前追溯 10 个交易日，用于确保能获取 T-1 的排名信息
        lookback_days = 10
        date_before_start = start_date - timedelta(days=lookback_days)

        # 加载数据（向前追溯）
        lf = dp.load_pool_data(
            self._pool,
            date_before_start,
            end_date,
            exprs=[],
            actions=None,
        )

        # 确保包含必要的列
        required_cols = [F.DATE, F.ASSET, F.TOTAL_MV, F.RET, F.AMOUNT, F.TURNOVER_RATE]
        available_schema = lf.collect_schema()
        for col in required_cols:
            if col not in available_schema:
                logger.warning(f"⚠️ 缺少必需列: {col}，跳过此指标计算")

        # 第零步：先过滤停牌股和新股（提升数据质量）
        # 这确保我们只计算可交易股票的基准
        lf = lf.filter(
            ~pl.col(F.IS_SUSPENDED)  # 非停牌
            & (pl.col(F.LIST_DAYS) >= 180)  # 上市超过 180 天
            & (~pl.col("IS_ST"))
        )

        # 第一步：按 ASSET 排序，计算每日排名，然后取 T-1 的排名
        # 关键：对每一行数据，我们需要知道这只股票在 T-1（前一天）的排名
        # 使用 shift(1).over(ASSET) 获取同一股票前一天的值
        lf = (
            lf.sort([F.ASSET, F.DATE])
            .with_columns(
                [
                    # 当日排名
                    pl.col(F.TOTAL_MV)
                    .rank("ordinal")
                    .over(F.DATE)
                    .alias("_mv_rank_today"),
                    # T-1 的排名：shift(1) 获取前一行（前一交易日）
                    pl.col(F.TOTAL_MV)
                    .rank("ordinal")
                    .over(F.DATE)
                    .shift(1)
                    .over(F.ASSET)
                    .alias("_mv_rank_prev"),
                ]
            )
            # 第二步：过滤【在 T-1 被选中的股票】
            .filter(pl.col("_mv_rank_prev") <= 400)
        )

        # 第三步：按 DATE 分组，计算统计指标
        stats_df = (
            lf.group_by(F.DATE)
            .agg(
                [
                    # 基础指标：等权平均收益
                    pl.col(F.RET).mean().cast(pl.Float32).alias("ret_mean"),
                    # 中位数：剔除极端值
                    pl.col(F.RET).median().cast(pl.Float32).alias("ret_median"),
                    # 平均市值：监控风格漂移
                    pl.col(F.TOTAL_MV).mean().cast(pl.Float32).alias("avg_mcap"),
                    # 成交额中位数：评估资金容纳能力
                    pl.col(F.AMOUNT).median().cast(pl.Float32).alias("median_amt"),
                    # 平均换手率：评估活跃度
                    pl.col(F.TURNOVER_RATE)
                    .mean()
                    .cast(pl.Float32)
                    .alias("turnover_rate_avg"),
                ]
            )
            .sort(F.DATE)
        )

        # 第四步：计算累计收益（使用 Lazy API 的 cum_prod）
        stats_df = stats_df.with_columns(
            [
                # cum_prod: (1 + ret_mean).cum_prod() - 1
                ((1 + pl.col("ret_mean")).cum_prod() - 1)
                .cast(pl.Float32)
                .alias("cum_ret"),
            ]
        )

        stats_collected = stats_df.collect()

        if stats_collected.is_empty():
            logger.warning("⚠️ 微盘股基准计算结果为空")
            return pl.DataFrame(
                schema={
                    "DATE": pl.Date,
                    "ret": pl.Float32,
                    "ret_mean": pl.Float32,
                    "ret_median": pl.Float32,
                    "cum_ret": pl.Float32,
                    "avg_mcap": pl.Float32,
                    "median_amt": pl.Float32,
                    "turnover_rate_avg": pl.Float32,
                }
            )

        # 第五步：组合结果（添加 ret 别名用于向后兼容）
        result = stats_collected.with_columns(
            [
                pl.col("ret_mean").alias("ret"),  # 向后兼容：ret = ret_mean
            ]
        ).select(
            [
                F.DATE,
                "ret",
                "ret_mean",
                "ret_median",
                "cum_ret",
                "avg_mcap",
                "median_amt",
                "turnover_rate_avg",
            ]
        )

        # 第六步：过滤回到原始日期范围（移除向前追溯的数据）
        result = result.filter(
            (pl.col(F.DATE) >= start_date) & (pl.col(F.DATE) <= end_date)
        )

        logger.info(f"✓ 成功计算 {result.height} 个交易日的统计数据")
        return result
