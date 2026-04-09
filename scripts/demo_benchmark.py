#!/usr/bin/env python
"""Benchmark 架构演示脚本

展示：
1. 初始化和增量更新 HS300 基准
2. 在因子表达式中 join 基准数据
3. 在评估时使用基准进行对比
"""

from datetime import date, timedelta
from alpha_factory.data_provider import HS300Benchmark
from alpha_factory.data_provider.pool import MainSmallPool

# ============================================================================
# 演示 1：初始化 HS300 基准并同步数据
# ============================================================================

print("=" * 70)
print("演示 1: Benchmark 数据同步")
print("=" * 70)

bench = HS300Benchmark()
print(f"\n📊 Benchmark 名称: {bench.name}")
print(f"📁 缓存路径: {bench._cache_path}")

# 首次同步：从 2024-01-01 到今天
print("\n第一次同步（会从 API 拉取数据）...")
try:
    # 为演示起见，只拉取最近 5 天的数据
    end_date = date.today()
    start_date = end_date - timedelta(days=5)
    bench.update(end_date=end_date)
    print("✅ 初次同步完成")
except Exception as e:
    print(f"⚠️  同步失败（可能是因为未配置 TUSHARE_TOKEN）: {e}")
    print("   请在 .env 或 settings 中配置 TUSHARE_TOKEN")

# ============================================================================
# 演示 2：加载基准数据并进行 join
# ============================================================================

print("\n" + "=" * 70)
print("演示 2: 在因子表中 join 基准数据")
print("=" * 70)

pool = MainSmallPool()
print(f"\n📊 Stock Pool: {pool.name}")

# 模拟日期范围
start_dt = date(2024, 1, 1)
end_dt = date(2024, 1, 31)

print(f"📅 日期范围: {start_dt} ~ {end_dt}")

# 假设已有 LazyFrame (在实际代码中通过 dp.load_pool_data 获取)
try:
    # 加载基准收益率
    bench_lf = bench.load_returns(start_dt, end_dt)
    print("\n✅ 基准数据加载成功")

    # 显示 LazyFrame 结构
    bench_collect = bench_lf.collect()
    print(f"   行数: {bench_collect.height}")
    print(f"   列: {bench_collect.columns}")
    if bench_collect.height > 0:
        print("\n   样本数据:")
        print(bench_collect.head(3))
except Exception as e:
    print(f"⚠️  加载失败（数据可能不存在）: {e}")

# ============================================================================
# 演示 3：展示如何在因子表达式中使用基准
# ============================================================================

print("\n" + "=" * 70)
print("演示 3: 因子表达式用法（代码示例）")
print("=" * 70)

usage_code = """
# 在实际使用中，可以这样写：

from alpha_factory.data_provider import DataProvider, HS300Benchmark
from alpha_factory.data_provider.pool import MainSmallPool

dp = DataProvider()
pool = MainSmallPool()
bench = HS300Benchmark()

# 第一步：加载池数据
start_date = "20240101"
end_date = "20240131"
lf = dp.load_pool_data(pool, start_date, end_date, exprs=[])

# 第二步：加载基准并 join
# 方式 A: 直接使用 pool 的 join_benchmark 方法
from datetime import datetime as dt
start_dt = dt.strptime(start_date, "%Y%m%d").date()
end_dt = dt.strptime(end_date, "%Y%m%d").date()
lf = pool.join_benchmark(lf, bench, start_dt, end_dt, col_name="BENCH_RET")

# 第三步：在表达式中使用基准
exprs = [
    "excess_ret = RET - BENCH_RET",  # 超额收益
    "relative_vol = CLOSE.ts_std(20) / BENCH_RET.ts_std(20)",  # 相对波动率
]
result_lf = dp.build_factors_view(pool, lf, exprs=exprs)
result_df = result_lf.collect()
"""

print(usage_code)

# ============================================================================
# 演示 4：展示评估时的基准用法
# ============================================================================

print("\n" + "=" * 70)
print("演示 4: 评估时使用基准（代码示例）")
print("=" * 70)

eval_code = """
# 在回测报告中使用基准：

from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.data_provider import HS300Benchmark

# 假设已有回测结果 result
result = {...}  # 包含 'series' 键的回测结果

# 使用基准进行对比
bench = HS300Benchmark()
generate_and_open_report(
    result,
    factor_name="my_factor_v1",
    benchmark=bench,
    start_date="20240101",  # 需要与回测日期对齐
    end_date="20240131",
)
# ✅ 报告会包含 Benchmark vs Strategy 的对比分析
"""

print(eval_code)

# ============================================================================
# 演示 5：展示多 benchmark 的支持
# ============================================================================

print("\n" + "=" * 70)
print("演示 5: 支持多个基准（扩展示例）")
print("=" * 70)

extend_code = '''
# 如果需要支持多个基准（如 HS300, ZZ500, CSI1000），可以：

class ZZ500Benchmark(Benchmark):
    """中证500指数"""
    def __init__(self):
        super().__init__(name="ZZ500", default_start_date=date(2010, 1, 1))

    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        # 从 Tushare 拉取 000905.SH 数据
        ...

class CSI1000Benchmark(Benchmark):
    """中证1000指数"""
    def __init__(self):
        super().__init__(name="CSI1000", default_start_date=date(2015, 1, 1))

    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        # 从 Tushare 拉取 000852.SH 数据
        ...

# 使用时灵活选择
bench_hs300 = HS300Benchmark()
bench_zz500 = ZZ500Benchmark()

# 都支持增量更新和 join
bench_hs300.update()
bench_zz500.update()

lf = pool.join_benchmark(lf, bench_hs300, start_dt, end_dt, col_name="HS300_RET")
lf = pool.join_benchmark(lf, bench_zz500, start_dt, end_dt, col_name="ZZ500_RET")
'''

print(extend_code)

print("\n" + "=" * 70)
print("✅ 演示完成！")
print("=" * 70)

# ============================================================================
# 演示 6：MicroCapBenchmark - 微盘股基准（新增）
# ============================================================================

print("\n" + "=" * 70)
print("演示 6: MicroCapBenchmark - 微盘股基准（新增）⭐️")
print("=" * 70)

microcap_code = """
# MicroCapBenchmark 是一个特殊的基准，定义为：
# - T-1 日市值排名 <= 400 的主板+创业板股票
# - T 日计算这些股票的多维统计指标

from alpha_factory.data_provider import MicroCapBenchmark
from alpha_factory.data_provider.pool import MainSmallPool
from datetime import datetime as dt

# 第一步：初始化并更新
bench = MicroCapBenchmark()
print(f"Benchmark 名称: {bench.name}")
bench.update()  # 计算并缓存多指标数据

# 第二步：在因子表达式中使用（与 HS300 相同接口）
# 优点：完全基于本地数据计算，零 API 消耗
pool = MainSmallPool()
lf = dp.load_pool_data(pool, "20240101", "20240131")
start_dt = dt.strptime("20240101", "%Y%m%d").date()
end_dt = dt.strptime("20240131", "%Y%m%d").date()
lf = pool.join_benchmark(lf, bench, start_dt, end_dt, col_name="MICROCAP_RET")

# 即可使用 MICROCAP_RET 在表达式中
exprs = [
    "vs_microcap = RET - MICROCAP_RET",  # 相对微盘股基准的超额收益
]

# 第三步：风险监控（新增功能）
# MicroCapBenchmark 提供 8 个统计指标，用于风险评估
stats = bench.load_statistics(start_dt, end_dt).collect()
print(f"\\n📊 微盘股基准统计指标（{stats.height} 个交易日）：")
print(f"  平均收益: {stats['ret_mean'].mean():.2%}")
print(f"  收益中位数: {stats['ret_median'].mean():.2%}")
print(f"  平均市值: ¥{stats['avg_mcap'].mean() / 1e8:.1f} 亿")
print(f"  成交额中位数: ¥{stats['median_amt'].median() / 1e7:.1f} 千万")
print(f"  平均换手率: {stats['turnover_rate_avg'].mean():.2%}")

# 第四步：识别异常
median_amt = stats["median_amt"].median()
if median_amt < 3e7:
    print(f"⚠️  流动性风险：成交额中位数 {median_amt/1e7:.1f} 千万，难以承载大资金")

# 平均市值漂移
avg_mcap_min = stats["avg_mcap"].min()
avg_mcap_max = stats["avg_mcap"].max()
drift_ratio = (avg_mcap_max - avg_mcap_min) / avg_mcap_min
if drift_ratio > 0.1:  # 10% 以上
    print(f"⚠️  风格漂移：市值范围 {avg_mcap_min/1e8:.1f}~{avg_mcap_max/1e8:.1f} 亿，漂移 {drift_ratio:.1%}")

# 极端行情识别
for date, mean, median in zip(stats["DATE"], stats["ret_mean"], stats["ret_median"]):
    deviation = mean - median
    if abs(deviation) > 0.02:  # 2% 以上偏差
        print(f"⚠️  {date}：检测到极端行情（均值-中位数={deviation:.2%}）")
"""

print(microcap_code)

print("\n" + "=" * 70)
print("✅ 演示完成！")
print("=" * 70)
print("\n关键特性总结：")
print("  1️⃣  增量更新：自动检测本地最新日期，只拉取之后的数据")
print("  2️⃣  LazyFrame join：高效地在因子表中添加基准列")
print("  3️⃣  因子表达式支持：基准可以像其他列一样参与表达式计算")
print("  4️⃣  评估集成：report 可以自动对比基准")
print("  5️⃣  扩展性：轻松添加新的基准子类（ZZ500, CSI1000 等）")
print("  6️⃣  ⭐️ 微盘股基准：本地计算、多维统计、零 API 消耗、风险监控")
