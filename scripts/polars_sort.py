import polars as pl
import time

# 1. 创建物理有序的数据 (100万行)
n = 1_000_000
df = pl.DataFrame(
    {
        "date": pl.int_range(0, n, eager=True),
        "value": pl.int_range(0, n, eager=True).cast(pl.Float64) * 0.5,
    }
)

# --- 场景 A：不声明有序 (默认走 Hash 路径) ---
lazy_a = df.lazy().group_by("date").agg(pl.col("value").mean())

# --- 场景 B：显式声明已排序 (走 Sorted 路径) ---
lazy_b = (
    df.lazy()
    .with_columns(pl.col("date").set_sorted())
    .group_by("date")
    .agg(pl.col("value").mean())
)

# 2. 开始压力测试：重复执行 100 次计算
print("正在测试 100 万行数据的 100 次聚合计算...\n")

# 测试 A
start_a = time.perf_counter()
for _ in range(100):
    _ = lazy_a.collect()
end_a = time.perf_counter()
time_a = end_a - start_a

# 测试 B
start_b = time.perf_counter()
for _ in range(100):
    _ = lazy_b.collect()
end_b = time.perf_counter()
time_b = end_b - start_b

# 3. 输出结果对比
print(f"{'策略':<20} | {'总耗时 (100次)':<15}")
print("-" * 40)
print(f"{'未标记 (Hash)':<20} | {time_a:>12.4f}s")
print(f"{'已标记 (Sorted)':<20} | {time_b:>12.4f}s")
print("-" * 40)
print(f"性能提升: {((time_a - time_b) / time_a * 100):.2f}%")

# 4. 打印执行计划确认
print("\n--- 验证底层算子 (请观察 AGGREGATE 是否带 sorted 标记) ---")
print(f"场景 B 计划片段:\n{lazy_b.explain().split('---')[0]}")
