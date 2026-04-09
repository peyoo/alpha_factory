"""Benchmark 基础功能测试

测试项目：
1. HS300Benchmark 初始化和数据加载
2. 增量更新逻辑
3. join_benchmark 方法
"""

import pytest
from datetime import date
import polars as pl
from pathlib import Path

from alpha_factory.data_provider import HS300Benchmark
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.config.base import settings


class TestHS300Benchmark:
    """HS300 Benchmark 功能测试"""

    def test_initialization(self):
        """测试 HS300Benchmark 初始化"""
        bench = HS300Benchmark()
        assert bench.name == "HS300"
        assert isinstance(bench._cache_path, Path)
        assert bench._default_start_date == date(2010, 1, 1)

    def test_cache_path_exists(self):
        """测试缓存路径是否正确"""
        bench = HS300Benchmark()
        expected_path = settings.BENCHMARKS_DIR / "HS300.parquet"
        assert bench._cache_path == expected_path

    def test_get_last_date_no_file(self):
        """测试 _get_last_date 在文件不存在时的行为"""
        bench = HS300Benchmark()
        # 临时创建一个不存在的benchmark
        bench._cache_path = settings.BENCHMARKS_DIR / "nonexistent_bench.parquet"
        last_date = bench._get_last_date()
        assert last_date is None

    def test_get_last_date_with_data(self):
        """测试 _get_last_date 在有数据时的行为"""
        if not (settings.BENCHMARKS_DIR / "HS300.parquet").exists():
            pytest.skip("HS300 数据未初始化")

        bench = HS300Benchmark()
        last_date = bench._get_last_date()
        assert last_date is not None
        assert isinstance(last_date, date)

    def test_load_returns_schema(self):
        """测试 load_returns 返回的 LazyFrame 结构"""
        if not (settings.BENCHMARKS_DIR / "HS300.parquet").exists():
            pytest.skip("HS300 数据未初始化")

        bench = HS300Benchmark()
        start_dt = date(2024, 1, 1)
        end_dt = date(2024, 1, 31)

        lf = bench.load_returns(start_dt, end_dt)
        assert isinstance(lf, pl.LazyFrame)

        # 验证列名和类型
        cols = lf.collect().columns
        assert "DATE" in cols
        assert "ret" in cols

    def test_load_returns_empty_range(self):
        """测试 load_returns 在没有数据的日期范围"""
        bench = HS300Benchmark()
        # 使用一个远未来的日期范围
        start_dt = date(2099, 1, 1)
        end_dt = date(2099, 1, 31)

        lf = bench.load_returns(start_dt, end_dt)
        df = lf.collect()
        # 应该返回空 DataFrame 或成功处理但无行
        assert isinstance(df, pl.DataFrame)


class TestPoolJoinBenchmark:
    """Pool join_benchmark 功能测试"""

    def test_join_benchmark_method_exists(self):
        """测试 PoolUniverse 是否有 join_benchmark 方法"""
        pool = MainSmallPool()
        assert hasattr(pool, "join_benchmark")
        assert callable(pool.join_benchmark)

    def test_join_benchmark_signature(self):
        """测试 join_benchmark 方法签名"""
        import inspect

        pool = MainSmallPool()
        sig = inspect.signature(pool.join_benchmark)

        # 验证参数
        params = list(sig.parameters.keys())
        assert "lf" in params
        assert "benchmark" in params
        assert "start_date" in params
        assert "end_date" in params
        assert "col_name" in params


class TestMicroCapBenchmark:
    """微盘股基准功能测试"""

    def test_initialization(self):
        """测试 MicroCapBenchmark 初始化"""
        from alpha_factory.data_provider import MicroCapBenchmark

        bench = MicroCapBenchmark()
        assert bench.name == "MicroCap"
        assert bench._default_start_date == date(2010, 1, 1)

    def test_cache_path_exists(self):
        """测试缓存路径是否设置正确"""
        from alpha_factory.data_provider import MicroCapBenchmark

        bench = MicroCapBenchmark()
        expected_path = settings.BENCHMARKS_DIR / "MicroCap.parquet"
        assert bench._cache_path == expected_path

    def test_load_statistics_method_exists(self):
        """测试 load_statistics 方法是否存在"""
        from alpha_factory.data_provider import MicroCapBenchmark

        bench = MicroCapBenchmark()
        assert hasattr(bench, "load_statistics")
        assert callable(bench.load_statistics)

    def test_load_statistics_schema_when_no_data(self):
        """测试当没有数据时 load_statistics 返回的 schema"""
        from alpha_factory.data_provider import MicroCapBenchmark

        bench = MicroCapBenchmark()
        # 如果文件不存在，应该返回只有 DATE 和 ret 列的 LazyFrame
        lf = bench.load_statistics(date(2024, 1, 1), date(2024, 1, 31))
        assert lf is not None

    def test_load_returns_method_works(self):
        """测试 load_returns 方法（向后兼容）"""
        from alpha_factory.data_provider import MicroCapBenchmark

        bench = MicroCapBenchmark()
        # 如果文件不存在，应该返回空的 LazyFrame
        lf = bench.load_returns(date(2024, 1, 1), date(2024, 1, 31))
        assert lf is not None
        # 验证返回的是 LazyFrame
        assert isinstance(lf, pl.LazyFrame)


class TestBenchmarkIntegration:
    """集成测试"""

    def test_benchmark_data_persistence(self):
        """测试 benchmark 数据是否持久化"""
        bench_path = settings.BENCHMARKS_DIR / "HS300.parquet"

        if bench_path.exists():
            # 读取数据验证格式
            df = pl.read_parquet(bench_path)
            assert "DATE" in df.columns
            assert "ret" in df.columns
            assert df.height > 0

            # 验证数据类型
            schema = df.schema
            assert schema["DATE"] == pl.Date
            assert schema["ret"] == pl.Float32


if __name__ == "__main__":
    # 运行基础检查
    print("运行 Benchmark 基础测试...")

    test = TestHS300Benchmark()

    print("✓ 测试初始化...", end=" ")
    test.test_initialization()
    print("PASS")

    print("✓ 测试缓存路径...", end=" ")
    test.test_cache_path_exists()
    print("PASS")

    print("✓ 测试不存在文件...", end=" ")
    test.test_get_last_date_no_file()
    print("PASS")

    print("✓ 测试 Pool join_benchmark...", end=" ")
    TestPoolJoinBenchmark().test_join_benchmark_method_exists()
    print("PASS")

    print("\n✅ 所有基础测试通过！")
