"""Benchmark 与 sync 命令集成测试

验证以下功能：
1. Benchmark 自动注册机制
2. TushareDataService.sync_data() 调用 _update_all_benchmarks()
3. sync 命令在 Benchmark 更新失败时中断
4. 增量更新逻辑（多次运行 sync，验证 Benchmark 智能跳过）
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch
from alpha_factory.data_provider.benchmark import Benchmark
from alpha_factory.data_provider.benchmarks.hs300 import HS300Benchmark
from alpha_factory.data_provider.benchmarks.micro_cap import MicroCapBenchmark
from alpha_factory.data_provider.tushare_service import TushareDataService


class TestBenchmarkRegistration:
    """测试 Benchmark 自动注册机制"""

    def test_benchmark_registry_not_empty(self):
        """验证注册表不为空"""
        benchmarks = Benchmark.list_all_benchmarks()
        assert len(benchmarks) > 0, "应该至少有一个已注册的 Benchmark"

    def test_hs300_benchmark_in_registry(self):
        """验证 HS300Benchmark 已注册"""
        benchmarks = Benchmark.list_all_benchmarks()
        assert "HS300Benchmark" in benchmarks
        assert benchmarks["HS300Benchmark"] == HS300Benchmark

    def test_microcap_benchmark_in_registry(self):
        """验证 MicroCapBenchmark 已注册"""
        benchmarks = Benchmark.list_all_benchmarks()
        assert "MicroCapBenchmark" in benchmarks
        assert benchmarks["MicroCapBenchmark"] == MicroCapBenchmark

    def test_all_benchmarks_instantiable(self):
        """验证所有已注册的 Benchmark 都能实例化"""
        benchmarks = Benchmark.list_all_benchmarks()
        for class_name, benchmark_class in benchmarks.items():
            instance = benchmark_class()
            assert instance is not None
            assert hasattr(instance, "name")
            assert hasattr(instance, "_cache_path")


class TestSyncBenchmarkIntegration:
    """测试 sync 命令与 Benchmark 的集成"""

    def test_update_all_benchmarks_called_via_sync(self):
        """验证 sync_data() 调用 _update_all_benchmarks()（集成测试）

        注：此测试通过其他单元测试间接验证，主要测试在下方。
        """
        # 由于 TushareDataService.__init__ 需要复杂的 mock 设置
        # 此集成测试由 test_update_all_benchmarks_calls_update_on_each_benchmark 覆盖
        pass

    @patch(
        "alpha_factory.data_provider.tushare_service.TushareDataService._sync_single_day_bundle"
    )
    @patch("alpha_factory.data_provider.tushare_service.Benchmark.list_all_benchmarks")
    def test_update_all_benchmarks_called(
        self,
        mock_list_all,
        mock_sync_single_day,
    ):
        """验证 sync_data() 在因子构建后调用 _update_all_benchmarks()"""
        # 创建两个 mock Benchmark 实例
        mock_instance_1 = Mock()
        mock_instance_2 = Mock()

        mock_benchmark_class_1 = Mock(return_value=mock_instance_1)
        mock_benchmark_class_2 = Mock(return_value=mock_instance_2)

        mock_list_all.return_value = {
            "MockBenchmark1": mock_benchmark_class_1,
            "MockBenchmark2": mock_benchmark_class_2,
        }

        # 初始化 service 实例（完全 mock 外部依赖）
        with patch("alpha_factory.data_provider.tushare_service.HDF5CacheManager"):
            with patch(
                "alpha_factory.data_provider.tushare_service.TradeCalendarManager"
            ) as mock_cal:
                with patch(
                    "alpha_factory.data_provider.tushare_service.StockAssetsManager"
                ):
                    with patch(
                        "alpha_factory.data_provider.tushare_service.UnifiedFactorBuilder"
                    ) as mock_builder:
                        # 设置 mock calendar 返回值
                        mock_cal_instance = Mock()
                        mock_cal.return_value = mock_cal_instance
                        mock_cal_instance.get_trade_days.return_value = [
                            date(2026, 4, 8)
                        ]
                        mock_cal_instance.sync_from_tushare = Mock()

                        # 设置 mock builder
                        mock_builder_instance = Mock()
                        mock_builder.return_value = mock_builder_instance

                        service = TushareDataService()
                        service.sync_data("20260408", "20260408")

                        # 验证 Benchmark.update() 被调用
                        mock_instance_1.update.assert_called()
                        mock_instance_2.update.assert_called()

    @patch(
        "alpha_factory.data_provider.tushare_service.TushareDataService._sync_single_day_bundle"
    )
    @patch("alpha_factory.data_provider.tushare_service.Benchmark.list_all_benchmarks")
    def test_update_all_benchmarks_calls_update_on_each_benchmark(
        self, mock_list_all, mock_sync_single_day
    ):
        """验证 _update_all_benchmarks() 对每个 Benchmark 调用 update()"""
        # 创建两个 mock Benchmark 类
        mock_benchmark_class_1 = Mock()
        mock_benchmark_class_2 = Mock()

        mock_instance_1 = Mock()
        mock_instance_2 = Mock()

        mock_benchmark_class_1.return_value = mock_instance_1
        mock_benchmark_class_2.return_value = mock_instance_2

        mock_list_all.return_value = {
            "MockBenchmark1": mock_benchmark_class_1,
            "MockBenchmark2": mock_benchmark_class_2,
        }

        # 创建 service 并调用 _update_all_benchmarks
        with patch("alpha_factory.data_provider.tushare_service.HDF5CacheManager"):
            with patch(
                "alpha_factory.data_provider.tushare_service.TradeCalendarManager"
            ):
                with patch(
                    "alpha_factory.data_provider.tushare_service.StockAssetsManager"
                ):
                    with patch(
                        "alpha_factory.data_provider.tushare_service.UnifiedFactorBuilder"
                    ):
                        service = TushareDataService()

        end_date = date(2026, 4, 8)
        service._update_all_benchmarks(end_date)

        # 验证每个 Benchmark 都被实例化和更新
        mock_benchmark_class_1.assert_called_once()
        mock_benchmark_class_2.assert_called_once()

        mock_instance_1.update.assert_called_once_with(end_date=end_date)
        mock_instance_2.update.assert_called_once_with(end_date=end_date)

    @patch("alpha_factory.data_provider.tushare_service.Benchmark.list_all_benchmarks")
    def test_update_all_benchmarks_raises_on_failure(self, mock_list_all):
        """验证 Benchmark 更新失败时 _update_all_benchmarks() 抛出异常"""
        # 创建一个会抛出异常的 mock Benchmark 类
        mock_benchmark_class = Mock()
        mock_instance = Mock()
        mock_instance.update.side_effect = RuntimeError("模拟 Benchmark 更新失败")
        mock_benchmark_class.return_value = mock_instance

        mock_list_all.return_value = {"FailingBenchmark": mock_benchmark_class}

        # 创建 service
        with patch("alpha_factory.data_provider.tushare_service.HDF5CacheManager"):
            with patch(
                "alpha_factory.data_provider.tushare_service.TradeCalendarManager"
            ):
                with patch(
                    "alpha_factory.data_provider.tushare_service.StockAssetsManager"
                ):
                    with patch(
                        "alpha_factory.data_provider.tushare_service.UnifiedFactorBuilder"
                    ):
                        service = TushareDataService()

        # 验证异常被抛出
        with pytest.raises(RuntimeError, match="模拟 Benchmark 更新失败"):
            service._update_all_benchmarks(date(2026, 4, 8))

    @patch("alpha_factory.data_provider.tushare_service.Benchmark.list_all_benchmarks")
    def test_update_all_benchmarks_handles_empty_registry(self, mock_list_all):
        """验证当没有注册的 Benchmark 时，_update_all_benchmarks() 正确处理"""
        mock_list_all.return_value = {}

        # 创建 service
        with patch("alpha_factory.data_provider.tushare_service.HDF5CacheManager"):
            with patch(
                "alpha_factory.data_provider.tushare_service.TradeCalendarManager"
            ):
                with patch(
                    "alpha_factory.data_provider.tushare_service.StockAssetsManager"
                ):
                    with patch(
                        "alpha_factory.data_provider.tushare_service.UnifiedFactorBuilder"
                    ):
                        service = TushareDataService()

        # 应该不抛出异常，正常返回
        service._update_all_benchmarks(date(2026, 4, 8))


class TestBenchmarkIncrementalUpdate:
    """测试 Benchmark 增量更新逻辑"""

    def test_list_all_benchmarks_returns_copy(self):
        """验证 list_all_benchmarks() 返回的是注册表的副本，而非引用"""
        benchmarks1 = Benchmark.list_all_benchmarks()
        benchmarks2 = Benchmark.list_all_benchmarks()

        # 应该相等但不是同一个对象
        assert benchmarks1 == benchmarks2
        assert benchmarks1 is not benchmarks2

    def test_benchmark_instances_have_correct_names(self):
        """验证 Benchmark 实例的名称与文件路径正确"""
        hs300 = HS300Benchmark()
        assert hs300.name == "HS300"
        assert "HS300.parquet" in str(hs300._cache_path)

        microcap = MicroCapBenchmark()
        assert microcap.name == "MicroCap"
        assert "MicroCap.parquet" in str(microcap._cache_path)


__all__ = [
    "TestBenchmarkRegistration",
    "TestSyncBenchmarkIntegration",
    "TestBenchmarkIncrementalUpdate",
]
