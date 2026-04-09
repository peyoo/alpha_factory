"""基准管理模块

提供预定义的基准实现，支持因子表达式计算与策略择时。
"""

from alpha_factory.data_provider.benchmarks.hs300 import HS300Benchmark

__all__ = ["HS300Benchmark"]
