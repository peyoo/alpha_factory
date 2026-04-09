"""
数据接入层 (Data Provider) - L0-L4 ETL 管道

【模块职责】
- L0 (API层): 从 Tushare 异步/批量获取原始行情与基本面数据。
- L1 (缓存层): HDF5 热缓存管理，支撑快速实验并减轻 API 额度压力。
- L2/L3 (加工层): 统一因子库构建。执行清洗、对齐、前向填充及复权计算，按年分区存储。
- L4 (接口层): 为下游提供标准化的 DataProvider 接口，支持 Lazy Mode 自动查询优化。

【核心契约】
- 坐标系: 始终以 (DATE, ASSET) 为主键。
- 单位制: 统一为 (元 / 股 / 倍)，消除万元、手、千元等量纲陷阱。
- 性能制: 深度集成 Polars LazyFrame，支持谓词下压与列过滤优化。
"""

from __future__ import annotations

# 1. 基础服务与缓存管理 (L0/L1)
from alpha_factory.data_provider.tushare_service import TushareDataService

# 2. 基准数据管理
from alpha_factory.data_provider.benchmark import Benchmark
from alpha_factory.data_provider.benchmarks import HS300Benchmark, MicroCapBenchmark

# 3. 统一读取接口 (L4)
from alpha_factory.data_provider.data_provider import DataProvider

# 显式暴露接口，方便 from alpha.data_provider import *
__all__ = [
    "TushareDataService",
    "DataProvider",
    "Benchmark",
    "HS300Benchmark",
    "MicroCapBenchmark",
]


# --- 快速诊断信息 ---
def info():
    """打印数据层核心状态简报"""
    from alpha_factory.config.base import settings
    import polars as pl

    print("=" * 40)
    print("📊 ALPHA DATA PROVIDER ENGINE STATUS")
    print("-" * 40)
    print(f"📦 Warehouse: {settings.WAREHOUSE_DIR}")
    print(f"🔥 L1 Cache : {settings.RAW_DATA_DIR}")
    print(f"🚀 Engine   : Polars {pl.__version__}")
    print("=" * 40)
