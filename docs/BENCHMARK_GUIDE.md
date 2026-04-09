# Benchmark 架构实现指南

## 概述

Benchmark 架构为 Alpha-Factory 提供了灵活的基准数据管理系统，支持：

1. **因子表达式中的 join**：基准数据可以作为虚拟列参与因子计算
2. **策略择时**：评估时可以对标基准，进行对比分析
3. **增量更新**：自动增量同步，避免重复拉取数据
4. **扩展性**：轻松添加新基准（ZZ500、CSI1000 等）

---

## 核心组件

### 1. `Benchmark` 基类
**位置**：`src/alpha_factory/data_provider/benchmark.py`

**职责**：
- 定义基准的标准接口
- 管理增量更新逻辑
- 提供 LazyFrame 加载接口

**关键方法**：
```python
class Benchmark(ABC):
    def __init__(self, name: str, default_start_date: date = date(2010, 1, 1))
    def update(self, end_date: Optional[date] = None) -> None
    def load_returns(self, start_date: date, end_date: date) -> pl.LazyFrame

    @abstractmethod
    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame
```

### 2. `HS300Benchmark` 实现
**位置**：`src/alpha_factory/data_provider/benchmarks/hs300.py`

**数据源**：Tushare Pro / index_daily
**指数代码**：000300.SH（沪深300指数）

**使用**：
```python
from alpha_factory.data_provider import HS300Benchmark

bench = HS300Benchmark()
bench.update()  # 自动增量同步
```

### 3. `Pool.join_benchmark()` 方法
**位置**：`src/alpha_factory/data_provider/pool.py`

**职责**：
- 将基准数据 left-join 到因子 LazyFrame
- 广播基准列到所有 ASSET

**用法**：
```python
from datetime import datetime as dt
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.data_provider import HS300Benchmark

pool = MainSmallPool()
bench = HS300Benchmark()

# 基准数据会被 join 到 lf，新增 BENCH_RET 列
start_dt = dt.strptime("20240101", "%Y%m%d").date()
end_dt = dt.strptime("20240131", "%Y%m%d").date()
lf = pool.join_benchmark(lf, bench, start_dt, end_dt, col_name="BENCH_RET")
```

### 4. 评估层集成
**位置**：`src/alpha_factory/evaluation/backtest/utils.py`

**功能**：`generate_and_open_report` 函数支持传入 benchmark 参数

**用法**：
```python
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.data_provider import HS300Benchmark

bench = HS300Benchmark()
generate_and_open_report(
    result,
    factor_name="my_factor_v1",
    benchmark=bench,
    start_date="20240101",
    end_date="20240131",
)
# ✅ 生成的 HTML 报告包含 Benchmark vs Strategy 的对标分析
```

---

## 数据存储结构

### Parquet Schema
```
文件：data/warehouse/benchmarks/{name}.parquet
结构：
  DATE: Date       # 交易日期
  ret:  Float32    # 日收益率（小数形式，如 0.01 = 1%）
```

### 增量更新策略
- 首次同步：从 `default_start_date` 拉取至今天
- 后续同步：自动检测本地最新日期，仅拉取之后的数据
- 冲突处理：使用 `unique("DATE")` 去重，新数据优先

---

## 使用场景

### 场景 1：因子优化 - 计算超额收益

```python
from alpha_factory.data_provider import DataProvider, HS300Benchmark
from alpha_factory.data_provider.pool import MainSmallPool

dp = DataProvider()
pool = MainSmallPool()
bench = HS300Benchmark()

# 第一步：加载池数据
lf = dp.load_pool_data(pool, "20240101", "20240131", exprs=[])

# 第二步：加载基准并 join
from datetime import datetime as dt
start_dt = dt.strptime("20240101", "%Y%m%d").date()
end_dt = dt.strptime("20240131", "%Y%m%d").date()
lf = pool.join_benchmark(lf, bench, start_dt, end_dt, col_name="BENCH_RET")

# 第三步：在表达式中计算超额收益
exprs = [
    "excess_ret = RET - BENCH_RET",          # 超额收益
    "tracking_error = (RET - BENCH_RET)^2",  # 跟踪误差平方
    "ir = excess_ret / tracking_error",      # 信息比（简化）
]

# 因子表达式可以直接使用 BENCH_RET 列
result_lf = dp.build_factors_view(pool, lf, exprs=exprs)
result_df = result_lf.collect()
```

### 场景 2：回测评估 - 对标分析

```python
from alpha_factory.evaluation.backtest.utils import generate_and_open_report
from alpha_factory.data_provider import HS300Benchmark

# 假设已有回测结果
result = backtest_engine.run(...)  # {'series': pl.DataFrame}

bench = HS300Benchmark()
bench.update()  # 确保数据最新

# 生成包含基准对标的 HTML 报告
generate_and_open_report(
    result,
    factor_name="smart_factor_v2",
    benchmark=bench,
    start_date="20240101",
    end_date="20240131",
)
```

### 场景 3：多基准对比

```python
from alpha_factory.data_provider import HS300Benchmark
from alpha_factory.data_provider.benchmarks import CSI1000Benchmark  # 假设已实现

# 同时加载多个基准
bench_hs300 = HS300Benchmark()
bench_csi1000 = CSI1000Benchmark()

# 都支持增量更新
bench_hs300.update()
bench_csi1000.update()

# 各自 join 到因子表中
lf = pool.join_benchmark(lf, bench_hs300, start_dt, end_dt, col_name="BENCH_HS300")
lf = pool.join_benchmark(lf, bench_csi1000, start_dt, end_dt, col_name="BENCH_CSI1000")

# 可以创建相对基准的因子
exprs = [
    "vs_hs300 = RET - BENCH_HS300",
    "vs_csi1000 = RET - BENCH_CSI1000",
    "benchmark_choice = case(vs_hs300 > vs_csi1000, BENCH_HS300, BENCH_CSI1000)",
]
```

---

## 扩展阶段：添加新基准

### 步骤 1：创建子类

```python
# src/alpha_factory/data_provider/benchmarks/zz500.py

from datetime import date
import polars as pl
import tushare as ts
from alpha_factory.data_provider.benchmark import Benchmark
from alpha_factory.config.base import settings

class ZZ500Benchmark(Benchmark):
    """中证500指数"""

    def __init__(self):
        super().__init__(name="ZZ500", default_start_date=date(2012, 1, 1))
        self._token = getattr(settings, "TUSHARE_TOKEN", None)
        self._is_vip = getattr(settings, "IS_VIP", True)

    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        pro = ts.pro_api(self._token)
        df = pro.index_daily(
            ts_code="000905.SH",  # 中证500 代码
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            fields="trade_date,pct_chg",
        )

        return (
            pl.from_pandas(df)
            .with_columns(
                pl.col("trade_date").str.to_date("%Y%m%d").alias("DATE"),
                (pl.col("pct_chg") / 100.0).cast(pl.Float32).alias("ret"),
            )
            .select(["DATE", "ret"])
            .sort("DATE")
        )
```

### 步骤 2：在 `__init__.py` 中暴露

```python
# src/alpha_factory/data_provider/benchmarks/__init__.py

from alpha_factory.data_provider.benchmarks.hs300 import HS300Benchmark
from alpha_factory.data_provider.benchmarks.zz500 import ZZ500Benchmark

__all__ = ["HS300Benchmark", "ZZ500Benchmark"]
```

### 步骤 3：在 data_provider 中暴露

```python
# src/alpha_factory/data_provider/__init__.py

from alpha_factory.data_provider.benchmarks import HS300Benchmark, ZZ500Benchmark

__all__ = [
    "TushareDataService",
    "DataProvider",
    "Benchmark",
    "HS300Benchmark",
    "ZZ500Benchmark",
]
```

### 步骤 4：使用

```python
from alpha_factory.data_provider import ZZ500Benchmark

bench = ZZ500Benchmark()
bench.update()
lf = pool.join_benchmark(lf, bench, start_dt, end_dt, col_name="ZZ500_RET")
```

---

## API 参考

### `Benchmark` 类

#### 方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `update` | `update(end_date: Optional[date] = None)` | 增量同步数据至指定日期，默认为今天 |
| `load_returns` | `load_returns(start_date: date, end_date: date) -> pl.LazyFrame` | 加载指定范围的基准收益率 |
| `_fetch` | `_fetch(start_date: date, end_date: date) -> pl.DataFrame` | 抽象方法，子类实现具体数据源 |
| `load` | `load()` | 向后兼容方法，返回 self |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 基准名称 |
| `_cache_path` | Path | 本地缓存文件路径 |
| `_default_start_date` | date | 历史数据起始日期 |

### `PoolUniverse.join_benchmark()` 方法

```python
def join_benchmark(
    self,
    lf: pl.LazyFrame,
    benchmark: Benchmark,
    start_date: date,
    end_date: date,
    col_name: str = "BENCH_RET",
) -> pl.LazyFrame:
    """
    Args:
        lf: 因子 LazyFrame（必须含 DATE 列）
        benchmark: Benchmark 实例
        start_date: 基准数据起始日期
        end_date: 基准数据结束日期
        col_name: 新增列名，默认 "BENCH_RET"

    Returns:
        新增 col_name 列的 LazyFrame
    """
```

### `generate_and_open_report()` 函数

```python
def generate_and_open_report(
    result: Dict[str, Any],
    factor_name: str,
    benchmark: Optional[Benchmark] = None,
    start_date: Optional[str] = None,  # YYYYMMDD
    end_date: Optional[str] = None,    # YYYYMMDD
):
    """
    Args:
        result: 回测结果字典，包含 'series' 键
        factor_name: 因子名称
        benchmark: 可选的 Benchmark 实例
        start_date: 基准数据起始日期（格式：YYYYMMDD）
        end_date: 基准数据结束日期（格式：YYYYMMDD）

    Returns:
        None（自动生成并打开 HTML 报告）
    """
```

---

## 常见问题

### Q1: Benchmark 数据更新的频率如何？

**A**: 根据需要手动调用 `benchmark.update()`。建议：
- 开发阶段：每周更新一次
- 生产环境：每日定时任务更新（例如交易日 18:00）

### Q2: 能否使用自定义数据源（非 Tushare）？

**A**: 完全支持。只需继承 `Benchmark` 基类，实现 `_fetch()` 方法：

```python
class CustomBenchmark(Benchmark):
    def _fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        # 从你的数据源获取数据
        df = load_from_my_database(start_date, end_date)
        return pl.from_pandas(df).select(["DATE", "ret"])
```

### Q3: 基准数据占多少磁盘空间？

**A**: 取决于数据跨度和精度：
- HS300（15 年历史）：约 24KB（Parquet 压缩）
- 一年数据：约 2-3KB

### Q4: join_benchmark 会影响性能吗？

**A**: 否。Polars 的 LazyFrame join 是下压优化的，只加载需要的列和日期，性能影响微乎其微。

### Q5: 如何处理基准数据缺失（停牌、休市）？

**A**:
- `load_returns()` 返回的是稀疏时间序列（仅包含有数据的日期）
- join 时采用 left-join，股票表中无对应基准日期的行会填充 null
- 可以使用 `fill_null()` 或 `forward_fill()` 处理

```python
lf = pool.join_benchmark(lf, bench, start_dt, end_dt)
lf = lf.with_columns(pl.col("BENCH_RET").forward_fill().over("DATE"))
```

---

## 测试

运行 benchmark 相关测试：
```bash
uv run pytest tests/test_benchmark.py -v
```

演示脚本：
```bash
uv run python scripts/demo_benchmark.py
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/alpha_factory/data_provider/benchmark.py` | 基类定义 |
| `src/alpha_factory/data_provider/benchmarks/hs300.py` | HS300 实现 |
| `src/alpha_factory/data_provider/benchmarks/__init__.py` | 模块暴露 |
| `src/alpha_factory/data_provider/pool.py` | Pool join_benchmark 方法 |
| `src/alpha_factory/evaluation/backtest/utils.py` | 评估层集成 |
| `tests/test_benchmark.py` | 单元测试 |
| `scripts/demo_benchmark.py` | 演示脚本 |

---

## 后续改进方向

1. **多基准支持**：实现 ZZ500、CSI1000、创业板指等
2. **本地化基准**：支持自定义合成基准（多基准加权组合）
3. **择时模块**：基于基准的动态头寸调整
4. **性能对标**：自动计算 Sharpe Ratio、Information Ratio 等相对指标
5. **风险管理**：基于基准的最大跟踪误差控制
