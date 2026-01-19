# 测试规范 (Testing Standards)

**版本**: 1.0
**最后更新**: 2026-01-19
**目的**: 定义 Alpha-Factory 项目的测试框架、覆盖范围及量化特定场景的测试策略。

---

## 1. 测试框架与工具链

### 1.1 核心工具
- **测试框架**: `pytest` (Python 原生)
- **Mock 数据**: `polars` 原生 `pl.DataFrame` 构造
- **断言库**: `assert` (Python 原生) + `pytest.approx()` (浮点比较)
- **覆盖率**: `pytest-cov`
- **性能测试**: `pytest-benchmark`

### 1.2 依赖管理
测试依赖应在 `requirements-dev.txt` 中定义：
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-benchmark>=4.0.0
```

---

## 2. 测试覆盖范围与策略

### 2.1 必测模块 (Critical Path)
所有以下模块的核心功能必须有单元测试覆盖：

| 模块 | 最小覆盖率 | 关键测试点 |
| :--- | :--- | :--- |
| `alpha.data_provider.schema` | 95% | Schema 验证、字段映射、单位转换 |
| `alpha.data_provider.cleaner` | 90% | Pandas->Polars 转换、后复权计算、空值处理 |
| `alpha.data_provider.tushare_source` | 80% | API 限流、并发下载、增量更新逻辑 |
| `alpha.data_provider.reader` | 90% | 数据加载、时间过滤、表关联 |
| `alpha.gp.*` | 85% | 因子计算、表达式树、代码生成 |
| `alpha.ml.*` | 80% | 模型训练、预测、评估 |

### 2.2 可选但推荐的模块
- `alpha.utils.config` - 配置管理
- `alpha.utils.logger` - 日志系统

---

## 3. 量化特定场景的测试 (Domain-Specific Test Cases)

### 3.1 数据质量测试

#### 3.1.1 停牌处理
```python
def test_suspend_stock_nan_handling():
    """验证停牌股票的 NaN 处理是否正确"""
    df = pl.DataFrame({
        "_DATE_": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "_ASSET_": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "CLOSE": [10.0, None, 11.0]  # 第二天停牌
    }).with_columns(pl.col("_DATE_").str.strptime(pl.Date, "%Y-%m-%d"))

    result = clean_daily_bars(df.to_pandas())
    assert result.collect()["CLOSE"].null_count() == 1, "❌ 停牌 NaN 未正确保留"
    logger.info("✓ 停牌 NaN 处理验证通过")
```

#### 3.1.2 涨跌停处理
```python
def test_limit_up_down_detection():
    """验证涨跌停检测逻辑"""
    df = pl.DataFrame({
        "_DATE_": ["2023-01-01", "2023-01-02"],
        "_ASSET_": ["000001.SZ", "000001.SZ"],
        "CLOSE": [10.0, 11.0],  # 第二天涨 10%
        "up_limit": [11.0, 11.0],
    })

    # 检查是否触及涨停
    assert (df["CLOSE"] >= df["up_limit"]).any(), "❌ 涨停检测失败"
    logger.info("✓ 涨跌停检测验证通过")
```

#### 3.1.3 退市股票处理
```python
def test_delisting_survivorship_bias():
    """验证是否正确处理了退市股票（生存偏误）"""
    df = pl.DataFrame({
        "_DATE_": ["2023-01-01", "2023-06-01", "2023-12-01"],
        "_ASSET_": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "CLOSE": [10.0, 11.0, None],  # 2023-12-01 之后退市
    })

    # 确保不会前向填充到退市后
    result = df.with_columns(
        pl.col("CLOSE").forward_fill().over("_ASSET_")
    )
    assert result[-1]["CLOSE"].is_null(), "❌ 检测到未来数据泄露"
    logger.info("✓ 退市处理验证通过")
```

### 3.2 因子计算测试

#### 3.2.1 时间对齐测试
```python
def test_factor_time_alignment():
    """验证因子计算中是否存在前向泄露"""
    df = pl.DataFrame({
        "_DATE_": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "_ASSET_": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "CLOSE": [10.0, 11.0, 12.0],
        "target_return": [0.05, 0.09, None],  # 下一日收益
    }).with_columns(pl.col("_DATE_").str.strptime(pl.Date, "%Y-%m-%d"))

    # 特征应使用当日收盘价，标签应使用次日收益
    # 验证：不应在第 3 行有标签（因为没有次日数据）
    assert df[-1]["target_return"].is_null(), "❌ 数据对齐异常"
    logger.info("✓ 时间对齐验证通过")
```

#### 3.2.2 秩变换边界测试
```python
def test_rank_transform_boundaries():
    """验证秩变换是否在 [0, 1] 范围内"""
    df = pl.DataFrame({
        "factor": [1.0, 5.0, 10.0, 2.0, 8.0]
    })

    ranked = df.with_columns(
        pl.col("factor").rank(method="average").truediv(len(df)).alias("f_rank")
    )

    assert (ranked["f_rank"] >= 0.0).all(), "❌ 秩变换下界超出"
    assert (ranked["f_rank"] <= 1.0).all(), "❌ 秩变换上界超出"
    logger.info("✓ 秩变换边界验证通过")
```

### 3.3 浮点精度测试

#### 3.3.1 Float32 精度验证
```python
def test_float32_precision():
    """验证 Float32 精度是否足够"""
    large_value = 1e8
    small_delta = 1e-6

    # 模拟大规模回测中的累积误差
    result = float(np.float32(large_value) * (1.0 + small_delta))
    expected = large_value * (1.0 + small_delta)

    # Float32 可能丢失小数点后的精度
    relative_error = abs(result - expected) / expected
    assert relative_error < 0.001, f"❌ Float32 精度不足: 相对误差 {relative_error}"
    logger.info(f"✓ Float32 精度验证通过 (相对误差: {relative_error})")
```

#### 3.3.2 对数变换精度
```python
def test_log_transform_precision():
    """验证对数变换是否保持精度"""
    prices = pl.Series([1.0, 1.1, 1.2, 1.05, 0.95])

    # 计算对数收益率
    log_returns = prices.log().diff()

    # 验证是否有 NaN 或 inf
    assert not log_returns.is_nan().any(), "❌ 对数变换产生 NaN"
    assert not log_returns.is_infinite().any(), "❌ 对数变换产生 inf"
    logger.info("✓ 对数变换精度验证通过")
```

---

## 4. 测试执行与报告

### 4.1 本地测试运行
```bash
# 运行所有测试，显示覆盖率
pytest tests/ --cov=alpha --cov-report=html

# 运行特定模块的测试
pytest tests/test_data_provider.py -v

# 运行性能基准测试
pytest tests/ -m benchmark --benchmark-only
```

### 4.2 测试报告格式
每个测试函数的输出应包含：
```
✓ [test_name] - PASS
  └─ 覆盖的核心逻辑: 字段映射、类型转换、空值处理
  └─ 执行时间: 0.05s
  └─ 测试数据规模: shape=(100, 5)
```

### 4.3 CI/CD 集成
提交前必须通过：
```bash
pytest tests/ --cov=alpha --cov-report=term-missing
```

覆盖率低于阈值则拒绝提交。

---

## 5. 测试数据集

### 5.1 Mock 数据生成工具
```python
def create_mock_daily_data(
    dates: list[str],
    assets: list[str],
    price_range: tuple = (8.0, 12.0)
) -> pl.LazyFrame:
    """生成用于测试的模拟日线数据"""
    # 见 tests/conftest.py
```

### 5.2 样本数据集
- `tests/fixtures/daily_bars_2023.parquet` - 2023 年完整日线数据（100 只股票）
- `tests/fixtures/market_status_2023.parquet` - 2023 年市场状态数据
- `tests/fixtures/calendar_2023.parquet` - 2023 年交易日历

---

## 6. TDD 工作流示例

### 6.1 新功能开发流程
```
1. 编写测试 (Write Test)
   def test_rsi_algorithm():
       input_df = create_mock_prices([50, 50.5, 51, ...])
       result = calc_rsi(input_df, period=14)
       assert result["f_rsi_14"].max() <= 100.0

2. 验证测试失败 (Red Phase)
   pytest tests/test_factors.py::test_rsi_algorithm
   # 预期：FAIL (函数未实现)

3. 实现功能 (Green Phase)
   def calc_rsi(df, period=14):
       # 实现 RSI 算法

4. 运行测试 (Verify)
   pytest tests/test_factors.py::test_rsi_algorithm
   # 预期：PASS

5. 重构代码 (Refactor)
   # 优化实现，确保测试仍通过
```

---

## 7. 量化风险检查清单 (Quantitative Risk Checklist)

在提交代码前，确保：
- [ ] 是否处理了所有 `null` 值？
- [ ] 是否检查了涨跌停/停牌情况？
- [ ] 是否验证了时间对齐（无未来数据泄露）？
- [ ] 是否处理了退市股票（生存偏误）？
- [ ] 是否验证了浮点精度（Float32 足够）？
- [ ] 是否测试了极端市场情况（全跌停、全涨停）？
- [ ] 是否运行了完整的单元测试套件？
- [ ] 是否更新了 `docs/progress.txt` 中的测试报告？

---

## 8. 常见问题 (FAQ)

**Q: 如果测试覆盖率低于阈值怎么办？**
A: 提交被拒绝。必须补充缺失的测试用例直到达到阈值。

**Q: 如何处理第三方库引入的不确定性？**
A: 使用 `pytest.mark.flaky()` 标记，但不推荐。优先重构代码降低依赖。

**Q: 是否需要性能测试？**
A: 关键路径（数据加载、因子计算）需要 Benchmark。目标：单股票日线 1000 条 < 10ms。

---

## 9. 参考资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [Polars 测试最佳实践](https://pola-rs.github.io/)
- [量化开发测试策略](https://en.wikipedia.org/wiki/Survivorship_bias)
