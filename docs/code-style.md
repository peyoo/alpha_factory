# Alpha-Factory 代码风格规范 (Code Style)

**版本**: 1.0  
**适用对象**: AI 助手 (Cursor/Copilot) 及 核心开发人员  
**核心原则**: **性能至上 (Polars)**、**类型安全**、**显式优于隐式**。

## 1. 核心技术准则 (Core Tech Stack)
- **计算库**: 强制使用 `polars`。禁止引入 pandas（除非第三方库强制要求输入）。
- **Python 版本**: 必须兼容 Python 3.12+ 语法。
- **异步/并行**: 涉及磁盘 IO 或网络请求（Tushare）必须考虑 `concurrent.futures`。

## 2. 命名规范 (Naming Conventions)

### 2.1 变量与函数
- **变量/函数**: `snake_case` (下划线命名)。
- **类名**: `PascalCase` (大驼峰命名)。
- **常量**: `UPPER_SNAKE_CASE` (大写下划线)。

### 2.2 量化字段契约 (The Contract)
所有 Polars DataFrame 必须遵循以下列名规范，以确保模块间解耦：
- **索引列**: `_DATE_` (日期类型), `_ASSET_` (字符串代码)。
- **基础价量**: `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOLUME`, `AMOUNT` (全部大写)。
- **特征/因子**: 前缀 `f_` (如 `f_rsi_14`)。
- **标签/目标**: 前缀 `target_` (如 `target_return_5d`)。

## 3. Polars 编程风格 (Polars Best Practices)

### 3.1 表达式优先
- **禁止使用循环**: 严禁使用 Python `for` 循环遍历行。
- **禁止 .apply()**: 除非无法用表达式实现，否则禁止使用 `map_elements`。
- **链式调用**: 推荐使用链式调用，逻辑复杂的步骤需换行。

### 3.2 内存与性能
- **显式 Schema**: 读取数据时应指定 `dtypes` 或使用 `schema_overrides`。
- **类型选择**:
    - 价格/成交量使用 `pl.Float32` 以节省空间。
    - 股票代码使用 `pl.Categorical` 或 `pl.Enum`。
- **Lazy 执行**: 函数应尽量接收并返回 `pl.LazyFrame`，由调用者决定何时 `.collect()`。

## 4. 模块结构与 IO (Architecture & IO)

### 4.1 路径管理
- **禁止硬编码字符串**: 严禁直接书写如 `"./data/..."`。
- **必须调用设置**: 统一使用 `from src.utils.config import settings`。
- **Pathlib**: 强制使用 `pathlib.Path` 进行路径拼接。

### 4.2 配置管理
- **环境变量**: 使用 `pydantic-settings` 自动加载 `.env`。
- **Secrets**: 严禁将 `TUSHARE_TOKEN` 写在代码中。

## 5. 文档与测试 (Docs & Testing)

### 5.1 函数定义模板
```python
def fetch_tushare_data(
    ts_code: str, 
    start_date: str, 
    end_date: str
) -> pl.LazyFrame:
    """从 Tushare 获取原始数据并转换为 LazyFrame。

    Args:
        ts_code: 股票代码 (例如 '000001.SZ')
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        包含基础行情的 Polars LazyFrame。
    """
    ...
```

### 5.2 日志规范
- 使用 `loguru` 代替 `print`。
- 关键路径必须有 `logger.info`；计算密集型内部使用 `logger.debug`。
