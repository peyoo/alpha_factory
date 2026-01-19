# 数据接入层技术规格说明书 (Data Provider)

**版本**: 1.0
**模块路径**: `alpha.data_provider`
**责任人**: AI Copilot

## 1. 概述
本模块负责建立从外部数据源（以 Tushare 为主）到本地高性能数据仓库的数据管道。其核心职责包括：API 请求调度、原始数据清洗、Schema 标准化以及 Parquet 落盘存储。

## 2. 设计原则
1.  **原子性与幂等性**：数据下载任务应支持断点续传，重复运行不应破坏现有数据。
2.  **高性能 IO**：
    *   网络 IO：使用 `concurrent.futures.ThreadPoolExecutor` 并发请求多只股票数据。
    *   磁盘 IO：利用 Polars 的 Parquet 读写能力，避免 Pandas 的低效序列化。
3.  **严格类型 (Strict Typing)**：所有输出数据必须强制符合系统的 Schema 契约，杜绝 `object` 类型。

## 3. 核心架构

### 3.1 模块结构
```text
alpha/data_provider/
├── __init__.py          # 暴露核心接口
├── tushare_source.py    # Tushare API 交互与限流逻辑
├── schema.py            # 字段映射定义与类型转换器
└── cleaner.py           # Polars 数据清洗流水线
```

### 3.2 数据流向
`Tushare HTTP API` -> `Pandas DataFrame (内存过渡)` -> `Polars DataFrame` -> `Schema 转换/清洗` -> `Parquet 文件`

> **注意**: 尽管本项目原则上禁用 Pandas，但由于 Tushare SDK 原生返回 Pandas DataFrame，此处允许作为中间过渡格式，必须并在第一时间转换为 Polars LazyFrame。

### 3.3 核心接口定义 (Public Interface)

本模块对外暴露的核心类为 `TushareDataService`，建议通过 `alpha.data_provider` 直接导入。

```python
class TushareDataService:
    """Tushare 数据接入服务"""

    def __init__(self, token: str = None, is_vip: bool = True):
        """
        初始化服务
        Args:
            token: Tushare API Token (默认读取环境变量 TUSHARE_TOKEN)
            is_vip: 是否为 VIP 账户 (默认 True，采用高并发模式)
        """

    def sync_basic_info(self) -> None:
        """同步全市场股票基础列表 (stock_basic)"""
        ...

    def sync_calendar(self, start_year: str = None, end_year: str = None) -> None:
        """同步交易日历"""
        ...

    def sync_daily_bars(self, start_date: str, end_date: str) -> None:
        """
        同步日线行情 (OHLCV)
        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
        """
        ...

    def sync_adj_factors(self, start_date: str, end_date: str) -> None:
        """同步复权因子"""
        ...

    def sync_daily_basic(self, start_date: str, end_date: str) -> None:
        """
        同步每日基础指标 (daily_basic)
        包括估值(PE/PB/PS/DV)、市值、股本、换手率等
        """
        ...

    def sync_market_status(self, start_date: str, end_date: str) -> None:
        """同步市场状态 (涨跌停、ST、停牌)"""
        ...

    def daily_update(self) -> None:
        """
        [增量更新] 自动检测最新日期，追加至当天。
        包含基础信息、行情、复权因子及市场状态的全流程更新。
        """
        ...


class DataProvider:
    """本地数据仓库读取服务"""

    def load_data(
        self,
        start_date: str,
        end_date: str,
        columns: list[str] = None
    ) -> pl.LazyFrame:
        """
        加载本地清洗后的数据 (Lazy Mode)
        自动关联行情、因子、状态等所有数据表。

        Args:
            start_date: YYYYMMDD (闭区间)
            end_date: YYYYMMDD (闭区间)
            columns: 需要加载的列名列表 (如 ['OPEN', 'CLOSE', 'pe', 'is_st'])。
                     None 表示加载所有列。
                     如果不显式包含 _DATE_ 和 _ASSET_，会自动添加。

        Returns:
            pl.LazyFrame: 已过滤时间范围并选列的 LazyFrame，准备好进行 collect()。
        """
        ...
```

## 4. 数据契约 (Data Contract)

所有存储于 `data/warehouse/` 的数据必须符合以下标准：

### 4.1 日线行情 Schema (Daily Bars)

| Tushare 原字段 | 系统标准字段 | Polars 类型 | 备注                 |
| :--- | :--- | :--- |:-------------------|
| `trade_date` | **`_DATE_`** | `pl.Date` | 核心索引，YYYY-MM-DD    |
| `ts_code` | **`_ASSET_`** | `pl.Categorical` | 核心索引，如 "000001.SZ" |
| `open` | `OPEN` | `pl.Float32` | 后复权 (可选，视配置而定)     |
| `high` | `HIGH` | `pl.Float32` |                    |
| `low` | `LOW` | `pl.Float32` |                    |
| `close` | `CLOSE` | `pl.Float32` |                    |
| `vol` | `VOLUME` | `pl.Float32` | 单位：股               |
| `amount` | `AMOUNT` | `pl.Float32` | 单位：元               |

*注：所有价格字段建议默认使用**不复权**数据，但在清洗阶段需分别存储复权因子，或根据配置直接存储前/后复权数据。初期版本简化为：存储不复权价格 + 复权因子。*

### 4.2 衍生/辅助数据 Schema

**交易日历 (Trade Calendar)**
- 存储路径: `data/warehouse/meta/calendar.parquet`
- 关键字段: `_DATE_`, `is_open` (Boolean)

**复权因子 (Adjustment Factors)**
- 存储路径: `data/warehouse/daily_adj.parquet` (或按年分区)
- 关键字段: `_DATE_`, `_ASSET_`, `adj_factor` (Float32)

**每日基础指标 (Daily Basic)**
- 存储路径: `data/warehouse/daily_basic.parquet`
- 关键字段: `_DATE_`, `_ASSET_`
- 包含字段 (Polars Type: Float32):
  - `turnover_rate`, `turnover_rate_f`: 换手率，自由流通换手率
  - `volume_ratio`: 量比
  - `pe`, `pe_ttm`: 市盈率，市盈率TTM
  - `pb`: 市净率
  - `ps`, `ps_ttm`: 市销率，市销率TTM
  - `dv_ratio`, `dv_ttm`: 股息率，股息率TTM
  - `total_share`, `float_share`, `free_share`: 总股本，流通股本，自由流通股本 (万股)
  - `total_mv`, `circ_mv`: 总市值，流通市值 (万元)

**市场状态因子 (Market Status)**
- 存储路径: `data/warehouse/daily_status.parquet`
- 包含字段:
  - `up_limit`, `down_limit`: 涨跌停价 (`pl.Float32`)
  - `is_st`: 是否 ST (`pl.Boolean`, 源自 `stock_st` 接口)
  - `is_suspended`: 是否停牌 (`pl.Boolean`, 需补全非交易日数据或标记状态)

## 5. 关键功能规格

### 5.1 全市场股票列表同步
- **功能**: 获取当前所有上市股票的基础信息。
- **输出**: `data/warehouse/stock_basic.parquet`
- **频率**: 每日一次。

### 5.2 交易日历同步 (Sync Calendar)
- **输入**: 年份范围。
- **逻辑**: 调用 Tushare `trade_cal` 接口，转换格式并存储。确保包含交易所的所有交易日信息。

### 5.3 日线行情同步 (Sync Daily)
- **输入**: 起始日期 `start_date`，结束日期 `end_date`。
- **逻辑**:
    1.  从 `stock_basic` 读取股票代码列表。
    2.  分批次（Chunk）并发请求 Tushare `daily` 接口。
    3.  每批次数据转换为 Polars DataFrame，进行类型转换和重命名。
    4.  按 `YYYY` 年份分区写入 Parquet，或按 `_ASSET_` 分桶写入（视数据量决定，初期按年份分区 `data/warehouse/daily/2023.parquet`）。

### 5.4 辅助/状态因子同步
1.  **复权因子**:
    - 调用 `adj_factor` 接口。
    - 独立存储，计算复权价格时与 OHLCV 表进行 Join。
2.  **每日基础指标**:
    - 调用 `daily_basic` 接口。
    - 包含换手率、PE/PB、市值等关键估值和流动性因子。
3.  **涨跌停价**:
    - 调用 `stk_limit` 接口 (限制较大，需分段请求)。
4.  **ST 与停牌状态**:
    - **ST**: 调用 `stock_st` 接口获取每日 ST 股票列表，生成每日 `is_st` 布尔序列。
    - **停牌**: 调用 `suspend_d` 获取停牌记录，生成每日 `is_suspended` 布尔序列。

### 5.5 错误处理与限流
- **模式区分**:
  - **VIP 模式 (默认)**: 假设拥有高权限，采用极低延迟的限流策略（或仅做错误重试），支持高并发。
  - **普通模式**: 严格遵守 Tushare 免费/积分限制（如每分钟几百次），触发限制时自动 `sleep`。
- **实现**: `RateLimiter` 根据 `is_vip` 标志动态调整阈值。

### 5.6 增量更新模式 (Incremental Update Mode)
- **场景**: 每日收盘后自动运行，追加最新数据。
- **逻辑**:
    1.  **扫描**: 读取 `data/warehouse/daily/` 下所有（或最新年份）文件，获取全局 `max(_DATE_)`。
    2.  **调度**: 设定 `start_date = max(_DATE_) + 1`，`end_date = 今日`。
    3.  **合并**:
        - 若新数据跨年（如 12月31日 -> 1月1日），自动创建新文件。
        - 若属于同一年，读取现有 Parquet -> Concat 新数据 -> 去重 (`upsert`) -> 覆盖写入。
- **一致性**: 必须保证 `_DATE_` + `_ASSET_` 唯一。

## 6. 使用示例 (Draft)

```python
from alpha.data_provider import TushareDataService

# 1. 初始化
# 自动读取环境变量 TUSHARE_TOKEN
service = TushareDataService()

# 2. 首次初始化 (历史数据落地)
# 建议分步骤执行以避免长时间阻塞
service.sync_basic_info()
service.sync_calendar(start_year="2010", end_year="2025")

# 同步最近5年的数据
service.sync_daily_bars(start_date="20200101", end_date="20241231")
service.sync_adj_factors(start_date="20200101", end_date="20241231")
service.sync_daily_basic(start_date="20200101", end_date="20241231")
service.sync_market_status(start_date="20200101", end_date="20241231")

# 3. 每日定时任务 (Crontab)
# 只需调用此方法，会自动计算增量范围并更新所有相关表
service.daily_update()


# 4. 数据读取 (Analysis/Backtest)
from alpha.data_provider import DataProvider
import polars as pl

provider = DataProvider()

# 获取指定时间段的收盘价、PE和ST状态
# 返回 LazyFrame，尚未执行计算
lf = provider.load_data(
    start_date="20230101",
    end_date="20231231",
    columns=["CLOSE", "pe", "is_st"]
)

# 执行查询
df = lf.collect()
print(df)
```

## 7. 下一步开发任务
1.  实现 `alpha.data_provider.schema` 定义常量映射。
2.  实现 `alpha.data_provider.tushare_source` 核心下载逻辑。
3.  编写单元测试验证 Tushare 连接与 Polars 转换。
