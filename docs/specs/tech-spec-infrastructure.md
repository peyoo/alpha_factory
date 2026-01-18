# 基础设施层技术规格说明书 (Infrastructure Layer)

**版本**: 1.0
**模块路径**: `alpha.utils`
**责任人**: AI Copilot

## 1. 概述
基础设施层为整个量化平台提供底层支持，不涉及具体的业务逻辑（如回测、挖掘），但负责全局的**配置管理**、**日志记录**、**环境隔离**以及**工具函数**。该层的稳定性直接决定了上层模块的开发效率和运行安全。

## 2. 核心组件设计

### 2.1 配置管理 (Configuration)
- **实现方案**: 基于 `pydantic-settings` 实现强类型的配置管理。
- **文件路径**: `alpha/utils/config.py`
- **设计模式**: 单例模式 (`settings` 对象)。
- **功能特性**:
    - **自动加载**: 优先读取环境变量，其次加载 `.env` 文件。
    - **路径解析**: 使用 `pathlib.Path` 动态计算项目根目录，避免硬编码路径。
    - **自动创建**: 初始化时自动检查并创建必要的运行时目录 (`data/`, `output/` 等)。
- **关键配置项**:
    - `BASE_DIR`: 项目根目录。
    - `DATA_DIR` / `RAW_DATA_DIR` / `WAREHOUSE_DIR`: 数据存储路径。
    - `OUTPUT_DIR` (Logs, Models, Reports): 产出物路径。
    - `TUSHARE_TOKEN`: 敏感凭证 (从环境变量读取)。

#### 代码示例
```python
from alpha.utils.config import settings

# 获取数据目录
parquet_path = settings.WAREHOUSE_DIR / "daily_bars.parquet"

# 获取 API Token
token = settings.TUSHARE_TOKEN
```

### 2.2 日志系统 (Logging)
- **实现方案**: 基于 `loguru`，替代标准库 `logging`。
- **文件路径**: `alpha/utils/logger.py`
- **功能特性**:
    - **多路输出**:
        - console (stderr): INFO 级别，绿色高亮，实时监控。
        - file (logs/): DEBUG 级别，详细记录调试信息。
    - **日志轮转 (Rotation)**: 每天 00:00 自动切割日志文件。
    - **日志保留 (Retention)**: 只保留最近 7 天的日志，自动清理旧文件。
    - **压缩 (Compression)**: 历史日志自动压缩为 `.zip` 节省空间。

#### 代码示例
```python
from loguru import logger

logger.info("Starting data sync process...")
logger.debug(f"Processing chunk: {chunk_id}")
logger.error("Download failed", exc_info=True)
```

## 3. 项目目录结构规范
系统运行依赖以下严格的目录结构，由 `settings.make_dirs()` 保证存在：

```text
alpha_factory/
├── config/                # 配置文件 (非敏感配置)
├── data/                  # 数据存储 (本地状态，Git忽略)
│   ├── raw/               # 落地原始数据 (临时缓存)
│   └── warehouse/         # 列式存储数据库 (Parquet)
│       ├── daily/         # 日线行情
│       ├── meta/          # 基础信息/日历
│       └── factors/       # 因子库
├── output/                # 运行时产出 (Git忽略)
│   ├── logs/              # 运行日志
│   ├── codegen/           # GP 生成的代码
│   └── reports/           # HTML 分析报告
└── .env                   # 敏感环境变量 (Git忽略)
```

## 4. 环境与依赖管理
- **Python 版本**: 3.11+
- **核心依赖**:
    - `pydantic-settings`: 配置管理。
    - `loguru`: 日志管理。
    - `polars`: 核心数据结构。
- **环境隔离**:
    - 使用 `.gitignore` 排除 `data/`, `output/`, `.env` 等文件，确保代码库纯净和安全。

## 5. 开发指引
1.  **新增配置**: 在 `alpha.utils.config.Settings` 类中添加字段，并给定默认值或在 `.env` 中设置。
2.  **路径引用**: 严禁使用字符串拼接路径，必须通过 `settings.XXX_DIR / "filename"` 的方式。
3.  **日志打印**: 禁止使用 `print()`，生产环境代码必须使用 `logger`。
