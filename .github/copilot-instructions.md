# 📜 Alpha-Factory 仓库指南（AI 与开发者速查）

本文件为仓库级的可执行指导：面向人类开发者与 AI 编码代理（例：Copilot）。内容简洁、可执行且与仓库约定一致。

## 目录与关键模块
- 源代码: `src/alpha_factory/`
  - CLI: `src/alpha_factory/cli/` （入口: `src/alpha_factory/cli/main.py`）
  - 数据提供与构建: `src/alpha_factory/data_provider/`（核心: `data_provider.py`, `unified_factor_builder.py`）
  - 因子/评估: `src/alpha_factory/evaluation/`（单因子分析: `single/`）
  - 遗传编程 / 表达式生成: `src/alpha_factory/gp/`
  - 回测脚本与示例: `scripts/`（例如 `script_single_factor_analysis.py`, `backtest_top_n.py`）
  - 测试: `tests/`（文件名格式 `test_*.py`）

## 代码风格与约定
- 语言: Python 3.11+
- 强制使用类型注解（Type Hints）并优先使用小而清晰的函数。
- 常用库: `polars`（Lazy 优先）、`typer`（CLI）、`rich`（终端展示）。
- 保持模块内函数可单元测试，工具函数集中在 `utils`。

参见例子: [src/alpha_factory/data_provider/data_provider.py](src/alpha_factory/data_provider/data_provider.py)

## 构建、运行与测试（必会命令）
 - 环境同步: `uv sync`
 - 运行 CLI: `uv run quant [commands]`
 - 静态检查: `uv run ruff check .`
 - 运行测试: `uv run pytest --maxfail=1 -q`

AI 代理注意：在尝试运行命令前，先检查本仓库是否激活虚拟环境（见 workspace 根目录的 `.venv`）。

## 项目结构与架构要点（“为什么”）
- 数据层（`data_provider`）负责将原始 HDF/Parquet 数据转换为统一因子表（`UnifiedFactorBuilder`），并提供质量自检接口。
- 挖掘层（`gp`）生成或搜索表达式/因子；生成结果需要可序列化到 `codegen/` 与 `html_reports/`。
- 评估层（`evaluation`）计算分位收益、换手率、成本扣除等，输出到 `reports/` 与 `html_reports/`。
- 脚本目录（`scripts/`）用于快速回测、批量分析与开发验证；生产化功能应迁移到 `cli`。

## 项目惯例（请严格遵守）
- 数据操作优先使用 `polars` 的 Lazy API，避免在主逻辑中大量 collect。
- 关键逻辑加入 `assert` 作为自检（例如：日期连续性、非空、索引一致性）。
- 所有非-trivial 更改必须配套测试（`tests/`），并使用小规模样本数据验证边界情形。

## 集成点与外部依赖
- 数据源：仓库 `data/raw/` 包含 HDF 数据（示例: `daily.h5` 等），ETL 输出建议写入 `data/tmp_cache/warehouse/unified_factors/`。
- 报告/展示：`html_reports/` 存放自动化生成的可视化报告。

## 安全与敏感数据
- 仓库内不要提交凭证、API Key 或数据库密码。
- 数据目录仅包含示例或脱敏数据，真实数据应由运维在 CI/CD 环境注入。

## 对 AI 编码代理的具体指导
- 在更改任何核心模块前，先运行并报告现有测试结果（`uv run pytest`）。
- 对于建议的代码变更，先生成小规模补丁并附带相应单元测试；在提交前运行 `uv run ruff check .` 与 `uv run pytest`。
- 引用并复用现有函数与约定（参考上文关键文件路径）。
- 当信息不明确时，提出不超过三项澄清问题再继续实现。

## 快速反馈与迭代
- 修改本文件请保持 <60 行，聚焦可执行内容；如果需要更长的指南，创建 `docs/agent-guides.md` 并在此处创建摘要与链接。

---
