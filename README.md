# Alpha Factory — 量化核心算子工厂

本仓库为 Alpha Factory 项目的源码与说明，面向量化因子研究与高性能算子开发（Python 3.11+, Polars 优先）。本文档提供使用 uv 管理依赖和快速上手的指引。

## 先决条件
- Python 3.11.x（项目在 `pyproject.toml` 中要求 `==3.11.*`）
- 已安装 `uv`（请参照 uv 官方安装说明）

## 快速开始（uv）
1. 克隆仓库并进入目录：

```bash
git clone <repo-url>
cd alpha_factory
```

2. 初始化/生成锁并同步依赖：

```bash
# 如首次使用，初始化 uv（可选）
uv init

# 生成或更新锁文件（uv.lock）
uv lock

# 同步依赖并创建/更新本地虚拟环境 `.venv`
uv sync
```

3. 运行测试：

```bash
# 使用 uv 运行 pytest
uv run pytest -q
```

## 常用命令对照（Poetry -> uv）
- 添加依赖：
  - 旧：`poetry add <pkg>`
  - 新：`uv add <pkg>`
- 运行命令：
  - 旧：`poetry run <cmd>`
  - 新：`uv run <cmd>`

## 开发者工作流建议
1. 新建分支：
```bash
git checkout -b feat/your-feature
```
2. 同步依赖并进入虚拟环境（可选激活）：
```bash
uv sync
source .venv/bin/activate
```
3. 运行本地测试：
```bash
uv run pytest -q
```

## CI 推荐片段（示例）
在 CI 中推荐使用 `python -m uv` 调用以避免 PATH 问题：

```bash
python -m uv lock
python -m uv sync
python -m uv run pytest -q
```

## 注意事项
- 系统级二进制依赖（如 `ta-lib`、某些 MKL/BLAS 绑定）可能无法在所有平台通过 PyPI wheel 安装。若遇到此类包，请在 CI/部署环境使用 Conda 或系统包管理器安装所需系统依赖；历史的 Conda 参考文件已保留在 `backup/migrate-to-uv/`。
- `pyproject.toml` 已迁移为 PEP-621 的 `[project]` 格式，仓库中使用 `uv.lock` 作为依赖锁定文件。

## 参考文档
- `docs/PRD.md`, `docs/code-style.md`, `docs/deps-audit.md`
