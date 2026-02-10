# 📜 Alpha-Factory 仓库指南 (宪法)

## 📂 项目结构 & 模块组织
- **源代码**: `src/alpha_factory/`
  - CLI相关代码在 `cli/`
  - 数据处理在 `data/`
  - 因子评估在 `evaluation/`
  - 回测相关代码在 `backtest/`
  - 遗传算法相关在 `gp/`
  - 机器学习相关在 `ml/`
  - 公共工具函数在 `utils/`
- **测试**: `tests/` (文件名格式 `test_*.py`)。
- **文档**: `docs/` (进度记录 `progress.txt`)。

## 🛠️ 构建、测试与开发命令
- **环境同步**: `uv sync`
- **运行 CLI**: `uv run quant [commands]`
- **代码检查**: `uv run ruff check .`
- **执行测试**: `uv run pytest` (带覆盖率: `uv run pytest --cov`)

## ⌨️ 编码风格 & 命名约定
- **语言**: Python 3.11+。强制使用类型标注 (Type Hints)。
- **核心库**: `polars` (Lazy 优先), `typer` (CLI), `rich` (UI)。
- **原则**: 保持文件简洁；工具函数提取至 `utils.py`；关键逻辑必须包含 `assert` 自检。

## 🧪 测试指南
- **框架**: `pytest`。
- **流程**: 修改逻辑后，必须运行 `uv run pytest` 确保回归正常。

## 📝 提交 & 协作准则
- **提交信息**: 遵循简洁、面向操作的 Conventional Commits (如 `feat(cli): add sync command [test]`)。
- **原子交付**: **一次只交付一个文件的代码块**。严禁预测未来步骤。
- **同步验证**: 给出代码块 -> 厂长物理验证 -> 确认后进入下一步。禁止输出冗长的 TDD 计划文档。
