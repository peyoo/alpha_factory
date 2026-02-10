# Git Commit 指令规范

## 核心原则
- **语言**：使用中文。
- **格式**：严格遵守 Conventional Commits 规范。
- **结构**：`<类型>(<范围>): <描述>`。

## 类型定义 (Type)
- `feat`: 新增量化因子、新 Skill 或新功能。
- `fix`: 修复代码 Bug、因子计算错误或连接中断问题。
- `refactor`: 代码重构（未改变功能，如优化 OpenClaw 插件结构）。
- `perf`: 提高性能（如优化向量化计算、减少 API 调用延迟）。
- `docs`: 修改文档、注释或 SKILL.md。
- `chore`: 更新依赖 (uv/pnpm)、修改 .gitignore。

## 范围定义 (Scope)
- `factor`: 涉及因子计算逻辑。
- `mcp`: 涉及 MCP Server 或工具协议。
- `gateway`: 涉及 OpenClaw 核心网关。
- `deps`: 涉及依赖管理。

## 写作要求
1. **简洁有力**：第一行（Subject）不超过 50 个字符。
2. **描述变动原因**：如果变动复杂，在正文中说明“为什么要改”，而不仅仅是“改了什么”。
3. **禁止废话**：不要包含 "Initial commit", "Fixed bug" 这种无意义的话。
4. **量化特化**：如果修改了因子，请注明因子名称（如：feat(factor): 引入 alpha-101 波动率信号）。
