# 因子开发平台 实施计划

**目标：** 在现有仓库中搭建一套可复现、可测试、可扩展的量化因子开发与评估流水线，包含数据同步、基于遗传算法的因子挖掘、批量与单因子评估，以及深度学习驱动的因子选择与合成。

**架构概览：** 使用模块化 CLI 与脚本（`src/alpha_factory/cli/factor.py`），统一数据接口（`data_loader`），遗传因子挖掘模块（`gp/`），批量评估与报告生成（复用 `backtest/` 与 `evaluation/`），以及深度学习模型服务（轻量训练/推理脚本）。

**技术栈：** Python 3.11+, `polars`（数据，强制使用 lazy API 与内存友好流程）、`typer`（CLI）、`pytest`（测试）、`pytorch` 或 `tensorflow`（深度学习），现有回测代码复用。

**数据处理要求：** 所有数据加载、清洗与因子计算首选使用 `polars` lazy API；对 HDF5 源数据建议转换为 `parquet` 以获得更好的 `polars` 性能与可分片处理能力。测试与 CI 使用小样本 `parquet` fixture 加速。

---

## 必要功能清单（MVP）
- 数据更新与同步：`scripts/sync_data.py`（或扩展 `load_data.py`），包含数据校验与增量更新。
- 遗传算法因子挖掘：`src/alpha_factory/gp_mining.py`（集成 `gp/` 中逻辑，输出因子 candidates metadata + score 文件）。
- 因子批量评估：`src/alpha_factory/cli/factor.py run --batch`（并行/分片执行，输出汇总CSV/HTML报告）。
- 单因子详细评估：`src/alpha_factory/evaluation/detailed.py`（IC、RankIC、分组收益、turnover、时序图）。
- 深度学习选取与合成：`src/alpha_factory/ml/factor_selector.py`（训练脚本 + 推理，用于筛选/组合候选因子）。

## 可选增强功能（Roadmap）
- 因子元数据与版本管理（JSON/YAML + hash）
- 因子市场/共享库（内部存储与检索）
- Notebook / Web UI 快速预览
- CI 轻量回测（使用小样本数据）

---

## 任务分解（每步均细化为 2–10 分钟子步骤，可直接逐项执行）

任务 A：数据更新与同步（`scripts/sync_data.py`）
- 文件：创建 `scripts/sync_data.py`；修改：无
- 步骤 1：编写失败测试 `tests/test_sync_data.py::test_sync_incremental`
- 步骤 2：实现增量读取与写入（使用 `polars` 读取 HDF5 或转换为 parquet）
- 步骤 3：运行测试并修正
- 步骤 4：添加 CLI 入口 `quant data sync`（`typer`）

任务 B：遗传算法因子挖掘（`src/alpha_factory/gp_mining.py`）
- 文件：创建 `src/alpha_factory/gp_mining.py`，示例配置 `examples/gp_config.yaml`
- 步骤 1：编写失败测试 `tests/test_gp_mining.py::test_generate_candidates_shape`
- 步骤 2：封装数据接口以供 GP 调用（日期区间、股票池）
- 步骤 3：集成现有 `gp/` 生成器，输出候选因子及其 metadata（参数、hash、样本 IC）
- 步骤 4：运行小样本以验证输出文件

任务 C：批量评估与报告（`src/alpha_factory/cli/factor.py run --batch`）
- 文件：创建/修改 `src/alpha_factory/cli/factor.py`，添加 `run --batch`、`report` 子命令
- 步骤 1：编写失败测试 `tests/test_cli_factor.py::test_batch_run_cli_help`
- 步骤 2：实现批量读取候选列表并并行提交评估任务（multiprocessing 或 joblib）
- 步骤 3：合并结果并生成汇总 HTML/CSV（复用 `html_reports/` 模板）

任务 D：单因子详细评估（`src/alpha_factory/evaluation/detailed.py`）
- 文件：创建 `src/alpha_factory/evaluation/detailed.py`
- 步骤 1：编写失败测试 `tests/test_detailed_eval.py::test_ic_calculation`
- 步骤 2：实现 IC/RankIC、分组收益、turnover 的计算函数
- 步骤 3：实现绘图与 HTML 报告片段（保存至 `html_reports/`）

任务 E：深度学习因子选择与合成（`src/alpha_factory/ml/factor_selector.py`）
- 文件：创建 `src/alpha_factory/ml/factor_selector.py`，训练脚本 `examples/train_selector.py`
- 步骤 1：编写失败测试 `tests/test_selector.py::test_model_training_runs`（仅验证小样本流程）
- 步骤 2：准备训练数据接口（features matrix, target future returns）
- 步骤 3：实现训练与保存模型（支持 PyTorch/TorchScript 或 ONNX 导出）
- 步骤 4：集成到批量评估流水线，使用模型对候选因子进行打分与合成

### 共同约束与测试
- 所有新增模块必须带类型注解并包含 `pytest` 测试。
- 增量开发：先在小样本数据上验证，再扩展到全量数据。
- CI 建议：创建轻量数据 fixture（`tests/fixtures/sample_data/`）供 PR 自动测试使用。

---

## 交付物（初始提交）
- `docs/plans/2026-02-22-factor-dev-platform.md`（本文件）
- CLI scaffold：`src/alpha_factory/cli/factor.py`（init/run/report 占位）
- 数据同步脚本：`scripts/sync_data.py`
- GP 挖掘模块：`src/alpha_factory/gp_mining.py`
- 评估模块：`src/alpha_factory/evaluation/detailed.py`
- ML 选择器：`src/alpha_factory/ml/factor_selector.py`
- 测试样例：若干 `tests/test_*.py`

---

## 时间估算（粗略）
- MVP（数据同步 + 批量评估 + 单因子评估 + GP 输出）：约 2–3 周（单人）
- 加入深度学习选择与合成：额外 1–2 周（取决于模型复杂度和数据清洗）

---

## 下一步（建议操作）
1. 我可以现在为您：
   - A1: 直接在仓库中添加计划文件（已完成）并生成 CLI scaffold 补丁；
   - A2: 仅添加计划文件（已完成），等待您选择下一步实现；
   - A3: 生成按 2–5 分钟粒度的详细 TODO 列表并开始实现第一项（例如 `scripts/sync_data.py`）。

请在 A1/A2/A3 中选择或给出其它指示。
