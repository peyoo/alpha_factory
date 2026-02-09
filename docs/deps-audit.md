# 依赖审计报告 (初版)

- 生成时间: 2026-02-09
- 目标 Python 版本: 3.11
- 目标平台: linux-64, osx-64, osx-arm64

> 说明：本报告为 Step 1 的初步静态审计结果，基于仓库根 `requirements.txt` 与对 `alpha/` 下源码的静态 import 扫描得到的候选依赖清单。原计划通过在隔离 venv 与 Conda 环境中逐项做 `pip --no-cache-dir` 与 `conda-forge` 安装探测（并记录日志），但在当前运行代理环境下部分自动化探测未能完成或日志不完整。请把本文件作为迁移的“起点”，我已在 `output/` 下预置了探测脚本与日志位置建议，下一步建议在本地运行完整探测（或允许我在你的工作区继续执行完整探测）。

---

## 工具与版本（运行代理时记录）

```
(注：以下为探测时收集的工具信息，若为空请在本地重新执行探测脚本以获得完整日志)
```

从 `output/tool-versions.txt`（若存在）摘录示例：

```
Python 3.11.13
/Users/yongpeng/opt/anaconda3/envs/alpha_py311/bin/python
pip 25.3 (python 3.11)
conda 4.14.0
mamba: 未检测到（如本机有 mamba 建议优先使用）
poetry: 未检测到（请在本地安装）
conda-lock: 未检测到（若需要生成 conda-lock.yml 请安装）
```

---

## 说明和判定规则（简要）

- verdict: poetry = 建议交给 Poetry (pyproject.toml) 管理；conda = 建议由 Conda 提供（写入 environment.yml）；conda-forced = 明确强制为 Conda（如 TA-Lib、cvxpy）；manual-check = 需手工复核/平台测试（可能存在导入名差异或仅在特定平台有 wheel）。
- 我在审计中特别关注二进制依赖（例如 TA-Lib、cvxpy、某些 SciPy/BLAS 相关包、pytorch/cuda 生态），并将它们 preferentially 标为 Conda。
- 本报告未包含真实的 pip/conda 探测日志（若需完整探测请在本地运行 `scripts` 下的探测脚本或允许我继续本地执行）。

---

## 汇总表（来自 `requirements.txt` 与静态导入）

| package | version_spec (requirements.txt) | detected_from | verdict | notes | suggested_channel |
|---|---:|---|---|---|---|
| polars | >=0.20.0 | requirements.txt | poetry | Polars 在 pip/conda 均有分发，建议由 Poetry 管理 | pypi / conda-forge (optional) |
| pydantic-settings | >=2.0.0 | requirements.txt | poetry | 纯 Python 包，Poetry 管理 | pypi |
| loguru | >=0.7.0 | requirements.txt | poetry | 纯 Python，Poetry 管理 | pypi |
| polars-ta | >=0.1.0 | requirements.txt | poetry | 基于 Polars 的扩展库 | pypi |
| numpy | >=1.24.0 | requirements.txt | manual-check | NumPy 与 BLAS/MKL 有平台依赖；pip wheel 在多数平台可用，但若需要特定 BLAS 建议用 Conda | pypi / conda-forge |
| pandas | >=2.0.0 | requirements.txt | poetry | pip wheel 可用 | pypi |
| tushare | >=1.2.89 | requirements.txt | poetry | 纯 Python 与网络接口 | pypi |
| pyarrow | >=14.0.0 | requirements.txt | manual-check | pyarrow 常依赖平台二进制，conda-forge 提供稳定二进制包；建议对目标平台验证 | conda-forge (recommended) |
| deap | >=1.4.0 | requirements.txt | poetry | 纯 Python，多为源代码 | pypi |
| lightgbm | >=4.0.0 | requirements.txt | manual-check | lightgbm 可能需要编译或使用 conda 提供的二进制，建议验证 | conda-forge |
| scikit-learn | >=1.3.0 | requirements.txt | manual-check | 含二进制依赖，pip wheels 通常可用，若需针对特定 BLAS/MKL 使用 Conda | pypi / conda-forge |
| pytest | >=7.0.0 | requirements.txt | poetry | dev dependency | pypi |
| alphainspect | (no spec) | requirements.txt | manual-check | 未在 PyPI 普遍可见，需确认来源 | manual-check |
| pre-commit | >=3.5.0 | requirements.txt | poetry | dev tooling | pypi |
| ruff | >=0.1.0 | requirements.txt | poetry | dev tooling (format/lint)；注意 ruff 版本与 pre-commit 配置的兼容性 | pypi |
| sympy | ~=1.14.0 | requirements.txt | poetry | 纯 Python | pypi |
| more-itertools | ~=10.8.0 | requirements.txt | poetry | 纯 Python | pypi |
| fastcluster | ~=1.3.0 | requirements.txt | manual-check | C 扩展包，pip 有时提供 wheel，但部分平台可能需要 conda | conda-forge (verify) |
| scipy | ~=1.17.0 | requirements.txt | manual-check | 含 Fortran/C/BLAS 绑定，推荐通过 conda-forge 安装以获得稳定二进制 | conda-forge (recommended) |
| tensorboardx | ~=2.6.4 | requirements.txt | poetry | 纯 Python 封装 | pypi |
| pydantic | ~=2.11.7 | requirements.txt | poetry | 纯 Python | pypi |
| quantstats | ~=0.0.81 | requirements.txt | manual-check | 依赖 pandas/scipy 等，需验证 | pypi / conda-forge |


## 静态扫描到的常见 imports（样例）

（注：只列出 alpha/ 下出现频率较高或与本仓库直接相关的 top-level imports）

- polars
- numpy
- pandas
- scipy
- fastcluster
- loguru
- deap
- lightgbm
- tensorboardX

---

## 强制保留在 Conda 列表的包（用户特别要求）

- TA-Lib (talib)：需依赖 C 库 `ta-lib`，强烈建议通过 `conda-forge` 安装并列入 `environment.yml`。
- cvxpy：含复杂的 C/C++/BLAS 绑定与编译步骤，建议使用 `conda-forge` 的二进制分发。

> 建议：在 `environment.yml` 中把上述包列为 Conda 依赖（channel: conda-forge），并在 Poetry 的 `pyproject.toml` 中不列出它们，或在 `[tool.poetry.extras]` 中单独说明。

---

## 后续建议（Step 2 列表）

1. 在你本机（或我继续本地 Mode B 完整运行）按原计划运行探测脚本（`pip --no-cache-dir` 与 `conda-forge` 安装尝试），以产出真实的 `output/pip-audit.json` 与 `output/conda-audit.json`。这些文件将写入 `output/` 供审阅。
2. 依据探测结果生成 `pyproject.toml`（Poetry）与 `environment.yml`（Conda），并确保 `python = "3.11"` 在两个文件中对齐。
3. 运行 `conda-lock` 生成 `conda-lock.yml`（目标平台：linux-64, osx-64, osx-arm64）。
4. 更新 CI（GitHub Actions）：先用 `mamba`/`conda` 安装 `environment.yml`，再在同一环境内运行 `poetry install`（`poetry config virtualenvs.create false --local`），最后运行 `pre-commit` 与 `pytest`。

---

## 附录：本地自动探测脚本（建议在本地执行以获得完整日志）

我已在审计计划中准备好用于探测的脚本片段（会在 `output/` 生成 `pip-audit.json` 与 `conda-audit.json`），示例脚本位于 PRD 列表/聊天记录中。如需我继续在本地完整运行探测并提交 `poetry`/`environment.yml` 生成补丁，请授权我在 Mode B 下继续执行 Step 2（我会在本地创建分支并提交但不 push）。

---

报告结束
