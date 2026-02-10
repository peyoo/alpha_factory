# 依赖迁移与审计报告 (uv) — Migration Audit

日期: 2026-02-09

概述
----
本次迁移将项目从原先的 (Conda + legacy-tool) 混合依赖管理，迁移到 `uv`（使用 PyPI wheels + `uv` 的 lock/sync），并对部分二进制/复杂编译包保留审慎处理策略（将 `ta-lib` 视为需由 Conda/系统包特殊安装）。本报告记录了决策依据、跨平台锁摘要、验证步骤与 PyCharm 使用建议。

1) 断/舍/离 决策（为什么移除 `ta-lib`）
-------------------------------------
- 原因：`ta-lib` 在 PyPI 上多依赖系统 C 库（TA-Lib C），在不同平台上构建耗时且易失败；对于可在 pip 上安装的项目，若依赖库可用 wheel（无编译）则迁移顺利，否则会导致 `uv sync` 构建失败或强制本地编译。
- 决策：将 `ta-lib` 从 `pyproject.toml` 移除，保留 `environment.yml`（放入 `backup/migrate-to-uv/`）作为历史与“若需 Conda 安装”参考。推荐的安装策略为：对需要系统包的二进制依赖（如 ta-lib、某些 LAPACK/MKL 绑定）在生产/CI 环境使用 Conda/系统包管理安装，而开发者本地可选择 apt/brew 或 Conda 场景安装。

如何缓解 `quantstats` 中的 `cvxpy` 编译压力
------------------------------------------
- 现象：`quantstats` 可能间接依赖或与 `cvxpy` 一起使用，`cvxpy` 曾因含有可选的本地求解器或 C 扩展（如 OSQP、SCS）而导致编译/链接问题。
- 措施：
  1. 在 `pyproject.toml` 中显式添加 `cvxpy` 作为依赖，并在 `uv.lock` 中允许解析为已有 wheel 的稳定版本（本次解析得到 `cvxpy==1.7.5`），优先使用 PyPI 上的 wheel（macOS universal2 / manylinux aarch64 / x86_64）以避免源码编译。 uv 在解算时已为 `cvxpy` 选取 pre-built wheels（参见 `uv.lock`）。
  2. 对于仍需系统库支持的求解器（比如某些 BLAS、MKL 或 ta-lib），建议在 CI/部署时用 Conda 提供系统级二进制依赖；uv 管理纯 Python 轮子和大多数预构建 binary wheel，减少编译需求。

2) 多平台锁（B）核心摘要
-------------------------
- 目标平台：
  - x86_64-unknown-linux-gnu (Linux x86_64)
  - x86_64-apple-darwin (macOS Intel)
  - aarch64-apple-darwin (macOS Apple Silicon)

- 结果摘要（来自仓库根 `uv.lock`，已解析）：
  - `quantstats` resolved version: 0.0.81 — wheel: universal py3 (pure python) (works on all platforms)
  - `cvxpy` resolved version: 1.7.5 — available wheels included:
    - macOS universal2 wheel (e.g. `cvxpy-1.7.5-cp311-cp311-macosx_10_9_universal2.whl`)
    - macOS x86_64 wheel (e.g. `cvxpy-1.7.5-cp311-cp311-macosx_10_9_x86_64.whl`)
    - manylinux aarch64 wheel (e.g. `cvxpy-1.7.5-cp311-cp311-manylinux_2_24_aarch64.whl`)
    - manylinux x86_64 wheel (e.g. `cvxpy-1.7.5-cp311-cp311-manylinux_2_24_x86_64.whl`)
  - 结论：`uv.lock` 已包含跨平台的预构建 wheel（macOS universal / linux manylinux），能覆盖目标三平台。实测 `uv sync` 在 macOS (Intel) 创建 `.venv` 并成功安装 `cvxpy==1.7.5` 与 `quantstats==0.0.81`（见 `output/uv-sync.log` 与 `output/import-test.log`）。

兼容性注意事项与风险
- 虽然 `uv.lock` 列出了跨平台 wheel，但实际在不同平台上仍可能遇到特殊系统库缺失（例如 `cvxpy` 的可选求解器或底层 BLAS/OSQP 的系统依赖）。因此建议在目标部署环境执行一次 `uv sync` 以验证完整安装。若某平台上缺少 wheel，uv 将回退为源码构建，可能失败。

3) 操作步骤（如何在三平台生成/验证锁）
--------------------------------------
- macOS (本机 Intel) — 已完成：
  - python -m uv lock  -> 生成 `uv.lock`（含多平台 wheel 列表）
  - python -m uv sync  -> 在 `.venv` 成功安装并测试 import（quantstats, cvxpy）
- macOS (Apple Silicon) — 验证建议：
  - 在 Apple Silicon 机器上运行同样命令（uv lock 或 uv sync）以确认本地解析/安装。macOS universal2 wheel 通常支持两类 Mac CPU。若无本机可用，可在 CI 上跑 arm64 runner。
- Linux x86_64 — 验证建议（CI / Docker）：
  - 在 linux x86_64 runner 或 Docker 镜像中运行 `python -m uv lock` 或 `uv sync` 来验证 manylinux wheel 可用性。

示例（复现）命令
```bash
# 在 repo 根
python -m uv lock       # update uv.lock (reads pyproject.toml / [project])
python -m uv sync       # create/update .venv and install
.venv/bin/python -c "import quantstats,cvxpy; print(quantstats.__version__, cvxpy.__version__)"
```

4) 架构对比（旧: Conda+Poetry vs 新: uv）
-----------------------------------------
- 安装速度
  - Conda: 初次创建环境（含二进制包）通常较快，因为 Conda 提供了预编译的二进制包，但对某些包（非 Conda 仓库）仍需构建。
  - legacy-tool+pip: 取决于 PyPI wheel 是否可用，源码构建会慢很多。legacy-tool 的 dependency solver 在部分版本上通常较慢于 uv（取决于具体实现和版本）。
  - uv: 通过解析 PyPI wheel 并尽可能使用 pre-built wheels，通常在解析/锁定速度上比纯 Poetry 快（本次 uv lock/resolution 在本机为数秒，uv sync 完成安装约 27s）

- 磁盘占用
  - Conda: 可能更大（Conda 本身、conda envs、共享库、MKL/BLAS 等）。
  - uv/pip: 更轻量，但依赖系统级库（如 OpenBLAS/MKL）在不同环境中可能需要手工安装。

- 开发流与 CI
  - 旧流 (Conda+Poetry): 适合重度二进制依赖与科学计算栈（numpy/scipy/ta-lib）统一管理，便于在服务器上维持一致性。Conda 在多平台上有更稳定的二进制保障。
  - 新流 (uv): 适合以 PyPI wheel 为主的项目，简化 lock/sync（无额外 Conda 安装），并能快速在开发者机器上复现。对于少数系统级依赖，建议在 CI 或部署时用 Conda/OS-level install 补充。

- 结论性建议
  - 对于纯 Python 或提供 universal/manylinux wheel 的依赖，`uv` 是简洁且快速的选择；
  - 对于强依赖本机 C 库（ta-lib、某些 MKL-linked packages），继续在生产部署或 CI 使用 Conda/system package 来提供这些库（混合方案）。

5) PyCharm 配置指南（如何绑定 `.venv`）
---------------------------------------
1. 打开 PyCharm，选择 Preferences / Settings -> Project -> Python Interpreter。
2. 点击齿轮 -> Add... -> Existing environment。
3. 指向仓库下的 `.venv/bin/python`（macOS/Linux）或 `.venv\Scripts\python.exe`（Windows）。
4. 确认后 PyCharm 会索引依赖；若你切换到不同分支并重建 `.venv`，在 PyCharm 中需要重新指向该解释器或刷新索引。

6) 后续建议与 TODO
-------------------
- 在 CI（github actions / GitLab runners）上配置矩阵 runner：
  - ubuntu-latest (x86_64) 执行 `python -m uv lock` / `uv sync` 验证 Linux manylinux wheels
  - macos-latest (x86_64) 与 macos-arm64 验证 Mac-specific wheels
- 把 `environment.yml` 放入 `backup/migrate-to-uv/` 作为 Conda 安装参考，并在 README.md 或 docs 中给出“若需要 Conda 安装”的一键命令示例。

附录：本地可查的关键文件
- `uv.lock` (repo root)
- `output/uv-sync.log`
- `output/import-test.log`
- `backup/migrate-to-uv/environment.yml`


报告结束。
