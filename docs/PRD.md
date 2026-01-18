# Alpha-Factory：高性能自动化量化研发平台需求文档

## 1. 项目背景与目标
在传统量化开发中，Python 的 Pandas 性能瓶颈、因子挖掘的盲目性以及回测链路的脱节是核心痛点。本平台旨在通过 **Polars** 的并行计算能力和 **DEAP** 的遗传编程，构建一条从 **Tushare 数据接入** 到 **AlphaInspect 评估回测** 的全自动流水线。

## 2. 系统架构设计
系统分为五个核心层级，采用松耦合设计，确保每一层都可独立优化或替换。

### 2.1 数据接入层 (Data Ingestion) - 基于 Tushare
- **API 调度器**：支持 Tushare Pro 接口，具备频次限制自动排队与断点续传功能。
- **Parquet 仓库**：摒弃 CSV/SQL，所有行情、财务、指标数据统一存储为 Apache Parquet 格式，按年份或资产类型分区。
- **数据清洗引擎**：利用 Polars 显式处理复权（前/后复权）、停牌填充、除权除息调整。

### 2.2 计算引擎层 (Calculation Engine) - 基于 Polars & polars_ta
- **矢量化内核**：利用 `polars_ta` 实现基础算子（SMA, RSI, MACD 等），确保计算全程不离开 C++/Rust 层。
- **表达式翻译器**：将字符串公式（如 `close/ts_mean(close, 20)`）通过 `expr_codegen` 编译为高效的 Polars 表达式树。
- **Lazy 执行流**：构建计算图，仅在数据落地前触发一次 `.collect()`，最大化并行效率。

### 2.3 自动挖掘层 (Alpha Mining) - 基于 DEAP & expr_codegen
- **符号回归引擎**：使用 DEAP 构建符号树，定义变异、交叉与选择算子。
- **适应度评估 (Fitness)**：以 RankIC、ICIR、周转率 为核心适应度指标。
- **代码生成器**：`expr_codegen` 负责将最优进化个体持久化为 `output/codegen/` 下的可执行 Python 模块。

### 2.4 模型训练层 (ML Learning) - 基于 LightGBM
- **特征工程库**：自动将挖掘出的多因子对齐并拼接。
- **非线性合成**：利用 LightGBM 处理因子间的非线性关系，支持滚动训练（Rolling window）与验证。
- **中性化插件**：在模型训练前后，提供行业中性化、市值中性化处理。

### 2.5 审计评估层 (Evaluation) - 基于 AlphaInspect
- **绩效归因**：生成 IC 衰减图、分层累计收益曲线、最大回撤分析。
- **热力图检查**：多因子相关性热力图，防止过拟合与特征冗余。

## 3. 详细目录规范
```text
alpha_factory/
├── config/                # 环境变量与 API Tokens (config.yaml, .env)
├── docs/                  # 文档体系 (spec.md, progress.txt)
├── data/                  # 离线数据中心
│   ├── raw/               # Tushare 原始落地数据
│   └── warehouse/         # Polars 优化后的 Parquet 数据库
├── src/                   # 源代码
│   ├── data_provider/     # Tushare 接口逻辑
│   ├── engine/            # polars_ta 封装与代码生成模板
│   ├── mining/            # DEAP 进化逻辑
│   ├── models/            # LightGBM 训练与预测
│   └── utils/             # 统一配置加载、日志管理
├── output/                # 统一产出
│   ├── codegen/           # GP 生成的因子代码
│   ├── logs/              # 每日运行日志
│   ├── models/            # 训练好的模型权重
│   └── reports/           # AlphaInspect HTML 报告
├── tests/                 # 单元测试与环境验证
└── manage.py              # 平台总调度入口
```

## 4. 关键流程 (Workflow)
- **数据就绪**：运行 `manage.py --sync`，从 Tushare 同步数据并转化为 Parquet。
- **挖掘启动**：配置 `spec.md` 中的搜索空间，运行 `manage.py --mine` 开始进化。
- **代码落盘**：`expr_codegen` 自动在 `output/codegen/` 生成新的因子模块。
- **模型合成**：调用 `src/models/` 对选定因子进行 LightGBM 训练。
- **可视化审计**：运行 AlphaInspect，在浏览器查看因子的实战表现。

## 5. 验收标准
- **性能**：处理全市场（5000+支股票）日线数据，因子计算耗时 < 1 秒。
- **合规**：所有路径引用必须符合 `pathlib` 规范，API Key 不得出现在代码库。
- **稳定性**：支持断点计算，即 GP 挖掘中断后可根据 `log/` 缓存恢复。

## 💡 下一步行动计划
这份需求文档现在可以作为你 AI 开发的 “总纲”。

建议从“数据层”开始构建： 你想让我为你生成 `src/data_provider/tushare_to_parquet.py` 的核心代码吗？它将实现：
1. Tushare API 的高效请求。
2. 自动转换为 Polars DataFrame。
3. 按照 `_DATE_` 和 `_ASSET_` 排序并保存为极速 Parquet。
