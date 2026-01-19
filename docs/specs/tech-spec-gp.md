# 遗传规划因子挖掘模块技术规格说明书 (GP - Genetic Programming)

**版本**: 1.0
**模块路径**: `alpha.gp`
**责任人**: AI Copilot

## 1. 概述
本模块基于 **DEAP (Distributed Evolutionary Algorithms in Python)** 框架，实现符号回归式的因子挖掘。其核心目标是：通过进化算法自动搜索由基础算子（技术指标、数学运算）组合而成的高质量 Alpha 因子。

## 2. 设计原则
1.  **适应度驱动 (Fitness-Oriented)**：以 RankIC、ICIR、信息系数为核心评价指标，确保进化方向符合量化业务目标。
2.  **表达式树 (Expression Tree)**：因子表示为抽象语法树 (AST)，便于遗传操作（交叉、变异）与代码生成。
3.  **Polars 原生适配**：生成的代码必须严格遵循 Polars 表达式规范，避免循环和 Pandas 依赖。

## 3. 核心架构

### 3.1 模块结构
```text
alpha/gp/
├── __init__.py          # 暴露核心接口
├── primitives.py        # 基础算子库 (自动生成，基于 polars_ta)
├── main.py              # DEAP 主进化循环与配置
├── cs/                  # 截面算子 (Cross-Sectional)
│   ├── __init__.py
│   ├── base.py          # 基础工具 (sympy 转换、表达式简化)
│   ├── custom.py        # 自定义截面算子
│   ├── deap_patch.py    # DEAP 框架补丁 (重要!)
│   └── helper.py        # 辅助函数
└── ts/                  # 时间序列算子 (Time-Series)
    ├── __init__.py
    ├── custom.py        # 自定义时序算子
    └── helper.py        # 辅助函数
```

**关键设计特点**:
1.  **算子自动生成**: `primitives.py` 由 `codegen_primitive.py` 自动生成，直接对接 `polars_ta` 库的所有算子。
2.  **Sympy 集成**: 通过 `expr_codegen` 和 `sympy` 实现表达式简化与代码生成。
3.  **DEAP 补丁**: `cs/deap_patch.py` 对 DEAP 进行了关键修改，确保与 Polars 的兼容性。
4.  **模块化算子**: 区分截面算子 (`cs/`) 和时间序列算子 (`ts/`)，便于独立开发和测试。

### 3.2 进化流程
```
初始化种群 (Random Expressions)
    ↓
[循环] 评估适应度 (RankIC/ICIR)
    ↓
选择 (Tournament Selection)
    ↓
交叉/变异 (Crossover/Mutation)
    ↓
精英保留 (Elitism)
    ↓
收敛判断 → 输出最优因子
    ↓
代码生成 (output/codegen/alpha_xxx.py)
```

## 4. 技术规格

### 4.1 基础算子库 (Primitives)

**算子来源**: 基于 `polars_ta` 库，通过 `codegen_primitive.py` 自动生成。

#### 数学运算 (`polars_ta.wq.arithmetic`)
- **二元算子**: `add`, `subtract`, `multiply`, `divide` (保护除法)、`power`, `signed_power`
- **统计算子**: `mean`, `std`, `var`, `max_`, `min_`
- **一元算子**: `abs_`, `sign`, `sqrt`, `log`, `log10`, `log1p`, `exp`, `expm1`
- **三角函数**: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arc_sin`, `arc_cos`, `arc_tan`
- **其他**: `floor`, `ceiling`, `round_`, `round_down`, `softsign`, `sigmoid`

#### 时间序列算子 (`polars_ta.wq.time_series`)
- **滑动窗口**: `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_sum`, `ts_product`
- **滞后/差分**: `ts_delay`, `ts_delta`, `ts_returns`, `ts_log_diff`
- **排名**: `ts_rank`, `ts_zscore`, `ts_scale`
- **统计量**: `ts_skewness`, `ts_kurtosis`, `ts_median`, `ts_moment`, `ts_l2_norm`
- **累积**: `ts_cum_sum`, `ts_cum_prod`, `ts_cum_max`, `ts_cum_min`, `ts_cum_count`
- **回归**: `ts_regression_slope`, `ts_regression_intercept`, `ts_regression_pred`, `ts_regression_resid`
- **相关性**: `ts_corr`, `ts_covariance`, `ts_partial_corr`, `ts_triple_corr`
- **加权**: `ts_weighted_mean`, `ts_weighted_sum`, `ts_decay_linear`, `ts_decay_exp_window`
- **其他**: `ts_arg_max`, `ts_arg_min`, `ts_fill_null`, `ts_count_nans`, `ts_ir`

#### 截面算子 (`polars_ta.wq.cross_sectional`)
- **排名**: `cs_rank` (全市场排名)
- **标准化**: `cs_demean`, `cs_zscore`, `cs_minmax`, `cs_scale`
- **稳健处理**: `cs_mad`, `cs_mad_zscore`, `cs_mad_rank`, `cs_3sigma`
- **回归**: `cs_regression_neut` (中性化)、`cs_regression_proj`、`cs_resid`
- **分组**: `cs_qcut`, `cs_top_bottom`, `cs_one_side`
- **填充**: `cs_fill_null`, `cs_fill_mean`, `cs_fill_except_all_null`

#### 技术指标 (`polars_ta.ta`)
- **动量**: `ts_RSI`, `ts_MOM`, `ts_ROC`, `ts_ROCP`, `ts_ROCR`
- **趋势**: `ts_MACD_macd`, `ts_MACD_macdsignal`, `ts_MACD_macdhist`
- **摆动**: `ts_WILLR`, `ts_RSV`, `ts_STOCHF_fastd`
- **其他**: `ts_APO`, `ts_PPO`, `ts_AROON_aroonup`, `ts_AROON_aroondown`

#### 逻辑算子 (`polars_ta.wq.logical`)
- **比较**: `equal`, `less`, `and_`, `or_`, `xor`, `not_`
- **空值检查**: `is_null`, `is_not_null`, `is_nan`, `is_not_nan`, `is_finite`
- **条件**: `if_else` (三元运算)

**约束**:
- 所有算子均由 `polars_ta` 提供，确保返回 `pl.Expr` 类型。
- 算子参数类型包括 `Expr`、`int`、`float`、`bool` 等，由 DEAP 的类型系统自动匹配。
- 禁止在 GP 过程中使用 `map_elements` 或 Python 循环。

### 4.2 适应度评估 (Fitness Evaluation)

#### 核心指标
1.  **RankIC (Rank Information Coefficient)**:
    - 计算因子值与未来收益率的秩相关系数。
    - 公式: `spearmanr(factor_rank, forward_return_rank)`
    - 目标: **最大化** 均值 RankIC。

2.  **ICIR (IC Information Ratio)**:
    - `ICIR = mean(IC) / std(IC)`
    - 衡量 IC 的稳定性，惩罚波动过大的因子。

#### 适应度函数
```python
def map_exprs(evaluate, invalid_ind, gen, label, split_date)
```

### 4.3 遗传算子 (Genetic Operators)

#### 交叉 (Crossover)
- **方法**: 子树交换 (`cxOnePoint` 或自定义)
- **概率**: 0.7

#### 变异 (Mutation)
- **节点替换**: 随机替换算子或终端节点
- **子树变异**: 随机生成新子树替换原节点
- **概率**: 0.3

#### 选择 (Selection)
- **锦标赛选择 (Tournament)**: 比赛规模 = 3
- **精英保留 (Elitism)**: 每代保留前 5% 最优个体

### 4.4 代码生成 (Code Generation)

#### 输出格式
生成的代码文件应符合以下模板：
```python
"""
自动生成的 Alpha 因子
生成时间: 2025-01-19 15:30:00
RankIC: 0.0423
ICIR: 1.85
"""
import polars as pl

def alpha_001(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    因子描述: (CLOSE - ts_mean(CLOSE, 20)) / ts_std(CLOSE, 20)

    Args:
        lf: 包含 _DATE_, _ASSET_, OHLCV 的 LazyFrame

    Returns:
        新增列 f_alpha_001 的 LazyFrame
    """
    return lf.with_columns(
        ((pl.col("CLOSE") - pl.col("CLOSE").rolling_mean(20).over("_ASSET_"))
         / pl.col("CLOSE").rolling_std(20).over("_ASSET_")
        ).cast(pl.Float32).alias("f_alpha_001")
    )
```

#### 存储路径
- `output/codegen/alpha_{timestamp}.py`
- 元数据 JSON: `output/codegen/metadata.json` (记录所有因子的评估指标)

## 5. 关键功能规格

### 5.1 初始化种群
- **种群规模**: 100-300 个体
- **最大深度**: 5-7 层
- **生成方式**: Half-and-half (50% full, 50% grow)

### 5.2 进化参数
- **代数 (Generations)**: 50-100
- **早停 (Early Stopping)**: 连续 10 代无改进则停止
- **并行化**: 使用 `multiprocessing.Pool` 并行评估适应度

### 5.3 约束条件
- **复杂度惩罚**: 节点数 > 30 时降低适应度
- **有效性检查**: 剔除产生全 NaN 或常数的因子
- **去重**: 基于哈希值避免重复个体

## 6. 使用示例 (Draft)

```python
from alpha.gp import GeneticProgramming
from alpha.data_provider import DataProvider

# 1. 加载数据
provider = DataProvider()
data = provider.load_data(
    start_date="20200101",
    end_date="20231231",
    columns=["CLOSE", "VOLUME", "target_1d_return"]
)

# 2. 初始化 GP 引擎
gp = GeneticProgramming(
    population_size=200,
    generations=50,
    crossover_prob=0.7,
    mutation_prob=0.3
)

# 3. 运行进化
best_factors = gp.evolve(data, n_best=10)

# 4. 生成代码
for idx, factor in enumerate(best_factors):
    gp.codegen(factor, output_path=f"output/codegen/alpha_{idx:03d}.py")
```

## 7. 下一步开发任务
1.  实现 `alpha.gp.primitives` 定义基础算子库。
2.  实现 `alpha.gp.evaluator` 适应度评估逻辑。
3.  实现 `alpha.gp.engine` DEAP 主循环。
4.  实现 `alpha.gp.codegen` 代码生成器。
5.  编写单元测试验证因子有效性。

## 8. 现有实现说明

### 8.1 已实现的组件
- ✅ **primitives.py**: 已由代码生成工具自动生成，包含 `polars_ta.wq` 和 `polars_ta.ta` 的全部算子
- ✅ **cs/deap_patch.py**: DEAP 框架的关键补丁，解决类型系统兼容性问题
- ✅ **cs/base.py**: Sympy 集成工具，实现表达式树到 Sympy 符号表达式的转换
  - `stringify_for_sympy()`: 将 DEAP 个体转换为 Sympy 可识别的字符串
  - `convert_inverse_prim()`: 处理减法/除法到加法/乘法的转换
  - `convert_inverse_sympy()`: Sympy 符号的反向转换
- ✅ **main.py**: 主进化循环框架（部分实现）

### 8.2 关键技术细节

#### Sympy 表达式转换
现有代码通过 `expr_codegen.codes.sources_to_exprs` 实现从字符串到 Polars 表达式的转换：
```python
from expr_codegen.codes import sources_to_exprs

# DEAP 个体 -> Sympy 字符串 -> Polars Expr
sympy_str = stringify_for_sympy(individual)
polars_expr = sources_to_exprs(sympy_str)
```

#### 算子命名约定
为支持类型重载，算子命名包含类型前缀：
- `oo_add`: 两个 `Expr` 类型的加法
- `oi_add`: `Expr` 和 `int` 的加法
- `of_add`: `Expr` 和 `float` 的加法
- `io_add`: `int` 和 `Expr` 的加法

#### 数据预处理建议
根据 `main.py` 中的注释，推荐数据预处理策略：
1. **特征预处理**: 行业中性化等操作应在 GP 外完成，避免重复计算
2. **样本内外划分**: 无需像机器学习那样严格分训练/测试集，IC 可事后分段统计
3. **标签设计**:
   - `returns`: 简单收益率，用于分组收益计算（横截面等权）
   - `labels`: 因变量，可能是超额收益、对数收益或分类标签

### 8.3 待完善的功能
- [ ] 适应度评估函数的完整实现（RankIC/ICIR 计算）
- [ ] 遗传算子的参数调优（交叉/变异概率）
- [ ] 代码生成器的模板完善
- [ ] 并行化评估的实现（`multiprocessing.Pool`）
- [ ] 早停机制与收敛判断
- [ ] 因子去重与有效性检查
