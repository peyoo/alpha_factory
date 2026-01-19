
## 1. 模块目标

通过 **Lasso** 进行高维特征的稀疏化筛选（剔除共线性与噪音），随后利用 **LightGBM** 捕捉深层非线性信号，最终合成具备高 ICIR 的 Alpha 预测模型。

---

## 2. 挖掘流水线架构 (Mining Pipeline)

模块采用“线性粗筛 + 非线性精炼”的双塔结构：

1. **特征池化 (Pooling)**：输入由 GP 或手工构造的 500+ 原始因子。
2. **Lasso 降维 (Sparsity)**：利用 L1 正则化强制将弱相关因子的系数归零。
3. **LGBM 组合 (Interaction)**：对 Lasso 留下的核心因子进行树模型集成。

---

## 3. Lasso 筛选规范 (The Linear Filter)

### 3.1 预处理要求

* **标准化**：Lasso 对量纲极度敏感。进入 Lasso 前必须执行 `StandardScaler`（均值为0，方差为1）。
* **时序对齐**：使用 \$T\$ 期的因子预测 \$T+N\$ 期的超额收益。

### 3.2 算子配置

* **正则化路径**：使用 `LassoCV` 自动搜索最优的 \$\\alpha\$ 参数。
* **筛选标准**：保留系数 \$\\beta \\neq 0\$ 的特征。
* **稳定性检查**：采用 **Stability Selection**（多次子采样），仅保留在 80% 以上子样本中都被选中的特征。

---

## 4. LightGBM 挖掘规范 (The Non-linear Refiner)

### 4.1 训练目标 (Objective)

* **优先使用 `lambdarank`**：在截面上学习股票的相对排序，而非绝对收益数值。
* **指标**：关注 `NDCG` 和 `Top-K Recall`。

### 4.2 防止过拟合策略

* **单调性约束 (Monotonic Constraints)**：对于逻辑上明确正相关或负相关的因子（如：动量、价值），强制设定单调性约束，防止噪声拟合。
* **列采样**：设定 `feature_fraction` < 0.8，强制模型探索不同的因子组合。

---

## 5. 验证与归档规范 (Validation & Archiving)

### 5.1 走入式校验 (Walk-forward Validation)

必须遵循严格的时间轴：

1. **训练窗口**：12-24 个月。
2. **验证窗口**：3 个月（用于 Early Stopping）。
3. **测试窗口**：1 个月（模拟 OOS 表现）。
4. **滚动频率**：按月或按季重新训练。

### 5.2 产出物清单

每次挖掘任务结束后，必须在 `output/models/` 生成：

* `model_v1.bin`: 训练好的 LightGBM 模型。
* `selected_features.json`: Lasso 选出的特征清单及其系数。
* `importance_plot.png`: 特征增益排名的可视化图表。

---

## 6. 开发者准则 (Guidelines)

* **Polars 集成**：特征变换（Rank, Z-Score）必须使用 Polars 表达式完成，严禁在循环中处理单只股票。
* **影子变量一致性**：训练 `Lasso` 和 `LGBM` 时，`target` 必须使用 `docs/specs/01_infrastructure.md` 定义的复权后收益。
* **日志控制**：`LassoCV` 的收敛信息和 `LGBM` 的迭代日志必须重定向至 `loguru`，严禁污染控制台。
