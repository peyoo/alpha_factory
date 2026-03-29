"""strategy.py — 策略配置模型

将 YAML 策略文件（如 output/main_small_pool/s1.yaml）映射为强类型 Pydantic 模型，
提供加载、校验与序列化能力。

典型用法::

    from alpha_factory.config.strategy import StrategyConfig

    cfg = StrategyConfig.from_yaml("output/main_small_pool/s1.yaml")
    print(cfg.name, cfg.hold_num, cfg.ranks[0].expression)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from alpha_factory.utils.schema import F


# ---------------------------------------------------------------------------
# 命名工具
# ---------------------------------------------------------------------------


class AutoNameGenerator:
    """统一的自动命名生成器。

    为所有内部列名（因子、过滤条件、复合器等）提供一致的命名策略。
    格式：{prefix}_{counter}

    示例::
        gen = AutoNameGenerator()
        gen.next_name("rank_f")      # "rank_f1"
        gen.next_name("rank_f")      # "rank_f2"
        gen.next_name("pool_mask")   # "pool_mask1"
        gen.next_name("pool_mask")   # "pool_mask2"
    """

    def __init__(self) -> None:
        """初始化计数器字典。"""
        self._counters: dict[str, int] = defaultdict(int)

    def next_name(self, prefix: str) -> str:
        """生成下一个名字。

        Args:
            prefix: 名字前缀，如 "rank_f"、"pool_mask"、"buy_able" 等。

        Returns:
            格式为 "{prefix}{counter}" 的名字。
        """
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def reset(self) -> None:
        """重置所有计数器（仅用于测试）。"""
        self._counters.clear()


# ---------------------------------------------------------------------------
# 子模型
# ---------------------------------------------------------------------------


class ExprCondition(BaseModel):
    """单条表达式规则。

    用于 ``pool_mask`` / ``not_sell_able`` / ``sell_able`` /
    ``not_buy_able`` / ``buy_able`` 列表中的每个元素。

    示例::

        pool_mask:
          - expression: "TOTAL_MV > 1000000000"
    """

    opt: ClassVar[dict] = {}
    expression: str = ""
    name: str = Field(default="", description="规则名称（可选）")


class FactorRank(BaseModel):
    """多因子策略中单个因子的完整定义。

    示例::

        ranks:
          - name: "f1"          # 可选，缺省时由 StrategyConfig 自动生成 rank_f1/rank_f2...
            expression: "ts_mean(AMOUNT, 60)"
            weight: 0.12
            direction: -1
    """

    expression: str
    name: Optional[str] = Field(
        default=None, description="因子名称，留空则由策略配置自动生成"
    )
    weight: float = Field(default=1.0, ge=0.0, description="因子权重，优化后写回此处")
    direction: Optional[Literal[1, -1]] = Field(
        default=None,
        description="1 表示因子值越大越好；-1 表示因子值越小越好；留空由 quant opt 通过 IC 自动推断",
    )

    @property
    def expr_str(self) -> str:
        """返回 DataProvider exprs 所需的 ``name = expression`` 格式字符串。

        如果 direction == -1 且表达式不以负号开头，在外加括号和负号；
        否则返回原始表达式，避免双重否定。
        """
        expr = self.expression.strip()
        if self.direction == -1 and not expr.startswith("-"):
            expr = f"-({expr})"
        return f"{self.name} = {expr}"


# ---------------------------------------------------------------------------
# 主配置模型
# ---------------------------------------------------------------------------


class StrategyConfig(BaseModel):
    """策略完整配置，一一对应 YAML 文件结构（如 s1.yaml）。

    字段说明
    --------
    name
        策略唯一标识符。
    pool
        股票池名称，对应 ``PoolUniverseEnum`` 可选值。
    pool_mask
        股票池过滤规则列表，逐条 AND 叠加。
    preprocess
        预处理函数名列表（字符串），将依次应用于所有 ranks 因子。
        如果缺省则表示因子不需要预处理。示例: ``['my_cs_mad_zscore_resid']``
    ranks
        参与合成排名的因子列表，至少需要 2 个因子方可执行权重优化。
    bt_mode
        回测模式，目前支持 ``"daily_top_n"``（逐日演进）。
    ls_mode
        多空模式，目前支持 ``"close"``（收盘价换仓）。
    exe_price
        实际执行价格字段，支持 ``"close"`` / ``"vwap"``。
    cost
        单边交易成本率（含冲击成本与佣金），例如 ``0.003``。
    n_bins
        分层回测的分层数。
    hold_num
        最大持仓股票数量。
    hold_rank
        持仓排名阈值：排名 ≤ hold_rank 的标的允许继续持有不触发卖出。
    sell_rank
        卖出排名阈值：排名 > sell_rank 且已持仓则触发卖出信号。
    not_sell_able
        不可卖出条件（表达式列表），任意一条成立则当日禁止卖出。
    sell_able
        强制可卖出白名单（表达式列表）。
    not_buy_able
        不可买入条件（表达式列表），任意一条成立则当日禁止买入。
    buy_able
        强制可买入白名单（表达式列表）。
    """

    # ---------- 基础信息 ----------
    name: str = Field(default="strategy", description="策略名称")
    pool: str = Field(default="main_small_pool", description="股票池名称")

    # ---------- 股票池过滤 ----------
    pool_mask: List[ExprCondition] = Field(
        default_factory=list,
        description="股票池额外过滤规则（AND 叠加）",
    )

    # ---------- 因子预处理 ----------
    preprocess: List[str] = Field(
        default_factory=list,
        description="预处理函数名列表，依次应用于所有 ranks 因子",
    )

    # ---------- 因子列表 ----------
    ranks: List[FactorRank] = Field(
        default_factory=list,
        description="参与合成排名的因子定义，顺序对应权重向量",
    )

    @model_validator(mode="after")
    def _validate(self) -> "StrategyConfig":
        """验证阶段（命名已在 from_yaml 时统一生成）。"""
        return self

    # ---------- 回测模式 ----------
    bt_mode: str = Field(default="daily_top_n", description="回测模式")
    ls_mode: str = Field(default="close", description="多空换仓模式")
    exe_price: str = Field(default="close", description="实际执行价格字段")

    # ---------- 成本与分层 ----------
    cost: float = Field(default=0.003, ge=0.0, le=1.0, description="单边交易成本率")
    n_bins: int = Field(default=10, ge=2, description="分层回测层数")

    # ---------- 仓位管理 ----------
    hold_num: int = Field(default=10, ge=1, description="最大持仓股数")
    buy_rank: int = Field(default=10, ge=1, description="买入排名阈值")
    sell_rank: int = Field(default=30, ge=1, description="触发卖出排名阈值")

    # ---------- 交易约束 ----------
    not_sell_able: List[ExprCondition] = Field(
        default_factory=list, description="不可卖出条件列表,or 叠加"
    )
    sell_able: List[ExprCondition] = Field(
        default_factory=list, description="可卖出白名单条件列表,and 叠加"
    )
    not_buy_able: List[ExprCondition] = Field(
        default_factory=list, description="不可买入条件列表,or 叠加"
    )
    buy_able: List[ExprCondition] = Field(
        default_factory=list, description="可买入白名单条件列表,and 叠加"
    )

    # ---------- 评估相关配置（用于 evals 命令及其他评估工具） ----------
    start_date: Optional[str] = Field(
        default=None, description="数据起始日期（YYYYMMDD），可由命令行覆盖"
    )
    end_date: Optional[str] = Field(
        default=None, description="数据结束日期（YYYYMMDD），可由命令行覆盖"
    )
    ic_decay: bool = Field(default=False, description="是否计算 IC Decay（衰减评估）")
    turnover_decay: bool = Field(
        default=False, description="是否计算 Turnover Decay（换手率衰减）"
    )
    cluster: bool = Field(default=False, description="是否进行因子聚类分析")
    relevance_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="聚类相关性阈值（0~1），越高聚类越严格"
    )

    # ---------------------------------------------------------------------------
    # I/O helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StrategyConfig":
        """从 YAML 文件加载策略配置并进行 Pydantic 校验。

        加载后会统一生成所有内部名字（因子、过滤条件等），覆盖 YAML 中可能存在的
        自定义名字，以确保名字的一致性和可预测性。

        Args:
            path: YAML 文件路径（绝对或相对于当前工作目录）。

        Returns:
            经过校验和名字统一化的 :class:`StrategyConfig` 实例。

        Raises:
            FileNotFoundError: 文件不存在。
            pydantic.ValidationError: 字段类型/约束不满足。
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"策略 YAML 文件不存在: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            data = {}

        # 统一生成所有内部名字（覆盖 YAML 中可能存在的名字）
        gen = AutoNameGenerator()

        # 1. 为 ranks（因子）统一生成名字
        if "ranks" in data and isinstance(data["ranks"], list):
            for rank in data["ranks"]:
                if isinstance(rank, dict):
                    rank["name"] = gen.next_name("rank_f")

        # 2. 为所有过滤条件统一生成名字
        for filter_key in [
            "pool_mask",
            "buy_able",
            "not_buy_able",
            "sell_able",
            "not_sell_able",
        ]:
            if filter_key in data and isinstance(data[filter_key], list):
                for condition in data[filter_key]:
                    if isinstance(condition, dict):
                        condition["name"] = gen.next_name(filter_key)

        return cls.model_validate(data)

    def to_yaml(
        self,
        path: Union[str, Path],
        *,
        preserve_comments: bool = True,
    ) -> None:
        """将策略配置序列化写回 YAML 文件。

        注释保留原理：
        - 若原文件存在且 preserve_comments=True，从原文件用 ruamel.yaml 加载 CommentedMap
        - CommentedMap 保留原始注释信息
        - 使用智能合并：只更新已存在的字段，但允许更新嵌套对象的属性值
          （如 ranks 列表中的 weight、direction）
        - 若原文件不存在，使用标准 PyYAML 输出（无注释）

        合并策略：
        - 顶级字段：只在已存在的字段上修改值，不新增字段
        - 嵌套列表对象：按位置匹配（ranks[0]、ranks[1]...），更新其所有字段值
        - 这样既保留注释结构，又允许优化结果（weight、direction）被保存

        Args:
            path: 目标文件路径。
            preserve_comments: 若为 True 且原文件存在，则保留原有注释；
                否则生成无注释的 YAML。
        """
        path = Path(path)
        data = self.model_dump()

        # 尝试保留原文件的注释
        if preserve_comments and path.exists():
            try:
                from ruamel.yaml import YAML as RuamelYAML
            except ImportError as exc:
                raise ImportError(
                    "preserve_comments=True 需要 ruamel.yaml，请执行 `uv sync`"
                ) from exc

            try:
                # 加载原文件为 CommentedMap（保留注释）
                yaml_obj = RuamelYAML()
                yaml_obj.preserve_quotes = True
                with path.open("r", encoding="utf-8") as f:
                    commented = yaml_obj.load(f)

                if commented is None:
                    commented = {}

                # 智能合并：保留结构同时允许更新嵌套对象的字段值
                def merge_preserve_structure(target: dict, source: dict) -> None:
                    """智能递归合并，保留目标结构同时更新值。

                    策略：
                    1. 只更新目标中已有的字段（不新增）
                    2. 对于嵌套列表，按位置匹配并更新列表项的字段
                    3. 对于嵌套字典，递归处理
                    """
                    for key in list(target.keys()):
                        if key not in source:
                            continue

                        src_value = source[key]
                        tgt_value = target[key]

                        # 情况1：都是列表 -> 按位置匹配更新列表项
                        if isinstance(tgt_value, list) and isinstance(src_value, list):
                            # 对每个位置的列表项进行更新
                            for i, (tgt_item, src_item) in enumerate(
                                zip(tgt_value, src_value)
                            ):
                                if isinstance(tgt_item, dict) and isinstance(
                                    src_item, dict
                                ):
                                    # 列表中的字典对象：更新所有字段
                                    tgt_item.update(src_item)
                                else:
                                    # 列表中的简单值：直接替换
                                    tgt_value[i] = src_item
                        # 情况2：都是字典 -> 递归处理
                        elif isinstance(tgt_value, dict) and isinstance(
                            src_value, dict
                        ):
                            merge_preserve_structure(tgt_value, src_value)
                        # 情况3：其他类型 -> 直接替换
                        else:
                            target[key] = src_value

                merge_preserve_structure(commented, data)

                # 写回 CommentedMap（注释被保留）
                with path.open("w", encoding="utf-8") as f:
                    yaml_obj.dump(commented, f)
            except Exception:  # noqa: BLE001
                # 如果处理失败，回退到基础 YAML 输出
                import yaml

                with path.open("w", encoding="utf-8") as f:
                    yaml.dump(
                        data,
                        f,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    )
        else:
            # 原文件不存在或用户不需要保留注释，直接输出
            import yaml

            with path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )

    # ---------------------------------------------------------------------------
    # 便捷属性
    # ---------------------------------------------------------------------------

    @property
    def factor_names(self) -> List[str]:
        """返回所有因子名称列表。"""
        return [r.name for r in self.ranks]

    @property
    def factor_weights(self) -> List[float]:
        """返回所有因子权重列表（顺序与 :attr:`ranks` 一致）。"""
        return [r.weight for r in self.ranks]

    @property
    def factor_directions(self) -> List[int]:
        """返回所有因子方向列表（1 / -1）。"""
        return [r.direction for r in self.ranks]

    @property
    def ranked_factor_exprs(self) -> List[str]:
        """
        返回所有线性加权因子的 ``name = expression`` 字符串列表，供 DataProvider 使用。
        """
        return [r.expr_str for r in self.ranks]

    def get_condition_exprs(self) -> List[str]:
        """
        收集所有条件表达式（pool_mask, buy_able, not_buy_able, sell_able, not_sell_able）。

        名字已在 from_yaml 时统一生成，此方法只需收集表达式。
        返回 ``name = expression`` 格式的列表。
        这些表达式会被 DataProvider 一起计算成列。

        Returns:
            表达式列表，格式：["name_1 = expr_1", "name_2 = expr_2", ...]
        """
        exprs: List[str] = []

        def _process_filters(filters: List[ExprCondition]) -> None:
            """处理一组条件规则，直接使用已生成的名称。"""
            for f in filters:
                if not f.expression.strip():
                    continue
                exprs.append(f"{f.name} = {f.expression.strip()}")

        # 按顺序收集所有过滤表达式
        _process_filters(self.pool_mask)
        _process_filters(self.buy_able)
        _process_filters(self.not_buy_able)
        _process_filters(self.sell_able)
        _process_filters(self.not_sell_able)

        return exprs

    def build_actions(self) -> List:
        """构建完整的 actions 管道。

        返回顺序（关键：因子处理在 pool_mask 之后，后续过滤之前）：
        1. pool_mask AND 合成
        2. 因子预处理（多因子）
        3. 因子合成（多因子）
        4. buy_able AND 合成（全部条件满足）
        5. not_buy_able OR 合成（任一条件成立则禁止）
        6. sell_able AND 合成（全部条件满足）
        7. not_sell_able OR 合成（任一条件成立则禁止）

        所有内部列名已在 from_yaml 时统一生成。

        Returns:
            FactorsAction 对象列表，供 DataProvider.load_pool_data 使用。
        """
        from alpha_factory.data_provider.factorsprocessor import (
            FactorsPreProcessor,
            FactorsRankComposite,
        )
        from alpha_factory.data_provider.prcoessors import And, Or
        from alpha_factory.cli.opt import _COMPOSITE_COL

        actions: List = []

        def _collect_filter_names(filters: List[ExprCondition]) -> List[str]:
            """收集过滤条件的已生成名称。"""
            return [f.name for f in filters if f.expression.strip()]

        # 1. Pool mask AND 合成
        if self.pool_mask:
            names = _collect_filter_names(self.pool_mask)
            if names:
                actions.append(And(factors=names, name=F.POOL_MASK))

        # 2. 因子预处理（多因子）
        if len(self.ranks) > 1:
            factor_names = self.factor_names

            # 预处理（如果配置了）
            if self.preprocess:
                actions.append(
                    FactorsPreProcessor(factors=factor_names, actions=self.preprocess)
                )

            # 3. 因子合成（权重直接使用，FactorsRankComposite 内部会做自动归一化）
            signed_weights = {
                name: float(w) for name, w in zip(factor_names, self.factor_weights)
            }
            actions.append(
                FactorsRankComposite(
                    factors=factor_names,
                    name=_COMPOSITE_COL,
                    weights=signed_weights,
                    use_rank=False,  # 因子已经过 FactorsPreProcessor 预处理，直接加权求和
                )
            )

        # 4. buy_able AND 合成（全部条件满足才允许购买）
        if self.buy_able:
            names = _collect_filter_names(self.buy_able)
            if names:
                actions.append(And(factors=names, name="buy_able"))

        # 5. not_buy_able OR 合成（任一条件成立则禁止购买）
        if self.not_buy_able:
            names = _collect_filter_names(self.not_buy_able)
            if names:
                actions.append(Or(factors=names, name="not_buy_able"))

        # 6. sell_able AND 合成（全部条件满足才允许卖出）
        if self.sell_able:
            names = _collect_filter_names(self.sell_able)
            if names:
                actions.append(And(factors=names, name="sell_able"))

        # 7. not_sell_able OR 合成（任一条件成立则禁止卖出）
        if self.not_sell_able:
            names = _collect_filter_names(self.not_sell_able)
            if names:
                actions.append(Or(factors=names, name="not_sell_able"))

        return actions
