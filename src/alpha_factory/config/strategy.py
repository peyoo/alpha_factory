"""strategy.py — 策略配置模型

将 YAML 策略文件（如 output/main_small_pool/s1.yaml）映射为强类型 Pydantic 模型，
提供加载、校验与序列化能力。

典型用法::

    from alpha_factory.config.strategy import StrategyConfig

    cfg = StrategyConfig.from_yaml("output/main_small_pool/s1.yaml")
    print(cfg.name, cfg.hold_num, cfg.ranks[0].expression)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 子模型
# ---------------------------------------------------------------------------


class ExprFilter(BaseModel):
    """单条表达式过滤规则。

    用于 ``pool_mask`` / ``not_sell_able`` / ``sell_able`` /
    ``not_buy_able`` / ``buy_able`` 列表中的每个元素。

    示例::

        pool_mask:
          - expression: "TOTAL_MV > 1000000000"
    """

    expression: str = ""
    name: str = Field(default="", description="过滤规则名称（可选）")


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
    pool_mask: List[ExprFilter] = Field(
        default_factory=list,
        description="股票池额外过滤规则（AND 叠加）",
    )

    # ---------- 因子列表 ----------
    ranks: List[FactorRank] = Field(
        default_factory=list,
        description="参与合成排名的因子定义，顺序对应权重向量",
    )

    @model_validator(mode="after")
    def _auto_fill_rank_names(self) -> "StrategyConfig":
        """为未命名的因子自动生成 rank_f1 / rank_f2 … 形式的名称。"""
        for idx, rank in enumerate(self.ranks, start=1):
            if not rank.name:
                rank.name = f"rank_f{idx}"
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
    not_sell_able: List[ExprFilter] = Field(
        default_factory=list, description="不可卖出条件列表,or 叠加"
    )
    sell_able: List[ExprFilter] = Field(
        default_factory=list, description="可卖出白名单条件列表,and 叠加"
    )
    not_buy_able: List[ExprFilter] = Field(
        default_factory=list, description="不可买入条件列表,or 叠加"
    )
    buy_able: List[ExprFilter] = Field(
        default_factory=list, description="可买入白名单条件列表,and 叠加"
    )

    # ---------------------------------------------------------------------------
    # I/O helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StrategyConfig":
        """从 YAML 文件加载策略配置并进行 Pydantic 校验。

        Args:
            path: YAML 文件路径（绝对或相对于当前工作目录）。

        Returns:
            经过校验的 :class:`StrategyConfig` 实例。

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

        return cls.model_validate(data or {})

    def to_yaml(
        self,
        path: Union[str, Path],
        *,
        preserve_comments: bool = True,
    ) -> None:
        """将策略配置序列化写回 YAML 文件。

        Args:
            path: 目标文件路径。
            preserve_comments: 若为 ``True`` 则使用 ``ruamel.yaml`` 保留原有注释
                （需要已安装 ``ruamel.yaml``）；否则使用标准 ``PyYAML``。
        """
        path = Path(path)
        data = self.model_dump()

        if preserve_comments:
            try:
                from ruamel.yaml import YAML as RuamelYAML
            except ImportError as exc:
                raise ImportError(
                    "preserve_comments=True 需要 ruamel.yaml，请执行 `uv sync`"
                ) from exc

            _yaml = RuamelYAML()
            _yaml.preserve_quotes = True
            with path.open("w", encoding="utf-8") as f:
                _yaml.dump(data, f)
        else:
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
    def factor_exprs(self) -> List[str]:
        """返回所有因子的 ``name = expression`` 字符串列表，供 DataProvider 使用。"""
        return [r.expr_str for r in self.ranks]
