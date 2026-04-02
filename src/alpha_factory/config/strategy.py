"""strategy.py — 策略配置模型（YAML ↔ Pydantic 双向映射）"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from alpha_factory.utils.schema import F


class AutoNameGenerator:
    """为因子列、过滤条件等生成全局唯一名称，格式 ``{prefix}{n}``。"""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def next_name(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def reset(self) -> None:
        self._counters.clear()


_COND_GROUPS = ("pool_mask", "buy_able", "not_buy_able", "sell_able", "not_sell_able")


class ExprCondition(BaseModel):
    """单条过滤表达式，支持 ``{param}`` 占位符参数化以供 Optuna 优化。

    带可优化参数的 YAML 示例::

        pool_mask:
          - expression: "TOTAL_MV > {v1}"
            opt:
              v1:
                type: float      # float | int | categorical
                low: 500000000
                high: 3000000000
                default: 1000000000  # 非优化模式使用；优化后覆盖写回
    """

    expression: str = ""
    name: str = ""
    opt: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_opt_params(self) -> bool:
        return bool(self.opt)

    @property
    def resolved_expression(self) -> str:
        """用 ``default`` 值替换占位符；无 opt 时返回原 expression。"""
        if not self.opt:
            return self.expression
        expr = self.expression
        for param_name, spec in self.opt.items():
            if "default" not in spec:
                raise ValueError(
                    f"ExprCondition '{self.name}'.opt['{param_name}'] 缺少必填字段 'default'"
                )
            expr = expr.replace(f"{{{param_name}}}", str(spec["default"]))
        return expr

    def resolve_expression(self, trial: Any) -> str:
        """用 Optuna trial 采样值替换占位符；key 格式 ``"{name}__{param_name}"``。"""
        if not self.opt:
            return self.expression
        expr = self.expression
        for param_name, spec in self.opt.items():
            key = f"{self.name}__{param_name}"
            t = spec.get("type", "float")
            if t == "float":
                v = trial.suggest_float(
                    key, float(spec["low"]), float(spec["high"]), step=spec.get("step")
                )
            elif t == "int":
                v = trial.suggest_int(
                    key,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec.get("step", 1)),
                )
            elif t == "categorical":
                v = trial.suggest_categorical(key, spec["choices"])
            else:
                raise ValueError(
                    f"ExprCondition '{self.name}'.opt['{param_name}'] 不支持 type='{t}'"
                )
            expr = expr.replace(f"{{{param_name}}}", str(v))
        return expr


class FactorRank(BaseModel):
    """单个因子定义（表达式 / 权重 / 方向）。"""

    expression: str
    name: Optional[str] = None
    weight: float = Field(default=1.0, ge=0.0)
    direction: Optional[Literal[1, -1]] = None

    @property
    def expr_str(self) -> str:
        """返回 ``name = [±]expression`` 格式；direction=-1 时在外加负号。"""
        expr = self.expression.strip()
        if self.direction == -1 and not expr.startswith("-"):
            expr = f"-({expr})"
        return f"{self.name} = {expr}"


class StrategyConfig(BaseModel):
    """策略完整配置，对应一个 YAML 文件（如 ``s1.yaml``）。"""

    name: str = "strategy"
    pool: str = "main_small_pool"

    pool_mask: List[ExprCondition] = Field(default_factory=list)
    preprocess: List[str] = Field(default_factory=list)
    ranks: List[FactorRank] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "StrategyConfig":
        return self

    bt_mode: str = "daily_top_n"
    ls_mode: str = "close"
    exe_price: str = "close"
    cost: float = Field(default=0.003, ge=0.0, le=1.0)
    n_bins: int = Field(default=10, ge=2)
    hold_num: int = Field(default=10, ge=1)
    buy_rank: int = Field(default=10, ge=1)
    sell_rank: int = Field(default=30, ge=1)

    not_sell_able: List[ExprCondition] = Field(default_factory=list)
    sell_able: List[ExprCondition] = Field(default_factory=list)
    not_buy_able: List[ExprCondition] = Field(default_factory=list)
    buy_able: List[ExprCondition] = Field(default_factory=list)

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ic_decay: bool = False
    turnover_decay: bool = False
    cluster: bool = False
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # 运行时状态，不序列化到 YAML；quant opt 启动时手动设为 True
    opt_mode: bool = Field(default=False, exclude=True)

    # ---------------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StrategyConfig":
        """从 YAML 加载并为所有因子/条件统一生成内部列名。"""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"策略 YAML 文件不存在: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        gen = AutoNameGenerator()
        if "ranks" in data:
            for r in data["ranks"]:
                if isinstance(r, dict):
                    r["name"] = gen.next_name("rank_f")
        for key in _COND_GROUPS:
            if key in data:
                for c in data[key]:
                    if isinstance(c, dict):
                        c["name"] = gen.next_name(key)

        return cls.model_validate(data)

    def to_yaml(
        self, path: Union[str, Path], *, preserve_comments: bool = True
    ) -> None:
        """序列化写回 YAML；preserve_comments=True 时用 ruamel.yaml 保留原始注释。"""
        path = Path(path)
        data = self.model_dump()

        if preserve_comments and path.exists():
            try:
                from ruamel.yaml import YAML as RuamelYAML
            except ImportError as exc:
                raise ImportError("需要 ruamel.yaml，请执行 `uv sync`") from exc

            try:
                yaml_obj = RuamelYAML()
                yaml_obj.preserve_quotes = True
                with path.open("r", encoding="utf-8") as f:
                    commented = yaml_obj.load(f) or {}

                def _merge(target: dict, source: dict) -> None:
                    for k in list(target.keys()):
                        if k not in source:
                            continue
                        sv, tv = source[k], target[k]
                        if isinstance(tv, list) and isinstance(sv, list):
                            for i, (ti, si) in enumerate(zip(tv, sv)):
                                if isinstance(ti, dict) and isinstance(si, dict):
                                    ti.update(si)
                                else:
                                    tv[i] = si
                        elif isinstance(tv, dict) and isinstance(sv, dict):
                            _merge(tv, sv)
                        else:
                            target[k] = sv

                _merge(commented, data)
                with path.open("w", encoding="utf-8") as f:
                    yaml_obj.dump(commented, f)
                return
            except Exception:  # noqa: BLE001
                pass

        import yaml

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                data, f, allow_unicode=True, sort_keys=False, default_flow_style=False
            )

    # ---------------------------------------------------------------------------
    # 便捷属性
    # ---------------------------------------------------------------------------

    @property
    def factor_names(self) -> List[str]:
        return [r.name for r in self.ranks]

    @property
    def factor_weights(self) -> List[float]:
        return [r.weight for r in self.ranks]

    @property
    def factor_directions(self) -> List[int]:
        return [r.direction for r in self.ranks]

    @property
    def ranked_factor_exprs(self) -> List[str]:
        return [r.expr_str for r in self.ranks]

    def _iter_all_conditions(self):
        return (getattr(self, k) for k in _COND_GROUPS)

    def get_condition_exprs(self) -> List[str]:
        """收集全部条件表达式（占位符已用 default 替换）。"""
        return [
            f"{c.name} = {c.resolved_expression}"
            for group in self._iter_all_conditions()
            for c in group
            if c.expression.strip()
        ]

    def has_condition_opt_params(self) -> bool:
        return any(c.has_opt_params for g in self._iter_all_conditions() for c in g)

    def get_static_condition_exprs(self) -> List[str]:
        """收集无 opt 参数的条件表达式（用于优化预计算阶段）。"""
        return [
            f"{c.name} = {c.resolved_expression}"
            for group in self._iter_all_conditions()
            for c in group
            if c.expression.strip() and not c.has_opt_params
        ]

    # ---------------------------------------------------------------------------
    # Optuna 相关
    # ---------------------------------------------------------------------------

    def run_opt_trial(self, trial: Any, dp: Any, base_df: Any) -> Any:
        """一次 Optuna trial：动态条件采样 → 含动态条件的组 And/Or 重聚合 → 权重采样。

        base_df 已包含：因子预处理结果 + 静态条件列 + 无动态参数组的 And/Or 聚合列。
        此方法只处理每 trial 变化的部分。
        """
        import polars as pl
        from alpha_factory.data_provider.factorsprocessor import FactorsRankComposite
        from alpha_factory.data_provider.prcoessors import And, Or
        from alpha_factory.cli.opt import _COMPOSITE_COL

        df: pl.DataFrame = base_df

        dynamic_exprs = [
            f"{c.name} = {c.resolve_expression(trial)}"
            for group in self._iter_all_conditions()
            for c in group
            if c.expression.strip() and c.has_opt_params
        ]
        if dynamic_exprs:
            df = dp.eval_exprs_on_df(df, dynamic_exprs)

        # 只对含动态条件的组重新聚合；静态组 And/Or 已在 base_df 中预计算
        def _names(g):
            return [c.name for c in g if c.expression.strip()]

        def _has_dyn(g):
            return any(c.has_opt_params for c in g)

        if _has_dyn(self.pool_mask) and (ns := _names(self.pool_mask)):
            df = And(factors=ns, name=F.POOL_MASK).process(df)
        if _has_dyn(self.buy_able) and (ns := _names(self.buy_able)):
            df = And(factors=ns, name="buy_able").process(df)
        if _has_dyn(self.not_buy_able) and (ns := _names(self.not_buy_able)):
            df = Or(factors=ns, name="not_buy_able").process(df)
        if _has_dyn(self.sell_able) and (ns := _names(self.sell_able)):
            df = And(factors=ns, name="sell_able").process(df)
        if _has_dyn(self.not_sell_able) and (ns := _names(self.not_sell_able)):
            df = Or(factors=ns, name="not_sell_able").process(df)

        df = FactorsRankComposite(
            factors=self.factor_names,
            name=_COMPOSITE_COL,
            opt=True,
            trial=trial,
            use_rank=False,
        ).process(df)
        return df

    def apply_best_condition_params(self, best_params: Dict[str, Any]) -> None:
        """将 study.best_params 写回各条件的 opt.default（供下次非优化模式直接使用）。"""
        for group in self._iter_all_conditions():
            for c in group:
                for param_name in c.opt:
                    key = f"{c.name}__{param_name}"
                    if key in best_params:
                        c.opt[param_name]["default"] = best_params[key]

    # ---------------------------------------------------------------------------
    # Actions 管道
    # ---------------------------------------------------------------------------

    def build_actions(self) -> List:
        """构建 DataProvider actions 管道（opt_mode=True 时返回空列表）。

        执行顺序：pool_mask AND → 因子预处理 → 因子合成 → buy_able AND →
        not_buy_able OR → sell_able AND → not_sell_able OR。
        """
        if self.opt_mode:
            return []

        from alpha_factory.data_provider.factorsprocessor import (
            FactorsPreProcessor,
            FactorsRankComposite,
        )
        from alpha_factory.data_provider.prcoessors import And, Or
        from alpha_factory.cli.opt import _COMPOSITE_COL

        actions: List = []
        names = lambda filters: [f.name for f in filters if f.expression.strip()]  # noqa: E731

        if ns := names(self.pool_mask):
            actions.append(And(factors=ns, name=F.POOL_MASK))

        if len(self.ranks) > 1:
            fnames = self.factor_names
            if self.preprocess:
                actions.append(
                    FactorsPreProcessor(factors=fnames, actions=self.preprocess)
                )
            weights = {n: float(w) for n, w in zip(fnames, self.factor_weights)}
            actions.append(
                FactorsRankComposite(
                    factors=fnames, name=_COMPOSITE_COL, weights=weights, use_rank=False
                )
            )

        if ns := names(self.buy_able):
            actions.append(And(factors=ns, name="buy_able"))
        if ns := names(self.not_buy_able):
            actions.append(Or(factors=ns, name="not_buy_able"))
        if ns := names(self.sell_able):
            actions.append(And(factors=ns, name="sell_able"))
        if ns := names(self.not_sell_able):
            actions.append(Or(factors=ns, name="not_sell_able"))

        return actions

    # ---------------------------------------------------------------------------
    # 潜在交易生成
    # ---------------------------------------------------------------------------

    def generate_potential_trades(
        self,
        df: "pl.DataFrame",  # noqa: F821
        factor_col: Optional[str] = None,
        *,
        ascending: bool = False,
        include_open: bool = True,
    ) -> "pl.DataFrame":  # noqa: F821
        """对每个资产运行买卖信号状态机，返回所有潜在交易记录。

        信号规则（与 ``backtest_quick_daily`` 对齐）：
          - 买入：``rank <= buy_rank`` 且当前未持仓
          - 卖出：``rank > sell_rank`` 且当前持仓中

        参数:
            df: 已 collect 的 DataFrame，须包含 ``DATE, ASSET, POOL_MASK,
                CLOSE`` 及 ``factor_col`` 列。数据加载方式参考
                ``quant bt``（``DataProvider.load_pool_data`` + ``build_actions``）。
            factor_col: 用于排名的因子列名。单因子策略可省略（自动取
                ``ranks[0].name``）；多因子策略须与 ``_COMPOSITE_COL`` 保持一致。
            ascending: 排名方向，默认 ``False``（值越大排名越靠前）。
            include_open: 末尾仍持仓的交易是否纳入结果（sell_date 等字段为
                ``null``）。

        返回:
            ``pl.DataFrame``，列：

            * ``ASSET`` — 资产代码
            * ``buy_date`` — 买入信号触发日
            * ``sell_date`` — 卖出信号触发日（开放持仓为 ``null``）
            * ``hold_days`` — 持仓交易日数（开放持仓为 ``null``）
            * ``rank_val`` — 买入日排名值
            * ``sell_rank_val`` — 卖出日排名值（开放持仓为 ``null``）
            * ``pnl_ret`` — 收益率 ``sell_close/buy_close - 1``（开放持仓为 ``null``）
        """
        import polars as pl

        # --- 1. factor_col 推断 ---
        if factor_col is None:
            if len(self.ranks) != 1:
                raise ValueError(
                    "多因子策略须显式传入 factor_col（通常为 _COMPOSITE_COL）"
                )
            factor_col = self.ranks[0].name

        # --- 2. 计算截面 RANK（与 backtest_quick_daily 保持一致）---
        df_ranked = df.with_columns(
            pl.when(pl.col(F.POOL_MASK))
            .then(pl.col(factor_col))
            .otherwise(None)
            .rank(descending=not ascending, method="random")
            .over(F.DATE)
            .fill_null(999999)
            .alias("_RANK")
        ).sort([F.DATE, F.ASSET])

        buy_rank = self.buy_rank
        sell_rank = self.sell_rank

        # --- 3. per-asset 状态机 ---
        records: list[dict] = []

        for asset_df in df_ranked.partition_by(F.ASSET, maintain_order=True):
            in_position = False
            buy_date = None
            buy_close = None
            buy_rank_val = None
            dates = asset_df[F.DATE].to_list()
            closes = asset_df[F.CLOSE].to_list()
            ranks = asset_df["_RANK"].to_list()

            for date, close, rank in zip(dates, closes, ranks):
                if not in_position:
                    if rank <= buy_rank:
                        in_position = True
                        buy_date = date
                        buy_close = close
                        buy_rank_val = rank
                else:
                    if rank > sell_rank:
                        records.append(
                            {
                                F.ASSET: asset_df[F.ASSET][0],
                                "buy_date": buy_date,
                                "sell_date": date,
                                "hold_days": None,  # 先 None，下面用 polars 计算
                                "rank_val": float(buy_rank_val),
                                "sell_rank_val": float(rank),
                                "buy_close": float(buy_close)
                                if buy_close is not None
                                else None,
                                "sell_close": float(close)
                                if close is not None
                                else None,
                            }
                        )
                        in_position = False
                        buy_date = buy_close = buy_rank_val = None

            # 末尾仍持仓
            if in_position and include_open:
                records.append(
                    {
                        F.ASSET: asset_df[F.ASSET][0],
                        "buy_date": buy_date,
                        "sell_date": None,
                        "hold_days": None,
                        "rank_val": float(buy_rank_val),
                        "sell_rank_val": None,
                        "buy_close": float(buy_close)
                        if buy_close is not None
                        else None,
                        "sell_close": None,
                    }
                )

        if not records:
            return pl.DataFrame(
                {
                    F.ASSET: pl.Series([], dtype=pl.Utf8),
                    "buy_date": pl.Series([], dtype=pl.Date),
                    "sell_date": pl.Series([], dtype=pl.Date),
                    "hold_days": pl.Series([], dtype=pl.Int32),
                    "rank_val": pl.Series([], dtype=pl.Float64),
                    "sell_rank_val": pl.Series([], dtype=pl.Float64),
                    "pnl_ret": pl.Series([], dtype=pl.Float64),
                }
            )

        # --- 4. 构建 DataFrame，计算 hold_days + pnl_ret ---
        # 显式 cast 日期列，避免全为 null 时 Polars 推断为 Null 类型导致减法失败
        result = (
            (
                pl.DataFrame(records).with_columns(
                    pl.col("buy_date").cast(pl.Date),
                    pl.col("sell_date").cast(pl.Date),
                )
            )
            .with_columns(
                pl.when(
                    pl.col("sell_date").is_not_null() & pl.col("buy_date").is_not_null()
                )
                .then(
                    (pl.col("sell_date") - pl.col("buy_date"))
                    .dt.total_days()
                    .cast(pl.Int32)
                )
                .otherwise(None)
                .alias("hold_days"),
                pl.when(
                    pl.col("sell_close").is_not_null()
                    & pl.col("buy_close").is_not_null()
                )
                .then(pl.col("sell_close") / pl.col("buy_close") - 1.0)
                .otherwise(None)
                .alias("pnl_ret"),
            )
            .select(
                [
                    F.ASSET,
                    "buy_date",
                    "sell_date",
                    "hold_days",
                    "rank_val",
                    "sell_rank_val",
                    "pnl_ret",
                ]
            )
        )

        return result
