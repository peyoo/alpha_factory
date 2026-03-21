from typing import List, Callable, Dict
import re

import polars as pl

from alpha_factory.utils.schema import F


class FactorsAction:
    def __init__(self, factors: List[str] | str):
        # 待处理的因子
        # 可以为字符串（正则表达式）或列表（明确列名）
        self.factors = factors

    def _cols_to_process(self, df: pl.DataFrame) -> List[str]:
        if self.factors is None:
            return df.columns
        elif isinstance(self.factors, str):
            pattern = re.compile(self.factors)
            return [c for c in df.columns if pattern.match(c)]
        else:
            return [c for c in self.factors if c in df.columns]

    """因子处理器接口，定义了 process 方法，接受一个 Polars DataFrame 作为输入，并返回一个新的 DataFrame 作为输出"""

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("Subclasses must implement the process method")


class FactorsComposite(FactorsAction):
    """因子合成器，继承自 FactorsAction，重写 process 方法实现因子合成"""

    def __init__(self, factors: List[str] | str, name: str):
        super().__init__(factors)
        # 合成后的因子名
        self.name = name


class FactorsPreProcessor(FactorsAction):
    """
    因子预处理器，继承自 FactorsAction，重写 process 方法实现预处理逻辑
    actions 是一个函数列表，每个函数接受一个 DataFrame 并返回一个 DataFrame，预处理器会依次应用这些函数到指定的因子列上
    比如，actions 可以包含去极值、标准化、回归残差等函数，这些函数会被依次应用到 factors 指定的列上，生成新的列覆盖原列

    """

    def __init__(
        self,
        factors: List[str] | str,
        actions: List[Callable[[pl.Expr], pl.Expr]],
        use_pool_mask: bool = True,
    ):
        super().__init__(factors)
        # 预处理动作列表，每个动作都是一个函数，接受一个 pl.Expr 并返回一个 pl.Expr
        # 所有的预处理动作都是截面函数，在 DATE 维度上应用
        # https://github.com/wukan1986/polars_ta/blob/main/polars_ta/wq/preprocess.py
        self.actions = actions
        self.use_pool_mask = use_pool_mask

    def _apply_actions_to_cols(self, cols: List[str]) -> List[pl.Expr]:
        """构建嵌套应用所有actions后的表达式列表"""
        final_exprs = []
        for c in cols:
            expr = pl.col(c)
            for action in self.actions:
                expr = action(expr)
            final_exprs.append(expr.over(F.DATE).alias(c))
        return final_exprs

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = self._cols_to_process(df)
        if not cols:
            return df

        has_mask = self.use_pool_mask and F.POOL_MASK in df.columns
        final_exprs = self._apply_actions_to_cols(cols)

        # 无mask 情况：直接应用预处理
        if not has_mask:
            return df.lazy().with_columns(final_exprs).collect()

        # 有mask 情况：Step 1 & 2
        # Step 1: 只保留池内股票，截面统计量只含池内数据（正确）
        # Step 2: 在池内截面上应用预处理，.over(DATE) 确保截面统计量仅含池内股票
        lf_pool = df.lazy().filter(pl.col(F.POOL_MASK)).with_columns(final_exprs)

        # Step 3: join 回原表
        # 使用 when-then-otherwise 确保非池内行保持原值，池内行使用新值
        keys = [F.DATE, F.ASSET]
        result = (
            df.lazy()
            .join(lf_pool.select(keys + cols), on=keys, how="left", suffix="_new")
            .with_columns(
                pl.when(pl.col(F.POOL_MASK))
                .then(pl.col(f"{c}_new"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in cols
            )
            .drop([f"{c}_new" for c in cols])
        )
        return result.collect()


class FactorsRankComposite(FactorsComposite):
    """因子排名合成器，继承自 FactorsComposite，重写 process 方法实现因子排名合成"""

    def __init__(
        self,
        factors: List[str] | str,
        weights: Dict[str, float],
        name: str = "RANK",
        ascending: bool = False,
    ):
        super().__init__(factors, name)
        self.ascending = ascending
        self.weights = weights

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = self._cols_to_process(df)
        if not cols:
            return df

        resolved_weights = [float(self.weights.get(col, 0.0)) for col in cols]
        total_weight = sum(resolved_weights)
        if total_weight == 0:
            resolved_weights = [1.0 / len(cols)] * len(cols)  # 等权重
        else:
            resolved_weights = [
                w / total_weight for w in resolved_weights
            ]  # 权重归一化

        # 修复：descending 逻辑应为 self.ascending 的相反值
        terms = [
            pl.col(col).rank(descending=self.ascending).over(F.DATE) * w
            for col, w in zip(cols, resolved_weights)
        ]
        composite_expr = pl.sum_horizontal(terms)

        return df.with_columns(composite_expr.alias(self.name))


class FactorsProcessor:
    def __init__(self, factors: List[str] | str, name: str):
        # 合成后的因子名，默认为None，表示不修改列名（不合成，不生成新列）
        self.name = name
        # 待处理的因子，默认为 None，表示处理所有列（不筛选）
        # 也可以为字符串（正则表达式）或列表（明确列名）
        self.factors = factors

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _cols_to_process(self, df: pl.DataFrame) -> List[str]:
        if self.factors is None:
            return df.columns
        elif isinstance(self.factors, str):
            pattern = re.compile(self.factors)
            return [c for c in df.columns if pattern.match(c)]
        else:
            return [c for c in self.factors if c in df.columns]
