from typing import List
import re

import polars as pl


class FactorsProcessor:
    def __init__(self, factors: List[str] = None, name: str = None):
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
