from typing import Dict, List, Optional

import polars as pl

from alpha_factory.data_provider.factorsprocessor import FactorsProcessor


class Rank(FactorsProcessor):
    def __init__(
        self,
        factors: str | List[str] = None,
        name: str = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(factors, name)
        self.weights = weights

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """对筛选出的因子列做截面加权求和，结果写入 self.name 列。

        权重（self.weights）为列名 → 权重的字典，与列顺序无关；
        若未指定权重则等权；字典中不存在的列默认权重为 0。
        合成公式：COMPOSITE = Σ (weight_i × col_i)
        """
        cols = self._cols_to_process(df)
        if not cols:
            return df

        resolved: List[float]
        if self.weights is not None:
            resolved = [float(self.weights.get(col, 0.0)) for col in cols]
        else:
            resolved = [1.0 / len(cols)] * len(cols)

        terms = [pl.col(col).cast(pl.Float64) * w for col, w in zip(cols, resolved)]
        composite_expr = terms[0]
        for t in terms[1:]:
            composite_expr = composite_expr + t

        out_col = self.name if self.name else "COMPOSITE"
        return df.with_columns(composite_expr.alias(out_col))
