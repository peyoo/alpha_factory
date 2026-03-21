import polars as pl

from alpha_factory.data_provider.factorsprocessor import FactorsComposite


class Or(FactorsComposite):
    """逻辑或合成器，继承自 FactorsProcessor，重写 process 方法实现逻辑或合成"""

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        factor_cols = self._cols_to_process(df)
        if not factor_cols:
            return df

        # 1. 计算逻辑或：只要任一因子为 True（非零），结果即为 True（1）
        lf = df.lazy()
        lf = lf.with_columns(pl.any_horizontal(*factor_cols).alias(self.name))
        return lf.collect()
